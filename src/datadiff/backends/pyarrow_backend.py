from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any

from datadiff.backends.base import Backend, BackendResult
from datadiff.backends.dataframe_semantics import (
    aggregate_triplets,
    case_when_plan,
    coalesce_plan,
    distinct_columns,
    filter_plan,
    groupby_plan,
    mutate_plan,
    select_columns,
)
from datadiff.backends.probe_semantics import EXTENDED_FALSE_PROBE_KINDS, resolve_bool_probe
from datadiff.csv_roundtrip import (
    csv_long_numeric_roundtrip_mismatch,
    csv_long_numeric_values,
    long_numeric_csv_path,
)
from datadiff.dsl import Program, SortKey, TableData, normalize_sort_keys
from datadiff.filtering import parse_filter_comparator
from datadiff.join_keys import join_key_arg, join_key_pairs
from datadiff.operation_semantics import (
    join_how,
    op_ascending,
    op_column,
    op_columns,
    op_kind,
    op_n,
    op_nulls,
    op_output_alias,
    op_right_columns,
    op_table,
    op_value,
)
from datadiff.pathing import path_basename
from datadiff.running import (
    running_sum_partition_columns,
    running_sum_sort_keys,
    sort_rows_for_running,
    stable_running_sum_values,
)
from datadiff.sortedness import is_sorted_values
from datadiff.tuple_logic import evaluate_tuple_absence
from datadiff.windowing import row_number_filter_rows


def _single_bool_table(pa: Any, alias: str, value: bool):
    return pa.Table.from_pydict({alias: [value]})


def _pyarrow_probe_handlers(pa: Any, op: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset_isin_all_match_probe": lambda: _pyarrow_dataset_isin_all_match_mismatch(pa),
        "run_end_null_compute_probe": lambda: _pyarrow_run_end_null_compute_mismatch(pa),
        "large_string_partition_probe": lambda: _pyarrow_large_string_partition_mismatch(pa),
        "hash_pivot_wider_probe": lambda: _pyarrow_hash_pivot_wider_mismatch(pa),
        "list_flatten_parent_indices_probe": lambda: _pyarrow_list_flatten_parent_indices_mismatch(pa),
        "csv_long_numeric_roundtrip_probe": lambda: _pyarrow_csv_long_numeric_roundtrip_mismatch(pa, op),
    }


class PyArrowBackend(Backend):
    name = "pyarrow"

    def _to_table(self, table: TableData):
        import pyarrow as pa

        schema = pa.schema(
            [pa.field(column.name, _arrow_type(pa, column.type), nullable=column.nullable) for column in table.columns]
        )
        return pa.Table.from_pylist(table.rows, schema=schema)

    def run(self, tables: list[TableData], program: Program, timeout_s: float = 5.0) -> BackendResult:
        start = time.perf_counter()
        try:
            with _suppress_native_stderr():
                import pyarrow as pa
                import pyarrow.compute as pc
                import pyarrow.csv as pacsv

                table_by_name = {table.name: table for table in tables}
                arrow_tables = {table.name: self._to_table(table) for table in tables}
                current_cols = [column.name for column in tables[0].columns]
                current = arrow_tables[tables[0].name]

                for op in program.operations:
                    kind = op_kind(op)
                    if kind == "join":
                        right_name = op_table(op)
                        right_data = table_by_name[right_name]
                        right = arrow_tables[right_name]
                        left_keys, right_keys = join_key_pairs(op)
                        right_key_set = set(right_keys)
                        right_cols = [
                            column.name
                            for column in right_data.columns
                            if column.name not in right_key_set and column.name not in current_cols
                        ]
                        join_type = "left outer" if join_how(op) == "left" else "inner"
                        current = current.join(
                            right.select([*right_keys, *right_cols]),
                            keys=join_key_arg(left_keys),
                            right_keys=join_key_arg(right_keys),
                            join_type=join_type,
                            coalesce_keys=True,
                            use_threads=False,
                        )
                        current_cols = [*current_cols, *right_cols]
                        current = _select_existing(current, current_cols)
                    elif kind == "union_all":
                        right = arrow_tables[op_table(op)].select(current_cols)
                        current = pa.concat_tables([current.select(current_cols), right], promote_options="default")
                    elif kind in {"semi_join", "anti_join"}:
                        join_type = "left semi" if kind == "semi_join" else "left anti"
                        left_keys, right_keys = join_key_pairs(op)
                        right_key_table = arrow_tables[op_table(op)].select(right_keys)
                        valid_mask = pc.is_valid(right_key_table[right_keys[0]])
                        for right_key in right_keys[1:]:
                            valid_mask = pc.and_(valid_mask, pc.is_valid(right_key_table[right_key]))
                        right_key_table = right_key_table.filter(valid_mask)
                        current = current.join(
                            right_key_table,
                            keys=join_key_arg(left_keys),
                            right_keys=join_key_arg(right_keys),
                            join_type=join_type,
                            coalesce_keys=True,
                            use_threads=False,
                        )
                        current = _select_existing(current, current_cols)
                    elif kind == "drop_nulls":
                        columns = op_columns(op)
                        mask = pc.is_valid(current[columns[0]])
                        for column in columns[1:]:
                            mask = pc.and_(mask, pc.is_valid(current[column]))
                        current = current.filter(mask)
                    elif kind == "filter":
                        column, comparator, value = filter_plan(op)
                        mask = _comparison_mask(pa, pc, current[column], comparator, value)
                        current = current.filter(mask)
                    elif kind == "tuple_absence_filter":
                        right_columns = op_right_columns(op)
                        right_rows = arrow_tables[op_table(op)].select(right_columns).to_pylist()
                        left_columns = op_columns(op)
                        rows = [
                            row
                            for row in current.to_pylist()
                            if evaluate_tuple_absence(row, left_columns, right_rows, right_columns)
                        ]
                        current = pa.Table.from_pylist(rows, schema=current.schema)
                    elif kind == "row_number_filter":
                        rows = row_number_filter_rows(current.to_pylist(), op)
                        current = pa.Table.from_pylist(rows, schema=current.schema)
                    elif kind == "running_sum":
                        out_column = op_column(op)
                        rows = sort_rows_for_running(current.to_pylist(), running_sum_sort_keys(op))
                        values = stable_running_sum_values(rows, op.source, running_sum_partition_columns(op))
                        rows = [{**row, out_column: value} for row, value in zip(rows, values)]
                        current_cols = [col for col in current_cols if col != out_column] + [out_column]
                        fields = [
                            field for field in current.schema if field.name != out_column
                        ] + [pa.field(out_column, pa.float64(), nullable=True)]
                        current = pa.Table.from_pylist(rows, schema=pa.schema(fields))
                    elif kind == "sortedness_check":
                        alias = op_output_alias(op)
                        ok = is_sorted_values(
                            current[op_column(op)].to_pylist(),
                            ascending=op_ascending(op),
                            nulls=op_nulls(op),
                        )
                        current_cols = [alias]
                        current = _single_bool_table(pa, alias, ok)
                    else:
                        probe_value = resolve_bool_probe(
                            kind,
                            handlers=_pyarrow_probe_handlers(pa, op),
                            fallback_false_kinds=EXTENDED_FALSE_PROBE_KINDS,
                        )
                        if probe_value is not None:
                            alias = op_output_alias(op)
                            current_cols = [alias]
                            current = _single_bool_table(pa, alias, probe_value)
                        elif kind == "select":
                            current_cols = select_columns(op)
                            current = current.select(current_cols)
                        elif kind == "distinct":
                            current_cols = distinct_columns(op)
                            current = current.select(current_cols).group_by(current_cols, use_threads=False).aggregate([])
                        elif kind == "fill_null":
                            column = op_column(op)
                            values = pc.fill_null(current[column], pa.scalar(op_value(op)))
                            current, current_cols = _replace_column(current, current_cols, column, values)
                        elif kind == "coalesce":
                            alias, sources, has_fallback, fallback = coalesce_plan(op)
                            values = [current[column] for column in sources]
                            if has_fallback:
                                values.append(pa.scalar(fallback))
                            current, current_cols = _replace_column(current, current_cols, alias, pc.coalesce(*values))
                        elif kind == "case_when":
                            alias, column, comparator, value, then_value, else_value = case_when_plan(op)
                            mask = _comparison_mask(pa, pc, current[column], comparator, value)
                            mask = pc.fill_null(mask, False)
                            values = pc.if_else(mask, pa.scalar(then_value), pa.scalar(else_value))
                            current, current_cols = _replace_column(current, current_cols, alias, values)
                        elif kind == "sort":
                            current = _sort_table(pa, current, normalize_sort_keys(op))
                        elif kind == "limit":
                            current = current.slice(0, op_n(op))
                        elif kind == "offset":
                            current = current.slice(op_n(op))
                        elif kind == "mutate":
                            plan = mutate_plan(op)
                            values = _eval_expr_plan(pa, pc, current, plan)
                            current, current_cols = _replace_column(current, current_cols, plan.column, values)
                        elif kind == "groupby":
                            keys, triplets = groupby_plan(op)
                            aggregates = [(column, _arrow_aggregate_func(func)) for column, func, _alias in triplets]
                            current = current.group_by(keys, use_threads=False).aggregate(aggregates)
                            source_names = [*keys, *[f"{column}_{_arrow_aggregate_func(func)}" for column, func, _alias in triplets]]
                            target_names = [*keys, *[alias for _column, _func, alias in triplets]]
                            current = _select_existing(current, source_names).rename_columns(target_names)
                            current_cols = target_names
                        elif kind == "aggregate":
                            values = {}
                            fields = []
                            aliases: list[str] = []
                            for column, func, alias in aggregate_triplets(op):
                                source = current[column]
                                values[alias] = [_global_aggregate(pc, source, func)]
                                aliases.append(alias)
                                fields.append(
                                    pa.field(
                                        alias,
                                        _global_aggregate_output_type(pa, source.type, func),
                                        nullable=True,
                                    )
                                )
                            current = pa.Table.from_pydict(values, schema=pa.schema(fields))
                            current_cols = aliases
                        else:
                            raise ValueError(kind)

                data = current.to_pandas(use_threads=False)

            return BackendResult(
                self.name,
                "ok",
                data=data,
                duration_ms=(time.perf_counter() - start) * 1000,
            )
        except Exception as exc:  # noqa: BLE001
            return BackendResult(
                self.name,
                "error",
                error_type=type(exc).__name__,
                error=str(exc),
                duration_ms=(time.perf_counter() - start) * 1000,
            )


def _pyarrow_dataset_isin_all_match_mismatch(pa) -> bool:
    import tempfile

    import pyarrow.compute as pc
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq

    expected = pa.table({"x": [0.0]})
    filter_expr = pc.field("x").isin([0.0])
    with tempfile.TemporaryDirectory() as tmpdir:
        pq.write_to_dataset(expected, tmpdir)
        observed = ds.dataset(tmpdir).filter(filter_expr).to_table()
    return observed.to_pydict() != expected.to_pydict()


def _pyarrow_run_end_null_compute_mismatch(pa) -> bool:
    import pyarrow.compute as pc

    run_ends = pa.array([1, 2, 4, 5], type=pa.int16())
    values = pa.array([True, None, False, True], type=pa.bool_())
    encoded = pa.RunEndEncodedArray.from_arrays(run_ends, values)
    expected = [True, None, True, True, True]
    try:
        observed = pc.true_unless_null(encoded).to_pylist()
    except Exception:  # noqa: BLE001
        return True
    return observed != expected


def _pyarrow_large_string_partition_mismatch(pa) -> bool:
    import tempfile
    from pathlib import Path

    import pyarrow.dataset as ds
    import pyarrow.parquet as pq

    expected = pa.table(
        {
            "part": pa.array(["a", "a", "b", "b"], type=pa.large_string()),
            "col": [1, 2, 3, 4],
        }
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir) / "large_string_dataset"
        part_a = root / "a" / "data.parquet"
        part_b = root / "b" / "data.parquet"
        part_a.parent.mkdir(parents=True)
        part_b.parent.mkdir(parents=True)
        pq.write_table(expected.slice(0, 2), part_a)
        pq.write_table(expected.slice(2, 2), part_b)
        try:
            observed = ds.dataset(root, partitioning=["part"], partition_base_dir=str(root)).to_table()
        except pa.ArrowTypeError:
            return True
    return observed.num_rows != expected.num_rows


def _pyarrow_hash_pivot_wider_mismatch(pa) -> bool:
    import pyarrow.compute as pc
    from pyarrow.acero import AggregateNodeOptions, Declaration, TableSourceNodeOptions

    data = {
        "foo": ["A", "A", "B", "B", "C"],
        "bar": ["k", "l", "m", "n", "o"],
        "N1": [1, 2, 2, 4, 2],
        "N2": [1, 2, 2, 4, 2],
    }
    table = pa.Table.from_pydict(data)
    source = Declaration("table_source", options=TableSourceNodeOptions(table))
    key_names = table.column("bar").unique().to_pylist()
    options = pc.PivotWiderOptions(key_names, unexpected_key_behavior="raise")
    aggregates = [(["bar", "N1"], "hash_pivot_wider", options, "N1")]
    pivot = Declaration("aggregate", AggregateNodeOptions(aggregates, ["foo"]))
    observed = Declaration.from_sequence([source, pivot]).to_table().flatten().to_pydict()
    expected = {
        "foo": ["A", "B", "C"],
        "N1.k": [1, None, None],
        "N1.l": [2, None, None],
        "N1.m": [None, 2, None],
        "N1.n": [None, 4, None],
        "N1.o": [None, None, 2],
    }
    return observed != expected


def _pyarrow_list_flatten_parent_indices_mismatch(pa) -> bool:
    import pyarrow.compute as pc

    values = pa.array([[1, 2], None, [], [None, 3]], type=pa.list_(pa.int64()))
    expected_flattened = [1, 2, None, 3]
    expected_parent_indices = [0, 0, 3, 3]
    try:
        flattened = pc.list_flatten(values).to_pylist()
        parent_indices = pc.list_parent_indices(values).to_pylist()
    except Exception:  # noqa: BLE001
        return True
    return flattened != expected_flattened or parent_indices != expected_parent_indices


def _pyarrow_csv_long_numeric_roundtrip_mismatch(pa, op: dict[str, Any]) -> bool:
    import pyarrow.csv as pacsv

    expected_values = csv_long_numeric_values(op)
    with long_numeric_csv_path(expected_values) as csv_path:
        observed_table = pacsv.read_csv(csv_path)
        observed_values = observed_table.column("value").to_pylist()
    return csv_long_numeric_roundtrip_mismatch(observed_values, expected_values)


def _comparison_mask(pa, pc, array: Any, comparator: str, value: Any):
    parsed = parse_filter_comparator(comparator)
    if parsed is None:
        raise ValueError(comparator)
    if parsed.base == "in_set":
        return _membership_mask(pa, pc, array, value)
    if parsed.base == "not_in_set":
        return pc.and_(pc.invert(_membership_mask(pa, pc, array, value)), pc.invert(pc.is_null(array)))
    if parsed.base == "is_null":
        return pc.is_null(array)
    if parsed.base == "is_not_null":
        return pc.invert(pc.is_null(array))
    if parsed.base == "bool_predicate":
        return _apply_boolean_truth_test(pc, array, parsed.truth_test)
    if parsed.base == "range_closed":
        lower, upper = value
        return pc.and_(pc.greater_equal(array, lower), pc.less_equal(array, upper))
    if parsed.base == "str_contains":
        return pc.fill_null(pc.match_substring(array, pattern=value), False)
    if parsed.base == "str_starts_with":
        return pc.fill_null(pc.starts_with(array, pattern=value), False)
    if parsed.base == "str_ends_with":
        return pc.fill_null(pc.ends_with(array, pattern=value), False)
    scalar = value
    if scalar is None:
        mask = pa.array([None] * len(array), type=pa.bool_())
    elif parsed.base == ">":
        mask = pc.greater(array, scalar)
    elif parsed.base == ">=":
        mask = pc.greater_equal(array, scalar)
    elif parsed.base == "<":
        mask = pc.less(array, scalar)
    elif parsed.base == "<=":
        mask = pc.less_equal(array, scalar)
    elif parsed.base == "==":
        mask = pc.equal(array, scalar)
    elif parsed.base == "!=":
        mask = pc.not_equal(array, scalar)
    else:
        raise ValueError(parsed.base)
    if parsed.truth_test is None:
        return mask
    if parsed.truth_test == "is_true":
        return pc.fill_null(mask, False)
    if parsed.truth_test == "is_not_true":
        return pc.invert(pc.fill_null(mask, False))
    if parsed.truth_test == "is_false":
        return pc.invert(pc.fill_null(mask, True))
    if parsed.truth_test == "is_not_false":
        return pc.fill_null(mask, True)
    if parsed.truth_test == "is_unknown":
        return pc.is_null(mask)
    if parsed.truth_test == "is_not_unknown":
        return pc.invert(pc.is_null(mask))
    raise ValueError(parsed.truth_test)


def _membership_mask(pa, pc, array: Any, values: Any):
    mask = pa.array([False] * len(array), type=pa.bool_())
    for value in values:
        try:
            value_mask = pc.equal(array, value)
        except (pa.ArrowException, TypeError):
            continue
        mask = pc.or_(mask, pc.fill_null(value_mask, False))
    return mask


def _apply_boolean_truth_test(pc, mask: Any, truth_test: str | None):
    if truth_test == "is_true":
        return pc.fill_null(mask, False)
    if truth_test == "is_not_true":
        return pc.invert(pc.fill_null(mask, False))
    if truth_test == "is_false":
        return pc.invert(pc.fill_null(mask, True))
    if truth_test == "is_not_false":
        return pc.fill_null(mask, True)
    if truth_test == "is_unknown":
        return pc.is_null(mask)
    if truth_test == "is_not_unknown":
        return pc.invert(pc.is_null(mask))
    raise ValueError(truth_test)


def _eval_expr_plan(pa, pc, table: Any, plan: Any):
    source = table[plan.source]
    if plan.kind == "add_const":
        return pc.add(source, plan.value)
    if plan.kind == "arith_const":
        operator = plan.operator
        if operator == "sub":
            return pc.subtract(source, plan.value)
        if operator == "mul":
            return pc.multiply(source, plan.value)
        if operator == "div":
            return pc.divide(pc.cast(source, "float64"), plan.value)
        if operator == "mod":
            raise ValueError("pyarrow backend does not support modulo in the common DSL subset")
        raise ValueError(operator)
    if plan.kind == "reverse_division_columns":
        return pc.divide(pc.cast(table[plan.numerator], "float64"), pc.cast(source, "float64"))
    if plan.kind == "abs":
        return pc.abs(source)
    if plan.kind == "clip":
        lower = plan.lower
        upper = plan.upper
        lower_clipped = pc.if_else(pc.less(source, lower), lower, source)
        return pc.if_else(pc.greater(lower_clipped, upper), upper, lower_clipped)
    if plan.kind == "bool_not":
        return pc.invert(source)
    if plan.kind == "cast":
        if plan.target_type == "float":
            return pc.cast(source, "float64")
        if plan.target_type == "int":
            return pc.cast(source, "int64")
        if plan.target_type == "str":
            return pc.cast(source, "string")
        raise ValueError(plan.target_type)
    if plan.kind == "string_length":
        return pc.utf8_length(source)
    if plan.kind == "string_lower":
        return pc.utf8_lower(source)
    if plan.kind == "string_upper":
        return pc.utf8_upper(source)
    if plan.kind == "string_strip":
        return pc.utf8_trim_whitespace(source)
    if plan.kind == "string_null_if_empty":
        return pc.if_else(pc.equal(source, ""), pa.scalar(None, type=source.type), source)
    if plan.kind == "string_replace":
        return pc.replace_substring(source, pattern=plan.old, replacement=plan.new)
    if plan.kind == "string_slice":
        start = plan.start
        return pc.utf8_slice_codeunits(source, start=start, stop=start + plan.length)
    if plan.kind == "string_split_part":
        return pc.list_element(pc.split_pattern(source, pattern=plan.separator, max_splits=1), plan.index)
    if plan.kind == "string_concat":
        return pc.binary_join_element_wise(source, table[plan.other], plan.separator or "")
    if plan.kind == "string_contains":
        return pc.match_substring(source, pattern=plan.needle)
    if plan.kind == "string_starts_with":
        return pc.starts_with(source, pattern=plan.needle)
    if plan.kind == "string_ends_with":
        return pc.ends_with(source, pattern=plan.needle)
    if plan.kind == "date_part":
        spans = {"year": (0, 4), "month": (5, 7), "day": (8, 10)}
        start, stop = spans[plan.part]
        return pc.cast(pc.utf8_slice_codeunits(source, start=start, stop=stop), "int64")
    if plan.kind == "string_basename":
        return pa.array([
            path_basename(value)
            for value in source.to_pylist()
        ], type=pa.string())
    raise ValueError(plan.kind)


def _global_aggregate(pc, array: Any, func: str) -> Any:
    if func == "count":
        return pc.count(array, mode="only_valid").as_py()
    if func == "nunique":
        return pc.count_distinct(array).as_py()
    if func == "sum":
        return pc.sum(array).as_py()
    if func == "mean":
        return pc.mean(array).as_py()
    if func == "any":
        return pc.any(array).as_py()
    if func == "all":
        return pc.all(array).as_py()
    if func == "min":
        return pc.min(array).as_py()
    if func == "max":
        return pc.max(array).as_py()
    raise ValueError(func)


def _global_aggregate_output_type(pa, source_type: Any, func: str) -> Any:
    if func in {"count", "nunique"}:
        return pa.int64()
    if func == "mean":
        return pa.float64()
    if func in {"any", "all"}:
        return pa.bool_()
    if func in {"sum", "min", "max"}:
        return source_type
    raise ValueError(func)


def _arrow_aggregate_func(func: str) -> str:
    return "count_distinct" if func == "nunique" else func


def _sort_table(pa: Any, table: Any, sort_keys: list[SortKey]) -> Any:
    null_placements = {key.nulls for key in sort_keys}
    if len(null_placements) <= 1:
        return table.sort_by(
            [(key.column, "ascending" if key.ascending else "descending") for key in sort_keys],
            null_placement="at_start" if sort_keys and sort_keys[0].nulls == "first" else "at_end",
        )

    df = table.to_pandas(use_threads=False)
    for key in reversed(sort_keys):
        df = df.sort_values(
            key.column,
            ascending=key.ascending,
            na_position=key.nulls,
            kind="mergesort",
        )
    return pa.Table.from_pandas(df, schema=table.schema, preserve_index=False)


def _replace_column(table: Any, cols: list[str], column: str, values: Any) -> tuple[Any, list[str]]:
    kept_cols = [col for col in cols if col != column]
    out = table.select(kept_cols)
    out = out.append_column(column, values)
    return out, [*kept_cols, column]


def _select_existing(table: Any, columns: list[str]):
    return table.select([column for column in columns if column in table.column_names])


def _arrow_type(pa, kind: str):
    if kind == "int":
        return pa.int64()
    if kind == "float":
        return pa.float64()
    if kind == "bool":
        return pa.bool_()
    return pa.string()


@contextmanager
def _suppress_native_stderr():
    import os
    import sys

    sys.stderr.flush()
    original_fd = os.dup(2)
    try:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            os.dup2(devnull.fileno(), 2)
            yield
    finally:
        os.dup2(original_fd, 2)
        os.close(original_fd)
