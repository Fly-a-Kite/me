from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any

from datadiff.backends.base import Backend, BackendResult
from datadiff.dsl import Program, SortKey, TableData, normalize_sort_keys
from datadiff.filtering import parse_filter_comparator
from datadiff.running import sort_rows_for_running, stable_running_sum_values
from datadiff.sortedness import is_sorted_values
from datadiff.tuple_logic import evaluate_tuple_absence


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

                table_by_name = {table.name: table for table in tables}
                arrow_tables = {table.name: self._to_table(table) for table in tables}
                current_cols = [column.name for column in tables[0].columns]
                current = arrow_tables[tables[0].name]

                for op in program.operations:
                    kind = op["op"]
                    if kind == "join":
                        right_data = table_by_name[op["table"]]
                        right = arrow_tables[op["table"]]
                        right_cols = [
                            column.name
                            for column in right_data.columns
                            if column.name != op["right_on"] and column.name not in current_cols
                        ]
                        join_type = "left outer" if op["how"] == "left" else "inner"
                        current = current.join(
                            right.select([op["right_on"], *right_cols]),
                            keys=op["left_on"],
                            right_keys=op["right_on"],
                            join_type=join_type,
                            coalesce_keys=True,
                            use_threads=False,
                        )
                        current_cols = [*current_cols, *right_cols]
                        current = _select_existing(current, current_cols)
                    elif kind == "filter":
                        mask = _comparison_mask(pa, pc, current[op["column"]], op["cmp"], op["value"])
                        current = current.filter(mask)
                    elif kind == "tuple_absence_filter":
                        right_rows = arrow_tables[op["table"]].select(list(op["right_columns"])).to_pylist()
                        left_columns = list(op["columns"])
                        right_columns = list(op["right_columns"])
                        rows = [
                            row
                            for row in current.to_pylist()
                            if evaluate_tuple_absence(row, left_columns, right_rows, right_columns)
                        ]
                        current = pa.Table.from_pylist(rows, schema=current.schema)
                    elif kind == "running_sum":
                        rows = sort_rows_for_running(current.to_pylist(), normalize_sort_keys({"keys": op["order_by"]}))
                        values = stable_running_sum_values(rows, op["source"])
                        rows = [{**row, op["column"]: value} for row, value in zip(rows, values)]
                        current_cols = [col for col in current_cols if col != op["column"]] + [op["column"]]
                        fields = [
                            field for field in current.schema if field.name != op["column"]
                        ] + [pa.field(op["column"], pa.float64(), nullable=True)]
                        current = pa.Table.from_pylist(rows, schema=pa.schema(fields))
                    elif kind == "sortedness_check":
                        ok = is_sorted_values(
                            current[op["column"]].to_pylist(),
                            ascending=bool(op.get("ascending", True)),
                            nulls=str(op.get("nulls", "last")),
                        )
                        current_cols = [op["as"]]
                        current = pa.Table.from_pydict({op["as"]: [ok]})
                    elif kind == "random_case_probe":
                        current_cols = [op["as"]]
                        current = pa.Table.from_pydict({op["as"]: [False]})
                    elif kind == "group_quantile_probe":
                        current_cols = [op["as"]]
                        current = pa.Table.from_pydict({op["as"]: [False]})
                    elif kind == "scalar_subquery_probe":
                        current_cols = [op["as"]]
                        current = pa.Table.from_pydict({op["as"]: [False]})
                    elif kind == "window_avg_probe":
                        current_cols = [op["as"]]
                        current = pa.Table.from_pydict({op["as"]: [False]})
                    elif kind == "struct_distinct_probe":
                        current_cols = [op["as"]]
                        current = pa.Table.from_pydict({op["as"]: [False]})
                    elif kind == "bit_compare_probe":
                        current_cols = [op["as"]]
                        current = pa.Table.from_pydict({op["as"]: [False]})
                    elif kind == "round_even_probe":
                        current_cols = [op["as"]]
                        current = pa.Table.from_pydict({op["as"]: [False]})
                    elif kind == "series_rtruediv_probe":
                        current_cols = [op["as"]]
                        current = pa.Table.from_pydict({op["as"]: [False]})
                    elif kind == "uint64_isin_probe":
                        current_cols = [op["as"]]
                        current = pa.Table.from_pydict({op["as"]: [False]})
                    elif kind == "tuple_anti_null_probe":
                        current_cols = [op["as"]]
                        current = pa.Table.from_pydict({op["as"]: [False]})
                    elif kind == "sparse_mask_probe":
                        current_cols = [op["as"]]
                        current = pa.Table.from_pydict({op["as"]: [False]})
                    elif kind == "float_wrap_probe":
                        current_cols = [op["as"]]
                        current = pa.Table.from_pydict({op["as"]: [False]})
                    elif kind == "index_bool_probe":
                        current_cols = [op["as"]]
                        current = pa.Table.from_pydict({op["as"]: [False]})
                    elif kind == "select":
                        current_cols = list(op["columns"])
                        current = current.select(current_cols)
                    elif kind == "sort":
                        current = _sort_table(pa, current, normalize_sort_keys(op))
                    elif kind == "limit":
                        current = current.slice(0, int(op["n"]))
                    elif kind == "offset":
                        current = current.slice(int(op["n"]))
                    elif kind == "mutate":
                        expr = op["expr"]
                        values = _eval_expr(pc, current, expr)
                        current, current_cols = _replace_column(current, current_cols, op["column"], values)
                    elif kind == "groupby":
                        keys = list(op["keys"])
                        aggregates = [(agg["column"], _arrow_aggregate_func(agg["func"])) for agg in op["aggs"]]
                        current = current.group_by(keys, use_threads=False).aggregate(aggregates)
                        source_names = [*keys, *[f"{agg['column']}_{_arrow_aggregate_func(agg['func'])}" for agg in op["aggs"]]]
                        target_names = [*keys, *[agg["as"] for agg in op["aggs"]]]
                        current = _select_existing(current, source_names).rename_columns(target_names)
                        current_cols = target_names
                    elif kind == "aggregate":
                        values = {}
                        for agg in op["aggs"]:
                            values[agg["as"]] = [_global_aggregate(pc, current[agg["column"]], agg["func"])]
                        current = pa.Table.from_pydict(values)
                        current_cols = [agg["as"] for agg in op["aggs"]]
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


def _comparison_mask(pa, pc, array: Any, comparator: str, value: Any):
    parsed = parse_filter_comparator(comparator)
    if parsed is None:
        raise ValueError(comparator)
    if parsed.base == "in_set":
        values = list(value)
        value_set = pa.array(values, type=array.type)
        return pc.fill_null(pc.is_in(array, value_set=value_set), False)
    if parsed.base == "is_null":
        return pc.is_null(array)
    if parsed.base == "is_not_null":
        return pc.invert(pc.is_null(array))
    if parsed.base == "bool_predicate":
        return _apply_boolean_truth_test(pc, array, parsed.truth_test)
    if parsed.base == "range_closed":
        lower, upper = value
        return pc.and_(pc.greater_equal(array, lower), pc.less_equal(array, upper))
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


def _eval_expr(pc, table: Any, expr: dict[str, Any]):
    source = table[expr["source"]]
    if expr["kind"] == "add_const":
        return pc.add(source, expr["value"])
    if expr["kind"] == "arith_const":
        op = expr["op"]
        if op == "sub":
            return pc.subtract(source, expr["value"])
        if op == "mul":
            return pc.multiply(source, expr["value"])
        if op == "div":
            return pc.divide(pc.cast(source, "float64"), expr["value"])
        if op == "mod":
            raise ValueError("pyarrow backend does not support modulo in the common DSL subset")
        raise ValueError(op)
    if expr["kind"] == "cast" and expr["to"] == "float":
        return pc.cast(source, "float64")
    if expr["kind"] == "string_length":
        return pc.utf8_length(source)
    if expr["kind"] == "string_lower":
        return pc.utf8_lower(source)
    raise ValueError(expr["kind"])


def _global_aggregate(pc, array: Any, func: str) -> Any:
    if func == "count":
        return pc.count(array, mode="only_valid").as_py()
    if func == "nunique":
        return pc.count_distinct(array).as_py()
    if func == "sum":
        return pc.sum(array).as_py()
    if func == "min":
        return pc.min(array).as_py()
    if func == "max":
        return pc.max(array).as_py()
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
