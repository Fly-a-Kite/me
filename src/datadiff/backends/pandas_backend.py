from __future__ import annotations

import time
import warnings
from typing import Any

from datadiff.backends.base import Backend, BackendResult, PreparedTable, prepare_table
from datadiff.backends.probe_semantics import EXTENDED_FALSE_PROBE_KINDS, resolve_bool_probe
from datadiff.csv_roundtrip import (
    csv_long_numeric_roundtrip_mismatch,
    csv_long_numeric_values,
    long_numeric_csv_path,
)
from datadiff.dsl import Program, TableData, normalize_sort_keys
from datadiff.filtering import evaluate_filter_predicate
from datadiff.join_keys import join_key_arg, join_key_pairs
from datadiff.operation_semantics import (
    aggregate_alias,
    aggregate_column,
    aggregate_func,
    aggregate_specs,
    case_else_value,
    case_then_value,
    coalesce_fallback,
    coalesce_has_fallback,
    coalesce_sources,
    condition_cmp,
    condition_column,
    condition_value,
    expr_index,
    expr_input_domain,
    expr_kind,
    expr_length,
    expr_lower,
    expr_needle,
    expr_new,
    expr_numerator,
    expr_old,
    expr_operator,
    expr_other,
    expr_part,
    expr_separator,
    expr_source,
    expr_start,
    expr_target_type,
    expr_upper,
    expr_value,
    groupby_keys,
    join_how,
    op_ascending,
    op_column,
    op_columns,
    op_kind,
    op_n,
    op_nulls,
    op_output_alias,
    op_source,
    op_table,
    op_value,
)
from datadiff.operation_type_semantics import case_when_output_type
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


def _records_without_duplicate_column_warning(df: Any) -> list[dict[str, Any]]:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"DataFrame columns are not unique, some columns will be omitted\.",
            category=UserWarning,
        )
        return df.to_dict("records")


def _bool_reduction(values: Any, func: str) -> bool | None:
    import pandas as pd

    valid = [bool(value) for value in values if not pd.isna(value)]
    if not valid:
        return None
    if func == "any":
        return any(valid)
    if func == "all":
        return all(valid)
    raise ValueError(func)


def _single_bool_frame(pd: Any, alias: str, value: bool) -> Any:
    return pd.DataFrame([{alias: value}], columns=[alias])


def _pandas_bool_probe_handlers(pd: Any, op: dict[str, Any]) -> dict[str, Any]:
    return {
        "uint64_isin_probe": lambda: _pandas_uint64_isin_probe_value(pd),
        "sparse_mask_probe": lambda: _pandas_sparse_mask_probe_mismatch(pd),
        "index_bool_probe": lambda: _pandas_index_bool_probe_mismatch(pd),
        "arrow_string_eq_sum_probe": lambda: _pandas_arrow_string_eq_sum_mismatch(pd),
        "arrow_string_contains_na_probe": lambda: _pandas_arrow_string_contains_na_mismatch(pd),
        "arrow_timestamp_loc_slice_probe": lambda: _pandas_arrow_timestamp_loc_slice_mismatch(pd),
        "arrow_timestamp_index_attr_probe": lambda: _pandas_arrow_timestamp_index_attr_mismatch(pd),
        "eval_inplace_alias_probe": lambda: _pandas_eval_inplace_alias_mismatch(pd),
        "bool_reduction_skipna_probe": lambda: _pandas_bool_reduction_skipna_mismatch(pd, op),
        "arrow_bool_groupby_reduction_probe": lambda: _pandas_arrow_bool_groupby_reduction_mismatch(pd, op),
        "csv_long_numeric_roundtrip_probe": lambda: _pandas_csv_long_numeric_roundtrip_mismatch(pd, op),
    }


class PandasBackend(Backend):
    name = "pandas"

    def _to_df(self, table: TableData | PreparedTable):
        import pandas as pd
        prepared = prepare_table(table)
        data = {}
        for column in prepared.columns:
            values = prepared.columns_data[column.name]
            if column.type == "int":
                data[column.name] = pd.array(values, dtype="Int64")
            elif column.type == "float":
                data[column.name] = pd.array(values, dtype="Float64")
            elif column.type == "bool":
                data[column.name] = pd.array(values, dtype="boolean")
            elif column.type == "str":
                data[column.name] = pd.array(values, dtype="string")
            else:
                data[column.name] = values
        return pd.DataFrame(data, columns=[c.name for c in prepared.columns])

    def execute_lowered(
        self,
        tables: list[TableData | PreparedTable],
        program: Program,
        timeout_s: float = 5.0,
    ) -> BackendResult:
        started_at = time.perf_counter()
        try:
            import pandas as pd  # noqa: F401
            frames = {table.name: self._to_df(table) for table in tables}
            df = frames[tables[0].name]
            for op in program.operations:
                kind = op_kind(op)
                if kind == "join":
                    right = frames[op_table(op)]
                    before_cols = set(df.columns)
                    left_keys, right_keys = join_key_pairs(op)
                    df = df.merge(
                        right,
                        how=join_how(op),
                        left_on=join_key_arg(left_keys),
                        right_on=join_key_arg(right_keys),
                        suffixes=("", "_r"),
                        sort=False,
                    )
                    drop_cols = [c for c in df.columns if str(c).endswith("_r")]
                    for left_key, right_key in zip(left_keys, right_keys):
                        if right_key != left_key and right_key not in before_cols and right_key in df.columns:
                            drop_cols.append(right_key)
                    if drop_cols:
                        df = df.drop(columns=drop_cols)
                elif kind == "union_all":
                    right = frames[op_table(op)]
                    df = pd.concat([df, right[list(df.columns)]], ignore_index=True)
                elif kind in {"semi_join", "anti_join"}:
                    right = frames[op_table(op)]
                    left_keys, right_keys = join_key_pairs(op)
                    right_values = {
                        tuple(values)
                        for values in right[right_keys].itertuples(index=False, name=None)
                        if all(evaluate_filter_predicate(value, "is_not_null", None) for value in values)
                    }
                    matches = [
                        all(evaluate_filter_predicate(value, "is_not_null", None) for value in values)
                        and tuple(values) in right_values
                        for values in df[left_keys].itertuples(index=False, name=None)
                    ]
                    keep = matches if kind == "semi_join" else [not matched for matched in matches]
                    df = df.loc[pd.Series(keep, index=df.index, dtype=bool)]
                elif kind == "drop_nulls":
                    df = df.dropna(subset=op_columns(op))
                elif kind == "filter":
                    col = op_column(op)
                    val = op_value(op)
                    comparator = condition_cmp(op)
                    series = df[col]
                    column_values = _predicate_values(series)
                    keep_mask = pd.Series(
                        [
                            evaluate_filter_predicate(column_value, comparator, val)
                            for column_value in column_values
                        ],
                        index=df.index,
                        dtype=bool,
                    )
                    df = df.loc[keep_mask]
                elif kind == "tuple_absence_filter":
                    right = frames[op_table(op)]
                    right_rows = _records_without_duplicate_column_warning(right[list(op.right_columns)])
                    left_columns = list(op.columns)
                    right_columns = list(op.right_columns)
                    keep_mask = pd.Series(
                        [
                            evaluate_tuple_absence(row, left_columns, right_rows, right_columns)
                            for row in _records_without_duplicate_column_warning(df)
                        ],
                        index=df.index,
                        dtype=bool,
                    )
                    df = df.loc[keep_mask]
                elif kind == "row_number_filter":
                    rows = row_number_filter_rows(_records_without_duplicate_column_warning(df), op)
                    df = pd.DataFrame(rows, columns=list(df.columns))
                elif kind == "running_sum":
                    out_column = op_column(op)
                    columns = [column for column in df.columns if column != out_column] + [out_column]
                    rows = sort_rows_for_running(
                        _records_without_duplicate_column_warning(df),
                        running_sum_sort_keys(op),
                    )
                    values = stable_running_sum_values(rows, op_source(op), running_sum_partition_columns(op))
                    rows = [{**row, out_column: value} for row, value in zip(rows, values)]
                    df = pd.DataFrame(rows, columns=columns)
                elif kind == "sortedness_check":
                    ok = is_sorted_values(
                        df[op_column(op)].tolist(),
                        ascending=op_ascending(op),
                        nulls=op_nulls(op),
                    )
                    df = _single_bool_frame(pd, op_output_alias(op), ok)
                else:
                    probe_value = resolve_bool_probe(
                        kind,
                        handlers=_pandas_bool_probe_handlers(pd, op),
                        fallback_false_kinds=EXTENDED_FALSE_PROBE_KINDS,
                    )
                    if probe_value is not None:
                        df = _single_bool_frame(pd, op_output_alias(op), probe_value)
                    elif kind == "select":
                        df = df[op_columns(op)]
                    elif kind == "distinct":
                        cols = op_columns(op)
                        df = df[cols].drop_duplicates().reset_index(drop=True)
                    elif kind == "fill_null":
                        df = df.copy()
                        column = op_column(op)
                        df[column] = df[column].fillna(op_value(op))
                    elif kind == "coalesce":
                        df = df.copy()
                        alias = op_output_alias(op)
                        columns = coalesce_sources(op)
                        values = df[columns[0]].copy()
                        for column in columns[1:]:
                            values = values.combine_first(df[column])
                        if coalesce_has_fallback(op):
                            values = values.fillna(coalesce_fallback(op))
                        df[alias] = values
                    elif kind == "case_when":
                        alias = op_output_alias(op)
                        condition_values = _predicate_values(df[condition_column(op)])
                        df = df.copy()
                        values = [
                            case_then_value(op)
                            if evaluate_filter_predicate(value, condition_cmp(op), condition_value(op))
                            else case_else_value(op)
                            for value in condition_values
                        ]
                        if values:
                            df[alias] = values
                        else:
                            dtype = {
                                "bool": "boolean",
                                "float": "float64",
                                "int": "Int64",
                                "str": "string",
                            }.get(
                                case_when_output_type(
                                    case_then_value(op),
                                    case_else_value(op),
                                ),
                                "object",
                            )
                            df[alias] = pd.Series(index=df.index, dtype=dtype)
                    elif kind == "sort":
                        for key in reversed(normalize_sort_keys(op)):
                            df = df.sort_values(
                                key.column,
                                ascending=key.ascending,
                                na_position=key.nulls,
                                kind="mergesort",
                            )
                    elif kind == "limit":
                        df = df.head(op_n(op))
                    elif kind == "offset":
                        df = df.iloc[op_n(op):]
                    elif kind == "mutate":
                        df = df.copy()
                        out_column = op_column(op)
                        source = expr_source(op)
                        if expr_kind(op) == "add_const":
                            df[out_column] = df[source] + expr_value(op)
                        elif expr_kind(op) == "arith_const":
                            if expr_operator(op) == "sub":
                                df[out_column] = df[source] - expr_value(op)
                            elif expr_operator(op) == "mul":
                                df[out_column] = df[source] * expr_value(op)
                            elif expr_operator(op) == "div":
                                df[out_column] = df[source] / expr_value(op)
                            elif expr_operator(op) == "mod":
                                df[out_column] = df[source] % expr_value(op)
                            else:
                                raise ValueError(expr_operator(op))
                        elif expr_kind(op) == "reverse_division_columns":
                            df[out_column] = df[expr_numerator(op)] / df[source]
                        elif expr_kind(op) == "abs":
                            df[out_column] = df[source].abs()
                        elif expr_kind(op) == "clip":
                            df[out_column] = df[source].clip(lower=expr_lower(op), upper=expr_upper(op))
                        elif expr_kind(op) == "bool_not":
                            # Relational operations such as a left join or the
                            # row-oriented running-sum implementation can turn a
                            # nullable BooleanDtype column into ``object``.  Python
                            # applies ``~`` element-wise to object values, so an
                            # unmatched ``None`` raises TypeError (and bools become
                            # integers).  Restore pandas' three-valued boolean
                            # representation before applying logical NOT.
                            boolean_values = pd.array(
                                _predicate_values(df[source]),
                                dtype="boolean",
                            )
                            df[out_column] = ~boolean_values
                        elif expr_kind(op) == "cast":
                            if expr_target_type(op) == "float":
                                if expr_input_domain(op) in {"numeric_string", "integer_string"}:
                                    df[out_column] = pd.to_numeric(df[source], errors="raise").astype("float64")
                                else:
                                    df[out_column] = df[source].astype("float64")
                            elif expr_target_type(op) == "int":
                                df[out_column] = pd.to_numeric(df[source], errors="raise").astype("Int64")
                            elif expr_target_type(op) == "str":
                                df[out_column] = df[source].astype("string")
                            else:
                                raise ValueError(expr_target_type(op))
                        elif expr_kind(op) == "string_length":
                            df[out_column] = df[source].str.len()
                        elif expr_kind(op) == "string_lower":
                            df[out_column] = df[source].str.lower()
                        elif expr_kind(op) == "string_upper":
                            df[out_column] = df[source].str.upper()
                        elif expr_kind(op) == "string_strip":
                            df[out_column] = df[source].str.strip()
                        elif expr_kind(op) == "string_null_if_empty":
                            df[out_column] = df[source].mask(df[source] == "")
                        elif expr_kind(op) == "string_replace":
                            df[out_column] = df[source].str.replace(expr_old(op), expr_new(op), regex=False)
                        elif expr_kind(op) == "string_slice":
                            slice_start = int(expr_start(op))
                            df[out_column] = df[source].str.slice(
                                slice_start,
                                slice_start + int(expr_length(op)),
                            )
                        elif expr_kind(op) == "string_split_part":
                            df[out_column] = df[source].str.split(expr_separator(op), n=1, regex=False).str[int(expr_index(op))]
                        elif expr_kind(op) == "string_concat":
                            df[out_column] = df[source].str.cat(df[expr_other(op)], sep=expr_separator(op))
                        elif expr_kind(op) == "string_contains":
                            df[out_column] = df[source].str.contains(expr_needle(op), regex=False)
                        elif expr_kind(op) == "string_starts_with":
                            df[out_column] = df[source].str.startswith(expr_needle(op))
                        elif expr_kind(op) == "string_ends_with":
                            df[out_column] = df[source].str.endswith(expr_needle(op))
                        elif expr_kind(op) == "date_part":
                            spans = {"year": (0, 4), "month": (5, 7), "day": (8, 10)}
                            start_idx, stop_idx = spans[expr_part(op)]
                            extracted = df[source].str.slice(start_idx, stop_idx)
                            df[out_column] = pd.to_numeric(extracted, errors="coerce").astype("Int64")
                        elif expr_kind(op) == "string_basename":
                            df[out_column] = df[source].map(path_basename)
                        else:
                            raise ValueError(expr_kind(op))
                    elif kind == "groupby":
                        keys = groupby_keys(op)
                        group = df.groupby(keys, dropna=False, sort=False)
                        pieces = []
                        for agg in aggregate_specs(op):
                            col = aggregate_column(agg)
                            func = aggregate_func(agg)
                            alias = aggregate_alias(agg)
                            if func == "count":
                                series = group[col].count().rename(alias)
                            elif func == "nunique":
                                series = group[col].nunique(dropna=True).rename(alias)
                            elif func in {"any", "all"}:
                                series = group[col].agg(lambda values, current_func=func: _bool_reduction(values, current_func)).rename(alias)
                            elif func == "sum":
                                series = group[col].sum(min_count=1).rename(alias)
                            else:
                                series = getattr(group[col], func)().rename(alias)
                            pieces.append(series)
                        df = pd.concat(pieces, axis=1).reset_index()
                    elif kind == "aggregate":
                        values = {}
                        columns = []
                        for agg in aggregate_specs(op):
                            col = aggregate_column(agg)
                            func = aggregate_func(agg)
                            alias = aggregate_alias(agg)
                            columns.append(alias)
                            if func == "count":
                                values[alias] = int(df[col].count())
                            elif func == "nunique":
                                values[alias] = int(df[col].nunique(dropna=True))
                            elif func in {"any", "all"}:
                                values[alias] = _bool_reduction(df[col], func)
                            elif func == "sum":
                                values[alias] = df[col].sum(min_count=1)
                            else:
                                values[alias] = getattr(df[col], func)()
                        df = pd.DataFrame([values], columns=columns)
                    else:
                        raise ValueError(kind)
            return BackendResult(self.name, "ok", data=df, duration_ms=(time.perf_counter() - started_at) * 1000)
        except Exception as exc:  # noqa: BLE001
            return BackendResult(
                self.name,
                "error",
                error_type=type(exc).__name__,
                error=str(exc),
                duration_ms=(time.perf_counter() - started_at) * 1000,
            )


def _predicate_values(series):
    array = getattr(series, "array", None)
    if array is not None:
        return array.tolist()
    return series.tolist()


def _pandas_uint64_isin_probe_value(pd) -> bool:
    import numpy as np

    observed = pd.Series([635554097106142143], dtype="UInt64").isin(
        np.array([635554097106142079])
    ).iloc[0]
    return bool(observed)


def _pandas_sparse_mask_probe_mismatch(pd) -> bool:
    import numpy as np

    sparse = pd.arrays.SparseArray([1, 2, 3, 4, np.nan, np.nan], fill_value=np.nan)
    mask = sparse > [3, 3, 4, 1, 0, 0]
    observed = sparse[mask].to_numpy().tolist()
    expected = [4.0]
    return len(observed) != len(expected) or any(left != right for left, right in zip(observed, expected))


def _pandas_index_bool_probe_mismatch(pd) -> bool:
    observed = pd.Index([3, 5, 8], name="i") == 5
    return not isinstance(observed, pd.Index)


def _pandas_csv_long_numeric_roundtrip_mismatch(pd, op: dict[str, Any]) -> bool:
    expected_values = csv_long_numeric_values(op)
    with long_numeric_csv_path(expected_values) as csv_path:
        observed_values = pd.read_csv(csv_path)["value"].tolist()
    return csv_long_numeric_roundtrip_mismatch(observed_values, expected_values)


def _pandas_arrow_string_eq_sum_mismatch(pd) -> bool:
    values = pd.array(["a"], dtype="string[pyarrow]")
    try:
        observed = (values == values).sum()
    except AttributeError:
        return True
    return int(observed) != 1


def _pandas_arrow_string_contains_na_mismatch(pd) -> bool:
    values = pd.Series(["alpha", pd.NA, "BETA", ""], dtype="string[pyarrow]")
    try:
        contains = values.str.contains("a", case=False, regex=False, na=False)
        starts = values.str.startswith("a", na=False)
    except (AttributeError, TypeError, ValueError):
        return True
    return contains.tolist() != [True, False, True, False] or starts.tolist() != [True, False, False, False]


def _pandas_arrow_timestamp_loc_slice_mismatch(pd) -> bool:
    values = pd.Series(["2015-07-01 00:00:00"])
    frame = (
        values.astype("timestamp[ns][pyarrow]")
        .dt.tz_localize("UTC")
        .rename("event_time")
        .to_frame()
        .set_index("event_time")
        .sort_index()
    )
    try:
        observed = frame.loc["2015":]
    except TypeError:
        return True
    return len(observed) != 1


def _pandas_arrow_timestamp_index_attr_mismatch(pd) -> bool:
    values = pd.Series(["2001-05-07 01:00:00", "2001-06-08 02:00:00"])
    index = (
        values.astype("timestamp[ns][pyarrow]")
        .dt.tz_localize("UTC")
        .rename("event_time")
        .to_frame()
        .set_index("event_time")
        .index
    )
    try:
        observed = list(index.month)
    except AttributeError:
        return True
    return observed != [5, 6]


def _pandas_eval_inplace_alias_mismatch(pd) -> bool:
    import numpy as np

    original = np.array([-1, 1, -1], dtype=np.int32)
    frame = pd.DataFrame({"nums": original.copy()})
    frame.eval("nums2 = nums", inplace=True)
    frame.loc[[True, False, True], "nums2"] = 0
    return not np.array_equal(frame["nums"].to_numpy(), original)


def _pandas_bool_reduction_skipna_mismatch(
    pd, op: dict[str, Any] | None = None
) -> bool:
    if (op or {}).get("family_id") == "pandas_nullable_bool_reduction":
        return _pandas_nullable_bool_family_mismatch(pd, op or {}, groupby=False)
    frame = pd.DataFrame(
        {
            "has_signal": pd.array([True, pd.NA], dtype="boolean"),
            "all_missing": pd.array([pd.NA, pd.NA], dtype="boolean"),
        }
    )
    for reducer in ("any", "all"):
        for axis in (0, None):
            try:
                getattr(frame, reducer)(axis=axis, skipna=False)
            except (TypeError, ValueError):
                return True
    return False


def _pandas_arrow_bool_groupby_reduction_mismatch(
    pd, op: dict[str, Any] | None = None
) -> bool:
    if (op or {}).get("family_id") == "pandas_nullable_bool_reduction":
        return _pandas_nullable_bool_family_mismatch(pd, op or {}, groupby=True)

    def normalize(mapping: dict[str, Any]) -> dict[str, bool | None]:
        normalized: dict[str, bool | None] = {}
        for key, value in mapping.items():
            if pd.isna(value):
                normalized[str(key)] = None
            else:
                normalized[str(key)] = bool(value)
        return normalized

    expected = {
        "any_skipna_true": {"a": True, "b": False, "c": False, "d": True},
        "all_skipna_true": {"a": True, "b": False, "c": True, "d": False},
        "any_skipna_false": {"a": True, "b": None, "c": None, "d": True},
        "all_skipna_false": {"a": None, "b": False, "c": None, "d": False},
    }
    try:
        frame = pd.DataFrame(
            {
                "g": pd.array(["a", "a", "b", "b", "c", "c", "d", "d"], dtype="string"),
                "flag": pd.array(
                    [True, pd.NA, False, pd.NA, pd.NA, pd.NA, True, False],
                    dtype="bool[pyarrow]",
                ),
            }
        )
        grouped = frame.groupby("g", dropna=False)["flag"]
        observed = {
            "any_skipna_true": normalize(grouped.any(skipna=True).astype("object").to_dict()),
            "all_skipna_true": normalize(grouped.all(skipna=True).astype("object").to_dict()),
            "any_skipna_false": normalize(grouped.any(skipna=False).astype("object").to_dict()),
            "all_skipna_false": normalize(grouped.all(skipna=False).astype("object").to_dict()),
        }
    except Exception:  # noqa: BLE001
        return True
    return observed != expected


def _pandas_nullable_bool_family_mismatch(
    pd,
    op: dict[str, Any],
    *,
    groupby: bool,
) -> bool:
    reducer = str(op.get("reduction", "") or "")
    values = list(op.get("values", []) or [])
    if reducer not in {"any", "all"} or not values:
        return True
    has_true = any(value is True for value in values)
    has_false = any(value is False for value in values)
    has_null = any(value is None or value is pd.NA for value in values)
    if reducer == "any":
        expected: bool | None = True if has_true else None if has_null else False
    else:
        expected = False if has_false else None if has_null else True
    try:
        flags = pd.array(values, dtype="bool[pyarrow]" if groupby else "boolean")
        if groupby:
            frame = pd.DataFrame(
                {"g": pd.array(["a"] * len(values), dtype="string"), "flag": flags}
            )
            observed = getattr(frame.groupby("g")["flag"], reducer)(
                skipna=False
            ).iloc[0]
        else:
            observed = getattr(pd.Series(flags), reducer)(skipna=False)
    except Exception:  # noqa: BLE001
        return True
    normalized = None if pd.isna(observed) else bool(observed)
    return normalized is not expected
