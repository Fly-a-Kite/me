from __future__ import annotations

import time
from typing import Any

from datadiff.backends.base import Backend, BackendResult
from datadiff.csv_roundtrip import (
    csv_long_numeric_roundtrip_mismatch,
    csv_long_numeric_values,
    long_numeric_csv_path,
)
from datadiff.dsl import Program, TableData, normalize_sort_keys
from datadiff.filtering import evaluate_filter_predicate
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


class PandasBackend(Backend):
    name = "pandas"

    def _to_df(self, table: TableData):
        import pandas as pd
        data = {}
        for column in table.columns:
            values = [row.get(column.name) for row in table.rows]
            if column.type == "int":
                data[column.name] = pd.array(values, dtype="Int64")
            elif column.type == "bool":
                data[column.name] = pd.array(values, dtype="boolean")
            elif column.type == "str":
                data[column.name] = pd.array(values, dtype="string")
            else:
                data[column.name] = values
        return pd.DataFrame(data, columns=[c.name for c in table.columns])

    def run(self, tables: list[TableData], program: Program, timeout_s: float = 5.0) -> BackendResult:
        start = time.perf_counter()
        try:
            import pandas as pd  # noqa: F401
            frames = {table.name: self._to_df(table) for table in tables}
            df = frames[tables[0].name]
            for op in program.operations:
                kind = op["op"]
                if kind == "join":
                    right = frames[op["table"]]
                    before_cols = set(df.columns)
                    df = df.merge(
                        right,
                        how=op["how"],
                        left_on=op["left_on"],
                        right_on=op["right_on"],
                        suffixes=("", "_r"),
                        sort=False,
                    )
                    drop_cols = [c for c in df.columns if str(c).endswith("_r")]
                    if (
                        op["right_on"] != op["left_on"]
                        and op["right_on"] not in before_cols
                        and op["right_on"] in df.columns
                    ):
                        drop_cols.append(op["right_on"])
                    if drop_cols:
                        df = df.drop(columns=drop_cols)
                elif kind == "filter":
                    col = op["column"]
                    val = op["value"]
                    comparator = op["cmp"]
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
                    right = frames[op["table"]]
                    right_rows = right[list(op["right_columns"])].to_dict("records")
                    left_columns = list(op["columns"])
                    right_columns = list(op["right_columns"])
                    keep_mask = pd.Series(
                        [
                            evaluate_tuple_absence(row, left_columns, right_rows, right_columns)
                            for row in df.to_dict("records")
                        ],
                        index=df.index,
                        dtype=bool,
                    )
                    df = df.loc[keep_mask]
                elif kind == "row_number_filter":
                    rows = row_number_filter_rows(df.to_dict("records"), op)
                    df = pd.DataFrame(rows, columns=list(df.columns))
                elif kind == "running_sum":
                    columns = [column for column in df.columns if column != op["column"]] + [op["column"]]
                    rows = sort_rows_for_running(df.to_dict("records"), running_sum_sort_keys(op))
                    values = stable_running_sum_values(rows, op["source"], running_sum_partition_columns(op))
                    rows = [{**row, op["column"]: value} for row, value in zip(rows, values)]
                    df = pd.DataFrame(rows, columns=columns)
                elif kind == "sortedness_check":
                    ok = is_sorted_values(
                        df[op["column"]].tolist(),
                        ascending=bool(op.get("ascending", True)),
                        nulls=str(op.get("nulls", "last")),
                    )
                    df = pd.DataFrame([{op["as"]: ok}], columns=[op["as"]])
                elif kind == "random_case_probe":
                    df = pd.DataFrame([{op["as"]: False}], columns=[op["as"]])
                elif kind == "group_quantile_probe":
                    df = pd.DataFrame([{op["as"]: False}], columns=[op["as"]])
                elif kind == "scalar_subquery_probe":
                    df = pd.DataFrame([{op["as"]: False}], columns=[op["as"]])
                elif kind == "window_avg_probe":
                    df = pd.DataFrame([{op["as"]: False}], columns=[op["as"]])
                elif kind == "struct_distinct_probe":
                    df = pd.DataFrame([{op["as"]: False}], columns=[op["as"]])
                elif kind == "bit_compare_probe":
                    df = pd.DataFrame([{op["as"]: False}], columns=[op["as"]])
                elif kind == "round_even_probe":
                    df = pd.DataFrame([{op["as"]: False}], columns=[op["as"]])
                elif kind == "float_literal_precision_probe":
                    df = pd.DataFrame([{op["as"]: False}], columns=[op["as"]])
                elif kind == "timestamp_precision_filter_probe":
                    df = pd.DataFrame([{op["as"]: False}], columns=[op["as"]])
                elif kind == "series_rtruediv_probe":
                    df = pd.DataFrame([{op["as"]: False}], columns=[op["as"]])
                elif kind == "uint64_isin_probe":
                    import numpy as np

                    observed = pd.Series([635554097106142143], dtype="UInt64").isin(
                        np.array([635554097106142079])
                    ).iloc[0]
                    df = pd.DataFrame([{op["as"]: bool(observed)}], columns=[op["as"]])
                elif kind == "tuple_anti_null_probe":
                    df = pd.DataFrame([{op["as"]: False}], columns=[op["as"]])
                elif kind == "setop_all_duplicate_probe":
                    df = pd.DataFrame([{op["as"]: False}], columns=[op["as"]])
                elif kind == "json_predicate_order_probe":
                    df = pd.DataFrame([{op["as"]: False}], columns=[op["as"]])
                elif kind == "sparse_mask_probe":
                    import numpy as np

                    sparse = pd.arrays.SparseArray([1, 2, 3, 4, np.nan, np.nan], fill_value=np.nan)
                    mask = sparse > [3, 3, 4, 1, 0, 0]
                    observed = sparse[mask].to_numpy().tolist()
                    expected = [4.0]
                    mismatch = len(observed) != len(expected) or any(
                        left != right for left, right in zip(observed, expected)
                    )
                    df = pd.DataFrame([{op["as"]: mismatch}], columns=[op["as"]])
                elif kind == "float_wrap_probe":
                    df = pd.DataFrame([{op["as"]: False}], columns=[op["as"]])
                elif kind == "index_bool_probe":
                    observed = pd.Index([3, 5, 8], name="i") == 5
                    df = pd.DataFrame([{op["as"]: not isinstance(observed, pd.Index)}], columns=[op["as"]])
                elif kind == "empty_literal_groupby_probe":
                    df = pd.DataFrame([{op["as"]: False}], columns=[op["as"]])
                elif kind == "arrow_string_eq_sum_probe":
                    mismatch = _pandas_arrow_string_eq_sum_mismatch(pd)
                    df = pd.DataFrame([{op["as"]: mismatch}], columns=[op["as"]])
                elif kind == "arrow_timestamp_loc_slice_probe":
                    mismatch = _pandas_arrow_timestamp_loc_slice_mismatch(pd)
                    df = pd.DataFrame([{op["as"]: mismatch}], columns=[op["as"]])
                elif kind == "arrow_timestamp_index_attr_probe":
                    mismatch = _pandas_arrow_timestamp_index_attr_mismatch(pd)
                    df = pd.DataFrame([{op["as"]: mismatch}], columns=[op["as"]])
                elif kind == "eval_inplace_alias_probe":
                    mismatch = _pandas_eval_inplace_alias_mismatch(pd)
                    df = pd.DataFrame([{op["as"]: mismatch}], columns=[op["as"]])
                elif kind == "bool_reduction_skipna_probe":
                    mismatch = _pandas_bool_reduction_skipna_mismatch(pd)
                    df = pd.DataFrame([{op["as"]: mismatch}], columns=[op["as"]])
                elif kind == "dataset_isin_all_match_probe":
                    df = pd.DataFrame([{op["as"]: False}], columns=[op["as"]])
                elif kind == "run_end_null_compute_probe":
                    df = pd.DataFrame([{op["as"]: False}], columns=[op["as"]])
                elif kind == "large_string_partition_probe":
                    df = pd.DataFrame([{op["as"]: False}], columns=[op["as"]])
                elif kind == "hash_pivot_wider_probe":
                    df = pd.DataFrame([{op["as"]: False}], columns=[op["as"]])
                elif kind == "rolling_mean_by_null_count_probe":
                    df = pd.DataFrame([{op["as"]: False}], columns=[op["as"]])
                elif kind == "csv_long_numeric_roundtrip_probe":
                    expected_values = csv_long_numeric_values(op)
                    with long_numeric_csv_path(expected_values) as csv_path:
                        observed_values = pd.read_csv(csv_path)["value"].tolist()
                    mismatch = csv_long_numeric_roundtrip_mismatch(observed_values, expected_values)
                    df = pd.DataFrame([{op["as"]: mismatch}], columns=[op["as"]])
                elif kind == "select":
                    df = df[list(op["columns"])]
                elif kind == "sort":
                    for key in reversed(normalize_sort_keys(op)):
                        df = df.sort_values(
                            key.column,
                            ascending=key.ascending,
                            na_position=key.nulls,
                            kind="mergesort",
                        )
                elif kind == "limit":
                    df = df.head(int(op["n"]))
                elif kind == "offset":
                    df = df.iloc[int(op["n"]):]
                elif kind == "mutate":
                    expr = op["expr"]
                    df = df.copy()
                    if expr["kind"] == "add_const":
                        df[op["column"]] = df[expr["source"]] + expr["value"]
                    elif expr["kind"] == "arith_const":
                        if expr["op"] == "sub":
                            df[op["column"]] = df[expr["source"]] - expr["value"]
                        elif expr["op"] == "mul":
                            df[op["column"]] = df[expr["source"]] * expr["value"]
                        elif expr["op"] == "div":
                            df[op["column"]] = df[expr["source"]] / expr["value"]
                        elif expr["op"] == "mod":
                            df[op["column"]] = df[expr["source"]] % expr["value"]
                        else:
                            raise ValueError(expr["op"])
                    elif expr["kind"] == "reverse_division_columns":
                        df[op["column"]] = df[expr["numerator"]] / df[expr["source"]]
                    elif expr["kind"] == "cast" and expr["to"] == "float":
                        df[op["column"]] = df[expr["source"]].astype("float64")
                    elif expr["kind"] == "string_length":
                        df[op["column"]] = df[expr["source"]].str.len()
                    elif expr["kind"] == "string_lower":
                        df[op["column"]] = df[expr["source"]].str.lower()
                    elif expr["kind"] == "string_basename":
                        df[op["column"]] = df[expr["source"]].map(path_basename)
                    else:
                        raise ValueError(expr["kind"])
                elif kind == "groupby":
                    keys = list(op["keys"])
                    group = df.groupby(keys, dropna=False, sort=False)
                    pieces = []
                    for agg in op["aggs"]:
                        col, func, alias = agg["column"], agg["func"], agg["as"]
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
                    for agg in op["aggs"]:
                        col, func, alias = agg["column"], agg["func"], agg["as"]
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
                    df = pd.DataFrame([values], columns=[agg["as"] for agg in op["aggs"]])
                else:
                    raise ValueError(kind)
            return BackendResult(self.name, "ok", data=df, duration_ms=(time.perf_counter()-start)*1000)
        except Exception as exc:  # noqa: BLE001
            return BackendResult(self.name, "error", error_type=type(exc).__name__, error=str(exc), duration_ms=(time.perf_counter()-start)*1000)


def _predicate_values(series):
    array = getattr(series, "array", None)
    if array is not None:
        return array.tolist()
    return series.tolist()


def _pandas_arrow_string_eq_sum_mismatch(pd) -> bool:
    values = pd.array(["a"], dtype="string[pyarrow]")
    try:
        observed = (values == values).sum()
    except AttributeError:
        return True
    return int(observed) != 1


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


def _pandas_bool_reduction_skipna_mismatch(pd) -> bool:
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
