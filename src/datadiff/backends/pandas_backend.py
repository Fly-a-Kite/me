from __future__ import annotations

import time
from typing import Any

from datadiff.backends.base import Backend, BackendResult
from datadiff.dsl import Program, TableData, normalize_sort_keys
from datadiff.filtering import evaluate_filter_predicate
from datadiff.running import sort_rows_for_running, stable_running_sum_values
from datadiff.sortedness import is_sorted_values
from datadiff.tuple_logic import evaluate_tuple_absence


class PandasBackend(Backend):
    name = "pandas"

    def _to_df(self, table: TableData):
        import pandas as pd
        return pd.DataFrame(table.rows, columns=[c.name for c in table.columns])

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
                    mask = series.map(lambda value: evaluate_filter_predicate(value, comparator, val)).astype(bool)
                    df = df[mask]
                elif kind == "tuple_absence_filter":
                    right = frames[op["table"]]
                    right_rows = right[list(op["right_columns"])].to_dict("records")
                    left_columns = list(op["columns"])
                    right_columns = list(op["right_columns"])
                    mask = [
                        evaluate_tuple_absence(row, left_columns, right_rows, right_columns)
                        for row in df.to_dict("records")
                    ]
                    df = df[mask]
                elif kind == "running_sum":
                    columns = [column for column in df.columns if column != op["column"]] + [op["column"]]
                    rows = sort_rows_for_running(df.to_dict("records"), normalize_sort_keys({"keys": op["order_by"]}))
                    values = stable_running_sum_values(rows, op["source"])
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
                elif kind == "dataset_isin_all_match_probe":
                    df = pd.DataFrame([{op["as"]: False}], columns=[op["as"]])
                elif kind == "large_string_partition_probe":
                    df = pd.DataFrame([{op["as"]: False}], columns=[op["as"]])
                elif kind == "hash_pivot_wider_probe":
                    df = pd.DataFrame([{op["as"]: False}], columns=[op["as"]])
                elif kind == "rolling_mean_by_null_count_probe":
                    df = pd.DataFrame([{op["as"]: False}], columns=[op["as"]])
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
                    elif expr["kind"] == "cast" and expr["to"] == "float":
                        df[op["column"]] = df[expr["source"]].astype("float64")
                    elif expr["kind"] == "string_length":
                        df[op["column"]] = df[expr["source"]].str.len()
                    elif expr["kind"] == "string_lower":
                        df[op["column"]] = df[expr["source"]].str.lower()
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
