from __future__ import annotations

import time
from typing import Any

from datadiff.backends.base import Backend, BackendResult
from datadiff.dsl import Program, TableData, normalize_sort_keys


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
                    cmp = op["cmp"]
                    series = df[col]
                    if cmp == ">": mask = series > val
                    elif cmp == ">=": mask = series >= val
                    elif cmp == "<": mask = series < val
                    elif cmp == "<=": mask = series <= val
                    elif cmp == "==": mask = series == val
                    elif cmp == "!=": mask = series != val
                    else: raise ValueError(cmp)
                    if val is not None:
                        mask = mask & series.notna()
                    df = df[mask.fillna(False)]
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
