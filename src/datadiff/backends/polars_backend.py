from __future__ import annotations

import time
import warnings

from datadiff.backends.base import Backend, BackendResult
from datadiff.dsl import Program, TableData, normalize_sort_keys
from datadiff.filtering import parse_filter_comparator
from datadiff.sortedness import is_sorted_values
from datadiff.tuple_logic import evaluate_tuple_absence


class PolarsBackend(Backend):
    name = "polars"

    def _to_df(self, table: TableData):
        import polars as pl

        data = {c.name: [row.get(c.name) for row in table.rows] for c in table.columns}
        schema = {c.name: _polars_dtype(pl, c.type) for c in table.columns}
        return pl.DataFrame(data, schema=schema)

    def run(self, tables: list[TableData], program: Program, timeout_s: float = 5.0) -> BackendResult:
        start = time.perf_counter()
        try:
            import polars as pl
            frames = {table.name: self._to_df(table) for table in tables}
            df = frames[tables[0].name]
            for op in program.operations:
                kind = op["op"]
                if kind == "join":
                    before_cols = set(df.columns)
                    df = df.join(
                        frames[op["table"]],
                        left_on=op["left_on"],
                        right_on=op["right_on"],
                        how=op["how"],
                        suffix="_r",
                        maintain_order="left",
                    )
                    drop_cols = [c for c in df.columns if c.endswith("_r")]
                    if (
                        op["right_on"] != op["left_on"]
                        and op["right_on"] not in before_cols
                        and op["right_on"] in df.columns
                    ):
                        drop_cols.append(op["right_on"])
                    if drop_cols:
                        df = df.drop(drop_cols)
                elif kind == "filter":
                    col = pl.col(op["column"])
                    val = op["value"]
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore",
                            message="Comparisons with None always result in null.*",
                            category=UserWarning,
                        )
                        expr = _polars_filter_expr(col, op["cmp"], val)
                    df = df.filter(expr)
                elif kind == "tuple_absence_filter":
                    mask = _tuple_absence_mask(pl, df, frames[op["table"]], op)
                    df = df.filter(mask)
                elif kind == "running_sum":
                    keys = normalize_sort_keys({"keys": op["order_by"]})
                    df = df.sort(
                        [key.column for key in keys],
                        descending=[not key.ascending for key in keys],
                        nulls_last=[key.nulls == "last" for key in keys],
                    ).with_columns(_polars_running_sum_expr(pl, op))
                elif kind == "sortedness_check":
                    series = df.get_column(op["column"])
                    ok = _polars_is_sorted(series, op)
                    df = pl.DataFrame({op["as"]: [ok]})
                elif kind == "random_case_probe":
                    df = pl.DataFrame({op["as"]: [False]})
                elif kind == "select":
                    df = df.select(list(op["columns"]))
                elif kind == "sort":
                    keys = normalize_sort_keys(op)
                    df = df.sort(
                        [key.column for key in keys],
                        descending=[not key.ascending for key in keys],
                        nulls_last=[key.nulls == "last" for key in keys],
                    )
                elif kind == "limit":
                    df = df.head(int(op["n"]))
                elif kind == "offset":
                    df = df.slice(int(op["n"]))
                elif kind == "mutate":
                    expr = op["expr"]
                    if expr["kind"] == "add_const":
                        df = df.with_columns((pl.col(expr["source"]) + expr["value"]).alias(op["column"]))
                    elif expr["kind"] == "arith_const":
                        src = pl.col(expr["source"])
                        if expr["op"] == "sub":
                            out = src - expr["value"]
                        elif expr["op"] == "mul":
                            out = src * expr["value"]
                        elif expr["op"] == "div":
                            out = src / expr["value"]
                        elif expr["op"] == "mod":
                            out = src % expr["value"]
                        else:
                            raise ValueError(expr["op"])
                        df = df.with_columns(out.alias(op["column"]))
                    elif expr["kind"] == "cast" and expr["to"] == "float":
                        df = df.with_columns(pl.col(expr["source"]).cast(pl.Float64).alias(op["column"]))
                    elif expr["kind"] == "string_length":
                        # Polars string lengths are unsigned by default. Cast to
                        # signed Int64 so later arithmetic follows the DSL's
                        # common signed-integer semantics instead of wrapping.
                        df = df.with_columns(
                            pl.col(expr["source"]).str.len_chars().cast(pl.Int64).alias(op["column"])
                        )
                    elif expr["kind"] == "string_lower":
                        df = df.with_columns(pl.col(expr["source"]).str.to_lowercase().alias(op["column"]))
                    else:
                        raise ValueError(expr["kind"])
                elif kind == "groupby":
                    keys = list(op["keys"])
                    aggs = []
                    for agg in op["aggs"]:
                        col, func, alias = agg["column"], agg["func"], agg["as"]
                        if func == "count":
                            aggs.append(pl.col(col).count().alias(alias))
                        elif func == "nunique":
                            aggs.append(pl.col(col).drop_nulls().n_unique().alias(alias))
                        elif func == "sum":
                            aggs.append(
                                pl.when(pl.col(col).count() == 0)
                                .then(None)
                                .otherwise(pl.col(col).sum())
                                .alias(alias)
                            )
                        else:
                            aggs.append(getattr(pl.col(col), func)().alias(alias))
                    df = df.group_by(keys, maintain_order=True).agg(aggs)
                elif kind == "aggregate":
                    aggs = []
                    for agg in op["aggs"]:
                        col, func, alias = agg["column"], agg["func"], agg["as"]
                        if func == "count":
                            aggs.append(pl.col(col).count().alias(alias))
                        elif func == "nunique":
                            aggs.append(pl.col(col).drop_nulls().n_unique().alias(alias))
                        elif func == "sum":
                            aggs.append(
                                pl.when(pl.col(col).count() == 0)
                                .then(None)
                                .otherwise(pl.col(col).sum())
                                .alias(alias)
                            )
                        else:
                            aggs.append(getattr(pl.col(col), func)().alias(alias))
                    df = df.select(aggs)
                else:
                    raise ValueError(kind)
            return BackendResult(self.name, "ok", data=df, duration_ms=(time.perf_counter()-start)*1000)
        except Exception as exc:  # noqa: BLE001
            return BackendResult(self.name, "error", error_type=type(exc).__name__, error=str(exc), duration_ms=(time.perf_counter()-start)*1000)


class PolarsLazyBackend(PolarsBackend):
    name = "polars_lazy"

    def run(self, tables: list[TableData], program: Program, timeout_s: float = 5.0) -> BackendResult:
        start = time.perf_counter()
        try:
            import polars as pl

            frames = {table.name: self._to_df(table).lazy() for table in tables}
            lf = frames[tables[0].name]
            for op in program.operations:
                kind = op["op"]
                if kind == "join":
                    before_cols = set(lf.collect_schema().names())
                    lf = lf.join(
                        frames[op["table"]],
                        left_on=op["left_on"],
                        right_on=op["right_on"],
                        how=op["how"],
                        suffix="_r",
                    )
                    drop_cols = [c for c in lf.collect_schema().names() if c.endswith("_r")]
                    if (
                        op["right_on"] != op["left_on"]
                        and op["right_on"] not in before_cols
                        and op["right_on"] in lf.collect_schema().names()
                    ):
                        drop_cols.append(op["right_on"])
                    if drop_cols:
                        lf = lf.drop(drop_cols)
                elif kind == "filter":
                    col = pl.col(op["column"])
                    val = op["value"]
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore",
                            message="Comparisons with None always result in null.*",
                            category=UserWarning,
                        )
                        expr = _polars_filter_expr(col, op["cmp"], val)
                    lf = lf.filter(expr)
                elif kind == "tuple_absence_filter":
                    left_df = lf.collect()
                    right_df = frames[op["table"]].collect()
                    mask = _tuple_absence_mask(pl, left_df, right_df, op)
                    lf = left_df.filter(mask).lazy()
                elif kind == "running_sum":
                    keys = normalize_sort_keys({"keys": op["order_by"]})
                    lf = lf.sort(
                        [key.column for key in keys],
                        descending=[not key.ascending for key in keys],
                        nulls_last=[key.nulls == "last" for key in keys],
                    ).with_columns(_polars_running_sum_expr(pl, op))
                elif kind == "sortedness_check":
                    series = lf.collect().get_column(op["column"])
                    ok = _polars_is_sorted(series, op)
                    lf = pl.DataFrame({op["as"]: [ok]}).lazy()
                elif kind == "random_case_probe":
                    lf = pl.DataFrame({op["as"]: [False]}).lazy()
                elif kind == "select":
                    lf = lf.select(list(op["columns"]))
                elif kind == "sort":
                    keys = normalize_sort_keys(op)
                    lf = lf.sort(
                        [key.column for key in keys],
                        descending=[not key.ascending for key in keys],
                        nulls_last=[key.nulls == "last" for key in keys],
                    )
                elif kind == "limit":
                    lf = lf.head(int(op["n"]))
                elif kind == "offset":
                    lf = lf.slice(int(op["n"]))
                elif kind == "mutate":
                    expr = op["expr"]
                    if expr["kind"] == "add_const":
                        lf = lf.with_columns((pl.col(expr["source"]) + expr["value"]).alias(op["column"]))
                    elif expr["kind"] == "arith_const":
                        src = pl.col(expr["source"])
                        if expr["op"] == "sub":
                            out = src - expr["value"]
                        elif expr["op"] == "mul":
                            out = src * expr["value"]
                        elif expr["op"] == "div":
                            out = src / expr["value"]
                        elif expr["op"] == "mod":
                            out = src % expr["value"]
                        else:
                            raise ValueError(expr["op"])
                        lf = lf.with_columns(out.alias(op["column"]))
                    elif expr["kind"] == "cast" and expr["to"] == "float":
                        lf = lf.with_columns(pl.col(expr["source"]).cast(pl.Float64).alias(op["column"]))
                    elif expr["kind"] == "string_length":
                        lf = lf.with_columns(
                            pl.col(expr["source"]).str.len_chars().cast(pl.Int64).alias(op["column"])
                        )
                    elif expr["kind"] == "string_lower":
                        lf = lf.with_columns(pl.col(expr["source"]).str.to_lowercase().alias(op["column"]))
                    else:
                        raise ValueError(expr["kind"])
                elif kind == "groupby":
                    keys = list(op["keys"])
                    aggs = []
                    for agg in op["aggs"]:
                        col, func, alias = agg["column"], agg["func"], agg["as"]
                        if func == "count":
                            aggs.append(pl.col(col).count().alias(alias))
                        elif func == "nunique":
                            aggs.append(pl.col(col).drop_nulls().n_unique().alias(alias))
                        elif func == "sum":
                            aggs.append(
                                pl.when(pl.col(col).count() == 0)
                                .then(None)
                                .otherwise(pl.col(col).sum())
                                .alias(alias)
                            )
                        else:
                            aggs.append(getattr(pl.col(col), func)().alias(alias))
                    lf = lf.group_by(keys).agg(aggs)
                elif kind == "aggregate":
                    aggs = []
                    for agg in op["aggs"]:
                        col, func, alias = agg["column"], agg["func"], agg["as"]
                        if func == "count":
                            aggs.append(pl.col(col).count().alias(alias))
                        elif func == "nunique":
                            aggs.append(pl.col(col).drop_nulls().n_unique().alias(alias))
                        elif func == "sum":
                            aggs.append(
                                pl.when(pl.col(col).count() == 0)
                                .then(None)
                                .otherwise(pl.col(col).sum())
                                .alias(alias)
                            )
                        else:
                            aggs.append(getattr(pl.col(col), func)().alias(alias))
                    lf = lf.select(aggs)
                else:
                    raise ValueError(kind)
            return BackendResult(self.name, "ok", data=lf.collect(), duration_ms=(time.perf_counter()-start)*1000)
        except Exception as exc:  # noqa: BLE001
            return BackendResult(self.name, "error", error_type=type(exc).__name__, error=str(exc), duration_ms=(time.perf_counter()-start)*1000)


def _polars_dtype(pl, kind: str):
    if kind == "int":
        return pl.Int64
    if kind == "float":
        return pl.Float64
    if kind == "bool":
        return pl.Boolean
    return pl.Utf8


def _tuple_absence_mask(pl, df, right_df, op: dict):
    left_columns = list(op["columns"])
    right_columns = list(op["right_columns"])
    right_rows = right_df.select(right_columns).to_dicts()
    mask = [
        evaluate_tuple_absence(row, left_columns, right_rows, right_columns)
        for row in df.to_dicts()
    ]
    return pl.Series("__datadiff_tuple_absence", mask, dtype=pl.Boolean)


def _polars_running_sum_expr(pl, op: dict):
    source = pl.col(op["source"])
    if op.get("input_dtype") == "float32":
        source = source.cast(pl.Float32)
    else:
        source = source.cast(pl.Float64)
    return source.cum_sum().alias(op["column"])


def _polars_is_sorted(series, op: dict) -> bool:
    ascending = bool(op.get("ascending", True))
    nulls = str(op.get("nulls", "last"))
    try:
        return bool(
            series.is_sorted(
                descending=not ascending,
                nulls_last=nulls == "last",
            )
        )
    except TypeError:
        return is_sorted_values(
            series.to_list(),
            ascending=ascending,
            nulls=nulls,
        )


def _polars_filter_expr(col, comparator: str, value):
    parsed = parse_filter_comparator(comparator)
    if parsed is None:
        raise ValueError(comparator)
    if parsed.base == "in_set":
        expr = col.is_in(list(value))
    elif parsed.base == "is_null":
        expr = col.is_null()
    elif parsed.base == "is_not_null":
        expr = col.is_not_null()
    elif parsed.base == "bool_predicate":
        expr = col
    elif parsed.base == "range_closed":
        lower, upper = value
        expr = (col >= lower) & (col <= upper)
    elif parsed.base == ">":
        expr = col > value
    elif parsed.base == ">=":
        expr = col >= value
    elif parsed.base == "<":
        expr = col < value
    elif parsed.base == "<=":
        expr = col <= value
    elif parsed.base == "==":
        expr = col == value
    elif parsed.base == "!=":
        expr = col != value
    else:
        raise ValueError(parsed.base)
    if parsed.truth_test is None or parsed.truth_test == "is_true":
        return expr.fill_null(False)
    if parsed.truth_test == "is_not_true":
        return ~expr.fill_null(False)
    if parsed.truth_test == "is_false":
        return ~expr.fill_null(True)
    if parsed.truth_test == "is_not_false":
        return expr.fill_null(True)
    if parsed.truth_test == "is_unknown":
        return expr.is_null()
    if parsed.truth_test == "is_not_unknown":
        return expr.is_not_null()
    raise ValueError(parsed.truth_test)
