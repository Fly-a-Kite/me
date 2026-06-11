from __future__ import annotations

import math
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from datadiff.backends.base import Backend, BackendResult, PreparedTable, prepare_table
from datadiff.backends.dataframe_semantics import running_sum_plan, tuple_absence_plan
from datadiff.backends.polars_lowering import (
    apply_polars_common_op,
    polars_agg_expr,
    polars_all_not_null,
    polars_filter_expr as _polars_filter_expr,
    polars_running_sum_expr as _polars_running_sum_expr,
    single_bool_frame as _single_bool_frame,
    single_bool_lazy_frame as _single_bool_lazy_frame,
)
from datadiff.backends.probe_semantics import EXTENDED_FALSE_PROBE_KINDS, resolve_bool_probe
from datadiff.csv_roundtrip import (
    csv_long_numeric_roundtrip_mismatch,
    csv_long_numeric_values,
    long_numeric_csv_path,
)
from datadiff.dsl import Program, TableData, normalize_sort_keys
from datadiff.join_keys import join_key_arg, join_key_pairs
from datadiff.operation_semantics import (
    join_how,
    op_ascending,
    op_column,
    op_columns,
    op_kind,
    op_nulls,
    op_right_columns,
    op_table,
    op_values,
)
from datadiff.running import running_sum_sort_keys
from datadiff.sortedness import is_sorted_values
from datadiff.tuple_logic import evaluate_tuple_absence
from datadiff.windowing import row_number_filter_rows

def _polars_probe_handlers(
    pl,
    op: dict[str, Any],
    *,
    lazy: bool,
    collect: Callable | None = None,
):
    return {
        "group_quantile_probe": lambda: _polars_group_quantile_key_mismatch(pl, op, lazy=lazy),
        "window_avg_probe": lambda: _polars_sql_window_avg_mismatch(pl),
        "timestamp_precision_filter_probe": (
            lambda: _polars_timestamp_precision_filter_mismatch(pl, collect=collect) if lazy
            else _polars_timestamp_precision_filter_mismatch(pl)
        ),
        "series_rtruediv_probe": lambda: _polars_series_rtruediv_mismatch(pl) if not lazy else False,
        "float_wrap_probe": lambda: _polars_float_wrap_mismatch(pl, lazy=lazy),
        "polars_timezone_filter_probe": lambda: _polars_timezone_filter_mismatch(pl, lazy=lazy),
        "empty_literal_groupby_probe": lambda: _polars_empty_literal_groupby_mismatch(pl, lazy=lazy),
        "rolling_mean_by_null_count_probe": lambda: _polars_rolling_mean_by_null_count_mismatch(pl, lazy=lazy),
        "csv_long_numeric_roundtrip_probe": (
            lambda: _polars_csv_long_numeric_roundtrip_mismatch(pl, op, lazy=lazy, collect=collect)
        ),
    }


class PolarsBackend(Backend):
    name = "polars"

    def _to_df(self, table: TableData | PreparedTable):
        import polars as pl

        prepared = prepare_table(table)
        data = {column.name: prepared.columns_data[column.name] for column in prepared.columns}
        schema = {column.name: _polars_dtype(pl, column.type) for column in prepared.columns}
        return pl.DataFrame(data, schema=schema)

    def run(
        self,
        tables: list[TableData | PreparedTable],
        program: Program,
        timeout_s: float = 5.0,
    ) -> BackendResult:
        start = time.perf_counter()
        try:
            import polars as pl
            frames = {table.name: self._to_df(table) for table in tables}
            df = frames[tables[0].name]
            for op in program.operations:
                kind = op_kind(op)
                if kind == "join":
                    before_cols = set(df.columns)
                    left_keys, right_keys = join_key_pairs(op)
                    df = df.join(
                        frames[op_table(op)],
                        left_on=join_key_arg(left_keys),
                        right_on=join_key_arg(right_keys),
                        how=join_how(op),
                        suffix="_r",
                        maintain_order="left",
                    )
                    drop_cols = [c for c in df.columns if c.endswith("_r")]
                    for left_key, right_key in zip(left_keys, right_keys):
                        if right_key != left_key and right_key not in before_cols and right_key in df.columns:
                            drop_cols.append(right_key)
                    if drop_cols:
                        df = df.drop(drop_cols)
                elif kind == "union_all":
                    df = pl.concat([df, frames[op_table(op)].select(df.columns)], how="vertical")
                elif kind in {"semi_join", "anti_join"}:
                    how = "semi" if kind == "semi_join" else "anti"
                    left_keys, right_key_cols = join_key_pairs(op)
                    right_key_frame = (
                        frames[op_table(op)]
                        .filter(polars_all_not_null(pl, right_key_cols))
                        .select(right_key_cols)
                        .unique(maintain_order=True)
                    )
                    df = df.join(
                        right_key_frame,
                        left_on=join_key_arg(left_keys),
                        right_on=join_key_arg(right_key_cols),
                        how=how,
                        maintain_order="left",
                    )
                elif kind == "drop_nulls":
                    df = df.drop_nulls(subset=op_columns(op))
                elif kind == "tuple_absence_filter":
                    right_table, _, _ = tuple_absence_plan(op)
                    mask = _tuple_absence_mask(pl, df, frames[right_table], op)
                    df = df.filter(mask)
                elif kind == "row_number_filter":
                    df = _polars_row_number_filter(pl, df, op)
                elif kind == "running_sum":
                    plan = running_sum_plan(op)
                    df = apply_polars_common_op(
                        pl,
                        df,
                        op,
                        reverse_division_mode="series_rtruediv",
                        groupby_maintain_order=True,
                    )
                elif kind == "sortedness_check":
                    alias = op_output_alias(op)
                    series = df.get_column(op_column(op))
                    ok = _polars_is_sorted(series, op)
                    df = _single_bool_frame(pl, alias, ok)
                else:
                    probe_value = resolve_bool_probe(
                        kind,
                        handlers=_polars_probe_handlers(pl, op, lazy=False),
                        fallback_false_kinds=EXTENDED_FALSE_PROBE_KINDS,
                    )
                    if probe_value is not None:
                        df = _single_bool_frame(pl, op_output_alias(op), probe_value)
                    else:
                        next_df = apply_polars_common_op(
                            pl,
                            df,
                            op,
                            reverse_division_mode="series_rtruediv",
                            groupby_maintain_order=True,
                        )
                        if next_df is None:
                            raise ValueError(kind)
                        df = next_df
            return BackendResult(self.name, "ok", data=df, duration_ms=(time.perf_counter()-start)*1000)
        except Exception as exc:  # noqa: BLE001
            return BackendResult(self.name, "error", error_type=type(exc).__name__, error=str(exc), duration_ms=(time.perf_counter()-start)*1000)


class PolarsLazyBackend(PolarsBackend):
    name = "polars_lazy"
    collect_engine: str | None = None

    def _collect_lazy_frame(self, lazy_frame):
        if self.collect_engine is None:
            return lazy_frame.collect()
        return lazy_frame.collect(engine=self.collect_engine)

    def run(
        self,
        tables: list[TableData | PreparedTable],
        program: Program,
        timeout_s: float = 5.0,
    ) -> BackendResult:
        start = time.perf_counter()
        try:
            import polars as pl

            frames = {table.name: self._to_df(table).lazy() for table in tables}
            lf = frames[tables[0].name]
            for op in program.operations:
                kind = op_kind(op)
                if kind == "join":
                    before_cols = set(lf.collect_schema().names())
                    left_keys, right_keys = join_key_pairs(op)
                    lf = lf.join(
                        frames[op_table(op)],
                        left_on=join_key_arg(left_keys),
                        right_on=join_key_arg(right_keys),
                        how=join_how(op),
                        suffix="_r",
                    )
                    drop_cols = [c for c in lf.collect_schema().names() if c.endswith("_r")]
                    current_names = lf.collect_schema().names()
                    for left_key, right_key in zip(left_keys, right_keys):
                        if right_key != left_key and right_key not in before_cols and right_key in current_names:
                            drop_cols.append(right_key)
                    if drop_cols:
                        lf = lf.drop(drop_cols)
                elif kind == "union_all":
                    lf = pl.concat([lf, frames[op_table(op)].select(lf.collect_schema().names())], how="vertical")
                elif kind in {"semi_join", "anti_join"}:
                    how = "semi" if kind == "semi_join" else "anti"
                    left_keys, right_key_cols = join_key_pairs(op)
                    right_key_frame = (
                        frames[op_table(op)]
                        .filter(polars_all_not_null(pl, right_key_cols))
                        .select(right_key_cols)
                        .unique(maintain_order=True)
                    )
                    lf = lf.join(
                        right_key_frame,
                        left_on=join_key_arg(left_keys),
                        right_on=join_key_arg(right_key_cols),
                        how=how,
                    )
                elif kind == "drop_nulls":
                    lf = lf.drop_nulls(subset=op_columns(op))
                elif kind == "tuple_absence_filter":
                    left_df = lf.collect()
                    right_df = frames[op_table(op)].collect()
                    mask = _tuple_absence_mask(pl, left_df, right_df, op)
                    lf = left_df.filter(mask).lazy()
                elif kind == "row_number_filter":
                    lf = _polars_row_number_filter(pl, lf.collect(), op).lazy()
                elif kind == "running_sum":
                    lf = apply_polars_common_op(
                        pl,
                        lf,
                        op,
                        reverse_division_mode="expr_division",
                        groupby_maintain_order=None,
                    )
                elif kind == "sortedness_check":
                    alias = op_output_alias(op)
                    series = lf.collect().get_column(op_column(op))
                    ok = _polars_is_sorted(series, op)
                    lf = _single_bool_lazy_frame(pl, alias, ok)
                else:
                    probe_value = resolve_bool_probe(
                        kind,
                        handlers=_polars_probe_handlers(
                            pl,
                            op,
                            lazy=True,
                            collect=self._collect_lazy_frame,
                        ),
                        fallback_false_kinds=EXTENDED_FALSE_PROBE_KINDS,
                    )
                    if probe_value is not None:
                        lf = _single_bool_lazy_frame(pl, op_output_alias(op), probe_value)
                    else:
                        next_lf = apply_polars_common_op(
                            pl,
                            lf,
                            op,
                            reverse_division_mode="expr_division",
                            groupby_maintain_order=None,
                        )
                        if next_lf is None:
                            raise ValueError(kind)
                        lf = next_lf
            return BackendResult(self.name, "ok", data=self._collect_lazy_frame(lf), duration_ms=(time.perf_counter()-start)*1000)
        except Exception as exc:  # noqa: BLE001
            return BackendResult(self.name, "error", error_type=type(exc).__name__, error=str(exc), duration_ms=(time.perf_counter()-start)*1000)


class PolarsStreamingBackend(PolarsLazyBackend):
    name = "polars_streaming"
    collect_engine = "streaming"


def _polars_dtype(pl, kind: str):
    if kind == "int":
        return pl.Int64
    if kind == "float":
        return pl.Float64
    if kind == "bool":
        return pl.Boolean
    return pl.Utf8


def _tuple_absence_mask(pl, df, right_df, op: dict):
    left_columns = op_columns(op)
    right_columns = op_right_columns(op)
    right_rows = right_df.select(right_columns).to_dicts()
    mask = [
        evaluate_tuple_absence(row, left_columns, right_rows, right_columns)
        for row in df.to_dicts()
    ]
    return pl.Series("__datadiff_tuple_absence", mask, dtype=pl.Boolean)


def _polars_row_number_filter(pl, df, op: dict):
    rows = row_number_filter_rows(df.to_dicts(), op)
    return pl.DataFrame(rows, schema=df.schema)

def _polars_is_sorted(series, op: dict) -> bool:
    ascending = op_ascending(op)
    nulls = op_nulls(op)
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


def _polars_group_quantile_key_mismatch(pl, op: dict, *, lazy: bool) -> bool:
    values = [float(value) for value in (op_values(op) or [1, 2, 3])]
    quantiles = [float(value) for value in (op_quantiles(op) or [0.0, 0.5, 1.0])]
    expected = [_nearest_quantile(values, quantile) for quantile in quantiles]
    value_frame = pl.DataFrame({"x": values})
    quantile_frame = pl.DataFrame({"q": quantiles})
    if lazy:
        value_frame = value_frame.lazy()
        quantile_frame = quantile_frame.lazy()
    result = (
        value_frame.join(quantile_frame, how="cross")
        .group_by("q")
        .agg(pl.col("x").quantile(pl.col("q").first()).alias("observed_quantile"))
        .sort("q")
    )
    if lazy:
        result = result.collect()
    observed = result.get_column("observed_quantile").to_list()
    return len(observed) != len(expected) or any(
        actual is None or not math.isclose(float(actual), want, rel_tol=0.0, abs_tol=1e-12)
        for actual, want in zip(observed, expected)
    )


def _nearest_quantile(values: list[float], quantile: float) -> float:
    sorted_values = sorted(values)
    index = int(math.floor(quantile * (len(sorted_values) - 1) + 0.5))
    index = max(0, min(index, len(sorted_values) - 1))
    return float(sorted_values[index])


def _polars_sql_window_avg_mismatch(pl) -> bool:
    expected = [10.0, 15.0, 20.0, 25.0, 30.0]
    ctx = pl.SQLContext()
    ctx.register(
        "df",
        pl.DataFrame(
            {
                "foo": [1, 2, 3, 4, 5],
                "bar": [10.0, 20.0, 30.0, 40.0, 50.0],
            }
        ).lazy(),
    )
    result = ctx.execute(
        """
        SELECT
            foo,
            bar,
            AVG(bar) OVER (
                ORDER BY foo
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS mean_avg
        FROM df
        ORDER BY foo
        """
    ).collect()
    observed = result.get_column("mean_avg").to_list()
    return len(observed) != len(expected) or any(
        actual is None or not math.isclose(float(actual), want, rel_tol=0.0, abs_tol=1e-12)
        for actual, want in zip(observed, expected)
    )


def _polars_series_rtruediv_mismatch(pl) -> bool:
    expected = [2.0, 1.5, 4.0 / 3.0]
    observed = pl.Series([1, 2, 3]).__rtruediv__(pl.Series([2, 3, 4])).to_list()
    return len(observed) != len(expected) or any(
        actual is None or not math.isclose(float(actual), want, rel_tol=0.0, abs_tol=1e-12)
        for actual, want in zip(observed, expected)
    )


def _polars_float_wrap_mismatch(pl, *, lazy: bool) -> bool:
    expected = [100, 44]
    frame = pl.DataFrame({"float_value": [100.0, 300.0], "int_value": [100, 300]})
    if lazy:
        frame = frame.lazy()
    result = frame.with_columns(
        wrapped_float=pl.col("float_value").cast(pl.UInt8, wrap_numerical=True),
        wrapped_int=pl.col("int_value").cast(pl.UInt8, wrap_numerical=True),
    )
    if lazy:
        result = result.collect()
    observed_float = result.get_column("wrapped_float").to_list()
    observed_int = result.get_column("wrapped_int").to_list()
    return observed_float != expected or observed_int != expected


def _polars_timestamp_precision_filter_mismatch(pl, collect=None) -> bool:
    frame = pl.DataFrame({"ts": [1]}, schema={"ts": pl.Datetime("us")})
    predicate = pl.col("ts") < pl.lit(1_500, dtype=pl.Datetime("ns"))
    if collect is None:
        result = frame.filter(predicate)
    else:
        result = collect(frame.lazy().filter(predicate))
    return result.height != 1


def _polars_timezone_filter_mismatch(pl, *, lazy: bool) -> bool:
    london = ZoneInfo("Europe/London")
    frame = pl.DataFrame(
        {
            "ts": [
                datetime(2024, 6, 1, 0, 30, tzinfo=ZoneInfo("UTC")),
                datetime(2024, 5, 31, 22, 30, tzinfo=ZoneInfo("UTC")),
            ]
        }
    )
    if lazy:
        frame = frame.lazy()
    result = frame.with_columns(
        local=pl.col("ts").dt.convert_time_zone("Europe/London")
    ).filter(
        pl.col("local") >= pl.lit(datetime(2024, 6, 1, 0, 0, tzinfo=london))
    )
    if lazy:
        result = result.collect()
    return result.height != 1


def _polars_empty_literal_groupby_mismatch(pl, *, lazy: bool) -> bool:
    frame = pl.DataFrame({})
    if lazy:
        frame = frame.lazy()
    result = frame.group_by(pl.lit("A")).agg(pl.len())
    if lazy:
        result = result.collect()
    return result.height != 0


def _polars_rolling_mean_by_null_count_mismatch(pl, *, lazy: bool) -> bool:
    from datetime import datetime

    frame = pl.DataFrame(
        {
            "timestamp": pl.datetime_range(
                datetime(year=2026, month=1, day=9, hour=5, minute=23, second=37),
                datetime(year=2026, month=1, day=9, hour=5, minute=23, second=39),
                interval="1s",
                time_unit="ms",
                eager=True,
            ),
            "power": [303, 498, None],
        }
    )
    if lazy:
        frame = frame.lazy()
    result = frame.with_columns(
        observed=pl.col("power").rolling_mean_by(
            "timestamp",
            window_size="2s",
            min_samples=2,
        )
    )
    if lazy:
        result = result.collect()
    return result.get_column("observed").to_list() != [None, 400.5, None]


def _polars_csv_long_numeric_roundtrip_mismatch(pl, op: dict, *, lazy: bool, collect=None) -> bool:
    expected_values = csv_long_numeric_values(op)
    with long_numeric_csv_path(expected_values) as csv_path:
        if lazy:
            frame = pl.scan_csv(str(csv_path))
            frame = collect(frame) if collect is not None else frame.collect()
        else:
            frame = pl.read_csv(str(csv_path))
        observed_values = frame.get_column("value").to_list()
    return csv_long_numeric_roundtrip_mismatch(observed_values, expected_values)
