from __future__ import annotations

import warnings
from typing import Any

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
from datadiff.dsl import normalize_sort_keys
from datadiff.filtering import parse_filter_comparator
from datadiff.operation_semantics import (
    op_column,
    op_columns,
    op_input_dtype,
    op_kind,
    op_n,
    op_output_alias,
    op_partition_columns,
    op_source,
    op_value,
)
from datadiff.running import running_sum_sort_keys


def polars_agg_expr(pl, column: str, func: str, alias: str):
    if func == "count":
        return pl.col(column).count().cast(pl.Int64).alias(alias)
    if func == "nunique":
        return pl.col(column).drop_nulls().n_unique().cast(pl.Int64).alias(alias)
    if func == "sum":
        return (
            pl.when(pl.col(column).count() == 0)
            .then(None)
            .otherwise(pl.col(column).sum())
            .alias(alias)
        )
    if func == "any":
        return (
            pl.when(pl.col(column).count() == 0)
            .then(None)
            .otherwise(pl.col(column).max())
            .alias(alias)
        )
    if func == "all":
        return (
            pl.when(pl.col(column).count() == 0)
            .then(None)
            .otherwise(pl.col(column).min())
            .alias(alias)
        )
    return getattr(pl.col(column), func)().alias(alias)


def polars_all_not_null(pl, columns: list[str]):
    expr = pl.col(columns[0]).is_not_null()
    for column in columns[1:]:
        expr = expr & pl.col(column).is_not_null()
    return expr


def single_bool_frame(pl, alias: str, value: bool):
    return pl.DataFrame({alias: [value]})


def single_bool_lazy_frame(pl, alias: str, value: bool):
    return single_bool_frame(pl, alias, value).lazy()


def polars_running_sum_expr(pl, op: dict):
    source = pl.col(op_source(op))
    if op_input_dtype(op) == "float32":
        source = source.cast(pl.Float32)
    else:
        source = source.cast(pl.Float64)
    partition_columns = op_partition_columns(op)
    running = source.fill_null(0).cum_sum()
    non_null_count = source.is_not_null().cum_sum()
    if partition_columns:
        running = running.over(partition_columns)
        non_null_count = non_null_count.over(partition_columns)
    expr = pl.when(non_null_count == 0).then(None).otherwise(running)
    return expr.alias(op_column(op))


def polars_filter_expr(col, comparator: str, value):
    parsed = parse_filter_comparator(comparator)
    if parsed is None:
        raise ValueError(comparator)
    if parsed.base == "in_set":
        expr = col.is_in(list(value))
    elif parsed.base == "not_in_set":
        expr = (~col.is_in(list(value))) & col.is_not_null()
    elif parsed.base == "is_null":
        expr = col.is_null()
    elif parsed.base == "is_not_null":
        expr = col.is_not_null()
    elif parsed.base == "bool_predicate":
        expr = col
    elif parsed.base == "range_closed":
        lower, upper = value
        expr = (col >= lower) & (col <= upper)
    elif parsed.base == "str_contains":
        expr = col.str.contains(value, literal=True)
    elif parsed.base == "str_starts_with":
        expr = col.str.starts_with(value)
    elif parsed.base == "str_ends_with":
        expr = col.str.ends_with(value)
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


def apply_polars_common_op(
    pl,
    frame,
    op: dict[str, Any],
    *,
    reverse_division_mode: str,
    groupby_maintain_order: bool | None = None,
):
    kind = op_kind(op)
    if kind == "drop_nulls":
        return frame.drop_nulls(subset=op_columns(op))
    if kind == "filter":
        column, comparator, value = filter_plan(op)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Comparisons with None always result in null.*",
                category=UserWarning,
            )
            expr = polars_filter_expr(pl.col(column), comparator, value)
        return frame.filter(expr)
    if kind == "running_sum":
        keys = running_sum_sort_keys(op)
        return frame.sort(
            [key.column for key in keys],
            descending=[not key.ascending for key in keys],
            nulls_last=[key.nulls == "last" for key in keys],
        ).with_columns(polars_running_sum_expr(pl, op))
    if kind == "select":
        return frame.select(select_columns(op))
    if kind == "distinct":
        columns = distinct_columns(op)
        return frame.select(columns).unique(maintain_order=True)
    if kind == "fill_null":
        column = op_column(op)
        return frame.with_columns(pl.col(column).fill_null(op_value(op)).alias(column))
    if kind == "coalesce":
        alias, sources, has_fallback, fallback = coalesce_plan(op)
        exprs = [pl.col(column) for column in sources]
        if has_fallback:
            exprs.append(pl.lit(fallback))
        return frame.with_columns(pl.coalesce(exprs).alias(alias))
    if kind == "case_when":
        alias, column, comparator, value, then_value, else_value = case_when_plan(op)
        expr = polars_filter_expr(pl.col(column), comparator, value)
        return frame.with_columns(
            pl.when(expr)
            .then(pl.lit(then_value))
            .otherwise(pl.lit(else_value))
            .alias(alias)
        )
    if kind == "sort":
        keys = normalize_sort_keys(op)
        return frame.sort(
            [key.column for key in keys],
            descending=[not key.ascending for key in keys],
            nulls_last=[key.nulls == "last" for key in keys],
        )
    if kind == "limit":
        return frame.head(op_n(op))
    if kind == "offset":
        return frame.slice(op_n(op))
    if kind == "mutate":
        return _apply_polars_mutate(pl, frame, op, reverse_division_mode=reverse_division_mode)
    if kind == "groupby":
        keys, triplets = groupby_plan(op)
        aggs = [polars_agg_expr(pl, column, func, alias) for column, func, alias in triplets]
        if groupby_maintain_order is None:
            return frame.group_by(keys).agg(aggs)
        return frame.group_by(keys, maintain_order=groupby_maintain_order).agg(aggs)
    if kind == "aggregate":
        aggs = [polars_agg_expr(pl, column, func, alias) for column, func, alias in aggregate_triplets(op)]
        return frame.select(aggs)
    return None


def _apply_polars_mutate(pl, frame, op: dict[str, Any], *, reverse_division_mode: str):
    plan = mutate_plan(op)
    expr = _polars_mutate_expr(pl, plan, reverse_division_mode=reverse_division_mode)
    if expr is not None:
        return frame.with_columns(expr)
    if plan.kind == "reverse_division_columns" and reverse_division_mode == "series_rtruediv":
        out = frame[plan.source].__rtruediv__(frame[plan.numerator])
        return frame.with_columns(out.alias(plan.column))
    raise ValueError(plan.kind)


def _polars_mutate_expr(pl, plan: Any, *, reverse_division_mode: str):
    if plan.kind == "add_const":
        return (pl.col(plan.source) + plan.value).alias(plan.column)
    if plan.kind == "arith_const":
        source = pl.col(plan.source)
        if plan.operator == "sub":
            out = source - plan.value
        elif plan.operator == "mul":
            out = source * plan.value
        elif plan.operator == "div":
            out = source / plan.value
        elif plan.operator == "mod":
            out = source % plan.value
        else:
            raise ValueError(plan.operator)
        return out.alias(plan.column)
    if plan.kind == "reverse_division_columns" and reverse_division_mode == "expr_division":
        return (pl.col(plan.numerator) / pl.col(plan.source)).alias(plan.column)
    if plan.kind == "abs":
        return pl.col(plan.source).abs().alias(plan.column)
    if plan.kind == "clip":
        return pl.col(plan.source).clip(plan.lower, plan.upper).alias(plan.column)
    if plan.kind == "bool_not":
        return pl.col(plan.source).not_().alias(plan.column)
    if plan.kind == "cast":
        if plan.target_type == "float":
            out = pl.col(plan.source).cast(pl.Float64)
        elif plan.target_type == "int":
            out = pl.col(plan.source).cast(pl.Int64)
        elif plan.target_type == "str":
            out = pl.col(plan.source).cast(pl.Utf8)
        else:
            raise ValueError(plan.target_type)
        return out.alias(plan.column)
    if plan.kind == "string_length":
        return pl.col(plan.source).str.len_chars().cast(pl.Int64).alias(plan.column)
    if plan.kind == "string_lower":
        return pl.col(plan.source).str.to_lowercase().alias(plan.column)
    if plan.kind == "string_upper":
        return pl.col(plan.source).str.to_uppercase().alias(plan.column)
    if plan.kind == "string_strip":
        return pl.col(plan.source).str.strip_chars().alias(plan.column)
    if plan.kind == "string_null_if_empty":
        return (
            pl.when(pl.col(plan.source) == "")
            .then(None)
            .otherwise(pl.col(plan.source))
            .alias(plan.column)
        )
    if plan.kind == "string_replace":
        return pl.col(plan.source).str.replace_all(plan.old, plan.new, literal=True).alias(plan.column)
    if plan.kind == "string_slice":
        return pl.col(plan.source).str.slice(plan.start, plan.length).alias(plan.column)
    if plan.kind == "string_split_part":
        return pl.col(plan.source).str.split(plan.separator).list.get(plan.index).alias(plan.column)
    if plan.kind == "string_concat":
        return pl.concat_str(
            [pl.col(plan.source), pl.col(plan.other)],
            separator=plan.separator or "",
            ignore_nulls=False,
        ).alias(plan.column)
    if plan.kind == "string_contains":
        return pl.col(plan.source).str.contains(plan.needle, literal=True).alias(plan.column)
    if plan.kind == "string_starts_with":
        return pl.col(plan.source).str.starts_with(plan.needle).alias(plan.column)
    if plan.kind == "string_ends_with":
        return pl.col(plan.source).str.ends_with(plan.needle).alias(plan.column)
    if plan.kind == "date_part":
        spans = {"year": (0, 4), "month": (5, 2), "day": (8, 2)}
        start_idx, length = spans[plan.part]
        return pl.col(plan.source).str.slice(start_idx, length).cast(pl.Int64).alias(plan.column)
    if plan.kind == "string_basename":
        return pl.col(plan.source).str.replace(r"^.*[\\/]", "").alias(plan.column)
    return None
