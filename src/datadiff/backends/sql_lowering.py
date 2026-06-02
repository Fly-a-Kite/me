from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from datadiff.filtering import sql_filter_condition
from datadiff.dsl import SortKey
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
    op_column,
    op_output_alias,
    op_value,
)


@dataclass(frozen=True, slots=True)
class SqlDialect:
    float_cast_type: str
    int_cast_type: str
    str_cast_type: str
    string_slice_fn: str
    string_startswith_fn: str
    string_endswith_fn: str
    date_part_spans: dict[str, tuple[int, int]]
    basename_sql: Callable[[str], str]
    split_part_sql: Callable[[str, str, int], str]
    division_sql: Callable[[str, str], str]
    reverse_division_sql: Callable[[str, str], str]


def render_fill_null_expr(op: Mapping[str, Any], quote: Callable[[str], str], lit: Callable[[Any], str]) -> tuple[str, str]:
    column = op_column(op)
    expr_sql = f"COALESCE(q.{quote(column)}, {lit(op_value(op))})"
    return column, expr_sql


def replace_projection(
    cols: list[str],
    column: str,
    expr_sql: str,
    quote: Callable[[str], str],
) -> tuple[str, list[str]]:
    kept_cols = [col for col in cols if col != column]
    select_parts = [f"q.{quote(col)}" for col in kept_cols]
    select_parts.append(f"{expr_sql} AS {quote(column)}")
    return ", ".join(select_parts), kept_cols + [column]


def replace_visible_column(visible_cols: list[str], column: str) -> list[str]:
    return [col for col in visible_cols if col != column] + [column]


def pending_order_mentions(pending_order: list[SortKey] | None, column: str) -> bool:
    return pending_order is not None and column in {key.column for key in pending_order}


def visible_projection(visible_cols: list[str], quote: Callable[[str], str], *, table_alias: str = "q") -> str:
    return ", ".join(f"{table_alias}.{quote(col)}" for col in visible_cols)


def select_with_pending_order(
    cols: list[str],
    pending_order: list[SortKey] | None,
    hidden_order_cols: list[str],
    quote: Callable[[str], str],
    *,
    table_alias: str = "q",
) -> tuple[str, list[SortKey] | None, list[str]]:
    if pending_order is None:
        projection = ", ".join(f"{table_alias}.{quote(col)}" for col in cols)
        return projection, None, list(hidden_order_cols)
    projection = [f"{table_alias}.{quote(col)}" for col in cols]
    selected = set(cols)
    updated_order: list[SortKey] = []
    next_hidden_order_cols = list(hidden_order_cols)
    for idx, key in enumerate(pending_order):
        if key.column in selected:
            updated_order.append(key)
            continue
        hidden = (
            key.column
            if key.column in next_hidden_order_cols
            else f"__datadiff_order_{len(next_hidden_order_cols)}_{idx}"
        )
        if hidden not in next_hidden_order_cols:
            projection.append(f"{table_alias}.{quote(key.column)} AS {quote(hidden)}")
            next_hidden_order_cols.append(hidden)
        else:
            projection.append(f"{table_alias}.{quote(hidden)}")
        updated_order.append(SortKey(hidden, key.ascending, key.nulls))
    return ", ".join(projection), updated_order, next_hidden_order_cols


@dataclass(slots=True)
class SqlPipelineState:
    current_cols: list[str]
    visible_cols: list[str]
    hidden_order_cols: list[str]
    pending_order: list[SortKey] | None = None

    @classmethod
    def from_columns(cls, cols: Sequence[str]) -> "SqlPipelineState":
        current = list(cols)
        return cls(current_cols=current, visible_cols=list(current), hidden_order_cols=[])

    def reset_projection(self, cols: Sequence[str], *, clear_pending_order: bool = True) -> None:
        next_cols = list(cols)
        self.current_cols = next_cols
        self.visible_cols = list(next_cols)
        self.hidden_order_cols = []
        if clear_pending_order:
            self.pending_order = None

    def reset_to_alias(self, alias: str) -> None:
        self.reset_projection([alias])

    def visible_projection(
        self,
        quote: Callable[[str], str],
        *,
        table_alias: str = "q",
    ) -> str:
        return visible_projection(self.visible_cols, quote, table_alias=table_alias)

    def select_with_pending_order(
        self,
        cols: list[str],
        quote: Callable[[str], str],
        *,
        table_alias: str = "q",
    ) -> str:
        projection, self.pending_order, self.hidden_order_cols = select_with_pending_order(
            cols,
            self.pending_order,
            self.hidden_order_cols,
            quote,
            table_alias=table_alias,
        )
        self.visible_cols = list(cols)
        self.current_cols = cols + [col for col in self.hidden_order_cols if col not in cols]
        return projection

    def drop_hidden_order_cols(self) -> bool:
        if not self.hidden_order_cols:
            return False
        self.current_cols = list(self.visible_cols)
        self.hidden_order_cols = []
        return True

    def freeze_pending_order(self) -> tuple[str, list[SortKey]] | None:
        if self.pending_order is None:
            return None
        prior_order = list(self.pending_order)
        ordinal = f"__datadiff_order_freeze_{len(self.hidden_order_cols)}"
        self.current_cols = [*self.current_cols, ordinal]
        self.hidden_order_cols = [*self.hidden_order_cols, ordinal]
        self.pending_order = [SortKey(ordinal, True, "last")]
        return ordinal, prior_order

    def replace_projection_expr(
        self,
        column: str,
        expr_sql: str,
        quote: Callable[[str], str],
    ) -> str:
        projection, self.current_cols = replace_projection(self.current_cols, column, expr_sql, quote)
        return projection

    def replace_visible_column(self, column: str) -> None:
        self.visible_cols = replace_visible_column(self.visible_cols, column)

    def pending_order_mentions(self, column: str) -> bool:
        return pending_order_mentions(self.pending_order, column)

    def clear_pending_order(self) -> None:
        self.pending_order = None


def render_coalesce_expr(op: Mapping[str, Any], quote: Callable[[str], str], lit: Callable[[Any], str]) -> tuple[str, str]:
    alias = op_output_alias(op)
    parts = [f"q.{quote(column)}" for column in coalesce_sources(op)]
    if coalesce_has_fallback(op):
        parts.append(lit(coalesce_fallback(op)))
    return alias, f"COALESCE({', '.join(parts)})"


def render_case_when_expr(op: Mapping[str, Any], quote: Callable[[str], str], lit: Callable[[Any], str]) -> tuple[str, str]:
    alias = op_output_alias(op)
    condition_sql = sql_filter_condition(
        f"q.{quote(condition_column(op))}",
        lit(condition_value(op)),
        condition_cmp(op),
    )
    expr_sql = f"CASE WHEN {condition_sql} THEN {lit(case_then_value(op))} ELSE {lit(case_else_value(op))} END"
    return alias, expr_sql


def render_mutate_expr(
    op: Mapping[str, Any],
    dialect: SqlDialect,
    quote: Callable[[str], str],
    lit: Callable[[Any], str],
) -> tuple[str, str]:
    column = op_column(op)
    source = expr_source(op)
    kind = expr_kind(op)
    if kind == "add_const":
        return column, f"q.{quote(source)} + {lit(expr_value(op))}"
    if kind == "arith_const":
        operator = expr_operator(op)
        if operator == "div":
            return column, dialect.division_sql(f"q.{quote(source)}", lit(expr_value(op)))
        op_sql = {"sub": "-", "mul": "*", "mod": "%"}[operator]
        return column, f"q.{quote(source)} {op_sql} {lit(expr_value(op))}"
    if kind == "reverse_division_columns":
        return column, dialect.reverse_division_sql(f"q.{quote(expr_numerator(op))}", f"q.{quote(source)}")
    if kind == "abs":
        return column, f"ABS(q.{quote(source)})"
    if kind == "clip":
        source_sql = f"q.{quote(source)}"
        expr_sql = (
            f"CASE WHEN {source_sql} IS NULL THEN NULL "
            f"WHEN {source_sql} < {lit(expr_lower(op))} THEN {lit(expr_lower(op))} "
            f"WHEN {source_sql} > {lit(expr_upper(op))} THEN {lit(expr_upper(op))} "
            f"ELSE {source_sql} END"
        )
        return column, expr_sql
    if kind == "bool_not":
        return column, f"NOT q.{quote(source)}"
    if kind == "cast":
        cast_type = {
            "float": dialect.float_cast_type,
            "int": dialect.int_cast_type,
            "str": dialect.str_cast_type,
        }.get(expr_target_type(op))
        if cast_type is None:
            raise ValueError(expr_target_type(op))
        return column, f"CAST(q.{quote(source)} AS {cast_type})"
    if kind == "string_length":
        return column, f"LENGTH(q.{quote(source)})"
    if kind == "string_lower":
        return column, f"LOWER(q.{quote(source)})"
    if kind == "string_upper":
        return column, f"UPPER(q.{quote(source)})"
    if kind == "string_strip":
        return column, f"TRIM(q.{quote(source)})"
    if kind == "string_null_if_empty":
        return column, f"NULLIF(q.{quote(source)}, '')"
    if kind == "string_replace":
        return column, f"REPLACE(q.{quote(source)}, {lit(expr_old(op))}, {lit(expr_new(op))})"
    if kind == "string_slice":
        start = int(expr_start(op)) + 1
        length = int(expr_length(op))
        return column, f"{dialect.string_slice_fn}(q.{quote(source)}, {start}, {length})"
    if kind == "string_split_part":
        return column, dialect.split_part_sql(f"q.{quote(source)}", lit(expr_separator(op)), int(expr_index(op)))
    if kind == "string_concat":
        return column, f"q.{quote(source)} || {lit(expr_separator(op))} || q.{quote(expr_other(op))}"
    if kind == "string_contains":
        expr_sql = (
            f"CASE WHEN q.{quote(source)} IS NULL THEN NULL "
            f"ELSE INSTR(q.{quote(source)}, {lit(expr_needle(op))}) > 0 END"
        )
        return column, expr_sql
    if kind == "string_starts_with":
        return column, dialect.string_startswith_fn(f"q.{quote(source)}", lit(expr_needle(op)))
    if kind == "string_ends_with":
        return column, dialect.string_endswith_fn(f"q.{quote(source)}", lit(expr_needle(op)))
    if kind == "date_part":
        start_idx, length = dialect.date_part_spans[expr_part(op)]
        return column, f"CAST({dialect.string_slice_fn}(q.{quote(source)}, {start_idx}, {length}) AS {dialect.int_cast_type})"
    if kind == "string_basename":
        return column, dialect.basename_sql(f"q.{quote(source)}")
    raise ValueError(kind)


def render_aggregate_sql(
    op: Mapping[str, Any],
    agg_expr: Callable[[str, str], str],
    quote: Callable[[str], str],
) -> tuple[list[str], list[str]]:
    agg_sql = [f"{agg_expr(aggregate_column(agg), aggregate_func(agg))} AS {quote(aggregate_alias(agg))}" for agg in aggregate_specs(op)]
    aliases = [aggregate_alias(agg) for agg in aggregate_specs(op)]
    return agg_sql, aliases


def render_groupby_sql(
    op: Mapping[str, Any],
    agg_expr: Callable[[str, str], str],
    quote: Callable[[str], str],
    keys: list[str],
) -> tuple[str, list[str]]:
    key_sql = ", ".join(quote(key) for key in keys)
    agg_sql, aliases = render_aggregate_sql(op, agg_expr, quote)
    select_sql = ", ".join([part for part in [key_sql, *agg_sql] if part])
    return select_sql, aliases
