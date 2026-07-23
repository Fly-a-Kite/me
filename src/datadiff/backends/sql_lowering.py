from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from datadiff.dsl import SortKey
from datadiff.filtering import parse_filter_comparator, sql_filter_condition
from datadiff.join_keys import join_key_pairs
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
    op_table,
    op_value,
)


class HarnessLoweringError(RuntimeError):
    """A violation in DataDiff's SQL-lowering harness, never a backend result.

    Backends catch this as a normal execution error, which keeps the raw
    observation lossless.  The classifier recognises the error type and
    excludes it before semantic-boundary rules are considered.
    """


@dataclass(frozen=True, slots=True)
class SqlDialect:
    logical_type_sql: Mapping[str, str]
    string_slice_fn: str
    string_length_sql: Callable[[str], str]
    string_contains_sql: Callable[[str, str], str]
    string_startswith_fn: Callable[[str, str], str]
    string_endswith_fn: Callable[[str, str], str]
    date_part_spans: dict[str, tuple[int, int]]
    basename_sql: Callable[[str], str]
    split_part_sql: Callable[[str, str, int], str]
    division_sql: Callable[[str, str], str]
    reverse_division_sql: Callable[[str, str], str]
    semantic_cast_sql: Callable[[str, str | None, str], str] | None = None

    def cast_type(self, logical_type: str) -> str:
        try:
            return self.logical_type_sql[logical_type]
        except KeyError as exc:
            raise ValueError(f"SQL dialect has no mapping for logical type {logical_type!r}") from exc

    def render_semantic_cast(
        self,
        expression_sql: str,
        source_type: str | None,
        target_type: str,
    ) -> str:
        if self.semantic_cast_sql is not None:
            return self.semantic_cast_sql(expression_sql, source_type, target_type)
        return cast_logical_expression(expression_sql, target_type, self)


def cast_logical_expression(expression_sql: str, logical_type: str, dialect: SqlDialect) -> str:
    return f"CAST({expression_sql} AS {dialect.cast_type(logical_type)})"


def typed_column_sql(
    alias: str,
    column: str,
    logical_type: str,
    dialect: SqlDialect,
    quote: Callable[[str], str],
) -> str:
    prefix = f"{alias}." if alias else ""
    return cast_logical_expression(f"{prefix}{quote(column)}", logical_type, dialect)


def render_filter_condition(
    column_sql: str,
    literal_sql: str,
    comparator: Any,
    dialect: SqlDialect,
) -> str:
    parsed = parse_filter_comparator(comparator)
    if parsed is None:
        raise ValueError(comparator)
    if parsed.base == "str_contains":
        return dialect.string_contains_sql(column_sql, literal_sql)
    if parsed.base == "str_starts_with":
        return dialect.string_startswith_fn(column_sql, literal_sql)
    if parsed.base == "str_ends_with":
        return dialect.string_endswith_fn(column_sql, literal_sql)
    return sql_filter_condition(column_sql, literal_sql, comparator)


def join_projection_columns(
    right_columns: Sequence[str],
    right_keys: Sequence[str],
    left_columns: Sequence[str],
) -> list[str]:
    excluded = set(right_keys) | set(left_columns)
    return [column for column in right_columns if column not in excluded]


def render_join_right_projection(
    right_columns: Sequence[str],
    right_keys: Sequence[str],
    left_columns: Sequence[str],
    quote: Callable[[str], str],
    *,
    table_alias: str = "r",
) -> tuple[str, list[str]]:
    projected = join_projection_columns(right_columns, right_keys, left_columns)
    parts = [f"{table_alias}.{quote(column)} AS {quote(column)}" for column in projected]
    return (", " + ", ".join(parts) if parts else ""), projected


def render_join_condition(
    op: Mapping[str, Any],
    dialect: SqlDialect,
    quote: Callable[[str], str],
    left_types: Mapping[str, str],
    right_types: Mapping[str, str],
    *,
    left_alias: str = "q",
    right_alias: str = "r",
) -> str:
    left_keys, right_keys = join_key_pairs(op)
    predicates: list[str] = []
    for left, right in zip(left_keys, right_keys):
        left_type = left_types.get(left)
        right_type = right_types.get(right)
        if left_type is None or right_type is None:
            raise ValueError(
                f"join key type is undeclared: {left!r}:{left_type!r}, {right!r}:{right_type!r}"
            )
        if left_type != right_type:
            raise ValueError(
                f"join key type mismatch: {left!r}:{left_type} != {right!r}:{right_type}"
            )
        predicates.append(
            f"{typed_column_sql(left_alias, left, left_type, dialect, quote)} = "
            f"{typed_column_sql(right_alias, right, right_type, dialect, quote)}"
        )
    if not predicates:
        raise ValueError("join requires at least one key pair")
    return " AND ".join(predicates)


def render_semi_anti_join_condition(
    op: Mapping[str, Any],
    kind: str,
    dialect: SqlDialect,
    quote: Callable[[str], str],
    left_types: Mapping[str, str],
    right_types: Mapping[str, str],
) -> str:
    if kind not in {"semi_join", "anti_join"}:
        raise ValueError(kind)
    left_keys, right_keys = join_key_pairs(op)
    equality = render_join_condition(op, dialect, quote, left_types, right_types)
    not_null = [
        f"{typed_column_sql('r', right, right_types[right], dialect, quote)} IS NOT NULL"
        for right in right_keys
    ]
    exists_sql = (
        f"EXISTS (SELECT 1 FROM {quote(op_table(op))} r WHERE "
        f"{' AND '.join([*not_null, equality])})"
    )
    return exists_sql if kind == "semi_join" else f"NOT {exists_sql}"


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
        return projection, None, []
    projection = [f"{table_alias}.{quote(col)}" for col in cols]
    selected = set(cols)
    updated_order: list[SortKey] = []
    next_hidden_order_cols: list[str] = []
    unavailable_aliases = set(cols)
    for idx, key in enumerate(pending_order):
        if key.column in selected:
            updated_order.append(key)
            continue
        # A previous projection may already have materialised this key under a
        # hidden alias.  Otherwise choose a fresh alias that cannot collide
        # with a visible or existing hidden column.
        hidden = key.column if key.column in hidden_order_cols else ""
        if not hidden:
            # Retain the historic deterministic spelling for reproducible SQL
            # exports while still probing for collisions below.
            base = f"__datadiff_order_{len(hidden_order_cols) + len(next_hidden_order_cols)}_{idx}"
            hidden = base
            suffix = 0
            while hidden in unavailable_aliases or hidden in hidden_order_cols:
                suffix += 1
                hidden = f"{base}_{suffix}"
        if hidden not in next_hidden_order_cols:
            projection.append(f"{table_alias}.{quote(key.column)} AS {quote(hidden)}")
            next_hidden_order_cols.append(hidden)
        else:
            projection.append(f"{table_alias}.{quote(hidden)}")
        unavailable_aliases.add(hidden)
        updated_order.append(SortKey(hidden, key.ascending, key.nulls))
    return ", ".join(projection), updated_order, next_hidden_order_cols


@dataclass(slots=True)
class SqlPipelineState:
    current_cols: list[str]
    visible_cols: list[str]
    hidden_order_cols: list[str]
    pending_order: list[SortKey] | None = None

    def __post_init__(self) -> None:
        self.assert_invariants()

    @classmethod
    def from_columns(cls, cols: Sequence[str]) -> "SqlPipelineState":
        current = list(cols)
        return cls(current_cols=current, visible_cols=list(current), hidden_order_cols=[])

    def assert_invariants(self) -> None:
        """Validate the SQL symbol table at every lowering boundary.

        ``current_cols`` represents exactly the subquery output.  The two
        projections are distinct subsets of it, and no pending ``ORDER BY``
        expression may reference a historical alias that is no longer output.
        """

        current = list(self.current_cols)
        visible = list(self.visible_cols)
        hidden = list(self.hidden_order_cols)
        if len(current) != len(set(current)):
            raise HarnessLoweringError(f"duplicate current SQL columns: {current!r}")
        if len(visible) != len(set(visible)):
            raise HarnessLoweringError(f"duplicate visible SQL columns: {visible!r}")
        if len(hidden) != len(set(hidden)):
            raise HarnessLoweringError(f"duplicate hidden order columns: {hidden!r}")
        current_set = set(current)
        if not set(visible).issubset(current_set):
            raise HarnessLoweringError(
                f"visible columns are not subquery outputs: {sorted(set(visible) - current_set)!r}"
            )
        if not set(hidden).issubset(current_set):
            raise HarnessLoweringError(
                f"hidden order columns are not subquery outputs: {sorted(set(hidden) - current_set)!r}"
            )
        overlap = set(visible) & set(hidden)
        if overlap:
            raise HarnessLoweringError(
                f"visible and hidden SQL columns overlap: {sorted(overlap)!r}"
            )
        if self.pending_order is not None:
            missing = [key.column for key in self.pending_order if key.column not in current_set]
            if missing:
                raise HarnessLoweringError(
                    f"pending order references missing SQL columns: {missing!r}"
                )

    def reset_projection(self, cols: Sequence[str], *, clear_pending_order: bool = True) -> None:
        next_cols = list(cols)
        self.current_cols = next_cols
        self.visible_cols = list(next_cols)
        self.hidden_order_cols = []
        if clear_pending_order:
            self.pending_order = None
        self.assert_invariants()

    def reset_to_alias(self, alias: str) -> None:
        self.reset_projection([alias])

    def visible_projection(
        self,
        quote: Callable[[str], str],
        *,
        table_alias: str = "q",
    ) -> str:
        self.assert_invariants()
        return visible_projection(self.visible_cols, quote, table_alias=table_alias)

    def select_with_pending_order(
        self,
        cols: list[str],
        quote: Callable[[str], str],
        *,
        table_alias: str = "q",
    ) -> str:
        self.assert_invariants()
        if len(cols) != len(set(cols)):
            raise HarnessLoweringError(f"select contains duplicate columns: {cols!r}")
        unavailable = [column for column in cols if column not in self.visible_cols]
        if unavailable:
            raise HarnessLoweringError(
                f"select references columns absent from visible projection: {unavailable!r}"
            )
        projection, self.pending_order, self.hidden_order_cols = select_with_pending_order(
            cols,
            self.pending_order,
            self.hidden_order_cols,
            quote,
            table_alias=table_alias,
        )
        self.visible_cols = list(cols)
        self.current_cols = cols + [col for col in self.hidden_order_cols if col not in cols]
        self.assert_invariants()
        return projection

    def drop_hidden_order_cols(self) -> bool:
        if not self.hidden_order_cols:
            return False
        self.current_cols = list(self.visible_cols)
        self.hidden_order_cols = []
        # Once the materialised ordering keys have been removed there is no
        # sound pending ORDER BY left to apply.
        self.pending_order = None
        self.assert_invariants()
        return True

    def freeze_pending_order(self) -> tuple[str, list[SortKey]] | None:
        self.assert_invariants()
        if self.pending_order is None:
            return None
        prior_order = list(self.pending_order)
        ordinal = f"__datadiff_order_freeze_{len(self.hidden_order_cols)}"
        suffix = 0
        while ordinal in self.current_cols:
            suffix += 1
            ordinal = f"__datadiff_order_freeze_{len(self.hidden_order_cols)}_{suffix}"
        self.current_cols = [*self.current_cols, ordinal]
        self.hidden_order_cols = [*self.hidden_order_cols, ordinal]
        self.pending_order = [SortKey(ordinal, True, "last")]
        self.assert_invariants()
        return ordinal, prior_order

    def replace_projection_expr(
        self,
        column: str,
        expr_sql: str,
        quote: Callable[[str], str],
    ) -> str:
        self.assert_invariants()
        # ``mutate`` creates a fresh visible alias, while fill/coalesce/case
        # replace an existing one.  Both transitions are valid as long as the
        # resulting projection is a real output symbol table.
        projection, self.current_cols = replace_projection(self.current_cols, column, expr_sql, quote)
        self.assert_invariants()
        return projection

    def replace_visible_column(self, column: str) -> None:
        self.visible_cols = replace_visible_column(self.visible_cols, column)
        self.assert_invariants()

    def pending_order_mentions(self, column: str) -> bool:
        return pending_order_mentions(self.pending_order, column)

    def clear_pending_order(self) -> None:
        self.pending_order = None
        self.assert_invariants()


def render_coalesce_expr(op: Mapping[str, Any], quote: Callable[[str], str], lit: Callable[[Any], str]) -> tuple[str, str]:
    alias = op_output_alias(op)
    parts = [f"q.{quote(column)}" for column in coalesce_sources(op)]
    if coalesce_has_fallback(op):
        parts.append(lit(coalesce_fallback(op)))
    return alias, f"COALESCE({', '.join(parts)})"


def render_case_when_expr(
    op: Mapping[str, Any],
    dialect: SqlDialect,
    quote: Callable[[str], str],
    lit: Callable[[Any], str],
) -> tuple[str, str]:
    alias = op_output_alias(op)
    condition_sql = render_filter_condition(
        f"q.{quote(condition_column(op))}",
        lit(condition_value(op)),
        condition_cmp(op),
        dialect,
    )
    expr_sql = f"CASE WHEN {condition_sql} THEN {lit(case_then_value(op))} ELSE {lit(case_else_value(op))} END"
    return alias, expr_sql


def render_mutate_expr(
    op: Mapping[str, Any],
    dialect: SqlDialect,
    quote: Callable[[str], str],
    lit: Callable[[Any], str],
    column_types: Mapping[str, str] | None = None,
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
        target_type = expr_target_type(op)
        source_type = (column_types or {}).get(source)
        return column, dialect.render_semantic_cast(
            f"q.{quote(source)}",
            source_type,
            target_type,
        )
    if kind == "string_length":
        return column, dialect.string_length_sql(f"q.{quote(source)}")
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
        return column, dialect.string_contains_sql(f"q.{quote(source)}", lit(expr_needle(op)))
    if kind == "string_starts_with":
        return column, dialect.string_startswith_fn(f"q.{quote(source)}", lit(expr_needle(op)))
    if kind == "string_ends_with":
        return column, dialect.string_endswith_fn(f"q.{quote(source)}", lit(expr_needle(op)))
    if kind == "date_part":
        start_idx, length = dialect.date_part_spans[expr_part(op)]
        expression_sql = f"{dialect.string_slice_fn}(q.{quote(source)}, {start_idx}, {length})"
        return column, cast_logical_expression(expression_sql, "int", dialect)
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
