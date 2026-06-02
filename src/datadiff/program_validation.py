from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from datadiff.dsl import SortKey, TableData, normalize_sort_keys
from datadiff.expression_semantics import expr_output_type
from datadiff.filtering import filter_comparator_supports_type, is_filter_comparator, parse_filter_comparator
from datadiff.identifiers import is_reserved_output_name
from datadiff.join_keys import join_key_pairs
from datadiff.operation_semantics import (
    aggregate_alias,
    aggregate_column,
    aggregate_func,
    aggregate_specs,
    case_else_value,
    case_then_value,
    coalesce_fallback,
    coalesce_sources,
    condition_cmp,
    condition_column,
    condition_value,
    groupby_keys,
    join_how,
    normalized_order_by_keys,
    op_kind,
    op_ascending,
    op_column,
    op_columns,
    op_input_dtype,
    op_n,
    op_nulls,
    op_output_alias,
    op_partition_columns,
    op_right_columns,
    op_source,
    op_table,
    op_value,
)
from datadiff.operation_type_semantics import aggregate_accepts_type, aggregate_result_type, case_when_output_type
from datadiff.program_state import ProgramState
from datadiff.util import unique_preserve_order


@dataclass(slots=True)
class ValidationContext:
    tables: dict[str, TableData]
    state: ProgramState
    errors: list[str] = field(default_factory=list)


def validate_core_operation(
    ctx: ValidationContext,
    op: Any,
    idx: int,
) -> bool:
    kind = op_kind(op)
    if kind == "join":
        return _validate_join(ctx, op, idx)
    if kind in {"semi_join", "anti_join"}:
        return _validate_semi_anti_join(ctx, op, idx)
    if kind == "union_all":
        return _validate_union_all(ctx, op, idx)
    if kind == "drop_nulls":
        return _validate_drop_nulls(ctx, op, idx)
    if kind == "filter":
        return _validate_filter(ctx, op, idx)
    if kind == "tuple_absence_filter":
        return _validate_tuple_absence_filter(ctx, op, idx)
    if kind == "row_number_filter":
        return _validate_row_number_filter(ctx, op, idx)
    if kind == "running_sum":
        return _validate_running_sum(ctx, op, idx)
    if kind == "sortedness_check":
        return _validate_sortedness_check(ctx, op, idx)
    if kind == "select":
        return _validate_select(ctx, op, idx)
    if kind == "sort":
        return _validate_sort(ctx, op, idx)
    if kind == "limit":
        return _validate_limit(ctx, op, idx)
    if kind == "offset":
        return _validate_offset(ctx, op, idx)
    if kind == "mutate":
        return _validate_mutate(ctx, op, idx)
    if kind == "distinct":
        return _validate_distinct(ctx, op, idx)
    if kind == "fill_null":
        return _validate_fill_null(ctx, op, idx)
    if kind == "coalesce":
        return _validate_coalesce(ctx, op, idx)
    if kind == "case_when":
        return _validate_case_when(ctx, op, idx)
    if kind == "groupby":
        return _validate_groupby(ctx, op, idx)
    if kind == "aggregate":
        return _validate_aggregate(ctx, op, idx)
    return False


def filter_literal_error(column_type: str, comparator: Any, value: Any) -> str:
    if not is_filter_comparator(comparator):
        return f"unsupported comparator {comparator!r}"
    if not filter_comparator_supports_type(column_type, comparator):
        return f"comparator {comparator!r} is not supported for {column_type} filter"
    parsed = parse_filter_comparator(comparator)
    if parsed is not None and parsed.base in {"in_set", "not_in_set"}:
        if not isinstance(value, list) or not value:
            return f"filter literal {value!r} is not a non-empty list for {parsed.base}"
        if any(item is None for item in value):
            return f"{parsed.base} filter literals must not contain NULL"
        for item in value:
            item_error = scalar_filter_literal_error(column_type, item)
            if item_error:
                return item_error
        return ""
    if parsed is not None and parsed.base == "range_closed":
        if not isinstance(value, list) or len(value) != 2:
            return f"filter literal {value!r} must contain two bounds for range_closed"
        if any(item is None for item in value):
            return "range_closed filter bounds must not contain NULL"
        for item in value:
            item_error = scalar_filter_literal_error(column_type, item)
            if item_error:
                return item_error
        if value[0] > value[1]:
            return "range_closed lower bound must be <= upper bound"
        return ""
    if parsed is not None and parsed.base in {"str_contains", "str_starts_with", "str_ends_with"}:
        if not isinstance(value, str) or value == "":
            return f"filter literal {value!r} must be a non-empty string for {parsed.base}"
        return ""
    if parsed is not None and parsed.base in {"is_null", "is_not_null"}:
        return "" if value is None else f"filter literal {value!r} must be NULL for {parsed.base}"
    if parsed is not None and parsed.base == "bool_predicate":
        return "" if value is None else f"filter literal {value!r} must be NULL for {comparator}"
    if value is None:
        return ""
    return scalar_filter_literal_error(column_type, value)


def scalar_filter_literal_error(column_type: str, value: Any) -> str:
    if column_type == "str" and not isinstance(value, str):
        return f"filter literal {value!r} is not compatible with str column"
    if column_type == "bool" and not isinstance(value, bool):
        return f"filter literal {value!r} is not compatible with bool column"
    if column_type in {"int", "float"} and (isinstance(value, bool) or not isinstance(value, (int, float))):
        return f"filter literal {value!r} is not compatible with {column_type} column"
    return ""


def valid_float_literal_text(value: Any) -> bool:
    text = str(value or "").strip()
    if not text or not any(ch.isdigit() for ch in text):
        return False
    allowed = set("0123456789+-.eE")
    if any(ch not in allowed for ch in text):
        return False
    try:
        float(text)
    except ValueError:
        return False
    return True


def valid_group_quantile_values(values: Any, quantiles: Any) -> bool:
    if not isinstance(values, list) or not isinstance(quantiles, list):
        return False
    if len(values) < 2 or len(quantiles) < 2:
        return False
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
        return False
    return all(
        not isinstance(quantile, bool)
        and isinstance(quantile, (int, float))
        and 0.0 <= float(quantile) <= 1.0
        for quantile in quantiles
    )


def valid_digit_string_values(values: Any) -> bool:
    return (
        isinstance(values, list)
        and bool(values)
        and all(isinstance(value, str) and value.isdigit() for value in values)
    )


def _validate_join(ctx: ValidationContext, op: Any, idx: int) -> bool:
    right = ctx.tables.get(op_table(op))
    if right is None:
        ctx.errors.append(f"op {idx}: unknown join table {op_table(op)!r}")
        return False
    left_keys, right_keys = join_key_pairs(op)
    right_cols = {column.name for column in right.columns}
    right_types = {column.name: column.type for column in right.columns}
    ok = True
    if not left_keys or not right_keys or len(left_keys) != len(right_keys):
        ctx.errors.append(f"op {idx}: join key count mismatch")
        ok = False
    if len(unique_preserve_order(left_keys)) != len(left_keys):
        ctx.errors.append(f"op {idx}: join left keys contain duplicates")
        ok = False
    if len(unique_preserve_order(right_keys)) != len(right_keys):
        ctx.errors.append(f"op {idx}: join right keys contain duplicates")
        ok = False
    for left_key, right_key in zip(left_keys, right_keys):
        if left_key not in ctx.state.available:
            ctx.errors.append(f"op {idx}: join left key {left_key!r} is unavailable")
            ok = False
            continue
        if right_key not in right_cols:
            ctx.errors.append(f"op {idx}: join right key {right_key!r} is unavailable")
            ok = False
            continue
        if ctx.state.column_types.get(left_key) != right_types.get(right_key):
            ctx.errors.append(
                f"op {idx}: join key type mismatch: {left_key!r}:{ctx.state.column_types.get(left_key)}"
                f"!={right_key!r}:{right_types.get(right_key)}"
            )
            ok = False
    if join_how(op) not in {"inner", "left"}:
        ctx.errors.append(f"op {idx}: unsupported join kind {join_how(op)!r}")
        ok = False
    return ok


def _validate_semi_anti_join(ctx: ValidationContext, op: Any, idx: int) -> bool:
    kind = op_kind(op)
    right = ctx.tables.get(op_table(op))
    if right is None:
        ctx.errors.append(f"op {idx}: unknown {kind} table {op_table(op)!r}")
        return False
    left_keys, right_keys = join_key_pairs(op)
    right_types = {column.name: column.type for column in right.columns}
    ok = True
    if not left_keys or len(left_keys) != len(right_keys):
        ctx.errors.append(f"op {idx}: {kind} join key arity mismatch")
        return False
    missing_left = [left_key for left_key in left_keys if left_key not in ctx.state.available]
    missing_right = [right_key for right_key in right_keys if right_key not in right_types]
    if missing_left:
        ctx.errors.append(f"op {idx}: {kind} left keys unavailable: {missing_left}")
        ok = False
    if missing_right:
        ctx.errors.append(f"op {idx}: {kind} right keys unavailable: {missing_right}")
        ok = False
    mismatched = [
        f"{left_key!r}:{ctx.state.column_types.get(left_key)}!={right_key!r}:{right_types.get(right_key)}"
        for left_key, right_key in zip(left_keys, right_keys)
        if left_key in ctx.state.available
        and right_key in right_types
        and ctx.state.column_types.get(left_key) != right_types.get(right_key)
    ]
    if mismatched:
        ctx.errors.append(f"op {idx}: {kind} key type mismatch: {mismatched}")
        ok = False
    return ok


def _validate_union_all(ctx: ValidationContext, op: Any, idx: int) -> bool:
    right = ctx.tables.get(op_table(op))
    if right is None:
        ctx.errors.append(f"op {idx}: unknown union_all table {op_table(op)!r}")
        return False
    right_types = {column.name: column.type for column in right.columns}
    missing = [column for column in ctx.state.available if column not in right_types]
    if missing:
        ctx.errors.append(f"op {idx}: union_all columns unavailable in right table: {missing}")
    mismatched = [
        f"{column!r}:{ctx.state.column_types.get(column)}!={right_types[column]}"
        for column in ctx.state.available
        if column in right_types and ctx.state.column_types.get(column) != right_types[column]
    ]
    if mismatched:
        ctx.errors.append(f"op {idx}: union_all column type mismatch: {mismatched}")
    return not missing and not mismatched


def _validate_drop_nulls(ctx: ValidationContext, op: Any, idx: int) -> bool:
    cols = op_columns(op)
    missing = [column for column in cols if column not in ctx.state.available]
    if missing:
        ctx.errors.append(f"op {idx}: drop_nulls columns unavailable: {missing}")
    if not cols:
        ctx.errors.append(f"op {idx}: drop_nulls has no columns")
    if len(unique_preserve_order(cols)) != len(cols):
        ctx.errors.append(f"op {idx}: drop_nulls contains duplicate columns")
    return not missing and bool(cols) and len(unique_preserve_order(cols)) == len(cols)


def _validate_filter(ctx: ValidationContext, op: Any, idx: int) -> bool:
    column = op_column(op)
    ok = True
    if column not in ctx.state.available:
        ctx.errors.append(f"op {idx}: filter column {column!r} is unavailable")
        ok = False
    comparator = condition_cmp(op)
    if not is_filter_comparator(comparator):
        ctx.errors.append(f"op {idx}: unsupported comparator {comparator!r}")
        ok = False
    column_type = ctx.state.column_types.get(column)
    if column_type is not None:
        literal_error = filter_literal_error(column_type, comparator, op_value(op))
        if literal_error:
            ctx.errors.append(f"op {idx}: {literal_error}")
            ok = False
    return ok


def _validate_tuple_absence_filter(ctx: ValidationContext, op: Any, idx: int) -> bool:
    right = ctx.tables.get(op_table(op))
    if right is None:
        ctx.errors.append(f"op {idx}: unknown tuple absence table {op_table(op)!r}")
        return False
    columns = op_columns(op)
    right_columns = op_right_columns(op)
    right_types = {column.name: column.type for column in right.columns}
    ok = True
    if not columns:
        ctx.errors.append(f"op {idx}: tuple absence filter has no columns")
        ok = False
    if len(columns) != len(right_columns):
        ctx.errors.append(f"op {idx}: tuple absence column count mismatch")
        ok = False
    if len(unique_preserve_order(columns)) != len(columns):
        ctx.errors.append(f"op {idx}: tuple absence filter contains duplicate left columns")
        ok = False
    missing = [column for column in columns if column not in ctx.state.available]
    if missing:
        ctx.errors.append(f"op {idx}: tuple absence columns unavailable: {missing}")
        ok = False
    missing_right = [column for column in right_columns if column not in right_types]
    if missing_right:
        ctx.errors.append(f"op {idx}: tuple absence right columns unavailable: {missing_right}")
        ok = False
    for left, right_column in zip(columns, right_columns):
        left_type = ctx.state.column_types.get(left)
        right_type = right_types.get(right_column)
        if left_type is not None and right_type is not None and left_type != right_type:
            ctx.errors.append(
                f"op {idx}: tuple absence type mismatch {left!r}:{left_type} vs {right_column!r}:{right_type}"
            )
            ok = False
    return ok


def _validate_row_number_filter(ctx: ValidationContext, op: Any, idx: int) -> bool:
    partition_by = op_partition_columns(op)
    ok = True
    if len(unique_preserve_order(partition_by)) != len(partition_by):
        ctx.errors.append(f"op {idx}: row_number_filter partition_by contains duplicate columns")
        ok = False
    missing_partition = [column for column in partition_by if column not in ctx.state.available]
    if missing_partition:
        ctx.errors.append(f"op {idx}: row_number_filter partition_by columns unavailable: {missing_partition}")
        ok = False
    try:
        order_keys = normalized_order_by_keys(op)
    except ValueError as exc:
        ctx.errors.append(f"op {idx}: invalid row_number_filter order_by: {exc}")
        return False
    order_columns = [key.column for key in order_keys]
    missing_order = [column for column in order_columns if column not in ctx.state.available]
    if missing_order:
        ctx.errors.append(f"op {idx}: row_number_filter order_by columns unavailable: {missing_order}")
        ok = False
    if not order_columns:
        ctx.errors.append(f"op {idx}: row_number_filter has no order_by columns")
        ok = False
    if len(unique_preserve_order(order_columns)) != len(order_columns):
        ctx.errors.append(f"op {idx}: row_number_filter order_by contains duplicate columns")
        ok = False
    comparator = condition_cmp(op)
    if comparator not in {"==", "<", "<="}:
        ctx.errors.append(f"op {idx}: row_number_filter has unsupported comparator {comparator!r}")
        ok = False
    try:
        if int(op_value(op) if op_value(op) is not None else 0) <= 0:
            ctx.errors.append(f"op {idx}: row_number_filter value must be positive")
            ok = False
    except (TypeError, ValueError):
        ctx.errors.append(f"op {idx}: row_number_filter value must be an integer")
        ok = False
    return ok


def _validate_running_sum(ctx: ValidationContext, op: Any, idx: int) -> bool:
    source = op_source(op)
    column = op_column(op)
    ok = True
    if source not in ctx.state.available:
        ctx.errors.append(f"op {idx}: running_sum source {source!r} is unavailable")
        ok = False
    elif source not in ctx.state.numeric:
        ctx.errors.append(f"op {idx}: running_sum source {source!r} is not numeric")
        ok = False
    if not column:
        ctx.errors.append(f"op {idx}: running_sum output column is empty")
        ok = False
    elif is_reserved_output_name(column):
        ctx.errors.append(f"op {idx}: running_sum output column {column!r} is reserved")
        ok = False
    try:
        order_keys = normalized_order_by_keys(op)
    except ValueError as exc:
        ctx.errors.append(f"op {idx}: invalid running_sum order_by: {exc}")
        return False
    partition_by = op_partition_columns(op)
    missing_partition = [name for name in partition_by if name not in ctx.state.available]
    if missing_partition:
        ctx.errors.append(f"op {idx}: running_sum partition_by columns unavailable: {missing_partition}")
        ok = False
    if len(unique_preserve_order(partition_by)) != len(partition_by):
        ctx.errors.append(f"op {idx}: running_sum partition_by contains duplicate columns")
        ok = False
    order_columns = [key.column for key in order_keys]
    missing_order = [name for name in order_columns if name not in ctx.state.available]
    if missing_order:
        ctx.errors.append(f"op {idx}: running_sum order_by columns unavailable: {missing_order}")
        ok = False
    if not order_columns:
        ctx.errors.append(f"op {idx}: running_sum has no order_by columns")
        ok = False
    if len(unique_preserve_order(order_columns)) != len(order_columns):
        ctx.errors.append(f"op {idx}: running_sum order_by contains duplicate columns")
        ok = False
    return ok


def _validate_sortedness_check(ctx: ValidationContext, op: Any, idx: int) -> bool:
    column = op_column(op)
    alias = op_output_alias(op)
    ascending = op.get("ascending", True)
    nulls = op.get("nulls", "last")
    ok = True
    if column not in ctx.state.available:
        ctx.errors.append(f"op {idx}: sortedness_check column {column!r} is unavailable")
        ok = False
    if not alias:
        ctx.errors.append(f"op {idx}: sortedness_check output alias is empty")
        ok = False
    elif is_reserved_output_name(alias):
        ctx.errors.append(f"op {idx}: sortedness_check output alias {alias!r} is reserved")
        ok = False
    if not isinstance(ascending, bool):
        ctx.errors.append(f"op {idx}: sortedness_check ascending must be boolean")
        ok = False
    if nulls not in {"first", "last"}:
        ctx.errors.append(f"op {idx}: sortedness_check nulls must be 'first' or 'last'")
        ok = False
    return ok


def _validate_select(ctx: ValidationContext, op: Any, idx: int) -> bool:
    cols = op_columns(op)
    missing = [column for column in cols if column not in ctx.state.available]
    if missing:
        ctx.errors.append(f"op {idx}: select columns unavailable: {missing}")
    if not cols:
        ctx.errors.append(f"op {idx}: select has no columns")
    if len(unique_preserve_order(cols)) != len(cols):
        ctx.errors.append(f"op {idx}: select contains duplicate columns")
    return not missing and bool(cols) and len(unique_preserve_order(cols)) == len(cols)


def _validate_sort(ctx: ValidationContext, op: Any, idx: int) -> bool:
    try:
        keys = normalize_sort_keys(op)
    except ValueError as exc:
        ctx.errors.append(f"op {idx}: invalid sort keys: {exc}")
        return False
    cols = [key.column for key in keys]
    missing = [column for column in cols if column not in ctx.state.available]
    if missing:
        ctx.errors.append(f"op {idx}: sort columns unavailable: {missing}")
    if not cols:
        ctx.errors.append(f"op {idx}: sort has no columns")
    if len(unique_preserve_order(cols)) != len(cols):
        ctx.errors.append(f"op {idx}: sort contains duplicate columns")
    return not missing and bool(cols) and len(unique_preserve_order(cols)) == len(cols)


def _validate_limit(ctx: ValidationContext, op: Any, idx: int) -> bool:
    try:
        if op_n(op, -1) < 0:
            ctx.errors.append(f"op {idx}: negative limit")
            return False
    except (TypeError, ValueError):
        ctx.errors.append(f"op {idx}: non-integer limit {op.get('n')!r}")
        return False
    return True


def _validate_offset(ctx: ValidationContext, op: Any, idx: int) -> bool:
    try:
        if op_n(op, -1) < 0:
            ctx.errors.append(f"op {idx}: negative offset")
            return False
    except (TypeError, ValueError):
        ctx.errors.append(f"op {idx}: non-integer offset {op.get('n')!r}")
        return False
    return True


def _validate_mutate(ctx: ValidationContext, op: Any, idx: int) -> bool:
    expr = getattr(op, "expression", None) or op.get("expr", {})
    out_type = expr_output_type(expr, ctx.state.column_types)
    if out_type is None:
        ctx.errors.append(f"op {idx}: invalid mutate expression {expr!r}")
        return False
    column = op_column(op)
    if not column:
        ctx.errors.append(f"op {idx}: mutate output column is empty")
        return False
    if is_reserved_output_name(column):
        ctx.errors.append(f"op {idx}: mutate output column {column!r} is reserved")
        return False
    return True


def _validate_distinct(ctx: ValidationContext, op: Any, idx: int) -> bool:
    cols = op_columns(op)
    missing = [column for column in cols if column not in ctx.state.available]
    if missing:
        ctx.errors.append(f"op {idx}: distinct columns unavailable: {missing}")
    if not cols:
        ctx.errors.append(f"op {idx}: distinct has no columns")
    if len(unique_preserve_order(cols)) != len(cols):
        ctx.errors.append(f"op {idx}: distinct contains duplicate columns")
    return not missing and bool(cols) and len(unique_preserve_order(cols)) == len(cols)


def _validate_fill_null(ctx: ValidationContext, op: Any, idx: int) -> bool:
    column = op_column(op)
    if column not in ctx.state.available:
        ctx.errors.append(f"op {idx}: fill_null column unavailable: {column!r}")
        return False
    value = op_value(op)
    if value is None:
        ctx.errors.append(f"op {idx}: fill_null value must not be NULL")
        return False
    literal_error = scalar_filter_literal_error(ctx.state.column_types.get(column, "derived"), value)
    if literal_error:
        ctx.errors.append(f"op {idx}: fill_null literal {value!r} is incompatible with {ctx.state.column_types.get(column, 'derived')}")
        return False
    return True


def _validate_coalesce(ctx: ValidationContext, op: Any, idx: int) -> bool:
    cols = op_columns(op)
    alias = op_output_alias(op)
    missing = [column for column in cols if column not in ctx.state.available]
    ok = True
    if missing:
        ctx.errors.append(f"op {idx}: coalesce columns unavailable: {missing}")
        ok = False
    if len(cols) < 2:
        ctx.errors.append(f"op {idx}: coalesce needs at least two columns")
        ok = False
    if len(unique_preserve_order(cols)) != len(cols):
        ctx.errors.append(f"op {idx}: coalesce contains duplicate columns")
        ok = False
    if not alias:
        ctx.errors.append(f"op {idx}: coalesce output alias is empty")
        ok = False
    elif is_reserved_output_name(alias):
        ctx.errors.append(f"op {idx}: coalesce output alias {alias!r} is reserved")
        ok = False
    present = [column for column in coalesce_sources(op) if column in ctx.state.column_types]
    output_type = ctx.state.column_types.get(present[0]) if present else None
    mismatched = [column for column in present if ctx.state.column_types.get(column) != output_type]
    if mismatched:
        ctx.errors.append(f"op {idx}: coalesce column type mismatch: {mismatched}")
        ok = False
    fallback = coalesce_fallback(op)
    if output_type is not None and "fallback" in op and fallback is not None:
        literal_error = scalar_filter_literal_error(output_type, fallback)
        if literal_error:
            ctx.errors.append(f"op {idx}: coalesce fallback literal {fallback!r} is incompatible with {output_type}")
            ok = False
    return ok


def _validate_case_when(ctx: ValidationContext, op: Any, idx: int) -> bool:
    alias = op_output_alias(op)
    predicate_column = condition_column(op)
    ok = True
    if predicate_column not in ctx.state.available:
        ctx.errors.append(f"op {idx}: case_when condition column unavailable: {predicate_column!r}")
        return False
    if not alias:
        ctx.errors.append(f"op {idx}: case_when output alias is empty")
        return False
    if is_reserved_output_name(alias):
        ctx.errors.append(f"op {idx}: case_when output alias {alias!r} is reserved")
        return False
    literal_error = filter_literal_error(
        ctx.state.column_types.get(predicate_column, "derived"),
        condition_cmp(op),
        condition_value(op),
    )
    if literal_error:
        ctx.errors.append(f"op {idx}: case_when {literal_error}")
        ok = False
    output_type = case_when_output_type(case_then_value(op), case_else_value(op))
    if output_type is None:
        ctx.errors.append(f"op {idx}: case_when branch literals are incompatible")
        ok = False
    return ok


def _validate_groupby(ctx: ValidationContext, op: Any, idx: int) -> bool:
    keys = groupby_keys(op)
    aggs = aggregate_specs(op)
    ok = True
    missing_keys = [key for key in keys if key not in ctx.state.available]
    if missing_keys:
        ctx.errors.append(f"op {idx}: groupby keys unavailable: {missing_keys}")
        ok = False
    if not keys:
        ctx.errors.append(f"op {idx}: groupby has no keys")
        ok = False
    if len(unique_preserve_order(keys)) != len(keys):
        ctx.errors.append(f"op {idx}: groupby contains duplicate keys")
        ok = False
    if not aggs:
        ctx.errors.append(f"op {idx}: groupby has no aggregations")
        ok = False
    aliases = [aggregate_alias(agg) for agg in aggs if aggregate_alias(agg)]
    if len(unique_preserve_order(aliases)) != len(aliases):
        ctx.errors.append(f"op {idx}: groupby contains duplicate aggregation aliases")
        ok = False
    key_set = set(keys)
    colliding_aliases = [alias for alias in aliases if alias in key_set]
    if colliding_aliases:
        ctx.errors.append(f"op {idx}: groupby aggregation aliases collide with keys: {colliding_aliases}")
        ok = False
    reserved_aliases = [alias for alias in aliases if is_reserved_output_name(alias)]
    if reserved_aliases:
        ctx.errors.append(f"op {idx}: groupby aggregation aliases use reserved names: {reserved_aliases}")
        ok = False
    for aggregate in aggs:
        column = aggregate_column(aggregate)
        func = aggregate_func(aggregate)
        if column not in ctx.state.available:
            ctx.errors.append(f"op {idx}: aggregation column {column!r} is unavailable")
            ok = False
        if not aggregate_accepts_type(
            ctx.state.column_types.get(column, "derived"),
            func,
            column in ctx.state.numeric,
        ):
            ctx.errors.append(f"op {idx}: aggregation column {column!r} is not numeric")
            ok = False
        if func not in {"sum", "mean", "min", "max", "count", "nunique", "any", "all"}:
            ctx.errors.append(f"op {idx}: unsupported aggregation {func!r}")
            ok = False
    return ok


def _validate_aggregate(ctx: ValidationContext, op: Any, idx: int) -> bool:
    aggs = aggregate_specs(op)
    ok = True
    if not aggs:
        ctx.errors.append(f"op {idx}: aggregate has no aggregations")
        ok = False
    aliases = [aggregate_alias(agg) for agg in aggs if aggregate_alias(agg)]
    if len(unique_preserve_order(aliases)) != len(aliases):
        ctx.errors.append(f"op {idx}: aggregate contains duplicate aggregation aliases")
        ok = False
    reserved_aliases = [alias for alias in aliases if is_reserved_output_name(alias)]
    if reserved_aliases:
        ctx.errors.append(f"op {idx}: aggregate aliases use reserved names: {reserved_aliases}")
        ok = False
    for aggregate in aggs:
        column = aggregate_column(aggregate)
        func = aggregate_func(aggregate)
        if column not in ctx.state.available:
            ctx.errors.append(f"op {idx}: aggregation column {column!r} is unavailable")
            ok = False
        if not aggregate_accepts_type(
            ctx.state.column_types.get(column, "derived"),
            func,
            column in ctx.state.numeric,
        ):
            ctx.errors.append(f"op {idx}: aggregation column {column!r} is not numeric")
            ok = False
        if func not in {"sum", "mean", "min", "max", "count", "nunique", "any", "all"}:
            ctx.errors.append(f"op {idx}: unsupported aggregation {func!r}")
            ok = False
    return ok


def apply_core_operation_state(state: ProgramState, op: Any) -> None:
    kind = op_kind(op)
    if kind == "join":
        return
    if kind in {"select", "distinct"}:
        state.replace_projection(op_columns(op))
        return
    if kind == "mutate":
        expr = getattr(op, "expression", None) or op.get("expr", {})
        state.upsert_column(op_column(op), expr_output_type(expr, state.column_types))
        return
    if kind == "running_sum":
        state.upsert_column(op_column(op), "float")
        return
    if kind == "fill_null":
        return
    if kind == "coalesce":
        alias = op_output_alias(op)
        sources = [column for column in coalesce_sources(op) if column in state.column_types]
        output_type = state.column_types.get(sources[0]) if sources else None
        state.upsert_column(alias, output_type)
        return
    if kind == "case_when":
        state.upsert_column(
            op_output_alias(op),
            case_when_output_type(case_then_value(op), case_else_value(op)),
        )
        return
    if kind == "groupby":
        next_columns = list(groupby_keys(op))
        next_types = {
            key: state.column_types[key]
            for key in next_columns
            if key in state.column_types
        }
        for aggregate in aggregate_specs(op):
            alias = aggregate_alias(aggregate)
            if not alias:
                continue
            next_columns.append(alias)
            next_types[alias] = aggregate_result_type(
                state.column_types.get(aggregate_column(aggregate)),
                aggregate_func(aggregate),
            )
        state.columns = unique_preserve_order(next_columns)
        state.column_types = next_types
        return
    if kind == "aggregate":
        next_columns: list[str] = []
        next_types: dict[str, str] = {}
        for aggregate in aggregate_specs(op):
            alias = aggregate_alias(aggregate)
            if not alias:
                continue
            next_columns.append(alias)
            next_types[alias] = aggregate_result_type(
                state.column_types.get(aggregate_column(aggregate)),
                aggregate_func(aggregate),
            )
        state.columns = unique_preserve_order(next_columns)
        state.column_types = next_types
        return
    if kind == "sortedness_check":
        state.collapse_to_alias(op_output_alias(op), "bool")


def normalized_row_number_filter_op(op: Any, available: set[str]) -> dict[str, Any] | None:
    partition_by = [column for column in unique_preserve_order(op_partition_columns(op)) if column in available]
    try:
        order_keys = normalized_order_by_keys(op)
    except ValueError:
        return None
    order_keys = dedupe_sort_keys([key for key in order_keys if key.column in available])
    if not order_keys:
        return None
    comparator = condition_cmp(op, "==")
    try:
        value = int(op_value(op) if op_value(op) is not None else 1)
    except (TypeError, ValueError):
        return None
    if comparator not in {"==", "<", "<="} or value <= 0:
        return None
    return {
        "op": "row_number_filter",
        "partition_by": partition_by,
        "order_by": [key.to_dict() for key in order_keys],
        "cmp": comparator,
        "value": value,
    }


def normalized_running_sum_op(op: Any, available: set[str]) -> dict[str, Any] | None:
    partition_by = [column for column in unique_preserve_order(op_partition_columns(op)) if column in available]
    try:
        order_keys = normalized_order_by_keys(op)
    except ValueError:
        return None
    order_keys = dedupe_sort_keys([key for key in order_keys if key.column in available])
    if not order_keys:
        return None
    repaired = {
        "op": "running_sum",
        "source": op_source(op),
        "column": op_column(op),
        "order_by": [key.to_dict() for key in order_keys],
        "input_dtype": op_input_dtype(op, "float64"),
    }
    if partition_by:
        repaired["partition_by"] = partition_by
    return repaired


def normalized_sort_op(op: Any, available: set[str]) -> dict[str, Any] | None:
    try:
        keys = normalize_sort_keys(op)
    except ValueError:
        return None
    keys = dedupe_sort_keys([key for key in keys if key.column in available])
    if not keys:
        return None
    existing = {key.column for key in keys}
    tail = [SortKey(column=column) for column in sorted(column for column in available if column not in existing)]
    full_keys = keys + tail
    if "keys" in op:
        return {"op": "sort", "keys": [key.to_dict() for key in full_keys]}
    repaired = dict(op.to_dict()) if hasattr(op, "to_dict") else dict(op)
    repaired["columns"] = [key.column for key in full_keys]
    return repaired


def dedupe_sort_keys(keys: list[SortKey]) -> list[SortKey]:
    seen: set[str] = set()
    output: list[SortKey] = []
    for key in keys:
        if key.column in seen:
            continue
        seen.add(key.column)
        output.append(key)
    return output
