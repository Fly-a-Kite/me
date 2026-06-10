from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping
import math
from typing import Any

from datadiff.dsl import AggregateSpec, Expression, SortKey, SortKeySpec, normalize_sort_keys

OperationLike = Mapping[str, Any]
AggregateLike = AggregateSpec | Mapping[str, Any]

DEFAULT_FALSE_PROBE_KINDS: frozenset[str] = frozenset(
    {
        "random_case_probe",
        "scalar_subquery_probe",
        "struct_distinct_probe",
        "bit_compare_probe",
        "round_even_probe",
        "float_literal_precision_probe",
        "tuple_anti_null_probe",
        "setop_all_duplicate_probe",
        "json_predicate_order_probe",
        "sparse_mask_probe",
        "index_bool_probe",
        "eval_inplace_alias_probe",
        "arrow_bool_groupby_reduction_probe",
        "dataset_isin_all_match_probe",
        "run_end_null_compute_probe",
        "large_string_partition_probe",
        "hash_pivot_wider_probe",
    }
)


def op_kind(op: OperationLike, default: str = "") -> str:
    value = op.get("op", default)
    if value in {None, ""}:
        value = op.get("kind", default)
    if value in {None, ""}:
        return default
    return str(value)


def operation_names(operations: Iterable[OperationLike], *, default: str = "") -> list[str]:
    return [op_kind(operation, default=default) for operation in operations]


def operation_count(operations: Iterable[OperationLike], kind: str) -> int:
    wanted = str(kind)
    return sum(1 for operation in operations if op_kind(operation) == wanted)


def has_operation(operations: Iterable[OperationLike], kind: str) -> bool:
    return any(op_kind(operation) == kind for operation in operations)


def has_any_operation(operations: Iterable[OperationLike], kinds: Collection[str]) -> bool:
    wanted = {str(kind) for kind in kinds}
    return any(op_kind(operation) in wanted for operation in operations)


ORDER_OBSERVING_OPS: frozenset[str] = frozenset(
    {
        "row_number_filter",
        "running_sum",
        "sort",
        "limit",
        "offset",
        "sortedness_check",
    }
)


def has_order_observer(program_or_operations: Any) -> bool:
    operations = getattr(program_or_operations, "operations", program_or_operations)
    if operations is None:
        return False
    return has_any_operation(operations, ORDER_OBSERVING_OPS)


def is_default_false_probe_kind(kind: str) -> bool:
    return kind in DEFAULT_FALSE_PROBE_KINDS


def has_post_topk_filter(operations: Iterable[OperationLike]) -> bool:
    ops = list(operations)
    for sort_idx, operation in enumerate(ops):
        if op_kind(operation) != "sort":
            continue
        for topk_idx in range(sort_idx + 1, len(ops)):
            if op_kind(ops[topk_idx]) not in {"limit", "offset"}:
                continue
            if any(op_kind(later) == "filter" for later in ops[topk_idx + 1 :]):
                return True
    return False


def expr_payload(op: OperationLike) -> Mapping[str, Any]:
    value = op.get("expr", {})
    return value if isinstance(value, Mapping) else {}


def expr_kind(op: OperationLike, default: str = "") -> str:
    expr = getattr(op, "expression", None)
    if isinstance(expr, Expression):
        value = expr.kind or default
    else:
        value = expr_payload(op).get("kind", default)
    if value in {None, ""}:
        return default
    return str(value)


def expr_kinds(operations: Iterable[OperationLike], *, op: str = "mutate") -> set[str]:
    kinds: set[str] = set()
    for operation in operations:
        if op_kind(operation) != op:
            continue
        kind = expr_kind(operation)
        if kind:
            kinds.add(kind)
    return kinds


def has_expr_kind(operations: Iterable[OperationLike], kind: str, *, op: str = "mutate") -> bool:
    return kind in expr_kinds(operations, op=op)


def has_any_expr_kind(operations: Iterable[OperationLike], kinds: Collection[str], *, op: str = "mutate") -> bool:
    return bool(expr_kinds(operations, op=op) & {str(kind) for kind in kinds})


def condition_payload(op: OperationLike) -> Mapping[str, Any]:
    value = op.get("condition", {})
    return value if isinstance(value, Mapping) else {}


def condition_cmp(op: OperationLike, default: str = "") -> str:
    condition = op.get("condition")
    value = getattr(condition, "comparator", None)
    if value in {None, ""}:
        value = condition_payload(op).get("cmp", default)
    if value in {None, ""}:
        value = op.get("cmp", default)
    if value in {None, ""}:
        return default
    return str(value)


def condition_column(op: OperationLike, default: str = "") -> str:
    condition = op.get("condition")
    value = getattr(condition, "column", None)
    if value in {None, ""}:
        value = condition_payload(op).get("column", default)
    if value in {None, ""}:
        value = op.get("column", default)
    if value in {None, ""}:
        return default
    return str(value)


def condition_value(op: OperationLike) -> Any:
    condition = op.get("condition")
    if condition is not None and hasattr(condition, "value"):
        return getattr(condition, "value")
    value = condition_payload(op).get("value")
    if value is not None or "value" in condition_payload(op):
        return value
    return op.get("value")


def op_table(op: OperationLike, default: str = "") -> str:
    value = getattr(op, "table", None)
    if value in {None, ""}:
        value = op.get("table", default)
    if value in {None, ""}:
        return default
    return str(value)


def op_output_alias(op: OperationLike, default: str = "") -> str:
    value = getattr(op, "output_alias", None)
    if value in {None, ""}:
        value = op.get("as", default)
    if value in {None, ""}:
        return default
    return str(value)


def op_column(op: OperationLike, default: str = "") -> str:
    value = getattr(op, "column", None)
    if value in {None, ""}:
        value = op.get("column", default)
    if value in {None, ""}:
        return default
    return str(value)


def op_source(op: OperationLike, default: str = "") -> str:
    value = getattr(op, "source", None)
    if value in {None, ""}:
        value = op.get("source", default)
    if value in {None, ""}:
        return default
    return str(value)


def op_comparator(op: OperationLike, default: str = "") -> str:
    value = getattr(op, "comparator", None)
    if value in {None, ""}:
        value = op.get("cmp", default)
    if value in {None, ""}:
        return default
    return str(value)


def op_columns(op: OperationLike) -> list[str]:
    value = getattr(op, "columns", None)
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    raw = op.get("columns", [])
    if isinstance(raw, (list, tuple)):
        return [str(item) for item in raw if str(item)]
    column = str(raw or "")
    return [column] if column else []


def op_value(op: OperationLike) -> Any:
    if hasattr(op, "value"):
        return getattr(op, "value")
    return op.get("value")


def op_right_columns(op: OperationLike) -> list[str]:
    value = getattr(op, "right_columns", None)
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    raw = op.get("right_columns", [])
    if isinstance(raw, (list, tuple)):
        return [str(item) for item in raw if str(item)]
    column = str(raw or "")
    return [column] if column else []


def op_n(op: OperationLike, default: int = 0) -> int:
    value = getattr(op, "n", None)
    if value in {None, ""}:
        value = op.get("n", default)
    if value in {None, ""}:
        return default
    return int(value)


def op_rows(op: OperationLike, default: int = 0) -> int:
    value = getattr(op, "rows", None)
    if value in {None, ""}:
        value = op.get("rows", default)
    if value in {None, ""}:
        return default
    return int(value)


def op_branches(op: OperationLike, default: int = 0) -> int:
    value = getattr(op, "branches", None)
    if value in {None, ""}:
        value = op.get("branches", default)
    if value in {None, ""}:
        return default
    return int(value)


def op_literal(op: OperationLike, default: str = "") -> str:
    value = getattr(op, "literal", None)
    if value in {None, ""}:
        value = op.get("literal", default)
    if value in {None, ""}:
        return default
    return str(value)


def op_values(op: OperationLike) -> list[Any]:
    raw = getattr(op, "probe_values", None)
    if raw is None:
        raw = op.get("values", [])
    if isinstance(raw, (list, tuple)):
        return list(raw)
    return [raw] if raw not in {None, ""} else []


def op_quantiles(op: OperationLike) -> list[Any]:
    raw = getattr(op, "quantiles", None)
    if raw is None:
        raw = op.get("quantiles", [])
    if isinstance(raw, (list, tuple)):
        return list(raw)
    return [raw] if raw not in {None, ""} else []


def op_input_dtype(op: OperationLike, default: str = "") -> str:
    value = getattr(op, "input_dtype", None)
    if value in {None, ""}:
        value = op.get("input_dtype", default)
    if value in {None, ""}:
        return default
    return str(value)


def op_partition_columns(op: OperationLike) -> list[str]:
    value = getattr(op, "partition_columns", None)
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    raw = op.get("partition_by", [])
    if isinstance(raw, (list, tuple)):
        return [str(item) for item in raw if str(item)]
    column = str(raw or "")
    return [column] if column else []


def groupby_keys(op: OperationLike) -> list[str]:
    value = getattr(op, "group_keys", None)
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    raw = op.get("keys", [])
    return [str(item) for item in raw if str(item)]


def join_how(op: OperationLike, default: str = "") -> str:
    value = getattr(op, "how", None)
    if value in {None, ""}:
        value = op.get("how", default)
    if value in {None, ""}:
        return default
    return str(value)


def op_ascending(op: OperationLike, default: bool = True) -> bool:
    value = getattr(op, "ascending", None)
    if value is None:
        value = op.get("ascending", default)
    if value is None:
        return default
    return value if isinstance(value, bool) else bool(value)


def op_nulls(op: OperationLike, default: str = "last") -> str:
    value = getattr(op, "nulls", None)
    if value in {None, ""}:
        value = op.get("nulls", default)
    if value in {None, ""}:
        return default
    return str(value)


def coalesce_sources(op: OperationLike) -> list[str]:
    value = getattr(op, "sources", None)
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    return op_columns(op)


def coalesce_has_fallback(op: OperationLike) -> bool:
    return "fallback" in op


def coalesce_fallback(op: OperationLike) -> Any:
    if hasattr(op, "fallback"):
        return getattr(op, "fallback")
    return op.get("fallback")


def case_then_value(op: OperationLike) -> Any:
    if hasattr(op, "then_value"):
        return getattr(op, "then_value")
    return op.get("then")


def case_else_value(op: OperationLike) -> Any:
    if hasattr(op, "else_value"):
        return getattr(op, "else_value")
    return op.get("else")


def is_left_join(op: OperationLike) -> bool:
    return op_kind(op) == "join" and str(getattr(op, "how", op.get("how", ""))) == "left"


def is_multi_key_join(op: OperationLike) -> bool:
    if op_kind(op) not in {"join", "semi_join", "anti_join"}:
        return False
    left_keys = getattr(op, "left_keys", None)
    right_keys = getattr(op, "right_keys", None)
    if isinstance(left_keys, list) or isinstance(right_keys, list):
        return len(left_keys or []) > 1 or len(right_keys or []) > 1
    left_on = op.get("left_on")
    right_on = op.get("right_on")
    return isinstance(left_on, (list, tuple)) or isinstance(right_on, (list, tuple))


def is_multi_key_groupby(op: OperationLike) -> bool:
    if op_kind(op) != "groupby":
        return False
    keys = getattr(op, "group_keys", None)
    if isinstance(keys, list):
        return len(keys) > 1
    return len(op.get("keys", []) or []) > 1


def is_string_input_cast(op: OperationLike) -> bool:
    return op_kind(op) == "mutate" and expr_kind(op) == "cast" and expr_payload(op).get("input_domain") in {
        "numeric_string",
        "integer_string",
    }


def groupby_agg_aliases(op: OperationLike) -> set[str]:
    aggregate_aliases = getattr(op, "aggregate_aliases", None)
    if isinstance(aggregate_aliases, list):
        return {alias for alias in aggregate_aliases if alias}
    aliases: set[str] = set()
    for aggregate in aggregate_specs(op):
        alias = aggregate_alias(aggregate)
        if alias:
            aliases.add(alias)
    return aliases


def aggregate_specs(op: OperationLike) -> list[AggregateLike]:
    value = getattr(op, "aggregates", None)
    if isinstance(value, list):
        return value
    aggregates = op.get("aggs", []) or []
    return [aggregate for aggregate in aggregates if isinstance(aggregate, Mapping)]


def aggregate_func(aggregate: AggregateLike, default: str = "") -> str:
    if isinstance(aggregate, AggregateSpec):
        value = aggregate.func
    else:
        value = aggregate.get("func", default)
    if value in {None, ""}:
        return default
    return str(value)


def aggregate_column(aggregate: AggregateLike, default: str = "") -> str:
    if isinstance(aggregate, AggregateSpec):
        value = aggregate.column
    else:
        value = aggregate.get("column", default)
    if value in {None, ""}:
        return default
    return str(value)


def aggregate_alias(aggregate: AggregateLike, default: str = "") -> str:
    if isinstance(aggregate, AggregateSpec):
        value = aggregate.alias
    else:
        value = aggregate.get("as", default)
    if value in {None, ""}:
        return default
    return str(value)


def aggregate_functions(op: OperationLike) -> list[str]:
    funcs: list[str] = []
    for aggregate in aggregate_specs(op):
        func = aggregate_func(aggregate)
        if func:
            funcs.append(func)
    return funcs


def expr_operator(op: OperationLike, default: str = "") -> str:
    expr = getattr(op, "expression", None)
    value = getattr(expr, "operator", None)
    if value in {None, ""}:
        value = expr_payload(op).get("op", default)
    if value in {None, ""}:
        return default
    return str(value)


def expr_target_type(op: OperationLike, default: str = "") -> str:
    expr = getattr(op, "expression", None)
    value = getattr(expr, "target_type", None)
    if value in {None, ""}:
        value = expr_payload(op).get("to", default)
    if value in {None, ""}:
        return default
    return str(value)


def expr_source(op: OperationLike, default: str = "") -> str:
    expr = getattr(op, "expression", None)
    value = getattr(expr, "source", None)
    if value in {None, ""}:
        value = expr_payload(op).get("source", default)
    if value in {None, ""}:
        return default
    return str(value)


def expr_value(op: OperationLike) -> Any:
    expr = getattr(op, "expression", None)
    if expr is not None and hasattr(expr, "value"):
        return getattr(expr, "value")
    return expr_payload(op).get("value")


def expr_other(op: OperationLike, default: str = "") -> str:
    expr = getattr(op, "expression", None)
    value = getattr(expr, "other", None)
    if value in {None, ""}:
        value = expr_payload(op).get("other", default)
    if value in {None, ""}:
        return default
    return str(value)


def expr_separator(op: OperationLike, default: str = "") -> str:
    expr = getattr(op, "expression", None)
    value = getattr(expr, "separator", None)
    if value in {None, ""}:
        value = expr_payload(op).get("sep", default)
    if value in {None, ""}:
        return default
    return str(value)


def expr_needle(op: OperationLike) -> Any:
    expr = getattr(op, "expression", None)
    if expr is not None and hasattr(expr, "needle"):
        return getattr(expr, "needle")
    return expr_payload(op).get("needle")


def expr_part(op: OperationLike, default: str = "") -> str:
    expr = getattr(op, "expression", None)
    value = getattr(expr, "part", None)
    if value in {None, ""}:
        value = expr_payload(op).get("part", default)
    if value in {None, ""}:
        return default
    return str(value)


def expr_input_domain(op: OperationLike, default: str = "") -> str:
    expr = getattr(op, "expression", None)
    value = getattr(expr, "input_domain", None)
    if value in {None, ""}:
        value = expr_payload(op).get("input_domain", default)
    if value in {None, ""}:
        return default
    return str(value)


def expr_old(op: OperationLike) -> Any:
    expr = getattr(op, "expression", None)
    if expr is not None and hasattr(expr, "old"):
        return getattr(expr, "old")
    return expr_payload(op).get("old")


def expr_new(op: OperationLike) -> Any:
    expr = getattr(op, "expression", None)
    if expr is not None and hasattr(expr, "new"):
        return getattr(expr, "new")
    return expr_payload(op).get("new")


def expr_start(op: OperationLike) -> Any:
    expr = getattr(op, "expression", None)
    if expr is not None and hasattr(expr, "start"):
        return getattr(expr, "start")
    return expr_payload(op).get("start")


def expr_length(op: OperationLike) -> Any:
    expr = getattr(op, "expression", None)
    if expr is not None and hasattr(expr, "length"):
        return getattr(expr, "length")
    return expr_payload(op).get("length")


def expr_index(op: OperationLike) -> Any:
    expr = getattr(op, "expression", None)
    if expr is not None and hasattr(expr, "index"):
        return getattr(expr, "index")
    return expr_payload(op).get("index")


def expr_lower(op: OperationLike) -> Any:
    expr = getattr(op, "expression", None)
    if expr is not None and hasattr(expr, "lower"):
        return getattr(expr, "lower")
    return expr_payload(op).get("lower")


def expr_upper(op: OperationLike) -> Any:
    expr = getattr(op, "expression", None)
    if expr is not None and hasattr(expr, "upper"):
        return getattr(expr, "upper")
    return expr_payload(op).get("upper")


def expr_numerator(op: OperationLike, default: str = "") -> str:
    expr = getattr(op, "expression", None)
    value = getattr(expr, "numerator", None)
    if value in {None, ""}:
        value = expr_payload(op).get("numerator", default)
    if value in {None, ""}:
        return default
    return str(value)


def join_left_keys(op: OperationLike) -> list[str]:
    value = getattr(op, "left_keys", None)
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    left_on = op.get("left_on")
    if isinstance(left_on, (list, tuple)):
        return [str(item) for item in left_on if str(item)]
    column = str(left_on or "")
    return [column] if column else []


def join_right_keys(op: OperationLike) -> list[str]:
    value = getattr(op, "right_keys", None)
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    right_on = op.get("right_on")
    if isinstance(right_on, (list, tuple)):
        return [str(item) for item in right_on if str(item)]
    column = str(right_on or "")
    return [column] if column else []


def sort_keys(op: OperationLike) -> list[SortKeySpec | Any]:
    value = getattr(op, "get", lambda _key, _default=None: _default)("keys", [])
    if isinstance(value, list):
        return value
    return []


def op_sort_keys(op: OperationLike) -> list[Any]:
    typed_keys = getattr(op, "order_keys", None)
    if isinstance(typed_keys, list):
        return typed_keys
    raw = op.get("order_by")
    if isinstance(raw, list):
        return raw
    return sort_keys(op)


def normalized_order_by_keys(op: OperationLike) -> list[SortKey]:
    typed_keys = getattr(op, "order_keys", None)
    if isinstance(typed_keys, list):
        return [
            SortKey(
                column=str(getattr(key, "column", "")),
                ascending=bool(getattr(key, "ascending", True)),
                nulls="first" if str(getattr(key, "nulls", "last")) == "first" else "last",
            )
            for key in typed_keys
        ]
    return normalize_sort_keys({"keys": op.get("order_by", [])})


def sort_has_nulls_first(op: OperationLike) -> bool:
    try:
        return any(key.nulls == "first" for key in normalize_sort_keys(op))
    except ValueError:
        return False


def sort_key_columns(op: OperationLike) -> set[str]:
    try:
        return {key.column for key in normalize_sort_keys(op)}
    except ValueError:
        return set()


def sort_null_placement_mismatch(
    sort_op: OperationLike | None,
    column: str,
    check_nulls: str,
) -> bool:
    if sort_op is None:
        return False
    try:
        keys = normalize_sort_keys(sort_op)
    except ValueError:
        return False
    return any(key.column == column and key.nulls != check_nulls for key in keys)


def aggregate_feature_type(source_type: str, func: str) -> str:
    if func in {"count", "nunique"}:
        return "int"
    if func in {"any", "all"}:
        return "bool"
    if func in {"min", "max"} and source_type in {"int", "float", "str", "bool"}:
        return source_type
    return "float"


def has_fractional_float_literal(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value) and not value.is_integer()
    if isinstance(value, (list, tuple, set, frozenset)):
        return any(has_fractional_float_literal(item) for item in value)
    return False
