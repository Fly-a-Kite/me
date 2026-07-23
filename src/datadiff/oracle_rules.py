from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

_STRING_EXPR_ROOT_KINDS = frozenset(
    {
        "string_length",
        "string_lower",
        "string_upper",
        "string_strip",
        "string_null_if_empty",
        "string_replace",
        "string_slice",
        "string_split_part",
        "string_basename",
        "string_concat",
        "string_contains",
        "string_starts_with",
        "string_ends_with",
    }
)


SignalValue = bool | Callable[[], bool]


@dataclass(slots=True)
class RootCauseContext:
    kind: str
    op_set: frozenset[str]
    expr_kind_set: frozenset[str]
    probe_root: str | None = None
    signals: Mapping[str, SignalValue] = field(default_factory=dict)
    _signal_cache: dict[str, bool] = field(default_factory=dict, init=False, repr=False)

    def signal(self, name: str) -> bool:
        if name in self._signal_cache:
            return self._signal_cache[name]
        value = self.signals.get(name, False)
        resolved = bool(value() if callable(value) else value)
        self._signal_cache[name] = resolved
        return resolved


@dataclass(frozen=True, slots=True)
class RootCauseRule:
    name: str
    root_cause: str
    predicate: Callable[[RootCauseContext], bool]


ROOT_CAUSE_RULES: tuple[RootCauseRule, ...] = (
    RootCauseRule("exception_mismatch", "exception_taxonomy", lambda ctx: ctx.kind == "exception_mismatch"),
    RootCauseRule("probe_root", "", lambda ctx: ctx.probe_root is not None),
    RootCauseRule("path_projection", "path_projection_keyed_pick", lambda ctx: ctx.signal("path_projection_keyed_pick")),
    RootCauseRule("running_sum", "running_sum_precision", lambda ctx: ctx.signal("running_sum")),
    RootCauseRule("special_float", "nan_inf_semantics", lambda ctx: ctx.signal("contains_special_float")),
    RootCauseRule("modulo", "arithmetic_expression", lambda ctx: ctx.signal("uses_modulo")),
    RootCauseRule("reverse_division", "reverse_division_operand_order", lambda ctx: ctx.signal("reverse_division_columns")),
    RootCauseRule(
        "tuple_absence_nullable",
        "tuple_absence_null_filter",
        lambda ctx: ctx.signal("tuple_absence_nullable_row_value"),
    ),
    RootCauseRule("grouped_topk_null_sort", "grouped_topk_null_sort_key", lambda ctx: ctx.signal("grouped_topk_null_sort_key")),
    RootCauseRule("distinct_null_topk", "distinct_null_topk", lambda ctx: ctx.signal("distinct_null_topk")),
    RootCauseRule("float_group_key", "float_group_key_instability", lambda ctx: ctx.signal("float_group_key_instability")),
    RootCauseRule("negative_zero", "negative_zero_comparison", lambda ctx: ctx.signal("negative_zero_comparison")),
    RootCauseRule("tuple_absence", "tuple_absence_null_filter", lambda ctx: ctx.signal("tuple_absence_filter")),
    RootCauseRule("unicode_case", "unicode_case_mapping", lambda ctx: ctx.signal("unicode_case_mapping")),
    RootCauseRule("outer_join_truth", "outer_join_truth_filter", lambda ctx: ctx.signal("outer_join_truth_filter")),
    RootCauseRule("post_topk_filter", "topk_filter_pushdown", lambda ctx: ctx.signal("post_topk_filter")),
    RootCauseRule("joined_order_offset", "joined_order_offset_projection", lambda ctx: ctx.signal("joined_order_offset_projection")),
    RootCauseRule("ordered_topk", "ordered_topk_projection", lambda ctx: ctx.signal("ordered_topk_projection")),
    RootCauseRule("float_precision_boundary", "arithmetic_expression", lambda ctx: ctx.signal("float_precision_boundary")),
    RootCauseRule("case_when", "conditional_expression", lambda ctx: "case_when" in ctx.op_set),
    RootCauseRule("union_all", "union_all_row_append", lambda ctx: "union_all" in ctx.op_set),
    RootCauseRule("drop_nulls", "drop_nulls_null_filter", lambda ctx: "drop_nulls" in ctx.op_set),
    RootCauseRule("semi_join", "semi_join_membership", lambda ctx: "semi_join" in ctx.op_set),
    RootCauseRule("anti_join", "anti_join_exclusion", lambda ctx: "anti_join" in ctx.op_set),
    RootCauseRule("coalesce", "coalesce_null_semantics", lambda ctx: "coalesce" in ctx.op_set),
    RootCauseRule(
        "pyarrow_sliced_bool_groupby",
        "pyarrow_sliced_bool_groupby_any_all",
        lambda ctx: ctx.signal("pyarrow_sliced_bool_groupby"),
    ),
    RootCauseRule("groupby", "groupby_aggregation", lambda ctx: bool(ctx.op_set & {"groupby", "aggregate"})),
    RootCauseRule("join", "join_semantics", lambda ctx: "join" in ctx.op_set),
    RootCauseRule("fill_null", "fill_null_null_semantics", lambda ctx: "fill_null" in ctx.op_set),
    RootCauseRule("distinct", "distinct_duplicate_elimination", lambda ctx: "distinct" in ctx.op_set),
    RootCauseRule("filter", "filter_predicate", lambda ctx: "filter" in ctx.op_set),
    RootCauseRule("mutate_string", "string_expression", lambda ctx: "mutate" in ctx.op_set and bool(ctx.expr_kind_set & _STRING_EXPR_ROOT_KINDS)),
    RootCauseRule("mutate_date", "datetime_expression", lambda ctx: "mutate" in ctx.op_set and "date_part" in ctx.expr_kind_set),
    RootCauseRule("mutate_bool", "nullable_boolean_expression", lambda ctx: "mutate" in ctx.op_set and "bool_not" in ctx.expr_kind_set),
    RootCauseRule("mutate_cast", "type_cast", lambda ctx: "mutate" in ctx.op_set and "cast" in ctx.expr_kind_set),
    RootCauseRule("mutate_arithmetic", "arithmetic_expression", lambda ctx: "mutate" in ctx.op_set),
    RootCauseRule("ordering", "ordering_or_limit", lambda ctx: bool(ctx.op_set & {"sort", "limit", "offset"})),
    RootCauseRule("schema_projection", "schema_projection", lambda ctx: ctx.signal("ok_result_columns_differ")),
    RootCauseRule("nulls", "null_semantics", lambda ctx: ctx.signal("contains_null")),
)


def classify_root_cause_from_context(context: RootCauseContext) -> str:
    for rule in ROOT_CAUSE_RULES:
        if not rule.predicate(context):
            continue
        if rule.name == "probe_root":
            return str(context.probe_root or "")
        return rule.root_cause
    return "unknown"
