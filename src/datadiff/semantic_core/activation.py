"""Deterministic semantic-activation witnesses for goal-first generation.

Syntactic goal reach only proves that required types and operation families are
present.  This module records a stricter, data-aware witness without changing
the generated case or consulting backend outcomes.  Evaluators are deliberately
small and fail closed: unsupported goals are ``not_evaluated`` rather than being
reported as inactive.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Mapping

from datadiff.dsl import Case, normalize_sort_keys
from datadiff.operation_semantics import op_kind
from datadiff.semantic_family_universe_v3 import expansion_v3_family_definitions


SEMANTIC_ACTIVATION_SCHEMA_VERSION = "semantic-activation-v1"
ActivationStatus = Literal["activated", "not_activated", "not_evaluated"]


@dataclass(frozen=True, slots=True)
class SemanticActivation:
    goal_id: str
    syntactic_reached: bool
    evaluation_status: ActivationStatus
    semantically_activated: bool | None
    evaluator_id: str
    required_tokens: tuple[str, ...] = ()
    activation_tokens: tuple[str, ...] = ()
    missing_tokens: tuple[str, ...] = ()
    witness_summary: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    schema_version: str = SEMANTIC_ACTIVATION_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "goal_id": self.goal_id,
            "syntactic_reached": self.syntactic_reached,
            "evaluation_status": self.evaluation_status,
            "semantically_activated": self.semantically_activated,
            "evaluator_id": self.evaluator_id,
            "required_tokens": list(self.required_tokens),
            "activation_tokens": list(self.activation_tokens),
            "missing_tokens": list(self.missing_tokens),
            "witness_summary": dict(self.witness_summary),
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "SemanticActivation":
        """Load current or absent/legacy evidence without treating unknown as false."""

        payload = data if isinstance(data, Mapping) else {}
        raw_activated = payload.get("semantically_activated")
        activated = raw_activated if isinstance(raw_activated, bool) else None
        raw_status = str(payload.get("evaluation_status", "") or "")
        if raw_status not in {"activated", "not_activated", "not_evaluated"}:
            raw_status = (
                "activated"
                if activated is True
                else "not_activated"
                if activated is False
                else "not_evaluated"
            )
        if raw_status == "not_evaluated":
            activated = None
        elif activated is None:
            activated = raw_status == "activated"
        return cls(
            goal_id=str(payload.get("goal_id", "") or ""),
            syntactic_reached=bool(payload.get("syntactic_reached", False)),
            evaluation_status=raw_status,
            semantically_activated=activated,
            evaluator_id=str(payload.get("evaluator_id", "none") or "none"),
            required_tokens=_string_tuple(payload.get("required_tokens")),
            activation_tokens=_string_tuple(payload.get("activation_tokens")),
            missing_tokens=_string_tuple(payload.get("missing_tokens")),
            witness_summary=dict(payload.get("witness_summary", {}) or {}),
            reason=str(payload.get("reason", "") or ""),
            schema_version=str(
                payload.get("schema_version", SEMANTIC_ACTIVATION_SCHEMA_VERSION)
                or SEMANTIC_ACTIVATION_SCHEMA_VERSION
            ),
        )


@dataclass(frozen=True, slots=True)
class _WitnessEvaluation:
    evaluator_id: str
    required_tokens: tuple[str, ...]
    observed_tokens: tuple[str, ...]
    witness_summary: dict[str, Any]


ActivationEvaluator = Callable[[Case], _WitnessEvaluation]


def evaluate_semantic_activation(
    case: Case,
    *,
    goal_id: str,
    syntactic_reached: bool,
) -> SemanticActivation:
    """Evaluate a goal using static program/data witnesses only.

    The result is observational evidence.  It never mutates ``case`` and never
    uses an execution result, learned model, network service, or random choice.
    """

    resolved_goal_id = str(goal_id or "")
    evaluator = _EVALUATORS.get(resolved_goal_id)
    if evaluator is None:
        return SemanticActivation(
            goal_id=resolved_goal_id,
            syntactic_reached=bool(syntactic_reached),
            evaluation_status="not_evaluated",
            semantically_activated=None,
            evaluator_id="none",
            witness_summary={"registered_evaluator": False},
            reason="no_registered_evaluator",
        )

    evaluation = evaluator(case)
    observed = tuple(dict.fromkeys(evaluation.observed_tokens))
    missing = tuple(
        token for token in evaluation.required_tokens if token not in observed
    )
    activated = bool(syntactic_reached) and not missing
    if activated:
        reason = "all_required_witnesses_observed"
    elif not syntactic_reached:
        reason = "syntactic_goal_not_reached"
    else:
        reason = "missing_required_witnesses:" + ",".join(missing)
    return SemanticActivation(
        goal_id=resolved_goal_id,
        syntactic_reached=bool(syntactic_reached),
        evaluation_status="activated" if activated else "not_activated",
        semantically_activated=activated,
        evaluator_id=evaluation.evaluator_id,
        required_tokens=evaluation.required_tokens,
        activation_tokens=observed,
        missing_tokens=missing,
        witness_summary=evaluation.witness_summary,
        reason=reason,
    )


def _evaluate_order_offset_aggregate(case: Case) -> _WitnessEvaluation:
    required = (
        "ordered_cut_pipeline",
        "positive_offset",
        "active_cardinality_cut",
        "nonempty_cut_survivor",
        "tie_competition",
        "deterministic_tie_breaker",
        "nullable_payload",
        "grouped_observation",
    )
    operations = list(case.program.operations)
    group_index = _first_index(operations, {"groupby", "aggregate"})
    sort_index = _first_index(operations, {"sort"}, before=group_index)
    cut_rows = _cut_witness(
        operations,
        input_rows=_primary_row_count(case),
        start=sort_index,
        end=group_index,
    )
    cut_indices = cut_rows["operation_indices"]
    ordered_pipeline = (
        sort_index is not None
        and group_index is not None
        and bool(cut_indices)
        and sort_index < min(cut_indices) < group_index
    )
    positive_offset = any(
        op_kind(operations[index]) == "offset"
        and _nonnegative_int(operations[index].get("n")) > 0
        for index in cut_indices
    )

    sort_keys = (
        normalize_sort_keys(operations[sort_index]) if sort_index is not None else []
    )
    leading_column = sort_keys[0].column if sort_keys else ""
    leading_values = _column_values_before(
        case,
        leading_column,
        operation_index=sort_index,
    )
    tie_multiplicities = sorted(
        (count for count in Counter(_stable_value_key(value) for value in leading_values).values() if count > 1),
        reverse=True,
    )
    deterministic_tie_breaker = len(sort_keys) >= 2

    group_operation = operations[group_index] if group_index is not None else None
    observed_columns = {key.column for key in sort_keys}
    aggregate_count = 0
    if group_operation is not None:
        observed_columns.update(_column_list(group_operation.get("keys")))
        aggregates = list(group_operation.get("aggs", []) or [])
        aggregate_count = len(aggregates)
        observed_columns.update(
            str(aggregate.get("column", "") or "")
            for aggregate in aggregates
            if isinstance(aggregate, Mapping)
        )
    null_counts = _null_counts(case, observed_columns)

    observed: list[str] = []
    _observe(observed, "ordered_cut_pipeline", ordered_pipeline)
    _observe(observed, "positive_offset", positive_offset)
    _observe(
        observed,
        "active_cardinality_cut",
        int(cut_rows["removed_rows"]) > 0,
    )
    _observe(
        observed,
        "nonempty_cut_survivor",
        int(cut_rows["rows_after_cut"]) > 0,
    )
    _observe(observed, "tie_competition", bool(tie_multiplicities))
    _observe(observed, "deterministic_tie_breaker", deterministic_tie_breaker)
    _observe(observed, "nullable_payload", sum(null_counts.values()) > 0)
    _observe(observed, "grouped_observation", aggregate_count > 0)
    return _WitnessEvaluation(
        evaluator_id="order-offset-aggregate-static-v1",
        required_tokens=required,
        observed_tokens=tuple(observed),
        witness_summary={
            "sort_index": sort_index,
            "cut_operation_indices": cut_indices,
            "group_index": group_index,
            "input_row_count": _primary_row_count(case),
            "rows_after_cut": cut_rows["rows_after_cut"],
            "cut_removed_rows": cut_rows["removed_rows"],
            "positive_offset": positive_offset,
            "leading_sort_column": leading_column,
            "tie_multiplicities": tie_multiplicities,
            "sort_key_count": len(sort_keys),
            "null_counts": null_counts,
            "aggregate_count": aggregate_count,
        },
    )


def _evaluate_nullable_membership_join(case: Case) -> _WitnessEvaluation:
    required = (
        "multi_key_membership",
        "null_guards_before_membership",
        "null_key_rows",
        "matched_keys",
        "unmatched_keys",
        "duplicate_key_pressure",
        "downstream_grouped_observation",
    )
    operations = list(case.program.operations)
    join_index = _first_index(operations, {"semi_join", "anti_join"})
    join = operations[join_index] if join_index is not None else None
    left_keys = _column_list(join.get("left_on")) if join is not None else []
    right_keys = _column_list(join.get("right_on")) if join is not None else []
    right_table_name = str(join.get("table", "") or "") if join is not None else ""
    left_table = case.tables[0] if case.tables else None
    right_table = next(
        (table for table in case.tables if table.name == right_table_name),
        None,
    )
    left_rows = list(left_table.rows) if left_table is not None else []
    right_rows = list(right_table.rows) if right_table is not None else []

    guarded_columns = {
        str(operation.get("column", "") or "")
        for operation in operations[: join_index or 0]
        if op_kind(operation) == "filter"
        and str(operation.get("cmp", "") or "") == "is_not_null"
    }
    null_key_rows = sum(
        any(row.get(column) is None for column in left_keys) for row in left_rows
    )
    eligible_left_keys = [
        _stable_row_key(row, left_keys)
        for row in left_rows
        if left_keys and all(row.get(column) is not None for column in left_keys)
    ]
    eligible_right_keys = [
        _stable_row_key(row, right_keys)
        for row in right_rows
        if right_keys and all(row.get(column) is not None for column in right_keys)
    ]
    right_key_set = set(eligible_right_keys)
    matched_count = sum(key in right_key_set for key in eligible_left_keys)
    unmatched_count = sum(key not in right_key_set for key in eligible_left_keys)
    exact_duplicate_count = _duplicate_count(eligible_left_keys) + _duplicate_count(
        eligible_right_keys
    )
    component_collision_count = _component_collision_count(
        eligible_left_keys
    ) + _component_collision_count(eligible_right_keys)
    downstream_group_index = _first_index(
        operations,
        {"groupby", "aggregate"},
        after=join_index,
    )

    observed: list[str] = []
    _observe(
        observed,
        "multi_key_membership",
        join is not None and len(left_keys) >= 2 and len(left_keys) == len(right_keys),
    )
    _observe(
        observed,
        "null_guards_before_membership",
        bool(left_keys) and set(left_keys) <= guarded_columns,
    )
    _observe(observed, "null_key_rows", null_key_rows > 0)
    _observe(observed, "matched_keys", matched_count > 0)
    _observe(observed, "unmatched_keys", unmatched_count > 0)
    _observe(
        observed,
        "duplicate_key_pressure",
        exact_duplicate_count > 0 or component_collision_count > 0,
    )
    _observe(
        observed,
        "downstream_grouped_observation",
        downstream_group_index is not None,
    )
    return _WitnessEvaluation(
        evaluator_id="nullable-membership-join-static-v1",
        required_tokens=required,
        observed_tokens=tuple(observed),
        witness_summary={
            "join_index": join_index,
            "join_kind": op_kind(join) if join is not None else "",
            "left_keys": left_keys,
            "right_keys": right_keys,
            "guarded_columns": sorted(guarded_columns),
            "left_row_count": len(left_rows),
            "right_row_count": len(right_rows),
            "null_key_rows": null_key_rows,
            "eligible_left_key_count": len(eligible_left_keys),
            "matched_key_count": matched_count,
            "unmatched_key_count": unmatched_count,
            "exact_duplicate_count": exact_duplicate_count,
            "component_collision_count": component_collision_count,
            "downstream_group_index": downstream_group_index,
        },
    )


def _evaluate_union_distinct_window(case: Case) -> _WitnessEvaluation:
    required = (
        "union_distinct_window_pipeline",
        "duplicate_bag_to_set_transition",
        "anti_join_match_and_survivor",
        "nullable_window_input",
        "multirow_window_partition",
        "deterministic_window_order",
    )
    operations = list(case.program.operations)
    union_index = _first_index(operations, {"union_all"})
    distinct_index = _first_index(operations, {"distinct"}, after=union_index)
    anti_index = _first_index(operations, {"anti_join"}, after=distinct_index)
    window_index = _first_index(operations, {"running_sum"}, after=anti_index)
    union = operations[union_index] if union_index is not None else None
    distinct = operations[distinct_index] if distinct_index is not None else None
    anti = operations[anti_index] if anti_index is not None else None
    window = operations[window_index] if window_index is not None else None

    base = case.tables[0] if case.tables else None
    append = _table_by_name(case, str(union.get("table", "") or "")) if union else None
    combined_rows = [
        *([] if base is None else base.rows),
        *([] if append is None else append.rows),
    ]
    distinct_columns = _column_list(distinct.get("columns")) if distinct else []
    distinct_keys = [
        _stable_row_key(row, distinct_columns)
        for row in combined_rows
        if distinct_columns
    ]
    duplicate_count = _duplicate_count(distinct_keys)

    candidate_rows = list(combined_rows)
    if distinct_columns:
        seen: set[tuple[Any, ...]] = set()
        deduplicated: list[Mapping[str, Any]] = []
        for row in candidate_rows:
            key = _stable_row_key(row, distinct_columns)
            if key in seen:
                continue
            seen.add(key)
            deduplicated.append(row)
        candidate_rows = deduplicated
    if distinct_index is not None and anti_index is not None:
        for operation in operations[distinct_index + 1 : anti_index]:
            if op_kind(operation) == "filter":
                candidate_rows = [
                    row for row in candidate_rows if _row_passes_filter(row, operation)
                ]

    anti_left_keys = _column_list(anti.get("left_on")) if anti else []
    anti_right_keys = _column_list(anti.get("right_on")) if anti else []
    anti_table = _table_by_name(case, str(anti.get("table", "") or "")) if anti else None
    excluded_keys = {
        _stable_row_key(row, anti_right_keys)
        for row in ([] if anti_table is None else anti_table.rows)
        if anti_right_keys
    }
    anti_matches = [
        row
        for row in candidate_rows
        if _stable_row_key(row, anti_left_keys) in excluded_keys
    ]
    anti_survivors = [
        row
        for row in candidate_rows
        if _stable_row_key(row, anti_left_keys) not in excluded_keys
    ]
    window_source = str(window.get("source", "") or "") if window else ""
    nullable_window_count = sum(row.get(window_source) is None for row in anti_survivors)
    partition_columns = _column_list(window.get("partition_by")) if window else []
    partition_counts = Counter(
        _stable_row_key(row, partition_columns)
        for row in anti_survivors
        if partition_columns
    )
    order_keys = normalize_sort_keys({"keys": window.get("order_by", [])}) if window else []

    observed: list[str] = []
    _observe(
        observed,
        "union_distinct_window_pipeline",
        None not in {union_index, distinct_index, anti_index, window_index}
        and union_index < distinct_index < anti_index < window_index,
    )
    _observe(observed, "duplicate_bag_to_set_transition", duplicate_count > 0)
    _observe(
        observed,
        "anti_join_match_and_survivor",
        bool(anti_matches) and bool(anti_survivors),
    )
    _observe(observed, "nullable_window_input", nullable_window_count > 0)
    _observe(
        observed,
        "multirow_window_partition",
        any(count >= 2 for count in partition_counts.values()),
    )
    _observe(observed, "deterministic_window_order", len(order_keys) >= 2)
    return _WitnessEvaluation(
        evaluator_id="union-distinct-window-static-v1",
        required_tokens=required,
        observed_tokens=tuple(observed),
        witness_summary={
            "union_index": union_index,
            "distinct_index": distinct_index,
            "anti_join_index": anti_index,
            "window_index": window_index,
            "combined_row_count": len(combined_rows),
            "distinct_columns": distinct_columns,
            "duplicate_row_count": duplicate_count,
            "anti_match_count": len(anti_matches),
            "anti_survivor_count": len(anti_survivors),
            "nullable_window_input_count": nullable_window_count,
            "max_partition_size": max(partition_counts.values(), default=0),
            "window_order_key_count": len(order_keys),
        },
    )


def _evaluate_large_integer_cast_window(case: Case) -> _WitnessEvaluation:
    required = (
        "cast_membership_window_pipeline",
        "integer_string_exact_cast",
        "adjacent_precision_boundary_values",
        "membership_match_and_rejection",
        "null_cast_guard",
        "signed_numeric_domain",
        "multirow_window_partition",
        "deterministic_window_order",
    )
    operations = list(case.program.operations)
    cast_index = next(
        (
            index
            for index, operation in enumerate(operations)
            if op_kind(operation) == "mutate"
            and str((operation.get("expr") or {}).get("kind", "") or "") == "cast"
        ),
        None,
    )
    membership_index = _first_index(operations, {"semi_join"}, after=cast_index)
    window_index = _first_index(operations, {"running_sum"}, after=membership_index)
    cast = operations[cast_index] if cast_index is not None else None
    membership = operations[membership_index] if membership_index is not None else None
    window = operations[window_index] if window_index is not None else None
    expression = cast.get("expr") if cast is not None else None
    source_column = str(expression.get("source", "") or "") if isinstance(expression, Mapping) else ""
    output_column = str(cast.get("column", "") or "") if cast is not None else ""
    text_values = [
        row.get(source_column)
        for row in ([] if not case.tables else case.tables[0].rows)
    ]
    parsed_values: list[int] = []
    for value in text_values:
        if value is None:
            continue
        try:
            parsed_values.append(int(str(value)))
        except ValueError:
            continue
    precision_edge = 2**53
    adjacent_precision_values = {
        value for value in parsed_values if precision_edge - 1 <= abs(value) <= precision_edge + 1
    }
    membership_right_key = _column_list(membership.get("right_on")) if membership else []
    membership_table = (
        _table_by_name(case, str(membership.get("table", "") or ""))
        if membership is not None
        else None
    )
    membership_values = {
        row.get(membership_right_key[0])
        for row in ([] if membership_table is None else membership_table.rows)
        if membership_right_key
    }
    matched_values = [value for value in parsed_values if value in membership_values]
    rejected_values = [value for value in parsed_values if value not in membership_values]
    primary_rows = [] if not case.tables else case.tables[0].rows
    partition_columns = _column_list(window.get("partition_by")) if window else []
    matched_rows = []
    for row in primary_rows:
        raw = row.get(source_column)
        if raw is None:
            continue
        try:
            parsed = int(str(raw))
        except ValueError:
            continue
        if parsed in membership_values:
            matched_rows.append(row)
    partition_counts = Counter(
        _stable_row_key(row, partition_columns[:1])
        for row in matched_rows
        if partition_columns
    )
    order_keys = normalize_sort_keys({"keys": window.get("order_by", [])}) if window else []
    exact_cast = bool(
        isinstance(expression, Mapping)
        and str(expression.get("to", "") or "") == "int"
        and str(expression.get("input_domain", "") or "") == "integer_string"
        and output_column
    )
    signed_domain = (
        any(value < 0 for value in parsed_values)
        and any(value == 0 for value in parsed_values)
        and any(value > 0 for value in parsed_values)
    )

    observed: list[str] = []
    _observe(
        observed,
        "cast_membership_window_pipeline",
        None not in {cast_index, membership_index, window_index}
        and cast_index < membership_index < window_index,
    )
    _observe(observed, "integer_string_exact_cast", exact_cast)
    _observe(
        observed,
        "adjacent_precision_boundary_values",
        len(adjacent_precision_values) >= 3,
    )
    _observe(
        observed,
        "membership_match_and_rejection",
        bool(matched_values) and bool(rejected_values),
    )
    _observe(observed, "null_cast_guard", any(value is None for value in text_values))
    _observe(observed, "signed_numeric_domain", signed_domain)
    _observe(
        observed,
        "multirow_window_partition",
        any(count >= 2 for count in partition_counts.values()),
    )
    _observe(observed, "deterministic_window_order", len(order_keys) >= 2)
    return _WitnessEvaluation(
        evaluator_id="large-integer-cast-window-static-v1",
        required_tokens=required,
        observed_tokens=tuple(observed),
        witness_summary={
            "cast_index": cast_index,
            "membership_index": membership_index,
            "window_index": window_index,
            "cast_source_column": source_column,
            "cast_output_column": output_column,
            "parsed_value_count": len(parsed_values),
            "adjacent_precision_values": sorted(adjacent_precision_values),
            "matched_value_count": len(matched_values),
            "rejected_value_count": len(rejected_values),
            "null_text_count": sum(value is None for value in text_values),
            "max_partition_size": max(partition_counts.values(), default=0),
            "window_order_key_count": len(order_keys),
        },
    )


def _evaluate_empty_union_groupby(case: Case) -> _WitnessEvaluation:
    required = (
        "empty_union_groupby_pipeline",
        "filter_produces_empty_branch",
        "nonempty_compatible_union_branch",
        "nullable_union_payload",
        "multiple_group_observation",
        "deterministic_group_order",
    )
    operations = list(case.program.operations)
    filter_index = _first_index(operations, {"filter"})
    union_index = _first_index(operations, {"union_all"}, after=filter_index)
    group_index = _first_index(operations, {"groupby", "aggregate"}, after=union_index)
    sort_index = _first_index(operations, {"sort"}, after=group_index)
    filter_operation = operations[filter_index] if filter_index is not None else None
    union = operations[union_index] if union_index is not None else None
    group = operations[group_index] if group_index is not None else None
    base = case.tables[0] if case.tables else None
    append = _table_by_name(case, str(union.get("table", "") or "")) if union else None
    filtered_rows = [
        row
        for row in ([] if base is None else base.rows)
        if filter_operation is not None and _row_passes_filter(row, filter_operation)
    ]
    base_schema = [] if base is None else [(column.name, column.type) for column in base.columns]
    append_schema = [] if append is None else [(column.name, column.type) for column in append.columns]
    append_rows = [] if append is None else append.rows
    null_payload_count = sum(
        value is None for row in append_rows for value in row.values()
    )
    group_keys = _column_list(group.get("keys")) if group else []
    fill_values: dict[str, Any] = {}
    if union_index is not None and group_index is not None:
        for operation in operations[union_index + 1 : group_index]:
            if op_kind(operation) == "fill_null":
                fill_values[str(operation.get("column", "") or "")] = operation.get("value")
    group_values = set()
    for row in append_rows:
        key = tuple(
            _stable_value_key(
                fill_values.get(column) if row.get(column) is None and column in fill_values else row.get(column)
            )
            for column in group_keys
        )
        if group_keys:
            group_values.add(key)
    sort_keys = (
        normalize_sort_keys(operations[sort_index]) if sort_index is not None else []
    )

    observed: list[str] = []
    _observe(
        observed,
        "empty_union_groupby_pipeline",
        None not in {filter_index, union_index, group_index, sort_index}
        and filter_index < union_index < group_index < sort_index,
    )
    _observe(observed, "filter_produces_empty_branch", not filtered_rows)
    _observe(
        observed,
        "nonempty_compatible_union_branch",
        bool(append_rows) and base_schema == append_schema,
    )
    _observe(observed, "nullable_union_payload", null_payload_count > 0)
    _observe(observed, "multiple_group_observation", len(group_values) >= 2)
    _observe(observed, "deterministic_group_order", bool(sort_keys))
    return _WitnessEvaluation(
        evaluator_id="empty-union-groupby-static-v1",
        required_tokens=required,
        observed_tokens=tuple(observed),
        witness_summary={
            "filter_index": filter_index,
            "union_index": union_index,
            "group_index": group_index,
            "sort_index": sort_index,
            "base_row_count": 0 if base is None else len(base.rows),
            "filtered_base_row_count": len(filtered_rows),
            "append_row_count": len(append_rows),
            "schema_compatible": base_schema == append_schema,
            "nullable_append_value_count": null_payload_count,
            "observed_group_count": len(group_values),
            "sort_key_count": len(sort_keys),
        },
    )


def _evaluate_nullable_grouped_topk(case: Case) -> _WitnessEvaluation:
    required = (
        "grouped_topk_pipeline",
        "nullable_aggregate",
        "aggregate_drives_sort",
        "null_and_non_null_aggregate_groups",
        "active_topk_cut",
    )
    operations = list(case.program.operations)
    group_index = _first_index(operations, {"groupby", "aggregate"})
    sort_index = _first_index(operations, {"sort"}, after=group_index)
    limit_index = _first_index(operations, {"limit"}, after=sort_index)
    group = operations[group_index] if group_index is not None else None
    sort = operations[sort_index] if sort_index is not None else None
    limit = operations[limit_index] if limit_index is not None else None
    aggregates = list(group.get("aggs", []) or []) if group is not None else []
    sort_columns = [key.column for key in normalize_sort_keys(sort)] if sort is not None else []
    nullable_functions = {
        "min",
        "max",
        "sum",
        "mean",
        "median",
        "first",
        "last",
        "any",
        "all",
    }
    relevant_aggregates = [
        aggregate
        for aggregate in aggregates
        if isinstance(aggregate, Mapping)
        and str(aggregate.get("as", "") or "") in sort_columns
        and str(aggregate.get("func", "") or "") in nullable_functions
    ]
    primary = case.tables[0] if case.tables else None
    rows = list(primary.rows) if primary is not None else []
    nullable_input_count = sum(
        row.get(str(aggregate.get("column", "") or "")) is None
        for aggregate in relevant_aggregates
        for row in rows
    )
    group_keys = _column_list(group.get("keys")) if group is not None else []
    null_group_count, non_null_group_count, potential_group_count = _aggregate_group_witness(
        rows,
        group_keys=group_keys,
        aggregate=(relevant_aggregates[0] if relevant_aggregates else None),
    )
    limit_n = _nonnegative_int(limit.get("n")) if limit is not None else 0
    active_topk = limit is not None and potential_group_count > limit_n

    observed: list[str] = []
    _observe(
        observed,
        "grouped_topk_pipeline",
        group_index is not None
        and sort_index is not None
        and limit_index is not None
        and group_index < sort_index < limit_index,
    )
    _observe(
        observed,
        "nullable_aggregate",
        bool(relevant_aggregates) and nullable_input_count > 0,
    )
    _observe(observed, "aggregate_drives_sort", bool(relevant_aggregates))
    _observe(
        observed,
        "null_and_non_null_aggregate_groups",
        null_group_count > 0 and non_null_group_count > 0,
    )
    _observe(observed, "active_topk_cut", active_topk)
    return _WitnessEvaluation(
        evaluator_id="nullable-grouped-topk-static-v1",
        required_tokens=required,
        observed_tokens=tuple(observed),
        witness_summary={
            "group_index": group_index,
            "sort_index": sort_index,
            "limit_index": limit_index,
            "group_keys": group_keys,
            "sort_columns": sort_columns,
            "aggregate_aliases": [
                str(aggregate.get("as", "") or "")
                for aggregate in aggregates
                if isinstance(aggregate, Mapping)
            ],
            "relevant_aggregate_aliases": [
                str(aggregate.get("as", "") or "")
                for aggregate in relevant_aggregates
            ],
            "nullable_input_count": nullable_input_count,
            "null_aggregate_group_count": null_group_count,
            "non_null_aggregate_group_count": non_null_group_count,
            "potential_group_count": potential_group_count,
            "limit_n": limit_n,
        },
    )


def _evaluate_pyarrow_layout_bool_groupby(case: Case) -> _WitnessEvaluation:
    required = (
        "registered_physical_layout",
        "layout_directive_reaches_primary_input",
        "boolean_groupby_any_all",
        "nullable_false_bitmap_pattern",
        "registered_variant_and_data_pattern",
        "bounded_layout_palette_cell",
        "static_root_guidance_without_corpus_io",
    )
    metadata = case.metadata if isinstance(case.metadata, Mapping) else {}
    witness = metadata.get("physical_layout_witness", {})
    if not isinstance(witness, Mapping):
        witness = {}
    layout = str(witness.get("layout", "") or "")
    variant_id = str(witness.get("variant_id", "") or "")
    data_pattern_id = str(witness.get("data_pattern_id", "") or "")
    layout_map = metadata.get("input_layouts", {})
    primary_layout = (
        layout_map.get(case.tables[0].name, {})
        if case.tables and isinstance(layout_map, Mapping)
        else {}
    )
    representation = str(
        primary_layout.get("representation", "")
        if isinstance(primary_layout, Mapping)
        else ""
    )
    group = next(
        (
            operation
            for operation in case.program.operations
            if op_kind(operation) == "groupby"
        ),
        None,
    )
    aggregates = list(group.get("aggs", []) or []) if group is not None else []
    aggregate_functions = {
        str(aggregate.get("func", "") or "")
        for aggregate in aggregates
        if isinstance(aggregate, Mapping)
        and str(aggregate.get("column", "") or "") == "flag"
    }
    primary = case.tables[0] if case.tables else None
    flag_type = next(
        (
            column.type
            for column in ([] if primary is None else primary.columns)
            if column.name == "flag"
        ),
        "",
    )
    flag_values = [
        row.get("flag") for row in ([] if primary is None else primary.rows)
    ]
    cell_index = witness.get("cell_index")
    bounded_cell = (
        isinstance(cell_index, int)
        and 0 <= cell_index < 18
    )
    root_ids = tuple(str(value) for value in witness.get("root_ids", ()) or ())

    observed: list[str] = []
    _observe(
        observed,
        "registered_physical_layout",
        layout in {"contiguous", "sliced", "chunked"},
    )
    _observe(
        observed,
        "layout_directive_reaches_primary_input",
        bool(layout) and representation == layout,
    )
    _observe(
        observed,
        "boolean_groupby_any_all",
        group is not None
        and flag_type == "bool"
        and {"any", "all"} <= aggregate_functions,
    )
    _observe(
        observed,
        "nullable_false_bitmap_pattern",
        len(flag_values) == 2
        and False in flag_values
        and any(value is None for value in flag_values),
    )
    _observe(
        observed,
        "registered_variant_and_data_pattern",
        variant_id in {"any_all", "all_any"}
        and data_pattern_id
        in {
            "single_group_false_null",
            "two_groups_false_null",
            "null_key_false_null",
        },
    )
    _observe(observed, "bounded_layout_palette_cell", bounded_cell)
    _observe(
        observed,
        "static_root_guidance_without_corpus_io",
        "pyarrow-sliced-bool-hash-aggregate-001" in root_ids
        and not bool(witness.get("canonical_case_replay", True))
        and not bool(witness.get("runtime_corpus_io", True)),
    )
    return _WitnessEvaluation(
        evaluator_id="pyarrow-layout-bool-groupby-static-v1",
        required_tokens=required,
        observed_tokens=tuple(observed),
        witness_summary={
            "layout": layout,
            "input_representation": representation,
            "variant_id": variant_id,
            "data_pattern_id": data_pattern_id,
            "cell_index": cell_index,
            "aggregate_functions": sorted(aggregate_functions),
            "flag_values": flag_values,
            "root_ids": list(root_ids),
            "canonical_case_replay": bool(
                witness.get("canonical_case_replay", False)
            ),
            "runtime_corpus_io": bool(witness.get("runtime_corpus_io", False)),
        },
    )


def _evaluate_polars_reflected_arithmetic(case: Case) -> _WitnessEvaluation:
    from datadiff.semantic_reflected_arithmetic_witness import (
        REFLECTED_ARITHMETIC_CELL_COUNT,
        REFLECTED_ARITHMETIC_DATA_PATTERNS,
        REFLECTED_ARITHMETIC_NAME_MODES,
        REFLECTED_ARITHMETIC_OPERATORS,
        REFLECTED_ARITHMETIC_ROOT_ID,
        reflected_arithmetic_expected_values,
        reflected_arithmetic_reversed_values,
    )

    required = (
        "registered_reflected_operator",
        "native_series_probe",
        "registered_name_mode_and_data_pattern",
        "static_noncommutative_operands",
        "bounded_family_palette_cell",
        "family_axes_match_operation",
        "static_root_guidance_without_corpus_io",
    )
    metadata = case.metadata if isinstance(case.metadata, Mapping) else {}
    witness = metadata.get("reflected_arithmetic_witness", {})
    family_witness = metadata.get("family_witness", {})
    if not isinstance(witness, Mapping):
        witness = {}
    if not isinstance(family_witness, Mapping):
        family_witness = {}
    operation = next(
        (
            item
            for item in case.program.operations
            if op_kind(item) == "series_reflected_arithmetic_probe"
        ),
        None,
    )
    operator = str(witness.get("operator", "") or "")
    name_mode = str(witness.get("name_mode", "") or "")
    data_pattern_id = str(witness.get("data_pattern_id", "") or "")
    lhs_values = list(witness.get("lhs_values", []) or [])
    rhs_values = list(witness.get("rhs_values", []) or [])
    expected_values: list[Any] = []
    reversed_values: list[Any] = []
    try:
        expected_values = reflected_arithmetic_expected_values(
            operator,
            lhs_values,
            rhs_values,
        )
        reversed_values = reflected_arithmetic_reversed_values(
            operator,
            lhs_values,
            rhs_values,
        )
    except (TypeError, ValueError, ZeroDivisionError, OverflowError):
        pass
    cell_index = witness.get("cell_index")
    family_axes = family_witness.get("axes", {})
    if not isinstance(family_axes, Mapping):
        family_axes = {}
    root_ids = tuple(str(value) for value in witness.get("root_ids", ()) or ())

    observed: list[str] = []
    _observe(observed, "registered_reflected_operator", operator in REFLECTED_ARITHMETIC_OPERATORS)
    _observe(
        observed,
        "native_series_probe",
        operation is not None
        and str(operation.get("operator", "") or "") == operator
        and list(operation.get("lhs_values", []) or []) == lhs_values
        and list(operation.get("rhs_values", []) or []) == rhs_values,
    )
    _observe(
        observed,
        "registered_name_mode_and_data_pattern",
        name_mode in REFLECTED_ARITHMETIC_NAME_MODES
        and data_pattern_id in REFLECTED_ARITHMETIC_DATA_PATTERNS,
    )
    _observe(
        observed,
        "static_noncommutative_operands",
        bool(expected_values)
        and expected_values != reversed_values
        and expected_values == list(witness.get("expected_values", []) or []),
    )
    _observe(
        observed,
        "bounded_family_palette_cell",
        isinstance(cell_index, int)
        and 0 <= cell_index < REFLECTED_ARITHMETIC_CELL_COUNT,
    )
    _observe(
        observed,
        "family_axes_match_operation",
        str(family_witness.get("family_id", "") or "")
        == "polars_reflected_arithmetic_operand_order"
        and str(family_axes.get("operator", "") or "") == operator
        and str(family_axes.get("name_mode", "") or "") == name_mode
        and str(family_axes.get("data_pattern", "") or "") == data_pattern_id,
    )
    _observe(
        observed,
        "static_root_guidance_without_corpus_io",
        REFLECTED_ARITHMETIC_ROOT_ID in root_ids
        and not bool(witness.get("canonical_case_replay", True))
        and not bool(witness.get("runtime_corpus_io", True)),
    )
    return _WitnessEvaluation(
        evaluator_id="polars-reflected-arithmetic-static-v1",
        required_tokens=required,
        observed_tokens=tuple(observed),
        witness_summary={
            "operator": operator,
            "name_mode": name_mode,
            "data_pattern_id": data_pattern_id,
            "cell_index": cell_index,
            "lhs_values": lhs_values,
            "rhs_values": rhs_values,
            "expected_values": expected_values,
            "reversed_values": reversed_values,
            "root_ids": list(root_ids),
            "canonical_case_replay": bool(
                witness.get("canonical_case_replay", False)
            ),
            "runtime_corpus_io": bool(witness.get("runtime_corpus_io", False)),
        },
    )


def _evaluate_datafusion_grouped_null_topk(case: Case) -> _WitnessEvaluation:
    from datadiff.semantic_datafusion_grouped_null_topk_witness import (
        DATAFUSION_GROUPED_NULL_TOPK_AGGREGATES,
        DATAFUSION_GROUPED_NULL_TOPK_CELL_COUNT,
        DATAFUSION_GROUPED_NULL_TOPK_DATA_PATTERNS,
        DATAFUSION_GROUPED_NULL_TOPK_EXPOSURE_MODES,
        DATAFUSION_GROUPED_NULL_TOPK_FAMILY_ID,
        DATAFUSION_GROUPED_NULL_TOPK_ROOT_ID,
        datafusion_grouped_null_topk_expected_values,
    )

    required = (
        "registered_aggregate_exposure_pattern",
        "native_datafusion_probe",
        "all_null_group_present",
        "aggregate_specific_bug_direction",
        "topk_exposes_null_group",
        "bounded_family_palette_cell",
        "family_axes_match_operation",
        "static_root_guidance_without_corpus_io",
    )
    metadata = case.metadata if isinstance(case.metadata, Mapping) else {}
    witness = metadata.get("datafusion_grouped_null_topk_witness", {})
    family_witness = metadata.get("family_witness", {})
    if not isinstance(witness, Mapping):
        witness = {}
    if not isinstance(family_witness, Mapping):
        family_witness = {}
    operation = next(
        (
            item
            for item in case.program.operations
            if op_kind(item) == "datafusion_grouped_null_topk_probe"
        ),
        None,
    )
    aggregate = str(witness.get("aggregate", "") or "")
    exposure_mode = str(witness.get("exposure_mode", "") or "")
    data_pattern_id = str(witness.get("data_pattern_id", "") or "")
    direction = str(witness.get("direction", "") or "")
    nulls = str(witness.get("nulls", "") or "")
    limit = witness.get("limit")
    raw_rows = list(witness.get("rows", []) or [])
    rows = [
        (str(row.get("g", "") or ""), row.get("x"))
        for row in raw_rows
        if isinstance(row, Mapping)
    ]
    expected_values: list[Any] = []
    try:
        expected_values = datafusion_grouped_null_topk_expected_values(
            aggregate=aggregate,
            exposure_mode=exposure_mode,
            rows=rows,
        )
    except (TypeError, ValueError):
        pass
    all_null_groups = {
        group
        for group, _value in rows
        if all(
            candidate is None
            for candidate_group, candidate in rows
            if candidate_group == group
        )
    }
    family_axes = family_witness.get("axes", {})
    if not isinstance(family_axes, Mapping):
        family_axes = {}
    cell_index = witness.get("cell_index")
    root_ids = tuple(str(value) for value in witness.get("root_ids", ()) or ())

    observed: list[str] = []
    _observe(
        observed,
        "registered_aggregate_exposure_pattern",
        aggregate in DATAFUSION_GROUPED_NULL_TOPK_AGGREGATES
        and exposure_mode in DATAFUSION_GROUPED_NULL_TOPK_EXPOSURE_MODES
        and data_pattern_id in DATAFUSION_GROUPED_NULL_TOPK_DATA_PATTERNS,
    )
    _observe(
        observed,
        "native_datafusion_probe",
        operation is not None
        and str(operation.get("aggregate", "") or "") == aggregate
        and str(operation.get("exposure_mode", "") or "") == exposure_mode
        and str(operation.get("data_pattern_id", "") or "")
        == data_pattern_id
        and list(operation.get("expected_values", []) or [])
        == expected_values,
    )
    _observe(observed, "all_null_group_present", bool(all_null_groups))
    _observe(
        observed,
        "aggregate_specific_bug_direction",
        (aggregate == "min" and direction == "asc")
        or (aggregate == "max" and direction == "desc"),
    )
    _observe(
        observed,
        "topk_exposes_null_group",
        None in expected_values
        and (
            exposure_mode == "full_nulls_last"
            and nulls == "last"
            and isinstance(limit, int)
            and limit >= len({group for group, _value in rows})
            or exposure_mode == "top1_nulls_first"
            and nulls == "first"
            and limit == 1
            and expected_values[:1] == [None]
        ),
    )
    _observe(
        observed,
        "bounded_family_palette_cell",
        isinstance(cell_index, int)
        and 0 <= cell_index < DATAFUSION_GROUPED_NULL_TOPK_CELL_COUNT,
    )
    _observe(
        observed,
        "family_axes_match_operation",
        str(family_witness.get("family_id", "") or "")
        == DATAFUSION_GROUPED_NULL_TOPK_FAMILY_ID
        and str(family_axes.get("aggregate", "") or "") == aggregate
        and str(family_axes.get("exposure_mode", "") or "")
        == exposure_mode
        and str(family_axes.get("data_pattern", "") or "")
        == data_pattern_id,
    )
    _observe(
        observed,
        "static_root_guidance_without_corpus_io",
        DATAFUSION_GROUPED_NULL_TOPK_ROOT_ID in root_ids
        and not bool(witness.get("canonical_case_replay", True))
        and not bool(witness.get("runtime_corpus_io", True)),
    )
    return _WitnessEvaluation(
        evaluator_id="datafusion-grouped-null-topk-static-v1",
        required_tokens=required,
        observed_tokens=tuple(observed),
        witness_summary={
            "aggregate": aggregate,
            "exposure_mode": exposure_mode,
            "data_pattern_id": data_pattern_id,
            "cell_index": cell_index,
            "direction": direction,
            "nulls": nulls,
            "limit": limit,
            "expected_values": expected_values,
            "all_null_groups": sorted(all_null_groups),
            "root_ids": list(root_ids),
            "canonical_case_replay": bool(
                witness.get("canonical_case_replay", False)
            ),
            "runtime_corpus_io": bool(witness.get("runtime_corpus_io", False)),
        },
    )


def _evaluate_confirmed_root_witness(case: Case) -> _WitnessEvaluation:
    from datadiff.family_witness_registry import family_witness_registration
    from datadiff.semantic_confirmed_root_witness import (
        CONFIRMED_ROOT_PROBE_KIND,
        confirmed_root_witness_static_preconditions,
    )

    required = (
        "registered_witness_identity",
        "native_target_probe",
        "registered_axes",
        "bounded_family_palette_cell",
        "family_axes_match_operation",
        "mechanism_preconditions",
        "static_root_guidance_without_corpus_io",
    )
    metadata = case.metadata if isinstance(case.metadata, Mapping) else {}
    witness = metadata.get("confirmed_root_witness", {})
    family_witness = metadata.get("family_witness", {})
    if not isinstance(witness, Mapping):
        witness = {}
    if not isinstance(family_witness, Mapping):
        family_witness = {}
    family_id = str(witness.get("family_id", "") or "")
    root_ids = tuple(str(value) for value in witness.get("root_ids", ()) or ())
    root_id = root_ids[0] if len(root_ids) == 1 else ""
    axes_raw = witness.get("axes", {})
    axes = (
        {str(key): str(value) for key, value in axes_raw.items()}
        if isinstance(axes_raw, Mapping)
        else {}
    )
    operation = next(
        (
            item
            for item in case.program.operations
            if op_kind(item) == CONFIRMED_ROOT_PROBE_KIND
        ),
        None,
    )
    try:
        registration = family_witness_registration(family_id)
    except KeyError:
        registration = None
    cell_index = witness.get("cell_index")
    encoded_cell_index: int | None = None
    if registration is not None and isinstance(cell_index, int):
        try:
            encoded_cell_index, encoded_axes = registration.cell_for_seed(cell_index)
        except (TypeError, ValueError):
            encoded_axes = {}
    else:
        encoded_axes = {}
    family_axes_raw = family_witness.get("axes", {})
    family_axes = (
        {str(key): str(value) for key, value in family_axes_raw.items()}
        if isinstance(family_axes_raw, Mapping)
        else {}
    )
    operation_axes_raw = operation.get("axes", {}) if operation is not None else {}
    operation_axes = (
        {str(key): str(value) for key, value in operation_axes_raw.items()}
        if isinstance(operation_axes_raw, Mapping)
        else {}
    )
    preconditions = bool(
        operation is not None
        and confirmed_root_witness_static_preconditions(
            root_id,
            axes,
            operation,
        )
    )

    observed: list[str] = []
    _observe(
        observed,
        "registered_witness_identity",
        registration is not None
        and registration.root_id == root_id
        and registration.goal_id == str(witness.get("goal_id", "") or "")
        and str(family_witness.get("family_id", "") or "") == family_id,
    )
    _observe(
        observed,
        "native_target_probe",
        operation is not None
        and str(operation.get("root_id", "") or "") == root_id
        and str(operation.get("target_backend", "") or "")
        == str(witness.get("target_backend", "") or "")
        and str(operation.get("root_cause", "") or "")
        == str(witness.get("root_cause", "") or ""),
    )
    _observe(
        observed,
        "registered_axes",
        registration is not None
        and axes == encoded_axes
        and set(axes) == set(registration.axis_names),
    )
    _observe(
        observed,
        "bounded_family_palette_cell",
        registration is not None
        and isinstance(cell_index, int)
        and encoded_cell_index == cell_index
        and 0 <= cell_index < registration.cell_count,
    )
    _observe(
        observed,
        "family_axes_match_operation",
        family_axes == axes == operation_axes,
    )
    _observe(observed, "mechanism_preconditions", preconditions)
    _observe(
        observed,
        "static_root_guidance_without_corpus_io",
        bool(root_id)
        and not bool(witness.get("canonical_case_replay", True))
        and not bool(witness.get("runtime_corpus_io", True))
        and not bool(family_witness.get("canonical_case_replay", True))
        and not bool(family_witness.get("runtime_corpus_io", True)),
    )
    return _WitnessEvaluation(
        evaluator_id="confirmed-root-family-static-v1",
        required_tokens=required,
        observed_tokens=tuple(observed),
        witness_summary={
            "family_id": family_id,
            "root_id": root_id,
            "target_backend": str(witness.get("target_backend", "") or ""),
            "root_cause": str(witness.get("root_cause", "") or ""),
            "cell_index": cell_index,
            "axes": axes,
            "registered": registration is not None,
            "mechanism_preconditions": preconditions,
            "canonical_case_replay": bool(
                witness.get("canonical_case_replay", False)
            ),
            "runtime_corpus_io": bool(witness.get("runtime_corpus_io", False)),
        },
    )


def _evaluate_semantic_family_expansion(case: Case) -> _WitnessEvaluation:
    from datadiff.family_witness_registry import family_witness_registration
    from datadiff.semantic_family_expansion_witness import (
        semantic_family_expansion_static_preconditions,
    )
    from datadiff.semantic_family_expansion_v2_witness import (
        semantic_family_expansion_v2_static_preconditions,
    )
    from datadiff.semantic_family_expansion_v3_witness import (
        semantic_family_expansion_v3_static_preconditions,
    )

    required = (
        "registered_family_identity",
        "registered_axes",
        "bounded_family_palette_cell",
        "target_and_controls_declared",
        "mechanism_preconditions",
        "static_target_guidance_without_corpus_io",
    )
    metadata = case.metadata if isinstance(case.metadata, Mapping) else {}
    witness = metadata.get("semantic_family_expansion_witness", {})
    family_witness = metadata.get("family_witness", {})
    if not isinstance(witness, Mapping):
        witness = {}
    if not isinstance(family_witness, Mapping):
        family_witness = {}
    family_id = str(witness.get("family_id", "") or "")
    axes_raw = witness.get("axes", {})
    axes = (
        {str(key): str(value) for key, value in axes_raw.items()}
        if isinstance(axes_raw, Mapping)
        else {}
    )
    try:
        registration = family_witness_registration(family_id)
    except KeyError:
        registration = None
    cell_index = witness.get("cell_index")
    encoded_index: int | None = None
    encoded_axes: dict[str, str] = {}
    if registration is not None and isinstance(cell_index, int):
        try:
            encoded_index, encoded_axes = registration.cell_for_seed(cell_index)
        except (TypeError, ValueError):
            pass
    v1_preconditions = bool(
        registration is not None
        and semantic_family_expansion_static_preconditions(family_id, axes, case)
    )
    v2_preconditions = bool(
        registration is not None
        and semantic_family_expansion_v2_static_preconditions(
            family_id,
            axes,
            case,
        )
    )
    v3_preconditions = bool(
        registration is not None
        and semantic_family_expansion_v3_static_preconditions(
            family_id,
            axes,
            case,
        )
    )
    preconditions = v1_preconditions or v2_preconditions or v3_preconditions

    observed: list[str] = []
    _observe(
        observed,
        "registered_family_identity",
        registration is not None
        and registration.goal_id == str(witness.get("goal_id", "") or "")
        and str(family_witness.get("family_id", "") or "") == family_id,
    )
    _observe(
        observed,
        "registered_axes",
        registration is not None
        and axes == encoded_axes
        and family_witness.get("axes") == axes,
    )
    _observe(
        observed,
        "bounded_family_palette_cell",
        registration is not None
        and isinstance(cell_index, int)
        and encoded_index == cell_index
        and 0 <= cell_index < registration.cell_count,
    )
    _observe(
        observed,
        "target_and_controls_declared",
        registration is not None
        and str(witness.get("target_backend", "") or "")
        == registration.target_backend
        and set(witness.get("control_backends", ()) or ())
        == (set(registration.backends) - {registration.target_backend}),
    )
    _observe(observed, "mechanism_preconditions", preconditions)
    _observe(
        observed,
        "static_target_guidance_without_corpus_io",
        bool(witness.get("root_ids", ()) or ())
        and not bool(witness.get("canonical_case_replay", True))
        and not bool(witness.get("runtime_corpus_io", True))
        and not bool(family_witness.get("canonical_case_replay", True))
        and not bool(family_witness.get("runtime_corpus_io", True)),
    )
    return _WitnessEvaluation(
        evaluator_id=(
            "semantic-family-expansion-static-v3"
            if v3_preconditions
            else "semantic-family-expansion-static-v2"
            if v2_preconditions
            else "semantic-family-expansion-static-v1"
        ),
        required_tokens=required,
        observed_tokens=tuple(observed),
        witness_summary={
            "family_id": family_id,
            "mechanism_id": str(witness.get("mechanism_id", "") or ""),
            "target_backend": str(witness.get("target_backend", "") or ""),
            "control_backends": list(witness.get("control_backends", ()) or ()),
            "cell_index": cell_index,
            "axes": axes,
            "registered": registration is not None,
            "mechanism_preconditions": preconditions,
            "operation_kinds": list(witness.get("operation_kinds", ()) or ()),
            "canonical_case_replay": bool(
                witness.get("canonical_case_replay", False)
            ),
            "runtime_corpus_io": bool(witness.get("runtime_corpus_io", False)),
        },
    )


_EVALUATORS: dict[str, ActivationEvaluator] = {
    "order_offset_aggregate": _evaluate_order_offset_aggregate,
    "nullable_membership_join": _evaluate_nullable_membership_join,
    "union_distinct_window": _evaluate_union_distinct_window,
    "large_integer_cast_window": _evaluate_large_integer_cast_window,
    "empty_union_groupby": _evaluate_empty_union_groupby,
    "nullable_grouped_topk": _evaluate_nullable_grouped_topk,
    "pyarrow_layout_bool_groupby": _evaluate_pyarrow_layout_bool_groupby,
    "polars_reflected_arithmetic": _evaluate_polars_reflected_arithmetic,
    "datafusion_grouped_null_topk": _evaluate_datafusion_grouped_null_topk,
    "datafusion_limit_offset_pushdown": _evaluate_confirmed_root_witness,
    "datafusion_negative_zero_comparison": _evaluate_confirmed_root_witness,
    "datafusion_distinct_null_topk": _evaluate_confirmed_root_witness,
    "datafusion_ordered_limit_idempotence": _evaluate_confirmed_root_witness,
    "polars_grouped_max_sort_metadata": _evaluate_confirmed_root_witness,
    "duckdb_join_filter_pushdown_limit": _evaluate_confirmed_root_witness,
    "pandas_nullable_bool_reduction": _evaluate_semantic_family_expansion,
    "sqlite_affinity_null_ordered_cut": _evaluate_semantic_family_expansion,
    "polars_lazy_filter_groupby_window": _evaluate_semantic_family_expansion,
    "polars_lazy_temporal_cast_boundary": _evaluate_semantic_family_expansion,
    "pyarrow_encoded_nested_compute": _evaluate_semantic_family_expansion,
    "datafusion_window_order_join_interaction": _evaluate_semantic_family_expansion,
    "pandas_nullable_string_normalization": _evaluate_semantic_family_expansion,
    "pyarrow_string_layout_predicate": _evaluate_semantic_family_expansion,
    "polars_arithmetic_cast_sortedness": _evaluate_semantic_family_expansion,
    "polars_lazy_case_string_membership": _evaluate_semantic_family_expansion,
    "duckdb_union_duplicate_global_aggregate": _evaluate_semantic_family_expansion,
    "sqlite_three_valued_membership": _evaluate_semantic_family_expansion,
    "datafusion_null_setop_aggregate": _evaluate_semantic_family_expansion,
}

_EVALUATORS.update(
    {
        definition.family_id: _evaluate_semantic_family_expansion
        for definition in expansion_v3_family_definitions()
    }
)


def _first_index(
    operations: list[Any],
    kinds: set[str],
    *,
    before: int | None = None,
    after: int | None = None,
) -> int | None:
    for index, operation in enumerate(operations):
        if before is not None and index >= before:
            break
        if after is not None and index <= after:
            continue
        if op_kind(operation) in kinds:
            return index
    return None


def _cut_witness(
    operations: list[Any],
    *,
    input_rows: int,
    start: int | None,
    end: int | None,
) -> dict[str, Any]:
    row_count = max(0, int(input_rows))
    removed_rows = 0
    operation_indices: list[int] = []
    if start is None or end is None:
        return {
            "operation_indices": operation_indices,
            "rows_after_cut": row_count,
            "removed_rows": removed_rows,
        }
    for index in range(start + 1, end):
        operation = operations[index]
        kind = op_kind(operation)
        if kind not in {"limit", "offset"}:
            continue
        operation_indices.append(index)
        previous = row_count
        n = _nonnegative_int(operation.get("n"))
        row_count = min(row_count, n) if kind == "limit" else max(0, row_count - n)
        removed_rows += previous - row_count
    return {
        "operation_indices": operation_indices,
        "rows_after_cut": row_count,
        "removed_rows": removed_rows,
    }


def _column_values_before(
    case: Case,
    column: str,
    *,
    operation_index: int | None,
) -> list[Any]:
    if not column or not case.tables:
        return []
    values = [row.get(column) for row in case.tables[0].rows]
    if operation_index is None:
        return values
    for operation in case.program.operations[:operation_index]:
        if op_kind(operation) != "fill_null":
            continue
        if str(operation.get("column", "") or "") != column:
            continue
        replacement = operation.get("value")
        values = [replacement if value is None else value for value in values]
    return values


def _aggregate_group_witness(
    rows: list[Mapping[str, Any]],
    *,
    group_keys: list[str],
    aggregate: Mapping[str, Any] | None,
) -> tuple[int, int, int]:
    if not group_keys or aggregate is None:
        return 0, 0, 0
    source = str(aggregate.get("column", "") or "")
    if not source or any(any(key not in row for key in group_keys) for row in rows):
        return 0, 0, 0
    groups: dict[tuple[Any, ...], list[Any]] = defaultdict(list)
    for row in rows:
        groups[_stable_row_key(row, group_keys)].append(row.get(source))
    null_groups = sum(not any(value is not None for value in values) for values in groups.values())
    non_null_groups = sum(any(value is not None for value in values) for values in groups.values())
    return null_groups, non_null_groups, len(groups)


def _row_passes_filter(row: Mapping[str, Any], operation: Mapping[str, Any]) -> bool:
    column = str(operation.get("column", "") or "")
    comparator = str(operation.get("cmp", "") or "")
    actual = row.get(column)
    expected = operation.get("value")
    if comparator == "is_not_null":
        return actual is not None
    if comparator == "is_null":
        return actual is None
    if actual is None:
        return False
    try:
        if comparator == "<":
            return bool(actual < expected)
        if comparator == "<=":
            return bool(actual <= expected)
        if comparator == ">":
            return bool(actual > expected)
        if comparator == ">=":
            return bool(actual >= expected)
        if comparator == "==":
            return bool(actual == expected)
        if comparator == "!=":
            return bool(actual != expected)
    except TypeError:
        return False
    return True


def _table_by_name(case: Case, name: str) -> Any | None:
    return next((table for table in case.tables if table.name == name), None)


def _component_collision_count(keys: list[tuple[Any, ...]]) -> int:
    if not keys:
        return 0
    return sum(
        _duplicate_count([key[index] for key in keys])
        for index in range(len(keys[0]))
    )


def _duplicate_count(values: list[Any]) -> int:
    return sum(max(0, count - 1) for count in Counter(values).values())


def _null_counts(case: Case, columns: set[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for column in sorted(column for column in columns if column):
        count = sum(row.get(column) is None for table in case.tables for row in table.rows)
        if count:
            out[column] = count
    return out


def _primary_row_count(case: Case) -> int:
    return len(case.tables[0].rows) if case.tables else 0


def _column_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item)]
    text = str(value or "")
    return [text] if text else []


def _stable_row_key(row: Mapping[str, Any], columns: list[str]) -> tuple[Any, ...]:
    return tuple(_stable_value_key(row.get(column)) for column in columns)


def _stable_value_key(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value):
            return ("float", "nan")
        if value == 0.0:
            return ("float", "-0" if math.copysign(1.0, value) < 0 else "+0")
        return ("float", repr(value))
    if isinstance(value, (str, int, bool)) or value is None:
        return (type(value).__name__, value)
    if isinstance(value, (list, tuple)):
        return ("sequence", tuple(_stable_value_key(item) for item in value))
    if isinstance(value, Mapping):
        return (
            "mapping",
            tuple(
                (str(key), _stable_value_key(item))
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            ),
        )
    return (type(value).__name__, repr(value))


def _nonnegative_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _observe(tokens: list[str], token: str, condition: bool) -> None:
    if condition:
        tokens.append(token)


def _string_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(str(item) for item in value if str(item))
