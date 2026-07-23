from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from datadiff.boundary_values import apply_targeted_boundary_profile
from datadiff.dsl import Case
from datadiff.goal_first_variants import (
    WitnessBuilderVariant,
    witness_builder_variants,
)
from datadiff.operation_semantics import op_kind
from datadiff.profile_generators import (
    generate_empty_then_union_groupby_case,
    generate_large_int_text_membership_window_case,
    generate_multi_key_anti_join_null_guard_case,
    generate_nested_topk_offset_aggregate_case,
    generate_null_groupby_topk_case,
    generate_union_distinct_anti_running_sum_case,
)
from datadiff.semantic_core.activation import evaluate_semantic_activation


GOAL_FIRST_SCHEMA_VERSION = "goal-first-backward-generation-v1"
ACTIVATION_REPAIR_SCHEMA_VERSION = "semantic-activation-repair-v1"
SEMANTIC_WITNESS_EPOCH_SPAN = 512
SEMANTIC_WITNESS_VARIANT_LANE_COUNT = 3
SEMANTIC_WITNESS_ACTIVE_VARIANTS_PER_EPOCH = 2
SEMANTIC_WITNESS_DATA_PATTERN_LANE_COUNT = 3
SEMANTIC_WITNESS_PHASE_COUNT = (
    SEMANTIC_WITNESS_VARIANT_LANE_COUNT
    * SEMANTIC_WITNESS_DATA_PATTERN_LANE_COUNT
)
GoalBuilder = Callable[[int], Case]


@dataclass(frozen=True, slots=True)
class GenerationGoal:
    goal_id: str
    target_contract: str
    fault_models: tuple[str, ...]
    interaction: tuple[str, ...]
    required_types: tuple[str, ...]
    required_operation_groups: tuple[tuple[str, ...], ...]
    preferred_boundary_profiles: tuple[str, ...]
    backward_steps: tuple[str, ...]
    builder: GoalBuilder
    witness_variants: tuple[WitnessBuilderVariant, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal_id": self.goal_id,
            "target_contract": self.target_contract,
            "fault_models": list(self.fault_models),
            "interaction": list(self.interaction),
            "required_types": list(self.required_types),
            "required_operation_groups": [
                list(group) for group in self.required_operation_groups
            ],
            "preferred_boundary_profiles": list(self.preferred_boundary_profiles),
            "backward_steps": list(self.backward_steps),
            "witness_variants": [
                variant.to_dict() for variant in self.witness_variants
            ],
        }


@dataclass(frozen=True, slots=True)
class GoalFirstGenerationResult:
    case: Case | None
    trace: dict[str, Any]


_GOALS: tuple[GenerationGoal, ...] = (
    GenerationGoal(
        goal_id="order_offset_aggregate",
        target_contract="observable_order_survives_limit_offset_and_aggregation",
        fault_models=("ordering_stability", "limit_pushdown", "offset_semantics"),
        interaction=("sort", "limit", "offset", "groupby"),
        required_types=("int", "str"),
        required_operation_groups=(("sort",), ("limit", "offset"), ("groupby", "aggregate")),
        preferred_boundary_profiles=("empty_singleton_transition", "duplicate_heavy_keys"),
        backward_steps=(
            "choose an aggregate whose input is a limited/offset ordered relation",
            "synthesize an observable sort key and deterministic tie breaker",
            "create nullable and duplicate rows around the offset boundary",
            "derive the source schema containing group, order and aggregate columns",
        ),
        builder=generate_nested_topk_offset_aggregate_case,
        witness_variants=witness_builder_variants("order_offset_aggregate"),
    ),
    GenerationGoal(
        goal_id="nullable_membership_join",
        target_contract="multi_key_membership_respects_null_guards_and_three_valued_logic",
        fault_models=("join_null_semantics", "three_valued_logic", "join_cardinality"),
        interaction=("filter", "anti_or_semi_join", "case_when", "groupby"),
        required_types=("int", "str", "bool"),
        required_operation_groups=(("filter",), ("semi_join", "anti_join"), ("groupby",)),
        preferred_boundary_profiles=("null_truth_table", "duplicate_heavy_keys"),
        backward_steps=(
            "choose membership semantics and grouped survivor observation",
            "derive compatible multi-key left/right schemas",
            "insert explicit non-null guards before membership",
            "construct matched, unmatched, partially-null and duplicate keys",
        ),
        builder=generate_multi_key_anti_join_null_guard_case,
        witness_variants=witness_builder_variants("nullable_membership_join"),
    ),
    GenerationGoal(
        goal_id="union_distinct_window",
        target_contract="bag_to_set_transition_preserves_partitioned_window_inputs",
        fault_models=("union_all_semantics", "duplicate_semantics", "ordering_stability"),
        interaction=("union_all", "distinct", "anti_join", "running_sum"),
        required_types=("int", "float", "str", "bool"),
        required_operation_groups=(("union_all",), ("distinct",), ("running_sum",)),
        preferred_boundary_profiles=("duplicate_heavy_keys", "signed_zero_nan_infinity"),
        backward_steps=(
            "choose a partitioned running result as the terminal observation",
            "derive stable partition/order keys and a nullable numeric source",
            "place membership and distinct transitions before the window",
            "construct union inputs with controlled duplicates and nulls",
        ),
        builder=generate_union_distinct_anti_running_sum_case,
        witness_variants=witness_builder_variants("union_distinct_window"),
    ),
    GenerationGoal(
        goal_id="large_integer_cast_window",
        target_contract="integer_text_cast_remains exact through membership_and_window_order",
        fault_models=("cast_semantics", "numeric_precision", "ordering_stability"),
        interaction=("cast", "semi_join", "case_when", "running_sum"),
        required_types=("int", "str"),
        required_operation_groups=(("mutate",), ("semi_join",), ("running_sum",)),
        preferred_boundary_profiles=("int64_overflow_edges",),
        backward_steps=(
            "choose an exact integer order key consumed by a partitioned window",
            "derive membership and range predicates over the cast result",
            "derive a string source column for the cast",
            "construct values around 2^53 and int64 boundaries",
        ),
        builder=generate_large_int_text_membership_window_case,
        witness_variants=witness_builder_variants("large_integer_cast_window"),
    ),
    GenerationGoal(
        goal_id="empty_union_groupby",
        target_contract="empty_to_nonempty_union_preserves_grouped_null_semantics",
        fault_models=("empty_input_handling", "union_all_semantics", "aggregation_null_semantics"),
        interaction=("filter_to_empty", "union_all", "groupby", "sort"),
        required_types=("int", "str"),
        required_operation_groups=(("filter",), ("union_all",), ("groupby",)),
        preferred_boundary_profiles=("empty_singleton_transition", "null_truth_table"),
        backward_steps=(
            "choose grouped output over a union",
            "derive one branch that becomes empty under a predicate",
            "derive a compatible non-empty append schema",
            "construct null and singleton groups around the cardinality transition",
        ),
        builder=generate_empty_then_union_groupby_case,
        witness_variants=witness_builder_variants("empty_union_groupby"),
    ),
    GenerationGoal(
        goal_id="nullable_grouped_topk",
        target_contract="null_aggregate_ordering_is_observed_by_topk",
        fault_models=("aggregation_null_semantics", "null_placement", "ordering_stability"),
        interaction=("groupby", "nullable_aggregate", "sort", "limit"),
        required_types=("int", "str"),
        required_operation_groups=(("groupby",), ("sort",), ("limit",)),
        preferred_boundary_profiles=("null_truth_table", "duplicate_heavy_keys"),
        backward_steps=(
            "choose top-k as the terminal order observation",
            "derive a nullable aggregate used as the leading sort key",
            "derive grouped key and aggregate input columns",
            "construct all-null, mixed-null and duplicate-heavy groups",
        ),
        builder=generate_null_groupby_topk_case,
        witness_variants=witness_builder_variants("nullable_grouped_topk"),
    ),
)


def generation_goals() -> tuple[GenerationGoal, ...]:
    return _GOALS


def generation_goal(goal_id: str) -> GenerationGoal:
    return next(goal for goal in _GOALS if goal.goal_id == goal_id)


def generation_goal_variants(
    goal: GenerationGoal | str,
    *,
    variant_family: str = "v2",
) -> tuple[WitnessBuilderVariant, ...]:
    resolved = generation_goal(goal) if isinstance(goal, str) else goal
    return witness_builder_variants(resolved.goal_id, family=variant_family)


def semantic_witness_epoch_key(seed: int) -> int:
    """Return the stable coarse-grained construction epoch for a case seed."""

    return int(seed) // SEMANTIC_WITNESS_EPOCH_SPAN


def semantic_witness_epoch_lane(seed: int) -> int:
    """Return the registered three-way variant lane for a raw case seed."""

    return semantic_witness_epoch_key(seed) % SEMANTIC_WITNESS_VARIANT_LANE_COUNT


def semantic_witness_data_pattern_lane(seed: int) -> int:
    """Return the decoupled bounded data-pattern lane for a raw case seed."""

    return (
        semantic_witness_epoch_key(seed) // SEMANTIC_WITNESS_VARIANT_LANE_COUNT
    ) % SEMANTIC_WITNESS_DATA_PATTERN_LANE_COUNT


def semantic_witness_epoch_phase(seed: int) -> int:
    """Return the combined variant/data phase in the bounded nine-phase cycle."""

    return semantic_witness_epoch_key(seed) % SEMANTIC_WITNESS_PHASE_COUNT


def semantic_witness_palette_construction_seed(
    data_pattern_lane: int,
    goal: GenerationGoal | str,
    variant_index: int,
    *,
    variant_family: str = "v2",
) -> int:
    """Return one stable builder seed per epoch/goal/variant cache cell."""

    resolved = generation_goal(goal) if isinstance(goal, str) else goal
    goal_index = _GOALS.index(resolved)
    lane_count = len(
        generation_goal_variants(resolved, variant_family=variant_family)
    )
    resolved_variant = int(variant_index)
    if resolved_variant < 0 or resolved_variant >= lane_count:
        raise IndexError(
            f"variant index {resolved_variant} outside [0, {lane_count})"
        )
    palette_size = len(_GOALS) * SEMANTIC_WITNESS_VARIANT_LANE_COUNT
    resolved_pattern_lane = int(data_pattern_lane)
    if (
        resolved_pattern_lane < 0
        or resolved_pattern_lane >= SEMANTIC_WITNESS_DATA_PATTERN_LANE_COUNT
    ):
        raise IndexError(
            "data pattern lane "
            f"{resolved_pattern_lane} outside "
            f"[0, {SEMANTIC_WITNESS_DATA_PATTERN_LANE_COUNT})"
        )
    return (
        resolved_pattern_lane * palette_size
        + goal_index * SEMANTIC_WITNESS_VARIANT_LANE_COUNT
        + resolved_variant
    )


def select_generation_variant(
    seed: int,
    goal: GenerationGoal | str,
    *,
    diversity_epoch_seed: int | None = None,
    variant_family: str = "v2",
) -> tuple[WitnessBuilderVariant, int, str]:
    resolved = generation_goal(goal) if isinstance(goal, str) else goal
    variants = generation_goal_variants(
        resolved,
        variant_family=variant_family,
    )
    if not variants:
        raise ValueError(f"goal {resolved.goal_id!r} has no witness builder variants")
    goal_index = _GOALS.index(resolved)
    # The quotient term breaks the correlation between ``seed % len(_GOALS)``
    # and variant selection.  Every goal therefore rotates through every
    # variant on the default seeded goal cycle.
    if diversity_epoch_seed is not None:
        epoch_lane = (
            int(diversity_epoch_seed) % SEMANTIC_WITNESS_VARIANT_LANE_COUNT
        )
        active_variant_indexes = (
            epoch_lane,
            (epoch_lane + 1) % SEMANTIC_WITNESS_VARIANT_LANE_COUNT,
        )
        palette_rotation = (
            int(seed)
            + int(seed) // len(_GOALS)
            + goal_index
        ) % len(active_variant_indexes)
        variant_index = active_variant_indexes[palette_rotation]
        selection_reason = "cache_aware_two_variant_palette_cycle_v4"
    else:
        variant_index = (
            int(seed) + int(seed) // len(_GOALS) + goal_index
        ) % len(variants)
        selection_reason = "seeded_goal_variant_cycle_v1"
    return variants[variant_index], variant_index, selection_reason


def select_generation_goal(seed: int, *, profile: str = "") -> tuple[GenerationGoal, str]:
    text = str(profile or "").lower()
    profile_hints = (
        (("offset", "topk", "ordered"), "order_offset_aggregate"),
        (("anti", "semi", "join_null", "membership"), "nullable_membership_join"),
        (("union", "running", "window"), "union_distinct_window"),
        (("cast", "large_int", "numeric_text"), "large_integer_cast_window"),
        (("empty",), "empty_union_groupby"),
        (("null_groupby", "null_agg"), "nullable_grouped_topk"),
    )
    for hints, goal_id in profile_hints:
        if any(hint in text for hint in hints):
            return generation_goal(goal_id), f"profile_hint:{profile}"
    return _GOALS[int(seed) % len(_GOALS)], "seeded_goal_cycle"


def generate_goal_first_case(
    seed: int,
    *,
    profile: str = "",
    schema_spec: Any | None = None,
    boundary_mode: str = "disabled",
    collect_activation_evidence: bool = True,
    require_semantic_activation: bool = False,
    diversity_preserving_witness: bool = False,
    diversity_epoch_seed: int | None = None,
    witness_variant_family: str = "v2",
) -> GoalFirstGenerationResult:
    goal, selection_reason = select_generation_goal(seed, profile=profile)
    selected_goal_variants = generation_goal_variants(
        goal,
        variant_family=witness_variant_family,
    )
    selected_variant: WitnessBuilderVariant | None = None
    variant_index = 0
    variant_reason = "legacy_single_builder"
    if diversity_preserving_witness:
        selected_variant, variant_index, variant_reason = select_generation_variant(
            int(seed),
            goal,
            diversity_epoch_seed=diversity_epoch_seed,
            variant_family=witness_variant_family,
        )
    construction_seed = int(seed)
    if (
        diversity_preserving_witness
        and diversity_epoch_seed is not None
        and selected_variant is not None
    ):
        construction_seed = semantic_witness_palette_construction_seed(
            (
                int(diversity_epoch_seed)
                // SEMANTIC_WITNESS_VARIANT_LANE_COUNT
            )
            % SEMANTIC_WITNESS_DATA_PATTERN_LANE_COUNT,
            goal,
            variant_index,
            variant_family=witness_variant_family,
        )
    goal_index = _GOALS.index(goal)
    palette_cell_index = (
        goal_index * SEMANTIC_WITNESS_VARIANT_LANE_COUNT + variant_index
        if selected_variant is not None
        else None
    )
    selected_goal_payload = goal.to_dict()
    selected_goal_payload["witness_variant_family"] = str(witness_variant_family)
    selected_goal_payload["witness_variants"] = [
        variant.to_dict() for variant in selected_goal_variants
    ]
    trace: dict[str, Any] = {
        "schema_version": GOAL_FIRST_SCHEMA_VERSION,
        "generation_mode": "goal_first",
        "selected_goal": selected_goal_payload,
        "selection_reason": selection_reason,
        "builder_variant": (
            {
                **selected_variant.to_dict(),
                "selection_index": variant_index,
                "selection_reason": variant_reason,
                "variant_family": str(witness_variant_family),
                "diversity_preserving": True,
                "construction_seed": construction_seed,
                "case_seed": int(seed),
                "epoch_reuse": construction_seed != int(seed),
            }
            if selected_variant is not None
            else {
                "variant_id": "legacy_single_builder",
                "selection_index": 0,
                "selection_reason": variant_reason,
                "variant_family": "legacy",
                "diversity_preserving": False,
                "construction_seed": int(seed),
                "case_seed": int(seed),
                "epoch_reuse": False,
            }
        ),
        "requested_profile": str(profile or ""),
        "requested_lhs_schema": (
            schema_spec.to_dict() if hasattr(schema_spec, "to_dict") else None
        ),
        "schema_resolution": "goal_preconditions_override_forward_lhs_shape",
        "constructible": False,
        "valid": False,
        "skip_reason": "",
        "boundary_application": {},
        "activation_requirement": {
            "required": bool(require_semantic_activation),
            "policy": (
                "deterministic_root_guided_static_witness"
                if diversity_preserving_witness and witness_variant_family == "v3"
                else "deterministic_diversity_preserving_static_witness"
                if diversity_preserving_witness
                else "deterministic_static_witness"
            ),
        },
        "diversity_epoch": {
            "enabled": bool(
                diversity_preserving_witness and diversity_epoch_seed is not None
            ),
            "epoch_key": (
                int(diversity_epoch_seed)
                if diversity_preserving_witness and diversity_epoch_seed is not None
                else None
            ),
            "seed_span": SEMANTIC_WITNESS_EPOCH_SPAN,
            "epoch_lane": (
                int(diversity_epoch_seed) % SEMANTIC_WITNESS_VARIANT_LANE_COUNT
                if diversity_preserving_witness and diversity_epoch_seed is not None
                else None
            ),
            "data_pattern_lane": (
                (
                    int(diversity_epoch_seed)
                    // SEMANTIC_WITNESS_VARIANT_LANE_COUNT
                )
                % SEMANTIC_WITNESS_DATA_PATTERN_LANE_COUNT
                if diversity_preserving_witness and diversity_epoch_seed is not None
                else None
            ),
            "palette_cell_index": (
                palette_cell_index
                if diversity_preserving_witness and diversity_epoch_seed is not None
                else None
            ),
            "palette_size": (
                len(_GOALS) * SEMANTIC_WITNESS_ACTIVE_VARIANTS_PER_EPOCH
                if diversity_preserving_witness and diversity_epoch_seed is not None
                else 0
            ),
            "registered_palette_size": (
                len(_GOALS) * SEMANTIC_WITNESS_VARIANT_LANE_COUNT
                if diversity_preserving_witness and diversity_epoch_seed is not None
                else 0
            ),
            "active_variant_indexes": (
                [
                    int(diversity_epoch_seed)
                    % SEMANTIC_WITNESS_VARIANT_LANE_COUNT,
                    (
                        int(diversity_epoch_seed)
                        % SEMANTIC_WITNESS_VARIANT_LANE_COUNT
                        + 1
                    )
                    % SEMANTIC_WITNESS_VARIANT_LANE_COUNT,
                ]
                if diversity_preserving_witness and diversity_epoch_seed is not None
                else []
            ),
        },
    }
    try:
        builder = selected_variant.builder if selected_variant is not None else goal.builder
        case = builder(construction_seed)
    except Exception as exc:  # noqa: BLE001 - explicit generation skip provenance
        trace["skip_reason"] = f"builder_error:{type(exc).__name__}:{exc}"
        return GoalFirstGenerationResult(case=None, trace=trace)
    if selected_variant is not None:
        data_pattern = (
            case.metadata.get("semantic_witness_data_pattern", {})
            if isinstance(case.metadata, dict)
            else {}
        )
        if isinstance(data_pattern, dict):
            trace["builder_variant"]["data_pattern"] = dict(data_pattern)
    validation = _validate_goal_case(case, goal)
    trace["precondition_validation"] = validation
    trace["constructible"] = bool(validation["valid"])
    if not validation["valid"]:
        trace["skip_reason"] = "unsatisfied_backward_preconditions"
        return GoalFirstGenerationResult(case=None, trace=trace)
    if boundary_mode == "fault_model_targeted":
        if require_semantic_activation:
            application, boundary_trials = _apply_activation_aware_boundary_profile(
                case,
                goal,
                seed=construction_seed,
                diversity_tie_break=diversity_preserving_witness,
                variant_index=variant_index,
            )
            trace["boundary_activation_trials"] = boundary_trials
        else:
            application = apply_targeted_boundary_profile(
                case,
                seed=construction_seed,
                fault_models=goal.fault_models,
                preferred_profile_ids=goal.preferred_boundary_profiles,
            )
        case = application.case
        trace["boundary_application"] = application.trace
    if selected_variant is not None and construction_seed != int(seed):
        case.seed = int(seed)
        case.program.seed = int(seed)
        case.case_id = f"case-{int(seed):08d}-{selected_variant.variant_id}"
        case.program.program_id = (
            f"prog-{int(seed):08d}-{selected_variant.variant_id}"
        )
    case.case_id = f"{case.case_id}-goal-first"
    case.metadata = dict(case.metadata or {})
    case.metadata["goal_first_generation"] = trace
    case.metadata["generation_mode"] = "goal_first"
    case.metadata["goal_id"] = goal.goal_id
    case.metadata["goal_fault_models"] = list(goal.fault_models)
    case.metadata["goal_builder_variant"] = dict(trace["builder_variant"])
    case.metadata["requested_generator_profile"] = str(profile or "")
    trace["valid"] = True
    activation: dict[str, Any] = {}
    if collect_activation_evidence or require_semantic_activation:
        activation = _attach_semantic_activation(case, goal, trace=trace)
    if require_semantic_activation:
        repair = _repair_semantic_activation(case, goal, activation)
        trace["activation_repair"] = repair
        witness_mode = (
            "goal_first_witness_v3"
            if diversity_preserving_witness and witness_variant_family == "v3"
            else "goal_first_witness_v2"
            if diversity_preserving_witness
            else "goal_first_witness"
        )
        trace["generation_mode"] = witness_mode
        case.metadata["generation_mode"] = witness_mode
        case.case_id = (
            f"{case.case_id}-witness-v3"
            if diversity_preserving_witness and witness_variant_family == "v3"
            else f"{case.case_id}-witness-v2"
            if diversity_preserving_witness
            else f"{case.case_id}-witness"
        )
        if repair["changed"]:
            activation = _attach_semantic_activation(case, goal, trace=trace)
        repair["after_status"] = str(
            activation.get("evaluation_status", "not_evaluated") or "not_evaluated"
        )
        repair["after_missing_tokens"] = list(
            activation.get("missing_tokens", []) or []
        )
        repair["satisfied"] = repair["after_status"] == "activated"
    return GoalFirstGenerationResult(case=case, trace=trace)


def case_reaches_generation_goal(case: Case, goal: GenerationGoal | str) -> bool:
    resolved = generation_goal(goal) if isinstance(goal, str) else goal
    return bool(_validate_goal_case(case, resolved)["valid"])


def refresh_goal_semantic_activation(case: Case) -> dict[str, Any]:
    """Recompute evidence after preflight/mutation while preserving old cases."""

    metadata = case.metadata if isinstance(case.metadata, dict) else {}
    goal_id = str(metadata.get("goal_id", "") or "")
    goal_trace = metadata.get("goal_first_generation", {})
    if not goal_id and isinstance(goal_trace, dict):
        selected_goal = goal_trace.get("selected_goal", {})
        if isinstance(selected_goal, dict):
            goal_id = str(selected_goal.get("goal_id", "") or "")
    if not goal_id:
        return {}
    try:
        goal = generation_goal(goal_id)
    except StopIteration:
        stored_activation = metadata.get("semantic_activation", {})
        syntactic_reached = bool(
            metadata.get("semantic_activation_syntactic_reached", False)
            or (
                stored_activation.get("syntactic_reached", False)
                if isinstance(stored_activation, dict)
                else False
            )
        )
        activation = evaluate_semantic_activation(
            case,
            goal_id=goal_id,
            syntactic_reached=syntactic_reached,
        ).to_dict()
        case.metadata = dict(metadata)
        case.metadata["semantic_activation"] = activation
        return activation
    return _attach_semantic_activation(
        case,
        goal,
        trace=goal_trace if isinstance(goal_trace, dict) else None,
    )


def _attach_semantic_activation(
    case: Case,
    goal: GenerationGoal,
    *,
    trace: dict[str, Any] | None,
) -> dict[str, Any]:
    validation = _validate_goal_case(case, goal)
    activation = evaluate_semantic_activation(
        case,
        goal_id=goal.goal_id,
        syntactic_reached=bool(validation["valid"]),
    ).to_dict()
    case.metadata = dict(case.metadata or {})
    case.metadata["semantic_activation"] = activation
    if trace is not None:
        trace["semantic_activation"] = activation
    stored_trace = case.metadata.get("goal_first_generation")
    if isinstance(stored_trace, dict):
        stored_trace["semantic_activation"] = activation
    return activation


def _repair_semantic_activation(
    case: Case,
    goal: GenerationGoal,
    activation: dict[str, Any],
) -> dict[str, Any]:
    before_status = str(
        activation.get("evaluation_status", "not_evaluated") or "not_evaluated"
    )
    missing = tuple(str(item) for item in activation.get("missing_tokens", []) or [])
    repair = {
        "schema_version": ACTIVATION_REPAIR_SCHEMA_VERSION,
        "goal_id": goal.goal_id,
        "attempted": before_status != "activated",
        "changed": False,
        "repair_id": "",
        "before_status": before_status,
        "before_missing_tokens": list(missing),
        "after_status": before_status,
        "after_missing_tokens": list(missing),
        "satisfied": before_status == "activated",
        "reason": "already_activated" if before_status == "activated" else "",
    }
    if before_status == "activated":
        return repair
    repair_fn = _ACTIVATION_REPAIRS.get(goal.goal_id)
    if repair_fn is None:
        repair["reason"] = "no_registered_repair"
        return repair
    repair_id, changed, reason = repair_fn(case, missing)
    repair["repair_id"] = repair_id
    repair["changed"] = bool(changed)
    repair["reason"] = reason
    return repair


def _repair_order_offset_aggregate(
    case: Case,
    missing: tuple[str, ...],
) -> tuple[str, bool, str]:
    repair_id = "positive-offset-v1"
    if "positive_offset" not in missing:
        return repair_id, False, "target_witness_not_missing"
    for operation in case.program.operations:
        if op_kind(operation) != "offset":
            continue
        try:
            current = int(operation.get("n", 0) or 0)
        except (TypeError, ValueError):
            current = 0
        if current > 0:
            return repair_id, False, "offset_already_positive"
        operation["n"] = 1
        return repair_id, True, "set_zero_offset_to_one"
    return repair_id, False, "offset_operation_missing"


def _repair_union_distinct_window(
    case: Case,
    missing: tuple[str, ...],
) -> tuple[str, bool, str]:
    repair_id = "duplicate-union-row-v1"
    union = next(
        (operation for operation in case.program.operations if op_kind(operation) == "union_all"),
        None,
    )
    if union is None or not case.tables or not case.tables[0].rows:
        return repair_id, False, "union_source_unavailable"
    append_name = str(union.get("table", "") or "")
    append = next((table for table in case.tables if table.name == append_name), None)
    if append is None:
        return repair_id, False, "union_append_table_missing"
    changes: list[str] = []
    if "duplicate_bag_to_set_transition" in missing:
        append.rows.append(copy.deepcopy(case.tables[0].rows[0]))
        changes.append("exact_cross_branch_duplicate")
    if {
        "nullable_window_input",
        "multirow_window_partition",
    }.intersection(missing):
        sample_ids = [
            row.get("sample_id")
            for table in case.tables
            for row in table.rows
            if isinstance(row.get("sample_id"), int)
        ]
        next_id = max(sample_ids, default=0) + 1
        for offset, x_value in enumerate((None, 1.0)):
            row = {column.name: None for column in append.columns}
            row.update(
                {
                    "sample_id": next_id + offset,
                    "grp": "activation_survivor",
                    "seq": 100 + offset,
                    "x": x_value,
                    "flag": offset == 1,
                }
            )
            append.rows.append(row)
        changes.append("nullable_multirow_window_partition")
    if not changes:
        return repair_id, False, "target_witness_not_missing"
    return repair_id, True, "injected_" + "+".join(changes)


def _repair_large_integer_cast_window(
    case: Case,
    missing: tuple[str, ...],
) -> tuple[str, bool, str]:
    repair_id = "multirow-exact-membership-partition-v1"
    if "multirow_window_partition" not in missing:
        return repair_id, False, "target_witness_not_missing"
    if len(case.tables) < 2:
        return repair_id, False, "membership_table_missing"
    left = case.tables[0]
    membership = next(
        (table for table in case.tables if table.name == "t_large_int_member"),
        case.tables[1],
    )
    candidates: list[int] = []
    for row in left.rows:
        if row.get("acct") != "a" or row.get("num_s") is None:
            continue
        try:
            candidates.append(int(str(row["num_s"])))
        except ValueError:
            continue
    candidates = list(dict.fromkeys(candidates))
    if len(candidates) < 2:
        return repair_id, False, "same_partition_cast_values_unavailable"
    existing = {row.get("num_value") for row in membership.rows}
    added = 0
    for value in candidates[:2]:
        if value in existing:
            continue
        membership.rows.append({"num_value": value})
        existing.add(value)
        added += 1
    if sum(value in existing for value in candidates[:2]) < 2:
        return repair_id, False, "membership_values_not_materialized"
    return repair_id, added > 0, "added_exact_values_for_multirow_partition"


def _repair_nullable_grouped_topk(
    case: Case,
    missing: tuple[str, ...],
) -> tuple[str, bool, str]:
    repair_id = "nullable-aggregate-topk-v1"
    target_tokens = {
        "nullable_aggregate",
        "aggregate_drives_sort",
        "null_and_non_null_aggregate_groups",
        "active_topk_cut",
    }
    if not target_tokens.intersection(missing):
        return repair_id, False, "target_witness_not_missing"
    if not case.tables:
        return repair_id, False, "primary_table_missing"
    group_index = next(
        (
            index
            for index, operation in enumerate(case.program.operations)
            if op_kind(operation) == "groupby"
        ),
        None,
    )
    if group_index is None:
        return repair_id, False, "groupby_operation_missing"
    sort_index = next(
        (
            index
            for index, operation in enumerate(case.program.operations)
            if index > group_index and op_kind(operation) == "sort"
        ),
        None,
    )
    limit_index = next(
        (
            index
            for index, operation in enumerate(case.program.operations)
            if sort_index is not None
            and index > sort_index
            and op_kind(operation) == "limit"
        ),
        None,
    )
    if sort_index is None or limit_index is None:
        return repair_id, False, "topk_pipeline_missing"
    table = case.tables[0]
    rows = table.rows
    if not rows:
        rows.append({column.name: None for column in table.columns})
    rows[0]["s"] = "activation_all_null"
    rows[0]["x"] = None
    if len(rows) < 2:
        rows.append({column.name: None for column in table.columns})
    rows[1]["s"] = "activation_non_null"
    rows[1]["x"] = 1

    group = case.program.operations[group_index]
    group["keys"] = ["s"]
    group["aggs"] = [{"column": "x", "func": "min", "as": "min_x"}]
    select = next(
        (
            operation
            for index, operation in enumerate(case.program.operations)
            if group_index < index < sort_index and op_kind(operation) == "select"
        ),
        None,
    )
    if select is not None:
        select["columns"] = ["s", "min_x"]
    sort = case.program.operations[sort_index]
    sort.pop("columns", None)
    sort.pop("ascending", None)
    sort["keys"] = [
        {"column": "min_x", "ascending": True, "nulls": "first"},
        {"column": "s", "ascending": True, "nulls": "last"},
    ]
    group_count = len({row.get("s") for row in rows})
    case.program.operations[limit_index]["n"] = max(1, group_count - 1)
    return repair_id, True, "rewired_nullable_aggregate_as_active_topk_key"


_ACTIVATION_REPAIRS: dict[
    str,
    Callable[[Case, tuple[str, ...]], tuple[str, bool, str]],
] = {
    "order_offset_aggregate": _repair_order_offset_aggregate,
    "union_distinct_window": _repair_union_distinct_window,
    "large_integer_cast_window": _repair_large_integer_cast_window,
    "nullable_grouped_topk": _repair_nullable_grouped_topk,
}


def _apply_activation_aware_boundary_profile(
    case: Case,
    goal: GenerationGoal,
    *,
    seed: int,
    diversity_tie_break: bool = False,
    variant_index: int = 0,
) -> tuple[Any, list[dict[str, Any]]]:
    """Choose among declared boundary profiles using only static witness quality."""

    applications: list[tuple[tuple[int, int, int], int, Any]] = []
    trials: list[dict[str, Any]] = []
    profile_ids = goal.preferred_boundary_profiles or ("",)
    for order, profile_id in enumerate(profile_ids):
        application = apply_targeted_boundary_profile(
            case,
            seed=seed,
            fault_models=goal.fault_models,
            preferred_profile_ids=(profile_id,) if profile_id else (),
            _trial_copy=True,
        )
        validation = _validate_goal_case(application.case, goal)
        activation = evaluate_semantic_activation(
            application.case,
            goal_id=goal.goal_id,
            syntactic_reached=bool(validation["valid"]),
        ).to_dict()
        quality = (
            int(activation["evaluation_status"] == "activated"),
            len(activation.get("activation_tokens", []) or []),
            -len(activation.get("missing_tokens", []) or []),
        )
        score = (*quality, -order)
        applications.append((quality, order, application))
        trials.append(
            {
                "profile_id": str(application.trace.get("profile_id", "") or ""),
                "evaluation_status": activation["evaluation_status"],
                "activation_token_count": len(
                    activation.get("activation_tokens", []) or []
                ),
                "missing_tokens": list(activation.get("missing_tokens", []) or []),
                "score": list(score),
                "quality": list(quality),
            }
        )
    if diversity_tie_break:
        best_quality = max(item[0] for item in applications)
        tied = [item for item in applications if item[0] == best_quality]
        rotation = (
            int(seed) + int(seed) // len(_GOALS) + int(variant_index)
        ) % len(tied)
        selected = tied[rotation][2]
        selection_policy = "activation_quality_then_seeded_tie_rotation_v1"
    else:
        applications.sort(key=lambda item: (*item[0], -item[1]), reverse=True)
        tied = [applications[0]]
        rotation = 0
        selected = applications[0][2]
        selection_policy = "activation_quality_then_declared_profile_order_v1"
    selected_profile = str(selected.trace.get("profile_id", "") or "")
    for trial in trials:
        trial["selected"] = trial["profile_id"] == selected_profile
        trial["selection_policy"] = selection_policy
        trial["best_quality_tie_count"] = len(tied)
        trial["tie_rotation_index"] = rotation
    return selected, trials


def _validate_goal_case(case: Case, goal: GenerationGoal) -> dict[str, Any]:
    logical_types = {column.type for table in case.tables for column in table.columns}
    operations = [op_kind(operation) for operation in case.program.operations]
    missing_types = sorted(set(goal.required_types) - logical_types)
    missing_groups = [
        list(group)
        for group in goal.required_operation_groups
        if not set(group).intersection(operations)
    ]
    return {
        "valid": not missing_types and not missing_groups,
        "logical_types": sorted(logical_types),
        "operation_sequence": operations,
        "missing_types": missing_types,
        "missing_operation_groups": missing_groups,
    }
