from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from datadiff.canonicalization import short_canonical_hash


COMPLETE_BUILDER_ORDER = "complete_builder_order"
CCS_RISK_PRIORITY = "ccs_risk_priority"
CCS_RISK_PRIORITY_V2 = "ccs_risk_priority_v2"
OBLIGATION_PRIORITY_MODES = frozenset(
    {COMPLETE_BUILDER_ORDER, CCS_RISK_PRIORITY, CCS_RISK_PRIORITY_V2}
)
OBLIGATION_PRIORITY_SCHEMA_VERSION = "ccs-obligation-priority-v2"

FAULT_MODEL_WEIGHTS: dict[str, float] = {
    "null_semantics": 4.0,
    "join_null_semantics": 4.0,
    "aggregation_null_semantics": 4.0,
    "numeric_precision": 4.0,
    "cast_boundary": 4.0,
    "dtype_lowering": 4.0,
    "ordering_stability": 4.0,
    "null_placement": 4.0,
    "string_semantics": 4.0,
    "partitioning": 4.0,
    "union_all_semantics": 4.0,
    "offset_semantics": 4.0,
    "limit_pushdown": 4.0,
    "join_cardinality": 3.0,
    "input_materialization": 3.0,
    "filter_pushdown": 3.0,
    "predicate_semantics": 3.0,
    "aggregation_strategy": 3.0,
    "aggregation_projection": 3.0,
    "schema_ordering": 3.0,
    "empty_input_handling": 3.0,
    "limit_overflow": 3.0,
    "input_normalization": 3.0,
    "optimizer_equivalence": 2.0,
    "operation_reordering": 2.0,
    "predicate_simplification": 2.0,
    "input_cardinality": 2.0,
    "bag_semantics": 2.0,
    "join_ordering": 2.0,
}
SCOPE_WEIGHTS: dict[str, float] = {
    "node_contract": 1.0,
    "input_relation": 1.5,
    "operation_sequence": 2.0,
    "program_tail": 1.75,
    "whole_program": 2.0,
}
DEFAULT_FAULT_MODEL_WEIGHT = 1.0
OPERATION_PREFIX_FAULT_MODEL_WEIGHT = 0.5
ESTIMATED_COST_PENALTY = 0.75

# Frozen from the development/canonical corpus before the P5 final holdout.
# These are semantic/optimizer-risk features, not current-case outcomes.
V2_INTERACTION_WEIGHTS: dict[str, float] = {
    "order_limit_offset": 4.5,
    "null_aggregation_order": 4.0,
    "join_predicate_limit": 3.75,
    "cast_membership_window": 3.5,
    "set_bag_window": 3.25,
    "nullable_membership": 3.0,
    "numeric_window": 2.75,
    "layout_sensitive": 2.25,
    "empty_union_aggregate": 2.0,
    "multi_operation": 1.0,
}
V2_PLAN_TRANSITION_WEIGHTS: dict[str, float] = {
    "sort_to_limit_or_offset": 3.0,
    "limit_or_offset_to_aggregate": 3.0,
    "aggregate_to_sort": 2.5,
    "filter_to_join": 2.5,
    "join_to_filter_or_limit": 2.25,
    "distinct_to_order": 2.0,
    "cast_to_membership": 2.0,
    "setop_to_distinct": 1.75,
}
V2_HISTORICAL_ROOT_SUPPORT: dict[str, float] = {
    "ordering_stability": 3.0,
    "null_placement": 2.75,
    "limit_pushdown": 2.75,
    "offset_semantics": 2.5,
    "aggregation_null_semantics": 2.5,
    "numeric_precision": 2.25,
    "join_cardinality": 2.0,
    "filter_pushdown": 2.0,
    "join_null_semantics": 1.75,
    "duplicate_semantics": 1.5,
    "union_all_semantics": 1.5,
}
V2_COST_NORMALIZATION = 0.65
V2_SCOPE_SCALE = 0.5


@dataclass(frozen=True, slots=True)
class ObligationPriorityInput:
    obligation_id: str
    target_fault_models: tuple[str, ...]
    estimated_cost: float
    scopes: tuple[str, ...]
    anchor_node_count: int
    source_relation_count: int
    interaction_features: tuple[str, ...] = ()
    plan_transition_features: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ObligationPriorityScore:
    obligation_id: str
    rank: int
    total_score: float
    fault_model_score: float
    scope_score: float
    breadth_score: float
    cost_penalty: float
    tie_breaker: str
    interaction_score: float = 0.0
    plan_transition_score: float = 0.0
    historical_support_score: float = 0.0
    cost_denominator: float = 1.0
    feature_explanation: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "obligation_id": self.obligation_id,
            "rank": self.rank,
            "total_score": self.total_score,
            "fault_model_score": self.fault_model_score,
            "scope_score": self.scope_score,
            "breadth_score": self.breadth_score,
            "cost_penalty": self.cost_penalty,
            "tie_breaker": self.tie_breaker,
            "interaction_score": self.interaction_score,
            "plan_transition_score": self.plan_transition_score,
            "historical_support_score": self.historical_support_score,
            "cost_denominator": self.cost_denominator,
            "feature_explanation": list(self.feature_explanation),
        }


@dataclass(frozen=True, slots=True)
class ObligationPriorityPlan:
    mode: str
    semantic_digest: str
    relation_order: tuple[str, ...]
    scores: tuple[ObligationPriorityScore, ...]
    digest: str
    policy: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": OBLIGATION_PRIORITY_SCHEMA_VERSION,
            "mode": self.mode,
            "semantic_digest": self.semantic_digest,
            "relation_order": list(self.relation_order),
            "scores": [score.to_dict() for score in self.scores],
            "digest": self.digest,
        }
        if self.policy:
            payload["policy"] = dict(self.policy)
        return payload


def build_obligation_priority_plan(
    inputs: Sequence[ObligationPriorityInput],
    *,
    mode: str,
    complete_builder_order: Iterable[str],
    semantic_digest: str,
    preferred_order: Iterable[str] = (),
) -> ObligationPriorityPlan:
    normalized_mode = str(mode or COMPLETE_BUILDER_ORDER)
    if normalized_mode not in OBLIGATION_PRIORITY_MODES:
        raise ValueError(
            f"unsupported obligation priority mode: {normalized_mode}"
        )
    by_id = {item.obligation_id: item for item in inputs}
    preferred = _eligible_unique(preferred_order, by_id)
    complete = _eligible_unique(complete_builder_order, by_id)
    remaining = [
        obligation_id
        for obligation_id in by_id
        if obligation_id not in preferred and obligation_id not in complete
    ]
    complete_order = [
        *preferred,
        *(item for item in complete if item not in preferred),
        *remaining,
    ]
    score_fn = _score_v2 if normalized_mode == CCS_RISK_PRIORITY_V2 else _score
    raw_scores = {
        obligation_id: score_fn(item, semantic_digest=semantic_digest)
        for obligation_id, item in by_id.items()
    }
    if normalized_mode == COMPLETE_BUILDER_ORDER:
        relation_order = complete_order
    else:
        relation_order = [
            *preferred,
            *sorted(
                (
                    obligation_id
                    for obligation_id in by_id
                    if obligation_id not in preferred
                ),
                key=lambda obligation_id: (
                    -raw_scores[obligation_id]["total_score"],
                    raw_scores[obligation_id]["tie_breaker"],
                ),
            ),
        ]
    scores = tuple(
        ObligationPriorityScore(
            obligation_id=obligation_id,
            rank=rank,
            **raw_scores[obligation_id],
        )
        for rank, obligation_id in enumerate(relation_order)
    )
    policy = _priority_policy(normalized_mode)
    digest_payload = {
        "schema_version": OBLIGATION_PRIORITY_SCHEMA_VERSION,
        "mode": normalized_mode,
        "semantic_digest": semantic_digest,
        "relation_order": relation_order,
        "scores": [score.to_dict() for score in scores],
    }
    if policy:
        digest_payload["policy"] = policy
    return ObligationPriorityPlan(
        mode=normalized_mode,
        semantic_digest=semantic_digest,
        relation_order=tuple(relation_order),
        scores=scores,
        digest=f"obligation-priority-{short_canonical_hash(digest_payload, 64)}",
        policy=policy,
    )


def _score(
    item: ObligationPriorityInput,
    *,
    semantic_digest: str,
) -> dict[str, float | str]:
    fault_model_score = sum(
        _fault_model_weight(model)
        for model in dict.fromkeys(item.target_fault_models)
    )
    scope_score = max(
        (SCOPE_WEIGHTS.get(scope, 0.0) for scope in item.scopes),
        default=0.0,
    )
    breadth_score = min(
        1.0,
        0.25 * max(0, item.source_relation_count - 1)
        + 0.10 * max(0, item.anchor_node_count),
    )
    cost_penalty = ESTIMATED_COST_PENALTY * max(
        0.0, float(item.estimated_cost)
    )
    total_score = round(
        fault_model_score + scope_score + breadth_score - cost_penalty,
        6,
    )
    return {
        "total_score": total_score,
        "fault_model_score": fault_model_score,
        "scope_score": scope_score,
        "breadth_score": breadth_score,
        "cost_penalty": cost_penalty,
        "tie_breaker": short_canonical_hash(
            {
                "semantic_digest": semantic_digest,
                "obligation_id": item.obligation_id,
            },
            16,
        ),
        "interaction_score": 0.0,
        "plan_transition_score": 0.0,
        "historical_support_score": 0.0,
        "cost_denominator": 1.0,
        "feature_explanation": (),
    }


def _score_v2(
    item: ObligationPriorityInput,
    *,
    semantic_digest: str,
) -> dict[str, Any]:
    fault_model_score = 0.25 * sum(
        _fault_model_weight(model)
        for model in dict.fromkeys(item.target_fault_models)
    )
    interaction_score = sum(
        V2_INTERACTION_WEIGHTS.get(feature, 0.0)
        for feature in dict.fromkeys(item.interaction_features)
    )
    plan_transition_score = sum(
        V2_PLAN_TRANSITION_WEIGHTS.get(feature, 0.0)
        for feature in dict.fromkeys(item.plan_transition_features)
    )
    historical_support_score = sum(
        V2_HISTORICAL_ROOT_SUPPORT.get(model, 0.0)
        for model in dict.fromkeys(item.target_fault_models)
    )
    scope_score = V2_SCOPE_SCALE * max(
        (SCOPE_WEIGHTS.get(scope, 0.0) for scope in item.scopes),
        default=0.0,
    )
    breadth_score = min(
        1.5,
        0.35 * max(0, item.source_relation_count - 1)
        + 0.20 * max(0, item.anchor_node_count),
    )
    raw_risk = (
        fault_model_score
        + interaction_score
        + plan_transition_score
        + historical_support_score
        + scope_score
        + breadth_score
    )
    cost_denominator = 1.0 + V2_COST_NORMALIZATION * max(
        0.0, float(item.estimated_cost)
    )
    total_score = round(raw_risk / cost_denominator, 6)
    cost_penalty = round(raw_risk - total_score, 6)
    explanation = tuple(
        [
            *(f"interaction:{feature}={V2_INTERACTION_WEIGHTS.get(feature, 0.0)}" for feature in sorted(set(item.interaction_features))),
            *(f"plan_transition:{feature}={V2_PLAN_TRANSITION_WEIGHTS.get(feature, 0.0)}" for feature in sorted(set(item.plan_transition_features))),
            *(f"historical:{model}={V2_HISTORICAL_ROOT_SUPPORT.get(model, 0.0)}" for model in sorted(set(item.target_fault_models)) if model in V2_HISTORICAL_ROOT_SUPPORT),
            f"cost_denominator={round(cost_denominator, 6)}",
        ]
    )
    return {
        "total_score": total_score,
        "fault_model_score": round(fault_model_score, 6),
        "scope_score": round(scope_score, 6),
        "breadth_score": round(breadth_score, 6),
        "cost_penalty": cost_penalty,
        "tie_breaker": short_canonical_hash(
            {
                "semantic_digest": semantic_digest,
                "obligation_id": item.obligation_id,
                "mode": CCS_RISK_PRIORITY_V2,
            },
            16,
        ),
        "interaction_score": round(interaction_score, 6),
        "plan_transition_score": round(plan_transition_score, 6),
        "historical_support_score": round(historical_support_score, 6),
        "cost_denominator": round(cost_denominator, 6),
        "feature_explanation": explanation,
    }


def _priority_policy(mode: str) -> dict[str, Any] | None:
    if mode != CCS_RISK_PRIORITY_V2:
        return None
    coefficients = {
        "interaction": V2_INTERACTION_WEIGHTS,
        "plan_transition": V2_PLAN_TRANSITION_WEIGHTS,
        "historical_root_support": V2_HISTORICAL_ROOT_SUPPORT,
        "cost_normalization": V2_COST_NORMALIZATION,
        "scope_scale": V2_SCOPE_SCALE,
    }
    return {
        "policy_version": "priority-v2-development-freeze-1",
        "coefficient_digest": short_canonical_hash(coefficients, 64),
        "design_source": "canonical_confirmed_roots_and_development_corpus_only",
        "uses_current_full_outcome": False,
        "uses_final_holdout_outcome": False,
        "cost_normalized": True,
    }


def _fault_model_weight(fault_model: str) -> float:
    normalized = str(fault_model)
    if normalized.startswith("operation:"):
        return OPERATION_PREFIX_FAULT_MODEL_WEIGHT
    return FAULT_MODEL_WEIGHTS.get(
        normalized,
        DEFAULT_FAULT_MODEL_WEIGHT,
    )


def _eligible_unique(
    values: Iterable[str],
    allowed: Mapping[str, ObligationPriorityInput],
) -> list[str]:
    return [
        value
        for value in dict.fromkeys(str(item) for item in values if str(item))
        if value in allowed
    ]
