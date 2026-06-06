from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from datadiff.bandit_selection import (
    _record_adaptive_action_feedback,
    _record_backend_pair_feedback,
    _record_generator_profile_feedback,
)
from datadiff.config import ExperimentConfig
from datadiff.dsl import Case
from datadiff.feedback_policy import (
    FeedbackStorageContext,
    FeedbackStoragePolicy,
    feedback_storage_decision,
    source_scheduler_snapshot,
)
from datadiff.feedback_signals import (
    build_feedback_discovery_signals,
    feedback_source_new_behavior,
)
from datadiff.reward import (
    feedback_summary_for_case,
    source_reward_adjustment_from_summary,
)
from datadiff.run_metadata import _feedback_target_keys


@dataclass(frozen=True, slots=True)
class FeedbackUpdateResult:
    finding_outcomes: Any | None
    elapsed_ms: float = 0.0


def apply_feedback_updates(
    *,
    row: dict[str, Any],
    case: Case,
    feedback: Any,
    config: ExperimentConfig,
    selected_meta: dict[str, Any],
    guidance_row: dict[str, Any],
    preflight_row: dict[str, Any],
    behavior_signature: str,
    signal_signature: str,
    discovery_signature: str,
    version_id: str,
    generator_profile_context_features: tuple[str, ...],
    target_capabilities: list[str] | tuple[str, ...],
) -> FeedbackUpdateResult:
    known_families = config.known_saturated_bug_families
    discovery_signals = build_feedback_discovery_signals(
        row,
        known_saturated_bug_families=known_families,
    )
    finding_outcomes = discovery_signals.finding_outcomes
    if feedback is None:
        _apply_feedback_disabled_row(
            row,
            config=config,
            finding_outcomes=finding_outcomes,
        )
        return FeedbackUpdateResult(finding_outcomes=finding_outcomes, elapsed_ms=0.0)

    started = time.perf_counter()
    reward_signals = discovery_signals.reward_signals
    row_candidate_families = discovery_signals.candidate_bug_families
    row_candidate_signatures = discovery_signals.candidate_bug_signatures
    feedback_eligible, feedback_skip_reason = _feedback_storage_decision(
        case,
        candidate_source=selected_meta["source"],
        seed_lineage=selected_meta["seed_lineage"],
        candidate_bug_families=row_candidate_families,
        allow_feedback_mutation_child_candidate_bug=True,
    )
    if feedback_eligible and discovery_signals.resolved_semantic_only:
        feedback_eligible = False
        feedback_skip_reason = "resolved_semantic_divergence"
    row["feedback_eligible"] = feedback_eligible
    row["feedback_skip_reason"] = feedback_skip_reason
    feedback_summary = feedback_summary_for_case(
        row,
        known_saturated_bug_families=known_families,
        finding_outcomes=finding_outcomes,
    )
    if feedback_eligible:
        row_target_keys = _feedback_target_keys(
            guidance_row,
            selected_meta["operation_combo"],
            target_capabilities=target_capabilities,
        )
        row["stored_in_feedback_corpus"] = feedback.record(
            case,
            behavior_signature,
            discovery_signals.feedback_finding,
            novelty_signature=signal_signature,
            discovery_signature=discovery_signature,
            candidate_bug_families=row_candidate_families,
            target_keys=row_target_keys,
            disagreement_descriptor=row.get("disagreement_descriptor"),
            case_fingerprint=row.get("case_fingerprint"),
            schedule_delta=float(feedback_summary.get("seed_schedule_delta", 0.0) or 0.0),
        )
        row["feedback_corpus_persisted"] = feedback.last_persisted_to_disk
        row["feedback_record_skip_reason"] = feedback.last_record_skip_reason
    else:
        feedback.last_persisted_to_disk = False
        row["stored_in_feedback_corpus"] = False
        row["feedback_corpus_persisted"] = False
        row["feedback_record_skip_reason"] = ""
    feedback_summary["stored_in_feedback_corpus"] = bool(row["stored_in_feedback_corpus"])
    feedback_summary["source_reward_adjustment"] = source_reward_adjustment_from_summary(
        feedback_summary,
        candidate_source=selected_meta["source"],
    )
    row["feedback_summary"] = feedback_summary
    feedback_outcome_recorder = getattr(feedback, "record_candidate_outcome", None) or getattr(
        feedback,
        "record_candidate_result",
    )
    row["source_reward"] = feedback_outcome_recorder(
        selected_meta["source"],
        has_finding=discovery_signals.feedback_finding,
        is_new_behavior=feedback_source_new_behavior(row, discovery_signals),
        preflight=preflight_row,
        candidate_bug=bool(reward_signals["candidate_bug"]),
        semantic_divergence=discovery_signals.rewardable_semantic_divergence,
        false_positive=bool(reward_signals["false_positive"]),
        candidate_bug_families=row_candidate_families,
        candidate_bug_signatures=row_candidate_signatures,
        reward_adjustment=float(feedback_summary.get("source_reward_adjustment", 0.0) or 0.0),
    )
    profile_reward = _record_generator_profile_feedback(
        feedback,
        selected_meta,
        row,
        context_features=generator_profile_context_features,
        reward_signals=reward_signals,
    )
    if profile_reward is not None:
        row["generator_profile_selection"]["reward"] = profile_reward
    for selection_key in (
        "semantic_objective_selection",
        "metamorphic_relation_selection",
        "version_pair_selection",
    ):
        learning_reward = _record_adaptive_action_feedback(
            feedback,
            row.get(selection_key, {}),
            row,
            context_features=tuple(row.get("case_learning_context", []) or ()),
            version_id=version_id,
            reward_signals=reward_signals,
        )
        if learning_reward is not None and isinstance(row.get(selection_key), dict):
            row[selection_key]["reward"] = learning_reward
    row["backend_pair_feedback"] = _record_backend_pair_feedback(
        feedback,
        row,
        context_features=tuple(row.get("backend_pair_context", []) or ()),
        version_id=version_id,
        reward_signals=reward_signals,
    )
    row["source_scheduler"] = _source_scheduler_snapshot(feedback)
    return FeedbackUpdateResult(
        finding_outcomes=finding_outcomes,
        elapsed_ms=(time.perf_counter() - started) * 1000,
    )


def _apply_feedback_disabled_row(
    row: dict[str, Any],
    *,
    config: ExperimentConfig,
    finding_outcomes: Any | None,
) -> None:
    row["stored_in_feedback_corpus"] = False
    row["feedback_corpus_persisted"] = False
    row["feedback_eligible"] = False
    row["feedback_skip_reason"] = "feedback_disabled"
    row["feedback_record_skip_reason"] = ""
    row["source_reward"] = None
    row["backend_pair_feedback"] = {"recorded": 0, "rewarded_pairs": [], "disagree_pairs": []}
    row["source_scheduler"] = []
    row["feedback_summary"] = feedback_summary_for_case(
        row,
        known_saturated_bug_families=config.known_saturated_bug_families,
        finding_outcomes=finding_outcomes,
    )


def _source_scheduler_snapshot(feedback: Any | None) -> list[dict[str, Any]]:
    return source_scheduler_snapshot(feedback)


def _feedback_storage_decision(
    case: Case,
    *,
    candidate_source: str = "generated",
    seed_lineage: dict[str, Any] | None = None,
    candidate_bug_families: list[str] | tuple[str, ...] | None = None,
    allow_feedback_mutation_child_candidate_bug: bool = False,
    max_feedback_mutation_child_depth: int = 2,
) -> tuple[bool, str]:
    return feedback_storage_decision(
        case,
        FeedbackStorageContext(
            candidate_source=str(candidate_source or "generated"),
            seed_lineage=seed_lineage,
            candidate_bug_families=tuple(
                str(family).strip()
                for family in (candidate_bug_families or ())
                if str(family).strip()
            ),
        ),
        FeedbackStoragePolicy(
            allow_feedback_mutation_child_candidate_bug=allow_feedback_mutation_child_candidate_bug,
            max_feedback_mutation_child_depth=max_feedback_mutation_child_depth,
        ),
    )
