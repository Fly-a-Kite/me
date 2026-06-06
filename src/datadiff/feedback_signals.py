from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datadiff.finding_outcomes import (
    FindingOutcomeAnalysis,
    analyze_finding_outcomes,
    row_has_rewardable_new_behavior,
    row_reward_signals,
)


@dataclass(frozen=True, slots=True)
class FeedbackDiscoverySignals:
    finding_outcomes: FindingOutcomeAnalysis | None
    reward_signals: dict[str, Any]
    candidate_bug_families: list[str]
    candidate_bug_signatures: list[str]
    rewardable_semantic_divergence: bool
    resolved_semantic_only: bool
    feedback_finding: bool
    rewardable_new_behavior: bool


def build_feedback_discovery_signals(
    row: dict[str, Any],
    *,
    known_saturated_bug_families: list[str] | tuple[str, ...] | None = None,
    finding_outcomes: FindingOutcomeAnalysis | None = None,
) -> FeedbackDiscoverySignals:
    outcomes = finding_outcomes
    findings = row.get("findings") or []
    if outcomes is None and findings:
        outcomes = analyze_finding_outcomes(
            findings,
            known_saturated_bug_families=known_saturated_bug_families,
        )
    signals = row_reward_signals(
        row,
        known_saturated_bug_families=known_saturated_bug_families,
        finding_outcomes=outcomes,
    )
    candidate_families = list(outcomes.candidate_bug_families) if outcomes else []
    candidate_signatures = list(outcomes.candidate_bug_signatures) if outcomes else []
    rewardable_semantic = bool(signals["rewardable_semantic_divergence"])
    resolved_semantic_only = (
        bool(signals["resolved_semantic_divergence_count"])
        and not bool(signals["candidate_bug"])
        and not rewardable_semantic
    )
    return FeedbackDiscoverySignals(
        finding_outcomes=outcomes,
        reward_signals=signals,
        candidate_bug_families=candidate_families,
        candidate_bug_signatures=candidate_signatures,
        rewardable_semantic_divergence=rewardable_semantic,
        resolved_semantic_only=resolved_semantic_only,
        feedback_finding=bool(candidate_families) or rewardable_semantic,
        rewardable_new_behavior=row_has_rewardable_new_behavior(
            row,
            known_saturated_bug_families=known_saturated_bug_families,
            finding_outcomes=outcomes,
        ),
    )


def feedback_source_new_behavior(row: dict[str, Any], signals: FeedbackDiscoverySignals) -> bool:
    return signals.rewardable_new_behavior
