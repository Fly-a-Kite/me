from __future__ import annotations

from collections import Counter
from typing import Any

from datadiff.exploration_objectives import EXPLORATION_OBJECTIVE_PREFIX
from datadiff.family_novelty import candidate_family_novelty_reward
from datadiff.finding_outcomes import (
    CANDIDATE_BUG_VERDICT,
    EMPTY_FINDING_REWARD_SIGNALS,
    FALSE_POSITIVE_VERDICTS,
    ISSUE_REPLAY_ORIGIN,
    NEEDS_CONFIRMATION_VERDICTS,
    OFFLINE_BUCKET_FALSE_POSITIVE,
    OFFLINE_BUCKET_KNOWN_BUG,
    OFFLINE_BUCKET_NEEDS_TRIAGE,
    OFFLINE_BUCKET_NEW_BUG,
    OFFLINE_BUCKET_SEMANTIC_DIVERGENCE,
    OFFLINE_BUCKET_UNCLASSIFIED,
    RESOLVED_SEMANTIC_DIVERGENCE_VERDICTS,
    SEMANTIC_DIVERGENCE_VERDICTS,
    FindingOutcomeAnalysis,
    analyze_finding_outcomes,
    backend_group_key,
    candidate_issue_family_key,
    candidate_issue_family_keys,
    candidate_issue_signatures,
    family_key_matches_known_family,
    is_candidate_issue_finding,
    is_false_positive_finding,
    is_issue_replay_finding,
    is_known_saturated_candidate_issue_finding,
    is_resolved_semantic_divergence_finding,
    is_rewardable_candidate_issue_finding,
    is_semantic_divergence_finding,
    is_source_issue_candidate_issue_finding,
    issue_replay_candidate_issue_family_keys,
    offline_finding_bucket,
    offline_finding_buckets,
    row_has_rewardable_new_behavior,
    row_reward_signals,
    split_family_key,
    suspicious_key,
)
from datadiff.mutator import mutation_operator_profiles
from datadiff.multi_objective import (
    CostVector,
    ObjectiveVector,
    SEED_OBJECTIVE_SPEC,
    bounded_ratio,
    constrained_objective_score,
)
from datadiff.semantic_signal import CANONICAL_SEMANTIC_SIGNAL_PREFIX, canonical_target_key


def online_case_reward(
    row: dict[str, Any],
    known_saturated_bug_families: list[str] | tuple[str, ...] | None = None,
    *,
    finding_outcomes: FindingOutcomeAnalysis | None = None,
) -> float:
    signals = row_reward_signals(
        row,
        known_saturated_bug_families=known_saturated_bug_families,
        finding_outcomes=finding_outcomes,
    )
    preflight = row.get("preflight") or {}
    signal_new_behavior = row_has_rewardable_new_behavior(
        row,
        known_saturated_bug_families=known_saturated_bug_families,
        finding_outcomes=finding_outcomes,
    )
    reward = (
        4.0 * signals["candidate_bug_count"]
        + 0.20 * signals["semantic_divergence_needs_confirmation_count"]
        + (0.5 if signal_new_behavior else 0.0)
        - 0.15 * signals["resolved_semantic_divergence_count"]
        - 2.5 * signals["false_positive_count"]
    )
    if not bool(preflight.get("valid", True)) or bool(preflight.get("fallback_used", False)):
        reward -= 0.5
    if reward == 0.0:
        reward -= 0.1
    return reward


def feedback_summary_for_case(
    row: dict[str, Any],
    known_saturated_bug_families: list[str] | tuple[str, ...] | None = None,
    *,
    finding_outcomes: FindingOutcomeAnalysis | None = None,
) -> dict[str, Any]:
    known_families = known_saturated_bug_families or ()
    findings = row.get("findings") or []
    analysis = None
    if finding_outcomes is not None:
        analysis = finding_outcomes
        signals = analysis.reward_signals()
    elif findings:
        analysis = analyze_finding_outcomes(
            findings,
            known_saturated_bug_families=known_families,
        )
        signals = analysis.reward_signals()
    else:
        signals = dict(EMPTY_FINDING_REWARD_SIGNALS)
    quality = _quality_oracle_signals(row.get("quality_oracles") or [])
    summary = {
        **signals,
        "candidate_source": str(row.get("candidate_source", "generated")),
        "has_finding": bool(findings),
        "is_new_behavior": bool(row.get("is_new_behavior")),
        "raw_signal_new_behavior": bool(row.get("signal_new_behavior", row.get("is_new_behavior"))),
        "signal_new_behavior": row_has_rewardable_new_behavior(
            row,
            known_saturated_bug_families=known_families,
            finding_outcomes=analysis,
        ),
        "candidate_bug_families": list(analysis.candidate_bug_families) if analysis is not None else [],
        "candidate_bug_signatures": list(analysis.candidate_bug_signatures) if analysis is not None else [],
        "stored_in_feedback_corpus": bool(row.get("stored_in_feedback_corpus")),
        "preflight_valid": bool((row.get("preflight") or {}).get("valid", True)),
        "preflight_fallback_used": bool((row.get("preflight") or {}).get("fallback_used", False)),
        **quality,
    }
    summary.update(_feedback_decision_summary(row))
    summary["source_reward_adjustment"] = source_reward_adjustment_from_summary(
        summary,
        candidate_source=str(row.get("candidate_source", "generated")),
    )
    summary["guidance_reward_adjustment"] = guidance_reward_adjustment_from_summary(summary)
    summary["seed_schedule_delta"] = seed_schedule_delta_from_summary(summary)
    return summary


def resolved_case_feedback_summary(
    row: dict[str, Any],
    known_saturated_bug_families: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    base = feedback_summary_for_case(
        row,
        known_saturated_bug_families=known_saturated_bug_families,
    )
    existing = row.get("feedback_summary")
    if not isinstance(existing, dict):
        return base
    merged = dict(existing)
    merged.update(base)
    merged["candidate_source"] = str(
        row.get("candidate_source", existing.get("candidate_source", merged["candidate_source"]))
    )
    if not row.get("quality_oracles"):
        _inherit_historical_quality_summary(merged, existing)
        merged["source_reward_adjustment"] = source_reward_adjustment_from_summary(
            merged,
            candidate_source=str(merged.get("candidate_source", "generated")),
        )
        merged["guidance_reward_adjustment"] = guidance_reward_adjustment_from_summary(merged)
        merged["seed_schedule_delta"] = seed_schedule_delta_from_summary(merged)
    return merged


def _inherit_historical_quality_summary(summary: dict[str, Any], existing: dict[str, Any]) -> None:
    for key in (
        "quality_oracle_count",
        "quality_pass_count",
        "quality_fail_count",
        "quality_score_total",
        "mutation_oracle_verdict",
        "mutation_oracle_score",
        "mutation_oracle_passed",
        "feedback_oracle_verdict",
        "feedback_oracle_score",
        "feedback_oracle_passed",
        "guidance_oracle_verdict",
        "guidance_oracle_score",
        "guidance_oracle_passed",
    ):
        if key in existing:
            summary[key] = existing[key]


def aggregate_feedback_summary(
    rows: list[dict[str, Any]],
    known_saturated_bug_families: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    out = {
        "feedback_case_count": 0,
        "feedback_mutation_cases": 0,
        "stored_in_feedback_corpus_cases": 0,
        "quality_oracle_count": 0,
        "quality_pass_count": 0,
        "quality_fail_count": 0,
        "quality_score_total": 0.0,
        "source_reward_adjustment_total": 0.0,
        "guidance_reward_adjustment_total": 0.0,
        "seed_schedule_delta_total": 0.0,
        "productive_mutation_cases": 0,
        "invalid_mutation_cases": 0,
        "redundant_mutation_cases": 0,
        "feedback_finding_yield_cases": 0,
        "feedback_new_behavior_yield_cases": 0,
        "feedback_redundant_behavior_cases": 0,
        "guided_productive_cases": 0,
        "guided_target_miss_cases": 0,
        "guided_redundant_cases": 0,
        "feedback_target_key_count": 0,
        "feedback_semantic_family_target_count": 0,
        "feedback_semantic_signal_target_count": 0,
        "feedback_exploration_objective_target_count": 0,
        "feedback_operator_affinity_hit_cases": 0,
        "feedback_selected_operator_count": 0,
        "feedback_selected_operator_score_total": 0.0,
        "top_feedback_selected_operators": "none",
        "top_feedback_semantic_target_keys": "none",
    }
    selected_operator_counter: Counter[str] = Counter()
    semantic_target_key_counter: Counter[str] = Counter()
    for row in rows:
        summary = resolved_case_feedback_summary(
            row,
            known_saturated_bug_families=known_saturated_bug_families,
        )
        out["feedback_case_count"] += 1
        out["feedback_mutation_cases"] += int(summary.get("candidate_source") == "feedback_mutation")
        out["stored_in_feedback_corpus_cases"] += int(bool(summary.get("stored_in_feedback_corpus")))
        out["quality_oracle_count"] += int(summary.get("quality_oracle_count", 0) or 0)
        out["quality_pass_count"] += int(summary.get("quality_pass_count", 0) or 0)
        out["quality_fail_count"] += int(summary.get("quality_fail_count", 0) or 0)
        out["quality_score_total"] += float(summary.get("quality_score_total", 0.0) or 0.0)
        out["source_reward_adjustment_total"] += float(summary.get("source_reward_adjustment", 0.0) or 0.0)
        out["guidance_reward_adjustment_total"] += float(summary.get("guidance_reward_adjustment", 0.0) or 0.0)
        out["seed_schedule_delta_total"] += float(summary.get("seed_schedule_delta", 0.0) or 0.0)
        out["feedback_target_key_count"] += int(summary.get("feedback_target_key_count", 0) or 0)
        out["feedback_semantic_family_target_count"] += int(summary.get("feedback_semantic_family_target_count", 0) or 0)
        out["feedback_semantic_signal_target_count"] += int(summary.get("feedback_semantic_signal_target_count", 0) or 0)
        out["feedback_exploration_objective_target_count"] += int(
            summary.get("feedback_exploration_objective_target_count", 0) or 0
        )
        out["feedback_operator_affinity_hit_cases"] += int(bool(summary.get("feedback_operator_affinity_hit", False)))
        selected_operator = str(summary.get("feedback_selected_operator", "")).strip()
        if selected_operator:
            out["feedback_selected_operator_count"] += 1
            out["feedback_selected_operator_score_total"] += float(
                summary.get("feedback_selected_operator_score", 0.0) or 0.0
            )
            selected_operator_counter[selected_operator] += 1
        semantic_target_key_counter.update(_string_items(summary.get("feedback_semantic_target_keys", [])))

        mutation_verdict = str(summary.get("mutation_oracle_verdict", "")).strip()
        if mutation_verdict == "productive_mutation":
            out["productive_mutation_cases"] += 1
        elif mutation_verdict == "invalid_mutation_repaired":
            out["invalid_mutation_cases"] += 1
        elif mutation_verdict == "redundant_mutation":
            out["redundant_mutation_cases"] += 1

        feedback_verdict = str(summary.get("feedback_oracle_verdict", "")).strip()
        if feedback_verdict == "finding_yield":
            out["feedback_finding_yield_cases"] += 1
        elif feedback_verdict == "new_behavior_yield":
            out["feedback_new_behavior_yield_cases"] += 1
        elif feedback_verdict == "redundant_behavior":
            out["feedback_redundant_behavior_cases"] += 1

        guidance_verdict = str(summary.get("guidance_oracle_verdict", "")).strip()
        if guidance_verdict == "guided_productive":
            out["guided_productive_cases"] += 1
        elif guidance_verdict == "guided_target_miss":
            out["guided_target_miss_cases"] += 1
        elif guidance_verdict == "guided_redundant":
            out["guided_redundant_cases"] += 1
    out["top_feedback_selected_operators"] = _counter_summary(selected_operator_counter)
    out["top_feedback_semantic_target_keys"] = _counter_summary(semantic_target_key_counter)
    return out


def source_reward_adjustment_from_summary(summary: dict[str, Any], *, candidate_source: str) -> float:
    adjustment = 0.0
    mutation_verdict = str(summary.get("mutation_oracle_verdict", ""))
    feedback_verdict = str(summary.get("feedback_oracle_verdict", ""))
    guidance_verdict = str(summary.get("guidance_oracle_verdict", ""))
    suppress_positive_signal = _suppress_auxiliary_positive_feedback(summary)
    if candidate_source == "feedback_mutation":
        if mutation_verdict == "productive_mutation" and not suppress_positive_signal:
            adjustment += 0.35
        elif mutation_verdict == "invalid_mutation_repaired":
            adjustment -= 0.35
        elif mutation_verdict == "redundant_mutation":
            adjustment -= 0.10
    if feedback_verdict == "finding_yield" and not suppress_positive_signal:
        adjustment += 0.10
    elif feedback_verdict == "redundant_behavior":
        adjustment -= 0.05
    if guidance_verdict == "guided_productive" and not suppress_positive_signal:
        adjustment += 0.05
    elif guidance_verdict == "guided_target_miss":
        adjustment -= 0.05
    if bool(summary.get("stored_in_feedback_corpus")) and not suppress_positive_signal:
        adjustment += 0.05
    return max(-0.50, min(0.50, adjustment))


def guidance_reward_adjustment_from_summary(summary: dict[str, Any]) -> float:
    verdict = str(summary.get("guidance_oracle_verdict", ""))
    suppress_positive_signal = _suppress_auxiliary_positive_feedback(summary)
    if verdict == "guided_productive" and not suppress_positive_signal:
        return 0.25
    if verdict == "guided_target_miss":
        return -0.15
    if verdict == "guided_redundant":
        return -0.05
    return 0.0


def seed_schedule_delta_from_summary(summary: dict[str, Any]) -> float:
    signal_new_behavior = bool(summary.get("signal_new_behavior", summary.get("is_new_behavior")))
    suppress_positive_signal = _suppress_auxiliary_positive_feedback(summary)
    discovery_signal = min(1.0, 0.70 * float(summary.get("candidate_bug_count", 0)))
    semantic_signal = min(
        1.0,
        0.60 * float(summary.get("semantic_divergence_needs_confirmation_count", 0)),
    )
    behavior_signal = 0.0 if suppress_positive_signal else (
        1.0 if signal_new_behavior else 0.10 * float(summary.get("quality_pass_count", 0))
    )
    structural_signal = 0.0 if suppress_positive_signal else (
        0.25 * bool(summary.get("stored_in_feedback_corpus"))
        + 0.15 * bool(summary.get("feedback_operator_affinity_hit"))
    )
    vector = ObjectiveVector(
        discovery=discovery_signal,
        semantic=semantic_signal,
        novelty=0.0,
        expandability=0.25 * bool(summary.get("stored_in_feedback_corpus")),
        structural_risk=0.15 * bool(summary.get("feedback_operator_affinity_hit")),
        coverage_gain=structural_signal,
    )
    cost = CostVector(
        invalidity=1.0 if (
            not bool(summary.get("preflight_valid", True))
            or bool(summary.get("preflight_fallback_used", False))
        ) else 0.0,
        false_positive=bounded_ratio(float(summary.get("false_positive_count", 0)), 1.0),
        redundancy=bounded_ratio(float(summary.get("resolved_semantic_divergence_count", 0)), 2.0)
        + bounded_ratio(float(summary.get("quality_fail_count", 0)), 4.0),
    )
    delta = constrained_objective_score(
        vector,
        cost=cost,
        spec=SEED_OBJECTIVE_SPEC,
        validity=1.0 - cost.invalidity,
        false_positive_risk=cost.false_positive,
    )
    if bool(summary.get("resolved_semantic_divergence_count", 0)) and not bool(summary.get("rewardable_semantic_divergence", False)):
        delta -= 0.17
    if bool(summary.get("source_issue_candidate_bug_count", 0)):
        delta -= 0.05
    if not bool(summary.get("preflight_valid", True)) or bool(summary.get("preflight_fallback_used", False)):
        delta -= 0.20
    return max(-2.0, min(6.0, delta))


def _suppress_auxiliary_positive_feedback(summary: dict[str, Any]) -> bool:
    has_rewardable_candidate = bool(summary.get("candidate_bug_count", 0))
    has_rewardable_semantic = bool(summary.get("rewardable_semantic_divergence", False))
    if bool(summary.get("false_positive_count", 0)):
        return True
    if (
        bool(summary.get("resolved_semantic_divergence_count", 0))
        and not has_rewardable_candidate
        and not has_rewardable_semantic
    ):
        return True
    known_candidate_only = (
        (
            bool(summary.get("issue_replay_candidate_bug_count", 0))
            or bool(summary.get("known_saturated_candidate_bug_count", 0))
            or bool(summary.get("source_issue_candidate_bug_count", 0))
        )
        and not has_rewardable_candidate
        and not has_rewardable_semantic
    )
    return known_candidate_only


def _quality_oracle_signals(oracles: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {
        "quality_oracle_count": 0,
        "quality_pass_count": 0,
        "quality_fail_count": 0,
        "quality_score_total": 0.0,
        "mutation_oracle_verdict": "",
        "mutation_oracle_score": 0.0,
        "mutation_oracle_passed": False,
        "feedback_oracle_verdict": "",
        "feedback_oracle_score": 0.0,
        "feedback_oracle_passed": False,
        "guidance_oracle_verdict": "",
        "guidance_oracle_score": 0.0,
        "guidance_oracle_passed": False,
    }
    for oracle in oracles:
        name = str(oracle.get("name", "")).strip()
        verdict = str(oracle.get("verdict", "")).strip()
        score = float(oracle.get("score", 0.0) or 0.0)
        passed = bool(oracle.get("passed", False))
        summary["quality_oracle_count"] += 1
        summary["quality_score_total"] += score
        if verdict != "not_applicable":
            summary["quality_pass_count"] += int(passed)
            summary["quality_fail_count"] += int(not passed)
        if name in {"mutation", "feedback", "guidance"}:
            summary[f"{name}_oracle_verdict"] = verdict
            summary[f"{name}_oracle_score"] = score
            summary[f"{name}_oracle_passed"] = passed
    return summary


def _feedback_decision_summary(row: dict[str, Any]) -> dict[str, Any]:
    selection = row.get("feedback_decision", {})
    if not isinstance(selection, dict):
        selection = {}
    target_keys = [_canonical_feedback_target_key(item) for item in selection.get("target_keys", []) or []]
    target_keys = [item for item in target_keys if item]
    semantic_family_targets: list[str] = []
    semantic_signal_targets: list[str] = []
    exploration_objective_targets: list[str] = []
    semantic_target_keys: list[str] = []
    for key in target_keys:
        if key.startswith("semantic_family:"):
            family = key.removeprefix("semantic_family:").strip()
            if family:
                semantic_family_targets.append(family)
                semantic_target_keys.append(f"semantic_family:{family}")
        elif key.startswith(CANONICAL_SEMANTIC_SIGNAL_PREFIX):
            signal = key.removeprefix(CANONICAL_SEMANTIC_SIGNAL_PREFIX).strip()
            if signal:
                semantic_signal_targets.append(signal)
                semantic_target_keys.append(f"semantic_signal:{signal}")
        elif key.startswith(EXPLORATION_OBJECTIVE_PREFIX):
            objective = key.removeprefix(EXPLORATION_OBJECTIVE_PREFIX).strip()
            if objective:
                exploration_objective_targets.append(objective)
        elif key.startswith("feature:semantic_family:"):
            family = key.removeprefix("feature:semantic_family:").strip()
            if family:
                semantic_family_targets.append(family)
                semantic_target_keys.append(f"semantic_family:{family}")
        elif key.startswith(f"feature:{EXPLORATION_OBJECTIVE_PREFIX}"):
            objective = key.removeprefix(f"feature:{EXPLORATION_OBJECTIVE_PREFIX}").strip()
            if objective:
                exploration_objective_targets.append(objective)
    selected_operator = str(selection.get("selected_operator", "")).strip()
    selected_operator_score = float(selection.get("selected_operator_score", 0.0) or 0.0)
    operator_profiles = mutation_operator_profiles(allow_probe_operators=False)
    selected_profile = operator_profiles.get(selected_operator)
    operator_family_affinity = _string_items(
        getattr(selected_profile, "semantic_family_affinity", getattr(selected_profile, "semantic_affinity", ()))
        if selected_profile is not None
        else ()
    )
    operator_signal_affinity = _string_items(
        getattr(selected_profile, "semantic_signal_affinity", ())
        if selected_profile is not None
        else ()
    )
    operator_objective_affinity = _string_items(
        getattr(selected_profile, "exploration_objective_affinity", ())
        if selected_profile is not None
        else ()
    )
    family_affinity_hit = bool(set(operator_family_affinity) & set(semantic_family_targets))
    signal_affinity_hit = bool(set(operator_signal_affinity) & set(semantic_signal_targets))
    objective_affinity_hit = bool(
        set(operator_objective_affinity) & set(exploration_objective_targets)
    )
    return {
        "feedback_parent_case_id": str(selection.get("parent_case_id", "")).strip(),
        "feedback_selected_operator": selected_operator,
        "feedback_selected_operator_score": selected_operator_score,
        "feedback_target_key_count": len(target_keys),
        "feedback_semantic_family_target_count": len(semantic_family_targets),
        "feedback_semantic_signal_target_count": len(semantic_signal_targets),
        "feedback_exploration_objective_target_count": len(exploration_objective_targets),
        "feedback_semantic_family_targets": semantic_family_targets,
        "feedback_semantic_signal_targets": semantic_signal_targets,
        "feedback_exploration_objective_targets": exploration_objective_targets,
        "feedback_semantic_target_keys": semantic_target_keys,
        "feedback_operator_family_affinity": operator_family_affinity,
        "feedback_operator_signal_affinity": operator_signal_affinity,
        "feedback_operator_objective_affinity": operator_objective_affinity,
        "feedback_operator_family_affinity_hit": family_affinity_hit,
        "feedback_operator_signal_affinity_hit": signal_affinity_hit,
        "feedback_operator_objective_affinity_hit": objective_affinity_hit,
        "feedback_operator_affinity_hit": (
            family_affinity_hit or signal_affinity_hit or objective_affinity_hit
        ),
    }


_feedback_selection_summary = _feedback_decision_summary


def _canonical_feedback_target_key(value: Any) -> str:
    return str(canonical_target_key(value)).strip()


def _counter_summary(counter: Counter[str], *, limit: int = 8) -> str:
    if not counter:
        return "none"
    items = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    return "; ".join(f"{key}:{count}" for key, count in items[:limit])


def _string_items(values: list[Any] | tuple[Any, ...] | set[Any] | None) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values or []:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


is_candidate_bug_finding = is_candidate_issue_finding
candidate_bug_family_key = candidate_issue_family_key
is_known_saturated_candidate_bug_finding = is_known_saturated_candidate_issue_finding
is_rewardable_candidate_bug_finding = is_rewardable_candidate_issue_finding
candidate_bug_family_keys = candidate_issue_family_keys
issue_replay_candidate_bug_family_keys = issue_replay_candidate_issue_family_keys
candidate_bug_signatures = candidate_issue_signatures
summarize_case_feedback = feedback_summary_for_case
coerce_case_feedback_summary = resolved_case_feedback_summary
aggregate_feedback_summaries = aggregate_feedback_summary
resolved_feedback_summary = resolved_case_feedback_summary
