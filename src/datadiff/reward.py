from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from datadiff.mutator import mutation_operator_profiles
from datadiff.semantic_signal import CANONICAL_SEMANTIC_SIGNAL_PREFIX, canonical_target_key

CANDIDATE_BUG_VERDICT = "candidate_implementation_bug"
ISSUE_REPLAY_ORIGIN = "issue_replay"
SEMANTIC_DIVERGENCE_VERDICTS = {
    "documented_semantic_divergence",
    "expected_semantic_divergence",
    "semantic_divergence_needs_confirmation",
}
RESOLVED_SEMANTIC_DIVERGENCE_VERDICTS = {
    "documented_semantic_divergence",
    "expected_semantic_divergence",
}
FALSE_POSITIVE_VERDICTS = {
    "generator_false_positive",
    "normalizer_false_positive",
}
NEEDS_CONFIRMATION_VERDICTS = {
    "needs_manual_confirmation",
    "semantic_divergence_needs_confirmation",
}
OFFLINE_BUCKET_NEW_BUG = "new_bug"
OFFLINE_BUCKET_KNOWN_BUG = "known_bug"
OFFLINE_BUCKET_FALSE_POSITIVE = "false_positive"
OFFLINE_BUCKET_SEMANTIC_DIVERGENCE = "semantic_divergence"
OFFLINE_BUCKET_NEEDS_TRIAGE = "needs_triage"
OFFLINE_BUCKET_UNCLASSIFIED = "unclassified"
EMPTY_FINDING_REWARD_SIGNALS = {
    "candidate_bug": False,
    "candidate_bug_count": 0,
    "issue_replay_candidate_bug_count": 0,
    "known_saturated_candidate_bug_count": 0,
    "semantic_divergence": False,
    "semantic_divergence_count": 0,
    "resolved_semantic_divergence_count": 0,
    "rewardable_semantic_divergence": False,
    "false_positive": False,
    "false_positive_count": 0,
    "needs_confirmation": False,
    "needs_confirmation_count": 0,
}


@dataclass(slots=True)
class FindingOutcomeAnalysis:
    candidate_bug_count: int = 0
    issue_replay_candidate_bug_count: int = 0
    known_saturated_candidate_bug_count: int = 0
    semantic_divergence_count: int = 0
    resolved_semantic_divergence_count: int = 0
    false_positive_count: int = 0
    needs_confirmation_count: int = 0
    root_cause_counts: Counter[str] = field(default_factory=Counter)
    candidate_bug_families: Counter[str] = field(default_factory=Counter)
    issue_replay_candidate_bug_families: Counter[str] = field(default_factory=Counter)
    candidate_bug_signatures: Counter[str] = field(default_factory=Counter)

    def reward_signals(self) -> dict[str, Any]:
        if (
            self.candidate_bug_count == 0
            and self.issue_replay_candidate_bug_count == 0
            and self.known_saturated_candidate_bug_count == 0
            and self.semantic_divergence_count == 0
            and self.resolved_semantic_divergence_count == 0
            and self.false_positive_count == 0
            and self.needs_confirmation_count == 0
        ):
            return dict(EMPTY_FINDING_REWARD_SIGNALS)
        return {
            "candidate_bug": self.candidate_bug_count > 0,
            "candidate_bug_count": self.candidate_bug_count,
            "issue_replay_candidate_bug_count": self.issue_replay_candidate_bug_count,
            "known_saturated_candidate_bug_count": self.known_saturated_candidate_bug_count,
            "semantic_divergence": self.semantic_divergence_count > 0,
            "semantic_divergence_count": self.semantic_divergence_count,
            "resolved_semantic_divergence_count": self.resolved_semantic_divergence_count,
            "rewardable_semantic_divergence": self.needs_confirmation_count > 0,
            "false_positive": self.false_positive_count > 0,
            "false_positive_count": self.false_positive_count,
            "needs_confirmation": self.needs_confirmation_count > 0,
            "needs_confirmation_count": self.needs_confirmation_count,
        }


def analyze_finding_outcomes(
    findings: list[dict[str, Any]],
    known_saturated_bug_families: list[str] | tuple[str, ...] | None = None,
) -> FindingOutcomeAnalysis:
    known_families = known_saturated_bug_families or ()
    analysis = FindingOutcomeAnalysis()
    root_by_backend_group: dict[str, str] = {}
    rewardable_findings: list[tuple[str, str, str]] = []
    for finding in findings:
        root = str(finding.get("root_cause", "unknown"))
        verdict = str(finding.get("triage_verdict", "unclassified"))
        analysis.root_cause_counts[root] += 1
        if verdict in SEMANTIC_DIVERGENCE_VERDICTS:
            analysis.semantic_divergence_count += 1
        if verdict in RESOLVED_SEMANTIC_DIVERGENCE_VERDICTS:
            analysis.resolved_semantic_divergence_count += 1
        if verdict in NEEDS_CONFIRMATION_VERDICTS:
            analysis.needs_confirmation_count += 1
        if is_false_positive_finding(finding):
            analysis.false_positive_count += 1

        if not is_candidate_issue_finding(finding):
            continue
        backend_group = backend_group_key(finding)
        family_key = f"{root}@{backend_group}"
        is_issue_replay = is_issue_replay_finding(finding)
        is_known_saturated = family_key_matches_known_family(family_key, known_families)
        if is_issue_replay:
            analysis.issue_replay_candidate_bug_count += 1
            if not root.startswith("metamorphic_"):
                analysis.issue_replay_candidate_bug_families[family_key] += 1
        if is_known_saturated:
            analysis.known_saturated_candidate_bug_count += 1
        if is_issue_replay or is_known_saturated:
            continue
        analysis.candidate_bug_count += 1
        if not root.startswith("metamorphic_"):
            root_by_backend_group.setdefault(backend_group, root)
        rewardable_findings.append((root, backend_group, str(finding.get("signature", "")).strip()))
    for root, backend_group, signature in rewardable_findings:
        normalized_root = root
        if normalized_root.startswith("metamorphic_") and backend_group in root_by_backend_group:
            normalized_root = root_by_backend_group[backend_group]
        analysis.candidate_bug_families[f"{normalized_root}@{backend_group}"] += 1
        if signature:
            analysis.candidate_bug_signatures[signature] += 1
    return analysis


def candidate_family_novelty_reward(
    previous_hits: int,
    *,
    enable_family_saturation: bool = True,
    family_saturation_threshold: int = 8,
    saturated_family_reward: float = 0.02,
) -> float:
    if enable_family_saturation and family_saturation_threshold > 0 and previous_hits >= family_saturation_threshold:
        return max(0.0, saturated_family_reward)
    if previous_hits <= 0:
        return 4.0
    if previous_hits <= 2:
        return 1.5
    if previous_hits <= 8:
        return 0.75 / max(1.0, previous_hits**0.5)
    return 0.10


def is_candidate_issue_finding(finding: dict[str, Any]) -> bool:
    return finding.get("triage_verdict") == CANDIDATE_BUG_VERDICT and not finding.get("false_positive")


def is_issue_replay_finding(finding: dict[str, Any]) -> bool:
    return str(finding.get("discovery_origin", "")).strip() == ISSUE_REPLAY_ORIGIN


def backend_group_key(finding: dict[str, Any]) -> str:
    return ",".join(sorted(finding.get("suspicious_backends", []) or [])) or "unknown"


def candidate_issue_family_key(finding: dict[str, Any]) -> str:
    root = str(finding.get("root_cause", "unknown"))
    backend_group = backend_group_key(finding)
    return f"{root}@{backend_group}"


def family_key_matches_known_family(candidate_family: str, known_bug_families: list[str] | tuple[str, ...]) -> bool:
    candidate_root, candidate_backends = _split_family_key(candidate_family)
    for known_family in known_bug_families:
        known_root, known_backends = _split_family_key(known_family)
        if known_root != candidate_root:
            continue
        if not known_backends or not candidate_backends or known_backends & candidate_backends:
            return True
    return False


def is_known_saturated_candidate_issue_finding(
    finding: dict[str, Any],
    known_saturated_bug_families: list[str] | tuple[str, ...] | None = None,
) -> bool:
    if not is_candidate_issue_finding(finding):
        return False
    known_families = known_saturated_bug_families or ()
    return family_key_matches_known_family(candidate_issue_family_key(finding), known_families)


def is_rewardable_candidate_issue_finding(
    finding: dict[str, Any],
    known_saturated_bug_families: list[str] | tuple[str, ...] | None = None,
) -> bool:
    return (
        is_candidate_issue_finding(finding)
        and not is_issue_replay_finding(finding)
        and not is_known_saturated_candidate_issue_finding(finding, known_saturated_bug_families)
    )


def is_semantic_divergence_finding(finding: dict[str, Any]) -> bool:
    return str(finding.get("triage_verdict", "unclassified")) in SEMANTIC_DIVERGENCE_VERDICTS


def is_resolved_semantic_divergence_finding(finding: dict[str, Any]) -> bool:
    return str(finding.get("triage_verdict", "unclassified")) in RESOLVED_SEMANTIC_DIVERGENCE_VERDICTS


def is_false_positive_finding(finding: dict[str, Any]) -> bool:
    return bool(finding.get("false_positive")) or str(finding.get("triage_verdict", "unclassified")) in FALSE_POSITIVE_VERDICTS


def offline_finding_bucket(
    finding: dict[str, Any],
    known_saturated_bug_families: list[str] | tuple[str, ...] | None = None,
) -> str:
    if is_false_positive_finding(finding):
        return OFFLINE_BUCKET_FALSE_POSITIVE
    if is_semantic_divergence_finding(finding):
        return OFFLINE_BUCKET_SEMANTIC_DIVERGENCE
    if is_candidate_issue_finding(finding):
        if (
            is_known_saturated_candidate_issue_finding(finding, known_saturated_bug_families)
            or is_issue_replay_finding(finding)
            or str(finding.get("source_issue", "")).strip()
        ):
            return OFFLINE_BUCKET_KNOWN_BUG
        return OFFLINE_BUCKET_NEW_BUG
    if str(finding.get("triage_verdict", "unclassified")) in NEEDS_CONFIRMATION_VERDICTS:
        return OFFLINE_BUCKET_NEEDS_TRIAGE
    return OFFLINE_BUCKET_UNCLASSIFIED


def offline_finding_buckets(
    findings: list[dict[str, Any]],
    known_saturated_bug_families: list[str] | tuple[str, ...] | None = None,
) -> Counter[str]:
    return Counter(
        offline_finding_bucket(finding, known_saturated_bug_families)
        for finding in findings
    )


def suspicious_key(finding: dict[str, Any]) -> str:
    return backend_group_key(finding)


def candidate_issue_family_keys(
    findings: list[dict[str, Any]],
    known_saturated_bug_families: list[str] | tuple[str, ...] | None = None,
) -> Counter[str]:
    return analyze_finding_outcomes(
        findings,
        known_saturated_bug_families=known_saturated_bug_families,
    ).candidate_bug_families


def issue_replay_candidate_issue_family_keys(findings: list[dict[str, Any]]) -> Counter[str]:
    return analyze_finding_outcomes(findings).issue_replay_candidate_bug_families


def candidate_issue_signatures(
    findings: list[dict[str, Any]],
    known_saturated_bug_families: list[str] | tuple[str, ...] | None = None,
) -> Counter[str]:
    return analyze_finding_outcomes(
        findings,
        known_saturated_bug_families=known_saturated_bug_families,
    ).candidate_bug_signatures


def row_reward_signals(
    row: dict[str, Any],
    known_saturated_bug_families: list[str] | tuple[str, ...] | None = None,
    *,
    finding_outcomes: FindingOutcomeAnalysis | None = None,
) -> dict[str, Any]:
    findings = row.get("findings") or []
    if finding_outcomes is None and not findings:
        return dict(EMPTY_FINDING_REWARD_SIGNALS)
    analysis = finding_outcomes or analyze_finding_outcomes(
        findings,
        known_saturated_bug_families=known_saturated_bug_families,
    )
    return analysis.reward_signals()


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
    signal_new_behavior = bool(row.get("signal_new_behavior", row.get("is_new_behavior")))
    reward = (
        4.0 * signals["candidate_bug_count"]
        + 0.20 * signals["needs_confirmation_count"]
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
        "signal_new_behavior": bool(row.get("signal_new_behavior", row.get("is_new_behavior"))),
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
    merged = dict(base)
    merged.update(existing)
    merged["candidate_source"] = str(
        existing.get("candidate_source", row.get("candidate_source", merged["candidate_source"]))
    )
    merged["candidate_bug_families"] = list(existing.get("candidate_bug_families", merged["candidate_bug_families"]))
    merged["candidate_bug_signatures"] = list(
        existing.get("candidate_bug_signatures", merged["candidate_bug_signatures"])
    )
    return merged


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
    suppress_positive_signal = bool(summary.get("false_positive_count", 0)) or (
        bool(summary.get("resolved_semantic_divergence_count", 0))
        and not bool(summary.get("candidate_bug_count", 0))
        and not bool(summary.get("needs_confirmation_count", 0))
    )
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
    if verdict == "guided_productive":
        return 0.25
    if verdict == "guided_target_miss":
        return -0.15
    if verdict == "guided_redundant":
        return -0.05
    return 0.0


def seed_schedule_delta_from_summary(summary: dict[str, Any]) -> float:
    signal_new_behavior = bool(summary.get("signal_new_behavior", summary.get("is_new_behavior")))
    delta = (
        2.5 * float(summary.get("candidate_bug_count", 0))
        + 0.5 * float(summary.get("needs_confirmation_count", 0))
        + (0.5 if signal_new_behavior else 0.0)
        - 1.0 * float(summary.get("false_positive_count", 0))
        - 0.25 * float(summary.get("resolved_semantic_divergence_count", 0))
        + 0.25 * float(summary.get("quality_pass_count", 0))
        - 0.10 * float(summary.get("quality_fail_count", 0))
    )
    if not bool(summary.get("preflight_valid", True)) or bool(summary.get("preflight_fallback_used", False)):
        delta -= 0.25
    return max(-2.0, min(6.0, delta))


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
    selection = row.get("feedback_selection")
    if not isinstance(selection, dict):
        selection = row.get("feedback_decision", {})
    if not isinstance(selection, dict):
        selection = {}
    target_keys = [_canonical_feedback_target_key(item) for item in selection.get("target_keys", []) or []]
    target_keys = [item for item in target_keys if item]
    semantic_family_targets: list[str] = []
    semantic_signal_targets: list[str] = []
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
        elif key.startswith("feature:semantic_family:"):
            family = key.removeprefix("feature:semantic_family:").strip()
            if family:
                semantic_family_targets.append(family)
                semantic_target_keys.append(f"semantic_family:{family}")
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
    family_affinity_hit = bool(set(operator_family_affinity) & set(semantic_family_targets))
    signal_affinity_hit = bool(set(operator_signal_affinity) & set(semantic_signal_targets))
    return {
        "feedback_parent_case_id": str(selection.get("parent_case_id", "")).strip(),
        "feedback_selected_operator": selected_operator,
        "feedback_selected_operator_score": selected_operator_score,
        "feedback_target_key_count": len(target_keys),
        "feedback_semantic_family_target_count": len(semantic_family_targets),
        "feedback_semantic_signal_target_count": len(semantic_signal_targets),
        "feedback_semantic_family_targets": semantic_family_targets,
        "feedback_semantic_signal_targets": semantic_signal_targets,
        "feedback_semantic_target_keys": semantic_target_keys,
        "feedback_operator_family_affinity": operator_family_affinity,
        "feedback_operator_signal_affinity": operator_signal_affinity,
        "feedback_operator_family_affinity_hit": family_affinity_hit,
        "feedback_operator_signal_affinity_hit": signal_affinity_hit,
        "feedback_operator_affinity_hit": family_affinity_hit or signal_affinity_hit,
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


def _split_family_key(family_key: str) -> tuple[str, set[str]]:
    root, _, backend_part = str(family_key).partition("@")
    backends = {backend.strip() for backend in backend_part.split(",") if backend.strip()}
    return root.strip(), backends


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
