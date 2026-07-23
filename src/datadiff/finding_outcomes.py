from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from datadiff.family_novelty import family_key_matches_known_family, split_family_key

CANDIDATE_BUG_VERDICT = "candidate_implementation_bug"
ISSUE_REPLAY_ORIGIN = "issue_replay"
NON_FRESH_LIFECYCLE_STATES = {
    "submitted",
    "acknowledged",
    "confirmed",
    "fixed",
    "duplicate",
    "invalid",
    "saturated",
}
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
    "harness_lowering_error",
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
    "source_issue_candidate_bug_count": 0,
    "nonfresh_lifecycle_candidate_bug_count": 0,
    "duplicate_candidate_bug_count": 0,
    "semantic_divergence": False,
    "semantic_divergence_count": 0,
    "resolved_semantic_divergence_count": 0,
    "semantic_divergence_needs_confirmation_count": 0,
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
    source_issue_candidate_bug_count: int = 0
    nonfresh_lifecycle_candidate_bug_count: int = 0
    duplicate_candidate_bug_count: int = 0
    semantic_divergence_count: int = 0
    resolved_semantic_divergence_count: int = 0
    semantic_divergence_needs_confirmation_count: int = 0
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
            and self.source_issue_candidate_bug_count == 0
            and self.nonfresh_lifecycle_candidate_bug_count == 0
            and self.duplicate_candidate_bug_count == 0
            and self.semantic_divergence_count == 0
            and self.resolved_semantic_divergence_count == 0
            and self.semantic_divergence_needs_confirmation_count == 0
            and self.false_positive_count == 0
            and self.needs_confirmation_count == 0
        ):
            return dict(EMPTY_FINDING_REWARD_SIGNALS)
        return {
            "candidate_bug": self.candidate_bug_count > 0,
            "candidate_bug_count": self.candidate_bug_count,
            "issue_replay_candidate_bug_count": self.issue_replay_candidate_bug_count,
            "known_saturated_candidate_bug_count": self.known_saturated_candidate_bug_count,
            "source_issue_candidate_bug_count": self.source_issue_candidate_bug_count,
            "nonfresh_lifecycle_candidate_bug_count": self.nonfresh_lifecycle_candidate_bug_count,
            "duplicate_candidate_bug_count": self.duplicate_candidate_bug_count,
            "semantic_divergence": self.semantic_divergence_count > 0,
            "semantic_divergence_count": self.semantic_divergence_count,
            "resolved_semantic_divergence_count": self.resolved_semantic_divergence_count,
            "semantic_divergence_needs_confirmation_count": (
                self.semantic_divergence_needs_confirmation_count
            ),
            "rewardable_semantic_divergence": self.semantic_divergence_needs_confirmation_count > 0,
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
        if verdict == "semantic_divergence_needs_confirmation":
            analysis.semantic_divergence_needs_confirmation_count += 1
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
        is_source_issue = has_source_issue(finding)
        lifecycle = finding_lifecycle(finding)
        if is_issue_replay:
            analysis.issue_replay_candidate_bug_count += 1
            if not root.startswith("metamorphic_"):
                analysis.issue_replay_candidate_bug_families[family_key] += 1
        if is_known_saturated:
            analysis.known_saturated_candidate_bug_count += 1
        if is_source_issue:
            analysis.source_issue_candidate_bug_count += 1
        if lifecycle in NON_FRESH_LIFECYCLE_STATES:
            analysis.nonfresh_lifecycle_candidate_bug_count += 1
            analysis.duplicate_candidate_bug_count += int(lifecycle == "duplicate")
        if is_issue_replay or is_known_saturated or is_source_issue or lifecycle in NON_FRESH_LIFECYCLE_STATES:
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
    signals = analysis.reward_signals()
    lane = evidence_lane_for_row(row)
    if lane in {"seeded_sensitivity", "replay", "known_regression"}:
        # Keep the raw observation counts in explicit lane fields, but prevent
        # historical/seeded/known evidence from entering fresh discovery reward
        # or report totals.
        signals["evidence_lane"] = lane
        signals["excluded_lane_candidate_bug_count"] = int(signals["candidate_bug_count"])
        signals["seeded_candidate_bug_count"] = (
            int(signals["candidate_bug_count"])
            if lane == "seeded_sensitivity"
            else 0
        )
        signals["candidate_bug"] = False
        signals["candidate_bug_count"] = 0
        signals["rewardable_semantic_divergence"] = False
    return signals


def evidence_lane_for_row(row: dict[str, Any]) -> str:
    source = str(row.get("candidate_source", "") or "").strip().lower()
    suite = str(row.get("target_suite", "") or "").strip().lower()
    metadata = row.get("metadata")
    metadata_text = ""
    if isinstance(metadata, dict):
        metadata_text = " ".join(
            str(metadata.get(key, "") or "")
            for key in ("source", "family_lifecycle", "generator_profile")
        ).lower()
    text = f"{source} {suite} {metadata_text}"
    if "seeded" in text or "fault_sensitivity" in text:
        return "seeded_sensitivity"
    if "known_regression" in text or "known replay" in text:
        return "known_regression"
    if "replay" in text or "historical" in text or "exact_reproducer" in text:
        return "replay"
    if "known" in text or "regression" in text:
        return "known_regression"
    if "feedback" in text:
        return "feedback"
    if "metamorphic" in text or "semantic" in text:
        return "semantic_metamorphic"
    return "fresh"


def reward_signals_have_rewardable_finding(signals: dict[str, Any]) -> bool:
    return bool(signals.get("candidate_bug_count", 0)) or bool(
        signals.get("rewardable_semantic_divergence", False)
    )


def row_has_rewardable_finding(
    row: dict[str, Any],
    known_saturated_bug_families: list[str] | tuple[str, ...] | None = None,
    *,
    finding_outcomes: FindingOutcomeAnalysis | None = None,
) -> bool:
    return reward_signals_have_rewardable_finding(
        row_reward_signals(
            row,
            known_saturated_bug_families=known_saturated_bug_families,
            finding_outcomes=finding_outcomes,
        )
    )


def row_has_rewardable_new_behavior(
    row: dict[str, Any],
    known_saturated_bug_families: list[str] | tuple[str, ...] | None = None,
    *,
    finding_outcomes: FindingOutcomeAnalysis | None = None,
) -> bool:
    if not bool(row.get("signal_new_behavior", row.get("is_new_behavior"))):
        return False
    findings = row.get("findings") or []
    if not findings:
        return True
    return row_has_rewardable_finding(
        row,
        known_saturated_bug_families=known_saturated_bug_families,
        finding_outcomes=finding_outcomes,
    )


def is_candidate_issue_finding(finding: dict[str, Any]) -> bool:
    return finding.get("triage_verdict") == CANDIDATE_BUG_VERDICT and not finding.get("false_positive")


def is_issue_replay_finding(finding: dict[str, Any]) -> bool:
    return str(finding.get("discovery_origin", "")).strip() == ISSUE_REPLAY_ORIGIN


def has_source_issue(finding: dict[str, Any]) -> bool:
    return bool(str(finding.get("source_issue", "")).strip())


def finding_lifecycle(finding: dict[str, Any]) -> str:
    raw = (
        finding.get("family_lifecycle")
        or finding.get("lifecycle_status")
        or finding.get("candidate_status")
        or finding.get("bug_status")
        or "novel"
    )
    return str(raw).strip().lower().replace("-", "_").replace(" ", "_") or "novel"


def backend_group_key(finding: dict[str, Any]) -> str:
    return ",".join(sorted(finding.get("suspicious_backends", []) or [])) or "unknown"


def candidate_issue_family_key(finding: dict[str, Any]) -> str:
    root = str(finding.get("root_cause", "unknown"))
    backend_group = backend_group_key(finding)
    return f"{root}@{backend_group}"


def is_known_saturated_candidate_issue_finding(
    finding: dict[str, Any],
    known_saturated_bug_families: list[str] | tuple[str, ...] | None = None,
) -> bool:
    if not is_candidate_issue_finding(finding):
        return False
    known_families = known_saturated_bug_families or ()
    return family_key_matches_known_family(candidate_issue_family_key(finding), known_families)


def is_source_issue_candidate_issue_finding(finding: dict[str, Any]) -> bool:
    return is_candidate_issue_finding(finding) and has_source_issue(finding)


def is_rewardable_candidate_issue_finding(
    finding: dict[str, Any],
    known_saturated_bug_families: list[str] | tuple[str, ...] | None = None,
) -> bool:
    return (
        is_candidate_issue_finding(finding)
        and not is_issue_replay_finding(finding)
        and not has_source_issue(finding)
        and finding_lifecycle(finding) not in NON_FRESH_LIFECYCLE_STATES
        and not is_known_saturated_candidate_issue_finding(finding, known_saturated_bug_families)
    )


def is_semantic_divergence_finding(finding: dict[str, Any]) -> bool:
    return str(finding.get("triage_verdict", "unclassified")) in SEMANTIC_DIVERGENCE_VERDICTS


def is_resolved_semantic_divergence_finding(finding: dict[str, Any]) -> bool:
    return str(finding.get("triage_verdict", "unclassified")) in RESOLVED_SEMANTIC_DIVERGENCE_VERDICTS


def is_false_positive_finding(finding: dict[str, Any]) -> bool:
    return bool(finding.get("false_positive")) or str(
        finding.get("triage_verdict", "unclassified")
    ) in FALSE_POSITIVE_VERDICTS


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
            or has_source_issue(finding)
            or finding_lifecycle(finding) in NON_FRESH_LIFECYCLE_STATES
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


is_candidate_bug_finding = is_candidate_issue_finding
candidate_bug_family_key = candidate_issue_family_key
is_known_saturated_candidate_bug_finding = is_known_saturated_candidate_issue_finding
is_source_issue_candidate_bug_finding = is_source_issue_candidate_issue_finding
is_rewardable_candidate_bug_finding = is_rewardable_candidate_issue_finding
candidate_bug_family_keys = candidate_issue_family_keys
issue_replay_candidate_bug_family_keys = issue_replay_candidate_issue_family_keys
candidate_bug_signatures = candidate_issue_signatures
