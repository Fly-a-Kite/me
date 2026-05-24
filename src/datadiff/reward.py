from __future__ import annotations

from collections import Counter
from typing import Any

CANDIDATE_BUG_VERDICT = "candidate_implementation_bug"
ISSUE_REPLAY_ORIGIN = "issue_replay"
SEMANTIC_DIVERGENCE_VERDICTS = {
    "documented_semantic_divergence",
    "expected_semantic_divergence",
    "semantic_divergence_needs_confirmation",
}
FALSE_POSITIVE_VERDICTS = {
    "generator_false_positive",
    "normalizer_false_positive",
}
NEEDS_CONFIRMATION_VERDICTS = {
    "needs_manual_confirmation",
    "semantic_divergence_needs_confirmation",
}


def is_candidate_bug_finding(finding: dict[str, Any]) -> bool:
    return finding.get("triage_verdict") == CANDIDATE_BUG_VERDICT and not finding.get("false_positive")


def is_issue_replay_finding(finding: dict[str, Any]) -> bool:
    return str(finding.get("discovery_origin", "")).strip() == ISSUE_REPLAY_ORIGIN


def candidate_bug_family_key(finding: dict[str, Any]) -> str:
    root = str(finding.get("root_cause", "unknown"))
    suspicious = suspicious_key(finding)
    return f"{root}@{suspicious}"


def family_key_matches_known_family(candidate_family: str, known_bug_families: list[str] | tuple[str, ...]) -> bool:
    candidate_root, candidate_backends = _split_family_key(candidate_family)
    for known_family in known_bug_families:
        known_root, known_backends = _split_family_key(known_family)
        if known_root != candidate_root:
            continue
        if not known_backends or not candidate_backends or known_backends & candidate_backends:
            return True
    return False


def is_known_saturated_candidate_bug_finding(
    finding: dict[str, Any],
    known_saturated_bug_families: list[str] | tuple[str, ...] | None = None,
) -> bool:
    if not is_candidate_bug_finding(finding):
        return False
    known_families = known_saturated_bug_families or ()
    return family_key_matches_known_family(candidate_bug_family_key(finding), known_families)


def is_rewardable_candidate_bug_finding(
    finding: dict[str, Any],
    known_saturated_bug_families: list[str] | tuple[str, ...] | None = None,
) -> bool:
    return (
        is_candidate_bug_finding(finding)
        and not is_issue_replay_finding(finding)
        and not is_known_saturated_candidate_bug_finding(finding, known_saturated_bug_families)
    )


def is_semantic_divergence_finding(finding: dict[str, Any]) -> bool:
    return str(finding.get("triage_verdict", "unclassified")) in SEMANTIC_DIVERGENCE_VERDICTS


def is_false_positive_finding(finding: dict[str, Any]) -> bool:
    return bool(finding.get("false_positive")) or str(finding.get("triage_verdict", "unclassified")) in FALSE_POSITIVE_VERDICTS


def suspicious_key(finding: dict[str, Any]) -> str:
    return ",".join(sorted(finding.get("suspicious_backends", []) or [])) or "unknown"


def candidate_bug_family_keys(
    findings: list[dict[str, Any]],
    known_saturated_bug_families: list[str] | tuple[str, ...] | None = None,
) -> Counter[str]:
    keys: Counter[str] = Counter()
    root_by_suspicious: dict[str, str] = {}
    for finding in findings:
        if not is_rewardable_candidate_bug_finding(finding, known_saturated_bug_families):
            continue
        root = str(finding.get("root_cause", "unknown"))
        if root.startswith("metamorphic_"):
            continue
        root_by_suspicious.setdefault(suspicious_key(finding), root)
    for finding in findings:
        if not is_rewardable_candidate_bug_finding(finding, known_saturated_bug_families):
            continue
        root = str(finding.get("root_cause", "unknown"))
        suspicious = suspicious_key(finding)
        if root.startswith("metamorphic_") and suspicious in root_by_suspicious:
            root = root_by_suspicious[suspicious]
        keys[f"{root}@{suspicious}"] += 1
    return keys


def issue_replay_candidate_bug_family_keys(findings: list[dict[str, Any]]) -> Counter[str]:
    keys: Counter[str] = Counter()
    for finding in findings:
        if not is_candidate_bug_finding(finding) or not is_issue_replay_finding(finding):
            continue
        root = str(finding.get("root_cause", "unknown"))
        if root.startswith("metamorphic_"):
            continue
        keys[f"{root}@{suspicious_key(finding)}"] += 1
    return keys


def candidate_bug_signatures(
    findings: list[dict[str, Any]],
    known_saturated_bug_families: list[str] | tuple[str, ...] | None = None,
) -> Counter[str]:
    signatures: Counter[str] = Counter()
    for finding in findings:
        if not is_rewardable_candidate_bug_finding(finding, known_saturated_bug_families):
            continue
        signature = str(finding.get("signature", "")).strip()
        if signature:
            signatures[signature] += 1
    return signatures


def row_reward_signals(
    row: dict[str, Any],
    known_saturated_bug_families: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    findings = row.get("findings") or []
    candidate_bug_count = sum(
        1 for finding in findings if is_rewardable_candidate_bug_finding(finding, known_saturated_bug_families)
    )
    issue_replay_candidate_bug_count = sum(
        1 for finding in findings if is_candidate_bug_finding(finding) and is_issue_replay_finding(finding)
    )
    known_saturated_candidate_bug_count = sum(
        1
        for finding in findings
        if is_known_saturated_candidate_bug_finding(finding, known_saturated_bug_families)
    )
    semantic_divergence_count = sum(1 for finding in findings if is_semantic_divergence_finding(finding))
    false_positive_count = sum(1 for finding in findings if is_false_positive_finding(finding))
    needs_confirmation_count = sum(
        1
        for finding in findings
        if str(finding.get("triage_verdict", "unclassified")) in NEEDS_CONFIRMATION_VERDICTS
    )
    return {
        "candidate_bug": candidate_bug_count > 0,
        "candidate_bug_count": candidate_bug_count,
        "issue_replay_candidate_bug_count": issue_replay_candidate_bug_count,
        "known_saturated_candidate_bug_count": known_saturated_candidate_bug_count,
        "semantic_divergence": semantic_divergence_count > 0,
        "semantic_divergence_count": semantic_divergence_count,
        "false_positive": false_positive_count > 0,
        "false_positive_count": false_positive_count,
        "needs_confirmation": needs_confirmation_count > 0,
        "needs_confirmation_count": needs_confirmation_count,
    }


def online_case_reward(
    row: dict[str, Any],
    known_saturated_bug_families: list[str] | tuple[str, ...] | None = None,
) -> float:
    signals = row_reward_signals(row, known_saturated_bug_families=known_saturated_bug_families)
    preflight = row.get("preflight") or {}
    reward = (
        4.0 * signals["candidate_bug_count"]
        + 0.20 * signals["semantic_divergence_count"]
        + (0.5 if row.get("is_new_behavior") else 0.0)
        - 0.25 * signals["needs_confirmation_count"]
        - 2.5 * signals["false_positive_count"]
    )
    if not bool(preflight.get("valid", True)) or bool(preflight.get("fallback_used", False)):
        reward -= 0.5
    if reward == 0.0:
        reward -= 0.1
    return reward


def _split_family_key(family_key: str) -> tuple[str, set[str]]:
    root, _, backend_part = str(family_key).partition("@")
    backends = {backend.strip() for backend in backend_part.split(",") if backend.strip()}
    return root.strip(), backends
