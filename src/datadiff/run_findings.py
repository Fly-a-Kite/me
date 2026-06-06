from __future__ import annotations

from typing import Any

from datadiff.adjudication import build_adjudication, counts_as_bug_evidence
from datadiff.config import ExperimentConfig
from datadiff.oracle import Finding


def _finding_recheck_key(finding: dict[str, Any]) -> tuple[str, str, tuple[str, ...], str]:
    return (
        str(finding.get("kind", "")),
        str(finding.get("root_cause", "unknown")),
        tuple(sorted(str(backend) for backend in finding.get("suspicious_backends", []) or [])),
        str(finding.get("mismatch_class", "")),
    )


def _format_recheck_key(key: tuple[str, str, tuple[str, ...], str]) -> str:
    kind, root, backends, mismatch = key
    backend_part = ",".join(backends) or "unknown"
    suffix = f":{mismatch}" if mismatch else ""
    return f"{kind}:{root}@{backend_part}{suffix}"


def _mark_finding_non_reproducible(finding: Finding, attempts: int) -> None:
    finding.triage_verdict = "non_reproducible_candidate"
    finding.paper_status = "exclude_unreproducible_candidate"
    finding.triage_confidence = "high"
    finding.false_positive = True
    finding.false_positive_reason = "candidate_not_reproduced_on_immediate_recheck"
    finding.triage_evidence = (
        f"Initial finding did not reproduce in {attempts} immediate fresh recheck run(s); "
        "exclude it from latest-version bug evidence until a stable reproducer exists."
    )
    finding.adjudication = build_adjudication(
        "non_reproducible_candidate",
        validity_gate="recheck_failed",
        semantic_gate="unknown",
        attribution_gate="reproduction_failed",
        recheck_status="failed",
        exclusion_reason="candidate_not_reproduced_on_immediate_recheck",
        needs_manual_review=False,
    )


def _countable_row_findings(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        finding
        for finding in row.get("findings", []) or []
        if _is_countable_finding_dict(finding)
    ]


def _countable_finding_objects(findings: list[Finding]) -> list[Finding]:
    return [finding for finding in findings if _is_countable_finding_dict(finding.to_dict())]


def _is_countable_finding_dict(finding: dict[str, Any]) -> bool:
    return counts_as_bug_evidence(finding)


def _artifact_budget_available(config: ExperimentConfig, saved_count: int) -> bool:
    if not config.enable_artifact:
        return False
    if config.artifact_limit is None:
        return True
    return saved_count < max(0, int(config.artifact_limit))
