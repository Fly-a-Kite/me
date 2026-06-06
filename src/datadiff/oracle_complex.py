from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from datadiff.oracle import Finding


@dataclass(frozen=True, slots=True)
class OracleCrossValidationSummary:
    differential_count: int = 0
    metamorphic_count: int = 0
    cross_validated_count: int = 0
    differential_needs_recheck_count: int = 0
    metamorphic_only_count: int = 0
    corroborated_backends: tuple[str, ...] = ()
    relation_support: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "differential_count": self.differential_count,
            "metamorphic_count": self.metamorphic_count,
            "cross_validated_count": self.cross_validated_count,
            "differential_needs_recheck_count": self.differential_needs_recheck_count,
            "metamorphic_only_count": self.metamorphic_only_count,
            "corroborated_backends": list(self.corroborated_backends),
            "relation_support": {
                relation: list(backends)
                for relation, backends in sorted(self.relation_support.items())
            },
        }


def cross_validate_oracle_findings(
    differential_findings: list[Finding],
    metamorphic_findings: list[Finding],
) -> OracleCrossValidationSummary:
    relation_support: dict[str, set[str]] = {}
    corroborated_backends: set[str] = set()
    cross_validated = 0
    needs_recheck = 0

    for finding in metamorphic_findings:
        relation = _metamorphic_relation(finding)
        for backend in _backend_set(finding):
            relation_support.setdefault(relation, set()).add(backend)

    for finding in differential_findings:
        matching = _corroborating_metamorphic_findings(finding, metamorphic_findings)
        if matching:
            cross_validated += 1
            matched_backends = sorted(
                _backend_set(finding).intersection(
                    backend
                    for match in matching
                    for backend in _backend_set(match)
                )
            )
            corroborated_backends.update(matched_backends)
            _merge_finding_adjudication(
                finding,
                cross_validated=True,
                metamorphic_support="corroborated",
                needs_recheck=False,
                corroborated_backends=matched_backends,
                corroborating_relations=sorted({_metamorphic_relation(match) for match in matching}),
            )
            finding.confidence = "high"
        else:
            needs_recheck += 1
            _merge_finding_adjudication(
                finding,
                cross_validated=False,
                metamorphic_support="not_observed",
                needs_recheck=True,
            )

    covered_metamorphic: set[str] = {
        finding.finding_id
        for differential in differential_findings
        for finding in _corroborating_metamorphic_findings(differential, metamorphic_findings)
    }
    metamorphic_only = 0
    for finding in metamorphic_findings:
        if finding.finding_id in covered_metamorphic:
            _merge_finding_adjudication(
                finding,
                cross_validated=True,
                metamorphic_support="corroborates_differential",
                needs_recheck=False,
            )
            continue
        metamorphic_only += 1
        finding.discovery_origin = "metamorphic_only"
        if finding.confidence not in {"high", "medium"}:
            finding.confidence = "low"
        _merge_finding_adjudication(
            finding,
            cross_validated=False,
            metamorphic_support="metamorphic_only",
            needs_recheck=True,
        )

    return OracleCrossValidationSummary(
        differential_count=len(differential_findings),
        metamorphic_count=len(metamorphic_findings),
        cross_validated_count=cross_validated,
        differential_needs_recheck_count=needs_recheck,
        metamorphic_only_count=metamorphic_only,
        corroborated_backends=tuple(sorted(corroborated_backends)),
        relation_support={
            relation: sorted(backends)
            for relation, backends in relation_support.items()
        },
    )


def _corroborating_metamorphic_findings(
    finding: Finding,
    metamorphic_findings: list[Finding],
) -> list[Finding]:
    finding_backends = _backend_set(finding)
    if not finding_backends:
        return []
    root = _root_family(finding.root_cause)
    out: list[Finding] = []
    for candidate in metamorphic_findings:
        candidate_backends = _backend_set(candidate)
        if not finding_backends.intersection(candidate_backends):
            continue
        candidate_root = _root_family(candidate.root_cause)
        if root and candidate_root and root != candidate_root:
            relation = _metamorphic_relation(candidate)
            if root not in relation and relation not in root:
                continue
        out.append(candidate)
    return out


def _backend_set(finding: Finding) -> set[str]:
    return {str(backend) for backend in finding.suspicious_backends if str(backend)}


def _metamorphic_relation(finding: Finding) -> str:
    root = str(finding.root_cause or "")
    if root.startswith("metamorphic_"):
        return root[len("metamorphic_") :]
    kind = str(finding.kind or "")
    if kind.startswith("metamorphic_") and kind.endswith("_violation"):
        return kind[len("metamorphic_") : -len("_violation")]
    return root or kind or "unknown"


def _root_family(root_cause: str) -> str:
    root = str(root_cause or "")
    if root.startswith("metamorphic_"):
        root = root[len("metamorphic_") :]
    return root


def _merge_finding_adjudication(
    finding: Finding,
    *,
    cross_validated: bool,
    metamorphic_support: str,
    needs_recheck: bool,
    corroborated_backends: list[str] | None = None,
    corroborating_relations: list[str] | None = None,
) -> None:
    adjudication = dict(finding.adjudication or {})
    oracle_complex = dict(adjudication.get("oracle_complex", {}) or {})
    oracle_complex.update(
        {
            "cross_validated": bool(cross_validated),
            "needs_recheck": bool(needs_recheck),
            "corroborated_backends": list(corroborated_backends or []),
            "corroborating_relations": list(corroborating_relations or []),
        }
    )
    adjudication["oracle_complex"] = oracle_complex
    adjudication["metamorphic_support"] = metamorphic_support
    if needs_recheck:
        adjudication["recheck_status"] = "needed"
    finding.adjudication = adjudication
