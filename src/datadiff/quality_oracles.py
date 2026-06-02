from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from datadiff.dsl import Case
from datadiff.guidance import derive_case_features


@dataclass(slots=True)
class QualityOracleResult:
    name: str
    verdict: str
    passed: bool
    score: float
    evidence: str
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_quality_oracles(
    case: Case,
    row: dict[str, Any],
    *,
    candidate_source: str,
    preflight: dict[str, Any],
    guidance_decision: dict[str, Any],
    guidance_strategy: str,
    guidance_targets: list[str],
) -> list[QualityOracleResult]:
    return [
        mutation_oracle(case, row, candidate_source=candidate_source, preflight=preflight),
        feedback_oracle(row),
        guidance_oracle(
            case,
            row,
            guidance_decision=guidance_decision,
            guidance_strategy=guidance_strategy,
            guidance_targets=guidance_targets,
        ),
    ]


def mutation_oracle(
    case: Case,
    row: dict[str, Any],
    *,
    candidate_source: str,
    preflight: dict[str, Any],
) -> QualityOracleResult:
    if candidate_source != "feedback_mutation":
        return QualityOracleResult(
            name="mutation",
            verdict="not_applicable",
            passed=True,
            score=0.0,
            evidence="Case was generated directly, not selected from feedback mutation.",
            metrics={"candidate_source": candidate_source},
        )

    signal_new_behavior = bool(row.get("signal_new_behavior", row.get("is_new_behavior")))
    valid = bool(preflight.get("valid", True))
    productive = bool(row.get("findings") or signal_new_behavior)
    fallback_used = bool(preflight.get("fallback_used", False))
    passed = valid and productive and not fallback_used
    if passed:
        verdict = "productive_mutation"
        evidence = "Feedback mutation produced a valid case with new behavior or a finding."
    elif not valid or fallback_used:
        verdict = "invalid_mutation_repaired"
        evidence = "Feedback mutation required fallback repair before execution."
    else:
        verdict = "redundant_mutation"
        evidence = "Feedback mutation executed but did not add new behavior or findings."
    score = (1.0 if row.get("findings") else 0.0) + (0.5 if signal_new_behavior else 0.0)
    if not valid or fallback_used:
        score -= 0.5
    return QualityOracleResult(
        name="mutation",
        verdict=verdict,
        passed=passed,
        score=score,
        evidence=evidence,
        metrics={
            "candidate_source": candidate_source,
            "case_id": case.case_id,
            "preflight_valid": valid,
            "preflight_repaired": bool(preflight.get("repaired", False)),
            "preflight_fallback_used": fallback_used,
            "new_behavior": signal_new_behavior,
            "findings": len(row.get("findings", [])),
        },
    )


def feedback_oracle(row: dict[str, Any]) -> QualityOracleResult:
    has_finding = bool(row.get("findings"))
    new_behavior = bool(row.get("signal_new_behavior", row.get("is_new_behavior")))
    stored = bool(row.get("stored_in_feedback_corpus"))
    passed = has_finding or new_behavior
    if has_finding:
        verdict = "finding_yield"
        evidence = "Case produced one or more findings and should influence later mutation."
    elif new_behavior:
        verdict = "new_behavior_yield"
        evidence = "Case expanded behavior coverage."
    else:
        verdict = "redundant_behavior"
        evidence = "Case repeated known behavior and produced no finding."
    return QualityOracleResult(
        name="feedback",
        verdict=verdict,
        passed=passed,
        score=(1.0 if has_finding else 0.0) + (0.5 if new_behavior else 0.0),
        evidence=evidence,
        metrics={
            "new_behavior": new_behavior,
            "findings": len(row.get("findings", [])),
            "stored_in_feedback_corpus": stored,
            "behavior_signature": row.get("behavior_signature", ""),
        },
    )


def guidance_oracle(
    case: Case,
    row: dict[str, Any],
    *,
    guidance_decision: dict[str, Any],
    guidance_strategy: str,
    guidance_targets: list[str],
) -> QualityOracleResult:
    if guidance_strategy != "guided":
        return QualityOracleResult(
            name="guidance",
            verdict="not_applicable",
            passed=True,
            score=0.0,
            evidence="Guidance strategy is random.",
            metrics={"strategy": guidance_strategy},
        )

    signal_new_behavior = bool(row.get("signal_new_behavior", row.get("is_new_behavior")))
    matched_targets = list(guidance_decision.get("matched_targets", []))
    features = derive_case_features(case)
    target_hit = bool(matched_targets) if guidance_targets else True
    productive = bool(row.get("findings") or signal_new_behavior)
    passed = target_hit and productive
    if passed:
        verdict = "guided_productive"
        evidence = "Guidance selected a target-matching case that yielded new behavior or a finding."
    elif not target_hit:
        verdict = "guided_target_miss"
        evidence = "Guidance selected a case that did not match configured targets."
    else:
        verdict = "guided_redundant"
        evidence = "Guidance matched targets but did not produce new behavior or findings."
    return QualityOracleResult(
        name="guidance",
        verdict=verdict,
        passed=passed,
        score=float(guidance_decision.get("score", 0.0)) + (1.0 if productive else 0.0),
        evidence=evidence,
        metrics={
            "strategy": guidance_strategy,
            "configured_targets": guidance_targets,
            "matched_targets": matched_targets,
            "candidate_count": int(guidance_decision.get("candidate_count", 1)),
            "contributing_candidate_count": int(
                guidance_decision.get("contributing_candidate_count", guidance_decision.get("candidate_count", 1))
            ),
            "pruned_candidate_count": int(guidance_decision.get("pruned_candidate_count", 0)),
            "frontier_bucket_count": len(guidance_decision.get("frontier_buckets", [])),
            "discovery_bucket_count": len(guidance_decision.get("discovery_buckets", [])),
            "frontier_conformance": float(guidance_decision.get("score_breakdown", {}).get("frontier_conformance", 0.0)),
            "discovery_diversity_bonus": float(
                guidance_decision.get("score_breakdown", {}).get("discovery_diversity_bonus", 0.0)
            ),
            "candidate_pool_diversity_bonus": float(
                guidance_decision.get("score_breakdown", {}).get("candidate_pool_diversity_bonus", 0.0)
            ),
            "discovery_stale_penalty": float(
                guidance_decision.get("score_breakdown", {}).get("discovery_stale_penalty", 0.0)
            ),
            "contribution_potential": float(
                guidance_decision.get("score_breakdown", {}).get("contribution_potential", 0.0)
            ),
            "feature_count": len(features),
            "new_behavior": signal_new_behavior,
            "findings": len(row.get("findings", [])),
        },
    )
