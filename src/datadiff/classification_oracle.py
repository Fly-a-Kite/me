from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from datadiff.adjudication import build_adjudication
from datadiff.canonicalization import compare_result_against_anchor
from datadiff.case_features import (
    case_contains_non_ascii_string as _case_contains_non_ascii_string,
    case_contains_special_float as _case_contains_special_float,
    case_has_null_filter_literal as _case_has_null_filter_literal,
    case_uses_modulo as _case_uses_modulo,
    case_uses_unicode_case_mapping as _case_uses_unicode_case_mapping,
)
from datadiff.case_policy import case_discovery_origin, primary_source_issue
from datadiff.case_validation import validate_case_program
from datadiff.classification_signals import (
    all_backends_rejected_due_to_generated_invalidity as _all_backends_rejected_due_to_generated_invalidity,
    has_clear_minority_backend as _has_clear_minority_backend,
    has_limit_or_offset_without_defined_order as _has_limit_or_offset_without_defined_order,
    is_float_precision_boundary_mismatch as _is_float_precision_boundary_mismatch,
    is_order_only_mismatch as _is_order_only_mismatch,
    is_pyarrow_empty_global_bool_aggregate_adapter_error as _is_pyarrow_empty_global_bool_aggregate_adapter_error,
    is_sort_tie_order_only_mismatch as _is_sort_tie_order_only_mismatch,
    is_sort_topk_tie_cutoff_mismatch as _is_sort_topk_tie_cutoff_mismatch,
    join_keys_contain_null as _join_keys_contain_null,
    normalizer_errors as _normalizer_errors,
)
from datadiff.dsl import Case
from datadiff.normalizer import NormalizedResult
from datadiff.operation_semantics import has_order_observer
from datadiff.oracle import Finding
from datadiff.reference_semantics import reference_result
from datadiff.semantic_boundaries import (
    SemanticBoundaryMatch,
    build_documented_semantic_rules,
    build_semantic_boundary_rules,
    matching_semantic_rules,
    ordered_rules_from_snapshot,
    semantic_boundary_reasons,
    semantic_rule_records,
)
from datadiff.dynamic_strategy import StrategyRuleRecord


@dataclass(slots=True)
class Classification:
    verdict: str
    paper_status: str
    confidence: str
    false_positive: bool = False
    false_positive_reason: str = ""
    evidence: str = ""
    recommendation: list[str] = field(default_factory=list)
    documentation_refs: list[dict[str, str]] = field(default_factory=list)
    implicated_backends: list[str] = field(default_factory=list)
    adjudication: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.adjudication:
            self.adjudication = build_adjudication(
                self.verdict,
                exclusion_reason=self.false_positive_reason or None,
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _classification(
    verdict: str,
    paper_status: str,
    confidence: str,
    *,
    false_positive: bool = False,
    false_positive_reason: str = "",
    evidence: str = "",
    recommendation: list[str] | None = None,
    documentation_refs: list[dict[str, str]] | None = None,
    implicated_backends: list[str] | None = None,
    adjudication: dict[str, Any] | None = None,
) -> Classification:
    return Classification(
        verdict=verdict,
        paper_status=paper_status,
        confidence=confidence,
        false_positive=false_positive,
        false_positive_reason=false_positive_reason,
        evidence=evidence,
        recommendation=list(recommendation or []),
        documentation_refs=list(documentation_refs or []),
        implicated_backends=list(implicated_backends or []),
        adjudication=adjudication or build_adjudication(
            verdict,
            exclusion_reason=false_positive_reason or None,
        ),
    )


def annotate_findings(
    case: Case,
    findings: list[Finding],
    normalized: dict[str, NormalizedResult],
    raw_results: dict[str, dict[str, Any]],
    config: dict[str, Any],
    backends: list[str],
) -> None:
    discovery_origin = _discovery_origin(case)
    source_issue = _source_issue(case)
    for finding in findings:
        classification = classify_finding(case, finding, normalized, raw_results, config, backends)
        finding.triage_verdict = classification.verdict
        finding.paper_status = classification.paper_status
        finding.triage_confidence = classification.confidence
        finding.false_positive = classification.false_positive
        finding.false_positive_reason = classification.false_positive_reason
        finding.triage_evidence = classification.evidence
        finding.recommendation = classification.recommendation
        finding.documentation_refs = classification.documentation_refs
        finding.adjudication = classification.adjudication
        finding.discovery_origin = discovery_origin
        finding.source_issue = source_issue
        if classification.implicated_backends:
            finding.suspicious_backends = classification.implicated_backends


def classify_finding(
    case: Case,
    finding: Finding | dict[str, Any],
    normalized: dict[str, NormalizedResult | dict[str, Any]],
    raw_results: dict[str, dict[str, Any]],
    config: dict[str, Any],
    backends: list[str],
) -> Classification:
    validity_errors = validate_case_program(case)
    if validity_errors:
        return _classification(
            "generator_false_positive",
            "exclude_generator_invalid_case",
            "high",
            false_positive=True,
            false_positive_reason="invalid_generated_program",
            evidence="; ".join(validity_errors[:3]),
            recommendation=[
                "Do not count this finding as a backend bug.",
                "Fix the generator/repair logic so it emits only executable DSL programs.",
            ],
            adjudication=build_adjudication(
                "generator_false_positive",
                validity_gate="invalid_generated_case",
                semantic_gate="out_of_scope",
                attribution_gate="generator_fault",
                exclusion_reason="invalid_generated_program",
            ),
        )

    normalizer_errors = _normalizer_errors(normalized)
    if normalizer_errors:
        return _classification(
            "normalizer_false_positive",
            "exclude_normalizer_failure",
            "high",
            false_positive=True,
            false_positive_reason="normalization_error",
            evidence=f"Normalizer failed for: {normalizer_errors}",
            recommendation=[
                "Do not count this finding as a backend bug.",
                "Fix normalization or rerun with the raw backend outputs before triage.",
            ],
            adjudication=build_adjudication(
                "normalizer_false_positive",
                validity_gate="harness_failure",
                semantic_gate="out_of_scope",
                attribution_gate="normalizer_or_adapter_fault",
                exclusion_reason="normalization_error",
            ),
        )

    if _all_backends_rejected_due_to_generated_invalidity(raw_results):
        return _classification(
            "generator_false_positive",
            "exclude_generator_invalid_case",
            "medium",
            false_positive=True,
            false_positive_reason="all_backends_rejected_generated_case",
            evidence="All backends rejected with schema/name/type errors, indicating an invalid generated DSL program.",
            recommendation=[
                "Do not count this finding as a backend bug.",
                "Minimize the case and add a generator regression test.",
            ],
            adjudication=build_adjudication(
                "generator_false_positive",
                validity_gate="invalid_generated_case",
                semantic_gate="out_of_scope",
                attribution_gate="generator_fault",
                exclusion_reason="all_backends_rejected_generated_case",
            ),
        )

    if _is_pyarrow_empty_global_bool_aggregate_adapter_error(case, finding, normalized, raw_results):
        return _classification(
            "normalizer_false_positive",
            "exclude_harness_adapter_failure",
            "high",
            false_positive=True,
            false_positive_reason="pyarrow_empty_global_bool_aggregate_adapter_error",
            evidence=(
                "Stored evidence matches an older DataDiffFuzz PyArrow adapter failure for an empty-input "
                "global bool any/all aggregate; rerun the case with the current adapter before triage."
            ),
            recommendation=[
                "Do not count stale stored evidence for this signature as a backend bug.",
                "Regenerate or rerun the case with the current PyArrow adapter before creating issue evidence.",
            ],
            adjudication=build_adjudication(
                "normalizer_false_positive",
                validity_gate="harness_failure",
                semantic_gate="out_of_scope",
                attribution_gate="normalizer_or_adapter_fault",
                exclusion_reason="pyarrow_empty_global_bool_aggregate_adapter_error",
            ),
        )

    if _is_order_only_mismatch(normalized):
        if not case.program.order_sensitive:
            return _classification(
                "normalizer_false_positive",
                "exclude_normalizer_failure",
                "high",
                false_positive=True,
                false_positive_reason="order_only_normalization_mismatch",
                evidence="Backends returned the same row multiset, but normalized rows are ordered differently.",
                recommendation=[
                    "Do not count this as a backend bug.",
                    "Fix canonical row ordering or compare normalized outputs as bags for this oracle.",
                ],
                adjudication=build_adjudication(
                    "normalizer_false_positive",
                    validity_gate="harness_failure",
                    semantic_gate="out_of_scope",
                    attribution_gate="normalizer_or_adapter_fault",
                    exclusion_reason="order_only_normalization_mismatch",
                ),
            )
        if _is_sort_tie_order_only_mismatch(case):
            return _classification(
                "normalizer_false_positive",
                "exclude_order_underconstrained_case",
                "high",
                false_positive=True,
                false_positive_reason="sort_tie_order_underconstrained",
                evidence=(
                    "Backends returned the same row multiset, but the observable order depends on "
                    "tie ordering for non-unique sort keys."
                ),
                recommendation=[
                    "Do not count this as a backend bug.",
                    "Regenerate the case with explicit tie-breaker sort keys before using it as latest-version evidence.",
                ],
                adjudication=build_adjudication(
                    "normalizer_false_positive",
                    validity_gate="underconstrained_case",
                    semantic_gate="out_of_scope",
                    attribution_gate="ordering_underconstrained",
                    exclusion_reason="sort_tie_order_underconstrained",
                ),
            )

    if _has_limit_or_offset_without_defined_order(case):
        return _classification(
            "generator_false_positive",
            "exclude_order_underconstrained_case",
            "high",
            false_positive=True,
            false_positive_reason="limit_offset_order_underconstrained",
            evidence=(
                "The case applies LIMIT/OFFSET before any portable order-defining operation, "
                "or after an operation that invalidates row order."
            ),
            recommendation=[
                "Do not count this finding as a backend bug.",
                "Regenerate the case with an explicit sort or window order before LIMIT/OFFSET.",
            ],
            adjudication=build_adjudication(
                "generator_false_positive",
                validity_gate="underconstrained_case",
                semantic_gate="out_of_scope",
                attribution_gate="generator_fault",
                exclusion_reason="limit_offset_order_underconstrained",
            ),
        )

    if _is_sort_topk_tie_cutoff_mismatch(case):
        return _classification(
            "normalizer_false_positive",
            "exclude_order_underconstrained_case",
            "high",
            false_positive=True,
            false_positive_reason="sort_tie_order_underconstrained",
            evidence=(
                "The case applies LIMIT/OFFSET over a sort whose key is not unique at the "
                "cutoff boundary, so multiple output row sets are valid."
            ),
            recommendation=[
                "Do not count this as a backend bug.",
                "Regenerate the case with an explicit tie-breaker sort key before using it as latest-version evidence.",
            ],
            adjudication=build_adjudication(
                "normalizer_false_positive",
                validity_gate="underconstrained_case",
                semantic_gate="out_of_scope",
                attribution_gate="ordering_underconstrained",
                exclusion_reason="sort_tie_order_underconstrained",
            ),
        )

    documented_matches = _documented_semantic_matches(case, finding, config)
    if documented_matches:
        return _classification(
            "documented_semantic_divergence",
            "valid_finding_not_bug",
            "high",
            evidence="Finding matches a known/documented semantic boundary.",
            recommendation=[
                "Keep as a semantic-divergence benchmark finding.",
                "Do not count as a confirmed implementation bug.",
            ],
            documentation_refs=_documentation_refs(case, finding),
            adjudication=build_adjudication(
                "documented_semantic_divergence",
                validity_gate="valid_case",
                semantic_gate="documented_boundary",
                attribution_gate="documented_semantics",
                documentation_support="documented",
                countable_as_valid_finding=True,
                needs_manual_review=False,
                documented_rule_ids=[match.rule_id for match in documented_matches],
                boundary_rule_ids=[match.rule_id for match in documented_matches],
            ),
        )

    if _is_float_precision_boundary_mismatch(case, normalized):
        return _classification(
            "expected_semantic_divergence",
            "valid_finding_not_bug",
            "high",
            evidence=(
                "Backends differ only in full-precision floating-point arithmetic or aggregate "
                "ordering; the relaxed numeric row multisets agree."
            ),
            recommendation=[
                "Do not count this finding as an implementation bug.",
                "Use exact decimal inputs or backend-specific numeric semantics before making a bug claim.",
            ],
            adjudication=build_adjudication(
                "expected_semantic_divergence",
                validity_gate="valid_case",
                semantic_gate="expected_boundary",
                attribution_gate="numeric_precision_boundary",
                reference_support="not_used",
                countable_as_valid_finding=True,
                needs_manual_review=False,
                boundary_rule_ids=["boundary:float_precision_relaxed_match"],
            ),
        )

    metamorphic_classification = _metamorphic_classification(case, finding, backends)
    if metamorphic_classification is not None:
        return metamorphic_classification

    semantic_boundary_matches = _semantic_boundary_matches(case, finding, config)
    semantic_boundary_reasons = [match.reason for match in semantic_boundary_matches]
    if semantic_boundary_reasons and str(_get(finding, "root_cause", "")) != "ordering_or_limit":
        return _classification(
            "expected_semantic_divergence",
            "valid_finding_not_bug",
            "medium",
            evidence="; ".join(semantic_boundary_reasons),
            recommendation=[
                "Keep as a valid semantic-divergence finding.",
                "Do not count as an implementation bug unless a backend-specific specification is contradicted.",
                "Use a separate boundary-semantics experiment for this class.",
            ],
            adjudication=build_adjudication(
                "expected_semantic_divergence",
                validity_gate="valid_case",
                semantic_gate="expected_boundary",
                attribution_gate="semantic_boundary",
                countable_as_valid_finding=True,
                needs_manual_review=False,
                boundary_rule_ids=[match.rule_id for match in semantic_boundary_matches],
            ),
        )

    reference_classification = _reference_classification(case, finding, normalized, backends)
    if reference_classification is not None:
        return reference_classification

    if semantic_boundary_reasons:
        return _classification(
            "expected_semantic_divergence",
            "valid_finding_not_bug",
            "medium",
            evidence="; ".join(semantic_boundary_reasons),
            recommendation=[
                "Keep as a valid semantic-divergence finding.",
                "Do not count as an implementation bug unless a backend-specific specification is contradicted.",
                "Use a separate boundary-semantics experiment for this class.",
            ],
            adjudication=build_adjudication(
                "expected_semantic_divergence",
                validity_gate="valid_case",
                semantic_gate="expected_boundary",
                attribution_gate="semantic_boundary",
                countable_as_valid_finding=True,
                needs_manual_review=False,
                boundary_rule_ids=[match.rule_id for match in semantic_boundary_matches],
            ),
        )

    if _has_clear_minority_backend(finding, backends):
        return _classification(
            "candidate_implementation_bug",
            "candidate_bug_needs_external_confirmation",
            "high",
            evidence=f"Clear suspicious minority backend(s): {_get(finding, 'suspicious_backends', [])}",
            recommendation=[
                "Minimize the artifact and make a backend-specific reproducer.",
                "Check backend documentation/release notes, then file upstream if behavior contradicts the expected semantics.",
            ],
            adjudication=build_adjudication(
                "candidate_implementation_bug",
                validity_gate="valid_case",
                semantic_gate="common_subset_or_backend_specific",
                attribution_gate="backend_candidate_bug",
                countable_as_bug_evidence=True,
                countable_as_valid_finding=True,
                needs_manual_review=False,
                needs_external_confirmation=True,
            ),
        )

    return _classification(
        "needs_manual_confirmation",
        "valid_finding_needs_triage",
        "medium",
        evidence="No generator/normalizer failure detected, but no clear implementation-bug signal was found.",
        recommendation=[
            "Deduplicate by signature, minimize the case, and inspect backend-specific outputs.",
        ],
        adjudication=build_adjudication(
            "needs_manual_confirmation",
            validity_gate="valid_case",
            semantic_gate="unknown",
            attribution_gate="manual_triage_required",
            countable_as_valid_finding=True,
            needs_manual_review=True,
        ),
    )


def _source_issue(case: Case) -> str:
    return primary_source_issue(case)


def _discovery_origin(case: Case) -> str:
    return case_discovery_origin(case)


def _metamorphic_classification(
    case: Case,
    finding: Finding | dict[str, Any],
    backends: list[str],
) -> Classification | None:
    if _get(finding, "oracle", "") != "metamorphic":
        return None
    suspicious = list(_get(finding, "suspicious_backends", []) or [])
    if has_order_observer(case.program):
        return _classification(
            "semantic_divergence_needs_confirmation",
            "valid_finding_not_confirmed_bug",
            "medium",
            evidence=(
                "Metamorphic relation violation occurs in a program with an order-observing "
                "operation; row ordering or tie-breaking may be part of the observed behavior."
            ),
            recommendation=[
                "Inspect whether the metamorphic relation is valid for this order-sensitive case before counting it as a bug.",
                "Regenerate without row-number/running-sum/sort/top-k observers or add explicit tie-breakers.",
            ],
            adjudication=build_adjudication(
                "semantic_divergence_needs_confirmation",
                validity_gate="valid_case",
                semantic_gate="boundary_needs_confirmation",
                attribution_gate="semantic_boundary",
                metamorphic_support="order_observer_needs_confirmation",
                countable_as_valid_finding=True,
                needs_manual_review=True,
            ),
        )
    mismatch_class = str(_get(finding, "mismatch_class", "") or "")
    if mismatch_class in {"status", "error_type", "row_order"}:
        if mismatch_class == "row_order":
            mismatch_evidence = (
                "Metamorphic relation violation is order-only; row ordering or tie-breaking may be "
                "underconstrained for this relation."
            )
            metamorphic_support = "row_order_needs_confirmation"
        else:
            mismatch_evidence = (
                "Metamorphic relation violation changes execution status or exception taxonomy "
                f"({mismatch_class}); the variant may have crossed a generated-program or adapter boundary."
            )
            metamorphic_support = "accept_reject_needs_confirmation"
        return _classification(
            "semantic_divergence_needs_confirmation",
            "valid_finding_not_confirmed_bug",
            "medium",
            evidence=mismatch_evidence,
            recommendation=[
                "Inspect whether the metamorphic relation is valid for this ordering-sensitive behavior before counting it as a bug.",
                "Prefer ok-vs-ok value/schema/row-count metamorphic violations with explicit ordering constraints for automatic candidate bug evidence.",
            ],
            adjudication=build_adjudication(
                "semantic_divergence_needs_confirmation",
                validity_gate="valid_case",
                semantic_gate="boundary_needs_confirmation",
                attribution_gate="semantic_boundary",
                metamorphic_support=metamorphic_support,
                countable_as_valid_finding=True,
                needs_manual_review=True,
            ),
        )
    if len(suspicious) == 1 and len(backends) >= 1:
        return _classification(
            "candidate_implementation_bug",
            "candidate_bug_needs_external_confirmation",
            "high",
            evidence=(
                "Single backend violates a semantics-preserving metamorphic relation: "
                f"{suspicious[0]}"
            ),
            recommendation=[
                "Minimize the metamorphic pair and rerun the same backend on base and variant cases.",
                "If the relation is still violated, file as a backend/adaptor implementation bug.",
            ],
            adjudication=build_adjudication(
                "candidate_implementation_bug",
                validity_gate="valid_case",
                semantic_gate="common_subset_or_backend_specific",
                attribution_gate="backend_candidate_bug",
                metamorphic_support="single_backend_violation",
                countable_as_bug_evidence=True,
                countable_as_valid_finding=True,
                needs_manual_review=False,
                needs_external_confirmation=True,
            ),
        )
    return _classification(
        "semantic_divergence_needs_confirmation",
        "valid_finding_not_confirmed_bug",
        "medium",
        evidence="Metamorphic relation violation involves multiple or ambiguous backends.",
        recommendation=[
            "Inspect whether the metamorphic relation is valid for this case before counting it as a bug.",
        ],
        adjudication=build_adjudication(
            "semantic_divergence_needs_confirmation",
            validity_gate="valid_case",
            semantic_gate="boundary_needs_confirmation",
            attribution_gate="semantic_boundary",
            metamorphic_support="ambiguous_violation",
            countable_as_valid_finding=True,
            needs_manual_review=True,
        ),
    )


def _reference_classification(
    case: Case,
    finding: Finding | dict[str, Any],
    normalized: dict[str, NormalizedResult | dict[str, Any]],
    backends: list[str],
) -> Classification | None:
    if not normalized:
        return None
    reference = reference_result(case)
    if reference is None:
        return None
    comparable_items = [
        (backend, result)
        for backend, result in normalized.items()
        if _result_get(result, "status") not in {"normalization_error", ""}
    ]
    if not comparable_items:
        return None
    comparable_backends = [backend for backend, _ in comparable_items]
    anchor_comparison = compare_result_against_anchor([reference, *[result for _, result in comparable_items]])
    backend_labels = ["__reference__", *comparable_backends]
    matching, mismatching, confidence = anchor_comparison.label_summary(
        backend_labels,
        exclude_anchor_from_matches=True,
    )
    if not mismatching:
        return None

    suspicious = set(_get(finding, "suspicious_backends", []) or [])
    implicated = sorted((set(mismatching) & suspicious) or set(mismatching))
    if matching and implicated:
        return _classification(
            "candidate_implementation_bug",
            "candidate_bug_needs_external_confirmation",
            confidence,
            evidence=(
                "Independent DSL reference agrees with "
                f"{sorted(matching)} and disagrees with {implicated}."
            ),
            implicated_backends=implicated,
            recommendation=[
                "Minimize the case and include the DSL reference output in the artifact.",
                "Treat as confirmed only after backend documentation or maintainers establish the expected behavior.",
            ],
            adjudication=build_adjudication(
                "candidate_implementation_bug",
                validity_gate="valid_case",
                semantic_gate="common_subset_or_backend_specific",
                attribution_gate="backend_candidate_bug",
                reference_support="reference_majority_support",
                countable_as_bug_evidence=True,
                countable_as_valid_finding=True,
                needs_manual_review=False,
                needs_external_confirmation=True,
            ),
        )

    if not matching:
        return _classification(
            "needs_manual_confirmation",
            "valid_finding_needs_triage",
            "medium",
            evidence="No tested backend matches the independent DSL reference output.",
            recommendation=[
                "Inspect the DSL reference semantics before making a backend bug claim.",
                "This may indicate a reference-oracle bug or an underspecified DSL operation.",
            ],
            adjudication=build_adjudication(
                "needs_manual_confirmation",
                validity_gate="valid_case",
                semantic_gate="unknown",
                attribution_gate="manual_triage_required",
                reference_support="reference_disagrees_with_all",
                countable_as_valid_finding=True,
                needs_manual_review=True,
            ),
        )
    return None


def _documented_polars_nan_semantics(case: Case, finding: Finding | dict[str, Any], config: dict[str, Any]) -> bool:
    suspicious = set(_get(finding, "suspicious_backends", []) or [])
    return (
        str(_get(finding, "root_cause", "")) == "nan_inf_semantics"
        and suspicious == {"polars"}
        and _case_contains_special_float(case)
    )


def _join_null_semantics_boundary(case: Case, finding: Finding | dict[str, Any], config: dict[str, Any]) -> bool:
    return _join_keys_contain_null(case)


def _null_filter_literal_boundary(case: Case, finding: Finding | dict[str, Any], config: dict[str, Any]) -> bool:
    return _case_has_null_filter_literal(case)


def _modulo_boundary(case: Case, finding: Finding | dict[str, Any], config: dict[str, Any]) -> bool:
    return _case_uses_modulo(case)


def _unicode_case_mapping_boundary(case: Case, finding: Finding | dict[str, Any], config: dict[str, Any]) -> bool:
    return _case_uses_unicode_case_mapping(case) and _case_contains_non_ascii_string(case)


DOCUMENTED_SEMANTIC_RULES = build_documented_semantic_rules(
    documented_polars_nan_semantics=_documented_polars_nan_semantics,
)


SEMANTIC_BOUNDARY_RULES = build_semantic_boundary_rules(
    get_finding_value=lambda finding, key, default=None: _get(finding, key, default),
    case_contains_special_float=_case_contains_special_float,
    join_null_semantics_boundary=_join_null_semantics_boundary,
    null_filter_literal_boundary=_null_filter_literal_boundary,
    modulo_boundary=_modulo_boundary,
    unicode_case_mapping_boundary=_unicode_case_mapping_boundary,
)


def documented_semantic_rule_records() -> tuple[StrategyRuleRecord, ...]:
    return semantic_rule_records(DOCUMENTED_SEMANTIC_RULES, kind="documented_semantic_rule")


def semantic_boundary_rule_records() -> tuple[StrategyRuleRecord, ...]:
    return semantic_rule_records(SEMANTIC_BOUNDARY_RULES, kind="semantic_boundary_rule")


def _is_documented_semantic_divergence(case: Case, finding: Finding | dict[str, Any], config: dict[str, Any]) -> bool:
    return bool(_documented_semantic_matches(case, finding, config))


def _is_semantic_boundary(case: Case, finding: Finding | dict[str, Any], config: dict[str, Any]) -> bool:
    return bool(_semantic_boundary_matches(case, finding, config))


def _documented_semantic_matches(
    case: Case,
    finding: Finding | dict[str, Any],
    config: dict[str, Any],
) -> list[SemanticBoundaryMatch]:
    return matching_semantic_rules(
        ordered_rules_from_snapshot(
            config=config,
            snapshot_key="classification_documented_rules",
            default_rules=DOCUMENTED_SEMANTIC_RULES,
        ),
        case,
        finding,
        config,
    )


def _semantic_boundary_matches(
    case: Case,
    finding: Finding | dict[str, Any],
    config: dict[str, Any],
) -> list[SemanticBoundaryMatch]:
    return matching_semantic_rules(
        ordered_rules_from_snapshot(
            config=config,
            snapshot_key="classification_boundary_rules",
            default_rules=SEMANTIC_BOUNDARY_RULES,
        ),
        case,
        finding,
        config,
    )


def _semantic_boundary_reasons(case: Case, finding: Finding | dict[str, Any], config: dict[str, Any]) -> list[str]:
    return semantic_boundary_reasons(_semantic_boundary_matches(case, finding, config))


def _documentation_refs(case: Case, finding: Finding | dict[str, Any]) -> list[dict[str, str]]:
    root = str(_get(finding, "root_cause", ""))
    if root == "nan_inf_semantics":
        return [
            {
                "title": "Polars floating point numbers",
                "url": "https://docs.pola.rs/user-guide/concepts/data-types-and-structures/#floating-point-numbers",
                "note": "Polars documents NaN ordering/comparison behavior as distinct from regular missing data.",
            },
            {
                "title": "Polars missing data",
                "url": "https://docs.pola.rs/user-guide/expressions/missing-data/#not-a-number-or-nan-values",
                "note": "Polars documents null as missing data and NaN as a floating-point value.",
            },
        ]
    return []


def _get(finding: Finding | dict[str, Any], key: str, default: Any = None) -> Any:
    if isinstance(finding, dict):
        return finding.get(key, default)
    return getattr(finding, key, default)


def _result_get(result: NormalizedResult | dict[str, Any], key: str, default: Any = "") -> Any:
    if isinstance(result, dict):
        return result.get(key, default)
    return getattr(result, key, default)
