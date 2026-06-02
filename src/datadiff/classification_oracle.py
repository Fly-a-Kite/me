from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from datadiff.adjudication import build_adjudication
from datadiff.canonicalization import (
    compare_result_batch,
    compare_result_against_anchor,
    canonical_key,
    compare_row_set_batch,
    compare_row_sets,
    profile_row_sets,
    profile_rows,
)
from datadiff.case_features import (
    case_contains_non_ascii_string as _case_contains_non_ascii_string,
    case_contains_special_float as _case_contains_special_float,
    case_has_null_filter_literal as _case_has_null_filter_literal,
    case_uses_modulo as _case_uses_modulo,
    case_uses_unicode_case_mapping as _case_uses_unicode_case_mapping,
)
from datadiff.case_policy import case_discovery_origin, primary_source_issue
from datadiff.dsl import Case, Program, SortKey, normalize_sort_keys
from datadiff.expression_semantics import expr_output_type
from datadiff.filtering import (
    evaluate_filter_predicate,
    filter_comparator_supports_type,
    is_filter_comparator,
    parse_filter_comparator,
)
from datadiff.identifiers import is_reserved_output_name
from datadiff.join_keys import join_key_pairs, join_key_value
from datadiff.normalizer import NormalizedResult, _norm_value
from datadiff.ordering_semantics import rows_have_duplicate_sort_key, sort_row_mappings, sort_window_boundary_splits_tie
from datadiff.operation_semantics import (
    aggregate_alias,
    aggregate_column,
    aggregate_func,
    aggregate_functions,
    aggregate_specs,
    case_else_value,
    case_then_value,
    coalesce_fallback,
    condition_cmp,
    condition_column,
    condition_value,
    expr_input_domain,
    expr_kind,
    expr_operator,
    expr_part,
    expr_source,
    expr_target_type,
    expr_value,
    groupby_keys,
    op_branches,
    op_column,
    op_columns,
    op_kind,
    op_literal,
    op_n,
    op_output_alias,
    op_partition_columns,
    op_quantiles,
    op_right_columns,
    op_rows,
    op_table,
    op_value,
    op_values,
    operation_names,
)
from datadiff.operation_type_semantics import aggregate_accepts_type, aggregate_result_type, case_when_output_type
from datadiff.oracle import Finding
from datadiff.pathing import path_basename
from datadiff.program_analysis import case_uses_precision_sensitive_float_arithmetic
from datadiff.program_state import ProgramState, apply_operation_state
from datadiff.program_validation import (
    ValidationContext,
    valid_digit_string_values,
    valid_float_literal_text,
    valid_group_quantile_values,
    validate_core_operation,
)
from datadiff.reference_semantics import reference_result
from datadiff.running import (
    running_sum_partition_columns,
    running_sum_sort_keys,
    sort_rows_for_running,
    stable_running_sum_values,
)
from datadiff.sortedness import is_sorted_values
from datadiff.tuple_logic import evaluate_tuple_absence
from datadiff.util import unique_preserve_order
from datadiff.windowing import row_number_filter_rows


BoundaryPredicate = Callable[[Case, Finding | dict[str, Any], dict[str, Any]], bool]


@dataclass(frozen=True, slots=True)
class SemanticBoundaryRule:
    rule_id: str
    reason: str
    predicate: BoundaryPredicate


@dataclass(frozen=True, slots=True)
class SemanticBoundaryMatch:
    rule_id: str
    reason: str


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

    metamorphic_classification = _metamorphic_classification(finding, backends)
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


def validate_case_program(case: Case) -> list[str]:
    if not case.tables:
        return ["case has no tables"]
    errors: list[str] = []
    tables = {table.name: table for table in case.tables}
    state = ProgramState.from_table(case.tables[0])
    bool_alias_probes = {
        "scalar_subquery_probe",
        "window_avg_probe",
        "struct_distinct_probe",
        "bit_compare_probe",
        "round_even_probe",
        "timestamp_precision_filter_probe",
        "series_rtruediv_probe",
        "uint64_isin_probe",
        "tuple_anti_null_probe",
        "setop_all_duplicate_probe",
        "json_predicate_order_probe",
        "sparse_mask_probe",
        "float_wrap_probe",
        "index_bool_probe",
        "empty_literal_groupby_probe",
        "arrow_string_eq_sum_probe",
        "arrow_timestamp_loc_slice_probe",
        "arrow_timestamp_index_attr_probe",
        "eval_inplace_alias_probe",
        "bool_reduction_skipna_probe",
        "dataset_isin_all_match_probe",
        "run_end_null_compute_probe",
        "large_string_partition_probe",
        "hash_pivot_wider_probe",
        "list_flatten_parent_indices_probe",
        "rolling_mean_by_null_count_probe",
    }

    for idx, op in enumerate(case.program.operations):
        kind = op_kind(op)
        ctx = ValidationContext(tables=tables, state=state, errors=errors)
        is_core_operation = kind in {
            "join",
            "semi_join",
            "anti_join",
            "union_all",
            "drop_nulls",
            "filter",
            "tuple_absence_filter",
            "row_number_filter",
            "running_sum",
            "sortedness_check",
            "select",
            "sort",
            "limit",
            "offset",
            "mutate",
            "distinct",
            "fill_null",
            "coalesce",
            "case_when",
            "groupby",
            "aggregate",
        }
        if is_core_operation:
            if validate_core_operation(ctx, op, idx):
                apply_operation_state(state, op)
                if kind == "join":
                    right = tables.get(op_table(op))
                    if right is not None:
                        _, right_keys = join_key_pairs(op)
                        right_key_set = set(right_keys)
                        for column in right.columns:
                            if column.name in right_key_set or column.name in state.available:
                                continue
                            state.columns.append(column.name)
                            state.column_types[column.name] = column.type
            continue
        if kind == "random_case_probe":
            alias = op_output_alias(op)
            if not alias:
                errors.append(f"op {idx}: random_case_probe output alias is empty")
            elif is_reserved_output_name(alias):
                errors.append(f"op {idx}: random_case_probe output alias {alias!r} is reserved")
            try:
                if op_rows(op) <= 0:
                    errors.append(f"op {idx}: random_case_probe rows must be positive")
            except (TypeError, ValueError):
                errors.append(f"op {idx}: random_case_probe rows must be an integer")
            try:
                branches = op_branches(op)
                if branches <= 0 or branches > 16:
                    errors.append(f"op {idx}: random_case_probe branches must be between 1 and 16")
            except (TypeError, ValueError):
                errors.append(f"op {idx}: random_case_probe branches must be an integer")
            if alias:
                state.collapse_to_alias(alias, "bool")
        elif kind == "group_quantile_probe":
            alias = op_output_alias(op)
            if not alias:
                errors.append(f"op {idx}: group_quantile_probe output alias is empty")
            elif is_reserved_output_name(alias):
                errors.append(f"op {idx}: group_quantile_probe output alias {alias!r} is reserved")
            values = op_values(op)
            quantiles = op_quantiles(op)
            if not valid_group_quantile_values(values, quantiles):
                errors.append(f"op {idx}: group_quantile_probe requires numeric values and quantiles in [0, 1]")
            if alias:
                state.collapse_to_alias(alias, "bool")
        elif kind == "scalar_subquery_probe":
            alias = op_output_alias(op)
            if not alias:
                errors.append(f"op {idx}: scalar_subquery_probe output alias is empty")
            elif is_reserved_output_name(alias):
                errors.append(f"op {idx}: scalar_subquery_probe output alias {alias!r} is reserved")
            if alias:
                state.collapse_to_alias(alias, "bool")
        elif kind == "window_avg_probe":
            alias = op_output_alias(op)
            if not alias:
                errors.append(f"op {idx}: window_avg_probe output alias is empty")
            elif is_reserved_output_name(alias):
                errors.append(f"op {idx}: window_avg_probe output alias {alias!r} is reserved")
            if alias:
                state.collapse_to_alias(alias, "bool")
        elif kind == "float_literal_precision_probe":
            alias = op_output_alias(op)
            literal = op_literal(op)
            if not alias:
                errors.append(f"op {idx}: float_literal_precision_probe output alias is empty")
            elif is_reserved_output_name(alias):
                errors.append(f"op {idx}: float_literal_precision_probe output alias {alias!r} is reserved")
            if not valid_float_literal_text(literal):
                errors.append(f"op {idx}: float_literal_precision_probe literal is invalid")
            if alias:
                state.collapse_to_alias(alias, "bool")
        elif kind in bool_alias_probes:
            alias = op_output_alias(op)
            if not alias:
                errors.append(f"op {idx}: {kind} output alias is empty")
            elif is_reserved_output_name(alias):
                errors.append(f"op {idx}: {kind} output alias {alias!r} is reserved")
            if alias:
                state.collapse_to_alias(alias, "bool")
        elif kind == "csv_long_numeric_roundtrip_probe":
            alias = op_output_alias(op)
            if not alias:
                errors.append(f"op {idx}: csv_long_numeric_roundtrip_probe output alias is empty")
            elif is_reserved_output_name(alias):
                errors.append(f"op {idx}: csv_long_numeric_roundtrip_probe output alias {alias!r} is reserved")
            if not valid_digit_string_values(op_values(op)):
                errors.append(f"op {idx}: csv_long_numeric_roundtrip_probe requires non-empty digit-string values")
            if alias:
                state.collapse_to_alias(alias, "bool")
        else:
            errors.append(f"op {idx}: unknown operation {kind!r}")
    return errors


def _metamorphic_classification(
    finding: Finding | dict[str, Any],
    backends: list[str],
) -> Classification | None:
    if _get(finding, "oracle", "") != "metamorphic":
        return None
    suspicious = list(_get(finding, "suspicious_backends", []) or [])
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


def _mutate_output_type(
    expr: dict[str, Any],
    available: set[str],
    numeric: set[str],
    strings: set[str],
    col_types: dict[str, str],
) -> str | None:
    kind = expr_kind({"expr": expr})
    src = expr_source({"expr": expr})
    if src not in available:
        return None
    return expr_output_type(expr, col_types)


def _cast_output_type(source_type: str, expr: dict[str, Any]) -> str | None:
    target = expr_target_type({"expr": expr})
    input_domain = expr_input_domain({"expr": expr})
    if target == "float" and source_type in {"int", "float"}:
        return "float"
    if target == "float" and source_type == "str" and input_domain in {"numeric_string", "integer_string"}:
        return "float"
    if target == "int" and source_type == "int":
        return "int"
    if target == "int" and source_type == "str" and input_domain == "integer_string":
        return "int"
    if target == "str" and source_type in {"int", "float", "str"}:
        return "str"
    return None


def _case_when_output_type(then_value: Any, else_value: Any) -> str | None:
    return case_when_output_type(then_value, else_value)


def _valid_clip_bounds(source_type: str, lower: Any, upper: Any) -> bool:
    if lower is None or upper is None:
        return False
    if not all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in (lower, upper)):
        return False
    if lower > upper:
        return False
    if source_type == "int":
        return isinstance(lower, int) and isinstance(upper, int)
    return source_type == "float"


def _literal_output_type(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    return None


def _filter_literal_error(column_type: str, comparator: Any, value: Any) -> str:
    if not filter_comparator_supports_type(column_type, comparator):
        return f"comparator {comparator!r} is not supported for {column_type} filter"
    parsed = parse_filter_comparator(comparator)
    if parsed is not None and parsed.base in {"in_set", "not_in_set"}:
        if not isinstance(value, list) or not value:
            return f"filter literal {value!r} is not a non-empty list for {parsed.base}"
        if any(item is None for item in value):
            return f"{parsed.base} filter literals must not contain NULL"
        for item in value:
            item_error = _scalar_filter_literal_error(column_type, item)
            if item_error:
                return item_error
        return ""
    if parsed is not None and parsed.base == "range_closed":
        if not isinstance(value, list) or len(value) != 2:
            return f"filter literal {value!r} must contain two bounds for range_closed"
        if any(item is None for item in value):
            return "range_closed filter bounds must not contain NULL"
        for item in value:
            item_error = _scalar_filter_literal_error(column_type, item)
            if item_error:
                return item_error
        if value[0] > value[1]:
            return "range_closed lower bound must be <= upper bound"
        return ""
    if parsed is not None and parsed.base in {"str_contains", "str_starts_with", "str_ends_with"}:
        if not isinstance(value, str) or value == "":
            return f"filter literal {value!r} must be a non-empty string for {parsed.base}"
        return ""
    if parsed is not None and parsed.base in {"is_null", "is_not_null"}:
        return "" if value is None else f"filter literal {value!r} must be NULL for {parsed.base}"
    if parsed is not None and parsed.base == "bool_predicate":
        return "" if value is None else f"filter literal {value!r} must be NULL for {comparator}"
    if value is None:
        return ""
    return _scalar_filter_literal_error(column_type, value)


def _scalar_filter_literal_error(column_type: str, value: Any) -> str:
    if column_type == "str" and not isinstance(value, str):
        return f"filter literal {value!r} is not compatible with str column"
    if column_type == "bool" and not isinstance(value, bool):
        return f"filter literal {value!r} is not compatible with bool column"
    if column_type in {"int", "float"} and (isinstance(value, bool) or not isinstance(value, (int, float))):
        return f"filter literal {value!r} is not compatible with {column_type} column"
    return ""


def _stable_row_key(row: list[Any]) -> str:
    return canonical_key(row)


def _aggregate_accepts_type(source_type: str, func: str, is_numeric_column: bool) -> bool:
    return aggregate_accepts_type(source_type, func, is_numeric_column)


def _aggregate_result_type(source_type: str, func: str) -> str:
    return aggregate_result_type(source_type, func)


def _normalizer_errors(normalized: dict[str, NormalizedResult | dict[str, Any]]) -> list[str]:
    out = []
    for backend, result in normalized.items():
        if _result_get(result, "status") == "normalization_error":
            out.append(f"{backend}:{_result_get(result, 'error_type')}")
    return out


def _is_pyarrow_empty_global_bool_aggregate_adapter_error(
    case: Case,
    finding: Finding | dict[str, Any],
    normalized: dict[str, NormalizedResult | dict[str, Any]],
    raw_results: dict[str, dict[str, Any]],
) -> bool:
    suspicious = {str(backend) for backend in (_get(finding, "suspicious_backends", []) or [])}
    if "pyarrow" not in suspicious:
        return False
    pyarrow_raw = raw_results.get("pyarrow", {})
    pyarrow_normalized = normalized.get("pyarrow")
    status = str(pyarrow_raw.get("status") or _result_get(pyarrow_normalized, "status", ""))
    if status != "error":
        return False
    error_text = " ".join(
        str(part)
        for part in [
            pyarrow_raw.get("error_type", ""),
            pyarrow_raw.get("error", ""),
            _result_get(pyarrow_normalized, "error_type", ""),
            _result_get(pyarrow_normalized, "error", ""),
        ]
    ).lower()
    if "arrowinvalid" not in error_text or "invalid null value" not in error_text:
        return False
    return _has_empty_input_global_bool_aggregate(case)


def _has_empty_input_global_bool_aggregate(case: Case) -> bool:
    rows = [dict(row) for row in case.tables[0].rows] if case.tables else []
    for op in case.program.operations:
        kind = op_kind(op)
        if kind == "filter":
            try:
                rows = [
                    row
                    for row in rows
                    if evaluate_filter_predicate(row.get(condition_column(op)), condition_cmp(op), condition_value(op))
                ]
            except Exception:  # noqa: BLE001
                return False
            continue
        if kind != "aggregate":
            return False
        return not rows and bool(set(aggregate_functions(op)) & {"any", "all"})
    return False


def _all_backends_rejected_due_to_generated_invalidity(raw_results: dict[str, dict[str, Any]]) -> bool:
    if not raw_results:
        return False
    if any(result.get("status") == "ok" for result in raw_results.values()):
        return False
    text = " ".join(
        f"{result.get('error_type', '')} {result.get('error', '')}".lower()
        for result in raw_results.values()
    )
    invalid_markers = [
        "column",
        "not found",
        "no such",
        "binder",
        "schema",
        "keyerror",
        "invalid operation",
        "cannot resolve",
    ]
    return any(marker in text for marker in invalid_markers)


def _finding_root_is(expected_root: str) -> BoundaryPredicate:
    return lambda case, finding, config: str(_get(finding, "root_cause", "")) == expected_root


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


DOCUMENTED_SEMANTIC_RULES: tuple[SemanticBoundaryRule, ...] = (
    SemanticBoundaryRule(
        rule_id="documented:polars_nan_inf_semantics",
        reason="Polars documents NaN/NULL floating-point behavior for this mismatch class.",
        predicate=_documented_polars_nan_semantics,
    ),
)


SEMANTIC_BOUNDARY_RULES: tuple[SemanticBoundaryRule, ...] = (
    SemanticBoundaryRule(
        rule_id="boundary:root_nan_inf_semantics",
        reason="root cause nan_inf_semantics is a known cross-engine semantic boundary",
        predicate=_finding_root_is("nan_inf_semantics"),
    ),
    SemanticBoundaryRule(
        rule_id="boundary:root_null_semantics",
        reason="root cause null_semantics is a known cross-engine semantic boundary",
        predicate=_finding_root_is("null_semantics"),
    ),
    SemanticBoundaryRule(
        rule_id="boundary:root_ordering_or_limit",
        reason="root cause ordering_or_limit is a known cross-engine semantic boundary",
        predicate=_finding_root_is("ordering_or_limit"),
    ),
    SemanticBoundaryRule(
        rule_id="boundary:special_float_values",
        reason="case contains NaN or Infinity values",
        predicate=lambda case, finding, config: (
            str(_get(finding, "root_cause", "")) == "nan_inf_semantics" and _case_contains_special_float(case)
        ),
    ),
    SemanticBoundaryRule(
        rule_id="boundary:join_null_keys",
        reason="join key contains NULL values; NULL join semantics differ across target families",
        predicate=_join_null_semantics_boundary,
    ),
    SemanticBoundaryRule(
        rule_id="boundary:null_filter_literal",
        reason="filter compares against NULL; engines intentionally differ on NULL predicate semantics",
        predicate=_null_filter_literal_boundary,
    ),
    SemanticBoundaryRule(
        rule_id="boundary:modulo_semantics",
        reason="case uses modulo; negative/float remainder semantics differ across engines",
        predicate=_modulo_boundary,
    ),
    SemanticBoundaryRule(
        rule_id="boundary:unicode_case_mapping",
        reason="case changes case for non-ASCII text; Unicode case mapping support differs across engines",
        predicate=_unicode_case_mapping_boundary,
    ),
)


def _matching_semantic_rules(
    rules: tuple[SemanticBoundaryRule, ...],
    case: Case,
    finding: Finding | dict[str, Any],
    config: dict[str, Any],
) -> list[SemanticBoundaryMatch]:
    matches: list[SemanticBoundaryMatch] = []
    for rule in rules:
        if rule.predicate(case, finding, config):
            matches.append(SemanticBoundaryMatch(rule_id=rule.rule_id, reason=rule.reason))
    return matches


def _is_documented_semantic_divergence(case: Case, finding: Finding | dict[str, Any], config: dict[str, Any]) -> bool:
    return bool(_documented_semantic_matches(case, finding, config))


def _is_semantic_boundary(case: Case, finding: Finding | dict[str, Any], config: dict[str, Any]) -> bool:
    return bool(_semantic_boundary_matches(case, finding, config))


def _documented_semantic_matches(
    case: Case,
    finding: Finding | dict[str, Any],
    config: dict[str, Any],
) -> list[SemanticBoundaryMatch]:
    return _matching_semantic_rules(DOCUMENTED_SEMANTIC_RULES, case, finding, config)


def _semantic_boundary_matches(
    case: Case,
    finding: Finding | dict[str, Any],
    config: dict[str, Any],
) -> list[SemanticBoundaryMatch]:
    return _matching_semantic_rules(SEMANTIC_BOUNDARY_RULES, case, finding, config)


def _semantic_boundary_reasons(case: Case, finding: Finding | dict[str, Any], config: dict[str, Any]) -> list[str]:
    return unique_preserve_order(match.reason for match in _semantic_boundary_matches(case, finding, config))


def _is_float_precision_boundary_mismatch(
    case: Case,
    normalized: dict[str, NormalizedResult | dict[str, Any]],
) -> bool:
    if not _case_uses_precision_sensitive_float_arithmetic(case):
        return False
    ok_results = [result for result in normalized.values() if _result_get(result, "status") == "ok"]
    if len(ok_results) < 2:
        return False
    column_sets = [_result_get(result, "columns", []) for result in ok_results]
    first_columns = column_sets[0]
    if any(columns != first_columns for columns in column_sets):
        return False
    comparison = compare_row_set_batch(
        [_result_get(result, "rows", []) for result in ok_results],
        column_sets=column_sets,
    )
    if not comparison.has_mismatch:
        return False
    relaxed_rows = [
        [_relaxed_float_precision_row(row) for row in _result_get(result, "rows", [])]
        for result in ok_results
    ]
    relaxed_comparison = compare_row_set_batch(
        relaxed_rows,
        column_sets=column_sets,
    )
    if relaxed_comparison.is_exact_match():
        return True
    return relaxed_comparison.is_order_only_mismatch()


def _relaxed_float_precision_row(row: list[Any]) -> list[Any]:
    return [_norm_value(value, preserve_float_precision=False) for value in row]


def _case_uses_precision_sensitive_float_arithmetic(case: Case) -> bool:
    return case_uses_precision_sensitive_float_arithmetic(case)


def _has_clear_minority_backend(finding: Finding | dict[str, Any], backends: list[str]) -> bool:
    suspicious = list(_get(finding, "suspicious_backends", []) or [])
    confidence = _get(finding, "confidence", "")
    return 0 < len(suspicious) < max(1, len(backends)) and confidence == "high"


def _is_order_only_mismatch(normalized: dict[str, NormalizedResult | dict[str, Any]]) -> bool:
    ok_results = [result for result in normalized.values() if _result_get(result, "status") == "ok"]
    if len(ok_results) < 2:
        return False
    comparison = compare_row_set_batch(
        [_result_get(result, "rows", []) for result in ok_results],
        column_sets=[_result_get(result, "columns", []) for result in ok_results],
    )
    return comparison.is_order_only_mismatch()


def _is_sort_tie_order_only_mismatch(case: Case) -> bool:
    op_index = _last_order_defining_operation_index(case)
    if op_index is None:
        return False
    op = case.program.operations[op_index]
    if op_kind(op) != "sort":
        return False
    try:
        sort_keys = normalize_sort_keys(op)
    except ValueError:
        return False
    rows = _reference_rows_before_operation(case, op_index)
    if rows is None:
        return False
    return rows_have_duplicate_sort_key(rows, sort_keys)


def _is_sort_topk_tie_cutoff_mismatch(case: Case) -> bool:
    op_index = _last_order_defining_operation_index(case)
    if op_index is None:
        return False
    op = case.program.operations[op_index]
    if op_kind(op) != "sort":
        return False
    try:
        sort_keys = normalize_sort_keys(op)
    except ValueError:
        return False
    rows = _reference_rows_before_operation(case, op_index)
    if rows is None or len(rows) < 2:
        return False
    sorted_rows = list(sort_row_mappings(rows, sort_keys))
    start = 0
    end = len(sorted_rows)
    saw_topk = False
    sort_columns = {key.column for key in sort_keys}
    for later in case.program.operations[op_index + 1 :]:
        kind = op_kind(later)
        if kind == "select":
            continue
        if kind in {"fill_null", "coalesce", "case_when"}:
            continue
        if kind == "mutate":
            if op_column(later) in sort_columns:
                return False
            continue
        if kind == "limit":
            try:
                limit = max(0, op_n(later))
            except (TypeError, ValueError):
                return False
            end = min(end, start + limit)
            saw_topk = True
            if sort_window_boundary_splits_tie(sorted_rows, sort_keys, start, end):
                return True
            continue
        if kind == "offset":
            try:
                offset = max(0, op_n(later))
            except (TypeError, ValueError):
                return False
            start = min(end, start + offset)
            saw_topk = True
            if sort_window_boundary_splits_tie(sorted_rows, sort_keys, start, end):
                return True
            continue
        if saw_topk:
            continue
        return False
    return False


def _has_limit_or_offset_without_defined_order(case: Case) -> bool:
    order_defined = False
    for op in case.program.operations:
        kind = op_kind(op)
        if kind in {"sort", "running_sum", "row_number_filter"}:
            order_defined = True
            continue
        if kind == "limit":
            try:
                limit = op_n(op)
            except (TypeError, ValueError):
                limit = 1
            if limit > 0 and not order_defined:
                return True
            continue
        if kind == "offset":
            try:
                offset = op_n(op)
            except (TypeError, ValueError):
                offset = 1
            if offset > 0 and not order_defined:
                return True
            continue
        if kind in {
            "filter",
            "tuple_absence_filter",
            "select",
            "mutate",
        }:
            continue
        if kind == "sortedness_check":
            order_defined = True
            continue
        order_defined = False
    return False


def _last_order_defining_operation_index(case: Case) -> int | None:
    row_order_preserving = {
        "filter",
        "tuple_absence_filter",
        "running_sum",
        "row_number_filter",
        "select",
        "mutate",
        "limit",
        "offset",
    }
    for idx in range(len(case.program.operations) - 1, -1, -1):
        kind = op_kind(case.program.operations[idx])
        if kind in {"sort", "running_sum", "row_number_filter", "sortedness_check"}:
            return idx
        if kind in row_order_preserving:
            continue
        return None
    return None


def _reference_rows_before_operation(case: Case, op_index: int) -> list[dict[str, Any]] | None:
    prefix = Case(
        case_id=f"{case.case_id}-prefix-{op_index}",
        seed=case.seed,
        tables=case.tables,
        program=Program(
            program_id=f"{case.program.program_id}-prefix-{op_index}",
            seed=case.program.seed,
            operations=list(case.program.operations[:op_index]),
        ),
        metadata=case.metadata,
    )
    reference = reference_result(prefix)
    if reference is None or reference.status != "ok":
        return None
    return [
        {column: row[idx] for idx, column in enumerate(reference.columns)}
        for row in reference.rows
    ]
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


def _join_keys_contain_null(case: Case) -> bool:
    tables = {table.name: table for table in case.tables}
    for op in case.program.operations:
        if op_kind(op) != "join":
            continue
        left_keys, right_keys = join_key_pairs(op)
        right = tables.get(op_table(op))
        left_has_null = any(any(row.get(left_key) is None for left_key in left_keys) for row in case.tables[0].rows)
        right_has_null = bool(
            right and any(any(row.get(right_key) is None for right_key in right_keys) for row in right.rows)
        )
        if left_has_null or right_has_null:
            return True
    return False


def _get(finding: Finding | dict[str, Any], key: str, default: Any = None) -> Any:
    if isinstance(finding, dict):
        return finding.get(key, default)
    return getattr(finding, key, default)


def _result_get(result: NormalizedResult | dict[str, Any], key: str, default: Any = "") -> Any:
    if isinstance(result, dict):
        return result.get(key, default)
    return getattr(result, key, default)
