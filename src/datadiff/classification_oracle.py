from __future__ import annotations

import math
from functools import cmp_to_key
from dataclasses import asdict, dataclass, field
from typing import Any

from datadiff.dsl import Case, Program, SortKey, normalize_sort_keys
from datadiff.filtering import evaluate_filter_predicate, filter_comparator_supports_type, is_filter_comparator, parse_filter_comparator
from datadiff.identifiers import is_reserved_output_name
from datadiff.normalizer import NormalizedResult, _norm_value
from datadiff.case_policy import case_discovery_origin, primary_source_issue
from datadiff.oracle import Finding
from datadiff.pathing import path_basename
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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
        return Classification(
            verdict="generator_false_positive",
            paper_status="exclude_generator_invalid_case",
            confidence="high",
            false_positive=True,
            false_positive_reason="invalid_generated_program",
            evidence="; ".join(validity_errors[:3]),
            recommendation=[
                "Do not count this finding as a backend bug.",
                "Fix the generator/repair logic so it emits only executable DSL programs.",
            ],
        )

    normalizer_errors = _normalizer_errors(normalized)
    if normalizer_errors:
        return Classification(
            verdict="normalizer_false_positive",
            paper_status="exclude_normalizer_failure",
            confidence="high",
            false_positive=True,
            false_positive_reason="normalization_error",
            evidence=f"Normalizer failed for: {normalizer_errors}",
            recommendation=[
                "Do not count this finding as a backend bug.",
                "Fix normalization or rerun with the raw backend outputs before triage.",
            ],
        )

    if _all_backends_rejected_due_to_generated_invalidity(raw_results):
        return Classification(
            verdict="generator_false_positive",
            paper_status="exclude_generator_invalid_case",
            confidence="medium",
            false_positive=True,
            false_positive_reason="all_backends_rejected_generated_case",
            evidence="All backends rejected with schema/name/type errors, indicating an invalid generated DSL program.",
            recommendation=[
                "Do not count this finding as a backend bug.",
                "Minimize the case and add a generator regression test.",
            ],
        )

    if _is_order_only_mismatch(normalized):
        if not case.program.order_sensitive:
            return Classification(
                verdict="normalizer_false_positive",
                paper_status="exclude_normalizer_failure",
                confidence="high",
                false_positive=True,
                false_positive_reason="order_only_normalization_mismatch",
                evidence="Backends returned the same row multiset, but normalized rows are ordered differently.",
                recommendation=[
                    "Do not count this as a backend bug.",
                    "Fix canonical row ordering or compare normalized outputs as bags for this oracle.",
                ],
            )
        if _is_sort_tie_order_only_mismatch(case):
            return Classification(
                verdict="normalizer_false_positive",
                paper_status="exclude_order_underconstrained_case",
                confidence="high",
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
            )

    if _has_limit_or_offset_without_defined_order(case):
        return Classification(
            verdict="generator_false_positive",
            paper_status="exclude_order_underconstrained_case",
            confidence="high",
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
        )

    if _is_documented_semantic_divergence(case, finding, config):
        return Classification(
            verdict="documented_semantic_divergence",
            paper_status="valid_finding_not_bug",
            confidence="high",
            evidence="Finding matches a known/documented semantic boundary.",
            recommendation=[
                "Keep as a semantic-divergence benchmark finding.",
                "Do not count as a confirmed implementation bug.",
            ],
            documentation_refs=_documentation_refs(case, finding),
        )

    if _is_float_precision_boundary_mismatch(case, normalized):
        return Classification(
            verdict="expected_semantic_divergence",
            paper_status="valid_finding_not_bug",
            confidence="high",
            evidence=(
                "Backends differ only in full-precision floating-point arithmetic or aggregate "
                "ordering; the relaxed numeric row multisets agree."
            ),
            recommendation=[
                "Do not count this finding as an implementation bug.",
                "Use exact decimal inputs or backend-specific numeric semantics before making a bug claim.",
            ],
        )

    metamorphic_classification = _metamorphic_classification(finding, backends)
    if metamorphic_classification is not None:
        return metamorphic_classification

    semantic_boundary_reasons = _semantic_boundary_reasons(case, finding, config)
    if semantic_boundary_reasons and str(_get(finding, "root_cause", "")) != "ordering_or_limit":
        return Classification(
            verdict="expected_semantic_divergence",
            paper_status="valid_finding_not_bug",
            confidence="medium",
            evidence="; ".join(semantic_boundary_reasons),
            recommendation=[
                "Keep as a valid semantic-divergence finding.",
                "Do not count as an implementation bug unless a backend-specific specification is contradicted.",
                "Use a separate boundary-semantics experiment for this class.",
            ],
        )

    reference_classification = _reference_classification(case, finding, normalized, backends)
    if reference_classification is not None:
        return reference_classification

    if semantic_boundary_reasons:
        return Classification(
            verdict="expected_semantic_divergence",
            paper_status="valid_finding_not_bug",
            confidence="medium",
            evidence="; ".join(semantic_boundary_reasons),
            recommendation=[
                "Keep as a valid semantic-divergence finding.",
                "Do not count as an implementation bug unless a backend-specific specification is contradicted.",
                "Use a separate boundary-semantics experiment for this class.",
            ],
        )

    if _has_clear_minority_backend(finding, backends):
        return Classification(
            verdict="candidate_implementation_bug",
            paper_status="candidate_bug_needs_external_confirmation",
            confidence="high",
            evidence=f"Clear suspicious minority backend(s): {_get(finding, 'suspicious_backends', [])}",
            recommendation=[
                "Minimize the artifact and make a backend-specific reproducer.",
                "Check backend documentation/release notes, then file upstream if behavior contradicts the expected semantics.",
            ],
        )

    return Classification(
        verdict="needs_manual_confirmation",
        paper_status="valid_finding_needs_triage",
        confidence="medium",
        evidence="No generator/normalizer failure detected, but no clear implementation-bug signal was found.",
        recommendation=[
            "Deduplicate by signature, minimize the case, and inspect backend-specific outputs.",
        ],
    )


def _source_issue(case: Case) -> str:
    return primary_source_issue(case)


def _discovery_origin(case: Case) -> str:
    return case_discovery_origin(case)


def validate_case_program(case: Case) -> list[str]:
    errors: list[str] = []
    if not case.tables:
        return ["case has no tables"]
    tables = {table.name: table for table in case.tables}
    available = {column.name for column in case.tables[0].columns}
    col_types = {column.name: column.type for column in case.tables[0].columns}
    numeric = {column.name for column in case.tables[0].columns if column.type in {"int", "float"}}
    strings = {column.name for column in case.tables[0].columns if column.type == "str"}

    for idx, op in enumerate(case.program.operations):
        kind = op.get("op")
        if kind == "join":
            right = tables.get(str(op.get("table", "")))
            if right is None:
                errors.append(f"op {idx}: unknown join table {op.get('table')!r}")
                continue
            right_cols = {column.name for column in right.columns}
            if op.get("left_on") not in available:
                errors.append(f"op {idx}: join left key {op.get('left_on')!r} is unavailable")
            if op.get("right_on") not in right_cols:
                errors.append(f"op {idx}: join right key {op.get('right_on')!r} is unavailable")
            if op.get("how") not in {"inner", "left"}:
                errors.append(f"op {idx}: unsupported join kind {op.get('how')!r}")
            for column in right.columns:
                if column.name == op.get("right_on") or column.name in available:
                    continue
                available.add(column.name)
                col_types[column.name] = column.type
                if column.type in {"int", "float"}:
                    numeric.add(column.name)
                if column.type == "str":
                    strings.add(column.name)
        elif kind == "filter":
            if op.get("column") not in available:
                errors.append(f"op {idx}: filter column {op.get('column')!r} is unavailable")
            if not is_filter_comparator(op.get("cmp")):
                errors.append(f"op {idx}: unsupported comparator {op.get('cmp')!r}")
            column_type = col_types.get(str(op.get("column")))
            if column_type is not None:
                literal_error = _filter_literal_error(column_type, op.get("cmp"), op.get("value"))
                if literal_error:
                    errors.append(f"op {idx}: {literal_error}")
        elif kind == "tuple_absence_filter":
            right = tables.get(str(op.get("table", "")))
            columns = list(op.get("columns", []))
            right_columns = list(op.get("right_columns", []))
            if right is None:
                errors.append(f"op {idx}: unknown tuple absence table {op.get('table')!r}")
                continue
            right_types = {column.name: column.type for column in right.columns}
            if not columns:
                errors.append(f"op {idx}: tuple absence filter has no columns")
            if len(columns) != len(right_columns):
                errors.append(f"op {idx}: tuple absence column count mismatch")
            if len(unique_preserve_order(columns)) != len(columns):
                errors.append(f"op {idx}: tuple absence filter contains duplicate left columns")
            missing = [column for column in columns if column not in available]
            if missing:
                errors.append(f"op {idx}: tuple absence columns unavailable: {missing}")
            missing_right = [column for column in right_columns if column not in right_types]
            if missing_right:
                errors.append(f"op {idx}: tuple absence right columns unavailable: {missing_right}")
            for left, right_column in zip(columns, right_columns):
                left_type = col_types.get(str(left))
                right_type = right_types.get(str(right_column))
                if left_type is not None and right_type is not None and left_type != right_type:
                    errors.append(
                        f"op {idx}: tuple absence type mismatch {left!r}:{left_type} vs {right_column!r}:{right_type}"
                    )
        elif kind == "row_number_filter":
            partition_by = [str(column) for column in op.get("partition_by", []) or []]
            if len(unique_preserve_order(partition_by)) != len(partition_by):
                errors.append(f"op {idx}: row_number_filter partition_by contains duplicate columns")
            missing_partition = [column for column in partition_by if column not in available]
            if missing_partition:
                errors.append(f"op {idx}: row_number_filter partition_by columns unavailable: {missing_partition}")
            try:
                order_keys = normalize_sort_keys({"keys": op.get("order_by", [])})
            except ValueError as exc:
                errors.append(f"op {idx}: invalid row_number_filter order_by: {exc}")
                order_keys = []
            order_columns = [key.column for key in order_keys]
            missing_order = [column for column in order_columns if column not in available]
            if missing_order:
                errors.append(f"op {idx}: row_number_filter order_by columns unavailable: {missing_order}")
            if not order_columns:
                errors.append(f"op {idx}: row_number_filter has no order_by columns")
            if len(unique_preserve_order(order_columns)) != len(order_columns):
                errors.append(f"op {idx}: row_number_filter order_by contains duplicate columns")
            if op.get("cmp") not in {"==", "<", "<="}:
                errors.append(f"op {idx}: row_number_filter has unsupported comparator {op.get('cmp')!r}")
            try:
                if int(op.get("value", 0)) <= 0:
                    errors.append(f"op {idx}: row_number_filter value must be positive")
            except (TypeError, ValueError):
                errors.append(f"op {idx}: row_number_filter value must be an integer")
        elif kind == "running_sum":
            source = str(op.get("source", ""))
            column = str(op.get("column", ""))
            if source not in available:
                errors.append(f"op {idx}: running_sum source {source!r} is unavailable")
            elif source not in numeric:
                errors.append(f"op {idx}: running_sum source {source!r} is not numeric")
            if not column:
                errors.append(f"op {idx}: running_sum output column is empty")
            elif is_reserved_output_name(column):
                errors.append(f"op {idx}: running_sum output column {column!r} is reserved")
            try:
                order_keys = normalize_sort_keys({"keys": op.get("order_by", [])})
            except ValueError as exc:
                errors.append(f"op {idx}: invalid running_sum order_by: {exc}")
                order_keys = []
            partition_by = [str(column) for column in op.get("partition_by", []) or []]
            missing_partition = [column for column in partition_by if column not in available]
            if missing_partition:
                errors.append(f"op {idx}: running_sum partition_by columns unavailable: {missing_partition}")
            if len(unique_preserve_order(partition_by)) != len(partition_by):
                errors.append(f"op {idx}: running_sum partition_by contains duplicate columns")
            order_columns = [key.column for key in order_keys]
            missing_order = [key for key in order_columns if key not in available]
            if missing_order:
                errors.append(f"op {idx}: running_sum order_by columns unavailable: {missing_order}")
            if not order_columns:
                errors.append(f"op {idx}: running_sum has no order_by columns")
            if len(unique_preserve_order(order_columns)) != len(order_columns):
                errors.append(f"op {idx}: running_sum order_by contains duplicate columns")
            if column:
                available.add(column)
                col_types[column] = "float"
                numeric.add(column)
                strings.discard(column)
        elif kind == "sortedness_check":
            column = str(op.get("column", ""))
            alias = str(op.get("as", ""))
            ascending = op.get("ascending", True)
            nulls = str(op.get("nulls", "last"))
            if column not in available:
                errors.append(f"op {idx}: sortedness_check column {column!r} is unavailable")
            if not alias:
                errors.append(f"op {idx}: sortedness_check output alias is empty")
            elif is_reserved_output_name(alias):
                errors.append(f"op {idx}: sortedness_check output alias {alias!r} is reserved")
            if not isinstance(ascending, bool):
                errors.append(f"op {idx}: sortedness_check ascending must be boolean")
            if nulls not in {"first", "last"}:
                errors.append(f"op {idx}: sortedness_check nulls must be 'first' or 'last'")
            if alias:
                available = {alias}
                col_types = {alias: "bool"}
                numeric = set()
                strings = set()
        elif kind == "random_case_probe":
            alias = str(op.get("as", ""))
            if not alias:
                errors.append(f"op {idx}: random_case_probe output alias is empty")
            elif is_reserved_output_name(alias):
                errors.append(f"op {idx}: random_case_probe output alias {alias!r} is reserved")
            try:
                if int(op.get("rows", 0)) <= 0:
                    errors.append(f"op {idx}: random_case_probe rows must be positive")
            except (TypeError, ValueError):
                errors.append(f"op {idx}: random_case_probe rows must be an integer")
            try:
                branches = int(op.get("branches", 0))
                if branches <= 0 or branches > 16:
                    errors.append(f"op {idx}: random_case_probe branches must be between 1 and 16")
            except (TypeError, ValueError):
                errors.append(f"op {idx}: random_case_probe branches must be an integer")
            if alias:
                available = {alias}
                col_types = {alias: "bool"}
                numeric = set()
                strings = set()
        elif kind == "group_quantile_probe":
            alias = str(op.get("as", ""))
            if not alias:
                errors.append(f"op {idx}: group_quantile_probe output alias is empty")
            elif is_reserved_output_name(alias):
                errors.append(f"op {idx}: group_quantile_probe output alias {alias!r} is reserved")
            values = op.get("values", [])
            quantiles = op.get("quantiles", [])
            if not _valid_group_quantile_values(values, quantiles):
                errors.append(f"op {idx}: group_quantile_probe requires numeric values and quantiles in [0, 1]")
            if alias:
                available = {alias}
                col_types = {alias: "bool"}
                numeric = set()
                strings = set()
        elif kind == "scalar_subquery_probe":
            alias = str(op.get("as", ""))
            if not alias:
                errors.append(f"op {idx}: scalar_subquery_probe output alias is empty")
            elif is_reserved_output_name(alias):
                errors.append(f"op {idx}: scalar_subquery_probe output alias {alias!r} is reserved")
            if alias:
                available = {alias}
                col_types = {alias: "bool"}
                numeric = set()
                strings = set()
        elif kind == "window_avg_probe":
            alias = str(op.get("as", ""))
            if not alias:
                errors.append(f"op {idx}: window_avg_probe output alias is empty")
            elif is_reserved_output_name(alias):
                errors.append(f"op {idx}: window_avg_probe output alias {alias!r} is reserved")
            if alias:
                available = {alias}
                col_types = {alias: "bool"}
                numeric = set()
                strings = set()
        elif kind == "struct_distinct_probe":
            alias = str(op.get("as", ""))
            if not alias:
                errors.append(f"op {idx}: struct_distinct_probe output alias is empty")
            elif is_reserved_output_name(alias):
                errors.append(f"op {idx}: struct_distinct_probe output alias {alias!r} is reserved")
            if alias:
                available = {alias}
                col_types = {alias: "bool"}
                numeric = set()
                strings = set()
        elif kind == "bit_compare_probe":
            alias = str(op.get("as", ""))
            if not alias:
                errors.append(f"op {idx}: bit_compare_probe output alias is empty")
            elif is_reserved_output_name(alias):
                errors.append(f"op {idx}: bit_compare_probe output alias {alias!r} is reserved")
            if alias:
                available = {alias}
                col_types = {alias: "bool"}
                numeric = set()
                strings = set()
        elif kind == "round_even_probe":
            alias = str(op.get("as", ""))
            if not alias:
                errors.append(f"op {idx}: round_even_probe output alias is empty")
            elif is_reserved_output_name(alias):
                errors.append(f"op {idx}: round_even_probe output alias {alias!r} is reserved")
            if alias:
                available = {alias}
                col_types = {alias: "bool"}
                numeric = set()
                strings = set()
        elif kind == "float_literal_precision_probe":
            alias = str(op.get("as", ""))
            literal = str(op.get("literal", ""))
            if not alias:
                errors.append(f"op {idx}: float_literal_precision_probe output alias is empty")
            elif is_reserved_output_name(alias):
                errors.append(f"op {idx}: float_literal_precision_probe output alias {alias!r} is reserved")
            if not _valid_float_literal_text(literal):
                errors.append(f"op {idx}: float_literal_precision_probe literal is invalid")
            if alias:
                available = {alias}
                col_types = {alias: "bool"}
                numeric = set()
                strings = set()
        elif kind == "timestamp_precision_filter_probe":
            alias = str(op.get("as", ""))
            if not alias:
                errors.append(f"op {idx}: timestamp_precision_filter_probe output alias is empty")
            elif is_reserved_output_name(alias):
                errors.append(f"op {idx}: timestamp_precision_filter_probe output alias {alias!r} is reserved")
            if alias:
                available = {alias}
                col_types = {alias: "bool"}
                numeric = set()
                strings = set()
        elif kind == "series_rtruediv_probe":
            alias = str(op.get("as", ""))
            if not alias:
                errors.append(f"op {idx}: series_rtruediv_probe output alias is empty")
            elif is_reserved_output_name(alias):
                errors.append(f"op {idx}: series_rtruediv_probe output alias {alias!r} is reserved")
            if alias:
                available = {alias}
                col_types = {alias: "bool"}
                numeric = set()
                strings = set()
        elif kind == "uint64_isin_probe":
            alias = str(op.get("as", ""))
            if not alias:
                errors.append(f"op {idx}: uint64_isin_probe output alias is empty")
            elif is_reserved_output_name(alias):
                errors.append(f"op {idx}: uint64_isin_probe output alias {alias!r} is reserved")
            if alias:
                available = {alias}
                col_types = {alias: "bool"}
                numeric = set()
                strings = set()
        elif kind == "tuple_anti_null_probe":
            alias = str(op.get("as", ""))
            if not alias:
                errors.append(f"op {idx}: tuple_anti_null_probe output alias is empty")
            elif is_reserved_output_name(alias):
                errors.append(f"op {idx}: tuple_anti_null_probe output alias {alias!r} is reserved")
            if alias:
                available = {alias}
                col_types = {alias: "bool"}
                numeric = set()
                strings = set()
        elif kind == "setop_all_duplicate_probe":
            alias = str(op.get("as", ""))
            if not alias:
                errors.append(f"op {idx}: setop_all_duplicate_probe output alias is empty")
            elif is_reserved_output_name(alias):
                errors.append(f"op {idx}: setop_all_duplicate_probe output alias {alias!r} is reserved")
            if alias:
                available = {alias}
                col_types = {alias: "bool"}
                numeric = set()
                strings = set()
        elif kind == "json_predicate_order_probe":
            alias = str(op.get("as", ""))
            if not alias:
                errors.append(f"op {idx}: json_predicate_order_probe output alias is empty")
            elif is_reserved_output_name(alias):
                errors.append(f"op {idx}: json_predicate_order_probe output alias {alias!r} is reserved")
            if alias:
                available = {alias}
                col_types = {alias: "bool"}
                numeric = set()
                strings = set()
        elif kind == "sparse_mask_probe":
            alias = str(op.get("as", ""))
            if not alias:
                errors.append(f"op {idx}: sparse_mask_probe output alias is empty")
            elif is_reserved_output_name(alias):
                errors.append(f"op {idx}: sparse_mask_probe output alias {alias!r} is reserved")
            if alias:
                available = {alias}
                col_types = {alias: "bool"}
                numeric = set()
                strings = set()
        elif kind == "float_wrap_probe":
            alias = str(op.get("as", ""))
            if not alias:
                errors.append(f"op {idx}: float_wrap_probe output alias is empty")
            elif is_reserved_output_name(alias):
                errors.append(f"op {idx}: float_wrap_probe output alias {alias!r} is reserved")
            if alias:
                available = {alias}
                col_types = {alias: "bool"}
                numeric = set()
                strings = set()
        elif kind == "index_bool_probe":
            alias = str(op.get("as", ""))
            if not alias:
                errors.append(f"op {idx}: index_bool_probe output alias is empty")
            elif is_reserved_output_name(alias):
                errors.append(f"op {idx}: index_bool_probe output alias {alias!r} is reserved")
            if alias:
                available = {alias}
                col_types = {alias: "bool"}
                numeric = set()
                strings = set()
        elif kind == "empty_literal_groupby_probe":
            alias = str(op.get("as", ""))
            if not alias:
                errors.append(f"op {idx}: empty_literal_groupby_probe output alias is empty")
            elif is_reserved_output_name(alias):
                errors.append(f"op {idx}: empty_literal_groupby_probe output alias {alias!r} is reserved")
            if alias:
                available = {alias}
                col_types = {alias: "bool"}
                numeric = set()
                strings = set()
        elif kind == "arrow_string_eq_sum_probe":
            alias = str(op.get("as", ""))
            if not alias:
                errors.append(f"op {idx}: arrow_string_eq_sum_probe output alias is empty")
            elif is_reserved_output_name(alias):
                errors.append(f"op {idx}: arrow_string_eq_sum_probe output alias {alias!r} is reserved")
            if alias:
                available = {alias}
                col_types = {alias: "bool"}
                numeric = set()
                strings = set()
        elif kind == "arrow_timestamp_loc_slice_probe":
            alias = str(op.get("as", ""))
            if not alias:
                errors.append(f"op {idx}: arrow_timestamp_loc_slice_probe output alias is empty")
            elif is_reserved_output_name(alias):
                errors.append(f"op {idx}: arrow_timestamp_loc_slice_probe output alias {alias!r} is reserved")
            if alias:
                available = {alias}
                col_types = {alias: "bool"}
                numeric = set()
                strings = set()
        elif kind == "arrow_timestamp_index_attr_probe":
            alias = str(op.get("as", ""))
            if not alias:
                errors.append(f"op {idx}: arrow_timestamp_index_attr_probe output alias is empty")
            elif is_reserved_output_name(alias):
                errors.append(f"op {idx}: arrow_timestamp_index_attr_probe output alias {alias!r} is reserved")
            if alias:
                available = {alias}
                col_types = {alias: "bool"}
                numeric = set()
                strings = set()
        elif kind == "eval_inplace_alias_probe":
            alias = str(op.get("as", ""))
            if not alias:
                errors.append(f"op {idx}: eval_inplace_alias_probe output alias is empty")
            elif is_reserved_output_name(alias):
                errors.append(f"op {idx}: eval_inplace_alias_probe output alias {alias!r} is reserved")
            if alias:
                available = {alias}
                col_types = {alias: "bool"}
                numeric = set()
                strings = set()
        elif kind == "bool_reduction_skipna_probe":
            alias = str(op.get("as", ""))
            if not alias:
                errors.append(f"op {idx}: bool_reduction_skipna_probe output alias is empty")
            elif is_reserved_output_name(alias):
                errors.append(f"op {idx}: bool_reduction_skipna_probe output alias {alias!r} is reserved")
            if alias:
                available = {alias}
                col_types = {alias: "bool"}
                numeric = set()
                strings = set()
        elif kind == "dataset_isin_all_match_probe":
            alias = str(op.get("as", ""))
            if not alias:
                errors.append(f"op {idx}: dataset_isin_all_match_probe output alias is empty")
            elif is_reserved_output_name(alias):
                errors.append(f"op {idx}: dataset_isin_all_match_probe output alias {alias!r} is reserved")
            if alias:
                available = {alias}
                col_types = {alias: "bool"}
                numeric = set()
                strings = set()
        elif kind == "run_end_null_compute_probe":
            alias = str(op.get("as", ""))
            if not alias:
                errors.append(f"op {idx}: run_end_null_compute_probe output alias is empty")
            elif is_reserved_output_name(alias):
                errors.append(f"op {idx}: run_end_null_compute_probe output alias {alias!r} is reserved")
            if alias:
                available = {alias}
                col_types = {alias: "bool"}
                numeric = set()
                strings = set()
        elif kind == "large_string_partition_probe":
            alias = str(op.get("as", ""))
            if not alias:
                errors.append(f"op {idx}: large_string_partition_probe output alias is empty")
            elif is_reserved_output_name(alias):
                errors.append(f"op {idx}: large_string_partition_probe output alias {alias!r} is reserved")
            if alias:
                available = {alias}
                col_types = {alias: "bool"}
                numeric = set()
                strings = set()
        elif kind == "hash_pivot_wider_probe":
            alias = str(op.get("as", ""))
            if not alias:
                errors.append(f"op {idx}: hash_pivot_wider_probe output alias is empty")
            elif is_reserved_output_name(alias):
                errors.append(f"op {idx}: hash_pivot_wider_probe output alias {alias!r} is reserved")
            if alias:
                available = {alias}
                col_types = {alias: "bool"}
                numeric = set()
                strings = set()
        elif kind == "rolling_mean_by_null_count_probe":
            alias = str(op.get("as", ""))
            if not alias:
                errors.append(f"op {idx}: rolling_mean_by_null_count_probe output alias is empty")
            elif is_reserved_output_name(alias):
                errors.append(f"op {idx}: rolling_mean_by_null_count_probe output alias {alias!r} is reserved")
            if alias:
                available = {alias}
                col_types = {alias: "bool"}
                numeric = set()
                strings = set()
        elif kind == "csv_long_numeric_roundtrip_probe":
            alias = str(op.get("as", ""))
            if not alias:
                errors.append(f"op {idx}: csv_long_numeric_roundtrip_probe output alias is empty")
            elif is_reserved_output_name(alias):
                errors.append(f"op {idx}: csv_long_numeric_roundtrip_probe output alias {alias!r} is reserved")
            if not _valid_digit_string_values(op.get("values", [])):
                errors.append(f"op {idx}: csv_long_numeric_roundtrip_probe requires non-empty digit-string values")
            if alias:
                available = {alias}
                col_types = {alias: "bool"}
                numeric = set()
                strings = set()
        elif kind == "select":
            cols = list(op.get("columns", []))
            missing = [col for col in cols if col not in available]
            if missing:
                errors.append(f"op {idx}: select columns unavailable: {missing}")
            if not cols:
                errors.append(f"op {idx}: select has no columns")
            deduped = unique_preserve_order(cols)
            if len(deduped) != len(cols):
                errors.append(f"op {idx}: select contains duplicate columns")
            available = set(deduped) & available
            numeric &= available
            strings &= available
        elif kind == "sort":
            try:
                keys = normalize_sort_keys(op)
            except ValueError as exc:
                errors.append(f"op {idx}: invalid sort keys: {exc}")
                continue
            cols = [key.column for key in keys]
            missing = [col for col in cols if col not in available]
            if missing:
                errors.append(f"op {idx}: sort columns unavailable: {missing}")
            if not cols:
                errors.append(f"op {idx}: sort has no columns")
            if len(unique_preserve_order(cols)) != len(cols):
                errors.append(f"op {idx}: sort contains duplicate columns")
        elif kind == "limit":
            try:
                if int(op.get("n", -1)) < 0:
                    errors.append(f"op {idx}: negative limit")
            except (TypeError, ValueError):
                errors.append(f"op {idx}: non-integer limit {op.get('n')!r}")
        elif kind == "offset":
            try:
                if int(op.get("n", -1)) < 0:
                    errors.append(f"op {idx}: negative offset")
            except (TypeError, ValueError):
                errors.append(f"op {idx}: non-integer offset {op.get('n')!r}")
        elif kind == "mutate":
            out_type = _mutate_output_type(op.get("expr", {}), available, numeric, strings, col_types)
            if out_type is None:
                errors.append(f"op {idx}: invalid mutate expression {op.get('expr')!r}")
                continue
            column = str(op.get("column", ""))
            if not column:
                errors.append(f"op {idx}: mutate output column is empty")
                continue
            if is_reserved_output_name(column):
                errors.append(f"op {idx}: mutate output column {column!r} is reserved")
                continue
            available.add(column)
            col_types[column] = out_type
            if out_type in {"int", "float"}:
                numeric.add(column)
            if out_type == "str":
                strings.add(column)
        elif kind == "groupby":
            keys = list(op.get("keys", []))
            aggs = list(op.get("aggs", []))
            missing_keys = [key for key in keys if key not in available]
            if missing_keys:
                errors.append(f"op {idx}: groupby keys unavailable: {missing_keys}")
            if not keys:
                errors.append(f"op {idx}: groupby has no keys")
            if len(unique_preserve_order(keys)) != len(keys):
                errors.append(f"op {idx}: groupby contains duplicate keys")
            if not aggs:
                errors.append(f"op {idx}: groupby has no aggregations")
            aliases = [str(agg.get("as")) for agg in aggs if agg.get("as")]
            if len(unique_preserve_order(aliases)) != len(aliases):
                errors.append(f"op {idx}: groupby contains duplicate aggregation aliases")
            key_set = {str(key) for key in keys}
            colliding_aliases = [alias for alias in aliases if alias in key_set]
            if colliding_aliases:
                errors.append(f"op {idx}: groupby aggregation aliases collide with keys: {colliding_aliases}")
            reserved_aliases = [alias for alias in aliases if is_reserved_output_name(alias)]
            if reserved_aliases:
                errors.append(f"op {idx}: groupby aggregation aliases use reserved names: {reserved_aliases}")
            for agg in aggs:
                col = agg.get("column")
                if col not in available:
                    errors.append(f"op {idx}: aggregation column {col!r} is unavailable")
                func = agg.get("func")
                if not _aggregate_accepts_type(col_types.get(str(col), "derived"), str(func), col in numeric):
                    errors.append(f"op {idx}: aggregation column {col!r} is not numeric")
                if func not in {"sum", "mean", "min", "max", "count", "nunique", "any", "all"}:
                    errors.append(f"op {idx}: unsupported aggregation {func!r}")
            available = set(keys) | {str(agg.get("as")) for agg in aggs if agg.get("as")}
            numeric = {key for key in keys if col_types.get(key) in {"int", "float"}}
            strings = {key for key in keys if col_types.get(key) == "str"}
            for agg in aggs:
                if agg.get("as"):
                    output_type = _aggregate_result_type(
                        col_types.get(str(agg.get("column")), "float"),
                        str(agg.get("func", "")),
                    )
                    col_types[str(agg["as"])] = output_type
                    if output_type in {"int", "float"}:
                        numeric.add(str(agg["as"]))
        elif kind == "aggregate":
            aggs = list(op.get("aggs", []))
            if not aggs:
                errors.append(f"op {idx}: aggregate has no aggregations")
            aliases = [str(agg.get("as")) for agg in aggs if agg.get("as")]
            if len(unique_preserve_order(aliases)) != len(aliases):
                errors.append(f"op {idx}: aggregate contains duplicate aggregation aliases")
            reserved_aliases = [alias for alias in aliases if is_reserved_output_name(alias)]
            if reserved_aliases:
                errors.append(f"op {idx}: aggregate aliases use reserved names: {reserved_aliases}")
            for agg in aggs:
                col = agg.get("column")
                if col not in available:
                    errors.append(f"op {idx}: aggregation column {col!r} is unavailable")
                func = agg.get("func")
                if not _aggregate_accepts_type(col_types.get(str(col), "derived"), str(func), col in numeric):
                    errors.append(f"op {idx}: aggregation column {col!r} is not numeric")
                if func not in {"sum", "mean", "min", "max", "count", "nunique", "any", "all"}:
                    errors.append(f"op {idx}: unsupported aggregation {func!r}")
            available = {str(agg.get("as")) for agg in aggs if agg.get("as")}
            numeric = set()
            strings = set()
            for agg in aggs:
                if agg.get("as"):
                    output_type = _aggregate_result_type(
                        col_types.get(str(agg.get("column")), "float"),
                        str(agg.get("func", "")),
                    )
                    col_types[str(agg["as"])] = output_type
                    if output_type in {"int", "float"}:
                        numeric.add(str(agg["as"]))
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
        return Classification(
            verdict="candidate_implementation_bug",
            paper_status="candidate_bug_needs_external_confirmation",
            confidence="high",
            evidence=(
                "Single backend violates a semantics-preserving metamorphic relation: "
                f"{suspicious[0]}"
            ),
            recommendation=[
                "Minimize the metamorphic pair and rerun the same backend on base and variant cases.",
                "If the relation is still violated, file as a backend/adaptor implementation bug.",
            ],
        )
    return Classification(
        verdict="semantic_divergence_needs_confirmation",
        paper_status="valid_finding_not_confirmed_bug",
        confidence="medium",
        evidence="Metamorphic relation violation involves multiple or ambiguous backends.",
        recommendation=[
            "Inspect whether the metamorphic relation is valid for this case before counting it as a bug.",
        ],
    )


def _reference_classification(
    case: Case,
    finding: Finding | dict[str, Any],
    normalized: dict[str, NormalizedResult | dict[str, Any]],
    backends: list[str],
) -> Classification | None:
    if not normalized:
        return None
    reference = _reference_result(case)
    if reference is None:
        return None
    expected = _result_payload(reference)
    matching: list[str] = []
    mismatching: list[str] = []
    for backend, result in normalized.items():
        if _result_get(result, "status") == "normalization_error":
            continue
        if _result_payload(result) == expected:
            matching.append(backend)
        else:
            mismatching.append(backend)
    if not mismatching:
        return None

    suspicious = set(_get(finding, "suspicious_backends", []) or [])
    implicated = sorted((set(mismatching) & suspicious) or set(mismatching))
    if matching and implicated:
        confidence = "high" if len(implicated) < max(1, len(backends)) else "medium"
        return Classification(
            verdict="candidate_implementation_bug",
            paper_status="candidate_bug_needs_external_confirmation",
            confidence=confidence,
            evidence=(
                "Independent DSL reference agrees with "
                f"{sorted(matching)} and disagrees with {implicated}."
            ),
            implicated_backends=implicated,
            recommendation=[
                "Minimize the case and include the DSL reference output in the artifact.",
                "Treat as confirmed only after backend documentation or maintainers establish the expected behavior.",
            ],
        )

    if not matching:
        return Classification(
            verdict="needs_manual_confirmation",
            paper_status="valid_finding_needs_triage",
            confidence="medium",
            evidence="No tested backend matches the independent DSL reference output.",
            recommendation=[
                "Inspect the DSL reference semantics before making a backend bug claim.",
                "This may indicate a reference-oracle bug or an underspecified DSL operation.",
            ],
        )
    return None


def _mutate_output_type(
    expr: dict[str, Any],
    available: set[str],
    numeric: set[str],
    strings: set[str],
    col_types: dict[str, str],
) -> str | None:
    kind = expr.get("kind")
    src = expr.get("source")
    if src not in available:
        return None
    if kind == "add_const":
        return col_types[src] if src in numeric else None
    if kind == "arith_const":
        if src not in numeric or expr.get("op") not in {"sub", "mul", "div", "mod"}:
            return None
        if expr.get("op") in {"div", "mod"} and expr.get("value") == 0:
            return None
        return "float" if expr.get("op") == "div" or col_types[src] == "float" else col_types[src]
    if kind == "reverse_division_columns":
        numerator = expr.get("numerator")
        if src not in numeric or numerator not in numeric:
            return None
        return "float"
    if kind == "cast":
        return "float" if src in numeric and expr.get("to") == "float" else None
    if kind == "string_length":
        return "int" if src in strings else None
    if kind == "string_lower":
        return "str" if src in strings else None
    if kind == "string_basename":
        return "str" if src in strings else None
    return None


def _valid_group_quantile_values(values: Any, quantiles: Any) -> bool:
    if not isinstance(values, list) or not isinstance(quantiles, list):
        return False
    if len(values) < 2 or len(quantiles) < 2:
        return False
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
        return False
    return all(
        not isinstance(quantile, bool)
        and isinstance(quantile, (int, float))
        and 0.0 <= float(quantile) <= 1.0
        for quantile in quantiles
    )


def _valid_float_literal_text(value: Any) -> bool:
    text = str(value or "").strip()
    if not text or not any(ch.isdigit() for ch in text):
        return False
    allowed = set("0123456789+-.eE")
    if any(ch not in allowed for ch in text):
        return False
    try:
        float(text)
    except ValueError:
        return False
    return True


def _valid_digit_string_values(values: Any) -> bool:
    return (
        isinstance(values, list)
        and bool(values)
        and all(isinstance(value, str) and value.isdigit() for value in values)
    )


def _filter_literal_error(column_type: str, comparator: Any, value: Any) -> str:
    if not filter_comparator_supports_type(column_type, comparator):
        return f"comparator {comparator!r} is not supported for {column_type} filter"
    parsed = parse_filter_comparator(comparator)
    if parsed is not None and parsed.base == "in_set":
        if not isinstance(value, list) or not value:
            return f"filter literal {value!r} is not a non-empty list for in_set"
        if any(item is None for item in value):
            return "in_set filter literals must not contain NULL"
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


def _reference_result(case: Case) -> NormalizedResult | None:
    try:
        if validate_case_program(case):
            return None
        columns = [column.name for column in case.tables[0].columns]
        rows = [
            {column: row.get(column) for column in columns}
            for row in case.tables[0].rows
        ]
        tables = {table.name: table for table in case.tables}
        for op in case.program.operations:
            kind = op.get("op")
            if kind == "join":
                right = tables[str(op["table"])]
                right_columns = [
                    column.name
                    for column in right.columns
                    if column.name != op["right_on"] and column.name not in columns
                ]
                index: dict[Any, list[dict[str, Any]]] = {}
                for right_row in right.rows:
                    key = right_row.get(op["right_on"])
                    if key is None:
                        continue
                    index.setdefault(key, []).append(right_row)
                joined = []
                for left_row in rows:
                    key = left_row.get(op["left_on"])
                    matches = [] if key is None else index.get(key, [])
                    if matches:
                        for right_row in matches:
                            out = dict(left_row)
                            for column in right_columns:
                                out[column] = right_row.get(column)
                            joined.append(out)
                    elif op["how"] == "left":
                        out = dict(left_row)
                        for column in right_columns:
                            out[column] = None
                        joined.append(out)
                rows = joined
                columns.extend(right_columns)
            elif kind == "filter":
                rows = [row for row in rows if _reference_compare(row.get(op["column"]), op["cmp"], op.get("value"))]
            elif kind == "tuple_absence_filter":
                right = tables[str(op["table"])]
                left_columns = list(op["columns"])
                right_columns = list(op["right_columns"])
                rows = [
                    row
                    for row in rows
                    if evaluate_tuple_absence(row, left_columns, right.rows, right_columns)
                ]
            elif kind == "row_number_filter":
                rows = row_number_filter_rows(rows, op)
            elif kind == "running_sum":
                sort_keys = running_sum_sort_keys(op)
                rows = sort_rows_for_running(rows, sort_keys)
                column = str(op["column"])
                values = stable_running_sum_values(rows, str(op["source"]), running_sum_partition_columns(op))
                rows = [{**row, column: value} for row, value in zip(rows, values)]
                columns = [name for name in columns if name != column] + [column]
            elif kind == "sortedness_check":
                alias = str(op["as"])
                ok = is_sorted_values(
                    [row.get(op["column"]) for row in rows],
                    ascending=bool(op.get("ascending", True)),
                    nulls=str(op.get("nulls", "last")),
                )
                columns = [alias]
                rows = [{alias: ok}]
            elif kind == "random_case_probe":
                alias = str(op["as"])
                columns = [alias]
                rows = [{alias: False}]
            elif kind == "group_quantile_probe":
                alias = str(op["as"])
                columns = [alias]
                rows = [{alias: False}]
            elif kind == "scalar_subquery_probe":
                alias = str(op["as"])
                columns = [alias]
                rows = [{alias: False}]
            elif kind == "window_avg_probe":
                alias = str(op["as"])
                columns = [alias]
                rows = [{alias: False}]
            elif kind == "struct_distinct_probe":
                alias = str(op["as"])
                columns = [alias]
                rows = [{alias: False}]
            elif kind == "bit_compare_probe":
                alias = str(op["as"])
                columns = [alias]
                rows = [{alias: False}]
            elif kind == "round_even_probe":
                alias = str(op["as"])
                columns = [alias]
                rows = [{alias: False}]
            elif kind == "float_literal_precision_probe":
                alias = str(op["as"])
                columns = [alias]
                rows = [{alias: False}]
            elif kind == "timestamp_precision_filter_probe":
                alias = str(op["as"])
                columns = [alias]
                rows = [{alias: False}]
            elif kind == "series_rtruediv_probe":
                alias = str(op["as"])
                columns = [alias]
                rows = [{alias: False}]
            elif kind == "uint64_isin_probe":
                alias = str(op["as"])
                columns = [alias]
                rows = [{alias: False}]
            elif kind == "tuple_anti_null_probe":
                alias = str(op["as"])
                columns = [alias]
                rows = [{alias: False}]
            elif kind == "setop_all_duplicate_probe":
                alias = str(op["as"])
                columns = [alias]
                rows = [{alias: False}]
            elif kind == "json_predicate_order_probe":
                alias = str(op["as"])
                columns = [alias]
                rows = [{alias: False}]
            elif kind == "sparse_mask_probe":
                alias = str(op["as"])
                columns = [alias]
                rows = [{alias: False}]
            elif kind == "float_wrap_probe":
                alias = str(op["as"])
                columns = [alias]
                rows = [{alias: False}]
            elif kind == "index_bool_probe":
                alias = str(op["as"])
                columns = [alias]
                rows = [{alias: False}]
            elif kind == "empty_literal_groupby_probe":
                alias = str(op["as"])
                columns = [alias]
                rows = [{alias: False}]
            elif kind == "arrow_string_eq_sum_probe":
                alias = str(op["as"])
                columns = [alias]
                rows = [{alias: False}]
            elif kind == "arrow_timestamp_loc_slice_probe":
                alias = str(op["as"])
                columns = [alias]
                rows = [{alias: False}]
            elif kind == "arrow_timestamp_index_attr_probe":
                alias = str(op["as"])
                columns = [alias]
                rows = [{alias: False}]
            elif kind == "eval_inplace_alias_probe":
                alias = str(op["as"])
                columns = [alias]
                rows = [{alias: False}]
            elif kind == "bool_reduction_skipna_probe":
                alias = str(op["as"])
                columns = [alias]
                rows = [{alias: False}]
            elif kind == "dataset_isin_all_match_probe":
                alias = str(op["as"])
                columns = [alias]
                rows = [{alias: False}]
            elif kind == "run_end_null_compute_probe":
                alias = str(op["as"])
                columns = [alias]
                rows = [{alias: False}]
            elif kind == "large_string_partition_probe":
                alias = str(op["as"])
                columns = [alias]
                rows = [{alias: False}]
            elif kind == "hash_pivot_wider_probe":
                alias = str(op["as"])
                columns = [alias]
                rows = [{alias: False}]
            elif kind == "rolling_mean_by_null_count_probe":
                alias = str(op["as"])
                columns = [alias]
                rows = [{alias: False}]
            elif kind == "csv_long_numeric_roundtrip_probe":
                alias = str(op["as"])
                columns = [alias]
                rows = [{alias: False}]
            elif kind == "select":
                columns = list(op["columns"])
                rows = [{column: row.get(column) for column in columns} for row in rows]
            elif kind == "sort":
                sort_keys = normalize_sort_keys(op)
                rows = sorted(
                    rows,
                    key=cmp_to_key(lambda left, right: _compare_rows(left, right, sort_keys)),
                )
            elif kind == "limit":
                rows = rows[: int(op["n"])]
            elif kind == "offset":
                rows = rows[int(op["n"]):]
            elif kind == "mutate":
                column = str(op["column"])
                rows = [{**row, column: _reference_eval_expr(row, op["expr"])} for row in rows]
                if column not in columns:
                    columns.append(column)
            elif kind == "groupby":
                keys = list(op["keys"])
                grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
                order: list[tuple[Any, ...]] = []
                for row in rows:
                    key = tuple(row.get(column) for column in keys)
                    if key not in grouped:
                        grouped[key] = []
                        order.append(key)
                    grouped[key].append(row)
                out_rows = []
                for key in order:
                    group_rows = grouped[key]
                    out = {column: value for column, value in zip(keys, key)}
                    for agg in op["aggs"]:
                        out[agg["as"]] = _reference_aggregate(group_rows, agg["column"], agg["func"])
                    out_rows.append(out)
                columns = keys + [agg["as"] for agg in op["aggs"]]
                rows = out_rows
            elif kind == "aggregate":
                out = {}
                for agg in op["aggs"]:
                    out[agg["as"]] = _reference_aggregate(rows, agg["column"], agg["func"])
                columns = [agg["as"] for agg in op["aggs"]]
                rows = [out]
            else:
                return None
        return _normalize_reference_rows(columns, rows, preserve_order=case.program.order_sensitive)
    except Exception:
        return None


def _normalize_reference_rows(
    columns: list[str],
    rows: list[dict[str, Any]],
    *,
    preserve_order: bool = False,
) -> NormalizedResult:
    column_positions = sorted(enumerate(columns), key=lambda item: (item[1], item[0]))
    out_columns = [name for _, name in column_positions]
    out_rows = [
        [
            _norm_value(row.get(columns[idx]), preserve_float_precision=preserve_order)
            for idx, _ in column_positions
        ]
        for row in rows
    ]
    if not preserve_order:
        out_rows = sorted(out_rows, key=_stable_row_key)
    return NormalizedResult("dsl_reference", "ok", out_columns, out_rows)


def _stable_row_key(row: list[Any]) -> str:
    import json

    return json.dumps(row, ensure_ascii=False, sort_keys=True)


def _reference_compare(left: Any, comparator: str, right: Any) -> bool:
    return evaluate_filter_predicate(left, comparator, right)


def _reference_eval_expr(row: dict[str, Any], expr: dict[str, Any]) -> Any:
    value = row.get(expr.get("source"))
    if value is None:
        return None
    kind = expr.get("kind")
    if kind == "add_const":
        return value + expr["value"]
    if kind == "arith_const":
        op = expr["op"]
        rhs = expr["value"]
        if op == "sub":
            return value - rhs
        if op == "mul":
            return value * rhs
        if op == "div":
            return value / rhs
        if op == "mod":
            return value % rhs
    if kind == "reverse_division_columns":
        numerator = row.get(expr.get("numerator"))
        if numerator is None or value == 0:
            return None
        return numerator / value
    if kind == "cast" and expr.get("to") == "float":
        return float(value)
    if kind == "string_length":
        return len(value)
    if kind == "string_lower":
        return value.lower()
    if kind == "string_basename":
        return path_basename(value)
    raise ValueError(kind)


def _reference_aggregate(rows: list[dict[str, Any]], column: str, func: str) -> Any:
    values = [row.get(column) for row in rows if row.get(column) is not None]
    if func == "count":
        return len(values)
    if func == "nunique":
        return len(set(values))
    if not values:
        return None
    if func == "any":
        return any(bool(value) for value in values)
    if func == "all":
        return all(bool(value) for value in values)
    if func == "sum":
        return sum(values)
    if func == "mean":
        return sum(values) / len(values)
    if func == "min":
        return min(values)
    if func == "max":
        return max(values)
    raise ValueError(func)


def _aggregate_accepts_type(source_type: str, func: str, is_numeric_column: bool) -> bool:
    if func in {"count", "nunique"}:
        return True
    if func in {"any", "all"}:
        return source_type == "bool"
    if func in {"min", "max"} and source_type == "bool":
        return True
    return is_numeric_column


def _aggregate_result_type(source_type: str, func: str) -> str:
    if func in {"count", "nunique"}:
        return "int"
    if func in {"any", "all"}:
        return "bool"
    if func == "mean":
        return "float"
    return source_type


def _compare_rows(left: dict[str, Any], right: dict[str, Any], sort_keys: list[SortKey]) -> int:
    for key in sort_keys:
        left_value = left.get(key.column)
        right_value = right.get(key.column)
        if left_value is None and right_value is None:
            continue
        if left_value is None:
            return -1 if key.nulls == "first" else 1
        if right_value is None:
            return 1 if key.nulls == "first" else -1
        cmp = _compare_values(left_value, right_value)
        if cmp:
            return cmp if key.ascending else -cmp
    return 0


def _compare_values(left: Any, right: Any) -> int:
    if left < right:
        return -1
    if left > right:
        return 1
    return 0


def _normalizer_errors(normalized: dict[str, NormalizedResult | dict[str, Any]]) -> list[str]:
    out = []
    for backend, result in normalized.items():
        if _result_get(result, "status") == "normalization_error":
            out.append(f"{backend}:{_result_get(result, 'error_type')}")
    return out


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


def _is_documented_semantic_divergence(case: Case, finding: Finding | dict[str, Any], config: dict[str, Any]) -> bool:
    root = str(_get(finding, "root_cause", ""))
    suspicious = set(_get(finding, "suspicious_backends", []) or [])
    if (
        root == "nan_inf_semantics"
        and suspicious == {"polars"}
        and _case_contains_special_float(case)
    ):
        return True
    return False


def _is_semantic_boundary(case: Case, finding: Finding | dict[str, Any], config: dict[str, Any]) -> bool:
    return bool(_semantic_boundary_reasons(case, finding, config))


def _semantic_boundary_reasons(case: Case, finding: Finding | dict[str, Any], config: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    root = str(_get(finding, "root_cause", ""))
    if root in {"nan_inf_semantics", "null_semantics", "ordering_or_limit"}:
        reasons.append(f"root cause {root} is a known cross-engine semantic boundary")
    if root == "nan_inf_semantics" and _case_contains_special_float(case):
        reasons.append("case contains NaN or Infinity values")
    if _join_keys_contain_null(case):
        reasons.append("join key contains NULL values; NULL join semantics differ across target families")
    if _case_has_null_filter_literal(case):
        reasons.append("filter compares against NULL; engines intentionally differ on NULL predicate semantics")
    if _case_uses_modulo(case):
        reasons.append("case uses modulo; negative/float remainder semantics differ across engines")
    if _case_uses_string_lower(case) and _case_contains_non_ascii_string(case):
        reasons.append("case lowercases non-ASCII text; Unicode case mapping support differs across engines")
    return unique_preserve_order(reasons)


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
    ordered_rows = [_stable_rows(_result_get(result, "rows", [])) for result in ok_results]
    if len({tuple(rows) for rows in ordered_rows}) <= 1:
        return False
    relaxed_rows = [
        [_relaxed_float_precision_row(row) for row in _result_get(result, "rows", [])]
        for result in ok_results
    ]
    relaxed_ordered = [_stable_rows(rows) for rows in relaxed_rows]
    if len({tuple(rows) for rows in relaxed_ordered}) <= 1:
        return True
    relaxed_unordered = [tuple(sorted(rows)) for rows in relaxed_ordered]
    return len(set(relaxed_unordered)) == 1


def _relaxed_float_precision_row(row: list[Any]) -> list[Any]:
    return [_norm_value(value, preserve_float_precision=False) for value in row]


def _case_uses_precision_sensitive_float_arithmetic(case: Case) -> bool:
    if not case.tables:
        return False
    table_by_name = {table.name: table for table in case.tables}
    column_types = {column.name: column.type for column in case.tables[0].columns}
    float_lineage = {column.name for column in case.tables[0].columns if column.type == "float"}
    precision_columns: set[str] = set(float_lineage)
    saw_precision_arithmetic = False
    for operation in case.program.operations:
        operation_kind = operation.get("op")
        if operation_kind == "join":
            right_table = table_by_name.get(str(operation.get("table", "")))
            if right_table is None:
                continue
            right_key = str(operation.get("right_on", ""))
            for column in right_table.columns:
                if column.name == right_key or column.name in column_types:
                    continue
                column_types[column.name] = column.type
                if column.type == "float":
                    float_lineage.add(column.name)
                    precision_columns.add(column.name)
        elif operation_kind == "select":
            selected_columns = {str(column) for column in operation.get("columns", [])}
            column_types = {name: value_type for name, value_type in column_types.items() if name in selected_columns}
            float_lineage &= selected_columns
            precision_columns &= selected_columns
        elif operation_kind == "mutate":
            output_column = str(operation.get("column", ""))
            expression = operation.get("expr", {})
            source_name = str(expression.get("source", ""))
            result_type = _mutate_float_precision_result_type(expression, column_types)
            if not output_column or result_type is None:
                continue
            column_types[output_column] = result_type
            if result_type != "float":
                continue
            source_is_float = source_name in float_lineage or column_types.get(source_name) == "float"
            expression_kind = expression.get("kind")
            expression_operator = expression.get("op")
            precision_sensitive = (
                expression_kind == "cast"
                or source_is_float
                or expression_operator in {"div", "mul"}
            )
            if precision_sensitive:
                saw_precision_arithmetic = True
                float_lineage.add(output_column)
                precision_columns.add(output_column)
        elif operation_kind in {"groupby", "aggregate"}:
            input_column_types = dict(column_types)
            aggregate_outputs = _float_precision_aggregate_outputs(operation, input_column_types, precision_columns)
            if aggregate_outputs:
                saw_precision_arithmetic = True
            if operation_kind == "groupby":
                group_keys = [str(key) for key in operation.get("keys", [])]
                column_types = {key: input_column_types.get(key, "derived") for key in group_keys}
                precision_columns = {key for key in group_keys if key in precision_columns}
                float_lineage = {key for key in group_keys if key in float_lineage}
            else:
                column_types = {}
                precision_columns = set()
                float_lineage = set()
            for aggregate in operation.get("aggs", []):
                output_column = str(aggregate.get("as", ""))
                if not output_column:
                    continue
                source_type = input_column_types.get(str(aggregate.get("column", "")), "float")
                result_type = _aggregate_result_type(source_type, str(aggregate.get("func", "")))
                column_types[output_column] = result_type
                if output_column in aggregate_outputs:
                    precision_columns.add(output_column)
                    float_lineage.add(output_column)
        elif operation_kind == "sort":
            try:
                sort_columns = {sort_key.column for sort_key in normalize_sort_keys(operation)}
            except ValueError:
                sort_columns = set()
            if saw_precision_arithmetic and sort_columns & precision_columns:
                return True
    return saw_precision_arithmetic


def _mutate_float_precision_result_type(
    expression: dict[str, Any],
    column_types: dict[str, str],
) -> str | None:
    source_name = str(expression.get("source", ""))
    source_type = column_types.get(source_name)
    if source_type is None:
        return None
    expression_kind = expression.get("kind")
    if expression_kind == "add_const":
        return source_type if source_type in {"int", "float"} else None
    if expression_kind == "arith_const":
        expression_operator = expression.get("op")
        if source_type not in {"int", "float"} or expression_operator not in {"sub", "mul", "div", "mod"}:
            return None
        return "float" if expression_operator == "div" or source_type == "float" else source_type
    if expression_kind == "cast" and expression.get("to") == "float":
        return "float" if source_type in {"int", "float"} else None
    return None


def _float_precision_aggregate_outputs(
    operation: dict[str, Any],
    column_types: dict[str, str],
    precision_columns: set[str],
) -> set[str]:
    outputs: set[str] = set()
    for aggregate in operation.get("aggs", []):
        source_name = str(aggregate.get("column", ""))
        aggregate_function = str(aggregate.get("func", ""))
        output_column = str(aggregate.get("as", ""))
        if not output_column:
            continue
        source_type = column_types.get(source_name)
        if aggregate_function in {"sum", "mean"} and (
            source_name in precision_columns or source_type == "float"
        ):
            outputs.add(output_column)
    return outputs


def _has_clear_minority_backend(finding: Finding | dict[str, Any], backends: list[str]) -> bool:
    suspicious = list(_get(finding, "suspicious_backends", []) or [])
    confidence = _get(finding, "confidence", "")
    return 0 < len(suspicious) < max(1, len(backends)) and confidence == "high"


def _is_order_only_mismatch(normalized: dict[str, NormalizedResult | dict[str, Any]]) -> bool:
    ok_results = [result for result in normalized.values() if _result_get(result, "status") == "ok"]
    if len(ok_results) < 2:
        return False
    payloads = [(_result_get(result, "columns", []), _result_get(result, "rows", [])) for result in ok_results]
    first_columns = payloads[0][0]
    if any(columns != first_columns for columns, _ in payloads):
        return False
    ordered = [_stable_rows(rows) for _, rows in payloads]
    if len({tuple(rows) for rows in ordered}) <= 1:
        return False
    unordered = [tuple(sorted(rows)) for rows in ordered]
    return len(set(unordered)) == 1


def _stable_rows(rows: list[list[Any]]) -> list[str]:
    import json

    return [json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows]


def _is_sort_tie_order_only_mismatch(case: Case) -> bool:
    op_index = _last_order_defining_operation_index(case)
    if op_index is None:
        return False
    op = case.program.operations[op_index]
    if op.get("op") != "sort":
        return False
    try:
        sort_keys = normalize_sort_keys(op)
    except ValueError:
        return False
    rows = _reference_rows_before_operation(case, op_index)
    if rows is None:
        return False
    return _rows_have_duplicate_sort_key(rows, sort_keys)


def _has_limit_or_offset_without_defined_order(case: Case) -> bool:
    order_defined = False
    for op in case.program.operations:
        kind = op.get("op")
        if kind in {"sort", "running_sum", "row_number_filter"}:
            order_defined = True
            continue
        if kind == "limit":
            try:
                limit = int(op.get("n", 0))
            except (TypeError, ValueError):
                limit = 1
            if limit > 0 and not order_defined:
                return True
            continue
        if kind == "offset":
            try:
                offset = int(op.get("n", 0))
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
        kind = case.program.operations[idx].get("op")
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
    reference = _reference_result(prefix)
    if reference is None or reference.status != "ok":
        return None
    return [
        {column: row[idx] for idx, column in enumerate(reference.columns)}
        for row in reference.rows
    ]


def _rows_have_duplicate_sort_key(rows: list[dict[str, Any]], sort_keys: list[SortKey]) -> bool:
    seen: set[str] = set()
    for row in rows:
        key = _stable_row_key([row.get(sort_key.column) for sort_key in sort_keys])
        if key in seen:
            return True
        seen.add(key)
    return False


def _result_payload(result: NormalizedResult | dict[str, Any]) -> dict[str, Any]:
    return {
        "status": _result_get(result, "status"),
        "columns": _result_get(result, "columns", []),
        "rows": _result_get(result, "rows", []),
        "error_type": _result_get(result, "error_type", ""),
    }


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


def _case_contains_special_float(case: Case) -> bool:
    return any(
        isinstance(value, float) and (math.isnan(value) or math.isinf(value))
        for table in case.tables
        for row in table.rows
        for value in row.values()
    )


def _join_keys_contain_null(case: Case) -> bool:
    tables = {table.name: table for table in case.tables}
    for op in case.program.operations:
        if op.get("op") != "join":
            continue
        left_key = op.get("left_on")
        right = tables.get(str(op.get("table", "")))
        right_key = op.get("right_on")
        left_has_null = any(row.get(left_key) is None for row in case.tables[0].rows)
        right_has_null = bool(right and any(row.get(right_key) is None for row in right.rows))
        if left_has_null or right_has_null:
            return True
    return False


def _case_uses_string_lower(case: Case) -> bool:
    return any(
        op.get("op") == "mutate" and op.get("expr", {}).get("kind") == "string_lower"
        for op in case.program.operations
    )


def _case_uses_modulo(case: Case) -> bool:
    return any(
        op.get("op") == "mutate"
        and op.get("expr", {}).get("kind") == "arith_const"
        and op.get("expr", {}).get("op") == "mod"
        for op in case.program.operations
    )


def _case_has_null_filter_literal(case: Case) -> bool:
    return any(
        op.get("op") == "filter"
        and op.get("value") is None
        and _filter_comparator_base(op) not in {"is_null", "is_not_null", "bool_predicate"}
        for op in case.program.operations
    )


def _filter_comparator_base(op: dict[str, Any]) -> str:
    parsed = parse_filter_comparator(op.get("cmp"))
    return parsed.base if parsed is not None else ""


def _case_contains_non_ascii_string(case: Case) -> bool:
    return any(
        isinstance(value, str) and any(ord(ch) > 127 for ch in value)
        for table in case.tables
        for row in table.rows
        for value in row.values()
    )


def _get(finding: Finding | dict[str, Any], key: str, default: Any = None) -> Any:
    if isinstance(finding, dict):
        return finding.get(key, default)
    return getattr(finding, key, default)


def _result_get(result: NormalizedResult | dict[str, Any], key: str, default: Any = "") -> Any:
    if isinstance(result, dict):
        return result.get(key, default)
    return getattr(result, key, default)
