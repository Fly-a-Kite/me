from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

from datadiff.canonicalization import (
    compare_row_set_batch,
    short_canonical_hash,
)
from datadiff.case_features import (
    PROBE_ROOTS,
    case_contains_non_ascii_string as _case_contains_non_ascii_string,
    case_contains_null as _case_contains_null,
    case_contains_special_float as _case_contains_special_float,
    case_has_outer_join_truth_filter as _case_has_outer_join_truth_filter,
    case_has_path_projection_keyed_pick as _case_has_path_projection_keyed_pick,
    case_has_post_topk_filter as _case_has_post_topk_filter,
    case_has_running_sum as _case_has_running_sum,
    case_uses_modulo as _case_uses_modulo,
    case_uses_unicode_case_mapping as _case_uses_unicode_case_mapping,
    last_probe_root as _last_probe_root,
)
from datadiff.dsl import Case, normalize_sort_keys
from datadiff.expression_semantics import cast_output_type, eval_expr_on_row, expr_output_type
from datadiff.filtering import evaluate_filter_predicate, parse_filter_comparator
from datadiff.join_keys import join_key_pairs, join_key_value
from datadiff.normalizer import NormalizedResult
from datadiff.operation_semantics import (
    aggregate_alias,
    aggregate_column,
    aggregate_func,
    aggregate_specs,
    condition_cmp,
    expr_payload,
    expr_numerator,
    expr_kinds,
    expr_input_domain,
    expr_kind,
    expr_other,
    expr_operator,
    expr_source,
    expr_target_type,
    expr_value,
    groupby_keys,
    join_how,
    op_column,
    op_comparator,
    op_columns,
    op_kind,
    op_output_alias,
    op_right_columns,
    op_table,
    op_value,
    operation_names,
)
from datadiff.program_analysis import case_uses_arithmetic_float_lineage
from datadiff.program_state import state_before_first_operation
from datadiff.sample_semantics import (
    distinct_output_samples,
    eval_expr_samples,
    filter_samples,
    groupby_output_samples,
    join_samples,
    rows_from_samples,
    samples_from_rows,
)


@dataclass(slots=True)
class Finding:
    finding_id: str
    kind: str
    severity: str
    suspicious_backends: list[str]
    evidence: str
    signature: str
    root_cause: str = "unknown"
    oracle: str = "differential"
    confidence: str = "medium"
    triage_verdict: str = "unclassified"
    paper_status: str = "unclassified"
    triage_confidence: str = "low"
    false_positive: bool = False
    false_positive_reason: str = ""
    triage_evidence: str = ""
    recommendation: list[str] = field(default_factory=list)
    documentation_refs: list[dict[str, str]] = field(default_factory=list)
    adjudication: dict[str, Any] = field(default_factory=dict)
    mismatch_class: str = ""
    discovery_origin: str = "organic"
    source_issue: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _payload(norm: NormalizedResult) -> dict[str, Any]:
    return norm.comparison_payload()


def _signature(case: Case, normalized: dict[str, NormalizedResult], kind: str) -> str:
    payload = {
        "kind": kind,
        "ops": case.program.op_sequence(),
        "results": {k: _payload(v) for k, v in sorted(normalized.items())},
    }
    return short_canonical_hash(payload, 16)


def classify_root_cause(case: Case, normalized: dict[str, NormalizedResult], kind: str) -> str:
    ops = case.program.op_sequence()
    if kind == "exception_mismatch":
        return "exception_taxonomy"
    probe_root = _last_probe_root(case)
    if probe_root is not None:
        return probe_root
    if _case_has_path_projection_keyed_pick(case):
        return "path_projection_keyed_pick"
    if _case_has_running_sum(case):
        return "running_sum_precision"
    if _case_contains_special_float(case):
        return "nan_inf_semantics"
    if _case_uses_modulo(case):
        return "arithmetic_expression"
    if _case_has_reverse_division_columns(case):
        return "reverse_division_operand_order"
    if _case_has_grouped_topk_null_sort_key(case):
        return "grouped_topk_null_sort_key"
    if _case_has_distinct_null_topk(case):
        return "distinct_null_topk"
    if _case_has_float_group_key_instability(case, normalized):
        return "float_group_key_instability"
    if _case_has_negative_zero_comparison(case):
        return "negative_zero_comparison"
    if _case_has_tuple_absence_filter(case):
        return "tuple_absence_null_filter"
    if _case_uses_unicode_case_mapping(case) and _case_contains_non_ascii_string(case):
        return "unicode_case_mapping"
    if _case_has_outer_join_truth_filter(case):
        return "outer_join_truth_filter"
    if _case_has_post_topk_filter(case):
        return "topk_filter_pushdown"
    if _case_has_joined_order_offset_projection(case):
        return "joined_order_offset_projection"
    if _case_has_ordered_topk_projection(case):
        return "ordered_topk_projection"
    if any(op == "case_when" for op in ops):
        return "conditional_expression"
    if any(op == "union_all" for op in ops):
        return "union_all_row_append"
    if any(op == "drop_nulls" for op in ops):
        return "drop_nulls_null_filter"
    if any(op == "semi_join" for op in ops):
        return "semi_join_membership"
    if any(op == "anti_join" for op in ops):
        return "anti_join_exclusion"
    if any(op == "coalesce" for op in ops):
        return "coalesce_null_semantics"
    if any(op in {"groupby", "aggregate"} for op in ops):
        return "groupby_aggregation"
    if any(op == "join" for op in ops):
        return "join_semantics"
    if any(op == "fill_null" for op in ops):
        return "fill_null_null_semantics"
    if any(op == "distinct" for op in ops):
        return "distinct_duplicate_elimination"
    if any(op == "filter" for op in ops):
        return "filter_predicate"
    if any(op == "mutate" for op in ops):
        kinds = expr_kinds(case.program.operations)
        if kinds & {
            "string_length",
            "string_lower",
            "string_upper",
            "string_strip",
            "string_null_if_empty",
            "string_replace",
            "string_slice",
            "string_split_part",
            "string_basename",
            "string_concat",
            "string_contains",
            "string_starts_with",
            "string_ends_with",
        }:
            return "string_expression"
        if "date_part" in kinds:
            return "datetime_expression"
        if "bool_not" in kinds:
            return "nullable_boolean_expression"
        if "cast" in kinds:
            return "type_cast"
        return "arithmetic_expression"
    if any(op in {"sort", "limit", "offset"} for op in ops):
        return "ordering_or_limit"
    ok_results = [r for r in normalized.values() if r.status == "ok"]
    if ok_results and len({tuple(r.columns) for r in ok_results}) > 1:
        return "schema_projection"
    if _case_contains_null(case):
        return "null_semantics"
    return "unknown"


def _case_has_sortedness_check(case: Case) -> bool:
    return "sortedness_check" in operation_names(case.program.operations)


def _case_has_random_case_probe(case: Case) -> bool:
    return "random_case_probe" in operation_names(case.program.operations)


def _case_has_group_quantile_probe(case: Case) -> bool:
    return "group_quantile_probe" in operation_names(case.program.operations)


def _case_has_scalar_subquery_probe(case: Case) -> bool:
    return "scalar_subquery_probe" in operation_names(case.program.operations)


def _case_has_window_avg_probe(case: Case) -> bool:
    return "window_avg_probe" in operation_names(case.program.operations)


def _case_has_struct_distinct_probe(case: Case) -> bool:
    return "struct_distinct_probe" in operation_names(case.program.operations)


def _case_has_bit_compare_probe(case: Case) -> bool:
    return "bit_compare_probe" in operation_names(case.program.operations)


def _case_has_round_even_probe(case: Case) -> bool:
    return "round_even_probe" in operation_names(case.program.operations)


def _case_has_series_rtruediv_probe(case: Case) -> bool:
    return "series_rtruediv_probe" in operation_names(case.program.operations)


def _case_has_uint64_isin_probe(case: Case) -> bool:
    return "uint64_isin_probe" in operation_names(case.program.operations)


def _case_has_tuple_anti_null_probe(case: Case) -> bool:
    return "tuple_anti_null_probe" in operation_names(case.program.operations)


def _case_has_json_predicate_order_probe(case: Case) -> bool:
    return "json_predicate_order_probe" in operation_names(case.program.operations)


def _case_has_sparse_mask_probe(case: Case) -> bool:
    return "sparse_mask_probe" in operation_names(case.program.operations)


def _case_has_float_wrap_probe(case: Case) -> bool:
    return "float_wrap_probe" in operation_names(case.program.operations)


def _case_has_index_bool_probe(case: Case) -> bool:
    return "index_bool_probe" in operation_names(case.program.operations)


def _case_has_empty_literal_groupby_probe(case: Case) -> bool:
    return "empty_literal_groupby_probe" in operation_names(case.program.operations)


def _case_has_arrow_string_eq_sum_probe(case: Case) -> bool:
    return "arrow_string_eq_sum_probe" in operation_names(case.program.operations)


def _case_has_arrow_timestamp_loc_slice_probe(case: Case) -> bool:
    return "arrow_timestamp_loc_slice_probe" in operation_names(case.program.operations)


def _case_has_arrow_timestamp_index_attr_probe(case: Case) -> bool:
    return "arrow_timestamp_index_attr_probe" in operation_names(case.program.operations)


def _case_has_eval_inplace_alias_probe(case: Case) -> bool:
    return "eval_inplace_alias_probe" in operation_names(case.program.operations)


def _case_has_dataset_isin_all_match_probe(case: Case) -> bool:
    return "dataset_isin_all_match_probe" in operation_names(case.program.operations)


def _case_has_large_string_partition_probe(case: Case) -> bool:
    return "large_string_partition_probe" in operation_names(case.program.operations)


def _case_has_hash_pivot_wider_probe(case: Case) -> bool:
    return "hash_pivot_wider_probe" in operation_names(case.program.operations)


def _case_has_list_flatten_parent_indices_probe(case: Case) -> bool:
    return "list_flatten_parent_indices_probe" in operation_names(case.program.operations)


def _case_has_rolling_mean_by_null_count_probe(case: Case) -> bool:
    return "rolling_mean_by_null_count_probe" in operation_names(case.program.operations)


def _case_has_reverse_division_columns(case: Case) -> bool:
    return any(
        op_kind == "mutate" and expr_kind(op) == "reverse_division_columns"
        for op_kind, op in zip(operation_names(case.program.operations), case.program.operations)
    )


def _case_has_joined_order_offset_projection(case: Case) -> bool:
    saw_join = False
    saw_order_after_join = False
    for op in case.program.operations:
        kind = op_kind(op)
        if kind == "join":
            saw_join = True
        elif kind == "sort" and saw_join:
            saw_order_after_join = True
        elif kind == "offset" and saw_order_after_join:
            return True
    return False


def _case_has_ordered_topk_projection(case: Case) -> bool:
    saw_order = False
    saw_topk_after_order = False
    for op in case.program.operations:
        kind = op_kind(op)
        if kind == "sort":
            if saw_topk_after_order:
                return True
            saw_order = True
        elif kind in {"limit", "offset"} and saw_order:
            saw_topk_after_order = True
        elif kind == "select" and saw_topk_after_order:
            return True
    return False


def _case_has_negative_zero_comparison(case: Case) -> bool:
    zero_source_columns = _columns_with_float_zero(case)
    negative_zero_columns = set()
    for op in case.program.operations:
        kind = op_kind(op)
        if kind == "mutate":
            if (
                expr_kind(op) == "arith_const"
                and expr_operator(op) == "mul"
                and _is_negative_numeric_value(expr_value(op))
                and expr_source(op) in zero_source_columns
            ):
                negative_zero_columns.add(op_column(op))
        elif kind == "filter" and op_column(op) in negative_zero_columns:
            parsed = parse_filter_comparator(condition_cmp(op) or op_comparator(op))
            if parsed is not None and _filter_comparator_touches_zero(parsed.base, op_value(op)):
                return True
    return False


def _filter_comparator_touches_zero(base: str, value: Any) -> bool:
    if base in {">", ">=", "<", "<=", "==", "!="}:
        return _is_numeric_value(value, 0.0)
    if base in {"in_set", "not_in_set"} and isinstance(value, (list, tuple, frozenset, set)):
        return any(_is_numeric_value(item, 0.0) for item in value)
    if base == "range_closed" and isinstance(value, (list, tuple)) and len(value) == 2:
        lower, upper = value
        if _is_numeric_value(lower, 0.0) or _is_numeric_value(upper, 0.0):
            return True
        if _is_orderable_number(lower) and _is_orderable_number(upper):
            return float(lower) < 0.0 < float(upper)
    return False


def _columns_with_float_zero(case: Case) -> set[str]:
    columns = set()
    for table in case.tables:
        float_columns = {column.name for column in table.columns if column.type == "float"}
        for row in table.rows:
            for column, value in row.items():
                if column in float_columns and _is_numeric_value(value, 0.0):
                    columns.add(column)
    return columns


def _is_negative_numeric_value(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return float(value) < 0.0
    except Exception:
        return False


def _is_numeric_value(value: Any, target: float) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return float(value) == target
    except Exception:
        return False


def _is_orderable_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        number = float(value)
    except Exception:
        return False
    return not math.isnan(number)


def _case_has_tuple_absence_filter(case: Case) -> bool:
    return "tuple_absence_filter" in operation_names(case.program.operations)


def _case_has_grouped_topk_null_sort_key(case: Case) -> bool:
    if not case.tables:
        return False
    table_samples = {
        table.name: {
            column.name: [row.get(column.name) for row in table.rows]
            for column in table.columns
        }
        for table in case.tables
    }
    samples = dict(table_samples.get(case.tables[0].name, {}))
    grouped = False
    ops = case.program.operations
    for idx, op in enumerate(ops):
        kind = op_kind(op)
        if kind == "join":
            samples = join_samples(samples, table_samples.get(op_table(op), {}), op)
        elif kind == "filter":
            samples = filter_samples(samples, op)
        elif kind == "select":
            selected = [column for column in op_columns(op) if column in samples]
            samples = {column: samples[column] for column in selected}
        elif kind == "mutate":
            column = op_column(op)
            values = eval_expr_samples(samples, expr_payload(op))
            if column and values is not None:
                samples[column] = values
        elif kind == "groupby":
            samples = groupby_output_samples(samples, op)
            grouped = True
        elif kind == "sort" and grouped:
            try:
                sort_columns = [key.column for key in normalize_sort_keys(op) if key.column in samples]
            except ValueError:
                sort_columns = []
            if any(any(value is None for value in samples[column]) for column in sort_columns):
                if any(operation_names(ops[idx + 1 :])[j] in {"limit", "offset"} for j in range(len(ops[idx + 1 :]))):
                    return True
    return False


def _case_has_distinct_null_topk(case: Case) -> bool:
    if not case.tables:
        return False
    table_samples = {
        table.name: {
            column.name: [row.get(column.name) for row in table.rows]
            for column in table.columns
        }
        for table in case.tables
    }
    samples = dict(table_samples.get(case.tables[0].name, {}))
    saw_distinct = False
    ops = case.program.operations
    for idx, op in enumerate(ops):
        kind = op_kind(op)
        if kind == "join":
            samples = join_samples(samples, table_samples.get(op_table(op), {}), op)
        elif kind == "filter":
            samples = filter_samples(samples, op)
        elif kind == "select":
            selected = [column for column in op_columns(op) if column in samples]
            samples = {column: samples[column] for column in selected}
        elif kind == "mutate":
            column = op_column(op)
            values = eval_expr_samples(samples, expr_payload(op))
            if column and values is not None:
                samples[column] = values
        elif kind == "distinct":
            samples = distinct_output_samples(samples, op)
            saw_distinct = True
        elif kind == "sort" and saw_distinct:
            try:
                sort_keys = [key for key in normalize_sort_keys(op) if key.column in samples]
            except ValueError:
                sort_keys = []
            nulls_first_keys = [key.column for key in sort_keys if key.nulls == "first"]
            if any(any(value is None for value in samples[column]) for column in nulls_first_keys):
                if any(operation_names(ops[idx + 1 :])[j] in {"limit", "offset"} for j in range(len(ops[idx + 1 :]))):
                    return True
    return False


def _case_has_float_group_key_instability(
    case: Case,
    normalized: dict[str, NormalizedResult],
) -> bool:
    groupby_keys = _final_groupby_keys(case)
    if not groupby_keys:
        return False
    column_types = _column_types_before_groupby(case)
    if not any(column_types.get(key) == "float" for key in groupby_keys):
        return False
    if not _case_uses_arithmetic_float_lineage(case, groupby_keys):
        return False

    ok_results = [result for result in normalized.values() if result.status == "ok"]
    if len(ok_results) < 2:
        return False
    row_counts = {len(result.rows) for result in ok_results}
    if len(row_counts) < 2:
        return False
    return any(_has_duplicate_normalized_rows(result) for result in ok_results)


def _final_groupby_keys(case: Case) -> list[str]:
    for op in reversed(case.program.operations):
        if op_kind(op) == "groupby":
            return groupby_keys(op)
    return []


def _column_types_before_groupby(case: Case) -> dict[str, str]:
    return state_before_first_operation(case, "groupby").column_types


def _case_uses_arithmetic_float_lineage(case: Case, groupby_keys: list[str]) -> bool:
    return case_uses_arithmetic_float_lineage(case, groupby_keys)


def _expr_output_type(expr: dict[str, Any], col_types: dict[str, str]) -> str | None:
    return expr_output_type(expr, col_types)


def _cast_output_type(source_type: str, expr: dict[str, Any]) -> str | None:
    return cast_output_type(source_type, expr)


def _has_duplicate_normalized_rows(result: NormalizedResult) -> bool:
    return result.has_duplicate_rows


def _compare_value(left: Any, comparator: str, right: Any) -> bool:
    try:
        return evaluate_filter_predicate(left, comparator, right)
    except Exception:
        return False


def evaluate_case(case: Case, normalized: dict[str, NormalizedResult]) -> list[Finding]:
    findings: list[Finding] = []
    statuses = {b: r.status for b, r in normalized.items()}
    ok = {b: r for b, r in normalized.items() if r.status == "ok"}
    non_ok = {b: r for b, r in normalized.items() if r.status != "ok"}

    if ok and non_ok:
        sig = _signature(case, normalized, "accept_reject_mismatch")
        kind = "accept_reject_mismatch"
        findings.append(Finding(
            finding_id=f"finding-{sig}",
            kind=kind,
            severity="high",
            suspicious_backends=list(non_ok),
            evidence=f"Some backends accepted while others rejected: {statuses}",
            signature=sig,
            root_cause=classify_root_cause(case, normalized, kind),
            oracle="differential",
            confidence="high",
            mismatch_class="accept_reject",
        ))
        return findings

    if len(non_ok) == len(normalized) and len(set((r.status, r.error_type) for r in non_ok.values())) > 1:
        sig = _signature(case, normalized, "exception_mismatch")
        kind = "exception_mismatch"
        findings.append(Finding(
            finding_id=f"finding-{sig}",
            kind=kind,
            severity="medium",
            suspicious_backends=list(non_ok),
            evidence=f"All backends rejected but with different errors: { {b: r.error_type for b, r in non_ok.items()} }",
            signature=sig,
            root_cause=classify_root_cause(case, normalized, kind),
            oracle="differential",
            confidence="medium",
            mismatch_class="exception_taxonomy",
        ))
        return findings

    if len(ok) >= 2:
        ok_items = list(ok.items())
        ok_backends = [backend for backend, _ in ok_items]
        comparison = compare_row_set_batch(
            [result.rows for _, result in ok_items],
            column_sets=[result.columns for _, result in ok_items],
        )
        if comparison.has_mismatch:
            suspicious = sorted(comparison.suspicious_labels(ok_backends))
            sig = _signature(case, normalized, "semantic_output_mismatch")
            kind = "semantic_output_mismatch"
            shapes = {b: (len(r.rows), len(r.columns)) for b, r in ok.items()}
            mismatch_class = comparison.mismatch_class
            findings.append(Finding(
                finding_id=f"finding-{sig}",
                kind=kind,
                severity="critical",
                suspicious_backends=suspicious,
                evidence=f"Backends returned different canonical tables; mismatch_class={mismatch_class}; shapes={shapes}",
                signature=sig,
                root_cause=classify_root_cause(case, normalized, kind),
                oracle="differential",
                confidence=comparison.default_confidence(),
                mismatch_class=mismatch_class,
            ))
    return findings


def _semantic_mismatch_class(ok_results: dict[str, NormalizedResult]) -> str:
    result_rows = list(ok_results.values())
    if len(result_rows) < 2:
        return "none"
    comparison = compare_row_set_batch(
        [result.rows for result in result_rows],
        column_sets=[result.columns for result in result_rows],
    )
    return comparison.mismatch_class
