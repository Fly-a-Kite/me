from __future__ import annotations

from typing import Any

CANONICAL_OPERATION_ORDER = (
    "join",
    "filter",
    "tuple_absence_filter",
    "mutate",
    "running_sum",
    "random_case_probe",
    "group_quantile_probe",
    "scalar_subquery_probe",
    "window_avg_probe",
    "struct_distinct_probe",
    "bit_compare_probe",
    "round_even_probe",
    "series_rtruediv_probe",
    "groupby",
    "aggregate",
    "select",
    "sort",
    "sortedness_check",
    "offset",
    "limit",
)

HIGH_FREQUENCY_COMBOS = {
    "filter_select",
    "filter_select_sort_limit",
    "filter_sort_limit",
    "filter_mutate_select",
    "filter_mutate_select_sort_limit",
    "groupby_select",
    "groupby_select_sort_limit",
    "filter_groupby_select",
    "filter_groupby_select_sort_limit",
    "join_filter_select",
    "join_filter_mutate_groupby_select_sort_limit",
    "join_mutate_groupby_select_sort_limit",
    "sort_limit",
}

MEDIUM_FREQUENCY_COMBOS = {
    "filter",
    "filter_mutate",
    "filter_mutate_groupby",
    "filter_mutate_groupby_select",
    "groupby",
    "groupby_sort_limit",
    "join_filter",
    "join_filter_groupby",
    "join_filter_groupby_select_sort_limit",
    "join_mutate_groupby",
    "join_select",
    "join_select_sort_limit",
    "join_sort_limit",
    "mutate_groupby",
    "mutate_groupby_select_sort_limit",
    "join_groupby_aggregate",
}


def classify_operation_combo(operations: list[dict[str, Any]]) -> dict[str, Any]:
    sequence = [str(op.get("op", "unknown")) for op in operations]
    op_set = set(sequence)
    template = "_".join(op for op in CANONICAL_OPERATION_ORDER if op in op_set) if sequence else "empty"
    risks = _correctness_risks(sequence)
    frequency_bucket = _frequency_bucket(template)
    priority = _priority_score(frequency_bucket, risks, len(sequence))
    return {
        "template": template,
        "sequence": sequence,
        "operation_count": len(sequence),
        "frequency_bucket": frequency_bucket,
        "priority": priority,
        "correctness_risks": risks,
        "has_join": "join" in op_set,
        "has_groupby": "groupby" in op_set,
        "has_aggregate": "aggregate" in op_set,
        "has_sort_limit": "sort" in op_set and bool({"offset", "limit"} & op_set),
    }


def _frequency_bucket(template: str) -> str:
    if template in HIGH_FREQUENCY_COMBOS:
        return "high"
    if template in MEDIUM_FREQUENCY_COMBOS:
        return "medium"
    return "exploratory"


def _correctness_risks(sequence: list[str]) -> list[str]:
    op_set = set(sequence)
    risks = []
    if "join" in op_set:
        risks.append("join_cardinality")
    if "join" in op_set and "filter" in op_set:
        risks.append("join_filter_pushdown")
    if "filter" in op_set and "mutate" in op_set:
        risks.append("filter_mutate_dependency")
    if "tuple_absence_filter" in op_set:
        risks.append("tuple_absence_null_filter")
    if "running_sum" in op_set:
        risks.append("running_sum_precision")
    if "sortedness_check" in op_set:
        risks.append("sortedness_null_placement")
    if "random_case_probe" in op_set:
        risks.append("simple_case_random_subject")
    if "group_quantile_probe" in op_set:
        risks.append("group_quantile_key_expression")
    if "scalar_subquery_probe" in op_set:
        risks.append("scalar_subquery_double_parentheses")
    if "window_avg_probe" in op_set:
        risks.append("window_avg_rows_frame")
    if "struct_distinct_probe" in op_set:
        risks.append("struct_distinct_unnest")
    if "bit_compare_probe" in op_set:
        risks.append("bit_compare_unequal_length")
    if "round_even_probe" in op_set:
        risks.append("round_even_float_scale")
    if "series_rtruediv_probe" in op_set:
        risks.append("series_rtruediv_operand_order")
    if "groupby" in op_set:
        risks.append("groupby_aggregation")
    if "aggregate" in op_set:
        risks.append("global_aggregation")
    if sequence.count("join") >= 2 and "groupby" in op_set and "aggregate" in op_set:
        risks.append("join_groupby_pipeline")
    if "groupby" in op_set and "sort" in op_set and bool({"offset", "limit"} & op_set):
        risks.append("grouped_topk")
    if "sort" in op_set and bool({"offset", "limit"} & op_set):
        risks.append("topk_ordering")
    if _has_post_topk_filter_sequence(sequence):
        risks.append("topk_filter_pushdown")
    if "select" in op_set and ("sort" in op_set or "limit" in op_set or "offset" in op_set):
        risks.append("projection_ordering")
    return risks


def _has_post_topk_filter_sequence(sequence: list[str]) -> bool:
    for sort_idx, op in enumerate(sequence):
        if op != "sort":
            continue
        for topk_idx in range(sort_idx + 1, len(sequence)):
            if sequence[topk_idx] not in {"limit", "offset"}:
                continue
            if "filter" in sequence[topk_idx + 1:]:
                return True
    return False


def _priority_score(frequency_bucket: str, correctness_risks: list[str], operation_count: int) -> float:
    frequency_score = {
        "high": 1.0,
        "medium": 0.65,
        "exploratory": 0.35,
    }[frequency_bucket]
    risk_score = min(0.60, 0.12 * len(correctness_risks))
    depth_score = min(0.25, 0.04 * max(0, operation_count - 2))
    return round(frequency_score + risk_score + depth_score, 6)
