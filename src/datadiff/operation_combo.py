from __future__ import annotations

from typing import Any

from datadiff.join_keys import join_key_pairs
from datadiff.operation_semantics import (
    aggregate_functions,
    case_else_value,
    case_then_value,
    condition_cmp,
    condition_value,
    expr_kind,
    expr_operator,
    expr_target_type,
    is_left_join as op_is_left_join,
    is_multi_key_groupby as op_is_multi_key_groupby,
    is_multi_key_join as op_is_multi_key_join,
    is_string_input_cast as op_is_string_input_cast,
    op_comparator,
    op_kind,
    op_partition_columns,
    operation_names,
    sort_has_nulls_first,
)

CANONICAL_OPERATION_ORDER = (
    "join",
    "semi_join",
    "anti_join",
    "union_all",
    "filter",
    "tuple_absence_filter",
    "drop_nulls",
    "fill_null",
    "coalesce",
    "case_when",
    "mutate",
    "row_number_filter",
    "running_sum",
    "random_case_probe",
    "group_quantile_probe",
    "scalar_subquery_probe",
    "window_avg_probe",
    "struct_distinct_probe",
    "bit_compare_probe",
    "round_even_probe",
    "float_literal_precision_probe",
    "timestamp_precision_filter_probe",
    "series_rtruediv_probe",
    "series_reflected_arithmetic_probe",
    "datafusion_grouped_null_topk_probe",
    "confirmed_root_witness_probe",
    "uint64_isin_probe",
    "tuple_anti_null_probe",
    "setop_all_duplicate_probe",
    "json_predicate_order_probe",
    "sparse_mask_probe",
    "float_wrap_probe",
    "index_bool_probe",
    "empty_literal_groupby_probe",
    "arrow_string_eq_sum_probe",
    "arrow_string_contains_na_probe",
    "arrow_timestamp_loc_slice_probe",
    "arrow_timestamp_index_attr_probe",
    "eval_inplace_alias_probe",
    "bool_reduction_skipna_probe",
    "polars_timezone_filter_probe",
    "dataset_isin_all_match_probe",
    "run_end_null_compute_probe",
    "large_string_partition_probe",
    "hash_pivot_wider_probe",
    "list_flatten_parent_indices_probe",
    "rolling_mean_by_null_count_probe",
    "csv_long_numeric_roundtrip_probe",
    "groupby",
    "aggregate",
    "select",
    "distinct",
    "sort",
    "sortedness_check",
    "offset",
    "limit",
)

HIGH_FREQUENCY_COMBOS = {
    "filter_select",
    "filter_select_sort_limit",
    "filter_sort_limit",
    "filter_distinct_sort_limit",
    "filter_fill_null_distinct_sort_limit",
    "filter_fill_null_groupby_sort_limit",
    "fill_null_coalesce_groupby_select_sort_limit",
    "filter_fill_null_groupby_select_sort_limit",
    "coalesce_groupby_sort",
    "coalesce_select_sort_limit",
    "coalesce_distinct_sort_offset_limit",
    "coalesce_mutate_distinct_sort_offset_limit",
    "filter_case_when_select_sort_limit",
    "filter_case_when_groupby_sort_limit",
    "union_all_filter_groupby_sort",
    "union_all_case_when_select_sort_limit",
    "union_all_coalesce_mutate_distinct_sort_offset_limit",
    "drop_nulls_groupby_sort",
    "union_all_drop_nulls_case_when_select_sort_limit",
    "semi_join_filter_select_sort_limit",
    "anti_join_fill_null_groupby_sort",
    "semi_join_groupby_sort",
    "anti_join_groupby_sort",
    "join_semi_join_fill_null_coalesce_groupby_sort",
    "join_anti_join_fill_null_coalesce_sort_limit",
    "join_anti_join_fill_null_coalesce_select_sort_limit",
    "filter_mutate_select",
    "fill_null_groupby_sort",
    "case_when_groupby_sort",
    "case_when_mutate_distinct_sort_limit",
    "filter_mutate_select_sort_limit",
    "groupby_select",
    "groupby_select_sort_limit",
    "distinct_sort_limit",
    "filter_groupby_select",
    "filter_groupby_sort_limit",
    "filter_groupby_select_sort_limit",
    "filter_aggregate",
    "join_filter_aggregate",
    "join_case_when_groupby_sort",
    "join_coalesce_case_when_groupby_sort",
    "join_filter_coalesce_case_when_groupby_sort",
    "join_filter_select",
    "join_fill_null_groupby_sort",
    "semi_join_filter_case_when_groupby_sort",
    "semi_join_filter_case_when_mutate_groupby_sort",
    "anti_join_filter_case_when_groupby_sort",
    "semi_join_filter_mutate_groupby_sort",
    "anti_join_filter_case_when_mutate_groupby_sort",
    "coalesce_case_when_mutate_groupby_distinct_sort",
    "join_fill_null_mutate_groupby_sort",
    "join_filter_mutate_groupby_select_sort_limit",
    "join_mutate_groupby_select_sort_limit",
    "row_number_filter_select_sort",
    "running_sum_select_sort",
    "sort_limit",
}

MEDIUM_FREQUENCY_COMBOS = {
    "filter",
    "filter_mutate",
    "fill_null_distinct_sort_limit",
    "fill_null_filter_distinct_sort_limit",
    "fill_null_coalesce_groupby_sort_limit",
    "coalesce_filter_sort_limit",
    "coalesce_groupby",
    "case_when_filter_sort_limit",
    "union_all_filter",
    "union_all_groupby_sort",
    "drop_nulls_filter_sort_limit",
    "drop_nulls_groupby",
    "semi_join_filter",
    "semi_join_sort_limit",
    "anti_join_filter",
    "anti_join_groupby",
    "filter_mutate_groupby",
    "filter_mutate_groupby_select",
    "filter_aggregate_select_sort_limit",
    "fill_null_mutate_groupby_select_sort_limit",
    "groupby",
    "groupby_sort_limit",
    "join_filter",
    "join_filter_groupby",
    "join_filter_groupby_select_sort_limit",
    "join_mutate_groupby",
    "join_select",
    "join_select_sort_limit",
    "join_sort_limit",
    "filter_offset_select_sort_limit",
    "filter_select_sort_offset_limit",
    "filter_distinct_groupby_sort",
    "mutate_groupby",
    "mutate_groupby_select_sort_limit",
    "join_groupby_aggregate",
}


def describe_operation_combo(operations: list[dict[str, Any]] | list[Any]) -> dict[str, Any]:
    sequence = operation_names(operations, default="unknown")
    op_set = set(sequence)
    template = "_".join(op for op in CANONICAL_OPERATION_ORDER if op in op_set) if sequence else "empty"
    semantic_signals = _semantic_signals(operations, sequence)
    frequency_bucket = _frequency_bucket(template)
    priority = _priority_score(frequency_bucket, semantic_signals, len(sequence))
    return {
        "template": template,
        "sequence": sequence,
        "operation_count": len(sequence),
        "frequency_bucket": frequency_bucket,
        "priority": priority,
        "semantic_signal_candidates": semantic_signals,
        "correctness_risks": semantic_signals,
        "semantic_signals": semantic_signals,
        "has_join": "join" in op_set,
        "has_groupby": "groupby" in op_set,
        "has_aggregate": "aggregate" in op_set,
        "has_union_all": "union_all" in op_set,
        "has_drop_nulls": "drop_nulls" in op_set,
        "has_semi_join": "semi_join" in op_set,
        "has_anti_join": "anti_join" in op_set,
        "has_distinct": "distinct" in op_set,
        "has_fill_null": "fill_null" in op_set,
        "has_coalesce": "coalesce" in op_set,
        "has_case_when": "case_when" in op_set,
        "has_sort_limit": "sort" in op_set and bool({"offset", "limit"} & op_set),
    }

def combo_semantic_signals(combo: dict[str, Any]) -> list[str]:
    values = combo.get("semantic_signal_candidates")
    if isinstance(values, list):
        return [str(value) for value in values]
    values = combo.get("semantic_signals")
    if isinstance(values, list):
        return [str(value) for value in values]
    legacy_values = combo.get("correctness_risks")
    if isinstance(legacy_values, list):
        return [str(value) for value in legacy_values]
    return []


def _frequency_bucket(template: str) -> str:
    if template in HIGH_FREQUENCY_COMBOS:
        return "high"
    if template in MEDIUM_FREQUENCY_COMBOS:
        return "medium"
    return "exploratory"


def _semantic_signals(operations: list[dict[str, Any]] | list[Any], sequence: list[str]) -> list[str]:
    op_set = set(sequence)
    signals = []
    if "join" in op_set:
        signals.append("join_cardinality")
        if any(_is_multi_key_join(op) for op in operations):
            signals.append("multi_key_join")
    if "groupby" in op_set and any(_is_multi_key_groupby(op) for op in operations):
        signals.append("multi_key_groupby")
    if "union_all" in op_set:
        signals.append("union_all_row_append")
        signals.append("union_all_schema_alignment")
    if "semi_join" in op_set:
        signals.append("semi_join_membership")
        signals.append("semi_anti_join_null_keys")
    if "anti_join" in op_set:
        signals.append("anti_join_exclusion")
        signals.append("semi_anti_join_null_keys")
    if "join" in op_set and "filter" in op_set:
        signals.append("join_filter_pushdown")
        if _has_null_predicate_filter(operations):
            signals.append("join_null_predicate_filter")
            if _has_left_join(operations):
                signals.append("left_join_null_predicate_filter")
            if "groupby" in op_set or "aggregate" in op_set:
                signals.append("left_join_null_predicate_aggregation")
    if "join" in op_set and "fill_null" in op_set and ("groupby" in op_set or "aggregate" in op_set):
        signals.append("left_join_fill_null_aggregation")
    if "join" in op_set and "coalesce" in op_set and ("semi_join" in op_set or "anti_join" in op_set):
        signals.append("join_coalesce_membership")
        if _has_left_join(operations):
            signals.append("left_join_coalesce_membership")
        if "groupby" in op_set or "aggregate" in op_set:
            signals.append("join_coalesce_membership_aggregation")
        if "sort" in op_set or "limit" in op_set or "offset" in op_set:
            signals.append("join_coalesce_membership_topk")
    if "join" in op_set and "case_when" in op_set and _has_case_when_membership_condition(operations):
        signals.append("join_case_when_membership")
        if _has_left_join(operations):
            signals.append("left_join_case_when_membership")
        if "groupby" in op_set or "aggregate" in op_set:
            signals.append("join_case_when_membership_aggregation")
        if "sort" in op_set or "limit" in op_set or "offset" in op_set:
            signals.append("join_case_when_membership_topk")
    if "case_when" in op_set and _has_boolean_case_condition(operations):
        signals.append("boolean_case_when_predicate")
        if ("semi_join" in op_set or "anti_join" in op_set) and ("groupby" in op_set or "aggregate" in op_set):
            signals.append("boolean_membership_case_aggregation")
        if "coalesce" in op_set and ("groupby" in op_set or "aggregate" in op_set):
            signals.append("boolean_coalesce_case_aggregation")
            if "join" in op_set and _has_left_join(operations):
                signals.append("left_join_boolean_coalesce_aggregation")
    if "coalesce" in op_set and "filter" in op_set and _has_boolean_filter(operations):
        signals.append("boolean_coalesce_filter")
        if "groupby" in op_set or "aggregate" in op_set:
            signals.append("boolean_coalesce_filter_aggregation")
            if "join" in op_set and _has_left_join(operations):
                signals.append("left_join_boolean_coalesce_filter_aggregation")
    if "case_when" in op_set and _has_boolean_case_when(operations):
        signals.append("boolean_case_when")
        if "groupby" in op_set or "aggregate" in op_set:
            signals.append("boolean_case_when_aggregation")
        if ("semi_join" in op_set or "anti_join" in op_set) and ("groupby" in op_set or "aggregate" in op_set):
            signals.append("boolean_membership_case_aggregation")
        if "join" in op_set and _has_left_join(operations) and ("groupby" in op_set or "aggregate" in op_set):
            signals.append("left_join_boolean_case_aggregation")
    string_join_mutates = {
        expr_kind(op)
        for op in operations
        if op_kind(op) == "mutate"
        and expr_kind(op) in {"string_lower", "string_upper", "string_strip", "string_replace"}
    }
    if "join" in op_set and string_join_mutates:
        signals.append("cleaned_join_key")
    if "join" in op_set and "string_lower" in string_join_mutates:
        signals.append("normalized_string_join_key")
        if {"string_strip", "string_replace"} & string_join_mutates:
            signals.append("chained_string_normalized_join_key")
        if "groupby" in op_set or "aggregate" in op_set:
            signals.append("normalized_string_join_aggregation")
        if "sort" in op_set or "limit" in op_set or "offset" in op_set:
            signals.append("normalized_string_join_topk")
    if ("semi_join" in op_set or "anti_join" in op_set) and string_join_mutates:
        signals.append("normalized_string_membership_key")
        if {"string_strip", "string_replace"} & string_join_mutates:
            signals.append("chained_string_normalized_membership_key")
        if "groupby" in op_set or "aggregate" in op_set:
            signals.append("normalized_string_membership_aggregation")
        if "sort" in op_set or "limit" in op_set or "offset" in op_set:
            signals.append("normalized_string_membership_topk")
    if any(op_kind(op) == "mutate" and expr_kind(op) in {"string_lower", "string_upper"} for op in operations):
        signals.append("string_case_normalization")
        if "groupby" in op_set or "aggregate" in op_set:
            signals.append("string_case_aggregation_keys")
        if "sort" in op_set or "limit" in op_set or "offset" in op_set:
            signals.append("string_case_topk_keys")
    if any(op_kind(op) == "mutate" and expr_kind(op) == "string_length" for op in operations):
        signals.append("string_length")
        signals.append("string_length_null_boundary")
        if "filter" in op_set:
            signals.append("string_length_filter")
        if "groupby" in op_set or "aggregate" in op_set:
            signals.append("string_length_aggregation_keys")
        if "sort" in op_set or "limit" in op_set or "offset" in op_set:
            signals.append("string_length_topk_keys")
    if ("semi_join" in op_set or "anti_join" in op_set) and "filter" in op_set:
        signals.append("semi_anti_join_filter_pushdown")
    if _has_multi_key_membership_join(operations):
        signals.append("multi_key_semi_anti_join")
        if "groupby" in op_set or "aggregate" in op_set:
            signals.append("multi_key_membership_aggregation")
    if "union_all" in op_set and "filter" in op_set:
        signals.append("union_all_filter_pushdown")
    if "union_all" in op_set and ("groupby" in op_set or "aggregate" in op_set):
        signals.append("union_all_aggregation_cardinality")
    if any(func == "nunique" for op in operations for func in aggregate_functions(op)):
        signals.append("distinct_count_aggregation")
        if "join" in op_set:
            signals.append("join_distinct_count_aggregation")
        if "aggregate" in op_set:
            signals.append("global_distinct_count")
    if ("semi_join" in op_set or "anti_join" in op_set) and ("groupby" in op_set or "aggregate" in op_set):
        signals.append("semi_anti_join_aggregation_cardinality")
    if ("semi_join" in op_set or "anti_join" in op_set) and (
        "sort" in op_set or "limit" in op_set or "offset" in op_set
    ):
        signals.append("semi_anti_join_topk_cardinality")
    if "filter" in op_set and "mutate" in op_set:
        signals.append("filter_mutate_dependency")
    cast_targets = {
        expr_target_type(op, "unknown")
        for op in operations
        if op_kind(op) == "mutate" and expr_kind(op) == "cast"
    }
    if cast_targets:
        signals.append("type_cast")
        for target in sorted(cast_targets):
            signals.append(f"type_cast_to_{target}")
        if any(_is_string_input_cast(op) for op in operations):
            signals.append("type_cast_numeric_string")
        if "groupby" in op_set or "aggregate" in op_set:
            signals.append("type_cast_aggregation_keys")
        if "sort" in op_set or "limit" in op_set or "offset" in op_set:
            signals.append("type_cast_topk_keys")
        if ("semi_join" in op_set or "anti_join" in op_set) and ("groupby" in op_set or "aggregate" in op_set):
            signals.append("type_cast_membership_aggregation")
            if any(_is_string_input_cast(op) for op in operations):
                signals.append("numeric_text_cast_membership_aggregation")
    if any(op_kind(op) == "filter" and op_comparator(op) == "not_in_set" for op in operations):
        signals.append("negative_set_membership")
    if any(op_kind(op) == "mutate" and expr_kind(op) == "abs" for op in operations):
        signals.append("numeric_abs_sign")
        if "groupby" in op_set or "aggregate" in op_set:
            signals.append("numeric_abs_aggregation_keys")
        if "sort" in op_set or "limit" in op_set or "offset" in op_set:
            signals.append("numeric_abs_topk_keys")
    if any(op_kind(op) == "mutate" and expr_kind(op) == "clip" for op in operations):
        signals.append("numeric_clip_bounds")
        if "groupby" in op_set or "aggregate" in op_set:
            signals.append("numeric_clip_aggregation_keys")
        if "sort" in op_set or "limit" in op_set or "offset" in op_set:
            signals.append("numeric_clip_topk_keys")
    if any(op_kind(op) == "mutate" and expr_kind(op) == "bool_not" for op in operations):
        signals.append("nullable_boolean_negation")
        if "filter" in op_set:
            signals.append("nullable_boolean_filter")
        if "groupby" in op_set or "aggregate" in op_set:
            signals.append("nullable_boolean_aggregation_keys")
        if "sort" in op_set or "limit" in op_set or "offset" in op_set:
            signals.append("nullable_boolean_topk_keys")
    if any(
        func in {"any", "all"}
        for op in operations
        for func in aggregate_functions(op)
    ):
        signals.append("nullable_boolean_reduction")
        if "filter" in op_set:
            signals.append("nullable_boolean_reduction_filter")
        if "groupby" in op_set or "aggregate" in op_set:
            signals.append("nullable_boolean_reduction_aggregation")
        if "sort" in op_set or "limit" in op_set or "offset" in op_set:
            signals.append("nullable_boolean_reduction_topk")
    if any(
        op_kind(op) == "filter" and op_comparator(op) in {"str_contains", "str_starts_with", "str_ends_with"}
        for op in operations
    ):
        signals.append("string_pattern_filter")
        if "groupby" in op_set or "aggregate" in op_set:
            signals.append("string_pattern_aggregation_cardinality")
        if "sort" in op_set or "limit" in op_set or "offset" in op_set:
            signals.append("string_pattern_topk_cardinality")
    if any(op_kind(op) == "mutate" and expr_kind(op) == "string_strip" for op in operations):
        signals.append("string_strip_whitespace")
        if "groupby" in op_set or "aggregate" in op_set:
            signals.append("string_strip_aggregation_keys")
        if "sort" in op_set or "limit" in op_set or "offset" in op_set:
            signals.append("string_strip_topk_keys")
    if any(op_kind(op) == "mutate" and expr_kind(op) == "string_null_if_empty" for op in operations):
        signals.append("string_null_if_empty")
        signals.append("empty_string_null_boundary")
        if "filter" in op_set:
            signals.append("string_null_if_empty_filter")
        if "groupby" in op_set or "aggregate" in op_set:
            signals.append("string_null_if_empty_aggregation_keys")
        if "sort" in op_set or "limit" in op_set or "offset" in op_set:
            signals.append("string_null_if_empty_topk_keys")
    if any(op_kind(op) == "mutate" and expr_kind(op) == "string_replace" for op in operations):
        signals.append("string_replace_literal")
        if "groupby" in op_set or "aggregate" in op_set:
            signals.append("string_replace_aggregation_keys")
        if "sort" in op_set or "limit" in op_set or "offset" in op_set:
            signals.append("string_replace_topk_keys")
    if any(op_kind(op) == "mutate" and expr_kind(op) == "string_slice" for op in operations):
        signals.append("string_slice_prefix")
        if "groupby" in op_set or "aggregate" in op_set:
            signals.append("string_slice_aggregation_keys")
        if "sort" in op_set or "limit" in op_set or "offset" in op_set:
            signals.append("string_slice_topk_keys")
    if any(op_kind(op) == "mutate" and expr_kind(op) == "string_split_part" for op in operations):
        signals.append("string_split_part_first_token")
        if "groupby" in op_set or "aggregate" in op_set:
            signals.append("string_split_part_aggregation_keys")
        if "sort" in op_set or "limit" in op_set or "offset" in op_set:
            signals.append("string_split_part_topk_keys")
    if any(op_kind(op) == "mutate" and expr_kind(op) == "string_basename" for op in operations):
        signals.append("path_projection")
        signals.append("string_basename_path_separator")
        if "groupby" in op_set or "aggregate" in op_set:
            signals.append("string_basename_aggregation_keys")
        if "sort" in op_set or "limit" in op_set or "offset" in op_set:
            signals.append("string_basename_topk_keys")
    if any(op_kind(op) == "mutate" and expr_kind(op) == "string_concat" for op in operations):
        signals.append("string_concat_null_propagation")
        if "groupby" in op_set or "aggregate" in op_set:
            signals.append("string_concat_aggregation_keys")
        if "sort" in op_set or "limit" in op_set or "offset" in op_set:
            signals.append("string_concat_topk_keys")
    if any(op_kind(op) == "mutate" and expr_kind(op) == "date_part" for op in operations):
        signals.append("date_part_extraction")
        if "filter" in op_set:
            signals.append("date_part_filter")
        if "groupby" in op_set or "aggregate" in op_set:
            signals.append("date_part_aggregation_keys")
        if "sort" in op_set or "limit" in op_set or "offset" in op_set:
            signals.append("date_part_topk_keys")
    string_pattern_expr_kinds = {"string_contains", "string_starts_with", "string_ends_with"}
    if any(
        op_kind(op) == "mutate" and expr_kind(op) in string_pattern_expr_kinds
        for op in operations
    ):
        signals.append("string_pattern_nullable_bool")
        if "filter" in op_set:
            signals.append("string_pattern_bool_filter")
        if "groupby" in op_set or "aggregate" in op_set:
            signals.append("string_pattern_aggregation_keys")
        if "sort" in op_set or "limit" in op_set or "offset" in op_set:
            signals.append("string_pattern_topk_keys")
    if any(op_kind(op) == "mutate" and expr_kind(op) == "string_contains" for op in operations):
        signals.append("string_contains_nullable_bool")
        if "filter" in op_set:
            signals.append("string_contains_bool_filter")
        if "groupby" in op_set or "aggregate" in op_set:
            signals.append("string_contains_aggregation_keys")
        if "sort" in op_set or "limit" in op_set or "offset" in op_set:
            signals.append("string_contains_topk_keys")
    if (
        "case_when" in op_set
        and any(op_kind(op) == "mutate" and expr_kind(op) in {"string_strip", "string_lower"} for op in operations)
        and _has_case_when_membership_condition(operations)
    ):
        signals.append("normalized_string_case_when")
        if "groupby" in op_set or "aggregate" in op_set:
            signals.append("normalized_string_case_when_aggregation")
        if "sort" in op_set or "limit" in op_set or "offset" in op_set:
            signals.append("normalized_string_case_when_topk")
    if "case_when" in op_set and _has_boolean_case_condition(operations) and any(
        op_kind(op) == "mutate" and expr_kind(op) in string_pattern_expr_kinds for op in operations
    ):
        signals.append("string_pattern_case_when")
        if "groupby" in op_set or "aggregate" in op_set:
            signals.append("string_pattern_case_when_aggregation")
        if "sort" in op_set or "limit" in op_set or "offset" in op_set:
            signals.append("string_pattern_case_when_topk")
    if "tuple_absence_filter" in op_set:
        signals.append("tuple_absence_null_filter")
    if "drop_nulls" in op_set:
        signals.append("drop_nulls_null_filter")
    if "drop_nulls" in op_set and ("groupby" in op_set or "aggregate" in op_set):
        signals.append("drop_nulls_aggregation_cardinality")
    if "drop_nulls" in op_set and ("sort" in op_set or "limit" in op_set or "offset" in op_set):
        signals.append("drop_nulls_topk_cardinality")
    if "fill_null" in op_set:
        signals.append("fill_null_null_semantics")
    if "coalesce" in op_set:
        signals.append("coalesce_null_semantics")
        signals.append("coalesce_column_precedence")
    if "case_when" in op_set:
        signals.append("conditional_expression")
        if _has_case_when_membership_condition(operations) or any(
            op_kind(op) == "filter" and op_comparator(op) in {"in_set", "not_in_set"} for op in operations
        ):
            signals.append("case_when_membership_predicate")
            if "groupby" in op_set or "aggregate" in op_set:
                signals.append("case_when_membership_aggregation")
            if "sort" in op_set or "limit" in op_set or "offset" in op_set:
                signals.append("case_when_membership_topk")
    if "distinct" in op_set:
        signals.append("distinct_duplicate_elimination")
        if "sort" in op_set and ("limit" in op_set or "offset" in op_set):
            signals.append("distinct_topk_ordering")
            if _has_distinct_sort_nulls_first_topk(operations):
                signals.append("distinct_null_topk_ordering")
            if _has_sql_distinct_null_topk_pattern(operations):
                signals.append("sql_distinct_null_topk")
            if "coalesce" in op_set:
                signals.append("coalesced_distinct_topk")
            if "union_all" in op_set and "coalesce" in op_set:
                signals.append("union_coalesce_distinct_topk")
    if "fill_null" in op_set and "groupby" in op_set:
        signals.append("fill_null_groupby_keys")
    if "fill_null" in op_set and ("sort" in op_set or "limit" in op_set or "offset" in op_set):
        signals.append("fill_null_topk_keys")
    if "coalesce" in op_set and ("groupby" in op_set or "aggregate" in op_set):
        signals.append("coalesce_aggregation_keys")
    if "coalesce" in op_set and ("sort" in op_set or "limit" in op_set or "offset" in op_set):
        signals.append("coalesce_topk_keys")
    if "case_when" in op_set and "groupby" in op_set:
        signals.append("case_when_groupby_keys")
    if {"coalesce", "case_when", "distinct", "groupby"} <= op_set:
        signals.append("coalesce_case_distinct_aggregation")
    if "row_number_filter" in op_set:
        signals.append("keyed_row_pick")
        if any(op_kind(op) == "row_number_filter" and op_partition_columns(op) for op in operations):
            signals.append("topn_per_group")
    if "row_number_filter" in op_set and any(
        op_kind(op) == "mutate" and expr_kind(op) == "string_basename"
        for op in operations
    ):
        signals.append("path_projection_keyed_pick")
    if "running_sum" in op_set:
        signals.append("running_sum_precision")
    if any(op_kind(op) == "running_sum" and op_partition_columns(op) for op in operations):
        signals.append("partitioned_running_sum")
    if "aggregate" in op_set and "filter" in op_set:
        signals.append("filtered_global_aggregation")
    if "groupby" in op_set and "filter" in op_set and _filter_after_groupby(sequence):
        signals.append("having_filter_after_aggregation")
    if "sortedness_check" in op_set:
        signals.append("sortedness_null_placement")
    if "random_case_probe" in op_set:
        signals.append("simple_case_random_subject")
    if "group_quantile_probe" in op_set:
        signals.append("group_quantile_key_expression")
    if "scalar_subquery_probe" in op_set:
        signals.append("scalar_subquery_double_parentheses")
    if "window_avg_probe" in op_set:
        signals.append("window_avg_rows_frame")
    if "struct_distinct_probe" in op_set:
        signals.append("struct_distinct_unnest")
    if "bit_compare_probe" in op_set:
        signals.append("bit_compare_unequal_length")
    if "round_even_probe" in op_set:
        signals.append("round_even_float_scale")
    if "float_literal_precision_probe" in op_set:
        signals.append("duckdb_float_literal_precision")
    if "timestamp_precision_filter_probe" in op_set:
        signals.append("polars_timestamp_precision_filter")
    if "series_rtruediv_probe" in op_set:
        signals.append("series_rtruediv_operand_order")
    if "series_reflected_arithmetic_probe" in op_set:
        signals.append("polars_reflected_arithmetic_operand_order")
    if "datafusion_grouped_null_topk_probe" in op_set:
        signals.append("grouped_topk_null_sort_key")
    if "confirmed_root_witness_probe" in op_set:
        signals.extend(
            str(operation.get("root_cause", "") or "")
            for operation in operations
            if op_kind(operation) == "confirmed_root_witness_probe"
            and str(operation.get("root_cause", "") or "")
        )
    if "uint64_isin_probe" in op_set:
        signals.append("pandas_uint64_isin_precision")
    if "tuple_anti_null_probe" in op_set:
        signals.append("duckdb_tuple_anti_null_semantics")
    if "setop_all_duplicate_probe" in op_set:
        signals.append("datafusion_setop_all_duplicate_count")
    if "json_predicate_order_probe" in op_set:
        signals.append("duckdb_json_predicate_order_semantics")
    if "sparse_mask_probe" in op_set:
        signals.append("pandas_sparse_array_mask_semantics")
    if "float_wrap_probe" in op_set:
        signals.append("polars_float_wrap_numerical_semantics")
    if "index_bool_probe" in op_set:
        signals.append("pandas_index_bool_result_type")
    if "empty_literal_groupby_probe" in op_set:
        signals.append("polars_empty_literal_groupby_semantics")
    if "arrow_string_eq_sum_probe" in op_set:
        signals.append("pandas_arrow_string_eq_sum_semantics")
    if "arrow_string_contains_na_probe" in op_set:
        signals.append("pandas_arrow_string_contains_na_semantics")
    if "arrow_timestamp_loc_slice_probe" in op_set:
        signals.append("pandas_arrow_timestamp_loc_slice_semantics")
    if "arrow_timestamp_index_attr_probe" in op_set:
        signals.append("pandas_arrow_timestamp_index_attr_semantics")
    if "eval_inplace_alias_probe" in op_set:
        signals.append("pandas_eval_inplace_aliasing_semantics")
    if "bool_reduction_skipna_probe" in op_set:
        signals.append("pandas_bool_reduction_skipna_semantics")
    if "polars_timezone_filter_probe" in op_set:
        signals.append("polars_timezone_filter_semantics")
    if "dataset_isin_all_match_probe" in op_set:
        signals.append("pyarrow_dataset_isin_all_match_semantics")
    if "run_end_null_compute_probe" in op_set:
        signals.append("pyarrow_run_end_null_compute_semantics")
    if "large_string_partition_probe" in op_set:
        signals.append("pyarrow_large_string_partition_schema_semantics")
    if "hash_pivot_wider_probe" in op_set:
        signals.append("pyarrow_hash_pivot_wider_order_semantics")
    if "list_flatten_parent_indices_probe" in op_set:
        signals.append("pyarrow_list_flatten_parent_indices_semantics")
    if "rolling_mean_by_null_count_probe" in op_set:
        signals.append("polars_rolling_mean_by_null_count_semantics")
    if "csv_long_numeric_roundtrip_probe" in op_set:
        signals.append("csv_long_numeric_roundtrip")
    if "groupby" in op_set:
        signals.append("groupby_aggregation")
    if "aggregate" in op_set:
        signals.append("global_aggregation")
    if sequence.count("join") >= 2 and "groupby" in op_set and "aggregate" in op_set:
        signals.append("join_groupby_pipeline")
    if "groupby" in op_set and "sort" in op_set and bool({"offset", "limit"} & op_set):
        signals.append("grouped_topk")
    if "sort" in op_set and bool({"offset", "limit"} & op_set):
        signals.append("topk_ordering")
    if _has_post_topk_filter_sequence(sequence):
        signals.append("topk_filter_pushdown")
    if "select" in op_set and ("sort" in op_set or "limit" in op_set or "offset" in op_set):
        signals.append("projection_ordering")
    return signals


def _has_distinct_sort_nulls_first_topk(operations: list[dict[str, Any]] | list[Any]) -> bool:
    distinct_idx = next((idx for idx, op in enumerate(operations) if op_kind(op) == "distinct"), -1)
    if distinct_idx < 0:
        return False
    for sort_idx in range(distinct_idx + 1, len(operations)):
        sort_op = operations[sort_idx]
        if op_kind(sort_op) != "sort":
            continue
        has_nulls_first = sort_has_nulls_first(sort_op)
        if has_nulls_first and any(op_kind(later) in {"limit", "offset"} for later in operations[sort_idx + 1 :]):
            return True
    return False


def _has_sql_distinct_null_topk_pattern(operations: list[dict[str, Any]] | list[Any]) -> bool:
    if not _has_distinct_sort_nulls_first_topk(operations):
        return False
    null_sensitive_ops = {"fill_null", "coalesce", "drop_nulls"}
    if any(op_kind(op) in null_sensitive_ops for op in operations):
        return True
    null_sensitive_exprs = {"string_null_if_empty", "string_strip", "string_lower", "bool_not", "cast"}
    return any(
        op_kind(op) == "mutate" and expr_kind(op) in null_sensitive_exprs
        for op in operations
    )


def _has_case_when_membership_condition(operations: list[dict[str, Any]] | list[Any]) -> bool:
    return any(
        op_kind(op) == "case_when"
        and condition_cmp(op) in {"in_set", "not_in_set"}
        for op in operations
    )


def _has_boolean_case_when(operations: list[dict[str, Any]] | list[Any]) -> bool:
    return any(
        op_kind(op) == "case_when"
        and isinstance(case_then_value(op), bool)
        and isinstance(case_else_value(op), bool)
        for op in operations
    )


def _has_boolean_case_condition(operations: list[dict[str, Any]] | list[Any]) -> bool:
    return any(
        op_kind(op) == "case_when"
        and condition_cmp(op).startswith("bool_")
        for op in operations
    )


def _has_boolean_filter(operations: list[dict[str, Any]] | list[Any]) -> bool:
    return any(
        op_kind(op) == "filter"
        and (op_comparator(op).startswith("bool_") or isinstance(condition_value(op), bool))
        for op in operations
    )


def _has_multi_key_membership_join(operations: list[dict[str, Any]] | list[Any]) -> bool:
    for op in operations:
        if op_kind(op) not in {"semi_join", "anti_join"}:
            continue
        left_keys, right_keys = join_key_pairs(op)
        if len(left_keys) > 1 and len(left_keys) == len(right_keys):
            return True
    return False


def _has_null_predicate_filter(operations: list[dict[str, Any]] | list[Any]) -> bool:
    return any(
        op_kind(op) == "filter" and op_comparator(op) in {"is_null", "is_not_null"}
        for op in operations
    )


def _has_left_join(operations: list[dict[str, Any]] | list[Any]) -> bool:
    return any(op_is_left_join(op) for op in operations)


def _filter_after_groupby(sequence: list[str]) -> bool:
    try:
        return sequence.index("groupby") < sequence.index("filter")
    except ValueError:
        return False


def _is_multi_key_join(op: dict[str, Any] | Any) -> bool:
    return op_is_multi_key_join(op)


def _is_multi_key_groupby(op: dict[str, Any] | Any) -> bool:
    return op_is_multi_key_groupby(op)


def _is_string_input_cast(op: dict[str, Any] | Any) -> bool:
    return op_is_string_input_cast(op)


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


def _priority_score(frequency_bucket: str, semantic_signals: list[str], operation_count: int) -> float:
    frequency_score = {
        "high": 1.0,
        "medium": 0.65,
        "exploratory": 0.35,
    }[frequency_bucket]
    signal_density_score = min(0.60, 0.12 * len(semantic_signals))
    depth_score = min(0.25, 0.04 * max(0, operation_count - 2))
    return round(frequency_score + signal_density_score + depth_score, 6)


classify_operation_combo = describe_operation_combo
summarize_operation_combo = describe_operation_combo
