from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from functools import cmp_to_key
from typing import Any

from datadiff.dsl import Case, normalize_sort_keys
from datadiff.filtering import evaluate_filter_predicate, parse_filter_comparator
from datadiff.operation_combo import classify_operation_combo
from datadiff.reward import online_case_reward
from datadiff.running import sort_rows_for_running, stable_running_sum_values
from datadiff.sortedness import is_sorted_values
from datadiff.tuple_logic import evaluate_tuple_absence
from datadiff.util import unique_preserve_order

TARGET_ALIASES: dict[str, set[str]] = {
    "filter": {"op:filter"},
    "truth_filter": {"filter:truth-test"},
    "groupby": {"op:groupby"},
    "mutate": {"op:mutate"},
    "sort": {"op:sort"},
    "limit": {"op:limit"},
    "offset": {"op:offset"},
    "sort_limit": {"op:sort", "op:limit"},
    "sort_offset": {"op:sort", "op:offset"},
    "nulls": {"has:null"},
    "strings": {"type:str", "has:empty_string", "has:unicode_string", "has:space_string"},
    "numeric": {"type:int", "type:float", "has:negative_number", "has:fractional_float"},
    "edge_float": {"has:special_float", "has:fractional_float"},
    "empty": {"rows:empty", "op:limit_zero"},
    "aggregation": {"op:groupby", "op:aggregate", "agg:sum", "agg:min", "agg:max", "agg:count"},
    "global_aggregation": {"op:aggregate"},
    "null_groupby_topk": {"pattern:null_groupby_topk"},
    "null_agg_topk": {"pattern:null_agg_topk"},
    "filter_null_agg_topk": {"pattern:filter_null_agg_topk"},
    "join_null_agg_topk": {"pattern:join_null_agg_topk"},
    "join_null_key_topk": {"pattern:join_null_key_topk"},
    "wide_offset_topk": {"pattern:wide_offset_topk"},
    "empty_filter_groupby": {"pattern:empty_filter_groupby"},
    "join_filter_groupby": {"pattern:join_filter_groupby"},
    "join_null_truth_filter": {"pattern:join_null_truth_filter"},
    "float_group_key": {"pattern:float_group_key"},
    "join_null_sort": {"pattern:join_null_sort"},
    "ordered_groupby_sort": {"pattern:ordered_groupby_sort"},
    "topk_resort": {"pattern:topk_resort"},
    "join_ordered_agg_topk": {"pattern:join_ordered_agg_topk"},
    "global_null_aggregate": {"pattern:global_null_aggregate"},
    "string_count_groupby": {"pattern:string_count_groupby"},
    "unique_count_groupby": {"pattern:unique_count_groupby"},
    "unique_count": {"agg:nunique"},
    "set_membership_filter": {"pattern:set_membership_filter"},
    "set_membership": {"filter:set-membership"},
    "null_predicate_filter": {"pattern:null_predicate_filter"},
    "null_predicate": {"filter:null-predicate"},
    "boolean_predicate_filter": {"pattern:boolean_predicate_filter"},
    "boolean_predicate": {"filter:boolean-predicate"},
    "post_topk_range_filter": {"pattern:post_topk_range_filter"},
    "range_filter": {"filter:range-closed"},
    "tuple_absence_filter": {"pattern:tuple_absence_filter"},
    "tuple_absence": {"filter:tuple-absence"},
    "running_sum_precision": {"pattern:running_sum_precision"},
    "running_sum": {"op:running_sum"},
    "sortedness_null_placement": {"pattern:sortedness_null_placement"},
    "sortedness": {"op:sortedness_check"},
    "simple_case_random_subject": {"pattern:simple_case_random_subject"},
    "random_case_probe": {"op:random_case_probe"},
    "case_expression": {"op:random_case_probe"},
    "group_quantile_key_probe": {"pattern:group_quantile_key_probe"},
    "group_quantile_probe": {"op:group_quantile_probe"},
    "dynamic_quantile": {"quantile:dynamic-key"},
    "scalar_subquery_double_parentheses": {"pattern:scalar_subquery_double_parentheses"},
    "scalar_subquery_probe": {"op:scalar_subquery_probe"},
    "correlated_subquery": {"subquery:correlated-scalar"},
    "window_avg_rows_frame": {"pattern:window_avg_rows_frame"},
    "window_avg_probe": {"op:window_avg_probe"},
    "window_frame": {"window:rows-frame"},
    "struct_distinct_unnest": {"pattern:struct_distinct_unnest"},
    "struct_distinct_probe": {"op:struct_distinct_probe"},
    "struct_unnest": {"struct:unnest"},
    "bit_compare_unequal_length": {"pattern:bit_compare_unequal_length"},
    "bit_compare_probe": {"op:bit_compare_probe"},
    "bit_ordering": {"bit:unequal-length", "comparison:bit-order"},
    "round_even_float_scale": {"pattern:round_even_float_scale"},
    "round_even_probe": {"op:round_even_probe"},
    "rounding": {"numeric:round-even", "float:decimal-scale"},
    "series_rtruediv_operand_order": {"pattern:series_rtruediv_operand_order"},
    "series_rtruediv_probe": {"op:series_rtruediv_probe"},
    "reverse_division": {"series:reverse-division", "arithmetic:operand-order"},
    "pandas_uint64_isin_precision": {"pattern:pandas_uint64_isin_precision"},
    "uint64_isin_probe": {"op:uint64_isin_probe"},
    "unsigned_membership": {"pandas:uint64-isin", "membership:unsigned-precision"},
    "duckdb_tuple_anti_null_semantics": {"pattern:duckdb_tuple_anti_null_semantics"},
    "tuple_anti_null_probe": {"op:tuple_anti_null_probe"},
    "tuple_null_membership": {"duckdb:tuple-anti-null", "nulls:ternary-membership"},
    "pandas_sparse_array_mask_semantics": {"pattern:pandas_sparse_array_mask_semantics"},
    "sparse_mask_probe": {"op:sparse_mask_probe"},
    "sparse_masking": {"pandas:sparse-mask", "mask:sparse-array"},
    "polars_float_wrap_numerical_semantics": {"pattern:polars_float_wrap_numerical_semantics"},
    "float_wrap_probe": {"op:float_wrap_probe"},
    "wrap_numerical": {"polars:wrap-numerical", "cast:float-overflow"},
    "pandas_index_bool_result_type": {"pattern:pandas_index_bool_result_type"},
    "index_bool_probe": {"op:index_bool_probe"},
    "index_boolean_result": {"pandas:index-bool", "api:result-type"},
    "polars_empty_literal_groupby_semantics": {"pattern:polars_empty_literal_groupby_semantics"},
    "empty_literal_groupby_probe": {"op:empty_literal_groupby_probe"},
    "literal_empty_groupby": {"polars:empty-literal-groupby", "groupby:empty-literal"},
    "pandas_arrow_string_eq_sum_semantics": {"pattern:pandas_arrow_string_eq_sum_semantics"},
    "arrow_string_eq_sum_probe": {"op:arrow_string_eq_sum_probe"},
    "arrow_string_reduction": {"pandas:arrow-string-eq-sum", "arrow:string-bool-reduction"},
    "pandas_arrow_timestamp_loc_slice_semantics": {"pattern:pandas_arrow_timestamp_loc_slice_semantics"},
    "arrow_timestamp_loc_slice_probe": {"op:arrow_timestamp_loc_slice_probe"},
    "arrow_timestamp_indexing": {"pandas:arrow-timestamp-loc-slice", "arrow:timestamp-index-slice"},
    "pandas_arrow_timestamp_index_attr_semantics": {"pattern:pandas_arrow_timestamp_index_attr_semantics"},
    "arrow_timestamp_index_attr_probe": {"op:arrow_timestamp_index_attr_probe"},
    "arrow_timestamp_attributes": {"pandas:arrow-timestamp-index-attr", "arrow:timestamp-index-attribute"},
    "join": {"op:join", "tables:multi"},
    "common_workflow": {"combo_frequency:high"},
    "operation_combo": {"combo_frequency:high", "combo_frequency:medium"},
    "topk": {"combo_risk:topk_ordering", "combo_risk:grouped_topk", "combo_risk:topk_filter_pushdown"},
    "expressions": {"expr:add_const", "expr:arith_const", "expr:string_length", "expr:string_lower", "expr:cast"},
    "casts": {"expr:cast"},
}

PATTERN_TARGET_WEIGHT = 8.0
GENERIC_COMPANION_TARGET_WEIGHT = 0.25
TEMPLATE_TARGET_BONUS = 3.0


def parse_guidance_targets(value: str | list[str] | tuple[str, ...] | None) -> list[str]:
    if value is None:
        return []
    raw = value if isinstance(value, (list, tuple)) else value.split(",")
    targets = []
    for item in raw:
        text = str(item).strip()
        if text:
            targets.append(text)
    return targets


def extract_case_features(case: Case) -> set[str]:
    features: set[str] = set()
    table = case.tables[0]
    features.add("tables:multi" if len(case.tables) > 1 else "tables:single")
    generator_profile = str(case.metadata.get("generator_profile", "")).strip()
    if generator_profile:
        features.add(f"generator_profile:{generator_profile}")
    mixed_generator_profile = str(case.metadata.get("mixed_generator_profile", "")).strip()
    if mixed_generator_profile:
        features.add(f"mixed_generator_profile:{mixed_generator_profile}")
    row_count = len(table.rows)
    col_count = len(table.columns)
    features.add(_bucket("rows", row_count, [(0, "empty"), (3, "tiny"), (10, "small"), (20, "medium")], "large"))
    features.add(_bucket("cols", col_count, [(3, "narrow"), (5, "medium")], "wide"))
    for column in table.columns:
        features.add(f"type:{column.type}")
        features.add(f"nullable:{column.type}:{column.nullable}")

    for row in table.rows:
        for value in row.values():
            if value is None:
                features.add("has:null")
            elif isinstance(value, bool):
                features.add(f"bool:{str(value).lower()}")
            elif isinstance(value, int):
                if value < 0:
                    features.add("has:negative_number")
                elif value == 0:
                    features.add("has:zero")
            elif isinstance(value, float):
                if math.isnan(value) or math.isinf(value):
                    features.add("has:special_float")
                elif not value.is_integer():
                    features.add("has:fractional_float")
                if value < 0:
                    features.add("has:negative_number")
                elif value == 0:
                    features.add("has:zero")
            elif isinstance(value, str):
                if value == "":
                    features.add("has:empty_string")
                if any(ord(ch) > 127 for ch in value):
                    features.add("has:unicode_string")
                if " " in value:
                    features.add("has:space_string")

    op_names = []
    available_types = {column.name: column.type for column in table.columns}
    has_string_count_groupby = False
    has_unique_count_groupby = False
    has_set_membership_filter = False
    has_null_predicate_filter = False
    has_boolean_predicate_filter = False
    has_range_filter = False
    has_tuple_absence_filter = False
    has_running_sum_precision = False
    has_sortedness_check = False
    has_random_case_probe = False
    has_group_quantile_probe = False
    has_scalar_subquery_probe = False
    has_window_avg_probe = False
    has_struct_distinct_probe = False
    has_bit_compare_probe = False
    has_round_even_probe = False
    has_series_rtruediv_probe = False
    has_uint64_isin_probe = False
    has_tuple_anti_null_probe = False
    has_sparse_mask_probe = False
    has_float_wrap_probe = False
    has_index_bool_probe = False
    has_empty_literal_groupby_probe = False
    has_arrow_string_eq_sum_probe = False
    has_arrow_timestamp_loc_slice_probe = False
    has_arrow_timestamp_index_attr_probe = False
    for op in case.program.operations:
        kind = str(op.get("op", "unknown"))
        op_names.append(kind)
        features.add(f"op:{kind}")
        if kind == "filter":
            cmp = str(op.get("cmp", "unknown"))
            column = str(op.get("column", "unknown"))
            features.add(f"cmp:{cmp}")
            features.add(f"filter_type:{available_types.get(column, 'derived')}")
            parsed = parse_filter_comparator(cmp)
            if parsed is not None and parsed.base == "in_set":
                features.add("filter:set-membership")
                has_set_membership_filter = True
            if parsed is not None and parsed.base in {"is_null", "is_not_null"}:
                features.add("filter:null-predicate")
                features.add(f"filter:null-predicate:{parsed.base}")
                has_null_predicate_filter = True
            if parsed is not None and parsed.base == "bool_predicate":
                features.add("filter:boolean-predicate")
                features.add(f"filter:boolean-predicate:{parsed.truth_test}")
                has_boolean_predicate_filter = True
            if parsed is not None and parsed.base == "range_closed":
                features.add("filter:range-closed")
                has_range_filter = True
            if parsed is not None and parsed.truth_test is not None:
                features.add("filter:truth-test")
                features.add(f"filter:truth:{parsed.truth_test}")
        elif kind == "tuple_absence_filter":
            features.add("filter:tuple-absence")
            has_tuple_absence_filter = True
        elif kind == "running_sum":
            source = str(op.get("source", ""))
            input_dtype = str(op.get("input_dtype", "float64"))
            features.add(f"running:{input_dtype}")
            features.add(f"running_source_type:{available_types.get(source, 'derived')}")
            if input_dtype == "float32":
                has_running_sum_precision = True
            available_types[str(op.get("column", "derived"))] = "float"
        elif kind == "sortedness_check":
            column = str(op.get("column", ""))
            nulls = str(op.get("nulls", "last"))
            features.add(f"sortedness:nulls:{nulls}")
            features.add(f"sortedness:{'asc' if op.get('ascending', True) else 'desc'}")
            features.add(f"sortedness_source_type:{available_types.get(column, 'derived')}")
            available_types = {str(op.get("as", "derived")): "bool"}
            has_sortedness_check = True
        elif kind == "random_case_probe":
            features.add("case_expr:simple")
            features.add("case_expr:random-subject")
            features.add(_bucket("case_probe_rows", int(op.get("rows", 0)), [(1000, "small"), (10000, "medium")], "large"))
            available_types = {str(op.get("as", "derived")): "bool"}
            has_random_case_probe = True
        elif kind == "group_quantile_probe":
            features.add("quantile:dynamic-key")
            features.add(_bucket("quantile_probe_values", len(op.get("values", [])), [(2, "tiny"), (4, "small")], "medium"))
            available_types = {str(op.get("as", "derived")): "bool"}
            has_group_quantile_probe = True
        elif kind == "scalar_subquery_probe":
            features.add("subquery:correlated-scalar")
            features.add("subquery:nested-aggregate")
            available_types = {str(op.get("as", "derived")): "bool"}
            has_scalar_subquery_probe = True
        elif kind == "window_avg_probe":
            features.add("window:rows-frame")
            features.add("window:avg")
            available_types = {str(op.get("as", "derived")): "bool"}
            has_window_avg_probe = True
        elif kind == "struct_distinct_probe":
            features.add("struct:unnest")
            features.add("struct:distinct")
            available_types = {str(op.get("as", "derived")): "bool"}
            has_struct_distinct_probe = True
        elif kind == "bit_compare_probe":
            features.add("bit:unequal-length")
            features.add("comparison:bit-order")
            available_types = {str(op.get("as", "derived")): "bool"}
            has_bit_compare_probe = True
        elif kind == "round_even_probe":
            features.add("numeric:round-even")
            features.add("float:decimal-scale")
            available_types = {str(op.get("as", "derived")): "bool"}
            has_round_even_probe = True
        elif kind == "series_rtruediv_probe":
            features.add("series:reverse-division")
            features.add("arithmetic:operand-order")
            available_types = {str(op.get("as", "derived")): "bool"}
            has_series_rtruediv_probe = True
        elif kind == "uint64_isin_probe":
            features.add("pandas:uint64-isin")
            features.add("membership:unsigned-precision")
            available_types = {str(op.get("as", "derived")): "bool"}
            has_uint64_isin_probe = True
        elif kind == "tuple_anti_null_probe":
            features.add("duckdb:tuple-anti-null")
            features.add("nulls:ternary-membership")
            available_types = {str(op.get("as", "derived")): "bool"}
            has_tuple_anti_null_probe = True
        elif kind == "sparse_mask_probe":
            features.add("pandas:sparse-mask")
            features.add("mask:sparse-array")
            available_types = {str(op.get("as", "derived")): "bool"}
            has_sparse_mask_probe = True
        elif kind == "float_wrap_probe":
            features.add("polars:wrap-numerical")
            features.add("cast:float-overflow")
            available_types = {str(op.get("as", "derived")): "bool"}
            has_float_wrap_probe = True
        elif kind == "index_bool_probe":
            features.add("pandas:index-bool")
            features.add("api:result-type")
            available_types = {str(op.get("as", "derived")): "bool"}
            has_index_bool_probe = True
        elif kind == "empty_literal_groupby_probe":
            features.add("polars:empty-literal-groupby")
            features.add("groupby:empty-literal")
            available_types = {str(op.get("as", "derived")): "bool"}
            has_empty_literal_groupby_probe = True
        elif kind == "arrow_string_eq_sum_probe":
            features.add("pandas:arrow-string-eq-sum")
            features.add("arrow:string-bool-reduction")
            available_types = {str(op.get("as", "derived")): "bool"}
            has_arrow_string_eq_sum_probe = True
        elif kind == "arrow_timestamp_loc_slice_probe":
            features.add("pandas:arrow-timestamp-loc-slice")
            features.add("arrow:timestamp-index-slice")
            available_types = {str(op.get("as", "derived")): "bool"}
            has_arrow_timestamp_loc_slice_probe = True
        elif kind == "arrow_timestamp_index_attr_probe":
            features.add("pandas:arrow-timestamp-index-attr")
            features.add("arrow:timestamp-index-attribute")
            available_types = {str(op.get("as", "derived")): "bool"}
            has_arrow_timestamp_index_attr_probe = True
        elif kind == "select":
            width = len(op.get("columns", []))
            features.add(_bucket("select_width", width, [(1, "one"), (3, "few")], "many"))
        elif kind == "sort":
            try:
                sort_keys = normalize_sort_keys(op)
            except ValueError:
                sort_keys = []
            directions = {key.ascending for key in sort_keys}
            if len(directions) > 1:
                features.add("sort:mixed")
            else:
                features.add(f"sort:{'asc' if (not sort_keys or sort_keys[0].ascending) else 'desc'}")
        elif kind == "join":
            features.add(f"join:{op.get('how', 'unknown')}")
            features.add(f"join_table:{op.get('table', 'unknown')}")
        elif kind == "limit":
            limit = int(op.get("n", 0))
            if limit == 0:
                features.add("op:limit_zero")
            features.add(_bucket("limit", limit, [(0, "zero"), (3, "tiny"), (10, "small")], "large"))
        elif kind == "offset":
            offset = int(op.get("n", 0))
            if offset == 0:
                features.add("op:offset_zero")
            features.add(_bucket("offset", offset, [(0, "zero"), (3, "tiny"), (10, "small")], "large"))
        elif kind == "mutate":
            expr = op.get("expr", {})
            expr_kind = expr.get("kind", "unknown")
            features.add(f"mutate:{expr_kind}")
            features.add(f"expr:{expr_kind}")
            if expr_kind == "arith_const":
                features.add(f"arith:{expr.get('op', 'unknown')}")
            if expr_kind == "cast":
                features.add(f"cast_to:{expr.get('to', 'unknown')}")
        elif kind == "groupby":
            for key in op.get("keys", []):
                features.add(f"group_key_type:{available_types.get(key, 'derived')}")
            for agg in op.get("aggs", []):
                source_type = available_types.get(str(agg.get("column", "")), "derived")
                features.add(f"agg:{agg.get('func', 'unknown')}")
                features.add(f"agg_source_type:{source_type}")
                if agg.get("func") == "count" and source_type == "str":
                    features.add("agg:count:str")
                    has_string_count_groupby = True
                if agg.get("func") == "nunique":
                    features.add(f"agg:nunique:{source_type}")
                    has_unique_count_groupby = True
                available_types[str(agg.get("as", "derived"))] = (
                    "int" if agg.get("func") in {"count", "nunique"} else "float"
                )
        elif kind == "aggregate":
            for agg in op.get("aggs", []):
                source_type = available_types.get(str(agg.get("column", "")), "derived")
                features.add(f"agg:{agg.get('func', 'unknown')}")
                features.add(f"agg_source_type:{source_type}")
                if agg.get("func") == "count" and source_type == "str":
                    features.add("agg:count:str")
                if agg.get("func") == "nunique":
                    features.add(f"agg:nunique:{source_type}")
                available_types[str(agg.get("as", "derived"))] = (
                    "int" if agg.get("func") in {"count", "nunique"} else "float"
                )
    if op_names:
        features.add("opseq:" + ">".join(op_names))
        features.add(_bucket("op_count", len(op_names), [(1, "one"), (3, "few"), (5, "many")], "deep"))
    combo = classify_operation_combo(case.program.operations)
    features.add(f"combo:{combo['template']}")
    features.add(f"combo_frequency:{combo['frequency_bucket']}")
    for risk in combo["correctness_risks"]:
        features.add(f"combo_risk:{risk}")
    _, frontier_buckets = _frontier_signature(case)
    features.update(frontier_buckets)
    if _has_null_groupby_topk_pattern(op_names, frontier_buckets):
        features.add("pattern:null_groupby_topk")
    if _has_null_agg_topk_pattern(case.program.operations, frontier_buckets):
        features.add("pattern:null_agg_topk")
    if _has_filter_null_agg_topk_pattern(case.program.operations, frontier_buckets):
        features.add("pattern:filter_null_agg_topk")
    if _has_join_null_agg_topk_pattern(case.program.operations, frontier_buckets):
        features.add("pattern:join_null_agg_topk")
    if _has_join_null_key_topk_pattern(case.program.operations, frontier_buckets):
        features.add("pattern:join_null_key_topk")
    if _has_wide_offset_topk_pattern(case.program.operations):
        features.add("pattern:wide_offset_topk")
    if _has_empty_filter_groupby_pattern(case.program.operations, frontier_buckets):
        features.add("pattern:empty_filter_groupby")
    if _has_join_filter_groupby_pattern(case.program.operations):
        features.add("pattern:join_filter_groupby")
    if _has_join_null_truth_filter_pattern(case.program.operations):
        features.add("pattern:join_null_truth_filter")
    if _has_float_group_key_pattern(case.program.operations):
        features.add("pattern:float_group_key")
    if _has_join_null_sort_pattern(case.program.operations, frontier_buckets):
        features.add("pattern:join_null_sort")
    if _has_ordered_groupby_sort_pattern(case.program.operations):
        features.add("pattern:ordered_groupby_sort")
    if _has_topk_resort_pattern(case.program.operations):
        features.add("pattern:topk_resort")
    if _has_join_ordered_agg_topk_pattern(case.program.operations):
        features.add("pattern:join_ordered_agg_topk")
    if _has_global_null_aggregate_pattern(case.program.operations, frontier_buckets):
        features.add("pattern:global_null_aggregate")
    if has_string_count_groupby:
        features.add("pattern:string_count_groupby")
    if has_unique_count_groupby:
        features.add("pattern:unique_count_groupby")
    if has_set_membership_filter:
        features.add("pattern:set_membership_filter")
    if has_null_predicate_filter:
        features.add("pattern:null_predicate_filter")
    if has_boolean_predicate_filter:
        features.add("pattern:boolean_predicate_filter")
    if has_range_filter:
        features.add("pattern:range_filter")
    if has_range_filter and _has_post_topk_filter_pattern(case.program.operations):
        features.add("pattern:post_topk_range_filter")
    if has_tuple_absence_filter:
        features.add("pattern:tuple_absence_filter")
    if has_running_sum_precision or _has_running_sum_precision_pattern(case.program.operations):
        features.add("pattern:running_sum_precision")
    if has_sortedness_check and _has_sortedness_null_placement_pattern(case.program.operations):
        features.add("pattern:sortedness_null_placement")
    if has_random_case_probe:
        features.add("pattern:simple_case_random_subject")
    if has_group_quantile_probe:
        features.add("pattern:group_quantile_key_probe")
    if has_scalar_subquery_probe:
        features.add("pattern:scalar_subquery_double_parentheses")
    if has_window_avg_probe:
        features.add("pattern:window_avg_rows_frame")
    if has_struct_distinct_probe:
        features.add("pattern:struct_distinct_unnest")
    if has_bit_compare_probe:
        features.add("pattern:bit_compare_unequal_length")
    if has_round_even_probe:
        features.add("pattern:round_even_float_scale")
    if has_series_rtruediv_probe:
        features.add("pattern:series_rtruediv_operand_order")
    if has_uint64_isin_probe:
        features.add("pattern:pandas_uint64_isin_precision")
    if has_tuple_anti_null_probe:
        features.add("pattern:duckdb_tuple_anti_null_semantics")
    if has_sparse_mask_probe:
        features.add("pattern:pandas_sparse_array_mask_semantics")
    if has_float_wrap_probe:
        features.add("pattern:polars_float_wrap_numerical_semantics")
    if has_index_bool_probe:
        features.add("pattern:pandas_index_bool_result_type")
    if has_empty_literal_groupby_probe:
        features.add("pattern:polars_empty_literal_groupby_semantics")
    if has_arrow_string_eq_sum_probe:
        features.add("pattern:pandas_arrow_string_eq_sum_semantics")
    if has_arrow_timestamp_loc_slice_probe:
        features.add("pattern:pandas_arrow_timestamp_loc_slice_semantics")
    if has_arrow_timestamp_index_attr_probe:
        features.add("pattern:pandas_arrow_timestamp_index_attr_semantics")
    return features


@dataclass(slots=True)
class GuidanceDecision:
    case: Case
    score: float
    features: list[str]
    matched_targets: list[str]
    candidate_count: int
    contributing_candidate_count: int = 1
    pruned_candidate_count: int = 0
    frontier_buckets: list[str] = field(default_factory=list)
    score_breakdown: dict[str, float] = field(default_factory=dict)
    online_weights: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": round(self.score, 6),
            "features": self.features,
            "matched_targets": self.matched_targets,
            "candidate_count": self.candidate_count,
            "contributing_candidate_count": self.contributing_candidate_count,
            "pruned_candidate_count": self.pruned_candidate_count,
            "frontier_buckets": self.frontier_buckets,
            "score_breakdown": {
                key: round(value, 6) for key, value in sorted(self.score_breakdown.items())
            },
            "online_weights": self.online_weights,
        }


@dataclass(slots=True)
class FeatureRewardStats:
    pulls: int = 0
    total_reward: float = 0.0

    @property
    def mean_reward(self) -> float:
        return self.total_reward / self.pulls if self.pulls else 0.0


@dataclass(slots=True)
class OnlineFeatureWeights:
    exploration_weight: float = 0.12
    min_multiplier: float = 0.35
    max_multiplier: float = 2.50
    feature_stats: dict[str, FeatureRewardStats] = field(default_factory=dict)
    prefix_stats: dict[str, FeatureRewardStats] = field(default_factory=dict)
    total_updates: int = 0

    def record(self, features: set[str], reward: float) -> None:
        learnable = [feature for feature in features if _is_learnable_weight_feature(feature)]
        if not learnable:
            return
        self.total_updates += 1
        for feature in learnable:
            self._record_stat(self.feature_stats, feature, reward)
            self._record_stat(self.prefix_stats, _feature_prefix(feature), reward)

    def multiplier(self, feature: str) -> float:
        exact = self.feature_stats.get(feature)
        prefix = self.prefix_stats.get(_feature_prefix(feature))
        if exact is None and prefix is None:
            return 1.0
        exact_signal = exact.mean_reward if exact is not None else 0.0
        prefix_signal = prefix.mean_reward if prefix is not None else 0.0
        pulls = exact.pulls if exact is not None else 0
        exploration = self.exploration_weight * math.sqrt(
            math.log(self.total_updates + 2.0) / max(1, pulls)
        )
        value = 1.0 + (0.65 * exact_signal) + (0.35 * prefix_signal) + exploration
        return min(self.max_multiplier, max(self.min_multiplier, value))

    def snapshot(self, limit: int = 12) -> list[dict[str, Any]]:
        rows = [
            {
                "feature": feature,
                "pulls": stats.pulls,
                "mean_reward": stats.mean_reward,
                "multiplier": self.multiplier(feature),
            }
            for feature, stats in self.feature_stats.items()
        ]
        return sorted(rows, key=lambda row: (row["multiplier"], row["pulls"]), reverse=True)[:limit]

    @staticmethod
    def _record_stat(stats: dict[str, FeatureRewardStats], key: str, reward: float) -> None:
        stat = stats.setdefault(key, FeatureRewardStats())
        stat.pulls += 1
        stat.total_reward += reward


@dataclass(slots=True)
class GuidanceState:
    targets: list[str] = field(default_factory=list)
    feature_counts: Counter[str] = field(default_factory=Counter)
    finding_feature_counts: Counter[str] = field(default_factory=Counter)
    root_cause_counts: Counter[str] = field(default_factory=Counter)
    frontier_bucket_counts: Counter[str] = field(default_factory=Counter)
    online_weights: OnlineFeatureWeights = field(default_factory=OnlineFeatureWeights)

    def choose_case(self, candidates: list[Case]) -> GuidanceDecision:
        if not candidates:
            raise ValueError("guided candidate pool cannot be empty")
        scored = [self._score_case(case, len(candidates)) for case in candidates]
        contributing = [decision for decision in scored if self._is_contributing_candidate(decision)]
        if not contributing:
            contributing = [max(scored, key=lambda decision: (decision.score, -decision.case.seed))]
        pruned = len(scored) - len(contributing)
        for decision in contributing:
            decision.contributing_candidate_count = len(contributing)
            decision.pruned_candidate_count = pruned
        targeted = [decision for decision in contributing if decision.matched_targets]
        if targeted:
            return max(
                targeted,
                key=lambda decision: (
                    decision.score_breakdown.get("specific_target_matches", 0.0),
                    decision.score_breakdown.get("target_priority", float(len(decision.matched_targets))),
                    decision.score,
                    len(decision.matched_targets),
                    -decision.case.seed,
                ),
            )
        return max(contributing, key=lambda decision: (decision.score, -decision.case.seed))

    def record_result(self, case: Case, row: dict[str, Any]) -> None:
        features = extract_case_features(case)
        self.feature_counts.update(features)
        _, frontier_buckets = _frontier_signature(case)
        self.frontier_bucket_counts.update(frontier_buckets)
        self.online_weights.record(features, _guidance_reward(row))
        findings = row.get("findings") or []
        if findings:
            self.finding_feature_counts.update(features)
            for finding in findings:
                root = str(finding.get("root_cause", "unknown"))
                self.root_cause_counts[root] += 1

    def _score_case(self, case: Case, candidate_count: int) -> GuidanceDecision:
        features = extract_case_features(case)
        matched_targets = _matched_targets(features, self.targets)
        target_priority = _target_match_priority(features, matched_targets, self.feature_counts)
        target_template_matches = _target_template_match_count(features, matched_targets)
        target_template_bonus = _target_template_bonus(features, matched_targets, self.feature_counts)
        specific_target_matches = _specific_target_count(matched_targets)
        path_coverage_proxy = _path_coverage_proxy(features, self.feature_counts, self.online_weights)
        data_sensitivity = _data_sensitivity_score(features, self.feature_counts, self.online_weights)
        frontier_conformance, frontier_buckets = _frontier_conformance(case, self.frontier_bucket_counts)
        target_bonus = 3.0 * target_priority
        finding_yield_bonus = (
            sum(
                _bounded_finding_signal(self.finding_feature_counts[f])
                * _finding_feature_weight(f, self.online_weights)
                for f in features
            )
            / math.sqrt(max(1, len(features)))
            * 0.50
        )
        feature_saturation_penalty = (
            sum(
                _feature_saturation(self.finding_feature_counts[f])
                * _saturation_feature_weight(f, self.online_weights)
                for f in features
            )
            / math.sqrt(max(1, len(features)))
            * 0.08
        )
        root_saturation_penalty = sum(
            _root_saturation(self.root_cause_counts[root]) for root in _predicted_roots(features)
        )
        if matched_targets:
            feature_saturation_penalty *= 0.35
            root_saturation_penalty *= 0.35
        contribution_potential = _contribution_potential(
            features,
            frontier_buckets,
            matched_targets,
            frontier_conformance,
            feature_counts=self.feature_counts,
            frontier_bucket_counts=self.frontier_bucket_counts,
            root_cause_counts=self.root_cause_counts,
        )
        combo_priority = classify_operation_combo(case.program.operations)["priority"] * 0.25
        online_multipliers = [
            self.online_weights.multiplier(feature)
            for feature in features
            if _is_learnable_weight_feature(feature)
        ]
        online_weight_mean = (
            sum(online_multipliers) / len(online_multipliers)
            if online_multipliers
            else 1.0
        )
        score = (
            path_coverage_proxy
            + data_sensitivity
            + frontier_conformance
            + target_bonus
            + finding_yield_bonus
            + combo_priority
        )
        score -= feature_saturation_penalty + root_saturation_penalty
        return GuidanceDecision(
            case=case,
            score=score,
            features=sorted(features),
            matched_targets=matched_targets,
            candidate_count=candidate_count,
            frontier_buckets=frontier_buckets,
            online_weights=self.online_weights.snapshot(limit=16),
            score_breakdown={
                "path_coverage_proxy": path_coverage_proxy,
                "data_sensitivity": data_sensitivity,
                "frontier_conformance": frontier_conformance,
                "target_bonus": target_bonus,
                "target_priority": target_priority,
                "target_template_matches": float(target_template_matches),
                "target_template_bonus": target_template_bonus,
                "specific_target_matches": float(specific_target_matches),
                "finding_yield_bonus": finding_yield_bonus,
                "combo_priority": combo_priority,
                "contribution_potential": contribution_potential,
                "online_weight_mean": online_weight_mean,
                "online_weight_max": max(online_multipliers, default=1.0),
                "online_weight_updates": float(self.online_weights.total_updates),
                "feature_saturation_penalty": -feature_saturation_penalty,
                "root_saturation_penalty": -root_saturation_penalty,
            },
        )

    def _is_contributing_candidate(self, decision: GuidanceDecision) -> bool:
        features = set(decision.features)
        unseen_path = any(_is_path_feature(feature) and self.feature_counts[feature] == 0 for feature in features)
        unseen_frontier = any(self.frontier_bucket_counts[bucket] == 0 for bucket in decision.frontier_buckets)
        frontier_conformance = decision.score_breakdown.get("frontier_conformance", 0.0)
        contribution_potential = decision.score_breakdown.get("contribution_potential", 0.0)

        if decision.matched_targets:
            return True
        if unseen_path or unseen_frontier:
            return True
        if frontier_conformance >= 0.80:
            return True
        return contribution_potential >= 1.15


def _matched_targets(features: set[str], targets: list[str]) -> list[str]:
    matched = []
    for target in targets:
        required = TARGET_ALIASES.get(target, {target})
        if features & required:
            matched.append(target)
    return matched


def _target_match_priority(
    features: set[str],
    matched_targets: list[str],
    feature_counts: Counter[str],
) -> float:
    has_specific_target = any(_is_specific_target(target) for target in matched_targets)
    generic_weight = GENERIC_COMPANION_TARGET_WEIGHT if has_specific_target else 1.0
    return sum(
        _target_match_weight(target, generic_weight=generic_weight) for target in matched_targets
    ) + _target_template_bonus(features, matched_targets, feature_counts)


def _target_template_match_count(features: set[str], matched_targets: list[str]) -> int:
    return sum(
        1
        for target in matched_targets
        if f"generator_profile:{target}" in features or f"mixed_generator_profile:{target}" in features
    )


def _target_template_bonus(
    features: set[str],
    matched_targets: list[str],
    feature_counts: Counter[str],
) -> float:
    bonus = 0.0
    for target in matched_targets:
        template_features = [
            feature
            for feature in (f"mixed_generator_profile:{target}", f"generator_profile:{target}")
            if feature in features
        ]
        for feature in template_features:
            bonus += TEMPLATE_TARGET_BONUS / math.sqrt(1.0 + feature_counts[feature])
    return bonus


def _target_match_weight(target: str, *, generic_weight: float) -> float:
    if _is_specific_target(target):
        return PATTERN_TARGET_WEIGHT
    return generic_weight


def _specific_target_count(matched_targets: list[str]) -> int:
    return sum(1 for target in matched_targets if _is_specific_target(target))


def _is_specific_target(target: str) -> bool:
    return any(feature.startswith("pattern:") for feature in TARGET_ALIASES.get(target, {target}))


def _bucket(prefix: str, value: int, limits: list[tuple[int, str]], fallback: str) -> str:
    for limit, name in limits:
        if value <= limit:
            return f"{prefix}:{name}"
    return f"{prefix}:{fallback}"


def _bounded_finding_signal(count: int) -> float:
    if count <= 0:
        return 0.0
    return min(math.log1p(count), 2.0) / (1.0 + count / 50.0)


def _feature_saturation(count: int) -> float:
    if count <= 25:
        return 0.0
    return math.log1p(count - 25)


def _root_saturation(count: int) -> float:
    if count <= 20:
        return 0.0
    return math.log1p(count - 20) * 0.22


def _guidance_reward(row: dict[str, Any]) -> float:
    return online_case_reward(row)


def _finding_feature_weight(feature: str, online_weights: OnlineFeatureWeights | None = None) -> float:
    if feature.startswith(("op:", "opseq:")):
        base = 1.0
    elif feature.startswith(("agg:", "cmp:", "expr:", "mutate:", "join:", "filter:", "filter_type:", "cast_to:", "combo:", "running:")):
        base = 0.75
    elif feature.startswith(("has:", "type:", "nullable:")):
        base = 0.35
    else:
        base = 0.50
    return base * _online_multiplier(feature, online_weights)


def _path_coverage_proxy(
    features: set[str],
    feature_counts: Counter[str],
    online_weights: OnlineFeatureWeights | None = None,
) -> float:
    path_features = [feature for feature in features if _is_path_feature(feature)]
    if not path_features:
        return 0.0
    novelty = sum(
        _path_feature_weight(feature, online_weights) / (1.0 + feature_counts[feature])
        for feature in path_features
    )
    op_diversity = sum(1 for feature in path_features if feature.startswith("op:")) * 0.12
    sequence_bonus = 0.20 if any(feature.startswith("opseq:") for feature in path_features) else 0.0
    return novelty / math.sqrt(len(path_features)) + op_diversity + sequence_bonus


def _data_sensitivity_score(
    features: set[str],
    feature_counts: Counter[str],
    online_weights: OnlineFeatureWeights | None = None,
) -> float:
    data_features = [feature for feature in features if _is_data_sensitivity_feature(feature)]
    if not data_features:
        return 0.0
    weighted = sum(
        _data_feature_weight(feature, online_weights) * (1.0 + 1.0 / (1.0 + feature_counts[feature]))
        for feature in data_features
    )
    return weighted / math.sqrt(len(data_features)) * 0.35


def _saturation_feature_weight(feature: str, online_weights: OnlineFeatureWeights | None = None) -> float:
    if feature.startswith("opseq:"):
        base = 1.0
    elif feature.startswith("op:"):
        base = 0.9
    elif feature.startswith(("agg:", "cmp:", "expr:", "mutate:", "join:", "filter_type:", "cast_to:", "combo:", "running:")):
        base = 0.65
    elif feature.startswith(("has:", "type:", "nullable:")):
        base = 0.15
    else:
        base = 0.25
    return base * _online_multiplier(feature, online_weights)


def _is_path_feature(feature: str) -> bool:
    return feature.startswith(
        (
            "op:",
            "opseq:",
            "pattern:",
            "cmp:",
            "filter:",
            "filter_type:",
            "select_width:",
            "sort:",
            "join:",
            "join_table:",
            "limit:",
            "offset:",
            "mutate:",
            "expr:",
            "arith:",
            "cast_to:",
            "group_key_type:",
            "agg:",
            "op_count:",
            "groupby:",
            "running:",
            "running_source_type:",
            "combo:",
            "combo_frequency:",
            "combo_risk:",
        )
    )


def _path_feature_weight(feature: str, online_weights: OnlineFeatureWeights | None = None) -> float:
    if feature.startswith("opseq:"):
        base = 1.4
    elif feature.startswith("pattern:"):
        base = 1.6
    elif feature.startswith("op:"):
        base = 1.0
    elif feature.startswith(("combo:", "combo_risk:")):
        base = 1.1
    elif feature.startswith("combo_frequency:"):
        base = 0.8
    elif feature.startswith(("running:", "running_source_type:")):
        base = 1.0
    elif feature.startswith(("join:", "agg:", "expr:", "mutate:", "cmp:", "filter:", "group_key_type:")):
        base = 0.9
    elif feature.startswith(("filter_type:", "select_width:", "sort:", "limit:", "offset:", "cast_to:", "arith:")):
        base = 0.7
    else:
        base = 0.5
    return base * _online_multiplier(feature, online_weights)


def _is_data_sensitivity_feature(feature: str) -> bool:
    return feature.startswith(
        (
            "tables:",
            "rows:",
            "cols:",
            "type:",
            "nullable:",
            "has:",
            "bool:",
        )
    ) or feature in {
        "op:limit_zero",
        "groupby:null-key",
        "groupby:null-agg-output",
        "filter:empty-output",
        "offset:row-boundary",
        "offset:large",
        "running:long",
        "running:small-increment",
    }


def _data_feature_weight(feature: str, online_weights: OnlineFeatureWeights | None = None) -> float:
    if feature == "groupby:null-key":
        base = 1.6
    elif feature == "groupby:null-agg-output":
        base = 1.7
    elif feature == "filter:empty-output":
        base = 1.5
    elif feature.startswith("offset:"):
        base = 1.1
    elif feature in {"running:long", "running:small-increment"}:
        base = 1.4
    elif feature in {"has:special_float", "has:null", "has:unicode_string"}:
        base = 1.8
    elif feature in {"has:fractional_float", "has:negative_number", "has:empty_string", "has:space_string"}:
        base = 1.2
    elif feature in {"tables:multi", "rows:empty", "rows:tiny", "cols:wide", "op:limit_zero"}:
        base = 1.0
    elif feature.startswith(("nullable:", "type:")):
        base = 0.6
    elif feature.startswith("bool:"):
        base = 0.4
    else:
        base = 0.5
    return base * _online_multiplier(feature, online_weights)


def _online_multiplier(feature: str, online_weights: OnlineFeatureWeights | None) -> float:
    if online_weights is None:
        return 1.0
    return online_weights.multiplier(feature)


def _is_learnable_weight_feature(feature: str) -> bool:
    return _is_path_feature(feature) or _is_data_sensitivity_feature(feature)


def _feature_prefix(feature: str) -> str:
    return feature.split(":", 1)[0] if ":" in feature else feature


def _frontier_conformance(case: Case, frontier_bucket_counts: Counter[str]) -> tuple[float, list[str]]:
    raw_score, frontier_buckets = _frontier_signature(case)
    if not frontier_buckets:
        return raw_score, frontier_buckets
    novelty = (
        sum(1.0 / (1.0 + frontier_bucket_counts[bucket]) for bucket in frontier_buckets)
        / math.sqrt(len(frontier_buckets))
        * 0.35
    )
    return raw_score + novelty, frontier_buckets


def _frontier_signature(case: Case) -> tuple[float, list[str]]:
    if not case.tables:
        return 0.0, []

    table_by_name = {table.name: table for table in case.tables}
    samples = {column.name: [row.get(column.name) for row in case.tables[0].rows] for column in case.tables[0].columns}
    scores: list[float] = []
    buckets: list[str] = []
    last_sort_op: dict[str, Any] | None = None

    for op in case.program.operations:
        kind = str(op.get("op", ""))
        if kind == "filter":
            score, op_buckets = _filter_frontier_score(samples, op)
            scores.append(score)
            buckets.extend(op_buckets)
            samples = _filter_output_samples(samples, op)
        elif kind == "tuple_absence_filter":
            right = table_by_name.get(str(op.get("table", "")))
            score, op_buckets = _tuple_absence_frontier_score(samples, right, op)
            scores.append(score)
            buckets.extend(op_buckets)
            samples = _tuple_absence_output_samples(samples, right, op)
        elif kind == "join":
            right = table_by_name.get(str(op.get("table", "")))
            score, op_buckets = _join_frontier_score(samples, right, op)
            scores.append(score)
            buckets.extend(op_buckets)
            if right is not None:
                samples = _join_output_samples(samples, right, op)
            last_sort_op = None
        elif kind == "running_sum":
            score, op_buckets, samples = _running_sum_frontier_score(samples, op)
            scores.append(score)
            buckets.extend(op_buckets)
            last_sort_op = {"op": "sort", "keys": op.get("order_by", [])}
        elif kind == "sortedness_check":
            score, op_buckets, samples = _sortedness_frontier_score(samples, op, last_sort_op)
            scores.append(score)
            buckets.extend(op_buckets)
            last_sort_op = None
        elif kind == "random_case_probe":
            score, op_buckets, samples = _random_case_frontier_score(op)
            scores.append(score)
            buckets.extend(op_buckets)
            last_sort_op = None
        elif kind == "group_quantile_probe":
            score, op_buckets, samples = _group_quantile_frontier_score(op)
            scores.append(score)
            buckets.extend(op_buckets)
            last_sort_op = None
        elif kind == "scalar_subquery_probe":
            score, op_buckets, samples = _scalar_subquery_frontier_score(op)
            scores.append(score)
            buckets.extend(op_buckets)
            last_sort_op = None
        elif kind == "window_avg_probe":
            score, op_buckets, samples = _window_avg_frontier_score(op)
            scores.append(score)
            buckets.extend(op_buckets)
            last_sort_op = None
        elif kind == "struct_distinct_probe":
            score, op_buckets, samples = _struct_distinct_frontier_score(op)
            scores.append(score)
            buckets.extend(op_buckets)
            last_sort_op = None
        elif kind == "bit_compare_probe":
            score, op_buckets, samples = _bit_compare_frontier_score(op)
            scores.append(score)
            buckets.extend(op_buckets)
            last_sort_op = None
        elif kind == "round_even_probe":
            score, op_buckets, samples = _round_even_frontier_score(op)
            scores.append(score)
            buckets.extend(op_buckets)
            last_sort_op = None
        elif kind == "series_rtruediv_probe":
            score, op_buckets, samples = _series_rtruediv_frontier_score(op)
            scores.append(score)
            buckets.extend(op_buckets)
            last_sort_op = None
        elif kind == "uint64_isin_probe":
            score, op_buckets, samples = _uint64_isin_frontier_score(op)
            scores.append(score)
            buckets.extend(op_buckets)
            last_sort_op = None
        elif kind == "tuple_anti_null_probe":
            score, op_buckets, samples = _tuple_anti_null_frontier_score(op)
            scores.append(score)
            buckets.extend(op_buckets)
            last_sort_op = None
        elif kind == "sparse_mask_probe":
            score, op_buckets, samples = _sparse_mask_frontier_score(op)
            scores.append(score)
            buckets.extend(op_buckets)
            last_sort_op = None
        elif kind == "float_wrap_probe":
            score, op_buckets, samples = _float_wrap_frontier_score(op)
            scores.append(score)
            buckets.extend(op_buckets)
            last_sort_op = None
        elif kind == "index_bool_probe":
            score, op_buckets, samples = _index_bool_frontier_score(op)
            scores.append(score)
            buckets.extend(op_buckets)
            last_sort_op = None
        elif kind == "empty_literal_groupby_probe":
            score, op_buckets, samples = _empty_literal_groupby_frontier_score(op)
            scores.append(score)
            buckets.extend(op_buckets)
            last_sort_op = None
        elif kind == "arrow_string_eq_sum_probe":
            score, op_buckets, samples = _arrow_string_eq_sum_frontier_score(op)
            scores.append(score)
            buckets.extend(op_buckets)
            last_sort_op = None
        elif kind == "arrow_timestamp_loc_slice_probe":
            score, op_buckets, samples = _arrow_timestamp_loc_slice_frontier_score(op)
            scores.append(score)
            buckets.extend(op_buckets)
            last_sort_op = None
        elif kind == "arrow_timestamp_index_attr_probe":
            score, op_buckets, samples = _arrow_timestamp_index_attr_frontier_score(op)
            scores.append(score)
            buckets.extend(op_buckets)
            last_sort_op = None
        elif kind == "select":
            cols = [str(column) for column in op.get("columns", []) if str(column) in samples]
            samples = {column: samples[column] for column in unique_preserve_order(cols)}
        elif kind == "sort":
            score, op_buckets = _sort_frontier_score(samples, op)
            scores.append(score)
            buckets.extend(op_buckets)
            samples = _sort_output_samples(samples, op)
            last_sort_op = op
        elif kind == "limit":
            score, op_buckets = _limit_frontier_score(samples, op)
            scores.append(score)
            buckets.extend(op_buckets)
            samples = _limit_output_samples(samples, op)
        elif kind == "offset":
            score, op_buckets = _offset_frontier_score(samples, op)
            scores.append(score)
            buckets.extend(op_buckets)
            samples = _offset_output_samples(samples, op)
        elif kind == "mutate":
            score, op_buckets, values = _mutate_frontier_score(samples, op)
            scores.append(score)
            buckets.extend(op_buckets)
            column = str(op.get("column", ""))
            if column:
                samples[column] = values
        elif kind == "groupby":
            score, op_buckets = _groupby_frontier_score(samples, op)
            scores.append(score)
            buckets.extend(op_buckets)
            samples = _groupby_output_samples(samples, op)
            last_sort_op = None
        elif kind == "aggregate":
            score, op_buckets = _aggregate_frontier_score(samples, op)
            scores.append(score)
            buckets.extend(op_buckets)
            samples = {str(agg.get("as", "")): [] for agg in op.get("aggs", []) if agg.get("as")}
            last_sort_op = None

    scores = [score for score in scores if score > 0.0]
    return (sum(scores) / len(scores) if scores else 0.0), unique_preserve_order(buckets)


def _contribution_potential(
    features: set[str],
    frontier_buckets: list[str],
    matched_targets: list[str],
    frontier_conformance: float,
    *,
    feature_counts: Counter[str],
    frontier_bucket_counts: Counter[str],
    root_cause_counts: Counter[str],
) -> float:
    path_novelty = sum(1 for feature in features if _is_path_feature(feature) and feature_counts[feature] == 0)
    data_novelty = sum(1 for feature in features if _is_data_sensitivity_feature(feature) and feature_counts[feature] == 0)
    frontier_novelty = sum(1 for bucket in frontier_buckets if frontier_bucket_counts[bucket] == 0)
    root_novelty = sum(1 for root in _predicted_roots(features) if root_cause_counts[root] == 0)
    root_saturation = sum(1 for root in _predicted_roots(features) if root_cause_counts[root] >= 40)
    return (
        frontier_conformance
        + 0.30 * path_novelty
        + 0.20 * data_novelty
        + 0.45 * frontier_novelty
        + 0.35 * root_novelty
        + 0.60 * len(matched_targets)
        - 0.20 * root_saturation
    )


def _filter_frontier_score(samples: dict[str, list[Any]], op: dict[str, Any]) -> tuple[float, list[str]]:
    column = str(op.get("column", ""))
    values = samples.get(column, [])
    comparator = str(op.get("cmp", ""))
    literal = op.get("value")
    buckets: list[str] = []
    if not values:
        return 0.0, buckets
    mask = _filter_mask(values, op)
    kept = sum(1 for keep in mask if keep)
    if kept == 0:
        buckets.append("filter:empty-output")
    elif kept == len(values):
        buckets.append("filter:all-pass")
    else:
        buckets.append("filter:partial-output")

    if any(value is None for value in values):
        buckets.append("filter:null-aware")
    parsed = parse_filter_comparator(comparator)
    if parsed is not None and parsed.base == "in_set":
        buckets.append("filter:set-membership")
    if parsed is not None and parsed.base in {"is_null", "is_not_null"}:
        buckets.append("filter:null-predicate")
        buckets.append(f"filter:null-predicate:{parsed.base}")
    if parsed is not None and parsed.base == "bool_predicate":
        buckets.append("filter:boolean-predicate")
        buckets.append(f"filter:boolean-predicate:{parsed.truth_test}")
    if parsed is not None and parsed.base == "range_closed":
        buckets.append("filter:range-closed")
    if parsed is not None and parsed.truth_test is not None:
        buckets.append("filter:truth-test")
        buckets.append(f"filter:truth:{parsed.truth_test}")
    numeric_values = _numeric_values(values)
    if numeric_values and isinstance(literal, (int, float)) and not isinstance(literal, bool):
        spread = max(numeric_values) - min(numeric_values) if len(numeric_values) > 1 else 0.0
        scale = max(1.0, abs(float(literal)), spread)
        min_diff = min(abs(value - float(literal)) for value in numeric_values)
        closeness = 1.0 / (1.0 + (min_diff / scale))
        if min_diff == 0:
            buckets.append("filter:exact-hit")
        elif closeness >= 0.65:
            buckets.append("filter:near-hit")
        else:
            buckets.append("filter:range-probe")
        if comparator in {">", "<", ">=", "<="}:
            buckets.append(f"filter:cmp:{comparator}")
        return min(
            1.0,
            0.40
            + 0.45 * closeness
            + (0.10 if "filter:null-aware" in buckets else 0.0)
            + (0.08 if "filter:empty-output" in buckets else 0.0),
        ), buckets

    scalar_values = [value for value in values if value is not None]
    if isinstance(literal, list):
        hits = sum(1 for value in scalar_values if value in literal)
        if hits:
            buckets.append("filter:set-hit")
        return min(
            1.0,
            0.55
            + (0.20 if hits and hits < len(scalar_values) else 0.0)
            + (0.10 if "filter:null-aware" in buckets else 0.0),
        ), buckets
    if literal in scalar_values:
        buckets.append("filter:exact-hit")
        return 0.85, buckets
    if isinstance(literal, str):
        buckets.append("filter:string-literal")
        return 0.55 + (0.10 if any(isinstance(value, str) and value == "" for value in scalar_values) else 0.0), buckets
    if isinstance(literal, bool):
        buckets.append("filter:bool-literal")
        return 0.60, buckets
    return 0.30, buckets


def _filter_output_samples(samples: dict[str, list[Any]], op: dict[str, Any]) -> dict[str, list[Any]]:
    column = str(op.get("column", ""))
    values = samples.get(column)
    if values is None:
        return samples
    mask = _filter_mask(values, op)
    return {
        name: [value for value, keep in zip(column_values, mask) if keep]
        for name, column_values in samples.items()
    }


def _filter_mask(values: list[Any], op: dict[str, Any]) -> list[bool]:
    comparator = str(op.get("cmp", ""))
    literal = op.get("value")
    mask: list[bool] = []
    for value in values:
        try:
            mask.append(evaluate_filter_predicate(value, comparator, literal))
        except Exception:
            mask.append(False)
    return mask


def _tuple_absence_frontier_score(
    samples: dict[str, list[Any]],
    right: Any,
    op: dict[str, Any],
) -> tuple[float, list[str]]:
    buckets = ["filter:tuple-absence"]
    left_columns = [str(column) for column in op.get("columns", [])]
    right_columns = [str(column) for column in op.get("right_columns", [])]
    if right is None or not left_columns or len(left_columns) != len(right_columns):
        return 0.0, buckets
    rows = _rows_from_samples(samples)
    right_rows = list(getattr(right, "rows", []))
    if _tuple_absence_has_nulls(rows, left_columns) or _tuple_absence_has_nulls(right_rows, right_columns):
        buckets.append("filter:null-aware")
        buckets.append("tuple_absence:null")
    mask = [
        evaluate_tuple_absence(row, left_columns, right_rows, right_columns)
        for row in rows
    ]
    passed = sum(1 for keep in mask if keep)
    if passed == 0:
        buckets.append("tuple_absence:empty-output")
    elif passed == len(mask):
        buckets.append("tuple_absence:all-pass")
    else:
        buckets.append("tuple_absence:partial-output")
    return min(1.0, 0.40 + 0.25 * int("tuple_absence:null" in buckets) + 0.20 * int("tuple_absence:partial-output" in buckets)), buckets


def _tuple_absence_output_samples(
    samples: dict[str, list[Any]],
    right: Any,
    op: dict[str, Any],
) -> dict[str, list[Any]]:
    left_columns = [str(column) for column in op.get("columns", [])]
    right_columns = [str(column) for column in op.get("right_columns", [])]
    if right is None or not left_columns or len(left_columns) != len(right_columns):
        return samples
    rows = _rows_from_samples(samples)
    right_rows = list(getattr(right, "rows", []))
    mask = [
        evaluate_tuple_absence(row, left_columns, right_rows, right_columns)
        for row in rows
    ]
    return {
        name: [value for value, keep in zip(column_values, mask) if keep]
        for name, column_values in samples.items()
    }


def _tuple_absence_has_nulls(rows: list[dict[str, Any]], columns: list[str]) -> bool:
    return any(row.get(column) is None for row in rows for column in columns)


def _join_frontier_score(
    samples: dict[str, list[Any]],
    right: Any,
    op: dict[str, Any],
) -> tuple[float, list[str]]:
    left_on = str(op.get("left_on", ""))
    right_on = str(op.get("right_on", ""))
    left_values = [value for value in samples.get(left_on, []) if value is not None]
    right_values = [] if right is None else [row.get(right_on) for row in right.rows if row.get(right_on) is not None]
    buckets: list[str] = []
    if right is None or (not left_values and not right_values):
        return 0.0, buckets

    left_set = set(left_values)
    right_set = set(right_values)
    union = left_set | right_set
    overlap = len(left_set & right_set) / max(1, len(union))
    partial_overlap = 1.0 - abs(overlap - 0.5) * 2.0
    if overlap == 0.0:
        buckets.append("join:no-overlap")
    elif overlap >= 0.95:
        buckets.append("join:full-overlap")
    else:
        buckets.append("join:partial-overlap")
    if len(left_values) != len(left_set) or len(right_values) != len(right_set):
        buckets.append("join:duplicate-keys")
    if any(value is None for value in samples.get(left_on, [])) or any(row.get(right_on) is None for row in getattr(right, "rows", [])):
        buckets.append("join:null-keys")
    return min(1.0, 0.35 + 0.45 * max(0.0, partial_overlap) + 0.10 * int("join:duplicate-keys" in buckets) + 0.05 * int("join:null-keys" in buckets)), buckets


def _join_output_samples(
    left_samples: dict[str, list[Any]],
    right: Any,
    op: dict[str, Any],
) -> dict[str, list[Any]]:
    left_on = str(op.get("left_on", ""))
    right_on = str(op.get("right_on", ""))
    if left_on not in left_samples:
        return left_samples
    right_rows = getattr(right, "rows", [])
    right_columns = [
        column.name
        for column in getattr(right, "columns", [])
        if column.name != right_on and column.name not in left_samples
    ]
    if not right_rows:
        return left_samples

    left_rows = _rows_from_samples(left_samples)
    right_index: dict[Any, list[dict[str, Any]]] = {}
    for row in right_rows:
        key = row.get(right_on)
        if key is not None:
            right_index.setdefault(key, []).append(row)

    how = str(op.get("how", "inner"))
    joined_rows: list[dict[str, Any]] = []
    for left_row in left_rows:
        key = left_row.get(left_on)
        matches = [] if key is None else right_index.get(key, [])
        if matches:
            for right_row in matches:
                out = dict(left_row)
                for column in right_columns:
                    out[column] = right_row.get(column)
                joined_rows.append(out)
        elif how == "left":
            out = dict(left_row)
            for column in right_columns:
                out[column] = None
            joined_rows.append(out)

    return _samples_from_rows(joined_rows, list(left_samples) + right_columns)


def _rows_from_samples(samples: dict[str, list[Any]]) -> list[dict[str, Any]]:
    row_count = min((len(values) for values in samples.values()), default=0)
    return [
        {column: values[idx] for column, values in samples.items()}
        for idx in range(row_count)
    ]


def _samples_from_rows(rows: list[dict[str, Any]], columns: list[str]) -> dict[str, list[Any]]:
    return {
        column: [row.get(column) for row in rows]
        for column in columns
    }


def _running_sum_frontier_score(
    samples: dict[str, list[Any]],
    op: dict[str, Any],
) -> tuple[float, list[str], dict[str, list[Any]]]:
    source = str(op.get("source", ""))
    column = str(op.get("column", ""))
    values = samples.get(source, [])
    buckets: list[str] = []
    if not source or not column or source not in samples:
        return 0.0, buckets, samples
    try:
        sort_keys = normalize_sort_keys({"keys": op.get("order_by", [])})
    except ValueError:
        return 0.0, buckets, samples

    rows = sort_rows_for_running(_rows_from_samples(samples), sort_keys)
    running_values = stable_running_sum_values(rows, source)
    out_rows = [{**row, column: value} for row, value in zip(rows, running_values)]
    out_columns = [name for name in samples if name != column] + [column]

    input_dtype = str(op.get("input_dtype", "float64"))
    buckets.append("running:ordered")
    buckets.append(f"running:{input_dtype}")
    if len(values) >= 10_000:
        buckets.append("running:long")
    numeric_values = _numeric_values(values)
    if numeric_values and max(abs(value) for value in numeric_values) <= 0.001:
        buckets.append("running:small-increment")
    if any(value is None for value in values):
        buckets.append("running:null-source")
    score = (
        0.35
        + 0.25 * int(input_dtype == "float32")
        + 0.20 * int("running:long" in buckets)
        + 0.15 * int("running:small-increment" in buckets)
        + 0.05 * int("running:null-source" in buckets)
    )
    return min(1.0, score), buckets, _samples_from_rows(out_rows, out_columns)


def _sortedness_frontier_score(
    samples: dict[str, list[Any]],
    op: dict[str, Any],
    last_sort_op: dict[str, Any] | None,
) -> tuple[float, list[str], dict[str, list[Any]]]:
    column = str(op.get("column", ""))
    alias = str(op.get("as", ""))
    values = list(samples.get(column, []))
    ascending = bool(op.get("ascending", True))
    nulls = str(op.get("nulls", "last"))
    ok = is_sorted_values(values, ascending=ascending, nulls=nulls)
    buckets = [
        "sortedness:check",
        f"sortedness:nulls:{nulls}",
        f"sortedness:{'true' if ok else 'false'}",
    ]
    if any(_is_null_like(value) for value in values):
        buckets.append("sortedness:null-aware")
    if _sort_null_placement_mismatch(last_sort_op, column, nulls):
        buckets.append("sortedness:null-placement-mismatch")
    score = (
        0.45
        + 0.25 * int("sortedness:null-aware" in buckets)
        + 0.25 * int("sortedness:null-placement-mismatch" in buckets)
        + 0.05 * int(not ok)
    )
    return min(1.0, score), buckets, {alias: [ok]} if alias else samples


def _random_case_frontier_score(op: dict[str, Any]) -> tuple[float, list[str], dict[str, list[Any]]]:
    alias = str(op.get("as", ""))
    row_count = int(op.get("rows", 0))
    branch_count = int(op.get("branches", 0))
    buckets = [
        "case_expr:simple",
        "case_expr:random-subject",
        _bucket("case_probe_rows", row_count, [(1000, "small"), (10000, "medium")], "large"),
    ]
    if branch_count >= 3:
        buckets.append("case_expr:multi-branch")
    score = 0.55 + 0.25 * int(row_count >= 10_000) + 0.10 * int(branch_count >= 3)
    return min(1.0, score), buckets, {alias: [False]} if alias else {}


def _group_quantile_frontier_score(op: dict[str, Any]) -> tuple[float, list[str], dict[str, list[Any]]]:
    alias = str(op.get("as", ""))
    values = list(op.get("values", []))
    quantiles = list(op.get("quantiles", []))
    buckets = [
        "quantile:dynamic-key",
        _bucket("quantile_probe_values", len(values), [(2, "tiny"), (4, "small")], "medium"),
    ]
    if len(set(quantiles)) > 1:
        buckets.append("quantile:multi-probability")
    score = 0.55 + 0.25 * int("quantile:multi-probability" in buckets) + 0.10 * int(len(values) >= 3)
    return min(1.0, score), buckets, {alias: [False]} if alias else {}


def _scalar_subquery_frontier_score(op: dict[str, Any]) -> tuple[float, list[str], dict[str, list[Any]]]:
    alias = str(op.get("as", ""))
    buckets = ["subquery:correlated-scalar", "subquery:nested-aggregate"]
    return 0.90, buckets, {alias: [False]} if alias else {}


def _window_avg_frontier_score(op: dict[str, Any]) -> tuple[float, list[str], dict[str, list[Any]]]:
    alias = str(op.get("as", ""))
    buckets = ["window:rows-frame", "window:avg"]
    return 0.90, buckets, {alias: [False]} if alias else {}


def _struct_distinct_frontier_score(op: dict[str, Any]) -> tuple[float, list[str], dict[str, list[Any]]]:
    alias = str(op.get("as", ""))
    buckets = ["struct:unnest", "struct:distinct"]
    return 0.90, buckets, {alias: [False]} if alias else {}


def _bit_compare_frontier_score(op: dict[str, Any]) -> tuple[float, list[str], dict[str, list[Any]]]:
    alias = str(op.get("as", ""))
    buckets = ["bit:unequal-length", "comparison:bit-order"]
    return 0.90, buckets, {alias: [False]} if alias else {}


def _round_even_frontier_score(op: dict[str, Any]) -> tuple[float, list[str], dict[str, list[Any]]]:
    alias = str(op.get("as", ""))
    buckets = ["numeric:round-even", "float:decimal-scale"]
    return 0.90, buckets, {alias: [False]} if alias else {}


def _series_rtruediv_frontier_score(op: dict[str, Any]) -> tuple[float, list[str], dict[str, list[Any]]]:
    alias = str(op.get("as", ""))
    buckets = ["series:reverse-division", "arithmetic:operand-order"]
    return 0.90, buckets, {alias: [False]} if alias else {}


def _uint64_isin_frontier_score(op: dict[str, Any]) -> tuple[float, list[str], dict[str, list[Any]]]:
    alias = str(op.get("as", ""))
    buckets = ["pandas:uint64-isin", "membership:unsigned-precision"]
    return 0.90, buckets, {alias: [False]} if alias else {}


def _tuple_anti_null_frontier_score(op: dict[str, Any]) -> tuple[float, list[str], dict[str, list[Any]]]:
    alias = str(op.get("as", ""))
    buckets = ["duckdb:tuple-anti-null", "nulls:ternary-membership"]
    return 0.90, buckets, {alias: [False]} if alias else {}


def _sparse_mask_frontier_score(op: dict[str, Any]) -> tuple[float, list[str], dict[str, list[Any]]]:
    alias = str(op.get("as", ""))
    buckets = ["pandas:sparse-mask", "mask:sparse-array"]
    return 0.90, buckets, {alias: [False]} if alias else {}


def _float_wrap_frontier_score(op: dict[str, Any]) -> tuple[float, list[str], dict[str, list[Any]]]:
    alias = str(op.get("as", ""))
    buckets = ["polars:wrap-numerical", "cast:float-overflow"]
    return 0.90, buckets, {alias: [False]} if alias else {}


def _index_bool_frontier_score(op: dict[str, Any]) -> tuple[float, list[str], dict[str, list[Any]]]:
    alias = str(op.get("as", ""))
    buckets = ["pandas:index-bool", "api:result-type"]
    return 0.90, buckets, {alias: [False]} if alias else {}


def _empty_literal_groupby_frontier_score(op: dict[str, Any]) -> tuple[float, list[str], dict[str, list[Any]]]:
    alias = str(op.get("as", ""))
    buckets = ["polars:empty-literal-groupby", "groupby:empty-literal"]
    return 0.90, buckets, {alias: [False]} if alias else {}


def _arrow_string_eq_sum_frontier_score(op: dict[str, Any]) -> tuple[float, list[str], dict[str, list[Any]]]:
    alias = str(op.get("as", ""))
    buckets = ["pandas:arrow-string-eq-sum", "arrow:string-bool-reduction"]
    return 0.90, buckets, {alias: [False]} if alias else {}


def _arrow_timestamp_loc_slice_frontier_score(op: dict[str, Any]) -> tuple[float, list[str], dict[str, list[Any]]]:
    alias = str(op.get("as", ""))
    buckets = ["pandas:arrow-timestamp-loc-slice", "arrow:timestamp-index-slice"]
    return 0.90, buckets, {alias: [False]} if alias else {}


def _arrow_timestamp_index_attr_frontier_score(op: dict[str, Any]) -> tuple[float, list[str], dict[str, list[Any]]]:
    alias = str(op.get("as", ""))
    buckets = ["pandas:arrow-timestamp-index-attr", "arrow:timestamp-index-attribute"]
    return 0.90, buckets, {alias: [False]} if alias else {}


def _groupby_frontier_score(samples: dict[str, list[Any]], op: dict[str, Any]) -> tuple[float, list[str]]:
    keys = [str(key) for key in op.get("keys", []) if str(key) in samples]
    buckets: list[str] = []
    if not keys:
        return 0.0, buckets
    row_count = min((len(samples[key]) for key in keys), default=0)
    if row_count <= 0:
        return 0.0, buckets
    tuples = [tuple(samples[key][idx] for key in keys) for idx in range(row_count)]
    distinct = len(set(tuples))
    distinct_ratio = distinct / max(1, row_count)
    mixed = 1.0 - abs(distinct_ratio - 0.5) * 2.0
    if distinct == 1:
        buckets.append("groupby:single-group")
    elif distinct == row_count:
        buckets.append("groupby:high-cardinality")
    else:
        buckets.append("groupby:mixed-cardinality")
    if any(any(value is None for value in group) for group in tuples):
        buckets.append("groupby:null-key")
    if len(op.get("aggs", [])) > 1:
        buckets.append("groupby:multi-agg")
    if _has_null_aggregate_output(samples, op, tuples):
        buckets.append("groupby:null-agg-output")
    return min(1.0, 0.35 + 0.45 * max(0.0, mixed) + 0.10 * int("groupby:multi-agg" in buckets) + 0.05 * int("groupby:null-key" in buckets) + 0.08 * int("groupby:null-agg-output" in buckets)), buckets


def _groupby_output_samples(samples: dict[str, list[Any]], op: dict[str, Any]) -> dict[str, list[Any]]:
    keys = [str(key) for key in op.get("keys", []) if str(key) in samples]
    row_count = min((len(samples[key]) for key in keys), default=0)
    if row_count <= 0:
        return {key: [] for key in keys}
    groups: dict[tuple[Any, ...], list[int]] = {}
    for idx in range(row_count):
        key_tuple = tuple(samples[key][idx] for key in keys)
        groups.setdefault(key_tuple, []).append(idx)

    out: dict[str, list[Any]] = {key: [] for key in keys}
    for key_tuple in groups:
        for idx, key in enumerate(keys):
            out[key].append(key_tuple[idx])
    for agg in op.get("aggs", []):
        alias = str(agg.get("as", ""))
        source = str(agg.get("column", ""))
        func = str(agg.get("func", ""))
        if not alias:
            continue
        source_values = samples.get(source, [])
        values = []
        for indices in groups.values():
            group_values = [source_values[idx] for idx in indices if idx < len(source_values)]
            non_null = [value for value in group_values if value is not None]
            if func == "count":
                values.append(len(non_null))
            elif not non_null:
                values.append(None)
            elif func == "sum":
                values.append(sum(non_null))
            elif func == "min":
                values.append(min(non_null))
            elif func == "max":
                values.append(max(non_null))
            else:
                values.append(None)
        out[alias] = values
    return out


def _aggregate_frontier_score(samples: dict[str, list[Any]], op: dict[str, Any]) -> tuple[float, list[str]]:
    buckets = ["aggregate:global"]
    if len(op.get("aggs", [])) > 1:
        buckets.append("aggregate:multi-agg")
    for agg in op.get("aggs", []):
        if agg.get("func") == "count":
            continue
        values = samples.get(str(agg.get("column", "")), [])
        if not values:
            buckets.append("aggregate:empty-input")
            break
        if all(value is None for value in values):
            buckets.append("aggregate:null-output")
            break
    score = (
        0.55
        + 0.12 * int("aggregate:multi-agg" in buckets)
        + 0.08 * int("aggregate:null-output" in buckets)
        + 0.08 * int("aggregate:empty-input" in buckets)
    )
    return min(1.0, score), buckets


def _has_null_aggregate_output(
    samples: dict[str, list[Any]],
    op: dict[str, Any],
    tuples: list[tuple[Any, ...]],
) -> bool:
    if not tuples:
        return False
    groups: dict[tuple[Any, ...], list[int]] = {}
    for idx, key_tuple in enumerate(tuples):
        groups.setdefault(key_tuple, []).append(idx)
    for agg in op.get("aggs", []):
        if agg.get("func") == "count":
            continue
        source_values = samples.get(str(agg.get("column", "")), [])
        for indices in groups.values():
            if all(idx >= len(source_values) or source_values[idx] is None for idx in indices):
                return True
    return False


def _mutate_frontier_score(samples: dict[str, list[Any]], op: dict[str, Any]) -> tuple[float, list[str], list[Any]]:
    expr = op.get("expr", {})
    kind = str(expr.get("kind", ""))
    source = str(expr.get("source", ""))
    values = list(samples.get(source, []))
    buckets: list[str] = []
    if not values:
        return 0.0, buckets, values

    if kind in {"add_const", "arith_const", "cast"}:
        numeric_values = _numeric_values(values)
        if any(value is None for value in values):
            buckets.append("mutate:null-propagation")
        if any(value < 0 for value in numeric_values):
            buckets.append("mutate:negative")
        if any(value == 0 for value in numeric_values):
            buckets.append("mutate:zero")
        if any(not float(value).is_integer() for value in numeric_values):
            buckets.append("mutate:fractional")
        if kind == "arith_const":
            buckets.append(f"mutate:arith:{expr.get('op', 'unknown')}")
        if kind == "cast":
            buckets.append("mutate:cast")
        score = 0.35 + 0.15 * int("mutate:null-propagation" in buckets) + 0.15 * int("mutate:negative" in buckets) + 0.15 * int("mutate:zero" in buckets) + 0.10 * int("mutate:fractional" in buckets)
        if kind == "arith_const" and expr.get("op") in {"div", "mod"}:
            score += 0.10
        return min(1.0, score), buckets, _evaluate_mutate_values(values, expr)

    string_values = [value for value in values if isinstance(value, str)]
    if any(value == "" for value in string_values):
        buckets.append("mutate:empty-string")
    if any(" " in value for value in string_values):
        buckets.append("mutate:space-string")
    if any(any(ord(ch) > 127 for ch in value) for value in string_values):
        buckets.append("mutate:unicode-string")
    if any(_has_mixed_case(value) for value in string_values):
        buckets.append("mutate:mixed-case")
    score = 0.35 + 0.15 * int("mutate:empty-string" in buckets) + 0.10 * int("mutate:space-string" in buckets) + 0.15 * int("mutate:unicode-string" in buckets) + 0.15 * int("mutate:mixed-case" in buckets)
    return min(1.0, score), buckets, _evaluate_mutate_values(values, expr)


def _sort_frontier_score(samples: dict[str, list[Any]], op: dict[str, Any]) -> tuple[float, list[str]]:
    try:
        keys = [key for key in normalize_sort_keys(op) if key.column in samples]
    except ValueError:
        return 0.0, []
    columns = [key.column for key in keys]
    buckets: list[str] = []
    if not columns:
        return 0.0, buckets
    values = samples[columns[0]]
    non_null = [value for value in values if value is not None]
    if len(non_null) != len(set(non_null)):
        buckets.append("sort:duplicate-key")
    if any(value is None for value in values):
        buckets.append("sort:null-order")
    if len({key.ascending for key in keys}) > 1:
        buckets.append("sort:mixed-direction")
    if len({key.nulls for key in keys}) > 1:
        buckets.append("sort:mixed-null-placement")
    if not buckets:
        return 0.0, buckets
    return min(1.0, 0.30 + 0.20 * len(buckets)), buckets


def _sort_output_samples(samples: dict[str, list[Any]], op: dict[str, Any]) -> dict[str, list[Any]]:
    try:
        keys = [key for key in normalize_sort_keys(op) if key.column in samples]
    except ValueError:
        return samples
    if not keys:
        return samples
    columns = list(samples)
    rows = _rows_from_samples(samples)
    rows = sorted(rows, key=cmp_to_key(lambda left, right: _compare_sample_rows(left, right, keys)))
    return _samples_from_rows(rows, columns)


def _compare_sample_rows(left: dict[str, Any], right: dict[str, Any], keys: list[Any]) -> int:
    for key in keys:
        left_value = left.get(key.column)
        right_value = right.get(key.column)
        if _is_null_like(left_value) and _is_null_like(right_value):
            continue
        if _is_null_like(left_value):
            return -1 if key.nulls == "first" else 1
        if _is_null_like(right_value):
            return 1 if key.nulls == "first" else -1
        cmp = _compare_sample_values(left_value, right_value)
        if cmp:
            return cmp if key.ascending else -cmp
    return 0


def _compare_sample_values(left: Any, right: Any) -> int:
    try:
        if left < right:
            return -1
        if left > right:
            return 1
        return 0
    except TypeError:
        left_text = repr(left)
        right_text = repr(right)
        if left_text < right_text:
            return -1
        if left_text > right_text:
            return 1
        return 0


def _sort_null_placement_mismatch(
    sort_op: dict[str, Any] | None,
    column: str,
    check_nulls: str,
) -> bool:
    if sort_op is None:
        return False
    try:
        keys = normalize_sort_keys(sort_op)
    except ValueError:
        return False
    return any(key.column == column and key.nulls != check_nulls for key in keys)


def _limit_frontier_score(samples: dict[str, list[Any]], op: dict[str, Any]) -> tuple[float, list[str]]:
    row_count = max((len(values) for values in samples.values()), default=0)
    try:
        limit = int(op.get("n", 0))
    except (TypeError, ValueError):
        return 0.0, []
    buckets: list[str] = []
    if limit == 0:
        buckets.append("limit:zero")
    if row_count and abs(limit - row_count) <= 1:
        buckets.append("limit:row-boundary")
    if row_count and abs(limit - (row_count + 1)) <= 1:
        buckets.append("limit:overflow-boundary")
    if not buckets:
        return 0.0, buckets
    return min(1.0, 0.30 + 0.20 * len(buckets)), buckets


def _offset_frontier_score(samples: dict[str, list[Any]], op: dict[str, Any]) -> tuple[float, list[str]]:
    row_count = max((len(values) for values in samples.values()), default=0)
    try:
        offset = int(op.get("n", 0))
    except (TypeError, ValueError):
        return 0.0, []
    buckets: list[str] = []
    if offset == 0:
        buckets.append("offset:zero")
    if row_count and offset >= max(1, int(row_count * 0.5)):
        buckets.append("offset:large")
    if row_count and abs(offset - row_count) <= 1:
        buckets.append("offset:row-boundary")
    if not buckets:
        return 0.0, buckets
    return min(1.0, 0.30 + 0.18 * len(buckets)), buckets


def _limit_output_samples(samples: dict[str, list[Any]], op: dict[str, Any]) -> dict[str, list[Any]]:
    try:
        limit = max(0, int(op.get("n", 0)))
    except (TypeError, ValueError):
        return samples
    return {name: values[:limit] for name, values in samples.items()}


def _offset_output_samples(samples: dict[str, list[Any]], op: dict[str, Any]) -> dict[str, list[Any]]:
    try:
        offset = max(0, int(op.get("n", 0)))
    except (TypeError, ValueError):
        return samples
    return {name: values[offset:] for name, values in samples.items()}


def _evaluate_mutate_values(values: list[Any], expr: dict[str, Any]) -> list[Any]:
    kind = str(expr.get("kind", ""))
    out: list[Any] = []
    for value in values:
        if value is None:
            out.append(None)
            continue
        try:
            if kind == "add_const":
                out.append(value + expr.get("value", 0))
            elif kind == "arith_const":
                operand = expr.get("value", 0)
                op = expr.get("op")
                if op == "sub":
                    out.append(value - operand)
                elif op == "mul":
                    out.append(value * operand)
                elif op == "div":
                    out.append(value / operand)
                elif op == "mod":
                    out.append(value % operand)
                else:
                    out.append(None)
            elif kind == "cast":
                out.append(float(value))
            elif kind == "string_length":
                out.append(len(value) if isinstance(value, str) else None)
            elif kind == "string_lower":
                out.append(value.lower() if isinstance(value, str) else None)
            else:
                out.append(None)
        except Exception:
            out.append(None)
    return out


def _numeric_values(values: list[Any]) -> list[float]:
    out = []
    for value in values:
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, (int, float)):
            out.append(float(value))
    return out


def _is_null_like(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _has_mixed_case(value: str) -> bool:
    letters = [ch for ch in value if ch.isalpha()]
    return any(ch.islower() for ch in letters) and any(ch.isupper() for ch in letters)


def _predicted_roots(features: set[str]) -> set[str]:
    roots = set()
    if "has:special_float" in features:
        roots.add("nan_inf_semantics")
    if "pattern:join_null_truth_filter" in features:
        roots.add("outer_join_truth_filter")
    if "pattern:boolean_predicate_filter" in features:
        roots.add("boolean_null_filter")
    if "pattern:post_topk_range_filter" in features:
        roots.add("topk_filter_pushdown")
    if "pattern:tuple_absence_filter" in features:
        roots.add("tuple_absence_null_filter")
    if "pattern:running_sum_precision" in features or "op:running_sum" in features:
        roots.add("running_sum_precision")
    if "pattern:sortedness_null_placement" in features or "op:sortedness_check" in features:
        roots.add("sortedness_null_placement")
    if "pattern:simple_case_random_subject" in features or "op:random_case_probe" in features:
        roots.add("simple_case_random_subject")
    if "pattern:group_quantile_key_probe" in features or "op:group_quantile_probe" in features:
        roots.add("group_quantile_key_expression")
    if "pattern:scalar_subquery_double_parentheses" in features or "op:scalar_subquery_probe" in features:
        roots.add("scalar_subquery_double_parentheses")
    if "pattern:window_avg_rows_frame" in features or "op:window_avg_probe" in features:
        roots.add("window_avg_rows_frame")
    if "pattern:struct_distinct_unnest" in features or "op:struct_distinct_probe" in features:
        roots.add("struct_distinct_unnest")
    if "pattern:bit_compare_unequal_length" in features or "op:bit_compare_probe" in features:
        roots.add("bit_compare_unequal_length")
    if "pattern:round_even_float_scale" in features or "op:round_even_probe" in features:
        roots.add("round_even_float_scale")
    if "pattern:series_rtruediv_operand_order" in features or "op:series_rtruediv_probe" in features:
        roots.add("series_rtruediv_operand_order")
    if "pattern:pandas_uint64_isin_precision" in features or "op:uint64_isin_probe" in features:
        roots.add("pandas_uint64_isin_precision")
    if "pattern:duckdb_tuple_anti_null_semantics" in features or "op:tuple_anti_null_probe" in features:
        roots.add("duckdb_tuple_anti_null_semantics")
    if "pattern:pandas_sparse_array_mask_semantics" in features or "op:sparse_mask_probe" in features:
        roots.add("pandas_sparse_array_mask_semantics")
    if "pattern:polars_float_wrap_numerical_semantics" in features or "op:float_wrap_probe" in features:
        roots.add("polars_float_wrap_numerical_semantics")
    if "pattern:pandas_index_bool_result_type" in features or "op:index_bool_probe" in features:
        roots.add("pandas_index_bool_result_type")
    if "pattern:polars_empty_literal_groupby_semantics" in features or "op:empty_literal_groupby_probe" in features:
        roots.add("polars_empty_literal_groupby_semantics")
    if "pattern:pandas_arrow_string_eq_sum_semantics" in features or "op:arrow_string_eq_sum_probe" in features:
        roots.add("pandas_arrow_string_eq_sum_semantics")
    if (
        "pattern:pandas_arrow_timestamp_loc_slice_semantics" in features
        or "op:arrow_timestamp_loc_slice_probe" in features
    ):
        roots.add("pandas_arrow_timestamp_loc_slice_semantics")
    if (
        "pattern:pandas_arrow_timestamp_index_attr_semantics" in features
        or "op:arrow_timestamp_index_attr_probe" in features
    ):
        roots.add("pandas_arrow_timestamp_index_attr_semantics")
    if features & {
        "pattern:null_groupby_topk",
        "pattern:null_agg_topk",
        "pattern:filter_null_agg_topk",
        "pattern:join_null_agg_topk",
        "pattern:join_null_key_topk",
    }:
        roots.add("grouped_topk_null_sort_key")
    if "pattern:empty_filter_groupby" in features:
        roots.add("groupby_aggregation")
    if "op:join" in features:
        roots.add("join_semantics")
    if "op:groupby" in features:
        roots.add("float_group_key_instability" if "pattern:float_group_key" in features else "groupby_aggregation")
    if "op:aggregate" in features:
        roots.add("groupby_aggregation")
    if "op:filter" in features:
        roots.add("filter_predicate")
    if "op:mutate" in features:
        if features & {"expr:string_length", "expr:string_lower"}:
            roots.add("string_expression")
        elif "expr:cast" in features:
            roots.add("type_cast")
        else:
            roots.add("arithmetic_expression")
    if features & {"op:sort", "op:limit", "op:offset"}:
        roots.add("ordering_or_limit")
    if "has:null" in features:
        roots.add("null_semantics")
    return roots


def _has_null_groupby_topk_pattern(op_names: list[str], frontier_buckets: list[str]) -> bool:
    if "groupby:null-key" not in frontier_buckets or "sort:null-order" not in frontier_buckets:
        return False
    try:
        groupby_idx = op_names.index("groupby")
        sort_idx = next(idx for idx, name in enumerate(op_names[groupby_idx + 1 :], start=groupby_idx + 1) if name == "sort")
        next(idx for idx, name in enumerate(op_names[sort_idx + 1 :], start=sort_idx + 1) if name == "limit")
    except (StopIteration, ValueError):
        return False
    return True


def _has_null_agg_topk_pattern(ops: list[dict[str, Any]], frontier_buckets: list[str]) -> bool:
    if "groupby:null-agg-output" not in frontier_buckets or "sort:null-order" not in frontier_buckets:
        return False
    for groupby_idx, op in enumerate(ops):
        if op.get("op") != "groupby":
            continue
        agg_aliases = {str(agg.get("as", "")) for agg in op.get("aggs", []) if agg.get("as")}
        if not agg_aliases:
            continue
        for sort_idx in range(groupby_idx + 1, len(ops)):
            sort_op = ops[sort_idx]
            if sort_op.get("op") != "sort":
                continue
            try:
                sort_columns = {key.column for key in normalize_sort_keys(sort_op)}
            except ValueError:
                sort_columns = set()
            if not (agg_aliases & sort_columns):
                continue
            if any(later.get("op") == "limit" for later in ops[sort_idx + 1 :]):
                return True
    return False


def _has_filter_null_agg_topk_pattern(ops: list[dict[str, Any]], frontier_buckets: list[str]) -> bool:
    if "groupby:null-agg-output" not in frontier_buckets or "sort:null-order" not in frontier_buckets:
        return False
    try:
        filter_idx = next(idx for idx, op in enumerate(ops) if op.get("op") == "filter")
        mutate_idx = next(
            idx for idx, op in enumerate(ops[filter_idx + 1 :], start=filter_idx + 1) if op.get("op") == "mutate"
        )
        select_idx = next(
            idx for idx, op in enumerate(ops[mutate_idx + 1 :], start=mutate_idx + 1) if op.get("op") == "select"
        )
        groupby_idx = next(
            idx for idx, op in enumerate(ops[select_idx + 1 :], start=select_idx + 1) if op.get("op") == "groupby"
        )
        post_group_select_idx = next(
            idx
            for idx, op in enumerate(ops[groupby_idx + 1 :], start=groupby_idx + 1)
            if op.get("op") == "select"
        )
        sort_idx = next(
            idx
            for idx, op in enumerate(ops[post_group_select_idx + 1 :], start=post_group_select_idx + 1)
            if op.get("op") == "sort"
        )
        next(idx for idx, op in enumerate(ops[sort_idx + 1 :], start=sort_idx + 1) if op.get("op") == "limit")
    except StopIteration:
        return False
    return True


def _has_join_null_agg_topk_pattern(ops: list[dict[str, Any]], frontier_buckets: list[str]) -> bool:
    if "groupby:null-agg-output" not in frontier_buckets or "sort:null-order" not in frontier_buckets:
        return False
    join_idx = None
    for idx, op in enumerate(ops):
        if op.get("op") == "join" and op.get("how") == "left":
            join_idx = idx
            break
    if join_idx is None:
        return False
    try:
        groupby_idx = next(idx for idx, op in enumerate(ops[join_idx + 1 :], start=join_idx + 1) if op.get("op") == "groupby")
        sort_idx = next(idx for idx, op in enumerate(ops[groupby_idx + 1 :], start=groupby_idx + 1) if op.get("op") == "sort")
        next(idx for idx, op in enumerate(ops[sort_idx + 1 :], start=sort_idx + 1) if op.get("op") == "limit")
    except StopIteration:
        return False
    return True


def _has_join_null_key_topk_pattern(ops: list[dict[str, Any]], frontier_buckets: list[str]) -> bool:
    if "groupby:null-key" not in frontier_buckets or "sort:null-order" not in frontier_buckets:
        return False
    join_idx = None
    for idx, op in enumerate(ops):
        if op.get("op") == "join" and op.get("how") == "left":
            join_idx = idx
            break
    if join_idx is None:
        return False
    try:
        groupby_idx = next(idx for idx, op in enumerate(ops[join_idx + 1 :], start=join_idx + 1) if op.get("op") == "groupby")
    except StopIteration:
        return False
    group_keys = {str(key) for key in ops[groupby_idx].get("keys", [])}
    if not group_keys:
        return False
    for sort_idx, sort_op in enumerate(ops[groupby_idx + 1 :], start=groupby_idx + 1):
        if sort_op.get("op") != "sort":
            continue
        try:
            sort_columns = {key.column for key in normalize_sort_keys(sort_op)}
        except ValueError:
            sort_columns = set()
        if not (sort_columns & group_keys):
            continue
        if any(later.get("op") == "limit" for later in ops[sort_idx + 1 :]):
            return True
    return False


def _has_wide_offset_topk_pattern(ops: list[dict[str, Any]]) -> bool:
    try:
        sort_idx = next(idx for idx, op in enumerate(ops) if op.get("op") == "sort")
        offset_idx = next(idx for idx, op in enumerate(ops[sort_idx + 1 :], start=sort_idx + 1) if op.get("op") == "offset")
        next(idx for idx, op in enumerate(ops[offset_idx + 1 :], start=offset_idx + 1) if op.get("op") == "limit")
    except StopIteration:
        return False
    return True


def _has_empty_filter_groupby_pattern(ops: list[dict[str, Any]], frontier_buckets: list[str]) -> bool:
    if "filter:empty-output" not in frontier_buckets:
        return False
    try:
        filter_idx = next(idx for idx, op in enumerate(ops) if op.get("op") == "filter")
        groupby_idx = next(idx for idx, op in enumerate(ops[filter_idx + 1 :], start=filter_idx + 1) if op.get("op") == "groupby")
    except StopIteration:
        return False
    return any(op.get("op") in {"select", "sort", "limit"} for op in ops[groupby_idx + 1 :])


def _has_join_filter_groupby_pattern(ops: list[dict[str, Any]]) -> bool:
    try:
        join_idx = next(idx for idx, op in enumerate(ops) if op.get("op") == "join" and op.get("how") == "inner")
        filter_idx = next(idx for idx, op in enumerate(ops[join_idx + 1 :], start=join_idx + 1) if op.get("op") == "filter")
        next(idx for idx, op in enumerate(ops[filter_idx + 1 :], start=filter_idx + 1) if op.get("op") == "groupby")
        sort_idx = next(idx for idx, op in enumerate(ops[filter_idx + 1 :], start=filter_idx + 1) if op.get("op") == "sort")
        next(idx for idx, op in enumerate(ops[sort_idx + 1 :], start=sort_idx + 1) if op.get("op") == "limit")
    except StopIteration:
        return False
    return True


def _has_join_null_truth_filter_pattern(ops: list[dict[str, Any]]) -> bool:
    try:
        join_idx = next(idx for idx, op in enumerate(ops) if op.get("op") == "join" and op.get("how") == "left")
        filter_op = next(
            op
            for op in ops[join_idx + 1 :]
            if op.get("op") == "filter"
        )
    except StopIteration:
        return False
    parsed = parse_filter_comparator(filter_op.get("cmp", ""))
    return parsed is not None and parsed.truth_test in {"is_not_true", "is_not_false", "is_unknown"}


def _has_float_group_key_pattern(ops: list[dict[str, Any]]) -> bool:
    div_columns: set[str] = set()
    for op in ops:
        kind = op.get("op")
        if kind == "mutate":
            expr = op.get("expr", {})
            if expr.get("kind") == "arith_const" and expr.get("op") == "div":
                div_columns.add(str(op.get("column", "")))
        elif kind == "groupby":
            keys = {str(key) for key in op.get("keys", [])}
            return bool(keys & div_columns)
    return False


def _has_ordered_groupby_sort_pattern(ops: list[dict[str, Any]]) -> bool:
    try:
        first_sort_idx = next(idx for idx, op in enumerate(ops) if op.get("op") == "sort")
        groupby_idx = next(
            idx for idx, op in enumerate(ops[first_sort_idx + 1 :], start=first_sort_idx + 1)
            if op.get("op") == "groupby"
        )
        groupby = ops[groupby_idx]
        agg_aliases = {str(agg.get("as", "")) for agg in groupby.get("aggs", []) if agg.get("as")}
        second_sort = next(
            op for op in ops[groupby_idx + 1 :]
            if op.get("op") == "sort"
            and bool(agg_aliases & {key.column for key in normalize_sort_keys(op)})
        )
    except (StopIteration, ValueError):
        return False
    return bool(second_sort)


def _has_topk_resort_pattern(ops: list[dict[str, Any]]) -> bool:
    try:
        first_sort_idx = next(idx for idx, op in enumerate(ops) if op.get("op") == "sort")
        limit_idx = next(
            idx for idx, op in enumerate(ops[first_sort_idx + 1 :], start=first_sort_idx + 1)
            if op.get("op") == "limit"
        )
        next_op = ops[limit_idx + 1]
    except StopIteration:
        return False
    except IndexError:
        return False
    return next_op.get("op") in {"sort", "offset"}


def _has_post_topk_filter_pattern(ops: list[dict[str, Any]]) -> bool:
    for sort_idx, op in enumerate(ops):
        if op.get("op") != "sort":
            continue
        for topk_idx in range(sort_idx + 1, len(ops)):
            if ops[topk_idx].get("op") not in {"limit", "offset"}:
                continue
            if any(later.get("op") == "filter" for later in ops[topk_idx + 1 :]):
                return True
    return False


def _has_running_sum_precision_pattern(ops: list[dict[str, Any]]) -> bool:
    return any(
        op.get("op") == "running_sum" and op.get("input_dtype") == "float32"
        for op in ops
    )


def _has_sortedness_null_placement_pattern(ops: list[dict[str, Any]]) -> bool:
    for idx, op in enumerate(ops):
        if op.get("op") != "sortedness_check":
            continue
        column = str(op.get("column", ""))
        check_nulls = str(op.get("nulls", "last"))
        for previous in reversed(ops[:idx]):
            if previous.get("op") != "sort":
                continue
            if _sort_null_placement_mismatch(previous, column, check_nulls):
                return True
    return False


def _has_join_ordered_agg_topk_pattern(ops: list[dict[str, Any]]) -> bool:
    try:
        join_idx = next(idx for idx, op in enumerate(ops) if op.get("op") == "join")
        first_sort_idx = next(
            idx for idx, op in enumerate(ops[join_idx + 1 :], start=join_idx + 1)
            if op.get("op") == "sort"
        )
        groupby_idx = next(
            idx for idx, op in enumerate(ops[first_sort_idx + 1 :], start=first_sort_idx + 1)
            if op.get("op") == "groupby"
        )
        groupby = ops[groupby_idx]
        agg_aliases = {str(agg.get("as", "")) for agg in groupby.get("aggs", []) if agg.get("as")}
        sort_idx = next(
            idx for idx, op in enumerate(ops[groupby_idx + 1 :], start=groupby_idx + 1)
            if op.get("op") == "sort"
            and bool(agg_aliases & {key.column for key in normalize_sort_keys(op)})
        )
        next(idx for idx, op in enumerate(ops[sort_idx + 1 :], start=sort_idx + 1) if op.get("op") == "limit")
    except (StopIteration, ValueError):
        return False
    return True


def _has_global_null_aggregate_pattern(ops: list[dict[str, Any]], frontier_buckets: list[str]) -> bool:
    if "aggregate:global" not in frontier_buckets:
        return False
    if not {"aggregate:null-output", "aggregate:empty-input"} & set(frontier_buckets):
        return False
    return any(op.get("op") == "aggregate" for op in ops)


def _has_join_null_sort_pattern(ops: list[dict[str, Any]], frontier_buckets: list[str]) -> bool:
    if "sort:null-order" not in frontier_buckets:
        return False
    join_idx = None
    for idx, op in enumerate(ops):
        if op.get("op") == "join" and op.get("how") == "left":
            join_idx = idx
            break
    if join_idx is None:
        return False
    try:
        sort_idx = next(idx for idx, op in enumerate(ops[join_idx + 1 :], start=join_idx + 1) if op.get("op") == "sort")
        next(idx for idx, op in enumerate(ops[sort_idx + 1 :], start=sort_idx + 1) if op.get("op") == "limit")
    except StopIteration:
        return False
    return True
