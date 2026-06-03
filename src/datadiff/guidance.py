from __future__ import annotations

import math
from collections import Counter, OrderedDict, deque
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable

from datadiff.case_features import PROBE_ROOTS, last_probe_root_for_operations
from datadiff.config import DiscoveryBias
from datadiff.dsl import Case, normalize_sort_keys
from datadiff.filtering import evaluate_filter_predicate, parse_filter_comparator
from datadiff.join_keys import join_key_pairs, join_key_value
from datadiff.ordering_semantics import (
    compare_scalar_values,
    is_null_like,
    reorder_samples_by_index,
    sort_sample_indices,
)
from datadiff.operation_combo import combo_semantic_signals, describe_operation_combo
from datadiff.operation_semantics import (
    aggregate_alias,
    aggregate_column,
    aggregate_feature_type,
    aggregate_func,
    aggregate_specs,
    case_else_value,
    case_then_value,
    condition_cmp,
    condition_column,
    condition_value,
    expr_input_domain,
    expr_index,
    expr_kind,
    expr_length,
    expr_lower,
    expr_new,
    expr_needle,
    expr_old,
    expr_other,
    expr_operator,
    expr_part,
    expr_separator,
    expr_source,
    expr_start,
    expr_target_type,
    expr_upper,
    expr_value,
    has_fractional_float_literal,
    groupby_keys,
    join_how,
    join_left_keys,
    join_right_keys,
    normalized_order_by_keys,
    op_ascending,
    op_branches,
    op_column,
    op_columns,
    op_comparator,
    op_input_dtype,
    op_kind,
    op_literal,
    op_n,
    op_nulls,
    op_output_alias,
    op_partition_columns,
    op_quantiles,
    op_right_columns,
    op_rows,
    op_source,
    op_table,
    op_value,
    op_values,
    operation_names,
    sort_null_placement_mismatch,
)
from datadiff.program_patterns import (
    has_float_group_key_pattern,
    has_join_null_truth_filter_pattern,
    has_reverse_division_columns_pattern,
    has_running_sum_precision_pattern,
    has_sortedness_null_placement_pattern,
    program_pattern_features,
)
from datadiff.exploration_objectives import (
    ExplorationObjectiveRule,
    active_exploration_objective_rules,
    derive_exploration_objective_features,
    merge_exploration_objective_rules,
)
from datadiff.semantic_family import derive_semantic_family_features
from datadiff.semantic_signal import (
    alias_expanded_semantic_features,
    canonical_semantic_feature,
    canonical_semantic_prefix,
    feature_matches_prefix_alias,
    semantic_feature_aliases,
    semantic_signal_aliases,
    semantic_signal_feature,
    semantic_signal_feature_bundle,
)
from datadiff.case_policy import canonical_source_issue_key, case_discovery_origin
from datadiff.reward import (
    FindingOutcomeAnalysis,
    analyze_finding_outcomes,
    is_candidate_issue_finding,
    is_rewardable_candidate_issue_finding,
    row_reward_signals,
    feedback_summary_for_case,
)
from datadiff.running import (
    running_sum_partition_columns,
    sort_samples_for_running,
    running_sum_sort_keys,
    stable_running_sum_values,
    stable_running_sum_sample_values,
)
from datadiff.sample_semantics import (
    distinct_output_samples as shared_distinct_output_samples,
    eval_expr_samples as shared_eval_expr_samples,
    filter_samples as shared_filter_samples,
    groupby_output_samples as shared_groupby_output_samples,
    join_samples as shared_join_samples,
    rows_from_samples as shared_rows_from_samples,
    samples_from_rows as shared_samples_from_rows,
)
from datadiff.sortedness import is_sorted_values
from datadiff.tuple_logic import evaluate_tuple_absence
from datadiff.util import unique_preserve_order


def _semantic_signal_feature(signal: str) -> str:
    return semantic_signal_feature(signal)


def _combo_signal_aliases(signal: str) -> set[str]:
    return semantic_signal_aliases(signal)


def _semantic_signal_aliases(*signals: str) -> set[str]:
    return semantic_signal_aliases(*signals)


def _semantic_signal_bundle(*signals: str, extra_features: set[str] | None = None) -> set[str]:
    return semantic_signal_feature_bundle(*signals, extra_features=extra_features)


def _sql_rewrite_semantic_bundle() -> set[str]:
    return _semantic_signal_bundle(
        "sql_distinct_null_topk",
        "left_join_coalesce_membership",
        "case_when_membership_predicate",
        "union_coalesce_distinct_topk",
        "left_join_case_when_membership",
        "left_join_null_predicate_aggregation",
        "coalesce_case_distinct_aggregation",
        "numeric_text_cast_membership_aggregation",
        "multi_key_membership_aggregation",
        "boolean_membership_case_aggregation",
        "left_join_boolean_case_aggregation",
        "boolean_coalesce_case_aggregation",
        "left_join_boolean_coalesce_aggregation",
        "boolean_coalesce_filter_aggregation",
        "left_join_boolean_coalesce_filter_aggregation",
    )


def _template_backed_signal_bundle(
    *signals: str,
    templates: tuple[str, ...] = (),
) -> set[str]:
    return _semantic_signal_bundle(
        *signals,
        extra_features={f"common_api_template:{template}" for template in templates if str(template).strip()},
    )


def _combo_signal_feature(feature: str) -> str:
    return canonical_semantic_feature(feature)


@lru_cache(maxsize=None)
def _feature_aliases(feature: str) -> frozenset[str]:
    return semantic_feature_aliases(feature)


def _alias_expanded_features(features: set[str]) -> set[str]:
    return alias_expanded_semantic_features(features)


@lru_cache(maxsize=None)
def _target_aliases(target: str) -> frozenset[str]:
    aliases = TARGET_ALIASES.get(target, {target})
    expanded: set[str] = set()
    for alias in aliases:
        expanded.update(_feature_aliases(alias))
    return frozenset(expanded)


def _bias_target_aliases(target: str) -> set[str]:
    text = str(target).strip()
    if not text:
        return set()
    aliases = set(_target_aliases(text))
    aliases.add(text)
    if not text.startswith(("semantic_family:", "semantic_signal:")):
        aliases.add(f"semantic_family:{text}")
        aliases.add(_semantic_signal_feature(text))
    return _alias_expanded_features(aliases)


def _canonical_feature(feature: str) -> str:
    return canonical_semantic_feature(feature)


def _canonical_features(features: set[str]) -> set[str]:
    return {_canonical_feature(feature) for feature in features}


def _canonical_feature_prefix_key(prefix: str) -> str:
    return canonical_semantic_prefix(prefix)


def _canonical_counter(counter_like: Counter[str] | dict[str, Any] | None) -> Counter[str]:
    canonical: Counter[str] = Counter()
    for feature, count in (counter_like or {}).items():
        canonical[_canonical_feature(str(feature))] += int(count or 0)
    return canonical


def _canonical_feature_stats(
    stats_by_feature: dict[str, FeatureRewardStats] | dict[str, dict[str, Any]] | None,
) -> dict[str, FeatureRewardStats]:
    canonical: dict[str, FeatureRewardStats] = {}
    for feature, stats in (stats_by_feature or {}).items():
        key = _canonical_feature(str(feature))
        current = canonical.setdefault(key, FeatureRewardStats())
        if isinstance(stats, FeatureRewardStats):
            current.pulls += int(stats.pulls or 0)
            current.total_reward += float(stats.total_reward or 0.0)
        else:
            payload = stats or {}
            current.pulls += int(payload.get("pulls", 0) or 0)
            current.total_reward += float(payload.get("total_reward", 0.0) or 0.0)
    return canonical


def _canonical_prefix_stats(
    stats_by_prefix: dict[str, FeatureRewardStats] | dict[str, dict[str, Any]] | None,
) -> dict[str, FeatureRewardStats]:
    canonical: dict[str, FeatureRewardStats] = {}
    for prefix, stats in (stats_by_prefix or {}).items():
        key = _canonical_feature_prefix_key(str(prefix))
        current = canonical.setdefault(key, FeatureRewardStats())
        if isinstance(stats, FeatureRewardStats):
            current.pulls += int(stats.pulls or 0)
            current.total_reward += float(stats.total_reward or 0.0)
        else:
            payload = stats or {}
            current.pulls += int(payload.get("pulls", 0) or 0)
            current.total_reward += float(payload.get("total_reward", 0.0) or 0.0)
    return canonical


def _feature_has_prefix(feature: str, prefix: str) -> bool:
    return feature_matches_prefix_alias(feature, prefix)


def _feature_prefix_token(feature_or_prefix: str) -> str:
    text = str(feature_or_prefix).strip()
    if ":" not in text:
        return ""
    return f"{text.split(':', 1)[0]}:"


def _feature_prefix_root(feature_or_prefix: str) -> str:
    canonical = _canonical_feature(str(feature_or_prefix).strip())
    return canonical.split(":", 1)[0] if ":" in canonical else canonical


TARGET_ALIASES: dict[str, set[str]] = {
    "filter": {"op:filter"},
    "truth_filter": {"filter:truth-test"},
    "groupby": {"op:groupby"},
    "groupby_sorted_input": {"pattern:groupby_sorted_input", "groupby:sorted-input"},
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
    "aggregation": {
        "op:groupby",
        "op:aggregate",
        "agg:sum",
        "agg:mean",
        "agg:min",
        "agg:max",
        "agg:count",
        "agg:nunique",
        "agg:any",
        "agg:all",
    },
    "numeric_mean_aggregate": {"agg:mean"},
    "global_aggregation": {"op:aggregate"},
    "null_groupby_topk": {"pattern:null_groupby_topk"},
    "null_agg_topk": {"pattern:null_agg_topk"},
    "filter_null_agg_topk": {"pattern:filter_null_agg_topk"},
    "join_null_agg_topk": {"pattern:join_null_agg_topk"},
    "join_null_key_topk": {"pattern:join_null_key_topk"},
    "distinct_null_topk": {"pattern:distinct_null_topk"},
    "wide_offset_topk": {"pattern:wide_offset_topk"},
    "empty_filter_groupby": {"pattern:empty_filter_groupby"},
    "empty_filter_aggregate": {"pattern:empty_filter_aggregate"},
    "empty_global_aggregation": {"pattern:empty_filter_aggregate"},
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
    "bool_null_groupby_agg": {"pattern:bool_null_groupby_agg"},
    "boolean_aggregation": {"agg:boolean", "agg:min:bool", "agg:max:bool", "agg:any:bool", "agg:all:bool"},
    "bool_any_all": {"agg:any:bool", "agg:all:bool"},
    "large_int_filter_groupby": {"pattern:large_int_filter_groupby"},
    "large_integer": {"int:large-magnitude"},
    "unique_count": {"agg:nunique"},
    "nunique": {"agg:nunique"},
    "count_distinct": {"agg:nunique"},
    "distinct_count_aggregation": _combo_signal_aliases("distinct_count_aggregation") | {"agg:nunique"},
    "set_membership_filter": {"pattern:set_membership_filter"},
    "set_membership": {"filter:set-membership"},
    "negative_set_membership_filter": {"filter:negative-set-membership"},
    "not_in_set_filter": {"filter:negative-set-membership"},
    "string_pattern_filter": {"pattern:string_pattern_filter"},
    "string_contains_filter": {"pattern:string_pattern_filter", "filter:string-contains"},
    "string_contains": {"filter:string-contains", "expr:string_contains", "string:contains"},
    "string_starts_with": {"filter:string-starts-with", "expr:string_starts_with", "string:starts-with"},
    "string_ends_with": {"filter:string-ends-with", "expr:string_ends_with", "string:ends-with"},
    "string_length": {"expr:string_length", "string:length"},
    "string_null_if_empty": {"expr:string_null_if_empty", "string:null-if-empty", "null:empty-string"},
    "string_split_part": {"expr:string_split_part", "string:split-first"},
    "string_basename": {"expr:string_basename", "path:basename"},
    "path_basename": {"expr:string_basename", "path:basename"},
    "pyarrow_groupby_filter_cast_membership": {
        "pattern:pyarrow_groupby_filter_cast_membership",
        "membership:int-column-fractional-literal",
    },
    "null_predicate_filter": {"pattern:null_predicate_filter"},
    "null_predicate": {"filter:null-predicate"},
    "boolean_predicate_filter": {"pattern:boolean_predicate_filter"},
    "boolean_predicate": {"filter:boolean-predicate"},
    "post_topk_range_filter": {"pattern:post_topk_range_filter"},
    "range_filter": {"filter:range-closed"},
    "tuple_absence_filter": {"pattern:tuple_absence_filter"},
    "tuple_absence": {"filter:tuple-absence"},
    "row_value_absence_filter": {"pattern:row_value_absence_filter"},
    "row_value_absence": {"filter:tuple-absence", "nulls:ternary-membership"},
    "running_sum_precision": {"pattern:running_sum_precision"},
    "running_sum": {"op:running_sum"},
    "partitioned_running_sum": {"pattern:partitioned_running_sum"},
    "running_sum_partitioned": {"running:partitioned"},
    "path_basename_keyed_pick": {"pattern:path_basename_keyed_pick"},
    "path_projection": {"expr:string_basename", "path:basename"},
    "keyed_row_pick": {"op:row_number_filter"},
    "row_number_filter": {"op:row_number_filter", "row_pick:keyed"},
    "top_n_per_group": {"op:row_number_filter"} | _combo_signal_aliases("topn_per_group"),
    "dedup_latest_per_id": {"op:row_number_filter", "row_pick:keyed"},
    "filtered_global_aggregation": _combo_signal_aliases("filtered_global_aggregation"),
    "left_join_fill_null_aggregation": _combo_signal_aliases("left_join_fill_null_aggregation"),
    "left_join_null_predicate_aggregation": _template_backed_signal_bundle(
        "left_join_null_predicate_aggregation",
        "left_join_null_predicate_filter",
        templates=("sql_left_join_null_predicate_aggregate",),
    ),
    "left_join_coalesce_membership": _template_backed_signal_bundle(
        "left_join_coalesce_membership",
        "join_coalesce_membership",
        templates=("sql_left_join_coalesce_membership",),
    ),
    "join_coalesce_membership": _semantic_signal_aliases(
        "join_coalesce_membership",
        "join_coalesce_membership_aggregation",
        "join_coalesce_membership_topk",
    ),
    "cleaned_join_key": _semantic_signal_aliases("cleaned_join_key"),
    "normalized_string_join": _semantic_signal_aliases("normalized_string_join_key"),
    "normalized_string_join_key": _semantic_signal_aliases("normalized_string_join_key"),
    "chained_string_normalized_join_key": _semantic_signal_aliases("chained_string_normalized_join_key"),
    "normalized_string_membership": _semantic_signal_aliases("normalized_string_membership_key"),
    "normalized_string_membership_key": _semantic_signal_aliases("normalized_string_membership_key"),
    "chained_string_normalized_membership_key": _semantic_signal_aliases("chained_string_normalized_membership_key"),
    "sql_distinct_null_topk": _template_backed_signal_bundle(
        "sql_distinct_null_topk",
        templates=("sql_distinct_null_coalesce_topk",),
    ),
    "coalesced_distinct_topk": _semantic_signal_aliases("coalesced_distinct_topk"),
    "case_when_membership": _template_backed_signal_bundle(
        "case_when_membership_predicate",
        "case_when_membership_aggregation",
        "case_when_membership_topk",
        templates=(
            "sql_case_membership_distinct_topk",
            "normalized_string_case_when_groupby",
            "normalized_string_case_when_topk",
        ),
    ),
    "string_pattern_case_when": _template_backed_signal_bundle(
        "string_pattern_case_when",
        "string_pattern_case_when_aggregation",
        "string_pattern_case_when_topk",
        templates=("string_pattern_case_when_groupby", "string_pattern_case_when_topk"),
    ),
    "union_coalesce_distinct_topk": _template_backed_signal_bundle(
        "union_coalesce_distinct_topk",
        templates=("sql_union_coalesce_distinct_topk",),
    ),
    "left_join_case_membership": _template_backed_signal_bundle(
        "left_join_case_when_membership",
        "join_case_when_membership",
        templates=("sql_left_join_case_membership_groupby",),
    ),
    "coalesce_case_distinct_aggregation": _template_backed_signal_bundle(
        "coalesce_case_distinct_aggregation",
        templates=("sql_coalesce_case_distinct_groupby",),
    ),
    "numeric_text_cast_membership": _template_backed_signal_bundle(
        "numeric_text_cast_membership_aggregation",
        "type_cast_membership_aggregation",
        templates=(
            "sql_numeric_text_cast_membership_groupby",
            "sql_numeric_text_cast_bool_antijoin_groupby",
        ),
    ),
    "multi_key_membership_aggregation": _template_backed_signal_bundle(
        "multi_key_membership_aggregation",
        "multi_key_semi_anti_join",
        templates=("sql_multi_key_semijoin_case_groupby", "sql_multi_key_antijoin_case_groupby"),
    ),
    "boolean_membership_case_aggregation": _template_backed_signal_bundle(
        "boolean_membership_case_aggregation",
        "boolean_case_when_predicate",
        templates=(
            "sql_bool_membership_case_aggregate",
            "sql_bool_antijoin_case_aggregate",
            "sql_numeric_text_cast_bool_antijoin_groupby",
            "sql_multi_key_semijoin_case_groupby",
            "sql_multi_key_antijoin_case_groupby",
        ),
    ),
    "left_join_boolean_case_aggregation": _template_backed_signal_bundle(
        "left_join_boolean_case_aggregation",
        "boolean_case_when_aggregation",
        templates=("sql_left_join_bool_case_groupby",),
    ),
    "boolean_coalesce_case_aggregation": _template_backed_signal_bundle(
        "boolean_coalesce_case_aggregation",
        "boolean_case_when_predicate",
        templates=("sql_left_join_bool_coalesce_case_groupby",),
    ),
    "left_join_boolean_coalesce_aggregation": _template_backed_signal_bundle(
        "left_join_boolean_coalesce_aggregation",
        "boolean_coalesce_case_aggregation",
        templates=("sql_left_join_bool_coalesce_case_groupby",),
    ),
    "boolean_coalesce_filter_aggregation": _template_backed_signal_bundle(
        "boolean_coalesce_filter_aggregation",
        templates=("sql_left_join_bool_coalesce_filter_groupby",),
    ),
    "left_join_boolean_coalesce_filter_aggregation": _template_backed_signal_bundle(
        "left_join_boolean_coalesce_filter_aggregation",
        "boolean_coalesce_filter_aggregation",
        templates=("sql_left_join_bool_coalesce_filter_groupby",),
    ),
    "embedded_sql_rewrite": _sql_rewrite_semantic_bundle(),
    "duckdb_sql_rewrite": _sql_rewrite_semantic_bundle(),
    "sqlite_sql_rewrite": _sql_rewrite_semantic_bundle(),
    "having_filter_after_aggregation": _semantic_signal_aliases("having_filter_after_aggregation"),
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
    "duckdb_float_literal_precision": {"pattern:duckdb_float_literal_precision"},
    "float_literal_precision_probe": {"op:float_literal_precision_probe"},
    "float_literal_precision": {"duckdb:float-literal-precision", "float:literal-cast-consistency"},
    "polars_timestamp_precision_filter": {"pattern:polars_timestamp_precision_filter"},
    "timestamp_precision_filter_probe": {"op:timestamp_precision_filter_probe"},
    "timestamp_precision_filter": {"polars:timestamp-precision-filter", "timestamp:precision-filter"},
    "series_rtruediv_operand_order": {"pattern:series_rtruediv_operand_order"},
    "series_rtruediv_probe": {"op:series_rtruediv_probe"},
    "polars_reverse_division_columns": {"pattern:polars_reverse_division_columns"},
    "reverse_division": {
        "series:reverse-division",
        "expr:reverse_division_columns",
        "arithmetic:operand-order",
    },
    "pandas_uint64_isin_precision": {"pattern:pandas_uint64_isin_precision"},
    "uint64_isin_probe": {"op:uint64_isin_probe"},
    "unsigned_membership": {"pandas:uint64-isin", "membership:unsigned-precision"},
    "duckdb_tuple_anti_null_semantics": {"pattern:duckdb_tuple_anti_null_semantics"},
    "tuple_anti_null_probe": {"op:tuple_anti_null_probe"},
    "tuple_null_membership": {"duckdb:tuple-anti-null", "nulls:ternary-membership"},
    "datafusion_setop_all_duplicate_count": {"pattern:datafusion_setop_all_duplicate_count"},
    "setop_all_duplicate_probe": {"op:setop_all_duplicate_probe"},
    "setop_all_duplicates": {"sql:setop-all", "setop:duplicate-count"},
    "duckdb_json_predicate_order_semantics": {"pattern:duckdb_json_predicate_order_semantics"},
    "json_predicate_order_probe": {"op:json_predicate_order_probe"},
    "json_predicate_order": {"duckdb:json-predicate-order", "json:predicate-reorder"},
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
    "pandas_eval_inplace_aliasing_semantics": {"pattern:pandas_eval_inplace_aliasing_semantics"},
    "eval_inplace_alias_probe": {"op:eval_inplace_alias_probe"},
    "eval_inplace_aliasing": {"pandas:eval-inplace-alias", "copy:on-write-alias"},
    "pandas_bool_reduction_skipna_semantics": {"pattern:pandas_bool_reduction_skipna_semantics"},
    "bool_reduction_skipna_probe": {"op:bool_reduction_skipna_probe"},
    "bool_reduction_skipna": {"pandas:bool-reduction-skipna", "nullable-bool:reduction"},
    "pyarrow_dataset_isin_all_match_semantics": {"pattern:pyarrow_dataset_isin_all_match_semantics"},
    "dataset_isin_all_match_probe": {"op:dataset_isin_all_match_probe"},
    "dataset_membership_filter": {"pyarrow:dataset-isin-all-match", "dataset:membership-filter"},
    "pyarrow_run_end_null_compute_semantics": {"pattern:pyarrow_run_end_null_compute_semantics"},
    "run_end_null_compute_probe": {"op:run_end_null_compute_probe"},
    "run_end_null_compute": {"pyarrow:run-end-null-compute", "run_end:null-compute"},
    "pyarrow_large_string_partition_schema_semantics": {"pattern:pyarrow_large_string_partition_schema_semantics"},
    "large_string_partition_probe": {"op:large_string_partition_probe"},
    "large_string_partition": {"pyarrow:large-string-partition", "dataset:partition-schema"},
    "pyarrow_hash_pivot_wider_order_semantics": {"pattern:pyarrow_hash_pivot_wider_order_semantics"},
    "hash_pivot_wider_probe": {"op:hash_pivot_wider_probe"},
    "hash_pivot_wider": {"pyarrow:hash-pivot-wider", "pivot:wider-order"},
    "pyarrow_list_flatten_parent_indices_semantics": {"pattern:pyarrow_list_flatten_parent_indices_semantics"},
    "list_flatten_parent_indices_probe": {"op:list_flatten_parent_indices_probe"},
    "list_layout": {"pyarrow:list-flatten-parent-indices", "arrow:list-layout"},
    "polars_rolling_mean_by_null_count_semantics": {"pattern:polars_rolling_mean_by_null_count_semantics"},
    "rolling_mean_by_null_count_probe": {"op:rolling_mean_by_null_count_probe"},
    "rolling_temporal_nulls": {"polars:rolling-mean-by-null-count", "rolling:temporal-min-samples"},
    "csv_long_numeric_roundtrip": {"pattern:csv_long_numeric_roundtrip"},
    "csv_long_numeric_roundtrip_probe": {"op:csv_long_numeric_roundtrip_probe"},
    "csv_numeric_inference": {"csv:long-numeric-roundtrip", "csv:numeric-inference", "numeric:long-identifier"},
    "join": {"op:join", "tables:multi"},
    "multi_key_join": {"join:multi-key"} | _semantic_signal_aliases("multi_key_join"),
    "multi_key_groupby": {"groupby:multi-key"} | _semantic_signal_aliases("multi_key_groupby"),
    "common_api_workflow": {"pattern:common_api_workflow", "generator_profile:common_api_workflow"},
    "daily_api": {"pattern:common_api_workflow", "combo_frequency:high"},
    "input_materialization": {"pattern:input_materialization_boundary", "materialization:input"},
    "materialization_boundary": {"pattern:input_materialization_boundary", "materialization:input"},
    "filter_input_materialization": {"pattern:filter_input_materialization", "materialization:filter"},
    "cleanup_input_materialization": {
        "pattern:cleanup_input_materialization",
        "materialization:leading-cleanup",
    },
    "drop_nulls_input_materialization": {
        "pattern:drop_nulls_input_materialization",
        "materialization:drop_nulls",
    },
    "fill_null_input_materialization": {
        "pattern:fill_null_input_materialization",
        "materialization:fill_null",
    },
    "distinct_input_materialization": {
        "pattern:distinct_input_materialization",
        "materialization:distinct",
    },
    "input_partition_union": {"pattern:input_partition_union_all", "materialization:input-partition"},
    "union_all": {"op:union_all", "table:row-append"},
    "concat": {"op:union_all", "table:row-append"},
    "drop_nulls": {"op:drop_nulls", "null:drop"},
    "dropna": {"op:drop_nulls", "null:drop"},
    "semi_join": {"op:semi_join", "join:existence", "membership:semi_join"},
    "anti_join": {"op:anti_join", "join:existence", "membership:anti_join"},
    "semi_anti_join_rewrite": {
        "pattern:semi_anti_join_rewrite",
        "join:existence",
    }
    | _semantic_signal_aliases("multi_key_semi_anti_join"),
    "membership_join": {"op:semi_join", "join:existence", "membership:semi_join"},
    "exclusion_join": {"op:anti_join", "join:existence", "membership:anti_join"},
    "distinct": {"op:distinct", "distinct:deduplicate"},
    "duplicate_elimination": {"op:distinct", "distinct:deduplicate"},
    "fill_null": {"op:fill_null", "null:fill"},
    "coalesce": {"op:coalesce", "null:coalesce"},
    "column_coalesce": {"op:coalesce", "null:coalesce", "coalesce:columns"},
    "case_when": {"op:case_when", "conditional:case_when"},
    "conditional_expression": {"op:case_when", "conditional:case_when"},
    "common_workflow": {"combo_frequency:high"},
    "operation_combo": {"combo_frequency:high", "combo_frequency:medium"},
    "topk": _semantic_signal_aliases("topk_ordering", "grouped_topk", "topk_filter_pushdown"),
    "string_lower": {"expr:string_lower", "string:lower"},
    "string_upper": {"expr:string_upper", "string:upper"},
    "string_case": {"expr:string_lower", "expr:string_upper", "string:lower", "string:upper"},
    "unicode_case_mapping": {"expr:string_lower", "expr:string_upper", "has:unicode_string"},
    "date_part": {"expr:date_part", "date:part"},
    "date_extraction": {"expr:date_part", "date:part"},
    "expressions": {"expr:add_const", "expr:arith_const", "expr:abs", "expr:clip", "expr:bool_not", "expr:string_length", "expr:string_lower", "expr:string_upper", "expr:string_strip", "expr:string_null_if_empty", "expr:string_replace", "expr:string_slice", "expr:string_split_part", "expr:string_basename", "expr:string_concat", "expr:string_contains", "expr:string_starts_with", "expr:string_ends_with", "expr:date_part", "expr:cast"},
    "casts": {"expr:cast"},
}

PATTERN_TARGET_WEIGHT = 8.0
GENERIC_COMPANION_TARGET_WEIGHT = 0.25
TEMPLATE_TARGET_BONUS = 3.0
RESOLVED_SEMANTIC_BOUNDARY_TARGETS = {
    "duckdb_float_literal_precision",
    "float_literal_precision",
    "float_literal_precision_probe",
    "unicode_case_mapping",
}
RESOLVED_SEMANTIC_BOUNDARY_ROOTS = {
    "arithmetic_expression",
    "nan_inf_semantics",
    "unicode_case_mapping",
}


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


def _case_source_issue(case: Case) -> str:
    metadata = case.metadata if isinstance(case.metadata, dict) else {}
    return str(metadata.get("source_issue") or metadata.get("source_issue_alt") or "").strip()


def _source_issue_key(value: Any) -> str:
    return canonical_source_issue_key(value)


def derive_case_features(
    case: Case,
    *,
    operation_combo: dict[str, Any] | None = None,
    frontier_buckets: list[str] | None = None,
    exploration_objective_rules: list[ExplorationObjectiveRule] | tuple[ExplorationObjectiveRule, ...] | None = None,
) -> set[str]:
    features: set[str] = set()
    table = case.tables[0]
    features.add("tables:multi" if len(case.tables) > 1 else "tables:single")
    generator_profile = str(case.metadata.get("generator_profile", "")).strip()
    if generator_profile:
        features.add(f"generator_profile:{generator_profile}")
    mixed_generator_profile = str(case.metadata.get("mixed_generator_profile", "")).strip()
    if mixed_generator_profile:
        features.add(f"mixed_generator_profile:{mixed_generator_profile}")
    candidate_source = str(case.metadata.get("candidate_source", "")).strip()
    if candidate_source:
        features.add(f"source:{candidate_source}")
    seed_lineage = case.metadata.get("seed_lineage", {})
    if isinstance(seed_lineage, dict):
        depth = int(seed_lineage.get("depth", 0) or 0)
        if depth > 0:
            features.add("source:feedback_mutation")
            features.add(_bucket("mutation_depth", depth, [(1, "one"), (3, "shallow")], "deep"))
    mutation = case.metadata.get("mutation", {})
    if isinstance(mutation, dict):
        operator_name = str(mutation.get("operator", "")).strip()
        if operator_name and operator_name != "generated":
            features.add(f"mutation_op:{operator_name}")
    features.update(_quality_archive_context_features(case.metadata.get("quality_archive_context", {})))
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
                if abs(value) > 2**53:
                    features.add("int:large-magnitude")
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
    table_by_name = {source_table.name: source_table for source_table in case.tables}
    available_types = {column.name: column.type for column in table.columns}
    has_string_count_groupby = False
    has_unique_count_groupby = False
    has_bool_groupby_agg = False
    has_exact_groupby_agg = False
    has_set_membership_filter = False
    has_null_predicate_filter = False
    has_boolean_predicate_filter = False
    has_range_filter = False
    has_string_contains_filter = False
    has_tuple_absence_filter = False
    has_union_all = False
    has_drop_nulls = False
    has_semi_join = False
    has_anti_join = False
    has_distinct = False
    has_fill_null = False
    has_coalesce = False
    has_case_when = False
    has_running_sum_precision = False
    has_partitioned_running_sum = False
    has_path_basename_keyed_pick = False
    has_string_basename_expr = False
    has_row_number_filter = False
    has_sortedness_check = False
    has_random_case_probe = False
    has_group_quantile_probe = False
    has_scalar_subquery_probe = False
    has_window_avg_probe = False
    has_struct_distinct_probe = False
    has_bit_compare_probe = False
    has_round_even_probe = False
    has_float_literal_precision_probe = False
    has_timestamp_precision_filter_probe = False
    has_series_rtruediv_probe = False
    has_uint64_isin_probe = False
    has_tuple_anti_null_probe = False
    has_setop_all_duplicate_probe = False
    has_json_predicate_order_probe = False
    has_sparse_mask_probe = False
    has_float_wrap_probe = False
    has_index_bool_probe = False
    has_empty_literal_groupby_probe = False
    has_arrow_string_eq_sum_probe = False
    has_arrow_timestamp_loc_slice_probe = False
    has_arrow_timestamp_index_attr_probe = False
    has_eval_inplace_alias_probe = False
    has_bool_reduction_skipna_probe = False
    has_dataset_isin_all_match_probe = False
    has_run_end_null_compute_probe = False
    has_large_string_partition_probe = False
    has_hash_pivot_wider_probe = False
    has_list_flatten_parent_indices_probe = False
    has_rolling_mean_by_null_count_probe = False
    has_csv_long_numeric_roundtrip_probe = False
    for op in case.program.operations:
        kind = op_kind(op, "unknown")
        op_names.append(kind)
        features.add(f"op:{kind}")
        if kind == "filter":
            cmp = op_comparator(op, "unknown")
            column = op_column(op, "unknown")
            features.add(f"cmp:{cmp}")
            features.add(f"filter_type:{available_types.get(column, 'derived')}")
            parsed = parse_filter_comparator(cmp)
            if parsed is not None and parsed.base in {"in_set", "not_in_set"}:
                features.add("filter:set-membership")
                if parsed.base == "not_in_set":
                    features.add("filter:negative-set-membership")
                has_set_membership_filter = True
                if has_fractional_float_literal(op_value(op)):
                    features.add("membership:fractional-literal")
                    if available_types.get(column) == "int":
                        features.add("membership:int-column-fractional-literal")
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
            if parsed is not None and parsed.base in {"str_contains", "str_starts_with", "str_ends_with"}:
                features.add("filter:string-pattern")
                features.add(f"filter:{parsed.base.replace('str_', 'string-').replace('_', '-')}")
                has_string_contains_filter = True
            if parsed is not None and parsed.truth_test is not None:
                features.add("filter:truth-test")
                features.add(f"filter:truth:{parsed.truth_test}")
        elif kind == "tuple_absence_filter":
            features.add("filter:tuple-absence")
            has_tuple_absence_filter = True
        elif kind == "union_all":
            features.add("table:row-append")
            features.add("union_all:append")
            has_union_all = True
        elif kind == "drop_nulls":
            columns = op_columns(op)
            features.add("null:drop")
            features.add("drop_nulls:subset")
            features.add(_bucket("drop_nulls_columns", len(columns), [(1, "one"), (2, "two")], "many"))
            has_drop_nulls = True
        elif kind in {"semi_join", "anti_join"}:
            features.add("join:existence")
            features.add("membership:semi_join" if kind == "semi_join" else "membership:anti_join")
            features.add(f"membership:left_key:{','.join(join_left_keys(op)) or 'unknown'}")
            features.add(f"membership:right_key:{','.join(join_right_keys(op)) or 'unknown'}")
            if kind == "semi_join":
                has_semi_join = True
            else:
                has_anti_join = True
        elif kind == "distinct":
            columns = op_columns(op)
            features.add("distinct:deduplicate")
            features.add(_bucket("distinct_columns", len(columns), [(1, "one"), (2, "two")], "many"))
            has_distinct = True
        elif kind == "fill_null":
            column = op_column(op)
            value = op_value(op)
            column_type = available_types.get(column, "derived")
            features.add("null:fill")
            features.add(f"fill_null_type:{column_type}")
            if value is False:
                features.add("fill_null:false")
            elif value == "":
                features.add("fill_null:empty-string")
            elif value == 0:
                features.add("fill_null:zero")
            has_fill_null = True
        elif kind == "coalesce":
            columns = op_columns(op)
            alias = op_output_alias(op)
            features.add("null:coalesce")
            features.add("coalesce:columns")
            features.add(_bucket("coalesce_columns", len(columns), [(2, "two"), (3, "three")], "many"))
            if columns:
                features.add(f"coalesce_type:{available_types.get(columns[0], 'derived')}")
            if alias in available_types:
                features.add("coalesce:overwrite")
            if "fallback" in op:
                features.add("coalesce:fallback")
            has_coalesce = True
        elif kind == "case_when":
            column = condition_column(op)
            cmp = condition_cmp(op, "unknown")
            alias = op_output_alias(op, "derived")
            output_type = _literal_feature_type(case_then_value(op), case_else_value(op))
            features.add("conditional:case_when")
            features.add(f"case_when_type:{available_types.get(column, 'derived')}")
            features.add(f"case_when_cmp:{cmp}")
            features.add(f"case_when_output:{output_type}")
            available_types[alias] = output_type
            has_case_when = True
        elif kind == "row_number_filter":
            partition_count = len(op_partition_columns(op))
            order_count = len(normalized_order_by_keys(op))
            features.add("row_pick:keyed")
            features.add(f"row_pick:cmp:{op_comparator(op, 'unknown')}")
            features.add(_bucket("row_pick_partition_count", partition_count, [(0, "none"), (1, "one")], "many"))
            features.add(_bucket("row_pick_order_count", order_count, [(1, "one"), (2, "two")], "many"))
            has_row_number_filter = True
            if has_string_basename_expr:
                has_path_basename_keyed_pick = True
        elif kind == "running_sum":
            source = op_source(op)
            input_dtype = op_input_dtype(op, "float64")
            features.add(f"running:{input_dtype}")
            features.add(f"running_source_type:{available_types.get(source, 'derived')}")
            if op_partition_columns(op):
                features.add("running:partitioned")
                has_partitioned_running_sum = True
            if input_dtype == "float32":
                has_running_sum_precision = True
            available_types[op_column(op, "derived")] = "float"
        elif kind == "sortedness_check":
            column = op_column(op)
            nulls = op_nulls(op, "last")
            features.add(f"sortedness:nulls:{nulls}")
            features.add(f"sortedness:{'asc' if op_ascending(op, True) else 'desc'}")
            features.add(f"sortedness_source_type:{available_types.get(column, 'derived')}")
            available_types = {op_output_alias(op, "derived"): "bool"}
            has_sortedness_check = True
        elif kind == "random_case_probe":
            features.add("case_expr:simple")
            features.add("case_expr:random-subject")
            features.add(_bucket("case_probe_rows", op_rows(op, 0), [(1000, "small"), (10000, "medium")], "large"))
            available_types = {op_output_alias(op, "derived"): "bool"}
            has_random_case_probe = True
        elif kind == "group_quantile_probe":
            features.add("quantile:dynamic-key")
            features.add(_bucket("quantile_probe_values", len(op_values(op)), [(2, "tiny"), (4, "small")], "medium"))
            available_types = {op_output_alias(op, "derived"): "bool"}
            has_group_quantile_probe = True
        elif kind == "scalar_subquery_probe":
            features.add("subquery:correlated-scalar")
            features.add("subquery:nested-aggregate")
            available_types = {op_output_alias(op, "derived"): "bool"}
            has_scalar_subquery_probe = True
        elif kind == "window_avg_probe":
            features.add("window:rows-frame")
            features.add("window:avg")
            available_types = {op_output_alias(op, "derived"): "bool"}
            has_window_avg_probe = True
        elif kind == "struct_distinct_probe":
            features.add("struct:unnest")
            features.add("struct:distinct")
            available_types = {op_output_alias(op, "derived"): "bool"}
            has_struct_distinct_probe = True
        elif kind == "bit_compare_probe":
            features.add("bit:unequal-length")
            features.add("comparison:bit-order")
            available_types = {op_output_alias(op, "derived"): "bool"}
            has_bit_compare_probe = True
        elif kind == "round_even_probe":
            features.add("numeric:round-even")
            features.add("float:decimal-scale")
            available_types = {op_output_alias(op, "derived"): "bool"}
            has_round_even_probe = True
        elif kind == "float_literal_precision_probe":
            features.add("duckdb:float-literal-precision")
            features.add("float:literal-cast-consistency")
            features.add("float:decimal-literal")
            available_types = {op_output_alias(op, "derived"): "bool"}
            has_float_literal_precision_probe = True
        elif kind == "timestamp_precision_filter_probe":
            features.add("polars:timestamp-precision-filter")
            features.add("timestamp:precision-filter")
            features.add("timestamp:unit-cast")
            available_types = {op_output_alias(op, "derived"): "bool"}
            has_timestamp_precision_filter_probe = True
        elif kind == "series_rtruediv_probe":
            features.add("series:reverse-division")
            features.add("arithmetic:operand-order")
            available_types = {op_output_alias(op, "derived"): "bool"}
            has_series_rtruediv_probe = True
        elif kind == "uint64_isin_probe":
            features.add("pandas:uint64-isin")
            features.add("membership:unsigned-precision")
            available_types = {op_output_alias(op, "derived"): "bool"}
            has_uint64_isin_probe = True
        elif kind == "tuple_anti_null_probe":
            features.add("duckdb:tuple-anti-null")
            features.add("nulls:ternary-membership")
            available_types = {op_output_alias(op, "derived"): "bool"}
            has_tuple_anti_null_probe = True
        elif kind == "setop_all_duplicate_probe":
            features.add("datafusion:setop-all-duplicate-count")
            features.add("sql:setop-all")
            features.add("setop:except-all")
            features.add("setop:intersect-all")
            features.add("setop:duplicate-count")
            available_types = {op_output_alias(op, "derived"): "bool"}
            has_setop_all_duplicate_probe = True
        elif kind == "json_predicate_order_probe":
            features.add("duckdb:json-predicate-order")
            features.add("json:predicate-reorder")
            available_types = {op_output_alias(op, "derived"): "bool"}
            has_json_predicate_order_probe = True
        elif kind == "sparse_mask_probe":
            features.add("pandas:sparse-mask")
            features.add("mask:sparse-array")
            available_types = {op_output_alias(op, "derived"): "bool"}
            has_sparse_mask_probe = True
        elif kind == "float_wrap_probe":
            features.add("polars:wrap-numerical")
            features.add("cast:float-overflow")
            available_types = {op_output_alias(op, "derived"): "bool"}
            has_float_wrap_probe = True
        elif kind == "index_bool_probe":
            features.add("pandas:index-bool")
            features.add("api:result-type")
            available_types = {op_output_alias(op, "derived"): "bool"}
            has_index_bool_probe = True
        elif kind == "empty_literal_groupby_probe":
            features.add("polars:empty-literal-groupby")
            features.add("groupby:empty-literal")
            available_types = {op_output_alias(op, "derived"): "bool"}
            has_empty_literal_groupby_probe = True
        elif kind == "arrow_string_eq_sum_probe":
            features.add("pandas:arrow-string-eq-sum")
            features.add("arrow:string-bool-reduction")
            available_types = {op_output_alias(op, "derived"): "bool"}
            has_arrow_string_eq_sum_probe = True
        elif kind == "arrow_timestamp_loc_slice_probe":
            features.add("pandas:arrow-timestamp-loc-slice")
            features.add("arrow:timestamp-index-slice")
            available_types = {op_output_alias(op, "derived"): "bool"}
            has_arrow_timestamp_loc_slice_probe = True
        elif kind == "arrow_timestamp_index_attr_probe":
            features.add("pandas:arrow-timestamp-index-attr")
            features.add("arrow:timestamp-index-attribute")
            available_types = {op_output_alias(op, "derived"): "bool"}
            has_arrow_timestamp_index_attr_probe = True
        elif kind == "eval_inplace_alias_probe":
            features.add("pandas:eval-inplace-alias")
            features.add("copy:on-write-alias")
            available_types = {op_output_alias(op, "derived"): "bool"}
            has_eval_inplace_alias_probe = True
        elif kind == "bool_reduction_skipna_probe":
            features.add("pandas:bool-reduction-skipna")
            features.add("nullable-bool:reduction")
            available_types = {op_output_alias(op, "derived"): "bool"}
            has_bool_reduction_skipna_probe = True
        elif kind == "dataset_isin_all_match_probe":
            features.add("pyarrow:dataset-isin-all-match")
            features.add("dataset:membership-filter")
            available_types = {op_output_alias(op, "derived"): "bool"}
            has_dataset_isin_all_match_probe = True
        elif kind == "run_end_null_compute_probe":
            features.add("pyarrow:run-end-null-compute")
            features.add("run_end:null-compute")
            available_types = {op_output_alias(op, "derived"): "bool"}
            has_run_end_null_compute_probe = True
        elif kind == "large_string_partition_probe":
            features.add("pyarrow:large-string-partition")
            features.add("dataset:partition-schema")
            available_types = {op_output_alias(op, "derived"): "bool"}
            has_large_string_partition_probe = True
        elif kind == "hash_pivot_wider_probe":
            features.add("pyarrow:hash-pivot-wider")
            features.add("pivot:wider-order")
            available_types = {op_output_alias(op, "derived"): "bool"}
            has_hash_pivot_wider_probe = True
        elif kind == "list_flatten_parent_indices_probe":
            features.add("pyarrow:list-flatten-parent-indices")
            features.add("arrow:list-layout")
            available_types = {op_output_alias(op, "derived"): "bool"}
            has_list_flatten_parent_indices_probe = True
        elif kind == "rolling_mean_by_null_count_probe":
            features.add("polars:rolling-mean-by-null-count")
            features.add("rolling:temporal-min-samples")
            available_types = {op_output_alias(op, "derived"): "bool"}
            has_rolling_mean_by_null_count_probe = True
        elif kind == "csv_long_numeric_roundtrip_probe":
            features.add("csv:long-numeric-roundtrip")
            features.add("csv:numeric-inference")
            features.add("numeric:long-identifier")
            features.add(_bucket("csv_probe_values", len(op_values(op)), [(3, "few"), (6, "several")], "many"))
            available_types = {op_output_alias(op, "derived"): "bool"}
            has_csv_long_numeric_roundtrip_probe = True
        elif kind == "select":
            width = len(op_columns(op))
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
            features.add(f"join:{join_how(op, 'unknown')}")
            features.add(f"join_table:{op_table(op, 'unknown')}")
            right_table = table_by_name.get(op_table(op, ""))
            right_keys = join_right_keys(op)
            right_key = right_keys[0] if right_keys else ""
            if right_table is not None:
                for column in right_table.columns:
                    if column.name == right_key or column.name in available_types:
                        continue
                    available_types[column.name] = column.type
        elif kind == "limit":
            limit = op_n(op, 0)
            if limit == 0:
                features.add("op:limit_zero")
            features.add(_bucket("limit", limit, [(0, "zero"), (3, "tiny"), (10, "small")], "large"))
        elif kind == "offset":
            offset = op_n(op, 0)
            if offset == 0:
                features.add("op:offset_zero")
            features.add(_bucket("offset", offset, [(0, "zero"), (3, "tiny"), (10, "small")], "large"))
        elif kind == "mutate":
            mutate_kind = expr_kind(op, "unknown")
            output_column = op_column(op, "derived")
            source_column = expr_source(op)
            features.add(f"mutate:{mutate_kind}")
            features.add(f"expr:{mutate_kind}")
            if mutate_kind == "arith_const":
                features.add(f"arith:{expr_operator(op, 'unknown')}")
                if expr_operator(op) == "div":
                    available_types[output_column] = "float"
                else:
                    available_types[output_column] = available_types.get(source_column, "derived")
            elif mutate_kind == "reverse_division_columns":
                features.add("arithmetic:operand-order")
                features.add("arithmetic:reverse-division")
                available_types[output_column] = "float"
            elif mutate_kind == "abs":
                features.add("numeric:abs")
                available_types[output_column] = available_types.get(source_column, "derived")
            elif mutate_kind == "clip":
                features.add("numeric:clip")
                available_types[output_column] = available_types.get(source_column, "derived")
            elif mutate_kind == "bool_not":
                features.add("boolean:not")
                available_types[output_column] = "bool"
            elif mutate_kind == "add_const":
                available_types[output_column] = available_types.get(source_column, "derived")
            if mutate_kind == "cast":
                target = expr_target_type(op, "unknown")
                features.add(f"cast_to:{target}")
                features.add(f"cast:{available_types.get(source_column, 'derived')}_to_{target}")
                if expr_input_domain(op):
                    features.add(f"cast_domain:{expr_input_domain(op)}")
                available_types[output_column] = target or "derived"
            elif mutate_kind == "string_length":
                features.add("string:length")
                available_types[output_column] = "int"
            elif mutate_kind == "string_lower":
                features.add("string:lower")
                available_types[output_column] = "str"
            elif mutate_kind == "string_upper":
                features.add("string:upper")
                available_types[output_column] = "str"
            elif mutate_kind == "string_strip":
                features.add("string:strip")
                available_types[output_column] = "str"
            elif mutate_kind == "string_null_if_empty":
                features.add("string:null-if-empty")
                features.add("null:empty-string")
                available_types[output_column] = "str"
            elif mutate_kind == "string_replace":
                features.add("string:replace")
                available_types[output_column] = "str"
            elif mutate_kind == "string_slice":
                features.add("string:slice")
                available_types[output_column] = "str"
            elif mutate_kind == "string_split_part":
                features.add("string:split-first")
                available_types[output_column] = "str"
            elif mutate_kind == "string_concat":
                features.add("string:concat")
                available_types[output_column] = "str"
            elif mutate_kind == "string_contains":
                features.add("string:contains")
                available_types[output_column] = "bool"
            elif mutate_kind == "string_starts_with":
                features.add("string:starts-with")
                available_types[output_column] = "bool"
            elif mutate_kind == "string_ends_with":
                features.add("string:ends-with")
                available_types[output_column] = "bool"
            elif mutate_kind == "date_part":
                features.add("date:part")
                features.add(f"date_part:{expr_part(op, 'unknown')}")
                available_types[output_column] = "int"
            elif mutate_kind == "string_basename":
                features.add("path:basename")
                available_types[output_column] = "str"
                has_string_basename_expr = True
                if has_row_number_filter:
                    has_path_basename_keyed_pick = True
        elif kind == "groupby":
            keys = groupby_keys(op)
            agg_funcs = {aggregate_func(agg, "unknown") for agg in aggregate_specs(op)}
            if agg_funcs and agg_funcs <= {"count", "nunique", "min", "max", "any", "all"}:
                features.add("groupby:exact-aggregate")
                features.add("groupby:sorted-input")
                has_exact_groupby_agg = True
            features.add(_bucket("groupby_keys", len(keys), [(1, "one"), (2, "two")], "many"))
            if len(keys) > 1:
                features.add("groupby:multi-key")
            for key in keys:
                features.add(f"group_key_type:{available_types.get(key, 'derived')}")
            for agg in aggregate_specs(op):
                source = aggregate_column(agg)
                func = aggregate_func(agg, "unknown")
                source_type = available_types.get(source, "derived")
                features.add(f"agg:{func}")
                features.add(f"agg_source_type:{source_type}")
                if source_type == "float" and func in {"sum", "mean"}:
                    features.add("agg:precision-float")
                if source_type == "bool":
                    features.add("agg:boolean")
                    features.add(f"agg:{func}:bool")
                    if func in {"min", "max", "count", "nunique", "any", "all"}:
                        has_bool_groupby_agg = True
                if func == "count" and source_type == "str":
                    features.add("agg:count:str")
                    has_string_count_groupby = True
                if func == "nunique":
                    features.add(f"agg:nunique:{source_type}")
                    has_unique_count_groupby = True
                available_types[aggregate_alias(agg, "derived")] = aggregate_feature_type(
                    source_type,
                    func,
                )
        elif kind == "aggregate":
            for agg in aggregate_specs(op):
                source = aggregate_column(agg)
                func = aggregate_func(agg, "unknown")
                source_type = available_types.get(source, "derived")
                features.add(f"agg:{func}")
                features.add(f"agg_source_type:{source_type}")
                if source_type == "float" and func in {"sum", "mean"}:
                    features.add("agg:precision-float")
                if source_type == "bool":
                    features.add("agg:boolean")
                    features.add(f"agg:{func}:bool")
                if func == "count" and source_type == "str":
                    features.add("agg:count:str")
                if func == "nunique":
                    features.add(f"agg:nunique:{source_type}")
                available_types[aggregate_alias(agg, "derived")] = aggregate_feature_type(
                    source_type,
                    func,
                )
    if op_names:
        features.add("opseq:" + ">".join(op_names))
        features.add(_bucket("op_count", len(op_names), [(1, "one"), (3, "few"), (5, "many")], "deep"))
        first_op = op_names[0]
        if first_op in {"filter", "drop_nulls", "fill_null", "distinct"}:
            features.add("materialization:input")
            features.add("pattern:input_materialization_boundary")
            if first_op == "filter":
                features.add("materialization:filter")
                features.add("pattern:filter_input_materialization")
            else:
                features.add("materialization:leading-cleanup")
                features.add(f"materialization:{first_op}")
                features.add("pattern:cleanup_input_materialization")
                features.add(f"pattern:{first_op}_input_materialization")
        if len(table.rows) >= 2 and "limit" not in op_names and "offset" not in op_names and not case.program.order_sensitive:
            features.add("materialization:input-partition")
            features.add("pattern:input_partition_union_all")
    if generator_profile == "common_api_workflow" or mixed_generator_profile == "common_api_workflow":
        features.add("pattern:common_api_workflow")
        template = str(case.metadata.get("workflow_template", "")).strip()
        if template:
            features.add(f"common_api_template:{template}")
    source_issue = _case_source_issue(case)
    if source_issue:
        features.add(f"source_issue:{_source_issue_key(source_issue)}")
        features.add(f"source:{case_discovery_origin(case)}")
    combo = operation_combo or describe_operation_combo(case.program.operations)
    features.add(f"combo:{combo['template']}")
    features.add(f"combo_frequency:{combo['frequency_bucket']}")
    for signal in combo_semantic_signals(combo):
        features.add(_semantic_signal_feature(signal))
        features.add(f"combo_risk:{signal}")
    resolved_frontier_buckets = frontier_buckets
    if resolved_frontier_buckets is None:
        _, resolved_frontier_buckets = _frontier_signature(case)
    features.update(resolved_frontier_buckets)
    features.update(
        program_pattern_features(
            case.program.operations,
            resolved_frontier_buckets,
            op_names=op_names,
            has_range_filter=has_range_filter,
            has_running_sum_precision=has_running_sum_precision,
            has_sortedness_check=has_sortedness_check,
        )
    )
    if has_string_count_groupby:
        features.add("pattern:string_count_groupby")
    if has_unique_count_groupby:
        features.add("pattern:unique_count_groupby")
    if has_bool_groupby_agg:
        features.add("pattern:bool_null_groupby_agg")
    if has_set_membership_filter:
        features.add("pattern:set_membership_filter")
    if has_null_predicate_filter:
        features.add("pattern:null_predicate_filter")
    if has_boolean_predicate_filter:
        features.add("pattern:boolean_predicate_filter")
    if has_range_filter:
        features.add("pattern:range_filter")
    if has_string_contains_filter:
        features.add("pattern:string_contains_filter")
        features.add("pattern:string_pattern_filter")
    if has_tuple_absence_filter:
        features.add("pattern:tuple_absence_filter")
    if has_union_all:
        features.add("pattern:union_all_row_append")
    if has_drop_nulls:
        features.add("pattern:drop_nulls_null_filter")
    if has_semi_join:
        features.add("pattern:semi_join_membership")
        features.add("pattern:semi_anti_join_null_keys")
        features.add("pattern:semi_anti_join_rewrite")
    if has_anti_join:
        features.add("pattern:anti_join_exclusion")
        features.add("pattern:semi_anti_join_null_keys")
        features.add("pattern:semi_anti_join_rewrite")
    if has_distinct:
        features.add("pattern:distinct_deduplicate")
    if has_fill_null:
        features.add("pattern:fill_null_null_semantics")
    if has_coalesce:
        features.add("pattern:coalesce_null_semantics")
    if has_case_when:
        features.add("pattern:conditional_expression")
    if has_exact_groupby_agg:
        features.add("pattern:groupby_sorted_input")
    if generator_profile == "row_value_absence_filter" or mixed_generator_profile == "row_value_absence_filter":
        features.add("pattern:row_value_absence_filter")
    if generator_profile == "large_int_filter_groupby" or mixed_generator_profile == "large_int_filter_groupby":
        features.add("pattern:large_int_filter_groupby")
    if has_partitioned_running_sum or generator_profile == "partitioned_running_sum":
        features.add("pattern:partitioned_running_sum")
    if (
        has_path_basename_keyed_pick
        or generator_profile == "path_basename_keyed_pick"
        or mixed_generator_profile == "path_basename_keyed_pick"
    ):
        features.add("pattern:path_basename_keyed_pick")
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
    if has_float_literal_precision_probe:
        features.add("pattern:duckdb_float_literal_precision")
    if has_timestamp_precision_filter_probe:
        features.add("pattern:polars_timestamp_precision_filter")
    if has_series_rtruediv_probe:
        features.add("pattern:series_rtruediv_operand_order")
    if has_uint64_isin_probe:
        features.add("pattern:pandas_uint64_isin_precision")
    if has_tuple_anti_null_probe:
        features.add("pattern:duckdb_tuple_anti_null_semantics")
    if has_setop_all_duplicate_probe:
        features.add("pattern:datafusion_setop_all_duplicate_count")
    if has_json_predicate_order_probe:
        features.add("pattern:duckdb_json_predicate_order_semantics")
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
    if has_eval_inplace_alias_probe:
        features.add("pattern:pandas_eval_inplace_aliasing_semantics")
    if has_bool_reduction_skipna_probe:
        features.add("pattern:pandas_bool_reduction_skipna_semantics")
    if has_dataset_isin_all_match_probe:
        features.add("pattern:pyarrow_dataset_isin_all_match_semantics")
    if has_run_end_null_compute_probe:
        features.add("pattern:pyarrow_run_end_null_compute_semantics")
    if has_large_string_partition_probe:
        features.add("pattern:pyarrow_large_string_partition_schema_semantics")
    if has_hash_pivot_wider_probe:
        features.add("pattern:pyarrow_hash_pivot_wider_order_semantics")
    if has_list_flatten_parent_indices_probe:
        features.add("pattern:pyarrow_list_flatten_parent_indices_semantics")
    if has_rolling_mean_by_null_count_probe:
        features.add("pattern:polars_rolling_mean_by_null_count_semantics")
    if has_csv_long_numeric_roundtrip_probe:
        features.add("pattern:csv_long_numeric_roundtrip")
    features.update(derive_semantic_family_features(features))
    features.update(
        derive_exploration_objective_features(
            features,
            rules=exploration_objective_rules,
        )
    )
    return features


def extract_case_features(case: Case) -> set[str]:
    return derive_case_features(case)


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
    discovery_buckets: list[str] = field(default_factory=list)
    discovery_bias_hits: list[str] = field(default_factory=list)
    score_breakdown: dict[str, float] = field(default_factory=dict)
    online_weights: list[dict[str, Any]] = field(default_factory=list)
    analysis: "CaseAnalysis | None" = field(default=None, repr=False)
    frontier_conformance_metric: float = 0.0
    contribution_potential_metric: float = 0.0
    target_priority_metric: float = 0.0
    specific_target_matches_metric: float = 0.0
    discovery_bias_bonus_metric: float = 0.0
    discovery_diversity_bonus_metric: float = 0.0
    profile_saturation_penalty_metric: float = 0.0
    target_no_yield_penalty_metric: float = 0.0
    discovery_stale_penalty_metric: float = 0.0
    recent_discovery_loop_penalty_metric: float = 0.0
    resolved_semantic_boundary_penalty_metric: float = 0.0
    candidate_pool_bias_bonus_metric: float = 0.0
    candidate_pool_shared_bonus_metric: float = 0.0
    candidate_pool_diversity_bonus_metric: float = 0.0
    discovery_bias_keep_in_pool_flag: bool = False
    family_saturation_active_flag: bool = False
    profile_saturation_active_flag: bool = False
    discovery_stale_active_flag: bool = False
    recent_discovery_loop_active_flag: bool = False
    issue_replay_global_saturation_active_flag: bool = False
    issue_inspired_source_saturation_active_flag: bool = False
    path_coverage_proxy_metric: float = 0.0
    data_sensitivity_metric: float = 0.0
    target_bonus_metric: float = 0.0
    target_template_matches_metric: float = 0.0
    target_template_bonus_metric: float = 0.0
    finding_yield_bonus_metric: float = 0.0
    combo_priority_metric: float = 0.0
    online_weight_mean_metric: float = 1.0
    online_weight_max_metric: float = 1.0
    feature_saturation_penalty_metric: float = 0.0
    root_saturation_penalty_metric: float = 0.0
    family_saturation_penalty_metric: float = 0.0
    issue_replay_saturation_penalty_metric: float = 0.0
    issue_replay_global_saturation_penalty_metric: float = 0.0
    issue_inspired_source_saturation_penalty_metric: float = 0.0
    issue_replay_saturation_active_flag: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": round(self.score, 6),
            "features": self.features,
            "matched_targets": self.matched_targets,
            "candidate_count": self.candidate_count,
            "contributing_candidate_count": self.contributing_candidate_count,
            "pruned_candidate_count": self.pruned_candidate_count,
            "frontier_buckets": self.frontier_buckets,
            "discovery_buckets": self.discovery_buckets,
            "discovery_bias_hits": self.discovery_bias_hits,
            "score_breakdown": {
                key: round(value, 6) for key, value in sorted(self.score_breakdown.items())
            },
            "online_weights": self.online_weights,
        }


def _materialize_score_breakdown(
    decision: GuidanceDecision,
    *,
    recent_discovery_window_count: float,
    online_weight_updates: float,
) -> dict[str, float]:
    return {
        "path_coverage_proxy": decision.path_coverage_proxy_metric,
        "data_sensitivity": decision.data_sensitivity_metric,
        "frontier_conformance": decision.frontier_conformance_metric,
        "discovery_diversity_bonus": decision.discovery_diversity_bonus_metric,
        "candidate_pool_diversity_bonus": decision.candidate_pool_diversity_bonus_metric,
        "candidate_pool_shared_bonus": decision.candidate_pool_shared_bonus_metric,
        "candidate_pool_bias_bonus": decision.candidate_pool_bias_bonus_metric,
        "discovery_bias_bonus": decision.discovery_bias_bonus_metric,
        "discovery_bias_hit_count": float(len(decision.discovery_bias_hits)),
        "discovery_bias_keep_in_pool": 1.0 if decision.discovery_bias_keep_in_pool_flag else 0.0,
        "discovery_stale_penalty": decision.discovery_stale_penalty_metric,
        "discovery_stale_active": 1.0 if decision.discovery_stale_active_flag else 0.0,
        "recent_discovery_loop_penalty": decision.recent_discovery_loop_penalty_metric,
        "recent_discovery_loop_active": 1.0 if decision.recent_discovery_loop_active_flag else 0.0,
        "recent_discovery_window_count": recent_discovery_window_count,
        "discovery_bucket_count": float(len(decision.discovery_buckets)),
        "target_bonus": decision.target_bonus_metric,
        "target_priority": decision.target_priority_metric,
        "target_template_matches": decision.target_template_matches_metric,
        "target_template_bonus": decision.target_template_bonus_metric,
        "target_no_yield_penalty": decision.target_no_yield_penalty_metric,
        "specific_target_matches": decision.specific_target_matches_metric,
        "finding_yield_bonus": decision.finding_yield_bonus_metric,
        "combo_priority": decision.combo_priority_metric,
        "contribution_potential": decision.contribution_potential_metric,
        "online_weight_mean": decision.online_weight_mean_metric,
        "online_weight_max": decision.online_weight_max_metric,
        "online_weight_updates": online_weight_updates,
        "feature_saturation_penalty": decision.feature_saturation_penalty_metric,
        "root_saturation_penalty": decision.root_saturation_penalty_metric,
        "resolved_semantic_boundary_penalty": decision.resolved_semantic_boundary_penalty_metric,
        "profile_saturation_penalty": decision.profile_saturation_penalty_metric,
        "profile_saturation_active": 1.0 if decision.profile_saturation_active_flag else 0.0,
        "family_saturation_penalty": decision.family_saturation_penalty_metric,
        "family_saturation_active": 1.0 if decision.family_saturation_active_flag else 0.0,
        "issue_replay_saturation_penalty": decision.issue_replay_saturation_penalty_metric,
        "issue_replay_saturation_active": 1.0 if decision.issue_replay_saturation_active_flag else 0.0,
        "issue_replay_global_saturation_penalty": decision.issue_replay_global_saturation_penalty_metric,
        "issue_replay_global_saturation_active": (
            1.0 if decision.issue_replay_global_saturation_active_flag else 0.0
        ),
        "issue_inspired_source_saturation_penalty": decision.issue_inspired_source_saturation_penalty_metric,
        "issue_inspired_source_saturation_active": (
            1.0 if decision.issue_inspired_source_saturation_active_flag else 0.0
        ),
    }


@dataclass(slots=True)
class FeatureRewardStats:
    pulls: int = 0
    total_reward: float = 0.0

    @property
    def mean_reward(self) -> float:
        return self.total_reward / self.pulls if self.pulls else 0.0


@dataclass(slots=True)
class CaseAnalysis:
    operation_combo: dict[str, Any]
    combo_priority_base: float
    features: set[str]
    ordered_features: list[str]
    canonical_features: set[str]
    expanded_features: frozenset[str]
    expanded_feature_prefixes: frozenset[str]
    expanded_feature_prefix_roots: frozenset[str]
    canonical_feature_weight_bases: tuple[tuple[str, float, float], ...]
    feature_score_specs: tuple["FeatureScoreSpec", ...]
    canonical_feature_count_sqrt: float
    path_features: tuple[str, ...]
    path_feature_weight_bases: tuple[tuple[str, float], ...]
    path_feature_count_sqrt: float
    path_operation_diversity_bonus: float
    path_sequence_bonus: float
    data_features: tuple[str, ...]
    data_feature_weight_bases: tuple[tuple[str, float], ...]
    data_feature_count_sqrt: float
    learnable_features: tuple[str, ...]
    learnable_feature_prefix_pairs: tuple[tuple[str, str], ...]
    learnable_feature_count: int
    mixed_profile_features: tuple[str, ...]
    matched_targets: list[str]
    matched_target_set: frozenset[str]
    matched_target_aliases: frozenset[str]
    matched_target_count: int
    specific_target_match_count: int
    generic_target_match_count: int
    target_match_priority_base: float
    target_template_match_count: int
    target_template_features: tuple[str, ...]
    target_penalty_specs: tuple["TargetPenaltySpec", ...]
    resolved_semantic_boundary_penalty: float
    frontier_raw_score: float
    frontier_buckets: list[str]
    frontier_bucket_count_sqrt: float
    discovery_buckets: list[str]
    discovery_bucket_weights: tuple[tuple[str, float], ...]
    discovery_bucket_count_sqrt: float
    predicted_roots: set[str]
    issue_inspired_source_keys: tuple[str, ...]
    has_issue_replay_source: bool
    has_issue_inspired_source: bool


@dataclass(frozen=True, slots=True)
class FeatureScoreSpec:
    feature: str
    finding_weight_base: float
    saturation_weight_base: float
    path_weight_base: float
    data_weight_base: float
    learnable_feature_prefix: str | None
    is_mixed_profile: bool


@dataclass(slots=True)
class CandidateRootContext:
    case: Case
    operations: tuple[Any, ...]
    op_sequence: tuple[str, ...]
    op_set: frozenset[str]
    hinted_roots: frozenset[str]
    probe_root: str | None


@dataclass(slots=True)
class TargetPenaltySpec:
    aliases: frozenset[str]
    weight: float


@dataclass(frozen=True, slots=True)
class CompiledDiscoveryBias:
    label: str
    targets: frozenset[str]
    target_aliases: frozenset[str]
    root_prefixes: frozenset[str]
    full_prefixes: tuple[str, ...]
    exact_feature_prefixes: frozenset[str]
    scan_feature_prefixes: tuple[str, ...]
    score_bonus: float
    novelty_bonus: float
    contribution_bonus: float
    candidate_pool_bonus: float
    keep_in_pool: bool


@dataclass(slots=True)
class OnlineFeatureWeights:
    exploration_weight: float = 0.12
    min_multiplier: float = 0.35
    max_multiplier: float = 2.50
    feature_stats: dict[str, FeatureRewardStats] = field(default_factory=dict)
    prefix_stats: dict[str, FeatureRewardStats] = field(default_factory=dict)
    total_updates: int = 0
    _multiplier_cache: dict[str, float] = field(default_factory=dict, repr=False)
    _prefix_only_multiplier_cache: dict[str, float] = field(default_factory=dict, repr=False)
    _snapshot_cache: dict[int, list[dict[str, Any]]] = field(default_factory=dict, repr=False)
    _log_total_updates_term: float = field(default=math.log(2.0), repr=False)

    def record(
        self,
        features: set[str],
        reward: float,
        *,
        learnable_feature_prefix_pairs: tuple[tuple[str, str], ...] | None = None,
    ) -> None:
        resolved_pairs = learnable_feature_prefix_pairs
        if resolved_pairs is None:
            learnable = [feature for feature in _canonical_features(features) if _is_learnable_weight_feature(feature)]
            if not learnable:
                return
            resolved_pairs = tuple((feature, _feature_prefix(feature)) for feature in learnable)
        if not resolved_pairs:
            return
        self.total_updates += 1
        self._multiplier_cache.clear()
        self._prefix_only_multiplier_cache.clear()
        self._snapshot_cache.clear()
        self._log_total_updates_term = math.log(self.total_updates + 2.0)
        for feature, feature_prefix in resolved_pairs:
            self._record_stat(self.feature_stats, feature, reward)
            self._record_stat(self.prefix_stats, feature_prefix, reward)

    def multiplier(self, feature: str) -> float:
        cached = self._multiplier_cache.get(feature)
        if cached is not None:
            return cached
        return self._multiplier_from_parts(feature, _feature_prefix(feature))

    def multiplier_for_prefix(self, feature: str, feature_prefix: str) -> float:
        cached = self._multiplier_cache.get(feature)
        if cached is not None:
            return cached
        return self._multiplier_from_parts(feature, feature_prefix)

    def _multiplier_from_parts(self, feature: str, feature_prefix: str) -> float:
        exact = self.feature_stats.get(feature)
        prefix = self.prefix_stats.get(feature_prefix)
        if exact is None and prefix is None:
            self._multiplier_cache[feature] = 1.0
            return 1.0
        if exact is None:
            cached_prefix = self._prefix_only_multiplier_cache.get(feature_prefix)
            if cached_prefix is not None:
                self._multiplier_cache[feature] = cached_prefix
                return cached_prefix
        exact_signal = self._reward_signal(exact, max_abs=2.0)
        prefix_signal = self._reward_signal(prefix, max_abs=1.5)
        pulls = exact.pulls if exact is not None else 0
        exploration = self.exploration_weight * math.sqrt(
            self._log_total_updates_term / max(1, pulls)
        )
        value = 1.0 + (0.65 * exact_signal) + (0.35 * prefix_signal) + exploration
        bounded = min(self.max_multiplier, max(self.min_multiplier, value))
        self._multiplier_cache[feature] = bounded
        if exact is None:
            self._prefix_only_multiplier_cache[feature_prefix] = bounded
        return bounded

    def snapshot(self, limit: int = 12) -> list[dict[str, Any]]:
        cached = self._snapshot_cache.get(limit)
        if cached is not None:
            return [dict(row) for row in cached]
        rows = [
            {
                "feature": feature,
                "pulls": stats.pulls,
                "mean_reward": stats.mean_reward,
                "reward_signal": self._reward_signal(stats, max_abs=2.0),
                "multiplier": self.multiplier(feature),
            }
            for feature, stats in self.feature_stats.items()
        ]
        snapshot = sorted(rows, key=lambda row: (row["multiplier"], row["pulls"]), reverse=True)[:limit]
        self._snapshot_cache[limit] = [dict(row) for row in snapshot]
        return [dict(row) for row in snapshot]

    def to_state_dict(self) -> dict[str, Any]:
        return {
            "exploration_weight": self.exploration_weight,
            "min_multiplier": self.min_multiplier,
            "max_multiplier": self.max_multiplier,
            "feature_stats": {
                feature: {
                    "pulls": stats.pulls,
                    "total_reward": stats.total_reward,
                }
                for feature, stats in self.feature_stats.items()
            },
            "prefix_stats": {
                prefix: {
                    "pulls": stats.pulls,
                    "total_reward": stats.total_reward,
                }
                for prefix, stats in self.prefix_stats.items()
            },
            "total_updates": self.total_updates,
        }

    @classmethod
    def from_state_dict(
        cls,
        data: dict[str, Any],
        *,
        exploration_weight: float | None = None,
        min_multiplier: float | None = None,
        max_multiplier: float | None = None,
    ) -> "OnlineFeatureWeights":
        state = cls(
            exploration_weight=float(
                exploration_weight
                if exploration_weight is not None
                else data.get("exploration_weight", 0.12) or 0.12
            ),
            min_multiplier=float(
                min_multiplier
                if min_multiplier is not None
                else data.get("min_multiplier", 0.35) or 0.35
            ),
            max_multiplier=float(
                max_multiplier
                if max_multiplier is not None
                else data.get("max_multiplier", 2.50) or 2.50
            ),
        )
        state.total_updates = int(data.get("total_updates", 0) or 0)
        state.feature_stats = _canonical_feature_stats(data.get("feature_stats", {}) or {})
        state.prefix_stats = _canonical_prefix_stats(data.get("prefix_stats", {}) or {})
        state._multiplier_cache.clear()
        state._prefix_only_multiplier_cache.clear()
        state._snapshot_cache.clear()
        state._log_total_updates_term = math.log(state.total_updates + 2.0)
        return state

    @staticmethod
    def _record_stat(stats: dict[str, FeatureRewardStats], key: str, reward: float) -> None:
        stat = stats.setdefault(key, FeatureRewardStats())
        stat.pulls += 1
        stat.total_reward += reward

    @staticmethod
    def _reward_signal(stats: FeatureRewardStats | None, *, max_abs: float) -> float:
        if stats is None or stats.pulls <= 0:
            return 0.0
        return _bounded_confident_mean_reward(
            stats.total_reward,
            stats.pulls,
            max_abs=max_abs,
        )


@dataclass(slots=True)
class GuidanceState:
    targets: list[str] = field(default_factory=list)
    discovery_biases: list[DiscoveryBias] = field(default_factory=list)
    feature_counts: Counter[str] = field(default_factory=Counter)
    finding_feature_counts: Counter[str] = field(default_factory=Counter)
    root_cause_counts: Counter[str] = field(default_factory=Counter)
    candidate_bug_family_counts: Counter[str] = field(default_factory=Counter)
    issue_replay_family_counts: Counter[str] = field(default_factory=Counter)
    issue_inspired_source_counts: Counter[str] = field(default_factory=Counter)
    candidate_bug_signature_counts: Counter[str] = field(default_factory=Counter)
    frontier_bucket_counts: Counter[str] = field(default_factory=Counter)
    discovery_bucket_counts: Counter[str] = field(default_factory=Counter)
    discovery_bucket_signal_counts: Counter[str] = field(default_factory=Counter)
    recent_discovery_windows: deque[tuple[tuple[str, ...], bool]] = field(default_factory=deque)
    recent_discovery_window_limit: int = 32
    online_weights: OnlineFeatureWeights = field(default_factory=OnlineFeatureWeights)
    enable_family_saturation: bool = True
    family_saturation_threshold: int = 8
    family_saturation_penalty: float = 1.25
    saturated_family_reward: float = 0.02
    known_saturated_bug_families: list[str] = field(default_factory=list)
    exploration_objective_rules: list[ExplorationObjectiveRule] = field(default_factory=list)
    issue_replay_saturation_threshold: int = 1
    issue_replay_saturation_penalty: float = 1.0
    issue_replay_global_saturation_threshold: int = 4
    issue_replay_global_saturation_penalty: float = 1.5
    issue_inspired_source_saturation_threshold: int = 3
    issue_inspired_source_saturation_penalty: float = 1.25
    issue_replay_count: int = 0
    active_backends: list[str] = field(default_factory=list)
    recent_discovery_stale_counts: Counter[str] = field(default_factory=Counter)
    recent_discovery_signal_counts: Counter[str] = field(default_factory=Counter)
    _active_backend_filter: frozenset[str] = field(default_factory=frozenset, repr=False)
    _known_saturated_roots: set[str] = field(default_factory=set, repr=False)
    _candidate_bug_root_hits: Counter[str] = field(default_factory=Counter, repr=False)
    _issue_replay_root_hits: Counter[str] = field(default_factory=Counter, repr=False)
    _case_analysis_cache: OrderedDict[tuple[int, tuple[Any, ...]], CaseAnalysis] = field(default_factory=OrderedDict, repr=False)
    _max_cached_case_analyses: int = field(default=128, repr=False)
    _compiled_discovery_biases: tuple[CompiledDiscoveryBias, ...] = field(default_factory=tuple, repr=False)
    _discovery_bias_keys: tuple[tuple[Any, ...], ...] = field(default_factory=tuple, repr=False)

    def __post_init__(self) -> None:
        self.exploration_objective_rules = merge_exploration_objective_rules(
            self.exploration_objective_rules
        )
        self._sync_compiled_discovery_biases()
        self._sync_family_saturation_state()

    def _sync_compiled_discovery_biases(self) -> None:
        keys = tuple(bias.key() for bias in self.discovery_biases)
        if keys == self._discovery_bias_keys:
            return
        self._discovery_bias_keys = keys
        self._compiled_discovery_biases = _compile_discovery_biases(self.discovery_biases)

    def _sync_family_saturation_state(self) -> None:
        self._active_backend_filter = _normalized_active_backends(self.active_backends)
        self._known_saturated_roots = _family_roots_for_active_backends(
            self.known_saturated_bug_families,
            active_backends=self._active_backend_filter,
        )
        self._candidate_bug_root_hits = _root_hit_counts_for_active_backends(
            self.candidate_bug_family_counts,
            active_backends=self._active_backend_filter,
        )
        self._issue_replay_root_hits = _root_hit_counts_for_active_backends(
            self.issue_replay_family_counts,
            active_backends=self._active_backend_filter,
        )

    def _record_family_hits(
        self,
        updates: Counter[str] | dict[str, int],
        *,
        family_counts: Counter[str],
        root_hits: Counter[str],
    ) -> None:
        if not updates:
            return
        family_counts.update(updates)
        for family_key, count in updates.items():
            hits = int(count or 0)
            if hits <= 0:
                continue
            root = _family_root_for_active_backends(
                family_key,
                active_backends=self._active_backend_filter,
            )
            if root:
                root_hits[root] += hits

    def _populate_decision_diagnostics(
        self,
        decision: GuidanceDecision,
        *,
        include_online_weight_snapshot: bool = True,
    ) -> GuidanceDecision:
        if not decision.score_breakdown or "path_coverage_proxy" not in decision.score_breakdown:
            decision.score_breakdown = _materialize_score_breakdown(
                decision,
                recent_discovery_window_count=len(self.recent_discovery_windows),
                online_weight_updates=float(self.online_weights.total_updates),
            )
        if include_online_weight_snapshot and not decision.online_weights:
            decision.online_weights = self.online_weights.snapshot(limit=16)
        return decision

    def predicted_saturated_family_roots(self, case: Case, *, include_known_families: bool = True) -> list[str]:
        if not self.enable_family_saturation or self.family_saturation_threshold <= 0:
            return []
        analysis = self._case_analysis(case)
        return [
            root
            for root in sorted(analysis.predicted_roots)
            if _predicted_family_hit_count(
                root,
                root_hit_counts=self._candidate_bug_root_hits,
                known_saturated_roots=self._known_saturated_roots,
                threshold=self.family_saturation_threshold,
                include_known_families=include_known_families,
            )
            >= self.family_saturation_threshold
        ]

    def predicted_saturated_family_roots_for_candidate(
        self,
        case: Case,
        *,
        include_known_families: bool = True,
    ) -> list[str]:
        if not self.enable_family_saturation or self.family_saturation_threshold <= 0:
            return []
        saturated_roots = self._candidate_prefilter_saturated_roots(
            include_known_families=include_known_families,
        )
        if not saturated_roots:
            return []
        context = _candidate_root_context(case)
        return [
            root
            for root in sorted(saturated_roots)
            if _candidate_prefilter_matches_root(context, root)
        ]

    def select_case(
        self,
        candidates: list[Case],
        *,
        include_online_weight_snapshot: bool = True,
    ) -> GuidanceDecision:
        if not candidates:
            raise ValueError("guided candidate pool cannot be empty")
        self._sync_compiled_discovery_biases()
        discovery_bucket_metric_cache: dict[str, tuple[float, float, float]] = {}
        recent_discovery_window_count = len(self.recent_discovery_windows)
        scored = [
            self._score_case(
                case,
                len(candidates),
                compiled_discovery_biases=self._compiled_discovery_biases,
                discovery_bucket_metric_cache=discovery_bucket_metric_cache,
                recent_discovery_window_count=recent_discovery_window_count,
                include_score_breakdown=False,
            )
            for case in candidates
        ]
        _apply_candidate_pool_discovery_balance(
            scored,
            self.discovery_bucket_counts,
            self.recent_discovery_stale_counts,
        )
        contributing = [decision for decision in scored if self._is_contributing_candidate(decision)]
        if not contributing:
            contributing = [max(scored, key=lambda decision: (decision.score, -decision.case.seed))]
        contributing = _prefer_unsaturated_decisions(
            contributing,
            scored,
            _decision_has_issue_replay_global_saturation,
        )
        contributing = _prefer_unsaturated_decisions(
            contributing,
            scored,
            _decision_has_issue_inspired_source_saturation,
        )
        keep_in_pool = [decision for decision in contributing if _decision_has_keep_in_pool_bias(decision)]
        if keep_in_pool:
            contributing = keep_in_pool
        contributing = _prefer_unsaturated_decisions(
            contributing,
            scored,
            _decision_has_profile_saturation,
        )
        contributing = _prefer_unsaturated_decisions(
            contributing,
            scored,
            _decision_has_discovery_stale,
        )
        contributing = _prefer_unsaturated_decisions(
            contributing,
            scored,
            _decision_has_recent_discovery_loop,
        )
        contributing = _prefer_unsaturated_decisions(
            contributing,
            scored,
            _decision_has_resolved_semantic_boundary,
        )
        contributing = _prefer_unsaturated_decisions(
            contributing,
            scored,
            _decision_has_family_saturation,
        )
        pruned = len(scored) - len(contributing)
        for decision in contributing:
            decision.contributing_candidate_count = len(contributing)
            decision.pruned_candidate_count = pruned
        matched = [decision for decision in contributing if decision.matched_targets]
        bias_matched = [decision for decision in contributing if decision.discovery_bias_hits]
        if bias_matched:
            matched = matched + [decision for decision in bias_matched if decision not in matched]
        if matched:
            unsaturated_matched = [
                decision for decision in matched if not _decision_has_family_saturation(decision)
            ]
            if unsaturated_matched:
                matched = unsaturated_matched
            elif len(matched) < len(contributing):
                unsaturated_contributing = [
                    decision for decision in contributing if not _decision_has_family_saturation(decision)
                ]
                if unsaturated_contributing:
                    return self._populate_decision_diagnostics(
                        max(
                            unsaturated_contributing,
                            key=lambda decision: (decision.score, -decision.case.seed),
                        ),
                        include_online_weight_snapshot=include_online_weight_snapshot,
                    )
                return self._populate_decision_diagnostics(
                    max(contributing, key=lambda decision: (decision.score, -decision.case.seed)),
                    include_online_weight_snapshot=include_online_weight_snapshot,
                )
            boundary_avoiding_matched = [
                decision for decision in matched if not _decision_has_resolved_semantic_boundary(decision)
            ]
            if boundary_avoiding_matched:
                matched = boundary_avoiding_matched
            return self._populate_decision_diagnostics(
                max(matched, key=_matched_decision_key),
                include_online_weight_snapshot=include_online_weight_snapshot,
            )
        return self._populate_decision_diagnostics(
            max(contributing, key=lambda decision: (decision.score, -decision.case.seed)),
            include_online_weight_snapshot=include_online_weight_snapshot,
        )

    def choose_case(
        self,
        candidates: list[Case],
        *,
        include_online_weight_snapshot: bool = True,
    ) -> GuidanceDecision:
        return self.select_case(
            candidates,
            include_online_weight_snapshot=include_online_weight_snapshot,
        )

    def record_result(
        self,
        case: Case,
        row: dict[str, Any],
        *,
        finding_outcomes: FindingOutcomeAnalysis | None = None,
    ) -> None:
        analysis = self._case_analysis(case)
        features = analysis.features
        canonical_features = analysis.canonical_features
        self.feature_counts.update(canonical_features)
        issue_inspired_case_sources = Counter(analysis.issue_inspired_source_keys)
        if issue_inspired_case_sources:
            self.issue_inspired_source_counts.update(issue_inspired_case_sources)
        frontier_buckets = analysis.frontier_buckets
        self.frontier_bucket_counts.update(frontier_buckets)
        reward = _guidance_reward(
            row,
            root_cause_counts=self.root_cause_counts,
            candidate_bug_family_counts=self.candidate_bug_family_counts,
            candidate_bug_signature_counts=self.candidate_bug_signature_counts,
            enable_family_saturation=self.enable_family_saturation,
            family_saturation_threshold=self.family_saturation_threshold,
            saturated_family_reward=self.saturated_family_reward,
            known_saturated_bug_families=self.known_saturated_bug_families,
            finding_outcomes=finding_outcomes,
        )
        self.online_weights.record(
            canonical_features,
            reward,
            learnable_feature_prefix_pairs=analysis.learnable_feature_prefix_pairs,
        )
        discovery_buckets = analysis.discovery_buckets
        self.discovery_bucket_counts.update(discovery_buckets)
        has_discovery_signal = _row_has_discovery_signal(
            row,
            reward,
            known_saturated_bug_families=self.known_saturated_bug_families,
            finding_outcomes=finding_outcomes,
        )
        self._record_recent_discovery_window(discovery_buckets, has_discovery_signal)
        if has_discovery_signal:
            self.discovery_bucket_signal_counts.update(discovery_buckets)
        findings = row.get("findings") or []
        if findings:
            outcomes = finding_outcomes or analyze_finding_outcomes(
                findings,
                known_saturated_bug_families=self.known_saturated_bug_families,
            )
            self.finding_feature_counts.update(canonical_features)
            self.root_cause_counts.update(outcomes.root_cause_counts)
            self._record_family_hits(
                outcomes.candidate_bug_families,
                family_counts=self.candidate_bug_family_counts,
                root_hits=self._candidate_bug_root_hits,
            )
            issue_replay_families = outcomes.issue_replay_candidate_bug_families
            self._record_family_hits(
                issue_replay_families,
                family_counts=self.issue_replay_family_counts,
                root_hits=self._issue_replay_root_hits,
            )
            self.issue_replay_count += sum(issue_replay_families.values())
            if not issue_inspired_case_sources:
                self.issue_inspired_source_counts.update(_issue_inspired_candidate_source_issue_keys(findings))
            self.candidate_bug_signature_counts.update(outcomes.candidate_bug_signatures)

    def _score_case(
        self,
        case: Case,
        candidate_count: int,
        *,
        compiled_discovery_biases: tuple[CompiledDiscoveryBias, ...] | None = None,
        discovery_bucket_metric_cache: dict[str, tuple[float, float, float]] | None = None,
        recent_discovery_window_count: int | None = None,
        include_score_breakdown: bool = True,
    ) -> GuidanceDecision:
        analysis = self._case_analysis(case)
        feature_score_specs = analysis.feature_score_specs
        matched_targets = analysis.matched_targets
        matched_target_count = analysis.matched_target_count
        specific_target_matches = analysis.specific_target_match_count
        target_template_matches = analysis.target_template_match_count
        target_template_bonus = _target_template_bonus_from_features(
            analysis.target_template_features,
            self.feature_counts,
        )
        target_priority = analysis.target_match_priority_base + target_template_bonus
        target_no_yield_penalty = _target_no_yield_penalty(
            analysis.canonical_features,
            matched_targets,
            self.feature_counts,
            self.finding_feature_counts,
            expanded_features=analysis.expanded_features,
            target_penalty_specs=analysis.target_penalty_specs,
        )
        feature_counts_get = self.feature_counts.get
        finding_feature_counts_get = self.finding_feature_counts.get
        root_cause_counts_get = self.root_cause_counts.get
        online_weight_total = 0.0
        online_weight_max = 1.0
        path_novelty_total = 0.0
        data_weighted_total = 0.0
        finding_yield_total = 0.0
        feature_saturation_total = 0.0
        profile_saturation_penalty = 0.0
        path_novelty_count = 0
        data_novelty_count = 0
        if feature_score_specs:
            feature_multiplier = self.online_weights.multiplier_for_prefix
            for spec in feature_score_specs:
                count = feature_counts_get(spec.feature, 0)
                if spec.learnable_feature_prefix is not None:
                    multiplier_value = feature_multiplier(spec.feature, spec.learnable_feature_prefix)
                    online_weight_total += multiplier_value
                    if multiplier_value > online_weight_max:
                        online_weight_max = multiplier_value
                else:
                    multiplier_value = 1.0
                if spec.path_weight_base > 0.0:
                    path_novelty_total += (spec.path_weight_base * multiplier_value) / (1.0 + count)
                    if count == 0:
                        path_novelty_count += 1
                if spec.data_weight_base > 0.0:
                    data_weighted_total += (spec.data_weight_base * multiplier_value) * (1.0 + 1.0 / (1.0 + count))
                    if count == 0:
                        data_novelty_count += 1
                finding_hits = finding_feature_counts_get(spec.feature, 0)
                finding_yield_total += _bounded_finding_signal(finding_hits) * spec.finding_weight_base * multiplier_value
                feature_saturation_total += _feature_saturation(finding_hits) * spec.saturation_weight_base * multiplier_value
                if spec.is_mixed_profile:
                    profile_saturation_penalty += _profile_saturation(count)
        path_coverage_proxy = 0.0
        if analysis.path_feature_weight_bases:
            path_coverage_proxy = (
                path_novelty_total / analysis.path_feature_count_sqrt
                + analysis.path_operation_diversity_bonus
                + analysis.path_sequence_bonus
            )
        data_sensitivity = 0.0
        if analysis.data_feature_weight_bases:
            data_sensitivity = (data_weighted_total / analysis.data_feature_count_sqrt) * 0.35
        frontier_buckets = analysis.frontier_buckets
        frontier_conformance = analysis.frontier_raw_score
        frontier_novelty_count = 0
        if frontier_buckets:
            frontier_counts_get = self.frontier_bucket_counts.get
            frontier_weighted_novelty = 0.0
            for bucket in frontier_buckets:
                bucket_count = frontier_counts_get(bucket, 0)
                frontier_weighted_novelty += 1.0 / (1.0 + bucket_count)
                if bucket_count == 0:
                    frontier_novelty_count += 1
            frontier_conformance += (frontier_weighted_novelty / analysis.frontier_bucket_count_sqrt) * 0.35
        discovery_buckets = analysis.discovery_buckets
        discovery_bucket_weights = analysis.discovery_bucket_weights
        (
            discovery_bias_bonus,
            discovery_bias_novelty_bonus,
            discovery_bias_contribution_bonus,
            discovery_bias_pool_bonus,
            discovery_bias_keep_in_pool,
            discovery_bias_hits,
        ) = _apply_discovery_biases(
            analysis,
            compiled_discovery_biases
            if compiled_discovery_biases is not None
            else self._compiled_discovery_biases,
        )
        (
            discovery_diversity_bonus,
            discovery_stale_penalty,
            recent_discovery_loop_penalty,
        ) = _discovery_metrics(
            discovery_bucket_weights,
            self.discovery_bucket_counts,
            self.discovery_bucket_signal_counts,
            self.recent_discovery_stale_counts,
            self.recent_discovery_signal_counts,
            recent_window_count=(
                recent_discovery_window_count
                if recent_discovery_window_count is not None
                else len(self.recent_discovery_windows)
            ),
            bucket_metric_cache=discovery_bucket_metric_cache,
        )
        discovery_diversity_bonus += discovery_bias_novelty_bonus
        discovery_stale_active = discovery_stale_penalty > 0.0
        recent_discovery_loop_active = recent_discovery_loop_penalty > 0.0
        target_bonus = 3.0 * target_priority
        finding_yield_bonus = (finding_yield_total / analysis.canonical_feature_count_sqrt) * 0.50
        feature_saturation_penalty = (feature_saturation_total / analysis.canonical_feature_count_sqrt) * 0.08
        profile_saturation_active = profile_saturation_penalty > 0.0
        predicted_roots = analysis.predicted_roots
        root_saturation_penalty = 0.0
        root_novelty = 0
        root_saturation = 0
        for root in predicted_roots:
            root_hits = root_cause_counts_get(root, 0)
            root_saturation_penalty += _root_saturation(root_hits)
            if root_hits == 0:
                root_novelty += 1
            elif root_hits >= 12:
                root_saturation += 1
        resolved_semantic_boundary_penalty = analysis.resolved_semantic_boundary_penalty
        family_saturation_penalty = 0.0
        family_saturation_active = False
        issue_replay_saturation_penalty = 0.0
        issue_replay_saturation_active = False
        issue_replay_global_saturation_penalty = 0.0
        issue_replay_global_saturation_active = False
        issue_inspired_source_saturation_penalty = 0.0
        issue_inspired_source_saturation_active = False
        if self.enable_family_saturation:
            family_saturation_penalty, family_saturation_active = _predicted_family_saturation_state(
                predicted_roots,
                root_hit_counts=self._candidate_bug_root_hits,
                known_saturated_roots=self._known_saturated_roots,
                threshold=self.family_saturation_threshold,
                penalty_weight=self.family_saturation_penalty,
            )
            issue_replay_saturation_penalty, issue_replay_saturation_active = _predicted_family_saturation_state(
                predicted_roots,
                root_hit_counts=self._issue_replay_root_hits,
                known_saturated_roots=_EMPTY_ROOTS,
                threshold=self.issue_replay_saturation_threshold,
                penalty_weight=self.issue_replay_saturation_penalty,
            )
            if analysis.has_issue_replay_source:
                issue_replay_global_saturation_penalty = _family_saturation_penalty(
                    self.issue_replay_count,
                    threshold=self.issue_replay_global_saturation_threshold,
                    penalty_weight=self.issue_replay_global_saturation_penalty,
                )
                issue_replay_global_saturation_active = _is_globally_saturated(
                    self.issue_replay_count,
                    threshold=self.issue_replay_global_saturation_threshold,
                )
            if analysis.has_issue_inspired_source:
                source_issue_hits = _max_source_issue_hits_for_keys(
                    analysis.issue_inspired_source_keys,
                    source_counts=self.issue_inspired_source_counts,
                )
                issue_inspired_source_saturation_penalty = _family_saturation_penalty(
                    source_issue_hits,
                    threshold=self.issue_inspired_source_saturation_threshold,
                    penalty_weight=self.issue_inspired_source_saturation_penalty,
                )
                issue_inspired_source_saturation_active = _is_globally_saturated(
                    source_issue_hits,
                    threshold=self.issue_inspired_source_saturation_threshold,
                )
        if matched_targets:
            saturation_multiplier = 0.35 if specific_target_matches else 0.85
            feature_saturation_penalty *= saturation_multiplier
            root_saturation_penalty *= saturation_multiplier
            profile_saturation_penalty *= saturation_multiplier
            family_saturation_penalty *= saturation_multiplier
            issue_replay_saturation_penalty *= saturation_multiplier
            issue_replay_global_saturation_penalty *= saturation_multiplier
            issue_inspired_source_saturation_penalty *= saturation_multiplier
        issue_replay_saturation_active_any = (
            issue_replay_saturation_active or issue_replay_global_saturation_active
        )
        contribution_potential = (
            frontier_conformance
            + 0.30 * path_novelty_count
            + 0.20 * data_novelty_count
            + 0.45 * frontier_novelty_count
            + 0.35 * root_novelty
            + 0.60 * matched_target_count
            - 0.20 * root_saturation
            + discovery_bias_contribution_bonus
        )
        combo_priority = analysis.combo_priority_base
        online_weight_mean = (
            online_weight_total / analysis.learnable_feature_count
            if analysis.learnable_feature_count
            else 1.0
        )
        score = (
            path_coverage_proxy
            + data_sensitivity
            + frontier_conformance
            + discovery_diversity_bonus
            + target_bonus
            + finding_yield_bonus
            + combo_priority
            + discovery_bias_bonus
        )
        score -= (
            feature_saturation_penalty
            + root_saturation_penalty
            + resolved_semantic_boundary_penalty
            + profile_saturation_penalty
            + family_saturation_penalty
            + issue_replay_saturation_penalty
            + issue_replay_global_saturation_penalty
            + issue_inspired_source_saturation_penalty
            + target_no_yield_penalty
            + discovery_stale_penalty
            + recent_discovery_loop_penalty
        )
        decision = GuidanceDecision(
            case=case,
            score=score,
            features=analysis.ordered_features,
            matched_targets=matched_targets,
            candidate_count=candidate_count,
            frontier_buckets=frontier_buckets,
            discovery_buckets=discovery_buckets,
            discovery_bias_hits=discovery_bias_hits,
            analysis=analysis,
            frontier_conformance_metric=frontier_conformance,
            contribution_potential_metric=contribution_potential,
            target_priority_metric=target_priority,
            specific_target_matches_metric=float(specific_target_matches),
            discovery_bias_bonus_metric=discovery_bias_bonus,
            discovery_diversity_bonus_metric=discovery_diversity_bonus,
            profile_saturation_penalty_metric=-profile_saturation_penalty,
            target_no_yield_penalty_metric=-target_no_yield_penalty,
            discovery_stale_penalty_metric=-discovery_stale_penalty,
            recent_discovery_loop_penalty_metric=-recent_discovery_loop_penalty,
            resolved_semantic_boundary_penalty_metric=-resolved_semantic_boundary_penalty,
            candidate_pool_bias_bonus_metric=discovery_bias_pool_bonus,
            discovery_bias_keep_in_pool_flag=discovery_bias_keep_in_pool,
            family_saturation_active_flag=(family_saturation_active or issue_replay_saturation_active_any),
            profile_saturation_active_flag=profile_saturation_active,
            discovery_stale_active_flag=discovery_stale_active,
            recent_discovery_loop_active_flag=recent_discovery_loop_active,
            issue_replay_global_saturation_active_flag=issue_replay_global_saturation_active,
            issue_inspired_source_saturation_active_flag=issue_inspired_source_saturation_active,
            path_coverage_proxy_metric=path_coverage_proxy,
            data_sensitivity_metric=data_sensitivity,
            target_bonus_metric=target_bonus,
            target_template_matches_metric=float(target_template_matches),
            target_template_bonus_metric=target_template_bonus,
            finding_yield_bonus_metric=finding_yield_bonus,
            combo_priority_metric=combo_priority,
            online_weight_mean_metric=online_weight_mean,
            online_weight_max_metric=online_weight_max,
            feature_saturation_penalty_metric=-feature_saturation_penalty,
            root_saturation_penalty_metric=-root_saturation_penalty,
            family_saturation_penalty_metric=-family_saturation_penalty,
            issue_replay_saturation_penalty_metric=-issue_replay_saturation_penalty,
            issue_replay_global_saturation_penalty_metric=-issue_replay_global_saturation_penalty,
            issue_inspired_source_saturation_penalty_metric=-issue_inspired_source_saturation_penalty,
            issue_replay_saturation_active_flag=issue_replay_saturation_active_any,
            score_breakdown={},
        )
        if include_score_breakdown:
            decision.score_breakdown = _materialize_score_breakdown(
                decision,
                recent_discovery_window_count=float(
                    recent_discovery_window_count
                    if recent_discovery_window_count is not None
                    else len(self.recent_discovery_windows)
                ),
                online_weight_updates=float(self.online_weights.total_updates),
            )
        return decision

    def _case_analysis(self, case: Case) -> CaseAnalysis:
        objective_rule_keys = tuple(
            rule.key()
            for rule in active_exploration_objective_rules(self.exploration_objective_rules)
        )
        key = (id(case), objective_rule_keys)
        cached = self._case_analysis_cache.get(key)
        if cached is not None:
            self._case_analysis_cache.move_to_end(key)
            return cached
        frontier_raw_score, frontier_buckets = _frontier_signature(case)
        operation_combo = describe_operation_combo(case.program.operations)
        features = derive_case_features(
            case,
            operation_combo=operation_combo,
            frontier_buckets=frontier_buckets,
            exploration_objective_rules=self.exploration_objective_rules,
        )
        canonical_features = _canonical_features(features)
        expanded_features = frozenset(_alias_expanded_features(canonical_features))
        expanded_feature_prefixes = frozenset(
            prefix
            for feature in expanded_features
            if (prefix := _feature_prefix_token(feature))
        )
        expanded_feature_prefix_roots = frozenset(
            _feature_prefix_root(feature)
            for feature in expanded_features
        )
        path_features = tuple(feature for feature in canonical_features if _is_path_feature(feature))
        path_feature_weight_map = {
            feature: _path_feature_weight_base(feature)
            for feature in path_features
        }
        path_feature_weight_bases = tuple(
            (feature, path_feature_weight_map[feature])
            for feature in path_features
        )
        data_features = tuple(feature for feature in canonical_features if _is_data_sensitivity_feature(feature))
        data_feature_weight_map = {
            feature: _data_feature_weight_base(feature)
            for feature in data_features
        }
        data_feature_weight_bases = tuple(
            (feature, data_feature_weight_map[feature])
            for feature in data_features
        )
        learnable_features = tuple(
            feature
            for feature in canonical_features
            if _is_learnable_weight_feature(feature)
        )
        learnable_feature_prefix_pairs = tuple(
            (feature, _feature_prefix(feature))
            for feature in learnable_features
        )
        learnable_feature_prefix_map = dict(learnable_feature_prefix_pairs)
        mixed_profile_features = tuple(
            feature for feature in canonical_features if feature.startswith("mixed_generator_profile:")
        )
        mixed_profile_feature_set = set(mixed_profile_features)
        canonical_feature_weight_bases = tuple(
            (
                feature,
                _finding_feature_weight_base(feature),
                _saturation_feature_weight_base(feature),
            )
            for feature in canonical_features
        )
        feature_score_specs = tuple(
            FeatureScoreSpec(
                feature=feature,
                finding_weight_base=finding_weight_base,
                saturation_weight_base=saturation_weight_base,
                path_weight_base=path_feature_weight_map.get(feature, 0.0),
                data_weight_base=data_feature_weight_map.get(feature, 0.0),
                learnable_feature_prefix=learnable_feature_prefix_map.get(feature),
                is_mixed_profile=feature in mixed_profile_feature_set,
            )
            for feature, finding_weight_base, saturation_weight_base in canonical_feature_weight_bases
        )
        matched_targets = _matched_targets(canonical_features, self.targets, expanded_features=expanded_features)
        matched_target_set = frozenset(matched_targets)
        matched_target_aliases = frozenset(_alias_expanded_features(set(matched_targets)))
        matched_target_count = len(matched_targets)
        specific_target_match_count = 0
        target_template_match_count = 0
        target_template_features_list: list[str] = []
        target_penalty_specs: list[TargetPenaltySpec] = []
        for target in matched_targets:
            is_specific_target = _is_specific_target(target)
            if is_specific_target:
                specific_target_match_count += 1
            template_features = [
                feature
                for feature in (f"mixed_generator_profile:{target}", f"generator_profile:{target}")
                if feature in features
            ]
            if template_features:
                target_template_match_count += 1
                target_template_features_list.extend(template_features)
            matched_aliases = expanded_features & _target_aliases(target)
            if matched_aliases:
                target_penalty_specs.append(
                    TargetPenaltySpec(
                        aliases=frozenset(matched_aliases),
                        weight=(0.30 if is_specific_target else 0.08),
                    )
                )
        generic_target_match_count = matched_target_count - specific_target_match_count
        generic_target_weight = GENERIC_COMPANION_TARGET_WEIGHT if specific_target_match_count else 1.0
        target_match_priority_base = (
            (specific_target_match_count * PATTERN_TARGET_WEIGHT)
            + (generic_target_match_count * generic_target_weight)
        )
        discovery_buckets = _discovery_buckets(features, matched_targets, frontier_buckets)
        discovery_bucket_weights = tuple(
            (bucket, _discovery_bucket_weight(bucket))
            for bucket in discovery_buckets
        )
        predicted_roots = _predicted_roots(canonical_features)
        issue_inspired_source_keys = tuple(
            sorted(
                _source_issue_key(feature.split(":", 1)[1])
                for feature in canonical_features
                if feature.startswith("source_issue:")
                and _source_issue_key(feature.split(":", 1)[1])
                and "source:issue_inspired" in features
            )
        )
        analysis = CaseAnalysis(
            operation_combo=operation_combo,
            combo_priority_base=float(operation_combo.get("priority", 0.0) or 0.0) * 0.25,
            features=features,
            ordered_features=sorted(features),
            canonical_features=canonical_features,
            expanded_features=expanded_features,
            expanded_feature_prefixes=expanded_feature_prefixes,
            expanded_feature_prefix_roots=expanded_feature_prefix_roots,
            canonical_feature_weight_bases=canonical_feature_weight_bases,
            feature_score_specs=feature_score_specs,
            canonical_feature_count_sqrt=math.sqrt(max(1, len(canonical_feature_weight_bases))),
            path_features=path_features,
            path_feature_weight_bases=path_feature_weight_bases,
            path_feature_count_sqrt=math.sqrt(max(1, len(path_feature_weight_bases))),
            path_operation_diversity_bonus=(
                sum(1 for feature in path_features if feature.startswith("op:")) * 0.12
            ),
            path_sequence_bonus=0.20 if any(feature.startswith("opseq:") for feature in path_features) else 0.0,
            data_features=data_features,
            data_feature_weight_bases=data_feature_weight_bases,
            data_feature_count_sqrt=math.sqrt(max(1, len(data_feature_weight_bases))),
            learnable_features=learnable_features,
            learnable_feature_prefix_pairs=learnable_feature_prefix_pairs,
            learnable_feature_count=len(learnable_feature_prefix_pairs),
            mixed_profile_features=mixed_profile_features,
            matched_targets=matched_targets,
            matched_target_set=matched_target_set,
            matched_target_aliases=matched_target_aliases,
            matched_target_count=matched_target_count,
            specific_target_match_count=specific_target_match_count,
            generic_target_match_count=generic_target_match_count,
            target_match_priority_base=target_match_priority_base,
            target_template_match_count=target_template_match_count,
            target_template_features=tuple(target_template_features_list),
            target_penalty_specs=tuple(target_penalty_specs),
            resolved_semantic_boundary_penalty=_resolved_semantic_boundary_penalty(
                canonical_features,
                predicted_roots,
                matched_targets,
            ),
            frontier_raw_score=frontier_raw_score,
            frontier_buckets=frontier_buckets,
            frontier_bucket_count_sqrt=math.sqrt(max(1, len(frontier_buckets))),
            discovery_buckets=discovery_buckets,
            discovery_bucket_weights=discovery_bucket_weights,
            discovery_bucket_count_sqrt=math.sqrt(max(1, len(discovery_bucket_weights))),
            predicted_roots=predicted_roots,
            issue_inspired_source_keys=issue_inspired_source_keys,
            has_issue_replay_source="source:issue_replay" in features,
            has_issue_inspired_source="source:issue_inspired" in features,
        )
        self._case_analysis_cache[key] = analysis
        self._case_analysis_cache.move_to_end(key)
        while len(self._case_analysis_cache) > self._max_cached_case_analyses:
            self._case_analysis_cache.popitem(last=False)
        return analysis

    def _candidate_prefilter_saturated_roots(self, *, include_known_families: bool) -> set[str]:
        roots = {
            root
            for root, hits in self._candidate_bug_root_hits.items()
            if hits >= self.family_saturation_threshold
        }
        if include_known_families:
            roots.update(self._known_saturated_roots)
        return roots & _CANDIDATE_PREFILTER_ROOTS

    def _is_contributing_candidate(self, decision: GuidanceDecision) -> bool:
        analysis = decision.analysis
        features = analysis.canonical_features if analysis is not None else _canonical_features(set(decision.features))
        path_features = analysis.path_features if analysis is not None else tuple(
            feature for feature in features if _is_path_feature(feature)
        )
        feature_counts_get = self.feature_counts.get
        unseen_path = False
        for feature in path_features:
            if feature_counts_get(feature, 0) == 0:
                unseen_path = True
                break
        frontier_counts_get = self.frontier_bucket_counts.get
        unseen_frontier = False
        for bucket in decision.frontier_buckets:
            if frontier_counts_get(bucket, 0) == 0:
                unseen_frontier = True
                break
        frontier_conformance = decision.frontier_conformance_metric
        contribution_potential = decision.contribution_potential_metric
        if decision.discovery_bias_keep_in_pool_flag:
            return True

        if decision.matched_targets:
            return True
        if unseen_path or unseen_frontier:
            return True
        if frontier_conformance >= 0.80:
            return True
        return contribution_potential >= 1.15

    def _record_recent_discovery_window(self, discovery_buckets: list[str], has_signal: bool) -> None:
        if self.recent_discovery_window_limit <= 0:
            self.recent_discovery_windows.clear()
            self.recent_discovery_stale_counts.clear()
            self.recent_discovery_signal_counts.clear()
            return
        bucket_tuple = tuple(discovery_buckets)
        self.recent_discovery_windows.append((bucket_tuple, has_signal))
        bucket_counts = self.recent_discovery_signal_counts if has_signal else self.recent_discovery_stale_counts
        bucket_counts.update(bucket_tuple)
        while len(self.recent_discovery_windows) > self.recent_discovery_window_limit:
            old_buckets, old_has_signal = self.recent_discovery_windows.popleft()
            old_counts = self.recent_discovery_signal_counts if old_has_signal else self.recent_discovery_stale_counts
            for bucket in old_buckets:
                old_counts[bucket] -= 1
                if old_counts[bucket] <= 0:
                    del old_counts[bucket]

    def to_state_dict(self) -> dict[str, Any]:
        return {
            "targets": list(self.targets),
            "discovery_biases": [bias.to_dict() for bias in self.discovery_biases],
            "feature_counts": dict(self.feature_counts),
            "finding_feature_counts": dict(self.finding_feature_counts),
            "root_cause_counts": dict(self.root_cause_counts),
            "candidate_bug_family_counts": dict(self.candidate_bug_family_counts),
            "issue_replay_family_counts": dict(self.issue_replay_family_counts),
            "issue_inspired_source_counts": dict(self.issue_inspired_source_counts),
            "candidate_bug_signature_counts": dict(self.candidate_bug_signature_counts),
            "frontier_bucket_counts": dict(self.frontier_bucket_counts),
            "discovery_bucket_counts": dict(self.discovery_bucket_counts),
            "discovery_bucket_signal_counts": dict(self.discovery_bucket_signal_counts),
            "recent_discovery_windows": [
                {
                    "buckets": list(buckets),
                    "has_signal": bool(has_signal),
                }
                for buckets, has_signal in self.recent_discovery_windows
            ],
            "recent_discovery_window_limit": self.recent_discovery_window_limit,
            "recent_discovery_stale_counts": dict(self.recent_discovery_stale_counts),
            "recent_discovery_signal_counts": dict(self.recent_discovery_signal_counts),
            "online_weights": self.online_weights.to_state_dict(),
            "enable_family_saturation": self.enable_family_saturation,
            "family_saturation_threshold": self.family_saturation_threshold,
            "family_saturation_penalty": self.family_saturation_penalty,
            "saturated_family_reward": self.saturated_family_reward,
            "known_saturated_bug_families": list(self.known_saturated_bug_families),
            "exploration_objective_rules": [
                rule.to_dict() for rule in self.exploration_objective_rules
            ],
            "issue_replay_saturation_threshold": self.issue_replay_saturation_threshold,
            "issue_replay_saturation_penalty": self.issue_replay_saturation_penalty,
            "issue_replay_global_saturation_threshold": self.issue_replay_global_saturation_threshold,
            "issue_replay_global_saturation_penalty": self.issue_replay_global_saturation_penalty,
            "issue_inspired_source_saturation_threshold": self.issue_inspired_source_saturation_threshold,
            "issue_inspired_source_saturation_penalty": self.issue_inspired_source_saturation_penalty,
            "issue_replay_count": self.issue_replay_count,
            "active_backends": list(self.active_backends),
        }

    @classmethod
    def from_state_dict(
        cls,
        data: dict[str, Any],
        *,
        targets: list[str] | None = None,
        discovery_biases: list[DiscoveryBias] | None = None,
        enable_family_saturation: bool | None = None,
        family_saturation_threshold: int | None = None,
        family_saturation_penalty: float | None = None,
        saturated_family_reward: float | None = None,
        known_saturated_bug_families: list[str] | None = None,
        exploration_objective_rules: list[ExplorationObjectiveRule] | None = None,
        issue_replay_saturation_threshold: int | None = None,
        issue_replay_saturation_penalty: float | None = None,
        issue_replay_global_saturation_threshold: int | None = None,
        issue_replay_global_saturation_penalty: float | None = None,
        issue_inspired_source_saturation_threshold: int | None = None,
        issue_inspired_source_saturation_penalty: float | None = None,
        active_backends: list[str] | None = None,
    ) -> "GuidanceState":
        state = cls(
            targets=list(targets if targets is not None else data.get("targets", []) or []),
            discovery_biases=list(
                discovery_biases
                if discovery_biases is not None
                else [
                    DiscoveryBias(**item)
                    for item in (data.get("discovery_biases", []) or [])
                    if isinstance(item, dict)
                ]
            ),
            enable_family_saturation=bool(
                enable_family_saturation
                if enable_family_saturation is not None
                else data.get("enable_family_saturation", True)
            ),
            family_saturation_threshold=int(
                family_saturation_threshold
                if family_saturation_threshold is not None
                else data.get("family_saturation_threshold", 8) or 8
            ),
            family_saturation_penalty=float(
                family_saturation_penalty
                if family_saturation_penalty is not None
                else data.get("family_saturation_penalty", 1.25) or 1.25
            ),
            saturated_family_reward=float(
                saturated_family_reward
                if saturated_family_reward is not None
                else data.get("saturated_family_reward", 0.02) or 0.02
            ),
            known_saturated_bug_families=list(
                known_saturated_bug_families
                if known_saturated_bug_families is not None
                else data.get("known_saturated_bug_families", []) or []
            ),
            exploration_objective_rules=merge_exploration_objective_rules(
                exploration_objective_rules
                if exploration_objective_rules is not None
                else data.get("exploration_objective_rules", []) or []
            ),
            issue_replay_saturation_threshold=int(
                issue_replay_saturation_threshold
                if issue_replay_saturation_threshold is not None
                else data.get("issue_replay_saturation_threshold", 1) or 1
            ),
            issue_replay_saturation_penalty=float(
                issue_replay_saturation_penalty
                if issue_replay_saturation_penalty is not None
                else data.get("issue_replay_saturation_penalty", 1.0) or 1.0
            ),
            issue_replay_global_saturation_threshold=int(
                issue_replay_global_saturation_threshold
                if issue_replay_global_saturation_threshold is not None
                else data.get("issue_replay_global_saturation_threshold", 4) or 4
            ),
            issue_replay_global_saturation_penalty=float(
                issue_replay_global_saturation_penalty
                if issue_replay_global_saturation_penalty is not None
                else data.get("issue_replay_global_saturation_penalty", 1.5) or 1.5
            ),
            issue_inspired_source_saturation_threshold=int(
                issue_inspired_source_saturation_threshold
                if issue_inspired_source_saturation_threshold is not None
                else data.get("issue_inspired_source_saturation_threshold", 3) or 3
            ),
            issue_inspired_source_saturation_penalty=float(
                issue_inspired_source_saturation_penalty
                if issue_inspired_source_saturation_penalty is not None
                else data.get("issue_inspired_source_saturation_penalty", 1.25) or 1.25
            ),
            active_backends=list(
                active_backends if active_backends is not None else data.get("active_backends", []) or []
            ),
        )
        state.feature_counts = _canonical_counter(data.get("feature_counts", {}) or {})
        state.finding_feature_counts = _canonical_counter(data.get("finding_feature_counts", {}) or {})
        state.root_cause_counts = Counter(data.get("root_cause_counts", {}) or {})
        state.candidate_bug_family_counts = Counter(data.get("candidate_bug_family_counts", {}) or {})
        state.issue_replay_family_counts = Counter(data.get("issue_replay_family_counts", {}) or {})
        state.issue_inspired_source_counts = Counter(data.get("issue_inspired_source_counts", {}) or {})
        state.candidate_bug_signature_counts = Counter(data.get("candidate_bug_signature_counts", {}) or {})
        state.frontier_bucket_counts = Counter(data.get("frontier_bucket_counts", {}) or {})
        state.discovery_bucket_counts = Counter(data.get("discovery_bucket_counts", {}) or {})
        state.discovery_bucket_signal_counts = Counter(data.get("discovery_bucket_signal_counts", {}) or {})
        state.recent_discovery_window_limit = max(
            0,
            int(data.get("recent_discovery_window_limit", state.recent_discovery_window_limit) or 0),
        )
        recent_discovery_windows = deque()
        for item in data.get("recent_discovery_windows", []) or []:
            if not isinstance(item, dict):
                continue
            buckets = tuple(str(bucket) for bucket in item.get("buckets", []) or [])
            has_signal = bool(item.get("has_signal", False))
            recent_discovery_windows.append((buckets, has_signal))
        while len(recent_discovery_windows) > state.recent_discovery_window_limit > 0:
            recent_discovery_windows.popleft()
        if state.recent_discovery_window_limit <= 0:
            recent_discovery_windows.clear()
        state.recent_discovery_windows = recent_discovery_windows
        state.recent_discovery_stale_counts = Counter(
            {
                str(key): int(value or 0)
                for key, value in (data.get("recent_discovery_stale_counts", {}) or {}).items()
            }
        )
        state.recent_discovery_signal_counts = Counter(
            {
                str(key): int(value or 0)
                for key, value in (data.get("recent_discovery_signal_counts", {}) or {}).items()
            }
        )
        state._sync_recent_discovery_counts()
        state._sync_family_saturation_state()
        online_weights_data = data.get("online_weights", {}) or {}
        if isinstance(online_weights_data, dict):
            state.online_weights = OnlineFeatureWeights.from_state_dict(online_weights_data)
        state.issue_replay_count = int(data.get("issue_replay_count", 0) or 0)
        return state

    def _sync_recent_discovery_counts(self) -> None:
        stale_expected = 0
        signal_expected = 0
        for buckets, has_signal in self.recent_discovery_windows:
            if has_signal:
                signal_expected += len(buckets)
            else:
                stale_expected += len(buckets)
        if sum(self.recent_discovery_stale_counts.values()) != stale_expected:
            self.recent_discovery_stale_counts = Counter(
                bucket
                for buckets, has_signal in self.recent_discovery_windows
                if not has_signal
                for bucket in buckets
            )
        if sum(self.recent_discovery_signal_counts.values()) != signal_expected:
            self.recent_discovery_signal_counts = Counter(
                bucket
                for buckets, has_signal in self.recent_discovery_windows
                if has_signal
                for bucket in buckets
            )


def _matched_decision_key(decision: GuidanceDecision) -> tuple[float, ...]:
    specific_matches = decision.specific_target_matches_metric
    target_priority = decision.target_priority_metric or float(len(decision.matched_targets))
    bias_bonus = decision.discovery_bias_bonus_metric
    profile_penalty = decision.profile_saturation_penalty_metric
    no_yield_penalty = decision.target_no_yield_penalty_metric
    discovery_penalty = decision.discovery_stale_penalty_metric
    recent_loop_penalty = decision.recent_discovery_loop_penalty_metric
    discovery_bonus = decision.discovery_diversity_bonus_metric
    if specific_matches > 0.0:
        return (
            1.0,
            bias_bonus,
            profile_penalty,
            no_yield_penalty,
            discovery_penalty,
            recent_loop_penalty,
            specific_matches,
            target_priority,
            discovery_bonus,
            decision.score,
            -decision.case.seed,
        )
    return (
        0.0,
        bias_bonus,
        profile_penalty,
        no_yield_penalty,
        discovery_penalty,
        recent_loop_penalty,
        discovery_bonus,
        decision.score,
        target_priority,
        float(len(decision.matched_targets)),
        -decision.case.seed,
    )


def _semantic_focus_decision_key(decision: GuidanceDecision) -> tuple[float, ...]:
    return _matched_decision_key(decision)


def _contrast_decision_key(decision: GuidanceDecision) -> tuple[float, ...]:
    return _semantic_focus_decision_key(decision)


def _decision_has_family_saturation(decision: GuidanceDecision) -> bool:
    return decision.family_saturation_active_flag


def _decision_has_profile_saturation(decision: GuidanceDecision) -> bool:
    return decision.profile_saturation_active_flag


def _decision_has_discovery_stale(decision: GuidanceDecision) -> bool:
    return decision.discovery_stale_active_flag


def _decision_has_recent_discovery_loop(decision: GuidanceDecision) -> bool:
    return decision.recent_discovery_loop_active_flag


def _decision_has_resolved_semantic_boundary(decision: GuidanceDecision) -> bool:
    return decision.resolved_semantic_boundary_penalty_metric < 0.0


def _decision_has_issue_replay_global_saturation(decision: GuidanceDecision) -> bool:
    return decision.issue_replay_global_saturation_active_flag


def _decision_has_issue_inspired_source_saturation(decision: GuidanceDecision) -> bool:
    return decision.issue_inspired_source_saturation_active_flag


def _decision_has_keep_in_pool_bias(decision: GuidanceDecision) -> bool:
    return decision.discovery_bias_keep_in_pool_flag


def _compile_discovery_biases(
    discovery_biases: list[DiscoveryBias],
) -> tuple[CompiledDiscoveryBias, ...]:
    compiled: list[CompiledDiscoveryBias] = []
    for bias in discovery_biases:
        full_prefixes = tuple(
            prefix
            for prefix in (_normalize_bias_prefix(prefix) for prefix in bias.feature_prefixes)
            if prefix
        )
        exact_feature_prefixes = frozenset(
            prefix for prefix in full_prefixes if _is_exact_feature_prefix(prefix)
        )
        compiled.append(
            CompiledDiscoveryBias(
                label=_discovery_bias_label(bias),
                targets=frozenset(bias.targets),
                target_aliases=frozenset(
                    alias
                    for target in bias.targets
                    for alias in _bias_target_aliases(target)
                ),
                root_prefixes=frozenset(_bias_prefix_root(prefix) for prefix in full_prefixes if prefix),
                full_prefixes=full_prefixes,
                exact_feature_prefixes=exact_feature_prefixes,
                scan_feature_prefixes=tuple(
                    prefix for prefix in full_prefixes if prefix not in exact_feature_prefixes
                ),
                score_bonus=float(bias.score_bonus),
                novelty_bonus=float(bias.novelty_bonus),
                contribution_bonus=float(bias.contribution_bonus),
                candidate_pool_bonus=float(bias.candidate_pool_bonus),
                keep_in_pool=bool(bias.keep_in_pool),
            )
        )
    return tuple(compiled)


def _normalize_bias_prefix(prefix: str) -> str:
    return str(prefix).strip()


def _is_exact_feature_prefix(prefix: str) -> bool:
    text = _normalize_bias_prefix(prefix)
    return text.endswith(":") and text.count(":") == 1


def _bias_prefix_root(prefix: str) -> str:
    text = _normalize_bias_prefix(prefix).rstrip(":")
    if not text:
        return ""
    return canonical_semantic_prefix(text.split(":", 1)[0])


def _apply_discovery_biases(
    analysis: CaseAnalysis,
    discovery_biases: tuple[CompiledDiscoveryBias, ...],
) -> tuple[float, float, float, float, bool, list[str]]:
    score_bonus = 0.0
    novelty_bonus = 0.0
    contribution_bonus = 0.0
    pool_bonus = 0.0
    keep_in_pool = False
    hits: list[str] = []
    for bias in discovery_biases:
        if not _compiled_discovery_bias_matches(analysis, bias):
            continue
        hits.append(bias.label)
        score_bonus += bias.score_bonus
        novelty_bonus += bias.novelty_bonus
        contribution_bonus += bias.contribution_bonus
        pool_bonus += bias.candidate_pool_bonus
        keep_in_pool = keep_in_pool or bias.keep_in_pool
    return score_bonus, novelty_bonus, contribution_bonus, pool_bonus, keep_in_pool, hits


def _compiled_discovery_bias_matches(
    analysis: CaseAnalysis,
    bias: CompiledDiscoveryBias,
) -> bool:
    target_match = not bias.targets or bool(
        bias.targets & analysis.matched_target_set
        or bias.target_aliases & (analysis.matched_target_aliases | analysis.expanded_features)
    )
    if not target_match:
        return False
    if not bias.full_prefixes:
        return True
    if not (analysis.expanded_feature_prefix_roots & bias.root_prefixes):
        return False
    if analysis.expanded_feature_prefixes & bias.exact_feature_prefixes:
        return True
    return any(
        any(_feature_has_prefix(feature, prefix) for feature in analysis.expanded_features)
        for prefix in bias.scan_feature_prefixes
    )


def _discovery_bias_label(bias: DiscoveryBias) -> str:
    target_part = "+".join(bias.targets) if bias.targets else "any-target"
    prefix_part = "+".join(bias.feature_prefixes) if bias.feature_prefixes else "any-feature"
    return f"{target_part}|{prefix_part}"


def _prefer_unsaturated_decisions(
    current: list[GuidanceDecision],
    scored: list[GuidanceDecision],
    is_saturated: Any,
) -> list[GuidanceDecision]:
    preferred = [decision for decision in current if not is_saturated(decision)]
    if preferred:
        return preferred
    fallback = [decision for decision in scored if not is_saturated(decision)]
    return fallback or current


def _apply_candidate_pool_discovery_balance(
    scored: list[GuidanceDecision],
    discovery_bucket_counts: Counter[str],
    recent_discovery_bucket_counts: Counter[str],
) -> None:
    if len(scored) <= 1:
        return
    pool_counts: Counter[str] = Counter()
    for decision in scored:
        pool_counts.update(decision.discovery_buckets)
    shared_bucket_factors = _candidate_pool_bucket_factors(
        pool_counts,
        discovery_bucket_counts,
        recent_discovery_bucket_counts,
    )
    for decision in scored:
        analysis = decision.analysis
        shared_bonus = _candidate_pool_discovery_bonus_from_weights(
            _decision_discovery_bucket_weights(decision),
            shared_bucket_factors,
            bucket_count_sqrt=(
                analysis.discovery_bucket_count_sqrt
                if analysis is not None
                else None
            ),
        )
        bias_bonus = decision.candidate_pool_bias_bonus_metric
        bonus = shared_bonus + bias_bonus
        decision.score += bonus
        decision.candidate_pool_shared_bonus_metric = shared_bonus
        decision.candidate_pool_diversity_bonus_metric = bonus
        if decision.score_breakdown:
            decision.score_breakdown["candidate_pool_shared_bonus"] = shared_bonus
            decision.score_breakdown["candidate_pool_diversity_bonus"] = bonus


def _candidate_pool_bucket_factors(
    pool_counts: Counter[str],
    discovery_bucket_counts: Counter[str],
    recent_discovery_bucket_counts: Counter[str],
) -> dict[str, float]:
    discovery_counts_get = discovery_bucket_counts.get
    recent_counts_get = recent_discovery_bucket_counts.get
    factors: dict[str, float] = {}
    for bucket, pool_count in pool_counts.items():
        factors[bucket] = _candidate_pool_bucket_factor(
            pool_count,
            discovery_counts_get(bucket, 0),
            recent_counts_get(bucket, 0),
        )
    return factors


def _candidate_pool_discovery_bonus_from_weights(
    discovery_bucket_weights: tuple[tuple[str, float], ...],
    shared_bucket_factors: dict[str, float],
    *,
    bucket_count_sqrt: float | None = None,
) -> float:
    if not discovery_bucket_weights:
        return 0.0
    shared_factors_get = shared_bucket_factors.get
    weighted = 0.0
    for bucket, bucket_weight in discovery_bucket_weights:
        weighted += bucket_weight * shared_factors_get(bucket, 0.0)
    count_sqrt = bucket_count_sqrt if bucket_count_sqrt is not None else math.sqrt(len(discovery_bucket_weights))
    return min(0.80, (weighted / count_sqrt) * 0.18)


def _matched_targets(
    features: set[str],
    targets: list[str],
    *,
    expanded_features: frozenset[str] | set[str] | None,
) -> list[str]:
    if expanded_features is None:
        resolved_expanded_features = _alias_expanded_features(features)
    else:
        resolved_expanded_features = set(expanded_features)
    matched = []
    for target in targets:
        normalized_target = str(target).strip()
        if not normalized_target:
            continue
        if normalized_target in resolved_expanded_features:
            matched.append(target)
            continue
        if f"semantic_family:{normalized_target}" in resolved_expanded_features:
            matched.append(target)
            continue
        if normalized_target.startswith("semantic_signal:"):
            signal = normalized_target.removeprefix("semantic_signal:").strip()
            if signal and _semantic_signal_feature(signal) in resolved_expanded_features:
                matched.append(target)
                continue
        if target == "unicode_case_mapping":
            if (
                "has:unicode_string" in resolved_expanded_features
                and resolved_expanded_features & {"expr:string_lower", "expr:string_upper"}
            ):
                matched.append(target)
            continue
        required = _target_aliases(target)
        if resolved_expanded_features & required:
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


def _target_template_bonus_from_features(
    template_features: tuple[str, ...] | list[str],
    feature_counts: Counter[str],
) -> float:
    if not template_features:
        return 0.0
    feature_counts_get = feature_counts.get
    bonus = 0.0
    for feature in template_features:
        bonus += TEMPLATE_TARGET_BONUS / math.sqrt(1.0 + feature_counts_get(feature, 0))
    return bonus


def _target_no_yield_penalty(
    features: set[str],
    matched_targets: list[str],
    feature_counts: Counter[str],
    finding_feature_counts: Counter[str],
    *,
    expanded_features: frozenset[str] | set[str] | None = None,
    target_penalty_specs: tuple[TargetPenaltySpec, ...] | None = None,
) -> float:
    resolved_expanded_features = (
        expanded_features
        if expanded_features is not None
        else _alias_expanded_features(features)
    )
    penalty = 0.0
    if target_penalty_specs is None:
        resolved_target_penalty_specs = tuple(
            TargetPenaltySpec(
                aliases=frozenset(matched_aliases),
                weight=(0.30 if _is_specific_target(target) else 0.08),
            )
            for target in matched_targets
            if (matched_aliases := resolved_expanded_features & _target_aliases(target))
        )
    else:
        resolved_target_penalty_specs = target_penalty_specs
    feature_counts_get = feature_counts.get
    finding_feature_counts_get = finding_feature_counts.get
    for spec in resolved_target_penalty_specs:
        matched_aliases = spec.aliases
        if not matched_aliases:
            continue
        pulls = 0
        for feature in matched_aliases:
            count = feature_counts_get(feature, 0)
            if count > pulls:
                pulls = count
        if pulls < 6:
            continue
        finding_hits = 0
        for feature in matched_aliases:
            count = finding_feature_counts_get(feature, 0)
            if count > finding_hits:
                finding_hits = count
        stale_pulls = max(0, pulls - (3 * finding_hits) - 5)
        if stale_pulls <= 0:
            continue
        penalty += math.log1p(stale_pulls) * spec.weight
    return min(4.0, penalty)


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
    template_features = tuple(
        feature
        for target in matched_targets
        for feature in (f"mixed_generator_profile:{target}", f"generator_profile:{target}")
        if feature in features
    )
    return _target_template_bonus_from_features(template_features, feature_counts)


def _discovery_buckets(
    features: set[str],
    matched_targets: list[str],
    frontier_buckets: list[str],
) -> list[str]:
    buckets: list[str] = []
    for target in matched_targets:
        buckets.append(f"target:{target}")
    for feature in sorted(features):
        if (
            feature.startswith("opseq:")
            or feature.startswith("agg:")
            or feature.startswith("cast:")
            or feature.startswith("combo:")
            or feature.startswith("combo_risk:")
            or feature.startswith("semantic_signal:")
            or feature.startswith("common_api_template:")
            or feature.startswith("cmp:")
            or feature.startswith("date_part:")
            or feature.startswith("expr:")
            or feature.startswith("filter:")
            or feature.startswith("generator_profile:")
            or feature.startswith("group_key_type:")
            or feature.startswith("join:")
            or feature.startswith("mixed_generator_profile:")
            or feature.startswith("mutation_op:")
            or feature.startswith("semantic_family:")
            or feature.startswith("source_issue:")
            or feature.startswith("pattern:")
            or feature.startswith("materialization:")
            or feature.startswith("sort:")
        ):
            buckets.append(feature)
    buckets.extend(f"frontier:{bucket}" for bucket in frontier_buckets)
    return unique_preserve_order(buckets)


def _row_has_discovery_signal(
    row: dict[str, Any],
    reward: float,
    *,
    known_saturated_bug_families: list[str],
    finding_outcomes: FindingOutcomeAnalysis | None = None,
) -> bool:
    signals = row_reward_signals(
        row,
        known_saturated_bug_families=known_saturated_bug_families,
        finding_outcomes=finding_outcomes,
    )
    resolved_semantic_only = (
        bool(signals["resolved_semantic_divergence_count"])
        and not bool(signals["candidate_bug"])
        and not bool(signals["rewardable_semantic_divergence"])
    )
    if row.get("signal_new_behavior", row.get("is_new_behavior")) and not resolved_semantic_only:
        return True
    return bool(signals["candidate_bug"] or signals["needs_confirmation"] or reward >= 1.0)


def _discovery_metrics(
    discovery_bucket_weights: tuple[tuple[str, float], ...],
    discovery_bucket_counts: Counter[str],
    discovery_bucket_signal_counts: Counter[str],
    recent_discovery_stale_counts: Counter[str],
    recent_discovery_signal_counts: Counter[str],
    *,
    recent_window_count: int,
    bucket_metric_cache: dict[str, tuple[float, float, float]] | None = None,
) -> tuple[float, float, float]:
    if not discovery_bucket_weights:
        return 0.0, 0.0, 0.0
    weighted_diversity = 0.0
    weighted_stale_penalty = 0.0
    weighted_recent_loop_penalty = 0.0
    recent_loop_enabled = recent_window_count >= 3
    bucket_count_sqrt = math.sqrt(len(discovery_bucket_weights))
    discovery_counts_get = discovery_bucket_counts.get
    signal_counts_get = discovery_bucket_signal_counts.get
    recent_stale_get = recent_discovery_stale_counts.get
    recent_signal_get = recent_discovery_signal_counts.get
    for bucket, bucket_weight in discovery_bucket_weights:
        metrics = bucket_metric_cache.get(bucket) if bucket_metric_cache is not None else None
        if metrics is None:
            metrics = _discovery_bucket_metric_components(
                discovery_counts_get(bucket, 0),
                signal_counts_get(bucket, 0),
                recent_stale_get(bucket, 0),
                recent_signal_get(bucket, 0),
                recent_loop_enabled,
            )
            if bucket_metric_cache is not None:
                bucket_metric_cache[bucket] = metrics
        diversity_factor, stale_penalty_base, recent_loop_penalty_base = metrics
        weighted_diversity += bucket_weight * diversity_factor
        weighted_stale_penalty += stale_penalty_base * bucket_weight
        weighted_recent_loop_penalty += recent_loop_penalty_base * bucket_weight
    diversity_bonus = min(1.50, (weighted_diversity / bucket_count_sqrt) * 0.45)
    stale_penalty = min(4.0, (weighted_stale_penalty / bucket_count_sqrt) * 0.40)
    recent_loop_penalty = min(2.50, (weighted_recent_loop_penalty / bucket_count_sqrt) * 0.55)
    return diversity_bonus, stale_penalty, recent_loop_penalty


def _discovery_diversity_bonus(
    discovery_buckets: list[str],
    discovery_bucket_counts: Counter[str],
) -> float:
    diversity_bonus, _, _ = _discovery_metrics(
        _discovery_bucket_weights_for_buckets(discovery_buckets),
        discovery_bucket_counts,
        Counter(),
        Counter(),
        Counter(),
        recent_window_count=0,
    )
    return diversity_bonus


def _discovery_stale_penalty(
    discovery_buckets: list[str],
    discovery_bucket_counts: Counter[str],
    discovery_bucket_signal_counts: Counter[str],
) -> float:
    _, stale_penalty, _ = _discovery_metrics(
        _discovery_bucket_weights_for_buckets(discovery_buckets),
        discovery_bucket_counts,
        discovery_bucket_signal_counts,
        Counter(),
        Counter(),
        recent_window_count=0,
    )
    return stale_penalty


def _recent_discovery_loop_penalty(
    discovery_buckets: list[str],
    recent_discovery_stale_counts: Counter[str],
    recent_discovery_signal_counts: Counter[str],
    *,
    recent_window_count: int,
) -> float:
    _, _, recent_loop_penalty = _discovery_metrics(
        _discovery_bucket_weights_for_buckets(discovery_buckets),
        Counter(),
        Counter(),
        recent_discovery_stale_counts,
        recent_discovery_signal_counts,
        recent_window_count=recent_window_count,
    )
    return recent_loop_penalty


def _discovery_bucket_weight(bucket: str) -> float:
    return _discovery_bucket_weight_cached(bucket)


def _discovery_bucket_weights_for_buckets(
    discovery_buckets: list[str],
) -> tuple[tuple[str, float], ...]:
    return tuple((bucket, _discovery_bucket_weight(bucket)) for bucket in discovery_buckets)


def _decision_discovery_bucket_weights(
    decision: GuidanceDecision,
) -> tuple[tuple[str, float], ...]:
    analysis = decision.analysis
    if analysis is not None:
        return analysis.discovery_bucket_weights
    return _discovery_bucket_weights_for_buckets(decision.discovery_buckets)


def _target_match_weight(target: str, *, generic_weight: float) -> float:
    if _is_specific_target(target):
        return PATTERN_TARGET_WEIGHT
    return generic_weight


def _specific_target_count(matched_targets: list[str]) -> int:
    return sum(1 for target in matched_targets if _is_specific_target(target))


def _is_specific_target(target: str) -> bool:
    if f"semantic_family:{target}" in _target_aliases(target):
        return True
    return any(feature.startswith("pattern:") for feature in _target_aliases(target))


def _bucket(prefix: str, value: int, limits: list[tuple[int, str]], fallback: str) -> str:
    for limit, name in limits:
        if value <= limit:
            return f"{prefix}:{name}"
    return f"{prefix}:{fallback}"


def _quality_archive_context_features(context: Any) -> set[str]:
    if not isinstance(context, dict) or not context:
        return set()
    prefix = "quality_archive"
    features = {
        f"{prefix}:{'known' if bool(context.get('archive_known', False)) else 'unknown'}",
        _bucket(f"{prefix}:seed_count", _as_nonnegative_int(context.get("archive_seed_count")), [(0, "zero"), (1, "one"), (3, "few")], "many"),
        _bucket(f"{prefix}:outcome_count", _as_nonnegative_int(context.get("archive_outcome_count")), [(0, "zero"), (2, "few"), (8, "some")], "many"),
        _bucket(f"{prefix}:cluster_count", _as_nonnegative_int(context.get("cluster_count")), [(0, "zero"), (1, "one"), (4, "few")], "many"),
        _bucket(f"{prefix}:recent_cluster_pulls", _as_nonnegative_int(context.get("recent_cluster_pulls")), [(0, "zero"), (2, "few"), (6, "some")], "many"),
    }
    features.add(f"{prefix}:reward:{_signed_signal_bucket(context.get('archive_cluster_reward'))}")
    features.add(f"{prefix}:feedback_reward:{_signed_signal_bucket(context.get('cluster_feedback_reward'))}")
    features.add(f"{prefix}:health:{_nonnegative_signal_bucket(context.get('archive_health_penalty'))}")
    features.add(f"{prefix}:novelty:{_nonnegative_signal_bucket(context.get('cluster_novelty_score'))}")
    elite_indexes = context.get("archive_elite_indexes")
    features.add(f"{prefix}:elite:{'present' if isinstance(elite_indexes, list) and elite_indexes else 'absent'}")
    return features


def _as_nonnegative_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _as_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _signed_signal_bucket(value: Any) -> str:
    number = _as_float(value)
    if number <= -0.5:
        return "negative"
    if number < -0.05:
        return "weak_negative"
    if number < 0.05:
        return "neutral"
    if number < 0.5:
        return "weak_positive"
    return "positive"


def _nonnegative_signal_bucket(value: Any) -> str:
    number = max(0.0, _as_float(value))
    if number <= 0.0:
        return "zero"
    if number < 0.15:
        return "low"
    if number < 0.5:
        return "medium"
    return "high"


def _literal_feature_type(*values: Any) -> str:
    types = set()
    for value in values:
        if value is None:
            types.add("null")
        elif isinstance(value, bool):
            types.add("bool")
        elif isinstance(value, int):
            types.add("int")
        elif isinstance(value, float):
            types.add("float")
        elif isinstance(value, str):
            types.add("str")
        else:
            types.add("other")
    if len(types) == 1:
        return next(iter(types))
    if types <= {"int", "float"}:
        return "numeric"
    return "mixed"


@lru_cache(maxsize=None)
def _bounded_finding_signal(count: int) -> float:
    if count <= 0:
        return 0.0
    return min(math.log1p(count), 2.0) / (1.0 + count / 50.0)


@lru_cache(maxsize=None)
def _feature_saturation(count: int) -> float:
    if count <= 25:
        return 0.0
    return math.log1p(count - 25)


@lru_cache(maxsize=None)
def _root_saturation(count: int) -> float:
    if count <= 6:
        return 0.0
    return math.log1p(count - 6) * 0.45


@lru_cache(maxsize=None)
def _candidate_pool_bucket_factor(
    pool_count: int,
    discovery_count: int,
    recent_count: int,
) -> float:
    return (
        1.0
        / math.sqrt(max(1, pool_count))
        / math.sqrt(1.0 + discovery_count)
        / math.sqrt(1.0 + recent_count)
    )


@lru_cache(maxsize=None)
def _discovery_bucket_metric_components(
    pulls: int,
    signal_hits: int,
    stale_window_pulls: int,
    recent_signal_hits: int,
    recent_loop_enabled: bool,
) -> tuple[float, float, float]:
    diversity_factor = 1.0 / math.sqrt(1.0 + pulls)
    stale_penalty_base = 0.0
    if pulls >= 6:
        stale_pulls = max(0, pulls - (3 * signal_hits) - 5)
        if stale_pulls > 0:
            stale_penalty_base = math.log1p(stale_pulls)
    recent_loop_penalty_base = 0.0
    if recent_loop_enabled and stale_window_pulls >= 3:
        repeated_pulls = max(0, stale_window_pulls - (2 * recent_signal_hits) - 2)
        if repeated_pulls > 0:
            recent_loop_penalty_base = math.log1p(repeated_pulls)
    return diversity_factor, stale_penalty_base, recent_loop_penalty_base


def _resolved_semantic_boundary_penalty(
    features: set[str],
    predicted_roots: set[str],
    matched_targets: list[str],
) -> float:
    if any(target in RESOLVED_SEMANTIC_BOUNDARY_TARGETS for target in matched_targets):
        return 0.0
    penalty = 0.0
    if "unicode_case_mapping" in predicted_roots:
        penalty += 2.0
    if "has:special_float" in features or "nan_inf_semantics" in predicted_roots:
        penalty += 1.25
    if "agg:precision-float" in features:
        penalty += 1.25
    if "mutate:arith:mod" in features:
        penalty += 1.25
    return min(3.0, penalty)


_EMPTY_ROOTS: frozenset[str] = frozenset()


def _predicted_family_saturation_state(
    predicted_roots: set[str],
    *,
    root_hit_counts: Counter[str],
    known_saturated_roots: set[str],
    threshold: int,
    penalty_weight: float,
) -> tuple[float, bool]:
    if threshold <= 0:
        return 0.0, False
    penalty = 0.0
    active = False
    positive_penalty_weight = max(0.0, penalty_weight)
    for root in predicted_roots:
        hit_count = _predicted_family_hit_count(
            root,
            root_hit_counts=root_hit_counts,
            known_saturated_roots=known_saturated_roots,
            threshold=threshold,
        )
        if hit_count >= threshold:
            active = True
            if positive_penalty_weight > 0.0:
                penalty += _family_saturation_penalty(
                    hit_count,
                    threshold=threshold,
                    penalty_weight=positive_penalty_weight,
                )
    return penalty, active


def _predicted_family_saturation_penalty(
    predicted_roots: set[str],
    *,
    root_hit_counts: Counter[str],
    known_saturated_roots: set[str],
    threshold: int,
    penalty_weight: float,
) -> float:
    return _predicted_family_saturation_state(
        predicted_roots,
        root_hit_counts=root_hit_counts,
        known_saturated_roots=known_saturated_roots,
        threshold=threshold,
        penalty_weight=penalty_weight,
    )[0]


def _has_predicted_saturated_family(
    predicted_roots: set[str],
    *,
    root_hit_counts: Counter[str],
    known_saturated_roots: set[str],
    threshold: int,
) -> bool:
    return _predicted_family_saturation_state(
        predicted_roots,
        root_hit_counts=root_hit_counts,
        known_saturated_roots=known_saturated_roots,
        threshold=threshold,
        penalty_weight=0.0,
    )[1]


def _predicted_family_hit_count(
    root: str,
    *,
    root_hit_counts: Counter[str],
    known_saturated_roots: set[str],
    threshold: int,
    include_known_families: bool = True,
) -> int:
    dynamic_hits = root_hit_counts[root]
    known_hits = (
        threshold
        if include_known_families and root in known_saturated_roots else 0
    )
    return max(dynamic_hits, known_hits)


def _issue_inspired_candidate_source_issue_keys(findings: list[dict[str, Any]]) -> Counter[str]:
    keys: Counter[str] = Counter()
    for finding in findings:
        if not is_candidate_issue_finding(finding):
            continue
        if str(finding.get("discovery_origin", "")).strip() != "issue_inspired":
            continue
        source_key = _source_issue_key(finding.get("source_issue"))
        if source_key:
            keys[source_key] += 1
    return keys


def _issue_inspired_case_source_issue_keys(features: set[str]) -> Counter[str]:
    keys: Counter[str] = Counter()
    if "source:issue_inspired" not in features:
        return keys
    for feature in features:
        if feature.startswith("source_issue:"):
            source_key = _source_issue_key(feature.split(":", 1)[1])
            if source_key:
                keys[source_key] += 1
    return keys


def _max_source_issue_hits(features: set[str], *, source_counts: Counter[str]) -> int:
    hits = [
        source_counts[feature.split(":", 1)[1]]
        for feature in features
        if feature.startswith("source_issue:")
    ]
    return max(hits, default=0)


def _max_source_issue_hits_for_keys(
    source_keys: tuple[str, ...] | list[str],
    *,
    source_counts: Counter[str],
) -> int:
    return max((source_counts[source_key] for source_key in source_keys), default=0)


def _family_saturation_penalty(count: int, *, threshold: int, penalty_weight: float) -> float:
    if count < threshold:
        return 0.0
    return max(0.0, penalty_weight) * (1.0 + math.log1p(count - threshold))


def _is_globally_saturated(count: int, *, threshold: int) -> bool:
    return threshold > 0 and count >= threshold


def _profile_saturation(count: int) -> float:
    if count <= 3:
        return 0.0
    return min(4.0, math.log1p(count - 3) * 1.15)


_targeted_decision_key = _contrast_decision_key
_decision_hits_family_diversity_guard = _decision_has_family_saturation
_predicted_family_diversity_guard_penalty = _predicted_family_saturation_penalty
_family_diversity_guard_penalty = _family_saturation_penalty
_family_saturation = _family_saturation_penalty


def _guidance_reward(
    row: dict[str, Any],
    *,
    root_cause_counts: Counter[str] | None = None,
    candidate_bug_family_counts: Counter[str] | None = None,
    candidate_bug_signature_counts: Counter[str] | None = None,
    enable_family_saturation: bool = True,
    family_saturation_threshold: int = 8,
    saturated_family_reward: float = 0.02,
    known_saturated_bug_families: list[str] | None = None,
    finding_outcomes: FindingOutcomeAnalysis | None = None,
) -> float:
    findings = row.get("findings") or []
    known_families = known_saturated_bug_families or []
    outcomes = (
        finding_outcomes
        if finding_outcomes is not None
        else (
            analyze_finding_outcomes(
                findings,
                known_saturated_bug_families=known_families,
            )
            if findings
            else None
        )
    )
    signals = row_reward_signals(
        row,
        known_saturated_bug_families=known_families,
        finding_outcomes=outcomes,
    )
    root_counts = (
        root_cause_counts
        if isinstance(root_cause_counts, Counter)
        else Counter(root_cause_counts or {})
    )
    family_counts = (
        candidate_bug_family_counts
        if isinstance(candidate_bug_family_counts, Counter)
        else Counter(candidate_bug_family_counts or {})
    )
    signature_counts = (
        candidate_bug_signature_counts
        if isinstance(candidate_bug_signature_counts, Counter)
        else Counter(candidate_bug_signature_counts or {})
    )
    reward = (
        _candidate_issue_guidance_reward(
            findings,
            root_cause_counts=root_counts,
            candidate_bug_family_counts=family_counts,
            candidate_bug_signature_counts=signature_counts,
            enable_family_saturation=enable_family_saturation,
            family_saturation_threshold=family_saturation_threshold,
            saturated_family_reward=saturated_family_reward,
            known_saturated_bug_families=known_families,
            finding_outcomes=outcomes,
        )
        + 0.20 * signals["needs_confirmation_count"]
        + (0.5 if row.get("signal_new_behavior", row.get("is_new_behavior")) else 0.0)
        - 0.15 * signals["resolved_semantic_divergence_count"]
        - 2.5 * signals["false_positive_count"]
    )
    preflight = row.get("preflight") or {}
    if not bool(preflight.get("valid", True)) or bool(preflight.get("fallback_used", False)):
        reward -= 0.5
    feedback_summary = row.get("feedback_summary")
    if not isinstance(feedback_summary, dict) and row.get("quality_oracles"):
        feedback_summary = feedback_summary_for_case(
            row,
            known_saturated_bug_families=known_families,
            finding_outcomes=outcomes,
        )
    if isinstance(feedback_summary, dict):
        reward += float(feedback_summary.get("guidance_reward_adjustment", 0.0) or 0.0)
    if reward == 0.0:
        reward -= 0.1
    return reward


def _candidate_issue_guidance_reward(
    findings: list[dict[str, Any]],
    *,
    root_cause_counts: Counter[str],
    candidate_bug_family_counts: Counter[str],
    candidate_bug_signature_counts: Counter[str],
    enable_family_saturation: bool,
    family_saturation_threshold: int,
    saturated_family_reward: float,
    known_saturated_bug_families: list[str],
    finding_outcomes: FindingOutcomeAnalysis | None = None,
) -> float:
    outcomes = (
        finding_outcomes
        if finding_outcomes is not None
        else (
            analyze_finding_outcomes(
                findings,
                known_saturated_bug_families=known_saturated_bug_families,
            )
            if findings
            else None
        )
    )
    families = outcomes.candidate_bug_families if outcomes is not None else ()
    signatures = outcomes.candidate_bug_signatures if outcomes is not None else ()
    if families:
        reward = 0.0
        for family in families:
            root = family.split("@", 1)[0]
            previous_hits = max(candidate_bug_family_counts[family], root_cause_counts[root])
            if enable_family_saturation and _family_key_matches_known_family(
                family,
                known_saturated_bug_families,
            ):
                previous_hits = max(previous_hits, family_saturation_threshold)
            reward += _candidate_issue_novelty_reward(
                previous_hits,
                enable_family_saturation=enable_family_saturation,
                family_saturation_threshold=family_saturation_threshold,
                saturated_family_reward=saturated_family_reward,
            )
        if signatures:
            duplicate_signatures = sum(1 for signature in signatures if candidate_bug_signature_counts[signature] > 0)
            if duplicate_signatures == len(signatures):
                reward *= 0.35
            elif duplicate_signatures:
                reward *= 0.75
        return reward
    reward = 0.0
    for finding in findings:
        if not is_rewardable_candidate_issue_finding(finding, known_saturated_bug_families):
            continue
        root = str(finding.get("root_cause", "unknown"))
        reward += _candidate_issue_novelty_reward(
            root_cause_counts[root],
            enable_family_saturation=enable_family_saturation,
            family_saturation_threshold=family_saturation_threshold,
            saturated_family_reward=saturated_family_reward,
        )
    return reward


def _candidate_issue_novelty_reward(
    previous_hits: int,
    *,
    enable_family_saturation: bool = True,
    family_saturation_threshold: int = 8,
    saturated_family_reward: float = 0.02,
) -> float:
    if enable_family_saturation and family_saturation_threshold > 0 and previous_hits >= family_saturation_threshold:
        return max(0.0, saturated_family_reward)
    if previous_hits <= 0:
        return 4.0
    if previous_hits <= 2:
        return 1.5
    if previous_hits <= 8:
        return 0.75 / math.sqrt(previous_hits)
    return 0.10


def _family_key_matches_root_backend(
    family_key: str,
    root: str,
    active_backends: list[str],
) -> bool:
    family_root, family_backends = _split_family_key(family_key)
    return family_root == root and _family_backends_active(family_backends, active_backends)


def _family_key_matches_known_family(candidate_family: str, known_saturated_bug_families: list[str]) -> bool:
    candidate_root, candidate_backends = _split_family_key(candidate_family)
    for known_family in known_saturated_bug_families:
        known_root, known_backends = _split_family_key(known_family)
        if known_root != candidate_root:
            continue
        if not known_backends or not candidate_backends or known_backends & candidate_backends:
            return True
    return False


def _family_backends_active(family_backends: set[str], active_backends: list[str]) -> bool:
    active = _normalized_active_backends(active_backends)
    if not active or not family_backends or family_backends == {"unknown"}:
        return True
    return bool(family_backends & active)


def _normalized_active_backends(active_backends: list[str] | frozenset[str] | set[str]) -> frozenset[str]:
    return frozenset(str(backend).strip() for backend in active_backends if str(backend).strip())


def _family_root_for_active_backends(
    family_key: str,
    *,
    active_backends: frozenset[str],
) -> str:
    root, family_backends = _split_family_key(family_key)
    if not root:
        return ""
    if not active_backends or not family_backends or family_backends == {"unknown"}:
        return root
    return root if family_backends & active_backends else ""


def _root_hit_counts_for_active_backends(
    family_counts: Counter[str],
    *,
    active_backends: frozenset[str],
) -> Counter[str]:
    root_hits: Counter[str] = Counter()
    for family_key, count in family_counts.items():
        hits = int(count or 0)
        if hits <= 0:
            continue
        root = _family_root_for_active_backends(family_key, active_backends=active_backends)
        if root:
            root_hits[root] += hits
    return root_hits


def _family_roots_for_active_backends(
    family_keys: list[str],
    *,
    active_backends: frozenset[str],
) -> set[str]:
    roots: set[str] = set()
    for family_key in family_keys:
        root = _family_root_for_active_backends(family_key, active_backends=active_backends)
        if root:
            roots.add(root)
    return roots


def _split_family_key(family_key: str) -> tuple[str, set[str]]:
    root, _, backend_part = str(family_key).partition("@")
    backends = {
        backend.strip()
        for backend in backend_part.split(",")
        if backend.strip()
    }
    return root.strip(), backends


def _finding_feature_weight(feature: str, online_weights: OnlineFeatureWeights | None = None) -> float:
    return _finding_feature_weight_base(feature) * _online_multiplier(feature, online_weights)


def _path_coverage_proxy(
    path_feature_weight_bases: tuple[tuple[str, float], ...],
    feature_counts: Counter[str],
    *,
    path_feature_count_sqrt: float,
    path_operation_diversity_bonus: float,
    path_sequence_bonus: float,
    online_weights: OnlineFeatureWeights | None = None,
    multipliers: dict[str, float] | None = None,
) -> float:
    if not path_feature_weight_bases:
        return 0.0
    feature_counts_get = feature_counts.get
    novelty = 0.0
    if multipliers is not None:
        multipliers_get = multipliers.get
        for feature, base_weight in path_feature_weight_bases:
            novelty += (base_weight * multipliers_get(feature, 1.0)) / (1.0 + feature_counts_get(feature, 0))
    else:
        for feature, base_weight in path_feature_weight_bases:
            novelty += (base_weight * _online_multiplier(feature, online_weights)) / (
                1.0 + feature_counts_get(feature, 0)
            )
    return novelty / path_feature_count_sqrt + path_operation_diversity_bonus + path_sequence_bonus


def _data_sensitivity_score(
    data_feature_weight_bases: tuple[tuple[str, float], ...],
    feature_counts: Counter[str],
    *,
    data_feature_count_sqrt: float,
    online_weights: OnlineFeatureWeights | None = None,
    multipliers: dict[str, float] | None = None,
) -> float:
    if not data_feature_weight_bases:
        return 0.0
    feature_counts_get = feature_counts.get
    weighted = 0.0
    if multipliers is not None:
        multipliers_get = multipliers.get
        for feature, base_weight in data_feature_weight_bases:
            count = feature_counts_get(feature, 0)
            weighted += (base_weight * multipliers_get(feature, 1.0)) * (1.0 + 1.0 / (1.0 + count))
    else:
        for feature, base_weight in data_feature_weight_bases:
            count = feature_counts_get(feature, 0)
            weighted += (base_weight * _online_multiplier(feature, online_weights)) * (1.0 + 1.0 / (1.0 + count))
    return weighted / data_feature_count_sqrt * 0.35


def _saturation_feature_weight(feature: str, online_weights: OnlineFeatureWeights | None = None) -> float:
    return _saturation_feature_weight_base(feature) * _online_multiplier(feature, online_weights)


def _is_path_feature(feature: str) -> bool:
    return _is_path_feature_cached(feature)


def _path_feature_weight(feature: str, online_weights: OnlineFeatureWeights | None = None) -> float:
    return _path_feature_weight_base(feature) * _online_multiplier(feature, online_weights)


def _is_data_sensitivity_feature(feature: str) -> bool:
    return _is_data_sensitivity_feature_cached(feature)


def _data_feature_weight(feature: str, online_weights: OnlineFeatureWeights | None = None) -> float:
    return _data_feature_weight_base(feature) * _online_multiplier(feature, online_weights)


def _online_multiplier(feature: str, online_weights: OnlineFeatureWeights | None) -> float:
    if online_weights is None:
        return 1.0
    return online_weights.multiplier(feature)


@lru_cache(maxsize=None)
def _is_path_feature_cached(feature: str) -> bool:
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
            "cast:",
            "cast_to:",
            "cast_domain:",
            "group_key_type:",
            "agg:",
            "op_count:",
            "groupby:",
            "running:",
            "running_source_type:",
            "combo:",
            "combo_frequency:",
            "combo_risk:",
            "semantic_signal:",
            "semantic_family:",
            "exploration_objective:",
        )
    )


@lru_cache(maxsize=None)
def _path_feature_weight_base(feature: str) -> float:
    if feature.startswith("opseq:"):
        return 1.4
    if feature.startswith("pattern:"):
        return 1.6
    if feature.startswith("semantic_family:"):
        return 1.3
    if feature.startswith("exploration_objective:"):
        return 1.2
    if feature.startswith("op:"):
        return 1.0
    if feature.startswith(("combo:", "combo_risk:", "semantic_signal:")):
        return 1.1
    if feature.startswith("combo_frequency:"):
        return 0.8
    if feature.startswith(("running:", "running_source_type:")):
        return 1.0
    if feature.startswith(("join:", "agg:", "expr:", "mutate:", "cmp:", "filter:", "group_key_type:")):
        return 0.9
    if feature.startswith(("filter_type:", "select_width:", "sort:", "limit:", "offset:", "cast:", "cast_to:", "cast_domain:", "arith:")):
        return 0.7
    return 0.5


@lru_cache(maxsize=None)
def _is_data_sensitivity_feature_cached(feature: str) -> bool:
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


@lru_cache(maxsize=None)
def _data_feature_weight_base(feature: str) -> float:
    if feature == "groupby:null-key":
        return 1.6
    if feature == "groupby:null-agg-output":
        return 1.7
    if feature == "filter:empty-output":
        return 1.5
    if feature.startswith("offset:"):
        return 1.1
    if feature in {"running:long", "running:small-increment"}:
        return 1.4
    if feature in {"has:special_float", "has:null", "has:unicode_string"}:
        return 1.8
    if feature in {"has:fractional_float", "has:negative_number", "has:empty_string", "has:space_string"}:
        return 1.2
    if feature in {"tables:multi", "rows:empty", "rows:tiny", "cols:wide", "op:limit_zero"}:
        return 1.0
    if feature.startswith(("nullable:", "type:")):
        return 0.6
    if feature.startswith("bool:"):
        return 0.4
    return 0.5


@lru_cache(maxsize=None)
def _finding_feature_weight_base(feature: str) -> float:
    if feature.startswith(("op:", "opseq:")):
        return 1.0
    if feature.startswith(
        ("agg:", "cmp:", "expr:", "mutate:", "join:", "filter:", "filter_type:", "cast:", "cast_to:", "cast_domain:", "combo:", "running:")
    ):
        return 0.75
    if feature.startswith(("has:", "type:", "nullable:")):
        return 0.35
    return 0.50


@lru_cache(maxsize=None)
def _saturation_feature_weight_base(feature: str) -> float:
    if feature.startswith("opseq:"):
        return 1.0
    if feature.startswith("semantic_family:"):
        return 0.8
    if feature.startswith("exploration_objective:"):
        return 0.75
    if feature.startswith("semantic_signal:"):
        return 0.7
    if feature.startswith("op:"):
        return 0.9
    if feature.startswith(("agg:", "cmp:", "expr:", "mutate:", "join:", "filter_type:", "cast:", "cast_to:", "cast_domain:", "combo:", "running:")):
        return 0.65
    if feature.startswith(("has:", "type:", "nullable:")):
        return 0.15
    return 0.25


@lru_cache(maxsize=None)
def _discovery_bucket_weight_cached(bucket: str) -> float:
    if bucket.startswith(("target:", "pattern:", "common_api_template:", "mixed_generator_profile:")):
        return 1.0
    if bucket.startswith(
        (
            "opseq:",
            "frontier:",
            "materialization:",
            "combo_risk:",
            "semantic_signal:",
            "semantic_family:",
            "exploration_objective:",
        )
    ):
        return 0.85
    if bucket.startswith(("agg:", "cast:", "cmp:", "expr:", "filter:", "combo:", "source_issue:", "mutation_op:")):
        return 0.65
    if bucket.startswith(("date_part:", "group_key_type:", "join:", "sort:")):
        return 0.50
    if bucket.startswith("generator_profile:"):
        return 0.45
    return 0.30


def _bounded_confident_mean_reward(total_reward: float, pulls: float, *, max_abs: float) -> float:
    if pulls <= 0:
        return 0.0
    mean_reward = float(total_reward) / float(pulls)
    bounded_mean_reward = max(-max_abs, min(max_abs, mean_reward))
    confidence = min(1.0, math.log1p(float(pulls)) / math.log(4.0))
    return bounded_mean_reward * confidence


def _is_learnable_weight_feature(feature: str) -> bool:
    return (
        _is_path_feature_cached(feature)
        or _is_data_sensitivity_feature_cached(feature)
        or feature.startswith("quality_archive:")
    )


def _feature_prefix(feature: str) -> str:
    canonical = _canonical_feature(feature)
    return canonical.split(":", 1)[0] if ":" in canonical else canonical


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


def _frontier_conformance_from_analysis(analysis: CaseAnalysis, frontier_bucket_counts: Counter[str]) -> float:
    if not analysis.frontier_buckets:
        return analysis.frontier_raw_score
    frontier_counts_get = frontier_bucket_counts.get
    novelty_total = 0.0
    for bucket in analysis.frontier_buckets:
        novelty_total += 1.0 / (1.0 + frontier_counts_get(bucket, 0))
    novelty = (novelty_total / math.sqrt(len(analysis.frontier_buckets))) * 0.35
    return analysis.frontier_raw_score + novelty


def _frontier_signature(case: Case) -> tuple[float, list[str]]:
    if not case.tables:
        return 0.0, []

    table_by_name = {table.name: table for table in case.tables}
    samples = {column.name: [row.get(column.name) for row in case.tables[0].rows] for column in case.tables[0].columns}
    scores: list[float] = []
    buckets: list[str] = []
    last_sort_op: Any | None = None

    for op in case.program.operations:
        kind = op_kind(op)
        if kind == "filter":
            score, op_buckets = _filter_frontier_score(samples, op)
            scores.append(score)
            buckets.extend(op_buckets)
            samples = _filter_output_samples(samples, op)
        elif kind == "tuple_absence_filter":
            right = table_by_name.get(op_table(op))
            score, op_buckets = _tuple_absence_frontier_score(samples, right, op)
            scores.append(score)
            buckets.extend(op_buckets)
            samples = _tuple_absence_output_samples(samples, right, op)
        elif kind == "join":
            right = table_by_name.get(op_table(op))
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
            last_sort_op = {"keys": normalized_order_by_keys(op)}
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
        elif kind == "setop_all_duplicate_probe":
            score, op_buckets, samples = _setop_all_duplicate_frontier_score(op)
            scores.append(score)
            buckets.extend(op_buckets)
            last_sort_op = None
        elif kind == "json_predicate_order_probe":
            score, op_buckets, samples = _json_predicate_order_frontier_score(op)
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
        elif kind == "eval_inplace_alias_probe":
            score, op_buckets, samples = _eval_inplace_alias_frontier_score(op)
            scores.append(score)
            buckets.extend(op_buckets)
            last_sort_op = None
        elif kind == "bool_reduction_skipna_probe":
            score, op_buckets, samples = _bool_reduction_skipna_frontier_score(op)
            scores.append(score)
            buckets.extend(op_buckets)
            last_sort_op = None
        elif kind == "dataset_isin_all_match_probe":
            score, op_buckets, samples = _dataset_isin_all_match_frontier_score(op)
            scores.append(score)
            buckets.extend(op_buckets)
            last_sort_op = None
        elif kind == "run_end_null_compute_probe":
            score, op_buckets, samples = _run_end_null_compute_frontier_score(op)
            scores.append(score)
            buckets.extend(op_buckets)
            last_sort_op = None
        elif kind == "large_string_partition_probe":
            score, op_buckets, samples = _large_string_partition_frontier_score(op)
            scores.append(score)
            buckets.extend(op_buckets)
            last_sort_op = None
        elif kind == "hash_pivot_wider_probe":
            score, op_buckets, samples = _hash_pivot_wider_frontier_score(op)
            scores.append(score)
            buckets.extend(op_buckets)
            last_sort_op = None
        elif kind == "list_flatten_parent_indices_probe":
            score, op_buckets, samples = _list_flatten_parent_indices_frontier_score(op)
            scores.append(score)
            buckets.extend(op_buckets)
            last_sort_op = None
        elif kind == "rolling_mean_by_null_count_probe":
            score, op_buckets, samples = _rolling_mean_by_null_count_frontier_score(op)
            scores.append(score)
            buckets.extend(op_buckets)
            last_sort_op = None
        elif kind == "csv_long_numeric_roundtrip_probe":
            score, op_buckets, samples = _csv_long_numeric_roundtrip_frontier_score(op)
            scores.append(score)
            buckets.extend(op_buckets)
            last_sort_op = None
        elif kind == "select":
            cols = [column for column in op_columns(op) if column in samples]
            samples = {column: samples[column] for column in unique_preserve_order(cols)}
        elif kind == "distinct":
            samples = shared_distinct_output_samples(samples, op)
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
            column = op_column(op)
            if column:
                samples[column] = values
        elif kind == "groupby":
            score, op_buckets = _groupby_frontier_score(samples, op)
            scores.append(score)
            buckets.extend(op_buckets)
            samples = shared_groupby_output_samples(samples, op)
            last_sort_op = None
        elif kind == "aggregate":
            score, op_buckets = _aggregate_frontier_score(samples, op)
            scores.append(score)
            buckets.extend(op_buckets)
            samples = {
                aggregate_alias(agg): []
                for agg in aggregate_specs(op)
                if aggregate_alias(agg)
            }
            last_sort_op = None

    scores = [score for score in scores if score > 0.0]
    return (sum(scores) / len(scores) if scores else 0.0), unique_preserve_order(buckets)


def _contribution_potential(
    features: set[str],
    frontier_buckets: list[str],
    matched_targets: list[str],
    frontier_conformance: float,
    *,
    path_features: tuple[str, ...] | list[str] | None = None,
    data_features: tuple[str, ...] | list[str] | None = None,
    predicted_roots: set[str] | None = None,
    feature_counts: Counter[str],
    frontier_bucket_counts: Counter[str],
    root_cause_counts: Counter[str],
) -> float:
    resolved_path_features = path_features if path_features is not None else [
        feature for feature in features if _is_path_feature(feature)
    ]
    resolved_data_features = data_features if data_features is not None else [
        feature for feature in features if _is_data_sensitivity_feature(feature)
    ]
    feature_counts_get = feature_counts.get
    frontier_counts_get = frontier_bucket_counts.get
    root_cause_counts_get = root_cause_counts.get
    path_novelty = 0
    for feature in resolved_path_features:
        if feature_counts_get(feature, 0) == 0:
            path_novelty += 1
    data_novelty = 0
    for feature in resolved_data_features:
        if feature_counts_get(feature, 0) == 0:
            data_novelty += 1
    frontier_novelty = 0
    for bucket in frontier_buckets:
        if frontier_counts_get(bucket, 0) == 0:
            frontier_novelty += 1
    resolved_predicted_roots = predicted_roots if predicted_roots is not None else _predicted_roots(features)
    root_novelty = 0
    root_saturation = 0
    for root in resolved_predicted_roots:
        root_hits = root_cause_counts_get(root, 0)
        if root_hits == 0:
            root_novelty += 1
        elif root_hits >= 12:
            root_saturation += 1
    return (
        frontier_conformance
        + 0.30 * path_novelty
        + 0.20 * data_novelty
        + 0.45 * frontier_novelty
        + 0.35 * root_novelty
        + 0.60 * len(matched_targets)
        - 0.20 * root_saturation
    )


def _filter_frontier_score(samples: dict[str, list[Any]], op: Any) -> tuple[float, list[str]]:
    column = condition_column(op)
    values = samples.get(column, [])
    comparator = condition_cmp(op)
    literal = condition_value(op)
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
    if parsed is not None and parsed.base in {"in_set", "not_in_set"}:
        buckets.append("filter:set-membership")
        if parsed.base == "not_in_set":
            buckets.append("filter:negative-set-membership")
    if parsed is not None and parsed.base in {"is_null", "is_not_null"}:
        buckets.append("filter:null-predicate")
        buckets.append(f"filter:null-predicate:{parsed.base}")
    if parsed is not None and parsed.base == "bool_predicate":
        buckets.append("filter:boolean-predicate")
        buckets.append(f"filter:boolean-predicate:{parsed.truth_test}")
    if parsed is not None and parsed.base == "range_closed":
        buckets.append("filter:range-closed")
    if parsed is not None and parsed.base in {"str_contains", "str_starts_with", "str_ends_with"}:
        buckets.append("filter:string-pattern")
        buckets.append(f"filter:{parsed.base.replace('str_', 'string-').replace('_', '-')}")
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


def _filter_output_samples(samples: dict[str, list[Any]], op: Any) -> dict[str, list[Any]]:
    column = condition_column(op)
    values = samples.get(column)
    if values is None:
        return samples
    return shared_filter_samples(samples, op)


def _filter_mask(values: list[Any], op: Any) -> list[bool]:
    comparator = condition_cmp(op)
    literal = condition_value(op)
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
    op: Any,
) -> tuple[float, list[str]]:
    buckets = ["filter:tuple-absence"]
    left_columns = op_columns(op)
    right_columns = op_right_columns(op)
    if right is None or not left_columns or len(left_columns) != len(right_columns):
        return 0.0, buckets
    rows = shared_rows_from_samples(samples)
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
    op: Any,
) -> dict[str, list[Any]]:
    left_columns = op_columns(op)
    right_columns = op_right_columns(op)
    if right is None or not left_columns or len(left_columns) != len(right_columns):
        return samples
    rows = shared_rows_from_samples(samples)
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
    op: Any,
) -> tuple[float, list[str]]:
    left_keys, right_keys = join_key_pairs(op)
    if (
        not left_keys
        or len(left_keys) != len(right_keys)
        or any(left_key not in samples for left_key in left_keys)
    ):
        return 0.0, []
    left_values = [
        key for key in (join_key_value(row, left_keys) for row in shared_rows_from_samples(samples)) if key is not None
    ]
    right_values = [] if right is None else [key for key in (join_key_value(row, right_keys) for row in right.rows) if key is not None]
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
    if any(
        any(row.get(left_key) is None for left_key in left_keys)
        for row in shared_rows_from_samples(samples)
    ) or any(
        any(row.get(right_key) is None for right_key in right_keys)
        for row in getattr(right, "rows", [])
    ):
        buckets.append("join:null-keys")
    if len(left_keys) > 1:
        buckets.append("join:multi-key")
    return min(1.0, 0.35 + 0.45 * max(0.0, partial_overlap) + 0.10 * int("join:duplicate-keys" in buckets) + 0.05 * int("join:null-keys" in buckets)), buckets


def _join_output_samples(
    left_samples: dict[str, list[Any]],
    right: Any,
    op: Any,
) -> dict[str, list[Any]]:
    if right is None:
        return left_samples
    right_samples = {
        column.name: [row.get(column.name) for row in getattr(right, "rows", [])]
        for column in getattr(right, "columns", [])
    }
    return shared_join_samples(left_samples, right_samples, op)


def _running_sum_frontier_score(
    samples: dict[str, list[Any]],
    op: Any,
) -> tuple[float, list[str], dict[str, list[Any]]]:
    source = op_source(op)
    column = op_column(op)
    values = samples.get(source, [])
    buckets: list[str] = []
    if not source or not column or source not in samples:
        return 0.0, buckets, samples
    try:
        sort_keys = running_sum_sort_keys(op)
    except ValueError:
        return 0.0, buckets, samples

    ordered_samples = sort_samples_for_running(samples, sort_keys)
    partition_columns = running_sum_partition_columns(op)
    running_values = stable_running_sum_sample_values(ordered_samples, source, partition_columns)
    out_samples = {
        name: list(values)
        for name, values in ordered_samples.items()
        if name != column
    }
    out_samples[column] = running_values

    input_dtype = op_input_dtype(op, "float64")
    buckets.append("running:ordered")
    buckets.append(f"running:{input_dtype}")
    if partition_columns:
        buckets.append("running:partitioned")
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
        + 0.10 * int("running:partitioned" in buckets)
    )
    return min(1.0, score), buckets, out_samples


def _sortedness_frontier_score(
    samples: dict[str, list[Any]],
    op: Any,
    last_sort_op: Any | None,
) -> tuple[float, list[str], dict[str, list[Any]]]:
    column = op_column(op)
    alias = op_output_alias(op)
    values = list(samples.get(column, []))
    ascending = op_ascending(op, True)
    nulls = op_nulls(op, "last")
    ok = is_sorted_values(values, ascending=ascending, nulls=nulls)
    buckets = [
        "sortedness:check",
        f"sortedness:nulls:{nulls}",
        f"sortedness:{'true' if ok else 'false'}",
    ]
    if any(_is_null_like(value) for value in values):
        buckets.append("sortedness:null-aware")
    if sort_null_placement_mismatch(last_sort_op, column, nulls):
        buckets.append("sortedness:null-placement-mismatch")
    score = (
        0.45
        + 0.25 * int("sortedness:null-aware" in buckets)
        + 0.25 * int("sortedness:null-placement-mismatch" in buckets)
        + 0.05 * int(not ok)
    )
    return min(1.0, score), buckets, {alias: [ok]} if alias else samples


def _random_case_frontier_score(op: Any) -> tuple[float, list[str], dict[str, list[Any]]]:
    alias = op_output_alias(op)
    row_count = op_rows(op, 0)
    branch_count = op_branches(op, 0)
    buckets = [
        "case_expr:simple",
        "case_expr:random-subject",
        _bucket("case_probe_rows", row_count, [(1000, "small"), (10000, "medium")], "large"),
    ]
    if branch_count >= 3:
        buckets.append("case_expr:multi-branch")
    score = 0.55 + 0.25 * int(row_count >= 10_000) + 0.10 * int(branch_count >= 3)
    return min(1.0, score), buckets, {alias: [False]} if alias else {}


def _group_quantile_frontier_score(op: Any) -> tuple[float, list[str], dict[str, list[Any]]]:
    alias = op_output_alias(op)
    values = op_values(op)
    quantiles = op_quantiles(op)
    buckets = [
        "quantile:dynamic-key",
        _bucket("quantile_probe_values", len(values), [(2, "tiny"), (4, "small")], "medium"),
    ]
    if len(set(quantiles)) > 1:
        buckets.append("quantile:multi-probability")
    score = 0.55 + 0.25 * int("quantile:multi-probability" in buckets) + 0.10 * int(len(values) >= 3)
    return min(1.0, score), buckets, {alias: [False]} if alias else {}


def _scalar_subquery_frontier_score(op: Any) -> tuple[float, list[str], dict[str, list[Any]]]:
    alias = op_output_alias(op)
    buckets = ["subquery:correlated-scalar", "subquery:nested-aggregate"]
    return 0.90, buckets, {alias: [False]} if alias else {}


def _window_avg_frontier_score(op: Any) -> tuple[float, list[str], dict[str, list[Any]]]:
    alias = op_output_alias(op)
    buckets = ["window:rows-frame", "window:avg"]
    return 0.90, buckets, {alias: [False]} if alias else {}


def _struct_distinct_frontier_score(op: Any) -> tuple[float, list[str], dict[str, list[Any]]]:
    alias = op_output_alias(op)
    buckets = ["struct:unnest", "struct:distinct"]
    return 0.90, buckets, {alias: [False]} if alias else {}


def _bit_compare_frontier_score(op: Any) -> tuple[float, list[str], dict[str, list[Any]]]:
    alias = op_output_alias(op)
    buckets = ["bit:unequal-length", "comparison:bit-order"]
    return 0.90, buckets, {alias: [False]} if alias else {}


def _round_even_frontier_score(op: Any) -> tuple[float, list[str], dict[str, list[Any]]]:
    alias = op_output_alias(op)
    buckets = ["numeric:round-even", "float:decimal-scale"]
    return 0.90, buckets, {alias: [False]} if alias else {}


def _series_rtruediv_frontier_score(op: Any) -> tuple[float, list[str], dict[str, list[Any]]]:
    alias = op_output_alias(op)
    buckets = ["series:reverse-division", "arithmetic:operand-order"]
    return 0.90, buckets, {alias: [False]} if alias else {}


def _uint64_isin_frontier_score(op: Any) -> tuple[float, list[str], dict[str, list[Any]]]:
    alias = op_output_alias(op)
    buckets = ["pandas:uint64-isin", "membership:unsigned-precision"]
    return 0.90, buckets, {alias: [False]} if alias else {}


def _tuple_anti_null_frontier_score(op: Any) -> tuple[float, list[str], dict[str, list[Any]]]:
    alias = op_output_alias(op)
    buckets = ["duckdb:tuple-anti-null", "nulls:ternary-membership"]
    return 0.90, buckets, {alias: [False]} if alias else {}


def _setop_all_duplicate_frontier_score(op: Any) -> tuple[float, list[str], dict[str, list[Any]]]:
    alias = op_output_alias(op)
    buckets = [
        "datafusion:setop-all-duplicate-count",
        "sql:setop-all",
        "setop:except-all",
        "setop:intersect-all",
        "setop:duplicate-count",
    ]
    return 0.95, buckets, {alias: [False]} if alias else {}


def _json_predicate_order_frontier_score(op: Any) -> tuple[float, list[str], dict[str, list[Any]]]:
    alias = op_output_alias(op)
    buckets = ["duckdb:json-predicate-order", "json:predicate-reorder"]
    return 0.90, buckets, {alias: [False]} if alias else {}


def _sparse_mask_frontier_score(op: Any) -> tuple[float, list[str], dict[str, list[Any]]]:
    alias = op_output_alias(op)
    buckets = ["pandas:sparse-mask", "mask:sparse-array"]
    return 0.90, buckets, {alias: [False]} if alias else {}


def _float_wrap_frontier_score(op: Any) -> tuple[float, list[str], dict[str, list[Any]]]:
    alias = op_output_alias(op)
    buckets = ["polars:wrap-numerical", "cast:float-overflow"]
    return 0.90, buckets, {alias: [False]} if alias else {}


def _index_bool_frontier_score(op: Any) -> tuple[float, list[str], dict[str, list[Any]]]:
    alias = op_output_alias(op)
    buckets = ["pandas:index-bool", "api:result-type"]
    return 0.90, buckets, {alias: [False]} if alias else {}


def _empty_literal_groupby_frontier_score(op: Any) -> tuple[float, list[str], dict[str, list[Any]]]:
    alias = op_output_alias(op)
    buckets = ["polars:empty-literal-groupby", "groupby:empty-literal"]
    return 0.90, buckets, {alias: [False]} if alias else {}


def _arrow_string_eq_sum_frontier_score(op: Any) -> tuple[float, list[str], dict[str, list[Any]]]:
    alias = op_output_alias(op)
    buckets = ["pandas:arrow-string-eq-sum", "arrow:string-bool-reduction"]
    return 0.90, buckets, {alias: [False]} if alias else {}


def _arrow_timestamp_loc_slice_frontier_score(op: Any) -> tuple[float, list[str], dict[str, list[Any]]]:
    alias = op_output_alias(op)
    buckets = ["pandas:arrow-timestamp-loc-slice", "arrow:timestamp-index-slice"]
    return 0.90, buckets, {alias: [False]} if alias else {}


def _arrow_timestamp_index_attr_frontier_score(op: Any) -> tuple[float, list[str], dict[str, list[Any]]]:
    alias = op_output_alias(op)
    buckets = ["pandas:arrow-timestamp-index-attr", "arrow:timestamp-index-attribute"]
    return 0.90, buckets, {alias: [False]} if alias else {}


def _eval_inplace_alias_frontier_score(op: Any) -> tuple[float, list[str], dict[str, list[Any]]]:
    alias = op_output_alias(op)
    buckets = ["pandas:eval-inplace-alias", "copy:on-write-alias"]
    return 0.90, buckets, {alias: [False]} if alias else {}


def _bool_reduction_skipna_frontier_score(op: Any) -> tuple[float, list[str], dict[str, list[Any]]]:
    alias = op_output_alias(op)
    buckets = ["pandas:bool-reduction-skipna", "nullable-bool:reduction"]
    return 0.90, buckets, {alias: [False]} if alias else {}


def _dataset_isin_all_match_frontier_score(op: Any) -> tuple[float, list[str], dict[str, list[Any]]]:
    alias = op_output_alias(op)
    buckets = ["pyarrow:dataset-isin-all-match", "dataset:membership-filter"]
    return 0.90, buckets, {alias: [False]} if alias else {}


def _run_end_null_compute_frontier_score(op: Any) -> tuple[float, list[str], dict[str, list[Any]]]:
    alias = op_output_alias(op)
    buckets = ["pyarrow:run-end-null-compute", "run_end:null-compute"]
    return 0.90, buckets, {alias: [False]} if alias else {}


def _large_string_partition_frontier_score(op: Any) -> tuple[float, list[str], dict[str, list[Any]]]:
    alias = op_output_alias(op)
    buckets = ["pyarrow:large-string-partition", "dataset:partition-schema"]
    return 0.90, buckets, {alias: [False]} if alias else {}


def _hash_pivot_wider_frontier_score(op: Any) -> tuple[float, list[str], dict[str, list[Any]]]:
    alias = op_output_alias(op)
    buckets = ["pyarrow:hash-pivot-wider", "pivot:wider-order"]
    return 0.90, buckets, {alias: [False]} if alias else {}


def _list_flatten_parent_indices_frontier_score(op: Any) -> tuple[float, list[str], dict[str, list[Any]]]:
    alias = op_output_alias(op)
    buckets = ["pyarrow:list-flatten-parent-indices", "arrow:list-layout"]
    return 0.90, buckets, {alias: [False]} if alias else {}


def _rolling_mean_by_null_count_frontier_score(op: Any) -> tuple[float, list[str], dict[str, list[Any]]]:
    alias = op_output_alias(op)
    buckets = ["polars:rolling-mean-by-null-count", "rolling:temporal-min-samples"]
    return 0.90, buckets, {alias: [False]} if alias else {}


def _csv_long_numeric_roundtrip_frontier_score(op: Any) -> tuple[float, list[str], dict[str, list[Any]]]:
    alias = op_output_alias(op)
    value_count = len(op_values(op))
    buckets = [
        "csv:long-numeric-roundtrip",
        "csv:numeric-inference",
        "numeric:long-identifier",
        _bucket("csv_probe_values", value_count, [(3, "few"), (6, "several")], "many"),
    ]
    return 0.88, buckets, {alias: [False]} if alias else {}


def _groupby_frontier_score(samples: dict[str, list[Any]], op: Any) -> tuple[float, list[str]]:
    keys = [key for key in groupby_keys(op) if key in samples]
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
    if len(aggregate_specs(op)) > 1:
        buckets.append("groupby:multi-agg")
    if len(keys) > 1:
        buckets.append("groupby:multi-key")
    if _has_null_aggregate_output(samples, op, tuples):
        buckets.append("groupby:null-agg-output")
    return min(1.0, 0.35 + 0.45 * max(0.0, mixed) + 0.10 * int("groupby:multi-agg" in buckets) + 0.06 * int("groupby:multi-key" in buckets) + 0.05 * int("groupby:null-key" in buckets) + 0.08 * int("groupby:null-agg-output" in buckets)), buckets


def _aggregate_frontier_score(samples: dict[str, list[Any]], op: Any) -> tuple[float, list[str]]:
    buckets = ["aggregate:global"]
    if len(aggregate_specs(op)) > 1:
        buckets.append("aggregate:multi-agg")
    for agg in aggregate_specs(op):
        if aggregate_func(agg) in {"count", "nunique"}:
            continue
        values = samples.get(aggregate_column(agg), [])
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
    op: Any,
    tuples: list[tuple[Any, ...]],
) -> bool:
    if not tuples:
        return False
    groups: dict[tuple[Any, ...], list[int]] = {}
    for idx, key_tuple in enumerate(tuples):
        groups.setdefault(key_tuple, []).append(idx)
    for agg in aggregate_specs(op):
        if aggregate_func(agg) in {"count", "nunique"}:
            continue
        source_values = samples.get(aggregate_column(agg), [])
        for indices in groups.values():
            if all(idx >= len(source_values) or source_values[idx] is None for idx in indices):
                return True
    return False


def _mutate_frontier_score(samples: dict[str, list[Any]], op: Any) -> tuple[float, list[str], list[Any]]:
    kind = expr_kind(op)
    source = expr_source(op)
    values = list(samples.get(source, []))
    buckets: list[str] = []
    if not values:
        return 0.0, buckets, values

    if kind in {"add_const", "arith_const", "abs", "clip", "cast"}:
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
            buckets.append(f"mutate:arith:{expr_operator(op, 'unknown')}")
        if kind == "abs":
            buckets.append("mutate:abs")
        if kind == "clip":
            buckets.append("mutate:clip")
        if kind == "cast":
            buckets.append("mutate:cast")
            buckets.append(f"mutate:cast-to:{expr_target_type(op, 'unknown')}")
            if expr_input_domain(op):
                buckets.append(f"mutate:cast-domain:{expr_input_domain(op)}")
        score = 0.35 + 0.15 * int("mutate:null-propagation" in buckets) + 0.15 * int("mutate:negative" in buckets) + 0.15 * int("mutate:zero" in buckets) + 0.10 * int("mutate:fractional" in buckets)
        if kind == "arith_const" and expr_operator(op) in {"div", "mod"}:
            score += 0.10
        return min(1.0, score), buckets, _evaluate_mutate_values(samples, op)

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
    return min(1.0, score), buckets, _evaluate_mutate_values(samples, op)


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
    order = sort_sample_indices(
        samples,
        keys,
        null_like=True,
        scalar_fallback_to_repr=True,
    )
    return reorder_samples_by_index(samples, order)


def _compare_sample_values(left: Any, right: Any) -> int:
    return compare_scalar_values(left, right, fallback_to_repr=True)


def _limit_frontier_score(samples: dict[str, list[Any]], op: Any) -> tuple[float, list[str]]:
    row_count = max((len(values) for values in samples.values()), default=0)
    try:
        limit = op_n(op, 0)
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


def _offset_frontier_score(samples: dict[str, list[Any]], op: Any) -> tuple[float, list[str]]:
    row_count = max((len(values) for values in samples.values()), default=0)
    try:
        offset = op_n(op, 0)
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


def _limit_output_samples(samples: dict[str, list[Any]], op: Any) -> dict[str, list[Any]]:
    try:
        limit = max(0, op_n(op, 0))
    except (TypeError, ValueError):
        return samples
    return {name: values[:limit] for name, values in samples.items()}


def _offset_output_samples(samples: dict[str, list[Any]], op: Any) -> dict[str, list[Any]]:
    try:
        offset = max(0, op_n(op, 0))
    except (TypeError, ValueError):
        return samples
    return {name: values[offset:] for name, values in samples.items()}


def _evaluate_mutate_values(samples: dict[str, list[Any]], op: Any) -> list[Any]:
    expr = getattr(op, "expression", None)
    if expr is None:
        return []
    values = shared_eval_expr_samples(samples, expr.to_dict())
    return list(values) if values is not None else []


def _numeric_values(values: list[Any]) -> list[float]:
    out = []
    for value in values:
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, (int, float)):
            out.append(float(value))
    return out


def _is_null_like(value: Any) -> bool:
    return is_null_like(value)


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
    if _feature_aliases("semantic_signal:topk_filter_pushdown") & features:
        roots.add("topk_filter_pushdown")
    if (
        _feature_aliases("semantic_signal:projection_ordering") & features
        and "op:sort" in features
        and ("op:limit" in features or "op:offset" in features)
    ):
        roots.add("ordered_topk_projection")
    if "pattern:tuple_absence_filter" in features or "pattern:row_value_absence_filter" in features:
        roots.add("tuple_absence_null_filter")
    if "pattern:union_all_row_append" in features or "op:union_all" in features:
        roots.add("union_all_row_append")
    if "pattern:drop_nulls_null_filter" in features or "op:drop_nulls" in features:
        roots.add("drop_nulls_null_filter")
    if "pattern:semi_join_membership" in features or "op:semi_join" in features:
        roots.add("semi_join_membership")
    if "pattern:anti_join_exclusion" in features or "op:anti_join" in features:
        roots.add("anti_join_exclusion")
    if "pattern:semi_anti_join_rewrite" in features:
        roots.add("semi_anti_join_rewrite")
    if (
        "has:unicode_string" in features
        and features & {"expr:string_lower", "expr:string_upper", "string:lower", "string:upper"}
    ):
        roots.add("unicode_case_mapping")
    if "pattern:fill_null_null_semantics" in features or "op:fill_null" in features:
        roots.add("fill_null_null_semantics")
    if "pattern:coalesce_null_semantics" in features or "op:coalesce" in features:
        roots.add("coalesce_null_semantics")
    if "pattern:conditional_expression" in features or "op:case_when" in features:
        roots.add("conditional_expression")
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
    if "pattern:duckdb_float_literal_precision" in features or "op:float_literal_precision_probe" in features:
        roots.add("duckdb_float_literal_precision")
    if "pattern:polars_timestamp_precision_filter" in features or "op:timestamp_precision_filter_probe" in features:
        roots.add("polars_timestamp_precision_filter")
    if "pattern:series_rtruediv_operand_order" in features or "op:series_rtruediv_probe" in features:
        roots.add("series_rtruediv_operand_order")
    if features & {
        "pattern:polars_reverse_division_columns",
        "expr:reverse_division_columns",
        "series:reverse-division",
        "arithmetic:reverse-division",
    }:
        roots.add("reverse_division_operand_order")
    if "pattern:pandas_uint64_isin_precision" in features or "op:uint64_isin_probe" in features:
        roots.add("pandas_uint64_isin_precision")
    if "pattern:duckdb_tuple_anti_null_semantics" in features or "op:tuple_anti_null_probe" in features:
        roots.add("duckdb_tuple_anti_null_semantics")
    if (
        "pattern:datafusion_setop_all_duplicate_count" in features
        or "op:setop_all_duplicate_probe" in features
    ):
        roots.add("datafusion_setop_all_duplicate_count")
    if "pattern:duckdb_json_predicate_order_semantics" in features or "op:json_predicate_order_probe" in features:
        roots.add("duckdb_json_predicate_order_semantics")
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
    if "pattern:pandas_eval_inplace_aliasing_semantics" in features or "op:eval_inplace_alias_probe" in features:
        roots.add("pandas_eval_inplace_aliasing_semantics")
    if "pattern:pandas_bool_reduction_skipna_semantics" in features or "op:bool_reduction_skipna_probe" in features:
        roots.add("pandas_bool_reduction_skipna_semantics")
    if "pattern:pyarrow_dataset_isin_all_match_semantics" in features or "op:dataset_isin_all_match_probe" in features:
        roots.add("pyarrow_dataset_isin_all_match_semantics")
    if "pattern:pyarrow_run_end_null_compute_semantics" in features or "op:run_end_null_compute_probe" in features:
        roots.add("pyarrow_run_end_null_compute_semantics")
    if (
        "pattern:pyarrow_large_string_partition_schema_semantics" in features
        or "op:large_string_partition_probe" in features
    ):
        roots.add("pyarrow_large_string_partition_schema_semantics")
    if (
        "pattern:pyarrow_hash_pivot_wider_order_semantics" in features
        or "op:hash_pivot_wider_probe" in features
    ):
        roots.add("pyarrow_hash_pivot_wider_order_semantics")
    if (
        "pattern:pyarrow_list_flatten_parent_indices_semantics" in features
        or "op:list_flatten_parent_indices_probe" in features
    ):
        roots.add("pyarrow_list_flatten_parent_indices_semantics")
    if (
        "pattern:polars_rolling_mean_by_null_count_semantics" in features
        or "op:rolling_mean_by_null_count_probe" in features
    ):
        roots.add("polars_rolling_mean_by_null_count_semantics")
    if "pattern:csv_long_numeric_roundtrip" in features or "op:csv_long_numeric_roundtrip_probe" in features:
        roots.add("csv_long_numeric_roundtrip")
    if features & {
        "pattern:null_groupby_topk",
        "pattern:null_agg_topk",
        "pattern:filter_null_agg_topk",
        "pattern:join_null_agg_topk",
        "pattern:join_null_key_topk",
    }:
        roots.add("grouped_topk_null_sort_key")
    if "pattern:distinct_null_topk" in features:
        roots.add("distinct_null_topk")
    if features & {"pattern:empty_filter_groupby", "pattern:empty_filter_aggregate"}:
        roots.add("groupby_aggregation")
    if "op:join" in features:
        roots.add("join_semantics")
        if {"op:sort", "op:offset"}.issubset(features):
            roots.add("joined_order_offset_projection")
    if (
        "mutate:arith:mul" in features
        and "mutate:negative" in features
        and "has:zero" in features
        and "filter:truth-test" in features
    ):
        roots.add("negative_zero_comparison")
    if "op:groupby" in features:
        roots.add("float_group_key_instability" if "pattern:float_group_key" in features else "groupby_aggregation")
    if "op:aggregate" in features:
        roots.add("groupby_aggregation")
    if "op:filter" in features:
        roots.add("filter_predicate")
    if "op:mutate" in features:
        matched_expr_root = False
        if features & {"expr:string_length", "expr:string_lower", "expr:string_upper", "expr:string_strip", "expr:string_null_if_empty", "expr:string_replace", "expr:string_slice", "expr:string_split_part", "expr:string_basename", "expr:string_concat", "expr:string_contains", "expr:string_starts_with", "expr:string_ends_with"}:
            roots.add("string_expression")
            matched_expr_root = True
        if "expr:date_part" in features:
            roots.add("datetime_expression")
            matched_expr_root = True
        if "expr:bool_not" in features:
            roots.add("nullable_boolean_expression")
            matched_expr_root = True
        if "expr:cast" in features:
            roots.add("type_cast")
            matched_expr_root = True
        if not matched_expr_root:
            roots.add("arithmetic_expression")
    if features & {"op:sort", "op:limit", "op:offset"}:
        roots.add("ordering_or_limit")
    if "has:null" in features:
        roots.add("null_semantics")
    return roots


def _candidate_root_context(case: Case) -> CandidateRootContext:
    operations = tuple(case.program.operations)
    op_sequence = tuple(operation_names(operations, default="unknown"))
    hinted_roots: set[str] = set()
    metadata = case.metadata if isinstance(case.metadata, dict) else {}
    for key in ("generator_profile", "mixed_generator_profile"):
        profile = str(metadata.get(key, "")).strip()
        if not profile:
            continue
        hinted_roots.update(_CANDIDATE_ROOT_HINTS_BY_PROFILE.get(profile, ()))
    probe_root = last_probe_root_for_operations(operations)
    if probe_root:
        hinted_roots.add(probe_root)
    return CandidateRootContext(
        case=case,
        operations=operations,
        op_sequence=op_sequence,
        op_set=frozenset(op_sequence),
        hinted_roots=frozenset(hinted_roots),
        probe_root=probe_root,
    )


def _candidate_prefilter_matches_root(context: CandidateRootContext, root: str) -> bool:
    if root in context.hinted_roots:
        return True
    matcher = _CANDIDATE_ROOT_MATCHERS.get(root)
    return bool(matcher(context)) if matcher is not None else False


def _sequence_contains_ordered_ops(
    op_sequence: tuple[str, ...],
    *steps: str | frozenset[str],
) -> bool:
    cursor = 0
    for step in steps:
        accepted = {step} if isinstance(step, str) else set(step)
        while cursor < len(op_sequence) and op_sequence[cursor] not in accepted:
            cursor += 1
        if cursor >= len(op_sequence):
            return False
        cursor += 1
    return True


def _candidate_matches_joined_order_offset_projection(context: CandidateRootContext) -> bool:
    return _sequence_contains_ordered_ops(
        context.op_sequence,
        "join",
        "sort",
        "offset",
    )


def _candidate_matches_ordered_topk_projection(context: CandidateRootContext) -> bool:
    return _sequence_contains_ordered_ops(
        context.op_sequence,
        "sort",
        frozenset({"limit", "offset"}),
        frozenset({"select", "sort"}),
    )


def _candidate_matches_topk_filter_pushdown(context: CandidateRootContext) -> bool:
    return _sequence_contains_ordered_ops(
        context.op_sequence,
        "sort",
        frozenset({"limit", "offset"}),
        "filter",
    )


def _candidate_matches_negative_zero_comparison(context: CandidateRootContext) -> bool:
    zero_source_columns = _candidate_zero_numeric_columns(context.case)
    if not zero_source_columns:
        return False
    negative_zero_columns: set[str] = set()
    for op in context.operations:
        kind = op_kind(op)
        if kind == "mutate":
            if (
                expr_kind(op) == "arith_const"
                and expr_operator(op) == "mul"
                and _candidate_is_negative_numeric(expr_value(op))
                and expr_source(op) in zero_source_columns
            ):
                negative_zero_columns.add(op_column(op))
            continue
        if kind != "filter" or op_column(op) not in negative_zero_columns:
            continue
        parsed = parse_filter_comparator(condition_cmp(op) or op_comparator(op))
        if parsed is not None and _candidate_filter_touches_zero(parsed.base, op_value(op)):
            return True
    return False


def _candidate_zero_numeric_columns(case: Case) -> set[str]:
    columns: set[str] = set()
    for table in case.tables:
        numeric_columns = {column.name for column in table.columns if column.type in {"int", "float"}}
        for row in table.rows:
            for column_name, value in row.items():
                if column_name in numeric_columns and _candidate_is_numeric_value(value, 0.0):
                    columns.add(column_name)
    return columns


def _candidate_is_negative_numeric(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return float(value) < 0.0
    except Exception:
        return False


def _candidate_is_numeric_value(value: Any, target: float) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return float(value) == target
    except Exception:
        return False


def _candidate_filter_touches_zero(base: str, value: Any) -> bool:
    if base in {
        ">",
        ">=",
        "<",
        "<=",
        "==",
        "!=",
        "gt_is_not_true",
        "ge_is_not_true",
        "lt_is_not_false",
        "le_is_not_false",
    }:
        return _candidate_is_numeric_value(value, 0.0)
    if base in {"in_set", "not_in_set"} and isinstance(value, (list, tuple, set, frozenset)):
        return any(_candidate_is_numeric_value(item, 0.0) for item in value)
    if base == "range_closed" and isinstance(value, (list, tuple)) and len(value) == 2:
        lower, upper = value
        if _candidate_is_numeric_value(lower, 0.0) or _candidate_is_numeric_value(upper, 0.0):
            return True
        if _candidate_is_orderable_number(lower) and _candidate_is_orderable_number(upper):
            return float(lower) < 0.0 < float(upper)
    return False


def _candidate_is_orderable_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        number = float(value)
    except Exception:
        return False
    return not math.isnan(number)


_CANDIDATE_ROOT_HINTS_BY_PROFILE: dict[str, frozenset[str]] = {
    "join_null_truth_filter": frozenset({"outer_join_truth_filter"}),
    "float_group_key": frozenset({"float_group_key_instability"}),
    "running_sum_precision": frozenset({"running_sum_precision"}),
    "partitioned_running_sum": frozenset({"running_sum_precision"}),
    "sortedness_null_placement": frozenset({"sortedness_null_placement"}),
    "polars_reverse_division_columns": frozenset({"reverse_division_operand_order"}),
    "null_groupby_topk": frozenset({"grouped_topk_null_sort_key"}),
    "null_agg_topk": frozenset({"grouped_topk_null_sort_key"}),
    "filter_null_agg_topk": frozenset({"grouped_topk_null_sort_key"}),
    "join_null_agg_topk": frozenset({"grouped_topk_null_sort_key"}),
    "join_null_key_topk": frozenset({"grouped_topk_null_sort_key"}),
    "distinct_null_topk": frozenset({"distinct_null_topk"}),
    "pyarrow_list_flatten_parent_indices_semantics": frozenset({"pyarrow_list_flatten_parent_indices_semantics"}),
    "csv_long_numeric_roundtrip": frozenset({"csv_long_numeric_roundtrip"}),
}

_CANDIDATE_ROOT_MATCHERS: dict[str, Callable[[CandidateRootContext], bool]] = {
    "outer_join_truth_filter": lambda context: has_join_null_truth_filter_pattern(context.operations),
    "topk_filter_pushdown": _candidate_matches_topk_filter_pushdown,
    "running_sum_precision": lambda context: has_running_sum_precision_pattern(context.operations),
    "sortedness_null_placement": lambda context: has_sortedness_null_placement_pattern(context.operations),
    "reverse_division_operand_order": lambda context: has_reverse_division_columns_pattern(context.operations),
    "float_group_key_instability": lambda context: has_float_group_key_pattern(context.operations),
    "joined_order_offset_projection": _candidate_matches_joined_order_offset_projection,
    "ordered_topk_projection": _candidate_matches_ordered_topk_projection,
    "negative_zero_comparison": _candidate_matches_negative_zero_comparison,
}

_CANDIDATE_PREFILTER_ROOTS = frozenset(
    set(PROBE_ROOTS.values())
    | {root for roots in _CANDIDATE_ROOT_HINTS_BY_PROFILE.values() for root in roots}
    | set(_CANDIDATE_ROOT_MATCHERS)
)
