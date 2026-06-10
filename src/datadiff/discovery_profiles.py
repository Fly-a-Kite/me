from __future__ import annotations

from collections.abc import Callable, Mapping

from .dsl import Case

ProfileGenerator = Callable[[int], Case]
ProfileCaseGenerator = Callable[[int, str], Case]


DEEP_PROBE_ROTATION_PROFILES = (
    "simple_case_random_subject",
    "group_quantile_key_probe",
    "scalar_subquery_double_parentheses",
    "window_avg_rows_frame",
    "struct_distinct_unnest",
    "bit_compare_unequal_length",
    "round_even_float_scale",
    "series_rtruediv_operand_order",
    "pandas_uint64_isin_precision",
    "duckdb_tuple_anti_null_semantics",
    "datafusion_setop_all_duplicate_count",
    "duckdb_json_predicate_order_semantics",
    "pandas_sparse_array_mask_semantics",
    "polars_float_wrap_numerical_semantics",
    "pandas_index_bool_result_type",
    "polars_empty_literal_groupby_semantics",
    "pandas_arrow_string_eq_sum_semantics",
    "pandas_arrow_timestamp_loc_slice_semantics",
    "pandas_arrow_timestamp_index_attr_semantics",
    "pandas_eval_inplace_aliasing_semantics",
    "pandas_bool_reduction_skipna_semantics",
    "pandas_arrow_bool_groupby_reduction_semantics",
    "pyarrow_dataset_isin_all_match_semantics",
    "pyarrow_run_end_null_compute_semantics",
    "pyarrow_large_string_partition_schema_semantics",
    "pyarrow_hash_pivot_wider_order_semantics",
    "pyarrow_list_flatten_parent_indices_semantics",
    "polars_rolling_mean_by_null_count_semantics",
    "pyarrow_groupby_filter_cast_membership",
    "partitioned_running_sum",
    "path_basename_keyed_pick",
    "csv_long_numeric_roundtrip",
)


ISSUE_FOCUS_MIXED_PROFILES = (
    "null_groupby_topk",
    "null_agg_topk",
    "filter_null_agg_topk",
    "join_null_agg_topk",
    "empty_filter_groupby",
    "join_filter_groupby",
    "join_null_sort",
    "ordered_groupby_sort",
    "topk_resort",
    "join_ordered_agg_topk",
    "global_null_aggregate",
    "string_count_groupby",
    "unique_count_groupby",
    "bool_null_groupby_agg",
    "large_int_filter_groupby",
    "set_membership_filter",
    "pyarrow_groupby_filter_cast_membership",
    "null_predicate_filter",
    "row_value_absence_filter",
    "path_basename_keyed_pick",
    "polars_reverse_division_columns",
    "pandas_bool_reduction_skipna_semantics",
    "csv_long_numeric_roundtrip",
)


def as_discovery_mixed_case(
    case: Case,
    seed: int,
    mixed_profile: str,
    *,
    generator_profile: str = "discovery",
) -> Case:
    metadata = dict(case.metadata)
    metadata["generator_profile"] = generator_profile
    metadata["mixed_generator_profile"] = mixed_profile
    profile_slug = generator_profile.replace("_", "-")
    return Case(
        case_id=f"case-{seed:08d}-{profile_slug}-{mixed_profile.replace('_', '-')}",
        seed=seed,
        tables=case.tables,
        program=case.program,
        metadata=metadata,
    )


def deep_probe_rotation_case(seed: int, *, generate_profile_case: ProfileCaseGenerator) -> Case:
    profile = DEEP_PROBE_ROTATION_PROFILES[seed % len(DEEP_PROBE_ROTATION_PROFILES)]
    case = generate_profile_case(seed, profile)
    return as_discovery_mixed_case(
        case,
        seed,
        profile,
        generator_profile="deep_probe_rotation",
    )


def issue_focus_case(
    seed: int,
    *,
    generate_profile_case: ProfileCaseGenerator,
    profile_generators: Mapping[str, ProfileGenerator],
) -> Case:
    issue_case = discovery_issue_inspired_case(seed, profile_generators=profile_generators)
    if issue_case is not None:
        mixed_profile = str(issue_case.metadata.get("mixed_generator_profile", "issue_inspired"))
        return as_discovery_mixed_case(
            issue_case,
            seed,
            mixed_profile,
            generator_profile="issue_focus",
        )
    start_index = seed % len(ISSUE_FOCUS_MIXED_PROFILES)
    for offset in range(len(ISSUE_FOCUS_MIXED_PROFILES)):
        mixed_profile = ISSUE_FOCUS_MIXED_PROFILES[(start_index + offset) % len(ISSUE_FOCUS_MIXED_PROFILES)]
        case = generate_profile_case(seed + offset, mixed_profile)
        return as_discovery_mixed_case(
            case,
            seed,
            mixed_profile,
            generator_profile="issue_focus",
        )
    fallback = generate_profile_case(seed, "discovery_fresh")
    return as_discovery_mixed_case(
        fallback,
        seed,
        "discovery_fresh",
        generator_profile="issue_focus",
    )


def discovery_issue_inspired_case(
    seed: int,
    *,
    profile_generators: Mapping[str, ProfileGenerator],
) -> Case | None:
    def mixed(profile: str) -> Case:
        return as_discovery_mixed_case(profile_generators[profile](seed), seed, profile)

    if seed % 211 == 111:
        return mixed("duckdb_float_literal_precision")
    if seed % 227 == 114:
        return mixed("polars_timestamp_precision_filter")
    selector = seed % 60
    if selector == 2:
        return mixed("join_null_truth_filter")
    if selector == 5:
        return mixed("global_null_aggregate")
    if selector == 11:
        return mixed("empty_filter_groupby")
    if selector == 20:
        return mixed("wide_offset_topk")
    if selector == 31:
        return mixed("ordered_groupby_sort")
    if selector == 40:
        return mixed("join_null_key_topk")
    if selector == 47:
        return mixed("string_count_groupby")
    if selector == 50:
        return mixed("set_membership_filter")
    if selector == 51:
        return mixed("null_predicate_filter")
    if selector == 52:
        return mixed("polars_reverse_division_columns")
    if selector == 54:
        return mixed("boolean_predicate_filter")
    if selector == 55:
        return mixed("post_topk_range_filter")
    if selector == 56:
        return mixed("unique_count_groupby")
    if selector == 57:
        return mixed("tuple_absence_filter")
    if selector == 53:
        return mixed("topk_resort")
    if selector == 58:
        return mixed("join_ordered_agg_topk")
    if selector == 59:
        return mixed("running_sum_precision")
    if seed % 71 == 60:
        return mixed("sortedness_null_placement")
    if seed % 73 == 61:
        return mixed("simple_case_random_subject")
    if seed % 79 == 63:
        return mixed("group_quantile_key_probe")
    if seed % 83 == 64:
        return mixed("scalar_subquery_double_parentheses")
    if seed % 89 == 66:
        return mixed("window_avg_rows_frame")
    if seed % 97 == 67:
        return mixed("struct_distinct_unnest")
    if seed % 101 == 68:
        return mixed("bit_compare_unequal_length")
    if seed % 103 == 69:
        return mixed("round_even_float_scale")
    if seed % 107 == 70:
        return mixed("series_rtruediv_operand_order")
    if seed % 109 == 72:
        return mixed("pandas_uint64_isin_precision")
    if seed % 113 == 73:
        return mixed("duckdb_tuple_anti_null_semantics")
    if seed % 199 == 102:
        return mixed("datafusion_setop_all_duplicate_count")
    if seed % 181 == 105:
        return mixed("duckdb_json_predicate_order_semantics")
    if seed % 181 == 101:
        return mixed("row_value_absence_filter")
    if seed % 127 == 74:
        return mixed("pandas_sparse_array_mask_semantics")
    if seed % 131 == 75:
        return mixed("polars_float_wrap_numerical_semantics")
    if seed % 137 == 76:
        return mixed("pandas_index_bool_result_type")
    if seed % 139 == 77:
        return mixed("polars_empty_literal_groupby_semantics")
    if seed % 149 == 78:
        return mixed("pandas_arrow_string_eq_sum_semantics")
    if seed % 151 == 79:
        return mixed("pandas_arrow_timestamp_loc_slice_semantics")
    if seed % 157 == 81:
        return mixed("pandas_arrow_timestamp_index_attr_semantics")
    if seed % 181 == 108:
        return mixed("pandas_eval_inplace_aliasing_semantics")
    if seed % 193 == 110:
        return mixed("pandas_bool_reduction_skipna_semantics")
    if seed % 239 == 84:
        return mixed("pandas_arrow_bool_groupby_reduction_semantics")
    if seed % 163 == 82:
        return mixed("pyarrow_dataset_isin_all_match_semantics")
    if seed % 197 == 112:
        return mixed("pyarrow_run_end_null_compute_semantics")
    if seed % 173 == 104:
        return mixed("pyarrow_large_string_partition_schema_semantics")
    if seed % 191 == 106:
        return mixed("pyarrow_hash_pivot_wider_order_semantics")
    if seed % 233 == 8:
        return mixed("pyarrow_list_flatten_parent_indices_semantics")
    if seed % 167 == 83:
        return mixed("polars_rolling_mean_by_null_count_semantics")
    if seed % 197 == 109:
        return mixed("pyarrow_groupby_filter_cast_membership")
    if seed % 223 == 103:
        return mixed("partitioned_running_sum")
    if seed % 229 == 107:
        return mixed("path_basename_keyed_pick")
    if seed % 251 == 48:
        return mixed("csv_long_numeric_roundtrip")
    return None


def discovery_no_groupby_issue_inspired_case(
    seed: int,
    *,
    profile_generators: Mapping[str, ProfileGenerator],
) -> Case | None:
    if seed % 53 == 18:
        return as_discovery_mixed_case(
            profile_generators["datafusion_setop_all_duplicate_count"](seed),
            seed,
            "datafusion_setop_all_duplicate_count",
            generator_profile="discovery_no_groupby",
        )
    return None
