from __future__ import annotations

from collections.abc import Callable, Mapping

from .dsl import Case

ProfileGenerator = Callable[[int], Case]
ProfileCaseGenerator = Callable[[int, str], Case]


ORTHOGONAL_STRESS_ROTATION_PROFILES = (
    "bitmap_boundary_bool_aggregate",
    "vector_boundary_groupby_distinct",
    "wide_schema_projection_boundary",
    "skewed_join_multiplicity",
    "utf8_slice_length_groupby",
    "unique_order_window_tiebreak",
)

PANDAS_TARGETED_ROTATION_PROFILES = (
    "wide_schema_projection_boundary",
    "bitmap_boundary_bool_aggregate",
    "utf8_slice_length_groupby",
    "vector_boundary_groupby_distinct",
    "skewed_join_multiplicity",
)

POLARS_TARGETED_ROTATION_PROFILES = (
    "unique_order_window_tiebreak",
    "bitmap_boundary_bool_aggregate",
    "wide_schema_projection_boundary",
    "vector_boundary_groupby_distinct",
    "skewed_join_multiplicity",
    "utf8_slice_length_groupby",
)

DATAFUSION_TARGETED_ROTATION_PROFILES = (
    "vector_boundary_groupby_distinct",
    "skewed_join_multiplicity",
    "unique_order_window_tiebreak",
    "wide_schema_projection_boundary",
    "utf8_slice_length_groupby",
    "bitmap_boundary_bool_aggregate",
)


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
    "case_when_join_key_membership",
    "coalesce_union_distinct_type_boundary",
    "multi_key_anti_join_null_guard",
    "empty_then_union_groupby",
    "boolean_coalesce_case_membership",
    "numeric_text_cast_membership_aggregation",
    "partitioned_running_sum",
    "path_basename_keyed_pick",
    "csv_long_numeric_roundtrip",
    "string_token_join_distinct",
    "date_part_row_number_union",
    "post_groupby_join_global_aggregate",
    "distinct_anti_join_case_topk",
    "coalesce_row_number_topk",
    "union_distinct_anti_running_sum",
    "null_case_semi_join_groupby",
    "date_string_cast_row_number",
    "drop_nulls_coalesce_distinct_join_topk",
    "date_part_distinct_offset",
    "bool_fill_null_membership_row_number",
    "string_numeric_cast_anti_join_aggregate",
    "post_aggregate_case_membership",
    "multi_key_nullable_membership_window",
    "string_empty_pattern_membership_distinct",
    "date_cast_union_running_sum_topk",
    "coalesce_anti_join_union_topk",
    "bool_null_distinct_running_sum",
    "date_string_membership_offset_window",
    "empty_union_window_aggregate",
    "duplicate_key_join_distinct_anti_topk",
    "large_int_text_membership_window",
    "nested_topk_offset_aggregate",
    "union_distinct_empty_string_window",
    "multi_key_semi_join_window_aggregate",
    "string_contains_anti_join_offset",
    "bool_case_distinct_groupby_union",
    "left_join_filter_distinct_window",
    "cast_groupby_membership",
    "null_sort_window_union",
    "date_part_membership_distinct_join",
    "coalesce_case_anti_join_aggregate",
    "string_token_transform_join_window",
    "prefix_suffix_bool_membership",
    "numeric_clip_division_anti_window",
    "bool_not_union_distinct_aggregate",
    "outer_join_coalesce_distinct_topk",
    "chained_string_cleanup_membership_window",
    "cast_date_union_anti_running",
    "post_groupby_filter_membership_topk",
    "duplicate_key_left_join_window_aggregate",
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
    "case_when_join_key_membership",
    "coalesce_union_distinct_type_boundary",
    "multi_key_anti_join_null_guard",
    "empty_then_union_groupby",
    "boolean_coalesce_case_membership",
    "numeric_text_cast_membership_aggregation",
    "null_predicate_filter",
    "row_value_absence_filter",
    "path_basename_keyed_pick",
    "polars_reverse_division_columns",
    "pandas_bool_reduction_skipna_semantics",
    "csv_long_numeric_roundtrip",
    "string_token_join_distinct",
    "date_part_row_number_union",
    "post_groupby_join_global_aggregate",
    "distinct_anti_join_case_topk",
    "coalesce_row_number_topk",
    "union_distinct_anti_running_sum",
    "null_case_semi_join_groupby",
    "date_string_cast_row_number",
    "drop_nulls_coalesce_distinct_join_topk",
    "date_part_distinct_offset",
    "bool_fill_null_membership_row_number",
    "string_numeric_cast_anti_join_aggregate",
    "post_aggregate_case_membership",
    "multi_key_nullable_membership_window",
    "string_empty_pattern_membership_distinct",
    "date_cast_union_running_sum_topk",
    "coalesce_anti_join_union_topk",
    "bool_null_distinct_running_sum",
    "date_string_membership_offset_window",
    "empty_union_window_aggregate",
    "duplicate_key_join_distinct_anti_topk",
    "large_int_text_membership_window",
    "nested_topk_offset_aggregate",
    "union_distinct_empty_string_window",
    "multi_key_semi_join_window_aggregate",
    "string_contains_anti_join_offset",
    "bool_case_distinct_groupby_union",
    "left_join_filter_distinct_window",
    "cast_groupby_membership",
    "null_sort_window_union",
    "date_part_membership_distinct_join",
    "coalesce_case_anti_join_aggregate",
    "string_token_transform_join_window",
    "prefix_suffix_bool_membership",
    "numeric_clip_division_anti_window",
    "bool_not_union_distinct_aggregate",
    "outer_join_coalesce_distinct_topk",
    "chained_string_cleanup_membership_window",
    "cast_date_union_anti_running",
    "post_groupby_filter_membership_topk",
    "duplicate_key_left_join_window_aggregate",
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


def orthogonal_stress_rotation_case(seed: int, *, generate_profile_case: ProfileCaseGenerator) -> Case:
    return _rotation_case(
        seed,
        profiles=ORTHOGONAL_STRESS_ROTATION_PROFILES,
        generator_profile="orthogonal_stress_rotation",
        generate_profile_case=generate_profile_case,
    )


def pandas_targeted_rotation_case(seed: int, *, generate_profile_case: ProfileCaseGenerator) -> Case:
    return _rotation_case(
        seed,
        profiles=PANDAS_TARGETED_ROTATION_PROFILES,
        generator_profile="pandas_targeted_rotation",
        generate_profile_case=generate_profile_case,
    )


def polars_targeted_rotation_case(seed: int, *, generate_profile_case: ProfileCaseGenerator) -> Case:
    return _rotation_case(
        seed,
        profiles=POLARS_TARGETED_ROTATION_PROFILES,
        generator_profile="polars_targeted_rotation",
        generate_profile_case=generate_profile_case,
    )


def datafusion_targeted_rotation_case(seed: int, *, generate_profile_case: ProfileCaseGenerator) -> Case:
    return _rotation_case(
        seed,
        profiles=DATAFUSION_TARGETED_ROTATION_PROFILES,
        generator_profile="datafusion_targeted_rotation",
        generate_profile_case=generate_profile_case,
    )


def _rotation_case(
    seed: int,
    *,
    profiles: tuple[str, ...],
    generator_profile: str,
    generate_profile_case: ProfileCaseGenerator,
) -> Case:
    profile = profiles[seed % len(profiles)]
    case = generate_profile_case(seed, profile)
    return as_discovery_mixed_case(
        case,
        seed,
        profile,
        generator_profile=generator_profile,
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
    if seed % 257 == 116:
        return mixed("case_when_join_key_membership")
    if seed % 263 == 118:
        return mixed("coalesce_union_distinct_type_boundary")
    if seed % 269 == 120:
        return mixed("multi_key_anti_join_null_guard")
    if seed % 271 == 122:
        return mixed("empty_then_union_groupby")
    if seed % 277 == 124:
        return mixed("boolean_coalesce_case_membership")
    if seed % 281 == 126:
        return mixed("numeric_text_cast_membership_aggregation")
    if seed % 283 == 128:
        return mixed("string_token_join_distinct")
    if seed % 293 == 130:
        return mixed("date_part_row_number_union")
    if seed % 307 == 132:
        return mixed("post_groupby_join_global_aggregate")
    if seed % 311 == 134:
        return mixed("distinct_anti_join_case_topk")
    if seed % 313 == 136:
        return mixed("coalesce_row_number_topk")
    if seed % 317 == 138:
        return mixed("union_distinct_anti_running_sum")
    if seed % 331 == 140:
        return mixed("null_case_semi_join_groupby")
    if seed % 337 == 142:
        return mixed("date_string_cast_row_number")
    if seed % 347 == 144:
        return mixed("drop_nulls_coalesce_distinct_join_topk")
    if seed % 349 == 146:
        return mixed("date_part_distinct_offset")
    if seed % 353 == 148:
        return mixed("bool_fill_null_membership_row_number")
    if seed % 359 == 150:
        return mixed("string_numeric_cast_anti_join_aggregate")
    if seed % 367 == 152:
        return mixed("post_aggregate_case_membership")
    if seed % 373 == 154:
        return mixed("multi_key_nullable_membership_window")
    if seed % 379 == 156:
        return mixed("string_empty_pattern_membership_distinct")
    if seed % 383 == 158:
        return mixed("date_cast_union_running_sum_topk")
    if seed % 389 == 160:
        return mixed("coalesce_anti_join_union_topk")
    if seed % 397 == 162:
        return mixed("bool_null_distinct_running_sum")
    if seed % 401 == 164:
        return mixed("date_string_membership_offset_window")
    if seed % 409 == 166:
        return mixed("empty_union_window_aggregate")
    if seed % 419 == 168:
        return mixed("duplicate_key_join_distinct_anti_topk")
    if seed % 421 == 170:
        return mixed("large_int_text_membership_window")
    if seed % 431 == 172:
        return mixed("nested_topk_offset_aggregate")
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
