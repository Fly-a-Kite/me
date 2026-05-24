from pathlib import Path

from datadiff.case_policy import case_discovery_origin, replay_bug_filter_reason
from datadiff.cli import _preset_config
from datadiff.config import DEFAULT_REPLAY_BUG_SOURCE_ISSUES
from datadiff.datagen import generate_case
from datadiff.targets import TARGETS, common_capabilities, resolve_target_backends


CORE_METHOD_CAPABILITIES = {
    "op:filter",
    "op:mutate",
    "op:groupby",
    "op:aggregate",
    "op:join",
    "op:sort",
    "op:limit",
    "expr:arith_const",
    "expr:string_lower",
    "agg:min",
    "agg:mean",
    "agg:count",
    "agg:any",
    "agg:all",
    "nulls",
}


def test_methodology_targets_span_multiple_framework_families():
    implemented = [target for target in TARGETS.values() if target.status == "implemented"]
    families = {target.family for target in implemented}
    layers = {target.layer for target in implemented}

    assert {"dataframe", "embedded_sql", "query_engine", "arrow"}.issubset(families)
    assert {"python_dataframe", "embedded_analytical_engine", "arrow_query_engine", "arrow_compute"}.issubset(layers)
    assert len([target for target in implemented if target.family != "seeded_fault"]) >= 7


def test_methodology_cross_family_suites_keep_common_dsl_contract():
    suites = {
        "core": resolve_target_backends(target_suite="core"),
        "core_lazy": resolve_target_backends(target_suite="core_lazy"),
        "core_datafusion": resolve_target_backends(target_suite="core_datafusion"),
        "core_arrow": resolve_target_backends(target_suite="core_arrow"),
        "datafusion_cross": resolve_target_backends(target_suite="datafusion_cross"),
        "arrow_cross": resolve_target_backends(target_suite="arrow_cross"),
        "latest_all_engines": resolve_target_backends(target_suite="latest_all_engines"),
        "latest_no_datafusion": resolve_target_backends(target_suite="latest_no_datafusion"),
        "embedded_sql_cross": resolve_target_backends(target_suite="embedded_sql_cross"),
    }

    for suite, backends in suites.items():
        families = {TARGETS[backend].family for backend in backends}
        assert len(families) >= 2, suite
        assert CORE_METHOD_CAPABILITIES.issubset(common_capabilities(backends)), suite


def test_methodology_latest_live_suite_covers_all_real_targets():
    backends = resolve_target_backends(target_suite="latest_all_engines")
    target_backends = {"pyarrow", "polars", "polars_lazy", "duckdb", "sqlite", "datafusion"}
    assert target_backends.issubset(backends)
    assert "pandas" in backends

    families = {TARGETS[backend].family for backend in backends}
    assert {"arrow", "dataframe", "embedded_sql", "query_engine"}.issubset(families)

    live = _preset_config("live_cross_family")
    assert live.guidance_strategy == "guided"
    assert live.enable_feedback is True
    assert live.enable_normalizer is True
    assert live.enable_local_source_scheduler is True
    assert {"operation_combo", "join", "filter", "groupby", "sort_limit", "topk"}.issubset(live.guidance_targets)

    focus = _preset_config("live_issue_focus")
    assert focus.generator_profile == "issue_focus"
    assert focus.enable_replay_bug is False
    assert {
        "row_value_absence_filter",
        "polars_reverse_division_columns",
        "join_filter_groupby",
        "pandas_bool_reduction_skipna_semantics",
        "csv_long_numeric_roundtrip",
    }.issubset(focus.guidance_targets)


def test_methodology_experiment_presets_cover_required_ablation_axes():
    baseline = _preset_config("baseline")
    no_type = _preset_config("no_type_aware")
    no_normalizer = _preset_config("no_normalizer")
    no_feedback = _preset_config("no_feedback")
    metamorphic = _preset_config("metamorphic")
    guided = _preset_config("guided")
    guided_join = _preset_config("guided_join")

    assert baseline.enable_type_aware_generation
    assert baseline.enable_normalizer
    assert baseline.enable_feedback
    assert not no_type.enable_type_aware_generation
    assert not no_normalizer.enable_normalizer
    assert not no_feedback.enable_feedback
    assert metamorphic.enable_metamorphic_oracle
    assert metamorphic.oracle_mode == "both"
    assert guided.guidance_strategy == "guided"
    assert guided.guidance_candidate_pool > 1
    assert guided_join.generator_profile == "bughunt_no_groupby"
    assert "join" in guided_join.guidance_targets


def test_methodology_fresh_and_replay_share_case_policy_gate():
    fresh = _preset_config("live_datafusion")
    replay = _preset_config("live_datafusion_replay")

    assert fresh.enable_replay_bug is False
    assert replay.enable_replay_bug is True
    assert replay.generator_profile == fresh.generator_profile
    assert replay.guidance_targets == fresh.guidance_targets
    assert replay.guidance_strategy == fresh.guidance_strategy
    assert replay.guidance_candidate_pool == fresh.guidance_candidate_pool

    replay_profiles = [
        "datafusion_setop_all_duplicate_count",
        "duckdb_tuple_anti_null_semantics",
        "polars_rolling_mean_by_null_count_semantics",
        "pandas_bool_reduction_skipna_semantics",
        "pyarrow_run_end_null_compute_semantics",
    ]
    for seed, profile in enumerate(replay_profiles, start=137):
        case = generate_case(seed, profile=profile)
        assert case_discovery_origin(case) == "issue_replay"
        assert (
            replay_bug_filter_reason(
                case,
                enable_replay_bug=False,
                replay_bug_source_issues=DEFAULT_REPLAY_BUG_SOURCE_ISSUES,
            )
            == "issue_replay_probe"
        )
        assert (
            replay_bug_filter_reason(
                case,
                enable_replay_bug=True,
                replay_bug_source_issues=DEFAULT_REPLAY_BUG_SOURCE_ISSUES,
            )
            == ""
        )

    fresh_policy_rejections = []
    for seed in range(20):
        fresh_case = generate_case(seed, profile="issue_focus")
        reason = replay_bug_filter_reason(
            fresh_case,
            enable_replay_bug=False,
            replay_bug_source_issues=DEFAULT_REPLAY_BUG_SOURCE_ISSUES,
        )
        if reason:
            fresh_policy_rejections.append(reason)

    assert {"known_replay_source_issue", "issue_replay_probe"}.issubset(set(fresh_policy_rejections))


def test_methodology_bottom_layer_does_not_import_middle_policy_modules():
    repo_root = Path(__file__).resolve().parents[1]
    bottom_layer_paths = [
        repo_root / "src/datadiff/datagen.py",
        repo_root / "src/datadiff/csv_roundtrip.py",
        repo_root / "src/datadiff/dsl.py",
        repo_root / "src/datadiff/normalizer.py",
        repo_root / "src/datadiff/oracle.py",
        repo_root / "src/datadiff/pathing.py",
        repo_root / "src/datadiff/windowing.py",
        repo_root / "src/datadiff/running.py",
    ]
    bottom_layer_paths.extend((repo_root / "src/datadiff/backends").glob("*.py"))
    forbidden_imports = (
        "datadiff.case_policy",
        "datadiff.config",
        "datadiff.guidance",
        "datadiff.scheduler",
        "datadiff.final_readiness",
        "datadiff.reporter",
        "datadiff.reward",
        "datadiff.targets",
    )

    for path in bottom_layer_paths:
        source = path.read_text(encoding="utf-8")
        assert not any(forbidden in source for forbidden in forbidden_imports), path


def test_methodology_replay_source_gate_spans_historical_projects():
    required_sources = {
        "https://github.com/apache/datafusion/issues/22190",
        "https://github.com/duckdb/duckdb/issues/22075",
        "https://github.com/duckdb/duckdb/issues/22656",
        "https://github.com/duckdb/duckdb/issues/22837",
        "https://github.com/apache/arrow/issues/42231",
    }

    assert required_sources.issubset(set(DEFAULT_REPLAY_BUG_SOURCE_ISSUES))


def test_methodology_bug_hunting_presets_target_distinct_semantic_risks():
    null_groupby = _preset_config("null_groupby_topk")
    null_agg = _preset_config("null_agg_topk")
    filter_null_agg = _preset_config("filter_null_agg_topk")
    join_null_agg = _preset_config("join_null_agg_topk")
    join_null_key = _preset_config("join_null_key_topk")
    wide_offset = _preset_config("wide_offset_topk")
    empty_filter_groupby = _preset_config("empty_filter_groupby")
    join_filter_groupby = _preset_config("join_filter_groupby")
    join_null_truth_filter = _preset_config("join_null_truth_filter")
    join_groupby_stress = _preset_config("join_groupby_stress")
    storage_offset = _preset_config("storage_offset")
    float_group = _preset_config("float_group_key")
    float_group_meta = _preset_config("float_group_key_metamorphic")
    join_null_sort = _preset_config("join_null_sort")
    ordered_groupby_sort = _preset_config("ordered_groupby_sort")
    topk_resort = _preset_config("topk_resort")
    join_ordered_agg_topk = _preset_config("join_ordered_agg_topk")
    global_null_aggregate = _preset_config("global_null_aggregate")
    string_count_groupby = _preset_config("string_count_groupby")
    unique_count_groupby = _preset_config("unique_count_groupby")
    bool_null_groupby_agg = _preset_config("bool_null_groupby_agg")
    large_int_filter_groupby = _preset_config("large_int_filter_groupby")
    set_membership_filter = _preset_config("set_membership_filter")
    null_predicate_filter = _preset_config("null_predicate_filter")
    boolean_predicate_filter = _preset_config("boolean_predicate_filter")
    post_topk_range_filter = _preset_config("post_topk_range_filter")
    tuple_absence_filter = _preset_config("tuple_absence_filter")
    row_value_absence_filter = _preset_config("row_value_absence_filter")
    running_sum_precision = _preset_config("running_sum_precision")
    partitioned_running_sum = _preset_config("partitioned_running_sum")
    path_basename_keyed_pick = _preset_config("path_basename_keyed_pick")
    sortedness_null_placement = _preset_config("sortedness_null_placement")
    simple_case_random_subject = _preset_config("simple_case_random_subject")
    group_quantile_key_probe = _preset_config("group_quantile_key_probe")
    scalar_subquery_double_parentheses = _preset_config("scalar_subquery_double_parentheses")
    window_avg_rows_frame = _preset_config("window_avg_rows_frame")
    struct_distinct_unnest = _preset_config("struct_distinct_unnest")
    bit_compare_unequal_length = _preset_config("bit_compare_unequal_length")
    round_even_float_scale = _preset_config("round_even_float_scale")
    duckdb_float_literal_precision = _preset_config("duckdb_float_literal_precision")
    polars_timestamp_precision_filter = _preset_config("polars_timestamp_precision_filter")
    series_rtruediv_operand_order = _preset_config("series_rtruediv_operand_order")
    pandas_uint64_isin_precision = _preset_config("pandas_uint64_isin_precision")
    duckdb_tuple_anti_null_semantics = _preset_config("duckdb_tuple_anti_null_semantics")
    datafusion_setop_all_duplicate_count = _preset_config("datafusion_setop_all_duplicate_count")
    duckdb_json_predicate_order_semantics = _preset_config("duckdb_json_predicate_order_semantics")
    pandas_sparse_array_mask_semantics = _preset_config("pandas_sparse_array_mask_semantics")
    polars_float_wrap_numerical_semantics = _preset_config("polars_float_wrap_numerical_semantics")
    pandas_index_bool_result_type = _preset_config("pandas_index_bool_result_type")
    polars_empty_literal_groupby_semantics = _preset_config("polars_empty_literal_groupby_semantics")
    pandas_arrow_string_eq_sum_semantics = _preset_config("pandas_arrow_string_eq_sum_semantics")
    pandas_arrow_timestamp_loc_slice_semantics = _preset_config("pandas_arrow_timestamp_loc_slice_semantics")
    pandas_arrow_timestamp_index_attr_semantics = _preset_config("pandas_arrow_timestamp_index_attr_semantics")
    pandas_eval_inplace_aliasing_semantics = _preset_config("pandas_eval_inplace_aliasing_semantics")
    pandas_bool_reduction_skipna_semantics = _preset_config("pandas_bool_reduction_skipna_semantics")
    pyarrow_dataset_isin_all_match_semantics = _preset_config("pyarrow_dataset_isin_all_match_semantics")
    pyarrow_run_end_null_compute_semantics = _preset_config("pyarrow_run_end_null_compute_semantics")
    pyarrow_large_string_partition_schema_semantics = _preset_config(
        "pyarrow_large_string_partition_schema_semantics"
    )
    pyarrow_hash_pivot_wider_order_semantics = _preset_config(
        "pyarrow_hash_pivot_wider_order_semantics"
    )
    polars_rolling_mean_by_null_count_semantics = _preset_config("polars_rolling_mean_by_null_count_semantics")
    csv_long_numeric_roundtrip = _preset_config("csv_long_numeric_roundtrip")

    assert null_groupby.generator_profile == "null_groupby_topk"
    assert {"groupby", "nulls", "sort_limit"}.issubset(null_groupby.guidance_targets)
    assert null_agg.generator_profile == "null_agg_topk"
    assert {"aggregation", "nulls", "sort_limit"}.issubset(null_agg.guidance_targets)
    assert filter_null_agg.generator_profile == "filter_null_agg_topk"
    assert {"filter", "aggregation", "nulls", "sort_limit", "expressions"}.issubset(filter_null_agg.guidance_targets)
    assert join_null_agg.generator_profile == "join_null_agg_topk"
    assert {"join", "aggregation", "nulls", "sort_limit"}.issubset(join_null_agg.guidance_targets)
    assert join_null_key.generator_profile == "join_null_key_topk"
    assert {"join", "groupby", "nulls", "sort_limit", "topk"}.issubset(join_null_key.guidance_targets)
    assert wide_offset.generator_profile == "wide_offset_topk"
    assert {"sort_offset", "offset", "topk"}.issubset(wide_offset.guidance_targets)
    assert empty_filter_groupby.generator_profile == "empty_filter_groupby"
    assert {"filter", "groupby", "aggregation", "empty"}.issubset(empty_filter_groupby.guidance_targets)
    assert join_filter_groupby.generator_profile == "join_filter_groupby"
    assert {"join", "filter", "groupby", "aggregation", "sort_limit"}.issubset(join_filter_groupby.guidance_targets)
    assert join_null_truth_filter.generator_profile == "join_null_truth_filter"
    assert {"join", "filter", "truth_filter", "nulls"}.issubset(join_null_truth_filter.guidance_targets)
    assert join_groupby_stress.generator_profile == "join_groupby_stress"
    assert {"join", "groupby", "aggregation", "global_aggregation"}.issubset(join_groupby_stress.guidance_targets)
    assert storage_offset.generator_profile == "storage_offset"
    assert {"sort_offset", "offset"}.issubset(storage_offset.guidance_targets)
    assert float_group.generator_profile == "float_group_key"
    assert {"join", "mutate", "groupby", "expressions"}.issubset(float_group.guidance_targets)
    assert float_group_meta.enable_metamorphic_oracle
    assert float_group_meta.metamorphic_variant_limit > float_group.metamorphic_variant_limit
    assert join_null_sort.generator_profile == "join_null_sort"
    assert {"join", "nulls", "sort_limit"}.issubset(join_null_sort.guidance_targets)
    assert ordered_groupby_sort.generator_profile == "ordered_groupby_sort"
    assert {"groupby", "aggregation", "sort_limit"}.issubset(ordered_groupby_sort.guidance_targets)
    assert topk_resort.generator_profile == "topk_resort"
    assert {"sort_limit", "topk", "nulls"}.issubset(topk_resort.guidance_targets)
    assert join_ordered_agg_topk.generator_profile == "join_ordered_agg_topk"
    assert {"join", "groupby", "aggregation", "sort_limit", "topk"}.issubset(join_ordered_agg_topk.guidance_targets)
    assert global_null_aggregate.generator_profile == "global_null_aggregate"
    assert {"global_aggregation", "aggregation", "nulls", "sort_limit"}.issubset(global_null_aggregate.guidance_targets)
    assert string_count_groupby.generator_profile == "string_count_groupby"
    assert {"groupby", "strings", "nulls", "aggregation", "sort_limit"}.issubset(string_count_groupby.guidance_targets)
    assert unique_count_groupby.generator_profile == "unique_count_groupby"
    assert {"groupby", "strings", "unique_count", "aggregation", "sort_limit"}.issubset(unique_count_groupby.guidance_targets)
    assert bool_null_groupby_agg.generator_profile == "bool_null_groupby_agg"
    assert {"groupby", "nulls", "boolean_aggregation", "bool_any_all", "aggregation", "sort_limit"}.issubset(
        bool_null_groupby_agg.guidance_targets
    )
    assert large_int_filter_groupby.generator_profile == "large_int_filter_groupby"
    assert {"filter", "groupby", "large_integer", "numeric", "aggregation", "sort_limit"}.issubset(
        large_int_filter_groupby.guidance_targets
    )
    assert set_membership_filter.generator_profile == "set_membership_filter"
    assert {"filter", "strings", "set_membership", "aggregation", "sort_limit"}.issubset(set_membership_filter.guidance_targets)
    assert null_predicate_filter.generator_profile == "null_predicate_filter"
    assert {"filter", "null_predicate", "aggregation", "sort_limit"}.issubset(null_predicate_filter.guidance_targets)
    assert boolean_predicate_filter.generator_profile == "boolean_predicate_filter"
    assert {"filter", "boolean_predicate", "truth_filter", "aggregation", "sort_limit"}.issubset(boolean_predicate_filter.guidance_targets)
    assert post_topk_range_filter.generator_profile == "post_topk_range_filter"
    assert {"filter", "range_filter", "sort_limit", "topk"}.issubset(post_topk_range_filter.guidance_targets)
    assert tuple_absence_filter.generator_profile == "tuple_absence_filter"
    assert {"filter", "tuple_absence", "nulls", "join"}.issubset(tuple_absence_filter.guidance_targets)
    assert row_value_absence_filter.generator_profile == "row_value_absence_filter"
    assert {"filter", "row_value_absence", "tuple_absence", "nulls", "join"}.issubset(
        row_value_absence_filter.guidance_targets
    )
    assert running_sum_precision.generator_profile == "running_sum_precision"
    assert {"running_sum", "numeric", "sort_limit"}.issubset(running_sum_precision.guidance_targets)
    assert partitioned_running_sum.generator_profile == "partitioned_running_sum"
    assert {"running_sum_partitioned", "running_sum", "numeric"}.issubset(partitioned_running_sum.guidance_targets)
    assert path_basename_keyed_pick.generator_profile == "path_basename_keyed_pick"
    assert {"path_projection", "keyed_row_pick", "strings"}.issubset(path_basename_keyed_pick.guidance_targets)
    assert _preset_config("path_basename_keyed_pick_replay").enable_replay_bug is True
    assert sortedness_null_placement.generator_profile == "sortedness_null_placement"
    assert {"sortedness", "nulls", "sort_limit"}.issubset(sortedness_null_placement.guidance_targets)
    assert simple_case_random_subject.generator_profile == "simple_case_random_subject"
    assert {"random_case_probe", "case_expression"}.issubset(simple_case_random_subject.guidance_targets)
    assert group_quantile_key_probe.generator_profile == "group_quantile_key_probe"
    assert {"group_quantile_probe", "dynamic_quantile", "groupby"}.issubset(group_quantile_key_probe.guidance_targets)
    assert scalar_subquery_double_parentheses.generator_profile == "scalar_subquery_double_parentheses"
    assert {"scalar_subquery_probe", "correlated_subquery"}.issubset(
        scalar_subquery_double_parentheses.guidance_targets
    )
    assert window_avg_rows_frame.generator_profile == "window_avg_rows_frame"
    assert {"window_avg_probe", "window_frame", "numeric"}.issubset(window_avg_rows_frame.guidance_targets)
    assert struct_distinct_unnest.generator_profile == "struct_distinct_unnest"
    assert {"struct_distinct_probe", "struct_unnest"}.issubset(struct_distinct_unnest.guidance_targets)
    assert bit_compare_unequal_length.generator_profile == "bit_compare_unequal_length"
    assert {"bit_compare_probe", "bit_ordering"}.issubset(bit_compare_unequal_length.guidance_targets)
    assert round_even_float_scale.generator_profile == "round_even_float_scale"
    assert {"round_even_probe", "rounding", "numeric"}.issubset(round_even_float_scale.guidance_targets)
    assert duckdb_float_literal_precision.generator_profile == "duckdb_float_literal_precision"
    assert {"float_literal_precision_probe", "float_literal_precision", "numeric"}.issubset(
        duckdb_float_literal_precision.guidance_targets
    )
    assert polars_timestamp_precision_filter.generator_profile == "polars_timestamp_precision_filter"
    assert {"timestamp_precision_filter_probe", "timestamp_precision_filter", "casts"}.issubset(
        polars_timestamp_precision_filter.guidance_targets
    )
    assert series_rtruediv_operand_order.generator_profile == "series_rtruediv_operand_order"
    assert {"series_rtruediv_probe", "reverse_division", "numeric"}.issubset(
        series_rtruediv_operand_order.guidance_targets
    )
    assert pandas_uint64_isin_precision.generator_profile == "pandas_uint64_isin_precision"
    assert {"uint64_isin_probe", "unsigned_membership", "numeric"}.issubset(
        pandas_uint64_isin_precision.guidance_targets
    )
    assert duckdb_tuple_anti_null_semantics.generator_profile == "duckdb_tuple_anti_null_semantics"
    assert {"tuple_anti_null_probe", "tuple_null_membership", "nulls"}.issubset(
        duckdb_tuple_anti_null_semantics.guidance_targets
    )
    assert datafusion_setop_all_duplicate_count.generator_profile == "datafusion_setop_all_duplicate_count"
    assert {"setop_all_duplicate_probe", "setop_all_duplicates", "aggregation"}.issubset(
        datafusion_setop_all_duplicate_count.guidance_targets
    )
    assert duckdb_json_predicate_order_semantics.generator_profile == "duckdb_json_predicate_order_semantics"
    assert {"json_predicate_order_probe", "json_predicate_order", "filter"}.issubset(
        duckdb_json_predicate_order_semantics.guidance_targets
    )
    assert pandas_sparse_array_mask_semantics.generator_profile == "pandas_sparse_array_mask_semantics"
    assert {"sparse_mask_probe", "sparse_masking", "filter"}.issubset(
        pandas_sparse_array_mask_semantics.guidance_targets
    )
    assert polars_float_wrap_numerical_semantics.generator_profile == "polars_float_wrap_numerical_semantics"
    assert {"float_wrap_probe", "wrap_numerical", "casts"}.issubset(
        polars_float_wrap_numerical_semantics.guidance_targets
    )
    assert pandas_index_bool_result_type.generator_profile == "pandas_index_bool_result_type"
    assert {"index_bool_probe", "index_boolean_result", "filter"}.issubset(
        pandas_index_bool_result_type.guidance_targets
    )
    assert polars_empty_literal_groupby_semantics.generator_profile == "polars_empty_literal_groupby_semantics"
    assert {"empty_literal_groupby_probe", "literal_empty_groupby", "groupby"}.issubset(
        polars_empty_literal_groupby_semantics.guidance_targets
    )
    assert pandas_arrow_string_eq_sum_semantics.generator_profile == "pandas_arrow_string_eq_sum_semantics"
    assert {"arrow_string_eq_sum_probe", "arrow_string_reduction", "strings"}.issubset(
        pandas_arrow_string_eq_sum_semantics.guidance_targets
    )
    assert (
        pandas_arrow_timestamp_loc_slice_semantics.generator_profile
        == "pandas_arrow_timestamp_loc_slice_semantics"
    )
    assert {"arrow_timestamp_loc_slice_probe", "arrow_timestamp_indexing", "sort_limit"}.issubset(
        pandas_arrow_timestamp_loc_slice_semantics.guidance_targets
    )
    assert (
        pandas_arrow_timestamp_index_attr_semantics.generator_profile
        == "pandas_arrow_timestamp_index_attr_semantics"
    )
    assert {"arrow_timestamp_index_attr_probe", "arrow_timestamp_attributes", "sort_limit"}.issubset(
        pandas_arrow_timestamp_index_attr_semantics.guidance_targets
    )
    assert pandas_eval_inplace_aliasing_semantics.generator_profile == "pandas_eval_inplace_aliasing_semantics"
    assert {"eval_inplace_alias_probe", "eval_inplace_aliasing", "mutate"}.issubset(
        pandas_eval_inplace_aliasing_semantics.guidance_targets
    )
    assert pandas_bool_reduction_skipna_semantics.generator_profile == "pandas_bool_reduction_skipna_semantics"
    assert {"bool_reduction_skipna_probe", "bool_reduction_skipna", "nulls"}.issubset(
        pandas_bool_reduction_skipna_semantics.guidance_targets
    )
    assert pyarrow_dataset_isin_all_match_semantics.generator_profile == "pyarrow_dataset_isin_all_match_semantics"
    assert {"dataset_isin_all_match_probe", "dataset_membership_filter", "filter"}.issubset(
        pyarrow_dataset_isin_all_match_semantics.guidance_targets
    )
    assert pyarrow_run_end_null_compute_semantics.generator_profile == "pyarrow_run_end_null_compute_semantics"
    assert {"run_end_null_compute_probe", "run_end_null_compute", "nulls"}.issubset(
        pyarrow_run_end_null_compute_semantics.guidance_targets
    )
    assert (
        pyarrow_large_string_partition_schema_semantics.generator_profile
        == "pyarrow_large_string_partition_schema_semantics"
    )
    assert {"large_string_partition_probe", "large_string_partition", "strings"}.issubset(
        pyarrow_large_string_partition_schema_semantics.guidance_targets
    )
    assert (
        pyarrow_hash_pivot_wider_order_semantics.generator_profile
        == "pyarrow_hash_pivot_wider_order_semantics"
    )
    assert {"hash_pivot_wider_probe", "hash_pivot_wider", "aggregation"}.issubset(
        pyarrow_hash_pivot_wider_order_semantics.guidance_targets
    )
    assert (
        polars_rolling_mean_by_null_count_semantics.generator_profile
        == "polars_rolling_mean_by_null_count_semantics"
    )
    assert {"rolling_mean_by_null_count_probe", "rolling_temporal_nulls", "nulls"}.issubset(
        polars_rolling_mean_by_null_count_semantics.guidance_targets
    )
    assert csv_long_numeric_roundtrip.generator_profile == "csv_long_numeric_roundtrip"
    assert {"csv_long_numeric_roundtrip_probe", "csv_numeric_inference", "numeric"}.issubset(
        csv_long_numeric_roundtrip.guidance_targets
    )


def test_methodology_seeded_fault_suites_support_sensitivity_evaluation():
    assert resolve_target_backends(target_suite="seeded_filter") == ["pandas", "buggy_filter"]
    assert resolve_target_backends(target_suite="seeded_groupby") == ["pandas", "buggy_groupby"]
    assert resolve_target_backends(target_suite="seeded_join") == ["pandas", "buggy_join"]
    assert resolve_target_backends(target_suite="seeded_mutate") == ["pandas", "buggy_mutate"]
