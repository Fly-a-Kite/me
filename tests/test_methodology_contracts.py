from datadiff.cli import _preset_config
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
    "agg:count",
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


def test_methodology_seeded_fault_suites_support_sensitivity_evaluation():
    assert resolve_target_backends(target_suite="seeded_filter") == ["pandas", "buggy_filter"]
    assert resolve_target_backends(target_suite="seeded_groupby") == ["pandas", "buggy_groupby"]
    assert resolve_target_backends(target_suite="seeded_join") == ["pandas", "buggy_join"]
    assert resolve_target_backends(target_suite="seeded_mutate") == ["pandas", "buggy_mutate"]
