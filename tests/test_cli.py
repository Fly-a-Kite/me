import json
import os
from pathlib import Path

import pytest

from datadiff import cli
from datadiff.cli import _experiment_target_runs, _preset_config, build_parser
from datadiff.dsl import Case, ColumnSpec, Program, TableData
from datadiff.util import append_jsonl, run_meta_path


def test_cli_parses_fuzz_ablation_flags():
    parser = build_parser()
    args = parser.parse_args(
        [
            "fuzz",
            "--cases",
            "10",
            "--seed",
            "5",
            "--duration",
            "10s",
            "--profile",
            "edge_float",
            "--disable-normalizer",
            "--disable-feedback",
            "--disable-preflight-repair",
            "--persist-feedback-corpus",
            "--feedback-persist-limit",
            "12",
            "--enable-local-source-scheduler",
            "--local-source-exploration-weight",
            "0.25",
            "--metamorphic-variant-limit",
            "9",
            "--log-level",
            "minimal",
        ]
    )
    assert args.cmd == "fuzz"
    assert args.cases == 10
    assert args.seed == 5
    assert args.duration == "10s"
    assert args.profile == "edge_float"
    assert args.disable_normalizer is True
    assert args.disable_feedback is True
    assert args.disable_preflight_repair is True
    assert args.persist_feedback_corpus is True
    assert args.feedback_persist_limit == 12
    assert args.enable_local_source_scheduler is True
    assert args.local_source_exploration_weight == 0.25
    assert args.metamorphic_variant_limit == 9
    assert args.log_level == "minimal"
    assert args.no_compress_run_log is False


def test_cli_parses_no_compress_run_log():
    parser = build_parser()
    args = parser.parse_args(["fuzz", "--no-compress-run-log"])
    assert args.cmd == "fuzz"
    assert args.no_compress_run_log is True


def test_cli_parses_artifact_limit():
    parser = build_parser()
    args = parser.parse_args(["longrun", "--artifact-limit", "25"])
    assert args.cmd == "longrun"
    assert args.artifact_limit == 25


def test_cli_parses_guided_fuzz_options():
    parser = build_parser()
    args = parser.parse_args(
        [
            "fuzz",
            "--strategy",
            "guided",
            "--candidate-pool",
            "12",
            "--targets",
            "groupby,nulls",
        ]
    )
    assert args.cmd == "fuzz"
    assert args.strategy == "guided"
    assert args.candidate_pool == 12
    assert args.targets == "groupby,nulls"


def test_cli_parses_workflow_profile():
    parser = build_parser()
    args = parser.parse_args(["fuzz", "--profile", "workflow"])
    assert args.cmd == "fuzz"
    assert args.profile == "workflow"


def test_cli_parses_bughunt_profile():
    parser = build_parser()
    args = parser.parse_args(["fuzz", "--profile", "bughunt"])
    assert args.cmd == "fuzz"
    assert args.profile == "bughunt"


def test_cli_parses_bughunt_no_groupby_profile():
    parser = build_parser()
    args = parser.parse_args(["fuzz", "--profile", "bughunt_no_groupby"])
    assert args.cmd == "fuzz"
    assert args.profile == "bughunt_no_groupby"


def test_cli_parses_null_groupby_topk_profile():
    parser = build_parser()
    args = parser.parse_args(["fuzz", "--profile", "null_groupby_topk"])
    assert args.cmd == "fuzz"
    assert args.profile == "null_groupby_topk"

    args = parser.parse_args(["longrun", "--profile", "null_groupby_topk"])
    assert args.cmd == "longrun"
    assert args.profile == "null_groupby_topk"


def test_cli_parses_null_agg_topk_profile():
    parser = build_parser()
    args = parser.parse_args(["fuzz", "--profile", "null_agg_topk"])
    assert args.cmd == "fuzz"
    assert args.profile == "null_agg_topk"

    args = parser.parse_args(["longrun", "--profile", "null_agg_topk"])
    assert args.cmd == "longrun"
    assert args.profile == "null_agg_topk"


def test_cli_parses_filter_null_agg_topk_profile():
    parser = build_parser()
    args = parser.parse_args(["fuzz", "--profile", "filter_null_agg_topk"])
    assert args.cmd == "fuzz"
    assert args.profile == "filter_null_agg_topk"

    args = parser.parse_args(["longrun", "--profile", "filter_null_agg_topk"])
    assert args.cmd == "longrun"
    assert args.profile == "filter_null_agg_topk"


def test_cli_parses_join_null_agg_topk_profile():
    parser = build_parser()
    args = parser.parse_args(["fuzz", "--profile", "join_null_agg_topk"])
    assert args.cmd == "fuzz"
    assert args.profile == "join_null_agg_topk"

    args = parser.parse_args(["longrun", "--profile", "join_null_agg_topk"])
    assert args.cmd == "longrun"
    assert args.profile == "join_null_agg_topk"


def test_cli_parses_additional_issue_inspired_profiles():
    parser = build_parser()
    for profile in ["join_null_key_topk", "wide_offset_topk", "empty_filter_groupby"]:
        args = parser.parse_args(["fuzz", "--profile", profile])
        assert args.cmd == "fuzz"
        assert args.profile == profile

        args = parser.parse_args(["longrun", "--profile", profile])
        assert args.cmd == "longrun"
        assert args.profile == profile


def test_cli_parses_join_filter_groupby_profile():
    parser = build_parser()
    args = parser.parse_args(["fuzz", "--profile", "join_filter_groupby"])
    assert args.cmd == "fuzz"
    assert args.profile == "join_filter_groupby"

    args = parser.parse_args(["longrun", "--profile", "join_filter_groupby"])
    assert args.cmd == "longrun"
    assert args.profile == "join_filter_groupby"


def test_cli_parses_join_null_truth_filter_profile():
    parser = build_parser()
    args = parser.parse_args(["fuzz", "--profile", "join_null_truth_filter"])
    assert args.cmd == "fuzz"
    assert args.profile == "join_null_truth_filter"

    args = parser.parse_args(["longrun", "--profile", "join_null_truth_filter"])
    assert args.cmd == "longrun"
    assert args.profile == "join_null_truth_filter"


def test_cli_parses_join_groupby_stress_profile():
    parser = build_parser()
    args = parser.parse_args(["fuzz", "--profile", "join_groupby_stress"])
    assert args.cmd == "fuzz"
    assert args.profile == "join_groupby_stress"

    args = parser.parse_args(["longrun", "--profile", "join_groupby_stress"])
    assert args.cmd == "longrun"
    assert args.profile == "join_groupby_stress"


def test_cli_parses_storage_offset_profile():
    parser = build_parser()
    args = parser.parse_args(["fuzz", "--profile", "storage_offset"])
    assert args.cmd == "fuzz"
    assert args.profile == "storage_offset"

    args = parser.parse_args(["longrun", "--profile", "storage_offset"])
    assert args.cmd == "longrun"
    assert args.profile == "storage_offset"


def test_cli_parses_float_group_key_profile():
    parser = build_parser()
    args = parser.parse_args(["fuzz", "--profile", "float_group_key"])
    assert args.cmd == "fuzz"
    assert args.profile == "float_group_key"

    args = parser.parse_args(["longrun", "--profile", "float_group_key"])
    assert args.cmd == "longrun"
    assert args.profile == "float_group_key"


def test_cli_parses_join_null_sort_profile():
    parser = build_parser()
    args = parser.parse_args(["fuzz", "--profile", "join_null_sort"])
    assert args.cmd == "fuzz"
    assert args.profile == "join_null_sort"

    args = parser.parse_args(["longrun", "--profile", "join_null_sort"])
    assert args.cmd == "longrun"
    assert args.profile == "join_null_sort"


def test_cli_parses_order_sensitive_bug_hunt_profiles():
    parser = build_parser()
    for profile in [
        "ordered_groupby_sort",
        "topk_resort",
        "join_ordered_agg_topk",
        "global_null_aggregate",
        "string_count_groupby",
        "unique_count_groupby",
        "set_membership_filter",
        "null_predicate_filter",
        "boolean_predicate_filter",
        "post_topk_range_filter",
        "tuple_absence_filter",
        "running_sum_precision",
        "sortedness_null_placement",
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
        "duckdb_json_predicate_order_semantics",
        "pandas_sparse_array_mask_semantics",
        "polars_float_wrap_numerical_semantics",
        "pandas_index_bool_result_type",
        "polars_empty_literal_groupby_semantics",
        "pandas_arrow_string_eq_sum_semantics",
        "pandas_arrow_timestamp_loc_slice_semantics",
        "pandas_arrow_timestamp_index_attr_semantics",
        "pandas_eval_inplace_aliasing_semantics",
        "pyarrow_dataset_isin_all_match_semantics",
        "pyarrow_large_string_partition_schema_semantics",
        "pyarrow_hash_pivot_wider_order_semantics",
        "polars_rolling_mean_by_null_count_semantics",
    ]:
        args = parser.parse_args(["fuzz", "--profile", profile])
        assert args.cmd == "fuzz"
        assert args.profile == profile

        args = parser.parse_args(["longrun", "--profile", profile])
        assert args.cmd == "longrun"
        assert args.profile == profile


def test_cli_parses_target_suite():
    parser = build_parser()
    args = parser.parse_args(["fuzz", "--target-suite", "dataframe"])
    assert args.cmd == "fuzz"
    assert args.target_suite == "dataframe"
    assert args.backends is None


def test_cli_parses_targets_command():
    parser = build_parser()
    args = parser.parse_args(["targets"])
    assert args.cmd == "targets"


def test_cli_parses_targets_json_command():
    parser = build_parser()
    args = parser.parse_args(["targets", "--json"])
    assert args.cmd == "targets"
    assert args.json is True


def test_cli_prune_corpus_dry_run_and_yes(tmp_path, monkeypatch, capsys):
    corpus_dir = tmp_path / "corpus"
    interesting = corpus_dir / "interesting"
    interesting.mkdir(parents=True)
    files = []
    for idx in range(3):
        path = interesting / f"{idx}.json"
        path.write_text("{}", encoding="utf-8")
        os.utime(path, (idx + 1, idx + 1))
        files.append(path)
    monkeypatch.setattr(cli, "CORPUS_DIR", corpus_dir)

    parser = build_parser()
    args = parser.parse_args(["prune-corpus", "--keep", "1"])
    assert args.func(args) == 0
    assert all(path.exists() for path in files)
    assert "dry_run=true" in capsys.readouterr().out

    args = parser.parse_args(["prune-corpus", "--keep", "1", "--yes"])
    assert args.func(args) == 0
    remaining = sorted(path.name for path in interesting.glob("*.json"))
    assert remaining == ["2.json"]


def test_cli_parses_experiment_command():
    parser = build_parser()
    args = parser.parse_args(
        [
            "experiment",
            "--cases",
            "5",
            "--seeds",
            "1,2",
            "--presets",
            "baseline,metamorphic",
        ]
    )
    assert args.cmd == "experiment"
    assert args.cases == 5
    assert args.seeds == "1,2"
    assert args.presets == "baseline,metamorphic"
    assert args.no_compress_run_log is False
    assert args.target_suites is None
    assert args.metamorphic_variant_limit is None


def test_cli_parses_multi_target_suite_experiment():
    parser = build_parser()
    args = parser.parse_args(
        [
            "experiment",
            "--cases",
            "5",
            "--target-suites",
            "dataframe,embedded_sql,cross_family",
        ]
    )
    assert args.cmd == "experiment"
    assert args.target_suites == "dataframe,embedded_sql,cross_family"
    assert _experiment_target_runs(args) == [
        ("dataframe", ["pandas", "polars"]),
        ("embedded_sql", ["duckdb", "sqlite"]),
        ("cross_family", ["pandas", "duckdb"]),
    ]


def test_cli_parses_workflow_experiment_preset():
    parser = build_parser()
    args = parser.parse_args(["experiment", "--presets", "workflow"])
    assert args.cmd == "experiment"
    assert args.presets == "workflow"


def test_cli_parses_workflow_metamorphic_experiment_preset():
    parser = build_parser()
    args = parser.parse_args(["experiment", "--presets", "workflow_metamorphic"])
    assert args.cmd == "experiment"
    assert args.presets == "workflow_metamorphic"
    config = _preset_config(args.presets)
    assert config.generator_profile == "workflow"
    assert config.enable_metamorphic_oracle is True
    assert config.oracle_mode == "both"


def test_cli_parses_edge_float_guided_experiment_preset():
    parser = build_parser()
    args = parser.parse_args(["experiment", "--presets", "edge_float_guided"])
    assert args.cmd == "experiment"
    config = _preset_config(args.presets)
    assert config.generator_profile == "edge_float"
    assert config.guidance_strategy == "guided"
    assert config.guidance_candidate_pool == 8
    assert config.guidance_targets == ["edge_float", "numeric", "expressions"]


def test_cli_parses_edge_float_metamorphic_experiment_preset():
    parser = build_parser()
    args = parser.parse_args(["experiment", "--presets", "edge_float_metamorphic"])
    assert args.cmd == "experiment"
    config = _preset_config(args.presets)
    assert config.generator_profile == "edge_float"
    assert config.enable_metamorphic_oracle is True
    assert config.oracle_mode == "both"


def test_cli_parses_targeted_guided_experiment_presets():
    assert _preset_config("guided_filter").guidance_targets == ["filter"]
    assert _preset_config("guided_join").generator_profile == "bughunt_no_groupby"
    assert _preset_config("guided_join").guidance_targets == ["join", "sort_limit"]
    assert _preset_config("guided_mutate").guidance_targets == ["mutate", "expressions"]
    assert _preset_config("null_groupby_topk").generator_profile == "null_groupby_topk"
    assert _preset_config("null_groupby_topk").guidance_targets[0] == "null_groupby_topk"
    assert _preset_config("null_agg_topk").generator_profile == "null_agg_topk"
    assert _preset_config("null_agg_topk").guidance_targets[0] == "null_agg_topk"
    assert _preset_config("null_agg_topk_metamorphic").enable_metamorphic_oracle is True
    assert _preset_config("filter_null_agg_topk").generator_profile == "filter_null_agg_topk"
    assert _preset_config("filter_null_agg_topk").guidance_targets[0] == "filter_null_agg_topk"
    assert _preset_config("filter_null_agg_topk_metamorphic").enable_metamorphic_oracle is True
    assert _preset_config("join_null_agg_topk").generator_profile == "join_null_agg_topk"
    assert _preset_config("join_null_agg_topk").guidance_targets[0] == "join_null_agg_topk"
    assert _preset_config("join_null_agg_topk_metamorphic").enable_metamorphic_oracle is True
    assert _preset_config("join_null_key_topk").generator_profile == "join_null_key_topk"
    assert _preset_config("join_null_key_topk").guidance_targets[0] == "join_null_key_topk"
    assert _preset_config("join_null_key_topk_metamorphic").enable_metamorphic_oracle is True
    assert _preset_config("wide_offset_topk").generator_profile == "wide_offset_topk"
    assert _preset_config("wide_offset_topk").guidance_targets[0] == "wide_offset_topk"
    assert _preset_config("wide_offset_topk_metamorphic").enable_metamorphic_oracle is True
    assert _preset_config("empty_filter_groupby").generator_profile == "empty_filter_groupby"
    assert _preset_config("empty_filter_groupby").guidance_targets[0] == "empty_filter_groupby"
    assert _preset_config("empty_filter_groupby_metamorphic").enable_metamorphic_oracle is True
    assert _preset_config("join_filter_groupby").generator_profile == "join_filter_groupby"
    assert _preset_config("join_filter_groupby").guidance_targets[0] == "join_filter_groupby"
    assert _preset_config("join_filter_groupby_metamorphic").enable_metamorphic_oracle is True
    assert _preset_config("join_null_truth_filter").generator_profile == "join_null_truth_filter"
    assert _preset_config("join_null_truth_filter").guidance_targets[0] == "join_null_truth_filter"
    assert _preset_config("join_null_truth_filter_metamorphic").enable_metamorphic_oracle is True
    assert _preset_config("join_groupby_stress").generator_profile == "join_groupby_stress"
    assert "global_aggregation" in _preset_config("join_groupby_stress").guidance_targets
    assert _preset_config("join_groupby_stress_metamorphic").enable_metamorphic_oracle is True
    assert _preset_config("storage_offset").generator_profile == "storage_offset"
    assert {"sort_offset", "offset"}.issubset(_preset_config("storage_offset").guidance_targets)
    assert _preset_config("float_group_key").generator_profile == "float_group_key"
    assert _preset_config("float_group_key").guidance_targets[0] == "float_group_key"
    assert _preset_config("float_group_key_metamorphic").enable_metamorphic_oracle is True
    assert _preset_config("global_null_aggregate").generator_profile == "global_null_aggregate"
    assert _preset_config("global_null_aggregate").guidance_targets[0] == "global_null_aggregate"
    assert _preset_config("global_null_aggregate_metamorphic").enable_metamorphic_oracle is True
    assert _preset_config("string_count_groupby").generator_profile == "string_count_groupby"
    assert _preset_config("string_count_groupby").guidance_targets[0] == "string_count_groupby"
    assert _preset_config("string_count_groupby_metamorphic").enable_metamorphic_oracle is True
    assert _preset_config("unique_count_groupby").generator_profile == "unique_count_groupby"
    assert _preset_config("unique_count_groupby").guidance_targets[0] == "unique_count_groupby"
    assert _preset_config("unique_count_groupby_metamorphic").enable_metamorphic_oracle is True
    assert _preset_config("set_membership_filter").generator_profile == "set_membership_filter"
    assert _preset_config("set_membership_filter").guidance_targets[0] == "set_membership_filter"
    assert _preset_config("set_membership_filter_metamorphic").enable_metamorphic_oracle is True
    assert _preset_config("null_predicate_filter").generator_profile == "null_predicate_filter"
    assert _preset_config("null_predicate_filter").guidance_targets[0] == "null_predicate_filter"
    assert _preset_config("null_predicate_filter_metamorphic").enable_metamorphic_oracle is True
    assert _preset_config("boolean_predicate_filter").generator_profile == "boolean_predicate_filter"
    assert _preset_config("boolean_predicate_filter").guidance_targets[0] == "boolean_predicate_filter"
    assert _preset_config("boolean_predicate_filter_metamorphic").enable_metamorphic_oracle is True
    assert _preset_config("post_topk_range_filter").generator_profile == "post_topk_range_filter"
    assert _preset_config("post_topk_range_filter").guidance_targets[0] == "post_topk_range_filter"
    assert _preset_config("post_topk_range_filter_metamorphic").enable_metamorphic_oracle is True
    assert _preset_config("tuple_absence_filter").generator_profile == "tuple_absence_filter"
    assert _preset_config("tuple_absence_filter").guidance_targets[0] == "tuple_absence_filter"
    assert _preset_config("tuple_absence_filter_metamorphic").enable_metamorphic_oracle is True
    assert _preset_config("running_sum_precision").generator_profile == "running_sum_precision"
    assert _preset_config("running_sum_precision").guidance_targets[0] == "running_sum_precision"
    assert _preset_config("running_sum_precision_metamorphic").enable_metamorphic_oracle is True
    assert _preset_config("sortedness_null_placement").generator_profile == "sortedness_null_placement"
    assert _preset_config("sortedness_null_placement").guidance_targets[0] == "sortedness_null_placement"
    assert _preset_config("sortedness_null_placement_metamorphic").enable_metamorphic_oracle is True
    assert _preset_config("simple_case_random_subject").generator_profile == "simple_case_random_subject"
    assert _preset_config("simple_case_random_subject").guidance_targets[0] == "simple_case_random_subject"
    assert _preset_config("simple_case_random_subject_metamorphic").enable_metamorphic_oracle is True
    assert _preset_config("group_quantile_key_probe").generator_profile == "group_quantile_key_probe"
    assert _preset_config("group_quantile_key_probe").guidance_targets[0] == "group_quantile_key_probe"
    assert _preset_config("group_quantile_key_probe_metamorphic").enable_metamorphic_oracle is True
    assert _preset_config("scalar_subquery_double_parentheses").generator_profile == "scalar_subquery_double_parentheses"
    assert _preset_config("scalar_subquery_double_parentheses").guidance_targets[0] == "scalar_subquery_double_parentheses"
    assert _preset_config("scalar_subquery_double_parentheses_metamorphic").enable_metamorphic_oracle is True
    assert _preset_config("window_avg_rows_frame").generator_profile == "window_avg_rows_frame"
    assert _preset_config("window_avg_rows_frame").guidance_targets[0] == "window_avg_rows_frame"
    assert _preset_config("window_avg_rows_frame_metamorphic").enable_metamorphic_oracle is True
    assert _preset_config("struct_distinct_unnest").generator_profile == "struct_distinct_unnest"
    assert _preset_config("struct_distinct_unnest").guidance_targets[0] == "struct_distinct_unnest"
    assert _preset_config("struct_distinct_unnest_metamorphic").enable_metamorphic_oracle is True
    assert _preset_config("bit_compare_unequal_length").generator_profile == "bit_compare_unequal_length"
    assert _preset_config("bit_compare_unequal_length").guidance_targets[0] == "bit_compare_unequal_length"
    assert _preset_config("bit_compare_unequal_length_metamorphic").enable_metamorphic_oracle is True
    assert _preset_config("round_even_float_scale").generator_profile == "round_even_float_scale"
    assert _preset_config("round_even_float_scale").guidance_targets[0] == "round_even_float_scale"
    assert _preset_config("round_even_float_scale_metamorphic").enable_metamorphic_oracle is True
    assert _preset_config("series_rtruediv_operand_order").generator_profile == "series_rtruediv_operand_order"
    assert _preset_config("series_rtruediv_operand_order").guidance_targets[0] == "series_rtruediv_operand_order"
    assert _preset_config("series_rtruediv_operand_order_metamorphic").enable_metamorphic_oracle is True
    assert _preset_config("pandas_uint64_isin_precision").generator_profile == "pandas_uint64_isin_precision"
    assert _preset_config("pandas_uint64_isin_precision").guidance_targets[0] == "pandas_uint64_isin_precision"
    assert _preset_config("pandas_uint64_isin_precision_metamorphic").enable_metamorphic_oracle is True
    assert _preset_config("duckdb_tuple_anti_null_semantics").generator_profile == "duckdb_tuple_anti_null_semantics"
    assert _preset_config("duckdb_tuple_anti_null_semantics").guidance_targets[0] == "duckdb_tuple_anti_null_semantics"
    assert _preset_config("duckdb_tuple_anti_null_semantics_metamorphic").enable_metamorphic_oracle is True
    assert _preset_config("duckdb_json_predicate_order_semantics").generator_profile == "duckdb_json_predicate_order_semantics"
    assert _preset_config("duckdb_json_predicate_order_semantics").guidance_targets[0] == "duckdb_json_predicate_order_semantics"
    assert _preset_config("duckdb_json_predicate_order_semantics_metamorphic").enable_metamorphic_oracle is True
    assert _preset_config("pandas_sparse_array_mask_semantics").generator_profile == "pandas_sparse_array_mask_semantics"
    assert _preset_config("pandas_sparse_array_mask_semantics").guidance_targets[0] == "pandas_sparse_array_mask_semantics"
    assert _preset_config("pandas_sparse_array_mask_semantics_metamorphic").enable_metamorphic_oracle is True
    assert _preset_config("polars_float_wrap_numerical_semantics").generator_profile == "polars_float_wrap_numerical_semantics"
    assert _preset_config("polars_float_wrap_numerical_semantics").guidance_targets[0] == "polars_float_wrap_numerical_semantics"
    assert _preset_config("polars_float_wrap_numerical_semantics_metamorphic").enable_metamorphic_oracle is True
    assert _preset_config("pandas_index_bool_result_type").generator_profile == "pandas_index_bool_result_type"
    assert _preset_config("pandas_index_bool_result_type").guidance_targets[0] == "pandas_index_bool_result_type"
    assert _preset_config("pandas_index_bool_result_type_metamorphic").enable_metamorphic_oracle is True
    assert _preset_config("polars_empty_literal_groupby_semantics").generator_profile == "polars_empty_literal_groupby_semantics"
    assert (
        _preset_config("polars_empty_literal_groupby_semantics").guidance_targets[0]
        == "polars_empty_literal_groupby_semantics"
    )
    assert _preset_config("polars_empty_literal_groupby_semantics_metamorphic").enable_metamorphic_oracle is True
    assert _preset_config("pandas_arrow_string_eq_sum_semantics").generator_profile == "pandas_arrow_string_eq_sum_semantics"
    assert (
        _preset_config("pandas_arrow_string_eq_sum_semantics").guidance_targets[0]
        == "pandas_arrow_string_eq_sum_semantics"
    )
    assert _preset_config("pandas_arrow_string_eq_sum_semantics_metamorphic").enable_metamorphic_oracle is True
    assert (
        _preset_config("pandas_arrow_timestamp_loc_slice_semantics").generator_profile
        == "pandas_arrow_timestamp_loc_slice_semantics"
    )
    assert (
        _preset_config("pandas_arrow_timestamp_loc_slice_semantics").guidance_targets[0]
        == "pandas_arrow_timestamp_loc_slice_semantics"
    )
    assert _preset_config("pandas_arrow_timestamp_loc_slice_semantics_metamorphic").enable_metamorphic_oracle is True
    assert (
        _preset_config("pandas_arrow_timestamp_index_attr_semantics").generator_profile
        == "pandas_arrow_timestamp_index_attr_semantics"
    )
    assert (
        _preset_config("pandas_arrow_timestamp_index_attr_semantics").guidance_targets[0]
        == "pandas_arrow_timestamp_index_attr_semantics"
    )
    assert _preset_config("pandas_arrow_timestamp_index_attr_semantics_metamorphic").enable_metamorphic_oracle is True
    assert (
        _preset_config("pandas_eval_inplace_aliasing_semantics").generator_profile
        == "pandas_eval_inplace_aliasing_semantics"
    )
    assert (
        _preset_config("pandas_eval_inplace_aliasing_semantics").guidance_targets[0]
        == "pandas_eval_inplace_aliasing_semantics"
    )
    assert _preset_config("pandas_eval_inplace_aliasing_semantics_metamorphic").enable_metamorphic_oracle is True
    assert (
        _preset_config("pyarrow_dataset_isin_all_match_semantics").generator_profile
        == "pyarrow_dataset_isin_all_match_semantics"
    )
    assert (
        _preset_config("pyarrow_dataset_isin_all_match_semantics").guidance_targets[0]
        == "pyarrow_dataset_isin_all_match_semantics"
    )
    assert _preset_config("pyarrow_dataset_isin_all_match_semantics_metamorphic").enable_metamorphic_oracle is True
    assert (
        _preset_config("pyarrow_large_string_partition_schema_semantics").generator_profile
        == "pyarrow_large_string_partition_schema_semantics"
    )
    assert (
        _preset_config("pyarrow_large_string_partition_schema_semantics").guidance_targets[0]
        == "pyarrow_large_string_partition_schema_semantics"
    )
    assert _preset_config("pyarrow_large_string_partition_schema_semantics_metamorphic").enable_metamorphic_oracle is True
    assert (
        _preset_config("pyarrow_hash_pivot_wider_order_semantics").generator_profile
        == "pyarrow_hash_pivot_wider_order_semantics"
    )
    assert (
        _preset_config("pyarrow_hash_pivot_wider_order_semantics").guidance_targets[0]
        == "pyarrow_hash_pivot_wider_order_semantics"
    )
    assert _preset_config("pyarrow_hash_pivot_wider_order_semantics_metamorphic").enable_metamorphic_oracle is True
    assert (
        _preset_config("polars_rolling_mean_by_null_count_semantics").generator_profile
        == "polars_rolling_mean_by_null_count_semantics"
    )
    assert (
        _preset_config("polars_rolling_mean_by_null_count_semantics").guidance_targets[0]
        == "polars_rolling_mean_by_null_count_semantics"
    )
    assert _preset_config("polars_rolling_mean_by_null_count_semantics_metamorphic").enable_metamorphic_oracle is True
    assert _preset_config("join_null_sort").generator_profile == "join_null_sort"
    assert _preset_config("join_null_sort").guidance_targets[0] == "join_null_sort"
    assert _preset_config("join_null_sort_metamorphic").enable_metamorphic_oracle is True


def test_cli_parses_bughunt_guided_metamorphic_preset():
    config = _preset_config("bughunt_guided_metamorphic")
    assert config.generator_profile == "bughunt"
    assert config.enable_metamorphic_oracle is True
    assert config.guidance_strategy == "guided"
    assert config.metamorphic_variant_limit == 8


def test_cli_parses_bughunt_no_groupby_guided_metamorphic_preset():
    config = _preset_config("bughunt_no_groupby_guided_metamorphic")
    assert config.generator_profile == "bughunt_no_groupby"
    assert config.enable_metamorphic_oracle is True
    assert config.guidance_strategy == "guided"
    assert "groupby" not in config.guidance_targets
    assert config.metamorphic_variant_limit == 8


def test_cli_parses_bughunt_experiment_presets():
    assert _preset_config("bughunt").generator_profile == "bughunt"
    assert _preset_config("bughunt_no_groupby").generator_profile == "bughunt_no_groupby"
    guided = _preset_config("bughunt_guided")
    assert guided.generator_profile == "bughunt"
    assert guided.guidance_strategy == "guided"
    assert guided.guidance_targets == ["join", "groupby", "mutate", "filter", "expressions"]
    no_groupby_guided = _preset_config("bughunt_no_groupby_guided")
    assert no_groupby_guided.generator_profile == "bughunt_no_groupby"
    assert no_groupby_guided.guidance_targets == ["join", "mutate", "filter", "expressions", "sort_limit"]
    metamorphic = _preset_config("bughunt_metamorphic")
    assert metamorphic.generator_profile == "bughunt"
    assert metamorphic.enable_metamorphic_oracle is True
    no_groupby_metamorphic = _preset_config("bughunt_no_groupby_metamorphic")
    assert no_groupby_metamorphic.generator_profile == "bughunt_no_groupby"
    assert no_groupby_metamorphic.enable_metamorphic_oracle is True


def test_cli_parses_live_datafusion_presets():
    live = _preset_config("live_datafusion")
    assert live.generator_profile == "bughunt"
    assert live.guidance_strategy == "guided"
    assert live.guidance_candidate_pool == 12
    assert live.enable_local_source_scheduler is True
    assert live.local_source_exploration_weight == 0.35
    assert {"common_workflow", "operation_combo", "topk", "join", "groupby", "join_null_key_topk"}.issubset(live.guidance_targets)

    metamorphic = _preset_config("live_datafusion_metamorphic")
    assert metamorphic.enable_metamorphic_oracle is True
    assert metamorphic.oracle_mode == "both"
    assert metamorphic.metamorphic_variant_limit == 6


def test_cli_parses_non_datafusion_live_presets():
    arrow = _preset_config("live_arrow")
    assert arrow.generator_profile == "bughunt"
    assert arrow.guidance_strategy == "guided"
    assert {
        "join",
        "groupby",
        "strings",
        "casts",
        "topk",
        "global_null_aggregate",
        "string_count_groupby",
        "unique_count_groupby",
        "set_membership_filter",
        "null_predicate_filter",
        "boolean_predicate_filter",
        "post_topk_range_filter",
        "tuple_absence_filter",
        "running_sum_precision",
        "sortedness_null_placement",
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
        "duckdb_json_predicate_order_semantics",
        "pandas_sparse_array_mask_semantics",
        "polars_float_wrap_numerical_semantics",
        "pandas_index_bool_result_type",
        "polars_empty_literal_groupby_semantics",
        "pandas_arrow_string_eq_sum_semantics",
        "pandas_arrow_timestamp_loc_slice_semantics",
        "pandas_arrow_timestamp_index_attr_semantics",
        "pandas_eval_inplace_aliasing_semantics",
        "pyarrow_dataset_isin_all_match_semantics",
        "pyarrow_large_string_partition_schema_semantics",
        "pyarrow_hash_pivot_wider_order_semantics",
        "polars_rolling_mean_by_null_count_semantics",
    }.issubset(arrow.guidance_targets)
    assert arrow.enable_local_source_scheduler is True

    polars_lazy = _preset_config("live_polars_lazy")
    assert polars_lazy.generator_profile == "bughunt"
    assert {"join", "filter", "mutate", "sort_limit", "topk", "global_aggregation"}.issubset(polars_lazy.guidance_targets)
    assert polars_lazy.local_source_exploration_weight == 0.45

    embedded_sql = _preset_config("live_embedded_sql")
    assert embedded_sql.generator_profile == "bughunt"
    assert {"join", "filter", "groupby", "aggregation", "casts"}.issubset(embedded_sql.guidance_targets)

    cross_family = _preset_config("live_cross_family")
    assert cross_family.generator_profile == "bughunt"
    assert {"common_workflow", "operation_combo", "join", "groupby", "topk"}.issubset(cross_family.guidance_targets)


def test_cli_parses_non_datafusion_live_metamorphic_presets():
    for preset in [
        "live_arrow_metamorphic",
        "live_polars_lazy_metamorphic",
        "live_embedded_sql_metamorphic",
        "live_cross_family_metamorphic",
    ]:
        config = _preset_config(preset)
        assert config.enable_metamorphic_oracle is True
        assert config.oracle_mode == "both"
        assert config.guidance_strategy == "guided"
        assert config.enable_local_source_scheduler is True


def test_cli_experiment_duration_can_run_without_case_cap():
    parser = build_parser()
    args = parser.parse_args(["experiment", "--duration", "1s", "--seeds", "1"])
    assert args.cmd == "experiment"
    assert args.cases is None
    assert args.duration == "1s"


def test_cli_parses_analyze_experiment_command():
    parser = build_parser()
    args = parser.parse_args(
        [
            "analyze-experiment",
            "--manifest",
            "runs/experiment-x.json",
            "--baseline-preset",
            "baseline",
            "--compare-presets",
            "guided_filter,guided_join",
            "--refresh",
        ]
    )
    assert args.cmd == "analyze-experiment"
    assert args.manifest == "runs/experiment-x.json"
    assert args.baseline_preset == "baseline"
    assert args.compare_presets == "guided_filter,guided_join"
    assert args.refresh is True


def test_cli_parses_analyze_seeded_sensitivity_command():
    parser = build_parser()
    args = parser.parse_args(
        [
            "analyze-seeded-sensitivity",
            "--manifest",
            "runs/experiment-seeded.json",
        ]
    )
    assert args.cmd == "analyze-seeded-sensitivity"
    assert args.manifest == "runs/experiment-seeded.json"


def test_cli_parses_analyze_ablation_audit_command():
    parser = build_parser()
    args = parser.parse_args(
        [
            "analyze-ablation-audit",
            "--manifest",
            "runs/experiment-ablation.json",
            "--trusted-presets",
            "baseline,guided",
            "--ablation-presets",
            "no_type_aware,no_normalizer",
            "--refresh",
        ]
    )
    assert args.cmd == "analyze-ablation-audit"
    assert args.manifest == "runs/experiment-ablation.json"
    assert args.trusted_presets == "baseline,guided"
    assert args.ablation_presets == "no_type_aware,no_normalizer"
    assert args.refresh is True


def test_cli_parses_analyze_pattern_variants_command():
    parser = build_parser()
    args = parser.parse_args(
        [
            "analyze-pattern-variants",
            "--manifest",
            "runs/experiment-pattern.json",
            "--pattern",
            "null_agg_topk",
        ]
    )
    assert args.cmd == "analyze-pattern-variants"
    assert args.manifest == "runs/experiment-pattern.json"
    assert args.pattern == "null_agg_topk"


def test_classify_run_refresh_recomputes_current_oracle_roots(tmp_path, capsys):
    case = Case(
        "case-stale-root",
        14,
        [TableData("t0", [ColumnSpec("x", "int")], [{"x": 0}, {"x": 0}])],
        Program(
            "prog-stale-root",
            14,
            [
                {"op": "mutate", "column": "m_0", "expr": {"kind": "add_const", "source": "x", "value": -1}},
                {"op": "filter", "column": "m_0", "cmp": "==", "value": -1},
                {"op": "mutate", "column": "m_1", "expr": {"kind": "arith_const", "op": "mul", "source": "m_0", "value": 10}},
                {"op": "mutate", "column": "m_3", "expr": {"kind": "arith_const", "op": "div", "source": "m_1", "value": 3}},
                {"op": "groupby", "keys": ["m_3"], "aggs": [{"column": "m_0", "func": "min", "as": "min_m_0"}]},
            ],
        ),
    )
    run_file = tmp_path / "run-stale.jsonl"
    append_jsonl(
        {
            "status": "bug",
            "case": case.to_dict(),
            "findings": [
                {
                    "kind": "semantic_output_mismatch",
                    "root_cause": "groupby_aggregation",
                    "triage_verdict": "candidate_implementation_bug",
                    "suspicious_backends": ["b"],
                    "signature": "stale",
                    "confidence": "high",
                }
            ],
            "normalized": {
                "a": {"backend": "a", "status": "ok", "columns": ["m_3", "min_m_0"], "rows": [[-3.3333333333, -1]]},
                "c": {"backend": "c", "status": "ok", "columns": ["m_3", "min_m_0"], "rows": [[-3.3333333333, -1]]},
                "b": {
                    "backend": "b",
                    "status": "ok",
                    "columns": ["m_3", "min_m_0"],
                    "rows": [[-3.3333333333, -1], [-3.3333333333, -1]],
                },
            },
        },
        run_file,
    )

    parser = build_parser()
    args = parser.parse_args(["classify-run", "--run-file", str(run_file), "--refresh"])
    assert args.func(args) == 0
    out = capsys.readouterr().out

    assert "refresh=true" in out
    assert "float_group_key_instability@b: 1" in out
    assert "root=float_group_key_instability" in out


def test_cli_experiment_parses_no_compress_run_log():
    parser = build_parser()
    args = parser.parse_args(["experiment", "--no-compress-run-log"])
    assert args.cmd == "experiment"
    assert args.no_compress_run_log is True


def test_cli_experiment_parses_artifact_limit():
    parser = build_parser()
    args = parser.parse_args(["experiment", "--artifact-limit", "20"])
    assert args.cmd == "experiment"
    assert args.artifact_limit == 20


def test_cli_experiment_parses_skip_run_reports():
    parser = build_parser()
    args = parser.parse_args(["experiment", "--skip-run-reports"])
    assert args.cmd == "experiment"
    assert args.skip_run_reports is True


def test_cli_experiment_parses_jobs():
    parser = build_parser()
    args = parser.parse_args(["experiment", "--jobs", "4"])
    assert args.cmd == "experiment"
    assert args.jobs == 4


def test_cli_experiment_parses_auto_jobs_and_parallel_cost():
    parser = build_parser()
    args = parser.parse_args(["experiment", "--jobs", "auto", "--max-parallel-cost", "9.5"])
    assert args.cmd == "experiment"
    assert args.jobs == "auto"
    assert args.max_parallel_cost == 9.5


def test_cli_experiment_parses_adaptive_schedule_flags():
    parser = build_parser()
    args = parser.parse_args(
        [
            "experiment",
            "--schedule",
            "adaptive",
            "--batch-cases",
            "25",
            "--batch-duration",
            "30s",
            "--warmup-batches",
            "2",
            "--exploration-weight",
            "0.5",
            "--local-source-exploration-weight",
            "0.2",
        ]
    )
    assert args.cmd == "experiment"
    assert args.schedule == "adaptive"
    assert args.batch_cases == 25
    assert args.batch_duration == "30s"
    assert args.warmup_batches == 2
    assert args.exploration_weight == 0.5
    assert args.local_source_exploration_weight == 0.2


def test_cli_experiment_parses_evidence_mode_flags():
    parser = build_parser()
    args = parser.parse_args(
        [
            "experiment",
            "--evidence-mode",
            "historical",
            "--known-bug-id",
            "datafusion-22190",
            "--target-version",
            "pre-fix-sha",
        ]
    )
    assert args.evidence_mode == "historical"
    assert args.known_bug_id == "datafusion-22190"
    assert args.target_version == "pre-fix-sha"


def test_cli_parses_paper_run_journal_flags():
    parser = build_parser()
    args = parser.parse_args(
        [
            "experiment",
            "--run-theme",
            "final-live:datafusion",
            "--paper-notes",
            "24h latest-version run",
            "--skip-paper-journal",
        ]
    )
    assert args.run_theme == "final-live:datafusion"
    assert args.paper_notes == "24h latest-version run"
    assert args.skip_paper_journal is True


def test_run_experiment_job_propagates_local_source_scheduler(monkeypatch):
    captured = {}

    def fake_run_fuzz(*, cases, seed, backends, config, duration_s):
        captured["enable_local_source_scheduler"] = config.enable_local_source_scheduler
        captured["local_source_exploration_weight"] = config.local_source_exploration_weight
        return Path("runs/fake.jsonl")

    monkeypatch.setattr(cli, "run_fuzz", fake_run_fuzz)
    monkeypatch.setattr(cli, "write_report", lambda run_file: (Path(""), Path("")))

    result = cli._run_experiment_job(
        {
            "order": 0,
            "target_suite": "core",
            "backends": ["pandas"],
            "preset": "baseline",
            "seed": 1,
            "cases": 1,
            "duration_s": None,
            "log_level": "compact",
            "compress_run_log": True,
            "artifact_limit": None,
            "metamorphic_variant_limit": None,
            "enable_local_source_scheduler": True,
            "local_source_exploration_weight": 0.125,
            "skip_run_reports": True,
        }
    )

    assert captured["enable_local_source_scheduler"] is True
    assert captured["local_source_exploration_weight"] == 0.125
    assert result["run"]["run_file"] == "runs/fake.jsonl"


def test_run_experiment_job_preserves_live_preset_source_scheduler(monkeypatch):
    captured = {}

    def fake_run_fuzz(*, cases, seed, backends, config, duration_s):
        captured["enable_local_source_scheduler"] = config.enable_local_source_scheduler
        captured["local_source_exploration_weight"] = config.local_source_exploration_weight
        captured["guidance_candidate_pool"] = config.guidance_candidate_pool
        return Path("runs/fake-live.jsonl")

    monkeypatch.setattr(cli, "run_fuzz", fake_run_fuzz)
    monkeypatch.setattr(cli, "write_report", lambda run_file: (Path(""), Path("")))

    cli._run_experiment_job(
        {
            "order": 0,
            "target_suite": "datafusion_cross",
            "backends": ["pandas", "duckdb", "datafusion"],
            "preset": "live_datafusion",
            "seed": 1,
            "cases": 1,
            "duration_s": None,
            "log_level": "compact",
            "compress_run_log": True,
            "artifact_limit": None,
            "metamorphic_variant_limit": None,
            "enable_local_source_scheduler": False,
            "local_source_exploration_weight": 0.5,
            "skip_run_reports": True,
        }
    )

    assert captured["enable_local_source_scheduler"] is True
    assert captured["local_source_exploration_weight"] == 0.35
    assert captured["guidance_candidate_pool"] == 12


def test_cli_experiment_static_manifest_records_local_source_scheduler(tmp_path, monkeypatch, capsys):
    runs_dir = tmp_path / "runs"
    reports_dir = tmp_path / "reports"
    bugs_dir = tmp_path / "bugs"
    corpus_dir = tmp_path / "corpus"
    monkeypatch.setattr(cli, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(cli, "REPORTS_DIR", reports_dir)
    monkeypatch.setattr(cli, "BUGS_DIR", bugs_dir)
    monkeypatch.setattr(cli, "CORPUS_DIR", corpus_dir)

    def fake_run_fuzz(*, cases, seed, backends, config, duration_s):
        run_file = runs_dir / "run-static.jsonl"
        append_jsonl(
            {
                "case": {"case_id": "case-0", "seed": seed},
                "is_new_behavior": False,
                "findings": [],
            },
            run_file,
        )
        meta_path = Path(str(run_file).replace(".jsonl", ".meta.json"))
        meta_path.write_text(
            json.dumps(
                {
                    "elapsed_s": 0.1,
                    "throughput_cases_s": 10.0,
                    "next_seed": seed + (cases or 1),
                }
            ),
            encoding="utf-8",
        )
        return run_file

    monkeypatch.setattr(cli, "run_fuzz", fake_run_fuzz)
    monkeypatch.setattr(cli, "write_report", lambda run_file: (Path(""), Path("")))

    parser = build_parser()
    args = parser.parse_args(
        [
            "experiment",
            "--target-suites",
            "core",
            "--presets",
            "baseline",
            "--seeds",
            "1",
            "--cases",
            "1",
            "--enable-local-source-scheduler",
            "--local-source-exploration-weight",
            "0.25",
            "--skip-run-reports",
        ]
    )

    assert args.func(args) == 0
    manifests = sorted(runs_dir.glob("experiment-*.json"))
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert manifest["schedule"] == "matrix_order"
    assert manifest["evidence_mode"] == "live"
    assert manifest["local_source_scheduler"]["enabled"] is True
    assert manifest["local_source_scheduler"]["exploration_weight"] == 0.25


def test_cli_experiment_manifest_records_historical_evidence_metadata(tmp_path, monkeypatch, capsys):
    runs_dir = tmp_path / "runs"
    reports_dir = tmp_path / "reports"
    monkeypatch.setattr(cli, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(cli, "REPORTS_DIR", reports_dir)

    def fake_run_fuzz(*, cases, seed, backends, config, duration_s):
        run_file = runs_dir / "run-historical.jsonl"
        append_jsonl({"case": {"case_id": "case-0", "seed": seed}, "findings": []}, run_file)
        run_meta_path(run_file).write_text(
            json.dumps({"elapsed_s": 0.1, "throughput_cases_s": 10.0, "next_seed": seed + 1}),
            encoding="utf-8",
        )
        return run_file

    monkeypatch.setattr(cli, "run_fuzz", fake_run_fuzz)
    monkeypatch.setattr(cli, "write_report", lambda run_file: (Path(""), Path("")))

    parser = build_parser()
    args = parser.parse_args(
        [
            "experiment",
            "--target-suites",
            "core",
            "--presets",
            "baseline",
            "--seeds",
            "1",
            "--cases",
            "1",
            "--evidence-mode",
            "historical",
            "--known-bug-id",
            "datafusion-22190",
            "--target-version",
            "pre-fix-sha",
            "--skip-run-reports",
        ]
    )

    assert args.func(args) == 0
    manifest = json.loads(next(runs_dir.glob("experiment-*.json")).read_text(encoding="utf-8"))
    assert manifest["evidence_mode"] == "historical"
    assert manifest["known_bug_id"] == "datafusion-22190"
    assert manifest["target_version"] == "pre-fix-sha"
    assert manifest["runs"][0]["evidence_mode"] == "historical"


def test_cli_replay_fixture_records_single_case_run_and_journal(tmp_path, monkeypatch, capsys):
    pyarrow = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    from datadiff.fixture_replay import fixture_sha256

    runs_dir = tmp_path / "runs"
    reports_dir = tmp_path / "reports"
    monkeypatch.setattr(cli, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(cli, "REPORTS_DIR", reports_dir)
    monkeypatch.setattr(cli, "BUGS_DIR", tmp_path / "bugs")
    monkeypatch.setattr(cli, "CORPUS_DIR", tmp_path / "corpus")

    fixture_path = tmp_path / "fixture.parquet"
    table = pyarrow.table(
        {
            "a": pyarrow.array([None, 2, 1], type=pyarrow.int64()),
            "b": pyarrow.array(["n", "z", "a"], type=pyarrow.string()),
        }
    )
    pq.write_table(table, fixture_path)
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(
        json.dumps(
            {
                "case_id": "case-cli-fixture",
                "seed": 77,
                "target_suite": "cross_family",
                "fixture": {
                    "name": "declared",
                    "columns": ["a", "b"],
                    "sha256": fixture_sha256(fixture_path),
                },
                "operations": [
                    {
                        "op": "sort",
                        "keys": [
                            {"column": "a", "ascending": True, "nulls": "first"},
                            {"column": "b", "ascending": True, "nulls": "last"},
                        ],
                    },
                    {"op": "limit", "n": 2},
                ],
            }
        ),
        encoding="utf-8",
    )

    parser = build_parser()
    args = parser.parse_args(
        [
            "replay-fixture",
            "--spec",
            str(spec_path),
            "--fixture",
            str(fixture_path),
            "--backends",
            "pandas",
            "--known-bug-id",
            "fixture-test",
            "--target-version",
            "target==1.0",
            "--artifact-limit",
            "0",
            "--no-compress-run-log",
        ]
    )

    assert args.func(args) == 0
    run_file = next(runs_dir.glob("run-fixture-*.jsonl"))
    meta = json.loads(run_meta_path(run_file).read_text(encoding="utf-8"))
    journal = (reports_dir / "paper-run-journal.jsonl").read_text(encoding="utf-8")
    output = capsys.readouterr().out
    assert "status=ok" in output
    assert meta["preset"] == "fixture_replay"
    assert meta["evidence_mode"] == "historical"
    assert meta["known_bug_id"] == "fixture-test"
    assert meta["fixture_sha256"] == fixture_sha256(fixture_path)
    assert '"known_bug_id": "fixture-test"' in journal


def test_cli_historical_status_marks_counted_and_pending(capsys):
    parser = build_parser()
    args = parser.parse_args(["historical-status", "--include-pending", "--json"])

    assert args.func(args) == 0
    payload = json.loads(capsys.readouterr().out)
    by_id = {row["bug_id"]: row for row in payload["historical_bugs"]}
    assert by_id["duckdb-22075"]["counted"] is True
    assert by_id["duckdb-22656"]["counted"] is True
    assert by_id["duckdb-22656"]["target_suite"] == "duckdb_storage_cross"
    assert by_id["duckdb-3015"]["counted"] is False
    assert by_id["duckdb-11261"]["counted"] is False
    assert by_id["duckdb-11261"]["target_suite"] == "duckdb_storage_cross"
    assert by_id["arrow-42231"]["counted"] is False
    assert by_id["arrow-42231"]["target_suite"] == "arrow_cross"
    assert by_id["duckdb-3015"]["replay_kind"] == "fixture"
    assert by_id["duckdb-3015"]["fixture_env_status"] in {"set", "unset"}


def test_cli_experiment_manifest_names_do_not_collide_within_same_second(tmp_path, monkeypatch):
    runs_dir = tmp_path / "runs"
    reports_dir = tmp_path / "reports"
    bugs_dir = tmp_path / "bugs"
    corpus_dir = tmp_path / "corpus"
    monkeypatch.setattr(cli, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(cli, "REPORTS_DIR", reports_dir)
    monkeypatch.setattr(cli, "BUGS_DIR", bugs_dir)
    monkeypatch.setattr(cli, "CORPUS_DIR", corpus_dir)
    monkeypatch.setattr(cli, "utc_now", lambda: "2026-05-19T15:42:27Z")
    ticks = iter([111, 222])
    monkeypatch.setattr(cli.time, "time_ns", lambda: next(ticks))

    def fake_run_fuzz(*, cases, seed, backends, config, duration_s):
        run_file = runs_dir / f"run-{seed}.jsonl"
        append_jsonl(
            {
                "case": {"case_id": f"case-{seed}", "seed": seed},
                "is_new_behavior": False,
                "findings": [],
            },
            run_file,
        )
        meta_path = Path(str(run_file).replace(".jsonl", ".meta.json"))
        meta_path.write_text(
            json.dumps(
                {
                    "elapsed_s": 0.1,
                    "throughput_cases_s": 10.0,
                    "next_seed": seed + (cases or 1),
                }
            ),
            encoding="utf-8",
        )
        return run_file

    monkeypatch.setattr(cli, "run_fuzz", fake_run_fuzz)
    monkeypatch.setattr(cli, "write_report", lambda run_file: (Path(""), Path("")))

    parser = build_parser()
    argv = [
        "experiment",
        "--target-suites",
        "core",
        "--presets",
        "baseline",
        "--seeds",
        "1",
        "--cases",
        "1",
        "--skip-run-reports",
    ]

    args = parser.parse_args(argv)
    assert args.func(args) == 0
    args = parser.parse_args(argv)
    assert args.func(args) == 0

    manifests = sorted(runs_dir.glob("experiment-*.json"))
    assert [path.name for path in manifests] == [
        "experiment-20260519T154227-111.json",
        "experiment-20260519T154227-222.json",
    ]


def test_experiment_parallel_scheduler_starts_heavy_jobs_first():
    fast = {
        "order": 0,
        "preset": "null_groupby_topk",
        "backends": ["pandas", "duckdb", "datafusion"],
    }
    slow = {
        "order": 1,
        "preset": "float_group_key_metamorphic",
        "backends": ["pandas", "polars", "polars_lazy", "duckdb", "sqlite", "datafusion"],
    }

    assert sorted([fast, slow], key=cli._experiment_job_sort_key) == [slow, fast]


def test_experiment_parallelism_auto_uses_bounded_workers(monkeypatch):
    monkeypatch.setattr(cli.os, "cpu_count", lambda: 20)
    parser = build_parser()
    args = parser.parse_args(["experiment", "--jobs", "auto"])
    planned = [
        {"order": 0, "preset": "live_cross_family", "backends": ["pandas", "duckdb", "datafusion"]},
        {"order": 1, "preset": "live_polars_lazy", "backends": ["polars", "polars_lazy"]},
        {"order": 2, "preset": "live_arrow", "backends": ["pandas", "duckdb", "pyarrow"]},
        {"order": 3, "preset": "baseline", "backends": ["pandas", "duckdb"]},
        {"order": 4, "preset": "baseline", "backends": ["pandas", "sqlite"]},
    ]

    parallelism = cli._resolve_experiment_parallelism(args, planned)

    assert parallelism["worker_count"] == 5
    assert parallelism["bounded_submission"] is True
    assert parallelism["cost_limited"] is True
    assert parallelism["worker_thread_limit"] == 4


def test_cli_experiment_adaptive_scheduler_reuses_budget_on_high_yield_arm(tmp_path, monkeypatch, capsys):
    runs_dir = tmp_path / "runs"
    reports_dir = tmp_path / "reports"
    bugs_dir = tmp_path / "bugs"
    corpus_dir = tmp_path / "corpus"
    monkeypatch.setattr(cli, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(cli, "REPORTS_DIR", reports_dir)
    monkeypatch.setattr(cli, "BUGS_DIR", bugs_dir)
    monkeypatch.setattr(cli, "CORPUS_DIR", corpus_dir)

    counter = {"index": 0}

    def fake_run_fuzz(*, cases, seed, backends, config, duration_s):
        idx = counter["index"]
        counter["index"] += 1
        assert config.enable_local_source_scheduler is True
        assert config.local_source_exploration_weight == 0.125
        run_file = runs_dir / f"run-{idx}.jsonl"
        target_suite = "datafusion_cross" if "datafusion" in backends else "core"
        if target_suite == "datafusion_cross":
            append_jsonl(
                {
                    "case": {"case_id": f"case-{idx}", "seed": seed},
                    "is_new_behavior": True,
                    "findings": [
                        {
                            "root_cause": "grouped_topk_null_sort_key",
                            "triage_verdict": "candidate_implementation_bug",
                            "suspicious_backends": ["datafusion"],
                            "signature": f"sig-{idx}",
                        }
                    ],
                },
                run_file,
            )
        else:
            append_jsonl(
                {
                    "case": {"case_id": f"case-{idx}", "seed": seed},
                    "is_new_behavior": False,
                    "findings": [],
                },
                run_file,
            )
        meta_path = Path(str(run_file).replace(".jsonl", ".meta.json"))
        meta_path.write_text(
            json.dumps(
                {
                    "elapsed_s": 0.1,
                    "throughput_cases_s": 10.0,
                    "next_seed": seed + cases,
                }
            ),
            encoding="utf-8",
        )
        return run_file

    monkeypatch.setattr(cli, "run_fuzz", fake_run_fuzz)
    monkeypatch.setattr(cli, "write_report", lambda run_file: (Path(""), Path("")))

    parser = build_parser()
    args = parser.parse_args(
        [
            "experiment",
            "--target-suites",
            "core,datafusion_cross",
            "--presets",
            "baseline",
            "--seeds",
            "1",
            "--cases",
            "2",
            "--schedule",
            "adaptive",
            "--batch-cases",
            "1",
            "--jobs",
            "1",
            "--local-source-exploration-weight",
            "0.125",
            "--skip-run-reports",
        ]
    )

    assert args.func(args) == 0
    out = capsys.readouterr().out
    assert "reward=" in out

    manifests = sorted(runs_dir.glob("experiment-*.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert manifest["schedule"] == "adaptive"
    assert manifest["adaptive_config"]["fine_grained_local_source_scheduler"] is True
    assert manifest["adaptive_config"]["local_source_exploration_weight"] == 0.125
    suites = [run["target_suite"] for run in manifest["runs"]]
    assert suites.count("datafusion_cross") > suites.count("core")


def test_cli_experiment_adaptive_scheduler_supports_parallel_rounds(tmp_path, monkeypatch, capsys):
    runs_dir = tmp_path / "runs"
    reports_dir = tmp_path / "reports"
    bugs_dir = tmp_path / "bugs"
    corpus_dir = tmp_path / "corpus"
    monkeypatch.setattr(cli, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(cli, "REPORTS_DIR", reports_dir)
    monkeypatch.setattr(cli, "BUGS_DIR", bugs_dir)
    monkeypatch.setattr(cli, "CORPUS_DIR", corpus_dir)
    monkeypatch.setattr(cli, "ProcessPoolExecutor", cli.ThreadPoolExecutor)

    counter = {"index": 0}

    def fake_run_fuzz(*, cases, seed, backends, config, duration_s):
        idx = counter["index"]
        counter["index"] += 1
        run_file = runs_dir / f"run-{idx}.jsonl"
        target_suite = (
            "datafusion_cross"
            if "datafusion" in backends
            else "dataframe"
            if "polars" in backends
            else "core"
        )
        if target_suite == "datafusion_cross":
            payload = {
                "case": {"case_id": f"case-{idx}", "seed": seed},
                "is_new_behavior": True,
                "findings": [
                    {
                        "root_cause": "grouped_topk_null_sort_key",
                        "triage_verdict": "candidate_implementation_bug",
                        "suspicious_backends": ["datafusion"],
                        "signature": f"sig-{idx}",
                    }
                ],
            }
        elif target_suite == "dataframe":
            payload = {
                "case": {"case_id": f"case-{idx}", "seed": seed},
                "is_new_behavior": True,
                "findings": [],
            }
        else:
            payload = {
                "case": {"case_id": f"case-{idx}", "seed": seed},
                "is_new_behavior": False,
                "findings": [],
            }
        append_jsonl(payload, run_file)
        meta_path = Path(str(run_file).replace(".jsonl", ".meta.json"))
        meta_path.write_text(
            json.dumps(
                {
                    "elapsed_s": 0.1,
                    "throughput_cases_s": 10.0,
                    "next_seed": seed + cases,
                }
            ),
            encoding="utf-8",
        )
        return run_file

    monkeypatch.setattr(cli, "run_fuzz", fake_run_fuzz)
    monkeypatch.setattr(cli, "write_report", lambda run_file: (Path(""), Path("")))

    parser = build_parser()
    args = parser.parse_args(
        [
            "experiment",
            "--target-suites",
            "core,dataframe,datafusion_cross",
            "--presets",
            "baseline",
            "--seeds",
            "1",
            "--cases",
            "2",
            "--schedule",
            "adaptive",
            "--batch-cases",
            "1",
            "--jobs",
            "2",
            "--skip-run-reports",
        ]
    )

    assert args.func(args) == 0
    out = capsys.readouterr().out
    assert "reward=" in out

    manifests = sorted(runs_dir.glob("experiment-*.json"))
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert manifest["adaptive_config"]["jobs"] == 2
    suites = [run["target_suite"] for run in manifest["runs"]]
    assert suites.count("datafusion_cross") >= suites.count("core")
    assert any(run["scheduler_reward"] > 0 for run in manifest["runs"])


def test_cli_experiment_summary_parses_refresh():
    parser = build_parser()
    args = parser.parse_args(["experiment-summary", "--refresh"])
    assert args.cmd == "experiment-summary"
    assert args.refresh is True


def test_cli_parses_longrun_defaults():
    parser = build_parser()
    args = parser.parse_args(["longrun"])
    assert args.cmd == "longrun"
    assert args.duration == "24h"
    assert args.cases is None
    assert args.strategy == "guided"
    assert args.candidate_pool == 8
    assert args.checkpoint_interval == "60s"
    assert args.progress_interval == "60s"
    assert args.save_cases is False
    assert args.no_save_cases is False
    assert args.log_level == "compact"


def test_cli_parses_experiment_summary_command():
    parser = build_parser()
    args = parser.parse_args(["experiment-summary", "--manifest", "runs/experiment-x.json"])
    assert args.cmd == "experiment-summary"
    assert args.manifest == "runs/experiment-x.json"


def test_cli_parses_report_csv_limit():
    parser = build_parser()
    args = parser.parse_args(["report", "--run-file", "runs/run-x.jsonl.gz", "--csv-limit", "100"])
    assert args.cmd == "report"
    assert args.csv_limit == 100


def test_cli_parses_artifact_validation_command():
    parser = build_parser()
    args = parser.parse_args(["validate-artifact", "--bug", "bugs/bug_x"])
    assert args.cmd == "validate-artifact"
    assert args.bug == "bugs/bug_x"


def test_cli_parses_classify_run_command():
    parser = build_parser()
    args = parser.parse_args(["classify-run", "--run-file", "runs/run-x.jsonl", "--limit", "2"])
    assert args.cmd == "classify-run"
    assert args.run_file == "runs/run-x.jsonl"
    assert args.limit == 2


def test_cli_classify_run_reads_compressed_jsonl(tmp_path, capsys):
    run_file = tmp_path / "run-x.jsonl.gz"
    append_jsonl({"case": {"case_id": "case-x", "seed": 1}, "findings": []}, run_file)

    parser = build_parser()
    args = parser.parse_args(["classify-run", "--run-file", str(run_file), "--limit", "2"])

    assert args.func(args) == 0
    out = capsys.readouterr().out
    assert f"run_file={run_file}" in out
    assert "- none" in out


def test_cli_classify_run_reports_candidate_bug_families(tmp_path, capsys):
    run_file = tmp_path / "run-family.jsonl.gz"
    append_jsonl(
        {
            "case": {"case_id": "case-x", "seed": 1},
            "findings": [
                {
                    "kind": "semantic_output_mismatch",
                    "root_cause": "grouped_topk_null_sort_key",
                    "triage_verdict": "candidate_implementation_bug",
                    "suspicious_backends": ["datafusion"],
                    "signature": "sig-x",
                }
            ],
        },
        run_file,
    )

    parser = build_parser()
    args = parser.parse_args(["classify-run", "--run-file", str(run_file), "--limit", "1"])

    assert args.func(args) == 0
    out = capsys.readouterr().out
    assert "candidate bug families:" in out
    assert "- grouped_topk_null_sort_key@datafusion: 1" in out


def test_cli_parses_artifact_triage_command():
    parser = build_parser()
    args = parser.parse_args(
        ["triage-artifact", "--bug", "bugs/bug_x", "--reduce", "--standalone-reproducer"]
    )
    assert args.cmd == "triage-artifact"
    assert args.bug == "bugs/bug_x"
    assert args.reduce is True
    assert args.standalone_reproducer is True


def test_cli_triage_artifact_writes_datafusion_standalone(tmp_path, monkeypatch, capsys):
    bug_dir = tmp_path / "bug_datafusion"
    bug_dir.mkdir()
    case = Case(
        "case-datafusion",
        1,
        [TableData("t0", [ColumnSpec("g", "str"), ColumnSpec("x", "int", nullable=True)], [{"g": "a", "x": None}])],
        Program(
            "prog-datafusion",
            1,
            [
                {"op": "groupby", "keys": ["g"], "aggs": [{"column": "x", "func": "min", "as": "min_x"}]},
                {"op": "sort", "columns": ["min_x"], "ascending": True},
                {"op": "limit", "n": 20},
            ],
        ),
    )
    (bug_dir / "case.json").write_text(json.dumps(case.to_dict()), encoding="utf-8")
    (bug_dir / "findings.json").write_text(
        json.dumps(
            [
                {
                    "kind": "semantic_output_mismatch",
                    "root_cause": "grouped_topk_null_sort_key",
                    "confidence": "high",
                    "suspicious_backends": ["datafusion"],
                }
            ]
        ),
        encoding="utf-8",
    )
    (bug_dir / "config.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        cli,
        "run_loaded_case",
        lambda *args, **kwargs: {
            "status": "bug",
            "findings": [
                {
                    "kind": "semantic_output_mismatch",
                    "root_cause": "grouped_topk_null_sort_key",
                    "confidence": "high",
                    "suspicious_backends": ["datafusion"],
                }
            ],
        },
    )

    parser = build_parser()
    args = parser.parse_args(
        [
            "triage-artifact",
            "--bug",
            str(bug_dir),
            "--backends",
            "pandas,duckdb,datafusion",
            "--standalone-reproducer",
        ]
    )

    assert args.func(args) == 0
    out = capsys.readouterr().out
    standalone = bug_dir / "standalone_datafusion_groupby_null_sortkey_limit.py"
    triage = json.loads((bug_dir / "triage.json").read_text(encoding="utf-8"))
    assert f"standalone_reproducer={standalone}" in out
    assert standalone.exists()
    assert "ORDER BY min_x ASC NULLS LAST LIMIT 20" in standalone.read_text(encoding="utf-8")
    assert triage["verdict"] == "candidate_implementation_bug"
