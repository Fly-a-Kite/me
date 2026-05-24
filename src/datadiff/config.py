from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

OracleMode = Literal["differential", "metamorphic", "both"]
GeneratorProfile = Literal[
    "common",
    "edge_float",
    "workflow",
    "bughunt",
    "bughunt_fresh",
    "bughunt_no_groupby",
    "null_groupby_topk",
    "null_agg_topk",
    "filter_null_agg_topk",
    "join_null_agg_topk",
    "join_null_key_topk",
    "wide_offset_topk",
    "empty_filter_groupby",
    "join_filter_groupby",
    "join_null_truth_filter",
    "join_groupby_stress",
    "storage_offset",
    "float_group_key",
    "join_null_sort",
    "ordered_groupby_sort",
    "topk_resort",
    "join_ordered_agg_topk",
    "global_null_aggregate",
    "string_count_groupby",
    "unique_count_groupby",
    "set_membership_filter",
    "pyarrow_groupby_filter_cast_membership",
    "null_predicate_filter",
    "boolean_predicate_filter",
    "post_topk_range_filter",
    "tuple_absence_filter",
    "row_value_absence_filter",
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
    "polars_reverse_division_columns",
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
    "pyarrow_dataset_isin_all_match_semantics",
    "pyarrow_large_string_partition_schema_semantics",
    "pyarrow_hash_pivot_wider_order_semantics",
    "polars_rolling_mean_by_null_count_semantics",
]
GuidanceStrategy = Literal["random", "guided"]
LogLevel = Literal["full", "compact", "minimal"]

DEFAULT_KNOWN_SATURATED_BUG_FAMILIES = [
    "groupby_aggregation@datafusion",
    "grouped_topk_null_sort_key@datafusion",
    "joined_order_offset_projection@datafusion",
    "negative_zero_comparison@datafusion",
    "ordered_topk_projection@datafusion",
    "outer_join_truth_filter@datafusion",
    "topk_filter_pushdown@datafusion",
    "reverse_division_operand_order@polars",
    "tuple_absence_null_filter@duckdb",
]

DEFAULT_REPLAY_BUG_SOURCE_ISSUES = [
    "https://github.com/apache/datafusion/issues/22190",
    "https://github.com/apache/datafusion/issues/22489",
    "https://github.com/apache/datafusion/issues/22441",
    "https://github.com/apache/datafusion/issues/12956",
    "https://github.com/apache/datafusion/issues/12955",
    "https://github.com/duckdb/duckdb/issues/3015",
    "https://github.com/duckdb/duckdb/issues/11261",
    "https://github.com/duckdb/duckdb/issues/22075",
    "https://github.com/duckdb/duckdb/issues/22656",
    "https://github.com/apache/arrow/issues/42231",
]


@dataclass(slots=True)
class ExperimentConfig:
    enable_type_aware_generation: bool = True
    enable_normalizer: bool = True
    enable_differential_oracle: bool = True
    enable_metamorphic_oracle: bool = False
    enable_feedback: bool = True
    enable_replay_bug: bool = False
    enable_reducer: bool = False
    enable_artifact: bool = True
    enable_preflight_validation: bool = True
    enable_preflight_repair: bool = True
    persist_feedback_corpus: bool = False
    feedback_persist_limit: int = 4096
    enable_local_source_scheduler: bool = False
    local_source_exploration_weight: float = 0.5
    compress_run_log: bool = True
    artifact_limit: int | None = None
    oracle_mode: OracleMode = "differential"
    generator_profile: GeneratorProfile = "common"
    guidance_strategy: GuidanceStrategy = "random"
    guidance_candidate_pool: int = 1
    guidance_targets: list[str] = field(default_factory=list)
    enable_family_saturation: bool = True
    family_saturation_threshold: int = 8
    family_saturation_penalty: float = 1.25
    saturated_family_reward: float = 0.02
    known_saturated_bug_families: list[str] = field(default_factory=list)
    replay_bug_source_issues: list[str] = field(
        default_factory=lambda: list(DEFAULT_REPLAY_BUG_SOURCE_ISSUES)
    )
    issue_replay_saturation_threshold: int = 1
    issue_replay_saturation_penalty: float = 1.0
    issue_replay_global_saturation_threshold: int = 4
    issue_replay_global_saturation_penalty: float = 1.5
    issue_inspired_source_saturation_threshold: int = 3
    issue_inspired_source_saturation_penalty: float = 1.25
    metamorphic_variant_limit: int = 4
    log_level: LogLevel = "compact"

    def to_dict(self) -> dict:
        return asdict(self)
