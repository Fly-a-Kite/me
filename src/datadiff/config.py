from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

OracleMode = Literal["differential", "metamorphic", "both"]
GeneratorProfile = Literal[
    "common",
    "edge_float",
    "workflow",
    "bughunt",
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
    "pyarrow_dataset_isin_all_match_semantics",
    "pyarrow_large_string_partition_schema_semantics",
    "polars_rolling_mean_by_null_count_semantics",
]
GuidanceStrategy = Literal["random", "guided"]
LogLevel = Literal["full", "compact", "minimal"]


@dataclass(slots=True)
class ExperimentConfig:
    enable_type_aware_generation: bool = True
    enable_normalizer: bool = True
    enable_differential_oracle: bool = True
    enable_metamorphic_oracle: bool = False
    enable_feedback: bool = True
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
    metamorphic_variant_limit: int = 4
    log_level: LogLevel = "compact"

    def to_dict(self) -> dict:
        return asdict(self)
