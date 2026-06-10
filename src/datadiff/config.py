from __future__ import annotations

from collections.abc import Iterable
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
from typing import Any
from typing import Literal

from datadiff.exploration_objectives import (
    ExplorationObjectiveRule,
    merge_exploration_objective_rules,
)

OracleMode = Literal["differential", "metamorphic", "both"]
GeneratorProfile = Literal[
    "common",
    "edge_float",
    "typed_grammar",
    "workflow",
    "discovery",
    "discovery_fresh",
    "discovery_no_groupby",
    "common_api_workflow",
    "issue_focus",
    "deep_probe_rotation",
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
    "bool_null_groupby_agg",
    "large_int_filter_groupby",
    "set_membership_filter",
    "pyarrow_groupby_filter_cast_membership",
    "null_predicate_filter",
    "boolean_predicate_filter",
    "post_topk_range_filter",
    "tuple_absence_filter",
    "row_value_absence_filter",
    "running_sum_precision",
    "partitioned_running_sum",
    "path_basename_keyed_pick",
    "sortedness_null_placement",
    "simple_case_random_subject",
    "group_quantile_key_probe",
    "scalar_subquery_double_parentheses",
    "window_avg_rows_frame",
    "struct_distinct_unnest",
    "bit_compare_unequal_length",
    "round_even_float_scale",
    "duckdb_float_literal_precision",
    "polars_timestamp_precision_filter",
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
    "pandas_bool_reduction_skipna_semantics",
    "pandas_arrow_bool_groupby_reduction_semantics",
    "pyarrow_dataset_isin_all_match_semantics",
    "pyarrow_run_end_null_compute_semantics",
    "pyarrow_large_string_partition_schema_semantics",
    "pyarrow_hash_pivot_wider_order_semantics",
    "pyarrow_list_flatten_parent_indices_semantics",
    "polars_rolling_mean_by_null_count_semantics",
    "csv_long_numeric_roundtrip",
]
GuidanceStrategy = Literal["random", "guided"]
LogLevel = Literal["full", "compact", "minimal"]


@dataclass(slots=True)
class DiscoveryBias:
    targets: list[str] = field(default_factory=list)
    feature_prefixes: list[str] = field(default_factory=list)
    score_bonus: float = 0.0
    novelty_bonus: float = 0.0
    contribution_bonus: float = 0.0
    candidate_pool_bonus: float = 0.0
    keep_in_pool: bool = False

    def __post_init__(self) -> None:
        self.targets = _normalize_string_list(self.targets)
        self.feature_prefixes = _normalize_string_list(self.feature_prefixes)
        self.score_bonus = float(self.score_bonus or 0.0)
        self.novelty_bonus = float(self.novelty_bonus or 0.0)
        self.contribution_bonus = float(self.contribution_bonus or 0.0)
        self.candidate_pool_bonus = float(self.candidate_pool_bonus or 0.0)
        self.keep_in_pool = bool(self.keep_in_pool)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def key(self) -> tuple[Any, ...]:
        return (
            tuple(self.targets),
            tuple(self.feature_prefixes),
            self.score_bonus,
            self.novelty_bonus,
            self.contribution_bonus,
            self.candidate_pool_bonus,
            self.keep_in_pool,
        )

    def is_noop(self) -> bool:
        return (
            not self.targets
            and not self.feature_prefixes
            and self.score_bonus == 0.0
            and self.novelty_bonus == 0.0
            and self.contribution_bonus == 0.0
            and self.candidate_pool_bonus == 0.0
            and not self.keep_in_pool
        )


def _normalize_string_list(values: Iterable[Any] | None) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values or []:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        normalized.append(text)
    return normalized


def coerce_discovery_bias(value: DiscoveryBias | Mapping[str, Any]) -> DiscoveryBias:
    if isinstance(value, DiscoveryBias):
        return DiscoveryBias(**value.to_dict())
    if isinstance(value, Mapping):
        return DiscoveryBias(**dict(value))
    raise TypeError(f"unsupported discovery bias payload: {type(value)!r}")


def merge_discovery_biases(
    *groups: Iterable[DiscoveryBias | Mapping[str, Any] | None] | None,
) -> list[DiscoveryBias]:
    merged: list[DiscoveryBias] = []
    seen: set[tuple[Any, ...]] = set()
    for group in groups:
        if group is None:
            continue
        for item in group:
            if item is None:
                continue
            bias = coerce_discovery_bias(item)
            if bias.is_noop():
                continue
            key = bias.key()
            if key in seen:
                continue
            seen.add(key)
            merged.append(bias)
    return merged

DEFAULT_KNOWN_SATURATED_BUG_FAMILIES = [
    "csv_long_numeric_roundtrip@duckdb",
    "csv_long_numeric_roundtrip@pyarrow",
    "distinct_null_topk@datafusion",
    "groupby_aggregation@datafusion",
    "grouped_topk_null_sort_key@datafusion",
    "datafusion_limit_idempotence@datafusion",
    "metamorphic_limit_idempotence@datafusion",
    "joined_order_offset_projection@datafusion",
    "negative_zero_comparison@datafusion",
    "ordered_topk_projection@datafusion",
    "outer_join_truth_filter@datafusion",
    "pandas_arrow_timestamp_index_attr_semantics@pandas",
    "path_projection_keyed_pick@duckdb",
    "polars_reflected_arithmetic_operand_order@polars",
    "polars_vector_division_rounding@polars",
    "pyarrow_dataset_isin_all_match_semantics@pyarrow",
    "pyarrow_sliced_bool_groupby_any_all@pyarrow",
    "topk_filter_pushdown@datafusion",
    "reverse_division_operand_order@polars",
    "tuple_absence_null_filter@duckdb",
]

DEFAULT_REPLAY_BUG_SOURCE_ISSUES = [
    "https://github.com/apache/arrow/issues/32171",
    "https://github.com/apache/arrow/issues/36149",
    "https://github.com/apache/arrow/issues/42231",
    "https://github.com/apache/arrow/issues/46183",
    "https://github.com/apache/arrow/issues/47177",
    "https://github.com/apache/arrow/issues/48679",
    "https://github.com/apache/arrow/issues/49889",
    "https://github.com/apache/datafusion/issues/12955",
    "https://github.com/apache/datafusion/issues/12956",
    "https://github.com/apache/datafusion/issues/22190",
    "https://github.com/apache/datafusion/issues/22441",
    "https://github.com/apache/datafusion/issues/22489",
    "https://github.com/apache/datafusion/issues/22554",
    "https://github.com/duckdb/duckdb/issues/3015",
    "https://github.com/duckdb/duckdb/issues/4978",
    "https://github.com/duckdb/duckdb/issues/11261",
    "https://github.com/duckdb/duckdb/issues/17278",
    "https://github.com/duckdb/duckdb/issues/19491",
    "https://github.com/duckdb/duckdb/issues/19851",
    "https://github.com/duckdb/duckdb/issues/20366",
    "https://github.com/duckdb/duckdb/issues/22075",
    "https://github.com/duckdb/duckdb/issues/22418",
    "https://github.com/duckdb/duckdb/issues/22527",
    "https://github.com/duckdb/duckdb/issues/22576",
    "https://github.com/duckdb/duckdb/issues/22656",
    "https://github.com/duckdb/duckdb/issues/22676",
    "https://github.com/duckdb/duckdb/issues/22750",
    "https://github.com/duckdb/duckdb/issues/22837",
    "https://github.com/duckdb/duckdb/issues/22849",
    "https://github.com/pandas-dev/pandas/issues/43767",
    "https://github.com/pandas-dev/pandas/issues/45284",
    "https://github.com/pandas-dev/pandas/issues/59609",
    "https://github.com/pandas-dev/pandas/issues/62766",
    "https://github.com/pandas-dev/pandas/issues/63458",
    "https://github.com/pandas-dev/pandas/issues/63526",
    "https://github.com/pandas-dev/pandas/issues/63527",
    "https://github.com/pandas-dev/pandas/issues/65664",
    "https://github.com/pandas-dev/pandas/issues/65710",
    "https://github.com/pola-rs/polars/issues/8516",
    "https://github.com/pola-rs/polars/issues/17760",
    "https://github.com/pola-rs/polars/issues/18546",
    "https://github.com/pola-rs/polars/issues/22149",
    "https://github.com/pola-rs/polars/issues/23870",
    "https://github.com/pola-rs/polars/issues/25888",
    "https://github.com/pola-rs/polars/issues/26065",
    "https://github.com/pola-rs/polars/issues/26671",
    "https://github.com/pola-rs/polars/issues/26800",
    "https://github.com/pola-rs/polars/issues/26803",
    "https://github.com/pola-rs/polars/issues/26993",
    "https://github.com/pola-rs/polars/issues/27661",
    "https://github.com/pola-rs/polars/issues/27662",
    "https://github.com/pola-rs/polars/issues/27726",
]


@dataclass(frozen=True, slots=True)
class OracleConfig:
    mode: OracleMode = "differential"
    enable_differential: bool = True
    enable_metamorphic: bool = False
    metamorphic_variant_limit: int = 4
    metamorphic_relation_order: tuple[str, ...] = ()
    candidate_recheck_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "enable_differential": self.enable_differential,
            "enable_metamorphic": self.enable_metamorphic,
            "metamorphic_variant_limit": self.metamorphic_variant_limit,
            "metamorphic_relation_order": list(self.metamorphic_relation_order),
            "candidate_recheck_count": self.candidate_recheck_count,
        }


@dataclass(frozen=True, slots=True)
class FeedbackConfig:
    enabled: bool = True
    persist_corpus: bool = False
    persist_limit: int = 4096
    max_cases_per_profile: int = 6
    enable_local_source_scheduler: bool = False
    local_source_exploration_weight: float = 0.5
    enable_mutation_operator_learning: bool = True
    enable_operator_swarm: bool = True
    enable_ir_rewrite_mutations: bool = True
    enable_divergence_conditioned_mutations: bool = True
    enable_shrink_mutations: bool = True
    enable_value_catalog: bool = True
    enable_quality_archive: bool = True
    enable_hierarchical_archive: bool = True
    enable_bd_axis_bandit: bool = True
    enable_bayesian_exploration: bool = True
    enable_seed_quota: bool = True
    enable_seed_energy_batch: bool = True
    enable_seed_energy_tier_bandit: bool = True
    enable_per_operator_energy: bool = True
    enable_lineage_rarity: bool = True
    enable_minhash_dedup: bool = True
    enable_disagreement_bd_axis: bool = True
    enable_champion_corpus: bool = True
    enable_champion_graft_donor_bandit: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class GuidanceConfig:
    strategy: GuidanceStrategy = "random"
    candidate_pool: int = 1
    targets: tuple[str, ...] = ()
    semantic_focus_families: tuple[str, ...] = ()
    semantic_focus_signals: tuple[str, ...] = ()
    discovery_biases: tuple[DiscoveryBias, ...] = ()
    enable_family_saturation: bool = True
    family_saturation_threshold: int = 8
    family_saturation_penalty: float = 1.25
    saturated_family_reward: float = 0.02
    known_saturated_bug_families: tuple[str, ...] = ()
    replay_bug_source_issues: tuple[str, ...] = ()
    issue_replay_saturation_threshold: int = 1
    issue_replay_saturation_penalty: float = 1.0
    issue_replay_global_saturation_threshold: int = 4
    issue_replay_global_saturation_penalty: float = 1.5
    issue_inspired_source_saturation_threshold: int = 3
    issue_inspired_source_saturation_penalty: float = 1.25

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "candidate_pool": self.candidate_pool,
            "targets": list(self.targets),
            "semantic_focus_families": list(self.semantic_focus_families),
            "semantic_focus_signals": list(self.semantic_focus_signals),
            "discovery_biases": [bias.to_dict() for bias in self.discovery_biases],
            "enable_family_saturation": self.enable_family_saturation,
            "family_saturation_threshold": self.family_saturation_threshold,
            "family_saturation_penalty": self.family_saturation_penalty,
            "saturated_family_reward": self.saturated_family_reward,
            "known_saturated_bug_families": list(self.known_saturated_bug_families),
            "replay_bug_source_issues": list(self.replay_bug_source_issues),
            "issue_replay_saturation_threshold": self.issue_replay_saturation_threshold,
            "issue_replay_saturation_penalty": self.issue_replay_saturation_penalty,
            "issue_replay_global_saturation_threshold": self.issue_replay_global_saturation_threshold,
            "issue_replay_global_saturation_penalty": self.issue_replay_global_saturation_penalty,
            "issue_inspired_source_saturation_threshold": self.issue_inspired_source_saturation_threshold,
            "issue_inspired_source_saturation_penalty": self.issue_inspired_source_saturation_penalty,
        }


@dataclass(frozen=True, slots=True)
class LearningConfig:
    generator_profile_pool: tuple[str, ...] = ()
    version_pair_pool: tuple[str, ...] = ()
    generator_profile_learning_weight: float = 0.0
    semantic_objective_learning_weight: float = 0.0
    metamorphic_relation_learning_weight: float = 0.0
    version_pair_learning_weight: float = 0.0
    backend_pair_learning_weight: float = 0.0
    backend_pair_priority_limit: int = 3
    enable_generator_profile_learning: bool = True
    enable_profile_capability_filter: bool = True
    enable_semantic_objective_learning: bool = True
    enable_metamorphic_relation_learning: bool = True
    enable_backend_pair_learning: bool = True
    enable_lhs_seeding: bool = True
    target_version: str = ""
    fixed_version: str = ""
    strategy_snapshot_path: str = ""
    strategy_learning_path: str = ""
    freeze_strategy_snapshot: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "generator_profile_pool": list(self.generator_profile_pool),
            "version_pair_pool": list(self.version_pair_pool),
            "generator_profile_learning_weight": self.generator_profile_learning_weight,
            "semantic_objective_learning_weight": self.semantic_objective_learning_weight,
            "metamorphic_relation_learning_weight": self.metamorphic_relation_learning_weight,
            "version_pair_learning_weight": self.version_pair_learning_weight,
            "backend_pair_learning_weight": self.backend_pair_learning_weight,
            "backend_pair_priority_limit": self.backend_pair_priority_limit,
            "enable_generator_profile_learning": self.enable_generator_profile_learning,
            "enable_profile_capability_filter": self.enable_profile_capability_filter,
            "enable_semantic_objective_learning": self.enable_semantic_objective_learning,
            "enable_metamorphic_relation_learning": self.enable_metamorphic_relation_learning,
            "enable_backend_pair_learning": self.enable_backend_pair_learning,
            "enable_lhs_seeding": self.enable_lhs_seeding,
            "target_version": self.target_version,
            "fixed_version": self.fixed_version,
            "strategy_snapshot_path": self.strategy_snapshot_path,
            "strategy_learning_path": self.strategy_learning_path,
            "freeze_strategy_snapshot": self.freeze_strategy_snapshot,
        }


@dataclass(frozen=True, slots=True)
class LoggingConfig:
    log_level: LogLevel = "compact"
    compress_run_log: bool = True
    enable_artifact: bool = True
    artifact_limit: int | None = None
    enable_reducer: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ExecutionConfig:
    enable_parallel_backend_execution: bool = True
    enable_preflight_validation: bool = True
    enable_preflight_repair: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
    enable_parallel_backend_execution: bool = True
    enable_preflight_validation: bool = True
    enable_preflight_repair: bool = True
    persist_feedback_corpus: bool = False
    feedback_persist_limit: int = 4096
    feedback_max_cases_per_profile: int = 6
    enable_local_source_scheduler: bool = False
    local_source_exploration_weight: float = 0.5
    compress_run_log: bool = True
    artifact_limit: int | None = None
    oracle_mode: OracleMode = "differential"
    generator_profile: GeneratorProfile = "common"
    generator_profile_pool: list[str] = field(default_factory=list)
    version_pair_pool: list[str] = field(default_factory=list)
    generator_profile_learning_weight: float = 0.0
    semantic_objective_learning_weight: float = 0.0
    metamorphic_relation_learning_weight: float = 0.0
    version_pair_learning_weight: float = 0.0
    backend_pair_learning_weight: float = 0.0
    backend_pair_priority_limit: int = 3
    enable_generator_profile_learning: bool = True
    enable_profile_capability_filter: bool = True
    enable_semantic_objective_learning: bool = True
    enable_metamorphic_relation_learning: bool = True
    enable_backend_pair_learning: bool = True
    enable_mutation_operator_learning: bool = True
    enable_operator_swarm: bool = True
    enable_ir_rewrite_mutations: bool = True
    enable_divergence_conditioned_mutations: bool = True
    enable_shrink_mutations: bool = True
    enable_value_catalog: bool = True
    enable_quality_archive: bool = True
    enable_hierarchical_archive: bool = True
    enable_bd_axis_bandit: bool = True
    enable_bayesian_exploration: bool = True
    enable_seed_quota: bool = True
    enable_seed_energy_batch: bool = True
    enable_seed_energy_tier_bandit: bool = True
    enable_per_operator_energy: bool = True
    enable_lineage_rarity: bool = True
    enable_minhash_dedup: bool = True
    enable_disagreement_bd_axis: bool = True
    enable_lhs_seeding: bool = True
    enable_champion_corpus: bool = True
    enable_champion_graft_donor_bandit: bool = True
    guidance_strategy: GuidanceStrategy = "random"
    guidance_candidate_pool: int = 1
    guidance_targets: list[str] = field(default_factory=list)
    semantic_focus_families: list[str] = field(default_factory=list)
    semantic_focus_signals: list[str] = field(default_factory=list)
    exploration_objective_rules: list[ExplorationObjectiveRule] = field(default_factory=list)
    discovery_biases: list[DiscoveryBias] = field(default_factory=list)
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
    candidate_recheck_count: int = 0
    metamorphic_variant_limit: int = 4
    metamorphic_relation_order: list[str] = field(default_factory=list)
    target_version: str = ""
    fixed_version: str = ""
    log_level: LogLevel = "compact"
    strategy_snapshot_path: str = ""
    strategy_learning_path: str = ""
    freeze_strategy_snapshot: bool = False

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any] | None) -> "ExperimentConfig":
        if not isinstance(payload, Mapping) or not payload:
            return cls()
        allowed = {item.name for item in fields(cls)}
        return cls(**{str(key): value for key, value in payload.items() if str(key) in allowed})

    def __post_init__(self) -> None:
        self.guidance_targets = _normalize_string_list(self.guidance_targets)
        self.semantic_focus_families = _normalize_string_list(self.semantic_focus_families)
        self.semantic_focus_signals = _normalize_string_list(self.semantic_focus_signals)
        self.generator_profile_pool = _normalize_string_list(self.generator_profile_pool)
        self.version_pair_pool = _normalize_string_list(self.version_pair_pool)
        self.generator_profile_learning_weight = max(0.0, float(self.generator_profile_learning_weight or 0.0))
        self.semantic_objective_learning_weight = max(0.0, float(self.semantic_objective_learning_weight or 0.0))
        self.metamorphic_relation_learning_weight = max(0.0, float(self.metamorphic_relation_learning_weight or 0.0))
        self.version_pair_learning_weight = max(0.0, float(self.version_pair_learning_weight or 0.0))
        self.backend_pair_learning_weight = max(0.0, float(self.backend_pair_learning_weight or 0.0))
        self.backend_pair_priority_limit = max(1, int(self.backend_pair_priority_limit or 3))
        self.enable_generator_profile_learning = bool(self.enable_generator_profile_learning)
        self.enable_profile_capability_filter = bool(self.enable_profile_capability_filter)
        self.enable_semantic_objective_learning = bool(self.enable_semantic_objective_learning)
        self.enable_metamorphic_relation_learning = bool(self.enable_metamorphic_relation_learning)
        self.enable_backend_pair_learning = bool(self.enable_backend_pair_learning)
        self.enable_parallel_backend_execution = bool(self.enable_parallel_backend_execution)
        self.enable_mutation_operator_learning = bool(self.enable_mutation_operator_learning)
        self.enable_operator_swarm = bool(self.enable_operator_swarm)
        self.enable_ir_rewrite_mutations = bool(self.enable_ir_rewrite_mutations)
        self.enable_divergence_conditioned_mutations = bool(self.enable_divergence_conditioned_mutations)
        self.enable_shrink_mutations = bool(self.enable_shrink_mutations)
        self.enable_value_catalog = bool(self.enable_value_catalog)
        self.enable_quality_archive = bool(self.enable_quality_archive)
        self.enable_hierarchical_archive = bool(self.enable_hierarchical_archive)
        self.enable_bd_axis_bandit = bool(self.enable_bd_axis_bandit)
        self.enable_bayesian_exploration = bool(self.enable_bayesian_exploration)
        self.enable_seed_quota = bool(self.enable_seed_quota)
        self.enable_seed_energy_batch = bool(self.enable_seed_energy_batch)
        self.enable_seed_energy_tier_bandit = bool(self.enable_seed_energy_tier_bandit)
        self.enable_per_operator_energy = bool(self.enable_per_operator_energy)
        self.enable_lineage_rarity = bool(self.enable_lineage_rarity)
        self.enable_minhash_dedup = bool(self.enable_minhash_dedup)
        self.enable_disagreement_bd_axis = bool(self.enable_disagreement_bd_axis)
        self.enable_lhs_seeding = bool(self.enable_lhs_seeding)
        self.enable_champion_corpus = bool(self.enable_champion_corpus)
        self.enable_champion_graft_donor_bandit = bool(self.enable_champion_graft_donor_bandit)
        self.metamorphic_relation_order = _normalize_string_list(self.metamorphic_relation_order)
        self.target_version = str(self.target_version or "").strip()
        self.fixed_version = str(self.fixed_version or "").strip()
        self.exploration_objective_rules = merge_exploration_objective_rules(
            self.exploration_objective_rules
        )
        self.discovery_biases = merge_discovery_biases(self.discovery_biases)
        self.known_saturated_bug_families = _normalize_string_list(self.known_saturated_bug_families)
        self.replay_bug_source_issues = _normalize_string_list(self.replay_bug_source_issues)
        self.strategy_snapshot_path = str(self.strategy_snapshot_path or "").strip()
        self.strategy_learning_path = str(self.strategy_learning_path or "").strip()
        self.freeze_strategy_snapshot = bool(self.freeze_strategy_snapshot)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["discovery_biases"] = [bias.to_dict() for bias in self.discovery_biases]
        payload["exploration_objective_rules"] = [
            rule.to_dict() for rule in self.exploration_objective_rules
        ]
        return payload

    @property
    def oracle(self) -> OracleConfig:
        return OracleConfig(
            mode=self.oracle_mode,
            enable_differential=self.enable_differential_oracle,
            enable_metamorphic=self.enable_metamorphic_oracle,
            metamorphic_variant_limit=self.metamorphic_variant_limit,
            metamorphic_relation_order=tuple(self.metamorphic_relation_order),
            candidate_recheck_count=self.candidate_recheck_count,
        )

    @property
    def feedback(self) -> FeedbackConfig:
        return FeedbackConfig(
            enabled=self.enable_feedback,
            persist_corpus=self.persist_feedback_corpus,
            persist_limit=self.feedback_persist_limit,
            max_cases_per_profile=self.feedback_max_cases_per_profile,
            enable_local_source_scheduler=self.enable_local_source_scheduler,
            local_source_exploration_weight=self.local_source_exploration_weight,
            enable_mutation_operator_learning=self.enable_mutation_operator_learning,
            enable_operator_swarm=self.enable_operator_swarm,
            enable_ir_rewrite_mutations=self.enable_ir_rewrite_mutations,
            enable_divergence_conditioned_mutations=self.enable_divergence_conditioned_mutations,
            enable_shrink_mutations=self.enable_shrink_mutations,
            enable_value_catalog=self.enable_value_catalog,
            enable_quality_archive=self.enable_quality_archive,
            enable_hierarchical_archive=self.enable_hierarchical_archive,
            enable_bd_axis_bandit=self.enable_bd_axis_bandit,
            enable_bayesian_exploration=self.enable_bayesian_exploration,
            enable_seed_quota=self.enable_seed_quota,
            enable_seed_energy_batch=self.enable_seed_energy_batch,
            enable_seed_energy_tier_bandit=self.enable_seed_energy_tier_bandit,
            enable_per_operator_energy=self.enable_per_operator_energy,
            enable_lineage_rarity=self.enable_lineage_rarity,
            enable_minhash_dedup=self.enable_minhash_dedup,
            enable_disagreement_bd_axis=self.enable_disagreement_bd_axis,
            enable_champion_corpus=self.enable_champion_corpus,
            enable_champion_graft_donor_bandit=self.enable_champion_graft_donor_bandit,
        )

    @property
    def guidance(self) -> GuidanceConfig:
        return GuidanceConfig(
            strategy=self.guidance_strategy,
            candidate_pool=self.guidance_candidate_pool,
            targets=tuple(self.guidance_targets),
            semantic_focus_families=tuple(self.semantic_focus_families),
            semantic_focus_signals=tuple(self.semantic_focus_signals),
            discovery_biases=tuple(self.discovery_biases),
            enable_family_saturation=self.enable_family_saturation,
            family_saturation_threshold=self.family_saturation_threshold,
            family_saturation_penalty=self.family_saturation_penalty,
            saturated_family_reward=self.saturated_family_reward,
            known_saturated_bug_families=tuple(self.known_saturated_bug_families),
            replay_bug_source_issues=tuple(self.replay_bug_source_issues),
            issue_replay_saturation_threshold=self.issue_replay_saturation_threshold,
            issue_replay_saturation_penalty=self.issue_replay_saturation_penalty,
            issue_replay_global_saturation_threshold=self.issue_replay_global_saturation_threshold,
            issue_replay_global_saturation_penalty=self.issue_replay_global_saturation_penalty,
            issue_inspired_source_saturation_threshold=self.issue_inspired_source_saturation_threshold,
            issue_inspired_source_saturation_penalty=self.issue_inspired_source_saturation_penalty,
        )

    @property
    def learning(self) -> LearningConfig:
        return LearningConfig(
            generator_profile_pool=tuple(self.generator_profile_pool),
            version_pair_pool=tuple(self.version_pair_pool),
            generator_profile_learning_weight=self.generator_profile_learning_weight,
            semantic_objective_learning_weight=self.semantic_objective_learning_weight,
            metamorphic_relation_learning_weight=self.metamorphic_relation_learning_weight,
            version_pair_learning_weight=self.version_pair_learning_weight,
            backend_pair_learning_weight=self.backend_pair_learning_weight,
            backend_pair_priority_limit=self.backend_pair_priority_limit,
            enable_generator_profile_learning=self.enable_generator_profile_learning,
            enable_profile_capability_filter=self.enable_profile_capability_filter,
            enable_semantic_objective_learning=self.enable_semantic_objective_learning,
            enable_metamorphic_relation_learning=self.enable_metamorphic_relation_learning,
            enable_backend_pair_learning=self.enable_backend_pair_learning,
            enable_lhs_seeding=self.enable_lhs_seeding,
            target_version=self.target_version,
            fixed_version=self.fixed_version,
            strategy_snapshot_path=self.strategy_snapshot_path,
            strategy_learning_path=self.strategy_learning_path,
            freeze_strategy_snapshot=self.freeze_strategy_snapshot,
        )

    @property
    def logging(self) -> LoggingConfig:
        return LoggingConfig(
            log_level=self.log_level,
            compress_run_log=self.compress_run_log,
            enable_artifact=self.enable_artifact,
            artifact_limit=self.artifact_limit,
            enable_reducer=self.enable_reducer,
        )

    @property
    def execution(self) -> ExecutionConfig:
        return ExecutionConfig(
            enable_parallel_backend_execution=self.enable_parallel_backend_execution,
            enable_preflight_validation=self.enable_preflight_validation,
            enable_preflight_repair=self.enable_preflight_repair,
        )

    def to_nested_dict(self) -> dict[str, Any]:
        return {
            "generation": {
                "generator_profile": self.generator_profile,
                "enable_type_aware_generation": self.enable_type_aware_generation,
            },
            "oracle": self.oracle.to_dict(),
            "feedback": self.feedback.to_dict(),
            "guidance": self.guidance.to_dict(),
            "learning": self.learning.to_dict(),
            "logging": self.logging.to_dict(),
            "execution": self.execution.to_dict(),
            "exploration_objective_rules": [
                rule.to_dict() for rule in self.exploration_objective_rules
            ],
            "flat": self.to_dict(),
        }
