from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any
from collections.abc import Mapping
from collections.abc import Sequence

from datadiff.config import (
    DEFAULT_KNOWN_SATURATED_BUG_FAMILIES,
    DEFAULT_REPLAY_BUG_SOURCE_ISSUES,
    DiscoveryBias,
    ExperimentConfig,
    merge_discovery_biases,
)
from datadiff.live_target_catalog import DEFAULT_LIVE_TARGETS_BY_NAME
from datadiff.strategy_registry import discovery_biases_for_lanes
from datadiff.strategy_registry import discovery_lane_spec


@dataclass(frozen=True, slots=True)
class LivePresetSpec:
    target_key: str
    generator_profile: str = "discovery"
    candidate_pool: int = 12
    local_source_exploration_weight: float = 0.40
    metamorphic_variant_limit: int = 4
    known_saturated_bug_families: tuple[str, ...] = ()
    replay_bug_source_issues: tuple[str, ...] = ()
    enable_replay_bug: bool = False
    family_saturation_threshold: int = 4
    family_saturation_penalty: float = 6.0
    saturated_family_reward: float = 0.0
    candidate_recheck_count: int = 2
    issue_replay_global_saturation_threshold: int = 2
    issue_replay_global_saturation_penalty: float = 2.0
    discovery_lanes: tuple[str, ...] = ()
    semantic_focus_families: tuple[str, ...] = ()
    semantic_focus_signals: tuple[str, ...] = ()
    discovery_biases: tuple[DiscoveryBias | Mapping[str, Any], ...] = ()


@dataclass(frozen=True, slots=True)
class PresetSpec:
    base_preset: str = ""
    overlays: tuple[str, ...] = ()
    config: dict[str, Any] = field(default_factory=dict)
    live: LivePresetSpec | None = None


@dataclass(frozen=True, slots=True)
class ConfigOverlay:
    overlay_id: str
    updates: dict[str, Any] = field(default_factory=dict)
    notes: str = ""


def _config_spec(*, base_preset: str = "", **config: Any) -> PresetSpec:
    return PresetSpec(base_preset=base_preset, config=dict(config))


def _live_preset(target_key: str, **kwargs: Any) -> PresetSpec:
    return PresetSpec(live=LivePresetSpec(target_key=target_key, **kwargs))


def _overlay_preset(*, base_preset: str, overlays: Sequence[str]) -> PresetSpec:
    return PresetSpec(base_preset=base_preset, overlays=tuple(str(name).strip() for name in overlays if str(name).strip()))


def _metamorphic_overlay(*, base_preset: str, variant_limit: int, candidate_pool: int | None = None) -> PresetSpec:
    config: dict[str, Any] = {
        "enable_metamorphic_oracle": True,
        "oracle_mode": "both",
        "metamorphic_variant_limit": variant_limit,
    }
    if candidate_pool is not None:
        config["guidance_candidate_pool"] = candidate_pool
    return _config_spec(base_preset=base_preset, **config)


def _guided_profile(
    *,
    generator_profile: str = "common",
    guidance_targets: Sequence[str],
    guidance_candidate_pool: int = 8,
    candidate_recheck_count: int | None = None,
    metamorphic_variant_limit: int | None = None,
) -> PresetSpec:
    config: dict[str, Any] = dict(
        generator_profile=generator_profile,
        guidance_strategy="guided",
        guidance_candidate_pool=guidance_candidate_pool,
        guidance_targets=list(guidance_targets),
    )
    if candidate_recheck_count is not None:
        config["candidate_recheck_count"] = int(candidate_recheck_count)
    if metamorphic_variant_limit is not None:
        config["metamorphic_variant_limit"] = int(metamorphic_variant_limit)
    return _config_spec(**config)


def _guided_profile_from_name(
    *,
    name: str,
    guidance_targets: Sequence[str],
    guidance_candidate_pool: int,
    metamorphic_variant_limit: int,
    candidate_recheck_count: int | None = None,
) -> PresetSpec:
    return _guided_profile(
        generator_profile=name,
        guidance_targets=guidance_targets,
        guidance_candidate_pool=guidance_candidate_pool,
        metamorphic_variant_limit=metamorphic_variant_limit,
        candidate_recheck_count=candidate_recheck_count,
    )


def _profile_metamorphic_overlay(
    *,
    base_preset: str,
    guidance_candidate_pool: int,
    metamorphic_variant_limit: int,
    candidate_recheck_count: int | None = None,
) -> PresetSpec:
    config: dict[str, Any] = {
        "enable_metamorphic_oracle": True,
        "oracle_mode": "both",
        "guidance_candidate_pool": guidance_candidate_pool,
        "metamorphic_variant_limit": metamorphic_variant_limit,
    }
    if candidate_recheck_count is not None:
        config["candidate_recheck_count"] = int(candidate_recheck_count)
    return _config_spec(base_preset=base_preset, **config)


CONFIG_OVERLAYS: dict[str, ConfigOverlay] = {
    "disable_type_aware_generation": ConfigOverlay(
        overlay_id="disable_type_aware_generation",
        updates={"enable_type_aware_generation": False},
        notes="Disable type-aware generation for module-ablation or comparison contrasts.",
    ),
    "disable_normalizer": ConfigOverlay(
        overlay_id="disable_normalizer",
        updates={"enable_normalizer": False},
        notes="Disable semantic/result normalization while preserving the rest of the harness.",
    ),
    "disable_feedback_corpus": ConfigOverlay(
        overlay_id="disable_feedback_corpus",
        updates={"enable_feedback": False},
        notes="Disable the feedback corpus and closed-loop mutation reuse path.",
    ),
    "enable_metamorphic_oracle": ConfigOverlay(
        overlay_id="enable_metamorphic_oracle",
        updates={"enable_metamorphic_oracle": True, "oracle_mode": "both"},
        notes="Enable metamorphic checking alongside the differential oracle.",
    ),
    "disable_differential_oracle": ConfigOverlay(
        overlay_id="disable_differential_oracle",
        updates={"enable_differential_oracle": False},
        notes="Disable differential-only checking while leaving other oracle settings intact.",
    ),
    "enable_reducer": ConfigOverlay(
        overlay_id="enable_reducer",
        updates={"enable_reducer": True},
        notes="Enable reducer-backed artifact minimization.",
    ),
    "enable_guidance": ConfigOverlay(
        overlay_id="enable_guidance",
        updates={"guidance_strategy": "guided", "guidance_candidate_pool": 8},
        notes="Turn on guided selection with the standard final-harness candidate pool.",
    ),
    "target_filter": ConfigOverlay(
        overlay_id="target_filter",
        updates={"guidance_targets": ["filter"]},
        notes="Focus guidance on filter semantics.",
    ),
    "target_groupby": ConfigOverlay(
        overlay_id="target_groupby",
        updates={
            "generator_profile": "discovery",
            "guidance_targets": [
                "groupby",
                "aggregation",
                "groupby_aggregation",
                "groupby_sorted_input",
                "multi_key_groupby",
                "multi_key_groupby_topk",
                "null_groupby_topk",
                "null_agg_topk",
                "filter_null_agg_topk",
                "join_null_agg_topk",
                "empty_filter_groupby",
                "join_filter_groupby",
                "bool_null_groupby_agg",
                "nunique_groupby",
                "groupby_having_topk",
            ],
            "discovery_biases": [
                DiscoveryBias(
                    targets=[
                        "groupby_aggregation",
                        "multi_key_groupby",
                        "groupby_sorted_input",
                        "null_agg_topk",
                        "join_filter_groupby",
                    ],
                    feature_prefixes=[
                        "op:groupby",
                        "groupby:",
                        "pattern:multi_key_groupby",
                        "pattern:null_agg_topk",
                        "pattern:join_filter_groupby",
                    ],
                    score_bonus=0.35,
                    novelty_bonus=0.20,
                    contribution_bonus=0.20,
                    candidate_pool_bonus=0.25,
                    keep_in_pool=True,
                )
            ],
            "family_saturation_threshold": 16,
        },
        notes="Focus guidance on grouped aggregation semantics.",
    ),
    "target_join": ConfigOverlay(
        overlay_id="target_join",
        updates={
            "generator_profile": "discovery_no_groupby",
            "guidance_targets": ["join", "sort_limit"],
        },
        notes="Focus guidance on join and ordered top-k semantics.",
    ),
    "target_mutate": ConfigOverlay(
        overlay_id="target_mutate",
        updates={"guidance_targets": ["mutate", "expressions"]},
        notes="Focus guidance on mutate/expression semantics.",
    ),
    "target_discovery_guided": ConfigOverlay(
        overlay_id="target_discovery_guided",
        updates={"guidance_targets": ["join", "groupby", "mutate", "filter", "expressions"]},
        notes="Use the standard discovery guided semantic target set.",
    ),
    "generator_discovery": ConfigOverlay(
        overlay_id="generator_discovery",
        updates={"generator_profile": "discovery"},
        notes="Switch the generator profile to discovery.",
    ),
    "generator_workflow": ConfigOverlay(
        overlay_id="generator_workflow",
        updates={"generator_profile": "workflow"},
        notes="Switch the generator profile to workflow.",
    ),
    "metamorphic_variant_limit_2": ConfigOverlay(
        overlay_id="metamorphic_variant_limit_2",
        updates={"metamorphic_variant_limit": 2},
    ),
    "metamorphic_variant_limit_4": ConfigOverlay(
        overlay_id="metamorphic_variant_limit_4",
        updates={"metamorphic_variant_limit": 4},
    ),
    "metamorphic_variant_limit_6": ConfigOverlay(
        overlay_id="metamorphic_variant_limit_6",
        updates={"metamorphic_variant_limit": 6},
    ),
    "metamorphic_variant_limit_8": ConfigOverlay(
        overlay_id="metamorphic_variant_limit_8",
        updates={"metamorphic_variant_limit": 8},
    ),
    "candidate_pool_4": ConfigOverlay(
        overlay_id="candidate_pool_4",
        updates={"guidance_candidate_pool": 4},
    ),
    "candidate_pool_8": ConfigOverlay(
        overlay_id="candidate_pool_8",
        updates={"guidance_candidate_pool": 8},
    ),
    "candidate_pool_10": ConfigOverlay(
        overlay_id="candidate_pool_10",
        updates={"guidance_candidate_pool": 10},
    ),
}


def _profile_base_spec(
    *,
    name: str,
    spec: Mapping[str, Any],
) -> PresetSpec:
    generator_profile = str(spec.get("generator_profile", name))
    guidance_targets = list(spec["guidance_targets"])
    guidance_candidate_pool = int(spec["guidance_candidate_pool"])
    metamorphic_variant_limit = int(spec["metamorphic_variant_limit"])
    candidate_recheck_count = spec.get("candidate_recheck_count")
    if not bool(spec.get("guided", True)):
        return _config_spec(generator_profile=generator_profile)
    return _guided_profile(
        generator_profile=generator_profile,
        guidance_targets=guidance_targets,
        guidance_candidate_pool=guidance_candidate_pool,
        candidate_recheck_count=candidate_recheck_count,
        metamorphic_variant_limit=metamorphic_variant_limit,
    )


def _profile_overlay_spec(
    *,
    name: str,
    spec: Mapping[str, Any],
) -> PresetSpec:
    base_preset = str(spec.get("metamorphic_base_preset", name))
    guidance_candidate_pool = int(spec["guidance_candidate_pool"])
    metamorphic_variant_limit = int(
        spec.get(
            "metamorphic_overlay_variant_limit",
            max(int(spec["metamorphic_variant_limit"]) + 2, 2),
        )
    )
    candidate_recheck_count = spec.get("candidate_recheck_count")
    return _profile_metamorphic_overlay(
        base_preset=base_preset,
        guidance_candidate_pool=guidance_candidate_pool,
        metamorphic_variant_limit=metamorphic_variant_limit,
        candidate_recheck_count=candidate_recheck_count,
    )


def _semantic_focus_guidance_targets(
    *,
    base_targets: Sequence[str] | None = None,
    semantic_focus_families: Sequence[str] | None = None,
    semantic_focus_signals: Sequence[str] | None = None,
) -> list[str]:
    merged = [
        *(str(item).strip() for item in (base_targets or [])),
        *(str(item).strip() for item in (semantic_focus_families or [])),
        *(str(item).strip() for item in (semantic_focus_signals or [])),
    ]
    out: list[str] = []
    seen: set[str] = set()
    for item in merged:
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _live_semantic_focus(
    spec: LivePresetSpec,
) -> tuple[list[str], list[str]]:
    families = [str(item).strip() for item in spec.semantic_focus_families if str(item).strip()]
    signals = [str(item).strip() for item in spec.semantic_focus_signals if str(item).strip()]
    for lane_id in spec.discovery_lanes:
        lane = discovery_lane_spec(lane_id)
        for family in lane.semantic_focus_families:
            item = str(family).strip()
            if item and item not in families:
                families.append(item)
        for signal in lane.semantic_focus_signals:
            item = str(signal).strip()
            if item and item not in signals:
                signals.append(item)
    return families, signals


PROFILE_PRESET_SPECS: dict[str, dict[str, Any]] = {
    "edge_float": {
        "generator_profile": "edge_float",
        "guidance_targets": ["edge_float", "numeric", "expressions"],
        "guidance_candidate_pool": 8,
        "metamorphic_variant_limit": 0,
        "metamorphic_overlay_variant_limit": 4,
        "guided": False,
    },
    "edge_float_guided": {
        "generator_profile": "edge_float",
        "guidance_targets": ["edge_float", "numeric", "expressions"],
        "guidance_candidate_pool": 8,
        "metamorphic_variant_limit": 0,
        "register_metamorphic_overlay": False,
    },
    "null_groupby_topk": {
        "guidance_targets": ["null_groupby_topk", "groupby", "nulls", "sort_limit"],
        "guidance_candidate_pool": 4,
        "metamorphic_variant_limit": 4,
        "metamorphic_overlay_variant_limit": 8,
    },
    "null_agg_topk": {
        "guidance_targets": ["null_agg_topk", "groupby", "nulls", "aggregation", "sort_limit"],
        "guidance_candidate_pool": 4,
        "metamorphic_variant_limit": 4,
        "metamorphic_overlay_variant_limit": 8,
    },
    "filter_null_agg_topk": {
        "guidance_targets": [
            "filter_null_agg_topk",
            "filter",
            "groupby",
            "nulls",
            "aggregation",
            "sort_limit",
            "expressions",
        ],
        "guidance_candidate_pool": 4,
        "metamorphic_variant_limit": 4,
        "metamorphic_overlay_variant_limit": 8,
    },
    "join_null_agg_topk": {
        "guidance_targets": ["join_null_agg_topk", "join", "nulls", "aggregation", "sort_limit", "expressions"],
        "guidance_candidate_pool": 4,
        "metamorphic_variant_limit": 4,
        "metamorphic_overlay_variant_limit": 8,
    },
    "join_null_key_topk": {
        "guidance_targets": ["join_null_key_topk", "join", "groupby", "nulls", "sort_limit", "topk"],
        "guidance_candidate_pool": 4,
        "metamorphic_variant_limit": 4,
        "metamorphic_overlay_variant_limit": 8,
    },
    "wide_offset_topk": {
        "guidance_targets": ["wide_offset_topk", "sort_limit", "sort_offset", "offset", "topk", "nulls"],
        "guidance_candidate_pool": 2,
        "metamorphic_variant_limit": 2,
        "metamorphic_overlay_variant_limit": 4,
    },
    "empty_filter_groupby": {
        "guidance_targets": ["empty_filter_groupby", "filter", "groupby", "aggregation", "empty", "sort_limit"],
        "guidance_candidate_pool": 4,
        "metamorphic_variant_limit": 4,
        "metamorphic_overlay_variant_limit": 8,
    },
    "join_filter_groupby": {
        "guidance_targets": ["join_filter_groupby", "join", "filter", "groupby", "aggregation", "sort_limit", "expressions"],
        "guidance_candidate_pool": 4,
        "metamorphic_variant_limit": 4,
        "metamorphic_overlay_variant_limit": 8,
    },
    "join_null_truth_filter": {
        "guidance_targets": ["join_null_truth_filter", "join", "filter", "truth_filter", "nulls"],
        "guidance_candidate_pool": 4,
        "metamorphic_variant_limit": 4,
        "metamorphic_overlay_variant_limit": 8,
    },
    "join_groupby_stress": {
        "guidance_targets": ["join", "groupby", "aggregation", "global_aggregation"],
        "guidance_candidate_pool": 1,
        "metamorphic_variant_limit": 2,
        "metamorphic_overlay_variant_limit": 4,
    },
    "storage_offset": {
        "guidance_targets": ["sort_limit", "sort_offset", "offset"],
        "guidance_candidate_pool": 1,
        "metamorphic_variant_limit": 0,
        "register_metamorphic_overlay": False,
    },
    "float_group_key": {
        "guidance_targets": ["float_group_key", "join", "mutate", "groupby", "expressions"],
        "guidance_candidate_pool": 4,
        "metamorphic_variant_limit": 4,
        "metamorphic_overlay_variant_limit": 8,
    },
    "join_null_sort": {
        "guidance_targets": ["join_null_sort", "join", "nulls", "sort_limit", "expressions"],
        "guidance_candidate_pool": 4,
        "metamorphic_variant_limit": 4,
        "metamorphic_overlay_variant_limit": 8,
    },
    "ordered_groupby_sort": {
        "guidance_targets": ["ordered_groupby_sort", "groupby", "aggregation", "sort_limit"],
        "guidance_candidate_pool": 4,
        "metamorphic_variant_limit": 4,
        "metamorphic_overlay_variant_limit": 8,
    },
    "topk_resort": {
        "guidance_targets": ["topk_resort", "sort_limit", "topk", "nulls"],
        "guidance_candidate_pool": 4,
        "metamorphic_variant_limit": 4,
        "metamorphic_overlay_variant_limit": 8,
    },
    "join_ordered_agg_topk": {
        "guidance_targets": ["join_ordered_agg_topk", "join", "groupby", "aggregation", "sort_limit", "topk"],
        "guidance_candidate_pool": 4,
        "metamorphic_variant_limit": 4,
        "metamorphic_overlay_variant_limit": 8,
    },
    "global_null_aggregate": {
        "guidance_targets": ["global_null_aggregate", "global_aggregation", "aggregation", "nulls", "sort_limit"],
        "guidance_candidate_pool": 4,
        "metamorphic_variant_limit": 4,
        "metamorphic_overlay_variant_limit": 8,
    },
    "string_count_groupby": {
        "guidance_targets": ["string_count_groupby", "groupby", "strings", "nulls", "aggregation", "sort_limit"],
        "guidance_candidate_pool": 4,
        "metamorphic_variant_limit": 4,
        "metamorphic_overlay_variant_limit": 8,
    },
    "unique_count_groupby": {
        "guidance_targets": ["unique_count_groupby", "unique_count", "groupby", "strings", "nulls", "aggregation", "sort_limit"],
        "guidance_candidate_pool": 4,
        "metamorphic_variant_limit": 4,
        "metamorphic_overlay_variant_limit": 8,
    },
    "bool_null_groupby_agg": {
        "guidance_targets": [
            "bool_null_groupby_agg",
            "boolean_aggregation",
            "bool_any_all",
            "groupby",
            "nulls",
            "aggregation",
            "sort_limit",
        ],
        "guidance_candidate_pool": 4,
        "metamorphic_variant_limit": 4,
        "metamorphic_overlay_variant_limit": 8,
    },
    "large_int_filter_groupby": {
        "guidance_targets": [
            "large_int_filter_groupby",
            "large_integer",
            "filter",
            "groupby",
            "numeric",
            "aggregation",
            "sort_limit",
        ],
        "guidance_candidate_pool": 4,
        "metamorphic_variant_limit": 4,
        "metamorphic_overlay_variant_limit": 8,
    },
    "set_membership_filter": {
        "guidance_targets": ["set_membership_filter", "set_membership", "filter", "strings", "nulls", "aggregation", "sort_limit"],
        "guidance_candidate_pool": 4,
        "metamorphic_variant_limit": 4,
        "metamorphic_overlay_variant_limit": 8,
    },
    "pyarrow_groupby_filter_cast_membership": {
        "guidance_targets": [
            "pyarrow_groupby_filter_cast_membership",
            "set_membership",
            "filter",
            "groupby",
            "casts",
            "aggregation",
        ],
        "guidance_candidate_pool": 4,
        "metamorphic_variant_limit": 4,
        "metamorphic_overlay_variant_limit": 8,
    },
    "null_predicate_filter": {
        "guidance_targets": ["null_predicate_filter", "null_predicate", "filter", "nulls", "aggregation", "sort_limit"],
        "guidance_candidate_pool": 4,
        "metamorphic_variant_limit": 4,
        "metamorphic_overlay_variant_limit": 8,
    },
    "boolean_predicate_filter": {
        "guidance_targets": ["boolean_predicate_filter", "boolean_predicate", "truth_filter", "filter", "nulls", "aggregation", "sort_limit"],
        "guidance_candidate_pool": 4,
        "metamorphic_variant_limit": 4,
        "metamorphic_overlay_variant_limit": 8,
    },
    "post_topk_range_filter": {
        "guidance_targets": ["post_topk_range_filter", "range_filter", "filter", "sort_limit", "topk", "nulls"],
        "guidance_candidate_pool": 4,
        "metamorphic_variant_limit": 4,
        "metamorphic_overlay_variant_limit": 8,
    },
    "tuple_absence_filter": {
        "guidance_targets": ["tuple_absence_filter", "tuple_absence", "filter", "nulls", "join"],
        "guidance_candidate_pool": 4,
        "metamorphic_variant_limit": 4,
        "metamorphic_overlay_variant_limit": 8,
    },
    "row_value_absence_filter": {
        "guidance_targets": ["row_value_absence_filter", "row_value_absence", "tuple_absence", "filter", "nulls", "join"],
        "guidance_candidate_pool": 4,
        "metamorphic_variant_limit": 4,
        "metamorphic_overlay_variant_limit": 8,
    },
    "running_sum_precision": {
        "guidance_targets": ["running_sum_precision", "running_sum", "numeric", "sort_limit"],
        "guidance_candidate_pool": 4,
        "metamorphic_variant_limit": 2,
        "metamorphic_overlay_variant_limit": 4,
    },
    "partitioned_running_sum": {
        "guidance_targets": [
            "partitioned_running_sum",
            "running_sum_partitioned",
            "running_sum",
            "numeric",
            "sort_limit",
        ],
        "guidance_candidate_pool": 4,
        "metamorphic_variant_limit": 2,
        "metamorphic_overlay_variant_limit": 4,
    },
    "path_basename_keyed_pick": {
        "guidance_targets": [
            "path_basename_keyed_pick",
            "path_projection",
            "keyed_row_pick",
            "strings",
            "sort_limit",
        ],
        "guidance_candidate_pool": 4,
        "metamorphic_variant_limit": 2,
        "register_metamorphic_overlay": False,
    },
    "sortedness_null_placement": {
        "guidance_targets": ["sortedness_null_placement", "sortedness", "sort_limit", "nulls"],
        "guidance_candidate_pool": 4,
        "metamorphic_variant_limit": 2,
        "metamorphic_overlay_variant_limit": 4,
    },
    "simple_case_random_subject": {
        "guidance_targets": ["simple_case_random_subject", "random_case_probe", "case_expression"],
        "guidance_candidate_pool": 2,
        "metamorphic_variant_limit": 0,
        "metamorphic_overlay_variant_limit": 2,
    },
    "group_quantile_key_probe": {
        "guidance_targets": ["group_quantile_key_probe", "group_quantile_probe", "dynamic_quantile", "groupby"],
        "guidance_candidate_pool": 2,
        "metamorphic_variant_limit": 0,
        "metamorphic_overlay_variant_limit": 2,
    },
    "scalar_subquery_double_parentheses": {
        "guidance_targets": ["scalar_subquery_double_parentheses", "scalar_subquery_probe", "correlated_subquery"],
        "guidance_candidate_pool": 2,
        "metamorphic_variant_limit": 0,
        "metamorphic_overlay_variant_limit": 2,
    },
    "window_avg_rows_frame": {
        "guidance_targets": ["window_avg_rows_frame", "window_avg_probe", "window_frame", "numeric"],
        "guidance_candidate_pool": 2,
        "metamorphic_variant_limit": 0,
        "metamorphic_overlay_variant_limit": 2,
    },
    "struct_distinct_unnest": {
        "guidance_targets": ["struct_distinct_unnest", "struct_distinct_probe", "struct_unnest"],
        "guidance_candidate_pool": 2,
        "metamorphic_variant_limit": 0,
        "metamorphic_overlay_variant_limit": 2,
    },
    "bit_compare_unequal_length": {
        "guidance_targets": ["bit_compare_unequal_length", "bit_compare_probe", "bit_ordering"],
        "guidance_candidate_pool": 2,
        "metamorphic_variant_limit": 0,
        "metamorphic_overlay_variant_limit": 2,
    },
    "round_even_float_scale": {
        "guidance_targets": ["round_even_float_scale", "round_even_probe", "rounding", "numeric"],
        "guidance_candidate_pool": 2,
        "metamorphic_variant_limit": 0,
        "metamorphic_overlay_variant_limit": 2,
    },
    "duckdb_float_literal_precision": {
        "guidance_targets": [
            "duckdb_float_literal_precision",
            "float_literal_precision_probe",
            "float_literal_precision",
            "numeric",
        ],
        "guidance_candidate_pool": 2,
        "metamorphic_variant_limit": 0,
        "metamorphic_overlay_variant_limit": 2,
    },
    "polars_timestamp_precision_filter": {
        "guidance_targets": [
            "polars_timestamp_precision_filter",
            "timestamp_precision_filter_probe",
            "timestamp_precision_filter",
            "casts",
        ],
        "guidance_candidate_pool": 2,
        "metamorphic_variant_limit": 0,
        "metamorphic_overlay_variant_limit": 2,
    },
    "series_rtruediv_operand_order": {
        "guidance_targets": [
            "series_rtruediv_operand_order",
            "series_rtruediv_probe",
            "reverse_division",
            "numeric",
        ],
        "guidance_candidate_pool": 2,
        "metamorphic_variant_limit": 0,
        "metamorphic_overlay_variant_limit": 2,
    },
    "polars_reverse_division_columns": {
        "guidance_targets": [
            "polars_reverse_division_columns",
            "reverse_division",
            "numeric",
        ],
        "guidance_candidate_pool": 2,
        "metamorphic_variant_limit": 0,
        "metamorphic_overlay_variant_limit": 2,
    },
    "pandas_uint64_isin_precision": {
        "guidance_targets": [
            "pandas_uint64_isin_precision",
            "uint64_isin_probe",
            "unsigned_membership",
            "numeric",
        ],
        "guidance_candidate_pool": 2,
        "metamorphic_variant_limit": 0,
        "metamorphic_overlay_variant_limit": 2,
    },
    "duckdb_tuple_anti_null_semantics": {
        "guidance_targets": [
            "duckdb_tuple_anti_null_semantics",
            "tuple_anti_null_probe",
            "tuple_null_membership",
            "nulls",
        ],
        "guidance_candidate_pool": 2,
        "metamorphic_variant_limit": 0,
        "metamorphic_overlay_variant_limit": 2,
    },
    "datafusion_setop_all_duplicate_count": {
        "guidance_targets": [
            "datafusion_setop_all_duplicate_count",
            "setop_all_duplicate_probe",
            "setop_all_duplicates",
            "aggregation",
        ],
        "guidance_candidate_pool": 2,
        "metamorphic_variant_limit": 0,
        "metamorphic_overlay_variant_limit": 2,
    },
    "duckdb_json_predicate_order_semantics": {
        "guidance_targets": [
            "duckdb_json_predicate_order_semantics",
            "json_predicate_order_probe",
            "json_predicate_order",
            "filter",
        ],
        "guidance_candidate_pool": 2,
        "metamorphic_variant_limit": 0,
        "metamorphic_overlay_variant_limit": 2,
    },
    "pandas_sparse_array_mask_semantics": {
        "guidance_targets": [
            "pandas_sparse_array_mask_semantics",
            "sparse_mask_probe",
            "sparse_masking",
            "filter",
        ],
        "guidance_candidate_pool": 2,
        "metamorphic_variant_limit": 0,
        "metamorphic_overlay_variant_limit": 2,
    },
    "polars_float_wrap_numerical_semantics": {
        "guidance_targets": [
            "polars_float_wrap_numerical_semantics",
            "float_wrap_probe",
            "wrap_numerical",
            "casts",
        ],
        "guidance_candidate_pool": 2,
        "metamorphic_variant_limit": 0,
        "metamorphic_overlay_variant_limit": 2,
    },
    "pandas_index_bool_result_type": {
        "guidance_targets": [
            "pandas_index_bool_result_type",
            "index_bool_probe",
            "index_boolean_result",
            "filter",
        ],
        "guidance_candidate_pool": 2,
        "metamorphic_variant_limit": 0,
        "metamorphic_overlay_variant_limit": 2,
    },
    "polars_empty_literal_groupby_semantics": {
        "guidance_targets": [
            "polars_empty_literal_groupby_semantics",
            "empty_literal_groupby_probe",
            "literal_empty_groupby",
            "groupby",
        ],
        "guidance_candidate_pool": 2,
        "metamorphic_variant_limit": 0,
        "metamorphic_overlay_variant_limit": 2,
    },
    "pandas_arrow_string_eq_sum_semantics": {
        "guidance_targets": [
            "pandas_arrow_string_eq_sum_semantics",
            "arrow_string_eq_sum_probe",
            "arrow_string_reduction",
            "strings",
        ],
        "guidance_candidate_pool": 2,
        "metamorphic_variant_limit": 0,
        "metamorphic_overlay_variant_limit": 2,
    },
    "pandas_arrow_timestamp_loc_slice_semantics": {
        "guidance_targets": [
            "pandas_arrow_timestamp_loc_slice_semantics",
            "arrow_timestamp_loc_slice_probe",
            "arrow_timestamp_indexing",
            "sort_limit",
        ],
        "guidance_candidate_pool": 2,
        "metamorphic_variant_limit": 0,
        "metamorphic_overlay_variant_limit": 2,
    },
    "pandas_arrow_timestamp_index_attr_semantics": {
        "guidance_targets": [
            "pandas_arrow_timestamp_index_attr_semantics",
            "arrow_timestamp_index_attr_probe",
            "arrow_timestamp_attributes",
            "sort_limit",
        ],
        "guidance_candidate_pool": 2,
        "metamorphic_variant_limit": 0,
        "metamorphic_overlay_variant_limit": 2,
    },
    "pyarrow_dataset_isin_all_match_semantics": {
        "guidance_targets": [
            "pyarrow_dataset_isin_all_match_semantics",
            "dataset_isin_all_match_probe",
            "dataset_membership_filter",
            "filter",
        ],
        "guidance_candidate_pool": 2,
        "metamorphic_variant_limit": 0,
    },
    "pandas_eval_inplace_aliasing_semantics": {
        "guidance_targets": [
            "pandas_eval_inplace_aliasing_semantics",
            "eval_inplace_alias_probe",
            "eval_inplace_aliasing",
            "mutate",
        ],
        "guidance_candidate_pool": 2,
        "metamorphic_variant_limit": 0,
        "metamorphic_overlay_variant_limit": 2,
    },
    "pandas_bool_reduction_skipna_semantics": {
        "guidance_targets": [
            "pandas_bool_reduction_skipna_semantics",
            "bool_reduction_skipna_probe",
            "bool_reduction_skipna",
            "nulls",
        ],
        "guidance_candidate_pool": 2,
        "metamorphic_variant_limit": 0,
        "metamorphic_overlay_variant_limit": 2,
    },
    "pandas_arrow_bool_groupby_reduction_semantics": {
        "guidance_targets": [
            "pandas_arrow_bool_groupby_reduction_semantics",
            "arrow_bool_groupby_reduction_probe",
            "arrow_bool_groupby_reduction",
            "aggregation",
            "nulls",
        ],
        "guidance_candidate_pool": 2,
        "metamorphic_variant_limit": 0,
        "metamorphic_overlay_variant_limit": 2,
    },
    "pyarrow_run_end_null_compute_semantics": {
        "guidance_targets": [
            "pyarrow_run_end_null_compute_semantics",
            "run_end_null_compute_probe",
            "run_end_null_compute",
            "nulls",
        ],
        "guidance_candidate_pool": 2,
        "metamorphic_variant_limit": 0,
        "metamorphic_overlay_variant_limit": 2,
    },
    "pyarrow_large_string_partition_schema_semantics": {
        "guidance_targets": [
            "pyarrow_large_string_partition_schema_semantics",
            "large_string_partition_probe",
            "large_string_partition",
            "strings",
        ],
        "guidance_candidate_pool": 2,
        "metamorphic_variant_limit": 0,
        "metamorphic_overlay_variant_limit": 2,
    },
    "pyarrow_hash_pivot_wider_order_semantics": {
        "guidance_targets": [
            "pyarrow_hash_pivot_wider_order_semantics",
            "hash_pivot_wider_probe",
            "hash_pivot_wider",
            "aggregation",
        ],
        "guidance_candidate_pool": 2,
        "metamorphic_variant_limit": 0,
    },
    "pyarrow_list_flatten_parent_indices_semantics": {
        "guidance_targets": [
            "pyarrow_list_flatten_parent_indices_semantics",
            "list_flatten_parent_indices_probe",
            "list_layout",
            "nulls",
        ],
        "guidance_candidate_pool": 2,
        "metamorphic_variant_limit": 0,
        "metamorphic_overlay_variant_limit": 2,
    },
    "polars_rolling_mean_by_null_count_semantics": {
        "guidance_targets": [
            "polars_rolling_mean_by_null_count_semantics",
            "rolling_mean_by_null_count_probe",
            "rolling_temporal_nulls",
            "nulls",
        ],
        "guidance_candidate_pool": 2,
        "metamorphic_variant_limit": 0,
        "metamorphic_overlay_variant_limit": 2,
    },
    "csv_long_numeric_roundtrip": {
        "guidance_targets": [
            "csv_long_numeric_roundtrip",
            "csv_long_numeric_roundtrip_probe",
            "csv_numeric_inference",
            "numeric",
        ],
        "guidance_candidate_pool": 2,
        "metamorphic_variant_limit": 0,
        "candidate_recheck_count": 2,
        "metamorphic_overlay_variant_limit": 2,
    },
}


PRESET_CATALOG: dict[str, PresetSpec] = {
    "baseline": _config_spec(),
    "no_type_aware": _overlay_preset(base_preset="baseline", overlays=("disable_type_aware_generation",)),
    "no_normalizer": _overlay_preset(base_preset="baseline", overlays=("disable_normalizer",)),
    "no_feedback": _overlay_preset(base_preset="baseline", overlays=("disable_feedback_corpus",)),
    "metamorphic": _overlay_preset(base_preset="baseline", overlays=("enable_metamorphic_oracle",)),
    "reducer": _overlay_preset(base_preset="baseline", overlays=("enable_reducer",)),
    "oracle_only_metamorphic": _overlay_preset(
        base_preset="baseline",
        overlays=("disable_differential_oracle", "enable_metamorphic_oracle"),
    ),
    "workflow": _overlay_preset(base_preset="baseline", overlays=("generator_workflow",)),
    "workflow_metamorphic": _metamorphic_overlay(base_preset="workflow", variant_limit=4),
    "discovery": _overlay_preset(base_preset="baseline", overlays=("generator_discovery",)),
    "discovery_no_groupby": _config_spec(generator_profile="discovery_no_groupby"),
    "discovery_guided": _overlay_preset(base_preset="discovery", overlays=("enable_guidance", "target_discovery_guided")),
    "discovery_no_groupby_guided": _guided_profile(
        generator_profile="discovery_no_groupby",
        guidance_targets=["join", "mutate", "filter", "expressions", "sort_limit"],
    ),
    "discovery_metamorphic": _metamorphic_overlay(base_preset="discovery", variant_limit=8),
    "discovery_no_groupby_metamorphic": _metamorphic_overlay(
        base_preset="discovery_no_groupby",
        variant_limit=8,
    ),
    "discovery_guided_metamorphic": _metamorphic_overlay(
        base_preset="discovery_guided",
        variant_limit=8,
    ),
    "discovery_no_groupby_guided_metamorphic": _metamorphic_overlay(
        base_preset="discovery_no_groupby_guided",
        variant_limit=8,
    ),
    "guided": _overlay_preset(base_preset="baseline", overlays=("enable_guidance",)),
    "guided_filter": _overlay_preset(base_preset="baseline", overlays=("enable_guidance", "target_filter")),
    "guided_groupby": _overlay_preset(base_preset="baseline", overlays=("enable_guidance", "target_groupby")),
    "guided_join": _overlay_preset(base_preset="baseline", overlays=("enable_guidance", "target_join")),
    "guided_mutate": _overlay_preset(base_preset="baseline", overlays=("enable_guidance", "target_mutate")),
    "live_datafusion": _live_preset(
        "live_datafusion",
        local_source_exploration_weight=0.35,
    ),
    "live_datafusion_metamorphic": _metamorphic_overlay(
        base_preset="live_datafusion",
        candidate_pool=8,
        variant_limit=6,
    ),
    "live_datafusion_fresh": _live_preset(
        "live_datafusion_fresh",
        generator_profile="discovery_no_groupby",
        local_source_exploration_weight=0.45,
        discovery_lanes=("datafusion_optimizer",),
    ),
    "live_datafusion_fresh_metamorphic": _metamorphic_overlay(
        base_preset="live_datafusion_fresh",
        candidate_pool=8,
        variant_limit=6,
    ),
    "live_arrow": _live_preset(
        "live_arrow",
        local_source_exploration_weight=0.45,
    ),
    "live_arrow_metamorphic": _metamorphic_overlay(
        base_preset="live_arrow",
        candidate_pool=8,
        variant_limit=6,
    ),
    "live_polars_lazy": _live_preset(
        "live_polars_lazy",
        local_source_exploration_weight=0.45,
    ),
    "live_polars_lazy_metamorphic": _metamorphic_overlay(
        base_preset="live_polars_lazy",
        candidate_pool=8,
        variant_limit=8,
    ),
    "live_polars_streaming": _live_preset(
        "live_polars_streaming",
        local_source_exploration_weight=0.45,
    ),
    "live_polars_streaming_metamorphic": _metamorphic_overlay(
        base_preset="live_polars_streaming",
        candidate_pool=8,
        variant_limit=8,
    ),
    "live_embedded_sql": _live_preset(
        "live_embedded_sql",
        local_source_exploration_weight=0.40,
    ),
    "live_embedded_sql_metamorphic": _metamorphic_overlay(
        base_preset="live_embedded_sql",
        candidate_pool=8,
        variant_limit=8,
    ),
    "live_cross_family": _live_preset(
        "live_cross_family",
        local_source_exploration_weight=0.40,
    ),
    "live_cross_family_metamorphic": _config_spec(
        base_preset="live_cross_family",
        enable_metamorphic_oracle=True,
        oracle_mode="both",
        guidance_candidate_pool=8,
        metamorphic_variant_limit=6,
        discovery_biases=discovery_biases_for_lanes("cross_family"),
    ),
    "live_deep_organic": _live_preset(
        "live_deep_organic",
        generator_profile="discovery_fresh",
        candidate_pool=14,
        local_source_exploration_weight=0.45,
        discovery_lanes=("cross_family",),
    ),
    "live_deep_organic_metamorphic": _metamorphic_overlay(
        base_preset="live_deep_organic",
        candidate_pool=10,
        variant_limit=8,
    ),
    "live_common_api_workflow": _live_preset(
        "live_common_api_workflow",
        generator_profile="common_api_workflow",
        candidate_pool=8,
        local_source_exploration_weight=0.35,
        metamorphic_variant_limit=0,
        discovery_lanes=("common_api_workflow",),
    ),
    "live_common_api_workflow_metamorphic": _metamorphic_overlay(
        base_preset="live_common_api_workflow",
        candidate_pool=8,
        variant_limit=6,
    ),
    "live_datafusion_common_api": _live_preset(
        "live_datafusion_common_api",
        generator_profile="common_api_workflow",
        candidate_pool=8,
        local_source_exploration_weight=0.35,
        metamorphic_variant_limit=0,
        discovery_lanes=("datafusion_common_api",),
    ),
    "live_datafusion_common_api_metamorphic": _metamorphic_overlay(
        base_preset="live_datafusion_common_api",
        candidate_pool=8,
        variant_limit=6,
    ),
    "live_arrow_deep_organic": _live_preset(
        "live_arrow_deep_organic",
        generator_profile="discovery_fresh",
        candidate_pool=10,
        local_source_exploration_weight=0.45,
    ),
    "live_arrow_deep_organic_metamorphic": _config_spec(
        base_preset="live_arrow_deep_organic",
        enable_metamorphic_oracle=True,
        oracle_mode="both",
        guidance_candidate_pool=8,
        metamorphic_variant_limit=6,
        discovery_biases=discovery_biases_for_lanes("arrow_layout"),
    ),
    "live_polars_deep_organic": _live_preset(
        "live_polars_deep_organic",
        generator_profile="discovery_fresh",
        candidate_pool=10,
        local_source_exploration_weight=0.45,
    ),
    "live_polars_deep_organic_metamorphic": _config_spec(
        base_preset="live_polars_deep_organic",
        enable_metamorphic_oracle=True,
        oracle_mode="both",
        guidance_candidate_pool=8,
        metamorphic_variant_limit=8,
        discovery_biases=discovery_biases_for_lanes("polars_lazy"),
    ),
    "live_polars_streaming_deep_organic": _live_preset(
        "live_polars_streaming_deep_organic",
        generator_profile="discovery_fresh",
        candidate_pool=10,
        local_source_exploration_weight=0.45,
    ),
    "live_polars_streaming_deep_organic_metamorphic": _config_spec(
        base_preset="live_polars_streaming_deep_organic",
        enable_metamorphic_oracle=True,
        oracle_mode="both",
        guidance_candidate_pool=8,
        metamorphic_variant_limit=8,
        discovery_biases=discovery_biases_for_lanes("polars_streaming"),
    ),
    "live_datafusion_deep_organic": _live_preset(
        "live_datafusion_deep_organic",
        generator_profile="discovery_fresh",
        candidate_pool=10,
        local_source_exploration_weight=0.45,
    ),
    "live_datafusion_deep_organic_metamorphic": _config_spec(
        base_preset="live_datafusion_deep_organic",
        enable_metamorphic_oracle=True,
        oracle_mode="both",
        guidance_candidate_pool=8,
        metamorphic_variant_limit=6,
        discovery_biases=discovery_biases_for_lanes("datafusion_optimizer"),
    ),
    "live_embedded_sql_deep_organic": _live_preset(
        "live_embedded_sql_deep_organic",
        generator_profile="discovery_fresh",
        candidate_pool=10,
        local_source_exploration_weight=0.45,
    ),
    "live_embedded_sql_deep_organic_metamorphic": _config_spec(
        base_preset="live_embedded_sql_deep_organic",
        enable_metamorphic_oracle=True,
        oracle_mode="both",
        guidance_candidate_pool=8,
        metamorphic_variant_limit=8,
        discovery_biases=discovery_biases_for_lanes("embedded_sql"),
    ),
    "live_issue_focus": _live_preset(
        "live_issue_focus",
        generator_profile="issue_focus",
        candidate_pool=14,
        local_source_exploration_weight=0.50,
    ),
    "live_issue_focus_metamorphic": _metamorphic_overlay(
        base_preset="live_issue_focus",
        candidate_pool=10,
        variant_limit=6,
    ),
    "live_polars_issue_focus": _live_preset(
        "live_polars_issue_focus",
        generator_profile="issue_focus",
        candidate_pool=12,
        local_source_exploration_weight=0.50,
    ),
    "live_polars_issue_focus_metamorphic": _metamorphic_overlay(
        base_preset="live_polars_issue_focus",
        candidate_pool=8,
        variant_limit=8,
    ),
    "live_duckdb_issue_focus": _live_preset(
        "live_duckdb_issue_focus",
        generator_profile="issue_focus",
        candidate_pool=12,
        local_source_exploration_weight=0.50,
    ),
    "live_duckdb_issue_focus_metamorphic": _metamorphic_overlay(
        base_preset="live_duckdb_issue_focus",
        candidate_pool=8,
        variant_limit=8,
    ),
    "live_arrow_issue_focus": _live_preset(
        "live_arrow_issue_focus",
        generator_profile="issue_focus",
        candidate_pool=12,
        local_source_exploration_weight=0.50,
    ),
    "live_arrow_issue_focus_metamorphic": _metamorphic_overlay(
        base_preset="live_arrow_issue_focus",
        candidate_pool=8,
        variant_limit=6,
    ),
    "live_deep_probe_rotation": _live_preset(
        "live_deep_probe_rotation",
        generator_profile="deep_probe_rotation",
        candidate_pool=6,
        local_source_exploration_weight=0.25,
        metamorphic_variant_limit=0,
    ),
    "live_deep_probe_rotation_metamorphic": _metamorphic_overlay(
        base_preset="live_deep_probe_rotation",
        candidate_pool=4,
        variant_limit=2,
    ),
}

for _profile_name, _profile_spec in PROFILE_PRESET_SPECS.items():
    PRESET_CATALOG.setdefault(
        _profile_name,
        _profile_base_spec(name=_profile_name, spec=_profile_spec),
    )
    if not bool(_profile_spec.get("register_metamorphic_overlay", True)):
        continue
    PRESET_CATALOG.setdefault(
        f"{_profile_name}_metamorphic",
        _profile_overlay_spec(name=_profile_name, spec=_profile_spec),
    )

CATALOG_PRESET_NAMES = frozenset(PRESET_CATALOG)


def catalog_preset_semantic_focus(name: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    normalized_name = str(name or "").strip()
    if not normalized_name:
        return (), ()
    return _resolved_semantic_focus(normalized_name, trail=())


def catalog_preset_metadata(
    name: str,
    *,
    base_preset: str = "",
    overlays: Sequence[str] | None = None,
    live_targets_by_name: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    preset_id = str(name or "").strip()
    normalized_base = str(base_preset or "").strip()
    overlay_list = [str(item).strip() for item in overlays or () if str(item).strip()]
    replay_overlay = False
    catalog_name = preset_id
    if catalog_name.endswith("_replay"):
        catalog_name = catalog_name[: -len("_replay")]
        replay_overlay = True
    if not normalized_base and overlay_list:
        normalized_base = catalog_name
    config: ExperimentConfig | None
    registered = False
    if normalized_base or overlay_list:
        try:
            config = build_experiment_config(
                normalized_base,
                overlay_list,
                live_targets_by_name=live_targets_by_name,
            )
            registered = bool(normalized_base in PRESET_CATALOG)
        except ValueError:
            config = None
    else:
        config = build_catalog_preset(catalog_name, live_targets_by_name=live_targets_by_name)
        registered = config is not None
        normalized_base = catalog_name
    if config is None:
        return {
            "schema_version": "preset-methodology-metadata-v1",
            "preset_id": preset_id,
            "registered": False,
            "base_catalog_preset": normalized_base,
            "overlays": overlay_list,
            "is_replay_overlay": replay_overlay,
        }
    if replay_overlay:
        config.enable_replay_bug = True
    base_spec = PRESET_CATALOG.get(normalized_base)
    resolved_base_spec = _resolved_terminal_spec(normalized_base, trail=()) if normalized_base else None
    live_spec = _resolved_live_spec(normalized_base, trail=()) if normalized_base else None
    families, signals = _metadata_semantic_focus(
        normalized_base,
        overlay_list,
        config=config,
    )
    discovery_biases = [bias.to_dict() for bias in config.discovery_biases]
    methodology_tags = _methodology_tags(
        preset_id=preset_id,
        base_preset=normalized_base,
        overlays=overlay_list,
        config=config,
        live_spec=live_spec,
        replay_overlay=replay_overlay,
    )
    return {
        "schema_version": "preset-methodology-metadata-v1",
        "preset_id": preset_id,
        "registered": registered,
        "base_catalog_preset": normalized_base,
        "base_preset": base_spec.base_preset if base_spec is not None else "",
        "resolved_catalog_preset": _resolved_catalog_preset_id(normalized_base, trail=()),
        "overlays": overlay_list or (list(base_spec.overlays) if base_spec is not None else []),
        "catalog_overlays": list(base_spec.overlays) if base_spec is not None else [],
        "is_replay_overlay": replay_overlay,
        "is_live_preset": live_spec is not None,
        "live_target_key": live_spec.target_key if live_spec is not None else "",
        "discovery_lanes": list(live_spec.discovery_lanes) if live_spec is not None else [],
        "target_suite_affinity": _target_suite_affinity(live_spec),
        "generator_profile": str(config.generator_profile or ""),
        "generator_profile_pool": list(config.generator_profile_pool),
        "guidance_strategy": str(config.guidance_strategy or ""),
        "guidance_candidate_pool": int(config.guidance_candidate_pool or 0),
        "guidance_targets": list(config.guidance_targets),
        "semantic_focus_families": list(families),
        "semantic_focus_signals": list(signals),
        "discovery_bias_count": len(discovery_biases),
        "discovery_biases": discovery_biases,
        "discovery_bias_targets": _discovery_bias_targets(config.discovery_biases),
        "enable_differential_oracle": bool(config.enable_differential_oracle),
        "enable_metamorphic_oracle": bool(config.enable_metamorphic_oracle),
        "oracle_mode": str(config.oracle_mode or ""),
        "metamorphic_variant_limit": int(config.metamorphic_variant_limit or 0),
        "enable_feedback": bool(config.enable_feedback),
        "enable_replay_bug": bool(config.enable_replay_bug),
        "enable_local_source_scheduler": bool(config.enable_local_source_scheduler),
        "local_source_exploration_weight": float(config.local_source_exploration_weight or 0.0),
        "candidate_recheck_count": int(config.candidate_recheck_count or 0),
        "known_saturated_bug_family_count": len(config.known_saturated_bug_families),
        "replay_bug_source_issue_count": len(config.replay_bug_source_issues),
        "methodology_tags": methodology_tags,
        "methodology_claim": _metadata_methodology_claim(methodology_tags),
        "resolved_from_live_base": (
            resolved_base_spec.live is not None
            if resolved_base_spec is not None
            else False
        ),
    }


def _resolved_terminal_spec(name: str, *, trail: tuple[str, ...]) -> PresetSpec | None:
    spec = PRESET_CATALOG.get(name)
    if spec is None:
        return None
    if name in trail:
        chain = " -> ".join((*trail, name))
        raise ValueError(f"cyclic preset catalog chain: {chain}")
    if spec.base_preset:
        return _resolved_terminal_spec(spec.base_preset, trail=(*trail, name)) or spec
    return spec


def _resolved_live_spec(name: str, *, trail: tuple[str, ...]) -> LivePresetSpec | None:
    spec = PRESET_CATALOG.get(name)
    if spec is None:
        return None
    if name in trail:
        chain = " -> ".join((*trail, name))
        raise ValueError(f"cyclic preset catalog chain: {chain}")
    if spec.live is not None:
        return spec.live
    if spec.base_preset:
        return _resolved_live_spec(spec.base_preset, trail=(*trail, name))
    return None


def _resolved_catalog_preset_id(name: str, *, trail: tuple[str, ...]) -> str:
    spec = PRESET_CATALOG.get(name)
    if spec is None:
        return ""
    if name in trail:
        chain = " -> ".join((*trail, name))
        raise ValueError(f"cyclic preset catalog chain: {chain}")
    if spec.live is not None or not spec.base_preset:
        return name
    return _resolved_catalog_preset_id(spec.base_preset, trail=(*trail, name)) or name


def _metadata_semantic_focus(
    base_preset: str,
    overlays: Sequence[str],
    *,
    config: ExperimentConfig,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    families, signals = catalog_preset_semantic_focus(base_preset)
    if overlays:
        families = tuple(dict.fromkeys([*families, *config.semantic_focus_families]))
        signals = tuple(dict.fromkeys([*signals, *config.semantic_focus_signals]))
    return families, signals


def _target_suite_affinity(live_spec: LivePresetSpec | None) -> list[str]:
    if live_spec is None:
        return []
    suites = []
    if live_spec.target_key:
        suites.append(live_spec.target_key)
    for lane_id in live_spec.discovery_lanes:
        lane = discovery_lane_spec(lane_id)
        if lane.target_suite and lane.target_suite not in suites:
            suites.append(lane.target_suite)
    return suites


def _discovery_bias_targets(discovery_biases: Sequence[DiscoveryBias]) -> list[str]:
    targets: list[str] = []
    for bias in discovery_biases:
        for target in bias.targets:
            if target and target not in targets:
                targets.append(target)
    return targets


def _methodology_tags(
    *,
    preset_id: str,
    base_preset: str,
    overlays: Sequence[str],
    config: ExperimentConfig,
    live_spec: LivePresetSpec | None,
    replay_overlay: bool,
) -> list[str]:
    tags: list[str] = []
    name_parts = [preset_id, base_preset, *(str(item) for item in overlays)]
    joined = " ".join(name_parts)
    if live_spec is not None or preset_id.startswith("live_"):
        tags.append("latest_live_discovery")
    if replay_overlay or config.enable_replay_bug:
        tags.append("cross_version_replay")
    if config.guidance_strategy == "guided":
        tags.append("guided_search")
    if config.enable_feedback:
        tags.append("closed_loop_feedback")
    if config.enable_metamorphic_oracle or config.oracle_mode in {"metamorphic", "both"}:
        tags.append("metamorphic_oracle")
    if config.enable_differential_oracle:
        tags.append("differential_oracle")
    if config.enable_local_source_scheduler:
        tags.append("local_source_scheduler")
    if config.discovery_biases:
        tags.append("discovery_bias")
    if "common_api" in joined:
        tags.append("common_api_workflow")
    if "deep_organic" in joined:
        tags.append("deep_organic")
    if "issue_focus" in joined:
        tags.append("issue_inspired")
    if "ablation" in joined or any(str(item).startswith("disable_") for item in overlays):
        tags.append("ablation_support")
    if not (replay_overlay or config.enable_replay_bug):
        tags.append("fresh_candidate_discovery")
    return list(dict.fromkeys(tags))


def _metadata_methodology_claim(tags: Sequence[str]) -> str:
    tag_set = set(tags)
    if "cross_version_replay" in tag_set:
        return "Replay-oriented preset metadata supports cross-version rediscovery and regression accounting."
    if "latest_live_discovery" in tag_set:
        return "Live preset metadata records the generator, guidance, oracle, and bias choices used for latest-version discovery."
    if "ablation_support" in tag_set:
        return "Ablation preset metadata records controlled component changes for paper comparison tables."
    return "Preset metadata records the reproducible harness configuration used by the experiment run."


def _resolved_semantic_focus(name: str, *, trail: tuple[str, ...]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    spec = PRESET_CATALOG.get(name)
    if spec is None:
        return (), ()
    if name in trail:
        chain = " -> ".join((*trail, name))
        raise ValueError(f"cyclic preset catalog chain: {chain}")

    families: list[str] = []
    signals: list[str] = []
    if spec.base_preset:
        base_families, base_signals = _resolved_semantic_focus(spec.base_preset, trail=(*trail, name))
        families.extend(base_families)
        signals.extend(base_signals)
    if spec.live is not None:
        live_families, live_signals = _live_semantic_focus(spec.live)
        families = list(live_families)
        signals = list(live_signals)

    config = spec.config if isinstance(spec.config, dict) else {}
    families.extend(str(item).strip() for item in config.get("semantic_focus_families", []) or [])
    signals.extend(str(item).strip() for item in config.get("semantic_focus_signals", []) or [])

    return (
        tuple(dict.fromkeys(item for item in families if item)),
        tuple(dict.fromkeys(item for item in signals if item)),
    )


def _resolved_live_targets_by_name(
    live_targets_by_name: Mapping[str, Sequence[str]] | None = None,
) -> Mapping[str, Sequence[str]]:
    return live_targets_by_name or DEFAULT_LIVE_TARGETS_BY_NAME


def _build_live_config(
    spec: LivePresetSpec,
    live_targets_by_name: Mapping[str, Sequence[str]] | None = None,
) -> ExperimentConfig:
    live_targets_by_name = _resolved_live_targets_by_name(live_targets_by_name)
    guidance_targets = live_targets_by_name.get(spec.target_key)
    if guidance_targets is None:
        raise KeyError(f"missing live target catalog entry for preset key: {spec.target_key}")
    semantic_focus_families, semantic_focus_signals = _live_semantic_focus(spec)
    return ExperimentConfig(
        enable_replay_bug=spec.enable_replay_bug,
        generator_profile=spec.generator_profile,
        guidance_strategy="guided",
        guidance_candidate_pool=spec.candidate_pool,
        guidance_targets=list(guidance_targets),
        semantic_focus_families=semantic_focus_families,
        semantic_focus_signals=semantic_focus_signals,
        discovery_biases=merge_discovery_biases(
            spec.discovery_biases,
            discovery_biases_for_lanes(*spec.discovery_lanes) if spec.discovery_lanes else None,
        ),
        enable_local_source_scheduler=True,
        local_source_exploration_weight=spec.local_source_exploration_weight,
        metamorphic_variant_limit=spec.metamorphic_variant_limit,
        family_saturation_threshold=spec.family_saturation_threshold,
        family_saturation_penalty=spec.family_saturation_penalty,
        saturated_family_reward=spec.saturated_family_reward,
        candidate_recheck_count=max(0, int(spec.candidate_recheck_count)),
        known_saturated_bug_families=list(
            spec.known_saturated_bug_families or DEFAULT_KNOWN_SATURATED_BUG_FAMILIES
        ),
        replay_bug_source_issues=list(spec.replay_bug_source_issues or DEFAULT_REPLAY_BUG_SOURCE_ISSUES),
        issue_replay_global_saturation_threshold=spec.issue_replay_global_saturation_threshold,
        issue_replay_global_saturation_penalty=spec.issue_replay_global_saturation_penalty,
    )


def _apply_overlay_updates(payload: dict[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    merged = deepcopy(payload)
    for key, value in updates.items():
        merged[key] = deepcopy(value)
    return merged


def build_experiment_config(
    base_preset: str,
    overlays: Sequence[str] | None = None,
    *,
    live_targets_by_name: Mapping[str, Sequence[str]] | None = None,
) -> ExperimentConfig:
    live_targets_by_name = _resolved_live_targets_by_name(live_targets_by_name)
    normalized_base = str(base_preset or "").strip()
    if not normalized_base:
        payload = ExperimentConfig().to_dict()
    else:
        base_payload = _resolved_payload(
            normalized_base,
            live_targets_by_name=live_targets_by_name,
            trail=(),
        )
        if base_payload is None:
            raise ValueError(f"unknown base preset in catalog: {normalized_base}")
        payload = deepcopy(base_payload)
    for overlay_name in overlays or ():
        normalized_overlay = str(overlay_name).strip()
        if not normalized_overlay:
            continue
        overlay = CONFIG_OVERLAYS.get(normalized_overlay)
        if overlay is None:
            raise ValueError(f"unknown config overlay: {normalized_overlay}")
        payload = _apply_overlay_updates(payload, overlay.updates)
    return ExperimentConfig(**payload)


def _resolved_payload(
    name: str,
    *,
    live_targets_by_name: Mapping[str, Sequence[str]] | None,
    trail: tuple[str, ...],
) -> dict[str, Any] | None:
    live_targets_by_name = _resolved_live_targets_by_name(live_targets_by_name)
    spec = PRESET_CATALOG.get(name)
    if spec is None:
        return None
    if name in trail:
        chain = " -> ".join((*trail, name))
        raise ValueError(f"cyclic preset catalog chain: {chain}")
    if spec.live is not None:
        payload = _build_live_config(spec.live, live_targets_by_name).to_dict()
    elif spec.base_preset:
        base_payload = _resolved_payload(
            spec.base_preset,
            live_targets_by_name=live_targets_by_name,
            trail=(*trail, name),
        )
        if base_payload is None:
            raise ValueError(f"unknown base preset in catalog: {spec.base_preset}")
        payload = deepcopy(base_payload)
    else:
        payload = ExperimentConfig().to_dict()
    for overlay_name in spec.overlays:
        overlay = CONFIG_OVERLAYS.get(overlay_name)
        if overlay is None:
            raise ValueError(f"unknown config overlay: {overlay_name}")
        payload = _apply_overlay_updates(payload, overlay.updates)
    payload.update(deepcopy(spec.config))
    return payload


def build_catalog_preset(
    name: str,
    *,
    live_targets_by_name: Mapping[str, Sequence[str]] | None = None,
) -> ExperimentConfig | None:
    payload = _resolved_payload(name, live_targets_by_name=live_targets_by_name, trail=())
    if payload is None:
        return None
    return ExperimentConfig(**payload)
