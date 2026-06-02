from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from datadiff.config import DiscoveryBias
from datadiff.config import merge_discovery_biases
from datadiff.semantic_signal import semantic_signal_bias_prefixes


@dataclass(frozen=True, slots=True)
class DiscoveryLaneSpec:
    lane_id: str
    target_suite: str
    preset: str
    theme: str
    default: bool = False
    semantic_focus_families: tuple[str, ...] = ()
    semantic_focus_signals: tuple[str, ...] = ()
    discovery_biases: tuple[DiscoveryBias, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.lane_id,
            "default": self.default,
            "target_suite": self.target_suite,
            "preset": self.preset,
            "theme": self.theme,
            "semantic_focus_families": list(self.semantic_focus_families),
            "semantic_focus_signals": list(self.semantic_focus_signals),
            "discovery_biases": [bias.to_dict() for bias in self.discovery_biases],
        }


def _bias(
    *,
    targets: list[str] | None = None,
    feature_prefixes: list[str] | None = None,
    score_bonus: float = 0.0,
    novelty_bonus: float = 0.0,
    contribution_bonus: float = 0.0,
    candidate_pool_bonus: float = 0.0,
    keep_in_pool: bool = False,
) -> DiscoveryBias:
    return DiscoveryBias(
        targets=list(targets or []),
        feature_prefixes=list(feature_prefixes or []),
        score_bonus=score_bonus,
        novelty_bonus=novelty_bonus,
        contribution_bonus=contribution_bonus,
        candidate_pool_bonus=candidate_pool_bonus,
        keep_in_pool=keep_in_pool,
    )


DISCOVERY_LANE_SPECS: tuple[DiscoveryLaneSpec, ...] = (
    DiscoveryLaneSpec(
        lane_id="arrow_layout",
        target_suite="arrow_cross",
        preset="live_arrow_deep_organic_metamorphic",
        theme="Organic Arrow table/list layout, nullable aggregation, string/cast, and compute equivalence",
        default=True,
        semantic_focus_families=("null_semantics", "type_coercion", "backend_specific_semantics"),
        semantic_focus_signals=("pyarrow_list_flatten_parent_indices_semantics", "pyarrow_hash_pivot_wider_order_semantics"),
        discovery_biases=(
            _bias(
                targets=["pyarrow_list_flatten_parent_indices_semantics", "pyarrow_hash_pivot_wider_order_semantics"],
                feature_prefixes=["op:", "pattern:"],
                score_bonus=0.35,
                novelty_bonus=0.15,
                contribution_bonus=0.10,
                candidate_pool_bonus=0.05,
                keep_in_pool=True,
            ),
        ),
    ),
    DiscoveryLaneSpec(
        lane_id="polars_lazy",
        target_suite="polars_cross",
        preset="live_polars_deep_organic_metamorphic",
        theme="Organic Polars eager/lazy state, sortedness, nullable dtype, and expression equivalence",
        default=True,
        semantic_focus_families=("stateful_ordering", "backend_specific_semantics"),
        semantic_focus_signals=("polars_reverse_division_columns", "polars_rolling_mean_by_null_count_semantics"),
        discovery_biases=(
            _bias(
                targets=["polars_reverse_division_columns", "polars_rolling_mean_by_null_count_semantics"],
                feature_prefixes=["expr:", "pattern:"],
                score_bonus=0.30,
                novelty_bonus=0.10,
                contribution_bonus=0.10,
                keep_in_pool=True,
            ),
        ),
    ),
    DiscoveryLaneSpec(
        lane_id="polars_streaming",
        target_suite="polars_streaming_cross",
        preset="live_polars_streaming_deep_organic_metamorphic",
        theme="Organic Polars lazy versus streaming execution state equivalence",
        default=True,
        semantic_focus_families=("stateful_ordering", "topk_ordering"),
        semantic_focus_signals=("partitioned_running_sum", "groupby_sorted_input"),
        discovery_biases=(
            _bias(
                targets=["partitioned_running_sum", "groupby_sorted_input"],
                feature_prefixes=["running:", "sort:"],
                score_bonus=0.25,
                novelty_bonus=0.10,
                contribution_bonus=0.15,
            ),
        ),
    ),
    DiscoveryLaneSpec(
        lane_id="datafusion_optimizer",
        target_suite="datafusion_cross",
        preset="live_datafusion_deep_organic_metamorphic",
        theme="Organic DataFusion optimizer, join/filter/projection, and SQL metamorphic equivalence",
        default=True,
        semantic_focus_families=("join_membership", "string_semantics", "topk_ordering"),
        semantic_focus_signals=("row_value_absence_filter", "normalized_string_join", "negative_set_membership_filter"),
        discovery_biases=(
            _bias(
                targets=["row_value_absence_filter", "normalized_string_join", "negative_set_membership_filter"],
                feature_prefixes=["join:", "filter:", "pattern:"],
                score_bonus=0.45,
                novelty_bonus=0.20,
                contribution_bonus=0.20,
                candidate_pool_bonus=0.10,
                keep_in_pool=True,
            ),
        ),
    ),
    DiscoveryLaneSpec(
        lane_id="datafusion_common_api",
        target_suite="datafusion_cross",
        preset="live_datafusion_common_api_metamorphic",
        theme="DataFusion daily low-complexity SQL/DataFrame workflows, including nullable DISTINCT top-k",
        default=True,
        semantic_focus_families=("materialization_boundary", "set_semantics", "join_membership"),
        semantic_focus_signals=("distinct_null_topk", "boolean_aggregation", "multi_key_membership_aggregation"),
        discovery_biases=(
            _bias(
                targets=["distinct_null_topk", "boolean_aggregation", "multi_key_membership_aggregation"],
                feature_prefixes=semantic_signal_bias_prefixes("common_api_template:", "materialization:"),
                score_bonus=0.40,
                novelty_bonus=0.20,
                contribution_bonus=0.15,
                keep_in_pool=True,
            ),
        ),
    ),
    DiscoveryLaneSpec(
        lane_id="embedded_sql",
        target_suite="embedded_sql_cross",
        preset="live_embedded_sql_deep_organic_metamorphic",
        theme="Organic DuckDB/SQLite lightweight SQL semantics compared with DataFrame references",
        default=True,
        semantic_focus_families=("join_membership", "stateful_ordering"),
        semantic_focus_signals=("row_value_absence_filter", "groupby_sorted_input"),
        discovery_biases=(
            _bias(
                targets=["row_value_absence_filter", "groupby_sorted_input"],
                feature_prefixes=["join:", "group_key_type:", "sort:"],
                score_bonus=0.30,
                novelty_bonus=0.10,
                contribution_bonus=0.15,
            ),
        ),
    ),
    DiscoveryLaneSpec(
        lane_id="duckdb_storage",
        target_suite="duckdb_storage_cross",
        preset="live_embedded_sql_deep_organic_metamorphic",
        theme="DuckDB persistent database execution path versus pandas references, including storage-backed joins, aggregation, sort, and projection",
        default=True,
        semantic_focus_families=("stateful_ordering", "join_membership", "materialization_boundary"),
        semantic_focus_signals=("storage_offset", "join_null_sort"),
        discovery_biases=(
            _bias(
                targets=["storage_offset", "join_null_sort"],
                feature_prefixes=["sort:", "join:", "materialization:"],
                score_bonus=0.30,
                novelty_bonus=0.10,
                contribution_bonus=0.15,
            ),
        ),
    ),
    DiscoveryLaneSpec(
        lane_id="cross_family",
        target_suite="latest_no_datafusion",
        preset="live_deep_organic_metamorphic",
        theme="Organic cross-family Python API, Arrow, DataFrame, and embedded SQL semantics without DataFusion saturation",
        default=True,
        semantic_focus_families=("stateful_ordering", "join_membership", "string_semantics"),
        semantic_focus_signals=("groupby_sorted_input", "running_sum_precision", "normalized_string_membership"),
        discovery_biases=(
            _bias(
                targets=["groupby_sorted_input", "running_sum_precision", "normalized_string_membership"],
                feature_prefixes=["pattern:", "expr:", "join:"],
                score_bonus=0.35,
                novelty_bonus=0.15,
                contribution_bonus=0.20,
                keep_in_pool=True,
            ),
        ),
    ),
    DiscoveryLaneSpec(
        lane_id="common_api_workflow",
        target_suite="latest_no_datafusion",
        preset="live_common_api_workflow_metamorphic",
        theme="Organic low-complexity daily DataFrame/Arrow/embedded SQL workflows: filter, string clean/replace/slice/concat/pattern flags, ISO date-part extraction, int/float/string casts, nullable boolean not/reductions, drop-null, mutate, union-all, semi/anti join, fill-null/coalesce, case-when, distinct, join, groupby, sort, and slice",
        default=True,
        semantic_focus_families=("materialization_boundary", "string_semantics", "join_membership", "set_semantics"),
        semantic_focus_signals=(
            "common_api_workflow",
            "filter_input_materialization",
            "distinct_input_materialization",
            "case_when_membership",
            "string_pattern_case_when",
        ),
        discovery_biases=(
            _bias(
                targets=[
                    "common_api_workflow",
                    "filter_input_materialization",
                    "distinct_input_materialization",
                    "case_when_membership",
                    "string_pattern_case_when",
                ],
                feature_prefixes=semantic_signal_bias_prefixes(
                    "common_api_template:",
                    "materialization:",
                    "expr:string_",
                ),
                score_bonus=0.50,
                novelty_bonus=0.25,
                contribution_bonus=0.20,
                candidate_pool_bonus=0.10,
                keep_in_pool=True,
            ),
        ),
    ),
    DiscoveryLaneSpec(
        lane_id="arrow_probe_stress",
        target_suite="arrow_cross",
        preset="live_arrow_issue_focus_metamorphic",
        theme="Arrow probe stress lane; issue-inspired sources remain separated from fresh counts",
        discovery_biases=(),
    ),
    DiscoveryLaneSpec(
        lane_id="polars_probe_stress",
        target_suite="polars_cross",
        preset="live_polars_issue_focus_metamorphic",
        theme="Polars probe stress lane; issue-inspired sources remain separated from fresh counts",
        discovery_biases=(),
    ),
    DiscoveryLaneSpec(
        lane_id="duckdb_probe_stress",
        target_suite="embedded_sql_cross",
        preset="live_duckdb_issue_focus_metamorphic",
        theme="DuckDB/SQLite probe stress lane; issue-inspired sources remain separated from fresh counts",
        discovery_biases=(),
    ),
    DiscoveryLaneSpec(
        lane_id="deep_probe_rotation",
        target_suite="latest_all_engines",
        preset="live_deep_probe_rotation_metamorphic",
        theme="Deterministic deep probe rotation for audit/seed calibration; known probe families remain separated from fresh discovery counts",
        default=False,
        discovery_biases=(),
    ),
)


DISCOVERY_LANES_BY_ID: dict[str, DiscoveryLaneSpec] = {
    spec.lane_id: spec for spec in DISCOVERY_LANE_SPECS
}


DEFAULT_DISCOVERY_LANE_IDS: tuple[str, ...] = tuple(
    spec.lane_id for spec in DISCOVERY_LANE_SPECS if spec.default
)


def discovery_lane_spec(lane_id: str) -> DiscoveryLaneSpec:
    if lane_id not in DISCOVERY_LANES_BY_ID:
        raise ValueError(f"unknown bug-sprint lane id: {lane_id}")
    return DISCOVERY_LANES_BY_ID[lane_id]


def discovery_lane_catalog() -> dict[str, dict[str, Any]]:
    return {
        lane_id: spec.to_dict()
        for lane_id, spec in sorted(DISCOVERY_LANES_BY_ID.items())
    }


def discovery_biases_for_lanes(*lane_ids: str) -> list[DiscoveryBias]:
    return merge_discovery_biases(
        *(
            discovery_lane_spec(lane_id).discovery_biases
            for lane_id in lane_ids
            if lane_id in DISCOVERY_LANES_BY_ID
        )
    )


BugSprintLaneSpec = DiscoveryLaneSpec
BUG_SPRINT_LANE_SPECS = DISCOVERY_LANE_SPECS
BUG_SPRINT_LANES_BY_ID = DISCOVERY_LANES_BY_ID
DEFAULT_BUG_SPRINT_LANE_IDS = DEFAULT_DISCOVERY_LANE_IDS


def bug_sprint_lane_spec(lane_id: str) -> DiscoveryLaneSpec:
    return discovery_lane_spec(lane_id)


def bug_sprint_lane_catalog() -> dict[str, dict[str, Any]]:
    return discovery_lane_catalog()
