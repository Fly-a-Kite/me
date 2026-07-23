"""Private declaration-derived Phase-6 lane and focus registry.

This module deliberately does not alter the legacy discovery defaults.  It
projects the reviewed formal selector through ``DISCOVERY_LANES_BY_ID`` and
binds every complete executable declaration by digest.  The resulting values
are inputs for later Root context verification; they are not gate authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from types import MappingProxyType
from typing import Any, Iterable

from datadiff.strategy_registry import (
    DEFAULT_DISCOVERY_LANE_IDS,
    DISCOVERY_LANES_BY_ID,
    DiscoveryLaneSpec,
)

from datadiff_osc._canonical import stable_digest, to_primitive


FORMAL_LANE_REGISTRY_SCHEMA_VERSION = "osc-private-formal-lane-registry-v1"
FORMAL_LANE_DECLARATION_SCHEMA_VERSION = (
    "osc-private-formal-lane-declaration-v1"
)
FORMAL_FOCUS_OBLIGATION_SCHEMA_VERSION = (
    "osc-private-formal-focus-obligation-v1"
)
FORMAL_FOCUS_RULE_SCHEMA_VERSION = "osc-private-formal-focus-rule-v1"

# This selector is separate from, and must never replace, the legacy defaults.
FORMAL_PHASE6_DISCOVERY_LANE_IDS: tuple[str, ...] = (
    "pandas_targeted_boundaries",
    "polars_targeted_boundaries",
    "datafusion_targeted_boundaries",
    "chdb_targeted_boundaries",
    "orthogonal_stress",
    "arrow_layout",
    "polars_streaming",
    "embedded_sql",
    "duckdb_storage",
    "common_api_workflow",
    "cross_family",
)

EXPECTED_LEGACY_DEFAULT_DISCOVERY_LANE_IDS: tuple[str, ...] = (
    "arrow_layout",
    "polars_lazy",
    "polars_streaming",
    "datafusion_optimizer",
    "datafusion_common_api",
    "embedded_sql",
    "duckdb_storage",
    "cross_family",
    "common_api_workflow",
)


# These are executable source rules, not caller-provided generator metadata.
# Each rule deterministically rebuilds a typed Case and its metadata-independent
# extraction from the formal case SeedLineage.  The six signals that are not
# direct generator profile names are bound to reviewed semantic source cases.
_FOCUS_SOURCE_GENERATORS = MappingProxyType({
    "bitmap_boundary_bool_aggregate": "bitmap_boundary_bool_aggregate",
    "case_when_membership": "case_when_join_key_membership",
    "common_api_workflow": "common_api_workflow",
    "distinct_input_materialization": "vector_boundary_groupby_distinct",
    "filter_input_materialization": "empty_filter_groupby",
    "groupby_sorted_input": "ordered_groupby_sort",
    "join_null_sort": "join_null_sort",
    "normalized_string_membership": (
        "chained_string_cleanup_membership_window"
    ),
    "partitioned_running_sum": "partitioned_running_sum",
    "pyarrow_hash_pivot_wider_order_semantics": (
        "pyarrow_hash_pivot_wider_order_semantics"
    ),
    "pyarrow_list_flatten_parent_indices_semantics": (
        "pyarrow_list_flatten_parent_indices_semantics"
    ),
    "row_value_absence_filter": "row_value_absence_filter",
    "running_sum_precision": "running_sum_precision",
    "skewed_join_multiplicity": "skewed_join_multiplicity",
    "storage_offset": "storage_offset",
    "string_pattern_case_when": "string_empty_pattern_membership_distinct",
    "unique_order_window_tiebreak": "unique_order_window_tiebreak",
    "utf8_slice_length_groupby": "utf8_slice_length_groupby",
    "vector_boundary_groupby_distinct": "vector_boundary_groupby_distinct",
    "wide_schema_projection_boundary": "wide_schema_projection_boundary",
})

# Each rule also carries one feature that must be recomputed from the exact
# generated Case.  The feature is deliberately independent of a caller's
# claimed focus label.  Some legacy focus names are broader than the modern
# structural feature vocabulary, so those names bind a reviewed structural
# witness rather than a same-spelled metadata string.
_FOCUS_REQUIRED_FEATURES = MappingProxyType({
    "bitmap_boundary_bool_aggregate": (
        "semantic_signal:nullable_boolean_reduction"
    ),
    "case_when_membership": (
        "semantic_signal:case_when_membership_predicate"
    ),
    "common_api_workflow": "pattern:common_api_workflow",
    "distinct_input_materialization": "pattern:distinct_input_materialization",
    "filter_input_materialization": "pattern:filter_input_materialization",
    "groupby_sorted_input": "pattern:groupby_sorted_input",
    "join_null_sort": "pattern:join_null_sort",
    "normalized_string_membership": (
        "semantic_signal:normalized_string_membership_key"
    ),
    "partitioned_running_sum": "semantic_signal:partitioned_running_sum",
    "pyarrow_hash_pivot_wider_order_semantics": (
        "semantic_signal:pyarrow_hash_pivot_wider_order_semantics"
    ),
    "pyarrow_list_flatten_parent_indices_semantics": (
        "semantic_signal:pyarrow_list_flatten_parent_indices_semantics"
    ),
    "row_value_absence_filter": "pattern:tuple_absence_nullable_row_value",
    "running_sum_precision": "semantic_signal:running_sum_precision",
    "skewed_join_multiplicity": "semantic_signal:join_cardinality",
    "storage_offset": "semantic_signal:topk_ordering",
    "string_pattern_case_when": "semantic_signal:string_pattern_case_when",
    "unique_order_window_tiebreak": "semantic_signal:keyed_row_pick",
    "utf8_slice_length_groupby": "semantic_signal:string_slice_prefix",
    "vector_boundary_groupby_distinct": (
        "semantic_signal:distinct_duplicate_elimination"
    ),
    "wide_schema_projection_boundary": "semantic_signal:projection_ordering",
})


def executable_lane_declaration_digest(spec: DiscoveryLaneSpec) -> str:
    """Bind every field in the executable legacy declaration."""

    if not isinstance(spec, DiscoveryLaneSpec):
        raise TypeError("lane declaration digest requires DiscoveryLaneSpec")
    return stable_digest(
        "osc-private-executable-lane-declaration-v1",
        spec.to_dict(),
    )


@dataclass(frozen=True, slots=True)
class FormalLaneDeclaration:
    lane_id: str
    target_suite: str
    preset: str
    theme: str
    legacy_default: bool
    semantic_focus_families: tuple[str, ...]
    semantic_focus_signals: tuple[str, ...]
    executable_declaration_digest: str
    schema_version: str = FORMAL_LANE_DECLARATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != FORMAL_LANE_DECLARATION_SCHEMA_VERSION:
            raise ValueError("formal lane declaration schema mismatch")
        if not all((self.lane_id, self.target_suite, self.preset, self.theme)):
            raise ValueError("formal lane declaration identity must be non-empty")
        if not self.semantic_focus_families or not self.semantic_focus_signals:
            raise ValueError("formal lane declaration requires focus declarations")
        for name, values in (
            ("families", self.semantic_focus_families),
            ("signals", self.semantic_focus_signals),
        ):
            if any(not isinstance(item, str) or not item for item in values):
                raise ValueError(f"formal lane focus {name} must be non-empty strings")
            if len(values) != len(set(values)):
                raise ValueError(f"formal lane focus {name} must be unique")
        if not self.executable_declaration_digest:
            raise ValueError("formal lane declaration requires executable digest")

    @property
    def digest(self) -> str:
        return stable_digest("osc-private-formal-lane-declaration", self)

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(self)


@dataclass(frozen=True, slots=True)
class FormalFocusObligation:
    obligation_id: str
    lane_id: str
    signal_id: str
    lane_declaration_digest: str
    schema_version: str = FORMAL_FOCUS_OBLIGATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != FORMAL_FOCUS_OBLIGATION_SCHEMA_VERSION:
            raise ValueError("formal focus obligation schema mismatch")
        if not all(
            (
                self.obligation_id,
                self.lane_id,
                self.signal_id,
                self.lane_declaration_digest,
            )
        ):
            raise ValueError("formal focus obligation fields must be non-empty")
        expected = _focus_obligation_id(
            lane_id=self.lane_id,
            signal_id=self.signal_id,
            lane_declaration_digest=self.lane_declaration_digest,
        )
        if self.obligation_id != expected:
            raise ValueError("formal focus obligation identity mismatch")

    @property
    def digest(self) -> str:
        return stable_digest("osc-private-formal-focus-obligation", self)

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(self)


def _focus_rule_id(
    signal_id: str,
    source_generator: str,
    required_feature: str,
) -> str:
    return stable_digest(
        "osc-private-formal-focus-rule-id-v1",
        {
            "signal_id": signal_id,
            "source_generator": source_generator,
            "required_feature": required_feature,
            "source_kind": "canonical_case_and_atom_extraction",
        },
    )


@dataclass(frozen=True, slots=True)
class FormalFocusRule:
    rule_id: str
    signal_id: str
    source_generator: str
    required_feature: str
    source_kind: str = "canonical_case_and_atom_extraction"
    schema_version: str = FORMAL_FOCUS_RULE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != FORMAL_FOCUS_RULE_SCHEMA_VERSION:
            raise ValueError("formal focus rule schema mismatch")
        if self.signal_id not in _FOCUS_SOURCE_GENERATORS:
            raise ValueError("formal focus rule signal is not frozen")
        expected_generator = _FOCUS_SOURCE_GENERATORS[self.signal_id]
        if self.source_generator != expected_generator:
            raise ValueError("formal focus rule source is caller-substituted")
        expected_feature = _FOCUS_REQUIRED_FEATURES.get(self.signal_id)
        if self.required_feature != expected_feature:
            raise ValueError("formal focus rule feature is caller-substituted")
        if self.source_kind != "canonical_case_and_atom_extraction":
            raise ValueError("formal focus rule source kind is not intrinsic")
        expected_id = _focus_rule_id(
            self.signal_id,
            expected_generator,
            expected_feature,
        )
        if self.rule_id != expected_id:
            raise ValueError("formal focus rule identity mismatch")

    @property
    def digest(self) -> str:
        return stable_digest("osc-private-formal-focus-rule", self)

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(self)


@dataclass(frozen=True, slots=True)
class FormalLaneRegistry:
    lane_declarations: tuple[FormalLaneDeclaration, ...]
    focus_obligations: tuple[FormalFocusObligation, ...]
    unique_focus_signal_ids: tuple[str, ...]
    schema_version: str = FORMAL_LANE_REGISTRY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != FORMAL_LANE_REGISTRY_SCHEMA_VERSION:
            raise ValueError("formal lane registry schema mismatch")
        expected_declarations = _declarations_for_selector(
            FORMAL_PHASE6_DISCOVERY_LANE_IDS
        )
        if self.lane_declarations != expected_declarations:
            raise ValueError("formal lane declaration drift or selector substitution")
        expected_obligations = _obligations_for(expected_declarations)
        if self.focus_obligations != expected_obligations:
            raise ValueError("formal lane focus obligation drift")
        expected_signals = tuple(
            sorted({item.signal_id for item in expected_obligations})
        )
        if self.unique_focus_signal_ids != expected_signals:
            raise ValueError("formal lane unique focus signal drift")
        if len(self.lane_declarations) != 11:
            raise ValueError("formal lane registry must contain exactly 11 lanes")
        if len(self.focus_obligations) != 37:
            raise ValueError(
                "formal lane registry must contain exactly 37 focus obligations"
            )
        if len(self.unique_focus_signal_ids) != 20:
            raise ValueError(
                "formal lane registry must contain exactly 20 unique signals"
            )

    @property
    def lane_ids(self) -> tuple[str, ...]:
        return tuple(item.lane_id for item in self.lane_declarations)

    @property
    def digest(self) -> str:
        return stable_digest("osc-private-formal-lane-registry", self)

    def declaration_for(self, lane_id: str) -> FormalLaneDeclaration:
        for item in self.lane_declarations:
            if item.lane_id == lane_id:
                return item
        raise KeyError(lane_id)

    def obligations_for(self, lane_id: str) -> tuple[FormalFocusObligation, ...]:
        self.declaration_for(lane_id)
        return tuple(item for item in self.focus_obligations if item.lane_id == lane_id)

    def obligation(self, obligation_id: str) -> FormalFocusObligation:
        for item in self.focus_obligations:
            if item.obligation_id == obligation_id:
                return item
        raise KeyError(obligation_id)

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(self)


def _declaration_from_spec(spec: DiscoveryLaneSpec) -> FormalLaneDeclaration:
    return FormalLaneDeclaration(
        lane_id=spec.lane_id,
        target_suite=spec.target_suite,
        preset=spec.preset,
        theme=spec.theme,
        legacy_default=bool(spec.default),
        semantic_focus_families=tuple(spec.semantic_focus_families),
        semantic_focus_signals=tuple(spec.semantic_focus_signals),
        executable_declaration_digest=executable_lane_declaration_digest(spec),
    )


def _declarations_for_selector(
    selector: Iterable[str],
) -> tuple[FormalLaneDeclaration, ...]:
    materialized = tuple(selector)
    if materialized != FORMAL_PHASE6_DISCOVERY_LANE_IDS:
        raise ValueError("formal Phase-6 lane selector is reordered or substituted")
    if len(materialized) != len(set(materialized)):
        raise ValueError("formal Phase-6 lane selector contains duplicates")
    missing = tuple(item for item in materialized if item not in DISCOVERY_LANES_BY_ID)
    if missing:
        raise ValueError(f"formal Phase-6 lanes are undeclared: {missing}")
    return tuple(
        _declaration_from_spec(DISCOVERY_LANES_BY_ID[lane_id])
        for lane_id in materialized
    )


def _focus_obligation_id(
    *,
    lane_id: str,
    signal_id: str,
    lane_declaration_digest: str,
) -> str:
    return stable_digest(
        "osc-private-formal-focus-obligation-id-v1",
        {
            "lane_id": lane_id,
            "signal_id": signal_id,
            "lane_declaration_digest": lane_declaration_digest,
        },
    )


def _obligations_for(
    declarations: tuple[FormalLaneDeclaration, ...],
) -> tuple[FormalFocusObligation, ...]:
    return tuple(
        FormalFocusObligation(
            obligation_id=_focus_obligation_id(
                lane_id=lane.lane_id,
                signal_id=signal_id,
                lane_declaration_digest=lane.digest,
            ),
            lane_id=lane.lane_id,
            signal_id=signal_id,
            lane_declaration_digest=lane.digest,
        )
        for lane in declarations
        for signal_id in lane.semantic_focus_signals
    )


def build_formal_lane_registry(
    selector: Iterable[str] = FORMAL_PHASE6_DISCOVERY_LANE_IDS,
) -> FormalLaneRegistry:
    """Compile the reviewed selector, rejecting legacy/default drift."""

    if tuple(DEFAULT_DISCOVERY_LANE_IDS) != EXPECTED_LEGACY_DEFAULT_DISCOVERY_LANE_IDS:
        raise ValueError("legacy default discovery lanes changed")
    declarations = _declarations_for_selector(selector)
    obligations = _obligations_for(declarations)
    return FormalLaneRegistry(
        lane_declarations=declarations,
        focus_obligations=obligations,
        unique_focus_signal_ids=tuple(
            sorted({item.signal_id for item in obligations})
        ),
    )


@cache
def formal_lane_registry() -> FormalLaneRegistry:
    return build_formal_lane_registry()


@cache
def formal_focus_rules() -> tuple[FormalFocusRule, ...]:
    """Return one frozen executable proof rule for each formal focus signal."""

    signals = formal_lane_registry().unique_focus_signal_ids
    if tuple(sorted(_FOCUS_SOURCE_GENERATORS)) != signals:
        raise ValueError("formal focus source rules do not cover the registry")
    if tuple(sorted(_FOCUS_REQUIRED_FEATURES)) != signals:
        raise ValueError("formal focus feature rules do not cover the registry")
    return tuple(
        FormalFocusRule(
            rule_id=_focus_rule_id(
                signal_id,
                _FOCUS_SOURCE_GENERATORS[signal_id],
                _FOCUS_REQUIRED_FEATURES[signal_id],
            ),
            signal_id=signal_id,
            source_generator=_FOCUS_SOURCE_GENERATORS[signal_id],
            required_feature=_FOCUS_REQUIRED_FEATURES[signal_id],
        )
        for signal_id in signals
    )


def formal_focus_rule(signal_id: str) -> FormalFocusRule:
    for rule in formal_focus_rules():
        if rule.signal_id == signal_id:
            return rule
    raise KeyError(signal_id)
