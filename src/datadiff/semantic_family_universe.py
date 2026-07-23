"""Authoritative semantic-family universe for the global-v2 witness portfolio.

The confirmed-root corpus is evidence about bugs already found; it is not a
complete definition of the testing surface.  This module therefore freezes an
explicit backend-by-mechanism universe for ``latest_all_engines``.  Every
required family has one native target, one or more deterministic controls, and
declared capability/dimension obligations.  Coverage audits consume this
registry and fail closed when a required family is absent.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Literal

from datadiff.targets import resolve_target_backends


SEMANTIC_FAMILY_UNIVERSE_SCHEMA_VERSION = "semantic-family-universe-v1"
SEMANTIC_FAMILY_PAIR_SCHEMA_VERSION = "semantic-family-backend-pair-v1"
SEMANTIC_FAMILY_UNIVERSE_ID = "family_universe_v1"
SEMANTIC_FAMILY_UNIVERSE_TARGET_SUITE = "latest_all_engines"
SEMANTIC_FAMILY_GLOBAL_V2_GENERATION_MODE = "goal_first_witness_global_v2"
SEMANTIC_FAMILY_GLOBAL_V2_METHOD_ARM_ID = "p8_semantic_witness_global_v2"

FamilySource = Literal["confirmed_root", "coverage_expansion"]
BackendRole = Literal["target", "control", "out_of_scope"]


@dataclass(frozen=True, slots=True)
class SemanticFamilyDefinition:
    family_id: str
    mechanism_id: str
    target_backend: str
    control_backends: tuple[str, ...]
    required_capabilities: tuple[str, ...]
    semantic_dimensions: tuple[str, ...]
    source: FamilySource
    root_id: str
    minimum_cell_count: int
    priority: int
    rationale: str

    def __post_init__(self) -> None:
        if not self.family_id or not self.mechanism_id or not self.target_backend:
            raise ValueError("semantic family identity fields must be non-empty")
        if not self.control_backends:
            raise ValueError("semantic family must declare at least one control backend")
        if self.target_backend in self.control_backends:
            raise ValueError("semantic family target cannot also be a control")
        if len(set(self.control_backends)) != len(self.control_backends):
            raise ValueError("semantic family control backends must be unique")
        if not self.required_capabilities:
            raise ValueError("semantic family must declare required capabilities")
        if not self.semantic_dimensions:
            raise ValueError("semantic family must declare semantic dimensions")
        if self.source not in {"confirmed_root", "coverage_expansion"}:
            raise ValueError(f"unsupported semantic family source: {self.source}")
        if not self.root_id:
            raise ValueError("semantic family must declare a root or target identity")
        if self.source == "coverage_expansion" and not self.root_id.startswith(
            "target-family-"
        ):
            raise ValueError(
                "coverage-expansion identities must use the target-family- prefix"
            )
        if self.minimum_cell_count < 1:
            raise ValueError("semantic family minimum cell count must be positive")
        if self.priority < 1:
            raise ValueError("semantic family priority must be positive")

    @property
    def execution_backends(self) -> tuple[str, ...]:
        return (self.target_backend, *self.control_backends)

    def backend_role(self, backend: str) -> BackendRole:
        if backend == self.target_backend:
            return "target"
        if backend in self.control_backends:
            return "control"
        return "out_of_scope"

    def manifest(self) -> dict[str, Any]:
        return {
            "family_id": self.family_id,
            "mechanism_id": self.mechanism_id,
            "target_backend": self.target_backend,
            "control_backends": list(self.control_backends),
            "execution_backends": list(self.execution_backends),
            "required_capabilities": list(self.required_capabilities),
            "semantic_dimensions": list(self.semantic_dimensions),
            "source": self.source,
            "root_id": self.root_id,
            "minimum_cell_count": self.minimum_cell_count,
            "priority": self.priority,
            "rationale": self.rationale,
        }


_CONFIRMED_ROOT_FAMILIES: tuple[SemanticFamilyDefinition, ...] = (
    SemanticFamilyDefinition(
        family_id="pyarrow_sliced_bool_groupby_any_all",
        mechanism_id="physical_layout_boolean_aggregate",
        target_backend="pyarrow",
        control_backends=("pandas",),
        required_capabilities=("op:groupby", "agg:any", "agg:all", "type:bool", "nulls"),
        semantic_dimensions=("physical_layout", "null_semantics", "operation"),
        source="confirmed_root",
        root_id="pyarrow-sliced-bool-hash-aggregate-001",
        minimum_cell_count=18,
        priority=1,
        rationale="Boolean validity/value bitmap offsets across bounded Arrow layouts.",
    ),
    SemanticFamilyDefinition(
        family_id="polars_reflected_arithmetic_operand_order",
        mechanism_id="reflected_arithmetic_operand_order",
        target_backend="polars",
        control_backends=("polars_lazy",),
        required_capabilities=("op:series_reflected_arithmetic_probe", "type:float"),
        semantic_dimensions=("operation", "execution_mode"),
        source="confirmed_root",
        root_id="polars-reflected-arithmetic-operand-order-001",
        minimum_cell_count=30,
        priority=1,
        rationale="Non-commutative reflected Series operators and name dispatch.",
    ),
    SemanticFamilyDefinition(
        family_id="datafusion_grouped_null_topk",
        mechanism_id="grouped_null_topk",
        target_backend="datafusion",
        control_backends=("pandas", "duckdb"),
        required_capabilities=("op:datafusion_grouped_null_topk_probe", "nulls"),
        semantic_dimensions=("null_semantics", "order_semantics", "plan_kind"),
        source="confirmed_root",
        root_id="datafusion-grouped-null-topk-001",
        minimum_cell_count=12,
        priority=1,
        rationale="All-null grouped aggregates remain observable through ordered TopK.",
    ),
    SemanticFamilyDefinition(
        family_id="datafusion_limit_offset_pushdown",
        mechanism_id="limit_offset_pushdown",
        target_backend="datafusion",
        control_backends=("pandas", "duckdb"),
        required_capabilities=("op:confirmed_root_witness_probe", "op:limit", "op:offset"),
        semantic_dimensions=("order_semantics", "plan_kind", "cross_operation"),
        source="confirmed_root",
        root_id="datafusion-limit-offset-pushdown-001",
        minimum_cell_count=12,
        priority=1,
        rationale="Outer OFFSET must not leak into an inner ordered LIMIT.",
    ),
    SemanticFamilyDefinition(
        family_id="datafusion_negative_zero_comparison",
        mechanism_id="negative_zero_comparison",
        target_backend="datafusion",
        control_backends=("pandas", "duckdb"),
        required_capabilities=("op:confirmed_root_witness_probe", "type:float"),
        semantic_dimensions=("logical_type", "operation", "plan_kind"),
        source="confirmed_root",
        root_id="datafusion-negative-zero-comparison-001",
        minimum_cell_count=12,
        priority=1,
        rationale="SQL equality/order semantics at the signed-zero boundary.",
    ),
    SemanticFamilyDefinition(
        family_id="datafusion_distinct_null_topk",
        mechanism_id="distinct_null_topk",
        target_backend="datafusion",
        control_backends=("pandas", "duckdb"),
        required_capabilities=("op:confirmed_root_witness_probe", "op:distinct", "nulls"),
        semantic_dimensions=("null_semantics", "order_semantics", "plan_kind"),
        source="confirmed_root",
        root_id="datafusion-distinct-null-topk-001",
        minimum_cell_count=12,
        priority=1,
        rationale="Distinct TopK retains the NULL heap entry under bounded projections.",
    ),
    SemanticFamilyDefinition(
        family_id="datafusion_ordered_limit_idempotence",
        mechanism_id="ordered_limit_idempotence",
        target_backend="datafusion",
        control_backends=("pandas", "duckdb"),
        required_capabilities=("op:confirmed_root_witness_probe", "op:sort", "op:limit"),
        semantic_dimensions=("order_semantics", "plan_kind", "cross_operation"),
        source="confirmed_root",
        root_id="datafusion-ordered-limit-idempotence-001",
        minimum_cell_count=12,
        priority=1,
        rationale="Repeated identical ordered cuts must remain idempotent.",
    ),
    SemanticFamilyDefinition(
        family_id="polars_grouped_max_sort_metadata",
        mechanism_id="grouped_aggregate_sort_metadata",
        target_backend="polars",
        control_backends=("polars_lazy",),
        required_capabilities=("op:confirmed_root_witness_probe", "op:groupby", "agg:max"),
        semantic_dimensions=("order_semantics", "execution_mode", "cross_operation"),
        source="confirmed_root",
        root_id="polars-grouped-max-sort-metadata-001",
        minimum_cell_count=12,
        priority=1,
        rationale="Grouped aggregation must invalidate stale sortedness metadata.",
    ),
    SemanticFamilyDefinition(
        family_id="duckdb_join_filter_pushdown_limit",
        mechanism_id="join_filter_pushdown_limit",
        target_backend="duckdb",
        control_backends=("pandas", "sqlite"),
        required_capabilities=("op:confirmed_root_witness_probe", "op:join", "op:limit"),
        semantic_dimensions=("plan_kind", "order_semantics", "cross_operation"),
        source="confirmed_root",
        root_id="duckdb-join-filter-pushdown-limit-001",
        minimum_cell_count=24,
        priority=1,
        rationale="Runtime join filters must not cross an ordered cut boundary.",
    ),
)


_EXPANSION_FAMILIES: tuple[SemanticFamilyDefinition, ...] = (
    SemanticFamilyDefinition(
        family_id="pandas_nullable_bool_reduction",
        mechanism_id="nullable_extension_groupby_reduction",
        target_backend="pandas",
        control_backends=("polars",),
        required_capabilities=("op:bool_reduction_skipna_probe", "type:bool", "nulls"),
        semantic_dimensions=("logical_type", "null_semantics", "cross_operation"),
        source="coverage_expansion",
        root_id="target-family-pandas-nullable-bool-reduction-v1",
        minimum_cell_count=12,
        priority=2,
        rationale="Give Pandas a native nullable-extension reduction target rather than only a control role.",
    ),
    SemanticFamilyDefinition(
        family_id="sqlite_affinity_null_ordered_cut",
        mechanism_id="type_affinity_null_collation_limit",
        target_backend="sqlite",
        control_backends=("pandas", "duckdb"),
        required_capabilities=("expr:cast", "op:sort", "op:limit", "op:offset", "nulls"),
        semantic_dimensions=("logical_type", "null_semantics", "order_semantics"),
        source="coverage_expansion",
        root_id="target-family-sqlite-affinity-null-ordered-cut-v1",
        minimum_cell_count=12,
        priority=2,
        rationale="Exercise SQLite affinity and NULL ordering through an active LIMIT/OFFSET cut.",
    ),
    SemanticFamilyDefinition(
        family_id="polars_lazy_filter_groupby_window",
        mechanism_id="lazy_optimizer_filter_groupby_window",
        target_backend="polars_lazy",
        control_backends=("polars",),
        required_capabilities=("op:filter", "op:groupby", "op:running_sum", "op:sort"),
        semantic_dimensions=("plan_kind", "execution_mode", "cross_operation"),
        source="coverage_expansion",
        root_id="target-family-polars-lazy-filter-groupby-window-v1",
        minimum_cell_count=12,
        priority=2,
        rationale="Make lazy optimizer behavior a native target with eager Polars as control.",
    ),
    SemanticFamilyDefinition(
        family_id="polars_lazy_temporal_cast_boundary",
        mechanism_id="decimal_temporal_cast_boundary",
        target_backend="polars_lazy",
        control_backends=("polars",),
        required_capabilities=("op:timestamp_precision_filter_probe", "expr:cast"),
        semantic_dimensions=("logical_type", "execution_mode", "operation"),
        source="coverage_expansion",
        root_id="target-family-polars-lazy-temporal-cast-boundary-v1",
        minimum_cell_count=12,
        priority=2,
        rationale="Bound temporal precision and cast-direction behavior at unit boundaries.",
    ),
    SemanticFamilyDefinition(
        family_id="pyarrow_encoded_nested_compute",
        mechanism_id="dictionary_chunked_nested_encoding",
        target_backend="pyarrow",
        control_backends=("pandas",),
        required_capabilities=(
            "op:run_end_null_compute_probe",
            "op:list_flatten_parent_indices_probe",
            "op:large_string_partition_probe",
        ),
        semantic_dimensions=("physical_layout", "logical_type", "operation"),
        source="coverage_expansion",
        root_id="target-family-pyarrow-encoded-nested-compute-v1",
        minimum_cell_count=12,
        priority=2,
        rationale="Cover run-end, list/nested, large-string, dictionary, and chunked execution paths.",
    ),
    SemanticFamilyDefinition(
        family_id="datafusion_window_order_join_interaction",
        mechanism_id="window_order_join_interaction",
        target_backend="datafusion",
        control_backends=("pandas", "duckdb"),
        required_capabilities=("op:join", "op:row_number_filter", "op:running_sum", "op:sort"),
        semantic_dimensions=("plan_kind", "order_semantics", "cross_operation"),
        source="coverage_expansion",
        root_id="target-family-datafusion-window-order-join-interaction-v1",
        minimum_cell_count=12,
        priority=2,
        rationale="Compose join, ordered window observation, and an active cardinality boundary.",
    ),
)


_FAMILIES = (*_CONFIRMED_ROOT_FAMILIES, *_EXPANSION_FAMILIES)


def semantic_family_definitions() -> tuple[SemanticFamilyDefinition, ...]:
    return _FAMILIES


def confirmed_root_family_definitions() -> tuple[SemanticFamilyDefinition, ...]:
    return _CONFIRMED_ROOT_FAMILIES


def expansion_family_definitions() -> tuple[SemanticFamilyDefinition, ...]:
    return _EXPANSION_FAMILIES


def semantic_family_definition(family_id: str) -> SemanticFamilyDefinition:
    matches = [item for item in _FAMILIES if item.family_id == str(family_id)]
    if len(matches) != 1:
        raise KeyError(family_id)
    return matches[0]


def semantic_family_universe_manifest() -> dict[str, Any]:
    backends = tuple(
        resolve_target_backends(target_suite=SEMANTIC_FAMILY_UNIVERSE_TARGET_SUITE)
    )
    families = [item.manifest() for item in _FAMILIES]
    backend_pairs = [
        {
            "schema_version": SEMANTIC_FAMILY_PAIR_SCHEMA_VERSION,
            "family_id": family.family_id,
            "mechanism_id": family.mechanism_id,
            "backend": backend,
            "role": family.backend_role(backend),
            "required": family.backend_role(backend) != "out_of_scope",
        }
        for family in _FAMILIES
        for backend in backends
    ]
    payload: dict[str, Any] = {
        "schema_version": SEMANTIC_FAMILY_UNIVERSE_SCHEMA_VERSION,
        "universe_id": SEMANTIC_FAMILY_UNIVERSE_ID,
        "target_suite": SEMANTIC_FAMILY_UNIVERSE_TARGET_SUITE,
        "backends": list(backends),
        "global_v2_generation_mode": SEMANTIC_FAMILY_GLOBAL_V2_GENERATION_MODE,
        "global_v2_method_arm_id": SEMANTIC_FAMILY_GLOBAL_V2_METHOD_ARM_ID,
        "families": families,
        "backend_pairs": backend_pairs,
        "summary": {
            "backend_count": len(backends),
            "family_count": len(_FAMILIES),
            "confirmed_root_family_count": len(_CONFIRMED_ROOT_FAMILIES),
            "coverage_expansion_family_count": len(_EXPANSION_FAMILIES),
            "backend_pair_count": len(backend_pairs),
            "required_backend_pair_count": sum(
                bool(item["required"]) for item in backend_pairs
            ),
        },
    }
    payload["digest"] = _digest(payload)
    return payload


def _digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "semantic-family-universe-" + hashlib.sha256(encoded).hexdigest()
