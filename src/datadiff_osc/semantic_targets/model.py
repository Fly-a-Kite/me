from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from datadiff_osc._canonical import frozen_pairs, stable_digest, to_primitive
from datadiff_osc.schemas import SeedLineage, TargetFingerprint


TARGET_SCHEMA_VERSION = "osc-semantic-target-v1"


class ProvenanceClass(str, Enum):
    FRESH_DISCOVERY = "fresh_discovery"
    KNOWN_REGRESSION = "known_regression"
    SENSITIVITY_CONTROL = "sensitivity_control"


class ContrastRelation(str, Enum):
    METAMORPHIC_EQUIVALENCE = "metamorphic_equivalence"
    DIFFERENTIAL_ISOLATION = "differential_isolation"
    MONOTONIC_BOUNDARY = "monotonic_boundary"


@dataclass(frozen=True, slots=True)
class AxisSpec:
    name: str
    values: tuple[str, ...]
    baseline: str
    relation: ContrastRelation = ContrastRelation.DIFFERENTIAL_ISOLATION

    def __post_init__(self) -> None:
        if not self.name or not self.values:
            raise ValueError("target axis name and values must be non-empty")
        if len(set(self.values)) != len(self.values):
            raise ValueError(f"target axis {self.name} values must be unique")
        if self.baseline not in self.values:
            raise ValueError(f"target axis {self.name} baseline is not a value")


@dataclass(frozen=True, slots=True)
class TargetTemplate:
    template_id: str
    test_family_id: str
    provenance_class: ProvenanceClass
    semantic_dimensions: tuple[str, ...]
    axes: tuple[AxisSpec, ...]
    required_all_atoms: frozenset[str]
    required_any_atom_groups: tuple[frozenset[str], ...]
    forbidden_atoms: frozenset[str]
    target_backend: str
    control_backends: tuple[str, ...]
    required_capabilities: frozenset[str]
    risk_class: str
    priority: int
    cost_class: str
    constructor_provider: str
    oracle_relation: str
    provenance_evidence_id: str = ""
    schema_version: str = TARGET_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.template_id or not self.test_family_id:
            raise ValueError("target template identity must be non-empty")
        if not self.axes or not self.control_backends:
            raise ValueError("target template needs axes and controls")
        if self.target_backend in self.control_backends:
            raise ValueError("target backend cannot be a control")
        if len({axis.name for axis in self.axes}) != len(self.axes):
            raise ValueError("target template axis names must be unique")
        if self.priority < 1:
            raise ValueError("target priority must be positive")
        if (
            self.provenance_class == ProvenanceClass.FRESH_DISCOVERY
            and any(token in self.provenance_evidence_id.lower() for token in ("confirmed-root", "confirmed_root", "issue-replay", "known-root"))
        ):
            raise ValueError("fresh target template contains regression provenance")

    @property
    def cell_count(self) -> int:
        count = 1
        for axis in self.axes:
            count *= len(axis.values)
        return count

    @property
    def digest(self) -> str:
        return stable_digest("osc-target-template", self)

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(self)


@dataclass(frozen=True, slots=True)
class TargetCell:
    target_cell_id: str
    template_id: str
    test_family_id: str
    provenance_class: ProvenanceClass
    coordinates: tuple[tuple[str, str], ...]
    required_all_atoms: frozenset[str]
    required_any_atom_groups: tuple[frozenset[str], ...]
    forbidden_atoms: frozenset[str]
    target_backend: str
    control_backends: tuple[str, ...]
    required_capabilities: frozenset[str]
    risk_class: str
    cost_class: str
    construction_contract: str
    activation_contract: str
    observation_contract: str
    construction_index: int
    schema_version: str = TARGET_SCHEMA_VERSION

    @property
    def coordinate_map(self) -> dict[str, str]:
        return dict(self.coordinates)

    @property
    def digest(self) -> str:
        return stable_digest("osc-target-cell", self)

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(self)


@dataclass(frozen=True, slots=True)
class BackendPairObligation:
    obligation_id: str
    target_cell_id: str
    provenance_class: ProvenanceClass
    target_backend: str
    control_backend: str
    observation_contract: str
    schema_version: str = "osc-backend-pair-obligation-v1"

    @property
    def digest(self) -> str:
        return stable_digest("osc-backend-pair-obligation", self)


@dataclass(frozen=True, slots=True)
class ContrastEdge:
    contrast_edge_id: str
    base_cell_id: str
    sibling_cell_id: str
    changed_axis: str
    base_value: str
    sibling_value: str
    relation: ContrastRelation
    activation_contract: str
    observation_contract: str
    schema_version: str = "osc-contrast-edge-v1"

    @property
    def digest(self) -> str:
        return stable_digest("osc-contrast-edge", self)


@dataclass(frozen=True, slots=True)
class ContrastSet:
    contrast_set_id: str
    base_cell_id: str
    sibling_cell_ids: tuple[str, ...]
    edge_ids: tuple[str, ...]
    assignment_digest: str
    seed_lineage: SeedLineage
    schema_version: str = "osc-contrast-set-v1"

    @property
    def digest(self) -> str:
        return stable_digest("osc-contrast-set", self)


@dataclass(frozen=True, slots=True)
class InteractionTile:
    tile_id: str
    test_family_id: str
    axis_a: str
    axis_b: str
    base_cell_id: str
    a_only_cell_id: str
    b_only_cell_id: str
    joint_cell_id: str
    non_target_coordinates: tuple[tuple[str, str], ...]
    relation: str = "interaction_tile"
    scheduler_status: str = "shadow"
    schema_version: str = "osc-interaction-tile-v1"

    def __post_init__(self) -> None:
        endpoints = (
            self.base_cell_id,
            self.a_only_cell_id,
            self.b_only_cell_id,
            self.joint_cell_id,
        )
        if len(set(endpoints)) != 4:
            raise ValueError("interaction tile requires four distinct endpoints")
        if self.scheduler_status not in {"shadow", "primary", "deleted"}:
            raise ValueError("invalid interaction tile scheduler status")

    @property
    def digest(self) -> str:
        return stable_digest("osc-interaction-tile", self)


@dataclass(frozen=True, slots=True)
class TargetAssignment:
    selected_cell_ids: tuple[str, ...]
    seed_lineage: SeedLineage
    selected_edge_ids: tuple[str, ...] = ()
    selected_tile_ids: tuple[str, ...] = ()
    parameters: tuple[tuple[str, Any], ...] = ()

    @classmethod
    def build(
        cls,
        *,
        selected_cell_ids: tuple[str, ...],
        seed_lineage: SeedLineage,
        selected_edge_ids: tuple[str, ...] = (),
        selected_tile_ids: tuple[str, ...] = (),
        parameters: dict[str, Any] | None = None,
    ) -> "TargetAssignment":
        return cls(
            selected_cell_ids=tuple(selected_cell_ids),
            seed_lineage=seed_lineage,
            selected_edge_ids=tuple(selected_edge_ids),
            selected_tile_ids=tuple(selected_tile_ids),
            parameters=frozen_pairs(parameters),
        )

    @property
    def digest(self) -> str:
        return stable_digest("osc-target-assignment", self)


@dataclass(frozen=True, slots=True)
class ActivationCertificate:
    target_fingerprint: TargetFingerprint
    assignment_digest: str
    selected_cell_ids: tuple[str, ...]
    activated_cell_ids: tuple[str, ...]
    required_atoms: tuple[str, ...]
    observed_atoms: tuple[str, ...]
    missing_atoms: tuple[str, ...]
    forbidden_atoms: tuple[str, ...]
    static_facts: tuple[str, ...]
    semantic_atom_digests: tuple[str, ...]
    preflight_valid: bool
    mutation_preserved: bool
    degraded_reasons: tuple[str, ...] = ()
    schema_version: str = "osc-activation-certificate-v1"

    @property
    def valid(self) -> bool:
        return (
            self.preflight_valid
            and self.mutation_preserved
            and bool(self.assignment_digest)
            and bool(self.semantic_atom_digests)
            and not self.missing_atoms
            and not self.forbidden_atoms
            and set(self.selected_cell_ids) <= set(self.activated_cell_ids)
        )

    @property
    def digest(self) -> str:
        return stable_digest("osc-activation-certificate", self)


@dataclass(frozen=True, slots=True)
class CompiledTargetUniverse:
    fresh_cells: tuple[TargetCell, ...]
    regression_cells: tuple[TargetCell, ...]
    fresh_edges: tuple[ContrastEdge, ...]
    fresh_backend_pair_obligations: tuple[BackendPairObligation, ...]
    regression_backend_pair_obligations: tuple[BackendPairObligation, ...]
    shadow_tiles: tuple[InteractionTile, ...]
    taxonomy_digest: str
    template_digest: str
    schema_version: str = "osc-compiled-target-universe-v1"

    def __post_init__(self) -> None:
        if any(
            item.provenance_class != ProvenanceClass.FRESH_DISCOVERY
            for item in self.fresh_backend_pair_obligations
        ):
            raise ValueError("fresh obligation collection contains non-fresh evidence")
        if any(
            item.provenance_class != ProvenanceClass.KNOWN_REGRESSION
            for item in self.regression_backend_pair_obligations
        ):
            raise ValueError("regression obligation collection contains non-regression evidence")
        fresh_ids = {item.obligation_id for item in self.fresh_backend_pair_obligations}
        regression_ids = {
            item.obligation_id for item in self.regression_backend_pair_obligations
        }
        if fresh_ids & regression_ids:
            raise ValueError("fresh and regression obligation identities overlap")

    @property
    def digest(self) -> str:
        return stable_digest("osc-compiled-target-universe", self)

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(self)
