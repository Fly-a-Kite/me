"""Trusted observation-policy bindings for OSC semantic targets.

``observation_contract`` is a policy identity, not evidence supplied by a run.
This module is the only resolver from that identity to an executable
``HyperContract``.  Resolution is recomputed from the frozen target objects,
real extraction provenance and original runtime endpoints.

The name is intentionally retained from the Phase-4 probe-contract finding,
but the binding also covers ordinary cell, edge and backend-pair policies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from datadiff_osc._canonical import stable_digest
from datadiff_osc.contract_engine.model import (
    Endpoint,
    EndpointRequirement,
    HyperContract,
    RelationObligation,
)
from datadiff_osc.contract_engine.relations import default_relation_registry
from datadiff_osc.generation.extraction import ContrastExtraction, ExtractionResult
from datadiff_osc.semantic_targets.model import (
    BackendPairObligation,
    CompiledTargetUniverse,
    ContrastEdge,
    ContrastRelation,
    TargetCell,
)


OBSERVATION_POLICY_SCHEMA_VERSION = "osc-observation-policy-v1"


class ObservationPolicyError(ValueError):
    """A target policy cannot be resolved without weakening authority."""


@dataclass(frozen=True, slots=True)
class ProbePredicateContract:
    operation_atom: str
    predicate_id: str
    relation_id: str = "witness_false"
    evaluator_identity: str = "osc-relation-evaluator:witness_false:v1"
    schema_version: str = "osc-probe-predicate-contract-v1"

    @property
    def digest(self) -> str:
        return stable_digest("osc-probe-predicate-contract", self)


_PROBE_PREDICATES = tuple(
    ProbePredicateContract(
        operation_atom=f"op:{operation}",
        predicate_id=f"osc-probe-predicate:{operation}:v1",
    )
    for operation in (
        "large_string_partition_probe",
        "list_flatten_parent_indices_probe",
        "polars_timezone_filter_probe",
        "run_end_null_compute_probe",
        "timestamp_precision_filter_probe",
    )
)

_PROBE_BY_ATOM = {item.operation_atom: item for item in _PROBE_PREDICATES}
_DECLARED_RELATION_IDS = ("bag_equal", "sequence_equal", "set_equal_unique")
_RELATION_ALIASES = {
    "bag_equal": "bag_equal",
    "sequence_equal": "sequence_equal",
    "set_equal": "set_equal_unique",
    "set_equal_unique": "set_equal_unique",
}


def probe_predicate_contracts() -> tuple[ProbePredicateContract, ...]:
    return _PROBE_PREDICATES


def _probe_operation_atoms(atoms: Iterable[str]) -> tuple[str, ...]:
    return tuple(
        sorted(
            atom
            for atom in atoms
            if atom.startswith("op:") and atom.removeprefix("op:").endswith("_probe")
        )
    )


def resolve_probe_predicate(
    atoms: Iterable[str],
) -> ProbePredicateContract | None:
    operation_atoms = _probe_operation_atoms(atoms)
    if not operation_atoms:
        return None
    if len(operation_atoms) != 1:
        raise ObservationPolicyError(
            "a cell must bind exactly one probe predicate: "
            + ",".join(operation_atoms)
        )
    try:
        return _PROBE_BY_ATOM[operation_atoms[0]]
    except KeyError as exc:
        raise ObservationPolicyError(
            f"unregistered probe predicate: {operation_atoms[0]}"
        ) from exc


@dataclass(frozen=True, slots=True)
class CellObservationPolicy:
    target_cell_id: str
    target_backend: str
    control_backends: tuple[str, ...]
    declared_relation_id: str
    primary_relation_id: str
    probe_predicate_digest: str = ""
    schema_version: str = OBSERVATION_POLICY_SCHEMA_VERSION

    @property
    def digest(self) -> str:
        return stable_digest("osc-cell-observation-policy", self)


@dataclass(frozen=True, slots=True)
class EdgeObservationPolicy:
    contrast_edge_id: str
    base_cell_id: str
    sibling_cell_id: str
    changed_axis: str
    base_value: str
    sibling_value: str
    relation: str
    base_policy_digest: str
    sibling_policy_digest: str
    schema_version: str = OBSERVATION_POLICY_SCHEMA_VERSION

    @property
    def digest(self) -> str:
        return stable_digest("osc-edge-observation-policy", self)


def compile_cell_observation_policy(
    *,
    target_cell_id: str,
    target_backend: str,
    control_backends: tuple[str, ...],
    required_all_atoms: frozenset[str],
    declared_relation_id: str,
) -> CellObservationPolicy:
    try:
        normalized_relation = _RELATION_ALIASES[declared_relation_id]
    except KeyError as exc:
        raise ObservationPolicyError(
            f"unregistered target observation relation: {declared_relation_id}"
        ) from exc
    predicate = resolve_probe_predicate(required_all_atoms)
    return CellObservationPolicy(
        target_cell_id=target_cell_id,
        target_backend=target_backend,
        control_backends=tuple(control_backends),
        declared_relation_id=normalized_relation,
        primary_relation_id=(
            predicate.relation_id if predicate is not None else normalized_relation
        ),
        probe_predicate_digest=predicate.digest if predicate is not None else "",
    )


def resolve_cell_observation_policy(cell: TargetCell) -> CellObservationPolicy:
    matches = tuple(
        policy
        for relation_id in _DECLARED_RELATION_IDS
        for policy in (
            compile_cell_observation_policy(
                target_cell_id=cell.target_cell_id,
                target_backend=cell.target_backend,
                control_backends=cell.control_backends,
                required_all_atoms=cell.required_all_atoms,
                declared_relation_id=relation_id,
            ),
        )
        if policy.digest == cell.observation_contract
    )
    if len(matches) != 1:
        raise ObservationPolicyError(
            f"cell observation policy is not uniquely resolvable: {cell.target_cell_id}"
        )
    return matches[0]


def compile_edge_observation_policy(
    *,
    contrast_edge_id: str,
    base_cell_id: str,
    sibling_cell_id: str,
    changed_axis: str,
    base_value: str,
    sibling_value: str,
    relation: ContrastRelation,
    base_policy_digest: str,
    sibling_policy_digest: str,
) -> EdgeObservationPolicy:
    return EdgeObservationPolicy(
        contrast_edge_id=contrast_edge_id,
        base_cell_id=base_cell_id,
        sibling_cell_id=sibling_cell_id,
        changed_axis=changed_axis,
        base_value=base_value,
        sibling_value=sibling_value,
        relation=relation.value,
        base_policy_digest=base_policy_digest,
        sibling_policy_digest=sibling_policy_digest,
    )


def resolve_edge_observation_policy(
    edge: ContrastEdge,
    *,
    cells_by_id: dict[str, TargetCell],
) -> EdgeObservationPolicy:
    try:
        base = cells_by_id[edge.base_cell_id]
        sibling = cells_by_id[edge.sibling_cell_id]
    except KeyError as exc:
        raise ObservationPolicyError("edge references an unknown target cell") from exc
    base_policy = resolve_cell_observation_policy(base)
    sibling_policy = resolve_cell_observation_policy(sibling)
    expected = compile_edge_observation_policy(
        contrast_edge_id=edge.contrast_edge_id,
        base_cell_id=edge.base_cell_id,
        sibling_cell_id=edge.sibling_cell_id,
        changed_axis=edge.changed_axis,
        base_value=edge.base_value,
        sibling_value=edge.sibling_value,
        relation=edge.relation,
        base_policy_digest=base_policy.digest,
        sibling_policy_digest=sibling_policy.digest,
    )
    if edge.observation_contract != expected.digest:
        raise ObservationPolicyError(
            f"edge observation policy does not match its direction: {edge.contrast_edge_id}"
        )
    return expected


def _target_maps(universe: CompiledTargetUniverse) -> tuple[
    dict[str, TargetCell],
    dict[str, ContrastEdge],
    dict[str, BackendPairObligation],
]:
    cells = {
        item.target_cell_id: item
        for item in (*universe.fresh_cells, *universe.regression_cells)
    }
    edges = {item.contrast_edge_id: item for item in universe.fresh_edges}
    pairs = {
        item.obligation_id: item
        for item in (
            *universe.fresh_backend_pair_obligations,
            *universe.regression_backend_pair_obligations,
        )
    }
    return cells, edges, pairs


def _case_digests(
    extraction: ExtractionResult | ContrastExtraction,
    required_cell_ids: frozenset[str],
    *,
    edge_ids: tuple[str, ...],
) -> dict[str, str]:
    if isinstance(extraction, ContrastExtraction):
        available = {
            item.target_cell_id: item.extraction.source_digest
            for item in extraction.endpoint_extractions
        }
        if set(available) != set(extraction.endpoint_ids):
            raise ObservationPolicyError("contrast extraction endpoint binding is malformed")
        missing = required_cell_ids - set(available)
        if missing:
            raise ObservationPolicyError(
                "coverage extraction misses target contexts: " + ",".join(sorted(missing))
            )
        return {cell_id: available[cell_id] for cell_id in required_cell_ids}
    if not isinstance(extraction, ExtractionResult):
        raise ObservationPolicyError("coverage authority requires real extraction evidence")
    if edge_ids:
        raise ObservationPolicyError("edge authority requires contrast extraction evidence")
    if len(required_cell_ids) != 1:
        raise ObservationPolicyError(
            "one extraction can bind exactly one credited cell context"
        )
    return {next(iter(required_cell_ids)): extraction.source_digest}


def _endpoint_key(endpoint: Endpoint) -> tuple[str, str]:
    return endpoint.case_digest, endpoint.backend


def _relation_components(relation_id: str) -> tuple[str, ...]:
    if relation_id == "witness_false":
        return ("status", "row_membership")
    if relation_id == "sequence_equal":
        return (
            "status",
            "schema_names",
            "schema_types",
            "nullability",
            "cardinality",
            "row_membership",
            "duplicate_multiplicity",
            "presentation_order",
        )
    if relation_id in {"bag_equal", "set_equal_unique"}:
        return (
            "status",
            "schema_names",
            "schema_types",
            "nullability",
            "cardinality",
            "row_membership",
            "duplicate_multiplicity",
        )
    if relation_id in {"cardinality_nonincreasing", "cardinality_nondecreasing"}:
        return ("cardinality",)
    raise ObservationPolicyError(f"unregistered materialized relation: {relation_id}")


def _obligation(
    *,
    scope: str,
    relation_id: str,
    endpoint_ids: tuple[str, ...],
    components: tuple[str, ...] | None = None,
    parameters: dict[str, object] | None = None,
) -> RelationObligation:
    return RelationObligation.build(
        obligation_id="target-" + stable_digest(
            "osc-target-obligation-id",
            {
                "scope": scope,
                "relation": relation_id,
                "endpoints": endpoint_ids,
                "parameters": parameters or {},
            },
        )[-32:],
        relation_id=relation_id,
        endpoint_ids=endpoint_ids,
        components=components or _relation_components(relation_id),
        parameters=parameters,
        strength=f"target_policy:{relation_id}",
    )


def _equality_obligations(
    *, scope: str, relation_id: str, endpoint_ids: tuple[str, ...]
) -> tuple[RelationObligation, ...]:
    if relation_id == "witness_false":
        return (
            _obligation(
                scope=scope,
                relation_id="witness_false",
                endpoint_ids=endpoint_ids,
            ),
        )
    return (
        _obligation(
            scope=f"{scope}:schema",
            relation_id="schema_equal",
            endpoint_ids=endpoint_ids,
            components=("schema_names", "schema_types", "nullability"),
            parameters={"mode": "full"},
        ),
        _obligation(
            scope=f"{scope}:cardinality",
            relation_id="cardinality_equal",
            endpoint_ids=endpoint_ids,
            components=("cardinality",),
        ),
        _obligation(
            scope=scope,
            relation_id=relation_id,
            endpoint_ids=endpoint_ids,
        ),
    )


def materialize_coverage_hypercontract(
    universe: CompiledTargetUniverse,
    *,
    cell_ids: tuple[str, ...],
    edge_ids: tuple[str, ...],
    backend_pair_obligation_ids: tuple[str, ...],
    extraction: ExtractionResult | ContrastExtraction,
    endpoints: tuple[Endpoint, ...],
) -> HyperContract:
    """Resolve one coverage event to its only admissible HyperContract.

    Cell credit is backed by a pair for that cell.  Edge credit additionally
    requires a pair and a distinct extraction context for each directional
    endpoint.  The runtime endpoints must be exactly the case/backend cross
    product selected by those pair obligations; unrelated or extra endpoints
    fail closed.
    """

    if not endpoints or any(not isinstance(item, Endpoint) for item in endpoints):
        raise ObservationPolicyError("coverage authority requires original Endpoints")
    if len({item.endpoint_id for item in endpoints}) != len(endpoints):
        raise ObservationPolicyError("coverage endpoints contain duplicate identities")
    cells_by_id, edges_by_id, pairs_by_id = _target_maps(universe)
    try:
        credited_cells = tuple(cells_by_id[item] for item in cell_ids)
        credited_edges = tuple(edges_by_id[item] for item in edge_ids)
        credited_pairs = tuple(pairs_by_id[item] for item in backend_pair_obligation_ids)
    except KeyError as exc:
        raise ObservationPolicyError("coverage authority references an unknown target") from exc
    if not credited_pairs:
        raise ObservationPolicyError("executed coverage requires declared backend pairs")

    pair_cells = {item.target_cell_id for item in credited_pairs}
    required_cell_ids = (
        {item.target_cell_id for item in credited_cells}
        | {
            target_cell_id
            for edge in credited_edges
            for target_cell_id in (edge.base_cell_id, edge.sibling_cell_id)
        }
        | pair_cells
    )
    explicitly_required_cells = {
        item.target_cell_id for item in credited_cells
    } | {
        target_cell_id
        for edge in credited_edges
        for target_cell_id in (edge.base_cell_id, edge.sibling_cell_id)
    }
    missing_pair_cells = explicitly_required_cells - pair_cells
    if missing_pair_cells:
        raise ObservationPolicyError(
            "coverage target lacks backend-pair authority: "
            + ",".join(sorted(missing_pair_cells))
        )
    case_by_cell = _case_digests(
        extraction,
        frozenset(required_cell_ids),
        edge_ids=edge_ids,
    )

    cell_policies = {
        cell_id: resolve_cell_observation_policy(cells_by_id[cell_id])
        for cell_id in required_cell_ids
    }
    for pair in credited_pairs:
        cell = cells_by_id[pair.target_cell_id]
        if (
            pair.target_backend != cell.target_backend
            or pair.control_backend not in cell.control_backends
            or pair.observation_contract != cell_policies[cell.target_cell_id].digest
        ):
            raise ObservationPolicyError(
                f"backend-pair policy is not bound to its cell: {pair.obligation_id}"
            )
    for edge in credited_edges:
        resolve_edge_observation_policy(edge, cells_by_id=cells_by_id)

    expected_keys = {
        (case_by_cell[pair.target_cell_id], backend)
        for pair in credited_pairs
        for backend in (pair.target_backend, pair.control_backend)
    }
    actual_by_key = {_endpoint_key(item): item for item in endpoints}
    if len(actual_by_key) != len(endpoints):
        raise ObservationPolicyError("coverage endpoints duplicate a case/backend role")
    if set(actual_by_key) != expected_keys:
        raise ObservationPolicyError(
            "coverage endpoints do not match extraction and backend-pair roles"
        )
    for pair in credited_pairs:
        cell = cells_by_id[pair.target_cell_id]
        target_endpoint = actual_by_key[
            (case_by_cell[pair.target_cell_id], pair.target_backend)
        ]
        coordinates = cell.coordinate_map
        expected_layout = coordinates.get(
            "physical_layout", coordinates.get("layout", "")
        )
        expected_mode = coordinates.get(
            "execution_mode", coordinates.get("mode", "")
        )
        if expected_layout and target_endpoint.physical_layout != expected_layout:
            raise ObservationPolicyError(
                f"target endpoint physical layout does not match cell: {cell.target_cell_id}"
            )
        if expected_mode and target_endpoint.execution_mode != expected_mode:
            raise ObservationPolicyError(
                f"target endpoint execution mode does not match cell: {cell.target_cell_id}"
            )

    cells_for_endpoint: dict[str, set[str]] = {}
    for pair in credited_pairs:
        for backend in (pair.target_backend, pair.control_backend):
            endpoint = actual_by_key[(case_by_cell[pair.target_cell_id], backend)]
            cells_for_endpoint.setdefault(endpoint.endpoint_id, set()).add(
                pair.target_cell_id
            )
    requirements = tuple(
        EndpointRequirement(
            endpoint_id=endpoint.endpoint_id,
            backend=endpoint.backend,
            version_spec=endpoint.backend_version,
            execution_modes=frozenset({endpoint.execution_mode}),
            physical_layouts=frozenset({endpoint.physical_layout}),
            required_capabilities=frozenset(
                capability
                for cell_id in cells_for_endpoint[endpoint.endpoint_id]
                for capability in cells_by_id[cell_id].required_capabilities
            ),
        )
        for endpoint in endpoints
    )

    obligations: list[RelationObligation] = [
        _obligation(
            scope="coverage:endpoint-status",
            relation_id="status_ok",
            endpoint_ids=tuple(item.endpoint_id for item in endpoints),
            components=("status",),
        )
    ]
    pair_endpoint_ids: dict[str, tuple[str, str]] = {}
    for pair in sorted(credited_pairs, key=lambda item: item.obligation_id):
        case_digest = case_by_cell[pair.target_cell_id]
        pair_ids = (
            actual_by_key[(case_digest, pair.target_backend)].endpoint_id,
            actual_by_key[(case_digest, pair.control_backend)].endpoint_id,
        )
        pair_endpoint_ids[pair.obligation_id] = pair_ids
        obligations.extend(
            _equality_obligations(
                scope=f"pair:{pair.obligation_id}",
                relation_id=cell_policies[pair.target_cell_id].primary_relation_id,
                endpoint_ids=pair_ids,
            )
        )

    pairs_by_cell: dict[str, tuple[BackendPairObligation, ...]] = {
        cell_id: tuple(
            pair for pair in credited_pairs if pair.target_cell_id == cell_id
        )
        for cell_id in required_cell_ids
    }
    for edge in sorted(credited_edges, key=lambda item: item.contrast_edge_id):
        base = cells_by_id[edge.base_cell_id]
        sibling = cells_by_id[edge.sibling_cell_id]
        base_backends = {
            backend
            for pair in pairs_by_cell[base.target_cell_id]
            for backend in (pair.target_backend, pair.control_backend)
        }
        sibling_backends = {
            backend
            for pair in pairs_by_cell[sibling.target_cell_id]
            for backend in (pair.target_backend, pair.control_backend)
        }
        shared_backends = tuple(sorted(base_backends & sibling_backends))
        if not shared_backends:
            raise ObservationPolicyError(
                f"edge has no shared backend contrast role: {edge.contrast_edge_id}"
            )
        base_policy = cell_policies[base.target_cell_id]
        sibling_policy = cell_policies[sibling.target_cell_id]
        if edge.relation is ContrastRelation.METAMORPHIC_EQUIVALENCE:
            if base_policy.primary_relation_id != sibling_policy.primary_relation_id:
                raise ObservationPolicyError("metamorphic edge changes observation relation")
            for backend in shared_backends:
                obligations.extend(
                    _equality_obligations(
                        scope=f"edge:{edge.contrast_edge_id}:{backend}",
                        relation_id=base_policy.primary_relation_id,
                        endpoint_ids=(
                            actual_by_key[
                                (case_by_cell[base.target_cell_id], backend)
                            ].endpoint_id,
                            actual_by_key[
                                (case_by_cell[sibling.target_cell_id], backend)
                            ].endpoint_id,
                        ),
                    )
                )
        elif edge.relation is ContrastRelation.DIFFERENTIAL_ISOLATION:
            # The two per-cell pair obligations above are the finite product
            # contract: each context is adjudicated independently.  Raw outputs
            # across the changed axis are deliberately not equated.
            pass
        elif edge.relation is ContrastRelation.MONOTONIC_BOUNDARY:
            if not (
                base_policy.probe_predicate_digest
                and sibling_policy.probe_predicate_digest
                and base_policy.primary_relation_id == "witness_false"
                and sibling_policy.primary_relation_id == "witness_false"
            ):
                raise ObservationPolicyError(
                    "non-probe monotonic edge has no versioned direction policy"
                )
            obligations.append(
                _obligation(
                    scope=f"edge:{edge.contrast_edge_id}:predicate-boundary",
                    relation_id="witness_false",
                    endpoint_ids=tuple(
                        actual_by_key[(case_by_cell[cell_id], backend)].endpoint_id
                        for cell_id in (base.target_cell_id, sibling.target_cell_id)
                        for backend in shared_backends
                    ),
                )
            )
        else:  # pragma: no cover - enum exhaustiveness guard
            raise ObservationPolicyError("unknown contrast relation")

    obligation_ids = tuple(item.obligation_id for item in obligations)
    if len(obligation_ids) != len(set(obligation_ids)):
        raise ObservationPolicyError("materialized target obligations are not unique")
    policy_payload = {
        "schema_version": OBSERVATION_POLICY_SCHEMA_VERSION,
        "universe_digest": universe.digest,
        "cell_policies": tuple(
            sorted(cell_policies[cell_id].digest for cell_id in required_cell_ids)
        ),
        "edge_policies": tuple(
            sorted(
                resolve_edge_observation_policy(edge, cells_by_id=cells_by_id).digest
                for edge in credited_edges
            )
        ),
        "pair_obligations": tuple(
            sorted(pair.digest for pair in credited_pairs)
        ),
        "endpoint_scopes": requirements,
        "obligations": tuple(obligations),
    }
    derivation_digest = stable_digest(
        "osc-target-observation-derivation", policy_payload
    )
    contract_id = "target-contract-" + stable_digest(
        "osc-target-observation-contract-id",
        {"derivation": derivation_digest, "endpoints": tuple(item.endpoint_id for item in endpoints)},
    )[-32:]
    observations = tuple(
        dict.fromkeys(
            component for obligation in obligations for component in obligation.components
        )
    )
    return HyperContract(
        contract_id=contract_id,
        endpoint_requirements=requirements,
        preconditions=("no_unresolved_rules", "endpoint_scope_matches"),
        observations=observations,
        obligations=tuple(obligations),
        strength="osc-target-observation-authority-v1",
        derivation_digest=derivation_digest,
        registry_digest=default_relation_registry().digest,
    )


__all__ = [
    "CellObservationPolicy",
    "EdgeObservationPolicy",
    "ObservationPolicyError",
    "ProbePredicateContract",
    "compile_cell_observation_policy",
    "compile_edge_observation_policy",
    "materialize_coverage_hypercontract",
    "probe_predicate_contracts",
    "resolve_cell_observation_policy",
    "resolve_edge_observation_policy",
    "resolve_probe_predicate",
]
