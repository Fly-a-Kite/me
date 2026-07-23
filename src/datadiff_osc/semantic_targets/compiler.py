from __future__ import annotations

from itertools import combinations, product
from typing import Iterable, Mapping

from datadiff.semantic_family_universe_v3 import pipeline_evidence_spec

from datadiff_osc._canonical import stable_digest
from datadiff_osc.probe_contracts import (
    compile_cell_observation_policy,
    compile_edge_observation_policy,
)

from .declarations import declarations_digest
from .model import (
    BackendPairObligation,
    CompiledTargetUniverse,
    ContrastEdge,
    InteractionTile,
    ProvenanceClass,
    TargetCell,
    TargetTemplate,
)
from .taxonomy import AtomTaxonomy, semantic_atoms_for_axis


COMPILER_SCHEMA_VERSION = "osc-target-compiler-v1"


def _pipeline_atoms(pipeline_id: str) -> frozenset[str]:
    try:
        evidence = pipeline_evidence_spec(pipeline_id)
    except KeyError as exc:
        raise ValueError(f"unknown target pipeline: {pipeline_id}") from exc
    atoms = {f"pipeline:{pipeline_id}"}
    atoms.update(f"op:{item}" for item in evidence.required_operations)
    atoms.update(f"expr:{item}" for item in evidence.required_expressions)
    atoms.update(f"agg:{item}" for item in evidence.required_aggregates)
    # aggregate_any_of is represented as a matcher group, not an all-of fact.
    for chain in evidence.required_operation_chains:
        atoms.add("chain:" + ">".join(chain))
    return frozenset(atoms)


def _cell_atoms(
    template: TargetTemplate, coordinates: Mapping[str, str]
) -> tuple[frozenset[str], tuple[frozenset[str], ...]]:
    required = set(template.required_all_atoms)
    groups = list(template.required_any_atom_groups)
    for name, value in coordinates.items():
        required.update(semantic_atoms_for_axis(name, value))
        if name == "pipeline":
            required.update(_pipeline_atoms(value))
            try:
                evidence = pipeline_evidence_spec(value)
            except KeyError as exc:
                raise ValueError(f"unknown target pipeline: {value}") from exc
            if evidence.aggregate_any_of:
                groups.append(
                    frozenset(f"agg:{item}" for item in evidence.aggregate_any_of)
                )
    return frozenset(required), tuple(groups)


def _compile_cells(
    template: TargetTemplate, taxonomy: AtomTaxonomy
) -> tuple[TargetCell, ...]:
    cells: list[TargetCell] = []
    names = tuple(axis.name for axis in template.axes)
    values = tuple(axis.values for axis in template.axes)
    for construction_index, selection in enumerate(product(*values)):
        coordinates = tuple(zip(names, selection, strict=True))
        coordinate_map = dict(coordinates)
        required, groups = _cell_atoms(template, coordinate_map)
        taxonomy.compile(required)
        for group in groups:
            taxonomy.compile(group)
        taxonomy.compile(template.forbidden_atoms)
        identity = {
            "schema_version": COMPILER_SCHEMA_VERSION,
            "template_id": template.template_id,
            "coordinates": coordinates,
        }
        cell_id = stable_digest("osc-target-cell-id", identity)
        contract_payload = {
            "cell_id": cell_id,
            "required_all": sorted(required),
            "required_any": [sorted(group) for group in groups],
            "forbidden": sorted(template.forbidden_atoms),
        }
        activation_contract = stable_digest(
            "osc-activation-contract", contract_payload
        )
        observation_policy = compile_cell_observation_policy(
            target_cell_id=cell_id,
            target_backend=template.target_backend,
            control_backends=template.control_backends,
            required_all_atoms=required,
            declared_relation_id=template.oracle_relation,
        )
        cells.append(
            TargetCell(
                target_cell_id=cell_id,
                template_id=template.template_id,
                test_family_id=template.test_family_id,
                provenance_class=template.provenance_class,
                coordinates=coordinates,
                required_all_atoms=required,
                required_any_atom_groups=groups,
                forbidden_atoms=template.forbidden_atoms,
                target_backend=template.target_backend,
                control_backends=template.control_backends,
                required_capabilities=template.required_capabilities,
                risk_class=template.risk_class,
                cost_class=template.cost_class,
                construction_contract=stable_digest(
                    "osc-construction-contract",
                    {
                        "provider": template.constructor_provider,
                        "coordinates": coordinates,
                    },
                ),
                activation_contract=activation_contract,
                observation_contract=observation_policy.digest,
                construction_index=construction_index,
            )
        )
    return tuple(cells)


def _cell_lookup(cells: Iterable[TargetCell]) -> dict[tuple[tuple[str, str], ...], TargetCell]:
    return {cell.coordinates: cell for cell in cells}


def _compile_edges(
    template: TargetTemplate, cells: tuple[TargetCell, ...]
) -> tuple[ContrastEdge, ...]:
    if template.provenance_class != ProvenanceClass.FRESH_DISCOVERY:
        return ()
    lookup = _cell_lookup(cells)
    edges: list[ContrastEdge] = []
    for axis in template.axes:
        for cell in cells:
            coordinates = cell.coordinate_map
            if coordinates[axis.name] != axis.baseline:
                continue
            for sibling_value in axis.values:
                if sibling_value == axis.baseline:
                    continue
                sibling_coordinates = dict(coordinates)
                sibling_coordinates[axis.name] = sibling_value
                sibling_key = tuple(
                    (candidate.name, sibling_coordinates[candidate.name])
                    for candidate in template.axes
                )
                sibling = lookup[sibling_key]
                edge_id = stable_digest(
                    "osc-contrast-edge-id",
                    {
                        "template_id": template.template_id,
                        "base": cell.target_cell_id,
                        "sibling": sibling.target_cell_id,
                        "axis": axis.name,
                        "relation": axis.relation.value,
                    },
                )
                edges.append(
                    ContrastEdge(
                        contrast_edge_id=edge_id,
                        base_cell_id=cell.target_cell_id,
                        sibling_cell_id=sibling.target_cell_id,
                        changed_axis=axis.name,
                        base_value=axis.baseline,
                        sibling_value=sibling_value,
                        relation=axis.relation,
                        activation_contract=stable_digest(
                            "osc-edge-activation-contract",
                            (cell.activation_contract, sibling.activation_contract),
                        ),
                        observation_contract=compile_edge_observation_policy(
                            contrast_edge_id=edge_id,
                            base_cell_id=cell.target_cell_id,
                            sibling_cell_id=sibling.target_cell_id,
                            changed_axis=axis.name,
                            base_value=axis.baseline,
                            sibling_value=sibling_value,
                            relation=axis.relation,
                            base_policy_digest=cell.observation_contract,
                            sibling_policy_digest=sibling.observation_contract,
                        ).digest,
                    )
                )
    return tuple(sorted(edges, key=lambda item: item.contrast_edge_id))


def _compile_tiles(
    template: TargetTemplate, cells: tuple[TargetCell, ...]
) -> tuple[InteractionTile, ...]:
    if template.provenance_class != ProvenanceClass.FRESH_DISCOVERY:
        return ()
    lookup = _cell_lookup(cells)
    tiles: list[InteractionTile] = []
    for axis_a, axis_b in combinations(template.axes, 2):
        other_axes = tuple(
            axis
            for axis in template.axes
            if axis.name not in {axis_a.name, axis_b.name}
        )
        other_values = tuple(axis.values for axis in other_axes)
        contexts = product(*other_values) if other_values else ((),)
        for context in contexts:
            non_target = tuple(
                (axis.name, value)
                for axis, value in zip(other_axes, context, strict=True)
            )
            for value_a in axis_a.values:
                if value_a == axis_a.baseline:
                    continue
                for value_b in axis_b.values:
                    if value_b == axis_b.baseline:
                        continue
                    base_map = dict(non_target)
                    base_map[axis_a.name] = axis_a.baseline
                    base_map[axis_b.name] = axis_b.baseline
                    a_map = {**base_map, axis_a.name: value_a}
                    b_map = {**base_map, axis_b.name: value_b}
                    joint_map = {**a_map, axis_b.name: value_b}

                    def cell_id(mapping: Mapping[str, str]) -> str:
                        key = tuple(
                            (axis.name, mapping[axis.name]) for axis in template.axes
                        )
                        return lookup[key].target_cell_id

                    endpoints = (
                        cell_id(base_map),
                        cell_id(a_map),
                        cell_id(b_map),
                        cell_id(joint_map),
                    )
                    tile_id = stable_digest(
                        "osc-interaction-tile-id",
                        {
                            "template_id": template.template_id,
                            "axes": (axis_a.name, axis_b.name),
                            "endpoints": endpoints,
                        },
                    )
                    tiles.append(
                        InteractionTile(
                            tile_id=tile_id,
                            test_family_id=template.test_family_id,
                            axis_a=axis_a.name,
                            axis_b=axis_b.name,
                            base_cell_id=endpoints[0],
                            a_only_cell_id=endpoints[1],
                            b_only_cell_id=endpoints[2],
                            joint_cell_id=endpoints[3],
                            non_target_coordinates=non_target,
                        )
                    )
    return tuple(sorted(tiles, key=lambda item: item.tile_id))


def compile_target_universe(
    templates: Iterable[TargetTemplate],
    *,
    taxonomy: AtomTaxonomy | None = None,
) -> CompiledTargetUniverse:
    taxonomy = taxonomy or AtomTaxonomy()
    materialized = tuple(sorted(templates, key=lambda item: item.template_id))
    if len({item.template_id for item in materialized}) != len(materialized):
        raise ValueError("duplicate target template identity")
    if len({item.test_family_id for item in materialized}) != len(materialized):
        raise ValueError("duplicate target family declaration")

    fresh_cells: list[TargetCell] = []
    regression_cells: list[TargetCell] = []
    fresh_edges: list[ContrastEdge] = []
    fresh_obligations: list[BackendPairObligation] = []
    regression_obligations: list[BackendPairObligation] = []
    tiles: list[InteractionTile] = []
    for template in materialized:
        cells = _compile_cells(template, taxonomy)
        destination = (
            fresh_cells
            if template.provenance_class == ProvenanceClass.FRESH_DISCOVERY
            else regression_cells
        )
        destination.extend(cells)
        fresh_edges.extend(_compile_edges(template, cells))
        tiles.extend(_compile_tiles(template, cells))
        obligation_destination = (
            fresh_obligations
            if template.provenance_class == ProvenanceClass.FRESH_DISCOVERY
            else regression_obligations
        )
        for cell in cells:
            for control in cell.control_backends:
                obligation_destination.append(
                    BackendPairObligation(
                        obligation_id=stable_digest(
                            "osc-backend-pair-obligation-id",
                            {
                                "cell_id": cell.target_cell_id,
                                "target": cell.target_backend,
                                "control": control,
                            },
                        ),
                        target_cell_id=cell.target_cell_id,
                        provenance_class=cell.provenance_class,
                        target_backend=cell.target_backend,
                        control_backend=control,
                        observation_contract=cell.observation_contract,
                    )
                )
    return CompiledTargetUniverse(
        fresh_cells=tuple(sorted(fresh_cells, key=lambda item: item.target_cell_id)),
        regression_cells=tuple(
            sorted(regression_cells, key=lambda item: item.target_cell_id)
        ),
        fresh_edges=tuple(
            sorted(fresh_edges, key=lambda item: item.contrast_edge_id)
        ),
        fresh_backend_pair_obligations=tuple(
            sorted(fresh_obligations, key=lambda item: item.obligation_id)
        ),
        regression_backend_pair_obligations=tuple(
            sorted(regression_obligations, key=lambda item: item.obligation_id)
        ),
        shadow_tiles=tuple(sorted(tiles, key=lambda item: item.tile_id)),
        taxonomy_digest=taxonomy.digest,
        template_digest=declarations_digest(materialized),
    )
