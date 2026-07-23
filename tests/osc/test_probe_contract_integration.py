from __future__ import annotations

from dataclasses import replace

import pytest

from datadiff.family_witness_registry import latest_family_witness_registrations

from datadiff_osc.contract_engine import (
    Endpoint,
    Observation,
    SchemaField,
    evaluate_applicability,
    execute_staged_comparison,
)
from datadiff_osc.generation.extraction import AtomExtractor, merge_extractions
from datadiff_osc.probe_contracts import (
    ObservationPolicyError,
    materialize_coverage_hypercontract,
    probe_predicate_contracts,
    resolve_cell_observation_policy,
    resolve_edge_observation_policy,
)
from datadiff_osc.schemas import (
    ExecutionStatus,
    FailureKind,
    StructuredExecutionOutcome,
    VerdictKind,
)
from datadiff_osc.semantic_targets.compiler import compile_target_universe
from datadiff_osc.semantic_targets.declarations import legacy_v4_target_templates
from datadiff_osc.semantic_targets.model import ContrastRelation


def _universe():
    return compile_target_universe(legacy_v4_target_templates())


def _registrations():
    return {
        item.family_id: item for item in latest_family_witness_registrations()
    }


def _extraction_for(cell, registrations, extractor):
    return extractor.extract(
        registrations[cell.test_family_id].generate_case(cell.construction_index)
    )


def _edge_authority_fixture(
    *,
    family_id="pandas_nullable_bool_reduction",
    relation=None,
):
    universe = _universe()
    cells = {item.target_cell_id: item for item in universe.fresh_cells}
    edge = next(
        item
        for item in universe.fresh_edges
        if cells[item.base_cell_id].test_family_id == family_id
        and (relation is None or item.relation is relation)
    )
    base = cells[edge.base_cell_id]
    sibling = cells[edge.sibling_cell_id]
    pairs = tuple(
        next(
            item
            for item in universe.fresh_backend_pair_obligations
            if item.target_cell_id == cell.target_cell_id
        )
        for cell in (base, sibling)
    )
    extractor = AtomExtractor()
    registrations = _registrations()
    contrast = merge_extractions(
        tuple(
            (
                cell.target_cell_id,
                _extraction_for(cell, registrations, extractor),
            )
            for cell in (base, sibling)
        )
    )
    endpoints = tuple(
        Endpoint.build(
            endpoint_id=f"{cell.target_cell_id[-12:]}:{backend}",
            case_digest=contrast.extraction_for(cell.target_cell_id).source_digest,
            backend=backend,
            backend_version="1.0.0",
            adapter_revision="adapter-v1",
            execution_mode="eager",
            physical_layout="contiguous",
            capabilities=cell.required_capabilities,
        )
        for cell, pair in zip((base, sibling), pairs, strict=True)
        for backend in (pair.target_backend, pair.control_backend)
    )
    return universe, base, sibling, edge, pairs, contrast, endpoints


def _execute(contract, endpoints, *, value=1):
    applicability = evaluate_applicability(
        contract,
        endpoints,
        facts=frozenset(contract.preconditions),
    )
    assert applicability.valid
    schema = (SchemaField("value", "bool" if isinstance(value, bool) else "int", False),)
    observations = tuple(
        Observation.build(
            endpoint_id=item.endpoint_id,
            status="ok",
            schema=schema,
            rows=[[value]],
            execution_metadata={"endpoint_digest": item.digest},
        )
        for item in endpoints
    )
    outcomes = tuple(
        StructuredExecutionOutcome(
            item.endpoint_id,
            ExecutionStatus.OK,
            FailureKind.NONE,
        )
        for item in endpoints
    )
    return execute_staged_comparison(
        contract,
        observations,
        applicability,
        evidence_tier="audit",
        endpoints=endpoints,
        force_exact=True,
        execution_outcomes=outcomes,
    )


def test_compiler_emits_unique_recomputable_cell_and_directional_edge_policies():
    universe = _universe()
    cells = (*universe.fresh_cells, *universe.regression_cells)
    cells_by_id = {item.target_cell_id: item for item in cells}

    assert len(cells) == 232 + 144
    assert len({item.observation_contract for item in cells}) == len(cells)
    assert len(universe.fresh_edges) == 384
    assert len({item.observation_contract for item in universe.fresh_edges}) == 384
    assert all(
        resolve_cell_observation_policy(item).digest == item.observation_contract
        for item in cells
    )
    assert all(
        resolve_edge_observation_policy(item, cells_by_id=cells_by_id).digest
        == item.observation_contract
        for item in universe.fresh_edges
    )

    reversed_edge = replace(
        universe.fresh_edges[0],
        base_cell_id=universe.fresh_edges[0].sibling_cell_id,
        sibling_cell_id=universe.fresh_edges[0].base_cell_id,
    )
    with pytest.raises(ObservationPolicyError, match="does not match its direction"):
        resolve_edge_observation_policy(reversed_edge, cells_by_id=cells_by_id)


def test_five_probe_predicates_are_versioned_witness_false_and_never_collapse():
    universe = _universe()
    probe_cells = []
    for cell in (*universe.fresh_cells, *universe.regression_cells):
        policy = resolve_cell_observation_policy(cell)
        if policy.probe_predicate_digest:
            probe_cells.append((cell, policy))

    registry = probe_predicate_contracts()
    assert len(registry) == 5
    assert len({item.predicate_id for item in registry}) == 5
    assert len({item.digest for item in registry}) == 5
    assert {item.relation_id for item in registry} == {"witness_false"}
    assert len(probe_cells) == 24
    assert {policy.primary_relation_id for _, policy in probe_cells} == {
        "witness_false"
    }
    assert len({policy.probe_predicate_digest for _, policy in probe_cells}) == 5
    monotonic_edges = tuple(
        item
        for item in universe.fresh_edges
        if item.relation is ContrastRelation.MONOTONIC_BOUNDARY
    )
    cells_by_id = {item.target_cell_id: item for item in universe.fresh_cells}
    assert len(monotonic_edges) == 14
    assert all(
        resolve_cell_observation_policy(cells_by_id[cell_id]).probe_predicate_digest
        for edge in monotonic_edges
        for cell_id in (edge.base_cell_id, edge.sibling_cell_id)
    )

    original, _ = probe_cells[0]
    known_probe_atom = next(
        atom for atom in original.required_all_atoms if atom.endswith("_probe")
    )
    unregistered = replace(
        original,
        required_all_atoms=frozenset(
            {
                *(original.required_all_atoms - {known_probe_atom}),
                "op:random_case_probe",
            }
        ),
    )
    with pytest.raises(ObservationPolicyError, match="unregistered probe predicate"):
        resolve_cell_observation_policy(unregistered)


def test_materialized_edge_contract_binds_both_cases_and_declared_backend_roles():
    universe, base, sibling, edge, pairs, contrast, endpoints = (
        _edge_authority_fixture()
    )
    contract = materialize_coverage_hypercontract(
        universe,
        cell_ids=(base.target_cell_id, sibling.target_cell_id),
        edge_ids=(edge.contrast_edge_id,),
        backend_pair_obligation_ids=tuple(item.obligation_id for item in pairs),
        extraction=contrast,
        endpoints=endpoints,
    )
    repeated = materialize_coverage_hypercontract(
        universe,
        cell_ids=(base.target_cell_id, sibling.target_cell_id),
        edge_ids=(edge.contrast_edge_id,),
        backend_pair_obligation_ids=tuple(item.obligation_id for item in pairs),
        extraction=contrast,
        endpoints=endpoints,
    )
    assert contract == repeated
    assert {item.backend for item in contract.endpoint_requirements} == {
        base.target_backend,
        pairs[0].control_backend,
    }
    assert _execute(contract, endpoints).verdict.kind is VerdictKind.SATISFIED

    with pytest.raises(ObservationPolicyError, match="lacks backend-pair authority"):
        materialize_coverage_hypercontract(
            universe,
            cell_ids=(base.target_cell_id,),
            edge_ids=(edge.contrast_edge_id,),
            backend_pair_obligation_ids=(pairs[0].obligation_id,),
            extraction=contrast,
            endpoints=endpoints[:2],
        )
    with pytest.raises(ObservationPolicyError, match="do not match extraction"):
        materialize_coverage_hypercontract(
            universe,
            cell_ids=(base.target_cell_id, sibling.target_cell_id),
            edge_ids=(edge.contrast_edge_id,),
            backend_pair_obligation_ids=tuple(item.obligation_id for item in pairs),
            extraction=contrast,
            endpoints=(replace(endpoints[0], case_digest="unrelated-case"), *endpoints[1:]),
        )
    with pytest.raises(ObservationPolicyError, match="do not match extraction"):
        materialize_coverage_hypercontract(
            universe,
            cell_ids=(base.target_cell_id, sibling.target_cell_id),
            edge_ids=(edge.contrast_edge_id,),
            backend_pair_obligation_ids=tuple(item.obligation_id for item in pairs),
            extraction=contrast,
            endpoints=(replace(endpoints[0], backend="unrelated_backend"), *endpoints[1:]),
        )


def test_probe_cell_materializes_witness_false_instead_of_bag_equality():
    universe = _universe()
    cell = next(
        item
        for item in universe.fresh_cells
        if resolve_cell_observation_policy(item).probe_predicate_digest
        and "layout" in item.coordinate_map
    )
    pair = next(
        item
        for item in universe.fresh_backend_pair_obligations
        if item.target_cell_id == cell.target_cell_id
    )
    extraction = _extraction_for(cell, _registrations(), AtomExtractor())
    endpoints = tuple(
        Endpoint.build(
            endpoint_id=f"probe:{backend}",
            case_digest=extraction.source_digest,
            backend=backend,
            backend_version="1.0.0",
            adapter_revision="adapter-v1",
            execution_mode="eager",
            physical_layout=cell.coordinate_map.get("layout", "contiguous"),
            capabilities=cell.required_capabilities,
        )
        for backend in (pair.target_backend, pair.control_backend)
    )
    contract = materialize_coverage_hypercontract(
        universe,
        cell_ids=(cell.target_cell_id,),
        edge_ids=(),
        backend_pair_obligation_ids=(pair.obligation_id,),
        extraction=extraction,
        endpoints=endpoints,
    )
    relation_ids = tuple(item.relation_id for item in contract.obligations)
    assert "witness_false" in relation_ids
    assert "bag_equal" not in relation_ids
    assert _execute(contract, endpoints, value=False).verdict.kind is VerdictKind.SATISFIED
    assert _execute(contract, endpoints, value=True).verdict.kind is VerdictKind.VIOLATED

    pair_only = materialize_coverage_hypercontract(
        universe,
        cell_ids=(),
        edge_ids=(),
        backend_pair_obligation_ids=(pair.obligation_id,),
        extraction=extraction,
        endpoints=endpoints,
    )
    assert pair_only == contract
    with pytest.raises(ObservationPolicyError, match="physical layout"):
        materialize_coverage_hypercontract(
            universe,
            cell_ids=(cell.target_cell_id,),
            edge_ids=(),
            backend_pair_obligation_ids=(pair.obligation_id,),
            extraction=extraction,
            endpoints=(
                replace(endpoints[0], physical_layout="unrelated-layout"),
                endpoints[1],
            ),
        )


def test_probe_monotonic_edge_materializes_both_directional_contexts():
    universe, base, sibling, edge, pairs, contrast, endpoints = (
        _edge_authority_fixture(
            family_id="polars_lazy_temporal_cast_boundary",
            relation=ContrastRelation.MONOTONIC_BOUNDARY,
        )
    )
    contract = materialize_coverage_hypercontract(
        universe,
        cell_ids=(base.target_cell_id, sibling.target_cell_id),
        edge_ids=(edge.contrast_edge_id,),
        backend_pair_obligation_ids=tuple(item.obligation_id for item in pairs),
        extraction=contrast,
        endpoints=endpoints,
    )
    assert tuple(item.relation_id for item in contract.obligations).count(
        "witness_false"
    ) == 3
    assert _execute(contract, endpoints, value=False).verdict.kind is VerdictKind.SATISFIED
