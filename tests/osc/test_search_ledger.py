from __future__ import annotations

from dataclasses import dataclass, replace

import pytest

from datadiff.family_witness_registry import latest_family_witness_registrations

from datadiff_osc.contract_engine import (
    Endpoint,
    Observation,
    SchemaField,
    evaluate_applicability,
    execute_staged_comparison,
)
from datadiff_osc.generation.extraction import (
    AtomExtractor,
    ContrastExtraction,
    ExtractionResult,
    merge_extractions,
)
from datadiff_osc.probe_contracts import materialize_coverage_hypercontract
from datadiff_osc.schemas import (
    CoverageLevel,
    ExecutionStatus,
    FailureKind,
    LedgerEvent,
    SeedLineage,
    SeedStage,
    StructuredExecutionOutcome,
    VerdictKind,
)
from datadiff_osc.search.ledger import CoverageLedger
from datadiff_osc.search.matcher import TargetMatcher
from datadiff_osc.semantic_targets.compiler import compile_target_universe
from datadiff_osc.semantic_targets.declarations import legacy_v4_target_templates
from datadiff_osc.semantic_targets.model import TargetAssignment


@dataclass(frozen=True)
class _EdgeFixture:
    universe: object
    base: object
    sibling: object
    edge: object
    pairs: tuple[object, object]
    contrast: ContrastExtraction
    endpoints: tuple[Endpoint, ...]
    assignment: TargetAssignment
    activation: object


@dataclass(frozen=True)
class _CellFixture:
    universe: object
    cell: object
    pair: object
    extraction: ExtractionResult
    endpoints: tuple[Endpoint, ...]
    assignment: TargetAssignment
    activation: object


def _universe():
    return compile_target_universe(legacy_v4_target_templates())


def _registrations():
    return {
        item.family_id: item for item in latest_family_witness_registrations()
    }


def _extract(cell, registrations, extractor):
    return extractor.extract(
        registrations[cell.test_family_id].generate_case(cell.construction_index)
    )


def _endpoints_for(cells, pairs, extractions):
    return tuple(
        Endpoint.build(
            endpoint_id=f"{cell.target_cell_id[-12:]}:{backend}",
            case_digest=extractions[cell.target_cell_id].source_digest,
            backend=backend,
            backend_version="1.0.0",
            adapter_revision="adapter-v1",
            execution_mode=cell.coordinate_map.get(
                "execution_mode", cell.coordinate_map.get("mode", "eager")
            ),
            physical_layout=cell.coordinate_map.get(
                "physical_layout", cell.coordinate_map.get("layout", "contiguous")
            ),
            capabilities=cell.required_capabilities,
        )
        for cell, pair in zip(cells, pairs, strict=True)
        for backend in (pair.target_backend, pair.control_backend)
    )


def _edge_fixture() -> _EdgeFixture:
    universe = _universe()
    cells_by_id = {item.target_cell_id: item for item in universe.fresh_cells}
    edge = next(
        item
        for item in universe.fresh_edges
        if cells_by_id[item.base_cell_id].test_family_id
        == "pandas_nullable_bool_reduction"
    )
    base = cells_by_id[edge.base_cell_id]
    sibling = cells_by_id[edge.sibling_cell_id]
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
    extractions = {
        cell.target_cell_id: _extract(cell, registrations, extractor)
        for cell in (base, sibling)
    }
    contrast = merge_extractions(
        tuple(
            (cell.target_cell_id, extractions[cell.target_cell_id])
            for cell in (base, sibling)
        )
    )
    endpoints = _endpoints_for((base, sibling), pairs, extractions)
    assignment = TargetAssignment(
        (base.target_cell_id, sibling.target_cell_id),
        SeedLineage("p", 1, "lane", 0, SeedStage.TARGET),
        selected_edge_ids=(edge.contrast_edge_id,),
    )
    activation = TargetMatcher(universe).match(assignment, contrast)
    assert activation.valid
    return _EdgeFixture(
        universe,
        base,
        sibling,
        edge,
        pairs,
        contrast,
        endpoints,
        assignment,
        activation,
    )


def _cell_fixture(edge_fixture: _EdgeFixture | None = None) -> _CellFixture:
    fixture = edge_fixture or _edge_fixture()
    extraction = fixture.contrast.extraction_for(fixture.base.target_cell_id)
    assignment = TargetAssignment(
        (fixture.base.target_cell_id,),
        SeedLineage("p", 2, "lane", 0, SeedStage.TARGET),
    )
    activation = TargetMatcher(fixture.universe).match(assignment, extraction)
    assert activation.valid
    return _CellFixture(
        fixture.universe,
        fixture.base,
        fixture.pairs[0],
        extraction,
        fixture.endpoints[:2],
        assignment,
        activation,
    )


def _authority(
    universe,
    *,
    cell_ids,
    edge_ids,
    pair_ids,
    extraction,
    endpoints,
):
    contract = materialize_coverage_hypercontract(
        universe,
        cell_ids=cell_ids,
        edge_ids=edge_ids,
        backend_pair_obligation_ids=pair_ids,
        extraction=extraction,
        endpoints=endpoints,
    )
    applicability = evaluate_applicability(
        contract,
        endpoints,
        facts=frozenset(contract.preconditions),
    )
    assert applicability.valid
    outcomes = tuple(
        StructuredExecutionOutcome(
            endpoint_id=item.endpoint_id,
            status=ExecutionStatus.OK,
            failure_kind=FailureKind.NONE,
        )
        for item in endpoints
    )
    schema = (SchemaField("x", "int", False),)
    observations = tuple(
        Observation.build(
            endpoint_id=item.endpoint_id,
            status="ok",
            schema=schema,
            rows=[[1], [2]],
            execution_metadata={"endpoint_digest": item.digest},
        )
        for item in endpoints
    )
    observation = execute_staged_comparison(
        contract,
        observations,
        applicability,
        evidence_tier="audit",
        endpoints=endpoints,
        force_exact=True,
        execution_outcomes=outcomes,
    )
    assert observation.valid
    assert observation.verdict.kind is VerdictKind.SATISFIED
    return contract, applicability, outcomes, observation


def _generic_authority(simple_compiled, simple_applicability, endpoints, int_schema):
    outcomes = tuple(
        StructuredExecutionOutcome(
            endpoint_id=item.endpoint_id,
            status=ExecutionStatus.OK,
            failure_kind=FailureKind.NONE,
        )
        for item in endpoints
    )
    observations = tuple(
        Observation.build(
            endpoint_id=item.endpoint_id,
            status="ok",
            schema=int_schema,
            rows=[[1], [2]],
            execution_metadata={"endpoint_digest": item.digest},
        )
        for item in endpoints
    )
    observation = execute_staged_comparison(
        simple_compiled.contract,
        observations,
        simple_applicability,
        evidence_tier="audit",
        endpoints=endpoints,
        force_exact=True,
        execution_outcomes=outcomes,
    )
    assert observation.valid
    return simple_compiled.contract, simple_applicability, outcomes, observation


def _event(
    level,
    order,
    activation,
    *,
    cell_ids=(),
    edge_ids=(),
    pair_ids=(),
    applicability=None,
    outcomes=(),
    observation=None,
):
    values = dict(
        task_id=f"task-{order}",
        level=level,
        event_order_key=(f"{order:04d}",),
        cell_ids=tuple(cell_ids),
        edge_ids=tuple(edge_ids),
        backend_pair_obligation_ids=tuple(pair_ids),
    )
    if level != CoverageLevel.CONSTRUCTED:
        values["activation_certificate_digest"] = activation.digest
    if level in {CoverageLevel.EXECUTED, CoverageLevel.OBSERVED}:
        values["applicability_certificate_digest"] = (
            applicability.digest if applicability is not None else "placeholder-app"
        )
        values["execution_outcome_digests"] = (
            tuple(item.digest for item in outcomes)
            if outcomes
            else ("placeholder-outcome",)
        )
    if level == CoverageLevel.OBSERVED:
        values["observation_certificate_digest"] = (
            observation.digest
            if observation is not None
            else "placeholder-observation"
        )
    return LedgerEvent(**values)


def _activation_kwargs(fixture):
    return {
        "target_assignment": fixture.assignment,
        "extraction": fixture.extraction
        if isinstance(fixture, _CellFixture)
        else fixture.contrast,
        "activation_certificate": fixture.activation,
    }


def _execution_kwargs(fixture, contract, applicability, outcomes, *, endpoints=None):
    return {
        **_activation_kwargs(fixture),
        "contract": contract,
        "endpoints": fixture.endpoints if endpoints is None else endpoints,
        "applicability_certificate": applicability,
        "execution_outcomes": outcomes,
    }


def _admit_prefix(ledger, fixture, *, cell_ids=(), edge_ids=(), pair_ids=()):
    ledger.admit(
        _event(
            CoverageLevel.CONSTRUCTED,
            0,
            fixture.activation,
            cell_ids=cell_ids,
            edge_ids=edge_ids,
            pair_ids=pair_ids,
        )
    )
    ledger.admit(
        _event(
            CoverageLevel.ACTIVATED,
            1,
            fixture.activation,
            cell_ids=cell_ids,
            edge_ids=edge_ids,
            pair_ids=pair_ids,
        ),
        **_activation_kwargs(fixture),
    )


def _admit_authority_levels(
    ledger,
    fixture,
    *,
    cell_ids=(),
    edge_ids=(),
    pair_ids=(),
    contract,
    applicability,
    outcomes,
    observation,
):
    _admit_prefix(
        ledger,
        fixture,
        cell_ids=cell_ids,
        edge_ids=edge_ids,
        pair_ids=pair_ids,
    )
    execution_kwargs = _execution_kwargs(
        fixture, contract, applicability, outcomes
    )
    ledger.admit(
        _event(
            CoverageLevel.EXECUTED,
            2,
            fixture.activation,
            cell_ids=cell_ids,
            edge_ids=edge_ids,
            pair_ids=pair_ids,
            applicability=applicability,
            outcomes=outcomes,
        ),
        **execution_kwargs,
    )
    ledger.admit(
        _event(
            CoverageLevel.OBSERVED,
            3,
            fixture.activation,
            cell_ids=cell_ids,
            edge_ids=edge_ids,
            pair_ids=pair_ids,
            applicability=applicability,
            outcomes=outcomes,
            observation=observation,
        ),
        **execution_kwargs,
        observation_certificate=observation,
    )


def test_cell_and_pair_observed_credit_uses_materialized_target_authority():
    fixture = _cell_fixture()
    cell_ids = (fixture.cell.target_cell_id,)
    pair_ids = (fixture.pair.obligation_id,)
    authority = _authority(
        fixture.universe,
        cell_ids=cell_ids,
        edge_ids=(),
        pair_ids=pair_ids,
        extraction=fixture.extraction,
        endpoints=fixture.endpoints,
    )
    ledger = CoverageLedger(fixture.universe)
    _admit_authority_levels(
        ledger,
        fixture,
        cell_ids=cell_ids,
        pair_ids=pair_ids,
        contract=authority[0],
        applicability=authority[1],
        outcomes=authority[2],
        observation=authority[3],
    )
    counts = ledger.counts(CoverageLevel.OBSERVED)
    assert (counts.cells, counts.edges, counts.backend_pairs) == (1, 0, 1)


def test_pair_only_observed_credit_is_independently_reachable():
    fixture = _cell_fixture()
    pair_ids = (fixture.pair.obligation_id,)
    authority = _authority(
        fixture.universe,
        cell_ids=(),
        edge_ids=(),
        pair_ids=pair_ids,
        extraction=fixture.extraction,
        endpoints=fixture.endpoints,
    )
    ledger = CoverageLedger(fixture.universe)
    _admit_authority_levels(
        ledger,
        fixture,
        pair_ids=pair_ids,
        contract=authority[0],
        applicability=authority[1],
        outcomes=authority[2],
        observation=authority[3],
    )
    counts = ledger.counts(CoverageLevel.OBSERVED)
    assert (counts.cells, counts.edges, counts.backend_pairs) == (0, 0, 1)


def test_full_directional_edge_observed_credit_requires_both_context_pairs():
    fixture = _edge_fixture()
    cell_ids = (fixture.base.target_cell_id, fixture.sibling.target_cell_id)
    edge_ids = (fixture.edge.contrast_edge_id,)
    pair_ids = tuple(item.obligation_id for item in fixture.pairs)
    authority = _authority(
        fixture.universe,
        cell_ids=cell_ids,
        edge_ids=edge_ids,
        pair_ids=pair_ids,
        extraction=fixture.contrast,
        endpoints=fixture.endpoints,
    )
    ledger = CoverageLedger(fixture.universe)
    _admit_authority_levels(
        ledger,
        fixture,
        cell_ids=cell_ids,
        edge_ids=edge_ids,
        pair_ids=pair_ids,
        contract=authority[0],
        applicability=authority[1],
        outcomes=authority[2],
        observation=authority[3],
    )
    counts = ledger.counts(CoverageLevel.OBSERVED)
    assert (counts.cells, counts.edges, counts.backend_pairs) == (2, 1, 2)


def test_all_384_edges_retain_materializable_direction_bound_authority():
    universe = _universe()
    cells = {item.target_cell_id: item for item in universe.fresh_cells}
    pairs_by_cell = {}
    for pair in universe.fresh_backend_pair_obligations:
        pairs_by_cell.setdefault(pair.target_cell_id, []).append(pair)
    extractor = AtomExtractor()
    registrations = _registrations()
    extractions = {
        cell.target_cell_id: _extract(cell, registrations, extractor)
        for cell in universe.fresh_cells
    }
    contract_digests = []
    for edge in universe.fresh_edges:
        selected_cells = (cells[edge.base_cell_id], cells[edge.sibling_cell_id])
        selected_pairs = (
            pairs_by_cell[edge.base_cell_id][0],
            pairs_by_cell[edge.sibling_cell_id][0],
        )
        contrast = merge_extractions(
            tuple(
                (cell.target_cell_id, extractions[cell.target_cell_id])
                for cell in selected_cells
            )
        )
        endpoints = _endpoints_for(selected_cells, selected_pairs, extractions)
        contract_digests.append(
            materialize_coverage_hypercontract(
                universe,
                cell_ids=tuple(item.target_cell_id for item in selected_cells),
                edge_ids=(edge.contrast_edge_id,),
                backend_pair_obligation_ids=tuple(
                    item.obligation_id for item in selected_pairs
                ),
                extraction=contrast,
                endpoints=endpoints,
            ).digest
        )
    assert len(contract_digests) == len(set(contract_digests)) == 384


def test_generic_fixture_contract_cannot_credit_real_target_pair(
    simple_compiled, simple_applicability, endpoints, int_schema
):
    fixture = _cell_fixture()
    contract, applicability, outcomes, _ = _generic_authority(
        simple_compiled, simple_applicability, endpoints, int_schema
    )
    pair_ids = (fixture.pair.obligation_id,)
    ledger = CoverageLedger(fixture.universe)
    _admit_prefix(ledger, fixture, pair_ids=pair_ids)
    event = _event(
        CoverageLevel.EXECUTED,
        2,
        fixture.activation,
        pair_ids=pair_ids,
        applicability=applicability,
        outcomes=outcomes,
    )
    with pytest.raises(ValueError, match="coverage observation policy"):
        ledger.admit(
            event,
            **_execution_kwargs(
                fixture,
                contract,
                applicability,
                outcomes,
                endpoints=endpoints,
            ),
        )


@pytest.mark.parametrize("credit_kind", ("cell", "edge", "one_sided_edge"))
def test_authority_credit_fails_closed_without_required_pair_contexts(credit_kind):
    edge_fixture = _edge_fixture()
    if credit_kind == "cell":
        fixture = _cell_fixture(edge_fixture)
        cell_ids = (fixture.cell.target_cell_id,)
        edge_ids = ()
        pair_ids = ()
        valid_pair_ids = (fixture.pair.obligation_id,)
        authority = _authority(
            fixture.universe,
            cell_ids=(fixture.cell.target_cell_id,),
            edge_ids=(),
            pair_ids=valid_pair_ids,
            extraction=fixture.extraction,
            endpoints=fixture.endpoints,
        )
    else:
        fixture = edge_fixture
        cell_ids = (fixture.base.target_cell_id, fixture.sibling.target_cell_id)
        edge_ids = (fixture.edge.contrast_edge_id,)
        pair_ids = (
            ()
            if credit_kind == "edge"
            else (fixture.pairs[0].obligation_id,)
        )
        valid_pair_ids = tuple(item.obligation_id for item in fixture.pairs)
        authority = _authority(
            fixture.universe,
            cell_ids=cell_ids,
            edge_ids=edge_ids,
            pair_ids=valid_pair_ids,
            extraction=fixture.contrast,
            endpoints=fixture.endpoints,
        )
    ledger = CoverageLedger(fixture.universe)
    _admit_prefix(
        ledger,
        fixture,
        cell_ids=cell_ids,
        edge_ids=edge_ids,
        pair_ids=pair_ids,
    )
    event = _event(
        CoverageLevel.EXECUTED,
        2,
        fixture.activation,
        cell_ids=cell_ids,
        edge_ids=edge_ids,
        pair_ids=pair_ids,
        applicability=authority[1],
        outcomes=authority[2],
    )
    with pytest.raises(ValueError, match="coverage observation policy"):
        ledger.admit(
            event,
            **_execution_kwargs(
                fixture, authority[0], authority[1], authority[2]
            ),
        )


@pytest.mark.parametrize("field,value", (("case_digest", "unrelated-case"), ("backend", "unrelated_backend")))
def test_unrelated_endpoint_case_or_backend_cannot_borrow_pair_authority(field, value):
    fixture = _cell_fixture()
    pair_ids = (fixture.pair.obligation_id,)
    authority = _authority(
        fixture.universe,
        cell_ids=(),
        edge_ids=(),
        pair_ids=pair_ids,
        extraction=fixture.extraction,
        endpoints=fixture.endpoints,
    )
    rebound = (replace(fixture.endpoints[0], **{field: value}), fixture.endpoints[1])
    ledger = CoverageLedger(fixture.universe)
    _admit_prefix(ledger, fixture, pair_ids=pair_ids)
    event = _event(
        CoverageLevel.EXECUTED,
        2,
        fixture.activation,
        pair_ids=pair_ids,
        applicability=authority[1],
        outcomes=authority[2],
    )
    with pytest.raises(ValueError, match="coverage observation policy"):
        ledger.admit(
            event,
            **_execution_kwargs(
                fixture,
                authority[0],
                authority[1],
                authority[2],
                endpoints=rebound,
            ),
        )


def test_materialized_contract_must_equal_the_unique_recomputed_object():
    fixture = _cell_fixture()
    pair_ids = (fixture.pair.obligation_id,)
    authority = _authority(
        fixture.universe,
        cell_ids=(),
        edge_ids=(),
        pair_ids=pair_ids,
        extraction=fixture.extraction,
        endpoints=fixture.endpoints,
    )
    forged_contract = replace(authority[0], derivation_digest="forged-derivation")
    ledger = CoverageLedger(fixture.universe)
    _admit_prefix(ledger, fixture, pair_ids=pair_ids)
    event = _event(
        CoverageLevel.EXECUTED,
        2,
        fixture.activation,
        pair_ids=pair_ids,
        applicability=authority[1],
        outcomes=authority[2],
    )
    with pytest.raises(ValueError, match="unique materialized HyperContract"):
        ledger.admit(
            event,
            **_execution_kwargs(
                fixture,
                forged_contract,
                authority[1],
                authority[2],
            ),
        )


def test_reversed_edge_with_retained_policy_cannot_reach_execution():
    fixture = _edge_fixture()
    reversed_edge = replace(
        fixture.edge,
        base_cell_id=fixture.edge.sibling_cell_id,
        sibling_cell_id=fixture.edge.base_cell_id,
    )
    forged_universe = replace(
        fixture.universe,
        fresh_edges=tuple(
            reversed_edge
            if item.contrast_edge_id == fixture.edge.contrast_edge_id
            else item
            for item in fixture.universe.fresh_edges
        ),
    )
    reversed_contrast = merge_extractions(
        tuple(
            (item.target_cell_id, item.extraction)
            for item in reversed(fixture.contrast.endpoint_extractions)
        )
    )
    assignment = TargetAssignment(
        (fixture.sibling.target_cell_id, fixture.base.target_cell_id),
        SeedLineage("p", 3, "lane", 0, SeedStage.TARGET),
        selected_edge_ids=(fixture.edge.contrast_edge_id,),
    )
    activation = TargetMatcher(forged_universe).match(
        assignment, reversed_contrast
    )
    forged_fixture = _EdgeFixture(
        forged_universe,
        fixture.sibling,
        fixture.base,
        reversed_edge,
        tuple(reversed(fixture.pairs)),
        reversed_contrast,
        tuple(reversed(fixture.endpoints)),
        assignment,
        activation,
    )
    cell_ids = assignment.selected_cell_ids
    edge_ids = (reversed_edge.contrast_edge_id,)
    pair_ids = tuple(item.obligation_id for item in forged_fixture.pairs)
    authority = _authority(
        fixture.universe,
        cell_ids=(fixture.base.target_cell_id, fixture.sibling.target_cell_id),
        edge_ids=(fixture.edge.contrast_edge_id,),
        pair_ids=tuple(item.obligation_id for item in fixture.pairs),
        extraction=fixture.contrast,
        endpoints=fixture.endpoints,
    )
    ledger = CoverageLedger(forged_universe)
    _admit_prefix(
        ledger,
        forged_fixture,
        cell_ids=cell_ids,
        edge_ids=edge_ids,
        pair_ids=pair_ids,
    )
    event = _event(
        CoverageLevel.EXECUTED,
        2,
        activation,
        cell_ids=cell_ids,
        edge_ids=edge_ids,
        pair_ids=pair_ids,
        applicability=authority[1],
        outcomes=authority[2],
    )
    with pytest.raises(ValueError, match="coverage observation policy"):
        ledger.admit(
            event,
            **_execution_kwargs(
                forged_fixture,
                authority[0],
                authority[1],
                authority[2],
                endpoints=fixture.endpoints,
            ),
        )


def test_ledger_rejects_skipped_levels_and_missing_real_objects():
    fixture = _cell_fixture()
    pair_ids = (fixture.pair.obligation_id,)
    ledger = CoverageLedger(fixture.universe)
    activated = _event(
        CoverageLevel.ACTIVATED,
        1,
        fixture.activation,
        pair_ids=pair_ids,
    )
    with pytest.raises(ValueError, match="lacks prior"):
        ledger.admit(activated, **_activation_kwargs(fixture))
    ledger.admit(
        _event(
            CoverageLevel.CONSTRUCTED,
            0,
            fixture.activation,
            pair_ids=pair_ids,
        )
    )
    with pytest.raises(ValueError, match="TargetAssignment"):
        ledger.admit(activated)


def test_ledger_rejects_forged_universe_assignment_and_atom_bindings():
    fixture = _cell_fixture()
    pair_ids = (fixture.pair.obligation_id,)
    ledger = CoverageLedger(fixture.universe)
    ledger.admit(
        _event(
            CoverageLevel.CONSTRUCTED,
            0,
            fixture.activation,
            pair_ids=pair_ids,
        )
    )
    forged = replace(
        fixture.activation,
        target_fingerprint=replace(
            fixture.activation.target_fingerprint,
            universe_digest="wrong-universe",
        ),
        observed_atoms=(),
    )
    activated = _event(
        CoverageLevel.ACTIVATED,
        1,
        forged,
        pair_ids=pair_ids,
    )
    with pytest.raises(ValueError, match="universe, assignment, or atoms"):
        ledger.admit(
            activated,
            target_assignment=fixture.assignment,
            extraction=fixture.extraction,
            activation_certificate=forged,
        )
    with pytest.raises(ValueError, match="universe, assignment, or atoms"):
        ledger.admit(
            _event(
                CoverageLevel.ACTIVATED,
                2,
                fixture.activation,
                pair_ids=pair_ids,
            ),
            target_assignment=replace(
                fixture.assignment,
                seed_lineage=SeedLineage(
                    "p", 4, "lane", 0, SeedStage.TARGET
                ),
            ),
            extraction=fixture.extraction,
            activation_certificate=fixture.activation,
        )


def test_executed_credit_rejects_missing_objects_and_non_ok_outcomes():
    fixture = _cell_fixture()
    pair_ids = (fixture.pair.obligation_id,)
    authority = _authority(
        fixture.universe,
        cell_ids=(),
        edge_ids=(),
        pair_ids=pair_ids,
        extraction=fixture.extraction,
        endpoints=fixture.endpoints,
    )
    ledger = CoverageLedger(fixture.universe)
    _admit_prefix(ledger, fixture, pair_ids=pair_ids)
    placeholder = _event(
        CoverageLevel.EXECUTED,
        2,
        fixture.activation,
        pair_ids=pair_ids,
    )
    with pytest.raises(ValueError, match="HyperContract"):
        ledger.admit(placeholder, **_activation_kwargs(fixture))

    timeout_outcomes = (
        StructuredExecutionOutcome(
            fixture.endpoints[0].endpoint_id,
            ExecutionStatus.TIMEOUT,
            FailureKind.TIMEOUT,
            reason="timeout",
        ),
        authority[2][1],
    )
    non_ok = _event(
        CoverageLevel.EXECUTED,
        3,
        fixture.activation,
        pair_ids=pair_ids,
        applicability=authority[1],
        outcomes=timeout_outcomes,
    )
    with pytest.raises(ValueError, match="non-OK"):
        ledger.admit(
            non_ok,
            **_execution_kwargs(
                fixture,
                authority[0],
                authority[1],
                timeout_outcomes,
            ),
        )


def test_observed_credit_rejects_misbound_observation_authority():
    fixture = _cell_fixture()
    pair_ids = (fixture.pair.obligation_id,)
    authority = _authority(
        fixture.universe,
        cell_ids=(),
        edge_ids=(),
        pair_ids=pair_ids,
        extraction=fixture.extraction,
        endpoints=fixture.endpoints,
    )
    ledger = CoverageLedger(fixture.universe)
    _admit_prefix(ledger, fixture, pair_ids=pair_ids)
    execution_kwargs = _execution_kwargs(
        fixture, authority[0], authority[1], authority[2]
    )
    ledger.admit(
        _event(
            CoverageLevel.EXECUTED,
            2,
            fixture.activation,
            pair_ids=pair_ids,
            applicability=authority[1],
            outcomes=authority[2],
        ),
        **execution_kwargs,
    )
    misbound = replace(
        authority[3], applicability_digest="another-applicability"
    )
    event = _event(
        CoverageLevel.OBSERVED,
        3,
        fixture.activation,
        pair_ids=pair_ids,
        applicability=authority[1],
        outcomes=authority[2],
        observation=misbound,
    )
    with pytest.raises(ValueError, match="authority-bound"):
        ledger.admit(
            event,
            **execution_kwargs,
            observation_certificate=misbound,
        )


def test_merge_is_order_independent_idempotent_and_universe_bound():
    fixture = _cell_fixture()
    pair_ids = (fixture.pair.obligation_id,)
    left = CoverageLedger(fixture.universe)
    right = CoverageLedger(fixture.universe)
    constructed = _event(
        CoverageLevel.CONSTRUCTED,
        0,
        fixture.activation,
        pair_ids=pair_ids,
    )
    activated = _event(
        CoverageLevel.ACTIVATED,
        1,
        fixture.activation,
        pair_ids=pair_ids,
    )
    left.admit(constructed)
    left.admit(activated, **_activation_kwargs(fixture))
    right.admit(constructed)
    merged_a = CoverageLedger.merge(fixture.universe, (left, right))
    merged_b = CoverageLedger.merge(fixture.universe, (right, left))
    assert [item.event_id for item in merged_a.events] == [
        item.event_id for item in merged_b.events
    ]
    assert merged_a.counts(CoverageLevel.ACTIVATED) == merged_b.counts(
        CoverageLevel.ACTIVATED
    )

    other = replace(fixture.universe, template_digest="another-template")
    with pytest.raises(ValueError, match="another universe"):
        CoverageLedger.merge(
            fixture.universe, (left, CoverageLedger(other))
        )
