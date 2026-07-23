from __future__ import annotations

from dataclasses import replace

import pytest

from datadiff.family_witness_registry import latest_family_witness_registrations

from datadiff_osc.generation.extraction import (
    AtomExtractor,
    ExtractionResult,
    merge_extractions,
)
from datadiff_osc.schemas import AtomProvenance, SeedLineage, SeedStage, SemanticAtom
from datadiff_osc.search.matcher import TargetMatcher
from datadiff_osc.semantic_targets.compiler import compile_target_universe
from datadiff_osc.semantic_targets.declarations import legacy_v4_target_templates
from datadiff_osc.semantic_targets.model import ContrastRelation, TargetAssignment


def _lineage():
    return SeedLineage("protocol", 11, "lane", 0, SeedStage.TARGET)


def _first_case_and_cell():
    universe = compile_target_universe(legacy_v4_target_templates())
    cell = next(item for item in universe.fresh_cells if item.test_family_id == "pandas_nullable_bool_reduction")
    registration = next(
        item for item in latest_family_witness_registrations()
        if item.family_id == cell.test_family_id
    )
    return universe, cell, registration.generate_case(cell.construction_index)


def test_exact_matcher_builds_fingerprint_bound_activation_certificate():
    universe, cell, case = _first_case_and_cell()
    extraction = AtomExtractor().extract(case)
    assignment = TargetAssignment((cell.target_cell_id,), _lineage())
    certificate = TargetMatcher(universe).match(assignment, extraction)
    assert certificate.valid
    assert certificate.assignment_digest == assignment.digest
    assert certificate.target_fingerprint.universe_digest == universe.digest
    assert certificate.activated_cell_ids == (cell.target_cell_id,)


def test_missing_atom_and_mutation_flag_fail_activation():
    universe, cell, case = _first_case_and_cell()
    extraction = AtomExtractor().extract(case)
    required = next(iter(cell.required_all_atoms))
    reduced = replace(
        extraction,
        atoms=tuple(item for item in extraction.atoms if item.atom_id != required),
    )
    assignment = TargetAssignment((cell.target_cell_id,), _lineage())
    missing = TargetMatcher(universe).match(assignment, reduced)
    assert not missing.valid
    assert required in missing.missing_atoms
    mutated = TargetMatcher(universe).match(assignment, extraction, mutation_preserved=False)
    assert not mutated.valid


def test_matcher_rejects_assignment_self_evidence():
    universe, cell, case = _first_case_and_cell()
    extraction = AtomExtractor().extract(case)
    forged_provenance = AtomProvenance("target_assignment", "s", "selected", "e")
    forged = SemanticAtom("axis:forged=yes", "axis", "forged=yes", (forged_provenance,))
    extraction = replace(extraction, atoms=(*extraction.atoms, forged))
    with pytest.raises(ValueError, match="cannot provide activation evidence"):
        TargetMatcher(universe).match(
            TargetAssignment((cell.target_cell_id,), _lineage()), extraction
        )


def test_match_cell_enforces_all_any_and_none_clauses_exactly():
    universe, cell, _case = _first_case_and_cell()
    del universe
    cell = replace(
        cell,
        required_all_atoms=frozenset({"op:filter"}),
        required_any_atom_groups=(frozenset({"agg:sum", "agg:count"}),),
        forbidden_atoms=frozenset({"op:limit"}),
    )
    assert TargetMatcher.match_cell(
        cell, frozenset({"op:filter", "agg:count"})
    ).matched
    missing_any = TargetMatcher.match_cell(cell, frozenset({"op:filter"}))
    assert not missing_any.matched and missing_any.missing_any_groups
    forbidden = TargetMatcher.match_cell(
        cell, frozenset({"op:filter", "agg:sum", "op:limit"})
    )
    assert not forbidden.matched and forbidden.observed_forbidden == ("op:limit",)


def _directional_edge_fixture():
    universe = compile_target_universe(legacy_v4_target_templates())
    cells = {item.target_cell_id: item for item in universe.fresh_cells}
    edge = next(
        item
        for item in universe.fresh_edges
        if item.relation is ContrastRelation.MONOTONIC_BOUNDARY
    )
    registrations = {
        item.family_id: item for item in latest_family_witness_registrations()
    }
    extractor = AtomExtractor()
    base_cell = cells[edge.base_cell_id]
    sibling_cell = cells[edge.sibling_cell_id]
    base = extractor.extract(
        registrations[base_cell.test_family_id].generate_case(
            base_cell.construction_index
        )
    )
    sibling = extractor.extract(
        registrations[sibling_cell.test_family_id].generate_case(
            sibling_cell.construction_index
        )
    )
    assignment = TargetAssignment(
        (edge.base_cell_id, edge.sibling_cell_id),
        _lineage(),
        selected_edge_ids=(edge.contrast_edge_id,),
    )
    return universe, cells, edge, assignment, base, sibling


def test_directional_edge_binds_base_sibling_and_rejects_swaps_or_atom_borrowing():
    universe, _cells, edge, assignment, base, sibling = _directional_edge_fixture()
    matcher = TargetMatcher(universe)
    proper = merge_extractions(
        ((edge.base_cell_id, base), (edge.sibling_cell_id, sibling))
    )
    certificate = matcher.match(assignment, proper)
    assert certificate.valid
    assert any(
        fact.startswith("contrast_edge_binding:")
        for fact in certificate.static_facts
    )

    swapped = merge_extractions(
        ((edge.sibling_cell_id, sibling), (edge.base_cell_id, base))
    )
    with pytest.raises(ValueError, match="direction"):
        matcher.match(assignment, swapped)

    borrowed = merge_extractions(
        ((edge.base_cell_id, sibling), (edge.sibling_cell_id, sibling))
    )
    borrowed_certificate = matcher.match(assignment, borrowed)
    assert not borrowed_certificate.valid
    assert edge.base_cell_id not in borrowed_certificate.activated_cell_ids


def test_directional_edge_requires_both_endpoints_to_activate_independently():
    universe, cells, edge, assignment, base, sibling = _directional_edge_fixture()
    required = cells[edge.sibling_cell_id].required_all_atoms & sibling.atom_ids
    assert required
    removed = next(iter(required))
    one_sided = replace(
        sibling,
        atoms=tuple(item for item in sibling.atoms if item.atom_id != removed),
    )
    evidence = merge_extractions(
        ((edge.base_cell_id, base), (edge.sibling_cell_id, one_sided))
    )
    certificate = TargetMatcher(universe).match(assignment, evidence)
    assert not certificate.valid
    assert edge.sibling_cell_id not in certificate.activated_cell_ids
