from __future__ import annotations

from datadiff.family_witness_registry import latest_family_witness_registrations

from datadiff_osc.generation.extraction import AtomExtractor
from datadiff_osc.generation.mutation import TargetPreservingMutator
from datadiff_osc.schemas import SeedLineage, SeedStage
from datadiff_osc.search.matcher import TargetMatcher
from datadiff_osc.semantic_targets.compiler import compile_target_universe
from datadiff_osc.semantic_targets.declarations import legacy_v4_target_templates
from datadiff_osc.semantic_targets.model import TargetAssignment


def _fixture():
    universe = compile_target_universe(legacy_v4_target_templates())
    cell = next(item for item in universe.fresh_cells if item.test_family_id == "polars_lazy_filter_groupby_window" and item.coordinate_map["data_pattern"] == "balanced")
    registration = next(item for item in latest_family_witness_registrations() if item.family_id == cell.test_family_id)
    case = registration.generate_case(cell.construction_index)
    assignment = TargetAssignment(
        (cell.target_cell_id,),
        SeedLineage("protocol", 19, "lane", cell.construction_index, SeedStage.TARGET),
    )
    extractor = AtomExtractor()
    matcher = TargetMatcher(universe)
    assert matcher.match(assignment, extractor.extract(case)).valid
    return case, assignment, TargetPreservingMutator(matcher, extractor=extractor)


def test_mutation_is_accepted_only_after_reextract_and_rematch():
    case, assignment, mutator = _fixture()

    def add_balanced_rows(candidate, _seed):
        candidate.tables[0].rows.extend(
            [
                {"id": 5, "g": "a", "x": 4.0},
                {"id": 6, "g": "b", "x": 5.0},
            ]
        )
        return candidate

    outcome = mutator.mutate(case, assignment, add_balanced_rows)
    assert outcome.accepted
    assert outcome.status == "accepted"
    assert outcome.certificate.mutation_preserved


def test_generic_program_repair_must_preserve_a_non_noop_mutation():
    case, assignment, mutator = _fixture()

    def remove_program_and_change_data(candidate, _seed):
        candidate.program.operations.clear()
        candidate.tables[0].rows[0]["x"] = 9.0
        return candidate

    outcome = mutator.mutate(case, assignment, remove_program_and_change_data)
    assert outcome.accepted
    assert outcome.status == "repaired"
    assert outcome.attempts == 2


def test_semantic_noop_repair_or_unpreserved_target_is_rejected():
    case, assignment, mutator = _fixture()

    def erase_program(candidate, _seed):
        candidate.program.operations.clear()
        return candidate

    outcome = mutator.mutate(case, assignment, erase_program)
    assert not outcome.accepted
    assert outcome.status == "rejected"
    assert not outcome.certificate.mutation_preserved
