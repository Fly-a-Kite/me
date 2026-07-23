from __future__ import annotations

from datadiff_osc.generation.construction import (
    BackwardConstructor,
    FragmentEffect,
    TypedFragment,
    plan_backward,
)
from datadiff_osc.schemas import SeedLineage, SeedStage
from datadiff_osc.semantic_targets.compiler import compile_target_universe
from datadiff_osc.semantic_targets.declarations import legacy_v4_target_templates
from datadiff_osc.semantic_targets.model import TargetAssignment


def _fixture():
    templates = legacy_v4_target_templates()
    universe = compile_target_universe(templates)
    cell = universe.fresh_cells[0]
    assignment = TargetAssignment(
        (cell.target_cell_id,),
        SeedLineage("protocol", 17, "lane", cell.construction_index, SeedStage.CONSTRUCTOR),
    )
    return templates, universe, cell, assignment


def test_backward_planner_starts_from_oracle_tail_and_prunes_dominated_cost():
    leaf = TypedFragment("leaf", FragmentEffect(frozenset(), frozenset({"op:leaf"})), 1)
    expensive = TypedFragment("expensive", FragmentEffect(frozenset(), frozenset({"op:leaf"})), 9)
    tail = TypedFragment(
        "tail",
        FragmentEffect(frozenset({"op:leaf"}), frozenset({"oracle:exact"})),
        1,
        True,
    )
    plan = plan_backward(frozenset({"oracle:exact"}), (expensive, leaf, tail))
    assert plan is not None
    assert plan.fragment_ids == ("tail", "leaf")
    assert plan.cost == 2


def test_declaration_provider_is_verified_by_independent_matcher():
    templates, universe, cell, assignment = _fixture()
    constructor = BackwardConstructor(universe, templates)
    outcome = constructor.construct(assignment)
    assert outcome.successful
    assert outcome.certificate is not None and outcome.certificate.valid
    assert outcome.certificate.activated_cell_ids == (cell.target_cell_id,)
    assert outcome.plan is not None and outcome.plan.fragment_ids


def test_capability_failure_is_explicit_infeasible_evidence():
    templates, universe, cell, assignment = _fixture()
    outcome = BackwardConstructor(universe, templates).construct(
        assignment,
        supported_capabilities=frozenset(),
    )
    assert not outcome.successful
    assert outcome.infeasible is not None
    assert outcome.infeasible.reason_code == "unsupported_capabilities"
    assert set(outcome.infeasible.missing_requirements) == set(cell.required_capabilities)


def test_regression_cell_cannot_be_used_as_fresh_generation_seed():
    templates, universe, _cell, assignment = _fixture()
    regression = universe.regression_cells[0]
    assignment = TargetAssignment((regression.target_cell_id,), assignment.seed_lineage)
    outcome = BackwardConstructor(universe, templates).construct(assignment)
    assert not outcome.successful
    assert outcome.infeasible is not None
    assert outcome.infeasible.reason_code == "fresh_cell_required"
