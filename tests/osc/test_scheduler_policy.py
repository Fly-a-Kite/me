from __future__ import annotations

from collections import Counter
from dataclasses import replace
import inspect

import pytest

from datadiff_osc.schemas import SeedLineage, SeedStage
from datadiff_osc.scheduler.policy import ConstraintFirstScheduler, ScheduleCandidate
from datadiff_osc.search.ledger import CoverageLedger
from datadiff_osc.semantic_targets.compiler import compile_target_universe
from datadiff_osc.semantic_targets.declarations import legacy_v4_target_templates


def _fixture():
    universe = compile_target_universe(legacy_v4_target_templates())
    lineage = SeedLineage("scheduler", 31, "lane", 0, SeedStage.TARGET)
    return (
        universe,
        ConstraintFirstScheduler(universe, lineage, pareto_bound=16),
        CoverageLedger(universe),
    )


def test_deterministic_coverage_floors_are_no_replacement_for_all_hard_views():
    universe, scheduler, _ledger = _fixture()
    for kind, expected in (("cells", 232), ("edges", 384), ("pairs", 502)):
        selected = [scheduler.floor_selection(kind, index).item_id for index in range(expected)]
        assert len(selected) == len(set(selected)) == expected
    assert set(
        scheduler.floor_selection("pairs", index).item_id for index in range(502)
    ) == {item.obligation_id for item in universe.fresh_backend_pair_obligations}


def test_coverage_floor_precedes_novelty_or_adaptive_score():
    universe, scheduler, ledger = _fixture()
    first = universe.fresh_cells[0]
    candidates = (
        ScheduleCandidate(
            "coverage-floor",
            cell_ids=(first.target_cell_id,),
            novelty=0,
            adaptive_shadow_score=0,
        ),
        ScheduleCandidate(
            "novel-adaptive-without-target",
            novelty=10_000,
            adaptive_shadow_score=10_000,
        ),
    )
    decision = scheduler.schedule(
        reversed(candidates),
        limit=1,
        ledger=ledger,
        enable_adaptive_shadow=True,
    )
    assert decision.candidate_ids == ("coverage-floor",)
    assert decision.coverage_floor_applied and decision.max_min_applied
    assert not decision.adaptive_shadow_used


def test_bounded_pareto_rejects_dominated_candidate_and_is_input_order_invariant():
    universe, scheduler, ledger = _fixture()
    cell = universe.fresh_cells[0]
    strong = ScheduleCandidate(
        "strong",
        cell_ids=(cell.target_cell_id,),
        novelty=4,
        semantic_risk=4,
        estimated_cost=1,
    )
    dominated = ScheduleCandidate(
        "dominated",
        cell_ids=(cell.target_cell_id,),
        novelty=1,
        semantic_risk=1,
        estimated_cost=3,
    )
    left = scheduler.schedule((strong, dominated), limit=1, ledger=ledger)
    right = scheduler.schedule((dominated, strong), limit=1, ledger=ledger)
    assert left.candidate_ids == right.candidate_ids == ("strong",)
    assert left.decision_digest == right.decision_digest


def test_scheduler_has_no_family_dispatch_or_verdict_authority():
    source = inspect.getsource(__import__("datadiff_osc.scheduler.policy", fromlist=["*"]))
    assert "evaluate_semantic_activation" not in source
    assert "Verdict" not in source
    _universe, scheduler, _ledger = _fixture()
    assert not scheduler.graph_heat_enabled
    assert not scheduler.archive_enabled


def test_scheduler_rejects_caller_invented_group_identity():
    universe, scheduler, ledger = _fixture()
    candidate = ScheduleCandidate(
        "forged-group",
        cell_ids=(universe.fresh_cells[0].target_cell_id,),
        group_ids=("unknown-group",),
    )
    with pytest.raises(ValueError, match="unknown|group"):
        scheduler.schedule(
            (candidate,),
            limit=1,
            ledger=ledger,
        )


def test_scheduler_recomputes_group_denominators_from_universe_and_ledger():
    universe, scheduler, ledger = _fixture()
    family_counts = Counter(item.test_family_id for item in universe.fresh_cells)
    family = next(name for name, total in family_counts.items() if total == 18)
    coverage = scheduler.group_coverage(ledger)
    assert coverage[f"family:cell:{family}"] == (0, 18)

    candidate = ScheduleCandidate(
        "fixed-denominator",
        cell_ids=(
            next(
                item.target_cell_id
                for item in universe.fresh_cells
                if item.test_family_id == family
            ),
        ),
    )
    with pytest.raises(TypeError, match="unexpected keyword"):
        scheduler.schedule(
            (candidate,),
            limit=1,
            ledger=ledger,
            group_coverage={f"family:cell:{family}": (1, 1)},
        )


def test_scheduler_rejects_ledger_from_another_universe():
    universe, scheduler, _ledger = _fixture()
    other = replace(universe, taxonomy_digest="another-taxonomy")
    with pytest.raises(ValueError, match="another target universe"):
        scheduler.schedule(
            (ScheduleCandidate("candidate"),),
            limit=1,
            ledger=CoverageLedger(other),
        )
