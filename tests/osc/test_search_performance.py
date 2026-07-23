from __future__ import annotations

from time import perf_counter

from datadiff.family_witness_registry import latest_family_witness_registrations

from datadiff_osc.generation.extraction import AtomExtractor
from datadiff_osc.schemas import SeedLineage, SeedStage
from datadiff_osc.scheduler.policy import ConstraintFirstScheduler, ScheduleCandidate
from datadiff_osc.search.ledger import CoverageLedger
from datadiff_osc.semantic_targets.compiler import compile_target_universe
from datadiff_osc.semantic_targets.declarations import legacy_v4_target_templates


def test_single_extraction_cache_boundary_is_constant_work_on_reuse():
    registration = next(
        item for item in latest_family_witness_registrations()
        if item.family_id == "polars_lazy_filter_groupby_window"
    )
    case = registration.generate_case(0)
    extractor = AtomExtractor()
    start = perf_counter()
    first = extractor.extract(case)
    for _ in range(1000):
        assert extractor.extract(case) is first
    elapsed = perf_counter() - start
    assert extractor.extraction_count == 1
    assert elapsed < 5.0


def test_bounded_scheduler_handles_large_candidate_batch_without_unbounded_archive():
    universe = compile_target_universe(legacy_v4_target_templates())
    scheduler = ConstraintFirstScheduler(
        universe,
        SeedLineage("perf", 37, "lane", 0, SeedStage.TARGET),
        pareto_bound=32,
    )
    cells = universe.fresh_cells
    candidates = tuple(
        ScheduleCandidate(
            f"candidate-{index:04d}",
            cell_ids=(cells[index % len(cells)].target_cell_id,),
            novelty=index % 17,
            semantic_risk=index % 7,
            estimated_cost=1 + index % 5,
        )
        for index in range(1000)
    )
    start = perf_counter()
    decision = scheduler.schedule(
        candidates,
        limit=32,
        ledger=CoverageLedger(universe),
    )
    elapsed = perf_counter() - start
    assert len(decision.candidate_ids) == 32
    assert decision.pareto_bound == 32
    assert not scheduler.graph_heat_enabled and not scheduler.archive_enabled
    assert elapsed < 10.0
