"""Constraint-first deterministic coverage scheduling.

Coverage floors and worst-group max-min debt are authority.  Pareto/novelty is
consulted only afterwards; adaptive feedback is an optional final tie-break and
never receives or returns an oracle verdict or bug classification.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable, Mapping

from datadiff_osc._canonical import stable_digest
from datadiff_osc.schemas import CoverageLevel, SeedLineage
from datadiff_osc.search.epochs import EpochSelection, NoReplacementEpoch
from datadiff_osc.search.ledger import CoverageLedger
from datadiff_osc.semantic_targets.model import CompiledTargetUniverse


@dataclass(frozen=True, slots=True)
class ScheduleCandidate:
    candidate_id: str
    cell_ids: tuple[str, ...] = ()
    edge_ids: tuple[str, ...] = ()
    backend_pair_ids: tuple[str, ...] = ()
    group_ids: tuple[str, ...] = ()
    estimated_cost: int = 1
    novelty: int = 0
    semantic_risk: int = 0
    adaptive_shadow_score: int = 0

    def __post_init__(self) -> None:
        if not self.candidate_id or self.estimated_cost < 1:
            raise ValueError("scheduler candidate identity/cost is invalid")
        for name, values in (
            ("cell IDs", self.cell_ids),
            ("edge IDs", self.edge_ids),
            ("backend-pair IDs", self.backend_pair_ids),
            ("group IDs", self.group_ids),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"candidate {name} must be unique")


@dataclass(frozen=True, slots=True)
class ScheduleDecision:
    candidate_ids: tuple[str, ...]
    decision_digest: str
    coverage_floor_applied: bool
    max_min_applied: bool
    pareto_bound: int
    adaptive_shadow_used: bool


@dataclass(frozen=True, slots=True)
class _Objectives:
    coverage_floor: int
    max_min_debt: Fraction
    cell_gain: int
    edge_gain: int
    pair_gain: int
    novelty: int
    semantic_risk: int
    negative_cost: int

    def values(self) -> tuple[Fraction, ...]:
        return (
            Fraction(self.coverage_floor),
            self.max_min_debt,
            Fraction(self.cell_gain),
            Fraction(self.edge_gain),
            Fraction(self.pair_gain),
            Fraction(self.novelty),
            Fraction(self.semantic_risk),
            Fraction(self.negative_cost),
        )


def _dominates(left: _Objectives, right: _Objectives) -> bool:
    left_values = left.values()
    right_values = right.values()
    return all(a >= b for a, b in zip(left_values, right_values, strict=True)) and any(
        a > b for a, b in zip(left_values, right_values, strict=True)
    )


class ConstraintFirstScheduler:
    graph_heat_enabled = False
    archive_enabled = False

    def __init__(
        self,
        universe: CompiledTargetUniverse,
        lineage: SeedLineage,
        *,
        pareto_bound: int = 64,
    ) -> None:
        if pareto_bound < 1:
            raise ValueError("Pareto bound must be positive")
        self.universe = universe
        self.lineage = lineage
        self.pareto_bound = int(pareto_bound)
        self._valid_cells = {item.target_cell_id for item in universe.fresh_cells}
        self._valid_edges = {item.contrast_edge_id for item in universe.fresh_edges}
        self._valid_pairs = {
            item.obligation_id for item in universe.fresh_backend_pair_obligations
        }
        self._group_members: dict[str, set[str]] = {}
        self._cell_groups: dict[str, frozenset[str]] = {}
        self._edge_groups: dict[str, frozenset[str]] = {}
        self._pair_groups: dict[str, frozenset[str]] = {}
        self._build_frozen_groups()
        self._floors = {
            "cells": NoReplacementEpoch(self._valid_cells, lineage, namespace="scheduler-cells"),
            "edges": NoReplacementEpoch(self._valid_edges, lineage, namespace="scheduler-edges"),
            "pairs": NoReplacementEpoch(self._valid_pairs, lineage, namespace="scheduler-pairs"),
        }

    def _register_groups(self, token: str, group_ids: Iterable[str]) -> frozenset[str]:
        frozen = frozenset(str(item) for item in group_ids)
        if not frozen or any(not item for item in frozen):
            raise ValueError("scheduler target must belong to frozen non-empty groups")
        for group_id in frozen:
            self._group_members.setdefault(group_id, set()).add(token)
        return frozen

    def _build_frozen_groups(self) -> None:
        cells = {item.target_cell_id: item for item in self.universe.fresh_cells}
        for cell in self.universe.fresh_cells:
            groups = {
                f"family:cell:{cell.test_family_id}",
                *(
                    f"axis:cell:{name}={value}"
                    for name, value in cell.coordinates
                ),
            }
            self._cell_groups[cell.target_cell_id] = self._register_groups(
                f"cell:{cell.target_cell_id}", groups
            )
        for edge in self.universe.fresh_edges:
            base = cells[edge.base_cell_id]
            groups = {
                f"family:edge:{base.test_family_id}",
                f"axis:edge:{edge.changed_axis}",
            }
            self._edge_groups[edge.contrast_edge_id] = self._register_groups(
                f"edge:{edge.contrast_edge_id}", groups
            )
        for pair in self.universe.fresh_backend_pair_obligations:
            cell = cells[pair.target_cell_id]
            groups = {
                f"family:pair:{cell.test_family_id}",
                f"backend-pair:{pair.target_backend}|{pair.control_backend}",
            }
            self._pair_groups[pair.obligation_id] = self._register_groups(
                f"pair:{pair.obligation_id}", groups
            )

    def floor_selection(self, kind: str, decision_index: int) -> EpochSelection:
        try:
            epoch = self._floors[kind]
        except KeyError as exc:
            raise ValueError(f"unknown scheduler floor kind: {kind}") from exc
        return epoch.select(decision_index)

    def _validate_candidate(self, candidate: ScheduleCandidate) -> None:
        unknown_cells = set(candidate.cell_ids) - self._valid_cells
        unknown_edges = set(candidate.edge_ids) - self._valid_edges
        unknown_pairs = set(candidate.backend_pair_ids) - self._valid_pairs
        if unknown_cells or unknown_edges or unknown_pairs:
            raise ValueError(
                "scheduler candidate contains unknown fresh targets: "
                f"cells={sorted(unknown_cells)} edges={sorted(unknown_edges)} "
                f"pairs={sorted(unknown_pairs)}"
            )

        derived = self._derived_group_ids(candidate)
        if candidate.group_ids:
            supplied = frozenset(candidate.group_ids)
            unknown_groups = supplied - set(self._group_members)
            if unknown_groups:
                raise ValueError(
                    "scheduler candidate contains unknown groups: "
                    f"{sorted(unknown_groups)}"
                )
            if supplied != derived:
                raise ValueError(
                    "scheduler candidate group membership does not match frozen universe"
                )

    def _derived_group_ids(self, candidate: ScheduleCandidate) -> frozenset[str]:
        return frozenset(
            group_id
            for target_id in candidate.cell_ids
            for group_id in self._cell_groups.get(target_id, ())
        ) | frozenset(
            group_id
            for target_id in candidate.edge_ids
            for group_id in self._edge_groups.get(target_id, ())
        ) | frozenset(
            group_id
            for target_id in candidate.backend_pair_ids
            for group_id in self._pair_groups.get(target_id, ())
        )

    def _validate_ledger(self, ledger: CoverageLedger) -> None:
        if not isinstance(ledger, CoverageLedger):
            raise TypeError("scheduler requires a real CoverageLedger")
        if ledger.universe.digest != self.universe.digest:
            raise ValueError("scheduler ledger belongs to another target universe")

    def _group_coverage_from_ids(
        self,
        cells: set[str],
        edges: set[str],
        pairs: set[str],
    ) -> dict[str, tuple[int, int]]:
        observed_tokens = {
            *(f"cell:{item}" for item in cells),
            *(f"edge:{item}" for item in edges),
            *(f"pair:{item}" for item in pairs),
        }
        return {
            group_id: (len(members & observed_tokens), len(members))
            for group_id, members in sorted(self._group_members.items())
        }

    def group_coverage(self, ledger: CoverageLedger) -> dict[str, tuple[int, int]]:
        """Recompute every fixed denominator from this universe and its ledger."""

        self._validate_ledger(ledger)
        return self._group_coverage_from_ids(
            set(ledger.credited_ids(CoverageLevel.OBSERVED, "cells")),
            set(ledger.credited_ids(CoverageLevel.OBSERVED, "edges")),
            set(ledger.credited_ids(CoverageLevel.OBSERVED, "pairs")),
        )

    @staticmethod
    def _group_debt(
        group_ids: Iterable[str],
        group_coverage: Mapping[str, tuple[int, int]],
    ) -> Fraction:
        debts: list[Fraction] = []
        for group_id in group_ids:
            if group_id not in group_coverage:
                raise ValueError(f"unknown frozen scheduler group: {group_id}")
            observed, total = group_coverage[group_id]
            if total < 1 or observed < 0 or observed > total:
                raise ValueError(f"invalid group coverage for {group_id}")
            debts.append(Fraction(total - observed, total))
        return max(debts, default=Fraction(0))

    def _objectives(
        self,
        candidate: ScheduleCandidate,
        observed_cells: set[str],
        observed_edges: set[str],
        observed_pairs: set[str],
        group_coverage: Mapping[str, tuple[int, int]],
    ) -> _Objectives:
        cell_gain = len(set(candidate.cell_ids) - observed_cells)
        edge_gain = len(set(candidate.edge_ids) - observed_edges)
        pair_gain = len(set(candidate.backend_pair_ids) - observed_pairs)
        return _Objectives(
            coverage_floor=int(cell_gain + edge_gain + pair_gain > 0),
            max_min_debt=self._group_debt(
                self._derived_group_ids(candidate), group_coverage
            ),
            cell_gain=cell_gain,
            edge_gain=edge_gain,
            pair_gain=pair_gain,
            novelty=max(0, int(candidate.novelty)),
            semantic_risk=max(0, int(candidate.semantic_risk)),
            negative_cost=-candidate.estimated_cost,
        )

    def _frontier(
        self,
        scored: list[tuple[ScheduleCandidate, _Objectives]],
    ) -> list[tuple[ScheduleCandidate, _Objectives]]:
        frontier: list[tuple[ScheduleCandidate, _Objectives]] = []
        ordered = sorted(
            scored,
            key=lambda item: (
                tuple(-value for value in item[1].values()),
                stable_digest("osc-scheduler-candidate-order", item[0].candidate_id),
            ),
        )
        for candidate, objectives in ordered:
            if any(_dominates(existing, objectives) for _item, existing in frontier):
                continue
            frontier = [
                (item, existing)
                for item, existing in frontier
                if not _dominates(objectives, existing)
            ]
            frontier.append((candidate, objectives))
            frontier.sort(
                key=lambda item: (
                    tuple(-value for value in item[1].values()),
                    stable_digest("osc-scheduler-frontier-order", item[0].candidate_id),
                )
            )
            if len(frontier) > self.pareto_bound:
                frontier = frontier[: self.pareto_bound]
        return frontier

    def schedule(
        self,
        candidates: Iterable[ScheduleCandidate],
        *,
        limit: int,
        ledger: CoverageLedger,
        enable_adaptive_shadow: bool = False,
    ) -> ScheduleDecision:
        if limit < 0:
            raise ValueError("schedule limit must be non-negative")
        materialized = tuple(candidates)
        remaining = {item.candidate_id: item for item in materialized}
        if len(remaining) != len(materialized):
            raise ValueError("scheduler candidate identities must be unique")
        for candidate in remaining.values():
            self._validate_candidate(candidate)
        self._validate_ledger(ledger)
        initial_cells = ledger.credited_ids(CoverageLevel.OBSERVED, "cells")
        initial_edges = ledger.credited_ids(CoverageLevel.OBSERVED, "edges")
        initial_pairs = ledger.credited_ids(CoverageLevel.OBSERVED, "pairs")
        cells = set(initial_cells)
        edges = set(initial_edges)
        pairs = set(initial_pairs)
        if not cells <= self._valid_cells or not edges <= self._valid_edges or not pairs <= self._valid_pairs:
            raise ValueError("observed scheduler state contains unknown targets")
        selected: list[str] = []
        floor_applied = False
        max_min_applied = False
        adaptive_used = False
        while remaining and len(selected) < limit:
            groups = self._group_coverage_from_ids(cells, edges, pairs)
            scored = [
                (candidate, self._objectives(candidate, cells, edges, pairs, groups))
                for candidate in remaining.values()
            ]
            frontier = self._frontier(scored)
            if not frontier:
                break
            best_objectives = max((item[1].values() for item in frontier))
            tied = [item for item in frontier if item[1].values() == best_objectives]
            if enable_adaptive_shadow and len(tied) > 1:
                top_adaptive = max(item[0].adaptive_shadow_score for item in tied)
                narrowed = [item for item in tied if item[0].adaptive_shadow_score == top_adaptive]
                adaptive_used |= len(narrowed) < len(tied)
                tied = narrowed
            candidate, objectives = min(
                tied,
                key=lambda item: stable_digest(
                    "osc-scheduler-final-tie", item[0].candidate_id
                ),
            )
            selected.append(candidate.candidate_id)
            floor_applied |= bool(objectives.coverage_floor)
            max_min_applied |= bool(objectives.max_min_debt)
            cells.update(candidate.cell_ids)
            edges.update(candidate.edge_ids)
            pairs.update(candidate.backend_pair_ids)
            remaining.pop(candidate.candidate_id)

        payload = {
            "candidate_ids": selected,
            "initial_observed_cells": sorted(initial_cells),
            "initial_observed_edges": sorted(initial_edges),
            "initial_observed_pairs": sorted(initial_pairs),
            "group_coverage": sorted(self.group_coverage(ledger).items()),
            "universe_digest": self.universe.digest,
            "limit": limit,
            "pareto_bound": self.pareto_bound,
            "adaptive_shadow_used": adaptive_used,
        }
        return ScheduleDecision(
            candidate_ids=tuple(selected),
            decision_digest=stable_digest("osc-schedule-decision", payload),
            coverage_floor_applied=floor_applied,
            max_min_applied=max_min_applied,
            pareto_bound=self.pareto_bound,
            adaptive_shadow_used=adaptive_used,
        )


__all__ = ["ConstraintFirstScheduler", "ScheduleCandidate", "ScheduleDecision"]
