from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

from datadiff.behavioral_descriptor import BehavioralDescriptor

QUALITY_ARCHIVE_SCHEMA_VERSION = "quality-diversity-archive-v2"


@dataclass(slots=True)
class ArchiveSeed:
    index: int
    utility: float = 0.0
    pulls: int = 0
    reward_total: float = 0.0
    reward_count: int = 0

    def quality_score(self) -> float:
        reward_signal = _bounded_confident_mean_reward(self.reward_total, self.reward_count, max_abs=2.0)
        exploration_bonus = 0.20 / math.sqrt(1.0 + max(0, self.pulls))
        return float(self.utility) + reward_signal + exploration_bonus

    def to_state_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "utility": self.utility,
            "pulls": self.pulls,
            "reward_total": self.reward_total,
            "reward_count": self.reward_count,
        }

    @classmethod
    def from_state_dict(cls, data: dict[str, Any]) -> "ArchiveSeed":
        return cls(
            index=int(data.get("index", 0) or 0),
            utility=float(data.get("utility", 0.0) or 0.0),
            pulls=int(data.get("pulls", 0) or 0),
            reward_total=float(data.get("reward_total", 0.0) or 0.0),
            reward_count=int(data.get("reward_count", 0) or 0),
        )


@dataclass(slots=True)
class QualityDiversityCell:
    cluster_key: str
    max_elites: int = 4
    level: int = 0
    parent_key: str = ""
    split_axis: str = ""
    split_value: str = ""
    seeds: dict[int, ArchiveSeed] = field(default_factory=dict)
    seed_axes: dict[int, tuple[tuple[str, str], ...]] = field(default_factory=dict)
    children: dict[str, "QualityDiversityCell"] = field(default_factory=dict)
    reward_total: float = 0.0
    reward_count: int = 0
    outcome_count: int = 0
    invalid_count: int = 0
    fallback_count: int = 0
    false_positive_count: int = 0
    _elite_indexes_cache: tuple[int, ...] | None = field(default=None, init=False, repr=False)
    _elite_rank_cache: dict[int, int] | None = field(default=None, init=False, repr=False)

    def record_seed(self, index: int, utility: float) -> None:
        seed = self.seeds.get(index)
        if seed is None:
            self.seeds[index] = ArchiveSeed(index=index, utility=float(utility))
            self._invalidate_elite_cache()
            return
        next_utility = max(seed.utility, float(utility))
        if next_utility != seed.utility:
            seed.utility = next_utility
            self._invalidate_elite_cache()

    def remove_seed(self, index: int) -> None:
        if index in self.seeds:
            del self.seeds[index]
            self.seed_axes.pop(index, None)
            for child_key in list(self.children):
                child = self.children[child_key]
                child.remove_seed(index)
                if not child.seeds and child.outcome_count <= 0:
                    del self.children[child_key]
            self._invalidate_elite_cache()

    def record_seed_axes(self, index: int, axis_tuples: tuple[tuple[str, str], ...]) -> None:
        if not axis_tuples:
            return
        self.seed_axes[int(index)] = tuple(axis_tuples)

    def record_pull(self, index: int) -> None:
        seed = self.seeds.get(index)
        if seed is not None:
            seed.pulls += 1
            self._invalidate_elite_cache()

    def record_outcome(
        self,
        *,
        index: int | None,
        reward: float,
        preflight_valid: bool = True,
        fallback_used: bool = False,
        false_positive: bool = False,
    ) -> None:
        self.reward_total += float(reward)
        self.reward_count += 1
        self.outcome_count += 1
        if not preflight_valid:
            self.invalid_count += 1
        if fallback_used:
            self.fallback_count += 1
        if false_positive:
            self.false_positive_count += 1
        if index is None:
            return
        seed = self.seeds.get(index)
        if seed is None:
            return
        seed.reward_total += float(reward)
        seed.reward_count += 1
        self._invalidate_elite_cache()

    def elite_indexes(self) -> list[int]:
        self._ensure_elite_cache()
        return list(self._elite_indexes_cache or ())

    def seed_elite_bonus(self, index: int) -> float:
        self._ensure_elite_cache()
        rank = (self._elite_rank_cache or {}).get(index)
        if rank is None:
            return 0.0
        return max(0.05, 0.25 / (1.0 + rank))

    def _invalidate_elite_cache(self) -> None:
        self._elite_indexes_cache = None
        self._elite_rank_cache = None

    def _ensure_elite_cache(self) -> None:
        if self._elite_indexes_cache is not None and self._elite_rank_cache is not None:
            return
        ranked = sorted(
            self.seeds.values(),
            key=lambda seed: (seed.quality_score(), -seed.pulls, -seed.index),
            reverse=True,
        )
        elite_indexes = tuple(seed.index for seed in ranked[: max(0, self.max_elites)])
        self._elite_indexes_cache = elite_indexes
        self._elite_rank_cache = {seed_index: rank for rank, seed_index in enumerate(elite_indexes)}

    def reward_signal(self) -> float:
        return _bounded_confident_mean_reward(self.reward_total, self.reward_count, max_abs=1.5)

    def variance(self) -> float:
        scores = [
            seed.utility
            + _bounded_confident_mean_reward(seed.reward_total, seed.reward_count, max_abs=2.0)
            for seed in self.seeds.values()
        ]
        return _variance(scores)

    def child_reward_variance(self) -> float:
        if len(self.children) < 2:
            return 0.0
        return _variance([child.reward_signal() for child in self.children.values()])

    def health_penalty(self) -> float:
        if self.outcome_count <= 0:
            return 0.0
        invalid_rate = self.invalid_count / self.outcome_count
        fallback_rate = self.fallback_count / self.outcome_count
        false_positive_rate = self.false_positive_count / self.outcome_count
        return min(0.75, 0.25 * invalid_rate + 0.15 * fallback_rate + 0.50 * false_positive_rate)

    def to_state_dict(self) -> dict[str, Any]:
        return {
            "cluster_key": self.cluster_key,
            "max_elites": self.max_elites,
            "level": self.level,
            "parent_key": self.parent_key,
            "split_axis": self.split_axis,
            "split_value": self.split_value,
            "seeds": [seed.to_state_dict() for seed in self.seeds.values()],
            "seed_axes": {
                str(index): [
                    {"axis_name": axis_name, "axis_value": axis_value}
                    for axis_name, axis_value in axis_tuples
                ]
                for index, axis_tuples in sorted(self.seed_axes.items())
            },
            "children": [child.to_state_dict() for child in self.children.values()],
            "reward_total": self.reward_total,
            "reward_count": self.reward_count,
            "outcome_count": self.outcome_count,
            "invalid_count": self.invalid_count,
            "fallback_count": self.fallback_count,
            "false_positive_count": self.false_positive_count,
        }

    @classmethod
    def from_state_dict(cls, data: dict[str, Any]) -> "QualityDiversityCell":
        cell = cls(
            cluster_key=str(data.get("cluster_key", "") or ""),
            max_elites=int(data.get("max_elites", 4) or 4),
            level=max(0, int(data.get("level", 0) or 0)),
            parent_key=str(data.get("parent_key", "") or ""),
            split_axis=str(data.get("split_axis", "") or ""),
            split_value=str(data.get("split_value", "") or ""),
            reward_total=float(data.get("reward_total", 0.0) or 0.0),
            reward_count=int(data.get("reward_count", 0) or 0),
            outcome_count=int(data.get("outcome_count", 0) or 0),
            invalid_count=int(data.get("invalid_count", 0) or 0),
            fallback_count=int(data.get("fallback_count", 0) or 0),
            false_positive_count=int(data.get("false_positive_count", 0) or 0),
        )
        for raw_seed in data.get("seeds", []) or []:
            if isinstance(raw_seed, dict):
                seed = ArchiveSeed.from_state_dict(raw_seed)
                cell.seeds[seed.index] = seed
        raw_seed_axes = data.get("seed_axes", {}) or {}
        if isinstance(raw_seed_axes, dict):
            for raw_index, raw_axes in raw_seed_axes.items():
                try:
                    index = int(raw_index)
                except (TypeError, ValueError):
                    continue
                axes = _state_axis_tuples(raw_axes)
                if axes:
                    cell.seed_axes[index] = axes
        for raw_child in data.get("children", []) or []:
            if not isinstance(raw_child, dict):
                continue
            child = cls.from_state_dict(raw_child)
            if child.split_value:
                child_key = child.split_value
            else:
                child_key = _child_value_from_cluster_key(child.cluster_key)
            if child_key:
                cell.children[child_key] = child
        return cell


@dataclass(slots=True)
class QualityDiversityArchive:
    max_elites_per_cluster: int = 4
    enable_hierarchical: bool = True
    split_min_seeds: int = 4
    split_variance_threshold: float = 0.20
    merge_variance_threshold: float = 0.02
    max_hierarchy_depth: int = 1
    cells: dict[str, QualityDiversityCell] = field(default_factory=dict)
    axis_cells: dict[str, dict[str, QualityDiversityCell]] = field(default_factory=dict)

    def record_seed(self, cluster_key: str, index: int, utility: float) -> None:
        if not cluster_key:
            return
        self._cell(cluster_key).record_seed(index, utility)

    def record_seed_multi(
        self,
        cluster_key: str,
        descriptor: BehavioralDescriptor | dict[str, Any] | None,
        index: int,
        utility: float,
    ) -> None:
        self.record_seed(cluster_key, index, utility)
        axis_tuples = _descriptor_axis_tuples(descriptor)
        cell = self.cells.get(cluster_key)
        if cell is not None:
            cell.record_seed_axes(index, axis_tuples)
        for axis_name, axis_value in axis_tuples:
            self._axis_cell(axis_name, axis_value).record_seed(index, utility)
        if self.enable_hierarchical:
            self.maybe_split_cell(cluster_key)
            self._record_seed_in_child(cluster_key, axis_tuples, index, utility)

    def remove_seed(self, cluster_key: str, index: int) -> None:
        if not cluster_key:
            return
        cell = self.cells.get(cluster_key)
        if cell is None:
            return
        cell.remove_seed(index)
        if not cell.seeds and cell.outcome_count <= 0:
            del self.cells[cluster_key]
        self.remove_seed_from_axes(index)

    def remove_seed_from_axes(self, index: int) -> None:
        for axis_name in list(self.axis_cells):
            cells = self.axis_cells[axis_name]
            for axis_value in list(cells):
                cell = cells[axis_value]
                cell.remove_seed(index)
                if not cell.seeds and cell.outcome_count <= 0:
                    del cells[axis_value]
            if not cells:
                del self.axis_cells[axis_name]

    def retain_seeds(self, valid_indexes_by_cluster: dict[str, set[int]]) -> None:
        live_indexes: set[int] = set()
        for cluster_key in list(self.cells):
            cell = self.cells[cluster_key]
            valid_indexes = valid_indexes_by_cluster.get(cluster_key, set())
            live_indexes.update(valid_indexes)
            for index in list(cell.seeds):
                if index not in valid_indexes:
                    cell.remove_seed(index)
            if not cell.seeds and cell.outcome_count <= 0:
                del self.cells[cluster_key]
        self.retain_axis_seeds(live_indexes)

    def retain_axis_seeds(self, valid_indexes: set[int]) -> None:
        for axis_name in list(self.axis_cells):
            cells = self.axis_cells[axis_name]
            for axis_value in list(cells):
                cell = cells[axis_value]
                for index in list(cell.seeds):
                    if index not in valid_indexes:
                        cell.remove_seed(index)
                if not cell.seeds and cell.outcome_count <= 0:
                    del cells[axis_value]
            if not cells:
                del self.axis_cells[axis_name]

    def record_pull(self, cluster_key: str, index: int) -> None:
        if not cluster_key:
            return
        self._cell(cluster_key).record_pull(index)

    def record_pull_multi(
        self,
        cluster_key: str,
        descriptor: BehavioralDescriptor | dict[str, Any] | None,
        index: int,
    ) -> None:
        self.record_pull(cluster_key, index)
        for axis_name, axis_value in _descriptor_axis_tuples(descriptor):
            self._axis_cell(axis_name, axis_value).record_pull(index)

    def record_outcome(
        self,
        cluster_key: str,
        *,
        index: int | None,
        reward: float,
        preflight_valid: bool = True,
        fallback_used: bool = False,
        false_positive: bool = False,
    ) -> None:
        if not cluster_key:
            return
        self._cell(cluster_key).record_outcome(
            index=index,
            reward=reward,
            preflight_valid=preflight_valid,
            fallback_used=fallback_used,
            false_positive=false_positive,
        )

    def record_outcome_multi(
        self,
        cluster_key: str,
        descriptor: BehavioralDescriptor | dict[str, Any] | None,
        *,
        index: int | None,
        reward: float,
        preflight_valid: bool = True,
        fallback_used: bool = False,
        false_positive: bool = False,
    ) -> None:
        self.record_outcome(
            cluster_key,
            index=index,
            reward=reward,
            preflight_valid=preflight_valid,
            fallback_used=fallback_used,
            false_positive=false_positive,
        )
        axis_tuples = _descriptor_axis_tuples(descriptor)
        for axis_name, axis_value in axis_tuples:
            self._axis_cell(axis_name, axis_value).record_outcome(
                index=index,
                reward=reward,
                preflight_valid=preflight_valid,
                fallback_used=fallback_used,
                false_positive=false_positive,
            )
        if self.enable_hierarchical:
            self._record_outcome_in_child(
                cluster_key,
                axis_tuples,
                index=index,
                reward=reward,
                preflight_valid=preflight_valid,
                fallback_used=fallback_used,
                false_positive=false_positive,
            )
            self.maybe_split_cell(cluster_key)
            self.maybe_merge_siblings(cluster_key)

    def seed_elite_bonus(self, cluster_key: str, index: int) -> float:
        cell = self.cells.get(cluster_key)
        if cell is None:
            return 0.0
        bonus = cell.seed_elite_bonus(index)
        if self.enable_hierarchical and cell.split_axis:
            child = self._child_for_seed(cell, index)
            if child is not None:
                bonus = max(bonus, child.seed_elite_bonus(index))
        return bonus

    def cluster_reward_signal(self, cluster_key: str) -> float:
        cell = self.cells.get(cluster_key)
        if cell is None:
            return 0.0
        return cell.reward_signal()

    def composite_reward_signal(
        self,
        descriptor: BehavioralDescriptor | dict[str, Any] | None,
        *,
        axis_weights: dict[str, float] | None = None,
    ) -> float:
        total = 0.0
        weight_total = 0.0
        weights = axis_weights or {}
        for axis_name, axis_value in _descriptor_axis_tuples(descriptor):
            cell = self.axis_cells.get(axis_name, {}).get(axis_value)
            if cell is None:
                continue
            weight = max(0.0, float(weights.get(axis_name, 1.0)))
            if weight <= 0.0:
                continue
            total += weight * cell.reward_signal()
            weight_total += weight
        if weight_total <= 0.0:
            return 0.0
        return total / weight_total

    def cluster_health_penalty(self, cluster_key: str) -> float:
        cell = self.cells.get(cluster_key)
        if cell is None:
            return 0.0
        return cell.health_penalty()

    def elite_indexes(self, cluster_key: str) -> list[int]:
        cell = self.cells.get(cluster_key)
        if cell is None:
            return []
        if self.enable_hierarchical and cell.split_axis and cell.children:
            ranked_children = sorted(
                cell.children.values(),
                key=lambda child: (child.reward_signal(), len(child.seeds), child.cluster_key),
                reverse=True,
            )
            indexes: list[int] = []
            seen: set[int] = set()
            for child in ranked_children:
                for index in child.elite_indexes():
                    if index in seen:
                        continue
                    seen.add(index)
                    indexes.append(index)
                    if len(indexes) >= self.max_elites_per_cluster:
                        return indexes
            if indexes:
                return indexes
        return cell.elite_indexes()

    def has_cluster(self, cluster_key: str) -> bool:
        return cluster_key in self.cells

    def seed_count(self, cluster_key: str) -> int:
        cell = self.cells.get(cluster_key)
        if cell is None:
            return 0
        return len(cell.seeds)

    def axis_seed_count(self, axis_name: str, axis_value: str) -> int:
        cell = self.axis_cells.get(axis_name, {}).get(axis_value)
        if cell is None:
            return 0
        return len(cell.seeds)

    def hierarchical_cell_count(self, cluster_key: str | None = None) -> int:
        if cluster_key is not None:
            cell = self.cells.get(cluster_key)
            return _descendant_cell_count(cell) if cell is not None else 0
        return sum(_descendant_cell_count(cell) for cell in self.cells.values())

    def split_cell_count(self) -> int:
        return sum(1 for cell in self.cells.values() if cell.split_axis)

    def maybe_split_cell(self, cluster_key: str) -> bool:
        if not self.enable_hierarchical:
            return False
        cell = self.cells.get(cluster_key)
        if cell is None or cell.split_axis:
            return False
        if cell.level >= max(0, int(self.max_hierarchy_depth)):
            return False
        if len(cell.seeds) < max(2, int(self.split_min_seeds)):
            return False
        if cell.variance() < max(0.0, float(self.split_variance_threshold)):
            return False
        split_axis = self._best_split_axis(cell)
        if not split_axis:
            return False
        self._split_cell(cell, split_axis)
        return True

    def maybe_merge_siblings(self, cluster_key: str) -> bool:
        if not self.enable_hierarchical:
            return False
        cell = self.cells.get(cluster_key)
        if cell is None or not cell.split_axis or len(cell.children) < 2:
            return False
        if cell.child_reward_variance() > max(0.0, float(self.merge_variance_threshold)):
            return False
        if any(child.variance() >= self.split_variance_threshold for child in cell.children.values()):
            return False
        cell.children.clear()
        cell.split_axis = ""
        return True

    def outcome_count(self, cluster_key: str) -> int:
        cell = self.cells.get(cluster_key)
        if cell is None:
            return 0
        return cell.outcome_count

    def is_empty(self) -> bool:
        return not self.cells and not self.axis_cells

    def to_state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": QUALITY_ARCHIVE_SCHEMA_VERSION,
            "max_elites_per_cluster": self.max_elites_per_cluster,
            "enable_hierarchical": self.enable_hierarchical,
            "split_min_seeds": self.split_min_seeds,
            "split_variance_threshold": self.split_variance_threshold,
            "merge_variance_threshold": self.merge_variance_threshold,
            "max_hierarchy_depth": self.max_hierarchy_depth,
            "cells": [cell.to_state_dict() for cell in self.cells.values()],
            "axis_cells": {
                axis_name: [cell.to_state_dict() for cell in cells.values()]
                for axis_name, cells in sorted(self.axis_cells.items())
            },
        }

    @classmethod
    def from_state_dict(cls, data: dict[str, Any] | None) -> "QualityDiversityArchive":
        if not isinstance(data, dict):
            return cls()
        archive = cls(
            max_elites_per_cluster=int(data.get("max_elites_per_cluster", 4) or 4),
            enable_hierarchical=bool(data.get("enable_hierarchical", True)),
            split_min_seeds=max(2, int(data.get("split_min_seeds", 4) or 4)),
            split_variance_threshold=max(0.0, float(data.get("split_variance_threshold", 0.20) or 0.20)),
            merge_variance_threshold=max(0.0, float(data.get("merge_variance_threshold", 0.02) or 0.02)),
            max_hierarchy_depth=max(0, int(data.get("max_hierarchy_depth", 1) or 1)),
        )
        for raw_cell in data.get("cells", []) or []:
            if not isinstance(raw_cell, dict):
                continue
            cell = QualityDiversityCell.from_state_dict(raw_cell)
            if not cell.cluster_key:
                continue
            cell.max_elites = archive.max_elites_per_cluster
            archive.cells[cell.cluster_key] = cell
        raw_axis_cells = data.get("axis_cells", {}) or {}
        if isinstance(raw_axis_cells, dict):
            for axis_name, raw_cells in raw_axis_cells.items():
                axis_key = str(axis_name)
                if not axis_key:
                    continue
                for raw_cell in raw_cells or []:
                    if not isinstance(raw_cell, dict):
                        continue
                    cell = QualityDiversityCell.from_state_dict(raw_cell)
                    if not cell.cluster_key:
                        continue
                    cell.max_elites = archive.max_elites_per_cluster
                    archive.axis_cells.setdefault(axis_key, {})[cell.cluster_key] = cell
        return archive

    def _cell(self, cluster_key: str) -> QualityDiversityCell:
        cell = self.cells.get(cluster_key)
        if cell is None:
            cell = QualityDiversityCell(cluster_key=cluster_key, max_elites=self.max_elites_per_cluster)
            self.cells[cluster_key] = cell
        return cell

    def _axis_cell(self, axis_name: str, axis_value: str) -> QualityDiversityCell:
        cells = self.axis_cells.setdefault(str(axis_name), {})
        cell = cells.get(axis_value)
        if cell is None:
            cell = QualityDiversityCell(cluster_key=axis_value, max_elites=self.max_elites_per_cluster)
            cells[axis_value] = cell
        return cell

    def _record_seed_in_child(
        self,
        cluster_key: str,
        axis_tuples: tuple[tuple[str, str], ...],
        index: int,
        utility: float,
    ) -> None:
        cell = self.cells.get(cluster_key)
        if cell is None or not cell.split_axis:
            return
        axis_value = _axis_value(axis_tuples, cell.split_axis)
        if not axis_value:
            return
        child = self._child_cell(cell, axis_value)
        child.record_seed(index, utility)
        child.record_seed_axes(index, axis_tuples)

    def _record_outcome_in_child(
        self,
        cluster_key: str,
        axis_tuples: tuple[tuple[str, str], ...],
        *,
        index: int | None,
        reward: float,
        preflight_valid: bool,
        fallback_used: bool,
        false_positive: bool,
    ) -> None:
        cell = self.cells.get(cluster_key)
        if cell is None or not cell.split_axis:
            return
        axis_value = _axis_value(axis_tuples, cell.split_axis)
        if not axis_value and index is not None:
            child = self._child_for_seed(cell, index)
        elif axis_value:
            child = self._child_cell(cell, axis_value)
        else:
            child = None
        if child is None:
            return
        child.record_outcome(
            index=index,
            reward=reward,
            preflight_valid=preflight_valid,
            fallback_used=fallback_used,
            false_positive=false_positive,
        )

    def _best_split_axis(self, cell: QualityDiversityCell) -> str:
        axis_values: dict[str, set[str]] = {}
        for axis_tuples in cell.seed_axes.values():
            for axis_name, axis_value in axis_tuples:
                if not axis_name or not axis_value:
                    continue
                axis_values.setdefault(axis_name, set()).add(axis_value)
        ranked = sorted(
            (
                (len(values), axis_name)
                for axis_name, values in axis_values.items()
                if len(values) >= 2
            ),
            reverse=True,
        )
        return ranked[0][1] if ranked else ""

    def _split_cell(self, cell: QualityDiversityCell, split_axis: str) -> None:
        cell.split_axis = split_axis
        cell.children.clear()
        for index, seed in sorted(cell.seeds.items()):
            axis_value = _axis_value(cell.seed_axes.get(index, ()), split_axis)
            if not axis_value:
                continue
            child = self._child_cell(cell, axis_value)
            child.seeds[index] = ArchiveSeed.from_state_dict(seed.to_state_dict())
            child.seed_axes[index] = tuple(cell.seed_axes.get(index, ()))
            child.reward_total += seed.reward_total
            child.reward_count += seed.reward_count
            child.outcome_count += seed.reward_count
            child._invalidate_elite_cache()

    def _child_cell(self, cell: QualityDiversityCell, axis_value: str) -> QualityDiversityCell:
        child = cell.children.get(axis_value)
        if child is None:
            child = QualityDiversityCell(
                cluster_key=f"{cell.cluster_key}|{cell.split_axis}={axis_value}",
                max_elites=cell.max_elites,
                level=cell.level + 1,
                parent_key=cell.cluster_key,
                split_value=axis_value,
            )
            cell.children[axis_value] = child
        return child

    def _child_for_seed(self, cell: QualityDiversityCell, index: int) -> QualityDiversityCell | None:
        axis_value = _axis_value(cell.seed_axes.get(index, ()), cell.split_axis)
        if axis_value:
            return cell.children.get(axis_value)
        for child in cell.children.values():
            if index in child.seeds:
                return child
        return None


def _descriptor_axis_tuples(
    descriptor: BehavioralDescriptor | dict[str, Any] | None,
) -> tuple[tuple[str, str], ...]:
    if descriptor is None:
        return ()
    if isinstance(descriptor, BehavioralDescriptor):
        return descriptor.axis_tuples()
    if isinstance(descriptor, dict):
        return BehavioralDescriptor.from_dict(descriptor).axis_tuples()
    return ()


def _state_axis_tuples(raw_axes: Any) -> tuple[tuple[str, str], ...]:
    if not isinstance(raw_axes, list):
        return ()
    axes: list[tuple[str, str]] = []
    for raw_axis in raw_axes:
        if not isinstance(raw_axis, dict):
            continue
        axis_name = str(raw_axis.get("axis_name", "") or "").strip()
        axis_value = str(raw_axis.get("axis_value", "") or "").strip()
        if axis_name and axis_value:
            axes.append((axis_name, axis_value))
    return tuple(axes)


def _axis_value(axis_tuples: tuple[tuple[str, str], ...], axis_name: str) -> str:
    for candidate_axis, axis_value in axis_tuples:
        if candidate_axis == axis_name:
            return axis_value
    return ""


def _variance(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / len(values)


def _descendant_cell_count(cell: QualityDiversityCell | None) -> int:
    if cell is None:
        return 0
    return len(cell.children) + sum(_descendant_cell_count(child) for child in cell.children.values())


def _child_value_from_cluster_key(cluster_key: str) -> str:
    tail = str(cluster_key or "").rsplit("|", 1)[-1]
    if "=" not in tail:
        return ""
    return tail.split("=", 1)[1]


def _bounded_confident_mean_reward(total_reward: float, pulls: int, *, max_abs: float) -> float:
    if pulls <= 0:
        return 0.0
    mean_reward = float(total_reward) / float(pulls)
    bounded_mean_reward = max(-max_abs, min(max_abs, mean_reward))
    confidence = min(1.0, math.log1p(float(pulls)) / math.log(4.0))
    return bounded_mean_reward * confidence
