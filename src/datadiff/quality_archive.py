from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

QUALITY_ARCHIVE_SCHEMA_VERSION = "quality-diversity-archive-v1"


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
    seeds: dict[int, ArchiveSeed] = field(default_factory=dict)
    reward_total: float = 0.0
    reward_count: int = 0
    outcome_count: int = 0
    invalid_count: int = 0
    fallback_count: int = 0
    false_positive_count: int = 0

    def record_seed(self, index: int, utility: float) -> None:
        seed = self.seeds.get(index)
        if seed is None:
            self.seeds[index] = ArchiveSeed(index=index, utility=float(utility))
            return
        seed.utility = max(seed.utility, float(utility))

    def remove_seed(self, index: int) -> None:
        self.seeds.pop(index, None)

    def record_pull(self, index: int) -> None:
        seed = self.seeds.get(index)
        if seed is not None:
            seed.pulls += 1

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

    def elite_indexes(self) -> list[int]:
        ranked = sorted(
            self.seeds.values(),
            key=lambda seed: (seed.quality_score(), -seed.pulls, -seed.index),
            reverse=True,
        )
        return [seed.index for seed in ranked[: max(0, self.max_elites)]]

    def seed_elite_bonus(self, index: int) -> float:
        elites = self.elite_indexes()
        if index not in elites:
            return 0.0
        rank = elites.index(index)
        return max(0.05, 0.25 / (1.0 + rank))

    def reward_signal(self) -> float:
        return _bounded_confident_mean_reward(self.reward_total, self.reward_count, max_abs=1.5)

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
            "seeds": [seed.to_state_dict() for seed in self.seeds.values()],
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
        return cell


@dataclass(slots=True)
class QualityDiversityArchive:
    max_elites_per_cluster: int = 4
    cells: dict[str, QualityDiversityCell] = field(default_factory=dict)

    def record_seed(self, cluster_key: str, index: int, utility: float) -> None:
        if not cluster_key:
            return
        self._cell(cluster_key).record_seed(index, utility)

    def remove_seed(self, cluster_key: str, index: int) -> None:
        if not cluster_key:
            return
        cell = self.cells.get(cluster_key)
        if cell is None:
            return
        cell.remove_seed(index)
        if not cell.seeds and cell.outcome_count <= 0:
            del self.cells[cluster_key]

    def retain_seeds(self, valid_indexes_by_cluster: dict[str, set[int]]) -> None:
        for cluster_key in list(self.cells):
            cell = self.cells[cluster_key]
            valid_indexes = valid_indexes_by_cluster.get(cluster_key, set())
            for index in list(cell.seeds):
                if index not in valid_indexes:
                    cell.remove_seed(index)
            if not cell.seeds and cell.outcome_count <= 0:
                del self.cells[cluster_key]

    def record_pull(self, cluster_key: str, index: int) -> None:
        if not cluster_key:
            return
        self._cell(cluster_key).record_pull(index)

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

    def seed_elite_bonus(self, cluster_key: str, index: int) -> float:
        cell = self.cells.get(cluster_key)
        if cell is None:
            return 0.0
        return cell.seed_elite_bonus(index)

    def cluster_reward_signal(self, cluster_key: str) -> float:
        cell = self.cells.get(cluster_key)
        if cell is None:
            return 0.0
        return cell.reward_signal()

    def cluster_health_penalty(self, cluster_key: str) -> float:
        cell = self.cells.get(cluster_key)
        if cell is None:
            return 0.0
        return cell.health_penalty()

    def elite_indexes(self, cluster_key: str) -> list[int]:
        cell = self.cells.get(cluster_key)
        if cell is None:
            return []
        return cell.elite_indexes()

    def has_cluster(self, cluster_key: str) -> bool:
        return cluster_key in self.cells

    def seed_count(self, cluster_key: str) -> int:
        cell = self.cells.get(cluster_key)
        if cell is None:
            return 0
        return len(cell.seeds)

    def outcome_count(self, cluster_key: str) -> int:
        cell = self.cells.get(cluster_key)
        if cell is None:
            return 0
        return cell.outcome_count

    def is_empty(self) -> bool:
        return not self.cells

    def to_state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": QUALITY_ARCHIVE_SCHEMA_VERSION,
            "max_elites_per_cluster": self.max_elites_per_cluster,
            "cells": [cell.to_state_dict() for cell in self.cells.values()],
        }

    @classmethod
    def from_state_dict(cls, data: dict[str, Any] | None) -> "QualityDiversityArchive":
        if not isinstance(data, dict):
            return cls()
        archive = cls(
            max_elites_per_cluster=int(data.get("max_elites_per_cluster", 4) or 4),
        )
        for raw_cell in data.get("cells", []) or []:
            if not isinstance(raw_cell, dict):
                continue
            cell = QualityDiversityCell.from_state_dict(raw_cell)
            if not cell.cluster_key:
                continue
            cell.max_elites = archive.max_elites_per_cluster
            archive.cells[cell.cluster_key] = cell
        return archive

    def _cell(self, cluster_key: str) -> QualityDiversityCell:
        cell = self.cells.get(cluster_key)
        if cell is None:
            cell = QualityDiversityCell(cluster_key=cluster_key, max_elites=self.max_elites_per_cluster)
            self.cells[cluster_key] = cell
        return cell


def _bounded_confident_mean_reward(total_reward: float, pulls: int, *, max_abs: float) -> float:
    if pulls <= 0:
        return 0.0
    mean_reward = float(total_reward) / float(pulls)
    bounded_mean_reward = max(-max_abs, min(max_abs, mean_reward))
    confidence = min(1.0, math.log1p(float(pulls)) / math.log(4.0))
    return bounded_mean_reward * confidence
