from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

from datadiff.multi_objective import (
    CostVector,
    ObjectiveVector,
    SEED_FRONTIER_OBJECTIVE_SPEC,
    bounded_ratio,
    constrained_objective_score,
)

SeedScalar = Callable[[int], float]
SeedInt = Callable[[int], int]
SeedText = Callable[[int], str]
SeedDict = Callable[[int], dict[str, Any]]
ClusterScalar = Callable[[str], float]
AxisWeights = Callable[[int], dict[str, float]]


@dataclass(slots=True)
class SeedScoringContext:
    seed_schedule_reward: SeedScalar
    target_novelty_score: SeedScalar
    cluster_key_at: SeedText
    behavioral_descriptor_at: SeedDict
    archive_cluster_reward: ClusterScalar
    archive_axis_reward: Callable[[int], float]
    cluster_feedback_reward: ClusterScalar
    cluster_novelty_score: SeedScalar
    elite_bonus: SeedScalar
    lineage_rarity: SeedScalar
    archive_health_penalty: SeedScalar
    family_reuse_penalty: SeedScalar
    mutation_pulls: SeedInt
    recent_parent_pulls: SeedInt
    recent_cluster_pulls: Callable[[str], int]
    seed_energy: SeedInt


@dataclass(slots=True)
class SeedScorer:
    context: SeedScoringContext

    def schedule_score(self, index: int) -> float:
        reward_prior = self.context.seed_schedule_reward(index)
        target_novelty = self.context.target_novelty_score(index)
        cluster_key = self.context.cluster_key_at(index)
        cluster_reward = max(
            self.context.cluster_feedback_reward(cluster_key),
            self.context.archive_cluster_reward(cluster_key),
            self.context.archive_axis_reward(index),
        )
        cluster_novelty = self.context.cluster_novelty_score(index)
        elite_bonus = self.context.elite_bonus(index)
        lineage_rarity = self.context.lineage_rarity(index)
        archive_health_penalty = self.context.archive_health_penalty(index)
        family_reuse_penalty = self.context.family_reuse_penalty(index)
        mutation_pulls = self.context.mutation_pulls(index)
        recent_parent_pulls = self.context.recent_parent_pulls(index)
        recent_cluster_pulls = self.context.recent_cluster_pulls(cluster_key)
        seed_energy = self.context.seed_energy(index)
        discovery_signal = min(
            1.75,
            bounded_ratio(reward_prior, 4.0, upper=1.5)
            + 0.55 * bounded_ratio(cluster_reward, 1.5)
            + 0.30 * bounded_ratio(cluster_novelty, 1.25)
            + 0.25 * bounded_ratio(lineage_rarity, 1.5),
        )
        semantic_signal = min(1.35, target_novelty)
        structural_signal = min(
            1.25,
            0.50 * bounded_ratio(cluster_novelty, 1.25)
            + 0.30 * bounded_ratio(elite_bonus, 1.0)
            + 0.30 * bounded_ratio(lineage_rarity, 1.5)
            + 0.20 * bounded_ratio(seed_energy, 4.0),
        )
        vector = ObjectiveVector(
            discovery=discovery_signal,
            semantic=semantic_signal,
            novelty=min(1.0, 0.70 * bounded_ratio(cluster_novelty, 1.25) + 0.30 * bounded_ratio(target_novelty, 1.35)),
            expandability=min(1.0, 0.60 * bounded_ratio(seed_energy, 4.0) + 0.25 * bounded_ratio(lineage_rarity, 1.5)),
            structural_risk=structural_signal,
            coverage_gain=min(1.0, 0.55 * bounded_ratio(cluster_reward, 1.5) + 0.45 * bounded_ratio(elite_bonus, 1.0)),
        )
        cost = CostVector(
            invalidity=archive_health_penalty,
            redundancy=(
                bounded_ratio(float(mutation_pulls), 8.0)
                + bounded_ratio(float(recent_parent_pulls), 4.0)
                + bounded_ratio(float(recent_cluster_pulls), 4.0)
                + bounded_ratio(max(0.0, family_reuse_penalty - 1.0), 1.0)
            ),
            target_miss=0.0 if target_novelty > 0.0 else 0.18,
        )
        score = max(
            0.05,
            constrained_objective_score(
                vector,
                cost=cost,
                spec=SEED_FRONTIER_OBJECTIVE_SPEC,
                validity=1.0 - cost.invalidity,
                false_positive_risk=0.0,
            ),
        )
        energy_multiplier = 1.0 + (0.04 * max(0, seed_energy - 1))
        return score * energy_multiplier

    def frontier_priority(self, index: int) -> tuple[float, float, float, int, int, int]:
        cluster_key = self.context.cluster_key_at(index)
        return (
            -float(self.schedule_score(index)),
            -float(self.context.seed_schedule_reward(index)),
            -float(self.context.target_novelty_score(index)),
            int(self.context.mutation_pulls(index)),
            int(self.context.recent_parent_pulls(index)) + int(self.context.recent_cluster_pulls(cluster_key)),
            int(index),
        )

    def frontier_row(self, index: int) -> dict[str, Any]:
        cluster_key = self.context.cluster_key_at(index)
        reward_prior = self.context.seed_schedule_reward(index)
        target_novelty = self.context.target_novelty_score(index)
        cluster_reward = max(
            self.context.cluster_feedback_reward(cluster_key),
            self.context.archive_cluster_reward(cluster_key),
            self.context.archive_axis_reward(index),
        )
        return {
            "index": index,
            "case_id": "",
            "schedule_score": self.schedule_score(index),
            "seed_energy": self.context.seed_energy(index),
            "lineage_rarity": self.context.lineage_rarity(index),
            "reward_prior": reward_prior,
            "target_novelty_score": target_novelty,
            "cluster_reward": cluster_reward,
            "archive_axis_reward": self.context.archive_axis_reward(index),
            "cluster_novelty_score": self.context.cluster_novelty_score(index),
            "elite_bonus": self.context.elite_bonus(index),
            "archive_health_penalty": self.context.archive_health_penalty(index),
            "mutation_pulls": self.context.mutation_pulls(index),
            "recent_parent_pulls": self.context.recent_parent_pulls(index),
            "recent_cluster_pulls": self.context.recent_cluster_pulls(cluster_key),
            "cluster_key": cluster_key,
        }
