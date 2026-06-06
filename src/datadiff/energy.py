from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class EnergyParameters:
    base_energy: float = 1.0
    family_diversity_weight: float = 0.5
    cold_seed_bonus: float = 1.5
    stale_penalty: float = 0.4
    max_energy: int = 16
    min_energy: int = 1


def seed_energy(
    *,
    pulls: int,
    mean_reward: float,
    family_breadth: int,
    cluster_outcome_count: int,
    since_last_finding_pulls: int,
    params: EnergyParameters | None = None,
) -> int:
    p = params or EnergyParameters()
    raw = max(0.0, float(p.base_energy))
    raw += max(0.0, float(p.cold_seed_bonus)) / math.sqrt(1.0 + max(0, int(pulls)))
    raw += max(0.0, float(mean_reward))
    raw += max(0.0, float(p.family_diversity_weight)) * math.log1p(max(0, int(family_breadth)))
    raw += 0.25 * math.log1p(max(0, int(cluster_outcome_count)))
    raw -= max(0.0, float(p.stale_penalty)) * math.log1p(max(0, int(since_last_finding_pulls)))
    return _bounded_energy(raw, p)


def operator_energy(
    *,
    pulls: int,
    mean_reward: float,
    recent_unproductive_streak: int,
    catalog_width: int,
    params: EnergyParameters | None = None,
) -> int:
    p = params or EnergyParameters()
    raw = max(0.0, float(p.base_energy))
    raw += max(0.0, float(mean_reward))
    raw += 0.20 * math.log1p(max(0, int(catalog_width)))
    raw += max(0.0, float(p.cold_seed_bonus)) / math.sqrt(1.0 + max(0, int(pulls)))
    raw -= max(0.0, float(p.stale_penalty)) * max(0, int(recent_unproductive_streak))
    return _bounded_energy(raw, p)


def cost_normalized_reward(reward: float, elapsed_s: float, *, floor_s: float = 0.05) -> float:
    return float(reward) / max(float(floor_s), float(elapsed_s or 0.0))


def _bounded_energy(value: float, params: EnergyParameters) -> int:
    lower = max(1, int(params.min_energy))
    upper = max(lower, int(params.max_energy))
    return min(upper, max(lower, int(round(float(value)))))
