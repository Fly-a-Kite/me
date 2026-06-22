from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from datadiff.energy import seed_energy

SEED_ENERGY_TIERS = ("low", "med", "high")
SEED_ENERGY_TIER_MULTIPLIERS = {
    "low": 0.75,
    "med": 1.0,
    "high": 1.35,
}


@dataclass(slots=True)
class SeedBudgetAllocator:
    tier_selector: Callable[[int], str]
    min_energy: int = 1
    max_energy: int = 16

    def energy_for_seed(
        self,
        index: int,
        *,
        pulls: int,
        mean_reward: float,
        family_breadth: int,
        cluster_outcome_count: int,
        since_last_finding_pulls: int,
    ) -> int:
        base_energy = seed_energy(
            pulls=pulls,
            mean_reward=mean_reward,
            family_breadth=family_breadth,
            cluster_outcome_count=cluster_outcome_count,
            since_last_finding_pulls=since_last_finding_pulls,
        )
        tier = self.normalized_tier(self.tier_selector(int(index)))
        multiplier = SEED_ENERGY_TIER_MULTIPLIERS.get(tier, 1.0)
        return self.bounded_energy(int(round(float(base_energy) * multiplier)))

    def normalized_tier(self, tier: str) -> str:
        return tier if tier in SEED_ENERGY_TIERS else "med"

    def bounded_energy(self, value: int) -> int:
        lower = max(1, int(self.min_energy))
        upper = max(lower, int(self.max_energy))
        return min(upper, max(lower, int(value)))
