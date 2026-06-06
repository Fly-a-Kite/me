from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

DISCOVERY_RATE_SCHEMA_VERSION = "discovery-rate-estimator-v1"


@dataclass(slots=True)
class DiscoveryRateEstimator:
    family_counts: Counter[str] = field(default_factory=Counter)
    total_observations: int = 0

    def observe(self, family_keys: Iterable[Any]) -> None:
        keys = [str(item).strip() for item in family_keys if str(item).strip()]
        if not keys:
            return
        for key in keys:
            self.family_counts[key] += 1
            self.total_observations += 1

    @property
    def unique_family_count(self) -> int:
        return len(self.family_counts)

    @property
    def singleton_family_count(self) -> int:
        return sum(1 for count in self.family_counts.values() if int(count) == 1)

    def unseen_probability(self) -> float:
        if self.total_observations <= 0:
            return 1.0
        return min(1.0, max(0.0, self.singleton_family_count / float(self.total_observations)))

    def adaptive_exploration_weight(self, base: float) -> float:
        base_weight = max(0.0, float(base or 0.0))
        if base_weight <= 0.0:
            return 0.0
        unseen = self.unseen_probability()
        return max(0.05, base_weight * (0.35 + (1.35 * unseen)))

    def bucket(self) -> str:
        unseen = self.unseen_probability()
        if unseen >= 0.75:
            return "very_high"
        if unseen >= 0.45:
            return "high"
        if unseen >= 0.20:
            return "medium"
        if unseen > 0.0:
            return "low"
        return "exhausted"

    def to_state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": DISCOVERY_RATE_SCHEMA_VERSION,
            "family_counts": dict(sorted(self.family_counts.items())),
            "total_observations": int(self.total_observations),
            "unique_family_count": self.unique_family_count,
            "singleton_family_count": self.singleton_family_count,
            "unseen_probability": self.unseen_probability(),
            "bucket": self.bucket(),
        }

    @classmethod
    def from_state_dict(cls, data: Mapping[str, Any] | None) -> "DiscoveryRateEstimator":
        if not isinstance(data, Mapping):
            return cls()
        estimator = cls(
            family_counts=Counter(
                {
                    str(key): int(value)
                    for key, value in (data.get("family_counts", {}) or {}).items()
                    if str(key) and int(value or 0) > 0
                }
            ),
            total_observations=int(data.get("total_observations", 0) or 0),
        )
        if estimator.total_observations <= 0:
            estimator.total_observations = sum(estimator.family_counts.values())
        return estimator
