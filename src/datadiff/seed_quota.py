from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

SEED_QUOTA_SCHEMA_VERSION = "seed-quota-v1"
UNKNOWN_CLUSTER_KEY = "__unknown__"


@dataclass(slots=True)
class SeedQuotaManager:
    """Quota-aware corpus eviction over behavioral cluster cells."""

    enabled: bool = True
    min_quota_per_active_cell: int = 1
    pull_decay: float = 1.0
    elite_protection_bonus: float = 0.35

    def quota_for_cell(
        self,
        cluster_key: str,
        *,
        total_capacity: int,
        active_cell_count: int,
        cell_scores: Mapping[str, float] | None = None,
    ) -> int:
        if not self.enabled:
            return 0
        key = _cluster_key(cluster_key)
        if cell_scores:
            return self.cell_quotas(cell_scores, total_capacity=total_capacity).get(key, 0)
        if total_capacity <= 0 or active_cell_count <= 0:
            return 0
        if active_cell_count <= total_capacity:
            return max(1, int(self.min_quota_per_active_cell))
        return 0

    def cell_quotas(
        self,
        cell_scores: Mapping[str, float],
        *,
        total_capacity: int,
    ) -> dict[str, int]:
        keys = sorted(_cluster_key(key) for key in cell_scores if _cluster_key(key))
        if not self.enabled or total_capacity <= 0 or not keys:
            return {key: 0 for key in keys}
        capacity = max(0, int(total_capacity))
        active_count = len(keys)
        quotas = {key: 0 for key in keys}
        if active_count <= capacity:
            base_quota = max(1, int(self.min_quota_per_active_cell))
            for key in keys:
                quotas[key] = base_quota
        used = sum(quotas.values())
        if used > capacity:
            for key in keys:
                quotas[key] = 0
            used = 0
        remaining = capacity - used
        scores = {
            key: max(0.0, float(cell_scores.get(key, 0.0) or 0.0))
            for key in keys
        }
        while remaining > 0:
            chosen = max(
                keys,
                key=lambda key: (
                    scores[key] / (1.0 + quotas[key]),
                    scores[key],
                    -quotas[key],
                    key,
                ),
            )
            quotas[chosen] += 1
            remaining -= 1
        return quotas

    def evict_candidate(
        self,
        *,
        case_cluster_keys: Sequence[str],
        case_utilities: Sequence[float],
        case_mutation_pulls: Sequence[int] | None = None,
        incoming_cluster_key: str,
        incoming_utility: float = 0.0,
        max_corpus: int | None = None,
        archive: Any | None = None,
    ) -> int | None:
        live_count = min(len(case_cluster_keys), len(case_utilities))
        if live_count <= 0:
            return None
        if not self.enabled:
            return _weakest_index(
                range(live_count),
                case_cluster_keys=case_cluster_keys,
                case_utilities=case_utilities,
                case_mutation_pulls=case_mutation_pulls,
                archive=archive,
                elite_protection_bonus=self.elite_protection_bonus,
            )
        capacity = live_count if max_corpus is None else max(0, min(int(max_corpus), live_count))
        if capacity <= 0:
            return None
        normalized_clusters = [_cluster_key(key) for key in case_cluster_keys[:live_count]]
        incoming_key = _cluster_key(incoming_cluster_key)
        counts_after = Counter(normalized_clusters)
        if incoming_key:
            counts_after[incoming_key] += 1
        cell_scores = _cell_scores(
            normalized_clusters,
            case_utilities,
            case_mutation_pulls,
            incoming_cluster_key=incoming_key,
            incoming_utility=incoming_utility,
            pull_decay=self.pull_decay,
        )
        quotas = self.cell_quotas(cell_scores, total_capacity=capacity)
        over_quota_clusters = {
            cluster_key
            for cluster_key, count in counts_after.items()
            if count > quotas.get(cluster_key, 0)
        }
        if over_quota_clusters:
            candidates = [
                index
                for index, cluster_key in enumerate(normalized_clusters)
                if cluster_key in over_quota_clusters
            ]
            if candidates:
                return _weakest_index(
                    candidates,
                    case_cluster_keys=normalized_clusters,
                    case_utilities=case_utilities,
                    case_mutation_pulls=case_mutation_pulls,
                    archive=archive,
                    elite_protection_bonus=self.elite_protection_bonus,
                )
        return _weakest_index(
            range(live_count),
            case_cluster_keys=normalized_clusters,
            case_utilities=case_utilities,
            case_mutation_pulls=case_mutation_pulls,
            archive=archive,
            elite_protection_bonus=self.elite_protection_bonus,
        )

    def to_state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SEED_QUOTA_SCHEMA_VERSION,
            "enabled": self.enabled,
            "min_quota_per_active_cell": self.min_quota_per_active_cell,
            "pull_decay": self.pull_decay,
            "elite_protection_bonus": self.elite_protection_bonus,
        }

    @classmethod
    def from_state_dict(cls, data: Mapping[str, Any] | None) -> "SeedQuotaManager":
        if not isinstance(data, Mapping):
            return cls()
        return cls(
            enabled=bool(data.get("enabled", True)),
            min_quota_per_active_cell=max(0, int(data.get("min_quota_per_active_cell", 1) or 0)),
            pull_decay=max(0.0, float(data.get("pull_decay", 1.0) or 0.0)),
            elite_protection_bonus=max(0.0, float(data.get("elite_protection_bonus", 0.35) or 0.0)),
        )


def _cell_scores(
    case_cluster_keys: Sequence[str],
    case_utilities: Sequence[float],
    case_mutation_pulls: Sequence[int] | None,
    *,
    incoming_cluster_key: str,
    incoming_utility: float,
    pull_decay: float,
) -> dict[str, float]:
    counts = Counter(case_cluster_keys)
    if incoming_cluster_key:
        counts[incoming_cluster_key] += 1
    scores: defaultdict[str, float] = defaultdict(float)
    for index, cluster_key in enumerate(case_cluster_keys):
        scores[cluster_key] += _seed_quota_score(
            cluster_key,
            index=index,
            case_utilities=case_utilities,
            case_mutation_pulls=case_mutation_pulls,
            cluster_counts=counts,
            pull_decay=pull_decay,
        )
    if incoming_cluster_key:
        scores[incoming_cluster_key] += _incoming_quota_score(
            incoming_cluster_key,
            incoming_utility=incoming_utility,
            cluster_counts=counts,
        )
    return dict(scores)


def _seed_quota_score(
    cluster_key: str,
    *,
    index: int,
    case_utilities: Sequence[float],
    case_mutation_pulls: Sequence[int] | None,
    cluster_counts: Counter[str],
    pull_decay: float,
) -> float:
    utility = float(case_utilities[index]) if index < len(case_utilities) else 0.0
    pulls = (
        int(case_mutation_pulls[index])
        if case_mutation_pulls is not None and index < len(case_mutation_pulls)
        else 0
    )
    novelty = 1.0 / max(1.0, float(cluster_counts[_cluster_key(cluster_key)]))
    pull_factor = 1.0 + max(0.0, float(pulls)) * max(0.0, pull_decay)
    return max(0.0, utility) * novelty / pull_factor


def _incoming_quota_score(
    cluster_key: str,
    *,
    incoming_utility: float,
    cluster_counts: Counter[str],
) -> float:
    novelty = 1.0 / max(1.0, float(cluster_counts[_cluster_key(cluster_key)]))
    return max(0.0, float(incoming_utility)) * novelty


def _weakest_index(
    candidates: Sequence[int] | range,
    *,
    case_cluster_keys: Sequence[str],
    case_utilities: Sequence[float],
    case_mutation_pulls: Sequence[int] | None,
    archive: Any | None,
    elite_protection_bonus: float,
) -> int | None:
    candidate_list = list(candidates)
    if not candidate_list:
        return None
    counts = Counter(_cluster_key(key) for key in case_cluster_keys)
    return min(
        candidate_list,
        key=lambda index: (
            _retention_score(
                index,
                case_cluster_keys=case_cluster_keys,
                case_utilities=case_utilities,
                case_mutation_pulls=case_mutation_pulls,
                cluster_counts=counts,
                archive=archive,
                elite_protection_bonus=elite_protection_bonus,
            ),
            index,
        ),
    )


def _retention_score(
    index: int,
    *,
    case_cluster_keys: Sequence[str],
    case_utilities: Sequence[float],
    case_mutation_pulls: Sequence[int] | None,
    cluster_counts: Counter[str],
    archive: Any | None,
    elite_protection_bonus: float,
) -> float:
    cluster_key = _cluster_key(case_cluster_keys[index]) if index < len(case_cluster_keys) else UNKNOWN_CLUSTER_KEY
    utility = float(case_utilities[index]) if index < len(case_utilities) else 0.0
    pulls = (
        int(case_mutation_pulls[index])
        if case_mutation_pulls is not None and index < len(case_mutation_pulls)
        else 0
    )
    novelty = 1.0 / max(1.0, float(cluster_counts[cluster_key]))
    score = max(0.0, utility) * novelty / (1.0 + max(0, pulls))
    if _is_archive_elite(archive, cluster_key, index):
        score += max(0.0, float(elite_protection_bonus))
    return score


def _is_archive_elite(archive: Any | None, cluster_key: str, index: int) -> bool:
    if archive is None or not cluster_key:
        return False
    elite_indexes = getattr(archive, "elite_indexes", None)
    if not callable(elite_indexes):
        return False
    try:
        return index in set(elite_indexes(cluster_key))
    except Exception:
        return False


def _cluster_key(value: Any) -> str:
    text = str(value or "").strip()
    return text or UNKNOWN_CLUSTER_KEY
