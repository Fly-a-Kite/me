from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from datadiff.dsl import Case


@dataclass(frozen=True, slots=True)
class SeedCorpusRecord:
    case: Case
    family_keys: list[str]
    profile_key: str
    target_keys: list[str]
    cluster_key: str
    behavioral_descriptor: dict[str, Any]
    utility: float
    schedule_reward: float
    parent_index: int | None = None


@dataclass(slots=True)
class SeedCorpus:
    max_corpus: int
    cases: list[Case]
    utilities: list[float]
    family_keys: list[list[str]]
    profile_keys: list[str]
    target_keys: list[list[str]]
    cluster_keys: list[str]
    behavioral_descriptors: list[dict[str, Any]]
    mutation_pulls: list[int]
    schedule_rewards: list[float]
    schedule_feedback_totals: list[float]
    schedule_feedback_counts: list[int]
    stored_candidate_bug_families: Counter[str]
    stored_profiles: Counter[str]
    stored_target_keys: Counter[str]
    stored_cluster_keys: Counter[str]
    quality_archive: Any | None = None
    lineage: Any | None = None
    quota_manager: Any | None = None

    def append_seed(self, record: SeedCorpusRecord) -> int:
        index = len(self.cases)
        self.cases.append(record.case)
        self.utilities.append(float(record.utility))
        self.family_keys.append(list(record.family_keys))
        self.profile_keys.append(record.profile_key)
        self.target_keys.append(list(record.target_keys))
        self.cluster_keys.append(record.cluster_key)
        self.behavioral_descriptors.append(dict(record.behavioral_descriptor))
        self.mutation_pulls.append(0)
        self.schedule_rewards.append(float(record.schedule_reward))
        self.schedule_feedback_totals.append(0.0)
        self.schedule_feedback_counts.append(0)
        if self.lineage is not None:
            self.lineage.add_seed(index, record.parent_index)
        if self.quality_archive is not None:
            self.quality_archive.record_seed_multi(
                record.cluster_key,
                record.behavioral_descriptor,
                index,
                float(record.utility),
            )
        return index

    def replace_seed(self, index: int, record: SeedCorpusRecord) -> int:
        bounded_index = index % max(1, int(self.max_corpus))
        old_cluster_key = (
            self.cluster_keys[bounded_index]
            if bounded_index < len(self.cluster_keys)
            else ""
        )
        if self.quality_archive is not None:
            self.quality_archive.remove_seed(old_cluster_key, bounded_index)
        if self.lineage is not None:
            self.lineage.remove_seed(bounded_index)
        self._decrement_counter_values(
            self.stored_candidate_bug_families,
            self.family_keys[bounded_index] if bounded_index < len(self.family_keys) else [],
        )
        self._decrement_counter_values(
            self.stored_profiles,
            [self.profile_keys[bounded_index]] if bounded_index < len(self.profile_keys) else [],
        )
        self._decrement_counter_values(
            self.stored_target_keys,
            self.target_keys[bounded_index] if bounded_index < len(self.target_keys) else [],
        )
        self._decrement_counter_values(
            self.stored_cluster_keys,
            [self.cluster_keys[bounded_index]] if bounded_index < len(self.cluster_keys) else [],
        )

        self.cases[bounded_index] = record.case
        self.utilities[bounded_index] = float(record.utility)
        self.family_keys[bounded_index] = list(record.family_keys)
        self.profile_keys[bounded_index] = record.profile_key
        self.target_keys[bounded_index] = list(record.target_keys)
        self.cluster_keys[bounded_index] = record.cluster_key
        self.behavioral_descriptors[bounded_index] = dict(record.behavioral_descriptor)
        self.mutation_pulls[bounded_index] = 0
        self.schedule_rewards[bounded_index] = float(record.schedule_reward)
        self.schedule_feedback_totals[bounded_index] = 0.0
        self.schedule_feedback_counts[bounded_index] = 0
        if self.lineage is not None:
            self.lineage.add_seed(bounded_index, record.parent_index)
        if self.quality_archive is not None:
            self.quality_archive.record_seed_multi(
                record.cluster_key,
                record.behavioral_descriptor,
                bounded_index,
                float(record.utility),
            )
        return bounded_index

    def record_stored_counters(self, record: SeedCorpusRecord) -> None:
        self.stored_candidate_bug_families.update(record.family_keys)
        if record.profile_key:
            self.stored_profiles[record.profile_key] += 1
        self.stored_target_keys.update(record.target_keys)
        self.stored_cluster_keys.update([record.cluster_key])

    def least_useful_seed_index(self) -> int | None:
        if not self.cases or not self.utilities:
            return None
        return min(range(len(self.cases)), key=lambda index: (self.utilities[index], index))

    def evict_seed_index(self, *, incoming_cluster_key: str, incoming_utility: float) -> int | None:
        if not self.cases or self.max_corpus <= 0:
            return None
        if self.quota_manager is None:
            return self.least_useful_seed_index()
        return self.quota_manager.evict_candidate(
            case_cluster_keys=self.cluster_keys,
            case_utilities=self.utilities,
            case_mutation_pulls=self.mutation_pulls,
            incoming_cluster_key=incoming_cluster_key,
            incoming_utility=float(incoming_utility),
            max_corpus=int(self.max_corpus),
            archive=self.quality_archive,
        )

    @staticmethod
    def _decrement_counter_values(counter: Counter[str], values: list[str]) -> None:
        for value in values:
            if not value:
                continue
            counter[value] -= 1
            if counter[value] <= 0:
                del counter[value]
