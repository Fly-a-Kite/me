from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
import math
from typing import Any

from datadiff.adaptive_learning import AdaptiveLearningState
from datadiff.dsl import Case
from datadiff.exploration_objectives import EXPLORATION_OBJECTIVE_PREFIX
from datadiff.mutator import mutate_case_with_metadata, mutation_operator_profiles
from datadiff.operation_semantics import operation_names
from datadiff.quality_archive import QualityDiversityArchive
from datadiff.scheduler import LocalSourceScheduler
from datadiff.semantic_signal import (
    CANONICAL_SEMANTIC_SIGNAL_PREFIX,
    canonical_target_key,
    legacy_target_key_alias as semantic_signal_legacy_target_key_alias,
    target_key_weight,
)
from datadiff.util import CORPUS_DIR, dump_json

MUTATION_OPERATOR_LEARNING_WEIGHT = 0.05


@dataclass(frozen=True, slots=True)
class _ContextualTargetDescriptor:
    key: str
    weight: float
    semantic_family: str = ""
    semantic_signal: str = ""
    exploration_objective: str = ""


@dataclass(slots=True)
class FeedbackState:
    max_corpus: int = 256
    persist_to_disk: bool = False
    max_persisted: int = 4096
    max_cases_per_candidate_family: int = 8
    max_cases_per_profile: int = 16
    seen_signatures: set[str] = field(default_factory=set)
    interesting_cases: list[Case] = field(default_factory=list)
    stored_candidate_bug_families: Counter[str] = field(default_factory=Counter)
    stored_profiles: Counter[str] = field(default_factory=Counter)
    case_utilities: list[float] = field(default_factory=list)
    case_family_keys: list[list[str]] = field(default_factory=list)
    case_profile_keys: list[str] = field(default_factory=list)
    case_target_keys: list[list[str]] = field(default_factory=list)
    case_cluster_keys: list[str] = field(default_factory=list)
    case_mutation_pulls: list[int] = field(default_factory=list)
    case_schedule_rewards: list[float] = field(default_factory=list)
    case_schedule_feedback_totals: list[float] = field(default_factory=list)
    case_schedule_feedback_counts: list[int] = field(default_factory=list)
    recent_mutation_parent_indexes: deque[int] = field(default_factory=lambda: deque(maxlen=32))
    recent_mutation_cluster_keys: deque[str] = field(default_factory=lambda: deque(maxlen=32))
    recent_mutation_operators: deque[str] = field(default_factory=lambda: deque(maxlen=64))
    recent_mutation_parent_counts: Counter[int] = field(default_factory=Counter)
    recent_mutation_cluster_counts: Counter[str] = field(default_factory=Counter)
    recent_mutation_operator_counts: Counter[str] = field(default_factory=Counter)
    last_feedback_parent_index: int | None = None
    last_feedback_operator: str = ""
    mutation_operator_rewards: Counter[str] = field(default_factory=Counter)
    mutation_operator_pulls: Counter[str] = field(default_factory=Counter)
    mutation_operator_target_rewards: Counter[str] = field(default_factory=Counter)
    mutation_operator_target_pulls: Counter[str] = field(default_factory=Counter)
    stored_target_keys: Counter[str] = field(default_factory=Counter)
    stored_cluster_keys: Counter[str] = field(default_factory=Counter)
    cluster_schedule_feedback_totals: Counter[str] = field(default_factory=Counter)
    cluster_schedule_feedback_counts: Counter[str] = field(default_factory=Counter)
    quality_archive: QualityDiversityArchive = field(default_factory=QualityDiversityArchive)
    adaptive_learning: AdaptiveLearningState = field(default_factory=AdaptiveLearningState)
    enable_mutation_operator_learning: bool = True
    enable_quality_archive: bool = True
    last_feedback_decision: dict[str, Any] = field(default_factory=dict)
    persisted_count: int = 0
    last_persisted_to_disk: bool = False
    last_record_skip_reason: str = ""
    source_scheduler: LocalSourceScheduler | None = None
    last_candidate_source: str = "generated"
    last_candidate_metadata: dict[str, Any] = field(default_factory=dict)
    last_source_reward: float | None = None

    def select_case(self, seed: int, generated: Case) -> Case:
        self.last_candidate_source = "generated"
        self.last_candidate_metadata = _generated_candidate_metadata(generated)
        self.last_feedback_parent_index = None
        self.last_feedback_operator = ""
        self.last_feedback_decision = {}
        if not self.interesting_cases:
            return generated
        if self.source_scheduler is None:
            if seed % 3 != 0:
                return generated
        else:
            source = self.source_scheduler.choose_source(feedback_available=bool(self.interesting_cases))
            self.last_candidate_source = source
            if source != "feedback_mutation":
                return generated
        for attempt in range(min(4, len(self.interesting_cases))):
            mutation_seed = seed + attempt
            base_index = self._choose_mutation_seed_index(mutation_seed)
            base = self.interesting_cases[base_index]
            parent_target_keys = self.case_target_keys[base_index] if base_index < len(self.case_target_keys) else []
            operator_scores = self._mutation_operator_score_snapshot(target_keys=parent_target_keys)
            result = mutate_case_with_metadata(
                base,
                mutation_seed,
                allow_probe_operators=False,
                operator_scores=operator_scores,
            )
            if result.metadata.get("mutation", {}).get("changed"):
                self._record_mutation_seed_pull(base_index)
                self.last_feedback_parent_index = base_index
                self.last_feedback_operator = str(result.metadata.get("mutation", {}).get("operator", ""))
                if self.last_feedback_operator:
                    self._record_recent_operator_pull(self.last_feedback_operator)
                self.last_feedback_decision = self._feedback_decision_snapshot(
                    base_index,
                    mutation_seed=mutation_seed,
                    attempt=attempt,
                    operator_scores=operator_scores,
                )
                self.last_feedback_decision["selected_operator"] = self.last_feedback_operator
                self.last_feedback_decision["selected_operator_score"] = operator_scores.get(
                    self.last_feedback_operator,
                    operator_scores.get("__untried__", 0.0),
                )
                self.last_feedback_decision["recent_operator_pulls"] = self._recent_operator_pull_count(
                    self.last_feedback_operator
                )
                self.last_candidate_source = "feedback_mutation"
                self.last_candidate_metadata = dict(result.metadata)
                self.last_candidate_metadata["feedback_selection"] = dict(self.last_feedback_decision)
                self.last_candidate_metadata["feedback_decision"] = dict(self.last_feedback_decision)
                return result.case
        self.last_candidate_source = "generated"
        self.last_candidate_metadata = _generated_candidate_metadata(generated)
        self.last_feedback_parent_index = None
        self.last_feedback_operator = ""
        self.last_feedback_decision = {}
        return generated

    def choose_case(self, seed: int, generated: Case) -> Case:
        return self.select_case(seed, generated)

    def candidate_quality_context(
        self,
        case: Case,
        *,
        target_keys: list[str] | tuple[str, ...] | None = None,
        profile_key: str | None = None,
    ) -> dict[str, Any]:
        if not self.enable_quality_archive:
            normalized_target_keys = _normalize_target_keys(
                list(target_keys) if target_keys is not None else _case_target_keys(case)
            )
            resolved_profile_key = str(profile_key).strip() if profile_key is not None else _case_profile_key(case)
            cluster_key = _case_cluster_key(
                case,
                profile_key=resolved_profile_key,
                target_keys=normalized_target_keys,
            )
            return {
                "cluster_key": cluster_key,
                "profile_key": resolved_profile_key,
                "target_keys": normalized_target_keys,
                "target_key_count": len(normalized_target_keys),
                "archive_enabled": False,
            }
        normalized_target_keys = _normalize_target_keys(
            list(target_keys) if target_keys is not None else _case_target_keys(case)
        )
        resolved_profile_key = str(profile_key).strip() if profile_key is not None else _case_profile_key(case)
        cluster_key = _case_cluster_key(
            case,
            profile_key=resolved_profile_key,
            target_keys=normalized_target_keys,
        )
        archive_elite_indexes = self.quality_archive.elite_indexes(cluster_key)
        recent_cluster_pulls = Counter(self.recent_mutation_cluster_keys)[cluster_key]
        cluster_feedback_count = int(self.cluster_schedule_feedback_counts[cluster_key])
        archive_cluster_reward = self.quality_archive.cluster_reward_signal(cluster_key)
        return {
            "cluster_key": cluster_key,
            "profile_key": resolved_profile_key,
            "target_keys": normalized_target_keys,
            "target_key_count": len(normalized_target_keys),
            "archive_enabled": True,
            "archive_known": self.quality_archive.has_cluster(cluster_key),
            "archive_elite_indexes": archive_elite_indexes,
            "archive_seed_count": self.quality_archive.seed_count(cluster_key),
            "archive_outcome_count": self.quality_archive.outcome_count(cluster_key),
            "archive_cluster_reward": archive_cluster_reward,
            "archive_health_penalty": self.quality_archive.cluster_health_penalty(cluster_key),
            "cluster_count": int(self.stored_cluster_keys[cluster_key]),
            "cluster_feedback_reward": self._cluster_feedback_reward_signal(cluster_key),
            "cluster_feedback_count": cluster_feedback_count,
            "cluster_novelty_score": min(1.25, 0.35 / (1.0 + self.stored_cluster_keys[cluster_key])),
            "recent_cluster_pulls": int(recent_cluster_pulls),
        }

    @property
    def last_feedback_selection(self) -> dict[str, Any]:
        return self.last_feedback_decision

    @last_feedback_selection.setter
    def last_feedback_selection(self, value: dict[str, Any]) -> None:
        self.last_feedback_decision = dict(value or {})

    def record(
        self,
        case: Case,
        behavior_signature: str,
        has_finding: bool,
        *,
        novelty_signature: str | None = None,
        discovery_signature: str | None = None,
        candidate_bug_families: list[str] | None = None,
        target_keys: list[str] | None = None,
        schedule_delta: float = 0.0,
    ) -> bool:
        self.last_persisted_to_disk = False
        self.last_record_skip_reason = ""
        novelty_key = str(novelty_signature or discovery_signature or behavior_signature)
        is_new = novelty_key not in self.seen_signatures
        self.seen_signatures.add(novelty_key)
        if not (is_new or has_finding):
            self.last_record_skip_reason = "duplicate_uninteresting_behavior"
            return False
        family_keys = _unique_nonempty(candidate_bug_families or [])
        normalized_target_keys = _normalize_target_keys(target_keys or _case_target_keys(case))
        family_limit = max(0, int(self.max_cases_per_candidate_family))
        if family_limit and family_keys and all(
            self.stored_candidate_bug_families[family] >= family_limit for family in family_keys
        ):
            self.last_record_skip_reason = "candidate_family_saturated"
            return False
        profile_key = _case_profile_key(case)
        cluster_key = _case_cluster_key(case, profile_key=profile_key, target_keys=normalized_target_keys)
        profile_limit = max(0, int(self.max_cases_per_profile))
        if (
            profile_limit
            and profile_key
            and not has_finding
            and self.stored_profiles[profile_key] >= profile_limit
        ):
            self.last_record_skip_reason = "profile_saturated"
            return False
        if len(self.interesting_cases) < self.max_corpus:
            self.interesting_cases.append(case)
            utility = _case_seed_utility(case, has_finding=has_finding, family_keys=family_keys)
            self.case_utilities.append(utility)
            self.case_family_keys.append(list(family_keys))
            self.case_profile_keys.append(profile_key)
            self.case_target_keys.append(list(normalized_target_keys))
            self.case_cluster_keys.append(cluster_key)
            self.case_mutation_pulls.append(0)
            self.case_schedule_rewards.append(
                _case_seed_schedule_prior(
                    has_finding=has_finding,
                    family_keys=family_keys,
                )
                + float(schedule_delta)
            )
            self.case_schedule_feedback_totals.append(0.0)
            self.case_schedule_feedback_counts.append(0)
            if self.enable_quality_archive:
                self.quality_archive.record_seed(cluster_key, len(self.interesting_cases) - 1, utility)
        elif has_finding:
            self._replace_feedback_seed(
                int(behavior_signature, 16) % self.max_corpus,
                case,
                has_finding=has_finding,
                family_keys=family_keys,
                profile_key=profile_key,
                target_keys=normalized_target_keys,
                cluster_key=cluster_key,
                schedule_delta=schedule_delta,
            )
        else:
            utility = _case_seed_utility(case, has_finding=has_finding, family_keys=family_keys)
            replace_index = self._least_useful_seed_index()
            if replace_index is None or utility <= self.case_utilities[replace_index]:
                self.last_record_skip_reason = "corpus_full_low_utility"
                return False
            self._replace_feedback_seed(
                replace_index,
                case,
                has_finding=has_finding,
                family_keys=family_keys,
                profile_key=profile_key,
                target_keys=normalized_target_keys,
                cluster_key=cluster_key,
                utility=utility,
                schedule_delta=schedule_delta,
            )
        self.stored_candidate_bug_families.update(family_keys)
        if profile_key:
            self.stored_profiles[profile_key] += 1
        self.stored_target_keys.update(normalized_target_keys)
        self.stored_cluster_keys.update([cluster_key])
        if self.persist_to_disk and self.persisted_count < max(0, self.max_persisted):
            self._write_interesting_case(
                case,
                behavior_signature,
                has_finding,
                discovery_signature=str(discovery_signature or novelty_key),
            )
            self.persisted_count += 1
            self.last_persisted_to_disk = True
        return True

    def _replace_feedback_seed(
        self,
        index: int,
        case: Case,
        *,
        has_finding: bool,
        family_keys: list[str],
        profile_key: str,
        target_keys: list[str],
        cluster_key: str,
        utility: float | None = None,
        schedule_delta: float = 0.0,
    ) -> None:
        bounded_index = index % max(1, self.max_corpus)
        self._align_seed_metadata_lengths()
        old_cluster_key = self.case_cluster_keys[bounded_index] if bounded_index < len(self.case_cluster_keys) else ""
        if self.enable_quality_archive:
            self.quality_archive.remove_seed(old_cluster_key, bounded_index)
        self._decrement_counter_values(self.stored_candidate_bug_families, self.case_family_keys[bounded_index])
        self._decrement_counter_values(self.stored_profiles, [self.case_profile_keys[bounded_index]])
        self._decrement_counter_values(self.stored_target_keys, self.case_target_keys[bounded_index])
        self._decrement_counter_values(self.stored_cluster_keys, [self.case_cluster_keys[bounded_index]])
        self.interesting_cases[bounded_index] = case
        self.case_utilities[bounded_index] = (
            utility if utility is not None else _case_seed_utility(case, has_finding=has_finding, family_keys=family_keys)
        )
        self.case_family_keys[bounded_index] = list(family_keys)
        self.case_profile_keys[bounded_index] = profile_key
        self.case_target_keys[bounded_index] = list(target_keys)
        self.case_cluster_keys[bounded_index] = cluster_key
        self.case_mutation_pulls[bounded_index] = 0
        self.case_schedule_rewards[bounded_index] = _case_seed_schedule_prior(
            has_finding=has_finding,
            family_keys=family_keys,
        ) + float(schedule_delta)
        self.case_schedule_feedback_totals[bounded_index] = 0.0
        self.case_schedule_feedback_counts[bounded_index] = 0
        if self.enable_quality_archive:
            self.quality_archive.record_seed(cluster_key, bounded_index, self.case_utilities[bounded_index])

    def _least_useful_seed_index(self) -> int | None:
        if not self.interesting_cases or not self.case_utilities:
            return None
        return min(range(len(self.interesting_cases)), key=lambda index: (self.case_utilities[index], index))

    def _choose_mutation_seed_index(self, seed: int) -> int:
        if not self.interesting_cases:
            raise ValueError("feedback corpus is empty")
        self._align_seed_metadata_lengths()
        window = min(len(self.interesting_cases), 16)
        start = seed % len(self.interesting_cases)
        candidate_indexes = [(start + offset) % len(self.interesting_cases) for offset in range(window)]
        return max(
            candidate_indexes,
            key=lambda index: (
                self._case_seed_schedule_score(index),
                -self.case_mutation_pulls[index],
                -index,
            ),
        )

    def _case_seed_schedule_score(self, index: int) -> float:
        reward_prior = self._case_seed_schedule_reward(index)
        target_novelty = self._case_target_novelty_score(index)
        cluster_key = self._case_cluster_key_at(index)
        archive_cluster_reward = (
            self.quality_archive.cluster_reward_signal(cluster_key)
            if self.enable_quality_archive
            else 0.0
        )
        cluster_reward = max(self._cluster_feedback_reward_signal(cluster_key), archive_cluster_reward)
        cluster_novelty = self._case_cluster_novelty_score(index)
        elite_bonus = (
            self.quality_archive.seed_elite_bonus(cluster_key, index)
            if self.enable_quality_archive
            else 0.0
        )
        archive_health_penalty = (
            self.quality_archive.cluster_health_penalty(cluster_key)
            if self.enable_quality_archive
            else 0.0
        )
        pull_penalty = (1.0 + self.case_mutation_pulls[index]) ** 0.5
        recent_penalty = 1.0 + self._recent_parent_pull_count(index)
        recent_cluster_penalty = 1.0 + (0.50 * self._recent_cluster_pull_count(cluster_key))
        return (
            1.0
            + reward_prior
            + target_novelty
            + cluster_reward
            + cluster_novelty
            + elite_bonus
        ) / (pull_penalty * recent_penalty * recent_cluster_penalty * (1.0 + archive_health_penalty))

    def _case_seed_schedule_reward(self, index: int) -> float:
        base_reward = self.case_schedule_rewards[index] if index < len(self.case_schedule_rewards) else 0.0
        feedback_reward = self._case_feedback_reward_signal(index)
        return base_reward + feedback_reward

    def _case_feedback_reward_signal(self, index: int) -> float:
        if index >= len(self.case_schedule_feedback_counts):
            return 0.0
        pulls = self.case_schedule_feedback_counts[index]
        if pulls <= 0:
            return 0.0
        total_reward = (
            self.case_schedule_feedback_totals[index]
            if index < len(self.case_schedule_feedback_totals)
            else 0.0
        )
        return _bounded_confident_mean_reward(total_reward, pulls, max_abs=2.5)

    def _case_target_novelty_score(self, index: int) -> float:
        if index >= len(self.case_target_keys):
            return 0.0
        target_keys = self.case_target_keys[index]
        if not target_keys:
            return 0.0
        novelty = sum(1.0 / (1.0 + self.stored_target_keys[target]) for target in target_keys[:8])
        return min(1.5, 0.25 * novelty)

    def _case_cluster_key_at(self, index: int) -> str:
        self._align_seed_metadata_lengths()
        if index < len(self.case_cluster_keys):
            return self.case_cluster_keys[index]
        return ""

    def _case_cluster_novelty_score(self, index: int) -> float:
        cluster_key = self._case_cluster_key_at(index)
        if not cluster_key:
            return 0.0
        return min(1.25, 0.35 / (1.0 + self.stored_cluster_keys[cluster_key]))

    def _cluster_feedback_reward_signal(self, cluster_key: str) -> float:
        if not cluster_key:
            return 0.0
        pulls = self.cluster_schedule_feedback_counts[cluster_key]
        if pulls <= 0:
            return 0.0
        return _bounded_confident_mean_reward(
            self.cluster_schedule_feedback_totals[cluster_key],
            pulls,
            max_abs=1.5,
        )

    def _record_mutation_seed_pull(self, index: int) -> None:
        self._align_seed_metadata_lengths()
        self.case_mutation_pulls[index] += 1
        self._record_recent_parent_pull(index)
        cluster_key = self._case_cluster_key_at(index)
        self._record_recent_cluster_pull(cluster_key)
        if self.enable_quality_archive:
            self.quality_archive.record_pull(cluster_key, index)

    def _recent_parent_pull_count(self, index: int) -> int:
        self._sync_recent_parent_counts()
        return self.recent_mutation_parent_counts[index]

    def _recent_cluster_pull_count(self, cluster_key: str) -> int:
        if not cluster_key:
            return 0
        self._sync_recent_cluster_counts()
        return self.recent_mutation_cluster_counts[cluster_key]

    def _align_seed_metadata_lengths(self) -> None:
        metadata_changed = False
        while len(self.case_utilities) < len(self.interesting_cases):
            self.case_utilities.append(
                _case_seed_utility(
                    self.interesting_cases[len(self.case_utilities)],
                    has_finding=False,
                    family_keys=[],
                )
            )
            metadata_changed = True
        while len(self.case_family_keys) < len(self.interesting_cases):
            self.case_family_keys.append([])
            metadata_changed = True
        while len(self.case_profile_keys) < len(self.interesting_cases):
            self.case_profile_keys.append("")
            metadata_changed = True
        while len(self.case_target_keys) < len(self.interesting_cases):
            self.case_target_keys.append(_normalize_target_keys(_case_target_keys(self.interesting_cases[len(self.case_target_keys)])))
            metadata_changed = True
        while len(self.case_cluster_keys) < len(self.interesting_cases):
            index = len(self.case_cluster_keys)
            profile_key = self.case_profile_keys[index] if index < len(self.case_profile_keys) else ""
            target_keys = self.case_target_keys[index] if index < len(self.case_target_keys) else []
            self.case_cluster_keys.append(
                _case_cluster_key(self.interesting_cases[index], profile_key=profile_key, target_keys=target_keys)
            )
            metadata_changed = True
        while len(self.case_mutation_pulls) < len(self.interesting_cases):
            self.case_mutation_pulls.append(0)
            metadata_changed = True
        while len(self.case_schedule_rewards) < len(self.interesting_cases):
            self.case_schedule_rewards.append(0.0)
            metadata_changed = True
        while len(self.case_schedule_feedback_totals) < len(self.interesting_cases):
            self.case_schedule_feedback_totals.append(0.0)
            metadata_changed = True
        while len(self.case_schedule_feedback_counts) < len(self.interesting_cases):
            self.case_schedule_feedback_counts.append(0)
            metadata_changed = True
        if self.enable_quality_archive and (
            metadata_changed or (self.interesting_cases and self.quality_archive.is_empty())
        ):
            self._sync_quality_archive()

    def _sync_quality_archive(self) -> None:
        if not self.enable_quality_archive:
            return
        valid_indexes_by_cluster: dict[str, set[int]] = {}
        for index in range(len(self.interesting_cases)):
            cluster_key = self.case_cluster_keys[index] if index < len(self.case_cluster_keys) else ""
            if not cluster_key:
                continue
            utility = self.case_utilities[index] if index < len(self.case_utilities) else 0.0
            valid_indexes_by_cluster.setdefault(cluster_key, set()).add(index)
            self.quality_archive.record_seed(cluster_key, index, utility)
        self.quality_archive.retain_seeds(valid_indexes_by_cluster)

    def _mutation_operator_score_snapshot(self, *, target_keys: list[str] | None = None) -> dict[str, float]:
        operator_profiles = mutation_operator_profiles(allow_probe_operators=False)
        self._sync_recent_operator_counts()
        known_operators = list(operator_profiles)
        known_operator_set = set(known_operators)
        for operator_source in (
            self.mutation_operator_rewards,
            self.mutation_operator_pulls,
            self.recent_mutation_operator_counts,
        ):
            for operator in operator_source:
                if operator and operator not in known_operator_set:
                    known_operators.append(operator)
                    known_operator_set.add(operator)
        if not known_operators:
            return {}
        mutation_operator_pulls = self.mutation_operator_pulls
        mutation_operator_rewards = self.mutation_operator_rewards
        recent_operator_counts = self.recent_mutation_operator_counts
        total_pulls = sum(mutation_operator_pulls.values())
        contextual_targets = _contextual_target_descriptors(target_keys or [])
        learning_context = self._mutation_operator_learning_context_features(contextual_targets)
        learning_active = self.enable_mutation_operator_learning and self._learning_scope_has_feedback(
            "mutation_operator"
        )
        scores: dict[str, float] = {}
        for operator in known_operators:
            pulls = mutation_operator_pulls[operator]
            mean_reward = _bounded_confident_mean_reward(
                mutation_operator_rewards[operator],
                pulls,
                max_abs=2.0,
            )
            exploration_bonus = 0.20 / ((1 + pulls) ** 0.5)
            recent_penalty = 0.15 * recent_operator_counts[operator]
            target_affinity_bonus = self._mutation_operator_target_affinity(operator, contextual_targets)
            structural_affinity_bonus = self._mutation_operator_structural_affinity(
                operator,
                contextual_targets,
                operator_profiles,
            )
            learning_bonus = (
                self._mutation_operator_learning_bonus(operator, learning_context)
                if learning_active
                else 0.0
            )
            scores[operator] = (
                mean_reward
                + exploration_bonus
                - recent_penalty
                + target_affinity_bonus
                + structural_affinity_bonus
                + learning_bonus
            )
        if total_pulls >= 8:
            scores["__untried__"] = 0.15
        return scores

    def _mutation_operator_learning_bonus(
        self,
        operator: str,
        context_features: tuple[str, ...],
    ) -> float:
        row = self.adaptive_learning.score_action(
            "mutation_operator",
            operator,
            context_features=context_features,
        )
        return MUTATION_OPERATOR_LEARNING_WEIGHT * (
            float(row.get("model_prediction", 0.0) or 0.0)
            + 0.25 * float(row.get("uncertainty", 0.0) or 0.0)
            + float(row.get("version_signal", 0.0) or 0.0)
            - float(row.get("health_penalty", 0.0) or 0.0)
        )

    def _mutation_operator_learning_context_features(
        self,
        target_keys: list[_ContextualTargetDescriptor],
    ) -> tuple[str, ...]:
        features: list[str] = []
        for target in target_keys[:8]:
            if target.key:
                features.append(target.key)
            if target.semantic_family:
                features.append(f"semantic_family:{target.semantic_family}")
            if target.semantic_signal:
                features.append(f"semantic_signal:{target.semantic_signal}")
            if target.exploration_objective:
                features.append(f"exploration_objective:{target.exploration_objective}")
        features.append(_bucket("feedback_corpus", len(self.interesting_cases), [(0, "empty"), (8, "small"), (64, "medium")], "large"))
        features.append(_bucket("target_key_count", len(target_keys), [(0, "none"), (2, "few"), (6, "some")], "many"))
        if not self.enable_quality_archive:
            features.append("quality_archive:disabled")
        elif self.quality_archive.is_empty():
            features.append("quality_archive:empty")
        else:
            features.append(_bucket("quality_archive_clusters", len(self.quality_archive.cells), [(4, "few"), (16, "some")], "many"))
        return tuple(_unique_nonempty(features))

    def _learning_scope_has_feedback(self, scope: str) -> bool:
        bandit = self.adaptive_learning.bandits.get(scope)
        return bool(bandit is not None and bandit.total_pulls > 0)

    def _recent_operator_pull_count(self, operator: str) -> int:
        self._sync_recent_operator_counts()
        return self.recent_mutation_operator_counts[operator]

    def _record_recent_parent_pull(self, index: int) -> None:
        if len(self.recent_mutation_parent_indexes) == self.recent_mutation_parent_indexes.maxlen:
            evicted = self.recent_mutation_parent_indexes[0]
            self._decrement_recent_parent_count(evicted)
        self.recent_mutation_parent_indexes.append(index)
        self.recent_mutation_parent_counts[index] += 1

    def _record_recent_cluster_pull(self, cluster_key: str) -> None:
        if not cluster_key:
            return
        if len(self.recent_mutation_cluster_keys) == self.recent_mutation_cluster_keys.maxlen:
            evicted = self.recent_mutation_cluster_keys[0]
            self._decrement_recent_cluster_count(evicted)
        self.recent_mutation_cluster_keys.append(cluster_key)
        self.recent_mutation_cluster_counts[cluster_key] += 1

    def _record_recent_operator_pull(self, operator: str) -> None:
        if len(self.recent_mutation_operators) == self.recent_mutation_operators.maxlen:
            evicted = self.recent_mutation_operators[0]
            self._decrement_recent_operator_count(evicted)
        self.recent_mutation_operators.append(operator)
        self.recent_mutation_operator_counts[operator] += 1

    def _decrement_recent_parent_count(self, index: int) -> None:
        self.recent_mutation_parent_counts[index] -= 1
        if self.recent_mutation_parent_counts[index] <= 0:
            del self.recent_mutation_parent_counts[index]

    def _decrement_recent_cluster_count(self, cluster_key: str) -> None:
        self.recent_mutation_cluster_counts[cluster_key] -= 1
        if self.recent_mutation_cluster_counts[cluster_key] <= 0:
            del self.recent_mutation_cluster_counts[cluster_key]

    def _decrement_recent_operator_count(self, operator: str) -> None:
        self.recent_mutation_operator_counts[operator] -= 1
        if self.recent_mutation_operator_counts[operator] <= 0:
            del self.recent_mutation_operator_counts[operator]

    def _sync_recent_parent_counts(self) -> None:
        if sum(self.recent_mutation_parent_counts.values()) == len(self.recent_mutation_parent_indexes):
            return
        self.recent_mutation_parent_counts = Counter(self.recent_mutation_parent_indexes)

    def _sync_recent_cluster_counts(self) -> None:
        if sum(self.recent_mutation_cluster_counts.values()) == len(self.recent_mutation_cluster_keys):
            return
        self.recent_mutation_cluster_counts = Counter(self.recent_mutation_cluster_keys)

    def _sync_recent_operator_counts(self) -> None:
        if sum(self.recent_mutation_operator_counts.values()) == len(self.recent_mutation_operators):
            return
        self.recent_mutation_operator_counts = Counter(self.recent_mutation_operators)

    def _mutation_operator_target_affinity(
        self,
        operator: str,
        target_keys: list[_ContextualTargetDescriptor],
    ) -> float:
        if not target_keys:
            return 0.0
        target_pulls = self.mutation_operator_target_pulls
        target_rewards = self.mutation_operator_target_rewards
        weighted_reward = 0.0
        weighted_pulls = 0.0
        for target in target_keys:
            reward, pulls = _operator_target_reward_pull(
                target_rewards,
                target_pulls,
                operator,
                target.key,
            )
            if pulls <= 0:
                continue
            weight = target.weight
            weighted_reward += weight * reward
            weighted_pulls += weight
        if weighted_pulls <= 0.0:
            return 0.0
        return _bounded_confident_mean_reward(weighted_reward, weighted_pulls, max_abs=0.75)

    def _mutation_operator_structural_affinity(
        self,
        operator: str,
        target_keys: list["_ContextualTargetDescriptor"],
        operator_profiles: dict[str, Any],
    ) -> float:
        if not target_keys:
            return 0.0
        profile = operator_profiles.get(operator)
        if profile is None:
            return 0.0
        family_affinity = tuple(
            str(value).strip()
            for value in getattr(profile, "semantic_family_affinity", getattr(profile, "semantic_affinity", ()))
            if str(value).strip()
        )
        signal_affinity = tuple(
            str(value).strip()
            for value in getattr(profile, "semantic_signal_affinity", ())
            if str(value).strip()
        )
        objective_affinity = tuple(
            str(value).strip()
            for value in getattr(profile, "exploration_objective_affinity", ())
            if str(value).strip()
        )
        if not family_affinity and not signal_affinity and not objective_affinity:
            return 0.0
        bonus = 0.0
        for target in target_keys:
            if target.semantic_family:
                if target.semantic_family in family_affinity:
                    bonus += 0.16 if target.key.startswith("semantic_family:") else 0.10
            elif target.semantic_signal and target.semantic_signal in signal_affinity:
                bonus += 0.18
            elif target.exploration_objective and target.exploration_objective in objective_affinity:
                bonus += 0.14 if target.key.startswith(EXPLORATION_OBJECTIVE_PREFIX) else 0.08
        return min(0.55, bonus)

    def _feedback_decision_snapshot(
        self,
        index: int,
        *,
        mutation_seed: int,
        attempt: int,
        operator_scores: dict[str, float],
    ) -> dict[str, Any]:
        self._align_seed_metadata_lengths()
        target_keys = self.case_target_keys[index] if index < len(self.case_target_keys) else []
        semantic_family_targets = [
            str(key).removeprefix("semantic_family:").strip()
            for key in target_keys
            if str(key).startswith("semantic_family:") and str(key).removeprefix("semantic_family:").strip()
        ]
        semantic_signal_targets = [
            str(key).removeprefix(CANONICAL_SEMANTIC_SIGNAL_PREFIX).strip()
            for key in target_keys
            if str(key).startswith(CANONICAL_SEMANTIC_SIGNAL_PREFIX)
            and str(key).removeprefix(CANONICAL_SEMANTIC_SIGNAL_PREFIX).strip()
        ]
        exploration_objective_targets = [
            str(key).removeprefix(EXPLORATION_OBJECTIVE_PREFIX).strip()
            for key in target_keys
            if str(key).startswith(EXPLORATION_OBJECTIVE_PREFIX)
            and str(key).removeprefix(EXPLORATION_OBJECTIVE_PREFIX).strip()
        ]
        return {
            "parent_index": index,
            "parent_case_id": self.interesting_cases[index].case_id,
            "mutation_seed": mutation_seed,
            "attempt": attempt,
            "retention_utility": self.case_utilities[index],
            "schedule_reward": self._case_seed_schedule_reward(index),
            "schedule_base_reward": self.case_schedule_rewards[index],
            "schedule_feedback_reward": self._case_feedback_reward_signal(index),
            "schedule_feedback_count": self.case_schedule_feedback_counts[index],
            "target_novelty_score": self._case_target_novelty_score(index),
            "schedule_score": self._case_seed_schedule_score(index),
            "target_keys": target_keys,
            "target_key_count": len(target_keys),
            "cluster_key": self._case_cluster_key_at(index),
            "cluster_reward": self._cluster_feedback_reward_signal(self._case_cluster_key_at(index)),
            "cluster_novelty_score": self._case_cluster_novelty_score(index),
            "recent_cluster_pulls": self._recent_cluster_pull_count(self._case_cluster_key_at(index)),
            "quality_archive_cluster_reward": self.quality_archive.cluster_reward_signal(
                self._case_cluster_key_at(index)
            ),
            "quality_archive_health_penalty": self.quality_archive.cluster_health_penalty(
                self._case_cluster_key_at(index)
            ),
            "quality_archive_elite_bonus": self.quality_archive.seed_elite_bonus(
                self._case_cluster_key_at(index),
                index,
            ),
            "quality_archive_elite_indexes": self.quality_archive.elite_indexes(self._case_cluster_key_at(index)),
            "semantic_family_targets": semantic_family_targets,
            "semantic_signal_targets": semantic_signal_targets,
            "exploration_objective_targets": exploration_objective_targets,
            "mutation_pulls": self.case_mutation_pulls[index],
            "recent_parent_pulls": self._recent_parent_pull_count(index),
            "operator_score_count": len(operator_scores),
            "recent_operator_pulls": 0,
            "operator_score_top": [
                {"operator": operator, "mean_reward": score}
                for operator, score in sorted(operator_scores.items(), key=lambda item: item[1], reverse=True)[:8]
            ],
        }

    _feedback_selection_snapshot = _feedback_decision_snapshot

    @staticmethod
    def _decrement_counter_values(counter: Counter[str], values: list[str]) -> None:
        for value in values:
            if not value:
                continue
            counter[value] -= 1
            if counter[value] <= 0:
                del counter[value]

    def record_candidate_outcome(
        self,
        candidate_source: str,
        *,
        has_finding: bool,
        is_new_behavior: bool,
        preflight: dict[str, Any],
        candidate_bug: bool = False,
        semantic_divergence: bool = False,
        false_positive: bool = False,
        candidate_bug_families: list[str] | None = None,
        candidate_bug_signatures: list[str] | None = None,
        reward_adjustment: float = 0.0,
    ) -> float | None:
        if self.source_scheduler is None:
            self.last_source_reward = None
            return None
        reward = self.source_scheduler.record_result(
            "feedback_mutation" if candidate_source == "feedback_mutation" else "generated",
            has_finding=has_finding,
            is_new_behavior=is_new_behavior,
            preflight_valid=bool(preflight.get("valid", True)),
            fallback_used=bool(preflight.get("fallback_used", False)),
            candidate_bug=candidate_bug,
            semantic_divergence=semantic_divergence,
            false_positive=false_positive,
            candidate_bug_families=candidate_bug_families,
            candidate_bug_signatures=candidate_bug_signatures,
            reward_adjustment=reward_adjustment,
        )
        self.last_source_reward = reward
        self.record_feedback_outcome_reward(
            candidate_source,
            reward,
            preflight_valid=bool(preflight.get("valid", True)),
            fallback_used=bool(preflight.get("fallback_used", False)),
            false_positive=false_positive,
        )
        return reward

    def record_candidate_result(
        self,
        candidate_source: str,
        *,
        has_finding: bool,
        is_new_behavior: bool,
        preflight: dict[str, Any],
        candidate_bug: bool = False,
        semantic_divergence: bool = False,
        false_positive: bool = False,
        candidate_bug_families: list[str] | None = None,
        candidate_bug_signatures: list[str] | None = None,
        reward_adjustment: float = 0.0,
    ) -> float | None:
        return self.record_candidate_outcome(
            candidate_source,
            has_finding=has_finding,
            is_new_behavior=is_new_behavior,
            preflight=preflight,
            candidate_bug=candidate_bug,
            semantic_divergence=semantic_divergence,
            false_positive=false_positive,
            candidate_bug_families=candidate_bug_families,
            candidate_bug_signatures=candidate_bug_signatures,
            reward_adjustment=reward_adjustment,
        )

    def record_feedback_outcome_reward(
        self,
        candidate_source: str,
        reward: float | None,
        *,
        preflight_valid: bool = True,
        fallback_used: bool = False,
        false_positive: bool = False,
    ) -> None:
        if candidate_source != "feedback_mutation" or reward is None:
            return
        self._align_seed_metadata_lengths()
        if self.last_feedback_parent_index is not None and self.last_feedback_parent_index < len(self.case_schedule_rewards):
            self.case_schedule_feedback_totals[self.last_feedback_parent_index] += float(reward)
            self.case_schedule_feedback_counts[self.last_feedback_parent_index] += 1
            cluster_key = self._case_cluster_key_at(self.last_feedback_parent_index)
            if cluster_key:
                self.cluster_schedule_feedback_totals[cluster_key] += float(reward)
                self.cluster_schedule_feedback_counts[cluster_key] += 1
                if self.enable_quality_archive:
                    self.quality_archive.record_outcome(
                        cluster_key,
                        index=self.last_feedback_parent_index,
                        reward=float(reward),
                        preflight_valid=preflight_valid,
                        fallback_used=fallback_used,
                        false_positive=false_positive,
                    )
        if self.last_feedback_operator:
            self.mutation_operator_pulls[self.last_feedback_operator] += 1
            self.mutation_operator_rewards[self.last_feedback_operator] += float(reward)
            target_keys = []
            if self.last_feedback_parent_index is not None and self.last_feedback_parent_index < len(self.case_target_keys):
                target_keys = _normalize_target_keys(self.case_target_keys[self.last_feedback_parent_index])
                for target_key in target_keys[:8]:
                    stat_key = _operator_target_stat_key(self.last_feedback_operator, target_key)
                    self.mutation_operator_target_pulls[stat_key] += 1
                    self.mutation_operator_target_rewards[stat_key] += float(reward)
            if self.enable_mutation_operator_learning:
                self.adaptive_learning.record_outcome(
                    "mutation_operator",
                    self.last_feedback_operator,
                    context_features=self._mutation_operator_learning_context_features(
                        _contextual_target_descriptors(target_keys)
                    ),
                    reward=float(reward),
                    preflight_valid=preflight_valid,
                    fallback_used=fallback_used,
                    false_positive=false_positive,
                )

    def record_feedback_candidate_reward(self, candidate_source: str, reward: float | None) -> None:
        self.record_feedback_outcome_reward(candidate_source, reward)

    def _write_interesting_case(
        self,
        case: Case,
        behavior_signature: str,
        has_finding: bool,
        *,
        discovery_signature: str,
    ) -> None:
        path = CORPUS_DIR / "interesting" / f"{behavior_signature}.json"
        dump_json(
            {
                "behavior_signature": behavior_signature,
                "discovery_signature": discovery_signature,
                "has_finding": has_finding,
                "case": case.to_dict(),
            },
            path,
        )

    def to_state_dict(self) -> dict[str, Any]:
        return {
            "max_corpus": self.max_corpus,
            "persist_to_disk": self.persist_to_disk,
            "max_persisted": self.max_persisted,
            "max_cases_per_candidate_family": self.max_cases_per_candidate_family,
            "max_cases_per_profile": self.max_cases_per_profile,
            "seen_signatures": sorted(self.seen_signatures),
            "interesting_cases": [case.to_dict() for case in self.interesting_cases],
            "stored_candidate_bug_families": dict(self.stored_candidate_bug_families),
            "stored_profiles": dict(self.stored_profiles),
            "case_utilities": list(self.case_utilities),
            "case_family_keys": [list(items) for items in self.case_family_keys],
            "case_profile_keys": list(self.case_profile_keys),
            "case_target_keys": [list(items) for items in self.case_target_keys],
            "case_cluster_keys": list(self.case_cluster_keys),
            "case_mutation_pulls": list(self.case_mutation_pulls),
            "case_schedule_rewards": list(self.case_schedule_rewards),
            "case_schedule_feedback_totals": list(self.case_schedule_feedback_totals),
            "case_schedule_feedback_counts": list(self.case_schedule_feedback_counts),
            "recent_mutation_parent_indexes": list(self.recent_mutation_parent_indexes),
            "recent_mutation_cluster_keys": list(self.recent_mutation_cluster_keys),
            "recent_mutation_operators": list(self.recent_mutation_operators),
            "recent_mutation_parent_counts": dict(self.recent_mutation_parent_counts),
            "recent_mutation_cluster_counts": dict(self.recent_mutation_cluster_counts),
            "recent_mutation_operator_counts": dict(self.recent_mutation_operator_counts),
            "last_feedback_parent_index": self.last_feedback_parent_index,
            "last_feedback_operator": self.last_feedback_operator,
            "mutation_operator_rewards": dict(self.mutation_operator_rewards),
            "mutation_operator_pulls": dict(self.mutation_operator_pulls),
            "mutation_operator_target_rewards": dict(self.mutation_operator_target_rewards),
            "mutation_operator_target_pulls": dict(self.mutation_operator_target_pulls),
            "stored_target_keys": dict(self.stored_target_keys),
            "stored_cluster_keys": dict(self.stored_cluster_keys),
            "cluster_schedule_feedback_totals": dict(self.cluster_schedule_feedback_totals),
            "cluster_schedule_feedback_counts": dict(self.cluster_schedule_feedback_counts),
            "quality_archive": self.quality_archive.to_state_dict(),
            "adaptive_learning": self.adaptive_learning.to_state_dict(),
            "enable_mutation_operator_learning": self.enable_mutation_operator_learning,
            "enable_quality_archive": self.enable_quality_archive,
            "last_feedback_selection": dict(self.last_feedback_decision),
            "last_feedback_decision": dict(self.last_feedback_decision),
            "persisted_count": self.persisted_count,
            "source_scheduler": (
                self.source_scheduler.to_state_dict() if self.source_scheduler is not None else None
            ),
        }

    @classmethod
    def from_state_dict(
        cls,
        data: dict[str, Any],
        *,
        persist_to_disk: bool,
        max_persisted: int,
        max_cases_per_profile: int,
        source_scheduler: LocalSourceScheduler | None,
        enable_mutation_operator_learning: bool | None = None,
        enable_quality_archive: bool | None = None,
    ) -> "FeedbackState":
        state = cls(
            max_corpus=int(data.get("max_corpus", 256) or 256),
            persist_to_disk=persist_to_disk,
            max_persisted=max_persisted,
            max_cases_per_candidate_family=int(data.get("max_cases_per_candidate_family", 8) or 8),
            max_cases_per_profile=max_cases_per_profile,
            source_scheduler=source_scheduler,
            enable_mutation_operator_learning=bool(
                enable_mutation_operator_learning
                if enable_mutation_operator_learning is not None
                else data.get("enable_mutation_operator_learning", True)
            ),
            enable_quality_archive=bool(
                enable_quality_archive
                if enable_quality_archive is not None
                else data.get("enable_quality_archive", True)
            ),
        )
        state.seen_signatures = {str(item) for item in data.get("seen_signatures", []) or []}
        state.interesting_cases = [
            Case.from_dict(item)
            for item in data.get("interesting_cases", []) or []
            if isinstance(item, dict)
        ]
        state.stored_candidate_bug_families = Counter(data.get("stored_candidate_bug_families", {}) or {})
        state.stored_profiles = Counter(data.get("stored_profiles", {}) or {})
        state.case_utilities = [float(item) for item in data.get("case_utilities", []) or []]
        state.case_family_keys = [list(items) for items in data.get("case_family_keys", []) or []]
        state.case_profile_keys = [str(item) for item in data.get("case_profile_keys", []) or []]
        state.case_target_keys = [_normalize_target_keys(items) for items in data.get("case_target_keys", []) or []]
        state.case_cluster_keys = [str(item) for item in data.get("case_cluster_keys", []) or []]
        state.case_mutation_pulls = [int(item) for item in data.get("case_mutation_pulls", []) or []]
        state.case_schedule_rewards = [float(item) for item in data.get("case_schedule_rewards", []) or []]
        state.case_schedule_feedback_totals = [
            float(item) for item in data.get("case_schedule_feedback_totals", []) or []
        ]
        state.case_schedule_feedback_counts = [
            int(item) for item in data.get("case_schedule_feedback_counts", []) or []
        ]
        state.recent_mutation_parent_indexes = deque(
            (int(item) for item in data.get("recent_mutation_parent_indexes", []) or []),
            maxlen=32,
        )
        state.recent_mutation_cluster_keys = deque(
            (str(item) for item in data.get("recent_mutation_cluster_keys", []) or []),
            maxlen=32,
        )
        state.recent_mutation_operators = deque(
            (str(item) for item in data.get("recent_mutation_operators", []) or []),
            maxlen=64,
        )
        state.recent_mutation_parent_counts = Counter(
            {
                int(key): int(value)
                for key, value in (data.get("recent_mutation_parent_counts", {}) or {}).items()
            }
        )
        state.recent_mutation_cluster_counts = Counter(
            {
                str(key): int(value)
                for key, value in (data.get("recent_mutation_cluster_counts", {}) or {}).items()
            }
        )
        state.recent_mutation_operator_counts = Counter(
            {
                str(key): int(value)
                for key, value in (data.get("recent_mutation_operator_counts", {}) or {}).items()
            }
        )
        parent_index = data.get("last_feedback_parent_index", None)
        state.last_feedback_parent_index = int(parent_index) if parent_index is not None else None
        state.last_feedback_operator = str(data.get("last_feedback_operator", "") or "")
        state.mutation_operator_rewards = _normalize_float_counter(data.get("mutation_operator_rewards", {}) or {})
        state.mutation_operator_pulls = _normalize_int_counter(data.get("mutation_operator_pulls", {}) or {})
        state.mutation_operator_target_rewards = _normalize_operator_target_counter(
            data.get("mutation_operator_target_rewards", {}) or {},
            value_type=float,
        )
        state.mutation_operator_target_pulls = _normalize_operator_target_counter(
            data.get("mutation_operator_target_pulls", {}) or {},
            value_type=int,
        )
        raw_stored_target_keys = data.get("stored_target_keys", None)
        if isinstance(raw_stored_target_keys, dict) and raw_stored_target_keys:
            state.stored_target_keys = _normalize_target_counter(raw_stored_target_keys)
        else:
            rebuilt_target_keys: Counter[str] = Counter()
            for target_keys in state.case_target_keys:
                rebuilt_target_keys.update(_normalize_target_keys(target_keys))
            state.stored_target_keys = rebuilt_target_keys
        raw_stored_cluster_keys = data.get("stored_cluster_keys", None)
        if isinstance(raw_stored_cluster_keys, dict) and raw_stored_cluster_keys:
            state.stored_cluster_keys = _normalize_int_counter(raw_stored_cluster_keys)
        else:
            rebuilt_cluster_keys: Counter[str] = Counter()
            rebuilt_cluster_keys.update(str(key) for key in state.case_cluster_keys if str(key))
            state.stored_cluster_keys = rebuilt_cluster_keys
        state.cluster_schedule_feedback_totals = _normalize_float_counter(
            data.get("cluster_schedule_feedback_totals", {}) or {}
        )
        state.cluster_schedule_feedback_counts = _normalize_int_counter(
            data.get("cluster_schedule_feedback_counts", {}) or {}
        )
        state.quality_archive = QualityDiversityArchive.from_state_dict(data.get("quality_archive"))
        state.adaptive_learning = AdaptiveLearningState.from_state_dict(data.get("adaptive_learning"))
        state.last_feedback_decision = dict(
            data.get("last_feedback_selection", data.get("last_feedback_decision", {})) or {}
        )
        state.persisted_count = int(data.get("persisted_count", 0) or 0)
        state._sync_recent_parent_counts()
        state._sync_recent_cluster_counts()
        state._sync_recent_operator_counts()
        state._align_seed_metadata_lengths()
        if state.enable_quality_archive:
            state._sync_quality_archive()
        if not state.stored_cluster_keys:
            state.stored_cluster_keys.update(str(key) for key in state.case_cluster_keys if str(key))
        return state


def _generated_candidate_metadata(case: Case) -> dict[str, Any]:
    return {
        "seed_lineage": {
            "root_seed": case.seed,
            "parent_seed": None,
            "parent_case_id": "",
            "mutation_seed": None,
            "depth": 0,
        },
        "mutation": {
            "operator": "generated",
            "detail": "generated",
            "changed": False,
        },
        "feedback_selection": {},
        "feedback_decision": {},
    }


def _unique_nonempty(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value).strip()
        if not item or item in seen:
            continue
        out.append(item)
        seen.add(item)
    return out


def _bucket(prefix: str, value: int, limits: list[tuple[int, str]], fallback: str) -> str:
    for limit, name in limits:
        if value <= limit:
            return f"{prefix}:{name}"
    return f"{prefix}:{fallback}"


def _operator_target_stat_key(operator: str, target_key: str) -> str:
    return f"{operator}|{target_key}"


def _normalize_target_key(value: Any) -> str:
    return canonical_target_key(value)


def _normalize_target_keys(values: list[Any] | tuple[Any, ...]) -> list[str]:
    return _unique_nonempty([_normalize_target_key(value) for value in values])


def _normalize_int_counter(raw: dict[str, Any]) -> Counter[str]:
    counter: Counter[str] = Counter()
    if not isinstance(raw, dict):
        return counter
    for key, count in raw.items():
        normalized_key = str(key).strip()
        if not normalized_key:
            continue
        try:
            normalized_count = int(count)
        except (TypeError, ValueError):
            continue
        if normalized_count > 0:
            counter[normalized_key] += normalized_count
    return counter


def _normalize_float_counter(raw: dict[str, Any]) -> Counter[str]:
    counter: Counter[str] = Counter()
    if not isinstance(raw, dict):
        return counter
    for key, count in raw.items():
        normalized_key = str(key).strip()
        if not normalized_key:
            continue
        try:
            normalized_count = float(count)
        except (TypeError, ValueError):
            continue
        if normalized_count:
            counter[normalized_key] += normalized_count
    return counter


def _normalize_target_counter(raw: dict[str, Any]) -> Counter[str]:
    counter: Counter[str] = Counter()
    if not isinstance(raw, dict):
        return counter
    for key, count in raw.items():
        normalized_key = _normalize_target_key(key)
        if not normalized_key:
            continue
        try:
            normalized_count = int(count)
        except (TypeError, ValueError):
            continue
        if normalized_count > 0:
            counter[normalized_key] += normalized_count
    return counter


def _normalize_operator_target_stat_key(value: Any) -> str:
    text = str(value).strip()
    if not text or "|" not in text:
        return text
    operator, target_key = text.split("|", 1)
    normalized_operator = operator.strip()
    normalized_target_key = _normalize_target_key(target_key)
    if not normalized_operator:
        return normalized_target_key
    if not normalized_target_key:
        return normalized_operator
    return _operator_target_stat_key(normalized_operator, normalized_target_key)


def _legacy_target_key_alias(target_key: str) -> str:
    return semantic_signal_legacy_target_key_alias(target_key)


def _contextual_target_descriptors(values: list[Any] | tuple[Any, ...]) -> list[_ContextualTargetDescriptor]:
    descriptors: list[_ContextualTargetDescriptor] = []
    for target_key in _normalize_target_keys(values)[:8]:
        semantic_family = ""
        semantic_signal = ""
        exploration_objective = ""
        if target_key.startswith("semantic_family:"):
            semantic_family = target_key.removeprefix("semantic_family:").strip()
        elif target_key.startswith("feature:semantic_family:"):
            semantic_family = target_key.removeprefix("feature:semantic_family:").strip()
        elif target_key.startswith(CANONICAL_SEMANTIC_SIGNAL_PREFIX):
            semantic_signal = target_key.removeprefix(CANONICAL_SEMANTIC_SIGNAL_PREFIX).strip()
        elif target_key.startswith(EXPLORATION_OBJECTIVE_PREFIX):
            exploration_objective = target_key.removeprefix(EXPLORATION_OBJECTIVE_PREFIX).strip()
        elif target_key.startswith(f"feature:{EXPLORATION_OBJECTIVE_PREFIX}"):
            exploration_objective = target_key.removeprefix(f"feature:{EXPLORATION_OBJECTIVE_PREFIX}").strip()
        descriptors.append(
            _ContextualTargetDescriptor(
                key=target_key,
                weight=target_key_weight(target_key),
                semantic_family=semantic_family,
                semantic_signal=semantic_signal,
                exploration_objective=exploration_objective,
            )
        )
    return descriptors


def _operator_target_stat_keys(operator: str, target_key: str) -> tuple[str, ...]:
    normalized_target_key = _normalize_target_key(target_key)
    keys = [_operator_target_stat_key(operator, normalized_target_key)]
    legacy_target_key = _legacy_target_key_alias(normalized_target_key)
    if legacy_target_key:
        keys.append(_operator_target_stat_key(operator, legacy_target_key))
    return tuple(keys)


def _normalize_operator_target_counter(raw: dict[str, Any], *, value_type: type[int] | type[float]) -> Counter[str]:
    counter: Counter[str] = Counter()
    if not isinstance(raw, dict):
        return counter
    for key, count in raw.items():
        normalized_key = _normalize_operator_target_stat_key(key)
        if not normalized_key:
            continue
        try:
            normalized_count = value_type(count)
        except (TypeError, ValueError):
            continue
        if value_type is int and normalized_count <= 0:
            continue
        if normalized_count:
            counter[normalized_key] += normalized_count
    return counter


def _target_key_weight(target_key: str) -> float:
    return target_key_weight(target_key)


def _operator_target_reward_pull(
    reward_counter: Counter[str],
    pull_counter: Counter[str],
    operator: str,
    normalized_target_key: str,
) -> tuple[float, float]:
    primary_stat_key = _operator_target_stat_key(operator, normalized_target_key)
    pulls = pull_counter.get(primary_stat_key)
    if pulls:
        return float(reward_counter.get(primary_stat_key, 0.0)), float(pulls)
    legacy_target_key = _legacy_target_key_alias(normalized_target_key)
    if legacy_target_key:
        legacy_stat_key = _operator_target_stat_key(operator, legacy_target_key)
        legacy_pulls = pull_counter.get(legacy_stat_key)
        if legacy_pulls:
            return float(reward_counter.get(legacy_stat_key, 0.0)), float(legacy_pulls)
    return 0.0, 0.0


def _operator_target_counter_value(counter: Counter[str], operator: str, target_key: str) -> float:
    for stat_key in _operator_target_stat_keys(operator, target_key):
        if stat_key in counter:
            return float(counter[stat_key])
    return 0.0


def _case_profile_key(case: Case) -> str:
    metadata = case.metadata if isinstance(case.metadata, dict) else {}
    for field_name in ("mixed_generator_profile", "generator_profile"):
        value = str(metadata.get(field_name, "")).strip()
        if value:
            return value
    return ""


def _case_target_keys(case: Case) -> list[str]:
    metadata = case.metadata if isinstance(case.metadata, dict) else {}
    raw_values: list[Any] = []
    for field_name in (
        "target_keys",
        "matched_targets",
        "guidance_targets",
        "semantic_families",
        "exploration_objectives",
    ):
        value = metadata.get(field_name)
        if isinstance(value, list):
            if field_name == "exploration_objectives":
                raw_values.extend(f"{EXPLORATION_OBJECTIVE_PREFIX}{item}" for item in value)
            else:
                raw_values.extend(value)
        elif isinstance(value, str):
            items = value.split(",")
            if field_name == "exploration_objectives":
                raw_values.extend(f"{EXPLORATION_OBJECTIVE_PREFIX}{item}" for item in items)
            else:
                raw_values.extend(items)
    profile = _case_profile_key(case)
    if profile:
        raw_values.append(f"profile:{profile}")
    return _unique_nonempty([str(value) for value in raw_values])


def _case_cluster_key(case: Case, *, profile_key: str, target_keys: list[str]) -> str:
    profile_part = _cluster_token(profile_key or "generic")
    target_part = _case_cluster_target_part(target_keys)
    op_part = _case_cluster_operation_part(operation_names(case.program.operations, default="unknown"))
    return f"profile={profile_part}|targets={target_part}|ops={op_part}"


def _case_cluster_target_part(target_keys: list[str]) -> str:
    normalized = [
        target
        for target in _normalize_target_keys(target_keys)
        if target and not target.startswith("profile:")
    ]
    prioritized_targets = sorted(
        enumerate(normalized),
        key=lambda item: (-_case_cluster_target_priority(item[1]), item[0]),
    )
    stable_targets = [_cluster_token(target) for _, target in prioritized_targets]
    if not stable_targets:
        return "none"
    return "+".join(stable_targets[:3])


def _case_cluster_target_priority(target_key: str) -> int:
    if target_key.startswith("semantic_family:"):
        return 5
    if target_key.startswith(CANONICAL_SEMANTIC_SIGNAL_PREFIX):
        return 4
    if target_key.startswith(EXPLORATION_OBJECTIVE_PREFIX):
        return 4
    if target_key.startswith("capability:"):
        return 3
    if target_key.startswith("combo:"):
        return 2
    if target_key.startswith("target:"):
        return 1
    return 0


def _case_cluster_operation_part(op_names: list[str]) -> str:
    compact: list[str] = []
    counts: Counter[str] = Counter()
    for raw_name in op_names:
        name = _cluster_token(raw_name or "unknown")
        counts[name] += 1
        if compact and compact[-1] == name:
            continue
        compact.append(name)
    if not compact:
        return "empty"
    skeleton = ">".join(compact[:6])
    overflow = len(compact) - 6
    if overflow > 0:
        skeleton = f"{skeleton}>more{min(overflow, 9)}"
    repeated = [
        f"{name}{min(count, 9)}"
        for name, count in sorted(counts.items())
        if count > 1
    ]
    if repeated:
        skeleton = f"{skeleton};rep={','.join(repeated[:4])}"
    return skeleton


def _cluster_token(value: Any) -> str:
    text = str(value).strip().lower()
    if not text:
        return "none"
    chars: list[str] = []
    last_was_sep = False
    for char in text:
        if char.isalnum():
            chars.append(char)
            last_was_sep = False
        elif not last_was_sep:
            chars.append("_")
            last_was_sep = True
    token = "".join(chars).strip("_")
    return token[:96] if token else "none"


def _case_seed_utility(case: Case, *, has_finding: bool, family_keys: list[str]) -> float:
    operations = list(case.program.operations)
    op_names = operation_names(operations)
    utility = 1.0 + (8.0 if has_finding else 0.0)
    utility += 2.0 * len(family_keys)
    utility += min(4.0, 0.40 * len(operations))
    utility += min(2.0, 0.25 * sum(1 for table in case.tables for row in table.rows for value in row.values() if value is None))
    utility += 0.50 * max(0, len(case.tables) - 1)
    utility += 0.35 * len(set(op_names) & {"join", "semi_join", "anti_join", "groupby", "aggregate", "sort", "filter"})
    utility += 0.25 * sum(1 for name in op_names if name in {"mutate", "coalesce", "case_when", "fill_null", "distinct"})
    if case.metadata.get("mixed_generator_profile") or case.metadata.get("workflow_template"):
        utility += 0.50
    return utility


def _case_seed_schedule_prior(
    *,
    has_finding: bool,
    family_keys: list[str],
) -> float:
    return (4.0 if has_finding else 0.0) + min(4.0, 1.5 * len(family_keys))


def _bounded_confident_mean_reward(total_reward: float, pulls: float, *, max_abs: float) -> float:
    if pulls <= 0:
        return 0.0
    mean_reward = float(total_reward) / float(pulls)
    bounded_mean_reward = max(-max_abs, min(max_abs, mean_reward))
    confidence = min(1.0, math.log1p(float(pulls)) / math.log(4.0))
    return bounded_mean_reward * confidence
