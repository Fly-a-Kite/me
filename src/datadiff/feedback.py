from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from functools import lru_cache
import math
from typing import Any, Mapping

from datadiff.adaptive_learning import AdaptiveLearningState
from datadiff.behavioral_descriptor import BehavioralDescriptor, compute_behavioral_descriptor
from datadiff.champion_corpus import ChampionRegistry, ChampionSeed, champion_signature
from datadiff.discovery_rate import DiscoveryRateEstimator
from datadiff.dsl import Case
from datadiff.energy import operator_energy, seed_energy
from datadiff.exploration_objectives import EXPLORATION_OBJECTIVE_PREFIX
from datadiff.fingerprint import jaccard_distance
from datadiff.lineage import LineageDAG
from datadiff.mutator import (
    IR_REWRITE_MUTATION_OPERATOR_NAMES,
    mutate_case_with_metadata,
    mutation_operator_profiles,
)
from datadiff.multi_objective import (
    CostVector,
    ObjectiveVector,
    OPERATOR_OBJECTIVE_SPEC,
    SEED_FRONTIER_OBJECTIVE_SPEC,
    bounded_ratio,
    constrained_objective_score,
    multiplicative_cost_modifier,
    novelty_decay,
)
from datadiff.mutator_swarm import OperatorSwarm
from datadiff.operation_semantics import operation_names
from datadiff.quality_archive import QualityDiversityArchive
from datadiff.seed_corpus import SeedCorpusRecord, SeedPool
from datadiff.seed_budget import SEED_ENERGY_TIERS, SeedBudgetAllocator
from datadiff.seed_frontier import SeedFrontier
from datadiff.source_scheduler import LocalSourceScheduler
from datadiff.seed_quota import SeedEvictionPolicy
from datadiff.semantic_signal import (
    CANONICAL_SEMANTIC_SIGNAL_PREFIX,
    canonical_target_key,
    legacy_target_key_alias as semantic_signal_legacy_target_key_alias,
    target_key_weight,
)
from datadiff.util import CORPUS_DIR, dump_json
from datadiff.value_catalog import ADVERSARIAL_CATALOG, entry_context_features

MUTATION_OPERATOR_LEARNING_WEIGHT = 0.05
FINGERPRINT_REDUNDANT_DISTANCE = 0.15
BD_AXIS_BANDIT_SCOPE = "bd_axis_weights"
SEED_ENERGY_TIER_SCOPE = "seed_energy_tier"
CHAMPION_GRAFT_DONOR_SCOPE = "champion_graft_donor"
BD_AXIS_WEIGHT_BONUS = 0.35
DISCOVERY_EXPLORATION_BASE_WEIGHT = 0.45


@dataclass(frozen=True, slots=True)
class _ContextualTargetDescriptor:
    key: str
    weight: float
    semantic_family: str = ""
    semantic_signal: str = ""
    exploration_objective: str = ""


@dataclass(frozen=True, slots=True)
class _SelectedCandidate:
    case: Case
    source: str
    metadata: dict[str, Any]
    parent_index: int | None = None
    operator: str = ""
    swarm_particle_id: int | None = None


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
    champion_family_hits: Counter[str] = field(default_factory=Counter)
    champion_promoted_families: set[str] = field(default_factory=set)
    stored_profiles: Counter[str] = field(default_factory=Counter)
    case_utilities: list[float] = field(default_factory=list)
    case_family_keys: list[list[str]] = field(default_factory=list)
    case_profile_keys: list[str] = field(default_factory=list)
    case_target_keys: list[list[str]] = field(default_factory=list)
    case_cluster_keys: list[str] = field(default_factory=list)
    case_behavioral_descriptors: list[dict[str, Any]] = field(default_factory=list)
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
    last_feedback_swarm_particle_id: int | None = None
    mutation_operator_rewards: Counter[str] = field(default_factory=Counter)
    mutation_operator_pulls: Counter[str] = field(default_factory=Counter)
    mutation_operator_target_rewards: Counter[str] = field(default_factory=Counter)
    mutation_operator_target_pulls: Counter[str] = field(default_factory=Counter)
    stored_target_keys: Counter[str] = field(default_factory=Counter)
    stored_cluster_keys: Counter[str] = field(default_factory=Counter)
    cluster_schedule_feedback_totals: Counter[str] = field(default_factory=Counter)
    cluster_schedule_feedback_counts: Counter[str] = field(default_factory=Counter)
    quality_archive: QualityDiversityArchive = field(default_factory=QualityDiversityArchive)
    lineage: LineageDAG = field(default_factory=LineageDAG)
    discovery_rate_estimator: DiscoveryRateEstimator = field(default_factory=DiscoveryRateEstimator)
    adaptive_learning: AdaptiveLearningState = field(default_factory=AdaptiveLearningState)
    operator_swarm: OperatorSwarm = field(
        default_factory=lambda: OperatorSwarm.init(
            mutation_operator_profiles(allow_probe_operators=False).keys(),
            n_particles=16,
        )
    )
    enable_mutation_operator_learning: bool = True
    enable_operator_swarm: bool = True
    enable_ir_rewrite_mutations: bool = True
    enable_divergence_conditioned_mutations: bool = True
    enable_shrink_mutations: bool = True
    enable_value_catalog: bool = True
    enable_quality_archive: bool = True
    enable_hierarchical_archive: bool = True
    enable_bd_axis_bandit: bool = True
    enable_bayesian_exploration: bool = True
    enable_seed_quota: bool = True
    enable_seed_energy_batch: bool = True
    enable_seed_energy_tier_bandit: bool = True
    enable_per_operator_energy: bool = True
    enable_lineage_rarity: bool = True
    enable_minhash_dedup: bool = True
    enable_disagreement_bd_axis: bool = True
    enable_champion_corpus: bool = True
    enable_champion_graft_donor_bandit: bool = True
    seed_eviction_policy: SeedEvictionPolicy = field(default_factory=SeedEvictionPolicy)
    champion_registry: ChampionRegistry | None = None
    champion_version_id: str = ""
    champion_promotion_threshold: int = 3
    last_feedback_decision: dict[str, Any] = field(default_factory=dict)
    persisted_count: int = 0
    last_persisted_to_disk: bool = False
    last_record_skip_reason: str = ""
    source_scheduler: LocalSourceScheduler | None = None
    last_candidate_source: str = "generated"
    last_candidate_metadata: dict[str, Any] = field(default_factory=dict)
    last_candidate_batch_metadata: list[dict[str, Any]] = field(default_factory=list)
    last_candidate_batch_sources: list[str] = field(default_factory=list)
    pending_candidate_batch: deque[_SelectedCandidate] = field(default_factory=deque, repr=False)
    last_source_reward: float | None = None
    seed_frontier: SeedFrontier = field(init=False, repr=False)
    seed_budget_allocator: SeedBudgetAllocator = field(init=False, repr=False)
    seed_frontier_heap: list[tuple[float, float, float, int, int, int]] = field(default_factory=list, repr=False)
    seed_frontier_dirty: bool = field(default=True, repr=False)
    seed_metadata_aligned_count: int = field(default=-1, repr=False)
    mutation_operator_score_cache: dict[tuple[Any, ...], dict[str, float]] = field(default_factory=dict, repr=False)
    value_catalog_score_cache: dict[tuple[Any, ...], dict[str, float]] = field(default_factory=dict, repr=False)
    mutation_swarm_choice_cache: dict[tuple[Any, ...], str] = field(default_factory=dict, repr=False)
    mutation_score_cache_epoch: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        self.enable_ir_rewrite_mutations = bool(self.enable_ir_rewrite_mutations)
        self.enable_seed_energy_tier_bandit = bool(self.enable_seed_energy_tier_bandit)
        self.enable_champion_graft_donor_bandit = bool(self.enable_champion_graft_donor_bandit)
        self.quality_archive.enable_hierarchical = bool(self.enable_hierarchical_archive)
        self.seed_frontier = SeedFrontier(
            priority_builder=self._seed_frontier_priority,
            row_builder=self._seed_frontier_row,
        )
        self.seed_budget_allocator = SeedBudgetAllocator(
            tier_selector=self._choose_seed_energy_tier,
        )
        self.seed_eviction_policy.enabled = bool(
            self.enable_seed_quota and self.seed_eviction_policy.enabled
        )
        self.operator_swarm.ensure_operators(self._mutation_operator_profiles())
        self._sync_seed_frontier_component_state()

    def _mutation_operator_profiles(self):
        return mutation_operator_profiles(
            allow_probe_operators=False,
            enable_ir_rewrite_mutations=self.enable_ir_rewrite_mutations,
            enable_shrink_mutations=self.enable_shrink_mutations,
        )

    def select_case(self, seed: int, generated: Case) -> Case:
        if self.pending_candidate_batch:
            return self._commit_selected_candidate(self.pending_candidate_batch.popleft()).case
        batch = self._select_case_batch_items(seed, generated, max_batch=1)
        if not batch:
            batch = [self._generated_selected_candidate(generated)]
        return self._commit_selected_candidate(batch[0]).case

    def select_case_batch(
        self,
        seed: int,
        generated: Case,
        *,
        max_batch: int | None = None,
        enqueue_remaining: bool = False,
    ) -> list[Case]:
        if self.pending_candidate_batch:
            selected = self._commit_selected_candidate(self.pending_candidate_batch.popleft())
            return [selected.case]
        batch = self._select_case_batch_items(seed, generated, max_batch=max_batch)
        if not batch:
            batch = [self._generated_selected_candidate(generated)]
        self.pending_candidate_batch.clear()
        if enqueue_remaining:
            self.pending_candidate_batch.extend(batch[1:])
        self._commit_selected_candidate(batch[0])
        self.last_candidate_batch_metadata = [dict(item.metadata) for item in batch]
        self.last_candidate_batch_sources = [item.source for item in batch]
        return [item.case for item in batch]

    def pop_pending_candidate(self) -> _SelectedCandidate | None:
        if not self.pending_candidate_batch:
            return None
        return self._commit_selected_candidate(self.pending_candidate_batch.popleft())

    def activate_selected_candidate(self, case: Case, metadata: dict[str, Any]) -> None:
        resolved_metadata = dict(metadata or {})
        source = str(resolved_metadata.get("source", "generated") or "generated")
        self.last_candidate_source = source
        self.last_candidate_metadata = resolved_metadata
        self.last_candidate_batch_metadata = [resolved_metadata]
        self.last_candidate_batch_sources = [source]
        if source != "feedback_mutation":
            self.last_feedback_parent_index = None
            self.last_feedback_operator = ""
            self.last_feedback_swarm_particle_id = None
            self.last_feedback_decision = {}
            return
        decision = (
            resolved_metadata.get("feedback_decision")
            or {}
        )
        if not isinstance(decision, dict):
            decision = {}
        lineage = resolved_metadata.get("seed_lineage", {})
        if not isinstance(lineage, dict):
            lineage = {}
        mutation = resolved_metadata.get("mutation", {})
        if not isinstance(mutation, dict):
            mutation = {}
        self.last_feedback_parent_index = _optional_nonnegative_int(
            decision.get("parent_index", lineage.get("parent_index"))
        )
        self.last_feedback_operator = str(
            decision.get("selected_operator", mutation.get("operator", "")) or ""
        )
        self.last_feedback_swarm_particle_id = _optional_nonnegative_int(
            decision.get("swarm_particle_id")
        )
        self.last_feedback_decision = dict(decision)

    def _select_case_batch_items(
        self,
        seed: int,
        generated: Case,
        *,
        max_batch: int | None,
    ) -> list[_SelectedCandidate]:
        requested_batch = max(1, int(max_batch or 1))
        if not self.enable_seed_energy_batch:
            requested_batch = 1
        if not self.interesting_cases:
            return [self._generated_selected_candidate(generated)]
        if self.source_scheduler is None:
            if seed % 3 != 0:
                return [self._generated_selected_candidate(generated)]
        else:
            source = self.source_scheduler.choose_source(feedback_available=bool(self.interesting_cases))
            if source != "feedback_mutation":
                return [self._generated_selected_candidate(generated)]
        champion_donors = self._champion_donors_for_mutation_batch()
        excluded_indexes: set[int] = set()
        batch: list[_SelectedCandidate] = []
        mutation_seed = int(seed)
        max_parent_attempts = min(4, len(self.interesting_cases))
        for parent_attempt in range(max_parent_attempts):
            base_index = self._choose_mutation_seed_index(mutation_seed, excluded_indexes=excluded_indexes)
            target_count = min(requested_batch - len(batch), max(1, self._case_seed_energy(base_index)))
            local_attempt_limit = max(target_count + 4, 4)
            for local_attempt in range(local_attempt_limit):
                if len(batch) >= requested_batch or len(batch) >= target_count:
                    break
                selected = self._mutate_selected_candidate(
                    base_index,
                    mutation_seed=mutation_seed,
                    attempt=parent_attempt + local_attempt,
                    champion_donors=champion_donors,
                )
                mutation_seed += 1
                if selected is not None:
                    batch.append(selected)
            if batch or len(batch) >= requested_batch:
                break
            excluded_indexes.add(base_index)
            mutation_seed += 1
        return batch or [self._generated_selected_candidate(generated)]

    def _mutate_selected_candidate(
        self,
        base_index: int,
        *,
        mutation_seed: int,
        attempt: int,
        champion_donors: list[ChampionSeed] | None = None,
    ) -> _SelectedCandidate | None:
        self._align_seed_metadata_lengths()
        base = self.interesting_cases[base_index]
        parent_target_keys = self.case_target_keys[base_index] if base_index < len(self.case_target_keys) else []
        swarm_particle_id = self._choose_mutation_swarm_particle_id(
            base_index,
            target_keys=parent_target_keys,
            mutation_seed=mutation_seed,
        )
        operator_scores = self._mutation_operator_score_snapshot(
            target_keys=parent_target_keys,
            swarm_particle_id=swarm_particle_id,
        )
        plan_depth = self._mutation_plan_depth(base_index, target_keys=parent_target_keys, operator_scores=operator_scores)
        disagreement_descriptor = (
            base.metadata.get("disagreement_descriptor") if isinstance(base.metadata, dict) else None
        )
        value_catalog_scores = (
            self._value_catalog_score_snapshot(
                target_keys=parent_target_keys,
                disagreement=disagreement_descriptor,
            )
            if self.enable_value_catalog
            else None
        )
        result = self._invoke_mutator(
            base,
            mutation_seed,
            operator_scores=operator_scores,
            target_keys=parent_target_keys,
            plan_depth=plan_depth,
            disagreement_descriptor=disagreement_descriptor,
            value_catalog_scores=value_catalog_scores,
            champion_donors=champion_donors,
        )
        if not result.metadata.get("mutation", {}).get("changed"):
            return None
        self._record_mutation_seed_pull(base_index)
        operator = str(result.metadata.get("mutation", {}).get("operator", ""))
        if operator:
            self._record_recent_operator_pull(operator)
        decision = self._feedback_decision_snapshot(
            base_index,
            mutation_seed=mutation_seed,
            attempt=attempt,
            operator_scores=operator_scores,
            plan_depth=plan_depth,
        )
        decision["selected_operator"] = operator
        decision["swarm_particle_id"] = swarm_particle_id
        decision["selected_operator_score"] = operator_scores.get(
            operator,
            operator_scores.get("__untried__", 0.0),
        )
        decision["recent_operator_pulls"] = self._recent_operator_pull_count(operator)
        decision["mutation_plan"] = dict(
            result.metadata.get("mutation_plan", result.metadata.get("mutation_selection", {}).get("plan", {})) or {}
        )
        decision["value_catalog_entry_count"] = len(
            result.metadata.get("mutation", {}).get("value_catalog_entries", []) or []
        )
        seed_lineage = dict(result.metadata.get("seed_lineage", {}) or {})
        seed_lineage["parent_index"] = base_index
        result.metadata["seed_lineage"] = seed_lineage
        metadata = dict(result.metadata)
        metadata["feedback_decision"] = dict(decision)
        if result.case is not base:
            result.case.metadata = metadata
        return _SelectedCandidate(
            case=result.case,
            source="feedback_mutation",
            metadata=metadata,
            parent_index=base_index,
            operator=operator,
            swarm_particle_id=swarm_particle_id,
        )

    def _invoke_mutator(
        self,
        base: Case,
        mutation_seed: int,
        *,
        operator_scores: dict[str, float],
        target_keys: list[str],
        plan_depth: int,
        disagreement_descriptor: Any | None,
        value_catalog_scores: dict[str, float] | None,
        champion_donors: list[ChampionSeed] | None = None,
    ) -> Any:
        try:
            resolved_champion_donors = list(champion_donors or [])
            champion_donor_selection: dict[str, Any] = {}
            if resolved_champion_donors:
                resolved_champion_donors, champion_donor_selection = self._choose_champion_graft_donors(
                    base,
                    resolved_champion_donors,
                    target_keys=target_keys,
                )
            result = mutate_case_with_metadata(
                base,
                mutation_seed,
                allow_probe_operators=False,
                operator_scores=operator_scores,
                target_keys=target_keys,
                plan_depth=plan_depth,
                disagreement=disagreement_descriptor,
                operator_pulls=self.mutation_operator_pulls,
                recent_operator_pulls=self.recent_mutation_operator_counts,
                champion_donors=resolved_champion_donors,
                enable_ir_rewrite_mutations=self.enable_ir_rewrite_mutations,
                enable_divergence_conditioned_mutations=self.enable_divergence_conditioned_mutations,
                enable_shrink_mutations=self.enable_shrink_mutations,
                enable_per_operator_energy=self.enable_per_operator_energy,
                enable_value_catalog=self.enable_value_catalog,
                value_catalog_scores=value_catalog_scores,
            )
            mutation_metadata = result.metadata.get("mutation", {}) if isinstance(result.metadata, dict) else {}
            mutation_operator = (
                str(mutation_metadata.get("operator", "") or "")
                if isinstance(mutation_metadata, dict)
                else ""
            )
            if champion_donor_selection and mutation_operator == "champion_graft":
                metadata = dict(result.metadata or {})
                metadata["champion_graft_selection"] = dict(champion_donor_selection)
                result.metadata = metadata
                if isinstance(result.case.metadata, dict):
                    result.case.metadata = {**result.case.metadata, "champion_graft_selection": dict(champion_donor_selection)}
            return result
        except TypeError as exc:
            if "unexpected keyword argument" not in str(exc):
                raise
            try:
                return mutate_case_with_metadata(
                    base,
                    mutation_seed,
                    allow_probe_operators=False,
                    operator_scores=operator_scores,
                    target_keys=target_keys,
                    plan_depth=plan_depth,
                    disagreement=disagreement_descriptor,
                    operator_pulls=self.mutation_operator_pulls,
                    recent_operator_pulls=self.recent_mutation_operator_counts,
                )
            except TypeError as retry_exc:
                if "unexpected keyword argument" not in str(retry_exc):
                    raise
                return mutate_case_with_metadata(
                    base,
                    mutation_seed,
                    allow_probe_operators=False,
                    operator_scores=operator_scores,
                )

    def _champion_donors_for_mutation_batch(self) -> list[ChampionSeed]:
        if not (self.enable_champion_corpus and self.champion_registry is not None):
            return []
        return self.champion_registry.champions_for_version(self.champion_version_id, limit=4)

    def _commit_selected_candidate(self, selected: _SelectedCandidate) -> _SelectedCandidate:
        self.last_candidate_source = selected.source
        self.last_candidate_metadata = dict(selected.metadata)
        self.last_candidate_batch_metadata = [dict(selected.metadata)]
        self.last_candidate_batch_sources = [selected.source]
        self.last_feedback_parent_index = selected.parent_index
        self.last_feedback_operator = selected.operator
        self.last_feedback_swarm_particle_id = selected.swarm_particle_id
        self.last_feedback_decision = dict(
            selected.metadata.get("feedback_decision", {}) or {}
        )
        return selected

    def _generated_selected_candidate(self, generated: Case) -> _SelectedCandidate:
        return _SelectedCandidate(
            case=generated,
            source="generated",
            metadata=_generated_candidate_metadata(generated),
        )

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
        recent_cluster_pulls = self._recent_cluster_pull_count(cluster_key)
        cluster_feedback_count = int(self.cluster_schedule_feedback_counts[cluster_key])
        archive_cluster_reward = self.quality_archive.cluster_reward_signal(cluster_key)
        descriptor = compute_behavioral_descriptor(
            case,
            profile_key=resolved_profile_key,
            target_keys=normalized_target_keys,
        ).to_dict()
        archive_axis_weights = self._bd_axis_weights(
            descriptor,
            cluster_key=cluster_key,
            target_keys=normalized_target_keys,
        )
        archive_axis_reward = self.quality_archive.composite_reward_signal(
            descriptor,
            axis_weights=archive_axis_weights,
        )
        return {
            "cluster_key": cluster_key,
            "behavioral_descriptor": descriptor,
            "profile_key": resolved_profile_key,
            "target_keys": normalized_target_keys,
            "target_key_count": len(normalized_target_keys),
            "archive_enabled": True,
            "archive_known": self.quality_archive.has_cluster(cluster_key),
            "archive_elite_indexes": archive_elite_indexes,
            "archive_seed_count": self.quality_archive.seed_count(cluster_key),
            "archive_outcome_count": self.quality_archive.outcome_count(cluster_key),
            "archive_cluster_reward": archive_cluster_reward,
            "archive_axis_reward": archive_axis_reward,
            "archive_axis_weights": dict(archive_axis_weights),
            "archive_health_penalty": self.quality_archive.cluster_health_penalty(cluster_key),
            "cluster_count": int(self.stored_cluster_keys[cluster_key]),
            "cluster_feedback_reward": self._cluster_feedback_reward_signal(cluster_key),
            "cluster_feedback_count": cluster_feedback_count,
            "cluster_novelty_score": min(1.25, 0.35 / (1.0 + self.stored_cluster_keys[cluster_key])),
            "recent_cluster_pulls": int(recent_cluster_pulls),
        }

    def _apply_discovery_exploration_weight(self) -> None:
        exploration_weight = (
            self.discovery_rate_estimator.adaptive_exploration_weight(
                DISCOVERY_EXPLORATION_BASE_WEIGHT
            )
            if self.enable_bayesian_exploration
            else DISCOVERY_EXPLORATION_BASE_WEIGHT
        )
        for bandit in self.adaptive_learning.bandits.values():
            bandit.exploration_weight = exploration_weight

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
        disagreement_descriptor: Any | None = None,
        case_fingerprint: Any | None = None,
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
        _attach_disagreement_descriptor(case, disagreement_descriptor)
        _attach_case_fingerprint(case, case_fingerprint)
        family_keys = _unique_nonempty(candidate_bug_families or [])
        self.discovery_rate_estimator.observe(family_keys)
        self._apply_discovery_exploration_weight()
        if not has_finding and not family_keys and self._is_fingerprint_redundant(case):
            self.last_record_skip_reason = "fingerprint_redundant"
            return False
        normalized_target_keys = _normalize_target_keys(target_keys or _case_target_keys(case))
        family_limit = max(0, int(self.max_cases_per_candidate_family))
        if family_limit and family_keys and all(
            self.stored_candidate_bug_families[family] >= family_limit for family in family_keys
        ):
            self.last_record_skip_reason = "candidate_family_saturated"
            return False
        profile_key = _case_profile_key(case)
        cluster_key = _case_cluster_key(case, profile_key=profile_key, target_keys=normalized_target_keys)
        descriptor = compute_behavioral_descriptor(
            case,
            profile_key=profile_key,
            target_keys=normalized_target_keys,
        ).to_dict()
        descriptor = self._normalize_behavioral_descriptor(descriptor)
        profile_limit = max(0, int(self.max_cases_per_profile))
        if (
            profile_limit
            and profile_key
            and not has_finding
            and self.stored_profiles[profile_key] >= profile_limit
        ):
            self.last_record_skip_reason = "profile_saturated"
            return False
        utility = _case_seed_utility(case, has_finding=has_finding, family_keys=family_keys)
        record = self._seed_corpus_record(
            case,
            has_finding=has_finding,
            family_keys=family_keys,
            profile_key=profile_key,
            target_keys=normalized_target_keys,
            cluster_key=cluster_key,
            descriptor=descriptor,
            utility=utility,
            schedule_delta=schedule_delta,
        )
        if len(self.interesting_cases) < self.max_corpus:
            self._seed_pool_view().append_seed(record)
        elif has_finding:
            replace_index = self._evict_seed_index(
                incoming_cluster_key=cluster_key,
                incoming_utility=utility,
            )
            if replace_index is None:
                self.last_record_skip_reason = "corpus_full_no_evictable_seed"
                return False
            self._replace_feedback_seed(
                replace_index,
                case,
                has_finding=has_finding,
                family_keys=family_keys,
                profile_key=profile_key,
                target_keys=normalized_target_keys,
                cluster_key=cluster_key,
                descriptor=descriptor,
                utility=utility,
                schedule_delta=schedule_delta,
            )
        else:
            replace_index = self._evict_seed_index(
                incoming_cluster_key=cluster_key,
                incoming_utility=utility,
            )
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
                descriptor=descriptor,
                utility=utility,
                schedule_delta=schedule_delta,
            )
        self._seed_pool_view().record_stored_counters(record)
        self.seed_metadata_aligned_count = len(self.interesting_cases)
        if has_finding:
            self._record_champion_family_hits(case, family_keys)
        if self.persist_to_disk and self.persisted_count < max(0, self.max_persisted):
            self._write_interesting_case(
                case,
                behavior_signature,
                has_finding,
                discovery_signature=str(discovery_signature or novelty_key),
            )
            self.persisted_count += 1
            self.last_persisted_to_disk = True
        self._mark_seed_frontier_dirty()
        return True

    def inject_champion_seed(self, champion: ChampionSeed) -> bool:
        case = champion.case()
        metadata = dict(case.metadata or {})
        metadata["candidate_source"] = "champion_corpus"
        metadata["seed_lineage"] = {
            "root_seed": case.seed,
            "parent_seed": case.seed,
            "parent_case_id": champion.case_id,
            "mutation_seed": case.seed,
            "depth": 0,
            "champion": True,
        }
        case.metadata = metadata
        return self.record(
            case,
            champion_signature(champion),
            False,
            novelty_signature=champion_signature(champion),
            discovery_signature=champion_signature(champion),
            candidate_bug_families=list(champion.bug_family_keys),
            target_keys=[f"champion_family:{family}" for family in champion.bug_family_keys],
            schedule_delta=1.0 + min(3.0, 0.25 * champion.stability),
        )

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
        descriptor: dict[str, Any],
        utility: float | None = None,
        schedule_delta: float = 0.0,
    ) -> None:
        self._align_seed_metadata_lengths()
        resolved_utility = (
            float(utility)
            if utility is not None
            else _case_seed_utility(case, has_finding=has_finding, family_keys=family_keys)
        )
        self._seed_pool_view().replace_seed(
            index,
            self._seed_corpus_record(
                case,
                has_finding=has_finding,
                family_keys=family_keys,
                profile_key=profile_key,
                target_keys=target_keys,
                cluster_key=cluster_key,
                descriptor=descriptor,
                utility=resolved_utility,
                schedule_delta=schedule_delta,
            ),
        )
        self._mark_seed_frontier_dirty()

    def _least_useful_seed_index(self) -> int | None:
        return self._seed_pool_view().least_useful_seed_index()

    def _evict_seed_index(self, *, incoming_cluster_key: str, incoming_utility: float) -> int | None:
        if not self.interesting_cases or self.max_corpus <= 0:
            return None
        self._align_seed_metadata_lengths()
        return self.seed_eviction_policy.evict_candidate(
            case_cluster_keys=self.case_cluster_keys,
            case_utilities=self.case_utilities,
            case_mutation_pulls=self.case_mutation_pulls,
            incoming_cluster_key=incoming_cluster_key,
            incoming_utility=incoming_utility,
            max_corpus=int(self.max_corpus),
            archive=self.quality_archive if self.enable_quality_archive else None,
        )

    def _seed_corpus_record(
        self,
        case: Case,
        *,
        has_finding: bool,
        family_keys: list[str],
        profile_key: str,
        target_keys: list[str],
        cluster_key: str,
        descriptor: dict[str, Any],
        utility: float,
        schedule_delta: float = 0.0,
    ) -> SeedCorpusRecord:
        return SeedCorpusRecord(
            case=case,
            family_keys=list(family_keys),
            profile_key=profile_key,
            target_keys=list(target_keys),
            cluster_key=cluster_key,
            behavioral_descriptor=dict(descriptor),
            utility=float(utility),
            schedule_reward=_case_seed_schedule_prior(
                has_finding=has_finding,
                family_keys=family_keys,
            )
            + float(schedule_delta),
            parent_index=_lineage_parent_index(case),
        )

    def _seed_pool_view(self) -> SeedPool:
        return SeedPool(
            max_corpus=self.max_corpus,
            cases=self.interesting_cases,
            utilities=self.case_utilities,
            family_keys=self.case_family_keys,
            profile_keys=self.case_profile_keys,
            target_keys=self.case_target_keys,
            cluster_keys=self.case_cluster_keys,
            behavioral_descriptors=self.case_behavioral_descriptors,
            mutation_pulls=self.case_mutation_pulls,
            schedule_rewards=self.case_schedule_rewards,
            schedule_feedback_totals=self.case_schedule_feedback_totals,
            schedule_feedback_counts=self.case_schedule_feedback_counts,
            stored_candidate_bug_families=self.stored_candidate_bug_families,
            stored_profiles=self.stored_profiles,
            stored_target_keys=self.stored_target_keys,
            stored_cluster_keys=self.stored_cluster_keys,
            quality_archive=self.quality_archive if self.enable_quality_archive else None,
            lineage=self.lineage,
            seed_eviction_policy=(
                self.seed_eviction_policy
                if self.enable_seed_quota
                else None
            ),
        )

    @property
    def seed_pool(self) -> SeedPool:
        return self._seed_pool_view()

    def _record_champion_family_hits(self, case: Case, family_keys: list[str]) -> None:
        if not family_keys:
            return
        if not self.enable_champion_corpus:
            return
        threshold = max(1, int(self.champion_promotion_threshold or 3))
        for family in family_keys:
            self.champion_family_hits[family] += 1
            if self.champion_family_hits[family] < threshold:
                continue
            if family in self.champion_promoted_families:
                continue
            if self.champion_registry is None:
                continue
            promoted = self.champion_registry.promote_if_stable(
                case,
                [family],
                threshold=threshold,
                version_id=self.champion_version_id,
                stability=self.champion_family_hits[family],
            )
            if promoted:
                self.champion_promoted_families.add(family)

    def _is_fingerprint_redundant(self, case: Case) -> bool:
        if not self.enable_minhash_dedup:
            return False
        fingerprint = _case_fingerprint_payload(case)
        if fingerprint is None:
            return False
        for stored in self.interesting_cases:
            stored_fingerprint = _case_fingerprint_payload(stored)
            if stored_fingerprint is None:
                continue
            if jaccard_distance(fingerprint, stored_fingerprint) < FINGERPRINT_REDUNDANT_DISTANCE:
                return True
        return False

    def _choose_mutation_seed_index(self, seed: int, *, excluded_indexes: set[int] | None = None) -> int:
        if not self.interesting_cases:
            raise ValueError("feedback corpus is empty")
        self._align_seed_metadata_lengths()
        excluded = excluded_indexes or set()
        snapshot = self._seed_frontier_snapshot(limit=max(8, len(excluded) + 4), excluded_indexes=excluded)
        if not snapshot:
            candidate_indexes = [index for index in range(len(self.interesting_cases)) if index not in excluded]
            if not candidate_indexes:
                raise ValueError("feedback corpus is empty after exclusions")
            return max(
                candidate_indexes,
                key=lambda index: (
                    self._case_seed_schedule_score(index),
                    -self.case_mutation_pulls[index],
                    -index,
                ),
            )
        top_score = float(snapshot[0]["schedule_score"])
        near_ties = [
            row for row in snapshot
            if (top_score - float(row["schedule_score"])) <= 0.03
        ]
        chosen = near_ties[seed % len(near_ties)] if near_ties else snapshot[0]
        return int(chosen["index"])

    def _case_seed_schedule_score(self, index: int) -> float:
        reward_prior = self._case_seed_schedule_reward(index)
        target_novelty = self._case_target_novelty_score(index)
        cluster_key = self._case_cluster_key_at(index)
        archive_cluster_reward = (
            self.quality_archive.cluster_reward_signal(cluster_key)
            if self.enable_quality_archive
            else 0.0
        )
        archive_axis_reward = (
            self.quality_archive.composite_reward_signal(
                self._case_behavioral_descriptor_at(index),
                axis_weights=self._bd_axis_weights_for_seed(index),
            )
            if self.enable_quality_archive
            else 0.0
        )
        cluster_reward = max(
            self._cluster_feedback_reward_signal(cluster_key),
            archive_cluster_reward,
            archive_axis_reward,
        )
        cluster_novelty = self._case_cluster_novelty_score(index)
        elite_bonus = (
            self.quality_archive.seed_elite_bonus(cluster_key, index)
            if self.enable_quality_archive
            else 0.0
        )
        lineage_rarity = self.lineage.rarity_score(index) if self.enable_lineage_rarity else 0.0
        archive_health_penalty = (
            self.quality_archive.cluster_health_penalty(cluster_key)
            if self.enable_quality_archive
            else 0.0
        )
        family_reuse_penalty = self._case_family_reuse_penalty(index)
        mutation_pulls = self.case_mutation_pulls[index]
        recent_parent_pulls = self._recent_parent_pull_count(index)
        recent_cluster_pulls = self._recent_cluster_pull_count(cluster_key)
        discovery_signal = min(
            1.75,
            bounded_ratio(reward_prior, 4.0, upper=1.5)
            + 0.55 * bounded_ratio(cluster_reward, 1.5)
            + 0.30 * bounded_ratio(cluster_novelty, 1.25)
            + 0.25 * bounded_ratio(lineage_rarity, 1.5),
        )
        semantic_signal = min(1.35, target_novelty)
        behavior_signal = min(
            1.10,
            0.45 * bounded_ratio(cluster_reward, 1.5)
            + 0.40 * bounded_ratio(elite_bonus, 1.0)
            + 0.25 * bounded_ratio(reward_prior, 4.0),
        )
        structural_signal = min(
            1.25,
            0.50 * bounded_ratio(cluster_novelty, 1.25)
            + 0.30 * bounded_ratio(elite_bonus, 1.0)
            + 0.30 * bounded_ratio(lineage_rarity, 1.5)
            + 0.20 * bounded_ratio(self._case_seed_energy(index), 4.0),
        )
        vector = ObjectiveVector(
            discovery=discovery_signal,
            semantic=semantic_signal,
            novelty=min(1.0, 0.70 * bounded_ratio(cluster_novelty, 1.25) + 0.30 * bounded_ratio(target_novelty, 1.35)),
            expandability=min(1.0, 0.60 * bounded_ratio(self._case_seed_energy(index), 4.0) + 0.25 * bounded_ratio(lineage_rarity, 1.5)),
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
        energy_multiplier = 1.0 + (0.04 * max(0, self._case_seed_energy(index) - 1))
        return score * energy_multiplier

    def _case_family_reuse_penalty(self, index: int) -> float:
        if index >= len(self.case_family_keys):
            return 1.0
        families = [family for family in self.case_family_keys[index] if family]
        if not families:
            return 1.0
        family_limit = max(1, int(self.max_cases_per_candidate_family or 8))
        max_family_count = max(int(self.stored_candidate_bug_families[family]) for family in families)
        if max_family_count <= 1:
            return 1.0
        saturation_ratio = max_family_count / float(family_limit)
        return 1.0 + min(2.0, 0.60 * math.log1p(saturation_ratio))

    def _case_seed_energy(self, index: int) -> int:
        self._align_seed_metadata_lengths()
        cluster_key = self._case_cluster_key_at(index)
        family_breadth = len(
            {
                *(
                    self.case_family_keys[index]
                    if index < len(self.case_family_keys)
                    else []
                ),
                *(
                    self.case_target_keys[index]
                    if index < len(self.case_target_keys)
                    else []
                ),
            }
        )
        cluster_outcome_count = (
            self.quality_archive.outcome_count(cluster_key)
            if self.enable_quality_archive and cluster_key
            else int(self.cluster_schedule_feedback_counts[cluster_key])
        )
        pulls = self.case_mutation_pulls[index] if index < len(self.case_mutation_pulls) else 0
        feedback_count = (
            self.case_schedule_feedback_counts[index]
            if index < len(self.case_schedule_feedback_counts)
            else 0
        )
        return self.seed_budget_allocator.energy_for_seed(
            index,
            pulls=pulls,
            mean_reward=self._case_seed_schedule_reward(index),
            family_breadth=family_breadth,
            cluster_outcome_count=cluster_outcome_count,
            since_last_finding_pulls=max(0, pulls - feedback_count),
        )

    def _choose_seed_energy_tier(self, index: int) -> str:
        if not (
            self.enable_seed_energy_batch
            and self.enable_seed_energy_tier_bandit
            and self.enable_mutation_operator_learning
            and self._learning_scope_has_feedback(SEED_ENERGY_TIER_SCOPE)
        ):
            return "med"
        choice = self.adaptive_learning.choose(
            SEED_ENERGY_TIER_SCOPE,
            action_ids=SEED_ENERGY_TIERS,
            context_features=self._seed_energy_tier_context_features(index),
        )
        return choice if choice in SEED_ENERGY_TIERS else "med"

    def _seed_energy_tier_context_features(self, index: int) -> tuple[str, ...]:
        self._align_seed_metadata_lengths()
        cluster_key = self._case_cluster_key_at(index)
        pulls = self.case_mutation_pulls[index] if index < len(self.case_mutation_pulls) else 0
        feedback_count = (
            self.case_schedule_feedback_counts[index]
            if index < len(self.case_schedule_feedback_counts)
            else 0
        )
        cluster_outcome_count = (
            self.quality_archive.outcome_count(cluster_key)
            if self.enable_quality_archive and cluster_key
            else int(self.cluster_schedule_feedback_counts[cluster_key])
        )
        target_keys = self.case_target_keys[index] if index < len(self.case_target_keys) else []
        profile_key = self.case_profile_keys[index] if index < len(self.case_profile_keys) else ""
        features = [
            _bucket("seed_pulls", pulls, [(0, "cold"), (2, "warm"), (8, "hot")], "saturated"),
            _bucket("seed_feedback", feedback_count, [(0, "none"), (2, "few"), (8, "some")], "many"),
            _bucket("cluster_outcomes", cluster_outcome_count, [(0, "none"), (2, "few"), (8, "some")], "many"),
            _bucket(
                "recent_pull_rate",
                self._recent_parent_pull_count(index) + self._recent_cluster_pull_count(cluster_key),
                [(0, "idle"), (2, "light"), (6, "busy")],
                "hot",
            ),
            _bucket("target_key_count", len(target_keys), [(0, "none"), (2, "few"), (6, "some")], "many"),
            _float_bucket("schedule_reward", self._case_seed_schedule_reward(index)),
        ]
        if cluster_key:
            features.append(f"cluster:{cluster_key}")
        if profile_key:
            features.append(f"profile:{profile_key}")
        for target_key in target_keys[:4]:
            features.append(f"target:{target_key}")
        return tuple(_unique_nonempty(features))

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
        novelty = 0.0
        for target in _contextual_target_descriptors(target_keys[:8]):
            novelty += target.weight * novelty_decay(
                self.stored_target_keys[target.key],
                scale=0.32,
                floor=0.02,
            )
        return min(1.5, novelty)

    def _case_cluster_key_at(self, index: int) -> str:
        self._align_seed_metadata_lengths()
        if index < len(self.case_cluster_keys):
            return self.case_cluster_keys[index]
        return ""

    def _case_behavioral_descriptor_at(self, index: int) -> dict[str, Any]:
        self._align_seed_metadata_lengths()
        if index < len(self.case_behavioral_descriptors):
            return self._normalize_behavioral_descriptor(self.case_behavioral_descriptors[index])
        return self._normalize_behavioral_descriptor(BehavioralDescriptor.empty().to_dict())

    def _normalize_behavioral_descriptor(self, descriptor: dict[str, Any]) -> dict[str, Any]:
        return _normalize_behavioral_descriptor(
            descriptor,
            enable_disagreement_bd_axis=self.enable_disagreement_bd_axis,
        )

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
            self.quality_archive.record_pull_multi(cluster_key, self._case_behavioral_descriptor_at(index), index)
        self._mark_seed_frontier_dirty()

    def _recent_parent_pull_count(self, index: int) -> int:
        self._sync_recent_parent_counts()
        return self.recent_mutation_parent_counts[index]

    def _recent_cluster_pull_count(self, cluster_key: str) -> int:
        if not cluster_key:
            return 0
        self._sync_recent_cluster_counts()
        return self.recent_mutation_cluster_counts[cluster_key]

    def _align_seed_metadata_lengths(self) -> None:
        case_count = len(self.interesting_cases)
        if (
            self.seed_metadata_aligned_count == case_count
            and len(self.lineage.nodes) >= case_count
            and (
                not self.enable_quality_archive
                or not self.interesting_cases
                or not self.quality_archive.is_empty()
            )
        ):
            return
        metadata_changed = False
        while len(self.case_utilities) < case_count:
            self.case_utilities.append(
                _case_seed_utility(
                    self.interesting_cases[len(self.case_utilities)],
                    has_finding=False,
                    family_keys=[],
                )
            )
            metadata_changed = True
        while len(self.case_family_keys) < case_count:
            self.case_family_keys.append([])
            metadata_changed = True
        while len(self.case_profile_keys) < case_count:
            self.case_profile_keys.append("")
            metadata_changed = True
        while len(self.case_target_keys) < case_count:
            self.case_target_keys.append(_normalize_target_keys(_case_target_keys(self.interesting_cases[len(self.case_target_keys)])))
            metadata_changed = True
        while len(self.case_cluster_keys) < case_count:
            index = len(self.case_cluster_keys)
            profile_key = self.case_profile_keys[index] if index < len(self.case_profile_keys) else ""
            target_keys = self.case_target_keys[index] if index < len(self.case_target_keys) else []
            self.case_cluster_keys.append(
                _case_cluster_key(self.interesting_cases[index], profile_key=profile_key, target_keys=target_keys)
            )
            metadata_changed = True
        while len(self.case_behavioral_descriptors) < case_count:
            index = len(self.case_behavioral_descriptors)
            profile_key = self.case_profile_keys[index] if index < len(self.case_profile_keys) else ""
            target_keys = self.case_target_keys[index] if index < len(self.case_target_keys) else []
            self.case_behavioral_descriptors.append(
                self._normalize_behavioral_descriptor(
                    compute_behavioral_descriptor(
                        self.interesting_cases[index],
                        profile_key=profile_key,
                        target_keys=target_keys,
                    ).to_dict()
                )
            )
            metadata_changed = True
        while len(self.case_mutation_pulls) < case_count:
            self.case_mutation_pulls.append(0)
            metadata_changed = True
        while len(self.case_schedule_rewards) < case_count:
            self.case_schedule_rewards.append(0.0)
            metadata_changed = True
        while len(self.case_schedule_feedback_totals) < case_count:
            self.case_schedule_feedback_totals.append(0.0)
            metadata_changed = True
        while len(self.case_schedule_feedback_counts) < case_count:
            self.case_schedule_feedback_counts.append(0)
            metadata_changed = True
        if self.enable_quality_archive and (
            metadata_changed or (self.interesting_cases and self.quality_archive.is_empty())
        ):
            self._sync_quality_archive()
        if metadata_changed or len(self.lineage.nodes) < case_count:
            self._sync_lineage()
        if metadata_changed:
            self._mark_seed_frontier_dirty()
        self.seed_metadata_aligned_count = case_count

    def _sync_lineage(self) -> None:
        for index, case in enumerate(self.interesting_cases):
            if index not in self.lineage.nodes:
                self.lineage.add_seed(index, _lineage_parent_index(case))

    def _sync_quality_archive(self) -> None:
        if not self.enable_quality_archive:
            return
        valid_indexes_by_cluster: dict[str, set[int]] = {}
        for index in range(len(self.interesting_cases)):
            cluster_key = self.case_cluster_keys[index] if index < len(self.case_cluster_keys) else ""
            if not cluster_key:
                continue
            utility = self.case_utilities[index] if index < len(self.case_utilities) else 0.0
            descriptor = (
                dict(self.case_behavioral_descriptors[index])
                if index < len(self.case_behavioral_descriptors)
                else BehavioralDescriptor.empty().to_dict()
            )
            valid_indexes_by_cluster.setdefault(cluster_key, set()).add(index)
            self.quality_archive.record_seed_multi(cluster_key, descriptor, index, utility)
        self.quality_archive.retain_seeds(valid_indexes_by_cluster)

    def _mutation_operator_score_snapshot(
        self,
        *,
        target_keys: list[str] | None = None,
        swarm_particle_id: int | None = None,
    ) -> dict[str, float]:
        self._sync_recent_operator_counts()
        normalized_targets = tuple(_normalize_target_keys(target_keys or []))
        cache_key = (
            self.mutation_score_cache_epoch,
            normalized_targets,
            swarm_particle_id,
            self.enable_mutation_operator_learning,
            self.enable_operator_swarm,
            self.enable_ir_rewrite_mutations,
            self.enable_shrink_mutations,
            self.enable_per_operator_energy,
        )
        cached = self.mutation_operator_score_cache.get(cache_key)
        if cached is not None:
            return dict(cached)
        operator_profiles = self._mutation_operator_profiles()
        if self.enable_operator_swarm and swarm_particle_id is not None:
            self.operator_swarm.ensure_operators(operator_profiles)
        known_operators = list(operator_profiles)
        known_operator_set = set(known_operators)
        for operator_source in (
            self.mutation_operator_rewards,
            self.mutation_operator_pulls,
            self.recent_mutation_operator_counts,
        ):
            for operator in operator_source:
                if not self.enable_ir_rewrite_mutations and operator in IR_REWRITE_MUTATION_OPERATOR_NAMES:
                    continue
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
            energy_bonus = (
                0.03
                * max(
                    0,
                    operator_energy(
                        pulls=pulls,
                        mean_reward=mean_reward,
                        recent_unproductive_streak=recent_operator_counts[operator],
                        catalog_width=len(known_operators),
                    )
                    - 1,
                )
                if self.enable_per_operator_energy
                else 0.0
            )
            swarm_bonus = (
                self.operator_swarm.score_bonus(operator, particle_id=swarm_particle_id)
                if self.enable_operator_swarm and swarm_particle_id is not None
                else 0.0
            )
            discovery_signal = min(
                1.75,
                bounded_ratio(mean_reward + exploration_bonus, 1.5, upper=1.5)
                + 0.25 * bounded_ratio(learning_bonus, 0.25)
                + 0.15 * bounded_ratio(swarm_bonus, 0.20),
            )
            semantic_signal = min(
                1.50,
                bounded_ratio(target_affinity_bonus, 0.75)
                + 0.65 * bounded_ratio(structural_affinity_bonus, 0.55),
            )
            structural_signal = min(
                1.20,
                0.50 * bounded_ratio(structural_affinity_bonus, 0.55)
                + 0.25 * bounded_ratio(energy_bonus, 0.12)
                + 0.20 * bounded_ratio(learning_bonus, 0.25)
                + 0.15 * bounded_ratio(swarm_bonus, 0.20),
            )
            profile = operator_profiles.get(operator)
            structural_tag_count = (
                len(getattr(profile, "structural_risk_tags", ()))
                if profile is not None
                else 0
            )
            coverage_axis_count = (
                len(getattr(profile, "coverage_axes", ()))
                if profile is not None
                else 0
            )
            expandability_bias = (
                float(getattr(profile, "expandability_bias", 0.0) or 0.0)
                if profile is not None
                else 0.0
            )
            validity_floor = (
                float(getattr(profile, "validity_floor", 1.0) or 1.0)
                if profile is not None
                else 1.0
            )
            vector = ObjectiveVector(
                discovery=discovery_signal,
                semantic=semantic_signal,
                novelty=exploration_bonus,
                expandability=min(
                    1.0,
                    0.45 * bounded_ratio(energy_bonus, 0.12)
                    + 0.25 * bounded_ratio(learning_bonus, 0.25)
                    + max(0.0, expandability_bias),
                ),
                structural_risk=min(1.0, structural_signal + (0.08 * structural_tag_count)),
                coverage_gain=min(
                    1.0,
                    0.50 * bounded_ratio(structural_affinity_bonus, 0.55)
                    + 0.25 * bounded_ratio(swarm_bonus, 0.20)
                    + 0.04 * min(4, coverage_axis_count),
                ),
            )
            if pulls > 0:
                vector = ObjectiveVector(
                    discovery=min(1.75, vector.discovery + (0.20 * bounded_ratio(mean_reward, 2.0, upper=1.0))),
                    semantic=vector.semantic,
                    novelty=vector.novelty,
                    expandability=vector.expandability,
                    structural_risk=vector.structural_risk,
                    coverage_gain=vector.coverage_gain,
                )
            modifier = multiplicative_cost_modifier(
                CostVector(
                    redundancy=bounded_ratio(float(recent_operator_counts[operator]), 4.0)
                    + bounded_ratio(float(pulls), 24.0, upper=0.5),
                ),
                weights=OPERATOR_OBJECTIVE_SPEC.cost,
                min_modifier=0.20,
                max_modifier=1.10,
            )
            scores[operator] = (
                (
                    constrained_objective_score(
                        vector,
                        cost=CostVector(),
                        spec=OPERATOR_OBJECTIVE_SPEC,
                        validity=validity_floor,
                        false_positive_risk=0.0,
                    )
                    * modifier
                )
                - recent_penalty
                + learning_bonus
                + energy_bonus
                + swarm_bonus
            )
        if total_pulls >= 8:
            scores["__untried__"] = 0.15
        self._store_mutation_operator_score_cache(cache_key, scores)
        return scores

    def _choose_mutation_swarm_particle_id(
        self,
        index: int,
        *,
        target_keys: list[str],
        mutation_seed: int,
    ) -> int | None:
        if not self.enable_operator_swarm:
            return None
        operator_profiles = self._mutation_operator_profiles()
        self.operator_swarm.ensure_operators(operator_profiles)
        if not self.operator_swarm.particles:
            return None
        action_ids = [str(particle.particle_id) for particle in self.operator_swarm.particles]
        if self.enable_mutation_operator_learning:
            context_features = self._mutation_swarm_context_features(index, target_keys=target_keys)
            cache_key = (
                self.mutation_score_cache_epoch,
                int(index),
                tuple(_normalize_target_keys(target_keys)),
                tuple(action_ids),
                context_features,
            )
            choice = self.mutation_swarm_choice_cache.get(cache_key)
            if choice is None:
                choice = self.adaptive_learning.choose(
                    "mutation_operator_swarm",
                    action_ids=action_ids,
                    context_features=context_features,
                )
                self._store_mutation_swarm_choice_cache(cache_key, choice)
        else:
            choice = action_ids[int(mutation_seed) % len(action_ids)]
        return self.operator_swarm.select_particle(choice).particle_id

    def _store_mutation_swarm_choice_cache(
        self,
        key: tuple[Any, ...],
        choice: str,
    ) -> None:
        if len(self.mutation_swarm_choice_cache) >= 128:
            self.mutation_swarm_choice_cache.clear()
        self.mutation_swarm_choice_cache[key] = str(choice)

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

    def _value_catalog_score_snapshot(
        self,
        *,
        target_keys: list[str] | tuple[str, ...] | None = None,
        disagreement: Any | None = None,
    ) -> dict[str, float]:
        if not self.enable_value_catalog:
            return {}
        cache_key = (
            self.mutation_score_cache_epoch,
            tuple(_normalize_target_keys(target_keys or [])),
            _cache_key_value(disagreement),
            self.enable_mutation_operator_learning,
        )
        cached = self.value_catalog_score_cache.get(cache_key)
        if cached is not None:
            return dict(cached)
        learning_active = self.enable_mutation_operator_learning and self._learning_scope_has_feedback(
            "value_catalog_entry"
        )
        scores: dict[str, float] = {}
        for entry in ADVERSARIAL_CATALOG:
            if not learning_active:
                scores[entry.entry_id] = 0.0
                continue
            row = self.adaptive_learning.score_action(
                "value_catalog_entry",
                entry.entry_id,
                context_features=entry_context_features(
                    entry.entry_id,
                    target_keys=target_keys or (),
                    descriptor=disagreement,
                ),
            )
            score = (
                float(row.get("model_prediction", 0.0) or 0.0)
                + 0.25 * float(row.get("uncertainty", 0.0) or 0.0)
                + float(row.get("version_signal", 0.0) or 0.0)
                - float(row.get("health_penalty", 0.0) or 0.0)
            )
            scores[entry.entry_id] = max(-2.0, min(6.0, score))
        self._store_value_catalog_score_cache(cache_key, scores)
        return scores

    def _invalidate_mutation_score_caches(self) -> None:
        self.mutation_score_cache_epoch += 1
        self.mutation_operator_score_cache.clear()
        self.value_catalog_score_cache.clear()
        self.mutation_swarm_choice_cache.clear()

    def _store_mutation_operator_score_cache(
        self,
        key: tuple[Any, ...],
        scores: dict[str, float],
    ) -> None:
        if len(self.mutation_operator_score_cache) >= 128:
            self.mutation_operator_score_cache.clear()
        self.mutation_operator_score_cache[key] = dict(scores)

    def _store_value_catalog_score_cache(
        self,
        key: tuple[Any, ...],
        scores: dict[str, float],
    ) -> None:
        if len(self.value_catalog_score_cache) >= 128:
            self.value_catalog_score_cache.clear()
        self.value_catalog_score_cache[key] = dict(scores)

    def _choose_champion_graft_donors(
        self,
        base: Case,
        donors: list[ChampionSeed],
        *,
        target_keys: list[str],
    ) -> tuple[list[ChampionSeed], dict[str, Any]]:
        if not donors:
            return [], {}
        if not (
            self.enable_champion_corpus
            and self.enable_champion_graft_donor_bandit
            and self.enable_mutation_operator_learning
        ):
            return donors, {}
        context_features = self._champion_graft_donor_context_features(
            base,
            target_keys=target_keys,
        )
        action_ids = [donor.case_id for donor in donors if str(donor.case_id).strip()]
        if not action_ids:
            return donors, {}
        learning_active = self._learning_scope_has_feedback(CHAMPION_GRAFT_DONOR_SCOPE)
        ranked = (
            self.adaptive_learning.rank(
                CHAMPION_GRAFT_DONOR_SCOPE,
                action_ids,
                context_features=context_features,
                version_id=self.champion_version_id,
            )
            if learning_active
            else []
        )
        if ranked:
            donor_by_id = {donor.case_id: donor for donor in donors}
            ordered = [
                donor_by_id[str(row.get("action_id", "") or "")]
                for row in ranked
                if str(row.get("action_id", "") or "") in donor_by_id
            ]
            seen = {donor.case_id for donor in ordered}
            ordered.extend(donor for donor in donors if donor.case_id not in seen)
            donors = ordered
        selected = donors[0]
        return donors, {
            "scope": CHAMPION_GRAFT_DONOR_SCOPE,
            "strategy": "contextual_bandit" if ranked else "fixed_warmup",
            "selected_donor_case_id": selected.case_id,
            "selected_donor_version_id": selected.version_id,
            "selected_donor_bug_family_keys": list(selected.bug_family_keys),
            "donor_pool": [donor.case_id for donor in donors],
            "context_features": list(context_features),
            "ranked": ranked,
        }

    def _champion_graft_donor_context_features(
        self,
        base: Case,
        *,
        target_keys: list[str],
    ) -> tuple[str, ...]:
        operations = operation_names(base.program.operations)
        type_mix = []
        for table in base.tables[:2]:
            for column in table.columns[:8]:
                type_mix.append(str(getattr(column, "dtype", getattr(column, "type", "")) or ""))
        metadata = base.metadata if isinstance(base.metadata, dict) else {}
        lineage = metadata.get("seed_lineage", {}) if isinstance(metadata.get("seed_lineage", {}), dict) else {}
        features = [
            _bucket("host_op_count", len(operations), [(0, "empty"), (2, "short"), (6, "medium")], "long"),
            _bucket("host_table_count", len(base.tables), [(1, "one"), (2, "two")], "many"),
            _bucket("host_column_count", sum(len(table.columns) for table in base.tables), [(1, "one"), (4, "few"), (12, "some")], "many"),
            _bucket("host_lineage_depth", int(lineage.get("depth", 0) or 0), [(0, "root"), (2, "shallow"), (6, "deep")], "very_deep"),
            _bucket("target_key_count", len(target_keys), [(0, "none"), (2, "few"), (6, "some")], "many"),
        ]
        if type_mix:
            features.append("host_type_mix:" + ",".join(sorted(set(type_mix))[:6]))
        for op_name in operations[:8]:
            features.append(f"host_op:{op_name}")
        for target_key in target_keys[:6]:
            features.append(f"target:{target_key}")
        return tuple(_unique_nonempty(features))

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

    def _mutation_swarm_context_features(
        self,
        index: int | None,
        *,
        target_keys: list[str] | tuple[str, ...] | None,
    ) -> tuple[str, ...]:
        contextual_targets = _contextual_target_descriptors(list(target_keys or []))
        features = list(self._mutation_operator_learning_context_features(contextual_targets))
        if index is not None and 0 <= index < len(self.interesting_cases):
            profile_key = self.case_profile_keys[index] if index < len(self.case_profile_keys) else ""
            cluster_key = self.case_cluster_keys[index] if index < len(self.case_cluster_keys) else ""
            features.append(_bucket("seed_pulls", self.case_mutation_pulls[index], [(0, "cold"), (2, "warm"), (8, "hot")], "saturated"))
            if profile_key:
                features.append(f"profile:{profile_key}")
            if cluster_key:
                features.append(f"cluster:{cluster_key}")
        return tuple(_unique_nonempty(features))

    def _learning_scope_has_feedback(self, scope: str) -> bool:
        bandit = self.adaptive_learning.bandits.get(scope)
        return bool(bandit is not None and bandit.total_pulls > 0)

    def _bd_axis_weights_for_seed(self, index: int) -> dict[str, float]:
        return self._bd_axis_weights(
            self._case_behavioral_descriptor_at(index),
            cluster_key=self._case_cluster_key_at(index),
            target_keys=(
                self.case_target_keys[index]
                if index < len(self.case_target_keys)
                else []
            ),
            seed_index=index,
        )

    def _bd_axis_weights(
        self,
        descriptor: BehavioralDescriptor | dict[str, Any] | None,
        *,
        cluster_key: str = "",
        target_keys: list[str] | tuple[str, ...] | None = None,
        seed_index: int | None = None,
    ) -> dict[str, float]:
        if not (
            self.enable_quality_archive
            and self.enable_bd_axis_bandit
            and self._learning_scope_has_feedback(BD_AXIS_BANDIT_SCOPE)
        ):
            return {}
        axis_tuples = _behavioral_axis_tuples(descriptor)
        if not axis_tuples:
            return {}
        context_features = self._bd_axis_context_features(
            descriptor,
            cluster_key=cluster_key,
            target_keys=target_keys or (),
            seed_index=seed_index,
        )
        weights: dict[str, float] = {}
        for axis_name, _axis_value in axis_tuples:
            row = self.adaptive_learning.score_action(
                BD_AXIS_BANDIT_SCOPE,
                axis_name,
                context_features=context_features,
            )
            signal = (
                float(row.get("reward_signal", 0.0) or 0.0)
                + float(row.get("model_prediction", 0.0) or 0.0)
                + 0.25 * float(row.get("uncertainty", 0.0) or 0.0)
                + float(row.get("version_signal", 0.0) or 0.0)
                - float(row.get("health_penalty", 0.0) or 0.0)
            )
            weights[axis_name] = max(0.10, min(3.0, 1.0 + (BD_AXIS_WEIGHT_BONUS * signal)))
        return weights

    def _bd_axis_context_features(
        self,
        descriptor: BehavioralDescriptor | dict[str, Any] | None,
        *,
        cluster_key: str = "",
        target_keys: list[str] | tuple[str, ...] | None = None,
        seed_index: int | None = None,
    ) -> tuple[str, ...]:
        features: list[str] = []
        if cluster_key:
            features.append(f"cluster:{cluster_key}")
        for target_key in _normalize_target_keys(tuple(target_keys or ())):
            features.append(f"target:{target_key}")
        for axis_name, axis_value in _behavioral_axis_tuples(descriptor):
            features.append(f"{axis_name}:{axis_value}")
        if seed_index is not None and 0 <= seed_index < len(self.interesting_cases):
            features.append(
                _bucket(
                    "seed_pulls",
                    self.case_mutation_pulls[seed_index] if seed_index < len(self.case_mutation_pulls) else 0,
                    [(0, "cold"), (2, "warm"), (8, "hot")],
                    "saturated",
                )
            )
            features.append(
                _bucket(
                    "seed_feedback",
                    self.case_schedule_feedback_counts[seed_index]
                    if seed_index < len(self.case_schedule_feedback_counts)
                    else 0,
                    [(0, "none"), (2, "few"), (8, "some")],
                    "many",
                )
            )
        features.append(
            _bucket(
                "quality_archive_axes",
                sum(len(cells) for cells in self.quality_archive.axis_cells.values()),
                [(0, "empty"), (8, "small"), (32, "medium")],
                "large",
            )
        )
        return tuple(_unique_nonempty(features))

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
        self._invalidate_mutation_score_caches()

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
        self._invalidate_mutation_score_caches()

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
        plan_depth: int,
    ) -> dict[str, Any]:
        self._align_seed_metadata_lengths()
        frontier_snapshot = self._seed_frontier_snapshot(limit=8)
        frontier_rank = self._seed_frontier_rank(index)
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
            "seed_energy": self._case_seed_energy(index),
            "seed_energy_tier": self._choose_seed_energy_tier(index),
            "seed_energy_tier_context": list(self._seed_energy_tier_context_features(index)),
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
            "frontier_rank": frontier_rank,
            "frontier_size": len(self.interesting_cases),
            "frontier_head": [
                {
                    "index": int(row["index"]),
                    "case_id": str(row["case_id"]),
                    "schedule_score": float(row["schedule_score"]),
                    "reward_prior": float(row["reward_prior"]),
                    "seed_energy": int(row["seed_energy"]),
                    "target_novelty_score": float(row["target_novelty_score"]),
                    "cluster_reward": float(row["cluster_reward"]),
                    "cluster_novelty_score": float(row["cluster_novelty_score"]),
                }
                for row in frontier_snapshot
            ],
            "planned_mutation_depth": int(plan_depth),
            "operator_score_count": len(operator_scores),
            "recent_operator_pulls": 0,
            "operator_score_top": [
                {"operator": operator, "mean_reward": score}
                for operator, score in sorted(operator_scores.items(), key=lambda item: item[1], reverse=True)[:8]
            ],
        }

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
        target_keys: list[str] = []
        if self.last_feedback_parent_index is not None and self.last_feedback_parent_index < len(self.case_target_keys):
            target_keys = _normalize_target_keys(self.case_target_keys[self.last_feedback_parent_index])
        if self.last_feedback_parent_index is not None and self.last_feedback_parent_index < len(self.case_schedule_rewards):
            self.case_schedule_feedback_totals[self.last_feedback_parent_index] += float(reward)
            self.case_schedule_feedback_counts[self.last_feedback_parent_index] += 1
            cluster_key = self._case_cluster_key_at(self.last_feedback_parent_index)
            if cluster_key:
                self.cluster_schedule_feedback_totals[cluster_key] += float(reward)
                self.cluster_schedule_feedback_counts[cluster_key] += 1
                if self.enable_quality_archive:
                    self.quality_archive.record_outcome_multi(
                        cluster_key,
                        self._case_behavioral_descriptor_at(self.last_feedback_parent_index),
                        index=self.last_feedback_parent_index,
                        reward=float(reward),
                        preflight_valid=preflight_valid,
                        fallback_used=fallback_used,
                        false_positive=false_positive,
                    )
                    self._record_bd_axis_outcome_reward(
                        float(reward),
                        descriptor=self._case_behavioral_descriptor_at(self.last_feedback_parent_index),
                        cluster_key=cluster_key,
                        target_keys=target_keys,
                        seed_index=self.last_feedback_parent_index,
                        preflight_valid=preflight_valid,
                        fallback_used=fallback_used,
                        false_positive=false_positive,
                    )
            family_keys = (
                self.case_family_keys[self.last_feedback_parent_index]
                if self.last_feedback_parent_index < len(self.case_family_keys)
                else []
            )
            self.lineage.record_pull(
                self.last_feedback_parent_index,
                reward=float(reward),
                family_keys=family_keys,
            )
            self._mark_seed_frontier_dirty()
        if self.last_feedback_operator:
            self.mutation_operator_pulls[self.last_feedback_operator] += 1
            self.mutation_operator_rewards[self.last_feedback_operator] += float(reward)
            for target_key in target_keys[:8]:
                stat_key = _operator_target_stat_key(self.last_feedback_operator, target_key)
                self.mutation_operator_target_pulls[stat_key] += 1
                self.mutation_operator_target_rewards[stat_key] += float(reward)
            if self.enable_operator_swarm:
                self.operator_swarm.update(
                    self.last_feedback_swarm_particle_id,
                    float(reward),
                    operator=self.last_feedback_operator,
                )
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
                if self.enable_operator_swarm and self.last_feedback_swarm_particle_id is not None:
                    self.adaptive_learning.record_outcome(
                        "mutation_operator_swarm",
                        str(self.last_feedback_swarm_particle_id),
                        context_features=self._mutation_swarm_context_features(
                            self.last_feedback_parent_index,
                            target_keys=target_keys,
                        ),
                        reward=float(reward),
                        preflight_valid=preflight_valid,
                        fallback_used=fallback_used,
                        false_positive=false_positive,
                    )
        self._record_value_catalog_outcome_reward(
            float(reward),
            target_keys=target_keys,
            preflight_valid=preflight_valid,
            fallback_used=fallback_used,
            false_positive=false_positive,
        )
        self._record_seed_energy_tier_outcome_reward(
            float(reward),
            preflight_valid=preflight_valid,
            fallback_used=fallback_used,
            false_positive=false_positive,
        )
        self._record_champion_graft_donor_outcome_reward(
            float(reward),
            preflight_valid=preflight_valid,
            fallback_used=fallback_used,
            false_positive=false_positive,
        )
        self._invalidate_mutation_score_caches()

    def _record_bd_axis_outcome_reward(
        self,
        reward: float,
        *,
        descriptor: BehavioralDescriptor | dict[str, Any] | None,
        cluster_key: str,
        target_keys: list[str],
        seed_index: int,
        preflight_valid: bool,
        fallback_used: bool,
        false_positive: bool,
    ) -> None:
        if not (self.enable_quality_archive and self.enable_bd_axis_bandit):
            return
        axis_tuples = _behavioral_axis_tuples(descriptor)
        if not axis_tuples:
            return
        context_features = self._bd_axis_context_features(
            descriptor,
            cluster_key=cluster_key,
            target_keys=target_keys,
            seed_index=seed_index,
        )
        seen: set[str] = set()
        for axis_name, _axis_value in axis_tuples:
            if axis_name in seen:
                continue
            seen.add(axis_name)
            self.adaptive_learning.record_outcome(
                BD_AXIS_BANDIT_SCOPE,
                axis_name,
                context_features=context_features,
                reward=float(reward),
                preflight_valid=preflight_valid,
                fallback_used=fallback_used,
                false_positive=false_positive,
            )

    def _record_value_catalog_outcome_reward(
        self,
        reward: float,
        *,
        target_keys: list[str],
        preflight_valid: bool,
        fallback_used: bool,
        false_positive: bool,
    ) -> None:
        if not (self.enable_value_catalog and self.enable_mutation_operator_learning):
            return
        mutation = self.last_candidate_metadata.get("mutation", {})
        if not isinstance(mutation, dict):
            return
        raw_entries = mutation.get("value_catalog_entries", [])
        if not isinstance(raw_entries, list):
            return
        descriptor = self.last_candidate_metadata.get("disagreement_descriptor")
        seen: set[str] = set()
        for raw_entry in raw_entries:
            if isinstance(raw_entry, dict):
                entry_id = str(raw_entry.get("entry_id", "") or "").strip()
            else:
                entry_id = str(raw_entry or "").strip()
            if not entry_id or entry_id in seen:
                continue
            seen.add(entry_id)
            self.adaptive_learning.record_outcome(
                "value_catalog_entry",
                entry_id,
                context_features=entry_context_features(
                    entry_id,
                    target_keys=target_keys,
                    descriptor=descriptor,
                ),
                reward=float(reward),
                preflight_valid=preflight_valid,
                fallback_used=fallback_used,
                false_positive=false_positive,
            )

    def _record_seed_energy_tier_outcome_reward(
        self,
        reward: float,
        *,
        preflight_valid: bool,
        fallback_used: bool,
        false_positive: bool,
    ) -> None:
        if not (
            self.enable_seed_energy_batch
            and self.enable_seed_energy_tier_bandit
            and self.enable_mutation_operator_learning
        ):
            return
        decision = self.last_feedback_decision if isinstance(self.last_feedback_decision, dict) else {}
        tier = str(decision.get("seed_energy_tier", "") or "").strip()
        if tier not in SEED_ENERGY_TIERS:
            if self.last_feedback_parent_index is None:
                return
            tier = self._choose_seed_energy_tier(self.last_feedback_parent_index)
        raw_context = decision.get("seed_energy_tier_context", ())
        context_features = (
            tuple(str(item) for item in raw_context)
            if isinstance(raw_context, (list, tuple))
            else ()
        )
        if not context_features and self.last_feedback_parent_index is not None:
            context_features = self._seed_energy_tier_context_features(self.last_feedback_parent_index)
        self.adaptive_learning.record_outcome(
            SEED_ENERGY_TIER_SCOPE,
            tier,
            context_features=context_features,
            reward=reward,
            preflight_valid=preflight_valid,
            fallback_used=fallback_used,
            false_positive=false_positive,
        )

    def _record_champion_graft_donor_outcome_reward(
        self,
        reward: float,
        *,
        preflight_valid: bool,
        fallback_used: bool,
        false_positive: bool,
    ) -> None:
        if not (
            self.enable_champion_corpus
            and self.enable_champion_graft_donor_bandit
            and self.enable_mutation_operator_learning
        ):
            return
        metadata = self.last_candidate_metadata if isinstance(self.last_candidate_metadata, dict) else {}
        selection = metadata.get("champion_graft_selection", {})
        if not isinstance(selection, dict):
            return
        donor_id = str(selection.get("selected_donor_case_id", "") or "").strip()
        if not donor_id:
            return
        raw_context = selection.get("context_features", ())
        context_features = (
            tuple(str(item) for item in raw_context)
            if isinstance(raw_context, (list, tuple))
            else ()
        )
        self.adaptive_learning.record_outcome(
            CHAMPION_GRAFT_DONOR_SCOPE,
            donor_id,
            context_features=context_features,
            version_id=str(selection.get("selected_donor_version_id", "") or ""),
            reward=reward,
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
            "champion_family_hits": dict(self.champion_family_hits),
            "champion_promoted_families": sorted(self.champion_promoted_families),
            "champion_version_id": self.champion_version_id,
            "champion_promotion_threshold": self.champion_promotion_threshold,
            "stored_profiles": dict(self.stored_profiles),
            "case_utilities": list(self.case_utilities),
            "case_family_keys": [list(items) for items in self.case_family_keys],
            "case_profile_keys": list(self.case_profile_keys),
            "case_target_keys": [list(items) for items in self.case_target_keys],
            "case_cluster_keys": list(self.case_cluster_keys),
            "case_behavioral_descriptors": [dict(item) for item in self.case_behavioral_descriptors],
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
            "last_feedback_swarm_particle_id": self.last_feedback_swarm_particle_id,
            "mutation_operator_rewards": dict(self.mutation_operator_rewards),
            "mutation_operator_pulls": dict(self.mutation_operator_pulls),
            "mutation_operator_target_rewards": dict(self.mutation_operator_target_rewards),
            "mutation_operator_target_pulls": dict(self.mutation_operator_target_pulls),
            "stored_target_keys": dict(self.stored_target_keys),
            "stored_cluster_keys": dict(self.stored_cluster_keys),
            "cluster_schedule_feedback_totals": dict(self.cluster_schedule_feedback_totals),
            "cluster_schedule_feedback_counts": dict(self.cluster_schedule_feedback_counts),
            "quality_archive": self.quality_archive.to_state_dict(),
            "lineage": self.lineage.to_state_dict(),
            "discovery_rate_estimator": self.discovery_rate_estimator.to_state_dict(),
            "adaptive_learning": self.adaptive_learning.to_state_dict(),
            "operator_swarm": self.operator_swarm.to_state_dict(),
            "enable_mutation_operator_learning": self.enable_mutation_operator_learning,
            "enable_operator_swarm": self.enable_operator_swarm,
            "enable_ir_rewrite_mutations": self.enable_ir_rewrite_mutations,
            "enable_divergence_conditioned_mutations": self.enable_divergence_conditioned_mutations,
            "enable_shrink_mutations": self.enable_shrink_mutations,
            "enable_value_catalog": self.enable_value_catalog,
            "enable_quality_archive": self.enable_quality_archive,
            "enable_hierarchical_archive": self.enable_hierarchical_archive,
            "enable_bd_axis_bandit": self.enable_bd_axis_bandit,
            "enable_bayesian_exploration": self.enable_bayesian_exploration,
            "enable_seed_quota": self.enable_seed_quota,
            "enable_seed_energy_batch": self.enable_seed_energy_batch,
            "enable_seed_energy_tier_bandit": self.enable_seed_energy_tier_bandit,
            "enable_per_operator_energy": self.enable_per_operator_energy,
            "enable_lineage_rarity": self.enable_lineage_rarity,
            "enable_minhash_dedup": self.enable_minhash_dedup,
            "enable_disagreement_bd_axis": self.enable_disagreement_bd_axis,
            "enable_champion_corpus": self.enable_champion_corpus,
            "enable_champion_graft_donor_bandit": self.enable_champion_graft_donor_bandit,
            "seed_eviction_policy": self.seed_eviction_policy.to_state_dict(),
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
        enable_operator_swarm: bool | None = None,
        enable_ir_rewrite_mutations: bool | None = None,
        enable_divergence_conditioned_mutations: bool | None = None,
        enable_shrink_mutations: bool | None = None,
        enable_value_catalog: bool | None = None,
        enable_quality_archive: bool | None = None,
        enable_hierarchical_archive: bool | None = None,
        enable_bd_axis_bandit: bool | None = None,
        enable_bayesian_exploration: bool | None = None,
        enable_seed_quota: bool | None = None,
        enable_seed_energy_batch: bool | None = None,
        enable_seed_energy_tier_bandit: bool | None = None,
        enable_per_operator_energy: bool | None = None,
        enable_lineage_rarity: bool | None = None,
        enable_minhash_dedup: bool | None = None,
        enable_disagreement_bd_axis: bool | None = None,
        enable_champion_corpus: bool | None = None,
        enable_champion_graft_donor_bandit: bool | None = None,
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
            enable_operator_swarm=bool(
                enable_operator_swarm
                if enable_operator_swarm is not None
                else data.get("enable_operator_swarm", True)
            ),
            enable_ir_rewrite_mutations=bool(
                enable_ir_rewrite_mutations
                if enable_ir_rewrite_mutations is not None
                else data.get("enable_ir_rewrite_mutations", True)
            ),
            enable_divergence_conditioned_mutations=bool(
                enable_divergence_conditioned_mutations
                if enable_divergence_conditioned_mutations is not None
                else data.get("enable_divergence_conditioned_mutations", True)
            ),
            enable_shrink_mutations=bool(
                enable_shrink_mutations
                if enable_shrink_mutations is not None
                else data.get("enable_shrink_mutations", True)
            ),
            enable_value_catalog=bool(
                enable_value_catalog
                if enable_value_catalog is not None
                else data.get("enable_value_catalog", True)
            ),
            enable_quality_archive=bool(
                enable_quality_archive
                if enable_quality_archive is not None
                else data.get("enable_quality_archive", True)
            ),
            enable_hierarchical_archive=bool(
                enable_hierarchical_archive
                if enable_hierarchical_archive is not None
                else data.get("enable_hierarchical_archive", True)
            ),
            enable_bd_axis_bandit=bool(
                enable_bd_axis_bandit
                if enable_bd_axis_bandit is not None
                else data.get("enable_bd_axis_bandit", True)
            ),
            enable_bayesian_exploration=bool(
                enable_bayesian_exploration
                if enable_bayesian_exploration is not None
                else data.get("enable_bayesian_exploration", True)
            ),
            enable_seed_quota=bool(
                enable_seed_quota
                if enable_seed_quota is not None
                else data.get("enable_seed_quota", True)
            ),
            enable_seed_energy_batch=bool(
                enable_seed_energy_batch
                if enable_seed_energy_batch is not None
                else data.get("enable_seed_energy_batch", True)
            ),
            enable_seed_energy_tier_bandit=bool(
                enable_seed_energy_tier_bandit
                if enable_seed_energy_tier_bandit is not None
                else data.get("enable_seed_energy_tier_bandit", True)
            ),
            enable_per_operator_energy=bool(
                enable_per_operator_energy
                if enable_per_operator_energy is not None
                else data.get("enable_per_operator_energy", True)
            ),
            enable_lineage_rarity=bool(
                enable_lineage_rarity
                if enable_lineage_rarity is not None
                else data.get("enable_lineage_rarity", True)
            ),
            enable_minhash_dedup=bool(
                enable_minhash_dedup
                if enable_minhash_dedup is not None
                else data.get("enable_minhash_dedup", True)
            ),
            enable_disagreement_bd_axis=bool(
                enable_disagreement_bd_axis
                if enable_disagreement_bd_axis is not None
                else data.get("enable_disagreement_bd_axis", True)
            ),
            enable_champion_corpus=bool(
                enable_champion_corpus
                if enable_champion_corpus is not None
                else data.get("enable_champion_corpus", True)
            ),
            enable_champion_graft_donor_bandit=bool(
                enable_champion_graft_donor_bandit
                if enable_champion_graft_donor_bandit is not None
                else data.get("enable_champion_graft_donor_bandit", True)
            ),
        )
        state.seen_signatures = {str(item) for item in data.get("seen_signatures", []) or []}
        state.interesting_cases = [
            Case.from_dict(item)
            for item in data.get("interesting_cases", []) or []
            if isinstance(item, dict)
        ]
        state.stored_candidate_bug_families = Counter(data.get("stored_candidate_bug_families", {}) or {})
        state.champion_family_hits = _normalize_int_counter(data.get("champion_family_hits", {}) or {})
        state.champion_promoted_families = {
            str(item)
            for item in data.get("champion_promoted_families", []) or []
            if str(item).strip()
        }
        state.champion_version_id = str(data.get("champion_version_id", "") or "")
        state.champion_promotion_threshold = max(1, int(data.get("champion_promotion_threshold", 3) or 3))
        state.stored_profiles = Counter(data.get("stored_profiles", {}) or {})
        state.case_utilities = [float(item) for item in data.get("case_utilities", []) or []]
        state.case_family_keys = [list(items) for items in data.get("case_family_keys", []) or []]
        state.case_profile_keys = [str(item) for item in data.get("case_profile_keys", []) or []]
        state.case_target_keys = [_normalize_target_keys(items) for items in data.get("case_target_keys", []) or []]
        state.case_cluster_keys = [str(item) for item in data.get("case_cluster_keys", []) or []]
        state.case_behavioral_descriptors = [
            state._normalize_behavioral_descriptor(BehavioralDescriptor.from_dict(item).to_dict())
            for item in data.get("case_behavioral_descriptors", []) or []
            if isinstance(item, dict)
        ]
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
        particle_id = data.get("last_feedback_swarm_particle_id", None)
        state.last_feedback_swarm_particle_id = int(particle_id) if particle_id is not None else None
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
        state.quality_archive.enable_hierarchical = bool(state.enable_hierarchical_archive)
        state.lineage = LineageDAG.from_state_dict(data.get("lineage"))
        state.discovery_rate_estimator = DiscoveryRateEstimator.from_state_dict(
            data.get("discovery_rate_estimator")
        )
        state.adaptive_learning = AdaptiveLearningState.from_state_dict(data.get("adaptive_learning"))
        state.operator_swarm = OperatorSwarm.from_state_dict(
            data.get("operator_swarm"),
            operator_names=state._mutation_operator_profiles().keys(),
        )
        state.operator_swarm.ensure_operators(state._mutation_operator_profiles())
        state.seed_eviction_policy = SeedEvictionPolicy.from_state_dict(
            data.get("seed_eviction_policy", data.get("seed_quota"))
        )
        state.seed_eviction_policy.enabled = bool(
            state.enable_seed_quota and state.seed_eviction_policy.enabled
        )
        state._apply_discovery_exploration_weight()
        state.last_feedback_decision = dict(data.get("last_feedback_decision", {}) or {})
        state.persisted_count = int(data.get("persisted_count", 0) or 0)
        state._sync_recent_parent_counts()
        state._sync_recent_cluster_counts()
        state._sync_recent_operator_counts()
        state._align_seed_metadata_lengths()
        if state.enable_quality_archive:
            state._sync_quality_archive()
        if not state.stored_cluster_keys:
            state.stored_cluster_keys.update(str(key) for key in state.case_cluster_keys if str(key))
        state._mark_seed_frontier_dirty()
        return state

    def _mark_seed_frontier_dirty(self) -> None:
        self.seed_frontier.mark_dirty()
        self._sync_seed_frontier_component_state()

    def _seed_frontier_snapshot(
        self,
        *,
        limit: int,
        excluded_indexes: set[int] | None = None,
    ) -> list[dict[str, Any]]:
        return self.seed_frontier.snapshot(
            indexes=range(len(self.interesting_cases)),
            limit=limit,
            excluded_indexes=excluded_indexes,
        )

    def _seed_frontier_rank(self, index: int) -> int:
        return self.seed_frontier.rank(
            indexes=range(len(self.interesting_cases)),
            index=index,
        )

    def _ensure_seed_frontier(self) -> None:
        self._align_seed_metadata_lengths()
        self.seed_frontier.ensure(range(len(self.interesting_cases)))
        self._sync_seed_frontier_component_state()

    def _sync_seed_frontier_component_state(self) -> None:
        self.seed_frontier_heap = list(self.seed_frontier.heap)
        self.seed_frontier_dirty = bool(self.seed_frontier.dirty)

    def _seed_frontier_priority(self, index: int) -> tuple[float, float, float, int, int, int]:
        cluster_key = self._case_cluster_key_at(index)
        return (
            -float(self._case_seed_schedule_score(index)),
            -float(self._case_seed_schedule_reward(index)),
            -float(self._case_target_novelty_score(index)),
            int(self.case_mutation_pulls[index]),
            int(self._recent_parent_pull_count(index)) + int(self._recent_cluster_pull_count(cluster_key)),
            int(index),
        )

    def _seed_frontier_row(self, index: int) -> dict[str, Any]:
        cluster_key = self._case_cluster_key_at(index)
        reward_prior = self._case_seed_schedule_reward(index)
        target_novelty = self._case_target_novelty_score(index)
        archive_cluster_reward = (
            self.quality_archive.cluster_reward_signal(cluster_key)
            if self.enable_quality_archive
            else 0.0
        )
        archive_axis_reward = (
            self.quality_archive.composite_reward_signal(
                self._case_behavioral_descriptor_at(index),
                axis_weights=self._bd_axis_weights_for_seed(index),
            )
            if self.enable_quality_archive
            else 0.0
        )
        cluster_reward = max(
            self._cluster_feedback_reward_signal(cluster_key),
            archive_cluster_reward,
            archive_axis_reward,
        )
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
        mutation_pulls = self.case_mutation_pulls[index]
        recent_parent_pulls = self._recent_parent_pull_count(index)
        recent_cluster_pulls = self._recent_cluster_pull_count(cluster_key)
        lineage_rarity = self.lineage.rarity_score(index) if self.enable_lineage_rarity else 0.0
        return {
            "index": index,
            "case_id": self.interesting_cases[index].case_id if index < len(self.interesting_cases) else "",
            "schedule_score": self._case_seed_schedule_score(index),
            "seed_energy": self._case_seed_energy(index),
            "lineage_rarity": lineage_rarity,
            "reward_prior": reward_prior,
            "target_novelty_score": target_novelty,
            "cluster_reward": cluster_reward,
            "archive_axis_reward": archive_axis_reward,
            "cluster_novelty_score": cluster_novelty,
            "elite_bonus": elite_bonus,
            "archive_health_penalty": archive_health_penalty,
            "mutation_pulls": mutation_pulls,
            "recent_parent_pulls": recent_parent_pulls,
            "recent_cluster_pulls": recent_cluster_pulls,
            "cluster_key": cluster_key,
        }

    def _mutation_plan_depth(
        self,
        index: int,
        *,
        target_keys: list[str],
        operator_scores: dict[str, float],
    ) -> int:
        depth = 1
        schedule_score = self._case_seed_schedule_score(index)
        if schedule_score >= 1.75:
            depth += 1
        if len(target_keys) >= 2:
            depth += 1
        if any(
            str(target).startswith("semantic_signal:") or str(target).startswith("exploration_objective:")
            for target in target_keys
        ):
            depth += 1
        positive_scores = [float(score) for key, score in operator_scores.items() if key != "__untried__" and float(score) > 0.0]
        if len(positive_scores) >= 3:
            depth += 1
        if self.case_mutation_pulls[index] >= 8:
            depth -= 1
        if self._recent_parent_pull_count(index) >= 3:
            depth -= 1
        return max(1, min(4, depth))


def _generated_candidate_metadata(case: Case) -> dict[str, Any]:
    return {
        "seed_lineage": {
            "root_seed": case.seed,
            "parent_index": None,
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


def _behavioral_axis_tuples(
    descriptor: BehavioralDescriptor | dict[str, Any] | None,
) -> tuple[tuple[str, str], ...]:
    if isinstance(descriptor, BehavioralDescriptor):
        return descriptor.axis_tuples()
    if not isinstance(descriptor, dict):
        return ()
    raw_axis_tuples = descriptor.get("axis_tuples", ())
    axis_tuples: list[tuple[str, str]] = []
    if isinstance(raw_axis_tuples, list):
        for item in raw_axis_tuples:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                continue
            axis_name = str(item[0]).strip()
            axis_value = str(item[1]).strip()
            if axis_name and axis_value:
                axis_tuples.append((axis_name, axis_value))
    if axis_tuples:
        return tuple(axis_tuples)
    return BehavioralDescriptor.from_dict(descriptor).axis_tuples()


def _normalize_behavioral_descriptor(
    descriptor: dict[str, Any],
    *,
    enable_disagreement_bd_axis: bool,
) -> dict[str, Any]:
    normalized = BehavioralDescriptor.from_dict(descriptor).to_dict()
    if enable_disagreement_bd_axis:
        return normalized
    normalized["backend_disagreement_axis"] = "pair:none"
    normalized["axis_tuples"] = [
        list(item) for item in BehavioralDescriptor.from_dict(normalized).axis_tuples()
    ]
    return normalized


def _bucket(prefix: str, value: int, limits: list[tuple[int, str]], fallback: str) -> str:
    for limit, name in limits:
        if value <= limit:
            return f"{prefix}:{name}"
    return f"{prefix}:{fallback}"


def _float_bucket(prefix: str, value: float) -> str:
    numeric = float(value or 0.0)
    if numeric <= -0.5:
        return f"{prefix}:negative"
    if numeric <= 0.0:
        return f"{prefix}:zero"
    if numeric <= 1.0:
        return f"{prefix}:low"
    if numeric <= 2.5:
        return f"{prefix}:medium"
    return f"{prefix}:high"


def _bounded_seed_energy(value: int) -> int:
    return min(16, max(1, int(value)))


def _operator_target_stat_key(operator: str, target_key: str) -> str:
    return f"{operator}|{target_key}"


def _normalize_target_key(value: Any) -> str:
    return _normalize_target_key_cached(str(value))


@lru_cache(maxsize=8192)
def _normalize_target_key_cached(value: str) -> str:
    return canonical_target_key(value)


def _normalize_target_keys(values: list[Any] | tuple[Any, ...]) -> list[str]:
    return list(_normalize_target_keys_cached(tuple(str(value) for value in values)))


@lru_cache(maxsize=2048)
def _normalize_target_keys_cached(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(_unique_nonempty([_normalize_target_key(value) for value in values]))


def _cache_key_value(value: Any) -> Any:
    feature_tokens = getattr(value, "feature_tokens", None)
    if callable(feature_tokens):
        return (
            tuple(feature_tokens()),
            str(getattr(value, "primary_root_cause", "") or ""),
            str(getattr(value, "mismatch_class", "") or ""),
        )
    if isinstance(value, Mapping):
        raw_tokens = value.get("feature_tokens")
        tokens = tuple(str(token) for token in raw_tokens) if isinstance(raw_tokens, list | tuple) else ()
        if tokens:
            return (
                tokens,
                str(value.get("primary_root_cause", "") or ""),
                str(value.get("mismatch_class", "") or ""),
            )
        return (
            tokens,
            str(value.get("primary_root_cause", "") or ""),
            str(value.get("mismatch_class", "") or ""),
            _cache_key_simple(value.get("backend_statuses")),
            _cache_key_simple(value.get("column_classes")),
            _cache_key_simple(value.get("backend_groups")),
            _cache_key_simple(value.get("pair_disagrees")),
        )
    return value


def _cache_key_simple(value: Any) -> Any:
    if isinstance(value, Mapping):
        return tuple(
            (str(key), _cache_key_simple(item))
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        )
    if isinstance(value, list):
        return tuple(_cache_key_simple(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_cache_key_simple(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted(_cache_key_simple(item) for item in value))
    if isinstance(value, float):
        if math.isnan(value):
            return ("float", "nan")
        if math.isinf(value):
            return ("float", "inf", 1 if value > 0 else -1)
    return value


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


@lru_cache(maxsize=4096)
def _legacy_target_key_alias(target_key: str) -> str:
    return semantic_signal_legacy_target_key_alias(target_key)


def _contextual_target_descriptors(values: list[Any] | tuple[Any, ...]) -> list[_ContextualTargetDescriptor]:
    return list(_contextual_target_descriptors_cached(tuple(str(value) for value in values)))


@lru_cache(maxsize=512)
def _contextual_target_descriptors_cached(values: tuple[str, ...]) -> tuple[_ContextualTargetDescriptor, ...]:
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
    return tuple(descriptors)


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


def _lineage_parent_index(case: Case) -> int | None:
    metadata = case.metadata if isinstance(case.metadata, dict) else {}
    lineage = metadata.get("seed_lineage", {})
    if not isinstance(lineage, dict):
        return None
    value = lineage.get("parent_index", None)
    if value is None:
        return None
    try:
        parent_index = int(value)
    except (TypeError, ValueError):
        return None
    return parent_index if parent_index >= 0 else None


def _optional_nonnegative_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


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


def _attach_disagreement_descriptor(case: Case, descriptor: Any | None) -> None:
    if descriptor is None or not isinstance(case.metadata, dict):
        return
    to_dict = getattr(descriptor, "to_dict", None)
    payload = to_dict() if callable(to_dict) else descriptor
    if isinstance(payload, dict):
        case.metadata["disagreement_descriptor"] = dict(payload)


def _attach_case_fingerprint(case: Case, fingerprint: Any | None) -> None:
    if fingerprint is None or not isinstance(case.metadata, dict):
        return
    to_dict = getattr(fingerprint, "to_dict", None)
    payload = to_dict() if callable(to_dict) else fingerprint
    if isinstance(payload, dict):
        case.metadata["case_fingerprint"] = dict(payload)


def _case_fingerprint_payload(case: Case) -> dict[str, Any] | None:
    metadata = case.metadata if isinstance(case.metadata, dict) else {}
    payload = metadata.get("case_fingerprint")
    if not isinstance(payload, dict):
        return None
    signature = payload.get("minhash_signature")
    if not isinstance(signature, list) or not signature:
        return None
    return payload


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
