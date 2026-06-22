from __future__ import annotations

import hashlib
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from datadiff.finding_outcomes import (
    FALSE_POSITIVE_VERDICTS,
    NEEDS_CONFIRMATION_VERDICTS,
    RESOLVED_SEMANTIC_DIVERGENCE_VERDICTS,
    SEMANTIC_DIVERGENCE_VERDICTS,
    backend_group_key,
    candidate_issue_family_keys,
    is_rewardable_candidate_issue_finding,
    row_has_rewardable_new_behavior,
)
from datadiff.reward import aggregate_feedback_summaries
from datadiff.adaptive_learning import AdaptiveLearningState, context_features_from_mapping
from datadiff.discovery_rate import DiscoveryRateEstimator
from datadiff.energy import cost_normalized_reward
from datadiff.multi_objective import (
    BATCH_OBJECTIVE_SPEC,
    CostVector,
    ObjectiveVector,
    bounded_ratio,
    constrained_objective_score,
)
from datadiff.source_scheduler import LocalSourceScheduler
from datadiff.util import iter_jsonl, load_json, read_jsonl, run_meta_path

@dataclass(slots=True)
class BatchObservation:
    cases: int
    elapsed_s: float
    throughput_cases_s: float
    findings: int
    candidate_bug_cases: int
    candidate_bug_families: set[str] = field(default_factory=set)
    semantic_divergence_count: int = 0
    rewardable_semantic_divergence_count: int = 0
    resolved_semantic_divergence_count: int = 0
    false_positive_count: int = 0
    needs_confirmation_count: int = 0
    new_behavior_cases: int = 0
    signal_new_behavior_cases: int = 0
    first_candidate_bug_case_index: int | None = None
    first_candidate_bug_elapsed_s: float | None = None
    candidate_bug_discovery_auc: float = 0.0
    feedback_case_count: int = 0
    feedback_mutation_cases: int = 0
    stored_in_feedback_corpus_cases: int = 0
    quality_oracle_count: int = 0
    quality_pass_count: int = 0
    quality_fail_count: int = 0
    quality_score_total: float = 0.0
    source_reward_adjustment_total: float = 0.0
    guidance_reward_adjustment_total: float = 0.0
    seed_schedule_delta_total: float = 0.0
    productive_mutation_cases: int = 0
    invalid_mutation_cases: int = 0
    redundant_mutation_cases: int = 0
    feedback_finding_yield_cases: int = 0
    feedback_new_behavior_yield_cases: int = 0
    feedback_redundant_behavior_cases: int = 0
    guided_productive_cases: int = 0
    guided_target_miss_cases: int = 0
    guided_redundant_cases: int = 0
    scheduler_feedback_share: float = 0.0


@dataclass(slots=True)
class AdaptiveScheduleConfig:
    batch_cases: int = 100
    batch_duration_s: float | None = None
    warmup_batches: int = 1
    exploration_weight: float = 0.75
    freshness_weight: float = 0.10
    stale_penalty: float = 0.12
    group_fairness_weight: float = 0.40
    max_group_pull_gap: int = 3
    prefer_group_diversity_in_round: bool = True
    learning_weight: float = 0.0
    record_learning_feedback: bool = True
    enable_runtime_cost_learning: bool = True
    enable_cost_normalized_reward: bool = True
    enable_active_learning: bool = True
    enable_online_reward_model: bool = True
    enable_continual_learning: bool = True
    enable_bayesian_exploration: bool = True
    annealing_initial_temperature: float = 0.0
    annealing_decay: float = 0.985
    annealing_min_temperature: float = 0.02


@dataclass(slots=True)
class ScheduledBatch:
    arm_id: str
    group_key: str
    batch_index: int
    seed: int
    cases: int | None
    duration_s: float | None
    job: dict[str, Any]


@dataclass(slots=True)
class AdaptiveArmState:
    arm_id: str
    group_key: str
    job: dict[str, Any]
    next_seed: int
    closed_loop_state: dict[str, Any] | None = None
    pulls: int = 0
    total_reward: float = 0.0
    last_reward: float = 0.0
    stale_batches: int = 0
    last_batch_index: int = -1
    candidate_bug_families: set[str] = field(default_factory=set)

    @property
    def mean_reward(self) -> float:
        return self.total_reward / self.pulls if self.pulls else 0.0


class AdaptiveBudgetScheduler:
    def __init__(
        self,
        jobs: list[dict[str, Any]],
        *,
        total_cases_budget: int | None,
        total_duration_budget_s: float | None,
        config: AdaptiveScheduleConfig,
        learning_state: AdaptiveLearningState | None = None,
    ) -> None:
        if not jobs:
            raise ValueError("adaptive scheduler requires at least one job")
        if total_cases_budget is None and total_duration_budget_s is None:
            raise ValueError("adaptive scheduler requires a case or duration budget")
        self.config = config
        self.remaining_cases_budget = total_cases_budget
        self.remaining_duration_budget_s = total_duration_budget_s
        self.reserved_cases_budget = 0
        self.reserved_duration_budget_s = 0.0
        self.total_batches_completed = 0
        self.total_batches_scheduled = 0
        self.global_candidate_bug_families: set[str] = set()
        self.learning_state = learning_state or AdaptiveLearningState()
        self.base_exploration_weight = max(0.0, float(config.exploration_weight or 0.0))
        self.current_exploration_weight = self.base_exploration_weight
        self.bayesian_exploration_observation_count = 0
        self.bayesian_unseen_probability = 0.0
        self.bayesian_exploration_bucket = ""
        self.arms = {
            str(job["arm_id"]): AdaptiveArmState(
                arm_id=str(job["arm_id"]),
                group_key=_adaptive_group_key(job),
                job=dict(job),
                next_seed=int(job["seed"]),
            )
            for job in jobs
        }

    def has_budget(self) -> bool:
        if self.remaining_cases_budget is not None:
            return self._available_cases_budget() > 0
        return bool(self._available_duration_budget_s() and self._available_duration_budget_s() > 0)

    def next_batch(self) -> ScheduledBatch:
        batches = self.next_round(1)
        if not batches:
            raise RuntimeError("adaptive scheduler has no remaining budget")
        return batches[0]

    def next_round(self, max_batches: int) -> list[ScheduledBatch]:
        batches: list[ScheduledBatch] = []
        excluded: set[str] = set()
        excluded_groups: set[str] = set()
        while len(batches) < max(1, int(max_batches)) and self.has_budget():
            batch = self._schedule_next_batch(excluded, excluded_groups)
            if batch is None:
                break
            batches.append(batch)
            excluded.add(batch.arm_id)
            excluded_groups.add(batch.group_key)
        return batches

    def _schedule_next_batch(self, excluded: set[str], excluded_groups: set[str]) -> ScheduledBatch | None:
        arm = self._choose_arm(excluded, excluded_groups)
        if arm is None:
            return None
        cases = None
        duration_s = None
        if self.remaining_cases_budget is not None:
            cases = min(
                max(1, int(self.config.batch_cases)),
                max(1, int(self._available_cases_budget())),
            )
        if self.remaining_duration_budget_s is not None:
            requested = self.config.batch_duration_s or self._available_duration_budget_s()
            duration_s = min(float(requested), float(self._available_duration_budget_s()))
        batch_index = self.total_batches_scheduled
        job = dict(arm.job)
        job["seed"] = arm.next_seed
        job["cases"] = cases
        job["duration_s"] = duration_s
        job["schedule_arm_id"] = arm.arm_id
        job["batch_index"] = batch_index
        job["persist_closed_loop_state"] = True
        if arm.closed_loop_state is not None:
            job["closed_loop_state"] = arm.closed_loop_state
        arm.last_batch_index = batch_index
        if cases is not None:
            self.reserved_cases_budget += int(cases)
        if duration_s is not None:
            self.reserved_duration_budget_s += float(duration_s)
        self.total_batches_scheduled += 1
        return ScheduledBatch(
            arm_id=arm.arm_id,
            group_key=arm.group_key,
            batch_index=batch_index,
            seed=arm.next_seed,
            cases=cases,
            duration_s=duration_s,
            job=job,
        )

    def record_result(
        self,
        batch: ScheduledBatch,
        observation: BatchObservation,
        *,
        next_seed: int,
        closed_loop_state: dict[str, Any] | None = None,
    ) -> float:
        arm = self.arms[batch.arm_id]
        new_global_families = observation.candidate_bug_families - self.global_candidate_bug_families
        new_local_families = observation.candidate_bug_families - arm.candidate_bug_families
        reward = _batch_reward(
            observation,
            new_global_family_count=len(new_global_families),
            new_local_family_count=len(new_local_families),
            enable_runtime_cost=bool(self.config.enable_runtime_cost_learning),
            enable_cost_normalized_reward=bool(self.config.enable_cost_normalized_reward),
        )
        arm.pulls += 1
        arm.total_reward += reward
        arm.last_reward = reward
        arm.next_seed = int(next_seed)
        arm.closed_loop_state = dict(closed_loop_state) if isinstance(closed_loop_state, dict) else None
        self._update_bayesian_exploration_from_closed_loop_state(arm.closed_loop_state)
        arm.candidate_bug_families.update(observation.candidate_bug_families)
        self.global_candidate_bug_families.update(observation.candidate_bug_families)
        self._record_learning_feedback(arm, observation, reward)
        signal = _observation_has_rewardable_signal(
            observation,
            new_global_family_count=len(new_global_families),
            new_local_family_count=len(new_local_families),
        )
        arm.stale_batches = 0 if signal else arm.stale_batches + 1
        if self.remaining_cases_budget is not None:
            self.reserved_cases_budget = max(0, self.reserved_cases_budget - int(batch.cases or 0))
            self.remaining_cases_budget = max(0, self.remaining_cases_budget - int(observation.cases))
        if self.remaining_duration_budget_s is not None:
            self.reserved_duration_budget_s = max(0.0, self.reserved_duration_budget_s - float(batch.duration_s or 0.0))
            self.remaining_duration_budget_s = max(
                0.0,
                float(self.remaining_duration_budget_s) - float(observation.elapsed_s),
            )
        self.total_batches_completed += 1
        return reward

    def snapshot(self) -> list[dict[str, Any]]:
        rows = []
        for arm in sorted(self.arms.values(), key=lambda item: item.arm_id):
            rows.append(
                {
                    "arm_id": arm.arm_id,
                    "target_suite": arm.job.get("target_suite", ""),
                    "preset": arm.job.get("preset", ""),
                    "group_key": arm.group_key,
                    "initial_seed": int(arm.job.get("seed", 0)),
                    "next_seed": arm.next_seed,
                    "pulls": arm.pulls,
                    "mean_reward": arm.mean_reward,
                    "reward_signal": self._arm_reward_signal(arm),
                    "learning_signal": self._learning_signal(arm),
                    "base_exploration_weight": self.base_exploration_weight,
                    "current_exploration_weight": self._exploration_weight(),
                    "bayesian_exploration_enabled": bool(self.config.enable_bayesian_exploration),
                    "bayesian_exploration_observation_count": self.bayesian_exploration_observation_count,
                    "bayesian_unseen_probability": self.bayesian_unseen_probability,
                    "bayesian_exploration_bucket": self.bayesian_exploration_bucket,
                    "annealing_temperature": self._annealing_temperature(),
                    "last_reward": arm.last_reward,
                    "stale_batches": arm.stale_batches,
                    "candidate_bug_families": sorted(arm.candidate_bug_families),
                    "closed_loop_state_present": arm.closed_loop_state is not None,
                }
            )
        return rows

    def _choose_arm(
        self,
        excluded: set[str] | None = None,
        excluded_groups: set[str] | None = None,
    ) -> AdaptiveArmState | None:
        excluded = excluded or set()
        excluded_groups = excluded_groups or set()
        available = [arm for arm in self.arms.values() if arm.arm_id not in excluded]
        if not available:
            return None
        group_diverse = self._available_group_diverse_arms(available, excluded_groups)
        candidate_pool = group_diverse or available
        warmup = [arm for arm in candidate_pool if arm.pulls < max(1, int(self.config.warmup_batches))]
        if warmup:
            return min(
                warmup,
                key=lambda arm: (
                    self._group_pull_count(arm.group_key),
                    arm.pulls,
                    arm.last_batch_index,
                    arm.arm_id,
                ),
            )
        fairness_pool = self._underrepresented_group_arms(candidate_pool)
        if fairness_pool:
            return self._select_scored_arm(fairness_pool)
        return self._select_scored_arm(candidate_pool)

    def _select_scored_arm(self, arms: list[AdaptiveArmState]) -> AdaptiveArmState:
        if len(arms) <= 1:
            return arms[0]
        temperature = self._annealing_temperature()
        if temperature <= 0.0:
            return max(arms, key=self._score_arm)
        rows = sorted(
            ((arm, self._score_arm(arm)) for arm in arms),
            key=lambda item: (item[1], item[0].arm_id),
            reverse=True,
        )
        max_score = rows[0][1]
        weights = [
            math.exp(max(-60.0, min(60.0, (score - max_score) / temperature)))
            for _arm, score in rows
        ]
        total = sum(weights)
        if total <= 0.0:
            return rows[0][0]
        threshold = (
            _stable_unit_interval(
                "adaptive-scheduler-annealing",
                self.total_batches_completed,
                self.total_batches_scheduled,
                ",".join(arm.arm_id for arm, _score in rows),
            )
            * total
        )
        cumulative = 0.0
        for (arm, _score), weight in zip(rows, weights):
            cumulative += weight
            if threshold <= cumulative:
                return arm
        return rows[-1][0]

    def _annealing_temperature(self) -> float:
        initial = max(0.0, float(self.config.annealing_initial_temperature or 0.0))
        if initial <= 0.0:
            return 0.0
        decay = min(1.0, max(0.0, float(self.config.annealing_decay or 0.0)))
        cooled = initial * (decay ** max(0, int(self.total_batches_completed)))
        minimum = max(0.0, float(self.config.annealing_min_temperature or 0.0))
        return max(minimum, cooled)

    def _score_arm(self, arm: AdaptiveArmState) -> float:
        explore = self._exploration_weight() * math.sqrt(
            math.log(self.total_batches_completed + 2.0) / max(1, arm.pulls)
        )
        freshness = self.config.freshness_weight * min(
            1.0,
            max(0, self.total_batches_completed - arm.last_batch_index - 1) / max(1, len(self.arms)),
        )
        stale = self.config.stale_penalty * min(arm.stale_batches, 5)
        fairness_gap = self._group_fairness_gap(arm.group_key)
        fairness_bonus = 0.0
        if fairness_gap > 0:
            fairness_bonus = self.config.group_fairness_weight * min(
                1.0,
                fairness_gap / max(1, int(self.config.max_group_pull_gap)),
            )
        learning = self.config.learning_weight * self._learning_signal(arm)
        return self._arm_reward_signal(arm) + explore + freshness + fairness_bonus + learning - stale

    def _exploration_weight(self) -> float:
        if not self.config.enable_bayesian_exploration:
            return self.base_exploration_weight
        return max(0.0, float(self.current_exploration_weight or 0.0))

    def _update_bayesian_exploration_from_closed_loop_state(
        self,
        closed_loop_state: dict[str, Any] | None,
    ) -> None:
        if not self.config.enable_bayesian_exploration:
            self.current_exploration_weight = self.base_exploration_weight
            return
        raw_estimator = _discovery_rate_state_from_closed_loop_state(closed_loop_state)
        if not raw_estimator:
            return
        estimator = DiscoveryRateEstimator.from_state_dict(raw_estimator)
        if estimator.total_observations <= 0:
            return
        self.current_exploration_weight = estimator.adaptive_exploration_weight(
            self.base_exploration_weight
        )
        self.bayesian_unseen_probability = estimator.unseen_probability()
        self.bayesian_exploration_bucket = estimator.bucket()
        self.bayesian_exploration_observation_count += 1

    def _learning_signal(self, arm: AdaptiveArmState) -> float:
        if self.config.learning_weight <= 0.0:
            return 0.0
        context_features = _arm_context_features(arm)
        version_id = _arm_version_id(arm)
        arm_row = self.learning_state.score_action(
            "batch_arm",
            arm.arm_id,
            context_features=context_features,
            version_id=version_id,
            enable_active_learning=bool(self.config.enable_active_learning),
            enable_reward_model=bool(self.config.enable_online_reward_model),
            enable_continual_learning=bool(self.config.enable_continual_learning),
        )
        action_signals = [
            _learning_row_signal(
                self.learning_state.score_action(
                    scope,
                    action_id,
                    context_features=context_features,
                    version_id=version_id,
                    enable_active_learning=bool(self.config.enable_active_learning),
                    enable_reward_model=bool(self.config.enable_online_reward_model),
                    enable_continual_learning=bool(self.config.enable_continual_learning),
                )
            )
            for scope, action_id in _job_learning_actions(arm.job)
        ]
        arm_signal = _learning_row_signal(arm_row)
        if not action_signals:
            return arm_signal
        transferable_signal = sum(action_signals) / len(action_signals)
        return (0.55 * arm_signal) + (0.45 * transferable_signal)

    def _record_learning_feedback(
        self,
        arm: AdaptiveArmState,
        observation: BatchObservation,
        reward: float,
    ) -> None:
        if not self.config.record_learning_feedback:
            return
        preflight_valid = observation.invalid_mutation_cases <= 0
        false_positive = observation.false_positive_count > 0
        runtime_cost = (
            _runtime_cost_penalty(observation)
            if self.config.enable_runtime_cost_learning
            else 0.0
        )
        version_id = _arm_version_id(arm)
        context_features = _arm_context_features(arm)
        self.learning_state.record_outcome(
            "batch_arm",
            arm.arm_id,
            context_features=context_features,
            reward=reward,
            runtime_cost=runtime_cost,
            version_id=version_id,
            preflight_valid=preflight_valid,
            false_positive=false_positive,
            enable_reward_model=bool(self.config.enable_online_reward_model),
        )
        for scope, action_id in _job_learning_actions(arm.job):
            self.learning_state.record_outcome(
                scope,
                action_id,
                context_features=context_features,
                reward=reward,
                runtime_cost=runtime_cost,
                version_id=version_id,
                preflight_valid=preflight_valid,
                false_positive=false_positive,
                enable_reward_model=bool(self.config.enable_online_reward_model),
            )

    @staticmethod
    def _arm_reward_signal(arm: AdaptiveArmState) -> float:
        return _bounded_confident_mean_reward(
            arm.total_reward,
            arm.pulls,
            max_abs=6.0,
        )

    def _available_group_diverse_arms(
        self,
        available: list[AdaptiveArmState],
        excluded_groups: set[str],
    ) -> list[AdaptiveArmState]:
        if not self.config.prefer_group_diversity_in_round or not excluded_groups:
            return available
        diverse = [arm for arm in available if arm.group_key not in excluded_groups]
        return diverse or available

    def _group_pull_count(self, group_key: str) -> int:
        return sum(arm.pulls for arm in self.arms.values() if arm.group_key == group_key)

    def _group_pull_counts(self) -> Counter[str]:
        counts: Counter[str] = Counter()
        for arm in self.arms.values():
            counts[arm.group_key] += arm.pulls
        return counts

    def _group_fairness_gap(self, group_key: str) -> int:
        counts = self._group_pull_counts()
        if not counts:
            return 0
        return max(counts.values()) - int(counts.get(group_key, 0))

    def _underrepresented_group_arms(self, available: list[AdaptiveArmState]) -> list[AdaptiveArmState]:
        if not available or self.config.max_group_pull_gap <= 0:
            return []
        counts = self._group_pull_counts()
        if not counts:
            return []
        min_pulls = min(int(counts.get(arm.group_key, 0)) for arm in available)
        max_pulls = max(int(counts.get(arm.group_key, 0)) for arm in available)
        if max_pulls - min_pulls < int(self.config.max_group_pull_gap):
            return []
        underrepresented = {
            arm.group_key
            for arm in available
            if int(counts.get(arm.group_key, 0)) == min_pulls
        }
        return [arm for arm in available if arm.group_key in underrepresented]

    def _available_cases_budget(self) -> int:
        if self.remaining_cases_budget is None:
            return 0
        return max(0, int(self.remaining_cases_budget) - int(self.reserved_cases_budget))

    def _available_duration_budget_s(self) -> float:
        if self.remaining_duration_budget_s is None:
            return 0.0
        return max(0.0, float(self.remaining_duration_budget_s) - float(self.reserved_duration_budget_s))


def summarize_batch_run(run_file: Path) -> BatchObservation:
    meta_path = run_meta_path(run_file)
    meta = load_json(meta_path) if meta_path.exists() else {}
    config = meta.get("config", {}) if isinstance(meta.get("config", {}), dict) else {}
    known_families = list(config.get("known_saturated_bug_families", []) or [])
    findings = 0
    case_count = 0
    candidate_bug_cases = 0
    candidate_bug_families: Counter[str] = Counter()
    semantic_divergence_count = 0
    rewardable_semantic_divergence_count = 0
    resolved_semantic_divergence_count = 0
    false_positive_count = 0
    needs_confirmation_count = 0
    new_behavior_cases = 0
    signal_new_behavior_cases = 0
    rows: list[dict[str, Any]] = []
    for row in iter_jsonl(run_file):
        rows.append(row)
        case_count += 1
        row_findings = row.get("findings", [])
        findings += len(row_findings)
        new_behavior_cases += int(bool(row.get("is_new_behavior")))
        signal_new_behavior_cases += int(row_has_rewardable_new_behavior(row, known_families))
        if any(is_rewardable_candidate_issue_finding(finding, known_families) for finding in row_findings):
            candidate_bug_cases += 1
        candidate_bug_families.update(
            candidate_issue_family_keys(row_findings, known_saturated_bug_families=known_families)
        )
        for finding in row_findings:
            verdict = str(finding.get("triage_verdict", "unclassified"))
            if verdict in SEMANTIC_DIVERGENCE_VERDICTS:
                semantic_divergence_count += 1
            if verdict == "semantic_divergence_needs_confirmation":
                rewardable_semantic_divergence_count += 1
            if verdict in RESOLVED_SEMANTIC_DIVERGENCE_VERDICTS:
                resolved_semantic_divergence_count += 1
            if verdict in FALSE_POSITIVE_VERDICTS:
                false_positive_count += 1
            if verdict in NEEDS_CONFIRMATION_VERDICTS:
                needs_confirmation_count += 1
    feedback = aggregate_feedback_summaries(rows, known_saturated_bug_families=known_families)
    first_candidate_idx, first_candidate_elapsed_s = _first_candidate_bug_position(rows, known_families)
    return BatchObservation(
        cases=case_count,
        elapsed_s=float(meta.get("elapsed_s", 0.0) or 0.0),
        throughput_cases_s=float(meta.get("throughput_cases_s", 0.0) or 0.0),
        findings=findings,
        candidate_bug_cases=candidate_bug_cases,
        candidate_bug_families=set(candidate_bug_families),
        semantic_divergence_count=semantic_divergence_count,
        rewardable_semantic_divergence_count=rewardable_semantic_divergence_count,
        resolved_semantic_divergence_count=resolved_semantic_divergence_count,
        false_positive_count=false_positive_count,
        needs_confirmation_count=needs_confirmation_count,
        new_behavior_cases=new_behavior_cases,
        signal_new_behavior_cases=signal_new_behavior_cases,
        first_candidate_bug_case_index=first_candidate_idx,
        first_candidate_bug_elapsed_s=first_candidate_elapsed_s,
        candidate_bug_discovery_auc=_candidate_bug_discovery_auc(rows, known_families),
        feedback_case_count=int(feedback["feedback_case_count"]),
        feedback_mutation_cases=int(feedback["feedback_mutation_cases"]),
        stored_in_feedback_corpus_cases=int(feedback["stored_in_feedback_corpus_cases"]),
        quality_oracle_count=int(feedback["quality_oracle_count"]),
        quality_pass_count=int(feedback["quality_pass_count"]),
        quality_fail_count=int(feedback["quality_fail_count"]),
        quality_score_total=float(feedback["quality_score_total"]),
        source_reward_adjustment_total=float(feedback["source_reward_adjustment_total"]),
        guidance_reward_adjustment_total=float(feedback["guidance_reward_adjustment_total"]),
        seed_schedule_delta_total=float(feedback["seed_schedule_delta_total"]),
        productive_mutation_cases=int(feedback["productive_mutation_cases"]),
        invalid_mutation_cases=int(feedback["invalid_mutation_cases"]),
        redundant_mutation_cases=int(feedback["redundant_mutation_cases"]),
        feedback_finding_yield_cases=int(feedback["feedback_finding_yield_cases"]),
        feedback_new_behavior_yield_cases=int(feedback["feedback_new_behavior_yield_cases"]),
        feedback_redundant_behavior_cases=int(feedback["feedback_redundant_behavior_cases"]),
        guided_productive_cases=int(feedback["guided_productive_cases"]),
        guided_target_miss_cases=int(feedback["guided_target_miss_cases"]),
        guided_redundant_cases=int(feedback["guided_redundant_cases"]),
        scheduler_feedback_share=_scheduler_feedback_share_from_meta(meta),
    )


def _batch_reward(
    observation: BatchObservation,
    *,
    new_global_family_count: int,
    new_local_family_count: int,
    enable_runtime_cost: bool = True,
    enable_cost_normalized_reward: bool = True,
) -> float:
    cases = max(1, int(observation.cases))
    findings = max(1, int(observation.findings))
    candidate_rate = observation.candidate_bug_cases / cases
    new_behavior_rate = observation.signal_new_behavior_cases / cases
    rewardable_semantic_rate = observation.rewardable_semantic_divergence_count / findings
    false_positive_rate = observation.false_positive_count / findings
    manual_confirmation_count = max(
        0,
        int(observation.needs_confirmation_count) - int(observation.rewardable_semantic_divergence_count),
    )
    manual_confirmation_rate = manual_confirmation_count / findings
    throughput_signal = math.log1p(max(0.0, float(observation.throughput_cases_s))) / 6.0
    has_rewardable_signal = _observation_has_rewardable_signal(
        observation,
        new_global_family_count=new_global_family_count,
        new_local_family_count=new_local_family_count,
    )
    source_adjustment_rate = _gated_positive_rate(
        float(observation.source_reward_adjustment_total),
        cases,
        allow_positive=has_rewardable_signal,
    )
    guidance_adjustment_rate = _gated_positive_rate(
        float(observation.guidance_reward_adjustment_total),
        cases,
        allow_positive=has_rewardable_signal,
    )
    seed_schedule_rate = _gated_positive_rate(
        float(observation.seed_schedule_delta_total),
        cases,
        allow_positive=has_rewardable_signal,
    )
    feedback_mutation_productive_rate = (
        float(observation.productive_mutation_cases) / max(1.0, float(observation.feedback_mutation_cases))
        if observation.feedback_mutation_cases
        else 0.0
    )
    guidance_productive_rate = float(observation.guided_productive_cases) / cases
    guidance_target_miss_rate = float(observation.guided_target_miss_cases) / cases
    redundant_feedback_rate = float(observation.feedback_redundant_behavior_cases) / cases
    early_case_bonus = 0.0
    if observation.first_candidate_bug_case_index is not None:
        early_case_bonus = max(
            0.0,
            1.0 - (float(observation.first_candidate_bug_case_index) / cases),
        )
    early_time_bonus = 0.0
    if observation.first_candidate_bug_elapsed_s is not None:
        elapsed = max(float(observation.elapsed_s), float(observation.first_candidate_bug_elapsed_s), 1e-9)
        early_time_bonus = max(0.0, 1.0 - (float(observation.first_candidate_bug_elapsed_s) / elapsed))
    discovery_signal = min(
        2.0,
        (1.25 * candidate_rate)
        + (0.55 * min(1.0, float(new_local_family_count)))
        + (1.00 * min(1.0, float(new_global_family_count)))
        + (0.40 * float(observation.candidate_bug_discovery_auc)),
    )
    semantic_signal = min(1.5, rewardable_semantic_rate + (0.35 * early_case_bonus))
    behavior_signal = min(
        1.25,
        new_behavior_rate
        + 0.35 * _gated_positive_value(feedback_mutation_productive_rate, allow_positive=has_rewardable_signal)
        + 0.20 * _gated_positive_value(guidance_productive_rate, allow_positive=has_rewardable_signal),
    )
    structural_signal = min(
        1.0,
        0.30 * throughput_signal
        + 0.20 * early_time_bonus
        + 0.18 * source_adjustment_rate
        + 0.14 * guidance_adjustment_rate
        + 0.18 * seed_schedule_rate,
    )
    vector = ObjectiveVector(
        discovery=discovery_signal,
        semantic=semantic_signal,
        novelty=min(
            1.0,
            0.40 * min(1.0, float(new_local_family_count))
            + 0.65 * min(1.0, float(new_global_family_count)),
        ),
        expandability=min(
            1.0,
            0.35 * _gated_positive_value(feedback_mutation_productive_rate, allow_positive=has_rewardable_signal)
            + 0.20 * seed_schedule_rate,
        ),
        structural_risk=structural_signal,
        coverage_gain=min(1.0, 0.50 * throughput_signal + 0.25 * early_time_bonus + 0.25 * early_case_bonus),
    )
    cost = CostVector(
        invalidity=_runtime_cost_penalty(observation) if enable_runtime_cost else 0.0,
        false_positive=false_positive_rate,
        redundancy=redundant_feedback_rate + bounded_ratio(
            float(observation.resolved_semantic_divergence_count),
            float(findings),
        ),
        target_miss=guidance_target_miss_rate + manual_confirmation_rate,
    )
    reward = constrained_objective_score(
        vector,
        cost=cost,
        spec=BATCH_OBJECTIVE_SPEC,
        validity=1.0 - cost.invalidity,
        false_positive_risk=cost.false_positive,
    )
    if observation.findings == 0 and observation.signal_new_behavior_cases == 0:
        reward -= 0.25
    if enable_cost_normalized_reward:
        return cost_normalized_reward(reward, observation.elapsed_s)
    return reward


def _observation_has_rewardable_signal(
    observation: BatchObservation,
    *,
    new_global_family_count: int,
    new_local_family_count: int,
) -> bool:
    return bool(
        observation.candidate_bug_cases
        or observation.signal_new_behavior_cases
        or observation.rewardable_semantic_divergence_count
        or new_global_family_count
        or new_local_family_count
    )


def _gated_positive_rate(total: float, cases: float, *, allow_positive: bool) -> float:
    return _gated_positive_value(float(total) / max(1.0, float(cases)), allow_positive=allow_positive)


def _gated_positive_value(value: float, *, allow_positive: bool) -> float:
    value = float(value)
    if allow_positive:
        return value
    return min(0.0, value)


def _runtime_cost_penalty(observation: BatchObservation) -> float:
    cases = max(1.0, float(observation.cases))
    feedback_mutation_cases = max(1.0, float(observation.feedback_mutation_cases))
    scheduler_feedback_share = max(0.0, float(observation.scheduler_feedback_share or 0.0))
    invalid_mutation_rate = max(0.0, float(observation.invalid_mutation_cases)) / feedback_mutation_cases
    redundant_mutation_rate = max(0.0, float(observation.redundant_mutation_cases)) / feedback_mutation_cases
    guided_redundant_rate = max(0.0, float(observation.guided_redundant_cases)) / cases
    return min(
        3.0,
        1.50 * scheduler_feedback_share
        + 0.75 * invalid_mutation_rate
        + 0.35 * redundant_mutation_rate
        + 0.25 * guided_redundant_rate,
    )


def _scheduler_feedback_share_from_meta(meta: dict[str, Any]) -> float:
    stage_profile = meta.get("stage_profile", {})
    if not isinstance(stage_profile, dict):
        return 0.0
    share = stage_profile.get("share_of_total", {})
    if isinstance(share, dict) and "scheduler_feedback_ms" in share:
        return max(0.0, _safe_float(share.get("scheduler_feedback_ms")))
    totals = stage_profile.get("totals_ms", {})
    if not isinstance(totals, dict):
        return 0.0
    total_ms = _safe_float(totals.get("total_case_wall_ms"))
    if total_ms <= 0.0:
        return 0.0
    return max(0.0, _safe_float(totals.get("scheduler_feedback_ms")) / total_ms)


def _learning_row_signal(row: dict[str, Any]) -> float:
    return (
        0.35 * float(row.get("reward_signal", 0.0))
        + float(row.get("model_prediction", 0.0))
        + 0.50 * float(row.get("uncertainty", 0.0))
        + float(row.get("version_signal", 0.0))
        + 0.75 * float(row.get("continual_priority_signal", 0.0))
        + 0.40 * float(row.get("exploration_bonus", 0.0))
        - float(row.get("health_penalty", 0.0))
    )


def _safe_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _candidate_issue_family_keys(findings: list[dict[str, Any]]) -> Counter[str]:
    return candidate_issue_family_keys(findings)


def _discovery_rate_state_from_closed_loop_state(
    closed_loop_state: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(closed_loop_state, dict):
        return None
    feedback = closed_loop_state.get("feedback")
    if isinstance(feedback, dict):
        estimator = feedback.get("discovery_rate_estimator")
        if isinstance(estimator, dict):
            return estimator
    estimator = closed_loop_state.get("discovery_rate_estimator")
    if isinstance(estimator, dict):
        return estimator
    return None


def _adaptive_group_key(job: dict[str, Any]) -> str:
    target_suite = str(job.get("target_suite", "") or "unknown")
    preset = str(job.get("preset", "") or "baseline")
    return f"{target_suite}:{preset}"


def _arm_context_features(arm: AdaptiveArmState) -> tuple[str, ...]:
    return context_features_from_mapping(arm.job)


def _arm_version_id(arm: AdaptiveArmState) -> str:
    target_version = str(arm.job.get("target_version", "") or "").strip()
    fixed_version = str(arm.job.get("fixed_version", "") or "").strip()
    if target_version and fixed_version:
        return f"{target_version}->{fixed_version}"
    return target_version or fixed_version


def _job_learning_actions(job: dict[str, Any]) -> list[tuple[str, str]]:
    actions: list[tuple[str, str]] = []

    def add(scope: str, action_id: Any) -> None:
        text = str(action_id).strip()
        if text:
            actions.append((scope, text))

    add("generator_profile", job.get("scheduler_generator_profile") or job.get("generator_profile") or job.get("profile"))
    add("guidance_strategy", job.get("scheduler_guidance_strategy") or job.get("guidance_strategy") or job.get("guidance"))
    add("oracle_mode", job.get("scheduler_oracle_mode") or job.get("oracle_mode"))
    for operator in _iter_string_values(
        job.get("mutation_operator")
        or job.get("mutation_operators")
        or job.get("mutation_operator_profile")
        or job.get("preferred_mutation_operators")
    ):
        add("mutation_operator", operator)
    metamorphic_enabled = bool(
        job.get(
            "scheduler_enable_metamorphic_oracle",
            job.get("enable_metamorphic_oracle", job.get("metamorphic_oracle", False)),
        )
    )
    if metamorphic_enabled:
        add("metamorphic_relation", "enabled")
    for relation in _iter_string_values(
        job.get("metamorphic_relation")
        or job.get("metamorphic_relations")
        or job.get("metamorphic_relation_type")
        or job.get("metamorphic_relation_types")
        or job.get("mr_type")
        or job.get("mr_types")
    ):
        add("metamorphic_relation", relation)
    metamorphic_limit = job.get(
        "scheduler_effective_metamorphic_variant_limit",
        job.get(
            "effective_metamorphic_variant_limit",
            job.get("metamorphic_variant_limit"),
        ),
    )
    if metamorphic_enabled and metamorphic_limit not in (None, ""):
        add("metamorphic_relation", f"limit_{metamorphic_limit}")
    for objective in _iter_string_values(
        job.get("scheduler_semantic_objectives")
        or job.get("semantic_objectives")
        or job.get("objectives")
    ):
        add("semantic_objective", objective)
    for target in _iter_string_values(job.get("scheduler_guidance_targets") or job.get("guidance_targets")):
        add("semantic_objective", target)
    for family in _iter_string_values(
        job.get("scheduler_semantic_focus_families")
        or job.get("semantic_focus_families")
    ):
        add("semantic_objective", f"semantic_family:{family}")
    for signal in _iter_string_values(
        job.get("scheduler_semantic_focus_signals")
        or job.get("semantic_focus_signals")
    ):
        add("semantic_objective", f"semantic_signal:{signal}")
    version_id = str(job.get("target_version", "") or "").strip()
    fixed_version = str(job.get("fixed_version", "") or "").strip()
    if version_id or fixed_version:
        add("version_pair", f"{version_id}->{fixed_version}" if fixed_version else version_id)
    return list(dict.fromkeys(actions))


def _iter_string_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw = value.split(",")
    elif isinstance(value, (list, tuple, set)):
        raw = value
    else:
        raw = [value]
    return _unique_nonempty([str(item).strip() for item in raw])


def _bounded_confident_mean_reward(total_reward: float, pulls: float, *, max_abs: float) -> float:
    if pulls <= 0:
        return 0.0
    mean_reward = float(total_reward) / float(pulls)
    bounded_mean_reward = max(-max_abs, min(max_abs, mean_reward))
    confidence = min(1.0, math.log1p(float(pulls)) / math.log(4.0))
    return bounded_mean_reward * confidence


def _stable_unit_interval(*parts: Any) -> float:
    payload = "\x1f".join(str(part) for part in parts)
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") / float(1 << 64)


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


def _first_candidate_issue_position(
    rows: list[dict[str, Any]],
    known_saturated_bug_families: list[str],
) -> tuple[int | None, float | None]:
    for idx, row in enumerate(rows):
        if any(
            is_rewardable_candidate_issue_finding(finding, known_saturated_bug_families)
            for finding in row.get("findings", [])
        ):
            elapsed = row.get("elapsed_s")
            return int(row.get("case_index", idx)), float(elapsed) if elapsed not in (None, "") else None
    return None, None


def _candidate_issue_discovery_auc(
    rows: list[dict[str, Any]],
    known_saturated_bug_families: list[str] | None = None,
) -> float:
    if not rows:
        return 0.0
    cumulative = 0
    area = 0
    total = 0
    per_row = []
    for row in rows:
        hit = int(
            any(
                is_rewardable_candidate_issue_finding(finding, known_saturated_bug_families)
                for finding in row.get("findings", [])
            )
        )
        per_row.append(hit)
        total += hit
    if total == 0:
        return 0.0
    for hit in per_row:
        cumulative += hit
        area += cumulative
    return area / (len(rows) * total)


_candidate_bug_family_keys = _candidate_issue_family_keys
_first_candidate_bug_position = _first_candidate_issue_position
_candidate_bug_discovery_auc = _candidate_issue_discovery_auc
