from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from datadiff.reward import (
    FALSE_POSITIVE_VERDICTS,
    NEEDS_CONFIRMATION_VERDICTS,
    SEMANTIC_DIVERGENCE_VERDICTS,
    candidate_bug_family_keys,
    is_candidate_bug_finding,
    suspicious_key,
)
from datadiff.util import load_json, read_jsonl, run_meta_path

CandidateSource = Literal["generated", "feedback_mutation"]

@dataclass(slots=True)
class BatchObservation:
    cases: int
    elapsed_s: float
    throughput_cases_s: float
    findings: int
    candidate_bug_cases: int
    candidate_bug_families: set[str] = field(default_factory=set)
    semantic_divergence_count: int = 0
    false_positive_count: int = 0
    needs_confirmation_count: int = 0
    new_behavior_cases: int = 0
    first_candidate_bug_case_index: int | None = None
    first_candidate_bug_elapsed_s: float | None = None
    candidate_bug_discovery_auc: float = 0.0


@dataclass(slots=True)
class AdaptiveScheduleConfig:
    batch_cases: int = 100
    batch_duration_s: float | None = None
    warmup_batches: int = 1
    exploration_weight: float = 0.75
    freshness_weight: float = 0.10
    stale_penalty: float = 0.12


@dataclass(slots=True)
class ScheduledBatch:
    arm_id: str
    batch_index: int
    seed: int
    cases: int | None
    duration_s: float | None
    job: dict[str, Any]


@dataclass(slots=True)
class AdaptiveArmState:
    arm_id: str
    job: dict[str, Any]
    next_seed: int
    pulls: int = 0
    total_reward: float = 0.0
    last_reward: float = 0.0
    stale_batches: int = 0
    last_batch_index: int = -1
    candidate_bug_families: set[str] = field(default_factory=set)

    @property
    def mean_reward(self) -> float:
        return self.total_reward / self.pulls if self.pulls else 0.0


@dataclass(slots=True)
class SourceArmState:
    name: CandidateSource
    pulls: int = 0
    total_reward: float = 0.0
    candidate_bug_families: Counter[str] = field(default_factory=Counter)
    candidate_bug_signatures: Counter[str] = field(default_factory=Counter)

    @property
    def mean_reward(self) -> float:
        return self.total_reward / self.pulls if self.pulls else 0.0


class LocalSourceScheduler:
    def __init__(
        self,
        *,
        exploration_weight: float = 0.5,
        min_feedback_share: float = 0.12,
        enable_family_saturation: bool = True,
        family_saturation_threshold: int = 8,
        saturated_family_reward: float = 0.02,
        known_saturated_bug_families: list[str] | None = None,
    ) -> None:
        self.exploration_weight = max(0.0, float(exploration_weight))
        self.min_feedback_share = min(0.5, max(0.0, float(min_feedback_share)))
        self.enable_family_saturation = bool(enable_family_saturation)
        self.family_saturation_threshold = max(0, int(family_saturation_threshold))
        self.saturated_family_reward = max(0.0, float(saturated_family_reward))
        self.known_saturated_bug_families = _unique_nonempty(known_saturated_bug_families or [])
        self.total_pulls = 0
        self.candidate_bug_families: Counter[str] = Counter()
        self.candidate_bug_signatures: Counter[str] = Counter()
        self.arms: dict[CandidateSource, SourceArmState] = {
            "generated": SourceArmState(name="generated"),
            "feedback_mutation": SourceArmState(name="feedback_mutation"),
        }

    def choose_source(self, *, feedback_available: bool) -> CandidateSource:
        if not feedback_available:
            return "generated"
        for source in ("generated", "feedback_mutation"):
            if self.arms[source].pulls == 0:
                return source
        feedback_pulls = self.arms["feedback_mutation"].pulls
        if self.total_pulls >= 4 and feedback_pulls / max(1, self.total_pulls) < self.min_feedback_share:
            return "feedback_mutation"
        return max(self.arms.values(), key=self._score_arm).name

    def record_result(
        self,
        source: CandidateSource,
        *,
        has_finding: bool,
        is_new_behavior: bool,
        preflight_valid: bool,
        fallback_used: bool,
        candidate_bug: bool = False,
        semantic_divergence: bool = False,
        false_positive: bool = False,
        candidate_bug_families: list[str] | None = None,
        candidate_bug_signatures: list[str] | None = None,
    ) -> float:
        family_keys = _unique_nonempty(candidate_bug_families or [])
        signature_keys = _unique_nonempty(candidate_bug_signatures or [])
        candidate_bug_reward = 0.0
        if candidate_bug:
            if family_keys:
                candidate_bug_reward = sum(
                    _candidate_family_reward(
                        self._previous_family_hits(family),
                        enable_family_saturation=self.enable_family_saturation,
                        family_saturation_threshold=self.family_saturation_threshold,
                        saturated_family_reward=self.saturated_family_reward,
                    )
                    for family in family_keys
                )
                if signature_keys and all(self.candidate_bug_signatures[signature] > 0 for signature in signature_keys):
                    candidate_bug_reward *= 0.5
            else:
                candidate_bug_reward = 4.0
        reward = (
            candidate_bug_reward
            + (0.20 if semantic_divergence else 0.0)
            + (0.05 if has_finding and not candidate_bug and not semantic_divergence and not false_positive else 0.0)
            + (0.5 if is_new_behavior else 0.0)
            - (2.5 if false_positive else 0.0)
        )
        if not preflight_valid or fallback_used:
            reward -= 0.75
        if reward == 0.0:
            reward -= 0.1
        arm = self.arms[source]
        arm.pulls += 1
        arm.total_reward += reward
        arm.candidate_bug_families.update(family_keys)
        arm.candidate_bug_signatures.update(signature_keys)
        self.candidate_bug_families.update(family_keys)
        self.candidate_bug_signatures.update(signature_keys)
        self.total_pulls += 1
        return reward

    def _previous_family_hits(self, family: str) -> int:
        previous_hits = self.candidate_bug_families[family]
        if self.enable_family_saturation and _family_key_matches_known_family(
            family,
            self.known_saturated_bug_families,
        ):
            previous_hits = max(previous_hits, self.family_saturation_threshold)
        return previous_hits

    def snapshot(self) -> list[dict[str, Any]]:
        return [
            {
                "source": arm.name,
                "pulls": arm.pulls,
                "mean_reward": arm.mean_reward,
                "total_reward": arm.total_reward,
                "candidate_bug_family_count": len(arm.candidate_bug_families),
                "candidate_bug_signature_count": len(arm.candidate_bug_signatures),
                "min_feedback_share": self.min_feedback_share,
            }
            for arm in sorted(self.arms.values(), key=lambda item: item.name)
        ]

    def _score_arm(self, arm: SourceArmState) -> float:
        explore = self.exploration_weight * math.sqrt(
            math.log(self.total_pulls + 1.0) / max(1, arm.pulls)
        )
        return arm.mean_reward + explore


class AdaptiveBudgetScheduler:
    def __init__(
        self,
        jobs: list[dict[str, Any]],
        *,
        total_cases_budget: int | None,
        total_duration_budget_s: float | None,
        config: AdaptiveScheduleConfig,
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
        self.arms = {
            str(job["arm_id"]): AdaptiveArmState(
                arm_id=str(job["arm_id"]),
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
        while len(batches) < max(1, int(max_batches)) and self.has_budget():
            batch = self._schedule_next_batch(excluded)
            if batch is None:
                break
            batches.append(batch)
            excluded.add(batch.arm_id)
        return batches

    def _schedule_next_batch(self, excluded: set[str]) -> ScheduledBatch | None:
        arm = self._choose_arm(excluded)
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
        arm.last_batch_index = batch_index
        if cases is not None:
            self.reserved_cases_budget += int(cases)
        if duration_s is not None:
            self.reserved_duration_budget_s += float(duration_s)
        self.total_batches_scheduled += 1
        return ScheduledBatch(
            arm_id=arm.arm_id,
            batch_index=batch_index,
            seed=arm.next_seed,
            cases=cases,
            duration_s=duration_s,
            job=job,
        )

    def record_result(self, batch: ScheduledBatch, observation: BatchObservation, *, next_seed: int) -> float:
        arm = self.arms[batch.arm_id]
        new_global_families = observation.candidate_bug_families - self.global_candidate_bug_families
        new_local_families = observation.candidate_bug_families - arm.candidate_bug_families
        reward = _batch_reward(
            observation,
            new_global_family_count=len(new_global_families),
            new_local_family_count=len(new_local_families),
        )
        arm.pulls += 1
        arm.total_reward += reward
        arm.last_reward = reward
        arm.next_seed = int(next_seed)
        arm.candidate_bug_families.update(observation.candidate_bug_families)
        self.global_candidate_bug_families.update(observation.candidate_bug_families)
        signal = bool(
            observation.candidate_bug_cases
            or observation.new_behavior_cases
            or new_global_families
            or new_local_families
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
                    "initial_seed": int(arm.job.get("seed", 0)),
                    "next_seed": arm.next_seed,
                    "pulls": arm.pulls,
                    "mean_reward": arm.mean_reward,
                    "last_reward": arm.last_reward,
                    "stale_batches": arm.stale_batches,
                    "candidate_bug_families": sorted(arm.candidate_bug_families),
                }
            )
        return rows

    def _choose_arm(self, excluded: set[str] | None = None) -> AdaptiveArmState | None:
        excluded = excluded or set()
        available = [arm for arm in self.arms.values() if arm.arm_id not in excluded]
        if not available:
            return None
        warmup = [arm for arm in available if arm.pulls < max(1, int(self.config.warmup_batches))]
        if warmup:
            return min(warmup, key=lambda arm: (arm.pulls, arm.last_batch_index, arm.arm_id))
        return max(available, key=self._score_arm)

    def _score_arm(self, arm: AdaptiveArmState) -> float:
        explore = self.config.exploration_weight * math.sqrt(
            math.log(self.total_batches_completed + 2.0) / max(1, arm.pulls)
        )
        freshness = self.config.freshness_weight * min(
            1.0,
            max(0, self.total_batches_completed - arm.last_batch_index - 1) / max(1, len(self.arms)),
        )
        stale = self.config.stale_penalty * min(arm.stale_batches, 5)
        return arm.mean_reward + explore + freshness - stale

    def _available_cases_budget(self) -> int:
        if self.remaining_cases_budget is None:
            return 0
        return max(0, int(self.remaining_cases_budget) - int(self.reserved_cases_budget))

    def _available_duration_budget_s(self) -> float:
        if self.remaining_duration_budget_s is None:
            return 0.0
        return max(0.0, float(self.remaining_duration_budget_s) - float(self.reserved_duration_budget_s))


def summarize_batch_run(run_file: Path) -> BatchObservation:
    rows = read_jsonl(run_file)
    meta_path = run_meta_path(run_file)
    meta = load_json(meta_path) if meta_path.exists() else {}
    findings = 0
    candidate_bug_cases = 0
    candidate_bug_families: Counter[str] = Counter()
    semantic_divergence_count = 0
    false_positive_count = 0
    needs_confirmation_count = 0
    new_behavior_cases = 0
    for row in rows:
        row_findings = row.get("findings", [])
        findings += len(row_findings)
        new_behavior_cases += int(bool(row.get("is_new_behavior")))
        if any(is_candidate_bug_finding(finding) for finding in row_findings):
            candidate_bug_cases += 1
        candidate_bug_families.update(candidate_bug_family_keys(row_findings))
        for finding in row_findings:
            verdict = str(finding.get("triage_verdict", "unclassified"))
            if verdict in SEMANTIC_DIVERGENCE_VERDICTS:
                semantic_divergence_count += 1
            if verdict in FALSE_POSITIVE_VERDICTS:
                false_positive_count += 1
            if verdict in NEEDS_CONFIRMATION_VERDICTS:
                needs_confirmation_count += 1
    first_candidate_idx, first_candidate_elapsed_s = _first_candidate_bug_position(rows)
    return BatchObservation(
        cases=len(rows),
        elapsed_s=float(meta.get("elapsed_s", 0.0) or 0.0),
        throughput_cases_s=float(meta.get("throughput_cases_s", 0.0) or 0.0),
        findings=findings,
        candidate_bug_cases=candidate_bug_cases,
        candidate_bug_families=set(candidate_bug_families),
        semantic_divergence_count=semantic_divergence_count,
        false_positive_count=false_positive_count,
        needs_confirmation_count=needs_confirmation_count,
        new_behavior_cases=new_behavior_cases,
        first_candidate_bug_case_index=first_candidate_idx,
        first_candidate_bug_elapsed_s=first_candidate_elapsed_s,
        candidate_bug_discovery_auc=_candidate_bug_discovery_auc(rows),
    )


def _batch_reward(
    observation: BatchObservation,
    *,
    new_global_family_count: int,
    new_local_family_count: int,
) -> float:
    cases = max(1, int(observation.cases))
    findings = max(1, int(observation.findings))
    candidate_rate = observation.candidate_bug_cases / cases
    new_behavior_rate = observation.new_behavior_cases / cases
    semantic_rate = observation.semantic_divergence_count / findings
    false_positive_rate = observation.false_positive_count / findings
    needs_confirmation_rate = observation.needs_confirmation_count / findings
    throughput_signal = math.log1p(max(0.0, float(observation.throughput_cases_s))) / 6.0
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
    reward = (
        10.0 * candidate_rate
        + 2.0 * new_behavior_rate
        + 0.10 * semantic_rate
        + 1.5 * new_local_family_count
        + 3.0 * new_global_family_count
        + 2.0 * early_case_bonus
        + 1.0 * early_time_bonus
        + 3.0 * observation.candidate_bug_discovery_auc
        + 0.2 * throughput_signal
        - 1.0 * needs_confirmation_rate
        - 6.0 * false_positive_rate
    )
    if observation.findings == 0 and observation.new_behavior_cases == 0:
        reward -= 0.25
    return reward


def _candidate_bug_family_keys(findings: list[dict[str, Any]]) -> Counter[str]:
    return candidate_bug_family_keys(findings)


def _suspicious_key(finding: dict[str, Any]) -> str:
    return suspicious_key(finding)


def _candidate_family_reward(
    previous_hits: int,
    *,
    enable_family_saturation: bool = True,
    family_saturation_threshold: int = 8,
    saturated_family_reward: float = 0.02,
) -> float:
    if enable_family_saturation and family_saturation_threshold > 0 and previous_hits >= family_saturation_threshold:
        return max(0.0, saturated_family_reward)
    if previous_hits <= 0:
        return 4.0
    if previous_hits <= 2:
        return 1.25
    return max(0.15, 0.75 / math.sqrt(previous_hits))


def _family_key_matches_known_family(candidate_family: str, known_saturated_bug_families: list[str]) -> bool:
    candidate_root, candidate_backends = _split_family_key(candidate_family)
    for known_family in known_saturated_bug_families:
        known_root, known_backends = _split_family_key(known_family)
        if known_root != candidate_root:
            continue
        if not known_backends or not candidate_backends or known_backends & candidate_backends:
            return True
    return False


def _split_family_key(family_key: str) -> tuple[str, set[str]]:
    root, _, backend_part = str(family_key).partition("@")
    backends = {backend.strip() for backend in backend_part.split(",") if backend.strip()}
    return root.strip(), backends


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


def _first_candidate_bug_position(rows: list[dict[str, Any]]) -> tuple[int | None, float | None]:
    for idx, row in enumerate(rows):
        if any(is_candidate_bug_finding(finding) for finding in row.get("findings", [])):
            elapsed = row.get("elapsed_s")
            return int(row.get("case_index", idx)), float(elapsed) if elapsed not in (None, "") else None
    return None, None


def _candidate_bug_discovery_auc(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    cumulative = 0
    area = 0
    total = 0
    per_row = []
    for row in rows:
        hit = int(any(is_candidate_bug_finding(finding) for finding in row.get("findings", [])))
        per_row.append(hit)
        total += hit
    if total == 0:
        return 0.0
    for hit in per_row:
        cumulative += hit
        area += cumulative
    return area / (len(rows) * total)
