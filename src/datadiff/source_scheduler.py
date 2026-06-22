from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Literal

from datadiff.family_novelty import candidate_family_novelty_reward, family_key_matches_known_family
from datadiff.multi_objective import (
    CostVector,
    ObjectiveVector,
    SOURCE_OBJECTIVE_SPEC,
    bounded_ratio,
    constrained_objective_score,
)

CandidateSource = Literal["generated", "feedback_mutation"]


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
        if self.arms["feedback_mutation"].pulls == 0 and self.arms["generated"].pulls > 0:
            generated_signal = self._arm_reward_signal(self.arms["generated"])
            if generated_signal <= 0.0:
                return "feedback_mutation"
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
        reward_adjustment: float = 0.0,
    ) -> float:
        family_keys = _unique_nonempty(candidate_bug_families or [])
        signature_keys = _unique_nonempty(candidate_bug_signatures or [])
        rewardable_semantic_divergence = semantic_divergence and not false_positive
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
        discovery_signal = min(2.0, float(candidate_bug_reward) / 2.6)
        semantic_signal = 0.14 if rewardable_semantic_divergence else 0.0
        behavior_signal = 0.10 if is_new_behavior else 0.0
        structural_signal = 0.08 if has_finding and not candidate_bug and not false_positive else 0.0
        vector = ObjectiveVector(
            discovery=discovery_signal,
            semantic=semantic_signal,
            novelty=bounded_ratio(float(candidate_bug_reward), 4.0) if candidate_bug else 0.0,
            expandability=0.10 if is_new_behavior else 0.0,
            structural_risk=structural_signal,
            coverage_gain=0.08 if family_keys else 0.0,
        )
        cost = CostVector(
            invalidity=1.0 if (not preflight_valid or fallback_used) else 0.0,
            false_positive=1.0 if false_positive else 0.0,
            redundancy=0.35 if family_keys and signature_keys and all(
                self.candidate_bug_signatures[signature] > 0 for signature in signature_keys
            ) else 0.0,
        )
        reward = constrained_objective_score(
            vector,
            cost=cost,
            spec=SOURCE_OBJECTIVE_SPEC,
            validity=1.0 - cost.invalidity,
            false_positive_risk=cost.false_positive,
        ) + float(reward_adjustment)
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
        if self.enable_family_saturation and family_key_matches_known_family(
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
                "reward_signal": self._arm_reward_signal(arm),
                "total_reward": arm.total_reward,
                "candidate_bug_family_count": len(arm.candidate_bug_families),
                "candidate_bug_signature_count": len(arm.candidate_bug_signatures),
                "min_feedback_share": self.min_feedback_share,
            }
            for arm in sorted(self.arms.values(), key=lambda item: item.name)
        ]

    def to_state_dict(self) -> dict[str, Any]:
        return {
            "exploration_weight": self.exploration_weight,
            "min_feedback_share": self.min_feedback_share,
            "enable_family_saturation": self.enable_family_saturation,
            "family_saturation_threshold": self.family_saturation_threshold,
            "saturated_family_reward": self.saturated_family_reward,
            "known_saturated_bug_families": list(self.known_saturated_bug_families),
            "total_pulls": self.total_pulls,
            "candidate_bug_families": dict(self.candidate_bug_families),
            "candidate_bug_signatures": dict(self.candidate_bug_signatures),
            "arms": {
                name: {
                    "pulls": arm.pulls,
                    "total_reward": arm.total_reward,
                    "candidate_bug_families": dict(arm.candidate_bug_families),
                    "candidate_bug_signatures": dict(arm.candidate_bug_signatures),
                }
                for name, arm in self.arms.items()
            },
        }

    @classmethod
    def from_state_dict(
        cls,
        data: dict[str, Any],
        *,
        exploration_weight: float,
        enable_family_saturation: bool,
        family_saturation_threshold: int,
        saturated_family_reward: float,
        known_saturated_bug_families: list[str] | None,
    ) -> "LocalSourceScheduler":
        scheduler = cls(
            exploration_weight=exploration_weight,
            min_feedback_share=float(data.get("min_feedback_share", 0.12) or 0.12),
            enable_family_saturation=enable_family_saturation,
            family_saturation_threshold=family_saturation_threshold,
            saturated_family_reward=saturated_family_reward,
            known_saturated_bug_families=known_saturated_bug_families,
        )
        scheduler.total_pulls = int(data.get("total_pulls", 0) or 0)
        scheduler.candidate_bug_families = Counter(data.get("candidate_bug_families", {}) or {})
        scheduler.candidate_bug_signatures = Counter(data.get("candidate_bug_signatures", {}) or {})
        for name, arm in scheduler.arms.items():
            arm_data = (data.get("arms", {}) or {}).get(name, {})
            arm.pulls = int(arm_data.get("pulls", 0) or 0)
            arm.total_reward = float(arm_data.get("total_reward", 0.0) or 0.0)
            arm.candidate_bug_families = Counter(arm_data.get("candidate_bug_families", {}) or {})
            arm.candidate_bug_signatures = Counter(arm_data.get("candidate_bug_signatures", {}) or {})
        return scheduler

    def _score_arm(self, arm: SourceArmState) -> float:
        explore = self.exploration_weight * math.sqrt(
            math.log(self.total_pulls + 1.0) / max(1, arm.pulls)
        )
        return self._arm_reward_signal(arm) + explore

    @staticmethod
    def _arm_reward_signal(arm: SourceArmState) -> float:
        return _bounded_confident_mean_reward(
            arm.total_reward,
            arm.pulls,
            max_abs=2.5,
        )


def _candidate_family_reward(
    previous_hits: int,
    *,
    enable_family_saturation: bool = True,
    family_saturation_threshold: int = 8,
    saturated_family_reward: float = 0.02,
) -> float:
    return candidate_family_novelty_reward(
        previous_hits,
        enable_family_saturation=enable_family_saturation,
        family_saturation_threshold=family_saturation_threshold,
        saturated_family_reward=saturated_family_reward,
    )


def _bounded_confident_mean_reward(total_reward: float, pulls: float, *, max_abs: float) -> float:
    if pulls <= 0:
        return 0.0
    mean_reward = float(total_reward) / float(pulls)
    bounded_mean_reward = max(-max_abs, min(max_abs, mean_reward))
    confidence = min(1.0, math.log1p(float(pulls)) / math.log(4.0))
    return bounded_mean_reward * confidence


def _unique_nonempty(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out
