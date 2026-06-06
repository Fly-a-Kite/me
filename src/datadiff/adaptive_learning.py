from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from functools import lru_cache
import math
from typing import Any

from datadiff.feature_interning import (
    export_feature_counter,
    export_feature_mapping,
    import_feature_counter,
    import_feature_float_mapping,
    intern_feature,
    intern_feature_ids,
    resolve_feature_id,
)

ADAPTIVE_LEARNING_SCHEMA_VERSION = "adaptive-learning-v1"
CONTINUAL_LEARNING_SCHEMA_VERSION = "cross-version-continual-learning-v1"
_SELF_CALIBRATED_COMPONENTS = (
    "reward_signal",
    "model_prediction",
    "uncertainty",
    "exploration",
    "version_signal",
    "continual_priority_signal",
    "exploration_bonus",
    "health_penalty",
)
_SELF_CALIBRATED_COMPONENT_INDEX = {
    component: index for index, component in enumerate(_SELF_CALIBRATED_COMPONENTS)
}
_SELF_CALIBRATED_COMPONENT_COUNT = len(_SELF_CALIBRATED_COMPONENTS)
_REWARD_SIGNAL_COMPONENT_INDEX = _SELF_CALIBRATED_COMPONENT_INDEX["reward_signal"]
_MODEL_PREDICTION_COMPONENT_INDEX = _SELF_CALIBRATED_COMPONENT_INDEX["model_prediction"]
_UNCERTAINTY_COMPONENT_INDEX = _SELF_CALIBRATED_COMPONENT_INDEX["uncertainty"]
_EXPLORATION_COMPONENT_INDEX = _SELF_CALIBRATED_COMPONENT_INDEX["exploration"]
_VERSION_SIGNAL_COMPONENT_INDEX = _SELF_CALIBRATED_COMPONENT_INDEX["version_signal"]
_CONTINUAL_PRIORITY_SIGNAL_COMPONENT_INDEX = _SELF_CALIBRATED_COMPONENT_INDEX["continual_priority_signal"]
_EXPLORATION_BONUS_COMPONENT_INDEX = _SELF_CALIBRATED_COMPONENT_INDEX["exploration_bonus"]
_HEALTH_PENALTY_COMPONENT_INDEX = _SELF_CALIBRATED_COMPONENT_INDEX["health_penalty"]


@dataclass(slots=True)
class OnlineRewardModel:
    learning_rate: float = 0.08
    l2: float = 0.001
    max_abs_reward: float = 6.0
    _weights_dense: list[float] = field(default_factory=list, repr=False)
    _feature_counts_dense: list[int] = field(default_factory=list, repr=False)
    total_updates: int = 0

    @property
    def weights(self) -> dict[str, float]:
        return export_feature_mapping(self._weights_by_id())

    @property
    def feature_counts(self) -> Counter[str]:
        return Counter(export_feature_counter(self._feature_counts_by_id()))

    def predict(self, features: Iterable[Any]) -> float:
        normalized = _normalize_features(features)
        return self.predict_normalized(normalized)

    def predict_normalized(self, normalized: tuple[str, ...]) -> float:
        if not normalized:
            return 0.0
        feature_ids = _feature_ids_from_normalized(normalized)
        scale = math.sqrt(len(normalized))
        return self._predict_from_feature_ids(feature_ids, scale)

    def uncertainty(self, features: Iterable[Any]) -> float:
        normalized = _normalize_features(features)
        return self.uncertainty_normalized(normalized)

    def uncertainty_normalized(self, normalized: tuple[str, ...]) -> float:
        if not normalized:
            return 1.0
        feature_ids = _feature_ids_from_normalized(normalized)
        coldness = self._coldness_from_feature_ids(feature_ids)
        return min(1.0, coldness / math.sqrt(len(normalized)))

    def update(self, features: Iterable[Any], reward: float) -> float:
        normalized = _normalize_features(features)
        return self.update_normalized(normalized, reward)

    def update_normalized(self, normalized: tuple[str, ...], reward: float) -> float:
        if not normalized:
            return 0.0
        target = _bounded(float(reward), self.max_abs_reward)
        feature_ids = _feature_ids_from_normalized(normalized)
        scale = math.sqrt(len(normalized))
        prediction = self._predict_from_feature_ids(feature_ids, scale)
        error = target - prediction
        self._ensure_dense_capacity(max(feature_ids))
        update_base = self.learning_rate * (error / scale)
        l2_decay = self.learning_rate * self.l2
        weights = self._weights_dense
        counts = self._feature_counts_dense
        for feature_id in feature_ids:
            current = weights[feature_id]
            weights[feature_id] = current + update_base - (l2_decay * current)
            counts[feature_id] += 1
        self.total_updates += 1
        return error

    def to_state_dict(self) -> dict[str, Any]:
        return {
            "learning_rate": self.learning_rate,
            "l2": self.l2,
            "max_abs_reward": self.max_abs_reward,
            "weights": export_feature_mapping(self._weights_by_id()),
            "feature_counts": export_feature_counter(self._feature_counts_by_id()),
            "total_updates": self.total_updates,
        }

    @classmethod
    def from_state_dict(cls, data: Mapping[str, Any] | None) -> "OnlineRewardModel":
        if not isinstance(data, Mapping):
            return cls()
        model = cls(
            learning_rate=_float_field(data, "learning_rate", 0.08),
            l2=_float_field(data, "l2", 0.001),
            max_abs_reward=_float_field(data, "max_abs_reward", 6.0),
        )
        model._load_dense_state(
            weights_by_id=import_feature_float_mapping(data.get("weights", {}) or {}),
            feature_counts_by_id=import_feature_counter(data.get("feature_counts", {}) or {}),
        )
        model.total_updates = int(data.get("total_updates", 0) or 0)
        return model

    def _predict_from_feature_ids(self, feature_ids: tuple[int, ...], scale: float) -> float:
        total = 0.0
        weights = self._weights_dense
        limit = len(weights)
        for feature_id in feature_ids:
            if feature_id < limit:
                total += weights[feature_id]
        return total / scale

    def _coldness_from_feature_ids(self, feature_ids: tuple[int, ...]) -> float:
        total = 0.0
        counts = self._feature_counts_dense
        limit = len(counts)
        for feature_id in feature_ids:
            count = counts[feature_id] if feature_id < limit else 0
            total += 1.0 / math.sqrt(1.0 + count)
        return total

    def _ensure_dense_capacity(self, feature_id: int) -> None:
        required = int(feature_id) + 1 - len(self._weights_dense)
        if required <= 0:
            return
        self._weights_dense.extend([0.0] * required)
        self._feature_counts_dense.extend([0] * required)

    def _load_dense_state(
        self,
        *,
        weights_by_id: Mapping[int, Any],
        feature_counts_by_id: Mapping[int, Any],
    ) -> None:
        self._weights_dense = []
        self._feature_counts_dense = []
        max_feature_id = -1
        for feature_id in weights_by_id:
            max_feature_id = max(max_feature_id, int(feature_id))
        for feature_id in feature_counts_by_id:
            max_feature_id = max(max_feature_id, int(feature_id))
        if max_feature_id >= 0:
            self._ensure_dense_capacity(max_feature_id)
        for feature_id, weight in weights_by_id.items():
            if int(feature_id) < 0:
                continue
            self._weights_dense[int(feature_id)] = float(weight or 0.0)
        for feature_id, count in feature_counts_by_id.items():
            if int(feature_id) < 0:
                continue
            self._feature_counts_dense[int(feature_id)] = int(count or 0)

    def _weights_by_id(self) -> dict[int, float]:
        return {
            feature_id: weight
            for feature_id, weight in enumerate(self._weights_dense)
            if float(weight) != 0.0
        }

    def _feature_counts_by_id(self) -> Counter[int]:
        return Counter(
            {
                feature_id: int(count)
                for feature_id, count in enumerate(self._feature_counts_dense)
                if int(count) > 0
            }
        )


@dataclass(slots=True)
class ActionStats:
    action_id: str
    pulls: int = 0
    total_reward: float = 0.0
    reward_sq_total: float = 0.0
    last_reward: float = 0.0
    runtime_cost_total: float = 0.0
    invalid_count: int = 0
    fallback_count: int = 0
    false_positive_count: int = 0
    last_step: int = -1

    @property
    def mean_reward(self) -> float:
        return self.total_reward / self.pulls if self.pulls else 0.0

    def reward_signal(self, *, max_abs: float = 6.0) -> float:
        return _bounded_confident_mean_reward(self.total_reward, self.pulls, max_abs=max_abs)

    def health_penalty(self) -> float:
        if self.pulls <= 0:
            return 0.0
        invalid_rate = self.invalid_count / self.pulls
        fallback_rate = self.fallback_count / self.pulls
        false_positive_rate = self.false_positive_count / self.pulls
        runtime_cost_rate = max(0.0, self.runtime_cost_total / self.pulls)
        return min(
            1.5,
            0.25 * invalid_rate
            + 0.15 * fallback_rate
            + 0.60 * false_positive_rate
            + 0.35 * runtime_cost_rate,
        )

    def to_state_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "pulls": self.pulls,
            "total_reward": self.total_reward,
            "reward_sq_total": self.reward_sq_total,
            "last_reward": self.last_reward,
            "runtime_cost_total": self.runtime_cost_total,
            "invalid_count": self.invalid_count,
            "fallback_count": self.fallback_count,
            "false_positive_count": self.false_positive_count,
            "last_step": self.last_step,
        }

    @classmethod
    def from_state_dict(cls, data: Mapping[str, Any]) -> "ActionStats":
        return cls(
            action_id=str(data.get("action_id", "") or ""),
            pulls=int(data.get("pulls", 0) or 0),
            total_reward=float(data.get("total_reward", 0.0) or 0.0),
            reward_sq_total=float(data.get("reward_sq_total", 0.0) or 0.0),
            last_reward=float(data.get("last_reward", 0.0) or 0.0),
            runtime_cost_total=float(data.get("runtime_cost_total", 0.0) or 0.0),
            invalid_count=int(data.get("invalid_count", 0) or 0),
            fallback_count=int(data.get("fallback_count", 0) or 0),
            false_positive_count=int(data.get("false_positive_count", 0) or 0),
            last_step=int(data.get("last_step", -1) if data.get("last_step", -1) is not None else -1),
        )


@dataclass(slots=True)
class DenseBanditChoice:
    action_id: str
    score: float
    pulls: int

    def sort_key(self) -> tuple[float, int, str]:
        return (self.score, -self.pulls, self.action_id)


@dataclass(slots=True)
class DenseBanditScore:
    action_id: str
    score: float
    pulls: int
    mean_reward: float
    signal_components_dense: tuple[float, ...] = field(repr=False)
    weighted_components_dense: tuple[float, ...] = field(repr=False)

    def sort_key(self) -> tuple[float, int, str]:
        return (self.score, -self.pulls, self.action_id)


@dataclass(slots=True)
class VersionFeedbackMemory:
    _reward_totals_by_id: Counter[int] = field(default_factory=Counter, repr=False)
    _reward_counts_by_id: Counter[int] = field(default_factory=Counter, repr=False)
    _runtime_cost_totals_by_id: Counter[int] = field(default_factory=Counter, repr=False)
    _invalid_counts_by_id: Counter[int] = field(default_factory=Counter, repr=False)
    _false_positive_counts_by_id: Counter[int] = field(default_factory=Counter, repr=False)
    _action_reward_totals_by_id: Counter[int] = field(default_factory=Counter, init=False, repr=False)
    _action_reward_counts_by_id: Counter[int] = field(default_factory=Counter, init=False, repr=False)

    def __post_init__(self) -> None:
        self._rebuild_action_indexes()

    @property
    def reward_totals(self) -> Counter[str]:
        return Counter(export_feature_mapping(self._reward_totals_by_id))

    @property
    def reward_counts(self) -> Counter[str]:
        return Counter(export_feature_counter(self._reward_counts_by_id))

    @property
    def runtime_cost_totals(self) -> Counter[str]:
        return Counter(export_feature_mapping(self._runtime_cost_totals_by_id))

    @property
    def invalid_counts(self) -> Counter[str]:
        return Counter(export_feature_counter(self._invalid_counts_by_id))

    @property
    def false_positive_counts(self) -> Counter[str]:
        return Counter(export_feature_counter(self._false_positive_counts_by_id))

    def record(
        self,
        *,
        scope: str,
        action_id: str,
        version_id: str,
        reward: float,
        runtime_cost: float = 0.0,
        preflight_valid: bool = True,
        false_positive: bool = False,
    ) -> None:
        if not version_id:
            return
        key_id = _version_key_id(scope, action_id, version_id)
        action_key_id = _scope_action_key_id(scope, action_id)
        self._reward_totals_by_id[key_id] += float(reward)
        self._reward_counts_by_id[key_id] += 1
        self._runtime_cost_totals_by_id[key_id] += max(0.0, float(runtime_cost))
        self._action_reward_totals_by_id[action_key_id] += float(reward)
        self._action_reward_counts_by_id[action_key_id] += 1
        if not preflight_valid:
            self._invalid_counts_by_id[key_id] += 1
        if false_positive:
            self._false_positive_counts_by_id[key_id] += 1

    def transfer_signal(self, *, scope: str, action_id: str, version_id: str) -> float:
        if not version_id:
            return 0.0
        exact_key_id = _version_key_id(scope, action_id, version_id)
        exact_count = self._reward_counts_by_id[exact_key_id]
        if exact_count > 0:
            return self._key_signal(exact_key_id)
        action_key_id = _scope_action_key_id(scope, action_id)
        totals = self._action_reward_totals_by_id[action_key_id]
        counts = self._action_reward_counts_by_id[action_key_id]
        if counts <= 0:
            return 0.0
        return 0.35 * _bounded_confident_mean_reward(totals, counts, max_abs=3.0)

    def health_penalty(self, *, scope: str, action_id: str, version_id: str) -> float:
        if not version_id:
            return 0.0
        key_id = _version_key_id(scope, action_id, version_id)
        count = self._reward_counts_by_id[key_id]
        if count <= 0:
            return 0.0
        invalid_rate = self._invalid_counts_by_id[key_id] / count
        false_positive_rate = self._false_positive_counts_by_id[key_id] / count
        runtime_cost_rate = max(0.0, float(self._runtime_cost_totals_by_id[key_id]) / count)
        return min(1.5, 0.25 * invalid_rate + 0.65 * false_positive_rate + 0.35 * runtime_cost_rate)

    def _key_signal(self, key_id: int) -> float:
        return _bounded_confident_mean_reward(
            self._reward_totals_by_id[key_id],
            self._reward_counts_by_id[key_id],
            max_abs=3.0,
        )

    def _rebuild_action_indexes(self) -> None:
        self._action_reward_totals_by_id.clear()
        self._action_reward_counts_by_id.clear()
        for key_id, total in self._reward_totals_by_id.items():
            action_key = _scope_action_prefix(_feature_text_from_id(key_id))
            if not action_key:
                continue
            action_key_id = intern_feature(action_key)
            self._action_reward_totals_by_id[action_key_id] += float(total)
        for key_id, count in self._reward_counts_by_id.items():
            action_key = _scope_action_prefix(_feature_text_from_id(key_id))
            if not action_key:
                continue
            action_key_id = intern_feature(action_key)
            self._action_reward_counts_by_id[action_key_id] += int(count)

    def to_state_dict(self) -> dict[str, Any]:
        return {
            "reward_totals": export_feature_mapping(self._reward_totals_by_id),
            "reward_counts": export_feature_counter(self._reward_counts_by_id),
            "runtime_cost_totals": export_feature_mapping(self._runtime_cost_totals_by_id),
            "invalid_counts": export_feature_counter(self._invalid_counts_by_id),
            "false_positive_counts": export_feature_counter(self._false_positive_counts_by_id),
        }

    @classmethod
    def from_state_dict(cls, data: Mapping[str, Any] | None) -> "VersionFeedbackMemory":
        if not isinstance(data, Mapping):
            return cls()
        memory = cls()
        memory._reward_totals_by_id = Counter(
            import_feature_float_mapping(data.get("reward_totals", {}) or {})
        )
        memory._reward_counts_by_id = import_feature_counter(data.get("reward_counts", {}) or {})
        memory._runtime_cost_totals_by_id = Counter(
            import_feature_float_mapping(data.get("runtime_cost_totals", {}) or {})
        )
        memory._invalid_counts_by_id = import_feature_counter(data.get("invalid_counts", {}) or {})
        memory._false_positive_counts_by_id = import_feature_counter(
            data.get("false_positive_counts", {}) or {}
        )
        memory._rebuild_action_indexes()
        return memory


@dataclass(slots=True)
class ContinualPriorityMemory:
    _feature_priority_totals_by_id: Counter[int] = field(default_factory=Counter, repr=False)
    _feature_counts_by_id: Counter[int] = field(default_factory=Counter, repr=False)
    _feature_health_penalty_totals_by_id: Counter[int] = field(default_factory=Counter, repr=False)
    _feature_health_counts_by_id: Counter[int] = field(default_factory=Counter, repr=False)
    family_priorities: dict[str, float] = field(default_factory=dict)
    family_health_penalties: dict[str, float] = field(default_factory=dict)
    status_counts: Counter[str] = field(default_factory=Counter)
    imported_ledger_count: int = 0
    imported_family_count: int = 0
    imported_health_feedback_count: int = 0

    @property
    def feature_priority_totals(self) -> Counter[str]:
        return Counter(export_feature_mapping(self._feature_priority_totals_by_id))

    @property
    def feature_counts(self) -> Counter[str]:
        return Counter(export_feature_counter(self._feature_counts_by_id))

    @property
    def feature_health_penalty_totals(self) -> Counter[str]:
        return Counter(export_feature_mapping(self._feature_health_penalty_totals_by_id))

    @property
    def feature_health_counts(self) -> Counter[str]:
        return Counter(export_feature_counter(self._feature_health_counts_by_id))

    def ingest_ledger(self, ledger: Mapping[str, Any] | None) -> dict[str, Any]:
        if not isinstance(ledger, Mapping):
            return self.summary()
        rows = _ledger_priority_rows(ledger)
        if not rows:
            return self.summary()
        self.imported_ledger_count += 1
        for row in rows:
            family = str(row.get("family", "") or "").strip()
            if not family:
                continue
            status = str(row.get("status", "") or "unknown").strip() or "unknown"
            priority = _bounded(_float_value(row.get("priority"), _continual_priority(status)), 2.0)
            self.family_priorities[family] = max(priority, float(self.family_priorities.get(family, 0.0)))
            health_penalty = _ledger_family_health_penalty(row)
            if health_penalty > 0.0:
                self.family_health_penalties[family] = max(
                    health_penalty,
                    float(self.family_health_penalties.get(family, 0.0)),
                )
                self.imported_health_feedback_count += 1
            self.status_counts[status] += 1
            self.imported_family_count += 1
            for feature in continual_priority_features(family=family, status=status):
                feature_id = intern_feature(feature)
                self._feature_priority_totals_by_id[feature_id] += priority
                self._feature_counts_by_id[feature_id] += 1
                if health_penalty > 0.0:
                    self._feature_health_penalty_totals_by_id[feature_id] += health_penalty
                    self._feature_health_counts_by_id[feature_id] += 1
        return self.summary()

    def merge(self, other: "ContinualPriorityMemory") -> dict[str, Any]:
        if not isinstance(other, ContinualPriorityMemory):
            return self.summary()
        self._feature_priority_totals_by_id.update(other._feature_priority_totals_by_id)
        self._feature_counts_by_id.update(other._feature_counts_by_id)
        self._feature_health_penalty_totals_by_id.update(other._feature_health_penalty_totals_by_id)
        self._feature_health_counts_by_id.update(other._feature_health_counts_by_id)
        self.status_counts.update(other.status_counts)
        self.imported_ledger_count += int(other.imported_ledger_count)
        self.imported_family_count += int(other.imported_family_count)
        self.imported_health_feedback_count += int(other.imported_health_feedback_count)
        for family, priority in other.family_priorities.items():
            if not str(family).strip():
                continue
            self.family_priorities[str(family)] = max(
                float(priority or 0.0),
                float(self.family_priorities.get(str(family), 0.0)),
            )
        for family, penalty in other.family_health_penalties.items():
            if not str(family).strip():
                continue
            self.family_health_penalties[str(family)] = max(
                float(penalty or 0.0),
                float(self.family_health_penalties.get(str(family), 0.0)),
            )
        return self.summary()

    def priority_signal(
        self,
        *,
        scope: str,
        action_id: str,
        context_features: Iterable[Any] = (),
        version_id: str = "",
    ) -> float:
        query_features = continual_priority_query_features(
            scope=scope,
            action_id=action_id,
            context_features=context_features,
            version_id=version_id,
        )
        weighted_total = 0.0
        matched = 0
        for feature_id in _feature_ids_from_normalized(query_features):
            count = int(self._feature_counts_by_id[feature_id])
            if count <= 0:
                continue
            weighted_total += float(self._feature_priority_totals_by_id[feature_id]) / count
            matched += 1
        if matched <= 0:
            return 0.0
        health_penalty_total = 0.0
        health_matched = 0
        for feature_id in _feature_ids_from_normalized(query_features):
            count = int(self._feature_health_counts_by_id[feature_id])
            if count <= 0:
                continue
            health_penalty_total += float(self._feature_health_penalty_totals_by_id[feature_id]) / count
            health_matched += 1
        health_penalty = health_penalty_total / health_matched if health_matched else 0.0
        return max(0.0, min(2.0, weighted_total / matched - health_penalty))

    def to_state_dict(self) -> dict[str, Any]:
        return {
            "feature_priority_totals": export_feature_mapping(self._feature_priority_totals_by_id),
            "feature_counts": export_feature_counter(self._feature_counts_by_id),
            "feature_health_penalty_totals": export_feature_mapping(
                self._feature_health_penalty_totals_by_id
            ),
            "feature_health_counts": export_feature_counter(self._feature_health_counts_by_id),
            "family_priorities": dict(sorted(self.family_priorities.items())),
            "family_health_penalties": dict(sorted(self.family_health_penalties.items())),
            "status_counts": dict(self.status_counts),
            "imported_ledger_count": int(self.imported_ledger_count),
            "imported_family_count": int(self.imported_family_count),
            "imported_health_feedback_count": int(self.imported_health_feedback_count),
        }

    @classmethod
    def from_state_dict(cls, data: Mapping[str, Any] | None) -> "ContinualPriorityMemory":
        if not isinstance(data, Mapping):
            return cls()
        return cls(
            _feature_priority_totals_by_id=Counter(
                import_feature_float_mapping(data.get("feature_priority_totals", {}) or {})
            ),
            _feature_counts_by_id=import_feature_counter(data.get("feature_counts", {}) or {}),
            _feature_health_penalty_totals_by_id=Counter(
                import_feature_float_mapping(data.get("feature_health_penalty_totals", {}) or {})
            ),
            _feature_health_counts_by_id=import_feature_counter(
                data.get("feature_health_counts", {}) or {}
            ),
            family_priorities={
                str(family): float(priority or 0.0)
                for family, priority in (data.get("family_priorities", {}) or {}).items()
                if str(family).strip()
            },
            family_health_penalties={
                str(family): float(penalty or 0.0)
                for family, penalty in (data.get("family_health_penalties", {}) or {}).items()
                if str(family).strip()
            },
            status_counts=_int_counter(data.get("status_counts", {}) or {}),
            imported_ledger_count=int(data.get("imported_ledger_count", 0) or 0),
            imported_family_count=int(data.get("imported_family_count", 0) or 0),
            imported_health_feedback_count=int(
                data.get("imported_health_feedback_count", 0) or 0
            ),
        )

    def summary(self) -> dict[str, Any]:
        top_families = sorted(
            (
                {"family": family, "priority": priority}
                for family, priority in self.family_priorities.items()
            ),
            key=lambda row: (-float(row["priority"]), row["family"]),
        )[:32]
        return {
            "schema_version": "continual-priority-memory-v1",
            "imported_ledger_count": int(self.imported_ledger_count),
            "imported_family_count": int(self.imported_family_count),
            "imported_health_feedback_count": int(self.imported_health_feedback_count),
            "family_count": len(self.family_priorities),
            "feature_count": len(self.feature_counts),
            "health_feedback_feature_count": len(self.feature_health_counts),
            "status_counts": dict(sorted(self.status_counts.items())),
            "top_families": top_families,
        }


@dataclass(slots=True)
class ExplorationMemory:
    _context_counts_by_id: Counter[int] = field(default_factory=Counter, repr=False)
    _action_counts_by_id: Counter[int] = field(default_factory=Counter, repr=False)
    _context_reward_totals_by_id: Counter[int] = field(default_factory=Counter, repr=False)
    _action_reward_totals_by_id: Counter[int] = field(default_factory=Counter, repr=False)
    total_records: int = 0

    @property
    def context_counts(self) -> Counter[str]:
        return Counter(export_feature_counter(self._context_counts_by_id))

    @property
    def action_counts(self) -> Counter[str]:
        return Counter(export_feature_counter(self._action_counts_by_id))

    @property
    def context_reward_totals(self) -> Counter[str]:
        return Counter(export_feature_mapping(self._context_reward_totals_by_id))

    @property
    def action_reward_totals(self) -> Counter[str]:
        return Counter(export_feature_mapping(self._action_reward_totals_by_id))

    def record(
        self,
        *,
        scope: str,
        action_id: str,
        context_features: Iterable[Any] = (),
        reward: float,
        version_id: str = "",
    ) -> None:
        features = action_context_features(scope, action_id, context_features, version_id=version_id)
        action_key_id = _action_key_id(scope, action_id, version_id)
        self._action_counts_by_id[action_key_id] += 1
        self._action_reward_totals_by_id[action_key_id] += float(reward)
        for feature_id in _feature_ids_from_normalized(features):
            self._context_counts_by_id[feature_id] += 1
            self._context_reward_totals_by_id[feature_id] += float(reward)
        self.total_records += 1

    def exploration_bonus(
        self,
        *,
        scope: str,
        action_id: str,
        context_features: Iterable[Any] = (),
        version_id: str = "",
    ) -> float:
        features = action_context_features(scope, action_id, context_features, version_id=version_id)
        if not features:
            return 0.0
        action_key_id = _action_key_id(scope, action_id, version_id)
        action_count = self._action_counts_by_id[action_key_id]
        novelty = 1.0 / math.sqrt(1.0 + action_count)
        feature_ids = _feature_ids_from_normalized(features)
        context_coldness = sum(
            1.0 / math.sqrt(1.0 + self._context_counts_by_id[feature_id])
            for feature_id in feature_ids
        )
        context_coldness /= math.sqrt(len(features))
        potential = max(0.0, self._action_mean(action_key_id))
        if potential <= 0.0:
            potential = max(0.0, self._context_mean(feature_ids))
        # Cold regions receive exploration pressure; early positive evidence keeps them from being ignored.
        return min(2.0, 0.55 * novelty + 0.35 * context_coldness + 0.20 * potential)

    def _action_mean(self, action_key_id: int) -> float:
        count = self._action_counts_by_id[action_key_id]
        return float(self._action_reward_totals_by_id[action_key_id]) / count if count else 0.0

    def _context_mean(self, feature_ids: tuple[int, ...]) -> float:
        reward = 0.0
        count = 0
        for feature_id in feature_ids:
            feature_count = self._context_counts_by_id[feature_id]
            if feature_count <= 0:
                continue
            reward += float(self._context_reward_totals_by_id[feature_id])
            count += int(feature_count)
        return reward / count if count else 0.0

    def to_state_dict(self) -> dict[str, Any]:
        return {
            "context_counts": export_feature_counter(self._context_counts_by_id),
            "action_counts": export_feature_counter(self._action_counts_by_id),
            "context_reward_totals": export_feature_mapping(self._context_reward_totals_by_id),
            "action_reward_totals": export_feature_mapping(self._action_reward_totals_by_id),
            "total_records": self.total_records,
        }

    @classmethod
    def from_state_dict(cls, data: Mapping[str, Any] | None) -> "ExplorationMemory":
        if not isinstance(data, Mapping):
            return cls()
        return cls(
            _context_counts_by_id=import_feature_counter(data.get("context_counts", {}) or {}),
            _action_counts_by_id=import_feature_counter(data.get("action_counts", {}) or {}),
            _context_reward_totals_by_id=Counter(
                import_feature_float_mapping(data.get("context_reward_totals", {}) or {})
            ),
            _action_reward_totals_by_id=Counter(
                import_feature_float_mapping(data.get("action_reward_totals", {}) or {})
            ),
            total_records=int(data.get("total_records", 0) or 0),
        )

    def summary(self) -> dict[str, Any]:
        return {
            "total_records": int(self.total_records),
            "context_count": len(self.context_counts),
            "action_count": len(self.action_counts),
            "cold_context_count": sum(1 for count in self.context_counts.values() if int(count) <= 1),
            "cold_action_count": sum(1 for count in self.action_counts.values() if int(count) <= 1),
        }


@dataclass(slots=True)
class TopLevelWeightCalibrator:
    learning_rate: float = 0.05
    l2: float = 0.002
    min_scale: float = 0.25
    max_scale: float = 3.0
    health_min_scale: float = 0.5
    health_max_scale: float = 3.0
    _component_scales_dense: list[float] = field(
        default_factory=lambda: [1.0] * _SELF_CALIBRATED_COMPONENT_COUNT,
        repr=False,
    )
    _component_update_counts_dense: list[int] = field(
        default_factory=lambda: [0] * _SELF_CALIBRATED_COMPONENT_COUNT,
        repr=False,
    )
    _component_reward_totals_dense: list[float] = field(
        default_factory=lambda: [0.0] * _SELF_CALIBRATED_COMPONENT_COUNT,
        repr=False,
    )
    total_updates: int = 0
    last_error: float = 0.0

    @property
    def component_scales(self) -> dict[str, float]:
        return {
            component: float(self._component_scales_dense[index])
            for index, component in enumerate(_SELF_CALIBRATED_COMPONENTS)
            if float(self._component_scales_dense[index]) != 1.0
        }

    @component_scales.setter
    def component_scales(self, values: Mapping[str, Any] | None) -> None:
        dense = [1.0] * _SELF_CALIBRATED_COMPONENT_COUNT
        if isinstance(values, Mapping):
            for component, scale in values.items():
                index = _SELF_CALIBRATED_COMPONENT_INDEX.get(str(component))
                if index is None:
                    continue
                dense[index] = _clip_component_scale(
                    str(component),
                    float(scale or 1.0),
                    min_scale=self.min_scale,
                    max_scale=self.max_scale,
                    health_min_scale=self.health_min_scale,
                    health_max_scale=self.health_max_scale,
                )
        self._component_scales_dense = dense

    @property
    def component_update_counts(self) -> Counter[str]:
        return Counter(
            {
                component: int(self._component_update_counts_dense[index])
                for index, component in enumerate(_SELF_CALIBRATED_COMPONENTS)
                if int(self._component_update_counts_dense[index]) > 0
            }
        )

    @component_update_counts.setter
    def component_update_counts(self, values: Mapping[str, Any] | None) -> None:
        dense = [0] * _SELF_CALIBRATED_COMPONENT_COUNT
        if isinstance(values, Mapping):
            for component, count in values.items():
                index = _SELF_CALIBRATED_COMPONENT_INDEX.get(str(component))
                if index is not None:
                    dense[index] = int(count or 0)
        self._component_update_counts_dense = dense

    @property
    def component_reward_totals(self) -> Counter[str]:
        return Counter(
            {
                component: float(self._component_reward_totals_dense[index])
                for index, component in enumerate(_SELF_CALIBRATED_COMPONENTS)
                if float(self._component_reward_totals_dense[index]) != 0.0
            }
        )

    @component_reward_totals.setter
    def component_reward_totals(self, values: Mapping[str, Any] | None) -> None:
        dense = [0.0] * _SELF_CALIBRATED_COMPONENT_COUNT
        if isinstance(values, Mapping):
            for component, total in values.items():
                index = _SELF_CALIBRATED_COMPONENT_INDEX.get(str(component))
                if index is not None:
                    dense[index] = float(total or 0.0)
        self._component_reward_totals_dense = dense

    def scale(self, component: str) -> float:
        index = _SELF_CALIBRATED_COMPONENT_INDEX.get(str(component))
        if index is None:
            return 1.0
        return float(self._component_scales_dense[index] or 1.0)

    def score(self, weighted_components: Mapping[str, float] | None) -> float:
        if not weighted_components:
            return 0.0
        return self.score_dense(self._dense_component_values(weighted_components))

    def score_dense(self, weighted_components_dense: tuple[float, ...]) -> float:
        if not weighted_components_dense:
            return 0.0
        total = 0.0
        scales = self._component_scales_dense
        for index, value in enumerate(weighted_components_dense):
            if value == 0.0:
                continue
            if index == _HEALTH_PENALTY_COMPONENT_INDEX:
                total -= scales[index] * value
            else:
                total += scales[index] * value
        return total

    def update(self, weighted_components: Mapping[str, float] | None, reward: float, *, max_abs_reward: float = 6.0) -> float:
        if not weighted_components:
            return 0.0
        return self.update_dense(
            self._dense_component_values(weighted_components),
            reward,
            max_abs_reward=max_abs_reward,
        )

    def update_dense(
        self,
        weighted_components_dense: tuple[float, ...],
        reward: float,
        *,
        max_abs_reward: float = 6.0,
    ) -> float:
        if not weighted_components_dense:
            return 0.0
        target = _bounded(float(reward), max_abs_reward)
        prediction = self.score_dense(weighted_components_dense)
        error = target - prediction
        norm = 1.0 + sum(abs(value) for value in weighted_components_dense)
        scales = self._component_scales_dense
        counts = self._component_update_counts_dense
        reward_totals = self._component_reward_totals_dense
        for index, value in enumerate(weighted_components_dense):
            if value == 0.0:
                continue
            current = scales[index]
            component = _SELF_CALIBRATED_COMPONENTS[index]
            sign = -1.0 if index == _HEALTH_PENALTY_COMPONENT_INDEX else 1.0
            proposal = current + self.learning_rate * (
                ((error * sign * value) / norm) - self.l2 * (current - 1.0)
            )
            scales[index] = _clip_component_scale(
                component,
                proposal,
                min_scale=self.min_scale,
                max_scale=self.max_scale,
                health_min_scale=self.health_min_scale,
                health_max_scale=self.health_max_scale,
            )
            counts[index] += 1
            reward_totals[index] += target
        self.total_updates += 1
        self.last_error = error
        return error

    def snapshot(self) -> list[dict[str, Any]]:
        return [
            {
                "component": component,
                "scale": self.scale(component),
                "updates": int(self._component_update_counts_dense[index]),
                "mean_reward": (
                    float(self._component_reward_totals_dense[index]) / int(self._component_update_counts_dense[index])
                    if self._component_update_counts_dense[index]
                    else 0.0
                ),
            }
            for index, component in enumerate(_SELF_CALIBRATED_COMPONENTS)
        ]

    def to_state_dict(self) -> dict[str, Any]:
        return {
            "learning_rate": self.learning_rate,
            "l2": self.l2,
            "min_scale": self.min_scale,
            "max_scale": self.max_scale,
            "health_min_scale": self.health_min_scale,
            "health_max_scale": self.health_max_scale,
            "component_scales": self.component_scales,
            "component_update_counts": dict(self.component_update_counts),
            "component_reward_totals": dict(self.component_reward_totals),
            "total_updates": int(self.total_updates),
            "last_error": float(self.last_error),
        }

    @classmethod
    def from_state_dict(cls, data: Mapping[str, Any] | None) -> "TopLevelWeightCalibrator":
        if not isinstance(data, Mapping):
            return cls()
        calibrator = cls(
            learning_rate=_float_field(data, "learning_rate", 0.05),
            l2=_float_field(data, "l2", 0.002),
            min_scale=_float_field(data, "min_scale", 0.25),
            max_scale=_float_field(data, "max_scale", 3.0),
            health_min_scale=_float_field(data, "health_min_scale", 0.5),
            health_max_scale=_float_field(data, "health_max_scale", 3.0),
        )
        calibrator.component_scales = data.get("component_scales", {}) or {}
        calibrator.component_update_counts = _int_counter(data.get("component_update_counts", {}) or {})
        calibrator.component_reward_totals = _float_counter(data.get("component_reward_totals", {}) or {})
        calibrator.total_updates = int(data.get("total_updates", 0) or 0)
        calibrator.last_error = float(data.get("last_error", 0.0) or 0.0)
        return calibrator

    def _dense_component_values(self, weighted_components: Mapping[str, float]) -> tuple[float, ...]:
        return tuple(
            float(weighted_components.get(component, 0.0) or 0.0)
            for component in _SELF_CALIBRATED_COMPONENTS
        )

    def scales_dense(self) -> tuple[float, ...]:
        return tuple(self._component_scales_dense)


@dataclass(slots=True)
class ContextualBandit:
    exploration_weight: float = 0.45
    model_weight: float = 0.35
    uncertainty_weight: float = 0.25
    version_weight: float = 0.30
    continual_priority_weight: float = 0.35
    active_learning_weight: float = 0.20
    reward_model: OnlineRewardModel = field(default_factory=OnlineRewardModel)
    weight_calibrator: TopLevelWeightCalibrator = field(default_factory=TopLevelWeightCalibrator)
    arms: dict[str, ActionStats] = field(default_factory=dict)
    total_pulls: int = 0

    def choose(
        self,
        action_ids: Iterable[Any],
        *,
        scope: str,
        context_features: Iterable[Any] = (),
        version_id: str = "",
        version_memory: VersionFeedbackMemory | None = None,
        continual_priority_memory: ContinualPriorityMemory | None = None,
        exploration_memory: ExplorationMemory | None = None,
        enable_reward_model: bool = True,
        enable_continual_learning: bool = True,
    ) -> str:
        best = self.choose_dense(
            action_ids,
            scope=scope,
            context_features=context_features,
            version_id=version_id,
            version_memory=version_memory,
            continual_priority_memory=continual_priority_memory,
            exploration_memory=exploration_memory,
            enable_reward_model=enable_reward_model,
            enable_continual_learning=enable_continual_learning,
        )
        return best.action_id if best is not None else ""

    def choose_dense(
        self,
        action_ids: Iterable[Any],
        *,
        scope: str,
        context_features: Iterable[Any] = (),
        version_id: str = "",
        version_memory: VersionFeedbackMemory | None = None,
        continual_priority_memory: ContinualPriorityMemory | None = None,
        exploration_memory: ExplorationMemory | None = None,
        enable_reward_model: bool = True,
        enable_continual_learning: bool = True,
    ) -> DenseBanditChoice | None:
        best: DenseBanditChoice | None = None
        for action_id in _unique_nonempty(action_ids):
            choice = self._choose_action_dense(
                action_id,
                scope=scope,
                context_features=context_features,
                version_id=version_id,
                version_memory=version_memory,
                continual_priority_memory=continual_priority_memory,
                exploration_memory=exploration_memory,
                enable_reward_model=enable_reward_model,
                enable_continual_learning=enable_continual_learning,
            )
            if best is None or choice.sort_key() > best.sort_key():
                best = choice
        return best

    def rank(
        self,
        action_ids: Iterable[Any],
        *,
        scope: str,
        context_features: Iterable[Any] = (),
        version_id: str = "",
        version_memory: VersionFeedbackMemory | None = None,
        continual_priority_memory: ContinualPriorityMemory | None = None,
        exploration_memory: ExplorationMemory | None = None,
        enable_reward_model: bool = True,
        enable_continual_learning: bool = True,
    ) -> list[dict[str, Any]]:
        dense_rows = self.rank_dense(
            action_ids,
            scope=scope,
            context_features=context_features,
            version_id=version_id,
            version_memory=version_memory,
            continual_priority_memory=continual_priority_memory,
            exploration_memory=exploration_memory,
            enable_reward_model=enable_reward_model,
            enable_continual_learning=enable_continual_learning,
        )
        calibrated_component_scales = _dense_component_dict(self.weight_calibrator.scales_dense())
        return [
            self._materialize_dense_score(
                dense_row,
                calibrated_component_scales=calibrated_component_scales,
            )
            for dense_row in dense_rows
        ]

    def rank_top(
        self,
        action_ids: Iterable[Any],
        *,
        scope: str,
        limit: int,
        context_features: Iterable[Any] = (),
        version_id: str = "",
        version_memory: VersionFeedbackMemory | None = None,
        continual_priority_memory: ContinualPriorityMemory | None = None,
        exploration_memory: ExplorationMemory | None = None,
        enable_reward_model: bool = True,
        enable_continual_learning: bool = True,
    ) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        dense_rows = self.rank_dense(
            action_ids,
            scope=scope,
            context_features=context_features,
            version_id=version_id,
            version_memory=version_memory,
            continual_priority_memory=continual_priority_memory,
            exploration_memory=exploration_memory,
            enable_reward_model=enable_reward_model,
            enable_continual_learning=enable_continual_learning,
        )[:limit]
        calibrated_component_scales = _dense_component_dict(self.weight_calibrator.scales_dense())
        return [
            self._materialize_dense_score(
                dense_row,
                calibrated_component_scales=calibrated_component_scales,
            )
            for dense_row in dense_rows
        ]

    def rank_dense(
        self,
        action_ids: Iterable[Any],
        *,
        scope: str,
        context_features: Iterable[Any] = (),
        version_id: str = "",
        version_memory: VersionFeedbackMemory | None = None,
        continual_priority_memory: ContinualPriorityMemory | None = None,
        exploration_memory: ExplorationMemory | None = None,
        enable_reward_model: bool = True,
        enable_continual_learning: bool = True,
    ) -> list[DenseBanditScore]:
        dense_rows = [
            self._score_action_dense(
                action_id,
                scope=scope,
                context_features=context_features,
                version_id=version_id,
                version_memory=version_memory,
                continual_priority_memory=continual_priority_memory,
                exploration_memory=exploration_memory,
                enable_reward_model=enable_reward_model,
                enable_continual_learning=enable_continual_learning,
            )
            for action_id in _unique_nonempty(action_ids)
        ]
        return sorted(dense_rows, key=DenseBanditScore.sort_key, reverse=True)

    def score_action(
        self,
        action_id: Any,
        *,
        scope: str,
        context_features: Iterable[Any] = (),
        version_id: str = "",
        version_memory: VersionFeedbackMemory | None = None,
        continual_priority_memory: ContinualPriorityMemory | None = None,
        exploration_memory: ExplorationMemory | None = None,
        enable_reward_model: bool = True,
        enable_continual_learning: bool = True,
    ) -> dict[str, Any]:
        dense_score = self._score_action_dense(
            action_id,
            scope=scope,
            context_features=context_features,
            version_id=version_id,
            version_memory=version_memory,
            continual_priority_memory=continual_priority_memory,
            exploration_memory=exploration_memory,
            enable_reward_model=enable_reward_model,
            enable_continual_learning=enable_continual_learning,
        )
        return self._materialize_dense_score(
            dense_score,
            calibrated_component_scales=_dense_component_dict(self.weight_calibrator.scales_dense()),
        )

    def record(
        self,
        action_id: Any,
        *,
        scope: str,
        context_features: Iterable[Any] = (),
        reward: float,
        runtime_cost: float = 0.0,
        version_id: str = "",
        preflight_valid: bool = True,
        fallback_used: bool = False,
        false_positive: bool = False,
        enable_reward_model: bool = True,
        version_memory: VersionFeedbackMemory | None = None,
        continual_priority_memory: ContinualPriorityMemory | None = None,
        exploration_memory: ExplorationMemory | None = None,
        enable_continual_learning: bool = True,
    ) -> float:
        normalized_action = str(action_id).strip()
        bounded_reward = float(reward)
        arm, features, _, weighted_components_dense = self._score_components_dense(
            normalized_action,
            scope=scope,
            context_features=context_features,
            version_id=version_id,
            version_memory=version_memory,
            continual_priority_memory=continual_priority_memory,
            exploration_memory=exploration_memory,
            enable_reward_model=enable_reward_model,
            enable_continual_learning=enable_continual_learning,
        )
        arm.pulls += 1
        arm.total_reward += bounded_reward
        arm.reward_sq_total += bounded_reward * bounded_reward
        arm.last_reward = bounded_reward
        arm.runtime_cost_total += max(0.0, float(runtime_cost))
        arm.last_step = self.total_pulls
        if not preflight_valid:
            arm.invalid_count += 1
        if fallback_used:
            arm.fallback_count += 1
        if false_positive:
            arm.false_positive_count += 1
        self.total_pulls += 1
        self.weight_calibrator.update_dense(weighted_components_dense, bounded_reward)
        if not enable_reward_model:
            return 0.0
        return self.reward_model.update_normalized(features, bounded_reward)

    def _arm(self, action_id: str) -> ActionStats:
        if action_id not in self.arms:
            self.arms[action_id] = ActionStats(action_id=action_id)
        return self.arms[action_id]

    def _choose_action_dense(
        self,
        action_id: Any,
        *,
        scope: str,
        context_features: Iterable[Any],
        version_id: str,
        version_memory: VersionFeedbackMemory | None,
        continual_priority_memory: ContinualPriorityMemory | None,
        exploration_memory: ExplorationMemory | None,
        enable_reward_model: bool,
        enable_continual_learning: bool,
    ) -> DenseBanditChoice:
        normalized_action = str(action_id).strip()
        arm, _, _, weighted_components_dense = self._score_components_dense(
            normalized_action,
            scope=scope,
            context_features=context_features,
            version_id=version_id,
            version_memory=version_memory,
            continual_priority_memory=continual_priority_memory,
            exploration_memory=exploration_memory,
            enable_reward_model=enable_reward_model,
            enable_continual_learning=enable_continual_learning,
        )
        return DenseBanditChoice(
            action_id=normalized_action,
            score=self.weight_calibrator.score_dense(weighted_components_dense),
            pulls=arm.pulls,
        )

    def _score_action_dense(
        self,
        action_id: Any,
        *,
        scope: str,
        context_features: Iterable[Any],
        version_id: str,
        version_memory: VersionFeedbackMemory | None,
        continual_priority_memory: ContinualPriorityMemory | None,
        exploration_memory: ExplorationMemory | None,
        enable_reward_model: bool,
        enable_continual_learning: bool,
    ) -> DenseBanditScore:
        normalized_action = str(action_id).strip()
        arm, _, signal_components_dense, weighted_components_dense = self._score_components_dense(
            normalized_action,
            scope=scope,
            context_features=context_features,
            version_id=version_id,
            version_memory=version_memory,
            continual_priority_memory=continual_priority_memory,
            exploration_memory=exploration_memory,
            enable_reward_model=enable_reward_model,
            enable_continual_learning=enable_continual_learning,
        )
        return DenseBanditScore(
            action_id=normalized_action,
            score=self.weight_calibrator.score_dense(weighted_components_dense),
            pulls=arm.pulls,
            mean_reward=arm.mean_reward,
            signal_components_dense=signal_components_dense,
            weighted_components_dense=weighted_components_dense,
        )

    def _materialize_dense_score(
        self,
        dense_score: DenseBanditScore,
        *,
        calibrated_component_scales: Mapping[str, float] | None = None,
    ) -> dict[str, Any]:
        row = _dense_component_dict(dense_score.signal_components_dense)
        row.update(
            {
                "weighted_components": _dense_component_dict(dense_score.weighted_components_dense),
                "calibrated_component_scales": (
                    dict(calibrated_component_scales)
                    if calibrated_component_scales is not None
                    else _dense_component_dict(self.weight_calibrator.scales_dense())
                ),
            }
        )
        row.update(
            {
                "action_id": dense_score.action_id,
                "score": dense_score.score,
                "pulls": dense_score.pulls,
                "mean_reward": dense_score.mean_reward,
            }
        )
        return row

    def _score_components_dense(
        self,
        action_id: str,
        *,
        scope: str,
        context_features: Iterable[Any],
        version_id: str,
        version_memory: VersionFeedbackMemory | None,
        continual_priority_memory: ContinualPriorityMemory | None,
        exploration_memory: ExplorationMemory | None,
        enable_reward_model: bool,
        enable_continual_learning: bool,
    ) -> tuple[ActionStats, tuple[str, ...], tuple[float, ...], tuple[float, ...]]:
        normalized_scope = str(scope or "")
        normalized_version = str(version_id or "").strip()
        normalized_context_features = tuple(str(feature) for feature in context_features)
        arm = self._arm(action_id)
        features = _action_context_features_cached(
            normalized_scope,
            action_id,
            normalized_context_features,
            bool(normalized_version),
        )
        model_prediction = self.reward_model.predict_normalized(features) if enable_reward_model else 0.0
        uncertainty = self.reward_model.uncertainty_normalized(features) if enable_reward_model else 0.0
        version_signal = (
            version_memory.transfer_signal(
                scope=normalized_scope,
                action_id=action_id,
                version_id=normalized_version,
            )
            if enable_continual_learning and version_memory is not None
            else 0.0
        )
        continual_priority_signal = (
            continual_priority_memory.priority_signal(
                scope=normalized_scope,
                action_id=action_id,
                context_features=normalized_context_features,
                version_id=normalized_version,
            )
            if enable_continual_learning and continual_priority_memory is not None
            else 0.0
        )
        exploration_bonus = (
            exploration_memory.exploration_bonus(
                scope=normalized_scope,
                action_id=action_id,
                context_features=normalized_context_features,
                version_id=normalized_version,
            )
            if exploration_memory is not None
            else 0.0
        )
        health_penalty = arm.health_penalty()
        if enable_continual_learning and version_memory is not None:
            health_penalty += version_memory.health_penalty(
                scope=normalized_scope,
                action_id=action_id,
                version_id=normalized_version,
            )
        signal_components_dense = (
            arm.reward_signal(),
            model_prediction,
            uncertainty,
            self.exploration_weight * math.sqrt(math.log(self.total_pulls + 2.0) / (1.0 + arm.pulls)),
            version_signal,
            continual_priority_signal,
            exploration_bonus,
            health_penalty,
        )
        weighted_components_dense = (
            signal_components_dense[_REWARD_SIGNAL_COMPONENT_INDEX],
            self.model_weight * signal_components_dense[_MODEL_PREDICTION_COMPONENT_INDEX],
            self.uncertainty_weight * signal_components_dense[_UNCERTAINTY_COMPONENT_INDEX],
            signal_components_dense[_EXPLORATION_COMPONENT_INDEX],
            self.version_weight * signal_components_dense[_VERSION_SIGNAL_COMPONENT_INDEX],
            self.continual_priority_weight * signal_components_dense[_CONTINUAL_PRIORITY_SIGNAL_COMPONENT_INDEX],
            self.active_learning_weight * signal_components_dense[_EXPLORATION_BONUS_COMPONENT_INDEX],
            signal_components_dense[_HEALTH_PENALTY_COMPONENT_INDEX],
        )
        return arm, features, signal_components_dense, weighted_components_dense

    def to_state_dict(self) -> dict[str, Any]:
        return {
            "exploration_weight": self.exploration_weight,
            "model_weight": self.model_weight,
            "uncertainty_weight": self.uncertainty_weight,
            "version_weight": self.version_weight,
            "continual_priority_weight": self.continual_priority_weight,
            "active_learning_weight": self.active_learning_weight,
            "total_pulls": self.total_pulls,
            "reward_model": self.reward_model.to_state_dict(),
            "weight_calibrator": self.weight_calibrator.to_state_dict(),
            "arms": [arm.to_state_dict() for arm in self.arms.values()],
        }

    @classmethod
    def from_state_dict(cls, data: Mapping[str, Any] | None) -> "ContextualBandit":
        if not isinstance(data, Mapping):
            return cls()
        bandit = cls(
            exploration_weight=_float_field(data, "exploration_weight", 0.45),
            model_weight=_float_field(data, "model_weight", 0.35),
            uncertainty_weight=_float_field(data, "uncertainty_weight", 0.25),
            version_weight=_float_field(data, "version_weight", 0.30),
            continual_priority_weight=_float_field(data, "continual_priority_weight", 0.35),
            active_learning_weight=_float_field(data, "active_learning_weight", 0.20),
            reward_model=OnlineRewardModel.from_state_dict(data.get("reward_model")),
            weight_calibrator=TopLevelWeightCalibrator.from_state_dict(data.get("weight_calibrator")),
        )
        bandit.total_pulls = int(data.get("total_pulls", 0) or 0)
        for raw_arm in data.get("arms", []) or []:
            if isinstance(raw_arm, Mapping):
                arm = ActionStats.from_state_dict(raw_arm)
                if arm.action_id:
                    bandit.arms[arm.action_id] = arm
        return bandit


@dataclass(slots=True)
class AdaptiveLearningState:
    bandits: dict[str, ContextualBandit] = field(default_factory=dict)
    version_memory: VersionFeedbackMemory = field(default_factory=VersionFeedbackMemory)
    continual_priority_memory: ContinualPriorityMemory = field(default_factory=ContinualPriorityMemory)
    exploration_memory: ExplorationMemory = field(default_factory=ExplorationMemory)

    def choose_dense(
        self,
        scope: str,
        action_ids: Iterable[Any],
        *,
        context_features: Iterable[Any] = (),
        version_id: str = "",
        enable_active_learning: bool = True,
        enable_reward_model: bool = True,
        enable_continual_learning: bool = True,
    ) -> DenseBanditChoice | None:
        return self._bandit(scope).choose_dense(
            action_ids,
            scope=scope,
            context_features=context_features,
            version_id=version_id,
            version_memory=self.version_memory,
            continual_priority_memory=self.continual_priority_memory,
            exploration_memory=self.exploration_memory if enable_active_learning else None,
            enable_reward_model=enable_reward_model,
            enable_continual_learning=enable_continual_learning,
        )

    def choose(
        self,
        scope: str,
        action_ids: Iterable[Any],
        *,
        context_features: Iterable[Any] = (),
        version_id: str = "",
        enable_active_learning: bool = True,
        enable_reward_model: bool = True,
        enable_continual_learning: bool = True,
    ) -> str:
        return self._bandit(scope).choose(
            action_ids,
            scope=scope,
            context_features=context_features,
            version_id=version_id,
            version_memory=self.version_memory,
            continual_priority_memory=self.continual_priority_memory,
            exploration_memory=self.exploration_memory if enable_active_learning else None,
            enable_reward_model=enable_reward_model,
            enable_continual_learning=enable_continual_learning,
        )

    def rank(
        self,
        scope: str,
        action_ids: Iterable[Any],
        *,
        context_features: Iterable[Any] = (),
        version_id: str = "",
        enable_active_learning: bool = True,
        enable_reward_model: bool = True,
        enable_continual_learning: bool = True,
    ) -> list[dict[str, Any]]:
        return self._bandit(scope).rank(
            action_ids,
            scope=scope,
            context_features=context_features,
            version_id=version_id,
            version_memory=self.version_memory,
            continual_priority_memory=self.continual_priority_memory,
            exploration_memory=self.exploration_memory if enable_active_learning else None,
            enable_reward_model=enable_reward_model,
            enable_continual_learning=enable_continual_learning,
        )

    def rank_top(
        self,
        scope: str,
        action_ids: Iterable[Any],
        *,
        limit: int,
        context_features: Iterable[Any] = (),
        version_id: str = "",
        enable_active_learning: bool = True,
        enable_reward_model: bool = True,
        enable_continual_learning: bool = True,
    ) -> list[dict[str, Any]]:
        return self._bandit(scope).rank_top(
            action_ids,
            scope=scope,
            limit=limit,
            context_features=context_features,
            version_id=version_id,
            version_memory=self.version_memory,
            continual_priority_memory=self.continual_priority_memory,
            exploration_memory=self.exploration_memory if enable_active_learning else None,
            enable_reward_model=enable_reward_model,
            enable_continual_learning=enable_continual_learning,
        )

    def score_action(
        self,
        scope: str,
        action_id: Any,
        *,
        context_features: Iterable[Any] = (),
        version_id: str = "",
        enable_active_learning: bool = True,
        enable_reward_model: bool = True,
        enable_continual_learning: bool = True,
    ) -> dict[str, Any]:
        return self._bandit(scope).score_action(
            action_id,
            scope=scope,
            context_features=context_features,
            version_id=version_id,
            version_memory=self.version_memory,
            continual_priority_memory=self.continual_priority_memory,
            exploration_memory=self.exploration_memory if enable_active_learning else None,
            enable_reward_model=enable_reward_model,
            enable_continual_learning=enable_continual_learning,
        )

    def ingest_continual_ledger(self, ledger: Mapping[str, Any] | None) -> dict[str, Any]:
        return self.continual_priority_memory.ingest_ledger(ledger)

    def record_outcome(
        self,
        scope: str,
        action_id: Any,
        *,
        context_features: Iterable[Any] = (),
        reward: float,
        runtime_cost: float = 0.0,
        version_id: str = "",
        preflight_valid: bool = True,
        fallback_used: bool = False,
        false_positive: bool = False,
        enable_reward_model: bool = True,
    ) -> float:
        normalized_scope = _token(scope or "default")
        normalized_action = str(action_id).strip()
        error = self._bandit(normalized_scope).record(
            normalized_action,
            scope=normalized_scope,
            context_features=context_features,
            reward=reward,
            runtime_cost=runtime_cost,
            version_id=version_id,
            preflight_valid=preflight_valid,
            fallback_used=fallback_used,
            false_positive=false_positive,
            enable_reward_model=enable_reward_model,
            version_memory=self.version_memory,
            continual_priority_memory=self.continual_priority_memory,
            exploration_memory=self.exploration_memory,
            enable_continual_learning=True,
        )
        self.version_memory.record(
            scope=normalized_scope,
            action_id=normalized_action,
            version_id=str(version_id or "").strip(),
            reward=reward,
            runtime_cost=runtime_cost,
            preflight_valid=preflight_valid,
            false_positive=false_positive,
        )
        self.exploration_memory.record(
            scope=normalized_scope,
            action_id=normalized_action,
            context_features=context_features,
            reward=reward,
            version_id=str(version_id or "").strip(),
        )
        return error

    def snapshot(self, *, scope: str, action_ids: Iterable[Any], context_features: Iterable[Any] = (), version_id: str = "") -> list[dict[str, Any]]:
        return self.rank(scope, action_ids, context_features=context_features, version_id=version_id)

    def health_summary(self) -> dict[str, Any]:
        bandit_count = len(self.bandits)
        arm_count = 0
        total_pulls = 0
        health_penalties: list[float] = []
        uncertainties: list[float] = []
        stale_arm_count = 0
        for scope, bandit in self.bandits.items():
            for action_id, arm in bandit.arms.items():
                arm_count += 1
                total_pulls += int(arm.pulls)
                if arm.pulls <= 0:
                    stale_arm_count += 1
                row = bandit.score_action(
                    action_id,
                    scope=scope,
                    context_features=(),
                    version_memory=self.version_memory,
                    continual_priority_memory=self.continual_priority_memory,
                    exploration_memory=self.exploration_memory,
                )
                health_penalties.append(float(row.get("health_penalty", 0.0) or 0.0))
                uncertainties.append(float(row.get("uncertainty", 0.0) or 0.0))
        return {
            "schema_version": "adaptive-learning-health-v1",
            "bandit_count": bandit_count,
            "arm_count": arm_count,
            "total_pulls": total_pulls,
            "stale_arm_count": stale_arm_count,
            "avg_health_penalty": _mean(health_penalties),
            "max_health_penalty": max(health_penalties) if health_penalties else 0.0,
            "avg_uncertainty": _mean(uncertainties),
            "version_memory_key_count": len(self.version_memory.reward_counts),
            "calibrated_component_scales": {
                scope: bandit.weight_calibrator.snapshot()
                for scope, bandit in sorted(self.bandits.items())
            },
            "continual_priority_memory": self.continual_priority_memory.summary(),
            "exploration_memory": self.exploration_memory.summary(),
        }

    def _bandit(self, scope: str) -> ContextualBandit:
        normalized_scope = _token(scope or "default")
        if normalized_scope not in self.bandits:
            self.bandits[normalized_scope] = ContextualBandit()
        return self.bandits[normalized_scope]

    def to_state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": ADAPTIVE_LEARNING_SCHEMA_VERSION,
            "bandits": {
                scope: bandit.to_state_dict()
                for scope, bandit in sorted(self.bandits.items())
            },
            "version_memory": self.version_memory.to_state_dict(),
            "continual_priority_memory": self.continual_priority_memory.to_state_dict(),
            "exploration_memory": self.exploration_memory.to_state_dict(),
        }

    @classmethod
    def from_state_dict(cls, data: Mapping[str, Any] | None) -> "AdaptiveLearningState":
        if not isinstance(data, Mapping):
            return cls()
        state = cls(
            version_memory=VersionFeedbackMemory.from_state_dict(data.get("version_memory")),
            continual_priority_memory=ContinualPriorityMemory.from_state_dict(
                data.get("continual_priority_memory")
            ),
            exploration_memory=ExplorationMemory.from_state_dict(data.get("exploration_memory")),
        )
        for scope, raw_bandit in (data.get("bandits", {}) or {}).items():
            state.bandits[_token(scope)] = ContextualBandit.from_state_dict(raw_bandit)
        return state


def action_context_features(
    scope: str,
    action_id: str,
    context_features: Iterable[Any],
    *,
    version_id: str = "",
) -> tuple[str, ...]:
    return _action_context_features_cached(
        str(scope or ""),
        str(action_id or ""),
        tuple(str(feature) for feature in context_features),
        bool(version_id),
    )


def context_features_from_mapping(data: Mapping[str, Any] | None) -> tuple[str, ...]:
    if not isinstance(data, Mapping):
        return ()
    features: list[str] = []
    for key in (
        "target_suite",
        "preset",
        "generator_profile",
        "guidance_strategy",
        "oracle_mode",
        "version_pair",
        "target_version",
        "fixed_version",
    ):
        value = str(data.get(f"scheduler_{key}", data.get(key, "")) or "").strip()
        if value:
            features.append(f"{_token(key)}:{_token(value)}")
    metamorphic_enabled = bool(
        data.get(
            "scheduler_enable_metamorphic_oracle",
            data.get("enable_metamorphic_oracle", data.get("metamorphic_oracle", False)),
        )
    )
    if metamorphic_enabled:
        features.append("metamorphic:enabled")
    if bool(data.get("scheduler_enable_feedback", data.get("enable_feedback", data.get("feedback", False)))):
        features.append("feedback:enabled")
    if bool(data.get("enable_local_source_scheduler", False)):
        features.append("source_scheduler:enabled")
    limit = _as_nonnegative_int(
        data.get(
            "scheduler_effective_metamorphic_variant_limit",
            data.get("effective_metamorphic_variant_limit", data.get("metamorphic_variant_limit")),
        )
    )
    if metamorphic_enabled and limit:
        features.append(_bucket("metamorphic_limit", limit, [(2, "small"), (6, "medium")], "large"))
    backends = data.get("backends", [])
    if isinstance(backends, (list, tuple, set)):
        backend_count = len([item for item in backends if str(item).strip()])
        features.append(_bucket("backend_count", backend_count, [(1, "single"), (3, "few")], "many"))
    for key in ("semantic_focus_families", "semantic_focus_signals", "guidance_targets"):
        values = data.get(f"scheduler_{key}", data.get(key, []))
        if isinstance(values, str):
            values = values.split(",")
        if isinstance(values, Iterable):
            for value in values:
                text = str(value).strip()
                if text:
                    features.append(f"{_token(key)}:{_token(text)}")
    return tuple(_unique_nonempty(features))


def continual_learning_summary(
    family_rows: Iterable[Mapping[str, Any]],
    *,
    version_order: Iterable[Any],
) -> dict[str, Any]:
    rows = [row for row in family_rows if isinstance(row, Mapping)]
    status_counts = Counter(str(row.get("status", "unknown") or "unknown") for row in rows)
    priorities = []
    for row in rows:
        family = str(row.get("family", "") or "").strip()
        if not family:
            continue
        status = str(row.get("status", "unknown") or "unknown")
        priorities.append(
            {
                "family": family,
                "status": status,
                "priority": _continual_priority(status),
                "suggested_action": _continual_suggested_action(status),
            }
        )
    priorities.sort(key=lambda row: (-float(row["priority"]), row["family"]))
    return {
        "schema_version": CONTINUAL_LEARNING_SCHEMA_VERSION,
        "version_count": len([item for item in version_order if str(item).strip()]),
        "status_counts": dict(sorted(status_counts.items())),
        "family_priorities": priorities,
        "methodology_claim": (
            "Cross-version outcomes are retained as continual-learning feedback so "
            "future schedules can prioritize regressions and new families while "
            "downweighting fixed, saturated, or unhealthy regions."
        ),
        "adaptive_learning_seed": continual_priority_seed(
            {
                "families": rows,
                "version_order": list(version_order),
                "continual_learning": {
                    "family_priorities": [
                        {
                            **priority,
                            "health_observations": list(
                                _family_health_observations(
                                    rows,
                                    str(priority.get("family", "") or ""),
                                )
                            ),
                        }
                        for priority in priorities
                    ]
                },
            }
        ),
    }


def continual_priority_seed(ledger: Mapping[str, Any] | None) -> dict[str, Any]:
    memory = ContinualPriorityMemory()
    memory.ingest_ledger(ledger)
    return {
        "schema_version": "continual-priority-seed-v1",
        "continual_priority_memory": memory.to_state_dict(),
        "summary": memory.summary(),
    }


def continual_priority_features(*, family: str, status: str = "") -> tuple[str, ...]:
    root, backends = _split_family_key(family)
    features: list[str] = []
    if root:
        features.append(f"family:{_token(root)}")
        for part in _split_semantic_parts(root):
            features.append(f"family_token:{_token(part)}")
    for backend in sorted(backends):
        features.append(f"backend:{_token(backend)}")
    if status:
        features.append(f"status:{_token(status)}")
    return tuple(_unique_nonempty(features))


def continual_priority_query_features(
    *,
    scope: str,
    action_id: str,
    context_features: Iterable[Any] = (),
    version_id: str = "",
) -> tuple[str, ...]:
    return _continual_priority_query_features_cached(
        str(scope or ""),
        str(action_id or ""),
        tuple(str(feature) for feature in context_features),
        str(version_id or ""),
    )


def _ledger_priority_rows(ledger: Mapping[str, Any]) -> list[dict[str, Any]]:
    family_health = _ledger_family_health_by_name(ledger)
    continual = ledger.get("continual_learning", {})
    rows = []
    if isinstance(continual, Mapping):
        raw_priorities = continual.get("family_priorities", [])
        if isinstance(raw_priorities, Iterable) and not isinstance(raw_priorities, (str, bytes)):
            for raw_row in raw_priorities:
                if isinstance(raw_row, Mapping):
                    family = str(raw_row.get("family", "") or "").strip()
                    if family:
                        row = {
                            "family": family,
                            "status": str(raw_row.get("status", "") or "unknown"),
                            "priority": _float_value(
                                raw_row.get("priority"),
                                _continual_priority(str(raw_row.get("status", "") or "unknown")),
                            ),
                        }
                        health_observations = _mapping_list(raw_row.get("health_observations", []))
                        if not health_observations:
                            health_observations = family_health.get(family, [])
                        if health_observations:
                            row["health_observations"] = health_observations
                        rows.append(row)
    if rows:
        return rows
    raw_families = ledger.get("families", [])
    if isinstance(raw_families, Iterable) and not isinstance(raw_families, (str, bytes)):
        for raw_row in raw_families:
            if not isinstance(raw_row, Mapping):
                continue
            family = str(raw_row.get("family", "") or "").strip()
            if not family:
                continue
            status = str(raw_row.get("status", "") or "unknown")
            row = {
                "family": family,
                "status": status,
                "priority": _continual_priority(status),
            }
            health_observations = _mapping_list(raw_row.get("health_observations", []))
            if health_observations:
                row["health_observations"] = health_observations
            rows.append(row)
    return rows


def _family_health_observations(
    rows: Iterable[Mapping[str, Any]],
    family: str,
) -> tuple[Mapping[str, Any], ...]:
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        if str(row.get("family", "") or "").strip() != family:
            continue
        return tuple(_mapping_list(row.get("health_observations", [])))
    return ()


def _ledger_family_health_by_name(ledger: Mapping[str, Any]) -> dict[str, list[Mapping[str, Any]]]:
    out: dict[str, list[Mapping[str, Any]]] = {}
    raw_families = ledger.get("families", [])
    if not isinstance(raw_families, Iterable) or isinstance(raw_families, (str, bytes)):
        return out
    for raw_row in raw_families:
        if not isinstance(raw_row, Mapping):
            continue
        family = str(raw_row.get("family", "") or "").strip()
        if not family:
            continue
        health = _mapping_list(raw_row.get("health_observations", []))
        if health:
            out[family] = health
    return out


def _mapping_list(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, Iterable) or isinstance(value, (str, bytes)):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _ledger_family_health_penalty(row: Mapping[str, Any]) -> float:
    observations = _mapping_list(row.get("health_observations", []))
    if not observations:
        return 0.0
    penalties = []
    for observation in observations:
        invalid = max(0.0, _float_value(observation.get("invalid_rate"), 0.0))
        fallback = max(0.0, _float_value(observation.get("fallback_rate"), 0.0))
        false_positive = max(0.0, _float_value(observation.get("false_positive_rate"), 0.0))
        runtime_ms = max(0.0, _float_value(observation.get("runtime_ms_per_case"), 0.0))
        runtime_cost = min(1.0, runtime_ms / 1000.0)
        penalties.append(
            min(
                1.5,
                0.30 * invalid
                + 0.15 * fallback
                + 0.75 * false_positive
                + 0.25 * runtime_cost,
            )
        )
    return max(penalties) if penalties else 0.0


def _split_family_key(family: str) -> tuple[str, set[str]]:
    root, _, backend_part = str(family or "").partition("@")
    backends = {item.strip() for item in backend_part.split(",") if item.strip()}
    return root.strip(), backends


def _split_semantic_parts(value: str) -> list[str]:
    text = _token(value)
    return [part for part in text.split("_") if len(part) >= 3]


def _continual_priority(status: str) -> float:
    return {
        "regression": 1.0,
        "new": 0.9,
        "persistent": 0.45,
        "fixed": 0.20,
        "absent": 0.05,
    }.get(status, 0.10)


def _continual_suggested_action(status: str) -> str:
    return {
        "regression": "prioritize_replay_and_mutation",
        "new": "expand_cluster_and_reduce",
        "persistent": "sample_for_drift_without_saturating",
        "fixed": "retain_low_rate_guardrail",
        "absent": "deprioritize_until_context_changes",
    }.get(status, "observe")


def _bounded_confident_mean_reward(total_reward: float, pulls: float, *, max_abs: float) -> float:
    if pulls <= 0:
        return 0.0
    mean_reward = float(total_reward) / float(pulls)
    bounded_mean = _bounded(mean_reward, max_abs)
    confidence = min(1.0, math.log1p(float(pulls)) / math.log(4.0))
    return bounded_mean * confidence


def _bounded(value: float, max_abs: float) -> float:
    return max(-max_abs, min(max_abs, float(value)))


def _clip_component_scale(
    component: str,
    value: float,
    *,
    min_scale: float,
    max_scale: float,
    health_min_scale: float,
    health_max_scale: float,
) -> float:
    if component == "health_penalty":
        return max(float(health_min_scale), min(float(health_max_scale), float(value)))
    return max(float(min_scale), min(float(max_scale), float(value)))


def _mean(values: Iterable[float]) -> float:
    items = [float(value) for value in values]
    return sum(items) / len(items) if items else 0.0


def _dense_component_dict(values: Iterable[float]) -> dict[str, float]:
    return {
        component: float(value)
        for component, value in zip(_SELF_CALIBRATED_COMPONENTS, values, strict=False)
    }


def _version_key(scope: str, action_id: str, version_id: str) -> str:
    return f"{_token(scope)}|{_token(action_id)}|{_token(version_id)}"


def _scope_action_key(scope: str, action_id: str) -> str:
    return f"{_token(scope)}|{_token(action_id)}"


def _scope_action_prefix(version_key: str) -> str:
    prefix, sep, _ = str(version_key or "").rpartition("|")
    return prefix if sep else ""


def _action_key(scope: str, action_id: str, version_id: str) -> str:
    return f"{_token(scope)}|{_token(action_id)}|{_token(version_id or 'any_version')}"


def _feature_text_from_id(feature_id: int) -> str:
    return resolve_feature_id(feature_id)


def _version_key_id(scope: str, action_id: str, version_id: str) -> int:
    return intern_feature(_version_key(scope, action_id, version_id))


def _scope_action_key_id(scope: str, action_id: str) -> int:
    return intern_feature(_scope_action_key(scope, action_id))


def _action_key_id(scope: str, action_id: str, version_id: str) -> int:
    return intern_feature(_action_key(scope, action_id, version_id))


def _normalize_features(features: Iterable[Any]) -> tuple[str, ...]:
    return _normalize_feature_texts(tuple(str(feature) for feature in features))


@lru_cache(maxsize=32768)
def _feature_ids_from_normalized(normalized: tuple[str, ...]) -> tuple[int, ...]:
    return intern_feature_ids(normalized)


def _feature_token(value: Any) -> str:
    return _cached_feature_token(str(value))


def _token(value: Any) -> str:
    return _cached_token(str(value))


@lru_cache(maxsize=32768)
def _cached_token(value: str) -> str:
    text = str(value).strip().lower()
    chars: list[str] = []
    last_sep = False
    for char in text:
        if char.isalnum():
            chars.append(char)
            last_sep = False
        elif not last_sep:
            chars.append("_")
            last_sep = True
    token = "".join(chars).strip("_")
    return token[:96] if token else "none"


@lru_cache(maxsize=32768)
def _cached_feature_token(value: str) -> str:
    text = str(value).strip()
    if not text:
        return ""
    if ":" in text:
        prefix, _, suffix = text.partition(":")
        return f"{_cached_token(prefix)}:{_cached_token(suffix)}"
    return _cached_token(text)


@lru_cache(maxsize=32768)
def _normalize_feature_texts(raw_features: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(_unique_nonempty(_cached_feature_token(feature) for feature in raw_features))


@lru_cache(maxsize=32768)
def _action_context_features_cached(
    scope: str,
    action_id: str,
    context_features: tuple[str, ...],
    version_present: bool,
) -> tuple[str, ...]:
    features = [f"scope:{_cached_token(scope)}", f"action:{_cached_token(action_id)}"]
    if version_present:
        features.append("version:present")
    features.extend(_normalize_feature_texts(context_features))
    return tuple(_unique_nonempty(features))


@lru_cache(maxsize=32768)
def _continual_priority_query_features_cached(
    scope: str,
    action_id: str,
    context_features: tuple[str, ...],
    version_id: str,
) -> tuple[str, ...]:
    raw_features = context_features + (scope, action_id, version_id)
    out: list[str] = []
    for text in raw_features:
        stripped = str(text or "").strip()
        if not stripped:
            continue
        normalized = _cached_feature_token(stripped)
        out.append(normalized)
        prefix, _, suffix = normalized.partition(":")
        candidates = [suffix] if suffix else [normalized]
        if prefix in {"semantic_family", "semantic_focus_families", "guidance_targets"}:
            candidates.append(suffix.removeprefix("semantic_family_"))
        if prefix in {"backend", "backends", "target_suite"}:
            candidates.append(suffix)
        for candidate in candidates:
            if not candidate:
                continue
            out.append(f"family:{_cached_token(candidate)}")
            out.append(f"backend:{_cached_token(candidate)}")
            for part in _split_semantic_parts(candidate):
                out.append(f"family_token:{_cached_token(part)}")
    return tuple(_unique_nonempty(out))


def _unique_nonempty(values: Iterable[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        out.append(text)
        seen.add(text)
    return out


def _bucket(prefix: str, value: int, limits: list[tuple[int, str]], fallback: str) -> str:
    for limit, name in limits:
        if value <= limit:
            return f"{prefix}:{name}"
    return f"{prefix}:{fallback}"


def _as_nonnegative_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _float_field(data: Mapping[str, Any], key: str, default: float) -> float:
    value = data.get(key, default)
    if value is None:
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _float_value(value: Any, default: float) -> float:
    if value is None:
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _float_counter(raw: Mapping[str, Any]) -> Counter[str]:
    return Counter({str(key): float(value or 0.0) for key, value in raw.items()})


def _int_counter(raw: Mapping[str, Any]) -> Counter[str]:
    return Counter({str(key): int(value or 0) for key, value in raw.items()})
