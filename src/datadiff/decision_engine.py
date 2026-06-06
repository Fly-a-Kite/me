from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

_CONTEXTUAL_BANDIT_STRATEGIES = frozenset(
    {
        "contextual_bandit",
        "contextual_bandit_warmup",
    }
)


@dataclass(frozen=True, slots=True)
class AdaptiveDecision:
    strategy: str
    action: str
    action_pool: tuple[str, ...]
    learning_weight: float
    ranked: tuple[dict[str, Any], ...]
    scope: str = ""

    def to_selection_dict(
        self,
        *,
        action_key: str,
        pool_key: str,
        include_scope: bool = True,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        selection: dict[str, Any] = {"strategy": self.strategy}
        if include_scope:
            selection["scope"] = self.scope
        selection[action_key] = self.action
        selection[pool_key] = list(self.action_pool)
        if extra:
            selection.update(dict(extra))
        selection["learning_weight"] = float(self.learning_weight)
        selection["ranked"] = list(self.ranked)
        return selection


@dataclass(frozen=True, slots=True)
class AdaptivePriorityDecision:
    strategy: str
    priority: tuple[str, ...]
    action_pool: tuple[str, ...]
    learning_weight: float
    ranked: tuple[dict[str, Any], ...]
    scope: str

    def to_selection_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "scope": self.scope,
            "priority": list(self.priority),
            "action_pool": list(self.action_pool),
            "learning_weight": float(self.learning_weight),
            "ranked": list(self.ranked),
        }


def choose_adaptive_action(
    feedback: Any,
    *,
    scope: str,
    action_pool: tuple[str, ...],
    context_features: tuple[str, ...],
    version_id: str,
    learning_weight: float,
    enabled: bool,
    fixed_strategy: str = "fixed",
) -> tuple[str, dict[str, Any]]:
    decision = _choose_decision(
        feedback,
        scope=scope,
        action_pool=action_pool,
        context_features=context_features,
        version_id=version_id,
        learning_weight=learning_weight,
        enabled=enabled,
        fixed_strategy=fixed_strategy,
    )
    return decision.action, decision.to_selection_dict(
        action_key="action",
        pool_key="action_pool",
        include_scope=True,
    )


def choose_priority_actions(
    feedback: Any,
    *,
    scope: str,
    action_pool: tuple[str, ...],
    context_features: tuple[str, ...],
    version_id: str,
    learning_weight: float,
    enabled: bool,
    limit: int,
    fixed_strategy: str = "fixed",
) -> AdaptivePriorityDecision:
    bounded_limit = max(0, min(int(limit or 0), len(action_pool)))
    if not action_pool or bounded_limit <= 0:
        return AdaptivePriorityDecision(
            strategy="none",
            scope=scope,
            priority=(),
            action_pool=tuple(action_pool),
            learning_weight=0.0,
            ranked=(),
        )
    if not enabled or feedback is None or learning_weight <= 0.0:
        return AdaptivePriorityDecision(
            strategy=fixed_strategy,
            scope=scope,
            priority=tuple(action_pool[:bounded_limit]),
            action_pool=tuple(action_pool),
            learning_weight=0.0,
            ranked=(),
        )
    learning = getattr(feedback, "adaptive_learning", None)
    if learning is None:
        return AdaptivePriorityDecision(
            strategy="fixed_no_learning_state",
            scope=scope,
            priority=tuple(action_pool[:bounded_limit]),
            action_pool=tuple(action_pool),
            learning_weight=0.0,
            ranked=(),
        )
    bandit = learning.bandits.get(scope)
    ranked = tuple(
        learning.rank_top(
            scope,
            action_pool,
            limit=max(bounded_limit, min(8, len(action_pool))),
            context_features=context_features,
            version_id=version_id,
        )
    )
    warmed = bool(
        bandit is not None
        and all(getattr(bandit.arms.get(action), "pulls", 0) > 0 for action in action_pool)
    )
    if warmed:
        priority = _priority_from_ranked(ranked, bounded_limit)
    else:
        arms = getattr(bandit, "arms", {}) if bandit is not None else {}
        priority = tuple(
            action
            for action in action_pool
            if getattr(arms.get(action), "pulls", 0) <= 0
        )[:bounded_limit]
    if len(priority) < bounded_limit:
        priority = _fill_priority(priority, action_pool, bounded_limit)
    return AdaptivePriorityDecision(
        strategy="contextual_bandit" if warmed else "contextual_bandit_warmup",
        scope=scope,
        priority=priority,
        action_pool=tuple(action_pool),
        learning_weight=float(learning_weight),
        ranked=ranked,
    )


def record_priority_action_feedback(
    feedback: Any,
    selection: Mapping[str, Any] | None,
    action: str,
    *,
    context_features: Iterable[Any] = (),
    version_id: str = "",
    reward: float,
    runtime_cost: float = 0.0,
    preflight_valid: bool = True,
    fallback_used: bool = False,
    false_positive: bool = False,
) -> float | None:
    return _record_decision_feedback(
        feedback,
        selection,
        scope="",
        action_key="",
        action_override=action,
        context_features=context_features,
        version_id=version_id,
        reward=reward,
        runtime_cost=runtime_cost,
        preflight_valid=preflight_valid,
        fallback_used=fallback_used,
        false_positive=false_positive,
    )


def choose_generator_profile(
    feedback: Any,
    profile_pool: tuple[str, ...],
    *,
    context_features: tuple[str, ...],
    learning_weight: float,
    pool_metadata: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    if not profile_pool:
        decision = AdaptiveDecision(
            strategy="fixed",
            scope="generator_profile",
            action="common",
            action_pool=(),
            learning_weight=0.0,
            ranked=(),
        )
        return decision.action, decision.to_selection_dict(
            action_key="profile",
            pool_key="profile_pool",
            include_scope=False,
            extra={"profile_pool_metadata": dict(pool_metadata or {})},
        )
    decision = _choose_decision(
        feedback,
        scope="generator_profile",
        action_pool=profile_pool,
        context_features=context_features,
        version_id="",
        learning_weight=learning_weight,
        enabled=True,
        fixed_strategy="fixed",
        fixed_when_pool_size_at_most=1,
    )
    return decision.action, decision.to_selection_dict(
        action_key="profile",
        pool_key="profile_pool",
        include_scope=False,
        extra={"profile_pool_metadata": dict(pool_metadata or {})},
    )


def record_adaptive_action_feedback(
    feedback: Any,
    selection: Mapping[str, Any] | None,
    *,
    context_features: Iterable[Any] = (),
    version_id: str = "",
    reward: float,
    runtime_cost: float = 0.0,
    preflight_valid: bool = True,
    fallback_used: bool = False,
    false_positive: bool = False,
) -> float | None:
    return _record_decision_feedback(
        feedback,
        selection,
        scope="",
        action_key="action",
        context_features=context_features,
        version_id=version_id,
        reward=reward,
        runtime_cost=runtime_cost,
        preflight_valid=preflight_valid,
        fallback_used=fallback_used,
        false_positive=false_positive,
    )


def record_generator_profile_feedback(
    feedback: Any,
    selection: Mapping[str, Any] | None,
    *,
    context_features: Iterable[Any] = (),
    reward: float,
    preflight_valid: bool = True,
    fallback_used: bool = False,
    false_positive: bool = False,
) -> float | None:
    return _record_decision_feedback(
        feedback,
        selection,
        scope="generator_profile",
        action_key="profile",
        context_features=context_features,
        version_id="",
        reward=reward,
        runtime_cost=0.0,
        preflight_valid=preflight_valid,
        fallback_used=fallback_used,
        false_positive=false_positive,
    )


def _choose_decision(
    feedback: Any,
    *,
    scope: str,
    action_pool: tuple[str, ...],
    context_features: tuple[str, ...],
    version_id: str,
    learning_weight: float,
    enabled: bool,
    fixed_strategy: str,
    fixed_when_pool_size_at_most: int = 0,
) -> AdaptiveDecision:
    if not action_pool:
        return AdaptiveDecision(
            strategy="none",
            scope=scope,
            action="",
            action_pool=(),
            learning_weight=0.0,
            ranked=(),
        )
    if len(action_pool) <= fixed_when_pool_size_at_most or not enabled or feedback is None or learning_weight <= 0.0:
        return AdaptiveDecision(
            strategy=fixed_strategy,
            scope=scope,
            action=action_pool[0],
            action_pool=tuple(action_pool),
            learning_weight=0.0,
            ranked=(),
        )
    learning = getattr(feedback, "adaptive_learning", None)
    if learning is None:
        return AdaptiveDecision(
            strategy="fixed_no_learning_state",
            scope=scope,
            action=action_pool[0],
            action_pool=tuple(action_pool),
            learning_weight=0.0,
            ranked=(),
        )
    bandit = learning.bandits.get(scope)
    if bandit is not None:
        for action in action_pool:
            arm = bandit.arms.get(action)
            if arm is None or arm.pulls <= 0:
                return AdaptiveDecision(
                    strategy="contextual_bandit_warmup",
                    scope=scope,
                    action=action,
                    action_pool=tuple(action_pool),
                    learning_weight=float(learning_weight),
                    ranked=tuple(
                        learning.rank_top(
                            scope,
                            action_pool,
                            limit=8,
                            context_features=context_features,
                            version_id=version_id,
                        )
                    ),
                )
    else:
        return AdaptiveDecision(
            strategy="contextual_bandit_warmup",
            scope=scope,
            action=action_pool[0],
            action_pool=tuple(action_pool),
            learning_weight=float(learning_weight),
            ranked=(),
        )
    best = learning.choose_dense(
        scope,
        action_pool,
        context_features=context_features,
        version_id=version_id,
    )
    action = str(best.action_id) if best is not None else action_pool[0]
    ranked = learning.rank_top(
        scope,
        action_pool,
        limit=8,
        context_features=context_features,
        version_id=version_id,
    )
    return AdaptiveDecision(
        strategy="contextual_bandit",
        scope=scope,
        action=action,
        action_pool=tuple(action_pool),
        learning_weight=float(learning_weight),
        ranked=tuple(ranked),
    )


def _record_decision_feedback(
    feedback: Any,
    selection: Mapping[str, Any] | None,
    *,
    scope: str,
    action_key: str,
    action_override: str = "",
    context_features: Iterable[Any],
    version_id: str,
    reward: float,
    runtime_cost: float,
    preflight_valid: bool,
    fallback_used: bool,
    false_positive: bool,
) -> float | None:
    learning = getattr(feedback, "adaptive_learning", None) if feedback is not None else None
    if learning is None or not isinstance(selection, Mapping):
        return None
    strategy = str(selection.get("strategy", "") or "")
    if strategy not in _CONTEXTUAL_BANDIT_STRATEGIES:
        return None
    resolved_scope = str(selection.get("scope", scope) or scope).strip()
    action = str(action_override or (selection.get(action_key, "") if action_key else "") or "").strip()
    if not resolved_scope or not action:
        return None
    learning.record_outcome(
        resolved_scope,
        action,
        context_features=context_features,
        version_id=version_id,
        reward=reward,
        runtime_cost=runtime_cost,
        preflight_valid=preflight_valid,
        fallback_used=fallback_used,
        false_positive=false_positive,
    )
    return reward


def _priority_from_ranked(ranked: tuple[dict[str, Any], ...], limit: int) -> tuple[str, ...]:
    priority: list[str] = []
    for row in ranked:
        if not isinstance(row, Mapping):
            continue
        action = str(row.get("action_id", "") or "").strip()
        if action and action not in priority:
            priority.append(action)
            if len(priority) >= limit:
                break
    return tuple(priority)


def _fill_priority(
    priority: tuple[str, ...],
    action_pool: tuple[str, ...],
    limit: int,
) -> tuple[str, ...]:
    out = list(priority)
    for action in action_pool:
        if action in out:
            continue
        out.append(action)
        if len(out) >= limit:
            break
    return tuple(out)
