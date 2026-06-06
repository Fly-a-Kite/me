from __future__ import annotations

from itertools import combinations
from typing import Any

from datadiff.config import ExperimentConfig
from datadiff.decision_engine import (
    choose_adaptive_action as _choose_adaptive_action,
    choose_generator_profile as _choose_generator_profile,
    choose_priority_actions as _choose_priority_actions,
    record_adaptive_action_feedback as _record_adaptive_feedback,
    record_generator_profile_feedback as _record_profile_feedback,
    record_priority_action_feedback as _record_priority_action_feedback,
)
from datadiff.dsl import Case
from datadiff.exploration_objectives import (
    EXPLORATION_OBJECTIVE_PREFIX,
    derive_exploration_objective_features,
    objective_features_for_rules,
)
from datadiff.finding_outcomes import reward_signals_have_rewardable_finding
from datadiff.guidance import derive_case_features
from datadiff.semantic_registry import filter_generator_profiles_by_capability


def _version_pair_id(config: ExperimentConfig) -> str:
    target_version = str(getattr(config, "target_version", "") or "").strip()
    fixed_version = str(getattr(config, "fixed_version", "") or "").strip()
    if target_version and fixed_version:
        return f"{target_version}->{fixed_version}"
    return target_version or fixed_version


def _parse_version_pair(value: str) -> tuple[str, str]:
    text = str(value or "").strip()
    if not text:
        return "", ""
    if "->" in text:
        target, fixed = text.split("->", 1)
        return str(target).strip(), str(fixed).strip()
    return text, ""


def _version_pair_pool(config: ExperimentConfig) -> tuple[tuple[str, ...], dict[str, Any]]:
    configured = [str(item).strip() for item in getattr(config, "version_pair_pool", []) if str(item).strip()]
    default_pair = _version_pair_id(config)
    if default_pair and default_pair not in configured:
        configured.insert(0, default_pair)
    selected = tuple(_unique_nonempty_strings(configured))
    if not selected and default_pair:
        selected = (default_pair,)
    metadata = {
        "requested": list(configured),
        "selected": list(selected),
        "default_pair": default_pair,
    }
    return selected, metadata


def _config_for_version_pair(config: ExperimentConfig, version_pair: str) -> ExperimentConfig:
    if not version_pair:
        return config
    target_version, fixed_version = _parse_version_pair(version_pair)
    if target_version == str(getattr(config, "target_version", "") or "").strip() and fixed_version == str(
        getattr(config, "fixed_version", "") or ""
    ).strip():
        return config
    config_data = config.to_dict()
    config_data["target_version"] = target_version
    config_data["fixed_version"] = fixed_version
    return ExperimentConfig(**config_data)


def _config_payload_for_version_pair(config_payload: dict[str, Any], version_pair: str) -> dict[str, Any]:
    payload = dict(config_payload)
    if not version_pair:
        return payload
    target_version, fixed_version = _parse_version_pair(version_pair)
    payload["target_version"] = target_version
    payload["fixed_version"] = fixed_version
    payload["selected_version_pair"] = version_pair
    return payload


def _case_learning_context_features(
    case: Case,
    config: ExperimentConfig,
    *,
    backends: list[str],
    target_capabilities: list[str] | tuple[str, ...] = (),
    operation_combo: dict[str, Any] | None = None,
    guidance_row: dict[str, Any] | None = None,
) -> tuple[str, ...]:
    features: list[str] = [
        f"oracle_mode:{config.oracle_mode}",
        f"guidance_strategy:{config.guidance_strategy}",
        _bucket_feature("backend_count", len(backends), [(0, "none"), (1, "single"), (3, "few")], "many"),
    ]
    if config.enable_metamorphic_oracle:
        features.append("metamorphic:enabled")
    if config.enable_feedback:
        features.append("feedback:enabled")
    version_pair = _version_pair_id(config)
    if version_pair:
        features.append("version_pair:present")
    for backend in backends[:8]:
        features.append(f"backend:{backend}")
    for capability in target_capabilities[:24]:
        features.append(f"capability:{capability}")
    if isinstance(guidance_row, dict):
        for target in guidance_row.get("matched_targets", []) or []:
            text = str(target).strip()
            if text:
                features.append(f"matched_target:{text}")
        for feature in guidance_row.get("features", []) or []:
            text = str(feature).strip()
            if text:
                features.append(text)
    case_features = derive_case_features(
        case,
        operation_combo=operation_combo,
        exploration_objective_rules=config.exploration_objective_rules,
    )
    features.extend(sorted(case_features))
    return tuple(_unique_nonempty_strings(features))


def _version_pair_context_features(
    *,
    config: ExperimentConfig,
    backends: list[str],
    target_capabilities: list[str] | tuple[str, ...] = (),
) -> tuple[str, ...]:
    features = [
        f"oracle_mode:{config.oracle_mode}",
        f"guidance_strategy:{config.guidance_strategy}",
        _bucket_feature("backend_count", len(backends), [(0, "none"), (1, "single"), (3, "few")], "many"),
    ]
    if config.enable_feedback:
        features.append("feedback:enabled")
    if config.enable_metamorphic_oracle:
        features.append("metamorphic:enabled")
    for backend in backends[:8]:
        features.append(f"backend:{backend}")
    for capability in target_capabilities[:24]:
        features.append(f"capability:{capability}")
    for family in config.semantic_focus_families[:8]:
        features.append(f"semantic_family:{family}")
    for signal in config.semantic_focus_signals[:8]:
        features.append(f"semantic_signal:{signal}")
    return tuple(_unique_nonempty_strings(features))


def _backend_pair_pool(backends: list[str]) -> tuple[str, ...]:
    normalized = sorted(_unique_nonempty_strings(str(backend).strip() for backend in backends))
    return tuple(f"{left}|{right}" for left, right in combinations(normalized, 2))


def _backend_pair_context_features(
    *,
    case_learning_context: tuple[str, ...] = (),
    case_fingerprint: dict[str, Any] | None = None,
    disagreement_descriptor: dict[str, Any] | None = None,
    selected_version_pair: str = "",
    target_capabilities: list[str] | tuple[str, ...] = (),
) -> tuple[str, ...]:
    features: list[str] = []
    features.extend(str(feature) for feature in case_learning_context[:32])
    if selected_version_pair:
        features.append(f"version_pair:{selected_version_pair}")
    if isinstance(case_fingerprint, dict):
        for token in case_fingerprint.get("feature_tokens", []) or []:
            text = str(token).strip()
            if text:
                features.append(text)
    if isinstance(disagreement_descriptor, dict):
        for token in disagreement_descriptor.get("feature_tokens", []) or []:
            text = str(token).strip()
            if text and not text.startswith("disagree_pair:"):
                features.append(text)
        mismatch = str(disagreement_descriptor.get("mismatch_class", "") or "").strip()
        if mismatch:
            features.append(f"mismatch:{mismatch}")
        root = str(disagreement_descriptor.get("primary_root_cause", "") or "").strip()
        if root:
            features.append(f"root:{root}")
    for capability in target_capabilities[:24]:
        features.append(f"capability:{capability}")
    return tuple(_unique_nonempty_strings(features))


def _rank_backend_pairs(
    feedback: Any,
    *,
    pair_pool: tuple[str, ...],
    context_features: tuple[str, ...],
    version_id: str,
    learning_weight: float,
    enabled: bool,
    limit: int,
) -> dict[str, Any]:
    decision = _choose_priority_actions(
        feedback,
        scope="backend_pair",
        action_pool=pair_pool,
        context_features=context_features,
        version_id=version_id,
        learning_weight=learning_weight,
        enabled=enabled,
        limit=limit,
    )
    selection = decision.to_selection_dict()
    selection["context_feature_count"] = len(context_features)
    return selection


def _pair_disagrees_from_descriptor(descriptor: Any) -> dict[str, bool]:
    if not isinstance(descriptor, dict):
        return {}
    raw_pairs = descriptor.get("pair_disagrees", [])
    out: dict[str, bool] = {}
    if isinstance(raw_pairs, dict):
        for key, value in raw_pairs.items():
            pair_id = _normalize_backend_pair_id(str(key))
            if pair_id:
                out[pair_id] = bool(value)
        return out
    if not isinstance(raw_pairs, list):
        return {}
    for item in raw_pairs:
        if not isinstance(item, dict):
            continue
        pair_id = _normalize_backend_pair_id(f"{item.get('left', '')}|{item.get('right', '')}")
        if pair_id:
            out[pair_id] = bool(item.get("disagrees"))
    return out


def _normalize_backend_pair_id(pair_id: str) -> str:
    parts = [part.strip() for part in str(pair_id or "").split("|", 1)]
    if len(parts) != 2 or not parts[0] or not parts[1] or parts[0] == parts[1]:
        return ""
    left, right = sorted(parts)
    return f"{left}|{right}"


def _backend_pair_reward(
    pair_id: str,
    *,
    disagrees: bool,
    reward_signals: dict[str, Any],
    priority_pairs: set[str],
) -> float:
    if disagrees:
        reward = 1.25
        if bool(reward_signals.get("candidate_bug", False)):
            reward += 1.75
        if bool(reward_signals.get("rewardable_semantic_divergence", False)):
            reward += 0.75
        if pair_id in priority_pairs:
            reward += 0.25
        if bool(reward_signals.get("false_positive", False)):
            reward -= 1.0
        return reward
    if bool(reward_signals.get("false_positive", False)):
        return -0.45
    return -0.02 if pair_id in priority_pairs else 0.0


def _record_backend_pair_feedback(
    feedback: Any,
    row: dict[str, Any],
    *,
    context_features: tuple[str, ...],
    version_id: str,
    reward_signals: dict[str, Any],
) -> dict[str, Any]:
    selection = row.get("backend_pair_selection", {})
    if feedback is None or not isinstance(selection, dict):
        return {"recorded": 0, "rewarded_pairs": [], "disagree_pairs": []}
    strategy = str(selection.get("strategy", "") or "")
    if strategy not in {"contextual_bandit", "contextual_bandit_warmup"}:
        return {"recorded": 0, "rewarded_pairs": [], "disagree_pairs": []}
    pair_pool = [
        pair_id
        for pair_id in (
            _normalize_backend_pair_id(str(item))
            for item in selection.get("action_pool", []) or []
        )
        if pair_id
    ]
    if not pair_pool:
        return {"recorded": 0, "rewarded_pairs": [], "disagree_pairs": []}
    pair_disagrees = _pair_disagrees_from_descriptor(row.get("disagreement_descriptor"))
    priority_pairs = {
        pair_id
        for pair_id in (
            _normalize_backend_pair_id(str(item))
            for item in selection.get("priority", []) or []
        )
        if pair_id
    }
    preflight = row.get("preflight", {}) if isinstance(row.get("preflight", {}), dict) else {}
    recorded = 0
    rewarded_pairs: list[dict[str, Any]] = []
    disagree_pairs: list[str] = []
    for pair_id in pair_pool:
        disagrees = bool(pair_disagrees.get(pair_id, False))
        reward = _backend_pair_reward(
            pair_id,
            disagrees=disagrees,
            reward_signals=reward_signals,
            priority_pairs=priority_pairs,
        )
        recorded_reward = _record_priority_action_feedback(
            feedback,
            selection,
            pair_id,
            context_features=context_features,
            version_id=version_id,
            reward=reward,
            runtime_cost=_runtime_cost_signal(row) / max(1, len(pair_pool)),
            preflight_valid=bool(preflight.get("valid", True)),
            fallback_used=bool(preflight.get("fallback_used", False)),
            false_positive=bool(reward_signals.get("false_positive", False)),
        )
        if recorded_reward is None:
            continue
        recorded += 1
        if disagrees:
            disagree_pairs.append(pair_id)
            rewarded_pairs.append({"pair": pair_id, "reward": reward})
    return {
        "recorded": recorded,
        "rewarded_pairs": rewarded_pairs,
        "disagree_pairs": disagree_pairs,
    }


def _semantic_objective_pool(
    case_features: tuple[str, ...],
    config: ExperimentConfig,
    guidance_row: dict[str, Any],
) -> tuple[str, ...]:
    candidates: list[str] = []
    for feature in case_features:
        text = str(feature).strip()
        if text.startswith(EXPLORATION_OBJECTIVE_PREFIX):
            candidates.append(text)
    for target in guidance_row.get("matched_targets", []) or []:
        text = str(target).strip()
        if text.startswith(EXPLORATION_OBJECTIVE_PREFIX):
            candidates.append(text)
    candidates.extend(objective_features_for_rules(config.exploration_objective_rules))
    candidates.extend(derive_exploration_objective_features(case_features, rules=config.exploration_objective_rules))
    return tuple(_unique_nonempty_strings(candidates))


def _select_adaptive_action(
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
    return _choose_adaptive_action(
        feedback,
        scope=scope,
        action_pool=action_pool,
        context_features=context_features,
        version_id=version_id,
        learning_weight=learning_weight,
        enabled=enabled,
        fixed_strategy=fixed_strategy,
    )


def _record_adaptive_action_feedback(
    feedback: Any,
    selection: dict[str, Any],
    row: dict[str, Any],
    *,
    context_features: tuple[str, ...],
    version_id: str,
    reward_signals: dict[str, Any],
) -> float | None:
    reward = _adaptive_exploration_reward(row, reward_signals=reward_signals)
    preflight = row.get("preflight", {}) if isinstance(row.get("preflight", {}), dict) else {}
    return _record_adaptive_feedback(
        feedback,
        selection,
        context_features=context_features,
        version_id=version_id,
        reward=reward,
        runtime_cost=_runtime_cost_signal(row),
        preflight_valid=bool(preflight.get("valid", True)),
        fallback_used=bool(preflight.get("fallback_used", False)),
        false_positive=bool(reward_signals.get("false_positive", False)),
    )


def _adaptive_exploration_reward(row: dict[str, Any], *, reward_signals: dict[str, Any]) -> float:
    return _generator_profile_reward(row, reward_signals=reward_signals)


def _runtime_cost_signal(row: dict[str, Any]) -> float:
    duration_ms = float(row.get("duration_ms", 0.0) or 0.0)
    return min(2.0, max(0.0, duration_ms / 1000.0))


def _metamorphic_relation_order_from_selection(
    selected_relation: str,
    *,
    configured_order: list[str] | tuple[str, ...],
    relation_pool: tuple[str, ...] | list[str] = (),
    rotation_seed: int | None = None,
    selection_strategy: str = "",
) -> list[str]:
    configured = _unique_nonempty_strings(configured_order)
    pool = _unique_nonempty_strings(relation_pool)
    selected = str(selected_relation or "").strip()
    strategy = str(selection_strategy or "").strip()

    if strategy in {"fixed", "fixed_no_learning_state"}:
        if configured:
            return _unique_nonempty_strings([*configured, *pool])
        rotated = _rotated_strings(pool, rotation_seed)
        if rotated:
            return rotated

    ordered = []
    if selected:
        ordered.append(selected)
    ordered.extend(configured)
    return _unique_nonempty_strings(ordered)


def _rotated_strings(values: list[str], rotation_seed: int | None) -> list[str]:
    if not values:
        return []
    try:
        offset = int(rotation_seed or 0) % len(values)
    except (TypeError, ValueError):
        offset = 0
    return values[offset:] + values[:offset]


def _effective_generator_profile(config: ExperimentConfig) -> str:
    return config.generator_profile


def _generator_profile_pool(
    config: ExperimentConfig,
    *,
    target_capabilities: list[str] | tuple[str, ...] = (),
) -> tuple[tuple[str, ...], dict[str, Any]]:
    configured = [str(item).strip() for item in config.generator_profile_pool if str(item).strip()]
    if not configured:
        configured = [str(config.generator_profile or "common").strip()]
    elif str(config.generator_profile or "").strip() not in configured:
        configured.insert(0, str(config.generator_profile or "common").strip())
    out: list[str] = []
    seen: set[str] = set()
    for profile in configured:
        if not profile or profile in seen:
            continue
        out.append(profile)
        seen.add(profile)
    original = out or ["common"]
    filter_result = filter_generator_profiles_by_capability(original, target_capabilities)
    if not config.enable_profile_capability_filter:
        filter_result = {
            "selected": list(original),
            "dropped": [],
            "target_capability_count": len(tuple(target_capabilities)),
        }
    selected = list(filter_result.get("selected", []) or [])
    if not selected:
        selected = [str(config.generator_profile or "common").strip() or "common"]
    metadata = {
        "requested": list(original),
        "selected": selected,
        "dropped": list(filter_result.get("dropped", []) or []),
        "target_capability_count": int(filter_result.get("target_capability_count", 0) or 0),
        "capability_aware": bool(target_capabilities) and config.enable_profile_capability_filter,
        "capability_filter_enabled": config.enable_profile_capability_filter,
    }
    return tuple(selected), metadata


def _generator_profile_context_features(
    *,
    config: ExperimentConfig,
    backends: list[str],
    target_specs: list[dict[str, Any]],
    target_capabilities: list[str],
    candidate_pool: int,
) -> tuple[str, ...]:
    features = [
        f"oracle_mode:{config.oracle_mode}",
        f"guidance_strategy:{config.guidance_strategy}",
        _bucket_feature("candidate_pool", candidate_pool, [(1, "single"), (4, "small"), (8, "medium")], "large"),
        _bucket_feature("backend_count", len(backends), [(0, "none"), (1, "single"), (3, "few")], "many"),
    ]
    if config.enable_metamorphic_oracle:
        features.append("metamorphic:enabled")
    if config.enable_feedback:
        features.append("feedback:enabled")
    for backend in backends[:8]:
        features.append(f"backend:{backend}")
    for target in target_specs[:8]:
        if isinstance(target, dict):
            family = str(target.get("family", "") or "").strip()
            layer = str(target.get("layer", "") or "").strip()
            if family:
                features.append(f"target_family:{family}")
            if layer:
                features.append(f"target_layer:{layer}")
    for capability in target_capabilities[:24]:
        features.append(f"capability:{capability}")
    for family in config.semantic_focus_families[:8]:
        features.append(f"semantic_family:{family}")
    for signal in config.semantic_focus_signals[:8]:
        features.append(f"semantic_signal:{signal}")
    for target in config.guidance_targets[:8]:
        features.append(f"guidance_target:{target}")
    return tuple(_unique_nonempty_strings(features))


def _select_generator_profile(
    feedback: Any,
    profile_pool: tuple[str, ...],
    *,
    context_features: tuple[str, ...],
    learning_weight: float,
    pool_metadata: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    return _choose_generator_profile(
        feedback,
        profile_pool,
        context_features=context_features,
        learning_weight=learning_weight,
        pool_metadata=pool_metadata,
    )


def _record_generator_profile_feedback(
    feedback: Any,
    selected_meta: dict[str, Any],
    row: dict[str, Any],
    *,
    context_features: tuple[str, ...],
    reward_signals: dict[str, Any],
) -> float | None:
    profile_selection = selected_meta.get("generator_profile_selection", {})
    reward = _generator_profile_reward(row, reward_signals=reward_signals)
    preflight = row.get("preflight", {}) if isinstance(row.get("preflight", {}), dict) else {}
    return _record_profile_feedback(
        feedback,
        profile_selection,
        context_features=context_features,
        reward=reward,
        preflight_valid=bool(preflight.get("valid", True)),
        fallback_used=bool(preflight.get("fallback_used", False)),
        false_positive=bool(reward_signals.get("false_positive", False)),
    )


def _generator_profile_reward(row: dict[str, Any], *, reward_signals: dict[str, Any]) -> float:
    candidate_bug = bool(reward_signals.get("candidate_bug", False))
    rewardable_semantic = bool(reward_signals.get("rewardable_semantic_divergence", False))
    false_positive = bool(reward_signals.get("false_positive", False))
    rewardable_finding_count = int(reward_signals.get("candidate_bug_count", 0) or 0) + int(
        reward_signals.get("semantic_divergence_needs_confirmation_count", 0) or 0
    )
    new_behavior = _reward_signals_have_rewardable_new_behavior(row, reward_signals=reward_signals)
    preflight = row.get("preflight", {}) if isinstance(row.get("preflight", {}), dict) else {}
    throughput_hint = 0.0
    duration_ms = float(row.get("duration_ms", 0.0) or 0.0)
    if duration_ms > 0.0:
        throughput_hint = min(0.25, 1.0 / duration_ms)
    return (
        (3.0 if candidate_bug else 0.0)
        + (0.7 if rewardable_semantic else 0.0)
        + (0.45 if new_behavior else 0.0)
        + min(0.4, 0.05 * rewardable_finding_count)
        + throughput_hint
        - (2.5 if false_positive else 0.0)
        - (0.5 if not bool(preflight.get("valid", True)) else 0.0)
        - (0.25 if bool(preflight.get("fallback_used", False)) else 0.0)
    )


def _reward_signals_have_rewardable_new_behavior(
    row: dict[str, Any],
    *,
    reward_signals: dict[str, Any],
) -> bool:
    if not bool(row.get("signal_new_behavior", row.get("is_new_behavior", False))):
        return False
    if not row.get("findings"):
        return True
    return reward_signals_have_rewardable_finding(reward_signals)


def _bucket_feature(prefix: str, value: int, limits: list[tuple[int, str]], fallback: str) -> str:
    for limit, name in limits:
        if value <= limit:
            return f"{prefix}:{name}"
    return f"{prefix}:{fallback}"


def _unique_nonempty_strings(values: Any) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value).strip()
        if not item or item in seen:
            continue
        out.append(item)
        seen.add(item)
    return out
