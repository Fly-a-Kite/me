from __future__ import annotations

from typing import Any

from datadiff.exploration_objectives import (
    EXPLORATION_OBJECTIVE_PREFIX,
    canonical_objective_key,
)

CANONICAL_SEMANTIC_SIGNAL_PREFIX = "semantic_signal:"
LEGACY_COMBO_RISK_PREFIX = "combo_risk:"
LEGACY_RISK_PREFIX = "risk:"
SEMANTIC_SIGNAL_PREFIX_ALIASES = (
    CANONICAL_SEMANTIC_SIGNAL_PREFIX,
    LEGACY_COMBO_RISK_PREFIX,
)


def semantic_signal_feature(signal: str) -> str:
    return f"{CANONICAL_SEMANTIC_SIGNAL_PREFIX}{signal}"


def semantic_signal_aliases(*signals: str) -> set[str]:
    aliases: set[str] = set()
    for signal in signals:
        text = str(signal).strip()
        if not text:
            continue
        aliases.add(semantic_signal_feature(text))
        aliases.add(f"{LEGACY_COMBO_RISK_PREFIX}{text}")
    return aliases


def semantic_signal_feature_bundle(*signals: str, extra_features: set[str] | None = None) -> set[str]:
    bundle = semantic_signal_aliases(*signals)
    if extra_features:
        bundle.update(str(feature).strip() for feature in extra_features if str(feature).strip())
    return bundle


def canonical_semantic_feature(feature: str) -> str:
    text = str(feature).strip()
    objective_key = canonical_objective_key(text)
    if objective_key.startswith(EXPLORATION_OBJECTIVE_PREFIX):
        return objective_key
    if text.startswith(LEGACY_COMBO_RISK_PREFIX):
        signal = text.removeprefix(LEGACY_COMBO_RISK_PREFIX).strip()
        if signal:
            return semantic_signal_feature(signal)
    return text


def semantic_feature_aliases(feature: str) -> set[str]:
    canonical = canonical_semantic_feature(feature)
    aliases = {canonical}
    if canonical.startswith(CANONICAL_SEMANTIC_SIGNAL_PREFIX):
        signal = canonical.removeprefix(CANONICAL_SEMANTIC_SIGNAL_PREFIX).strip()
        if signal:
            aliases.add(f"{LEGACY_COMBO_RISK_PREFIX}{signal}")
    return aliases


def alias_expanded_semantic_features(features: set[str]) -> set[str]:
    expanded: set[str] = set()
    for feature in features:
        expanded.update(semantic_feature_aliases(feature))
    return expanded


def canonical_semantic_prefix(prefix: str) -> str:
    text = str(prefix).strip()
    if text == LEGACY_COMBO_RISK_PREFIX.removesuffix(":"):
        return CANONICAL_SEMANTIC_SIGNAL_PREFIX.removesuffix(":")
    return text


def feature_matches_prefix_alias(feature: str, prefix: str) -> bool:
    text = str(feature).strip()
    normalized_prefix = str(prefix).strip()
    if text.startswith(normalized_prefix):
        return True
    if normalized_prefix == LEGACY_COMBO_RISK_PREFIX and text.startswith(CANONICAL_SEMANTIC_SIGNAL_PREFIX):
        return True
    if normalized_prefix == CANONICAL_SEMANTIC_SIGNAL_PREFIX and text.startswith(LEGACY_COMBO_RISK_PREFIX):
        return True
    return False


def canonical_target_key(value: Any) -> str:
    text = str(value).strip()
    objective_key = canonical_objective_key(text)
    if objective_key.startswith(EXPLORATION_OBJECTIVE_PREFIX):
        return objective_key
    if text.startswith(LEGACY_RISK_PREFIX):
        signal = text.removeprefix(LEGACY_RISK_PREFIX).strip()
        if signal:
            return semantic_signal_feature(signal)
    return canonical_semantic_feature(text)


def legacy_target_key_alias(target_key: str) -> str:
    normalized = canonical_target_key(target_key)
    if normalized.startswith(CANONICAL_SEMANTIC_SIGNAL_PREFIX):
        signal = normalized.removeprefix(CANONICAL_SEMANTIC_SIGNAL_PREFIX).strip()
        if signal:
            return f"{LEGACY_RISK_PREFIX}{signal}"
    return ""


def target_key_weight(target_key: str) -> float:
    canonical = canonical_target_key(target_key)
    if canonical.startswith(EXPLORATION_OBJECTIVE_PREFIX):
        return 1.15
    if canonical.startswith("semantic_family:"):
        return 1.20
    if canonical.startswith(CANONICAL_SEMANTIC_SIGNAL_PREFIX):
        return 1.25
    if canonical.startswith("combo:"):
        return 1.10
    if canonical.startswith("target:"):
        return 1.00
    if canonical.startswith("feature:semantic_family:"):
        return 0.95
    if canonical.startswith("feature:pattern:"):
        return 0.90
    if canonical.startswith("feature:source:"):
        return 0.60
    if canonical.startswith("profile:"):
        return 0.40
    return 0.50


def semantic_signal_bias_prefixes(*prefixes: str) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for prefix in (*prefixes, CANONICAL_SEMANTIC_SIGNAL_PREFIX, LEGACY_COMBO_RISK_PREFIX):
        text = str(prefix).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return ordered
