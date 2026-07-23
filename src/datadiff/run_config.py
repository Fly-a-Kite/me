from __future__ import annotations

from typing import Any

from datadiff.config import ExperimentConfig
from datadiff.experiment_manifest import finalize_config_payload
from datadiff.exploration_objectives import objective_features_for_rules
from datadiff.semantic_signal import canonical_target_key, semantic_signal_feature


def _effective_guidance_targets(config: ExperimentConfig) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for text in _configured_guidance_targets(config):
        if text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _configured_guidance_targets(config: ExperimentConfig) -> list[str]:
    configured: list[str] = []
    seen: set[str] = set()

    def _append(value: Any) -> None:
        text = str(value).strip()
        if not text or text in seen:
            return
        seen.add(text)
        configured.append(text)

    for value in config.guidance_targets or []:
        _append(value)
    for value in config.semantic_focus_families or []:
        family = str(value).strip()
        if not family:
            continue
        _append(f"semantic_family:{family}")
    for value in config.semantic_focus_signals or []:
        signal = str(value).strip()
        if not signal:
            continue
        semantic_target = str(canonical_target_key(semantic_signal_feature(signal))).strip()
        if semantic_target:
            _append(semantic_target)
    for objective in objective_features_for_rules(config.exploration_objective_rules):
        _append(objective)
    return configured


def _config_payload_with_effective_guidance_targets(config: ExperimentConfig) -> dict[str, Any]:
    payload = config.to_dict()
    payload["effective_guidance_targets"] = _configured_guidance_targets(config)
    payload["method_arm_manifest"] = config.method_arm_manifest
    return finalize_config_payload(payload)


def _config_layer_payload(config: ExperimentConfig) -> dict[str, Any]:
    payload = config.to_nested_dict()
    payload["guidance"]["effective_targets"] = _configured_guidance_targets(config)
    flat_payload = _config_payload_with_effective_guidance_targets(config)
    payload["config_digest"] = flat_payload["config_digest"]
    return payload
