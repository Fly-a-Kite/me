from __future__ import annotations

from typing import Any

from datadiff.config import ExperimentConfig
from datadiff.dsl import Case
from datadiff.exploration_objectives import EXPLORATION_OBJECTIVE_PREFIX
from datadiff.operation_combo import combo_semantic_signals, describe_operation_combo
from datadiff.semantic_signal import legacy_target_key_alias, semantic_signal_feature


def _selected_candidate_metadata(
    case: Case,
    config: ExperimentConfig,
    guidance_enabled: bool,
) -> dict[str, Any]:
    generated_metadata = _generated_candidate_metadata(case)
    case_metadata = case.metadata if isinstance(case.metadata, dict) else {}
    return {
        "source": "generated",
        "generated_seed": case.seed,
        "seed_lineage": generated_metadata["seed_lineage"],
        "mutation": generated_metadata["mutation"],
        "feedback_decision": {},
        "quality_archive_context": {},
        "goal_first_generation": dict(
            case_metadata.get("goal_first_generation", {}) or {}
        ),
        "semantic_activation": dict(
            case_metadata.get("semantic_activation", {}) or {}
        ),
        "boundary_application": dict(
            case_metadata.get("boundary_application", {}) or {}
        ),
        "operation_combo": describe_operation_combo(case.program.operations),
        "preflight": {
            "valid": True,
            "repaired": False,
            "fallback_used": False,
            "errors_before": [],
            "errors_after": [],
        },
        "replay_filter": {
            "enabled": not config.enable_replay_bug,
            "filtered_before_candidate": 0,
            "fallback_used": False,
            "last_skip_reason": "",
        },
        "family_saturation_filter": {
            "enabled": guidance_enabled,
            "filtered_before_candidate": 0,
            "fallback_used": False,
            "last_skip_reason": "",
        },
    }


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
        "feedback_decision": {},
    }


def _candidate_target_keys_from_metadata(metadata: dict[str, Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    if not isinstance(metadata, dict):
        return out
    for section_name in ("feedback_decision",):
        section = metadata.get(section_name)
        if not isinstance(section, dict):
            continue
        values = section.get("target_keys")
        if not isinstance(values, list):
            continue
        for value in values:
            text = str(value).strip()
            if not text or text in seen:
                continue
            out.append(text)
            seen.add(text)
    return out


def _candidate_quality_context(
    feedback: Any,
    case: Case,
    metadata: dict[str, Any],
    *,
    target_capabilities: list[str] | tuple[str, ...] = (),
) -> dict[str, Any]:
    if feedback is None:
        return {}
    context_getter = getattr(feedback, "candidate_quality_context", None)
    if context_getter is None:
        return {}
    target_keys = _candidate_target_keys_from_metadata(metadata)
    target_keys.extend(_capability_target_keys(target_capabilities))
    return dict(context_getter(case, target_keys=target_keys or None) or {})


def _attach_metadata_to_row_case(
    row: dict[str, Any],
    key: str,
    payload: dict[str, Any],
) -> None:
    row_case = row.get("case")
    if not isinstance(row_case, dict) or not isinstance(payload, dict):
        return
    metadata = row_case.get("metadata")
    row_case["metadata"] = dict(metadata) if isinstance(metadata, dict) else {}
    row_case["metadata"][key] = payload


def _attach_disagreement_descriptor_to_row_case(
    row: dict[str, Any],
    descriptor: dict[str, Any],
) -> None:
    _attach_metadata_to_row_case(row, "disagreement_descriptor", descriptor)


def _attach_case_fingerprint_to_row_case(
    row: dict[str, Any],
    fingerprint: dict[str, Any],
) -> None:
    _attach_metadata_to_row_case(row, "case_fingerprint", fingerprint)


def _attach_semantic_contract_lattice_to_row_case(
    row: dict[str, Any],
    lattice: dict[str, Any],
) -> None:
    _attach_metadata_to_row_case(row, "semantic_contract_lattice", lattice)


def _attach_interaction_descriptor_to_row_case(
    row: dict[str, Any],
    descriptor: dict[str, Any],
) -> None:
    _attach_metadata_to_row_case(row, "interaction_descriptor", descriptor)


def _fingerprint_anchor_result(normalized: dict[str, Any]) -> Any | None:
    ok_items = [
        (backend, result)
        for backend, result in sorted(normalized.items())
        if getattr(result, "status", "") == "ok"
    ]
    if ok_items:
        return ok_items[0][1]
    items = sorted(normalized.items())
    return items[0][1] if items else None


def _feedback_target_keys(
    guidance_row: dict[str, Any],
    operation_combo: dict[str, Any],
    *,
    target_capabilities: list[str] | tuple[str, ...] = (),
) -> list[str]:
    broad_targets = {
        "strings",
        "numeric",
        "nulls",
        "mutate",
        "aggregation",
        "expressions",
        "groupby",
        "filter",
        "sort_limit",
        "topk",
        "join",
    }
    keys: list[str] = []
    for target in guidance_row.get("matched_targets", []) or []:
        value = str(target).strip()
        if value and value not in broad_targets:
            keys.append(f"target:{value}")
    for feature in guidance_row.get("features", []) or []:
        value = str(feature).strip()
        if value and (
            value.startswith("pattern:")
            or value.startswith("semantic_family:")
            or value.startswith(EXPLORATION_OBJECTIVE_PREFIX)
            or value.startswith("source:")
            or value.startswith("profile:")
        ):
            keys.append(f"feature:{value}")
    template = str(operation_combo.get("template", "")).strip()
    if template:
        keys.append(f"combo:{template}")
    for family in guidance_row.get("features", []) or []:
        value = str(family).strip()
        if value.startswith("semantic_family:"):
            keys.append(value)
        elif value.startswith(EXPLORATION_OBJECTIVE_PREFIX):
            keys.append(value)
    for signal in combo_semantic_signals(operation_combo):
        value = str(signal).strip()
        if value:
            semantic_key = semantic_signal_feature(value)
            keys.append(semantic_key)
            legacy_key = legacy_target_key_alias(semantic_key)
            if legacy_key:
                keys.append(legacy_key)
    config_payload = guidance_row.get("config", {}) if isinstance(guidance_row.get("config", {}), dict) else {}
    for family in config_payload.get("semantic_focus_families", []) or []:
        value = str(family).strip()
        if value:
            keys.append(f"semantic_family:{value}")
    for signal in config_payload.get("semantic_focus_signals", []) or []:
        value = str(signal).strip()
        if value:
            semantic_key = semantic_signal_feature(value)
            keys.append(semantic_key)
            legacy_key = legacy_target_key_alias(semantic_key)
            if legacy_key:
                keys.append(legacy_key)
    keys.extend(_capability_target_keys(target_capabilities))
    out: list[str] = []
    seen: set[str] = set()
    for key in keys:
        if key in seen:
            continue
        out.append(key)
        seen.add(key)
    return out


def _capability_target_keys(target_capabilities: list[str] | tuple[str, ...]) -> list[str]:
    keys: list[str] = []
    seen: set[str] = set()
    for capability in target_capabilities:
        value = str(capability).strip()
        if not value:
            continue
        key = f"capability:{value}"
        if key in seen:
            continue
        keys.append(key)
        seen.add(key)
    return keys[:24]
