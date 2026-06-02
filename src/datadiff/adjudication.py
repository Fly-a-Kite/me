from __future__ import annotations

from typing import Any


_DEFAULT_POLICY = {
    "validity_gate": "unknown",
    "semantic_gate": "unknown",
    "attribution_gate": "unknown",
    "reference_support": "not_used",
    "metamorphic_support": "not_used",
    "documentation_support": "not_used",
    "recheck_status": "not_requested",
    "countable_as_bug_evidence": False,
    "countable_as_valid_finding": False,
    "needs_manual_review": True,
    "needs_external_confirmation": False,
    "exclusion_reason": "",
    "boundary_rule_ids": [],
    "documented_rule_ids": [],
}


_VERDICT_POLICIES: dict[str, dict[str, Any]] = {
    "unclassified": dict(_DEFAULT_POLICY),
    "generator_false_positive": {
        **_DEFAULT_POLICY,
        "validity_gate": "invalid_generated_case",
        "semantic_gate": "out_of_scope",
        "attribution_gate": "generator_fault",
        "needs_manual_review": False,
    },
    "normalizer_false_positive": {
        **_DEFAULT_POLICY,
        "validity_gate": "harness_failure",
        "semantic_gate": "out_of_scope",
        "attribution_gate": "normalizer_or_adapter_fault",
        "needs_manual_review": False,
    },
    "documented_semantic_divergence": {
        **_DEFAULT_POLICY,
        "validity_gate": "valid_case",
        "semantic_gate": "documented_boundary",
        "attribution_gate": "documented_semantics",
        "documentation_support": "documented",
        "countable_as_valid_finding": True,
        "needs_manual_review": False,
    },
    "expected_semantic_divergence": {
        **_DEFAULT_POLICY,
        "validity_gate": "valid_case",
        "semantic_gate": "expected_boundary",
        "attribution_gate": "semantic_boundary",
        "countable_as_valid_finding": True,
        "needs_manual_review": False,
    },
    "semantic_divergence_needs_confirmation": {
        **_DEFAULT_POLICY,
        "validity_gate": "valid_case",
        "semantic_gate": "boundary_needs_confirmation",
        "attribution_gate": "semantic_boundary",
        "countable_as_valid_finding": True,
        "needs_manual_review": True,
    },
    "candidate_implementation_bug": {
        **_DEFAULT_POLICY,
        "validity_gate": "valid_case",
        "semantic_gate": "common_subset_or_backend_specific",
        "attribution_gate": "backend_candidate_bug",
        "countable_as_bug_evidence": True,
        "countable_as_valid_finding": True,
        "needs_manual_review": False,
        "needs_external_confirmation": True,
    },
    "needs_manual_confirmation": {
        **_DEFAULT_POLICY,
        "validity_gate": "valid_case",
        "semantic_gate": "unknown",
        "attribution_gate": "manual_triage_required",
        "countable_as_valid_finding": True,
        "needs_manual_review": True,
    },
    "non_reproducible_candidate": {
        **_DEFAULT_POLICY,
        "validity_gate": "recheck_failed",
        "semantic_gate": "unknown",
        "attribution_gate": "reproduction_failed",
        "recheck_status": "failed",
        "needs_manual_review": False,
    },
    "not_reproduced": {
        **_DEFAULT_POLICY,
        "validity_gate": "not_reproduced",
        "semantic_gate": "unknown",
        "attribution_gate": "reproduction_missing",
        "needs_manual_review": False,
    },
}


def build_adjudication(
    verdict: str,
    *,
    validity_gate: str | None = None,
    semantic_gate: str | None = None,
    attribution_gate: str | None = None,
    reference_support: str | None = None,
    metamorphic_support: str | None = None,
    documentation_support: str | None = None,
    recheck_status: str | None = None,
    countable_as_bug_evidence: bool | None = None,
    countable_as_valid_finding: bool | None = None,
    needs_manual_review: bool | None = None,
    needs_external_confirmation: bool | None = None,
    exclusion_reason: str | None = None,
    boundary_rule_ids: list[str] | tuple[str, ...] | None = None,
    documented_rule_ids: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    payload = verdict_policy(verdict)
    payload["verdict"] = str(verdict or "unclassified")
    _override_str(payload, "validity_gate", validity_gate)
    _override_str(payload, "semantic_gate", semantic_gate)
    _override_str(payload, "attribution_gate", attribution_gate)
    _override_str(payload, "reference_support", reference_support)
    _override_str(payload, "metamorphic_support", metamorphic_support)
    _override_str(payload, "documentation_support", documentation_support)
    _override_str(payload, "recheck_status", recheck_status)
    _override_bool(payload, "countable_as_bug_evidence", countable_as_bug_evidence)
    _override_bool(payload, "countable_as_valid_finding", countable_as_valid_finding)
    _override_bool(payload, "needs_manual_review", needs_manual_review)
    _override_bool(payload, "needs_external_confirmation", needs_external_confirmation)
    if exclusion_reason is not None:
        payload["exclusion_reason"] = str(exclusion_reason)
    if boundary_rule_ids is not None:
        payload["boundary_rule_ids"] = _normalize_string_list(boundary_rule_ids)
    if documented_rule_ids is not None:
        payload["documented_rule_ids"] = _normalize_string_list(documented_rule_ids)
    return payload


def verdict_policy(verdict: str) -> dict[str, Any]:
    base = _VERDICT_POLICIES.get(str(verdict or "unclassified"), _DEFAULT_POLICY)
    return {
        "validity_gate": str(base.get("validity_gate", "unknown")),
        "semantic_gate": str(base.get("semantic_gate", "unknown")),
        "attribution_gate": str(base.get("attribution_gate", "unknown")),
        "reference_support": str(base.get("reference_support", "not_used")),
        "metamorphic_support": str(base.get("metamorphic_support", "not_used")),
        "documentation_support": str(base.get("documentation_support", "not_used")),
        "recheck_status": str(base.get("recheck_status", "not_requested")),
        "countable_as_bug_evidence": bool(base.get("countable_as_bug_evidence", False)),
        "countable_as_valid_finding": bool(base.get("countable_as_valid_finding", False)),
        "needs_manual_review": bool(base.get("needs_manual_review", True)),
        "needs_external_confirmation": bool(base.get("needs_external_confirmation", False)),
        "exclusion_reason": str(base.get("exclusion_reason", "")),
        "boundary_rule_ids": _normalize_string_list(base.get("boundary_rule_ids", [])),
        "documented_rule_ids": _normalize_string_list(base.get("documented_rule_ids", [])),
    }


def finding_adjudication(finding: dict[str, Any] | Any) -> dict[str, Any]:
    verdict = str(_get(finding, "triage_verdict", "unclassified") or "unclassified")
    payload = build_adjudication(verdict)
    existing = _get(finding, "adjudication", {})
    if isinstance(existing, dict) and existing:
        payload = build_adjudication(
            verdict,
            validity_gate=_as_optional_str(existing.get("validity_gate")),
            semantic_gate=_as_optional_str(existing.get("semantic_gate")),
            attribution_gate=_as_optional_str(existing.get("attribution_gate")),
            reference_support=_as_optional_str(existing.get("reference_support")),
            metamorphic_support=_as_optional_str(existing.get("metamorphic_support")),
            documentation_support=_as_optional_str(existing.get("documentation_support")),
            recheck_status=_as_optional_str(existing.get("recheck_status")),
            countable_as_bug_evidence=_as_optional_bool(existing.get("countable_as_bug_evidence")),
            countable_as_valid_finding=_as_optional_bool(existing.get("countable_as_valid_finding")),
            needs_manual_review=_as_optional_bool(existing.get("needs_manual_review")),
            needs_external_confirmation=_as_optional_bool(existing.get("needs_external_confirmation")),
            exclusion_reason=_as_optional_str(existing.get("exclusion_reason")),
            boundary_rule_ids=_as_optional_string_list(existing.get("boundary_rule_ids")),
            documented_rule_ids=_as_optional_string_list(existing.get("documented_rule_ids")),
        )
    false_positive_reason = str(_get(finding, "false_positive_reason", "") or "")
    if false_positive_reason and not str(payload.get("exclusion_reason", "")):
        payload["exclusion_reason"] = false_positive_reason
    return payload


def counts_as_bug_evidence(finding: dict[str, Any] | Any) -> bool:
    adjudication = finding_adjudication(finding)
    if bool(_get(finding, "false_positive", False)):
        return False
    return bool(adjudication.get("countable_as_bug_evidence", False))


def counts_as_valid_finding(finding: dict[str, Any] | Any) -> bool:
    adjudication = finding_adjudication(finding)
    return bool(adjudication.get("countable_as_valid_finding", False))


def _override_str(payload: dict[str, Any], key: str, value: str | None) -> None:
    if value is not None:
        payload[key] = str(value)


def _override_bool(payload: dict[str, Any], key: str, value: bool | None) -> None:
    if value is not None:
        payload[key] = bool(value)


def _normalize_string_list(values: Any) -> list[str]:
    if not isinstance(values, (list, tuple)):
        return []
    return [str(value) for value in values if str(value)]


def _get(finding: dict[str, Any] | Any, key: str, default: Any = None) -> Any:
    if isinstance(finding, dict):
        return finding.get(key, default)
    return getattr(finding, key, default)


def _as_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _as_optional_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _as_optional_string_list(value: Any) -> list[str] | None:
    if not isinstance(value, (list, tuple)):
        return None
    return _normalize_string_list(value)
