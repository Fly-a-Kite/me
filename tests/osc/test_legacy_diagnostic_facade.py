from __future__ import annotations

from copy import deepcopy

import pytest

from datadiff.env import collect_environment
from datadiff.experiment_manifest import (
    finalize_config_payload,
    stable_digest as legacy_stable_digest,
)
from datadiff.normalizer import NormalizedResult
from datadiff.osc_diagnostic_facade import (
    DIAGNOSTIC_AUTHORITY_SCOPE,
    DIAGNOSTIC_NOT_EVALUATED,
    OpaqueDiagnosticRefSet,
    _context_from_legacy_metadata,
    _translate_legacy_result,
    build_legacy_diagnostic_ref_set,
    consume_opaque_diagnostic_ref_set,
    not_evaluated_diagnostic_ref_set,
)
from datadiff.targets import describe_targets
from datadiff_osc.schemas import ExecutionStatus, FailureKind


def _environment() -> dict[str, str]:
    return collect_environment()


def _case_payload() -> dict[str, object]:
    return {"case_id": "case-1", "seed": 1, "tables": [], "program": {}}


def _case_digest() -> str:
    return legacy_stable_digest("case", _case_payload())


def _config() -> dict[str, object]:
    return finalize_config_payload(
        {
            "method_arm": "baseline",
            "method_arm_manifest": {"arm_id": "baseline", "digest": "arm-1"},
            "optimizer": {"enabled": False},
        }
    )


def _target() -> dict[str, object]:
    return describe_targets(["pandas"])[0]


def _raw(
    status: str = "error",
    *,
    error_type: str = "AdapterBoundaryError:RuntimeError",
    error: str = "boom",
    capability_decision: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "backend": "pandas",
        "status": status,
        "error_type": error_type,
        "error": error,
        "duration_ms": 1.0,
        "input_physical_layout": "dataframe_columns",
        "capability_decision": capability_decision,
    }


def _normalized(
    status: str = "error",
    *,
    error_type: str = "AdapterBoundaryError:RuntimeError",
    error: str = "boom",
) -> NormalizedResult:
    return NormalizedResult(
        backend="pandas",
        status=status,
        columns=[],
        rows=[],
        error_type=error_type,
        error=error,
    )


def _build(
    *,
    raw: dict[str, object] | None = None,
    normalized: NormalizedResult | None = None,
    required_capabilities: tuple[str, ...] | None = ("op:filter",),
    nullability: tuple[bool, ...] | None = None,
    order_provenance: str = "",
    target: dict[str, object] | None = None,
    environment: dict[str, str] | None = None,
    config: dict[str, object] | None = None,
) -> dict[str, object]:
    return build_legacy_diagnostic_ref_set(
        backends=["pandas"],
        case_digest=_case_digest(),
        case_payload=_case_payload(),
        raw_results={"pandas": raw or _raw()},
        normalized_results={"pandas": normalized or _normalized()},
        target_specs=[target or _target()],
        environment=environment or _environment(),
        optimizer_config=config or _config(),
        required_capabilities=required_capabilities,
        schema_nullability_by_backend=(
            None if nullability is None else {"pandas": nullability}
        ),
        row_order_provenance_by_backend={"pandas": order_provenance},
    )


def _only_ref(payload: dict[str, object]) -> dict[str, object]:
    refs = payload["refs"]
    assert isinstance(refs, list) and len(refs) == 1
    return refs[0]["ref"]


def _context(
    raw: dict[str, object],
    *,
    required_capabilities: tuple[str, ...] = ("op:filter",),
    nullability: tuple[bool, ...] | None = None,
    order_provenance: str = "",
):
    return _context_from_legacy_metadata(
        backend="pandas",
        case_digest=_case_digest(),
        case_payload=_case_payload(),
        raw_result=raw,
        target_payload=_target(),
        environment=_environment(),
        optimizer_config=_config(),
        required_capabilities=required_capabilities,
    )


@pytest.mark.parametrize(
    "spoof",
    [
        "timeout while checking an unsupported capability",
        "segmentation crash in an alleged semantic domain",
        "unsupported_capability:op:join",
    ],
)
def test_generic_legacy_error_maps_only_to_adapter_error(spoof: str) -> None:
    raw = _raw(status="error", error=spoof)
    diagnostic = _translate_legacy_result(_context(raw), raw, _normalized(error=spoof))

    assert diagnostic.outcome.status is ExecutionStatus.ADAPTER_ERROR
    assert diagnostic.outcome.failure_kind is FailureKind.ADAPTER_ERROR
    assert diagnostic.observation.status == ExecutionStatus.ADAPTER_ERROR.value


def test_normalization_error_maps_only_to_adapter_error() -> None:
    raw = _raw(status="ok", error_type="", error="")
    diagnostic = _translate_legacy_result(
        _context(raw),
        raw,
        _normalized(
            status="normalization_error",
            error_type="ValueError",
            error="timeout unsupported crash",
        ),
    )

    assert diagnostic.outcome.status is ExecutionStatus.ADAPTER_ERROR
    assert diagnostic.outcome.failure_kind is FailureKind.ADAPTER_ERROR


def test_forged_unbound_capability_claim_is_missing_not_unsupported() -> None:
    forged = {
        "schema_version": "target-capability-decision-v1",
        "supported": False,
        "skip_reason": "unsupported_capability:operation:op:not-real",
        "missing": ["operation:op:not-real"],
        "capability_digest": "forged",
    }
    raw = _raw(
        status="missing",
        error_type="UnsupportedCapability",
        error=str(forged["skip_reason"]),
        capability_decision=forged,
    )
    diagnostic = _translate_legacy_result(_context(raw), raw, _normalized("missing"))

    assert diagnostic.outcome.status is ExecutionStatus.MISSING
    assert diagnostic.outcome.failure_kind is FailureKind.MISSING_RESULT
    assert not diagnostic.outcome.unsupported_evidence_digest


def test_unsupported_requires_recomputed_bound_capability_evidence() -> None:
    raw_seed = _raw(status="missing")
    context = _context(raw_seed, required_capabilities=("op:not-real",))
    expected = {
        key: value for key, value in context.expected_legacy_capability_decision
    }
    raw = _raw(
        status="missing",
        error_type="UnsupportedCapability",
        error=str(expected["skip_reason"]),
        capability_decision=expected,
    )
    diagnostic = _translate_legacy_result(context, raw, _normalized("missing"))

    assert diagnostic.outcome.status is ExecutionStatus.UNSUPPORTED
    assert diagnostic.outcome.failure_kind is FailureKind.UNSUPPORTED_CAPABILITY
    assert diagnostic.unsupported_evidence
    assert diagnostic.outcome.unsupported_evidence_digest == (
        diagnostic.unsupported_evidence[0].digest
    )


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ("missing_required", "missing_required_capabilities"),
        ("missing_version", "missing_backend_version"),
        ("bad_config", "optimizer_config_digest_mismatch"),
        ("bad_capability_digest", "target_registry_mismatch"),
        ("missing_layout", "missing_physical_layout"),
    ],
)
def test_missing_or_mismatched_typed_context_fails_closed(
    mutation: str, reason: str
) -> None:
    raw = _raw()
    target = _target()
    environment = _environment()
    config = _config()
    required: tuple[str, ...] | None = ("op:filter",)
    if mutation == "missing_required":
        required = None
    elif mutation == "missing_version":
        environment.pop("pandas")
    elif mutation == "bad_config":
        config["optimizer"] = {"enabled": True}
    elif mutation == "bad_capability_digest":
        target = deepcopy(target)
        target["capability_model"]["digest"] = "forged"
    elif mutation == "missing_layout":
        raw.pop("input_physical_layout")

    payload = _build(
        raw=raw,
        target=target,
        environment=environment,
        config=config,
        required_capabilities=required,
    )
    ref = _only_ref(payload)

    assert payload["evaluation_status"] == DIAGNOSTIC_NOT_EVALUATED
    assert ref["evaluation_status"] == DIAGNOSTIC_NOT_EVALUATED
    assert ref["reason_code"] == reason
    assert ref["authority_scope"] == DIAGNOSTIC_AUTHORITY_SCOPE
    assert ref["authority_eligible"] is False


def test_ok_rejects_caller_claimed_nullability_and_row_order_provenance() -> None:
    raw = _raw(status="ok", error_type="", error="")
    normalized = NormalizedResult(
        backend="pandas",
        status="ok",
        columns=["value"],
        rows=[[1]],
        column_types=["int64"],
        lossless_rows=[[{}]],
        lossless_schema_version="caller-claimed-lossless-v1",
    )

    ref = _only_ref(
        _build(
            raw=raw,
            normalized=normalized,
            nullability=(False,),
            order_provenance="source_preserved",
        )
    )

    assert ref["evaluation_status"] == DIAGNOSTIC_NOT_EVALUATED
    assert ref["reason_code"] == "legacy_ok_exact_schema_unavailable"


def test_legacy_ok_not_evaluated_reference_is_deterministic() -> None:
    raw = _raw(status="ok", error_type="", error="")
    normalized = NormalizedResult(
        backend="pandas",
        status="ok",
        columns=["value"],
        rows=[[1], [1], [None]],
        column_types=["Int64"],
        lossless_rows=[[{}], [{}], [{}]],
        lossless_schema_version="caller-claimed-lossless-v1",
    )
    payload_a = _build(
        raw=raw,
        normalized=normalized,
        nullability=(True,),
        order_provenance="source_preserved",
    )
    payload_b = _build(
        raw=deepcopy(raw),
        normalized=NormalizedResult.from_dict(normalized.to_dict()),
        nullability=(True,),
        order_provenance="source_preserved",
    )

    assert payload_a == payload_b
    assert payload_a["evaluation_status"] == DIAGNOSTIC_NOT_EVALUATED


@pytest.mark.parametrize(
    ("raw_status", "normalized_status"),
    [
        ("timeout", "error"),
        ("error", "ok"),
        ("missing", "timeout"),
    ],
)
def test_raw_normalized_failure_status_mismatch_is_not_evaluated(
    raw_status: str, normalized_status: str
) -> None:
    payload = _build(
        raw=_raw(status=raw_status),
        normalized=_normalized(normalized_status),
    )

    assert _only_ref(payload)["reason_code"] == "raw_normalized_status_mismatch"


def test_untyped_semantic_error_status_is_not_evaluated() -> None:
    payload = _build(
        raw=_raw(status="semantic_error", error_type="DomainError"),
        normalized=_normalized("semantic_error", error_type="DomainError"),
    )

    assert _only_ref(payload)["reason_code"] == "unsupported_legacy_status"


def test_none_target_specs_fail_closed_without_raising() -> None:
    payload = build_legacy_diagnostic_ref_set(
        backends=["pandas"],
        case_digest=_case_digest(),
        case_payload=_case_payload(),
        raw_results={"pandas": _raw()},
        normalized_results={"pandas": _normalized()},
        target_specs=None,
        environment=_environment(),
        optimizer_config=_config(),
        required_capabilities=("op:filter",),
    )

    assert _only_ref(payload)["reason_code"] == "missing_or_ambiguous_target_spec"


@pytest.mark.parametrize("expected_backends", [None, "pandas", 7])
def test_consumer_malformed_backend_context_fails_closed(expected_backends) -> None:
    payload = consume_opaque_diagnostic_ref_set(
        None,
        expected_backends=expected_backends,
        case_digest=_case_digest(),
    )

    assert payload["evaluation_status"] == DIAGNOSTIC_NOT_EVALUATED
    assert payload["refs"] == []


def test_backend_context_rejects_values_with_hostile_string_conversion() -> None:
    class BadString:
        def __str__(self):
            raise RuntimeError("must not stringify untrusted backend identities")

    for expected_backends in ([BadString()], [None], [7], ["pandas", "pandas"]):
        payload = consume_opaque_diagnostic_ref_set(
            None,
            expected_backends=expected_backends,
            case_digest=_case_digest(),
        )
        assert payload["evaluation_status"] == DIAGNOSTIC_NOT_EVALUATED
        assert payload["refs"] == []


def test_all_public_string_fields_reject_hostile_objects_without_conversion() -> None:
    class BadString:
        def __str__(self):
            raise RuntimeError("public boundary must not call __str__")

    produced = _build()
    bad_backend = deepcopy(produced)
    bad_backend["refs"][0]["backend"] = BadString()
    bad_ref_id = deepcopy(produced)
    bad_ref_id["refs"][0]["ref"]["ref_id"] = BadString()

    for payload in (bad_backend, bad_ref_id):
        consumed = consume_opaque_diagnostic_ref_set(
            payload,
            expected_backends=["pandas"],
            case_digest=_case_digest(),
        )
        assert consumed["evaluation_status"] == DIAGNOSTIC_NOT_EVALUATED
        assert _only_ref(consumed)["reason_code"] == "invalid_diagnostic_refs"

    hostile_expected_case = consume_opaque_diagnostic_ref_set(
        None,
        expected_backends=["pandas"],
        case_digest=BadString(),
    )
    assert hostile_expected_case["case_digest"] == ""
    assert _only_ref(hostile_expected_case)["evaluation_status"] == (
        DIAGNOSTIC_NOT_EVALUATED
    )

    hostile_build_case = build_legacy_diagnostic_ref_set(
        backends=["pandas"],
        case_digest=BadString(),
        case_payload=_case_payload(),
        raw_results={"pandas": _raw()},
        normalized_results={"pandas": _normalized()},
        target_specs=[_target()],
        environment=_environment(),
        optimizer_config=_config(),
        required_capabilities=("op:filter",),
    )
    assert hostile_build_case["case_digest"] == ""
    assert _only_ref(hostile_build_case)["evaluation_status"] == (
        DIAGNOSTIC_NOT_EVALUATED
    )

    hostile_target = deepcopy(_target())
    hostile_target["backend"] = BadString()
    hostile_target_payload = _build(target=hostile_target)
    assert _only_ref(hostile_target_payload)["reason_code"] == (
        "missing_or_ambiguous_target_spec"
    )

    hostile_fallback = not_evaluated_diagnostic_ref_set(
        ["pandas"],
        case_digest=BadString(),
        reason_code=BadString(),
    )
    assert hostile_fallback["case_digest"] == ""
    assert _only_ref(hostile_fallback)["reason_code"] == (
        "invalid_diagnostic_context"
    )


def test_build_skips_hostile_target_mapping_and_fails_closed() -> None:
    from collections.abc import Mapping

    class BadMapping(Mapping):
        def __getitem__(self, key):
            raise RuntimeError("hostile mapping item")

        def __iter__(self):
            return iter(())

        def __len__(self):
            return 0

        def get(self, key, default=None):
            raise RuntimeError("hostile mapping get")

    payload = build_legacy_diagnostic_ref_set(
        backends=["pandas"],
        case_digest=_case_digest(),
        case_payload=_case_payload(),
        raw_results={"pandas": _raw()},
        normalized_results={"pandas": _normalized()},
        target_specs=[BadMapping()],
        environment=_environment(),
        optimizer_config=_config(),
        required_capabilities=("op:filter",),
    )

    assert _only_ref(payload)["reason_code"] == "missing_or_ambiguous_target_spec"


def test_consumer_requires_exact_expected_backend_context() -> None:
    consumed = consume_opaque_diagnostic_ref_set(
        _build(),
        expected_backends=None,
        case_digest=_case_digest(),
    )

    assert consumed["evaluation_status"] == DIAGNOSTIC_NOT_EVALUATED
    assert consumed["refs"] == []


def test_consumer_rejects_cross_case_opaque_ref_replay() -> None:
    original = _build()

    consumed = consume_opaque_diagnostic_ref_set(
        original,
        expected_backends=["pandas"],
        case_digest="case-forged-other",
    )

    assert original["case_digest"] == _case_digest()
    assert consumed["case_digest"] == "case-forged-other"
    assert consumed["evaluation_status"] == DIAGNOSTIC_NOT_EVALUATED
    assert _only_ref(consumed)["reason_code"] == "invalid_diagnostic_refs"
    assert consumed["ref_set_id"] != original["ref_set_id"]


def test_consumer_rejects_missing_or_tampered_case_binding() -> None:
    original = _build()
    missing_expected = consume_opaque_diagnostic_ref_set(
        original,
        expected_backends=["pandas"],
        case_digest="",
    )
    missing_field = deepcopy(original)
    missing_field.pop("case_digest")
    consumed_missing_field = consume_opaque_diagnostic_ref_set(
        missing_field,
        expected_backends=["pandas"],
        case_digest=_case_digest(),
    )
    tampered = deepcopy(original)
    tampered["case_digest"] = "case-tampered"
    consumed_tampered = consume_opaque_diagnostic_ref_set(
        tampered,
        expected_backends=["pandas"],
        case_digest="case-tampered",
    )

    assert missing_expected["evaluation_status"] == DIAGNOSTIC_NOT_EVALUATED
    assert missing_expected["case_digest"] == ""
    assert consumed_missing_field["case_digest"] == _case_digest()
    assert _only_ref(consumed_missing_field)["reason_code"] == (
        "invalid_diagnostic_refs"
    )
    assert consumed_tampered["case_digest"] == "case-tampered"
    assert _only_ref(consumed_tampered)["reason_code"] == "invalid_diagnostic_refs"


def test_consumer_downgrades_opaque_evaluated_claim_without_typed_producer() -> None:
    produced = _build()
    assert produced["evaluation_status"] == "evaluated"
    produced_ref_id = _only_ref(produced)["ref_id"]

    consumed = consume_opaque_diagnostic_ref_set(
        produced,
        expected_backends=["pandas"],
        case_digest=_case_digest(),
    )

    assert consumed["evaluation_status"] == DIAGNOSTIC_NOT_EVALUATED
    assert _only_ref(consumed) == {
        "schema_version": "datadiff-opaque-osc-diagnostic-ref-v1",
        "ref_id": produced_ref_id,
        "evaluation_status": DIAGNOSTIC_NOT_EVALUATED,
        "reason_code": "opaque_reference_not_revalidated",
        "authority_scope": DIAGNOSTIC_AUTHORITY_SCOPE,
        "authority_eligible": False,
    }


def test_target_registry_rebinding_is_not_evaluated() -> None:
    target = deepcopy(_target())
    target["description"] = "caller-authored replacement"

    ref = _only_ref(_build(target=target))

    assert ref["reason_code"] == "target_registry_mismatch"


def test_environment_rebinding_is_not_evaluated() -> None:
    environment = _environment()
    environment["source_tree_sha256"] = "f" * 64

    ref = _only_ref(_build(environment=environment))

    assert ref["reason_code"] == "environment_binding_mismatch"


def test_case_digest_rebinding_is_not_evaluated() -> None:
    payload = build_legacy_diagnostic_ref_set(
        backends=["pandas"],
        case_digest="case-forged",
        case_payload=_case_payload(),
        raw_results={"pandas": _raw()},
        normalized_results={"pandas": _normalized()},
        target_specs=[_target()],
        environment=_environment(),
        optimizer_config=_config(),
        required_capabilities=("op:filter",),
    )

    assert _only_ref(payload)["reason_code"] == "case_digest_mismatch"


def test_opaque_payload_contains_no_verdict_or_evidence_authority() -> None:
    payload = _build()
    forbidden_keys = {
        "admission",
        "coverage",
        "receipt",
        "evidence",
        "evidence_envelope",
        "verdict",
        "certificate",
        "candidate",
        "bug",
        "gate",
        "twenty_four_hour",
    }

    def keys(value):
        if isinstance(value, dict):
            yield from value
            for item in value.values():
                yield from keys(item)
        elif isinstance(value, list):
            for item in value:
                yield from keys(item)

    assert not (set(keys(payload)) & forbidden_keys)
    assert payload["authority_scope"] == DIAGNOSTIC_AUTHORITY_SCOPE
    assert payload["authority_eligible"] is False
    assert OpaqueDiagnosticRefSet.from_dict(payload).to_dict() == payload


def test_consumer_rejects_extra_or_forged_authority_fields() -> None:
    payload = _build()
    forged = deepcopy(payload)
    forged["authority_eligible"] = True
    forged["coverage"] = {"observed": 1}

    consumed = consume_opaque_diagnostic_ref_set(
        forged,
        expected_backends=["pandas"],
        case_digest=_case_digest(),
    )
    ref = _only_ref(consumed)

    assert consumed["evaluation_status"] == DIAGNOSTIC_NOT_EVALUATED
    assert consumed["authority_eligible"] is False
    assert ref["reason_code"] == "invalid_diagnostic_refs"
    assert "coverage" not in consumed
