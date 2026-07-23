"""Fail-closed legacy-to-OSC diagnostics with no verdict authority.

This module is intentionally owned by the legacy ``datadiff`` package.  It may
construct frozen OSC endpoint/observation/outcome values internally so the
legacy boundary has a typed failure taxonomy, but it exports only opaque
diagnostic references to downstream legacy consumers.  Those references are
never admission, evidence, coverage, candidate, gate, or run authority.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from datadiff.capability_model import CapabilityModel
from datadiff.env import collect_environment
from datadiff.experiment_manifest import (
    finalize_config_payload,
    stable_digest as legacy_stable_digest,
)
from datadiff.targets import target_spec
from datadiff_osc._canonical import (
    frozen_pairs,
    pairs_dict,
    stable_digest,
    to_primitive,
)
from datadiff_osc.contract_engine.capability import (
    UnsupportedEvidence,
    decide_capabilities,
)
from datadiff_osc.contract_engine.model import Endpoint, Observation
from datadiff_osc.schemas import (
    ExecutionStatus,
    FailureKind,
    StructuredExecutionOutcome,
)


OPAQUE_DIAGNOSTIC_REF_SCHEMA_VERSION = "datadiff-opaque-osc-diagnostic-ref-v1"
OPAQUE_DIAGNOSTIC_REF_SET_SCHEMA_VERSION = (
    "datadiff-opaque-osc-diagnostic-ref-set-v1"
)
LEGACY_DIAGNOSTIC_CONTEXT_SCHEMA_VERSION = (
    "datadiff-legacy-osc-diagnostic-context-v1"
)
LEGACY_TYPED_DIAGNOSTIC_SCHEMA_VERSION = "datadiff-legacy-typed-diagnostic-v1"
DIAGNOSTIC_AUTHORITY_SCOPE = "diagnostic_only"
DIAGNOSTIC_EVALUATED = "evaluated"
DIAGNOSTIC_NOT_EVALUATED = "not_evaluated"
DIAGNOSTIC_PARTIAL = "partial"

_REF_FIELDS = frozenset(
    {
        "schema_version",
        "ref_id",
        "evaluation_status",
        "reason_code",
        "authority_scope",
        "authority_eligible",
    }
)
_REF_SET_FIELDS = frozenset(
    {
        "schema_version",
        "ref_set_id",
        "case_digest",
        "evaluation_status",
        "authority_scope",
        "authority_eligible",
        "refs",
    }
)
_ADAPTER_MODULE_ENVIRONMENT_KEYS = {
    "datadiff.backends.chdb_backend": "chdb",
    "datadiff.backends.datafusion_backend": "datafusion",
    "datadiff.backends.duckdb_backend": "duckdb",
    "datadiff.backends.faulty_backend": "pandas",
    "datadiff.backends.numpy_backend": "numpy",
    "datadiff.backends.pandas_backend": "pandas",
    "datadiff.backends.polars_backend": "polars",
    "datadiff.backends.pyarrow_backend": "pyarrow",
    "datadiff.backends.sqlite_backend": "sqlite",
}


class _NotEvaluated(ValueError):
    def __init__(self, reason_code: str) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code


def _require_text(value: Any, reason_code: str) -> str:
    if not isinstance(value, str) or not value:
        raise _NotEvaluated(reason_code)
    return value


def _safe_text(value: Any, fallback: str = "") -> str:
    return value if isinstance(value, str) and value else fallback


def _mapping(value: Any, reason_code: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
        if isinstance(payload, Mapping):
            return dict(payload)
    raise _NotEvaluated(reason_code)


def _backend_names(value: Any) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    try:
        items = tuple(value)
    except Exception:
        return ()
    if any(not isinstance(item, str) or not item for item in items):
        return ()
    if len(items) != len(set(items)):
        return ()
    return tuple(sorted(items))


def _summary_mapping(value: Any, reason_code: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    summary_dict = getattr(value, "summary_dict", None)
    if callable(summary_dict):
        payload = summary_dict()
        if isinstance(payload, Mapping):
            return dict(payload)
    return _mapping(value, reason_code)


def _capability_model(payload: Any) -> CapabilityModel:
    model_payload = _mapping(payload, "invalid_capability_model")
    declared_digest = _require_text(
        model_payload.pop("digest", ""), "missing_capability_model_digest"
    )
    try:
        model = CapabilityModel(
            logical_types=tuple(model_payload["logical_types"]),
            operation_tokens=tuple(model_payload["operation_tokens"]),
            null_semantics=tuple(model_payload["null_semantics"]),
            order_semantics=tuple(model_payload["order_semantics"]),
            plan_kinds=tuple(model_payload["plan_kinds"]),
            execution_modes=tuple(model_payload["execution_modes"]),
            physical_layouts=tuple(model_payload["physical_layouts"]),
            operation_type_support={
                str(operation): tuple(types)
                for operation, types in dict(
                    model_payload["operation_type_support"]
                ).items()
            },
            error_statuses=tuple(
                model_payload.get("error_statuses", ("error", "missing", "ok", "timeout"))
            ),
            native_export_formats=tuple(
                model_payload.get("native_export_formats", ())
            ),
            schema_version=str(model_payload.get("schema_version", "")),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise _NotEvaluated("invalid_capability_model") from exc
    if model.digest != declared_digest:
        raise _NotEvaluated("capability_model_digest_mismatch")
    return model


def _backend_version(
    *, target_payload: Mapping[str, Any], environment: Mapping[str, Any]
) -> str:
    adapter = _require_text(target_payload.get("adapter"), "missing_adapter_identity")
    adapter_module, separator, _class_name = adapter.rpartition(".")
    if not separator or adapter_module not in _ADAPTER_MODULE_ENVIRONMENT_KEYS:
        raise _NotEvaluated("unbound_backend_version_source")
    environment_key = _ADAPTER_MODULE_ENVIRONMENT_KEYS[adapter_module]
    trusted_environment = collect_environment()
    version = _require_text(
        environment.get(environment_key), "missing_backend_version"
    )
    if (
        version != trusted_environment.get(environment_key)
        or environment.get("source_tree_sha256")
        != trusted_environment.get("source_tree_sha256")
    ):
        raise _NotEvaluated("environment_binding_mismatch")
    if version == "not-installed":
        raise _NotEvaluated("backend_version_not_installed")
    return version


def _validated_optimizer_config(config_payload: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(config_payload)
    declared_digest = _require_text(
        payload.get("config_digest"), "missing_optimizer_config_digest"
    )
    try:
        recomputed = finalize_config_payload(payload)
    except (TypeError, ValueError) as exc:
        raise _NotEvaluated("invalid_optimizer_config") from exc
    if recomputed.get("config_digest") != declared_digest or recomputed != payload:
        raise _NotEvaluated("optimizer_config_digest_mismatch")
    return payload


@dataclass(frozen=True, slots=True)
class LegacyDiagnosticContext:
    """Exact endpoint context; still permanently diagnostic-only."""

    endpoint: Endpoint
    required_capabilities: frozenset[str]
    capability_model_digest: str
    expected_legacy_capability_decision: tuple[tuple[str, Any], ...]
    authority_scope: str = field(default=DIAGNOSTIC_AUTHORITY_SCOPE, init=False)
    authority_eligible: bool = field(default=False, init=False)
    schema_version: str = field(
        default=LEGACY_DIAGNOSTIC_CONTEXT_SCHEMA_VERSION, init=False
    )

    def __post_init__(self) -> None:
        if not isinstance(self.endpoint, Endpoint):
            raise ValueError("legacy diagnostic context requires a typed Endpoint")
        if not self.capability_model_digest:
            raise ValueError("legacy diagnostic context requires a capability digest")
        if any(
            not isinstance(item, str) or not item
            for item in self.required_capabilities
        ):
            raise ValueError("required capabilities must be non-empty strings")
        if self.authority_scope != DIAGNOSTIC_AUTHORITY_SCOPE:
            raise ValueError("legacy diagnostic context is diagnostic-only")
        if self.authority_eligible:
            raise ValueError("legacy diagnostic context cannot be authority eligible")

    @property
    def digest(self) -> str:
        return stable_digest("legacy-diagnostic-context", self)


@dataclass(frozen=True, slots=True)
class _LegacyTypedDiagnostic:
    endpoint: Endpoint
    observation: Observation
    outcome: StructuredExecutionOutcome
    context_digest: str
    unsupported_evidence: tuple[UnsupportedEvidence, ...] = ()
    authority_scope: str = field(default=DIAGNOSTIC_AUTHORITY_SCOPE, init=False)
    authority_eligible: bool = field(default=False, init=False)
    schema_version: str = field(
        default=LEGACY_TYPED_DIAGNOSTIC_SCHEMA_VERSION, init=False
    )

    def __post_init__(self) -> None:
        endpoint_id = self.endpoint.endpoint_id
        if self.observation.endpoint_id != endpoint_id:
            raise ValueError("diagnostic observation endpoint mismatch")
        if self.outcome.endpoint_id != endpoint_id:
            raise ValueError("diagnostic outcome endpoint mismatch")
        if self.observation.status != self.outcome.status.value:
            raise ValueError("diagnostic observation/outcome status mismatch")
        evidence_digests = {
            item.digest for item in self.unsupported_evidence if item.valid
        }
        if self.outcome.status is ExecutionStatus.UNSUPPORTED:
            if self.outcome.unsupported_evidence_digest not in evidence_digests:
                raise ValueError("unsupported diagnostic evidence mismatch")
        elif self.unsupported_evidence:
            raise ValueError("only unsupported diagnostics carry capability evidence")
        if self.authority_scope != DIAGNOSTIC_AUTHORITY_SCOPE or self.authority_eligible:
            raise ValueError("legacy typed diagnostic cannot carry authority")

    @property
    def digest(self) -> str:
        return stable_digest("legacy-typed-diagnostic", self)


@dataclass(frozen=True, slots=True)
class OpaqueDiagnosticRef:
    ref_id: str
    evaluation_status: str
    reason_code: str
    authority_scope: str = field(default=DIAGNOSTIC_AUTHORITY_SCOPE, init=False)
    authority_eligible: bool = field(default=False, init=False)
    schema_version: str = field(
        default=OPAQUE_DIAGNOSTIC_REF_SCHEMA_VERSION, init=False
    )

    def __post_init__(self) -> None:
        prefix = "legacy-diagnostic-ref-"
        suffix = self.ref_id.removeprefix(prefix)
        if (
            not self.ref_id.startswith(prefix)
            or len(suffix) != 64
            or any(character not in "0123456789abcdef" for character in suffix)
        ):
            raise ValueError("invalid opaque diagnostic reference identity")
        if self.evaluation_status not in {
            DIAGNOSTIC_EVALUATED,
            DIAGNOSTIC_NOT_EVALUATED,
        }:
            raise ValueError("invalid diagnostic evaluation status")
        if not self.reason_code:
            raise ValueError("diagnostic reference requires a reason code")
        if self.authority_scope != DIAGNOSTIC_AUTHORITY_SCOPE:
            raise ValueError("diagnostic reference scope is immutable")
        if self.authority_eligible:
            raise ValueError("diagnostic reference cannot be authority eligible")

    @classmethod
    def evaluated(cls, diagnostic: _LegacyTypedDiagnostic) -> "OpaqueDiagnosticRef":
        return cls(
            ref_id=stable_digest("legacy-diagnostic-ref", diagnostic.digest),
            evaluation_status=DIAGNOSTIC_EVALUATED,
            reason_code="typed_diagnostic_available",
        )

    @classmethod
    def not_evaluated(
        cls, *, backend: str, case_digest: str, reason_code: str
    ) -> "OpaqueDiagnosticRef":
        safe_backend = _safe_text(backend)
        safe_case_digest = _safe_text(case_digest)
        safe_reason_code = _safe_text(
            reason_code, "invalid_diagnostic_context"
        )
        return cls(
            ref_id=stable_digest(
                "legacy-diagnostic-ref",
                {
                    "backend": safe_backend,
                    "case_digest": safe_case_digest,
                    "evaluation_status": DIAGNOSTIC_NOT_EVALUATED,
                    "reason_code": safe_reason_code,
                },
            ),
            evaluation_status=DIAGNOSTIC_NOT_EVALUATED,
            reason_code=safe_reason_code,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "ref_id": self.ref_id,
            "evaluation_status": self.evaluation_status,
            "reason_code": self.reason_code,
            "authority_scope": self.authority_scope,
            "authority_eligible": self.authority_eligible,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "OpaqueDiagnosticRef":
        if set(payload) != _REF_FIELDS:
            raise ValueError("opaque diagnostic reference fields do not match")
        if payload.get("schema_version") != OPAQUE_DIAGNOSTIC_REF_SCHEMA_VERSION:
            raise ValueError("unsupported opaque diagnostic reference schema")
        if payload.get("authority_scope") != DIAGNOSTIC_AUTHORITY_SCOPE:
            raise ValueError("opaque diagnostic reference scope mismatch")
        if payload.get("authority_eligible") is not False:
            raise ValueError("opaque diagnostic reference claims authority")
        ref_id = payload.get("ref_id")
        evaluation_status = payload.get("evaluation_status")
        reason_code = payload.get("reason_code")
        if not all(
            isinstance(item, str)
            for item in (ref_id, evaluation_status, reason_code)
        ):
            raise ValueError("opaque diagnostic reference fields must be strings")
        ref = cls(
            ref_id=ref_id,
            evaluation_status=evaluation_status,
            reason_code=reason_code,
        )
        if ref.to_dict() != dict(payload):
            raise ValueError("opaque diagnostic reference is not canonical")
        return ref


@dataclass(frozen=True, slots=True)
class OpaqueDiagnosticRefSet:
    case_digest: str
    refs: tuple[tuple[str, OpaqueDiagnosticRef], ...]
    authority_scope: str = field(default=DIAGNOSTIC_AUTHORITY_SCOPE, init=False)
    authority_eligible: bool = field(default=False, init=False)
    schema_version: str = field(
        default=OPAQUE_DIAGNOSTIC_REF_SET_SCHEMA_VERSION, init=False
    )

    def __post_init__(self) -> None:
        backends = tuple(backend for backend, _ref in self.refs)
        if backends != tuple(sorted(backends)) or len(backends) != len(set(backends)):
            raise ValueError("diagnostic reference backends must be unique and sorted")
        if any(
            not isinstance(backend, str)
            or not backend
            or not isinstance(ref, OpaqueDiagnosticRef)
            for backend, ref in self.refs
        ):
            raise ValueError("invalid diagnostic reference set entry")
        if any(
            ref.evaluation_status == DIAGNOSTIC_EVALUATED
            for _backend, ref in self.refs
        ) and not self.case_digest:
            raise ValueError("evaluated diagnostic references require a case binding")
        if self.authority_scope != DIAGNOSTIC_AUTHORITY_SCOPE or self.authority_eligible:
            raise ValueError("diagnostic reference set cannot carry authority")

    @property
    def evaluation_status(self) -> str:
        evaluated = sum(
            ref.evaluation_status == DIAGNOSTIC_EVALUATED for _backend, ref in self.refs
        )
        if self.refs and evaluated == len(self.refs):
            return DIAGNOSTIC_EVALUATED
        if evaluated:
            return DIAGNOSTIC_PARTIAL
        return DIAGNOSTIC_NOT_EVALUATED

    @property
    def ref_set_id(self) -> str:
        return stable_digest(
            "legacy-diagnostic-ref-set",
            {
                "case_digest": self.case_digest,
                "refs": [(backend, ref.to_dict()) for backend, ref in self.refs],
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "ref_set_id": self.ref_set_id,
            "case_digest": self.case_digest,
            "evaluation_status": self.evaluation_status,
            "authority_scope": self.authority_scope,
            "authority_eligible": self.authority_eligible,
            "refs": [
                {"backend": backend, "ref": ref.to_dict()}
                for backend, ref in self.refs
            ],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "OpaqueDiagnosticRefSet":
        if set(payload) != _REF_SET_FIELDS:
            raise ValueError("opaque diagnostic reference set fields do not match")
        if payload.get("schema_version") != OPAQUE_DIAGNOSTIC_REF_SET_SCHEMA_VERSION:
            raise ValueError("unsupported opaque diagnostic reference set schema")
        if payload.get("authority_scope") != DIAGNOSTIC_AUTHORITY_SCOPE:
            raise ValueError("opaque diagnostic reference set scope mismatch")
        if payload.get("authority_eligible") is not False:
            raise ValueError("opaque diagnostic reference set claims authority")
        rows = payload.get("refs")
        if not isinstance(rows, list):
            raise ValueError("opaque diagnostic references must be a list")
        refs: list[tuple[str, OpaqueDiagnosticRef]] = []
        for row in rows:
            if not isinstance(row, Mapping) or set(row) != {"backend", "ref"}:
                raise ValueError("invalid opaque diagnostic reference row")
            backend_value = row.get("backend")
            if not isinstance(backend_value, str) or not backend_value:
                raise ValueError("opaque diagnostic backend must be a non-empty string")
            backend = backend_value
            ref_payload = row.get("ref")
            if not isinstance(ref_payload, Mapping):
                raise ValueError("invalid opaque diagnostic reference payload")
            refs.append((backend, OpaqueDiagnosticRef.from_dict(ref_payload)))
        case_digest = payload.get("case_digest")
        if not isinstance(case_digest, str):
            raise ValueError("opaque diagnostic case binding must be a string")
        result = cls(case_digest=case_digest, refs=tuple(refs))
        if result.to_dict() != dict(payload):
            raise ValueError("opaque diagnostic reference set is not canonical")
        return result


def _context_from_legacy_metadata(
    *,
    backend: str,
    case_digest: str,
    case_payload: Mapping[str, Any],
    raw_result: Any,
    target_payload: Mapping[str, Any],
    environment: Mapping[str, Any],
    optimizer_config: Mapping[str, Any],
    required_capabilities: Sequence[str],
) -> LegacyDiagnosticContext:
    backend = _require_text(backend, "missing_backend_identity")
    case_digest = _require_text(case_digest, "missing_case_digest")
    case_mapping = _mapping(case_payload, "missing_case_payload")
    try:
        recomputed_case_digest = legacy_stable_digest("case", case_mapping)
    except (TypeError, ValueError) as exc:
        raise _NotEvaluated("invalid_case_payload") from exc
    if recomputed_case_digest != case_digest:
        raise _NotEvaluated("case_digest_mismatch")
    raw = _summary_mapping(raw_result, "missing_raw_result")
    if raw.get("backend") != backend:
        raise _NotEvaluated("raw_backend_mismatch")
    if target_payload.get("backend") != backend:
        raise _NotEvaluated("target_backend_mismatch")

    try:
        trusted_target_payload = target_spec(backend).to_dict()
    except (TypeError, ValueError) as exc:
        raise _NotEvaluated("unregistered_target_backend") from exc
    if to_primitive(dict(target_payload)) != to_primitive(trusted_target_payload):
        raise _NotEvaluated("target_registry_mismatch")
    target_payload = trusted_target_payload

    model = _capability_model(target_payload.get("capability_model"))
    target_capabilities = tuple(target_payload.get("capabilities", ()))
    if set(target_capabilities) != set(model.legacy_tokens):
        raise _NotEvaluated("target_capability_projection_mismatch")
    if len(model.execution_modes) != 1:
        raise _NotEvaluated("ambiguous_execution_mode")
    execution_mode = model.execution_modes[0]
    physical_layout = _require_text(
        raw.get("input_physical_layout"), "missing_physical_layout"
    )

    version = _backend_version(target_payload=target_payload, environment=environment)
    source_tree_digest = _require_text(
        environment.get("source_tree_sha256"), "missing_adapter_source_digest"
    )
    adapter = _require_text(target_payload.get("adapter"), "missing_adapter_identity")
    adapter_revision = stable_digest(
        "legacy-adapter-revision",
        {"adapter": adapter, "source_tree_sha256": source_tree_digest},
    )
    config = _validated_optimizer_config(optimizer_config)

    required = frozenset(
        _require_text(item, "invalid_required_capability")
        for item in required_capabilities
    )
    required = frozenset(
        {
            *required,
            f"execution_mode:{execution_mode}",
            f"physical_layout:{physical_layout}",
        }
    )
    actual = frozenset(
        {
            *model.legacy_tokens,
            *(f"execution_mode:{item}" for item in model.execution_modes),
            *(f"physical_layout:{item}" for item in model.physical_layouts),
        }
    )
    expected_legacy_decision = model.decision(
        required_tokens=tuple(
            item
            for item in required
            if not item.startswith("execution_mode:")
            and not item.startswith("physical_layout:")
        ),
        execution_mode=execution_mode,
        physical_layout=physical_layout,
    ).to_dict()
    endpoint_scope = {
        "case_digest": case_digest,
        "backend": backend,
        "backend_version": version,
        "adapter_revision": adapter_revision,
        "execution_mode": execution_mode,
        "physical_layout": physical_layout,
        "optimizer_config": config,
        "capabilities": sorted(actual),
    }
    endpoint = Endpoint.build(
        endpoint_id=stable_digest("legacy-diagnostic-endpoint", endpoint_scope),
        case_digest=case_digest,
        backend=backend,
        backend_version=version,
        adapter_revision=adapter_revision,
        execution_mode=execution_mode,
        physical_layout=physical_layout,
        optimizer_config=config,
        capabilities=actual,
    )
    return LegacyDiagnosticContext(
        endpoint=endpoint,
        required_capabilities=required,
        capability_model_digest=model.digest,
        expected_legacy_capability_decision=frozen_pairs(
            expected_legacy_decision
        ),
    )


def _failure_diagnostic(
    *,
    context: LegacyDiagnosticContext,
    raw: Mapping[str, Any],
    status: ExecutionStatus,
    failure_kind: FailureKind,
    unsupported_evidence: tuple[UnsupportedEvidence, ...] = (),
) -> _LegacyTypedDiagnostic:
    reason_value = raw.get("error", "")
    error_type_value = raw.get("error_type", "")
    if not isinstance(reason_value, str) or not isinstance(error_type_value, str):
        raise _NotEvaluated("invalid_legacy_error_payload")
    reason = reason_value[:500]
    error_type = error_type_value
    unsupported_evidence_digest = (
        unsupported_evidence[0].digest if unsupported_evidence else ""
    )
    observation = Observation.build(
        endpoint_id=context.endpoint.endpoint_id,
        status=status.value,
        error_category=failure_kind.value,
        error_type=error_type,
        error_message=reason,
        execution_metadata={
            "authority_scope": DIAGNOSTIC_AUTHORITY_SCOPE,
            "context_digest": context.digest,
            "endpoint_digest": context.endpoint.digest,
        },
    )
    outcome = StructuredExecutionOutcome(
        endpoint_id=context.endpoint.endpoint_id,
        status=status,
        failure_kind=failure_kind,
        reason=reason,
        unsupported_evidence_digest=unsupported_evidence_digest,
    )
    return _LegacyTypedDiagnostic(
        endpoint=context.endpoint,
        observation=observation,
        outcome=outcome,
        context_digest=context.digest,
        unsupported_evidence=unsupported_evidence,
    )


def _translate_legacy_result(
    context: LegacyDiagnosticContext,
    raw_result: Any,
    normalized_result: Any,
) -> _LegacyTypedDiagnostic:
    """Translate for diagnostics only; callers outside this module receive a ref."""

    raw = _summary_mapping(raw_result, "missing_raw_result")
    normalized = _mapping(normalized_result, "missing_normalized_result")
    backend = context.endpoint.backend
    if raw.get("backend") != backend or normalized.get("backend") != backend:
        raise _NotEvaluated("result_backend_mismatch")
    raw_status_value = raw.get("status")
    normalized_status_value = normalized.get("status")
    if not isinstance(raw_status_value, str) or not isinstance(
        normalized_status_value, str
    ):
        raise _NotEvaluated("invalid_legacy_status_type")
    raw_status = raw_status_value
    normalized_status = normalized_status_value

    if normalized_status == "normalization_error":
        if raw_status != "ok":
            raise _NotEvaluated("raw_normalized_status_mismatch")
        return _failure_diagnostic(
            context=context,
            raw={
                "error": normalized.get("error", ""),
                "error_type": normalized.get("error_type", ""),
            },
            status=ExecutionStatus.ADAPTER_ERROR,
            failure_kind=FailureKind.ADAPTER_ERROR,
        )

    if raw_status not in {"ok", "error", "timeout", "missing"}:
        raise _NotEvaluated("unsupported_legacy_status")
    if normalized_status != raw_status:
        raise _NotEvaluated("raw_normalized_status_mismatch")

    if raw_status == "error":
        return _failure_diagnostic(
            context=context,
            raw=raw,
            status=ExecutionStatus.ADAPTER_ERROR,
            failure_kind=FailureKind.ADAPTER_ERROR,
        )
    if raw_status == "timeout":
        return _failure_diagnostic(
            context=context,
            raw=raw,
            status=ExecutionStatus.TIMEOUT,
            failure_kind=FailureKind.TIMEOUT,
        )
    if raw_status == "missing":
        expected_legacy = to_primitive(
            pairs_dict(context.expected_legacy_capability_decision)
        )
        supplied_legacy = raw.get("capability_decision")
        capability = decide_capabilities(
            context.endpoint, context.required_capabilities
        )
        if (
            isinstance(supplied_legacy, Mapping)
            and to_primitive(dict(supplied_legacy)) == expected_legacy
            and not bool(expected_legacy.get("supported"))
            and capability.valid
            and not capability.supported
        ):
            return _failure_diagnostic(
                context=context,
                raw=raw,
                status=ExecutionStatus.UNSUPPORTED,
                failure_kind=FailureKind.UNSUPPORTED_CAPABILITY,
                unsupported_evidence=capability.evidence,
            )
        return _failure_diagnostic(
            context=context,
            raw=raw,
            status=ExecutionStatus.MISSING,
            failure_kind=FailureKind.MISSING_RESULT,
        )
    # The legacy normalizer records logical type strings and tagged values but
    # does not retain a trusted output-schema nullability declaration or the
    # original presentation order for every normalization mode.  Caller-authored
    # replacements are not exact typed context, so OK must remain unevaluated
    # until an execution-owned provenance producer exists.
    raise _NotEvaluated("legacy_ok_exact_schema_unavailable")


def not_evaluated_diagnostic_ref_set(
    backends: Sequence[str] | None = (),
    *,
    case_digest: str = "",
    reason_code: str = "missing_diagnostic_refs",
) -> dict[str, Any]:
    safe_case_digest = _safe_text(case_digest)
    safe_reason_code = _safe_text(reason_code, "invalid_diagnostic_context")
    refs = tuple(
        (
            backend,
            OpaqueDiagnosticRef.not_evaluated(
                backend=backend,
                case_digest=safe_case_digest,
                reason_code=safe_reason_code,
            ),
        )
        for backend in _backend_names(backends)
    )
    return OpaqueDiagnosticRefSet(
        case_digest=safe_case_digest,
        refs=refs,
    ).to_dict()


def build_legacy_diagnostic_ref_set(
    *,
    backends: Sequence[str],
    case_digest: str,
    case_payload: Mapping[str, Any] | None = None,
    raw_results: Mapping[str, Any],
    normalized_results: Mapping[str, Any],
    target_specs: Sequence[Mapping[str, Any]],
    environment: Mapping[str, Any],
    optimizer_config: Mapping[str, Any],
    required_capabilities: Sequence[str] | None,
    schema_nullability_by_backend: Mapping[str, Sequence[bool]] | None = None,
    row_order_provenance_by_backend: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build deterministic diagnostic refs without returning typed authority objects."""

    ordered_backends = _backend_names(backends)
    target_rows: dict[str, list[Mapping[str, Any]]] = {
        backend: [] for backend in ordered_backends
    }
    if isinstance(target_specs, Sequence) and not isinstance(
        target_specs, (str, bytes)
    ):
        try:
            target_values: Sequence[Any] = tuple(target_specs)
        except Exception:
            target_values = ()
    else:
        target_values = ()
    for target in target_values:
        try:
            if isinstance(target, Mapping):
                backend_value = target.get("backend")
                if not isinstance(backend_value, str) or not backend_value:
                    continue
                backend = backend_value
                if backend in target_rows:
                    target_rows[backend].append(target)
        except Exception:
            # A hostile Mapping implementation is malformed context, not an
            # alternate source of target identity.
            continue

    refs: list[tuple[str, OpaqueDiagnosticRef]] = []
    for backend in ordered_backends:
        try:
            if required_capabilities is None:
                raise _NotEvaluated("missing_required_capabilities")
            if isinstance(required_capabilities, (str, bytes)):
                raise _NotEvaluated("invalid_required_capabilities")
            if len(target_rows[backend]) != 1:
                raise _NotEvaluated("missing_or_ambiguous_target_spec")
            if backend not in raw_results:
                raise _NotEvaluated("missing_raw_result")
            if backend not in normalized_results:
                raise _NotEvaluated("missing_normalized_result")
            context = _context_from_legacy_metadata(
                backend=backend,
                case_digest=case_digest,
                case_payload=case_payload,
                raw_result=raw_results[backend],
                target_payload=target_rows[backend][0],
                environment=environment,
                optimizer_config=optimizer_config,
                required_capabilities=required_capabilities,
            )
            diagnostic = _translate_legacy_result(
                context,
                raw_results[backend],
                normalized_results[backend],
            )
            ref = OpaqueDiagnosticRef.evaluated(diagnostic)
        except _NotEvaluated as exc:
            ref = OpaqueDiagnosticRef.not_evaluated(
                backend=backend,
                case_digest=case_digest,
                reason_code=exc.reason_code,
            )
        except Exception:  # fail closed at the untrusted legacy payload boundary
            ref = OpaqueDiagnosticRef.not_evaluated(
                backend=backend,
                case_digest=case_digest,
                reason_code="invalid_typed_context",
            )
        refs.append((backend, ref))
    bound_case_digest = _safe_text(case_digest)
    return OpaqueDiagnosticRefSet(
        case_digest=bound_case_digest,
        refs=tuple(refs),
    ).to_dict()


def consume_opaque_diagnostic_ref_set(
    payload: Any,
    *,
    expected_backends: Sequence[str] | None = (),
    case_digest: str = "",
) -> dict[str, Any]:
    """Validate/copy a ref set; malformed or authority-claiming input fails closed."""

    if not isinstance(payload, Mapping):
        return not_evaluated_diagnostic_ref_set(
            expected_backends,
            case_digest=case_digest,
            reason_code="missing_diagnostic_refs",
        )
    try:
        parsed = OpaqueDiagnosticRefSet.from_dict(payload)
        expected = _backend_names(expected_backends)
        if not isinstance(case_digest, str) or not case_digest:
            return not_evaluated_diagnostic_ref_set(
                expected,
                case_digest="",
                reason_code="missing_expected_case_context",
            )
        if not expected:
            return not_evaluated_diagnostic_ref_set(
                (),
                case_digest=case_digest,
                reason_code="missing_expected_backend_context",
            )
        if parsed.case_digest != case_digest:
            raise ValueError("diagnostic reference case binding mismatch")
        observed = tuple(backend for backend, _ref in parsed.refs)
        if observed != expected:
            raise ValueError("diagnostic reference backend set mismatch")
        # An opaque serialized reference deliberately omits its typed diagnostic,
        # so an offline consumer cannot authenticate an ``evaluated`` claim.  It
        # preserves the reference identity but downgrades that claim.  This keeps
        # malformed or invented refs from becoming execution facts while still
        # allowing diagnostic correlation.
        consumer_refs = tuple(
            (
                backend,
                (
                    OpaqueDiagnosticRef(
                        ref_id=ref.ref_id,
                        evaluation_status=DIAGNOSTIC_NOT_EVALUATED,
                        reason_code="opaque_reference_not_revalidated",
                    )
                    if ref.evaluation_status == DIAGNOSTIC_EVALUATED
                    else ref
                ),
            )
            for backend, ref in parsed.refs
        )
        return OpaqueDiagnosticRefSet(
            case_digest=parsed.case_digest,
            refs=consumer_refs,
        ).to_dict()
    except Exception:  # public untrusted payload boundary must never escape
        return not_evaluated_diagnostic_ref_set(
            expected_backends,
            case_digest=case_digest,
            reason_code="invalid_diagnostic_refs",
        )


__all__ = [
    "DIAGNOSTIC_AUTHORITY_SCOPE",
    "DIAGNOSTIC_EVALUATED",
    "DIAGNOSTIC_NOT_EVALUATED",
    "DIAGNOSTIC_PARTIAL",
    "build_legacy_diagnostic_ref_set",
    "consume_opaque_diagnostic_ref_set",
    "not_evaluated_diagnostic_ref_set",
]
