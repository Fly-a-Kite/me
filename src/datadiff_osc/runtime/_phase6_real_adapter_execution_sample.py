"""Private, diagnostic-only capture of one real adapter-pair execution.

The capture deliberately has a much narrower scope than a Phase-6 evidence
producer.  It accepts a fully validated private R1 Search-to-Runtime binding,
re-derives its declaration-generated fresh case and pair, invokes the existing
direct serial execution entry point exactly once, and retains only ephemeral
canonical summaries in memory.  It has no source snapshot, raw artifact,
ledger event, provenance receipt, candidate promotion, or gate authority.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping

from datadiff.backends import make_backend
from datadiff.config import ExperimentConfig
from datadiff.execution import execute_case
from datadiff.family_witness_registry import latest_family_witness_registrations
from datadiff_osc._canonical import canonical_json, stable_digest
from datadiff_osc.runtime._phase6_reachability_execution_binding import (
    ReachabilityExecutionBinding,
)
from datadiff_osc.semantic_targets.compiler import compile_target_universe
from datadiff_osc.semantic_targets.declarations import legacy_v4_target_templates


REAL_ADAPTER_EXECUTION_SAMPLE_SCHEMA_VERSION = (
    "osc-private-real-adapter-execution-sample-v1"
)
REAL_ADAPTER_EXECUTION_RECORD_SCHEMA_VERSION = (
    "osc-private-real-adapter-execution-record-v1"
)
DIRECT_SERIAL_UNCACHED_EXECUTION_MODE = "direct-serial-uncached-execute-case-v1"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError(f"{name} must be non-empty text")
    return value


def _require_sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _canonical_mapping(payload: object, *, name: str) -> dict[str, Any]:
    if not isinstance(payload, str) or not payload:
        raise ValueError(f"{name} must be non-empty canonical JSON")
    try:
        decoded = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} is malformed canonical JSON") from exc
    if not isinstance(decoded, dict):
        raise ValueError(f"{name} must decode to a mapping")
    if canonical_json(decoded) != payload:
        raise ValueError(f"{name} is not canonical JSON")
    return decoded


def _adapter_type(adapter: object) -> str:
    adapter_type = type(adapter)
    return f"{adapter_type.__module__}:{adapter_type.__qualname__}"


def _revalidate_binding(
    binding: ReachabilityExecutionBinding,
) -> ReachabilityExecutionBinding:
    """Force all R1 context validation before any real adapter is constructed."""

    if not isinstance(binding, ReachabilityExecutionBinding):
        raise TypeError("real adapter capture requires ReachabilityExecutionBinding")
    try:
        rebuilt = ReachabilityExecutionBinding(
            scheduled_attempt=binding.scheduled_attempt,
            observation_context=binding.observation_context,
            canonical_case=binding.canonical_case,
            construction=binding.construction,
            extraction=binding.extraction,
            activation=binding.activation,
            backend_pair=binding.backend_pair,
            contract=binding.contract,
            endpoints=binding.endpoints,
            applicability=binding.applicability,
            results=binding.results,
            evidence=binding.evidence,
            observation=binding.observation,
            schema_version=binding.schema_version,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("real adapter capture context is not exact R1 context") from exc
    if rebuilt != binding:
        raise ValueError("real adapter capture context changed during revalidation")
    return rebuilt


def _derive_execution_input(
    binding: ReachabilityExecutionBinding,
) -> tuple[tuple[str, str], object, object]:
    """Recompute the one fresh generated Case and exact backend pair."""

    checked = _revalidate_binding(binding)
    universe = compile_target_universe(legacy_v4_target_templates())
    cell = next(
        (
            item
            for item in universe.fresh_cells
            if item.target_cell_id == checked.scheduled_attempt.target_cell_id
        ),
        None,
    )
    if cell is None:
        raise ValueError("real adapter capture requires a current fresh target cell")
    if (
        cell.test_family_id != checked.scheduled_attempt.target_family_id
        or cell.target_cell_id != checked.backend_pair.target_cell_id
        or cell.observation_contract != checked.backend_pair.observation_contract
    ):
        raise ValueError("real adapter capture fresh cell/pair context mismatch")

    expected_backends = (
        checked.backend_pair.target_backend,
        checked.backend_pair.control_backend,
    )
    if tuple(endpoint.backend for endpoint in checked.endpoints) != expected_backends:
        raise ValueError("real adapter capture endpoint order does not match pair")
    registrations = {
        item.family_id: item for item in latest_family_witness_registrations()
    }
    registration = registrations.get(cell.test_family_id)
    if registration is None:
        raise ValueError("real adapter capture registration is unavailable")
    if tuple(registration.backends) != expected_backends:
        raise ValueError("real adapter capture registration/backend pair mismatch")
    expected_case = registration.generate_case(cell.construction_index)
    expected_case_json = canonical_json(expected_case).encode("utf-8")
    if checked.canonical_case.canonical_case_json != expected_case_json:
        raise ValueError("real adapter capture case is not exact generated fresh case")
    if checked.canonical_case.source_case_id != expected_case.case_id:
        raise ValueError("real adapter capture canonical case identity mismatch")
    return expected_backends, expected_case, registration


def _close_adapters(adapters: tuple[object, ...]) -> None:
    """Close every constructed adapter and fail closed if any close fails."""

    failures: list[BaseException] = []
    for adapter in adapters:
        try:
            close = getattr(adapter, "close", None)
            if not callable(close):
                raise TypeError("adapter close is unavailable")
            close()
        except BaseException as exc:  # fail closed even for unusual adapter failures
            failures.append(exc)
    if failures:
        raise ValueError("real adapter close failed") from failures[0]


@dataclass(frozen=True, slots=True)
class RealAdapterExecutionRecord:
    """Canonical in-memory summary for one endpoint of the direct execution."""

    backend: str
    adapter_type: str
    raw_summary_json: str
    raw_summary_sha256: str
    normalized_result_json: str
    normalized_result_sha256: str
    schema_version: str = REAL_ADAPTER_EXECUTION_RECORD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_text(self.backend, name="real adapter record backend")
        _require_text(self.adapter_type, name="real adapter record adapter type")
        if self.schema_version != REAL_ADAPTER_EXECUTION_RECORD_SCHEMA_VERSION:
            raise ValueError("real adapter record schema version mismatch")
        _require_sha256(self.raw_summary_sha256, name="real adapter raw summary SHA")
        _require_sha256(
            self.normalized_result_sha256,
            name="real adapter normalized result SHA",
        )
        raw = _canonical_mapping(
            self.raw_summary_json,
            name="real adapter raw summary",
        )
        normalized = _canonical_mapping(
            self.normalized_result_json,
            name="real adapter normalized result",
        )
        if _sha256(self.raw_summary_json.encode("utf-8")) != self.raw_summary_sha256:
            raise ValueError("real adapter raw summary SHA mismatch")
        if (
            _sha256(self.normalized_result_json.encode("utf-8"))
            != self.normalized_result_sha256
        ):
            raise ValueError("real adapter normalized result SHA mismatch")
        if raw.get("backend") != self.backend:
            raise ValueError("real adapter raw summary backend mismatch")
        if normalized.get("backend") != self.backend:
            raise ValueError("real adapter normalized result backend mismatch")
        if raw.get("status") != "ok" or normalized.get("status") != "ok":
            raise ValueError("real adapter record status is not OK")
        if raw.get("error_type", "") or raw.get("error", ""):
            raise ValueError("real adapter raw summary carries an error")
        if normalized.get("error_type", "") or normalized.get("error", ""):
            raise ValueError("real adapter normalized result carries an error")

    @property
    def digest(self) -> str:
        return stable_digest("osc-private-real-adapter-execution-record", self)


@dataclass(frozen=True, slots=True)
class RealAdapterExecutionSample:
    """One exact real adapter-pair sample with permanent diagnostic-only scope."""

    binding: ReachabilityExecutionBinding
    execution_mode: str
    adapter_records: tuple[RealAdapterExecutionRecord, ...]
    schema_version: str = REAL_ADAPTER_EXECUTION_SAMPLE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.binding, ReachabilityExecutionBinding):
            raise TypeError("real adapter sample requires ReachabilityExecutionBinding")
        if self.schema_version != REAL_ADAPTER_EXECUTION_SAMPLE_SCHEMA_VERSION:
            raise ValueError("real adapter sample schema version mismatch")
        if self.execution_mode != DIRECT_SERIAL_UNCACHED_EXECUTION_MODE:
            raise ValueError("real adapter sample execution mode mismatch")
        if (
            not isinstance(self.adapter_records, tuple)
            or len(self.adapter_records) != 2
            or any(
                not isinstance(item, RealAdapterExecutionRecord)
                for item in self.adapter_records
            )
        ):
            raise ValueError("real adapter sample requires exactly two typed records")
        checked = _revalidate_binding(self.binding)
        expected_backends, _case, _registration = _derive_execution_input(checked)
        observed_backends = tuple(item.backend for item in self.adapter_records)
        if observed_backends != expected_backends:
            raise ValueError("real adapter sample record order does not match pair")
        if len(set(observed_backends)) != len(observed_backends):
            raise ValueError("real adapter sample records must be unique")
        if (
            self.binding.authority_eligible
            or self.binding.gate_credit
            or self.binding.bug_claimed
        ):
            raise ValueError("real adapter sample cannot wrap authoritative R1 context")

    @property
    def sample_id(self) -> str:
        return stable_digest(
            "osc-private-real-adapter-execution-sample-id-v1",
            {
                "binding_digest": self.binding.digest,
                "execution_mode": self.execution_mode,
                "record_digests": tuple(item.digest for item in self.adapter_records),
            },
        )

    @property
    def digest(self) -> str:
        return stable_digest("osc-private-real-adapter-execution-sample", self)

    @property
    def authority_eligible(self) -> bool:
        return False

    @property
    def gate_credit(self) -> bool:
        return False

    @property
    def raw_artifact_admitted(self) -> bool:
        return False

    @property
    def coverage_event_created(self) -> bool:
        return False

    @property
    def candidate_confirmed(self) -> bool:
        return False

    @property
    def bug_claimed(self) -> bool:
        return False


def _record_from_execution(
    *,
    backend: str,
    adapter: object,
    raw_summary: Mapping[str, Any],
    normalized_result: object,
) -> RealAdapterExecutionRecord:
    if not isinstance(raw_summary, Mapping):
        raise ValueError("real adapter raw summary must be a mapping")
    to_dict = getattr(normalized_result, "to_dict", None)
    if not callable(to_dict):
        raise ValueError("real adapter normalized result is unavailable")
    normalized_payload = to_dict()
    if not isinstance(normalized_payload, Mapping):
        raise ValueError("real adapter normalized result must be a mapping")
    raw_summary_json = canonical_json(raw_summary)
    normalized_result_json = canonical_json(normalized_payload)
    return RealAdapterExecutionRecord(
        backend=backend,
        adapter_type=_adapter_type(adapter),
        raw_summary_json=raw_summary_json,
        raw_summary_sha256=_sha256(raw_summary_json.encode("utf-8")),
        normalized_result_json=normalized_result_json,
        normalized_result_sha256=_sha256(normalized_result_json.encode("utf-8")),
    )


def capture_real_adapter_execution_sample(
    binding: ReachabilityExecutionBinding,
) -> RealAdapterExecutionSample:
    """Run the only permitted direct pair execution for this private sample.

    There is intentionally no caller supplied adapter, cache, session, raw
    artifact path, source snapshot, or authority flag.  A successful return is
    still diagnostic-only and cannot advance any Phase-6 gate.
    """

    expected_backends, case, registration = _derive_execution_input(binding)
    adapters: dict[str, object] = {}
    try:
        for backend_name in expected_backends:
            adapter = make_backend(backend_name)
            if getattr(adapter, "name", None) != backend_name:
                raise ValueError("real adapter identity does not match backend pair")
            adapters[backend_name] = adapter
        raw_results, normalized_results = execute_case(
            case,
            list(expected_backends),
            ExperimentConfig(method_arm=registration.method_arm_id),
            backend_instances=adapters,
            parallel=False,
        )
        if (
            not isinstance(raw_results, Mapping)
            or not isinstance(normalized_results, Mapping)
            or set(raw_results) != set(expected_backends)
            or set(normalized_results) != set(expected_backends)
        ):
            raise ValueError("real adapter execution result endpoints mismatch")
        records = tuple(
            _record_from_execution(
                backend=backend_name,
                adapter=adapters[backend_name],
                raw_summary=raw_results[backend_name],
                normalized_result=normalized_results[backend_name],
            )
            for backend_name in expected_backends
        )
    finally:
        _close_adapters(tuple(adapters.values()))
    return RealAdapterExecutionSample(
        binding=_revalidate_binding(binding),
        execution_mode=DIRECT_SERIAL_UNCACHED_EXECUTION_MODE,
        adapter_records=records,
    )
