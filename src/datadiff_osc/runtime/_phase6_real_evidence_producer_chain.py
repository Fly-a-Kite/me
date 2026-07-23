"""Private, fail-closed staging for future Phase-6 process evidence.

This module is deliberately diagnostic-only.  It can bind one observed local
subprocess transcript to one exact typed execution result, but it never writes
an artifact, producer receipt, dynamic plan, receipt index, authority bundle,
or gate result.  A later, separately authorized execution authority must add
the clean-source snapshot and durable output boundary before any record can be
considered for Phase-6 evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time
from typing import Any

from datadiff_osc._canonical import (
    assert_deeply_immutable,
    canonical_json,
    stable_digest,
)
from datadiff_osc.runtime._receipt_producers import TypedExecutionResult
from datadiff_osc.schemas import EvidenceEnvelope, EvidenceState


REAL_EVIDENCE_DIAGNOSTIC_CAPTURE_SCHEMA_VERSION = (
    "osc-private-phase6-real-evidence-diagnostic-capture-v1"
)
REAL_EVIDENCE_DIAGNOSTIC_MODE = "diagnostic-process-bound-no-authority-v1"
_ALLOWED_ENVIRONMENT_KEYS = frozenset(
    {"LANG", "LC_ALL", "PATH", "PYTHONDONTWRITEBYTECODE", "TZ"}
)
_FORBIDDEN_EVIDENCE_STATES = frozenset(
    {EvidenceState.UNIQUE_ROOT, EvidenceState.INDEPENDENTLY_CONFIRMED}
)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError(f"{name} must be non-empty text without NUL")
    return value


def _require_sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _require_nonnegative_int(value: object, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _require_immutable_bytes(value: object, *, name: str) -> bytes:
    if not isinstance(value, bytes):
        raise ValueError(f"{name} must be immutable bytes")
    return value


def _normalized_environment(
    value: object,
) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, tuple):
        raise ValueError("capture environment must be an immutable tuple")
    fields: list[tuple[str, str]] = []
    for item in value:
        if (
            not isinstance(item, tuple)
            or len(item) != 2
            or not isinstance(item[0], str)
            or not isinstance(item[1], str)
        ):
            raise ValueError("capture environment fields must be text pairs")
        key = _require_text(item[0], name="capture environment key")
        field_value = _require_text(item[1], name="capture environment value")
        if key not in _ALLOWED_ENVIRONMENT_KEYS:
            raise ValueError(f"capture environment key is not allowlisted: {key}")
        fields.append((key, field_value))
    if not fields or tuple(fields) != tuple(sorted(fields)):
        raise ValueError("capture environment fields must be non-empty and sorted")
    if len({key for key, _ in fields}) != len(fields):
        raise ValueError("capture environment keys must be unique")
    return tuple(fields)


def _normalized_command(value: object) -> tuple[str, ...]:
    if not isinstance(value, tuple) or not value:
        raise ValueError("capture command must be a non-empty immutable tuple")
    return tuple(_require_text(item, name="capture command argument") for item in value)


def _resolved_directory(value: object, *, name: str) -> Path:
    if isinstance(value, Path):
        path = value
    else:
        path = Path(_require_text(value, name=name))
    if not path.is_absolute():
        raise ValueError(f"{name} must be absolute")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"{name} cannot be resolved") from exc
    if not resolved.is_dir():
        raise ValueError(f"{name} must resolve to a directory")
    return resolved


def _resolved_executable(value: object) -> Path:
    text = _require_text(value, name="capture executable")
    path = Path(text)
    if not path.is_absolute():
        raise ValueError("capture executable must be an absolute path")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ValueError("capture executable cannot be resolved") from exc
    if not resolved.is_file():
        raise ValueError("capture executable must resolve to a regular file")
    return resolved


def _reject_synthetic_marker(*values: str) -> None:
    if any("synthetic" in value.lower() for value in values):
        raise ValueError("synthetic marker is not admissible for real evidence staging")


@dataclass(frozen=True, slots=True)
class ProcessTranscript:
    """Immutable transcript for one observed diagnostic subprocess.

    The transcript is intentionally unable to represent a persisted authority
    artifact.  Its output remains in memory and can only be bound to a
    diagnostic staging result below.
    """

    source_digest: str
    source_root: str
    cwd: str
    command: tuple[str, ...]
    executable_sha256: str
    environment: tuple[tuple[str, str], ...]
    started_ns: int
    ended_ns: int
    exit_code: int
    stdout: bytes
    stderr: bytes
    schema_version: str = REAL_EVIDENCE_DIAGNOSTIC_CAPTURE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_text(self.source_digest, name="capture source digest")
        _reject_synthetic_marker(self.source_digest)
        source_root = _resolved_directory(self.source_root, name="capture source root")
        cwd = _resolved_directory(self.cwd, name="capture cwd")
        try:
            cwd.relative_to(source_root)
        except ValueError as exc:
            raise ValueError("capture cwd escapes the declared source root") from exc
        command = _normalized_command(self.command)
        executable = _resolved_executable(command[0])
        if command[0] != str(executable):
            raise ValueError("capture command executable is not canonical")
        if _sha256(executable.read_bytes()) != self.executable_sha256:
            raise ValueError("capture executable SHA-256 mismatch")
        _normalized_environment(self.environment)
        started = _require_nonnegative_int(self.started_ns, name="capture start time")
        ended = _require_nonnegative_int(self.ended_ns, name="capture end time")
        if ended < started:
            raise ValueError("capture end time precedes start time")
        if not isinstance(self.exit_code, int) or isinstance(self.exit_code, bool):
            raise ValueError("capture exit code must be an integer")
        _require_immutable_bytes(self.stdout, name="capture stdout")
        _require_immutable_bytes(self.stderr, name="capture stderr")
        if self.schema_version != REAL_EVIDENCE_DIAGNOSTIC_CAPTURE_SCHEMA_VERSION:
            raise ValueError("capture schema version mismatch")
        assert_deeply_immutable(self)

    @property
    def command_digest(self) -> str:
        return stable_digest(
            "osc-private-phase6-diagnostic-command",
            {"command": self.command, "environment": self.environment},
        )

    @property
    def stdout_sha256(self) -> str:
        return _sha256(self.stdout)

    @property
    def stderr_sha256(self) -> str:
        return _sha256(self.stderr)

    @property
    def output_sha256(self) -> str:
        return self.stdout_sha256

    @property
    def digest(self) -> str:
        return stable_digest("osc-private-phase6-diagnostic-process-transcript", self)

    @property
    def diagnostic_only(self) -> bool:
        return True

    @property
    def authority_eligible(self) -> bool:
        return False

    @property
    def gate_credit(self) -> bool:
        return False


def capture_diagnostic_process(
    *,
    source_digest: str,
    source_root: Path,
    cwd: Path,
    command: tuple[str, ...],
    environment: tuple[tuple[str, str], ...],
    timeout_seconds: float = 30.0,
) -> ProcessTranscript:
    """Capture one bounded local command without creating durable evidence.

    ``shell`` is never used.  This helper is intentionally diagnostic-only and
    has a one-minute hard timeout so it cannot become a hidden long-running
    Phase-6 launcher.
    """

    source_text = _require_text(source_digest, name="capture source digest")
    _reject_synthetic_marker(source_text)
    source_directory = _resolved_directory(source_root, name="capture source root")
    cwd_directory = _resolved_directory(cwd, name="capture cwd")
    try:
        cwd_directory.relative_to(source_directory)
    except ValueError as exc:
        raise ValueError("capture cwd escapes the declared source root") from exc
    normalized_command = _normalized_command(command)
    executable = _resolved_executable(normalized_command[0])
    normalized_command = (str(executable), *normalized_command[1:])
    normalized_environment = _normalized_environment(environment)
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(float(timeout_seconds))
        or not 0.0 < float(timeout_seconds) <= 60.0
    ):
        raise ValueError("diagnostic capture timeout must be in (0, 60] seconds")
    started_ns = time.monotonic_ns()
    try:
        completed = subprocess.run(
            normalized_command,
            cwd=str(cwd_directory),
            env=dict(normalized_environment),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            shell=False,
            timeout=float(timeout_seconds),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError("diagnostic process did not produce a complete transcript") from exc
    ended_ns = time.monotonic_ns()
    if not isinstance(completed.stdout, bytes) or not isinstance(completed.stderr, bytes):
        raise ValueError("diagnostic process emitted a non-bytes transcript")
    return ProcessTranscript(
        source_digest=source_text,
        source_root=str(source_directory),
        cwd=str(cwd_directory),
        command=normalized_command,
        executable_sha256=_sha256(executable.read_bytes()),
        environment=normalized_environment,
        started_ns=started_ns,
        ended_ns=ended_ns,
        exit_code=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


@dataclass(frozen=True, slots=True)
class TypedEvidenceBinding:
    """One exact TaskSpec/result/outcome/evidence binding for a transcript."""

    source_digest: str
    subject_kind: str
    result: TypedExecutionResult
    evidence: EvidenceEnvelope
    expected_result_id: str
    expected_result_group_digest: str
    expected_task_spec_digest: str
    expected_seed_lineage_digest: str
    expected_execution_outcome_digest: str
    expected_output_sha256: str

    def __post_init__(self) -> None:
        _require_text(self.source_digest, name="typed binding source digest")
        _require_text(self.subject_kind, name="typed binding subject kind")
        _reject_synthetic_marker(self.source_digest, self.subject_kind)
        if not isinstance(self.result, TypedExecutionResult):
            raise TypeError("typed binding requires TypedExecutionResult")
        if not isinstance(self.evidence, EvidenceEnvelope):
            raise TypeError("typed binding requires EvidenceEnvelope")
        if self.evidence.state in _FORBIDDEN_EVIDENCE_STATES:
            raise ValueError("typed binding cannot contain confirmed/root evidence")
        _require_text(self.expected_result_id, name="typed binding result ID")
        _require_text(
            self.expected_result_group_digest,
            name="typed binding result-group digest",
        )
        _require_text(
            self.expected_task_spec_digest,
            name="typed binding task-spec digest",
        )
        _require_text(
            self.expected_seed_lineage_digest,
            name="typed binding seed-lineage digest",
        )
        _require_text(
            self.expected_execution_outcome_digest,
            name="typed binding execution-outcome digest",
        )
        _require_sha256(self.expected_output_sha256, name="typed binding output SHA")
        result = self.result
        evidence = self.evidence
        if self.expected_result_id != result.result_id:
            raise ValueError("typed binding result ID mismatch")
        if self.expected_result_group_digest != result.result_group.digest:
            raise ValueError("typed binding result-group mismatch")
        if self.expected_task_spec_digest != result.task.digest:
            raise ValueError("typed binding task-spec mismatch")
        if self.expected_seed_lineage_digest != result.seed_lineage.digest:
            raise ValueError("typed binding seed-lineage mismatch")
        if self.expected_execution_outcome_digest != result.outcome.digest:
            raise ValueError("typed binding execution-outcome mismatch")
        if evidence.result_group_digest != result.result_group.digest:
            raise ValueError("typed binding evidence/result-group mismatch")
        if evidence.task_id != result.task.identity.task_id:
            raise ValueError("typed binding evidence/task mismatch")
        if evidence.seed_lineage_digest != result.seed_lineage.digest:
            raise ValueError("typed binding evidence/seed mismatch")
        if evidence.contract_fingerprint != result.result_group.contract_fingerprint:
            raise ValueError("typed binding evidence/contract mismatch")
        if evidence.target_fingerprint != result.result_group.target_fingerprint:
            raise ValueError("typed binding evidence/target mismatch")
        outcomes = {item.endpoint_id: item for item in evidence.execution_outcomes}
        if outcomes.get(result.outcome.endpoint_id) != result.outcome:
            raise ValueError("typed binding evidence/outcome mismatch")
        assert_deeply_immutable(result)
        assert_deeply_immutable(evidence)
        assert_deeply_immutable(self)

    @property
    def subject_id(self) -> str:
        return self.result.result_id

    @property
    def digest(self) -> str:
        return stable_digest("osc-private-phase6-typed-evidence-binding", self)


def _diagnostic_payload(
    transcript: ProcessTranscript,
    binding: TypedEvidenceBinding,
) -> dict[str, object]:
    result = binding.result
    return {
        "schema_version": REAL_EVIDENCE_DIAGNOSTIC_CAPTURE_SCHEMA_VERSION,
        "mode": REAL_EVIDENCE_DIAGNOSTIC_MODE,
        "source_digest": transcript.source_digest,
        "process_transcript_digest": transcript.digest,
        "command_digest": transcript.command_digest,
        "executable_sha256": transcript.executable_sha256,
        "stdout_sha256": transcript.stdout_sha256,
        "stderr_sha256": transcript.stderr_sha256,
        "output_sha256": transcript.output_sha256,
        "subject_kind": binding.subject_kind,
        "typed_result_id": binding.subject_id,
        "result_group_digest": result.result_group.digest,
        "task_id": result.task.identity.task_id,
        "task_spec_digest": result.task.digest,
        "seed_lineage_digest": result.seed_lineage.digest,
        "execution_outcome_digest": result.outcome.digest,
        "evidence_envelope_digest": binding.evidence.digest,
        "diagnostic_only": True,
        "gate_credit": False,
        "bug_claimed": False,
    }


@dataclass(frozen=True, slots=True)
class DiagnosticEvidenceCapture:
    """In-memory result that is permanently not_evaluated for authority use."""

    transcript: ProcessTranscript
    binding: TypedEvidenceBinding
    payload_json: str
    schema_version: str = REAL_EVIDENCE_DIAGNOSTIC_CAPTURE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.transcript, ProcessTranscript):
            raise TypeError("diagnostic capture requires ProcessTranscript")
        if not isinstance(self.binding, TypedEvidenceBinding):
            raise TypeError("diagnostic capture requires TypedEvidenceBinding")
        if self.schema_version != REAL_EVIDENCE_DIAGNOSTIC_CAPTURE_SCHEMA_VERSION:
            raise ValueError("diagnostic capture schema version mismatch")
        if self.transcript.exit_code != 0:
            raise ValueError("diagnostic capture rejects a nonzero process exit")
        if not self.transcript.stdout:
            raise ValueError("diagnostic capture rejects an empty process output")
        if self.transcript.source_digest != self.binding.source_digest:
            raise ValueError("diagnostic capture source digest mismatch")
        if self.transcript.output_sha256 != self.binding.expected_output_sha256:
            raise ValueError("diagnostic capture output SHA-256 mismatch")
        expected = canonical_json(_diagnostic_payload(self.transcript, self.binding))
        if self.payload_json != expected:
            raise ValueError("diagnostic capture payload is not exact canonical staging")
        try:
            decoded = json.loads(self.payload_json)
        except json.JSONDecodeError as exc:
            raise ValueError("diagnostic capture payload is malformed") from exc
        if not isinstance(decoded, dict) or decoded.get("bug_claimed") is not False:
            raise ValueError("diagnostic capture payload has an invalid bug claim")
        assert_deeply_immutable(self)

    @property
    def capture_id(self) -> str:
        return stable_digest(
            "osc-private-phase6-diagnostic-capture-id",
            (self.transcript.digest, self.binding.digest),
        )

    @property
    def digest(self) -> str:
        return stable_digest("osc-private-phase6-diagnostic-evidence-capture", self)

    @property
    def status(self) -> str:
        return "not_evaluated"

    @property
    def diagnostic_only(self) -> bool:
        return True

    @property
    def authority_eligible(self) -> bool:
        return False

    @property
    def gate_credit(self) -> bool:
        return False

    @property
    def dynamic_plan_emitted(self) -> bool:
        return False

    @property
    def artifact_receipt_index_emitted(self) -> bool:
        return False

    @property
    def authority_bundle_emitted(self) -> bool:
        return False

    @property
    def candidate_confirmed(self) -> bool:
        return False

    @property
    def bug_claimed(self) -> bool:
        return False


def stage_diagnostic_capture(
    *,
    transcript: ProcessTranscript,
    binding: TypedEvidenceBinding,
) -> DiagnosticEvidenceCapture:
    """Bind an observed transcript in memory without persisting any evidence."""

    if not isinstance(transcript, ProcessTranscript):
        raise TypeError("diagnostic staging requires ProcessTranscript")
    if not isinstance(binding, TypedEvidenceBinding):
        raise TypeError("diagnostic staging requires TypedEvidenceBinding")
    return DiagnosticEvidenceCapture(
        transcript=transcript,
        binding=binding,
        payload_json=canonical_json(_diagnostic_payload(transcript, binding)),
    )


def diagnostic_runner_preview() -> dict[str, object]:
    """Return the runner's fixed no-execution/no-output declaration."""

    return {
        "schema_version": REAL_EVIDENCE_DIAGNOSTIC_CAPTURE_SCHEMA_VERSION,
        "mode": REAL_EVIDENCE_DIAGNOSTIC_MODE,
        "launches_formal_workload": False,
        "writes_durable_output": False,
        "emits_dynamic_plan": False,
        "emits_artifact_receipt_index": False,
        "publishes_authority_bundle": False,
        "gate_credit": False,
        "candidate_confirmed": False,
        "bug_claimed": False,
    }


__all__ = [
    "DiagnosticEvidenceCapture",
    "ProcessTranscript",
    "REAL_EVIDENCE_DIAGNOSTIC_CAPTURE_SCHEMA_VERSION",
    "REAL_EVIDENCE_DIAGNOSTIC_MODE",
    "TypedEvidenceBinding",
    "capture_diagnostic_process",
    "diagnostic_runner_preview",
    "stage_diagnostic_capture",
]
