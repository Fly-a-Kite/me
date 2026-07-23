"""Private transcript and raw-artifact binding for one Contract pair.

This construction is deliberately narrower than a real Phase-6 producer.  It
rebuilds one already typed R9 Contract-performance proof, derives one canonical
raw-artifact payload from that proof, and binds a strict zero-exit transcript to
those bytes.  It neither launches a process nor creates gate authority.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from typing import Any

from datadiff_osc._canonical import canonical_envelope, canonical_json, stable_digest, to_primitive
from datadiff_osc._replay_support import (
    ReplayValidationError,
    assert_payload_roundtrip,
    replay_immutable_value,
    require_exact_mapping,
    require_nonnegative_int,
    require_text,
    require_tuple,
)
from datadiff_osc.runtime._phase6_contract_performance_provenance import (
    ContractPerformanceProductionProvenance,
    reconstruct_contract_performance_production_provenance_payload,
)
from datadiff_osc.runtime._phase6_contract_performance_receipts import (
    ContractPerformanceEvidenceReceipt,
)
from datadiff_osc.runtime.gate_artifacts import (
    RAW_GATE_ARTIFACT_SCHEMA_VERSION,
    MetricReducer,
    RawMetricRecord,
    RawMetricSeries,
)


CONTRACT_PERFORMANCE_EXECUTION_PROVENANCE_SCHEMA_VERSION = (
    "osc-phase6-contract-performance-execution-provenance-v1"
)
CONTRACT_PERFORMANCE_EXECUTION_PROVENANCE_ENVELOPE_TYPE = (
    "ContractPerformanceExecutionProvenance"
)

_EXECUTION_PROVENANCE_FILENAME = "_phase6_contract_performance_execution_provenance.py"
_R9_PROVENANCE_FILENAME = "_phase6_contract_performance_provenance.py"
_PRODUCER_KIND = "performance_paired_replay"
_COMMAND = (
    "datadiff-osc-private-contract-performance-producer",
    "--emit-raw-artifact-v1",
)
_CWD = "."
_ENVIRONMENT = (("PYTHONDONTWRITEBYTECODE", "1"),)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_DIGEST_RE = re.compile(r"(?:[A-Za-z0-9_.:-]+-)?[0-9a-f]{64}")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _module_sha256(filename: str) -> str:
    try:
        raw = Path(__file__).with_name(filename).read_bytes()
    except OSError as exc:
        raise ValueError(
            "contract performance execution provenance implementation bytes "
            f"unavailable:{filename}"
        ) from exc
    return _sha256(raw)


def _require_sha256(value: object, *, path: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{path} must be a SHA-256")
    return value


def _require_digest(value: object, *, path: str) -> str:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise ValueError(f"{path} must be a stable digest")
    return value


def _require_bytes(value: object, *, path: str) -> bytes:
    if not isinstance(value, bytes):
        raise ValueError(f"{path} must be canonical bytes")
    return value


def _raw_artifact_id(receipt: ContractPerformanceEvidenceReceipt) -> str:
    return stable_digest(
        "osc-phase6-contract-performance-execution-artifact-id-v1",
        {
            "source_snapshot_digest": receipt.source_snapshot_digest,
            "sample_id": receipt.sample_id,
            "receipt_digest": receipt.digest,
        },
    )


def _raw_metric_series(
    receipt: ContractPerformanceEvidenceReceipt,
) -> tuple[RawMetricSeries, ...]:
    counters = (
        ("baseline", "materialized_bytes", receipt.baseline_materialized_bytes),
        (
            "treatment",
            "materialized_bytes",
            receipt.treatment_materialized_bytes,
        ),
        (
            "baseline",
            "backend_pair_comparisons",
            receipt.baseline_backend_pair_comparisons,
        ),
        (
            "treatment",
            "backend_pair_comparisons",
            receipt.treatment_backend_pair_comparisons,
        ),
        ("baseline", "comparison_cpu_ns", receipt.baseline_comparison_cpu_ns),
        ("treatment", "comparison_cpu_ns", receipt.treatment_comparison_cpu_ns),
    )
    return tuple(
        RawMetricSeries(
            metric_key=f"private_contract_performance_{arm}_{counter}",
            reducer=MetricReducer.SINGLE_VALUE,
            records=(
                RawMetricRecord(
                    stable_digest(
                        "osc-phase6-contract-performance-execution-record-id-v1",
                        {
                            "sample_id": receipt.sample_id,
                            "receipt_digest": receipt.digest,
                            "arm": arm,
                            "counter": counter,
                        },
                    ),
                    value,
                ),
            ),
        )
        for arm, counter, value in counters
    )


def _derive_raw_artifact(
    provenance: ContractPerformanceProductionProvenance,
) -> tuple[str, bytes, str, tuple[RawMetricSeries, ...]]:
    if not isinstance(provenance, ContractPerformanceProductionProvenance):
        raise TypeError("provenance must be ContractPerformanceProductionProvenance")
    receipt = provenance.receipt
    artifact_id = _raw_artifact_id(receipt)
    series = _raw_metric_series(receipt)
    payload = {
        "schema_version": RAW_GATE_ARTIFACT_SCHEMA_VERSION,
        "artifact_id": artifact_id,
        "source_digest": receipt.source_snapshot_digest,
        "series": to_primitive(series),
    }
    raw = canonical_json(payload).encode("utf-8")
    return artifact_id, raw, _sha256(raw), series


@dataclass(frozen=True, slots=True)
class ContractPerformanceExecutionTranscript:
    """Exact, zero-exit transcript whose stdout is the derived raw artifact."""

    command: tuple[str, ...]
    cwd: str
    environment: tuple[tuple[str, str], ...]
    stdout: bytes
    stderr: bytes
    exit_code: int

    def __post_init__(self) -> None:
        if self.command != _COMMAND:
            raise ValueError("execution transcript command mismatch")
        if self.cwd != _CWD:
            raise ValueError("execution transcript cwd mismatch")
        if self.environment != _ENVIRONMENT:
            raise ValueError("execution transcript environment mismatch")
        if not self.stdout:
            raise ValueError("execution transcript stdout must be non-empty")
        if self.stderr != b"":
            raise ValueError("execution transcript stderr must be exactly empty")
        if isinstance(self.exit_code, bool) or self.exit_code != 0:
            raise ValueError("execution transcript requires exact zero exit")

    @property
    def output_sha256(self) -> str:
        return _sha256(self.stdout)

    @property
    def digest(self) -> str:
        return stable_digest("osc-phase6-contract-performance-execution-transcript", self)

    def to_dict(self) -> dict[str, object]:
        value = to_primitive(self)
        assert isinstance(value, dict)
        return value


@dataclass(frozen=True, slots=True)
class ContractPerformanceExecutionProvenance:
    """Private exact binding of R9 proof, transcript, and raw artifact bytes."""

    production_provenance: ContractPerformanceProductionProvenance
    transcript: ContractPerformanceExecutionTranscript
    artifact_id: str
    artifact_bytes: bytes
    artifact_sha256: str
    execution_module_sha256: str
    schema_version: str = CONTRACT_PERFORMANCE_EXECUTION_PROVENANCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(
            self.production_provenance, ContractPerformanceProductionProvenance
        ):
            raise ValueError(
                "execution provenance requires a typed Contract-performance proof"
            )
        if not isinstance(self.transcript, ContractPerformanceExecutionTranscript):
            raise ValueError("execution provenance requires an exact transcript")
        if self.schema_version != CONTRACT_PERFORMANCE_EXECUTION_PROVENANCE_SCHEMA_VERSION:
            raise ValueError("execution provenance schema version mismatch")
        _require_digest(
            self.production_provenance.receipt.source_snapshot_digest,
            path="execution provenance source digest",
        )
        _require_sha256(self.artifact_sha256, path="execution provenance artifact SHA")
        expected_execution_module = _module_sha256(_EXECUTION_PROVENANCE_FILENAME)
        if self.execution_module_sha256 != expected_execution_module:
            raise ValueError("execution provenance implementation binding mismatch")
        if self.production_provenance.bridge_module_sha256 != _module_sha256(
            _R9_PROVENANCE_FILENAME
        ):
            raise ValueError("execution provenance R9 implementation binding mismatch")
        expected_id, expected_bytes, expected_sha, _ = _derive_raw_artifact(
            self.production_provenance
        )
        if self.artifact_id != expected_id:
            raise ValueError("execution provenance artifact identity mismatch")
        if self.artifact_bytes != expected_bytes:
            raise ValueError("execution provenance artifact bytes mismatch")
        if self.artifact_sha256 != expected_sha:
            raise ValueError("execution provenance artifact SHA mismatch")
        if self.transcript.stdout != self.artifact_bytes:
            raise ValueError("execution transcript output/artifact mismatch")
        if self.transcript.output_sha256 != self.artifact_sha256:
            raise ValueError("execution transcript output SHA mismatch")

    @property
    def source_snapshot_digest(self) -> str:
        return self.production_provenance.receipt.source_snapshot_digest

    @property
    def receipt(self) -> ContractPerformanceEvidenceReceipt:
        return self.production_provenance.receipt

    @property
    def provenance_id(self) -> str:
        return stable_digest(
            "osc-phase6-contract-performance-execution-provenance-id-v1",
            {
                "source_snapshot_digest": self.source_snapshot_digest,
                "sample_id": self.receipt.sample_id,
                "receipt_digest": self.receipt.digest,
                "production_provenance_id": self.production_provenance.provenance_id,
                "artifact_id": self.artifact_id,
                "artifact_sha256": self.artifact_sha256,
                "transcript_digest": self.transcript.digest,
                "execution_module_sha256": self.execution_module_sha256,
            },
        )

    @property
    def digest(self) -> str:
        return stable_digest("osc-phase6-contract-performance-execution-provenance", self)

    def to_dict(self) -> dict[str, object]:
        value = to_primitive(self)
        assert isinstance(value, dict)
        return value


def build_contract_performance_execution_provenance(
    production_provenance: ContractPerformanceProductionProvenance,
) -> ContractPerformanceExecutionProvenance:
    """Derive the only structural transcript/artifact pair for one R9 proof."""

    if not isinstance(production_provenance, ContractPerformanceProductionProvenance):
        raise TypeError("production_provenance must be ContractPerformanceProductionProvenance")
    artifact_id, artifact_bytes, artifact_sha256, _ = _derive_raw_artifact(
        production_provenance
    )
    return ContractPerformanceExecutionProvenance(
        production_provenance=production_provenance,
        transcript=ContractPerformanceExecutionTranscript(
            command=_COMMAND,
            cwd=_CWD,
            environment=_ENVIRONMENT,
            stdout=artifact_bytes,
            stderr=b"",
            exit_code=0,
        ),
        artifact_id=artifact_id,
        artifact_bytes=artifact_bytes,
        artifact_sha256=artifact_sha256,
        execution_module_sha256=_module_sha256(_EXECUTION_PROVENANCE_FILENAME),
    )


def _replay_text_pair(value: object, path: str) -> tuple[str, str]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ReplayValidationError(f"{path}: expected an exact text pair")
    return (
        require_text(value[0], path=f"{path}[0]"),
        require_text(value[1], path=f"{path}[1]"),
    )


def _replay_transcript(value: object, path: str) -> ContractPerformanceExecutionTranscript:
    data = require_exact_mapping(
        value,
        fields=("command", "cwd", "environment", "stdout", "stderr", "exit_code"),
        path=path,
    )
    command = tuple(
        require_tuple(
            data["command"],
            path=f"{path}.command",
            item_replayer=lambda item, item_path: require_text(item, path=item_path),
            min_length=1,
        )
    )
    environment = tuple(
        require_tuple(
            data["environment"],
            path=f"{path}.environment",
            item_replayer=_replay_text_pair,
            min_length=1,
        )
    )
    stdout = replay_immutable_value(data["stdout"], f"{path}.stdout")
    stderr = replay_immutable_value(data["stderr"], f"{path}.stderr")
    return ContractPerformanceExecutionTranscript(
        command=command,
        cwd=require_text(data["cwd"], path=f"{path}.cwd"),
        environment=environment,
        stdout=_require_bytes(stdout, path=f"{path}.stdout"),
        stderr=_require_bytes(stderr, path=f"{path}.stderr"),
        exit_code=require_nonnegative_int(data["exit_code"], path=f"{path}.exit_code"),
    )


def reconstruct_contract_performance_execution_provenance_payload(
    payload: object,
) -> ContractPerformanceExecutionProvenance:
    """Fail closed while re-reading a private execution/artifact envelope."""

    path = "payload.ContractPerformanceExecutionProvenance"
    data = require_exact_mapping(
        payload,
        fields=(
            "production_provenance",
            "transcript",
            "artifact_id",
            "artifact_bytes",
            "artifact_sha256",
            "execution_module_sha256",
            "schema_version",
        ),
        path=path,
    )
    try:
        production_provenance = reconstruct_contract_performance_production_provenance_payload(
            data["production_provenance"]
        )
    except (TypeError, ValueError) as exc:
        raise ReplayValidationError(
            f"{path}.production_provenance: reconstruction failed:{exc}"
        ) from exc
    artifact_bytes = replay_immutable_value(data["artifact_bytes"], f"{path}.artifact_bytes")
    provenance = ContractPerformanceExecutionProvenance(
        production_provenance=production_provenance,
        transcript=_replay_transcript(data["transcript"], f"{path}.transcript"),
        artifact_id=require_text(data["artifact_id"], path=f"{path}.artifact_id"),
        artifact_bytes=_require_bytes(artifact_bytes, path=f"{path}.artifact_bytes"),
        artifact_sha256=_require_sha256(
            data["artifact_sha256"], path=f"{path}.artifact_sha256"
        ),
        execution_module_sha256=_require_sha256(
            data["execution_module_sha256"], path=f"{path}.execution_module_sha256"
        ),
        schema_version=require_text(data["schema_version"], path=f"{path}.schema_version"),
    )
    assert_payload_roundtrip(payload, provenance, path=path)
    return provenance


def canonical_contract_performance_execution_provenance_envelope(
    provenance: ContractPerformanceExecutionProvenance,
) -> str:
    """Serialize a private structural proof without creating a public API."""

    if not isinstance(provenance, ContractPerformanceExecutionProvenance):
        raise TypeError("provenance must be ContractPerformanceExecutionProvenance")
    return canonical_envelope(
        CONTRACT_PERFORMANCE_EXECUTION_PROVENANCE_ENVELOPE_TYPE,
        CONTRACT_PERFORMANCE_EXECUTION_PROVENANCE_SCHEMA_VERSION,
        provenance,
    )
