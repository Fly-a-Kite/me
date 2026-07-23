"""Private re-readable provenance for one Phase-6 parallel-scaling receipt.

This module binds a complete typed production observation to the private R6
receipt it deterministically derives.  It is deliberately not a public API,
raw gate artifact, producer admission, metric, or gate decision.  Root must
still verify a separately hashed canonical envelope before retaining any
parallel-scaling admission.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

from datadiff_osc._canonical import canonical_envelope, stable_digest, to_primitive
from datadiff_osc._replay_support import (
    ReplayValidationError,
    assert_payload_roundtrip,
    require_exact_mapping,
    require_nonnegative_int,
    require_text,
    require_tuple,
)
from datadiff_osc.contract_engine.replay import _reconstruct_result_group
from datadiff_osc.runtime._phase6_parallel_scaling_producers import (
    ParallelScalingBenchmarkContext,
    ParallelScalingProductionObservation,
    build_parallel_scaling_evidence_receipt,
)
from datadiff_osc.runtime._phase6_parallel_scaling_receipts import (
    ParallelScalingEvidenceReceipt,
)
from datadiff_osc.runtime._receipt_producers import (
    PARALLEL_WORKER_BINDING_SCHEMA_VERSION,
    ParallelScalingObservation,
    RawCounterMeasurement,
    RawMonotonicInterval,
    TypedExecutionResult,
    WorkerBoundExecutionResult,
)
from datadiff_osc.runtime._semantic_replay import (
    _reconstruct_parallel_scaling_evidence_receipt_payload,
    _replay_evidence,
    _replay_outcome,
    _replay_seed_lineage,
    _replay_task_spec,
)
from datadiff_osc.schemas import EvidenceEnvelope


PARALLEL_SCALING_PRODUCTION_PROVENANCE_SCHEMA_VERSION = (
    "osc-phase6-parallel-scaling-production-provenance-v1"
)
PARALLEL_SCALING_PRODUCTION_PROVENANCE_ENVELOPE_TYPE = (
    "ParallelScalingProductionProvenance"
)
_R6_PRODUCER_FILENAME = "_phase6_parallel_scaling_producers.py"
_BRIDGE_FILENAME = "_phase6_parallel_scaling_provenance.py"


def _module_sha256(filename: str) -> str:
    try:
        raw = Path(__file__).with_name(filename).read_bytes()
    except OSError as exc:
        raise ValueError(f"parallel provenance implementation bytes unavailable:{filename}") from exc
    return hashlib.sha256(raw).hexdigest()


def _require_sha256(value: object, *, path: str) -> str:
    text = require_text(value, path=path)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ReplayValidationError(f"{path}: expected a lowercase SHA-256 digest")
    return text


def _replay_text_field_pair(value: object, path: str) -> tuple[str, str]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ReplayValidationError(f"{path}: expected a canonical text field pair")
    return (
        require_text(value[0], path=f"{path}[0]"),
        require_text(value[1], path=f"{path}[1]"),
    )


def _replay_text_fields(value: object, path: str) -> tuple[tuple[str, str], ...]:
    fields = tuple(
        require_tuple(
            value,
            path=path,
            item_replayer=_replay_text_field_pair,
            min_length=1,
        )
    )
    names = tuple(item[0] for item in fields)
    if names != tuple(sorted(names)) or len(names) != len(set(names)):
        raise ReplayValidationError(f"{path}: field names must be uniquely sorted")
    return fields


def _replay_benchmark_context(value: object, path: str) -> ParallelScalingBenchmarkContext:
    data = require_exact_mapping(
        value,
        fields=(
            "source_snapshot_digest",
            "benchmark_plan",
            "environment",
            "one_worker_config",
            "six_worker_config",
        ),
        path=path,
    )
    return ParallelScalingBenchmarkContext(
        source_snapshot_digest=require_text(
            data["source_snapshot_digest"],
            path=f"{path}.source_snapshot_digest",
        ),
        benchmark_plan=_replay_text_fields(data["benchmark_plan"], f"{path}.benchmark_plan"),
        environment=_replay_text_fields(data["environment"], f"{path}.environment"),
        one_worker_config=_replay_text_fields(
            data["one_worker_config"], f"{path}.one_worker_config"
        ),
        six_worker_config=_replay_text_fields(
            data["six_worker_config"], f"{path}.six_worker_config"
        ),
    )


def _replay_typed_execution_result(value: object, path: str) -> TypedExecutionResult:
    data = require_exact_mapping(
        value,
        fields=("case_id", "result_group", "task", "seed_lineage", "outcome"),
        path=path,
    )
    return TypedExecutionResult(
        case_id=require_text(data["case_id"], path=f"{path}.case_id"),
        result_group=_reconstruct_result_group(data["result_group"]),
        task=_replay_task_spec(data["task"], f"{path}.task"),
        seed_lineage=_replay_seed_lineage(data["seed_lineage"], f"{path}.seed_lineage"),
        outcome=_replay_outcome(data["outcome"], f"{path}.outcome"),
    )


def _replay_worker_binding(value: object, path: str) -> WorkerBoundExecutionResult:
    data = require_exact_mapping(
        value,
        fields=("worker_count", "result", "schema_version"),
        path=path,
    )
    return WorkerBoundExecutionResult(
        worker_count=require_nonnegative_int(
            data["worker_count"], path=f"{path}.worker_count"
        ),
        result=_replay_typed_execution_result(data["result"], f"{path}.result"),
        schema_version=require_text(data["schema_version"], path=f"{path}.schema_version"),
    )


def _replay_interval(value: object, path: str) -> RawMonotonicInterval:
    data = require_exact_mapping(
        value,
        fields=("clock_id", "start_ns", "end_ns"),
        path=path,
    )
    return RawMonotonicInterval(
        clock_id=require_text(data["clock_id"], path=f"{path}.clock_id"),
        start_ns=require_nonnegative_int(data["start_ns"], path=f"{path}.start_ns"),
        end_ns=require_nonnegative_int(data["end_ns"], path=f"{path}.end_ns"),
    )


def _replay_counter(value: object, path: str) -> RawCounterMeasurement:
    data = require_exact_mapping(
        value,
        fields=("unit", "start_count", "end_count", "interval"),
        path=path,
    )
    return RawCounterMeasurement(
        unit=require_text(data["unit"], path=f"{path}.unit"),
        start_count=require_nonnegative_int(
            data["start_count"], path=f"{path}.start_count"
        ),
        end_count=require_nonnegative_int(data["end_count"], path=f"{path}.end_count"),
        interval=_replay_interval(data["interval"], f"{path}.interval"),
    )


def _replay_parallel_observation(value: object, path: str) -> ParallelScalingObservation:
    data = require_exact_mapping(
        value,
        fields=(
            "ordinal",
            "workload_id",
            "one_worker_results",
            "six_worker_results",
            "one_worker_measurement",
            "six_worker_measurement",
        ),
        path=path,
    )
    return ParallelScalingObservation(
        ordinal=require_nonnegative_int(data["ordinal"], path=f"{path}.ordinal"),
        workload_id=require_text(data["workload_id"], path=f"{path}.workload_id"),
        one_worker_results=tuple(
            require_tuple(
                data["one_worker_results"],
                path=f"{path}.one_worker_results",
                item_replayer=_replay_worker_binding,
                min_length=1,
            )
        ),
        six_worker_results=tuple(
            require_tuple(
                data["six_worker_results"],
                path=f"{path}.six_worker_results",
                item_replayer=_replay_worker_binding,
                min_length=1,
            )
        ),
        one_worker_measurement=_replay_counter(
            data["one_worker_measurement"], f"{path}.one_worker_measurement"
        ),
        six_worker_measurement=_replay_counter(
            data["six_worker_measurement"], f"{path}.six_worker_measurement"
        ),
    )


def _replay_evidence_vector(value: object, path: str) -> tuple[EvidenceEnvelope, ...]:
    return tuple(
        require_tuple(
            value,
            path=path,
            item_replayer=_replay_evidence,
            min_length=1,
        )
    )


@dataclass(frozen=True, slots=True)
class ParallelScalingProductionProvenance:
    """Canonical raw typed context and its exact R6 receipt.

    The implementation hashes are derived from local bytes by the builder and
    re-checked by reconstruction.  They are not caller-selected declarations.
    """

    observation: ParallelScalingProductionObservation
    receipt: ParallelScalingEvidenceReceipt
    producer_module_sha256: str
    bridge_module_sha256: str
    schema_version: str = PARALLEL_SCALING_PRODUCTION_PROVENANCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.observation, ParallelScalingProductionObservation):
            raise ValueError("parallel provenance requires a typed production observation")
        if not isinstance(self.receipt, ParallelScalingEvidenceReceipt):
            raise ValueError("parallel provenance requires a ParallelScalingEvidenceReceipt")
        if self.schema_version != PARALLEL_SCALING_PRODUCTION_PROVENANCE_SCHEMA_VERSION:
            raise ValueError("parallel provenance schema version mismatch")
        for field_name, expected in (
            ("producer_module_sha256", _module_sha256(_R6_PRODUCER_FILENAME)),
            ("bridge_module_sha256", _module_sha256(_BRIDGE_FILENAME)),
        ):
            actual = getattr(self, field_name)
            if not isinstance(actual, str) or actual != expected:
                raise ValueError(f"parallel provenance implementation binding mismatch:{field_name}")
        rebuilt = build_parallel_scaling_evidence_receipt(self.observation)
        if rebuilt != self.receipt:
            raise ValueError("parallel provenance receipt does not match typed rebuild")
        if (
            self.receipt.source_snapshot_digest
            != self.observation.benchmark_context.source_snapshot_digest
        ):
            raise ValueError("parallel provenance source binding mismatch")

    @property
    def provenance_id(self) -> str:
        return stable_digest(
            "osc-phase6-parallel-scaling-production-provenance-id-v1",
            {
                "source_snapshot_digest": self.receipt.source_snapshot_digest,
                "sample_id": self.receipt.sample_id,
                "receipt_digest": self.receipt.digest,
                "producer_module_sha256": self.producer_module_sha256,
                "bridge_module_sha256": self.bridge_module_sha256,
            },
        )

    @property
    def digest(self) -> str:
        return stable_digest("osc-phase6-parallel-scaling-production-provenance", self)

    def to_dict(self) -> dict[str, object]:
        value = to_primitive(self)
        assert isinstance(value, dict)
        return value


def build_parallel_scaling_production_provenance(
    observation: ParallelScalingProductionObservation,
) -> ParallelScalingProductionProvenance:
    """Build the only private provenance object from complete typed input."""

    if not isinstance(observation, ParallelScalingProductionObservation):
        raise TypeError("observation must be a ParallelScalingProductionObservation")
    return ParallelScalingProductionProvenance(
        observation=observation,
        receipt=build_parallel_scaling_evidence_receipt(observation),
        producer_module_sha256=_module_sha256(_R6_PRODUCER_FILENAME),
        bridge_module_sha256=_module_sha256(_BRIDGE_FILENAME),
    )


def reconstruct_parallel_scaling_production_provenance_payload(
    payload: object,
) -> ParallelScalingProductionProvenance:
    """Fail-closed canonical reconstruction used only by the Root bridge."""

    path = "payload.ParallelScalingProductionProvenance"
    data = require_exact_mapping(
        payload,
        fields=(
            "observation",
            "receipt",
            "producer_module_sha256",
            "bridge_module_sha256",
            "schema_version",
        ),
        path=path,
    )
    observation_data = require_exact_mapping(
        data["observation"],
        fields=(
            "benchmark_context",
            "parallel_sample",
            "one_worker_evidence",
            "six_worker_evidence",
        ),
        path=f"{path}.observation",
    )
    observation = ParallelScalingProductionObservation(
        benchmark_context=_replay_benchmark_context(
            observation_data["benchmark_context"],
            f"{path}.observation.benchmark_context",
        ),
        parallel_sample=_replay_parallel_observation(
            observation_data["parallel_sample"],
            f"{path}.observation.parallel_sample",
        ),
        one_worker_evidence=_replay_evidence_vector(
            observation_data["one_worker_evidence"],
            f"{path}.observation.one_worker_evidence",
        ),
        six_worker_evidence=_replay_evidence_vector(
            observation_data["six_worker_evidence"],
            f"{path}.observation.six_worker_evidence",
        ),
    )
    provenance = ParallelScalingProductionProvenance(
        observation=observation,
        receipt=_reconstruct_parallel_scaling_evidence_receipt_payload(data["receipt"]),
        producer_module_sha256=_require_sha256(
            data["producer_module_sha256"],
            path=f"{path}.producer_module_sha256",
        ),
        bridge_module_sha256=_require_sha256(
            data["bridge_module_sha256"],
            path=f"{path}.bridge_module_sha256",
        ),
        schema_version=require_text(data["schema_version"], path=f"{path}.schema_version"),
    )
    assert_payload_roundtrip(payload, provenance, path=path)
    return provenance


def canonical_parallel_scaling_production_provenance_envelope(
    provenance: ParallelScalingProductionProvenance,
) -> str:
    if not isinstance(provenance, ParallelScalingProductionProvenance):
        raise TypeError("provenance must be a ParallelScalingProductionProvenance")
    return canonical_envelope(
        PARALLEL_SCALING_PRODUCTION_PROVENANCE_ENVELOPE_TYPE,
        PARALLEL_SCALING_PRODUCTION_PROVENANCE_SCHEMA_VERSION,
        provenance,
    )


__all__ = [
    "PARALLEL_SCALING_PRODUCTION_PROVENANCE_ENVELOPE_TYPE",
    "PARALLEL_SCALING_PRODUCTION_PROVENANCE_SCHEMA_VERSION",
    "ParallelScalingProductionProvenance",
    "build_parallel_scaling_production_provenance",
    "canonical_parallel_scaling_production_provenance_envelope",
    "reconstruct_parallel_scaling_production_provenance_payload",
]
