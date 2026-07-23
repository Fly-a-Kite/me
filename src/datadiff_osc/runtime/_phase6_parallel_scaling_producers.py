"""Private typed construction for one Phase-6 parallel-scaling receipt.

The builder deliberately consumes the existing typed Runtime worker bindings
and raw counters, but it does not produce an artifact or provenance receipt.
It therefore remains a local, non-authoritative construction boundary until a
later Root-owned bridge re-reads and admits canonical evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Final

from datadiff_osc._canonical import stable_digest
from datadiff_osc.runtime._phase6_parallel_scaling_receipts import (
    ParallelScalingEvidenceReceipt,
)
from datadiff_osc.runtime._receipt_producers import (
    ParallelScalingObservation,
    TypedExecutionResult,
    WorkerBoundExecutionResult,
    _parallel_arm_projections,
)
from datadiff_osc.schemas import (
    EvidenceEnvelope,
    EvidenceState,
    ExecutionStatus,
    TaskKind,
    VerdictKind,
)


_DIGEST_RE: Final = re.compile(r"(?:[A-Za-z0-9_.:-]+-)?[0-9a-f]{64}")
_AUTHORITATIVE_VERDICTS: Final = frozenset(
    {VerdictKind.SATISFIED, VerdictKind.VIOLATED}
)
_RESERVED_CONTEXT_KEYS: Final = frozenset(
    {
        "worker_count",
        "workers",
        "sample_id",
        "elapsed",
        "elapsed_ns",
        "efficiency",
        "speedup",
        "aggregate",
    }
)


def _require_digest(name: str, value: object) -> str:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a SHA-256 or namespaced stable digest")
    return value


def _require_fields(
    name: str,
    value: object,
    *,
    reject_reserved_keys: bool,
) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, tuple) or not value:
        raise ValueError(f"{name} must be a non-empty immutable field tuple")
    for item in value:
        if (
            not isinstance(item, tuple)
            or len(item) != 2
            or not isinstance(item[0], str)
            or not item[0]
            or "\x00" in item[0]
            or not isinstance(item[1], str)
            or "\x00" in item[1]
        ):
            raise ValueError(f"{name} contains an invalid field")
    keys = tuple(item[0] for item in value)
    if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
        raise ValueError(f"{name} field names must be uniquely sorted")
    if reject_reserved_keys and set(keys) & _RESERVED_CONTEXT_KEYS:
        raise ValueError(f"{name} cannot supply derived worker or metric fields")
    return value


@dataclass(frozen=True, slots=True)
class ParallelScalingBenchmarkContext:
    """Immutable input that supplies non-execution parallel benchmark context."""

    source_snapshot_digest: str
    benchmark_plan: tuple[tuple[str, str], ...]
    environment: tuple[tuple[str, str], ...]
    one_worker_config: tuple[tuple[str, str], ...]
    six_worker_config: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        _require_digest("source snapshot", self.source_snapshot_digest)
        _require_fields("benchmark plan", self.benchmark_plan, reject_reserved_keys=True)
        _require_fields("environment", self.environment, reject_reserved_keys=True)
        _require_fields(
            "one-worker configuration",
            self.one_worker_config,
            reject_reserved_keys=True,
        )
        _require_fields(
            "six-worker configuration",
            self.six_worker_config,
            reject_reserved_keys=True,
        )


@dataclass(frozen=True, slots=True)
class ParallelScalingProductionObservation:
    """Typed, non-authoritative input needed to construct one R5 receipt."""

    benchmark_context: ParallelScalingBenchmarkContext
    parallel_sample: ParallelScalingObservation
    one_worker_evidence: tuple[EvidenceEnvelope, ...]
    six_worker_evidence: tuple[EvidenceEnvelope, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.benchmark_context, ParallelScalingBenchmarkContext):
            raise ValueError("production observation requires benchmark context")
        if not isinstance(self.parallel_sample, ParallelScalingObservation):
            raise ValueError("production observation requires a parallel sample")
        _validate_production_observation(self)


def _require_evidence_binding(
    *,
    name: str,
    evidence: EvidenceEnvelope,
    result: TypedExecutionResult,
) -> VerdictKind:
    if not isinstance(evidence, EvidenceEnvelope):
        raise ValueError(f"{name} evidence must be an EvidenceEnvelope")
    if evidence.state is not EvidenceState.NOT_A_CANDIDATE:
        raise ValueError(f"{name} evidence must remain NOT_A_CANDIDATE")
    if evidence.verdict_kind not in _AUTHORITATIVE_VERDICTS:
        raise ValueError(f"{name} evidence verdict must be authoritative")
    if evidence.result_group_digest != result.result_group.digest:
        raise ValueError(f"{name} evidence/result-group mismatch")
    if evidence.task_id != result.task.identity.task_id:
        raise ValueError(f"{name} evidence/task mismatch")
    if evidence.seed_lineage_digest != result.seed_lineage.digest:
        raise ValueError(f"{name} evidence/seed mismatch")
    if evidence.contract_fingerprint != result.result_group.contract_fingerprint:
        raise ValueError(f"{name} evidence/contract mismatch")
    if evidence.target_fingerprint != result.result_group.target_fingerprint:
        raise ValueError(f"{name} evidence/target mismatch")
    if evidence.execution_outcomes != (result.outcome,):
        raise ValueError(f"{name} evidence/outcome mismatch")
    return evidence.verdict_kind


def _validate_arm_evidence(
    *,
    name: str,
    bindings: tuple[WorkerBoundExecutionResult, ...],
    evidence: tuple[EvidenceEnvelope, ...],
) -> tuple[EvidenceEnvelope, ...]:
    if not isinstance(evidence, tuple) or len(evidence) != len(bindings):
        raise ValueError(f"{name} evidence must have exactly one item per typed result")
    ordered_bindings = tuple(
        sorted(bindings, key=lambda value: value.result.task.identity.task_id)
    )
    expected_task_ids = tuple(
        value.result.task.identity.task_id for value in ordered_bindings
    )
    actual_task_ids = tuple(item.task_id for item in evidence if isinstance(item, EvidenceEnvelope))
    if len(actual_task_ids) != len(evidence):
        raise ValueError(f"{name} evidence must be typed")
    if actual_task_ids != expected_task_ids:
        raise ValueError(f"{name} evidence must be in canonical task order")
    verdicts = tuple(
        _require_evidence_binding(
            name=name,
            evidence=item,
            result=binding.result,
        )
        for binding, item in zip(ordered_bindings, evidence, strict=True)
    )
    if len(set(verdicts)) != 1:
        raise ValueError(f"{name} evidence verdicts must agree")
    return evidence


def _validate_production_observation(
    observation: ParallelScalingProductionObservation,
) -> None:
    sample = observation.parallel_sample
    one_worker = sample.one_worker_results
    six_worker = sample.six_worker_results
    if any(item.worker_count != 1 for item in one_worker):
        raise ValueError("one-worker arm must bind worker_count=1")
    if any(item.worker_count != 6 for item in six_worker):
        raise ValueError("six-worker arm must bind worker_count=6")

    all_bindings = one_worker + six_worker
    for binding in all_bindings:
        result = binding.result
        if result.outcome.status is not ExecutionStatus.OK:
            raise ValueError("parallel typed executions require OK outcomes")
        if result.task.identity.task_kind is not TaskKind.BACKEND_EXECUTION:
            raise ValueError("parallel typed executions require BACKEND_EXECUTION tasks")

    for label, values in (
        ("tasks", tuple(item.result.task.identity.task_id for item in all_bindings)),
        ("TaskSpecs", tuple(item.result.task.digest for item in all_bindings)),
        ("results", tuple(item.result.result_id for item in all_bindings)),
    ):
        if len(values) != len(set(values)):
            raise ValueError(f"parallel {label} must be globally distinct")

    one_evidence = _validate_arm_evidence(
        name="one-worker",
        bindings=one_worker,
        evidence=observation.one_worker_evidence,
    )
    six_evidence = _validate_arm_evidence(
        name="six-worker",
        bindings=six_worker,
        evidence=observation.six_worker_evidence,
    )
    evidence_ids = tuple(item.digest for item in one_evidence + six_evidence)
    if len(evidence_ids) != len(set(evidence_ids)):
        raise ValueError("parallel evidence identities must be globally distinct")
    if one_evidence[0].verdict_kind is not six_evidence[0].verdict_kind:
        raise ValueError("parallel arm evidence verdicts must agree")

    # Re-derive both projections at this boundary instead of trusting a caller digest.
    one_projection, six_projection = _parallel_arm_projections(one_worker, six_worker)
    if one_projection != six_projection:
        raise ValueError("parallel logical task-set projection mismatch")
    if sample.one_worker_measurement.delta != sample.six_worker_measurement.delta:
        raise ValueError("parallel counters must measure the same task count")
    if (
        sample.one_worker_measurement.interval.clock_id
        != sample.six_worker_measurement.interval.clock_id
    ):
        raise ValueError("parallel monotonic clocks must match")


def _logical_task_set_digest(projection: tuple[tuple[object, ...], ...]) -> str:
    return stable_digest("osc-phase6-parallel-scaling-logical-task-set-v1", projection)


def _arm_configuration_digest(
    *,
    worker_count: int,
    bindings: tuple[WorkerBoundExecutionResult, ...],
    fields: tuple[tuple[str, str], ...],
) -> str:
    return stable_digest(
        "osc-phase6-parallel-scaling-arm-configuration-v1",
        {
            "worker_count": worker_count,
            "task_specs": tuple(
                (
                    value.result.task.identity.task_id,
                    value.result.task.digest,
                    value.result.task.payload_digest,
                    value.result.task.resources.digest,
                )
                for value in sorted(
                    bindings, key=lambda item: item.result.task.identity.task_id
                )
            ),
            "fields": fields,
        },
    )


def _arm_result_digest(
    *,
    worker_count: int,
    bindings: tuple[WorkerBoundExecutionResult, ...],
    measurement: object,
) -> str:
    return stable_digest(
        "osc-phase6-parallel-scaling-execution-arm-v1",
        {
            "worker_count": worker_count,
            "typed_results": tuple(
                (
                    value.result.task.identity.task_id,
                    value.result.task.digest,
                    value.result.result_id,
                    value.result.outcome.digest,
                )
                for value in sorted(
                    bindings, key=lambda item: item.result.task.identity.task_id
                )
            ),
            "raw_counter": measurement,
        },
    )


def _arm_evidence_digest(
    *,
    worker_count: int,
    evidence: tuple[EvidenceEnvelope, ...],
) -> str:
    return stable_digest(
        "osc-phase6-parallel-scaling-evidence-arm-v1",
        {
            "worker_count": worker_count,
            "evidence": tuple((item.task_id, item.digest) for item in evidence),
        },
    )


def build_parallel_scaling_evidence_receipt(
    observation: ParallelScalingProductionObservation,
) -> ParallelScalingEvidenceReceipt:
    """Derive one private R5 receipt from exact typed, non-authoritative input."""

    if not isinstance(observation, ParallelScalingProductionObservation):
        raise TypeError("observation must be a ParallelScalingProductionObservation")
    _validate_production_observation(observation)
    context = observation.benchmark_context
    sample = observation.parallel_sample
    one_worker = sample.one_worker_results
    six_worker = sample.six_worker_results
    one_projection, six_projection = _parallel_arm_projections(one_worker, six_worker)
    one_task_set_digest = _logical_task_set_digest(one_projection)
    six_task_set_digest = _logical_task_set_digest(six_projection)
    if one_task_set_digest != six_task_set_digest:
        raise ValueError("parallel independently derived task-set digests mismatch")

    return ParallelScalingEvidenceReceipt(
        source_snapshot_digest=context.source_snapshot_digest,
        workload_id=sample.workload_id,
        benchmark_plan_digest=stable_digest(
            "osc-phase6-parallel-scaling-benchmark-plan-v1", context.benchmark_plan
        ),
        environment_digest=stable_digest(
            "osc-phase6-parallel-scaling-environment-v1", context.environment
        ),
        one_worker_config_digest=_arm_configuration_digest(
            worker_count=1,
            bindings=one_worker,
            fields=context.one_worker_config,
        ),
        six_worker_config_digest=_arm_configuration_digest(
            worker_count=6,
            bindings=six_worker,
            fields=context.six_worker_config,
        ),
        one_worker_task_set_digest=one_task_set_digest,
        six_worker_task_set_digest=six_task_set_digest,
        one_worker_result_digest=_arm_result_digest(
            worker_count=1,
            bindings=one_worker,
            measurement=sample.one_worker_measurement,
        ),
        six_worker_result_digest=_arm_result_digest(
            worker_count=6,
            bindings=six_worker,
            measurement=sample.six_worker_measurement,
        ),
        one_worker_evidence_digest=_arm_evidence_digest(
            worker_count=1,
            evidence=observation.one_worker_evidence,
        ),
        six_worker_evidence_digest=_arm_evidence_digest(
            worker_count=6,
            evidence=observation.six_worker_evidence,
        ),
        one_worker_clock_id=sample.one_worker_measurement.interval.clock_id,
        one_worker_start_ns=sample.one_worker_measurement.interval.start_ns,
        one_worker_end_ns=sample.one_worker_measurement.interval.end_ns,
        one_worker_task_counter_start=sample.one_worker_measurement.start_count,
        one_worker_task_counter_end=sample.one_worker_measurement.end_count,
        six_worker_clock_id=sample.six_worker_measurement.interval.clock_id,
        six_worker_start_ns=sample.six_worker_measurement.interval.start_ns,
        six_worker_end_ns=sample.six_worker_measurement.interval.end_ns,
        six_worker_task_counter_start=sample.six_worker_measurement.start_count,
        six_worker_task_counter_end=sample.six_worker_measurement.end_count,
    )


__all__ = [
    "ParallelScalingBenchmarkContext",
    "ParallelScalingProductionObservation",
    "build_parallel_scaling_evidence_receipt",
]
