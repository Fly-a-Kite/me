"""Private typed builders for Phase-6 Contract raw-pair receipts.

This module is deliberately a pure boundary: it turns already typed comparison
and execution observations into the R3 private receipt, but it does not write
an artifact, establish producer provenance, select a cohort, or calculate a
gate.  Root remains responsible for every later admission and cross-object
binding step.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Final

from datadiff_osc._canonical import stable_digest
from datadiff_osc.contract_engine._phase6_comparison_receipts import (
    ContractComparisonReceipt,
)
from datadiff_osc.runtime._phase6_contract_performance_receipts import (
    BASELINE_STRATEGY,
    TREATMENT_STRATEGY,
    ContractPerformanceEvidenceReceipt,
)
from datadiff_osc.runtime._receipt_producers import TypedExecutionResult
from datadiff_osc.schemas import (
    EvidenceEnvelope,
    EvidenceState,
    ExecutionStatus,
    TaskKind,
    VerdictKind,
)


_DIGEST_RE: Final = re.compile(r"(?:[A-Za-z0-9_.:-]+-)?[0-9a-f]{64}")
_RESERVED_CONFIG_KEYS: Final = frozenset(
    {"strategy", "comparison_strategy", "baseline_strategy", "treatment_strategy"}
)
_AUTHORITATIVE_VERDICTS: Final = frozenset(
    {VerdictKind.SATISFIED, VerdictKind.VIOLATED}
)


def _require_digest(name: str, value: object) -> str:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a SHA-256 or namespaced stable digest")
    return value


def _require_raw_counter(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _require_fields(
    name: str,
    value: object,
    *,
    reject_strategy_keys: bool,
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
    if reject_strategy_keys and set(keys) & _RESERVED_CONFIG_KEYS:
        raise ValueError(f"{name} cannot choose a comparison strategy")
    return value


def _comparison_partition_digest(
    *,
    kind: str,
    partition: tuple[tuple[str, ...], ...],
) -> str:
    """Use the same domain-separated shape that R3 Root later re-derives."""

    return stable_digest(
        "osc-phase6-contract-comparison-partition-binding-v1",
        {"kind": kind, "partition": partition},
    )


@dataclass(frozen=True, slots=True)
class RawContractPerformanceCounters:
    """One arm's unaggregated counter snapshot for one exact comparison."""

    materialized_bytes: int
    backend_pair_comparisons: int
    comparison_cpu_ns: int

    def __post_init__(self) -> None:
        _require_raw_counter("materialized bytes", self.materialized_bytes)
        _require_raw_counter(
            "backend pair comparisons", self.backend_pair_comparisons
        )
        _require_raw_counter("comparison CPU nanoseconds", self.comparison_cpu_ns)


@dataclass(frozen=True, slots=True)
class ContractPerformanceBenchmarkContext:
    """Immutable non-authoritative environment and configuration context."""

    benchmark_plan: tuple[tuple[str, str], ...]
    environment: tuple[tuple[str, str], ...]
    baseline_config: tuple[tuple[str, str], ...]
    treatment_config: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        _require_fields(
            "benchmark plan", self.benchmark_plan, reject_strategy_keys=False
        )
        _require_fields("environment", self.environment, reject_strategy_keys=False)
        _require_fields(
            "baseline configuration", self.baseline_config, reject_strategy_keys=True
        )
        _require_fields(
            "treatment configuration", self.treatment_config, reject_strategy_keys=True
        )


@dataclass(frozen=True, slots=True)
class ContractPerformanceProductionObservation:
    """Exact typed inputs required to derive one R3 raw-pair receipt.

    ``contract_comparison_producer_receipt_digest`` is intentionally opaque at
    this Runtime boundary.  It is retained solely for R3 Root's later exact
    cross-object check; this pure builder cannot use it as provenance.
    """

    comparison: ContractComparisonReceipt
    contract_comparison_producer_receipt_digest: str
    benchmark_context: ContractPerformanceBenchmarkContext
    baseline_result: TypedExecutionResult
    treatment_result: TypedExecutionResult
    baseline_evidence: EvidenceEnvelope
    treatment_evidence: EvidenceEnvelope
    baseline_counters: RawContractPerformanceCounters
    treatment_counters: RawContractPerformanceCounters

    def __post_init__(self) -> None:
        if not isinstance(self.comparison, ContractComparisonReceipt):
            raise ValueError("production observation requires a comparison receipt")
        _require_digest(
            "contract comparison producer receipt digest",
            self.contract_comparison_producer_receipt_digest,
        )
        if not isinstance(self.benchmark_context, ContractPerformanceBenchmarkContext):
            raise ValueError("production observation requires benchmark context")
        for name, value, expected_type in (
            ("baseline result", self.baseline_result, TypedExecutionResult),
            ("treatment result", self.treatment_result, TypedExecutionResult),
            ("baseline evidence", self.baseline_evidence, EvidenceEnvelope),
            ("treatment evidence", self.treatment_evidence, EvidenceEnvelope),
            (
                "baseline counters",
                self.baseline_counters,
                RawContractPerformanceCounters,
            ),
            (
                "treatment counters",
                self.treatment_counters,
                RawContractPerformanceCounters,
            ),
        ):
            if not isinstance(value, expected_type):
                raise ValueError(f"{name} must be {expected_type.__name__}")
        _validate_production_observation(self)


def _require_evidence_binding(
    *,
    name: str,
    evidence: EvidenceEnvelope,
    result: TypedExecutionResult,
    expected_verdict: VerdictKind,
) -> None:
    if evidence.state is not EvidenceState.NOT_A_CANDIDATE:
        raise ValueError(f"{name} evidence must remain NOT_A_CANDIDATE")
    if evidence.verdict_kind is not expected_verdict:
        raise ValueError(f"{name} evidence verdict mismatches exact comparison")
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
    outcomes = {item.endpoint_id: item for item in evidence.execution_outcomes}
    if outcomes.get(result.outcome.endpoint_id) != result.outcome:
        raise ValueError(f"{name} evidence/outcome mismatch")


def _validate_production_observation(
    observation: ContractPerformanceProductionObservation,
) -> None:
    comparison = observation.comparison
    baseline = observation.baseline_result
    treatment = observation.treatment_result

    if comparison.exact_result.verdict_kind not in _AUTHORITATIVE_VERDICTS:
        raise ValueError("comparison receipt must have an authoritative exact verdict")
    for name, value in (("baseline", baseline), ("treatment", treatment)):
        if value.outcome.status is not ExecutionStatus.OK:
            raise ValueError(f"{name} execution requires an OK outcome")
        if value.task.identity.task_kind is not TaskKind.FULL_DIFF:
            raise ValueError(f"{name} execution requires a FULL_DIFF task")
        if value.result_group != comparison.result_group:
            raise ValueError(f"{name} result group does not match comparison receipt")
        if value.seed_lineage.digest != comparison.exact_task.identity.seed_lineage_digest:
            raise ValueError(f"{name} seed lineage does not match comparison receipt")
        if value.task.identity.protocol_digest != comparison.exact_task.identity.protocol_digest:
            raise ValueError(f"{name} protocol does not match comparison receipt")
        if value.outcome.endpoint_id not in tuple(
            endpoint.endpoint_id for endpoint in comparison.endpoints
        ):
            raise ValueError(f"{name} outcome endpoint does not match comparison receipt")

    shared_fields = (
        ("case", baseline.case_id, treatment.case_id),
        ("result group", baseline.result_group, treatment.result_group),
        ("seed lineage", baseline.seed_lineage, treatment.seed_lineage),
        (
            "protocol",
            baseline.task.identity.protocol_digest,
            treatment.task.identity.protocol_digest,
        ),
        ("outcome endpoint", baseline.outcome.endpoint_id, treatment.outcome.endpoint_id),
        ("resource envelope", baseline.task.resources, treatment.task.resources),
    )
    for field, left, right in shared_fields:
        if left != right:
            raise ValueError(f"baseline and treatment {field} mismatch")

    distinct_fields = (
        ("task", baseline.task.identity.task_id, treatment.task.identity.task_id),
        ("TaskSpec", baseline.task.digest, treatment.task.digest),
        ("result", baseline.result_id, treatment.result_id),
        ("outcome", baseline.outcome.digest, treatment.outcome.digest),
        ("evidence", observation.baseline_evidence.digest, observation.treatment_evidence.digest),
    )
    for field, baseline_value, treatment_value in distinct_fields:
        if baseline_value == treatment_value:
            raise ValueError(f"baseline and treatment {field} identities must differ")

    _require_evidence_binding(
        name="baseline",
        evidence=observation.baseline_evidence,
        result=baseline,
        expected_verdict=comparison.exact_result.verdict_kind,
    )
    _require_evidence_binding(
        name="treatment",
        evidence=observation.treatment_evidence,
        result=treatment,
        expected_verdict=comparison.exact_result.verdict_kind,
    )

    for name, counters in (
        ("baseline", observation.baseline_counters),
        ("treatment", observation.treatment_counters),
    ):
        for counter_name, value in (
            ("materialized bytes", counters.materialized_bytes),
            ("backend pair comparisons", counters.backend_pair_comparisons),
            ("comparison CPU nanoseconds", counters.comparison_cpu_ns),
        ):
            _require_raw_counter(f"{name} {counter_name}", value)
    if any(
        value == 0
        for value in (
            observation.baseline_counters.materialized_bytes,
            observation.baseline_counters.backend_pair_comparisons,
            observation.baseline_counters.comparison_cpu_ns,
        )
    ):
        raise ValueError("baseline raw counters must be positive")


def _arm_config_digest(
    *,
    strategy: str,
    result: TypedExecutionResult,
    fields: tuple[tuple[str, str], ...],
) -> str:
    return stable_digest(
        "osc-phase6-contract-performance-arm-config-v1",
        {
            "strategy": strategy,
            "task_spec_digest": result.task.digest,
            "task_payload_digest": result.task.payload_digest,
            "resource_envelope_digest": result.task.resources.digest,
            "fields": fields,
        },
    )


def _comparison_algorithm_digest(comparison: ContractComparisonReceipt) -> str:
    return stable_digest(
        "osc-phase6-contract-performance-comparison-algorithm-v1",
        {
            "comparison_schema_version": comparison.schema_version,
            "contract_digest": comparison.contract.digest,
            "fingerprint_task_kind": comparison.fingerprint_task.identity.task_kind.value,
            "fingerprint_plan_digest": comparison.fingerprint_result.plan_digest,
            "exact_task_kind": comparison.exact_task.identity.task_kind.value,
            "exact_plan_digest": comparison.exact_result.plan_digest,
            "exact_escalated": comparison.exact_result.exact_escalated,
        },
    )


def build_contract_performance_evidence_receipt(
    observation: ContractPerformanceProductionObservation,
) -> ContractPerformanceEvidenceReceipt:
    """Derive one private R3 receipt from fully typed, non-authoritative input."""

    if not isinstance(observation, ContractPerformanceProductionObservation):
        raise TypeError("observation must be a ContractPerformanceProductionObservation")
    comparison = observation.comparison
    baseline = observation.baseline_result
    treatment = observation.treatment_result
    context = observation.benchmark_context
    return ContractPerformanceEvidenceReceipt(
        source_snapshot_digest=comparison.source_snapshot_digest,
        source_case_digest=comparison.source_case_digest,
        comparison_decision_id=comparison.comparison_decision_id,
        contract_comparison_producer_receipt_digest=(
            observation.contract_comparison_producer_receipt_digest
        ),
        result_group_id=comparison.result_group.result_group_id,
        result_group_digest=comparison.result_group.digest,
        contract_fingerprint_digest=comparison.result_group.contract_fingerprint.digest,
        endpoint_set_digest=comparison.endpoint_set_digest,
        fingerprint_task_id=comparison.fingerprint_task.identity.task_id,
        fingerprint_result_digest=comparison.fingerprint_result.digest,
        exact_task_id=comparison.exact_task.identity.task_id,
        exact_result_digest=comparison.exact_result.digest,
        fingerprint_partition_digest=_comparison_partition_digest(
            kind="fingerprint", partition=comparison.fingerprint_partition
        ),
        canonical_partition_digest=_comparison_partition_digest(
            kind="canonical", partition=comparison.pairwise_canonical_partition
        ),
        comparison_algorithm_digest=_comparison_algorithm_digest(comparison),
        benchmark_plan_digest=stable_digest(
            "osc-phase6-contract-performance-benchmark-plan-v1",
            context.benchmark_plan,
        ),
        environment_digest=stable_digest(
            "osc-phase6-contract-performance-environment-v1", context.environment
        ),
        baseline_config_digest=_arm_config_digest(
            strategy=BASELINE_STRATEGY,
            result=baseline,
            fields=context.baseline_config,
        ),
        treatment_config_digest=_arm_config_digest(
            strategy=TREATMENT_STRATEGY,
            result=treatment,
            fields=context.treatment_config,
        ),
        baseline_task_id=baseline.task.identity.task_id,
        baseline_task_spec_digest=baseline.task.digest,
        baseline_result_digest=baseline.result_id,
        baseline_seed_lineage_digest=baseline.seed_lineage.digest,
        baseline_execution_outcome_digest=baseline.outcome.digest,
        baseline_evidence_envelope_digest=observation.baseline_evidence.digest,
        treatment_task_id=treatment.task.identity.task_id,
        treatment_task_spec_digest=treatment.task.digest,
        treatment_result_digest=treatment.result_id,
        treatment_seed_lineage_digest=treatment.seed_lineage.digest,
        treatment_execution_outcome_digest=treatment.outcome.digest,
        treatment_evidence_envelope_digest=observation.treatment_evidence.digest,
        baseline_materialized_bytes=observation.baseline_counters.materialized_bytes,
        treatment_materialized_bytes=observation.treatment_counters.materialized_bytes,
        baseline_backend_pair_comparisons=(
            observation.baseline_counters.backend_pair_comparisons
        ),
        treatment_backend_pair_comparisons=(
            observation.treatment_counters.backend_pair_comparisons
        ),
        baseline_comparison_cpu_ns=observation.baseline_counters.comparison_cpu_ns,
        treatment_comparison_cpu_ns=observation.treatment_counters.comparison_cpu_ns,
    )


__all__ = [
    "ContractPerformanceBenchmarkContext",
    "ContractPerformanceProductionObservation",
    "RawContractPerformanceCounters",
    "build_contract_performance_evidence_receipt",
]
