"""Private raw paired observations for the Phase-6 Contract gate.

This module deliberately models one pair only.  A receipt supplies exact
identity and raw-observation context for later Root admission, but cannot by
itself establish producer provenance, a 1000-pair cohort, a bootstrap result,
or a gate decision.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from datadiff_osc._canonical import stable_digest, to_primitive


CONTRACT_PERFORMANCE_EVIDENCE_RECEIPT_SCHEMA_VERSION = (
    "osc-phase6-contract-performance-evidence-receipt-v1"
)
BASELINE_STRATEGY = "always_materialize_pairwise"
TREATMENT_STRATEGY = "staged_exact_escalation"

_DIGEST_RE = re.compile(r"(?:[A-Za-z0-9_.:-]+-)?[0-9a-f]{64}")


def _require_text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_digest(name: str, value: object) -> str:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a SHA-256 or namespaced stable digest")
    return value


def _require_counter(name: str, value: object, *, positive: bool) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < 0 or (positive and value == 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be {qualifier}")
    return value


@dataclass(frozen=True, slots=True)
class ContractPerformanceEvidenceReceipt:
    """One non-authoritative baseline/treatment raw measurement pair.

    ``sample_id`` is derived from all fields; callers therefore cannot reuse a
    sample identity for changed context or counters.  Root must still re-read
    canonical bytes and bind this private value to its independently admitted
    comparison receipt before it can contribute to a denominator.
    """

    source_snapshot_digest: str
    source_case_digest: str
    comparison_decision_id: str
    contract_comparison_producer_receipt_digest: str
    result_group_id: str
    result_group_digest: str
    contract_fingerprint_digest: str
    endpoint_set_digest: str
    fingerprint_task_id: str
    fingerprint_result_digest: str
    exact_task_id: str
    exact_result_digest: str
    fingerprint_partition_digest: str
    canonical_partition_digest: str
    comparison_algorithm_digest: str
    benchmark_plan_digest: str
    environment_digest: str
    baseline_config_digest: str
    treatment_config_digest: str
    baseline_task_id: str
    baseline_task_spec_digest: str
    baseline_result_digest: str
    baseline_seed_lineage_digest: str
    baseline_execution_outcome_digest: str
    baseline_evidence_envelope_digest: str
    treatment_task_id: str
    treatment_task_spec_digest: str
    treatment_result_digest: str
    treatment_seed_lineage_digest: str
    treatment_execution_outcome_digest: str
    treatment_evidence_envelope_digest: str
    baseline_materialized_bytes: int
    treatment_materialized_bytes: int
    baseline_backend_pair_comparisons: int
    treatment_backend_pair_comparisons: int
    baseline_comparison_cpu_ns: int
    treatment_comparison_cpu_ns: int
    baseline_strategy: str = BASELINE_STRATEGY
    treatment_strategy: str = TREATMENT_STRATEGY
    schema_version: str = CONTRACT_PERFORMANCE_EVIDENCE_RECEIPT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in (
            "source_snapshot_digest",
            "source_case_digest",
            "comparison_decision_id",
            "contract_comparison_producer_receipt_digest",
            "result_group_digest",
            "contract_fingerprint_digest",
            "endpoint_set_digest",
            "fingerprint_result_digest",
            "exact_result_digest",
            "fingerprint_partition_digest",
            "canonical_partition_digest",
            "comparison_algorithm_digest",
            "benchmark_plan_digest",
            "environment_digest",
            "baseline_config_digest",
            "treatment_config_digest",
            "baseline_task_spec_digest",
            "baseline_result_digest",
            "baseline_seed_lineage_digest",
            "baseline_execution_outcome_digest",
            "baseline_evidence_envelope_digest",
            "treatment_task_spec_digest",
            "treatment_result_digest",
            "treatment_seed_lineage_digest",
            "treatment_execution_outcome_digest",
            "treatment_evidence_envelope_digest",
        ):
            _require_digest(name, getattr(self, name))
        for name in (
            "result_group_id",
            "fingerprint_task_id",
            "exact_task_id",
            "baseline_task_id",
            "treatment_task_id",
        ):
            _require_text(name, getattr(self, name))

        if self.schema_version != CONTRACT_PERFORMANCE_EVIDENCE_RECEIPT_SCHEMA_VERSION:
            raise ValueError("contract performance receipt schema version mismatch")
        if self.baseline_strategy != BASELINE_STRATEGY:
            raise ValueError("baseline strategy must always materialize pairwise")
        if self.treatment_strategy != TREATMENT_STRATEGY:
            raise ValueError("treatment strategy must use staged exact escalation")
        if self.baseline_config_digest == self.treatment_config_digest:
            raise ValueError("baseline and treatment configurations must differ")

        pairs = (
            ("task", self.baseline_task_id, self.treatment_task_id),
            ("TaskSpec", self.baseline_task_spec_digest, self.treatment_task_spec_digest),
            ("result", self.baseline_result_digest, self.treatment_result_digest),
            (
                "execution outcome",
                self.baseline_execution_outcome_digest,
                self.treatment_execution_outcome_digest,
            ),
            (
                "evidence envelope",
                self.baseline_evidence_envelope_digest,
                self.treatment_evidence_envelope_digest,
            ),
        )
        for name, baseline, treatment in pairs:
            if baseline == treatment:
                raise ValueError(f"baseline and treatment {name} identities must differ")
        if self.baseline_seed_lineage_digest != self.treatment_seed_lineage_digest:
            raise ValueError("baseline and treatment seed lineage must match")

        for name in (
            "baseline_materialized_bytes",
            "baseline_backend_pair_comparisons",
            "baseline_comparison_cpu_ns",
        ):
            _require_counter(name, getattr(self, name), positive=True)
        for name in (
            "treatment_materialized_bytes",
            "treatment_backend_pair_comparisons",
            "treatment_comparison_cpu_ns",
        ):
            _require_counter(name, getattr(self, name), positive=False)

    @property
    def sample_id(self) -> str:
        return stable_digest("osc-phase6-contract-performance-sample-id-v1", self)

    @property
    def digest(self) -> str:
        return stable_digest("osc-phase6-contract-performance-evidence-receipt", self)

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(self)


__all__ = [
    "BASELINE_STRATEGY",
    "CONTRACT_PERFORMANCE_EVIDENCE_RECEIPT_SCHEMA_VERSION",
    "ContractPerformanceEvidenceReceipt",
    "TREATMENT_STRATEGY",
]
