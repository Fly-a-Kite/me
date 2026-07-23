"""Private raw parallel-scaling receipt for the Phase-6 authority boundary.

The receipt records one 1-worker versus 6-worker measurement shape.  It is
not an artifact, producer provenance record, performance conclusion, or gate
decision; Root must independently re-read and bind it before it can have any
future authority role.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from datadiff_osc._canonical import stable_digest, to_primitive


PARALLEL_SCALING_EVIDENCE_RECEIPT_SCHEMA_VERSION = (
    "osc-phase6-parallel-scaling-evidence-receipt-v1"
)

_DIGEST_RE = re.compile(r"(?:[A-Za-z0-9_.:-]+-)?[0-9a-f]{64}")


def _require_text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_digest(name: str, value: object) -> str:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a SHA-256 or namespaced stable digest")
    return value


def _require_nonnegative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


@dataclass(frozen=True, slots=True)
class ParallelScalingEvidenceReceipt:
    """One raw 1-worker/6-worker measurement, without authority by itself."""

    source_snapshot_digest: str
    workload_id: str
    benchmark_plan_digest: str
    environment_digest: str
    one_worker_config_digest: str
    six_worker_config_digest: str
    one_worker_task_set_digest: str
    six_worker_task_set_digest: str
    one_worker_result_digest: str
    six_worker_result_digest: str
    one_worker_evidence_digest: str
    six_worker_evidence_digest: str
    one_worker_clock_id: str
    one_worker_start_ns: int
    one_worker_end_ns: int
    one_worker_task_counter_start: int
    one_worker_task_counter_end: int
    six_worker_clock_id: str
    six_worker_start_ns: int
    six_worker_end_ns: int
    six_worker_task_counter_start: int
    six_worker_task_counter_end: int
    one_worker_count: int = 1
    six_worker_count: int = 6
    schema_version: str = PARALLEL_SCALING_EVIDENCE_RECEIPT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in (
            "source_snapshot_digest",
            "benchmark_plan_digest",
            "environment_digest",
            "one_worker_config_digest",
            "six_worker_config_digest",
            "one_worker_task_set_digest",
            "six_worker_task_set_digest",
            "one_worker_result_digest",
            "six_worker_result_digest",
            "one_worker_evidence_digest",
            "six_worker_evidence_digest",
        ):
            _require_digest(name, getattr(self, name))
        for name in ("workload_id", "one_worker_clock_id", "six_worker_clock_id"):
            _require_text(name, getattr(self, name))
        if self.schema_version != PARALLEL_SCALING_EVIDENCE_RECEIPT_SCHEMA_VERSION:
            raise ValueError("parallel scaling receipt schema version mismatch")
        if self.one_worker_count != 1 or self.six_worker_count != 6:
            raise ValueError("parallel scaling worker counts must be exactly 1 and 6")
        if self.one_worker_config_digest == self.six_worker_config_digest:
            raise ValueError("parallel scaling arm configurations must differ")
        if self.one_worker_task_set_digest != self.six_worker_task_set_digest:
            raise ValueError("parallel scaling task-set identity mismatch")
        for name, left, right in (
            ("result", self.one_worker_result_digest, self.six_worker_result_digest),
            ("evidence", self.one_worker_evidence_digest, self.six_worker_evidence_digest),
        ):
            if left == right:
                raise ValueError(f"parallel scaling arm {name} identities must differ")
        for name in (
            "one_worker_start_ns",
            "one_worker_end_ns",
            "one_worker_task_counter_start",
            "one_worker_task_counter_end",
            "six_worker_start_ns",
            "six_worker_end_ns",
            "six_worker_task_counter_start",
            "six_worker_task_counter_end",
        ):
            _require_nonnegative_int(name, getattr(self, name))
        if self.one_worker_clock_id != self.six_worker_clock_id:
            raise ValueError("parallel scaling monotonic clocks must match")
        if self.one_worker_end_ns <= self.one_worker_start_ns:
            raise ValueError("one-worker interval must be positive")
        if self.six_worker_end_ns <= self.six_worker_start_ns:
            raise ValueError("six-worker interval must be positive")
        if self.one_worker_task_counter_end <= self.one_worker_task_counter_start:
            raise ValueError("one-worker counter must make positive progress")
        if self.six_worker_task_counter_end <= self.six_worker_task_counter_start:
            raise ValueError("six-worker counter must make positive progress")
        if self.one_worker_task_count != self.six_worker_task_count:
            raise ValueError("parallel scaling counters must measure the same task count")

    @property
    def one_worker_elapsed_ns(self) -> int:
        return self.one_worker_end_ns - self.one_worker_start_ns

    @property
    def six_worker_elapsed_ns(self) -> int:
        return self.six_worker_end_ns - self.six_worker_start_ns

    @property
    def one_worker_task_count(self) -> int:
        return self.one_worker_task_counter_end - self.one_worker_task_counter_start

    @property
    def six_worker_task_count(self) -> int:
        return self.six_worker_task_counter_end - self.six_worker_task_counter_start

    @property
    def sample_id(self) -> str:
        return stable_digest("osc-phase6-parallel-scaling-sample-id-v1", self)

    @property
    def digest(self) -> str:
        return stable_digest("osc-phase6-parallel-scaling-evidence-receipt", self)

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(self)


__all__ = [
    "PARALLEL_SCALING_EVIDENCE_RECEIPT_SCHEMA_VERSION",
    "ParallelScalingEvidenceReceipt",
]
