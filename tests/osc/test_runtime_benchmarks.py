from __future__ import annotations

import time

import pytest

from datadiff_osc._canonical import stable_digest
from datadiff_osc.contract_engine.fingerprints import ComponentFingerprint
from datadiff_osc.parallel.executor import WorkerResult
from datadiff_osc.parallel.invariance import AuthorityEvidenceSnapshot
from datadiff_osc.parallel.resources import ResourceCapacity
from datadiff_osc.runtime.benchmarks import (
    audit_staged_exact_parity,
    benchmark_component_clustering,
    benchmark_parallel_scaling,
    percentile,
)
from datadiff_osc.schemas import ExecutionStatus, FailureKind, StructuredExecutionOutcome, VerdictKind

from test_parallel_dag import make_task


def _authority_snapshot(run, _worker_count):
    rows = tuple(
        (
            record.logical_task_digest,
            record.outcome.digest,
            record.payload_digest,
            record.authority_eligible,
        )
        for record in run.records
    )
    return AuthorityEvidenceSnapshot.from_run(
        run,
        authority_verdict_digest=stable_digest("benchmark-verdict", rows),
        coverage_bitmap_digest=stable_digest(
            "benchmark-coverage", tuple((row[0], row[3]) for row in rows)
        ),
        ledger_digest=stable_digest("benchmark-ledger", rows),
        certificate_digests=tuple(
            stable_digest("benchmark-certificate", row) for row in rows
        ),
    )


def test_parity_benchmark_records_discrepancies_without_relabeling_them():
    samples = tuple(range(5))

    def staged(value):
        return (VerdictKind.VIOLATED if value == 2 else VerdictKind.SATISFIED, value == 4)

    def exact(_):
        return VerdictKind.SATISFIED

    result = audit_staged_exact_parity(
        samples,
        sample_id=lambda value: f"sample-{value}",
        staged=staged,
        exact=exact,
    )
    assert result.group_count == 5
    assert result.discrepancy_count == 1
    assert result.discrepancy_ids == ("sample-2",)
    assert result.exact_escalation_count == 1
    assert result.zero_discrepancy is False
    assert result.phase6_sample_floor_met is False


def _fingerprints(count):
    return tuple(
        ComponentFingerprint(
            endpoint_id=f"endpoint-{index}",
            observer_id="bag",
            observer_digest="observer-1",
            contract_digest="contract-1",
            row_count=4,
            schema_digest="schema-1",
            payload_digest=f"payload-{index % 2}",
        )
        for index in range(count)
    )


def test_clustering_benchmark_matches_explicit_pairwise_partition():
    report = benchmark_component_clustering(
        (_fingerprints(2), _fingerprints(8)), repetitions=5
    )
    assert [item.backend_count for item in report.points] == [2, 8]
    assert report.all_pairwise_partitions_match
    assert all(item.nanoseconds_per_endpoint > 0 for item in report.points)


def test_parallel_scaling_records_raw_times_and_invariance():
    tasks = tuple(make_task(index) for index in range(6))
    capacity = ResourceCapacity.build(
        cpu_tokens=4,
        rss_bytes=4096,
        io_slots={"default": 4},
        backend_internal_threads=4,
    )

    def worker_factory(_):
        def worker(invocation):
            time.sleep(0.0005)
            outcome = StructuredExecutionOutcome(
                endpoint_id=invocation.task.identity.endpoint_id,
                status=ExecutionStatus.OK,
                failure_kind=FailureKind.NONE,
            )
            return WorkerResult(
                outcome,
                stable_digest("benchmark-output", invocation.task.payload_digest),
            )

        return worker

    report = benchmark_parallel_scaling(
        tasks,
        worker_factory,
        capacity=capacity,
        worker_counts=(1, 2),
        assignment_digest="assignment-1",
        authority_evidence_factory=_authority_snapshot,
    )
    assert report.invariance.passed
    assert report.invariance.authority_evidence_complete
    assert [item.workers for item in report.points] == [1, 2]
    assert all(item.elapsed_seconds > 0 for item in report.points)
    assert all(item.tasks_per_second > 0 for item in report.points)


@pytest.mark.parametrize("worker_counts", [(1,), (1, 1), (2, 6)])
def test_parallel_benchmark_rejects_invalid_design_before_worker_factory(
    worker_counts,
):
    calls = []

    def worker_factory(count):
        calls.append(count)

        def worker(invocation):
            outcome = StructuredExecutionOutcome(
                endpoint_id=invocation.task.identity.endpoint_id,
                status=ExecutionStatus.OK,
                failure_kind=FailureKind.NONE,
            )
            return WorkerResult(outcome, "payload")

        return worker

    with pytest.raises(ValueError, match="worker count"):
        benchmark_parallel_scaling(
            (make_task(0),),
            worker_factory,
            capacity=ResourceCapacity.build(
                cpu_tokens=6,
                rss_bytes=4096,
                io_slots={"default": 6},
                backend_internal_threads=6,
            ),
            worker_counts=worker_counts,
            assignment_digest="assignment-1",
            authority_evidence_factory=_authority_snapshot,
        )
    assert calls == []


def test_percentile_is_deterministic_and_validates_range():
    assert percentile((1.0, 2.0, 3.0), 50.0) == 2.0
    assert percentile((), 95.0) == 0.0
