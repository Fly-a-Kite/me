from __future__ import annotations

from dataclasses import replace
import time

import pytest

from datadiff_osc._canonical import stable_digest
from datadiff_osc.parallel.executor import (
    DeterministicTaskExecutor,
    RetryPolicy,
    WorkerCrashError,
    WorkerResult,
)
from datadiff_osc.parallel.invariance import (
    AuthorityEvidenceSnapshot,
    audit_execution_runs,
    run_worker_count_invariance,
)
from datadiff_osc.parallel.resources import ResourceCapacity
from datadiff_osc.schemas import ExecutionStatus, FailureKind, StructuredExecutionOutcome

from test_parallel_dag import make_task


def _capacity():
    return ResourceCapacity.build(
        cpu_tokens=6,
        rss_bytes=4096,
        io_slots={"default": 6},
        backend_internal_threads=6,
    )


def _authority_snapshot(run, _worker_count=0):
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
        authority_verdict_digest=stable_digest("test-authority-verdict", rows),
        coverage_bitmap_digest=stable_digest(
            "test-coverage-bitmap", tuple((row[0], row[3]) for row in rows)
        ),
        ledger_digest=stable_digest("test-ledger", rows),
        certificate_digests=tuple(
            stable_digest("test-certificate", row) for row in rows
        ),
    )


def _worker_factory(worker_count):
    def worker(invocation):
        index = invocation.task.identity.decision_index
        if index == 3 and invocation.task.identity.attempt == 0:
            raise WorkerCrashError("deterministic injected crash")
        time.sleep(((7 - index) if worker_count > 1 else index) * 0.0001)
        outcome = StructuredExecutionOutcome(
            endpoint_id=invocation.task.identity.endpoint_id,
            status=ExecutionStatus.OK,
            failure_kind=FailureKind.NONE,
        )
        return WorkerResult(
            outcome,
            stable_digest(
                "deterministic-worker-output", invocation.task.payload_digest
            ),
        )

    return worker


def _runs():
    tasks = tuple(make_task(index) for index in range(8))
    return run_worker_count_invariance(
        reversed(tasks),
        _worker_factory,
        capacity=_capacity(),
        worker_counts=(1, 2),
        retry_policy=RetryPolicy(max_retries=1),
        assignment_digest="assignment-1",
        authority_evidence_factory=_authority_snapshot,
    )


def test_workers_retry_and_completion_order_do_not_change_authority_outputs():
    tasks = tuple(make_task(index) for index in range(8))
    report, runs = run_worker_count_invariance(
        reversed(tasks),
        _worker_factory,
        capacity=_capacity(),
        worker_counts=(1, 2, 4, 6),
        retry_policy=RetryPolicy(max_retries=1),
        assignment_digest="assignment-1",
        authority_evidence_factory=_authority_snapshot,
    )
    assert report.passed
    assert report.authority_evidence_complete
    assert report.assignment_invariant
    assert report.epoch_invariant
    assert report.task_multiset_invariant
    assert report.authority_verdict_invariant
    assert report.coverage_bitmap_invariant
    assert report.ledger_invariant
    assert report.certificate_invariant
    assert report.worker_counts == tuple(run.worker_count for run in runs)
    assert all(run.schema_version == "osc-deterministic-execution-run-v2" for run in runs)
    assert all(
        _authority_snapshot(run).schema_version
        == "osc-authority-evidence-snapshot-v2"
        for run in runs
    )
    assert all(run.initial_task_ids == runs[0].initial_task_ids for run in runs)
    assert sum(len(record.attempts) == 2 for record in runs[0].records) == 1


def test_invariance_audit_fails_closed_on_changed_execution_outcome():
    _, runs = _runs()
    changed_attempt = replace(runs[1].records[0].attempts[0], payload_digest="changed")
    changed_record = replace(runs[1].records[0], attempts=(changed_attempt,))
    changed_run = replace(runs[1], records=(changed_record, *runs[1].records[1:]))
    report = audit_execution_runs(
        (runs[0], changed_run),
        worker_counts=(1, 2),
        authority_evidence=(
            _authority_snapshot(runs[0]),
            _authority_snapshot(changed_run),
        ),
    )
    assert report.passed is False
    assert report.outcome_invariant is False
    assert report.authority_verdict_invariant is False


def test_assignment_epoch_task_multiset_and_seed_changes_each_fail():
    _, runs = _runs()
    baseline = runs[0]
    comparison = runs[1]
    changed_record = replace(
        comparison.records[0], logical_task_digest="changed-logical-task"
    )
    changed_seed_record = replace(
        comparison.records[0], seed_lineage_digest="changed-seed-lineage"
    )
    cases = (
        (
            "assignment_invariant",
            replace(comparison, assignment_digest="assignment-2"),
        ),
        ("epoch_invariant", replace(comparison, epoch_digest="epoch-2")),
        (
            "task_multiset_invariant",
            replace(comparison, records=(changed_record, *comparison.records[1:])),
        ),
        (
            "seed_lineage_invariant",
            replace(
                comparison,
                records=(changed_seed_record, *comparison.records[1:]),
            ),
        ),
    )
    for field, changed_run in cases:
        report = audit_execution_runs(
            (baseline, changed_run),
            worker_counts=(1, 2),
            authority_evidence=(
                _authority_snapshot(baseline),
                _authority_snapshot(changed_run),
            ),
        )
        assert report.passed is False
        assert getattr(report, field) is False


def test_verdict_coverage_ledger_and_certificate_changes_each_fail():
    _, runs = _runs()
    first = _authority_snapshot(runs[0])
    second = _authority_snapshot(runs[1])
    cases = (
        (
            "authority_verdict_invariant",
            replace(second, authority_verdict_digest="changed-verdict"),
        ),
        (
            "coverage_bitmap_invariant",
            replace(second, coverage_bitmap_digest="changed-coverage"),
        ),
        ("ledger_invariant", replace(second, ledger_digest="changed-ledger")),
        (
            "certificate_invariant",
            replace(second, certificate_digests=("changed-certificate",)),
        ),
    )
    for field, changed_snapshot in cases:
        report = audit_execution_runs(
            runs,
            worker_counts=(1, 2),
            authority_evidence=(first, changed_snapshot),
        )
        assert report.passed is False
        assert getattr(report, field) is False


def test_missing_or_cross_run_authority_evidence_fails_closed():
    _, runs = _runs()
    missing = audit_execution_runs(
        runs,
        worker_counts=(1, 2),
        authority_evidence=(_authority_snapshot(runs[0]),),
    )
    cross_run = audit_execution_runs(
        runs,
        worker_counts=(1, 2),
        authority_evidence=(
            _authority_snapshot(runs[0]),
            replace(
                _authority_snapshot(runs[1]),
                execution_run_digest="wrong-run-digest",
            ),
        ),
    )
    assert missing.passed is False
    assert missing.authority_evidence_complete is False
    assert cross_run.passed is False
    assert cross_run.authority_evidence_complete is False
    assert any("execution_run_digest_mismatch" in item for item in cross_run.authority_evidence_errors)


@pytest.mark.parametrize("worker_counts", [(1,), (1, 1), (2, 6)])
def test_invariance_design_rejects_vacuous_or_unanchored_worker_counts(
    worker_counts,
):
    tasks = (make_task(0),)
    with pytest.raises(ValueError, match="worker count"):
        run_worker_count_invariance(
            tasks,
            _worker_factory,
            capacity=_capacity(),
            worker_counts=worker_counts,
            retry_policy=RetryPolicy(max_retries=0),
            assignment_digest="assignment-1",
            authority_evidence_factory=_authority_snapshot,
        )


def test_duplicate_workers_one_run_cannot_be_relabelled_as_one_and_six():
    run = DeterministicTaskExecutor(max_workers=1, capacity=_capacity()).execute(
        (make_task(0),),
        _worker_factory(1),
        assignment_digest="assignment-1",
    )
    snapshot = _authority_snapshot(run)
    report = audit_execution_runs(
        (run, run),
        worker_counts=(1, 6),
        authority_evidence=(snapshot, snapshot),
    )
    assert report.passed is False
    assert report.authority_evidence_complete is False
    assert any(
        "worker_count_label_mismatch" in error
        for error in report.authority_evidence_errors
    )


def test_authority_snapshot_worker_count_is_run_bound():
    _, runs = _runs()
    first = _authority_snapshot(runs[0])
    assert first.worker_count == runs[0].worker_count
    rebound = replace(first, worker_count=runs[1].worker_count)
    errors = rebound.binding_errors(runs[0])
    assert "worker_count_mismatch" in errors
