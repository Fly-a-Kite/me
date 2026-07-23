from __future__ import annotations

from collections import defaultdict
from threading import Lock
import time

import pytest

from datadiff_osc._canonical import stable_digest
from datadiff_osc.parallel.executor import (
    AdapterTaskError,
    DeterministicTaskExecutor,
    RetryPolicy,
    SemanticDomainTaskError,
    UnsupportedTaskError,
    WorkerCrashError,
    WorkerResult,
)
from datadiff_osc.parallel.resources import ResourceCapacity
from datadiff_osc.schemas import ExecutionStatus, FailureKind, ResourceTokens, StructuredExecutionOutcome

from test_parallel_dag import make_task


def _capacity():
    return ResourceCapacity.build(
        cpu_tokens=4,
        rss_bytes=1024,
        io_slots={"default": 4},
        backend_internal_threads=4,
    )


def _ok(invocation):
    endpoint = invocation.task.identity.endpoint_id
    outcome = StructuredExecutionOutcome(
        endpoint_id=endpoint,
        status=ExecutionStatus.OK,
        failure_kind=FailureKind.NONE,
    )
    return WorkerResult(outcome, stable_digest("payload", invocation.task.payload_digest))


def test_executor_merges_in_task_order_and_respects_dependency_barrier():
    first = make_task(0)
    second = make_task(1, dependencies=(first.identity.task_id,))
    seen = []

    def worker(invocation):
        seen.append((invocation.task.identity.decision_index, len(invocation.dependency_records)))
        return _ok(invocation)

    run = DeterministicTaskExecutor(max_workers=2, capacity=_capacity()).execute(
        (second, first), worker, assignment_digest="assignment-1"
    )
    assert run.initial_task_ids == tuple(sorted((first.identity.task_id, second.identity.task_id)))
    assert seen == [(0, 0), (1, 1)]
    assert all(record.outcome.status == ExecutionStatus.OK for record in run.records)
    assert run.worker_count == 2
    assert run.schema_version == "osc-deterministic-execution-run-v2"


def test_execution_run_worker_count_is_actual_and_digest_bound():
    task = make_task(0)
    run_one = DeterministicTaskExecutor(max_workers=1, capacity=_capacity()).execute(
        (task,), _ok, assignment_digest="assignment-1"
    )
    run_four = DeterministicTaskExecutor(max_workers=4, capacity=_capacity()).execute(
        (task,), _ok, assignment_digest="assignment-1"
    )
    assert run_one.worker_count == 1
    assert run_four.worker_count == 4
    assert run_one.digest != run_four.digest


def test_retry_changes_attempt_only_and_preserves_seed_and_final_result():
    task = make_task(0)
    calls = defaultdict(int)

    def worker(invocation):
        logical = invocation.task.identity.seed_lineage_digest
        calls[logical] += 1
        if invocation.task.identity.attempt == 0:
            raise WorkerCrashError("crash once")
        return _ok(invocation)

    run = DeterministicTaskExecutor(
        max_workers=2,
        capacity=_capacity(),
        retry_policy=RetryPolicy(max_retries=1),
    ).execute((task,), worker, assignment_digest="assignment-1")
    record = run.records[0]
    assert [item.attempt for item in record.attempts] == [0, 1]
    assert [item.outcome.status for item in record.attempts] == [ExecutionStatus.CRASH, ExecutionStatus.OK]
    assert record.seed_lineage_digest == task.identity.seed_lineage_digest
    assert calls[task.identity.seed_lineage_digest] == 2


@pytest.mark.parametrize(
    ("error", "status", "kind"),
    [
        (SemanticDomainTaskError("domain"), ExecutionStatus.SEMANTIC_ERROR, FailureKind.SEMANTIC_DOMAIN_ERROR),
        (TimeoutError("late"), ExecutionStatus.TIMEOUT, FailureKind.TIMEOUT),
        (WorkerCrashError("gone"), ExecutionStatus.CRASH, FailureKind.CRASH),
        (AdapterTaskError("adapter"), ExecutionStatus.ADAPTER_ERROR, FailureKind.ADAPTER_ERROR),
    ],
)
def test_worker_failures_remain_structurally_distinct(error, status, kind):
    def worker(_):
        raise error

    run = DeterministicTaskExecutor(
        max_workers=1,
        capacity=_capacity(),
        retry_policy=RetryPolicy(max_retries=0),
    ).execute((make_task(0),), worker, assignment_digest="assignment-1")
    assert run.records[0].outcome.status == status
    assert run.records[0].outcome.failure_kind == kind


def test_unsupported_requires_and_preserves_capability_evidence():
    def worker(_):
        raise UnsupportedTaskError("not supported", evidence_digest="capability-evidence-1")

    outcome = DeterministicTaskExecutor(max_workers=1, capacity=_capacity()).execute(
        (make_task(0),), worker, assignment_digest="assignment-1"
    ).records[0].outcome
    assert outcome.status == ExecutionStatus.UNSUPPORTED
    assert outcome.failure_kind == FailureKind.UNSUPPORTED_CAPABILITY
    assert outcome.unsupported_evidence_digest == "capability-evidence-1"


def test_exclusive_state_serializes_workers_without_hidden_state_leak():
    resources = ResourceTokens(
        cpu_tokens=1,
        rss_bytes=16,
        io_class="default",
        backend_internal_threads=0,
        exclusive_state="catalog-a",
    )
    tasks = tuple(make_task(index, resources=resources) for index in range(4))
    lock = Lock()
    active = 0
    maximum = 0

    def worker(invocation):
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        time.sleep(0.002)
        with lock:
            active -= 1
        return _ok(invocation)

    DeterministicTaskExecutor(max_workers=4, capacity=_capacity()).execute(
        tasks, worker, assignment_digest="assignment-1"
    )
    assert maximum == 1


def test_none_result_is_missing_and_retry_is_bounded():
    run = DeterministicTaskExecutor(
        max_workers=1,
        capacity=_capacity(),
        retry_policy=RetryPolicy(max_retries=2),
    ).execute((make_task(0),), lambda _: None, assignment_digest="assignment-1")
    record = run.records[0]
    assert len(record.attempts) == 3
    assert all(item.outcome.status == ExecutionStatus.MISSING for item in record.attempts)


@pytest.mark.parametrize(
    "failure",
    [
        SemanticDomainTaskError("domain"),
        UnsupportedTaskError("unsupported", evidence_digest="capability-evidence"),
        TimeoutError("late"),
        WorkerCrashError("crash"),
        AdapterTaskError("adapter"),
        None,
    ],
)
def test_any_non_ok_prerequisite_blocks_child_without_invoking_worker(failure):
    parent = make_task(0)
    child = make_task(1, dependencies=(parent.identity.task_id,))
    calls = []

    def worker(invocation):
        calls.append(invocation.task.identity.task_id)
        if invocation.task.identity.task_id == parent.identity.task_id:
            if failure is None:
                return None
            raise failure
        return _ok(invocation)

    run = DeterministicTaskExecutor(
        max_workers=2,
        capacity=_capacity(),
        retry_policy=RetryPolicy(max_retries=0),
    ).execute((child, parent), worker, assignment_digest="assignment-1")
    records = {record.initial_task_id: record for record in run.records}
    parent_record = records[parent.identity.task_id]
    child_record = records[child.identity.task_id]
    assert calls == [parent.identity.task_id]
    assert parent_record.outcome.status is not ExecutionStatus.OK
    assert child_record.blocked is True
    assert child_record.worker_invoked is False
    assert child_record.authority_eligible is False
    assert child_record.outcome.status is ExecutionStatus.MISSING
    assert child_record.outcome.failure_kind is FailureKind.MISSING_RESULT
    assert "blocked by non-OK prerequisite" in child_record.outcome.reason
    assert child_record.final_attempt.blocked_dependency_record_digests == (
        parent_record.digest,
    )


def test_prerequisite_blocking_propagates_transitively_without_execution_credit():
    parent = make_task(0)
    child = make_task(1, dependencies=(parent.identity.task_id,))
    grandchild = make_task(2, dependencies=(child.identity.task_id,))
    calls = []

    def worker(invocation):
        calls.append(invocation.task.identity.task_id)
        raise TimeoutError("root prerequisite timed out")

    run = DeterministicTaskExecutor(
        max_workers=3,
        capacity=_capacity(),
        retry_policy=RetryPolicy(max_retries=0),
    ).execute(
        (grandchild, child, parent),
        worker,
        assignment_digest="assignment-1",
    )
    records = {record.initial_task_id: record for record in run.records}
    assert calls == [parent.identity.task_id]
    assert records[child.identity.task_id].blocked
    assert records[grandchild.identity.task_id].blocked
    assert records[grandchild.identity.task_id].final_attempt.blocked_dependency_record_digests == (
        records[child.identity.task_id].digest,
    )
    assert all(
        record.authority_eligible is False
        for record in (
            records[parent.identity.task_id],
            records[child.identity.task_id],
            records[grandchild.identity.task_id],
        )
    )
