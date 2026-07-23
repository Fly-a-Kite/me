"""Resource-aware deterministic execution with structured bounded retries."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Callable, Iterable

from datadiff_osc._canonical import stable_digest, to_primitive
from datadiff_osc.parallel.dag import (
    FrozenEpoch,
    freeze_epoch,
    logical_task_digest,
    retry_task,
)
from datadiff_osc.parallel.resources import ResourceCapacity, ResourcePool
from datadiff_osc.schemas import (
    ExecutionStatus,
    FailureKind,
    StructuredExecutionOutcome,
    TaskSpec,
)


class UnsupportedTaskError(RuntimeError):
    def __init__(self, message: str, *, evidence_digest: str) -> None:
        super().__init__(message)
        if not evidence_digest:
            raise ValueError("unsupported task error requires evidence")
        self.evidence_digest = evidence_digest


class SemanticDomainTaskError(RuntimeError):
    pass


class AdapterTaskError(RuntimeError):
    pass


class WorkerCrashError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class WorkerResult:
    outcome: StructuredExecutionOutcome
    payload_digest: str

    def __post_init__(self) -> None:
        if not self.payload_digest:
            raise ValueError("worker result requires a payload digest")


@dataclass(frozen=True, slots=True)
class TaskAttemptRecord:
    concrete_task_id: str
    attempt: int
    outcome: StructuredExecutionOutcome
    payload_digest: str
    worker_invoked: bool = True
    blocked_dependency_record_digests: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.concrete_task_id or not self.payload_digest:
            raise ValueError("task attempt record bindings must be non-empty")
        if self.attempt < 0:
            raise ValueError("task attempt must be non-negative")
        if len(self.blocked_dependency_record_digests) != len(
            set(self.blocked_dependency_record_digests)
        ):
            raise ValueError("blocked dependency record digests must be unique")
        if tuple(sorted(self.blocked_dependency_record_digests)) != (
            self.blocked_dependency_record_digests
        ):
            raise ValueError("blocked dependency record digests must be sorted")
        if self.worker_invoked and self.blocked_dependency_record_digests:
            raise ValueError("an invoked worker cannot carry blocked dependencies")
        if not self.worker_invoked:
            if not self.blocked_dependency_record_digests:
                raise ValueError("a blocked attempt requires dependency evidence")
            if (
                self.outcome.status is not ExecutionStatus.MISSING
                or self.outcome.failure_kind is not FailureKind.MISSING_RESULT
            ):
                raise ValueError("a blocked attempt must be a structured missing result")

    @property
    def digest(self) -> str:
        return stable_digest("osc-task-attempt-record", self)


@dataclass(frozen=True, slots=True)
class TaskExecutionRecord:
    initial_task_id: str
    logical_task_digest: str
    seed_lineage_digest: str
    result_order_key: tuple[str, ...]
    attempts: tuple[TaskAttemptRecord, ...]

    @property
    def final_attempt(self) -> TaskAttemptRecord:
        return self.attempts[-1]

    @property
    def outcome(self) -> StructuredExecutionOutcome:
        return self.final_attempt.outcome

    @property
    def payload_digest(self) -> str:
        return self.final_attempt.payload_digest

    @property
    def worker_invoked(self) -> bool:
        return any(attempt.worker_invoked for attempt in self.attempts)

    @property
    def blocked(self) -> bool:
        return not self.worker_invoked

    @property
    def authority_eligible(self) -> bool:
        return self.worker_invoked and self.outcome.status is ExecutionStatus.OK

    @property
    def digest(self) -> str:
        return stable_digest("osc-task-execution-record", self)


@dataclass(frozen=True, slots=True)
class TaskInvocation:
    task: TaskSpec
    dependency_records: tuple[TaskExecutionRecord, ...]


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_retries: int = 1
    retryable_statuses: frozenset[ExecutionStatus] = frozenset(
        {
            ExecutionStatus.TIMEOUT,
            ExecutionStatus.CRASH,
            ExecutionStatus.ADAPTER_ERROR,
            ExecutionStatus.MISSING,
        }
    )

    def __post_init__(self) -> None:
        if self.max_retries < 0:
            raise ValueError("retry count must be non-negative")
        if ExecutionStatus.OK in self.retryable_statuses:
            raise ValueError("successful tasks cannot be retryable")


@dataclass(frozen=True, slots=True)
class ExecutionRun:
    epoch_digest: str
    assignment_digest: str
    worker_count: int
    records: tuple[TaskExecutionRecord, ...]
    task_set_digest: str
    schema_version: str = "osc-deterministic-execution-run-v2"

    def __post_init__(self) -> None:
        if (
            not isinstance(self.worker_count, int)
            or isinstance(self.worker_count, bool)
            or self.worker_count < 1
        ):
            raise ValueError("execution run worker count must be a positive integer")
        if self.schema_version != "osc-deterministic-execution-run-v2":
            raise ValueError("execution run schema version mismatch")

    @property
    def initial_task_ids(self) -> tuple[str, ...]:
        return tuple(record.initial_task_id for record in self.records)

    @property
    def seed_lineage_digests(self) -> tuple[str, ...]:
        return tuple(record.seed_lineage_digest for record in self.records)

    @property
    def seed_multiset_digest(self) -> str:
        return stable_digest(
            "osc-execution-run-seed-multiset",
            tuple(sorted(self.seed_lineage_digests)),
        )

    @property
    def result_order(self) -> tuple[tuple[str, ...], ...]:
        return tuple(record.result_order_key for record in self.records)

    @property
    def task_multiset_digest(self) -> str:
        return stable_digest(
            "osc-execution-run-task-multiset",
            tuple(sorted(record.logical_task_digest for record in self.records)),
        )

    @property
    def outcome_digest(self) -> str:
        return stable_digest(
            "osc-execution-run-outcomes",
            tuple(
                (
                    record.logical_task_digest,
                    record.outcome,
                    record.payload_digest,
                )
                for record in self.records
            ),
        )

    @property
    def retry_trace_digest(self) -> str:
        return stable_digest(
            "osc-execution-run-retry-trace",
            tuple(
                (
                    record.logical_task_digest,
                    tuple(
                        (attempt.attempt, attempt.outcome.status, attempt.payload_digest)
                        for attempt in record.attempts
                    ),
                )
                for record in self.records
            ),
        )

    @property
    def digest(self) -> str:
        return stable_digest("osc-deterministic-execution-run", self)

    def to_dict(self) -> dict[str, object]:
        return to_primitive(self)


Worker = Callable[[TaskInvocation], WorkerResult | None]


def _endpoint_id(task: TaskSpec) -> str:
    return task.identity.endpoint_id or task.identity.task_id


def _failure_result(task: TaskSpec, exc: Exception) -> WorkerResult:
    endpoint_id = _endpoint_id(task)
    reason = f"{type(exc).__name__}: {exc}"
    unsupported_digest = ""
    if isinstance(exc, UnsupportedTaskError):
        status = ExecutionStatus.UNSUPPORTED
        kind = FailureKind.UNSUPPORTED_CAPABILITY
        unsupported_digest = exc.evidence_digest
    elif isinstance(exc, SemanticDomainTaskError):
        status = ExecutionStatus.SEMANTIC_ERROR
        kind = FailureKind.SEMANTIC_DOMAIN_ERROR
    elif isinstance(exc, TimeoutError):
        status = ExecutionStatus.TIMEOUT
        kind = FailureKind.TIMEOUT
    elif isinstance(exc, WorkerCrashError):
        status = ExecutionStatus.CRASH
        kind = FailureKind.CRASH
    else:
        status = ExecutionStatus.ADAPTER_ERROR
        kind = FailureKind.ADAPTER_ERROR
    outcome = StructuredExecutionOutcome(
        endpoint_id=endpoint_id,
        status=status,
        failure_kind=kind,
        reason=reason,
        unsupported_evidence_digest=unsupported_digest,
    )
    return WorkerResult(
        outcome=outcome,
        payload_digest=stable_digest("osc-worker-failure-payload", outcome),
    )


def _missing_result(task: TaskSpec) -> WorkerResult:
    outcome = StructuredExecutionOutcome(
        endpoint_id=_endpoint_id(task),
        status=ExecutionStatus.MISSING,
        failure_kind=FailureKind.MISSING_RESULT,
        reason="worker returned no result",
    )
    return WorkerResult(
        outcome=outcome,
        payload_digest=stable_digest("osc-worker-missing-payload", outcome),
    )


class DeterministicTaskExecutor:
    """Execute frozen DAG layers and merge only in stable task-ID order."""

    def __init__(
        self,
        *,
        max_workers: int,
        capacity: ResourceCapacity,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        if (
            not isinstance(max_workers, int)
            or isinstance(max_workers, bool)
            or max_workers < 1
        ):
            raise ValueError("executor requires a positive integer worker count")
        self.max_workers = max_workers
        self.capacity = capacity
        self.retry_policy = retry_policy or RetryPolicy()

    def execute(
        self,
        tasks: Iterable[TaskSpec],
        worker: Worker,
        *,
        assignment_digest: str,
    ) -> ExecutionRun:
        epoch = freeze_epoch(tasks, assignment_digest=assignment_digest)
        pool = ResourcePool(self.capacity)
        for task in epoch.graph.tasks:
            pool.validate_request(task.resources)
        by_id = epoch.graph.task_map()
        completed: dict[str, TaskExecutionRecord] = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for layer in epoch.graph.layers:
                layer_tasks = tuple(by_id[task_id] for task_id in layer)
                completed.update(
                    self._execute_layer(layer_tasks, completed, worker, executor, pool)
                )
        if not pool.is_idle:
            raise RuntimeError("resource pool leaked reservations after execution")
        ordered = tuple(completed[task_id] for task_id in epoch.graph.task_ids)
        return ExecutionRun(
            epoch_digest=epoch.digest,
            assignment_digest=assignment_digest,
            worker_count=self.max_workers,
            records=ordered,
            task_set_digest=epoch.graph.task_set_digest,
        )

    def _execute_layer(
        self,
        tasks: tuple[TaskSpec, ...],
        completed: dict[str, TaskExecutionRecord],
        worker: Worker,
        executor: ThreadPoolExecutor,
        pool: ResourcePool,
    ) -> dict[str, TaskExecutionRecord]:
        pending = list(sorted(tasks, key=lambda item: item.identity.task_id))
        active: dict[Future[TaskExecutionRecord], TaskSpec] = {}
        layer_results: dict[str, TaskExecutionRecord] = {}
        while pending or active:
            scheduled = True
            while pending and len(active) < self.max_workers and scheduled:
                scheduled = False
                for index, task in enumerate(pending):
                    dependencies = tuple(
                        completed[task_id] for task_id in task.dependency_task_ids
                    )
                    blockers = tuple(
                        dependency
                        for dependency in dependencies
                        if dependency.outcome.status is not ExecutionStatus.OK
                    )
                    if blockers:
                        layer_results[task.identity.task_id] = (
                            self._blocked_by_prerequisites(task, blockers)
                        )
                        pending.pop(index)
                        scheduled = True
                        break
                    if not pool.try_acquire(task.resources):
                        continue
                    future = executor.submit(
                        self._execute_with_retries, task, dependencies, worker
                    )
                    active[future] = task
                    pending.pop(index)
                    scheduled = True
                    break
            if not active:
                if pending:
                    raise RuntimeError("resource scheduler made no progress")
                continue
            done, _ = wait(tuple(active), return_when=FIRST_COMPLETED)
            for future in sorted(done, key=lambda item: active[item].identity.task_id):
                task = active.pop(future)
                try:
                    layer_results[task.identity.task_id] = future.result()
                finally:
                    pool.release(task.resources)
        return layer_results

    @staticmethod
    def _blocked_by_prerequisites(
        task: TaskSpec,
        blockers: tuple[TaskExecutionRecord, ...],
    ) -> TaskExecutionRecord:
        blocking_summary = tuple(
            sorted(
                (
                    dependency.initial_task_id,
                    dependency.outcome.status.value,
                    dependency.digest,
                )
                for dependency in blockers
            )
        )
        blocking_digests = tuple(sorted(item[2] for item in blocking_summary))
        binding_digest = stable_digest(
            "osc-blocked-prerequisite-binding", blocking_summary
        )
        outcome = StructuredExecutionOutcome(
            endpoint_id=_endpoint_id(task),
            status=ExecutionStatus.MISSING,
            failure_kind=FailureKind.MISSING_RESULT,
            reason=(
                "authority execution blocked by non-OK prerequisite; "
                f"dependency_binding={binding_digest}"
            ),
        )
        attempt = TaskAttemptRecord(
            concrete_task_id=task.identity.task_id,
            attempt=task.identity.attempt,
            outcome=outcome,
            payload_digest=stable_digest(
                "osc-blocked-prerequisite-payload",
                {
                    "task_digest": logical_task_digest(task),
                    "blocking_summary": blocking_summary,
                    "outcome": outcome,
                },
            ),
            worker_invoked=False,
            blocked_dependency_record_digests=blocking_digests,
        )
        return TaskExecutionRecord(
            initial_task_id=task.identity.task_id,
            logical_task_digest=logical_task_digest(task),
            seed_lineage_digest=task.identity.seed_lineage_digest,
            result_order_key=(
                f"{task.identity.epoch_index:012d}",
                f"{task.identity.decision_index:012d}",
                task.identity.task_id,
            ),
            attempts=(attempt,),
        )

    def _execute_with_retries(
        self,
        initial: TaskSpec,
        dependencies: tuple[TaskExecutionRecord, ...],
        worker: Worker,
    ) -> TaskExecutionRecord:
        task = initial
        attempts: list[TaskAttemptRecord] = []
        while True:
            try:
                response = worker(TaskInvocation(task=task, dependency_records=dependencies))
                result = _missing_result(task) if response is None else response
                if not isinstance(result, WorkerResult):
                    raise AdapterTaskError("worker returned an invalid result type")
                if result.outcome.endpoint_id != _endpoint_id(task):
                    raise AdapterTaskError("worker result endpoint does not match task")
            except Exception as exc:  # structured worker boundary
                result = _failure_result(task, exc)
            attempts.append(
                TaskAttemptRecord(
                    concrete_task_id=task.identity.task_id,
                    attempt=task.identity.attempt,
                    outcome=result.outcome,
                    payload_digest=result.payload_digest,
                )
            )
            retries_used = len(attempts) - 1
            if (
                result.outcome.status not in self.retry_policy.retryable_statuses
                or retries_used >= self.retry_policy.max_retries
            ):
                break
            task = retry_task(task)
        return TaskExecutionRecord(
            initial_task_id=initial.identity.task_id,
            logical_task_digest=logical_task_digest(initial),
            seed_lineage_digest=initial.identity.seed_lineage_digest,
            result_order_key=(
                f"{initial.identity.epoch_index:012d}",
                f"{initial.identity.decision_index:012d}",
                initial.identity.task_id,
            ),
            attempts=tuple(attempts),
        )
