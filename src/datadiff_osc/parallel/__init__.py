"""Deterministic resource-aware task execution for OSC."""

from datadiff_osc.parallel.dag import (
    FrozenEpoch,
    TaskGraph,
    TaskGraphError,
    build_task_graph,
    freeze_epoch,
    logical_task_digest,
    retry_task,
)
from datadiff_osc.parallel.executor import (
    AdapterTaskError,
    DeterministicTaskExecutor,
    ExecutionRun,
    RetryPolicy,
    SemanticDomainTaskError,
    TaskInvocation,
    UnsupportedTaskError,
    WorkerCrashError,
    WorkerResult,
)
from datadiff_osc.parallel.invariance import (
    InvarianceReport,
    audit_execution_runs,
    run_worker_count_invariance,
)
from datadiff_osc.parallel.resources import (
    ResourceCapacity,
    ResourcePool,
    ResourceRequestTooLarge,
)

__all__ = [
    "AdapterTaskError",
    "DeterministicTaskExecutor",
    "ExecutionRun",
    "FrozenEpoch",
    "InvarianceReport",
    "ResourceCapacity",
    "ResourcePool",
    "ResourceRequestTooLarge",
    "RetryPolicy",
    "SemanticDomainTaskError",
    "TaskGraph",
    "TaskGraphError",
    "TaskInvocation",
    "UnsupportedTaskError",
    "WorkerCrashError",
    "WorkerResult",
    "audit_execution_runs",
    "build_task_graph",
    "freeze_epoch",
    "logical_task_digest",
    "retry_task",
    "run_worker_count_invariance",
]
