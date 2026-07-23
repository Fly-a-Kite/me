"""Immutable task graph validation and frozen epoch identities."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

from datadiff_osc._canonical import stable_digest, to_primitive
from datadiff_osc.schemas import TaskIdentity, TaskSpec


class TaskGraphError(ValueError):
    """A task set is incomplete, ambiguous, cyclic, or crosses epoch scope."""


def logical_task_digest(task: TaskSpec) -> str:
    """Identity of a logical task, excluding only the retry attempt."""

    identity = task.identity
    return stable_digest(
        "osc-logical-task",
        {
            "protocol_digest": identity.protocol_digest,
            "task_kind": identity.task_kind,
            "epoch_index": identity.epoch_index,
            "decision_index": identity.decision_index,
            "seed_lineage_digest": identity.seed_lineage_digest,
            "contrast_set_id": identity.contrast_set_id,
            "endpoint_id": identity.endpoint_id,
            "backend": identity.backend,
            "schema_version": identity.schema_version,
            "dependency_task_ids": task.dependency_task_ids,
            "resources": task.resources,
            "payload_digest": task.payload_digest,
            "task_schema_version": task.schema_version,
        },
    )


def retry_task(task: TaskSpec) -> TaskSpec:
    """Create the next concrete attempt without consuming a seed or case."""

    identity = replace(task.identity, attempt=task.identity.attempt + 1)
    retried = replace(task, identity=identity)
    if (
        retried.identity.seed_lineage_digest != task.identity.seed_lineage_digest
        or retried.payload_digest != task.payload_digest
        or retried.dependency_task_ids != task.dependency_task_ids
        or retried.resources != task.resources
        or logical_task_digest(retried) != logical_task_digest(task)
    ):
        raise RuntimeError("retry changed frozen logical task identity")
    return retried


@dataclass(frozen=True, slots=True)
class TaskGraph:
    tasks: tuple[TaskSpec, ...]
    layers: tuple[tuple[str, ...], ...]
    schema_version: str = "osc-runtime-task-graph-v1"

    @property
    def task_ids(self) -> tuple[str, ...]:
        return tuple(task.identity.task_id for task in self.tasks)

    @property
    def logical_task_digests(self) -> tuple[str, ...]:
        return tuple(logical_task_digest(task) for task in self.tasks)

    @property
    def task_set_digest(self) -> str:
        return stable_digest(
            "osc-logical-task-set", tuple(sorted(self.logical_task_digests))
        )

    @property
    def digest(self) -> str:
        return stable_digest("osc-runtime-task-graph", self)

    def task_map(self) -> dict[str, TaskSpec]:
        return {task.identity.task_id: task for task in self.tasks}

    def to_dict(self) -> dict[str, object]:
        return to_primitive(self)


def build_task_graph(tasks: Iterable[TaskSpec]) -> TaskGraph:
    materialized = tuple(tasks)
    if not materialized:
        raise TaskGraphError("task graph must be non-empty")
    by_id: dict[str, TaskSpec] = {}
    for task in materialized:
        task_id = task.identity.task_id
        if task_id in by_id:
            raise TaskGraphError(f"duplicate task identity: {task_id}")
        by_id[task_id] = task
    known = set(by_id)
    for task_id, task in by_id.items():
        missing = set(task.dependency_task_ids) - known
        if missing:
            raise TaskGraphError(
                f"task {task_id} has unknown dependencies: {sorted(missing)}"
            )

    indegree = {
        task_id: len(task.dependency_task_ids) for task_id, task in by_id.items()
    }
    dependents: dict[str, list[str]] = {task_id: [] for task_id in by_id}
    for task_id, task in by_id.items():
        for dependency in task.dependency_task_ids:
            dependents[dependency].append(task_id)

    layers: list[tuple[str, ...]] = []
    ready = tuple(sorted(task_id for task_id, degree in indegree.items() if degree == 0))
    visited = 0
    while ready:
        layers.append(ready)
        visited += len(ready)
        next_ready: list[str] = []
        for completed in ready:
            for dependent in dependents[completed]:
                indegree[dependent] -= 1
                if indegree[dependent] == 0:
                    next_ready.append(dependent)
        ready = tuple(sorted(next_ready))
    if visited != len(by_id):
        cyclic = sorted(task_id for task_id, degree in indegree.items() if degree > 0)
        raise TaskGraphError(f"task graph contains a cycle: {cyclic}")

    ordered = tuple(by_id[task_id] for task_id in sorted(by_id))
    return TaskGraph(tasks=ordered, layers=tuple(layers))


@dataclass(frozen=True, slots=True)
class FrozenEpoch:
    protocol_digest: str
    epoch_index: int
    assignment_digest: str
    graph: TaskGraph
    schema_version: str = "osc-frozen-runtime-epoch-v1"

    @property
    def digest(self) -> str:
        return stable_digest("osc-frozen-runtime-epoch", self)

    def to_dict(self) -> dict[str, object]:
        return to_primitive(self)


def freeze_epoch(tasks: Iterable[TaskSpec], *, assignment_digest: str) -> FrozenEpoch:
    if not assignment_digest:
        raise TaskGraphError("frozen epoch requires an assignment digest")
    graph = build_task_graph(tasks)
    protocols = {task.identity.protocol_digest for task in graph.tasks}
    epochs = {task.identity.epoch_index for task in graph.tasks}
    if len(protocols) != 1 or len(epochs) != 1:
        raise TaskGraphError("frozen epoch cannot mix protocol or epoch identities")
    return FrozenEpoch(
        protocol_digest=next(iter(protocols)),
        epoch_index=next(iter(epochs)),
        assignment_digest=assignment_digest,
        graph=graph,
    )
