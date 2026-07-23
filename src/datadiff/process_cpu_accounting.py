from __future__ import annotations

import math
import resource
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping


UsageFn = Callable[[], Mapping[str, float]]
ClockFn = Callable[[], float]


def _resource_usage(kind: int) -> dict[str, float]:
    usage = resource.getrusage(kind)
    return {
        "user_s": max(0.0, float(usage.ru_utime)),
        "system_s": max(0.0, float(usage.ru_stime)),
    }


def self_process_usage() -> dict[str, float]:
    return _resource_usage(resource.RUSAGE_SELF)


def reaped_children_usage() -> dict[str, float]:
    return _resource_usage(resource.RUSAGE_CHILDREN)


def _usage_value(payload: Mapping[str, Any], key: str) -> float:
    value = float(payload.get(key, 0.0) or 0.0)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"process CPU usage {key} must be finite and non-negative")
    return value


@dataclass(frozen=True, slots=True)
class ProcessCPUSnapshot:
    wall_s: float
    self_user_s: float
    self_system_s: float
    children_user_s: float
    children_system_s: float


@dataclass(frozen=True, slots=True)
class WorkerCPUObservation:
    measurement_id: str
    worker_id: str
    process_cpu_ms: float
    coverage: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "process-worker-cpu-observation-v1",
            "measurement_id": self.measurement_id,
            "worker_id": self.worker_id,
            "process_cpu_ms": self.process_cpu_ms,
            "coverage": self.coverage,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class ProcessCPUScopeRecord:
    scope_id: str
    status: str
    wall_ms: float
    self_user_cpu_ms: float
    self_system_cpu_ms: float
    reaped_children_user_cpu_ms: float
    reaped_children_system_cpu_ms: float
    explicit_worker_cpu_ms: float
    total_process_cpu_ms: float
    backend_reported_ms: float
    backend_calls: int
    worker_observations: tuple[WorkerCPUObservation, ...]
    metadata: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "process-cpu-scope-record-v1",
            "scope_id": self.scope_id,
            "status": self.status,
            "wall_ms": self.wall_ms,
            "self_user_cpu_ms": self.self_user_cpu_ms,
            "self_system_cpu_ms": self.self_system_cpu_ms,
            "reaped_children_user_cpu_ms": self.reaped_children_user_cpu_ms,
            "reaped_children_system_cpu_ms": self.reaped_children_system_cpu_ms,
            "explicit_worker_cpu_ms": self.explicit_worker_cpu_ms,
            "total_process_cpu_ms": self.total_process_cpu_ms,
            "backend_reported_ms": self.backend_reported_ms,
            "backend_calls": self.backend_calls,
            "worker_observations": [
                observation.to_dict() for observation in self.worker_observations
            ],
            "metadata": dict(self.metadata),
            "reconciliation": {
                "component_sum_ms": (
                    self.self_user_cpu_ms
                    + self.self_system_cpu_ms
                    + self.reaped_children_user_cpu_ms
                    + self.reaped_children_system_cpu_ms
                    + self.explicit_worker_cpu_ms
                ),
                "all_pass": math.isclose(
                    self.total_process_cpu_ms,
                    self.self_user_cpu_ms
                    + self.self_system_cpu_ms
                    + self.reaped_children_user_cpu_ms
                    + self.reaped_children_system_cpu_ms
                    + self.explicit_worker_cpu_ms,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                ),
            },
        }


@dataclass(slots=True)
class _ActiveScope:
    scope_id: str
    start: ProcessCPUSnapshot
    metadata: dict[str, Any]
    worker_observations: list[WorkerCPUObservation] = field(default_factory=list)


class ProcessCPUAccountingLedger:
    """Sequential additive process-tree CPU scopes with no overlapping charge."""

    WORKER_COVERAGE = {
        "external_process",
        "persistent_worker_excluded_from_rusage",
    }

    def __init__(
        self,
        *,
        wall_fn: ClockFn = time.perf_counter,
        self_usage_fn: UsageFn = self_process_usage,
        children_usage_fn: UsageFn = reaped_children_usage,
        include_reaped_children: bool = True,
    ) -> None:
        self.wall_fn = wall_fn
        self.self_usage_fn = self_usage_fn
        self.children_usage_fn = children_usage_fn
        self.include_reaped_children = bool(include_reaped_children)
        self.active: _ActiveScope | None = None
        self.records: list[ProcessCPUScopeRecord] = []
        self._scope_ids: set[str] = set()
        self._measurement_ids: set[str] = set()

    def _snapshot(self) -> ProcessCPUSnapshot:
        self_usage = self.self_usage_fn()
        child_usage = self.children_usage_fn()
        wall_s = float(self.wall_fn())
        if not math.isfinite(wall_s):
            raise ValueError("process CPU wall clock must be finite")
        return ProcessCPUSnapshot(
            wall_s=wall_s,
            self_user_s=_usage_value(self_usage, "user_s"),
            self_system_s=_usage_value(self_usage, "system_s"),
            children_user_s=_usage_value(child_usage, "user_s"),
            children_system_s=_usage_value(child_usage, "system_s"),
        )

    def start_scope(
        self,
        scope_id: str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        normalized = str(scope_id or "").strip()
        if not normalized:
            raise ValueError("process CPU scope_id cannot be empty")
        if self.active is not None:
            raise ValueError("process CPU scopes cannot overlap")
        if normalized in self._scope_ids:
            raise ValueError(f"duplicate process CPU scope_id: {normalized}")
        self.active = _ActiveScope(
            scope_id=normalized,
            start=self._snapshot(),
            metadata=dict(metadata or {}),
        )
        self._scope_ids.add(normalized)

    def record_worker_cpu(
        self,
        *,
        measurement_id: str,
        worker_id: str,
        process_cpu_ms: float,
        coverage: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if self.active is None:
            raise ValueError("worker CPU observation requires an active scope")
        normalized_measurement = str(measurement_id or "").strip()
        normalized_worker = str(worker_id or "").strip()
        normalized_coverage = str(coverage or "").strip()
        if not normalized_measurement or not normalized_worker:
            raise ValueError("worker CPU measurement_id and worker_id cannot be empty")
        if normalized_measurement in self._measurement_ids:
            raise ValueError(
                f"duplicate worker CPU measurement_id: {normalized_measurement}"
            )
        if normalized_coverage not in self.WORKER_COVERAGE:
            raise ValueError(
                "explicit worker CPU must be outside the selected rusage child coverage"
            )
        cpu_ms = float(process_cpu_ms)
        if not math.isfinite(cpu_ms) or cpu_ms < 0:
            raise ValueError("worker process_cpu_ms must be finite and non-negative")
        observation = WorkerCPUObservation(
            measurement_id=normalized_measurement,
            worker_id=normalized_worker,
            process_cpu_ms=cpu_ms,
            coverage=normalized_coverage,
            metadata=dict(metadata or {}),
        )
        self.active.worker_observations.append(observation)
        self._measurement_ids.add(normalized_measurement)

    @staticmethod
    def _delta_ms(end: float, start: float, *, field: str) -> float:
        delta = (float(end) - float(start)) * 1000.0
        if delta < -1e-9:
            raise ValueError(f"process CPU counter moved backwards: {field}")
        return max(0.0, delta)

    def finish_scope(
        self,
        *,
        status: str = "ok",
        backend_reported_ms: float = 0.0,
        backend_calls: int = 0,
        metadata: Mapping[str, Any] | None = None,
    ) -> ProcessCPUScopeRecord:
        if self.active is None:
            raise ValueError("no process CPU scope is active")
        active = self.active
        end = self._snapshot()
        normalized_status = str(status or "").strip()
        if not normalized_status:
            raise ValueError("process CPU scope status cannot be empty")
        wall_ms = self._delta_ms(end.wall_s, active.start.wall_s, field="wall")
        self_user_ms = self._delta_ms(
            end.self_user_s,
            active.start.self_user_s,
            field="self_user",
        )
        self_system_ms = self._delta_ms(
            end.self_system_s,
            active.start.self_system_s,
            field="self_system",
        )
        child_user_ms = (
            self._delta_ms(
                end.children_user_s,
                active.start.children_user_s,
                field="children_user",
            )
            if self.include_reaped_children
            else 0.0
        )
        child_system_ms = (
            self._delta_ms(
                end.children_system_s,
                active.start.children_system_s,
                field="children_system",
            )
            if self.include_reaped_children
            else 0.0
        )
        explicit_worker_ms = sum(
            observation.process_cpu_ms
            for observation in active.worker_observations
        )
        reported_ms = float(backend_reported_ms)
        calls = int(backend_calls)
        if not math.isfinite(reported_ms) or reported_ms < 0:
            raise ValueError("backend_reported_ms must be finite and non-negative")
        if calls < 0:
            raise ValueError("backend_calls must be non-negative")
        total_ms = (
            self_user_ms
            + self_system_ms
            + child_user_ms
            + child_system_ms
            + explicit_worker_ms
        )
        record = ProcessCPUScopeRecord(
            scope_id=active.scope_id,
            status=normalized_status,
            wall_ms=wall_ms,
            self_user_cpu_ms=self_user_ms,
            self_system_cpu_ms=self_system_ms,
            reaped_children_user_cpu_ms=child_user_ms,
            reaped_children_system_cpu_ms=child_system_ms,
            explicit_worker_cpu_ms=explicit_worker_ms,
            total_process_cpu_ms=total_ms,
            backend_reported_ms=reported_ms,
            backend_calls=calls,
            worker_observations=tuple(active.worker_observations),
            metadata={**active.metadata, **dict(metadata or {})},
        )
        self.records.append(record)
        self.active = None
        return record

    def summary(self) -> dict[str, Any]:
        status_counts = Counter(record.status for record in self.records)
        totals = {
            "wall_ms": sum(record.wall_ms for record in self.records),
            "self_user_cpu_ms": sum(
                record.self_user_cpu_ms for record in self.records
            ),
            "self_system_cpu_ms": sum(
                record.self_system_cpu_ms for record in self.records
            ),
            "reaped_children_user_cpu_ms": sum(
                record.reaped_children_user_cpu_ms for record in self.records
            ),
            "reaped_children_system_cpu_ms": sum(
                record.reaped_children_system_cpu_ms for record in self.records
            ),
            "explicit_worker_cpu_ms": sum(
                record.explicit_worker_cpu_ms for record in self.records
            ),
            "total_process_cpu_ms": sum(
                record.total_process_cpu_ms for record in self.records
            ),
            "backend_reported_ms": sum(
                record.backend_reported_ms for record in self.records
            ),
            "backend_calls": sum(record.backend_calls for record in self.records),
        }
        component_sum = (
            totals["self_user_cpu_ms"]
            + totals["self_system_cpu_ms"]
            + totals["reaped_children_user_cpu_ms"]
            + totals["reaped_children_system_cpu_ms"]
            + totals["explicit_worker_cpu_ms"]
        )
        return {
            "schema_version": "process-cpu-accounting-ledger-v1",
            "include_reaped_children": self.include_reaped_children,
            "active_scope_id": self.active.scope_id if self.active else "",
            "scope_count": len(self.records),
            "worker_measurement_count": len(self._measurement_ids),
            "status_counts": dict(sorted(status_counts.items())),
            "totals": totals,
            "reconciliation": {
                "component_sum_ms": component_sum,
                "all_pass": math.isclose(
                    totals["total_process_cpu_ms"],
                    component_sum,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                ),
                "wall_backend_process_cpu_separated": True,
                "overlapping_scopes_allowed": False,
            },
            "records": [record.to_dict() for record in self.records],
        }
