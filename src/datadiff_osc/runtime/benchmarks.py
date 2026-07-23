"""Reproducible parity, clustering, and parallel-scaling measurements."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Callable, Iterable, TypeVar

from datadiff_osc._canonical import stable_digest, to_primitive
from datadiff_osc.comparison.staged import cluster_component_fingerprints
from datadiff_osc.contract_engine.fingerprints import ComponentFingerprint
from datadiff_osc.parallel.executor import ExecutionRun, RetryPolicy, Worker
from datadiff_osc.parallel.invariance import (
    AuthorityEvidenceSnapshot,
    InvarianceReport,
    validate_worker_count_design,
)
from datadiff_osc.parallel.resources import ResourceCapacity
from datadiff_osc.schemas import TaskSpec, VerdictKind


T = TypeVar("T")


def percentile(values: Iterable[float], percentile_value: float) -> float:
    materialized = sorted(float(item) for item in values)
    if not materialized:
        return 0.0
    if not 0.0 <= percentile_value <= 100.0:
        raise ValueError("percentile must be in [0, 100]")
    rank = (len(materialized) - 1) * percentile_value / 100.0
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return materialized[lower]
    weight = rank - lower
    return materialized[lower] * (1.0 - weight) + materialized[upper] * weight


@dataclass(frozen=True, slots=True)
class ParityBenchmark:
    group_count: int
    discrepancy_count: int
    discrepancy_ids: tuple[str, ...]
    exact_escalation_count: int
    staged_p95_ms: float
    exact_p95_ms: float
    elapsed_seconds: float
    throughput_groups_per_second: float
    schema_version: str = "osc-staged-exact-parity-benchmark-v1"

    @property
    def zero_discrepancy(self) -> bool:
        return self.group_count > 0 and self.discrepancy_count == 0

    @property
    def phase6_sample_floor_met(self) -> bool:
        return self.group_count >= 100_000

    @property
    def digest(self) -> str:
        return stable_digest("osc-staged-exact-parity-benchmark", self)

    def to_dict(self) -> dict[str, object]:
        return to_primitive(self)


def audit_staged_exact_parity(
    samples: Iterable[T],
    *,
    sample_id: Callable[[T], str],
    staged: Callable[[T], tuple[VerdictKind, bool]],
    exact: Callable[[T], VerdictKind],
) -> ParityBenchmark:
    staged_times: list[float] = []
    exact_times: list[float] = []
    discrepancies: list[str] = []
    exact_escalations = 0
    count = 0
    started = time.perf_counter()
    for sample in samples:
        count += 1
        before = time.perf_counter()
        staged_verdict, escalated = staged(sample)
        staged_times.append((time.perf_counter() - before) * 1000.0)
        before = time.perf_counter()
        exact_verdict = exact(sample)
        exact_times.append((time.perf_counter() - before) * 1000.0)
        exact_escalations += int(escalated)
        if staged_verdict != exact_verdict:
            discrepancies.append(sample_id(sample))
    elapsed = time.perf_counter() - started
    return ParityBenchmark(
        group_count=count,
        discrepancy_count=len(discrepancies),
        discrepancy_ids=tuple(discrepancies),
        exact_escalation_count=exact_escalations,
        staged_p95_ms=percentile(staged_times, 95.0),
        exact_p95_ms=percentile(exact_times, 95.0),
        elapsed_seconds=elapsed,
        throughput_groups_per_second=(count / elapsed if elapsed > 0.0 else 0.0),
    )


@dataclass(frozen=True, slots=True)
class ClusteringBenchmarkPoint:
    backend_count: int
    repetitions: int
    elapsed_seconds: float
    nanoseconds_per_endpoint: float
    pairwise_partition_match: bool


@dataclass(frozen=True, slots=True)
class ClusteringBenchmark:
    points: tuple[ClusteringBenchmarkPoint, ...]
    schema_version: str = "osc-component-clustering-benchmark-v1"

    @property
    def all_pairwise_partitions_match(self) -> bool:
        return bool(self.points) and all(item.pairwise_partition_match for item in self.points)

    @property
    def digest(self) -> str:
        return stable_digest("osc-component-clustering-benchmark", self)

    def to_dict(self) -> dict[str, object]:
        return to_primitive(self)


def _partition(clusters: Iterable[Iterable[str]]) -> frozenset[frozenset[str]]:
    return frozenset(frozenset(cluster) for cluster in clusters)


def _pairwise_reference(fingerprints: tuple[ComponentFingerprint, ...]) -> frozenset[frozenset[str]]:
    remaining = list(fingerprints)
    clusters: list[set[str]] = []
    while remaining:
        representative = remaining.pop(0)
        group = {representative.endpoint_id}
        unmatched: list[ComponentFingerprint] = []
        for candidate in remaining:
            if (
                candidate.observer_id == representative.observer_id
                and candidate.observer_digest == representative.observer_digest
                and candidate.contract_digest == representative.contract_digest
                and candidate.payload_digest == representative.payload_digest
            ):
                group.add(candidate.endpoint_id)
            else:
                unmatched.append(candidate)
        remaining = unmatched
        clusters.append(group)
    return _partition(clusters)


def benchmark_component_clustering(
    fingerprint_groups: Iterable[tuple[ComponentFingerprint, ...]],
    *,
    repetitions: int = 100,
) -> ClusteringBenchmark:
    if repetitions < 1:
        raise ValueError("clustering benchmark repetitions must be positive")
    points: list[ClusteringBenchmarkPoint] = []
    for group in fingerprint_groups:
        if not group:
            raise ValueError("clustering benchmark group must be non-empty")
        endpoint_order = tuple(item.endpoint_id for item in group)
        before = time.perf_counter()
        clustered = None
        for _ in range(repetitions):
            clustered = cluster_component_fingerprints(group, endpoint_order=endpoint_order)
        elapsed = time.perf_counter() - before
        assert clustered is not None
        fast_partition = _partition(item.endpoint_ids for item in clustered.clusters)
        points.append(
            ClusteringBenchmarkPoint(
                backend_count=len(group),
                repetitions=repetitions,
                elapsed_seconds=elapsed,
                nanoseconds_per_endpoint=(
                    elapsed * 1_000_000_000.0 / (len(group) * repetitions)
                ),
                pairwise_partition_match=(fast_partition == _pairwise_reference(group)),
            )
        )
    return ClusteringBenchmark(points=tuple(points))


@dataclass(frozen=True, slots=True)
class ParallelScalingPoint:
    workers: int
    elapsed_seconds: float
    tasks_per_second: float
    parallel_efficiency: float


@dataclass(frozen=True, slots=True)
class ParallelScalingBenchmark:
    points: tuple[ParallelScalingPoint, ...]
    invariance: InvarianceReport
    schema_version: str = "osc-parallel-scaling-benchmark-v1"

    @property
    def digest(self) -> str:
        return stable_digest("osc-parallel-scaling-benchmark", self)

    def to_dict(self) -> dict[str, object]:
        return to_primitive(self)


def benchmark_parallel_scaling(
    tasks: Iterable[TaskSpec],
    worker_factory: Callable[[int], Worker],
    *,
    capacity: ResourceCapacity,
    worker_counts: tuple[int, ...] = (1, 2, 4, 6),
    retry_policy: RetryPolicy | None = None,
    assignment_digest: str,
    authority_evidence_factory: Callable[
        [ExecutionRun, int], AuthorityEvidenceSnapshot
    ],
) -> ParallelScalingBenchmark:
    worker_counts = validate_worker_count_design(worker_counts)
    materialized = tuple(tasks)
    elapsed: list[float] = []
    runs = []
    for count in worker_counts:
        started = time.perf_counter()
        from datadiff_osc.parallel.executor import DeterministicTaskExecutor

        runs.append(
            DeterministicTaskExecutor(
                max_workers=count,
                capacity=capacity,
                retry_policy=retry_policy,
            ).execute(
                materialized,
                worker_factory(count),
                assignment_digest=assignment_digest,
            )
        )
        elapsed.append(time.perf_counter() - started)
    from datadiff_osc.parallel.invariance import audit_execution_runs

    snapshots = tuple(
        authority_evidence_factory(run, count)
        for run, count in zip(runs, worker_counts, strict=True)
    )
    invariance = audit_execution_runs(
        tuple(runs),
        worker_counts=worker_counts,
        authority_evidence=snapshots,
    )
    baseline = elapsed[worker_counts.index(1)]
    points = tuple(
        ParallelScalingPoint(
            workers=count,
            elapsed_seconds=seconds,
            tasks_per_second=(len(materialized) / seconds if seconds > 0.0 else 0.0),
            parallel_efficiency=(
                baseline / (seconds * count) if seconds > 0.0 and count > 0 else 0.0
            ),
        )
        for count, seconds in zip(worker_counts, elapsed, strict=True)
    )
    return ParallelScalingBenchmark(points=points, invariance=invariance)
