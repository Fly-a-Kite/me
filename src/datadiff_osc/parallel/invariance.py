"""Authority-complete workers=1/N invariance audits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

from datadiff_osc._canonical import stable_digest, to_primitive
from datadiff_osc.parallel.executor import (
    DeterministicTaskExecutor,
    ExecutionRun,
    RetryPolicy,
    Worker,
)
from datadiff_osc.parallel.resources import ResourceCapacity
from datadiff_osc.schemas import TaskSpec


@dataclass(frozen=True, slots=True)
class AuthorityEvidenceSnapshot:
    """Authority outputs bound to one concrete execution run.

    Runtime scheduling alone cannot establish invariance.  The caller must
    bind the resulting verdict, coverage bitmap, ledger, and certificates to
    every run so a worker-count audit cannot silently omit an authority plane.
    """

    execution_run_digest: str
    worker_count: int
    assignment_digest: str
    epoch_digest: str
    task_multiset_digest: str
    seed_multiset_digest: str
    authority_verdict_digest: str
    coverage_bitmap_digest: str
    ledger_digest: str
    certificate_digests: tuple[str, ...]
    schema_version: str = "osc-authority-evidence-snapshot-v2"

    def __post_init__(self) -> None:
        if (
            not isinstance(self.worker_count, int)
            or isinstance(self.worker_count, bool)
            or self.worker_count < 1
        ):
            raise ValueError("authority evidence worker count must be positive")
        if self.schema_version != "osc-authority-evidence-snapshot-v2":
            raise ValueError("authority evidence snapshot schema version mismatch")
        for value in (
            self.execution_run_digest,
            self.assignment_digest,
            self.epoch_digest,
            self.task_multiset_digest,
            self.seed_multiset_digest,
            self.authority_verdict_digest,
            self.coverage_bitmap_digest,
            self.ledger_digest,
        ):
            if not isinstance(value, str) or not value:
                raise ValueError("authority evidence bindings must be non-empty")
        if not self.certificate_digests:
            raise ValueError("authority evidence requires certificate digests")
        if any(not isinstance(item, str) or not item for item in self.certificate_digests):
            raise ValueError("authority certificate digests must be non-empty")
        if len(self.certificate_digests) != len(set(self.certificate_digests)):
            raise ValueError("authority certificate digests must be unique")
        if tuple(sorted(self.certificate_digests)) != self.certificate_digests:
            raise ValueError("authority certificate digests must be sorted")

    @classmethod
    def from_run(
        cls,
        run: ExecutionRun,
        *,
        authority_verdict_digest: str,
        coverage_bitmap_digest: str,
        ledger_digest: str,
        certificate_digests: Iterable[str],
    ) -> AuthorityEvidenceSnapshot:
        return cls(
            execution_run_digest=run.digest,
            worker_count=run.worker_count,
            assignment_digest=run.assignment_digest,
            epoch_digest=run.epoch_digest,
            task_multiset_digest=run.task_multiset_digest,
            seed_multiset_digest=run.seed_multiset_digest,
            authority_verdict_digest=authority_verdict_digest,
            coverage_bitmap_digest=coverage_bitmap_digest,
            ledger_digest=ledger_digest,
            certificate_digests=tuple(sorted(certificate_digests)),
        )

    @property
    def digest(self) -> str:
        return stable_digest("osc-authority-evidence-snapshot", self)

    def binding_errors(self, run: ExecutionRun) -> tuple[str, ...]:
        checks = (
            ("execution_run_digest", self.execution_run_digest, run.digest),
            ("worker_count", self.worker_count, run.worker_count),
            ("assignment_digest", self.assignment_digest, run.assignment_digest),
            ("epoch_digest", self.epoch_digest, run.epoch_digest),
            ("task_multiset_digest", self.task_multiset_digest, run.task_multiset_digest),
            ("seed_multiset_digest", self.seed_multiset_digest, run.seed_multiset_digest),
        )
        return tuple(
            f"{name}_mismatch" for name, actual, expected in checks if actual != expected
        )


@dataclass(frozen=True, slots=True)
class InvarianceReport:
    worker_counts: tuple[int, ...]
    assignment_invariant: bool
    epoch_invariant: bool
    task_multiset_invariant: bool
    task_set_invariant: bool
    seed_lineage_invariant: bool
    result_order_invariant: bool
    outcome_invariant: bool
    retry_trace_invariant: bool
    authority_verdict_invariant: bool
    coverage_bitmap_invariant: bool
    ledger_invariant: bool
    certificate_invariant: bool
    authority_evidence_complete: bool
    authority_evidence_errors: tuple[str, ...]
    run_digests: tuple[str, ...]
    outcome_digests: tuple[str, ...]
    authority_evidence_digests: tuple[str, ...]
    schema_version: str = "osc-parallel-invariance-report-v2"

    @property
    def passed(self) -> bool:
        return all(
            (
                self.assignment_invariant,
                self.epoch_invariant,
                self.task_multiset_invariant,
                self.task_set_invariant,
                self.seed_lineage_invariant,
                self.result_order_invariant,
                self.outcome_invariant,
                self.retry_trace_invariant,
                self.authority_verdict_invariant,
                self.coverage_bitmap_invariant,
                self.ledger_invariant,
                self.certificate_invariant,
                self.authority_evidence_complete,
            )
        )

    def to_dict(self) -> dict[str, object]:
        return to_primitive(self)


def _all_equal(values: tuple[object, ...]) -> bool:
    return bool(values) and all(item == values[0] for item in values[1:])


def validate_worker_count_design(
    worker_counts: Iterable[int],
) -> tuple[int, ...]:
    """Validate a non-vacuous workers=1/N authority design before execution."""

    materialized = tuple(worker_counts)
    if any(
        not isinstance(count, int) or isinstance(count, bool) or count < 1
        for count in materialized
    ):
        raise ValueError("worker counts must be positive integers")
    if len(materialized) < 2:
        raise ValueError("worker count design requires at least two distinct counts")
    if len(materialized) != len(set(materialized)):
        raise ValueError("worker count design requires distinct counts")
    if 1 not in materialized:
        raise ValueError("worker count design requires a workers=1 baseline")
    if not any(count > 1 for count in materialized):
        raise ValueError("worker count design requires at least one N greater than 1")
    return materialized


def audit_execution_runs(
    runs: Iterable[ExecutionRun],
    *,
    worker_counts: tuple[int, ...],
    authority_evidence: Iterable[AuthorityEvidenceSnapshot],
) -> InvarianceReport:
    worker_counts = validate_worker_count_design(worker_counts)
    materialized = tuple(runs)
    if not materialized or len(materialized) != len(worker_counts):
        raise ValueError("invariance audit requires one run per worker count")
    snapshots = tuple(authority_evidence)
    evidence_errors: list[str] = []
    for index, (run, worker_count) in enumerate(
        zip(materialized, worker_counts, strict=True)
    ):
        if run.worker_count != worker_count:
            evidence_errors.append(
                "execution_run["
                f"{index}]:worker_count_label_mismatch:"
                f"label={worker_count}:run={run.worker_count}"
            )
    if len(snapshots) != len(materialized):
        evidence_errors.append(
            "authority_evidence_count_mismatch:"
            f"expected={len(materialized)}:actual={len(snapshots)}"
        )
    for index, (run, snapshot) in enumerate(zip(materialized, snapshots)):
        if not isinstance(snapshot, AuthorityEvidenceSnapshot):
            evidence_errors.append(f"authority_evidence[{index}]:invalid_type")
            continue
        evidence_errors.extend(
            f"authority_evidence[{index}]:{error}"
            for error in snapshot.binding_errors(run)
        )
    assignments = tuple(run.assignment_digest for run in materialized)
    epochs = tuple(run.epoch_digest for run in materialized)
    task_multisets = tuple(run.task_multiset_digest for run in materialized)
    task_sets = tuple(run.task_set_digest for run in materialized)
    lineages = tuple(run.seed_lineage_digests for run in materialized)
    orders = tuple(run.result_order for run in materialized)
    outcomes = tuple(run.outcome_digest for run in materialized)
    retries = tuple(run.retry_trace_digest for run in materialized)
    verdicts = tuple(
        snapshot.authority_verdict_digest
        for snapshot in snapshots
        if isinstance(snapshot, AuthorityEvidenceSnapshot)
    )
    coverage = tuple(
        snapshot.coverage_bitmap_digest
        for snapshot in snapshots
        if isinstance(snapshot, AuthorityEvidenceSnapshot)
    )
    ledgers = tuple(
        snapshot.ledger_digest
        for snapshot in snapshots
        if isinstance(snapshot, AuthorityEvidenceSnapshot)
    )
    certificates = tuple(
        snapshot.certificate_digests
        for snapshot in snapshots
        if isinstance(snapshot, AuthorityEvidenceSnapshot)
    )
    return InvarianceReport(
        worker_counts=worker_counts,
        assignment_invariant=_all_equal(assignments),
        epoch_invariant=_all_equal(epochs),
        task_multiset_invariant=_all_equal(task_multisets),
        task_set_invariant=_all_equal(task_sets),
        seed_lineage_invariant=_all_equal(lineages),
        result_order_invariant=_all_equal(orders),
        outcome_invariant=_all_equal(outcomes),
        retry_trace_invariant=_all_equal(retries),
        authority_verdict_invariant=_all_equal(verdicts),
        coverage_bitmap_invariant=_all_equal(coverage),
        ledger_invariant=_all_equal(ledgers),
        certificate_invariant=_all_equal(certificates),
        authority_evidence_complete=(
            len(snapshots) == len(materialized) and not evidence_errors
        ),
        authority_evidence_errors=tuple(evidence_errors),
        run_digests=tuple(run.digest for run in materialized),
        outcome_digests=outcomes,
        authority_evidence_digests=tuple(
            snapshot.digest
            for snapshot in snapshots
            if isinstance(snapshot, AuthorityEvidenceSnapshot)
        ),
    )


def run_worker_count_invariance(
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
) -> tuple[InvarianceReport, tuple[ExecutionRun, ...]]:
    worker_counts = validate_worker_count_design(worker_counts)
    materialized = tuple(tasks)
    runs = tuple(
        DeterministicTaskExecutor(
            max_workers=count,
            capacity=capacity,
            retry_policy=retry_policy,
        ).execute(
            materialized,
            worker_factory(count),
            assignment_digest=assignment_digest,
        )
        for count in worker_counts
    )
    snapshots = tuple(
        authority_evidence_factory(run, count)
        for run, count in zip(runs, worker_counts, strict=True)
    )
    return (
        audit_execution_runs(
            runs,
            worker_counts=worker_counts,
            authority_evidence=snapshots,
        ),
        runs,
    )
