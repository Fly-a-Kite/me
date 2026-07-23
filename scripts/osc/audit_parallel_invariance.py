#!/usr/bin/env python3
"""Audit worker-count, retry, and completion-order invariance on frozen tasks."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from datadiff_osc._canonical import canonical_json, stable_digest  # noqa: E402
from datadiff_osc.parallel import (  # noqa: E402
    ResourceCapacity,
    RetryPolicy,
    WorkerCrashError,
    WorkerResult,
    run_worker_count_invariance,
)
from datadiff_osc.parallel.invariance import (  # noqa: E402
    AuthorityEvidenceSnapshot,
    validate_worker_count_design,
)
from datadiff_osc.schemas import (  # noqa: E402
    ExecutionStatus,
    FailureKind,
    ResourceTokens,
    SeedLineage,
    SeedStage,
    StructuredExecutionOutcome,
    TaskIdentity,
    TaskKind,
    TaskSpec,
)


def _integers(value: str) -> tuple[int, ...]:
    parts = value.split(",")
    if not parts or any(not item for item in parts):
        raise argparse.ArgumentTypeError("worker count list contains an empty item")
    try:
        result = tuple(int(item) for item in parts)
        return validate_worker_count_design(result)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _tasks(count: int, seed: int) -> tuple[TaskSpec, ...]:
    tasks = []
    for index in range(count):
        lineage = SeedLineage(
            protocol_digest="osc-parallel-invariance-v1",
            master_seed=seed,
            lane_id="synthetic-invariance",
            case_index=index,
            stage_name=SeedStage.BACKEND,
        )
        identity = TaskIdentity(
            protocol_digest="osc-parallel-invariance-v1",
            task_kind=TaskKind.BACKEND_EXECUTION,
            epoch_index=0,
            decision_index=index,
            seed_lineage_digest=lineage.digest,
            endpoint_id=f"endpoint-{index}",
            backend="synthetic-backend",
        )
        tasks.append(
            TaskSpec(
                identity=identity,
                dependency_task_ids=(),
                resources=ResourceTokens(1, 1024, "default", 0),
                payload_digest=stable_digest("osc-invariance-task-payload", index),
            )
        )
    return tuple(tasks)


def _emit(payload, output: Path | None) -> None:
    text = canonical_json(payload) + "\n"
    if output is None:
        sys.stdout.write(text)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")


def _synthetic_authority_snapshot(run, _worker_count):
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
        authority_verdict_digest=stable_digest(
            "osc-invariance-synthetic-authority-verdicts", rows
        ),
        coverage_bitmap_digest=stable_digest(
            "osc-invariance-synthetic-coverage-bitmap",
            tuple((row[0], row[3]) for row in rows),
        ),
        ledger_digest=stable_digest("osc-invariance-synthetic-ledger", rows),
        certificate_digests=tuple(
            stable_digest("osc-invariance-synthetic-certificate", row)
            for row in rows
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=int, default=64)
    parser.add_argument("--workers", type=_integers, default=(1, 2, 4, 6))
    parser.add_argument("--seed", type=int, default=42000001)
    parser.add_argument("--retry-index", type=int, default=3)
    parser.add_argument("--completion-delay-ms", type=float, default=0.05)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        args.workers = validate_worker_count_design(args.workers)
    except ValueError as exc:
        parser.error(str(exc))
    if args.tasks < 1 or not 0 <= args.retry_index < args.tasks:
        parser.error("task count and retry index are inconsistent")
    tasks = _tasks(args.tasks, args.seed)
    capacity = ResourceCapacity.build(
        cpu_tokens=max(args.workers),
        rss_bytes=max(args.workers) * 2048,
        io_slots={"default": max(args.workers)},
        backend_internal_threads=max(args.workers),
    )

    def worker_factory(worker_count):
        def worker(invocation):
            index = invocation.task.identity.decision_index
            if index == args.retry_index and invocation.task.identity.attempt == 0:
                raise WorkerCrashError("deterministic retry injection")
            delay = ((args.tasks - index) if worker_count > 1 else index) * args.completion_delay_ms
            if delay > 0:
                time.sleep(delay / 1000.0)
            outcome = StructuredExecutionOutcome(
                endpoint_id=invocation.task.identity.endpoint_id,
                status=ExecutionStatus.OK,
                failure_kind=FailureKind.NONE,
            )
            return WorkerResult(
                outcome,
                stable_digest("osc-invariance-worker-result", invocation.task.payload_digest),
            )

        return worker

    report, runs = run_worker_count_invariance(
        tasks,
        worker_factory,
        capacity=capacity,
        worker_counts=args.workers,
        retry_policy=RetryPolicy(max_retries=1),
        assignment_digest=stable_digest("osc-invariance-assignment", (args.seed, args.tasks)),
        authority_evidence_factory=_synthetic_authority_snapshot,
    )
    _emit(
        {
            "schema_version": "osc-parallel-invariance-script-output-v1",
            "report": report,
            "run_digests": tuple(run.digest for run in runs),
            "synthetic_authority_evidence": True,
            "phase6_gate_eligible": False,
            "twenty_four_hour_run_authorized": False,
        },
        args.output,
    )
    return 0 if report.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
