from __future__ import annotations

import copy
import gc
import math
import os
import resource
import traceback
from collections.abc import Callable, Sequence
from concurrent.futures import as_completed
from typing import Any

from datadiff.process_runtime import create_safe_executor


WORKER_BATCHING_SCHEMA_VERSION = "experiment-worker-batching-v1"
WORKER_BATCH_RESULT_SCHEMA_VERSION = "experiment-worker-batch-result-v1"
WORKER_BATCH_ISOLATION_CONTRACT = (
    "each job receives a deep-copied plan and a fresh run_fuzz invocation; run-local backend "
    "sessions/caches and fresh-audit rules remain scoped to that invocation; the OS child exits "
    "after one bounded batch"
)


def resolve_worker_batching(
    args: Any,
    planned_runs: Sequence[dict[str, Any]],
    *,
    jobs: int,
    schedule: str,
) -> dict[str, Any]:
    requested_batch_size = max(0, int(getattr(args, "worker_batch_size", 4) or 0))
    max_rss_mib = max(0, int(getattr(args, "worker_max_rss_mib", 2048) or 0))
    retry_limit = max(0, int(getattr(args, "worker_retry_limit", 1) or 0))
    enabled = (
        requested_batch_size > 0
        and int(jobs) > 1
        and len(planned_runs) > 1
        and str(schedule) != "adaptive"
    )
    if requested_batch_size <= 0:
        reason = "disabled_by_worker_batch_size"
    elif str(schedule) == "adaptive":
        reason = "adaptive_scheduler_uses_dynamic_round_workers"
    elif int(jobs) <= 1 or len(planned_runs) <= 1:
        reason = "serial_execution_is_already_process_persistent"
    else:
        reason = "enabled"
    return {
        "schema_version": WORKER_BATCHING_SCHEMA_VERSION,
        "enabled": enabled,
        "reason": reason,
        "requested_batch_size": requested_batch_size,
        "effective_batch_size": (
            min(requested_batch_size, len(planned_runs)) if enabled else 0
        ),
        "max_rss_mib": max_rss_mib,
        "max_rss_kib": max_rss_mib * 1024,
        "retry_limit": retry_limit,
        "planned_run_count": len(planned_runs),
        "worker_count": int(jobs),
        "schedule": str(schedule),
        "executor_wave_batch_limit": 1 if enabled else None,
        "executor_recycle": "new process pool per bounded wave" if enabled else "disabled",
        "isolation_contract": WORKER_BATCH_ISOLATION_CONTRACT,
        "events": [],
        "summary": {},
    }


def build_worker_batches(
    planned_runs: Sequence[dict[str, Any]],
    *,
    worker_count: int,
    batch_size: int,
    experiment_job_sort_key_func: Callable[[dict[str, Any]], tuple[float, int]],
    job_estimated_cost_func: Callable[[dict[str, Any]], float],
) -> list[dict[str, Any]]:
    if not planned_runs:
        return []
    size = max(1, int(batch_size))
    scheduled = sorted(planned_runs, key=experiment_job_sort_key_func)
    batch_count = min(
        len(scheduled),
        max(max(1, int(worker_count)), math.ceil(len(scheduled) / size)),
    )
    bins = [
        {
            "batch_index": index,
            "jobs": [],
            "estimated_cost_sum": 0.0,
            "estimated_peak_cost": 0.0,
        }
        for index in range(batch_count)
    ]
    for job in scheduled:
        candidates = [row for row in bins if len(row["jobs"]) < size]
        if not candidates:
            raise RuntimeError("worker batch capacity was exhausted")
        target = min(
            candidates,
            key=lambda row: (
                float(row["estimated_cost_sum"]),
                len(row["jobs"]),
                int(row["batch_index"]),
            ),
        )
        cost = float(job_estimated_cost_func(job))
        target["jobs"].append(dict(job))
        target["estimated_cost_sum"] += cost
        target["estimated_peak_cost"] = max(
            float(target["estimated_peak_cost"]), cost
        )
    batches = []
    for row in bins:
        jobs = list(row["jobs"])
        if not jobs:
            continue
        batches.append(
            {
                "batch_id": f"worker-batch-{int(row['batch_index']):04d}",
                "batch_index": int(row["batch_index"]),
                "jobs": jobs,
                "job_orders": [int(job["order"]) for job in jobs],
                "estimated_cost_sum": round(float(row["estimated_cost_sum"]), 6),
                "estimated_peak_cost": round(float(row["estimated_peak_cost"]), 6),
                "replay_of": None,
                "replay_reason": "",
            }
        )
    return batches


def run_worker_batch(
    batch: dict[str, Any],
    *,
    run_experiment_job_func: Callable[[dict[str, Any]], dict[str, Any]],
    process_rss_kib_func: Callable[[], int] | None = None,
) -> dict[str, Any]:
    rss_func = process_rss_kib_func or process_max_rss_kib
    jobs = [copy.deepcopy(job) for job in batch.get("jobs", [])]
    max_rss_kib = max(0, int(batch.get("max_rss_kib", 0) or 0))
    started_rss_kib = int(rss_func())
    completed_results: list[dict[str, Any]] = []
    job_events: list[dict[str, Any]] = []
    for index, job in enumerate(jobs):
        before_rss_kib = int(rss_func())
        try:
            result = run_experiment_job_func(job)
        except Exception as exc:
            after_rss_kib = int(rss_func())
            return {
                "schema_version": WORKER_BATCH_RESULT_SCHEMA_VERSION,
                "batch_id": str(batch.get("batch_id", "")),
                "worker_pid": os.getpid(),
                "status": "job_failed",
                "completed_results": completed_results,
                "completed_job_orders": [
                    int(item["order"]) for item in completed_results
                ],
                "failed_job": job,
                "remaining_jobs": jobs[index + 1 :],
                "failure": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
                "job_events": job_events,
                "started_rss_kib": started_rss_kib,
                "final_rss_kib": after_rss_kib,
                "max_rss_kib": max(started_rss_kib, after_rss_kib),
                "recycle_reason": "job_exception",
                "isolation_contract": WORKER_BATCH_ISOLATION_CONTRACT,
            }
        after_rss_kib = int(rss_func())
        completed_results.append(result)
        job_events.append(
            {
                "order": int(job["order"]),
                "seed": int(job.get("seed", 0) or 0),
                "before_rss_kib": before_rss_kib,
                "after_rss_kib": after_rss_kib,
            }
        )
        gc.collect()
        remaining_jobs = jobs[index + 1 :]
        if max_rss_kib > 0 and after_rss_kib >= max_rss_kib and remaining_jobs:
            return {
                "schema_version": WORKER_BATCH_RESULT_SCHEMA_VERSION,
                "batch_id": str(batch.get("batch_id", "")),
                "worker_pid": os.getpid(),
                "status": "recycle_requested",
                "completed_results": completed_results,
                "completed_job_orders": [
                    int(item["order"]) for item in completed_results
                ],
                "failed_job": None,
                "remaining_jobs": remaining_jobs,
                "failure": None,
                "job_events": job_events,
                "started_rss_kib": started_rss_kib,
                "final_rss_kib": after_rss_kib,
                "max_rss_kib": max(
                    [started_rss_kib, after_rss_kib]
                    + [int(item["after_rss_kib"]) for item in job_events]
                ),
                "recycle_reason": "rss_limit",
                "isolation_contract": WORKER_BATCH_ISOLATION_CONTRACT,
            }
    final_rss_kib = int(rss_func())
    return {
        "schema_version": WORKER_BATCH_RESULT_SCHEMA_VERSION,
        "batch_id": str(batch.get("batch_id", "")),
        "worker_pid": os.getpid(),
        "status": "completed",
        "completed_results": completed_results,
        "completed_job_orders": [int(item["order"]) for item in completed_results],
        "failed_job": None,
        "remaining_jobs": [],
        "failure": None,
        "job_events": job_events,
        "started_rss_kib": started_rss_kib,
        "final_rss_kib": final_rss_kib,
        "max_rss_kib": max(
            [started_rss_kib, final_rss_kib]
            + [int(item["after_rss_kib"]) for item in job_events]
        ),
        "recycle_reason": "batch_limit",
        "isolation_contract": WORKER_BATCH_ISOLATION_CONTRACT,
    }


def run_batched_parallel(
    executor_cls: type,
    worker_count: int,
    planned_runs: Sequence[dict[str, Any]],
    *,
    max_parallel_cost: float,
    worker_batch_size: int,
    worker_max_rss_kib: int,
    worker_retry_limit: int,
    experiment_job_sort_key_func: Callable[[dict[str, Any]], tuple[float, int]],
    job_estimated_cost_func: Callable[[dict[str, Any]], float],
    run_experiment_job_batch_func: Callable[[dict[str, Any]], dict[str, Any]],
    worker_batch_manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    queued = build_worker_batches(
        planned_runs,
        worker_count=worker_count,
        batch_size=worker_batch_size,
        experiment_job_sort_key_func=experiment_job_sort_key_func,
        job_estimated_cost_func=job_estimated_cost_func,
    )
    attempts = {int(job["order"]): 0 for job in planned_runs}
    completed: list[dict[str, Any]] = []
    completed_orders: set[int] = set()
    events = worker_batch_manifest.setdefault("events", [])
    replay_counter = 0
    while queued:
        wave = select_batch_wave(
            queued,
            worker_count=worker_count,
            max_parallel_cost=max_parallel_cost,
        )
        for batch in wave:
            batch["max_rss_kib"] = int(worker_max_rss_kib)
        with create_recycling_executor(executor_cls, max_workers=len(wave)) as executor:
            futures = {
                executor.submit(run_experiment_job_batch_func, batch): batch
                for batch in wave
            }
            for future in as_completed(futures):
                batch = futures[future]
                try:
                    payload = future.result()
                except BaseException as exc:
                    retry_jobs = list(batch["jobs"])
                    exhausted = _increment_attempts(
                        attempts,
                        retry_jobs,
                        retry_limit=worker_retry_limit,
                    )
                    events.append(
                        _hard_failure_event(
                            batch,
                            exc,
                            retry_jobs=retry_jobs,
                            exhausted=exhausted,
                        )
                    )
                    if exhausted:
                        raise RuntimeError(
                            "persistent experiment worker batch exhausted retry limit: "
                            f"batch={batch['batch_id']} orders={exhausted}"
                        ) from exc
                    replay_counter += 1
                    queued.insert(
                        0,
                        replay_batch(
                            batch,
                            retry_jobs,
                            replay_counter=replay_counter,
                            reason="worker_process_failure",
                        ),
                    )
                    continue

                results = list(payload.get("completed_results", []))
                for result in results:
                    order = int(result["order"])
                    if order in completed_orders:
                        continue
                    completed_orders.add(order)
                    completed.append(result)
                    print(result["message"], flush=True)
                events.append(compact_batch_event(batch, payload))
                status = str(payload.get("status", ""))
                remaining_jobs = list(payload.get("remaining_jobs", []))
                if status == "job_failed":
                    failed_job = payload.get("failed_job")
                    retry_jobs = ([failed_job] if isinstance(failed_job, dict) else []) + remaining_jobs
                    failed_only = [failed_job] if isinstance(failed_job, dict) else []
                    exhausted = _increment_attempts(
                        attempts,
                        failed_only,
                        retry_limit=worker_retry_limit,
                    )
                    if exhausted:
                        failure = payload.get("failure", {}) or {}
                        raise RuntimeError(
                            "persistent experiment worker job exhausted retry limit: "
                            f"orders={exhausted} type={failure.get('type', '')} "
                            f"message={failure.get('message', '')}"
                        )
                    if retry_jobs:
                        replay_counter += 1
                        queued.insert(
                            0,
                            replay_batch(
                                batch,
                                retry_jobs,
                                replay_counter=replay_counter,
                                reason="job_exception",
                            ),
                        )
                elif status == "recycle_requested" and remaining_jobs:
                    replay_counter += 1
                    queued.insert(
                        0,
                        replay_batch(
                            batch,
                            remaining_jobs,
                            replay_counter=replay_counter,
                            reason="rss_limit",
                        ),
                    )
                elif status != "completed":
                    raise RuntimeError(
                        f"unsupported persistent worker batch status: {status!r}"
                    )
    expected_orders = {int(job["order"]) for job in planned_runs}
    if completed_orders != expected_orders:
        raise RuntimeError(
            "persistent worker batching did not complete every planned run: "
            f"missing={sorted(expected_orders - completed_orders)}"
        )
    worker_batch_manifest["summary"] = {
        "initial_batch_count": sum(
            1 for event in events if not event.get("replay_of")
        ),
        "event_count": len(events),
        "replay_count": sum(bool(event.get("replay_of")) for event in events),
        "rss_recycle_count": sum(
            event.get("recycle_reason") == "rss_limit" for event in events
        ),
        "job_failure_count": sum(
            event.get("status") == "job_failed" for event in events
        ),
        "hard_failure_count": sum(
            event.get("status") == "worker_process_failure" for event in events
        ),
        "completed_run_count": len(completed_orders),
        "worker_pids": sorted(
            {
                int(event["worker_pid"])
                for event in events
                if event.get("worker_pid") is not None
            }
        ),
        "max_observed_rss_kib": max(
            (int(event.get("max_rss_kib", 0) or 0) for event in events),
            default=0,
        ),
    }
    return completed


def select_batch_wave(
    queued: list[dict[str, Any]],
    *,
    worker_count: int,
    max_parallel_cost: float,
) -> list[dict[str, Any]]:
    wave: list[dict[str, Any]] = []
    running_cost = 0.0
    while queued and len(wave) < max(1, int(worker_count)):
        available = float(max_parallel_cost) - running_cost
        index = next(
            (
                index
                for index, batch in enumerate(queued)
                if float(batch["estimated_peak_cost"]) <= max(0.0, available)
            ),
            None,
        )
        if index is None:
            if wave:
                break
            index = 0
        batch = queued.pop(index)
        wave.append(batch)
        running_cost += float(batch["estimated_peak_cost"])
    return wave


def replay_batch(
    source: dict[str, Any],
    jobs: Sequence[dict[str, Any]],
    *,
    replay_counter: int,
    reason: str,
) -> dict[str, Any]:
    job_rows = [dict(job) for job in jobs]
    costs = [float(job.get("estimated_cost", 1.0) or 1.0) for job in job_rows]
    return {
        "batch_id": f"{source['batch_id']}-replay-{int(replay_counter):04d}",
        "batch_index": int(source.get("batch_index", 0)),
        "jobs": job_rows,
        "job_orders": [int(job["order"]) for job in job_rows],
        "estimated_cost_sum": round(sum(costs), 6),
        "estimated_peak_cost": round(max(costs, default=1.0), 6),
        "replay_of": str(source["batch_id"]),
        "replay_reason": str(reason),
    }


def compact_batch_event(
    batch: dict[str, Any], payload: dict[str, Any]
) -> dict[str, Any]:
    failure = payload.get("failure") or {}
    return {
        "batch_id": str(batch["batch_id"]),
        "replay_of": batch.get("replay_of"),
        "replay_reason": str(batch.get("replay_reason", "")),
        "planned_job_orders": [int(value) for value in batch["job_orders"]],
        "completed_job_orders": [
            int(value) for value in payload.get("completed_job_orders", [])
        ],
        "remaining_job_orders": [
            int(job["order"]) for job in payload.get("remaining_jobs", [])
        ],
        "status": str(payload.get("status", "")),
        "worker_pid": payload.get("worker_pid"),
        "started_rss_kib": int(payload.get("started_rss_kib", 0) or 0),
        "final_rss_kib": int(payload.get("final_rss_kib", 0) or 0),
        "max_rss_kib": int(payload.get("max_rss_kib", 0) or 0),
        "recycle_reason": str(payload.get("recycle_reason", "")),
        "failure_type": str(failure.get("type", "")),
        "failure_message": str(failure.get("message", "")),
        "job_events": list(payload.get("job_events", [])),
        "isolation_contract": str(payload.get("isolation_contract", "")),
    }


def create_recycling_executor(executor_cls: type, *, max_workers: int) -> Any:
    return create_safe_executor(
        executor_cls,
        max_workers=max_workers,
        max_tasks_per_child=1,
    )


def process_max_rss_kib() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _increment_attempts(
    attempts: dict[int, int],
    jobs: Sequence[dict[str, Any]],
    *,
    retry_limit: int,
) -> list[int]:
    exhausted = []
    for job in jobs:
        order = int(job["order"])
        attempts[order] = int(attempts.get(order, 0)) + 1
        if attempts[order] > max(0, int(retry_limit)):
            exhausted.append(order)
    return exhausted


def _hard_failure_event(
    batch: dict[str, Any],
    exc: BaseException,
    *,
    retry_jobs: Sequence[dict[str, Any]],
    exhausted: Sequence[int],
) -> dict[str, Any]:
    return {
        "batch_id": str(batch["batch_id"]),
        "replay_of": batch.get("replay_of"),
        "replay_reason": str(batch.get("replay_reason", "")),
        "planned_job_orders": [int(job["order"]) for job in retry_jobs],
        "completed_job_orders": [],
        "remaining_job_orders": [int(job["order"]) for job in retry_jobs],
        "status": "worker_process_failure",
        "worker_pid": None,
        "started_rss_kib": 0,
        "final_rss_kib": 0,
        "max_rss_kib": 0,
        "recycle_reason": "worker_process_failure",
        "failure_type": type(exc).__name__,
        "failure_message": str(exc),
        "retry_exhausted_orders": [int(order) for order in exhausted],
        "job_events": [],
        "isolation_contract": WORKER_BATCH_ISOLATION_CONTRACT,
    }
