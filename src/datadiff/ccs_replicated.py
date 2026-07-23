from __future__ import annotations

import random
import statistics
from collections.abc import Mapping, Sequence
from typing import Any

from datadiff.ccs_ablation import CCS_IR_ABLATION_ARMS
from datadiff.experiment_manifest import stable_digest
from datadiff.util import utc_now


CCS_FRESH_PROCESS_REPLICATION_SCHEMA_VERSION = "ccs-ir-fresh-process-replication-v1"
CCS_FRESH_PROCESS_WORKER_SCHEMA_VERSION = "ccs-ir-fresh-process-worker-v1"


def replication_schedule(repetitions: int) -> list[dict[str, Any]]:
    count = max(1, int(repetitions))
    return [
        {
            "replication": index,
            "process_order": list(
                CCS_IR_ABLATION_ARMS
                if index % 2 == 0
                else tuple(reversed(CCS_IR_ABLATION_ARMS))
            ),
        }
        for index in range(count)
    ]


def validate_worker_result(worker: Mapping[str, Any]) -> None:
    if worker.get("schema_version") != CCS_FRESH_PROCESS_WORKER_SCHEMA_VERSION:
        raise ValueError("unsupported CCS fresh-process worker schema")
    arm_run = worker.get("arm_run", {})
    if not isinstance(arm_run, Mapping):
        raise TypeError("worker arm_run must be a mapping")
    expected_arm_digest = stable_digest(
        "ccs-arm-run",
        {key: value for key, value in arm_run.items() if key != "result_digest"},
    )
    if arm_run.get("result_digest") != expected_arm_digest:
        raise ValueError("worker arm-run digest mismatch")
    expected_worker_digest = stable_digest(
        "ccs-worker",
        {key: value for key, value in worker.items() if key != "worker_digest"},
    )
    if worker.get("worker_digest") != expected_worker_digest:
        raise ValueError("worker result digest mismatch")


def combine_fresh_process_repetitions(
    repetitions: Sequence[Mapping[str, Any]],
    *,
    corpus_digest: str,
    backends: Sequence[str],
    thread_environment: Mapping[str, str],
    warmup_seed: int,
    bootstrap_seed: int = 20260713,
    bootstrap_resamples: int = 10000,
) -> dict[str, Any]:
    normalized_repetitions: list[dict[str, Any]] = []
    all_pairs: list[dict[str, Any]] = []
    for repetition in repetitions:
        replication = int(repetition.get("replication", len(normalized_repetitions)))
        workers = repetition.get("workers", {})
        if not isinstance(workers, Mapping):
            raise TypeError("replication workers must be a mapping")
        for arm_id in CCS_IR_ABLATION_ARMS:
            worker = workers.get(arm_id)
            if not isinstance(worker, Mapping):
                raise ValueError(f"replication {replication} is missing worker {arm_id}")
            validate_worker_result(worker)
        legacy = workers["contract_cartesian"]
        ccs = workers["contract_ccs_cartesian"]
        legacy_run = legacy["arm_run"]
        ccs_run = ccs["arm_run"]
        _validate_worker_pair(legacy_run, ccs_run, corpus_digest, backends)
        pairs = _worker_pairs(replication, legacy_run["runs"], ccs_run["runs"])
        all_pairs.extend(pairs)
        legacy_wall = float(legacy_run["summary"]["total_observed_wall_ms"])
        ccs_wall = float(ccs_run["summary"]["total_observed_wall_ms"])
        legacy_process = legacy.get("process_resources", {})
        ccs_process = ccs.get("process_resources", {})
        legacy_external = float(repetition["external_process_wall_ms"]["contract_cartesian"])
        ccs_external = float(repetition["external_process_wall_ms"]["contract_ccs_cartesian"])
        normalized_repetitions.append(
            {
                "replication": replication,
                "process_order": list(repetition.get("process_order", ())),
                "legacy_result_digest": legacy_run["result_digest"],
                "ccs_result_digest": ccs_run["result_digest"],
                "outcome_equivalence_rate": _rate(pairs, "outcome_equal"),
                "comparison_block_match_rate": _rate(
                    pairs, "comparison_block_equal"
                ),
                "backend_call_match_rate": _rate(pairs, "backend_calls_equal"),
                "measured_case_wall_ratio": ccs_wall / legacy_wall,
                "external_process_wall_ratio": ccs_external / legacy_external,
                "process_cpu_ratio": _ratio(
                    _process_cpu_seconds(ccs_process),
                    _process_cpu_seconds(legacy_process),
                ),
                "legacy": {
                    "summary": legacy_run["summary"],
                    "warmup": legacy_run["warmup"],
                    "process_resources": dict(legacy_process),
                    "external_process_wall_ms": legacy_external,
                },
                "ccs": {
                    "summary": ccs_run["summary"],
                    "warmup": ccs_run["warmup"],
                    "process_resources": dict(ccs_process),
                    "external_process_wall_ms": ccs_external,
                },
            }
        )

    measured_ratios = [
        float(item["measured_case_wall_ratio"]) for item in normalized_repetitions
    ]
    external_ratios = [
        float(item["external_process_wall_ratio"]) for item in normalized_repetitions
    ]
    cpu_ratios = [
        float(item["process_cpu_ratio"])
        for item in normalized_repetitions
        if item.get("process_cpu_ratio") is not None
    ]
    payload = {
        "schema_version": CCS_FRESH_PROCESS_REPLICATION_SCHEMA_VERSION,
        "generated_at": utc_now(),
        "corpus_digest": str(corpus_digest),
        "backends": [str(backend) for backend in backends],
        "arm_ids": list(CCS_IR_ABLATION_ARMS),
        "single_changed_dimension": "ir_mode",
        "warmup_seed": int(warmup_seed),
        "warmup_excluded_from_case_metrics": True,
        "thread_environment": dict(thread_environment),
        "replication_count": len(normalized_repetitions),
        "case_pair_count": len(all_pairs),
        "repetitions": normalized_repetitions,
        "validity_gates": {
            "outcome_equivalence_rate": _rate(all_pairs, "outcome_equal"),
            "comparison_block_match_rate": _rate(
                all_pairs, "comparison_block_equal"
            ),
            "backend_call_match_rate": _rate(all_pairs, "backend_calls_equal"),
        },
        "summary": {
            "median_measured_case_wall_ratio": statistics.median(measured_ratios),
            "mean_measured_case_wall_ratio": statistics.fmean(measured_ratios),
            "median_external_process_wall_ratio": statistics.median(external_ratios),
            "mean_external_process_wall_ratio": statistics.fmean(external_ratios),
            "median_process_cpu_ratio": statistics.median(cpu_ratios) if cpu_ratios else None,
            "mean_process_cpu_ratio": statistics.fmean(cpu_ratios) if cpu_ratios else None,
            "measured_case_wall_ratio_bootstrap_95": _bootstrap_median_interval(
                measured_ratios,
                seed=bootstrap_seed,
                resamples=bootstrap_resamples,
            ),
            "external_process_wall_ratio_bootstrap_95": _bootstrap_median_interval(
                external_ratios,
                seed=bootstrap_seed + 1,
                resamples=bootstrap_resamples,
            ),
            "bootstrap_seed": bootstrap_seed,
            "bootstrap_resamples": bootstrap_resamples,
        },
        "stopping_rule": "complete_all_preregistered_fresh_process_repetitions",
    }
    payload["result_digest"] = stable_digest("ccs-replicated", payload)
    return payload


def _validate_worker_pair(
    legacy: Mapping[str, Any],
    ccs: Mapping[str, Any],
    corpus_digest: str,
    backends: Sequence[str],
) -> None:
    for arm_run in (legacy, ccs):
        if arm_run.get("corpus_digest") != corpus_digest:
            raise ValueError("worker corpus digest does not match replication corpus")
        if list(arm_run.get("backends", ())) != list(backends):
            raise ValueError("worker backend order does not match replication protocol")
    if legacy.get("case_count") != ccs.get("case_count"):
        raise ValueError("worker case counts do not match")


def _worker_pairs(
    replication: int,
    legacy_runs: Sequence[Mapping[str, Any]],
    ccs_runs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    legacy_by_index = {int(run["case_index"]): run for run in legacy_runs}
    ccs_by_index = {int(run["case_index"]): run for run in ccs_runs}
    if set(legacy_by_index) != set(ccs_by_index):
        raise ValueError("worker case indices do not match")
    return [
        {
            "replication": replication,
            "case_index": case_index,
            "case_id": legacy_by_index[case_index]["case_id"],
            "outcome_equal": (
                legacy_by_index[case_index]["outcome_digest"]
                == ccs_by_index[case_index]["outcome_digest"]
            ),
            "comparison_block_equal": (
                legacy_by_index[case_index]["comparison_block_digest"]
                == ccs_by_index[case_index]["comparison_block_digest"]
            ),
            "backend_calls_equal": (
                legacy_by_index[case_index]["backend_calls"]
                == ccs_by_index[case_index]["backend_calls"]
            ),
        }
        for case_index in sorted(legacy_by_index)
    ]


def _bootstrap_median_interval(
    values: Sequence[float],
    *,
    seed: int,
    resamples: int,
) -> list[float] | None:
    if not values:
        return None
    rng = random.Random(int(seed))
    estimates = sorted(
        statistics.median(values[rng.randrange(len(values))] for _ in values)
        for _ in range(max(1, int(resamples)))
    )
    lower_index = max(0, int(0.025 * len(estimates)) - 1)
    upper_index = min(len(estimates) - 1, int(0.975 * len(estimates)) - 1)
    return [estimates[lower_index], estimates[upper_index]]


def _process_cpu_seconds(resources: Mapping[str, Any]) -> float:
    return float(resources.get("user_cpu_seconds", 0.0) or 0.0) + float(
        resources.get("system_cpu_seconds", 0.0) or 0.0
    )


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator > 0.0 else None


def _rate(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    return sum(1 for row in rows if bool(row.get(key))) / len(rows) if rows else 0.0
