from __future__ import annotations

import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from datadiff.config import ExperimentConfig
from datadiff.dsl import Case
from datadiff.env import collect_environment
from datadiff.execution import BackendExecutionSession
from datadiff.experiment_manifest import stable_digest
from datadiff.runner import run_loaded_case
from datadiff.targets import describe_targets
from datadiff.util import utc_now


FROZEN_CASE_CORPUS_SCHEMA_VERSION = "ccs-ir-frozen-case-corpus-v1"
CCS_IR_ABLATION_SCHEMA_VERSION = "ccs-ir-paired-ablation-v1"
CCS_IR_ARM_RUN_SCHEMA_VERSION = "ccs-ir-fresh-process-arm-run-v1"
CCS_IR_ABLATION_ARMS = ("contract_cartesian", "contract_ccs_cartesian")

RunCaseFn = Callable[..., dict[str, Any]]


def freeze_case_corpus(
    cases: Sequence[Case],
    *,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    case_payloads = [case.to_dict() for case in cases]
    payload = {
        "schema_version": FROZEN_CASE_CORPUS_SCHEMA_VERSION,
        "case_count": len(case_payloads),
        "case_ids": [str(case.get("case_id", "")) for case in case_payloads],
        "provenance": dict(provenance or {}),
        "cases": case_payloads,
    }
    payload["corpus_digest"] = stable_digest("frozen-corpus", payload)
    return payload


def load_frozen_cases(corpus: Mapping[str, Any]) -> list[Case]:
    validate_frozen_case_corpus(corpus)
    return [Case.from_dict(dict(payload)) for payload in corpus.get("cases", ())]


def validate_frozen_case_corpus(corpus: Mapping[str, Any]) -> None:
    if corpus.get("schema_version") != FROZEN_CASE_CORPUS_SCHEMA_VERSION:
        raise ValueError("unsupported frozen case corpus schema")
    cases = corpus.get("cases", ())
    if not isinstance(cases, list):
        raise TypeError("frozen case corpus cases must be a list")
    if int(corpus.get("case_count", -1)) != len(cases):
        raise ValueError("frozen case corpus count does not match cases")
    case_ids = [str(case.get("case_id", "")) for case in cases if isinstance(case, Mapping)]
    if len(case_ids) != len(cases) or case_ids != list(corpus.get("case_ids", ())):
        raise ValueError("frozen case corpus case order does not match case_ids")
    payload = {key: value for key, value in corpus.items() if key != "corpus_digest"}
    expected = stable_digest("frozen-corpus", payload)
    if str(corpus.get("corpus_digest", "")) != expected:
        raise ValueError("frozen case corpus digest mismatch")


def run_frozen_corpus_arm(
    corpus: Mapping[str, Any],
    backends: Sequence[str],
    arm_id: str,
    *,
    base_config: ExperimentConfig | None = None,
    warmup_case: Case | None = None,
    run_case_fn: RunCaseFn = run_loaded_case,
    collect_environment_fn: Callable[[], dict[str, str]] = collect_environment,
    describe_targets_fn: Callable[[list[str]], list[dict[str, Any]]] = describe_targets,
    perf_counter_fn: Callable[[], float] = time.perf_counter,
    use_execution_session: bool = True,
    execution_session_factory: Callable[..., BackendExecutionSession] = BackendExecutionSession,
) -> dict[str, Any]:
    validate_frozen_case_corpus(corpus)
    if arm_id not in CCS_IR_ABLATION_ARMS:
        raise ValueError(f"unsupported CCS-IR ablation arm: {arm_id}")
    resolved_backends = tuple(str(backend) for backend in backends if str(backend))
    if not resolved_backends:
        raise ValueError("fresh-process arm run requires at least one backend")
    config = _arm_config(base_config or ExperimentConfig(), arm_id)
    environment = collect_environment_fn()
    targets = describe_targets_fn(list(resolved_backends))
    session = (
        execution_session_factory(
            list(resolved_backends),
            environment=environment,
            adapter_revision=config.method_arm_manifest["digest"],
        )
        if use_execution_session
        else None
    )
    warmup: dict[str, Any] = {"executed": False}
    runs: list[dict[str, Any]] = []
    try:
        if warmup_case is not None:
            started = perf_counter_fn()
            row = run_case_fn(
                Case.from_dict(warmup_case.to_dict()),
                backends=list(resolved_backends),
                config=config,
                save_artifact=False,
                environment=environment,
                target_specs=targets,
                execution_session=session,
            )
            warmup = {
                "executed": True,
                "case_id": warmup_case.case_id,
                "seed": warmup_case.seed,
                "observed_wall_ms": max(
                    0.0, (perf_counter_fn() - started) * 1000.0
                ),
                "status": row.get("status", "unknown"),
                "finding_count": len(row.get("findings", ())),
                "excluded_from_metrics": True,
            }
        for case_index, case_payload in enumerate(corpus.get("cases", ())):
            case = Case.from_dict(dict(case_payload))
            started = perf_counter_fn()
            row = run_case_fn(
                case,
                backends=list(resolved_backends),
                config=config,
                save_artifact=False,
                environment=environment,
                target_specs=targets,
                execution_session=session,
            )
            runs.append(
                _run_record(
                    case_index,
                    arm_id,
                    row,
                    max(0.0, (perf_counter_fn() - started) * 1000.0),
                )
            )
    finally:
        if session is not None:
            session.close()
    payload = {
        "schema_version": CCS_IR_ARM_RUN_SCHEMA_VERSION,
        "generated_at": utc_now(),
        "corpus_digest": str(corpus.get("corpus_digest", "")),
        "case_count": len(runs),
        "backends": list(resolved_backends),
        "environment": environment,
        "targets": targets,
        "arm": config.method_arm_manifest,
        "warmup": warmup,
        "runs": runs,
        "summary": _single_arm_summary(runs),
        "stopping_rule": "execute_all_frozen_cases_without_outcome_based_early_stopping",
    }
    payload["result_digest"] = stable_digest("ccs-arm-run", payload)
    return payload


def run_ccs_ir_paired_ablation(
    corpus: Mapping[str, Any],
    backends: Sequence[str],
    *,
    base_config: ExperimentConfig | None = None,
    run_case_fn: RunCaseFn = run_loaded_case,
    collect_environment_fn: Callable[[], dict[str, str]] = collect_environment,
    describe_targets_fn: Callable[[list[str]], list[dict[str, Any]]] = describe_targets,
    perf_counter_fn: Callable[[], float] = time.perf_counter,
    use_execution_sessions: bool = True,
    execution_session_factory: Callable[..., BackendExecutionSession] = BackendExecutionSession,
) -> dict[str, Any]:
    validate_frozen_case_corpus(corpus)
    resolved_backends = tuple(str(backend) for backend in backends if str(backend))
    if not resolved_backends:
        raise ValueError("paired CCS-IR ablation requires at least one backend")
    config = base_config or ExperimentConfig()
    arm_configs = {
        arm_id: _arm_config(config, arm_id) for arm_id in CCS_IR_ABLATION_ARMS
    }
    environment = collect_environment_fn()
    targets = describe_targets_fn(list(resolved_backends))
    sessions = _execution_sessions(
        resolved_backends,
        arm_configs,
        environment,
        use_execution_sessions=use_execution_sessions,
        execution_session_factory=execution_session_factory,
    )
    runs: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    schedule: list[dict[str, Any]] = []
    try:
        for case_index, case_payload in enumerate(corpus.get("cases", ())):
            order = (
                CCS_IR_ABLATION_ARMS
                if case_index % 2 == 0
                else tuple(reversed(CCS_IR_ABLATION_ARMS))
            )
            schedule.append({"case_index": case_index, "arm_order": list(order)})
            rows_by_arm: dict[str, dict[str, Any]] = {}
            for arm_id in order:
                case = Case.from_dict(dict(case_payload))
                started = perf_counter_fn()
                row = run_case_fn(
                    case,
                    backends=list(resolved_backends),
                    config=arm_configs[arm_id],
                    save_artifact=False,
                    environment=environment,
                    target_specs=targets,
                    execution_session=sessions.get(arm_id),
                )
                observed_wall_ms = max(0.0, (perf_counter_fn() - started) * 1000.0)
                run = _run_record(case_index, arm_id, row, observed_wall_ms)
                runs.append(run)
                rows_by_arm[arm_id] = run
            pairs.append(_pair_record(case_index, rows_by_arm))
    finally:
        for session in sessions.values():
            session.close()

    payload = {
        "schema_version": CCS_IR_ABLATION_SCHEMA_VERSION,
        "generated_at": utc_now(),
        "corpus_digest": str(corpus.get("corpus_digest", "")),
        "case_count": int(corpus.get("case_count", 0)),
        "backends": list(resolved_backends),
        "environment": environment,
        "targets": targets,
        "arms": {
            arm_id: arm_configs[arm_id].method_arm_manifest
            for arm_id in CCS_IR_ABLATION_ARMS
        },
        "schedule_policy": "alternating_ab_ba",
        "schedule": schedule,
        "schedule_digest": stable_digest("schedule", schedule),
        "runs": runs,
        "pairs": pairs,
        "summary": _ablation_summary(runs, pairs),
        "stopping_rule": "execute_all_frozen_cases_without_outcome_based_early_stopping",
    }
    payload["manifest_digest"] = stable_digest("ccs-ablation", payload)
    return payload


def _arm_config(base: ExperimentConfig, arm_id: str) -> ExperimentConfig:
    payload = base.to_dict()
    payload.update(
        {
            "method_arm": arm_id,
            "method_arm_overrides": {},
            "enable_artifact": False,
            "candidate_recheck_count": 0,
        }
    )
    return ExperimentConfig.from_payload(payload)


def _execution_sessions(
    backends: tuple[str, ...],
    arm_configs: Mapping[str, ExperimentConfig],
    environment: Mapping[str, Any],
    *,
    use_execution_sessions: bool,
    execution_session_factory: Callable[..., BackendExecutionSession],
) -> dict[str, BackendExecutionSession]:
    if not use_execution_sessions:
        return {}
    return {
        arm_id: execution_session_factory(
            list(backends),
            environment=environment,
            adapter_revision=arm_configs[arm_id].method_arm_manifest["digest"],
        )
        for arm_id in CCS_IR_ABLATION_ARMS
    }


def _run_record(
    case_index: int,
    arm_id: str,
    row: Mapping[str, Any],
    observed_wall_ms: float,
) -> dict[str, Any]:
    execution = row.get("execution_profile", {})
    program_ir = row.get("program_ir", {})
    findings = row.get("findings", ())
    outcome_payload = {
        "status": row.get("status", "unknown"),
        "normalized": row.get("normalized", {}),
        "metamorphic": {
            name: {
                "relation": value.get("relation", ""),
                "normalized": value.get("normalized", {}),
            }
            for name, value in row.get("metamorphic", {}).items()
            if isinstance(value, Mapping)
        },
        "findings": [
            {
                "kind": finding.get("kind", ""),
                "signature": finding.get("signature", ""),
                "root_cause": finding.get("root_cause", ""),
                "mismatch_class": finding.get("mismatch_class", ""),
            }
            for finding in findings
            if isinstance(finding, Mapping)
        ],
    }
    return {
        "case_index": case_index,
        "case_id": row.get("case", {}).get("case_id", ""),
        "arm_id": arm_id,
        "ir_mode": program_ir.get("ir_mode", ""),
        "method_arm_digest": row.get("method_arm", {}).get("digest", ""),
        "comparison_block_digest": row.get("experiment_manifest", {}).get(
            "comparison_block_digest", ""
        ),
        "program_ir_digest": program_ir.get("digest", ""),
        "program_ir_resolution_ms": float(program_ir.get("resolution_ms", 0.0) or 0.0),
        "status": row.get("status", "unknown"),
        "finding_count": len(findings),
        "outcome_digest": stable_digest("outcome", outcome_payload),
        "observed_wall_ms": observed_wall_ms,
        "reported_case_ms": float(row.get("duration_ms", 0.0) or 0.0),
        "backend_calls": int(execution.get("backend_calls", 0) or 0),
        "backend_reported_ms": float(
            execution.get("backend_reported_total_ms", 0.0) or 0.0
        ),
        "execution_cache_hits": int(execution.get("execution_cache_hits", 0) or 0),
    }


def _pair_record(
    case_index: int,
    rows_by_arm: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    legacy = rows_by_arm["contract_cartesian"]
    ccs = rows_by_arm["contract_ccs_cartesian"]
    legacy_wall = float(legacy["observed_wall_ms"])
    ccs_wall = float(ccs["observed_wall_ms"])
    return {
        "case_index": case_index,
        "case_id": legacy["case_id"],
        "comparison_block_equal": (
            legacy["comparison_block_digest"] == ccs["comparison_block_digest"]
        ),
        "outcome_equal": legacy["outcome_digest"] == ccs["outcome_digest"],
        "status_equal": legacy["status"] == ccs["status"],
        "finding_count_equal": legacy["finding_count"] == ccs["finding_count"],
        "ccs_to_legacy_wall_ratio": ccs_wall / legacy_wall if legacy_wall > 0.0 else None,
        "ccs_minus_legacy_wall_ms": ccs_wall - legacy_wall,
        "ccs_minus_legacy_backend_calls": (
            int(ccs["backend_calls"]) - int(legacy["backend_calls"])
        ),
    }


def _ablation_summary(
    runs: Sequence[Mapping[str, Any]],
    pairs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    arms: dict[str, dict[str, Any]] = {}
    for arm_id in CCS_IR_ABLATION_ARMS:
        arm_runs = [run for run in runs if run.get("arm_id") == arm_id]
        arms[arm_id] = _single_arm_summary(arm_runs)
    ratios = [
        float(pair["ccs_to_legacy_wall_ratio"])
        for pair in pairs
        if pair.get("ccs_to_legacy_wall_ratio") is not None
    ]
    return {
        "arms": arms,
        "paired_case_count": len(pairs),
        "outcome_equivalence_rate": _rate(pairs, "outcome_equal"),
        "comparison_block_match_rate": _rate(pairs, "comparison_block_equal"),
        "status_equivalence_rate": _rate(pairs, "status_equal"),
        "finding_count_equivalence_rate": _rate(pairs, "finding_count_equal"),
        "median_ccs_to_legacy_wall_ratio": statistics.median(ratios) if ratios else None,
        "mean_ccs_to_legacy_wall_ratio": statistics.fmean(ratios) if ratios else None,
    }


def _single_arm_summary(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    wall = [float(run.get("observed_wall_ms", 0.0) or 0.0) for run in runs]
    total_wall_ms = sum(wall)
    return {
        "case_count": len(runs),
        "total_observed_wall_ms": total_wall_ms,
        "mean_observed_wall_ms": statistics.fmean(wall) if wall else 0.0,
        "median_observed_wall_ms": statistics.median(wall) if wall else 0.0,
        "cases_per_second": (
            len(runs) / (total_wall_ms / 1000.0) if total_wall_ms > 0.0 else 0.0
        ),
        "total_backend_calls": sum(int(run.get("backend_calls", 0)) for run in runs),
        "mean_program_ir_resolution_ms": statistics.fmean(
            float(run.get("program_ir_resolution_ms", 0.0) or 0.0) for run in runs
        )
        if runs
        else 0.0,
    }


def _rate(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    return (
        sum(1 for row in rows if bool(row.get(key))) / len(rows)
        if rows
        else 0.0
    )
