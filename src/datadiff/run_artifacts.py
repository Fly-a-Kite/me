from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

from datadiff.config import ExperimentConfig
from datadiff.dsl import Case
from datadiff.execution_accounting import (
    add_execution_profile_backend_work,
    execution_profile_backend_calls,
    execution_profile_backend_reported_ms,
    execution_profile_cache_hits,
)
from datadiff.run_findings import _artifact_budget_available, _countable_row_findings
from datadiff.run_logging import (
    _merge_process_cpu_profile,
    _merge_stage_profile,
    _merge_wall_time_profile,
    _process_cpu_profile_with_total,
    _stage_profile_with_total,
    _wall_time_profile_with_total,
)

RunLoadedCaseFn = Callable[..., dict[str, Any]]
ReduceCaseFn = Callable[..., Case]


@dataclass(frozen=True, slots=True)
class ArtifactProcessingResult:
    row: dict[str, Any]
    row_stage_profile: dict[str, float]
    row_wall_time_profile: dict[str, float]
    row_process_cpu_profile: dict[str, float]
    countable_row_findings: list[dict[str, Any]]
    scheduler_elapsed_ms: float = 0.0
    scheduler_process_cpu_ms: float = 0.0
    saved_artifact_delta: int = 0


def process_reducer_and_artifacts(
    *,
    row: dict[str, Any],
    case: Case,
    backends: list[str],
    config: ExperimentConfig,
    effective_config: ExperimentConfig,
    backend_instances: dict[str, Any] | None,
    environment: dict[str, str] | None,
    target_specs: list[dict[str, Any]],
    effective_config_payload: dict[str, Any],
    saved_artifacts_count: int,
    generate_mutate_elapsed_ms: float,
    run_loaded_case_fn: RunLoadedCaseFn,
    generate_mutate_process_cpu_ms: float = 0.0,
    reduce_case_fn: ReduceCaseFn | None = None,
    process_cpu_fn: Callable[[], float] = time.process_time,
) -> ArtifactProcessingResult:
    row_stage_profile = _stage_profile_with_total(row.get("stage_profile", {}))
    row_stage_profile["generate_mutate_ms"] += generate_mutate_elapsed_ms
    row_wall_time_profile = _wall_time_profile_with_total(
        row.get("wall_time_profile", {})
    )
    row_wall_time_profile["generate_mutate_ms"] += max(
        0.0,
        float(generate_mutate_elapsed_ms),
    )
    row_process_cpu_profile = _process_cpu_profile_with_total(
        row.get("process_cpu_profile", {})
    )
    row_process_cpu_profile["generate_mutate_ms"] += max(
        0.0,
        float(generate_mutate_process_cpu_ms),
    )
    countable_row_findings = _countable_row_findings(row)
    scheduler_elapsed_ms = 0.0
    scheduler_process_cpu_ms = 0.0

    if countable_row_findings and config.enable_reducer:
        original_row = row
        original_stage_profile = dict(row_stage_profile)
        original_wall_time_profile = dict(row_wall_time_profile)
        original_process_cpu_profile = dict(row_process_cpu_profile)
        original_execution_profile = dict(original_row.get("execution_profile", {}) or {})
        reducer_started = time.perf_counter()
        reducer_process_started = process_cpu_fn()
        reducer = reduce_case_fn or _default_reduce_case
        reduced = reducer(
            case,
            backends=backends,
            config=_reducer_config(effective_config),
            target_kinds=[finding["kind"] for finding in countable_row_findings],
            target_roots=[finding.get("root_cause", "unknown") for finding in countable_row_findings],
            target_suspicious_backends=[
                finding.get("suspicious_backends", []) for finding in countable_row_findings
            ],
        )
        scheduler_elapsed_ms += (time.perf_counter() - reducer_started) * 1000
        scheduler_process_cpu_ms += max(
            0.0,
            (process_cpu_fn() - reducer_process_started) * 1000,
        )
        reduced_row = run_loaded_case_fn(
            reduced,
            backends=backends,
            config=effective_config,
            save_artifact=_artifact_budget_available(config, saved_artifacts_count),
            backend_instances=backend_instances,
            environment=environment,
            target_specs=target_specs,
            config_payload=effective_config_payload,
        )
        reduced_row["original_case"] = case.to_dict()
        reduced_row["reduction"] = {
            "original_rows": len(case.tables[0].rows),
            "reduced_rows": len(reduced.tables[0].rows),
            "original_ops": len(case.program.operations),
            "reduced_ops": len(reduced.program.operations),
        }
        if "backend_sampling" in original_row:
            reduced_row["backend_sampling"] = original_row["backend_sampling"]
        reduced_execution_profile = dict(reduced_row.get("execution_profile", {}) or {})
        if original_execution_profile or reduced_execution_profile:
            reduced_row["execution_profile"] = add_execution_profile_backend_work(
                reduced_execution_profile,
                label="original_candidate",
                additional_ms=execution_profile_backend_reported_ms(original_execution_profile),
                additional_calls=execution_profile_backend_calls(original_execution_profile),
                additional_cache_hits=execution_profile_cache_hits(original_execution_profile),
            )
        row = reduced_row
        countable_row_findings = _countable_row_findings(row)
        reduced_stage_profile = _stage_profile_with_total(row.get("stage_profile", {}))
        row_stage_profile = _stage_profile_with_total(
            _merge_stage_profile(original_stage_profile, reduced_stage_profile)
        )
        row_wall_time_profile = _wall_time_profile_with_total(
            _merge_wall_time_profile(
                original_wall_time_profile,
                reduced_row.get("wall_time_profile", {}),
            )
        )
        row_process_cpu_profile = _process_cpu_profile_with_total(
            _merge_process_cpu_profile(
                original_process_cpu_profile,
                reduced_row.get("process_cpu_profile", {}),
            )
        )
        row["stage_profile"] = row_stage_profile
        row["wall_time_profile"] = row_wall_time_profile
        row["process_cpu_profile"] = row_process_cpu_profile
        row["duration_ms"] = row_stage_profile["total_case_wall_ms"]

    saved_artifact_delta = 0
    if countable_row_findings and row.get("bug_dir"):
        saved_artifact_delta = 1
        row["artifact_saved"] = True
    elif countable_row_findings and config.enable_artifact:
        row["artifact_saved"] = False
        row["artifact_skipped_reason"] = "artifact_limit_reached"

    return ArtifactProcessingResult(
        row=row,
        row_stage_profile=row_stage_profile,
        row_wall_time_profile=_wall_time_profile_with_total(row_wall_time_profile),
        row_process_cpu_profile=_process_cpu_profile_with_total(
            row_process_cpu_profile
        ),
        countable_row_findings=countable_row_findings,
        scheduler_elapsed_ms=scheduler_elapsed_ms,
        scheduler_process_cpu_ms=scheduler_process_cpu_ms,
        saved_artifact_delta=saved_artifact_delta,
    )


def _reducer_config(effective_config: ExperimentConfig) -> ExperimentConfig:
    return ExperimentConfig(
        method_arm=effective_config.method_arm,
        method_arm_overrides=dict(effective_config.method_arm_overrides),
        evidence_tier="native_reproduction",
        enable_type_aware_generation=effective_config.enable_type_aware_generation,
        enable_normalizer=effective_config.enable_normalizer,
        enable_differential_oracle=effective_config.enable_differential_oracle,
        enable_metamorphic_oracle=effective_config.enable_metamorphic_oracle,
        enable_feedback=False,
        enable_reducer=False,
        enable_artifact=False,
        oracle_mode=effective_config.oracle_mode,
        generator_profile=effective_config.generator_profile,
        metamorphic_variant_limit=effective_config.metamorphic_variant_limit,
        target_version=effective_config.target_version,
        fixed_version=effective_config.fixed_version,
    )


def _default_reduce_case(*args: Any, **kwargs: Any) -> Case:
    from datadiff.reducer import reduce_case

    return reduce_case(*args, **kwargs)
