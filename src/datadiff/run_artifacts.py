from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

from datadiff.config import ExperimentConfig
from datadiff.dsl import Case
from datadiff.run_findings import _artifact_budget_available, _countable_row_findings
from datadiff.run_logging import _stage_profile_with_total

RunLoadedCaseFn = Callable[..., dict[str, Any]]
ReduceCaseFn = Callable[..., Case]


@dataclass(frozen=True, slots=True)
class ArtifactProcessingResult:
    row: dict[str, Any]
    row_stage_profile: dict[str, float]
    countable_row_findings: list[dict[str, Any]]
    scheduler_elapsed_ms: float = 0.0
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
    reduce_case_fn: ReduceCaseFn | None = None,
) -> ArtifactProcessingResult:
    row_stage_profile = _stage_profile_with_total(row.get("stage_profile", {}))
    row_stage_profile["generate_mutate_ms"] += generate_mutate_elapsed_ms
    countable_row_findings = _countable_row_findings(row)
    scheduler_elapsed_ms = 0.0

    if countable_row_findings and config.enable_reducer:
        reducer_started = time.perf_counter()
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
        row = reduced_row
        countable_row_findings = _countable_row_findings(row)
        row_stage_profile = _stage_profile_with_total(row.get("stage_profile", {}))
        row_stage_profile["generate_mutate_ms"] += generate_mutate_elapsed_ms

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
        countable_row_findings=countable_row_findings,
        scheduler_elapsed_ms=scheduler_elapsed_ms,
        saved_artifact_delta=saved_artifact_delta,
    )


def _reducer_config(effective_config: ExperimentConfig) -> ExperimentConfig:
    return ExperimentConfig(
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
