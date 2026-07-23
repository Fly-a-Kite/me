from __future__ import annotations

from typing import Any, Callable

from datadiff.config import ExperimentConfig
from datadiff.dsl import Case
from datadiff.execution_accounting import (
    execution_profile_backend_calls,
    execution_profile_backend_reported_ms,
)
from datadiff.oracle import Finding
from datadiff.run_findings import (
    _finding_recheck_key,
    _format_recheck_key,
    _mark_finding_non_reproducible,
)
from datadiff.run_logging import (
    STAGE_PROFILE_KEYS,
    _empty_process_cpu_profile,
    _empty_stage_profile,
    _empty_wall_time_profile,
    _merge_process_cpu_profile,
    _merge_wall_time_profile,
    _process_cpu_profile_with_total,
    _stage_profile_with_total,
    _wall_time_profile_with_total,
)

RunLoadedCaseFn = Callable[..., dict[str, Any]]


def candidate_recheck_impl(
    case: Case,
    backends: list[str],
    config: ExperimentConfig,
    findings: list[Finding],
    *,
    run_loaded_case_fn: RunLoadedCaseFn,
) -> dict[str, Any]:
    attempts = max(0, int(getattr(config, "candidate_recheck_count", 0)))
    if attempts <= 0 or not findings:
        return {"enabled": False, "attempts": 0, "reproduced_keys": [], "non_reproduced_keys": []}

    recheck_config_data = config.to_dict()
    recheck_config_data["candidate_recheck_count"] = 0
    recheck_config_data["enable_artifact"] = False
    recheck_config_data["evidence_tier"] = "fresh_confirmation"
    recheck_config = ExperimentConfig.from_payload(recheck_config_data)
    original_keys = {_finding_recheck_key(finding.to_dict()) for finding in findings}
    reproduced_keys: set[tuple[str, str, tuple[str, ...], str]] | None = None
    attempt_summaries: list[dict[str, Any]] = []
    backend_reported_total_ms = 0.0
    backend_calls = 0
    stage_profile_totals = _empty_stage_profile()
    wall_time_profile_totals = _empty_wall_time_profile()
    process_cpu_profile_totals = _empty_process_cpu_profile()
    for attempt in range(attempts):
        row = run_loaded_case_fn(
            case,
            backends=backends,
            config=recheck_config,
            save_artifact=False,
            backend_instances=None,
            target_specs=[],
        )
        keys = {_finding_recheck_key(finding) for finding in row.get("findings", [])}
        current_reproduced = original_keys & keys
        attempt_backend_reported_ms = execution_profile_backend_reported_ms(
            row.get("execution_profile", {})
        )
        backend_reported_total_ms += attempt_backend_reported_ms
        attempt_backend_calls = execution_profile_backend_calls(
            row.get("execution_profile", {})
        )
        backend_calls += attempt_backend_calls
        attempt_stage_profile = row.get("stage_profile", {})
        if isinstance(attempt_stage_profile, dict):
            for key in STAGE_PROFILE_KEYS:
                if key == "total_case_wall_ms":
                    continue
                stage_profile_totals[key] += max(
                    0.0,
                    float(attempt_stage_profile.get(key, 0.0) or 0.0),
                )
        attempt_wall_time_profile = _wall_time_profile_with_total(
            row.get("wall_time_profile", {})
        )
        attempt_process_cpu_profile = _process_cpu_profile_with_total(
            row.get("process_cpu_profile", {})
        )
        wall_time_profile_totals = _merge_wall_time_profile(
            wall_time_profile_totals,
            attempt_wall_time_profile,
        )
        process_cpu_profile_totals = _merge_process_cpu_profile(
            process_cpu_profile_totals,
            attempt_process_cpu_profile,
        )
        reproduced_keys = current_reproduced if reproduced_keys is None else reproduced_keys & current_reproduced
        attempt_summaries.append(
            {
                "attempt": attempt + 1,
                "finding_count": len(row.get("findings", []) or []),
                "backend_reported_total_ms": attempt_backend_reported_ms,
                "backend_calls": attempt_backend_calls,
                "stage_profile": _stage_profile_with_total(
                    attempt_stage_profile if isinstance(attempt_stage_profile, dict) else {}
                ),
                "wall_time_profile": attempt_wall_time_profile,
                "process_cpu_profile": attempt_process_cpu_profile,
                "physical_plans": {
                    backend: raw["physical_plan"]
                    for backend, raw in (row.get("raw_results", {}) or {}).items()
                    if isinstance(raw, dict)
                    and isinstance(raw.get("physical_plan"), dict)
                },
                "reproduced_keys": [_format_recheck_key(key) for key in sorted(current_reproduced)],
            }
        )
    reproduced_keys = reproduced_keys or set()
    non_reproduced_keys = original_keys - reproduced_keys
    for finding in findings:
        key = _finding_recheck_key(finding.to_dict())
        if key in non_reproduced_keys:
            _mark_finding_non_reproducible(finding, attempts)
    return {
        "enabled": True,
        "attempts": attempts,
        "attempt_summaries": attempt_summaries,
        "backend_reported_total_ms": backend_reported_total_ms,
        "backend_calls": backend_calls,
        "stage_profile_totals": _stage_profile_with_total(stage_profile_totals),
        "wall_time_profile_totals": _wall_time_profile_with_total(
            wall_time_profile_totals
        ),
        "process_cpu_profile_totals": _process_cpu_profile_with_total(
            process_cpu_profile_totals
        ),
        "reproduced_keys": [_format_recheck_key(key) for key in sorted(reproduced_keys)],
        "non_reproduced_keys": [_format_recheck_key(key) for key in sorted(non_reproduced_keys)],
    }
