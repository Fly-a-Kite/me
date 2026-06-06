from __future__ import annotations

from typing import Any, Callable

from datadiff.config import ExperimentConfig
from datadiff.dsl import Case
from datadiff.oracle import Finding
from datadiff.run_findings import (
    _finding_recheck_key,
    _format_recheck_key,
    _mark_finding_non_reproducible,
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
    recheck_config = ExperimentConfig.from_payload(recheck_config_data)
    original_keys = {_finding_recheck_key(finding.to_dict()) for finding in findings}
    reproduced_keys: set[tuple[str, str, tuple[str, ...], str]] | None = None
    attempt_summaries: list[dict[str, Any]] = []
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
        reproduced_keys = current_reproduced if reproduced_keys is None else reproduced_keys & current_reproduced
        attempt_summaries.append(
            {
                "attempt": attempt + 1,
                "finding_count": len(row.get("findings", []) or []),
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
        "reproduced_keys": [_format_recheck_key(key) for key in sorted(reproduced_keys)],
        "non_reproduced_keys": [_format_recheck_key(key) for key in sorted(non_reproduced_keys)],
    }
