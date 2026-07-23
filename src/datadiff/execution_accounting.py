from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def backend_reported_ms_from_raw(raw_results: Mapping[str, Any] | None) -> float:
    total = 0.0
    for result in (raw_results or {}).values():
        if not isinstance(result, Mapping):
            continue
        if bool(result.get("execution_cache_hit", False)):
            continue
        total += _nonnegative_float(result.get("duration_ms", 0.0))
    return total


def backend_calls_from_raw(raw_results: Mapping[str, Any] | None) -> int:
    return sum(
        1
        for result in (raw_results or {}).values()
        if isinstance(result, Mapping) and not bool(result.get("execution_cache_hit", False))
    )


def execution_cache_hits_from_raw(raw_results: Mapping[str, Any] | None) -> int:
    return sum(
        1
        for result in (raw_results or {}).values()
        if isinstance(result, Mapping) and bool(result.get("execution_cache_hit", False))
    )


def execution_profile_backend_reported_ms(profile: Mapping[str, Any] | None) -> float:
    if not isinstance(profile, Mapping):
        return 0.0
    return _nonnegative_float(
        profile.get(
            "combined_backend_reported_total_ms",
            profile.get("backend_reported_total_ms", 0.0),
        )
    )


def execution_profile_backend_calls(profile: Mapping[str, Any] | None) -> int:
    if not isinstance(profile, Mapping):
        return 0
    return _nonnegative_int(
        profile.get(
            "combined_backend_calls",
            profile.get("backend_calls", 0),
        )
    )


def execution_profile_cache_hits(profile: Mapping[str, Any] | None) -> int:
    if not isinstance(profile, Mapping):
        return 0
    return _nonnegative_int(
        profile.get(
            "combined_execution_cache_hits",
            profile.get("execution_cache_hits", 0),
        )
    )


def add_execution_profile_backend_work(
    profile: Mapping[str, Any] | None,
    *,
    label: str,
    additional_ms: float,
    additional_calls: int = 0,
    additional_cache_hits: int = 0,
) -> dict[str, Any]:
    out = dict(profile or {})
    base_ms = execution_profile_backend_reported_ms(out)
    base_calls = execution_profile_backend_calls(out)
    base_cache_hits = execution_profile_cache_hits(out)
    added_ms = _nonnegative_float(additional_ms)
    added_calls = _nonnegative_int(additional_calls)
    added_cache_hits = _nonnegative_int(additional_cache_hits)
    normalized_label = str(label or "additional").strip().replace("-", "_") or "additional"
    out[f"{normalized_label}_backend_reported_total_ms"] = added_ms
    out[f"{normalized_label}_backend_calls"] = added_calls
    out[f"{normalized_label}_execution_cache_hits"] = added_cache_hits
    out["combined_backend_reported_total_ms"] = base_ms + added_ms
    out["combined_backend_calls"] = base_calls + added_calls
    out["combined_execution_cache_hits"] = base_cache_hits + added_cache_hits
    return out


def _nonnegative_float(value: Any) -> float:
    try:
        return max(0.0, float(value or 0.0))
    except (TypeError, ValueError):
        return 0.0


def _nonnegative_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0
