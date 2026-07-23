from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence

from datadiff.research_controls.rlcmf_shadow.rlcmf_manifest import ValidatedRLCMFManifest


RLCMF_PAIRED_ANALYSIS_SCHEMA = "rlcmf-paired-cost-analysis-v1"


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def _nonnegative(value: Any, *, field: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be finite and non-negative") from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError(f"{field} must be finite and non-negative")
    return parsed


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("cannot take a quantile of an empty sequence")
    position = probability * (len(sorted_values) - 1)
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    if lower_index == upper_index:
        return float(sorted_values[lower_index])
    fraction = position - lower_index
    return float(
        sorted_values[lower_index] * (1.0 - fraction)
        + sorted_values[upper_index] * fraction
    )


def _block_start(
    *,
    seed_sha256: str,
    replicate: int,
    block: int,
    case_count: int,
) -> int:
    material = {
        "schema_version": "rlcmf-bootstrap-block-draw-v1",
        "seed_sha256": seed_sha256,
        "replicate": replicate,
        "block": block,
    }
    draw = int.from_bytes(
        hashlib.sha256(_canonical_json_bytes(material)).digest()[:8],
        "big",
    )
    return draw % case_count


def circular_moving_block_indices(
    *,
    seed_sha256: str,
    replicate: int,
    case_count: int,
    block_length: int,
) -> tuple[int, ...]:
    if len(seed_sha256) != 64:
        raise ValueError("bootstrap seed must be a SHA-256 digest")
    if case_count <= 0:
        raise ValueError("case_count must be positive")
    if not 1 <= block_length <= case_count:
        raise ValueError("block_length must be within the case frame")
    indexes: list[int] = []
    block = 0
    while len(indexes) < case_count:
        start = _block_start(
            seed_sha256=seed_sha256,
            replicate=replicate,
            block=block,
            case_count=case_count,
        )
        indexes.extend(
            (start + offset) % case_count for offset in range(block_length)
        )
        block += 1
    return tuple(indexes[:case_count])


def _metric_pair(row: Mapping[str, Any], metric: str) -> tuple[float, float]:
    adaptive = row.get("adaptive")
    reference = row.get("full_reference_accounting")
    if not isinstance(adaptive, Mapping) or not isinstance(reference, Mapping):
        raise ValueError("every replay row requires paired accounting")
    adaptive_fields = {
        "backend_calls": "combined_backend_calls",
        "backend_reported_ms": "combined_backend_reported_ms",
        "wall_ms": "combined_wall_ms",
        "process_cpu_ms": "combined_process_cpu_ms",
    }
    reference_fields = {
        "backend_calls": "backend_calls",
        "backend_reported_ms": "backend_reported_ms",
        "wall_ms": "wall_ms",
        "process_cpu_ms": "process_cpu_ms",
    }
    try:
        adaptive_field = adaptive_fields[metric]
        reference_field = reference_fields[metric]
    except KeyError as exc:
        raise ValueError(f"unsupported paired cost metric: {metric}") from exc
    return (
        _nonnegative(adaptive.get(adaptive_field), field=f"adaptive.{metric}"),
        _nonnegative(reference.get(reference_field), field=f"reference.{metric}"),
    )


def _bootstrap_seed(manifest_sha256: str, analysis_id: str, metric: str) -> str:
    return hashlib.sha256(
        _canonical_json_bytes(
            {
                "schema_version": "rlcmf-bootstrap-seed-v1",
                "manifest_sha256": manifest_sha256,
                "analysis_id": analysis_id,
                "metric": metric,
            }
        )
    ).hexdigest()


def _analyze_metric(
    *,
    metric: str,
    pairs: Sequence[tuple[float, float]],
    manifest_sha256: str,
    analysis_id: str,
    confidence_level: float,
    replicates: int,
    block_length: int,
) -> dict[str, Any]:
    case_count = len(pairs)
    adaptive_total = math.fsum(pair[0] for pair in pairs)
    reference_total = math.fsum(pair[1] for pair in pairs)
    if reference_total <= 0:
        raise ValueError(f"reference total for {metric} must be positive")
    point_ratio = adaptive_total / reference_total
    point_difference = math.fsum(
        adaptive - reference for adaptive, reference in pairs
    ) / case_count
    seed_sha256 = _bootstrap_seed(manifest_sha256, analysis_id, metric)
    bootstrap_ratios: list[float] = []
    bootstrap_differences: list[float] = []
    for replicate in range(replicates):
        indexes = circular_moving_block_indices(
            seed_sha256=seed_sha256,
            replicate=replicate,
            case_count=case_count,
            block_length=block_length,
        )
        adaptive_sum = math.fsum(pairs[index][0] for index in indexes)
        reference_sum = math.fsum(pairs[index][1] for index in indexes)
        if reference_sum <= 0:
            raise ValueError(
                f"bootstrap reference total for {metric} must be positive"
            )
        bootstrap_ratios.append(adaptive_sum / reference_sum)
        bootstrap_differences.append(
            math.fsum(
                pairs[index][0] - pairs[index][1] for index in indexes
            )
            / case_count
        )
    bootstrap_ratios.sort()
    bootstrap_differences.sort()
    tail = (1.0 - confidence_level) / 2.0
    ratio_lower = _quantile(bootstrap_ratios, tail)
    ratio_upper = _quantile(bootstrap_ratios, 1.0 - tail)
    difference_lower = _quantile(bootstrap_differences, tail)
    difference_upper = _quantile(bootstrap_differences, 1.0 - tail)
    if ratio_upper < 1.0:
        state = "optimized"
    elif ratio_lower > 1.0:
        state = "regressed"
    else:
        state = "inconclusive"
    return {
        "metric": metric,
        "case_count": case_count,
        "adaptive_total": adaptive_total,
        "reference_total": reference_total,
        "ratio_of_totals": point_ratio,
        "relative_change": point_ratio - 1.0,
        "reference_over_adaptive_speedup": (
            reference_total / adaptive_total if adaptive_total > 0 else None
        ),
        "paired_mean_difference": point_difference,
        "ratio_confidence_interval": {
            "lower": ratio_lower,
            "upper": ratio_upper,
        },
        "paired_mean_difference_confidence_interval": {
            "lower": difference_lower,
            "upper": difference_upper,
        },
        "bootstrap_seed_sha256": seed_sha256,
        "decision": state,
    }


def analyze_rlcmf_paired_replay(
    manifest: ValidatedRLCMFManifest,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    if str(result.get("manifest_sha256", "")) != manifest.sha256:
        raise ValueError("paired replay result manifest digest mismatch")
    budget = manifest.payload.get("budget")
    if not isinstance(budget, Mapping):
        raise ValueError("manifest budget is missing")
    plan = budget.get("paired_analysis")
    if not isinstance(plan, Mapping):
        raise ValueError("manifest lacks a preregistered paired analysis")
    case_rows = result.get("case_results")
    if not isinstance(case_rows, list) or not case_rows:
        raise ValueError("paired replay result has no case rows")
    indexes = [int(row.get("trace_index", -1)) for row in case_rows]
    if indexes != list(range(len(case_rows))):
        raise ValueError("paired replay trace indexes must be contiguous")
    manifest_schedule = manifest.payload.get("seed_plan", {}).get("schedule", [])
    if len(manifest_schedule) != len(case_rows):
        raise ValueError("paired replay result does not cover the frozen schedule")
    trace = result.get("trace")
    if not isinstance(trace, Mapping):
        raise ValueError("paired replay result trace metadata is missing")
    expected_trace_sha = manifest.payload["fidelity"]["reference_policy"].get(
        "trace_sha256"
    )
    if str(trace.get("trace_sha256", "")) != str(expected_trace_sha):
        raise ValueError("paired replay result trace digest mismatch")

    confidence_level = float(plan["confidence_level"])
    replicates = int(plan["bootstrap_replicates"])
    block_length = int(plan["block_length"])
    analysis_id = str(plan["analysis_id"])
    metric_results: dict[str, dict[str, Any]] = {}
    paired_values: dict[str, list[dict[str, float]]] = {}
    for metric in plan["metrics"]:
        pairs = [_metric_pair(row, str(metric)) for row in case_rows]
        metric_results[str(metric)] = _analyze_metric(
            metric=str(metric),
            pairs=pairs,
            manifest_sha256=manifest.sha256,
            analysis_id=analysis_id,
            confidence_level=confidence_level,
            replicates=replicates,
            block_length=block_length,
        )
        paired_values[str(metric)] = [
            {"adaptive": adaptive, "reference": reference}
            for adaptive, reference in pairs
        ]

    primary_metric = str(plan["primary_cost_metric"])
    primary = metric_results[primary_metric]
    metrics_payload = result.get("metrics", {})
    return {
        "schema_version": RLCMF_PAIRED_ANALYSIS_SCHEMA,
        "manifest_id": manifest.payload["manifest_id"],
        "manifest_sha256": manifest.sha256,
        "analysis_plan": dict(plan),
        "trace": {
            "case_count": len(case_rows),
            "trace_sha256": trace["trace_sha256"],
            "paired_values_sha256": hashlib.sha256(
                _canonical_json_bytes(paired_values)
            ).hexdigest(),
        },
        "metrics": metric_results,
        "primary_decision": {
            "metric": primary_metric,
            "state": primary["decision"],
            "rule": plan["decision_rule"],
            "promotion_effect": "diagnostic_only",
        },
        "safety": {
            "final_state": result.get("safety", {}).get("final_state"),
            "controller_log_integrity": result.get("safety", {}).get(
                "controller_log_integrity"
            ),
        },
        "candidate_evidence": {
            "candidate_family_recall": metrics_payload.get(
                "candidate_family_recall"
            ),
            "candidate_root_recall": metrics_payload.get("candidate_root_recall"),
            "independently_confirmed_unique_real_roots_per_cpu_hour": (
                metrics_payload.get(
                    "independently_confirmed_unique_real_roots_per_cpu_hour"
                )
            ),
        },
        "limitations": [
            "moving-block bootstrap quantifies observed paired cost stability; it does not validate root recall",
            "backend-reported milliseconds remain a separate work proxy and are not substituted for process CPU",
            "candidate families and roots are not independently confirmed real bugs",
        ],
    }
