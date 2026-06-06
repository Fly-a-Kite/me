from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping

ICSE_EXPERIMENT_QUALITY_SCHEMA_VERSION = "icse-experiment-quality-v1"


@dataclass(frozen=True, slots=True)
class ICSEExperimentQualityThresholds:
    target_real_bug_families: int = 5
    target_confirmed_bug_families: int = 1
    target_reproduced_bug_families: int = 1
    target_throughput_cases_s: float = 5.0
    target_coverage_axes: int = 8
    target_live_suites: int = 4
    target_first_bug_elapsed_s: float = 6.0 * 60.0 * 60.0
    target_discovery_auc: float = 0.2
    target_reproducer_coverage: float = 1.0


DEFAULT_ICSE_EXPERIMENT_QUALITY_THRESHOLDS = ICSEExperimentQualityThresholds()

_DIMENSION_WEIGHTS = {
    "real_bug_yield": 0.30,
    "throughput": 0.20,
    "coverage": 0.20,
    "speed": 0.15,
    "reproducibility": 0.15,
}


def score_methodology_report(
    report: Mapping[str, Any],
    *,
    thresholds: ICSEExperimentQualityThresholds | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    threshold_obj = _thresholds(thresholds)
    discovery = _mapping(report.get("bug_discovery"))
    efficiency = _mapping(report.get("efficiency"))
    coverage = _mapping(report.get("coverage"))
    reproducibility = _mapping(report.get("reproducibility"))
    pipeline = _mapping(report.get("candidate_pipeline"))

    candidate_families = _count_mapping_or_value(
        discovery.get("candidate_bug_families"),
        discovery.get("candidate_bug_family_count"),
    )
    reproduced = _non_negative_int(pipeline.get("reproduced_count"))
    candidate_bug_verdicts = _non_negative_int(pipeline.get("candidate_bug_verdict_count"))
    bug_evidence = max(candidate_families, reproduced, candidate_bug_verdicts)
    confirmed_like = min(reproduced, candidate_bug_verdicts) if candidate_bug_verdicts else reproduced

    coverage_axes = _coverage_axis_count(
        coverage,
        axis_keys=(
            "target_suites",
            "presets",
            "matrix_ids",
            "comparison_groups",
            "variant_ids",
            "scope_kinds",
            "oracle_profiles",
            "rq_tags",
            "analysis_tags",
            "semantic_focus_families",
            "semantic_focus_signals",
            "evidence_modes",
        ),
    )
    first_elapsed = _first_candidate_elapsed_s(discovery)
    auc = _clean_float(discovery.get("avg_candidate_bug_discovery_auc"))
    run_log_total = _non_negative_int(reproducibility.get("run_logs_total"))
    run_log_existing = _non_negative_int(reproducibility.get("run_logs_exist"))
    run_log_coverage = run_log_existing / run_log_total if run_log_total else 0.0
    artifact_reproducer_coverage = _ratio(reproducibility.get("artifact_reproducer_coverage"))
    issue_bundle = _mapping(reproducibility.get("issue_bundle"))
    issue_bundle_clean = 1.0 if issue_bundle.get("clean_execution") is True else 0.0
    recheck_pass_rate = _ratio(pipeline.get("recheck_pass_rate"))
    reproducer_coverage = max(
        artifact_reproducer_coverage,
        issue_bundle_clean,
        recheck_pass_rate,
        1.0 if reproduced > 0 else 0.0,
    )

    dimensions = _score_dimensions(
        {
            "real_bug_yield": _bug_yield_dimension(
                observed=bug_evidence,
                confirmed=confirmed_like,
                target=threshold_obj.target_real_bug_families,
                confirmed_target=threshold_obj.target_reproduced_bug_families,
                evidence={
                    "candidate_bug_families": candidate_families,
                    "pipeline_reproduced_count": reproduced,
                    "candidate_bug_verdict_count": candidate_bug_verdicts,
                },
            ),
            "throughput": _ratio_dimension(
                observed=_clean_float(efficiency.get("cases_per_s")),
                target=threshold_obj.target_throughput_cases_s,
                evidence={
                    "cases": _non_negative_int(efficiency.get("cases")),
                    "elapsed_s": _clean_float(efficiency.get("elapsed_s")),
                },
            ),
            "coverage": _ratio_dimension(
                observed=coverage_axes,
                target=threshold_obj.target_coverage_axes,
                evidence={
                    "target_suite_count": _non_negative_int(coverage.get("target_suite_count")),
                    "preset_count": _non_negative_int(coverage.get("preset_count")),
                    "run_count": _non_negative_int(coverage.get("run_count")),
                    "axis_count": coverage_axes,
                },
            ),
            "speed": _speed_dimension(
                first_elapsed_s=first_elapsed,
                target_first_elapsed_s=threshold_obj.target_first_bug_elapsed_s,
                auc=auc,
                target_auc=threshold_obj.target_discovery_auc,
            ),
            "reproducibility": _ratio_dimension(
                observed=max(reproducer_coverage, run_log_coverage),
                target=threshold_obj.target_reproducer_coverage,
                evidence={
                    "run_log_coverage": run_log_coverage,
                    "artifact_reproducer_coverage": artifact_reproducer_coverage,
                    "issue_bundle_clean_execution": bool(issue_bundle.get("clean_execution", False)),
                    "candidate_pipeline_recheck_pass_rate": recheck_pass_rate,
                },
            ),
        }
    )
    return _build_quality_report(
        source="methodology_report",
        dimensions=dimensions,
        thresholds=threshold_obj,
    )


def score_final_readiness_summary(
    summary: Mapping[str, Any],
    *,
    thresholds: ICSEExperimentQualityThresholds | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    threshold_obj = _thresholds(thresholds)
    runtime = _mapping(summary.get("runtime_efficiency"))
    responsiveness = _mapping(summary.get("discovery_responsiveness"))
    matrix = _mapping(summary.get("final_matrix_coverage"))
    closed_loop_state = _mapping(summary.get("closed_loop_state_persistence"))

    rewardable = _count_mapping_or_value(summary.get("rewardable_live_candidate_families"))
    confirmed = _count_mapping_or_value(summary.get("confirmed_live_candidate_families"))
    external_confirmed = _count_mapping_or_value(summary.get("external_confirmed_live_candidate_families"))
    live_suites = len(_list(summary.get("live_suites")))
    missing_matrix = len(_list(matrix.get("missing_matrix_ids")))
    observed_matrix = len(_list(matrix.get("observed_matrix_ids")))
    semantic_focus_count = len(_list(summary.get("semantic_focus_families"))) + len(
        _list(summary.get("semantic_focus_signals"))
    )
    paper_required = _non_negative_int(summary.get("paper_run_journal_required_runs"))
    paper_covered = _non_negative_int(summary.get("paper_run_journal_covered_runs"))
    paper_coverage = paper_covered / paper_required if paper_required else 0.0
    state_required = _non_negative_int(closed_loop_state.get("required_run_count"))
    state_persisted = _non_negative_int(closed_loop_state.get("persisted_run_count"))
    state_coverage = state_persisted / state_required if state_required else 0.0

    dimensions = _score_dimensions(
        {
            "real_bug_yield": _bug_yield_dimension(
                observed=max(rewardable, confirmed, external_confirmed),
                confirmed=max(confirmed, external_confirmed),
                target=threshold_obj.target_real_bug_families,
                confirmed_target=threshold_obj.target_confirmed_bug_families,
                evidence={
                    "rewardable_live_candidate_families": rewardable,
                    "confirmed_live_candidate_families": confirmed,
                    "external_confirmed_live_candidate_families": external_confirmed,
                },
            ),
            "throughput": _ratio_dimension(
                observed=max(
                    _clean_float(runtime.get("avg_throughput_cases_s")),
                    _clean_float(runtime.get("min_throughput_cases_s")),
                ),
                target=threshold_obj.target_throughput_cases_s,
                evidence={
                    "min_throughput_cases_s": _clean_float(runtime.get("min_throughput_cases_s")),
                    "avg_throughput_cases_s": _clean_float(runtime.get("avg_throughput_cases_s")),
                    "issues": _list(runtime.get("issues")),
                },
            ),
            "coverage": _ratio_dimension(
                observed=observed_matrix + live_suites + semantic_focus_count,
                target=max(threshold_obj.target_coverage_axes, threshold_obj.target_live_suites),
                passed=(missing_matrix == 0 and live_suites >= threshold_obj.target_live_suites),
                evidence={
                    "observed_matrix_ids": observed_matrix,
                    "missing_matrix_ids": missing_matrix,
                    "live_suites": live_suites,
                    "semantic_focus_count": semantic_focus_count,
                },
            ),
            "speed": _speed_dimension(
                first_elapsed_s=_optional_float(responsiveness.get("best_first_candidate_elapsed_s")),
                target_first_elapsed_s=threshold_obj.target_first_bug_elapsed_s,
                auc=_clean_float(responsiveness.get("avg_candidate_bug_discovery_auc")),
                target_auc=threshold_obj.target_discovery_auc,
                evidence={
                    "observed_run_count": _non_negative_int(responsiveness.get("observed_run_count")),
                    "best_first_candidate_case_index": responsiveness.get("best_first_candidate_case_index"),
                    "best_run_label": responsiveness.get("best_run_label", ""),
                },
            ),
            "reproducibility": _ratio_dimension(
                observed=max(paper_coverage, state_coverage),
                target=threshold_obj.target_reproducer_coverage,
                evidence={
                    "paper_run_journal_coverage": paper_coverage,
                    "closed_loop_state_coverage": state_coverage,
                    "paper_run_journal_missing_run_count": _non_negative_int(
                        summary.get("paper_run_journal_missing_run_count")
                    ),
                },
            ),
        }
    )
    return _build_quality_report(
        source="final_readiness_summary",
        dimensions=dimensions,
        thresholds=threshold_obj,
    )


def _build_quality_report(
    *,
    source: str,
    dimensions: dict[str, dict[str, Any]],
    thresholds: ICSEExperimentQualityThresholds,
) -> dict[str, Any]:
    overall = sum(
        _DIMENSION_WEIGHTS[name] * _clean_float(dimension.get("score"))
        for name, dimension in dimensions.items()
    )
    overall = max(0.0, min(100.0, overall))
    priorities = sorted(
        (
            {
                "dimension": name,
                "score": round(_clean_float(dimension.get("score")), 3),
                "passed": bool(dimension.get("passed", False)),
                "reason": str(dimension.get("reason", "")),
            }
            for name, dimension in dimensions.items()
        ),
        key=lambda item: (item["passed"], item["score"], item["dimension"]),
    )
    ready = (
        overall >= 80.0
        and all(bool(dimension.get("passed", False)) for dimension in dimensions.values())
    )
    grade = _grade(overall)
    return {
        "schema_version": ICSE_EXPERIMENT_QUALITY_SCHEMA_VERSION,
        "source": source,
        "overall_score": round(overall, 3),
        "grade": grade,
        "ready_for_icse_claim": ready,
        "dimensions": dimensions,
        "optimization_priorities": priorities,
        "thresholds": asdict(thresholds),
        "methodology_claim": _methodology_claim(overall, grade, ready),
    }


def _score_dimensions(dimensions: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    scored = {}
    for name, dimension in dimensions.items():
        item = dict(dimension)
        item["weight"] = _DIMENSION_WEIGHTS[name]
        item["score"] = round(max(0.0, min(100.0, _clean_float(item.get("score")))), 3)
        scored[name] = item
    return scored


def _bug_yield_dimension(
    *,
    observed: int,
    confirmed: int,
    target: int,
    confirmed_target: int,
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    target = max(1, int(target))
    confirmed_target = max(0, int(confirmed_target))
    observed_score = min(1.0, _non_negative_int(observed) / target)
    confirmed_score = 1.0 if confirmed_target == 0 else min(1.0, _non_negative_int(confirmed) / confirmed_target)
    score = 100.0 * (0.75 * observed_score + 0.25 * confirmed_score)
    passed = observed >= target and (confirmed_target == 0 or confirmed >= confirmed_target)
    return {
        "score": score,
        "passed": bool(passed),
        "observed": _non_negative_int(observed),
        "target": target,
        "confirmed_observed": _non_negative_int(confirmed),
        "confirmed_target": confirmed_target,
        "evidence": dict(evidence),
        "reason": "needs more rewardable/reproduced bug families" if not passed else "target bug yield reached",
    }


def _ratio_dimension(
    *,
    observed: float,
    target: float,
    evidence: Mapping[str, Any],
    passed: bool | None = None,
) -> dict[str, Any]:
    target = max(0.0, _clean_float(target))
    observed = max(0.0, _clean_float(observed))
    ratio = 1.0 if target == 0.0 else min(1.0, observed / target)
    default_passed = observed >= target if target > 0.0 else observed > 0.0
    final_passed = default_passed if passed is None else bool(passed and default_passed)
    return {
        "score": 100.0 * ratio,
        "passed": bool(final_passed),
        "observed": observed,
        "target": target,
        "evidence": dict(evidence),
        "reason": "target reached" if final_passed else "below target",
    }


def _speed_dimension(
    *,
    first_elapsed_s: float | None,
    target_first_elapsed_s: float,
    auc: float,
    target_auc: float,
    evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    target_elapsed = max(0.001, _clean_float(target_first_elapsed_s))
    auc_target = max(0.0, _clean_float(target_auc))
    elapsed_score = 0.0
    if first_elapsed_s is not None and first_elapsed_s >= 0.0:
        elapsed_score = min(1.0, target_elapsed / max(first_elapsed_s, 0.001))
    auc_score = 1.0 if auc_target == 0.0 else min(1.0, max(0.0, auc) / auc_target)
    score = 100.0 * (0.70 * elapsed_score + 0.30 * auc_score)
    passed = (
        first_elapsed_s is not None
        and first_elapsed_s <= target_elapsed
        and (auc_target == 0.0 or auc >= auc_target)
    )
    payload = {
        "first_candidate_elapsed_s": first_elapsed_s,
        "avg_candidate_bug_discovery_auc": max(0.0, auc),
    }
    payload.update(dict(evidence or {}))
    return {
        "score": score,
        "passed": bool(passed),
        "observed": first_elapsed_s,
        "target": target_elapsed,
        "auc_observed": max(0.0, auc),
        "auc_target": auc_target,
        "evidence": payload,
        "reason": "fast candidate discovery reached" if passed else "needs earlier reproducible candidate discovery",
    }


def _coverage_axis_count(coverage: Mapping[str, Any], *, axis_keys: tuple[str, ...]) -> int:
    count = 0
    for key in axis_keys:
        value = coverage.get(key)
        if isinstance(value, Mapping):
            count += len([item for item in value if str(item).strip()])
        else:
            count += len(_list(value))
    return count


def _first_candidate_elapsed_s(discovery: Mapping[str, Any]) -> float | None:
    first = _mapping(discovery.get("first_candidate"))
    first_elapsed = _optional_float(first.get("elapsed_s"))
    if first_elapsed is not None:
        return first_elapsed
    values = []
    for item in _mapping(discovery.get("candidate_family_first_seen")).values():
        elapsed = _optional_float(_mapping(item).get("elapsed_s"))
        if elapsed is not None:
            values.append(elapsed)
    return min(values) if values else None


def _thresholds(value: ICSEExperimentQualityThresholds | Mapping[str, Any] | None) -> ICSEExperimentQualityThresholds:
    if value is None:
        return DEFAULT_ICSE_EXPERIMENT_QUALITY_THRESHOLDS
    if isinstance(value, ICSEExperimentQualityThresholds):
        return value
    data = asdict(DEFAULT_ICSE_EXPERIMENT_QUALITY_THRESHOLDS)
    for key, raw in value.items():
        if key not in data:
            continue
        if isinstance(data[key], int):
            data[key] = _non_negative_int(raw)
        else:
            data[key] = _clean_float(raw)
    return ICSEExperimentQualityThresholds(**data)


def _methodology_claim(score: float, grade: str, ready: bool) -> str:
    if ready:
        return f"ICSE experiment evidence is claim-ready at grade {grade} with score {score:.1f}."
    return f"ICSE experiment evidence is grade {grade} with score {score:.1f}; prioritize the weakest dimensions before final claims."


def _grade(score: float) -> str:
    if score >= 90.0:
        return "A"
    if score >= 80.0:
        return "B"
    if score >= 70.0:
        return "C"
    if score >= 60.0:
        return "D"
    return "F"


def _count_mapping_or_value(value: Any, fallback: Any = None) -> int:
    if isinstance(value, Mapping):
        return len([key for key, raw in value.items() if str(key).strip() and _non_negative_int(raw) > 0])
    return _non_negative_int(value if value is not None else fallback)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, Mapping):
        return [key for key in value if str(key).strip()]
    if isinstance(value, (list, tuple, set, frozenset)):
        return [item for item in value if str(item).strip()]
    return []


def _ratio(value: Any) -> float:
    return max(0.0, min(1.0, _clean_float(value)))


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    result = _clean_float(value)
    return result if math.isfinite(result) else None


def _clean_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if math.isfinite(result) else 0.0


def _non_negative_int(value: Any) -> int:
    try:
        result = int(float(value))
    except (TypeError, ValueError):
        return 0
    return max(0, result)
