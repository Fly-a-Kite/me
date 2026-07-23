"""Coverage-first promotion policy for bounded semantic regression portfolios."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable

from datadiff.method_arms import (
    DEFAULT_COVERAGE_METHOD_ARM_ID,
    DEFAULT_METHOD_ARM_ID,
)
from datadiff.semantic_family_universe_v2 import (
    SEMANTIC_FAMILY_GLOBAL_V3_METHOD_ARM_ID,
)


SEMANTIC_COVERAGE_PROMOTION_SCHEMA_VERSION = "semantic-coverage-promotion-v1"
SEMANTIC_COVERAGE_PROMOTION_V2_SCHEMA_VERSION = "semantic-coverage-promotion-v2"


def semantic_coverage_promotion_evaluator(
    expanded_audit: Mapping[str, Any],
) -> Callable[..., dict[str, Any]]:
    """Select the fail-closed policy version from the expanded audit shape."""

    backend_coverage = expanded_audit.get("backend_coverage", {})
    has_global_v4_coverage = (
        isinstance(backend_coverage, Mapping)
        and isinstance(backend_coverage.get("expanded_global_v4"), Mapping)
    )
    candidate_arm = str(
        expanded_audit.get("global_v4_method_arm_id", "") or ""
    )
    if has_global_v4_coverage or candidate_arm == "p8_semantic_witness_global_v4":
        return evaluate_semantic_coverage_promotion_v2
    return evaluate_semantic_coverage_promotion


def evaluate_semantic_coverage_promotion_auto(
    *,
    baseline_audit: Mapping[str, Any],
    expanded_audit: Mapping[str, Any],
    recall_result: Mapping[str, Any],
    paired_result: Mapping[str, Any],
    ratio_ci_upper_max: float = 1.25,
) -> dict[str, Any]:
    evaluator = semantic_coverage_promotion_evaluator(expanded_audit)
    result = evaluator(
        baseline_audit=baseline_audit,
        expanded_audit=expanded_audit,
        recall_result=recall_result,
        paired_result=paired_result,
        ratio_ci_upper_max=ratio_ci_upper_max,
    )
    result["evaluation_dispatch"] = {
        "mode": "automatic",
        "evaluator": evaluator.__name__,
        "schema_version": result.get("schema_version", ""),
    }
    return result


def evaluate_semantic_coverage_promotion(
    *,
    baseline_audit: Mapping[str, Any],
    expanded_audit: Mapping[str, Any],
    recall_result: Mapping[str, Any],
    paired_result: Mapping[str, Any],
    ratio_ci_upper_max: float = 1.25,
) -> dict[str, Any]:
    """Evaluate engineering promotion without using fresh yield as a veto.

    Fresh yield remains evidence for a separate discovery-yield claim.  A
    bounded regression portfolio is promoted when structural coverage strictly
    improves, correctness guardrails pass, and CPU/wall/evidence-I/O remain
    non-inferior.
    """

    baseline_summary = dict(baseline_audit.get("summary", {}) or {})
    expanded_summary = dict(expanded_audit.get("summary", {}) or {})
    baseline_portfolio = dict(baseline_audit.get("global_portfolio", {}) or {})
    expanded_portfolio = dict(expanded_audit.get("global_portfolio", {}) or {})
    coverage_profile = dict(expanded_audit.get("coverage_profile", {}) or {})
    old_profile = dict(coverage_profile.get("baseline_global_v2", {}) or {})
    new_profile = dict(coverage_profile.get("expanded_global_v3", {}) or {})

    coverage_checks = {
        "family_count_strictly_increased": _int(expanded_summary, "family_count")
        > _int(baseline_summary, "family_count"),
        "cell_count_strictly_increased": _int(expanded_portfolio, "cell_count")
        > _int(baseline_portfolio, "cell_count"),
        "required_backend_pair_count_strictly_increased": _int(
            expanded_summary,
            "required_backend_pair_count",
        )
        > _int(baseline_summary, "required_backend_pair_count"),
        "operation_kind_count_strictly_increased": len(
            new_profile.get("operation_kinds", []) or []
        )
        > len(old_profile.get("operation_kinds", []) or []),
        "expression_kind_count_strictly_increased": len(
            new_profile.get("expression_kinds", []) or []
        )
        > len(old_profile.get("expression_kinds", []) or []),
        "aggregate_function_count_strictly_increased": len(
            new_profile.get("aggregate_functions", []) or []
        )
        > len(old_profile.get("aggregate_functions", []) or []),
    }

    recall = dict(recall_result.get("decision", {}) or {})
    aggregates = dict(paired_result.get("aggregates", {}) or {})
    treatment = dict(
        aggregates.get("p8_semantic_witness_global_v3", {}) or {}
    )
    paired_decision = dict(paired_result.get("decision", {}) or {})
    integrity = dict(paired_decision.get("integrity_checks", {}) or {})
    backend_execution = dict(expanded_audit.get("backend_execution", {}) or {})
    correctness_checks = {
        "expanded_universe_complete": expanded_summary.get("complete") is True,
        "all_backend_results_ok": (
            backend_execution.get("complete") is True
            and not list(backend_execution.get("errors", []) or [])
        ),
        "paired_integrity_passed": paired_decision.get("integrity_gate_passed")
        is True,
        "treatment_activation_complete": float(
            treatment.get("activation_rate", 0.0) or 0.0
        )
        == 1.0,
        "preflight_repaired_zero": _int(treatment, "preflight_repaired_cases")
        == 0,
        "preflight_invalid_zero": _int(treatment, "preflight_invalid_cases") == 0,
        "preflight_fallback_zero": _int(treatment, "preflight_fallback_cases")
        == 0,
        "paired_activation_regressions_zero": _int(
            paired_result.get("paired", {}) or {},
            "activation_regressions",
        )
        == 0,
        "canonical_recall_non_regression": recall.get(
            "canonical_recall_non_regression"
        )
        is True,
        "low_io_contract_honored": integrity.get("low_io_contract_honored")
        is True,
        "normalized_results_not_retained": integrity.get(
            "normalized_results_not_retained"
        )
        is True,
    }

    statistics = dict(
        (paired_result.get("paired", {}) or {}).get("statistics", {}) or {}
    )
    performance_checks = {
        name: _ci_upper(statistics, metric) <= float(ratio_ci_upper_max)
        for name, metric in (
            ("process_cpu_noninferiority", "process_cpu_ratio"),
            ("wall_time_noninferiority", "wall_ratio"),
            ("evidence_io_noninferiority", "evidence_byte_ratio"),
        )
    }
    performance_checks["per_case_sidecars_zero"] = _int(
        treatment,
        "per_case_sidecar_files",
    ) == 0
    performance_checks["retained_normalized_rows_zero"] = _int(
        treatment,
        "retained_normalized_rows",
    ) == 0

    fresh_survivors = _int(treatment, "strict_recheck_surviving_candidate_cases")
    fresh_families = _int(treatment, "unique_candidate_family_count")
    promotion_ready = all(
        [
            *coverage_checks.values(),
            *correctness_checks.values(),
            *performance_checks.values(),
        ]
    )
    return {
        "schema_version": SEMANTIC_COVERAGE_PROMOTION_SCHEMA_VERSION,
        "policy": {
            "promotion_basis": (
                "strict coverage improvement plus correctness and CPU/wall/evidence-I/O "
                "non-inferiority"
            ),
            "fresh_yield_role": (
                "yield-claim evidence only; zero fresh yield is neutral for engineering "
                "coverage promotion"
            ),
            "ratio_ci_upper_max": float(ratio_ci_upper_max),
        },
        "coverage_checks": coverage_checks,
        "correctness_checks": correctness_checks,
        "performance_io_checks": performance_checks,
        "coverage_delta": {
            "families": [
                _int(baseline_summary, "family_count"),
                _int(expanded_summary, "family_count"),
            ],
            "cells": [
                _int(baseline_portfolio, "cell_count"),
                _int(expanded_portfolio, "cell_count"),
            ],
            "required_backend_pairs": [
                _int(baseline_summary, "required_backend_pair_count"),
                _int(expanded_summary, "required_backend_pair_count"),
            ],
            "operation_kinds": [
                len(old_profile.get("operation_kinds", []) or []),
                len(new_profile.get("operation_kinds", []) or []),
            ],
            "expression_kinds": [
                len(old_profile.get("expression_kinds", []) or []),
                len(new_profile.get("expression_kinds", []) or []),
            ],
            "aggregate_functions": [
                len(old_profile.get("aggregate_functions", []) or []),
                len(new_profile.get("aggregate_functions", []) or []),
            ],
        },
        "ratios": {
            metric: dict(statistics.get(metric, {}) or {})
            for metric in (
                "process_cpu_ratio",
                "wall_ratio",
                "evidence_byte_ratio",
            )
        },
        "fresh_signal": {
            "strict_survivors": fresh_survivors,
            "unique_candidate_families": fresh_families,
            "positive": fresh_survivors > 0 or fresh_families > 0,
            "promotion_veto": False,
        },
        "decision": {
            "coverage_promotion_ready": promotion_ready,
            "promoted_method_arm": (
                SEMANTIC_FAMILY_GLOBAL_V3_METHOD_ARM_ID
                if promotion_ready
                else ""
            ),
            "promotion_scope": "coverage_regression_screening",
            "open_ended_discovery_default": DEFAULT_METHOD_ARM_ID,
            "bug_yield_improvement_claim_allowed": bool(
                fresh_survivors > 0 or fresh_families > 0
            ),
            "reason": (
                "Coverage strictly improved and correctness/performance/I-O guardrails "
                "passed; zero fresh candidates do not block coverage promotion."
                if promotion_ready
                else "At least one coverage, correctness, or non-inferiority guardrail failed."
            ),
        },
    }


def evaluate_semantic_coverage_promotion_v2(
    *,
    baseline_audit: Mapping[str, Any],
    expanded_audit: Mapping[str, Any],
    recall_result: Mapping[str, Any],
    paired_result: Mapping[str, Any],
    ratio_ci_upper_max: float = 1.25,
) -> dict[str, Any]:
    """Evaluate Global-v4 with per-backend and interaction-level evidence.

    Global operation and aggregate breadth may remain equal once the generic
    vocabulary is saturated.  Promotion instead requires every backend's
    generic breadth to improve, new operation-chain interactions to be covered,
    exact dtype/layout cells to be exercised successfully, and the usual
    correctness/recall/performance guards to pass.  Fresh yield is never a veto.
    """

    baseline_summary = dict(baseline_audit.get("summary", {}) or {})
    expanded_summary = dict(expanded_audit.get("summary", {}) or {})
    baseline_portfolio = dict(baseline_audit.get("global_portfolio", {}) or {})
    expanded_portfolio = dict(expanded_audit.get("global_portfolio", {}) or {})
    backend_coverage = dict(expanded_audit.get("backend_coverage", {}) or {})
    baseline_profile = dict(
        backend_coverage.get("baseline_global_v3", {}) or {}
    )
    expanded_profile = dict(
        backend_coverage.get("expanded_global_v4", {}) or {}
    )
    baseline_backends = dict(baseline_profile.get("backends", {}) or {})
    expanded_backends = dict(expanded_profile.get("backends", {}) or {})
    backend_names = sorted(set(baseline_backends) | set(expanded_backends))

    per_backend_checks: dict[str, dict[str, bool]] = {}
    for backend in backend_names:
        old = dict(baseline_backends.get(backend, {}) or {})
        new = dict(expanded_backends.get(backend, {}) or {})
        per_backend_checks[backend] = {
            "operation_kinds_strictly_increased": len(
                new.get("operation_kinds", []) or []
            )
            > len(old.get("operation_kinds", []) or []),
            "expression_kinds_strictly_increased": len(
                new.get("expression_kinds", []) or []
            )
            > len(old.get("expression_kinds", []) or []),
            "aggregate_functions_strictly_increased": len(
                new.get("aggregate_functions", []) or []
            )
            > len(old.get("aggregate_functions", []) or []),
            "risk_pipeline_cells_strictly_increased": _int(
                new,
                "risk_pipeline_cell_count",
            )
            > _int(old, "risk_pipeline_cell_count"),
            "risk_operation_chains_strictly_increased": len(
                new.get("risk_operation_chains", []) or []
            )
            > len(old.get("risk_operation_chains", []) or []),
            "boundary_profiles_strictly_increased": len(
                new.get("boundary_profiles", []) or []
            )
            > len(old.get("boundary_profiles", []) or []),
            "probe_and_generic_coverage_separated": not set(
                new.get("operation_kinds", []) or []
            ).intersection(new.get("probe_operation_kinds", []) or []),
        }

    old_union = _union_backend_coverage(baseline_backends)
    new_union = _union_backend_coverage(expanded_backends)
    coverage_checks = {
        "family_count_strictly_increased": _int(expanded_summary, "family_count")
        > _int(baseline_summary, "family_count"),
        "cell_count_strictly_increased": _int(expanded_portfolio, "cell_count")
        > _int(baseline_portfolio, "cell_count"),
        "required_backend_pair_count_strictly_increased": _int(
            expanded_summary,
            "required_backend_pair_count",
        )
        > _int(baseline_summary, "required_backend_pair_count"),
        "global_operation_kinds_non_decreasing": len(new_union["operation_kinds"])
        >= len(old_union["operation_kinds"]),
        "global_expression_kinds_non_decreasing": len(
            new_union["expression_kinds"]
        )
        >= len(old_union["expression_kinds"]),
        "global_aggregate_functions_non_decreasing": len(
            new_union["aggregate_functions"]
        )
        >= len(old_union["aggregate_functions"]),
        "every_backend_strictly_improved": bool(per_backend_checks)
        and all(
            all(checks.values())
            for checks in per_backend_checks.values()
        ),
    }

    backend_execution = dict(expanded_audit.get("backend_execution", {}) or {})
    optimizer = dict(
        backend_execution.get("optimizer_plan_observation", {}) or {}
    )
    structural_checks = {
        "expanded_universe_complete": expanded_summary.get("complete") is True,
        "all_new_backend_results_ok": (
            backend_execution.get("complete") is True
            and not list(backend_execution.get("errors", []) or [])
        ),
        "exact_dtype_cells_complete_and_equal": _int(
            backend_execution,
            "exact_dtype_equal_cell_count",
        )
        == 10,
        "layout_cells_complete_and_equal": _int(
            backend_execution,
            "layout_equal_cell_count",
        )
        == 24,
        "pyarrow_layouts_actually_applied": _int(
            backend_execution,
            "pyarrow_applied_layout_cell_count",
        )
        == 24,
        "datafusion_optimizer_rewrite_observed": _optimizer_complete(
            optimizer,
            "datafusion",
        ),
        "polars_lazy_optimizer_rewrite_observed": _optimizer_complete(
            optimizer,
            "polars_lazy",
        ),
    }

    candidate_arm = str(
        expanded_audit.get("global_v4_method_arm_id", "")
        or "p8_semantic_witness_global_v4"
    )
    recall = dict(recall_result.get("decision", {}) or {})
    aggregates = dict(paired_result.get("aggregates", {}) or {})
    treatment = dict(aggregates.get(candidate_arm, {}) or {})
    paired_decision = dict(paired_result.get("decision", {}) or {})
    integrity = dict(paired_decision.get("integrity_checks", {}) or {})
    correctness_checks = {
        **structural_checks,
        "paired_integrity_passed": paired_decision.get("integrity_gate_passed")
        is True,
        "treatment_activation_complete": float(
            treatment.get("activation_rate", 0.0) or 0.0
        )
        == 1.0,
        "preflight_repaired_zero": _int(treatment, "preflight_repaired_cases")
        == 0,
        "preflight_invalid_zero": _int(treatment, "preflight_invalid_cases") == 0,
        "preflight_fallback_zero": _int(treatment, "preflight_fallback_cases")
        == 0,
        "paired_activation_regressions_zero": _int(
            paired_result.get("paired", {}) or {},
            "activation_regressions",
        )
        == 0,
        "canonical_recall_non_regression": recall.get(
            "canonical_recall_non_regression"
        )
        is True,
        "low_io_contract_honored": integrity.get("low_io_contract_honored")
        is True,
        "normalized_results_not_retained": integrity.get(
            "normalized_results_not_retained"
        )
        is True,
    }

    statistics = dict(
        (paired_result.get("paired", {}) or {}).get("statistics", {}) or {}
    )
    performance_checks = {
        name: _ci_upper(statistics, metric) <= float(ratio_ci_upper_max)
        for name, metric in (
            ("process_cpu_noninferiority", "process_cpu_ratio"),
            ("wall_time_noninferiority", "wall_ratio"),
            ("evidence_io_noninferiority", "evidence_byte_ratio"),
        )
    }
    performance_checks["per_case_sidecars_zero"] = _int(
        treatment,
        "per_case_sidecar_files",
    ) == 0
    performance_checks["retained_normalized_rows_zero"] = _int(
        treatment,
        "retained_normalized_rows",
    ) == 0

    fresh_survivors = _int(treatment, "strict_recheck_surviving_candidate_cases")
    fresh_families = _int(treatment, "unique_candidate_family_count")
    promotion_ready = all(
        [
            *coverage_checks.values(),
            *correctness_checks.values(),
            *performance_checks.values(),
        ]
    )
    return {
        "schema_version": SEMANTIC_COVERAGE_PROMOTION_V2_SCHEMA_VERSION,
        "policy": {
            "promotion_basis": (
                "per-backend breadth and interaction improvement plus exact dtype/layout, "
                "optimizer activation, correctness, recall, and CPU/wall/evidence-I/O "
                "non-inferiority"
            ),
            "saturated_global_vocabulary_rule": (
                "global operation/expression/aggregate counts may remain equal but may not decline"
            ),
            "fresh_yield_role": (
                "yield-claim evidence only; zero fresh yield is neutral for engineering "
                "coverage promotion"
            ),
            "ratio_ci_upper_max": float(ratio_ci_upper_max),
        },
        "coverage_checks": coverage_checks,
        "per_backend_coverage_checks": per_backend_checks,
        "correctness_checks": correctness_checks,
        "performance_io_checks": performance_checks,
        "coverage_delta": {
            "families": [
                _int(baseline_summary, "family_count"),
                _int(expanded_summary, "family_count"),
            ],
            "cells": [
                _int(baseline_portfolio, "cell_count"),
                _int(expanded_portfolio, "cell_count"),
            ],
            "required_backend_pairs": [
                _int(baseline_summary, "required_backend_pair_count"),
                _int(expanded_summary, "required_backend_pair_count"),
            ],
            "global_operation_kinds": [
                len(old_union["operation_kinds"]),
                len(new_union["operation_kinds"]),
            ],
            "global_expression_kinds": [
                len(old_union["expression_kinds"]),
                len(new_union["expression_kinds"]),
            ],
            "global_aggregate_functions": [
                len(old_union["aggregate_functions"]),
                len(new_union["aggregate_functions"]),
            ],
        },
        "ratios": {
            metric: dict(statistics.get(metric, {}) or {})
            for metric in (
                "process_cpu_ratio",
                "wall_ratio",
                "evidence_byte_ratio",
            )
        },
        "fresh_signal": {
            "strict_survivors": fresh_survivors,
            "unique_candidate_families": fresh_families,
            "positive": fresh_survivors > 0 or fresh_families > 0,
            "promotion_veto": False,
        },
        "decision": {
            "coverage_promotion_ready": promotion_ready,
            "candidate_method_arm": candidate_arm,
            "promoted_method_arm": candidate_arm if promotion_ready else "",
            "default_switch_allowed": promotion_ready,
            "current_coverage_default": DEFAULT_COVERAGE_METHOD_ARM_ID,
            "open_ended_discovery_default": DEFAULT_METHOD_ARM_ID,
            "promotion_scope": "coverage_regression_screening",
            "bug_yield_improvement_claim_allowed": bool(
                fresh_survivors > 0 or fresh_families > 0
            ),
            "reason": (
                "Every backend and interaction layer improved while correctness, recall, "
                "and performance/I-O guardrails passed; zero fresh candidates are neutral."
                if promotion_ready
                else "At least one per-backend coverage, correctness, recall, or non-inferiority guardrail failed."
            ),
        },
    }

def _union_backend_coverage(
    backends: Mapping[str, Any],
) -> dict[str, set[str]]:
    out = {
        "operation_kinds": set(),
        "expression_kinds": set(),
        "aggregate_functions": set(),
    }
    for raw in backends.values():
        row = dict(raw or {}) if isinstance(raw, Mapping) else {}
        for key in out:
            out[key].update(str(item) for item in row.get(key, []) or [])
    return out


def _optimizer_complete(
    optimizer: Mapping[str, Any],
    backend: str,
) -> bool:
    row = dict(optimizer.get(backend, {}) or {})
    return (
        _int(row, "observed_cell_count") == 36
        and _int(row, "changed_cell_count") == 36
        and len(row.get("pipelines", []) or []) == 18
    )


def _int(payload: Mapping[str, Any], key: str) -> int:
    return int(payload.get(key, 0) or 0)


def _ci_upper(statistics: Mapping[str, Any], metric: str) -> float:
    row = dict(statistics.get(metric, {}) or {})
    interval = list(row.get("confidence_interval_95", []) or [])
    return float(interval[1]) if len(interval) == 2 else float("inf")
