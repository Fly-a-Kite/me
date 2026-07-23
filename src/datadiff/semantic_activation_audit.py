"""Paired, backend-free audit for goal-first semantic activation."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from datadiff.goal_first import generate_goal_first_case, generation_goals
from datadiff.method_arms import arm_difference, method_arm, resolve_method_arm
from datadiff.preflight import preflight_case


SEMANTIC_ACTIVATION_AUDIT_SCHEMA_VERSION = "semantic-activation-paired-audit-v1"


def audit_semantic_activation_pair(
    *,
    seed_start: int = 0,
    cases: int = 84,
    profile: str = "",
    boundary_mode: str = "fault_model_targeted",
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for seed in range(int(seed_start), int(seed_start) + max(0, int(cases))):
        control = generate_goal_first_case(
            seed,
            profile=profile,
            boundary_mode=boundary_mode,
            require_semantic_activation=False,
        )
        treatment = generate_goal_first_case(
            seed,
            profile=profile,
            boundary_mode=boundary_mode,
            require_semantic_activation=True,
        )
        if control.case is None or treatment.case is None:
            rows.append(
                {
                    "seed": seed,
                    "constructible": False,
                    "control_skip_reason": control.trace.get("skip_reason", ""),
                    "treatment_skip_reason": treatment.trace.get("skip_reason", ""),
                }
            )
            continue
        control_activation = dict(control.trace.get("semantic_activation", {}) or {})
        treatment_activation = dict(
            treatment.trace.get("semantic_activation", {}) or {}
        )
        treatment_preflight = preflight_case(
            treatment.case,
            enable_validation=True,
            enable_repair=True,
        )
        control_goal = str(
            (control.trace.get("selected_goal", {}) or {}).get("goal_id", "") or ""
        )
        treatment_goal = str(
            (treatment.trace.get("selected_goal", {}) or {}).get("goal_id", "")
            or ""
        )
        repair = dict(treatment.trace.get("activation_repair", {}) or {})
        rows.append(
            {
                "seed": seed,
                "constructible": True,
                "goal_id": control_goal,
                "selected_goal_match": control_goal == treatment_goal,
                "control": {
                    "generation_mode": control.trace.get("generation_mode", ""),
                    "evaluation_status": control_activation.get(
                        "evaluation_status", "not_evaluated"
                    ),
                    "semantically_activated": control_activation.get(
                        "semantically_activated"
                    ),
                    "missing_tokens": list(
                        control_activation.get("missing_tokens", []) or []
                    ),
                    "boundary_profile": str(
                        (control.trace.get("boundary_application", {}) or {}).get(
                            "profile_id", ""
                        )
                        or ""
                    ),
                },
                "treatment": {
                    "generation_mode": treatment.trace.get("generation_mode", ""),
                    "evaluation_status": treatment_activation.get(
                        "evaluation_status", "not_evaluated"
                    ),
                    "semantically_activated": treatment_activation.get(
                        "semantically_activated"
                    ),
                    "missing_tokens": list(
                        treatment_activation.get("missing_tokens", []) or []
                    ),
                    "boundary_profile": str(
                        (treatment.trace.get("boundary_application", {}) or {}).get(
                            "profile_id", ""
                        )
                        or ""
                    ),
                    "repair_id": str(repair.get("repair_id", "") or ""),
                    "repair_changed": bool(repair.get("changed", False)),
                    "repair_satisfied": bool(repair.get("satisfied", False)),
                    "preflight_valid": bool(treatment_preflight.valid),
                    "preflight_fallback_used": bool(
                        treatment_preflight.fallback_used
                    ),
                },
            }
        )

    constructible_rows = [row for row in rows if row.get("constructible")]
    control_summary = _arm_summary(constructible_rows, arm="control")
    treatment_summary = _arm_summary(constructible_rows, arm="treatment")
    selected_goal_mismatches = sum(
        not bool(row.get("selected_goal_match", False)) for row in constructible_rows
    )
    paired_improvements = sum(
        row["control"]["evaluation_status"] != "activated"
        and row["treatment"]["evaluation_status"] == "activated"
        for row in constructible_rows
    )
    paired_regressions = sum(
        row["control"]["evaluation_status"] == "activated"
        and row["treatment"]["evaluation_status"] != "activated"
        for row in constructible_rows
    )
    treatment_preflight_valid = sum(
        bool(row["treatment"]["preflight_valid"]) for row in constructible_rows
    )
    treatment_preflight_fallback = sum(
        bool(row["treatment"]["preflight_fallback_used"])
        for row in constructible_rows
    )
    registered_goal_count = len(generation_goals())
    method_difference = arm_difference(
        method_arm("p8_candidate_v1"),
        method_arm("p8_semantic_witness_v1"),
    )
    gate = {
        "single_declared_method_dimension": method_difference
        == {"generation_mode": ("goal_first", "goal_first_witness")},
        "all_cases_constructible": len(constructible_rows) == len(rows),
        "all_registered_goals_evaluated": (
            treatment_summary["evaluated_goal_count"] == registered_goal_count
        ),
        "treatment_activation_rate_is_one": (
            treatment_summary["activation_rate_all_cases"] == 1.0
        ),
        "treatment_preflight_valid_rate_is_one": (
            treatment_preflight_valid == len(constructible_rows)
        ),
        "treatment_preflight_fallback_count_is_zero": (
            treatment_preflight_fallback == 0
        ),
        "selected_goal_mismatch_count_is_zero": selected_goal_mismatches == 0,
        "paired_activation_regression_count_is_zero": paired_regressions == 0,
    }
    gate["passed"] = all(gate.values())
    return {
        "schema_version": SEMANTIC_ACTIVATION_AUDIT_SCHEMA_VERSION,
        "design": {
            "seed_start": int(seed_start),
            "cases": max(0, int(cases)),
            "profile": str(profile or ""),
            "boundary_mode": str(boundary_mode),
            "control_method_arm": "p8_candidate_v1",
            "treatment_method_arm": "p8_semantic_witness_v1",
            "changed_dimension": "generation_mode",
            "method_difference": {
                key: list(values) for key, values in sorted(method_difference.items())
            },
            "control_method_manifest": resolve_method_arm(
                "p8_candidate_v1"
            ).manifest(),
            "treatment_method_manifest": resolve_method_arm(
                "p8_semantic_witness_v1"
            ).manifest(),
            "backend_execution": False,
        },
        "summary": {
            "registered_goal_count": registered_goal_count,
            "constructible_case_count": len(constructible_rows),
            "selected_goal_mismatch_count": selected_goal_mismatches,
            "paired_activation_improvement_count": paired_improvements,
            "paired_activation_regression_count": paired_regressions,
            "treatment_preflight_valid_count": treatment_preflight_valid,
            "treatment_preflight_fallback_count": treatment_preflight_fallback,
            "control": control_summary,
            "treatment": treatment_summary,
        },
        "gate": gate,
        "rows": rows,
    }


def render_semantic_activation_audit_markdown(audit: dict[str, Any]) -> str:
    design = audit["design"]
    summary = audit["summary"]
    control = summary["control"]
    treatment = summary["treatment"]
    lines = [
        "# Semantic Activation Paired Audit",
        "",
        "This is a generation/preflight audit; it does not claim backend bug-yield improvement.",
        "",
        "## Design",
        "",
        f"- Seeds: `{design['seed_start']}` to "
        f"`{design['seed_start'] + design['cases'] - 1}` ({design['cases']} cases)",
        f"- Boundary mode: `{design['boundary_mode']}`",
        f"- Control: `{design['control_method_arm']}`",
        f"- Treatment: `{design['treatment_method_arm']}`",
        f"- Single changed dimension: `{design['changed_dimension']}`",
        "",
        "## Results",
        "",
        "| Arm | Evaluated | Activated | Not activated | Activation rate |",
        "|---|---:|---:|---:|---:|",
        f"| Control | {control['evaluated_cases']} | {control['activated_cases']} | "
        f"{control['not_activated_cases']} | {control['activation_rate_all_cases']:.3f} |",
        f"| Treatment | {treatment['evaluated_cases']} | {treatment['activated_cases']} | "
        f"{treatment['not_activated_cases']} | {treatment['activation_rate_all_cases']:.3f} |",
        "",
        f"- Paired improvements: `{summary['paired_activation_improvement_count']}`",
        f"- Paired regressions: `{summary['paired_activation_regression_count']}`",
        f"- Treatment preflight valid: `{summary['treatment_preflight_valid_count']}`/"
        f"`{summary['constructible_case_count']}`",
        f"- Gate: `{'PASS' if audit['gate']['passed'] else 'FAIL'}`",
        "",
        "## By Goal",
        "",
        "| Goal | Control activated/cases | Treatment activated/cases |",
        "|---|---:|---:|",
    ]
    goal_ids = sorted(
        set(control["by_goal"]) | set(treatment["by_goal"])
    )
    for goal_id in goal_ids:
        control_goal = control["by_goal"].get(goal_id, {})
        treatment_goal = treatment["by_goal"].get(goal_id, {})
        lines.append(
            f"| `{goal_id}` | {control_goal.get('activated', 0)}/"
            f"{control_goal.get('cases', 0)} | {treatment_goal.get('activated', 0)}/"
            f"{treatment_goal.get('cases', 0)} |"
        )
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "Passing this gate only proves deterministic semantic-witness construction and "
            "preflight validity. Promotion still requires paired backend experiments, "
            "historical recall, fresh-only confirmed-root yield, and cost measurements.",
            "",
        ]
    )
    return "\n".join(lines)


def _arm_summary(rows: list[dict[str, Any]], *, arm: str) -> dict[str, Any]:
    statuses = Counter(str(row[arm]["evaluation_status"]) for row in rows)
    by_goal: dict[str, Counter[str]] = defaultdict(Counter)
    repair_counts: Counter[str] = Counter()
    boundary_counts: Counter[str] = Counter()
    for row in rows:
        goal_id = str(row.get("goal_id", "unknown") or "unknown")
        status = str(row[arm]["evaluation_status"])
        by_goal[goal_id]["cases"] += 1
        by_goal[goal_id][status] += 1
        boundary_counts[str(row[arm].get("boundary_profile", "") or "none")] += 1
        if arm == "treatment":
            repair_counts[str(row[arm].get("repair_id", "") or "already_activated")] += 1
    case_count = len(rows)
    evaluated = statuses["activated"] + statuses["not_activated"]
    return {
        "cases": case_count,
        "evaluated_cases": evaluated,
        "evaluated_goal_count": sum(
            any(key in counts for key in ("activated", "not_activated"))
            for counts in by_goal.values()
        ),
        "activated_cases": statuses["activated"],
        "not_activated_cases": statuses["not_activated"],
        "not_evaluated_cases": statuses["not_evaluated"],
        "activation_rate_all_cases": (
            statuses["activated"] / case_count if case_count else 0.0
        ),
        "activation_rate_evaluated": (
            statuses["activated"] / evaluated if evaluated else 0.0
        ),
        "by_goal": {
            goal_id: dict(sorted(counts.items()))
            for goal_id, counts in sorted(by_goal.items())
        },
        "boundary_profile_counts": dict(sorted(boundary_counts.items())),
        "repair_counts": dict(sorted(repair_counts.items())),
    }
