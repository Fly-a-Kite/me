"""Backend-free diversity and efficiency audit for semantic witness v2."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from collections.abc import Mapping
from typing import Any

from datadiff.canonicalization import short_canonical_hash
from datadiff.compact_novelty import SeedRangeBitmap
from datadiff.goal_first import (
    generate_goal_first_case,
    generation_goals,
    semantic_witness_epoch_key,
)
from datadiff.method_arms import arm_difference, method_arm, resolve_method_arm
from datadiff.operation_semantics import op_kind
from datadiff.preflight import preflight_case
from datadiff.semantic_trigger_bitmap import SemanticTriggerBitmap
from datadiff.goal_first_variants import SEMANTIC_WITNESS_DATA_PATTERN_COUNT


SEMANTIC_WITNESS_DIVERSITY_AUDIT_SCHEMA_VERSION = (
    "semantic-witness-diversity-paired-audit-v1"
)
CONTROL_ARM = "p8_semantic_witness_v1"
TREATMENT_ARM = "p8_semantic_witness_v2"


def audit_semantic_witness_diversity_pair(
    *,
    seed_start: int = 0,
    cases: int = 840,
    profile: str = "",
    boundary_mode: str = "fault_model_targeted",
    cache_aware_epochs: bool = False,
    max_cpu_ratio: float = 1.25,
    max_wall_ratio: float = 1.25,
) -> dict[str, Any]:
    """Compare v1 and v2 without retaining cases or writing per-case files."""

    normalized_cases = max(0, int(cases))
    arm_metrics = {
        CONTROL_ARM: _new_arm_metrics(),
        TREATMENT_ARM: _new_arm_metrics(),
    }
    paired_improvements = 0
    paired_regressions = 0
    constructibility_failures: list[dict[str, Any]] = []
    preflight_failures: list[dict[str, Any]] = []
    row_hasher = hashlib.sha256()
    seed_bitmap = SeedRangeBitmap(seed_start, normalized_cases)
    duplicate_seed_observations = 0
    trigger_bitmaps = {
        CONTROL_ARM: SemanticTriggerBitmap(),
        TREATMENT_ARM: SemanticTriggerBitmap(),
    }

    for seed in range(int(seed_start), int(seed_start) + normalized_cases):
        duplicate_seed_observations += int(not seed_bitmap.observe(seed))
        generated_by_arm: dict[str, Any] = {}
        # Alternate order to avoid a systematic warm-cache advantage in the
        # generation/preflight efficiency measurements.
        arm_order = (
            (CONTROL_ARM, TREATMENT_ARM)
            if seed % 2 == 0
            else (TREATMENT_ARM, CONTROL_ARM)
        )
        for arm in arm_order:
            diverse = arm == TREATMENT_ARM
            started_wall = time.perf_counter_ns()
            started_cpu = time.process_time_ns()
            generated = generate_goal_first_case(
                seed,
                profile=profile,
                boundary_mode=boundary_mode,
                require_semantic_activation=True,
                diversity_preserving_witness=diverse,
                diversity_epoch_seed=(
                    semantic_witness_epoch_key(seed)
                    if diverse and cache_aware_epochs
                    else None
                ),
            )
            generation_cpu_ns = time.process_time_ns() - started_cpu
            generation_wall_ns = time.perf_counter_ns() - started_wall
            metrics = arm_metrics[arm]
            metrics["generation_cpu_ns"] += generation_cpu_ns
            metrics["generation_wall_ns"] += generation_wall_ns
            generated_by_arm[arm] = generated
            if generated.case is None:
                constructibility_failures.append(
                    {
                        "seed": seed,
                        "arm": arm,
                        "skip_reason": str(generated.trace.get("skip_reason", "") or ""),
                    }
                )
                continue

            preflight_started_wall = time.perf_counter_ns()
            preflight_started_cpu = time.process_time_ns()
            preflight = preflight_case(
                generated.case,
                enable_validation=True,
                enable_repair=True,
            )
            metrics["preflight_cpu_ns"] += time.process_time_ns() - preflight_started_cpu
            metrics["preflight_wall_ns"] += (
                time.perf_counter_ns() - preflight_started_wall
            )
            if not preflight.valid or preflight.fallback_used:
                preflight_failures.append(
                    {
                        "seed": seed,
                        "arm": arm,
                        "valid": bool(preflight.valid),
                        "fallback_used": bool(preflight.fallback_used),
                        "reason": str(preflight.reason or ""),
                    }
                )
            _observe_case(metrics, generated.trace, generated.case, preflight)
            trigger_bitmaps[arm].observe_trace(generated.trace)

        control_status = _activation_status(generated_by_arm.get(CONTROL_ARM))
        treatment_status = _activation_status(generated_by_arm.get(TREATMENT_ARM))
        paired_improvements += int(
            control_status != "activated" and treatment_status == "activated"
        )
        paired_regressions += int(
            control_status == "activated" and treatment_status != "activated"
        )
        compact_pair = {
            "seed": seed,
            "control": _compact_generation_identity(generated_by_arm.get(CONTROL_ARM)),
            "treatment": _compact_generation_identity(
                generated_by_arm.get(TREATMENT_ARM)
            ),
        }
        row_hasher.update(
            json.dumps(
                compact_pair,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("utf-8")
        )
        row_hasher.update(b"\n")

    summaries = {
        arm: _finalize_arm_metrics(metrics, cases=normalized_cases)
        for arm, metrics in arm_metrics.items()
    }
    control = summaries[CONTROL_ARM]
    treatment = summaries[TREATMENT_ARM]
    parent_difference = arm_difference(
        method_arm("p8_candidate_v1"),
        method_arm(TREATMENT_ARM),
    )
    cpu_ratio = _safe_ratio(
        treatment["total_cpu_seconds"],
        control["total_cpu_seconds"],
    )
    wall_ratio = _safe_ratio(
        treatment["total_wall_seconds"],
        control["total_wall_seconds"],
    )
    registered_variant_counts = {
        goal.goal_id: len(goal.witness_variants) for goal in generation_goals()
    }
    per_goal_variant_coverage = all(
        len(treatment["by_goal"].get(goal_id, {}).get("variants", []))
        == registered_count
        for goal_id, registered_count in registered_variant_counts.items()
    )
    per_goal_operation_sequences = all(
        treatment["by_goal"].get(goal.goal_id, {}).get(
            "operation_sequence_count", 0
        )
        >= 3
        for goal in generation_goals()
    )
    expected_palette_cells = sum(
        len(goal.witness_variants) * SEMANTIC_WITNESS_DATA_PATTERN_COUNT
        for goal in generation_goals()
    )
    balanced_goal_pattern_coverage = all(
        treatment["by_goal"].get(goal.goal_id, {}).get(
            "combined_signature_count", 0
        )
        >= len(goal.witness_variants) * SEMANTIC_WITNESS_DATA_PATTERN_COUNT
        for goal in generation_goals()
    )
    balanced_goal_value_coverage = all(
        treatment["by_goal"].get(goal.goal_id, {}).get(
            "data_value_count", 0
        )
        >= 2 * len(goal.witness_variants)
        for goal in generation_goals()
    )
    treatment_trigger_snapshot = trigger_bitmaps[TREATMENT_ARM].snapshot()
    data_value_requirement_met = (
        balanced_goal_value_coverage
        if cache_aware_epochs
        else treatment["data_value_count"] > control["data_value_count"]
    )
    combined_coverage_requirement_met = (
        treatment["combined_signature_count"] >= expected_palette_cells
        and balanced_goal_pattern_coverage
        if cache_aware_epochs
        else treatment["combined_signature_count"]
        >= 2 * max(1, control["combined_signature_count"])
    )
    gate = {
        "single_declared_parent_dimension": parent_difference
        == {"generation_mode": ("goal_first", "goal_first_witness_v2")},
        "all_cases_constructible": not constructibility_failures,
        "treatment_activation_rate_is_one": treatment["activation_rate"] == 1.0,
        "treatment_preflight_valid_rate_is_one": treatment["preflight_valid_rate"]
        == 1.0,
        "treatment_preflight_fallback_count_is_zero": treatment[
            "preflight_fallback_cases"
        ]
        == 0,
        "paired_activation_regression_count_is_zero": paired_regressions == 0,
        "all_registered_variants_covered_per_goal": per_goal_variant_coverage,
        "at_least_three_operation_sequences_per_goal": per_goal_operation_sequences,
        "operation_sequence_diversity_improves": treatment[
            "operation_sequence_count"
        ]
        > control["operation_sequence_count"],
        "program_template_diversity_improves": treatment[
            "program_template_count"
        ]
        > control["program_template_count"],
        "data_shape_diversity_improves": treatment["data_shape_count"]
        > control["data_shape_count"],
        "data_value_coverage_requirement_met": data_value_requirement_met,
        "combined_coverage_requirement_met": combined_coverage_requirement_met,
        "bounded_palette_trigger_cells_complete": (
            not cache_aware_epochs
            or treatment_trigger_snapshot["observed_count"]
            >= expected_palette_cells
        ),
        "boundary_profile_diversity_does_not_regress": treatment[
            "boundary_profile_count"
        ]
        >= control["boundary_profile_count"],
        "cpu_ratio_within_limit": cpu_ratio is not None
        and cpu_ratio <= float(max_cpu_ratio),
        "wall_ratio_within_limit": wall_ratio is not None
        and wall_ratio <= float(max_wall_ratio),
        "seed_bitmap_has_no_duplicate_observations": duplicate_seed_observations
        == 0,
        "trigger_bitmap_has_no_unregistered_dimensions": all(
            bitmap.unregistered_observations == 0
            for bitmap in trigger_bitmaps.values()
        ),
    }
    gate["passed"] = all(gate.values())
    return {
        "schema_version": SEMANTIC_WITNESS_DIVERSITY_AUDIT_SCHEMA_VERSION,
        "design": {
            "seed_start": int(seed_start),
            "cases": normalized_cases,
            "profile": str(profile or ""),
            "boundary_mode": str(boundary_mode),
            "cache_aware_epochs": bool(cache_aware_epochs),
            "expected_bounded_palette_cells": expected_palette_cells,
            "cache_aware_coverage_policy": (
                "balanced_registered_goal_variant_pattern_cells"
                if cache_aware_epochs
                else "unbounded_seeded_signature_growth"
            ),
            "control_method_arm": CONTROL_ARM,
            "treatment_method_arm": TREATMENT_ARM,
            "treatment_parent_arm": "p8_candidate_v1",
            "changed_dimension": "generation_mode",
            "parent_method_difference": {
                key: list(values) for key, values in sorted(parent_difference.items())
            },
            "control_method_manifest": resolve_method_arm(CONTROL_ARM).manifest(),
            "treatment_method_manifest": resolve_method_arm(TREATMENT_ARM).manifest(),
            "backend_execution": False,
            "case_retention": False,
            "per_case_file_output": False,
            "streaming_pair_digest": row_hasher.hexdigest(),
            "seed_bitmap": seed_bitmap.snapshot(),
            "performance_limits": {
                "cpu_ratio_max": float(max_cpu_ratio),
                "wall_ratio_max": float(max_wall_ratio),
            },
        },
        "summary": {
            "paired_activation_improvement_count": paired_improvements,
            "paired_activation_regression_count": paired_regressions,
            "constructibility_failure_count": len(constructibility_failures),
            "preflight_failure_count": len(preflight_failures),
            "registered_variant_counts": registered_variant_counts,
            "generation_and_preflight_cpu_ratio": cpu_ratio,
            "generation_and_preflight_wall_ratio": wall_ratio,
            "duplicate_seed_observation_count": duplicate_seed_observations,
            "trigger_bitmaps": {
                arm: bitmap.snapshot() for arm, bitmap in trigger_bitmaps.items()
            },
            "control": control,
            "treatment": treatment,
        },
        "gate": gate,
        "failures": {
            "constructibility": constructibility_failures,
            "preflight": preflight_failures,
        },
    }


def render_semantic_witness_diversity_markdown(audit: Mapping[str, Any]) -> str:
    design = audit["design"]
    summary = audit["summary"]
    control = summary["control"]
    treatment = summary["treatment"]
    lines = [
        "# Semantic Witness v2 Diversity Audit",
        "",
        "This is a deterministic generation/preflight audit. It does not claim improved bug yield.",
        "",
        "## Design",
        "",
        f"- Seeds: `{design['seed_start']}` to "
        f"`{design['seed_start'] + design['cases'] - 1}` (`{design['cases']}` cases per arm)",
        f"- Control: `{design['control_method_arm']}`",
        f"- Treatment: `{design['treatment_method_arm']}`",
        f"- Cache-aware bounded epochs: `{design['cache_aware_epochs']}`",
        f"- Diversity coverage policy: `{design['cache_aware_coverage_policy']}`",
        "- Per-case retention/file output: disabled; only aggregate evidence and a streaming digest are stored.",
        f"- Exact seed bitmap: `{design['seed_bitmap']['byte_size']}` bytes for "
        f"`{design['seed_bitmap']['size']}` frozen seeds.",
        "",
        "## Activation and Efficiency",
        "",
        "| Arm | Activated | Preflight valid | CPU s | Wall s |",
        "|---|---:|---:|---:|---:|",
        f"| Control | {control['activated_cases']}/{control['cases']} | "
        f"{control['preflight_valid_cases']}/{control['cases']} | "
        f"{control['total_cpu_seconds']:.6f} | {control['total_wall_seconds']:.6f} |",
        f"| Treatment | {treatment['activated_cases']}/{treatment['cases']} | "
        f"{treatment['preflight_valid_cases']}/{treatment['cases']} | "
        f"{treatment['total_cpu_seconds']:.6f} | {treatment['total_wall_seconds']:.6f} |",
        "",
        f"- CPU ratio: `{summary['generation_and_preflight_cpu_ratio']:.4f}`",
        f"- Wall ratio: `{summary['generation_and_preflight_wall_ratio']:.4f}`",
        f"- Exact v2 trigger bitmap: "
        f"`{summary['trigger_bitmaps'][TREATMENT_ARM]['byte_size']}` bytes; "
        f"`{summary['trigger_bitmaps'][TREATMENT_ARM]['observed_count']}` cells observed.",
        "",
        "## Diversity",
        "",
        "| Metric | v1 | v2 |",
        "|---|---:|---:|",
        f"| Builder variants | {control['variant_count']} | {treatment['variant_count']} |",
        f"| Operation sequences | {control['operation_sequence_count']} | {treatment['operation_sequence_count']} |",
        f"| Program templates | {control['program_template_count']} | {treatment['program_template_count']} |",
        f"| Interaction signatures | {control['interaction_signature_count']} | {treatment['interaction_signature_count']} |",
        f"| Boundary profiles | {control['boundary_profile_count']} | {treatment['boundary_profile_count']} |",
        f"| Data-shape signatures | {control['data_shape_count']} | {treatment['data_shape_count']} |",
        f"| Data-value signatures | {control['data_value_count']} | {treatment['data_value_count']} |",
        f"| Combined signatures | {control['combined_signature_count']} | {treatment['combined_signature_count']} |",
        "",
        "## Per Goal",
        "",
        "| Goal | v1 op seq | v2 op seq | v2 variants | v2 shapes | v2 values |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for goal in generation_goals():
        goal_id = goal.goal_id
        left = control["by_goal"].get(goal_id, {})
        right = treatment["by_goal"].get(goal_id, {})
        lines.append(
            f"| `{goal_id}` | {left.get('operation_sequence_count', 0)} | "
            f"{right.get('operation_sequence_count', 0)} | "
            f"{len(right.get('variants', []))} | {right.get('data_shape_count', 0)} | "
            f"{right.get('data_value_count', 0)} |"
        )
    lines.extend(
        [
            "",
            f"- Gate: `{'PASS' if audit['gate']['passed'] else 'FAIL'}`",
            "",
            "## Claim Boundary",
            "",
            "Passing proves deterministic variant coverage, activation, preflight validity, "
            "and backend-free generation efficiency. Backend execution cost and fresh bug "
            "yield require separate paired screening/campaign evidence.",
            "",
        ]
    )
    return "\n".join(lines)


def _new_arm_metrics() -> dict[str, Any]:
    return {
        "cases": 0,
        "activated_cases": 0,
        "preflight_valid_cases": 0,
        "preflight_fallback_cases": 0,
        "generation_cpu_ns": 0,
        "generation_wall_ns": 0,
        "preflight_cpu_ns": 0,
        "preflight_wall_ns": 0,
        "variants": set(),
        "operation_sequences": set(),
        "program_templates": set(),
        "interaction_signatures": set(),
        "interaction_edges": set(),
        "boundary_profiles": set(),
        "data_shapes": set(),
        "data_values": set(),
        "combined_signatures": set(),
        "by_goal": defaultdict(_new_goal_metrics),
    }


def _new_goal_metrics() -> dict[str, Any]:
    return {
        "cases": 0,
        "activated_cases": 0,
        "variants": Counter(),
        "operation_sequences": set(),
        "program_templates": set(),
        "interaction_signatures": set(),
        "boundary_profiles": set(),
        "data_shapes": set(),
        "data_values": set(),
        "combined_signatures": set(),
    }


def _observe_case(
    metrics: dict[str, Any],
    trace: Mapping[str, Any],
    case: Any,
    preflight: Any,
) -> None:
    goal_id = str((trace.get("selected_goal", {}) or {}).get("goal_id", "") or "")
    activation = dict(trace.get("semantic_activation", {}) or {})
    variant_id = str((trace.get("builder_variant", {}) or {}).get("variant_id", "") or "")
    boundary_profile = str(
        (trace.get("boundary_application", {}) or {}).get("profile_id", "") or ""
    )
    operation_sequence = tuple(op_kind(operation) for operation in case.program.operations)
    program_template = _program_template_signature(case)
    interaction_edges = tuple(
        sorted(
            {
                f"{left}>{right}"
                for left, right in zip(operation_sequence, operation_sequence[1:])
            }
        )
    )
    interaction_signature = short_canonical_hash(interaction_edges, 16)
    data_shape = _data_shape_signature(case)
    data_values = _data_value_signature(case)
    combined = short_canonical_hash(
        {
            "goal": goal_id,
            "variant": variant_id,
            "ops": operation_sequence,
            "template": program_template,
            "boundary": boundary_profile,
            "shape": data_shape,
            "values": data_values,
        },
        24,
    )

    metrics["cases"] += 1
    metrics["activated_cases"] += int(
        activation.get("evaluation_status") == "activated"
    )
    metrics["preflight_valid_cases"] += int(bool(preflight.valid))
    metrics["preflight_fallback_cases"] += int(bool(preflight.fallback_used))
    metrics["variants"].add(variant_id)
    metrics["operation_sequences"].add(operation_sequence)
    metrics["program_templates"].add(program_template)
    metrics["interaction_signatures"].add(interaction_signature)
    metrics["interaction_edges"].update(interaction_edges)
    metrics["boundary_profiles"].add(boundary_profile)
    metrics["data_shapes"].add(data_shape)
    metrics["data_values"].add(data_values)
    metrics["combined_signatures"].add(combined)

    goal = metrics["by_goal"][goal_id]
    goal["cases"] += 1
    goal["activated_cases"] += int(
        activation.get("evaluation_status") == "activated"
    )
    goal["variants"][variant_id] += 1
    goal["operation_sequences"].add(operation_sequence)
    goal["program_templates"].add(program_template)
    goal["interaction_signatures"].add(interaction_signature)
    goal["boundary_profiles"].add(boundary_profile)
    goal["data_shapes"].add(data_shape)
    goal["data_values"].add(data_values)
    goal["combined_signatures"].add(combined)


def _finalize_arm_metrics(metrics: dict[str, Any], *, cases: int) -> dict[str, Any]:
    observed_cases = int(metrics["cases"])
    total_cpu_ns = int(metrics["generation_cpu_ns"]) + int(metrics["preflight_cpu_ns"])
    total_wall_ns = int(metrics["generation_wall_ns"]) + int(metrics["preflight_wall_ns"])
    return {
        "cases": observed_cases,
        "requested_cases": int(cases),
        "activated_cases": int(metrics["activated_cases"]),
        "activation_rate": _rate(metrics["activated_cases"], observed_cases),
        "preflight_valid_cases": int(metrics["preflight_valid_cases"]),
        "preflight_valid_rate": _rate(metrics["preflight_valid_cases"], observed_cases),
        "preflight_fallback_cases": int(metrics["preflight_fallback_cases"]),
        "generation_cpu_seconds": int(metrics["generation_cpu_ns"]) / 1e9,
        "generation_wall_seconds": int(metrics["generation_wall_ns"]) / 1e9,
        "preflight_cpu_seconds": int(metrics["preflight_cpu_ns"]) / 1e9,
        "preflight_wall_seconds": int(metrics["preflight_wall_ns"]) / 1e9,
        "total_cpu_seconds": total_cpu_ns / 1e9,
        "total_wall_seconds": total_wall_ns / 1e9,
        "cpu_microseconds_per_case": (
            total_cpu_ns / max(1, observed_cases) / 1_000.0
        ),
        "wall_microseconds_per_case": (
            total_wall_ns / max(1, observed_cases) / 1_000.0
        ),
        "variant_count": len(metrics["variants"]),
        "variants": sorted(metrics["variants"]),
        "operation_sequence_count": len(metrics["operation_sequences"]),
        "program_template_count": len(metrics["program_templates"]),
        "interaction_signature_count": len(metrics["interaction_signatures"]),
        "interaction_edge_count": len(metrics["interaction_edges"]),
        "interaction_edges": sorted(metrics["interaction_edges"]),
        "boundary_profile_count": len(metrics["boundary_profiles"]),
        "boundary_profiles": sorted(metrics["boundary_profiles"]),
        "data_shape_count": len(metrics["data_shapes"]),
        "data_value_count": len(metrics["data_values"]),
        "combined_signature_count": len(metrics["combined_signatures"]),
        "by_goal": {
            goal_id: {
                "cases": int(goal["cases"]),
                "activated_cases": int(goal["activated_cases"]),
                "variants": sorted(goal["variants"]),
                "variant_counts": dict(sorted(goal["variants"].items())),
                "operation_sequence_count": len(goal["operation_sequences"]),
                "program_template_count": len(goal["program_templates"]),
                "interaction_signature_count": len(goal["interaction_signatures"]),
                "boundary_profile_count": len(goal["boundary_profiles"]),
                "boundary_profiles": sorted(goal["boundary_profiles"]),
                "data_shape_count": len(goal["data_shapes"]),
                "data_value_count": len(goal["data_values"]),
                "combined_signature_count": len(goal["combined_signatures"]),
            }
            for goal_id, goal in sorted(metrics["by_goal"].items())
        },
    }


def _program_template_signature(case: Any) -> str:
    normalized = [
        _normalize_template_value(operation.to_dict(), key="")
        for operation in case.program.operations
    ]
    return short_canonical_hash(normalized, 24)


def _normalize_template_value(value: Any, *, key: str) -> Any:
    if isinstance(value, Mapping):
        return {
            str(item_key): _normalize_template_value(item, key=str(item_key))
            for item_key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_template_value(item, key=key) for item in value]
    if key in {"n", "value", "fallback", "then", "else"}:
        return f"<{_value_category(value)}>"
    return value


def _data_shape_signature(case: Any) -> str:
    payload = []
    for table in case.tables:
        row_keys = [
            tuple(_stable_value(row.get(column.name)) for column in table.columns)
            for row in table.rows
        ]
        payload.append(
            {
                "columns": [
                    (column.type, bool(column.nullable)) for column in table.columns
                ],
                "rows": len(table.rows),
                "nulls": sum(
                    row.get(column.name) is None
                    for row in table.rows
                    for column in table.columns
                ),
                "duplicate_rows": len(row_keys) - len(set(row_keys)),
            }
        )
    return short_canonical_hash(payload, 24)


def _data_value_signature(case: Any) -> str:
    payload = []
    for table in case.tables:
        columns = []
        for column in table.columns:
            values = [row.get(column.name) for row in table.rows]
            category_counts = Counter(_value_category(value) for value in values)
            columns.append(
                {
                    "type": column.type,
                    "categories": dict(sorted(category_counts.items())),
                    "unique_bucket": _count_bucket(
                        len({_stable_value(value) for value in values})
                    ),
                }
            )
        payload.append(columns)
    return short_canonical_hash(payload, 24)


def _value_category(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool_true" if value else "bool_false"
    if isinstance(value, int):
        if value == 0:
            return "int_zero"
        magnitude = abs(value)
        if magnitude >= 2**63 - 1:
            bucket = "int64_edge"
        elif magnitude >= 2**53 - 1:
            bucket = "precision_edge"
        elif magnitude >= 1_000:
            bucket = "large"
        else:
            bucket = "small"
        return f"int_{'negative' if value < 0 else 'positive'}_{bucket}"
    if isinstance(value, float):
        if math.isnan(value):
            return "float_nan"
        if math.isinf(value):
            return "float_negative_inf" if value < 0 else "float_positive_inf"
        if value == 0.0:
            return "float_negative_zero" if math.copysign(1.0, value) < 0 else "float_zero"
        return "float_negative" if value < 0 else "float_positive"
    if isinstance(value, str):
        if value == "":
            return "str_empty"
        if value.strip() != value:
            return "str_outer_space"
        if len(value) <= 4:
            return "str_short"
        if len(value) <= 16:
            return "str_medium"
        return "str_long"
    return type(value).__name__


def _stable_value(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value):
            return ("float", "nan")
        if value == 0.0:
            return (
                "float",
                "negative_zero" if math.copysign(1.0, value) < 0 else "zero",
            )
        return ("float", repr(value))
    if isinstance(value, (str, int, bool)) or value is None:
        return (type(value).__name__, value)
    return (type(value).__name__, repr(value))


def _compact_generation_identity(generated: Any) -> dict[str, Any]:
    if generated is None or generated.case is None:
        return {
            "constructible": False,
            "skip_reason": "" if generated is None else generated.trace.get("skip_reason", ""),
        }
    trace = generated.trace
    return {
        "constructible": True,
        "goal": str((trace.get("selected_goal", {}) or {}).get("goal_id", "") or ""),
        "variant": str((trace.get("builder_variant", {}) or {}).get("variant_id", "") or ""),
        "activation": _activation_status(generated),
        "operation_sequence": [
            op_kind(operation) for operation in generated.case.program.operations
        ],
        "boundary": str(
            (trace.get("boundary_application", {}) or {}).get("profile_id", "") or ""
        ),
        "shape": _data_shape_signature(generated.case),
        "values": _data_value_signature(generated.case),
    }


def _activation_status(generated: Any) -> str:
    if generated is None:
        return "not_evaluated"
    return str(
        (generated.trace.get("semantic_activation", {}) or {}).get(
            "evaluation_status", "not_evaluated"
        )
        or "not_evaluated"
    )


def _count_bucket(value: int) -> str:
    count = max(0, int(value))
    if count <= 1:
        return str(count)
    if count <= 3:
        return "2-3"
    if count <= 7:
        return "4-7"
    if count <= 15:
        return "8-15"
    return "16+"


def _rate(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator) / float(denominator)
