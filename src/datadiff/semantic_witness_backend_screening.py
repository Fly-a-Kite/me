"""Aggregate-only Pandas/DuckDB compatibility and cost screening for witness v2."""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping
from typing import Any

from datadiff.backends import make_backend
from datadiff.backends.base import Backend, prepare_tables
from datadiff.compact_novelty import DenseBitmap, MonotonicSeedTracker
from datadiff.goal_first import (
    generate_goal_first_case,
    generation_goal_variants,
    generation_goals,
)
from datadiff.preflight import preflight_case
from datadiff.semantic_trigger_bitmap import SemanticTriggerBitmap


SEMANTIC_WITNESS_BACKEND_SCREENING_SCHEMA_VERSION = (
    "semantic-witness-backend-screening-v1"
)
CONTROL_ARM = "p8_semantic_witness_v1"
TREATMENT_ARM = "p8_semantic_witness_v2"
ARMS = (CONTROL_ARM, TREATMENT_ARM)
DEFAULT_BACKENDS = ("pandas", "duckdb")
BackendFactory = Callable[[str], Backend]


def build_screening_seed_blocks(
    *,
    seed_start: int = 12_000_000,
    replicates: int = 3,
    cases_per_replicate: int = 42,
    replicate_stride: int = 10_000,
    arms: tuple[str, str] = ARMS,
) -> tuple[dict[str, Any], ...]:
    blocks = []
    for replicate in range(max(0, int(replicates))):
        start = int(seed_start) + replicate * int(replicate_stride)
        arm_order = list(arms) if replicate % 2 == 0 else list(reversed(arms))
        blocks.append(
            {
                "replicate": replicate,
                "seed_start": start,
                "seed_end": start + max(0, int(cases_per_replicate)) - 1,
                "cases": max(0, int(cases_per_replicate)),
                "arm_order": arm_order,
            }
        )
    return tuple(blocks)


def run_semantic_witness_backend_screening(
    *,
    seed_start: int = 12_000_000,
    replicates: int = 3,
    cases_per_replicate: int = 42,
    replicate_stride: int = 10_000,
    backends: tuple[str, ...] = DEFAULT_BACKENDS,
    max_cpu_ratio: float = 1.25,
    max_wall_ratio: float = 1.25,
    max_backend_duration_ratio: float = 1.25,
    backend_factory: BackendFactory = make_backend,
    warm_up: bool = True,
    control_arm: str = CONTROL_ARM,
    treatment_arm: str = TREATMENT_ARM,
    control_variant_family: str | None = None,
    treatment_variant_family: str = "v2",
) -> dict[str, Any]:
    """Run a low-I/O status/cost screen with one shared table preparation per case."""

    resolved_control = str(control_arm or "").strip()
    resolved_treatment = str(treatment_arm or "").strip()
    if not resolved_control or not resolved_treatment or resolved_control == resolved_treatment:
        raise ValueError("screening arms must be distinct non-empty identifiers")
    arms = (resolved_control, resolved_treatment)
    arm_variant_families = {
        resolved_control: control_variant_family,
        resolved_treatment: str(treatment_variant_family),
    }
    backend_names = tuple(dict.fromkeys(str(name) for name in backends if str(name)))
    blocks = build_screening_seed_blocks(
        seed_start=seed_start,
        replicates=replicates,
        cases_per_replicate=cases_per_replicate,
        replicate_stride=replicate_stride,
        arms=arms,
    )
    if warm_up and blocks and backend_names:
        _warm_backends(
            seed=int(blocks[0]["seed_start"]),
            backend_names=backend_names,
            backend_factory=backend_factory,
            variant_family=str(treatment_variant_family),
        )

    arm_metrics = {arm: _new_arm_metrics() for arm in arms}
    replicate_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    row_hasher = hashlib.sha256()
    block_completion_bitmaps = {arm: DenseBitmap(len(blocks)) for arm in arms}
    duplicate_seed_observations = Counter()
    trigger_bitmaps = {arm: SemanticTriggerBitmap() for arm in arms}
    for block in blocks:
        replicate = int(block["replicate"])
        replicate_metrics: dict[str, dict[str, Any]] = {}
        for arm in block["arm_order"]:
            variant_family = arm_variant_families[arm]
            diverse = variant_family is not None
            seed_tracker = MonotonicSeedTracker(
                int(block["seed_start"]),
                int(block["cases"]),
            )
            adapters = {
                backend_name: backend_factory(backend_name)
                for backend_name in backend_names
            }
            started_wall = time.perf_counter_ns()
            started_cpu = time.process_time_ns()
            local = _new_arm_metrics()
            try:
                for seed in range(
                    int(block["seed_start"]),
                    int(block["seed_start"]) + int(block["cases"]),
                ):
                    duplicate_seed_observations[arm] += int(
                        not seed_tracker.observe(seed)
                    )
                    generated = generate_goal_first_case(
                        seed,
                        boundary_mode="fault_model_targeted",
                        require_semantic_activation=True,
                        diversity_preserving_witness=diverse,
                        witness_variant_family=(
                            str(variant_family) if variant_family is not None else "v2"
                        ),
                    )
                    if generated.case is None:
                        failure = {
                            "replicate": replicate,
                            "seed": seed,
                            "arm": arm,
                            "stage": "generation",
                            "reason": str(
                                generated.trace.get("skip_reason", "") or ""
                            ),
                        }
                        failures.append(failure)
                        local["generation_failures"] += 1
                        continue
                    preflight = preflight_case(
                        generated.case,
                        enable_validation=True,
                        enable_repair=True,
                    )
                    if not preflight.valid or preflight.fallback_used:
                        failure = {
                            "replicate": replicate,
                            "seed": seed,
                            "arm": arm,
                            "stage": "preflight",
                            "valid": bool(preflight.valid),
                            "fallback_used": bool(preflight.fallback_used),
                            "errors": list(preflight.errors_after),
                        }
                        failures.append(failure)
                        local["preflight_failures"] += 1
                        continue

                    case = preflight.case
                    trigger_bitmaps[arm].observe_trace(generated.trace)
                    goal_id = str(
                        (generated.trace.get("selected_goal", {}) or {}).get(
                            "goal_id", ""
                        )
                        or ""
                    )
                    variant_id = str(
                        (generated.trace.get("builder_variant", {}) or {}).get(
                            "variant_id", ""
                        )
                        or ""
                    )
                    # This is the central efficiency invariant: build the row/
                    # column materialization once and reuse it for every backend.
                    prepared = prepare_tables(case.tables)
                    local["input_preparation_count"] += 1
                    statuses: dict[str, str] = {}
                    for backend_name in backend_names:
                        result = adapters[backend_name].run(prepared, case.program)
                        status = str(result.status or "unknown")
                        statuses[backend_name] = status
                        local["backend_calls"] += 1
                        local["backend_statuses"][backend_name][status] += 1
                        local["backend_duration_ms"][backend_name] += float(
                            result.duration_ms or 0.0
                        )
                        if status == "ok":
                            local["goal_backend_coverage"][goal_id].add(
                                backend_name
                            )
                            local["variant_backend_coverage"][variant_id].add(
                                backend_name
                            )
                        else:
                            failures.append(
                                {
                                    "replicate": replicate,
                                    "seed": seed,
                                    "arm": arm,
                                    "stage": "backend",
                                    "goal_id": goal_id,
                                    "variant_id": variant_id,
                                    "backend": backend_name,
                                    "status": status,
                                    "error_type": str(result.error_type or ""),
                                    "error": str(result.error or "")[:1000],
                                }
                            )
                    local["cases"] += 1
                    local["goals"][goal_id] += 1
                    local["variants"][variant_id] += 1
                    row_hasher.update(
                        json.dumps(
                            {
                                "replicate": replicate,
                                "seed": seed,
                                "arm": arm,
                                "goal": goal_id,
                                "variant": variant_id,
                                "statuses": statuses,
                            },
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode("utf-8")
                    )
                    row_hasher.update(b"\n")
            finally:
                for adapter in adapters.values():
                    adapter.close()
            local["process_cpu_ns"] = time.process_time_ns() - started_cpu
            local["wall_ns"] = time.perf_counter_ns() - started_wall
            if seed_tracker.complete:
                block_completion_bitmaps[arm].observe(replicate)
            _merge_arm_metrics(arm_metrics[arm], local)
            replicate_metrics[arm] = _finalize_arm_metrics(
                local,
                backend_names=backend_names,
            )
        replicate_rows.append(
            {
                "replicate": replicate,
                "seed_start": int(block["seed_start"]),
                "seed_end": int(block["seed_end"]),
                "arm_order": list(block["arm_order"]),
                "arms": replicate_metrics,
                "cpu_ratio": _safe_ratio(
                    replicate_metrics[resolved_treatment]["process_cpu_seconds"],
                    replicate_metrics[resolved_control]["process_cpu_seconds"],
                ),
                "wall_ratio": _safe_ratio(
                    replicate_metrics[resolved_treatment]["wall_seconds"],
                    replicate_metrics[resolved_control]["wall_seconds"],
                ),
            }
        )

    summaries = {
        arm: _finalize_arm_metrics(metrics, backend_names=backend_names)
        for arm, metrics in arm_metrics.items()
    }
    control = summaries[resolved_control]
    treatment = summaries[resolved_treatment]
    cpu_ratio = _safe_ratio(
        treatment["process_cpu_seconds"], control["process_cpu_seconds"]
    )
    wall_ratio = _safe_ratio(treatment["wall_seconds"], control["wall_seconds"])
    duration_ratio = _safe_ratio(
        treatment["backend_duration_ms_total"],
        control["backend_duration_ms_total"],
    )
    expected_calls = sum(int(block["cases"]) for block in blocks) * len(
        backend_names
    )
    expected_preparations = sum(int(block["cases"]) for block in blocks)
    expected_goal_ids = {goal.goal_id for goal in generation_goals()}
    expected_variants = {
        variant.variant_id
        for goal in generation_goals()
        for variant in generation_goal_variants(
            goal,
            variant_family=str(treatment_variant_family),
        )
    }
    backend_set = set(backend_names)
    treatment_goal_coverage = treatment["goal_backend_coverage"]
    treatment_variant_coverage = treatment["variant_backend_coverage"]
    gate = {
        "treatment_generation_failure_count_is_zero": treatment[
            "generation_failures"
        ]
        == 0,
        "treatment_preflight_failure_count_is_zero": treatment[
            "preflight_failures"
        ]
        == 0,
        "treatment_backend_call_count_complete": treatment["backend_calls"]
        == expected_calls,
        "treatment_all_backend_calls_ok": treatment["ok_backend_calls"]
        == expected_calls,
        "all_goals_execute_on_all_backends": expected_goal_ids
        <= set(treatment_goal_coverage)
        and all(
            set(treatment_goal_coverage[goal_id]) == backend_set
            for goal_id in expected_goal_ids
        ),
        "all_variants_execute_on_all_backends": expected_variants
        <= set(treatment_variant_coverage)
        and all(
            set(treatment_variant_coverage[variant_id]) == backend_set
            for variant_id in expected_variants
        ),
        "one_input_preparation_per_case": treatment["input_preparation_count"]
        == expected_preparations,
        "cpu_ratio_within_limit": cpu_ratio is not None
        and cpu_ratio <= float(max_cpu_ratio),
        "wall_ratio_within_limit": wall_ratio is not None
        and wall_ratio <= float(max_wall_ratio),
        "backend_duration_ratio_within_limit": duration_ratio is not None
        and duration_ratio <= float(max_backend_duration_ratio),
        "seed_trackers_have_no_duplicate_observations": all(
            duplicate_seed_observations[arm] == 0 for arm in arms
        ),
        "all_worker_blocks_completed": all(
            bitmap.count == len(blocks)
            for bitmap in block_completion_bitmaps.values()
        ),
        "trigger_bitmaps_have_no_unregistered_dimensions": all(
            bitmap.unregistered_observations == 0
            for bitmap in trigger_bitmaps.values()
        ),
    }
    gate["passed"] = all(gate.values())
    return {
        "schema_version": SEMANTIC_WITNESS_BACKEND_SCREENING_SCHEMA_VERSION,
        "design": {
            "kind": "paired_aggregate_only_backend_compatibility_cost_screening",
            "control_method_arm": resolved_control,
            "treatment_method_arm": resolved_treatment,
            "control_variant_family": control_variant_family,
            "treatment_variant_family": str(treatment_variant_family),
            "backends": list(backend_names),
            "seed_blocks": list(blocks),
            "total_cases_per_arm": expected_preparations,
            "total_backend_calls_per_arm": expected_calls,
            "backend_session_reuse_within_arm": True,
            "input_preparation_policy": "once_per_case_shared_across_backends",
            "case_retention": False,
            "normalized_result_retention": False,
            "per_case_file_output": False,
            "streaming_execution_digest": row_hasher.hexdigest(),
            "seed_tracking_policy": "monotonic_cursor_per_worker_block",
            "block_completion_bitmaps": {
                arm: bitmap.snapshot()
                for arm, bitmap in block_completion_bitmaps.items()
            },
            "claim_boundary": (
                "This screening establishes adapter compatibility and steady-state cost only. "
                "It does not perform oracle adjudication or claim improved fresh bug yield."
            ),
            "performance_limits": {
                "cpu_ratio_max": float(max_cpu_ratio),
                "wall_ratio_max": float(max_wall_ratio),
                "backend_duration_ratio_max": float(max_backend_duration_ratio),
            },
        },
        "summary": {
            "control": control,
            "treatment": treatment,
            "process_cpu_ratio": cpu_ratio,
            "wall_ratio": wall_ratio,
            "backend_duration_ratio": duration_ratio,
            "input_preparation_reuse_saved_count_per_arm": expected_preparations
            * max(0, len(backend_names) - 1),
            "failure_count": len(failures),
            "duplicate_seed_observation_counts": dict(
                sorted(duplicate_seed_observations.items())
            ),
            "trigger_bitmaps": {
                arm: bitmap.snapshot() for arm, bitmap in trigger_bitmaps.items()
            },
        },
        "replicates": replicate_rows,
        "gate": gate,
        "failures": failures[:100],
        "failure_rows_truncated": len(failures) > 100,
    }


def render_semantic_witness_backend_screening_markdown(
    screening: Mapping[str, Any],
) -> str:
    design = screening["design"]
    summary = screening["summary"]
    control = summary["control"]
    treatment = summary["treatment"]
    control_arm = str(design["control_method_arm"])
    treatment_arm = str(design["treatment_method_arm"])
    lines = [
        "# Semantic Witness Backend Screening",
        "",
        "## Design",
        "",
        f"- Backends: `{', '.join(design['backends'])}`",
        f"- Cases per arm: `{design['total_cases_per_arm']}`",
        f"- Backend calls per arm: `{design['total_backend_calls_per_arm']}`",
        "- Input preparation: once per case, reused across both backends.",
        "- Per-case/normalized-result retention and per-case file output: disabled.",
        "- Seed hot path: monotonic cursor per block; checkpoint state: one-bit block bitmap.",
        f"- Exact block bitmap per arm: "
        f"`{design['block_completion_bitmaps'][treatment_arm]['byte_size']}` bytes.",
        "",
        "## Results",
        "",
        "| Arm | OK calls | CPU s | Wall s | Backend ms | Preparations |",
        "|---|---:|---:|---:|---:|---:|",
        f"| `{control_arm}` | {control['ok_backend_calls']}/{control['backend_calls']} | "
        f"{control['process_cpu_seconds']:.3f} | {control['wall_seconds']:.3f} | "
        f"{control['backend_duration_ms_total']:.3f} | {control['input_preparation_count']} |",
        f"| `{treatment_arm}` | {treatment['ok_backend_calls']}/{treatment['backend_calls']} | "
        f"{treatment['process_cpu_seconds']:.3f} | {treatment['wall_seconds']:.3f} | "
        f"{treatment['backend_duration_ms_total']:.3f} | {treatment['input_preparation_count']} |",
        "",
        f"- CPU ratio: `{summary['process_cpu_ratio']:.4f}`",
        f"- Wall ratio: `{summary['wall_ratio']:.4f}`",
        f"- Backend-duration ratio: `{summary['backend_duration_ratio']:.4f}`",
        f"- Avoided duplicate input preparations per arm: "
        f"`{summary['input_preparation_reuse_saved_count_per_arm']}`",
        f"- Treatment goal/backend coverage: `{len(treatment['goal_backend_coverage'])}` goals",
        f"- Treatment variant/backend coverage: `{len(treatment['variant_backend_coverage'])}` variants",
        f"- Exact treatment trigger bitmap: "
        f"`{summary['trigger_bitmaps'][treatment_arm]['byte_size']}` bytes; "
        f"`{summary['trigger_bitmaps'][treatment_arm]['observed_count']}` cells observed.",
        f"- Gate: `{'PASS' if screening['gate']['passed'] else 'FAIL'}`",
        "",
        "## Claim Boundary",
        "",
        design["claim_boundary"],
        "",
    ]
    return "\n".join(lines)


def _warm_backends(
    *,
    seed: int,
    backend_names: tuple[str, ...],
    backend_factory: BackendFactory,
    variant_family: str = "v2",
) -> None:
    generated = generate_goal_first_case(
        seed,
        boundary_mode="fault_model_targeted",
        require_semantic_activation=True,
        diversity_preserving_witness=True,
        witness_variant_family=variant_family,
    )
    if generated.case is None:
        return
    prepared = prepare_tables(generated.case.tables)
    adapters = [backend_factory(name) for name in backend_names]
    try:
        for adapter in adapters:
            adapter.run(prepared, generated.case.program)
    finally:
        for adapter in adapters:
            adapter.close()


def _new_arm_metrics() -> dict[str, Any]:
    return {
        "cases": 0,
        "generation_failures": 0,
        "preflight_failures": 0,
        "backend_calls": 0,
        "input_preparation_count": 0,
        "process_cpu_ns": 0,
        "wall_ns": 0,
        "backend_statuses": defaultdict(Counter),
        "backend_duration_ms": Counter(),
        "goals": Counter(),
        "variants": Counter(),
        "goal_backend_coverage": defaultdict(set),
        "variant_backend_coverage": defaultdict(set),
    }


def _merge_arm_metrics(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key in (
        "cases",
        "generation_failures",
        "preflight_failures",
        "backend_calls",
        "input_preparation_count",
        "process_cpu_ns",
        "wall_ns",
    ):
        target[key] += source[key]
    target["backend_duration_ms"].update(source["backend_duration_ms"])
    target["goals"].update(source["goals"])
    target["variants"].update(source["variants"])
    for backend_name, statuses in source["backend_statuses"].items():
        target["backend_statuses"][backend_name].update(statuses)
    for goal_id, backend_names in source["goal_backend_coverage"].items():
        target["goal_backend_coverage"][goal_id].update(backend_names)
    for variant_id, backend_names in source["variant_backend_coverage"].items():
        target["variant_backend_coverage"][variant_id].update(backend_names)


def _finalize_arm_metrics(
    metrics: dict[str, Any],
    *,
    backend_names: tuple[str, ...],
) -> dict[str, Any]:
    backend_statuses = {
        backend_name: dict(sorted(metrics["backend_statuses"][backend_name].items()))
        for backend_name in backend_names
    }
    ok_calls = sum(
        metrics["backend_statuses"][backend_name]["ok"]
        for backend_name in backend_names
    )
    duration_by_backend = {
        backend_name: float(metrics["backend_duration_ms"][backend_name])
        for backend_name in backend_names
    }
    return {
        "cases": int(metrics["cases"]),
        "generation_failures": int(metrics["generation_failures"]),
        "preflight_failures": int(metrics["preflight_failures"]),
        "backend_calls": int(metrics["backend_calls"]),
        "ok_backend_calls": int(ok_calls),
        "backend_statuses": backend_statuses,
        "backend_duration_ms": duration_by_backend,
        "backend_duration_ms_total": sum(duration_by_backend.values()),
        "input_preparation_count": int(metrics["input_preparation_count"]),
        "process_cpu_seconds": int(metrics["process_cpu_ns"]) / 1e9,
        "wall_seconds": int(metrics["wall_ns"]) / 1e9,
        "cases_per_wall_second": (
            int(metrics["cases"]) / (int(metrics["wall_ns"]) / 1e9)
            if metrics["wall_ns"]
            else 0.0
        ),
        "goals": dict(sorted(metrics["goals"].items())),
        "variants": dict(sorted(metrics["variants"].items())),
        "goal_backend_coverage": {
            goal_id: sorted(names)
            for goal_id, names in sorted(metrics["goal_backend_coverage"].items())
        },
        "variant_backend_coverage": {
            variant_id: sorted(names)
            for variant_id, names in sorted(
                metrics["variant_backend_coverage"].items()
            )
        },
    }


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator) / float(denominator)
