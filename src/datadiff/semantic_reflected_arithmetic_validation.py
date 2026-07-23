from __future__ import annotations

import hashlib
import json
import random
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from datadiff.backends import make_backend
from datadiff.canonical_bug_corpus import DEFAULT_CANONICAL_BUG_CORPUS_MANIFEST
from datadiff.config import ExperimentConfig
from datadiff.execution import BackendExecutor
from datadiff.goal_first import generate_goal_first_case
from datadiff.method_arms import arm_difference, method_arm
from datadiff.preflight import preflight_case
from datadiff.semantic_reflected_arithmetic_witness import (
    REFLECTED_ARITHMETIC_CELL_COUNT,
    REFLECTED_ARITHMETIC_ROOT_ID,
    ReflectedArithmeticWitnessBitmap,
    generate_reflected_arithmetic_witness_case,
)
from datadiff.util import load_json


REFLECTED_ARITHMETIC_CANONICAL_RECALL_SCHEMA_VERSION = (
    "polars-reflected-arithmetic-canonical-recall-v1"
)
REFLECTED_ARITHMETIC_BACKEND_SCREENING_SCHEMA_VERSION = (
    "polars-reflected-arithmetic-backend-screening-v1"
)
CONTROL_ARM = "p8_candidate_v1"
TREATMENT_ARM = "p8_semantic_witness_v5"


def run_reflected_arithmetic_canonical_recall(
    *,
    manifest_path: Path = DEFAULT_CANONICAL_BUG_CORPUS_MANIFEST,
) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    root = next(
        item
        for item in manifest.get("confirmed_roots", [])
        if str(item.get("root_id", "")) == REFLECTED_ARITHMETIC_ROOT_ID
    )
    import polars as pl

    expected_bug_present = _expected_bug_state(
        root,
        polars_version=pl.__version__,
    )
    bitmap = ReflectedArithmeticWitnessBitmap()
    adapters = {
        name: make_backend(name) for name in ("polars", "polars_lazy")
    }
    executor = BackendExecutor(
        ["polars", "polars_lazy"],
        backend_instances=adapters,
        allow_backend_factory_fallback=False,
    )
    generation_failures: list[dict[str, Any]] = []
    backend_failures: list[dict[str, Any]] = []
    cells: list[dict[str, Any]] = []
    try:
        for seed in range(REFLECTED_ARITHMETIC_CELL_COUNT):
            generated = generate_reflected_arithmetic_witness_case(seed)
            case = generated.case
            preflight = preflight_case(
                case,
                enable_validation=True,
                enable_repair=True,
            )
            observation = bitmap.observe_case(case)
            if (
                generated.trace["semantic_activation"]["evaluation_status"]
                != "activated"
                or not preflight.valid
                or preflight.fallback_used
                or not observation.registered
            ):
                generation_failures.append(
                    {
                        "seed": seed,
                        "activation": generated.trace["semantic_activation"],
                        "preflight": preflight.to_dict(),
                        "bitmap": observation.to_dict(),
                    }
                )
            raw, normalized = executor.execute(
                case,
                _screening_config(TREATMENT_ARM),
            )
            eager = normalized["polars"]
            lazy = normalized["polars_lazy"]
            eager_mismatch = (
                eager.status == "ok"
                and eager.rows == [[True]]
            )
            lazy_control = (
                lazy.status == "ok"
                and lazy.rows == [[False]]
            )
            if eager.status != "ok" or lazy.status != "ok":
                backend_failures.append(
                    {
                        "seed": seed,
                        "polars": raw["polars"],
                        "polars_lazy": raw["polars_lazy"],
                    }
                )
            cells.append(
                {
                    "seed": seed,
                    "cell_index": observation.cell_index,
                    "axes": observation.axes,
                    "polars_status": eager.status,
                    "polars_lazy_status": lazy.status,
                    "eager_mismatch": eager_mismatch,
                    "lazy_control": lazy_control,
                }
            )
    finally:
        for adapter in adapters.values():
            adapter.close()

    bitmap_snapshot = bitmap.snapshot(include_data=True)
    observed_bug_present = any(row["eager_mismatch"] for row in cells)
    all_eager_match_version_state = (
        all(row["eager_mismatch"] for row in cells)
        if expected_bug_present is True
        else all(not row["eager_mismatch"] for row in cells)
        if expected_bug_present is False
        else False
    )
    gate = {
        "all_30_cells_activate_and_preflight": not generation_failures,
        "exact_family_bitmap_complete": (
            bitmap_snapshot["observed_count"]
            == REFLECTED_ARITHMETIC_CELL_COUNT
            and bitmap_snapshot["unregistered_observation_count"] == 0
        ),
        "backend_failures_zero": not backend_failures,
        "lazy_direct_controls_all_false": all(
            row["lazy_control"] for row in cells
        ),
        "canonical_version_state_is_registered": expected_bug_present is not None,
        "observed_state_matches_canonical_version": (
            expected_bug_present is not None
            and observed_bug_present is expected_bug_present
        ),
        "all_cells_match_registered_version_state": all_eager_match_version_state,
    }
    gate["passed"] = all(gate.values())
    return {
        "schema_version": REFLECTED_ARITHMETIC_CANONICAL_RECALL_SCHEMA_VERSION,
        "design": {
            "root_id": REFLECTED_ARITHMETIC_ROOT_ID,
            "manifest": str(manifest_path),
            "polars_version": pl.__version__,
            "backends": ["polars", "polars_lazy"],
            "generation_runtime_corpus_io": False,
            "canonical_manifest_read_stage": "validation_only",
            "per_case_file_output": False,
            "case_retention": False,
            "normalized_result_retention": False,
        },
        "expected_bug_present": expected_bug_present,
        "observed_bug_present": observed_bug_present,
        "cells": cells,
        "bitmap": bitmap_snapshot,
        "generation_failure_count": len(generation_failures),
        "generation_failures": generation_failures[:20],
        "backend_failure_count": len(backend_failures),
        "backend_failures": backend_failures[:20],
        "gate": gate,
    }


def run_reflected_arithmetic_backend_screening(
    *,
    seed_start: int = 17_500_000,
    replicates: int = 3,
    cases_per_replicate: int = 60,
    replicate_stride: int = 10_000,
    backends: Sequence[str] = ("polars", "polars_lazy"),
    max_cpu_ratio: float = 1.25,
    max_wall_ratio: float = 1.25,
    bootstrap_resamples: int = 5_000,
    bootstrap_seed: int = 20_260_717,
) -> dict[str, Any]:
    if int(replicates) < 3:
        raise ValueError("backend screening requires at least three replicates")
    if int(cases_per_replicate) < REFLECTED_ARITHMETIC_CELL_COUNT:
        raise ValueError(
            "backend screening requires at least "
            f"{REFLECTED_ARITHMETIC_CELL_COUNT} cases per replicate"
        )
    backend_names = tuple(dict.fromkeys(str(name) for name in backends if str(name)))
    if backend_names != ("polars", "polars_lazy"):
        raise ValueError("v5 screening requires the ordered polars,polars_lazy pair")

    _warm_backend_pair(backend_names)
    arm_metrics = {
        arm: _new_screening_metrics() for arm in (CONTROL_ARM, TREATMENT_ARM)
    }
    replicate_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    row_hasher = hashlib.sha256()
    for replicate in range(int(replicates)):
        start = int(seed_start) + replicate * int(replicate_stride)
        arm_order = (
            (CONTROL_ARM, TREATMENT_ARM)
            if replicate % 2 == 0
            else (TREATMENT_ARM, CONTROL_ARM)
        )
        by_arm: dict[str, dict[str, Any]] = {}
        for arm in arm_order:
            adapters = {name: make_backend(name) for name in backend_names}
            executor = BackendExecutor(
                list(backend_names),
                backend_instances=adapters,
                allow_backend_factory_fallback=False,
            )
            bitmap = ReflectedArithmeticWitnessBitmap() if arm == TREATMENT_ARM else None
            local = _new_screening_metrics()
            started_wall = time.perf_counter_ns()
            started_cpu = time.process_time_ns()
            try:
                for seed in range(start, start + int(cases_per_replicate)):
                    case = _generate_screening_case(seed, arm)
                    preflight = preflight_case(
                        case,
                        enable_validation=True,
                        enable_repair=True,
                    )
                    if not preflight.valid or preflight.fallback_used:
                        local["preflight_failures"] += 1
                        failures.append(
                            {
                                "replicate": replicate,
                                "arm": arm,
                                "seed": seed,
                                "stage": "preflight",
                                "preflight": preflight.to_dict(),
                            }
                        )
                        continue
                    case = preflight.case
                    if bitmap is not None:
                        observation = bitmap.observe_case(case)
                        if not observation.registered:
                            failures.append(
                                {
                                    "replicate": replicate,
                                    "arm": arm,
                                    "seed": seed,
                                    "stage": "bitmap",
                                    "observation": observation.to_dict(),
                                }
                            )
                    raw, normalized = executor.execute(
                        case,
                        _screening_config(arm),
                    )
                    local["input_preparation_count"] += 1
                    local["cases"] += 1
                    activation = case.metadata.get("semantic_activation", {})
                    local["activation_statuses"][
                        str(
                            activation.get("evaluation_status", "not_evaluated")
                            or "not_evaluated"
                        )
                    ] += 1
                    statuses: dict[str, str] = {}
                    observations: dict[str, Any] = {}
                    for backend_name in backend_names:
                        result = normalized[backend_name]
                        status = str(result.status or "unknown")
                        statuses[backend_name] = status
                        observations[backend_name] = (
                            result.rows[0][0]
                            if status == "ok" and result.rows and result.rows[0]
                            else None
                        )
                        local["backend_calls"] += 1
                        local["backend_statuses"][backend_name][status] += 1
                        local["backend_duration_ms"][backend_name] += float(
                            raw[backend_name].get("duration_ms", 0.0) or 0.0
                        )
                        if status != "ok":
                            failures.append(
                                {
                                    "replicate": replicate,
                                    "arm": arm,
                                    "seed": seed,
                                    "stage": "backend",
                                    "backend": backend_name,
                                    "status": status,
                                    "error_type": raw[backend_name].get(
                                        "error_type", ""
                                    ),
                                    "error": raw[backend_name].get("error", ""),
                                }
                            )
                    local["result_pairs"][
                        f"polars={observations.get('polars')}|"
                        f"polars_lazy={observations.get('polars_lazy')}"
                    ] += 1
                    row_hasher.update(
                        json.dumps(
                            {
                                "replicate": replicate,
                                "arm": arm,
                                "seed": seed,
                                "statuses": statuses,
                                "observations": observations,
                            },
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode("utf-8")
                    )
                    row_hasher.update(b"\n")
                    del normalized
            finally:
                for adapter in adapters.values():
                    adapter.close()
            local["process_cpu_ns"] = time.process_time_ns() - started_cpu
            local["wall_ns"] = time.perf_counter_ns() - started_wall
            if bitmap is not None:
                local["family_bitmap"] = bitmap.snapshot(include_data=True)
            _merge_screening_metrics(arm_metrics[arm], local)
            by_arm[arm] = _finalize_screening_metrics(local)
        replicate_rows.append(
            {
                "replicate": replicate,
                "seed_start": start,
                "seed_end": start + int(cases_per_replicate) - 1,
                "arm_order": list(arm_order),
                "arms": by_arm,
            }
        )

    finalized = {
        arm: _finalize_screening_metrics(metrics)
        for arm, metrics in arm_metrics.items()
    }
    ratio_rows = [
        {
            "control_cpu": row["arms"][CONTROL_ARM]["process_cpu_seconds"],
            "treatment_cpu": row["arms"][TREATMENT_ARM]["process_cpu_seconds"],
            "control_wall": row["arms"][CONTROL_ARM]["wall_seconds"],
            "treatment_wall": row["arms"][TREATMENT_ARM]["wall_seconds"],
        }
        for row in replicate_rows
    ]
    cpu = _paired_ratio_bootstrap(
        ratio_rows,
        control_key="control_cpu",
        treatment_key="treatment_cpu",
        resamples=bootstrap_resamples,
        seed=bootstrap_seed,
    )
    wall = _paired_ratio_bootstrap(
        ratio_rows,
        control_key="control_wall",
        treatment_key="treatment_wall",
        resamples=bootstrap_resamples,
        seed=bootstrap_seed + 1,
    )
    treatment = finalized[TREATMENT_ARM]
    parent_difference = arm_difference(
        method_arm(CONTROL_ARM),
        method_arm(TREATMENT_ARM),
    )
    treatment_bitmaps = [
        row["arms"][TREATMENT_ARM].get("family_bitmap", {})
        for row in replicate_rows
    ]
    gate = {
        "single_generation_dimension": parent_difference
        == {"generation_mode": ("goal_first", "goal_first_witness_v5")},
        "all_cases_complete": all(
            metrics["cases"] == int(replicates) * int(cases_per_replicate)
            for metrics in finalized.values()
        ),
        "preflight_failures_zero": not any(
            metrics["preflight_failures"] for metrics in finalized.values()
        ),
        "backend_failures_zero": not failures,
        "one_input_preparation_per_case": all(
            metrics["input_preparation_count"] == metrics["cases"]
            for metrics in finalized.values()
        ),
        "treatment_family_bitmap_complete": all(
            bitmap.get("observed_count") == REFLECTED_ARITHMETIC_CELL_COUNT
            and bitmap.get("unregistered_observation_count") == 0
            for bitmap in treatment_bitmaps
        ),
        "all_treatment_cells_reproduce_eager_vs_lazy": (
            treatment["result_pairs"]
            == {f"polars=True|polars_lazy=False": treatment["cases"]}
        ),
        "cpu_ratio_ci_upper_within_limit": (
            cpu["confidence_interval_95"] is not None
            and cpu["confidence_interval_95"][1] <= float(max_cpu_ratio)
        ),
        "wall_ratio_ci_upper_within_limit": (
            wall["confidence_interval_95"] is not None
            and wall["confidence_interval_95"][1] <= float(max_wall_ratio)
        ),
    }
    gate["passed"] = all(gate.values())
    return {
        "schema_version": REFLECTED_ARITHMETIC_BACKEND_SCREENING_SCHEMA_VERSION,
        "design": {
            "control_arm": CONTROL_ARM,
            "treatment_arm": TREATMENT_ARM,
            "changed_dimension": "generation_mode",
            "arm_difference": {
                key: list(value) for key, value in parent_difference.items()
            },
            "seed_start": int(seed_start),
            "replicates": int(replicates),
            "cases_per_replicate_per_arm": int(cases_per_replicate),
            "replicate_stride": int(replicate_stride),
            "backends": list(backend_names),
            "per_case_file_output": False,
            "case_retention": False,
            "normalized_result_retention": False,
            "input_preparation_policy": "once_per_case_shared_across_backends",
            "native_probe_materialization_policy": "polars_eager_adapter_only",
            "thread_environment": {
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
                "POLARS_MAX_THREADS": "1",
                "RAYON_NUM_THREADS": "1",
            },
            "streaming_row_digest": row_hasher.hexdigest(),
            "performance_limits": {
                "cpu_ratio_ci_upper_max": float(max_cpu_ratio),
                "wall_ratio_ci_upper_max": float(max_wall_ratio),
            },
        },
        "replicates": replicate_rows,
        "aggregates": finalized,
        "statistics": {
            "process_cpu_ratio": cpu,
            "wall_ratio": wall,
        },
        "failure_count": len(failures),
        "failures": failures[:50],
        "gate": gate,
    }


def render_reflected_arithmetic_canonical_recall_markdown(
    result: Mapping[str, Any],
) -> str:
    return "\n".join(
        [
            "# Polars Reflected-Arithmetic Canonical Recall",
            "",
            f"- Polars: `{result['design']['polars_version']}`",
            f"- Expected bug present: `{result['expected_bug_present']}`",
            f"- Observed bug present: `{result['observed_bug_present']}`",
            f"- Exact bitmap: `{result['bitmap']['observed_count']}/"
            f"{result['bitmap']['size']}` cells, `{result['bitmap']['byte_size']}` bytes",
            f"- Eager mismatch cells: `{sum(int(row['eager_mismatch']) for row in result['cells'])}`",
            f"- Lazy control cells: `{sum(int(row['lazy_control']) for row in result['cells'])}`",
            f"- Gate: `{'PASS' if result['gate']['passed'] else 'FAIL'}`",
            "",
        ]
    )


def render_reflected_arithmetic_backend_screening_markdown(
    result: Mapping[str, Any],
) -> str:
    control = result["aggregates"][CONTROL_ARM]
    treatment = result["aggregates"][TREATMENT_ARM]
    cpu = result["statistics"]["process_cpu_ratio"]
    wall = result["statistics"]["wall_ratio"]
    return "\n".join(
        [
            "# Polars Reflected-Arithmetic Backend Screening",
            "",
            f"- Gate: `{'PASS' if result['gate']['passed'] else 'FAIL'}`",
            f"- Process-CPU ratio: `{cpu['point_estimate']:.4f}`, 95% CI "
            f"`{cpu['confidence_interval_95']}`",
            f"- Wall ratio: `{wall['point_estimate']:.4f}`, 95% CI "
            f"`{wall['confidence_interval_95']}`",
            f"- Treatment bitmap: `{REFLECTED_ARITHMETIC_CELL_COUNT}/"
            f"{REFLECTED_ARITHMETIC_CELL_COUNT}` cells, `4` bytes",
            "",
            "| Arm | Cases | Backend calls | Input preparations | CPU s | Wall s |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
            f"| `{CONTROL_ARM}` | {control['cases']} | {control['backend_calls']} | "
            f"{control['input_preparation_count']} | {control['process_cpu_seconds']:.4f} | "
            f"{control['wall_seconds']:.4f} |",
            f"| `{TREATMENT_ARM}` | {treatment['cases']} | {treatment['backend_calls']} | "
            f"{treatment['input_preparation_count']} | {treatment['process_cpu_seconds']:.4f} | "
            f"{treatment['wall_seconds']:.4f} |",
            "",
            "Per-case files and retained normalized results are disabled.",
            "",
        ]
    )


def _expected_bug_state(
    root: Mapping[str, Any],
    *,
    polars_version: str,
) -> bool | None:
    expected_version = f"polars=={polars_version}"
    for observation in root.get("version_observations", []) or []:
        versions = observation.get("versions", {})
        if str(versions.get("polars", "") or "") == polars_version:
            return bool(observation.get("expected_bug_present", False))
    if expected_version in set(root.get("affected_versions", []) or []):
        return True
    if expected_version in set(root.get("fixed_versions", []) or []):
        return False
    return None


def _screening_config(arm: str) -> ExperimentConfig:
    return ExperimentConfig(
        method_arm=arm,
        enable_feedback=False,
        enable_artifact=False,
        enable_reducer=False,
        candidate_recheck_count=0,
        enable_parallel_backend_execution=False,
        enable_backend_session_reuse=False,
        enable_metamorphic_oracle=False,
        log_level="minimal",
    )


def _generate_screening_case(seed: int, arm: str):
    if arm == TREATMENT_ARM:
        return generate_reflected_arithmetic_witness_case(seed).case
    generated = generate_goal_first_case(
        seed,
        profile="common",
        boundary_mode="fault_model_targeted",
        collect_activation_evidence=True,
        require_semantic_activation=False,
    )
    if generated.case is None:
        raise RuntimeError(
            "control goal-first generation failed: "
            f"{generated.trace.get('skip_reason', '')}"
        )
    return generated.case


def _warm_backend_pair(backend_names: tuple[str, ...]) -> None:
    adapters = {name: make_backend(name) for name in backend_names}
    executor = BackendExecutor(
        list(backend_names),
        backend_instances=adapters,
        allow_backend_factory_fallback=False,
    )
    try:
        executor.execute(
            generate_reflected_arithmetic_witness_case(0).case,
            _screening_config(TREATMENT_ARM),
        )
    finally:
        for adapter in adapters.values():
            adapter.close()


def _new_screening_metrics() -> dict[str, Any]:
    return {
        "cases": 0,
        "input_preparation_count": 0,
        "backend_calls": 0,
        "preflight_failures": 0,
        "process_cpu_ns": 0,
        "wall_ns": 0,
        "backend_duration_ms": defaultdict(float),
        "backend_statuses": defaultdict(Counter),
        "activation_statuses": Counter(),
        "result_pairs": Counter(),
        "family_bitmap": {},
    }


def _merge_screening_metrics(target: dict[str, Any], source: Mapping[str, Any]) -> None:
    for key in (
        "cases",
        "input_preparation_count",
        "backend_calls",
        "preflight_failures",
        "process_cpu_ns",
        "wall_ns",
    ):
        target[key] += int(source.get(key, 0) or 0)
    target["activation_statuses"].update(source.get("activation_statuses", {}))
    target["result_pairs"].update(source.get("result_pairs", {}))
    for backend, duration in (source.get("backend_duration_ms", {}) or {}).items():
        target["backend_duration_ms"][backend] += float(duration)
    for backend, counts in (source.get("backend_statuses", {}) or {}).items():
        target["backend_statuses"][backend].update(counts)
    source_bitmap = source.get("family_bitmap", {})
    if isinstance(source_bitmap, Mapping) and source_bitmap:
        target["family_bitmap"] = dict(source_bitmap)


def _finalize_screening_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "cases": int(metrics.get("cases", 0) or 0),
        "input_preparation_count": int(
            metrics.get("input_preparation_count", 0) or 0
        ),
        "backend_calls": int(metrics.get("backend_calls", 0) or 0),
        "preflight_failures": int(metrics.get("preflight_failures", 0) or 0),
        "process_cpu_seconds": float(metrics.get("process_cpu_ns", 0) or 0)
        / 1_000_000_000.0,
        "wall_seconds": float(metrics.get("wall_ns", 0) or 0)
        / 1_000_000_000.0,
        "backend_duration_ms": {
            str(backend): float(duration)
            for backend, duration in sorted(
                (metrics.get("backend_duration_ms", {}) or {}).items()
            )
        },
        "backend_statuses": {
            str(backend): dict(sorted(counts.items()))
            for backend, counts in sorted(
                (metrics.get("backend_statuses", {}) or {}).items()
            )
        },
        "activation_statuses": dict(
            sorted((metrics.get("activation_statuses", {}) or {}).items())
        ),
        "result_pairs": dict(
            sorted((metrics.get("result_pairs", {}) or {}).items())
        ),
        "family_bitmap": dict(metrics.get("family_bitmap", {}) or {}),
    }


def _paired_ratio_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    *,
    control_key: str,
    treatment_key: str,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    resolved = list(rows)
    control_total = sum(float(row[control_key]) for row in resolved)
    treatment_total = sum(float(row[treatment_key]) for row in resolved)
    point = treatment_total / control_total if control_total > 0 else None
    if not resolved or point is None:
        return {"point_estimate": point, "confidence_interval_95": None}
    rng = random.Random(int(seed))
    samples = []
    for _index in range(max(1, int(resamples))):
        selected = [resolved[rng.randrange(len(resolved))] for _row in resolved]
        control = sum(float(row[control_key]) for row in selected)
        treatment = sum(float(row[treatment_key]) for row in selected)
        if control > 0:
            samples.append(treatment / control)
    samples.sort()
    return {
        "point_estimate": point,
        "confidence_interval_95": _percentile_interval(samples),
        "bootstrap_resamples": int(resamples),
        "bootstrap_seed": int(seed),
    }


def _percentile_interval(values: Sequence[float]) -> list[float] | None:
    if not values:
        return None
    ordered = list(values)
    lower_index = max(0, int(0.025 * (len(ordered) - 1)))
    upper_index = min(len(ordered) - 1, int(0.975 * (len(ordered) - 1)))
    return [ordered[lower_index], ordered[upper_index]]
