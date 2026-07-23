from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from datadiff.backends import make_backend
from datadiff.canonical_bug_corpus import DEFAULT_CANONICAL_BUG_CORPUS_MANIFEST
from datadiff.execution import BackendExecutor
from datadiff.goal_first import generate_goal_first_case
from datadiff.method_arms import arm_difference, method_arm
from datadiff.preflight import preflight_case
from datadiff.semantic_datafusion_grouped_null_topk_witness import (
    DATAFUSION_GROUPED_NULL_TOPK_CELL_COUNT,
    DATAFUSION_GROUPED_NULL_TOPK_ROOT_ID,
    DataFusionGroupedNullTopKBitmap,
    generate_datafusion_grouped_null_topk_witness_case,
)
from datadiff.semantic_reflected_arithmetic_validation import (
    _finalize_screening_metrics,
    _merge_screening_metrics,
    _new_screening_metrics,
    _paired_ratio_bootstrap,
    _screening_config,
)
from datadiff.util import load_json


DATAFUSION_GROUPED_NULL_TOPK_CANONICAL_RECALL_SCHEMA_VERSION = (
    "datafusion-grouped-null-topk-canonical-recall-v1"
)
DATAFUSION_GROUPED_NULL_TOPK_BACKEND_SCREENING_SCHEMA_VERSION = (
    "datafusion-grouped-null-topk-backend-screening-v1"
)
CONTROL_ARM = "p8_candidate_v1"
TREATMENT_ARM = "p8_semantic_witness_v6"


def run_datafusion_grouped_null_topk_canonical_recall(
    *,
    manifest_path: Path = DEFAULT_CANONICAL_BUG_CORPUS_MANIFEST,
) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    root = next(
        item
        for item in manifest.get("confirmed_roots", [])
        if str(item.get("root_id", ""))
        == DATAFUSION_GROUPED_NULL_TOPK_ROOT_ID
    )
    import datafusion
    import pyarrow as pa

    expected_bug_present = _expected_bug_state(
        root,
        datafusion_version=datafusion.__version__,
        pyarrow_version=pa.__version__,
    )
    bitmap = DataFusionGroupedNullTopKBitmap()
    backend_names = ("pandas", "duckdb", "datafusion")
    adapters = {name: make_backend(name) for name in backend_names}
    executor = BackendExecutor(
        list(backend_names),
        backend_instances=adapters,
        allow_backend_factory_fallback=False,
    )
    generation_failures: list[dict[str, Any]] = []
    backend_failures: list[dict[str, Any]] = []
    cells: list[dict[str, Any]] = []
    try:
        for seed in range(DATAFUSION_GROUPED_NULL_TOPK_CELL_COUNT):
            generated = generate_datafusion_grouped_null_topk_witness_case(seed)
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
            observations: dict[str, bool | None] = {}
            for backend_name in backend_names:
                result = normalized[backend_name]
                observations[backend_name] = (
                    bool(result.rows[0][0])
                    if result.status == "ok"
                    and result.rows
                    and result.rows[0]
                    else None
                )
                if result.status != "ok":
                    backend_failures.append(
                        {
                            "seed": seed,
                            "backend": backend_name,
                            "raw": raw[backend_name],
                        }
                    )
            cells.append(
                {
                    "seed": seed,
                    "cell_index": observation.cell_index,
                    "axes": observation.axes,
                    "observations": observations,
                    "datafusion_mismatch": observations["datafusion"] is True,
                    "controls_false": (
                        observations["pandas"] is False
                        and observations["duckdb"] is False
                    ),
                }
            )
    finally:
        for adapter in adapters.values():
            adapter.close()

    bitmap_snapshot = bitmap.snapshot(include_data=True)
    observed_bug_present = any(row["datafusion_mismatch"] for row in cells)
    all_cells_match_version_state = (
        all(row["datafusion_mismatch"] for row in cells)
        if expected_bug_present is True
        else all(not row["datafusion_mismatch"] for row in cells)
        if expected_bug_present is False
        else False
    )
    gate = {
        "all_12_cells_activate_and_preflight": not generation_failures,
        "exact_family_bitmap_complete": (
            bitmap_snapshot["observed_count"]
            == DATAFUSION_GROUPED_NULL_TOPK_CELL_COUNT
            and bitmap_snapshot["unregistered_observation_count"] == 0
        ),
        "backend_failures_zero": not backend_failures,
        "pandas_duckdb_controls_all_false": all(
            row["controls_false"] for row in cells
        ),
        "canonical_version_state_is_registered": expected_bug_present is not None,
        "observed_state_matches_canonical_version": (
            expected_bug_present is not None
            and observed_bug_present is expected_bug_present
        ),
        "all_cells_match_registered_version_state": all_cells_match_version_state,
    }
    gate["passed"] = all(gate.values())
    return {
        "schema_version": (
            DATAFUSION_GROUPED_NULL_TOPK_CANONICAL_RECALL_SCHEMA_VERSION
        ),
        "design": {
            "root_id": DATAFUSION_GROUPED_NULL_TOPK_ROOT_ID,
            "manifest": str(manifest_path),
            "datafusion_version": datafusion.__version__,
            "pyarrow_version": pa.__version__,
            "backends": list(backend_names),
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


def run_datafusion_grouped_null_topk_backend_screening(
    *,
    seed_start: int = 18_500_000,
    replicates: int = 3,
    cases_per_replicate: int = 60,
    replicate_stride: int = 10_000,
    backends: Sequence[str] = ("pandas", "duckdb", "datafusion"),
    max_cpu_ratio: float = 1.25,
    max_wall_ratio: float = 1.25,
    bootstrap_resamples: int = 5_000,
    bootstrap_seed: int = 20_260_717,
) -> dict[str, Any]:
    if int(replicates) < 3:
        raise ValueError("backend screening requires at least three replicates")
    if int(cases_per_replicate) < DATAFUSION_GROUPED_NULL_TOPK_CELL_COUNT:
        raise ValueError(
            "backend screening requires at least "
            f"{DATAFUSION_GROUPED_NULL_TOPK_CELL_COUNT} cases per replicate"
        )
    backend_names = tuple(
        dict.fromkeys(str(name) for name in backends if str(name))
    )
    if backend_names != ("pandas", "duckdb", "datafusion"):
        raise ValueError(
            "v6 screening requires the ordered pandas,duckdb,datafusion trio"
        )

    _warm_backend_trio(backend_names)
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
            bitmap = (
                DataFusionGroupedNullTopKBitmap()
                if arm == TREATMENT_ARM
                else None
            )
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
                            activation.get(
                                "evaluation_status",
                                "not_evaluated",
                            )
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
                            if status == "ok"
                            and result.rows
                            and result.rows[0]
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
                    pair_key = "|".join(
                        f"{name}={observations.get(name)}"
                        for name in backend_names
                    )
                    local["result_pairs"][pair_key] += 1
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
            "treatment_cpu": row["arms"][TREATMENT_ARM][
                "process_cpu_seconds"
            ],
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
    expected_pair = "pandas=False|duckdb=False|datafusion=True"
    gate = {
        "single_generation_dimension": parent_difference
        == {"generation_mode": ("goal_first", "goal_first_witness_v6")},
        "all_cases_complete": all(
            metrics["cases"]
            == int(replicates) * int(cases_per_replicate)
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
            bitmap.get("observed_count")
            == DATAFUSION_GROUPED_NULL_TOPK_CELL_COUNT
            and bitmap.get("unregistered_observation_count") == 0
            for bitmap in treatment_bitmaps
        ),
        "all_treatment_cells_reproduce_datafusion_vs_controls": (
            treatment["result_pairs"]
            == {expected_pair: treatment["cases"]}
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
        "schema_version": (
            DATAFUSION_GROUPED_NULL_TOPK_BACKEND_SCREENING_SCHEMA_VERSION
        ),
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
            "native_probe_materialization_policy": (
                "datafusion_in_memory_session_only"
            ),
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


def render_datafusion_grouped_null_topk_canonical_recall_markdown(
    result: Mapping[str, Any],
) -> str:
    return "\n".join(
        [
            "# DataFusion Grouped-Null TopK Canonical Recall",
            "",
            f"- DataFusion: `{result['design']['datafusion_version']}`",
            f"- PyArrow: `{result['design']['pyarrow_version']}`",
            f"- Expected bug present: `{result['expected_bug_present']}`",
            f"- Observed bug present: `{result['observed_bug_present']}`",
            f"- Exact bitmap: `{result['bitmap']['observed_count']}/"
            f"{result['bitmap']['size']}` cells, "
            f"`{result['bitmap']['byte_size']}` bytes",
            f"- DataFusion mismatch cells: "
            f"`{sum(int(row['datafusion_mismatch']) for row in result['cells'])}`",
            f"- Pandas/DuckDB control cells: "
            f"`{sum(int(row['controls_false']) for row in result['cells'])}`",
            f"- Gate: `{'PASS' if result['gate']['passed'] else 'FAIL'}`",
            "",
        ]
    )


def render_datafusion_grouped_null_topk_backend_screening_markdown(
    result: Mapping[str, Any],
) -> str:
    control = result["aggregates"][CONTROL_ARM]
    treatment = result["aggregates"][TREATMENT_ARM]
    cpu = result["statistics"]["process_cpu_ratio"]
    wall = result["statistics"]["wall_ratio"]
    return "\n".join(
        [
            "# DataFusion Grouped-Null TopK Backend Screening",
            "",
            f"- Gate: `{'PASS' if result['gate']['passed'] else 'FAIL'}`",
            f"- Process-CPU ratio: `{cpu['point_estimate']:.4f}`, 95% CI "
            f"`{cpu['confidence_interval_95']}`",
            f"- Wall ratio: `{wall['point_estimate']:.4f}`, 95% CI "
            f"`{wall['confidence_interval_95']}`",
            f"- Treatment bitmap: `{DATAFUSION_GROUPED_NULL_TOPK_CELL_COUNT}/"
            f"{DATAFUSION_GROUPED_NULL_TOPK_CELL_COUNT}` cells, `2` bytes",
            "",
            "| Arm | Cases | Backend calls | Input preparations | CPU s | Wall s |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
            f"| `{CONTROL_ARM}` | {control['cases']} | "
            f"{control['backend_calls']} | {control['input_preparation_count']} | "
            f"{control['process_cpu_seconds']:.4f} | "
            f"{control['wall_seconds']:.4f} |",
            f"| `{TREATMENT_ARM}` | {treatment['cases']} | "
            f"{treatment['backend_calls']} | "
            f"{treatment['input_preparation_count']} | "
            f"{treatment['process_cpu_seconds']:.4f} | "
            f"{treatment['wall_seconds']:.4f} |",
            "",
            "Per-case files and retained normalized results are disabled.",
            "",
        ]
    )


def _expected_bug_state(
    root: Mapping[str, Any],
    *,
    datafusion_version: str,
    pyarrow_version: str,
) -> bool | None:
    for observation in root.get("version_observations", []) or []:
        versions = observation.get("versions", {})
        if (
            str(versions.get("datafusion", "") or "") == datafusion_version
            and str(versions.get("pyarrow", "") or "") == pyarrow_version
        ):
            return bool(observation.get("expected_bug_present", False))
    return None


def _generate_screening_case(seed: int, arm: str):
    if arm == TREATMENT_ARM:
        return generate_datafusion_grouped_null_topk_witness_case(seed).case
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


def _warm_backend_trio(backend_names: tuple[str, ...]) -> None:
    adapters = {name: make_backend(name) for name in backend_names}
    executor = BackendExecutor(
        list(backend_names),
        backend_instances=adapters,
        allow_backend_factory_fallback=False,
    )
    try:
        executor.execute(
            generate_datafusion_grouped_null_topk_witness_case(0).case,
            _screening_config(TREATMENT_ARM),
        )
    finally:
        for adapter in adapters.values():
            adapter.close()
