"""Fail-closed coverage audit for the all-confirmed-root witness portfolio."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

from datadiff.backends import make_backend
from datadiff.canonical_bug_corpus import (
    DEFAULT_CANONICAL_BUG_CORPUS_MANIFEST,
    load_canonical_bug_corpus,
)
from datadiff.ccs_ir import case_to_ccs_ir
from datadiff.config import ExperimentConfig
from datadiff.execution import BackendExecutor
from datadiff.family_witness_registry import (
    GLOBAL_FAMILY_WITNESS_GENERATION_MODE,
    GLOBAL_FAMILY_WITNESS_METHOD_ARM_ID,
    GLOBAL_FAMILY_WITNESS_TARGET_SUITE,
    FamilyWitnessBitmap,
    FamilyWitnessRegistration,
    family_witness_registrations,
    generate_global_family_witness_case,
    global_family_witness_cell_count,
    global_family_witness_selection,
)
from datadiff.method_arms import DEFAULT_METHOD_ARM_ID, arm_difference, method_arm
from datadiff.preflight import preflight_case
from datadiff.targets import resolve_target_backends, target_spec


FAMILY_WITNESS_COVERAGE_SCHEMA_VERSION = "family-witness-coverage-audit-v1"
DEFAULT_CANONICAL_MANIFEST = DEFAULT_CANONICAL_BUG_CORPUS_MANIFEST


def audit_family_witness_coverage(
    repo_root: Path,
    *,
    manifest_path: Path | None = None,
    execute_backends: bool = False,
) -> dict[str, Any]:
    """Audit the complete canonical root set; any missing layer fails closed."""

    root = repo_root.resolve()
    manifest = (manifest_path or root / DEFAULT_CANONICAL_MANIFEST).resolve()
    corpus = load_canonical_bug_corpus(manifest)
    canonical_root_ids = tuple(
        str(item.get("root_id", "") or "")
        for item in corpus.get("confirmed_roots", ())
    )
    registrations = family_witness_registrations()
    registrations_by_root: dict[str, list[FamilyWitnessRegistration]] = defaultdict(list)
    for registration in registrations:
        registrations_by_root[registration.root_id].append(registration)

    global_errors: list[str] = []
    if len(set(canonical_root_ids)) != len(canonical_root_ids):
        global_errors.append("canonical_manifest_has_duplicate_root_ids")
    canonical_set = set(canonical_root_ids)
    registered_set = set(registrations_by_root)
    missing_roots = sorted(canonical_set - registered_set)
    extra_roots = sorted(registered_set - canonical_set)
    if missing_roots:
        global_errors.append("missing_registered_roots:" + ",".join(missing_roots))
    if extra_roots:
        global_errors.append("registered_roots_not_canonical:" + ",".join(extra_roots))
    duplicate_roots = sorted(
        root_id
        for root_id, root_registrations in registrations_by_root.items()
        if len(root_registrations) != 1
    )
    if duplicate_roots:
        global_errors.append(
            "root_registration_cardinality_not_one:" + ",".join(duplicate_roots)
        )
    if DEFAULT_METHOD_ARM_ID != "p8_candidate_v1":
        global_errors.append(f"default_method_arm_changed:{DEFAULT_METHOD_ARM_ID}")

    global_backends = tuple(
        resolve_target_backends(target_suite=GLOBAL_FAMILY_WITNESS_TARGET_SUITE)
    )
    root_rows: list[dict[str, Any]] = []
    family_bitmaps: dict[str, FamilyWitnessBitmap] = {}
    for root_id in canonical_root_ids:
        root_errors: list[str] = []
        matches = registrations_by_root.get(root_id, [])
        if len(matches) != 1:
            root_rows.append(
                {
                    "root_id": root_id,
                    "covered": False,
                    "errors": [f"registration_count:{len(matches)}"],
                    "cell_count": 0,
                    "activated_cell_count": 0,
                    "preflight_valid_cell_count": 0,
                    "bitmap_observed_cell_count": 0,
                }
            )
            continue
        registration = matches[0]
        family_bitmaps[registration.family_id] = FamilyWitnessBitmap(registration)
        suite_backends = tuple(
            resolve_target_backends(target_suite=registration.target_suite)
        )
        if suite_backends != registration.backends:
            root_errors.append(
                "target_suite_backend_mismatch:"
                f"suite={suite_backends},registered={registration.backends}"
            )
        if not set(registration.backends) <= set(global_backends):
            root_errors.append("family_backends_not_in_global_suite")
        try:
            arm = method_arm(registration.method_arm_id)
        except ValueError as exc:
            root_errors.append(f"method_arm_missing:{exc}")
        else:
            if arm.generation_mode != registration.generation_mode:
                root_errors.append(
                    "generation_mode_arm_mismatch:"
                    f"{arm.generation_mode}!={registration.generation_mode}"
                )
            if arm.parent_arm_id != "p8_candidate_v1":
                root_errors.append(f"unexpected_parent_arm:{arm.parent_arm_id}")
            difference = arm_difference(method_arm("p8_candidate_v1"), arm)
            if set(difference) != {"generation_mode"}:
                root_errors.append(
                    "arm_changes_non_generation_dimensions:"
                    + ",".join(sorted(difference))
                )

        activated_cells = 0
        preflight_cells = 0
        bitmap_cells = 0
        capability_cells = 0
        builder_errors: list[str] = []
        bitmap = family_bitmaps[registration.family_id]
        for seed in range(registration.cell_count):
            expected_cell, expected_axes = registration.cell_for_seed(seed)
            try:
                case = registration.generate_case(seed)
            except Exception as exc:  # noqa: BLE001 - audit records exact layer failure
                builder_errors.append(
                    f"cell={seed}:builder:{type(exc).__name__}:{exc}"
                )
                continue
            family_witness = case.metadata.get("family_witness", {})
            if not isinstance(family_witness, Mapping):
                root_errors.append(f"cell={seed}:family_witness_not_mapping")
                continue
            axes_raw = family_witness.get("axes", {})
            axes = (
                {str(key): str(value) for key, value in axes_raw.items()}
                if isinstance(axes_raw, Mapping)
                else {}
            )
            if int(family_witness.get("cell_index", -1)) != expected_cell:
                root_errors.append(f"cell={seed}:declared_cell_index_mismatch")
            if axes != expected_axes:
                root_errors.append(f"cell={seed}:declared_axes_mismatch")
            if family_witness.get("root_ids") != [root_id]:
                root_errors.append(f"cell={seed}:root_guidance_mismatch")
            if bool(family_witness.get("canonical_case_replay", True)):
                root_errors.append(f"cell={seed}:canonical_case_replay_enabled")
            if bool(family_witness.get("runtime_corpus_io", True)):
                root_errors.append(f"cell={seed}:runtime_corpus_io_enabled")

            preflight = preflight_case(
                case,
                enable_validation=True,
                enable_repair=True,
            )
            if preflight.valid and not preflight.fallback_used:
                preflight_cells += 1
            else:
                root_errors.append(
                    f"cell={seed}:preflight_invalid_or_fallback:"
                    f"{preflight.errors_after}"
                )
            activation = case.metadata.get("semantic_activation", {})
            if (
                isinstance(activation, Mapping)
                and activation.get("evaluation_status") == "activated"
                and activation.get("semantically_activated") is True
            ):
                activated_cells += 1
            else:
                root_errors.append(f"cell={seed}:semantic_activation_not_activated")
            observation = bitmap.observe_case(case)
            if observation.registered and observation.cell_index == expected_cell:
                bitmap_cells += 1
            else:
                root_errors.append(
                    f"cell={seed}:bitmap_rejected:{observation.reason}"
                )
            required = set(case_to_ccs_ir(case).required_capabilities)
            missing_by_backend = {
                backend: sorted(required - set(target_spec(backend).capabilities))
                for backend in registration.backends
            }
            missing_by_backend = {
                backend: missing
                for backend, missing in missing_by_backend.items()
                if missing
            }
            if missing_by_backend:
                root_errors.append(
                    f"cell={seed}:backend_capability_gap:{missing_by_backend}"
                )
            else:
                capability_cells += 1

        root_errors.extend(builder_errors)
        snapshot = bitmap.snapshot(include_data=True)
        if int(snapshot.get("observed_count", -1)) != registration.cell_count:
            root_errors.append(
                "bitmap_incomplete:"
                f"{snapshot.get('observed_count')}/{registration.cell_count}"
            )
        if int(snapshot.get("unregistered_observation_count", -1)) != 0:
            root_errors.append("bitmap_has_unregistered_observations")
        root_rows.append(
            {
                "root_id": root_id,
                "family_id": registration.family_id,
                "goal_id": registration.goal_id,
                "generation_mode": registration.generation_mode,
                "method_arm_id": registration.method_arm_id,
                "target_suite": registration.target_suite,
                "backends": list(registration.backends),
                "builder": registration.builder,
                "axes": [
                    {"name": name, "values": list(values)}
                    for name, values in registration.axes
                ],
                "cell_count": registration.cell_count,
                "activated_cell_count": activated_cells,
                "preflight_valid_cell_count": preflight_cells,
                "bitmap_observed_cell_count": bitmap_cells,
                "backend_capability_cell_count": capability_cells,
                "bitmap": snapshot,
                "covered": not root_errors,
                "errors": root_errors,
            }
        )

    global_portfolio = _audit_global_portfolio(
        canonical_root_ids,
        registrations,
        global_backends,
    )
    if not global_portfolio["complete"]:
        global_errors.extend(global_portfolio["errors"])

    backend_execution = {
        "requested": bool(execute_backends),
        "complete": not execute_backends,
        "executed_cell_count": 0,
        "backend_result_count": 0,
        "target_true_counts": {},
        "errors": [],
    }
    if execute_backends:
        backend_execution = _execute_global_portfolio(global_backends)
        if not backend_execution["complete"]:
            global_errors.extend(backend_execution["errors"])

    covered_roots = sum(bool(row.get("covered")) for row in root_rows)
    total_roots = len(canonical_root_ids)
    complete = bool(
        not global_errors
        and covered_roots == total_roots
        and total_roots > 0
        and global_portfolio["complete"]
        and backend_execution["complete"]
    )
    return {
        "schema_version": FAMILY_WITNESS_COVERAGE_SCHEMA_VERSION,
        "manifest": str(manifest.relative_to(root)),
        "fail_closed": True,
        "canonical_root_ids": list(canonical_root_ids),
        "registered_root_ids": [registration.root_id for registration in registrations],
        "default_method_arm": DEFAULT_METHOD_ARM_ID,
        "global_generation_mode": GLOBAL_FAMILY_WITNESS_GENERATION_MODE,
        "global_method_arm_id": GLOBAL_FAMILY_WITNESS_METHOD_ARM_ID,
        "global_target_suite": GLOBAL_FAMILY_WITNESS_TARGET_SUITE,
        "global_backends": list(global_backends),
        "roots": root_rows,
        "global_portfolio": global_portfolio,
        "backend_execution": backend_execution,
        "summary": {
            "canonical_root_count": total_roots,
            "registered_root_count": len(registrations),
            "covered_root_count": covered_roots,
            "root_coverage": f"{covered_roots}/{total_roots}",
            "family_cell_count": sum(
                registration.cell_count for registration in registrations
            ),
            "global_cell_count": global_family_witness_cell_count(),
            "complete": complete,
        },
        "errors": global_errors,
    }


def _audit_global_portfolio(
    canonical_root_ids: tuple[str, ...],
    registrations: tuple[FamilyWitnessRegistration, ...],
    global_backends: tuple[str, ...],
) -> dict[str, Any]:
    errors: list[str] = []
    expected_total = sum(registration.cell_count for registration in registrations)
    if expected_total != global_family_witness_cell_count():
        errors.append("global_cell_count_differs_from_registry_sum")
    root_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    observed_pairs: set[tuple[str, int]] = set()
    for global_index in range(expected_total):
        selection = global_family_witness_selection(global_index)
        case = generate_global_family_witness_case(global_index)
        global_witness = case.metadata.get("global_family_witness", {})
        family_witness = case.metadata.get("family_witness", {})
        activation = case.metadata.get("semantic_activation", {})
        if case.metadata.get("generation_mode") != GLOBAL_FAMILY_WITNESS_GENERATION_MODE:
            errors.append(f"global_cell={global_index}:generation_mode_mismatch")
        if not isinstance(global_witness, Mapping):
            errors.append(f"global_cell={global_index}:global_metadata_missing")
            continue
        if int(global_witness.get("global_cell_index", -1)) != global_index:
            errors.append(f"global_cell={global_index}:global_index_mismatch")
        if int(global_witness.get("global_cell_count", -1)) != expected_total:
            errors.append(f"global_cell={global_index}:global_count_mismatch")
        if not isinstance(family_witness, Mapping):
            errors.append(f"global_cell={global_index}:family_metadata_missing")
            continue
        family_id = str(family_witness.get("family_id", "") or "")
        root_ids = family_witness.get("root_ids", [])
        root_id = str(root_ids[0]) if isinstance(root_ids, list) and len(root_ids) == 1 else ""
        cell_index = int(family_witness.get("cell_index", -1))
        if family_id != selection.registration.family_id:
            errors.append(f"global_cell={global_index}:selected_family_mismatch")
        if root_id != selection.registration.root_id:
            errors.append(f"global_cell={global_index}:selected_root_mismatch")
        if cell_index != selection.family_cell_index:
            errors.append(f"global_cell={global_index}:selected_cell_mismatch")
        if (
            not isinstance(activation, Mapping)
            or activation.get("evaluation_status") != "activated"
        ):
            errors.append(f"global_cell={global_index}:activation_not_preserved")
        if bool(family_witness.get("canonical_case_replay", True)):
            errors.append(f"global_cell={global_index}:canonical_replay_enabled")
        if bool(family_witness.get("runtime_corpus_io", True)):
            errors.append(f"global_cell={global_index}:runtime_corpus_io_enabled")
        root_counts[root_id] += 1
        family_counts[family_id] += 1
        observed_pairs.add((family_id, cell_index))
    expected_pairs = {
        (registration.family_id, cell_index)
        for registration in registrations
        for cell_index in range(registration.cell_count)
    }
    if observed_pairs != expected_pairs:
        errors.append("global_family_cell_pairs_not_exact")
    if set(root_counts) != set(canonical_root_ids):
        errors.append("global_root_set_not_canonical")
    if tuple(global_backends) != tuple(
        resolve_target_backends(target_suite=GLOBAL_FAMILY_WITNESS_TARGET_SUITE)
    ):
        errors.append("global_backend_suite_resolution_unstable")
    return {
        "generation_mode": GLOBAL_FAMILY_WITNESS_GENERATION_MODE,
        "method_arm_id": GLOBAL_FAMILY_WITNESS_METHOD_ARM_ID,
        "target_suite": GLOBAL_FAMILY_WITNESS_TARGET_SUITE,
        "backends": list(global_backends),
        "cell_count": expected_total,
        "observed_family_cell_pair_count": len(observed_pairs),
        "root_cell_counts": dict(sorted(root_counts.items())),
        "family_cell_counts": dict(sorted(family_counts.items())),
        "complete": not errors,
        "errors": errors,
    }


def _execute_global_portfolio(global_backends: tuple[str, ...]) -> dict[str, Any]:
    errors: list[str] = []
    adapters = {backend: make_backend(backend) for backend in global_backends}
    executor = BackendExecutor(
        list(global_backends),
        backend_instances=adapters,
        allow_backend_factory_fallback=False,
    )
    config = ExperimentConfig(
        method_arm=GLOBAL_FAMILY_WITNESS_METHOD_ARM_ID,
        enable_parallel_backend_execution=False,
        enable_backend_session_reuse=True,
    )
    target_true_counts: Counter[str] = Counter()
    backend_result_count = 0
    executed_cells = 0
    try:
        for global_index in range(global_family_witness_cell_count()):
            case = generate_global_family_witness_case(global_index)
            _raw, normalized = executor.execute(case, config)
            executed_cells += 1
            backend_result_count += len(normalized)
            for backend, result in normalized.items():
                if result.status != "ok":
                    errors.append(
                        f"global_cell={global_index}:backend={backend}:"
                        f"status={result.status}:{result.error_type}:{result.error}"
                    )
            operation = case.program.operations[-1]
            if str(operation.get("op", "") or "").endswith("_probe"):
                target_backend = str(operation.get("target_backend", "") or "")
                if not target_backend:
                    target_backend = {
                        "series_reflected_arithmetic_probe": "polars",
                        "datafusion_grouped_null_topk_probe": "datafusion",
                    }.get(str(operation.get("op", "") or ""), "")
                for backend, result in normalized.items():
                    if result.status != "ok" or len(result.rows) != 1:
                        continue
                    value = result.rows[0][0] if result.rows[0] else None
                    if backend != target_backend and value is not False:
                        errors.append(
                            f"global_cell={global_index}:backend={backend}:"
                            f"non_target_probe_control={value!r}"
                        )
                    if backend == target_backend and value is True:
                        target_true_counts[case.metadata["family_witness"]["root_ids"][0]] += 1
    finally:
        for adapter in adapters.values():
            adapter.close()
    return {
        "requested": True,
        "complete": not errors,
        "executed_cell_count": executed_cells,
        "backend_result_count": backend_result_count,
        "target_true_counts": dict(sorted(target_true_counts.items())),
        "errors": errors,
    }


def render_family_witness_coverage_report(result: Mapping[str, Any]) -> str:
    summary = result["summary"]
    lines = [
        "# All-Family Semantic Witness Coverage Audit",
        "",
        f"- Root coverage: `{summary['root_coverage']}`",
        f"- Exact family cells: `{summary['family_cell_count']}`",
        f"- Global portfolio cells: `{summary['global_cell_count']}`",
        f"- Backend execution requested: `{result['backend_execution']['requested']}`",
        f"- Fail-closed audit complete: `{summary['complete']}`",
        "",
        "| Canonical root | Family | Cells | Activation | Preflight | Bitmap | Covered |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in result["roots"]:
        lines.append(
            f"| `{row['root_id']}` | `{row.get('family_id', '')}` | "
            f"{row.get('cell_count', 0)} | "
            f"{row.get('activated_cell_count', 0)} | "
            f"{row.get('preflight_valid_cell_count', 0)} | "
            f"{row.get('bitmap_observed_cell_count', 0)} | "
            f"`{row.get('covered', False)}` |"
        )
    if result.get("errors"):
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- `{error}`" for error in result["errors"])
    lines.append("")
    return "\n".join(lines)
