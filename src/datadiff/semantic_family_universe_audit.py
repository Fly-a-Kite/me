"""Fail-closed audit for :mod:`datadiff.semantic_family_universe`."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Any

import datadiff.family_witness_registry as witness_registry
from datadiff.backends import make_backend
from datadiff.ccs_ir import case_to_ccs_ir
from datadiff.config import ExperimentConfig
from datadiff.execution import BackendExecutor
from datadiff.method_arms import DEFAULT_METHOD_ARM_ID, arm_difference, method_arm
from datadiff.preflight import preflight_case
from datadiff.semantic_family_universe import (
    SEMANTIC_FAMILY_GLOBAL_V2_GENERATION_MODE,
    SEMANTIC_FAMILY_GLOBAL_V2_METHOD_ARM_ID,
    SEMANTIC_FAMILY_UNIVERSE_SCHEMA_VERSION,
    SEMANTIC_FAMILY_UNIVERSE_TARGET_SUITE,
    semantic_family_definitions,
    semantic_family_universe_manifest,
)
from datadiff.targets import resolve_target_backends, target_spec


SEMANTIC_FAMILY_UNIVERSE_AUDIT_SCHEMA_VERSION = "semantic-family-universe-audit-v1"


def audit_semantic_family_universe(
    *,
    execute_builders: bool = False,
    execute_backends: bool = False,
) -> dict[str, Any]:
    manifest = semantic_family_universe_manifest()
    definitions = semantic_family_definitions()
    backends = tuple(
        resolve_target_backends(target_suite=SEMANTIC_FAMILY_UNIVERSE_TARGET_SUITE)
    )
    errors: list[str] = []

    family_ids = [item.family_id for item in definitions]
    mechanism_ids = [item.mechanism_id for item in definitions]
    if len(set(family_ids)) != len(family_ids):
        errors.append("duplicate_family_ids")
    if len(set(mechanism_ids)) != len(mechanism_ids):
        errors.append("duplicate_mechanism_ids")
    if DEFAULT_METHOD_ARM_ID != "p8_candidate_v1":
        errors.append(f"default_method_arm_changed:{DEFAULT_METHOD_ARM_ID}")

    try:
        global_v2_arm = method_arm(SEMANTIC_FAMILY_GLOBAL_V2_METHOD_ARM_ID)
    except ValueError as exc:
        global_v2_arm = None
        errors.append(f"global_v2_method_arm_missing:{exc}")
    else:
        if global_v2_arm.generation_mode != SEMANTIC_FAMILY_GLOBAL_V2_GENERATION_MODE:
            errors.append("global_v2_generation_mode_mismatch")
        if global_v2_arm.parent_arm_id != "p8_candidate_v1":
            errors.append(f"global_v2_parent_mismatch:{global_v2_arm.parent_arm_id}")
        if set(arm_difference(method_arm("p8_candidate_v1"), global_v2_arm)) != {
            "generation_mode"
        }:
            errors.append("global_v2_changes_non_generation_dimensions")

    target_counts: Counter[str] = Counter()
    family_rows: list[dict[str, Any]] = []
    registered_ids: set[str] = set()
    all_registrations_fn = getattr(
        witness_registry,
        "all_family_witness_registrations",
        witness_registry.family_witness_registrations,
    )
    all_registrations = tuple(all_registrations_fn())
    registrations_by_id = {item.family_id: item for item in all_registrations}

    for definition in definitions:
        row_errors: list[str] = []
        target_counts[definition.target_backend] += 1
        if definition.target_backend not in backends:
            row_errors.append("target_backend_outside_universe")
        outside_controls = sorted(set(definition.control_backends) - set(backends))
        if outside_controls:
            row_errors.append("control_backends_outside_universe:" + ",".join(outside_controls))

        capability_rows: list[dict[str, Any]] = []
        for backend in definition.execution_backends:
            decision = target_spec(backend).capability_decision(
                required_tokens=definition.required_capabilities
            )
            capability_rows.append(
                {
                    "backend": backend,
                    "role": definition.backend_role(backend),
                    **decision.to_dict(),
                }
            )
            if not decision.supported:
                row_errors.append(
                    f"backend_capability_gap:{backend}:" + ",".join(decision.missing)
                )

        registration = registrations_by_id.get(definition.family_id)
        builder_summary: dict[str, Any] = {
            "requested": bool(execute_builders),
            "complete": not execute_builders,
            "generated_cell_count": 0,
            "activated_cell_count": 0,
            "preflight_repaired_cell_count": 0,
            "preflight_valid_cell_count": 0,
            "required_capability_cell_count": 0,
            "errors": [],
        }
        if registration is None:
            row_errors.append("family_registration_missing")
        else:
            registered_ids.add(registration.family_id)
            if registration.root_id != definition.root_id:
                row_errors.append("root_or_target_identity_mismatch")
            if registration.target_backend != definition.target_backend:
                row_errors.append("target_backend_registration_mismatch")
            if set(registration.backends) != set(definition.execution_backends):
                row_errors.append(
                    "execution_backends_registration_mismatch:"
                    f"{registration.backends}!={definition.execution_backends}"
                )
            if registration.cell_count < definition.minimum_cell_count:
                row_errors.append(
                    f"insufficient_cells:{registration.cell_count}/"
                    f"{definition.minimum_cell_count}"
                )
            try:
                arm = method_arm(registration.method_arm_id)
            except ValueError as exc:
                row_errors.append(f"registration_method_arm_missing:{exc}")
            else:
                if arm.generation_mode != registration.generation_mode:
                    row_errors.append("registration_generation_mode_arm_mismatch")
                if arm.parent_arm_id != "p8_candidate_v1":
                    row_errors.append(f"registration_parent_mismatch:{arm.parent_arm_id}")
                if set(arm_difference(method_arm("p8_candidate_v1"), arm)) != {
                    "generation_mode"
                }:
                    row_errors.append("registration_changes_non_generation_dimensions")
            if definition.source == "coverage_expansion" and (
                registration.generation_mode != SEMANTIC_FAMILY_GLOBAL_V2_GENERATION_MODE
                or registration.method_arm_id != SEMANTIC_FAMILY_GLOBAL_V2_METHOD_ARM_ID
            ):
                row_errors.append("expansion_family_not_owned_by_global_v2")
            if execute_builders:
                builder_summary = _audit_registration_builders(registration)
                if not builder_summary["complete"]:
                    row_errors.extend(builder_summary["errors"])

        family_rows.append(
            {
                **definition.manifest(),
                "registered": registration is not None,
                "registered_cell_count": 0 if registration is None else registration.cell_count,
                "capabilities": capability_rows,
                "builder_audit": builder_summary,
                "covered": not row_errors,
                "errors": row_errors,
            }
        )

    missing_target_backends = sorted(set(backends) - set(target_counts))
    if missing_target_backends:
        errors.append("backends_without_native_target:" + ",".join(missing_target_backends))
    extra_registered = sorted(set(registrations_by_id) - set(family_ids))
    if extra_registered:
        errors.append("registered_families_outside_universe:" + ",".join(extra_registered))

    global_portfolio = _audit_global_v2_portfolio(definitions)
    if not global_portfolio["complete"]:
        errors.extend(global_portfolio["errors"])

    backend_execution = {
        "requested": bool(execute_backends),
        "complete": not execute_backends,
        "executed_cell_count": 0,
        "backend_result_count": 0,
        "backend_statuses": {},
        "target_true_counts": {},
        "errors": [],
    }
    if execute_backends:
        backend_execution = _execute_global_v2_portfolio(backends)
        if not backend_execution["complete"]:
            errors.extend(backend_execution["errors"])

    covered_count = sum(bool(row["covered"]) for row in family_rows)
    errors.extend(
        f"family={row['family_id']}:{error}"
        for row in family_rows
        for error in row["errors"]
    )
    complete = bool(
        not errors
        and covered_count == len(definitions)
        and global_v2_arm is not None
        and global_portfolio["complete"]
        and backend_execution["complete"]
    )
    return {
        "schema_version": SEMANTIC_FAMILY_UNIVERSE_AUDIT_SCHEMA_VERSION,
        "universe_schema_version": SEMANTIC_FAMILY_UNIVERSE_SCHEMA_VERSION,
        "universe_digest": manifest["digest"],
        "target_suite": SEMANTIC_FAMILY_UNIVERSE_TARGET_SUITE,
        "backends": list(backends),
        "default_method_arm": DEFAULT_METHOD_ARM_ID,
        "global_v2_generation_mode": SEMANTIC_FAMILY_GLOBAL_V2_GENERATION_MODE,
        "global_v2_method_arm_id": SEMANTIC_FAMILY_GLOBAL_V2_METHOD_ARM_ID,
        "execute_builders": bool(execute_builders),
        "execute_backends": bool(execute_backends),
        "families": family_rows,
        "global_portfolio": global_portfolio,
        "backend_execution": backend_execution,
        "summary": {
            "backend_count": len(backends),
            "backend_target_count": len(target_counts),
            "family_count": len(definitions),
            "registered_family_count": len(registered_ids),
            "covered_family_count": covered_count,
            "coverage_expansion_family_count": sum(
                item.source == "coverage_expansion" for item in definitions
            ),
            "required_backend_pair_count": manifest["summary"][
                "required_backend_pair_count"
            ],
            "complete": complete,
        },
        "errors": errors,
    }


def _audit_registration_builders(registration: Any) -> dict[str, Any]:
    errors: list[str] = []
    generated = 0
    activated = 0
    preflight_valid = 0
    preflight_repaired = 0
    capability_valid = 0
    for seed in range(registration.cell_count):
        try:
            case = registration.generate_case(seed)
        except Exception as exc:  # noqa: BLE001 - audit records exact failure
            errors.append(f"cell={seed}:builder:{type(exc).__name__}:{exc}")
            continue
        generated += 1
        witness = case.metadata.get("family_witness", {})
        if not isinstance(witness, Mapping):
            errors.append(f"cell={seed}:family_witness_not_mapping")
            continue
        expected_index, expected_axes = registration.cell_for_seed(seed)
        if witness.get("cell_index") != expected_index:
            errors.append(f"cell={seed}:cell_index_mismatch")
        if witness.get("axes") != expected_axes:
            errors.append(f"cell={seed}:axes_mismatch")
        activation = case.metadata.get("semantic_activation", {})
        if (
            isinstance(activation, Mapping)
            and activation.get("evaluation_status") == "activated"
            and activation.get("semantically_activated") is True
        ):
            activated += 1
        else:
            errors.append(f"cell={seed}:activation_not_activated")
        preflight = preflight_case(case, enable_validation=True, enable_repair=True)
        if preflight.repaired:
            preflight_repaired += 1
            errors.append(f"cell={seed}:preflight_repaired")
        if preflight.valid and not preflight.fallback_used:
            preflight_valid += 1
        else:
            errors.append(f"cell={seed}:preflight_invalid_or_fallback")
        required = set(case_to_ccs_ir(case).required_capabilities)
        missing = {
            backend: sorted(required - set(target_spec(backend).capabilities))
            for backend in registration.backends
        }
        missing = {backend: values for backend, values in missing.items() if values}
        if missing:
            errors.append(f"cell={seed}:case_capability_gap:{missing}")
        else:
            capability_valid += 1
    return {
        "requested": True,
        "complete": not errors,
        "generated_cell_count": generated,
        "activated_cell_count": activated,
        "preflight_repaired_cell_count": preflight_repaired,
        "preflight_valid_cell_count": preflight_valid,
        "required_capability_cell_count": capability_valid,
        "errors": errors,
    }


def _audit_global_v2_portfolio(definitions: tuple[Any, ...]) -> dict[str, Any]:
    errors: list[str] = []
    registrations_fn = getattr(
        witness_registry,
        "global_family_witness_registrations",
        None,
    )
    cell_count_fn = getattr(witness_registry, "global_family_witness_cell_count", None)
    selection_fn = getattr(witness_registry, "global_family_witness_selection", None)
    if registrations_fn is None or cell_count_fn is None or selection_fn is None:
        return {
            "generation_mode": SEMANTIC_FAMILY_GLOBAL_V2_GENERATION_MODE,
            "registered_family_ids": [],
            "cell_count": 0,
            "complete": False,
            "errors": ["global_v2_portfolio_api_missing"],
        }
    try:
        registrations = tuple(
            registrations_fn(SEMANTIC_FAMILY_GLOBAL_V2_GENERATION_MODE)
        )
        cell_count = int(
            cell_count_fn(SEMANTIC_FAMILY_GLOBAL_V2_GENERATION_MODE)
        )
    except (KeyError, TypeError, ValueError) as exc:
        return {
            "generation_mode": SEMANTIC_FAMILY_GLOBAL_V2_GENERATION_MODE,
            "registered_family_ids": [],
            "cell_count": 0,
            "complete": False,
            "errors": [f"global_v2_portfolio_missing:{type(exc).__name__}:{exc}"],
        }
    expected_ids = [item.family_id for item in definitions]
    observed_ids = [item.family_id for item in registrations]
    if observed_ids != expected_ids:
        errors.append("global_v2_family_order_or_membership_mismatch")
    expected_cells = sum(item.cell_count for item in registrations)
    if cell_count != expected_cells:
        errors.append("global_v2_cell_count_mismatch")
    observed_pairs: set[tuple[str, int]] = set()
    try:
        for global_index in range(cell_count):
            selection = selection_fn(
                global_index,
                generation_mode=SEMANTIC_FAMILY_GLOBAL_V2_GENERATION_MODE,
            )
            observed_pairs.add(
                (selection.registration.family_id, selection.family_cell_index)
            )
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"global_v2_selection_failed:{type(exc).__name__}:{exc}")
    expected_pairs = {
        (registration.family_id, cell_index)
        for registration in registrations
        for cell_index in range(registration.cell_count)
    }
    if observed_pairs != expected_pairs:
        errors.append("global_v2_family_cell_pairs_not_exact")
    return {
        "generation_mode": SEMANTIC_FAMILY_GLOBAL_V2_GENERATION_MODE,
        "registered_family_ids": observed_ids,
        "cell_count": cell_count,
        "observed_family_cell_pair_count": len(observed_pairs),
        "complete": not errors,
        "errors": errors,
    }


def _execute_global_v2_portfolio(backends: tuple[str, ...]) -> dict[str, Any]:
    errors: list[str] = []
    adapters = {backend: make_backend(backend) for backend in backends}
    executors: dict[tuple[str, ...], BackendExecutor] = {}
    config = ExperimentConfig(
        method_arm=SEMANTIC_FAMILY_GLOBAL_V2_METHOD_ARM_ID,
        enable_parallel_backend_execution=False,
        enable_backend_session_reuse=True,
    )
    statuses: Counter[tuple[str, str]] = Counter()
    target_true_counts: Counter[str] = Counter()
    backend_result_count = 0
    executed_cells = 0
    try:
        cell_count = witness_registry.global_family_witness_cell_count(
            SEMANTIC_FAMILY_GLOBAL_V2_GENERATION_MODE
        )
        for global_index in range(cell_count):
            case = witness_registry.generate_global_family_witness_case(
                global_index,
                generation_mode=SEMANTIC_FAMILY_GLOBAL_V2_GENERATION_MODE,
            )
            family_witness = case.metadata.get("family_witness", {})
            family_id = str(family_witness.get("family_id", "") or "")
            registration = witness_registry.family_witness_registration(family_id)
            execution_backends = tuple(registration.backends)
            executor = executors.get(execution_backends)
            if executor is None:
                executor = BackendExecutor(
                    list(execution_backends),
                    backend_instances={
                        backend: adapters[backend] for backend in execution_backends
                    },
                    allow_backend_factory_fallback=False,
                )
                executors[execution_backends] = executor
            _raw, normalized = executor.execute(case, config)
            executed_cells += 1
            backend_result_count += len(normalized)
            for backend, result in normalized.items():
                statuses[(backend, result.status)] += 1
                if result.status != "ok":
                    errors.append(
                        f"global_cell={global_index}:family={family_id}:backend={backend}:"
                        f"status={result.status}:{result.error_type}:{result.error}"
                    )
                    continue
                if (
                    backend == registration.target_backend
                    and len(result.rows) == 1
                    and len(result.rows[0]) == 1
                    and result.rows[0][0] is True
                ):
                    target_true_counts[family_id] += 1
    finally:
        for adapter in adapters.values():
            adapter.close()
    backend_statuses: dict[str, dict[str, int]] = {}
    for (backend, status), count in sorted(statuses.items()):
        backend_statuses.setdefault(backend, {})[status] = count
    return {
        "requested": True,
        "complete": not errors,
        "executed_cell_count": executed_cells,
        "backend_result_count": backend_result_count,
        "backend_statuses": backend_statuses,
        "target_true_counts": dict(sorted(target_true_counts.items())),
        "errors": errors,
    }


def render_semantic_family_universe_report(result: Mapping[str, Any]) -> str:
    summary = result["summary"]
    lines = [
        "# Semantic Family Universe v1 Coverage Audit",
        "",
        f"- Target suite: `{result['target_suite']}`",
        f"- Backends with native targets: `{summary['backend_target_count']}` / `{summary['backend_count']}`",
        f"- Required families covered: `{summary['covered_family_count']}` / `{summary['family_count']}`",
        f"- Coverage-expansion families: `{summary['coverage_expansion_family_count']}`",
        f"- Required backend-family pairs: `{summary['required_backend_pair_count']}`",
        f"- Builder execution requested: `{result['execute_builders']}`",
        f"- Backend execution requested: `{result['execute_backends']}`",
        f"- Backend results: `{result['backend_execution']['backend_result_count']}`",
        f"- Fail-closed audit complete: `{summary['complete']}`",
        "",
        "| Family | Mechanism | Source | Target | Controls | Cells | Covered |",
        "| --- | --- | --- | --- | --- | ---: | --- |",
    ]
    for row in result["families"]:
        lines.append(
            f"| `{row['family_id']}` | `{row['mechanism_id']}` | `{row['source']}` | "
            f"`{row['target_backend']}` | `{','.join(row['control_backends'])}` | "
            f"{row['registered_cell_count']} | `{row['covered']}` |"
        )
    if result.get("errors"):
        lines.extend(["", "## Fail-closed gaps", ""])
        lines.extend(f"- `{error}`" for error in result["errors"])
    lines.append("")
    return "\n".join(lines)
