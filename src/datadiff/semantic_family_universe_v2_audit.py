"""Fail-closed audit for semantic-family universe v2 and global-v3."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Any

import datadiff.family_witness_registry as witness_registry
from datadiff.backends import make_backend
from datadiff.config import ExperimentConfig
from datadiff.execution import BackendExecutor
from datadiff.method_arms import DEFAULT_METHOD_ARM_ID, arm_difference, method_arm
from datadiff.operation_semantics import (
    aggregate_functions,
    expr_kinds,
    op_kind,
)
from datadiff.semantic_family_universe_v2 import (
    SEMANTIC_FAMILY_GLOBAL_V3_GENERATION_MODE,
    SEMANTIC_FAMILY_GLOBAL_V3_METHOD_ARM_ID,
    SEMANTIC_FAMILY_UNIVERSE_V2_SCHEMA_VERSION,
    SEMANTIC_FAMILY_UNIVERSE_V2_TARGET_SUITE,
    expansion_v2_family_definitions,
    semantic_family_universe_v2_manifest,
    semantic_family_v2_definitions,
)
from datadiff.semantic_family_universe_audit import _audit_registration_builders
from datadiff.targets import resolve_target_backends, target_spec


SEMANTIC_FAMILY_UNIVERSE_V2_AUDIT_SCHEMA_VERSION = (
    "semantic-family-universe-audit-v2"
)

_REQUIRED_NEW_OPERATION_KINDS = frozenset(
    {
        "aggregate",
        "anti_join",
        "case_when",
        "coalesce",
        "distinct",
        "drop_nulls",
        "fill_null",
        "semi_join",
        "sortedness_check",
        "tuple_absence_filter",
        "union_all",
    }
)
_REQUIRED_AGGREGATES = frozenset(
    {"all", "any", "count", "max", "mean", "min", "nunique", "sum"}
)
_REQUIRED_NEW_EXPRESSIONS = frozenset(
    {
        "abs",
        "add_const",
        "clip",
        "string_concat",
        "string_contains",
        "string_ends_with",
        "string_lower",
        "string_null_if_empty",
        "string_replace",
        "string_slice",
        "string_starts_with",
        "string_strip",
        "string_upper",
    }
)


def audit_semantic_family_universe_v2(
    *,
    execute_builders: bool = False,
    execute_backends: bool = False,
) -> dict[str, Any]:
    manifest = semantic_family_universe_v2_manifest()
    definitions = semantic_family_v2_definitions()
    new_family_ids = {
        item.family_id for item in expansion_v2_family_definitions()
    }
    backends = tuple(
        resolve_target_backends(
            target_suite=SEMANTIC_FAMILY_UNIVERSE_V2_TARGET_SUITE
        )
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
        global_v3_arm = method_arm(SEMANTIC_FAMILY_GLOBAL_V3_METHOD_ARM_ID)
    except ValueError as exc:
        global_v3_arm = None
        errors.append(f"global_v3_method_arm_missing:{exc}")
    else:
        if global_v3_arm.generation_mode != SEMANTIC_FAMILY_GLOBAL_V3_GENERATION_MODE:
            errors.append("global_v3_generation_mode_mismatch")
        if global_v3_arm.parent_arm_id != "p8_candidate_v1":
            errors.append(f"global_v3_parent_mismatch:{global_v3_arm.parent_arm_id}")
        if arm_difference(method_arm("p8_candidate_v1"), global_v3_arm) != {
            "generation_mode": (
                "goal_first",
                SEMANTIC_FAMILY_GLOBAL_V3_GENERATION_MODE,
            )
        }:
            errors.append("global_v3_changes_non_generation_dimensions")

    registrations = witness_registry.global_family_witness_registrations(
        SEMANTIC_FAMILY_GLOBAL_V3_GENERATION_MODE
    )
    registrations_by_id = {item.family_id: item for item in registrations}
    target_counts: Counter[str] = Counter()
    registered_ids: set[str] = set()
    family_rows: list[dict[str, Any]] = []
    for definition in definitions:
        row_errors: list[str] = []
        target_counts[definition.target_backend] += 1
        if definition.target_backend not in backends:
            row_errors.append("target_backend_outside_universe")
        outside_controls = sorted(set(definition.control_backends) - set(backends))
        if outside_controls:
            row_errors.append(
                "control_backends_outside_universe:" + ",".join(outside_controls)
            )

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
                    f"backend_capability_gap:{backend}:"
                    + ",".join(decision.missing)
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
                row_errors.append("execution_backends_registration_mismatch")
            if registration.cell_count < definition.minimum_cell_count:
                row_errors.append(
                    f"insufficient_cells:{registration.cell_count}/"
                    f"{definition.minimum_cell_count}"
                )
            try:
                registration_arm = method_arm(registration.method_arm_id)
            except ValueError as exc:
                row_errors.append(f"registration_method_arm_missing:{exc}")
            else:
                if registration_arm.generation_mode != registration.generation_mode:
                    row_errors.append("registration_generation_mode_arm_mismatch")
                if registration_arm.parent_arm_id != "p8_candidate_v1":
                    row_errors.append("registration_parent_mismatch")
                if set(
                    arm_difference(method_arm("p8_candidate_v1"), registration_arm)
                ) != {"generation_mode"}:
                    row_errors.append("registration_changes_non_generation_dimensions")
            if definition.family_id in new_family_ids and (
                registration.generation_mode
                != SEMANTIC_FAMILY_GLOBAL_V3_GENERATION_MODE
                or registration.method_arm_id
                != SEMANTIC_FAMILY_GLOBAL_V3_METHOD_ARM_ID
            ):
                row_errors.append("v2_expansion_family_not_owned_by_global_v3")
            if execute_builders:
                builder_summary = _audit_registration_builders(registration)
                if not builder_summary["complete"]:
                    row_errors.extend(builder_summary["errors"])

        family_rows.append(
            {
                **definition.manifest(),
                "registered": registration is not None,
                "registered_cell_count": (
                    0 if registration is None else registration.cell_count
                ),
                "capabilities": capability_rows,
                "builder_audit": builder_summary,
                "covered": not row_errors,
                "errors": row_errors,
            }
        )

    missing_target_backends = sorted(set(backends) - set(target_counts))
    if missing_target_backends:
        errors.append(
            "backends_without_native_target:" + ",".join(missing_target_backends)
        )
    extra_registered = sorted(set(registrations_by_id) - set(family_ids))
    if extra_registered:
        errors.append(
            "registered_families_outside_universe:" + ",".join(extra_registered)
        )

    global_portfolio = _audit_global_v3_portfolio(definitions)
    if not global_portfolio["complete"]:
        errors.extend(global_portfolio["errors"])

    coverage_profile = _coverage_profile()
    if not coverage_profile["complete"]:
        errors.extend(coverage_profile["errors"])

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
        backend_execution = _execute_global_v3_portfolio(backends)
        if not backend_execution["complete"]:
            errors.extend(backend_execution["errors"])

    errors.extend(
        f"family={row['family_id']}:{error}"
        for row in family_rows
        for error in row["errors"]
    )
    covered_count = sum(bool(row["covered"]) for row in family_rows)
    complete = bool(
        not errors
        and covered_count == len(definitions)
        and global_v3_arm is not None
        and global_portfolio["complete"]
        and coverage_profile["complete"]
        and backend_execution["complete"]
    )
    return {
        "schema_version": SEMANTIC_FAMILY_UNIVERSE_V2_AUDIT_SCHEMA_VERSION,
        "universe_schema_version": SEMANTIC_FAMILY_UNIVERSE_V2_SCHEMA_VERSION,
        "universe_digest": manifest["digest"],
        "target_suite": SEMANTIC_FAMILY_UNIVERSE_V2_TARGET_SUITE,
        "backends": list(backends),
        "default_method_arm": DEFAULT_METHOD_ARM_ID,
        "global_v3_generation_mode": SEMANTIC_FAMILY_GLOBAL_V3_GENERATION_MODE,
        "global_v3_method_arm_id": SEMANTIC_FAMILY_GLOBAL_V3_METHOD_ARM_ID,
        "execute_builders": bool(execute_builders),
        "execute_backends": bool(execute_backends),
        "families": family_rows,
        "global_portfolio": global_portfolio,
        "coverage_profile": coverage_profile,
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
            "new_coverage_expansion_family_count": len(new_family_ids),
            "required_backend_pair_count": manifest["summary"][
                "required_backend_pair_count"
            ],
            "complete": complete,
        },
        "errors": errors,
    }


def _audit_global_v3_portfolio(definitions: tuple[Any, ...]) -> dict[str, Any]:
    errors: list[str] = []
    registrations = witness_registry.global_family_witness_registrations(
        SEMANTIC_FAMILY_GLOBAL_V3_GENERATION_MODE
    )
    cell_count = witness_registry.global_family_witness_cell_count(
        SEMANTIC_FAMILY_GLOBAL_V3_GENERATION_MODE
    )
    expected_ids = [item.family_id for item in definitions]
    observed_ids = [item.family_id for item in registrations]
    if observed_ids != expected_ids:
        errors.append("global_v3_family_order_or_membership_mismatch")
    if cell_count != sum(item.cell_count for item in registrations):
        errors.append("global_v3_cell_count_mismatch")
    observed_pairs = {
        (
            witness_registry.global_family_witness_selection(
                index,
                generation_mode=SEMANTIC_FAMILY_GLOBAL_V3_GENERATION_MODE,
            ).registration.family_id,
            witness_registry.global_family_witness_selection(
                index,
                generation_mode=SEMANTIC_FAMILY_GLOBAL_V3_GENERATION_MODE,
            ).family_cell_index,
        )
        for index in range(cell_count)
    }
    expected_pairs = {
        (registration.family_id, cell_index)
        for registration in registrations
        for cell_index in range(registration.cell_count)
    }
    if observed_pairs != expected_pairs:
        errors.append("global_v3_family_cell_pairs_not_exact")
    return {
        "generation_mode": SEMANTIC_FAMILY_GLOBAL_V3_GENERATION_MODE,
        "registered_family_ids": observed_ids,
        "cell_count": cell_count,
        "observed_family_cell_pair_count": len(observed_pairs),
        "complete": not errors,
        "errors": errors,
    }


def _coverage_profile() -> dict[str, Any]:
    old = _collect_coverage(witness_registry.GLOBAL_FAMILY_WITNESS_V2_GENERATION_MODE)
    new = _collect_coverage(SEMANTIC_FAMILY_GLOBAL_V3_GENERATION_MODE)
    new_operation_kinds = sorted(set(new["operation_kinds"]) - set(old["operation_kinds"]))
    new_expression_kinds = sorted(set(new["expression_kinds"]) - set(old["expression_kinds"]))
    new_aggregate_functions = sorted(
        set(new["aggregate_functions"]) - set(old["aggregate_functions"])
    )
    errors: list[str] = []
    missing_operations = sorted(_REQUIRED_NEW_OPERATION_KINDS - set(new_operation_kinds))
    missing_expressions = sorted(_REQUIRED_NEW_EXPRESSIONS - set(new_expression_kinds))
    missing_aggregates = sorted(_REQUIRED_AGGREGATES - set(new["aggregate_functions"]))
    if missing_operations:
        errors.append("missing_new_operation_kinds:" + ",".join(missing_operations))
    if missing_expressions:
        errors.append("missing_new_expression_kinds:" + ",".join(missing_expressions))
    if missing_aggregates:
        errors.append("missing_aggregate_functions:" + ",".join(missing_aggregates))
    return {
        "baseline_global_v2": old,
        "expanded_global_v3": new,
        "new_operation_kinds": new_operation_kinds,
        "new_expression_kinds": new_expression_kinds,
        "new_aggregate_functions": new_aggregate_functions,
        "complete": not errors,
        "errors": errors,
    }


def _collect_coverage(generation_mode: str) -> dict[str, Any]:
    operations: set[str] = set()
    expressions: set[str] = set()
    aggregates: set[str] = set()
    cell_count = witness_registry.global_family_witness_cell_count(generation_mode)
    for seed in range(cell_count):
        case = witness_registry.generate_global_family_witness_case(
            seed,
            generation_mode=generation_mode,
        )
        operations.update(op_kind(operation) for operation in case.program.operations)
        expressions.update(expr_kinds(case.program.operations))
        aggregates.update(
            function
            for operation in case.program.operations
            for function in aggregate_functions(operation)
        )
    return {
        "generation_mode": generation_mode,
        "cell_count": cell_count,
        "operation_kinds": sorted(operations),
        "expression_kinds": sorted(expressions),
        "aggregate_functions": sorted(aggregates),
    }


def _execute_global_v3_portfolio(backends: tuple[str, ...]) -> dict[str, Any]:
    errors: list[str] = []
    adapters = {backend: make_backend(backend) for backend in backends}
    executors: dict[tuple[str, ...], BackendExecutor] = {}
    config = ExperimentConfig(
        method_arm=SEMANTIC_FAMILY_GLOBAL_V3_METHOD_ARM_ID,
        enable_parallel_backend_execution=False,
        enable_backend_session_reuse=True,
    )
    statuses: Counter[tuple[str, str]] = Counter()
    target_true_counts: Counter[str] = Counter()
    backend_result_count = 0
    executed_cells = 0
    try:
        cell_count = witness_registry.global_family_witness_cell_count(
            SEMANTIC_FAMILY_GLOBAL_V3_GENERATION_MODE
        )
        for global_index in range(cell_count):
            case = witness_registry.generate_global_family_witness_case(
                global_index,
                generation_mode=SEMANTIC_FAMILY_GLOBAL_V3_GENERATION_MODE,
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
                        f"global_cell={global_index}:family={family_id}:"
                        f"backend={backend}:status={result.status}:"
                        f"{result.error_type}:{result.error}"
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


def render_semantic_family_universe_v2_report(result: Mapping[str, Any]) -> str:
    summary = result["summary"]
    profile = result["coverage_profile"]
    lines = [
        "# Semantic Family Universe v2 Coverage Audit",
        "",
        f"- Target suite: `{result['target_suite']}`",
        f"- Backends with native targets: `{summary['backend_target_count']}` / `{summary['backend_count']}`",
        f"- Required families covered: `{summary['covered_family_count']}` / `{summary['family_count']}`",
        f"- New bounded families: `{summary['new_coverage_expansion_family_count']}`",
        f"- Exact global-v3 cells: `{result['global_portfolio']['cell_count']}`",
        f"- Required backend-family pairs: `{summary['required_backend_pair_count']}`",
        f"- Backend results: `{result['backend_execution']['backend_result_count']}`",
        f"- Fail-closed audit complete: `{summary['complete']}`",
        "",
        "## Structural breadth added",
        "",
        f"- New operation kinds: `{', '.join(profile['new_operation_kinds'])}`",
        f"- New expression kinds: `{', '.join(profile['new_expression_kinds'])}`",
        f"- New aggregate functions: `{', '.join(profile['new_aggregate_functions'])}`",
        "",
        "| Family | Mechanism | Target | Controls | Cells | Covered |",
        "| --- | --- | --- | --- | ---: | --- |",
    ]
    for row in result["families"]:
        lines.append(
            f"| `{row['family_id']}` | `{row['mechanism_id']}` | "
            f"`{row['target_backend']}` | `{','.join(row['control_backends'])}` | "
            f"{row['registered_cell_count']} | `{row['covered']}` |"
        )
    if result.get("errors"):
        lines.extend(["", "## Fail-closed gaps", ""])
        lines.extend(f"- `{error}`" for error in result["errors"])
    lines.append("")
    return "\n".join(lines)
