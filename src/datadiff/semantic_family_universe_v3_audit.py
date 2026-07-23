"""Fail-closed, per-backend audit for semantic-family universe v3/Global-v4."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Any

import datadiff.family_witness_registry as witness_registry
from datadiff.backends import make_backend
from datadiff.canonicalization import canonical_key, short_canonical_hash
from datadiff.config import ExperimentConfig
from datadiff.contract_comparison import comparison_payload_for_case
from datadiff.execution import BackendExecutor
from datadiff.execution_cache import ExecutionResultCache
from datadiff.method_arms import DEFAULT_METHOD_ARM_ID, arm_difference, method_arm
from datadiff.operation_semantics import aggregate_functions, expr_kinds, op_kind
from datadiff.semantic_family_universe_audit import _audit_registration_builders
from datadiff.semantic_family_universe_v3 import (
    CROSS_BACKEND_RISK_FAMILY_ID,
    EXACT_DTYPE_FAMILY_ID,
    GENERIC_AGGREGATE_FUNCTIONS,
    GENERIC_EXPRESSION_KINDS,
    GENERIC_OPERATION_KINDS,
    PYARROW_LAYOUT_RISK_FAMILY_ID,
    SEMANTIC_FAMILY_GLOBAL_V4_GENERATION_MODE,
    SEMANTIC_FAMILY_GLOBAL_V4_METHOD_ARM_ID,
    SEMANTIC_FAMILY_UNIVERSE_V3_SCHEMA_VERSION,
    SEMANTIC_FAMILY_UNIVERSE_V3_TARGET_SUITE,
    expansion_v3_family_definitions,
    pipeline_evidence_spec,
    pipeline_evidence_specs,
    semantic_family_universe_v3_manifest,
    semantic_family_v3_definitions,
    semantic_family_v3_witness_spec,
)
from datadiff.targets import resolve_target_backends, target_spec


SEMANTIC_FAMILY_UNIVERSE_V3_AUDIT_SCHEMA_VERSION = (
    "semantic-family-universe-audit-v3"
)


def audit_semantic_family_universe_v3(
    *,
    execute_builders: bool = False,
    execute_backends: bool = False,
) -> dict[str, Any]:
    manifest = semantic_family_universe_v3_manifest()
    definitions = semantic_family_v3_definitions()
    new_definitions = expansion_v3_family_definitions()
    new_family_ids = {item.family_id for item in new_definitions}
    backends = tuple(
        resolve_target_backends(
            target_suite=SEMANTIC_FAMILY_UNIVERSE_V3_TARGET_SUITE
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

    global_v4_arm = _audit_method_arm(errors)
    registrations = witness_registry.latest_family_witness_registrations()
    registrations_by_id = {item.family_id: item for item in registrations}
    registered_ids: set[str] = set()
    target_counts: Counter[str] = Counter()
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

        capability_rows = []
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
            if definition.family_id in new_family_ids:
                spec = semantic_family_v3_witness_spec(definition.family_id)
                if registration.axes != spec.axes:
                    row_errors.append("v3_axes_registration_mismatch")
                if registration.backends != spec.execution_backends:
                    row_errors.append("v3_backend_scope_registration_mismatch")
                if (
                    registration.generation_mode
                    != SEMANTIC_FAMILY_GLOBAL_V4_GENERATION_MODE
                    or registration.method_arm_id
                    != SEMANTIC_FAMILY_GLOBAL_V4_METHOD_ARM_ID
                ):
                    row_errors.append("v3_expansion_family_not_owned_by_global_v4")
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

    global_portfolio = _audit_global_v4_portfolio(definitions)
    errors.extend(global_portfolio["errors"])
    backend_coverage = _audit_backend_coverage(backends)
    errors.extend(backend_coverage["errors"])
    cache_reuse = _audit_declared_cache_reuse()
    errors.extend(cache_reuse["errors"])

    backend_execution = {
        "requested": bool(execute_backends),
        "complete": not execute_backends,
        "executed_cell_count": 0,
        "backend_result_count": 0,
        "backend_statuses": {},
        "exact_dtype_equal_cell_count": 0,
        "layout_equal_cell_count": 0,
        "pyarrow_applied_layout_cell_count": 0,
        "optimizer_plan_observation": {},
        "candidate_comparison_cells": [],
        "errors": [],
    }
    if execute_backends:
        backend_execution = _execute_new_v3_cells(backends)
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
        and global_v4_arm is not None
        and global_portfolio["complete"]
        and backend_coverage["complete"]
        and cache_reuse["complete"]
        and backend_execution["complete"]
    )
    return {
        "schema_version": SEMANTIC_FAMILY_UNIVERSE_V3_AUDIT_SCHEMA_VERSION,
        "universe_schema_version": SEMANTIC_FAMILY_UNIVERSE_V3_SCHEMA_VERSION,
        "universe_digest": manifest["digest"],
        "target_suite": SEMANTIC_FAMILY_UNIVERSE_V3_TARGET_SUITE,
        "backends": list(backends),
        "default_method_arm": DEFAULT_METHOD_ARM_ID,
        "global_v4_generation_mode": SEMANTIC_FAMILY_GLOBAL_V4_GENERATION_MODE,
        "global_v4_method_arm_id": SEMANTIC_FAMILY_GLOBAL_V4_METHOD_ARM_ID,
        "execute_builders": bool(execute_builders),
        "execute_backends": bool(execute_backends),
        "families": family_rows,
        "global_portfolio": global_portfolio,
        "backend_coverage": backend_coverage,
        "cache_reuse": cache_reuse,
        "coverage_profile": {
            "baseline_global_v3": backend_coverage["baseline_global_v3"],
            "expanded_global_v4": backend_coverage["expanded_global_v4"],
        },
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
            "new_witness_cell_count": manifest["summary"][
                "new_witness_cell_count"
            ],
            "complete": complete,
        },
        "errors": errors,
    }


def _audit_declared_cache_reuse() -> dict[str, Any]:
    errors: list[str] = []
    consumer = witness_registry.family_witness_registration(EXACT_DTYPE_FAMILY_ID)
    source_id = consumer.cache_reuse_source_family_id
    if source_id != CROSS_BACKEND_RISK_FAMILY_ID:
        errors.append(
            f"exact_dtype_cache_source:{source_id or 'missing'}"
        )
        return {
            "complete": False,
            "source_family_id": source_id,
            "consumer_family_id": consumer.family_id,
            "shared_cell_count": 0,
            "digest_equal_cell_count": 0,
            "scheduled_cache_hit_count": 0,
            "errors": errors,
        }
    source = witness_registry.family_witness_registration(source_id)
    portfolio = witness_registry.global_family_witness_registrations(
        SEMANTIC_FAMILY_GLOBAL_V4_GENERATION_MODE
    )
    source_position = portfolio.index(source)
    consumer_position = portfolio.index(consumer)
    if source_position >= consumer_position:
        errors.append("cache_reuse_source_not_scheduled_before_consumer")

    config = ExperimentConfig(method_arm=SEMANTIC_FAMILY_GLOBAL_V4_METHOD_ARM_ID)
    digest_equal = 0
    scheduled_hits = 0
    rows: list[dict[str, Any]] = []
    for consumer_cell in range(consumer.cell_count):
        consumer_case = consumer.generate_case(consumer_cell)
        _index, axes = consumer.cell_for_seed(consumer_cell)
        source_axes = {
            name: axes[name]
            for name in source.axis_names
        }
        source_cell = source.cell_index_for_axes(source_axes)
        source_case = source.generate_case(source_cell)
        source_digest = ExecutionResultCache.case_digest(source_case, config)
        consumer_digest = ExecutionResultCache.case_digest(consumer_case, config)
        digest_matches = source_digest == consumer_digest
        digest_equal += int(digest_matches)
        warmed_backends = tuple(
            backend
            for backend in witness_registry.family_witness_execution_backends(
                source_case
            )
            if backend in consumer.backends
        )
        scheduled_hits += len(warmed_backends) if digest_matches else 0
        rows.append(
            {
                "consumer_cell_index": consumer_cell,
                "source_cell_index": source_cell,
                "axes": dict(axes),
                "case_digest_equal": digest_matches,
                "scheduled_hit_backends": list(warmed_backends),
            }
        )
        if not digest_matches:
            errors.append(
                f"cache_reuse_digest_mismatch:consumer_cell={consumer_cell}:"
                f"source_cell={source_cell}"
            )
    if digest_equal != consumer.cell_count:
        errors.append(
            f"cache_reuse_digest_equal_cells:{digest_equal}/{consumer.cell_count}"
        )
    if scheduled_hits <= 0:
        errors.append("cache_reuse_scheduled_hits_zero")
    return {
        "complete": not errors,
        "source_family_id": source.family_id,
        "consumer_family_id": consumer.family_id,
        "reuse_axes": list(consumer.cache_reuse_axes),
        "source_portfolio_position": source_position,
        "consumer_portfolio_position": consumer_position,
        "shared_cell_count": consumer.cell_count,
        "digest_equal_cell_count": digest_equal,
        "scheduled_cache_hit_count": scheduled_hits,
        "cells": rows,
        "errors": errors,
    }


def _audit_method_arm(errors: list[str]) -> Any | None:
    try:
        arm = method_arm(SEMANTIC_FAMILY_GLOBAL_V4_METHOD_ARM_ID)
    except ValueError as exc:
        errors.append(f"global_v4_method_arm_missing:{exc}")
        return None
    if arm.generation_mode != SEMANTIC_FAMILY_GLOBAL_V4_GENERATION_MODE:
        errors.append("global_v4_generation_mode_mismatch")
    if arm.parent_arm_id != "p8_candidate_v1":
        errors.append(f"global_v4_parent_mismatch:{arm.parent_arm_id}")
    if arm_difference(method_arm("p8_candidate_v1"), arm) != {
        "generation_mode": (
            "goal_first",
            SEMANTIC_FAMILY_GLOBAL_V4_GENERATION_MODE,
        )
    }:
        errors.append("global_v4_changes_non_generation_dimensions")
    return arm


def _audit_global_v4_portfolio(definitions: tuple[Any, ...]) -> dict[str, Any]:
    errors: list[str] = []
    registrations = witness_registry.global_family_witness_registrations(
        SEMANTIC_FAMILY_GLOBAL_V4_GENERATION_MODE
    )
    cell_count = witness_registry.global_family_witness_cell_count(
        SEMANTIC_FAMILY_GLOBAL_V4_GENERATION_MODE
    )
    expected_ids = [item.family_id for item in definitions]
    observed_ids = [item.family_id for item in registrations]
    if observed_ids != expected_ids:
        errors.append("global_v4_family_order_or_membership_mismatch")
    if cell_count != 376:
        errors.append(f"global_v4_cell_count:{cell_count}/376")
    observed_pairs = {
        (
            selection.registration.family_id,
            selection.family_cell_index,
        )
        for index in range(cell_count)
        for selection in (
            witness_registry.global_family_witness_selection(
                index,
                generation_mode=SEMANTIC_FAMILY_GLOBAL_V4_GENERATION_MODE,
            ),
        )
    }
    expected_pairs = {
        (registration.family_id, cell_index)
        for registration in registrations
        for cell_index in range(registration.cell_count)
    }
    if observed_pairs != expected_pairs:
        errors.append("global_v4_family_cell_pairs_not_exact")
    return {
        "generation_mode": SEMANTIC_FAMILY_GLOBAL_V4_GENERATION_MODE,
        "registered_family_ids": observed_ids,
        "cell_count": cell_count,
        "observed_family_cell_pair_count": len(observed_pairs),
        "complete": not errors,
        "errors": errors,
    }


def _audit_backend_coverage(backends: tuple[str, ...]) -> dict[str, Any]:
    baseline = _collect_backend_coverage(
        witness_registry.GLOBAL_FAMILY_WITNESS_V3_GENERATION_MODE,
        backends,
    )
    expanded = _collect_backend_coverage(
        SEMANTIC_FAMILY_GLOBAL_V4_GENERATION_MODE,
        backends,
    )
    operation_target = set(GENERIC_OPERATION_KINDS)
    expression_target = set(GENERIC_EXPRESSION_KINDS)
    aggregate_target = set(GENERIC_AGGREGATE_FUNCTIONS)
    expected_chains = {
        ">".join(chain)
        for evidence in pipeline_evidence_specs()
        for chain in evidence.required_operation_chains
    }
    exact_scope = set(
        semantic_family_v3_witness_spec(EXACT_DTYPE_FAMILY_ID).execution_backends
    )
    layout_scope = set(
        semantic_family_v3_witness_spec(
            PYARROW_LAYOUT_RISK_FAMILY_ID
        ).execution_backends
    )
    errors: list[str] = []
    backend_checks: dict[str, dict[str, bool]] = {}
    for backend in backends:
        old = baseline["backends"][backend]
        new = expanded["backends"][backend]
        checks = {
            "generic_operations_complete": set(new["operation_kinds"])
            == operation_target,
            "generic_expressions_complete": set(new["expression_kinds"])
            == expression_target,
            "generic_aggregates_complete": set(new["aggregate_functions"])
            == aggregate_target,
            "operation_count_strictly_increased": len(new["operation_kinds"])
            > len(old["operation_kinds"]),
            "expression_count_strictly_increased": len(new["expression_kinds"])
            > len(old["expression_kinds"]),
            "aggregate_count_strictly_increased": len(new["aggregate_functions"])
            > len(old["aggregate_functions"]),
            "risk_pipeline_cells_complete": new["risk_pipeline_cell_count"]
            >= len(pipeline_evidence_specs()),
            "risk_pipelines_complete": len(new["risk_pipelines"]) == 18,
            "risk_classes_complete": len(new["risk_classes"]) == 7,
            "risk_operation_chains_complete": set(new["risk_operation_chains"])
            == expected_chains,
            "boundary_overlay_cells_complete": new["boundary_overlay_cell_count"]
            >= 5,
            "boundary_profiles_materially_diverse": len(new["boundary_profiles"])
            >= 5,
            "exact_dtype_scope_correct": new["exact_dtype_cell_count"]
            == (10 if backend in exact_scope else 0),
            "layout_scope_correct": new["layout_interaction_cell_count"]
            == (24 if backend in layout_scope else 0),
            "probe_and_generic_coverage_separated": not set(
                new["operation_kinds"]
            ).intersection(new["probe_operation_kinds"]),
            "risk_palette_contains_no_probe_operations": new[
                "risk_probe_cell_count"
            ]
            == 0,
        }
        backend_checks[backend] = checks
        errors.extend(
            f"backend={backend}:{name}"
            for name, passed in checks.items()
            if not passed
        )
    return {
        "baseline_global_v3": baseline,
        "expanded_global_v4": expanded,
        "required_generic_coverage": {
            "operation_kinds": list(GENERIC_OPERATION_KINDS),
            "expression_kinds": list(GENERIC_EXPRESSION_KINDS),
            "aggregate_functions": list(GENERIC_AGGREGATE_FUNCTIONS),
            "risk_operation_chains": sorted(expected_chains),
        },
        "backend_checks": backend_checks,
        "complete": not errors,
        "errors": errors,
    }


def _collect_backend_coverage(
    generation_mode: str,
    backends: tuple[str, ...],
) -> dict[str, Any]:
    state = {
        backend: {
            "family_cell_count": 0,
            "operations": set(),
            "expressions": set(),
            "aggregates": set(),
            "probes": set(),
            "risk_pipeline_cell_count": 0,
            "risk_pipelines": set(),
            "risk_classes": set(),
            "risk_operation_chains": set(),
            "risk_probe_cell_count": 0,
            "boundary_overlay_cell_count": 0,
            "boundary_profiles": set(),
            "exact_dtype_cell_count": 0,
            "layout_interaction_cell_count": 0,
            "layouts": set(),
        }
        for backend in backends
    }
    for registration in witness_registry.global_family_witness_registrations(
        generation_mode
    ):
        for seed in range(registration.cell_count):
            case = registration.generate_case(seed)
            kinds = tuple(op_kind(item) for item in case.program.operations)
            generic_operations = {kind for kind in kinds if not kind.endswith("_probe")}
            probe_operations = set(kinds) - generic_operations
            expressions = expr_kinds(case.program.operations)
            aggregates = {
                function
                for operation in case.program.operations
                for function in aggregate_functions(operation)
            }
            axes = registration.cell_for_seed(seed)[1]
            for backend in witness_registry.family_witness_execution_backends(case):
                row = state[backend]
                row["family_cell_count"] += 1
                row["operations"].update(generic_operations)
                row["expressions"].update(expressions)
                row["aggregates"].update(aggregates)
                row["probes"].update(probe_operations)
                if registration.family_id == CROSS_BACKEND_RISK_FAMILY_ID:
                    evidence = pipeline_evidence_spec(axes["pipeline"])
                    row["risk_pipeline_cell_count"] += 1
                    row["risk_pipelines"].add(evidence.pipeline_id)
                    row["risk_classes"].add(evidence.risk_class)
                    row["risk_operation_chains"].update(
                        ">".join(chain)
                        for chain in evidence.required_operation_chains
                        if _is_subsequence(chain, kinds)
                    )
                    row["risk_probe_cell_count"] += int(bool(probe_operations))
                    boundary = case.metadata.get("boundary_application", {}) or {}
                    if isinstance(boundary, Mapping) and boundary.get("applied") is True:
                        row["boundary_overlay_cell_count"] += 1
                        row["boundary_profiles"].add(
                            str(boundary.get("profile_id", ""))
                        )
                elif registration.family_id == EXACT_DTYPE_FAMILY_ID:
                    row["exact_dtype_cell_count"] += 1
                elif registration.family_id == PYARROW_LAYOUT_RISK_FAMILY_ID:
                    row["layout_interaction_cell_count"] += 1
                    row["layouts"].add(axes["layout"])
    return {
        "generation_mode": generation_mode,
        "cell_count": witness_registry.global_family_witness_cell_count(
            generation_mode
        ),
        "backends": {
            backend: {
                "family_cell_count": row["family_cell_count"],
                "operation_kinds": sorted(row["operations"]),
                "expression_kinds": sorted(row["expressions"]),
                "aggregate_functions": sorted(row["aggregates"]),
                "probe_operation_kinds": sorted(row["probes"]),
                "risk_pipeline_cell_count": row["risk_pipeline_cell_count"],
                "risk_pipelines": sorted(row["risk_pipelines"]),
                "risk_classes": sorted(row["risk_classes"]),
                "risk_operation_chains": sorted(row["risk_operation_chains"]),
                "risk_probe_cell_count": row["risk_probe_cell_count"],
                "boundary_overlay_cell_count": row[
                    "boundary_overlay_cell_count"
                ],
                "boundary_profiles": sorted(row["boundary_profiles"]),
                "exact_dtype_cell_count": row["exact_dtype_cell_count"],
                "layout_interaction_cell_count": row[
                    "layout_interaction_cell_count"
                ],
                "layouts": sorted(row["layouts"]),
            }
            for backend, row in state.items()
        },
    }


def _execute_new_v3_cells(backends: tuple[str, ...]) -> dict[str, Any]:
    errors: list[str] = []
    adapters = {backend: make_backend(backend) for backend in backends}
    executors: dict[tuple[str, ...], BackendExecutor] = {}
    config = ExperimentConfig(
        method_arm=SEMANTIC_FAMILY_GLOBAL_V4_METHOD_ARM_ID,
        evidence_tier="finding",
        enable_parallel_backend_execution=False,
        enable_backend_session_reuse=True,
    )
    statuses: Counter[tuple[str, str, str]] = Counter()
    optimizer = {
        backend: {
            "observed_cell_count": 0,
            "changed_cell_count": 0,
            "physical_plan_ok_cell_count": 0,
            "pipelines": set(),
        }
        for backend in ("polars_lazy", "datafusion")
    }
    candidate_cells: list[dict[str, Any]] = []
    exact_equal = 0
    layout_equal = 0
    pyarrow_layout_applied = 0
    executed_cells = 0
    backend_result_count = 0
    try:
        for definition in expansion_v3_family_definitions():
            registration = witness_registry.family_witness_registration(
                definition.family_id
            )
            executor = executors.get(registration.backends)
            if executor is None:
                executor = BackendExecutor(
                    list(registration.backends),
                    backend_instances={
                        backend: adapters[backend]
                        for backend in registration.backends
                    },
                    allow_backend_factory_fallback=False,
                )
                executors[registration.backends] = executor
            for seed in range(registration.cell_count):
                case = registration.generate_case(seed)
                axes = registration.cell_for_seed(seed)[1]
                raw, normalized = executor.execute(case, config)
                executed_cells += 1
                backend_result_count += len(normalized)
                if tuple(normalized) != registration.backends:
                    errors.append(
                        f"family={registration.family_id}:cell={seed}:backend_scope_mismatch"
                    )
                for backend, result in normalized.items():
                    statuses[(registration.family_id, backend, result.status)] += 1
                    if result.status != "ok":
                        errors.append(
                            f"family={registration.family_id}:cell={seed}:backend={backend}:"
                            f"status={result.status}:{result.error_type}:{result.error}"
                        )

                comparison_groups = _comparison_groups(case, normalized)
                if registration.family_id == EXACT_DTYPE_FAMILY_ID:
                    if len(comparison_groups) == 1:
                        exact_equal += 1
                    else:
                        errors.append(
                            f"exact_dtype_mismatch:cell={seed}:groups={len(comparison_groups)}"
                        )
                elif registration.family_id == PYARROW_LAYOUT_RISK_FAMILY_ID:
                    if len(comparison_groups) == 1:
                        layout_equal += 1
                    else:
                        errors.append(
                            f"layout_value_mismatch:cell={seed}:groups={len(comparison_groups)}"
                        )
                    applied = str(raw["pyarrow"].get("input_physical_layout", ""))
                    if applied == axes["layout"]:
                        pyarrow_layout_applied += 1
                    else:
                        errors.append(
                            f"layout_not_applied:cell={seed}:"
                            f"expected={axes['layout']}:observed={applied}"
                        )
                elif len(comparison_groups) > 1:
                    candidate_cells.append(
                        {
                            "family_id": registration.family_id,
                            "cell_index": seed,
                            "axes": dict(axes),
                            "comparison_group_count": len(comparison_groups),
                            "groups": comparison_groups,
                        }
                    )

                if registration.family_id == CROSS_BACKEND_RISK_FAMILY_ID:
                    for backend in optimizer:
                        plan = raw[backend].get("physical_plan", {})
                        observations = {
                            str(item.get("plan_kind", "")): item
                            for item in plan.get("observations", []) or []
                            if isinstance(item, Mapping)
                        }
                        logical = observations.get("logical", {})
                        optimized = observations.get("optimized_logical", {})
                        if (
                            logical.get("status") == "ok"
                            and optimized.get("status") == "ok"
                        ):
                            optimizer[backend]["observed_cell_count"] += 1
                            optimizer[backend]["pipelines"].add(axes["pipeline"])
                            if logical.get("fingerprint") != optimized.get(
                                "fingerprint"
                            ):
                                optimizer[backend]["changed_cell_count"] += 1
                        if observations.get("physical", {}).get("status") == "ok":
                            optimizer[backend]["physical_plan_ok_cell_count"] += 1
    finally:
        for adapter in adapters.values():
            adapter.close()

    expected_results = sum(
        witness_registry.family_witness_registration(item.family_id).cell_count
        * len(witness_registry.family_witness_registration(item.family_id).backends)
        for item in expansion_v3_family_definitions()
    )
    if executed_cells != 70:
        errors.append(f"executed_cell_count:{executed_cells}/70")
    if backend_result_count != expected_results:
        errors.append(
            f"backend_result_count:{backend_result_count}/{expected_results}"
        )
    if exact_equal != 10:
        errors.append(f"exact_dtype_equal_cell_count:{exact_equal}/10")
    if layout_equal != 24:
        errors.append(f"layout_equal_cell_count:{layout_equal}/24")
    if pyarrow_layout_applied != 24:
        errors.append(
            f"pyarrow_applied_layout_cell_count:{pyarrow_layout_applied}/24"
        )
    for backend, row in optimizer.items():
        if row["observed_cell_count"] != 36:
            errors.append(
                f"optimizer_plan_observed:{backend}:"
                f"{row['observed_cell_count']}/36"
            )
        if row["changed_cell_count"] != 36:
            errors.append(
                f"optimizer_plan_changed:{backend}:"
                f"{row['changed_cell_count']}/36"
            )
        if len(row["pipelines"]) != 18:
            errors.append(
                f"optimizer_pipeline_coverage:{backend}:{len(row['pipelines'])}/18"
            )
    if optimizer["datafusion"]["physical_plan_ok_cell_count"] != 36:
        errors.append(
            "datafusion_physical_plan_coverage:"
            f"{optimizer['datafusion']['physical_plan_ok_cell_count']}/36"
        )
    return {
        "requested": True,
        "complete": not errors,
        "executed_cell_count": executed_cells,
        "backend_result_count": backend_result_count,
        "backend_statuses": {
            f"{family}|{backend}|{status}": count
            for (family, backend, status), count in sorted(statuses.items())
        },
        "exact_dtype_equal_cell_count": exact_equal,
        "layout_equal_cell_count": layout_equal,
        "pyarrow_applied_layout_cell_count": pyarrow_layout_applied,
        "optimizer_plan_observation": {
            backend: {
                **row,
                "pipelines": sorted(row["pipelines"]),
            }
            for backend, row in optimizer.items()
        },
        "candidate_comparison_cells": candidate_cells,
        "candidate_comparison_cell_count": len(candidate_cells),
        "errors": errors,
    }


def _comparison_groups(case: Any, normalized: Mapping[str, Any]) -> list[dict[str, Any]]:
    groups: dict[str, list[str]] = {}
    for backend, result in normalized.items():
        payload = comparison_payload_for_case(case, result)
        key = canonical_key(payload)
        groups.setdefault(key, []).append(backend)
    return [
        {
            "comparison_fingerprint": short_canonical_hash(key, 16),
            "backends": sorted(group_backends),
        }
        for key, group_backends in sorted(groups.items())
    ]


def _is_subsequence(expected: tuple[str, ...], observed: tuple[str, ...]) -> bool:
    if not expected:
        return True
    index = 0
    for item in observed:
        if item == expected[index]:
            index += 1
            if index == len(expected):
                return True
    return False


def render_semantic_family_universe_v3_report(result: Mapping[str, Any]) -> str:
    summary = dict(result.get("summary", {}) or {})
    portfolio = dict(result.get("global_portfolio", {}) or {})
    coverage = dict(result.get("backend_coverage", {}) or {})
    execution = dict(result.get("backend_execution", {}) or {})
    cache_reuse = dict(result.get("cache_reuse", {}) or {})
    expanded = dict(coverage.get("expanded_global_v4", {}) or {})
    lines = [
        "# Semantic Family Universe v3 Audit",
        "",
        f"- Complete: `{summary.get('complete')}`",
        f"- Families: `{summary.get('family_count')}`",
        f"- Global-v4 cells: `{portfolio.get('cell_count')}`",
        f"- New bounded cells: `{summary.get('new_witness_cell_count')}`",
        f"- Backend execution results: `{execution.get('backend_result_count', 0)}`",
        f"- Exact dtype equal cells: `{execution.get('exact_dtype_equal_cell_count', 0)}/10`",
        f"- Layout equal/applied cells: `{execution.get('layout_equal_cell_count', 0)}/24`, "
        f"`{execution.get('pyarrow_applied_layout_cell_count', 0)}/24`",
        f"- Cross-backend comparison candidate cells: "
        f"`{execution.get('candidate_comparison_cell_count', 0)}`",
        f"- Exact/cross cache-shared cells: "
        f"`{cache_reuse.get('digest_equal_cell_count', 0)}/"
        f"{cache_reuse.get('shared_cell_count', 0)}`; scheduled hits "
        f"`{cache_reuse.get('scheduled_cache_hit_count', 0)}`",
        "",
        "| Backend | Generic ops | Expressions | Aggregates | Risk cells | Exact dtype | Layout |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for backend, row in (expanded.get("backends", {}) or {}).items():
        lines.append(
            f"| `{backend}` | {len(row.get('operation_kinds', []) or [])} "
            f"| {len(row.get('expression_kinds', []) or [])} "
            f"| {len(row.get('aggregate_functions', []) or [])} "
            f"| {int(row.get('risk_pipeline_cell_count', 0))} "
            f"| {int(row.get('exact_dtype_cell_count', 0))} "
            f"| {int(row.get('layout_interaction_cell_count', 0))} |"
        )
    optimizer = execution.get("optimizer_plan_observation", {}) or {}
    if optimizer:
        lines.extend(["", "## Optimizer plan activation", ""])
        for backend, row in optimizer.items():
            lines.append(
                f"- `{backend}`: logical/optimized observed "
                f"`{row.get('observed_cell_count', 0)}/36`, changed "
                f"`{row.get('changed_cell_count', 0)}/36`."
            )
    errors = list(result.get("errors", []) or [])
    lines.extend(["", "## Errors", ""])
    lines.extend([f"- `{item}`" for item in errors] or ["- None."])
    lines.append("")
    return "\n".join(lines)
