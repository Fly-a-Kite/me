from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any, Callable

from datadiff.artifact import save_issue_artifact as save_bug_artifact
from datadiff.classification_oracle import annotate_findings
from datadiff.config import ExperimentConfig
from datadiff.contract_comparison import comparison_profile_payload
from datadiff.disagreement import compute_descriptor
from datadiff.dsl import Case
from datadiff.env import collect_environment
from datadiff.execution import (
    build_execution_diagnostic_refs,
    execute_case as _execute_case_impl,
)
from datadiff.execution_accounting import (
    backend_calls_from_raw,
    backend_reported_ms_from_raw,
    execution_cache_hits_from_raw,
    execution_profile_backend_calls,
    execution_profile_backend_reported_ms,
)
from datadiff.fingerprint import compute_fingerprint
from datadiff.finding_outcomes import (
    is_known_saturated_candidate_issue_finding,
)
from datadiff.ir_runtime import resolve_case_program
from datadiff.interaction_descriptor import build_interaction_descriptor
from datadiff.localization import localize_first_divergence
from datadiff.experiment_manifest import (
    build_case_experiment_manifest,
    finalize_config_payload,
)
from datadiff.metamorphic import (
    all_metamorphic_variants,
    evaluate_metamorphic_variants,
    select_metamorphic_variants,
)
from datadiff.oracle import Finding, evaluate_case
from datadiff.oracle_complex import cross_validate_oracle_findings
from datadiff.run_config import _config_payload_with_effective_guidance_targets
from datadiff.run_findings import _countable_finding_objects
from datadiff.run_logging import (
    COST_PROFILE_STAGE_KEYS,
    STAGE_PROFILE_KEYS,
    _empty_process_cpu_profile,
    _empty_stage_profile,
    _empty_wall_time_profile,
    _process_cpu_profile_with_total,
    _wall_time_profile_with_total,
)
from datadiff.run_metadata import (
    _attach_case_fingerprint_to_row_case,
    _attach_disagreement_descriptor_to_row_case,
    _attach_semantic_contract_lattice_to_row_case,
    _attach_interaction_descriptor_to_row_case,
    _fingerprint_anchor_result,
)
from datadiff.run_signatures import behavior_signature, discovery_signature
from datadiff.semantic_contracts import semantic_contract_lattice_payload
from datadiff.targets import describe_targets
from datadiff.test_obligations import select_executable_test_obligations
from datadiff.util import utc_now
from datadiff.witness_oracle import evaluate_witness_contract, finding_from_witness_result

ExecuteCaseFn = Callable[..., tuple[dict[str, dict[str, Any]], dict[str, Any]]]
ExecuteCasesFn = Callable[..., list[tuple[dict[str, dict[str, Any]], dict[str, Any]]]]
FindingEvaluatorFn = Callable[..., list[Finding]]
AnnotateFindingsFn = Callable[..., None]
CandidateRecheckFn = Callable[[Case, list[str], ExperimentConfig, list[Finding]], dict[str, Any]]
ClockFn = Callable[[], float]


def _elapsed_ms(started: float, clock_fn: ClockFn) -> float:
    return max(0.0, (clock_fn() - started) * 1000)


def _record_execution_elapsed(
    stage_profile: dict[str, float],
    *,
    elapsed_ms: float,
    backend_reported_ms: float,
    parallel_execution_active: bool,
) -> None:
    elapsed_ms = max(0.0, float(elapsed_ms))
    backend_reported_ms = max(0.0, float(backend_reported_ms))
    if parallel_execution_active:
        stage_profile["backend_execution_ms"] += elapsed_ms
        return
    stage_profile["backend_execution_ms"] += min(elapsed_ms, backend_reported_ms)
    stage_profile["normalize_ms"] += max(0.0, elapsed_ms - backend_reported_ms)


def _merge_nested_stage_profile(
    stage_profile: dict[str, float],
    payload: Mapping[str, Any] | None,
) -> float:
    if not isinstance(payload, Mapping):
        return 0.0
    accounted_ms = 0.0
    for key in STAGE_PROFILE_KEYS:
        if key == "total_case_wall_ms":
            continue
        try:
            value = max(0.0, float(payload.get(key, 0.0) or 0.0))
        except (TypeError, ValueError):
            value = 0.0
        stage_profile[key] += value
        accounted_ms += value
    return accounted_ms


def _merge_nested_cost_profile(
    profile: dict[str, float],
    payload: Mapping[str, Any] | None,
) -> float:
    if not isinstance(payload, Mapping):
        return 0.0
    accounted_ms = 0.0
    for key in COST_PROFILE_STAGE_KEYS:
        try:
            value = max(0.0, float(payload.get(key, 0.0) or 0.0))
        except (TypeError, ValueError):
            value = 0.0
        profile[key] += value
        accounted_ms += value
    return accounted_ms


def _execution_reuse_payload(
    raw_results: Mapping[str, Any],
    metamorphic_rows: Mapping[str, Any],
) -> dict[str, Any]:
    events: list[dict[str, Any]] = []

    def add_group(label: str, rows: Mapping[str, Any]) -> None:
        for backend, raw in rows.items():
            if not isinstance(raw, Mapping):
                continue
            lookup = str(raw.get("execution_cache_lookup", "") or "")
            cache_hit = bool(raw.get("execution_cache_hit", False))
            events.append(
                {
                    "label": label,
                    "backend": str(backend),
                    "cache_hit": cache_hit,
                    "cache_lookup": lookup or "not_applicable",
                    "cache_key": str(raw.get("execution_cache_key", "") or ""),
                    "entry_bytes": max(
                        0,
                        int(raw.get("execution_cache_entry_bytes", 0) or 0),
                    ),
                    "reuse_reason": (
                        "complete_key_match"
                        if cache_hit
                        else "no_matching_complete_key"
                        if lookup == "miss"
                        else "cache_disabled_or_fresh_isolation"
                    ),
                    "provenance": dict(raw.get("execution_cache_provenance", {}) or {}),
                }
            )

    add_group("base", raw_results)
    for variant_name, variant in metamorphic_rows.items():
        if isinstance(variant, Mapping) and isinstance(
            variant.get("raw_results"), Mapping
        ):
            add_group(f"metamorphic:{variant_name}", variant["raw_results"])
    hit_count = sum(int(event["cache_hit"]) for event in events)
    miss_count = sum(int(event["cache_lookup"] == "miss") for event in events)
    return {
        "schema_version": "execution-reuse-provenance-v1",
        "events": events,
        "summary": {
            "event_count": len(events),
            "cache_hit_count": hit_count,
            "cache_miss_count": miss_count,
            "not_applicable_count": len(events) - hit_count - miss_count,
            "reused_entry_bytes": sum(
                int(event["entry_bytes"])
                for event in events
                if event["cache_hit"]
            ),
        },
    }


def _artifact_raw_results(
    raw_results: Mapping[str, Mapping[str, Any]],
    finding_diagnostics: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    out = {backend: dict(raw) for backend, raw in raw_results.items()}
    plans = finding_diagnostics.get("plans", {})
    if not isinstance(plans, Mapping):
        return out
    for backend, plan in plans.items():
        if backend in out and isinstance(plan, Mapping):
            out[backend]["physical_plan"] = dict(plan)
    return out


def parallel_backend_execution_active(config: ExperimentConfig, backends: list[str]) -> bool:
    return (
        bool(getattr(config, "enable_parallel_backend_execution", True))
        and len(backends) > 1
        and len(set(backends)) == len(backends)
    )


def execute_case_for_run_loaded(
    case: Case,
    backends: list[str],
    config: ExperimentConfig,
    backend_instances: dict[str, Any] | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    return _execute_case_impl(
        case,
        backends,
        config,
        backend_instances=backend_instances,
        parallel=config.enable_parallel_backend_execution,
    )


def run_loaded_case_impl(
    case: Case,
    backends: list[str],
    *,
    config: ExperimentConfig | None = None,
    save_artifact: bool = True,
    backend_instances: dict[str, Any] | None = None,
    environment: dict[str, str] | None = None,
    target_specs: list[dict[str, Any]] | None = None,
    config_payload: dict[str, Any] | None = None,
    metamorphic_relation_order: list[str] | tuple[str, ...] | None = None,
    execute_case_fn: ExecuteCaseFn = execute_case_for_run_loaded,
    execute_cases_fn: ExecuteCasesFn | None = None,
    parallel_backend_execution_active_fn: Callable[[ExperimentConfig, list[str]], bool] = parallel_backend_execution_active,
    evaluate_case_fn: FindingEvaluatorFn = evaluate_case,
    all_metamorphic_variants_fn: Callable[[Case], Any] = all_metamorphic_variants,
    select_metamorphic_variants_fn: Callable[..., Any] = select_metamorphic_variants,
    evaluate_metamorphic_variants_fn: FindingEvaluatorFn = evaluate_metamorphic_variants,
    annotate_findings_fn: AnnotateFindingsFn = annotate_findings,
    candidate_recheck_fn: CandidateRecheckFn | None = None,
    save_bug_artifact_fn: Callable[..., Any] = save_bug_artifact,
    collect_environment_fn: Callable[[], dict[str, str]] = collect_environment,
    describe_targets_fn: Callable[[list[str]], list[dict[str, Any]]] = describe_targets,
    perf_counter_fn: ClockFn = time.perf_counter,
    process_cpu_fn: ClockFn = time.process_time,
) -> dict[str, Any]:
    config = config or ExperimentConfig()
    method_policy = config.method_policy
    comparison_mode = method_policy.semantic.comparison_mode
    obligation_mode = method_policy.semantic.obligation_mode
    obligation_priority_mode = method_policy.semantic.obligation_priority_mode
    raw_config_payload = (
        dict(config_payload)
        if isinstance(config_payload, dict)
        else _config_payload_with_effective_guidance_targets(config)
    )
    raw_config_payload.setdefault("method_arm", config.method_arm)
    raw_config_payload.setdefault("method_arm_overrides", dict(config.method_arm_overrides))
    raw_config_payload["method_arm_manifest"] = config.method_arm_manifest
    resolved_config_payload = finalize_config_payload(raw_config_payload)
    resolved_environment = (
        dict(environment) if environment is not None else collect_environment_fn()
    )
    resolved_target_specs = (
        list(target_specs) if target_specs is not None else describe_targets_fn(backends)
    )
    resolved_program = resolve_case_program(case, config)
    program_ir_summary = resolved_program.summary()
    experiment_manifest = build_case_experiment_manifest(
        case_payload=case.to_dict(),
        backends=backends,
        target_specs=resolved_target_specs,
        environment=resolved_environment,
        config_payload=resolved_config_payload,
        ir_mode=resolved_program.ir_mode,
        program_ir_digest=resolved_program.program_ir_digest,
    )
    started = perf_counter_fn()
    process_started = process_cpu_fn()
    stage_profile = _empty_stage_profile()
    wall_time_profile = _empty_wall_time_profile()
    process_cpu_profile = _empty_process_cpu_profile()
    execute_started = perf_counter_fn()
    execute_process_started = process_cpu_fn()
    raw_results, normalized = execute_case_fn(
        case,
        backends,
        config,
        backend_instances=backend_instances,
    )
    base_diagnostic_refs = build_execution_diagnostic_refs(
        case,
        backends,
        case_digest=str(experiment_manifest["case_digest"]),
        raw_results=raw_results,
        normalized_results=normalized,
        target_specs=resolved_target_specs,
        environment=resolved_environment,
        optimizer_config=resolved_config_payload,
        required_capabilities=(
            tuple(resolved_program.ccs_ir.required_capabilities)
            if resolved_program.ccs_ir is not None
            else None
        ),
    )
    execute_elapsed = _elapsed_ms(execute_started, perf_counter_fn)
    execute_process_cpu_ms = _elapsed_ms(execute_process_started, process_cpu_fn)
    wall_time_profile["execution_pipeline_ms"] += execute_elapsed
    process_cpu_profile["execution_pipeline_ms"] += execute_process_cpu_ms
    parallel_execution_active = parallel_backend_execution_active_fn(config, backends)
    base_backend_reported_ms = backend_reported_ms_from_raw(raw_results)
    base_backend_calls = backend_calls_from_raw(raw_results)
    base_execution_cache_hits = execution_cache_hits_from_raw(raw_results)
    _record_execution_elapsed(
        stage_profile,
        elapsed_ms=execute_elapsed,
        backend_reported_ms=base_backend_reported_ms,
        parallel_execution_active=parallel_execution_active,
    )
    execution_profile = {
        "timing_schema_version": "execution-timing-v3",
        "wall_clock": "time.perf_counter",
        "process_cpu_clock": "time.process_time",
        "backend_reported_time_field": "raw_results[*].duration_ms",
        "method_arm_id": experiment_manifest["method_arm_id"],
        "comparison_mode": comparison_mode,
        "obligation_mode": obligation_mode,
        "obligation_priority_mode": obligation_priority_mode,
        "ir_mode": resolved_program.ir_mode,
        "program_ir_digest": resolved_program.program_ir_digest,
        "parallel_backend_execution": parallel_execution_active,
        "backend_wall_ms": execute_elapsed,
        "base_backend_wall_ms": execute_elapsed,
        "base_execution_pipeline_process_cpu_ms": execute_process_cpu_ms,
        "base_backend_reported_total_ms": base_backend_reported_ms,
        "backend_reported_total_ms": base_backend_reported_ms,
        "backend_count": len(backends),
        "base_backend_calls": base_backend_calls,
        "base_execution_cache_hits": base_execution_cache_hits,
        "backend_calls": base_backend_calls,
        "execution_cache_hits": base_execution_cache_hits,
    }

    differential_findings: list[Finding] = []
    metamorphic_findings: list[Finding] = []
    differential_oracle_ms = 0.0
    differential_oracle_process_cpu_ms = 0.0
    if config.enable_differential_oracle:
        differential_started = perf_counter_fn()
        differential_process_started = process_cpu_fn()
        differential_findings = (
            evaluate_case_fn(case, normalized, comparison_mode=comparison_mode)
            if evaluate_case_fn is evaluate_case
            else evaluate_case_fn(case, normalized)
        )
        differential_oracle_ms = _elapsed_ms(differential_started, perf_counter_fn)
        differential_oracle_process_cpu_ms = _elapsed_ms(
            differential_process_started,
            process_cpu_fn,
        )
    metamorphic_rows: dict[str, Any] = {}
    metamorphic_selection: dict[str, Any] = {}
    test_obligation_selection: dict[str, Any] = {
        "mode": obligation_mode,
        "priority_mode": obligation_priority_mode,
        "registry_applied": False,
    }
    metamorphic_backend_reported_ms = 0.0
    metamorphic_backend_calls = 0
    metamorphic_execution_cache_hits = 0
    metamorphic_backend_wall_ms = 0.0
    metamorphic_execution_process_cpu_ms = 0.0
    metamorphic_construction_ms = 0.0
    metamorphic_construction_process_cpu_ms = 0.0
    metamorphic_oracle_ms = 0.0
    metamorphic_oracle_process_cpu_ms = 0.0
    if config.enable_metamorphic_oracle:
        variant_results = {}
        construction_started = perf_counter_fn()
        construction_process_started = process_cpu_fn()
        if (
            obligation_mode == "ccs_guided"
            and resolved_program.ccs_ir is not None
            and all_metamorphic_variants_fn is all_metamorphic_variants
        ):
            obligation_selection = select_executable_test_obligations(
                case,
                resolved_program.ccs_ir,
                relation_order=(
                    metamorphic_relation_order or config.metamorphic_relation_order
                ),
                priority_mode=obligation_priority_mode,
            )
            applicable_variants = list(obligation_selection.variants)
            test_obligation_selection = {
                **obligation_selection.to_dict(),
                "registry_applied": True,
            }
        else:
            applicable_variants = list(all_metamorphic_variants_fn(case))
        variants = select_metamorphic_variants_fn(
            applicable_variants,
            limit=max(0, config.metamorphic_variant_limit),
            relation_order=metamorphic_relation_order or config.metamorphic_relation_order,
        )
        metamorphic_construction_ms += _elapsed_ms(construction_started, perf_counter_fn)
        metamorphic_construction_process_cpu_ms += _elapsed_ms(
            construction_process_started,
            process_cpu_fn,
        )
        metamorphic_execution_started = perf_counter_fn()
        metamorphic_execution_process_started = process_cpu_fn()
        variant_executions = (
            execute_cases_fn(
                [variant.case for variant in variants],
                backends,
                config,
                backend_instances=backend_instances,
            )
            if variants and execute_cases_fn is not None
            else [
                execute_case_fn(
                    variant.case,
                    backends,
                    config,
                    backend_instances=backend_instances,
                )
                for variant in variants
            ]
        )
        metamorphic_backend_wall_ms = _elapsed_ms(
            metamorphic_execution_started,
            perf_counter_fn,
        )
        metamorphic_execution_process_cpu_ms = _elapsed_ms(
            metamorphic_execution_process_started,
            process_cpu_fn,
        )
        wall_time_profile["execution_pipeline_ms"] += metamorphic_backend_wall_ms
        process_cpu_profile["execution_pipeline_ms"] += (
            metamorphic_execution_process_cpu_ms
        )
        assembly_started = perf_counter_fn()
        assembly_process_started = process_cpu_fn()
        for variant, (variant_raw, variant_norm) in zip(
            variants,
            variant_executions,
            strict=True,
        ):
            variant_program = resolve_case_program(variant.case, config)
            variant_manifest = build_case_experiment_manifest(
                case_payload=variant.case.to_dict(),
                backends=backends,
                target_specs=resolved_target_specs,
                environment=resolved_environment,
                config_payload=resolved_config_payload,
                ir_mode=variant_program.ir_mode,
                program_ir_digest=variant_program.program_ir_digest,
            )
            variant_diagnostic_refs = build_execution_diagnostic_refs(
                variant.case,
                backends,
                case_digest=str(variant_manifest["case_digest"]),
                raw_results=variant_raw,
                normalized_results=variant_norm,
                target_specs=resolved_target_specs,
                environment=resolved_environment,
                optimizer_config=resolved_config_payload,
                required_capabilities=(
                    tuple(variant_program.ccs_ir.required_capabilities)
                    if variant_program.ccs_ir is not None
                    else None
                ),
            )
            variant_results[variant.name] = variant_norm
            metamorphic_backend_reported_ms += backend_reported_ms_from_raw(variant_raw)
            metamorphic_backend_calls += backend_calls_from_raw(variant_raw)
            metamorphic_execution_cache_hits += execution_cache_hits_from_raw(variant_raw)
            metamorphic_rows[variant.name] = {
                "relation": variant.relation,
                "case": variant.case.to_dict(),
                "program_ir": variant_program.summary(),
                "experiment_manifest": variant_manifest,
                "raw_results": variant_raw,
                "normalized": {k: v.to_dict() for k, v in variant_norm.items()},
                "osc_diagnostic_refs": variant_diagnostic_refs,
            }
        metamorphic_construction_ms += _elapsed_ms(assembly_started, perf_counter_fn)
        metamorphic_construction_process_cpu_ms += _elapsed_ms(
            assembly_process_started,
            process_cpu_fn,
        )
        _record_execution_elapsed(
            stage_profile,
            elapsed_ms=metamorphic_backend_wall_ms,
            backend_reported_ms=metamorphic_backend_reported_ms,
            parallel_execution_active=parallel_execution_active,
        )
        metamorphic_oracle_started = perf_counter_fn()
        metamorphic_oracle_process_started = process_cpu_fn()
        metamorphic_findings = (
            evaluate_metamorphic_variants_fn(
                case,
                normalized,
                variant_results,
                comparison_mode=comparison_mode,
            )
            if evaluate_metamorphic_variants_fn is evaluate_metamorphic_variants
            else evaluate_metamorphic_variants_fn(case, normalized, variant_results)
        )
        selected_variant_keys = {
            (str(variant.name), str(variant.relation)) for variant in variants
        }
        applicable_relations = list(
            dict.fromkeys(str(variant.relation) for variant in applicable_variants)
        )
        executed_relations = list(
            dict.fromkeys(
                row["relation"]
                for row in metamorphic_rows.values()
                if isinstance(row, dict)
            )
        )
        omitted_variants = [
            variant
            for variant in applicable_variants
            if (str(variant.name), str(variant.relation)) not in selected_variant_keys
        ]
        metamorphic_selection = {
            "obligation_mode": obligation_mode,
            "obligation_priority_mode": obligation_priority_mode,
            "test_obligations": test_obligation_selection,
            "relation_order": list(
                metamorphic_relation_order or config.metamorphic_relation_order
            ),
            "applicable_relations": applicable_relations,
            "executed_relations": executed_relations,
            "omitted_relations": list(
                dict.fromkeys(str(variant.relation) for variant in omitted_variants)
            ),
            "applicable_variant_count": len(applicable_variants),
            "executed_variant_count": len(variants),
            "omitted_variant_count": len(omitted_variants),
            "executed_variants": [str(variant.name) for variant in variants],
            "omitted_variants": [str(variant.name) for variant in omitted_variants],
            "variant_limit": max(0, int(config.metamorphic_variant_limit)),
        }
        metamorphic_oracle_ms = _elapsed_ms(
            metamorphic_oracle_started,
            perf_counter_fn,
        )
        metamorphic_oracle_process_cpu_ms = _elapsed_ms(
            metamorphic_oracle_process_started,
            process_cpu_fn,
        )
    findings = [*differential_findings, *metamorphic_findings]
    witness_started = perf_counter_fn()
    witness_process_started = process_cpu_fn()
    witness_result = evaluate_witness_contract(
        case,
        normalized,
        enabled=bool(config.enable_witness_oracle),
        infer=bool(config.enable_witness_oracle),
    )
    witness_finding = (
        finding_from_witness_result(case, witness_result)
        if config.enable_witness_oracle
        else None
    )
    witness_oracle_ms = _elapsed_ms(witness_started, perf_counter_fn)
    witness_oracle_process_cpu_ms = _elapsed_ms(
        witness_process_started,
        process_cpu_fn,
    )
    if witness_finding is not None:
        findings.append(witness_finding)

    recheck_requested = False
    recheck_config = config
    classification_started = perf_counter_fn()
    classification_process_started = process_cpu_fn()
    if findings:
        annotate_findings_fn(
            case,
            findings,
            normalized=normalized,
            raw_results=raw_results,
            config=resolved_config_payload,
            backends=backends,
        )
        oracle_cross_validation = cross_validate_oracle_findings(
            differential_findings,
            metamorphic_findings,
        ).to_dict()
        countable_findings = _countable_finding_objects(findings)
        recheck_findings = (
            [
                finding
                for finding in countable_findings
                if not is_known_saturated_candidate_issue_finding(
                    finding.to_dict(),
                    config.known_saturated_bug_families,
                )
            ]
            if config.evidence_tier == "screening"
            else list(countable_findings)
        )
        if recheck_findings and candidate_recheck_fn is not None:
            relation_order = list(metamorphic_selection.get("relation_order", []) or [])
            if relation_order:
                recheck_config_data = config.to_dict()
                recheck_config_data["metamorphic_relation_order"] = relation_order
                recheck_config = ExperimentConfig.from_payload(recheck_config_data)
            recheck_requested = True
            recheck = {
                "enabled": False,
                "attempts": 0,
                "reproduced_keys": [],
                "non_reproduced_keys": [],
                "pending": True,
            }
        elif recheck_findings:
            recheck = {
                "enabled": False,
                "attempts": 0,
                "reproduced_keys": [],
                "non_reproduced_keys": [],
                "skip_reason": "recheck_callback_unavailable",
            }
        elif countable_findings:
            recheck = {
                "enabled": False,
                "attempts": 0,
                "reproduced_keys": [],
                "non_reproduced_keys": [],
                "skip_reason": (
                    "known_saturated_findings_do_not_require_screening_recheck"
                ),
            }
        else:
            recheck = {
                "enabled": False,
                "attempts": 0,
                "reproduced_keys": [],
                "non_reproduced_keys": [],
                "skip_reason": "no_countable_candidate_findings",
            }
    else:
        oracle_cross_validation = cross_validate_oracle_findings([], []).to_dict()
        countable_findings = []
        recheck_findings = []
        recheck = {"enabled": False, "attempts": 0, "reproduced_keys": [], "non_reproduced_keys": []}
    classification_annotation_ms = _elapsed_ms(classification_started, perf_counter_fn)
    classification_annotation_process_cpu_ms = _elapsed_ms(
        classification_process_started,
        process_cpu_fn,
    )

    finding_diagnostics: dict[str, Any] = {}
    if (
        recheck_findings
        and config.evidence_tier == "screening"
        and method_policy.execution.plan_collection.mode == "tiered"
    ):
        finding_diagnostics = {
            "schema_version": "finding-plan-sidecar-v1",
            "evidence_tier": "fresh_confirmation",
            "status": "pending_fresh_recheck",
            "plans": {},
            "skip_reason": "full_plan_reuses_fresh_confirmation_execution",
        }

    localization_report: dict[str, Any] = {
        "schema_version": "prefix-first-divergence-localization-v1",
        "enabled": False,
        "attempted": False,
        "skip_reason": "localization_mode_reducer_only",
        "extra_backend_calls": 0,
        "extra_backend_reported_ms": 0.0,
        "extra_execution_cache_hits": 0,
        "wall_ms": 0.0,
        "process_cpu_ms": 0.0,
    }
    localization_findings = (
        recheck_findings
        if config.evidence_tier == "screening"
        else findings
    )
    if (
        method_policy.evidence.localization_mode == "prefix_adaptive"
        and localization_findings
    ):
        localization_report = localize_first_divergence(
            case,
            backends=backends,
            config=config,
            raw_results=raw_results,
            normalized=normalized,
            findings=localization_findings,
            execute_prefix_fn=execute_case_fn,
            backend_instances=backend_instances,
            perf_counter_fn=perf_counter_fn,
            process_cpu_fn=process_cpu_fn,
        )
        localization_wall_ms = max(
            0.0, float(localization_report.get("wall_ms", 0.0) or 0.0)
        )
        localization_process_cpu_ms = max(
            0.0,
            float(localization_report.get("process_cpu_ms", 0.0) or 0.0),
        )
        localization_backend_reported_ms = max(
            0.0,
            float(
                localization_report.get("extra_backend_reported_ms", 0.0) or 0.0
            ),
        )
        wall_time_profile["execution_pipeline_ms"] += localization_wall_ms
        process_cpu_profile["execution_pipeline_ms"] += localization_process_cpu_ms
        _record_execution_elapsed(
            stage_profile,
            elapsed_ms=localization_wall_ms,
            backend_reported_ms=localization_backend_reported_ms,
            parallel_execution_active=parallel_execution_active,
        )
    elif method_policy.evidence.localization_mode == "prefix_adaptive":
        if (
            config.evidence_tier == "screening"
            and countable_findings
            and not recheck_findings
        ):
            localization_report["skip_reason"] = (
                "known_saturated_findings_do_not_require_screening_localization"
            )
        else:
            localization_report["skip_reason"] = (
                "no_countable_candidate_findings_at_screening_tier"
                if config.evidence_tier == "screening"
                else "no_findings"
            )

    candidate_recheck_wall_ms = 0.0
    candidate_recheck_process_cpu_ms = 0.0
    if recheck_requested and candidate_recheck_fn is not None:
        recheck_started = perf_counter_fn()
        recheck_process_started = process_cpu_fn()
        recheck = candidate_recheck_fn(case, backends, recheck_config, recheck_findings)
        candidate_recheck_wall_ms = _elapsed_ms(recheck_started, perf_counter_fn)
        candidate_recheck_process_cpu_ms = _elapsed_ms(
            recheck_process_started,
            process_cpu_fn,
        )
        post_recheck_started = perf_counter_fn()
        post_recheck_process_started = process_cpu_fn()
        countable_findings = _countable_finding_objects(findings)
        classification_annotation_ms += _elapsed_ms(
            post_recheck_started,
            perf_counter_fn,
        )
        classification_annotation_process_cpu_ms += _elapsed_ms(
            post_recheck_process_started,
            process_cpu_fn,
        )
        attempt_summaries = recheck.get("attempt_summaries", ())
        if finding_diagnostics and isinstance(attempt_summaries, list) and attempt_summaries:
            first_attempt = attempt_summaries[0]
            plans = (
                first_attempt.get("physical_plans", {})
                if isinstance(first_attempt, Mapping)
                else {}
            )
            if isinstance(plans, Mapping) and plans:
                finding_diagnostics = {
                    "schema_version": "finding-plan-sidecar-v1",
                    "evidence_tier": "fresh_confirmation",
                    "status": "collected",
                    "plans": dict(plans),
                    "source_attempt": 1,
                    "additional_backend_calls": 0,
                }

    local_oracle_ms = (
        differential_oracle_ms
        + metamorphic_construction_ms
        + metamorphic_oracle_ms
        + witness_oracle_ms
        + classification_annotation_ms
    )
    local_oracle_process_cpu_ms = (
        differential_oracle_process_cpu_ms
        + metamorphic_construction_process_cpu_ms
        + metamorphic_oracle_process_cpu_ms
        + witness_oracle_process_cpu_ms
        + classification_annotation_process_cpu_ms
    )
    stage_profile["oracle_classification_ms"] += local_oracle_ms
    wall_time_profile["oracle_classification_ms"] += local_oracle_ms
    process_cpu_profile["oracle_classification_ms"] += local_oracle_process_cpu_ms
    recheck_timing_source = "none"
    recheck_cost_profile_source = "none"
    if recheck_requested:
        nested_recheck_accounted_ms = _merge_nested_stage_profile(
            stage_profile,
            recheck.get("stage_profile_totals") if isinstance(recheck, Mapping) else None,
        )
        if nested_recheck_accounted_ms > 0.0:
            recheck_timing_source = "nested_stage_profile"
            stage_profile["logging_artifact_ms"] += max(
                0.0,
                candidate_recheck_wall_ms - nested_recheck_accounted_ms,
            )
        else:
            recheck_timing_source = "callback_wall_fallback"
            stage_profile["backend_execution_ms"] += candidate_recheck_wall_ms
        nested_recheck_wall_ms = _merge_nested_cost_profile(
            wall_time_profile,
            recheck.get("wall_time_profile_totals")
            if isinstance(recheck, Mapping)
            else None,
        )
        nested_recheck_process_cpu_ms = _merge_nested_cost_profile(
            process_cpu_profile,
            recheck.get("process_cpu_profile_totals")
            if isinstance(recheck, Mapping)
            else None,
        )
        if nested_recheck_wall_ms > 0.0 or nested_recheck_process_cpu_ms > 0.0:
            recheck_cost_profile_source = "nested_cost_profiles"
            wall_time_profile["logging_artifact_ms"] += max(
                0.0,
                candidate_recheck_wall_ms - nested_recheck_wall_ms,
            )
            process_cpu_profile["logging_artifact_ms"] += max(
                0.0,
                candidate_recheck_process_cpu_ms - nested_recheck_process_cpu_ms,
            )
        else:
            recheck_cost_profile_source = "callback_fallback"
            wall_time_profile["execution_pipeline_ms"] += candidate_recheck_wall_ms
            process_cpu_profile["execution_pipeline_ms"] += (
                candidate_recheck_process_cpu_ms
            )

    candidate_recheck_backend_reported_ms = execution_profile_backend_reported_ms(recheck)
    candidate_recheck_backend_calls = execution_profile_backend_calls(recheck)
    localization_backend_reported_ms = max(
        0.0,
        float(localization_report.get("extra_backend_reported_ms", 0.0) or 0.0),
    )
    localization_backend_calls = max(
        0, int(localization_report.get("extra_backend_calls", 0) or 0)
    )
    localization_cache_hits = max(
        0,
        int(localization_report.get("extra_execution_cache_hits", 0) or 0),
    )
    execution_profile["metamorphic_backend_reported_total_ms"] = metamorphic_backend_reported_ms
    execution_profile["candidate_recheck_backend_reported_total_ms"] = (
        candidate_recheck_backend_reported_ms
    )
    execution_profile["backend_reported_total_ms"] = (
        base_backend_reported_ms
        + metamorphic_backend_reported_ms
        + localization_backend_reported_ms
        + candidate_recheck_backend_reported_ms
    )
    execution_profile["metamorphic_backend_calls"] = metamorphic_backend_calls
    execution_profile["metamorphic_execution_cache_hits"] = metamorphic_execution_cache_hits
    execution_profile["candidate_recheck_backend_calls"] = (
        candidate_recheck_backend_calls
    )
    execution_profile["backend_calls"] = (
        base_backend_calls
        + metamorphic_backend_calls
        + localization_backend_calls
        + candidate_recheck_backend_calls
    )
    execution_profile["execution_cache_hits"] = (
        base_execution_cache_hits
        + metamorphic_execution_cache_hits
        + localization_cache_hits
    )
    execution_profile.update(
        {
            "differential_oracle_ms": differential_oracle_ms,
            "differential_oracle_process_cpu_ms": differential_oracle_process_cpu_ms,
            "metamorphic_construction_ms": metamorphic_construction_ms,
            "metamorphic_construction_process_cpu_ms": (
                metamorphic_construction_process_cpu_ms
            ),
            "metamorphic_backend_wall_ms": metamorphic_backend_wall_ms,
            "metamorphic_execution_pipeline_process_cpu_ms": (
                metamorphic_execution_process_cpu_ms
            ),
            "metamorphic_oracle_ms": metamorphic_oracle_ms,
            "metamorphic_oracle_process_cpu_ms": metamorphic_oracle_process_cpu_ms,
            "witness_oracle_ms": witness_oracle_ms,
            "witness_oracle_process_cpu_ms": witness_oracle_process_cpu_ms,
            "classification_annotation_ms": classification_annotation_ms,
            "classification_annotation_process_cpu_ms": (
                classification_annotation_process_cpu_ms
            ),
            "candidate_recheck_wall_ms": candidate_recheck_wall_ms,
            "candidate_recheck_process_cpu_ms": candidate_recheck_process_cpu_ms,
            "candidate_recheck_timing_source": recheck_timing_source,
            "candidate_recheck_cost_profile_source": recheck_cost_profile_source,
            "localization_backend_wall_ms": float(
                localization_report.get("wall_ms", 0.0) or 0.0
            ),
            "localization_execution_pipeline_process_cpu_ms": float(
                localization_report.get("process_cpu_ms", 0.0) or 0.0
            ),
            "localization_backend_reported_total_ms": localization_backend_reported_ms,
            "localization_backend_calls": localization_backend_calls,
            "localization_execution_cache_hits": localization_cache_hits,
            "direct_backend_wall_ms": (
                execute_elapsed
                + metamorphic_backend_wall_ms
                + float(localization_report.get("wall_ms", 0.0) or 0.0)
            ),
            "direct_execution_pipeline_process_cpu_ms": (
                execute_process_cpu_ms
                + metamorphic_execution_process_cpu_ms
                + float(localization_report.get("process_cpu_ms", 0.0) or 0.0)
            ),
            "direct_oracle_wall_ms": local_oracle_ms,
            "direct_oracle_process_cpu_ms": local_oracle_process_cpu_ms,
        }
    )

    row_finalization_started = perf_counter_fn()
    row_finalization_process_started = process_cpu_fn()
    row = {
        "run_at": utc_now(),
        "case": case.to_dict(),
        "targets": resolved_target_specs,
        "raw_results": raw_results,
        "normalized": {k: v.to_dict() for k, v in normalized.items()},
        "osc_diagnostic_refs": base_diagnostic_refs,
        "metamorphic": metamorphic_rows,
        "metamorphic_selection": metamorphic_selection,
        "test_obligation_selection": test_obligation_selection,
        "witness_oracle": witness_result.to_dict(),
        "oracle_cross_validation": oracle_cross_validation,
        "findings": [f.to_dict() for f in findings],
        "candidate_recheck": recheck,
        "localization": localization_report,
        "finding_diagnostics": finding_diagnostics,
        "config": resolved_config_payload,
        "method_arm": config.method_arm_manifest,
        "experiment_manifest": experiment_manifest,
        "program_ir": program_ir_summary,
        "ccs_ir_summary": (
            program_ir_summary if resolved_program.ccs_ir is not None else {}
        ),
        "ccs_ir_digest": (
            resolved_program.program_ir_digest if resolved_program.ccs_ir is not None else ""
        ),
        "ccs_ir": (
            resolved_program.ccs_ir.to_dict() if resolved_program.ccs_ir is not None else None
        ),
        "environment": resolved_environment,
        "status": "bug" if countable_findings else "ok",
        "duration_ms": stage_profile["total_case_wall_ms"],
        "stage_profile": stage_profile,
        "wall_time_profile": wall_time_profile,
        "process_cpu_profile": process_cpu_profile,
        "execution_profile": execution_profile,
    }
    row["behavior_signature"] = behavior_signature(row)
    row["discovery_signature"] = discovery_signature(row)
    row["execution_reuse"] = _execution_reuse_payload(raw_results, metamorphic_rows)
    disagreement_descriptor = compute_descriptor(normalized, findings).to_dict()
    row["disagreement_descriptor"] = disagreement_descriptor
    _attach_disagreement_descriptor_to_row_case(row, disagreement_descriptor)
    case_fingerprint = compute_fingerprint(case, _fingerprint_anchor_result(normalized)).to_dict()
    row["case_fingerprint"] = case_fingerprint
    _attach_case_fingerprint_to_row_case(row, case_fingerprint)
    semantic_contract_lattice = semantic_contract_lattice_payload(case)
    row["semantic_contract_lattice"] = semantic_contract_lattice
    _attach_semantic_contract_lattice_to_row_case(row, semantic_contract_lattice)
    method_settings = config.method_policy
    p5_descriptor_active = bool(
        method_settings.execution.plan_guidance == "physical"
        or method_settings.generation.mode
        in {
            "goal_first",
            "goal_first_witness",
            "goal_first_witness_v2",
            "goal_first_witness_v3",
            "goal_first_witness_v4",
            "goal_first_witness_v5",
            "goal_first_witness_v6",
            "goal_first_witness_v7",
            "goal_first_witness_global_v1",
            "goal_first_witness_global_v2",
            "goal_first_witness_global_v3",
            "goal_first_witness_global_v4",
        }
        or method_settings.generation.boundary_mode == "fault_model_targeted"
        or method_settings.semantic.obligation_priority_mode == "ccs_risk_priority_v2"
        or method_settings.generation.corpus_mode == "semantic_plan_qd"
        or method_settings.evidence.localization_mode == "prefix_adaptive"
    )
    interaction_descriptor: dict[str, Any] = {}
    if p5_descriptor_active and resolved_program.ccs_ir is not None:
        descriptor = build_interaction_descriptor(
            resolved_program.ccs_ir,
            raw_results,
        )
        interaction_descriptor = (
            descriptor.compact_dict()
            if config.method_arm == "p8_candidate_v1"
            and config.evidence_tier == "screening"
            and not countable_findings
            else descriptor.to_dict()
        )
        row["interaction_descriptor"] = interaction_descriptor
        _attach_interaction_descriptor_to_row_case(row, interaction_descriptor)
    semantic_comparison_profile = comparison_profile_payload(case)
    semantic_comparison_profile["applied_comparison_mode"] = comparison_mode
    row["semantic_comparison_profile"] = semantic_comparison_profile
    row_case = row.get("case", {})
    if isinstance(row_case, dict):
        row_case_metadata = row_case.setdefault("metadata", {})
        if isinstance(row_case_metadata, dict):
            row_case_metadata["semantic_comparison_profile"] = semantic_comparison_profile
    if isinstance(case.metadata, dict):
        case.metadata["disagreement_descriptor"] = disagreement_descriptor
        case.metadata["case_fingerprint"] = case_fingerprint
        case.metadata["semantic_contract_lattice"] = semantic_contract_lattice
        if interaction_descriptor:
            case.metadata["interaction_descriptor"] = interaction_descriptor
        case.metadata["semantic_comparison_profile"] = semantic_comparison_profile
    row_finalization_ms = _elapsed_ms(row_finalization_started, perf_counter_fn)
    row_finalization_process_cpu_ms = _elapsed_ms(
        row_finalization_process_started,
        process_cpu_fn,
    )
    stage_profile["logging_artifact_ms"] += row_finalization_ms
    wall_time_profile["logging_artifact_ms"] += row_finalization_ms
    process_cpu_profile["logging_artifact_ms"] += row_finalization_process_cpu_ms
    artifact_write_ms = 0.0
    artifact_write_process_cpu_ms = 0.0
    if countable_findings and save_artifact and config.enable_artifact:
        artifact_started = perf_counter_fn()
        artifact_process_started = process_cpu_fn()
        artifact_config_payload = dict(resolved_config_payload)
        artifact_config_payload["experiment_manifest"] = experiment_manifest
        bug_dir = save_bug_artifact_fn(
            case,
            raw_results=_artifact_raw_results(raw_results, finding_diagnostics),
            normalized={k: v.to_dict() for k, v in normalized.items()},
            findings=countable_findings,
            config=artifact_config_payload,
            ccs_ir=resolved_program.ccs_ir,
        )
        artifact_write_ms = _elapsed_ms(artifact_started, perf_counter_fn)
        artifact_write_process_cpu_ms = _elapsed_ms(
            artifact_process_started,
            process_cpu_fn,
        )
        stage_profile["logging_artifact_ms"] += artifact_write_ms
        wall_time_profile["logging_artifact_ms"] += artifact_write_ms
        process_cpu_profile["logging_artifact_ms"] += artifact_write_process_cpu_ms
        row["bug_dir"] = str(bug_dir)

    observed_case_wall_ms = _elapsed_ms(started, perf_counter_fn)
    accounted_before_gap_ms = sum(
        stage_profile[key]
        for key in STAGE_PROFILE_KEYS
        if key != "total_case_wall_ms"
    )
    unattributed_case_wall_ms = max(0.0, observed_case_wall_ms - accounted_before_gap_ms)
    stage_profile["logging_artifact_ms"] += unattributed_case_wall_ms
    stage_profile["total_case_wall_ms"] = sum(
        stage_profile[key]
        for key in STAGE_PROFILE_KEYS
        if key != "total_case_wall_ms"
    )
    wall_accounted_before_gap_ms = sum(
        wall_time_profile[key]
        for key in COST_PROFILE_STAGE_KEYS
    )
    wall_profile_gap_ms = max(
        0.0,
        observed_case_wall_ms - wall_accounted_before_gap_ms,
    )
    wall_time_profile["logging_artifact_ms"] += wall_profile_gap_ms
    wall_time_profile = _wall_time_profile_with_total(wall_time_profile)

    observed_case_process_cpu_ms = _elapsed_ms(process_started, process_cpu_fn)
    process_cpu_accounted_before_gap_ms = sum(
        process_cpu_profile[key]
        for key in COST_PROFILE_STAGE_KEYS
    )
    process_cpu_profile_gap_ms = max(
        0.0,
        observed_case_process_cpu_ms - process_cpu_accounted_before_gap_ms,
    )
    process_cpu_profile["logging_artifact_ms"] += process_cpu_profile_gap_ms
    process_cpu_profile = _process_cpu_profile_with_total(process_cpu_profile)
    row["wall_time_profile"] = wall_time_profile
    row["process_cpu_profile"] = process_cpu_profile
    execution_profile.update(
        {
            "row_finalization_ms": row_finalization_ms,
            "row_finalization_process_cpu_ms": row_finalization_process_cpu_ms,
            "artifact_write_ms": artifact_write_ms,
            "artifact_write_process_cpu_ms": artifact_write_process_cpu_ms,
            "unattributed_case_wall_ms": unattributed_case_wall_ms,
            "unattributed_wall_profile_ms": wall_profile_gap_ms,
            "unattributed_process_cpu_ms": process_cpu_profile_gap_ms,
            "observed_case_wall_ms": observed_case_wall_ms,
            "observed_case_process_cpu_ms": observed_case_process_cpu_ms,
            "accounted_case_wall_ms": stage_profile["total_case_wall_ms"],
            "accounted_wall_profile_ms": wall_time_profile["total_wall_ms"],
            "accounted_case_process_cpu_ms": process_cpu_profile[
                "total_process_cpu_ms"
            ],
            "accounting_overrun_ms": max(
                0.0,
                stage_profile["total_case_wall_ms"] - observed_case_wall_ms,
            ),
            "wall_profile_accounting_overrun_ms": max(
                0.0,
                wall_time_profile["total_wall_ms"] - observed_case_wall_ms,
            ),
            "process_cpu_accounting_overrun_ms": max(
                0.0,
                process_cpu_profile["total_process_cpu_ms"]
                - observed_case_process_cpu_ms,
            ),
            "backend_execution_stage_ms": stage_profile["backend_execution_ms"],
            "normalize_stage_ms": stage_profile["normalize_ms"],
            "oracle_classification_stage_ms": stage_profile["oracle_classification_ms"],
            "logging_artifact_stage_ms": stage_profile["logging_artifact_ms"],
        }
    )
    row["duration_ms"] = stage_profile["total_case_wall_ms"]
    return row
