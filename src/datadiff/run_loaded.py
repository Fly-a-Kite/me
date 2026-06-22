from __future__ import annotations

import time
from typing import Any, Callable

from datadiff.artifact import save_issue_artifact as save_bug_artifact
from datadiff.classification_oracle import annotate_findings
from datadiff.config import ExperimentConfig
from datadiff.disagreement import compute_descriptor
from datadiff.dsl import Case
from datadiff.env import collect_environment
from datadiff.execution import execute_case as _execute_case_impl
from datadiff.fingerprint import compute_fingerprint
from datadiff.metamorphic import (
    all_metamorphic_variants,
    evaluate_metamorphic_variants,
    select_metamorphic_variants,
)
from datadiff.oracle import Finding, evaluate_case
from datadiff.oracle_complex import cross_validate_oracle_findings
from datadiff.run_config import _config_payload_with_effective_guidance_targets
from datadiff.run_findings import _countable_finding_objects
from datadiff.run_logging import STAGE_PROFILE_KEYS, _empty_stage_profile
from datadiff.run_metadata import (
    _attach_case_fingerprint_to_row_case,
    _attach_disagreement_descriptor_to_row_case,
    _attach_semantic_contract_lattice_to_row_case,
    _fingerprint_anchor_result,
)
from datadiff.run_signatures import behavior_signature, discovery_signature
from datadiff.semantic_contracts import semantic_contract_lattice_payload
from datadiff.targets import describe_targets
from datadiff.util import utc_now
from datadiff.witness_oracle import evaluate_witness_contract, finding_from_witness_result

ExecuteCaseFn = Callable[..., tuple[dict[str, dict[str, Any]], dict[str, Any]]]
FindingEvaluatorFn = Callable[..., list[Finding]]
AnnotateFindingsFn = Callable[..., None]
CandidateRecheckFn = Callable[[Case, list[str], ExperimentConfig, list[Finding]], dict[str, Any]]


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
) -> dict[str, Any]:
    config = config or ExperimentConfig()
    resolved_config_payload = (
        dict(config_payload)
        if isinstance(config_payload, dict)
        else _config_payload_with_effective_guidance_targets(config)
    )
    started = time.perf_counter()
    stage_profile = _empty_stage_profile()
    execute_started = time.perf_counter()
    raw_results, normalized = execute_case_fn(
        case,
        backends,
        config,
        backend_instances=backend_instances,
    )
    execute_elapsed = (time.perf_counter() - execute_started) * 1000
    parallel_execution_active = parallel_backend_execution_active_fn(config, backends)
    backend_reported_ms = sum(
        float(result.get("duration_ms", 0.0) or 0.0)
        for result in raw_results.values()
    )
    if parallel_execution_active:
        stage_profile["backend_execution_ms"] += execute_elapsed
    else:
        stage_profile["backend_execution_ms"] += min(execute_elapsed, backend_reported_ms)
        stage_profile["normalize_ms"] += max(0.0, execute_elapsed - backend_reported_ms)
    execution_profile = {
        "parallel_backend_execution": parallel_execution_active,
        "backend_wall_ms": execute_elapsed,
        "backend_reported_total_ms": backend_reported_ms,
        "backend_count": len(backends),
    }

    differential_findings: list[Finding] = []
    metamorphic_findings: list[Finding] = []
    classification_started = time.perf_counter()
    if config.enable_differential_oracle:
        differential_findings = evaluate_case_fn(case, normalized)
    metamorphic_rows: dict[str, Any] = {}
    metamorphic_selection: dict[str, Any] = {}
    if config.enable_metamorphic_oracle:
        variant_results = {}
        variants = select_metamorphic_variants_fn(
            all_metamorphic_variants_fn(case),
            limit=max(0, config.metamorphic_variant_limit),
            relation_order=metamorphic_relation_order or config.metamorphic_relation_order,
        )
        for variant in variants:
            variant_raw, variant_norm = execute_case_fn(
                variant.case,
                backends,
                config,
                backend_instances=backend_instances,
            )
            variant_results[variant.name] = variant_norm
            metamorphic_rows[variant.name] = {
                "relation": variant.relation,
                "case": variant.case.to_dict(),
                "raw_results": variant_raw,
                "normalized": {k: v.to_dict() for k, v in variant_norm.items()},
            }
        metamorphic_findings = evaluate_metamorphic_variants_fn(case, normalized, variant_results)
        if metamorphic_rows:
            metamorphic_selection = {
                "relation_order": list(metamorphic_relation_order or config.metamorphic_relation_order),
                "executed_relations": [row["relation"] for row in metamorphic_rows.values() if isinstance(row, dict)],
                "variant_limit": max(0, int(config.metamorphic_variant_limit)),
            }
    findings = [*differential_findings, *metamorphic_findings]
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
    if witness_finding is not None:
        findings.append(witness_finding)
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
        if countable_findings and candidate_recheck_fn is not None:
            recheck = candidate_recheck_fn(case, backends, config, countable_findings)
            countable_findings = _countable_finding_objects(findings)
        elif countable_findings:
            recheck = {
                "enabled": False,
                "attempts": 0,
                "reproduced_keys": [],
                "non_reproduced_keys": [],
                "skip_reason": "recheck_callback_unavailable",
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
        recheck = {"enabled": False, "attempts": 0, "reproduced_keys": [], "non_reproduced_keys": []}
    stage_profile["oracle_classification_ms"] = (time.perf_counter() - classification_started) * 1000
    stage_profile["total_case_wall_ms"] = (time.perf_counter() - started) * 1000

    row = {
        "run_at": utc_now(),
        "case": case.to_dict(),
        "targets": target_specs if target_specs is not None else describe_targets_fn(backends),
        "raw_results": raw_results,
        "normalized": {k: v.to_dict() for k, v in normalized.items()},
        "metamorphic": metamorphic_rows,
        "metamorphic_selection": metamorphic_selection,
        "witness_oracle": witness_result.to_dict(),
        "oracle_cross_validation": oracle_cross_validation,
        "findings": [f.to_dict() for f in findings],
        "candidate_recheck": recheck,
        "config": resolved_config_payload,
        "environment": environment if environment is not None else collect_environment_fn(),
        "status": "bug" if countable_findings else "ok",
        "duration_ms": stage_profile["total_case_wall_ms"],
        "stage_profile": stage_profile,
        "execution_profile": execution_profile,
    }
    row["behavior_signature"] = behavior_signature(row)
    row["discovery_signature"] = discovery_signature(row)
    disagreement_descriptor = compute_descriptor(normalized, findings).to_dict()
    row["disagreement_descriptor"] = disagreement_descriptor
    _attach_disagreement_descriptor_to_row_case(row, disagreement_descriptor)
    case_fingerprint = compute_fingerprint(case, _fingerprint_anchor_result(normalized)).to_dict()
    row["case_fingerprint"] = case_fingerprint
    _attach_case_fingerprint_to_row_case(row, case_fingerprint)
    semantic_contract_lattice = semantic_contract_lattice_payload(case)
    row["semantic_contract_lattice"] = semantic_contract_lattice
    _attach_semantic_contract_lattice_to_row_case(row, semantic_contract_lattice)
    if isinstance(case.metadata, dict):
        case.metadata["disagreement_descriptor"] = disagreement_descriptor
        case.metadata["case_fingerprint"] = case_fingerprint
        case.metadata["semantic_contract_lattice"] = semantic_contract_lattice
    if countable_findings and save_artifact and config.enable_artifact:
        artifact_started = time.perf_counter()
        bug_dir = save_bug_artifact_fn(
            case,
            raw_results=raw_results,
            normalized={k: v.to_dict() for k, v in normalized.items()},
            findings=countable_findings,
            config=resolved_config_payload,
        )
        stage_profile["logging_artifact_ms"] += (time.perf_counter() - artifact_started) * 1000
        stage_profile["total_case_wall_ms"] = sum(
            stage_profile[key]
            for key in STAGE_PROFILE_KEYS
            if key != "total_case_wall_ms"
        )
        row["duration_ms"] = stage_profile["total_case_wall_ms"]
        row["bug_dir"] = str(bug_dir)
    return row
