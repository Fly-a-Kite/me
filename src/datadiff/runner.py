from __future__ import annotations

import hashlib
import json
import signal
import time
import traceback
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable

from datadiff.artifact import save_issue_artifact as save_bug_artifact
from datadiff.bandit_selection import (
    _adaptive_exploration_reward,
    _backend_pair_context_features,
    _backend_pair_pool,
    _backend_pair_reward,
    _bucket_feature,
    _case_learning_context_features,
    _config_for_version_pair,
    _config_payload_for_version_pair,
    _effective_generator_profile,
    _generator_profile_context_features,
    _generator_profile_pool,
    _generator_profile_reward,
    _metamorphic_relation_order_from_selection,
    _normalize_backend_pair_id,
    _pair_disagrees_from_descriptor,
    _parse_version_pair,
    _rank_backend_pairs,
    _record_backend_pair_feedback,
    _runtime_cost_signal,
    _select_adaptive_action,
    _select_generator_profile,
    _semantic_objective_pool,
    _unique_nonempty_strings,
    _version_pair_context_features,
    _version_pair_id,
    _version_pair_pool,
)
from datadiff.backends import make_backend
from datadiff.backends.base import Backend
from datadiff.backend_sampling import (
    CoverageAwareBackendSampler,
    attach_backend_sampling_result,
    record_backend_sampling_outcome,
)
from datadiff.candidate_pool_sampling import (
    AdaptiveCandidatePoolController,
    record_candidate_pool_outcome,
)
from datadiff.campaign_control import (
    coordinate_candidates,
    record_coordinator_outcome,
)
from datadiff.coordinator import Coordinator, CoordinatorPolicy
from datadiff.candidate_burst import row_has_candidate_signal
from datadiff import champion_corpus as _champion_corpus
from datadiff.champion_corpus import ChampionRegistry, DEFAULT_CHAMPION_CORPUS_PATH
from datadiff.classification_oracle import annotate_findings
from datadiff.config import DEFAULT_REPLAY_BUG_SOURCE_ISSUES, ExperimentConfig
from datadiff.datagen import generate_case
from datadiff.disagreement import compute_descriptor
from datadiff.dsl import Case
from datadiff.env import collect_environment
from datadiff.evidence import causal_signature_from_row
from datadiff.experiment_manifest import build_run_experiment_manifest
from datadiff.execution import BackendExecutionSession
from datadiff.execution import execute_case as _execute_case_impl
from datadiff.execution_cost_model import load_backend_cost_model
from datadiff.family_witness_registry import family_witness_execution_backends
from datadiff.finding_outcomes import (
    is_known_saturated_candidate_issue_finding,
)
from datadiff.fingerprint import compute_fingerprint
from datadiff.feedback import FeedbackState
from datadiff.fuzz_loop import FuzzBudget, IntervalGate, RunCounters, RunPaths
from datadiff.fuzz_loop import FuzzIteration
from datadiff.guidance import GuidanceState
from datadiff.lattice_runtime import LatticeExecutionController
from datadiff.metamorphic import (
    all_metamorphic_variants,
    evaluate_metamorphic_variants,
    select_metamorphic_variants,
)
from datadiff.oracle import Finding, evaluate_case
from datadiff.oracle_complex import cross_validate_oracle_findings
from datadiff.osc_diagnostic_facade import consume_opaque_diagnostic_ref_set
from datadiff.plan_fingerprint import fingerprint_plan
from datadiff.ir_runtime import resolve_case_program
from datadiff.run_config import (
    _config_layer_payload,
    _config_payload_with_effective_guidance_targets,
    _configured_guidance_targets,
)
from datadiff.run_artifacts import process_reducer_and_artifacts
from datadiff.run_candidates import (
    REPLAY_FILTER_EXTRA_ATTEMPTS_PER_CANDIDATE,
    SATURATED_FAMILY_FILTER_EXTRA_ATTEMPTS_PER_CANDIDATE,
    _generate_case_with_optional_schema,
    _known_replay_source_filter_reason,
    _replay_bug_filter_reason,
    generate_candidate_batch,
)
from datadiff.goal_first import semantic_witness_epoch_key
from datadiff.run_findings import (
    _artifact_budget_available,
    _countable_finding_objects,
    _countable_row_findings,
    _finding_recheck_key,
    _format_recheck_key,
    _is_countable_finding_dict,
    _mark_finding_non_reproducible,
)
from datadiff.run_logging import (
    CASE_LOG_BUFFER_LINES,
    RUN_LOG_BUFFER_LINES,
    STAGE_PROFILE_KEYS,
    _adaptive_learning_health_summary,
    _case_log_row,
    _case_summary,
    _champion_corpus_health_summary,
    _closed_loop_state_summary,
    _compact_log_row,
    _empty_process_cpu_profile,
    _empty_stage_profile,
    _empty_wall_time_profile,
    _finalize_process_cpu_profile_summary,
    _finalize_stage_profile_summary,
    _finalize_wall_time_profile_summary,
    _guidance_summary,
    _merge_process_cpu_profile,
    _merge_stage_profile,
    _merge_wall_time_profile,
    _normalized_summary,
    _quality_archive_health_summary,
    _quality_oracle_summary,
    _raw_results_summary,
    _seed_quota_health_summary,
    _process_cpu_profile_with_total,
    _stage_profile_with_total,
    _wall_time_profile_with_total,
)
from datadiff.run_loaded import (
    execute_case_for_run_loaded,
    parallel_backend_execution_active,
    run_loaded_case_impl,
)
from datadiff.run_metadata import (
    _attach_case_fingerprint_to_row_case,
    _attach_disagreement_descriptor_to_row_case,
    _attach_metadata_to_row_case,
    _candidate_quality_context,
    _candidate_target_keys_from_metadata,
    _capability_target_keys,
    _feedback_target_keys,
    _fingerprint_anchor_result,
    _generated_candidate_metadata,
    _selected_candidate_metadata,
)
from datadiff.run_feedback import (
    _feedback_storage_decision,
    _source_scheduler_snapshot,
    apply_feedback_updates,
)
from datadiff.run_row import apply_iteration_row_updates
from datadiff.run_recheck import candidate_recheck_impl
from datadiff.run_selection import select_iteration_case
from datadiff.run_signatures import (
    behavior_signature,
    discovery_signature,
    row_count_bucket as _row_count_bucket,
    signal_signature,
)
from datadiff.run_provenance import collect_run_provenance
from datadiff.run_event_evidence import (
    append_case_events,
    create_event_journal,
    event_evidence_dir,
)
from datadiff.run_state import (
    _build_closed_loop_state,
    _inject_champion_corpus,
    _restore_closed_loop_state,
)
from datadiff.synthesis.lhs_sampler import schema_spec_for_seed
from datadiff.targets import describe_targets, target_context
from datadiff.util import (
    CORPUS_DIR,
    RUNS_DIR,
    JsonlWriter,
    append_jsonl,
    closed_loop_state_path,
    dump_json,
    ensure_dirs,
    run_meta_path,
    utc_now,
)

ProgressCallback = Callable[[dict[str, Any]], None]
_RUNNER_IMPORT_DEFAULT_CHAMPION_CORPUS_PATH = Path(DEFAULT_CHAMPION_CORPUS_PATH)


def _default_champion_corpus_path() -> Path:
    configured_path = Path(DEFAULT_CHAMPION_CORPUS_PATH)
    if configured_path != _RUNNER_IMPORT_DEFAULT_CHAMPION_CORPUS_PATH:
        return configured_path
    module_configured_path = Path(_champion_corpus.DEFAULT_CHAMPION_CORPUS_PATH)
    if module_configured_path != _RUNNER_IMPORT_DEFAULT_CHAMPION_CORPUS_PATH:
        return module_configured_path
    return _champion_corpus.default_champion_corpus_path()


def _parallel_backend_execution_active(config: ExperimentConfig, backends: list[str]) -> bool:
    return parallel_backend_execution_active(config, backends)


def _diagnostic_log_manifest(row: Mapping[str, Any]) -> dict[str, Any]:
    try:
        manifest = row.get("experiment_manifest")
        return dict(manifest) if isinstance(manifest, Mapping) else {}
    except Exception:
        return {}


def _diagnostic_log_backend_status(row: Mapping[str, Any]) -> dict[str, str]:
    for key in ("normalized", "raw_results", "backend_status"):
        try:
            payload = row.get(key)
            items = tuple(payload.items()) if isinstance(payload, Mapping) else ()
        except Exception:
            continue
        status_by_backend: dict[str, str] = {}
        for backend, value in items:
            if not isinstance(backend, str) or not backend:
                continue
            try:
                status = value.get("status") if isinstance(value, Mapping) else value
            except Exception:
                status = None
            status_by_backend[backend] = (
                status if isinstance(status, str) and status else "unknown"
            )
        if status_by_backend:
            return status_by_backend
    return {}


def _diagnostic_log_backends(row: Mapping[str, Any]) -> list[str]:
    return list(_diagnostic_log_backend_status(row))


def _diagnostic_log_case_digest(row: Mapping[str, Any]) -> str:
    manifest = _diagnostic_log_manifest(row)
    case_digest = manifest.get("case_digest")
    return case_digest if isinstance(case_digest, str) else ""


def _diagnostic_log_projection(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "experiment_manifest": _diagnostic_log_manifest(row),
        "backend_status": _diagnostic_log_backend_status(row),
        "osc_diagnostic_refs": consume_opaque_diagnostic_ref_set(
            row.get("osc_diagnostic_refs"),
            expected_backends=_diagnostic_log_backends(row),
            case_digest=_diagnostic_log_case_digest(row),
        ),
    }


def _metamorphic_diagnostic_log_projections(
    row: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    try:
        variants = row.get("metamorphic")
        items = tuple(variants.items()) if isinstance(variants, Mapping) else ()
    except Exception:
        return {}
    projections: dict[str, dict[str, Any]] = {}
    for name, variant in items:
        if not isinstance(name, str) or not name or not isinstance(variant, Mapping):
            continue
        projections[name] = _diagnostic_log_projection(variant)
    return projections


def _attach_opaque_diagnostic_refs_for_log(
    source_row: Mapping[str, Any],
    log_row: dict[str, Any],
) -> None:
    """Copy diagnostics to logs only after fail-closed context validation."""

    base_projection = _diagnostic_log_projection(source_row)
    variant_projections = _metamorphic_diagnostic_log_projections(source_row)
    log_row["experiment_manifest"] = base_projection["experiment_manifest"]
    log_row["osc_diagnostic_refs"] = base_projection["osc_diagnostic_refs"]
    log_row["osc_metamorphic_diagnostic_refs"] = variant_projections

    nested = log_row.get("metamorphic")
    if isinstance(nested, Mapping):
        validated_nested: dict[str, Any] = {}
        try:
            nested_items = tuple(nested.items())
        except Exception:
            nested_items = ()
        for name, variant in nested_items:
            if not isinstance(name, str) or not isinstance(variant, Mapping):
                continue
            try:
                validated_variant = dict(variant)
            except Exception:
                validated_variant = {}
            projection = variant_projections.get(name)
            if projection is not None:
                validated_variant.update(projection)
            validated_nested[name] = validated_variant
        log_row["metamorphic"] = validated_nested


def _semantic_plan_fingerprint(case: Case, config: ExperimentConfig) -> dict[str, Any]:
    resolved = resolve_case_program(case, config)
    return fingerprint_plan(
        json.dumps(resolved.semantic_plan_payload(), ensure_ascii=False, sort_keys=True),
        backend="semantic_ir",
        plan_kind=resolved.ir_mode,
    ).to_dict()


def _legacy_evidence_refs(row: dict[str, Any]) -> dict[str, Any]:
    """Keep coordinator/root accounting available in the legacy-log control.

    The logging ablation disables the append-only event journal by design, but
    disabling evidence storage must not also change scheduling feedback or
    root grouping.  These in-row references are intentionally non-replayable;
    that missing recovery property is one of the measured treatment effects.
    """

    case = row.get("case", {}) if isinstance(row.get("case"), dict) else {}
    digest = hashlib.sha256(
        json.dumps(case, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    signatures = sorted(
        {
            causal_signature_from_row(row, finding, backend=str(backend)).group_key
            for finding in row.get("findings", ()) or ()
            if isinstance(finding, dict)
            for backend in finding.get("suspicious_backends", ()) or ()
            if str(backend)
        }
    )
    return {
        "case_manifest_digest": f"legacy-case-{digest}",
        "observation_ids": [],
        "verdict_ids": [],
        "causal_signatures": signatures,
        "replayable": False,
    }


def _execute_case(
    case: Case,
    backends: list[str],
    config: ExperimentConfig,
    backend_instances: dict[str, Backend] | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    return _execute_case_impl(
        case,
        backends,
        config,
        backend_instances=backend_instances,
        parallel=config.enable_parallel_backend_execution,
    )


def _invoke_run_loaded_case(
    case: Case,
    *,
    backends: list[str],
    config: ExperimentConfig,
    save_artifact: bool,
    backend_instances: dict[str, Backend] | None,
    environment: dict[str, str] | None,
    target_specs: list[dict[str, Any]],
    config_payload: dict[str, Any],
    metamorphic_relation_order: list[str] | tuple[str, ...],
    execution_session: BackendExecutionSession | None = None,
) -> dict[str, Any]:
    kwargs = {
        "backends": backends,
        "config": config,
        "save_artifact": save_artifact,
        "backend_instances": backend_instances,
        "environment": environment,
        "target_specs": target_specs,
        "config_payload": config_payload,
        "metamorphic_relation_order": metamorphic_relation_order,
    }
    if execution_session is not None:
        kwargs["execution_session"] = execution_session
    while True:
        try:
            return run_loaded_case(case, **kwargs)
        except TypeError as exc:
            message = str(exc)
            removable = next(
                (
                    key
                    for key in ("execution_session", "metamorphic_relation_order")
                    if key in kwargs and key in message
                ),
                "",
            )
            if not removable:
                raise
            kwargs.pop(removable)


def _sample_confirmation_config(
    config: ExperimentConfig,
    config_payload: dict[str, Any],
) -> tuple[ExperimentConfig, dict[str, Any]]:
    override = config.backend_sample_confirmation_recheck_count
    if override is None:
        return config, config_payload
    recheck_count = min(
        max(0, int(config.candidate_recheck_count)),
        max(0, int(override)),
    )
    if recheck_count == int(config.candidate_recheck_count):
        return config, config_payload
    payload = config.to_dict()
    payload["candidate_recheck_count"] = recheck_count
    confirmation_config = ExperimentConfig.from_payload(payload)
    confirmation_payload = dict(config_payload)
    confirmation_payload["candidate_recheck_count"] = recheck_count
    return confirmation_config, confirmation_payload


def run_loaded_case(
    case: Case,
    backends: list[str],
    config: ExperimentConfig | None = None,
    save_artifact: bool = True,
    backend_instances: dict[str, Backend] | None = None,
    environment: dict[str, str] | None = None,
    target_specs: list[dict[str, Any]] | None = None,
    config_payload: dict[str, Any] | None = None,
    metamorphic_relation_order: list[str] | tuple[str, ...] | None = None,
    execution_session: BackendExecutionSession | None = None,
) -> dict[str, Any]:
    execute_case_fn = execution_session.execute_case if execution_session is not None else _execute_case
    execute_cases_fn = execution_session.execute_cases if execution_session is not None else None
    return run_loaded_case_impl(
        case,
        backends,
        config=config,
        save_artifact=save_artifact,
        backend_instances=backend_instances,
        environment=environment,
        target_specs=target_specs,
        config_payload=config_payload,
        metamorphic_relation_order=metamorphic_relation_order,
        execute_case_fn=execute_case_fn,
        execute_cases_fn=execute_cases_fn,
        parallel_backend_execution_active_fn=_parallel_backend_execution_active,
        evaluate_case_fn=evaluate_case,
        all_metamorphic_variants_fn=all_metamorphic_variants,
        select_metamorphic_variants_fn=select_metamorphic_variants,
        evaluate_metamorphic_variants_fn=evaluate_metamorphic_variants,
        annotate_findings_fn=annotate_findings,
        candidate_recheck_fn=lambda recheck_case, recheck_backends, recheck_config, findings: _candidate_recheck(
            recheck_case,
            recheck_backends,
            recheck_config,
            findings,
            environment=environment,
            target_specs=target_specs,
        ),
        save_bug_artifact_fn=save_bug_artifact,
        collect_environment_fn=collect_environment,
        describe_targets_fn=describe_targets,
    )


def _candidate_recheck(
    case: Case,
    backends: list[str],
    config: ExperimentConfig,
    findings: list[Finding],
    *,
    environment: dict[str, str] | None = None,
    target_specs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    def rerun(recheck_case: Case, **kwargs: Any) -> dict[str, Any]:
        # Rechecks intentionally use fresh backend instances. Reusing the run
        # session can turn adapter state contamination into false stability.
        kwargs["backend_instances"] = None
        kwargs["environment"] = environment
        kwargs["target_specs"] = target_specs or []
        kwargs["execution_session"] = None
        return run_loaded_case(recheck_case, **kwargs)

    return candidate_recheck_impl(
        case,
        backends,
        config,
        findings,
        run_loaded_case_fn=rerun,
    )


def _row_has_only_known_saturated_candidate_findings(
    row: dict[str, Any],
    known_saturated_bug_families: list[str] | tuple[str, ...],
) -> bool:
    findings = [
        finding
        for finding in (row.get("findings", []) or [])
        if isinstance(finding, dict)
    ]
    return bool(findings) and all(
        is_known_saturated_candidate_issue_finding(
            finding,
            known_saturated_bug_families,
        )
        for finding in findings
    )


def run_fuzz(
    cases: int | None,
    seed: int,
    backends: list[str],
    config: ExperimentConfig | None = None,
    duration_s: float | None = None,
    save_cases: bool = False,
    case_log_file: Path | None = None,
    checkpoint_interval_s: float | None = None,
    progress_interval_s: float | None = None,
    progress_callback: ProgressCallback | None = None,
    closed_loop_state: dict[str, Any] | None = None,
    persist_closed_loop_state: bool = False,
    runs_dir: Path | None = None,
    corpus_dir: Path | None = None,
    generate_case_fn: Callable[..., Case] | None = None,
) -> Path:
    """Run a fuzz campaign, optionally placing its immutable logs in explicit dirs.

    The default keeps the historical ``RUNS_DIR``/``CORPUS_DIR`` behaviour.
    Declared experiment shards pass private directories here so independently
    retryable shards never need to mutate process-global runner paths.
    """
    ensure_dirs()
    config = config or ExperimentConfig()
    budget = FuzzBudget.start(
        cases=cases,
        duration_s=duration_s,
        now=time.perf_counter(),
    )
    cases = budget.cases
    run_id = f"run-{utc_now().replace(':', '').replace('-', '').replace('Z', '')}-{time.time_ns()}"
    run_paths = RunPaths.build(
        run_id=run_id,
        runs_dir=runs_dir or RUNS_DIR,
        corpus_dir=corpus_dir or CORPUS_DIR,
        compress_run_log=config.compress_run_log,
        save_cases=save_cases,
        case_log_file=case_log_file,
        checkpoint_interval_s=checkpoint_interval_s,
    )
    run_file = run_paths.run_file
    resolved_case_log_file = run_paths.case_log_file
    checkpoint_file = run_paths.checkpoint_file
    environment = collect_environment()
    run_provenance = collect_run_provenance()
    adapter_revision = str(
        (run_provenance.get("vcs", {}) or {}).get("git_commit", "")
        if isinstance(run_provenance, dict)
        else ""
    )
    backend_instances, execution_session, session_reuse_resolution = (
        _prepare_backend_execution_session(
            backends,
            requested=config.enable_backend_session_reuse,
            environment=environment,
            adapter_revision=adapter_revision,
        )
    )
    targets = target_context(backends)
    target_specs = targets.target_dicts()
    campaign_coordinator = Coordinator(
        targets.common_capabilities,
        policy=CoordinatorPolicy(
            fresh_min_share=config.fresh_generation_min_share,
            feedback_max_share=config.feedback_mutation_max_share,
            recheck_parallelism=config.coordinator_recheck_parallelism,
            coverage_debt_enabled=config.enable_coordinator_coverage_debt,
            semantic_novelty_enabled=config.enable_semantic_novelty_scheduler,
            semantic_novelty_weight=config.semantic_novelty_weight,
            semantic_depth_weight=config.semantic_depth_weight,
        ),
    )
    config_payload = _config_payload_with_effective_guidance_targets(config)
    config_layers_payload = _config_layer_payload(config)
    method_resolution = config.resolved_method_arm
    method_settings = method_resolution.effective_arm.policy
    method_arm_manifest = method_resolution.manifest()
    run_experiment_manifest = build_run_experiment_manifest(
        seed=seed,
        cases=cases,
        duration_s=duration_s,
        backends=backends,
        target_specs=target_specs,
        environment=environment,
        run_provenance=run_provenance,
        config_payload=config_payload,
    )
    event_journal = (
        create_event_journal(
            run_file=run_file,
            run_id=run_id,
            method_arm=method_arm_manifest,
            environment=environment,
            targets=target_specs,
            config=config_payload,
        )
        if config.enable_event_evidence
        else None
    )
    case_generator = generate_case_fn or generate_case
    guidance_targets = _configured_guidance_targets(config)
    guided = config.guidance_strategy == "guided"
    candidate_pool = max(
        1,
        config.coordinator_candidate_pool,
        config.guidance_candidate_pool if guided else 1,
    )
    champion_corpus_path = _default_champion_corpus_path()
    champion_registry = (
        ChampionRegistry(champion_corpus_path)
        if config.enable_champion_corpus
        else None
    )
    champion_version_id = _version_pair_id(config)
    seen, signal_seen, feedback, guidance = _restore_closed_loop_state(
        closed_loop_state,
        config=config,
        backends=backends,
        guidance_enabled=guided,
        feedback_enabled=config.enable_feedback,
        champion_registry=champion_registry,
        champion_version_id=champion_version_id,
        feedback_state_cls=FeedbackState,
        guidance_state_cls=GuidanceState,
    )
    started = budget.started
    checkpoint_gate = IntervalGate(checkpoint_interval_s, started)
    progress_gate = IntervalGate(progress_interval_s, started)
    counters = RunCounters(known_saturated_bug_families=tuple(config.known_saturated_bug_families))
    attempted_iterations = 0
    next_seed = seed
    stage_profile_totals = _empty_stage_profile()
    wall_time_profile_totals = _empty_wall_time_profile()
    process_cpu_profile_totals = _empty_process_cpu_profile()
    effective_generator_profile = _effective_generator_profile(config)
    generator_profile_pool, generator_profile_pool_metadata = _generator_profile_pool(
        config,
        target_capabilities=targets.common_capabilities,
    )
    version_pair_pool, version_pair_pool_metadata = _version_pair_pool(config)
    backend_pair_pool = _backend_pair_pool(backends)
    backend_sampler = CoverageAwareBackendSampler(
        backends,
        enabled=config.enable_backend_sampling,
        sample_size=config.backend_sample_size,
        full_sweep_interval=config.backend_full_sweep_interval,
        calibration_cases=config.backend_sampling_calibration_cases,
        candidate_burst_cases=config.backend_sampling_candidate_burst_cases,
        candidate_burst_novel_only=(
            config.backend_sampling_candidate_burst_novel_only
        ),
    )
    lattice_controller = (
        LatticeExecutionController(
            backends,
            selector_mode=method_settings.execution.selector,
            node_budget=float(method_settings.execution.node_budget or len(backends)),
            versions={
                backend: str(
                    environment.get(backend, environment.get(backend.split("_", 1)[0], ""))
                    or ""
                )
                for backend in backends
            },
            seed=seed,
            cost_model=(
                load_backend_cost_model(Path(__file__).resolve().parents[2])
                if method_settings.execution.selector == "shared_cost"
                else None
            ),
        )
        if method_settings.execution.mode == "lattice"
        else None
    )
    backend_selection_controller = lattice_controller or backend_sampler
    candidate_pool_controller = AdaptiveCandidatePoolController(
        candidate_pool,
        enabled=guided and config.enable_adaptive_candidate_pool,
        minimum_pool_size=config.adaptive_candidate_pool_min_size,
        full_sweep_interval=config.adaptive_candidate_pool_full_sweep_interval,
        calibration_cases=config.adaptive_candidate_pool_calibration_cases,
        candidate_burst_cases=config.adaptive_candidate_pool_candidate_burst_cases,
        candidate_burst_novel_only=(
            config.adaptive_candidate_pool_candidate_burst_novel_only
        ),
        preserve_seed_stride=config.adaptive_candidate_pool_preserve_seed_stride,
        compensate_seed_horizon=(
            config.adaptive_candidate_pool_compensate_seed_horizon
        ),
    )
    backend_sampling_priority: tuple[str, ...] = ()
    generator_profile_context_features = _generator_profile_context_features(
        config=config,
        backends=backends,
        target_specs=target_specs,
        target_capabilities=list(targets.common_capabilities),
        candidate_pool=candidate_pool,
    )
    version_pair_context_features = _version_pair_context_features(
        config=config,
        backends=backends,
        target_capabilities=list(targets.common_capabilities),
    )
    persisted_closed_loop_state_path = closed_loop_state_path(run_file) if persist_closed_loop_state else None
    champion_injected_count = _inject_champion_corpus(
        feedback,
        version_id=champion_version_id,
        limit=max(0, min(16, max(1, candidate_pool) * 4)),
    )

    def lhs_schema_spec(case_seed: int):
        if not config.enable_lhs_seeding or (case_seed - seed) >= 256:
            return None
        return schema_spec_for_seed(case_seed, sample_count=256)

    def snapshot(status: str) -> dict[str, Any]:
        elapsed_s = budget.elapsed_s(time.perf_counter())
        backend_cpu_hour_proxy = counters.backend_reported_total_ms / 3_600_000.0
        sample_survival_rate = (
            counters.sampled_candidate_confirmed / counters.sampled_candidate_screens
            if counters.sampled_candidate_screens
            else 0.0
        )
        out = {
            "run_id": run_id,
            "status": status,
            "run_file": str(run_file),
            "event_evidence_dir": (
                str(event_evidence_dir(run_file)) if event_journal is not None else ""
            ),
            "case_log_file": str(resolved_case_log_file) if resolved_case_log_file is not None else "",
            "checkpoint_file": str(checkpoint_file) if checkpoint_file is not None else "",
            "requested_cases": cases,
            "attempted_cases": attempted_iterations,
            "executed_cases": counters.executed,
            "case_iteration_failures": counters.case_iteration_failures,
            "last_case_iteration_error": counters.last_case_iteration_error,
            "duration_s": duration_s,
            "elapsed_s": elapsed_s,
            "throughput_cases_s": counters.executed / elapsed_s if elapsed_s else 0.0,
            "findings": counters.findings,
            "new_behavior_cases": counters.new_behavior_cases,
            "signal_new_behavior_cases": counters.signal_new_behavior_cases,
            "candidate_bug_cases": counters.candidate_bug_cases,
            "recheck_surviving_candidate_cases": counters.recheck_surviving_candidate_cases,
            "discovery_efficiency": {
                "signal_new_behavior_per_s": (
                    counters.signal_new_behavior_cases / elapsed_s if elapsed_s else 0.0
                ),
                "candidate_bug_cases_per_s": (
                    counters.candidate_bug_cases / elapsed_s if elapsed_s else 0.0
                ),
                "recheck_surviving_candidates_per_s": (
                    counters.recheck_surviving_candidate_cases / elapsed_s if elapsed_s else 0.0
                ),
                "backend_reported_cpu_hour_proxy": backend_cpu_hour_proxy,
                "recheck_surviving_candidates_per_backend_cpu_hour": (
                    counters.recheck_surviving_candidate_cases / backend_cpu_hour_proxy
                    if backend_cpu_hour_proxy
                    else 0.0
                ),
                "sampled_candidate_screens": counters.sampled_candidate_screens,
                "sampled_candidate_confirmed": counters.sampled_candidate_confirmed,
                "sampled_candidate_rejected": counters.sampled_candidate_rejected,
                "sample_confirmation_survival_rate": sample_survival_rate,
            },
            "saved_artifacts": counters.saved_artifacts,
            "preflight": counters.preflight_summary(),
            "replay_bug_filter": counters.replay_filter_summary(enabled=not config.enable_replay_bug),
            "family_saturation_filter": counters.family_saturation_filter_summary(
                enabled=bool(guidance is not None and config.enable_family_saturation),
            ),
            "quality_oracles": counters.quality_oracles,
            "semantic_activation": counters.semantic_activation_summary(),
            "stage_profile": _finalize_stage_profile_summary(stage_profile_totals, cases=counters.executed),
            "wall_time_profile": _finalize_wall_time_profile_summary(
                wall_time_profile_totals,
                cases=counters.executed,
            ),
            "process_cpu_profile": _finalize_process_cpu_profile_summary(
                process_cpu_profile_totals,
                cases=counters.executed,
            ),
            "seed": seed,
            "next_seed": next_seed,
            "guidance": {
                "strategy": config.guidance_strategy,
                "candidate_pool": candidate_pool,
                "adaptive_candidate_pool": candidate_pool_controller.summary(),
                "targets": config.guidance_targets,
            },
            "effective_generator_profile": effective_generator_profile,
            "generator_profile_pool": list(generator_profile_pool),
            "generator_profile_pool_metadata": dict(generator_profile_pool_metadata),
            "version_pair_pool": list(version_pair_pool),
            "version_pair_pool_metadata": dict(version_pair_pool_metadata),
            "backend_pair_pool": list(backend_pair_pool),
            "backend_pair_learning_weight": config.backend_pair_learning_weight,
            "backend_pair_priority_limit": config.backend_pair_priority_limit,
            "execution": {
                "parallel_backend_execution": _parallel_backend_execution_active(config, backends),
                "session_reuse_resolution": dict(session_reuse_resolution),
                "session": (
                    execution_session.summary()
                    if execution_session is not None
                    else {
                        "persistent": False,
                        "mode": "fresh_per_execution",
                        "execute_calls": 0,
                        "backend_calls": counters.backend_calls,
                        "batch_calls": 0,
                        "executor_variants": 0,
                        "reset_calls": 0,
                        "isolation_rejections": 0,
                        "cache": {
                            "enabled_for_lattice_arms": False,
                            "entries": 0,
                            "hits": 0,
                            "misses": 0,
                            "hit_rate": 0.0,
                        },
                    }
                ),
                "method_arm": method_arm_manifest,
                "lattice": lattice_controller.summary() if lattice_controller is not None else {
                    "enabled": False,
                    "selector_mode": "all",
                    "node_budget": None,
                },
                "backend_sampling": {
                    "enabled": backend_selection_controller.enabled,
                    "sample_size": backend_selection_controller.sample_size,
                    "full_sweep_interval": backend_selection_controller.full_sweep_interval,
                    "confirm_candidates": config.backend_sample_confirm_candidates,
                    "calibration_cases": backend_selection_controller.calibration_cases,
                    "candidate_burst_cases": backend_selection_controller.candidate_burst_cases,
                    "coverage": backend_selection_controller.coverage_summary(),
                },
            },
            "coordinator": campaign_coordinator.snapshot(
                omit_recomputable=config.log_level == "minimal",
            ),
            "generator_profile_learning_weight": config.generator_profile_learning_weight,
            "semantic_objective_learning_weight": config.semantic_objective_learning_weight,
            "metamorphic_relation_learning_weight": config.metamorphic_relation_learning_weight,
            "version_pair_learning_weight": config.version_pair_learning_weight,
            "version_pair": _version_pair_id(config),
            "target_version": config.target_version,
            "fixed_version": config.fixed_version,
            "champion_corpus": {
                "enabled": config.enable_champion_corpus,
                "donor_bandit_enabled": config.enable_champion_graft_donor_bandit,
                "path": str(champion_corpus_path),
                "version_id": champion_version_id,
                "injected_count": champion_injected_count,
            },
            "lhs_seeding": {
                "enabled": config.enable_lhs_seeding,
                "sample_count": 256,
            },
            "backends": backends,
            "targets": target_specs,
            "common_capabilities": list(targets.common_capabilities),
            "target_context": targets.to_dict(),
            "config": config_payload,
            "config_layers": config_layers_payload,
            "method_arm": method_arm_manifest,
            "experiment_manifest": run_experiment_manifest,
            "environment": environment,
            "run_provenance": run_provenance,
            "log_level": config.log_level,
            "updated_at": utc_now(),
        }
        if persist_closed_loop_state:
            closed_loop_state_payload = _build_closed_loop_state(
                seen=seen,
                signal_seen=signal_seen,
                feedback=feedback,
                guidance=guidance,
            )
            out["closed_loop_state_file"] = (
                str(persisted_closed_loop_state_path) if persisted_closed_loop_state_path is not None else ""
            )
            out["closed_loop_state_summary"] = _closed_loop_state_summary(closed_loop_state_payload)
        return out

    def write_checkpoint(status: str) -> None:
        try:
            run_writer.flush()
            if event_journal is not None:
                event_journal.flush()
            if case_writer is not None:
                case_writer.flush()
            if checkpoint_file is not None:
                if persist_closed_loop_state and persisted_closed_loop_state_path is not None:
                    dump_json(
                        _build_closed_loop_state(
                            seen=seen,
                            signal_seen=signal_seen,
                            feedback=feedback,
                            guidance=guidance,
                        ),
                        persisted_closed_loop_state_path,
                        compact=True,
                    )
                dump_json(snapshot(status), checkpoint_file, compact=True)
        except Exception:  # noqa: BLE001 — never let checkpoint IO crash a 24h run
            counters.checkpoint_write_failures += 1

    # Opaque diagnostic refs add per-case correlation digests.  Level 2 keeps
    # the complete fail-closed projection within the existing compact I/O
    # budget without dropping base or metamorphic reference identity.
    run_writer_context = JsonlWriter(
        run_file,
        compresslevel=2,
        buffer_lines=RUN_LOG_BUFFER_LINES,
    )
    run_writer = run_writer_context.__enter__()
    case_writer_context = (
        JsonlWriter(
            resolved_case_log_file,
            compresslevel=1,
            buffer_lines=CASE_LOG_BUFFER_LINES,
        )
        if resolved_case_log_file
        else None
    )
    case_writer = case_writer_context.__enter__() if case_writer_context is not None else None
    include_online_weight_snapshot = config.log_level == "full" or case_writer is not None

    # 24h-stability: a SIGTERM (e.g. tmux/systemd shutdown) must flush state, not lose it.
    graceful_stop = {"requested": False}

    def _on_graceful_signal(_signo: int, _frame: Any) -> None:
        graceful_stop["requested"] = True

    previous_sigterm = signal.getsignal(signal.SIGTERM)
    previous_sigusr1 = signal.getsignal(signal.SIGUSR1) if hasattr(signal, "SIGUSR1") else None
    try:
        signal.signal(signal.SIGTERM, _on_graceful_signal)
        if hasattr(signal, "SIGUSR1"):
            signal.signal(signal.SIGUSR1, _on_graceful_signal)
    except (ValueError, OSError):
        # Non-main thread / restricted environment: signals optional, do not abort.
        pass

    def _run_one_iteration() -> dict[str, Any]:
        """Single fuzz iteration body wrapped so a 24h run never dies on one bad case."""
        nonlocal next_seed, stage_profile_totals, wall_time_profile_totals
        nonlocal process_cpu_profile_totals, backend_sampling_priority
        candidate_seed_start = next_seed
        candidate_pool_selection = candidate_pool_controller.select()
        effective_candidate_pool = candidate_pool_selection.pool_size
        generate_mutate_started = time.perf_counter()
        generate_mutate_process_started = time.process_time()
        candidate_batch = generate_candidate_batch(
            seed_start=candidate_seed_start,
            candidate_pool=effective_candidate_pool,
            config=config,
            feedback=feedback,
            guidance=guidance,
            generator_profile_pool=generator_profile_pool,
            generator_profile_pool_metadata=generator_profile_pool_metadata,
            generator_profile_context_features=generator_profile_context_features,
            target_capabilities=targets.common_capabilities,
            schema_spec_for_seed=lhs_schema_spec,
            generate_case_fn=case_generator,
            replay_filter_fn=_known_replay_source_filter_reason,
            force_fresh_source=campaign_coordinator.requires_fresh_source(),
            generation_epoch_seed=semantic_witness_epoch_key(candidate_seed_start),
        )
        candidates = candidate_batch.candidates
        candidate_meta = candidate_batch.candidate_meta
        counters.replay_filtered_candidates += candidate_batch.replay_filtered_candidates
        counters.replay_fallback_candidates += candidate_batch.replay_fallback_candidates
        counters.saturated_family_filtered_candidates += candidate_batch.saturated_family_filtered_candidates
        counters.saturated_family_fallback_candidates += candidate_batch.saturated_family_fallback_candidates
        generate_mutate_elapsed_ms = (time.perf_counter() - generate_mutate_started) * 1000
        generate_mutate_process_cpu_ms = max(
            0.0,
            (time.process_time() - generate_mutate_process_started) * 1000,
        )
        generated_next_seed = int(candidate_batch.next_seed)
        seed_advance_decision = candidate_pool_controller.seed_advance_decision(
            generated_seed_advance=generated_next_seed - candidate_seed_start,
            selected_pool_size=effective_candidate_pool,
        )
        next_seed = candidate_seed_start + int(
            seed_advance_decision["applied_seed_advance"]
        )
        candidate_pool_controller.record_seed_advance(
            generated_seed_advance=int(
                seed_advance_decision["generated_seed_advance"]
            ),
            applied_seed_advance=int(seed_advance_decision["applied_seed_advance"]),
            seed_horizon_compensation_extra_slots=int(
                seed_advance_decision[
                    "seed_horizon_compensation_extra_slots"
                ]
            ),
        )
        candidate_pool_sampling = candidate_pool_selection.to_dict()
        candidate_pool_sampling.update(
            {
                "generated_next_seed": generated_next_seed,
                "applied_next_seed": next_seed,
                **seed_advance_decision,
            }
        )
        coordinated = coordinate_candidates(
            campaign_coordinator,
            candidates,
            candidate_meta,
            target_shard=",".join(backends),
        )
        candidates = list(coordinated.candidates)
        selection_wall_started = time.perf_counter()
        selection_process_started = time.process_time()
        selected_iteration = select_iteration_case(
            candidates=candidates,
            candidate_meta=candidate_meta,
            config=config,
            config_payload=config_payload,
            feedback=feedback,
            guidance=guidance,
            include_online_weight_snapshot=include_online_weight_snapshot,
            version_pair_pool=version_pair_pool,
            version_pair_pool_metadata=version_pair_pool_metadata,
            version_pair_context_features=version_pair_context_features,
            target_capabilities=targets.common_capabilities,
            backends=backends,
        )
        scheduler_profile_wall_ms = max(
            0.0,
            (time.perf_counter() - selection_wall_started) * 1000,
        )
        scheduler_profile_process_cpu_ms = max(
            0.0,
            (time.process_time() - selection_process_started) * 1000,
        )
        case = selected_iteration.case
        selected_meta = selected_iteration.selected_meta
        scheduled_case = coordinated.scheduled_by_case_id[id(case)]
        campaign_coordinator.admit(scheduled_case)
        selected_meta["coordinator"] = scheduled_case.to_dict()
        guidance_row = selected_iteration.guidance_row
        guidance_row["candidate_pool_sampling"] = candidate_pool_sampling
        preflight_row = selected_iteration.preflight_row
        case_seed = selected_iteration.case_seed
        effective_config = selected_iteration.effective_config
        effective_config_payload = selected_iteration.effective_config_payload
        version_pair_id = selected_iteration.version_pair_id
        scheduler_feedback_elapsed_ms = selected_iteration.scheduler_elapsed_ms
        logging_profile_wall_ms = 0.0
        logging_profile_process_cpu_ms = 0.0
        if case_writer is not None:
            case_log_started = time.perf_counter()
            case_log_process_started = time.process_time()
            case_writer.write(
                _case_log_row(
                    run_id=run_id,
                    case_index=counters.executed,
                    case_seed=case_seed,
                    candidate_seed_start=candidate_seed_start,
                    candidate_pool=effective_candidate_pool,
                    candidate_pool_sampling=candidate_pool_sampling,
                    guidance_row=guidance_row,
                    selected_meta=selected_meta,
                    backend_pair_pool=backend_pair_pool,
                    preflight_row=preflight_row,
                    generated_at=utc_now(),
                    case=case,
                )
            )
            logging_profile_wall_ms += max(
                0.0,
                (time.perf_counter() - case_log_started) * 1000,
            )
            logging_profile_process_cpu_ms += max(
                0.0,
                (time.process_time() - case_log_process_started) * 1000,
            )
        save_artifact_for_case = _artifact_budget_available(config, counters.saved_artifacts)
        backend_selection_started = time.perf_counter()
        backend_selection_process_started = time.process_time()
        registered_family_backends = family_witness_execution_backends(case)
        backend_selection = (
            lattice_controller.select(
                priority_pairs=backend_sampling_priority,
                semantic_plan_fingerprint=(
                    _semantic_plan_fingerprint(case, config)
                    if method_settings.execution.plan_guidance == "semantic"
                    else None
                ),
                required_backends=registered_family_backends,
            )
            if lattice_controller is not None
            else backend_sampler.select(
                priority_pairs=backend_sampling_priority,
                required_backends=registered_family_backends,
            )
        )
        scheduler_profile_wall_ms += max(
            0.0,
            (time.perf_counter() - backend_selection_started) * 1000,
        )
        scheduler_profile_process_cpu_ms += max(
            0.0,
            (time.process_time() - backend_selection_process_started) * 1000,
        )
        active_backends = list(backend_selection.active_backends)
        target_specs_by_name = {
            str(spec.get("name", "")): spec
            for spec in target_specs
            if isinstance(spec, dict)
        }
        active_target_specs = [
            target_specs_by_name[name]
            for name in active_backends
            if name in target_specs_by_name
        ]
        screening_config = effective_config
        screening_config_payload = effective_config_payload
        confirm_sampled_candidate = bool(
            backend_selection.sampled
            and (
                method_settings.execution.confirmation_mode == "full"
                if lattice_controller is not None
                else config.backend_sample_confirm_candidates
            )
        )
        if confirm_sampled_candidate and effective_config.candidate_recheck_count > 0:
            screening_config_data = effective_config.to_dict()
            screening_config_data["candidate_recheck_count"] = 0
            screening_config_data["enable_artifact"] = False
            screening_config = ExperimentConfig.from_payload(screening_config_data)
            screening_config_payload = dict(effective_config_payload)
            screening_config_payload["candidate_recheck_count"] = 0
            screening_config_payload["enable_artifact"] = False
        row = _invoke_run_loaded_case(
            case,
            backends=active_backends,
            config=screening_config,
            save_artifact=(
                not confirm_sampled_candidate
                and not config.enable_reducer
                and save_artifact_for_case
            ),
            backend_instances=backend_instances,
            environment=environment,
            target_specs=active_target_specs,
            config_payload=screening_config_payload,
            metamorphic_relation_order=selected_meta.get("metamorphic_relation_order", []),
            execution_session=execution_session,
        )
        screening_row = None
        confirmation_skip_reason = ""
        if confirm_sampled_candidate and row_has_candidate_signal(row):
            if (
                registered_family_backends
                and _row_has_only_known_saturated_candidate_findings(
                    row,
                    config.known_saturated_bug_families,
                )
            ):
                confirmation_skip_reason = (
                    "registered_family_known_saturated_finding"
                )
            else:
                screening_row = row
                confirmation_config, confirmation_config_payload = (
                    _sample_confirmation_config(
                        effective_config,
                        effective_config_payload,
                    )
                )
                row = _invoke_run_loaded_case(
                    case,
                    backends=backends,
                    config=confirmation_config,
                    save_artifact=not config.enable_reducer and save_artifact_for_case,
                    backend_instances=backend_instances,
                    environment=environment,
                    target_specs=target_specs,
                    config_payload=confirmation_config_payload,
                    metamorphic_relation_order=selected_meta.get("metamorphic_relation_order", []),
                    execution_session=execution_session,
                )
        row = attach_backend_sampling_result(
            row,
            selection=backend_selection,
            configured_backends=backends,
            screening_row=screening_row,
        )
        if confirmation_skip_reason:
            row.setdefault("backend_sampling", {})[
                "confirmation_skip_reason"
            ] = confirmation_skip_reason
        if lattice_controller is not None:
            row["execution_lattice_selection"] = backend_selection.to_dict()
        row["candidate_pool_sampling"] = candidate_pool_sampling
        artifact_result = process_reducer_and_artifacts(
            row=row,
            case=case,
            backends=backends,
            config=config,
            effective_config=effective_config,
            backend_instances=backend_instances,
            environment=environment,
            target_specs=target_specs,
            effective_config_payload=effective_config_payload,
            saved_artifacts_count=counters.saved_artifacts,
            generate_mutate_elapsed_ms=generate_mutate_elapsed_ms,
            generate_mutate_process_cpu_ms=generate_mutate_process_cpu_ms,
            run_loaded_case_fn=lambda *args, **kwargs: run_loaded_case(
                *args,
                **kwargs,
                execution_session=execution_session,
            ),
        )
        row = artifact_result.row
        row["candidate_pool_sampling"] = candidate_pool_sampling
        row_stage_profile = artifact_result.row_stage_profile
        row_wall_time_profile = artifact_result.row_wall_time_profile
        row_process_cpu_profile = artifact_result.row_process_cpu_profile
        scheduler_profile_wall_ms += artifact_result.scheduler_elapsed_ms
        scheduler_profile_process_cpu_ms += (
            artifact_result.scheduler_process_cpu_ms
        )
        burst_started = time.perf_counter()
        burst_process_started = time.process_time()
        backend_burst_decision = record_backend_sampling_outcome(
            backend_selection_controller,
            row=row,
            screening_row=screening_row,
        )
        row.setdefault("backend_sampling", {})[
            "candidate_burst_decision"
        ] = backend_burst_decision
        candidate_pool_burst_decision = record_candidate_pool_outcome(
            candidate_pool_controller,
            row=row,
            screening_row=screening_row,
        )
        candidate_pool_sampling[
            "candidate_burst_decision"
        ] = candidate_pool_burst_decision
        row["candidate_pool_sampling"] = candidate_pool_sampling
        scheduler_profile_wall_ms += max(
            0.0,
            (time.perf_counter() - burst_started) * 1000,
        )
        scheduler_profile_process_cpu_ms += max(
            0.0,
            (time.process_time() - burst_process_started) * 1000,
        )
        scheduler_feedback_elapsed_ms += artifact_result.scheduler_elapsed_ms
        counters.saved_artifacts += artifact_result.saved_artifact_delta

        row_update_started = time.perf_counter()
        row_update_process_started = time.process_time()
        row_update = apply_iteration_row_updates(
            row=row,
            case=case,
            selected_meta=selected_meta,
            preflight_row=preflight_row,
            feedback=feedback,
            config=config,
            guidance_row=guidance_row,
            guidance_targets=guidance_targets,
            backend_pair_pool=backend_pair_pool,
            version_pair_id=version_pair_id,
            target_capabilities=targets.common_capabilities,
            seen=seen,
            signal_seen=signal_seen,
            candidate_seed_start=candidate_seed_start,
            candidate_pool=effective_candidate_pool,
            case_index=counters.executed,
            elapsed_s=round(time.perf_counter() - started, 6),
        )
        scheduler_profile_wall_ms += max(
            0.0,
            (time.perf_counter() - row_update_started) * 1000,
        )
        scheduler_profile_process_cpu_ms += max(
            0.0,
            (time.process_time() - row_update_process_started) * 1000,
        )
        backend_sampling_priority = tuple(row.get("backend_pair_priority", []) or ())
        for oracle in row_update.quality_oracles:
            counters.record_quality_oracle(oracle)
        feedback_started = time.perf_counter()
        feedback_process_started = time.process_time()
        feedback_update = apply_feedback_updates(
            row=row,
            case=case,
            feedback=feedback,
            config=config,
            selected_meta=selected_meta,
            guidance_row=guidance_row,
            preflight_row=preflight_row,
            behavior_signature=row_update.behavior_signature,
            signal_signature=row_update.signal_signature,
            discovery_signature=row_update.discovery_signature,
            version_id=version_pair_id,
            generator_profile_context_features=generator_profile_context_features,
            target_capabilities=targets.common_capabilities,
        )
        scheduler_profile_wall_ms += max(
            0.0,
            (time.perf_counter() - feedback_started) * 1000,
        )
        scheduler_profile_process_cpu_ms += max(
            0.0,
            (time.process_time() - feedback_process_started) * 1000,
        )
        finding_outcomes = feedback_update.finding_outcomes
        scheduler_feedback_elapsed_ms += feedback_update.elapsed_ms
        if guidance is not None:
            guidance_record_started = time.perf_counter()
            guidance_record_process_started = time.process_time()
            guidance.record_result(case, row, finding_outcomes=finding_outcomes)
            scheduler_feedback_elapsed_ms += (time.perf_counter() - guidance_record_started) * 1000
            scheduler_profile_wall_ms += max(
                0.0,
                (time.perf_counter() - guidance_record_started) * 1000,
            )
            scheduler_profile_process_cpu_ms += max(
                0.0,
                (time.process_time() - guidance_record_process_started) * 1000,
            )
        row["guidance"] = guidance_row
        row_stage_profile["scheduler_feedback_ms"] += scheduler_feedback_elapsed_ms
        logging_started = time.perf_counter()
        logging_process_started = time.process_time()
        counters.record_completed_case(row=row, preflight=preflight_row)
        counter_logging_wall_ms = max(
            0.0,
            (time.perf_counter() - logging_started) * 1000,
        )
        counter_logging_process_cpu_ms = max(
            0.0,
            (time.process_time() - logging_process_started) * 1000,
        )
        row_stage_profile["logging_artifact_ms"] += counter_logging_wall_ms
        logging_profile_wall_ms += counter_logging_wall_ms
        logging_profile_process_cpu_ms += counter_logging_process_cpu_ms
        row_stage_profile["total_case_wall_ms"] = sum(
            row_stage_profile[key]
            for key in STAGE_PROFILE_KEYS
            if key != "total_case_wall_ms"
        )
        row_wall_time_profile["scheduler_feedback_ms"] += scheduler_profile_wall_ms
        row_process_cpu_profile["scheduler_feedback_ms"] += (
            scheduler_profile_process_cpu_ms
        )
        row_wall_time_profile["logging_artifact_ms"] += logging_profile_wall_ms
        row_process_cpu_profile["logging_artifact_ms"] += (
            logging_profile_process_cpu_ms
        )
        row_wall_time_profile = _wall_time_profile_with_total(
            row_wall_time_profile
        )
        row_process_cpu_profile = _process_cpu_profile_with_total(
            row_process_cpu_profile
        )
        row["stage_profile"] = row_stage_profile
        row["wall_time_profile"] = row_wall_time_profile
        row["process_cpu_profile"] = row_process_cpu_profile
        row["duration_ms"] = row_stage_profile["total_case_wall_ms"]
        event_evidence_started = time.perf_counter()
        event_evidence_process_started = time.process_time()
        event_refs = (
            append_case_events(
                event_journal,
                row=row,
                root_seed=seed,
                case_index=counters.executed,
                selected_meta=selected_meta,
                capability_units=targets.common_capabilities,
            )
            if event_journal is not None
            else _legacy_evidence_refs(row)
        )
        # The run log stays inexpensive to scan; authoritative raw execution
        # records live in the append-only event journal and are reached through
        # these stable identifiers.
        row["evidence_refs"] = event_refs
        record_coordinator_outcome(
            campaign_coordinator,
            scheduled_case,
            row=row,
            causal_signatures=event_refs["causal_signatures"],
            witness_id=event_refs["case_manifest_digest"],
        )
        event_evidence_wall_ms = max(
            0.0,
            (time.perf_counter() - event_evidence_started) * 1000,
        )
        event_evidence_process_cpu_ms = max(
            0.0,
            (time.process_time() - event_evidence_process_started) * 1000,
        )
        row_stage_profile["logging_artifact_ms"] += event_evidence_wall_ms
        row_wall_time_profile["logging_artifact_ms"] += event_evidence_wall_ms
        row_process_cpu_profile["logging_artifact_ms"] += event_evidence_process_cpu_ms
        row_stage_profile["total_case_wall_ms"] = sum(
            row_stage_profile[key]
            for key in STAGE_PROFILE_KEYS
            if key != "total_case_wall_ms"
        )
        row["stage_profile"] = row_stage_profile
        row["duration_ms"] = row_stage_profile["total_case_wall_ms"]
        row["fuzz_iteration"] = FuzzIteration.from_row(row, run_id=run_id).to_dict()
        compact_started = time.perf_counter()
        compact_process_started = time.process_time()
        compact_row = _compact_log_row(row, config.log_level)
        _attach_opaque_diagnostic_refs_for_log(row, compact_row)
        compact_wall_ms = max(
            0.0,
            (time.perf_counter() - compact_started) * 1000,
        )
        compact_process_cpu_ms = max(
            0.0,
            (time.process_time() - compact_process_started) * 1000,
        )
        row_wall_time_profile["logging_artifact_ms"] += compact_wall_ms
        row_process_cpu_profile["logging_artifact_ms"] += compact_process_cpu_ms
        row_wall_time_profile = _wall_time_profile_with_total(row_wall_time_profile)
        row_process_cpu_profile = _process_cpu_profile_with_total(
            row_process_cpu_profile
        )
        row["wall_time_profile"] = row_wall_time_profile
        row["process_cpu_profile"] = row_process_cpu_profile
        if compact_row is not row:
            compact_row["wall_time_profile"] = row_wall_time_profile
            compact_row["process_cpu_profile"] = row_process_cpu_profile
        run_writer.write(compact_row)
        stage_profile_totals = _merge_stage_profile(stage_profile_totals, row_stage_profile)
        wall_time_profile_totals = _merge_wall_time_profile(
            wall_time_profile_totals,
            row_wall_time_profile,
        )
        process_cpu_profile_totals = _merge_process_cpu_profile(
            process_cpu_profile_totals,
            row_process_cpu_profile,
        )

        now = time.perf_counter()
        if checkpoint_gate.due(now):
            write_checkpoint("running")
            checkpoint_gate.mark(now)
        if (
            progress_callback is not None
            and progress_gate.due(now)
        ):
            run_writer.flush()
            if event_journal is not None:
                event_journal.flush()
            if case_writer is not None:
                case_writer.flush()
            progress_callback(snapshot("running"))
            progress_gate.mark(now)

        return {"stop": bool(budget.should_stop_after_iteration(now=time.perf_counter()))}

    while budget.should_start_iteration(executed=attempted_iterations, now=time.perf_counter()):
        if graceful_stop["requested"]:
            break
        attempted_iterations += 1
        try:
            iteration_result = _run_one_iteration()
        except KeyboardInterrupt:
            graceful_stop["requested"] = True
            break
        except Exception as exc:  # noqa: BLE001 — protect 24h long run from single bad case
            counters.case_iteration_failures += 1
            counters.last_case_iteration_error = f"{type(exc).__name__}: {exc}"
            try:
                run_writer.write(
                    {
                        "kind": "case_iteration_error",
                        "case_seed": next_seed,
                        "case_index": counters.executed,
                        "attempt_index": attempted_iterations - 1,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc()[-2048:],
                        "method_arm": method_arm_manifest,
                        "experiment_manifest": run_experiment_manifest,
                        "logged_at": utc_now(),
                    }
                )
            except Exception:  # noqa: BLE001
                pass
            next_seed += 1
            now_after_failure = time.perf_counter()
            if checkpoint_gate.due(now_after_failure):
                write_checkpoint("running")
                checkpoint_gate.mark(now_after_failure)
            if budget.should_stop_after_iteration(now=now_after_failure):
                break
            continue
        if iteration_result.get("stop"):
            break

    try:
        signal.signal(signal.SIGTERM, previous_sigterm)
        if hasattr(signal, "SIGUSR1") and previous_sigusr1 is not None:
            signal.signal(signal.SIGUSR1, previous_sigusr1)
    except (ValueError, OSError):
        pass

    if case_writer_context is not None:
        case_writer_context.__exit__(None, None, None)
    run_writer_context.__exit__(None, None, None)
    if event_journal is not None:
        event_journal.close()
    if execution_session is not None:
        execution_session.close()

    elapsed_s = budget.elapsed_s(time.perf_counter())
    meta = snapshot("completed")
    meta["elapsed_s"] = elapsed_s
    meta["throughput_cases_s"] = counters.executed / elapsed_s if elapsed_s else 0.0
    if persist_closed_loop_state and persisted_closed_loop_state_path is not None:
        dump_json(
            _build_closed_loop_state(
                seen=seen,
                signal_seen=signal_seen,
                feedback=feedback,
                guidance=guidance,
            ),
            persisted_closed_loop_state_path,
            compact=True,
        )
    dump_json(meta, run_meta_path(run_file), compact=True)
    write_checkpoint("completed")
    if progress_callback is not None:
        progress_callback(meta)
    return run_file


def _prepare_backend_execution_session(
    backends: list[str],
    *,
    requested: bool,
    environment: dict[str, str],
    adapter_revision: str,
) -> tuple[
    dict[str, Backend] | None,
    BackendExecutionSession | None,
    dict[str, Any],
]:
    """Resolve safe session reuse without making fresh-only targets unrunnable."""

    if not requested:
        return None, None, {
            "requested": False,
            "effective": False,
            "mode": "fresh_per_execution",
            "reason": "disabled_by_config",
            "backend_policies": {},
            "fresh_only_backends": [],
        }

    backend_instances = {
        backend_name: make_backend(backend_name)
        for backend_name in backends
    }
    backend_policies = {
        backend_name: str(
            getattr(backend, "session_reuse_policy", "stateless") or "stateless"
        )
        for backend_name, backend in backend_instances.items()
    }
    fresh_only_backends = sorted(
        backend_name
        for backend_name, policy in backend_policies.items()
        if policy == "fresh_only"
    )
    if fresh_only_backends:
        close_errors: list[dict[str, str]] = []
        for backend_name, backend in backend_instances.items():
            try:
                backend.close()
            except Exception as exc:  # noqa: BLE001
                close_errors.append(
                    {
                        "backend": backend_name,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
        return None, None, {
            "requested": True,
            "effective": False,
            "mode": "fresh_per_execution",
            "reason": "fresh_only_backend_present",
            "backend_policies": dict(sorted(backend_policies.items())),
            "fresh_only_backends": fresh_only_backends,
            "probe_close_errors": close_errors,
        }

    execution_session = BackendExecutionSession(
        backends,
        backend_instances=backend_instances,
        environment=environment,
        adapter_revision=adapter_revision,
    )
    return backend_instances, execution_session, {
        "requested": True,
        "effective": True,
        "mode": "reuse",
        "reason": "all_backends_reusable",
        "backend_policies": dict(sorted(backend_policies.items())),
        "fresh_only_backends": [],
    }
