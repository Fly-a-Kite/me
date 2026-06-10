from __future__ import annotations

import hashlib
import json
import signal
import time
import traceback
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
from datadiff import champion_corpus as _champion_corpus
from datadiff.champion_corpus import ChampionRegistry, DEFAULT_CHAMPION_CORPUS_PATH
from datadiff.classification_oracle import annotate_findings
from datadiff.config import DEFAULT_REPLAY_BUG_SOURCE_ISSUES, ExperimentConfig
from datadiff.datagen import generate_case
from datadiff.disagreement import compute_descriptor
from datadiff.dsl import Case
from datadiff.env import collect_environment
from datadiff.execution import execute_case as _execute_case_impl
from datadiff.fingerprint import compute_fingerprint
from datadiff.feedback import FeedbackState
from datadiff.fuzz_loop import FuzzBudget, IntervalGate, RunCounters, RunPaths
from datadiff.fuzz_loop import FuzzIteration
from datadiff.guidance import GuidanceState
from datadiff.metamorphic import (
    all_metamorphic_variants,
    evaluate_metamorphic_variants,
    select_metamorphic_variants,
)
from datadiff.oracle import Finding, evaluate_case
from datadiff.oracle_complex import cross_validate_oracle_findings
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
    _empty_stage_profile,
    _finalize_stage_profile_summary,
    _guidance_summary,
    _merge_stage_profile,
    _normalized_summary,
    _quality_archive_health_summary,
    _quality_oracle_summary,
    _raw_results_summary,
    _seed_quota_health_summary,
    _stage_profile_with_total,
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
) -> dict[str, Any]:
    try:
        return run_loaded_case(
            case,
            backends=backends,
            config=config,
            save_artifact=save_artifact,
            backend_instances=backend_instances,
            environment=environment,
            target_specs=target_specs,
            config_payload=config_payload,
            metamorphic_relation_order=metamorphic_relation_order,
        )
    except TypeError as exc:
        if "metamorphic_relation_order" not in str(exc):
            raise
        return run_loaded_case(
            case,
            backends=backends,
            config=config,
            save_artifact=save_artifact,
            backend_instances=backend_instances,
            environment=environment,
            target_specs=target_specs,
            config_payload=config_payload,
    )


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
) -> dict[str, Any]:
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
        execute_case_fn=_execute_case,
        parallel_backend_execution_active_fn=_parallel_backend_execution_active,
        evaluate_case_fn=evaluate_case,
        all_metamorphic_variants_fn=all_metamorphic_variants,
        select_metamorphic_variants_fn=select_metamorphic_variants,
        evaluate_metamorphic_variants_fn=evaluate_metamorphic_variants,
        annotate_findings_fn=annotate_findings,
        candidate_recheck_fn=_candidate_recheck,
        save_bug_artifact_fn=save_bug_artifact,
        collect_environment_fn=collect_environment,
        describe_targets_fn=describe_targets,
    )


def _candidate_recheck(
    case: Case,
    backends: list[str],
    config: ExperimentConfig,
    findings: list[Finding],
) -> dict[str, Any]:
    return candidate_recheck_impl(
        case,
        backends,
        config,
        findings,
        run_loaded_case_fn=run_loaded_case,
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
) -> Path:
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
        runs_dir=RUNS_DIR,
        corpus_dir=CORPUS_DIR,
        compress_run_log=config.compress_run_log,
        save_cases=save_cases,
        case_log_file=case_log_file,
        checkpoint_interval_s=checkpoint_interval_s,
    )
    run_file = run_paths.run_file
    resolved_case_log_file = run_paths.case_log_file
    checkpoint_file = run_paths.checkpoint_file
    backend_instances = {backend_name: make_backend(backend_name) for backend_name in backends}
    environment = collect_environment()
    run_provenance = collect_run_provenance()
    targets = target_context(backends)
    target_specs = targets.target_dicts()
    config_payload = _config_payload_with_effective_guidance_targets(config)
    config_layers_payload = _config_layer_payload(config)
    guidance_targets = _configured_guidance_targets(config)
    guided = config.guidance_strategy == "guided"
    candidate_pool = max(1, config.guidance_candidate_pool if guided else 1)
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
    next_seed = seed
    stage_profile_totals = _empty_stage_profile()
    effective_generator_profile = _effective_generator_profile(config)
    generator_profile_pool, generator_profile_pool_metadata = _generator_profile_pool(
        config,
        target_capabilities=targets.common_capabilities,
    )
    version_pair_pool, version_pair_pool_metadata = _version_pair_pool(config)
    backend_pair_pool = _backend_pair_pool(backends)
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
        out = {
            "run_id": run_id,
            "status": status,
            "run_file": str(run_file),
            "case_log_file": str(resolved_case_log_file) if resolved_case_log_file is not None else "",
            "checkpoint_file": str(checkpoint_file) if checkpoint_file is not None else "",
            "requested_cases": cases,
            "executed_cases": counters.executed,
            "duration_s": duration_s,
            "elapsed_s": elapsed_s,
            "throughput_cases_s": counters.executed / elapsed_s if elapsed_s else 0.0,
            "findings": counters.findings,
            "new_behavior_cases": counters.new_behavior_cases,
            "signal_new_behavior_cases": counters.signal_new_behavior_cases,
            "saved_artifacts": counters.saved_artifacts,
            "preflight": counters.preflight_summary(),
            "replay_bug_filter": counters.replay_filter_summary(enabled=not config.enable_replay_bug),
            "family_saturation_filter": counters.family_saturation_filter_summary(
                enabled=bool(guidance is not None and config.enable_family_saturation),
            ),
            "quality_oracles": counters.quality_oracles,
            "stage_profile": _finalize_stage_profile_summary(stage_profile_totals, cases=counters.executed),
            "seed": seed,
            "next_seed": next_seed,
            "guidance": {
                "strategy": config.guidance_strategy,
                "candidate_pool": candidate_pool,
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
            },
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

    run_writer_context = JsonlWriter(run_file, compresslevel=1, buffer_lines=RUN_LOG_BUFFER_LINES)
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
        nonlocal next_seed, stage_profile_totals
        candidate_seed_start = next_seed
        generate_mutate_started = time.perf_counter()
        candidate_batch = generate_candidate_batch(
            seed_start=candidate_seed_start,
            candidate_pool=candidate_pool,
            config=config,
            feedback=feedback,
            guidance=guidance,
            generator_profile_pool=generator_profile_pool,
            generator_profile_pool_metadata=generator_profile_pool_metadata,
            generator_profile_context_features=generator_profile_context_features,
            target_capabilities=targets.common_capabilities,
            schema_spec_for_seed=lhs_schema_spec,
            generate_case_fn=generate_case,
            replay_filter_fn=_known_replay_source_filter_reason,
        )
        candidates = candidate_batch.candidates
        candidate_meta = candidate_batch.candidate_meta
        counters.replay_filtered_candidates += candidate_batch.replay_filtered_candidates
        counters.replay_fallback_candidates += candidate_batch.replay_fallback_candidates
        counters.saturated_family_filtered_candidates += candidate_batch.saturated_family_filtered_candidates
        counters.saturated_family_fallback_candidates += candidate_batch.saturated_family_fallback_candidates
        generate_mutate_elapsed_ms = (time.perf_counter() - generate_mutate_started) * 1000
        next_seed = candidate_batch.next_seed
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
        case = selected_iteration.case
        selected_meta = selected_iteration.selected_meta
        guidance_row = selected_iteration.guidance_row
        preflight_row = selected_iteration.preflight_row
        case_seed = selected_iteration.case_seed
        effective_config = selected_iteration.effective_config
        effective_config_payload = selected_iteration.effective_config_payload
        version_pair_id = selected_iteration.version_pair_id
        scheduler_feedback_elapsed_ms = selected_iteration.scheduler_elapsed_ms
        if case_writer is not None:
            case_writer.write(
                _case_log_row(
                    run_id=run_id,
                    case_index=counters.executed,
                    case_seed=case_seed,
                    candidate_seed_start=candidate_seed_start,
                    candidate_pool=candidate_pool,
                    guidance_row=guidance_row,
                    selected_meta=selected_meta,
                    backend_pair_pool=backend_pair_pool,
                    preflight_row=preflight_row,
                    generated_at=utc_now(),
                    case=case,
                )
            )
        save_artifact_for_case = _artifact_budget_available(config, counters.saved_artifacts)
        row = _invoke_run_loaded_case(
            case,
            backends=backends,
            config=effective_config,
            save_artifact=not config.enable_reducer and save_artifact_for_case,
            backend_instances=backend_instances,
            environment=environment,
            target_specs=target_specs,
            config_payload=effective_config_payload,
            metamorphic_relation_order=selected_meta.get("metamorphic_relation_order", []),
        )
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
            run_loaded_case_fn=run_loaded_case,
        )
        row = artifact_result.row
        row_stage_profile = artifact_result.row_stage_profile
        scheduler_feedback_elapsed_ms += artifact_result.scheduler_elapsed_ms
        counters.saved_artifacts += artifact_result.saved_artifact_delta

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
            candidate_pool=candidate_pool,
            case_index=counters.executed,
            elapsed_s=round(time.perf_counter() - started, 6),
        )
        for oracle in row_update.quality_oracles:
            counters.record_quality_oracle(oracle)
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
        finding_outcomes = feedback_update.finding_outcomes
        scheduler_feedback_elapsed_ms += feedback_update.elapsed_ms
        if guidance is not None:
            guidance_record_started = time.perf_counter()
            guidance.record_result(case, row, finding_outcomes=finding_outcomes)
            scheduler_feedback_elapsed_ms += (time.perf_counter() - guidance_record_started) * 1000
        row["guidance"] = guidance_row
        row_stage_profile["scheduler_feedback_ms"] += scheduler_feedback_elapsed_ms
        logging_started = time.perf_counter()
        counters.record_completed_case(row=row, preflight=preflight_row)
        row_stage_profile["logging_artifact_ms"] += (time.perf_counter() - logging_started) * 1000
        row_stage_profile["total_case_wall_ms"] = sum(
            row_stage_profile[key]
            for key in STAGE_PROFILE_KEYS
            if key != "total_case_wall_ms"
        )
        row["stage_profile"] = row_stage_profile
        row["duration_ms"] = row_stage_profile["total_case_wall_ms"]
        row["fuzz_iteration"] = FuzzIteration.from_row(row, run_id=run_id).to_dict()
        run_writer.write(_compact_log_row(row, config.log_level))
        stage_profile_totals = _merge_stage_profile(stage_profile_totals, row_stage_profile)

        now = time.perf_counter()
        if checkpoint_gate.due(now):
            write_checkpoint("running")
            checkpoint_gate.mark(now)
        if (
            progress_callback is not None
            and progress_gate.due(now)
        ):
            run_writer.flush()
            if case_writer is not None:
                case_writer.flush()
            progress_callback(snapshot("running"))
            progress_gate.mark(now)

        return {"stop": bool(budget.should_stop_after_iteration(now=time.perf_counter()))}

    while budget.should_start_iteration(executed=counters.executed, now=time.perf_counter()):
        if graceful_stop["requested"]:
            break
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
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc()[-2048:],
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
