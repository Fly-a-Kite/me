from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from typing import Any


def cmd_experiment_impl(
    args: argparse.Namespace,
    *,
    ensure_dirs_func: Callable[[], None],
    parse_presets_func: Callable[[str], list[str]],
    parse_seeds_func: Callable[[str], list[int]],
    experiment_target_runs_func: Callable[[argparse.Namespace], list[tuple[str, list[str]]]],
    parse_duration_func: Callable[[Any], float | None],
    resolve_evidence_mode_func: Callable[[str, list[str]], str],
    parse_adaptive_components_func: Callable[[Any], set[str]],
    adaptive_component_config_func: Callable[[set[str]], dict[str, bool]],
    parse_experiment_meta_func: Callable[[Any], dict[str, Any]],
    default_experiment_meta_for_runs_func: Callable[..., dict[str, Any]],
    merge_experiment_meta_func: Callable[..., dict[str, Any]],
    replay_bug_enabled_by_default_func: Callable[[str], bool],
    parse_guidance_targets_func: Callable[[str], list[str]],
    default_replay_bug_source_issues: Sequence[str],
    populate_job_learning_metadata_func: Callable[[dict[str, Any]], None],
    job_config_func: Callable[[dict[str, Any]], Any],
    experiment_job_weight_func: Callable[[dict[str, Any]], float],
    resolve_experiment_parallelism_func: Callable[..., dict[str, Any]],
    resolve_experiment_worker_batching_func: Callable[..., dict[str, Any]],
    resolve_experiment_schedule_func: Callable[..., str],
    invalid_live_adaptive_experiment_presets_func: Callable[[list[dict[str, Any]]], list[str]],
    effective_job_local_source_scheduler_func: Callable[[dict[str, Any]], tuple[bool, float]],
    target_context_func: Callable[[list[str]], Any],
    utc_now_func: Callable[[], str],
    run_experiment_adaptive_func: Callable[..., list[dict[str, Any]]],
    experiment_manifest_path_func: Callable[[], Any],
    dump_json_func: Callable[..., None],
    record_experiment_journal_func: Callable[..., tuple[Any, Any]],
    run_experiment_job_func: Callable[[dict[str, Any]], dict[str, Any]],
    run_experiment_jobs_parallel_func: Callable[..., list[dict[str, Any]]],
    process_pool_executor_cls: type,
    thread_pool_executor_cls: type,
) -> int:
    ensure_dirs_func()
    presets = parse_presets_func(args.presets)
    seeds = parse_seeds_func(args.seeds)
    target_runs = experiment_target_runs_func(args)
    suite_names = [suite for suite, _ in target_runs]
    backend_union = sorted({backend for _, backends in target_runs for backend in backends})
    duration_s = parse_duration_func(args.duration)
    evidence_mode = resolve_evidence_mode_func(getattr(args, "evidence_mode", "auto"), suite_names)
    disabled_adaptive_components = parse_adaptive_components_func(
        getattr(args, "disable_adaptive_components", "")
    )
    adaptive_components = adaptive_component_config_func(disabled_adaptive_components)
    explicit_experiment_meta = parse_experiment_meta_func(getattr(args, "experiment_meta", None))
    default_experiment_meta = default_experiment_meta_for_runs_func(
        evidence_mode=evidence_mode,
        target_runs=target_runs,
        presets=presets,
        known_bug_id=str(getattr(args, "known_bug_id", "") or ""),
        target_version=str(getattr(args, "target_version", "") or ""),
        include_pending_historical=True,
    )
    experiment_meta = merge_experiment_meta_func(default_experiment_meta, explicit_experiment_meta)
    planned_runs = [
        {
            "order": order,
            "arm_id": f"{target_suite}:{preset}:seed{seed}",
            "target_suite": target_suite,
            "backends": backends,
            "preset": preset,
            "seed": seed,
            "cases": args.cases,
            "duration_s": duration_s,
            "log_level": args.log_level,
            "compress_run_log": not args.no_compress_run_log,
            "artifact_limit": args.artifact_limit,
            "metamorphic_variant_limit": args.metamorphic_variant_limit,
            "evidence_mode": evidence_mode,
            "enable_replay_bug": bool(getattr(args, "enable_replay_bug", False))
            or replay_bug_enabled_by_default_func(evidence_mode),
            "replay_bug_source_issues": (
                parse_guidance_targets_func(getattr(args, "replay_bug_source_issues", ""))
                or list(default_replay_bug_source_issues)
            ),
            "profile_pool": str(getattr(args, "profile_pool", "") or ""),
            "profile_learning_weight": max(
                0.0,
                float(getattr(args, "profile_learning_weight", 0.0) or 0.0),
            ),
            "known_bug_id": str(getattr(args, "known_bug_id", "") or ""),
            "target_version": str(getattr(args, "target_version", "") or ""),
            "fixed_version": str(getattr(args, "fixed_version", "") or ""),
            "version_pair_pool": str(getattr(args, "version_pair_pool", "") or ""),
            "enable_metamorphic_oracle": bool(getattr(args, "enable_metamorphic_oracle", False)),
            "semantic_objective_learning_weight": max(
                0.0,
                float(getattr(args, "semantic_objective_learning_weight", 0.0) or 0.0),
            ),
            "metamorphic_relation_learning_weight": max(
                0.0,
                float(getattr(args, "metamorphic_relation_learning_weight", 0.0) or 0.0),
            ),
            "version_pair_learning_weight": max(
                0.0,
                float(getattr(args, "version_pair_learning_weight", 0.0) or 0.0),
            ),
            "backend_pair_learning_weight": max(
                0.0,
                float(getattr(args, "backend_pair_learning_weight", 0.0) or 0.0),
            ),
            "backend_pair_priority_limit": max(
                1,
                int(getattr(args, "backend_pair_priority_limit", 3) or 3),
            ),
            "metamorphic_relation_order": str(getattr(args, "metamorphic_relation_order", "") or ""),
            "enable_parallel_backend_execution": (
                bool(getattr(args, "enable_parallel_backend_execution", False))
                and not bool(
                    getattr(args, "disable_parallel_backend_execution", False)
                )
            ),
            "enable_backend_sampling": getattr(args, "enable_backend_sampling", None),
            "backend_sample_size": getattr(args, "backend_sample_size", None),
            "backend_full_sweep_interval": getattr(args, "backend_full_sweep_interval", None),
            "backend_sampling_calibration_cases": getattr(
                args,
                "backend_sampling_calibration_cases",
                None,
            ),
            "backend_sampling_candidate_burst_cases": getattr(
                args,
                "backend_sampling_candidate_burst_cases",
                None,
            ),
            "backend_sampling_candidate_burst_novel_only": getattr(
                args,
                "backend_sampling_candidate_burst_novel_only",
                None,
            ),
            "backend_sample_confirmation_recheck_count": getattr(
                args,
                "backend_sample_confirmation_recheck_count",
                None,
            ),
            "backend_sample_confirm_candidates": getattr(
                args,
                "backend_sample_confirm_candidates",
                None,
            ),
            "enable_adaptive_candidate_pool": getattr(
                args,
                "enable_adaptive_candidate_pool",
                None,
            ),
            "adaptive_candidate_pool_min_size": getattr(
                args,
                "adaptive_candidate_pool_min_size",
                None,
            ),
            "adaptive_candidate_pool_full_sweep_interval": getattr(
                args,
                "adaptive_candidate_pool_full_sweep_interval",
                None,
            ),
            "adaptive_candidate_pool_calibration_cases": getattr(
                args,
                "adaptive_candidate_pool_calibration_cases",
                None,
            ),
            "adaptive_candidate_pool_candidate_burst_cases": getattr(
                args,
                "adaptive_candidate_pool_candidate_burst_cases",
                None,
            ),
            "adaptive_candidate_pool_candidate_burst_novel_only": getattr(
                args,
                "adaptive_candidate_pool_candidate_burst_novel_only",
                None,
            ),
            "adaptive_candidate_pool_preserve_seed_stride": getattr(
                args,
                "adaptive_candidate_pool_preserve_seed_stride",
                None,
            ),
            "adaptive_candidate_pool_compensate_seed_horizon": getattr(
                args,
                "adaptive_candidate_pool_compensate_seed_horizon",
                None,
            ),
            "run_theme": str(getattr(args, "run_theme", "") or ""),
            "paper_notes": str(getattr(args, "paper_notes", "") or ""),
            "persist_closed_loop_state": bool(getattr(args, "persist_closed_loop_state", False)),
            "strategy_snapshot": str(getattr(args, "strategy_snapshot", "") or ""),
            "strategy_learning": str(getattr(args, "strategy_learning", "") or ""),
            "freeze_strategy_snapshot": bool(getattr(args, "freeze_strategy_snapshot", False)),
            "enable_local_source_scheduler": bool(getattr(args, "enable_local_source_scheduler", False)),
            "local_source_exploration_weight": max(
                0.0, float(getattr(args, "local_source_exploration_weight", 0.5))
            ),
            "disable_adaptive_components": sorted(disabled_adaptive_components),
            "adaptive_components": dict(adaptive_components),
            "experiment_meta": experiment_meta,
            "skip_run_reports": args.skip_run_reports,
        }
        for order, (target_suite, backends, preset, seed) in enumerate(
            (target_suite, backends, preset, seed)
            for target_suite, backends in target_runs
            for preset in presets
            for seed in seeds
        )
    ]
    for job in planned_runs:
        populate_job_learning_metadata_func(job)
        job["enable_replay_bug"] = bool(job.get("enable_replay_bug", False)) or job_config_func(job).enable_replay_bug
        job["estimated_cost"] = round(experiment_job_weight_func(job), 4)
    parallelism = resolve_experiment_parallelism_func(args, planned_runs)
    for job in planned_runs:
        job["worker_thread_limit"] = parallelism["worker_thread_limit"]
    jobs = int(parallelism["worker_count"])
    schedule = resolve_experiment_schedule_func(args, jobs=jobs)
    worker_batching = resolve_experiment_worker_batching_func(
        args,
        planned_runs,
        jobs=jobs,
        schedule=schedule,
    )
    invalid_live_adaptive_presets = invalid_live_adaptive_experiment_presets_func(planned_runs)
    if schedule == "adaptive" and evidence_mode == "live" and invalid_live_adaptive_presets:
        names = ",".join(invalid_live_adaptive_presets)
        print(
            "adaptive live experiments require guided presets so family-saturation and guidance stay active; "
            f"invalid presets: {names}. use live_deep_organic, live_cross_family, or another guided/live preset",
            flush=True,
        )
        return 2
    local_source_settings = [effective_job_local_source_scheduler_func(job) for job in planned_runs]
    local_source_enabled = (
        schedule == "adaptive" and adaptive_components["local_source_scheduler"]
    ) or any(enabled for enabled, _ in local_source_settings)
    local_source_weights = [weight for enabled, weight in local_source_settings if enabled]
    if schedule == "adaptive" and adaptive_components["local_source_scheduler"] and not local_source_weights:
        local_source_weights.append(max(0.0, float(getattr(args, "local_source_exploration_weight", 0.5))))
    backend_context = target_context_func(backend_union)
    manifest = {
        "created_at": utc_now_func(),
        "presets": presets,
        "seeds": seeds,
        "cases": args.cases,
        "duration_s": duration_s,
        "evidence_mode": evidence_mode,
        "known_bug_id": str(getattr(args, "known_bug_id", "") or ""),
        "target_version": str(getattr(args, "target_version", "") or ""),
        "run_theme": str(getattr(args, "run_theme", "") or ""),
        "paper_notes": str(getattr(args, "paper_notes", "") or ""),
        "experiment_meta": experiment_meta,
        "backends": backend_union,
        "target_suite": suite_names[0] if len(suite_names) == 1 else ",".join(suite_names),
        "target_suites": suite_names,
        "backends_by_suite": {suite: backends for suite, backends in target_runs},
        "targets": backend_context.target_dicts(),
        "common_capabilities": list(backend_context.common_capabilities),
        "target_context": backend_context.to_dict(),
        "log_level": args.log_level,
        "compress_run_log": not args.no_compress_run_log,
        "metamorphic_variant_limit": args.metamorphic_variant_limit,
        "replay_bug_policy": {
            "enable_replay_bug": any(bool(job.get("enable_replay_bug", False)) for job in planned_runs),
            "source_issues": sorted(
                {
                    source
                    for job in planned_runs
                    for source in job.get("replay_bug_source_issues", [])
                }
            ),
        },
        "jobs": jobs,
        "parallelism": parallelism,
        "worker_batching": worker_batching,
        "schedule": schedule,
        "local_source_scheduler": {
            "enabled": local_source_enabled,
            "exploration_weight": max(local_source_weights) if local_source_weights else 0.0,
        },
        "adaptive_methodology": {
            "components": dict(adaptive_components),
            "disabled_components": sorted(disabled_adaptive_components),
        },
        "runs": [],
    }
    if schedule == "adaptive":
        completed_runs = run_experiment_adaptive_func(args, manifest, planned_runs, duration_s, jobs=jobs)
        for result in completed_runs:
            manifest["runs"].append(result["run"])
        manifest["adaptive_state"] = completed_runs[-1]["scheduler_state"] if completed_runs else []
        manifest["adaptive_learning"] = completed_runs[-1]["adaptive_learning"] if completed_runs else {}
        manifest_path = experiment_manifest_path_func()
        dump_json_func(manifest, manifest_path)
        print(f"experiment manifest: {manifest_path}")
        if not getattr(args, "skip_paper_journal", False):
            journal_path, journal_md = record_experiment_journal_func(manifest_path, manifest)
            print(f"paper run journal: {journal_path}")
            print(f"paper run journal markdown: {journal_md}")
        return 0
    completed_runs = []
    if jobs == 1:
        for job in planned_runs:
            result = run_experiment_job_func(job)
            completed_runs.append(result)
            print(result["message"], flush=True)
    else:
        worker_count = min(jobs, len(planned_runs))
        try:
            completed_runs = run_experiment_jobs_parallel_func(
                process_pool_executor_cls,
                worker_count,
                planned_runs,
                max_parallel_cost=float(parallelism["max_parallel_cost"]),
                worker_batch_size=int(worker_batching["effective_batch_size"]),
                worker_max_rss_kib=int(worker_batching["max_rss_kib"]),
                worker_retry_limit=int(worker_batching["retry_limit"]),
                worker_batch_manifest=worker_batching,
            )
        except PermissionError:
            print("process parallelism unavailable; falling back to threaded workers", flush=True)
            worker_batching["enabled"] = False
            worker_batching["reason"] = "process_pool_unavailable_thread_fallback"
            worker_batching["effective_batch_size"] = 0
            completed_runs = run_experiment_jobs_parallel_func(
                thread_pool_executor_cls,
                worker_count,
                planned_runs,
                max_parallel_cost=float(parallelism["max_parallel_cost"]),
                worker_batch_size=0,
                worker_max_rss_kib=0,
                worker_retry_limit=0,
                worker_batch_manifest=worker_batching,
            )
    for result in sorted(completed_runs, key=lambda item: item["order"]):
        manifest["runs"].append(result["run"])
    manifest_path = experiment_manifest_path_func()
    dump_json_func(manifest, manifest_path)
    print(f"experiment manifest: {manifest_path}")
    if not getattr(args, "skip_paper_journal", False):
        journal_path, journal_md = record_experiment_journal_func(manifest_path, manifest)
        print(f"paper run journal: {journal_path}")
        print(f"paper run journal markdown: {journal_md}")
    return 0
