from __future__ import annotations

import argparse
import os
import time
from collections.abc import Mapping
from concurrent.futures import FIRST_COMPLETED, as_completed, wait
from pathlib import Path
from typing import Any, Callable

from datadiff.adaptive_learning import AdaptiveLearningState, ContinualPriorityMemory
from datadiff.config import ExperimentConfig
from datadiff.experiment_catalog import (
    registered_experiment_meta_defaults,
    registered_experiment_matrix_for_run,
    replay_bug_enabled_by_default,
)
from datadiff.experiment_metadata import normalize_experiment_meta, resolved_run_semantics
from datadiff.experiment_worker_batching import (
    resolve_worker_batching,
    run_batched_parallel,
    run_worker_batch,
)
from datadiff.process_runtime import create_safe_executor
from datadiff.exploration_objectives import objective_feature
from datadiff.guidance import parse_guidance_targets
from datadiff.preset_catalog import (
    build_catalog_preset,
    build_experiment_config,
    catalog_preset_metadata,
)
from datadiff.run_journal import (
    append_run_journal_entries,
    build_run_journal_entry,
    write_run_journal_markdown,
)
from datadiff.runner import _configured_guidance_targets
from datadiff.scheduler import AdaptiveBudgetScheduler, AdaptiveScheduleConfig, summarize_batch_run
from datadiff.targets import TARGET_SUITES, resolve_target_backends
from datadiff.util import (
    REPORTS_DIR,
    RUNS_DIR,
    closed_loop_state_path,
    load_json,
    parse_duration,
    run_meta_path,
    utc_now,
)


def parse_seeds(value: str) -> list[int]:
    seeds: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        seeds.append(int(part))
    if not seeds:
        raise ValueError("at least one seed is required")
    return seeds


def parse_presets(value: str) -> list[str]:
    presets = []
    for part in value.split(","):
        preset = part.strip()
        if preset:
            presets.append(preset)
    if not presets:
        raise ValueError("at least one preset is required")
    return presets


def parse_target_suites(
    value: str | None,
    *,
    target_suites: dict[str, Any] | None = None,
) -> list[str]:
    if not value:
        return []
    known_suites = target_suites or TARGET_SUITES
    suites: list[str] = []
    for part in value.split(","):
        suite = part.strip()
        if not suite:
            continue
        if suite not in known_suites:
            raise ValueError(f"unknown target suite: {suite}")
        if suite not in suites:
            suites.append(suite)
    return suites


def experiment_target_runs(
    args: argparse.Namespace,
    *,
    resolve_run_backends_func: Callable[[argparse.Namespace], list[str]],
    parse_target_suites_func: Callable[[str | None], list[str]] = parse_target_suites,
    resolve_target_backends_func: Callable[..., list[str]] = resolve_target_backends,
) -> list[tuple[str, list[str]]]:
    if getattr(args, "backends", None):
        return [(getattr(args, "target_suite", "custom"), resolve_run_backends_func(args))]
    suites = parse_target_suites_func(getattr(args, "target_suites", None)) or [args.target_suite]
    return [(suite, resolve_target_backends_func(target_suite=suite)) for suite in suites]


def preset_config(
    name: str,
    *,
    build_catalog_preset_func: Callable[[str], ExperimentConfig | None] = build_catalog_preset,
) -> ExperimentConfig:
    if name.endswith("_replay"):
        config = preset_config(name[: -len("_replay")], build_catalog_preset_func=build_catalog_preset_func)
        config.enable_replay_bug = True
        return config
    if catalog_config := build_catalog_preset_func(name):
        return catalog_config
    raise ValueError(f"unknown experiment preset: {name}")


def job_experiment_meta(
    job: dict[str, Any],
    *,
    normalize_experiment_meta_func: Callable[[Any], dict[str, Any]] = normalize_experiment_meta,
) -> dict[str, Any]:
    return normalize_experiment_meta_func(job.get("experiment_meta", {}))


def job_run_semantics(
    job: dict[str, Any],
    *,
    job_experiment_meta_func: Callable[[dict[str, Any]], dict[str, Any]] = job_experiment_meta,
    resolved_run_semantics_func: Callable[..., dict[str, Any]] = resolved_run_semantics,
) -> dict[str, Any]:
    return resolved_run_semantics_func(job, job_experiment_meta_func(job))


def job_config(
    job: dict[str, Any],
    *,
    job_run_semantics_func: Callable[[dict[str, Any]], dict[str, Any]] = job_run_semantics,
    preset_config_func: Callable[[str], ExperimentConfig] = preset_config,
    apply_job_config_overrides_func: Callable[[ExperimentConfig, dict[str, Any]], None],
    build_catalog_preset_func: Callable[[str], ExperimentConfig | None] = build_catalog_preset,
    build_experiment_config_func: Callable[[str, list[str]], ExperimentConfig] = build_experiment_config,
) -> ExperimentConfig:
    run_semantics = job_run_semantics_func(job)
    base_preset = str(run_semantics.get("base_preset", "") or "").strip()
    overlays = [str(name).strip() for name in run_semantics.get("overlays", []) if str(name).strip()]
    config: ExperimentConfig | None = None
    if base_preset:
        try:
            config = (
                build_catalog_preset_func(base_preset)
                if not overlays
                else build_experiment_config_func(base_preset, overlays)
            )
        except ValueError:
            pass
    if config is None:
        config = preset_config_func(str(job["preset"]))
    apply_job_config_overrides_func(config, job)
    return config


def apply_job_config_overrides(
    config: ExperimentConfig,
    job: dict[str, Any],
    *,
    parse_guidance_targets_func: Callable[[str], list[str]] = parse_guidance_targets,
    parse_adaptive_components_func: Callable[[Any], set[str]],
    apply_adaptive_component_config_func: Callable[[ExperimentConfig, set[str]], None],
) -> None:
    config.target_version = str(job.get("target_version", config.target_version) or "").strip()
    config.fixed_version = str(job.get("fixed_version", config.fixed_version) or "").strip()
    profile_pool = parse_guidance_targets_func(
        str(job.get("profile_pool", job.get("generator_profile_pool", "")) or "")
    )
    if profile_pool:
        config.generator_profile_pool = profile_pool
    version_pair_pool = parse_guidance_targets_func(str(job.get("version_pair_pool", "") or ""))
    if version_pair_pool:
        config.version_pair_pool = version_pair_pool
    config.generator_profile_learning_weight = max(
        0.0,
        float(
            job.get(
                "profile_learning_weight",
                job.get(
                    "generator_profile_learning_weight",
                    config.generator_profile_learning_weight,
                ),
            )
            or 0.0
        ),
    )
    config.semantic_objective_learning_weight = max(
        0.0,
        float(job.get("semantic_objective_learning_weight", config.semantic_objective_learning_weight) or 0.0),
    )
    config.metamorphic_relation_learning_weight = max(
        0.0,
        float(job.get("metamorphic_relation_learning_weight", config.metamorphic_relation_learning_weight) or 0.0),
    )
    config.version_pair_learning_weight = max(
        0.0,
        float(job.get("version_pair_learning_weight", config.version_pair_learning_weight) or 0.0),
    )
    config.backend_pair_learning_weight = max(
        0.0,
        float(job.get("backend_pair_learning_weight", config.backend_pair_learning_weight) or 0.0),
    )
    config.backend_pair_priority_limit = max(
        1,
        int(job.get("backend_pair_priority_limit", config.backend_pair_priority_limit) or 3),
    )
    config.enable_parallel_backend_execution = bool(
        job.get(
            "enable_parallel_backend_execution",
            config.enable_parallel_backend_execution,
        )
    )
    if job.get("enable_backend_sampling") is not None:
        config.enable_backend_sampling = bool(job["enable_backend_sampling"])
    if job.get("backend_sample_size") is not None:
        config.backend_sample_size = max(2, int(job["backend_sample_size"]))
    if job.get("backend_full_sweep_interval") is not None:
        config.backend_full_sweep_interval = max(0, int(job["backend_full_sweep_interval"]))
    if job.get("backend_sampling_calibration_cases") is not None:
        config.backend_sampling_calibration_cases = max(
            0,
            int(job["backend_sampling_calibration_cases"]),
        )
    if job.get("backend_sampling_candidate_burst_cases") is not None:
        config.backend_sampling_candidate_burst_cases = max(
            0,
            int(job["backend_sampling_candidate_burst_cases"]),
        )
    if job.get("backend_sampling_candidate_burst_novel_only") is not None:
        config.backend_sampling_candidate_burst_novel_only = bool(
            job["backend_sampling_candidate_burst_novel_only"]
        )
    if job.get("backend_sample_confirmation_recheck_count") is not None:
        config.backend_sample_confirmation_recheck_count = max(
            0,
            int(job["backend_sample_confirmation_recheck_count"]),
        )
    if job.get("backend_sample_confirm_candidates") is not None:
        config.backend_sample_confirm_candidates = bool(job["backend_sample_confirm_candidates"])
    if job.get("enable_adaptive_candidate_pool") is not None:
        config.enable_adaptive_candidate_pool = bool(job["enable_adaptive_candidate_pool"])
    if job.get("adaptive_candidate_pool_min_size") is not None:
        config.adaptive_candidate_pool_min_size = max(
            1,
            int(job["adaptive_candidate_pool_min_size"]),
        )
    if job.get("adaptive_candidate_pool_full_sweep_interval") is not None:
        config.adaptive_candidate_pool_full_sweep_interval = max(
            0,
            int(job["adaptive_candidate_pool_full_sweep_interval"]),
        )
    if job.get("adaptive_candidate_pool_calibration_cases") is not None:
        config.adaptive_candidate_pool_calibration_cases = max(
            0,
            int(job["adaptive_candidate_pool_calibration_cases"]),
        )
    if job.get("adaptive_candidate_pool_candidate_burst_cases") is not None:
        config.adaptive_candidate_pool_candidate_burst_cases = max(
            0,
            int(job["adaptive_candidate_pool_candidate_burst_cases"]),
        )
    if job.get("adaptive_candidate_pool_candidate_burst_novel_only") is not None:
        config.adaptive_candidate_pool_candidate_burst_novel_only = bool(
            job["adaptive_candidate_pool_candidate_burst_novel_only"]
        )
    if job.get("adaptive_candidate_pool_preserve_seed_stride") is not None:
        config.adaptive_candidate_pool_preserve_seed_stride = bool(
            job["adaptive_candidate_pool_preserve_seed_stride"]
        )
    if job.get("adaptive_candidate_pool_compensate_seed_horizon") is not None:
        config.adaptive_candidate_pool_compensate_seed_horizon = bool(
            job["adaptive_candidate_pool_compensate_seed_horizon"]
        )
        if config.adaptive_candidate_pool_compensate_seed_horizon:
            config.adaptive_candidate_pool_preserve_seed_stride = True
    if not config.adaptive_candidate_pool_preserve_seed_stride:
        config.adaptive_candidate_pool_compensate_seed_horizon = False
    relation_order = parse_guidance_targets_func(str(job.get("metamorphic_relation_order", "") or ""))
    if relation_order:
        config.metamorphic_relation_order = relation_order
    if bool(job.get("enable_metamorphic_oracle", False)):
        config.enable_metamorphic_oracle = True
        config.oracle_mode = "both"
    config.strategy_snapshot_path = str(
        job.get("strategy_snapshot", config.strategy_snapshot_path) or ""
    ).strip()
    config.strategy_learning_path = str(
        job.get("strategy_learning", config.strategy_learning_path) or ""
    ).strip()
    config.freeze_strategy_snapshot = bool(
        job.get("freeze_strategy_snapshot", config.freeze_strategy_snapshot)
    )
    disabled_components = parse_adaptive_components_func(job.get("disable_adaptive_components", ""))
    apply_adaptive_component_config_func(config, disabled_components)


def populate_job_learning_metadata(
    job: dict[str, Any],
    *,
    job_config_func: Callable[[dict[str, Any]], ExperimentConfig],
    objective_feature_func: Callable[[str], str] = objective_feature,
) -> None:
    config = job_config_func(job)
    override_limit = job.get("metamorphic_variant_limit")
    effective_metamorphic_limit = (
        override_limit
        if override_limit is not None
        else config.metamorphic_variant_limit
    )
    job["scheduler_generator_profile"] = str(config.generator_profile or "")
    job["scheduler_guidance_strategy"] = str(config.guidance_strategy or "")
    job["scheduler_oracle_mode"] = str(config.oracle_mode or "")
    job["profile_pool"] = ",".join(config.generator_profile_pool)
    job["generator_profile_learning_weight"] = float(config.generator_profile_learning_weight)
    job["target_version"] = str(job.get("target_version", config.target_version) or "")
    job["fixed_version"] = str(job.get("fixed_version", config.fixed_version) or "")
    job["version_pair_pool"] = ",".join(config.version_pair_pool)
    job["semantic_objective_learning_weight"] = float(config.semantic_objective_learning_weight)
    job["metamorphic_relation_learning_weight"] = float(config.metamorphic_relation_learning_weight)
    job["version_pair_learning_weight"] = float(config.version_pair_learning_weight)
    job["backend_pair_learning_weight"] = float(config.backend_pair_learning_weight)
    job["backend_pair_priority_limit"] = int(config.backend_pair_priority_limit)
    job["enable_parallel_backend_execution"] = bool(config.enable_parallel_backend_execution)
    job["enable_backend_sampling"] = bool(config.enable_backend_sampling)
    job["backend_sample_size"] = int(config.backend_sample_size)
    job["backend_full_sweep_interval"] = int(config.backend_full_sweep_interval)
    job["backend_sampling_calibration_cases"] = int(config.backend_sampling_calibration_cases)
    job["backend_sampling_candidate_burst_cases"] = int(
        config.backend_sampling_candidate_burst_cases
    )
    job["backend_sampling_candidate_burst_novel_only"] = bool(
        config.backend_sampling_candidate_burst_novel_only
    )
    job["backend_sample_confirmation_recheck_count"] = (
        None
        if config.backend_sample_confirmation_recheck_count is None
        else int(config.backend_sample_confirmation_recheck_count)
    )
    job["backend_sample_confirm_candidates"] = bool(config.backend_sample_confirm_candidates)
    job["enable_adaptive_candidate_pool"] = bool(config.enable_adaptive_candidate_pool)
    job["adaptive_candidate_pool_min_size"] = int(config.adaptive_candidate_pool_min_size)
    job["adaptive_candidate_pool_full_sweep_interval"] = int(
        config.adaptive_candidate_pool_full_sweep_interval
    )
    job["adaptive_candidate_pool_calibration_cases"] = int(
        config.adaptive_candidate_pool_calibration_cases
    )
    job["adaptive_candidate_pool_candidate_burst_cases"] = int(
        config.adaptive_candidate_pool_candidate_burst_cases
    )
    job["adaptive_candidate_pool_candidate_burst_novel_only"] = bool(
        config.adaptive_candidate_pool_candidate_burst_novel_only
    )
    job["adaptive_candidate_pool_preserve_seed_stride"] = bool(
        config.adaptive_candidate_pool_preserve_seed_stride
    )
    job["adaptive_candidate_pool_compensate_seed_horizon"] = bool(
        config.adaptive_candidate_pool_compensate_seed_horizon
    )
    job["scheduler_enable_feedback"] = bool(config.enable_feedback)
    job["scheduler_enable_metamorphic_oracle"] = bool(config.enable_metamorphic_oracle)
    job["scheduler_effective_metamorphic_variant_limit"] = (
        max(0, int(effective_metamorphic_limit or 0))
        if config.enable_metamorphic_oracle
        else 0
    )
    job["scheduler_guidance_targets"] = list(config.guidance_targets)
    job["scheduler_semantic_focus_families"] = list(config.semantic_focus_families)
    job["scheduler_semantic_focus_signals"] = list(config.semantic_focus_signals)
    job["scheduler_semantic_objectives"] = [
        objective_feature_func(getattr(rule, "objective", ""))
        for rule in config.exploration_objective_rules
        if objective_feature_func(getattr(rule, "objective", ""))
    ]


def effective_job_local_source_scheduler(
    job: dict,
    *,
    parse_adaptive_components_func: Callable[[Any], set[str]],
    job_config_func: Callable[[dict[str, Any]], ExperimentConfig],
) -> tuple[bool, float]:
    disabled_components = parse_adaptive_components_func(job.get("disable_adaptive_components", ""))
    if "local_source_scheduler" in disabled_components:
        return False, 0.0
    preset_config = job_config_func(job)
    job_enabled = bool(job.get("enable_local_source_scheduler", False))
    enabled = preset_config.enable_local_source_scheduler or job_enabled
    if job_enabled:
        weight = job.get("local_source_exploration_weight", preset_config.local_source_exploration_weight)
    else:
        weight = preset_config.local_source_exploration_weight
    return enabled, max(0.0, float(weight))


def record_experiment_journal(
    manifest_path: Path,
    manifest: dict,
    *,
    reports_dir: Path = REPORTS_DIR,
    experiment_run_theme_func: Callable[[dict, dict], str],
    build_run_journal_entry_func: Callable[[Path, dict], dict] = build_run_journal_entry,
    append_run_journal_entries_func: Callable[[list[dict], Path], None] = append_run_journal_entries,
    write_run_journal_markdown_func: Callable[[Path], Path] = write_run_journal_markdown,
) -> tuple[Path, Path]:
    journal_file = reports_dir / "paper-run-journal.jsonl"
    entries = []
    for run in manifest.get("runs", []):
        run_file = Path(run.get("run_file", ""))
        context = {
            "command": "experiment",
            "theme": experiment_run_theme_func(manifest, run),
            "notes": str(manifest.get("paper_notes", "") or ""),
            "evidence_mode": str(run.get("evidence_mode") or manifest.get("evidence_mode", "")),
            "known_bug_id": str(run.get("known_bug_id") or manifest.get("known_bug_id", "")),
            "target_version": str(run.get("target_version") or manifest.get("target_version", "")),
            "target_suite": str(run.get("target_suite", "")),
            "preset": str(run.get("preset", "")),
            "seed": run.get("seed", ""),
            "backends": run.get("backends", []),
            "manifest_file": str(manifest_path),
            "experiment_meta": manifest.get("experiment_meta", {}),
        }
        entries.append(build_run_journal_entry_func(run_file, context))
    append_run_journal_entries_func(entries, journal_file)
    md_path = write_run_journal_markdown_func(journal_file)
    return journal_file, md_path


def experiment_run_theme(
    manifest: dict,
    run: dict,
    *,
    replay_bug_enabled_by_default_func: Callable[[str], bool] = replay_bug_enabled_by_default,
) -> str:
    base = str(manifest.get("run_theme", "") or "").strip()
    suffix = f"{run.get('target_suite', '')}:{run.get('preset', '')}:seed{run.get('seed', '')}"
    if base:
        return f"{base} | {suffix}"
    evidence_mode = str(run.get("evidence_mode") or manifest.get("evidence_mode", "live"))
    known_bug_id = str(run.get("known_bug_id") or manifest.get("known_bug_id", "") or "")
    if replay_bug_enabled_by_default_func(evidence_mode) and known_bug_id:
        return f"historical:{known_bug_id}:{suffix}"
    return f"{evidence_mode}:{suffix}"


def run_experiment_adaptive(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    planned_runs: list[dict[str, Any]],
    duration_s: float | None,
    *,
    jobs: int,
    parse_adaptive_components_func: Callable[[Any], set[str]],
    adaptive_component_config_func: Callable[[set[str]], dict[str, bool]],
    parse_duration_func: Callable[[Any], float | None] = parse_duration,
    adaptive_schedule_config_cls: type = AdaptiveScheduleConfig,
    adaptive_budget_scheduler_cls: type = AdaptiveBudgetScheduler,
    adaptive_learning_state_from_ledgers_func: Callable[[Any], tuple[Any, list[dict[str, Any]]]],
    run_experiment_job_func: Callable[[dict[str, Any]], dict[str, Any]],
    adaptive_job_func: Callable[[Any], dict[str, Any]],
    complete_adaptive_round_func: Callable[[Any, list[dict[str, Any]], list[Any]], list[dict[str, Any]]],
    run_experiment_adaptive_parallel_func: Callable[[type, int, Any], list[dict[str, Any]]],
    process_pool_executor_cls: type,
    thread_pool_executor_cls: type,
) -> list[dict[str, Any]]:
    disabled_components = parse_adaptive_components_func(
        getattr(args, "disable_adaptive_components", "")
    )
    adaptive_components = adaptive_component_config_func(disabled_components)
    default_batch_cases = min(100, max(1, int(args.cases or 100)))
    batch_cases = max(1, int(getattr(args, "batch_cases", 0) or default_batch_cases))
    batch_duration_s = parse_duration_func(getattr(args, "batch_duration", None))
    if batch_duration_s is None and duration_s is not None and args.cases is None:
        batch_duration_s = min(duration_s, 30.0)
    if args.cases is not None:
        total_cases_budget = len(planned_runs) * max(1, int(args.cases))
    elif duration_s is None:
        total_cases_budget = len(planned_runs) * 100
    else:
        total_cases_budget = None
    total_duration_budget_s = None if args.cases is not None or duration_s is None else duration_s
    schedule_config = adaptive_schedule_config_cls(
        batch_cases=batch_cases,
        batch_duration_s=batch_duration_s,
        warmup_batches=max(1, int(getattr(args, "warmup_batches", 1))),
        exploration_weight=max(0.0, float(getattr(args, "exploration_weight", 0.75))),
        group_fairness_weight=max(0.0, float(getattr(args, "group_fairness_weight", 0.40))),
        max_group_pull_gap=max(0, int(getattr(args, "max_group_pull_gap", 3))),
        learning_weight=(
            max(0.0, float(getattr(args, "adaptive_learning_weight", 0.0)))
            if adaptive_components["scheduler_learning"]
            else 0.0
        ),
        record_learning_feedback=adaptive_components["scheduler_learning"],
        enable_runtime_cost_learning=adaptive_components["runtime_cost_learning"],
        enable_active_learning=adaptive_components["active_learning"],
        enable_online_reward_model=adaptive_components["online_reward_model"],
        enable_continual_learning=adaptive_components["continual_learning"],
        enable_bayesian_exploration=adaptive_components["bayesian_exploration"],
        enable_cost_normalized_reward=adaptive_components["cost_normalized_reward"],
        annealing_initial_temperature=(
            max(0.0, float(getattr(args, "scheduler_annealing_temperature", 0.0) or 0.0))
            if adaptive_components["scheduler_annealing"]
            else 0.0
        ),
        annealing_decay=max(0.0, float(getattr(args, "scheduler_annealing_decay", 0.985) or 0.0)),
        annealing_min_temperature=max(
            0.0,
            float(getattr(args, "scheduler_annealing_min_temperature", 0.02) or 0.0),
        ),
    )
    learning_state, continual_learning_sources = adaptive_learning_state_from_ledgers_func(
        getattr(args, "continual_learning_ledgers", "")
    )
    scheduler = adaptive_budget_scheduler_cls(
        planned_runs,
        total_cases_budget=total_cases_budget,
        total_duration_budget_s=total_duration_budget_s,
        config=schedule_config,
        learning_state=learning_state,
    )
    manifest["adaptive_config"] = {
        "total_cases_budget": total_cases_budget,
        "total_duration_budget_s": total_duration_budget_s,
        "batch_cases": batch_cases,
        "batch_duration_s": batch_duration_s,
        "warmup_batches": schedule_config.warmup_batches,
        "exploration_weight": schedule_config.exploration_weight,
        "group_fairness_weight": schedule_config.group_fairness_weight,
        "max_group_pull_gap": schedule_config.max_group_pull_gap,
        "prefer_group_diversity_in_round": schedule_config.prefer_group_diversity_in_round,
        "learning_weight": schedule_config.learning_weight,
        "record_learning_feedback": schedule_config.record_learning_feedback,
        "runtime_cost_learning": schedule_config.enable_runtime_cost_learning,
        "active_learning": schedule_config.enable_active_learning,
        "online_reward_model": schedule_config.enable_online_reward_model,
        "continual_learning": schedule_config.enable_continual_learning,
        "bayesian_exploration": schedule_config.enable_bayesian_exploration,
        "cost_normalized_reward": schedule_config.enable_cost_normalized_reward,
        "scheduler_annealing": adaptive_components["scheduler_annealing"],
        "annealing_initial_temperature": schedule_config.annealing_initial_temperature,
        "annealing_decay": schedule_config.annealing_decay,
        "annealing_min_temperature": schedule_config.annealing_min_temperature,
        "fine_grained_local_source_scheduler": adaptive_components["local_source_scheduler"],
        "local_source_exploration_weight": max(
            0.0,
            float(getattr(args, "local_source_exploration_weight", 0.5))
            if adaptive_components["local_source_scheduler"]
            else 0.0,
        ),
        "components": dict(adaptive_components),
        "disabled_components": sorted(disabled_components),
        "continual_learning_sources": continual_learning_sources,
        "jobs": jobs,
        "parallelism": manifest.get("parallelism", {}),
    }
    completed_runs = []
    worker_count = min(max(1, jobs), len(planned_runs))
    if worker_count == 1:
        while scheduler.has_budget():
            batches = scheduler.next_round(1)
            if not batches:
                break
            completed_runs.extend(
                complete_adaptive_round_func(
                    scheduler,
                    [run_experiment_job_func(adaptive_job_func(batches[0]))],
                    batches,
                )
            )
            print(completed_runs[-1]["message"], flush=True)
        return completed_runs
    try:
        completed_runs.extend(run_experiment_adaptive_parallel_func(process_pool_executor_cls, worker_count, scheduler))
    except PermissionError:
        print("process parallelism unavailable; falling back to threaded workers", flush=True)
        completed_runs.extend(run_experiment_adaptive_parallel_func(thread_pool_executor_cls, worker_count, scheduler))
    return completed_runs


def adaptive_learning_state_from_ledgers(
    value: Any,
    *,
    adaptive_learning_state_cls: type = AdaptiveLearningState,
    continual_priority_memory_cls: type = ContinualPriorityMemory,
    parse_guidance_targets_func: Callable[[str], list[str]] = parse_guidance_targets,
    load_json_func: Callable[[Path], Any] = load_json,
    validation_error_func: Callable[[dict[str, Any]], str],
) -> tuple[Any, list[dict[str, Any]]]:
    state = adaptive_learning_state_cls()
    sources: list[dict[str, Any]] = []
    for path_text in parse_guidance_targets_func(str(value or "")):
        path = Path(path_text)
        if not path.is_file():
            sources.append({"path": str(path), "loaded": False, "reason": "missing"})
            continue
        payload = load_json_func(path)
        if not isinstance(payload, dict):
            sources.append({"path": str(path), "loaded": False, "reason": "not_object"})
            continue
        ledger_error = validation_error_func(payload)
        if ledger_error:
            sources.append(
                {
                    "path": str(path),
                    "loaded": False,
                    "reason": ledger_error,
                    "schema_version": str(payload.get("schema_version", "") or ""),
                }
            )
            continue
        seed = payload.get("adaptive_learning_seed", {})
        if isinstance(seed, dict) and isinstance(seed.get("continual_priority_memory"), dict):
            summary = state.continual_priority_memory.merge(
                continual_priority_memory_cls.from_state_dict(seed.get("continual_priority_memory"))
            )
        else:
            summary = state.ingest_continual_ledger(payload)
        sources.append(
            {
                "path": str(path),
                "loaded": True,
                "schema_version": str(payload.get("schema_version", "") or ""),
                "family_count": int(summary.get("family_count", 0) or 0),
                "feature_count": int(summary.get("feature_count", 0) or 0),
                "status_counts": dict(summary.get("status_counts", {}) or {}),
            }
        )
    return state, sources


def continual_learning_ledger_validation_error(payload: dict[str, Any]) -> str:
    if str(payload.get("schema_version", "") or "") != "version-ledger-v1":
        return "schema_mismatch"
    health = payload.get("health", {}) if isinstance(payload.get("health", {}), dict) else {}
    if str(health.get("schema_version", "") or "") != "version-ledger-health-v1":
        return "missing_health_feedback"
    report = (
        payload.get("health_feedback_report", {})
        if isinstance(payload.get("health_feedback_report", {}), dict)
        else {}
    )
    if str(report.get("schema_version", "") or "") != "version-ledger-health-feedback-report-v1":
        return "missing_health_feedback_report"
    return ""


def resolve_experiment_parallelism(
    args: argparse.Namespace,
    planned_runs: list[dict[str, Any]],
    *,
    cpu_count_func: Callable[[], int | None] = os.cpu_count,
    experiment_job_weight_func: Callable[[dict[str, Any]], float],
) -> dict[str, Any]:
    cpu_count = max(1, cpu_count_func() or 1)
    requested = getattr(args, "jobs", "auto")
    if requested == "auto":
        worker_count = max(1, min(len(planned_runs), max(1, cpu_count // 4), 6))
    else:
        worker_count = max(1, int(requested))
        worker_count = min(worker_count, max(1, len(planned_runs)))
    costs = [experiment_job_weight_func(job) for job in planned_runs] or [1.0]
    max_job_cost = max(costs)
    requested_cost = getattr(args, "max_parallel_cost", None)
    if requested_cost is None:
        max_parallel_cost = max(max_job_cost, cpu_count * 0.75)
    else:
        max_parallel_cost = max(max_job_cost, float(requested_cost))
    worker_thread_limit = max(1, min(4, cpu_count // max(1, worker_count)))
    return {
        "requested_jobs": requested,
        "worker_count": worker_count,
        "cpu_count": cpu_count,
        "max_parallel_cost": round(max_parallel_cost, 4),
        "max_job_cost": round(max_job_cost, 4),
        "worker_thread_limit": worker_thread_limit,
        "bounded_submission": True,
        "cost_limited": True,
    }


def resolve_experiment_worker_batching(
    args: argparse.Namespace,
    planned_runs: list[dict[str, Any]],
    *,
    jobs: int,
    schedule: str,
) -> dict[str, Any]:
    return resolve_worker_batching(
        args,
        planned_runs,
        jobs=jobs,
        schedule=schedule,
    )


def run_experiment_adaptive_parallel(
    executor_cls: type,
    worker_count: int,
    scheduler: AdaptiveBudgetScheduler,
    *,
    run_experiment_job_func: Callable[[dict[str, Any]], dict[str, Any]],
    adaptive_job_func: Callable[[Any], dict[str, Any]],
    complete_adaptive_round_func: Callable[[Any, list[dict[str, Any]], list[Any]], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    completed_runs: list[dict[str, Any]] = []
    with create_safe_executor(executor_cls, max_workers=worker_count) as executor:
        while scheduler.has_budget():
            batches = scheduler.next_round(worker_count)
            if not batches:
                break
            futures = {
                executor.submit(run_experiment_job_func, adaptive_job_func(batch)): batch
                for batch in batches
            }
            round_results = [future.result() for future in as_completed(futures)]
            completed_runs.extend(complete_adaptive_round_func(scheduler, round_results, batches))
            for item in completed_runs[-len(batches):]:
                print(item["message"], flush=True)
    return completed_runs


def adaptive_job(
    batch: Any,
    *,
    parse_adaptive_components_func: Callable[[Any], set[str]],
) -> dict[str, Any]:
    job = dict(batch.job)
    disabled_components = parse_adaptive_components_func(job.get("disable_adaptive_components", ""))
    job["enable_local_source_scheduler"] = "local_source_scheduler" not in disabled_components
    return job


def complete_adaptive_round(
    scheduler: AdaptiveBudgetScheduler,
    round_results: list[dict[str, Any]],
    batches: list[Any],
    *,
    summarize_batch_run_func: Callable[[Path], Any] = summarize_batch_run,
    run_meta_path_func: Callable[[Path], Path] = run_meta_path,
    load_json_func: Callable[[Path], Any] = load_json,
    load_closed_loop_state_from_meta_func: Callable[..., dict[str, Any] | None],
) -> list[dict[str, Any]]:
    results_by_batch = {int(item["run"]["batch_index"]): item for item in round_results}
    batch_by_index = {int(batch.batch_index): batch for batch in batches}
    completed: list[dict[str, Any]] = []
    for batch_index in sorted(batch_by_index):
        batch = batch_by_index[batch_index]
        result = results_by_batch[batch_index]
        run_file = Path(result["run"]["run_file"])
        observation = summarize_batch_run_func(run_file)
        meta_path = run_meta_path_func(run_file)
        meta = load_json_func(meta_path) if meta_path.exists() else {}
        loaded_closed_loop_state = load_closed_loop_state_from_meta_func(meta, run_file=run_file)
        reward = scheduler.record_result(
            batch,
            observation,
            next_seed=int(meta.get("next_seed", batch.seed + max(1, observation.cases))),
            closed_loop_state=loaded_closed_loop_state,
        )
        result["run"].update(
            {
                "schedule_arm_id": batch.arm_id,
                "batch_index": batch.batch_index,
                "closed_loop_state_present": isinstance(loaded_closed_loop_state, dict),
                "scheduler_reward": reward,
                "scheduler_observation": {
                    "cases": observation.cases,
                    "elapsed_s": observation.elapsed_s,
                    "throughput_cases_s": observation.throughput_cases_s,
                    "findings": observation.findings,
                    "candidate_bug_cases": observation.candidate_bug_cases,
                    "candidate_bug_families": sorted(observation.candidate_bug_families),
                    "semantic_divergence_count": observation.semantic_divergence_count,
                    "false_positive_count": observation.false_positive_count,
                    "new_behavior_cases": observation.new_behavior_cases,
                    "signal_new_behavior_cases": observation.signal_new_behavior_cases,
                    "first_candidate_bug_case_index": observation.first_candidate_bug_case_index,
                    "first_candidate_bug_elapsed_s": observation.first_candidate_bug_elapsed_s,
                    "candidate_bug_discovery_auc": observation.candidate_bug_discovery_auc,
                    "feedback_case_count": observation.feedback_case_count,
                    "feedback_mutation_cases": observation.feedback_mutation_cases,
                    "stored_in_feedback_corpus_cases": observation.stored_in_feedback_corpus_cases,
                    "quality_oracle_count": observation.quality_oracle_count,
                    "quality_pass_count": observation.quality_pass_count,
                    "quality_fail_count": observation.quality_fail_count,
                    "quality_score_total": observation.quality_score_total,
                    "source_reward_adjustment_total": observation.source_reward_adjustment_total,
                    "guidance_reward_adjustment_total": observation.guidance_reward_adjustment_total,
                    "seed_schedule_delta_total": observation.seed_schedule_delta_total,
                    "productive_mutation_cases": observation.productive_mutation_cases,
                    "invalid_mutation_cases": observation.invalid_mutation_cases,
                    "redundant_mutation_cases": observation.redundant_mutation_cases,
                    "feedback_finding_yield_cases": observation.feedback_finding_yield_cases,
                    "feedback_new_behavior_yield_cases": observation.feedback_new_behavior_yield_cases,
                    "feedback_redundant_behavior_cases": observation.feedback_redundant_behavior_cases,
                    "guided_productive_cases": observation.guided_productive_cases,
                    "guided_target_miss_cases": observation.guided_target_miss_cases,
                    "guided_redundant_cases": observation.guided_redundant_cases,
                },
            }
        )
        completed.append(
            {
                **result,
                "scheduler_state": scheduler.snapshot(),
                "adaptive_learning": scheduler.learning_state.to_state_dict(),
                "message": (
                    f"{result['message']} batch={batch.batch_index} "
                    f"reward={reward:.3f} remaining_cases={scheduler.remaining_cases_budget} "
                    f"remaining_duration_s={scheduler.remaining_duration_budget_s}"
                ),
            }
        )
    return completed


def load_closed_loop_state_from_meta(
    meta: dict[str, Any],
    *,
    run_file: Path,
    closed_loop_state_path_func: Callable[[Path], Path] = closed_loop_state_path,
    load_json_func: Callable[[Path], Any] = load_json,
) -> dict[str, Any] | None:
    if not isinstance(meta, dict):
        return None
    inline_state = meta.get("closed_loop_state")
    if isinstance(inline_state, dict):
        return inline_state
    state_file = str(meta.get("closed_loop_state_file", "") or "").strip()
    candidate_paths = [Path(state_file)] if state_file else []
    candidate_paths.append(closed_loop_state_path_func(run_file))
    for path in candidate_paths:
        if not str(path):
            continue
        if path.exists():
            loaded = load_json_func(path)
            if isinstance(loaded, dict):
                return loaded
    return None


def resolve_experiment_schedule(args: argparse.Namespace, *, jobs: int) -> str:
    schedule = getattr(args, "schedule", None)
    if schedule:
        return str(schedule)
    return "longest_first" if jobs > 1 else "matrix_order"


def resolve_evidence_mode(value: str, suite_names: list[str]) -> str:
    if value != "auto":
        return value
    if suite_names and all(suite.startswith("seeded_") for suite in suite_names):
        return "seeded"
    return "live"


def default_experiment_meta_for_runs(
    *,
    evidence_mode: str,
    target_runs: list[tuple[str, list[str]]],
    presets: list[str],
    known_bug_id: str,
    target_version: str,
    include_pending_historical: bool = True,
    registered_experiment_meta_defaults_func: Callable[..., dict[str, Any]] = registered_experiment_meta_defaults,
    registered_experiment_matrix_for_run_func: Callable[..., Any] = registered_experiment_matrix_for_run,
) -> dict[str, Any]:
    if evidence_mode == "historical":
        target_suite = target_runs[0][0] if len(target_runs) == 1 else ",".join(suite for suite, _ in target_runs)
        return registered_experiment_meta_defaults_func(
            evidence_mode=evidence_mode,
            known_bug_id=known_bug_id,
            target_suite=target_suite,
            target_version=target_version,
            include_pending_historical=include_pending_historical,
        )
    target_suites = tuple(
        dict.fromkeys(str(suite or "").strip() for suite, _ in target_runs if str(suite or "").strip())
    )
    if not target_suites:
        return {}
    resolved_matrices = []
    for suite in target_suites:
        for preset in presets:
            matrix = registered_experiment_matrix_for_run_func(
                evidence_mode=evidence_mode,
                target_suite=suite,
                preset=str(preset or "").strip(),
            )
            if matrix is None:
                return {}
            resolved_matrices.append(matrix)
    if not resolved_matrices:
        return {}
    matrix_ids = {matrix.id for matrix in resolved_matrices}
    if len(matrix_ids) != 1:
        return {}
    return resolved_matrices[0].to_experiment_meta(target_suites=target_suites)


def invalid_live_adaptive_experiment_presets(
    planned_runs: list[dict[str, Any]],
    *,
    job_config_func: Callable[[dict[str, Any]], ExperimentConfig],
) -> list[str]:
    invalid: list[str] = []
    for job in planned_runs:
        if job_config_func(job).guidance_strategy != "guided":
            invalid.append(str(job["preset"]))
    return invalid


def experiment_manifest_path(
    *,
    runs_dir: Path = RUNS_DIR,
    utc_now_func: Callable[[], str] = utc_now,
    time_ns_func: Callable[[], int] = time.time_ns,
) -> Path:
    ts = utc_now_func().replace(":", "").replace("-", "").replace("Z", "")
    return runs_dir / f"experiment-{ts}-{time_ns_func()}.json"


def run_experiment_jobs_parallel(
    executor_cls: type,
    worker_count: int,
    planned_runs: list[dict],
    *,
    max_parallel_cost: float,
    experiment_job_sort_key_func: Callable[[dict], tuple[float, int]],
    next_schedulable_job_index_func: Callable[[list[dict], float], int | None],
    job_estimated_cost_func: Callable[[dict], float],
    run_experiment_job_func: Callable[[dict], dict],
    run_experiment_job_batch_func: Callable[[dict], dict] | None = None,
    worker_batch_size: int = 0,
    worker_max_rss_kib: int = 0,
    worker_retry_limit: int = 0,
    worker_batch_manifest: dict[str, Any] | None = None,
) -> list[dict]:
    if int(worker_batch_size) > 0 and run_experiment_job_batch_func is not None:
        return run_batched_parallel(
            executor_cls,
            worker_count,
            planned_runs,
            max_parallel_cost=max_parallel_cost,
            worker_batch_size=int(worker_batch_size),
            worker_max_rss_kib=int(worker_max_rss_kib),
            worker_retry_limit=int(worker_retry_limit),
            experiment_job_sort_key_func=experiment_job_sort_key_func,
            job_estimated_cost_func=job_estimated_cost_func,
            run_experiment_job_batch_func=run_experiment_job_batch_func,
            worker_batch_manifest=(
                worker_batch_manifest if worker_batch_manifest is not None else {}
            ),
        )
    completed_runs = []
    scheduled_runs = sorted(planned_runs, key=experiment_job_sort_key_func)
    queued_runs = list(scheduled_runs)
    running = {}
    running_cost = 0.0
    with create_safe_executor(executor_cls, max_workers=worker_count) as executor:
        while queued_runs or running:
            while len(running) < worker_count and queued_runs:
                available_cost = max_parallel_cost - running_cost
                index = next_schedulable_job_index_func(queued_runs, available_cost)
                if index is None:
                    break
                job = queued_runs.pop(index)
                cost = job_estimated_cost_func(job)
                future = executor.submit(run_experiment_job_func, job)
                running[future] = cost
                running_cost += cost
            if not running and queued_runs:
                job = queued_runs.pop(0)
                cost = job_estimated_cost_func(job)
                future = executor.submit(run_experiment_job_func, job)
                running[future] = cost
                running_cost += cost
            done, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in done:
                running_cost -= running.pop(future)
                result = future.result()
                completed_runs.append(result)
                print(result["message"], flush=True)
    return completed_runs


def run_experiment_job_batch(
    batch: dict[str, Any],
    *,
    run_experiment_job_func: Callable[[dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    return run_worker_batch(
        batch,
        run_experiment_job_func=run_experiment_job_func,
    )


def next_schedulable_job_index(
    queued_runs: list[dict],
    available_cost: float,
    *,
    job_estimated_cost_func: Callable[[dict], float],
) -> int | None:
    for index, job in enumerate(queued_runs):
        if job_estimated_cost_func(job) <= max(0.0, available_cost):
            return index
    return None


def job_estimated_cost(
    job: dict,
    *,
    experiment_job_weight_func: Callable[[dict], float],
) -> float:
    return float(job.get("estimated_cost", experiment_job_weight_func(job)))


def experiment_job_sort_key(
    job: dict,
    *,
    experiment_job_weight_func: Callable[[dict], float],
) -> tuple[float, int]:
    return (-experiment_job_weight_func(job), int(job["order"]))


def experiment_job_weight(
    job: dict,
    *,
    job_config_func: Callable[[dict[str, Any]], ExperimentConfig],
    backend_cost_func: Callable[[str], float],
) -> float:
    config = job_config_func(job)
    backend_cost_total = sum(backend_cost_func(str(backend)) for backend in job["backends"])
    metamorphic_multiplier = 1.0
    if config.enable_metamorphic_oracle:
        metamorphic_multiplier += max(1, int(config.metamorphic_variant_limit))
    profile_multiplier = {
        "float_group_key": 1.6,
        "join_null_sort": 1.5,
        "discovery": 1.3,
        "common_api_workflow": 0.9,
        "discovery_no_groupby": 1.2,
        "workflow": 1.2,
        "edge_float": 1.1,
        "null_groupby_topk": 1.0,
        "null_agg_topk": 1.0,
        "filter_null_agg_topk": 1.2,
        "join_null_agg_topk": 1.3,
        "join_null_key_topk": 1.3,
        "wide_offset_topk": 1.4,
        "empty_filter_groupby": 1.1,
        "join_filter_groupby": 1.4,
        "join_null_truth_filter": 1.3,
        "join_groupby_stress": 2.0,
        "storage_offset": 2.0,
        "ordered_groupby_sort": 1.2,
        "topk_resort": 1.1,
        "join_ordered_agg_topk": 1.5,
        "boolean_predicate_filter": 1.1,
        "post_topk_range_filter": 1.1,
        "tuple_absence_filter": 1.2,
        "row_value_absence_filter": 1.2,
        "running_sum_precision": 1.4,
        "partitioned_running_sum": 1.2,
        "path_basename_keyed_pick": 1.4,
        "sortedness_null_placement": 1.0,
        "simple_case_random_subject": 1.0,
        "group_quantile_key_probe": 1.0,
        "scalar_subquery_double_parentheses": 1.0,
        "window_avg_rows_frame": 1.0,
        "struct_distinct_unnest": 1.0,
        "bit_compare_unequal_length": 1.0,
        "round_even_float_scale": 1.0,
        "duckdb_float_literal_precision": 1.0,
        "polars_timestamp_precision_filter": 1.0,
        "series_rtruediv_operand_order": 1.0,
        "polars_reverse_division_columns": 1.0,
        "pandas_uint64_isin_precision": 1.0,
        "duckdb_tuple_anti_null_semantics": 1.0,
        "datafusion_setop_all_duplicate_count": 1.0,
        "duckdb_json_predicate_order_semantics": 1.0,
        "pandas_sparse_array_mask_semantics": 1.0,
        "polars_float_wrap_numerical_semantics": 1.0,
        "pandas_index_bool_result_type": 1.0,
        "polars_empty_literal_groupby_semantics": 1.0,
        "pandas_arrow_string_eq_sum_semantics": 1.0,
        "pandas_arrow_timestamp_loc_slice_semantics": 1.0,
        "pandas_arrow_timestamp_index_attr_semantics": 1.0,
        "pandas_eval_inplace_aliasing_semantics": 1.0,
        "pandas_bool_reduction_skipna_semantics": 1.0,
        "pyarrow_dataset_isin_all_match_semantics": 1.0,
        "pyarrow_run_end_null_compute_semantics": 1.0,
        "pyarrow_large_string_partition_schema_semantics": 1.0,
        "pyarrow_hash_pivot_wider_order_semantics": 1.0,
        "pyarrow_list_flatten_parent_indices_semantics": 1.0,
        "polars_rolling_mean_by_null_count_semantics": 1.0,
        "csv_long_numeric_roundtrip": 1.0,
        "deep_probe_rotation": 1.0,
        "common": 1.0,
    }.get(config.generator_profile, 1.0)
    guidance_multiplier = 1.0 + 0.03 * max(0, int(config.guidance_candidate_pool) - 1)
    return backend_cost_total * metamorphic_multiplier * profile_multiplier * guidance_multiplier


def backend_cost(backend: str) -> float:
    return {
        "datafusion": 1.5,
        "polars_lazy": 1.4,
        "duckdb": 1.2,
        "duckdb_persistent": 1.4,
        "sqlite": 1.0,
        "polars": 1.0,
        "pandas": 1.0,
        "pyarrow": 1.0,
    }.get(backend, 1.0)


def run_experiment_job(
    job: dict,
    *,
    apply_native_thread_limits_func: Callable[[int], None],
    job_config_func: Callable[[dict[str, Any]], ExperimentConfig],
    parse_adaptive_components_func: Callable[[Any], set[str]],
    adaptive_component_config_func: Callable[[set[str]], dict[str, bool]],
    apply_adaptive_component_config_func: Callable[[ExperimentConfig, set[str]], None],
    run_fuzz_func: Callable[..., Path],
    write_report_func: Callable[[Path], tuple[Path, Path]],
    job_experiment_meta_func: Callable[[dict[str, Any]], dict[str, Any]] = job_experiment_meta,
    resolved_run_semantics_func: Callable[..., dict[str, Any]] = resolved_run_semantics,
    catalog_preset_metadata_func: Callable[..., dict[str, Any]] = catalog_preset_metadata,
    configured_guidance_targets_func: Callable[[ExperimentConfig], list[str]] = _configured_guidance_targets,
) -> dict:
    apply_native_thread_limits_func(int(job.get("worker_thread_limit", 1) or 1))
    preset_config_value = job_config_func(job)
    disabled_components = parse_adaptive_components_func(job.get("disable_adaptive_components", ""))
    adaptive_components = adaptive_component_config_func(disabled_components)
    apply_adaptive_component_config_func(preset_config_value, disabled_components)
    preset_config_value.log_level = str(job["log_level"])
    preset_config_value.compress_run_log = bool(job["compress_run_log"])
    preset_config_value.artifact_limit = job["artifact_limit"]
    job_source_scheduler_enabled = bool(job.get("enable_local_source_scheduler", False))
    preset_config_value.enable_local_source_scheduler = (
        preset_config_value.enable_local_source_scheduler or job_source_scheduler_enabled
    ) and adaptive_components["local_source_scheduler"]
    if job_source_scheduler_enabled:
        preset_config_value.local_source_exploration_weight = max(
            0.0,
            float(job.get("local_source_exploration_weight", preset_config_value.local_source_exploration_weight)),
        )
    if not adaptive_components["local_source_scheduler"]:
        preset_config_value.local_source_exploration_weight = 0.0
    if job["metamorphic_variant_limit"] is not None:
        preset_config_value.metamorphic_variant_limit = max(0, int(job["metamorphic_variant_limit"]))
    preset_config_value.enable_replay_bug = preset_config_value.enable_replay_bug or bool(
        job.get("enable_replay_bug", False)
    )
    replay_bug_sources = list(job.get("replay_bug_source_issues", []) or [])
    if replay_bug_sources:
        preset_config_value.replay_bug_source_issues = replay_bug_sources
    run_kwargs = {
        "cases": job["cases"],
        "seed": int(job["seed"]),
        "backends": list(job["backends"]),
        "config": preset_config_value,
        "duration_s": job["duration_s"],
        "checkpoint_interval_s": float(job.get("checkpoint_interval_s", 60.0) or 0.0),
        "progress_interval_s": float(job.get("progress_interval_s", 60.0) or 0.0),
    }
    if bool(job.get("persist_closed_loop_state", False)):
        run_kwargs["persist_closed_loop_state"] = True
    if isinstance(job.get("closed_loop_state"), dict):
        run_kwargs["closed_loop_state"] = job["closed_loop_state"]
    run_file = run_fuzz_func(**run_kwargs)
    md_path = csv_path = ""
    if not job["skip_run_reports"]:
        md_path, csv_path = write_report_func(run_file)
    experiment_meta = job_experiment_meta_func(job)
    run_semantics = resolved_run_semantics_func(job, experiment_meta)
    preset_metadata = catalog_preset_metadata_func(
        str(job["preset"]),
        base_preset=str(run_semantics.get("base_preset", "") or ""),
        overlays=list(run_semantics.get("overlays", []) or []),
    )
    preset_semantic_focus_families = list(preset_config_value.semantic_focus_families)
    preset_semantic_focus_signals = list(preset_config_value.semantic_focus_signals)
    configured_guidance_targets = list(preset_config_value.guidance_targets)
    configured_effective_guidance_targets = configured_guidance_targets_func(preset_config_value)
    run = {
        "target_suite": job["target_suite"],
        "backends": job["backends"],
        "preset": job["preset"],
        "seed": job["seed"],
        "evidence_mode": job.get("evidence_mode", ""),
        "known_bug_id": job.get("known_bug_id", ""),
        "target_version": job.get("target_version", ""),
        "batch_index": job.get("batch_index"),
        "schedule_arm_id": job.get("schedule_arm_id", ""),
        "estimated_cost": job.get("estimated_cost", ""),
        "worker_thread_limit": job.get("worker_thread_limit", ""),
        "closed_loop_state_present": isinstance(job.get("closed_loop_state"), dict),
        "adaptive_components": dict(adaptive_components),
        "disabled_adaptive_components": sorted(disabled_components),
        "experiment_meta": experiment_meta,
        "matrix_id": run_semantics["matrix_id"],
        "matrix_title": run_semantics["matrix_title"],
        "comparison_group": run_semantics["comparison_group"],
        "purpose": run_semantics["purpose"],
        "counts_as_real_bugs": run_semantics["counts_as_real_bugs"],
        "rq_tags": list(run_semantics["rq_tags"]),
        "analysis_tags": list(run_semantics["analysis_tags"]),
        "variant_id": run_semantics["variant_id"],
        "variant_title": run_semantics["variant_title"],
        "base_preset": run_semantics["base_preset"],
        "comparison_role": run_semantics["comparison_role"],
        "canonical_comparison_role": run_semantics["canonical_comparison_role"],
        "component_focus": run_semantics["component_focus"],
        "overlays": list(run_semantics["overlays"]),
        "semantic_focus_families": list(run_semantics["semantic_focus_families"]),
        "semantic_focus_signals": list(run_semantics["semantic_focus_signals"]),
        "factors": dict(run_semantics["factors"]),
        "oracle_profile": run_semantics["oracle_profile"],
        "scope_kind": run_semantics["scope_kind"],
        "preset_metadata": preset_metadata,
        "configured_guidance_targets": configured_guidance_targets,
        "configured_effective_guidance_targets": configured_effective_guidance_targets,
        "configured_semantic_focus_families": preset_semantic_focus_families,
        "configured_semantic_focus_signals": preset_semantic_focus_signals,
        "run_file": str(run_file),
        "report": str(md_path),
        "csv": str(csv_path),
    }
    return {
        "order": int(job["order"]),
        "run": run,
        "message": f"{job['target_suite']} {job['preset']} seed={job['seed']} run={run_file}",
    }


NATIVE_THREAD_LIMIT_ENV_NAMES = (
    "DATADIFF_DUCKDB_THREADS",
    "POLARS_MAX_THREADS",
    "RAYON_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "ARROW_NUM_THREADS",
)


def capture_native_thread_limits() -> dict[str, str | None]:
    return {name: os.environ.get(name) for name in NATIVE_THREAD_LIMIT_ENV_NAMES}


def restore_native_thread_limits(previous: Mapping[str, str | None]) -> None:
    for name in NATIVE_THREAD_LIMIT_ENV_NAMES:
        value = previous.get(name)
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = str(value)


def apply_native_thread_limits(thread_limit: int) -> None:
    value = str(max(1, int(thread_limit)))
    for name in NATIVE_THREAD_LIMIT_ENV_NAMES:
        os.environ[name] = value
