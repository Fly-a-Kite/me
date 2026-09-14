from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor as _ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from pathlib import Path
from typing import get_args


class ProcessPoolExecutor(_ProcessPoolExecutor):
    """Process pool that never forks a potentially multi-threaded Python parent."""

    def __init__(self, *args, mp_context=None, **kwargs):
        super().__init__(
            *args,
            mp_context=mp_context or multiprocessing.get_context("spawn"),
            **kwargs,
        )

from datadiff.ablation_audit import analyze_ablation_audit
from datadiff.adaptive_learning import AdaptiveLearningState, ContinualPriorityMemory
from datadiff.adaptive_benchmark import (
    run_adaptive_benchmark,
    write_adaptive_benchmark_markdown,
)
from datadiff.bug_audit import list_audit_probe_ids, run_probe_audit, write_probe_issue_drafts
from datadiff.candidate_pipeline import DEFAULT_CANDIDATE_PIPELINE_DIR, build_candidate_pipeline
from datadiff.bug_status import build_issue_status, write_issue_status_outputs
from datadiff.commands.analysis import AnalysisCommandHandlers, register as register_analysis_commands
from datadiff.commands.analysis_handlers import (
    cmd_adaptive_benchmark_impl,
    cmd_analyze_ablation_audit_impl,
    cmd_analyze_experiment_impl,
    cmd_analyze_pattern_variants_impl,
    cmd_analyze_seeded_sensitivity_impl,
    cmd_experiment_summary_impl,
)
from datadiff.commands.artifacts import ArtifactCommandHandlers, register as register_artifact_commands
from datadiff.commands.artifact_handlers import (
    cmd_historical_status_impl,
    cmd_reduce_impl,
    cmd_replay_fixture_impl,
    cmd_reproduce_impl,
    cmd_triage_artifact_impl,
    cmd_validate_artifact_impl,
)
from datadiff.commands.core import CoreCommandHandlers, register as register_core_commands
from datadiff.commands.cross_version import register as register_cross_version_commands
from datadiff.commands.core_handlers import (
    cmd_init_impl,
    cmd_prune_corpus_impl,
    cmd_semantic_registry_impl,
    cmd_target_version_audit_impl,
    cmd_targets_impl,
)
from datadiff.commands.discovery import DiscoveryCommandHandlers, register as register_discovery_commands
from datadiff.commands.discovery_handlers import (
    cmd_candidate_pipeline_impl,
    cmd_discovery_campaign_aggregate_impl,
    cmd_discovery_campaign_impl,
    cmd_discovery_campaign_status_impl,
    cmd_discovery_run_impl,
)
from datadiff.commands.experiment import ExperimentCommandHandlers, register as register_experiment_commands
from datadiff.commands.experiment_handlers import cmd_experiment_impl
from datadiff.commands.fuzzing import FuzzingCommandHandlers, register as register_fuzzing_commands
from datadiff.commands.fuzzing_handlers import cmd_fuzz_impl, cmd_longrun_impl
from datadiff.commands.readiness import ReadinessCommandHandlers, register as register_readiness_commands
from datadiff.commands.readiness_handlers import (
    cmd_final_readiness_impl,
    cmd_review_readiness_impl,
)
from datadiff.commands.reporting import ReportingCommandHandlers, register as register_reporting_commands
from datadiff.commands.reporting_handlers import (
    cmd_bug_audit_impl,
    cmd_bug_status_impl,
    cmd_issue_bundle_impl,
    cmd_issue_readiness_impl,
    cmd_methodology_report_impl,
    cmd_report_impl,
    cmd_classify_run_impl,
    cmd_run_health_impl,
    cmd_show_bugs_impl,
)
from datadiff.commands.version_ledger import (
    _manifest_run_version_label,
    _version_ledger_runs_from_manifest_indexes,
    _write_version_ledger_evidence_manifest,
    cmd_version_ledger,
    evidence_manifest_requested,
)
from datadiff import cli_config as _cli_config
from datadiff import cli_flags as _cli_flags
from datadiff.config import (
    DEFAULT_REPLAY_BUG_SOURCE_ISSUES,
    DiscoveryBias,
    ExperimentConfig,
    GeneratorProfile,
    merge_discovery_biases,
)
from datadiff.discovery_campaign_summary import (
    DEFAULT_DISCOVERY_CAMPAIGN_HISTORY_WINDOW,
    DEFAULT_DISCOVERY_CAMPAIGN_SCORE_WEIGHTS,
    _current_discovery_campaign_run,
    _discovery_campaign_lane_rows,
    _discovery_campaign_scheduler_snapshot,
    _latest_discovery_campaign_run_file,
    _load_discovery_campaign_history_manifests,
    _parse_manifest_utc_timestamp,
    _resolve_project_cli_path,
    _summarize_discovery_campaign_aggregate,
    _summarize_discovery_campaign_status,
)
from datadiff import discovery_runtime as _discovery_runtime
from datadiff.dsl import Case
from datadiff.experiment_catalog import (
    EXPERIMENT_EVIDENCE_MODES,
    FIXTURE_REPLAY_EVIDENCE_MODES,
    registered_experiment_meta_defaults,
    registered_experiment_matrix_for_run,
    registered_experiment_meta_for_run,
    replay_bug_enabled_by_default,
)
from datadiff.experiment_analysis import analyze_experiment
from datadiff.experiment_metadata import (
    merge_experiment_meta,
    normalize_experiment_meta,
    parse_experiment_meta,
    resolved_run_semantics,
)
from datadiff import experiment_runtime as _experiment_runtime
from datadiff.exploration_objectives import (
    ExplorationObjectiveRule,
    objective_feature,
)
from datadiff.final_readiness import (
    DEFAULT_A_LEVEL_READINESS_POLICY,
    DEFAULT_FINAL_READINESS_MANIFEST_LIMIT,
    ReadinessPolicy,
    ReadinessThresholds,
    analyze_final_readiness,
)
from datadiff.fixture_replay import build_fixture_replay_case, load_fixture_replay_spec
from datadiff.historical import get_historical_bug, list_historical_bugs
from datadiff.issue_bundle import (
    DEFAULT_ISSUE_BUNDLE_STATUSES,
    build_issue_bundle,
)
from datadiff.issue_readiness import (
    build_issue_readiness,
    write_issue_readiness_outputs,
)
from datadiff.live_target_catalog import LIVE_PRESET_TARGETS_BY_NAME
from datadiff.manifest_index import (
    dedupe_paths as _dedupe_paths,
    manifest_index_files as _final_readiness_manifest_index_files,
    paths_from_index_value as _paths_from_index_value,
)
from datadiff.methodology_report import write_methodology_report
from datadiff.guidance import parse_guidance_targets
from datadiff.operation_combo import describe_operation_combo
from datadiff.pattern_analysis import analyze_pattern_variants
from datadiff.preset_catalog import build_catalog_preset
from datadiff.preset_catalog import build_experiment_config
from datadiff.preset_catalog import catalog_preset_metadata
from datadiff.reporter import (
    latest_run_log_path,
    write_experiment_summary_report,
    write_run_report,
)
from datadiff.reducer import reduce_case
from datadiff.review_readiness import (
    ReviewThresholds,
    build_review_readiness,
    write_review_readiness_outputs,
)
from datadiff.reproducer_scripts import write_reduced_reproducer
from datadiff.run_journal import (
    append_run_journal_entries,
    build_run_journal_entry,
    record_run_journal,
    write_run_journal_markdown,
)
from datadiff.run_summaries import (
    _candidate_bug_family_key,
    _candidate_bug_family_keys,
    _candidate_issue_family_key,
    _candidate_issue_family_keys,
    _candidate_row_origin,
    _case_has_non_ascii_data,
    _case_has_null_data,
    _case_has_special_float_data,
    _classification_cache_key,
    _classify_run_row,
    _is_candidate_bug_finding,
    _is_candidate_issue_finding,
    _is_known_saturated_family_key,
    _normalized_from_mapping,
    _normalized_from_row,
    _refresh_differential_findings,
    _run_health_runtime_summary,
    _summarize_run_classification,
    _summarize_run_health,
)
from datadiff.runner import _compact_log_row, _configured_guidance_targets, run_fuzz, run_loaded_case
from datadiff.scheduler import AdaptiveBudgetScheduler, AdaptiveScheduleConfig, summarize_batch_run
from datadiff.seeded_analysis import analyze_seeded_sensitivity
from datadiff.semantic_registry import semantic_registry_payload
from datadiff.strategy_registry import (
    DEFAULT_DISCOVERY_LANE_IDS,
    discovery_lane_catalog,
    discovery_lane_spec,
    discovery_biases_for_lanes,
)
from datadiff.target_version_audit import (
    build_target_version_audit,
    parse_latest_version_overrides,
    write_target_version_audit,
)
from datadiff.targets import (
    TARGETS,
    TARGET_SUITES,
    common_capabilities,
    describe_methodology,
    describe_targets,
    list_target_suites,
    parse_backend_names,
    resolve_target_backends,
    target_context,
    target_capability_matrix,
)
from datadiff.triage import (
    build_triage_report,
    supports_standalone_reproducer,
    write_standalone_reproducer,
    write_triage_artifact,
)
from datadiff.util import (
    BUGS_DIR,
    CORPUS_DIR,
    JsonlWriter,
    PROJECT_ROOT,
    REPORTS_DIR,
    RUNS_DIR,
    closed_loop_state_path,
    dump_json,
    ensure_dirs,
    load_json,
    parse_duration,
    read_jsonl,
    run_meta_path,
    slugify,
    utc_now,
    unique_preserve_order,
)

latest_run_file = latest_run_log_path
write_report = write_run_report
write_experiment_summary = write_experiment_summary_report
available_bug_audit_probe_ids = list_audit_probe_ids
run_bug_audit = run_probe_audit
write_bug_audit_issue_drafts = write_probe_issue_drafts
build_bug_status = build_issue_status
write_bug_status_outputs = write_issue_status_outputs

PROFILE_CHOICES = list(get_args(GeneratorProfile))
ADAPTIVE_COMPONENTS = _cli_config.ADAPTIVE_COMPONENTS
DEFAULT_FINAL_READINESS_THRESHOLDS = ReadinessThresholds()


def _parse_backends(value: str) -> list[str]:
    return parse_backend_names(value)


def _parse_jobs(value: str) -> int | str:
    text = str(value).strip().lower()
    if text == "auto":
        return "auto"
    jobs = int(text)
    if jobs < 1:
        raise argparse.ArgumentTypeError("--jobs must be a positive integer or 'auto'")
    return jobs


def _parse_exploration_objective_rules(value: str | None) -> list[ExplorationObjectiveRule]:
    return _cli_config.parse_exploration_objective_rules(value)


def _resolve_run_backends(args: argparse.Namespace) -> list[str]:
    return resolve_target_backends(
        getattr(args, "backends", None),
        target_suite=getattr(args, "target_suite", "core"),
    )


def _parse_adaptive_components(value: str | list[str] | tuple[str, ...] | set[str] | None) -> set[str]:
    return _cli_config.parse_adaptive_components(value)


def _parse_adaptive_component_tuple(value: str | list[str] | tuple[str, ...] | set[str] | None) -> tuple[str, ...]:
    return _cli_config.parse_adaptive_component_tuple(value)


def _adaptive_component_config(disabled_components: set[str]) -> dict[str, bool]:
    return _cli_config.adaptive_component_config(disabled_components)


def _config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    return _cli_config.config_from_args(args)


def _apply_adaptive_component_config(config: ExperimentConfig, disabled_components: set[str]) -> None:
    _cli_config.apply_adaptive_component_config(config, disabled_components)


def add_ablation_flags(parser: argparse.ArgumentParser) -> None:
    _cli_flags.add_ablation_flags(
        parser,
        parse_adaptive_components_func=_parse_adaptive_components,
        adaptive_components=ADAPTIVE_COMPONENTS,
    )


def add_guidance_flags(
    parser: argparse.ArgumentParser,
    *,
    default_strategy: str,
    default_candidate_pool: int,
) -> None:
    _cli_flags.add_guidance_flags(
        parser,
        default_strategy=default_strategy,
        default_candidate_pool=default_candidate_pool,
    )


def add_target_suite_flags(parser: argparse.ArgumentParser) -> None:
    _cli_flags.add_target_suite_flags(parser, target_suites=TARGET_SUITES)


def add_paper_journal_flags(parser: argparse.ArgumentParser) -> None:
    _cli_flags.add_paper_journal_flags(parser)


def cmd_init(args: argparse.Namespace) -> int:
    return cmd_init_impl(
        args,
        ensure_dirs_func=ensure_dirs,
        runs_dir=RUNS_DIR,
        bugs_dir=BUGS_DIR,
        reports_dir=REPORTS_DIR,
    )


def cmd_fuzz(args: argparse.Namespace) -> int:
    return cmd_fuzz_impl(
        args,
        run_fuzz_func=run_fuzz,
        resolve_run_backends_func=_resolve_run_backends,
        config_from_args_func=_config_from_args,
        parse_duration_func=parse_duration,
        record_cli_run_journal_func=_record_cli_run_journal,
    )


def _record_cli_run_journal(run_file: Path, args: argparse.Namespace, *, command: str) -> tuple[Path, Path | None]:
    context = {
        "command": command,
        "theme": _single_run_theme(args, command=command),
        "notes": str(getattr(args, "paper_notes", "") or ""),
        "evidence_mode": "live",
        "target_suite": str(getattr(args, "target_suite", "") or ""),
        "profile": str(getattr(args, "profile", "") or ""),
        "seed": getattr(args, "seed", ""),
        "backends": _resolve_run_backends(args),
    }
    return record_run_journal(
        run_file,
        context=context,
        journal_file=REPORTS_DIR / "paper-run-journal.jsonl",
    )


def _single_run_theme(args: argparse.Namespace, *, command: str) -> str:
    explicit = str(getattr(args, "run_theme", "") or "").strip()
    if explicit:
        return explicit
    target_suite = str(getattr(args, "target_suite", "") or "core")
    profile = str(getattr(args, "profile", "") or "common")
    seed = getattr(args, "seed", "")
    return f"{command}:{target_suite}:{profile}:seed{seed}"


def _print_longrun_progress(snapshot: dict) -> None:
    print(
        "progress "
        f"cases={snapshot['executed_cases']} "
        f"elapsed_s={snapshot['elapsed_s']:.1f} "
        f"cases_s={snapshot['throughput_cases_s']:.3f} "
        f"findings={snapshot['findings']} "
        f"next_seed={snapshot['next_seed']}",
        flush=True,
    )


def cmd_longrun(args: argparse.Namespace) -> int:
    return cmd_longrun_impl(
        args,
        run_fuzz_func=run_fuzz,
        resolve_run_backends_func=_resolve_run_backends,
        config_from_args_func=_config_from_args,
        parse_duration_func=parse_duration,
        print_longrun_progress_func=_print_longrun_progress,
        record_cli_run_journal_func=_record_cli_run_journal,
    )


def cmd_report(args: argparse.Namespace) -> int:
    return cmd_report_impl(
        args,
        latest_run_log_path_func=latest_run_log_path,
        write_report_func=write_report,
    )


def cmd_bug_audit(args: argparse.Namespace) -> int:
    return cmd_bug_audit_impl(
        args,
        parse_guidance_targets_func=parse_guidance_targets,
        run_probe_audit_func=run_probe_audit,
        write_probe_issue_drafts_func=write_probe_issue_drafts,
    )


def cmd_discovery_run(args: argparse.Namespace) -> int:
    return cmd_discovery_run_impl(
        args,
        ensure_dirs_func=ensure_dirs,
        resolve_run_backends_func=_resolve_run_backends,
        discovery_run_config_from_args_func=_discovery_run_config_from_args,
        parse_guidance_targets_func=parse_guidance_targets,
        run_probe_audit_func=run_probe_audit,
        write_probe_issue_drafts_func=write_probe_issue_drafts,
        project_relative_path_func=_project_relative_cli_path,
        run_fuzz_func=run_fuzz,
        parse_duration_func=parse_duration,
        write_report_func=write_report,
        summarize_run_classification_func=_summarize_run_classification,
        write_discovery_run_fresh_candidate_evidence_func=_write_discovery_run_fresh_candidate_evidence,
        run_candidate_pipeline_for_evidence_func=_run_candidate_pipeline_for_evidence,
        dump_json_func=dump_json,
        utc_now_func=utc_now,
    )


def _discovery_campaign_score_weights_from_args(args: argparse.Namespace) -> dict[str, float]:
    return _discovery_runtime.discovery_campaign_score_weights_from_args(
        args,
        default_weights=DEFAULT_DISCOVERY_CAMPAIGN_SCORE_WEIGHTS,
    )


def _next_discovery_campaign_lane(
    scheduler: dict[str, Any],
    pending_by_lane: dict[str, list[int]],
) -> dict[str, Any] | None:
    return _discovery_runtime.next_discovery_campaign_lane(scheduler, pending_by_lane)


def _aggregate_candidate_pipeline_summary(items: list[dict[str, Any]]) -> dict[str, Any]:
    return _discovery_runtime.aggregate_candidate_pipeline_summary(items)


def cmd_discovery_campaign(args: argparse.Namespace) -> int:
    return cmd_discovery_campaign_impl(
        args,
        discovery_campaign_lane_catalog_func=_discovery_campaign_lane_catalog,
        ensure_dirs_func=ensure_dirs,
        discovery_campaign_lanes_from_args_func=_discovery_campaign_lanes_from_args,
        parse_seeds_func=_parse_seeds,
        parse_duration_func=parse_duration,
        parse_guidance_targets_func=parse_guidance_targets,
        run_probe_audit_func=run_probe_audit,
        write_probe_issue_drafts_func=write_probe_issue_drafts,
        project_relative_path_func=_project_relative_cli_path,
        utc_now_func=utc_now,
        discovery_campaign_score_weights_from_args_func=_discovery_campaign_score_weights_from_args,
        default_campaign_history_window=DEFAULT_DISCOVERY_CAMPAIGN_HISTORY_WINDOW,
        discovery_campaign_scheduler_snapshot_func=_discovery_campaign_scheduler_snapshot,
        build_discovery_campaign_manifest_func=_build_discovery_campaign_manifest,
        dump_json_func=dump_json,
        next_discovery_campaign_lane_func=_next_discovery_campaign_lane,
        resolve_target_backends_func=resolve_target_backends,
        discovery_campaign_config_from_args_func=_discovery_campaign_config_from_args,
        run_fuzz_func=run_fuzz,
        write_report_func=write_report,
        summarize_run_classification_func=_summarize_run_classification,
        write_discovery_run_fresh_candidate_evidence_func=_write_discovery_run_fresh_candidate_evidence,
        run_candidate_pipeline_for_evidence_func=_run_candidate_pipeline_for_evidence,
        summarize_run_health_func=_summarize_run_health,
    )


def _build_discovery_campaign_manifest(
    *,
    manifest_path: Path,
    selected_lanes: list[dict[str, str]],
    seeds: list[int],
    audit_summary: dict[str, Any],
    runs: list[dict[str, Any]],
    aggregate_fresh: Counter[str],
    aggregate_issue_inspired: Counter[str],
    aggregate_known: Counter[str],
    aggregate_verdicts: Counter[str],
    stopped_by_health: bool,
    health_stop_reason: str,
    cases_per_lane_seed: int,
    duration: str | None,
    started_at: str,
    status: str,
    total_run_count: int,
    current_run: dict[str, Any] | None = None,
    scheduler: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return _discovery_runtime.build_discovery_campaign_manifest(
        manifest_path=manifest_path,
        selected_lanes=selected_lanes,
        seeds=seeds,
        audit_summary=audit_summary,
        runs=runs,
        aggregate_fresh=aggregate_fresh,
        aggregate_issue_inspired=aggregate_issue_inspired,
        aggregate_known=aggregate_known,
        aggregate_verdicts=aggregate_verdicts,
        stopped_by_health=stopped_by_health,
        health_stop_reason=health_stop_reason,
        cases_per_lane_seed=cases_per_lane_seed,
        duration=duration,
        started_at=started_at,
        status=status,
        total_run_count=total_run_count,
        current_run=current_run,
        scheduler=scheduler,
        utc_now_func=utc_now,
        aggregate_candidate_pipeline_summary_func=_aggregate_candidate_pipeline_summary,
    )


def _discovery_campaign_lanes_from_args(value: str) -> list[dict[str, str]]:
    return _discovery_runtime.discovery_campaign_lanes_from_args(
        value,
        parse_guidance_targets_func=parse_guidance_targets,
        default_discovery_lane_ids=DEFAULT_DISCOVERY_LANE_IDS,
        discovery_lane_spec_func=discovery_lane_spec,
    )


def _discovery_campaign_lane_catalog() -> dict[str, dict[str, Any]]:
    return _discovery_runtime.discovery_campaign_lane_catalog(
        discovery_lane_catalog_func=discovery_lane_catalog,
    )


def _discovery_campaign_config_from_args(
    args: argparse.Namespace,
    preset: str,
    *,
    lane_discovery_biases: list[dict[str, Any]] | None = None,
    lane_semantic_focus_families: list[str] | None = None,
    lane_semantic_focus_signals: list[str] | None = None,
) -> ExperimentConfig:
    return _discovery_runtime.discovery_campaign_config_from_args(
        args,
        preset,
        preset_config_func=_preset_config,
        parse_guidance_targets_func=parse_guidance_targets,
        merge_discovery_biases_func=merge_discovery_biases,
        lane_discovery_biases=lane_discovery_biases,
        lane_semantic_focus_families=lane_semantic_focus_families,
        lane_semantic_focus_signals=lane_semantic_focus_signals,
    )


def cmd_bug_status(args: argparse.Namespace) -> int:
    return cmd_bug_status_impl(
        args,
        parse_guidance_targets_func=parse_guidance_targets,
        build_bug_status_func=build_bug_status,
        write_bug_status_outputs_func=write_bug_status_outputs,
        project_relative_path_func=_project_relative_cli_path,
    )


def cmd_issue_readiness(args: argparse.Namespace) -> int:
    return cmd_issue_readiness_impl(
        args,
        parse_guidance_targets_func=parse_guidance_targets,
        build_issue_readiness_func=build_issue_readiness,
        write_issue_readiness_outputs_func=write_issue_readiness_outputs,
        project_relative_path_func=_project_relative_cli_path,
    )


def cmd_issue_bundle(args: argparse.Namespace) -> int:
    return cmd_issue_bundle_impl(
        args,
        parse_guidance_targets_func=parse_guidance_targets,
        build_issue_bundle_func=build_issue_bundle,
        default_issue_bundle_statuses=DEFAULT_ISSUE_BUNDLE_STATUSES,
    )


def cmd_candidate_pipeline(args: argparse.Namespace) -> int:
    return cmd_candidate_pipeline_impl(
        args,
        parse_guidance_targets_func=parse_guidance_targets,
        build_candidate_pipeline_func=build_candidate_pipeline,
        default_candidate_pipeline_dir=DEFAULT_CANDIDATE_PIPELINE_DIR,
    )


def _discovery_run_config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    return _discovery_runtime.discovery_run_config_from_args(
        args,
        preset_config_func=_preset_config,
        parse_guidance_targets_func=parse_guidance_targets,
    )


def _project_relative_cli_path(value: str | Path) -> str:
    return _discovery_runtime.project_relative_cli_path(value)


def _write_discovery_run_fresh_candidate_evidence(
    run_file: Path,
    *,
    classification: dict[str, Any],
    output_path: Path,
    refresh: bool = False,
) -> dict[str, Any]:
    return _discovery_runtime.write_discovery_run_fresh_candidate_evidence(
        run_file,
        classification=classification,
        output_path=output_path,
        refresh=refresh,
        read_jsonl_func=read_jsonl,
        candidate_issue_family_key_func=_candidate_issue_family_key,
        project_relative_path_func=_project_relative_cli_path,
        utc_now_func=utc_now,
        dump_json_func=dump_json,
    )


def _run_candidate_pipeline_for_evidence(
    args: argparse.Namespace,
    *,
    evidence_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    return _discovery_runtime.run_candidate_pipeline_for_evidence(
        args,
        evidence_path=evidence_path,
        manifest_path=manifest_path,
        build_candidate_pipeline_func=build_candidate_pipeline,
        default_candidate_pipeline_dir=DEFAULT_CANDIDATE_PIPELINE_DIR,
    )


def cmd_experiment_summary(args: argparse.Namespace) -> int:
    return cmd_experiment_summary_impl(
        args,
        write_experiment_summary_func=write_experiment_summary,
    )


def cmd_analyze_experiment(args: argparse.Namespace) -> int:
    return cmd_analyze_experiment_impl(
        args,
        parse_presets_func=_parse_presets,
        analyze_experiment_func=analyze_experiment,
    )


def cmd_analyze_seeded_sensitivity(args: argparse.Namespace) -> int:
    return cmd_analyze_seeded_sensitivity_impl(
        args,
        analyze_seeded_sensitivity_func=analyze_seeded_sensitivity,
    )


def cmd_analyze_ablation_audit(args: argparse.Namespace) -> int:
    return cmd_analyze_ablation_audit_impl(
        args,
        parse_presets_func=_parse_presets,
        analyze_ablation_audit_func=analyze_ablation_audit,
    )


def cmd_methodology_report(args: argparse.Namespace) -> int:
    return cmd_methodology_report_impl(
        args,
        write_methodology_report_func=write_methodology_report,
        load_json_func=load_json,
    )


def cmd_adaptive_benchmark(args: argparse.Namespace) -> int:
    return cmd_adaptive_benchmark_impl(
        args,
        utc_now_func=utc_now,
        run_adaptive_benchmark_func=run_adaptive_benchmark,
        dump_json_func=dump_json,
        write_adaptive_benchmark_markdown_func=write_adaptive_benchmark_markdown,
        project_relative_path_func=_project_relative_cli_path,
    )


def cmd_final_readiness(args: argparse.Namespace) -> int:
    return cmd_final_readiness_impl(
        args,
        parse_presets_func=_parse_presets,
        parse_adaptive_component_tuple_func=_parse_adaptive_component_tuple,
        manifest_index_files_func=_final_readiness_manifest_index_files,
        dedupe_paths_func=_dedupe_paths,
        readiness_policy_factory=ReadinessPolicy,
        readiness_thresholds_factory=ReadinessThresholds,
        default_readiness_policy=DEFAULT_A_LEVEL_READINESS_POLICY,
        default_manifest_limit=DEFAULT_FINAL_READINESS_MANIFEST_LIMIT,
        analyze_final_readiness_func=analyze_final_readiness,
        load_json_func=load_json,
    )


def cmd_review_readiness(args: argparse.Namespace) -> int:
    return cmd_review_readiness_impl(
        args,
        review_thresholds_factory=ReviewThresholds,
        build_review_readiness_func=build_review_readiness,
        write_review_readiness_outputs_func=write_review_readiness_outputs,
        project_relative_path_func=_project_relative_cli_path,
    )


def cmd_analyze_pattern_variants(args: argparse.Namespace) -> int:
    return cmd_analyze_pattern_variants_impl(
        args,
        analyze_pattern_variants_func=analyze_pattern_variants,
    )


def cmd_targets(args: argparse.Namespace) -> int:
    return cmd_targets_impl(
        args,
        targets=TARGETS,
        list_target_suites_func=list_target_suites,
        target_context_func=target_context,
        target_capability_matrix_func=target_capability_matrix,
    )


def cmd_semantic_registry(args: argparse.Namespace) -> int:
    return cmd_semantic_registry_impl(
        args,
        targets=TARGETS,
        resolve_run_backends_func=_resolve_run_backends,
        parse_objective_rules_func=_parse_exploration_objective_rules,
        experiment_config_factory=ExperimentConfig,
        target_context_func=target_context,
        semantic_registry_payload_func=semantic_registry_payload,
    )


def cmd_target_version_audit(args: argparse.Namespace) -> int:
    return cmd_target_version_audit_impl(
        args,
        parse_guidance_targets_func=parse_guidance_targets,
        parse_latest_version_overrides_func=parse_latest_version_overrides,
        build_target_version_audit_func=build_target_version_audit,
        write_target_version_audit_func=write_target_version_audit,
    )


def cmd_prune_corpus(args: argparse.Namespace) -> int:
    return cmd_prune_corpus_impl(args, corpus_dir=CORPUS_DIR)


def cmd_show_bugs(args: argparse.Namespace) -> int:
    return cmd_show_bugs_impl(
        args,
        latest_run_log_path_func=latest_run_log_path,
        read_jsonl_func=read_jsonl,
    )


def cmd_classify_run(args: argparse.Namespace) -> int:
    return cmd_classify_run_impl(
        args,
        latest_run_log_path_func=latest_run_log_path,
        summarize_run_classification_func=_summarize_run_classification,
    )


def cmd_run_health(args: argparse.Namespace) -> int:
    return cmd_run_health_impl(
        args,
        latest_run_log_path_func=latest_run_log_path,
        summarize_run_health_func=_summarize_run_health,
    )


def cmd_discovery_campaign_status(args: argparse.Namespace) -> int:
    return cmd_discovery_campaign_status_impl(
        args,
        summarize_discovery_campaign_status_func=_summarize_discovery_campaign_status,
        runs_dir=RUNS_DIR,
    )


def cmd_discovery_campaign_aggregate(args: argparse.Namespace) -> int:
    return cmd_discovery_campaign_aggregate_impl(
        args,
        summarize_discovery_campaign_aggregate_func=_summarize_discovery_campaign_aggregate,
        dump_json_func=dump_json,
    )


def cmd_reproduce(args: argparse.Namespace) -> int:
    return cmd_reproduce_impl(
        args,
        case_from_dict_func=Case.from_dict,
        load_json_func=load_json,
        experiment_config_factory=ExperimentConfig,
        parse_backends_func=_parse_backends,
        run_loaded_case_func=run_loaded_case,
    )


def cmd_validate_artifact(args: argparse.Namespace) -> int:
    return cmd_validate_artifact_impl(
        args,
        case_from_dict_func=Case.from_dict,
        load_json_func=load_json,
        experiment_config_factory=ExperimentConfig,
        parse_backends_func=_parse_backends,
        run_loaded_case_func=run_loaded_case,
    )


def cmd_triage_artifact(args: argparse.Namespace) -> int:
    return cmd_triage_artifact_impl(
        args,
        case_from_dict_func=Case.from_dict,
        load_json_func=load_json,
        dump_json_func=dump_json,
        experiment_config_factory=ExperimentConfig,
        parse_backends_func=_parse_backends,
        load_artifact_config_func=_load_artifact_config,
        reduce_case_func=reduce_case,
        run_loaded_case_func=run_loaded_case,
        build_triage_report_func=build_triage_report,
        write_triage_artifact_func=write_triage_artifact,
        supports_standalone_reproducer_func=supports_standalone_reproducer,
        write_standalone_reproducer_func=write_standalone_reproducer,
        write_reduced_reproducer_func=write_reduced_reproducer,
    )


def cmd_reduce(args: argparse.Namespace) -> int:
    return cmd_reduce_impl(
        args,
        case_from_dict_func=Case.from_dict,
        load_json_func=load_json,
        dump_json_func=dump_json,
        experiment_config_factory=ExperimentConfig,
        parse_backends_func=_parse_backends,
        load_artifact_config_func=_load_artifact_config,
        reduce_case_func=reduce_case,
        run_loaded_case_func=run_loaded_case,
        write_reduced_reproducer_func=write_reduced_reproducer,
    )


def cmd_replay_fixture(args: argparse.Namespace) -> int:
    return cmd_replay_fixture_impl(
        args,
        ensure_dirs_func=ensure_dirs,
        load_fixture_replay_spec_func=load_fixture_replay_spec,
        resolve_fixture_replay_path_func=_resolve_fixture_replay_path,
        build_fixture_replay_case_func=build_fixture_replay_case,
        parse_backends_func=_parse_backends,
        resolve_target_backends_func=resolve_target_backends,
        parse_experiment_meta_func=parse_experiment_meta,
        normalize_experiment_meta_func=normalize_experiment_meta,
        replay_bug_enabled_by_default_func=replay_bug_enabled_by_default,
        registered_experiment_meta_defaults_func=registered_experiment_meta_defaults,
        merge_experiment_meta_func=merge_experiment_meta,
        experiment_config_factory=ExperimentConfig,
        parse_adaptive_components_func=_parse_adaptive_components,
        adaptive_component_config_func=_adaptive_component_config,
        run_loaded_case_func=run_loaded_case,
        fixture_artifact_budget_allows_func=_fixture_artifact_budget_allows,
        describe_targets_func=describe_targets,
        fixture_replay_run_id_func=_fixture_replay_run_id,
        runs_dir=RUNS_DIR,
        jsonl_writer_factory=JsonlWriter,
        compact_log_row_func=_compact_log_row,
        target_context_func=target_context,
        experiment_manifest_path_func=_experiment_manifest_path,
        resolved_run_semantics_func=resolved_run_semantics,
        catalog_preset_metadata_func=catalog_preset_metadata,
        configured_guidance_targets_func=_configured_guidance_targets,
        describe_operation_combo_func=describe_operation_combo,
        dump_json_func=dump_json,
        run_meta_path_func=run_meta_path,
        record_run_journal_func=record_run_journal,
        reports_dir=REPORTS_DIR,
        utc_now_func=utc_now,
    )


def cmd_historical_status(args: argparse.Namespace) -> int:
    return cmd_historical_status_impl(
        args,
        list_historical_bugs_func=list_historical_bugs,
        historical_status_row_func=_historical_status_row,
    )


def _historical_status_row(spec: object) -> dict:
    fixture_env = str(getattr(spec, "fixture_env", "") or "")
    fixture_env_value = os.environ.get(fixture_env) if fixture_env else ""
    return {
        "bug_id": str(getattr(spec, "bug_id")),
        "project": str(getattr(spec, "project")),
        "status": str(getattr(spec, "status")),
        "counted": getattr(spec, "status") == "confirmed_fixed",
        "replay_kind": str(getattr(spec, "replay_kind", "experiment")),
        "target_suite": str(getattr(spec, "target_suite")),
        "target_version": str(getattr(spec, "target_version")),
        "fixed_version": str(getattr(spec, "fixed_version", "")),
        "issue_url": str(getattr(spec, "issue_url", "")),
        "default_presets": list(getattr(spec, "default_presets", ())),
        "default_cases": int(getattr(spec, "default_cases", 0)),
        "default_seeds": list(getattr(spec, "default_seeds", ())),
        "expected_root_causes": list(getattr(spec, "expected_root_causes", ())),
        "expected_suspicious_backends": list(getattr(spec, "expected_suspicious_backends", ())),
        "fixture_spec": str(getattr(spec, "fixture_spec", "")),
        "fixture_env": fixture_env,
        "fixture_env_set": bool(fixture_env_value),
        "fixture_env_status": "set" if fixture_env_value else "unset" if fixture_env else "n/a",
        "notes": str(getattr(spec, "notes", "")),
    }


def _resolve_fixture_replay_path(args: argparse.Namespace) -> Path:
    if args.fixture and args.fixture_env:
        raise ValueError("use only one of --fixture or --fixture-env")
    if args.fixture:
        return Path(args.fixture)
    if args.fixture_env:
        value = os.environ.get(args.fixture_env)
        if not value:
            raise ValueError(f"environment variable {args.fixture_env} is not set")
        return Path(value)
    raise ValueError("one of --fixture or --fixture-env is required")


def _fixture_artifact_budget_allows(artifact_limit: int | None) -> bool:
    return artifact_limit is None or int(artifact_limit) > 0


def _fixture_replay_run_id(label: str) -> str:
    ts = utc_now().replace(":", "").replace("-", "").replace("Z", "")
    return f"run-fixture-{slugify(label)}-{ts}-{time.time_ns()}"


def _load_artifact_config(bug_dir: Path) -> dict:
    config_path = bug_dir / "config.json"
    return load_json(config_path) if config_path.exists() else {}


def _parse_seeds(value: str) -> list[int]:
    return _experiment_runtime.parse_seeds(value)


def _parse_presets(value: str) -> list[str]:
    return _experiment_runtime.parse_presets(value)


def _parse_target_suites(value: str | None) -> list[str]:
    return _experiment_runtime.parse_target_suites(value, target_suites=TARGET_SUITES)


def _experiment_target_runs(args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    return _experiment_runtime.experiment_target_runs(
        args,
        resolve_run_backends_func=_resolve_run_backends,
        parse_target_suites_func=_parse_target_suites,
        resolve_target_backends_func=resolve_target_backends,
    )


def _preset_config(name: str) -> ExperimentConfig:
    return _experiment_runtime.preset_config(
        name,
        build_catalog_preset_func=build_catalog_preset,
    )


def _job_experiment_meta(job: dict[str, Any]) -> dict[str, Any]:
    return _experiment_runtime.job_experiment_meta(
        job,
        normalize_experiment_meta_func=normalize_experiment_meta,
    )


def _job_run_semantics(job: dict[str, Any]) -> dict[str, Any]:
    return _experiment_runtime.job_run_semantics(
        job,
        job_experiment_meta_func=_job_experiment_meta,
        resolved_run_semantics_func=resolved_run_semantics,
    )


def _job_config(job: dict[str, Any]) -> ExperimentConfig:
    return _experiment_runtime.job_config(
        job,
        job_run_semantics_func=_job_run_semantics,
        preset_config_func=_preset_config,
        apply_job_config_overrides_func=_apply_job_config_overrides,
        build_catalog_preset_func=build_catalog_preset,
        build_experiment_config_func=build_experiment_config,
    )


def _apply_job_config_overrides(config: ExperimentConfig, job: dict[str, Any]) -> None:
    _experiment_runtime.apply_job_config_overrides(
        config,
        job,
        parse_guidance_targets_func=parse_guidance_targets,
        parse_adaptive_components_func=_parse_adaptive_components,
        apply_adaptive_component_config_func=_apply_adaptive_component_config,
    )


def _populate_job_learning_metadata(job: dict[str, Any]) -> None:
    _experiment_runtime.populate_job_learning_metadata(
        job,
        job_config_func=_job_config,
        objective_feature_func=objective_feature,
    )


def _effective_job_local_source_scheduler(job: dict) -> tuple[bool, float]:
    return _experiment_runtime.effective_job_local_source_scheduler(
        job,
        parse_adaptive_components_func=_parse_adaptive_components,
        job_config_func=_job_config,
    )


def cmd_experiment(args: argparse.Namespace) -> int:
    return cmd_experiment_impl(
        args,
        ensure_dirs_func=ensure_dirs,
        parse_presets_func=_parse_presets,
        parse_seeds_func=_parse_seeds,
        experiment_target_runs_func=_experiment_target_runs,
        parse_duration_func=parse_duration,
        resolve_evidence_mode_func=_resolve_evidence_mode,
        parse_adaptive_components_func=_parse_adaptive_components,
        adaptive_component_config_func=_adaptive_component_config,
        parse_experiment_meta_func=parse_experiment_meta,
        default_experiment_meta_for_runs_func=_default_experiment_meta_for_runs,
        merge_experiment_meta_func=merge_experiment_meta,
        replay_bug_enabled_by_default_func=replay_bug_enabled_by_default,
        parse_guidance_targets_func=parse_guidance_targets,
        default_replay_bug_source_issues=DEFAULT_REPLAY_BUG_SOURCE_ISSUES,
        populate_job_learning_metadata_func=_populate_job_learning_metadata,
        job_config_func=_job_config,
        experiment_job_weight_func=_experiment_job_weight,
        resolve_experiment_parallelism_func=_resolve_experiment_parallelism,
        resolve_experiment_worker_batching_func=_resolve_experiment_worker_batching,
        resolve_experiment_schedule_func=_resolve_experiment_schedule,
        invalid_live_adaptive_experiment_presets_func=_invalid_live_adaptive_experiment_presets,
        effective_job_local_source_scheduler_func=_effective_job_local_source_scheduler,
        target_context_func=target_context,
        utc_now_func=utc_now,
        run_experiment_adaptive_func=_run_experiment_adaptive,
        experiment_manifest_path_func=_experiment_manifest_path,
        dump_json_func=dump_json,
        record_experiment_journal_func=_record_experiment_journal,
        run_experiment_job_func=_run_experiment_job,
        run_experiment_jobs_parallel_func=_run_experiment_jobs_parallel,
        process_pool_executor_cls=ProcessPoolExecutor,
        thread_pool_executor_cls=ThreadPoolExecutor,
    )


def _record_experiment_journal(manifest_path: Path, manifest: dict) -> tuple[Path, Path]:
    return _experiment_runtime.record_experiment_journal(
        manifest_path,
        manifest,
        reports_dir=REPORTS_DIR,
        experiment_run_theme_func=_experiment_run_theme,
        build_run_journal_entry_func=build_run_journal_entry,
        append_run_journal_entries_func=append_run_journal_entries,
        write_run_journal_markdown_func=write_run_journal_markdown,
    )


def _experiment_run_theme(manifest: dict, run: dict) -> str:
    return _experiment_runtime.experiment_run_theme(
        manifest,
        run,
        replay_bug_enabled_by_default_func=replay_bug_enabled_by_default,
    )


def _run_experiment_adaptive(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    planned_runs: list[dict[str, Any]],
    duration_s: float | None,
    *,
    jobs: int,
) -> list[dict[str, Any]]:
    return _experiment_runtime.run_experiment_adaptive(
        args,
        manifest,
        planned_runs,
        duration_s,
        jobs=jobs,
        parse_adaptive_components_func=_parse_adaptive_components,
        adaptive_component_config_func=_adaptive_component_config,
        parse_duration_func=parse_duration,
        adaptive_schedule_config_cls=AdaptiveScheduleConfig,
        adaptive_budget_scheduler_cls=AdaptiveBudgetScheduler,
        adaptive_learning_state_from_ledgers_func=_adaptive_learning_state_from_ledgers,
        run_experiment_job_func=_run_experiment_job,
        adaptive_job_func=_adaptive_job,
        complete_adaptive_round_func=_complete_adaptive_round,
        run_experiment_adaptive_parallel_func=_run_experiment_adaptive_parallel,
        process_pool_executor_cls=ProcessPoolExecutor,
        thread_pool_executor_cls=ThreadPoolExecutor,
    )


def _adaptive_learning_state_from_ledgers(value: Any) -> tuple[AdaptiveLearningState, list[dict[str, Any]]]:
    return _experiment_runtime.adaptive_learning_state_from_ledgers(
        value,
        adaptive_learning_state_cls=AdaptiveLearningState,
        continual_priority_memory_cls=ContinualPriorityMemory,
        parse_guidance_targets_func=parse_guidance_targets,
        load_json_func=load_json,
        validation_error_func=_continual_learning_ledger_validation_error,
    )


def _continual_learning_ledger_validation_error(payload: dict[str, Any]) -> str:
    return _experiment_runtime.continual_learning_ledger_validation_error(payload)


def _resolve_experiment_parallelism(args: argparse.Namespace, planned_runs: list[dict[str, Any]]) -> dict[str, Any]:
    return _experiment_runtime.resolve_experiment_parallelism(
        args,
        planned_runs,
        cpu_count_func=os.cpu_count,
        experiment_job_weight_func=_experiment_job_weight,
    )


def _resolve_experiment_worker_batching(
    args: argparse.Namespace,
    planned_runs: list[dict[str, Any]],
    *,
    jobs: int,
    schedule: str,
) -> dict[str, Any]:
    return _experiment_runtime.resolve_experiment_worker_batching(
        args,
        planned_runs,
        jobs=jobs,
        schedule=schedule,
    )


def _run_experiment_adaptive_parallel(
    executor_cls: type,
    worker_count: int,
    scheduler: AdaptiveBudgetScheduler,
) -> list[dict[str, Any]]:
    return _experiment_runtime.run_experiment_adaptive_parallel(
        executor_cls,
        worker_count,
        scheduler,
        run_experiment_job_func=_run_experiment_job,
        adaptive_job_func=_adaptive_job,
        complete_adaptive_round_func=_complete_adaptive_round,
    )


def _adaptive_job(batch: Any) -> dict[str, Any]:
    return _experiment_runtime.adaptive_job(
        batch,
        parse_adaptive_components_func=_parse_adaptive_components,
    )


def _complete_adaptive_round(
    scheduler: AdaptiveBudgetScheduler,
    round_results: list[dict[str, Any]],
    batches: list[Any],
) -> list[dict[str, Any]]:
    return _experiment_runtime.complete_adaptive_round(
        scheduler,
        round_results,
        batches,
        summarize_batch_run_func=summarize_batch_run,
        run_meta_path_func=run_meta_path,
        load_json_func=load_json,
        load_closed_loop_state_from_meta_func=_load_closed_loop_state_from_meta,
    )


def _load_closed_loop_state_from_meta(meta: dict[str, Any], *, run_file: Path) -> dict[str, Any] | None:
    return _experiment_runtime.load_closed_loop_state_from_meta(
        meta,
        run_file=run_file,
        closed_loop_state_path_func=closed_loop_state_path,
        load_json_func=load_json,
    )


def _resolve_experiment_schedule(args: argparse.Namespace, *, jobs: int) -> str:
    return _experiment_runtime.resolve_experiment_schedule(args, jobs=jobs)


def _resolve_evidence_mode(value: str, suite_names: list[str]) -> str:
    return _experiment_runtime.resolve_evidence_mode(value, suite_names)


def _default_experiment_meta_for_runs(
    *,
    evidence_mode: str,
    target_runs: list[tuple[str, list[str]]],
    presets: list[str],
    known_bug_id: str,
    target_version: str,
    include_pending_historical: bool = True,
) -> dict[str, Any]:
    return _experiment_runtime.default_experiment_meta_for_runs(
        evidence_mode=evidence_mode,
        target_runs=target_runs,
        presets=presets,
        known_bug_id=known_bug_id,
        target_version=target_version,
        include_pending_historical=include_pending_historical,
        registered_experiment_meta_defaults_func=registered_experiment_meta_defaults,
        registered_experiment_matrix_for_run_func=registered_experiment_matrix_for_run,
    )


def _invalid_live_adaptive_experiment_presets(planned_runs: list[dict[str, Any]]) -> list[str]:
    return _experiment_runtime.invalid_live_adaptive_experiment_presets(
        planned_runs,
        job_config_func=_job_config,
    )


def _experiment_manifest_path() -> Path:
    return _experiment_runtime.experiment_manifest_path(
        runs_dir=RUNS_DIR,
        utc_now_func=utc_now,
        time_ns_func=time.time_ns,
    )


def _run_experiment_jobs_parallel(
    executor_cls: type,
    worker_count: int,
    planned_runs: list[dict],
    *,
    max_parallel_cost: float,
    worker_batch_size: int = 0,
    worker_max_rss_kib: int = 0,
    worker_retry_limit: int = 0,
    worker_batch_manifest: dict[str, Any] | None = None,
) -> list[dict]:
    return _experiment_runtime.run_experiment_jobs_parallel(
        executor_cls,
        worker_count,
        planned_runs,
        max_parallel_cost=max_parallel_cost,
        experiment_job_sort_key_func=_experiment_job_sort_key,
        next_schedulable_job_index_func=_next_schedulable_job_index,
        job_estimated_cost_func=_job_estimated_cost,
        run_experiment_job_func=_run_experiment_job,
        run_experiment_job_batch_func=_run_experiment_job_batch,
        worker_batch_size=worker_batch_size,
        worker_max_rss_kib=worker_max_rss_kib,
        worker_retry_limit=worker_retry_limit,
        worker_batch_manifest=worker_batch_manifest,
    )


def _next_schedulable_job_index(queued_runs: list[dict], available_cost: float) -> int | None:
    return _experiment_runtime.next_schedulable_job_index(
        queued_runs,
        available_cost,
        job_estimated_cost_func=_job_estimated_cost,
    )


def _job_estimated_cost(job: dict) -> float:
    return _experiment_runtime.job_estimated_cost(
        job,
        experiment_job_weight_func=_experiment_job_weight,
    )


def _experiment_job_sort_key(job: dict) -> tuple[float, int]:
    return _experiment_runtime.experiment_job_sort_key(
        job,
        experiment_job_weight_func=_experiment_job_weight,
    )


def _experiment_job_weight(job: dict) -> float:
    return _experiment_runtime.experiment_job_weight(
        job,
        job_config_func=_job_config,
        backend_cost_func=_backend_cost,
    )


def _backend_cost(backend: str) -> float:
    return _experiment_runtime.backend_cost(backend)


def _run_experiment_job(job: dict) -> dict:
    previous_thread_limits = _experiment_runtime.capture_native_thread_limits()
    try:
        return _experiment_runtime.run_experiment_job(
            job,
            apply_native_thread_limits_func=_apply_native_thread_limits,
            job_config_func=_job_config,
            parse_adaptive_components_func=_parse_adaptive_components,
            adaptive_component_config_func=_adaptive_component_config,
            apply_adaptive_component_config_func=_apply_adaptive_component_config,
            run_fuzz_func=run_fuzz,
            write_report_func=write_report,
            job_experiment_meta_func=_job_experiment_meta,
            resolved_run_semantics_func=resolved_run_semantics,
            catalog_preset_metadata_func=catalog_preset_metadata,
            configured_guidance_targets_func=_configured_guidance_targets,
        )
    finally:
        _experiment_runtime.restore_native_thread_limits(previous_thread_limits)


def _run_experiment_job_batch(batch: dict[str, Any]) -> dict[str, Any]:
    return _experiment_runtime.run_experiment_job_batch(
        batch,
        run_experiment_job_func=_run_experiment_job,
    )


def _apply_native_thread_limits(thread_limit: int) -> None:
    _experiment_runtime.apply_native_thread_limits(thread_limit)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="datadiff",
        description="Semantic differential fuzzing for DataFrame and embedded analytical engines.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    register_core_commands(
        sub,
        handlers=CoreCommandHandlers(
            init=cmd_init,
            targets=cmd_targets,
            semantic_registry=cmd_semantic_registry,
            target_version_audit=cmd_target_version_audit,
            version_ledger=cmd_version_ledger,
            prune_corpus=cmd_prune_corpus,
        ),
        add_target_suite_flags=add_target_suite_flags,
    )

    register_cross_version_commands(sub)

    register_fuzzing_commands(
        sub,
        handlers=FuzzingCommandHandlers(fuzz=cmd_fuzz, longrun=cmd_longrun),
        profile_choices=PROFILE_CHOICES,
        add_target_suite_flags=add_target_suite_flags,
        add_guidance_flags=add_guidance_flags,
        add_ablation_flags=add_ablation_flags,
        add_paper_journal_flags=add_paper_journal_flags,
    )

    register_reporting_commands(
        sub,
        handlers=ReportingCommandHandlers(
            report=cmd_report,
            bug_audit=cmd_bug_audit,
            bug_status=cmd_bug_status,
            issue_readiness=cmd_issue_readiness,
            issue_bundle=cmd_issue_bundle,
            methodology_report=cmd_methodology_report,
            show_bugs=cmd_show_bugs,
            classify_run=cmd_classify_run,
            run_health=cmd_run_health,
        ),
        list_audit_probe_ids=list_audit_probe_ids,
        default_issue_bundle_statuses=DEFAULT_ISSUE_BUNDLE_STATUSES,
    )

    register_discovery_commands(
        sub,
        handlers=DiscoveryCommandHandlers(
            discovery_run=cmd_discovery_run,
            discovery_campaign=cmd_discovery_campaign,
            discovery_campaign_status=cmd_discovery_campaign_status,
            discovery_campaign_aggregate=cmd_discovery_campaign_aggregate,
            candidate_pipeline=cmd_candidate_pipeline,
        ),
        target_suites=TARGET_SUITES,
        list_audit_probe_ids=list_audit_probe_ids,
        default_discovery_lane_ids=DEFAULT_DISCOVERY_LANE_IDS,
        default_campaign_history_window=DEFAULT_DISCOVERY_CAMPAIGN_HISTORY_WINDOW,
        default_campaign_score_weights=DEFAULT_DISCOVERY_CAMPAIGN_SCORE_WEIGHTS,
        candidate_pipeline_output_dir_default=str(DEFAULT_CANDIDATE_PIPELINE_DIR.relative_to(PROJECT_ROOT)),
    )

    register_analysis_commands(
        sub,
        handlers=AnalysisCommandHandlers(
            experiment_summary=cmd_experiment_summary,
            analyze_experiment=cmd_analyze_experiment,
            analyze_seeded_sensitivity=cmd_analyze_seeded_sensitivity,
            analyze_ablation_audit=cmd_analyze_ablation_audit,
            adaptive_benchmark=cmd_adaptive_benchmark,
            analyze_pattern_variants=cmd_analyze_pattern_variants,
        ),
    )

    register_readiness_commands(
        sub,
        handlers=ReadinessCommandHandlers(
            final_readiness=cmd_final_readiness,
            review_readiness=cmd_review_readiness,
        ),
        default_final_readiness_manifest_limit=DEFAULT_FINAL_READINESS_MANIFEST_LIMIT,
        required_live_suites=DEFAULT_A_LEVEL_READINESS_POLICY.required_live_suites,
        required_live_families=DEFAULT_A_LEVEL_READINESS_POLICY.required_live_families,
        final_readiness_thresholds=DEFAULT_FINAL_READINESS_THRESHOLDS,
    )

    register_artifact_commands(
        sub,
        handlers=ArtifactCommandHandlers(
            reproduce=cmd_reproduce,
            validate_artifact=cmd_validate_artifact,
            triage_artifact=cmd_triage_artifact,
            reduce=cmd_reduce,
            historical_status=cmd_historical_status,
            replay_fixture=cmd_replay_fixture,
        ),
        target_suites=TARGET_SUITES,
        fixture_replay_evidence_modes=FIXTURE_REPLAY_EVIDENCE_MODES,
        add_ablation_flags=add_ablation_flags,
    )

    register_experiment_commands(
        sub,
        handlers=ExperimentCommandHandlers(experiment=cmd_experiment),
        experiment_evidence_modes=EXPERIMENT_EVIDENCE_MODES,
        adaptive_components=ADAPTIVE_COMPONENTS,
        parse_jobs=_parse_jobs,
        parse_adaptive_components=_parse_adaptive_components,
        add_target_suite_flags=add_target_suite_flags,
        add_paper_journal_flags=add_paper_journal_flags,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
