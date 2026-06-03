from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, as_completed, wait
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import get_args

from datadiff.ablation_audit import analyze_ablation_audit
from datadiff.adaptive_learning import AdaptiveLearningState, ContinualPriorityMemory
from datadiff.bug_audit import list_audit_probe_ids, run_probe_audit, write_probe_issue_drafts
from datadiff.candidate_pipeline import DEFAULT_CANDIDATE_PIPELINE_DIR, build_candidate_pipeline
from datadiff.bug_status import build_issue_status, write_issue_status_outputs
from datadiff.classification_oracle import classify_finding
from datadiff.config import (
    DEFAULT_KNOWN_SATURATED_BUG_FAMILIES,
    DEFAULT_REPLAY_BUG_SOURCE_ISSUES,
    DiscoveryBias,
    ExperimentConfig,
    GeneratorProfile,
    merge_discovery_biases,
)
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
from datadiff.exploration_objectives import (
    ExplorationObjectiveRule,
    merge_exploration_objective_rules,
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
from datadiff.methodology_report import write_methodology_report
from datadiff.normalizer import NormalizedResult, normalized_results_from_mapping
from datadiff.guidance import parse_guidance_targets
from datadiff.operation_combo import describe_operation_combo
from datadiff.operation_semantics import operation_names
from datadiff.oracle import evaluate_case
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
from datadiff.reward import backend_group_key, offline_finding_bucket
from datadiff.run_journal import (
    append_run_journal_entries,
    build_run_journal_entry,
    record_run_journal,
    write_run_journal_markdown,
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
    read_jsonl_partial,
    run_meta_path,
    slugify,
    utc_now,
    unique_preserve_order,
)
from datadiff.version_ledger import (
    build_version_ledger,
    observations_from_run_logs,
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
ADAPTIVE_COMPONENTS = (
    "scheduler_learning",
    "profile_learning",
    "semantic_objective_learning",
    "metamorphic_relation_learning",
    "version_pair_learning",
    "profile_capability_filter",
    "mutation_operator_learning",
    "quality_archive",
    "local_source_scheduler",
    "runtime_cost_learning",
    "active_learning",
    "online_reward_model",
    "continual_learning",
    "scheduler_annealing",
)
DEFAULT_FINAL_READINESS_THRESHOLDS = ReadinessThresholds()
DEFAULT_DISCOVERY_CAMPAIGN_HISTORY_WINDOW = 8
DEFAULT_DISCOVERY_CAMPAIGN_SCORE_WEIGHTS = {
    "yield_rate": 1.0,
    "novelty_rate": 0.75,
    "false_positive_penalty": 1.25,
}


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
    text = str(value or "").strip()
    if not text:
        return []
    try:
        payload = load_json(Path(text[1:])) if text.startswith("@") else json.loads(text)
    except Exception as exc:  # noqa: BLE001
        raise argparse.ArgumentTypeError(
            f"invalid exploration objective rule payload: {exc}"
        ) from exc
    if isinstance(payload, dict):
        if isinstance(payload.get("exploration_objective_rules"), list):
            payload = payload["exploration_objective_rules"]
        elif isinstance(payload.get("rules"), list):
            payload = payload["rules"]
        else:
            payload = [payload]
    if not isinstance(payload, list):
        raise argparse.ArgumentTypeError(
            "--exploration-objective-rules must be a JSON object/list or @path"
        )
    try:
        return merge_exploration_objective_rules(payload)
    except Exception as exc:  # noqa: BLE001
        raise argparse.ArgumentTypeError(
            f"invalid exploration objective rule payload: {exc}"
        ) from exc


def _resolve_run_backends(args: argparse.Namespace) -> list[str]:
    return resolve_target_backends(
        getattr(args, "backends", None),
        target_suite=getattr(args, "target_suite", "core"),
    )


def _parse_adaptive_components(value: str | list[str] | tuple[str, ...] | set[str] | None) -> set[str]:
    raw_items: list[str] = []
    if isinstance(value, str):
        raw_items = [item.strip() for item in value.split(",")]
    elif value:
        raw_items = [str(item).strip() for item in value]
    aliases = {
        "scheduler": "scheduler_learning",
        "bandit": "scheduler_learning",
        "contextual_bandit": "scheduler_learning",
        "generator_profile_learning": "profile_learning",
        "semantic_objective": "semantic_objective_learning",
        "semantic_objective_learning": "semantic_objective_learning",
        "objective_learning": "semantic_objective_learning",
        "metamorphic_relation": "metamorphic_relation_learning",
        "metamorphic_relation_learning": "metamorphic_relation_learning",
        "mr_learning": "metamorphic_relation_learning",
        "mr_type_learning": "metamorphic_relation_learning",
        "version_pair": "version_pair_learning",
        "version_pair_learning": "version_pair_learning",
        "cross_version_pair_learning": "version_pair_learning",
        "profile_filter": "profile_capability_filter",
        "capability_filter": "profile_capability_filter",
        "mutation_learning": "mutation_operator_learning",
        "operator_learning": "mutation_operator_learning",
        "map_elites": "quality_archive",
        "quality_diversity": "quality_archive",
        "source_scheduler": "local_source_scheduler",
        "feedback_source_scheduler": "local_source_scheduler",
        "runtime_cost": "runtime_cost_learning",
        "cost_learning": "runtime_cost_learning",
        "cost_aware_learning": "runtime_cost_learning",
        "runtime_cost_penalty": "runtime_cost_learning",
        "active": "active_learning",
        "active_learning": "active_learning",
        "uncertainty_sampling": "active_learning",
        "online_exploration": "active_learning",
        "exploration_memory": "active_learning",
        "reward_model": "online_reward_model",
        "online_reward": "online_reward_model",
        "online_reward_model": "online_reward_model",
        "statistical_reward_model": "online_reward_model",
        "continual": "continual_learning",
        "continual_learning": "continual_learning",
        "cross_version_learning": "continual_learning",
        "version_transfer": "continual_learning",
        "annealing": "scheduler_annealing",
        "scheduler_annealing": "scheduler_annealing",
        "simulated_annealing": "scheduler_annealing",
        "annealed_scheduler": "scheduler_annealing",
    }
    components: set[str] = set()
    for raw in raw_items:
        item = raw.strip().lower().replace("-", "_")
        if not item or item in {"none", "off", "false", "0"}:
            continue
        resolved = aliases.get(item, item)
        if resolved not in ADAPTIVE_COMPONENTS:
            allowed = ",".join(ADAPTIVE_COMPONENTS)
            raise argparse.ArgumentTypeError(
                f"unknown adaptive component '{raw}'; expected one of: {allowed}"
            )
        components.add(resolved)
    return components


def _parse_adaptive_component_tuple(value: str | list[str] | tuple[str, ...] | set[str] | None) -> tuple[str, ...]:
    raw_items: list[str] = []
    if isinstance(value, str):
        raw_items = [item.strip() for item in value.split(",")]
    elif value:
        for item in value:
            raw_items.extend(part.strip() for part in str(item).split(","))
    ordered: list[str] = []
    seen: set[str] = set()
    for raw in raw_items:
        for component in _parse_adaptive_components(raw):
            if component not in seen:
                ordered.append(component)
                seen.add(component)
    return tuple(ordered)


def _adaptive_component_config(disabled_components: set[str]) -> dict[str, bool]:
    return {component: component not in disabled_components for component in ADAPTIVE_COMPONENTS}


def _config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    disabled_adaptive_components = _parse_adaptive_components(
        getattr(args, "disable_adaptive_components", "")
    )
    return ExperimentConfig(
        enable_type_aware_generation=not args.disable_type_aware_generation,
        enable_normalizer=not args.disable_normalizer,
        enable_differential_oracle=not args.disable_differential_oracle,
        enable_metamorphic_oracle=args.enable_metamorphic_oracle,
        enable_feedback=not args.disable_feedback,
        enable_replay_bug=bool(getattr(args, "enable_replay_bug", False)),
        enable_reducer=args.enable_reducer,
        enable_artifact=not args.disable_artifact,
        enable_preflight_validation=not args.disable_preflight_validation,
        enable_preflight_repair=not args.disable_preflight_repair,
        persist_feedback_corpus=args.persist_feedback_corpus,
        feedback_persist_limit=max(0, int(getattr(args, "feedback_persist_limit", 4096))),
        enable_local_source_scheduler=bool(getattr(args, "enable_local_source_scheduler", False)),
        local_source_exploration_weight=max(
            0.0, float(getattr(args, "local_source_exploration_weight", 0.5))
        ),
        compress_run_log=not args.no_compress_run_log,
        artifact_limit=args.artifact_limit,
        oracle_mode="both" if args.enable_metamorphic_oracle else "differential",
        generator_profile=args.profile,
        generator_profile_pool=parse_guidance_targets(getattr(args, "profile_pool", "")),
        version_pair_pool=parse_guidance_targets(getattr(args, "version_pair_pool", "")),
        generator_profile_learning_weight=max(
            0.0,
            float(getattr(args, "profile_learning_weight", 0.0) or 0.0),
        ),
        semantic_objective_learning_weight=max(
            0.0,
            float(getattr(args, "semantic_objective_learning_weight", 0.0) or 0.0),
        ),
        metamorphic_relation_learning_weight=max(
            0.0,
            float(getattr(args, "metamorphic_relation_learning_weight", 0.0) or 0.0),
        ),
        version_pair_learning_weight=max(
            0.0,
            float(getattr(args, "version_pair_learning_weight", 0.0) or 0.0),
        ),
        enable_generator_profile_learning="profile_learning" not in disabled_adaptive_components,
        enable_semantic_objective_learning="semantic_objective_learning" not in disabled_adaptive_components,
        enable_metamorphic_relation_learning="metamorphic_relation_learning" not in disabled_adaptive_components,
        enable_profile_capability_filter="profile_capability_filter" not in disabled_adaptive_components,
        enable_mutation_operator_learning="mutation_operator_learning" not in disabled_adaptive_components,
        enable_quality_archive="quality_archive" not in disabled_adaptive_components,
        guidance_strategy=getattr(args, "strategy", "random"),
        guidance_candidate_pool=max(1, int(getattr(args, "candidate_pool", 1))),
        guidance_targets=parse_guidance_targets(getattr(args, "targets", "")),
        exploration_objective_rules=_parse_exploration_objective_rules(
            getattr(args, "exploration_objective_rules", "")
        ),
        enable_family_saturation=not bool(getattr(args, "disable_family_saturation", False)),
        family_saturation_threshold=max(1, int(getattr(args, "family_saturation_threshold", 8))),
        family_saturation_penalty=max(0.0, float(getattr(args, "family_saturation_penalty", 1.25))),
        saturated_family_reward=max(0.0, float(getattr(args, "saturated_family_reward", 0.02))),
        known_saturated_bug_families=parse_guidance_targets(
            getattr(args, "known_saturated_bug_families", "")
        ),
        replay_bug_source_issues=(
            parse_guidance_targets(getattr(args, "replay_bug_source_issues", ""))
            or list(DEFAULT_REPLAY_BUG_SOURCE_ISSUES)
        ),
        issue_replay_saturation_threshold=max(1, int(getattr(args, "issue_replay_saturation_threshold", 1))),
        issue_replay_saturation_penalty=max(0.0, float(getattr(args, "issue_replay_saturation_penalty", 1.0))),
        issue_replay_global_saturation_threshold=max(
            1,
            int(getattr(args, "issue_replay_global_saturation_threshold", 4)),
        ),
        issue_replay_global_saturation_penalty=max(
            0.0,
            float(getattr(args, "issue_replay_global_saturation_penalty", 1.5)),
        ),
        issue_inspired_source_saturation_threshold=max(
            1,
            int(getattr(args, "issue_inspired_source_saturation_threshold", 3)),
        ),
        issue_inspired_source_saturation_penalty=max(
            0.0,
            float(getattr(args, "issue_inspired_source_saturation_penalty", 1.25)),
        ),
        candidate_recheck_count=max(0, int(getattr(args, "candidate_recheck_count", 0))),
        metamorphic_variant_limit=max(0, int(getattr(args, "metamorphic_variant_limit", 4))),
        metamorphic_relation_order=parse_guidance_targets(getattr(args, "metamorphic_relation_order", "")),
        target_version=str(getattr(args, "target_version", "") or ""),
        fixed_version=str(getattr(args, "fixed_version", "") or ""),
        log_level=getattr(args, "log_level", "compact"),
        strategy_snapshot_path=str(getattr(args, "strategy_snapshot", "") or ""),
        strategy_learning_path=str(getattr(args, "strategy_learning", "") or ""),
        freeze_strategy_snapshot=bool(getattr(args, "freeze_strategy_snapshot", False)),
    )


def _apply_adaptive_component_config(config: ExperimentConfig, disabled_components: set[str]) -> None:
    config.enable_generator_profile_learning = "profile_learning" not in disabled_components
    config.enable_semantic_objective_learning = "semantic_objective_learning" not in disabled_components
    config.enable_metamorphic_relation_learning = "metamorphic_relation_learning" not in disabled_components
    config.enable_profile_capability_filter = "profile_capability_filter" not in disabled_components
    config.enable_mutation_operator_learning = "mutation_operator_learning" not in disabled_components
    config.enable_quality_archive = "quality_archive" not in disabled_components
    if "profile_learning" in disabled_components:
        config.generator_profile_learning_weight = 0.0
    if "semantic_objective_learning" in disabled_components:
        config.semantic_objective_learning_weight = 0.0
    if "metamorphic_relation_learning" in disabled_components:
        config.metamorphic_relation_learning_weight = 0.0
    if "version_pair_learning" in disabled_components:
        config.version_pair_learning_weight = 0.0
    if "local_source_scheduler" in disabled_components:
        config.enable_local_source_scheduler = False


def add_ablation_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--disable-type-aware-generation", action="store_true")
    parser.add_argument("--disable-normalizer", action="store_true")
    parser.add_argument("--disable-differential-oracle", action="store_true")
    parser.add_argument("--enable-metamorphic-oracle", action="store_true")
    parser.add_argument("--disable-feedback", action="store_true")
    parser.add_argument(
        "--enable-replay-bug",
        action="store_true",
        help="allow known issue-replay cases; default fresh mode filters submitted or replay-only bug targets",
    )
    parser.add_argument("--enable-reducer", action="store_true")
    parser.add_argument("--disable-artifact", action="store_true")
    parser.add_argument("--disable-preflight-validation", action="store_true")
    parser.add_argument("--disable-preflight-repair", action="store_true")
    parser.add_argument("--persist-feedback-corpus", action="store_true")
    parser.add_argument(
        "--feedback-persist-limit",
        type=int,
        default=4096,
        help="maximum interesting feedback cases to write to corpus/interesting for this run",
    )
    parser.add_argument(
        "--enable-local-source-scheduler",
        action="store_true",
        help="adaptively choose between generated candidates and feedback mutations within a run",
    )
    parser.add_argument(
        "--local-source-exploration-weight",
        type=float,
        default=0.5,
        help="exploration weight for the within-run generated-vs-feedback source scheduler",
    )
    parser.add_argument(
        "--disable-adaptive-components",
        type=_parse_adaptive_components,
        default="",
        help=(
            "comma-separated adaptive components to disable for ablation: "
            + ",".join(ADAPTIVE_COMPONENTS)
        ),
    )
    parser.add_argument("--no-compress-run-log", action="store_true")
    parser.add_argument(
        "--artifact-limit",
        type=int,
        default=None,
        help="maximum bug artifact directories to write for this run; 0 keeps only run-log finding summaries",
    )
    parser.add_argument(
        "--metamorphic-variant-limit",
        type=int,
        default=4,
        help="maximum metamorphic variants to execute per base case",
    )
    parser.add_argument(
        "--metamorphic-relation-order",
        default="",
        help="comma-separated MR relation priority order used before adaptive per-case MR learning",
    )
    parser.add_argument(
        "--semantic-objective-learning-weight",
        type=float,
        default=0.0,
        help="per-case semantic objective contextual-learning weight; 0 keeps objective selection observational",
    )
    parser.add_argument(
        "--metamorphic-relation-learning-weight",
        type=float,
        default=0.0,
        help="per-case MR type contextual-learning weight; 0 keeps configured MR order",
    )
    parser.add_argument(
        "--version-pair-learning-weight",
        type=float,
        default=0.0,
        help="per-case target-version-pair contextual-learning weight; 0 records no version-pair arm feedback",
    )
    parser.add_argument(
        "--candidate-recheck-count",
        type=int,
        default=0,
        help="rerun finding cases this many times and mark non-reproduced findings as false positives",
    )
    parser.add_argument(
        "--log-level",
        choices=["full", "compact", "minimal"],
        default="compact",
        help="run JSONL detail level; compact keeps full details only for finding rows",
    )
    parser.add_argument(
        "--strategy-snapshot",
        default="",
        help="path to a frozen dynamic strategy snapshot used by classification/reproduction logic",
    )
    parser.add_argument(
        "--strategy-learning",
        default="",
        help="path to a strategy-learning ledger used for evidence-driven updates outside frozen runs",
    )
    parser.add_argument(
        "--exploration-objective-rules",
        default="",
        help=(
            "JSON or @path defining neutral exploration objective rules; "
            "each rule has objective, exact_features, prefix_features, and fragments"
        ),
    )
    parser.add_argument(
        "--freeze-strategy-snapshot",
        action="store_true",
        help="treat the configured strategy snapshot as frozen and disable runtime learning drift for final runs",
    )


def add_guidance_flags(
    parser: argparse.ArgumentParser,
    *,
    default_strategy: str,
    default_candidate_pool: int,
) -> None:
    parser.add_argument("--strategy", choices=["random", "guided"], default=default_strategy)
    parser.add_argument(
        "--candidate-pool",
        type=int,
        default=default_candidate_pool,
        help="number of cheap generated candidates scored before executing one case",
    )
    parser.add_argument(
        "--targets",
        default="",
        help=(
            "comma-separated guided targets such as groupby,filter,mutate,sort_limit,"
            "nulls,strings,numeric,edge_float,aggregation,join,running_sum,sortedness,expressions,casts"
        ),
    )
    parser.add_argument(
        "--disable-family-saturation",
        action="store_true",
        help="disable repeated candidate bug family downweighting in guided selection and online rewards",
    )
    parser.add_argument(
        "--family-saturation-threshold",
        type=int,
        default=8,
        help="candidate bug family hit count where guidance starts treating the family as saturated",
    )
    parser.add_argument(
        "--family-saturation-penalty",
        type=float,
        default=1.25,
        help="score penalty scale for predicted cases in saturated candidate bug families",
    )
    parser.add_argument(
        "--saturated-family-reward",
        type=float,
        default=0.02,
        help="online reward assigned to a candidate bug family after saturation",
    )
    parser.add_argument(
        "--known-saturated-bug-families",
        default="",
        help="comma-separated root@backend families already considered saturated before this run",
    )
    parser.add_argument(
        "--replay-bug-source-issues",
        default="",
        help="comma-separated upstream issue URLs treated as known replay bugs in fresh mode",
    )
    parser.add_argument(
        "--issue-replay-saturation-threshold",
        type=int,
        default=1,
        help="issue-replay family hit count where guidance starts downweighting repeated replay probes",
    )
    parser.add_argument(
        "--issue-replay-saturation-penalty",
        type=float,
        default=1.0,
        help="score penalty scale for predicted cases in saturated issue-replay families",
    )
    parser.add_argument(
        "--issue-replay-global-saturation-threshold",
        type=int,
        default=4,
        help="total issue-replay candidate bug count where guidance starts downweighting replay probes",
    )
    parser.add_argument(
        "--issue-replay-global-saturation-penalty",
        type=float,
        default=1.5,
        help="score penalty scale for replay probes after the global replay budget is saturated",
    )
    parser.add_argument(
        "--issue-inspired-source-saturation-threshold",
        type=int,
        default=3,
        help="candidate bug count per source issue where guidance starts downweighting issue-inspired cases",
    )
    parser.add_argument(
        "--issue-inspired-source-saturation-penalty",
        type=float,
        default=1.25,
        help="score penalty scale for issue-inspired cases after their source issue is saturated",
    )


def add_target_suite_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--target-suite",
        choices=sorted(TARGET_SUITES),
        default="core",
        help="backend target suite to execute when --backends is not provided",
    )
    parser.add_argument(
        "--backends",
        default=None,
        help="explicit comma-separated backend targets; overrides --target-suite",
    )


def add_paper_journal_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--run-theme",
        default="",
        help="paper-facing run theme recorded in reports/paper-run-journal.*",
    )
    parser.add_argument(
        "--paper-notes",
        default="",
        help="short paper-facing notes recorded with the run journal entry",
    )
    parser.add_argument(
        "--skip-paper-journal",
        action="store_true",
        help="skip paper-run journal writes for IO-sensitive long runs; summarize later from the manifest/run log",
    )


def cmd_init(args: argparse.Namespace) -> int:
    ensure_dirs()
    print("Initialized DataDiffFuzz")
    print(f"runs:    {RUNS_DIR}")
    print(f"bugs:    {BUGS_DIR}")
    print(f"reports: {REPORTS_DIR}")
    return 0


def cmd_fuzz(args: argparse.Namespace) -> int:
    out = run_fuzz(
        cases=args.cases,
        seed=args.seed,
        backends=_resolve_run_backends(args),
        config=_config_from_args(args),
        duration_s=parse_duration(args.duration),
    )
    print(f"run log written: {out}")
    if not getattr(args, "skip_paper_journal", False):
        journal_path, journal_md = _record_cli_run_journal(out, args, command="fuzz")
        print(f"paper run journal: {journal_path}")
        if journal_md is not None:
            print(f"paper run journal markdown: {journal_md}")
    return 0


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
    case_log_file = Path(args.case_log) if args.case_log else None
    out = run_fuzz(
        cases=args.cases,
        seed=args.seed,
        backends=_resolve_run_backends(args),
        config=_config_from_args(args),
        duration_s=parse_duration(args.duration),
        save_cases=args.save_cases and not args.no_save_cases,
        case_log_file=case_log_file,
        checkpoint_interval_s=parse_duration(args.checkpoint_interval),
        progress_interval_s=parse_duration(args.progress_interval),
        progress_callback=_print_longrun_progress if not args.quiet else None,
    )
    print(f"run log written: {out}")
    if not getattr(args, "skip_paper_journal", False):
        journal_path, journal_md = _record_cli_run_journal(out, args, command="longrun")
        print(f"paper run journal: {journal_path}")
        if journal_md is not None:
            print(f"paper run journal markdown: {journal_md}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    run_file = Path(args.run_file) if args.run_file else latest_run_log_path()
    md_path, csv_path = write_report(run_file, csv_limit=args.csv_limit)
    print(f"markdown report: {md_path}")
    print(f"csv findings:    {csv_path}")
    return 0


def cmd_bug_audit(args: argparse.Namespace) -> int:
    probe_ids = parse_guidance_targets(getattr(args, "probes", "") or "")
    run = run_probe_audit(probe_ids=probe_ids or None)
    if getattr(args, "write_issues", False):
        issue_paths = write_probe_issue_drafts(
            run,
            issue_dir=Path(getattr(args, "issue_dir", "new_issue/generated")),
            overwrite=bool(getattr(args, "overwrite_issues", False)),
        )
    else:
        issue_paths = []
    print(f"bug audit json:     {run.output_json}")
    print(f"bug audit markdown: {run.output_markdown}")
    if issue_paths:
        print("issue drafts:")
        for path in issue_paths:
            print(f"- {path}")
    if run.candidate_bug_families:
        print("candidate bug families:")
        for family in run.candidate_bug_families:
            print(f"- {family}")
    else:
        print("candidate bug families: none")
    if getattr(args, "fail_on_candidate", False) and run.candidate_bug_families:
        return 2
    return 0


def cmd_discovery_run(args: argparse.Namespace) -> int:
    ensure_dirs()
    backends = _resolve_run_backends(args)
    config = _discovery_run_config_from_args(args)
    probe_ids = parse_guidance_targets(getattr(args, "probes", "") or "")
    audit_summary: dict[str, Any] = {"skipped": True}

    if not getattr(args, "skip_bug_audit", False):
        audit_run = run_probe_audit(probe_ids=probe_ids or None)
        issue_paths: list[Path] = []
        if getattr(args, "write_issues", True):
            issue_paths = write_probe_issue_drafts(
                audit_run,
                issue_dir=Path(getattr(args, "issue_dir", "new_issue/generated")),
                overwrite=bool(getattr(args, "overwrite_issues", True)),
            )
        audit_summary = {
            "skipped": False,
            "generated_at": audit_run.generated_at,
            "output_json": _project_relative_cli_path(audit_run.output_json),
            "output_markdown": _project_relative_cli_path(audit_run.output_markdown),
            "candidate_bug_families": list(audit_run.candidate_bug_families),
            "issue_files": [_project_relative_cli_path(path) for path in issue_paths],
            "environment": dict(audit_run.environment),
            "results": list(audit_run.results),
        }

    run_file = run_fuzz(
        cases=args.cases,
        seed=args.seed,
        backends=backends,
        config=config,
        duration_s=parse_duration(args.duration),
    )
    if getattr(args, "skip_run_report", False):
        md_path = csv_path = None
    else:
        md_path, csv_path = write_report(run_file)
    classification = _summarize_run_classification(
        run_file,
        limit=max(0, int(getattr(args, "classify_limit", 3))),
        refresh=bool(getattr(args, "refresh_classification", False)),
    )
    manifest_path = Path(getattr(args, "output_manifest", "") or "new_issue/generated/discovery-run-manifest.json")
    fresh_evidence_path = manifest_path.with_name(f"{manifest_path.stem}-fresh-candidates.json")
    fresh_evidence = _write_discovery_run_fresh_candidate_evidence(
        run_file,
        classification=classification,
        output_path=fresh_evidence_path,
    )
    candidate_pipeline = {}
    if fresh_evidence.get("candidate_row_count", 0):
        candidate_pipeline = _run_candidate_pipeline_for_evidence(
            args,
            evidence_path=fresh_evidence_path,
            manifest_path=manifest_path,
        )
    manifest = {
        "schema_version": "discovery-run-v1",
        "generated_at": utc_now(),
        "generated_by": "datadiff discovery-run",
        "workflow": [
            "deterministic_bug_audit",
            "fresh_guided_fuzz",
            "run_report",
            "candidate_classification",
            "candidate_pipeline",
        ],
        "target_suite": getattr(args, "target_suite", "latest_all_engines"),
        "backends": backends,
        "preset": str(getattr(args, "preset", "live_deep_organic")),
        "cases": args.cases,
        "duration": args.duration,
        "seed": args.seed,
        "config": config.to_dict(),
        "bug_audit": audit_summary,
        "fuzz_run": {
            "run_file": _project_relative_cli_path(run_file),
            "report": _project_relative_cli_path(md_path) if md_path is not None else "",
            "csv": _project_relative_cli_path(csv_path) if csv_path is not None else "",
            "fresh_candidate_evidence": _project_relative_cli_path(fresh_evidence_path),
            "fresh_candidate_evidence_rows": fresh_evidence["candidate_row_count"],
            "candidate_pipeline": candidate_pipeline,
        },
        "classification": classification,
        "candidate_pipeline": candidate_pipeline,
    }
    dump_json(manifest, manifest_path)

    print(f"discovery run manifest: {manifest_path}")
    print(f"run log written:   {run_file}")
    if not getattr(args, "skip_run_report", False):
        print(f"markdown report:   {md_path}")
        print(f"csv findings:      {csv_path}")
    if not audit_summary.get("skipped"):
        print("audit candidate families:")
        families = audit_summary.get("candidate_bug_families", [])
        if families:
            for family in families:
                print(f"- {family}")
        else:
            print("- none")
    print("fresh fuzz candidate families:")
    fresh = classification.get("fresh_candidate_bug_families", {})
    if fresh:
        for family, count in fresh.items():
            print(f"- {family}: {count}")
    else:
        print("- none")
    if candidate_pipeline:
        print(f"candidate pipeline: {candidate_pipeline.get('manifest_path', candidate_pipeline.get('status', ''))}")
    print("issue-inspired unsaturated candidate families:")
    inspired = classification.get("issue_inspired_unsaturated_candidate_bug_families", {})
    if inspired:
        for family, count in inspired.items():
            print(f"- {family}: {count}")
    else:
        print("- none")
    print("known saturated fuzz candidate families:")
    known = classification.get("known_saturated_candidate_bug_families", {})
    if known:
        for family, count in known.items():
            print(f"- {family}: {count}")
    else:
        print("- none")
    if getattr(args, "fail_on_fresh_candidate", False) and fresh:
        return 2
    return 0


def _discovery_campaign_score_weights_from_args(args: argparse.Namespace) -> dict[str, float]:
    return {
        "yield_rate": max(0.0, float(getattr(args, "lane_yield_weight", DEFAULT_DISCOVERY_CAMPAIGN_SCORE_WEIGHTS["yield_rate"]))),
        "novelty_rate": max(0.0, float(getattr(args, "lane_novelty_weight", DEFAULT_DISCOVERY_CAMPAIGN_SCORE_WEIGHTS["novelty_rate"]))),
        "false_positive_penalty": max(
            0.0,
            float(
                getattr(
                    args,
                    "lane_false_positive_penalty",
                    DEFAULT_DISCOVERY_CAMPAIGN_SCORE_WEIGHTS["false_positive_penalty"],
                )
            ),
        ),
    }


def _load_discovery_campaign_history_manifests(
    generated_issue_dir: Path,
    *,
    exclude_manifest: Path | None = None,
    limit: int = DEFAULT_DISCOVERY_CAMPAIGN_HISTORY_WINDOW,
) -> list[dict[str, Any]]:
    if limit <= 0 or not generated_issue_dir.is_dir():
        return []
    exclude = exclude_manifest.resolve() if exclude_manifest is not None and exclude_manifest.exists() else None
    paths = sorted(
        [*generated_issue_dir.glob("discovery-campaign*-manifest.json")],
        key=lambda path: (path.stat().st_mtime, path.name),
        reverse=True,
    )
    manifests: list[dict[str, Any]] = []
    for path in paths:
        if exclude is not None and path.resolve() == exclude:
            continue
        data = load_json(path)
        if isinstance(data, dict):
            manifests.append(data)
        if len(manifests) >= limit:
            break
    return manifests


def _discovery_campaign_scheduler_snapshot(
    *,
    selected_lanes: list[dict[str, str]],
    runs: list[dict[str, Any]],
    pending_by_lane: dict[str, list[int]],
    generated_issue_dir: Path,
    manifest_path: Path,
    history_limit: int,
    score_weights: dict[str, float],
) -> dict[str, Any]:
    history_manifests = _load_discovery_campaign_history_manifests(
        generated_issue_dir,
        exclude_manifest=manifest_path,
        limit=history_limit,
    )
    lane_rows = _discovery_campaign_lane_rows(
        selected_lanes,
        history_manifests=history_manifests,
        current_runs=runs,
        pending_by_lane=pending_by_lane,
        score_weights=score_weights,
    )
    return {
        "strategy": "adaptive_lane_yield_novelty_false_positive_weighting",
        "generated_at": utc_now(),
        "history_manifest_count": len(history_manifests),
        "history_window": history_limit,
        "score_weights": score_weights,
        "lanes": lane_rows,
    }


def _discovery_campaign_lane_rows(
    selected_lanes: list[dict[str, str]],
    *,
    history_manifests: list[dict[str, Any]],
    current_runs: list[dict[str, Any]],
    pending_by_lane: dict[str, list[int]],
    score_weights: dict[str, float],
) -> list[dict[str, Any]]:
    lane_ids = [lane["id"] for lane in selected_lanes]
    metrics = {
        lane["id"]: {
            "lane_id": lane["id"],
            "theme": lane["theme"],
            "target_suite": lane["target_suite"],
            "preset": lane["preset"],
            "completed_runs": 0,
            "history_runs": 0,
            "current_runs": 0,
            "fresh_candidate_total": 0,
            "issue_inspired_total": 0,
            "known_saturated_total": 0,
            "candidate_total": 0,
            "false_positive_total": 0,
            "unique_fresh_families": set(),
            "first_seen_families": set(),
        }
        for lane in selected_lanes
    }
    events: list[dict[str, Any]] = []

    def ingest_run(run: dict[str, Any], *, source: str, fallback_timestamp: str) -> None:
        lane_id = str(run.get("lane_id", ""))
        if lane_id not in metrics or str(run.get("status", "")) != "completed":
            return
        classification = run.get("classification", {}) if isinstance(run.get("classification"), dict) else {}
        fresh = Counter(classification.get("fresh_candidate_bug_families", {}) or {})
        issue_inspired = Counter(classification.get("issue_inspired_unsaturated_candidate_bug_families", {}) or {})
        known = Counter(classification.get("known_saturated_candidate_bug_families", {}) or {})
        false_positive = Counter(classification.get("false_positive_reasons", {}) or {})
        candidate_total = sum((classification.get("candidate_bug_families", {}) or {}).values())
        if not candidate_total:
            candidate_total = sum(fresh.values()) + sum(issue_inspired.values()) + sum(known.values())
        row = metrics[lane_id]
        row["completed_runs"] += 1
        row[f"{source}_runs"] += 1
        row["fresh_candidate_total"] += sum(fresh.values())
        row["issue_inspired_total"] += sum(issue_inspired.values())
        row["known_saturated_total"] += sum(known.values())
        row["candidate_total"] += candidate_total
        row["false_positive_total"] += sum(false_positive.values())
        row["unique_fresh_families"].update(fresh)
        events.append(
            {
                "lane_id": lane_id,
                "timestamp": str(run.get("completed_at", "") or run.get("started_at", "") or fallback_timestamp),
                "families": sorted(fresh),
            }
        )

    for manifest in history_manifests:
        fallback_timestamp = str(manifest.get("generated_at", "") or manifest.get("completed_at", ""))
        for run in manifest.get("runs", []) or []:
            if isinstance(run, dict):
                ingest_run(run, source="history", fallback_timestamp=fallback_timestamp)
    for run in current_runs:
        ingest_run(run, source="current", fallback_timestamp=utc_now())

    seen_families: set[str] = set()
    for event in sorted(
        events,
        key=lambda item: (
            _parse_manifest_utc_timestamp(item.get("timestamp", "")) or datetime.min.replace(tzinfo=timezone.utc),
            lane_ids.index(item["lane_id"]) if item["lane_id"] in lane_ids else len(lane_ids),
        ),
    ):
        for family in event["families"]:
            if family in seen_families:
                continue
            seen_families.add(family)
            metrics[event["lane_id"]]["first_seen_families"].add(family)

    rows: list[dict[str, Any]] = []
    for lane in selected_lanes:
        row = metrics[lane["id"]]
        completed_runs = int(row["completed_runs"])
        unique_fresh_family_count = len(row["unique_fresh_families"])
        first_seen_family_count = len(row["first_seen_families"])
        yield_rate = min(3.0, row["fresh_candidate_total"] / completed_runs) if completed_runs else 0.0
        novelty_rate = (
            first_seen_family_count / unique_fresh_family_count if unique_fresh_family_count else 0.0
        )
        false_positive_rate = (
            row["false_positive_total"] / (row["candidate_total"] + row["false_positive_total"])
            if (row["candidate_total"] + row["false_positive_total"]) > 0
            else 0.0
        )
        score = max(
            0.1,
            1.0
            + score_weights["yield_rate"] * yield_rate
            + score_weights["novelty_rate"] * novelty_rate
            - score_weights["false_positive_penalty"] * false_positive_rate,
        )
        rows.append(
            {
                "lane_id": lane["id"],
                "theme": lane["theme"],
                "target_suite": lane["target_suite"],
                "preset": lane["preset"],
                "pending_runs": len(pending_by_lane.get(lane["id"], [])),
                "completed_runs": completed_runs,
                "history_runs": int(row["history_runs"]),
                "current_runs": int(row["current_runs"]),
                "fresh_candidate_total": int(row["fresh_candidate_total"]),
                "unique_fresh_family_count": unique_fresh_family_count,
                "first_seen_family_count": first_seen_family_count,
                "issue_inspired_total": int(row["issue_inspired_total"]),
                "known_saturated_total": int(row["known_saturated_total"]),
                "candidate_total": int(row["candidate_total"]),
                "false_positive_total": int(row["false_positive_total"]),
                "yield_rate": yield_rate,
                "novelty_rate": novelty_rate,
                "false_positive_rate": false_positive_rate,
                "score": score,
            }
        )

    avg_score = sum(row["score"] for row in rows) / len(rows) if rows else 1.0
    ranked = sorted(
        rows,
        key=lambda row: (
            row["pending_runs"] <= 0,
            -row["score"],
            -row["novelty_rate"],
            -row["yield_rate"],
            row["false_positive_rate"],
            lane_ids.index(row["lane_id"]) if row["lane_id"] in lane_ids else len(lane_ids),
        ),
    )
    for index, row in enumerate(ranked, start=1):
        row["priority_rank"] = index
        row["budget_multiplier"] = max(0.5, min(2.5, row["score"] / avg_score if avg_score else 1.0))
    return ranked


def _next_discovery_campaign_lane(
    scheduler: dict[str, Any],
    pending_by_lane: dict[str, list[int]],
) -> dict[str, Any] | None:
    for lane in scheduler.get("lanes", []) or []:
        lane_id = str(lane.get("lane_id", ""))
        if pending_by_lane.get(lane_id):
            return lane
    return None


def _aggregate_candidate_pipeline_summary(items: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate: Counter[str] = Counter()
    pipeline_count = 0
    for item in items:
        summary = item.get("summary", {}) if isinstance(item.get("summary"), dict) else {}
        if not summary:
            continue
        pipeline_count += 1
        aggregate.update(
            {
                "candidate_count": int(summary.get("candidate_count", 0) or 0),
                "rechecked_count": int(summary.get("rechecked_count", 0) or 0),
                "reproduced_count": int(summary.get("reproduced_count", 0) or 0),
                "reduced_count": int(summary.get("reduced_count", 0) or 0),
                "candidate_bug_verdict_count": int(summary.get("candidate_bug_verdict_count", 0) or 0),
                "issue_draft_count": int(summary.get("issue_draft_count", 0) or 0),
                "needs_dedup_check_count": int(summary.get("needs_dedup_check_count", 0) or 0),
                "already_submitted_or_confirmed_count": int(
                    summary.get("already_submitted_or_confirmed_count", 0) or 0
                ),
            }
        )
    return {"pipeline_count": pipeline_count, **dict(aggregate)}


def cmd_discovery_campaign(args: argparse.Namespace) -> int:
    if getattr(args, "list_lanes", False):
        catalog = _discovery_campaign_lane_catalog()
        if getattr(args, "json", False):
            print(json.dumps(catalog, ensure_ascii=False, indent=2, sort_keys=True))
        else:
            for lane_id, lane in catalog.items():
                default_marker = " default" if lane["default"] else ""
                print(
                    f"{lane_id}:{default_marker} suite={lane['target_suite']} "
                    f"preset={lane['preset']} theme={lane['theme']}"
                )
        return 0

    ensure_dirs()
    selected_lanes = _discovery_campaign_lanes_from_args(getattr(args, "lanes", ""))
    seeds = _parse_seeds(str(getattr(args, "seeds", "1")))
    duration_s = parse_duration(getattr(args, "duration", None))
    probe_ids = parse_guidance_targets(getattr(args, "probes", "") or "")
    audit_summary: dict[str, Any] = {"skipped": True}
    if not getattr(args, "skip_bug_audit", False):
        audit_run = run_probe_audit(probe_ids=probe_ids or None)
        issue_paths: list[Path] = []
        if getattr(args, "write_issues", True):
            issue_paths = write_probe_issue_drafts(
                audit_run,
                issue_dir=Path(getattr(args, "issue_dir", "new_issue/generated")),
                overwrite=bool(getattr(args, "overwrite_issues", True)),
            )
        audit_summary = {
            "skipped": False,
            "generated_at": audit_run.generated_at,
            "output_json": _project_relative_cli_path(audit_run.output_json),
            "output_markdown": _project_relative_cli_path(audit_run.output_markdown),
            "candidate_bug_families": list(audit_run.candidate_bug_families),
            "issue_files": [_project_relative_cli_path(path) for path in issue_paths],
            "environment": dict(audit_run.environment),
            "results": list(audit_run.results),
        }

    manifest_path = Path(getattr(args, "output_manifest", "") or "new_issue/generated/discovery-campaign-manifest.json")
    runs: list[dict[str, Any]] = []
    aggregate_fresh: Counter[str] = Counter()
    aggregate_issue_inspired: Counter[str] = Counter()
    aggregate_known: Counter[str] = Counter()
    aggregate_verdicts: Counter[str] = Counter()
    stopped_by_health = False
    health_stop_reason = ""
    started_at = utc_now()
    total_run_count = len(selected_lanes) * len(seeds)
    generated_issue_dir = manifest_path.parent
    score_weights = _discovery_campaign_score_weights_from_args(args)
    history_limit = max(0, int(getattr(args, "lane_history_window", DEFAULT_DISCOVERY_CAMPAIGN_HISTORY_WINDOW)))
    lane_by_id = {lane["id"]: lane for lane in selected_lanes}
    pending_by_lane = {lane["id"]: list(seeds) for lane in selected_lanes}

    def write_manifest_snapshot(*, status: str, current_run: dict[str, Any] | None = None) -> None:
        scheduler = _discovery_campaign_scheduler_snapshot(
            selected_lanes=selected_lanes,
            runs=runs,
            pending_by_lane=pending_by_lane,
            generated_issue_dir=generated_issue_dir,
            manifest_path=manifest_path,
            history_limit=history_limit,
            score_weights=score_weights,
        )
        manifest = _build_discovery_campaign_manifest(
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
            cases_per_lane_seed=int(getattr(args, "cases", 100)),
            duration=getattr(args, "duration", None),
            started_at=started_at,
            status=status,
            total_run_count=total_run_count,
            current_run=current_run,
            scheduler=scheduler,
        )
        dump_json(manifest, manifest_path)

    write_manifest_snapshot(status="running")

    while True:
        scheduler = _discovery_campaign_scheduler_snapshot(
            selected_lanes=selected_lanes,
            runs=runs,
            pending_by_lane=pending_by_lane,
            generated_issue_dir=generated_issue_dir,
            manifest_path=manifest_path,
            history_limit=history_limit,
            score_weights=score_weights,
        )
        next_lane = _next_discovery_campaign_lane(scheduler, pending_by_lane)
        if next_lane is None:
            break
        lane_id = str(next_lane["lane_id"])
        lane = lane_by_id[lane_id]
        seed = pending_by_lane[lane_id].pop(0)
        target_suite = lane["target_suite"]
        preset = lane["preset"]
        backends = resolve_target_backends(target_suite=target_suite)
        config = _discovery_campaign_config_from_args(
            args,
            preset,
            lane_discovery_biases=list(lane.get("discovery_biases", []) or []),
            lane_semantic_focus_families=list(lane.get("semantic_focus_families", []) or []),
            lane_semantic_focus_signals=list(lane.get("semantic_focus_signals", []) or []),
        )
        run_record = {
            "lane_id": lane["id"],
            "theme": lane["theme"],
            "target_suite": target_suite,
            "backends": backends,
            "preset": preset,
            "semantic_focus_families": list(config.semantic_focus_families),
            "semantic_focus_signals": list(config.semantic_focus_signals),
            "seed": seed,
            "cases": int(getattr(args, "cases", 100)),
            "duration": getattr(args, "duration", None),
            "status": "running",
            "started_at": utc_now(),
            "scheduler": {
                "priority_rank": next_lane.get("priority_rank", 0),
                "score": next_lane.get("score", 0.0),
                "budget_multiplier": next_lane.get("budget_multiplier", 1.0),
                "yield_rate": next_lane.get("yield_rate", 0.0),
                "novelty_rate": next_lane.get("novelty_rate", 0.0),
                "false_positive_rate": next_lane.get("false_positive_rate", 0.0),
                "history_runs": next_lane.get("history_runs", 0),
                "current_runs": next_lane.get("current_runs", 0),
            },
        }
        runs.append(run_record)
        print(
            f"starting discovery-campaign lane={lane['id']} suite={target_suite} preset={preset} seed={seed} "
            f"score={next_lane.get('score', 0.0):.2f} budget={next_lane.get('budget_multiplier', 1.0):.2f}",
            flush=True,
        )
        write_manifest_snapshot(status="running", current_run=run_record)
        run_file = run_fuzz(
            cases=int(getattr(args, "cases", 100)),
            seed=int(seed),
            backends=backends,
            config=config,
            duration_s=duration_s,
        )
        if getattr(args, "skip_run_report", False):
            md_path = csv_path = None
        else:
            md_path, csv_path = write_report(run_file)
        classification = _summarize_run_classification(
            run_file,
            limit=max(0, int(getattr(args, "classify_limit", 3))),
            refresh=bool(getattr(args, "refresh_classification", False)),
        )
        evidence_path = manifest_path.with_name(
            f"{manifest_path.stem}-{lane['id']}-seed{seed}-fresh-candidates.json"
        )
        fresh_evidence = _write_discovery_run_fresh_candidate_evidence(
            run_file,
            classification=classification,
            output_path=evidence_path,
        )
        candidate_pipeline = {}
        if fresh_evidence.get("candidate_row_count", 0):
            candidate_pipeline = _run_candidate_pipeline_for_evidence(
                args,
                evidence_path=evidence_path,
                manifest_path=manifest_path,
            )
        aggregate_fresh.update(classification.get("fresh_candidate_bug_families", {}))
        aggregate_issue_inspired.update(
            classification.get("issue_inspired_unsaturated_candidate_bug_families", {})
        )
        aggregate_known.update(classification.get("known_saturated_candidate_bug_families", {}))
        aggregate_verdicts.update(classification.get("triage_verdicts", {}))
        health = _summarize_run_health(run_file, limit=max(0, int(getattr(args, "classify_limit", 3))))
        run_record.update(
            {
                "run_file": _project_relative_cli_path(run_file),
                "report": _project_relative_cli_path(md_path) if md_path is not None else "",
                "csv": _project_relative_cli_path(csv_path) if csv_path is not None else "",
                "fresh_candidate_evidence": _project_relative_cli_path(evidence_path),
                "fresh_candidate_evidence_rows": fresh_evidence["candidate_row_count"],
                "candidate_pipeline": candidate_pipeline,
                "classification": classification,
                "health": health,
                "status": "completed",
                "completed_at": utc_now(),
            }
        )
        print(
            f"discovery-campaign lane={lane['id']} suite={target_suite} preset={preset} seed={seed} run={run_file}",
            flush=True,
        )
        write_manifest_snapshot(status="running")
        if getattr(args, "watch_health", False):
            if health.get("fresh_candidate_bug_families"):
                stopped_by_health = True
                health_stop_reason = "fresh_candidate"
            elif health.get("statuses", {}).get("bug", 0):
                stopped_by_health = True
                health_stop_reason = "bug_status"
            if stopped_by_health:
                print(
            f"discovery-campaign health stop: reason={health_stop_reason} lane={lane['id']} seed={seed}",
                    flush=True,
                )
                break

    write_manifest_snapshot(status="completed")

    print(f"discovery campaign manifest: {manifest_path}")
    print("fresh candidate families:")
    if aggregate_fresh:
        for family, count in sorted(aggregate_fresh.items()):
            print(f"- {family}: {count}")
    else:
        print("- none")
    print("issue-inspired unsaturated candidate families:")
    if aggregate_issue_inspired:
        for family, count in sorted(aggregate_issue_inspired.items()):
            print(f"- {family}: {count}")
    else:
        print("- none")
    print("known saturated candidate families:")
    if aggregate_known:
        for family, count in sorted(aggregate_known.items()):
            print(f"- {family}: {count}")
    else:
        print("- none")
    if getattr(args, "fail_on_fresh_candidate", False) and aggregate_fresh:
        return 2
    return 0


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
    completed_run_count = sum(1 for run in runs if run.get("status") == "completed")
    candidate_pipeline_summary = _aggregate_candidate_pipeline_summary(
        [
            run.get("candidate_pipeline", {})
            for run in runs
            if isinstance(run.get("candidate_pipeline"), dict)
        ]
    )
    return {
        "schema_version": "discovery-campaign-v1",
        "generated_at": utc_now(),
        "generated_by": "datadiff discovery-campaign",
        "status": status,
        "started_at": started_at,
        "completed_at": utc_now() if status == "completed" else "",
        "workflow": [
            "deterministic_bug_audit",
            "guided_discovery_campaign",
            "run_report",
            "candidate_classification",
            "fresh_candidate_evidence",
            "candidate_pipeline",
        ],
        "stopped_by_health": stopped_by_health,
        "health_stop_reason": health_stop_reason,
        "cases_per_lane_seed": cases_per_lane_seed,
        "duration": duration,
        "seeds": seeds,
        "lane_ids": [lane["id"] for lane in selected_lanes],
        "lanes": selected_lanes,
        "bug_audit": audit_summary,
        "progress": {
            "planned_run_count": total_run_count,
            "completed_run_count": completed_run_count,
            "remaining_run_count": max(0, total_run_count - completed_run_count),
            "current_lane_id": str((current_run or {}).get("lane_id", "")),
            "current_seed": (current_run or {}).get("seed", ""),
        },
        "scheduler": scheduler or {},
        "runs": runs,
        "summary": {
            "run_count": completed_run_count,
            "fresh_candidate_bug_families": dict(sorted(aggregate_fresh.items())),
            "issue_inspired_unsaturated_candidate_bug_families": dict(sorted(aggregate_issue_inspired.items())),
            "known_saturated_candidate_bug_families": dict(sorted(aggregate_known.items())),
            "triage_verdicts": dict(aggregate_verdicts.most_common()),
            "candidate_pipeline": candidate_pipeline_summary,
        },
    }


def _discovery_campaign_lanes_from_args(value: str) -> list[dict[str, str]]:
    raw_lane_ids = parse_guidance_targets(value or "")
    lane_ids = raw_lane_ids or list(DEFAULT_DISCOVERY_LANE_IDS)
    lanes = []
    seen = set()
    for lane_id in lane_ids:
        if lane_id in seen:
            continue
        seen.add(lane_id)
        lane = discovery_lane_spec(lane_id).to_dict()
        lanes.append(lane)
    return lanes


def _discovery_campaign_lane_catalog() -> dict[str, dict[str, Any]]:
    return discovery_lane_catalog()


def _discovery_campaign_config_from_args(
    args: argparse.Namespace,
    preset: str,
    *,
    lane_discovery_biases: list[dict[str, Any]] | None = None,
    lane_semantic_focus_families: list[str] | None = None,
    lane_semantic_focus_signals: list[str] | None = None,
) -> ExperimentConfig:
    config = _preset_config(preset)
    config.log_level = str(getattr(args, "log_level", "compact"))
    config.compress_run_log = not bool(getattr(args, "no_compress_run_log", False))
    config.artifact_limit = getattr(args, "artifact_limit", None)
    if getattr(args, "candidate_recheck_count", None) is not None:
        config.candidate_recheck_count = max(0, int(args.candidate_recheck_count))
    if getattr(args, "metamorphic_variant_limit", None) is not None:
        config.metamorphic_variant_limit = max(0, int(args.metamorphic_variant_limit))
    extra_known = parse_guidance_targets(getattr(args, "extra_known_saturated_bug_families", "") or "")
    if extra_known:
        config.known_saturated_bug_families = list(
            dict.fromkeys([*config.known_saturated_bug_families, *extra_known])
        )
    if lane_semantic_focus_families:
        config.semantic_focus_families = list(
            dict.fromkeys([*config.semantic_focus_families, *lane_semantic_focus_families])
        )
    if lane_semantic_focus_signals:
        config.semantic_focus_signals = list(
            dict.fromkeys([*config.semantic_focus_signals, *lane_semantic_focus_signals])
        )
    if lane_discovery_biases:
        config.discovery_biases = merge_discovery_biases(config.discovery_biases, lane_discovery_biases)
    return config


def cmd_bug_status(args: argparse.Namespace) -> int:
    latest_confirmation_files = [
        Path(item) for item in parse_guidance_targets(getattr(args, "latest_confirmations", "") or "")
    ]
    status = build_issue_status(
        latest_confirmation_files=latest_confirmation_files or None,
        new_issue_dir=Path(getattr(args, "new_issue_dir", "new_issue")),
        old_issue_dir=Path(getattr(args, "old_issue_dir", "old_issue")),
        generated_issue_dir=Path(getattr(args, "generated_issue_dir", "new_issue/generated")),
    )
    if getattr(args, "write_report", False):
        json_path, md_path = write_issue_status_outputs(
            status,
            output_dir=Path(getattr(args, "output_dir", "reports")),
        )
        status["output_json"] = _project_relative_cli_path(json_path)
        status["output_markdown"] = _project_relative_cli_path(md_path)
    if getattr(args, "json", False):
        print(json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    summary = status["summary"]
    print(f"confirmed latest families: {summary['confirmed_latest_count']}")
    for family in summary["confirmed_latest_families"]:
        print(f"- {family}")
    print(f"audit candidate families: {summary['audit_candidate_family_count']}")
    for family in summary["audit_candidate_families"]:
        print(f"- {family}")
    print(f"currently unsaturated fresh candidate families: {summary['fresh_candidate_family_count']}")
    fresh = summary["fresh_candidate_families"]
    if fresh:
        for family, count in fresh.items():
            print(f"- {family}: {count}")
    else:
        print("- none")
    print(f"recorded fresh candidate families: {summary['recorded_fresh_candidate_family_count']}")
    recorded_fresh = summary["recorded_fresh_candidate_families"]
    if recorded_fresh:
        for family, count in recorded_fresh.items():
            print(f"- {family}: {count}")
    else:
        print("- none")
    print(f"discovery workflow manifests: {summary.get('discovery_workflow_manifest_count', 0)}")
    print(
        f"issue bundle: {summary.get('issue_bundle_family_count', 0)} families, "
        f"{summary.get('issue_bundle_reproducer_count', 0)} reproducers, "
        f"{summary.get('issue_bundle_compile_failure_count', 0)} compile failures"
    )
    print(f"pending manual issue drafts: {summary['pending_manual_issue_draft_count']}")
    for path in summary["pending_manual_issue_drafts"]:
        print(f"- {path}")
    print(f"old known upstream issues: {summary['old_known_upstream_issue_count']}")
    if getattr(args, "write_report", False):
        print(f"bug status json:     {status['output_json']}")
        print(f"bug status markdown: {status['output_markdown']}")
    return 0


def cmd_issue_readiness(args: argparse.Namespace) -> int:
    latest_confirmation_files = [
        Path(item) for item in parse_guidance_targets(getattr(args, "latest_confirmations", "") or "")
    ]
    audit = build_issue_readiness(
        latest_confirmation_files=latest_confirmation_files or None,
        new_issue_dir=Path(getattr(args, "new_issue_dir", "new_issue")),
        old_issue_dir=Path(getattr(args, "old_issue_dir", "old_issue")),
        generated_issue_dir=Path(getattr(args, "generated_issue_dir", "new_issue/generated")),
        include_generated=bool(getattr(args, "include_generated", False)),
    )
    if getattr(args, "write_report", False):
        json_path, md_path = write_issue_readiness_outputs(
            audit,
            output_dir=Path(getattr(args, "output_dir", "reports")),
        )
        audit["output_json"] = _project_relative_cli_path(json_path)
        audit["output_markdown"] = _project_relative_cli_path(md_path)
    if getattr(args, "json", False):
        print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))
        if getattr(args, "fail_on_no_ready", False) and not audit["summary"]["ready_to_submit_count"]:
            return 2
        return 0

    summary = audit["summary"]
    print(f"issue documents: {summary['issue_document_count']}")
    print(
        f"ready to submit: {summary['ready_to_submit_count']} documents, "
        f"{summary['ready_to_submit_family_count']} families"
    )
    for path in summary["ready_to_submit"]:
        print(f"- {path}")
    print(
        f"needs dedup check: {summary['needs_dedup_check_count']} documents, "
        f"{summary['needs_dedup_check_family_count']} families"
    )
    for path in summary["needs_dedup_check"]:
        print(f"- {path}")
    print(f"needs reproducer/evidence: {summary['needs_reproducer_or_evidence_count']}")
    for path in summary["needs_reproducer_or_evidence"]:
        print(f"- {path}")
    print(f"already submitted or confirmed: {summary['already_submitted_or_confirmed_count']}")
    print(f"not latest reproducible: {summary['not_latest_reproducible_count']}")
    if getattr(args, "write_report", False):
        print(f"issue readiness json:     {audit['output_json']}")
        print(f"issue readiness markdown: {audit['output_markdown']}")
    if getattr(args, "fail_on_no_ready", False) and not summary["ready_to_submit_count"]:
        return 2
    return 0


def cmd_issue_bundle(args: argparse.Namespace) -> int:
    latest_confirmation_files = [
        Path(item) for item in parse_guidance_targets(getattr(args, "latest_confirmations", "") or "")
    ]
    statuses = parse_guidance_targets(getattr(args, "statuses", "") or "")
    manifest = build_issue_bundle(
        latest_confirmation_files=latest_confirmation_files or None,
        new_issue_dir=Path(getattr(args, "new_issue_dir", "new_issue")),
        old_issue_dir=Path(getattr(args, "old_issue_dir", "old_issue")),
        generated_issue_dir=Path(getattr(args, "generated_issue_dir", "new_issue/generated")),
        output_dir=Path(getattr(args, "output_dir", "new_issue/generated/issue-bundles")),
        statuses=statuses or list(DEFAULT_ISSUE_BUNDLE_STATUSES),
        run_reproducers=bool(getattr(args, "run_reproducers", False)),
        timeout_s=float(getattr(args, "timeout", 20.0)),
        repeat_count=int(getattr(args, "repeat", 1) or 1),
        primary_per_family=bool(getattr(args, "primary_per_family", False)),
    )
    if getattr(args, "json", False):
        print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        summary = manifest["summary"]
        print(f"issue bundle manifest: {manifest['manifest_path']}")
        print(f"issue bundle markdown: {manifest['markdown_path']}")
        print(f"bundled issues: {summary['issue_count']} documents, {summary['family_count']} families")
        if summary.get("bundled_primary_per_family"):
            print(
                f"primary per family: selected {summary.get('selected_issue_count', summary['issue_count'])} "
                f"of {summary.get('available_issue_count', summary['issue_count'])} eligible documents, "
                f"skipped supporting drafts: {summary.get('skipped_supporting_duplicate_count', 0)}"
            )
        print(f"extracted reproducers: {summary['extracted_reproducer_count']}")
        print(f"compile failures: {summary['compile_failure_count']}")
        if summary["executed_reproducer_count"]:
            print(
                f"executed reproducers: {summary['executed_reproducer_count']}, "
                f"attempts: {summary.get('executed_reproducer_attempt_count', summary['executed_reproducer_count'])}, "
                f"flaky: {summary.get('flaky_reproducer_count', 0)}, "
                f"nonzero exits: {summary['nonzero_exit_count']}, timeouts: {summary['timeout_count']}"
            )
    if getattr(args, "fail_on_missing_reproducer", False) and manifest["summary"]["missing_reproducer_count"]:
        return 2
    if getattr(args, "fail_on_compile_error", False) and manifest["summary"]["compile_failure_count"]:
        return 2
    return 0


def cmd_candidate_pipeline(args: argparse.Namespace) -> int:
    manifest_file = Path(args.manifest) if getattr(args, "manifest", None) else None
    evidence_files = [Path(item) for item in parse_guidance_targets(getattr(args, "evidence_files", "") or "")]
    pipeline = build_candidate_pipeline(
        evidence_files=evidence_files or None,
        manifest_file=manifest_file,
        output_dir=Path(getattr(args, "output_dir", DEFAULT_CANDIDATE_PIPELINE_DIR)),
        recheck_attempts=max(0, int(getattr(args, "recheck_attempts", 2))),
        reduce_artifacts=not bool(getattr(args, "no_reduce", False)),
        standalone_reproducer=not bool(getattr(args, "no_standalone_reproducer", False)),
    )
    if getattr(args, "json", False):
        print(json.dumps(pipeline, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        summary = pipeline.get("summary", {})
        print(f"candidate pipeline manifest: {pipeline.get('manifest_path', '')}")
        print(f"candidate pipeline markdown: {pipeline.get('markdown_path', '')}")
        print(
            f"candidates={summary.get('candidate_count', 0)} "
            f"reproduced={summary.get('reproduced_count', 0)} "
            f"reduced={summary.get('reduced_count', 0)} "
            f"needs_dedup={summary.get('needs_dedup_check_count', 0)}"
        )
    if getattr(args, "fail_on_ready", False) and pipeline.get("summary", {}).get("ready_to_submit_count", 0):
        return 2
    return 0


def _discovery_run_config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    config = _preset_config(str(getattr(args, "preset", "live_deep_organic")))
    config.log_level = str(getattr(args, "log_level", "compact"))
    config.compress_run_log = not bool(getattr(args, "no_compress_run_log", False))
    config.artifact_limit = getattr(args, "artifact_limit", None)
    if getattr(args, "candidate_recheck_count", None) is not None:
        config.candidate_recheck_count = max(0, int(args.candidate_recheck_count))
    if getattr(args, "metamorphic_variant_limit", None) is not None:
        config.metamorphic_variant_limit = max(0, int(args.metamorphic_variant_limit))
    extra_known = parse_guidance_targets(getattr(args, "extra_known_saturated_bug_families", "") or "")
    if extra_known:
        config.known_saturated_bug_families = list(
            dict.fromkeys([*config.known_saturated_bug_families, *extra_known])
        )
    return config


def _project_relative_cli_path(value: str | Path) -> str:
    if not value:
        return ""
    path = Path(value)
    if not path.is_absolute():
        path = Path.cwd() / path
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def _write_discovery_run_fresh_candidate_evidence(
    run_file: Path,
    *,
    classification: dict[str, Any],
    output_path: Path,
) -> dict[str, Any]:
    fresh_families = set(classification.get("fresh_candidate_bug_families", {}))
    candidate_rows: list[dict[str, Any]] = []
    if fresh_families:
        for row in read_jsonl(run_file):
            matching_findings = [
                finding
                for finding in row.get("findings", [])
                if _candidate_issue_family_key(finding) in fresh_families
            ]
            if matching_findings:
                candidate_rows.append(
                    {
                        "case": row.get("case", {}),
                        "findings": matching_findings,
                        "normalized": row.get("normalized", {}),
                        "raw_results": row.get("raw_results", {}),
                        "config": row.get("config", {}),
                        "candidate_recheck": row.get("candidate_recheck", {}),
                        "bug_dir": row.get("bug_dir", ""),
                        "status": row.get("status", ""),
                        "case_index": row.get("case_index", ""),
                        "elapsed_s": row.get("elapsed_s", ""),
                    }
                )
    evidence = {
        "schema_version": "discovery-run-fresh-candidates-v1",
        "generated_at": utc_now(),
        "generated_by": "datadiff discovery-run",
        "source_run_file": _project_relative_cli_path(run_file),
        "fresh_candidate_bug_families": classification.get("fresh_candidate_bug_families", {}),
        "candidate_row_count": len(candidate_rows),
        "candidate_rows": candidate_rows,
    }
    dump_json(evidence, output_path)
    return evidence


def _run_candidate_pipeline_for_evidence(
    args: argparse.Namespace,
    *,
    evidence_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    if getattr(args, "skip_candidate_pipeline", False):
        return {}
    try:
        pipeline = build_candidate_pipeline(
            evidence_files=[evidence_path],
            manifest_file=manifest_path,
            output_dir=Path(getattr(args, "candidate_pipeline_output_dir", DEFAULT_CANDIDATE_PIPELINE_DIR)),
            recheck_attempts=max(0, int(getattr(args, "candidate_pipeline_recheck_attempts", 2))),
            reduce_artifacts=not bool(getattr(args, "no_candidate_pipeline_reduce", False)),
            standalone_reproducer=not bool(
                getattr(args, "no_candidate_pipeline_standalone_reproducer", False)
            ),
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "error",
            "error_type": exc.__class__.__name__,
            "error": str(exc),
        }
    return {
        "status": "ok",
        "manifest_path": str(pipeline.get("manifest_path", "")),
        "markdown_path": str(pipeline.get("markdown_path", "")),
        "summary": dict(pipeline.get("summary", {}) or {}),
    }


def cmd_experiment_summary(args: argparse.Namespace) -> int:
    manifest_file = Path(args.manifest) if args.manifest else None
    md_path, csv_path = write_experiment_summary(
        manifest_file,
        refresh=bool(getattr(args, "refresh", False)),
    )
    aggregate_csv_path = md_path.with_name(f"{md_path.stem}-aggregates.csv")
    print(f"markdown summary: {md_path}")
    print(f"csv summary:      {csv_path}")
    if aggregate_csv_path.exists():
        print(f"aggregate csv:    {aggregate_csv_path}")
    return 0


def cmd_analyze_experiment(args: argparse.Namespace) -> int:
    manifest_file = Path(args.manifest) if args.manifest else None
    compare_presets = _parse_presets(args.compare_presets) if args.compare_presets else None
    reference_preset = getattr(args, "reference_preset", None) or getattr(args, "baseline_preset", "baseline")
    md_path, csv_path = analyze_experiment(
        manifest_file,
        reference_preset=reference_preset,
        legacy_reference_preset=getattr(args, "baseline_preset", None),
        baseline_preset=getattr(args, "baseline_preset", None),
        compare_presets=compare_presets,
        refresh=bool(getattr(args, "refresh", False)),
    )
    print(f"analysis markdown: {md_path}")
    print(f"analysis csv:      {csv_path}")
    return 0


def cmd_analyze_seeded_sensitivity(args: argparse.Namespace) -> int:
    manifest_file = Path(args.manifest) if args.manifest else None
    md_path, csv_path = analyze_seeded_sensitivity(manifest_file)
    print(f"seeded sensitivity markdown: {md_path}")
    print(f"seeded sensitivity csv:      {csv_path}")
    return 0


def cmd_analyze_ablation_audit(args: argparse.Namespace) -> int:
    manifest_file = Path(args.manifest) if args.manifest else None
    reference_presets = _parse_presets(args.reference_presets) if getattr(args, "reference_presets", None) else None
    trusted_presets = _parse_presets(args.trusted_presets) if args.trusted_presets else None
    ablation_presets = _parse_presets(args.ablation_presets) if args.ablation_presets else None
    md_path, csv_path = analyze_ablation_audit(
        manifest_file,
        reference_presets=reference_presets,
        legacy_reference_presets=trusted_presets,
        trusted_presets=trusted_presets,
        ablation_presets=ablation_presets,
        refresh=bool(getattr(args, "refresh", False)),
    )
    print(f"ablation audit markdown: {md_path}")
    print(f"ablation audit csv:      {csv_path}")
    return 0


def cmd_methodology_report(args: argparse.Namespace) -> int:
    manifest_file = Path(args.manifest) if args.manifest else None
    md_path, json_path = write_methodology_report(
        manifest_file,
        refresh=bool(getattr(args, "refresh", False)),
        scan_run_logs=not bool(getattr(args, "summary_only", False)),
    )
    if getattr(args, "json", False):
        print(json.dumps(load_json(json_path), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    print(f"methodology report markdown: {md_path}")
    print(f"methodology report json:     {json_path}")
    return 0


def _final_readiness_manifest_index_files(
    index_files: list[str] | tuple[str, ...],
) -> tuple[list[Path], list[Path], list[Path]]:
    manifest_files: list[Path] = []
    extra_manifest_files: list[Path] = []
    paper_run_journal_files: list[Path] = []
    for raw_path in index_files or []:
        index_path = Path(raw_path)
        data = load_json(index_path)
        if not isinstance(data, dict):
            raise ValueError(f"manifest index is not a JSON object: {index_path}")
        manifest_files.extend(_paths_from_index_value(data.get("manifest_files", [])))
        extra_manifest_files.extend(_paths_from_index_value(data.get("extra_manifest_files", [])))
        for command in data.get("commands", []) or []:
            if not isinstance(command, dict):
                continue
            manifest_files.extend(_paths_from_index_value(command.get("manifest_files", [])))
            extra_manifest_files.extend(_paths_from_index_value(command.get("extra_manifest_files", [])))
            paper_run_journal_files.extend(_paths_from_index_value(command.get("paper_run_journal_files", [])))
    return (
        _dedupe_paths(manifest_files),
        _dedupe_paths(extra_manifest_files),
        _dedupe_paths(paper_run_journal_files),
    )


def _paths_from_index_value(value: object) -> list[Path]:
    if isinstance(value, str):
        items = [part.strip() for part in value.split(",")]
    elif isinstance(value, (list, tuple, set)):
        items = [str(item).strip() for item in value]
    else:
        items = []
    return [Path(item) for item in items if item]


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    deduped: list[Path] = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


def cmd_final_readiness(args: argparse.Namespace) -> int:
    manifests = [Path(path) for path in getattr(args, "manifest", [])]
    extra_manifests = [Path(path) for path in getattr(args, "extra_manifest", [])]
    index_manifests, index_extra_manifests, index_paper_run_journals = _final_readiness_manifest_index_files(
        getattr(args, "manifest_index", [])
    )
    manifests = _dedupe_paths([*manifests, *index_manifests])
    extra_manifests = _dedupe_paths([*extra_manifests, *index_extra_manifests])
    paper_run_journal_files = index_paper_run_journals or None
    latest_confirmation_files = [Path(path) for path in getattr(args, "latest_confirmation_file", [])]
    required_live_suites = (
        tuple(_parse_presets(args.required_live_suites))
        if args.required_live_suites
        else DEFAULT_A_LEVEL_READINESS_POLICY.required_live_suites
    )
    required_live_families = (
        tuple(_parse_presets(args.required_live_families))
        if args.required_live_families
        else DEFAULT_A_LEVEL_READINESS_POLICY.required_live_families
    )
    policy = ReadinessPolicy(
        required_live_suites=required_live_suites,
        required_live_families=required_live_families,
        confirmed_live_paper_statuses=DEFAULT_A_LEVEL_READINESS_POLICY.confirmed_live_paper_statuses,
    )
    thresholds = ReadinessThresholds(
        min_live_cases_per_suite=max(0, int(args.min_live_cases_per_suite)),
        min_live_duration_hours=max(0.0, float(args.min_live_duration_hours)),
        min_live_candidate_families=max(0, int(args.min_live_candidate_families)),
        min_confirmed_live_families=max(0, int(args.min_confirmed_live_families)),
        min_historical_confirmed=max(0, int(args.min_historical_confirmed)),
        require_validation=not bool(getattr(args, "no_require_validation", False)),
        require_seeded=not bool(getattr(args, "no_require_seeded", False)),
        require_ablation=not bool(getattr(args, "no_require_ablation", False)),
        require_comparison=not bool(getattr(args, "no_require_comparison", False)),
        require_adaptive_component_ablation=not bool(
            getattr(args, "no_require_adaptive_component_ablation", False)
        ),
        min_adaptive_component_ablations=max(0, int(args.min_adaptive_component_ablations)),
        required_adaptive_component_ablations=_parse_adaptive_component_tuple(
            getattr(args, "required_adaptive_component_ablations", "")
        ),
        require_transferability_scope=not bool(getattr(args, "no_require_transferability_scope", False)),
        min_transfer_target_families=max(0, int(args.min_transfer_target_families)),
        require_cross_version_ledger=not bool(getattr(args, "no_require_cross_version_ledger", False)),
        min_cross_version_ledger_versions=max(0, int(args.min_cross_version_ledger_versions)),
        min_cross_version_ledger_families=max(0, int(args.min_cross_version_ledger_families)),
        require_cross_version_health_feedback=not bool(
            getattr(args, "no_require_cross_version_health_feedback", False)
        ),
        require_runtime_efficiency=not bool(getattr(args, "no_require_runtime_efficiency", False)),
        min_throughput_cases_s=max(0.0, float(args.min_throughput_cases_s)),
        max_scheduler_feedback_share=max(0.0, float(args.max_scheduler_feedback_share)),
        min_scheduler_feedback_cases=max(1, int(args.min_scheduler_feedback_cases)),
        require_discovery_responsiveness=not bool(
            getattr(args, "no_require_discovery_responsiveness", False)
        ),
        max_first_candidate_elapsed_s=max(0.0, float(args.max_first_candidate_elapsed_s)),
        require_closed_loop_state_persistence=not bool(
            getattr(args, "no_require_closed_loop_state_persistence", False)
        ),
        require_adaptive_live_component_evidence=not bool(
            getattr(args, "no_require_adaptive_live_component_evidence", False)
        ),
    )
    manifest_limit = (
        None
        if manifests or getattr(args, "all_manifests", False)
        else max(1, int(getattr(args, "latest_manifests", DEFAULT_FINAL_READINESS_MANIFEST_LIMIT)))
    )
    scan_run_logs = (
        bool(manifests) or bool(getattr(args, "full_run_log_scan", False))
    ) and not bool(getattr(args, "summary_only", False))
    md_path, json_path = analyze_final_readiness(
        manifests or None,
        extra_manifest_files=extra_manifests or None,
        latest_confirmation_files=latest_confirmation_files or None,
        paper_run_journal_files=paper_run_journal_files,
        manifest_limit=manifest_limit,
        scan_run_logs=scan_run_logs,
        thresholds=thresholds,
        policy=policy,
    )
    audit = load_json(json_path)
    if getattr(args, "json", False):
        print(json.dumps(audit, indent=2, sort_keys=True))
    else:
        print(f"final readiness markdown: {md_path}")
        print(f"final readiness json:     {json_path}")
    return 2 if getattr(args, "fail_on_missing", False) and not audit.get("ready", False) else 0


def cmd_review_readiness(args: argparse.Namespace) -> int:
    latest_confirmation_files = [
        Path(path) for path in getattr(args, "latest_confirmation_file", [])
    ]
    thresholds = ReviewThresholds(
        target_confirmed_bug_families=max(0, int(getattr(args, "target_confirmed", 20))),
        min_audit_candidate_families=max(0, int(getattr(args, "min_audit_candidates", 1))),
        min_discovery_workflow_manifests=max(0, int(getattr(args, "min_discovery_workflows", 1))),
        min_generated_issue_drafts=max(0, int(getattr(args, "min_generated_issue_drafts", 1))),
        min_issue_bundle_families=max(0, int(getattr(args, "min_issue_bundle_families", 1))),
        min_pending_issue_drafts=max(0, int(getattr(args, "min_pending_issue_drafts", 1))),
        min_old_known_upstream_issues=max(0, int(getattr(args, "min_old_known_issues", 1))),
    )
    audit = build_review_readiness(
        latest_confirmation_files=latest_confirmation_files or None,
        thresholds=thresholds,
    )
    if getattr(args, "write_report", False):
        json_path, md_path = write_review_readiness_outputs(
            audit,
            output_dir=Path(getattr(args, "output_dir", "reports")),
        )
        audit["output_json"] = _project_relative_cli_path(json_path)
        audit["output_markdown"] = _project_relative_cli_path(md_path)
    if getattr(args, "json", False):
        print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        summary = audit["summary"]
        print(f"review ready: {audit['ready']}")
        print(f"required gates: {summary['required_passed']}/{summary['required_total']}")
        print(
            "confirmed latest families: "
            f"{summary['confirmed_latest_count']}/{summary['target_confirmed_bug_families']}"
        )
        for item in audit["criteria"]:
            print(f"- {item['id']}: {item['status']} ({item['evidence']})")
        if audit["recommendations"]:
            print("recommendations:")
            for item in audit["recommendations"]:
                print(f"- {item}")
        if getattr(args, "write_report", False):
            print(f"review readiness json:     {audit['output_json']}")
            print(f"review readiness markdown: {audit['output_markdown']}")
    if getattr(args, "fail_on_missing", False) and not audit["ready"]:
        return 2
    return 0


def cmd_analyze_pattern_variants(args: argparse.Namespace) -> int:
    manifest_file = Path(args.manifest) if args.manifest else None
    md_path, csv_path = analyze_pattern_variants(manifest_file, pattern=args.pattern)
    print(f"pattern variant markdown: {md_path}")
    print(f"pattern variant csv:      {csv_path}")
    return 0


def cmd_targets(args: argparse.Namespace) -> int:
    context = target_context(sorted(TARGETS))
    payload = {
        "suites": list_target_suites(),
        "targets": context.target_dicts(),
        "capability_matrix": target_capability_matrix(),
        "methodology": context.methodology_summary(),
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    print("target suites:")
    for suite in payload["suites"]:
        print(
            f"- {suite['suite']}: "
            f"backends={','.join(suite['backends'])} "
            f"families={','.join(suite['families'])} "
            f"common_capabilities={len(suite['common_capabilities'])}"
        )
    print("targets:")
    for target in sorted(TARGETS.values(), key=lambda item: item.name):
        print(
            f"- {target.name}: family={target.family} "
            f"layer={target.layer} status={target.status} "
            f"capabilities={len(target.capabilities)} adapter={target.adapter}"
        )
    print("methodology:")
    print(f"- name={payload['methodology']['name']}")
    print(f"- reusable_layers={len(payload['methodology']['reusable_layers'])}")
    print(f"- shared_extension_contract={len(payload['methodology']['shared_extension_contract'])}")
    return 0


def cmd_semantic_registry(args: argparse.Namespace) -> int:
    backends = _resolve_run_backends(args) if getattr(args, "backends", None) else sorted(TARGETS)
    context = target_context(backends)
    objective_rules = _parse_exploration_objective_rules(
        getattr(args, "exploration_objective_rules", "")
    )
    if not objective_rules:
        objective_rules = list(ExperimentConfig().exploration_objective_rules)
    payload = semantic_registry_payload(
        objective_rules=objective_rules,
        target_context=context,
        metadata={
            "target_suite": str(getattr(args, "target_suite", "core") or "core"),
            "backends": backends,
        },
    )
    if getattr(args, "json", False):
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    print(f"schema={payload['schema_version']}")
    print(f"objectives={len(payload['objectives'])}")
    print(f"semantic_families={len(payload['semantic_families'])}")
    print(f"target_capabilities={len(payload['target_capabilities'])}")
    for objective in payload["objectives"]:
        print(
            f"- {objective['feature']}: "
            f"rules={len(objective['rules'])} "
            f"operators={len(objective['mutation_operator_affinity'])} "
            f"oracles={','.join(objective['oracle_roles'])}"
        )
    return 0


def cmd_version_ledger(args: argparse.Namespace) -> int:
    run_files = [Path(item) for item in parse_guidance_targets(getattr(args, "run_files", ""))]
    if not run_files and getattr(args, "run_file", None):
        run_files = [Path(getattr(args, "run_file"))]
    versions = parse_guidance_targets(getattr(args, "versions", ""))
    manifest_indexes = list(getattr(args, "manifest_index", []) or [])
    if not run_files:
        run_files, auto_versions = _version_ledger_runs_from_manifest_indexes(
            manifest_indexes
        )
        if auto_versions and not versions:
            versions = auto_versions
        if manifest_indexes and not run_files:
            print("no run logs found in --manifest-index for version-ledger", file=sys.stderr)
            return 2
    if not run_files:
        run_files = [latest_run_log_path()]
    observations = observations_from_run_logs(run_files, versions=versions)
    unique_versions = [
        observation.version_id
        for observation in observations
        if str(observation.version_id).strip()
    ]
    unique_versions = list(dict.fromkeys(unique_versions))
    if (manifest_indexes or evidence_manifest_requested(args)) and len(unique_versions) < 2:
        print(
            "version-ledger evidence requires run logs from at least two versions",
            file=sys.stderr,
        )
        return 2
    previous_ledger = {}
    previous_ledger_text = str(getattr(args, "previous_ledger", "") or "").strip()
    if previous_ledger_text:
        previous_ledger_path = Path(previous_ledger_text)
        if previous_ledger_path.is_file():
            loaded = load_json(previous_ledger_path)
            previous_ledger = loaded if isinstance(loaded, dict) else {}
    payload = build_version_ledger(
        observations,
        baseline_version=str(getattr(args, "baseline_version", "") or ""),
        previous_ledger=previous_ledger,
    )
    output = str(getattr(args, "output", "") or "").strip()
    if output:
        dump_json(payload, Path(output))
    evidence_manifest_output = str(getattr(args, "evidence_manifest_output", "") or "").strip()
    if evidence_manifest_output:
        if not output:
            raise SystemExit("--evidence-manifest-output requires --output")
        _write_version_ledger_evidence_manifest(
            Path(evidence_manifest_output),
            ledger_file=Path(output),
            run_files=run_files,
            versions=versions,
        )
    if getattr(args, "json", False):
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    summary = payload["summary"]
    print(f"schema={payload['schema_version']}")
    print(f"versions={summary['version_count']}")
    print(f"families={summary['family_count']}")
    print(f"new={summary['new_family_count']}")
    print(f"fixed={summary['fixed_family_count']}")
    print(f"regression={summary['regression_family_count']}")
    print(f"persistent={summary['persistent_family_count']}")
    print(f"health_observations={summary.get('health_observation_count', 0)}")
    print(f"invalid_cases={summary.get('invalid_case_count', 0)}")
    print(f"fallback_cases={summary.get('fallback_case_count', 0)}")
    print(f"false_positives={summary.get('false_positive_count', 0)}")
    print(f"min_throughput_cases_s={summary.get('min_throughput_cases_s', 0.0):.6g}")
    print(f"max_invalid_rate={summary.get('max_invalid_rate', 0.0):.6g}")
    print(f"max_false_positive_rate={summary.get('max_false_positive_rate', 0.0):.6g}")
    if output:
        print(f"ledger={output}")
    if evidence_manifest_output:
        print(f"evidence_manifest={evidence_manifest_output}")
    return 0


def evidence_manifest_requested(args: argparse.Namespace) -> bool:
    return bool(str(getattr(args, "evidence_manifest_output", "") or "").strip())


def _version_ledger_runs_from_manifest_indexes(index_files: list[str] | tuple[str, ...]) -> tuple[list[Path], list[str]]:
    manifest_files, _, _ = _final_readiness_manifest_index_files(index_files)
    candidates: list[tuple[Path, str]] = []
    for manifest_file in manifest_files:
        if not manifest_file.is_file():
            continue
        manifest = load_json(manifest_file)
        if not isinstance(manifest, dict):
            continue
        manifest_target_version = str(manifest.get("target_version", "") or "").strip()
        manifest_evidence_mode = str(manifest.get("evidence_mode", "") or "").strip()
        for run in manifest.get("runs", []) or []:
            if not isinstance(run, dict):
                continue
            if str(run.get("evidence_kind", "") or manifest.get("evidence_kind", "") or "").strip():
                continue
            run_file_text = str(run.get("run_file", "") or "").strip()
            if not run_file_text:
                continue
            run_file = Path(run_file_text)
            version = (
                str(run.get("target_version", "") or "").strip()
                or manifest_target_version
                or _manifest_run_version_label(manifest, run)
            )
            if not version:
                continue
            candidates.append((run_file, version))
    seen: set[tuple[str, str]] = set()
    run_files: list[Path] = []
    versions: list[str] = []
    for run_file, version in candidates:
        key = (str(run_file), version)
        if key in seen:
            continue
        seen.add(key)
        run_files.append(run_file)
        versions.append(version)
    return run_files, versions


def _manifest_run_version_label(manifest: dict[str, Any], run: dict[str, Any]) -> str:
    for source in (run, manifest):
        experiment_meta = source.get("experiment_meta", {}) if isinstance(source, dict) else {}
        if not isinstance(experiment_meta, dict):
            continue
        historical = experiment_meta.get("historical", {})
        if isinstance(historical, dict):
            version = str(historical.get("target_version", "") or "").strip()
            if version:
                return version
        variant = experiment_meta.get("variant", {})
        if isinstance(variant, dict):
            version = str(variant.get("target_version", "") or "").strip()
            if version:
                return version
    return ""


def _write_version_ledger_evidence_manifest(
    manifest_path: Path,
    *,
    ledger_file: Path,
    run_files: list[Path],
    versions: list[str],
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    run_payload = {
        "target_suite": "cross_version",
        "preset": "version_ledger",
        "seed": "",
        "run_file": str(run_files[-1]) if run_files else "",
        "backends": [],
        "report": "",
        "evidence_mode": "comparison",
        "evidence_kind": "postprocess_ledger",
        "version_ledger_file": str(ledger_file),
    }
    dump_json(
        {
            "schema_version": "version-ledger-evidence-manifest-v1",
            "created_at": utc_now(),
            "evidence_mode": "comparison",
            "evidence_kind": "postprocess_ledger",
            "target_suite": "cross_version",
            "target_suites": ["cross_version"],
            "backends": [],
            "targets": [],
            "runs": [run_payload],
            "version_ledger_file": str(ledger_file),
            "version_ledger_inputs": {
                "run_files": [str(path) for path in run_files],
                "versions": versions,
            },
            "experiment_meta": {
                "matrix_id": "baseline_scope_comparison",
                "comparison_group": "cross_version_continual_learning",
                "variant": {
                    "variant_id": "version_ledger",
                    "comparison_role": "support",
                    "component_focus": "cross_version_continual_learning",
                },
                "analysis_tags": ["cross_version", "continual_learning", "regression_ledger"],
                "counts_as_real_bugs": False,
            },
        },
        manifest_path,
    )


def cmd_prune_corpus(args: argparse.Namespace) -> int:
    keep = max(0, int(args.keep))
    interesting_dir = CORPUS_DIR / "interesting"
    files = sorted(
        [path for path in interesting_dir.glob("*.json") if path.is_file()],
        key=lambda path: (path.stat().st_mtime, path.name),
        reverse=True,
    )
    victims = files[keep:]
    bytes_to_free = sum(path.stat().st_size for path in victims)
    print(f"corpus_interesting={interesting_dir}")
    print(f"total_files={len(files)}")
    print(f"keep_latest={keep}")
    print(f"delete_candidates={len(victims)}")
    print(f"bytes_to_free={bytes_to_free}")
    if not args.yes:
        print("dry_run=true")
        print("rerun with --yes to delete candidates")
        return 0

    for path in victims:
        path.unlink()
    print("dry_run=false")
    print(f"deleted={len(victims)}")
    return 0


def cmd_show_bugs(args: argparse.Namespace) -> int:
    run_file = Path(args.run_file) if args.run_file else latest_run_log_path()
    rows = read_jsonl(run_file)
    shown = 0
    for row in rows:
        if not row.get("findings"):
            continue
        print("=" * 88)
        print(f"{row['case']['case_id']} seed={row['case']['seed']} status={row['status']}")
        print(f"ops={[op['op'] for op in row['case']['program']['operations']]}")
        print(f"bug_dir={row.get('bug_dir', '')}")
        for backend, norm in row.get("normalized", {}).items():
            print(
                f"  {backend}: {norm.get('status')} "
                f"rows={len(norm.get('rows', []))} cols={norm.get('columns', [])}"
            )
        for finding in row.get("findings", []):
            print(
                f"- [{finding['severity']}] {finding['kind']} "
                f"root={finding.get('root_cause', 'unknown')} "
                f"oracle={finding.get('oracle', 'unknown')} "
                f"triage={finding.get('triage_verdict', 'unclassified')} "
                f"suspicious={finding.get('suspicious_backends', [])}: {finding['evidence']}"
            )
            if finding.get("false_positive_reason"):
                print(f"  false_positive_reason={finding['false_positive_reason']}")
        shown += 1
        if shown >= args.limit:
            break
    if shown == 0:
        print("No bugs in selected run.")
    return 0


def cmd_classify_run(args: argparse.Namespace) -> int:
    run_file = Path(args.run_file) if args.run_file else latest_run_log_path()
    summary = _summarize_run_classification(
        run_file,
        limit=max(0, int(args.limit)),
        refresh=bool(getattr(args, "refresh", False)),
    )
    if getattr(args, "json", False):
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    print(f"run_file={run_file}")
    if summary["refresh"]:
        print("refresh=true")
    print("offline paper buckets:")
    if not summary["offline_buckets"]:
        print("- none")
    for bucket, count in summary["offline_buckets"].items():
        print(f"- {bucket}: {count}")
    print("triage verdicts:")
    if not summary["triage_verdicts"]:
        print("- none")
    for verdict, count in summary["triage_verdicts"].items():
        print(f"- {verdict}: {count}")
    print("candidate bug families:")
    if not summary["candidate_bug_families"]:
        print("- none")
    for family, count in summary["candidate_bug_families"].items():
        print(f"- {family}: {count}")
    print("fresh candidate bug families:")
    if not summary["fresh_candidate_bug_families"]:
        print("- none")
    for family, count in summary["fresh_candidate_bug_families"].items():
        print(f"- {family}: {count}")
    print("issue-inspired unsaturated candidate bug families:")
    if not summary["issue_inspired_unsaturated_candidate_bug_families"]:
        print("- none")
    for family, count in summary["issue_inspired_unsaturated_candidate_bug_families"].items():
        print(f"- {family}: {count}")
    print("known saturated candidate bug families:")
    if not summary["known_saturated_candidate_bug_families"]:
        print("- none")
    for family, count in summary["known_saturated_candidate_bug_families"].items():
        print(f"- {family}: {count}")
    print("false positive reasons:")
    if not summary["false_positive_reasons"]:
        print("- none")
    for reason, count in summary["false_positive_reasons"].items():
        print(f"- {reason}: {count}")
    for verdict, items in summary["examples"].items():
        print(f"examples[{verdict}]:")
        for item in items:
            print(
                f"- {item['case_id']} seed={item['seed']} kind={item['kind']} "
                f"root={item['root']} suspicious={item['suspicious']} signature={item['signature']} "
                f"evidence={item['evidence']}"
            )
    return 0


def cmd_run_health(args: argparse.Namespace) -> int:
    run_file = Path(args.run_file) if args.run_file else latest_run_log_path()
    summary = _summarize_run_health(run_file, limit=max(0, int(args.limit)))
    if getattr(args, "json", False):
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    print(f"run_file={summary['run_file']}")
    print(f"partial={str(summary['partial']).lower()}")
    print(f"bytes={summary['bytes']}")
    print(f"rows={summary['rows']}")
    runtime = summary.get("runtime", {})
    if runtime:
        print("runtime:")
        print(f"- status: {runtime.get('status', '') or '-'}")
        print(f"- elapsed_s: {runtime.get('elapsed_s', 0.0)}")
        print(f"- executed_cases: {runtime.get('executed_cases', 0)}")
        print(f"- throughput_cases_s: {runtime.get('throughput_cases_s', 0.0)}")
        print(f"- evidence_bytes_per_case: {runtime.get('evidence_bytes_per_case', 0.0)}")
        print(f"- run_log_bytes: {runtime.get('run_log_bytes', 0)}")
        if runtime.get("meta_file"):
            print(f"- meta_file: {runtime['meta_file']}")
        if runtime.get("checkpoint_file"):
            print(f"- checkpoint_file: {runtime['checkpoint_file']}")
        if runtime.get("closed_loop_state_file"):
            print(f"- closed_loop_state_file: {runtime['closed_loop_state_file']}")
            print(f"- closed_loop_state_bytes: {runtime.get('closed_loop_state_bytes', 0)}")
        stage_profile = runtime.get("stage_profile", {})
        if stage_profile:
            print("- stage_profile:")
            avg_ms = stage_profile.get("avg_ms_per_case", {})
            share = stage_profile.get("share_of_total", {})
            for key in sorted(avg_ms):
                print(
                    f"  - {key}: avg_ms_per_case={float(avg_ms.get(key, 0.0) or 0.0):.6f} "
                    f"share_of_total={float(share.get(key, 0.0) or 0.0):.6f}"
                )
    print("statuses:")
    if not summary["statuses"]:
        print("- none")
    for status, count in summary["statuses"].items():
        print(f"- {status}: {count}")
    print("candidate bug families:")
    if not summary["candidate_bug_families"]:
        print("- none")
    for family, count in summary["candidate_bug_families"].items():
        print(f"- {family}: {count}")
    print("fresh candidate bug families:")
    if not summary["fresh_candidate_bug_families"]:
        print("- none")
    for family, count in summary["fresh_candidate_bug_families"].items():
        print(f"- {family}: {count}")
    print("known saturated candidate bug families:")
    if not summary["known_saturated_candidate_bug_families"]:
        print("- none")
    for family, count in summary["known_saturated_candidate_bug_families"].items():
        print(f"- {family}: {count}")
    print("false positive reasons:")
    if not summary["false_positive_reasons"]:
        print("- none")
    for reason, count in summary["false_positive_reasons"].items():
        print(f"- {reason}: {count}")
    if summary["examples"]:
        print("examples:")
        for item in summary["examples"]:
            print(
                f"- {item['case_id']} seed={item['seed']} status={item['status']} "
                f"kind={item['kind']} root={item['root']} suspicious={item['suspicious']}"
            )
    if getattr(args, "fail_on_fresh_candidate", False) and summary["fresh_candidate_bug_families"]:
        return 2
    if getattr(args, "fail_on_bug", False) and summary["statuses"].get("bug", 0):
        return 2
    return 0


def cmd_discovery_campaign_status(args: argparse.Namespace) -> int:
    manifest_arg = getattr(args, "manifest", "") or "new_issue/generated/discovery-campaign-manifest.json"
    summary = _summarize_discovery_campaign_status(Path(manifest_arg), limit=max(0, int(args.limit)))
    if getattr(args, "json", False):
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(f"manifest_file={summary['manifest_file']}")
        print(f"manifest_status={summary['manifest_status']}")
        print(f"started_at={summary['started_at']}")
        print(f"completed_at={summary['completed_at'] or '-'}")
        print(f"stopped_by_health={str(summary['stopped_by_health']).lower()}")
        print(f"health_stop_reason={summary['health_stop_reason'] or '-'}")
        progress = summary.get("progress", {})
        print(f"planned_runs={progress.get('planned_run_count', 0)}")
        print(f"completed_runs={progress.get('completed_run_count', 0)}")
        print(f"remaining_runs={progress.get('remaining_run_count', 0)}")
        print(f"current_lane={progress.get('current_lane_id', '') or '-'}")
        print(f"current_seed={progress.get('current_seed', '') or '-'}")
        current_run = summary.get("current_run", {})
        print(f"current_run_status={current_run.get('status', '') or '-'}")
        print(f"latest_observed_run_file={summary.get('latest_observed_run_file', '') or '-'}")
        latest_health = summary.get("latest_observed_run_health", {})
        if latest_health:
            print("latest observed run health:")
            print(f"- partial: {str(latest_health.get('partial', False)).lower()}")
            print(f"- rows: {latest_health.get('rows', 0)}")
            print("- statuses:")
            statuses = latest_health.get("statuses", {})
            if statuses:
                for key, count in statuses.items():
                    print(f"  - {key}: {count}")
            else:
                print("  - none")
        print("aggregate fresh candidate families:")
        fresh = summary.get("fresh_candidate_bug_families", {})
        if fresh:
            for family, count in fresh.items():
                print(f"- {family}: {count}")
        else:
            print("- none")
        print("aggregate issue-inspired unsaturated candidate families:")
        inspired = summary.get("issue_inspired_unsaturated_candidate_bug_families", {})
        if inspired:
            for family, count in inspired.items():
                print(f"- {family}: {count}")
        else:
            print("- none")
        print("aggregate known saturated candidate families:")
        known = summary.get("known_saturated_candidate_bug_families", {})
        if known:
            for family, count in known.items():
                print(f"- {family}: {count}")
        else:
            print("- none")
        print("recent lane yield summary:")
        lane_summary = summary.get("recent_lane_yield_summary", [])
        if lane_summary:
            for lane in lane_summary:
                print(
                    "- {lane_id}: rank={rank} score={score:.2f} budget={budget:.2f} "
                    "yield={yield_rate:.2f} novelty={novelty:.2f} fp={fp:.2f} pending={pending}".format(
                        lane_id=lane.get("lane_id", ""),
                        rank=int(lane.get("priority_rank", 0) or 0),
                        score=float(lane.get("score", 0.0) or 0.0),
                        budget=float(lane.get("budget_multiplier", 1.0) or 1.0),
                        yield_rate=float(lane.get("yield_rate", 0.0) or 0.0),
                        novelty=float(lane.get("novelty_rate", 0.0) or 0.0),
                        fp=float(lane.get("false_positive_rate", 0.0) or 0.0),
                        pending=int(lane.get("pending_runs", 0) or 0),
                    )
                )
        else:
            print("- none")
        pipeline_summary = summary.get("candidate_pipeline", {})
        if pipeline_summary:
            print(
                "candidate pipeline summary: "
                f"pipelines={pipeline_summary.get('pipeline_count', 0)} "
                f"candidates={pipeline_summary.get('candidate_count', 0)} "
                f"reproduced={pipeline_summary.get('reproduced_count', 0)} "
                f"needs_dedup={pipeline_summary.get('needs_dedup_check_count', 0)}"
            )
    fresh = summary.get("fresh_candidate_bug_families", {}) or summary.get("latest_observed_run_health", {}).get(
        "fresh_candidate_bug_families", {}
    )
    if getattr(args, "fail_on_fresh_candidate", False) and fresh:
        return 2
    if getattr(args, "fail_on_bug", False) and summary.get("latest_observed_run_health", {}).get("statuses", {}).get(
        "bug", 0
    ):
        return 2
    return 0


def _summarize_run_health(run_file: Path, *, limit: int = 3) -> dict[str, Any]:
    rows, partial = read_jsonl_partial(run_file)
    statuses: Counter[str] = Counter()
    candidate_bug_families: Counter[str] = Counter()
    false_positive_reasons: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    known_saturated = set(DEFAULT_KNOWN_SATURATED_BUG_FAMILIES)
    candidate_origins: dict[str, Counter[str]] = {
        "organic": Counter(),
        "issue_inspired": Counter(),
    }

    for row in rows:
        statuses[row.get("status", "unknown")] += 1
        known_saturated.update(row.get("config", {}).get("known_saturated_bug_families", []) or [])
        candidate_findings: list[dict] = []
        for finding in row.get("findings", []) or []:
            if finding.get("false_positive_reason"):
                false_positive_reasons[finding["false_positive_reason"]] += 1
            if _is_candidate_issue_finding(finding):
                candidate_findings.append(finding)
                if len(examples) < limit:
                    examples.append(
                        {
                            "case_id": row.get("case", {}).get("case_id", ""),
                            "seed": row.get("case", {}).get("seed", ""),
                            "status": row.get("status", ""),
                            "kind": finding.get("kind", ""),
                            "root": finding.get("root_cause", "unknown"),
                            "suspicious": finding.get("suspicious_backends", []),
                        }
                    )
        candidate_keys = _candidate_issue_family_keys(candidate_findings)
        candidate_bug_families.update(candidate_keys)
        if candidate_keys:
            origin = _candidate_row_origin(row, candidate_findings)
            candidate_origins.setdefault(origin, Counter()).update(candidate_keys)

    candidate_items = dict(candidate_bug_families.most_common())
    known_items = {
        family: count
        for family, count in candidate_items.items()
        if _is_known_saturated_family_key(family, known_saturated)
    }
    unsaturated_items = {
        family: count
        for family, count in candidate_items.items()
        if not _is_known_saturated_family_key(family, known_saturated)
    }
    fresh_items = {
        family: count
        for family, count in candidate_origins["organic"].most_common()
        if family in unsaturated_items
    }
    runtime = _run_health_runtime_summary(run_file)
    return {
        "run_file": _project_relative_cli_path(run_file),
        "partial": partial,
        "bytes": run_file.stat().st_size if run_file.exists() else 0,
        "rows": len(rows),
        "statuses": dict(statuses.most_common()),
        "candidate_bug_families": candidate_items,
        "fresh_candidate_bug_families": fresh_items,
        "known_saturated_candidate_bug_families": known_items,
        "false_positive_reasons": dict(false_positive_reasons.most_common()),
        "examples": examples,
        "runtime": runtime,
    }


def _run_health_runtime_summary(run_file: Path) -> dict[str, Any]:
    meta_path = run_meta_path(run_file)
    meta = load_json(meta_path) if meta_path.exists() else {}
    checkpoint_text = str(meta.get("checkpoint_file", "") or "").strip() if isinstance(meta, dict) else ""
    checkpoint_path = Path(checkpoint_text).expanduser() if checkpoint_text else run_file.with_name(
        f"{run_file.stem.split('.jsonl', 1)[0]}.checkpoint.json"
    )
    checkpoint = load_json(checkpoint_path) if checkpoint_path is not None and checkpoint_path.exists() else {}
    snapshot = checkpoint if isinstance(checkpoint, dict) and checkpoint else meta if isinstance(meta, dict) else {}
    if not isinstance(snapshot, dict) or not snapshot:
        return {}
    elapsed_s = float(snapshot.get("elapsed_s", 0.0) or 0.0)
    executed = int(snapshot.get("executed_cases", 0) or 0)
    run_bytes = run_file.stat().st_size if run_file.exists() else 0
    throughput = float(snapshot.get("throughput_cases_s", 0.0) or 0.0)
    stage_profile = snapshot.get("stage_profile", {}) if isinstance(snapshot.get("stage_profile", {}), dict) else {}
    avg_ms = stage_profile.get("avg_ms_per_case", {}) if isinstance(stage_profile.get("avg_ms_per_case", {}), dict) else {}
    share = stage_profile.get("share_of_total", {}) if isinstance(stage_profile.get("share_of_total", {}), dict) else {}
    stage_profile_summary = {
        "avg_ms_per_case": {key: float(value or 0.0) for key, value in avg_ms.items()},
        "share_of_total": {key: float(value or 0.0) for key, value in share.items()},
        "case_count": int(stage_profile.get("case_count", executed) or executed),
    }
    closed_loop_state_file = str(snapshot.get("closed_loop_state_file", "") or "").strip()
    state_path = Path(closed_loop_state_file).expanduser() if closed_loop_state_file else None
    state_bytes = state_path.stat().st_size if state_path is not None and state_path.exists() else 0
    return {
        "status": str(snapshot.get("status", "")).strip(),
        "elapsed_s": elapsed_s,
        "executed_cases": executed,
        "throughput_cases_s": throughput,
        "next_seed": int(snapshot.get("next_seed", 0) or 0),
        "evidence_bytes_per_case": (run_bytes / executed) if executed else 0.0,
        "run_log_bytes": run_bytes,
        "checkpoint_file": (
            _project_relative_cli_path(checkpoint_path) if checkpoint_path is not None and checkpoint_path.exists() else ""
        ),
        "meta_file": _project_relative_cli_path(meta_path) if meta_path.exists() else "",
        "stage_profile": stage_profile_summary,
        "closed_loop_state_file": _project_relative_cli_path(state_path) if state_path is not None and state_path.exists() else "",
        "closed_loop_state_bytes": state_bytes,
        "closed_loop_state_summary": snapshot.get("closed_loop_state_summary", {}),
    }


def _summarize_discovery_campaign_status(manifest_file: Path, *, limit: int = 3) -> dict[str, Any]:
    manifest = load_json(manifest_file)
    lanes = list(manifest.get("lanes", []) or [])
    pending_by_lane: dict[str, list[int]] = {str(lane.get("id", "")): [] for lane in lanes if lane.get("id")}
    for run in manifest.get("runs", []) or []:
        lane_id = str(run.get("lane_id", ""))
        if not lane_id or lane_id not in pending_by_lane:
            continue
        if str(run.get("status", "")) == "completed":
            continue
        seed = run.get("seed")
        if seed not in pending_by_lane[lane_id]:
            pending_by_lane[lane_id].append(seed)
    progress = manifest.get("progress", {}) if isinstance(manifest.get("progress"), dict) else {}
    current_lane_id = str(progress.get("current_lane_id", ""))
    current_seed = progress.get("current_seed")
    if current_lane_id and current_seed not in (None, "") and current_seed not in pending_by_lane.get(current_lane_id, []):
        pending_by_lane.setdefault(current_lane_id, []).insert(0, current_seed)
    scheduler = manifest.get("scheduler", {}) if isinstance(manifest.get("scheduler"), dict) else {}
    if lanes:
        scheduler = _discovery_campaign_scheduler_snapshot(
            selected_lanes=[lane for lane in lanes if isinstance(lane, dict) and lane.get("id")],
            runs=[run for run in manifest.get("runs", []) or [] if isinstance(run, dict)],
            pending_by_lane=pending_by_lane,
            generated_issue_dir=manifest_file.parent,
            manifest_path=manifest_file,
            history_limit=int(
                scheduler.get("history_window", DEFAULT_DISCOVERY_CAMPAIGN_HISTORY_WINDOW)
                or DEFAULT_DISCOVERY_CAMPAIGN_HISTORY_WINDOW
            ),
            score_weights=dict(scheduler.get("score_weights", {}) or DEFAULT_DISCOVERY_CAMPAIGN_SCORE_WEIGHTS),
        )
    current_run = _current_discovery_campaign_run(manifest)
    latest_observed_run_file = _latest_discovery_campaign_run_file(manifest)
    latest_observed_run_health = (
        _summarize_run_health(latest_observed_run_file, limit=limit) if latest_observed_run_file is not None else {}
    )
    return {
        "schema_version": "discovery-campaign-status-v1",
        "generated_at": utc_now(),
        "manifest_file": _project_relative_cli_path(manifest_file),
        "manifest_status": str(manifest.get("status", "")),
        "started_at": str(manifest.get("started_at", "")),
        "completed_at": str(manifest.get("completed_at", "")),
        "stopped_by_health": bool(manifest.get("stopped_by_health", False)),
        "health_stop_reason": str(manifest.get("health_stop_reason", "")),
        "progress": dict(manifest.get("progress", {})),
        "scheduler": scheduler,
        "current_run": current_run,
        "latest_observed_run_file": (
            _project_relative_cli_path(latest_observed_run_file) if latest_observed_run_file is not None else ""
        ),
        "latest_observed_run_health": latest_observed_run_health,
        "fresh_candidate_bug_families": dict(manifest.get("summary", {}).get("fresh_candidate_bug_families", {})),
        "issue_inspired_unsaturated_candidate_bug_families": dict(
            manifest.get("summary", {}).get("issue_inspired_unsaturated_candidate_bug_families", {})
        ),
        "known_saturated_candidate_bug_families": dict(
            manifest.get("summary", {}).get("known_saturated_candidate_bug_families", {})
        ),
        "triage_verdicts": dict(manifest.get("summary", {}).get("triage_verdicts", {})),
        "recent_lane_yield_summary": list(scheduler.get("lanes", []) or []),
        "candidate_pipeline": dict(manifest.get("summary", {}).get("candidate_pipeline", {}) or {}),
    }


def _current_discovery_campaign_run(manifest: dict[str, Any]) -> dict[str, Any]:
    runs = list(manifest.get("runs", []) or [])
    for run in reversed(runs):
        if run.get("status") == "running":
            return dict(run)
    return dict(runs[-1]) if runs else {}


def _latest_discovery_campaign_run_file(manifest: dict[str, Any]) -> Path | None:
    recorded: list[Path] = []
    for run in manifest.get("runs", []) or []:
        resolved = _resolve_project_cli_path(run.get("run_file", ""))
        if resolved is not None and resolved.is_file():
            recorded.append(resolved)
    if recorded:
        return max(recorded, key=lambda path: (path.stat().st_mtime, path.name))

    started_at = _parse_manifest_utc_timestamp(str(manifest.get("started_at", "")))
    candidates = sorted(
        [*RUNS_DIR.glob("run-*.jsonl"), *RUNS_DIR.glob("run-*.jsonl.gz")],
        key=lambda path: (path.stat().st_mtime, path.name),
    )
    if not candidates:
        return None
    if started_at is None:
        return candidates[-1]
    threshold = started_at.timestamp() - 1.0
    recent = [path for path in candidates if path.stat().st_mtime >= threshold]
    return recent[-1] if recent else None


def _resolve_project_cli_path(value: str | Path | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _parse_manifest_utc_timestamp(value: str) -> datetime | None:
    cleaned = value.strip()
    if not cleaned:
        return None
    try:
        return datetime.fromisoformat(cleaned.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def _summarize_run_classification(
    run_file: Path,
    *,
    limit: int = 3,
    refresh: bool = False,
) -> dict[str, Any]:
    verdicts: Counter[str] = Counter()
    offline_buckets: Counter[str] = Counter()
    candidate_bug_families: Counter[str] = Counter()
    false_positive_reasons: Counter[str] = Counter()
    examples: dict[str, list[dict]] = {}
    cache: dict[tuple, dict] = {}
    known_saturated = set(DEFAULT_KNOWN_SATURATED_BUG_FAMILIES)
    candidate_origins: dict[str, Counter[str]] = {
        "organic": Counter(),
        "issue_inspired": Counter(),
    }

    rows = read_jsonl(run_file)
    for row in rows:
        known_saturated.update(row.get("config", {}).get("known_saturated_bug_families", []) or [])
    for row in rows:
        _classify_run_row(
            row,
            verdicts,
            offline_buckets,
            candidate_bug_families,
            false_positive_reasons,
            examples,
            limit,
            cache,
            refresh=refresh,
            candidate_origins=candidate_origins,
            known_saturated_bug_families=known_saturated,
        )
    candidate_items = dict(candidate_bug_families.most_common())
    known_items = {
        family: count
        for family, count in candidate_items.items()
        if _is_known_saturated_family_key(family, known_saturated)
    }
    unsaturated_items = {
        family: count
        for family, count in candidate_items.items()
        if not _is_known_saturated_family_key(family, known_saturated)
    }
    inspired_items = {
        family: count
        for family, count in candidate_origins["issue_inspired"].most_common()
        if family in unsaturated_items
    }
    organic_items = {
        family: count
        for family, count in candidate_origins["organic"].most_common()
        if family in unsaturated_items
    }
    return {
        "run_file": _project_relative_cli_path(run_file),
        "refresh": refresh,
        "offline_buckets": dict(offline_buckets.most_common()),
        "triage_verdicts": dict(verdicts.most_common()),
        "candidate_bug_families": candidate_items,
        "unsaturated_candidate_bug_families": unsaturated_items,
        "fresh_candidate_bug_families": organic_items,
        "issue_inspired_unsaturated_candidate_bug_families": inspired_items,
        "known_saturated_candidate_bug_families": known_items,
        "known_saturated_reference_count": len(known_saturated),
        "false_positive_reasons": dict(false_positive_reasons.most_common()),
        "examples": examples,
    }


def _is_known_saturated_family_key(family: str, known_saturated: set[str]) -> bool:
    if family in known_saturated:
        return True
    root, sep, backend_key = family.rpartition("@")
    if not sep:
        return False
    backends = [backend for backend in backend_key.split(",") if backend]
    return bool(backends) and all(f"{root}@{backend}" in known_saturated for backend in backends)


def _classify_run_row(
    row: dict,
    verdicts: Counter[str],
    offline_buckets: Counter[str],
    candidate_bug_families: Counter[str],
    false_positive_reasons: Counter[str],
    examples: dict[str, list[dict]],
    limit: int,
    cache: dict[tuple, dict],
    *,
    refresh: bool = False,
    candidate_origins: dict[str, Counter[str]] | None = None,
    known_saturated_bug_families: set[str] | None = None,
) -> None:
    if not row.get("findings"):
        return
    case = None
    normalized = None
    raw_results = row.get("raw_results", {})
    config = row.get("config", {})
    backends = list(row.get("normalized", {}))
    candidate_findings: list[dict] = []
    findings = row.get("findings", [])
    if refresh:
        refreshed = _refresh_differential_findings(row)
        if refreshed is not None:
            case, normalized, findings, config, backends = refreshed
    if refresh and normalized:
        refreshed = [finding.to_dict() for finding in evaluate_case(case, normalized)]
        if refreshed:
            findings = refreshed
    for finding in findings:
        if finding.get("triage_verdict") and finding.get("triage_verdict") != "unclassified":
            classification = {
                "verdict": finding.get("triage_verdict", "unclassified"),
                "false_positive_reason": finding.get("false_positive_reason", ""),
                "evidence": finding.get("triage_evidence", ""),
            }
        else:
            cache_key = _classification_cache_key(row, finding)
            classification = cache.get(cache_key)
            if classification is None:
                if case is None:
                    case = Case.from_dict(row["case"])
                if normalized is None:
                    normalized = _normalized_from_row(row)
                c = classify_finding(case, finding, normalized, raw_results, config, backends)
                classification = c.to_dict()
                cache[cache_key] = classification
        verdict = classification["verdict"]
        verdicts[verdict] += 1
        classified_finding = dict(finding)
        classified_finding["triage_verdict"] = verdict
        classified_finding["false_positive"] = bool(
            classification.get("false_positive", finding.get("false_positive", False))
        )
        if not classified_finding.get("source_issue"):
            metadata = row.get("case", {}).get("metadata", {}) or {}
            classified_finding["source_issue"] = (
                metadata.get("source_issue")
                or metadata.get("source_issue_alt")
                or finding.get("source_issue", "")
            )
        offline_buckets[
            offline_finding_bucket(
                classified_finding,
                tuple(sorted(known_saturated_bug_families or set())),
            )
        ] += 1
        if verdict == "candidate_implementation_bug" and not classification.get("false_positive"):
            candidate_finding = dict(finding)
            candidate_finding["triage_verdict"] = verdict
            candidate_finding["false_positive"] = False
            candidate_findings.append(candidate_finding)
        if classification.get("false_positive_reason"):
            false_positive_reasons[classification["false_positive_reason"]] += 1
        if len(examples.setdefault(verdict, [])) < limit:
            examples[verdict].append(
                {
                    "case_id": row["case"]["case_id"],
                    "seed": row["case"]["seed"],
                    "kind": finding.get("kind", ""),
                    "root": finding.get("root_cause", "unknown"),
                    "suspicious": finding.get("suspicious_backends", []),
                    "signature": finding.get("signature", ""),
                    "evidence": classification.get("evidence", ""),
                }
            )
    candidate_keys = _candidate_issue_family_keys(candidate_findings)
    candidate_bug_families.update(candidate_keys)
    if candidate_origins is not None and candidate_keys:
        origin = _candidate_row_origin(row, candidate_findings)
        candidate_origins.setdefault(origin, Counter()).update(candidate_keys)


def _candidate_row_origin(row: dict, candidate_findings: list[dict]) -> str:
    if any(
        finding.get("discovery_origin") == "issue_inspired" or finding.get("source_issue")
        for finding in candidate_findings
    ):
        return "issue_inspired"
    metadata = row.get("case", {}).get("metadata", {})
    if metadata.get("source_issue") or metadata.get("source_issue_alt"):
        return "issue_inspired"
    return "organic"


def _candidate_issue_family_key(finding: dict) -> str:
    return next(iter(_candidate_issue_family_keys([finding])), "")


def _candidate_issue_family_keys(findings: list[dict]) -> Counter:
    keys: Counter = Counter()
    root_by_backend_group: dict[str, str] = {}
    for finding in findings:
        if not _is_candidate_issue_finding(finding):
            continue
        root = str(finding.get("root_cause", "unknown"))
        if root.startswith("metamorphic_"):
            continue
        backend_group = backend_group_key(finding)
        root_by_backend_group.setdefault(backend_group, root)
    for finding in findings:
        if not _is_candidate_issue_finding(finding):
            continue
        root = str(finding.get("root_cause", "unknown"))
        backend_group = backend_group_key(finding)
        if root.startswith("metamorphic_") and backend_group in root_by_backend_group:
            root = root_by_backend_group[backend_group]
        keys[f"{root}@{backend_group}"] += 1
    return keys


def _is_candidate_issue_finding(finding: dict) -> bool:
    return finding.get("triage_verdict") == "candidate_implementation_bug" and not finding.get("false_positive")


_candidate_bug_family_key = _candidate_issue_family_key
_candidate_bug_family_keys = _candidate_issue_family_keys
_is_candidate_bug_finding = _is_candidate_issue_finding


def _classification_cache_key(row: dict, finding: dict) -> tuple:
    case = row.get("case", {})
    program = case.get("program", {})
    tables = case.get("tables", [])
    return (
        finding.get("signature", ""),
        finding.get("kind", ""),
        finding.get("root_cause", ""),
        finding.get("confidence", ""),
        tuple(finding.get("suspicious_backends", [])),
        row.get("config", {}).get("generator_profile", ""),
        tuple(operation_names(program.get("operations", []))),
        _case_has_special_float_data(tables),
        _case_has_null_data(tables),
        _case_has_non_ascii_data(tables),
    )


def _case_has_special_float_data(tables: list[dict]) -> bool:
    # JSONL stores NaN/Infinity as non-standard JSON tokens; Python json restores
    # them as floats, so this detects old and new run files.
    import math

    for table in tables:
        for row in table.get("rows", []):
            for value in row.values():
                if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                    return True
    return False


def _case_has_null_data(tables: list[dict]) -> bool:
    return any(value is None for table in tables for row in table.get("rows", []) for value in row.values())


def _case_has_non_ascii_data(tables: list[dict]) -> bool:
    return any(
        isinstance(value, str) and any(ord(ch) > 127 for ch in value)
        for table in tables
        for row in table.get("rows", [])
        for value in row.values()
    )


def _normalized_from_row(row: dict) -> dict[str, NormalizedResult]:
    return _normalized_from_mapping(row.get("normalized", {}))


def _refresh_differential_findings(
    row: dict,
) -> tuple[Case, dict[str, NormalizedResult], list[dict], dict, list[str]] | None:
    config = row.get("config", {})
    if row.get("normalized"):
        try:
            case = Case.from_dict(row["case"])
        except Exception:
            case = None
        if case is not None:
            normalized = _normalized_from_mapping(row.get("normalized", {}))
            return case, normalized, [finding.to_dict() for finding in evaluate_case(case, normalized)], config, list(normalized)

    bug_dir_text = row.get("bug_dir", "")
    if not bug_dir_text:
        return None
    bug_dir = Path(bug_dir_text)
    if not bug_dir.exists():
        bug_dir = Path.cwd() / bug_dir_text
    case_path = bug_dir / "case.json"
    normalized_path = bug_dir / "normalized.json"
    if not case_path.exists() or not normalized_path.exists():
        return None
    case = Case.from_dict(load_json(case_path))
    normalized = _normalized_from_mapping(load_json(normalized_path))
    config_path = bug_dir / "config.json"
    if config_path.exists():
        config = load_json(config_path)
    return case, normalized, [finding.to_dict() for finding in evaluate_case(case, normalized)], config, list(normalized)


def _normalized_from_mapping(mapping: dict) -> dict[str, NormalizedResult]:
    return normalized_results_from_mapping(mapping)


def cmd_reproduce(args: argparse.Namespace) -> int:
    bug_dir = Path(args.bug)
    repro = bug_dir / "reproduce.py"
    if not repro.exists():
        raise FileNotFoundError(repro)
    if args.print_command:
        print(f"Run this command to reproduce:\npython {repro}")
        return 0
    case = Case.from_dict(load_json(bug_dir / "case.json"))
    config_path = bug_dir / "config.json"
    config_data = load_json(config_path) if config_path.exists() else {}
    config = ExperimentConfig(**config_data) if config_data else ExperimentConfig()
    backends = _parse_backends(args.backends) if args.backends else list(load_json(bug_dir / "results.json"))
    result = run_loaded_case(case, backends=backends, config=config, save_artifact=False)
    print(f"status={result['status']}")
    for finding in result["findings"]:
        print(
            f"- {finding['kind']} root={finding.get('root_cause', 'unknown')} "
            f"oracle={finding.get('oracle', 'unknown')} signature={finding.get('signature', '')}"
        )
    return 0


def cmd_validate_artifact(args: argparse.Namespace) -> int:
    bug_dir = Path(args.bug)
    case = Case.from_dict(load_json(bug_dir / "case.json"))
    original_findings = load_json(bug_dir / "findings.json")
    config_path = bug_dir / "config.json"
    config_data = load_json(config_path) if config_path.exists() else {}
    config = ExperimentConfig(**config_data) if config_data else ExperimentConfig()
    backends = _parse_backends(args.backends) if args.backends else list(load_json(bug_dir / "results.json"))
    result = run_loaded_case(case, backends=backends, config=config, save_artifact=False)

    original_kinds = {f.get("kind", "") for f in original_findings}
    reproduced_kinds = {f.get("kind", "") for f in result.get("findings", [])}
    original_roots = {f.get("root_cause", "unknown") for f in original_findings}
    reproduced_roots = {f.get("root_cause", "unknown") for f in result.get("findings", [])}
    kind_ok = bool(original_kinds & reproduced_kinds)
    root_ok = bool(original_roots & reproduced_roots) if original_roots else True
    ok = result["status"] == "bug" and kind_ok
    status = "valid" if ok and root_ok else "valid-root-changed" if ok else "not-reproduced"
    print(f"artifact={bug_dir}")
    print(f"status={status}")
    print(f"original_kinds={sorted(original_kinds)}")
    print(f"reproduced_kinds={sorted(reproduced_kinds)}")
    print(f"original_roots={sorted(original_roots)}")
    print(f"reproduced_roots={sorted(reproduced_roots)}")
    return 0 if ok else 1


def cmd_triage_artifact(args: argparse.Namespace) -> int:
    bug_dir = Path(args.bug)
    case = Case.from_dict(load_json(bug_dir / "case.json"))
    original_findings = load_json(bug_dir / "findings.json")
    config_data = _load_artifact_config(bug_dir)
    config = ExperimentConfig(**config_data) if config_data else ExperimentConfig()
    backends = _parse_backends(args.backends) if args.backends else list(load_json(bug_dir / "results.json"))

    triage_case = case
    if args.reduce:
        target_roots = [] if args.reduce_ignore_roots else [
            finding.get("root_cause", "unknown") for finding in original_findings
        ]
        triage_case = reduce_case(
            case,
            backends=backends,
            config=config,
            target_kinds=[finding.get("kind", "") for finding in original_findings],
            target_roots=target_roots,
        )
        dump_json(triage_case.to_dict(), bug_dir / "reduced_case.json")
        _write_reduced_reproducer(bug_dir, backends)

    result = run_loaded_case(triage_case, backends=backends, config=config, save_artifact=False)
    report = build_triage_report(
        triage_case,
        original_findings=original_findings,
        reproduced_findings=result.get("findings", []),
        config=config.to_dict(),
        backends=backends,
    )
    report["artifact"] = str(bug_dir)
    report["reduced"] = args.reduce
    report["rows"] = len(triage_case.tables[0].rows)
    report["operations"] = len(triage_case.program.operations)
    json_path, md_path = write_triage_artifact(bug_dir, report)
    print(f"verdict={report['verdict']}")
    print(f"paper_status={report['paper_status']}")
    print(f"triage_json={json_path}")
    print(f"triage_md={md_path}")
    if args.standalone_reproducer:
        if supports_standalone_reproducer(report):
            standalone_path = write_standalone_reproducer(bug_dir, report)
            print(f"standalone_reproducer={standalone_path}")
        else:
            print("standalone_reproducer=skipped (no standalone template for this root cause)")
    return 0


def cmd_reduce(args: argparse.Namespace) -> int:
    bug_dir = Path(args.bug)
    case = Case.from_dict(load_json(bug_dir / "case.json"))
    backends = _parse_backends(args.backends)
    original_findings = load_json(bug_dir / "findings.json")
    config_data = _load_artifact_config(bug_dir)
    config = ExperimentConfig(**config_data) if config_data else ExperimentConfig(enable_artifact=False)
    config.enable_artifact = False
    config.enable_reducer = False
    target_roots = [] if args.ignore_roots else [
        finding.get("root_cause", "unknown") for finding in original_findings
    ]
    reduced = reduce_case(
        case,
        backends=backends,
        config=config,
        target_kinds=[finding.get("kind", "") for finding in original_findings],
        target_roots=target_roots,
    )
    result = run_loaded_case(reduced, backends=backends, config=config)
    dump_json(reduced.to_dict(), bug_dir / "reduced_case.json")
    _write_reduced_reproducer(bug_dir, backends)
    print(f"original rows={len(case.tables[0].rows)} ops={len(case.program.operations)}")
    print(f"reduced rows={len(reduced.tables[0].rows)} ops={len(reduced.program.operations)}")
    print(f"status={result['status']} bug_dir={result.get('bug_dir', '')}")
    return 0


def cmd_replay_fixture(args: argparse.Namespace) -> int:
    ensure_dirs()
    spec = load_fixture_replay_spec(args.spec)
    fixture_path = _resolve_fixture_replay_path(args)
    case = build_fixture_replay_case(spec, fixture_path)
    suite = str(args.target_suite or spec.get("target_suite") or "core")
    backends = _parse_backends(args.backends) if args.backends else resolve_target_backends(None, suite)
    evidence_mode = str(args.evidence_mode)
    known_bug_id = str(args.known_bug_id or spec.get("known_bug_id") or "")
    target_version = str(args.target_version or spec.get("target_version") or "")
    run_theme = str(args.run_theme or f"{evidence_mode}-fixture:{known_bug_id or case.case_id}")
    paper_notes = str(args.paper_notes or spec.get("paper_notes") or "")
    explicit_experiment_meta = parse_experiment_meta(getattr(args, "experiment_meta", None))
    experiment_meta = normalize_experiment_meta(explicit_experiment_meta)
    if replay_bug_enabled_by_default(evidence_mode):
        experiment_defaults = registered_experiment_meta_defaults(
            evidence_mode=evidence_mode,
            known_bug_id=known_bug_id,
            target_suite=suite,
            target_version=target_version,
            include_pending_historical=True,
        )
        if experiment_defaults:
            experiment_meta = merge_experiment_meta(experiment_defaults, explicit_experiment_meta)
    config = ExperimentConfig(
        enable_replay_bug=replay_bug_enabled_by_default(evidence_mode),
        enable_artifact=not args.disable_artifact,
        compress_run_log=not args.no_compress_run_log,
        artifact_limit=args.artifact_limit,
        log_level=args.log_level,
    )
    disabled_components = _parse_adaptive_components(getattr(args, "disable_adaptive_components", ""))
    adaptive_components = _adaptive_component_config(disabled_components)
    started = time.perf_counter()
    row = run_loaded_case(
        case,
        backends=backends,
        config=config,
        save_artifact=not args.disable_artifact and _fixture_artifact_budget_allows(args.artifact_limit),
        target_specs=describe_targets(backends),
    )
    elapsed_s = time.perf_counter() - started
    row["case_index"] = 0
    row["elapsed_s"] = round(elapsed_s, 6)
    row["candidate_source"] = "fixture"
    row["seed_lineage"] = {
        "root_seed": case.seed,
        "parent_seed": None,
        "parent_case_id": "",
        "mutation_seed": None,
        "depth": 0,
    }
    row["mutation"] = {"operator": "fixture", "detail": "fixture_replay", "changed": False}
    row["operation_combo"] = describe_operation_combo(case.program.operations)
    row["preflight"] = {
        "valid": True,
        "repaired": False,
        "fallback_used": False,
        "errors_before": [],
        "errors_after": [],
    }
    row["guidance"] = {
        "score": 0.0,
        "features": [],
        "matched_targets": [],
        "candidate_count": 1,
    }
    row["candidate_seed_start"] = case.seed
    row["candidate_pool_size"] = 1
    row["is_new_behavior"] = bool(row.get("findings"))
    row["signal_new_behavior"] = bool(row.get("findings"))
    row["stored_in_feedback_corpus"] = False
    row["feedback_corpus_persisted"] = False
    row["source_reward"] = None
    row["source_scheduler"] = []

    run_id = _fixture_replay_run_id(known_bug_id or case.case_id)
    suffix = ".jsonl.gz" if config.compress_run_log else ".jsonl"
    run_file = RUNS_DIR / f"{run_id}{suffix}"
    with JsonlWriter(run_file, compresslevel=1) as writer:
        writer.write(_compact_log_row(row, config.log_level))

    backend_context = target_context(backends)
    manifest_path = _experiment_manifest_path()
    run_identity = {
        "target_suite": suite,
        "backends": backends,
        "preset": "fixture_replay",
        "seed": case.seed,
        "evidence_mode": evidence_mode,
        "known_bug_id": known_bug_id,
        "target_version": target_version,
        "experiment_meta": experiment_meta,
    }
    run_semantics = resolved_run_semantics(run_identity, experiment_meta)
    preset_metadata = catalog_preset_metadata(
        "fixture_replay",
        base_preset=str(run_semantics.get("base_preset", "") or ""),
        overlays=list(run_semantics.get("overlays", []) or []),
    )
    configured_guidance_targets = list(config.guidance_targets)
    configured_effective_guidance_targets = _configured_guidance_targets(config)
    run_payload = {
        "target_suite": suite,
        "backends": backends,
        "preset": "fixture_replay",
        "seed": case.seed,
        "evidence_mode": evidence_mode,
        "known_bug_id": known_bug_id,
        "target_version": target_version,
        "batch_index": None,
        "schedule_arm_id": "",
        "estimated_cost": "",
        "worker_thread_limit": "",
        "closed_loop_state_present": False,
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
        "configured_semantic_focus_families": list(config.semantic_focus_families),
        "configured_semantic_focus_signals": list(config.semantic_focus_signals),
        "fixture_spec": str(args.spec),
        "fixture_path": str(fixture_path),
        "fixture_sha256": case.metadata.get("fixture_sha256", ""),
        "run_file": str(run_file),
        "report": "",
        "csv": "",
    }
    manifest = {
        "created_at": utc_now(),
        "presets": ["fixture_replay"],
        "seeds": [case.seed],
        "cases": 1,
        "duration_s": None,
        "evidence_mode": evidence_mode,
        "known_bug_id": known_bug_id,
        "target_version": target_version,
        "run_theme": run_theme,
        "paper_notes": paper_notes,
        "experiment_meta": experiment_meta,
        "backends": backends,
        "target_suite": suite,
        "target_suites": [suite],
        "backends_by_suite": {suite: backends},
        "targets": backend_context.target_dicts(),
        "common_capabilities": list(backend_context.common_capabilities),
        "target_context": backend_context.to_dict(),
        "log_level": config.log_level,
        "compress_run_log": config.compress_run_log,
        "metamorphic_variant_limit": config.metamorphic_variant_limit,
        "replay_bug_policy": {
            "enable_replay_bug": config.enable_replay_bug,
            "source_issues": list(config.replay_bug_source_issues),
        },
        "fixture_spec": str(args.spec),
        "fixture_path": str(fixture_path),
        "fixture_sha256": case.metadata.get("fixture_sha256", ""),
        "jobs": 1,
        "parallelism": {"requested_jobs": 1, "worker_count": 1},
        "schedule": "fixture_replay",
        "adaptive_methodology": {
            "components": dict(adaptive_components),
            "disabled_components": sorted(disabled_components),
        },
        "runs": [run_payload],
    }
    meta = {
        "run_id": run_id,
        "status": "completed",
        "run_file": str(run_file),
        "manifest_file": str(manifest_path),
        "case_log_file": "",
        "checkpoint_file": "",
        "requested_cases": 1,
        "executed_cases": 1,
        "duration_s": None,
        "elapsed_s": elapsed_s,
        "throughput_cases_s": 1 / elapsed_s if elapsed_s else 0.0,
        "findings": len(row.get("findings", [])),
        "new_behavior_cases": int(bool(row.get("findings"))),
        "signal_new_behavior_cases": int(bool(row.get("findings"))),
        "saved_artifacts": int(bool(row.get("bug_dir"))),
        "preflight": {"fixture_cases": 1},
        "quality_oracles": {},
        "seed": case.seed,
        "next_seed": case.seed + 1,
        "backends": backends,
        "targets": backend_context.target_dicts(),
        "common_capabilities": list(backend_context.common_capabilities),
        "target_context": backend_context.to_dict(),
        "config": config.to_dict(),
        "environment": row.get("environment", {}),
        "log_level": config.log_level,
        "target_suite": suite,
        "preset": "fixture_replay",
        "evidence_mode": evidence_mode,
        "known_bug_id": known_bug_id,
        "target_version": target_version,
        "run_theme": run_theme,
        "paper_notes": paper_notes,
        "experiment_meta": experiment_meta,
        "fixture_spec": str(args.spec),
        "fixture_path": str(fixture_path),
        "fixture_sha256": case.metadata.get("fixture_sha256", ""),
        "updated_at": utc_now(),
    }
    dump_json(meta, run_meta_path(run_file))
    dump_json(manifest, manifest_path)
    journal_path, journal_md = record_run_journal(
        run_file,
        context={
            "command": "replay-fixture",
            "theme": run_theme,
            "notes": paper_notes,
            "evidence_mode": evidence_mode,
            "known_bug_id": known_bug_id,
            "target_version": target_version,
            "target_suite": suite,
            "preset": "fixture_replay",
            "seed": case.seed,
            "backends": backends,
            "manifest_file": str(manifest_path),
            "experiment_meta": experiment_meta,
        },
        journal_file=REPORTS_DIR / "paper-run-journal.jsonl",
    )

    print(f"status={row['status']}")
    print(f"run_file={run_file}")
    print(f"meta_file={run_meta_path(run_file)}")
    print(f"experiment manifest: {manifest_path}")
    print(f"paper_run_journal={journal_path}")
    print(f"paper_run_journal_markdown={journal_md}")
    for finding in row.get("findings", []):
        print(
            f"- {finding.get('kind', '')} root={finding.get('root_cause', 'unknown')} "
            f"verdict={finding.get('triage_verdict', '')} "
            f"suspicious={','.join(finding.get('suspicious_backends', []) or [])}"
        )
    return 0


def cmd_historical_status(args: argparse.Namespace) -> int:
    specs = list_historical_bugs(include_pending=bool(args.include_pending))
    rows = [_historical_status_row(spec) for spec in specs]
    if args.json:
        print(json.dumps({"historical_bugs": rows}, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    print("bug_id\tstatus\tcounted\treplay_kind\ttarget_version\tfixture_env")
    for row in rows:
        print(
            "\t".join(
                [
                    row["bug_id"],
                    row["status"],
                    "yes" if row["counted"] else "no",
                    row["replay_kind"],
                    row["target_version"],
                    row["fixture_env_status"],
                ]
            )
        )
    return 0


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


def _write_reduced_reproducer(bug_dir: Path, backends: list[str]) -> None:
    repro = f'''#!/usr/bin/env python3
from datadiff.config import ExperimentConfig
from datadiff.dsl import Case
from datadiff.runner import run_loaded_case
from datadiff.util import load_json

here = __import__("pathlib").Path(__file__).parent
case = Case.from_dict(load_json(here / "reduced_case.json"))
config_data = load_json(here / "config.json")
config = ExperimentConfig(**config_data) if config_data else ExperimentConfig()
result = run_loaded_case(case, backends={backends!r}, config=config, save_artifact=False)
print(result["status"])
for finding in result["findings"]:
    print(finding)
'''
    path = bug_dir / "reproduce_reduced.py"
    path.write_text(repro, encoding="utf-8")
    path.chmod(0o755)


def _parse_seeds(value: str) -> list[int]:
    seeds: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        seeds.append(int(part))
    if not seeds:
        raise ValueError("at least one seed is required")
    return seeds


def _parse_presets(value: str) -> list[str]:
    presets = []
    for part in value.split(","):
        preset = part.strip()
        if preset:
            presets.append(preset)
    if not presets:
        raise ValueError("at least one preset is required")
    return presets


def _parse_target_suites(value: str | None) -> list[str]:
    if not value:
        return []
    suites: list[str] = []
    for part in value.split(","):
        suite = part.strip()
        if not suite:
            continue
        if suite not in TARGET_SUITES:
            raise ValueError(f"unknown target suite: {suite}")
        if suite not in suites:
            suites.append(suite)
    return suites


def _experiment_target_runs(args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    if getattr(args, "backends", None):
        return [(getattr(args, "target_suite", "custom"), _resolve_run_backends(args))]
    suites = _parse_target_suites(getattr(args, "target_suites", None)) or [args.target_suite]
    return [(suite, resolve_target_backends(target_suite=suite)) for suite in suites]

def _preset_config(name: str) -> ExperimentConfig:
    if name.endswith("_replay"):
        config = _preset_config(name[: -len("_replay")])
        config.enable_replay_bug = True
        return config
    if catalog_config := build_catalog_preset(name):
        return catalog_config
    raise ValueError(f"unknown experiment preset: {name}")


def _job_experiment_meta(job: dict[str, Any]) -> dict[str, Any]:
    return normalize_experiment_meta(job.get("experiment_meta", {}))


def _job_run_semantics(job: dict[str, Any]) -> dict[str, Any]:
    return resolved_run_semantics(job, _job_experiment_meta(job))


def _job_config(job: dict[str, Any]) -> ExperimentConfig:
    run_semantics = _job_run_semantics(job)
    base_preset = str(run_semantics.get("base_preset", "") or "").strip()
    overlays = [str(name).strip() for name in run_semantics.get("overlays", []) if str(name).strip()]
    config: ExperimentConfig | None = None
    if base_preset:
        try:
            config = build_catalog_preset(base_preset) if not overlays else build_experiment_config(
                base_preset,
                overlays,
            )
        except ValueError:
            pass
    if config is None:
        config = _preset_config(str(job["preset"]))
    _apply_job_config_overrides(config, job)
    return config


def _apply_job_config_overrides(config: ExperimentConfig, job: dict[str, Any]) -> None:
    config.target_version = str(job.get("target_version", config.target_version) or "").strip()
    config.fixed_version = str(job.get("fixed_version", config.fixed_version) or "").strip()
    version_pair_pool = parse_guidance_targets(str(job.get("version_pair_pool", "") or ""))
    if version_pair_pool:
        config.version_pair_pool = version_pair_pool
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
    relation_order = parse_guidance_targets(str(job.get("metamorphic_relation_order", "") or ""))
    if relation_order:
        config.metamorphic_relation_order = relation_order
    disabled_components = _parse_adaptive_components(job.get("disable_adaptive_components", ""))
    _apply_adaptive_component_config(config, disabled_components)


def _populate_job_learning_metadata(job: dict[str, Any]) -> None:
    config = _job_config(job)
    override_limit = job.get("metamorphic_variant_limit")
    effective_metamorphic_limit = (
        override_limit
        if override_limit is not None
        else config.metamorphic_variant_limit
    )
    job["scheduler_generator_profile"] = str(config.generator_profile or "")
    job["scheduler_guidance_strategy"] = str(config.guidance_strategy or "")
    job["scheduler_oracle_mode"] = str(config.oracle_mode or "")
    job["target_version"] = str(job.get("target_version", config.target_version) or "")
    job["fixed_version"] = str(job.get("fixed_version", config.fixed_version) or "")
    job["version_pair_pool"] = ",".join(config.version_pair_pool)
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
        objective_feature(getattr(rule, "objective", ""))
        for rule in config.exploration_objective_rules
        if objective_feature(getattr(rule, "objective", ""))
    ]


def _effective_job_local_source_scheduler(job: dict) -> tuple[bool, float]:
    disabled_components = _parse_adaptive_components(job.get("disable_adaptive_components", ""))
    if "local_source_scheduler" in disabled_components:
        return False, 0.0
    preset_config = _job_config(job)
    job_enabled = bool(job.get("enable_local_source_scheduler", False))
    enabled = preset_config.enable_local_source_scheduler or job_enabled
    if job_enabled:
        weight = job.get("local_source_exploration_weight", preset_config.local_source_exploration_weight)
    else:
        weight = preset_config.local_source_exploration_weight
    return enabled, max(0.0, float(weight))


def cmd_experiment(args: argparse.Namespace) -> int:
    ensure_dirs()
    presets = _parse_presets(args.presets)
    seeds = _parse_seeds(args.seeds)
    target_runs = _experiment_target_runs(args)
    suite_names = [suite for suite, _ in target_runs]
    backend_union = sorted({backend for _, backends in target_runs for backend in backends})
    duration_s = parse_duration(args.duration)
    evidence_mode = _resolve_evidence_mode(getattr(args, "evidence_mode", "auto"), suite_names)
    disabled_adaptive_components = _parse_adaptive_components(
        getattr(args, "disable_adaptive_components", "")
    )
    adaptive_components = _adaptive_component_config(disabled_adaptive_components)
    explicit_experiment_meta = parse_experiment_meta(getattr(args, "experiment_meta", None))
    default_experiment_meta = _default_experiment_meta_for_runs(
        evidence_mode=evidence_mode,
        target_runs=target_runs,
        presets=presets,
        known_bug_id=str(getattr(args, "known_bug_id", "") or ""),
        target_version=str(getattr(args, "target_version", "") or ""),
        include_pending_historical=True,
    )
    experiment_meta = merge_experiment_meta(default_experiment_meta, explicit_experiment_meta)
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
            or replay_bug_enabled_by_default(evidence_mode),
            "replay_bug_source_issues": (
                parse_guidance_targets(getattr(args, "replay_bug_source_issues", ""))
                or list(DEFAULT_REPLAY_BUG_SOURCE_ISSUES)
            ),
            "known_bug_id": str(getattr(args, "known_bug_id", "") or ""),
            "target_version": str(getattr(args, "target_version", "") or ""),
            "fixed_version": str(getattr(args, "fixed_version", "") or ""),
            "version_pair_pool": str(getattr(args, "version_pair_pool", "") or ""),
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
            "metamorphic_relation_order": str(getattr(args, "metamorphic_relation_order", "") or ""),
            "run_theme": str(getattr(args, "run_theme", "") or ""),
            "paper_notes": str(getattr(args, "paper_notes", "") or ""),
            "persist_closed_loop_state": bool(getattr(args, "persist_closed_loop_state", False)),
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
        _populate_job_learning_metadata(job)
        job["enable_replay_bug"] = bool(job.get("enable_replay_bug", False)) or _job_config(job).enable_replay_bug
        job["estimated_cost"] = round(_experiment_job_weight(job), 4)
    parallelism = _resolve_experiment_parallelism(args, planned_runs)
    for job in planned_runs:
        job["worker_thread_limit"] = parallelism["worker_thread_limit"]
    jobs = int(parallelism["worker_count"])
    schedule = _resolve_experiment_schedule(args, jobs=jobs)
    invalid_live_adaptive_presets = _invalid_live_adaptive_experiment_presets(planned_runs)
    if schedule == "adaptive" and evidence_mode == "live" and invalid_live_adaptive_presets:
        names = ",".join(invalid_live_adaptive_presets)
        print(
            "adaptive live experiments require guided presets so family-saturation and guidance stay active; "
            f"invalid presets: {names}. use live_deep_organic, live_cross_family, or another guided/live preset",
            flush=True,
        )
        return 2
    local_source_settings = [_effective_job_local_source_scheduler(job) for job in planned_runs]
    local_source_enabled = (
        schedule == "adaptive" and adaptive_components["local_source_scheduler"]
    ) or any(enabled for enabled, _ in local_source_settings)
    local_source_weights = [weight for enabled, weight in local_source_settings if enabled]
    if schedule == "adaptive" and adaptive_components["local_source_scheduler"] and not local_source_weights:
        local_source_weights.append(max(0.0, float(getattr(args, "local_source_exploration_weight", 0.5))))
    backend_context = target_context(backend_union)
    manifest = {
        "created_at": utc_now(),
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
        completed_runs = _run_experiment_adaptive(args, manifest, planned_runs, duration_s, jobs=jobs)
        for result in completed_runs:
            manifest["runs"].append(result["run"])
        manifest["adaptive_state"] = completed_runs[-1]["scheduler_state"] if completed_runs else []
        manifest["adaptive_learning"] = completed_runs[-1]["adaptive_learning"] if completed_runs else {}
        manifest_path = _experiment_manifest_path()
        dump_json(manifest, manifest_path)
        print(f"experiment manifest: {manifest_path}")
        if not getattr(args, "skip_paper_journal", False):
            journal_path, journal_md = _record_experiment_journal(manifest_path, manifest)
            print(f"paper run journal: {journal_path}")
            print(f"paper run journal markdown: {journal_md}")
        return 0
    completed_runs = []
    if jobs == 1:
        for job in planned_runs:
            result = _run_experiment_job(job)
            completed_runs.append(result)
            print(result["message"], flush=True)
    else:
        worker_count = min(jobs, len(planned_runs))
        try:
            completed_runs = _run_experiment_jobs_parallel(
                ProcessPoolExecutor,
                worker_count,
                planned_runs,
                max_parallel_cost=float(parallelism["max_parallel_cost"]),
            )
        except PermissionError:
            print("process parallelism unavailable; falling back to threaded workers", flush=True)
            completed_runs = _run_experiment_jobs_parallel(
                ThreadPoolExecutor,
                worker_count,
                planned_runs,
                max_parallel_cost=float(parallelism["max_parallel_cost"]),
            )
    for result in sorted(completed_runs, key=lambda item: item["order"]):
        manifest["runs"].append(result["run"])
    manifest_path = _experiment_manifest_path()
    dump_json(manifest, manifest_path)
    print(f"experiment manifest: {manifest_path}")
    if not getattr(args, "skip_paper_journal", False):
        journal_path, journal_md = _record_experiment_journal(manifest_path, manifest)
        print(f"paper run journal: {journal_path}")
        print(f"paper run journal markdown: {journal_md}")
    return 0


def _record_experiment_journal(manifest_path: Path, manifest: dict) -> tuple[Path, Path]:
    journal_file = REPORTS_DIR / "paper-run-journal.jsonl"
    entries = []
    for run in manifest.get("runs", []):
        run_file = Path(run.get("run_file", ""))
        context = {
            "command": "experiment",
            "theme": _experiment_run_theme(manifest, run),
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
        entries.append(build_run_journal_entry(run_file, context))
    append_run_journal_entries(entries, journal_file)
    md_path = write_run_journal_markdown(journal_file)
    return journal_file, md_path


def _experiment_run_theme(manifest: dict, run: dict) -> str:
    base = str(manifest.get("run_theme", "") or "").strip()
    suffix = f"{run.get('target_suite', '')}:{run.get('preset', '')}:seed{run.get('seed', '')}"
    if base:
        return f"{base} | {suffix}"
    evidence_mode = str(run.get("evidence_mode") or manifest.get("evidence_mode", "live"))
    known_bug_id = str(run.get("known_bug_id") or manifest.get("known_bug_id", "") or "")
    if replay_bug_enabled_by_default(evidence_mode) and known_bug_id:
        return f"historical:{known_bug_id}:{suffix}"
    return f"{evidence_mode}:{suffix}"


def _run_experiment_adaptive(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    planned_runs: list[dict[str, Any]],
    duration_s: float | None,
    *,
    jobs: int,
) -> list[dict[str, Any]]:
    disabled_components = _parse_adaptive_components(
        getattr(args, "disable_adaptive_components", "")
    )
    adaptive_components = _adaptive_component_config(disabled_components)
    default_batch_cases = min(100, max(1, int(args.cases or 100)))
    batch_cases = max(1, int(getattr(args, "batch_cases", 0) or default_batch_cases))
    batch_duration_s = parse_duration(getattr(args, "batch_duration", None))
    if batch_duration_s is None and duration_s is not None and args.cases is None:
        batch_duration_s = min(duration_s, 30.0)
    if args.cases is not None:
        total_cases_budget = len(planned_runs) * max(1, int(args.cases))
    elif duration_s is None:
        total_cases_budget = len(planned_runs) * 100
    else:
        total_cases_budget = None
    total_duration_budget_s = None if args.cases is not None or duration_s is None else len(planned_runs) * duration_s
    schedule_config = AdaptiveScheduleConfig(
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
    learning_state, continual_learning_sources = _adaptive_learning_state_from_ledgers(
        getattr(args, "continual_learning_ledgers", "")
    )
    scheduler = AdaptiveBudgetScheduler(
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
            completed_runs.extend(_complete_adaptive_round(scheduler, [_run_experiment_job(_adaptive_job(batches[0]))], batches))
            print(completed_runs[-1]["message"], flush=True)
        return completed_runs
    try:
        completed_runs.extend(_run_experiment_adaptive_parallel(ProcessPoolExecutor, worker_count, scheduler))
    except PermissionError:
        print("process parallelism unavailable; falling back to threaded workers", flush=True)
        completed_runs.extend(_run_experiment_adaptive_parallel(ThreadPoolExecutor, worker_count, scheduler))
    return completed_runs


def _adaptive_learning_state_from_ledgers(value: Any) -> tuple[AdaptiveLearningState, list[dict[str, Any]]]:
    state = AdaptiveLearningState()
    sources: list[dict[str, Any]] = []
    for path_text in parse_guidance_targets(str(value or "")):
        path = Path(path_text)
        if not path.is_file():
            sources.append({"path": str(path), "loaded": False, "reason": "missing"})
            continue
        payload = load_json(path)
        if not isinstance(payload, dict):
            sources.append({"path": str(path), "loaded": False, "reason": "not_object"})
            continue
        ledger_error = _continual_learning_ledger_validation_error(payload)
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
                ContinualPriorityMemory.from_state_dict(seed.get("continual_priority_memory"))
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


def _continual_learning_ledger_validation_error(payload: dict[str, Any]) -> str:
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


def _resolve_experiment_parallelism(args: argparse.Namespace, planned_runs: list[dict[str, Any]]) -> dict[str, Any]:
    cpu_count = max(1, os.cpu_count() or 1)
    requested = getattr(args, "jobs", "auto")
    if requested == "auto":
        # Each matrix job runs several native engines that may use their own
        # thread pools. A conservative default gives better 24h throughput than
        # letting every target compete for every core.
        worker_count = max(1, min(len(planned_runs), max(1, cpu_count // 4), 6))
    else:
        worker_count = max(1, int(requested))
        worker_count = min(worker_count, max(1, len(planned_runs)))
    costs = [_experiment_job_weight(job) for job in planned_runs] or [1.0]
    max_job_cost = max(costs)
    requested_cost = getattr(args, "max_parallel_cost", None)
    if requested_cost is None:
        # Cost units are backend-weighted, not CPU cores. This budget allows
        # several light runs together but usually keeps latest_all_engines-style
        # jobs from stacking on top of each other.
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


def _run_experiment_adaptive_parallel(
    executor_cls: type,
    worker_count: int,
    scheduler: AdaptiveBudgetScheduler,
) -> list[dict[str, Any]]:
    completed_runs: list[dict[str, Any]] = []
    with executor_cls(max_workers=worker_count) as executor:
        while scheduler.has_budget():
            batches = scheduler.next_round(worker_count)
            if not batches:
                break
            futures = {
                executor.submit(_run_experiment_job, _adaptive_job(batch)): batch
                for batch in batches
            }
            round_results = [future.result() for future in as_completed(futures)]
            completed_runs.extend(_complete_adaptive_round(scheduler, round_results, batches))
            for item in completed_runs[-len(batches):]:
                print(item["message"], flush=True)
    return completed_runs


def _adaptive_job(batch: Any) -> dict[str, Any]:
    job = dict(batch.job)
    disabled_components = _parse_adaptive_components(job.get("disable_adaptive_components", ""))
    job["enable_local_source_scheduler"] = "local_source_scheduler" not in disabled_components
    return job


def _complete_adaptive_round(
    scheduler: AdaptiveBudgetScheduler,
    round_results: list[dict[str, Any]],
    batches: list[Any],
) -> list[dict[str, Any]]:
    results_by_batch = {int(item["run"]["batch_index"]): item for item in round_results}
    batch_by_index = {int(batch.batch_index): batch for batch in batches}
    completed: list[dict[str, Any]] = []
    for batch_index in sorted(batch_by_index):
        batch = batch_by_index[batch_index]
        result = results_by_batch[batch_index]
        run_file = Path(result["run"]["run_file"])
        observation = summarize_batch_run(run_file)
        meta_path = run_meta_path(run_file)
        meta = load_json(meta_path) if meta_path.exists() else {}
        loaded_closed_loop_state = _load_closed_loop_state_from_meta(meta, run_file=run_file)
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


def _load_closed_loop_state_from_meta(meta: dict[str, Any], *, run_file: Path) -> dict[str, Any] | None:
    if not isinstance(meta, dict):
        return None
    inline_state = meta.get("closed_loop_state")
    if isinstance(inline_state, dict):
        return inline_state
    state_file = str(meta.get("closed_loop_state_file", "") or "").strip()
    candidate_paths = [Path(state_file)] if state_file else []
    candidate_paths.append(closed_loop_state_path(run_file))
    for path in candidate_paths:
        if not str(path):
            continue
        if path.exists():
            loaded = load_json(path)
            if isinstance(loaded, dict):
                return loaded
    return None


def _resolve_experiment_schedule(args: argparse.Namespace, *, jobs: int) -> str:
    schedule = getattr(args, "schedule", None)
    if schedule:
        return str(schedule)
    return "longest_first" if jobs > 1 else "matrix_order"


def _resolve_evidence_mode(value: str, suite_names: list[str]) -> str:
    if value != "auto":
        return value
    if suite_names and all(suite.startswith("seeded_") for suite in suite_names):
        return "seeded"
    return "live"


def _default_experiment_meta_for_runs(
    *,
    evidence_mode: str,
    target_runs: list[tuple[str, list[str]]],
    presets: list[str],
    known_bug_id: str,
    target_version: str,
    include_pending_historical: bool = True,
) -> dict[str, Any]:
    if evidence_mode == "historical":
        target_suite = target_runs[0][0] if len(target_runs) == 1 else ",".join(suite for suite, _ in target_runs)
        return registered_experiment_meta_defaults(
            evidence_mode=evidence_mode,
            known_bug_id=known_bug_id,
            target_suite=target_suite,
            target_version=target_version,
            include_pending_historical=include_pending_historical,
        )
    target_suites = tuple(dict.fromkeys(str(suite or "").strip() for suite, _ in target_runs if str(suite or "").strip()))
    if not target_suites:
        return {}
    resolved_matrices = []
    for suite in target_suites:
        for preset in presets:
            matrix = registered_experiment_matrix_for_run(
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


def _invalid_live_adaptive_experiment_presets(planned_runs: list[dict[str, Any]]) -> list[str]:
    invalid: list[str] = []
    for job in planned_runs:
        if _job_config(job).guidance_strategy != "guided":
            invalid.append(str(job["preset"]))
    return invalid


def _experiment_manifest_path() -> Path:
    ts = utc_now().replace(":", "").replace("-", "").replace("Z", "")
    return RUNS_DIR / f"experiment-{ts}-{time.time_ns()}.json"


def _run_experiment_jobs_parallel(
    executor_cls: type,
    worker_count: int,
    planned_runs: list[dict],
    *,
    max_parallel_cost: float,
) -> list[dict]:
    completed_runs = []
    scheduled_runs = sorted(planned_runs, key=_experiment_job_sort_key)
    queued_runs = list(scheduled_runs)
    running = {}
    running_cost = 0.0
    with executor_cls(max_workers=worker_count) as executor:
        while queued_runs or running:
            while len(running) < worker_count and queued_runs:
                available_cost = max_parallel_cost - running_cost
                index = _next_schedulable_job_index(queued_runs, available_cost)
                if index is None:
                    break
                job = queued_runs.pop(index)
                cost = _job_estimated_cost(job)
                future = executor.submit(_run_experiment_job, job)
                running[future] = cost
                running_cost += cost
            if not running and queued_runs:
                # A single run can exceed the current cost budget; allow it so
                # the queue cannot deadlock.
                job = queued_runs.pop(0)
                cost = _job_estimated_cost(job)
                future = executor.submit(_run_experiment_job, job)
                running[future] = cost
                running_cost += cost
            done, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in done:
                running_cost -= running.pop(future)
                result = future.result()
                completed_runs.append(result)
                print(result["message"], flush=True)
    return completed_runs


def _next_schedulable_job_index(queued_runs: list[dict], available_cost: float) -> int | None:
    for index, job in enumerate(queued_runs):
        if _job_estimated_cost(job) <= max(0.0, available_cost):
            return index
    return None


def _job_estimated_cost(job: dict) -> float:
    return float(job.get("estimated_cost", _experiment_job_weight(job)))


def _experiment_job_sort_key(job: dict) -> tuple[float, int]:
    return (-_experiment_job_weight(job), int(job["order"]))


def _experiment_job_weight(job: dict) -> float:
    config = _job_config(job)
    backend_cost = sum(_backend_cost(str(backend)) for backend in job["backends"])
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
    return backend_cost * metamorphic_multiplier * profile_multiplier * guidance_multiplier


def _backend_cost(backend: str) -> float:
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


def _run_experiment_job(job: dict) -> dict:
    _apply_native_thread_limits(int(job.get("worker_thread_limit", 1) or 1))
    preset_config = _job_config(job)
    disabled_components = _parse_adaptive_components(job.get("disable_adaptive_components", ""))
    adaptive_components = _adaptive_component_config(disabled_components)
    _apply_adaptive_component_config(preset_config, disabled_components)
    preset_config.log_level = str(job["log_level"])
    preset_config.compress_run_log = bool(job["compress_run_log"])
    preset_config.artifact_limit = job["artifact_limit"]
    job_source_scheduler_enabled = bool(job.get("enable_local_source_scheduler", False))
    preset_config.enable_local_source_scheduler = (
        preset_config.enable_local_source_scheduler or job_source_scheduler_enabled
    ) and adaptive_components["local_source_scheduler"]
    if job_source_scheduler_enabled:
        preset_config.local_source_exploration_weight = max(
            0.0,
            float(job.get("local_source_exploration_weight", preset_config.local_source_exploration_weight)),
        )
    if not adaptive_components["local_source_scheduler"]:
        preset_config.local_source_exploration_weight = 0.0
    if job["metamorphic_variant_limit"] is not None:
        preset_config.metamorphic_variant_limit = max(0, int(job["metamorphic_variant_limit"]))
    preset_config.enable_replay_bug = preset_config.enable_replay_bug or bool(job.get("enable_replay_bug", False))
    replay_bug_sources = list(job.get("replay_bug_source_issues", []) or [])
    if replay_bug_sources:
        preset_config.replay_bug_source_issues = replay_bug_sources
    run_kwargs = {
        "cases": job["cases"],
        "seed": int(job["seed"]),
        "backends": list(job["backends"]),
        "config": preset_config,
        "duration_s": job["duration_s"],
        "checkpoint_interval_s": float(job.get("checkpoint_interval_s", 60.0) or 0.0),
        "progress_interval_s": float(job.get("progress_interval_s", 60.0) or 0.0),
    }
    if bool(job.get("persist_closed_loop_state", False)):
        run_kwargs["persist_closed_loop_state"] = True
    if isinstance(job.get("closed_loop_state"), dict):
        run_kwargs["closed_loop_state"] = job["closed_loop_state"]
    run_file = run_fuzz(
        **run_kwargs,
    )
    md_path = csv_path = ""
    if not job["skip_run_reports"]:
        md_path, csv_path = write_report(run_file)
    experiment_meta = _job_experiment_meta(job)
    run_semantics = resolved_run_semantics(job, experiment_meta)
    preset_metadata = catalog_preset_metadata(
        str(job["preset"]),
        base_preset=str(run_semantics.get("base_preset", "") or ""),
        overlays=list(run_semantics.get("overlays", []) or []),
    )
    preset_semantic_focus_families = list(preset_config.semantic_focus_families)
    preset_semantic_focus_signals = list(preset_config.semantic_focus_signals)
    configured_guidance_targets = list(preset_config.guidance_targets)
    configured_effective_guidance_targets = _configured_guidance_targets(preset_config)
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


def _apply_native_thread_limits(thread_limit: int) -> None:
    value = str(max(1, int(thread_limit)))
    for name in [
        "DATADIFF_DUCKDB_THREADS",
        "POLARS_MAX_THREADS",
        "RAYON_NUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "ARROW_NUM_THREADS",
    ]:
        os.environ[name] = value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="datadiff",
        description="Semantic differential fuzzing for DataFrame and embedded analytical engines.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init", help="create project runtime directories")
    p_init.set_defaults(func=cmd_init)

    p_targets = sub.add_parser("targets", help="list supported backend targets and target suites")
    p_targets.add_argument("--json", action="store_true", help="emit target registry as JSON")
    p_targets.set_defaults(func=cmd_targets)

    p_semantic_registry = sub.add_parser(
        "semantic-registry",
        help="emit the reusable semantic objective/capability/oracle methodology registry",
    )
    add_target_suite_flags(p_semantic_registry)
    p_semantic_registry.add_argument(
        "--exploration-objective-rules",
        default="",
        help="JSON or @path defining extra neutral exploration objective rules",
    )
    p_semantic_registry.add_argument("--json", action="store_true", help="emit registry as JSON")
    p_semantic_registry.set_defaults(func=cmd_semantic_registry)

    p_version_ledger = sub.add_parser(
        "version-ledger",
        help="build a cross-version candidate-family ledger from one or more run logs",
    )
    p_version_ledger.add_argument("--run-file", default=None, help="single run log; defaults to latest run")
    p_version_ledger.add_argument("--run-files", default="", help="comma-separated run logs in version order")
    p_version_ledger.add_argument(
        "--manifest-index",
        action="append",
        default=[],
        help=(
            "final experiment manifest index to scan for run logs when --run-files is omitted; "
            "may be repeated"
        ),
    )
    p_version_ledger.add_argument("--versions", default="", help="comma-separated version ids matching --run-files")
    p_version_ledger.add_argument("--baseline-version", default="", help="baseline version id; defaults to first run")
    p_version_ledger.add_argument("--previous-ledger", default="", help="previous ledger JSON for regression detection")
    p_version_ledger.add_argument("--output", default="", help="optional ledger JSON output path")
    p_version_ledger.add_argument(
        "--evidence-manifest-output",
        default="",
        help="optional final-readiness evidence manifest that references the written ledger",
    )
    p_version_ledger.add_argument("--json", action="store_true", help="emit ledger as JSON")
    p_version_ledger.set_defaults(func=cmd_version_ledger)

    p_prune = sub.add_parser("prune-corpus", help="dry-run prune of persisted feedback corpus cases")
    p_prune.add_argument(
        "--keep",
        type=int,
        default=4096,
        help="number of newest corpus/interesting JSON files to keep",
    )
    p_prune.add_argument("--yes", action="store_true", help="delete files beyond --keep")
    p_prune.set_defaults(func=cmd_prune_corpus)

    p_fuzz = sub.add_parser("fuzz", help="run differential fuzzing")
    p_fuzz.add_argument("--cases", type=int, default=None, help="maximum cases; defaults to 100 when --duration is absent")
    p_fuzz.add_argument("--duration", default=None, help="wall-clock budget such as 10s, 5m, 24h")
    p_fuzz.add_argument("--seed", type=int, default=1)
    add_target_suite_flags(p_fuzz)
    p_fuzz.add_argument("--profile", choices=PROFILE_CHOICES, default="common")
    p_fuzz.add_argument(
        "--profile-pool",
        default="",
        help="comma-separated generator profiles for adaptive per-case profile selection",
    )
    p_fuzz.add_argument(
        "--profile-learning-weight",
        type=float,
        default=0.0,
        help="per-case generator profile contextual-learning weight; 0 keeps fixed --profile",
    )
    p_fuzz.add_argument(
        "--version-pair-pool",
        default="",
        help="comma-separated target-version pairs for adaptive per-case cross-version selection",
    )
    p_fuzz.add_argument("--target-version", default="", help="target backend/dependency version label for learning context")
    p_fuzz.add_argument("--fixed-version", default="", help="fixed/backend comparison version label for learning context")
    add_guidance_flags(p_fuzz, default_strategy="random", default_candidate_pool=8)
    add_ablation_flags(p_fuzz)
    add_paper_journal_flags(p_fuzz)
    p_fuzz.set_defaults(func=cmd_fuzz)

    p_long = sub.add_parser("longrun", help="run long-duration fuzzing and persist generated test cases")
    p_long.add_argument("--cases", type=int, default=None, help="optional maximum cases; duration is the primary budget")
    p_long.add_argument("--duration", default="24h", help="wall-clock budget such as 10m, 24h, 2d")
    p_long.add_argument("--seed", type=int, default=1)
    add_target_suite_flags(p_long)
    p_long.add_argument("--profile", choices=PROFILE_CHOICES, default="common")
    p_long.add_argument(
        "--profile-pool",
        default="",
        help="comma-separated generator profiles for adaptive per-case profile selection",
    )
    p_long.add_argument(
        "--profile-learning-weight",
        type=float,
        default=0.0,
        help="per-case generator profile contextual-learning weight; 0 keeps fixed --profile",
    )
    p_long.add_argument(
        "--version-pair-pool",
        default="",
        help="comma-separated target-version pairs for adaptive per-case cross-version selection",
    )
    p_long.add_argument("--target-version", default="", help="target backend/dependency version label for learning context")
    p_long.add_argument("--fixed-version", default="", help="fixed/backend comparison version label for learning context")
    add_guidance_flags(p_long, default_strategy="guided", default_candidate_pool=8)
    p_long.add_argument("--case-log", default=None, help="optional JSONL path for generated test cases")
    p_long.add_argument("--checkpoint-interval", default="60s", help="checkpoint write interval")
    p_long.add_argument("--progress-interval", default="60s", help="stdout progress interval")
    p_long.add_argument("--save-cases", action="store_true", help="persist every generated test case separately")
    p_long.add_argument("--no-save-cases", action="store_true", help="do not persist generated test cases separately")
    p_long.add_argument("--quiet", action="store_true", help="suppress periodic progress output")
    add_ablation_flags(p_long)
    add_paper_journal_flags(p_long)
    p_long.set_defaults(func=cmd_longrun)

    p_report = sub.add_parser("report", help="generate markdown/csv report")
    p_report.add_argument("--run-file", default=None)
    p_report.add_argument(
        "--csv-limit",
        type=int,
        default=None,
        help="maximum finding rows to export to CSV; useful for large longrun logs",
    )
    p_report.set_defaults(func=cmd_report)

    p_bug_audit = sub.add_parser(
        "bug-audit",
        help="run deterministic latest-version bug probes and write a machine-readable evidence manifest",
    )
    p_bug_audit.add_argument(
        "--probes",
        default="",
        help=f"comma-separated probe ids; defaults to all: {','.join(list_audit_probe_ids())}",
    )
    p_bug_audit.add_argument(
        "--fail-on-candidate",
        action="store_true",
        help="exit with code 2 when any candidate implementation bug is detected",
    )
    p_bug_audit.add_argument(
        "--write-issues",
        action="store_true",
        help="write candidate issue drafts into --issue-dir using the automated audit manifest",
    )
    p_bug_audit.add_argument(
        "--issue-dir",
        default="new_issue/generated",
        help="directory for --write-issues output; defaults to new_issue/generated",
    )
    p_bug_audit.add_argument(
        "--overwrite-issues",
        action="store_true",
        help="overwrite existing audit-generated issue drafts",
    )
    p_bug_audit.set_defaults(func=cmd_bug_audit)

    p_discovery_run = sub.add_parser(
        "discovery-run",
        help="run the integrated latest-version discovery workflow: audit, fuzz, report, classify, and manifest",
    )
    p_discovery_run.add_argument("--cases", type=int, default=500, help="fresh fuzz case budget")
    p_discovery_run.add_argument("--duration", default=None, help="optional wall-clock budget such as 10m or 24h")
    p_discovery_run.add_argument("--seed", type=int, default=1)
    p_discovery_run.add_argument(
        "--target-suite",
        choices=sorted(TARGET_SUITES),
        default="latest_all_engines",
        help="backend target suite for the fresh fuzz stage",
    )
    p_discovery_run.add_argument(
        "--backends",
        default=None,
        help="explicit comma-separated backend targets; overrides --target-suite",
    )
    p_discovery_run.add_argument(
        "--preset",
        default="live_deep_organic",
        help="experiment preset for the fresh fuzz stage; defaults to live_deep_organic",
    )
    p_discovery_run.add_argument(
        "--probes",
        default="",
        help=f"comma-separated audit probe ids; defaults to all: {','.join(list_audit_probe_ids())}",
    )
    p_discovery_run.add_argument("--skip-bug-audit", action="store_true", help="skip deterministic audit stage")
    p_discovery_run.add_argument(
        "--write-issues",
        dest="write_issues",
        action="store_true",
        default=True,
        help="write audit candidate issue drafts into --issue-dir; enabled by default",
    )
    p_discovery_run.add_argument(
        "--no-write-issues",
        dest="write_issues",
        action="store_false",
        help="do not write audit issue drafts",
    )
    p_discovery_run.add_argument(
        "--issue-dir",
        default="new_issue/generated",
        help="directory for generated audit issue drafts",
    )
    p_discovery_run.add_argument(
        "--overwrite-issues",
        dest="overwrite_issues",
        action="store_true",
        default=True,
        help="overwrite existing generated audit issue drafts; enabled by default",
    )
    p_discovery_run.add_argument(
        "--no-overwrite-issues",
        dest="overwrite_issues",
        action="store_false",
        help="keep existing generated audit issue drafts",
    )
    p_discovery_run.add_argument(
        "--output-manifest",
        default="new_issue/generated/discovery-run-manifest.json",
        help="portable manifest for the integrated discovery run",
    )
    p_discovery_run.add_argument("--skip-run-report", action="store_true", help="skip markdown/csv report generation")
    p_discovery_run.add_argument("--classify-limit", type=int, default=3, help="example count per triage verdict")
    p_discovery_run.add_argument(
        "--refresh-classification",
        action="store_true",
        help="recompute differential findings from stored normalized outputs before classifying",
    )
    p_discovery_run.add_argument(
        "--extra-known-saturated-bug-families",
        default="",
        help="additional comma-separated root@backend families to exclude from fresh counts",
    )
    p_discovery_run.add_argument(
        "--candidate-recheck-count",
        type=int,
        default=None,
        help="override preset candidate recheck count",
    )
    p_discovery_run.add_argument(
        "--metamorphic-variant-limit",
        type=int,
        default=None,
        help="override preset metamorphic variant limit",
    )
    p_discovery_run.add_argument("--artifact-limit", type=int, default=None)
    p_discovery_run.add_argument("--no-compress-run-log", action="store_true")
    p_discovery_run.add_argument(
        "--log-level",
        choices=["full", "compact", "minimal"],
        default="compact",
        help="fresh fuzz JSONL detail level",
    )
    p_discovery_run.add_argument(
        "--fail-on-fresh-candidate",
        action="store_true",
        help="exit with code 2 when the fuzz stage finds a non-saturated candidate family",
    )
    p_discovery_run.add_argument(
        "--skip-candidate-pipeline",
        action="store_true",
        help="skip the automatic freeze/recheck/reduce/dedup/issue-readiness pipeline for fresh candidates",
    )
    p_discovery_run.add_argument("--candidate-pipeline-recheck-attempts", type=int, default=2)
    p_discovery_run.add_argument(
        "--candidate-pipeline-output-dir",
        default=str(DEFAULT_CANDIDATE_PIPELINE_DIR.relative_to(PROJECT_ROOT)),
    )
    p_discovery_run.add_argument("--no-candidate-pipeline-reduce", action="store_true")
    p_discovery_run.add_argument("--no-candidate-pipeline-standalone-reproducer", action="store_true")
    p_discovery_run.set_defaults(func=cmd_discovery_run)

    p_discovery_campaign = sub.add_parser(
        "discovery-campaign",
        help="run multiple narrow latest-version discovery lanes and write one evidence manifest",
    )
    p_discovery_campaign.add_argument("--cases", type=int, default=100, help="case budget per lane/seed")
    p_discovery_campaign.add_argument("--duration", default=None, help="optional wall-clock budget per lane/seed")
    p_discovery_campaign.add_argument("--seeds", default="1", help="comma-separated seeds for every selected lane")
    p_discovery_campaign.add_argument(
        "--lanes",
        default="",
        help=f"comma-separated lane ids; defaults to: {','.join(DEFAULT_DISCOVERY_LANE_IDS)}",
    )
    p_discovery_campaign.add_argument("--list-lanes", action="store_true", help="print available discovery lanes and exit")
    p_discovery_campaign.add_argument("--json", action="store_true", help="with --list-lanes, emit lane catalog as JSON")
    p_discovery_campaign.add_argument(
        "--probes",
        default="",
        help=f"comma-separated audit probe ids; defaults to all: {','.join(list_audit_probe_ids())}",
    )
    p_discovery_campaign.add_argument("--skip-bug-audit", action="store_true", help="skip deterministic audit stage")
    p_discovery_campaign.add_argument(
        "--write-issues",
        dest="write_issues",
        action="store_true",
        default=True,
        help="write audit candidate issue drafts into --issue-dir; enabled by default",
    )
    p_discovery_campaign.add_argument(
        "--no-write-issues",
        dest="write_issues",
        action="store_false",
        help="do not write audit issue drafts",
    )
    p_discovery_campaign.add_argument("--issue-dir", default="new_issue/generated")
    p_discovery_campaign.add_argument(
        "--overwrite-issues",
        dest="overwrite_issues",
        action="store_true",
        default=True,
        help="overwrite existing generated audit issue drafts; enabled by default",
    )
    p_discovery_campaign.add_argument(
        "--no-overwrite-issues",
        dest="overwrite_issues",
        action="store_false",
        help="keep existing generated audit issue drafts",
    )
    p_discovery_campaign.add_argument(
        "--output-manifest",
        default="new_issue/generated/discovery-campaign-manifest.json",
        help="portable manifest for the guided discovery campaign",
    )
    p_discovery_campaign.add_argument("--skip-run-report", action="store_true", help="skip markdown/csv report generation")
    p_discovery_campaign.add_argument("--classify-limit", type=int, default=3, help="example count per triage verdict")
    p_discovery_campaign.add_argument(
        "--refresh-classification",
        action="store_true",
        help="recompute differential findings from stored normalized outputs before classifying",
    )
    p_discovery_campaign.add_argument(
        "--extra-known-saturated-bug-families",
        default="",
        help="additional comma-separated root@backend families to exclude from fresh counts",
    )
    p_discovery_campaign.add_argument("--candidate-recheck-count", type=int, default=None)
    p_discovery_campaign.add_argument("--metamorphic-variant-limit", type=int, default=None)
    p_discovery_campaign.add_argument("--artifact-limit", type=int, default=None)
    p_discovery_campaign.add_argument("--no-compress-run-log", action="store_true")
    p_discovery_campaign.add_argument(
        "--log-level",
        choices=["full", "compact", "minimal"],
        default="compact",
        help="fresh fuzz JSONL detail level",
    )
    p_discovery_campaign.add_argument(
        "--fail-on-fresh-candidate",
        action="store_true",
        help="exit with code 2 when any lane finds a non-saturated candidate family",
    )
    p_discovery_campaign.add_argument(
        "--watch-health",
        action="store_true",
        help="stop remaining lanes after any completed lane/seed run contains a bug row or organic fresh candidate",
    )
    p_discovery_campaign.add_argument("--lane-history-window", type=int, default=DEFAULT_DISCOVERY_CAMPAIGN_HISTORY_WINDOW)
    p_discovery_campaign.add_argument(
        "--lane-yield-weight",
        type=float,
        default=DEFAULT_DISCOVERY_CAMPAIGN_SCORE_WEIGHTS["yield_rate"],
    )
    p_discovery_campaign.add_argument(
        "--lane-novelty-weight",
        type=float,
        default=DEFAULT_DISCOVERY_CAMPAIGN_SCORE_WEIGHTS["novelty_rate"],
    )
    p_discovery_campaign.add_argument(
        "--lane-false-positive-penalty",
        type=float,
        default=DEFAULT_DISCOVERY_CAMPAIGN_SCORE_WEIGHTS["false_positive_penalty"],
    )
    p_discovery_campaign.add_argument(
        "--skip-candidate-pipeline",
        action="store_true",
        help="skip the automatic freeze/recheck/reduce/dedup/issue-readiness pipeline for fresh candidates",
    )
    p_discovery_campaign.add_argument("--candidate-pipeline-recheck-attempts", type=int, default=2)
    p_discovery_campaign.add_argument(
        "--candidate-pipeline-output-dir",
        default=str(DEFAULT_CANDIDATE_PIPELINE_DIR.relative_to(PROJECT_ROOT)),
    )
    p_discovery_campaign.add_argument("--no-candidate-pipeline-reduce", action="store_true")
    p_discovery_campaign.add_argument("--no-candidate-pipeline-standalone-reproducer", action="store_true")
    p_discovery_campaign.set_defaults(func=cmd_discovery_campaign)

    p_discovery_campaign_status = sub.add_parser(
        "discovery-campaign-status",
        help="summarize a running or completed discovery-campaign manifest and its latest observed run health",
    )
    p_discovery_campaign_status.add_argument("--manifest", default="new_issue/generated/discovery-campaign-manifest.json")
    p_discovery_campaign_status.add_argument("--limit", type=int, default=3, help="candidate examples to show from latest run")
    p_discovery_campaign_status.add_argument("--json", action="store_true", help="emit machine-readable status JSON")
    p_discovery_campaign_status.add_argument(
        "--fail-on-fresh-candidate",
        action="store_true",
        help="exit with code 2 when the discovery-campaign manifest or latest observed run contains an unsaturated organic candidate",
    )
    p_discovery_campaign_status.add_argument(
        "--fail-on-bug",
        action="store_true",
        help="exit with code 2 when the latest observed run contains any status=bug row",
    )
    p_discovery_campaign_status.set_defaults(func=cmd_discovery_campaign_status)

    p_bug_status = sub.add_parser(
        "bug-status",
        help="summarize confirmed, candidate, generated, and old-known bug evidence without scanning run logs",
    )
    p_bug_status.add_argument("--json", action="store_true", help="emit machine-readable status JSON")
    p_bug_status.add_argument(
        "--latest-confirmations",
        default="",
        help="comma-separated latest confirmation JSON files; defaults to experiments/latest_confirmations.json",
    )
    p_bug_status.add_argument("--new-issue-dir", default="new_issue")
    p_bug_status.add_argument("--old-issue-dir", default="old_issue")
    p_bug_status.add_argument("--generated-issue-dir", default="new_issue/generated")
    p_bug_status.add_argument("--write-report", action="store_true", help="write reports/bug-status-*.json and .md")
    p_bug_status.add_argument("--output-dir", default="reports", help="directory for --write-report output")
    p_bug_status.set_defaults(func=cmd_bug_status)

    p_issue_readiness = sub.add_parser(
        "issue-readiness",
        help="audit local issue drafts for submission readiness without scanning run logs",
    )
    p_issue_readiness.add_argument("--json", action="store_true", help="emit machine-readable readiness JSON")
    p_issue_readiness.add_argument(
        "--latest-confirmations",
        default="",
        help="comma-separated latest confirmation JSON files; defaults to experiments/latest_confirmations.json",
    )
    p_issue_readiness.add_argument("--new-issue-dir", default="new_issue")
    p_issue_readiness.add_argument("--old-issue-dir", default="old_issue")
    p_issue_readiness.add_argument("--generated-issue-dir", default="new_issue/generated")
    p_issue_readiness.add_argument(
        "--include-generated",
        action="store_true",
        help="also audit raw generated issue drafts under --generated-issue-dir",
    )
    p_issue_readiness.add_argument(
        "--write-report",
        action="store_true",
        help="write reports/issue-readiness-*.json and .md",
    )
    p_issue_readiness.add_argument("--output-dir", default="reports", help="directory for --write-report output")
    p_issue_readiness.add_argument(
        "--fail-on-no-ready",
        action="store_true",
        help="exit with code 2 when no local issue draft is ready to submit",
    )
    p_issue_readiness.set_defaults(func=cmd_issue_readiness)

    p_issue_bundle = sub.add_parser(
        "issue-bundle",
        help="extract local issue reproducers and write a portable evidence manifest",
    )
    p_issue_bundle.add_argument("--json", action="store_true", help="emit the generated bundle manifest as JSON")
    p_issue_bundle.add_argument(
        "--latest-confirmations",
        default="",
        help="comma-separated latest confirmation JSON files; defaults to experiments/latest_confirmations.json",
    )
    p_issue_bundle.add_argument("--new-issue-dir", default="new_issue")
    p_issue_bundle.add_argument("--old-issue-dir", default="old_issue")
    p_issue_bundle.add_argument("--generated-issue-dir", default="new_issue/generated")
    p_issue_bundle.add_argument(
        "--statuses",
        default=",".join(DEFAULT_ISSUE_BUNDLE_STATUSES),
        help="comma-separated issue-readiness statuses to bundle",
    )
    p_issue_bundle.add_argument(
        "--output-dir",
        default="new_issue/generated/issue-bundles",
        help="directory for manifest and extracted reproducers",
    )
    p_issue_bundle.add_argument(
        "--run-reproducers",
        action="store_true",
        help="execute extracted reproducers and capture stdout/stderr in the manifest",
    )
    p_issue_bundle.add_argument("--timeout", type=float, default=20.0, help="per-reproducer timeout in seconds")
    p_issue_bundle.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="number of times to execute each reproducer when --run-reproducers is set",
    )
    p_issue_bundle.add_argument(
        "--primary-per-family",
        action="store_true",
        help="bundle only the primary selected issue draft for each issue-readiness submission family",
    )
    p_issue_bundle.add_argument(
        "--fail-on-missing-reproducer",
        action="store_true",
        help="exit with code 2 if any selected issue lacks a Python reproducer block",
    )
    p_issue_bundle.add_argument(
        "--fail-on-compile-error",
        action="store_true",
        help="exit with code 2 if any extracted reproducer has a syntax error",
    )
    p_issue_bundle.set_defaults(func=cmd_issue_bundle)

    p_candidate_pipeline = sub.add_parser(
        "candidate-pipeline",
        help="freeze fresh candidates and run recheck/reduce/dedup/issue-readiness automatically",
    )
    p_candidate_pipeline.add_argument("--manifest", default=None)
    p_candidate_pipeline.add_argument(
        "--evidence-files",
        default="",
        help="comma-separated fresh candidate evidence JSON files; overrides --manifest discovery when provided",
    )
    p_candidate_pipeline.add_argument(
        "--output-dir",
        default=str(DEFAULT_CANDIDATE_PIPELINE_DIR.relative_to(PROJECT_ROOT)),
    )
    p_candidate_pipeline.add_argument("--recheck-attempts", type=int, default=2)
    p_candidate_pipeline.add_argument("--no-reduce", action="store_true")
    p_candidate_pipeline.add_argument("--no-standalone-reproducer", action="store_true")
    p_candidate_pipeline.add_argument("--json", action="store_true")
    p_candidate_pipeline.add_argument(
        "--fail-on-ready",
        action="store_true",
        help="exit with code 2 when the pipeline produces any ready-to-submit draft",
    )
    p_candidate_pipeline.set_defaults(func=cmd_candidate_pipeline)

    p_exp_summary = sub.add_parser("experiment-summary", help="summarize an experiment manifest")
    p_exp_summary.add_argument("--manifest", default=None)
    p_exp_summary.add_argument(
        "--refresh",
        action="store_true",
        help="recompute differential findings from stored normalized outputs or bug artifacts with the current oracle",
    )
    p_exp_summary.set_defaults(func=cmd_experiment_summary)

    p_exp_analysis = sub.add_parser(
        "analyze-experiment",
        help="compare experiment aggregate metrics against a reference preset",
    )
    p_exp_analysis.add_argument("--manifest", default=None)
    p_exp_analysis.add_argument("--reference-preset", default="baseline")
    p_exp_analysis.add_argument(
        "--baseline-preset",
        default=None,
        help="legacy alias for --reference-preset",
    )
    p_exp_analysis.add_argument(
        "--compare-presets",
        default=None,
        help="optional comma-separated preset subset to compare against the reference run",
    )
    p_exp_analysis.add_argument(
        "--refresh",
        action="store_true",
        help="recompute experiment summary findings with the current oracle before analysis",
    )
    p_exp_analysis.set_defaults(func=cmd_analyze_experiment)

    p_seeded_analysis = sub.add_parser(
        "analyze-seeded-sensitivity",
        help="analyze expected-root detection for seeded fault experiments",
    )
    p_seeded_analysis.add_argument("--manifest", default=None)
    p_seeded_analysis.set_defaults(func=cmd_analyze_seeded_sensitivity)

    p_ablation_audit = sub.add_parser(
        "analyze-ablation-audit",
        help="audit candidate families and false positives introduced by ablation presets",
    )
    p_ablation_audit.add_argument("--manifest", default=None)
    p_ablation_audit.add_argument(
        "--reference-presets",
        default=None,
        help="comma-separated presets treated as the reference soundness boundary",
    )
    p_ablation_audit.add_argument(
        "--trusted-presets",
        default=None,
        help="legacy alias for --reference-presets",
    )
    p_ablation_audit.add_argument(
        "--ablation-presets",
        default=None,
        help="comma-separated weakened presets whose candidates should not be counted without triage",
    )
    p_ablation_audit.add_argument(
        "--refresh",
        action="store_true",
        help="recompute experiment summary findings with the current oracle before auditing",
    )
    p_ablation_audit.set_defaults(func=cmd_analyze_ablation_audit)

    p_methodology_report = sub.add_parser(
        "methodology-report",
        help="write a paper-facing methodology report from an experiment manifest",
    )
    p_methodology_report.add_argument("--manifest", default=None)
    p_methodology_report.add_argument(
        "--refresh",
        action="store_true",
        help="recompute experiment summary findings with the current oracle before reporting",
    )
    p_methodology_report.add_argument(
        "--summary-only",
        action="store_true",
        help="reuse existing experiment-summary CSVs when available and skip run-log-derived report sections",
    )
    p_methodology_report.add_argument("--json", action="store_true", help="emit the generated report JSON")
    p_methodology_report.set_defaults(func=cmd_methodology_report)

    p_final_ready = sub.add_parser(
        "final-readiness",
        help="audit final experiment breadth, depth, replay policy, and bug evidence readiness",
    )
    p_final_ready.add_argument(
        "--manifest",
        action="append",
        default=[],
        help=(
            "experiment manifest to include; may be repeated; when omitted, defaults to the latest "
            f"{DEFAULT_FINAL_READINESS_MANIFEST_LIMIT} runs/experiment-*.json files"
        ),
    )
    p_final_ready.add_argument(
        "--extra-manifest",
        action="append",
        default=[],
        help=(
            "additional support manifest to audit alongside the default/latest run manifests; "
            "use this for reports/experiment-final-version-ledger.json"
        ),
    )
    p_final_ready.add_argument(
        "--manifest-index",
        action="append",
        default=[],
        help=(
            "JSON manifest index produced by scripts/run_final_experiments.py --execute; "
            "may be repeated and is used instead of sweeping stale runs/experiment-*.json files"
        ),
    )
    p_final_ready.add_argument(
        "--latest-manifests",
        type=int,
        default=DEFAULT_FINAL_READINESS_MANIFEST_LIMIT,
        help="number of most-recent experiment manifests to audit when --manifest is omitted",
    )
    p_final_ready.add_argument(
        "--all-manifests",
        action="store_true",
        help="audit every runs/experiment-*.json manifest when --manifest is omitted",
    )
    p_final_ready.add_argument(
        "--summary-only",
        action="store_true",
        help="skip run-log scans and audit only manifest/meta/confirmation metadata",
    )
    p_final_ready.add_argument(
        "--full-run-log-scan",
        action="store_true",
        help="scan run logs even when --manifest is omitted; required for a final paper readiness claim",
    )
    p_final_ready.add_argument(
        "--latest-confirmation-file",
        action="append",
        default=[],
        help=(
            "JSON file with upstream-confirmed latest bug families; may be repeated; "
            "defaults to experiments/latest_confirmations.json when present"
        ),
    )
    p_final_ready.add_argument("--min-live-cases-per-suite", type=int, default=1)
    p_final_ready.add_argument("--min-live-duration-hours", type=float, default=24.0)
    p_final_ready.add_argument("--min-live-candidate-families", type=int, default=1)
    p_final_ready.add_argument("--min-confirmed-live-families", type=int, default=1)
    p_final_ready.add_argument("--min-historical-confirmed", type=int, default=2)
    p_final_ready.add_argument(
        "--required-live-suites",
        default=",".join(DEFAULT_A_LEVEL_READINESS_POLICY.required_live_suites),
        help="comma-separated live target suites required by the top-level experiment policy",
    )
    p_final_ready.add_argument(
        "--required-live-families",
        default=",".join(DEFAULT_A_LEVEL_READINESS_POLICY.required_live_families),
        help="comma-separated backend families required by the top-level experiment policy",
    )
    p_final_ready.add_argument("--no-require-validation", action="store_true")
    p_final_ready.add_argument("--no-require-seeded", action="store_true")
    p_final_ready.add_argument("--no-require-ablation", action="store_true")
    p_final_ready.add_argument("--no-require-comparison", action="store_true")
    p_final_ready.add_argument("--no-require-adaptive-component-ablation", action="store_true")
    p_final_ready.add_argument(
        "--min-adaptive-component-ablations",
        type=int,
        default=DEFAULT_FINAL_READINESS_THRESHOLDS.min_adaptive_component_ablations,
    )
    p_final_ready.add_argument(
        "--required-adaptive-component-ablations",
        default=",".join(DEFAULT_FINAL_READINESS_THRESHOLDS.required_adaptive_component_ablations),
        help=(
            "comma-separated adaptive components that must each have an ablation contrast; "
            "hyphenated aliases are accepted"
        ),
    )
    p_final_ready.add_argument("--no-require-transferability-scope", action="store_true")
    p_final_ready.add_argument(
        "--min-transfer-target-families",
        type=int,
        default=DEFAULT_FINAL_READINESS_THRESHOLDS.min_transfer_target_families,
    )
    p_final_ready.add_argument("--no-require-cross-version-ledger", action="store_true")
    p_final_ready.add_argument(
        "--min-cross-version-ledger-versions",
        type=int,
        default=DEFAULT_FINAL_READINESS_THRESHOLDS.min_cross_version_ledger_versions,
    )
    p_final_ready.add_argument(
        "--min-cross-version-ledger-families",
        type=int,
        default=DEFAULT_FINAL_READINESS_THRESHOLDS.min_cross_version_ledger_families,
    )
    p_final_ready.add_argument("--no-require-cross-version-health-feedback", action="store_true")
    p_final_ready.add_argument("--no-require-runtime-efficiency", action="store_true")
    p_final_ready.add_argument(
        "--min-throughput-cases-s",
        type=float,
        default=DEFAULT_FINAL_READINESS_THRESHOLDS.min_throughput_cases_s,
    )
    p_final_ready.add_argument(
        "--max-scheduler-feedback-share",
        type=float,
        default=DEFAULT_FINAL_READINESS_THRESHOLDS.max_scheduler_feedback_share,
    )
    p_final_ready.add_argument(
        "--min-scheduler-feedback-cases",
        type=int,
        default=DEFAULT_FINAL_READINESS_THRESHOLDS.min_scheduler_feedback_cases,
        help="minimum executed cases before scheduler feedback share is enforced for a run",
    )
    p_final_ready.add_argument("--no-require-discovery-responsiveness", action="store_true")
    p_final_ready.add_argument(
        "--max-first-candidate-elapsed-s",
        type=float,
        default=DEFAULT_FINAL_READINESS_THRESHOLDS.max_first_candidate_elapsed_s,
    )
    p_final_ready.add_argument("--no-require-closed-loop-state-persistence", action="store_true")
    p_final_ready.add_argument("--no-require-adaptive-live-component-evidence", action="store_true")
    p_final_ready.add_argument("--json", action="store_true", help="emit the generated readiness JSON")
    p_final_ready.add_argument(
        "--fail-on-missing",
        action="store_true",
        help="exit with code 2 when any required final-readiness gate is not satisfied",
    )
    p_final_ready.set_defaults(func=cmd_final_readiness)

    p_review_ready = sub.add_parser(
        "review-readiness",
        help="audit ISCE-style review readiness from lightweight repository and bug evidence",
    )
    p_review_ready.add_argument("--json", action="store_true", help="emit machine-readable readiness JSON")
    p_review_ready.add_argument("--write-report", action="store_true", help="write reports/review-readiness-*.json and .md")
    p_review_ready.add_argument("--output-dir", default="reports", help="directory for --write-report output")
    p_review_ready.add_argument(
        "--latest-confirmation-file",
        action="append",
        default=[],
        help="latest confirmation JSON file; may be repeated; defaults to experiments/latest_confirmations.json",
    )
    p_review_ready.add_argument("--target-confirmed", type=int, default=20)
    p_review_ready.add_argument("--min-audit-candidates", type=int, default=1)
    p_review_ready.add_argument("--min-discovery-workflows", type=int, default=1)
    p_review_ready.add_argument("--min-generated-issue-drafts", type=int, default=1)
    p_review_ready.add_argument("--min-issue-bundle-families", type=int, default=1)
    p_review_ready.add_argument("--min-pending-issue-drafts", type=int, default=1)
    p_review_ready.add_argument("--min-old-known-issues", type=int, default=1)
    p_review_ready.add_argument(
        "--fail-on-missing",
        action="store_true",
        help="exit with code 2 when any required review gate is not satisfied",
    )
    p_review_ready.set_defaults(func=cmd_review_readiness)

    p_pattern_variants = sub.add_parser(
        "analyze-pattern-variants",
        help="analyze generated pattern variants and candidate findings in an experiment",
    )
    p_pattern_variants.add_argument("--manifest", default=None)
    p_pattern_variants.add_argument(
        "--pattern",
        choices=["null_agg_topk"],
        default="null_agg_topk",
    )
    p_pattern_variants.set_defaults(func=cmd_analyze_pattern_variants)

    p_show = sub.add_parser("show-bugs", help="print bug findings")
    p_show.add_argument("--run-file", default=None)
    p_show.add_argument("--limit", type=int, default=10)
    p_show.set_defaults(func=cmd_show_bugs)

    p_classify = sub.add_parser("classify-run", help="classify findings as bugs, semantic divergences, or false positives")
    p_classify.add_argument("--run-file", default=None)
    p_classify.add_argument("--limit", type=int, default=3, help="examples per verdict")
    p_classify.add_argument(
        "--refresh",
        action="store_true",
        help="recompute differential findings from stored normalized outputs with the current oracle",
    )
    p_classify.add_argument("--json", action="store_true", help="emit machine-readable classification summary")
    p_classify.set_defaults(func=cmd_classify_run)

    p_health = sub.add_parser("run-health", help="summarize an in-progress or completed run log")
    p_health.add_argument("--run-file", default=None)
    p_health.add_argument("--limit", type=int, default=3, help="candidate examples to show")
    p_health.add_argument("--json", action="store_true", help="emit machine-readable health summary")
    p_health.add_argument(
        "--fail-on-fresh-candidate",
        action="store_true",
        help="exit with code 2 when the run contains an unsaturated organic candidate family",
    )
    p_health.add_argument("--fail-on-bug", action="store_true", help="exit with code 2 when any row has status=bug")
    p_health.set_defaults(func=cmd_run_health)

    p_repro = sub.add_parser("reproduce", help="show reproduce command for a bug artifact")
    p_repro.add_argument("--bug", required=True)
    p_repro.add_argument("--backends", default=None)
    p_repro.add_argument("--print-command", action="store_true")
    p_repro.set_defaults(func=cmd_reproduce)

    p_validate = sub.add_parser("validate-artifact", help="rerun a bug artifact and check finding preservation")
    p_validate.add_argument("--bug", required=True)
    p_validate.add_argument("--backends", default=None)
    p_validate.set_defaults(func=cmd_validate_artifact)

    p_triage = sub.add_parser("triage-artifact", help="classify a reproduced artifact for paper use")
    p_triage.add_argument("--bug", required=True)
    p_triage.add_argument("--backends", default=None)
    p_triage.add_argument("--reduce", action="store_true")
    p_triage.add_argument(
        "--reduce-ignore-roots",
        action="store_true",
        help="when reducing, preserve finding kind only instead of the original root-cause label",
    )
    p_triage.add_argument(
        "--standalone-reproducer",
        action="store_true",
        help="also write an optional standalone diagnostic script when this root cause is supported",
    )
    p_triage.set_defaults(func=cmd_triage_artifact)

    p_reduce = sub.add_parser("reduce", help="minimize a bug artifact while preserving findings")
    p_reduce.add_argument("--bug", required=True)
    p_reduce.add_argument("--backends", default="pandas,polars,duckdb,sqlite")
    p_reduce.add_argument(
        "--ignore-roots",
        action="store_true",
        help="preserve finding kind only instead of the original root-cause label",
    )
    p_reduce.set_defaults(func=cmd_reduce)

    p_hist = sub.add_parser("historical-status", help="show historical replay registry admission status")
    p_hist.add_argument(
        "--include-pending",
        action="store_true",
        help="include candidate and pending historical case studies",
    )
    p_hist.add_argument("--json", action="store_true", help="emit historical registry status as JSON")
    p_hist.set_defaults(func=cmd_historical_status)

    p_fixture = sub.add_parser(
        "replay-fixture",
        help="run a declared fixture-backed case through normal differential oracle and journal recording",
    )
    p_fixture.add_argument("--spec", required=True, help="fixture replay spec JSON")
    p_fixture.add_argument("--fixture", default=None, help="path to the external fixture data file")
    p_fixture.add_argument(
        "--fixture-env",
        default=None,
        help="environment variable containing the external fixture data file path",
    )
    p_fixture.add_argument("--backends", default=None, help="explicit comma-separated backend targets")
    p_fixture.add_argument(
        "--target-suite",
        choices=sorted(TARGET_SUITES),
        default=None,
        help="backend target suite used when --backends is not provided; defaults to the spec target_suite",
    )
    p_fixture.add_argument(
        "--evidence-mode",
        choices=list(FIXTURE_REPLAY_EVIDENCE_MODES),
        default="historical",
        help="paper evidence layer for this replay",
    )
    p_fixture.add_argument("--known-bug-id", default="", help="historical bug id recorded in the run journal")
    p_fixture.add_argument("--target-version", default="", help="target dependency version or commit under replay")
    p_fixture.add_argument("--run-theme", default="", help="short paper-facing run theme")
    p_fixture.add_argument("--paper-notes", default="", help="brief paper-facing run notes")
    p_fixture.add_argument(
        "--experiment-meta",
        default="",
        help="JSON object describing structured experiment metadata for this fixture replay",
    )
    add_ablation_flags(p_fixture)
    p_fixture.set_defaults(func=cmd_replay_fixture)

    p_exp = sub.add_parser("experiment", help="run repeatable ablation experiment matrix")
    p_exp.add_argument("--cases", type=int, default=None, help="maximum cases; defaults to 100 when --duration is absent")
    p_exp.add_argument("--duration", default=None, help="optional per-run wall-clock budget such as 10s, 5m, 24h")
    p_exp.add_argument("--seeds", default="1,1001,2001")
    p_exp.add_argument(
        "--evidence-mode",
        choices=["auto", *EXPERIMENT_EVIDENCE_MODES],
        default="auto",
        help=(
            "experiment evidence layer: validation smoke, live latest-version finding, "
            "historical fixed-bug replay, seeded fault sensitivity, module ablation, "
            "or baseline/related-scope comparison"
        ),
    )
    p_exp.add_argument(
        "--known-bug-id",
        default="",
        help="identifier for a replayed historical/upstream bug when --evidence-mode=historical",
    )
    p_exp.add_argument(
        "--target-version",
        default="",
        help="target dependency version or commit used by a historical replay run",
    )
    p_exp.add_argument(
        "--version-pair-pool",
        default="",
        help="comma-separated target-version pairs for adaptive per-case cross-version selection",
    )
    p_exp.add_argument(
        "--fixed-version",
        default="",
        help="fixed dependency version or commit paired with --target-version for cross-version learning",
    )
    add_target_suite_flags(p_exp)
    p_exp.add_argument(
        "--target-suites",
        default=None,
        help="comma-separated backend target suites to run as an extra experiment dimension",
    )
    p_exp.add_argument(
        "--artifact-limit",
        type=int,
        default=None,
        help="maximum bug artifact directories to write for each preset run",
    )
    p_exp.add_argument("--log-level", choices=["full", "compact", "minimal"], default="compact")
    p_exp.add_argument(
        "--jobs",
        type=_parse_jobs,
        default="auto",
        help="number of experiment matrix runs to execute in parallel, or 'auto'",
    )
    p_exp.add_argument(
        "--max-parallel-cost",
        type=float,
        default=None,
        help="cost-token budget for concurrently running experiment jobs; defaults to a CPU-based budget",
    )
    p_exp.add_argument(
        "--schedule",
        choices=["matrix_order", "longest_first", "adaptive"],
        default=None,
        help="experiment scheduler; adaptive shares the matrix budget across runs",
    )
    p_exp.add_argument(
        "--batch-cases",
        type=int,
        default=None,
        help="adaptive-schedule batch size in cases; ignored by static schedules",
    )
    p_exp.add_argument(
        "--batch-duration",
        default=None,
        help="adaptive-schedule batch wall-clock budget such as 30s; ignored by static schedules",
    )
    p_exp.add_argument(
        "--warmup-batches",
        type=int,
        default=1,
        help="minimum adaptive batches to allocate to each arm before exploitation",
    )
    p_exp.add_argument(
        "--exploration-weight",
        type=float,
        default=0.75,
        help="adaptive scheduler exploration weight",
    )
    p_exp.add_argument(
        "--group-fairness-weight",
        type=float,
        default=0.40,
        help="adaptive scheduler bonus for underrepresented target-suite/preset groups",
    )
    p_exp.add_argument(
        "--max-group-pull-gap",
        type=int,
        default=3,
        help="adaptive scheduler rebalances once a target-suite/preset group trails by this many pulls",
    )
    p_exp.add_argument(
        "--adaptive-learning-weight",
        type=float,
        default=0.0,
        help="adaptive scheduler contextual-learning score weight; 0 keeps legacy adaptive scheduling",
    )
    p_exp.add_argument(
        "--scheduler-annealing-temperature",
        type=float,
        default=0.0,
        help="initial adaptive scheduler annealing temperature; 0 keeps deterministic greedy selection",
    )
    p_exp.add_argument(
        "--scheduler-annealing-decay",
        type=float,
        default=0.985,
        help="per-completed-batch decay for adaptive scheduler annealing temperature",
    )
    p_exp.add_argument(
        "--scheduler-annealing-min-temperature",
        type=float,
        default=0.02,
        help="minimum nonzero adaptive scheduler annealing temperature",
    )
    p_exp.add_argument(
        "--continual-learning-ledgers",
        default="",
        help="comma-separated version-ledger JSON files used to cold-start adaptive continual-learning priority",
    )
    p_exp.add_argument(
        "--disable-adaptive-components",
        type=_parse_adaptive_components,
        default="",
        help=(
            "comma-separated adaptive components to disable for ablation: "
            + ",".join(ADAPTIVE_COMPONENTS)
        ),
    )
    p_exp.add_argument(
        "--enable-local-source-scheduler",
        action="store_true",
        help="enable within-run generated-vs-feedback source scheduling for non-adaptive experiment jobs",
    )
    p_exp.add_argument(
        "--local-source-exploration-weight",
        type=float,
        default=0.5,
        help="exploration weight for the within-run generated-vs-feedback source scheduler",
    )
    p_exp.add_argument(
        "--metamorphic-variant-limit",
        type=int,
        default=None,
        help="maximum metamorphic variants to execute per base case",
    )
    p_exp.add_argument(
        "--enable-replay-bug",
        action="store_true",
        help="allow submitted or historical issue replay cases in experiment presets",
    )
    p_exp.add_argument(
        "--replay-bug-source-issues",
        default="",
        help="comma-separated upstream issue URLs treated as known replay bugs in fresh experiment mode",
    )
    p_exp.add_argument("--no-compress-run-log", action="store_true")
    p_exp.add_argument(
        "--persist-closed-loop-state",
        action="store_true",
        help="write a resumable closed-loop learning state file for each experiment run",
    )
    p_exp.add_argument(
        "--skip-run-reports",
        action="store_true",
        help="do not write per-run markdown/csv reports during the matrix; use experiment-summary after completion",
    )
    p_exp.add_argument(
        "--presets",
        default="baseline,no_type_aware,no_normalizer,no_feedback,metamorphic,reducer",
        help="comma-separated presets",
    )
    p_exp.add_argument(
        "--experiment-meta",
        default="",
        help="JSON object describing structured experiment catalog metadata for the whole matrix",
    )
    add_paper_journal_flags(p_exp)
    p_exp.set_defaults(func=cmd_experiment)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
