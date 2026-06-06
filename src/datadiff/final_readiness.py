from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from datadiff.config import ExperimentConfig
from datadiff.experiment_catalog import (
    FINAL_ADAPTIVE_COMPONENT_ABLATION_MATRIX,
    FINAL_COMPARISON_MATRIX,
    FINAL_MODULE_ABLATION_MATRIX,
    FINAL_LIVE_DISCOVERY_MATRIX,
    FINAL_PROTOCOL_TRACKS,
    FINAL_SEEDED_SENSITIVITY_MATRIX,
    FINAL_VALIDATION_MATRIX,
    registered_experiment_matrix_for_run,
    replay_bug_enabled_by_default,
)
from datadiff.experiment_metadata import (
    experiment_row_group_id,
    experiment_row_variant_label,
    is_ablation_experiment_row,
    is_comparison_experiment_row,
    is_seeded_experiment_row,
    manifest_experiment_meta,
    resolved_run_semantics,
)
from datadiff.finding_outcomes import (
    candidate_issue_family_key,
    is_issue_replay_finding,
    is_known_saturated_candidate_issue_finding,
    is_rewardable_candidate_issue_finding,
)
from datadiff.historical import list_historical_bugs
from datadiff.icse_experiment_quality import score_final_readiness_summary
from datadiff.run_provenance import current_workspace_git_commit
from datadiff.runner import _configured_guidance_targets
from datadiff.target_version_audit import TARGET_VERSION_AUDIT_SCHEMA_VERSION
from datadiff.targets import TARGETS, TARGET_SUITES, target_context
from datadiff.util import REPORTS_DIR, RUNS_DIR, dump_json, ensure_dirs, iter_jsonl, load_json, read_jsonl, run_meta_path, utc_now

DEFAULT_LATEST_CONFIRMATIONS_FILE = Path("experiments/latest_confirmations.json")
DEFAULT_FINAL_READINESS_MANIFEST_LIMIT = 25
FINAL_READINESS_SCHEMA_VERSION = "final-readiness-v1"
DEFAULT_PAPER_RUN_JOURNAL_NAME = "paper-run-journal.jsonl"
ADAPTIVE_LIVE_EVIDENCE_COMPONENTS = frozenset(
    {
        "scheduler_learning",
        "online_reward_model",
        "active_learning",
        "continual_learning",
        "backend_pair_learning",
        "value_catalog",
        "quality_archive",
        "hierarchical_archive",
        "bd_axis_bandit",
        "bayesian_exploration",
        "seed_quota",
        "seed_energy_tier",
        "lhs_seeding",
        "champion_corpus",
        "champion_graft_donor",
        "runtime_cost_learning",
        "scheduler_annealing",
    }
)
CONFIRMED_LATEST_UPSTREAM_STATUSES = frozenset(
    {
        "upstream_labeled_bug",
        "maintainer_confirmed_bug",
        "confirmed_bug",
        "fixed_upstream",
    }
)


def _default_required_live_suites() -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(campaign.suite for campaign in FINAL_LIVE_DISCOVERY_MATRIX.campaigns)
    )


def _default_required_live_families() -> tuple[str, ...]:
    suites = _default_required_live_suites()
    families = {
        family
        for suite in suites
        for family in target_context(target_suite=suite).families
    }
    return tuple(sorted(families))

# Layering: the audit engine below is middle-layer analysis over manifests and
# run logs. A-level requirements are supplied as policy data and never alter the
# bottom-layer generator, runner, oracle, normalizer, or backend adapters.
@dataclass(frozen=True, slots=True)
class ReadinessPolicy:
    required_live_suites: tuple[str, ...] = _default_required_live_suites()
    required_live_families: tuple[str, ...] = _default_required_live_families()
    required_transfer_families: tuple[str, ...] = ("dataframe", "embedded_sql")
    primary_method_family: str = "dataframe"
    required_final_matrix_ids: tuple[str, ...] = (
        FINAL_VALIDATION_MATRIX.id,
        FINAL_LIVE_DISCOVERY_MATRIX.id,
        "historical_replay",
        FINAL_SEEDED_SENSITIVITY_MATRIX.id,
        FINAL_MODULE_ABLATION_MATRIX.id,
        FINAL_ADAPTIVE_COMPONENT_ABLATION_MATRIX.id,
        FINAL_COMPARISON_MATRIX.id,
    )
    confirmed_live_paper_statuses: tuple[str, ...] = (
        "confirmed_bug",
        "confirmed_implementation_bug",
        "fixed_upstream",
        "maintainer_confirmed_bug",
    )


DEFAULT_A_LEVEL_READINESS_POLICY = ReadinessPolicy()
EVIDENCE_MODES = frozenset(FINAL_PROTOCOL_TRACKS)


@dataclass(frozen=True, slots=True)
class ReadinessThresholds:
    min_live_cases_per_suite: int = 1
    min_live_duration_hours: float = 24.0
    min_live_candidate_families: int = 1
    min_confirmed_live_families: int = 1
    min_historical_confirmed: int = 2
    require_validation: bool = True
    require_seeded: bool = True
    require_ablation: bool = True
    require_comparison: bool = True
    require_adaptive_component_ablation: bool = True
    min_adaptive_component_ablations: int = 1
    required_adaptive_component_ablations: tuple[str, ...] = (
        "scheduler_learning",
        "online_reward_model",
        "backend_pair_learning",
        "continual_learning",
        "active_learning",
        "ir_rewrite_mutations",
        "operator_swarm",
        "divergence_conditioned",
        "shrink_mutations",
        "value_catalog",
        "quality_archive",
        "hierarchical_archive",
        "bd_axis_bandit",
        "bayesian_exploration",
        "seed_quota",
        "seed_energy_batch",
        "seed_energy_tier",
        "per_operator_energy",
        "lineage_rarity",
        "minhash_dedup",
        "disagreement_bd_axis",
        "lhs_seeding",
        "champion_corpus",
        "champion_graft_donor",
        "runtime_cost_learning",
        "cost_normalized_reward",
        "scheduler_annealing",
    )
    require_transferability_scope: bool = True
    min_transfer_target_families: int = 2
    require_cross_version_ledger: bool = True
    min_cross_version_ledger_versions: int = 2
    min_cross_version_ledger_families: int = 1
    require_cross_version_health_feedback: bool = True
    require_runtime_efficiency: bool = True
    min_throughput_cases_s: float = 0.001
    max_scheduler_feedback_share: float = 0.50
    min_scheduler_feedback_cases: int = 10
    require_discovery_responsiveness: bool = True
    max_first_candidate_elapsed_s: float = 6 * 3600.0
    require_closed_loop_state_persistence: bool = True
    require_adaptive_live_component_evidence: bool = True
    require_target_version_audit: bool = True


STAGE_PROFILE_FIELDS: tuple[str, ...] = (
    "generate_mutate_ms",
    "backend_execution_ms",
    "normalize_ms",
    "oracle_classification_ms",
    "scheduler_feedback_ms",
    "logging_artifact_ms",
    "total_case_wall_ms",
)


def analyze_final_readiness(
    manifest_files: list[Path] | None = None,
    *,
    extra_manifest_files: list[Path] | None = None,
    latest_confirmation_files: list[Path] | None = None,
    paper_run_journal_files: list[Path] | None = None,
    manifest_limit: int | None = DEFAULT_FINAL_READINESS_MANIFEST_LIMIT,
    scan_run_logs: bool = True,
    thresholds: ReadinessThresholds | None = None,
    policy: ReadinessPolicy | None = None,
) -> tuple[Path, Path]:
    ensure_dirs()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    thresholds = thresholds or ReadinessThresholds()
    policy = policy or DEFAULT_A_LEVEL_READINESS_POLICY
    manifest_files = _resolve_manifest_files(
        manifest_files,
        manifest_limit=manifest_limit,
        extra_manifest_files=extra_manifest_files,
    )
    latest_confirmation_files = _resolve_latest_confirmation_files(latest_confirmation_files)
    paper_run_journal_files = _resolve_paper_run_journal_files(paper_run_journal_files, manifest_files)
    audit = build_final_readiness(
        manifest_files,
        latest_confirmation_files=latest_confirmation_files,
        paper_run_journal_files=paper_run_journal_files,
        scan_run_logs=scan_run_logs,
        thresholds=thresholds,
        policy=policy,
    )
    stamp = utc_now().replace(":", "").replace("-", "").replace("Z", "")
    md_path = REPORTS_DIR / f"final-readiness-{stamp}.md"
    json_path = REPORTS_DIR / f"final-readiness-{stamp}.json"
    dump_json(audit, json_path)
    md_path.write_text(_render_markdown(audit), encoding="utf-8")
    return md_path, json_path


def build_final_readiness(
    manifest_files: list[Path],
    *,
    latest_confirmation_files: list[Path] | None = None,
    paper_run_journal_files: list[Path] | None = None,
    scan_run_logs: bool = True,
    thresholds: ReadinessThresholds,
    policy: ReadinessPolicy | None = None,
) -> dict[str, Any]:
    policy = policy or DEFAULT_A_LEVEL_READINESS_POLICY
    manifests = [_load_manifest(path) for path in manifest_files]
    latest_confirmations = _load_latest_confirmations(latest_confirmation_files or [])
    paper_run_journal_files = _resolve_paper_run_journal_files(paper_run_journal_files, manifest_files)
    paper_run_journal_entries = _load_paper_run_journal_entries(paper_run_journal_files)
    paper_run_journal_run_files = _paper_run_journal_run_file_keys(paper_run_journal_entries)
    runs = [
        run
        for manifest in manifests
        for run in _manifest_runs(manifest, policy=policy, scan_run_logs=scan_run_logs)
    ]
    for run in runs:
        run["paper_run_journal_present"] = _run_file_key(run.get("run_file", "")) in paper_run_journal_run_files
    validation_runs = [run for run in runs if run["evidence_mode"] == "validation"]
    explicit_live_runs = [run for run in runs if run["evidence_mode"] == "live"]
    live_runs = [run for run in explicit_live_runs if _fresh_replay_policy_ok(run)]
    replay_policy_rejected_live_runs = [run for run in explicit_live_runs if not _fresh_replay_policy_ok(run)]
    historical_runs = [run for run in runs if run["evidence_mode"] == "historical"]
    seeded_runs = [run for run in runs if run["evidence_mode"] == "seeded"]
    ablation_runs = [run for run in runs if _is_module_ablation_run(run)]
    comparison_runs = [run for run in runs if _is_contrast_scope_run(run)]
    module_ablation_comparison = _support_track_comparison_summary(ablation_runs)
    baseline_comparison = _support_track_comparison_summary(comparison_runs)
    adaptive_component_ablation = _adaptive_component_ablation_summary(runs)
    ignored_runs = [run for run in runs if run["evidence_mode"] not in EVIDENCE_MODES]
    audited_runs = [run for run in runs if run["evidence_mode"] in EVIDENCE_MODES]
    paper_run_journal_missing_runs = _paper_run_journal_issues(audited_runs)
    paper_run_journal_covered_runs = sum(1 for run in audited_runs if bool(run.get("paper_run_journal_present")))

    live_by_suite = _live_suite_summary(live_runs)
    live_families = sorted({family for run in live_runs for family in run["target_families"]})
    final_matrix_coverage = _final_matrix_coverage_summary(audited_runs, policy=policy)
    transferability_scope = _transferability_scope_summary(audited_runs, policy=policy)
    cross_version_ledger = _cross_version_ledger_summary(audited_runs)
    runtime_efficiency = _runtime_efficiency_summary(
        [run for run in audited_runs if run["evidence_mode"] != "historical"],
        thresholds=thresholds,
    )
    discovery_responsiveness = _discovery_responsiveness_summary(live_runs)
    closed_loop_state_persistence = _closed_loop_state_persistence_summary(live_runs)
    adaptive_live_component_evidence = _adaptive_live_component_evidence_summary(live_runs)
    continual_learning_absorption = _continual_learning_absorption_summary(audited_runs)
    target_version_audit = _target_version_audit_summary(manifests)
    rewardable_live_families = Counter()
    confirmed_live_families = Counter()
    external_confirmed_live_families = _confirmed_latest_family_counter(latest_confirmations)
    confirmed_live_families.update(external_confirmed_live_families)
    issue_replay_live_families = Counter()
    known_saturated_live_families = Counter()
    for run in live_runs:
        rewardable_live_families.update(run["rewardable_candidate_families"])
        confirmed_live_families.update(run["confirmed_candidate_families"])
        issue_replay_live_families.update(run["issue_replay_candidate_families"])
        known_saturated_live_families.update(run["known_saturated_candidate_families"])

    historical_confirmed = _historical_confirmed_runs(historical_runs)
    workspace_git_commit = current_workspace_git_commit()
    gates = _readiness_gates(
        thresholds=thresholds,
        live_by_suite=live_by_suite,
        live_families=live_families,
        audited_runs=audited_runs,
        validation_runs=validation_runs,
        live_runs=live_runs,
        historical_runs=historical_runs,
        historical_confirmed=historical_confirmed,
        seeded_runs=seeded_runs,
        ablation_runs=ablation_runs,
        comparison_runs=comparison_runs,
        module_ablation_comparison=module_ablation_comparison,
        baseline_comparison=baseline_comparison,
        scan_run_logs=scan_run_logs,
        rewardable_live_families=rewardable_live_families,
        confirmed_live_families=confirmed_live_families,
        paper_run_journal_files=paper_run_journal_files,
        paper_run_journal_covered_runs=paper_run_journal_covered_runs,
        paper_run_journal_issues=paper_run_journal_missing_runs,
        policy=policy,
        workspace_git_commit=workspace_git_commit,
        adaptive_component_ablation=adaptive_component_ablation,
        final_matrix_coverage=final_matrix_coverage,
        transferability_scope=transferability_scope,
        cross_version_ledger=cross_version_ledger,
        runtime_efficiency=runtime_efficiency,
        discovery_responsiveness=discovery_responsiveness,
        closed_loop_state_persistence=closed_loop_state_persistence,
        adaptive_live_component_evidence=adaptive_live_component_evidence,
        continual_learning_absorption=continual_learning_absorption,
        target_version_audit=target_version_audit,
    )
    audit = {
        "schema_version": FINAL_READINESS_SCHEMA_VERSION,
        "created_at": utc_now(),
        "ready": all(gate["passed"] for gate in gates),
        "thresholds": asdict(thresholds),
        "policy": asdict(policy),
        "manifest_files": [str(path) for path in manifest_files],
        "latest_confirmation_files": [str(path) for path in latest_confirmation_files or []],
        "paper_run_journal_files": [str(path) for path in paper_run_journal_files],
        "workspace_revision": {
            "git_commit": workspace_git_commit,
        },
        "gates": gates,
        "summary": {
            "live_runs": len(live_runs),
            "validation_runs": len(validation_runs),
            "validation_cases": sum(int(run["cases"]) for run in validation_runs),
            "validation_suites": sorted({run["target_suite"] for run in validation_runs}),
            "explicit_live_runs": len(explicit_live_runs),
            "replay_policy_rejected_live_runs": len(replay_policy_rejected_live_runs),
            "historical_runs": len(historical_runs),
            "seeded_runs": len(seeded_runs),
            "ablation_runs": len(ablation_runs),
            "ablation_cases": sum(int(run["cases"]) for run in ablation_runs),
            "ablation_suites": sorted({run["target_suite"] for run in ablation_runs}),
            "ablation_presets": sorted({run["preset"] for run in ablation_runs}),
            "ablation_variants": sorted({run["variant_label"] for run in ablation_runs if run.get("variant_label")}),
            "ablation_matrix_ids": _sorted_matrix_ids(ablation_runs),
            "module_ablation_comparison": module_ablation_comparison,
            "adaptive_component_ablation": adaptive_component_ablation,
            "comparison_runs": len(comparison_runs),
            "comparison_cases": sum(int(run["cases"]) for run in comparison_runs),
            "comparison_suites": sorted({run["target_suite"] for run in comparison_runs}),
            "comparison_presets": sorted({run["preset"] for run in comparison_runs}),
            "comparison_variants": sorted(
                {run["variant_label"] for run in comparison_runs if run.get("variant_label")}
            ),
            "comparison_matrix_ids": _sorted_matrix_ids(comparison_runs),
            "baseline_comparison": baseline_comparison,
            "ignored_evidence_runs": len(ignored_runs),
            "run_logs_scanned": sum(1 for run in runs if run.get("run_log_scanned")),
            "run_logs_scan_skipped": sum(1 for run in runs if run.get("run_log_scan_skipped")),
            "paper_run_journal_entries": len(paper_run_journal_entries),
            "paper_run_journal_required_runs": len(audited_runs),
            "paper_run_journal_covered_runs": paper_run_journal_covered_runs,
            "paper_run_journal_missing_run_count": len(paper_run_journal_missing_runs),
            "paper_run_journal_missing_runs": paper_run_journal_missing_runs,
            "live_suites": sorted(live_by_suite),
            "live_families": live_families,
            "final_matrix_coverage": final_matrix_coverage,
            "transferability_scope": transferability_scope,
            "cross_version_ledger": cross_version_ledger,
            "continual_learning_absorption": continual_learning_absorption,
            "target_version_audit": target_version_audit,
            "runtime_efficiency": runtime_efficiency,
            "discovery_responsiveness": discovery_responsiveness,
            "closed_loop_state_persistence": closed_loop_state_persistence,
            "adaptive_live_component_evidence": adaptive_live_component_evidence,
            "semantic_focus_families": sorted(
                {
                    family
                    for run in audited_runs
                    for family in run.get("semantic_focus_families", [])
                    if str(family).strip()
                }
            ),
            "semantic_focus_signals": sorted(
                {
                    signal
                    for run in audited_runs
                    for signal in run.get("semantic_focus_signals", [])
                    if str(signal).strip()
                }
            ),
            "rewardable_live_candidate_families": dict(sorted(rewardable_live_families.items())),
            "confirmed_live_candidate_families": dict(sorted(confirmed_live_families.items())),
            "external_confirmed_live_candidate_families": dict(
                sorted(external_confirmed_live_families.items())
            ),
            "issue_replay_live_candidate_families": dict(sorted(issue_replay_live_families.items())),
            "known_saturated_live_candidate_families": dict(sorted(known_saturated_live_families.items())),
            "historical_confirmed_bug_ids": sorted(historical_confirmed),
            "total_live_cases": sum(item["cases"] for item in live_by_suite.values()),
            "total_live_elapsed_s": sum(item["elapsed_s"] for item in live_by_suite.values()),
        },
        "live_suites": live_by_suite,
        "runs": audited_runs,
        "latest_confirmations": latest_confirmations,
        "ignored_runs": ignored_runs,
        "replay_policy_rejected_live_runs": replay_policy_rejected_live_runs,
    }
    icse_quality = score_final_readiness_summary(audit["summary"])
    audit["summary"]["icse_experiment_quality"] = icse_quality
    audit["icse_experiment_quality"] = icse_quality
    return audit


def _resolve_manifest_files(
    manifest_files: list[Path] | None,
    *,
    manifest_limit: int | None = DEFAULT_FINAL_READINESS_MANIFEST_LIMIT,
    extra_manifest_files: list[Path] | None = None,
) -> list[Path]:
    extras = [Path(path) for path in extra_manifest_files or []]
    if manifest_files:
        return list(dict.fromkeys([*(Path(path) for path in manifest_files), *extras]))
    manifests = sorted(RUNS_DIR.glob("experiment-*.json"))
    if manifest_limit is None:
        selected = manifests
    else:
        limit = max(1, int(manifest_limit))
        selected = sorted(manifests, key=lambda path: (path.stat().st_mtime_ns, path.name))[-limit:]
    return list(dict.fromkeys([*selected, *extras]))


def _resolve_latest_confirmation_files(latest_confirmation_files: list[Path] | None) -> list[Path]:
    if latest_confirmation_files:
        return [Path(path) for path in latest_confirmation_files]
    return [DEFAULT_LATEST_CONFIRMATIONS_FILE] if DEFAULT_LATEST_CONFIRMATIONS_FILE.is_file() else []


def _resolve_paper_run_journal_files(
    paper_run_journal_files: list[Path] | None,
    manifest_files: list[Path],
) -> list[Path]:
    if paper_run_journal_files:
        return [Path(path) for path in paper_run_journal_files]
    candidates: list[Path] = []
    for manifest_path in manifest_files:
        candidate = Path(manifest_path).parent.parent / "reports" / DEFAULT_PAPER_RUN_JOURNAL_NAME
        if candidate.is_file():
            candidates.append(candidate)
    return list(dict.fromkeys(candidates))


def _load_manifest(path: Path) -> dict[str, Any]:
    data = load_json(path)
    data["_manifest_file"] = str(path)
    return data


def _load_latest_confirmations(paths: list[Path]) -> list[dict[str, Any]]:
    confirmations: list[dict[str, Any]] = []
    for path in paths:
        if not path.is_file():
            continue
        data = load_json(path)
        raw_items = data.get("confirmations", data) if isinstance(data, dict) else data
        if not isinstance(raw_items, list):
            continue
        for raw_item in raw_items:
            if not isinstance(raw_item, dict):
                continue
            item = dict(raw_item)
            item["_confirmation_file"] = str(path)
            confirmations.append(item)
    return confirmations


def _load_paper_run_journal_entries(paths: list[Path]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for path in paths:
        if not path.is_file():
            continue
        for row in iter_jsonl(path):
            if isinstance(row, dict):
                entries.append(dict(row))
    return entries


def _run_file_key(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    path = Path(text).expanduser()
    return str(path.resolve(strict=False)) if path.is_absolute() else str(path)


def _paper_run_journal_run_file_keys(entries: list[dict[str, Any]]) -> set[str]:
    return {
        key
        for entry in entries
        if isinstance(entry, dict)
        for key in (_run_file_key(entry.get("run_file", "")),)
        if key
    }


def _paper_run_journal_issues(runs: list[dict[str, Any]]) -> list[str]:
    issues: list[str] = []
    for run in runs:
        if _is_postprocess_evidence_run(run):
            continue
        if bool(run.get("paper_run_journal_present")):
            continue
        issues.append(f"{run['evidence_mode']}:{run['target_suite']}:{run['preset']}")
    return sorted(set(issues))


def _manifest_runs(
    manifest: dict[str, Any],
    *,
    policy: ReadinessPolicy,
    scan_run_logs: bool,
) -> list[dict[str, Any]]:
    manifest_mode = _normalize_evidence_mode(manifest.get("evidence_mode", ""))
    experiment_meta = manifest_experiment_meta(manifest)
    manifest_replay_policy = (
        manifest.get("replay_bug_policy", {}) if isinstance(manifest.get("replay_bug_policy", {}), dict) else {}
    )
    out = []
    for run in manifest.get("runs", []):
        run_file = Path(run.get("run_file", ""))
        meta_path = run_meta_path(run_file)
        meta = load_json(meta_path) if meta_path.is_file() else {}
        run_log_scanned = bool(scan_run_logs and run_file.is_file())
        rows = read_jsonl(run_file) if run_log_scanned else []
        config = meta.get("config", {}) if isinstance(meta.get("config", {}), dict) else {}
        known_bug_families = list(config.get("known_saturated_bug_families", []) or [])
        replay_filter = (
            meta.get("replay_bug_filter", {}) if isinstance(meta.get("replay_bug_filter", {}), dict) else {}
        )
        run_provenance = (
            meta.get("run_provenance", {}) if isinstance(meta.get("run_provenance", {}), dict) else {}
        )
        target_suite = str(run.get("target_suite", manifest.get("target_suite", "")) or "")
        target_families = _target_families(run, manifest, meta)
        evidence_mode = _normalize_evidence_mode(run.get("evidence_mode", manifest_mode))
        if evidence_mode == "live" and _has_seeded_target(target_suite, target_families):
            evidence_mode = "unknown"
        run_semantics = resolved_run_semantics(run, experiment_meta)
        run_enable_replay_bug = (
            bool(config.get("enable_replay_bug"))
            if "enable_replay_bug" in config
            else replay_bug_enabled_by_default(evidence_mode)
        )
        manifest_enable_replay_bug = (
            bool(manifest_replay_policy.get("enable_replay_bug"))
            if "enable_replay_bug" in manifest_replay_policy
            else replay_bug_enabled_by_default(evidence_mode)
        )
        findings = [finding for row in rows for finding in row.get("findings", [])]
        findings_count = len(findings) if run_log_scanned else int(meta.get("findings", 0) or 0)
        first_candidate_index, first_candidate_elapsed_s = _first_candidate_from_run_rows(
            rows,
            known_bug_families,
        )
        candidate_discovery_auc = _candidate_discovery_auc_from_run_rows(
            rows,
            known_bug_families,
        )
        adaptive_selection_telemetry = _adaptive_selection_telemetry_from_rows(rows)
        registered_matrix = registered_experiment_matrix_for_run(
            evidence_mode=evidence_mode,
            target_suite=target_suite,
            preset=str(run.get("preset", "")),
        )
        out.append(
            {
                "manifest_file": manifest.get("_manifest_file", ""),
                "run_file": str(run_file),
                "target_suite": target_suite,
                "preset": str(run.get("preset", "")),
                "matrix_id": run_semantics["matrix_id"],
                "comparison_group": run_semantics["comparison_group"],
                "variant_id": run_semantics["variant_id"],
                "variant_label": experiment_row_variant_label(run_semantics),
                "variant_group_id": "|".join(experiment_row_group_id(run_semantics)),
                "base_preset": run_semantics["base_preset"],
                "comparison_role": run_semantics["comparison_role"],
                "canonical_comparison_role": run_semantics["canonical_comparison_role"],
                "component_focus": run_semantics["component_focus"],
                "analysis_tags": list(run_semantics["analysis_tags"]),
                "scope_kind": run_semantics["scope_kind"],
                "oracle_profile": run_semantics["oracle_profile"],
                "semantic_focus_families": list(run_semantics["semantic_focus_families"]),
                "semantic_focus_signals": list(run_semantics["semantic_focus_signals"]),
                "registered_matrix_id": registered_matrix.id if registered_matrix is not None else "",
                "registered_experiment": bool(registered_matrix is not None),
                "seed": run.get("seed", ""),
                "evidence_mode": evidence_mode,
                "evidence_kind": str(run.get("evidence_kind", manifest.get("evidence_kind", "")) or ""),
                "known_bug_id": str(run.get("known_bug_id", manifest.get("known_bug_id", "")) or ""),
                "target_version": str(run.get("target_version", manifest.get("target_version", "")) or ""),
                "cases": int(meta.get("executed_cases", len(rows)) or 0),
                "elapsed_s": float(meta.get("elapsed_s", 0.0) or 0.0),
                "throughput_cases_s": float(meta.get("throughput_cases_s", 0.0) or 0.0),
                "first_candidate_bug_case_index": _optional_int(
                    meta.get("first_candidate_bug_case_index"),
                    default=first_candidate_index,
                ),
                "first_candidate_bug_elapsed_s": _optional_float(
                    meta.get("first_candidate_bug_elapsed_s"),
                    default=first_candidate_elapsed_s,
                ),
                "candidate_bug_discovery_auc": _optional_float(
                    meta.get("candidate_bug_discovery_auc"),
                    default=candidate_discovery_auc,
                )
                or 0.0,
                "duration_s": meta.get("duration_s"),
                "findings": findings_count,
                "config": dict(config),
                "closed_loop_state_file": str(meta.get("closed_loop_state_file", "") or ""),
                "closed_loop_state_summary": (
                    dict(meta.get("closed_loop_state_summary", {}))
                    if isinstance(meta.get("closed_loop_state_summary", {}), dict)
                    else {}
                ),
                "run_log_scanned": run_log_scanned,
                "run_log_scan_skipped": bool(not run_log_scanned and run_file.is_file()),
                "adaptive_selection_telemetry": adaptive_selection_telemetry,
                "run_meta": dict(meta),
                "stage_profile_present": _stage_profile_present(meta),
                "stage_profile_totals": _stage_profile_totals(meta),
                "version_ledger_files": _version_ledger_files(run, manifest, meta),
                "target_families": target_families,
                "enable_replay_bug": run_enable_replay_bug,
                "manifest_enable_replay_bug": manifest_enable_replay_bug,
                "replay_filter_enabled": replay_filter.get("enabled", ""),
                "replay_filter_filtered_candidates": int(replay_filter.get("filtered_candidates", 0) or 0),
                "known_saturated_bug_families": known_bug_families,
                "run_provenance": run_provenance,
                "schedule": str(run.get("schedule", manifest.get("schedule", "")) or ""),
                "adaptive_config": _mapping_copy(manifest.get("adaptive_config", {})),
                "adaptive_scheduler_state": [
                    dict(item)
                    for item in manifest.get("adaptive_state", []) or []
                    if isinstance(item, dict)
                ],
                "adaptive_learning_state": _mapping_copy(manifest.get("adaptive_learning", {})),
                "adaptive_components": _adaptive_components(run, manifest),
                "disabled_adaptive_components": _disabled_adaptive_components(run, manifest),
                "rewardable_candidate_families": _candidate_family_counter(
                    findings,
                    rewardable=True,
                    known_saturated_bug_families=known_bug_families,
                ),
                "confirmed_candidate_families": _candidate_family_counter(
                    findings,
                    confirmed=True,
                    known_saturated_bug_families=known_bug_families,
                    policy=policy,
                ),
                "issue_replay_candidate_families": _candidate_family_counter(findings, issue_replay=True),
                "known_saturated_candidate_families": _candidate_family_counter(
                    findings,
                    known_saturated=True,
                    known_saturated_bug_families=known_bug_families,
                ),
            }
        )
    return out


def _normalize_evidence_mode(value: Any) -> str:
    mode = str(value or "").strip()
    return mode if mode in EVIDENCE_MODES else "unknown"


def _has_seeded_target(target_suite: str, target_families: list[str]) -> bool:
    return is_seeded_experiment_row(
        {
            "target_suite": target_suite,
            "target_families": list(target_families),
        }
    )


def _target_families(run: dict[str, Any], manifest: dict[str, Any], meta: dict[str, Any]) -> list[str]:
    targets = meta.get("targets") or manifest.get("targets") or []
    families = {str(target.get("family", "")) for target in targets if isinstance(target, dict)}
    if families:
        return sorted(family for family in families if family)
    backends = run.get("backends") or manifest.get("backends_by_suite", {}).get(run.get("target_suite"), [])
    if not backends:
        backends = TARGET_SUITES.get(str(run.get("target_suite", "")), [])
    return list(target_context(backends).families)


def _candidate_family_counter(
    findings: list[dict[str, Any]],
    *,
    rewardable: bool = False,
    confirmed: bool = False,
    issue_replay: bool = False,
    known_saturated: bool = False,
    known_saturated_bug_families: list[str] | tuple[str, ...] | None = None,
    policy: ReadinessPolicy | None = None,
) -> Counter[str]:
    policy = policy or DEFAULT_A_LEVEL_READINESS_POLICY
    confirmed_statuses = set(policy.confirmed_live_paper_statuses)
    counter: Counter[str] = Counter()
    for finding in findings:
        if str(finding.get("triage_verdict", "")) != "candidate_implementation_bug":
            continue
        if rewardable and not is_rewardable_candidate_issue_finding(finding, known_saturated_bug_families):
            continue
        if confirmed and str(finding.get("paper_status", "")) not in confirmed_statuses:
            continue
        if confirmed and is_known_saturated_candidate_issue_finding(finding, known_saturated_bug_families):
            continue
        if issue_replay and not is_issue_replay_finding(finding):
            continue
        if known_saturated and not is_known_saturated_candidate_issue_finding(
            finding,
            known_saturated_bug_families,
        ):
            continue
        counter[_candidate_family_key(finding)] += 1
    return counter


def _candidate_family_key(finding: dict[str, Any]) -> str:
    return candidate_issue_family_key(finding)


def _first_candidate_from_run_rows(
    rows: list[dict[str, Any]],
    known_saturated_bug_families: list[str] | tuple[str, ...],
) -> tuple[int | None, float | None]:
    for index, row in enumerate(rows):
        findings = row.get("findings", []) if isinstance(row, dict) else []
        if not any(
            isinstance(finding, dict)
            and is_rewardable_candidate_issue_finding(finding, known_saturated_bug_families)
            for finding in findings
        ):
            continue
        return _optional_int(row.get("case_index"), default=index), _optional_float(row.get("elapsed_s"))
    return None, None


def _adaptive_selection_telemetry_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scope_specs = (
        ("generator_profile", "generator_profile_selection", ("profile", "action"), "selected_generator_profile"),
        ("semantic_objective", "semantic_objective_selection", ("action",), "selected_semantic_objective"),
        ("metamorphic_relation", "metamorphic_relation_selection", ("action",), "selected_metamorphic_relation"),
        ("version_pair", "version_pair_selection", ("action",), "selected_version_pair"),
        ("backend_pair", "backend_pair_selection", ("action", "priority"), "backend_pair_priority"),
    )
    scope_counts: Counter[str] = Counter()
    strategy_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    reward_total = 0.0
    reward_count = 0
    ranked_metric_totals: Counter[str] = Counter()
    ranked_metric_counts: Counter[str] = Counter()
    for row in rows:
        if not isinstance(row, dict):
            continue
        for scope, selection_key, action_keys, fallback_key in scope_specs:
            selection = row.get(selection_key, {})
            if not isinstance(selection, dict) or not selection:
                continue
            action = _adaptive_selection_action(selection, action_keys, row.get(fallback_key, ""))
            strategy = str(selection.get("strategy", "") or "").strip()
            if not action and not strategy:
                continue
            scope_counts[scope] += 1
            if strategy:
                strategy_counts[f"{scope}:{strategy}"] += 1
            if action:
                action_counts[f"{scope}:{action}"] += 1
            reward = _optional_float(selection.get("reward"))
            if reward is not None:
                reward_total += reward
                reward_count += 1
            rank_row = _adaptive_selection_rank_row(selection.get("ranked", []), action)
            for metric in (
                "score",
                "model_prediction",
                "uncertainty",
                "exploration_bonus",
                "version_signal",
                "continual_priority_signal",
                "health_penalty",
            ):
                value = _optional_float(rank_row.get(metric) if isinstance(rank_row, dict) else None)
                if value is None:
                    continue
                ranked_metric_totals[metric] += value
                ranked_metric_counts[metric] += 1
    total = sum(scope_counts.values())
    return {
        "total_count": int(total),
        "scope_counts": dict(sorted(scope_counts.items())),
        "scope_count": len(scope_counts),
        "strategy_counts": dict(sorted(strategy_counts.items())),
        "top_actions": dict(action_counts.most_common(16)),
        "avg_reward": reward_total / reward_count if reward_count else 0.0,
        "avg_score": _counter_avg(ranked_metric_totals, ranked_metric_counts, "score"),
        "avg_model_prediction": _counter_avg(ranked_metric_totals, ranked_metric_counts, "model_prediction"),
        "avg_uncertainty": _counter_avg(ranked_metric_totals, ranked_metric_counts, "uncertainty"),
        "avg_exploration_bonus": _counter_avg(ranked_metric_totals, ranked_metric_counts, "exploration_bonus"),
        "avg_version_signal": _counter_avg(ranked_metric_totals, ranked_metric_counts, "version_signal"),
        "avg_continual_priority_signal": _counter_avg(
            ranked_metric_totals,
            ranked_metric_counts,
            "continual_priority_signal",
        ),
        "avg_health_penalty": _counter_avg(ranked_metric_totals, ranked_metric_counts, "health_penalty"),
    }


def _adaptive_selection_action(selection: dict[str, Any], keys: tuple[str, ...], fallback: Any = "") -> str:
    for key in keys:
        value = selection.get(key, "")
        if isinstance(value, list):
            value = next((item for item in value if str(item).strip()), "")
        text = str(value or "").strip()
        if text:
            return text
    if isinstance(fallback, list):
        fallback = next((item for item in fallback if str(item).strip()), "")
    return str(fallback or "").strip()


def _adaptive_selection_rank_row(ranked: Any, action: str) -> dict[str, Any]:
    if not isinstance(ranked, list):
        return {}
    if action:
        for row in ranked:
            if isinstance(row, dict) and str(row.get("action_id", "") or "").strip() == action:
                return row
    for row in ranked:
        if isinstance(row, dict):
            return row
    return {}


def _counter_avg(totals: Counter[str], counts: Counter[str], key: str) -> float:
    count = int(counts.get(key, 0) or 0)
    return float(totals.get(key, 0.0) or 0.0) / count if count else 0.0


def _candidate_discovery_auc_from_run_rows(
    rows: list[dict[str, Any]],
    known_saturated_bug_families: list[str] | tuple[str, ...],
) -> float:
    if not rows:
        return 0.0
    hits: list[int] = []
    for row in rows:
        findings = row.get("findings", []) if isinstance(row, dict) else []
        hits.append(
            int(
                any(
                    isinstance(finding, dict)
                    and is_rewardable_candidate_issue_finding(finding, known_saturated_bug_families)
                    for finding in findings
                )
            )
        )
    total = sum(hits)
    if total <= 0:
        return 0.0
    cumulative = 0
    area = 0
    for hit in hits:
        cumulative += hit
        area += cumulative
    return area / (len(rows) * total)


def _adaptive_components(run: dict[str, Any], manifest: dict[str, Any]) -> dict[str, bool]:
    run_components = run.get("adaptive_components", {})
    if isinstance(run_components, dict) and run_components:
        return {str(key): bool(value) for key, value in run_components.items() if str(key).strip()}
    methodology = manifest.get("adaptive_methodology", {})
    if isinstance(methodology, dict) and isinstance(methodology.get("components"), dict):
        return {
            str(key): bool(value)
            for key, value in methodology.get("components", {}).items()
            if str(key).strip()
        }
    adaptive_config = manifest.get("adaptive_config", {})
    if isinstance(adaptive_config, dict) and isinstance(adaptive_config.get("components"), dict):
        return {
            str(key): bool(value)
            for key, value in adaptive_config.get("components", {}).items()
            if str(key).strip()
        }
    return {}


def _disabled_adaptive_components(run: dict[str, Any], manifest: dict[str, Any]) -> list[str]:
    explicit = _string_list(run.get("disabled_adaptive_components", []))
    if explicit:
        return explicit
    components = _adaptive_components(run, manifest)
    disabled = sorted(component for component, enabled in components.items() if not enabled)
    if disabled:
        return disabled
    methodology = manifest.get("adaptive_methodology", {})
    if isinstance(methodology, dict):
        return _string_list(methodology.get("disabled_components", []))
    return []


def _adaptive_component_ablation_summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    disabled_counter: Counter[str] = Counter()
    component_counter: Counter[str] = Counter()
    run_labels: list[str] = []
    reference_labels: list[str] = []
    contrast_labels: list[str] = []
    reference_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    for run in runs:
        if str(run.get("matrix_id", "") or "").strip() != FINAL_ADAPTIVE_COMPONENT_ABLATION_MATRIX.id:
            continue
        components = run.get("adaptive_components", {})
        if isinstance(components, dict):
            component_counter.update(str(component) for component in components if str(component).strip())
        disabled = _string_list(run.get("disabled_adaptive_components", []))
        label = _run_label(run)
        row = _adaptive_component_ablation_run_row(run, disabled_components=disabled)
        run_labels.append(label)
        if disabled:
            disabled_counter.update(disabled)
            contrast_labels.append(label)
            contrast_rows.append(row)
        else:
            reference_labels.append(label)
            reference_rows.append(row)
    return {
        "run_count": len(run_labels),
        "reference_run_count": len(reference_rows),
        "contrast_run_count": len(contrast_rows),
        "components": sorted(component_counter),
        "disabled_components": sorted(disabled_counter),
        "disabled_component_counts": dict(sorted(disabled_counter.items())),
        "run_labels": sorted(set(run_labels)),
        "reference_run_labels": sorted(set(reference_labels)),
        "contrast_run_labels": sorted(set(contrast_labels)),
        "reference_metrics": _adaptive_component_ablation_metrics(reference_rows),
        "contrast_metrics": _adaptive_component_ablation_metrics(contrast_rows),
        "run_metrics": (reference_rows + contrast_rows)[:32],
    }


def _run_label(run: dict[str, Any]) -> str:
    return f"{run['evidence_mode']}:{run['target_suite']}:{run['preset']}"


def _adaptive_component_ablation_run_row(
    run: dict[str, Any],
    *,
    disabled_components: list[str],
) -> dict[str, Any]:
    return {
        "label": _run_label(run),
        "variant_label": str(run.get("variant_label", "") or ""),
        "disabled_components": list(disabled_components),
        "cases": int(run.get("cases", 0) or 0),
        "elapsed_s": float(run.get("elapsed_s", 0.0) or 0.0),
        "throughput_cases_s": float(run.get("throughput_cases_s", 0.0) or 0.0),
        "findings": int(run.get("findings", 0) or 0),
        "rewardable_candidate_family_count": len(run.get("rewardable_candidate_families", {}) or {}),
        "confirmed_candidate_family_count": len(run.get("confirmed_candidate_families", {}) or {}),
        "first_candidate_bug_case_index": run.get("first_candidate_bug_case_index"),
        "first_candidate_bug_elapsed_s": run.get("first_candidate_bug_elapsed_s"),
    }


def _adaptive_component_ablation_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cases = sum(int(row.get("cases", 0) or 0) for row in rows)
    elapsed_s = sum(float(row.get("elapsed_s", 0.0) or 0.0) for row in rows)
    throughputs = [float(row.get("throughput_cases_s", 0.0) or 0.0) for row in rows]
    return {
        "run_count": len(rows),
        "cases": cases,
        "elapsed_s": elapsed_s,
        "avg_throughput_cases_s": sum(throughputs) / len(throughputs) if throughputs else 0.0,
        "min_throughput_cases_s": min(throughputs) if throughputs else 0.0,
        "findings": sum(int(row.get("findings", 0) or 0) for row in rows),
        "rewardable_candidate_family_count": sum(
            int(row.get("rewardable_candidate_family_count", 0) or 0) for row in rows
        ),
        "confirmed_candidate_family_count": sum(
            int(row.get("confirmed_candidate_family_count", 0) or 0) for row in rows
        ),
    }


def _support_track_comparison_summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, dict[str, Any]] = {}
    reference_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    for run in runs:
        group_key = _support_group_key(run)
        group = groups.setdefault(
            group_key,
            {
                "group": group_key,
                "run_count": 0,
                "reference_run_count": 0,
                "contrast_run_count": 0,
                "reference_variants": set(),
                "contrast_variants": set(),
                "matrix_ids": set(),
                "target_suites": set(),
            },
        )
        group["run_count"] += 1
        group["matrix_ids"].add(str(run.get("matrix_id", "") or ""))
        group["target_suites"].add(str(run.get("target_suite", "") or ""))
        role = str(run.get("canonical_comparison_role", "") or "").strip()
        row = _support_track_run_row(run)
        if role == "baseline":
            group["reference_run_count"] += 1
            group["reference_variants"].add(str(run.get("variant_label", "") or run.get("preset", "") or ""))
            reference_rows.append(row)
        elif role == "contrast":
            group["contrast_run_count"] += 1
            group["contrast_variants"].add(str(run.get("variant_label", "") or run.get("preset", "") or ""))
            contrast_rows.append(row)
    group_rows = []
    missing_reference: list[str] = []
    missing_contrast: list[str] = []
    for group_key, group in sorted(groups.items()):
        reference_count = int(group["reference_run_count"])
        contrast_count = int(group["contrast_run_count"])
        if reference_count <= 0:
            missing_reference.append(group_key)
        if contrast_count <= 0:
            missing_contrast.append(group_key)
        group_rows.append(
            {
                "group": group_key,
                "run_count": int(group["run_count"]),
                "reference_run_count": reference_count,
                "contrast_run_count": contrast_count,
                "reference_variants": sorted(value for value in group["reference_variants"] if value),
                "contrast_variants": sorted(value for value in group["contrast_variants"] if value),
                "matrix_ids": sorted(value for value in group["matrix_ids"] if value),
                "target_suites": sorted(value for value in group["target_suites"] if value),
            }
        )
    return {
        "group_count": len(group_rows),
        "complete_group_count": sum(
            1 for row in group_rows if row["reference_run_count"] > 0 and row["contrast_run_count"] > 0
        ),
        "run_count": len(runs),
        "reference_run_count": len(reference_rows),
        "contrast_run_count": len(contrast_rows),
        "missing_reference_groups": missing_reference,
        "missing_contrast_groups": missing_contrast,
        "reference_metrics": _support_track_metrics(reference_rows),
        "contrast_metrics": _support_track_metrics(contrast_rows),
        "groups": group_rows[:64],
    }


def _support_group_key(run: dict[str, Any]) -> str:
    return "|".join(
        (
            str(run.get("target_suite", "") or ""),
            str(run.get("comparison_group", "") or ""),
            str(run.get("matrix_id", "") or ""),
        )
    )


def _support_track_run_row(run: dict[str, Any]) -> dict[str, Any]:
    return {
        "label": _run_label(run),
        "variant_label": str(run.get("variant_label", "") or ""),
        "cases": int(run.get("cases", 0) or 0),
        "elapsed_s": float(run.get("elapsed_s", 0.0) or 0.0),
        "throughput_cases_s": float(run.get("throughput_cases_s", 0.0) or 0.0),
        "findings": int(run.get("findings", 0) or 0),
    }


def _support_track_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cases = sum(int(row.get("cases", 0) or 0) for row in rows)
    elapsed_s = sum(float(row.get("elapsed_s", 0.0) or 0.0) for row in rows)
    throughputs = [float(row.get("throughput_cases_s", 0.0) or 0.0) for row in rows]
    return {
        "run_count": len(rows),
        "cases": cases,
        "elapsed_s": elapsed_s,
        "avg_throughput_cases_s": sum(throughputs) / len(throughputs) if throughputs else 0.0,
        "findings": sum(int(row.get("findings", 0) or 0) for row in rows),
    }


def _final_matrix_coverage_summary(
    runs: list[dict[str, Any]],
    *,
    policy: ReadinessPolicy,
) -> dict[str, Any]:
    matrix_counter: Counter[str] = Counter()
    run_labels_by_matrix: dict[str, list[str]] = defaultdict(list)
    for run in runs:
        matrix_id = str(run.get("matrix_id", "") or "").strip()
        if not matrix_id:
            continue
        matrix_counter[matrix_id] += 1
        run_labels_by_matrix[matrix_id].append(f"{run['evidence_mode']}:{run['target_suite']}:{run['preset']}")
    missing = sorted(set(policy.required_final_matrix_ids) - set(matrix_counter))
    return {
        "required_matrix_ids": list(policy.required_final_matrix_ids),
        "observed_matrix_ids": sorted(matrix_counter),
        "observed_matrix_counts": dict(sorted(matrix_counter.items())),
        "missing_matrix_ids": missing,
        "run_labels_by_matrix": {
            matrix_id: sorted(set(labels))
            for matrix_id, labels in sorted(run_labels_by_matrix.items())
        },
    }


def _transferability_scope_summary(
    runs: list[dict[str, Any]],
    *,
    policy: ReadinessPolicy,
) -> dict[str, Any]:
    family_counter: Counter[str] = Counter()
    suite_by_family: dict[str, set[str]] = defaultdict(set)
    run_labels: list[str] = []
    for run in runs:
        if run.get("evidence_mode") not in {"live", "comparison", "ablation", "validation"}:
            continue
        families = _string_list(run.get("target_families", []))
        if not families:
            continue
        run_labels.append(f"{run['evidence_mode']}:{run['target_suite']}:{run['preset']}")
        for family in families:
            family_counter[family] += 1
            suite_by_family[family].add(str(run.get("target_suite", "")))
    non_primary = sorted(family for family in family_counter if family != policy.primary_method_family)
    missing_required = sorted(set(policy.required_transfer_families) - set(family_counter))
    return {
        "families": sorted(family_counter),
        "family_count": len(family_counter),
        "family_run_counts": dict(sorted(family_counter.items())),
        "suites_by_family": {
            family: sorted(suites)
            for family, suites in sorted(suite_by_family.items())
        },
        "primary_method_family": policy.primary_method_family,
        "non_primary_families": non_primary,
        "non_primary_family_count": len(non_primary),
        "required_families": list(policy.required_transfer_families),
        "missing_required_families": missing_required,
        "run_labels": sorted(set(run_labels)),
    }


def _version_ledger_files(run: dict[str, Any], manifest: dict[str, Any], meta: dict[str, Any]) -> list[str]:
    candidates: list[str] = []
    for source in (run, manifest, meta):
        for key in ("version_ledger_file", "regression_ledger_file", "continual_learning_ledger_file"):
            value = source.get(key) if isinstance(source, dict) else None
            if value:
                candidates.extend(_string_list(value))
        for key in ("version_ledger_files", "regression_ledger_files", "continual_learning_ledger_files"):
            value = source.get(key) if isinstance(source, dict) else None
            if value:
                candidates.extend(_string_list(value))
    return list(dict.fromkeys(candidates))


def _cross_version_ledger_summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    ledger_paths = sorted(
        {
            path
            for run in runs
            for path in _string_list(run.get("version_ledger_files", []))
        }
    )
    ledgers = [_load_version_ledger(path) for path in ledger_paths]
    valid_ledgers = [ledger for ledger in ledgers if ledger.get("valid")]
    transition_counts: Counter[str] = Counter()
    priority_families: list[dict[str, Any]] = []
    version_counts: list[int] = []
    family_counts: list[int] = []
    health_observation_counts: list[int] = []
    health_feedback_reports: list[dict[str, Any]] = []
    invalid_case_count = 0
    fallback_case_count = 0
    false_positive_count = 0
    min_throughputs: list[float] = []
    max_invalid_rate = 0.0
    max_false_positive_rate = 0.0
    for ledger in valid_ledgers:
        summary = ledger.get("summary", {})
        if isinstance(summary, dict):
            version_counts.append(int(summary.get("version_count", 0) or 0))
            family_counts.append(int(summary.get("family_count", 0) or 0))
            transitions = summary.get("transition_counts", {})
            if isinstance(transitions, dict):
                transition_counts.update({str(key): int(value or 0) for key, value in transitions.items()})
        health = ledger.get("health", {})
        if isinstance(health, dict):
            health_observation_counts.append(int(health.get("health_observation_count", 0) or 0))
            invalid_case_count += int(health.get("invalid_case_count", 0) or 0)
            fallback_case_count += int(health.get("fallback_case_count", 0) or 0)
            false_positive_count += int(health.get("false_positive_count", 0) or 0)
            throughput = float(health.get("min_throughput_cases_s", 0.0) or 0.0)
            if throughput > 0.0:
                min_throughputs.append(throughput)
            max_invalid_rate = max(max_invalid_rate, float(health.get("max_invalid_rate", 0.0) or 0.0))
            max_false_positive_rate = max(
                max_false_positive_rate,
                float(health.get("max_false_positive_rate", 0.0) or 0.0),
            )
        health_feedback_report = ledger.get("health_feedback_report", {})
        if isinstance(health_feedback_report, dict):
            health_feedback_reports.append(
                {
                    "ledger_file": str(ledger.get("ledger_file", "")),
                    "schema_version": str(health_feedback_report.get("schema_version", "") or ""),
                    "has_health_feedback": bool(health_feedback_report.get("has_health_feedback", False)),
                    "health_observation_count": int(
                        health_feedback_report.get("health_observation_count", 0) or 0
                    ),
                    "invalid_case_count": int(health_feedback_report.get("invalid_case_count", 0) or 0),
                    "fallback_case_count": int(health_feedback_report.get("fallback_case_count", 0) or 0),
                    "false_positive_count": int(health_feedback_report.get("false_positive_count", 0) or 0),
                    "min_throughput_cases_s": float(
                        health_feedback_report.get("min_throughput_cases_s", 0.0) or 0.0
                    ),
                    "max_invalid_rate": float(health_feedback_report.get("max_invalid_rate", 0.0) or 0.0),
                    "max_false_positive_rate": float(
                        health_feedback_report.get("max_false_positive_rate", 0.0) or 0.0
                    ),
                }
            )
        continual = ledger.get("continual_learning", {})
        if isinstance(continual, dict):
            for row in continual.get("family_priorities", []) or []:
                if isinstance(row, dict) and str(row.get("family", "")).strip():
                    priority_families.append(
                        {
                            "family": str(row.get("family", "")).strip(),
                            "status": str(row.get("status", "") or ""),
                            "priority": float(row.get("priority", 0.0) or 0.0),
                            "ledger_file": str(ledger.get("ledger_file", "")),
                        }
                    )
    priority_families.sort(key=lambda row: (-float(row["priority"]), row["family"], row["ledger_file"]))
    invalid_ledgers = [ledger for ledger in ledgers if not ledger.get("valid")]
    return {
        "ledger_files": ledger_paths,
        "valid_ledger_files": [str(ledger.get("ledger_file", "")) for ledger in valid_ledgers],
        "invalid_ledger_files": [str(ledger.get("ledger_file", "")) for ledger in invalid_ledgers],
        "invalid_reasons": {
            str(ledger.get("ledger_file", "")): str(ledger.get("reason", "invalid"))
            for ledger in invalid_ledgers
        },
        "ledger_count": len(valid_ledgers),
        "max_version_count": max(version_counts) if version_counts else 0,
        "max_family_count": max(family_counts) if family_counts else 0,
        "transition_counts": dict(sorted(transition_counts.items())),
        "priority_families": priority_families[:10],
        "health_observation_count": sum(health_observation_counts),
        "max_health_observation_count": max(health_observation_counts) if health_observation_counts else 0,
        "health_feedback_report_count": len(health_feedback_reports),
        "health_feedback_reports": health_feedback_reports[:10],
        "invalid_case_count": invalid_case_count,
        "fallback_case_count": fallback_case_count,
        "false_positive_count": false_positive_count,
        "min_throughput_cases_s": min(min_throughputs) if min_throughputs else 0.0,
        "max_invalid_rate": max_invalid_rate,
        "max_false_positive_rate": max_false_positive_rate,
    }


def _load_version_ledger(path_value: str) -> dict[str, Any]:
    path = Path(str(path_value)).expanduser()
    if not path.is_file():
        return {"ledger_file": str(path_value), "valid": False, "reason": "missing"}
    data = load_json(path)
    if not isinstance(data, dict):
        return {"ledger_file": str(path), "valid": False, "reason": "not_object"}
    if str(data.get("schema_version", "")) != "version-ledger-v1":
        return {"ledger_file": str(path), "valid": False, "reason": "schema_mismatch"}
    summary = data.get("summary", {}) if isinstance(data.get("summary", {}), dict) else {}
    health = data.get("health", {}) if isinstance(data.get("health", {}), dict) else {}
    if str(health.get("schema_version", "") or "") != "version-ledger-health-v1":
        return {"ledger_file": str(path), "valid": False, "reason": "missing_health_feedback"}
    if "health_observation_count" not in health:
        return {"ledger_file": str(path), "valid": False, "reason": "missing_health_feedback"}
    health_feedback_report = (
        data.get("health_feedback_report", {})
        if isinstance(data.get("health_feedback_report", {}), dict)
        else {}
    )
    if str(health_feedback_report.get("schema_version", "") or "") != (
        "version-ledger-health-feedback-report-v1"
    ):
        return {"ledger_file": str(path), "valid": False, "reason": "missing_health_feedback_report"}
    if "health_observation_count" not in health_feedback_report:
        return {"ledger_file": str(path), "valid": False, "reason": "missing_health_feedback_report"}
    return {
        "ledger_file": str(path),
        "valid": True,
        "summary": summary,
        "health": health,
        "health_feedback_report": health_feedback_report,
        "continual_learning": data.get("continual_learning", {}),
    }


def _target_version_audit_summary(manifests: list[dict[str, Any]]) -> dict[str, Any]:
    audits = [
        manifest
        for manifest in manifests
        if str(manifest.get("schema_version", "") or "") == TARGET_VERSION_AUDIT_SCHEMA_VERSION
    ]
    valid: list[dict[str, Any]] = []
    invalid_reasons: dict[str, str] = {}
    for audit in audits:
        path = str(audit.get("_manifest_file", "") or "")
        summary = audit.get("summary", {}) if isinstance(audit.get("summary", {}), dict) else {}
        packages = audit.get("target_packages", []) if isinstance(audit.get("target_packages", []), list) else []
        if not summary:
            invalid_reasons[path] = "missing_summary"
            continue
        if not packages:
            invalid_reasons[path] = "missing_target_packages"
            continue
        valid.append(audit)
    outdated: list[dict[str, Any]] = []
    unknown: list[dict[str, Any]] = []
    package_count = 0
    up_to_date_count = 0
    for audit in valid:
        summary = audit.get("summary", {}) if isinstance(audit.get("summary", {}), dict) else {}
        package_count = max(package_count, int(summary.get("target_package_count", 0) or 0))
        up_to_date_count = max(up_to_date_count, int(summary.get("up_to_date_target_package_count", 0) or 0))
        outdated.extend(
            item
            for item in summary.get("outdated_target_packages", []) or []
            if isinstance(item, dict)
        )
        unknown.extend(
            item
            for item in summary.get("unknown_latest_target_packages", []) or []
            if isinstance(item, dict)
        )
    all_up_to_date = bool(valid) and not outdated and not unknown and not invalid_reasons
    return {
        "schema_version": "final-readiness-target-version-audit-summary-v1",
        "audit_files": [str(audit.get("_manifest_file", "") or "") for audit in audits],
        "valid_audit_files": [str(audit.get("_manifest_file", "") or "") for audit in valid],
        "invalid_files": sorted(invalid_reasons),
        "invalid_reasons": invalid_reasons,
        "audit_count": len(audits),
        "valid_audit_count": len(valid),
        "target_package_count": package_count,
        "up_to_date_target_package_count": up_to_date_count,
        "outdated_target_package_count": len(outdated),
        "unknown_latest_target_package_count": len(unknown),
        "all_target_packages_up_to_date": all_up_to_date,
        "outdated_target_packages": outdated,
        "unknown_latest_target_packages": unknown,
    }


def _runtime_efficiency_summary(
    runs: list[dict[str, Any]],
    *,
    thresholds: ReadinessThresholds,
) -> dict[str, Any]:
    throughputs = [float(run.get("throughput_cases_s", 0.0) or 0.0) for run in runs]
    scheduler_share_rows, scheduler_share_skipped = _scheduler_feedback_share_rows(
        runs,
        thresholds=thresholds,
    )
    scheduler_shares = [
        float(row["share"])
        for row in scheduler_share_rows
        if row.get("share") is not None
    ]
    return {
        "run_count": len(runs),
        "min_throughput_cases_s": min(throughputs) if throughputs else 0.0,
        "avg_throughput_cases_s": sum(throughputs) / len(throughputs) if throughputs else 0.0,
        "max_scheduler_feedback_share": max(scheduler_shares) if scheduler_shares else 0.0,
        "scheduler_feedback_share_run_count": len(scheduler_share_rows),
        "scheduler_feedback_share_skipped_run_count": len(scheduler_share_skipped),
        "scheduler_feedback_share_skipped_runs": scheduler_share_skipped[:32],
        "min_required_throughput_cases_s": thresholds.min_throughput_cases_s,
        "max_allowed_scheduler_feedback_share": thresholds.max_scheduler_feedback_share,
        "min_scheduler_feedback_cases": thresholds.min_scheduler_feedback_cases,
        "issues": _runtime_efficiency_issues(runs, thresholds=thresholds),
    }


def _discovery_responsiveness_summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    observed = [
        run
        for run in runs
        if run.get("first_candidate_bug_case_index") is not None
        or run.get("first_candidate_bug_elapsed_s") is not None
    ]
    elapsed_values = [
        float(run["first_candidate_bug_elapsed_s"])
        for run in observed
        if run.get("first_candidate_bug_elapsed_s") is not None
    ]
    case_indexes = [
        int(run["first_candidate_bug_case_index"])
        for run in observed
        if run.get("first_candidate_bug_case_index") is not None
    ]
    auc_values = [float(run.get("candidate_bug_discovery_auc", 0.0) or 0.0) for run in runs]
    best_run = min(
        observed,
        key=lambda run: (
            float(run.get("first_candidate_bug_elapsed_s"))
            if run.get("first_candidate_bug_elapsed_s") is not None
            else float("inf"),
            int(run.get("first_candidate_bug_case_index"))
            if run.get("first_candidate_bug_case_index") is not None
            else 10**12,
            str(run.get("target_suite", "")),
            str(run.get("preset", "")),
        ),
        default=None,
    )
    return {
        "run_count": len(runs),
        "observed_run_count": len(observed),
        "best_first_candidate_elapsed_s": min(elapsed_values) if elapsed_values else None,
        "best_first_candidate_case_index": min(case_indexes) if case_indexes else None,
        "avg_candidate_bug_discovery_auc": sum(auc_values) / len(auc_values) if auc_values else 0.0,
        "best_run_label": (
            f"{best_run['evidence_mode']}:{best_run['target_suite']}:{best_run['preset']}"
            if best_run is not None
            else ""
        ),
    }


def _closed_loop_state_persistence_summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    required = [run for run in runs if not _is_postprocess_evidence_run(run)]
    persisted: list[str] = []
    missing: list[str] = []
    weak: list[str] = []
    missing_health: list[str] = []
    total_bytes = 0
    health_rows: list[dict[str, Any]] = []
    for run in required:
        label = f"{run['evidence_mode']}:{run['target_suite']}:{run['preset']}"
        path_text = str(run.get("closed_loop_state_file", "") or "").strip()
        if not path_text:
            missing.append(f"{label}:missing_state_file")
            continue
        path = Path(path_text).expanduser()
        if not path.is_file():
            missing.append(f"{label}:state_file_not_found")
            continue
        size = path.stat().st_size
        total_bytes += size
        if size <= 0:
            missing.append(f"{label}:empty_state_file")
            continue
        summary = run.get("closed_loop_state_summary", {})
        if not _closed_loop_state_summary_has_signal(summary):
            weak.append(f"{label}:weak_state_summary")
            continue
        health = _adaptive_learning_health_from_summary(summary)
        if not health:
            missing_health.append(f"{label}:missing_adaptive_learning_health")
            continue
        health_rows.append({"label": label, **health})
        persisted.append(label)
    return {
        "required_run_count": len(required),
        "persisted_run_count": len(persisted),
        "persisted_run_labels": sorted(set(persisted)),
        "missing": sorted(set(missing)),
        "weak": sorted(set(weak)),
        "missing_health": sorted(set(missing_health)),
        "total_state_bytes": total_bytes,
        "adaptive_learning_health": _adaptive_learning_health_rollup(health_rows),
    }


def _adaptive_live_component_evidence_summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    declared_components: Counter[str] = Counter()
    proven_components: Counter[str] = Counter()
    enabled_run_count = 0
    for run in runs:
        if _is_postprocess_evidence_run(run):
            continue
        components = _declared_adaptive_live_components(run)
        if not components:
            continue
        enabled_run_count += 1
        declared_components.update(components)
        row = _adaptive_live_component_evidence_row(run, components)
        rows.append(row)
        proven_components.update(row["proven_components"])
        label = str(row["label"])
        for component in row["missing_components"]:
            missing.append(f"{label}:{component}")
        if int(row.get("adaptive_selection_total_count", 0) or 0) <= 0:
            missing.append(f"{label}:adaptive_selection_telemetry")
    return {
        "declared_run_count": len(rows),
        "enabled_run_count": enabled_run_count,
        "declared_components": sorted(declared_components),
        "proven_components": sorted(proven_components),
        "missing": sorted(set(missing)),
        "run_labels": sorted(str(row["label"]) for row in rows),
        "scheduler_learning_total_pulls": sum(int(row.get("scheduler_learning_pulls", 0) or 0) for row in rows),
        "reward_model_update_count": sum(int(row.get("reward_model_update_count", 0) or 0) for row in rows),
        "exploration_records": sum(int(row.get("exploration_records", 0) or 0) for row in rows),
        "version_memory_key_count": sum(int(row.get("version_memory_key_count", 0) or 0) for row in rows),
        "continual_imported_family_count": sum(int(row.get("continual_imported_family_count", 0) or 0) for row in rows),
        "quality_archive_cell_count": sum(int(row.get("quality_archive_cell_count", 0) or 0) for row in rows),
        "quality_archive_child_cell_count": sum(
            int(row.get("quality_archive_child_cell_count", 0) or 0) for row in rows
        ),
        "quality_archive_split_cell_count": sum(
            int(row.get("quality_archive_split_cell_count", 0) or 0) for row in rows
        ),
        "quality_archive_seed_count": sum(int(row.get("quality_archive_seed_count", 0) or 0) for row in rows),
        "quality_archive_outcome_count": sum(int(row.get("quality_archive_outcome_count", 0) or 0) for row in rows),
        "value_catalog_entry_pulls": sum(int(row.get("value_catalog_entry_pulls", 0) or 0) for row in rows),
        "value_catalog_entry_arm_count": sum(int(row.get("value_catalog_entry_arm_count", 0) or 0) for row in rows),
        "bd_axis_weight_pulls": sum(int(row.get("bd_axis_weight_pulls", 0) or 0) for row in rows),
        "bd_axis_weight_arm_count": sum(int(row.get("bd_axis_weight_arm_count", 0) or 0) for row in rows),
        "seed_energy_tier_pulls": sum(int(row.get("seed_energy_tier_pulls", 0) or 0) for row in rows),
        "seed_energy_tier_arm_count": sum(int(row.get("seed_energy_tier_arm_count", 0) or 0) for row in rows),
        "champion_graft_donor_pulls": sum(int(row.get("champion_graft_donor_pulls", 0) or 0) for row in rows),
        "champion_graft_donor_arm_count": sum(
            int(row.get("champion_graft_donor_arm_count", 0) or 0) for row in rows
        ),
        "seed_quota_active_run_count": sum(1 for row in rows if bool(row.get("seed_quota_active", False))),
        "seed_quota_cluster_count": sum(int(row.get("seed_quota_cluster_count", 0) or 0) for row in rows),
        "seed_quota_seed_count": sum(int(row.get("seed_quota_seed_count", 0) or 0) for row in rows),
        "lhs_seeding_enabled_run_count": sum(1 for row in rows if bool(row.get("lhs_seeding_enabled", False))),
        "champion_corpus_enabled_run_count": sum(
            1 for row in rows if bool(row.get("champion_corpus_enabled", False))
        ),
        "champion_corpus_injected_count": sum(
            int(row.get("champion_corpus_injected_count", 0) or 0) for row in rows
        ),
        "champion_corpus_promoted_family_count": sum(
            int(row.get("champion_corpus_promoted_family_count", 0) or 0) for row in rows
        ),
        "runtime_cost_observation_count": sum(int(row.get("runtime_cost_observation_count", 0) or 0) for row in rows),
        "annealing_observation_count": sum(int(row.get("annealing_observation_count", 0) or 0) for row in rows),
        "adaptive_selection_run_count": sum(
            1 for row in rows if int(row.get("adaptive_selection_total_count", 0) or 0) > 0
        ),
        "adaptive_selection_total_count": sum(
            int(row.get("adaptive_selection_total_count", 0) or 0) for row in rows
        ),
        "adaptive_selection_scopes": sorted(
            {
                scope
                for row in rows
                for scope, count in (row.get("adaptive_selection_scope_counts", {}) or {}).items()
                if int(count or 0) > 0
            }
        ),
        "rows": rows[:32],
    }


def _declared_adaptive_live_components(run: dict[str, Any]) -> set[str]:
    components = run.get("adaptive_components", {})
    declared = {
        str(component).strip()
        for component, enabled in components.items()
        if str(component).strip() in ADAPTIVE_LIVE_EVIDENCE_COMPONENTS and bool(enabled)
    } if isinstance(components, dict) else set()
    run_config = run.get("config", {})
    if isinstance(run_config, dict):
        case_selection_learning = _run_config_enables_case_selection_learning(run_config)
        if case_selection_learning:
            declared.add("scheduler_learning")
        if (
            float(run_config.get("backend_pair_learning_weight", 0.0) or 0.0) > 0.0
            and bool(run_config.get("enable_backend_pair_learning", True))
        ):
            declared.add("backend_pair_learning")
        if case_selection_learning and bool(run_config.get("enable_quality_archive", False)):
            declared.add("quality_archive")
    config = run.get("adaptive_config", {})
    if isinstance(config, dict):
        for component in ADAPTIVE_LIVE_EVIDENCE_COMPONENTS:
            if component in declared:
                continue
            config_key = _adaptive_component_config_key(component)
            if config_key and bool(config.get(config_key, False)):
                declared.add(component)
        if float(config.get("learning_weight", 0.0) or 0.0) > 0.0:
            declared.add("scheduler_learning")
        if float(config.get("annealing_initial_temperature", 0.0) or 0.0) > 0.0:
            declared.add("scheduler_annealing")
    return declared


def _run_config_enables_case_selection_learning(config: dict[str, Any]) -> bool:
    pairs = (
        ("generator_profile_learning_weight", "enable_generator_profile_learning"),
        ("semantic_objective_learning_weight", "enable_semantic_objective_learning"),
        ("metamorphic_relation_learning_weight", "enable_metamorphic_relation_learning"),
        ("version_pair_learning_weight", ""),
        ("backend_pair_learning_weight", "enable_backend_pair_learning"),
    )
    return any(
        float(config.get(weight_key, 0.0) or 0.0) > 0.0
        and (not enabled_key or bool(config.get(enabled_key, True)))
        for weight_key, enabled_key in pairs
    )


def _adaptive_component_config_key(component: str) -> str:
    return {
        "active_learning": "active_learning",
        "continual_learning": "continual_learning",
        "online_reward_model": "online_reward_model",
        "runtime_cost_learning": "runtime_cost_learning",
        "scheduler_annealing": "scheduler_annealing",
        "value_catalog": "value_catalog",
        "bd_axis_bandit": "bd_axis_bandit",
        "bayesian_exploration": "bayesian_exploration",
        "seed_quota": "seed_quota",
        "seed_energy_tier": "seed_energy_tier",
        "lhs_seeding": "lhs_seeding",
        "champion_corpus": "champion_corpus",
        "champion_graft_donor": "champion_graft_donor",
    }.get(component, "")


def _backend_pair_learning_observed(run: dict[str, Any], health: dict[str, Any]) -> bool:
    selection = run.get("adaptive_selection_telemetry", {})
    if isinstance(selection, dict):
        scope_counts = selection.get("scope_counts", {})
        if isinstance(scope_counts, dict) and int(scope_counts.get("backend_pair", 0) or 0) > 0:
            return True
    state = run.get("adaptive_learning_state", {})
    if isinstance(state, dict):
        bandits = state.get("bandits", {})
        if isinstance(bandits, dict):
            backend_pair = bandits.get("backend_pair", {})
            if isinstance(backend_pair, dict) and int(backend_pair.get("total_pulls", 0) or 0) > 0:
                return True
    return int(health.get("backend_pair_pulls", 0) or 0) > 0


def _adaptive_live_component_evidence_row(run: dict[str, Any], components: set[str]) -> dict[str, Any]:
    summary = run.get("closed_loop_state_summary", {})
    health = _adaptive_learning_health_from_summary(summary)
    quality = _quality_archive_health_from_summary(summary)
    seed_quota = _seed_quota_health_from_summary(summary)
    champion_corpus = _champion_corpus_health_from_run(run)
    selection = run.get("adaptive_selection_telemetry", {})
    selection = selection if isinstance(selection, dict) else {}
    scheduler_rows = [
        item
        for item in run.get("adaptive_scheduler_state", []) or []
        if isinstance(item, dict)
    ]
    scheduler_learning_pulls = sum(int(item.get("pulls", 0) or 0) for item in scheduler_rows)
    annealing_observation_count = sum(
        1
        for item in scheduler_rows
        if float(item.get("annealing_temperature", 0.0) or 0.0) > 0.0
    )
    bayesian_exploration_observation_count = sum(
        int(item.get("bayesian_exploration_observation_count", 0) or 0)
        for item in scheduler_rows
    )
    continual_imported_family_count = int(health.get("continual_imported_family_count", 0) or 0)
    continual_family_count = int(health.get("continual_family_count", 0) or 0)
    component_signals = {
        "scheduler_learning": (
            int(health.get("total_pulls", 0) or 0) > 0
            or int(health.get("bandit_count", 0) or 0) > 0
            or scheduler_learning_pulls > 0
        ),
        "online_reward_model": (
            int(health.get("reward_model_update_count", 0) or 0) > 0
            and int(health.get("reward_model_feature_count", 0) or 0) > 0
        ),
        "active_learning": int(health.get("exploration_records", 0) or 0) > 0,
        "continual_learning": (
            int(health.get("version_memory_key_count", 0) or 0) > 0
            or continual_imported_family_count > 0
            or continual_family_count > 0
        ),
        "backend_pair_learning": _backend_pair_learning_observed(run, health),
        "value_catalog": int(health.get("value_catalog_entry_pulls", 0) or 0) > 0,
        "bd_axis_bandit": int(health.get("bd_axis_weight_pulls", 0) or 0) > 0,
        "seed_energy_tier": int(health.get("seed_energy_tier_pulls", 0) or 0) > 0,
        "champion_graft_donor": int(health.get("champion_graft_donor_pulls", 0) or 0) > 0,
        "bayesian_exploration": bayesian_exploration_observation_count > 0,
        "quality_archive": (
            int(quality.get("cell_count", 0) or 0) > 0
            and int(quality.get("seed_count", 0) or 0) > 0
        ),
        "hierarchical_archive": (
            bool(quality.get("hierarchical_enabled", False))
            and int(quality.get("child_cell_count", 0) or 0) > 0
        ),
        "seed_quota": bool(seed_quota.get("active", False)),
        "lhs_seeding": _lhs_seeding_observed(run),
        "champion_corpus": bool(champion_corpus.get("active", False)),
        "runtime_cost_learning": (
            int(health.get("runtime_cost_observation_count", 0) or 0) > 0
            or float(health.get("runtime_cost_total", 0.0) or 0.0) > 0.0
        ),
        "scheduler_annealing": annealing_observation_count > 0,
    }
    proven = sorted(component for component in components if component_signals.get(component, False))
    missing = sorted(component for component in components if component not in proven)
    return {
        "label": _run_label(run),
        "components": sorted(components),
        "proven_components": proven,
        "missing_components": missing,
        "scheduler_learning_pulls": scheduler_learning_pulls,
        "reward_model_update_count": int(health.get("reward_model_update_count", 0) or 0),
        "reward_model_feature_count": int(health.get("reward_model_feature_count", 0) or 0),
        "exploration_records": int(health.get("exploration_records", 0) or 0),
        "version_memory_key_count": int(health.get("version_memory_key_count", 0) or 0),
        "continual_imported_family_count": continual_imported_family_count,
        "continual_family_count": continual_family_count,
        "quality_archive_cell_count": int(quality.get("cell_count", 0) or 0),
        "quality_archive_child_cell_count": int(quality.get("child_cell_count", 0) or 0),
        "quality_archive_split_cell_count": int(quality.get("split_cell_count", 0) or 0),
        "quality_archive_seed_count": int(quality.get("seed_count", 0) or 0),
        "quality_archive_outcome_count": int(quality.get("outcome_count", 0) or 0),
        "value_catalog_entry_pulls": int(health.get("value_catalog_entry_pulls", 0) or 0),
        "value_catalog_entry_arm_count": int(health.get("value_catalog_entry_arm_count", 0) or 0),
        "bd_axis_weight_pulls": int(health.get("bd_axis_weight_pulls", 0) or 0),
        "bd_axis_weight_arm_count": int(health.get("bd_axis_weight_arm_count", 0) or 0),
        "seed_energy_tier_pulls": int(health.get("seed_energy_tier_pulls", 0) or 0),
        "seed_energy_tier_arm_count": int(health.get("seed_energy_tier_arm_count", 0) or 0),
        "champion_graft_donor_pulls": int(health.get("champion_graft_donor_pulls", 0) or 0),
        "champion_graft_donor_arm_count": int(health.get("champion_graft_donor_arm_count", 0) or 0),
        "seed_quota_active": bool(seed_quota.get("active", False)),
        "seed_quota_cluster_count": int(seed_quota.get("cluster_count", 0) or 0),
        "seed_quota_seed_count": int(seed_quota.get("seed_count", 0) or 0),
        "lhs_seeding_enabled": _lhs_seeding_observed(run),
        "champion_corpus_enabled": bool(champion_corpus.get("enabled", False)),
        "champion_corpus_injected_count": int(champion_corpus.get("injected_count", 0) or 0),
        "champion_corpus_promoted_family_count": int(champion_corpus.get("promoted_family_count", 0) or 0),
        "runtime_cost_observation_count": int(health.get("runtime_cost_observation_count", 0) or 0),
        "runtime_cost_total": float(health.get("runtime_cost_total", 0.0) or 0.0),
        "annealing_observation_count": annealing_observation_count,
        "bayesian_exploration_observation_count": bayesian_exploration_observation_count,
        "adaptive_selection_total_count": int(selection.get("total_count", 0) or 0),
        "adaptive_selection_scope_counts": dict(selection.get("scope_counts", {}) or {}),
        "adaptive_selection_top_actions": dict(selection.get("top_actions", {}) or {}),
        "adaptive_selection_avg_reward": float(selection.get("avg_reward", 0.0) or 0.0),
        "adaptive_selection_avg_uncertainty": float(selection.get("avg_uncertainty", 0.0) or 0.0),
        "adaptive_selection_avg_version_signal": float(selection.get("avg_version_signal", 0.0) or 0.0),
        "adaptive_selection_avg_health_penalty": float(selection.get("avg_health_penalty", 0.0) or 0.0),
    }


def _continual_learning_absorption_summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    unloaded_sources: list[str] = []
    for run in runs:
        if _is_postprocess_evidence_run(run):
            continue
        config = run.get("adaptive_config", {})
        if not isinstance(config, dict):
            continue
        sources = [
            source
            for source in config.get("continual_learning_sources", []) or []
            if isinstance(source, dict)
        ]
        if not sources:
            continue
        continual_learning_enabled = bool(config.get("continual_learning", True))
        label = _run_label(run)
        loaded_sources = [source for source in sources if bool(source.get("loaded"))]
        for source in sources:
            if continual_learning_enabled and not bool(source.get("loaded")):
                unloaded_sources.append(f"{label}:{source.get('path', '')}:{source.get('reason', 'not_loaded')}")
        state = run.get("adaptive_learning_state", {})
        memory = state.get("continual_priority_memory", {}) if isinstance(state, dict) else {}
        imported_ledgers = int(memory.get("imported_ledger_count", 0) or 0) if isinstance(memory, dict) else 0
        imported_families = int(memory.get("imported_family_count", 0) or 0) if isinstance(memory, dict) else 0
        family_priorities = memory.get("family_priorities", {}) if isinstance(memory, dict) else {}
        family_count = len(family_priorities) if isinstance(family_priorities, dict) else 0
        feature_counts = memory.get("feature_counts", {}) if isinstance(memory, dict) else {}
        feature_count = len(feature_counts) if isinstance(feature_counts, dict) else 0
        row = {
            "label": label,
            "continual_learning_enabled": continual_learning_enabled,
            "source_count": len(sources),
            "loaded_source_count": len(loaded_sources),
            "imported_ledger_count": imported_ledgers,
            "imported_family_count": imported_families,
            "family_count": family_count,
            "feature_count": feature_count,
        }
        rows.append(row)
        if (
            continual_learning_enabled
            and loaded_sources
            and (imported_ledgers <= 0 or imported_families <= 0 or family_count <= 0)
        ):
            missing.append(f"{label}:loaded_sources_not_absorbed")
    absorbed = [
        row
        for row in rows
        if bool(row.get("continual_learning_enabled", True))
        and int(row["loaded_source_count"]) > 0
        and int(row["imported_ledger_count"]) > 0
        and int(row["imported_family_count"]) > 0
        and int(row["family_count"]) > 0
    ]
    enabled_rows = [row for row in rows if bool(row.get("continual_learning_enabled", True))]
    return {
        "declared_run_count": len(rows),
        "enabled_declared_run_count": len(enabled_rows),
        "absorbed_run_count": len(absorbed),
        "declared_source_count": sum(int(row["source_count"]) for row in rows),
        "enabled_declared_source_count": sum(int(row["source_count"]) for row in enabled_rows),
        "loaded_source_count": sum(int(row["loaded_source_count"]) for row in enabled_rows),
        "max_imported_ledger_count": max((int(row["imported_ledger_count"]) for row in rows), default=0),
        "max_imported_family_count": max((int(row["imported_family_count"]) for row in rows), default=0),
        "max_family_count": max((int(row["family_count"]) for row in rows), default=0),
        "max_feature_count": max((int(row["feature_count"]) for row in rows), default=0),
        "missing": sorted(set(missing)),
        "unloaded_sources": sorted(set(unloaded_sources)),
        "rows": rows[:32],
    }


def _closed_loop_state_summary_has_signal(summary: Any) -> bool:
    if not isinstance(summary, dict):
        return False
    for key in (
        "seen_signature_count",
        "signal_seen_signature_count",
        "feedback_interesting_case_count",
        "feedback_stored_candidate_family_count",
        "feedback_stored_target_key_count",
        "guidance_feature_count",
        "guidance_frontier_bucket_count",
        "guidance_candidate_bug_family_count",
    ):
        if int(summary.get(key, 0) or 0) > 0:
            return True
    return False


def _adaptive_learning_health_from_summary(summary: Any) -> dict[str, Any]:
    if not isinstance(summary, dict):
        return {}
    health = summary.get("adaptive_learning_health", {})
    if not isinstance(health, dict):
        return {}
    if str(health.get("schema_version", "") or "") != "adaptive-learning-health-v1":
        return {}
    total_pulls = int(health.get("total_pulls", 0) or 0)
    exploration = health.get("exploration_memory", {})
    exploration_records = (
        int(exploration.get("total_records", 0) or 0)
        if isinstance(exploration, dict)
        else 0
    )
    if total_pulls <= 0 and exploration_records <= 0:
        return {}
    continual = health.get("continual_priority_memory", {})
    continual_imported_family_count = 0
    continual_family_count = 0
    if isinstance(continual, dict):
        continual_imported_family_count = int(continual.get("imported_family_count", 0) or 0)
        continual_family_count = int(continual.get("family_count", 0) or 0)
    return {
        "total_pulls": total_pulls,
        "bandit_count": int(health.get("bandit_count", 0) or 0),
        "arm_count": int(health.get("arm_count", 0) or 0),
        "reward_model_update_count": int(health.get("reward_model_update_count", 0) or 0),
        "reward_model_feature_count": int(health.get("reward_model_feature_count", 0) or 0),
        "version_memory_key_count": int(health.get("version_memory_key_count", 0) or 0),
        "runtime_cost_observation_count": int(health.get("runtime_cost_observation_count", 0) or 0),
        "runtime_cost_total": float(health.get("runtime_cost_total", 0.0) or 0.0),
        "value_catalog_entry_pulls": int(health.get("value_catalog_entry_pulls", 0) or 0),
        "value_catalog_entry_arm_count": int(health.get("value_catalog_entry_arm_count", 0) or 0),
        "bd_axis_weight_pulls": int(health.get("bd_axis_weight_pulls", 0) or 0),
        "bd_axis_weight_arm_count": int(health.get("bd_axis_weight_arm_count", 0) or 0),
        "seed_energy_tier_pulls": int(health.get("seed_energy_tier_pulls", 0) or 0),
        "seed_energy_tier_arm_count": int(health.get("seed_energy_tier_arm_count", 0) or 0),
        "champion_graft_donor_pulls": int(health.get("champion_graft_donor_pulls", 0) or 0),
        "champion_graft_donor_arm_count": int(health.get("champion_graft_donor_arm_count", 0) or 0),
        "continual_imported_family_count": continual_imported_family_count,
        "continual_family_count": continual_family_count,
        "avg_health_penalty": float(health.get("avg_health_penalty", 0.0) or 0.0),
        "max_health_penalty": float(health.get("max_health_penalty", 0.0) or 0.0),
        "avg_uncertainty": float(health.get("avg_uncertainty", 0.0) or 0.0),
        "exploration_records": exploration_records,
    }


def _quality_archive_health_from_summary(summary: Any) -> dict[str, Any]:
    if not isinstance(summary, dict):
        return {}
    health = summary.get("quality_archive_health", {})
    if isinstance(health, dict) and str(health.get("schema_version", "") or "") == "quality-archive-health-v1":
        return {
            "cell_count": int(health.get("cell_count", 0) or 0),
            "hierarchical_enabled": bool(health.get("hierarchical_enabled", False)),
            "child_cell_count": int(health.get("child_cell_count", 0) or 0),
            "split_cell_count": int(health.get("split_cell_count", 0) or 0),
            "seed_count": int(health.get("seed_count", 0) or 0),
            "elite_seed_count": int(health.get("elite_seed_count", 0) or 0),
            "reward_count": int(health.get("reward_count", 0) or 0),
            "outcome_count": int(health.get("outcome_count", 0) or 0),
            "invalid_count": int(health.get("invalid_count", 0) or 0),
            "fallback_count": int(health.get("fallback_count", 0) or 0),
            "false_positive_count": int(health.get("false_positive_count", 0) or 0),
        }
    archive = summary.get("quality_archive", {})
    if isinstance(archive, dict) and str(archive.get("schema_version", "") or "") in {
        "quality-diversity-archive-v1",
        "quality-diversity-archive-v2",
    }:
        cells = [cell for cell in archive.get("cells", []) or [] if isinstance(cell, dict)]
        return {
            "cell_count": len(cells),
            "hierarchical_enabled": bool(archive.get("enable_hierarchical", False)),
            "child_cell_count": sum(
                len([child for child in cell.get("children", []) or [] if isinstance(child, dict)])
                for cell in cells
            ),
            "split_cell_count": sum(1 for cell in cells if str(cell.get("split_axis", "") or "")),
            "seed_count": sum(
                len([seed for seed in cell.get("seeds", []) or [] if isinstance(seed, dict)])
                for cell in cells
            ),
            "elite_seed_count": 0,
            "reward_count": sum(int(cell.get("reward_count", 0) or 0) for cell in cells),
            "outcome_count": sum(int(cell.get("outcome_count", 0) or 0) for cell in cells),
            "invalid_count": sum(int(cell.get("invalid_count", 0) or 0) for cell in cells),
            "fallback_count": sum(int(cell.get("fallback_count", 0) or 0) for cell in cells),
            "false_positive_count": sum(int(cell.get("false_positive_count", 0) or 0) for cell in cells),
        }
    return {}


def _seed_quota_health_from_summary(summary: Any) -> dict[str, Any]:
    if not isinstance(summary, dict):
        return {}
    health = summary.get("seed_quota_health", {})
    if not isinstance(health, dict):
        return {}
    return {
        "enabled": bool(health.get("enabled", False)),
        "active": bool(health.get("active", False)),
        "cluster_count": int(health.get("cluster_count", 0) or 0),
        "seed_count": int(health.get("seed_count", 0) or 0),
    }


def _lhs_seeding_observed(run: dict[str, Any]) -> bool:
    meta = run.get("run_meta", {})
    if isinstance(meta, dict):
        lhs = meta.get("lhs_seeding", {})
        if isinstance(lhs, dict) and bool(lhs.get("enabled", False)):
            return True
    config = run.get("config", {})
    return isinstance(config, dict) and bool(config.get("enable_lhs_seeding", False))


def _champion_corpus_health_from_run(run: dict[str, Any]) -> dict[str, Any]:
    health: dict[str, Any] = {}
    meta = run.get("run_meta", {})
    if isinstance(meta, dict):
        corpus = meta.get("champion_corpus", {})
        if isinstance(corpus, dict):
            health.update(
                {
                    "enabled": bool(corpus.get("enabled", False)),
                    "injected_count": int(corpus.get("injected_count", 0) or 0),
                }
            )
    summary = run.get("closed_loop_state_summary", {})
    if isinstance(summary, dict):
        corpus_summary = summary.get("champion_corpus_health", {})
        if isinstance(corpus_summary, dict):
            health.setdefault("enabled", bool(corpus_summary.get("enabled", False)))
            health["promoted_family_count"] = int(corpus_summary.get("promoted_family_count", 0) or 0)
            health["family_hit_count"] = int(corpus_summary.get("family_hit_count", 0) or 0)
    health["active"] = bool(
        health.get("enabled", False)
        and (
            int(health.get("injected_count", 0) or 0) > 0
            or int(health.get("promoted_family_count", 0) or 0) > 0
            or int(health.get("family_hit_count", 0) or 0) > 0
        )
    )
    return health


def _adaptive_learning_health_rollup(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "run_count": 0,
            "total_pulls": 0,
            "reward_model_update_count": 0,
            "reward_model_feature_count": 0,
            "version_memory_key_count": 0,
            "runtime_cost_observation_count": 0,
            "runtime_cost_total": 0.0,
            "bd_axis_weight_pulls": 0,
            "bd_axis_weight_arm_count": 0,
            "seed_energy_tier_pulls": 0,
            "seed_energy_tier_arm_count": 0,
            "champion_graft_donor_pulls": 0,
            "champion_graft_donor_arm_count": 0,
            "continual_imported_family_count": 0,
            "continual_family_count": 0,
            "max_health_penalty": 0.0,
            "avg_uncertainty": 0.0,
            "exploration_records": 0,
            "labels": [],
        }
    return {
        "run_count": len(rows),
        "total_pulls": sum(int(row.get("total_pulls", 0) or 0) for row in rows),
        "reward_model_update_count": sum(int(row.get("reward_model_update_count", 0) or 0) for row in rows),
        "reward_model_feature_count": sum(int(row.get("reward_model_feature_count", 0) or 0) for row in rows),
        "version_memory_key_count": sum(int(row.get("version_memory_key_count", 0) or 0) for row in rows),
        "runtime_cost_observation_count": sum(int(row.get("runtime_cost_observation_count", 0) or 0) for row in rows),
        "runtime_cost_total": sum(float(row.get("runtime_cost_total", 0.0) or 0.0) for row in rows),
        "bd_axis_weight_pulls": sum(int(row.get("bd_axis_weight_pulls", 0) or 0) for row in rows),
        "bd_axis_weight_arm_count": sum(int(row.get("bd_axis_weight_arm_count", 0) or 0) for row in rows),
        "seed_energy_tier_pulls": sum(int(row.get("seed_energy_tier_pulls", 0) or 0) for row in rows),
        "seed_energy_tier_arm_count": sum(int(row.get("seed_energy_tier_arm_count", 0) or 0) for row in rows),
        "champion_graft_donor_pulls": sum(int(row.get("champion_graft_donor_pulls", 0) or 0) for row in rows),
        "champion_graft_donor_arm_count": sum(
            int(row.get("champion_graft_donor_arm_count", 0) or 0) for row in rows
        ),
        "continual_imported_family_count": sum(int(row.get("continual_imported_family_count", 0) or 0) for row in rows),
        "continual_family_count": sum(int(row.get("continual_family_count", 0) or 0) for row in rows),
        "max_health_penalty": max(float(row.get("max_health_penalty", 0.0) or 0.0) for row in rows),
        "avg_uncertainty": (
            sum(float(row.get("avg_uncertainty", 0.0) or 0.0) for row in rows) / len(rows)
        ),
        "exploration_records": sum(int(row.get("exploration_records", 0) or 0) for row in rows),
        "labels": sorted(str(row.get("label", "") or "") for row in rows if str(row.get("label", "") or "")),
    }


def _string_list(value: Any) -> list[str]:
    raw_items = value.split(",") if isinstance(value, str) else value or []
    out: list[str] = []
    seen: set[str] = set()
    for raw in raw_items:
        item = str(raw).strip()
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _mapping_copy(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _confirmed_latest_family_counter(confirmations: list[dict[str, Any]]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for confirmation in confirmations:
        family = _latest_confirmation_family(confirmation)
        if family:
            counter[family] += 1
    return counter


def _latest_confirmation_family(confirmation: dict[str, Any]) -> str:
    status = str(confirmation.get("upstream_status", "")).strip()
    if status not in CONFIRMED_LATEST_UPSTREAM_STATUSES:
        return ""
    issue_url = str(confirmation.get("issue_url", "")).strip()
    if not issue_url:
        return ""
    family = str(confirmation.get("family", "")).strip()
    if family:
        return family
    root = str(confirmation.get("root_cause", "")).strip()
    suspicious = confirmation.get("suspicious_backends", [])
    if not root or not isinstance(suspicious, list):
        return ""
    backends = ",".join(sorted(str(backend).strip() for backend in suspicious if str(backend).strip()))
    return f"{root}@{backends}" if backends else ""


def _live_suite_summary(live_runs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_suite: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "runs": 0,
            "cases": 0,
            "elapsed_s": 0.0,
            "findings": 0,
            "seeds": [],
            "presets": [],
            "families": set(),
            "replay_policy_ok": True,
        }
    )
    for run in live_runs:
        item = by_suite[run["target_suite"]]
        item["runs"] += 1
        item["cases"] += int(run["cases"])
        item["elapsed_s"] += float(run["elapsed_s"])
        item["findings"] += int(run["findings"])
        item["seeds"].append(run["seed"])
        item["presets"].append(run["preset"])
        item["families"].update(run["target_families"])
        item["replay_policy_ok"] = item["replay_policy_ok"] and _fresh_replay_policy_ok(run)
    return {
        suite: {
            **item,
            "seeds": sorted({str(seed) for seed in item["seeds"]}),
            "presets": sorted({str(preset) for preset in item["presets"]}),
            "families": sorted(item["families"]),
        }
        for suite, item in sorted(by_suite.items())
    }


def _fresh_replay_policy_ok(run: dict[str, Any]) -> bool:
    return (
        run["evidence_mode"] == "live"
        and run["enable_replay_bug"] is False
        and run["manifest_enable_replay_bug"] is False
        and run["replay_filter_enabled"] is True
    )


def _run_fails_fresh_replay_policy(run: dict[str, Any]) -> bool:
    return not _fresh_replay_policy_ok(run)


def _historical_confirmed_runs(historical_runs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    confirmed_ids = {spec.bug_id for spec in list_historical_bugs(include_pending=False)}
    out = {}
    for run in historical_runs:
        bug_id = run["known_bug_id"]
        if bug_id in confirmed_ids and run["enable_replay_bug"] is True:
            out[bug_id] = run
    return out


def _is_module_ablation_run(run: dict[str, Any]) -> bool:
    return (
        is_ablation_experiment_row(run)
        and str(run.get("matrix_id", "") or "").strip() == FINAL_MODULE_ABLATION_MATRIX.id
    )


def _is_contrast_scope_run(run: dict[str, Any]) -> bool:
    if _is_postprocess_evidence_run(run):
        return False
    return is_comparison_experiment_row(run) and str(run.get("evidence_mode") or "") == "comparison"


def _is_postprocess_evidence_run(run: dict[str, Any]) -> bool:
    return str(run.get("evidence_kind", "") or "").strip() == "postprocess_ledger"


def _is_baseline_scope_comparison_run(run: dict[str, Any]) -> bool:
    return _is_contrast_scope_run(run)


_is_reference_scope_comparison_run = _is_contrast_scope_run
_is_comparison_scope_reference_run = _is_contrast_scope_run


def _readiness_gates(
    *,
    thresholds: ReadinessThresholds,
    live_by_suite: dict[str, dict[str, Any]],
    live_families: list[str],
    audited_runs: list[dict[str, Any]],
    validation_runs: list[dict[str, Any]],
    live_runs: list[dict[str, Any]],
    historical_runs: list[dict[str, Any]],
    historical_confirmed: dict[str, dict[str, Any]],
    seeded_runs: list[dict[str, Any]],
    ablation_runs: list[dict[str, Any]],
    comparison_runs: list[dict[str, Any]],
    module_ablation_comparison: dict[str, Any],
    baseline_comparison: dict[str, Any],
    scan_run_logs: bool,
    rewardable_live_families: Counter[str],
    confirmed_live_families: Counter[str],
    paper_run_journal_files: list[Path],
    paper_run_journal_covered_runs: int,
    paper_run_journal_issues: list[str],
    policy: ReadinessPolicy,
    workspace_git_commit: str,
    adaptive_component_ablation: dict[str, Any],
    final_matrix_coverage: dict[str, Any],
    transferability_scope: dict[str, Any],
    cross_version_ledger: dict[str, Any],
    runtime_efficiency: dict[str, Any],
    discovery_responsiveness: dict[str, Any],
    closed_loop_state_persistence: dict[str, Any],
    adaptive_live_component_evidence: dict[str, Any],
    continual_learning_absorption: dict[str, Any],
    target_version_audit: dict[str, Any],
) -> list[dict[str, Any]]:
    required_suites = set(policy.required_live_suites)
    missing_suites = sorted(required_suites - set(live_by_suite))
    shallow_suites = sorted(
        suite
        for suite in required_suites & set(live_by_suite)
        if live_by_suite[suite]["cases"] < thresholds.min_live_cases_per_suite
        or live_by_suite[suite]["elapsed_s"] < thresholds.min_live_duration_hours * 3600
    )
    missing_families = sorted(set(policy.required_live_families) - set(live_families))
    replay_policy_issues = sorted({run["target_suite"] for run in live_runs if _run_fails_fresh_replay_policy(run)})
    registered_runs = _registered_experiment_runs(audited_runs)
    missing_adaptive_component_ablations = sorted(
        set(thresholds.required_adaptive_component_ablations)
        - set(adaptive_component_ablation.get("disabled_components", []))
    )
    return [
        _gate(
            "run_log_scan",
            scan_run_logs,
            "full run-log scan is required before claiming final paper readiness",
        ),
        _gate(
            "paper_run_journal",
            not paper_run_journal_issues,
            (
                f"journal_files={len(paper_run_journal_files)} "
                f"required_runs={len(audited_runs)} "
                f"covered_runs={paper_run_journal_covered_runs}"
            ),
            issues=paper_run_journal_issues,
            missing=paper_run_journal_issues,
        ),
        _gate(
            "target_version_audit",
            (
                not live_runs
                or not thresholds.require_target_version_audit
                or (
                    target_version_audit.get("valid_audit_count", 0) > 0
                    and target_version_audit.get("all_target_packages_up_to_date", False)
                    and not target_version_audit.get("invalid_files", [])
                )
            ),
            (
                f"required={thresholds.require_target_version_audit} "
                f"live_runs={len(live_runs)} "
                f"valid_audits={target_version_audit.get('valid_audit_count', 0)} "
                f"target_packages={target_version_audit.get('target_package_count', 0)} "
                f"outdated={target_version_audit.get('outdated_target_package_count', 0)} "
                f"unknown_latest={target_version_audit.get('unknown_latest_target_package_count', 0)}"
            ),
            audit_files=target_version_audit.get("audit_files", []),
            valid_audit_files=target_version_audit.get("valid_audit_files", []),
            invalid_files=target_version_audit.get("invalid_files", []),
            invalid_reasons=target_version_audit.get("invalid_reasons", {}),
            outdated_target_packages=target_version_audit.get("outdated_target_packages", []),
            unknown_latest_target_packages=target_version_audit.get("unknown_latest_target_packages", []),
            all_target_packages_up_to_date=target_version_audit.get("all_target_packages_up_to_date", False),
        ),
        _gate(
            "short_validation",
            bool(validation_runs) or not thresholds.require_validation,
            f"required={thresholds.require_validation} observed_runs={len(validation_runs)}",
            suites=sorted({run["target_suite"] for run in validation_runs}),
        ),
        _gate(
            "live_suite_breadth",
            not missing_suites,
            f"required={len(policy.required_live_suites)} covered={len(required_suites - set(missing_suites))}",
            missing=missing_suites,
        ),
        _gate(
            "live_target_family_breadth",
            not missing_families,
            f"required={','.join(policy.required_live_families)} covered={','.join(live_families) or 'none'}",
            missing=missing_families,
        ),
        _gate(
            "transferability_scope",
            (
                transferability_scope.get("family_count", 0) >= thresholds.min_transfer_target_families
                and not transferability_scope.get("missing_required_families", [])
                and transferability_scope.get("non_primary_family_count", 0) > 0
            )
            or not thresholds.require_transferability_scope,
            (
                f"required={thresholds.require_transferability_scope} "
                f"min_families={thresholds.min_transfer_target_families} "
                f"observed_families={transferability_scope.get('family_count', 0)} "
                f"non_primary={transferability_scope.get('non_primary_family_count', 0)}"
            ),
            families=transferability_scope.get("families", []),
            required_families=transferability_scope.get("required_families", []),
            missing=transferability_scope.get("missing_required_families", []),
            non_primary_families=transferability_scope.get("non_primary_families", []),
        ),
        _gate(
            "live_depth",
            not shallow_suites and not missing_suites,
            (
                f"min_cases_per_suite={thresholds.min_live_cases_per_suite} "
                f"min_elapsed_hours={thresholds.min_live_duration_hours:g}"
            ),
            missing=missing_suites,
            shallow=shallow_suites,
        ),
        _gate(
            "fresh_replay_policy",
            not replay_policy_issues and bool(live_runs),
            "live runs must keep enable_replay_bug=false and replay filter enabled",
            issues=replay_policy_issues,
            bad_suites=replay_policy_issues,
        ),
        _gate(
            "structured_experiment_identity",
            not _structured_identity_issues(registered_runs),
            "registered final-harness runs should carry matrix/comparison/variant/canonical role/scope/oracle structured identity",
            issues=_structured_identity_issues(registered_runs),
            missing=_structured_identity_issues(registered_runs),
        ),
        _gate(
            "final_matrix_coverage",
            not final_matrix_coverage.get("missing_matrix_ids", []),
            (
                f"required={len(final_matrix_coverage.get('required_matrix_ids', []))} "
                f"observed={len(final_matrix_coverage.get('observed_matrix_ids', []))}"
            ),
            required_matrix_ids=final_matrix_coverage.get("required_matrix_ids", []),
            observed_matrix_ids=final_matrix_coverage.get("observed_matrix_ids", []),
            missing=final_matrix_coverage.get("missing_matrix_ids", []),
            observed_matrix_counts=final_matrix_coverage.get("observed_matrix_counts", {}),
        ),
        _gate(
            "structured_semantic_focus",
            not _structured_semantic_focus_issues(registered_runs),
            "registered final-harness runs should preserve structured semantic focus metadata from the experiment catalog",
            issues=_structured_semantic_focus_issues(registered_runs),
        ),
        _gate(
            "effective_guidance_targets",
            not _effective_guidance_target_issues(registered_runs),
            "registered final-harness runs should preserve the runtime-effective guidance target set implied by config and structured semantic focus",
            issues=_effective_guidance_target_issues(registered_runs),
        ),
        _gate(
            "stage_level_profiling",
            not _stage_profile_issues(
                [
                    run
                    for run in audited_runs
                    if run["evidence_mode"] != "historical" and not _is_postprocess_evidence_run(run)
                ]
            ),
            "all non-historical runs should expose stage-level profiling and throughput in run meta",
            issues=_stage_profile_issues(
                [
                    run
                    for run in audited_runs
                    if run["evidence_mode"] != "historical" and not _is_postprocess_evidence_run(run)
                ]
            ),
            missing=_stage_profile_issues(
                [
                    run
                    for run in audited_runs
                    if run["evidence_mode"] != "historical" and not _is_postprocess_evidence_run(run)
                ]
            ),
        ),
        _gate(
            "runtime_efficiency",
            not runtime_efficiency.get("issues", []) or not thresholds.require_runtime_efficiency,
            (
                f"required={thresholds.require_runtime_efficiency} "
                f"min_throughput_cases_s={thresholds.min_throughput_cases_s:g} "
                f"max_scheduler_feedback_share={thresholds.max_scheduler_feedback_share:g} "
                f"min_scheduler_feedback_cases={thresholds.min_scheduler_feedback_cases} "
                f"observed_min_throughput_cases_s={runtime_efficiency.get('min_throughput_cases_s', 0.0):.6g} "
                f"observed_max_scheduler_feedback_share={runtime_efficiency.get('max_scheduler_feedback_share', 0.0):.6g}"
            ),
            issues=runtime_efficiency.get("issues", []),
            min_throughput_cases_s=runtime_efficiency.get("min_throughput_cases_s", 0.0),
            avg_throughput_cases_s=runtime_efficiency.get("avg_throughput_cases_s", 0.0),
            max_scheduler_feedback_share=runtime_efficiency.get("max_scheduler_feedback_share", 0.0),
            scheduler_feedback_share_run_count=runtime_efficiency.get("scheduler_feedback_share_run_count", 0),
            scheduler_feedback_share_skipped_run_count=runtime_efficiency.get(
                "scheduler_feedback_share_skipped_run_count",
                0,
            ),
            min_scheduler_feedback_cases=runtime_efficiency.get("min_scheduler_feedback_cases", 0),
        ),
        _gate(
            "discovery_responsiveness",
            (
                discovery_responsiveness.get("observed_run_count", 0) > 0
                and discovery_responsiveness.get("best_first_candidate_elapsed_s") is not None
                and float(discovery_responsiveness.get("best_first_candidate_elapsed_s", 0.0))
                <= thresholds.max_first_candidate_elapsed_s
            )
            or not thresholds.require_discovery_responsiveness,
            (
                f"required={thresholds.require_discovery_responsiveness} "
                f"max_first_candidate_elapsed_s={thresholds.max_first_candidate_elapsed_s:g} "
                f"observed_runs={discovery_responsiveness.get('observed_run_count', 0)} "
                f"best_elapsed_s={discovery_responsiveness.get('best_first_candidate_elapsed_s')}"
            ),
            observed_run_count=discovery_responsiveness.get("observed_run_count", 0),
            best_first_candidate_elapsed_s=discovery_responsiveness.get("best_first_candidate_elapsed_s"),
            best_first_candidate_case_index=discovery_responsiveness.get("best_first_candidate_case_index"),
            avg_candidate_bug_discovery_auc=discovery_responsiveness.get("avg_candidate_bug_discovery_auc", 0.0),
            best_run_label=discovery_responsiveness.get("best_run_label", ""),
        ),
        _gate(
            "closed_loop_state_persistence",
            (
                closed_loop_state_persistence.get("required_run_count", 0) > 0
                and closed_loop_state_persistence.get("persisted_run_count", 0)
                == closed_loop_state_persistence.get("required_run_count", 0)
                and not closed_loop_state_persistence.get("missing", [])
                and not closed_loop_state_persistence.get("weak", [])
                and not closed_loop_state_persistence.get("missing_health", [])
            )
            or not thresholds.require_closed_loop_state_persistence,
            (
                f"required={thresholds.require_closed_loop_state_persistence} "
                f"required_runs={closed_loop_state_persistence.get('required_run_count', 0)} "
                f"persisted_runs={closed_loop_state_persistence.get('persisted_run_count', 0)} "
                f"state_bytes={closed_loop_state_persistence.get('total_state_bytes', 0)}"
            ),
            required_run_count=closed_loop_state_persistence.get("required_run_count", 0),
            persisted_run_count=closed_loop_state_persistence.get("persisted_run_count", 0),
            persisted_run_labels=closed_loop_state_persistence.get("persisted_run_labels", []),
            missing=closed_loop_state_persistence.get("missing", []),
            weak=closed_loop_state_persistence.get("weak", []),
            missing_health=closed_loop_state_persistence.get("missing_health", []),
            total_state_bytes=closed_loop_state_persistence.get("total_state_bytes", 0),
            adaptive_learning_health=closed_loop_state_persistence.get("adaptive_learning_health", {}),
        ),
        _gate(
            "adaptive_live_component_evidence",
            (
                adaptive_live_component_evidence.get("declared_run_count", 0) <= 0
                or (
                    adaptive_live_component_evidence.get("enabled_run_count", 0) > 0
                    and not adaptive_live_component_evidence.get("missing", [])
                )
            )
            or not thresholds.require_adaptive_live_component_evidence,
            (
                f"required={thresholds.require_adaptive_live_component_evidence} "
                f"declared_runs={adaptive_live_component_evidence.get('declared_run_count', 0)} "
                f"enabled_runs={adaptive_live_component_evidence.get('enabled_run_count', 0)} "
                f"declared_components={','.join(adaptive_live_component_evidence.get('declared_components', [])) or 'none'} "
                f"proven_components={','.join(adaptive_live_component_evidence.get('proven_components', [])) or 'none'} "
                f"adaptive_selection_count={adaptive_live_component_evidence.get('adaptive_selection_total_count', 0)}"
            ),
            declared_run_count=adaptive_live_component_evidence.get("declared_run_count", 0),
            enabled_run_count=adaptive_live_component_evidence.get("enabled_run_count", 0),
            declared_components=adaptive_live_component_evidence.get("declared_components", []),
            proven_components=adaptive_live_component_evidence.get("proven_components", []),
            missing=adaptive_live_component_evidence.get("missing", []),
            run_labels=adaptive_live_component_evidence.get("run_labels", []),
            scheduler_learning_total_pulls=adaptive_live_component_evidence.get("scheduler_learning_total_pulls", 0),
            reward_model_update_count=adaptive_live_component_evidence.get("reward_model_update_count", 0),
            exploration_records=adaptive_live_component_evidence.get("exploration_records", 0),
            version_memory_key_count=adaptive_live_component_evidence.get("version_memory_key_count", 0),
            continual_imported_family_count=adaptive_live_component_evidence.get("continual_imported_family_count", 0),
            quality_archive_cell_count=adaptive_live_component_evidence.get("quality_archive_cell_count", 0),
            quality_archive_child_cell_count=adaptive_live_component_evidence.get(
                "quality_archive_child_cell_count",
                0,
            ),
            quality_archive_split_cell_count=adaptive_live_component_evidence.get(
                "quality_archive_split_cell_count",
                0,
            ),
            quality_archive_seed_count=adaptive_live_component_evidence.get("quality_archive_seed_count", 0),
            value_catalog_entry_pulls=adaptive_live_component_evidence.get("value_catalog_entry_pulls", 0),
            value_catalog_entry_arm_count=adaptive_live_component_evidence.get("value_catalog_entry_arm_count", 0),
            seed_energy_tier_pulls=adaptive_live_component_evidence.get("seed_energy_tier_pulls", 0),
            seed_energy_tier_arm_count=adaptive_live_component_evidence.get("seed_energy_tier_arm_count", 0),
            champion_graft_donor_pulls=adaptive_live_component_evidence.get("champion_graft_donor_pulls", 0),
            champion_graft_donor_arm_count=adaptive_live_component_evidence.get("champion_graft_donor_arm_count", 0),
            seed_quota_active_run_count=adaptive_live_component_evidence.get("seed_quota_active_run_count", 0),
            seed_quota_cluster_count=adaptive_live_component_evidence.get("seed_quota_cluster_count", 0),
            seed_quota_seed_count=adaptive_live_component_evidence.get("seed_quota_seed_count", 0),
            lhs_seeding_enabled_run_count=adaptive_live_component_evidence.get("lhs_seeding_enabled_run_count", 0),
            champion_corpus_enabled_run_count=adaptive_live_component_evidence.get("champion_corpus_enabled_run_count", 0),
            champion_corpus_injected_count=adaptive_live_component_evidence.get("champion_corpus_injected_count", 0),
            champion_corpus_promoted_family_count=adaptive_live_component_evidence.get(
                "champion_corpus_promoted_family_count",
                0,
            ),
            runtime_cost_observation_count=adaptive_live_component_evidence.get("runtime_cost_observation_count", 0),
            annealing_observation_count=adaptive_live_component_evidence.get("annealing_observation_count", 0),
            adaptive_selection_run_count=adaptive_live_component_evidence.get("adaptive_selection_run_count", 0),
            adaptive_selection_total_count=adaptive_live_component_evidence.get("adaptive_selection_total_count", 0),
            adaptive_selection_scopes=adaptive_live_component_evidence.get("adaptive_selection_scopes", []),
            rows=adaptive_live_component_evidence.get("rows", []),
        ),
        _gate(
            "cross_version_continual_learning",
            (
                cross_version_ledger.get("ledger_count", 0) > 0
                and cross_version_ledger.get("max_version_count", 0) >= thresholds.min_cross_version_ledger_versions
                and cross_version_ledger.get("max_family_count", 0) >= thresholds.min_cross_version_ledger_families
                and (
                    cross_version_ledger.get("health_observation_count", 0) > 0
                    or not thresholds.require_cross_version_health_feedback
                )
                and not cross_version_ledger.get("invalid_ledger_files", [])
            )
            or not thresholds.require_cross_version_ledger,
            (
                f"required={thresholds.require_cross_version_ledger} "
                f"health_feedback_required={thresholds.require_cross_version_health_feedback} "
                f"min_versions={thresholds.min_cross_version_ledger_versions} "
                f"min_families={thresholds.min_cross_version_ledger_families} "
                f"observed_ledgers={cross_version_ledger.get('ledger_count', 0)} "
                f"observed_max_versions={cross_version_ledger.get('max_version_count', 0)} "
                f"observed_max_families={cross_version_ledger.get('max_family_count', 0)} "
                f"health_observations={cross_version_ledger.get('health_observation_count', 0)}"
            ),
            ledger_files=cross_version_ledger.get("ledger_files", []),
            valid_ledger_files=cross_version_ledger.get("valid_ledger_files", []),
            invalid_ledger_files=cross_version_ledger.get("invalid_ledger_files", []),
            invalid_reasons=cross_version_ledger.get("invalid_reasons", {}),
            transition_counts=cross_version_ledger.get("transition_counts", {}),
            priority_families=cross_version_ledger.get("priority_families", []),
            health_feedback_required=thresholds.require_cross_version_health_feedback,
            health_observation_count=cross_version_ledger.get("health_observation_count", 0),
            max_health_observation_count=cross_version_ledger.get("max_health_observation_count", 0),
            invalid_case_count=cross_version_ledger.get("invalid_case_count", 0),
            fallback_case_count=cross_version_ledger.get("fallback_case_count", 0),
            false_positive_count=cross_version_ledger.get("false_positive_count", 0),
            min_throughput_cases_s=cross_version_ledger.get("min_throughput_cases_s", 0.0),
            max_invalid_rate=cross_version_ledger.get("max_invalid_rate", 0.0),
            max_false_positive_rate=cross_version_ledger.get("max_false_positive_rate", 0.0),
        ),
        _gate(
            "continual_learning_absorption",
            (
                continual_learning_absorption.get("declared_run_count", 0) <= 0
                or continual_learning_absorption.get("enabled_declared_run_count", 0) <= 0
                or (
                    continual_learning_absorption.get("absorbed_run_count", 0) > 0
                    and continual_learning_absorption.get("loaded_source_count", 0) > 0
                    and not continual_learning_absorption.get("missing", [])
                    and not continual_learning_absorption.get("unloaded_sources", [])
                )
            )
            or not thresholds.require_cross_version_ledger,
            (
                f"declared_runs={continual_learning_absorption.get('declared_run_count', 0)} "
                f"absorbed_runs={continual_learning_absorption.get('absorbed_run_count', 0)} "
                f"loaded_sources={continual_learning_absorption.get('loaded_source_count', 0)} "
                f"max_imported_ledgers={continual_learning_absorption.get('max_imported_ledger_count', 0)} "
                f"max_imported_families={continual_learning_absorption.get('max_imported_family_count', 0)}"
            ),
            declared_run_count=continual_learning_absorption.get("declared_run_count", 0),
            enabled_declared_run_count=continual_learning_absorption.get("enabled_declared_run_count", 0),
            absorbed_run_count=continual_learning_absorption.get("absorbed_run_count", 0),
            declared_source_count=continual_learning_absorption.get("declared_source_count", 0),
            enabled_declared_source_count=continual_learning_absorption.get("enabled_declared_source_count", 0),
            loaded_source_count=continual_learning_absorption.get("loaded_source_count", 0),
            max_imported_ledger_count=continual_learning_absorption.get("max_imported_ledger_count", 0),
            max_imported_family_count=continual_learning_absorption.get("max_imported_family_count", 0),
            max_family_count=continual_learning_absorption.get("max_family_count", 0),
            max_feature_count=continual_learning_absorption.get("max_feature_count", 0),
            missing=continual_learning_absorption.get("missing", []),
            unloaded_sources=continual_learning_absorption.get("unloaded_sources", []),
            rows=continual_learning_absorption.get("rows", []),
        ),
        _gate(
            "latest_live_provenance",
            not _live_provenance_issues(live_runs, workspace_git_commit=workspace_git_commit) and bool(live_runs),
            "live runs must record a clean frozen authority provenance and match the current workspace revision",
            issues=_live_provenance_issues(live_runs, workspace_git_commit=workspace_git_commit),
            workspace_git_commit=workspace_git_commit,
        ),
        _gate(
            "latest_candidate_bug_families",
            len(rewardable_live_families) >= thresholds.min_live_candidate_families,
            f"required={thresholds.min_live_candidate_families} observed={len(rewardable_live_families)}",
            families=sorted(rewardable_live_families),
        ),
        _gate(
            "latest_confirmed_bug_families",
            len(confirmed_live_families) >= thresholds.min_confirmed_live_families,
            f"required={thresholds.min_confirmed_live_families} observed={len(confirmed_live_families)}",
            families=sorted(confirmed_live_families),
        ),
        _gate(
            "historical_confirmed_replay",
            len(historical_confirmed) >= thresholds.min_historical_confirmed,
            f"required={thresholds.min_historical_confirmed} observed={len(historical_confirmed)}",
            bug_ids=sorted(historical_confirmed),
        ),
        _gate(
            "seeded_sensitivity",
            bool(seeded_runs) or not thresholds.require_seeded,
            f"required={thresholds.require_seeded} observed_runs={len(seeded_runs)}",
        ),
        _gate(
            "module_ablation",
            (
                module_ablation_comparison.get("complete_group_count", 0) > 0
                and not module_ablation_comparison.get("missing_reference_groups", [])
                and not module_ablation_comparison.get("missing_contrast_groups", [])
            )
            or not thresholds.require_ablation,
            (
                f"required={thresholds.require_ablation} observed_runs={len(ablation_runs)} "
                f"complete_groups={module_ablation_comparison.get('complete_group_count', 0)} "
                f"groups={module_ablation_comparison.get('group_count', 0)}"
            ),
            suites=sorted({run["target_suite"] for run in ablation_runs}),
            presets=sorted({run["preset"] for run in ablation_runs}),
            variants=sorted({run["variant_label"] for run in ablation_runs if run.get("variant_label")}),
            matrix_ids=_sorted_matrix_ids(ablation_runs),
            reference_run_count=module_ablation_comparison.get("reference_run_count", 0),
            contrast_run_count=module_ablation_comparison.get("contrast_run_count", 0),
            missing_reference_groups=module_ablation_comparison.get("missing_reference_groups", []),
            missing_contrast_groups=module_ablation_comparison.get("missing_contrast_groups", []),
            groups=module_ablation_comparison.get("groups", []),
            reference_metrics=module_ablation_comparison.get("reference_metrics", {}),
            contrast_metrics=module_ablation_comparison.get("contrast_metrics", {}),
        ),
        _gate(
            "adaptive_component_ablation",
            (
                len(adaptive_component_ablation.get("disabled_components", []))
                >= thresholds.min_adaptive_component_ablations
                and not missing_adaptive_component_ablations
                and int(adaptive_component_ablation.get("reference_run_count", 0) or 0) > 0
                and int(adaptive_component_ablation.get("contrast_run_count", 0) or 0) > 0
            )
            or not thresholds.require_adaptive_component_ablation,
            (
                f"required={thresholds.require_adaptive_component_ablation} "
                f"min_components={thresholds.min_adaptive_component_ablations} "
                f"required_components={','.join(thresholds.required_adaptive_component_ablations) or 'none'} "
                f"observed_components={len(adaptive_component_ablation.get('disabled_components', []))} "
                f"observed_runs={adaptive_component_ablation.get('run_count', 0)} "
                f"reference_runs={adaptive_component_ablation.get('reference_run_count', 0)} "
                f"contrast_runs={adaptive_component_ablation.get('contrast_run_count', 0)}"
            ),
            components=adaptive_component_ablation.get("components", []),
            disabled_components=adaptive_component_ablation.get("disabled_components", []),
            required_components=list(thresholds.required_adaptive_component_ablations),
            missing_required_components=missing_adaptive_component_ablations,
            run_labels=adaptive_component_ablation.get("run_labels", []),
            reference_run_count=adaptive_component_ablation.get("reference_run_count", 0),
            contrast_run_count=adaptive_component_ablation.get("contrast_run_count", 0),
            reference_run_labels=adaptive_component_ablation.get("reference_run_labels", []),
            contrast_run_labels=adaptive_component_ablation.get("contrast_run_labels", []),
            reference_metrics=adaptive_component_ablation.get("reference_metrics", {}),
            contrast_metrics=adaptive_component_ablation.get("contrast_metrics", {}),
        ),
        _gate(
            "baseline_comparison",
            (
                baseline_comparison.get("complete_group_count", 0) > 0
                and not baseline_comparison.get("missing_reference_groups", [])
                and not baseline_comparison.get("missing_contrast_groups", [])
            )
            or not thresholds.require_comparison,
            (
                f"required={thresholds.require_comparison} observed_runs={len(comparison_runs)} "
                f"complete_groups={baseline_comparison.get('complete_group_count', 0)} "
                f"groups={baseline_comparison.get('group_count', 0)}"
            ),
            suites=sorted({run["target_suite"] for run in comparison_runs}),
            presets=sorted({run["preset"] for run in comparison_runs}),
            variants=sorted({run["variant_label"] for run in comparison_runs if run.get("variant_label")}),
            matrix_ids=_sorted_matrix_ids(comparison_runs),
            reference_run_count=baseline_comparison.get("reference_run_count", 0),
            contrast_run_count=baseline_comparison.get("contrast_run_count", 0),
            missing_reference_groups=baseline_comparison.get("missing_reference_groups", []),
            missing_contrast_groups=baseline_comparison.get("missing_contrast_groups", []),
            groups=baseline_comparison.get("groups", []),
            reference_metrics=baseline_comparison.get("reference_metrics", {}),
            contrast_metrics=baseline_comparison.get("contrast_metrics", {}),
        ),
    ]


def _gate(name: str, passed: bool, detail: str, **extra: Any) -> dict[str, Any]:
    return {"name": name, "passed": passed, "detail": detail, **extra}


def _sorted_matrix_ids(runs: list[dict[str, Any]]) -> list[str]:
    return sorted({str(run["matrix_id"]).strip() for run in runs if str(run["matrix_id"]).strip()})


def _registered_experiment_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [run for run in runs if bool(run.get("registered_experiment"))]


def _structured_identity_issues(runs: list[dict[str, Any]]) -> list[str]:
    issues: list[str] = []
    for run in runs:
        required_fields = (
            str(run.get("matrix_id", "")).strip(),
            str(run.get("comparison_group", "")).strip(),
            str(run.get("variant_id", "")).strip(),
            str(run.get("canonical_comparison_role", "")).strip(),
            str(run.get("scope_kind", "")).strip(),
            str(run.get("oracle_profile", "")).strip(),
        )
        if all(required_fields):
            continue
        issues.append(f"{run['evidence_mode']}:{run['target_suite']}:{run['preset']}")
    return sorted(set(issues))


def _live_provenance_issues(runs: list[dict[str, Any]], *, workspace_git_commit: str) -> list[str]:
    issues: list[str] = []
    if not workspace_git_commit:
        return ["workspace:missing_git_commit"]
    for run in runs:
        provenance = run.get("run_provenance", {}) if isinstance(run.get("run_provenance", {}), dict) else {}
        vcs = provenance.get("vcs", {}) if isinstance(provenance.get("vcs", {}), dict) else {}
        harness = provenance.get("harness", {}) if isinstance(provenance.get("harness", {}), dict) else {}
        freeze_artifacts = (
            provenance.get("freeze_artifacts", {}) if isinstance(provenance.get("freeze_artifacts", {}), dict) else {}
        )
        run_label = f"{run['evidence_mode']}:{run['target_suite']}:{run['preset']}"
        git_commit = str(vcs.get("git_commit", "") or "").strip()
        if not git_commit:
            issues.append(f"{run_label}:missing_git_commit")
        elif git_commit != workspace_git_commit:
            issues.append(f"{run_label}:head_mismatch")
        if vcs.get("workspace_dirty") is not False:
            issues.append(f"{run_label}:dirty_or_unknown_workspace")
        if not bool(harness.get("authority", False)):
            issues.append(f"{run_label}:non_authority")
        if not bool(harness.get("freeze_intent", False)):
            issues.append(f"{run_label}:freeze_not_declared")
        if not bool(harness.get("latest_code_claim", False)):
            issues.append(f"{run_label}:latest_code_not_declared")
        for artifact_name in ("manifest", "pip_freeze", "git_status", "launcher_env", "strategy_snapshot"):
            artifact_path = str(freeze_artifacts.get(artifact_name, "") or "").strip()
            if not artifact_path:
                issues.append(f"{run_label}:missing_{artifact_name}_artifact")
            elif not Path(artifact_path).is_file():
                issues.append(f"{run_label}:missing_{artifact_name}_file")
    return sorted(set(issues))


def _expected_registered_semantic_focus(run: dict[str, Any]) -> tuple[list[str], list[str]] | None:
    matrix = registered_experiment_matrix_for_run(
        evidence_mode=str(run.get("evidence_mode", "") or ""),
        target_suite=str(run.get("target_suite", "") or ""),
        preset=str(run.get("preset", "") or ""),
    )
    if matrix is None:
        return None
    preset = str(run.get("preset", "") or "")
    variant = next((item for item in matrix.variants if item.preset == preset), None)
    if variant is None:
        return None
    return (
        [str(item).strip() for item in variant.semantic_focus_families if str(item).strip()],
        [str(item).strip() for item in variant.semantic_focus_signals if str(item).strip()],
    )


def _structured_semantic_focus_issues(runs: list[dict[str, Any]]) -> list[str]:
    issues: list[str] = []
    for run in runs:
        expected = _expected_registered_semantic_focus(run)
        if expected is None:
            continue
        expected_families, expected_signals = expected
        actual_families = [str(item).strip() for item in run.get("semantic_focus_families", []) if str(item).strip()]
        actual_signals = [str(item).strip() for item in run.get("semantic_focus_signals", []) if str(item).strip()]
        if actual_families == expected_families and actual_signals == expected_signals:
            continue
        issues.append(f"{run['evidence_mode']}:{run['target_suite']}:{run['preset']}")
    return sorted(set(issues))


def _effective_guidance_target_issues(runs: list[dict[str, Any]]) -> list[str]:
    issues: list[str] = []
    for run in runs:
        config = run.get("config", {})
        if not isinstance(config, dict):
            continue
        actual_targets = [str(item).strip() for item in config.get("effective_guidance_targets", []) if str(item).strip()]
        if not actual_targets:
            continue
        try:
            expected_targets = _configured_guidance_targets(
                ExperimentConfig(**{key: value for key, value in config.items() if key != "effective_guidance_targets"})
            )
        except Exception:
            issues.append(f"{run['evidence_mode']}:{run['target_suite']}:{run['preset']}")
            continue
        if actual_targets != expected_targets:
            issues.append(f"{run['evidence_mode']}:{run['target_suite']}:{run['preset']}")
    return sorted(set(issues))


def _stage_profile_present(meta: dict[str, Any]) -> bool:
    stage_profile = meta.get("stage_profile", {})
    if not isinstance(stage_profile, dict):
        return False
    totals = stage_profile.get("totals_ms", {})
    if not isinstance(totals, dict):
        return False
    return all(key in totals for key in STAGE_PROFILE_FIELDS)


def _stage_profile_issues(runs: list[dict[str, Any]]) -> list[str]:
    issues: list[str] = []
    for run in runs:
        if bool(run.get("stage_profile_present")):
            continue
        issues.append(f"{run['evidence_mode']}:{run['target_suite']}:{run['preset']}")
    return sorted(set(issues))


def _stage_profile_totals(meta: dict[str, Any]) -> dict[str, float]:
    stage_profile = meta.get("stage_profile", {})
    if not isinstance(stage_profile, dict):
        return {}
    totals = stage_profile.get("totals_ms", {})
    if not isinstance(totals, dict):
        return {}
    parsed: dict[str, float] = {}
    for field in STAGE_PROFILE_FIELDS:
        if field not in totals:
            continue
        try:
            parsed[field] = float(totals.get(field, 0.0) or 0.0)
        except (TypeError, ValueError):
            parsed[field] = 0.0
    return parsed


def _optional_float(value: Any, *, default: float | None = None) -> float | None:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _optional_int(value: Any, *, default: int | None = None) -> int | None:
    if value in (None, ""):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _scheduler_feedback_share(run: dict[str, Any]) -> float | None:
    totals = run.get("stage_profile_totals", {})
    if not isinstance(totals, dict):
        return None
    total_ms = float(totals.get("total_case_wall_ms", 0.0) or 0.0)
    if total_ms <= 0.0:
        return None
    return max(0.0, float(totals.get("scheduler_feedback_ms", 0.0) or 0.0)) / total_ms


def _scheduler_feedback_share_rows(
    runs: list[dict[str, Any]],
    *,
    thresholds: ReadinessThresholds,
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    skipped: list[str] = []
    min_cases = max(1, int(thresholds.min_scheduler_feedback_cases or 0))
    for run in runs:
        if not bool(run.get("stage_profile_present")):
            continue
        label = f"{run['evidence_mode']}:{run['target_suite']}:{run['preset']}"
        cases = int(run.get("cases", 0) or 0)
        if cases < min_cases:
            skipped.append(f"{label}:cases_below_scheduler_share_min:{cases}/{min_cases}")
            continue
        share = _scheduler_feedback_share(run)
        if share is None:
            rows.append({"label": label, "cases": cases, "share": None})
            continue
        rows.append({"label": label, "cases": cases, "share": float(share)})
    return rows, sorted(set(skipped))


def _runtime_efficiency_issues(
    runs: list[dict[str, Any]],
    *,
    thresholds: ReadinessThresholds,
) -> list[str]:
    issues: list[str] = []
    for run in runs:
        label = f"{run['evidence_mode']}:{run['target_suite']}:{run['preset']}"
        throughput = float(run.get("throughput_cases_s", 0.0) or 0.0)
        if throughput < thresholds.min_throughput_cases_s:
            issues.append(f"{label}:throughput_below_min")
    scheduler_share_rows, _ = _scheduler_feedback_share_rows(runs, thresholds=thresholds)
    for row in scheduler_share_rows:
        label = str(row.get("label", "") or "")
        share = row.get("share")
        if share is None:
            issues.append(f"{label}:stage_total_unusable")
        elif float(share) > thresholds.max_scheduler_feedback_share:
            issues.append(f"{label}:scheduler_feedback_share_high")
    return sorted(set(issues))


def _gate_by_name(audit: dict[str, Any], name: str) -> dict[str, Any]:
    for gate in audit.get("gates", []) or []:
        if isinstance(gate, dict) and gate.get("name") == name:
            return gate
    return {}


def _render_markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# Final Experiment Readiness",
        "",
        f"- Created at: `{audit['created_at']}`",
        f"- Ready: `{str(audit['ready']).lower()}`",
        f"- Manifests: `{len(audit['manifest_files'])}`",
        "",
        "## Gates",
        "",
        "| gate | status | detail |",
        "|---|---|---|",
    ]
    for gate in audit["gates"]:
        lines.append(f"| {gate['name']} | {'pass' if gate['passed'] else 'fail'} | {gate['detail']} |")
    summary = audit["summary"]
    icse_quality = summary.get("icse_experiment_quality", {})
    icse_priorities = ", ".join(
        item.get("dimension", "")
        for item in (icse_quality.get("optimization_priorities", []) or [])[:3]
        if item.get("dimension")
    ) or "none"
    lines.extend(
        [
            "",
            "## Summary",
            "",
            (
                "- ICSE experiment quality: "
                f"score `{icse_quality.get('overall_score', 0.0):.6g}`; "
                f"grade `{icse_quality.get('grade', 'F')}`; "
                f"claim-ready `{str(icse_quality.get('ready_for_icse_claim', False)).lower()}`; "
                f"priorities `{icse_priorities}`"
            ),
            f"- Validation runs: `{summary['validation_runs']}`; validation cases: `{summary['validation_cases']}`",
            (
                f"- Ablation runs: `{summary['ablation_runs']}`; ablation cases: `{summary['ablation_cases']}`; "
                f"variants: `{', '.join(summary.get('ablation_variants', [])) or 'none'}`"
            ),
            (
                "- Adaptive component ablations: "
                f"`{summary.get('adaptive_component_ablation', {}).get('run_count', 0)}` runs; "
                f"reference `{summary.get('adaptive_component_ablation', {}).get('reference_run_count', 0)}`; "
                f"contrast `{summary.get('adaptive_component_ablation', {}).get('contrast_run_count', 0)}`; "
                "disabled: "
                f"`{', '.join(summary.get('adaptive_component_ablation', {}).get('disabled_components', [])) or 'none'}`; "
                "required missing: "
                f"`{', '.join(_gate_by_name(audit, 'adaptive_component_ablation').get('missing_required_components', [])) or 'none'}`"
            ),
            (
                "- Transferability scope: "
                f"`{', '.join(summary.get('transferability_scope', {}).get('families', [])) or 'none'}`; "
                "non-primary: "
                f"`{', '.join(summary.get('transferability_scope', {}).get('non_primary_families', [])) or 'none'}`"
            ),
            (
                "- Runtime efficiency: "
                f"min throughput `{summary.get('runtime_efficiency', {}).get('min_throughput_cases_s', 0.0):.6g}` cases/s; "
                f"avg throughput `{summary.get('runtime_efficiency', {}).get('avg_throughput_cases_s', 0.0):.6g}` cases/s; "
                "max scheduler feedback share "
                f"`{summary.get('runtime_efficiency', {}).get('max_scheduler_feedback_share', 0.0):.6g}`; "
                "scheduler-share runs "
                f"`{summary.get('runtime_efficiency', {}).get('scheduler_feedback_share_run_count', 0)}`; "
                "skipped short runs "
                f"`{summary.get('runtime_efficiency', {}).get('scheduler_feedback_share_skipped_run_count', 0)}`"
            ),
            (
                "- Discovery responsiveness: "
                f"observed `{summary.get('discovery_responsiveness', {}).get('observed_run_count', 0)}` live runs; "
                f"best elapsed `{summary.get('discovery_responsiveness', {}).get('best_first_candidate_elapsed_s')}` s; "
                f"avg AUC `{summary.get('discovery_responsiveness', {}).get('avg_candidate_bug_discovery_auc', 0.0):.6g}`"
            ),
            (
                "- Closed-loop state persistence: "
                f"`{summary.get('closed_loop_state_persistence', {}).get('persisted_run_count', 0)}` / "
                f"`{summary.get('closed_loop_state_persistence', {}).get('required_run_count', 0)}` runs; "
                f"bytes `{summary.get('closed_loop_state_persistence', {}).get('total_state_bytes', 0)}`"
            ),
            (
                "- Adaptive learning health: "
                f"runs `{summary.get('closed_loop_state_persistence', {}).get('adaptive_learning_health', {}).get('run_count', 0)}`; "
                f"pulls `{summary.get('closed_loop_state_persistence', {}).get('adaptive_learning_health', {}).get('total_pulls', 0)}`; "
                "reward-model updates "
                f"`{summary.get('closed_loop_state_persistence', {}).get('adaptive_learning_health', {}).get('reward_model_update_count', 0)}`; "
                "max health penalty "
                f"`{summary.get('closed_loop_state_persistence', {}).get('adaptive_learning_health', {}).get('max_health_penalty', 0.0):.6g}`; "
                "avg uncertainty "
                f"`{summary.get('closed_loop_state_persistence', {}).get('adaptive_learning_health', {}).get('avg_uncertainty', 0.0):.6g}`; "
                "exploration records "
                f"`{summary.get('closed_loop_state_persistence', {}).get('adaptive_learning_health', {}).get('exploration_records', 0)}`"
            ),
            (
                "- Adaptive live component evidence: "
                f"declared runs `{summary.get('adaptive_live_component_evidence', {}).get('declared_run_count', 0)}`; "
                f"proven `{', '.join(summary.get('adaptive_live_component_evidence', {}).get('proven_components', [])) or 'none'}`; "
                "selection count "
                f"`{summary.get('adaptive_live_component_evidence', {}).get('adaptive_selection_total_count', 0)}`; "
                "selection scopes "
                f"`{', '.join(summary.get('adaptive_live_component_evidence', {}).get('adaptive_selection_scopes', [])) or 'none'}`; "
                f"missing `{', '.join(_gate_by_name(audit, 'adaptive_live_component_evidence').get('missing', [])) or 'none'}`"
            ),
            (
                "- Cross-version ledger: "
                f"`{summary.get('cross_version_ledger', {}).get('ledger_count', 0)}` valid ledgers; "
                f"max versions `{summary.get('cross_version_ledger', {}).get('max_version_count', 0)}`; "
                f"max families `{summary.get('cross_version_ledger', {}).get('max_family_count', 0)}`; "
                f"health observations `{summary.get('cross_version_ledger', {}).get('health_observation_count', 0)}`; "
                "health feedback reports "
                f"`{summary.get('cross_version_ledger', {}).get('health_feedback_report_count', 0)}`; "
                "invalid/fallback/false-positive "
                f"`{summary.get('cross_version_ledger', {}).get('invalid_case_count', 0)}`/"
                f"`{summary.get('cross_version_ledger', {}).get('fallback_case_count', 0)}`/"
                f"`{summary.get('cross_version_ledger', {}).get('false_positive_count', 0)}`; "
                "min throughput "
                f"`{summary.get('cross_version_ledger', {}).get('min_throughput_cases_s', 0.0):.6g}`; "
                "invalid ledgers "
                f"`{len(summary.get('cross_version_ledger', {}).get('invalid_ledger_files', []))}`; "
                f"transitions `{summary.get('cross_version_ledger', {}).get('transition_counts', {})}`"
            ),
            (
                "- Continual-learning absorption: "
                f"declared runs `{summary.get('continual_learning_absorption', {}).get('declared_run_count', 0)}`; "
                f"absorbed runs `{summary.get('continual_learning_absorption', {}).get('absorbed_run_count', 0)}`; "
                f"loaded sources `{summary.get('continual_learning_absorption', {}).get('loaded_source_count', 0)}`; "
                "max imported families "
                f"`{summary.get('continual_learning_absorption', {}).get('max_imported_family_count', 0)}`"
            ),
            (
            f"- Comparison runs: `{summary['comparison_runs']}`; comparison cases: `{summary['comparison_cases']}`; "
            f"variants: `{', '.join(summary.get('comparison_variants', [])) or 'none'}`"
        ),
        f"- Run logs scanned: `{summary['run_logs_scanned']}`; skipped: `{summary['run_logs_scan_skipped']}`",
        (
            f"- Paper run journal files: `{len(audit.get('paper_run_journal_files', []))}`; "
            f"entries: `{summary.get('paper_run_journal_entries', 0)}`; "
            f"covered runs: `{summary.get('paper_run_journal_covered_runs', 0)}` / "
            f"`{summary.get('paper_run_journal_required_runs', 0)}`"
        ),
            f"- Live runs: `{summary['live_runs']}`; live cases: `{summary['total_live_cases']}`; live elapsed_s: `{summary['total_live_elapsed_s']:.3f}`",
            (
                "- Final matrix coverage: "
                f"`{', '.join(summary.get('final_matrix_coverage', {}).get('observed_matrix_ids', [])) or 'none'}`; "
                "missing: "
                f"`{', '.join(summary.get('final_matrix_coverage', {}).get('missing_matrix_ids', [])) or 'none'}`"
            ),
            f"- Explicit live runs: `{summary['explicit_live_runs']}`; replay-policy rejected live runs: `{summary['replay_policy_rejected_live_runs']}`",
            f"- Ignored evidence runs: `{summary['ignored_evidence_runs']}`",
            f"- Live suites: `{', '.join(summary['live_suites']) or 'none'}`",
            f"- Live families: `{', '.join(summary['live_families']) or 'none'}`",
            f"- Structured semantic focus families: `{', '.join(summary.get('semantic_focus_families', [])) or 'none'}`",
            f"- Structured semantic focus signals: `{', '.join(summary.get('semantic_focus_signals', [])) or 'none'}`",
            f"- Rewardable latest candidate families: `{len(summary['rewardable_live_candidate_families'])}`",
            f"- External confirmed latest candidate families: `{len(summary.get('external_confirmed_live_candidate_families', {}))}`",
            f"- Known/saturated latest candidate families: `{len(summary['known_saturated_live_candidate_families'])}`",
            f"- Confirmed latest candidate families: `{len(summary['confirmed_live_candidate_families'])}`",
            f"- Historical confirmed replay ids: `{', '.join(summary['historical_confirmed_bug_ids']) or 'none'}`",
            "",
            "## Live Suites",
            "",
            "| suite | runs | cases | elapsed_s | findings | families | replay policy |",
            "|---|---:|---:|---:|---:|---|---|",
        ]
    )
    for suite, row in audit["live_suites"].items():
        lines.append(
            f"| {suite} | {row['runs']} | {row['cases']} | {row['elapsed_s']:.3f} | "
            f"{row['findings']} | {', '.join(row['families'])} | {'ok' if row['replay_policy_ok'] else 'bad'} |"
        )
    lines.append("")
    return "\n".join(lines)


_structured_identity_missing = _structured_identity_issues
_stage_profile_missing = _stage_profile_issues
