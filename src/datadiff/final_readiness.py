from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from datadiff.config import ExperimentConfig
from datadiff.experiment_catalog import (
    FINAL_LIVE_DISCOVERY_MATRIX,
    FINAL_PROTOCOL_TRACKS,
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
from datadiff.historical import list_historical_bugs
from datadiff.reward import (
    candidate_issue_family_key,
    is_issue_replay_finding,
    is_known_saturated_candidate_issue_finding,
    is_rewardable_candidate_issue_finding,
)
from datadiff.run_provenance import current_workspace_git_commit
from datadiff.runner import _configured_guidance_targets
from datadiff.targets import TARGETS, TARGET_SUITES, target_context
from datadiff.util import REPORTS_DIR, RUNS_DIR, dump_json, ensure_dirs, load_json, read_jsonl, run_meta_path, utc_now

DEFAULT_LATEST_CONFIRMATIONS_FILE = Path("experiments/latest_confirmations.json")
DEFAULT_FINAL_READINESS_MANIFEST_LIMIT = 25
FINAL_READINESS_SCHEMA_VERSION = "final-readiness-v1"
DEFAULT_PAPER_RUN_JOURNAL_NAME = "paper-run-journal.jsonl"
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
    manifest_files = _resolve_manifest_files(manifest_files, manifest_limit=manifest_limit)
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
    ignored_runs = [run for run in runs if run["evidence_mode"] not in EVIDENCE_MODES]
    audited_runs = [run for run in runs if run["evidence_mode"] in EVIDENCE_MODES]
    paper_run_journal_missing_runs = _paper_run_journal_issues(audited_runs)
    paper_run_journal_covered_runs = sum(1 for run in audited_runs if bool(run.get("paper_run_journal_present")))

    live_by_suite = _live_suite_summary(live_runs)
    live_families = sorted({family for run in live_runs for family in run["target_families"]})
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
        scan_run_logs=scan_run_logs,
        rewardable_live_families=rewardable_live_families,
        confirmed_live_families=confirmed_live_families,
        paper_run_journal_files=paper_run_journal_files,
        paper_run_journal_covered_runs=paper_run_journal_covered_runs,
        paper_run_journal_issues=paper_run_journal_missing_runs,
        policy=policy,
        workspace_git_commit=workspace_git_commit,
    )
    return {
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
            "comparison_runs": len(comparison_runs),
            "comparison_cases": sum(int(run["cases"]) for run in comparison_runs),
            "comparison_suites": sorted({run["target_suite"] for run in comparison_runs}),
            "comparison_presets": sorted({run["preset"] for run in comparison_runs}),
            "comparison_variants": sorted(
                {run["variant_label"] for run in comparison_runs if run.get("variant_label")}
            ),
            "comparison_matrix_ids": _sorted_matrix_ids(comparison_runs),
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
        "runs": validation_runs + live_runs + historical_runs + seeded_runs + ablation_runs + comparison_runs,
        "latest_confirmations": latest_confirmations,
        "ignored_runs": ignored_runs,
        "replay_policy_rejected_live_runs": replay_policy_rejected_live_runs,
    }


def _resolve_manifest_files(
    manifest_files: list[Path] | None,
    *,
    manifest_limit: int | None = DEFAULT_FINAL_READINESS_MANIFEST_LIMIT,
) -> list[Path]:
    if manifest_files:
        return [Path(path) for path in manifest_files]
    manifests = sorted(RUNS_DIR.glob("experiment-*.json"))
    if manifest_limit is None:
        return manifests
    limit = max(1, int(manifest_limit))
    return sorted(manifests, key=lambda path: (path.stat().st_mtime_ns, path.name))[-limit:]


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
        for row in read_jsonl(path):
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
                "known_bug_id": str(run.get("known_bug_id", manifest.get("known_bug_id", "")) or ""),
                "target_version": str(run.get("target_version", manifest.get("target_version", "")) or ""),
                "cases": int(meta.get("executed_cases", len(rows)) or 0),
                "elapsed_s": float(meta.get("elapsed_s", 0.0) or 0.0),
                "throughput_cases_s": float(meta.get("throughput_cases_s", 0.0) or 0.0),
                "duration_s": meta.get("duration_s"),
                "findings": findings_count,
                "config": dict(config),
                "run_log_scanned": run_log_scanned,
                "run_log_scan_skipped": bool(not run_log_scanned and run_file.is_file()),
                "stage_profile_present": _stage_profile_present(meta),
                "target_families": target_families,
                "enable_replay_bug": run_enable_replay_bug,
                "manifest_enable_replay_bug": manifest_enable_replay_bug,
                "replay_filter_enabled": replay_filter.get("enabled", ""),
                "replay_filter_filtered_candidates": int(replay_filter.get("filtered_candidates", 0) or 0),
                "known_saturated_bug_families": known_bug_families,
                "run_provenance": run_provenance,
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
    return is_ablation_experiment_row(run)


def _is_contrast_scope_run(run: dict[str, Any]) -> bool:
    return is_comparison_experiment_row(run) and str(run.get("evidence_mode") or "") == "comparison"


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
    scan_run_logs: bool,
    rewardable_live_families: Counter[str],
    confirmed_live_families: Counter[str],
    paper_run_journal_files: list[Path],
    paper_run_journal_covered_runs: int,
    paper_run_journal_issues: list[str],
    policy: ReadinessPolicy,
    workspace_git_commit: str,
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
            not _stage_profile_issues([run for run in audited_runs if run["evidence_mode"] != "historical"]),
            "all non-historical runs should expose stage-level profiling and throughput in run meta",
            issues=_stage_profile_issues([run for run in audited_runs if run["evidence_mode"] != "historical"]),
            missing=_stage_profile_issues([run for run in audited_runs if run["evidence_mode"] != "historical"]),
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
            bool(ablation_runs) or not thresholds.require_ablation,
            f"required={thresholds.require_ablation} observed_runs={len(ablation_runs)}",
            suites=sorted({run["target_suite"] for run in ablation_runs}),
            presets=sorted({run["preset"] for run in ablation_runs}),
            variants=sorted({run["variant_label"] for run in ablation_runs if run.get("variant_label")}),
            matrix_ids=_sorted_matrix_ids(ablation_runs),
        ),
        _gate(
            "baseline_comparison",
            bool(comparison_runs) or not thresholds.require_comparison,
            f"required={thresholds.require_comparison} observed_runs={len(comparison_runs)}",
            suites=sorted({run["target_suite"] for run in comparison_runs}),
            presets=sorted({run["preset"] for run in comparison_runs}),
            variants=sorted({run["variant_label"] for run in comparison_runs if run.get("variant_label")}),
            matrix_ids=_sorted_matrix_ids(comparison_runs),
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
    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- Validation runs: `{summary['validation_runs']}`; validation cases: `{summary['validation_cases']}`",
            (
                f"- Ablation runs: `{summary['ablation_runs']}`; ablation cases: `{summary['ablation_cases']}`; "
                f"variants: `{', '.join(summary.get('ablation_variants', [])) or 'none'}`"
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
