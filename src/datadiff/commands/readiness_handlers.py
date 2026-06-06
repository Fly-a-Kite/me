from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any


def cmd_final_readiness_impl(
    args: argparse.Namespace,
    *,
    parse_presets_func: Callable[[str], list[str]],
    parse_adaptive_component_tuple_func: Callable[[Any], tuple[str, ...]],
    manifest_index_files_func: Callable[[list[str] | tuple[str, ...]], tuple[list[Path], list[Path], list[Path]]],
    dedupe_paths_func: Callable[[list[Path]], list[Path]],
    readiness_policy_factory: Callable[..., Any],
    readiness_thresholds_factory: Callable[..., Any],
    default_readiness_policy: Any,
    default_manifest_limit: int,
    analyze_final_readiness_func: Callable[..., tuple[Path, Path]],
    load_json_func: Callable[[Path], Any],
) -> int:
    manifests = [Path(path) for path in getattr(args, "manifest", [])]
    extra_manifests = [Path(path) for path in getattr(args, "extra_manifest", [])]
    index_manifests, index_extra_manifests, index_paper_run_journals = manifest_index_files_func(
        getattr(args, "manifest_index", [])
    )
    manifests = dedupe_paths_func([*manifests, *index_manifests])
    extra_manifests = dedupe_paths_func([*extra_manifests, *index_extra_manifests])
    paper_run_journal_files = index_paper_run_journals or None
    latest_confirmation_files = [Path(path) for path in getattr(args, "latest_confirmation_file", [])]
    required_live_suites = (
        tuple(parse_presets_func(args.required_live_suites))
        if args.required_live_suites
        else default_readiness_policy.required_live_suites
    )
    required_live_families = (
        tuple(parse_presets_func(args.required_live_families))
        if args.required_live_families
        else default_readiness_policy.required_live_families
    )
    policy = readiness_policy_factory(
        required_live_suites=required_live_suites,
        required_live_families=required_live_families,
        confirmed_live_paper_statuses=default_readiness_policy.confirmed_live_paper_statuses,
    )
    thresholds = readiness_thresholds_factory(
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
        required_adaptive_component_ablations=parse_adaptive_component_tuple_func(
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
        require_target_version_audit=not bool(getattr(args, "no_require_target_version_audit", False)),
    )
    manifest_limit = (
        None
        if manifests or getattr(args, "all_manifests", False)
        else max(1, int(getattr(args, "latest_manifests", default_manifest_limit)))
    )
    scan_run_logs = (
        bool(manifests) or bool(getattr(args, "full_run_log_scan", False))
    ) and not bool(getattr(args, "summary_only", False))
    md_path, json_path = analyze_final_readiness_func(
        manifests or None,
        extra_manifest_files=extra_manifests or None,
        latest_confirmation_files=latest_confirmation_files or None,
        paper_run_journal_files=paper_run_journal_files,
        manifest_limit=manifest_limit,
        scan_run_logs=scan_run_logs,
        thresholds=thresholds,
        policy=policy,
    )
    audit = load_json_func(json_path)
    if getattr(args, "json", False):
        print(json.dumps(audit, indent=2, sort_keys=True))
    else:
        print(f"final readiness markdown: {md_path}")
        print(f"final readiness json:     {json_path}")
    return 2 if getattr(args, "fail_on_missing", False) and not audit.get("ready", False) else 0


def cmd_review_readiness_impl(
    args: argparse.Namespace,
    *,
    review_thresholds_factory: Callable[..., Any],
    build_review_readiness_func: Callable[..., dict[str, Any]],
    write_review_readiness_outputs_func: Callable[..., tuple[Path, Path]],
    project_relative_path_func: Callable[[str | Path], str],
) -> int:
    latest_confirmation_files = [
        Path(path) for path in getattr(args, "latest_confirmation_file", [])
    ]
    thresholds = review_thresholds_factory(
        target_confirmed_bug_families=max(0, int(getattr(args, "target_confirmed", 20))),
        min_audit_candidate_families=max(0, int(getattr(args, "min_audit_candidates", 1))),
        min_discovery_workflow_manifests=max(0, int(getattr(args, "min_discovery_workflows", 1))),
        min_generated_issue_drafts=max(0, int(getattr(args, "min_generated_issue_drafts", 1))),
        min_issue_bundle_families=max(0, int(getattr(args, "min_issue_bundle_families", 1))),
        min_pending_issue_drafts=max(0, int(getattr(args, "min_pending_issue_drafts", 1))),
        min_old_known_upstream_issues=max(0, int(getattr(args, "min_old_known_issues", 1))),
    )
    audit = build_review_readiness_func(
        latest_confirmation_files=latest_confirmation_files or None,
        thresholds=thresholds,
    )
    if getattr(args, "write_report", False):
        json_path, md_path = write_review_readiness_outputs_func(
            audit,
            output_dir=Path(getattr(args, "output_dir", "reports")),
        )
        audit["output_json"] = project_relative_path_func(json_path)
        audit["output_markdown"] = project_relative_path_func(md_path)
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
