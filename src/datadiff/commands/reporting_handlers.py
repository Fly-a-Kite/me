from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any


def cmd_report_impl(
    args: argparse.Namespace,
    *,
    latest_run_log_path_func: Callable[[], Path],
    write_report_func: Callable[..., tuple[Path, Path]],
) -> int:
    run_file = Path(args.run_file) if args.run_file else latest_run_log_path_func()
    md_path, csv_path = write_report_func(run_file, csv_limit=args.csv_limit)
    print(f"markdown report: {md_path}")
    print(f"csv findings:    {csv_path}")
    return 0


def cmd_bug_audit_impl(
    args: argparse.Namespace,
    *,
    parse_guidance_targets_func: Callable[[str], list[str]],
    run_probe_audit_func: Callable[..., Any],
    write_probe_issue_drafts_func: Callable[..., list[Path]],
) -> int:
    probe_ids = parse_guidance_targets_func(getattr(args, "probes", "") or "")
    run = run_probe_audit_func(probe_ids=probe_ids or None)
    if getattr(args, "write_issues", False):
        issue_paths = write_probe_issue_drafts_func(
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


def cmd_bug_status_impl(
    args: argparse.Namespace,
    *,
    parse_guidance_targets_func: Callable[[str], list[str]],
    build_bug_status_func: Callable[..., dict[str, Any]],
    write_bug_status_outputs_func: Callable[..., tuple[Path, Path]],
    project_relative_path_func: Callable[[str | Path], str],
) -> int:
    latest_confirmation_files = [
        Path(item) for item in parse_guidance_targets_func(getattr(args, "latest_confirmations", "") or "")
    ]
    status = build_bug_status_func(
        latest_confirmation_files=latest_confirmation_files or None,
        new_issue_dir=Path(getattr(args, "new_issue_dir", "new_issue")),
        old_issue_dir=Path(getattr(args, "old_issue_dir", "old_issue")),
        generated_issue_dir=Path(getattr(args, "generated_issue_dir", "new_issue/generated")),
    )
    if getattr(args, "write_report", False):
        json_path, md_path = write_bug_status_outputs_func(
            status,
            output_dir=Path(getattr(args, "output_dir", "reports")),
        )
        status["output_json"] = project_relative_path_func(json_path)
        status["output_markdown"] = project_relative_path_func(md_path)
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


def cmd_issue_readiness_impl(
    args: argparse.Namespace,
    *,
    parse_guidance_targets_func: Callable[[str], list[str]],
    build_issue_readiness_func: Callable[..., dict[str, Any]],
    write_issue_readiness_outputs_func: Callable[..., tuple[Path, Path]],
    project_relative_path_func: Callable[[str | Path], str],
) -> int:
    latest_confirmation_files = [
        Path(item) for item in parse_guidance_targets_func(getattr(args, "latest_confirmations", "") or "")
    ]
    audit = build_issue_readiness_func(
        latest_confirmation_files=latest_confirmation_files or None,
        new_issue_dir=Path(getattr(args, "new_issue_dir", "new_issue")),
        old_issue_dir=Path(getattr(args, "old_issue_dir", "old_issue")),
        generated_issue_dir=Path(getattr(args, "generated_issue_dir", "new_issue/generated")),
        include_generated=bool(getattr(args, "include_generated", False)),
    )
    if getattr(args, "write_report", False):
        json_path, md_path = write_issue_readiness_outputs_func(
            audit,
            output_dir=Path(getattr(args, "output_dir", "reports")),
        )
        audit["output_json"] = project_relative_path_func(json_path)
        audit["output_markdown"] = project_relative_path_func(md_path)
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


def cmd_issue_bundle_impl(
    args: argparse.Namespace,
    *,
    parse_guidance_targets_func: Callable[[str], list[str]],
    build_issue_bundle_func: Callable[..., dict[str, Any]],
    default_issue_bundle_statuses: Sequence[str],
) -> int:
    latest_confirmation_files = [
        Path(item) for item in parse_guidance_targets_func(getattr(args, "latest_confirmations", "") or "")
    ]
    statuses = parse_guidance_targets_func(getattr(args, "statuses", "") or "")
    manifest = build_issue_bundle_func(
        latest_confirmation_files=latest_confirmation_files or None,
        new_issue_dir=Path(getattr(args, "new_issue_dir", "new_issue")),
        old_issue_dir=Path(getattr(args, "old_issue_dir", "old_issue")),
        generated_issue_dir=Path(getattr(args, "generated_issue_dir", "new_issue/generated")),
        output_dir=Path(getattr(args, "output_dir", "new_issue/generated/issue-bundles")),
        statuses=statuses or list(default_issue_bundle_statuses),
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


def cmd_methodology_report_impl(
    args: argparse.Namespace,
    *,
    write_methodology_report_func: Callable[..., tuple[Path, Path]],
    load_json_func: Callable[[Path], Any],
) -> int:
    manifest_file = Path(args.manifest) if args.manifest else None
    md_path, json_path = write_methodology_report_func(
        manifest_file,
        refresh=bool(getattr(args, "refresh", False)),
        scan_run_logs=not bool(getattr(args, "summary_only", False)),
    )
    if getattr(args, "json", False):
        print(json.dumps(load_json_func(json_path), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    print(f"methodology report markdown: {md_path}")
    print(f"methodology report json:     {json_path}")
    return 0


def cmd_show_bugs_impl(
    args: argparse.Namespace,
    *,
    latest_run_log_path_func: Callable[[], Path],
    read_jsonl_func: Callable[[Path], list[dict[str, Any]]],
) -> int:
    run_file = Path(args.run_file) if args.run_file else latest_run_log_path_func()
    rows = read_jsonl_func(run_file)
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


def cmd_classify_run_impl(
    args: argparse.Namespace,
    *,
    latest_run_log_path_func: Callable[[], Path],
    summarize_run_classification_func: Callable[..., dict[str, Any]],
) -> int:
    run_file = Path(args.run_file) if args.run_file else latest_run_log_path_func()
    summary = summarize_run_classification_func(
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


def cmd_run_health_impl(
    args: argparse.Namespace,
    *,
    latest_run_log_path_func: Callable[[], Path],
    summarize_run_health_func: Callable[..., dict[str, Any]],
) -> int:
    run_file = Path(args.run_file) if args.run_file else latest_run_log_path_func()
    summary = summarize_run_health_func(run_file, limit=max(0, int(args.limit)))
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
