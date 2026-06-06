from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any


def cmd_discovery_run_impl(
    args: argparse.Namespace,
    *,
    ensure_dirs_func: Callable[[], None],
    resolve_run_backends_func: Callable[[argparse.Namespace], list[str]],
    discovery_run_config_from_args_func: Callable[[argparse.Namespace], Any],
    parse_guidance_targets_func: Callable[[str], list[str]],
    run_probe_audit_func: Callable[..., Any],
    write_probe_issue_drafts_func: Callable[..., list[Path]],
    project_relative_path_func: Callable[[str | Path], str],
    run_fuzz_func: Callable[..., Path],
    parse_duration_func: Callable[[Any], float | None],
    write_report_func: Callable[..., tuple[Path, Path]],
    summarize_run_classification_func: Callable[..., dict[str, Any]],
    write_discovery_run_fresh_candidate_evidence_func: Callable[..., dict[str, Any]],
    run_candidate_pipeline_for_evidence_func: Callable[..., dict[str, Any]],
    dump_json_func: Callable[..., None],
    utc_now_func: Callable[[], str],
) -> int:
    ensure_dirs_func()
    backends = resolve_run_backends_func(args)
    config = discovery_run_config_from_args_func(args)
    probe_ids = parse_guidance_targets_func(getattr(args, "probes", "") or "")
    audit_summary: dict[str, Any] = {"skipped": True}

    if not getattr(args, "skip_bug_audit", False):
        audit_run = run_probe_audit_func(probe_ids=probe_ids or None)
        issue_paths: list[Path] = []
        if getattr(args, "write_issues", True):
            issue_paths = write_probe_issue_drafts_func(
                audit_run,
                issue_dir=Path(getattr(args, "issue_dir", "new_issue/generated")),
                overwrite=bool(getattr(args, "overwrite_issues", True)),
            )
        audit_summary = {
            "skipped": False,
            "generated_at": audit_run.generated_at,
            "output_json": project_relative_path_func(audit_run.output_json),
            "output_markdown": project_relative_path_func(audit_run.output_markdown),
            "candidate_bug_families": list(audit_run.candidate_bug_families),
            "issue_files": [project_relative_path_func(path) for path in issue_paths],
            "environment": dict(audit_run.environment),
            "results": list(audit_run.results),
        }

    run_file = run_fuzz_func(
        cases=args.cases,
        seed=args.seed,
        backends=backends,
        config=config,
        duration_s=parse_duration_func(args.duration),
    )
    if getattr(args, "skip_run_report", False):
        md_path = csv_path = None
    else:
        md_path, csv_path = write_report_func(run_file)
    classification = summarize_run_classification_func(
        run_file,
        limit=max(0, int(getattr(args, "classify_limit", 3))),
        refresh=bool(getattr(args, "refresh_classification", False)),
    )
    manifest_path = Path(getattr(args, "output_manifest", "") or "new_issue/generated/discovery-run-manifest.json")
    fresh_evidence_path = manifest_path.with_name(f"{manifest_path.stem}-fresh-candidates.json")
    fresh_evidence = write_discovery_run_fresh_candidate_evidence_func(
        run_file,
        classification=classification,
        output_path=fresh_evidence_path,
        refresh=bool(getattr(args, "refresh_classification", False)),
    )
    candidate_pipeline = {}
    if fresh_evidence.get("candidate_row_count", 0):
        candidate_pipeline = run_candidate_pipeline_for_evidence_func(
            args,
            evidence_path=fresh_evidence_path,
            manifest_path=manifest_path,
        )
    manifest = {
        "schema_version": "discovery-run-v1",
        "generated_at": utc_now_func(),
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
            "run_file": project_relative_path_func(run_file),
            "report": project_relative_path_func(md_path) if md_path is not None else "",
            "csv": project_relative_path_func(csv_path) if csv_path is not None else "",
            "fresh_candidate_evidence": project_relative_path_func(fresh_evidence_path),
            "fresh_candidate_evidence_rows": fresh_evidence["candidate_row_count"],
            "candidate_pipeline": candidate_pipeline,
        },
        "classification": classification,
        "candidate_pipeline": candidate_pipeline,
    }
    dump_json_func(manifest, manifest_path)

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


def cmd_discovery_campaign_impl(
    args: argparse.Namespace,
    *,
    discovery_campaign_lane_catalog_func: Callable[[], dict[str, dict[str, Any]]],
    ensure_dirs_func: Callable[[], None],
    discovery_campaign_lanes_from_args_func: Callable[[str], list[dict[str, Any]]],
    parse_seeds_func: Callable[[str], list[int]],
    parse_duration_func: Callable[[Any], float | None],
    parse_guidance_targets_func: Callable[[str], list[str]],
    run_probe_audit_func: Callable[..., Any],
    write_probe_issue_drafts_func: Callable[..., list[Path]],
    project_relative_path_func: Callable[[str | Path], str],
    utc_now_func: Callable[[], str],
    discovery_campaign_score_weights_from_args_func: Callable[[argparse.Namespace], dict[str, float]],
    default_campaign_history_window: int,
    discovery_campaign_scheduler_snapshot_func: Callable[..., dict[str, Any]],
    build_discovery_campaign_manifest_func: Callable[..., dict[str, Any]],
    dump_json_func: Callable[..., None],
    next_discovery_campaign_lane_func: Callable[..., dict[str, Any] | None],
    resolve_target_backends_func: Callable[..., list[str]],
    discovery_campaign_config_from_args_func: Callable[..., Any],
    run_fuzz_func: Callable[..., Path],
    write_report_func: Callable[..., tuple[Path, Path]],
    summarize_run_classification_func: Callable[..., dict[str, Any]],
    write_discovery_run_fresh_candidate_evidence_func: Callable[..., dict[str, Any]],
    run_candidate_pipeline_for_evidence_func: Callable[..., dict[str, Any]],
    summarize_run_health_func: Callable[..., dict[str, Any]],
) -> int:
    if getattr(args, "list_lanes", False):
        catalog = discovery_campaign_lane_catalog_func()
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

    ensure_dirs_func()
    selected_lanes = discovery_campaign_lanes_from_args_func(getattr(args, "lanes", ""))
    seeds = parse_seeds_func(str(getattr(args, "seeds", "1")))
    duration_s = parse_duration_func(getattr(args, "duration", None))
    probe_ids = parse_guidance_targets_func(getattr(args, "probes", "") or "")
    audit_summary: dict[str, Any] = {"skipped": True}
    if not getattr(args, "skip_bug_audit", False):
        audit_run = run_probe_audit_func(probe_ids=probe_ids or None)
        issue_paths: list[Path] = []
        if getattr(args, "write_issues", True):
            issue_paths = write_probe_issue_drafts_func(
                audit_run,
                issue_dir=Path(getattr(args, "issue_dir", "new_issue/generated")),
                overwrite=bool(getattr(args, "overwrite_issues", True)),
            )
        audit_summary = {
            "skipped": False,
            "generated_at": audit_run.generated_at,
            "output_json": project_relative_path_func(audit_run.output_json),
            "output_markdown": project_relative_path_func(audit_run.output_markdown),
            "candidate_bug_families": list(audit_run.candidate_bug_families),
            "issue_files": [project_relative_path_func(path) for path in issue_paths],
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
    started_at = utc_now_func()
    total_run_count = len(selected_lanes) * len(seeds)
    generated_issue_dir = manifest_path.parent
    score_weights = discovery_campaign_score_weights_from_args_func(args)
    history_limit = max(0, int(getattr(args, "lane_history_window", default_campaign_history_window)))
    lane_by_id = {lane["id"]: lane for lane in selected_lanes}
    pending_by_lane = {lane["id"]: list(seeds) for lane in selected_lanes}

    def write_manifest_snapshot(*, status: str, current_run: dict[str, Any] | None = None) -> None:
        scheduler = discovery_campaign_scheduler_snapshot_func(
            selected_lanes=selected_lanes,
            runs=runs,
            pending_by_lane=pending_by_lane,
            generated_issue_dir=generated_issue_dir,
            manifest_path=manifest_path,
            history_limit=history_limit,
            score_weights=score_weights,
        )
        manifest = build_discovery_campaign_manifest_func(
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
        dump_json_func(manifest, manifest_path)

    write_manifest_snapshot(status="running")

    while True:
        scheduler = discovery_campaign_scheduler_snapshot_func(
            selected_lanes=selected_lanes,
            runs=runs,
            pending_by_lane=pending_by_lane,
            generated_issue_dir=generated_issue_dir,
            manifest_path=manifest_path,
            history_limit=history_limit,
            score_weights=score_weights,
        )
        next_lane = next_discovery_campaign_lane_func(scheduler, pending_by_lane)
        if next_lane is None:
            break
        lane_id = str(next_lane["lane_id"])
        lane = lane_by_id[lane_id]
        seed = pending_by_lane[lane_id].pop(0)
        target_suite = lane["target_suite"]
        preset = lane["preset"]
        backends = resolve_target_backends_func(target_suite=target_suite)
        config = discovery_campaign_config_from_args_func(
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
            "started_at": utc_now_func(),
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
        run_file = run_fuzz_func(
            cases=int(getattr(args, "cases", 100)),
            seed=int(seed),
            backends=backends,
            config=config,
            duration_s=duration_s,
        )
        if getattr(args, "skip_run_report", False):
            md_path = csv_path = None
        else:
            md_path, csv_path = write_report_func(run_file)
        classification = summarize_run_classification_func(
            run_file,
            limit=max(0, int(getattr(args, "classify_limit", 3))),
            refresh=bool(getattr(args, "refresh_classification", False)),
        )
        evidence_path = manifest_path.with_name(
            f"{manifest_path.stem}-{lane['id']}-seed{seed}-fresh-candidates.json"
        )
        fresh_evidence = write_discovery_run_fresh_candidate_evidence_func(
            run_file,
            classification=classification,
            output_path=evidence_path,
            refresh=bool(getattr(args, "refresh_classification", False)),
        )
        candidate_pipeline = {}
        if fresh_evidence.get("candidate_row_count", 0):
            candidate_pipeline = run_candidate_pipeline_for_evidence_func(
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
        health = summarize_run_health_func(run_file, limit=max(0, int(getattr(args, "classify_limit", 3))))
        run_record.update(
            {
                "run_file": project_relative_path_func(run_file),
                "report": project_relative_path_func(md_path) if md_path is not None else "",
                "csv": project_relative_path_func(csv_path) if csv_path is not None else "",
                "fresh_candidate_evidence": project_relative_path_func(evidence_path),
                "fresh_candidate_evidence_rows": fresh_evidence["candidate_row_count"],
                "candidate_pipeline": candidate_pipeline,
                "classification": classification,
                "health": health,
                "status": "completed",
                "completed_at": utc_now_func(),
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


def cmd_candidate_pipeline_impl(
    args: argparse.Namespace,
    *,
    parse_guidance_targets_func: Callable[[str], list[str]],
    build_candidate_pipeline_func: Callable[..., dict[str, Any]],
    default_candidate_pipeline_dir: Path,
) -> int:
    manifest_file = Path(args.manifest) if getattr(args, "manifest", None) else None
    evidence_files = [Path(item) for item in parse_guidance_targets_func(getattr(args, "evidence_files", "") or "")]
    pipeline = build_candidate_pipeline_func(
        evidence_files=evidence_files or None,
        manifest_file=manifest_file,
        output_dir=Path(getattr(args, "output_dir", default_candidate_pipeline_dir)),
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


def cmd_discovery_campaign_aggregate_impl(
    args: argparse.Namespace,
    *,
    summarize_discovery_campaign_aggregate_func: Callable[..., dict[str, Any]],
    dump_json_func: Callable[..., None],
) -> int:
    manifest_files = _discovery_campaign_aggregate_manifest_files(str(getattr(args, "manifests", "") or ""))
    aggregate = summarize_discovery_campaign_aggregate_func(
        manifest_files,
        limit=max(0, int(getattr(args, "limit", 10))),
    )
    output = str(getattr(args, "output", "") or "").strip()
    if output:
        dump_json_func(aggregate, Path(output))
    if getattr(args, "json", False):
        print(json.dumps(aggregate, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        progress = aggregate.get("progress", {}) if isinstance(aggregate.get("progress"), dict) else {}
        pipeline = aggregate.get("candidate_pipeline", {}) if isinstance(aggregate.get("candidate_pipeline"), dict) else {}
        quality = (
            aggregate.get("icse_experiment_quality", {})
            if isinstance(aggregate.get("icse_experiment_quality"), dict)
            else {}
        )
        print(f"discovery campaign manifests: {aggregate.get('manifest_count', 0)}")
        print(
            "runs: "
            f"completed={progress.get('completed_run_count', 0)} "
            f"running={progress.get('running_run_count', 0)} "
            f"remaining={progress.get('remaining_run_count', 0)}"
        )
        print(f"fresh families: {aggregate.get('fresh_candidate_bug_family_count', 0)}")
        print(f"fresh evidence rows: {aggregate.get('fresh_candidate_evidence_rows', 0)}")
        print(
            "candidate pipeline: "
            f"candidates={pipeline.get('candidate_count', 0)} "
            f"reproduced={pipeline.get('reproduced_count', 0)} "
            f"reduced={pipeline.get('reduced_count', 0)} "
            f"issue_drafts={pipeline.get('issue_draft_count', 0)}"
        )
        print(
            "icse quality: "
            f"score={float(quality.get('overall_score', 0.0) or 0.0):.1f} "
            f"grade={quality.get('grade', '')} "
            f"claim_ready={str(quality.get('claim_ready', False)).lower()}"
        )
        fresh = aggregate.get("fresh_candidate_bug_families", {})
        print("top fresh families:")
        if fresh:
            for family, count in fresh.items():
                print(f"- {family}: {count}")
        else:
            print("- none")
    return 0


def cmd_discovery_campaign_status_impl(
    args: argparse.Namespace,
    *,
    summarize_discovery_campaign_status_func: Callable[..., dict[str, Any]],
    runs_dir: Path,
) -> int:
    manifest_arg = getattr(args, "manifest", "") or "new_issue/generated/discovery-campaign-manifest.json"
    summary = summarize_discovery_campaign_status_func(
        Path(manifest_arg),
        limit=max(0, int(args.limit)),
        runs_dir=runs_dir,
    )
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


def _discovery_campaign_aggregate_manifest_files(value: str) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for raw_token in value.split(","):
        token = raw_token.strip()
        if not token:
            continue
        matches = [Path(match) for match in glob.glob(token)] if any(ch in token for ch in "*?[]") else [Path(token)]
        for path in sorted(matches, key=lambda item: str(item)):
            key = path.resolve() if path.exists() else path.absolute()
            if key in seen:
                continue
            seen.add(key)
            paths.append(path)
    return paths
