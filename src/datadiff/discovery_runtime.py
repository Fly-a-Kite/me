from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from datadiff.candidate_pipeline import DEFAULT_CANDIDATE_PIPELINE_DIR, build_candidate_pipeline
from datadiff.config import ExperimentConfig, merge_discovery_biases
from datadiff.discovery_campaign_summary import DEFAULT_DISCOVERY_CAMPAIGN_SCORE_WEIGHTS
from datadiff.finding_outcomes import (
    candidate_issue_family_key,
    candidate_issue_family_keys,
    is_rewardable_candidate_issue_finding,
)
from datadiff.guidance import parse_guidance_targets
from datadiff.strategy_registry import (
    DEFAULT_DISCOVERY_LANE_IDS,
    discovery_lane_catalog,
    discovery_lane_spec,
)
from datadiff.util import dump_json, read_jsonl, utc_now


def discovery_campaign_score_weights_from_args(
    args: argparse.Namespace,
    *,
    default_weights: dict[str, float] = DEFAULT_DISCOVERY_CAMPAIGN_SCORE_WEIGHTS,
) -> dict[str, float]:
    weights = {key: max(0.0, float(value)) for key, value in default_weights.items()}
    weights["yield_rate"] = max(0.0, float(getattr(args, "lane_yield_weight", weights["yield_rate"])))
    weights["novelty_rate"] = max(0.0, float(getattr(args, "lane_novelty_weight", weights["novelty_rate"])))
    weights["false_positive_penalty"] = max(
        0.0,
        float(
            getattr(
                args,
                "lane_false_positive_penalty",
                weights["false_positive_penalty"],
            )
        ),
    )
    return weights


def next_discovery_campaign_lane(
    scheduler: dict[str, Any],
    pending_by_lane: dict[str, list[int]],
) -> dict[str, Any] | None:
    for lane in scheduler.get("lanes", []) or []:
        lane_id = str(lane.get("lane_id", ""))
        if pending_by_lane.get(lane_id):
            return lane
    return None


def aggregate_candidate_pipeline_summary(items: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate: Counter[str] = Counter()
    pipeline_count = 0
    for item in items:
        summary = item.get("summary", {}) if isinstance(item.get("summary"), dict) else {}
        if not summary:
            continue
        pipeline_count += 1
        candidate_count = int(summary.get("candidate_count", 0) or 0)
        skipped_duplicate_count = int(summary.get("skipped_duplicate_candidate_count", 0) or 0)
        processed_candidate_count = int(
            summary.get(
                "processed_candidate_count",
                max(0, candidate_count - skipped_duplicate_count),
            )
            or 0
        )
        aggregate.update(
            {
                "candidate_count": candidate_count,
                "processed_candidate_count": processed_candidate_count,
                "skipped_duplicate_candidate_count": skipped_duplicate_count,
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


def build_discovery_campaign_manifest(
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
    utc_now_func: Callable[[], str] = utc_now,
    aggregate_candidate_pipeline_summary_func: Callable[[list[dict[str, Any]]], dict[str, Any]]
    = aggregate_candidate_pipeline_summary,
) -> dict[str, Any]:
    completed_run_count = sum(1 for run in runs if run.get("status") == "completed")
    candidate_pipeline_summary = aggregate_candidate_pipeline_summary_func(
        [
            run.get("candidate_pipeline", {})
            for run in runs
            if isinstance(run.get("candidate_pipeline"), dict)
        ]
    )
    return {
        "schema_version": "discovery-campaign-v1",
        "generated_at": utc_now_func(),
        "generated_by": "datadiff discovery-campaign",
        "status": status,
        "started_at": started_at,
        "completed_at": utc_now_func() if status == "completed" else "",
        "workflow": [
            "deterministic_bug_audit",
            "guided_discovery_campaign",
            "true_bug_acquisition_scheduler",
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


def discovery_campaign_lanes_from_args(
    value: str,
    *,
    parse_guidance_targets_func: Callable[[str], list[str]] = parse_guidance_targets,
    default_discovery_lane_ids: tuple[str, ...] = DEFAULT_DISCOVERY_LANE_IDS,
    discovery_lane_spec_func: Callable[[str], Any] = discovery_lane_spec,
) -> list[dict[str, str]]:
    raw_lane_ids = parse_guidance_targets_func(value or "")
    lane_ids = raw_lane_ids or list(default_discovery_lane_ids)
    lanes = []
    seen = set()
    for lane_id in lane_ids:
        if lane_id in seen:
            continue
        seen.add(lane_id)
        lane = discovery_lane_spec_func(lane_id).to_dict()
        lanes.append(lane)
    return lanes


def discovery_campaign_lane_catalog(
    *,
    discovery_lane_catalog_func: Callable[[], dict[str, dict[str, Any]]] = discovery_lane_catalog,
) -> dict[str, dict[str, Any]]:
    return discovery_lane_catalog_func()


def discovery_campaign_config_from_args(
    args: argparse.Namespace,
    preset: str,
    *,
    preset_config_func: Callable[[str], ExperimentConfig],
    parse_guidance_targets_func: Callable[[str], list[str]] = parse_guidance_targets,
    merge_discovery_biases_func: Callable[..., Any] = merge_discovery_biases,
    lane_discovery_biases: list[dict[str, Any]] | None = None,
    lane_semantic_focus_families: list[str] | None = None,
    lane_semantic_focus_signals: list[str] | None = None,
) -> ExperimentConfig:
    config = preset_config_func(preset)
    config.log_level = str(getattr(args, "log_level", "compact"))
    config.compress_run_log = not bool(getattr(args, "no_compress_run_log", False))
    config.enable_parallel_backend_execution = not bool(
        getattr(args, "disable_parallel_backend_execution", False)
    )
    config.artifact_limit = getattr(args, "artifact_limit", None)
    if getattr(args, "candidate_recheck_count", None) is not None:
        config.candidate_recheck_count = max(0, int(args.candidate_recheck_count))
    if getattr(args, "metamorphic_variant_limit", None) is not None:
        config.metamorphic_variant_limit = max(0, int(args.metamorphic_variant_limit))
    extra_known = parse_guidance_targets_func(getattr(args, "extra_known_saturated_bug_families", "") or "")
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
        config.discovery_biases = merge_discovery_biases_func(config.discovery_biases, lane_discovery_biases)
    return config


def discovery_run_config_from_args(
    args: argparse.Namespace,
    *,
    preset_config_func: Callable[[str], ExperimentConfig],
    parse_guidance_targets_func: Callable[[str], list[str]] = parse_guidance_targets,
) -> ExperimentConfig:
    config = preset_config_func(str(getattr(args, "preset", "live_deep_organic")))
    config.log_level = str(getattr(args, "log_level", "compact"))
    config.compress_run_log = not bool(getattr(args, "no_compress_run_log", False))
    config.enable_parallel_backend_execution = not bool(
        getattr(args, "disable_parallel_backend_execution", False)
    )
    config.artifact_limit = getattr(args, "artifact_limit", None)
    if getattr(args, "candidate_recheck_count", None) is not None:
        config.candidate_recheck_count = max(0, int(args.candidate_recheck_count))
    if getattr(args, "metamorphic_variant_limit", None) is not None:
        config.metamorphic_variant_limit = max(0, int(args.metamorphic_variant_limit))
    extra_known = parse_guidance_targets_func(getattr(args, "extra_known_saturated_bug_families", "") or "")
    if extra_known:
        config.known_saturated_bug_families = list(
            dict.fromkeys([*config.known_saturated_bug_families, *extra_known])
        )
    return config


def project_relative_cli_path(value: str | Path) -> str:
    if not value:
        return ""
    path = Path(value)
    if not path.is_absolute():
        path = Path.cwd() / path
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def write_discovery_run_fresh_candidate_evidence(
    run_file: Path,
    *,
    classification: dict[str, Any],
    output_path: Path,
    refresh: bool = False,
    read_jsonl_func: Callable[[Path], list[dict[str, Any]]] = read_jsonl,
    candidate_issue_family_key_func: Callable[[dict[str, Any]], str] = candidate_issue_family_key,
    candidate_issue_family_keys_func: Callable[..., Counter[str]] = candidate_issue_family_keys,
    is_rewardable_candidate_issue_finding_func: Callable[..., bool] = is_rewardable_candidate_issue_finding,
    classified_candidate_rows_func: Callable[..., list[dict[str, Any]]] | None = None,
    project_relative_path_func: Callable[[str | Path], str] = project_relative_cli_path,
    utc_now_func: Callable[[], str] = utc_now,
    dump_json_func: Callable[..., None] = dump_json,
) -> dict[str, Any]:
    fresh_families = set(classification.get("fresh_candidate_bug_families", {}))
    candidate_rows: list[dict[str, Any]] = []
    if fresh_families:
        use_injected_raw_rows = (
            not refresh
            and classified_candidate_rows_func is None
            and read_jsonl_func is not read_jsonl
        )
        if use_injected_raw_rows:
            source_rows = read_jsonl_func(run_file)
        else:
            if classified_candidate_rows_func is None:
                from datadiff.run_summaries import _classified_candidate_rows_for_run

                classified_candidate_rows_func = _classified_candidate_rows_for_run
            source_rows = classified_candidate_rows_func(run_file, refresh=refresh)
        if not source_rows and not refresh:
            source_rows = read_jsonl_func(run_file)
        for row in source_rows:
            findings = list(row.get("findings", []) or [])
            known_saturated = row.get("config", {}).get("known_saturated_bug_families", []) or []
            row_family_counts = candidate_issue_family_keys_func(
                findings,
                known_saturated_bug_families=known_saturated,
            )
            row_families = set(row_family_counts)
            if not row_families.intersection(fresh_families):
                continue
            matching_findings = [
                finding
                for finding in findings
                if is_rewardable_candidate_issue_finding_func(
                    finding,
                    known_saturated_bug_families=known_saturated,
                )
                and (
                    candidate_issue_family_key_func(finding) in fresh_families
                    or row_families.intersection(fresh_families)
                )
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
                        "candidate_bug_families": dict(row_family_counts),
                    }
                )
    evidence = {
        "schema_version": "discovery-run-fresh-candidates-v1",
        "generated_at": utc_now_func(),
        "generated_by": "datadiff discovery-run",
        "source_run_file": project_relative_path_func(run_file),
        "refresh_classification": refresh,
        "fresh_candidate_bug_families": classification.get("fresh_candidate_bug_families", {}),
        "candidate_row_count": len(candidate_rows),
        "candidate_rows": candidate_rows,
    }
    dump_json_func(evidence, output_path)
    return evidence


def run_candidate_pipeline_for_evidence(
    args: argparse.Namespace,
    *,
    evidence_path: Path,
    manifest_path: Path,
    build_candidate_pipeline_func: Callable[..., dict[str, Any]] = build_candidate_pipeline,
    default_candidate_pipeline_dir: Path = DEFAULT_CANDIDATE_PIPELINE_DIR,
) -> dict[str, Any]:
    if getattr(args, "skip_candidate_pipeline", False):
        return {}
    try:
        pipeline = build_candidate_pipeline_func(
            evidence_files=[evidence_path],
            manifest_file=manifest_path,
            output_dir=Path(getattr(args, "candidate_pipeline_output_dir", default_candidate_pipeline_dir)),
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
