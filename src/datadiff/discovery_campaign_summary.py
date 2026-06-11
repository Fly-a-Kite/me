from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from datadiff.bug_discovery_system import (
    BugDiscoveryAcquisitionWeights,
    boltzmann_budget_multipliers,
    bug_discovery_system_descriptor,
    lane_true_bug_acquisition,
)
from datadiff.icse_experiment_quality import score_methodology_report
from datadiff.run_summaries import _summarize_run_health
from datadiff.util import PROJECT_ROOT, RUNS_DIR, load_json, utc_now


DEFAULT_DISCOVERY_CAMPAIGN_HISTORY_WINDOW = 8
_DEFAULT_DISCOVERY_ACQUISITION_WEIGHTS = BugDiscoveryAcquisitionWeights()
DEFAULT_DISCOVERY_CAMPAIGN_SCORE_WEIGHTS = {
    "yield_rate": _DEFAULT_DISCOVERY_ACQUISITION_WEIGHTS.yield_rate,
    "novelty_rate": _DEFAULT_DISCOVERY_ACQUISITION_WEIGHTS.novelty_rate,
    "false_positive_penalty": _DEFAULT_DISCOVERY_ACQUISITION_WEIGHTS.false_positive_penalty,
    "uncertainty_weight": _DEFAULT_DISCOVERY_ACQUISITION_WEIGHTS.uncertainty_weight,
    "proof_weight": _DEFAULT_DISCOVERY_ACQUISITION_WEIGHTS.proof_weight,
    "pipeline_actionable_weight": _DEFAULT_DISCOVERY_ACQUISITION_WEIGHTS.pipeline_actionable_weight,
    "duplicate_waste_penalty": _DEFAULT_DISCOVERY_ACQUISITION_WEIGHTS.duplicate_waste_penalty,
    "entropy_weight": _DEFAULT_DISCOVERY_ACQUISITION_WEIGHTS.entropy_weight,
    "saturation_penalty": _DEFAULT_DISCOVERY_ACQUISITION_WEIGHTS.saturation_penalty,
    "issue_inspired_weight": _DEFAULT_DISCOVERY_ACQUISITION_WEIGHTS.issue_inspired_weight,
    "boltzmann_temperature": _DEFAULT_DISCOVERY_ACQUISITION_WEIGHTS.boltzmann_temperature,
}
DISCOVERY_CAMPAIGN_SCHEDULER_STRATEGY = "true_bug_acquisition_good_turing_ucb_boltzmann"
_PIPELINE_COUNT_FIELDS = {
    "pipeline_count",
    "candidate_count",
    "processed_candidate_count",
    "skipped_duplicate_candidate_count",
    "rechecked_count",
    "reproduced_count",
    "reduced_count",
    "candidate_bug_verdict_count",
    "issue_draft_count",
    "needs_dedup_check_count",
    "already_submitted_or_confirmed_count",
    "artifact_created_count",
    "local_duplicate_candidate_count",
    "ready_to_submit_count",
    "not_latest_reproducible_count",
    "needs_reproducer_or_evidence_count",
    "submission_group_count",
    "unique_family_count",
}


def _project_relative_campaign_path(value: str | Path | None) -> str:
    if not value:
        return ""
    path = Path(value)
    if not path.is_absolute():
        path = Path.cwd() / path
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


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
    resolved_score_weights = _resolved_discovery_campaign_score_weights(score_weights)
    lane_rows = _discovery_campaign_lane_rows(
        selected_lanes,
        history_manifests=history_manifests,
        current_runs=runs,
        pending_by_lane=pending_by_lane,
        score_weights=resolved_score_weights,
    )
    return {
        "strategy": DISCOVERY_CAMPAIGN_SCHEDULER_STRATEGY,
        "generated_at": utc_now(),
        "history_manifest_count": len(history_manifests),
        "history_window": history_limit,
        "score_weights": resolved_score_weights,
        "bug_discovery_system": bug_discovery_system_descriptor(),
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
            "fresh_family_counts": Counter(),
            "candidate_pipeline": Counter(),
        }
        for lane in selected_lanes
    }
    events: list[dict[str, Any]] = []

    def ingest_run(run: dict[str, Any], *, source: str, fallback_timestamp: str) -> None:
        lane_id = str(run.get("lane_id", ""))
        if lane_id not in metrics or str(run.get("status", "")) != "completed":
            return
        classification = run.get("classification", {}) if isinstance(run.get("classification"), dict) else {}
        fresh = _positive_counter(classification.get("fresh_candidate_bug_families", {}) or {})
        issue_inspired = _positive_counter(
            classification.get("issue_inspired_unsaturated_candidate_bug_families", {}) or {}
        )
        known = _positive_counter(classification.get("known_saturated_candidate_bug_families", {}) or {})
        false_positive = _positive_counter(classification.get("false_positive_reasons", {}) or {})
        candidate_total = sum(_positive_counter(classification.get("candidate_bug_families", {}) or {}).values())
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
        row["fresh_family_counts"].update(fresh)
        candidate_pipeline = run.get("candidate_pipeline", {}) if isinstance(run.get("candidate_pipeline"), dict) else {}
        pipeline_summary = (
            candidate_pipeline.get("summary", {})
            if isinstance(candidate_pipeline.get("summary", {}), dict)
            else {}
        )
        _update_pipeline_counts(row["candidate_pipeline"], pipeline_summary)
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
    total_completed_runs = sum(int(row["completed_runs"]) for row in metrics.values())
    weights = BugDiscoveryAcquisitionWeights.from_mapping(score_weights)
    for lane in selected_lanes:
        row = metrics[lane["id"]]
        completed_runs = int(row["completed_runs"])
        unique_fresh_family_count = len(row["unique_fresh_families"])
        first_seen_family_count = len(row["first_seen_families"])
        pipeline_summary = row["candidate_pipeline"]
        acquisition = lane_true_bug_acquisition(
            {
                "completed_runs": completed_runs,
                "fresh_candidate_total": row["fresh_candidate_total"],
                "issue_inspired_total": row["issue_inspired_total"],
                "known_saturated_total": row["known_saturated_total"],
                "candidate_total": row["candidate_total"],
                "false_positive_total": row["false_positive_total"],
                "first_seen_family_count": first_seen_family_count,
                "unique_fresh_family_count": unique_fresh_family_count,
                "fresh_family_counts": row["fresh_family_counts"],
                "pipeline_candidate_count": pipeline_summary.get("candidate_count", 0),
                "pipeline_processed_candidate_count": pipeline_summary.get("processed_candidate_count", 0),
                "pipeline_skipped_duplicate_candidate_count": pipeline_summary.get(
                    "skipped_duplicate_candidate_count",
                    0,
                ),
                "pipeline_candidate_bug_verdict_count": pipeline_summary.get("candidate_bug_verdict_count", 0),
                "pipeline_issue_draft_count": pipeline_summary.get("issue_draft_count", 0),
                "pipeline_needs_dedup_check_count": pipeline_summary.get("needs_dedup_check_count", 0),
            },
            total_completed_runs=total_completed_runs,
            weights=weights,
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
                "pipeline_candidate_count": int(pipeline_summary.get("candidate_count", 0)),
                "pipeline_processed_candidate_count": int(pipeline_summary.get("processed_candidate_count", 0)),
                "pipeline_skipped_duplicate_candidate_count": int(
                    pipeline_summary.get("skipped_duplicate_candidate_count", 0)
                ),
                "pipeline_candidate_bug_verdict_count": int(
                    pipeline_summary.get("candidate_bug_verdict_count", 0)
                ),
                "pipeline_issue_draft_count": int(pipeline_summary.get("issue_draft_count", 0)),
                "yield_rate": acquisition["yield_rate"],
                "novelty_rate": acquisition["novelty_rate"],
                "false_positive_rate": acquisition["false_positive_rate"],
                "score": acquisition["acquisition_score"],
                "acquisition_strategy": DISCOVERY_CAMPAIGN_SCHEDULER_STRATEGY,
                "good_turing_unseen_probability": acquisition["good_turing_unseen_probability"],
                "ucb_bonus": acquisition["ucb_bonus"],
                "family_entropy": acquisition["family_entropy"],
                "saturation_rate": acquisition["saturation_rate"],
                "issue_inspired_rate": acquisition["issue_inspired_rate"],
                "proof_yield": acquisition["proof_yield"],
                "pipeline_actionable_rate": acquisition["pipeline_actionable_rate"],
                "pipeline_duplicate_waste_rate": acquisition["pipeline_duplicate_waste_rate"],
                "free_energy": acquisition["free_energy"],
            }
        )

    multipliers = boltzmann_budget_multipliers(
        [float(row["score"]) for row in rows],
        temperature=weights.boltzmann_temperature,
    )
    for row, multiplier in zip(rows, multipliers):
        row["budget_multiplier"] = multiplier
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
    return ranked


def _summarize_discovery_campaign_status(
    manifest_file: Path,
    *,
    limit: int = 3,
    runs_dir: Path = RUNS_DIR,
) -> dict[str, Any]:
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
    latest_observed_run_file = _latest_discovery_campaign_run_file(manifest, runs_dir=runs_dir)
    latest_observed_run_health = (
        _summarize_run_health(latest_observed_run_file, limit=limit) if latest_observed_run_file is not None else {}
    )
    return {
        "schema_version": "discovery-campaign-status-v1",
        "generated_at": utc_now(),
        "manifest_file": _project_relative_campaign_path(manifest_file),
        "manifest_status": str(manifest.get("status", "")),
        "started_at": str(manifest.get("started_at", "")),
        "completed_at": str(manifest.get("completed_at", "")),
        "stopped_by_health": bool(manifest.get("stopped_by_health", False)),
        "health_stop_reason": str(manifest.get("health_stop_reason", "")),
        "progress": dict(manifest.get("progress", {})),
        "scheduler": scheduler,
        "current_run": current_run,
        "latest_observed_run_file": (
            _project_relative_campaign_path(latest_observed_run_file) if latest_observed_run_file is not None else ""
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


def _summarize_discovery_campaign_aggregate(
    manifest_files: list[Path],
    *,
    limit: int = 10,
) -> dict[str, Any]:
    manifests: list[tuple[Path, dict[str, Any]]] = []
    for path in manifest_files:
        if not path.is_file():
            continue
        payload = load_json(path)
        if isinstance(payload, dict) and payload.get("schema_version") == "discovery-campaign-v1":
            manifests.append((path, payload))

    status_counts: Counter[str] = Counter()
    fresh: Counter[str] = Counter()
    issue_inspired: Counter[str] = Counter()
    known: Counter[str] = Counter()
    verdicts: Counter[str] = Counter()
    candidate_pipeline: Counter[str] = Counter()
    lane_rows: dict[str, dict[str, Any]] = {}
    target_suites: set[str] = set()
    presets: set[str] = set()
    seeds: set[int] = set()
    run_logs_total = 0
    run_logs_exist = 0
    total_cases = 0
    executed_cases = 0
    elapsed_s = 0.0
    run_throughputs: list[float] = []
    window_start: datetime | None = None
    window_end: datetime | None = None
    planned_runs = completed_runs = remaining_runs = running_runs = 0
    evidence_rows = 0
    first_candidate: tuple[tuple[float, int, str], dict[str, Any]] | None = None
    auc_contribution_sum = 0.0
    auc_hit_count = 0

    for manifest_path, manifest in manifests:
        manifest_status = str(manifest.get("status", "unknown") or "unknown")
        status_counts[manifest_status] += 1
        started_at = _parse_manifest_utc_timestamp(str(manifest.get("started_at", "") or ""))
        completed_at = _parse_manifest_utc_timestamp(
            str(manifest.get("completed_at", "") or manifest.get("generated_at", "") or "")
        )
        if started_at is not None:
            window_start = started_at if window_start is None else min(window_start, started_at)
        if completed_at is not None:
            window_end = completed_at if window_end is None else max(window_end, completed_at)
        progress = manifest.get("progress", {}) if isinstance(manifest.get("progress"), dict) else {}
        planned_runs += _non_negative_int(progress.get("planned_run_count"))
        completed_runs += _non_negative_int(progress.get("completed_run_count"))
        remaining_runs += _non_negative_int(progress.get("remaining_run_count"))
        summary = manifest.get("summary", {}) if isinstance(manifest.get("summary"), dict) else {}
        fresh.update(_positive_counter(summary.get("fresh_candidate_bug_families", {}) or {}))
        issue_inspired.update(_positive_counter(summary.get("issue_inspired_unsaturated_candidate_bug_families", {}) or {}))
        known.update(_positive_counter(summary.get("known_saturated_candidate_bug_families", {}) or {}))
        verdicts.update(_positive_counter(summary.get("triage_verdicts", {}) or {}))
        _update_pipeline_counts(candidate_pipeline, summary.get("candidate_pipeline", {}) or {})
        for seed in manifest.get("seeds", []) or []:
            try:
                seeds.add(int(seed))
            except (TypeError, ValueError):
                continue
        for run in manifest.get("runs", []) or []:
            if not isinstance(run, dict):
                continue
            lane_id = str(run.get("lane_id", "unknown") or "unknown")
            lane = lane_rows.setdefault(
                lane_id,
                {
                    "lane_id": lane_id,
                    "theme": str(run.get("theme", "")),
                    "target_suites": set(),
                    "presets": set(),
                    "seeds": set(),
                    "planned_runs": 0,
                    "completed_runs": 0,
                    "running_runs": 0,
                    "fresh_candidate_total": 0,
                    "unique_fresh_families": set(),
                    "fresh_candidate_evidence_rows": 0,
                    "candidate_pipeline": Counter(),
                    "first_candidate": None,
                },
            )
            lane["planned_runs"] += 1
            status = str(run.get("status", "unknown") or "unknown")
            if status == "completed":
                lane["completed_runs"] += 1
            elif status == "running":
                lane["running_runs"] += 1
                running_runs += 1
            target_suite = str(run.get("target_suite", "") or "")
            preset = str(run.get("preset", "") or "")
            if target_suite:
                target_suites.add(target_suite)
                lane["target_suites"].add(target_suite)
            if preset:
                presets.add(preset)
                lane["presets"].add(preset)
            try:
                seed = int(run.get("seed"))
                seeds.add(seed)
                lane["seeds"].add(seed)
            except (TypeError, ValueError):
                pass
            total_cases += _non_negative_int(run.get("cases"))
            run_evidence_rows = _non_negative_int(run.get("fresh_candidate_evidence_rows"))
            evidence_rows += run_evidence_rows
            lane["fresh_candidate_evidence_rows"] += run_evidence_rows
            evidence_path = _resolve_project_cli_path(run.get("fresh_candidate_evidence", ""))
            evidence_metrics = _candidate_evidence_metrics(evidence_path, manifest_path=manifest_path, run=run)
            auc_contribution_sum += _non_negative_float(evidence_metrics.get("auc_contribution_sum"))
            auc_hit_count += _non_negative_int(evidence_metrics.get("auc_hit_count"))
            evidence_first = evidence_metrics.get("first_candidate")
            if evidence_first is not None:
                first_key = (
                    _non_negative_float(evidence_first.get("elapsed_s")),
                    _non_negative_int(evidence_first.get("case_index")),
                    str(evidence_first.get("case_id", "")),
                )
                if first_candidate is None or first_key < first_candidate[0]:
                    first_candidate = (first_key, evidence_first)
                lane_first = lane.get("first_candidate")
                if lane_first is None or first_key < lane_first[0]:
                    lane["first_candidate"] = (first_key, evidence_first)
            classification = run.get("classification", {}) if isinstance(run.get("classification"), dict) else {}
            run_fresh = _positive_counter(classification.get("fresh_candidate_bug_families", {}) or {})
            lane["fresh_candidate_total"] += sum(run_fresh.values())
            lane["unique_fresh_families"].update(run_fresh)
            run_pipeline = run.get("candidate_pipeline", {}) if isinstance(run.get("candidate_pipeline"), dict) else {}
            run_pipeline_summary = (
                run_pipeline.get("summary", {}) if isinstance(run_pipeline.get("summary"), dict) else {}
            )
            _update_pipeline_counts(lane["candidate_pipeline"], run_pipeline_summary)
            run_file = _resolve_project_cli_path(run.get("run_file", ""))
            if status == "completed":
                run_logs_total += 1
                if run_file is not None and run_file.is_file():
                    run_logs_exist += 1
            runtime = (
                run.get("health", {}).get("runtime", {})
                if isinstance(run.get("health", {}), dict)
                and isinstance(run.get("health", {}).get("runtime", {}), dict)
                else {}
            )
            executed_cases += _non_negative_int(runtime.get("executed_cases"))
            run_elapsed_s = _non_negative_float(runtime.get("elapsed_s"))
            elapsed_s += run_elapsed_s
            run_throughput = _non_negative_float(runtime.get("throughput_cases_s"))
            if run_throughput > 0:
                run_throughputs.append(run_throughput)

    lane_summary = []
    for lane in lane_rows.values():
        lane_summary.append(
            {
                "lane_id": lane["lane_id"],
                "theme": lane["theme"],
                "target_suites": sorted(lane["target_suites"]),
                "presets": sorted(lane["presets"]),
                "seed_count": len(lane["seeds"]),
                "planned_runs": int(lane["planned_runs"]),
                "completed_runs": int(lane["completed_runs"]),
                "running_runs": int(lane["running_runs"]),
                "fresh_candidate_total": int(lane["fresh_candidate_total"]),
                "unique_fresh_family_count": len(lane["unique_fresh_families"]),
                "fresh_candidate_evidence_rows": int(lane["fresh_candidate_evidence_rows"]),
                "candidate_pipeline": dict(lane["candidate_pipeline"].most_common()),
                "first_candidate": lane["first_candidate"][1] if lane.get("first_candidate") else None,
            }
        )
    lane_summary.sort(
        key=lambda row: (
            -int(row["fresh_candidate_evidence_rows"]),
            -int(row["fresh_candidate_total"]),
            -int(row["completed_runs"]),
            row["lane_id"],
        )
    )

    cases = executed_cases or total_cases
    serial_cases_per_s = (cases / elapsed_s) if elapsed_s > 0 else 0.0
    wall_elapsed_s = (
        max(0.0, (window_end - window_start).total_seconds())
        if window_start is not None and window_end is not None
        else 0.0
    )
    parallel_cases_per_s = (cases / wall_elapsed_s) if wall_elapsed_s > 0 else 0.0
    avg_run_throughput_cases_s = sum(run_throughputs) / len(run_throughputs) if run_throughputs else 0.0
    parallel_capacity_cases_s = _running_parallel_capacity_cases_s(manifests)
    cases_per_s = max(
        serial_cases_per_s,
        parallel_cases_per_s,
        avg_run_throughput_cases_s,
        parallel_capacity_cases_s,
    )
    avg_candidate_bug_discovery_auc = auc_contribution_sum / auc_hit_count if auc_hit_count else 0.0
    candidate_pipeline_payload = dict(candidate_pipeline.most_common())
    if _non_negative_int(candidate_pipeline_payload.get("rechecked_count")):
        candidate_pipeline_payload["recheck_pass_rate"] = _non_negative_int(
            candidate_pipeline_payload.get("reproduced_count")
        ) / _non_negative_int(candidate_pipeline_payload.get("rechecked_count"))
    aggregate = {
        "schema_version": "discovery-campaign-aggregate-v1",
        "generated_at": utc_now(),
        "manifest_count": len(manifests),
        "manifest_files": [_project_relative_campaign_path(path) for path, _ in manifests],
        "status_counts": dict(status_counts.most_common()),
        "progress": {
            "planned_run_count": planned_runs,
            "completed_run_count": completed_runs,
            "remaining_run_count": remaining_runs,
            "running_run_count": running_runs,
        },
        "coverage": {
            "lane_count": len(lane_rows),
            "target_suite_count": len(target_suites),
            "preset_count": len(presets),
            "seed_count": len(seeds),
            "target_suites": sorted(target_suites),
            "presets": sorted(presets),
        },
        "efficiency": {
            "cases": cases,
            "executed_cases": executed_cases,
            "planned_cases": total_cases,
            "elapsed_s": elapsed_s,
            "serial_elapsed_s": elapsed_s,
            "wall_elapsed_s": wall_elapsed_s,
            "cases_per_s": cases_per_s,
            "serial_cases_per_s": serial_cases_per_s,
            "parallel_cases_per_s": parallel_cases_per_s,
            "parallel_capacity_cases_s": parallel_capacity_cases_s,
            "avg_run_throughput_cases_s": avg_run_throughput_cases_s,
        },
        "fresh_candidate_bug_families": dict(fresh.most_common(limit)),
        "fresh_candidate_bug_family_count": len(fresh),
        "issue_inspired_unsaturated_candidate_bug_families": dict(issue_inspired.most_common(limit)),
        "known_saturated_candidate_bug_families": dict(known.most_common(limit)),
        "triage_verdicts": dict(verdicts.most_common()),
        "fresh_candidate_evidence_rows": evidence_rows,
        "avg_candidate_bug_discovery_auc": avg_candidate_bug_discovery_auc,
        "candidate_pipeline": candidate_pipeline_payload,
        "first_candidate": first_candidate[1] if first_candidate else None,
        "lanes": lane_summary[:limit],
    }
    aggregate["icse_experiment_quality"] = score_methodology_report(
        {
            "bug_discovery": {
                "candidate_bug_families": dict(fresh),
                "candidate_bug_family_count": len(fresh),
                "first_candidate": first_candidate[1] if first_candidate else None,
                "avg_candidate_bug_discovery_auc": avg_candidate_bug_discovery_auc,
            },
            "efficiency": {
                "cases": cases,
                "elapsed_s": elapsed_s,
                "cases_per_s": cases_per_s,
            },
            "coverage": {
                "target_suites": sorted(target_suites),
                "presets": sorted(presets),
                "run_count": completed_runs,
            },
            "reproducibility": {
                "run_logs_total": run_logs_total,
                "run_logs_exist": run_logs_exist,
                "artifact_reproducer_coverage": 1.0 if _non_negative_int(candidate_pipeline.get("reduced_count")) else 0.0,
            },
            "candidate_pipeline": candidate_pipeline_payload,
        }
    )
    return aggregate


def _current_discovery_campaign_run(manifest: dict[str, Any]) -> dict[str, Any]:
    runs = list(manifest.get("runs", []) or [])
    for run in reversed(runs):
        if run.get("status") == "running":
            return dict(run)
    return dict(runs[-1]) if runs else {}


def _latest_discovery_campaign_run_file(
    manifest: dict[str, Any],
    *,
    runs_dir: Path = RUNS_DIR,
) -> Path | None:
    recorded: list[Path] = []
    for run in manifest.get("runs", []) or []:
        resolved = _resolve_project_cli_path(run.get("run_file", ""))
        if resolved is not None and resolved.is_file():
            recorded.append(resolved)
    if recorded:
        return max(recorded, key=lambda path: (path.stat().st_mtime, path.name))

    started_at = _parse_manifest_utc_timestamp(str(manifest.get("started_at", "")))
    candidates = sorted(
        [*runs_dir.glob("run-*.jsonl"), *runs_dir.glob("run-*.jsonl.gz")],
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


def _resolved_discovery_campaign_score_weights(score_weights: dict[str, float]) -> dict[str, float]:
    weights = BugDiscoveryAcquisitionWeights.from_mapping(score_weights)
    return {
        "yield_rate": weights.yield_rate,
        "novelty_rate": weights.novelty_rate,
        "false_positive_penalty": weights.false_positive_penalty,
        "uncertainty_weight": weights.uncertainty_weight,
        "proof_weight": weights.proof_weight,
        "pipeline_actionable_weight": weights.pipeline_actionable_weight,
        "duplicate_waste_penalty": weights.duplicate_waste_penalty,
        "entropy_weight": weights.entropy_weight,
        "saturation_penalty": weights.saturation_penalty,
        "issue_inspired_weight": weights.issue_inspired_weight,
        "boltzmann_temperature": weights.boltzmann_temperature,
    }


def _positive_counter(value: Any) -> Counter[str]:
    if not isinstance(value, dict):
        return Counter()
    result: Counter[str] = Counter()
    for key, raw_count in value.items():
        try:
            count = int(raw_count)
        except (TypeError, ValueError):
            continue
        if count > 0:
            result[str(key)] += count
    return result


def _update_pipeline_counts(counter: Counter[str], value: Any) -> None:
    if not isinstance(value, dict):
        return
    if "processed_candidate_count" not in value:
        candidate_count = _non_negative_int(value.get("candidate_count"))
        skipped_duplicate_count = _non_negative_int(value.get("skipped_duplicate_candidate_count"))
        processed_candidate_count = max(0, candidate_count - skipped_duplicate_count)
        if processed_candidate_count > 0:
            counter["processed_candidate_count"] += processed_candidate_count
    for key, raw_count in value.items():
        if str(key) not in _PIPELINE_COUNT_FIELDS:
            continue
        count = _non_negative_int(raw_count)
        if count > 0:
            counter[str(key)] += count


def _running_parallel_capacity_cases_s(manifests: list[tuple[Path, dict[str, Any]]]) -> float:
    total = 0.0
    for _, manifest in manifests:
        if str(manifest.get("status", "") or "") != "running":
            continue
        latest: tuple[datetime, float] | None = None
        for run in manifest.get("runs", []) or []:
            if not isinstance(run, dict):
                continue
            runtime = (
                run.get("health", {}).get("runtime", {})
                if isinstance(run.get("health", {}), dict)
                and isinstance(run.get("health", {}).get("runtime", {}), dict)
                else {}
            )
            throughput = _non_negative_float(runtime.get("throughput_cases_s"))
            if throughput <= 0.0:
                elapsed = _non_negative_float(runtime.get("elapsed_s"))
                executed = _non_negative_int(runtime.get("executed_cases"))
                throughput = executed / elapsed if elapsed > 0.0 else 0.0
            if throughput <= 0.0:
                continue
            timestamp = _parse_manifest_utc_timestamp(
                str(run.get("completed_at", "") or run.get("started_at", "") or manifest.get("generated_at", ""))
            ) or datetime.min.replace(tzinfo=timezone.utc)
            if latest is None or timestamp > latest[0]:
                latest = (timestamp, throughput)
        if latest is not None:
            total += latest[1]
    return total


def _candidate_evidence_metrics(
    evidence_path: Path | None,
    *,
    manifest_path: Path,
    run: dict[str, Any],
) -> dict[str, Any]:
    if evidence_path is None or not evidence_path.is_file():
        return {"first_candidate": None, "auc_contribution_sum": 0.0, "auc_hit_count": 0}
    payload = load_json(evidence_path)
    if not isinstance(payload, dict):
        return {"first_candidate": None, "auc_contribution_sum": 0.0, "auc_hit_count": 0}
    best: tuple[tuple[float, int, str], dict[str, Any]] | None = None
    payload_families = sorted(_positive_counter(payload.get("fresh_candidate_bug_families", {}) or {}))
    case_budget = max(1, _non_negative_int(run.get("cases")))
    auc_contribution_sum = 0.0
    auc_hit_count = 0
    for row in payload.get("candidate_rows", []) or []:
        if not isinstance(row, dict):
            continue
        raw_case_index = _optional_non_negative_int(row.get("case_index"))
        if raw_case_index is not None:
            case_index_for_auc = min(case_budget - 1, raw_case_index)
            auc_contribution_sum += (case_budget - case_index_for_auc) / case_budget
            auc_hit_count += 1
        elapsed = _optional_non_negative_float(row.get("elapsed_s"))
        if elapsed is None:
            continue
        case = row.get("case", {}) if isinstance(row.get("case", {}), dict) else {}
        case_id = str(case.get("case_id", "") or "")
        case_index = _non_negative_int(row.get("case_index"))
        families = sorted(_positive_counter(row.get("candidate_bug_families", {}) or {})) or payload_families
        candidate = {
            "manifest_file": _project_relative_campaign_path(manifest_path),
            "evidence_file": _project_relative_campaign_path(evidence_path),
            "source_run_file": str(payload.get("source_run_file", "") or run.get("run_file", "")),
            "lane_id": str(run.get("lane_id", "") or ""),
            "seed": run.get("seed", ""),
            "case_id": case_id,
            "case_index": case_index,
            "elapsed_s": elapsed,
            "families": families,
        }
        key = (elapsed, case_index, case_id)
        if best is None or key < best[0]:
            best = (key, candidate)
    return {
        "first_candidate": best[1] if best else None,
        "auc_contribution_sum": auc_contribution_sum,
        "auc_hit_count": auc_hit_count,
    }


def _non_negative_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _non_negative_float(value: Any) -> float:
    try:
        return max(0.0, float(value or 0.0))
    except (TypeError, ValueError):
        return 0.0


def _optional_non_negative_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return None


def _optional_non_negative_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return None


def _parse_manifest_utc_timestamp(value: str) -> datetime | None:
    cleaned = value.strip()
    if not cleaned:
        return None
    try:
        return datetime.fromisoformat(cleaned.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None
