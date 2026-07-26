#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FINAL_READINESS = PROJECT_ROOT / "reports" / "final-readiness-20260614T103547.json"
DEFAULT_ISSUE_BUNDLE = PROJECT_ROOT / "reports" / "duckdb-issue-ready-bundles" / "p0-duckdb-live-20260615" / "manifest.json"
DEFAULT_SQLANCER_QP_PILOT = PROJECT_ROOT / "reports" / "external-baselines" / "sqlancer-fair-2h-3seed-20260614-duckdb-query-partitioning-7200s.json"
DEFAULT_SQLANCER_NOREC_PILOT = PROJECT_ROOT / "reports" / "external-baselines" / "sqlancer-fair-2h-3seed-20260614-duckdb-norec-7200s.json"
DEFAULT_SQLANCER_STRICT_GLOB = "reports/external-baselines/sqlancer-fair-duckdb153-2h-3seed-20260615-*.json"
DEFAULT_LATEST_CONFIRMATIONS = PROJECT_ROOT / "experiments" / "latest_confirmations.json"
DEFAULT_OUTPUT_BASE = PROJECT_ROOT / "reports" / "external-baselines" / "sota-gap-snapshot-20260615"
OWNED_DISCOVERY_CREDITS = {
    "datadiff_found",
    "datadiff_submitted",
    "user_found",
    "user_submitted",
}


def main() -> int:
    return run_with_args(parse_args())


def run_with_args(args: argparse.Namespace) -> int:
    final_readiness_path = resolve_project_path(Path(str(args.final_readiness)))
    issue_bundle_path = resolve_project_path(Path(str(args.issue_bundle)))
    latest_confirmations_path = resolve_project_path(Path(str(args.latest_confirmations)))
    pilot_paths = [resolve_project_path(Path(str(path))) for path in args.sqlancer_pilot_manifest]
    strict_paths = sorted(PROJECT_ROOT.glob(str(args.sqlancer_strict_glob)))
    final_readiness = load_json(final_readiness_path)
    issue_bundle = load_json(issue_bundle_path) if issue_bundle_path.is_file() else {}
    latest_confirmations = load_json(latest_confirmations_path) if latest_confirmations_path.is_file() else {}
    pilot_manifests = [load_json(path) for path in pilot_paths if path.is_file()]
    strict_manifests = [load_json(path) for path in strict_paths if path.is_file()]
    payload = {
        "schema_version": "sota-gap-snapshot-v1",
        "generated_at": utc_now_iso(),
        "inputs": {
            "final_readiness": project_relative(final_readiness_path),
            "issue_bundle": project_relative(issue_bundle_path),
            "latest_confirmations": project_relative(latest_confirmations_path) if latest_confirmations_path.is_file() else "",
            "sqlancer_pilot_manifests": [project_relative(path) for path in pilot_paths if path.is_file()],
            "sqlancer_strict_glob": str(args.sqlancer_strict_glob),
            "sqlancer_strict_manifests": [project_relative(path) for path in strict_paths],
        },
        "counting_policy": (
            "Primary comparison count is latest-version bug families found/submitted by this project "
            "and independently confirmed by upstream. Similar pre-existing upstream issues are support/dedup "
            "evidence only. Throughput, queries, cases, candidates, and issue-ready bundles are support metrics."
        ),
        "sqlancer_pqs_historical": {
            "reports": 121,
            "true_previously_unknown_bugs": 96,
            "scope": "SQLite/MySQL/PostgreSQL in SQLancer/PQS OSDI 2020; calibration only, not local DuckDB evidence",
        },
        "datadiff": datadiff_metrics(final_readiness, issue_bundle, latest_confirmations),
        "sqlancer_pilot": sqlancer_metrics(pilot_manifests, strict_required=False),
        "sqlancer_strict": sqlancer_metrics(strict_manifests, strict_required=True),
    }
    payload["gap"] = gap_assessment(payload)
    output_base = resolve_output_base(Path(str(args.output_base)))
    write_outputs(output_base, payload)
    print(f"sota gap snapshot json: {output_base.with_suffix('.json')}")
    print(f"sota gap snapshot md:   {output_base.with_suffix('.md')}")
    return 0


def datadiff_metrics(
    final_readiness: dict[str, Any],
    issue_bundle: dict[str, Any],
    latest_confirmations: dict[str, Any],
) -> dict[str, Any]:
    summary = final_readiness.get("summary", {}) if isinstance(final_readiness.get("summary", {}), dict) else {}
    quality = final_readiness.get("icse_experiment_quality", {}) if isinstance(final_readiness.get("icse_experiment_quality", {}), dict) else {}
    throughput_evidence = (
        quality.get("dimensions", {}).get("throughput", {}).get("evidence", {})
        if isinstance(quality.get("dimensions", {}), dict)
        else {}
    )
    real_bug_evidence = (
        quality.get("dimensions", {}).get("real_bug_yield", {}).get("evidence", {})
        if isinstance(quality.get("dimensions", {}), dict)
        else {}
    )
    bundle_summary = issue_bundle.get("summary", {}) if isinstance(issue_bundle.get("summary", {}), dict) else {}
    latest_confirmation_count = count_latest_confirmations(latest_confirmations)
    confirmed_families = (
        latest_confirmation_count
        if latest_confirmation_count
        else first_count(
            summary.get("confirmed_latest_version_bug_families"),
            final_readiness.get("latest_confirmations"),
            summary.get("confirmed_live_candidate_families"),
            real_bug_evidence.get("confirmed_live_candidate_families") if isinstance(real_bug_evidence, dict) else None,
        )
    )
    rewardable_families = first_count(
        summary.get("rewardable_live_candidate_families"),
        summary.get("rewardable_live_candidate_family_count"),
        real_bug_evidence.get("rewardable_live_candidate_families") if isinstance(real_bug_evidence, dict) else None,
    )
    return {
        "final_readiness_ready": bool(final_readiness.get("ready")),
        "confirmed_latest_version_bug_families": int(confirmed_families or 0),
        "rewardable_live_candidate_families": int(rewardable_families or 0),
        "live_runs": int(summary.get("live_runs", 0) or 0),
        "live_cases": int(summary.get("total_live_cases", 0) or 0),
        "live_elapsed_seconds": float(summary.get("total_live_elapsed_s", 0.0) or 0.0),
        "avg_throughput_cases_s": float(
            throughput_evidence.get("avg_throughput_cases_s", summary.get("runtime_efficiency", {}).get("avg_throughput_cases_s", 0.0))
            or 0.0
        ),
        "target_throughput_cases_s": float(
            quality.get("thresholds", {}).get("target_throughput_cases_s", 5.0)
            if isinstance(quality.get("thresholds", {}), dict)
            else 5.0
        ),
        "duckdb_issue_ready_selected_count": int(bundle_summary.get("selected_count", 0) or 0),
        "duckdb_issue_ready_skipped_count": int(bundle_summary.get("skipped_count", 0) or 0),
    }


def first_count(*values: Any) -> int | None:
    for value in values:
        count = count_value(value)
        if count is not None:
            return count
    return None


def count_value(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return len(value)
    if isinstance(value, list):
        return len(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def count_latest_confirmations(latest_confirmations: dict[str, Any]) -> int:
    confirmations = latest_confirmations.get("confirmations", [])
    if not isinstance(confirmations, list):
        return 0
    families = {
        str(item.get("family"))
        for item in confirmations
        if (
            isinstance(item, dict)
            and item.get("family")
            and str(item.get("discovery_credit", "")).strip() in OWNED_DISCOVERY_CREDITS
        )
    }
    return len(families)


def sqlancer_metrics(manifests: list[dict[str, Any]], *, strict_required: bool) -> dict[str, Any]:
    run_count = 0
    successful = 0
    reported_failures = 0
    total_queries = 0
    total_databases = 0
    total_elapsed = 0.0
    strict_match_count = 0
    for manifest in manifests:
        strict_match = bool(manifest.get("strict_target_version_match"))
        for run in manifest.get("runs", []) or []:
            if not isinstance(run, dict):
                continue
            run_count += 1
            if int(run.get("returncode", 1) or 0) == 0:
                successful += 1
            stats = run.get("stats", {}) if isinstance(run.get("stats", {}), dict) else {}
            failure = run.get("failure_signal", {}) if isinstance(run.get("failure_signal", {}), dict) else {}
            total_queries += int(stats.get("summary_queries", 0) or 0)
            total_databases += int(stats.get("summary_databases", 0) or 0)
            total_elapsed += float(run.get("elapsed_seconds", 0.0) or 0.0)
            reported_failures += int(bool(failure.get("has_signal")))
            strict_match_count += int(strict_match)
    throughput = total_queries / total_elapsed if total_elapsed > 0 else 0.0
    expected_strict_runs = 6 if strict_required else run_count
    complete = bool(run_count >= expected_strict_runs and run_count > 0)
    if strict_required and strict_match_count < run_count:
        complete = False
    return {
        "manifest_count": len(manifests),
        "run_count": run_count,
        "successful_run_count": successful,
        "reported_failure_count": reported_failures,
        "confirmed_bug_family_count": 0,
        "total_queries": total_queries,
        "total_databases": total_databases,
        "total_elapsed_seconds": round(total_elapsed, 6),
        "throughput_queries_s": round(throughput, 6),
        "strict_target_match_run_count": strict_match_count,
        "expected_strict_run_count": expected_strict_runs if strict_required else "",
        "complete": complete,
        "status": "complete" if complete else "incomplete",
    }


def gap_assessment(payload: dict[str, Any]) -> dict[str, Any]:
    datadiff = payload["datadiff"]
    pilot = payload["sqlancer_pilot"]
    strict = payload["sqlancer_strict"]
    avg_cases_s = float(datadiff.get("avg_throughput_cases_s", 0.0) or 0.0)
    sqlancer_pilot_qps = float(pilot.get("throughput_queries_s", 0.0) or 0.0)
    throughput_ratio = sqlancer_pilot_qps / avg_cases_s if avg_cases_s > 0 else None
    return {
        "strict_comparison_ready": bool(strict.get("complete")),
        "strict_comparison_blocker": "" if strict.get("complete") else "DuckDB 1.5.3 SQLancer strict fair run has not produced all required manifests.",
        "local_sqlancer_confirmed_bug_families": int(strict.get("confirmed_bug_family_count", 0) or 0),
        "datadiff_confirmed_bug_families": int(datadiff.get("confirmed_latest_version_bug_families", 0) or 0),
        "datadiff_issue_ready_duckdb_candidates": int(datadiff.get("duckdb_issue_ready_selected_count", 0) or 0),
        "sqlancer_pilot_query_throughput_vs_datadiff_case_throughput_ratio": round(throughput_ratio, 3) if throughput_ratio is not None else "",
        "claim": (
            "Use SQLancer pilot throughput only as support evidence. Do not claim a strict head-to-head win until "
            "the DuckDB 1.5.3 strict run completes and both sides are triaged under the same confirmation policy."
        ),
    }


def write_outputs(output_base: Path, payload: dict[str, Any]) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    output_base.with_suffix(".json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_base.with_suffix(".md").write_text(render_markdown(payload), encoding="utf-8")


def render_markdown(payload: dict[str, Any]) -> str:
    datadiff = payload.get("datadiff", {})
    pilot = payload.get("sqlancer_pilot", {})
    strict = payload.get("sqlancer_strict", {})
    gap = payload.get("gap", {})
    lines = [
        "# SOTA Gap Snapshot",
        "",
        f"- Generated at: `{payload.get('generated_at', '')}`",
        f"- Strict comparison ready: `{str(gap.get('strict_comparison_ready', False)).lower()}`",
        f"- Strict blocker: {gap.get('strict_comparison_blocker', '') or 'none'}",
        "",
        "## Current Metrics",
        "",
        "| metric | value |",
        "| --- | ---: |",
        f"| DataDiff confirmed latest-version families | {datadiff.get('confirmed_latest_version_bug_families', 0)} |",
        f"| DataDiff rewardable live candidate families | {datadiff.get('rewardable_live_candidate_families', 0)} |",
        f"| DataDiff live cases | {datadiff.get('live_cases', 0)} |",
        f"| DataDiff avg cases/s | {datadiff.get('avg_throughput_cases_s', 0)} |",
        f"| DataDiff DuckDB issue-ready candidates | {datadiff.get('duckdb_issue_ready_selected_count', 0)} |",
        f"| SQLancer pilot runs | {pilot.get('successful_run_count', 0)} / {pilot.get('run_count', 0)} |",
        f"| SQLancer pilot queries | {pilot.get('total_queries', 0)} |",
        f"| SQLancer pilot queries/s | {pilot.get('throughput_queries_s', 0)} |",
        f"| SQLancer strict runs | {strict.get('successful_run_count', 0)} / {strict.get('expected_strict_run_count', 0)} |",
        f"| SQLancer strict reported failures | {strict.get('reported_failure_count', 0)} |",
        "",
        "## Claim Boundary",
        "",
        str(gap.get("claim", "")),
        "",
    ]
    return "\n".join(lines)


def resolve_project_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def resolve_output_base(path: Path) -> Path:
    if path.is_absolute():
        return path
    if path.parent != Path("."):
        return PROJECT_ROOT / path
    return PROJECT_ROOT / "reports" / "external-baselines" / path


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"expected JSON object: {path}")
    return data


def project_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a current DataDiffFuzz-vs-SQLancer SOTA gap snapshot.")
    parser.add_argument("--final-readiness", default=str(DEFAULT_FINAL_READINESS))
    parser.add_argument("--issue-bundle", default=str(DEFAULT_ISSUE_BUNDLE))
    parser.add_argument("--latest-confirmations", default=str(DEFAULT_LATEST_CONFIRMATIONS))
    parser.add_argument("--sqlancer-pilot-manifest", action="append", default=[
        str(DEFAULT_SQLANCER_QP_PILOT),
        str(DEFAULT_SQLANCER_NOREC_PILOT),
    ])
    parser.add_argument("--sqlancer-strict-glob", default=DEFAULT_SQLANCER_STRICT_GLOB)
    parser.add_argument("--output-base", default=str(DEFAULT_OUTPUT_BASE))
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
