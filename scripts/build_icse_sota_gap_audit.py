#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FINAL_READINESS = PROJECT_ROOT / "reports" / "final-readiness-20260614T103547.json"
DEFAULT_LATEST_CONFIRMATIONS = PROJECT_ROOT / "experiments" / "latest_confirmations.json"
DEFAULT_SOTA_SNAPSHOT = PROJECT_ROOT / "reports" / "external-baselines" / "sota-gap-snapshot-20260615.json"
DEFAULT_TRIAGE_PLAN = PROJECT_ROOT / "reports" / "bug-20plus-triage-plan-20260614.json"
DEFAULT_MINIMIZED_DUCKDB = PROJECT_ROOT / "reports" / "duckdb-sql-reproducers" / "p0-duckdb-live-20260615-minimized" / "manifest.json"
DEFAULT_DUCKDB_ISSUE_BUNDLE = PROJECT_ROOT / "reports" / "duckdb-issue-ready-bundles" / "p0-duckdb-live-20260615" / "manifest.json"
DEFAULT_OUTPUT_BASE = PROJECT_ROOT / "reports" / "icse-sota-gap-audit-20260615"
OWNED_DISCOVERY_CREDITS = {
    "datadiff_found",
    "datadiff_submitted",
    "user_found",
    "user_submitted",
}


def main() -> int:
    return run_with_args(parse_args())


def run_with_args(args: argparse.Namespace) -> int:
    paths = {
        "final_readiness": resolve_project_path(Path(str(args.final_readiness))),
        "latest_confirmations": resolve_project_path(Path(str(args.latest_confirmations))),
        "sota_snapshot": resolve_project_path(Path(str(args.sota_snapshot))),
        "triage_plan": resolve_project_path(Path(str(args.triage_plan))),
        "minimized_duckdb": resolve_project_path(Path(str(args.minimized_duckdb))),
        "duckdb_issue_bundle": resolve_project_path(Path(str(args.duckdb_issue_bundle))),
    }
    data = {name: load_json(path) if path.is_file() else {} for name, path in paths.items()}
    payload = build_payload(data, paths)
    output_base = resolve_output_base(Path(str(args.output_base)))
    write_outputs(output_base, payload)
    print(f"icse sota gap audit json: {output_base.with_suffix('.json')}")
    print(f"icse sota gap audit md:   {output_base.with_suffix('.md')}")
    return 0


def build_payload(data: dict[str, dict[str, Any]], paths: dict[str, Path]) -> dict[str, Any]:
    readiness = data["final_readiness"]
    confirmations = data["latest_confirmations"]
    snapshot = data["sota_snapshot"]
    triage = data["triage_plan"]
    minimized = data["minimized_duckdb"]
    issue_bundle = data["duckdb_issue_bundle"]
    metrics = derive_metrics(readiness, confirmations, snapshot, triage, minimized, issue_bundle)
    payload = {
        "schema_version": "icse-sota-gap-audit-v1",
        "generated_at": utc_now_iso(),
        "inputs": {name: project_relative(path) if path.is_file() else "" for name, path in paths.items()},
        "counting_policy": (
            "Count only independently confirmed latest-version bug families as real bugs. "
            "Candidate families, SQLancer raw failures, SQLancer queries, DataDiff cases, "
            "and issue-ready bundles are support or triage metrics until confirmed."
        ),
        "metrics": metrics,
        "sota_gap": sota_gap(metrics),
        "project_gaps": project_gaps(metrics),
        "code_and_pipeline_defects": code_and_pipeline_defects(metrics),
        "real_evidence_gaps": real_evidence_gaps(metrics),
        "claim_adjustments": claim_adjustments(metrics),
        "next_reinforcement_plan": next_reinforcement_plan(metrics),
    }
    payload["readiness_verdict"] = readiness_verdict(payload)
    return payload


def derive_metrics(
    readiness: dict[str, Any],
    confirmations: dict[str, Any],
    snapshot: dict[str, Any],
    triage: dict[str, Any],
    minimized: dict[str, Any],
    issue_bundle: dict[str, Any],
) -> dict[str, Any]:
    summary = mapping(readiness.get("summary"))
    quality = mapping(readiness.get("icse_experiment_quality") or summary.get("icse_experiment_quality"))
    dimensions = mapping(quality.get("dimensions"))
    throughput = mapping(dimensions.get("throughput"))
    throughput_evidence = mapping(throughput.get("evidence"))
    datadiff = mapping(snapshot.get("datadiff"))
    strict = mapping(snapshot.get("sqlancer_strict"))
    pilot = mapping(snapshot.get("sqlancer_pilot"))
    gap = mapping(snapshot.get("gap"))
    minimized_summary = mapping(minimized.get("summary"))
    bundle_summary = mapping(issue_bundle.get("summary"))
    latest_confirmation_count = count_confirmations(confirmations)
    readiness_confirmation_count = count_value(readiness.get("latest_confirmations"))
    if readiness_confirmation_count == 0:
        readiness_confirmation_count = count_value(summary.get("confirmed_live_candidate_families"))
    target_20plus = int(triage.get("target_confirmed_count", 20) or 20)
    strong_lower_target = 12
    confirmed_by_backend = confirmation_backend_counts(confirmations)
    avg_cases_s = float(
        datadiff.get("avg_throughput_cases_s")
        or throughput_evidence.get("avg_throughput_cases_s")
        or 0.0
    )
    target_cases_s = float(
        datadiff.get("target_throughput_cases_s")
        or mapping(quality.get("thresholds")).get("target_throughput_cases_s")
        or 5.0
    )
    return {
        "final_readiness_ready": bool(readiness.get("ready")),
        "readiness_failed_gate_count": sum(1 for gate in readiness.get("gates", []) or [] if isinstance(gate, dict) and not gate.get("passed")),
        "icse_quality_grade": quality.get("grade", ""),
        "icse_quality_score": float(quality.get("overall_score", quality.get("score", 0.0)) or 0.0),
        "icse_quality_claim_ready": bool(quality.get("ready_for_icse_claim")),
        "live_runs": int(datadiff.get("live_runs") or summary.get("live_runs") or 0),
        "live_cases": int(datadiff.get("live_cases") or summary.get("total_live_cases") or 0),
        "live_elapsed_seconds": float(datadiff.get("live_elapsed_seconds") or summary.get("total_live_elapsed_s") or 0.0),
        "rewardable_candidate_families": int(datadiff.get("rewardable_live_candidate_families") or count_value(summary.get("rewardable_live_candidate_families")) or 0),
        "latest_confirmed_families": latest_confirmation_count,
        "readiness_embedded_confirmed_families": readiness_confirmation_count,
        "confirmation_registry_delta_vs_readiness": latest_confirmation_count - readiness_confirmation_count,
        "confirmed_backend_counts": dict(sorted(confirmed_by_backend.items())),
        "strong_icse_lower_target": strong_lower_target,
        "strong_icse_lower_gap": max(0, strong_lower_target - latest_confirmation_count),
        "target_20plus_confirmed_families": target_20plus,
        "target_20plus_gap": max(0, target_20plus - latest_confirmation_count),
        "avg_throughput_cases_s": avg_cases_s,
        "target_throughput_cases_s": target_cases_s,
        "throughput_gap_cases_s": round(max(0.0, target_cases_s - avg_cases_s), 6),
        "throughput_target_ratio": round(avg_cases_s / target_cases_s, 6) if target_cases_s > 0 else 0.0,
        "sqlancer_pilot_run_count": int(pilot.get("run_count", 0) or 0),
        "sqlancer_pilot_successful_run_count": int(pilot.get("successful_run_count", 0) or 0),
        "sqlancer_pilot_total_queries": int(pilot.get("total_queries", 0) or 0),
        "sqlancer_pilot_queries_s": float(pilot.get("throughput_queries_s", 0.0) or 0.0),
        "sqlancer_pilot_reported_failures": int(pilot.get("reported_failure_count", 0) or 0),
        "sqlancer_strict_complete": bool(strict.get("complete")),
        "sqlancer_strict_run_count": int(strict.get("run_count", 0) or 0),
        "sqlancer_strict_expected_run_count": int(strict.get("expected_strict_run_count", 0) or 0),
        "sqlancer_strict_reported_failures": int(strict.get("reported_failure_count", 0) or 0),
        "sqlancer_strict_confirmed_bug_families": int(strict.get("confirmed_bug_family_count", 0) or 0),
        "sqlancer_strict_blocker": str(gap.get("strict_comparison_blocker", "")),
        "duckdb_minimized_family_count": int(minimized_summary.get("family_count", 0) or 0),
        "duckdb_minimized_target_reproduced_count": int(minimized_summary.get("target_reproduced_count", 0) or 0),
        "duckdb_minimized_local_sql_ok_count": int(minimized_summary.get("local_sql_execution_passed_count", 0) or 0),
        "duckdb_minimized_native_sql_match_count": int(minimized_summary.get("native_sql_match_count", 0) or 0),
        "duckdb_minimized_native_sql_mismatch_count": int(minimized_summary.get("native_sql_mismatch_count", 0) or 0),
        "duckdb_issue_ready_selected_count": int(bundle_summary.get("selected_count", datadiff.get("duckdb_issue_ready_selected_count", 0)) or 0),
        "duckdb_issue_ready_skipped_count": int(bundle_summary.get("skipped_count", datadiff.get("duckdb_issue_ready_skipped_count", 0)) or 0),
        "triage_candidate_count": len(triage.get("candidates", []) or []),
        "triage_needed_confirmations": int(triage.get("needed_confirmations", max(0, target_20plus - latest_confirmation_count)) or 0),
    }


def sota_gap(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        gap_row(
            "confirmed_latest_version_families",
            metrics["latest_confirmed_families"],
            f"{metrics['strong_icse_lower_target']}-20 strong, {metrics['target_20plus_confirmed_families']}+ stretch",
            "short_of_strong_lower_bar" if metrics["strong_icse_lower_gap"] else "meets_strong_lower_bar",
            f"Need {metrics['strong_icse_lower_gap']} more for the 12-family lower strong ICSE bar and {metrics['target_20plus_gap']} more for 20+.",
        ),
        gap_row(
            "external_sota_strict_sqlancer",
            f"{metrics['sqlancer_strict_run_count']} / {metrics['sqlancer_strict_expected_run_count']} strict runs",
            "complete strict DuckDB 1.5.3 SQLancer comparison",
            "incomplete" if not metrics["sqlancer_strict_complete"] else "complete",
            metrics["sqlancer_strict_blocker"] or "Strict SQLancer comparison completed; triage confirmed families before claiming.",
        ),
        gap_row(
            "throughput",
            f"{metrics['avg_throughput_cases_s']:.6g} cases/s",
            f"{metrics['target_throughput_cases_s']:.6g} cases/s",
            "below_target" if metrics["throughput_gap_cases_s"] > 0 else "meets_target",
            f"Current throughput reaches {metrics['throughput_target_ratio']:.3f} of the aspirational target.",
        ),
        gap_row(
            "duckdb_actionability",
            f"{metrics['duckdb_issue_ready_selected_count']} issue-ready / {metrics['duckdb_minimized_family_count']} minimized candidates",
            "all selected P0 candidates either issue-ready or explicitly downscoped",
            "partial",
            f"{metrics['duckdb_minimized_native_sql_mismatch_count']} minimized DuckDB candidates still need ingestion-path or precision-path reproducers.",
        ),
    ]


def project_gaps(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        item(
            "external_baseline_scope",
            "strict SQLancer baseline is not yet complete",
            "Blocks any head-to-head DuckDB SQL-overlap claim.",
            "Wait for strict 2h x 3-seed run, summarize, triage failures, then decide whether to scale to 24h x 5-10 seeds.",
            "high",
        ),
        item(
            "confirmation_depth",
            "confirmed latest-version count is below the 20+ stretch target",
            "Limits result-strength claims relative to SQLancer-family SOTA papers.",
            f"Use the top {metrics['triage_candidate_count']} triage queue and focus on {metrics['target_20plus_gap']} additional confirmations.",
            "high",
        ),
        item(
            "case_study_balance",
            "confirmed set is DataFusion-heavy",
            f"Backend distribution: {metrics['confirmed_backend_counts']}",
            "Prioritize DuckDB, PyArrow/Arrow, Pandas, and Polars candidates before adding more DataFusion cases.",
            "medium",
        ),
        item(
            "performance_claim_scope",
            "throughput is below aspirational ICSE quality target",
            "Performance should be reported honestly as a tradeoff of cross-ecosystem execution and evidence capture.",
            "Avoid SOTA-speed wording unless a targeted optimization pass improves cases/s.",
            "medium",
        ),
    ]


def code_and_pipeline_defects(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    defects = [
        item(
            "readiness_confirmation_registry_skew",
            "latest confirmation registry and embedded readiness snapshot disagree",
            (
                f"Registry has {metrics['latest_confirmed_families']} confirmations; "
                f"readiness artifact embeds {metrics['readiness_embedded_confirmed_families']}."
            ),
            "Rerun final-readiness after confirmation-registry updates so the authoritative readiness JSON and derived reports share the same count.",
            "medium" if metrics["confirmation_registry_delta_vs_readiness"] else "info",
        ),
        item(
            "native_sql_mismatch_tracking",
            "two minimized DuckDB candidates execute as SQL but do not match reduced DuckDB rerun output",
            "Prevents accidental SQL-only upstream claims for ingestion/precision-path evidence.",
            "Keep strict_native_sql_match gating in the bundle builder; build narrower ingestion/DataFrame-load reproducers for mismatches.",
            "high" if metrics["duckdb_minimized_native_sql_mismatch_count"] else "info",
        ),
        item(
            "strict_baseline_completion_gate",
            "SOTA snapshot correctly marks strict SQLancer comparison incomplete",
            "Prevents version-mismatched or incomplete pilot data from becoming a paper claim.",
            "Keep strict_target_version_match and expected-run-count checks mandatory in comparison reports.",
            "medium" if not metrics["sqlancer_strict_complete"] else "info",
        ),
    ]
    return defects


def real_evidence_gaps(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        item(
            "upstream_confirmations",
            f"{metrics['target_20plus_gap']} additional confirmed/fixed latest-version families needed for 20+",
            "Candidate families do not count without maintainer label, acknowledgement, merged fix, or equivalent upstream confirmation.",
            "Deduplicate and submit polished issue-ready candidates in small batches, then update latest_confirmations only after upstream action.",
            "high",
        ),
        item(
            "duckdb_issue_ready_submission",
            f"{metrics['duckdb_issue_ready_selected_count']} DuckDB issue-ready bundles exist but are not confirmed bugs yet",
            "They are candidate submission evidence, not confirmed latest-version bug counts.",
            "Run upstream duplicate search, submit the strict native SQL matches, and track labels/fixes before counting.",
            "high",
        ),
        item(
            "sqlancer_reproducible_bug_import",
            "local SQLancer DuckDB pilot has 0 reported failures and strict run is incomplete",
            "There is no local SQLancer-found DuckDB bug to reproduce inside DataDiffFuzz yet.",
            "If strict SQLancer later finds a failure, convert it through the same family-dedup and confirmation protocol before comparison.",
            "medium",
        ),
    ]


def claim_adjustments(metrics: dict[str, Any]) -> dict[str, list[str]]:
    return {
        "safe_claims": [
            "DataDiffFuzz has a green final-readiness audit for the current evidence set.",
            (
                "DataDiffFuzz currently records "
                f"{metrics['latest_confirmed_families']} confirmed latest-version bug families "
                "across DataFusion, DuckDB, Polars, and PyArrow."
            ),
            "DataDiffFuzz provides cross-ecosystem semantic differential testing plus an evidence pipeline with queue, witness, reduction, and issue-bundle artifacts.",
            "SQLancer is used as a scope-limited DuckDB SQL-overlap baseline; historical SQLancer/PQS numbers are calibration only.",
        ],
        "unsafe_claims": [
            "Do not claim global outperformance over SQLancer/SQLancer-family DBMS fuzzers.",
            "Do not claim 20+ confirmed latest-version bugs yet.",
            "Do not claim strict DuckDB SQLancer head-to-head results until all strict manifests exist and are triaged.",
            "Do not claim the two native-SQL mismatch DuckDB candidates as SQL-only DuckDB bugs.",
            "Do not claim SOTA-level throughput while average throughput remains below the aspirational 5 cases/s target.",
        ],
        "paper_wording": [
            (
                "Use: We evaluate SQLancer as a scope-limited DuckDB SQL-overlap baseline under equal "
                "wall-clock, seed, thread, target-version, and confirmation rules."
            ),
            (
                "Use: Our primary metric is independently confirmed unique latest-version bug families; "
                "raw queries, cases, and candidate findings are reported as support metrics."
            ),
        ],
    }


def next_reinforcement_plan(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        action(
            1,
            "Finish strict SQLancer fair comparison",
            "Let the active DuckDB 1.5.3 run complete, summarize all manifests, and triage any failure signals.",
            ["sqlancer strict manifests", "fair comparison summary", "updated SOTA gap snapshot"],
        ),
        action(
            2,
            "Refresh authoritative readiness after confirmation updates",
            "Rerun final-readiness with the final manifest index and version ledger so the readiness artifact embeds the current 9-confirmation registry.",
            ["new final-readiness JSON/MD", "0 failed gates", "confirmation count synchronized"],
        ),
        action(
            3,
            "Convert issue-ready DuckDB candidates into upstream evidence",
            "Deduplicate and submit the 2 strict native SQL matches; keep the 2 mismatch cases scoped to ingestion/precision reproducer work.",
            ["DuckDB duplicate-search notes", "submitted issue URLs or explicit duplicate/invalid outcome", "updated latest_confirmations only if confirmed"],
        ),
        action(
            4,
            "Expand confirmed-family count toward 12-20 and 20+",
            f"Work through the ranked triage plan and target {metrics['target_20plus_gap']} additional confirmations for the 20+ stretch bar.",
            ["native/minimized reproducers", "upstream confirmations", "updated 20+ triage report"],
        ),
        action(
            5,
            "Protect claim boundaries in the paper draft",
            "Align abstract/results language with confirmed counts, strict baseline status, and throughput limits.",
            ["claim checklist", "tables using support metrics separately from real-bug counts"],
        ),
    ]


def readiness_verdict(payload: dict[str, Any]) -> dict[str, Any]:
    metrics = payload["metrics"]
    blockers = [
        row["id"]
        for row in payload["sota_gap"] + payload["project_gaps"] + payload["real_evidence_gaps"]
        if row.get("severity") == "high" or row.get("status") in {"incomplete", "below_target", "partial", "short_of_strong_lower_bar"}
    ]
    return {
        "final_readiness_green": bool(metrics["final_readiness_ready"] and metrics["readiness_failed_gate_count"] == 0),
        "high_probability_icse_ready": False,
        "reason": "Final-readiness is green, but strict SOTA comparison, 20+ confirmation depth, throughput target, and some DuckDB actionability evidence remain incomplete.",
        "open_blockers": blockers,
    }


def gap_row(metric: str, current: Any, target: Any, status: str, evidence: str) -> dict[str, Any]:
    return {
        "id": metric,
        "current": current,
        "target": target,
        "status": status,
        "evidence": evidence,
    }


def item(identifier: str, finding: str, impact: str, next_step: str, severity: str) -> dict[str, Any]:
    return {
        "id": identifier,
        "finding": finding,
        "impact": impact,
        "next_step": next_step,
        "severity": severity,
    }


def action(order: int, title: str, action_text: str, evidence: list[str]) -> dict[str, Any]:
    return {
        "order": order,
        "title": title,
        "action": action_text,
        "done_when": evidence,
    }


def write_outputs(output_base: Path, payload: dict[str, Any]) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    output_base.with_suffix(".json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_base.with_suffix(".md").write_text(render_markdown(payload), encoding="utf-8")


def render_markdown(payload: dict[str, Any]) -> str:
    metrics = payload["metrics"]
    verdict = payload["readiness_verdict"]
    lines = [
        "# ICSE SOTA Gap Audit",
        "",
        f"- Generated at: `{payload.get('generated_at', '')}`",
        f"- Final-readiness green: `{str(verdict['final_readiness_green']).lower()}`",
        f"- High-probability ICSE ready: `{str(verdict['high_probability_icse_ready']).lower()}`",
        f"- Verdict: {verdict['reason']}",
        "",
        "## Core Metrics",
        "",
        "| metric | value |",
        "| --- | ---: |",
        f"| latest confirmed families | {metrics['latest_confirmed_families']} |",
        f"| rewardable candidate families | {metrics['rewardable_candidate_families']} |",
        f"| live runs | {metrics['live_runs']} |",
        f"| live cases | {metrics['live_cases']} |",
        f"| avg cases/s | {metrics['avg_throughput_cases_s']} |",
        f"| SQLancer pilot queries/s | {metrics['sqlancer_pilot_queries_s']} |",
        f"| SQLancer strict runs | {metrics['sqlancer_strict_run_count']} / {metrics['sqlancer_strict_expected_run_count']} |",
        f"| DuckDB issue-ready bundles | {metrics['duckdb_issue_ready_selected_count']} |",
        f"| confirmations needed for 20+ | {metrics['target_20plus_gap']} |",
        "",
        "## SOTA Gaps",
        "",
        "| item | current | target | status |",
        "| --- | --- | --- | --- |",
    ]
    for row in payload["sota_gap"]:
        lines.append(f"| {row['id']} | {row['current']} | {row['target']} | {row['status']} |")
    lines.extend(["", "## Project Gaps", ""])
    lines.extend(render_items(payload["project_gaps"]))
    lines.extend(["", "## Code And Pipeline Defects", ""])
    lines.extend(render_items(payload["code_and_pipeline_defects"]))
    lines.extend(["", "## Real Evidence Gaps", ""])
    lines.extend(render_items(payload["real_evidence_gaps"]))
    lines.extend(["", "## Claim Adjustments", "", "Safe claims:"])
    lines.extend(f"- {claim}" for claim in payload["claim_adjustments"]["safe_claims"])
    lines.extend(["", "Unsafe claims:"])
    lines.extend(f"- {claim}" for claim in payload["claim_adjustments"]["unsafe_claims"])
    lines.extend(["", "## Next Reinforcement Plan", ""])
    for row in payload["next_reinforcement_plan"]:
        lines.append(f"{row['order']}. {row['title']}: {row['action']}")
    lines.append("")
    return "\n".join(lines)


def render_items(items: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for row in items:
        lines.append(f"- `{row['id']}` ({row['severity']}): {row['finding']}")
        lines.append(f"  Impact: {row['impact']}")
        lines.append(f"  Next: {row['next_step']}")
    return lines


def count_confirmations(payload: dict[str, Any]) -> int:
    confirmations = payload.get("confirmations", [])
    if not isinstance(confirmations, list):
        return 0
    return len(
        {
            item.get("family")
            for item in confirmations
            if (
                isinstance(item, dict)
                and item.get("family")
                and str(item.get("discovery_credit", "")).strip() in OWNED_DISCOVERY_CREDITS
            )
        }
    )


def confirmation_backend_counts(payload: dict[str, Any]) -> Counter[str]:
    counts: Counter[str] = Counter()
    confirmations = payload.get("confirmations", [])
    if not isinstance(confirmations, list):
        return counts
    for item in confirmations:
        if not isinstance(item, dict):
            continue
        if str(item.get("discovery_credit", "")).strip() not in OWNED_DISCOVERY_CREDITS:
            continue
        for backend in item.get("suspicious_backends", []) or []:
            counts[str(backend)] += 1
    return counts


def count_value(value: Any) -> int:
    if isinstance(value, dict):
        return len(value)
    if isinstance(value, list):
        return len(value)
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def resolve_project_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def resolve_output_base(path: Path) -> Path:
    if path.is_absolute():
        return path
    if path.parent != Path("."):
        return PROJECT_ROOT / path
    return PROJECT_ROOT / "reports" / path


def project_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"expected JSON object: {path}")
    return data


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a consolidated ICSE/SOTA gap audit for DataDiffFuzz.")
    parser.add_argument("--final-readiness", default=str(DEFAULT_FINAL_READINESS))
    parser.add_argument("--latest-confirmations", default=str(DEFAULT_LATEST_CONFIRMATIONS))
    parser.add_argument("--sota-snapshot", default=str(DEFAULT_SOTA_SNAPSHOT))
    parser.add_argument("--triage-plan", default=str(DEFAULT_TRIAGE_PLAN))
    parser.add_argument("--minimized-duckdb", default=str(DEFAULT_MINIMIZED_DUCKDB))
    parser.add_argument("--duckdb-issue-bundle", default=str(DEFAULT_DUCKDB_ISSUE_BUNDLE))
    parser.add_argument("--output-base", default=str(DEFAULT_OUTPUT_BASE))
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
