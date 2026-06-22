#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "external-baselines"


def main() -> int:
    return run_with_args(parse_args())


def run_with_args(args: argparse.Namespace) -> int:
    rows: list[dict[str, object]] = []
    for manifest in args.sqlancer_manifest:
        rows.extend(sqlancer_rows(Path(str(manifest))))
    for manifest in args.datadiff_manifest:
        rows.extend(datadiff_rows(Path(str(manifest)), refresh=bool(args.refresh_datadiff_summary)))
    summary = build_summary(rows)
    output_base = resolve_output_base(Path(str(args.output_base)))
    payload = {
        "schema_version": "sqlancer-fair-comparison-summary-v1",
        "generated_at": utc_now_iso(),
        "scope": "DuckDB/embedded-SQL overlap support evidence",
        "counting_policy": (
            "Raw findings are not paper real-bug counts. Count only independently triaged "
            "unique confirmed bug families under the same confirmation protocol."
        ),
        "metric_policy": {
            "primary": [
                "unique confirmed latest-version bug families in the DuckDB SQL-overlap scope",
                "confirmed families per wall-clock hour",
                "time to first confirmed family",
                "upstream outcome: labeled/open, fixed/merged, duplicate, invalid",
            ],
            "supporting": [
                "SQLancer generated queries, generated databases, query throughput, and reported failures",
                "DataDiffFuzz executed cases, candidate bug cases, rewardable candidate families, and case throughput",
                "minimizer/reproducer/recheck/issue-bundle success",
                "semantic-class and target-scope coverage",
            ],
            "not_primary": [
                "raw SQLancer query count versus raw DataDiffFuzz case count",
                "raw failure/finding count without independent confirmation",
                "DataFrame/Arrow/Polars/PyArrow discoveries as head-to-head SQLancer wins",
            ],
        },
        "rows": rows,
        "summary": summary,
    }
    write_outputs(output_base, payload)
    print(f"fair comparison summary json: {output_base.with_suffix('.json')}")
    print(f"fair comparison summary csv:  {output_base.with_suffix('.csv')}")
    print(f"fair comparison summary md:   {output_base.with_suffix('.md')}")
    return 0


def sqlancer_rows(path: Path) -> list[dict[str, object]]:
    data = load_json(path)
    target_versions = data.get("target_versions", {}) if isinstance(data.get("target_versions"), dict) else {}
    normalized_versions = (
        target_versions.get("normalized", {}) if isinstance(target_versions.get("normalized"), dict) else {}
    )
    strict_target_version_match = bool(data.get("strict_target_version_match"))
    rows: list[dict[str, object]] = []
    for run in data.get("runs", []) or []:
        if not isinstance(run, dict):
            continue
        spec = run.get("spec", {}) if isinstance(run.get("spec"), dict) else {}
        stats = run.get("stats", {}) if isinstance(run.get("stats"), dict) else {}
        failure_signal = run.get("failure_signal", {}) if isinstance(run.get("failure_signal"), dict) else {}
        elapsed = float(run.get("elapsed_seconds", 0.0) or 0.0)
        queries = int(stats.get("summary_queries", 0) or 0)
        rows.append(
            {
                "tool": "sqlancer",
                "role": "external_sota_baseline",
                "manifest": project_relative(path),
                "suite": str(spec.get("suite", "")),
                "target": str(spec.get("dbms", "")),
                "oracle": str(spec.get("oracle", "")),
                "seed": int(spec.get("seed", 0) or 0),
                "status": str(run.get("status", "")),
                "returncode": int(run.get("returncode", 0) or 0),
                "duration_seconds": elapsed,
                "queries": queries,
                "databases": int(stats.get("summary_databases", 0) or 0),
                "reported_failure": bool(failure_signal.get("has_signal")),
                "reported_failure_lines": len(failure_signal.get("matched_lines", []) or []),
                "strict_target_version_match": strict_target_version_match,
                "datadiff_duckdb_version": str(normalized_versions.get("datadiff_duckdb", "")),
                "sqlancer_duckdb_version": str(normalized_versions.get("sqlancer_duckdb_jdbc", "")),
                "cases": "",
                "candidate_bug_cases": "",
                "candidate_bug_families": "",
                "confirmed_bug_families": "",
                "findings": "",
                "throughput_queries_s": round(queries / elapsed, 6) if elapsed > 0 else 0.0,
                "throughput_cases_s": "",
                "log_file": str(run.get("log_file", "")),
            }
        )
    return rows


def datadiff_rows(path: Path, *, refresh: bool) -> list[dict[str, object]]:
    if refresh:
        run_experiment_summary(path)
    aggregate_csv = PROJECT_ROOT / "reports" / f"experiment-summary-{path.stem}-aggregates.csv"
    if aggregate_csv.is_file():
        return datadiff_rows_from_aggregate_csv(path, aggregate_csv)
    return datadiff_rows_from_manifest(path)


def datadiff_rows_from_aggregate_csv(path: Path, aggregate_csv: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with aggregate_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            cases = int_float(row.get("cases"))
            elapsed = cases / int_float(row.get("avg_throughput_cases_s")) if int_float(row.get("avg_throughput_cases_s")) else 0.0
            rows.append(
                {
                    "tool": "datadiff",
                    "role": "subject_tool",
                    "manifest": project_relative(path),
                    "suite": row.get("target_suite", ""),
                    "target": row.get("backends", ""),
                    "oracle": row.get("oracle_profile", ""),
                    "seed": "",
                    "status": "completed",
                    "returncode": 0,
                    "duration_seconds": round(elapsed, 6),
                    "queries": "",
                    "databases": "",
                    "reported_failure": "",
                    "reported_failure_lines": "",
                    "strict_target_version_match": "",
                    "datadiff_duckdb_version": "",
                    "sqlancer_duckdb_version": "",
                    "cases": int(cases),
                    "candidate_bug_cases": int(int_float(row.get("candidate_bug_cases"))),
                    "candidate_bug_families": row.get("top_candidate_bug_families", "")
                    or row.get("candidate_bug_families", ""),
                    "confirmed_bug_families": row.get("confirmed_bug_families", ""),
                    "findings": int(int_float(row.get("findings"))),
                    "throughput_queries_s": "",
                    "throughput_cases_s": round(int_float(row.get("avg_throughput_cases_s")), 6),
                    "log_file": "",
                }
            )
    return rows


def datadiff_rows_from_manifest(path: Path) -> list[dict[str, object]]:
    data = load_json(path)
    rows: list[dict[str, object]] = []
    for run in data.get("runs", []) or []:
        if not isinstance(run, dict):
            continue
        rows.append(
            {
                "tool": "datadiff",
                "role": "subject_tool",
                "manifest": project_relative(path),
                "suite": str(run.get("target_suite", "")),
                "target": ",".join(str(item) for item in run.get("backends", []) or []),
                "oracle": str(run.get("oracle_profile", "")),
                "seed": int(run.get("seed", 0) or 0),
                "status": "planned_or_manifest_only",
                "returncode": "",
                "duration_seconds": "",
                "queries": "",
                "databases": "",
                "reported_failure": "",
                "reported_failure_lines": "",
                "strict_target_version_match": "",
                "datadiff_duckdb_version": "",
                "sqlancer_duckdb_version": "",
                "cases": "",
                "candidate_bug_cases": "",
                "candidate_bug_families": "",
                "confirmed_bug_families": "",
                "findings": "",
                "throughput_queries_s": "",
                "throughput_cases_s": "",
                "log_file": str(run.get("run_file", "")),
            }
        )
    return rows


def build_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    sqlancer = [row for row in rows if row.get("tool") == "sqlancer"]
    datadiff = [row for row in rows if row.get("tool") == "datadiff"]
    return {
        "row_count": len(rows),
        "sqlancer_run_count": len(sqlancer),
        "sqlancer_successful_run_count": sum(1 for row in sqlancer if row_returncode(row) == 0),
        "sqlancer_strict_target_match_run_count": sum(
            1 for row in sqlancer if bool(row.get("strict_target_version_match"))
        ),
        "sqlancer_version_mismatch_run_count": sum(
            1 for row in sqlancer if row.get("strict_target_version_match") is False
        ),
        "sqlancer_reported_failure_count": sum(1 for row in sqlancer if bool(row.get("reported_failure"))),
        "sqlancer_total_queries": sum(int(row.get("queries", 0) or 0) for row in sqlancer),
        "sqlancer_total_databases": sum(int(row.get("databases", 0) or 0) for row in sqlancer),
        "datadiff_run_count": len(datadiff),
        "datadiff_total_cases": sum(int(row.get("cases", 0) or 0) for row in datadiff),
        "datadiff_candidate_bug_cases": sum(int(row.get("candidate_bug_cases", 0) or 0) for row in datadiff),
        "datadiff_confirmed_bug_family_entries": sum(
            len(split_nonempty(row.get("confirmed_bug_families", ""))) for row in datadiff
        ),
    }


def write_outputs(output_base: Path, payload: dict[str, object]) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_base.with_suffix(".json")
    csv_path = output_base.with_suffix(".csv")
    md_path = output_base.with_suffix(".md")
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rows = list(payload.get("rows", []) or [])
    fieldnames = [
        "tool",
        "role",
        "suite",
        "target",
        "oracle",
        "seed",
        "status",
        "returncode",
        "duration_seconds",
        "queries",
        "databases",
        "reported_failure",
        "strict_target_version_match",
        "datadiff_duckdb_version",
        "sqlancer_duckdb_version",
        "cases",
        "candidate_bug_cases",
        "candidate_bug_families",
        "confirmed_bug_families",
        "findings",
        "throughput_queries_s",
        "throughput_cases_s",
        "manifest",
        "log_file",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    md_path.write_text(render_markdown(payload), encoding="utf-8")


def resolve_output_base(path: Path) -> Path:
    if path.is_absolute():
        return path
    if path.parent != Path("."):
        return PROJECT_ROOT / path
    return DEFAULT_OUTPUT_DIR / path


def render_markdown(payload: dict[str, object]) -> str:
    summary = payload.get("summary", {}) if isinstance(payload.get("summary"), dict) else {}
    rows = payload.get("rows", []) if isinstance(payload.get("rows"), list) else []
    lines = [
        "# SQLancer Fair Comparison Summary",
        "",
        f"- Generated at: `{payload.get('generated_at', '')}`",
        f"- Scope: {payload.get('scope', '')}",
        f"- SQLancer runs: `{summary.get('sqlancer_successful_run_count', 0)}` / `{summary.get('sqlancer_run_count', 0)}` successful",
        f"- SQLancer strict DuckDB-version runs: `{summary.get('sqlancer_strict_target_match_run_count', 0)}` / `{summary.get('sqlancer_run_count', 0)}`",
        f"- SQLancer reported failure signals: `{summary.get('sqlancer_reported_failure_count', 0)}`",
        f"- SQLancer total queries: `{summary.get('sqlancer_total_queries', 0)}`",
        f"- DataDiffFuzz rows: `{summary.get('datadiff_run_count', 0)}`; cases: `{summary.get('datadiff_total_cases', 0)}`; candidate cases: `{summary.get('datadiff_candidate_bug_cases', 0)}`",
        "",
        "Raw findings and candidate cases are support evidence only, not confirmed paper bug counts.",
        "The primary paper metric is independently confirmed unique latest-version bug families in the DuckDB SQL-overlap scope.",
        "",
        "Supporting metrics are reported to explain efficiency and triage workload: SQLancer queries/databases/throughput, DataDiffFuzz cases/candidates/throughput, reproducer quality, and semantic-class coverage.",
        "",
        "| tool | suite | oracle | seed | strict version | duration_s | queries | cases | candidate cases | throughput |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        throughput = row.get("throughput_queries_s") or row.get("throughput_cases_s") or ""
        lines.append(
            "| {tool} | {suite} | {oracle} | {seed} | {strict} | {duration} | {queries} | {cases} | {candidates} | {throughput} |".format(
                tool=row.get("tool", ""),
                suite=row.get("suite", ""),
                oracle=row.get("oracle", ""),
                seed=row.get("seed", ""),
                strict=_format_strict_version(row),
                duration=row.get("duration_seconds", ""),
                queries=row.get("queries", ""),
                cases=row.get("cases", ""),
                candidates=row.get("candidate_bug_cases", ""),
                throughput=throughput,
            )
        )
    lines.append("")
    return "\n".join(lines)


def _format_strict_version(row: dict[str, object]) -> str:
    value = row.get("strict_target_version_match", "")
    if value == "":
        return ""
    return "yes" if bool(value) else "no"


def run_experiment_summary(path: Path) -> None:
    subprocess.run(
        [
            str(PROJECT_ROOT / ".venv" / "bin" / "datadiff"),
            "experiment-summary",
            "--manifest",
            str(path),
            "--refresh",
        ],
        cwd=str(PROJECT_ROOT),
        check=True,
    )


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"expected JSON object: {path}")
    return data


def int_float(value: object) -> float:
    if value in (None, ""):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def split_nonempty(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [part.strip() for part in str(value or "").replace(";", ",").split(",") if part.strip()]


def row_returncode(row: dict[str, object]) -> int:
    value = row.get("returncode", 1)
    if value in (None, ""):
        return 1
    return int(value)


def project_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize SQLancer and DataDiffFuzz scope-limited fair-comparison manifests."
    )
    parser.add_argument("--sqlancer-manifest", action="append", default=[])
    parser.add_argument("--datadiff-manifest", action="append", default=[])
    parser.add_argument("--output-base", default=f"sqlancer-fair-comparison-summary-{utc_timestamp()}")
    parser.add_argument("--refresh-datadiff-summary", action="store_true")
    return parser.parse_args()


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


if __name__ == "__main__":
    sys.exit(main())
