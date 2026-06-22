from __future__ import annotations

import csv
import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "summarize_sqlancer_fair_comparison.py"
    spec = importlib.util.spec_from_file_location("summarize_sqlancer_fair_comparison", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_summary_counts_zero_returncode_as_success():
    module = _module()

    summary = module.build_summary(
        [
            {
                "tool": "sqlancer",
                "returncode": 0,
                "queries": 6000,
                "databases": 1,
            }
        ]
    )

    assert summary["sqlancer_successful_run_count"] == 1
    assert summary["sqlancer_total_queries"] == 6000
    assert summary["sqlancer_total_databases"] == 1


def test_run_with_args_writes_combined_outputs(tmp_path):
    module = _module()
    sqlancer_manifest = tmp_path / "sqlancer.json"
    sqlancer_manifest.write_text(
        json.dumps(
            {
                "strict_target_version_match": True,
                "target_versions": {
                    "normalized": {
                        "datadiff_duckdb": "1.5.3",
                        "sqlancer_duckdb_jdbc": "1.5.3",
                    }
                },
                "runs": [
                    {
                        "returncode": 0,
                        "status": "completed",
                        "elapsed_seconds": 20.0,
                        "log_file": "logs/sqlancer.log",
                        "failure_signal": {"has_signal": True, "matched_lines": ["--java.lang.AssertionError"]},
                        "spec": {
                            "suite": "duckdb-norec",
                            "dbms": "duckdb",
                            "oracle": "NOREC",
                            "seed": 1,
                        },
                        "stats": {
                            "summary_queries": 11000,
                            "summary_databases": 1,
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    datadiff_manifest = tmp_path / "experiment-demo.json"
    datadiff_manifest.write_text(json.dumps({"runs": []}), encoding="utf-8")
    reports_dir = Path(__file__).resolve().parents[1] / "reports"
    aggregate_csv = reports_dir / "experiment-summary-experiment-demo-aggregates.csv"
    aggregate_csv.write_text(
        "target_suite,backends,oracle_profile,cases,candidate_bug_cases,candidate_bug_families,top_candidate_bug_families,findings,avg_throughput_cases_s\n"
        "embedded_sql,\"duckdb,sqlite\",differential,23,4,1,topk_filter_pushdown@duckdb:4,4,1.15\n",
        encoding="utf-8",
    )
    try:
        rc = module.run_with_args(
            Namespace(
                sqlancer_manifest=[str(sqlancer_manifest)],
                datadiff_manifest=[str(datadiff_manifest)],
                output_base=str(tmp_path / "summary"),
                refresh_datadiff_summary=False,
            )
        )
    finally:
        aggregate_csv.unlink(missing_ok=True)

    assert rc == 0
    payload = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert payload["summary"]["sqlancer_successful_run_count"] == 1
    assert payload["summary"]["sqlancer_strict_target_match_run_count"] == 1
    assert payload["summary"]["sqlancer_reported_failure_count"] == 1
    assert payload["summary"]["datadiff_total_cases"] == 23
    with (tmp_path / "summary.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["tool"] for row in rows] == ["sqlancer", "datadiff"]
    assert rows[0]["strict_target_version_match"] == "True"
    assert rows[0]["datadiff_duckdb_version"] == "1.5.3"
    assert rows[0]["sqlancer_duckdb_version"] == "1.5.3"
    assert rows[1]["candidate_bug_cases"] == "4"
    assert rows[1]["candidate_bug_families"] == "topk_filter_pushdown@duckdb:4"
    markdown = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "SQLancer Fair Comparison Summary" in markdown
    assert "SQLancer strict DuckDB-version runs" in markdown


def test_summary_marks_version_mismatched_sqlancer_rows_as_support_only(tmp_path):
    module = _module()
    sqlancer_manifest = tmp_path / "sqlancer.json"
    sqlancer_manifest.write_text(
        json.dumps(
            {
                "strict_target_version_match": False,
                "target_versions": {
                    "normalized": {
                        "datadiff_duckdb": "1.5.3",
                        "sqlancer_duckdb_jdbc": "1.3.0",
                    }
                },
                "runs": [
                    {
                        "returncode": 0,
                        "status": "completed",
                        "elapsed_seconds": 10.0,
                        "spec": {"suite": "duckdb-norec", "dbms": "duckdb", "oracle": "NOREC", "seed": 1},
                        "stats": {"summary_queries": 100, "summary_databases": 1},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    rows = module.sqlancer_rows(sqlancer_manifest)
    summary = module.build_summary(rows)

    assert rows[0]["strict_target_version_match"] is False
    assert rows[0]["sqlancer_duckdb_version"] == "1.3.0"
    assert summary["sqlancer_strict_target_match_run_count"] == 0
    assert summary["sqlancer_version_mismatch_run_count"] == 1
