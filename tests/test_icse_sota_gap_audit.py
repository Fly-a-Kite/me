from __future__ import annotations

import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "build_icse_sota_gap_audit.py"
    spec = importlib.util.spec_from_file_location("build_icse_sota_gap_audit", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_icse_sota_gap_audit_reports_current_blockers(tmp_path, monkeypatch):
    module = _module()
    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    final_readiness = tmp_path / "final-readiness.json"
    latest_confirmations = tmp_path / "latest-confirmations.json"
    sota_snapshot = tmp_path / "sota-snapshot.json"
    triage_plan = tmp_path / "triage-plan.json"
    minimized = tmp_path / "minimized.json"
    issue_bundle = tmp_path / "issue-bundle.json"
    final_readiness.write_text(json.dumps(_final_readiness()), encoding="utf-8")
    latest_confirmations.write_text(json.dumps(_latest_confirmations(9)), encoding="utf-8")
    sota_snapshot.write_text(json.dumps(_sota_snapshot(strict_complete=False)), encoding="utf-8")
    triage_plan.write_text(
        json.dumps(
            {
                "target_confirmed_count": 20,
                "needed_confirmations": 11,
                "candidates": [{"family": f"candidate_{idx}@duckdb"} for idx in range(50)],
            }
        ),
        encoding="utf-8",
    )
    minimized.write_text(
        json.dumps(
            {
                "summary": {
                    "family_count": 4,
                    "target_reproduced_count": 4,
                    "local_sql_execution_passed_count": 4,
                    "native_sql_match_count": 2,
                    "native_sql_mismatch_count": 2,
                }
            }
        ),
        encoding="utf-8",
    )
    issue_bundle.write_text(json.dumps({"summary": {"selected_count": 2, "skipped_count": 2}}), encoding="utf-8")

    rc = module.run_with_args(
        Namespace(
            final_readiness=str(final_readiness),
            latest_confirmations=str(latest_confirmations),
            sota_snapshot=str(sota_snapshot),
            triage_plan=str(triage_plan),
            minimized_duckdb=str(minimized),
            duckdb_issue_bundle=str(issue_bundle),
            output_base=str(tmp_path / "audit"),
        )
    )

    assert rc == 0
    payload = json.loads((tmp_path / "audit.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == "icse-sota-gap-audit-v1"
    assert payload["metrics"]["latest_confirmed_families"] == 9
    assert payload["metrics"]["readiness_embedded_confirmed_families"] == 8
    assert payload["metrics"]["confirmation_registry_delta_vs_readiness"] == 1
    assert payload["metrics"]["target_20plus_gap"] == 11
    assert payload["metrics"]["sqlancer_strict_complete"] is False
    assert payload["metrics"]["duckdb_minimized_native_sql_mismatch_count"] == 2
    assert payload["readiness_verdict"]["final_readiness_green"] is True
    assert payload["readiness_verdict"]["high_probability_icse_ready"] is False
    assert "external_sota_strict_sqlancer" in payload["readiness_verdict"]["open_blockers"]
    defect_ids = {item["id"] for item in payload["code_and_pipeline_defects"]}
    assert "readiness_confirmation_registry_skew" in defect_ids
    markdown = (tmp_path / "audit.md").read_text(encoding="utf-8")
    assert "High-probability ICSE ready: `false`" in markdown
    assert "SQLancer strict runs | 0 / 6" in markdown


def test_icse_sota_gap_audit_allows_complete_strict_status_without_strict_blocker(tmp_path, monkeypatch):
    module = _module()
    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    paths = {
        "final_readiness": tmp_path / "final-readiness.json",
        "latest_confirmations": tmp_path / "latest-confirmations.json",
        "sota_snapshot": tmp_path / "sota-snapshot.json",
        "triage_plan": tmp_path / "triage-plan.json",
        "minimized_duckdb": tmp_path / "minimized.json",
        "duckdb_issue_bundle": tmp_path / "issue-bundle.json",
    }
    paths["final_readiness"].write_text(json.dumps(_final_readiness(embedded_confirmed=12)), encoding="utf-8")
    paths["latest_confirmations"].write_text(json.dumps(_latest_confirmations(12)), encoding="utf-8")
    paths["sota_snapshot"].write_text(json.dumps(_sota_snapshot(strict_complete=True)), encoding="utf-8")
    paths["triage_plan"].write_text(json.dumps({"target_confirmed_count": 20, "needed_confirmations": 8, "candidates": []}), encoding="utf-8")
    paths["minimized_duckdb"].write_text(json.dumps({"summary": {"family_count": 2, "native_sql_mismatch_count": 0}}), encoding="utf-8")
    paths["duckdb_issue_bundle"].write_text(json.dumps({"summary": {"selected_count": 2, "skipped_count": 0}}), encoding="utf-8")

    payload = module.build_payload({name: json.loads(path.read_text(encoding="utf-8")) for name, path in paths.items()}, paths)

    assert payload["metrics"]["sqlancer_strict_complete"] is True
    strict_gap = next(row for row in payload["sota_gap"] if row["id"] == "external_sota_strict_sqlancer")
    assert strict_gap["status"] == "complete"
    assert payload["metrics"]["confirmation_registry_delta_vs_readiness"] == 0


def _final_readiness(*, embedded_confirmed: int = 8) -> dict[str, object]:
    return {
        "ready": True,
        "gates": [{"name": "run_log_scan", "passed": True}],
        "latest_confirmations": [{"family": f"embedded_{idx}@duckdb"} for idx in range(embedded_confirmed)],
        "summary": {
            "live_runs": 1580,
            "total_live_cases": 184459,
            "total_live_elapsed_s": 950434.48,
            "rewardable_live_candidate_families": {f"candidate_{idx}@duckdb": 1 for idx in range(277)},
        },
        "icse_experiment_quality": {
            "overall_score": 85.197,
            "grade": "B",
            "ready_for_icse_claim": False,
            "thresholds": {"target_throughput_cases_s": 5.0},
            "dimensions": {
                "throughput": {
                    "evidence": {"avg_throughput_cases_s": 1.2993210941477638},
                }
            },
        },
    }


def _latest_confirmations(count: int) -> dict[str, object]:
    backends = ["datafusion", "duckdb", "polars", "pyarrow"]
    return {
        "schema_version": 1,
        "confirmations": [
            {
                "family": f"confirmed_{idx}@{backends[idx % len(backends)]}",
                "suspicious_backends": [backends[idx % len(backends)]],
            }
            for idx in range(count)
        ],
    }


def _sota_snapshot(*, strict_complete: bool) -> dict[str, object]:
    strict_runs = 6 if strict_complete else 0
    return {
        "datadiff": {
            "live_runs": 1580,
            "live_cases": 184459,
            "live_elapsed_seconds": 950434.48,
            "rewardable_live_candidate_families": 277,
            "avg_throughput_cases_s": 1.2993210941477638,
            "target_throughput_cases_s": 5.0,
            "duckdb_issue_ready_selected_count": 2,
            "duckdb_issue_ready_skipped_count": 2,
        },
        "sqlancer_pilot": {
            "run_count": 6,
            "successful_run_count": 6,
            "total_queries": 12808000,
            "throughput_queries_s": 296.418637,
            "reported_failure_count": 0,
        },
        "sqlancer_strict": {
            "complete": strict_complete,
            "run_count": strict_runs,
            "expected_strict_run_count": 6,
            "reported_failure_count": 0,
            "confirmed_bug_family_count": 0,
        },
        "gap": {
            "strict_comparison_blocker": "" if strict_complete else "DuckDB 1.5.3 SQLancer strict fair run has not produced all required manifests.",
        },
    }
