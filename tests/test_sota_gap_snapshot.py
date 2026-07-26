from __future__ import annotations

import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "summarize_sota_gap_snapshot.py"
    spec = importlib.util.spec_from_file_location("summarize_sota_gap_snapshot", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_snapshot_marks_strict_comparison_incomplete_without_required_manifests(tmp_path, monkeypatch):
    module = _module()
    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    final_readiness = tmp_path / "final.json"
    issue_bundle = tmp_path / "issue-bundle.json"
    pilot = tmp_path / "pilot.json"
    final_readiness.write_text(json.dumps(_final_readiness()), encoding="utf-8")
    issue_bundle.write_text(json.dumps({"summary": {"selected_count": 2, "skipped_count": 2}}), encoding="utf-8")
    pilot.write_text(json.dumps(_sqlancer_manifest(strict=True, queries=1200, elapsed=12.0)), encoding="utf-8")

    rc = module.run_with_args(
        Namespace(
            final_readiness=str(final_readiness),
            issue_bundle=str(issue_bundle),
            latest_confirmations="",
            sqlancer_pilot_manifest=[str(pilot)],
            sqlancer_strict_glob="missing-*.json",
            output_base=str(tmp_path / "snapshot"),
        )
    )

    assert rc == 0
    payload = json.loads((tmp_path / "snapshot.json").read_text(encoding="utf-8"))
    assert payload["datadiff"]["confirmed_latest_version_bug_families"] == 9
    assert payload["datadiff"]["duckdb_issue_ready_selected_count"] == 2
    assert payload["sqlancer_pilot"]["throughput_queries_s"] == 100.0
    assert payload["sqlancer_strict"]["status"] == "incomplete"
    assert payload["gap"]["strict_comparison_ready"] is False
    assert "not produced all required manifests" in payload["gap"]["strict_comparison_blocker"]
    assert payload["gap"]["sqlancer_pilot_query_throughput_vs_datadiff_case_throughput_ratio"] == 50.0
    markdown = (tmp_path / "snapshot.md").read_text(encoding="utf-8")
    assert "Strict comparison ready: `false`" in markdown


def test_snapshot_marks_strict_complete_when_all_six_strict_runs_exist(tmp_path, monkeypatch):
    module = _module()
    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    final_readiness = tmp_path / "final.json"
    issue_bundle = tmp_path / "issue-bundle.json"
    strict_dir = tmp_path / "reports" / "external-baselines"
    strict_dir.mkdir(parents=True)
    final_readiness.write_text(json.dumps(_final_readiness()), encoding="utf-8")
    issue_bundle.write_text(json.dumps({"summary": {"selected_count": 1, "skipped_count": 0}}), encoding="utf-8")
    for name in ["qp", "norec"]:
        (strict_dir / f"strict-{name}.json").write_text(
            json.dumps(_sqlancer_manifest(strict=True, queries=600, elapsed=6.0, run_count=3)),
            encoding="utf-8",
        )

    rc = module.run_with_args(
        Namespace(
            final_readiness=str(final_readiness),
            issue_bundle=str(issue_bundle),
            latest_confirmations="",
            sqlancer_pilot_manifest=[],
            sqlancer_strict_glob="reports/external-baselines/strict-*.json",
            output_base=str(tmp_path / "snapshot"),
        )
    )

    assert rc == 0
    payload = json.loads((tmp_path / "snapshot.json").read_text(encoding="utf-8"))
    assert payload["sqlancer_strict"]["run_count"] == 6
    assert payload["sqlancer_strict"]["strict_target_match_run_count"] == 6
    assert payload["sqlancer_strict"]["status"] == "complete"
    assert payload["gap"]["strict_comparison_ready"] is True


def test_snapshot_counts_confirmation_artifact_when_readiness_summary_lacks_count(tmp_path, monkeypatch):
    module = _module()
    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    final_readiness = tmp_path / "final.json"
    issue_bundle = tmp_path / "issue-bundle.json"
    latest_confirmations = tmp_path / "latest-confirmations.json"
    final_readiness.write_text(
        json.dumps(
            {
                "ready": True,
                "summary": {
                    "rewardable_live_candidate_families": {
                        "x@duckdb": 1,
                        "y@datafusion": 2,
                        "z@polars": 3,
                    },
                    "live_runs": 2,
                    "total_live_cases": 10,
                    "total_live_elapsed_s": 5.0,
                },
                "icse_experiment_quality": {
                    "dimensions": {
                        "throughput": {
                            "evidence": {"avg_throughput_cases_s": 2.0},
                        }
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    issue_bundle.write_text(json.dumps({"summary": {"selected_count": 0}}), encoding="utf-8")
    latest_confirmations.write_text(
        json.dumps(
            {
                "confirmations": [
                    {"family": "a@duckdb", "discovery_credit": "datadiff_submitted"},
                    {"family": "b@datafusion", "discovery_credit": "datadiff_submitted"},
                    {"family": "a@duckdb", "discovery_credit": "datadiff_submitted"},
                    {"family": "similar@pandas", "discovery_credit": "similar_existing"},
                ]
            }
        ),
        encoding="utf-8",
    )

    rc = module.run_with_args(
        Namespace(
            final_readiness=str(final_readiness),
            issue_bundle=str(issue_bundle),
            latest_confirmations=str(latest_confirmations),
            sqlancer_pilot_manifest=[],
            sqlancer_strict_glob="missing-*.json",
            output_base=str(tmp_path / "snapshot"),
        )
    )

    assert rc == 0
    payload = json.loads((tmp_path / "snapshot.json").read_text(encoding="utf-8"))
    assert payload["datadiff"]["confirmed_latest_version_bug_families"] == 2
    assert payload["datadiff"]["rewardable_live_candidate_families"] == 3


def _final_readiness() -> dict[str, object]:
    return {
        "ready": True,
        "summary": {
            "confirmed_latest_version_bug_families": 9,
            "rewardable_live_candidate_families": {
                f"candidate_{idx}@duckdb": 1 for idx in range(277)
            },
            "live_runs": 1580,
            "total_live_cases": 184459,
            "total_live_elapsed_s": 950434.48,
        },
        "icse_experiment_quality": {
            "dimensions": {
                "throughput": {
                    "evidence": {
                        "avg_throughput_cases_s": 2.0,
                    }
                }
            },
            "thresholds": {"target_throughput_cases_s": 5.0},
        },
    }


def _sqlancer_manifest(*, strict: bool, queries: int, elapsed: float, run_count: int = 1) -> dict[str, object]:
    return {
        "strict_target_version_match": strict,
        "runs": [
            {
                "returncode": 0,
                "elapsed_seconds": elapsed,
                "stats": {"summary_queries": queries, "summary_databases": 1},
                "failure_signal": {"has_signal": False},
            }
            for _ in range(run_count)
        ],
    }
