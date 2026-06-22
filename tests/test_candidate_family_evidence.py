from __future__ import annotations

import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "extract_candidate_family_evidence.py"
    spec = importlib.util.spec_from_file_location("extract_candidate_family_evidence", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_row_candidate_families_uses_root_and_sorted_suspicious_backends():
    module = _module()
    row = {
        "findings": [
            {
                "triage_verdict": "candidate_implementation_bug",
                "root_cause": "groupby_aggregation",
                "suspicious_backends": ["sqlite", "duckdb"],
            },
            {
                "triage_verdict": "generator_false_positive",
                "root_cause": "groupby_aggregation",
                "suspicious_backends": ["duckdb"],
            },
        ]
    }

    assert module.row_candidate_families(row) == {"groupby_aggregation@duckdb,sqlite": 1}


def test_candidate_run_records_filters_runs_by_rewardable_family():
    module = _module()
    final = {
        "runs": [
            {
                "run_file": "runs/run-a.jsonl.gz",
                "rewardable_candidate_families": {"union_all_row_append@duckdb": 2},
                "target_suite": "embedded_sql",
                "preset": "live_duckdb_issue_focus",
                "evidence_mode": "live",
                "seed": 1,
                "cases": 10,
                "elapsed_s": 1.5,
            },
            {
                "run_file": "runs/run-b.jsonl.gz",
                "rewardable_candidate_families": {"union_all_row_append@duckdb": 1},
                "evidence_mode": "validation",
            },
        ]
    }

    records = module.candidate_run_records(final, ["union_all_row_append@duckdb"])

    assert len(records) == 1
    assert records[0]["run_file"] == "runs/run-a.jsonl.gz"
    assert records[0]["families"] == ["union_all_row_append@duckdb"]


def test_row_evidence_contains_reproduce_hint_and_operation_sequence(tmp_path):
    module = _module()
    run_file = tmp_path / "run.jsonl"
    row = {
        "case": {
            "case_id": "case-1",
            "seed": 7,
            "program": {
                "operations": [
                    {"op": "union_all"},
                    {"op": "sort"},
                ]
            },
        },
        "status": "bug",
        "findings": [
            {
                "triage_verdict": "candidate_implementation_bug",
                "root_cause": "union_all_row_append",
                "suspicious_backends": ["duckdb"],
                "signature": "sig",
            }
        ],
    }

    evidence = module.row_evidence(
        row,
        family="union_all_row_append@duckdb",
        run_file=run_file,
        run_context={"target_suite": "embedded_sql", "preset": "p", "evidence_mode": "live", "seed": 1},
        row_index=3,
    )

    assert evidence["case_id"] == "case-1"
    assert evidence["operation_sequence"] == ["union_all", "sort"]
    assert evidence["finding_count"] == 1
    assert "row_index=3" in evidence["source_hint"]


def test_run_with_args_records_selection_metadata(tmp_path):
    module = _module()
    final = tmp_path / "final.json"
    final.write_text(json.dumps({"runs": []}), encoding="utf-8")

    rc = module.run_with_args(
        Namespace(
            final_readiness=str(final),
            family=["union_all_row_append@duckdb"],
            family_file=None,
            evidence_mode="live",
            per_family_limit=2,
            output_base=str(tmp_path / "evidence"),
            write_row_artifacts=False,
            row_output_dir="",
        )
    )

    assert rc == 0
    payload = json.loads((tmp_path / "evidence.json").read_text(encoding="utf-8"))
    assert payload["selection"] == {"evidence_mode": "live", "per_family_limit": 2}
    assert "Evidence mode: `live`" in (tmp_path / "evidence.md").read_text(encoding="utf-8")


def test_write_row_artifacts_extracts_original_run_rows(tmp_path, monkeypatch):
    module = _module()
    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    run_file = tmp_path / "runs" / "run.jsonl"
    run_file.parent.mkdir()
    original_row = {
        "case": {"case_id": "case-1"},
        "findings": [
            {
                "triage_verdict": "candidate_implementation_bug",
                "root_cause": "union_all_row_append",
                "suspicious_backends": ["duckdb"],
            }
        ],
    }
    run_file.write_text(json.dumps({"case": {"case_id": "skip"}}) + "\n" + json.dumps(original_row) + "\n", encoding="utf-8")

    result = module.write_row_artifacts(
        {
            "union_all_row_append@duckdb": [
                {
                    "run_file": "runs/run.jsonl",
                    "row_index": 1,
                    "case_id": "case-1",
                    "evidence_mode": "live",
                    "target_suite": "embedded_sql",
                    "preset": "p",
                }
            ]
        },
        row_output_dir=tmp_path / "rows",
    )

    assert result["family_count"] == 1
    assert result["row_count"] == 1
    artifact_path = tmp_path / "rows" / "union_all_row_append-duckdb.jsonl"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact["source"]["row_index"] == 1
    assert artifact["row"] == original_row
    manifest = json.loads((tmp_path / "rows" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["families"]["union_all_row_append@duckdb"]["rows"] == 1
