from __future__ import annotations

import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "plan_reproducer_queue.py"
    spec = importlib.util.spec_from_file_location("plan_reproducer_queue", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_queue_item_extracts_reproducer_priorities(tmp_path, monkeypatch):
    module = _module()
    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    artifact_path = tmp_path / "rows" / "family.jsonl"
    artifact_path.parent.mkdir()
    artifact = {
        "family": "union_all_row_append@duckdb",
        "source": {
            "run_file": "runs/run.jsonl.gz",
            "row_index": 7,
            "case_id": "case-1",
            "target_suite": "embedded_sql",
            "preset": "live",
        },
        "row": {
            "case": {
                "case_id": "case-1",
                "seed": 1,
                "tables": [{"name": "t", "rows": [{"x": 1}]}],
                "program": {
                    "operations": [{"op": "union_all"}, {"op": "aggregate"}],
                },
            },
            "candidate_recheck": {"attempts": 2, "reproduced_keys": ["k"], "non_reproduced_keys": []},
            "disagreement_descriptor": {"mismatch_class": "value"},
            "normalized": {
                "duckdb": {"status": "ok", "columns": ["x"], "rows": [[1]]},
                "sqlite": {"status": "ok", "columns": ["x"], "rows": [[2]]},
                "pandas": {"status": "ok", "columns": ["x"], "rows": [[2]]},
            },
            "findings": [
                {
                    "kind": "semantic_output_mismatch",
                    "triage_verdict": "candidate_implementation_bug",
                    "root_cause": "union_all_row_append",
                    "suspicious_backends": ["duckdb"],
                    "signature": "sig",
                    "triage_evidence": "reference agrees",
                }
            ],
        },
    }
    artifact_path.write_text(json.dumps(artifact) + "\n", encoding="utf-8")
    manifest = {
        "families": {
            "union_all_row_append@duckdb": {
                "path": "rows/family.jsonl",
                "rows": 1,
            }
        }
    }

    queue = module.collect_candidates(manifest, per_family_limit=1)

    item = queue["union_all_row_append@duckdb"][0]
    assert item["case_id"] == "case-1"
    assert item["case"]["case_id"] == "case-1"
    assert item["operation_count"] == 2
    assert item["reference_backends"] == ["pandas", "sqlite"]
    assert item["verification_backends"] == ["duckdb", "sqlite", "pandas"]
    assert item["expected_finding_keys"] == [
        "semantic_output_mismatch:union_all_row_append@duckdb"
    ]
    assert item["recheck_reproduced"] is True
    assert item["duckdb_rows"] == [[1]]
    assert item["reference_rows"] == {"pandas": [[2]], "sqlite": [[2]]}
    assert item["witness_plan"]["status"] == "available"
    assert item["witness_plan"]["contract"]["kind"] == "row_containment"
    assert item["witness_plan"]["contract"]["row"] == {"x": 2}
    assert item["witness_plan"]["failing_suspicious_backends"] == ["duckdb"]


def test_run_with_args_writes_queue_outputs(tmp_path, monkeypatch):
    module = _module()
    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    artifact_path = tmp_path / "rows" / "family.jsonl"
    artifact_path.parent.mkdir()
    artifact_path.write_text(
        json.dumps(
            {
                "family": "ordering_or_limit@duckdb",
                "source": {"row_index": 1, "case_id": "case-2"},
                "row": {
                    "case": {"case_id": "case-2", "seed": 2, "tables": [], "program": {"operations": [{"op": "limit"}]}},
                    "candidate_recheck": {"attempts": 1, "reproduced_keys": ["k"], "non_reproduced_keys": []},
                    "normalized": {
                        "duckdb": {"status": "ok", "columns": ["x"], "rows": [[1]]},
                        "sqlite": {"status": "ok", "columns": ["x"], "rows": [[2]]},
                    },
                    "findings": [
                        {
                            "kind": "semantic_output_mismatch",
                            "triage_verdict": "candidate_implementation_bug",
                            "root_cause": "ordering_or_limit",
                            "suspicious_backends": ["duckdb"],
                        }
                    ],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "rows" / "manifest.json"
    manifest_path.write_text(
        json.dumps({"families": {"ordering_or_limit@duckdb": {"path": "rows/family.jsonl", "rows": 1}}}),
        encoding="utf-8",
    )

    rc = module.run_with_args(
        Namespace(
            row_artifact_manifest=str(manifest_path),
            per_family_limit=1,
            output_base=str(tmp_path / "queue"),
        )
    )

    assert rc == 0
    payload = json.loads((tmp_path / "queue.json").read_text(encoding="utf-8"))
    assert payload["summary"]["candidate_count"] == 1
    assert payload["summary"]["witness_plan_available_count"] == 1
    assert "Reproducer Queue" in (tmp_path / "queue.md").read_text(encoding="utf-8")
    assert (tmp_path / "queue_reproduce.py").is_file()
    assert "evidence-queue-reproducer-results-v1" in (tmp_path / "queue_reproduce.py").read_text(encoding="utf-8")
