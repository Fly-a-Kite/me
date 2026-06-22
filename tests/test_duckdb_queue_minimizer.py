from __future__ import annotations

import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path

from datadiff.dsl import Case, ColumnSpec, Program, TableData


def test_minimize_duckdb_queue_sql_reproducers_writes_manifest(tmp_path, monkeypatch):
    module = _module()
    queue = tmp_path / "queue.json"
    validation = tmp_path / "validation.json"
    case = Case(
        "case-1",
        1,
        [TableData("t0", [ColumnSpec("x", "int")], [{"x": 1}, {"x": 2}])],
        Program("program-1", 1, [{"op": "aggregate", "aggs": [{"column": "x", "func": "sum", "as": "sum_x"}]}]),
    )
    row = {
        "family": "groupby_aggregation@duckdb",
        "case_id": "case-1",
        "case": case.to_dict(),
        "config": {},
        "kind": "semantic_output_mismatch",
        "root_cause": "groupby_aggregation",
        "suspicious_backends": ["duckdb"],
        "verification_backends": ["duckdb", "sqlite"],
        "expected_finding_keys": ["semantic_output_mismatch:groupby_aggregation@duckdb:value"],
        "duckdb_rows": [[99]],
        "reference_rows": {"sqlite": [[98]]},
        "output_columns": ["sum_x"],
        "witness_plan": _witness_plan(),
    }
    queue.write_text(json.dumps({"families": {"groupby_aggregation@duckdb": [row]}}), encoding="utf-8")
    validation.write_text(json.dumps({"results": [{"family": row["family"], "case_id": row["case_id"], "status": "passed"}]}), encoding="utf-8")
    monkeypatch.setattr(module, "reduce_case", lambda case, **kwargs: case)
    monkeypatch.setattr(
        module,
        "run_loaded_case",
        lambda case, **kwargs: {
            "status": "bug_candidate",
            "normalized": {
                "duckdb": {
                    "status": "ok",
                    "columns": ["sum_x"],
                    "rows": [[3]],
                    "row_count": 1,
                },
                "sqlite": {
                    "status": "ok",
                    "columns": ["sum_x"],
                    "rows": [[4]],
                    "row_count": 1,
                },
            },
            "findings": [
                {
                    "kind": "semantic_output_mismatch",
                    "root_cause": "groupby_aggregation",
                    "suspicious_backends": ["duckdb"],
                    "mismatch_class": "value",
                }
            ],
        },
    )
    out = tmp_path / "out"

    rc = module.run_with_args(
        Namespace(
            queue=str(queue),
            validation=str(validation),
            per_family_limit=1,
            output_dir=str(out),
            allow_failures=False,
        )
    )

    assert rc == 0
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "duckdb-minimized-sql-reproducer-export-v3"
    assert manifest["summary"]["exported_count"] == 1
    assert manifest["summary"]["target_reproduced_count"] == 1
    assert manifest["summary"]["local_sql_execution_passed_count"] == 1
    assert manifest["summary"]["local_sql_execution_failed_count"] == 0
    assert manifest["summary"]["native_sql_match_count"] == 1
    assert manifest["summary"]["native_sql_mismatch_count"] == 0
    assert manifest["exports"][0]["duckdb_rows"] == [[3]]
    assert manifest["exports"][0]["reference_rows"] == {"sqlite": [[4]]}
    assert manifest["exports"][0]["source_evidence"]["duckdb_rows"] == [[99]]
    assert manifest["exports"][0]["reduced_rerun_evidence"]["backend_statuses"]["duckdb"]["row_count"] == 1
    assert manifest["exports"][0]["local_sql_execution"]["status"] == "ok"
    assert manifest["exports"][0]["local_sql_execution"]["normalized"]["rows"] == [[3]]
    assert manifest["exports"][0]["native_sql_matches_rerun_duckdb"] is True
    assert manifest["exports"][0]["witness_plan"]["contract"]["aggregate"] == "sum_x"
    assert manifest["exports"][0]["witness_plan"]["contract"]["expected"] == 4
    assert manifest["exports"][0]["source_witness_plan"]["contract"]["aggregate"] == "sum_x"
    assert Path(manifest["exports"][0]["sql_path"]).is_file()
    assert "-- witness: kind=aggregate_value" in Path(manifest["exports"][0]["sql_path"]).read_text(encoding="utf-8")
    assert Path(manifest["exports"][0]["case_path"]).is_file()


def test_native_sql_match_uses_strict_json_equality():
    module = _module()
    assert not module.native_sql_matches_rerun_duckdb(
        {
            "status": "ok",
            "normalized": {
                "status": "ok",
                "columns": ["sum_x"],
                "rows": [[9007199254740992]],
            },
        },
        {
            "output_columns": ["sum_x"],
            "duckdb_rows": [[9007199254740992.0]],
        },
    )


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "minimize_duckdb_queue_sql_reproducers.py"
    spec = importlib.util.spec_from_file_location("minimize_duckdb_queue_sql_reproducers", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _witness_plan() -> dict[str, object]:
    return {
        "schema_version": "datadiff-reference-witness-plan-v1",
        "status": "available",
        "contract": {
            "schema_version": "datadiff-witness-contract-v1",
            "kind": "aggregate_value",
            "group_key": {},
            "aggregate": "sum_x",
            "expected": 3,
            "backends": ["duckdb", "sqlite"],
            "reason": "reference backends agree on aggregate witness value and suspicious backend violates it",
            "source": "reference_consensus",
        },
        "satisfied_reference_backends": ["sqlite"],
        "failing_suspicious_backends": ["duckdb"],
    }
