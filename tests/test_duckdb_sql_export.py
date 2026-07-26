from __future__ import annotations

import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path

from datadiff.backends.duckdb_backend import DuckDBBackend, _duckdb_threads
from datadiff.dsl import Case, ColumnSpec, Program, TableData
from datadiff.duckdb_sql_export import render_duckdb_case_sql


def test_duckdb_thread_limit_defaults_and_environment_override(monkeypatch):
    monkeypatch.delenv("DATADIFF_DUCKDB_THREADS", raising=False)
    assert _duckdb_threads() == 1

    monkeypatch.setenv("DATADIFF_DUCKDB_THREADS", "4")
    assert _duckdb_threads() == 4

    monkeypatch.setenv("DATADIFF_DUCKDB_THREADS", "invalid")
    assert _duckdb_threads() == 1


def test_render_duckdb_case_sql_executes_like_duckdb_backend():
    case = Case(
        "case-sql-export",
        1,
        [
            TableData(
                "t0",
                [
                    ColumnSpec("id", "int", nullable=False),
                    ColumnSpec("g", "str"),
                    ColumnSpec("x", "int"),
                ],
                [
                    {"id": 1, "g": "a", "x": 1},
                    {"id": 2, "g": "a", "x": 2},
                    {"id": 3, "g": "b", "x": None},
                ],
            ),
            TableData(
                "t1",
                [ColumnSpec("id", "int", nullable=False), ColumnSpec("j", "int")],
                [{"id": 1, "j": 10}, {"id": 2, "j": -1}, {"id": 3, "j": 7}],
            ),
        ],
        Program(
            "program-sql-export",
            1,
            [
                {"op": "join", "table": "t1", "how": "inner", "left_on": "id", "right_on": "id"},
                {"op": "filter", "column": "j", "cmp": ">=", "value": 0},
                {"op": "mutate", "column": "m_0", "expr": {"kind": "add_const", "source": "x", "value": 1}},
                {
                    "op": "groupby",
                    "keys": ["g"],
                    "aggs": [
                        {"column": "m_0", "func": "sum", "as": "sum_m_0"},
                        {"column": "j", "func": "count", "as": "count_j"},
                    ],
                },
                {"op": "sort", "columns": ["g"], "ascending": True},
            ],
        ),
    )

    sql = render_duckdb_case_sql(case)
    direct = DuckDBBackend().run(case.tables, case.program)
    exported_rows = _execute_duckdb_sql(sql)

    assert direct.status == "ok"
    direct_rows = _backend_result_rows(direct.data)
    exported_rows = [tuple(_normalize_cell(cell) for cell in row) for row in exported_rows]
    assert exported_rows == direct_rows
    assert "WITH" in sql
    assert 'CREATE TABLE "t0"' in sql


def test_export_duckdb_queue_sql_reproducers_writes_one_passing_row_per_family(tmp_path):
    module = _script_module()
    queue = tmp_path / "queue.json"
    validation = tmp_path / "validation.json"
    row = {
        "family": "groupby_aggregation@duckdb",
        "case_id": "case-1",
        "case": _queue_case().to_dict(),
        "verification_backends": ["duckdb", "sqlite"],
        "expected_finding_keys": ["semantic_output_mismatch:groupby_aggregation@duckdb:value"],
        "output_columns": ["sum_x"],
        "duckdb_rows": [[3]],
        "reference_rows": {"sqlite": [[3]]},
        "witness_plan": _witness_plan(),
        "source_artifact": "rows/family.jsonl",
        "source_row_index": 3,
    }
    queue.write_text(
        json.dumps({"schema_version": "reproducer-queue-v1", "families": {"groupby_aggregation@duckdb": [row]}}),
        encoding="utf-8",
    )
    validation.write_text(
        json.dumps({"results": [{"family": row["family"], "case_id": row["case_id"], "status": "passed"}]}),
        encoding="utf-8",
    )
    out = tmp_path / "sql"

    rc = module.run_with_args(
        Namespace(
            queue=str(queue),
            validation=str(validation),
            per_family_limit=1,
            output_dir=str(out),
            allow_skips=False,
            execute_sql=True,
        )
    )

    assert rc == 0
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["summary"]["exported_count"] == 1
    assert manifest["summary"]["local_sql_execution_passed_count"] == 1
    assert manifest["exports"][0]["local_sql_execution"]["status"] == "ok"
    assert manifest["exports"][0]["duckdb_rows"] == [[3]]
    assert manifest["exports"][0]["reference_rows"] == {"sqlite": [[3]]}
    assert manifest["exports"][0]["witness_plan"]["contract"]["kind"] == "aggregate_value"
    sql_path = Path(manifest["exports"][0]["sql_path"])
    if not sql_path.is_absolute():
        sql_path = Path(__file__).resolve().parents[1] / sql_path
    assert sql_path.is_file()
    text = sql_path.read_text(encoding="utf-8")
    assert "-- family: groupby_aggregation@duckdb" in text
    assert "-- witness: kind=aggregate_value" in text


def test_validate_duckdb_sql_export_treats_empty_stored_rows_as_reproducible(tmp_path):
    module = _validate_module()
    sql_path = tmp_path / "empty.sql"
    sql_path.write_text("SELECT 1 AS x WHERE FALSE;\n", encoding="utf-8")
    export = {
        "family": "grouped_topk_null_sort_key@duckdb",
        "case_id": "case-empty-duckdb-output",
        "sql_path": str(sql_path),
        "output_columns": ["x"],
        "duckdb_rows": [],
        "reference_rows": {"sqlite": [[1]]},
    }

    result = module.validate_export(export)

    assert result["status"] == "ok"
    assert result["has_stored_suspicious_rows"] is True
    assert result["current_matches_stored_suspicious"] is True
    assert result["current_differs_from_reference"] is True
    assert result["verdict"] == "current_reproduces_stored_suspicious_output"


def _queue_case() -> Case:
    return Case(
        "case-1",
        1,
        [TableData("t0", [ColumnSpec("x", "int")], [{"x": 1}, {"x": 2}])],
        Program("program-1", 1, [{"op": "aggregate", "aggs": [{"column": "x", "func": "sum", "as": "sum_x"}]}]),
    )


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


def _execute_duckdb_sql(sql: str) -> list[tuple[object, ...]]:
    import duckdb

    con = duckdb.connect(database=":memory:")
    try:
        return con.execute(sql).fetchall()
    finally:
        con.close()


def _normalize_cell(value):
    try:
        import math

        if isinstance(value, float) and math.isnan(value):
            return None
    except TypeError:
        pass
    return value


def _backend_result_rows(data):
    if hasattr(data, "row_values"):
        return [tuple(_normalize_cell(cell) for cell in row) for row in data.row_values]
    return [tuple(_normalize_cell(cell) for cell in row) for row in data.itertuples(index=False, name=None)]


def _script_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "export_duckdb_queue_sql_reproducers.py"
    spec = importlib.util.spec_from_file_location("export_duckdb_queue_sql_reproducers", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _validate_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "validate_duckdb_sql_exports.py"
    spec = importlib.util.spec_from_file_location("validate_duckdb_sql_exports", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
