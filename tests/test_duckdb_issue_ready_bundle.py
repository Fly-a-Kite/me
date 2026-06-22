from __future__ import annotations

import importlib.util
import json
import sys
import tarfile
from argparse import Namespace
from pathlib import Path


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "build_duckdb_issue_ready_bundle.py"
    spec = importlib.util.spec_from_file_location("build_duckdb_issue_ready_bundle", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_select_issue_ready_exports_requires_strict_native_sql_match():
    module = _module()
    ready = _export("ordering_or_limit@duckdb", native_match=True)
    mismatch = _export("union_all_row_append@duckdb", native_match=False)
    manifest = {"exports": [ready, mismatch]}

    selected, skipped = module.select_issue_ready_exports(manifest, include_families=[])

    assert [item["family"] for item in selected] == ["ordering_or_limit@duckdb"]
    assert skipped == [
        {
            "family": "union_all_row_append@duckdb",
            "case_id": "case-union_all_row_append",
            "reason": "native_sql_output_does_not_match_reduced_duckdb_rerun",
            "native_sql_matches_rerun_duckdb": False,
            "target_reproduced": True,
            "local_sql_status": "ok",
        }
    ]


def test_run_with_args_writes_issue_ready_bundle_and_tarball(tmp_path, monkeypatch):
    module = _module()
    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    ready = _materialized_export(
        tmp_path,
        family="ordering_or_limit@duckdb",
        native_match=True,
        witness_kind="row_containment",
    )
    mismatch = _materialized_export(
        tmp_path,
        family="union_all_row_append@duckdb",
        native_match=False,
        witness_kind="aggregate_value",
    )
    manifest_path = tmp_path / "minimized.json"
    manifest_path.write_text(
        json.dumps({"schema_version": "duckdb-minimized-sql-reproducer-export-v3", "exports": [ready, mismatch]}),
        encoding="utf-8",
    )
    out = tmp_path / "bundle"
    tarball = tmp_path / "bundle.tar.gz"

    rc = module.run_with_args(
        Namespace(
            minimized_manifest=str(manifest_path),
            output_dir=str(out),
            tarball=str(tarball),
            family=[],
            allow_empty=False,
        )
    )

    assert rc == 0
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "duckdb-issue-ready-bundle-v1"
    assert manifest["summary"]["selected_count"] == 1
    assert manifest["summary"]["skipped_count"] == 1
    issue = manifest["issues"][0]
    assert issue["family"] == "ordering_or_limit@duckdb"
    assert issue["witness_summary"].startswith("kind=row_containment")
    issue_dir = tmp_path / issue["issue_dir"]
    assert (issue_dir / "reproducer.sql").read_text(encoding="utf-8").startswith("-- sql")
    draft = (issue_dir / "upstream_issue_draft.md").read_text(encoding="utf-8")
    assert "Strict native SQL match to DuckDB rerun: `true`" in draft
    assert "candidate issue report" in draft
    assert tarball.is_file()
    assert tarball.with_suffix(tarball.suffix + ".sha256").is_file()
    with tarfile.open(tarball, "r:gz") as tar:
        names = tar.getnames()
    assert "bundle/manifest.json" in names
    assert any(name.endswith("/upstream_issue_draft.md") for name in names)
    assert any(name.endswith("/reproducer.sql") for name in names)


def test_run_with_args_fails_when_no_strict_candidates_and_empty_not_allowed(tmp_path, monkeypatch):
    module = _module()
    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    mismatch = _materialized_export(tmp_path, family="union_all_row_append@duckdb", native_match=False)
    manifest_path = tmp_path / "minimized.json"
    manifest_path.write_text(json.dumps({"exports": [mismatch]}), encoding="utf-8")

    rc = module.run_with_args(
        Namespace(
            minimized_manifest=str(manifest_path),
            output_dir=str(tmp_path / "bundle"),
            tarball="",
            family=[],
            allow_empty=False,
        )
    )

    assert rc == 1
    manifest = json.loads((tmp_path / "bundle" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["summary"]["selected_count"] == 0
    assert manifest["skipped"][0]["reason"] == "native_sql_output_does_not_match_reduced_duckdb_rerun"


def _materialized_export(
    tmp_path: Path,
    *,
    family: str,
    native_match: bool,
    witness_kind: str = "aggregate_value",
) -> dict[str, object]:
    slug = family.replace("@", "_").replace("/", "_")
    sql_path = tmp_path / f"{slug}.sql"
    case_path = tmp_path / f"{slug}.case.json"
    sql_path.write_text("-- sql\nSELECT 1;\n", encoding="utf-8")
    case_path.write_text(json.dumps({"case_id": f"case-{slug}"}), encoding="utf-8")
    export = _export(family, native_match=native_match, witness_kind=witness_kind)
    export["sql_path"] = str(sql_path)
    export["case_path"] = str(case_path)
    return export


def _export(
    family: str,
    *,
    native_match: bool,
    witness_kind: str = "aggregate_value",
) -> dict[str, object]:
    root = family.split("@", 1)[0]
    contract: dict[str, object] = {
        "kind": witness_kind,
        "backends": ["duckdb", "sqlite"],
        "source": "reference_consensus",
    }
    if witness_kind == "row_containment":
        contract["row"] = {"id": 1}
    else:
        contract["group_key"] = {}
        contract["aggregate"] = "sum_x"
        contract["expected"] = 4
    return {
        "family": family,
        "case_id": f"case-{root}",
        "target_reproduced": True,
        "native_sql_matches_rerun_duckdb": native_match,
        "local_sql_execution": {
            "status": "ok",
            "normalized": {"status": "ok", "columns": ["sum_x"], "rows": [[3]]},
        },
        "sql_path": "reproducer.sql",
        "case_path": "case.json",
        "source_artifact": "rows/source.jsonl",
        "source_row_index": 7,
        "expected_finding_keys": [f"semantic_output_mismatch:{family}:value"],
        "output_columns": ["sum_x"],
        "duckdb_rows": [[3]],
        "reference_rows": {"sqlite": [[4]]},
        "witness_plan": {
            "schema_version": "datadiff-reference-witness-plan-v1",
            "status": "available",
            "contract": contract,
            "failing_suspicious_backends": ["duckdb"],
            "satisfied_reference_backends": ["sqlite"],
        },
        "reduced": {"row_count": 2, "operation_count": 1},
        "original": {"row_count": 5, "operation_count": 3},
    }
