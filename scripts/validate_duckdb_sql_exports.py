#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "duckdb-sql-reproducers" / "fresh-p0-current-validation.json"


def main() -> int:
    return run_with_args(parse_args())


def run_with_args(args: argparse.Namespace) -> int:
    manifest_path = resolve_project_path(Path(str(args.manifest)))
    manifest = load_json(manifest_path)
    results = [validate_export(export) for export in manifest.get("exports", []) or [] if isinstance(export, dict)]
    payload = {
        "schema_version": "duckdb-sql-export-validation-v1",
        "generated_at": utc_now_iso(),
        "input_manifest": project_relative(manifest_path),
        "counting_policy": "Rows are candidate evidence only; count a bug only after current DuckDB still differs from references and upstream confirms.",
        "summary": summarize(results),
        "results": results,
    }
    output = resolve_project_path(Path(str(args.output)))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output.with_suffix(".md").write_text(render_markdown(payload), encoding="utf-8")
    print(f"duckdb sql validation json: {output}")
    print(f"duckdb sql validation md:   {output.with_suffix('.md')}")
    print(json.dumps(payload["summary"], sort_keys=True))
    return 0


def validate_export(export: dict[str, Any]) -> dict[str, Any]:
    family = str(export.get("family", ""))
    sql_path = resolve_project_path(Path(str(export.get("sql_path", ""))))
    output_columns = [str(item) for item in export.get("output_columns", []) or []]
    raw_stored_suspicious_rows = export.get("duckdb_rows")
    has_stored_suspicious_rows = isinstance(raw_stored_suspicious_rows, list)
    stored_suspicious_rows = normalize_rows(raw_stored_suspicious_rows if has_stored_suspicious_rows else [])
    reference_rows = first_reference_rows(export)
    execution = execute_sql(sql_path, output_columns=output_columns)
    current_rows = execution.get("rows", []) if execution.get("status") == "ok" else []
    matches_reference = bool(reference_rows) and rows_equal(current_rows, reference_rows)
    matches_stored_suspicious = has_stored_suspicious_rows and rows_equal(current_rows, stored_suspicious_rows)
    differs_from_reference = bool(reference_rows) and not rows_equal(current_rows, reference_rows)
    verdict = "current_matches_reference_not_latest_bug"
    if execution.get("status") != "ok":
        verdict = "current_sql_execution_error"
    elif matches_stored_suspicious and differs_from_reference:
        verdict = "current_reproduces_stored_suspicious_output"
    elif differs_from_reference:
        verdict = "current_differs_from_reference_but_not_stored_shape"
    return {
        "family": family,
        "case_id": str(export.get("case_id", "")),
        "sql_path": project_relative(sql_path),
        "status": execution.get("status", ""),
        "verdict": verdict,
        "current_matches_reference": matches_reference,
        "current_matches_stored_suspicious": matches_stored_suspicious,
        "has_stored_suspicious_rows": has_stored_suspicious_rows,
        "current_differs_from_reference": differs_from_reference,
        "current_rows": current_rows,
        "stored_suspicious_rows": stored_suspicious_rows,
        "reference_rows": reference_rows,
        "execution_error": execution.get("error", ""),
    }


def execute_sql(path: Path, *, output_columns: list[str]) -> dict[str, Any]:
    try:
        import duckdb

        con = duckdb.connect(database=":memory:")
        try:
            result = con.execute(path.read_text(encoding="utf-8"))
            raw_rows = result.fetchall()
            columns = [str(item[0]) for item in result.description or []]
        finally:
            con.close()
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}", "rows": []}
    rows = rows_in_output_order(raw_rows, columns=columns, output_columns=output_columns)
    return {"status": "ok", "columns": columns, "rows": rows}


def rows_in_output_order(
    rows: list[tuple[Any, ...]],
    *,
    columns: list[str],
    output_columns: list[str],
) -> list[list[Any]]:
    if output_columns and set(output_columns).issubset(set(columns)):
        index = [columns.index(column) for column in output_columns]
    else:
        index = list(range(len(columns)))
    return normalize_rows([[row[i] for i in index] for row in rows])


def first_reference_rows(export: dict[str, Any]) -> list[list[Any]]:
    refs = export.get("reference_rows", {}) if isinstance(export.get("reference_rows", {}), dict) else {}
    for _backend, rows in sorted(refs.items()):
        if isinstance(rows, list) and rows:
            return normalize_rows(rows)
    return []


def normalize_rows(rows: Any) -> list[list[Any]]:
    if not isinstance(rows, list):
        return []
    return [[normalize_cell(cell) for cell in row] for row in rows if isinstance(row, list)]


def normalize_cell(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if value.is_integer() and abs(value) <= 2**63 - 1:
            return int(value)
    return value


def rows_equal(left: list[list[Any]], right: list[list[Any]]) -> bool:
    return sorted(row_key(row) for row in left) == sorted(row_key(row) for row in right)


def row_key(row: list[Any]) -> str:
    return json.dumps(row, ensure_ascii=True, sort_keys=True, default=str)


def summarize(results: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "export_count": len(results),
        "sql_ok_count": sum(1 for row in results if row.get("status") == "ok"),
        "sql_error_count": sum(1 for row in results if row.get("status") != "ok"),
        "current_reproduces_stored_suspicious_count": sum(
            1 for row in results if row.get("verdict") == "current_reproduces_stored_suspicious_output"
        ),
        "current_matches_reference_count": sum(
            1 for row in results if row.get("verdict") == "current_matches_reference_not_latest_bug"
        ),
        "current_differs_other_shape_count": sum(
            1 for row in results if row.get("verdict") == "current_differs_from_reference_but_not_stored_shape"
        ),
    }


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {}) if isinstance(payload.get("summary"), dict) else {}
    lines = [
        "# DuckDB SQL Export Validation",
        "",
        f"- Generated at: `{payload.get('generated_at', '')}`",
        f"- Input manifest: `{payload.get('input_manifest', '')}`",
        f"- Current reproduces stored suspicious output: `{summary.get('current_reproduces_stored_suspicious_count', 0)}`",
        f"- Current matches reference: `{summary.get('current_matches_reference_count', 0)}`",
        f"- SQL errors: `{summary.get('sql_error_count', 0)}`",
        "",
        "| verdict | family | case | sql |",
        "| --- | --- | --- | --- |",
    ]
    for row in payload.get("results", []) or []:
        if not isinstance(row, dict):
            continue
        lines.append(
            "| `{}` | `{}` | `{}` | `{}` |".format(
                row.get("verdict", ""),
                row.get("family", ""),
                row.get("case_id", ""),
                row.get("sql_path", ""),
            )
        )
    lines.append("")
    return "\n".join(lines)


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"expected JSON object: {path}")
    return data


def resolve_project_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def project_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate exported DuckDB SQL against stored suspicious/reference rows.")
    parser.add_argument("--manifest", default="reports/duckdb-sql-reproducers/fresh-p0-current/manifest.json")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    sys.exit(main())
