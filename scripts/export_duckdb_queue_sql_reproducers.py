#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from datadiff.dsl import Case
from datadiff.duckdb_sql_export import UnsupportedDuckDBSqlExport, render_duckdb_case_sql


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "duckdb-sql-reproducers"


def main() -> int:
    return run_with_args(parse_args())


def run_with_args(args: argparse.Namespace) -> int:
    queue_path = resolve_project_path(Path(str(args.queue)))
    queue = load_json(queue_path)
    validation = load_json(resolve_project_path(Path(str(args.validation)))) if str(args.validation or "").strip() else {}
    output_dir = resolve_output_dir(Path(str(args.output_dir)))
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = selected_rows(queue, validation=validation, per_family_limit=int(args.per_family_limit))
    exported: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    for row in rows:
        family = str(row.get("family", ""))
        case_id = str(row.get("case_id", ""))
        try:
            case = Case.from_dict(row.get("case", {}) if isinstance(row.get("case"), dict) else {})
            sql = render_duckdb_case_sql(case, header_lines=header_lines(row))
        except (KeyError, TypeError, ValueError, UnsupportedDuckDBSqlExport) as exc:
            skipped.append({"family": family, "case_id": case_id, "reason": f"{type(exc).__name__}: {exc}"})
            continue
        path = output_dir / f"{slugify(family)}__{slugify(case_id)}.sql"
        path.write_text(sql, encoding="utf-8")
        execution = execute_duckdb_sql(sql) if bool(args.execute_sql) else {"status": "not_run"}
        exported.append(
            {
                "family": family,
                "case_id": case_id,
                "sql_path": project_relative(path),
                "local_sql_execution": execution,
                "duckdb_rows": row.get("duckdb_rows", []),
                "reference_rows": row.get("reference_rows", {}),
                "output_columns": row.get("output_columns", []),
                "expected_finding_keys": list(row.get("expected_finding_keys", []) or []),
                "witness_plan": row.get("witness_plan", {}) if isinstance(row.get("witness_plan", {}), dict) else {},
                "verification_backends": list(row.get("verification_backends", []) or []),
                "source_artifact": str(row.get("source_artifact", "")),
                "source_row_index": row.get("source_row_index", ""),
            }
        )
    manifest = {
        "schema_version": "duckdb-native-sql-reproducer-export-v1",
        "generated_at": utc_now_iso(),
        "queue": project_relative(queue_path),
        "validation": project_relative(resolve_project_path(Path(str(args.validation)))) if str(args.validation or "").strip() else "",
        "selection": {"per_family_limit": int(args.per_family_limit), "passed_validation_only": bool(validation)},
        "summary": {
            "selected_count": len(rows),
            "exported_count": len(exported),
            "skipped_count": len(skipped),
            "family_count": len({item["family"] for item in exported}),
            "local_sql_execution_passed_count": sum(
                1 for item in exported if item.get("local_sql_execution", {}).get("status") == "ok"
            ),
            "local_sql_execution_failed_count": sum(
                1 for item in exported if item.get("local_sql_execution", {}).get("status") == "error"
            ),
        },
        "exports": exported,
        "skipped": skipped,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "README.md").write_text(render_markdown(manifest), encoding="utf-8")
    print(f"duckdb sql reproducer manifest: {manifest_path}")
    print(f"exported: {len(exported)} skipped: {len(skipped)}")
    return 1 if skipped and not bool(args.allow_skips) else 0


def selected_rows(
    queue: dict[str, Any],
    *,
    validation: dict[str, Any],
    per_family_limit: int,
) -> list[dict[str, Any]]:
    passed = passed_validation_keys(validation)
    result: list[dict[str, Any]] = []
    families = queue.get("families", {}) if isinstance(queue.get("families"), dict) else {}
    for family, rows in families.items():
        family_rows: list[dict[str, Any]] = []
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            key = row_key(row)
            if passed and key not in passed:
                continue
            family_rows.append(row)
        if per_family_limit > 0:
            family_rows = family_rows[:per_family_limit]
        result.extend(family_rows)
    return result


def passed_validation_keys(validation: dict[str, Any]) -> set[tuple[str, str]]:
    return {
        (str(row.get("family", "")), str(row.get("case_id", "")))
        for row in validation.get("results", []) or []
        if isinstance(row, dict) and row.get("status") == "passed"
    }


def row_key(row: dict[str, Any]) -> tuple[str, str]:
    return (str(row.get("family", "")), str(row.get("case_id", "")))


def header_lines(row: dict[str, Any]) -> list[str]:
    lines = [
        f"family: {row.get('family', '')}",
        f"source: {row.get('source_artifact', '')}:{row.get('source_row_index', '')}",
        f"verification_backends: {', '.join(str(item) for item in row.get('verification_backends', []) or [])}",
    ]
    expected = ", ".join(str(item) for item in row.get("expected_finding_keys", []) or [])
    if expected:
        lines.append(f"expected_finding_keys: {expected}")
    witness = witness_summary(row.get("witness_plan", {}))
    if witness:
        lines.append(f"witness: {witness}")
    return lines


def witness_summary(witness_plan: object) -> str:
    if not isinstance(witness_plan, dict) or witness_plan.get("status") != "available":
        return ""
    contract = witness_plan.get("contract", {})
    if not isinstance(contract, dict):
        return ""
    kind = str(contract.get("kind", "") or "")
    target = contract.get("row") or contract.get("group_key") or {}
    aggregate = str(contract.get("aggregate", "") or "")
    expected = contract.get("expected", "")
    failing = ",".join(str(item) for item in witness_plan.get("failing_suspicious_backends", []) or [])
    parts = [f"kind={kind}"]
    if target:
        parts.append(f"target={json.dumps(target, sort_keys=True, ensure_ascii=True)}")
    if aggregate:
        parts.append(f"aggregate={aggregate}")
        parts.append(f"expected={json.dumps(expected, sort_keys=True, ensure_ascii=True)}")
    if failing:
        parts.append(f"failing={failing}")
    return "; ".join(parts)


def execute_duckdb_sql(sql: str) -> dict[str, object]:
    try:
        import duckdb

        con = duckdb.connect(database=":memory:")
        try:
            result = con.execute(sql)
            rows = result.fetchall()
            columns = [str(item[0]) for item in result.description or []]
        finally:
            con.close()
        return {
            "status": "ok",
            "columns": columns,
            "row_count": len(rows),
            "rows_preview": normalize_rows(rows[:10]),
        }
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


def normalize_rows(rows: list[tuple[object, ...]]) -> list[list[object]]:
    return [[normalize_cell(cell) for cell in row] for row in rows]


def normalize_cell(value: object) -> object:
    try:
        import math

        if isinstance(value, float) and math.isnan(value):
            return None
    except TypeError:
        pass
    return value


def render_markdown(manifest: dict[str, object]) -> str:
    summary = manifest.get("summary", {}) if isinstance(manifest.get("summary"), dict) else {}
    exports = manifest.get("exports", []) if isinstance(manifest.get("exports"), list) else []
    skipped = manifest.get("skipped", []) if isinstance(manifest.get("skipped"), list) else []
    lines = [
        "# DuckDB Native SQL Reproducers",
        "",
        f"- Generated at: `{manifest.get('generated_at', '')}`",
        f"- Exported: `{summary.get('exported_count', 0)}`",
        f"- Skipped: `{summary.get('skipped_count', 0)}`",
        f"- Local SQL execution passed: `{summary.get('local_sql_execution_passed_count', 0)}`",
        f"- Local SQL execution failed: `{summary.get('local_sql_execution_failed_count', 0)}`",
        "",
        "| family | case | SQL | local SQL | source |",
        "| --- | --- | --- | --- | --- |",
    ]
    for item in exports:
        lines.append(
            "| `{family}` | `{case}` | `{sql}` | `{status}` | `{source}:{row}` |".format(
                family=item.get("family", ""),
                case=item.get("case_id", ""),
                sql=item.get("sql_path", ""),
                status=(item.get("local_sql_execution", {}) or {}).get("status", ""),
                source=item.get("source_artifact", ""),
                row=item.get("source_row_index", ""),
            )
        )
    if exports:
        lines.extend(["", "## Normalized Evidence", ""])
        for item in exports:
            refs = item.get("reference_rows", {}) if isinstance(item.get("reference_rows"), dict) else {}
            lines.append(f"### `{item.get('family', '')}`")
            lines.append("")
            lines.append(f"- Output columns: `{', '.join(str(col) for col in item.get('output_columns', []) or [])}`")
            lines.append(f"- DuckDB normalized rows: `{json.dumps(item.get('duckdb_rows', []), ensure_ascii=False)}`")
            for backend, rows in refs.items():
                lines.append(f"- {backend} normalized rows: `{json.dumps(rows, ensure_ascii=False)}`")
            witness = witness_summary(item.get("witness_plan", {}))
            if witness:
                lines.append(f"- Witness: `{witness}`")
            lines.append("")
    if skipped:
        lines.extend(["", "## Skipped", ""])
        for item in skipped:
            lines.append(f"- `{item.get('family', '')}` `{item.get('case_id', '')}`: {item.get('reason', '')}")
    lines.append("")
    return "\n".join(lines)


def resolve_project_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def resolve_output_dir(path: Path) -> Path:
    if path.is_absolute():
        return path
    if path.parent != Path("."):
        return PROJECT_ROOT / path
    return DEFAULT_OUTPUT_DIR / path


def project_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"expected JSON object: {path}")
    return data


def slugify(value: str) -> str:
    safe = []
    for char in str(value):
        if char.isalnum() or char in {"-", "_"}:
            safe.append(char)
        else:
            safe.append("_")
    text = "".join(safe).strip("_")
    return text[:140] or "item"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export DataDiffFuzz DuckDB queue rows as native SQL reproducers.")
    parser.add_argument("--queue", required=True)
    parser.add_argument("--validation", default="")
    parser.add_argument("--per-family-limit", type=int, default=1)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--allow-skips", action="store_true")
    parser.add_argument("--execute-sql", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(main())
