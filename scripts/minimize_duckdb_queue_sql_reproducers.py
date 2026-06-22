#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from datadiff.backends.base import BackendResult
from datadiff.config import ExperimentConfig
from datadiff.dsl import Case
from datadiff.duckdb_sql_export import UnsupportedDuckDBSqlExport, render_duckdb_case_sql
from datadiff.normalizer import normalize_result
from datadiff.reducer import reduce_case
from datadiff.runner import run_loaded_case
from datadiff.util import dump_json
from datadiff.witness_oracle import build_reference_witness_plan


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "duckdb-sql-reproducers" / "minimized"


def main() -> int:
    return run_with_args(parse_args())


def run_with_args(args: argparse.Namespace) -> int:
    queue_path = resolve_project_path(Path(str(args.queue)))
    validation_path = resolve_project_path(Path(str(args.validation))) if str(args.validation or "").strip() else None
    queue = load_json(queue_path)
    validation = load_json(validation_path) if validation_path else {}
    output_dir = resolve_output_dir(Path(str(args.output_dir)))
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = selected_rows(queue, validation=validation, per_family_limit=int(args.per_family_limit))
    exports: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    for row in rows:
        family = str(row.get("family", ""))
        case_id = str(row.get("case_id", ""))
        try:
            original = Case.from_dict(row.get("case", {}) if isinstance(row.get("case"), dict) else {})
            backends = [str(item) for item in row.get("verification_backends", []) or [] if str(item)]
            config = reduction_config(row)
            reduced = reduce_case(
                original,
                backends=backends,
                config=config,
                target_kinds=[str(row.get("kind", "")) or "semantic_output_mismatch"],
                target_roots=[str(row.get("root_cause", ""))],
                target_suspicious_backends=[list(row.get("suspicious_backends", []) or [])],
            )
            rerun = run_loaded_case(reduced, backends=backends, config=config, save_artifact=False)
        except (KeyError, TypeError, ValueError, UnsupportedDuckDBSqlExport) as exc:
            skipped.append({"family": family, "case_id": case_id, "reason": f"{type(exc).__name__}: {exc}"})
            continue
        except Exception as exc:  # noqa: BLE001
            skipped.append({"family": family, "case_id": case_id, "reason": f"{type(exc).__name__}: {exc}"})
            continue
        base = f"{slugify(family)}__{slugify(case_id)}"
        case_path = output_dir / f"{base}.case.json"
        sql_path = output_dir / f"{base}.sql"
        dump_json(reduced.to_dict(), case_path)
        source_evidence = source_row_evidence(row)
        reduced_evidence = rerun_evidence(rerun, suspicious_backends=row.get("suspicious_backends", []) or [])
        reduced_witness_plan = build_reduced_witness_plan(row, reduced, rerun)
        sql = render_duckdb_case_sql(
            reduced,
            header_lines=header_lines(row, original=original, reduced=reduced, witness_plan=reduced_witness_plan),
        )
        sql_path.write_text(sql, encoding="utf-8")
        local_sql_execution = execute_duckdb_sql(sql, case=reduced, config=config)
        native_sql_matches = native_sql_matches_rerun_duckdb(local_sql_execution, reduced_evidence)
        exports.append(
            {
                "family": family,
                "case_id": case_id,
                "case_path": project_relative(case_path),
                "sql_path": project_relative(sql_path),
                "local_sql_execution": local_sql_execution,
                "native_sql_matches_rerun_duckdb": native_sql_matches,
                "verification_backends": backends,
                "expected_finding_keys": list(row.get("expected_finding_keys", []) or []),
                "witness_plan": reduced_witness_plan,
                "source_witness_plan": row.get("witness_plan", {}) if isinstance(row.get("witness_plan", {}), dict) else {},
                "source_artifact": str(row.get("source_artifact", "")),
                "source_row_index": row.get("source_row_index", ""),
                "original": case_size(original),
                "reduced": case_size(reduced),
                "rerun_status": str(rerun.get("status", "")),
                "rerun_finding_keys": finding_keys(rerun.get("findings", []) or []),
                "target_reproduced": target_reproduced(row, rerun.get("findings", []) or []),
                "duckdb_rows": reduced_evidence["duckdb_rows"],
                "reference_rows": reduced_evidence["reference_rows"],
                "output_columns": reduced_evidence["output_columns"],
                "reduced_rerun_evidence": reduced_evidence,
                "source_evidence": source_evidence,
            }
        )
    manifest = {
        "schema_version": "duckdb-minimized-sql-reproducer-export-v3",
        "generated_at": utc_now_iso(),
        "queue": project_relative(queue_path),
        "validation": project_relative(validation_path) if validation_path else "",
        "selection": {"per_family_limit": int(args.per_family_limit), "passed_validation_only": bool(validation)},
        "summary": {
            "selected_count": len(rows),
            "exported_count": len(exports),
            "skipped_count": len(skipped),
            "family_count": len({item["family"] for item in exports}),
            "target_reproduced_count": sum(1 for item in exports if bool(item.get("target_reproduced"))),
            "local_sql_execution_passed_count": sum(
                1 for item in exports if item.get("local_sql_execution", {}).get("status") == "ok"
            ),
            "local_sql_execution_failed_count": sum(
                1 for item in exports if item.get("local_sql_execution", {}).get("status") == "error"
            ),
            "native_sql_match_count": sum(1 for item in exports if bool(item.get("native_sql_matches_rerun_duckdb"))),
            "native_sql_mismatch_count": sum(
                1
                for item in exports
                if item.get("local_sql_execution", {}).get("status") == "ok"
                and not bool(item.get("native_sql_matches_rerun_duckdb"))
            ),
            "total_original_rows": sum(int(item.get("original", {}).get("row_count", 0)) for item in exports),
            "total_reduced_rows": sum(int(item.get("reduced", {}).get("row_count", 0)) for item in exports),
            "total_original_operations": sum(int(item.get("original", {}).get("operation_count", 0)) for item in exports),
            "total_reduced_operations": sum(int(item.get("reduced", {}).get("operation_count", 0)) for item in exports),
        },
        "exports": exports,
        "skipped": skipped,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "README.md").write_text(render_markdown(manifest), encoding="utf-8")
    print(f"minimized duckdb sql reproducer manifest: {manifest_path}")
    print(f"exported: {len(exports)} skipped: {len(skipped)}")
    failed = (
        skipped
        or any(not item.get("target_reproduced") for item in exports)
        or any(item.get("local_sql_execution", {}).get("status") == "error" for item in exports)
        or any(
            item.get("local_sql_execution", {}).get("status") == "ok"
            and not bool(item.get("native_sql_matches_rerun_duckdb"))
            for item in exports
        )
    )
    return 1 if failed and not bool(args.allow_failures) else 0


def selected_rows(
    queue: dict[str, Any],
    *,
    validation: dict[str, Any],
    per_family_limit: int,
) -> list[dict[str, Any]]:
    passed = {
        (str(row.get("family", "")), str(row.get("case_id", "")))
        for row in validation.get("results", []) or []
        if isinstance(row, dict) and row.get("status") == "passed"
    }
    result: list[dict[str, Any]] = []
    families = queue.get("families", {}) if isinstance(queue.get("families"), dict) else {}
    for _family, rows in families.items():
        family_rows = []
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            if passed and (str(row.get("family", "")), str(row.get("case_id", ""))) not in passed:
                continue
            family_rows.append(row)
        if per_family_limit > 0:
            family_rows = family_rows[:per_family_limit]
        result.extend(family_rows)
    return result


def reduction_config(row: dict[str, Any]) -> ExperimentConfig:
    payload = dict(row.get("config", {}) if isinstance(row.get("config", {}), dict) else {})
    payload["enable_artifact"] = False
    payload["candidate_recheck_count"] = 0
    return ExperimentConfig.from_payload(payload)


def case_size(case: Case) -> dict[str, int]:
    return {
        "table_count": len(case.tables),
        "row_count": sum(len(table.rows) for table in case.tables),
        "column_count": sum(len(table.columns) for table in case.tables),
        "operation_count": len(case.program.operations),
    }


def finding_keys(findings: list[Any]) -> list[str]:
    return sorted(
        {
            f"{finding.get('kind', '')}:{finding.get('root_cause', 'unknown')}@{','.join(sorted(str(item) for item in finding.get('suspicious_backends', []) or []))}:{finding.get('mismatch_class', '')}"
            for finding in findings
            if isinstance(finding, dict)
        }
    )


def target_reproduced(row: dict[str, Any], findings: list[Any]) -> bool:
    target_root = str(row.get("root_cause", ""))
    target_backends = sorted(str(item) for item in row.get("suspicious_backends", []) or [])
    target_kind = str(row.get("kind", ""))
    for finding in findings:
        if not isinstance(finding, dict):
            continue
        if target_kind and str(finding.get("kind", "")) != target_kind:
            continue
        if target_root and str(finding.get("root_cause", "")) != target_root:
            continue
        if target_backends and sorted(str(item) for item in finding.get("suspicious_backends", []) or []) != target_backends:
            continue
        return True
    return False


def source_row_evidence(row: dict[str, Any]) -> dict[str, object]:
    return {
        "duckdb_rows": row.get("duckdb_rows", []),
        "reference_rows": row.get("reference_rows", {}),
        "output_columns": row.get("output_columns", []),
    }


def rerun_evidence(rerun: dict[str, Any], *, suspicious_backends: list[Any]) -> dict[str, object]:
    normalized = rerun.get("normalized", {}) if isinstance(rerun.get("normalized", {}), dict) else {}
    suspicious = {str(item) for item in suspicious_backends}
    duckdb_payload = normalized.get("duckdb", {}) if isinstance(normalized.get("duckdb", {}), dict) else {}
    rows_by_backend = {
        str(backend): payload.get("rows", [])
        for backend, payload in normalized.items()
        if isinstance(payload, dict) and payload.get("status") == "ok"
    }
    reference_rows = {
        backend: rows
        for backend, rows in rows_by_backend.items()
        if backend not in suspicious
    }
    output_columns: list[str] = []
    if duckdb_payload.get("status") == "ok":
        output_columns = [str(column) for column in duckdb_payload.get("columns", []) or []]
    else:
        for payload in normalized.values():
            if isinstance(payload, dict) and payload.get("status") == "ok":
                output_columns = [str(column) for column in payload.get("columns", []) or []]
                break
    return {
        "backend_statuses": {
            str(backend): {
                "status": str(payload.get("status", "")),
                "row_count": int(payload.get("row_count", len(payload.get("rows", []) or [])) or 0),
                "error_type": str(payload.get("error_type", "")),
            }
            for backend, payload in normalized.items()
            if isinstance(payload, dict)
        },
        "duckdb_rows": duckdb_payload.get("rows", []) if duckdb_payload.get("status") == "ok" else [],
        "reference_rows": reference_rows,
        "output_columns": output_columns,
    }


def build_reduced_witness_plan(row: dict[str, Any], reduced: Case, rerun: dict[str, Any]) -> dict[str, Any]:
    normalized = rerun.get("normalized", {}) if isinstance(rerun.get("normalized", {}), dict) else {}
    if not normalized:
        return {
            "schema_version": "datadiff-reference-witness-plan-v1",
            "status": "not_available",
            "reason": "reduced rerun has no normalized outputs",
        }
    suspicious = [str(item) for item in row.get("suspicious_backends", []) or [] if str(item)]
    reference_backends = [
        str(backend)
        for backend in row.get("verification_backends", []) or []
        if str(backend) not in set(suspicious)
    ]
    operations = [operation.to_dict() if hasattr(operation, "to_dict") else dict(operation) for operation in reduced.program.operations]
    return build_reference_witness_plan(
        normalized,
        suspicious_backends=suspicious,
        reference_backends=reference_backends,
        operations=operations,
        root_cause=str(row.get("root_cause", "")),
        family=str(row.get("family", "")),
    )


def native_sql_matches_rerun_duckdb(
    local_sql_execution: dict[str, object],
    reduced_evidence: dict[str, object],
) -> bool:
    normalized = local_sql_execution.get("normalized") if isinstance(local_sql_execution, dict) else None
    if not isinstance(normalized, dict) or normalized.get("status") != "ok":
        return False
    return (
        strict_json_equal(normalized.get("columns", []), reduced_evidence.get("output_columns", []))
        and strict_json_equal(normalized.get("rows", []), reduced_evidence.get("duckdb_rows", []))
    )


def strict_json_equal(left: Any, right: Any) -> bool:
    return json.dumps(left, sort_keys=True, separators=(",", ":"), allow_nan=False) == json.dumps(
        right,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def execute_duckdb_sql(sql: str, *, case: Case, config: ExperimentConfig) -> dict[str, object]:
    try:
        import duckdb

        con = duckdb.connect(database=":memory:")
        try:
            result = con.execute(sql)
            rows = result.fetchall()
            columns = [str(item[0]) for item in result.description or []]
        finally:
            con.close()
        normalized = normalize_result(
            BackendResult("duckdb_native_sql", "ok", data=NativeSqlResult(columns, rows)),
            case.program,
            enable_normalizer=config.enable_normalizer,
        ).to_dict()
        return {
            "status": "ok",
            "columns": columns,
            "row_count": len(rows),
            "rows_preview": normalize_rows(rows[:10]),
            "normalized": normalized,
        }
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


class NativeSqlResult:
    def __init__(self, columns: list[str], rows: list[tuple[object, ...]]) -> None:
        self.columns = columns
        self._rows = rows

    def rows(self, named: bool = False) -> list[tuple[object, ...]]:  # noqa: FBT001, FBT002
        if named:
            return [dict(zip(self.columns, row)) for row in self._rows]  # type: ignore[return-value]
        return list(self._rows)


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


def header_lines(
    row: dict[str, Any],
    *,
    original: Case,
    reduced: Case,
    witness_plan: dict[str, Any] | None = None,
) -> list[str]:
    lines = [
        f"family: {row.get('family', '')}",
        f"source: {row.get('source_artifact', '')}:{row.get('source_row_index', '')}",
        f"original_rows: {case_size(original)['row_count']}; reduced_rows: {case_size(reduced)['row_count']}",
        f"original_operations: {case_size(original)['operation_count']}; reduced_operations: {case_size(reduced)['operation_count']}",
        f"expected_finding_keys: {', '.join(str(item) for item in row.get('expected_finding_keys', []) or [])}",
    ]
    witness = witness_summary(witness_plan or {})
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


def render_markdown(manifest: dict[str, object]) -> str:
    summary = manifest.get("summary", {}) if isinstance(manifest.get("summary"), dict) else {}
    exports = manifest.get("exports", []) if isinstance(manifest.get("exports"), list) else []
    skipped = manifest.get("skipped", []) if isinstance(manifest.get("skipped"), list) else []
    lines = [
        "# Minimized DuckDB SQL Reproducers",
        "",
        f"- Generated at: `{manifest.get('generated_at', '')}`",
        f"- Exported: `{summary.get('exported_count', 0)}`",
        f"- Target reproduced: `{summary.get('target_reproduced_count', 0)}`",
        f"- Local SQL execution passed: `{summary.get('local_sql_execution_passed_count', 0)}`",
        f"- Local SQL execution failed: `{summary.get('local_sql_execution_failed_count', 0)}`",
        f"- Native SQL matches reduced DuckDB rerun: `{summary.get('native_sql_match_count', 0)}`",
        f"- Native SQL mismatches reduced DuckDB rerun: `{summary.get('native_sql_mismatch_count', 0)}`",
        f"- Rows: `{summary.get('total_original_rows', 0)}` -> `{summary.get('total_reduced_rows', 0)}`",
        f"- Operations: `{summary.get('total_original_operations', 0)}` -> `{summary.get('total_reduced_operations', 0)}`",
        "",
        "| family | case | rows | ops | target | local SQL | native match | SQL |",
        "| --- | --- | ---: | ---: | --- | --- | --- | --- |",
    ]
    for item in exports:
        original = item.get("original", {}) if isinstance(item.get("original"), dict) else {}
        reduced = item.get("reduced", {}) if isinstance(item.get("reduced"), dict) else {}
        lines.append(
            "| `{family}` | `{case}` | {orows}->{rrows} | {oops}->{rops} | `{target}` | `{local_sql}` | `{native_match}` | `{sql}` |".format(
                family=item.get("family", ""),
                case=item.get("case_id", ""),
                orows=original.get("row_count", 0),
                rrows=reduced.get("row_count", 0),
                oops=original.get("operation_count", 0),
                rops=reduced.get("operation_count", 0),
                target=str(item.get("target_reproduced", False)).lower(),
                local_sql=(item.get("local_sql_execution", {}) or {}).get("status", ""),
                native_match=str(item.get("native_sql_matches_rerun_duckdb", False)).lower(),
                sql=item.get("sql_path", ""),
            )
        )
    if exports:
        lines.extend(["", "## Reduced Rerun Evidence", ""])
        for item in exports:
            reduced_evidence = (
                item.get("reduced_rerun_evidence", {})
                if isinstance(item.get("reduced_rerun_evidence", {}), dict)
                else {}
            )
            source_evidence = (
                item.get("source_evidence", {}) if isinstance(item.get("source_evidence", {}), dict) else {}
            )
            lines.append(f"### `{item.get('family', '')}`")
            lines.append("")
            lines.append(f"- Output columns: `{', '.join(str(col) for col in reduced_evidence.get('output_columns', []) or [])}`")
            lines.append(f"- DuckDB reduced rows: `{json.dumps(reduced_evidence.get('duckdb_rows', []), ensure_ascii=False)}`")
            refs = reduced_evidence.get("reference_rows", {})
            if isinstance(refs, dict):
                for backend, rows in refs.items():
                    lines.append(f"- {backend} reduced rows: `{json.dumps(rows, ensure_ascii=False)}`")
            lines.append(f"- Source DuckDB rows: `{json.dumps(source_evidence.get('duckdb_rows', []), ensure_ascii=False)}`")
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


def resolve_project_path(path: Path | None) -> Path:
    if path is None:
        raise ValueError("missing path")
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def resolve_output_dir(path: Path) -> Path:
    if path.is_absolute():
        return path
    if path.parent != Path("."):
        return PROJECT_ROOT / path
    return DEFAULT_OUTPUT_DIR / path


def project_relative(path: Path | None) -> str:
    if path is None:
        return ""
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
    parser = argparse.ArgumentParser(description="Reduce validated DuckDB queue rows and export native SQL reproducers.")
    parser.add_argument("--queue", required=True)
    parser.add_argument("--validation", default="")
    parser.add_argument("--per-family-limit", type=int, default=1)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--allow-failures", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(main())
