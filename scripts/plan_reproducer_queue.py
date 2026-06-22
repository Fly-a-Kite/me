#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from datadiff.reproducer_scripts import write_evidence_queue_reproducer
from datadiff.run_findings import _finding_recheck_key, _format_recheck_key
from datadiff.witness_oracle import build_reference_witness_plan


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports"


def main() -> int:
    return run_with_args(parse_args())


def run_with_args(args: argparse.Namespace) -> int:
    manifest_path = Path(str(args.row_artifact_manifest))
    manifest = load_json(manifest_path)
    candidates = collect_candidates(manifest, per_family_limit=int(args.per_family_limit))
    output_base = resolve_output_base(Path(str(args.output_base)))
    payload = {
        "schema_version": "reproducer-queue-v1",
        "generated_at": utc_now_iso(),
        "input_files": {"row_artifact_manifest": project_relative(manifest_path)},
        "selection_policy": {
            "per_family_limit": int(args.per_family_limit),
            "sort_order": [
                "recheck reproduced first",
                "fewest operations",
                "most corroborating/reference backends",
                "shortest row count",
                "case id",
            ],
        },
        "summary": summarize_queue(candidates),
        "families": candidates,
    }
    write_outputs(output_base, payload)
    script_path = write_evidence_queue_reproducer(output_base.with_suffix(".json"))
    print(f"reproducer queue json: {output_base.with_suffix('.json')}")
    print(f"reproducer queue md:   {output_base.with_suffix('.md')}")
    print(f"reproducer queue py:   {script_path}")
    return 0


def collect_candidates(manifest: dict[str, Any], *, per_family_limit: int) -> dict[str, list[dict[str, object]]]:
    families = manifest.get("families", {}) if isinstance(manifest.get("families"), dict) else {}
    result: dict[str, list[dict[str, object]]] = {}
    for family, meta in sorted(families.items()):
        if not isinstance(meta, dict):
            continue
        artifact_path = Path(str(meta.get("path", "")))
        if not artifact_path.is_absolute():
            artifact_path = PROJECT_ROOT / artifact_path
        rows = [queue_item(artifact, artifact_path=artifact_path) for artifact in iter_jsonl(artifact_path)]
        rows = [row for row in rows if row]
        rows.sort(key=queue_sort_key)
        result[str(family)] = rows[:per_family_limit]
    return result


def queue_item(artifact: dict[str, Any], *, artifact_path: Path) -> dict[str, object]:
    row = artifact.get("row", {}) if isinstance(artifact.get("row"), dict) else {}
    case = row.get("case", {}) if isinstance(row.get("case"), dict) else {}
    program = case.get("program", {}) if isinstance(case.get("program"), dict) else {}
    operations = [op for op in program.get("operations", []) or [] if isinstance(op, dict)]
    findings = [finding for finding in row.get("findings", []) or [] if is_candidate_finding(finding)]
    if not findings:
        return {}
    primary = findings[0]
    normalized = row.get("normalized", {}) if isinstance(row.get("normalized"), dict) else {}
    disagreement = row.get("disagreement_descriptor", {}) if isinstance(row.get("disagreement_descriptor"), dict) else {}
    recheck = row.get("candidate_recheck", {}) if isinstance(row.get("candidate_recheck"), dict) else {}
    source = artifact.get("source", {}) if isinstance(artifact.get("source"), dict) else {}
    suspicious = sorted(str(item) for item in primary.get("suspicious_backends", []) or [])
    reference_backends = [
        backend
        for backend in sorted(str(name) for name in normalized)
        if backend not in suspicious and normalized_status(normalized.get(backend)) == "ok"
    ]
    candidate_findings = [
        finding
        for finding in findings
        if str(finding.get("root_cause", "")) == str(primary.get("root_cause", ""))
        and sorted(str(item) for item in finding.get("suspicious_backends", []) or []) == suspicious
    ]
    return {
        "family": str(artifact.get("family", "")),
        "case_id": str(source.get("case_id") or case.get("case_id", "")),
        "case_seed": case.get("seed", ""),
        "case": case,
        "config": row.get("config", {}) if isinstance(row.get("config", {}), dict) else {},
        "source_artifact": project_relative(artifact_path),
        "source_run_file": str(source.get("run_file", "")),
        "source_row_index": source.get("row_index", ""),
        "target_suite": str(source.get("target_suite", "")),
        "preset": str(source.get("preset", "")),
        "root_cause": str(primary.get("root_cause", "")),
        "kind": str(primary.get("kind", "")),
        "signature": str(primary.get("signature", "")),
        "suspicious_backends": suspicious,
        "reference_backends": reference_backends,
        "verification_backends": verification_backends(row, suspicious=suspicious, reference_backends=reference_backends),
        "expected_finding_keys": expected_finding_keys(candidate_findings or [primary]),
        "recheck_attempts": int(recheck.get("attempts", 0) or 0),
        "recheck_reproduced": bool(recheck.get("reproduced_keys")) and not bool(recheck.get("non_reproduced_keys")),
        "operation_count": len(operations),
        "operation_sequence": [str(op.get("op", "")) for op in operations],
        "table_count": len(case.get("tables", []) or []),
        "input_row_count": sum(len(table.get("rows", []) or []) for table in case.get("tables", []) or [] if isinstance(table, dict)),
        "output_columns": output_columns(normalized, suspicious=suspicious, reference_backends=reference_backends),
        "duckdb_rows": normalized_rows(normalized, "duckdb"),
        "reference_rows": reference_rows(normalized, reference_backends),
        "witness_plan": build_reference_witness_plan(
            normalized,
            suspicious_backends=suspicious,
            reference_backends=reference_backends,
            operations=operations,
            root_cause=str(primary.get("root_cause", "")),
            family=str(artifact.get("family", "")),
        ),
        "mismatch_class": str(disagreement.get("mismatch_class", "")),
        "triage_evidence": str(primary.get("triage_evidence", "")),
        "next_step": next_step_for_family(str(artifact.get("family", "")), operations),
    }


def queue_sort_key(item: dict[str, object]) -> tuple[object, ...]:
    return (
        0 if item.get("recheck_reproduced") else 1,
        int(item.get("operation_count", 999)),
        -len(item.get("reference_backends", []) or []),
        int(item.get("input_row_count", 999999)),
        str(item.get("case_id", "")),
    )


def expected_finding_keys(findings: list[dict[str, Any]]) -> list[str]:
    return sorted(_format_recheck_key(_finding_recheck_key(finding)) for finding in findings)


def verification_backends(
    row: dict[str, Any],
    *,
    suspicious: list[str],
    reference_backends: list[str],
) -> list[str]:
    normalized = row.get("normalized", {})
    if isinstance(normalized, dict) and normalized:
        return [str(name) for name in normalized]
    raw_results = row.get("raw_results", {})
    if isinstance(raw_results, dict) and raw_results:
        return [str(name) for name in raw_results]
    return unique_strings([*suspicious, *reference_backends])


def unique_strings(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def is_candidate_finding(finding: dict[str, Any]) -> bool:
    return finding.get("triage_verdict") == "candidate_implementation_bug" and not finding.get("false_positive")


def normalized_status(value: object) -> str:
    if isinstance(value, dict):
        return str(value.get("status", ""))
    return ""


def normalized_rows(normalized: dict[str, Any], backend: str) -> list[object]:
    value = normalized.get(backend, {})
    if isinstance(value, dict):
        rows = value.get("rows", [])
        return rows if isinstance(rows, list) else []
    return []


def output_columns(normalized: dict[str, Any], *, suspicious: list[str], reference_backends: list[str]) -> list[str]:
    for backend in [*suspicious, *reference_backends]:
        value = normalized.get(backend, {})
        if isinstance(value, dict) and isinstance(value.get("columns"), list):
            return [str(item) for item in value.get("columns", [])]
    return []


def reference_rows(normalized: dict[str, Any], reference_backends: list[str]) -> dict[str, list[object]]:
    return {backend: normalized_rows(normalized, backend) for backend in reference_backends[:3]}


def next_step_for_family(family: str, operations: list[dict[str, Any]]) -> str:
    ops = ">".join(str(op.get("op", "")) for op in operations)
    if "@duckdb" in family:
        return f"Build a DuckDB SQL reproducer for `{ops}` and compare against sqlite/pandas reference rows."
    return f"Build a native reproducer for `{ops}` and compare against the listed reference backends."


def summarize_queue(families: dict[str, list[dict[str, object]]]) -> dict[str, object]:
    return {
        "family_count": len(families),
        "candidate_count": sum(len(rows) for rows in families.values()),
        "families_with_candidates": sum(1 for rows in families.values() if rows),
        "recheck_reproduced_count": sum(
            1 for rows in families.values() for row in rows if bool(row.get("recheck_reproduced"))
        ),
        "witness_plan_available_count": sum(
            1
            for rows in families.values()
            for row in rows
            if isinstance(row.get("witness_plan"), dict) and row.get("witness_plan", {}).get("status") == "available"
        ),
    }


def write_outputs(output_base: Path, payload: dict[str, object]) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    output_base.with_suffix(".json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_base.with_suffix(".md").write_text(render_markdown(payload), encoding="utf-8")


def resolve_output_base(path: Path) -> Path:
    if path.is_absolute():
        return path
    if path.parent != Path("."):
        return PROJECT_ROOT / path
    return DEFAULT_OUTPUT_DIR / path


def render_markdown(payload: dict[str, object]) -> str:
    summary = payload.get("summary", {}) if isinstance(payload.get("summary"), dict) else {}
    families = payload.get("families", {}) if isinstance(payload.get("families"), dict) else {}
    lines = [
        "# Reproducer Queue",
        "",
        f"- Generated at: `{payload.get('generated_at', '')}`",
        f"- Families with candidates: `{summary.get('families_with_candidates', 0)}` / `{summary.get('family_count', 0)}`",
        f"- Candidate rows: `{summary.get('candidate_count', 0)}`",
        f"- Recheck reproduced rows: `{summary.get('recheck_reproduced_count', 0)}`",
        f"- Witness plans available: `{summary.get('witness_plan_available_count', 0)}`",
        "",
    ]
    for family, rows in families.items():
        lines.append(f"## `{family}`")
        lines.append("")
        if not rows:
            lines.append("No candidates.")
            lines.append("")
            continue
        lines.append("| rank | case | ops | refs | witness | output | source | next |")
        lines.append("| ---: | --- | --- | --- | --- | --- | --- | --- |")
        for index, row in enumerate(rows, start=1):
            witness = row.get("witness_plan", {}) if isinstance(row.get("witness_plan"), dict) else {}
            contract = witness.get("contract", {}) if isinstance(witness.get("contract"), dict) else {}
            lines.append(
                "| {rank} | `{case}` | `{ops}` | `{refs}` | `{witness}` | `{output}` | `{source}:{row_index}` | {next_step} |".format(
                    rank=index,
                    case=row.get("case_id", ""),
                    ops=">".join(row.get("operation_sequence", []) or []),
                    refs=",".join(row.get("reference_backends", []) or []),
                    witness=str(contract.get("kind", witness.get("status", ""))),
                    output=",".join(row.get("output_columns", []) or []),
                    source=row.get("source_artifact", ""),
                    row_index=row.get("source_row_index", ""),
                    next_step=row.get("next_step", ""),
                )
            )
        lines.append("")
    return "\n".join(lines)


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            data = json.loads(text)
            if isinstance(data, dict):
                rows.append(data)
    return rows


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"expected JSON object: {path}")
    return data


def project_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan a native/minimized reproducer queue from candidate row artifacts.")
    parser.add_argument("--row-artifact-manifest", required=True)
    parser.add_argument("--per-family-limit", type=int, default=2)
    parser.add_argument("--output-base", default=f"reproducer-queue-{utc_timestamp()}")
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(main())
