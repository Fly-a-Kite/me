#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports"


def main() -> int:
    return run_with_args(parse_args())


def run_with_args(args: argparse.Namespace) -> int:
    final = load_json(Path(str(args.final_readiness)))
    families = selected_families(args.family, args.family_file)
    if not families:
        raise SystemExit("provide at least one --family or --family-file")
    run_records = candidate_run_records(final, families, evidence_mode=str(args.evidence_mode))
    evidence = collect_evidence(run_records, families=families, per_family_limit=int(args.per_family_limit))
    output_base = resolve_output_base(Path(str(args.output_base)))
    row_artifacts: dict[str, object] | None = None
    if bool(args.write_row_artifacts) or str(args.row_output_dir or "").strip():
        row_output_dir = Path(str(args.row_output_dir)) if str(args.row_output_dir or "").strip() else output_base.with_name(f"{output_base.name}-rows")
        if not row_output_dir.is_absolute():
            row_output_dir = PROJECT_ROOT / row_output_dir
        row_artifacts = write_row_artifacts(evidence, row_output_dir=row_output_dir)
    payload = {
        "schema_version": "candidate-family-evidence-v1",
        "generated_at": utc_now_iso(),
        "input_files": {"final_readiness": project_relative(Path(str(args.final_readiness)))},
        "selection": {
            "evidence_mode": str(args.evidence_mode),
            "per_family_limit": int(args.per_family_limit),
        },
        "families": families,
        "summary": summarize_evidence(evidence),
        "row_artifacts": row_artifacts or {},
        "evidence": evidence,
    }
    write_outputs(output_base, payload)
    print(f"candidate evidence json: {output_base.with_suffix('.json')}")
    print(f"candidate evidence md:   {output_base.with_suffix('.md')}")
    return 0


def candidate_run_records(final: dict[str, Any], families: list[str], *, evidence_mode: str = "live") -> list[dict[str, Any]]:
    family_set = set(families)
    records: list[dict[str, Any]] = []
    for run in final.get("runs", []) or []:
        if not isinstance(run, dict):
            continue
        if evidence_mode != "all" and str(run.get("evidence_mode", "")) != evidence_mode:
            continue
        rewardable = run.get("rewardable_candidate_families", {}) or {}
        if not isinstance(rewardable, dict):
            continue
        matched = sorted(family_set.intersection(rewardable))
        if not matched:
            continue
        run_file = str(run.get("run_file", "") or "")
        if not run_file:
            continue
        records.append(
            {
                "run_file": run_file,
                "families": matched,
                "target_suite": str(run.get("target_suite", "")),
                "preset": str(run.get("preset", "")),
                "evidence_mode": str(run.get("evidence_mode", "")),
                "seed": run.get("seed", ""),
                "cases": int(run.get("cases", 0) or 0),
                "elapsed_s": float(run.get("elapsed_s", 0.0) or 0.0),
                "manifest_file": str(run.get("manifest_file", "") or ""),
            }
        )
    return records


def collect_evidence(
    run_records: list[dict[str, Any]],
    *,
    families: list[str],
    per_family_limit: int,
) -> dict[str, list[dict[str, object]]]:
    targets = set(families)
    evidence: dict[str, list[dict[str, object]]] = {family: [] for family in families}
    scanned_files: set[str] = set()
    run_by_file: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in run_records:
        run_by_file[str(record["run_file"])].append(record)
    for run_file_text, records in run_by_file.items():
        if all(len(evidence[family]) >= per_family_limit for family in targets):
            break
        run_file = Path(run_file_text)
        if not run_file.is_absolute():
            run_file = PROJECT_ROOT / run_file
        if not run_file.exists() or str(run_file) in scanned_files:
            continue
        scanned_files.add(str(run_file))
        wanted_for_file = set()
        for record in records:
            wanted_for_file.update(record["families"])
        wanted_for_file = {family for family in wanted_for_file if len(evidence[family]) < per_family_limit}
        if not wanted_for_file:
            continue
        run_context = records[0]
        for index, row in iter_jsonl(run_file):
            row_families = row_candidate_families(row)
            matched = sorted(wanted_for_file.intersection(row_families))
            if not matched:
                continue
            for family in matched:
                if len(evidence[family]) >= per_family_limit:
                    continue
                evidence[family].append(row_evidence(row, family=family, run_file=run_file, run_context=run_context, row_index=index))
            wanted_for_file = {family for family in wanted_for_file if len(evidence[family]) < per_family_limit}
            if not wanted_for_file:
                break
    return evidence


def row_candidate_families(row: dict[str, Any]) -> Counter[str]:
    root_by_backend_group: dict[str, str] = {}
    findings = [finding for finding in row.get("findings", []) or [] if is_candidate_finding(finding)]
    for finding in findings:
        root = str(finding.get("root_cause", "unknown"))
        backend_group = backend_group_key(finding)
        if not root.startswith("metamorphic_"):
            root_by_backend_group.setdefault(backend_group, root)
    result: Counter[str] = Counter()
    for finding in findings:
        root = str(finding.get("root_cause", "unknown"))
        backend_group = backend_group_key(finding)
        if root.startswith("metamorphic_") and backend_group in root_by_backend_group:
            root = root_by_backend_group[backend_group]
        result[f"{root}@{backend_group}"] += 1
    return result


def row_evidence(
    row: dict[str, Any],
    *,
    family: str,
    run_file: Path,
    run_context: dict[str, Any],
    row_index: int,
) -> dict[str, object]:
    findings = []
    for finding in row.get("findings", []) or []:
        if not is_candidate_finding(finding):
            continue
        if family not in row_candidate_families({"findings": [finding]}):
            root, _, backend_text = family.partition("@")
            if str(finding.get("root_cause", "")) != root or backend_group_key(finding) != backend_text:
                continue
        findings.append(
            {
                "kind": finding.get("kind", ""),
                "root_cause": finding.get("root_cause", ""),
                "suspicious_backends": finding.get("suspicious_backends", []),
                "triage_verdict": finding.get("triage_verdict", ""),
                "triage_evidence": finding.get("triage_evidence", ""),
                "signature": finding.get("signature", ""),
                "backend_groups": finding.get("backend_groups", []),
            }
        )
    case = row.get("case", {}) if isinstance(row.get("case"), dict) else {}
    program = case.get("program", {}) if isinstance(case.get("program"), dict) else {}
    return {
        "family": family,
        "run_file": project_relative(run_file),
        "row_index": row_index,
        "case_id": case.get("case_id", ""),
        "case_seed": case.get("seed", ""),
        "status": row.get("status", ""),
        "target_suite": run_context.get("target_suite", ""),
        "preset": run_context.get("preset", ""),
        "evidence_mode": run_context.get("evidence_mode", ""),
        "run_seed": run_context.get("seed", ""),
        "manifest_file": project_relative(Path(str(run_context.get("manifest_file", "")))) if run_context.get("manifest_file") else "",
        "operation_sequence": [str(op.get("op", "")) for op in program.get("operations", []) if isinstance(op, dict)],
        "finding_count": len(findings),
        "findings": findings,
        "source_hint": (
            f"row_index={row_index} in {project_relative(run_file)}; "
            "extract this row for native reproducer/minimization"
        ),
    }


def backend_group_key(finding: dict[str, Any]) -> str:
    suspicious = finding.get("suspicious_backends", []) or []
    if isinstance(suspicious, str):
        suspicious = [suspicious]
    values = [str(item) for item in suspicious if str(item)]
    return ",".join(sorted(values)) if values else "unknown"


def is_candidate_finding(finding: dict[str, Any]) -> bool:
    return finding.get("triage_verdict") == "candidate_implementation_bug" and not finding.get("false_positive")


def summarize_evidence(evidence: dict[str, list[dict[str, object]]]) -> dict[str, object]:
    counts = {family: len(rows) for family, rows in evidence.items()}
    return {
        "family_count": len(evidence),
        "families_with_evidence": sum(1 for rows in evidence.values() if rows),
        "evidence_rows": sum(counts.values()),
        "counts_by_family": counts,
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


def write_row_artifacts(evidence: dict[str, list[dict[str, object]]], *, row_output_dir: Path) -> dict[str, object]:
    row_output_dir.mkdir(parents=True, exist_ok=True)
    row_cache_by_file: dict[str, dict[int, dict[str, Any]]] = {}
    manifest_families: dict[str, dict[str, object]] = {}
    for family, family_rows in evidence.items():
        output_path = row_output_dir / f"{slugify_family(family)}.jsonl"
        written = 0
        with output_path.open("w", encoding="utf-8") as handle:
            for evidence_row in family_rows:
                run_file = Path(str(evidence_row.get("run_file", "")))
                row_index = int(evidence_row.get("row_index", -1))
                if not run_file.is_absolute():
                    run_file = PROJECT_ROOT / run_file
                row_cache = row_cache_by_file.setdefault(str(run_file), {})
                if row_index not in row_cache:
                    row_cache[row_index] = read_jsonl_row(run_file, row_index)
                raw_row = row_cache[row_index]
                if not raw_row:
                    continue
                artifact_row = {
                    "schema_version": "candidate-family-row-artifact-v1",
                    "family": family,
                    "source": {
                        "run_file": project_relative(run_file),
                        "row_index": row_index,
                        "case_id": evidence_row.get("case_id", ""),
                        "evidence_mode": evidence_row.get("evidence_mode", ""),
                        "target_suite": evidence_row.get("target_suite", ""),
                        "preset": evidence_row.get("preset", ""),
                    },
                    "row": raw_row,
                }
                handle.write(json.dumps(artifact_row, ensure_ascii=False, sort_keys=True) + "\n")
                written += 1
        manifest_families[family] = {
            "path": project_relative(output_path),
            "rows": written,
        }
    manifest = {
        "schema_version": "candidate-family-row-artifacts-v1",
        "generated_at": utc_now_iso(),
        "output_dir": project_relative(row_output_dir),
        "families": manifest_families,
    }
    manifest_path = row_output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "manifest": project_relative(manifest_path),
        "output_dir": project_relative(row_output_dir),
        "family_count": len(manifest_families),
        "row_count": sum(int(item.get("rows", 0)) for item in manifest_families.values()),
    }


def render_markdown(payload: dict[str, object]) -> str:
    summary = payload.get("summary", {}) if isinstance(payload.get("summary"), dict) else {}
    selection = payload.get("selection", {}) if isinstance(payload.get("selection"), dict) else {}
    row_artifacts = payload.get("row_artifacts", {}) if isinstance(payload.get("row_artifacts"), dict) else {}
    evidence = payload.get("evidence", {}) if isinstance(payload.get("evidence"), dict) else {}
    lines = [
        "# Candidate Family Evidence",
        "",
        f"- Generated at: `{payload.get('generated_at', '')}`",
        f"- Evidence mode: `{selection.get('evidence_mode', '')}`",
        f"- Per-family limit: `{selection.get('per_family_limit', '')}`",
        f"- Families with evidence: `{summary.get('families_with_evidence', 0)}` / `{summary.get('family_count', 0)}`",
        f"- Evidence rows: `{summary.get('evidence_rows', 0)}`",
        f"- Row artifact manifest: `{row_artifacts.get('manifest', '')}`" if row_artifacts else "- Row artifact manifest: ``",
        "",
    ]
    for family, rows in evidence.items():
        lines.append(f"## `{family}`")
        if not rows:
            lines.append("")
            lines.append("No row-level evidence extracted from the selected final-readiness runs.")
            lines.append("")
            continue
        lines.append("")
        lines.append("| case | run | ops | finding roots | source |")
        lines.append("| --- | --- | --- | --- | --- |")
        for row in rows:
            roots = ", ".join(str(item.get("root_cause", "")) for item in row.get("findings", []) or [])
            ops = ">".join(row.get("operation_sequence", []) or [])
            lines.append(
                "| `{case}` | `{run}` | `{ops}` | `{roots}` | `{repro}` |".format(
                    case=row.get("case_id", ""),
                    run=row.get("run_file", ""),
                    ops=ops,
                    roots=roots,
                    repro=row.get("source_hint", ""),
                )
            )
        lines.append("")
    return "\n".join(lines)


def iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield index, row


def read_jsonl_row(path: Path, row_index: int) -> dict[str, Any]:
    if row_index < 0:
        return {}
    for index, row in iter_jsonl(path):
        if index == row_index:
            return row
        if index > row_index:
            break
    return {}


def slugify_family(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip("-") or "family"


def selected_families(values: list[str], family_file: str) -> list[str]:
    result: list[str] = []
    for value in values:
        for part in str(value).split(","):
            text = part.strip()
            if text and text not in result:
                result.append(text)
    if family_file:
        for line in Path(family_file).read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if text and not text.startswith("#") and text not in result:
                result.append(text)
    return result


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"expected JSON object: {path}")
    return data


def project_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except Exception:
        return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract representative run-log rows for candidate bug families.")
    parser.add_argument("--final-readiness", default="reports/final-readiness-20260614T103547.json")
    parser.add_argument("--family", action="append", default=[])
    parser.add_argument("--family-file", default="")
    parser.add_argument("--evidence-mode", choices=["live", "validation", "ablation", "comparison", "all"], default="live")
    parser.add_argument("--per-family-limit", type=int, default=3)
    parser.add_argument("--output-base", default=f"candidate-family-evidence-{utc_timestamp()}")
    parser.add_argument("--write-row-artifacts", action="store_true")
    parser.add_argument("--row-output-dir", default="")
    return parser.parse_args()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


if __name__ == "__main__":
    sys.exit(main())
