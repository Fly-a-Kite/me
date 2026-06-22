#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MINIMIZED_MANIFEST = (
    PROJECT_ROOT
    / "reports"
    / "duckdb-sql-reproducers"
    / "p0-duckdb-live-20260615-minimized"
    / "manifest.json"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "duckdb-issue-ready-bundles" / "p0-duckdb-live-20260615"


def main() -> int:
    return run_with_args(parse_args())


def run_with_args(args: argparse.Namespace) -> int:
    manifest_path = resolve_project_path(Path(str(args.minimized_manifest)))
    minimized = load_json(manifest_path)
    output_dir = resolve_output_dir(Path(str(args.output_dir)))
    clean_output_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected, skipped = select_issue_ready_exports(
        minimized,
        include_families=[str(item) for item in args.family or []],
    )
    issue_dirs: list[dict[str, Any]] = []
    for export in selected:
        issue_dirs.append(write_issue_dir(export, output_dir=output_dir))
    bundle_manifest = {
        "schema_version": "duckdb-issue-ready-bundle-v1",
        "generated_at": utc_now_iso(),
        "source_minimized_manifest": project_relative(manifest_path),
        "selection_policy": {
            "requires_target_reproduced": True,
            "requires_local_sql_execution_ok": True,
            "requires_strict_native_sql_match": True,
            "counting_policy": "candidate issue evidence only; not a confirmed bug until upstream labels, acknowledges, or fixes it",
        },
        "summary": {
            "source_export_count": len(minimized.get("exports", []) or []),
            "selected_count": len(selected),
            "skipped_count": len(skipped),
            "family_count": len({str(item.get("family", "")) for item in selected}),
        },
        "issues": issue_dirs,
        "skipped": skipped,
    }
    manifest_out = output_dir / "manifest.json"
    manifest_out.write_text(json.dumps(bundle_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "README.md").write_text(render_bundle_readme(bundle_manifest), encoding="utf-8")
    tarball_path = Path(str(args.tarball)) if str(args.tarball or "").strip() else output_dir.with_suffix(".tar.gz")
    tarball_path = resolve_output_dir(tarball_path)
    write_tarball(output_dir, tarball_path)
    checksum = sha256_file(tarball_path)
    checksum_path = tarball_path.with_suffix(tarball_path.suffix + ".sha256")
    checksum_path.write_text(f"{checksum}  {tarball_path.name}\n", encoding="utf-8")
    print(f"duckdb issue-ready bundle dir: {output_dir}")
    print(f"duckdb issue-ready tarball:    {tarball_path}")
    print(f"sha256:                       {checksum_path}")
    print(f"selected: {len(selected)} skipped: {len(skipped)}")
    if not selected and not bool(args.allow_empty):
        return 1
    return 0


def select_issue_ready_exports(
    manifest: dict[str, Any],
    *,
    include_families: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    family_filter = {str(item) for item in include_families if str(item)}
    for export in manifest.get("exports", []) or []:
        if not isinstance(export, dict):
            continue
        family = str(export.get("family", ""))
        if family_filter and family not in family_filter:
            skipped.append(skip_record(export, "not_requested_family"))
            continue
        reasons = issue_ready_blockers(export)
        if reasons:
            skipped.append(skip_record(export, ",".join(reasons)))
            continue
        selected.append(export)
    return selected, skipped


def issue_ready_blockers(export: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if not bool(export.get("target_reproduced")):
        reasons.append("target_not_reproduced")
    local_sql = export.get("local_sql_execution", {}) if isinstance(export.get("local_sql_execution", {}), dict) else {}
    if local_sql.get("status") != "ok":
        reasons.append("local_sql_execution_not_ok")
    if not bool(export.get("native_sql_matches_rerun_duckdb")):
        reasons.append("native_sql_output_does_not_match_reduced_duckdb_rerun")
    if not str(export.get("sql_path", "")).strip():
        reasons.append("missing_sql_path")
    if not str(export.get("case_path", "")).strip():
        reasons.append("missing_case_path")
    return reasons


def skip_record(export: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "family": str(export.get("family", "")),
        "case_id": str(export.get("case_id", "")),
        "reason": reason,
        "native_sql_matches_rerun_duckdb": bool(export.get("native_sql_matches_rerun_duckdb")),
        "target_reproduced": bool(export.get("target_reproduced")),
        "local_sql_status": str((export.get("local_sql_execution", {}) or {}).get("status", "")),
    }


def write_issue_dir(export: dict[str, Any], *, output_dir: Path) -> dict[str, Any]:
    family = str(export.get("family", ""))
    case_id = str(export.get("case_id", ""))
    slug = slugify(f"{family}__{case_id}")
    issue_dir = output_dir / slug
    issue_dir.mkdir(parents=True, exist_ok=True)
    sql_src = resolve_project_path(Path(str(export.get("sql_path", ""))))
    case_src = resolve_project_path(Path(str(export.get("case_path", ""))))
    sql_dst = issue_dir / "reproducer.sql"
    case_dst = issue_dir / "reduced_case.json"
    shutil.copyfile(sql_src, sql_dst)
    shutil.copyfile(case_src, case_dst)
    evidence = issue_evidence(export, sql_path=sql_dst, case_path=case_dst)
    evidence_path = issue_dir / "evidence.json"
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    issue_path = issue_dir / "upstream_issue_draft.md"
    issue_path.write_text(render_issue_draft(evidence), encoding="utf-8")
    checklist_path = issue_dir / "submission_checklist.md"
    checklist_path.write_text(render_checklist(evidence), encoding="utf-8")
    return {
        "family": family,
        "case_id": case_id,
        "issue_dir": project_relative(issue_dir),
        "upstream_issue_draft": project_relative(issue_path),
        "submission_checklist": project_relative(checklist_path),
        "sql_reproducer": project_relative(sql_dst),
        "reduced_case": project_relative(case_dst),
        "evidence_json": project_relative(evidence_path),
        "witness_summary": evidence.get("witness_summary", ""),
    }


def issue_evidence(export: dict[str, Any], *, sql_path: Path, case_path: Path) -> dict[str, Any]:
    return {
        "family": str(export.get("family", "")),
        "case_id": str(export.get("case_id", "")),
        "root_cause": str(export.get("family", "")).split("@", 1)[0],
        "backend": "duckdb",
        "sql_reproducer": project_relative(sql_path),
        "reduced_case": project_relative(case_path),
        "source_artifact": str(export.get("source_artifact", "")),
        "source_row_index": export.get("source_row_index", ""),
        "expected_finding_keys": list(export.get("expected_finding_keys", []) or []),
        "target_reproduced": bool(export.get("target_reproduced")),
        "native_sql_matches_rerun_duckdb": bool(export.get("native_sql_matches_rerun_duckdb")),
        "local_sql_execution": export.get("local_sql_execution", {}),
        "output_columns": list(export.get("output_columns", []) or []),
        "duckdb_rows": export.get("duckdb_rows", []),
        "reference_rows": export.get("reference_rows", {}),
        "witness_plan": export.get("witness_plan", {}),
        "witness_summary": witness_summary(export.get("witness_plan", {})),
        "reduced": export.get("reduced", {}),
        "original": export.get("original", {}),
        "counting_policy": "candidate latest-version DuckDB issue; count only after upstream confirmation/fix/bug label",
    }


def render_issue_draft(evidence: dict[str, Any]) -> str:
    family = str(evidence.get("family", ""))
    title = title_for_family(family)
    duckdb_rows = json.dumps(evidence.get("duckdb_rows", []), ensure_ascii=False, sort_keys=True)
    reference_rows = json.dumps(evidence.get("reference_rows", {}), ensure_ascii=False, sort_keys=True)
    local_sql = evidence.get("local_sql_execution", {}) if isinstance(evidence.get("local_sql_execution", {}), dict) else {}
    native_rows = json.dumps(
        (local_sql.get("normalized", {}) if isinstance(local_sql.get("normalized", {}), dict) else {}).get("rows", []),
        ensure_ascii=False,
        sort_keys=True,
    )
    return "\n".join(
        [
            f"# {title}",
            "",
            "## Summary",
            "",
            "DataDiffFuzz found a latest-version DuckDB wrong-result candidate where a reduced native SQL reproducer matches the DuckDB backend rerun exactly, while independent reference backends agree on a different result.",
            "",
            "This is a candidate issue report. It should not be counted as a confirmed bug until DuckDB maintainers label, acknowledge, fix, or otherwise confirm it.",
            "",
            "## Reproducer",
            "",
            f"- SQL reproducer: `{evidence.get('sql_reproducer', '')}`",
            f"- Reduced case JSON: `{evidence.get('reduced_case', '')}`",
            f"- Source row artifact: `{evidence.get('source_artifact', '')}:{evidence.get('source_row_index', '')}`",
            "",
            "Run:",
            "",
            "```bash",
            f"duckdb < {evidence.get('sql_reproducer', '')}",
            "```",
            "",
            "## Observed And Expected",
            "",
            f"- Output columns: `{', '.join(str(column) for column in evidence.get('output_columns', []) or [])}`",
            f"- DuckDB/DataDiff reduced rows: `{duckdb_rows}`",
            f"- Native SQL normalized rows: `{native_rows}`",
            f"- Reference backend rows: `{reference_rows}`",
            "",
            "## Witness",
            "",
            f"`{evidence.get('witness_summary', '')}`",
            "",
            "## DataDiffFuzz Evidence",
            "",
            f"- Family: `{family}`",
            f"- Case id: `{evidence.get('case_id', '')}`",
            f"- Target reproduced: `{str(evidence.get('target_reproduced', False)).lower()}`",
            f"- Strict native SQL match to DuckDB rerun: `{str(evidence.get('native_sql_matches_rerun_duckdb', False)).lower()}`",
            f"- Expected finding keys: `{', '.join(str(item) for item in evidence.get('expected_finding_keys', []) or [])}`",
            "",
            "## Environment",
            "",
            "This bundle was generated against the project latest DuckDB target recorded in the DataDiffFuzz final evidence set: DuckDB Python/engine 1.5.3.",
            "",
        ]
    )


def render_checklist(evidence: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Submission Checklist",
            "",
            "- [ ] Re-run `duckdb < reproducer.sql` on a clean latest DuckDB checkout or CLI.",
            "- [ ] Search DuckDB issues for the same reduced pattern/root cause.",
            "- [ ] Submit no more than one or two DuckDB reports at a time.",
            "- [ ] After upstream response, update `experiments/latest_confirmations.json` only if labeled/acknowledged/fixed.",
            "- [ ] Keep this as candidate evidence until upstream confirmation.",
            "",
            f"Family: `{evidence.get('family', '')}`",
            f"Witness: `{evidence.get('witness_summary', '')}`",
            "",
        ]
    )


def render_bundle_readme(manifest: dict[str, Any]) -> str:
    summary = manifest.get("summary", {}) if isinstance(manifest.get("summary", {}), dict) else {}
    lines = [
        "# DuckDB Issue-Ready Bundle",
        "",
        f"- Generated at: `{manifest.get('generated_at', '')}`",
        f"- Selected strict native-SQL issue candidates: `{summary.get('selected_count', 0)}`",
        f"- Skipped candidates: `{summary.get('skipped_count', 0)}`",
        "",
        "| family | case | draft | SQL | witness |",
        "| --- | --- | --- | --- | --- |",
    ]
    for issue in manifest.get("issues", []) or []:
        if not isinstance(issue, dict):
            continue
        lines.append(
            "| `{family}` | `{case}` | `{draft}` | `{sql}` | `{witness}` |".format(
                family=issue.get("family", ""),
                case=issue.get("case_id", ""),
                draft=issue.get("upstream_issue_draft", ""),
                sql=issue.get("sql_reproducer", ""),
                witness=issue.get("witness_summary", ""),
            )
        )
    skipped = manifest.get("skipped", []) if isinstance(manifest.get("skipped", []), list) else []
    if skipped:
        lines.extend(["", "## Skipped", ""])
        for item in skipped:
            lines.append(f"- `{item.get('family', '')}` `{item.get('case_id', '')}`: {item.get('reason', '')}")
    lines.append("")
    return "\n".join(lines)


def title_for_family(family: str) -> str:
    root = family.split("@", 1)[0]
    titles = {
        "grouped_topk_null_sort_key": "DuckDB drops a grouped top-k row after NULL-sensitive row-number/order processing",
        "ordering_or_limit": "DuckDB ROW_NUMBER/ORDER BY NULLS FIRST returns a different row than reference backends",
    }
    return titles.get(root, f"DuckDB wrong-result candidate for {root}")


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


def write_tarball(source_dir: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        with gzip.GzipFile(filename="", mode="wb", fileobj=handle, mtime=0) as gz_handle:
            with tarfile.open(fileobj=gz_handle, mode="w") as tar:
                for path in sorted(source_dir.rglob("*")):
                    if path.is_file():
                        tar.add(
                            path,
                            arcname=str(path.relative_to(source_dir.parent)),
                            recursive=False,
                            filter=normalize_tarinfo,
                        )


def normalize_tarinfo(tarinfo: tarfile.TarInfo) -> tarfile.TarInfo:
    tarinfo.uid = 0
    tarinfo.gid = 0
    tarinfo.uname = ""
    tarinfo.gname = ""
    tarinfo.mtime = 0
    tarinfo.mode = 0o755 if tarinfo.mode & 0o111 else 0o644
    return tarinfo


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_output_dir(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)


def resolve_project_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def resolve_output_dir(path: Path) -> Path:
    if path.is_absolute():
        return path
    if path.parent != Path("."):
        return PROJECT_ROOT / path
    return PROJECT_ROOT / path


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
    parser = argparse.ArgumentParser(description="Build issue-ready DuckDB bundles from strict native-SQL minimized reproducers.")
    parser.add_argument("--minimized-manifest", default=str(DEFAULT_MINIMIZED_MANIFEST))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--tarball", default="")
    parser.add_argument("--family", action="append", default=[])
    parser.add_argument("--allow-empty", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(main())
