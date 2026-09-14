#!/usr/bin/env python3
"""Build the paper-facing bug audit from the confirmation ledger, the canonical
confirmed-bug corpus, and curated audit metadata (A1/A2/A3).

Outputs:
  paper/experiments/results/bug_audit.json
  paper/experiments/results/bug_audit.md

Usage:
  python scripts/paper/build_bug_audit.py
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

_SEVERITY_WEIGHT = {"high": 3, "medium": 2, "low": 1}

_BACKEND_FAMILY = {
    "apache/datafusion": "query engine",
    "pola-rs/polars": "DataFrame API",
    "apache/arrow": "Arrow compute",
    "duckdb/duckdb": "embedded SQL",
}


def _load(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _project_from_url(url: str) -> str:
    # https://github.com/<org>/<repo>/issues/<n>
    parts = url.split("github.com/", 1)[-1].split("/")
    return f"{parts[0]}/{parts[1]}" if len(parts) >= 2 else url


def _still_affected(root: dict) -> bool | None:
    obs = root.get("version_observations") or []
    if not obs:
        return None
    return bool(obs[-1].get("expected_bug_present"))


def build(confirmations_path: Path, corpus_path: Path, metadata_path: Path) -> dict:
    confirmations = _load(confirmations_path)["confirmations"]
    corpus = _load(corpus_path)
    metadata = _load(metadata_path)

    by_issue = {c.get("issue_url"): c for c in confirmations}

    records: list[dict] = []
    for root in corpus["confirmed_roots"]:
        issue = root["issue_url"]
        ledger = by_issue.get(issue, {})
        meta = metadata["roots"].get(root["root_id"], {})
        project = root.get("project") or _project_from_url(issue)
        affected = root.get("affected_versions") or []
        fixed = root.get("fixed_versions") or []
        records.append(
            {
                "root_id": root["root_id"],
                "families": root.get("observed_families", []),
                "project": project,
                "backend_family": _BACKEND_FAMILY.get(project, "other"),
                "issue_url": issue,
                "upstream_status": ledger.get("upstream_status", root.get("ledger_status")),
                "state": root.get("state"),
                "labels": ledger.get("labels", []),
                "severity": meta.get("severity"),
                "severity_weight": _SEVERITY_WEIGHT.get(meta.get("severity", ""), 0),
                "impact_class": meta.get("impact_class"),
                "root_cause_group": meta.get("root_cause_group"),
                "first_violated_component": meta.get("first_violated_component"),
                "shared_subsystem_with": meta.get("shared_subsystem_with", []),
                "independence_note": meta.get("independence_note"),
                "affected_versions": affected,
                "fixed_versions": fixed,
                "fixed_by": root.get("fixed_by"),
                "still_affected_on_latest": _still_affected(root),
                "fault_model": root.get("fault_model"),
                "expected_signature": root.get("expected_signature"),
                "dsl_case": root.get("dsl_case"),
            }
        )

    # stable order: still-affected first, then severity weight, then project
    records.sort(
        key=lambda r: (
            not bool(r["still_affected_on_latest"]),
            -int(r["severity_weight"]),
            r["project"],
            r["root_id"],
        )
    )

    pending = []
    for root in corpus.get("pending_roots", []):
        issue = root["issue_url"]
        ledger = by_issue.get(issue, {})
        pending.append(
            {
                "root_id": root["root_id"],
                "families": root.get("observed_families", []),
                "project": root.get("project") or _project_from_url(issue),
                "issue_url": issue,
                "upstream_status": ledger.get("upstream_status", root.get("ledger_status")),
                "state": root.get("state"),
                "counts_as_confirmed": False,
            }
        )

    excluded = [
        {
            "family": family,
            "issue_url": info.get("issue_url"),
            "reason": info.get("reason"),
            "counts_as_confirmed": False,
        }
        for family, info in metadata.get("excluded", {}).items()
    ]

    by_backend = Counter(r["backend_family"] for r in records)
    by_project = Counter(r["project"] for r in records)
    by_severity = Counter(r["severity"] for r in records)
    by_group = Counter(r["root_cause_group"] for r in records)
    still_affected = sum(1 for r in records if r["still_affected_on_latest"])
    fixed_count = sum(1 for r in records if r["fixed_by"])

    summary = {
        "schema_version": "datadiff-bug-audit-v1",
        "confirmed_roots": len(records),
        "still_affected_on_latest": still_affected,
        "fixed_upstream": fixed_count,
        "severity_weighted_score": sum(int(r["severity_weight"]) for r in records),
        "distinct_root_cause_groups": len(by_group),
        "by_backend_family": dict(by_backend),
        "by_project": dict(by_project),
        "by_severity": dict(by_severity),
        "by_root_cause_group": dict(by_group),
        "pending_roots": len(pending),
        "excluded_records": len(excluded),
    }
    return {"summary": summary, "roots": records, "pending": pending, "excluded": excluded}


def render_markdown(audit: dict) -> str:
    s = audit["summary"]
    lines = [
        "# Bug audit (generated)",
        "",
        "Generated by `scripts/paper/build_bug_audit.py` from "
        "`experiments/latest_confirmations.json`, "
        "`experiments/canonical_confirmed_bug_corpus/v2/manifest.json`, and "
        "`experiments/bug_audit_metadata.json`. Do not edit by hand.",
        "",
        "## Summary",
        "",
        f"- Confirmed roots: **{s['confirmed_roots']}**",
        f"- Still affected on latest target: **{s['still_affected_on_latest']}**",
        f"- Fixed upstream: **{s['fixed_upstream']}**",
        f"- Severity-weighted score: **{s['severity_weighted_score']}**",
        f"- Distinct root-cause groups (coarse): **{s['distinct_root_cause_groups']}**",
        f"- Backend families: {s['by_backend_family']}",
        f"- Projects: {s['by_project']}",
        f"- Severity: {s['by_severity']}",
        f"- Pending (excluded from confirmed): {s['pending_roots']}",
        f"- Upstream-invalid / duplicate (excluded): {s['excluded_records']}",
        "",
        "## Confirmed roots",
        "",
        "| # | root | family | backend | upstream | severity | impact | still affected | fixed by | first violated component |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for i, r in enumerate(audit["roots"], 1):
        fam = ", ".join(r["families"])
        fixed_by = r["fixed_by"] or ""
        affected = "yes" if r["still_affected_on_latest"] else "no"
        lines.append(
            f"| {i} | `{r['root_id']}` | {fam} | {r['backend_family']} | "
            f"{r['upstream_status']} | {r['severity']} | {r['impact_class']} | "
            f"{affected} | {fixed_by} | {r['first_violated_component']} |"
        )
    lines += ["", "## Pending roots (not counted)", ""]
    for r in audit["pending"]:
        lines.append(
            f"- `{r['root_id']}` — {', '.join(r['families'])} — {r['upstream_status']} — {r['issue_url']}"
        )
    lines += ["", "## Excluded records", ""]
    for r in audit["excluded"]:
        lines.append(f"- {r['family']} — {r['reason']} — {r['issue_url']}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--confirmations",
        type=Path,
        default=REPO_ROOT / "experiments/latest_confirmations.json",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=REPO_ROOT / "experiments/canonical_confirmed_bug_corpus/v2/manifest.json",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=REPO_ROOT / "experiments/bug_audit_metadata.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "paper/experiments/results",
    )
    args = parser.parse_args()

    audit = build(args.confirmations, args.corpus, args.metadata)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "bug_audit.json"
    md_path = args.out_dir / "bug_audit.md"
    json_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(audit), encoding="utf-8")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    print(json.dumps(audit["summary"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
