from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from datadiff.dsl import normalize_sort_keys
from datadiff.reporter import latest_experiment_manifest
from datadiff.util import REPORTS_DIR, ensure_dirs, load_json, read_jsonl


def analyze_pattern_variants(
    manifest_file: Path | None = None,
    *,
    pattern: str = "null_agg_topk",
) -> tuple[Path, Path]:
    ensure_dirs()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest_file = manifest_file or latest_experiment_manifest()
    if pattern != "null_agg_topk":
        raise ValueError(f"unsupported pattern: {pattern}")
    manifest = load_json(manifest_file)
    rows = _collect_null_agg_topk_rows(manifest)

    md_path = REPORTS_DIR / f"pattern-variants-{pattern}-{manifest_file.stem}.md"
    csv_path = REPORTS_DIR / f"pattern-variants-{pattern}-{manifest_file.stem}.csv"
    _write_csv(csv_path, rows)
    _write_markdown(md_path, manifest_file, pattern, rows)
    return md_path, csv_path


def _collect_null_agg_topk_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str, str, str], Counter[str]] = defaultdict(Counter)
    family_counts: dict[tuple[str, str, str, str], Counter[str]] = defaultdict(Counter)
    for run in manifest.get("runs", []):
        run_file = Path(run["run_file"])
        target_suite = str(run.get("target_suite", ""))
        preset = str(run.get("preset", ""))
        for row in read_jsonl(run_file):
            variant = _null_agg_topk_variant(row.get("case", {}))
            if variant is None:
                continue
            key = (target_suite, preset, variant["agg_func"], variant["sort_direction"])
            bucket = buckets[key]
            bucket["cases"] += 1
            findings = row.get("findings", [])
            if findings:
                bucket["finding_cases"] += 1
            candidate_findings = [
                finding for finding in findings
                if finding.get("triage_verdict") == "candidate_implementation_bug"
                and not finding.get("false_positive")
            ]
            if candidate_findings:
                bucket["candidate_cases"] += 1
                family_counts[key].update(_candidate_bug_family_keys(candidate_findings))
            bucket["candidate_findings"] += len(candidate_findings)
            bucket["false_positive_findings"] += sum(1 for finding in findings if finding.get("false_positive"))
            bucket["semantic_divergence_findings"] += sum(
                1
                for finding in findings
                if str(finding.get("triage_verdict", "")).endswith("semantic_divergence")
                or finding.get("triage_verdict") in {"documented_semantic_divergence", "expected_semantic_divergence"}
            )
    rows = []
    for key in sorted(buckets):
        target_suite, preset, agg_func, sort_direction = key
        counts = buckets[key]
        rows.append(
            {
                "target_suite": target_suite,
                "preset": preset,
                "agg_func": agg_func,
                "sort_direction": sort_direction,
                "cases": counts["cases"],
                "finding_cases": counts["finding_cases"],
                "candidate_cases": counts["candidate_cases"],
                "candidate_findings": counts["candidate_findings"],
                "semantic_divergence_findings": counts["semantic_divergence_findings"],
                "false_positive_findings": counts["false_positive_findings"],
                "top_candidate_bug_families": _format_counter(family_counts[key]),
            }
        )
    return rows


def _null_agg_topk_variant(case: dict[str, Any]) -> dict[str, str] | None:
    ops = case.get("program", {}).get("operations", [])
    for groupby_idx, op in enumerate(ops):
        if op.get("op") != "groupby":
            continue
        aggs = op.get("aggs", [])
        if len(aggs) != 1:
            continue
        agg = aggs[0]
        alias = str(agg.get("as", ""))
        if not alias:
            continue
        for sort_idx in range(groupby_idx + 1, len(ops)):
            sort = ops[sort_idx]
            if sort.get("op") != "sort":
                continue
            try:
                sort_keys = normalize_sort_keys(sort)
            except ValueError:
                continue
            matching_key = next((key for key in sort_keys if key.column == alias), None)
            if matching_key is None:
                continue
            if not any(later.get("op") == "limit" for later in ops[sort_idx + 1 :]):
                continue
            return {
                "agg_func": str(agg.get("func", "unknown")),
                "sort_direction": "asc" if matching_key.ascending else "desc",
            }
    return None


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "target_suite",
        "preset",
        "agg_func",
        "sort_direction",
        "cases",
        "finding_cases",
        "candidate_cases",
        "candidate_findings",
        "semantic_divergence_findings",
        "false_positive_findings",
        "top_candidate_bug_families",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, manifest_file: Path, pattern: str, rows: list[dict[str, Any]]) -> None:
    total_cases = sum(int(row["cases"]) for row in rows)
    total_candidates = sum(int(row["candidate_cases"]) for row in rows)
    total_false_positives = sum(int(row["false_positive_findings"]) for row in rows)
    lines = [
        "# DataDiffFuzz Pattern Variant Analysis",
        "",
        f"- Manifest: `{manifest_file}`",
        f"- Pattern: `{pattern}`",
        f"- Cases: {total_cases}",
        f"- Candidate cases: {total_candidates}",
        f"- False-positive findings: {total_false_positives}",
        "",
        "| target suite | preset | agg | sort | cases | candidate cases | false positives | top families |",
        "|---|---|---|---|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {target_suite} | {preset} | {agg_func} | {sort_direction} | {cases} | {candidate_cases} | {false_positive_findings} | {top_candidate_bug_families} |".format(
                **row
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_counter(counter: Counter[str]) -> str:
    if not counter:
        return "none"
    return "; ".join(f"{family}:{count}" for family, count in counter.most_common())


def _candidate_bug_family_keys(findings: list[dict[str, Any]]) -> Counter[str]:
    keys: Counter[str] = Counter()
    root_by_suspicious: dict[str, str] = {}
    for finding in findings:
        root = str(finding.get("root_cause", "unknown"))
        suspicious = _suspicious_key(finding)
        if not root.startswith("metamorphic_"):
            root_by_suspicious.setdefault(suspicious, root)
    for finding in findings:
        root = str(finding.get("root_cause", "unknown"))
        suspicious = _suspicious_key(finding)
        if root.startswith("metamorphic_") and suspicious in root_by_suspicious:
            root = root_by_suspicious[suspicious]
        keys[f"{root}@{suspicious}"] += 1
    return keys


def _suspicious_key(finding: dict[str, Any]) -> str:
    return ",".join(sorted(finding.get("suspicious_backends", []) or [])) or "unknown"
