from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import median
from typing import Any

from datadiff.experiment_metadata import (
    contrast_variant_id_for_suite,
    experiment_row_group_id,
    experiment_row_variant_id,
    experiment_row_variant_key,
    experiment_row_variant_label,
    is_contrast_experiment_row,
    is_contrast_variant,
    manifest_experiment_meta,
    reference_row_for_group,
    resolved_run_semantics,
)
from datadiff.reporter import latest_experiment_manifest_path
from datadiff.util import REPORTS_DIR, ensure_dirs, load_json, read_jsonl, run_meta_path

latest_experiment_manifest = latest_experiment_manifest_path


EXPECTED_ROOTS_BY_SUITE: dict[str, set[str]] = {
    "seeded_filter": {"filter_predicate"},
    "seeded_groupby": {"groupby_aggregation"},
    "seeded_join": {"join_semantics"},
    "seeded_mutate": {"arithmetic_expression", "string_expression", "type_cast"},
}

EXPECTED_BACKEND_BY_SUITE: dict[str, str] = {
    "seeded_filter": "buggy_filter",
    "seeded_groupby": "buggy_groupby",
    "seeded_join": "buggy_join",
    "seeded_mutate": "buggy_mutate",
}


def analyze_seeded_sensitivity(manifest_file: Path | None = None) -> tuple[Path, Path]:
    ensure_dirs()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest_file = manifest_file or latest_experiment_manifest_path()
    manifest = load_json(manifest_file)
    experiment_meta = manifest_experiment_meta(manifest)
    run_rows = [_summarize_run(run, experiment_meta) for run in manifest.get("runs", [])]
    aggregate_rows = _aggregate(run_rows)
    comparisons = _contrast_variant_comparisons(aggregate_rows)

    md_path = REPORTS_DIR / f"seeded-sensitivity-{manifest_file.stem}.md"
    csv_path = REPORTS_DIR / f"seeded-sensitivity-{manifest_file.stem}.csv"
    json_path = REPORTS_DIR / f"seeded-sensitivity-{manifest_file.stem}.json"
    _write_json(json_path, manifest_file, aggregate_rows, comparisons)
    _write_markdown(md_path, manifest_file, json_path, aggregate_rows, comparisons)
    _write_csv(csv_path, aggregate_rows, comparisons)
    return md_path, csv_path


def _summarize_run(run: dict[str, Any], experiment_meta: dict[str, Any]) -> dict[str, Any]:
    suite = str(run["target_suite"])
    preset = str(run["preset"])
    run_file = Path(run["run_file"])
    run_semantics = resolved_run_semantics(run, experiment_meta)
    expected_roots = EXPECTED_ROOTS_BY_SUITE.get(suite, set())
    expected_backend = EXPECTED_BACKEND_BY_SUITE.get(suite, "")
    rows = read_jsonl(run_file)
    expected_indexes = []
    candidate_indexes = []
    for idx, row in enumerate(rows):
        findings = row.get("findings") or []
        if any(_is_candidate_finding(finding) for finding in findings):
            candidate_indexes.append(idx)
        if any(_is_expected_seeded_finding(finding, expected_roots, expected_backend) for finding in findings):
            expected_indexes.append(idx)
    elapsed_s = _elapsed_seconds(run_file)
    return {
        "target_suite": suite,
        "preset": preset,
        "matrix_id": run_semantics["matrix_id"],
        "comparison_group": run_semantics["comparison_group"],
        "variant_id": run_semantics["variant_id"],
        "variant_title": run_semantics["variant_title"],
        "base_preset": run_semantics["base_preset"],
        "comparison_role": run_semantics["comparison_role"],
        "canonical_comparison_role": run_semantics["canonical_comparison_role"],
        "analysis_tags": ",".join(run_semantics["analysis_tags"]),
        "seed": int(run["seed"]),
        "cases": len(rows),
        "expected_fault_cases": len(set(expected_indexes)),
        "candidate_bug_cases": len(set(candidate_indexes)),
        "first_expected_fault_case_index": min(expected_indexes) if expected_indexes else None,
        "elapsed_s": elapsed_s,
    }


def _is_candidate_finding(finding: dict[str, Any]) -> bool:
    return finding.get("triage_verdict") == "candidate_implementation_bug"


def _is_expected_seeded_finding(finding: dict[str, Any], expected_roots: set[str], expected_backend: str) -> bool:
    suspicious = set(finding.get("suspicious_backends") or [])
    return (
        _is_candidate_finding(finding)
        and finding.get("root_cause") in expected_roots
        and expected_backend in suspicious
    )


def _elapsed_seconds(run_file: Path) -> float:
    meta_path = run_meta_path(run_file)
    if not meta_path.exists():
        return 0.0
    return float(load_json(meta_path).get("elapsed_s", 0.0) or 0.0)


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(experiment_row_variant_key(row), []).append(row)
    out = []
    for (suite, comparison_group, matrix_id, variant_id, preset), items in sorted(grouped.items()):
        cases = sum(int(item["cases"]) for item in items)
        expected = sum(int(item["expected_fault_cases"]) for item in items)
        candidates = sum(int(item["candidate_bug_cases"]) for item in items)
        elapsed_s = sum(float(item["elapsed_s"]) for item in items)
        firsts = [
            int(item["first_expected_fault_case_index"])
            for item in items
            if item["first_expected_fault_case_index"] is not None
        ]
        out.append(
            {
                "target_suite": suite,
                "preset": preset,
                "matrix_id": matrix_id,
                "comparison_group": comparison_group,
                "variant_id": variant_id,
                "variant_title": next((item["variant_title"] for item in items if item.get("variant_title")), ""),
                "base_preset": next((item["base_preset"] for item in items if item.get("base_preset")), ""),
                "comparison_role": next((item["comparison_role"] for item in items if item.get("comparison_role")), ""),
                "canonical_comparison_role": next(
                    (item["canonical_comparison_role"] for item in items if item.get("canonical_comparison_role")),
                    "",
                ),
                "analysis_tags": next((item["analysis_tags"] for item in items if item.get("analysis_tags")), ""),
                "runs": len(items),
                "cases": cases,
                "expected_fault_cases": expected,
                "expected_fault_case_rate": expected / cases if cases else 0.0,
                "expected_fault_cases_per_s": expected / elapsed_s if elapsed_s > 0 else 0.0,
                "median_first_expected_fault_case_index": median(firsts) if firsts else None,
                "candidate_bug_cases": candidates,
                "candidate_bug_case_rate": candidates / cases if cases else 0.0,
            }
        )
    return out


def _contrast_variant_comparisons(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(experiment_row_group_id(row), []).append(row)

    comparisons = []
    for (_, _, _), group_rows in sorted(grouped.items()):
        reference_row = reference_row_for_group(group_rows)
        if reference_row is None:
            continue
        contrast_rows = [
            row
            for row in group_rows
            if experiment_row_variant_id(row) != experiment_row_variant_id(reference_row)
            and is_contrast_experiment_row(row)
            and is_contrast_variant(row)
        ]
        if not contrast_rows:
            suite = str(reference_row.get("target_suite", "") or "")
            contrast_variant = contrast_variant_id_for_suite(suite)
            contrast_rows = [
                row
                for row in group_rows
                if (
                    str(row.get("variant_id", "") or row.get("preset", "") or "") == contrast_variant
                    or str(row.get("preset", "") or "") == contrast_variant
                )
            ]
        for contrast in contrast_rows:
            comparisons.append(
                {
                    "target_suite": contrast["target_suite"],
                    "matrix_id": contrast.get("matrix_id", ""),
                    "comparison_group": contrast.get("comparison_group", ""),
                    "variant_id": experiment_row_variant_id(contrast),
                    "variant_label": experiment_row_variant_label(contrast),
                    "preset": contrast["preset"],
                    "comparison_role": contrast.get("comparison_role", ""),
                    "canonical_comparison_role": contrast.get("canonical_comparison_role", ""),
                    "reference_variant_id": experiment_row_variant_id(reference_row),
                    "reference_variant_label": experiment_row_variant_label(reference_row),
                    "reference_preset": reference_row.get("preset", ""),
                    "reference_expected_fault_case_rate": reference_row["expected_fault_case_rate"],
                    "expected_fault_case_rate": contrast["expected_fault_case_rate"],
                    "expected_fault_case_rate_delta": contrast["expected_fault_case_rate"]
                    - reference_row["expected_fault_case_rate"],
                    "expected_fault_case_rate_ratio": _ratio(
                        contrast["expected_fault_case_rate"],
                        reference_row["expected_fault_case_rate"],
                    ),
                    "baseline_median_first_expected": reference_row["median_first_expected_fault_case_index"],
                    "median_first_expected": contrast["median_first_expected_fault_case_index"],
                    "first_expected_delta": _optional_delta(
                        contrast["median_first_expected_fault_case_index"],
                        reference_row["median_first_expected_fault_case_index"],
                    ),
                }
            )
    return comparisons


def _write_markdown(
    path: Path,
    manifest_file: Path,
    json_path: Path,
    aggregate_rows: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
) -> None:
    lines = [
        "# Seeded Fault Sensitivity Analysis",
        "",
        f"- Manifest: `{manifest_file}`",
        f"- Aggregate JSON: `{json_path}`",
        "- Expected roots are counted separately from all candidate findings.",
        "",
        "## Contrast Variant Comparisons",
        "",
        "| target suite | variant | expected fault case % | delta | ratio | median first expected | first delta |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in comparisons:
        lines.append(
            "| {suite} | {variant} | {rate} | {delta} | {ratio} | {first} | {first_delta} |".format(
                suite=row["target_suite"],
                variant=row["variant_id"],
                rate=_fmt_percent(row["expected_fault_case_rate"]),
                delta=_fmt_signed_percent(row["expected_fault_case_rate_delta"]),
                ratio=_fmt_ratio(row["expected_fault_case_rate_ratio"]),
                first=_fmt_optional(row["median_first_expected"]),
                first_delta=_fmt_optional_signed(row["first_expected_delta"]),
            )
        )
    lines.extend(
        [
            "",
            "## Aggregate Rows",
            "",
            "| target suite | variant | preset | cases | expected fault cases | expected fault case % | expected cases/s | median first expected | all candidate case % |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in aggregate_rows:
        lines.append(
            "| {suite} | {variant} | {preset} | {cases} | {expected} | {rate} | {per_s} | {first} | {candidate_rate} |".format(
                suite=row["target_suite"],
                variant=experiment_row_variant_label(row),
                preset=row["preset"],
                cases=row["cases"],
                expected=row["expected_fault_cases"],
                rate=_fmt_percent(row["expected_fault_case_rate"]),
                per_s=_fmt_float(row["expected_fault_cases_per_s"]),
                first=_fmt_optional(row["median_first_expected_fault_case_index"]),
                candidate_rate=_fmt_percent(row["candidate_bug_case_rate"]),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_json(
    path: Path,
    manifest_file: Path,
    aggregate_rows: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
) -> None:
    payload = {
        "manifest": str(manifest_file),
        "variant_rows": aggregate_rows,
        "contrast_variant_comparisons": comparisons,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, aggregate_rows: list[dict[str, Any]], comparisons: list[dict[str, Any]]) -> None:
    rows = [{**row, "row_type": "aggregate"} for row in aggregate_rows]
    rows.extend({**_comparison_csv_row(row), "row_type": "contrast_comparison"} for row in comparisons)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _comparison_csv_row(row: dict[str, Any]) -> dict[str, Any]:
    exported = dict(row)
    exported["canonical_row_type"] = "contrast_variant_comparison"
    exported["comparison_row_type"] = "contrast_comparison"
    exported["legacy_row_type"] = "targeted_comparison"
    exported["contrast_variant"] = exported.get("variant_id", "")
    exported["contrast_variant_id"] = exported.get("variant_id", "")
    exported["contrast_variant_label"] = exported.get("variant_label", exported.get("variant_id", ""))
    exported["contrast_preset"] = exported.get("preset", "")
    exported["contrast_comparison_role"] = exported.get("comparison_role", "")
    exported["contrast_expected_fault_case_rate"] = exported.get("expected_fault_case_rate", 0.0)
    exported["contrast_median_first_expected"] = exported.get("median_first_expected")
    exported["reference_variant"] = exported.get("reference_variant_id", "")
    exported["reference_variant_id"] = exported.get("reference_variant_id", "")
    exported["reference_variant_label"] = exported.get("reference_variant_label", "")
    exported["reference_expected_fault_case_rate"] = exported.get("reference_expected_fault_case_rate", 0.0)
    exported["reference_selector_preset"] = exported.get("reference_preset", "")
    exported["baseline_preset"] = exported.get("reference_preset", "")
    exported["baseline_expected_fault_case_rate"] = exported.get("reference_expected_fault_case_rate", 0.0)
    exported["targeted_variant"] = exported.get("variant_id", "")
    exported["targeted_variant_id"] = exported.get("variant_id", "")
    exported["targeted_preset"] = exported.get("preset", "")
    exported["targeted_comparison_role"] = exported.get("comparison_role", "")
    exported["targeted_expected_fault_case_rate"] = exported.get("expected_fault_case_rate", 0.0)
    exported["targeted_median_first_expected"] = exported.get("median_first_expected")
    return exported


def _ratio(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _optional_delta(left: float | int | None, right: float | int | None) -> float | None:
    if left is None or right is None:
        return None
    return float(left) - float(right)


def _fmt_percent(value: float) -> str:
    return f"{value:.1%}"


def _fmt_signed_percent(value: float) -> str:
    return f"{value:+.1%}"


def _fmt_float(value: float) -> str:
    return f"{value:.2f}"


def _fmt_ratio(value: float | None) -> str:
    return "" if value is None else f"{value:.2f}x"


def _fmt_optional(value: float | int | None) -> str:
    return "" if value is None else f"{value:g}"


def _fmt_optional_signed(value: float | int | None) -> str:
    return "" if value is None else f"{value:+g}"
