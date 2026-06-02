from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from datadiff.experiment_metadata import (
    experiment_row_group_id,
    experiment_row_variant_id,
    experiment_row_variant_key,
    experiment_row_variant_label,
    is_contrast_variant,
    reference_row_for_group,
    row_string_value,
)
from datadiff.reporter import latest_experiment_manifest_path, write_experiment_summary_report
from datadiff.util import REPORTS_DIR, ensure_dirs

latest_experiment_manifest = latest_experiment_manifest_path
write_experiment_summary = write_experiment_summary_report


def analyze_experiment(
    manifest_file: Path | None = None,
    *,
    reference_preset: str = "baseline",
    legacy_reference_preset: str | None = None,
    baseline_preset: str | None = None,
    compare_presets: list[str] | None = None,
    refresh: bool = False,
) -> tuple[Path, Path]:
    ensure_dirs()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest_file = manifest_file or latest_experiment_manifest_path()
    effective_reference_preset = legacy_reference_preset or baseline_preset or reference_preset
    summary_md, _ = write_experiment_summary(manifest_file, refresh=refresh)
    aggregate_csv = summary_md.with_name(f"{summary_md.stem}-aggregates.csv")
    aggregate_json = summary_md.with_name(f"{summary_md.stem}-aggregates.json")
    aggregate_rows = _load_aggregate_rows(aggregate_json, aggregate_csv)
    comparisons = _build_variant_comparisons(aggregate_rows, effective_reference_preset, compare_presets)

    md_path = REPORTS_DIR / f"experiment-analysis-{manifest_file.stem}.md"
    csv_path = REPORTS_DIR / f"experiment-analysis-{manifest_file.stem}.csv"
    _write_analysis_markdown(
        md_path,
        manifest_file,
        aggregate_csv,
        aggregate_json,
        comparisons,
        aggregate_rows,
        effective_reference_preset,
    )
    _write_analysis_csv(csv_path, comparisons)
    return md_path, csv_path


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_aggregate_rows(aggregate_json: Path, aggregate_csv: Path) -> list[dict[str, Any]]:
    if aggregate_json.is_file():
        payload = json.loads(aggregate_json.read_text(encoding="utf-8"))
        variant_rows = payload.get("variant_rows", []) if isinstance(payload, dict) else []
        if isinstance(variant_rows, list) and all(isinstance(item, dict) for item in variant_rows):
            return variant_rows
    return _load_rows(aggregate_csv)


def _build_variant_comparisons(
    rows: list[dict[str, str]],
    reference_preset: str,
    compare_presets: list[str] | None,
) -> list[dict[str, Any]]:
    rows_by_variant = {experiment_row_variant_key(row): row for row in rows}
    group_ids = sorted({experiment_row_group_id(row) for row in rows})
    comparisons: list[dict[str, Any]] = []
    for suite, comparison_group, matrix_id in group_ids:
        group_rows = [
            row
            for row in rows
            if experiment_row_group_id(row) == (suite, comparison_group, matrix_id)
        ]
        reference_row = reference_row_for_group(group_rows, fallback_preset=reference_preset)
        if reference_row is None:
            continue
        selected_variant_keys = _selected_variant_keys(group_rows, reference_row, compare_presets)
        for variant_key in selected_variant_keys:
            comparison_row = rows_by_variant.get(variant_key)
            if comparison_row is None:
                continue
            comparisons.append(_build_variant_comparison(suite, reference_preset, reference_row, comparison_row))
    return comparisons


def _build_variant_comparison(
    suite: str,
    reference_preset: str,
    reference_row: dict[str, str],
    comparison_row: dict[str, str],
) -> dict[str, Any]:
    comparison_semantics = {
        key: row_string_value(comparison_row, key)
        for key in (
            "matrix_id",
            "comparison_group",
            "variant_id",
            "variant_title",
            "base_preset",
            "comparison_role",
            "canonical_comparison_role",
            "component_focus",
            "scope_kind",
            "oracle_profile",
            "rq_tags",
            "analysis_tags",
            "preset",
        )
    }
    reference_semantics = {
        key: row_string_value(reference_row, key)
        for key in ("variant_title", "preset")
    }
    is_contrast_run = is_contrast_variant(comparison_row)
    reference_variant_id = experiment_row_variant_id(reference_row)
    comparison_variant_id = experiment_row_variant_id(comparison_row)
    reference_variant_label = experiment_row_variant_label(reference_row)
    comparison_variant_label = experiment_row_variant_label(comparison_row)
    reference_rate = _float(reference_row, "candidate_bug_case_rate")
    current_rate = _float(comparison_row, "candidate_bug_case_rate")
    reference_yield = _float(reference_row, "candidate_bug_cases_per_s")
    current_yield = _float(comparison_row, "candidate_bug_cases_per_s")
    reference_first = _optional_float(reference_row.get("median_first_candidate_bug_case_index"))
    current_first = _optional_float(comparison_row.get("median_first_candidate_bug_case_index"))
    reference_first_s = _optional_float(reference_row.get("median_first_candidate_bug_elapsed_s"))
    current_first_s = _optional_float(comparison_row.get("median_first_candidate_bug_elapsed_s"))
    reference_auc = _float(reference_row, "avg_candidate_bug_discovery_auc")
    current_auc = _float(comparison_row, "avg_candidate_bug_discovery_auc")
    reference_space = _float(reference_row, "evidence_bytes_per_case")
    current_space = _float(comparison_row, "evidence_bytes_per_case")
    reference_run_log_space = _float(reference_row, "run_log_bytes_per_case")
    current_run_log_space = _float(comparison_row, "run_log_bytes_per_case")
    reference_artifact_space = _float(reference_row, "artifact_bytes_per_case")
    current_artifact_space = _float(comparison_row, "artifact_bytes_per_case")
    return {
        "target_suite": suite,
        "matrix_id": comparison_semantics["matrix_id"],
        "comparison_group": comparison_semantics["comparison_group"],
        "variant_id": comparison_semantics["variant_id"],
        "variant_title": comparison_semantics["variant_title"],
        "variant_label": comparison_variant_label,
        "base_preset": comparison_semantics["base_preset"],
        "comparison_role": comparison_semantics["comparison_role"],
        "canonical_comparison_role": comparison_semantics["canonical_comparison_role"],
        "component_focus": comparison_semantics["component_focus"],
        "scope_kind": comparison_semantics["scope_kind"],
        "oracle_profile": comparison_semantics["oracle_profile"],
        "rq_tags": comparison_semantics["rq_tags"],
        "analysis_tags": comparison_semantics["analysis_tags"],
        "reference_variant_id": reference_variant_id,
        "reference_variant_title": reference_semantics["variant_title"],
        "reference_variant_label": reference_variant_label,
        "reference_variant_preset": reference_semantics["preset"],
        "comparison_variant_id": comparison_variant_id,
        "comparison_variant_label": comparison_variant_label,
        "reference_preset": reference_preset,
        "preset": comparison_semantics["preset"],
        "is_contrast_run": is_contrast_run,
        "is_contrast_variant": is_contrast_run,
        "is_contrast_comparison": is_contrast_run,
        "is_targeted_variant": is_contrast_run,
        "is_targeted_comparison": is_contrast_run,
        "reference_candidate_bug_case_rate": reference_rate,
        "candidate_bug_case_rate": current_rate,
        "candidate_bug_case_rate_delta": current_rate - reference_rate,
        "candidate_bug_case_rate_ratio": _ratio(current_rate, reference_rate),
        "reference_candidate_bug_cases_per_s": reference_yield,
        "candidate_bug_cases_per_s": current_yield,
        "candidate_bug_cases_per_s_delta": current_yield - reference_yield,
        "candidate_bug_cases_per_s_ratio": _ratio(current_yield, reference_yield),
        "reference_median_first_candidate": reference_first,
        "median_first_candidate": current_first,
        "first_candidate_delta": _optional_delta(current_first, reference_first),
        "reference_median_first_candidate_s": reference_first_s,
        "median_first_candidate_s": current_first_s,
        "first_candidate_s_delta": _optional_delta(current_first_s, reference_first_s),
        "reference_discovery_auc": reference_auc,
        "discovery_auc": current_auc,
        "discovery_auc_delta": current_auc - reference_auc,
        "reference_evidence_bytes_per_case": reference_space,
        "evidence_bytes_per_case": current_space,
        "evidence_bytes_per_case_delta": current_space - reference_space,
        "evidence_bytes_per_case_ratio": _ratio(current_space, reference_space),
        "reference_run_log_bytes_per_case": reference_run_log_space,
        "run_log_bytes_per_case": current_run_log_space,
        "reference_artifact_bytes_per_case": reference_artifact_space,
        "artifact_bytes_per_case": current_artifact_space,
        "reference_cases": int(float(reference_row.get("cases", "0") or 0)),
        "cases": int(float(comparison_row.get("cases", "0") or 0)),
        "reference_candidate_bug_cases": int(float(reference_row.get("candidate_bug_cases", "0") or 0)),
        "candidate_bug_cases": int(float(comparison_row.get("candidate_bug_cases", "0") or 0)),
    }


def _selected_variant_keys(
    rows: list[dict[str, str]],
    reference_row: dict[str, str],
    compare_presets: list[str] | None,
) -> list[tuple[str, str, str, str, str]]:
    reference_key = experiment_row_variant_key(reference_row)
    if compare_presets:
        compare_set = set(compare_presets)
        selected: list[tuple[str, str, str, str, str]] = []
        for row in rows:
            variant_key = experiment_row_variant_key(row)
            if variant_key == reference_key:
                continue
            if row.get("preset", "") in compare_set or experiment_row_variant_id(row) in compare_set:
                selected.append(variant_key)
        return sorted(set(selected))
    structured_contrast_keys = {
        experiment_row_variant_key(row)
        for row in rows
        if experiment_row_variant_key(row) != reference_key and is_contrast_variant(row)
    }
    if structured_contrast_keys:
        return sorted(structured_contrast_keys)
    return sorted(
        {
            experiment_row_variant_key(row)
            for row in rows
            if experiment_row_variant_key(row) != reference_key
        }
    )


def _write_analysis_markdown(
    path: Path,
    manifest_file: Path,
    aggregate_csv: Path,
    aggregate_json: Path,
    comparisons: list[dict[str, Any]],
    aggregate_rows: list[dict[str, Any]],
    reference_preset: str,
) -> None:
    lines = [
        "# DataDiffFuzz Experiment Analysis",
        "",
        f"- Manifest: `{manifest_file}`",
        f"- Aggregate CSV: `{aggregate_csv}`",
        f"- Aggregate JSON: `{aggregate_json}`",
        f"- Reference selector fallback preset: `{reference_preset}`",
        "",
        "## Soundness Snapshot",
        "",
        _soundness_sentence(aggregate_rows),
        "",
        "## Contrast Variant Comparisons",
        "",
        "| target suite | variant | candidate case % | rate delta | rate ratio | candidate cases/s | yield delta | yield ratio | median first | median first s | discovery AUC |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparisons:
        if not row["is_contrast_run"]:
            continue
        lines.append(_comparison_markdown_row(row))
    lines.extend(
        [
            "",
            "## Space Efficiency Contrasts",
            "",
            "| target suite | variant | evidence bytes/case | bytes/case delta | bytes/case ratio | run log bytes/case | artifact bytes/case |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in comparisons:
        lines.append(
            "| {target_suite} | {variant} | {space} | {space_delta} | {space_ratio} | {run_log_space} | {artifact_space} |".format(
                target_suite=row["target_suite"],
                variant=row["comparison_variant_label"],
                space=_fmt_float(row["evidence_bytes_per_case"]),
                space_delta=_fmt_signed_float(row["evidence_bytes_per_case_delta"]),
                space_ratio=_fmt_ratio(row["evidence_bytes_per_case_ratio"]),
                run_log_space=_fmt_float(row["run_log_bytes_per_case"]),
                artifact_space=_fmt_float(row["artifact_bytes_per_case"]),
            )
        )
    lines.extend(
        [
            "",
            "## All Reference-Referenced Comparisons",
            "",
            "| target suite | variant | contrast-guided | candidate case % | rate ratio | candidate cases/s | yield ratio | median first | median first s | discovery AUC |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in comparisons:
        lines.append(
            "| {target_suite} | {variant} | {contrast_guided} | {rate} | {rate_ratio} | {yield_} | {yield_ratio} | {first} | {first_s} | {auc} |".format(
                target_suite=row["target_suite"],
                variant=row["comparison_variant_label"],
                contrast_guided="yes" if row["is_contrast_run"] else "no",
                rate=_fmt_percent(row["candidate_bug_case_rate"]),
                rate_ratio=_fmt_ratio(row["candidate_bug_case_rate_ratio"]),
                yield_=_fmt_float(row["candidate_bug_cases_per_s"]),
                yield_ratio=_fmt_ratio(row["candidate_bug_cases_per_s_ratio"]),
                first=_fmt_optional(row["median_first_candidate"]),
                first_s=_fmt_optional(row["median_first_candidate_s"]),
                auc=_fmt_float(row["discovery_auc"]),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_analysis_csv(path: Path, comparisons: list[dict[str, Any]]) -> None:
    fieldnames = [
        "target_suite",
        "matrix_id",
        "comparison_group",
        "variant_id",
        "variant_title",
        "variant_label",
        "base_preset",
        "comparison_role",
        "canonical_comparison_role",
        "component_focus",
        "scope_kind",
        "oracle_profile",
        "rq_tags",
        "analysis_tags",
        "reference_variant_id",
        "reference_variant_title",
        "reference_variant_label",
        "reference_variant_preset",
        "baseline_variant_id",
        "baseline_variant_title",
        "baseline_variant_label",
        "baseline_variant_preset",
        "comparison_variant_id",
        "comparison_variant_label",
        "reference_preset",
        "reference_selector_preset",
        "baseline_preset",
        "preset",
        "is_contrast_run",
        "is_contrast_variant",
        "is_contrast_comparison",
        "is_contrast_preset",
        "is_targeted_variant",
        "is_targeted_comparison",
        "is_targeted_preset",
        "reference_candidate_bug_case_rate",
        "baseline_candidate_bug_case_rate",
        "candidate_bug_case_rate",
        "candidate_bug_case_rate_delta",
        "candidate_bug_case_rate_ratio",
        "reference_candidate_bug_cases_per_s",
        "baseline_candidate_bug_cases_per_s",
        "candidate_bug_cases_per_s",
        "candidate_bug_cases_per_s_delta",
        "candidate_bug_cases_per_s_ratio",
        "reference_median_first_candidate",
        "baseline_median_first_candidate",
        "median_first_candidate",
        "first_candidate_delta",
        "reference_median_first_candidate_s",
        "baseline_median_first_candidate_s",
        "median_first_candidate_s",
        "first_candidate_s_delta",
        "reference_discovery_auc",
        "baseline_discovery_auc",
        "discovery_auc",
        "discovery_auc_delta",
        "reference_evidence_bytes_per_case",
        "baseline_evidence_bytes_per_case",
        "evidence_bytes_per_case",
        "evidence_bytes_per_case_delta",
        "evidence_bytes_per_case_ratio",
        "reference_run_log_bytes_per_case",
        "baseline_run_log_bytes_per_case",
        "run_log_bytes_per_case",
        "reference_artifact_bytes_per_case",
        "baseline_artifact_bytes_per_case",
        "artifact_bytes_per_case",
        "reference_cases",
        "baseline_cases",
        "cases",
        "reference_candidate_bug_cases",
        "baseline_candidate_bug_cases",
        "candidate_bug_cases",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(_export_comparison_row(row) for row in comparisons)


def _export_comparison_row(row: dict[str, Any]) -> dict[str, Any]:
    exported = dict(row)
    exported["baseline_variant_id"] = exported.get("reference_variant_id", "")
    exported["baseline_variant_title"] = exported.get("reference_variant_title", "")
    exported["baseline_variant_label"] = exported.get("reference_variant_label", "")
    exported["baseline_variant_preset"] = exported.get("reference_variant_preset", "")
    exported["baseline_preset"] = exported.get("reference_preset", "")
    exported["baseline_candidate_bug_case_rate"] = exported.get("reference_candidate_bug_case_rate", 0.0)
    exported["baseline_candidate_bug_cases_per_s"] = exported.get("reference_candidate_bug_cases_per_s", 0.0)
    exported["baseline_median_first_candidate"] = exported.get("reference_median_first_candidate")
    exported["baseline_median_first_candidate_s"] = exported.get("reference_median_first_candidate_s")
    exported["baseline_discovery_auc"] = exported.get("reference_discovery_auc", 0.0)
    exported["baseline_evidence_bytes_per_case"] = exported.get("reference_evidence_bytes_per_case", 0.0)
    exported["baseline_run_log_bytes_per_case"] = exported.get("reference_run_log_bytes_per_case", 0.0)
    exported["baseline_artifact_bytes_per_case"] = exported.get("reference_artifact_bytes_per_case", 0.0)
    exported["baseline_cases"] = exported.get("reference_cases", 0)
    exported["baseline_candidate_bug_cases"] = exported.get("reference_candidate_bug_cases", 0)
    exported["is_contrast_preset"] = exported.get("is_contrast_run", False)
    exported["is_targeted_preset"] = exported.get("is_contrast_preset", False)
    exported["reference_selector_preset"] = exported.get("reference_preset", "")
    return exported


def _comparison_markdown_row(row: dict[str, Any]) -> str:
    return (
        "| {target_suite} | {variant} | {rate} | {rate_delta} | {rate_ratio} | "
        "{yield_} | {yield_delta} | {yield_ratio} | {first} | {first_s} | {auc} |"
    ).format(
        target_suite=row["target_suite"],
        variant=row["comparison_variant_label"],
        rate=_fmt_percent(row["candidate_bug_case_rate"]),
        rate_delta=_fmt_signed_percent(row["candidate_bug_case_rate_delta"]),
        rate_ratio=_fmt_ratio(row["candidate_bug_case_rate_ratio"]),
        yield_=_fmt_float(row["candidate_bug_cases_per_s"]),
        yield_delta=_fmt_signed_float(row["candidate_bug_cases_per_s_delta"]),
        yield_ratio=_fmt_ratio(row["candidate_bug_cases_per_s_ratio"]),
        first=_fmt_optional(row["median_first_candidate"]),
        first_s=_fmt_optional(row["median_first_candidate_s"]),
        auc=_fmt_float(row["discovery_auc"]),
    )


def _soundness_sentence(rows: list[dict[str, str]]) -> str:
    real_rows = [row for row in rows if not row["target_suite"].startswith("seeded_")]
    if not real_rows:
        return "No non-seeded target suites are present in this experiment."
    cases = sum(int(float(row.get("cases", "0") or 0)) for row in real_rows)
    findings = sum(int(float(row.get("findings", "0") or 0)) for row in real_rows)
    false_positives = sum(int(float(row.get("false_positive_count", "0") or 0)) for row in real_rows)
    candidate = sum(int(float(row.get("candidate_implementation_bug_count", "0") or 0)) for row in real_rows)
    return (
        f"Non-seeded target suites executed {cases} cases with {findings} findings, "
        f"{candidate} candidate implementation bugs, and {false_positives} oracle false positives."
    )


def _float(row: dict[str, str], key: str) -> float:
    return float(row.get(key, "0") or 0.0)


def _optional_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _ratio(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _optional_delta(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def _fmt_percent(value: float) -> str:
    return f"{value:.1%}"


def _fmt_signed_percent(value: float) -> str:
    return f"{value:+.1%}"


def _fmt_float(value: float) -> str:
    return f"{value:.2f}"


def _fmt_signed_float(value: float) -> str:
    return f"{value:+.2f}"


def _fmt_ratio(value: float | None) -> str:
    return "" if value is None else f"{value:.2f}x"


def _fmt_optional(value: float | None) -> str:
    if value is None:
        return ""
    return str(int(value)) if value.is_integer() else f"{value:.1f}"


def _fmt_optional_signed(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:+.0f}" if value.is_integer() else f"{value:+.1f}"
