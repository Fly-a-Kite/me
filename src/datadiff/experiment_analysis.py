from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from datadiff.reporter import latest_experiment_manifest, write_experiment_summary
from datadiff.util import REPORTS_DIR, ensure_dirs


TARGETED_PRESET_BY_SUITE = {
    "seeded_filter": "guided_filter",
    "seeded_groupby": "guided_groupby",
    "seeded_join": "guided_join",
    "seeded_mutate": "guided_mutate",
    "arrow_cross": "live_arrow",
    "dataframe_lazy": "live_polars_lazy",
    "embedded_sql": "live_embedded_sql",
    "datafusion_cross": "live_datafusion",
    "latest_all_engines": "live_cross_family",
}


def analyze_experiment(
    manifest_file: Path | None = None,
    *,
    baseline_preset: str = "baseline",
    compare_presets: list[str] | None = None,
    refresh: bool = False,
) -> tuple[Path, Path]:
    ensure_dirs()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest_file = manifest_file or latest_experiment_manifest()
    summary_md, _ = write_experiment_summary(manifest_file, refresh=refresh)
    aggregate_csv = summary_md.with_name(f"{summary_md.stem}-aggregates.csv")
    rows = _read_csv(aggregate_csv)
    comparisons = _build_comparisons(rows, baseline_preset, compare_presets)

    md_path = REPORTS_DIR / f"experiment-analysis-{manifest_file.stem}.md"
    csv_path = REPORTS_DIR / f"experiment-analysis-{manifest_file.stem}.csv"
    _write_analysis_markdown(md_path, manifest_file, aggregate_csv, comparisons, rows, baseline_preset)
    _write_analysis_csv(csv_path, comparisons)
    return md_path, csv_path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _build_comparisons(
    rows: list[dict[str, str]],
    baseline_preset: str,
    compare_presets: list[str] | None,
) -> list[dict[str, Any]]:
    by_suite_preset = {
        (row["target_suite"], row["preset"]): row
        for row in rows
    }
    suites = sorted({row["target_suite"] for row in rows})
    comparisons: list[dict[str, Any]] = []
    for suite in suites:
        baseline = by_suite_preset.get((suite, baseline_preset))
        if baseline is None:
            continue
        selected = compare_presets or sorted(
            row["preset"]
            for row in rows
            if row["target_suite"] == suite and row["preset"] != baseline_preset
        )
        for preset in selected:
            current = by_suite_preset.get((suite, preset))
            if current is None:
                continue
            comparisons.append(_compare_rows(suite, baseline_preset, baseline, current))
    return comparisons


def _compare_rows(
    suite: str,
    baseline_preset: str,
    baseline: dict[str, str],
    current: dict[str, str],
) -> dict[str, Any]:
    base_rate = _float(baseline, "candidate_bug_case_rate")
    current_rate = _float(current, "candidate_bug_case_rate")
    base_yield = _float(baseline, "candidate_bug_cases_per_s")
    current_yield = _float(current, "candidate_bug_cases_per_s")
    base_first = _optional_float(baseline.get("median_first_candidate_bug_case_index"))
    current_first = _optional_float(current.get("median_first_candidate_bug_case_index"))
    base_first_s = _optional_float(baseline.get("median_first_candidate_bug_elapsed_s"))
    current_first_s = _optional_float(current.get("median_first_candidate_bug_elapsed_s"))
    base_auc = _float(baseline, "avg_candidate_bug_discovery_auc")
    current_auc = _float(current, "avg_candidate_bug_discovery_auc")
    return {
        "target_suite": suite,
        "baseline_preset": baseline_preset,
        "preset": current["preset"],
        "is_targeted_preset": current["preset"] == TARGETED_PRESET_BY_SUITE.get(suite),
        "baseline_candidate_bug_case_rate": base_rate,
        "candidate_bug_case_rate": current_rate,
        "candidate_bug_case_rate_delta": current_rate - base_rate,
        "candidate_bug_case_rate_ratio": _ratio(current_rate, base_rate),
        "baseline_candidate_bug_cases_per_s": base_yield,
        "candidate_bug_cases_per_s": current_yield,
        "candidate_bug_cases_per_s_delta": current_yield - base_yield,
        "candidate_bug_cases_per_s_ratio": _ratio(current_yield, base_yield),
        "baseline_median_first_candidate": base_first,
        "median_first_candidate": current_first,
        "first_candidate_delta": _optional_delta(current_first, base_first),
        "baseline_median_first_candidate_s": base_first_s,
        "median_first_candidate_s": current_first_s,
        "first_candidate_s_delta": _optional_delta(current_first_s, base_first_s),
        "baseline_discovery_auc": base_auc,
        "discovery_auc": current_auc,
        "discovery_auc_delta": current_auc - base_auc,
        "baseline_cases": int(float(baseline.get("cases", "0") or 0)),
        "cases": int(float(current.get("cases", "0") or 0)),
        "baseline_candidate_bug_cases": int(float(baseline.get("candidate_bug_cases", "0") or 0)),
        "candidate_bug_cases": int(float(current.get("candidate_bug_cases", "0") or 0)),
    }


def _write_analysis_markdown(
    path: Path,
    manifest_file: Path,
    aggregate_csv: Path,
    comparisons: list[dict[str, Any]],
    aggregate_rows: list[dict[str, str]],
    baseline_preset: str,
) -> None:
    lines = [
        "# DataDiffFuzz Experiment Analysis",
        "",
        f"- Manifest: `{manifest_file}`",
        f"- Aggregate CSV: `{aggregate_csv}`",
        f"- Baseline preset: `{baseline_preset}`",
        "",
        "## Soundness Snapshot",
        "",
        _soundness_sentence(aggregate_rows),
        "",
        "## Targeted Guidance Contrasts",
        "",
        "| target suite | preset | candidate case % | rate delta | rate ratio | candidate cases/s | yield delta | yield ratio | median first | median first s | discovery AUC |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparisons:
        if not row["is_targeted_preset"]:
            continue
        lines.append(_comparison_markdown_row(row))
    lines.extend(
        [
            "",
            "## All Baseline Comparisons",
            "",
            "| target suite | preset | targeted | candidate case % | rate ratio | candidate cases/s | yield ratio | median first | median first s | discovery AUC |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in comparisons:
        lines.append(
            "| {target_suite} | {preset} | {targeted} | {rate} | {rate_ratio} | {yield_} | {yield_ratio} | {first} | {first_s} | {auc} |".format(
                target_suite=row["target_suite"],
                preset=row["preset"],
                targeted="yes" if row["is_targeted_preset"] else "no",
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
        "baseline_preset",
        "preset",
        "is_targeted_preset",
        "baseline_candidate_bug_case_rate",
        "candidate_bug_case_rate",
        "candidate_bug_case_rate_delta",
        "candidate_bug_case_rate_ratio",
        "baseline_candidate_bug_cases_per_s",
        "candidate_bug_cases_per_s",
        "candidate_bug_cases_per_s_delta",
        "candidate_bug_cases_per_s_ratio",
        "baseline_median_first_candidate",
        "median_first_candidate",
        "first_candidate_delta",
        "baseline_median_first_candidate_s",
        "median_first_candidate_s",
        "first_candidate_s_delta",
        "baseline_discovery_auc",
        "discovery_auc",
        "discovery_auc_delta",
        "baseline_cases",
        "cases",
        "baseline_candidate_bug_cases",
        "candidate_bug_cases",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(comparisons)


def _comparison_markdown_row(row: dict[str, Any]) -> str:
    return (
        "| {target_suite} | {preset} | {rate} | {rate_delta} | {rate_ratio} | "
        "{yield_} | {yield_delta} | {yield_ratio} | {first} | {first_s} | {auc} |"
    ).format(
        target_suite=row["target_suite"],
        preset=row["preset"],
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
