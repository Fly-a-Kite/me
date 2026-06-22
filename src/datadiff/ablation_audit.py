from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from datadiff.experiment_catalog import FINAL_MODULE_ABLATION_MATRIX
from datadiff.experiment_metadata import (
    component_focus,
    is_ablation_contrast,
    is_reference_variant,
    row_tag_set,
)
from datadiff.reporter import latest_experiment_manifest_path, write_experiment_summary_report
from datadiff.util import REPORTS_DIR, ensure_dirs

latest_experiment_manifest = latest_experiment_manifest_path
write_experiment_summary = write_experiment_summary_report


CATALOG_REFERENCE_PRESETS = tuple(
    variant.preset
    for variant in FINAL_MODULE_ABLATION_MATRIX.variants
    if variant.comparison_role == "baseline"
)
CATALOG_ABLATION_PRESETS = tuple(
    variant.preset
    for variant in FINAL_MODULE_ABLATION_MATRIX.variants
    if variant.comparison_role == "contrast"
)
LEGACY_REFERENCE_PRESETS = ("baseline", "guided", "no_feedback", "metamorphic")
LEGACY_ABLATION_PRESETS = (
    "no_type_aware",
    "no_normalizer",
    "no_feedback",
    "metamorphic",
    "oracle_only_metamorphic",
    "reducer",
)
DEFAULT_REFERENCE_PRESETS = CATALOG_REFERENCE_PRESETS or LEGACY_REFERENCE_PRESETS
DEFAULT_ABLATION_PRESETS = tuple(
    dict.fromkeys([*CATALOG_ABLATION_PRESETS, *LEGACY_ABLATION_PRESETS])
)
REFERENCE_AND_ABLATION_DETECTED = "reference_and_ablation_detected"
REFERENCE_DETECTED = "reference_detected"
ABLATION_ONLY_REQUIRES_TRIAGE = "ablation_only_requires_triage"


def analyze_ablation_audit(
    manifest_file: Path | None = None,
    *,
    reference_presets: list[str] | None = None,
    ablation_presets: list[str] | None = None,
    refresh: bool = False,
) -> tuple[Path, Path]:
    ensure_dirs()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest_file = manifest_file or latest_experiment_manifest_path()
    summary_md, _ = write_experiment_summary(manifest_file, refresh=refresh)
    aggregate_csv = summary_md.with_name(f"{summary_md.stem}-aggregates.csv")
    aggregate_json = summary_md.with_name(f"{summary_md.stem}-aggregates.json")
    rows = _load_structured_aggregate_rows(aggregate_json, aggregate_csv)

    reference_selection = _resolve_reference_presets(reference_presets)
    ablation_selection = tuple(ablation_presets or DEFAULT_ABLATION_PRESETS)
    audit = _build_family_audit(rows, reference_selection, ablation_selection)

    md_path = REPORTS_DIR / f"ablation-audit-{manifest_file.stem}.md"
    csv_path = REPORTS_DIR / f"ablation-audit-{manifest_file.stem}.csv"
    _write_audit_markdown(
        md_path,
        manifest_file,
        aggregate_csv,
        aggregate_json,
        reference_selection,
        ablation_selection,
        audit,
    )
    _write_audit_csv(csv_path, audit["family_rows"])
    return md_path, csv_path


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_structured_aggregate_rows(
    aggregate_json_path: Path,
    aggregate_csv_path: Path,
) -> list[dict[str, Any]]:
    if aggregate_json_path.is_file():
        payload = json.loads(aggregate_json_path.read_text(encoding="utf-8"))
        variant_rows = payload.get("variant_rows", []) if isinstance(payload, dict) else []
        if isinstance(variant_rows, list) and all(isinstance(item, dict) for item in variant_rows):
            return variant_rows
    return _load_csv_rows(aggregate_csv_path)


def _resolve_reference_presets(
    reference_presets: list[str] | None,
) -> tuple[str, ...]:
    return tuple(reference_presets or DEFAULT_REFERENCE_PRESETS)


def _build_family_audit(
    rows: list[dict[str, str]],
    reference_selection: tuple[str, ...],
    ablation_selection: tuple[str, ...],
) -> dict[str, Any]:
    preset_totals: dict[str, Counter[str]] = defaultdict(Counter)
    family_by_preset: dict[str, Counter[str]] = defaultdict(Counter)
    family_by_component_focus: dict[str, Counter[str]] = defaultdict(Counter)
    family_suites: dict[str, set[str]] = defaultdict(set)
    family_presets: dict[str, set[str]] = defaultdict(set)

    for row in rows:
        preset = row.get("preset", "")
        totals = preset_totals[preset]
        for key in ("cases", "findings", "candidate_bug_cases", "semantic_divergence_count", "false_positive_count"):
            totals[key] += _int(row.get(key, "0"))
        for family, count in _parse_family_counts(row.get("top_candidate_bug_families", "")):
            family_by_preset[preset][family] += count
            focus = component_focus(row)
            if focus:
                family_by_component_focus[focus][family] += count
            family_suites[family].add(row.get("target_suite", ""))
            family_presets[family].add(preset)

    reference_presets = set(reference_selection)
    ablation_presets = set(ablation_selection)
    structured_reference_presets = {
        row.get("preset", "")
        for row in rows
        if _is_reference_variant(row)
    }
    structured_ablation_presets = {
        row.get("preset", "")
        for row in rows
        if _is_ablation_variant(row)
    }
    if structured_reference_presets and structured_ablation_presets:
        effective_reference_presets = structured_reference_presets
        effective_ablation_presets = structured_ablation_presets
    else:
        effective_reference_presets = reference_presets
        effective_ablation_presets = ablation_presets

    reference_families = set()
    for preset in effective_reference_presets:
        reference_families.update(family_by_preset.get(preset, {}))

    family_rows = []
    for family in sorted(set().union(*[set(counter) for counter in family_by_preset.values()]) if family_by_preset else set()):
        reference_count = sum(
            family_by_preset.get(preset, Counter()).get(family, 0)
            for preset in effective_reference_presets
        )
        ablation_count = sum(
            family_by_preset.get(preset, Counter()).get(family, 0)
            for preset in effective_ablation_presets
        )
        all_count = sum(counter.get(family, 0) for counter in family_by_preset.values())
        component_focuses = sorted(
            focus for focus, families in family_by_component_focus.items() if families.get(family, 0)
        )
        status = _family_status(family, reference_count, ablation_count)
        family_rows.append(
            {
                "family": family,
                "status": status,
                "reference_count": reference_count,
                "ablation_count": ablation_count,
                "total_count": all_count,
                "presets": ",".join(sorted(family_presets[family])),
                "target_suites": ",".join(sorted(suite for suite in family_suites[family] if suite)),
                "component_focuses": ",".join(component_focuses),
            }
        )

    return {
        "preset_totals": preset_totals,
        "family_rows": family_rows,
        "reference_presets": sorted(effective_reference_presets),
        "ablation_presets": sorted(effective_ablation_presets),
        "reference_families": reference_families,
        "reference_cases": sum(
            preset_totals.get(preset, Counter()).get("cases", 0)
            for preset in effective_reference_presets
        ),
        "reference_false_positives": sum(
            preset_totals.get(preset, Counter()).get("false_positive_count", 0)
            for preset in effective_reference_presets
        ),
        "ablation_cases": sum(
            preset_totals.get(preset, Counter()).get("cases", 0)
            for preset in effective_ablation_presets
        ),
        "ablation_false_positives": sum(
            preset_totals.get(preset, Counter()).get("false_positive_count", 0)
            for preset in effective_ablation_presets
        ),
    }


def _is_reference_row(row: dict[str, str]) -> bool:
    tags = row_tag_set(row)
    if is_reference_variant(row):
        return True
    if "baseline" in tags:
        return True
    return False


def _is_reference_variant(row: dict[str, str]) -> bool:
    return _is_reference_row(row)


def _is_ablation_row(row: dict[str, str]) -> bool:
    return is_ablation_contrast(row)


def _is_ablation_variant(row: dict[str, str]) -> bool:
    return _is_ablation_row(row)


def _parse_family_counts(text: str) -> list[tuple[str, int]]:
    if not text or text == "none":
        return []
    families = []
    for chunk in text.split(";"):
        chunk = chunk.strip()
        if not chunk or ":" not in chunk:
            continue
        family, count_text = chunk.rsplit(":", 1)
        try:
            count = int(float(count_text.strip()))
        except ValueError:
            continue
        families.append((family.strip(), count))
    return families


def _family_status(family: str, reference_count: int, ablation_count: int) -> str:
    if reference_count and ablation_count:
        return REFERENCE_AND_ABLATION_DETECTED
    if reference_count:
        return REFERENCE_DETECTED
    if ablation_count:
        return ABLATION_ONLY_REQUIRES_TRIAGE
    return "unclassified"


def _write_audit_markdown(
    path: Path,
    manifest_file: Path,
    aggregate_csv: Path,
    aggregate_json: Path,
    reference_selection: tuple[str, ...],
    ablation_selection: tuple[str, ...],
    audit: dict[str, Any],
) -> None:
    lines = [
        "# DataDiffFuzz Ablation Audit",
        "",
        f"- Manifest: `{manifest_file}`",
        f"- Aggregate CSV: `{aggregate_csv}`",
        f"- Aggregate JSON: `{aggregate_json}`",
        f"- Reference selection: `{','.join(audit['reference_presets'])}`",
        f"- Ablation selection: `{','.join(audit['ablation_presets'])}`",
        "",
        "## Soundness Boundary",
        "",
        (
            f"Reference variants executed {audit['reference_cases']} cases with "
            f"{audit['reference_false_positives']} oracle false positives. "
            f"Ablation variants executed {audit['ablation_cases']} cases with "
            f"{audit['ablation_false_positives']} oracle false positives."
        ),
        "",
        "## Variant Totals",
        "",
        "| variant | cases | findings | candidate cases | semantic divergences | false positives |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for preset in sorted(audit["preset_totals"]):
        totals = audit["preset_totals"][preset]
        lines.append(
            "| {variant} | {cases} | {findings} | {candidate} | {semantic} | {false_positive} |".format(
                variant=preset,
                cases=totals.get("cases", 0),
                findings=totals.get("findings", 0),
                candidate=totals.get("candidate_bug_cases", 0),
                semantic=totals.get("semantic_divergence_count", 0),
                false_positive=totals.get("false_positive_count", 0),
            )
        )
    lines.extend(
        [
            "",
            "## Candidate Family Audit",
            "",
            "| family | status | reference count | ablation count | total count | variants | target suites | component focuses |",
            "|---|---|---:|---:|---:|---|---|---|",
        ]
    )
    for row in audit["family_rows"]:
        lines.append(
            "| {family} | {status} | {reference_count} | {ablation_count} | {total_count} | {presets} | {target_suites} | {component_focuses} |".format(
                **row
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_audit_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "family",
        "status",
        "canonical_row_type",
        "reference_row_type",
        "reference_count",
        "reference_variant_count",
        "reference_selection_count",
        "trusted_count",
        "ablation_count",
        "total_count",
        "presets",
        "target_suites",
        "component_focuses",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(_export_audit_family_row(row) for row in rows)


def _export_audit_family_row(row: dict[str, Any]) -> dict[str, Any]:
    exported = dict(row)
    exported["canonical_row_type"] = "reference_family_audit"
    exported["reference_row_type"] = "reference_ablation_family_audit"
    exported["reference_variant_count"] = exported.get("reference_count", 0)
    exported["reference_selection_count"] = exported.get("reference_count", 0)
    exported["trusted_count"] = exported.get("reference_count", 0)
    return exported


def _int(value: str | None) -> int:
    return int(float(value or 0))
