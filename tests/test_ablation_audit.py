import csv
import json

from datadiff import ablation_audit
from datadiff.ablation_audit import (
    ABLATION_ONLY_REQUIRES_TRIAGE,
    CATALOG_ABLATION_PRESETS,
    CATALOG_REFERENCE_PRESETS,
    DEFAULT_ABLATION_PRESETS,
    DEFAULT_REFERENCE_PRESETS,
    REFERENCE_AND_ABLATION_DETECTED,
    analyze_ablation_audit,
)
from datadiff.util import dump_json


def test_analyze_ablation_audit_marks_ablation_only_families(tmp_path, monkeypatch):
    reports_dir = tmp_path / "reports"
    manifest = tmp_path / "runs" / "experiment-ablation.json"
    manifest.parent.mkdir(parents=True)
    dump_json({"runs": []}, manifest)
    monkeypatch.setattr(ablation_audit, "REPORTS_DIR", reports_dir)

    def fake_write_experiment_summary(manifest_file, *, refresh=False):
        reports_dir.mkdir(parents=True, exist_ok=True)
        md_path = reports_dir / f"experiment-summary-{manifest_file.stem}.md"
        csv_path = reports_dir / f"experiment-summary-{manifest_file.stem}.csv"
        aggregate_csv_path = reports_dir / f"{md_path.stem}-aggregates.csv"
        md_path.write_text("# Summary\n", encoding="utf-8")
        csv_path.write_text("", encoding="utf-8")
        aggregate_csv_path.write_text(
            "\n".join(
                [
                    "target_suite,preset,cases,findings,candidate_bug_cases,semantic_divergence_count,false_positive_count,top_candidate_bug_families",
                    "core,baseline,100,1,1,0,0,known_family@backend:1",
                    "core,no_type_aware,100,3,3,0,0,known_family@backend:1; weak_only@backend:2",
                    "core,no_normalizer,100,50,0,0,50,none",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        return md_path, csv_path

    monkeypatch.setattr(ablation_audit, "write_experiment_summary", fake_write_experiment_summary)

    md_path, csv_path = analyze_ablation_audit(manifest, refresh=True)

    md = md_path.read_text(encoding="utf-8")
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    known = next(row for row in rows if row["family"] == "known_family@backend")
    weak = next(row for row in rows if row["family"] == "weak_only@backend")
    assert "Reference variants executed 100 cases with 0 oracle false positives" in md
    assert "Ablation variants executed 200 cases with 50 oracle false positives" in md
    assert known["status"] == REFERENCE_AND_ABLATION_DETECTED
    assert weak["status"] == ABLATION_ONLY_REQUIRES_TRIAGE
    assert known["reference_count"] == "1"
    assert known["canonical_row_type"] == "reference_family_audit"
    assert known["reference_selection_count"] == "1"
    assert known["reference_row_type"] == "reference_ablation_family_audit"
    assert weak["ablation_count"] == "2"
    assert weak["component_focuses"] == "type_aware_generation"
    assert {"baseline"} == set(DEFAULT_REFERENCE_PRESETS)
    assert {"no_feedback", "metamorphic", "oracle_only_metamorphic", "reducer"}.issubset(
        set(DEFAULT_ABLATION_PRESETS)
    )


def test_analyze_ablation_audit_prefers_structured_metadata_defaults(tmp_path, monkeypatch):
    reports_dir = tmp_path / "reports"
    manifest = tmp_path / "runs" / "experiment-ablation-structured.json"
    manifest.parent.mkdir(parents=True)
    dump_json({"runs": []}, manifest)
    monkeypatch.setattr(ablation_audit, "REPORTS_DIR", reports_dir)

    def fake_write_experiment_summary(manifest_file, *, refresh=False):
        reports_dir.mkdir(parents=True, exist_ok=True)
        md_path = reports_dir / f"experiment-summary-{manifest_file.stem}.md"
        csv_path = reports_dir / f"experiment-summary-{manifest_file.stem}.csv"
        aggregate_csv_path = reports_dir / f"{md_path.stem}-aggregates.csv"
        md_path.write_text("# Summary\n", encoding="utf-8")
        csv_path.write_text("", encoding="utf-8")
        aggregate_csv_path.write_text(
            "\n".join(
                [
                    "target_suite,preset,matrix_id,comparison_group,comparison_role,component_focus,analysis_tags,cases,findings,candidate_bug_cases,semantic_divergence_count,false_positive_count,top_candidate_bug_families",
                    "core,stable_base,module_ablation,module_ablation,baseline,,ablation,100,1,1,0,0,known_family@backend:1",
                    "core,focus_variant,module_ablation,module_ablation,contrast,semantic_normalizer,\"ablation,noise_control\",100,3,3,0,0,known_family@backend:1; weak_only@backend:2",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        return md_path, csv_path

    monkeypatch.setattr(ablation_audit, "write_experiment_summary", fake_write_experiment_summary)

    _, csv_path = analyze_ablation_audit(manifest, refresh=True)

    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    known = next(row for row in rows if row["family"] == "known_family@backend")
    weak = next(row for row in rows if row["family"] == "weak_only@backend")
    assert known["reference_count"] == "1"
    assert known["canonical_row_type"] == "reference_family_audit"
    assert known["reference_selection_count"] == "1"
    assert known["reference_row_type"] == "reference_ablation_family_audit"
    assert weak["ablation_count"] == "2"
    assert weak["component_focuses"] == "semantic_normalizer"


def test_analyze_ablation_audit_prefers_structured_aggregate_json_variant_rows(tmp_path, monkeypatch):
    reports_dir = tmp_path / "reports"
    manifest = tmp_path / "runs" / "experiment-ablation-json-preferred.json"
    manifest.parent.mkdir(parents=True)
    dump_json({"runs": []}, manifest)
    monkeypatch.setattr(ablation_audit, "REPORTS_DIR", reports_dir)

    def fake_write_experiment_summary(manifest_file, *, refresh=False):
        reports_dir.mkdir(parents=True, exist_ok=True)
        md_path = reports_dir / f"experiment-summary-{manifest_file.stem}.md"
        csv_path = reports_dir / f"experiment-summary-{manifest_file.stem}.csv"
        aggregate_csv_path = reports_dir / f"{md_path.stem}-aggregates.csv"
        aggregate_json_path = reports_dir / f"{md_path.stem}-aggregates.json"
        md_path.write_text("# Summary\n", encoding="utf-8")
        csv_path.write_text("", encoding="utf-8")
        aggregate_csv_path.write_text(
            "\n".join(
                [
                    "target_suite,preset,cases,findings,candidate_bug_cases,semantic_divergence_count,false_positive_count,top_candidate_bug_families",
                    "csv_core,baseline,100,1,1,0,0,csv_only@backend:1",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        aggregate_json_path.write_text(
            json.dumps(
                {
                    "variant_rows": [
                        {
                            "target_suite": "core",
                            "preset": "stable_base",
                            "matrix_id": "module_ablation",
                            "comparison_group": "module_ablation",
                            "comparison_role": "baseline",
                            "canonical_comparison_role": "baseline",
                            "component_focus": "",
                            "analysis_tags": "ablation",
                            "cases": 100,
                            "findings": 1,
                            "candidate_bug_cases": 1,
                            "semantic_divergence_count": 0,
                            "false_positive_count": 0,
                            "top_candidate_bug_families": "known_family@backend:1",
                        },
                        {
                            "target_suite": "core",
                            "preset": "focus_variant",
                            "matrix_id": "module_ablation",
                            "comparison_group": "module_ablation",
                            "comparison_role": "contrast",
                            "canonical_comparison_role": "contrast",
                            "component_focus": "semantic_normalizer",
                            "analysis_tags": "ablation,noise_control",
                            "cases": 100,
                            "findings": 3,
                            "candidate_bug_cases": 3,
                            "semantic_divergence_count": 0,
                            "false_positive_count": 0,
                            "top_candidate_bug_families": "known_family@backend:1; weak_only@backend:2",
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )
        return md_path, csv_path

    monkeypatch.setattr(ablation_audit, "write_experiment_summary", fake_write_experiment_summary)

    md_path, csv_path = analyze_ablation_audit(manifest, refresh=True)

    md = md_path.read_text(encoding="utf-8")
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    known = next(row for row in rows if row["family"] == "known_family@backend")
    weak = next(row for row in rows if row["family"] == "weak_only@backend")
    assert "Aggregate JSON" in md
    assert known["reference_count"] == "1"
    assert weak["ablation_count"] == "2"
    assert weak["component_focuses"] == "semantic_normalizer"


def test_ablation_audit_catalog_defaults_follow_final_ablation_matrix() -> None:
    assert CATALOG_REFERENCE_PRESETS == ("baseline",)
    assert CATALOG_ABLATION_PRESETS == (
        "no_type_aware",
        "no_normalizer",
        "no_feedback",
        "metamorphic",
        "oracle_only_metamorphic",
        "reducer",
    )
    assert set(CATALOG_ABLATION_PRESETS).issubset(set(DEFAULT_ABLATION_PRESETS))
