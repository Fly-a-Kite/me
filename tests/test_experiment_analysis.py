import csv
import json

from datadiff import experiment_analysis, reporter
from datadiff.experiment_analysis import analyze_experiment
from datadiff.util import append_jsonl, dump_json, run_meta_path


def _write_run(path, *, candidate_indexes, throughput):
    for idx in range(2):
        findings = []
        if idx in candidate_indexes:
            findings.append(
                {
                    "kind": "semantic_output_mismatch",
                    "root_cause": "filter_predicate",
                    "triage_verdict": "candidate_implementation_bug",
                    "signature": f"sig-{path.stem}-{idx}",
                }
            )
        append_jsonl(
            {
                "status": "bug" if findings else "ok",
                "case_index": idx,
                "elapsed_s": (idx + 1) * 0.05,
                "case": {"case_id": f"case-{idx}", "seed": idx, "program": {"operations": []}},
                "findings": findings,
                "behavior_signature": f"behavior-{idx}",
                "backend_status": {},
                "quality_oracles": [],
                "is_new_behavior": True,
            },
            path,
        )
    dump_json(
        {
            "elapsed_s": 0.2,
            "throughput_cases_s": throughput,
            "backends": [],
            "targets": [],
            "common_capabilities": [],
        },
        run_meta_path(path),
    )


def test_analyze_experiment_writes_baseline_comparison(tmp_path, monkeypatch):
    runs_dir = tmp_path / "runs"
    reports_dir = tmp_path / "reports"
    monkeypatch.setattr(reporter, "REPORTS_DIR", reports_dir)
    monkeypatch.setattr(experiment_analysis, "REPORTS_DIR", reports_dir)
    baseline_run = runs_dir / "run-baseline.jsonl.gz"
    guided_run = runs_dir / "run-guided.jsonl.gz"
    _write_run(baseline_run, candidate_indexes={1}, throughput=10.0)
    _write_run(guided_run, candidate_indexes={0, 1}, throughput=8.0)
    manifest = runs_dir / "experiment-seeded.json"
    dump_json(
        {
            "presets": ["baseline", "guided_filter"],
            "seeds": [1],
            "backends": [],
            "target_suite": "seeded_filter",
            "target_suites": ["seeded_filter"],
            "targets": [],
            "common_capabilities": [],
            "runs": [
                {
                    "target_suite": "seeded_filter",
                    "preset": "baseline",
                    "matrix_id": "seeded_sensitivity",
                    "comparison_group": "seeded_sensitivity",
                    "variant_id": "baseline",
                    "variant_title": "baseline",
                    "base_preset": "baseline",
                    "analysis_tags": ["seeded", "baseline"],
                    "seed": 1,
                    "run_file": str(baseline_run),
                    "report": "",
                },
                {
                    "target_suite": "seeded_filter",
                    "preset": "guided_filter",
                    "matrix_id": "seeded_sensitivity",
                    "comparison_group": "seeded_sensitivity",
                    "variant_id": "guided_filter",
                    "variant_title": "guided_filter",
                    "base_preset": "baseline",
                    "analysis_tags": ["seeded", "guided"],
                    "seed": 1,
                    "run_file": str(guided_run),
                    "report": "",
                },
            ],
        },
        manifest,
    )

    md_path, csv_path = analyze_experiment(manifest)

    md = md_path.read_text(encoding="utf-8")
    row = next(csv.DictReader(csv_path.open(encoding="utf-8")))
    assert "## Contrast Variant Comparisons" in md
    assert "## Space Efficiency Contrasts" in md
    assert "| seeded_filter | guided_filter | 100.0% | +50.0% | 2.00x | 8.00 | +3.00 | 1.60x | 0 | 0.1 | 0.75 |" in md
    assert row["target_suite"] == "seeded_filter"
    assert row["preset"] == "guided_filter"
    assert row["comparison_variant_id"] == "guided_filter"
    assert row["comparison_variant_label"] == "guided_filter"
    assert row["baseline_variant_id"] == "baseline"
    assert row["baseline_variant_label"] == "baseline"
    assert row["is_contrast_run"] == "True"
    assert row["is_contrast_variant"] == "True"
    assert row["is_contrast_comparison"] == "True"
    assert row["is_contrast_preset"] == "True"
    assert row["is_targeted_comparison"] == "True"
    assert row["is_targeted_preset"] == "True"
    assert row["matrix_id"] == "seeded_sensitivity"
    assert row["comparison_group"] == "seeded_sensitivity"
    assert row["variant_id"] == "guided_filter"
    assert row["base_preset"] == "baseline"
    assert row["canonical_comparison_role"] == "contrast"
    assert row["candidate_bug_case_rate_ratio"] == "2.0"
    assert row["discovery_auc_delta"] == "0.25"
    assert float(row["evidence_bytes_per_case"]) > 0.0
    assert float(row["run_log_bytes_per_case"]) > 0.0
    assert "- Aggregate JSON: `" in md


def test_analyze_experiment_can_refresh_summary_before_analysis(tmp_path, monkeypatch):
    reports_dir = tmp_path / "reports"
    manifest = tmp_path / "runs" / "experiment-refresh.json"
    manifest.parent.mkdir(parents=True)
    dump_json({"runs": []}, manifest)
    monkeypatch.setattr(experiment_analysis, "REPORTS_DIR", reports_dir)
    calls = []

    def fake_write_experiment_summary(manifest_file, *, refresh=False):
        calls.append((manifest_file, refresh))
        reports_dir.mkdir(parents=True, exist_ok=True)
        md_path = reports_dir / f"experiment-summary-{manifest_file.stem}.md"
        csv_path = reports_dir / f"experiment-summary-{manifest_file.stem}.csv"
        aggregate_csv_path = reports_dir / f"{md_path.stem}-aggregates.csv"
        md_path.write_text("# Summary\n", encoding="utf-8")
        csv_path.write_text("", encoding="utf-8")
        aggregate_csv_path.write_text("target_suite,preset\n", encoding="utf-8")
        return md_path, csv_path

    monkeypatch.setattr(experiment_analysis, "write_experiment_summary", fake_write_experiment_summary)

    md_path, csv_path = analyze_experiment(manifest, refresh=True)

    assert calls == [(manifest, True)]
    assert md_path.exists()
    assert csv_path.exists()
    assert "No non-seeded target suites are present" in md_path.read_text(encoding="utf-8")


def test_analyze_experiment_prefers_structured_aggregate_json_variant_rows(tmp_path, monkeypatch):
    reports_dir = tmp_path / "reports"
    manifest = tmp_path / "runs" / "experiment-json-preferred.json"
    manifest.parent.mkdir(parents=True)
    dump_json({"runs": []}, manifest)
    monkeypatch.setattr(experiment_analysis, "REPORTS_DIR", reports_dir)

    def fake_write_experiment_summary(manifest_file, *, refresh=False):
        reports_dir.mkdir(parents=True, exist_ok=True)
        md_path = reports_dir / f"experiment-summary-{manifest_file.stem}.md"
        csv_path = reports_dir / f"experiment-summary-{manifest_file.stem}.csv"
        aggregate_csv_path = reports_dir / f"{md_path.stem}-aggregates.csv"
        aggregate_json_path = reports_dir / f"{md_path.stem}-aggregates.json"
        md_path.write_text("# Summary\n", encoding="utf-8")
        csv_path.write_text("", encoding="utf-8")
        aggregate_csv_path.write_text(
            "target_suite,preset,matrix_id,comparison_group,variant_id,variant_title,variant_label,base_preset,"
            "comparison_role,canonical_comparison_role,scope_kind,oracle_profile,rq_tags,analysis_tags,"
            "candidate_bug_case_rate,candidate_bug_cases_per_s,median_first_candidate_bug_case_index,"
            "median_first_candidate_bug_elapsed_s,avg_candidate_bug_discovery_auc,evidence_bytes_per_case,"
            "run_log_bytes_per_case,artifact_bytes_per_case,cases,candidate_bug_cases\n",
            encoding="utf-8",
        )
        aggregate_json_path.write_text(
            json.dumps(
                {
                    "variant_rows": [
                        {
                            "target_suite": "core",
                            "preset": "baseline",
                            "matrix_id": "module_ablation",
                            "comparison_group": "module_ablation",
                            "variant_id": "baseline",
                            "variant_title": "baseline",
                            "variant_label": "baseline",
                            "base_preset": "baseline",
                            "comparison_role": "baseline",
                            "canonical_comparison_role": "baseline",
                            "scope_kind": "core",
                            "oracle_profile": "differential",
                            "rq_tags": "RQ2",
                            "analysis_tags": "ablation",
                            "candidate_bug_case_rate": 0.25,
                            "candidate_bug_cases_per_s": 5.0,
                            "median_first_candidate_bug_case_index": 1.0,
                            "median_first_candidate_bug_elapsed_s": 0.1,
                            "avg_candidate_bug_discovery_auc": 0.5,
                            "evidence_bytes_per_case": 10.0,
                            "run_log_bytes_per_case": 6.0,
                            "artifact_bytes_per_case": 4.0,
                            "cases": 4,
                            "candidate_bug_cases": 1,
                        },
                        {
                            "target_suite": "core",
                            "preset": "contrast",
                            "matrix_id": "module_ablation",
                            "comparison_group": "module_ablation",
                            "variant_id": "contrast",
                            "variant_title": "contrast",
                            "variant_label": "contrast",
                            "base_preset": "baseline",
                            "comparison_role": "contrast",
                            "canonical_comparison_role": "contrast",
                            "scope_kind": "core",
                            "oracle_profile": "differential",
                            "rq_tags": "RQ2",
                            "analysis_tags": "ablation",
                            "candidate_bug_case_rate": 0.5,
                            "candidate_bug_cases_per_s": 8.0,
                            "median_first_candidate_bug_case_index": 0.0,
                            "median_first_candidate_bug_elapsed_s": 0.05,
                            "avg_candidate_bug_discovery_auc": 0.75,
                            "evidence_bytes_per_case": 12.0,
                            "run_log_bytes_per_case": 7.0,
                            "artifact_bytes_per_case": 5.0,
                            "cases": 4,
                            "candidate_bug_cases": 2,
                        },
                    ]
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return md_path, csv_path

    monkeypatch.setattr(experiment_analysis, "write_experiment_summary", fake_write_experiment_summary)

    md_path, csv_path = analyze_experiment(manifest, reference_preset="baseline")

    row = next(csv.DictReader(csv_path.open(encoding="utf-8")))
    md = md_path.read_text(encoding="utf-8")
    assert row["preset"] == "contrast"
    assert row["candidate_bug_case_rate_ratio"] == "2.0"
    assert row["candidate_bug_cases_per_s_ratio"] == "1.6"
    assert "- Aggregate JSON: `" in md


def test_analyze_experiment_falls_back_to_aggregate_csv_when_json_missing(tmp_path, monkeypatch):
    reports_dir = tmp_path / "reports"
    manifest = tmp_path / "runs" / "experiment-csv-fallback.json"
    manifest.parent.mkdir(parents=True)
    dump_json({"runs": []}, manifest)
    monkeypatch.setattr(experiment_analysis, "REPORTS_DIR", reports_dir)

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
                    "target_suite,preset,matrix_id,comparison_group,variant_id,variant_title,variant_label,base_preset,comparison_role,canonical_comparison_role,scope_kind,oracle_profile,rq_tags,analysis_tags,candidate_bug_case_rate,candidate_bug_cases_per_s,median_first_candidate_bug_case_index,median_first_candidate_bug_elapsed_s,avg_candidate_bug_discovery_auc,evidence_bytes_per_case,run_log_bytes_per_case,artifact_bytes_per_case,cases,candidate_bug_cases",
                    "core,baseline,module_ablation,module_ablation,baseline,baseline,baseline,baseline,baseline,baseline,core,differential,RQ2,ablation,0.25,5.0,1,0.1,0.5,10.0,6.0,4.0,4,1",
                    "core,contrast,module_ablation,module_ablation,contrast,contrast,contrast,baseline,contrast,contrast,core,differential,RQ2,ablation,0.5,8.0,0,0.05,0.75,12.0,7.0,5.0,4,2",
                ]
            ),
            encoding="utf-8",
        )
        return md_path, csv_path

    monkeypatch.setattr(experiment_analysis, "write_experiment_summary", fake_write_experiment_summary)

    _, csv_path = analyze_experiment(manifest, reference_preset="baseline")

    row = next(csv.DictReader(csv_path.open(encoding="utf-8")))
    assert row["preset"] == "contrast"
    assert row["candidate_bug_case_rate_ratio"] == "2.0"


def test_analyze_experiment_uses_structured_metadata_for_targeted_classification(tmp_path, monkeypatch):
    runs_dir = tmp_path / "runs"
    reports_dir = tmp_path / "reports"
    monkeypatch.setattr(reporter, "REPORTS_DIR", reports_dir)
    monkeypatch.setattr(experiment_analysis, "REPORTS_DIR", reports_dir)
    baseline_run = runs_dir / "run-base.jsonl.gz"
    targeted_run = runs_dir / "run-targeted.jsonl.gz"
    _write_run(baseline_run, candidate_indexes={1}, throughput=10.0)
    _write_run(targeted_run, candidate_indexes={0, 1}, throughput=8.0)
    manifest = runs_dir / "experiment-structured-targeted.json"
    dump_json(
        {
            "presets": ["stable_base", "organic_focus"],
            "seeds": [1],
            "backends": [],
            "target_suite": "seeded_filter",
            "target_suites": ["seeded_filter"],
            "targets": [],
            "common_capabilities": [],
            "experiment_meta": {
                "matrix_id": "seeded_sensitivity",
                "comparison_group": "seeded_sensitivity",
                "variant_by_preset": {
                    "stable_base": {
                        "variant_id": "stable_base",
                        "variant_title": "stable_base",
                        "base_preset": "stable_base",
                        "comparison_role": "baseline",
                        "analysis_tags": ["seeded", "baseline"],
                        "oracle_profile": "differential",
                    },
                    "organic_focus": {
                        "variant_id": "organic_focus",
                        "variant_title": "organic_focus",
                        "base_preset": "stable_base",
                        "comparison_role": "contrast",
                        "analysis_tags": ["seeded", "guided"],
                        "oracle_profile": "differential",
                    },
                },
            },
            "runs": [
                {
                    "target_suite": "seeded_filter",
                    "preset": "stable_base",
                    "seed": 1,
                    "run_file": str(baseline_run),
                    "report": "",
                },
                {
                    "target_suite": "seeded_filter",
                    "preset": "organic_focus",
                    "seed": 1,
                    "run_file": str(targeted_run),
                    "report": "",
                },
            ],
        },
        manifest,
    )

    _, csv_path = analyze_experiment(manifest, reference_preset="stable_base")

    row = next(csv.DictReader(csv_path.open(encoding="utf-8")))
    assert row["preset"] == "organic_focus"
    assert row["comparison_variant_id"] == "organic_focus"
    assert row["comparison_variant_label"] == "organic_focus"
    assert row["baseline_variant_id"] == "stable_base"
    assert row["baseline_variant_label"] == "stable_base"
    assert row["is_contrast_run"] == "True"
    assert row["is_contrast_variant"] == "True"
    assert row["is_contrast_comparison"] == "True"
    assert row["is_contrast_preset"] == "True"
    assert row["is_targeted_variant"] == "True"
    assert row["is_targeted_comparison"] == "True"
    assert row["is_targeted_preset"] == "True"
    assert row["comparison_role"] == "contrast"
    assert row["canonical_comparison_role"] == "contrast"
    assert row["reference_selector_preset"] == "stable_base"


def test_analyze_experiment_prefers_structured_contrast_variants_over_noncontrast_rows(tmp_path, monkeypatch):
    runs_dir = tmp_path / "runs"
    reports_dir = tmp_path / "reports"
    monkeypatch.setattr(reporter, "REPORTS_DIR", reports_dir)
    monkeypatch.setattr(experiment_analysis, "REPORTS_DIR", reports_dir)
    baseline_run = runs_dir / "run-base.jsonl.gz"
    contrast_run = runs_dir / "run-contrast.jsonl.gz"
    auxiliary_run = runs_dir / "run-aux.jsonl.gz"
    _write_run(baseline_run, candidate_indexes={1}, throughput=10.0)
    _write_run(contrast_run, candidate_indexes={0, 1}, throughput=8.0)
    _write_run(auxiliary_run, candidate_indexes={0}, throughput=9.0)
    manifest = runs_dir / "experiment-structured-contrast-preferred.json"
    dump_json(
        {
            "presets": ["stable_base", "focus_variant", "workflow_probe"],
            "seeds": [1],
            "backends": [],
            "target_suite": "core",
            "target_suites": ["core"],
            "targets": [],
            "common_capabilities": [],
            "experiment_meta": {
                "matrix_id": "module_ablation",
                "comparison_group": "module_ablation",
                "variant_by_preset": {
                    "stable_base": {
                        "variant_id": "stable_base",
                        "variant_title": "stable_base",
                        "base_preset": "stable_base",
                        "comparison_role": "baseline",
                        "analysis_tags": ["ablation", "baseline"],
                        "oracle_profile": "differential",
                    },
                    "focus_variant": {
                        "variant_id": "focus_variant",
                        "variant_title": "focus_variant",
                        "base_preset": "stable_base",
                        "comparison_role": "contrast",
                        "analysis_tags": ["ablation", "noise_control"],
                        "oracle_profile": "differential",
                    },
                    "workflow_probe": {
                        "variant_id": "workflow_probe",
                        "variant_title": "workflow_probe",
                        "base_preset": "stable_base",
                        "comparison_role": "",
                        "analysis_tags": ["ablation", "workflow"],
                        "oracle_profile": "differential",
                    },
                },
            },
            "runs": [
                {
                    "target_suite": "core",
                    "preset": "stable_base",
                    "seed": 1,
                    "run_file": str(baseline_run),
                    "report": "",
                },
                {
                    "target_suite": "core",
                    "preset": "focus_variant",
                    "seed": 1,
                    "run_file": str(contrast_run),
                    "report": "",
                },
                {
                    "target_suite": "core",
                    "preset": "workflow_probe",
                    "seed": 1,
                    "run_file": str(auxiliary_run),
                    "report": "",
                },
            ],
        },
        manifest,
    )

    _, csv_path = analyze_experiment(manifest, reference_preset="stable_base")

    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    assert [row["preset"] for row in rows] == ["focus_variant"]
