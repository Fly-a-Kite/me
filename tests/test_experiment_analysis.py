import csv

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
                    "seed": 1,
                    "run_file": str(baseline_run),
                    "report": "",
                },
                {
                    "target_suite": "seeded_filter",
                    "preset": "guided_filter",
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
    assert "## Targeted Guidance Contrasts" in md
    assert "| seeded_filter | guided_filter | 100.0% | +50.0% | 2.00x | 8.00 | +3.00 | 1.60x | 0 | 0.1 | 0.75 |" in md
    assert row["target_suite"] == "seeded_filter"
    assert row["preset"] == "guided_filter"
    assert row["is_targeted_preset"] == "True"
    assert row["candidate_bug_case_rate_ratio"] == "2.0"
    assert row["discovery_auc_delta"] == "0.25"


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
