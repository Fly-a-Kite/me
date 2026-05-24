import csv

from datadiff import reporter
from datadiff.dsl import Case, ColumnSpec, Program, TableData
from datadiff.util import append_jsonl, dump_json, run_meta_path


def _write_run(path, case_id):
    append_jsonl(
        {
            "status": "ok",
            "case": {"case_id": case_id, "seed": 1, "program": {"operations": []}},
            "findings": [],
            "behavior_signature": f"sig-{case_id}",
            "backend_status": {},
            "quality_oracles": [],
            "is_new_behavior": True,
        },
        path,
    )
    dump_json(
        {
            "elapsed_s": 0.1,
            "throughput_cases_s": 10.0,
            "backends": [],
            "targets": [],
            "common_capabilities": [],
        },
        run_meta_path(path),
    )


def test_write_report_uses_run_stem_for_unique_artifact_names(tmp_path, monkeypatch):
    runs_dir = tmp_path / "runs"
    reports_dir = tmp_path / "reports"
    monkeypatch.setattr(reporter, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(reporter, "REPORTS_DIR", reports_dir)
    run_a = runs_dir / "run-a.jsonl.gz"
    run_b = runs_dir / "run-b.jsonl.gz"
    _write_run(run_a, "case-a")
    _write_run(run_b, "case-b")

    md_a, csv_a = reporter.write_report(run_a)
    md_b, csv_b = reporter.write_report(run_b)

    assert md_a.name == "report-run-a.md"
    assert csv_a.name == "findings-run-a.csv"
    assert md_b.name == "report-run-b.md"
    assert csv_b.name == "findings-run-b.csv"
    assert md_a.exists()
    assert md_b.exists()
    assert md_a != md_b


def test_write_report_can_limit_findings_csv_rows(tmp_path, monkeypatch):
    runs_dir = tmp_path / "runs"
    reports_dir = tmp_path / "reports"
    monkeypatch.setattr(reporter, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(reporter, "REPORTS_DIR", reports_dir)
    run_file = runs_dir / "run-limit.jsonl.gz"
    for idx in range(3):
        append_jsonl(
            {
                "status": "bug",
                "case": {"case_id": f"case-{idx}", "seed": idx, "program": {"operations": []}},
                "findings": [
                    {
                        "kind": "semantic_output_mismatch",
                        "severity": "critical",
                        "root_cause": "filter_predicate",
                        "oracle": "differential",
                        "confidence": "high",
                        "triage_verdict": "semantic_divergence_needs_confirmation",
                        "paper_status": "valid_finding_not_confirmed_bug",
                        "triage_confidence": "medium",
                        "false_positive": False,
                        "false_positive_reason": "",
                        "suspicious_backends": ["duckdb"],
                        "signature": f"sig-{idx}",
                        "evidence": "different output",
                        "triage_evidence": "",
                    }
                ],
                "behavior_signature": f"behavior-{idx}",
                "backend_status": {},
                "quality_oracles": [],
                "is_new_behavior": True,
            },
            run_file,
        )
    dump_json(
        {
            "elapsed_s": 0.1,
            "throughput_cases_s": 10.0,
            "backends": [],
            "targets": [],
            "common_capabilities": [],
        },
        run_meta_path(run_file),
    )

    md_path, csv_path = reporter.write_report(run_file, csv_limit=1)

    assert "Findings CSV limit: 1" in md_path.read_text(encoding="utf-8")
    assert len(csv_path.read_text(encoding="utf-8").splitlines()) == 2


def test_write_experiment_summary_uses_manifest_stem(tmp_path, monkeypatch):
    runs_dir = tmp_path / "runs"
    reports_dir = tmp_path / "reports"
    monkeypatch.setattr(reporter, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(reporter, "REPORTS_DIR", reports_dir)
    run_file = runs_dir / "run-a.jsonl.gz"
    _write_run(run_file, "case-a")
    manifest = runs_dir / "experiment-x.json"
    dump_json(
        {
            "presets": ["baseline"],
            "seeds": [1],
            "backends": [],
            "target_suite": "core",
            "targets": [],
            "common_capabilities": [],
            "runs": [{"preset": "baseline", "seed": 1, "run_file": str(run_file), "report": ""}],
        },
        manifest,
    )

    md_path, csv_path = reporter.write_experiment_summary(manifest)

    assert md_path.name == "experiment-summary-experiment-x.md"
    assert csv_path.name == "experiment-summary-experiment-x.csv"
    assert md_path.exists()
    assert csv_path.exists()


def test_write_experiment_summary_aggregates_guidance_metrics(tmp_path, monkeypatch):
    runs_dir = tmp_path / "runs"
    reports_dir = tmp_path / "reports"
    monkeypatch.setattr(reporter, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(reporter, "REPORTS_DIR", reports_dir)
    run_file = runs_dir / "run-guided.jsonl.gz"
    for idx in range(2):
        append_jsonl(
            {
                "status": "ok",
                "case": {"case_id": f"case-{idx}", "seed": idx, "program": {"operations": []}},
                "findings": [],
                "behavior_signature": f"sig-{idx}",
                "backend_status": {},
                "quality_oracles": [],
                "is_new_behavior": idx == 0,
                "guidance": {
                    "score": 2.0 + idx,
                    "data_sensitivity": 0.5 + idx,
                    "path_coverage_proxy": 0.25 + idx,
                    "frontier_conformance": 0.75 + idx,
                    "contribution_potential": 1.0 + idx,
                    "candidate_count": 4,
                    "contributing_candidate_count": 2,
                    "pruned_candidate_count": 1,
                    "feature_count": 3 + idx,
                    "frontier_bucket_count": 1 + idx,
                },
            },
            run_file,
        )
    dump_json(
        {
            "elapsed_s": 0.1,
            "throughput_cases_s": 20.0,
            "backends": [],
            "targets": [],
            "common_capabilities": [],
        },
        run_meta_path(run_file),
    )
    manifest = runs_dir / "experiment-guided.json"
    dump_json(
        {
            "presets": ["guided"],
            "seeds": [1],
            "backends": [],
            "target_suite": "core",
            "targets": [],
            "common_capabilities": [],
            "runs": [{"preset": "guided", "seed": 1, "run_file": str(run_file), "report": ""}],
        },
        manifest,
    )

    md_path, csv_path = reporter.write_experiment_summary(manifest)
    row = next(csv.DictReader(csv_path.open(encoding="utf-8")))

    assert "data sensitivity" in md_path.read_text(encoding="utf-8")
    assert row["new_behavior_rate"] == "0.5"
    assert row["avg_guidance_score"] == "2.5"
    assert row["avg_data_sensitivity"] == "1.0"
    assert row["avg_path_coverage_proxy"] == "0.75"
    assert row["avg_frontier_conformance"] == "1.25"
    assert row["avg_contribution_potential"] == "1.5"
    assert row["pruned_candidate_rate"] == "0.25"


def test_write_experiment_summary_aggregates_triage_verdicts(tmp_path, monkeypatch):
    runs_dir = tmp_path / "runs"
    reports_dir = tmp_path / "reports"
    monkeypatch.setattr(reporter, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(reporter, "REPORTS_DIR", reports_dir)
    run_file = runs_dir / "run-triage.jsonl.gz"
    findings = [
        ("candidate_implementation_bug", "arithmetic_expression"),
        ("expected_semantic_divergence", "nan_inf_semantics"),
        ("documented_semantic_divergence", "nan_inf_semantics"),
        ("generator_false_positive", "invalid_program"),
    ]
    for idx, (verdict, root) in enumerate(findings):
        append_jsonl(
            {
                "status": "bug",
                "case_index": idx,
                "elapsed_s": (idx + 1) * 0.05,
                "case": {"case_id": f"case-{idx}", "seed": idx, "program": {"operations": []}},
                "findings": [
                    {
                        "kind": "semantic_output_mismatch",
                        "root_cause": root,
                        "triage_verdict": verdict,
                        "signature": f"sig-{idx}",
                    }
                ],
                "behavior_signature": f"sig-{idx}",
                "backend_status": {},
                "quality_oracles": [],
            },
            run_file,
        )
    dump_json(
        {
            "elapsed_s": 0.1,
            "throughput_cases_s": 20.0,
            "backends": [],
            "targets": [],
            "common_capabilities": [],
        },
        run_meta_path(run_file),
    )
    manifest = runs_dir / "experiment-triage.json"
    dump_json(
        {
            "presets": ["edge_float"],
            "seeds": [1],
            "backends": [],
            "target_suite": "core",
            "targets": [],
            "common_capabilities": [],
            "runs": [{"preset": "edge_float", "seed": 1, "run_file": str(run_file), "report": ""}],
        },
        manifest,
    )

    md_path, csv_path = reporter.write_experiment_summary(manifest)
    row = next(csv.DictReader(csv_path.open(encoding="utf-8")))
    aggregate_csv_path = reports_dir / "experiment-summary-experiment-triage-aggregates.csv"
    aggregate_row = next(csv.DictReader(aggregate_csv_path.open(encoding="utf-8")))
    md = md_path.read_text(encoding="utf-8")

    assert "candidate bugs" in md
    assert row["candidate_implementation_bug_count"] == "1"
    assert row["expected_semantic_divergence_count"] == "1"
    assert row["documented_semantic_divergence_count"] == "1"
    assert row["generator_false_positive_count"] == "1"
    assert row["candidate_bug_cases"] == "1"
    assert row["candidate_bug_case_rate"] == "0.25"
    assert row["first_finding_case_index"] == "0"
    assert row["first_candidate_bug_case_index"] == "0"
    assert row["first_candidate_bug_elapsed_s"] == "0.05"
    assert row["candidate_bug_discovery_auc"] == "1.0"
    assert "candidate_implementation_bug:1" in row["top_triage_verdicts"]
    assert "## Aggregates" in md
    assert (
        "| core | edge_float | 1 | 4 | 4 | 1 | 1 | 0 | 0 | 25.0% | "
        "5.00 | 0 | 0.1 | 1.00 | 0.00 | 2 | 1 |"
    ) in md
    assert aggregate_row["candidate_bug_case_rate"] == "0.25"
    assert aggregate_row["candidate_bug_cases_per_s"] == "5.0"
    assert aggregate_row["median_first_candidate_bug_case_index"] == "0.0"
    assert aggregate_row["median_first_candidate_bug_elapsed_s"] == "0.05"
    assert aggregate_row["avg_candidate_bug_discovery_auc"] == "1.0"


def test_write_experiment_summary_reports_candidate_bug_families(tmp_path, monkeypatch):
    runs_dir = tmp_path / "runs"
    reports_dir = tmp_path / "reports"
    monkeypatch.setattr(reporter, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(reporter, "REPORTS_DIR", reports_dir)
    run_file = runs_dir / "run-families.jsonl.gz"
    for idx in range(3):
        append_jsonl(
            {
                "status": "bug",
                "case_index": idx,
                "case": {"case_id": f"case-{idx}", "seed": idx, "program": {"operations": []}},
                "findings": [
                    {
                        "kind": "semantic_output_mismatch",
                        "root_cause": "grouped_topk_null_sort_key" if idx < 2 else "filter_predicate",
                        "triage_verdict": "candidate_implementation_bug",
                        "suspicious_backends": ["datafusion"] if idx < 2 else ["duckdb"],
                        "signature": f"sig-{idx}",
                    }
                ],
                "behavior_signature": f"sig-{idx}",
                "backend_status": {},
                "quality_oracles": [],
            },
            run_file,
        )
    dump_json(
        {
            "elapsed_s": 0.1,
            "throughput_cases_s": 10.0,
            "backends": [],
            "targets": [],
            "common_capabilities": [],
        },
        run_meta_path(run_file),
    )
    manifest = runs_dir / "experiment-families.json"
    dump_json(
        {
            "presets": ["null_agg_topk"],
            "seeds": [1],
            "backends": [],
            "target_suite": "datafusion_cross",
            "targets": [],
            "common_capabilities": [],
            "runs": [{"preset": "null_agg_topk", "seed": 1, "run_file": str(run_file), "report": ""}],
        },
        manifest,
    )

    md_path, csv_path = reporter.write_experiment_summary(manifest)
    row = next(csv.DictReader(csv_path.open(encoding="utf-8")))
    aggregate_csv_path = reports_dir / "experiment-summary-experiment-families-aggregates.csv"
    aggregate_row = next(csv.DictReader(aggregate_csv_path.open(encoding="utf-8")))
    md = md_path.read_text(encoding="utf-8")

    assert "## Candidate Bug-Family Deduplication" in md
    assert row["candidate_bug_families"] == "2"
    assert "grouped_topk_null_sort_key@datafusion:2" in row["top_candidate_bug_families"]
    assert aggregate_row["candidate_bug_families"] == "2"


def test_write_experiment_summary_separates_issue_replay_candidates(tmp_path, monkeypatch):
    runs_dir = tmp_path / "runs"
    reports_dir = tmp_path / "reports"
    monkeypatch.setattr(reporter, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(reporter, "REPORTS_DIR", reports_dir)
    run_file = runs_dir / "run-origin.jsonl.gz"
    append_jsonl(
        {
            "status": "bug",
            "case_index": 0,
            "case": {"case_id": "case-0", "seed": 0, "program": {"operations": []}},
            "findings": [
                {
                    "kind": "semantic_output_mismatch",
                    "root_cause": "groupby_aggregation",
                    "triage_verdict": "candidate_implementation_bug",
                    "suspicious_backends": ["datafusion"],
                    "signature": "sig-organic",
                    "discovery_origin": "organic",
                },
                {
                    "kind": "semantic_output_mismatch",
                    "root_cause": "group_quantile_key_expression",
                    "triage_verdict": "candidate_implementation_bug",
                    "suspicious_backends": ["polars"],
                    "signature": "sig-replay",
                    "discovery_origin": "issue_replay",
                },
            ],
            "behavior_signature": "sig-0",
            "backend_status": {},
            "quality_oracles": [],
        },
        run_file,
    )
    dump_json(
        {
            "elapsed_s": 0.1,
            "throughput_cases_s": 10.0,
            "backends": [],
            "targets": [],
            "common_capabilities": [],
            "config": {"enable_replay_bug": False},
            "replay_bug_filter": {
                "enabled": True,
                "filtered_candidates": 7,
                "fallback_candidates": 2,
            },
        },
        run_meta_path(run_file),
    )
    manifest = runs_dir / "experiment-origin.json"
    dump_json(
        {
            "presets": ["live_cross_family"],
            "seeds": [1],
            "backends": [],
            "target_suite": "latest_all_engines",
            "targets": [],
            "common_capabilities": [],
            "replay_bug_policy": {
                "enable_replay_bug": False,
                "source_issues": [
                    "https://github.com/apache/datafusion/issues/22190",
                    "https://github.com/duckdb/duckdb/issues/22075",
                ],
            },
            "runs": [{"preset": "live_cross_family", "seed": 1, "run_file": str(run_file), "report": ""}],
        },
        manifest,
    )

    md_path, csv_path = reporter.write_experiment_summary(manifest)
    row = next(csv.DictReader(csv_path.open(encoding="utf-8")))
    aggregate_csv_path = reports_dir / "experiment-summary-experiment-origin-aggregates.csv"
    aggregate_row = next(csv.DictReader(aggregate_csv_path.open(encoding="utf-8")))
    md = md_path.read_text(encoding="utf-8")

    assert row["candidate_implementation_bug_count"] == "2"
    assert row["rewardable_candidate_implementation_bug_count"] == "1"
    assert row["issue_replay_candidate_bug_count"] == "1"
    assert row["enable_replay_bug"] == "False"
    assert row["replay_filter_enabled"] == "True"
    assert row["replay_filter_filtered_candidates"] == "7"
    assert row["replay_filter_fallback_candidates"] == "2"
    assert "organic:1" in row["top_discovery_origins"]
    assert "issue_replay:1" in row["top_discovery_origins"]
    assert row["top_candidate_bug_families"] == "groupby_aggregation@datafusion:1"
    assert aggregate_row["rewardable_candidate_implementation_bug_count"] == "1"
    assert "rewardable candidates" in md
    assert "Replay bug policy: enable_replay_bug=false, source_issues=2" in md


def test_write_experiment_summary_excludes_known_saturated_families_from_rewardable_count(tmp_path, monkeypatch):
    runs_dir = tmp_path / "runs"
    reports_dir = tmp_path / "reports"
    monkeypatch.setattr(reporter, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(reporter, "REPORTS_DIR", reports_dir)
    run_file = runs_dir / "run-known-family.jsonl.gz"
    append_jsonl(
        {
            "status": "bug",
            "case_index": 0,
            "case": {"case_id": "case-0", "seed": 0, "program": {"operations": []}},
            "findings": [
                {
                    "kind": "semantic_output_mismatch",
                    "root_cause": "grouped_topk_null_sort_key",
                    "triage_verdict": "candidate_implementation_bug",
                    "suspicious_backends": ["datafusion"],
                    "signature": "known-sig",
                    "discovery_origin": "organic",
                }
            ],
            "behavior_signature": "sig-0",
            "backend_status": {},
            "quality_oracles": [],
        },
        run_file,
    )
    dump_json(
        {
            "elapsed_s": 0.1,
            "throughput_cases_s": 10.0,
            "backends": [],
            "targets": [],
            "common_capabilities": [],
            "config": {
                "enable_replay_bug": False,
                "known_saturated_bug_families": ["grouped_topk_null_sort_key@datafusion"],
            },
            "replay_bug_filter": {"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
        },
        run_meta_path(run_file),
    )
    manifest = runs_dir / "experiment-known-family.json"
    dump_json(
        {
            "presets": ["live_datafusion"],
            "seeds": [1],
            "backends": [],
            "target_suite": "datafusion_cross",
            "targets": [],
            "common_capabilities": [],
            "runs": [{"preset": "live_datafusion", "seed": 1, "run_file": str(run_file), "report": ""}],
        },
        manifest,
    )

    _, csv_path = reporter.write_experiment_summary(manifest)
    row = next(csv.DictReader(csv_path.open(encoding="utf-8")))
    aggregate_row = next(
        csv.DictReader((reports_dir / "experiment-summary-experiment-known-family-aggregates.csv").open(encoding="utf-8"))
    )

    assert row["candidate_implementation_bug_count"] == "1"
    assert row["known_saturated_candidate_bug_count"] == "1"
    assert row["rewardable_candidate_implementation_bug_count"] == "0"
    assert row["candidate_bug_families"] == "0"
    assert row["top_candidate_bug_families"] == "none"
    assert aggregate_row["known_saturated_candidate_bug_count"] == "1"
    assert aggregate_row["rewardable_candidate_implementation_bug_count"] == "0"


def test_write_experiment_summary_includes_adaptive_schedule_fields(tmp_path, monkeypatch):
    runs_dir = tmp_path / "runs"
    reports_dir = tmp_path / "reports"
    monkeypatch.setattr(reporter, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(reporter, "REPORTS_DIR", reports_dir)
    run_file = runs_dir / "run-adaptive.jsonl.gz"
    append_jsonl(
        {
            "status": "bug",
            "case_index": 0,
            "case": {"case_id": "case-0", "seed": 0, "program": {"operations": []}},
            "findings": [
                {
                    "kind": "semantic_output_mismatch",
                    "root_cause": "grouped_topk_null_sort_key",
                    "triage_verdict": "candidate_implementation_bug",
                    "suspicious_backends": ["datafusion"],
                    "signature": "sig-0",
                }
            ],
            "behavior_signature": "sig-0",
            "backend_status": {},
            "quality_oracles": [],
            "is_new_behavior": True,
        },
        run_file,
    )
    dump_json(
        {
            "elapsed_s": 0.1,
            "throughput_cases_s": 10.0,
            "backends": [],
            "targets": [],
            "common_capabilities": [],
        },
        run_meta_path(run_file),
    )
    manifest = runs_dir / "experiment-adaptive.json"
    dump_json(
        {
            "presets": ["baseline"],
            "seeds": [1],
            "backends": [],
            "target_suite": "datafusion_cross",
            "targets": [],
            "common_capabilities": [],
            "schedule": "adaptive",
            "local_source_scheduler": {"enabled": True, "exploration_weight": 0.25},
            "adaptive_config": {"jobs": 2, "batch_cases": 1},
            "adaptive_state": [{"arm_id": "datafusion_cross:baseline:seed1"}],
            "runs": [
                {
                    "target_suite": "datafusion_cross",
                    "preset": "baseline",
                    "seed": 1,
                    "batch_index": 3,
                    "schedule_arm_id": "datafusion_cross:baseline:seed1",
                    "scheduler_reward": 4.25,
                    "run_file": str(run_file),
                    "report": "",
                }
            ],
        },
        manifest,
    )

    md_path, csv_path = reporter.write_experiment_summary(manifest)
    row = next(csv.DictReader(csv_path.open(encoding="utf-8")))
    aggregate_csv_path = reports_dir / "experiment-summary-experiment-adaptive-aggregates.csv"
    aggregate_row = next(csv.DictReader(aggregate_csv_path.open(encoding="utf-8")))
    md = md_path.read_text(encoding="utf-8")

    assert "- Schedule: adaptive" in md
    assert "- Local source scheduler: {'enabled': True, 'exploration_weight': 0.25}" in md
    assert "## Adaptive Schedule" in md
    assert "- Global first candidate case: 1" in md
    assert "- Batches by suite: datafusion_cross:1" in md
    assert "- Cases by suite: datafusion_cross:1" in md
    assert "- Candidate cases by suite: datafusion_cross:1" in md
    assert row["batch_index"] == "3"
    assert row["schedule_arm_id"] == "datafusion_cross:baseline:seed1"
    assert row["scheduler_reward"] == "4.25"
    assert aggregate_row["avg_scheduler_reward"] == "4.25"


def test_write_experiment_summary_reports_preflight_integrity(tmp_path, monkeypatch):
    runs_dir = tmp_path / "runs"
    reports_dir = tmp_path / "reports"
    monkeypatch.setattr(reporter, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(reporter, "REPORTS_DIR", reports_dir)
    run_file = runs_dir / "run-preflight.jsonl.gz"
    append_jsonl(
        {
            "status": "ok",
            "case_index": 0,
            "case": {"case_id": "case-0", "seed": 0, "program": {"operations": []}},
            "findings": [],
            "behavior_signature": "sig-0",
            "backend_status": {},
            "quality_oracles": [],
            "is_new_behavior": False,
        },
        run_file,
    )
    append_jsonl(
        {
            "status": "ok",
            "case_index": 1,
            "case": {"case_id": "case-1", "seed": 1, "program": {"operations": []}},
            "findings": [],
            "behavior_signature": "sig-1",
            "backend_status": {},
            "quality_oracles": [],
            "is_new_behavior": True,
        },
        run_file,
    )
    dump_json(
        {
            "elapsed_s": 0.2,
            "throughput_cases_s": 10.0,
            "backends": [],
            "targets": [],
            "common_capabilities": [],
            "preflight": {"repaired_cases": 1, "fallback_cases": 1, "invalid_cases": 0},
        },
        run_meta_path(run_file),
    )
    manifest = runs_dir / "experiment-preflight.json"
    dump_json(
        {
            "presets": ["baseline"],
            "seeds": [1],
            "backends": [],
            "target_suite": "core",
            "targets": [],
            "common_capabilities": [],
            "runs": [
                {
                    "target_suite": "core",
                    "preset": "baseline",
                    "seed": 1,
                    "run_file": str(run_file),
                    "report": "",
                }
            ],
        },
        manifest,
    )

    md_path, csv_path = reporter.write_experiment_summary(manifest)
    row = next(csv.DictReader(csv_path.open(encoding="utf-8")))
    aggregate_csv_path = reports_dir / "experiment-summary-experiment-preflight-aggregates.csv"
    aggregate_row = next(csv.DictReader(aggregate_csv_path.open(encoding="utf-8")))
    md = md_path.read_text(encoding="utf-8")

    assert "## Preflight Integrity" in md
    assert "| core | baseline | 1 | 2 | 0.0% | 50.0% | 50.0% |" in md
    assert row["preflight_repaired_cases"] == "1"
    assert row["preflight_fallback_cases"] == "1"
    assert row["preflight_invalid_cases"] == "0"
    assert aggregate_row["preflight_repaired_rate"] == "0.5"
    assert aggregate_row["preflight_fallback_rate"] == "0.5"


def test_candidate_bug_family_dedup_maps_metamorphic_to_differential_root(tmp_path, monkeypatch):
    runs_dir = tmp_path / "runs"
    reports_dir = tmp_path / "reports"
    monkeypatch.setattr(reporter, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(reporter, "REPORTS_DIR", reports_dir)
    run_file = runs_dir / "run-metamorphic-family.jsonl.gz"
    append_jsonl(
        {
            "status": "bug",
            "case_index": 0,
            "case": {"case_id": "case-0", "seed": 0, "program": {"operations": []}},
            "findings": [
                {
                    "kind": "semantic_output_mismatch",
                    "root_cause": "float_group_key_instability",
                    "triage_verdict": "candidate_implementation_bug",
                    "suspicious_backends": ["polars_lazy"],
                    "signature": "sig-diff",
                },
                {
                    "kind": "metamorphic_join_inner_left_equivalence_violation",
                    "root_cause": "metamorphic_join_inner_left_equivalence",
                    "triage_verdict": "candidate_implementation_bug",
                    "suspicious_backends": ["polars_lazy"],
                    "signature": "sig-mr",
                },
            ],
            "behavior_signature": "sig-0",
            "backend_status": {},
            "quality_oracles": [],
        },
        run_file,
    )
    dump_json(
        {
            "elapsed_s": 0.1,
            "throughput_cases_s": 10.0,
            "backends": [],
            "targets": [],
            "common_capabilities": [],
        },
        run_meta_path(run_file),
    )
    manifest = runs_dir / "experiment-metamorphic-family.json"
    dump_json(
        {
            "presets": ["float_group_key_metamorphic"],
            "seeds": [1],
            "backends": [],
            "target_suite": "core_lazy",
            "targets": [],
            "common_capabilities": [],
            "runs": [{"preset": "float_group_key_metamorphic", "seed": 1, "run_file": str(run_file), "report": ""}],
        },
        manifest,
    )

    _, csv_path = reporter.write_experiment_summary(manifest)
    row = next(csv.DictReader(csv_path.open(encoding="utf-8")))

    assert row["candidate_bug_families"] == "1"
    assert row["top_candidate_bug_families"] == "float_group_key_instability@polars_lazy:2"


def test_write_experiment_summary_refreshes_from_bug_artifact(tmp_path, monkeypatch):
    runs_dir = tmp_path / "runs"
    reports_dir = tmp_path / "reports"
    bugs_dir = tmp_path / "bugs"
    monkeypatch.setattr(reporter, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(reporter, "REPORTS_DIR", reports_dir)
    bug_dir = bugs_dir / "bug-refresh"
    bug_dir.mkdir(parents=True)
    case = Case(
        "case-refresh",
        14,
        [TableData("t0", [ColumnSpec("x", "int")], [{"x": 0}, {"x": 0}])],
        Program(
            "prog-refresh",
            14,
            [
                {"op": "mutate", "column": "m_0", "expr": {"kind": "add_const", "source": "x", "value": -1}},
                {"op": "filter", "column": "m_0", "cmp": "==", "value": -1},
                {"op": "mutate", "column": "m_1", "expr": {"kind": "arith_const", "op": "mul", "source": "m_0", "value": 10}},
                {"op": "mutate", "column": "m_3", "expr": {"kind": "arith_const", "op": "div", "source": "m_1", "value": 3}},
                {"op": "groupby", "keys": ["m_3"], "aggs": [{"column": "m_0", "func": "min", "as": "min_m_0"}]},
            ],
        ),
    )
    dump_json(case.to_dict(), bug_dir / "case.json")
    dump_json(
        {
            "a": {"backend": "a", "status": "ok", "columns": ["m_3", "min_m_0"], "rows": [[-3.3333333333, -1]]},
            "c": {"backend": "c", "status": "ok", "columns": ["m_3", "min_m_0"], "rows": [[-3.3333333333, -1]]},
            "b": {
                "backend": "b",
                "status": "ok",
                "columns": ["m_3", "min_m_0"],
                "rows": [[-3.3333333333, -1], [-3.3333333333, -1]],
            },
        },
        bug_dir / "normalized.json",
    )
    dump_json({"generator_profile": "common"}, bug_dir / "config.json")
    run_file = runs_dir / "run-refresh.jsonl.gz"
    append_jsonl(
        {
            "status": "bug",
            "case_index": 0,
            "case": {"case_id": "case-refresh", "seed": 14, "row_count": 2, "table_count": 1, "program": {"operations": []}},
            "findings": [
                {
                    "kind": "semantic_output_mismatch",
                    "root_cause": "groupby_aggregation",
                    "triage_verdict": "candidate_implementation_bug",
                    "suspicious_backends": ["b"],
                    "signature": "stale",
                }
            ],
            "bug_dir": str(bug_dir),
            "behavior_signature": "sig-refresh",
            "backend_status": {"a": "ok", "b": "ok", "c": "ok"},
            "quality_oracles": [],
        },
        run_file,
    )
    dump_json(
        {
            "elapsed_s": 0.1,
            "throughput_cases_s": 10.0,
            "backends": ["a", "b", "c"],
            "targets": [],
            "common_capabilities": [],
        },
        run_meta_path(run_file),
    )
    manifest = runs_dir / "experiment-refresh.json"
    dump_json(
        {
            "presets": ["bughunt"],
            "seeds": [14],
            "backends": ["a", "b", "c"],
            "target_suite": "custom",
            "targets": [],
            "common_capabilities": [],
            "runs": [{"preset": "bughunt", "seed": 14, "run_file": str(run_file), "report": ""}],
        },
        manifest,
    )

    md_path, csv_path = reporter.write_experiment_summary(manifest, refresh=True)
    row = next(csv.DictReader(csv_path.open(encoding="utf-8")))
    md = md_path.read_text(encoding="utf-8")

    assert "Refreshed with current oracle: yes" in md
    assert row["top_root_causes"] == "float_group_key_instability:1"
    assert row["top_candidate_bug_families"] == "float_group_key_instability@b:1"
