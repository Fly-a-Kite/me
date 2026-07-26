import csv
import json

import pytest

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


def _polluted_signal_rows():
    return [
        {
            "status": "bug",
            "case": {"case_id": "resolved", "seed": 1, "program": {"operations": []}},
            "case_index": 0,
            "elapsed_s": 0.1,
            "behavior_signature": "resolved",
            "backend_status": {},
            "quality_oracles": [],
            "is_new_behavior": True,
            "signal_new_behavior": True,
            "findings": [
                {
                    "kind": "semantic_output_mismatch",
                    "severity": "medium",
                    "root_cause": "nan_inf_semantics",
                    "triage_verdict": "expected_semantic_divergence",
                    "suspicious_backends": ["duckdb"],
                    "signature": "sig-resolved",
                    "evidence": "expected backend semantic split",
                }
            ],
        },
        {
            "status": "bug",
            "case": {"case_id": "false-positive", "seed": 2, "program": {"operations": []}},
            "case_index": 1,
            "elapsed_s": 0.2,
            "behavior_signature": "false-positive",
            "backend_status": {},
            "quality_oracles": [],
            "is_new_behavior": True,
            "signal_new_behavior": True,
            "findings": [
                {
                    "kind": "semantic_output_mismatch",
                    "severity": "medium",
                    "root_cause": "order_only_normalization_mismatch",
                    "triage_verdict": "normalizer_false_positive",
                    "false_positive": True,
                    "suspicious_backends": ["sqlite"],
                    "signature": "sig-false-positive",
                    "evidence": "normalizer artifact",
                }
            ],
        },
        {
            "status": "bug",
            "case": {"case_id": "source-issue", "seed": 3, "program": {"operations": []}},
            "case_index": 2,
            "elapsed_s": 0.3,
            "behavior_signature": "source-issue",
            "backend_status": {},
            "quality_oracles": [],
            "is_new_behavior": True,
            "signal_new_behavior": True,
            "findings": [
                {
                    "kind": "semantic_output_mismatch",
                    "severity": "medium",
                    "root_cause": "csv_long_numeric_roundtrip",
                    "triage_verdict": "candidate_implementation_bug",
                    "source_issue": "duckdb/duckdb#12345",
                    "suspicious_backends": ["duckdb"],
                    "signature": "sig-source-issue",
                    "evidence": "known source issue",
                }
            ],
        },
        {
            "status": "ok",
            "case": {"case_id": "pure-behavior", "seed": 4, "program": {"operations": []}},
            "case_index": 3,
            "elapsed_s": 0.4,
            "behavior_signature": "pure-behavior",
            "backend_status": {},
            "quality_oracles": [],
            "is_new_behavior": True,
            "signal_new_behavior": True,
            "findings": [],
        },
        {
            "status": "bug",
            "case": {"case_id": "candidate", "seed": 5, "program": {"operations": []}},
            "case_index": 4,
            "elapsed_s": 0.5,
            "behavior_signature": "candidate",
            "backend_status": {},
            "quality_oracles": [],
            "is_new_behavior": True,
            "signal_new_behavior": True,
            "findings": [
                {
                    "kind": "semantic_output_mismatch",
                    "severity": "high",
                    "root_cause": "topk_filter_pushdown",
                    "triage_verdict": "candidate_implementation_bug",
                    "suspicious_backends": ["datafusion"],
                    "signature": "sig-candidate",
                    "evidence": "fresh candidate",
                }
            ],
        },
        {
            "status": "bug",
            "case": {"case_id": "semantic-needs-confirmation", "seed": 6, "program": {"operations": []}},
            "case_index": 5,
            "elapsed_s": 0.6,
            "behavior_signature": "semantic-needs-confirmation",
            "backend_status": {},
            "quality_oracles": [],
            "is_new_behavior": True,
            "signal_new_behavior": True,
            "findings": [
                {
                    "kind": "semantic_output_mismatch",
                    "severity": "medium",
                    "root_cause": "string_expression",
                    "triage_verdict": "semantic_divergence_needs_confirmation",
                    "suspicious_backends": ["sqlite"],
                    "signature": "sig-semantic-needs-confirmation",
                    "evidence": "needs semantic confirmation",
                }
            ],
        },
    ]


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


def test_latest_run_file_uses_mtime_not_lexical_order(tmp_path, monkeypatch):
    runs_dir = tmp_path / "runs"
    monkeypatch.setattr(reporter, "RUNS_DIR", runs_dir)
    older_lexically_later = runs_dir / "run-z.jsonl"
    newer_lexically_earlier = runs_dir / "run-a.jsonl.gz"
    _write_run(older_lexically_later, "case-old")
    _write_run(newer_lexically_earlier, "case-new")
    older_time = 1_700_000_000
    newer_time = older_time + 60
    older_lexically_later.touch()
    newer_lexically_earlier.touch()
    import os

    os.utime(older_lexically_later, (older_time, older_time))
    os.utime(newer_lexically_earlier, (newer_time, newer_time))

    assert reporter.latest_run_file() == newer_lexically_earlier
    assert reporter.latest_run_log_path() == newer_lexically_earlier


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


def test_write_report_filters_non_rewardable_signal_new_behavior(tmp_path, monkeypatch):
    runs_dir = tmp_path / "runs"
    reports_dir = tmp_path / "reports"
    monkeypatch.setattr(reporter, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(reporter, "REPORTS_DIR", reports_dir)
    run_file = runs_dir / "run-filtered-signal.jsonl.gz"
    for row in _polluted_signal_rows():
        append_jsonl(row, run_file)
    dump_json(
        {
            "executed_cases": 99,
            "new_behavior_cases": 99,
            "signal_new_behavior_cases": 99,
            "elapsed_s": 1.0,
            "throughput_cases_s": 6.0,
            "backends": [],
            "targets": [],
            "common_capabilities": [],
        },
        run_meta_path(run_file),
    )

    md_path, _ = reporter.write_report(run_file)
    md_text = md_path.read_text(encoding="utf-8")

    assert "- Raw new behavior cases: 6" in md_text
    assert "- Signal new behavior cases: 3" in md_text


def test_write_report_summarizes_adaptive_selection_telemetry(tmp_path, monkeypatch):
    runs_dir = tmp_path / "runs"
    reports_dir = tmp_path / "reports"
    monkeypatch.setattr(reporter, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(reporter, "REPORTS_DIR", reports_dir)
    run_file = runs_dir / "run-adaptive-selection.jsonl.gz"
    append_jsonl(
        {
            "status": "ok",
            "case": {"case_id": "case-0", "seed": 0, "program": {"operations": []}},
            "findings": [],
            "behavior_signature": "sig-0",
            "backend_status": {},
            "quality_oracles": [],
            "is_new_behavior": True,
            "generator_profile_selection": {
                "strategy": "contextual_bandit",
                "profile": "discovery_fresh",
                "profile_pool": ["common", "discovery_fresh"],
                "learning_weight": 1.0,
                "reward": 2.0,
                "ranked": [
                    {
                        "action_id": "discovery_fresh",
                        "score": 1.4,
                        "model_prediction": 0.6,
                        "uncertainty": 0.25,
                        "exploration_bonus": 0.3,
                        "version_signal": 0.2,
                        "continual_priority_signal": 0.1,
                        "health_penalty": 0.05,
                    }
                ],
            },
            "semantic_objective_selection": {
                "strategy": "contextual_bandit_warmup",
                "scope": "semantic_objective",
                "action": "objective:join_null_membership",
                "action_pool": ["objective:join_null_membership"],
                "learning_weight": 1.0,
                "reward": 1.5,
                "ranked": [{"action_id": "objective:join_null_membership", "uncertainty": 1.0}],
            },
            "metamorphic_relation_selection": {
                "strategy": "contextual_bandit",
                "scope": "metamorphic_relation",
                "action": "input_partition_union_all",
                "action_pool": ["input_partition_union_all"],
                "learning_weight": 1.0,
                "reward": 0.5,
                "ranked": [{"action_id": "input_partition_union_all", "exploration_bonus": 0.75}],
            },
            "version_pair_selection": {
                "strategy": "contextual_bandit",
                "scope": "version_pair",
                "action": "latest-smoke->fixed-smoke",
                "action_pool": ["latest-smoke->fixed-smoke"],
                "learning_weight": 1.0,
                "reward": 1.0,
                "ranked": [{"action_id": "latest-smoke->fixed-smoke", "version_signal": 0.8}],
            },
            "selected_generator_profile": "discovery_fresh",
            "selected_semantic_objective": "objective:join_null_membership",
            "selected_metamorphic_relation": "input_partition_union_all",
            "selected_version_pair": "latest-smoke->fixed-smoke",
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

    md_path, _ = reporter.write_report(run_file)
    md = md_path.read_text(encoding="utf-8")

    assert "## Adaptive Selection" in md
    assert "| generator_profile | 1 | discovery_fresh:1 | contextual_bandit:1 | 2.00 | 1.00 | 2.00 | 1.40 | 0.60 | 0.25 | 0.30 | 0.20 | 0.10 | 0.05 |" in md
    assert "objective:join_null_membership:1" in md
    assert "input_partition_union_all:1" in md
    assert "latest-smoke->fixed-smoke:1" in md


def test_reporter_canonical_helpers_match_compatibility_aliases(tmp_path, monkeypatch):
    runs_dir = tmp_path / "runs"
    reports_dir = tmp_path / "reports"
    monkeypatch.setattr(reporter, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(reporter, "REPORTS_DIR", reports_dir)
    run_file = runs_dir / "run-canonical.jsonl"
    _write_run(run_file, "case-canonical")

    assert reporter.write_run_report(run_file) == reporter.write_report(run_file)


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
    row = next(csv.DictReader(csv_path.open(encoding="utf-8")))
    aggregate_csv = reports_dir / "experiment-summary-experiment-x-aggregates.csv"
    aggregate_row = next(csv.DictReader(aggregate_csv.open(encoding="utf-8")))
    md_text = md_path.read_text(encoding="utf-8")

    assert md_path.name == "experiment-summary-experiment-x.md"
    assert csv_path.name == "experiment-summary-experiment-x.csv"
    assert md_path.exists()
    assert csv_path.exists()
    assert "## Storage Efficiency" in md_text
    assert "## Stage Profiling" in md_text
    assert int(row["run_log_bytes"]) > 0
    assert float(row["run_log_bytes_per_case"]) > 0.0
    assert "stage_backend_execution_avg_ms" in row
    assert int(aggregate_row["evidence_bytes"]) >= int(row["run_log_bytes"])
    assert float(aggregate_row["evidence_bytes_per_case"]) > 0.0
    assert "stage_total_case_wall_avg_ms" in aggregate_row


def test_write_experiment_summary_filters_non_rewardable_signal_new_behavior(tmp_path, monkeypatch):
    runs_dir = tmp_path / "runs"
    reports_dir = tmp_path / "reports"
    monkeypatch.setattr(reporter, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(reporter, "REPORTS_DIR", reports_dir)
    run_file = runs_dir / "run-filtered-signal.jsonl.gz"
    for row in _polluted_signal_rows():
        append_jsonl(row, run_file)
    dump_json(
        {
            "executed_cases": 99,
            "new_behavior_cases": 99,
            "signal_new_behavior_cases": 99,
            "elapsed_s": 1.0,
            "throughput_cases_s": 6.0,
            "backends": [],
            "targets": [],
            "common_capabilities": [],
        },
        run_meta_path(run_file),
    )
    manifest = runs_dir / "experiment-filtered-signal.json"
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

    _, csv_path = reporter.write_experiment_summary(manifest)
    row = next(csv.DictReader(csv_path.open(encoding="utf-8")))
    aggregate_csv = reports_dir / "experiment-summary-experiment-filtered-signal-aggregates.csv"
    aggregate_row = next(csv.DictReader(aggregate_csv.open(encoding="utf-8")))

    assert row["cases"] == "6"
    assert row["new_behavior_cases"] == "6"
    assert row["signal_new_behavior_cases"] == "3"
    assert row["signal_new_behavior_rate"] == "0.5"
    assert row["candidate_implementation_bug_count"] == "2"
    assert row["rewardable_candidate_implementation_bug_count"] == "1"
    assert row["candidate_bug_cases"] == "1"
    assert row["candidate_bug_case_rate"] == str(1 / 6)
    assert row["first_candidate_bug_case_index"] == "4"
    assert row["first_candidate_bug_elapsed_s"] == "0.5"
    assert row["candidate_bug_discovery_auc"] == str(1 / 3)
    assert aggregate_row["new_behavior_cases"] == "6"
    assert aggregate_row["signal_new_behavior_cases"] == "3"
    assert aggregate_row["avg_signal_new_behavior_rate"] == "0.5"
    assert aggregate_row["candidate_bug_cases"] == "1"
    assert aggregate_row["candidate_bug_case_rate"] == str(1 / 6)
    assert aggregate_row["candidate_bug_cases_per_s"] == "1.0"
    assert aggregate_row["median_first_candidate_bug_case_index"] == "4.0"
    assert aggregate_row["median_first_candidate_bug_elapsed_s"] == "0.5"
    assert aggregate_row["avg_candidate_bug_discovery_auc"] == str(1 / 3)


def test_write_experiment_summary_preserves_structured_experiment_metadata(tmp_path, monkeypatch):
    runs_dir = tmp_path / "runs"
    reports_dir = tmp_path / "reports"
    monkeypatch.setattr(reporter, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(reporter, "REPORTS_DIR", reports_dir)
    run_file = runs_dir / "run-structured.jsonl.gz"
    _write_run(run_file, "case-structured")
    manifest = runs_dir / "experiment-structured.json"
    dump_json(
        {
            "presets": ["baseline"],
            "seeds": [1],
            "backends": [],
            "target_suite": "core",
            "target_suites": ["core"],
            "targets": [],
            "common_capabilities": [],
            "experiment_meta": {
                "matrix_id": "module_ablation",
                "comparison_group": "module_ablation",
            },
            "runs": [
                {
                    "target_suite": "core",
                    "preset": "baseline",
                    "matrix_id": "module_ablation",
                    "matrix_title": "Module Ablation",
                    "comparison_group": "module_ablation",
                    "variant_id": "baseline",
                    "variant_title": "baseline",
                    "base_preset": "baseline",
                    "comparison_role": "baseline",
                    "component_focus": "",
                    "overlays": [],
                    "factors": {"type_aware_generation": True},
                    "scope_kind": "core",
                    "oracle_profile": "differential",
                    "rq_tags": ["RQ2", "RQ4"],
                    "analysis_tags": ["ablation"],
                    "counts_as_real_bugs": False,
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
    aggregate_csv = reports_dir / "experiment-summary-experiment-structured-aggregates.csv"
    aggregate_row = next(csv.DictReader(aggregate_csv.open(encoding="utf-8")))
    md_text = md_path.read_text(encoding="utf-8")

    assert row["matrix_id"] == "module_ablation"
    assert row["comparison_group"] == "module_ablation"
    assert row["variant_id"] == "baseline"
    assert row["base_preset"] == "baseline"
    assert row["comparison_role"] == "baseline"
    assert row["canonical_comparison_role"] == "baseline"
    assert row["variant_label"] == "baseline"
    assert row["variant_group_id"] == "core|module_ablation|module_ablation"
    assert row["component_focus"] == ""
    assert row["semantic_focus_families"] == ""
    assert row["semantic_focus_signals"] == ""
    assert row["scope_kind"] == "core"
    assert row["oracle_profile"] == "differential"
    assert row["rq_tags"] == "RQ2,RQ4"
    assert row["analysis_tags"] == "ablation"
    assert '"type_aware_generation": true' in row["factors"]
    assert aggregate_row["matrix_id"] == "module_ablation"
    assert aggregate_row["comparison_group"] == "module_ablation"
    assert aggregate_row["variant_id"] == "baseline"
    assert aggregate_row["base_preset"] == "baseline"
    assert aggregate_row["comparison_role"] == "baseline"
    assert aggregate_row["canonical_comparison_role"] == "baseline"
    assert aggregate_row["variant_label"] == "baseline"
    assert aggregate_row["variant_group_id"] == "core|module_ablation|module_ablation"
    assert aggregate_row["component_focus"] == ""
    assert aggregate_row["semantic_focus_families"] == ""
    assert aggregate_row["semantic_focus_signals"] == ""
    assert aggregate_row["oracle_profile"] == "differential"
    assert aggregate_row["analysis_tags"] == "ablation"
    assert '"type_aware_generation": true' in aggregate_row["factors"]
    assert "- Matrix id: module_ablation" in md_text
    assert "- Comparison group: module_ablation" in md_text
    assert "- Aggregate JSON: `" in md_text


def test_write_experiment_summary_backfills_run_semantics_from_experiment_meta(tmp_path, monkeypatch):
    runs_dir = tmp_path / "runs"
    reports_dir = tmp_path / "reports"
    monkeypatch.setattr(reporter, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(reporter, "REPORTS_DIR", reports_dir)
    run_file = runs_dir / "run-backfill.jsonl.gz"
    _write_run(run_file, "case-backfill")
    manifest = runs_dir / "experiment-backfill.json"
    dump_json(
        {
            "presets": ["focus_variant"],
            "seeds": [1],
            "backends": [],
            "target_suite": "core",
            "target_suites": ["core"],
            "targets": [],
            "common_capabilities": [],
            "experiment_meta": {
                "matrix_id": "module_ablation",
                "matrix_title": "Module Ablation",
                "comparison_group": "module_ablation",
                "rq_tags": ["RQ2"],
                "analysis_tags": ["ablation"],
                "scope_by_target_suite": {"core": "core"},
                "variant_by_preset": {
                    "focus_variant": {
                        "variant_id": "focus_variant",
                        "variant_title": "focus_variant",
                        "base_preset": "stable_base",
                        "comparison_role": "contrast",
                        "component_focus": "semantic_normalizer",
                        "semantic_focus_families": ["join_membership"],
                        "semantic_focus_signals": ["row_value_absence_filter"],
                        "factors": {"semantic_normalizer": False},
                        "oracle_profile": "differential",
                        "analysis_tags": ["noise_control"],
                    }
                },
            },
            "runs": [
                {
                    "target_suite": "core",
                    "preset": "focus_variant",
                    "seed": 1,
                    "run_file": str(run_file),
                    "report": "",
                }
            ],
        },
        manifest,
    )

    _, csv_path = reporter.write_experiment_summary(manifest)
    row = next(csv.DictReader(csv_path.open(encoding="utf-8")))
    assert row["matrix_id"] == "module_ablation"
    assert row["comparison_group"] == "module_ablation"
    assert row["variant_id"] == "focus_variant"
    assert row["variant_label"] == "focus_variant"
    assert row["variant_group_id"] == "core|module_ablation|module_ablation"
    assert row["base_preset"] == "stable_base"
    assert row["comparison_role"] == "contrast"
    assert row["canonical_comparison_role"] == "contrast"
    assert row["component_focus"] == "semantic_normalizer"
    assert row["semantic_focus_families"] == "join_membership"
    assert row["semantic_focus_signals"] == "row_value_absence_filter"
    assert row["scope_kind"] == "core"
    assert row["oracle_profile"] == "differential"
    assert row["analysis_tags"] == "noise_control,ablation"


def test_write_experiment_summary_writes_structured_aggregate_json(tmp_path, monkeypatch):
    runs_dir = tmp_path / "runs"
    reports_dir = tmp_path / "reports"
    monkeypatch.setattr(reporter, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(reporter, "REPORTS_DIR", reports_dir)
    run_file = runs_dir / "run-aggregate-json.jsonl.gz"
    _write_run(run_file, "case-aggregate-json")
    manifest = runs_dir / "experiment-aggregate-json.json"
    dump_json(
        {
            "presets": ["baseline"],
            "seeds": [1],
            "backends": [],
            "target_suite": "core",
            "target_suites": ["core"],
            "targets": [],
            "common_capabilities": [],
            "evidence_mode": "ablation",
            "experiment_meta": {
                "matrix_id": "module_ablation",
                "matrix_title": "Module Ablation",
                "comparison_group": "module_ablation",
                "rq_tags": ["RQ2", "RQ4"],
                "analysis_tags": ["ablation"],
            },
            "runs": [
                {
                    "target_suite": "core",
                    "preset": "baseline",
                    "matrix_id": "module_ablation",
                    "matrix_title": "Module Ablation",
                    "comparison_group": "module_ablation",
                    "variant_id": "baseline",
                    "variant_title": "baseline",
                    "base_preset": "baseline",
                    "comparison_role": "baseline",
                    "component_focus": "",
                    "overlays": ["disable_normalizer"],
                    "semantic_focus_families": ["join_membership"],
                    "semantic_focus_signals": ["row_value_absence_filter"],
                    "factors": {"semantic_normalizer": False, "type_aware_generation": True},
                    "scope_kind": "core",
                    "oracle_profile": "differential",
                    "rq_tags": ["RQ2", "RQ4"],
                    "analysis_tags": ["ablation"],
                    "counts_as_real_bugs": False,
                    "seed": 1,
                    "run_file": str(run_file),
                    "report": "",
                }
            ],
        },
        manifest,
    )
    dump_json(
        {
            "elapsed_s": 0.5,
            "throughput_cases_s": 2.0,
            "backends": [],
            "targets": [],
            "common_capabilities": [],
            "config": {
                "effective_guidance_targets": ["semantic_family:join_membership"],
                "discovery_biases": [],
            },
            "stage_profile": {
                "totals_ms": {
                    "generate_mutate_ms": 1.0,
                    "backend_execution_ms": 2.0,
                    "normalize_ms": 3.0,
                    "oracle_classification_ms": 4.0,
                    "scheduler_feedback_ms": 5.0,
                    "logging_artifact_ms": 6.0,
                    "total_case_wall_ms": 21.0,
                },
                "avg_ms_per_case": {
                    "generate_mutate_ms": 1.0,
                    "backend_execution_ms": 2.0,
                    "normalize_ms": 3.0,
                    "oracle_classification_ms": 4.0,
                    "scheduler_feedback_ms": 5.0,
                    "logging_artifact_ms": 6.0,
                    "total_case_wall_ms": 21.0,
                },
                "share_of_total": {
                    "generate_mutate_ms": 1.0 / 21.0,
                    "backend_execution_ms": 2.0 / 21.0,
                    "normalize_ms": 3.0 / 21.0,
                    "oracle_classification_ms": 4.0 / 21.0,
                    "scheduler_feedback_ms": 5.0 / 21.0,
                    "logging_artifact_ms": 6.0 / 21.0,
                    "total_case_wall_ms": 1.0,
                },
            },
        },
        run_meta_path(run_file),
    )

    reporter.write_experiment_summary(manifest)
    aggregate_json = reports_dir / "experiment-summary-experiment-aggregate-json-aggregates.json"
    payload = json.loads(aggregate_json.read_text(encoding="utf-8"))

    assert payload["schema_version"] == "experiment-summary-aggregates-v1"
    assert payload["experiment_meta"]["matrix_id"] == "module_ablation"
    assert payload["variant_count"] == 1
    assert payload["run_count"] == 1
    assert payload["by_matrix"][0]["group_value"] == "module_ablation"
    assert payload["by_matrix"][0]["candidate_bug_cases"] == 0
    assert payload["by_scope_kind"][0]["group_value"] == "core"
    assert payload["by_oracle_profile"][0]["group_value"] == "differential"
    assert {"RQ2", "RQ4"}.issubset({item["group_value"] for item in payload["by_rq"]})
    factor_groups = {(item["factor_name"], item["factor_value"]) for item in payload["by_factor"]}
    assert ("semantic_normalizer", False) in factor_groups
    assert ("type_aware_generation", True) in factor_groups
    assert payload["by_factor"][0]["stage_profile"]["totals_ms"]["total_case_wall_ms"] == 21.0


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
                "signal_new_behavior": False,
                "guidance": {
                    "score": 2.0 + idx,
                    "features": [
                        "semantic_family:join_membership",
                        "semantic_signal:topk_filter_pushdown",
                    ],
                    "matched_targets": ["groupby", "topk"] if idx == 0 else ["groupby"],
                    "matched_semantic_targets": ["groupby", "topk"] if idx == 0 else ["groupby"],
                    "discovery_bias_hits": ["groupby|semantic_signal:"] if idx == 0 else [],
                    "matched_semantic_target_count": 2 if idx == 0 else 1,
                    "discovery_bias_hit_count": 1 if idx == 0 else 0,
                    "data_sensitivity": 0.5 + idx,
                    "path_coverage_proxy": 0.25 + idx,
                    "frontier_conformance": 0.75 + idx,
                    "discovery_diversity_bonus": 0.1 + idx,
                    "candidate_pool_diversity_bonus": 0.05 + idx,
                    "discovery_stale_penalty": -0.2 - idx,
                    "contribution_potential": 1.0 + idx,
                    "candidate_count": 4,
                    "contributing_candidate_count": 2,
                    "pruned_candidate_count": 1,
                    "feature_count": 3 + idx,
                    "frontier_bucket_count": 1 + idx,
                    "discovery_bucket_count": 2 + idx,
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
            "config": {
                "guidance_targets": ["groupby", "topk"],
                "effective_guidance_targets": [
                    "groupby",
                    "topk",
                    "semantic_family:join_membership",
                    "semantic_signal:topk_filter_pushdown",
                ],
                "semantic_focus_families": ["join_membership"],
                "semantic_focus_signals": ["topk_filter_pushdown"],
                "discovery_biases": [
                    {
                        "targets": ["groupby"],
                        "feature_prefixes": ["semantic_signal:"],
                        "keep_in_pool": True,
                        "score_bonus": 1.0,
                    }
                ],
            },
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

    md_text = md_path.read_text(encoding="utf-8")
    assert "data sensitivity" in md_text
    assert "discovery bonus" in md_text
    assert "raw new behavior % | signal new behavior %" in md_text
    assert row["new_behavior_rate"] == "0.5"
    assert row["signal_new_behavior_rate"] == "0.0"
    assert row["avg_guidance_score"] == "2.5"
    assert row["avg_data_sensitivity"] == "1.0"
    assert row["avg_path_coverage_proxy"] == "0.75"
    assert row["avg_frontier_conformance"] == "1.25"
    assert float(row["avg_discovery_diversity_bonus"]) == pytest.approx(0.6)
    assert float(row["avg_candidate_pool_diversity_bonus"]) == pytest.approx(0.55)
    assert float(row["avg_discovery_stale_penalty"]) == pytest.approx(-0.7)
    assert row["avg_contribution_potential"] == "1.5"
    assert row["pruned_candidate_rate"] == "0.25"
    assert row["configured_guidance_targets"] == "groupby,topk"
    assert row["configured_effective_guidance_targets"] == "groupby,topk,semantic_family:join_membership,semantic_signal:topk_filter_pushdown"
    assert row["configured_semantic_focus_families"] == "join_membership"
    assert row["configured_semantic_focus_signals"] == "topk_filter_pushdown"
    assert row["configured_discovery_biases"] == "groupby|semantic_signal:|keep+score=1"
    assert row["matched_semantic_target_cases"] == "2"
    assert row["discovery_bias_hit_cases"] == "1"
    assert float(row["matched_semantic_target_case_rate"]) == pytest.approx(1.0)
    assert float(row["discovery_bias_hit_case_rate"]) == pytest.approx(0.5)
    assert float(row["avg_matched_semantic_target_count"]) == pytest.approx(1.5)
    assert float(row["avg_discovery_bias_hit_count"]) == pytest.approx(0.5)
    assert row["top_matched_semantic_targets"] == "groupby:2; topk:1"
    assert row["top_discovery_bias_hits"] == "groupby|semantic_signal::1"
    assert row["top_observed_semantic_families"] == "join_membership:2"
    assert row["top_observed_semantic_signals"] == "topk_filter_pushdown:2"
    aggregate_csv = reports_dir / f"experiment-summary-{manifest.stem}-aggregates.csv"
    aggregate_row = next(csv.DictReader(aggregate_csv.open(encoding="utf-8")))
    assert float(aggregate_row["avg_discovery_diversity_bonus"]) == pytest.approx(0.6)
    assert float(aggregate_row["avg_candidate_pool_diversity_bonus"]) == pytest.approx(0.55)
    assert float(aggregate_row["avg_discovery_stale_penalty"]) == pytest.approx(-0.7)
    assert float(aggregate_row["avg_discovery_bucket_count"]) == pytest.approx(2.5)
    assert aggregate_row["configured_guidance_targets"] == "groupby,topk"
    assert aggregate_row["configured_effective_guidance_targets"] == "groupby,topk,semantic_family:join_membership,semantic_signal:topk_filter_pushdown"
    assert aggregate_row["configured_semantic_focus_families"] == "join_membership"
    assert aggregate_row["configured_semantic_focus_signals"] == "topk_filter_pushdown"
    assert aggregate_row["configured_discovery_biases"] == "groupby|semantic_signal:|keep+score=1"
    assert aggregate_row["matched_semantic_target_cases"] == "2"
    assert aggregate_row["discovery_bias_hit_cases"] == "1"
    assert aggregate_row["top_matched_semantic_targets"] == "groupby:2; topk:1"
    assert aggregate_row["top_observed_semantic_families"] == "join_membership:2"
    assert "## Semantic Scheduling" in md_text
    assert "configured discovery biases" in md_text


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
        "| core | edge_float | edge_float | 1 | 4 | 4 | 1 | 1 | 0 | 0 | 25.0% | "
        "5.00 | 0 | 0.1 | 1.00 | 0.00 | 0.00 | 0.00 | 2 | 1 |"
    ) in md
    assert aggregate_row["candidate_bug_case_rate"] == "0.25"
    assert aggregate_row["candidate_bug_cases_per_s"] == "5.0"
    assert aggregate_row["median_first_candidate_bug_case_index"] == "0.0"
    assert aggregate_row["median_first_candidate_bug_elapsed_s"] == "0.05"
    assert aggregate_row["avg_candidate_bug_discovery_auc"] == "1.0"
    assert aggregate_row["avg_scheduler_reward"] == "0.0"
    assert aggregate_row["avg_scheduler_reward_signal"] == "0.0"
    assert aggregate_row["avg_scheduler_mean_reward"] == "0.0"


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
            "family_saturation_filter": {
                "enabled": True,
                "filtered_candidates": 11,
                "fallback_candidates": 3,
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
    assert row["family_saturation_filter_enabled"] == "True"
    assert row["family_saturation_filter_filtered_candidates"] == "11"
    assert row["family_saturation_filter_fallback_candidates"] == "3"
    assert "organic:1" in row["top_discovery_origins"]
    assert "issue_replay:1" in row["top_discovery_origins"]
    assert row["top_candidate_bug_families"] == "groupby_aggregation@datafusion:1"
    assert aggregate_row["rewardable_candidate_implementation_bug_count"] == "1"
    assert aggregate_row["family_saturation_filter_filtered_candidates"] == "11"
    assert aggregate_row["family_saturation_filter_fallback_candidates"] == "3"
    assert aggregate_row["family_saturation_filter_filtered_per_case"] == "11.0"
    assert aggregate_row["family_saturation_filter_fallback_per_case"] == "3.0"
    assert "rewardable candidates" in md
    assert "Replay bug policy: enable_replay_bug=false, source_issues=2" in md
    assert "Family Saturation Filter" in md
    assert "| latest_all_engines | live_cross_family | live_cross_family | 1 | 1 | 11 | 3 | 11.00 | 3.00 |" in md


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
            "quality_oracles": [
                {"name": "mutation", "verdict": "productive_mutation", "passed": True, "score": 1.5},
                {"name": "feedback", "verdict": "finding_yield", "passed": True, "score": 1.0},
                {"name": "guidance", "verdict": "guided_productive", "passed": True, "score": 3.0},
            ],
            "candidate_source": "feedback_mutation",
            "stored_in_feedback_corpus": True,
            "feedback_summary": {
                "candidate_source": "feedback_mutation",
                "stored_in_feedback_corpus": True,
                "quality_oracle_count": 3,
                "quality_pass_count": 3,
                "quality_fail_count": 0,
                "quality_score_total": 5.5,
                "source_reward_adjustment": 0.4,
                "guidance_reward_adjustment": 0.25,
                "seed_schedule_delta": 3.6,
                "mutation_oracle_verdict": "productive_mutation",
                "feedback_oracle_verdict": "finding_yield",
                "guidance_oracle_verdict": "guided_productive",
            },
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
            "adaptive_state": [
                {
                    "arm_id": "datafusion_cross:baseline:seed1",
                    "target_suite": "datafusion_cross",
                    "preset": "baseline",
                    "pulls": 3,
                    "mean_reward": 4.5,
                    "reward_signal": 3.75,
                    "last_reward": 4.25,
                    "stale_batches": 1,
                }
            ],
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
    assert "## Closed-Loop Feedback" in md
    assert "- Global first candidate case: 1" in md
    assert "- Batches by suite: datafusion_cross:1" in md
    assert "- Cases by suite: datafusion_cross:1" in md
    assert "- Candidate cases by suite: datafusion_cross:1" in md
    assert "| datafusion_cross:baseline:seed1 | datafusion_cross | 3 | 4.50 | 3.75 | 4.25 | 1 |" in md
    assert row["batch_index"] == "3"
    assert row["schedule_arm_id"] == "datafusion_cross:baseline:seed1"
    assert row["scheduler_reward"] == "4.25"
    assert row["scheduler_reward_signal"] == "3.75"
    assert row["scheduler_mean_reward"] == "4.5"
    assert row["scheduler_last_reward"] == "4.25"
    assert row["scheduler_pulls"] == "3"
    assert row["scheduler_stale_batches"] == "1"
    assert row["feedback_mutation_cases"] == "1"
    assert row["quality_pass_count"] == "3"
    assert row["source_reward_adjustment_per_case"] == "0.5"
    assert row["feedback_operator_affinity_hit_rate"] == "0.0"
    assert aggregate_row["feedback_mutation_case_rate"] == "1.0"
    assert aggregate_row["quality_pass_rate"] == "1.0"
    assert float(aggregate_row["seed_schedule_delta_per_case"]) == pytest.approx(1.17)
    assert aggregate_row["avg_scheduler_reward"] == "4.25"
    assert aggregate_row["avg_scheduler_reward_signal"] == "3.75"
    assert aggregate_row["avg_scheduler_mean_reward"] == "4.5"


def test_write_experiment_summary_reports_feedback_operator_selection_telemetry(tmp_path, monkeypatch):
    runs_dir = tmp_path / "runs"
    reports_dir = tmp_path / "reports"
    monkeypatch.setattr(reporter, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(reporter, "REPORTS_DIR", reports_dir)
    run_file = runs_dir / "run-feedback-selection.jsonl.gz"
    append_jsonl(
        {
            "status": "ok",
            "case_index": 0,
            "case": {"case_id": "case-0", "seed": 0, "program": {"operations": []}},
            "findings": [],
            "behavior_signature": "sig-0",
            "backend_status": {},
            "quality_oracles": [],
            "candidate_source": "feedback_mutation",
            "feedback_selection": {
                "target_keys": [
                    "semantic_family:conditional_semantics",
                    "semantic_signal:left_join_case_when_membership",
                ],
                "selected_operator": "append_left_join_case_membership",
                "selected_operator_score": 1.75,
            },
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
    manifest = runs_dir / "experiment-feedback-selection.json"
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
    row = next(csv.DictReader(csv_path.open(encoding="utf-8")))
    aggregate_csv_path = reports_dir / "experiment-summary-experiment-feedback-selection-aggregates.csv"
    aggregate_row = next(csv.DictReader(aggregate_csv_path.open(encoding="utf-8")))
    md = md_path.read_text(encoding="utf-8")

    assert row["feedback_target_key_count"] == "2"
    assert row["feedback_semantic_family_target_count"] == "1"
    assert row["feedback_semantic_signal_target_count"] == "1"
    assert row["feedback_operator_affinity_hit_cases"] == "1"
    assert row["top_feedback_selected_operators"] == "append_left_join_case_membership:1"
    assert "## Feedback Operator Selection" in md
    assert aggregate_row["feedback_operator_affinity_hit_rate"] == "1.0"
    assert aggregate_row["feedback_selected_operator_score_avg"] == "1.75"


def test_write_experiment_summary_reports_adaptive_selection_telemetry(tmp_path, monkeypatch):
    runs_dir = tmp_path / "runs"
    reports_dir = tmp_path / "reports"
    monkeypatch.setattr(reporter, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(reporter, "REPORTS_DIR", reports_dir)
    run_file = runs_dir / "run-adaptive-selection-summary.jsonl.gz"
    for idx, profile in enumerate(["common", "discovery_fresh"]):
        append_jsonl(
            {
                "status": "ok",
                "case_index": idx,
                "case": {"case_id": f"case-{idx}", "seed": idx, "program": {"operations": []}},
                "findings": [],
                "behavior_signature": f"sig-{idx}",
                "backend_status": {},
                "quality_oracles": [],
                "is_new_behavior": True,
                "generator_profile_selection": {
                    "strategy": "contextual_bandit",
                    "profile": profile,
                    "profile_pool": ["common", "discovery_fresh"],
                    "learning_weight": 1.0,
                    "reward": 1.0 + idx,
                    "ranked": [
                        {
                            "action_id": profile,
                            "score": 1.0 + idx,
                            "model_prediction": 0.5 + idx,
                            "uncertainty": 0.25 + idx,
                            "exploration_bonus": 0.4,
                            "version_signal": 0.2,
                            "continual_priority_signal": 0.1,
                            "health_penalty": 0.05,
                        }
                    ],
                },
                "semantic_objective_selection": {
                    "strategy": "contextual_bandit",
                    "scope": "semantic_objective",
                    "action": "exploration_objective:boundary_depth",
                    "action_pool": ["exploration_objective:boundary_depth"],
                    "learning_weight": 1.0,
                    "reward": 0.5,
                    "ranked": [{"action_id": "exploration_objective:boundary_depth", "uncertainty": 0.75}],
                },
                "metamorphic_relation_selection": {
                    "strategy": "contextual_bandit_warmup",
                    "scope": "metamorphic_relation",
                    "action": "input_partition_union_all",
                    "action_pool": ["input_partition_union_all"],
                    "learning_weight": 1.0,
                    "reward": 0.25,
                    "ranked": [{"action_id": "input_partition_union_all", "exploration_bonus": 0.9}],
                },
                "version_pair_selection": {
                    "strategy": "contextual_bandit",
                    "scope": "version_pair",
                    "action": "latest->fixed",
                    "action_pool": ["latest->fixed"],
                    "learning_weight": 1.0,
                    "reward": 0.75,
                    "ranked": [{"action_id": "latest->fixed", "version_signal": 0.8}],
                },
                "selected_generator_profile": profile,
                "selected_semantic_objective": "exploration_objective:boundary_depth",
                "selected_metamorphic_relation": "input_partition_union_all",
                "selected_version_pair": "latest->fixed",
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
        },
        run_meta_path(run_file),
    )
    manifest = runs_dir / "experiment-adaptive-selection.json"
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
    row = next(csv.DictReader(csv_path.open(encoding="utf-8")))
    aggregate_csv_path = reports_dir / "experiment-summary-experiment-adaptive-selection-aggregates.csv"
    aggregate_json_path = reports_dir / "experiment-summary-experiment-adaptive-selection-aggregates.json"
    aggregate_row = next(csv.DictReader(aggregate_csv_path.open(encoding="utf-8")))
    aggregate_payload = json.loads(aggregate_json_path.read_text(encoding="utf-8"))
    md = md_path.read_text(encoding="utf-8")

    assert "## Adaptive Selection Telemetry" in md
    assert "common:1; discovery_fresh:1" in md
    assert "latest->fixed:2" in md
    assert row["adaptive_selection_total_count"] == "8"
    assert row["adaptive_selection_total_per_case"] == "4.0"
    assert row["adaptive_generator_profile_selection_count"] == "2"
    assert row["adaptive_generator_profile_top_actions"] == "common:1; discovery_fresh:1"
    assert row["adaptive_generator_profile_avg_reward"] == "1.5"
    assert row["adaptive_generator_profile_avg_uncertainty"] == "0.75"
    assert aggregate_row["adaptive_selection_total_count"] == "8"
    assert aggregate_row["adaptive_selection_total_per_case"] == "4.0"
    assert aggregate_row["adaptive_version_pair_top_actions"] == "latest->fixed:2"
    assert aggregate_row["adaptive_version_pair_avg_version_signal"] == "0.8"
    assert (
        aggregate_payload["by_target_suite"][0]["adaptive_selection"]["adaptive_selection_total_count"]
        == 8
    )


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
    assert "| core | baseline | baseline | 1 | 2 | 0.0% | 50.0% | 50.0% |" in md
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
            "presets": ["discovery"],
            "seeds": [14],
            "backends": ["a", "b", "c"],
            "target_suite": "custom",
            "targets": [],
            "common_capabilities": [],
            "runs": [{"preset": "discovery", "seed": 14, "run_file": str(run_file), "report": ""}],
        },
        manifest,
    )

    md_path, csv_path = reporter.write_experiment_summary(manifest, refresh=True)
    row = next(csv.DictReader(csv_path.open(encoding="utf-8")))
    md = md_path.read_text(encoding="utf-8")

    assert "Refreshed with current oracle: yes" in md
    assert row["top_root_causes"] == "float_group_key_instability:1"
    assert row["top_candidate_bug_families"] == "float_group_key_instability@b:1"
