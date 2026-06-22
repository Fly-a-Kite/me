import json
from pathlib import Path

from datadiff.run_journal import build_run_journal_entry, record_run_journal
from datadiff.util import append_jsonl, read_jsonl, run_meta_path


def _polluted_signal_rows():
    return [
        {
            "case": {"case_id": "resolved", "seed": 1},
            "case_index": 0,
            "elapsed_s": 0.1,
            "is_new_behavior": True,
            "signal_new_behavior": True,
            "findings": [
                {
                    "root_cause": "nan_inf_semantics",
                    "triage_verdict": "expected_semantic_divergence",
                    "suspicious_backends": ["duckdb"],
                }
            ],
        },
        {
            "case": {"case_id": "false-positive", "seed": 2},
            "case_index": 1,
            "elapsed_s": 0.2,
            "is_new_behavior": True,
            "signal_new_behavior": True,
            "findings": [
                {
                    "root_cause": "order_only_normalization_mismatch",
                    "triage_verdict": "normalizer_false_positive",
                    "false_positive": True,
                    "suspicious_backends": ["sqlite"],
                }
            ],
        },
        {
            "case": {"case_id": "source-issue", "seed": 3},
            "case_index": 2,
            "elapsed_s": 0.3,
            "is_new_behavior": True,
            "signal_new_behavior": True,
            "findings": [
                {
                    "root_cause": "csv_long_numeric_roundtrip",
                    "triage_verdict": "candidate_implementation_bug",
                    "source_issue": "duckdb/duckdb#12345",
                    "suspicious_backends": ["duckdb"],
                }
            ],
        },
        {
            "case": {"case_id": "pure-behavior", "seed": 4},
            "case_index": 3,
            "elapsed_s": 0.4,
            "is_new_behavior": True,
            "signal_new_behavior": True,
            "findings": [],
        },
        {
            "case": {"case_id": "candidate", "seed": 5},
            "case_index": 4,
            "elapsed_s": 0.5,
            "is_new_behavior": True,
            "signal_new_behavior": True,
            "findings": [
                {
                    "root_cause": "topk_filter_pushdown",
                    "triage_verdict": "candidate_implementation_bug",
                    "suspicious_backends": ["datafusion"],
                }
            ],
        },
        {
            "case": {"case_id": "semantic-needs-confirmation", "seed": 6},
            "case_index": 5,
            "elapsed_s": 0.6,
            "is_new_behavior": True,
            "signal_new_behavior": True,
            "findings": [
                {
                    "root_cause": "string_expression",
                    "triage_verdict": "semantic_divergence_needs_confirmation",
                    "suspicious_backends": ["sqlite"],
                }
            ],
        },
    ]


def test_build_run_journal_entry_records_paper_facing_summary(tmp_path):
    run_file = tmp_path / "run-x.jsonl"
    append_jsonl(
        {
            "case": {"case_id": "case-1", "seed": 1},
            "case_index": 0,
            "elapsed_s": 1.25,
            "is_new_behavior": True,
            "signal_new_behavior": False,
            "findings": [
                {
                    "root_cause": "grouped_topk_null_sort_key",
                    "triage_verdict": "candidate_implementation_bug",
                    "suspicious_backends": ["datafusion"],
                    "false_positive": False,
                }
            ],
        },
        run_file,
    )
    run_meta_path(run_file).write_text(
        json.dumps(
            {
                "executed_cases": 1,
                "elapsed_s": 1.3,
                "throughput_cases_s": 0.77,
                "seed": 1,
                "backends": ["pandas", "duckdb", "datafusion"],
                "targets": [{"family": "sql", "layer": "query_engine"}],
                "common_capabilities": ["sort", "limit"],
                "config": {
                    "generator_profile": "null_agg_topk",
                    "enable_witness_oracle": True,
                    "enable_replay_bug": False,
                    "replay_bug_source_issues": [
                        "https://github.com/apache/datafusion/issues/22190",
                        "https://github.com/duckdb/duckdb/issues/22075",
                    ],
                    "guidance_strategy": "guided",
                    "guidance_candidate_pool": 8,
                    "guidance_targets": ["aggregation", "sort_limit"],
                },
                "replay_bug_filter": {
                    "enabled": True,
                    "filtered_candidates": 4,
                    "fallback_candidates": 1,
                },
                "experiment_meta": {
                    "matrix_id": "live_discovery",
                    "comparison_group": "latest_live_discovery",
                    "analysis_tags": ["live"],
                    "scope_by_target_suite": {"datafusion_cross": "cross_ecosystem"},
                    "variant_by_preset": {
                        "live_datafusion": {
                            "variant_id": "live_datafusion",
                            "base_preset": "baseline",
                            "comparison_role": "targeted",
                            "analysis_tags": ["guided"],
                            "oracle_profile": "differential",
                        }
                    },
                },
                "run_provenance": {
                    "vcs": {"git_commit": "abc123", "git_branch": "main", "workspace_dirty": False},
                    "launch": {"source": "closed_loop_tmux", "duration": "24h"},
                    "harness": {
                        "authority": True,
                        "freeze_intent": True,
                        "latest_code_claim": True,
                        "evidence_role": "latest_live_authority_24h",
                    },
                    "freeze_artifacts": {"strategy_snapshot": "reports/strategy-snapshots/frozen.json"},
                },
            }
        ),
        encoding="utf-8",
    )

    entry = build_run_journal_entry(
        run_file,
        {
            "command": "experiment",
            "theme": "paper run",
            "evidence_mode": "live",
            "target_suite": "datafusion_cross",
            "preset": "live_datafusion",
        },
    )

    assert entry["theme"] == "paper run"
    assert entry["evidence_mode"] == "live"
    assert entry["matrix_id"] == "live_discovery"
    assert entry["comparison_group"] == "latest_live_discovery"
    assert entry["variant_id"] == "live_datafusion"
    assert entry["variant_label"] == "live_datafusion"
    assert entry["variant_group_id"] == "datafusion_cross|latest_live_discovery|live_discovery"
    assert entry["comparison_role"] == "targeted"
    assert entry["canonical_comparison_role"] == "contrast"
    assert entry["scope_kind"] == "cross_ecosystem"
    assert entry["analysis_tags"] == ["guided", "live"]
    assert entry["executed_cases"] == 1
    assert entry["common_capabilities_count"] == 2
    assert entry["result_summary"]["candidate_bug_family_count"] == 1
    assert entry["result_summary"]["candidate_bug_families"] == {
        "grouped_topk_null_sort_key@datafusion": 1
    }
    assert entry["result_summary"]["new_behavior_cases"] == 1
    assert entry["result_summary"]["new_behavior_rate"] == 1.0
    assert entry["result_summary"]["signal_new_behavior_cases"] == 0
    assert entry["result_summary"]["signal_new_behavior_rate"] == 0.0
    assert entry["result_summary"]["first_candidate_bug_elapsed_s"] == 1.25
    assert entry["config_summary"]["enable_replay_bug"] is False
    assert entry["config_summary"]["witness_oracle"] is True
    assert entry["replay_bug_policy"] == {
        "enable_replay_bug": False,
        "source_issue_count": 2,
        "filter_enabled": True,
        "filtered_candidates": 4,
        "fallback_candidates": 1,
    }
    assert entry["run_provenance"] == {
        "git_commit": "abc123",
        "git_branch": "main",
        "workspace_dirty": False,
        "authority": True,
        "freeze_intent": True,
        "latest_code_claim": True,
        "evidence_role": "latest_live_authority_24h",
        "launch_source": "closed_loop_tmux",
        "launch_duration": "24h",
        "strategy_snapshot": "reports/strategy-snapshots/frozen.json",
    }


def test_build_run_journal_entry_filters_non_rewardable_signal_new_behavior(tmp_path):
    run_file = tmp_path / "run-filtered-signal.jsonl"
    for row in _polluted_signal_rows():
        append_jsonl(row, run_file)
    run_meta_path(run_file).write_text(
        json.dumps(
            {
                "executed_cases": 99,
                "new_behavior_cases": 99,
                "signal_new_behavior_cases": 99,
                "elapsed_s": 1.0,
                "throughput_cases_s": 6.0,
            }
        ),
        encoding="utf-8",
    )

    entry = build_run_journal_entry(run_file, {"theme": "filtered signal"})
    summary = entry["result_summary"]

    assert entry["executed_cases"] == 6
    assert summary["new_behavior_cases"] == 6
    assert summary["signal_new_behavior_cases"] == 3
    assert summary["signal_new_behavior_rate"] == 0.5
    assert summary["candidate_bug_cases"] == 1
    assert summary["candidate_bug_families"] == {"topk_filter_pushdown@datafusion": 1}
    assert summary["semantic_divergence_findings"] == 1
    assert summary["false_positive_findings"] == 1


def test_record_run_journal_appends_jsonl_and_markdown(tmp_path):
    run_file = tmp_path / "run-y.jsonl"
    journal_file = tmp_path / "paper-run-journal.jsonl"
    append_jsonl({"case": {"case_id": "case-2", "seed": 2}, "findings": []}, run_file)

    jsonl_path, md_path = record_run_journal(
        run_file,
        context={"theme": "empty live run", "evidence_mode": "live"},
        journal_file=journal_file,
    )

    assert jsonl_path == journal_file
    assert md_path == journal_file.with_suffix(".md")
    rows = read_jsonl(journal_file)
    assert rows[0]["theme"] == "empty live run"
    assert rows[0]["result_summary"]["raw_findings"] == 0
    md = Path(md_path).read_text(encoding="utf-8")
    assert "empty live run" in md
    assert "Raw novelty %" in md
    assert "Signal novelty %" in md
    assert "Raw/signal new behavior cases" in md
    assert "Replay policy" in md
    assert "Experiment identity" in md
    assert "Variant/preset" in md
    assert "canonical_role" in md


def test_run_journal_marks_validation_as_non_bug_evidence(tmp_path):
    run_file = tmp_path / "run-validation.jsonl"
    append_jsonl({"case": {"case_id": "case-3", "seed": 3}, "findings": []}, run_file)

    entry = build_run_journal_entry(
        run_file,
        context={"theme": "validation smoke", "evidence_mode": "validation"},
    )

    assert entry["evidence_mode"] == "validation"
    assert "do not count as final bug evidence" in entry["counting_policy"]


def test_run_journal_marks_ablation_and_comparison_as_rq_only_evidence(tmp_path):
    run_file = tmp_path / "run-support.jsonl"
    append_jsonl({"case": {"case_id": "case-4", "seed": 4}, "findings": []}, run_file)

    ablation = build_run_journal_entry(
        run_file,
        context={"theme": "module ablation", "evidence_mode": "ablation"},
    )
    comparison = build_run_journal_entry(
        run_file,
        context={"theme": "baseline comparison", "evidence_mode": "comparison"},
    )

    assert "RQ tables only" in ablation["counting_policy"]
    assert "RQ tables only" in comparison["counting_policy"]
    assert "do not count candidates as live bug evidence" in ablation["counting_policy"]
    assert "do not count candidates as live bug evidence" in comparison["counting_policy"]


def test_run_journal_prefers_context_experiment_meta_over_weak_meta_defaults(tmp_path):
    run_file = tmp_path / "run-context-meta.jsonl"
    append_jsonl({"case": {"case_id": "case-5", "seed": 5}, "findings": []}, run_file)
    run_meta_path(run_file).write_text(
        json.dumps(
            {
                "executed_cases": 1,
                "elapsed_s": 0.5,
                "throughput_cases_s": 2.0,
            }
        ),
        encoding="utf-8",
    )

    entry = build_run_journal_entry(
        run_file,
        context={
            "theme": "validation structured",
            "evidence_mode": "validation",
            "target_suite": "datafusion_cross",
            "preset": "live_datafusion_fresh",
            "experiment_meta": {
                "matrix_id": "final_validation",
                "comparison_group": "validation_smoke",
                "analysis_tags": ["validation"],
                "scope_by_target_suite": {"datafusion_cross": "query_engine"},
                "variant_by_preset": {
                    "live_datafusion_fresh": {
                        "variant_id": "live_datafusion_fresh",
                        "variant_title": "live_datafusion_fresh",
                        "base_preset": "live_datafusion",
                        "comparison_role": "contrast",
                        "analysis_tags": ["guided"],
                        "oracle_profile": "differential",
                        "scope_kind": "query_engine",
                    }
                },
            },
        },
    )

    assert entry["matrix_id"] == "final_validation"
    assert entry["comparison_group"] == "validation_smoke"
    assert entry["variant_id"] == "live_datafusion_fresh"
    assert entry["canonical_comparison_role"] == "contrast"
    assert entry["scope_kind"] == "query_engine"
    assert entry["analysis_tags"] == ["guided", "validation"]
