from datadiff.icse_experiment_quality import (
    ICSE_EXPERIMENT_QUALITY_SCHEMA_VERSION,
    score_final_readiness_summary,
    score_methodology_report,
)


def test_score_methodology_report_rewards_bug_yield_throughput_coverage_and_reproducibility():
    report = {
        "coverage": {
            "target_suites": ["core", "arrow_cross"],
            "presets": ["baseline", "guided"],
            "matrix_ids": ["final_live"],
            "comparison_groups": ["live"],
            "variant_ids": ["baseline", "guided"],
            "scope_kinds": ["dataframe"],
            "oracle_profiles": ["differential"],
            "rq_tags": ["RQ1"],
            "analysis_tags": ["live"],
            "semantic_focus_families": ["join"],
            "semantic_focus_signals": ["null_sort"],
            "evidence_modes": {"live": 2},
            "target_suite_count": 2,
            "preset_count": 2,
            "run_count": 2,
        },
        "efficiency": {"cases": 1000, "elapsed_s": 100.0, "cases_per_s": 10.0},
        "bug_discovery": {
            "candidate_bug_family_count": 2,
            "candidate_bug_families": {"family_a@duckdb": 3, "family_b@polars": 1},
            "first_candidate": {"elapsed_s": 30.0},
            "avg_candidate_bug_discovery_auc": 0.5,
        },
        "candidate_pipeline": {
            "reproduced_count": 1,
            "candidate_bug_verdict_count": 1,
            "recheck_pass_rate": 1.0,
        },
        "reproducibility": {
            "run_logs_exist": 2,
            "run_logs_total": 2,
            "artifact_reproducer_coverage": 1.0,
            "issue_bundle": {"clean_execution": True},
        },
    }

    quality = score_methodology_report(
        report,
        thresholds={
            "target_real_bug_families": 2,
            "target_reproduced_bug_families": 1,
            "target_throughput_cases_s": 5.0,
            "target_coverage_axes": 8,
            "target_first_bug_elapsed_s": 60.0,
            "target_discovery_auc": 0.2,
        },
    )

    assert quality["schema_version"] == ICSE_EXPERIMENT_QUALITY_SCHEMA_VERSION
    assert quality["ready_for_icse_claim"] is True
    assert quality["grade"] in {"A", "B"}
    assert set(quality["dimensions"]) == {
        "real_bug_yield",
        "throughput",
        "coverage",
        "speed",
        "reproducibility",
    }
    assert all(item["passed"] for item in quality["dimensions"].values())


def test_score_final_readiness_summary_identifies_weak_dimensions():
    summary = {
        "rewardable_live_candidate_families": {},
        "confirmed_live_candidate_families": {},
        "external_confirmed_live_candidate_families": {},
        "runtime_efficiency": {"avg_throughput_cases_s": 0.5, "min_throughput_cases_s": 0.2},
        "final_matrix_coverage": {"observed_matrix_ids": ["live"], "missing_matrix_ids": ["ablation"]},
        "live_suites": ["core"],
        "semantic_focus_families": [],
        "semantic_focus_signals": [],
        "discovery_responsiveness": {
            "observed_run_count": 1,
            "best_first_candidate_elapsed_s": None,
            "avg_candidate_bug_discovery_auc": 0.0,
        },
        "paper_run_journal_required_runs": 2,
        "paper_run_journal_covered_runs": 1,
        "closed_loop_state_persistence": {"required_run_count": 2, "persisted_run_count": 0},
    }

    quality = score_final_readiness_summary(summary)

    assert quality["ready_for_icse_claim"] is False
    assert quality["dimensions"]["real_bug_yield"]["passed"] is False
    assert quality["dimensions"]["speed"]["passed"] is False
    assert quality["optimization_priorities"][0]["dimension"] in {
        "real_bug_yield",
        "speed",
        "coverage",
    }


def test_score_methodology_report_tolerates_missing_and_dirty_numbers():
    quality = score_methodology_report(
        {
            "efficiency": {"cases_per_s": "nan"},
            "coverage": {"target_suites": "core,,"},
            "bug_discovery": {"candidate_bug_family_count": -1},
            "reproducibility": {"run_logs_exist": 1, "run_logs_total": 0},
        }
    )

    assert quality["schema_version"] == ICSE_EXPERIMENT_QUALITY_SCHEMA_VERSION
    assert quality["overall_score"] >= 0.0
    assert quality["overall_score"] <= 100.0
    assert quality["dimensions"]["throughput"]["observed"] == 0.0
