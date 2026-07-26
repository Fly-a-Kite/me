import pytest
from datadiff.reward import (
    analyze_finding_outcomes,
    aggregate_feedback_summaries,
    feedback_summary_for_case,
    candidate_bug_family_keys,
    candidate_bug_signatures,
    coerce_case_feedback_summary,
    is_known_saturated_candidate_bug_finding,
    is_rewardable_candidate_bug_finding,
    offline_finding_buckets,
    online_case_reward,
    row_reward_signals,
    summarize_case_feedback,
)


def test_reward_signals_do_not_count_issue_replay_as_discovery_candidate():
    row = {
        "findings": [
            {
                "triage_verdict": "candidate_implementation_bug",
                "root_cause": "groupby_aggregation",
                "suspicious_backends": ["datafusion"],
                "signature": "organic-sig",
                "discovery_origin": "organic",
            },
            {
                "triage_verdict": "candidate_implementation_bug",
                "root_cause": "group_quantile_key_expression",
                "suspicious_backends": ["polars", "polars_lazy"],
                "signature": "replay-sig",
                "discovery_origin": "issue_replay",
            },
        ]
    }

    signals = row_reward_signals(row)

    assert signals["candidate_bug"] is True
    assert signals["candidate_bug_count"] == 1
    assert signals["issue_replay_candidate_bug_count"] == 1


def test_candidate_family_and_signature_rewards_skip_issue_replays():
    findings = [
        {
            "triage_verdict": "candidate_implementation_bug",
            "root_cause": "groupby_aggregation",
            "suspicious_backends": ["datafusion"],
            "signature": "organic-sig",
            "discovery_origin": "organic",
        },
        {
            "triage_verdict": "candidate_implementation_bug",
            "root_cause": "group_quantile_key_expression",
            "suspicious_backends": ["polars", "polars_lazy"],
            "signature": "replay-sig",
            "discovery_origin": "issue_replay",
        },
    ]

    assert candidate_bug_family_keys(findings) == {"groupby_aggregation@datafusion": 1}
    assert candidate_bug_signatures(findings) == {"organic-sig": 1}


def test_finding_outcome_analysis_matches_reward_helpers():
    findings = [
        {
            "triage_verdict": "candidate_implementation_bug",
            "root_cause": "groupby_aggregation",
            "suspicious_backends": ["datafusion"],
            "signature": "organic-sig",
            "discovery_origin": "organic",
        },
        {
            "triage_verdict": "candidate_implementation_bug",
            "root_cause": "metamorphic_groupby_aggregation",
            "suspicious_backends": ["datafusion"],
            "signature": "meta-sig",
            "discovery_origin": "organic",
        },
        {
            "triage_verdict": "candidate_implementation_bug",
            "root_cause": "group_quantile_key_expression",
            "suspicious_backends": ["polars", "polars_lazy"],
            "signature": "replay-sig",
            "discovery_origin": "issue_replay",
        },
        {
            "triage_verdict": "semantic_divergence_needs_confirmation",
            "root_cause": "string_expression",
            "suspicious_backends": ["sqlite"],
            "signature": "needs-confirmation",
        },
    ]
    row = {"findings": findings}

    analysis = analyze_finding_outcomes(findings)

    assert analysis.reward_signals() == row_reward_signals(row)
    assert analysis.candidate_bug_families == candidate_bug_family_keys(findings)
    assert analysis.candidate_bug_signatures == candidate_bug_signatures(findings)


def test_reward_signals_do_not_count_known_saturated_family_as_fresh_candidate():
    finding = {
        "triage_verdict": "candidate_implementation_bug",
        "root_cause": "grouped_topk_null_sort_key",
        "suspicious_backends": ["datafusion"],
        "signature": "known-family-sig",
        "discovery_origin": "organic",
    }
    known = ["grouped_topk_null_sort_key@datafusion"]

    signals = row_reward_signals({"findings": [finding]}, known_saturated_bug_families=known)

    assert is_known_saturated_candidate_bug_finding(finding, known) is True
    assert signals["candidate_bug"] is False
    assert signals["candidate_bug_count"] == 0
    assert signals["known_saturated_candidate_bug_count"] == 1
    assert candidate_bug_family_keys([finding], known_saturated_bug_families=known) == {}
    assert candidate_bug_signatures([finding], known_saturated_bug_families=known) == {}


def test_known_saturated_family_matches_any_suspicious_backend():
    finding = {
        "triage_verdict": "candidate_implementation_bug",
        "root_cause": "csv_long_numeric_roundtrip",
        "suspicious_backends": ["duckdb", "pyarrow"],
        "signature": "known-csv-roundtrip",
        "discovery_origin": "organic",
    }
    known = ["csv_long_numeric_roundtrip@duckdb"]

    signals = row_reward_signals({"findings": [finding]}, known_saturated_bug_families=known)

    assert is_known_saturated_candidate_bug_finding(finding, known) is True
    assert signals["candidate_bug"] is False
    assert signals["candidate_bug_count"] == 0
    assert signals["known_saturated_candidate_bug_count"] == 1


def test_source_issue_candidates_are_not_fresh_online_discoveries():
    finding = {
        "triage_verdict": "candidate_implementation_bug",
        "root_cause": "csv_long_numeric_roundtrip",
        "suspicious_backends": ["duckdb"],
        "signature": "source-issue-sig",
        "discovery_origin": "organic",
        "source_issue": "duckdb/duckdb#12345",
    }
    row = {
        "findings": [finding],
        "preflight": {"valid": True, "fallback_used": False},
    }

    signals = row_reward_signals(row)
    buckets = offline_finding_buckets([finding])

    assert is_rewardable_candidate_bug_finding(finding) is False
    assert signals["candidate_bug"] is False
    assert signals["candidate_bug_count"] == 0
    assert signals["source_issue_candidate_bug_count"] == 1
    assert candidate_bug_family_keys([finding]) == {}
    assert candidate_bug_signatures([finding]) == {}
    assert buckets["known_bug"] == 1
    assert online_case_reward(row) < 0.0


def test_resolved_semantic_divergence_is_not_positive_reward_signal():
    row = {
        "findings": [
            {
                "triage_verdict": "expected_semantic_divergence",
                "root_cause": "groupby_aggregation",
                "suspicious_backends": ["duckdb"],
                "signature": "resolved-semantic-divergence",
            }
        ],
        "preflight": {"valid": True, "fallback_used": False},
    }

    signals = row_reward_signals(row)

    assert signals["semantic_divergence"] is True
    assert signals["semantic_divergence_count"] == 1
    assert signals["resolved_semantic_divergence_count"] == 1
    assert signals["rewardable_semantic_divergence"] is False
    assert signals["needs_confirmation"] is False
    assert online_case_reward(row) < 0.0


def test_semantic_divergence_needs_confirmation_gets_small_positive_reward():
    row = {
        "findings": [
            {
                "triage_verdict": "semantic_divergence_needs_confirmation",
                "root_cause": "string_expression",
                "suspicious_backends": ["sqlite"],
                "signature": "needs-confirmation",
            }
        ],
        "preflight": {"valid": True, "fallback_used": False},
    }

    signals = row_reward_signals(row)

    assert signals["semantic_divergence_count"] == 1
    assert signals["resolved_semantic_divergence_count"] == 0
    assert signals["rewardable_semantic_divergence"] is True
    assert signals["needs_confirmation"] is True
    assert online_case_reward(row) > 0.0


def test_manual_confirmation_is_not_rewardable_semantic_divergence():
    row = {
        "findings": [
            {
                "triage_verdict": "needs_manual_confirmation",
                "root_cause": "unknown",
                "suspicious_backends": ["duckdb", "pandas"],
                "signature": "manual-confirmation",
            }
        ],
        "preflight": {"valid": True, "fallback_used": False},
    }

    signals = row_reward_signals(row)

    assert signals["semantic_divergence"] is False
    assert signals["semantic_divergence_count"] == 0
    assert signals["rewardable_semantic_divergence"] is False
    assert signals["needs_confirmation"] is True
    assert online_case_reward(row) < 0.0


def test_offline_finding_buckets_match_paper_triage_categories():
    findings = [
        {
            "triage_verdict": "candidate_implementation_bug",
            "root_cause": "fresh_root",
            "suspicious_backends": ["polars"],
            "discovery_origin": "organic",
        },
        {
            "triage_verdict": "candidate_implementation_bug",
            "root_cause": "known_root",
            "suspicious_backends": ["duckdb"],
            "discovery_origin": "organic",
        },
        {
            "triage_verdict": "candidate_implementation_bug",
            "root_cause": "replayed_root",
            "suspicious_backends": ["datafusion"],
            "discovery_origin": "issue_replay",
        },
        {
            "triage_verdict": "generator_false_positive",
            "root_cause": "invalid_program",
            "false_positive": True,
        },
        {
            "triage_verdict": "expected_semantic_divergence",
            "root_cause": "unicode_case_mapping",
        },
    ]

    buckets = offline_finding_buckets(findings, known_saturated_bug_families=["known_root@duckdb"])

    assert buckets["new_bug"] == 1
    assert buckets["known_bug"] == 2
    assert buckets["false_positive"] == 1
    assert buckets["semantic_divergence"] == 1


def test_seeded_and_terminal_lifecycle_candidates_never_enter_fresh_reward_counts():
    seeded_row = {
        "candidate_source": "seeded_fault_sensitivity",
        "findings": [
            {
                "triage_verdict": "candidate_implementation_bug",
                "root_cause": "injected_filter",
                "suspicious_backends": ["buggy_filter"],
            }
        ],
        "preflight": {"valid": True, "fallback_used": False},
    }
    submitted = {
        "triage_verdict": "candidate_implementation_bug",
        "root_cause": "already_reported",
        "suspicious_backends": ["duckdb"],
        "family_lifecycle": "submitted",
    }

    seeded_signals = row_reward_signals(seeded_row)
    submitted_signals = row_reward_signals({"findings": [submitted]})

    assert seeded_signals["candidate_bug_count"] == 0
    assert seeded_signals["seeded_candidate_bug_count"] == 1
    assert online_case_reward(seeded_row) < 0.0
    assert submitted_signals["candidate_bug_count"] == 0
    assert submitted_signals["nonfresh_lifecycle_candidate_bug_count"] == 1
    known_signals = row_reward_signals(
        {
            "candidate_source": "known_regression_replay",
            "findings": [
                {
                    "triage_verdict": "candidate_implementation_bug",
                    "root_cause": "known_root",
                    "suspicious_backends": ["sqlite"],
                }
            ],
        }
    )
    assert known_signals["candidate_bug_count"] == 0
    assert known_signals["evidence_lane"] == "known_regression"


def test_summarize_case_feedback_exposes_closed_loop_adjustments():
    row = {
        "candidate_source": "feedback_mutation",
        "is_new_behavior": True,
        "stored_in_feedback_corpus": True,
        "preflight": {"valid": True, "fallback_used": False},
        "findings": [
            {
                "triage_verdict": "candidate_implementation_bug",
                "root_cause": "topk_filter_pushdown",
                "suspicious_backends": ["datafusion"],
                "signature": "sig-topk",
                "discovery_origin": "organic",
            }
        ],
        "quality_oracles": [
            {"name": "mutation", "verdict": "productive_mutation", "passed": True, "score": 1.5},
            {"name": "feedback", "verdict": "finding_yield", "passed": True, "score": 1.0},
            {"name": "guidance", "verdict": "guided_productive", "passed": True, "score": 3.0},
        ],
    }

    summary = feedback_summary_for_case(row)

    assert summary["candidate_bug_count"] == 1
    assert summary["quality_pass_count"] == 3
    assert summary["mutation_oracle_verdict"] == "productive_mutation"
    assert summary["source_reward_adjustment"] > 0.0
    assert summary["guidance_reward_adjustment"] > 0.0
    assert summary["seed_schedule_delta"] > 0.0


def test_feedback_summary_suppresses_auxiliary_rewards_for_resolved_semantic_only():
    row = {
        "candidate_source": "feedback_mutation",
        "is_new_behavior": True,
        "signal_new_behavior": True,
        "stored_in_feedback_corpus": True,
        "preflight": {"valid": True, "fallback_used": False},
        "findings": [
            {
                "triage_verdict": "expected_semantic_divergence",
                "root_cause": "nan_inf_semantics",
                "suspicious_backends": ["duckdb"],
                "signature": "resolved-semantic",
            }
        ],
        "quality_oracles": [
            {"name": "mutation", "verdict": "productive_mutation", "passed": True, "score": 1.0},
            {"name": "feedback", "verdict": "finding_yield", "passed": True, "score": 1.0},
            {"name": "guidance", "verdict": "guided_productive", "passed": True, "score": 1.0},
        ],
    }

    summary = feedback_summary_for_case(row)

    assert summary["candidate_bug_count"] == 0
    assert summary["resolved_semantic_divergence_count"] == 1
    assert summary["source_reward_adjustment"] == 0.0
    assert summary["guidance_reward_adjustment"] == 0.0
    assert summary["seed_schedule_delta"] == -0.19


def test_feedback_summary_suppresses_auxiliary_rewards_for_false_positive_only():
    row = {
        "candidate_source": "feedback_mutation",
        "is_new_behavior": True,
        "signal_new_behavior": True,
        "stored_in_feedback_corpus": True,
        "preflight": {"valid": True, "fallback_used": False},
        "findings": [
            {
                "triage_verdict": "normalizer_false_positive",
                "root_cause": "order_only_normalization_mismatch",
                "false_positive": True,
                "suspicious_backends": ["sqlite"],
                "signature": "false-positive",
            }
        ],
        "quality_oracles": [
            {"name": "mutation", "verdict": "productive_mutation", "passed": True, "score": 1.0},
            {"name": "feedback", "verdict": "finding_yield", "passed": True, "score": 1.0},
            {"name": "guidance", "verdict": "guided_productive", "passed": True, "score": 1.0},
        ],
    }

    summary = feedback_summary_for_case(row)

    assert summary["candidate_bug_count"] == 0
    assert summary["false_positive_count"] == 1
    assert summary["source_reward_adjustment"] == 0.0
    assert summary["guidance_reward_adjustment"] == 0.0
    assert summary["seed_schedule_delta"] == -1.29


def test_feedback_summary_suppresses_auxiliary_rewards_for_source_issue_only():
    row = {
        "candidate_source": "feedback_mutation",
        "is_new_behavior": True,
        "signal_new_behavior": True,
        "stored_in_feedback_corpus": True,
        "preflight": {"valid": True, "fallback_used": False},
        "findings": [
            {
                "triage_verdict": "candidate_implementation_bug",
                "root_cause": "csv_long_numeric_roundtrip",
                "suspicious_backends": ["duckdb"],
                "signature": "known-source-issue",
                "source_issue": "duckdb/duckdb#12345",
                "discovery_origin": "organic",
            }
        ],
        "quality_oracles": [
            {"name": "mutation", "verdict": "productive_mutation", "passed": True, "score": 1.0},
            {"name": "feedback", "verdict": "finding_yield", "passed": True, "score": 1.0},
            {"name": "guidance", "verdict": "guided_productive", "passed": True, "score": 1.0},
        ],
    }

    summary = feedback_summary_for_case(row)

    assert summary["candidate_bug_count"] == 0
    assert summary["source_issue_candidate_bug_count"] == 1
    assert summary["source_reward_adjustment"] == 0.0
    assert summary["guidance_reward_adjustment"] == 0.0
    assert summary["seed_schedule_delta"] == pytest.approx(0.01)


def test_feedback_summary_tracks_semantic_affinity_hits_and_selected_operator():
    row = {
        "candidate_source": "feedback_mutation",
        "feedback_decision": {
            "target_keys": [
                "semantic_family:conditional_semantics",
                "semantic_signal:left_join_case_when_membership",
                "exploration_objective:cross_model_consistency",
            ],
            "selected_operator": "append_left_join_case_membership",
            "selected_operator_score": 1.75,
        },
        "quality_oracles": [],
    }

    summary = feedback_summary_for_case(row)

    assert summary["feedback_target_key_count"] == 3
    assert summary["feedback_semantic_family_target_count"] == 1
    assert summary["feedback_semantic_signal_target_count"] == 1
    assert summary["feedback_exploration_objective_target_count"] == 1
    assert summary["feedback_selected_operator"] == "append_left_join_case_membership"
    assert summary["feedback_operator_family_affinity_hit"] is True
    assert summary["feedback_operator_signal_affinity_hit"] is True
    assert summary["feedback_operator_objective_affinity_hit"] is True
    assert summary["feedback_operator_affinity_hit"] is True


def test_aggregate_feedback_summaries_exposes_operator_and_semantic_target_telemetry():
    row = {
        "candidate_source": "feedback_mutation",
        "feedback_decision": {
            "target_keys": [
                "semantic_family:conditional_semantics",
                "semantic_signal:left_join_case_when_membership",
            ],
            "selected_operator": "append_left_join_case_membership",
            "selected_operator_score": 1.75,
        },
        "quality_oracles": [],
    }

    aggregate = aggregate_feedback_summaries([row])

    assert aggregate["feedback_target_key_count"] == 2
    assert aggregate["feedback_semantic_family_target_count"] == 1
    assert aggregate["feedback_semantic_signal_target_count"] == 1
    assert aggregate["feedback_operator_affinity_hit_cases"] == 1
    assert aggregate["feedback_selected_operator_count"] == 1
    assert aggregate["feedback_selected_operator_score_total"] == 1.75
    assert aggregate["top_feedback_selected_operators"] == "append_left_join_case_membership:1"
    assert aggregate["top_feedback_semantic_target_keys"] == (
        "semantic_family:conditional_semantics:1; semantic_signal:left_join_case_when_membership:1"
    )


def test_feedback_summary_helpers_preserve_and_aggregate_closed_loop_signals():
    row = {
        "candidate_source": "feedback_mutation",
        "is_new_behavior": True,
        "stored_in_feedback_corpus": True,
        "preflight": {"valid": True, "fallback_used": False},
        "findings": [
            {
                "triage_verdict": "candidate_implementation_bug",
                "root_cause": "topk_filter_pushdown",
                "suspicious_backends": ["datafusion"],
                "signature": "sig-topk",
                "discovery_origin": "organic",
            }
        ],
        "quality_oracles": [
            {"name": "mutation", "verdict": "productive_mutation", "passed": True, "score": 1.5},
            {"name": "feedback", "verdict": "finding_yield", "passed": True, "score": 1.0},
            {"name": "guidance", "verdict": "guided_productive", "passed": True, "score": 3.0},
        ],
        "feedback_summary": {"source_reward_adjustment": 0.42},
    }

    summary = coerce_case_feedback_summary(row)
    aggregate = aggregate_feedback_summaries([row])

    assert summary["candidate_source"] == "feedback_mutation"
    assert summary["source_reward_adjustment"] == 0.5
    assert aggregate["feedback_case_count"] == 1
    assert aggregate["feedback_mutation_cases"] == 1
    assert aggregate["stored_in_feedback_corpus_cases"] == 1
    assert aggregate["quality_pass_count"] == 3
    assert aggregate["productive_mutation_cases"] == 1
    assert aggregate["feedback_finding_yield_cases"] == 1
    assert aggregate["guided_productive_cases"] == 1
    assert aggregate["source_reward_adjustment_total"] == 0.5


def test_feedback_summary_helpers_recompute_polluted_existing_summary():
    row = {
        "candidate_source": "feedback_mutation",
        "is_new_behavior": True,
        "signal_new_behavior": True,
        "stored_in_feedback_corpus": True,
        "preflight": {"valid": True, "fallback_used": False},
        "findings": [
            {
                "triage_verdict": "normalizer_false_positive",
                "root_cause": "order_only_normalization_mismatch",
                "false_positive": True,
                "signature": "polluted-false-positive",
            }
        ],
        "quality_oracles": [
            {"name": "mutation", "verdict": "productive_mutation", "passed": True, "score": 1.0},
            {"name": "feedback", "verdict": "finding_yield", "passed": True, "score": 1.0},
            {"name": "guidance", "verdict": "guided_productive", "passed": True, "score": 1.0},
        ],
        "feedback_summary": {
            "candidate_bug_count": 99,
            "signal_new_behavior": True,
            "source_reward_adjustment": 0.5,
            "guidance_reward_adjustment": 0.25,
        },
    }

    summary = coerce_case_feedback_summary(row)
    aggregate = aggregate_feedback_summaries([row])

    assert summary["candidate_bug_count"] == 0
    assert summary["false_positive_count"] == 1
    assert summary["raw_signal_new_behavior"] is True
    assert summary["signal_new_behavior"] is False
    assert summary["source_reward_adjustment"] == 0.0
    assert summary["guidance_reward_adjustment"] == 0.0
    assert aggregate["source_reward_adjustment_total"] == 0.0
