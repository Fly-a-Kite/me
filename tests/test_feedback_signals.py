from datadiff.feedback_signals import (
    build_feedback_discovery_signals,
    feedback_source_new_behavior,
)


def test_feedback_signals_extract_candidate_bug_families_and_source_behavior():
    row = {
        "signal_new_behavior": True,
        "findings": [
            {
                "triage_verdict": "candidate_implementation_bug",
                "root_cause": "groupby_aggregation",
                "suspicious_backends": ["datafusion"],
                "signature": "sig-new",
                "discovery_origin": "organic",
            }
        ],
    }

    signals = build_feedback_discovery_signals(row)

    assert signals.feedback_finding is True
    assert signals.candidate_bug_families == ["groupby_aggregation@datafusion"]
    assert signals.candidate_bug_signatures == ["sig-new"]
    assert signals.rewardable_semantic_divergence is False
    assert signals.resolved_semantic_only is False
    assert signals.rewardable_new_behavior is True
    assert feedback_source_new_behavior(row, signals) is True


def test_feedback_signals_do_not_reward_manual_confirmation_as_finding():
    row = {
        "signal_new_behavior": True,
        "findings": [
            {
                "triage_verdict": "needs_manual_confirmation",
                "root_cause": "unknown",
                "suspicious_backends": ["duckdb", "pandas"],
                "signature": "sig-manual",
            }
        ],
    }

    signals = build_feedback_discovery_signals(row)

    assert signals.feedback_finding is False
    assert signals.candidate_bug_families == []
    assert signals.rewardable_semantic_divergence is False
    assert signals.reward_signals["needs_confirmation"] is True
    assert signals.rewardable_new_behavior is False
    assert feedback_source_new_behavior(row, signals) is False


def test_feedback_signals_keep_unresolved_semantic_divergence_explorable():
    row = {
        "signal_new_behavior": False,
        "findings": [
            {
                "triage_verdict": "semantic_divergence_needs_confirmation",
                "root_cause": "string_expression",
                "suspicious_backends": ["sqlite"],
                "signature": "sig-semantic",
            }
        ],
    }

    signals = build_feedback_discovery_signals(row)

    assert signals.feedback_finding is True
    assert signals.rewardable_semantic_divergence is True
    assert signals.resolved_semantic_only is False
    assert signals.candidate_bug_families == []
    assert signals.reward_signals["semantic_divergence_needs_confirmation_count"] == 1


def test_feedback_signals_suppress_source_new_behavior_for_resolved_semantics():
    row = {
        "signal_new_behavior": True,
        "findings": [
            {
                "triage_verdict": "expected_semantic_divergence",
                "root_cause": "unicode_case_mapping",
                "suspicious_backends": ["duckdb"],
                "signature": "sig-resolved",
            }
        ],
    }

    signals = build_feedback_discovery_signals(row)

    assert signals.feedback_finding is False
    assert signals.rewardable_semantic_divergence is False
    assert signals.resolved_semantic_only is True
    assert signals.rewardable_new_behavior is False
    assert feedback_source_new_behavior(row, signals) is False


def test_feedback_signals_suppress_source_new_behavior_for_false_positive():
    row = {
        "signal_new_behavior": True,
        "findings": [
            {
                "triage_verdict": "normalizer_false_positive",
                "root_cause": "order_only_normalization_mismatch",
                "false_positive": True,
                "signature": "sig-false-positive",
            }
        ],
    }

    signals = build_feedback_discovery_signals(row)

    assert signals.feedback_finding is False
    assert signals.reward_signals["false_positive_count"] == 1
    assert signals.rewardable_new_behavior is False
    assert feedback_source_new_behavior(row, signals) is False


def test_feedback_signals_suppress_source_new_behavior_for_source_issue():
    row = {
        "signal_new_behavior": True,
        "findings": [
            {
                "triage_verdict": "candidate_implementation_bug",
                "root_cause": "csv_long_numeric_roundtrip",
                "suspicious_backends": ["duckdb"],
                "source_issue": "duckdb/duckdb#12345",
                "signature": "sig-source-issue",
            }
        ],
    }

    signals = build_feedback_discovery_signals(row)

    assert signals.feedback_finding is False
    assert signals.reward_signals["source_issue_candidate_bug_count"] == 1
    assert signals.rewardable_new_behavior is False
    assert feedback_source_new_behavior(row, signals) is False


def test_feedback_signals_suppress_source_new_behavior_for_known_saturated_family():
    row = {
        "signal_new_behavior": True,
        "findings": [
            {
                "triage_verdict": "candidate_implementation_bug",
                "root_cause": "csv_long_numeric_roundtrip",
                "suspicious_backends": ["duckdb", "pyarrow"],
                "signature": "sig-known-family",
            }
        ],
    }

    signals = build_feedback_discovery_signals(
        row,
        known_saturated_bug_families=["csv_long_numeric_roundtrip@duckdb"],
    )

    assert signals.feedback_finding is False
    assert signals.reward_signals["known_saturated_candidate_bug_count"] == 1
    assert signals.rewardable_new_behavior is False
    assert feedback_source_new_behavior(row, signals) is False
