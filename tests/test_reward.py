from datadiff.reward import candidate_bug_family_keys, candidate_bug_signatures, row_reward_signals


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
