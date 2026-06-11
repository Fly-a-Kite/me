import pytest

from datadiff.bug_discovery_system import (
    BUG_DISCOVERY_SYSTEM_SCHEMA_VERSION,
    boltzmann_budget_multipliers,
    bug_discovery_system_descriptor,
    candidate_true_bug_acquisition,
    good_turing_unseen_probability,
    lane_true_bug_acquisition,
    normalized_entropy,
    rank_candidate_pipeline_rows,
)
from datadiff.discovery_campaign_summary import (
    DEFAULT_DISCOVERY_CAMPAIGN_SCORE_WEIGHTS,
    _discovery_campaign_lane_rows,
)


def _finding(root: str, *, backend: str = "duckdb", **extra):
    finding = {
        "triage_verdict": "candidate_implementation_bug",
        "root_cause": root,
        "suspicious_backends": [backend],
        "false_positive": False,
    }
    finding.update(extra)
    return finding


def _case(case_id: str = "case-fresh"):
    return {
        "case_id": case_id,
        "seed": 7,
        "tables": [
            {
                "name": "t0",
                "columns": [{"name": "x", "dtype": "int", "nullable": False}],
                "rows": [{"x": 1}, {"x": 2}, {"x": 3}],
            }
        ],
        "program": {
            "program_id": "prog-fresh",
            "seed": 7,
            "operations": [{"op": "filter", "predicate": {"column": "x", "op": ">", "value": 1}}],
        },
    }


def test_good_turing_and_entropy_track_unseen_family_mass():
    assert good_turing_unseen_probability({}) == 1.0
    assert good_turing_unseen_probability({"a": 1, "b": 1, "c": 4}) == pytest.approx(2 / 6)
    assert good_turing_unseen_probability({"a": "bad", "b": -2}) == 1.0
    assert normalized_entropy({"a": 1, "b": 1, "c": 1}) == pytest.approx(1.0)
    assert normalized_entropy({"a": 3}) == 0.0


def test_lane_acquisition_rewards_true_bug_yield_and_explores_sparse_lanes():
    productive = lane_true_bug_acquisition(
        {
            "completed_runs": 4,
            "fresh_candidate_total": 6,
            "candidate_total": 6,
            "false_positive_total": 0,
            "known_saturated_total": 0,
            "first_seen_family_count": 3,
            "unique_fresh_family_count": 3,
            "fresh_family_counts": {"fresh_a@duckdb": 1, "fresh_b@polars": 2, "fresh_c@pyarrow": 3},
        },
        total_completed_runs=18,
    )
    noisy = lane_true_bug_acquisition(
        {
            "completed_runs": 4,
            "fresh_candidate_total": 1,
            "candidate_total": 3,
            "false_positive_total": 5,
            "known_saturated_total": 4,
            "first_seen_family_count": 0,
            "unique_fresh_family_count": 1,
            "fresh_family_counts": {"old_family@duckdb": 5},
        },
        total_completed_runs=18,
    )
    sparse = lane_true_bug_acquisition(
        {"completed_runs": 0, "fresh_family_counts": {}},
        total_completed_runs=18,
    )

    assert productive["acquisition_score"] > noisy["acquisition_score"]
    assert noisy["false_positive_rate"] > productive["false_positive_rate"]
    assert noisy["saturation_rate"] > productive["saturation_rate"]
    assert sparse["ucb_bonus"] > productive["ucb_bonus"]
    assert 0.0 <= productive["good_turing_unseen_probability"] <= 1.0


def test_lane_acquisition_rewards_pipeline_conversion_and_penalizes_duplicate_waste():
    convertible = lane_true_bug_acquisition(
        {
            "completed_runs": 4,
            "fresh_candidate_total": 4,
            "candidate_total": 4,
            "first_seen_family_count": 2,
            "unique_fresh_family_count": 2,
            "fresh_family_counts": {"fresh_a@duckdb": 2, "fresh_b@pyarrow": 2},
            "pipeline_candidate_count": 4,
            "pipeline_processed_candidate_count": 4,
            "pipeline_candidate_bug_verdict_count": 3,
            "pipeline_issue_draft_count": 2,
        },
        total_completed_runs=8,
    )
    duplicate_heavy = lane_true_bug_acquisition(
        {
            "completed_runs": 4,
            "fresh_candidate_total": 4,
            "candidate_total": 4,
            "first_seen_family_count": 1,
            "unique_fresh_family_count": 1,
            "fresh_family_counts": {"same_family@duckdb": 4},
            "pipeline_candidate_count": 4,
            "pipeline_processed_candidate_count": 1,
            "pipeline_skipped_duplicate_candidate_count": 3,
            "pipeline_candidate_bug_verdict_count": 0,
        },
        total_completed_runs=8,
    )

    assert convertible["pipeline_actionable_rate"] == pytest.approx(3 / 4)
    assert duplicate_heavy["pipeline_duplicate_waste_rate"] == pytest.approx(3 / 4)
    assert convertible["acquisition_score"] > duplicate_heavy["acquisition_score"]


def test_boltzmann_budget_multipliers_preserve_budget_with_bounds():
    multipliers = boltzmann_budget_multipliers([5.0, 2.0, 1.0], temperature=0.7)

    assert multipliers[0] > multipliers[1] >= multipliers[2]
    assert sum(multipliers) == pytest.approx(3.0)
    assert all(0.5 <= value <= 2.5 for value in multipliers)
    assert boltzmann_budget_multipliers([]) == []


def test_boltzmann_budget_multipliers_tolerate_dirty_scores():
    multipliers = boltzmann_budget_multipliers([float("nan"), -3.0, 2.0], temperature=float("nan"))

    assert sum(multipliers) == pytest.approx(3.0)
    assert all(0.5 <= value <= 2.5 for value in multipliers)
    assert multipliers[2] > multipliers[0]


def test_candidate_acquisition_penalizes_duplicates_replays_sources_and_false_positives():
    fresh = {
        "candidate_id": "fresh",
        "case": _case(),
        "findings": [_finding("fresh_family")],
        "normalized": {"duckdb": {"status": "ok"}, "pandas": {"status": "ok"}},
        "raw_results": {"duckdb": {"status": "ok"}, "pandas": {"status": "ok"}},
        "bug_dir": "bugs/bug_fresh",
        "candidate_recheck": {"attempts": 2, "reproduced": True},
        "families": ["fresh_family@duckdb"],
    }
    stale = {
        "candidate_id": "stale",
        "case": _case("case-stale"),
        "findings": [
            _finding(
                "stale_family",
                false_positive=True,
                discovery_origin="issue_replay",
                source_issue="https://example.invalid/known",
            )
        ],
        "candidate_recheck": {"attempts": 1, "non_reproduced_keys": ["stale_family@duckdb"]},
        "families": ["stale_family@duckdb"],
    }

    fresh_score = candidate_true_bug_acquisition(
        fresh,
        confirmed_latest_families=frozenset(),
        existing_by_family={},
    )
    stale_score = candidate_true_bug_acquisition(
        stale,
        confirmed_latest_families=frozenset({"stale_family@duckdb"}),
        existing_by_family={"stale_family@duckdb": ["new_issue/stale.md"]},
    )

    assert fresh_score["schema_version"] == BUG_DISCOVERY_SYSTEM_SCHEMA_VERSION
    assert fresh_score["acquisition_score"] > stale_score["acquisition_score"]
    assert fresh_score["true_bug_probability"] > stale_score["true_bug_probability"]
    assert stale_score["false_positive_finding_count"] == 1
    assert stale_score["issue_replay_finding_count"] == 1
    assert stale_score["source_issue_finding_count"] == 1
    assert stale_score["duplicate_family_count"] == 1
    assert stale_score["confirmed_latest_family_count"] == 1


def test_rank_candidate_pipeline_rows_prioritizes_high_proof_fresh_candidates():
    duplicate_low_proof = {
        "candidate_id": "duplicate-low-proof",
        "case": _case("case-duplicate"),
        "findings": [_finding("duplicate_family")],
        "candidate_recheck": {"attempts": 1, "non_reproduced_keys": ["duplicate_family@duckdb"]},
        "families": ["duplicate_family@duckdb"],
    }
    fresh_high_proof = {
        "candidate_id": "fresh-high-proof",
        "case": _case("case-fresh"),
        "findings": [_finding("fresh_family")],
        "normalized": {"duckdb": {"status": "ok"}, "pandas": {"status": "ok"}},
        "raw_results": {"duckdb": {"status": "ok"}, "pandas": {"status": "ok"}},
        "bug_dir": "bugs/bug_fresh",
        "candidate_recheck": {"attempts": 2, "reproduced": True},
        "families": ["fresh_family@duckdb"],
    }

    ranked = rank_candidate_pipeline_rows(
        [duplicate_low_proof, fresh_high_proof],
        existing_by_family={"duplicate_family@duckdb": ["new_issue/duplicate.md"]},
        confirmed_latest_families=frozenset(),
    )

    assert ranked[0]["candidate_id"] == "fresh-high-proof"
    assert ranked[0]["candidate_acquisition"]["rewardable_candidate_finding_count"] == 1
    assert ranked[1]["candidate_acquisition"]["duplicate_family_count"] == 1


def test_bug_discovery_system_descriptor_documents_closed_loop():
    descriptor = bug_discovery_system_descriptor()

    assert descriptor["schema_version"] == BUG_DISCOVERY_SYSTEM_SCHEMA_VERSION
    assert "recheck_reproducibility" in descriptor["stages"]
    assert descriptor["theory"]["budget"] == "Boltzmann allocation over acquisition free energy"


def test_discovery_campaign_lane_rows_use_true_bug_acquisition_and_boltzmann_budgeting():
    lanes = [
        {
            "id": "productive",
            "theme": "Productive",
            "target_suite": "latest",
            "preset": "live_productive",
        },
        {
            "id": "noisy",
            "theme": "Noisy",
            "target_suite": "latest",
            "preset": "live_noisy",
        },
        {
            "id": "sparse",
            "theme": "Sparse",
            "target_suite": "latest",
            "preset": "live_sparse",
        },
    ]
    rows = _discovery_campaign_lane_rows(
        lanes,
        history_manifests=[
            {
                "generated_at": "2026-06-01T00:00:00Z",
                "runs": [
                    {
                        "lane_id": "productive",
                        "status": "completed",
                        "completed_at": "2026-06-01T00:10:00Z",
                        "classification": {
                            "fresh_candidate_bug_families": {"fresh_a@duckdb": 1, "fresh_b@polars": 1},
                            "candidate_bug_families": {"fresh_a@duckdb": 1, "fresh_b@polars": 1},
                            "false_positive_reasons": {},
                        },
                    },
                    {
                        "lane_id": "noisy",
                        "status": "completed",
                        "completed_at": "2026-06-01T00:20:00Z",
                        "classification": {
                            "fresh_candidate_bug_families": {"stale@duckdb": 1},
                            "candidate_bug_families": {"stale@duckdb": 1},
                            "known_saturated_candidate_bug_families": {"old@duckdb": 3},
                            "false_positive_reasons": {"normalizer_false_positive": 5},
                        },
                    },
                ],
            }
        ],
        current_runs=[
            {
                "lane_id": "productive",
                "status": "completed",
                "completed_at": "2026-06-01T00:30:00Z",
                "classification": {
                    "fresh_candidate_bug_families": {"fresh_c@pyarrow": 1, "fresh_d@duckdb": "bad"},
                    "candidate_bug_families": {"fresh_c@pyarrow": 1},
                    "false_positive_reasons": {"bad": -1},
                },
            },
        ],
        pending_by_lane={"productive": [1], "noisy": [1], "sparse": [1]},
        score_weights=DEFAULT_DISCOVERY_CAMPAIGN_SCORE_WEIGHTS,
    )
    by_lane = {row["lane_id"]: row for row in rows}

    assert rows[0]["lane_id"] == "productive"
    assert by_lane["productive"]["score"] > by_lane["noisy"]["score"]
    assert by_lane["productive"]["budget_multiplier"] > by_lane["noisy"]["budget_multiplier"]
    assert sum(row["budget_multiplier"] for row in rows) == pytest.approx(3.0)
    assert by_lane["sparse"]["ucb_bonus"] > by_lane["productive"]["ucb_bonus"]
    assert by_lane["productive"]["fresh_candidate_total"] == 3
    assert by_lane["productive"]["good_turing_unseen_probability"] > 0.0
    assert by_lane["noisy"]["false_positive_rate"] > by_lane["productive"]["false_positive_rate"]
    assert by_lane["productive"]["acquisition_strategy"] == "true_bug_acquisition_good_turing_ucb_boltzmann"


def test_discovery_campaign_lane_rows_feed_pipeline_signal_into_acquisition():
    lanes = [
        {
            "id": "convertible",
            "theme": "Convertible",
            "target_suite": "latest",
            "preset": "live_convertible",
        },
        {
            "id": "duplicate_heavy",
            "theme": "Duplicate heavy",
            "target_suite": "latest",
            "preset": "live_duplicate_heavy",
        },
    ]
    base_classification = {
        "fresh_candidate_bug_families": {"fresh_family@duckdb": 1},
        "candidate_bug_families": {"fresh_family@duckdb": 1},
        "false_positive_reasons": {},
    }

    rows = _discovery_campaign_lane_rows(
        lanes,
        history_manifests=[],
        current_runs=[
            {
                "lane_id": "convertible",
                "status": "completed",
                "completed_at": "2026-06-01T00:10:00Z",
                "classification": base_classification,
                "candidate_pipeline": {
                    "summary": {
                        "candidate_count": 4,
                        "processed_candidate_count": 4,
                        "candidate_bug_verdict_count": 3,
                        "issue_draft_count": 2,
                    }
                },
            },
            {
                "lane_id": "duplicate_heavy",
                "status": "completed",
                "completed_at": "2026-06-01T00:20:00Z",
                "classification": base_classification,
                "candidate_pipeline": {
                    "summary": {
                        "candidate_count": 4,
                        "processed_candidate_count": 1,
                        "skipped_duplicate_candidate_count": 3,
                    }
                },
            },
        ],
        pending_by_lane={"convertible": [1], "duplicate_heavy": [1]},
        score_weights=DEFAULT_DISCOVERY_CAMPAIGN_SCORE_WEIGHTS,
    )
    by_lane = {row["lane_id"]: row for row in rows}

    assert by_lane["convertible"]["pipeline_actionable_rate"] == pytest.approx(3 / 4)
    assert by_lane["duplicate_heavy"]["pipeline_duplicate_waste_rate"] == pytest.approx(3 / 4)
    assert by_lane["convertible"]["score"] > by_lane["duplicate_heavy"]["score"]
    assert rows[0]["lane_id"] == "convertible"
