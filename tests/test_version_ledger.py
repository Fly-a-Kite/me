import pytest

from datadiff.util import append_jsonl, dump_json, run_meta_path
from datadiff.version_ledger import VersionObservation, build_version_ledger, observation_from_run_log


def test_version_ledger_classifies_cross_version_family_transitions():
    ledger = build_version_ledger(
        [
            VersionObservation(
                version_id="v1",
                run_file="runs/v1.jsonl",
                case_count=10,
                candidate_families={
                    "fixed_family@engine": 2,
                    "persistent_family@engine": 1,
                },
            ),
            VersionObservation(
                version_id="v2",
                run_file="runs/v2.jsonl",
                case_count=10,
                candidate_families={
                    "persistent_family@engine": 3,
                    "new_family@engine": 1,
                },
            ),
        ]
    )

    statuses = {row["family"]: row["status"] for row in ledger["families"]}

    assert ledger["schema_version"] == "version-ledger-v1"
    assert statuses == {
        "fixed_family@engine": "fixed",
        "new_family@engine": "new",
        "persistent_family@engine": "persistent",
    }
    assert ledger["summary"]["fixed_family_count"] == 1
    assert ledger["summary"]["new_family_count"] == 1
    assert ledger["summary"]["persistent_family_count"] == 1


def test_version_ledger_marks_previous_family_reappearing_as_regression():
    previous = build_version_ledger(
        [
            VersionObservation(
                version_id="old",
                run_file="runs/old.jsonl",
                case_count=1,
                candidate_families={"regressed_family@engine": 1},
            )
        ]
    )
    current = build_version_ledger(
        [
            VersionObservation(
                version_id="v1",
                run_file="runs/v1.jsonl",
                case_count=1,
                candidate_families={},
            ),
            VersionObservation(
                version_id="v2",
                run_file="runs/v2.jsonl",
                case_count=1,
                candidate_families={"regressed_family@engine": 1},
            ),
        ],
        previous_ledger=previous,
    )

    assert current["families"][0]["status"] == "regression"
    assert current["summary"]["regression_family_count"] == 1


def test_version_ledger_exports_continual_learning_priorities():
    previous = build_version_ledger(
        [
            VersionObservation(
                version_id="old",
                run_file="runs/old.jsonl",
                case_count=1,
                candidate_families={"regressed_family@engine": 1},
            )
        ]
    )
    ledger = build_version_ledger(
        [
            VersionObservation(
                version_id="v1",
                run_file="runs/v1.jsonl",
                case_count=1,
                candidate_families={
                    "fixed_family@engine": 1,
                    "persistent_family@engine": 1,
                },
            ),
            VersionObservation(
                version_id="v2",
                run_file="runs/v2.jsonl",
                case_count=1,
                candidate_families={
                    "persistent_family@engine": 1,
                    "new_family@engine": 1,
                    "regressed_family@engine": 1,
                },
            ),
        ],
        previous_ledger=previous,
    )

    continual = ledger["continual_learning"]
    priorities = {row["family"]: row["priority"] for row in continual["family_priorities"]}

    assert continual["schema_version"] == "cross-version-continual-learning-v1"
    assert priorities["regressed_family@engine"] > priorities["new_family@engine"]
    assert priorities["new_family@engine"] > priorities["persistent_family@engine"]
    assert priorities["persistent_family@engine"] > priorities["fixed_family@engine"]


def test_version_ledger_exports_champion_transfer_evidence():
    ledger = build_version_ledger(
        [
            VersionObservation(
                version_id="v1",
                run_file="runs/v1.jsonl",
                case_count=4,
                candidate_families={"persistent_family@engine": 1},
                metadata={
                    "champion_transfer_observation": {
                        "champion_seed_case_count": 1,
                        "champion_graft_case_count": 0,
                        "champion_candidate_family_hit_count": 1,
                        "champion_candidate_families": {"persistent_family@engine": 1},
                        "champion_corpus_injected_count": 1,
                        "champion_promoted_family_count": 1,
                        "champion_graft_donor_pulls": 0,
                    }
                },
            ),
            VersionObservation(
                version_id="v2",
                run_file="runs/v2.jsonl",
                case_count=4,
                candidate_families={"persistent_family@engine": 1, "new_family@engine": 1},
                metadata={
                    "champion_transfer_observation": {
                        "champion_seed_case_count": 0,
                        "champion_graft_case_count": 1,
                        "champion_candidate_family_hit_count": 1,
                        "champion_candidate_families": {"persistent_family@engine": 1},
                        "champion_corpus_injected_count": 1,
                        "champion_promoted_family_count": 0,
                        "champion_graft_donor_pulls": 1,
                    }
                },
            ),
        ]
    )

    transfer = ledger["champion_transfer"]
    measured = transfer["measured_seed_level_transfer"]

    assert transfer["schema_version"] == "champion-transfer-evidence-v1"
    assert transfer["transferable_family_count"] == 1
    assert transfer["transferable_families"][0]["family"] == "persistent_family@engine"
    assert transfer["measured"] is True
    assert measured["champion_case_count"] == 2
    assert measured["champion_candidate_family_hit_count"] == 2
    assert measured["champion_graft_donor_pulls"] == 1
    assert ledger["summary"]["champion_transfer_measured"] is True


def test_version_ledger_exports_health_feedback_report():
    ledger = build_version_ledger(
        [
            VersionObservation(
                version_id="v1",
                run_file="runs/v1.jsonl",
                case_count=2,
                candidate_families={"persistent_family@engine": 1},
                duration_ms_total=4.0,
                throughput_cases_s=5.0,
            ),
            VersionObservation(
                version_id="v2",
                run_file="runs/v2.jsonl",
                case_count=2,
                candidate_families={
                    "persistent_family@engine": 1,
                    "new_family@engine": 1,
                },
                invalid_case_count=1,
                fallback_case_count=1,
                false_positive_count=1,
                duration_ms_total=8.0,
                throughput_cases_s=4.0,
            ),
        ]
    )

    report = ledger["health_feedback_report"]

    assert ledger["health"]["schema_version"] == "version-ledger-health-v1"
    assert report["schema_version"] == "version-ledger-health-feedback-report-v1"
    assert report["has_health_feedback"] is True
    assert report["health_observation_count"] == 2
    assert report["invalid_case_count"] == 1
    assert report["fallback_case_count"] == 1
    assert report["false_positive_count"] == 1
    assert report["min_throughput_cases_s"] == pytest.approx(4.0)
    assert ledger["summary"]["health_observation_count"] == 2
    assert ledger["summary"]["max_invalid_rate"] == pytest.approx(0.5)


def test_observation_from_run_log_extracts_health_feedback_from_rows_and_meta(tmp_path):
    run_file = tmp_path / "run-v2.jsonl"
    append_jsonl(
        {
            "duration_ms": 8.0,
            "preflight": {"valid": False, "fallback_used": True},
            "findings": [
                {
                    "triage_verdict": "candidate_implementation_bug",
                    "root_cause": "new_root",
                    "suspicious_backends": ["engine"],
                },
                {
                    "triage_verdict": "generator_false_positive",
                    "root_cause": "generator_noise",
                    "suspicious_backends": ["engine"],
                },
            ],
        },
        run_file,
    )
    dump_json(
        {
            "target_version": "v2-from-meta",
            "preflight": {"invalid_cases": 2, "fallback_cases": 3},
            "throughput_cases_s": 4.0,
        },
        run_meta_path(run_file),
    )

    observation = observation_from_run_log(run_file)

    assert observation.version_id == "v2-from-meta"
    assert observation.candidate_families == {"new_root@engine": 1}
    assert observation.invalid_case_count == 2
    assert observation.fallback_case_count == 3
    assert observation.false_positive_count == 1
    assert observation.duration_ms_total == pytest.approx(8.0)
    assert observation.throughput_cases_s == pytest.approx(4.0)


def test_observation_from_run_log_extracts_champion_transfer_observation(tmp_path):
    run_file = tmp_path / "run-v2.jsonl"
    append_jsonl(
        {
            "candidate_source": "champion_corpus",
            "case": {"metadata": {"champion_seed": {"source_version_id": "v1"}}},
            "findings": [
                {
                    "triage_verdict": "candidate_implementation_bug",
                    "root_cause": "persistent_root",
                    "suspicious_backends": ["engine"],
                }
            ],
        },
        run_file,
    )
    append_jsonl(
        {
            "mutation": {"operator": "champion_graft"},
            "case": {"metadata": {"champion_graft": {"donor_case_id": "case-old"}}},
            "findings": [],
        },
        run_file,
    )
    dump_json(
        {
            "target_version": "v2",
            "champion_corpus": {"enabled": True, "injected_count": 1},
            "closed_loop_state_summary": {
                "adaptive_learning_health": {
                    "champion_graft_donor_pulls": 2,
                    "champion_graft_donor_arm_count": 1,
                },
                "champion_corpus_health": {
                    "enabled": True,
                    "family_hit_count": 1,
                    "promoted_family_count": 1,
                },
            },
        },
        run_meta_path(run_file),
    )

    observation = observation_from_run_log(run_file)
    transfer = observation.metadata["champion_transfer_observation"]

    assert transfer["champion_seed_case_count"] == 1
    assert transfer["champion_graft_case_count"] == 1
    assert transfer["champion_candidate_family_hit_count"] == 1
    assert transfer["champion_candidate_families"] == {"persistent_root@engine": 1}
    assert transfer["champion_corpus_injected_count"] == 1
    assert transfer["champion_graft_donor_pulls"] == 2
