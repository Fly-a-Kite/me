import json
from pathlib import Path

from datadiff.discovery_runtime import aggregate_candidate_pipeline_summary, write_discovery_run_fresh_candidate_evidence
from datadiff.finding_outcomes import candidate_issue_family_key
from datadiff.osc_diagnostic_facade import not_evaluated_diagnostic_ref_set
from datadiff.run_summaries import _classified_candidate_rows_for_run
from datadiff.util import append_jsonl


def test_fresh_candidate_evidence_matches_row_level_normalized_family(tmp_path: Path):
    variant_refs = not_evaluated_diagnostic_ref_set(
        ["duckdb"],
        case_digest="case-discovery-variant-diagnostic",
        reason_code="legacy_variant_error_diagnostic",
    )
    variant_refs["authority_eligible"] = True
    row = {
        "case": {"case_id": "case-row-normalized", "seed": 1},
        "findings": [
            {
                "root_cause": "running_sum_precision",
                "triage_verdict": "candidate_implementation_bug",
                "false_positive": False,
                "suspicious_backends": ["sqlite", "duckdb"],
            },
            {
                "root_cause": "metamorphic_running_sum_precision",
                "triage_verdict": "candidate_implementation_bug",
                "false_positive": False,
                "suspicious_backends": ["duckdb", "sqlite"],
            },
        ],
        "normalized": {"duckdb": {"rows": [[1]]}},
        "raw_results": {},
        "config": {},
        "status": "bug",
        "case_index": 7,
        "elapsed_s": 0.25,
        "experiment_manifest": {"case_digest": "case-discovery-diagnostic"},
        "osc_diagnostic_refs": not_evaluated_diagnostic_ref_set(
            ["duckdb"],
            case_digest="case-discovery-diagnostic",
            reason_code="legacy_error_diagnostic",
        ),
        "osc_metamorphic_diagnostic_refs": {
            "target:b": {
                "experiment_manifest": {
                    "case_digest": "case-discovery-variant-diagnostic"
                },
                "backend_status": {"duckdb": "error"},
                "osc_diagnostic_refs": variant_refs,
            }
        },
    }
    row["osc_diagnostic_refs"]["authority_eligible"] = True
    evidence_path = tmp_path / "fresh.json"

    evidence = write_discovery_run_fresh_candidate_evidence(
        tmp_path / "run.jsonl",
        classification={
            "fresh_candidate_bug_families": {
                "running_sum_precision@duckdb,sqlite": 2,
            }
        },
        output_path=evidence_path,
        read_jsonl_func=lambda path: [row],
        candidate_issue_family_key_func=candidate_issue_family_key,
    )

    assert evidence["candidate_row_count"] == 1
    assert evidence["candidate_rows"][0]["case"]["case_id"] == "case-row-normalized"
    assert len(evidence["candidate_rows"][0]["findings"]) == 2
    refs = evidence["candidate_rows"][0]["osc_diagnostic_refs"]
    assert refs["evaluation_status"] == "not_evaluated"
    assert refs["authority_scope"] == "diagnostic_only"
    assert refs["authority_eligible"] is False
    assert refs["refs"][0]["backend"] == "duckdb"
    assert refs["refs"][0]["ref"]["reason_code"] == "invalid_diagnostic_refs"
    variant = evidence["candidate_rows"][0][
        "osc_metamorphic_diagnostic_refs"
    ]["target:b"]
    assert variant["experiment_manifest"]["case_digest"] == (
        "case-discovery-variant-diagnostic"
    )
    assert variant["osc_diagnostic_refs"]["authority_eligible"] is False
    assert variant["osc_diagnostic_refs"]["refs"][0]["ref"][
        "reason_code"
    ] == "invalid_diagnostic_refs"
    assert evidence["candidate_row_count"] == 1
    assert "confirmed_bug" not in evidence["candidate_rows"][0]
    persisted = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert persisted["candidate_row_count"] == 1


def test_fresh_candidate_evidence_keeps_known_saturated_rows_out(tmp_path: Path):
    row = {
        "case": {"case_id": "case-known", "seed": 1},
        "findings": [
            {
                "root_cause": "known_family",
                "triage_verdict": "candidate_implementation_bug",
                "false_positive": False,
                "suspicious_backends": ["duckdb"],
            }
        ],
        "config": {"known_saturated_bug_families": ["known_family@duckdb"]},
        "status": "bug",
    }

    evidence = write_discovery_run_fresh_candidate_evidence(
        tmp_path / "run.jsonl",
        classification={"fresh_candidate_bug_families": {"known_family@duckdb": 1}},
        output_path=tmp_path / "fresh.json",
        read_jsonl_func=lambda path: [row],
        candidate_issue_family_key_func=candidate_issue_family_key,
    )

    assert evidence["candidate_row_count"] == 0
    assert evidence["candidate_rows"] == []


def test_fresh_candidate_evidence_uses_refreshed_classified_rows(tmp_path: Path):
    raw_row = {
        "case": {"case_id": "case-stale", "seed": 1},
        "findings": [
            {
                "root_cause": "documented_boundary",
                "triage_verdict": "expected_semantic_divergence",
                "false_positive": False,
                "suspicious_backends": ["duckdb"],
            }
        ],
        "config": {},
    }
    refreshed_row = {
        "case": {"case_id": "case-stale", "seed": 1},
        "findings": [
            {
                "root_cause": "running_sum_precision",
                "triage_verdict": "candidate_implementation_bug",
                "false_positive": False,
                "suspicious_backends": ["pandas"],
            }
        ],
        "normalized": {"pandas": {"status": "ok", "rows": [[2]]}},
        "raw_results": {},
        "config": {},
        "status": "bug",
    }

    evidence = write_discovery_run_fresh_candidate_evidence(
        tmp_path / "run.jsonl",
        classification={"fresh_candidate_bug_families": {"running_sum_precision@pandas": 1}},
        output_path=tmp_path / "fresh.json",
        refresh=True,
        read_jsonl_func=lambda path: [raw_row],
        classified_candidate_rows_func=lambda path, refresh=False: [refreshed_row] if refresh else [],
    )

    assert evidence["refresh_classification"] is True
    assert evidence["candidate_row_count"] == 1
    assert evidence["candidate_rows"][0]["case"]["case_id"] == "case-stale"
    assert evidence["candidate_rows"][0]["findings"][0]["root_cause"] == "running_sum_precision"
    assert evidence["candidate_rows"][0]["candidate_bug_families"] == {
        "running_sum_precision@pandas": 1
    }


def test_classified_candidate_rows_for_run_honors_refreshed_findings(tmp_path: Path, monkeypatch):
    class FakeFinding:
        def to_dict(self):
            return {
                "kind": "semantic_output_mismatch",
                "root_cause": "running_sum_precision",
                "triage_verdict": "candidate_implementation_bug",
                "false_positive": False,
                "suspicious_backends": ["pandas"],
                "signature": "sig-refresh",
            }

    run_file = tmp_path / "run.jsonl"
    append_jsonl(
        {
            "case": {"case_id": "case-refresh", "seed": 1},
            "findings": [
                {
                    "kind": "semantic_output_mismatch",
                    "root_cause": "documented_boundary",
                    "triage_verdict": "expected_semantic_divergence",
                    "false_positive": False,
                    "suspicious_backends": ["duckdb"],
                    "signature": "sig-stale",
                }
            ],
            "normalized": {"duckdb": {"status": "ok", "rows": [[1]]}},
            "raw_results": {},
            "config": {},
            "status": "bug",
            "experiment_manifest": {"case_digest": "case-summary-diagnostic"},
            "osc_diagnostic_refs": not_evaluated_diagnostic_ref_set(
                ["duckdb"],
                case_digest="case-summary-diagnostic",
                reason_code="legacy_error_diagnostic",
            ),
            "osc_metamorphic_diagnostic_refs": {
                "target:b": {
                    "experiment_manifest": {
                        "case_digest": "case-summary-variant-diagnostic"
                    },
                    "backend_status": {"duckdb": "error"},
                    "osc_diagnostic_refs": not_evaluated_diagnostic_ref_set(
                        ["duckdb"],
                        case_digest="case-summary-variant-diagnostic",
                        reason_code="legacy_variant_error_diagnostic",
                    ),
                }
            },
        },
        run_file,
    )

    def fake_refresh(row):
        return object(), {"pandas": object()}, [FakeFinding().to_dict()], {}, ["pandas"]

    monkeypatch.setattr("datadiff.run_summaries._refresh_differential_findings", fake_refresh)
    monkeypatch.setattr("datadiff.run_summaries.evaluate_case", lambda case, normalized: [FakeFinding()])

    rows = _classified_candidate_rows_for_run(run_file, refresh=True)

    assert len(rows) == 1
    assert rows[0]["case"]["case_id"] == "case-refresh"
    assert rows[0]["findings"][0]["triage_verdict"] == "candidate_implementation_bug"
    assert rows[0]["findings"][0]["root_cause"] == "running_sum_precision"
    assert rows[0]["findings"][0]["suspicious_backends"] == ["pandas"]
    refs = rows[0]["osc_diagnostic_refs"]
    assert refs["case_digest"] == "case-summary-diagnostic"
    assert refs["evaluation_status"] == "not_evaluated"
    assert refs["authority_eligible"] is False
    assert refs["refs"][0]["backend"] == "duckdb"
    assert refs["refs"][0]["ref"]["reason_code"] == "legacy_error_diagnostic"
    variant = rows[0]["osc_metamorphic_diagnostic_refs"]["target:b"]
    assert variant["experiment_manifest"]["case_digest"] == (
        "case-summary-variant-diagnostic"
    )
    assert variant["osc_diagnostic_refs"]["authority_eligible"] is False
    assert variant["osc_diagnostic_refs"]["refs"][0]["ref"][
        "reason_code"
    ] == "legacy_variant_error_diagnostic"
    assert "confirmed_bug" not in rows[0]


def test_aggregate_candidate_pipeline_summary_keeps_duplicate_skip_metrics():
    summary = aggregate_candidate_pipeline_summary(
        [
            {
                "summary": {
                    "candidate_count": 5,
                    "processed_candidate_count": 2,
                    "skipped_duplicate_candidate_count": 3,
                    "rechecked_count": 2,
                    "reproduced_count": 1,
                    "reduced_count": 1,
                    "candidate_bug_verdict_count": 1,
                    "issue_draft_count": 1,
                }
            }
        ]
    )

    assert summary["pipeline_count"] == 1
    assert summary["candidate_count"] == 5
    assert summary["processed_candidate_count"] == 2
    assert summary["skipped_duplicate_candidate_count"] == 3
    assert summary["rechecked_count"] == 2
