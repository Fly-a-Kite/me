from datadiff.run_logging import (
    STAGE_PROFILE_KEYS,
    _adaptive_learning_health_summary,
    _case_log_row,
    _case_summary,
    _closed_loop_state_summary,
    _compact_log_row,
    _finalize_stage_profile_summary,
    _merge_stage_profile,
    _quality_archive_health_summary,
    _stage_profile_with_total,
)
from datadiff.dsl import Case, ColumnSpec, Program, TableData


def _sample_row() -> dict:
    return {
        "run_at": "2026-06-04T00:00:00Z",
        "status": "ok",
        "case": {
            "case_id": "case-1",
            "seed": 7,
            "tables": [
                {"rows": [{"x": 1}, {"x": 2}]},
                {"rows": [{"y": 3}]},
            ],
            "program": {
                "program_id": "program-1",
                "seed": 7,
                "operations": [{"op": "filter"}, {"op": "mutate"}],
            },
        },
        "raw_results": {
            "pandas": {"backend": "pandas", "status": "ok", "duration_ms": 1.25, "extra": "drop"},
            "duckdb": {"backend": "duckdb", "status": "ok", "duration_ms": 2.5, "extra": "drop"},
        },
        "normalized": {
            "pandas": {"backend": "pandas", "status": "ok", "columns": ["x"], "rows": [{"x": 1}]},
            "duckdb": {"backend": "duckdb", "status": "ok", "columns": ["x"], "rows": [{"x": 1}]},
        },
        "behavior_signature": "behavior",
        "discovery_signature": "discovery",
        "signal_signature": "signal",
        "duration_ms": 8.0,
        "stage_profile": {
            "generate_mutate_ms": 1.0,
            "backend_execution_ms": 2.0,
            "normalize_ms": 3.0,
        },
        "guidance": {
            "strategy": "guided",
            "score": 3.5,
            "matched_targets": ["semantic_family:window"],
            "features": ["op:filter", "op:mutate"],
            "score_breakdown": {
                "resolved_semantic_boundary_penalty": -0.5,
                "family_saturation_penalty": -0.25,
                "family_saturation_active": 1.0,
            },
        },
        "quality_oracles": [{"name": "oracle-a", "verdict": "pass", "passed": True, "score": 1.0}],
        "environment": {"python": "test"},
        "targets": [{"backend": "pandas"}],
    }


def test_case_log_row_keeps_selected_candidate_metadata_shape():
    case = Case(
        "case-1",
        7,
        [TableData("t0", [ColumnSpec("x", "int")], [{"x": 1}])],
        Program("program-1", 7, [{"op": "select", "columns": ["x"]}]),
        metadata={"quality_archive_context": {"archive_known": True}},
    )
    selected_meta = {
        "source": "feedback_mutation",
        "seed_lineage": {"root_seed": 7, "depth": 1},
        "mutation": {"operator": "value"},
        "feedback_decision": {"selected_operator": "value"},
        "quality_archive_context": {"archive_known": True},
        "generator_profile_selection": {"profile": "common"},
        "semantic_objective_selection": {"action": "objective:null_boundary"},
        "metamorphic_relation_selection": {"action": "row_order_invariance"},
        "version_pair_selection": {"action": "latest->fixed"},
        "selected_version_pair": "latest->fixed",
        "case_learning_context": ["op:select"],
        "operation_combo": {"template": "select_only", "operation_count": 1},
        "replay_filter": {"enabled": False},
        "family_saturation_filter": {"enabled": True},
    }

    row = _case_log_row(
        run_id="run-1",
        case_index=3,
        case_seed=7,
        candidate_seed_start=5,
        candidate_pool=4,
        guidance_row={"strategy": "guided"},
        selected_meta=selected_meta,
        backend_pair_pool=("duckdb|pandas",),
        preflight_row={"valid": True},
        generated_at="2026-06-04T00:00:00Z",
        case=case,
    )

    assert row["run_id"] == "run-1"
    assert row["case_index"] == 3
    assert row["seed"] == 7
    assert row["candidate_pool_size"] == 4
    assert row["candidate_source"] == "feedback_mutation"
    assert row["feedback_decision"] == {"selected_operator": "value"}
    assert row["backend_pair_pool"] == ["duckdb|pandas"]
    assert row["operation_combo"]["template"] == "select_only"
    assert row["preflight"] == {"valid": True}
    assert row["case"]["metadata"]["quality_archive_context"]["archive_known"] is True


def test_case_summary_keeps_log_shape_small():
    summary = _case_summary(_sample_row()["case"])

    assert summary["case_id"] == "case-1"
    assert summary["seed"] == 7
    assert summary["table_count"] == 2
    assert summary["row_count"] == 3
    assert summary["program"]["operations"] == [{"op": "filter"}, {"op": "mutate"}]


def test_stage_profile_helpers_fill_missing_keys_and_recompute_total():
    profile = _stage_profile_with_total({"generate_mutate_ms": 1.5, "backend_execution_ms": 2.5, "total_case_wall_ms": 99})

    assert set(profile) == set(STAGE_PROFILE_KEYS)
    assert profile["total_case_wall_ms"] == 4.0

    merged = _merge_stage_profile(profile, {"normalize_ms": 3.0, "logging_artifact_ms": 1.0})
    assert merged["generate_mutate_ms"] == 1.5
    assert merged["normalize_ms"] == 3.0
    assert merged["logging_artifact_ms"] == 1.0

    summary = _finalize_stage_profile_summary(merged, cases=2)
    assert summary["case_count"] == 2
    assert summary["avg_ms_per_case"]["normalize_ms"] == 1.5
    assert summary["totals_ms"]["total_case_wall_ms"] == 8.0
    assert summary["share_of_total"]["total_case_wall_ms"] == 1.0


def test_compact_log_row_summarizes_backend_payloads():
    compact = _compact_log_row(_sample_row(), "compact")

    assert compact["case"]["row_count"] == 3
    assert compact["stage_profile"]["total_case_wall_ms"] == 6.0
    assert compact["guidance"]["matched_semantic_targets"] == ["semantic_family:window"]
    assert compact["guidance"]["family_diversity_guard_penalty"] == -0.25
    assert compact["normalized"]["pandas"]["row_count"] == 1
    assert compact["raw_results"]["duckdb"]["duration_ms"] == 2.5
    assert "rows" not in compact["normalized"]["pandas"]
    assert "extra" not in compact["raw_results"]["pandas"]
    assert "environment" not in compact
    assert "targets" not in compact


def test_minimal_log_row_keeps_backend_status_only():
    minimal = _compact_log_row(_sample_row(), "minimal")

    assert minimal["backend_status"] == {"pandas": "ok", "duckdb": "ok"}
    assert "normalized" not in minimal
    assert "raw_results" not in minimal


def test_full_log_row_returns_original_object():
    row = _sample_row()

    assert _compact_log_row(row, "full") is row


def test_finding_rows_keep_reproduction_detail_but_drop_run_metadata():
    row = _sample_row()
    row["findings"] = [{"kind": "differential", "root_cause": "row_mismatch"}]

    compact = _compact_log_row(row, "compact")

    assert compact["case"] is row["case"]
    assert compact["normalized"] is row["normalized"]
    assert compact["findings"] == row["findings"]
    assert "environment" not in compact
    assert "targets" not in compact


def test_closed_loop_summary_includes_learning_archive_and_corpus_health():
    state = {
        "seen_signatures": ["a", "b"],
        "signal_seen_signatures": ["s"],
        "feedback": {
            "interesting_cases": [{"case_id": "case-1"}],
            "stored_candidate_bug_families": {"family": 1},
            "stored_target_keys": {"target": 2},
            "stored_cluster_keys": {"cluster": 1},
            "enable_seed_quota": True,
            "champion_family_hits": {"family": 3},
            "champion_promoted_families": ["family"],
            "champion_version_id": "v1",
            "adaptive_learning": {
                "schema_version": "adaptive-learning-v1",
                "bandits": {
                    "scope": {
                        "arms": [
                            {
                                "pulls": 4,
                                "runtime_cost_total": 2.0,
                                "false_positive_count": 1,
                                "invalid_count": 1,
                            }
                        ],
                        "reward_model": {"total_updates": 3, "feature_counts": {"f1": 4}},
                    },
                    "bd_axis_weights": {
                        "arms": [
                            {
                                "action_id": "bd_target_class",
                                "pulls": 2,
                                "runtime_cost_total": 0.0,
                            }
                        ],
                        "total_pulls": 2,
                        "reward_model": {"total_updates": 1, "feature_counts": {"bd": 2}},
                    },
                },
                "version_memory": {"reward_counts": {"v1": 2}},
            },
            "quality_archive": {
                "schema_version": "quality-diversity-archive-v1",
                "max_elites_per_cluster": 2,
                "cells": [
                    {
                        "seeds": [{"seed": 1}, {"seed": 2}, {"seed": 3}],
                        "reward_count": 5,
                        "outcome_count": 4,
                        "invalid_count": 1,
                    }
                ],
            },
            "seed_quota": {"enabled": True, "min_quota_per_active_cell": 2},
        },
        "guidance": {
            "feature_counts": {"f": 1},
            "frontier_bucket_counts": {"b": 2},
            "candidate_bug_family_counts": {"family": 1},
        },
    }

    summary = _closed_loop_state_summary(state)

    assert summary["seen_signature_count"] == 2
    assert summary["feedback_interesting_case_count"] == 1
    assert summary["adaptive_learning_health"]["total_pulls"] == 6
    assert summary["adaptive_learning_health"]["bd_axis_weight_pulls"] == 2
    assert summary["adaptive_learning_health"]["bd_axis_weight_arm_count"] == 1
    assert summary["quality_archive_health"]["elite_seed_count"] == 2
    assert summary["seed_quota_health"]["min_quota_per_active_cell"] == 2
    assert summary["champion_corpus_health"]["promoted_family_count"] == 1
    assert summary["guidance_feature_count"] == 1


def test_health_summaries_ignore_unrecognized_schema_versions():
    assert _adaptive_learning_health_summary({"schema_version": "old"}) == {}
    assert _quality_archive_health_summary({"schema_version": "old"}) == {}
