import json

from datadiff.run_logging import (
    MINIMAL_LOG_SCHEMA_VERSION,
    PROCESS_CPU_PROFILE_KEYS,
    STAGE_PROFILE_KEYS,
    WALL_TIME_PROFILE_KEYS,
    _adaptive_learning_health_summary,
    _case_log_row,
    _case_summary,
    _closed_loop_state_summary,
    _compact_log_row,
    _finalize_process_cpu_profile_summary,
    _finalize_stage_profile_summary,
    _merge_process_cpu_profile,
    _merge_stage_profile,
    _process_cpu_profile_with_total,
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
        "execution_reuse": {"schema_version": "execution-reuse-provenance-v1"},
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
        "goal_first_generation": {"constructible": True},
        "semantic_activation": {
            "goal_id": "nullable_membership_join",
            "evaluation_status": "activated",
        },
        "boundary_application": {"applied": True},
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
    assert row["candidate_pool_sampling"] == {}
    assert row["candidate_source"] == "feedback_mutation"
    assert row["feedback_decision"] == {"selected_operator": "value"}
    assert row["goal_first_generation"] == {"constructible": True}
    assert row["semantic_activation"]["evaluation_status"] == "activated"
    assert row["boundary_application"] == {"applied": True}
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


def test_process_cpu_profile_aggregation_is_additive_and_conservative():
    first = _process_cpu_profile_with_total(
        {"generate_mutate_ms": 1.0, "execution_pipeline_ms": 4.0}
    )
    second = _process_cpu_profile_with_total(
        {"oracle_classification_ms": 2.0, "logging_artifact_ms": 1.0}
    )

    merged = _merge_process_cpu_profile(first, second)
    summary = _finalize_process_cpu_profile_summary(merged, cases=2)

    assert set(merged) == set(PROCESS_CPU_PROFILE_KEYS)
    assert merged["total_process_cpu_ms"] == 8.0
    assert merged["total_process_cpu_ms"] == sum(
        merged[key]
        for key in PROCESS_CPU_PROFILE_KEYS
        if key != "total_process_cpu_ms"
    )
    assert summary["totals_ms"] == merged
    assert summary["avg_ms_per_case"]["total_process_cpu_ms"] == 4.0
    assert summary["share_of_total"]["total_process_cpu_ms"] == 1.0


def test_compact_log_row_summarizes_backend_payloads():
    row = _sample_row()
    row["candidate_recheck"] = {
        "enabled": True,
        "attempts": 1,
        "reproduced_keys": ["semantic_output_mismatch:root@duckdb"],
        "non_reproduced_keys": [],
    }
    row["semantic_activation"] = {
        "goal_id": "nullable_membership_join",
        "evaluation_status": "activated",
    }
    compact = _compact_log_row(row, "compact")

    assert compact["case"]["row_count"] == 3
    assert compact["stage_profile"]["total_case_wall_ms"] == 6.0
    assert compact["execution_reuse"]["schema_version"] == "execution-reuse-provenance-v1"
    assert compact["guidance"]["matched_semantic_targets"] == ["semantic_family:window"]
    assert compact["guidance"]["family_diversity_guard_penalty"] == -0.25
    assert compact["normalized"]["pandas"]["row_count"] == 1
    assert compact["raw_results"]["duckdb"]["duration_ms"] == 2.5
    assert "rows" not in compact["normalized"]["pandas"]
    assert "extra" not in compact["raw_results"]["pandas"]
    assert "environment" not in compact
    assert "targets" not in compact
    assert "wall_time_profile" not in compact
    assert "process_cpu_profile" not in compact
    assert compact["semantic_activation"]["evaluation_status"] == "activated"
    assert compact["candidate_recheck"]["reproduced_keys"] == [
        "semantic_output_mismatch:root@duckdb"
    ]


def test_compact_log_row_preserves_new_profiles_without_breaking_legacy_rows():
    row = _sample_row()
    row["wall_time_profile"] = {
        "execution_pipeline_ms": 4.0,
        "oracle_classification_ms": 1.0,
        "total_wall_ms": 999.0,
    }
    row["process_cpu_profile"] = {
        "execution_pipeline_ms": 3.0,
        "oracle_classification_ms": 0.5,
        "total_process_cpu_ms": 999.0,
    }

    compact = _compact_log_row(row, "compact")

    assert set(compact["wall_time_profile"]) == set(WALL_TIME_PROFILE_KEYS)
    assert compact["wall_time_profile"]["total_wall_ms"] == 5.0
    assert compact["process_cpu_profile"]["total_process_cpu_ms"] == 3.5


def test_minimal_log_row_keeps_backend_status_only():
    minimal = _compact_log_row(_sample_row(), "minimal")

    assert minimal["schema_version"] == MINIMAL_LOG_SCHEMA_VERSION
    assert minimal["backend_status"] == {"pandas": "ok", "duckdb": "ok"}
    assert "normalized" not in minimal
    assert "raw_results" not in minimal


def test_minimal_log_row_drops_high_volume_scheduler_duplicates():
    row = _sample_row()
    row.update(
        {
            "backend_sampling": {"rows": [{"payload": "x" * 4096}]},
            "candidate_pool_sampling": {"rows": [{"payload": "y" * 4096}]},
            "execution_lattice_selection": {"nodes": [{"payload": "z" * 4096}]},
            "case_learning_context": ["context" * 1024],
            "semantic_novelty": {"trace": ["novelty" * 1024]},
            "execution_reuse": {
                "schema_version": "execution-reuse-provenance-v1",
                "events": [{"payload": "reuse" * 1024}],
                "summary": {"cache_hit_count": 3, "cache_miss_count": 1},
            },
        }
    )

    minimal = _compact_log_row(row, "minimal")

    assert set(
        {
            "backend_sampling",
            "candidate_pool_sampling",
            "execution_lattice_selection",
            "case_learning_context",
            "semantic_novelty",
        }
    ).isdisjoint(minimal)
    assert minimal["execution_reuse"] == {
        "schema_version": "execution-reuse-provenance-v1",
        "summary": {"cache_hit_count": 3, "cache_miss_count": 1},
    }
    assert len(json.dumps(minimal)) < len(json.dumps(row)) * 0.35


def test_full_log_row_returns_original_object():
    row = _sample_row()

    assert _compact_log_row(row, "full") is row


def test_compact_log_keeps_ccs_summary_without_full_ir_payload():
    row = _sample_row()
    row["program_ir"] = {"ir_mode": "ccs_ir", "digest": "ccs-ir-a"}
    row["ccs_ir_summary"] = {"digest": "ccs-ir-a", "node_count": 3}
    row["ccs_ir_digest"] = "ccs-ir-a"
    row["ccs_ir"] = {"digest": "ccs-ir-a", "nodes": [{"large": "payload"}]}

    compact = _compact_log_row(row, "compact")

    assert compact["program_ir"]["ir_mode"] == "ccs_ir"
    assert compact["ccs_ir_summary"]["node_count"] == 3
    assert compact["ccs_ir_digest"] == "ccs-ir-a"
    assert "ccs_ir" not in compact


def test_finding_rows_keep_reproduction_detail_but_drop_run_metadata():
    row = _sample_row()
    row["findings"] = [{"kind": "differential", "root_cause": "row_mismatch"}]

    compact = _compact_log_row(row, "compact")

    assert compact["case"] is row["case"]
    assert compact["normalized"] is row["normalized"]
    assert compact["findings"] == row["findings"]
    assert "environment" not in compact
    assert "targets" not in compact


def test_minimal_finding_rows_keep_replay_inputs_and_witness_metadata_only():
    row = _sample_row()
    row["findings"] = [{"kind": "differential", "root_cause": "row_mismatch"}]
    row["case"]["metadata"] = {
        "case_fingerprint": {"hash": "recomputable"},
        "disagreement_descriptor": {"signature": "recomputable"},
        "goal_first_generation": {"trace": "duplicated-at-row-level"},
        "interaction_descriptor": {"plan_fingerprints": ["recomputable"]},
        "semantic_activation": {"evaluation_status": "activated"},
        "semantic_comparison_profile": {"mode": "recomputable"},
        "semantic_contract_lattice": {"nodes": ["recomputable"]},
        "boundary_application": {"profile_id": "recomputable"},
        "goal_builder_variant": {"variant_id": "recomputable"},
        "goal_fault_models": ["recomputable"],
        "goal_id": "recomputable",
        "generation_mode": "recomputable",
        "generator_profile": "recomputable",
        "semantic_activation_syntactic_reached": True,
        "semantic_witness_builder_variant": "recomputable",
        "semantic_witness_data_pattern": {"pattern_id": "recomputable"},
        "source_generator_profile": "recomputable",
        "canonical_case_replay": False,
        "runtime_corpus_io": False,
        "family_witness": {"family_id": "family-a", "axes": {"shape": "null"}},
        "confirmed_root_witness": {"root_id": "root-a", "expected": "mismatch"},
        "input_layouts": {"t0": {"representation": "sliced"}},
        "unknown_future_witness": {"classification_signal": "retain"},
    }

    minimal = _compact_log_row(row, "minimal")

    assert minimal["case"] is not row["case"]
    assert minimal["case"]["tables"] == row["case"]["tables"]
    assert minimal["case"]["program"] == row["case"]["program"]
    assert minimal["case"]["metadata"] == {
        "family_witness": {
            "family_id": "family-a",
            "axes": {"shape": "null"},
        },
        "confirmed_root_witness": {
            "root_id": "root-a",
            "expected": "mismatch",
        },
        "input_layouts": {"t0": {"representation": "sliced"}},
        "unknown_future_witness": {"classification_signal": "retain"},
    }
    assert "case_fingerprint" in row["case"]["metadata"]
    assert minimal["findings"] == row["findings"]
    assert minimal["backend_status"] == {"pandas": "ok", "duckdb": "ok"}
    assert "normalized" not in minimal
    assert "raw_results" not in minimal
    assert "environment" not in minimal
    assert "targets" not in minimal


def test_minimal_finding_row_preserves_full_plan_evidence_without_raw_payloads():
    row = _sample_row()
    full_plan = {
        "schema_version": "physical-plan-collector-v2",
        "detail": "full",
        "observations": [
            {
                "plan_kind": "physical",
                "status": "ok",
                "fingerprint": "plan-a",
                "raw_plan": "HASH_JOIN\nSEQ_SCAN",
            }
        ],
    }
    row["findings"] = [{"kind": "differential", "root_cause": "row_mismatch"}]
    row["raw_results"]["duckdb"]["physical_plan"] = full_plan
    row["finding_diagnostics"] = {
        "schema_version": "finding-plan-sidecar-v1",
        "status": "pending_fresh_recheck",
        "plans": {},
    }

    minimal = _compact_log_row(row, "minimal")

    assert minimal["findings"] == row["findings"]
    assert minimal["finding_diagnostics"]["plans"]["duckdb"] == full_plan
    assert minimal["finding_diagnostics"]["status"] == "pending_fresh_recheck"
    assert "raw_results" not in minimal
    assert "normalized" not in minimal


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
