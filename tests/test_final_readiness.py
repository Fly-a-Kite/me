import os
from pathlib import Path

from datadiff import final_readiness
from datadiff.final_readiness import (
    DEFAULT_A_LEVEL_READINESS_POLICY,
    DEFAULT_FINAL_READINESS_MANIFEST_LIMIT,
    ReadinessPolicy,
    ReadinessThresholds,
    build_final_readiness,
)
from datadiff.targets import TARGET_SUITES, describe_targets
from datadiff.util import append_jsonl, dump_json, run_meta_path


def test_final_readiness_passes_when_all_evidence_tracks_are_present(tmp_path):
    manifests = []
    manifests.append(
        _write_manifest(
            tmp_path,
            name="validation",
            evidence_mode="validation",
            target_suite="datafusion_cross",
            preset="validation_smoke",
            seed=0,
            config={"enable_replay_bug": False},
            replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
        )
    )
    for idx, suite in enumerate(DEFAULT_A_LEVEL_READINESS_POLICY.required_live_suites, start=1):
        finding = []
        if suite == "datafusion_cross":
            finding = [
                {
                    "triage_verdict": "candidate_implementation_bug",
                    "root_cause": "confirmed_root",
                    "suspicious_backends": ["datafusion"],
                    "discovery_origin": "organic",
                    "paper_status": "confirmed_bug",
                }
            ]
        manifests.append(
            _write_manifest(
                tmp_path,
                name=f"live-{suite}",
                evidence_mode="live",
                target_suite=suite,
                preset="live",
                seed=idx,
                findings=finding,
                config={"enable_replay_bug": False},
                replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
            )
        )
    manifests.append(
        _write_manifest(
            tmp_path,
            name="historical-duckdb-22075",
            evidence_mode="historical",
            target_suite="cross_family",
            preset="join_groupby_stress",
            seed=22075,
            known_bug_id="duckdb-22075",
            config={"enable_replay_bug": True},
        )
    )
    manifests.append(
        _write_manifest(
            tmp_path,
            name="historical-duckdb-22656",
            evidence_mode="historical",
            target_suite="duckdb_storage_cross",
            preset="storage_offset",
            seed=22656,
            known_bug_id="duckdb-22656",
            config={"enable_replay_bug": True},
        )
    )
    manifests.append(
        _write_manifest(
            tmp_path,
            name="seeded",
            evidence_mode="seeded",
            target_suite="seeded_filter",
            preset="guided_filter",
            seed=1,
        )
    )
    manifests.append(
        _write_manifest(
            tmp_path,
            name="ablation",
            evidence_mode="ablation",
            target_suite="core",
            preset="no_normalizer",
            seed=1,
            config={"enable_replay_bug": False},
            replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
        )
    )
    manifests.append(
        _write_manifest(
            tmp_path,
            name="comparison",
            evidence_mode="comparison",
            target_suite="embedded_sql",
            preset="baseline",
            seed=1,
            config={"enable_replay_bug": False},
            replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
        )
    )

    audit = build_final_readiness(
        manifests,
        thresholds=ReadinessThresholds(min_live_duration_hours=0.0),
    )

    assert audit["schema_version"] == "final-readiness-v1"
    assert audit["ready"] is True
    assert tuple(audit["policy"]["required_live_suites"]) == DEFAULT_A_LEVEL_READINESS_POLICY.required_live_suites
    assert {gate["name"]: gate["passed"] for gate in audit["gates"]}["short_validation"] is True
    assert {gate["name"]: gate["passed"] for gate in audit["gates"]}["live_suite_breadth"] is True
    assert audit["summary"]["validation_runs"] == 1
    assert audit["summary"]["validation_suites"] == ["datafusion_cross"]
    assert audit["summary"]["ablation_runs"] == 1
    assert audit["summary"]["comparison_runs"] == 1
    assert audit["summary"]["paper_run_journal_covered_runs"] == len(audit["runs"])
    assert audit["summary"]["paper_run_journal_missing_runs"] == []
    assert audit["summary"]["confirmed_live_candidate_families"] == {"confirmed_root@datafusion": 1}
    assert audit["summary"]["historical_confirmed_bug_ids"] == ["duckdb-22075", "duckdb-22656"]


def test_final_readiness_reports_missing_a_level_evidence(tmp_path):
    manifest = _write_manifest(
        tmp_path,
        name="live-only",
        evidence_mode="live",
        target_suite="datafusion_cross",
        preset="live_datafusion",
        seed=1,
        config={"enable_replay_bug": False},
        replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
    )

    audit = build_final_readiness(
        [manifest],
        thresholds=ReadinessThresholds(min_live_duration_hours=0.0),
    )

    gates = {gate["name"]: gate for gate in audit["gates"]}
    assert audit["ready"] is False
    assert gates["short_validation"]["passed"] is False
    assert gates["live_suite_breadth"]["passed"] is False
    assert gates["latest_confirmed_bug_families"]["passed"] is False
    assert gates["historical_confirmed_replay"]["passed"] is False
    assert gates["seeded_sensitivity"]["passed"] is False
    assert gates["module_ablation"]["passed"] is False
    assert gates["baseline_comparison"]["passed"] is False


def test_final_readiness_keeps_validation_runs_out_of_live_bug_evidence(tmp_path):
    manifest = _write_manifest(
        tmp_path,
        name="validation-with-candidate",
        evidence_mode="validation",
        target_suite="datafusion_cross",
        preset="validation_smoke",
        seed=3,
        findings=[
            {
                "triage_verdict": "candidate_implementation_bug",
                "root_cause": "validation_candidate",
                "suspicious_backends": ["datafusion"],
                "discovery_origin": "organic",
            }
        ],
        config={"enable_replay_bug": False},
        replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
    )

    audit = build_final_readiness(
        [manifest],
        thresholds=ReadinessThresholds(
            min_live_duration_hours=0.0,
            min_live_candidate_families=0,
            min_confirmed_live_families=0,
            min_historical_confirmed=0,
            require_seeded=False,
            require_ablation=False,
            require_comparison=False,
        ),
        policy=ReadinessPolicy(required_live_suites=("datafusion_cross",), required_live_families=("query_engine",)),
    )

    gates = {gate["name"]: gate for gate in audit["gates"]}
    assert gates["short_validation"]["passed"] is True
    assert audit["summary"]["validation_runs"] == 1
    assert audit["summary"]["validation_cases"] == 1
    assert audit["summary"]["explicit_live_runs"] == 0
    assert audit["summary"]["live_runs"] == 0
    assert audit["summary"]["rewardable_live_candidate_families"] == {}
    assert audit["summary"]["ignored_evidence_runs"] == 0


def test_final_readiness_keeps_support_tracks_out_of_live_bug_evidence(tmp_path):
    manifests = [
        _write_manifest(
            tmp_path,
            name="ablation-with-candidate",
            evidence_mode="ablation",
            target_suite="core",
            preset="no_normalizer",
            seed=4,
            findings=[
                {
                    "triage_verdict": "candidate_implementation_bug",
                    "root_cause": "ablation_candidate",
                    "suspicious_backends": ["datafusion"],
                    "discovery_origin": "organic",
                }
            ],
            config={"enable_replay_bug": False},
            replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
        ),
        _write_manifest(
            tmp_path,
            name="comparison-with-candidate",
            evidence_mode="comparison",
            target_suite="embedded_sql",
            preset="baseline",
            seed=5,
            findings=[
                {
                    "triage_verdict": "candidate_implementation_bug",
                    "root_cause": "comparison_candidate",
                    "suspicious_backends": ["duckdb"],
                    "discovery_origin": "organic",
                }
            ],
            config={"enable_replay_bug": False},
            replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
        ),
    ]

    audit = build_final_readiness(
        manifests,
        thresholds=ReadinessThresholds(
            min_live_duration_hours=0.0,
            min_live_candidate_families=0,
            min_confirmed_live_families=0,
            min_historical_confirmed=0,
            require_validation=False,
            require_seeded=False,
        ),
    )

    gates = {gate["name"]: gate for gate in audit["gates"]}
    assert gates["module_ablation"]["passed"] is True
    assert gates["baseline_comparison"]["passed"] is True
    assert audit["summary"]["ablation_runs"] == 1
    assert audit["summary"]["comparison_runs"] == 1
    assert audit["summary"]["live_runs"] == 0
    assert audit["summary"]["rewardable_live_candidate_families"] == {}
    assert audit["summary"]["ignored_evidence_runs"] == 0


def test_final_readiness_run_rows_include_canonical_comparison_role(tmp_path):
    manifest = _write_manifest(
        tmp_path,
        name="comparison-canonical-role",
        evidence_mode="comparison",
        target_suite="embedded_sql",
        preset="scope_variant",
        seed=9,
        config={"enable_replay_bug": False},
        replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
    )

    payload = final_readiness.load_json(manifest)
    payload["experiment_meta"] = {
        "matrix_id": "baseline_scope_comparison",
        "comparison_group": "scope_comparison",
        "analysis_tags": ["comparison"],
        "variant_by_preset": {
            "scope_variant": {
                "variant_id": "scope_variant",
                "base_preset": "stable_base",
                "comparison_role": "contrast",
                "analysis_tags": ["scope"],
            }
        },
    }
    final_readiness.dump_json(payload, manifest)

    audit = build_final_readiness(
        [manifest],
        thresholds=ReadinessThresholds(
            min_live_duration_hours=0.0,
            min_live_candidate_families=0,
            min_confirmed_live_families=0,
            min_historical_confirmed=0,
            require_validation=False,
            require_seeded=False,
            require_ablation=False,
        ),
    )

    run = next(item for item in audit["runs"] if item["evidence_mode"] == "comparison")
    assert run["comparison_role"] == "contrast"
    assert run["canonical_comparison_role"] == "contrast"
    assert run["variant_label"] == "scope_variant"
    assert run["variant_group_id"] == "embedded_sql|scope_comparison|baseline_scope_comparison"


def test_final_readiness_requires_structured_experiment_identity(tmp_path):
    manifest = _write_manifest(
        tmp_path,
        name="missing-structured-identity",
        evidence_mode="comparison",
        target_suite="embedded_sql",
        preset="guided",
        seed=10,
        config={"enable_replay_bug": False},
        replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
    )
    payload = final_readiness.load_json(manifest)
    payload["experiment_meta"] = {
        "matrix_id": "",
        "comparison_group": "",
        "analysis_tags": ["comparison"],
    }
    final_readiness.dump_json(payload, manifest)

    audit = build_final_readiness(
        [manifest],
        thresholds=ReadinessThresholds(
            min_live_duration_hours=0.0,
            min_live_candidate_families=0,
            min_confirmed_live_families=0,
            min_historical_confirmed=0,
            require_validation=False,
            require_seeded=False,
            require_ablation=False,
        ),
    )

    gates = {gate["name"]: gate for gate in audit["gates"]}
    assert gates["structured_experiment_identity"]["passed"] is False
    assert gates["structured_experiment_identity"]["missing"] == ["comparison:embedded_sql:guided"]


def test_final_readiness_requires_stage_level_profiling(tmp_path):
    manifest = _write_manifest(
        tmp_path,
        name="missing-stage-profile",
        evidence_mode="seeded",
        target_suite="seeded_filter",
        preset="guided_filter",
        seed=11,
        include_stage_profile=False,
    )

    audit = build_final_readiness(
        [manifest],
        thresholds=ReadinessThresholds(
            min_live_duration_hours=0.0,
            min_live_candidate_families=0,
            min_confirmed_live_families=0,
            min_historical_confirmed=0,
            require_validation=False,
            require_ablation=False,
            require_comparison=False,
        ),
    )

    gates = {gate["name"]: gate for gate in audit["gates"]}
    assert gates["stage_level_profiling"]["passed"] is False
    assert gates["stage_level_profiling"]["missing"] == ["seeded:seeded_filter:guided_filter"]


def test_final_readiness_requires_paper_run_journal_coverage(tmp_path):
    manifest = _write_manifest(
        tmp_path,
        name="missing-paper-journal",
        evidence_mode="validation",
        target_suite="datafusion_cross",
        preset="validation_smoke",
        seed=41,
        write_paper_run_journal=False,
    )

    audit = build_final_readiness(
        [manifest],
        thresholds=ReadinessThresholds(
            min_live_duration_hours=0.0,
            min_live_candidate_families=0,
            min_confirmed_live_families=0,
            min_historical_confirmed=0,
            require_seeded=False,
            require_ablation=False,
            require_comparison=False,
        ),
        policy=ReadinessPolicy(required_live_suites=("datafusion_cross",), required_live_families=("query_engine",)),
    )

    gates = {gate["name"]: gate for gate in audit["gates"]}
    assert gates["paper_run_journal"]["passed"] is False
    assert gates["paper_run_journal"]["missing"] == ["validation:datafusion_cross:validation_smoke"]
    assert audit["summary"]["paper_run_journal_covered_runs"] == 0
    assert audit["summary"]["paper_run_journal_missing_runs"] == ["validation:datafusion_cross:validation_smoke"]


def test_final_readiness_prefers_structured_matrix_identity_for_support_tracks(tmp_path):
    manifests = [
        _write_manifest(
            tmp_path,
            name="ablation-structured",
            evidence_mode="ablation",
            target_suite="core",
            preset="focus_variant",
            seed=11,
            config={"enable_replay_bug": False},
            replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
            run_updates={
                "matrix_id": "module_ablation",
                "comparison_group": "module_ablation",
                "variant_id": "focus_variant",
                "base_preset": "stable_base",
                "comparison_role": "contrast",
                "component_focus": "semantic_normalizer",
                "analysis_tags": ["ablation", "noise_control"],
            },
        ),
        _write_manifest(
            tmp_path,
            name="comparison-structured",
            evidence_mode="comparison",
            target_suite="embedded_sql",
            preset="scope_variant",
            seed=12,
            config={"enable_replay_bug": False},
            replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
            run_updates={
                "matrix_id": "baseline_scope_comparison",
                "comparison_group": "scope_comparison",
                "variant_id": "scope_variant",
                "base_preset": "stable_base",
                "comparison_role": "contrast",
                "analysis_tags": ["comparison", "scope"],
            },
        ),
    ]

    audit = build_final_readiness(
        manifests,
        thresholds=ReadinessThresholds(
            min_live_duration_hours=0.0,
            min_live_candidate_families=0,
            min_confirmed_live_families=0,
            min_historical_confirmed=0,
            require_validation=False,
            require_seeded=False,
        ),
    )

    gates = {gate["name"]: gate for gate in audit["gates"]}
    assert gates["module_ablation"]["passed"] is True
    assert gates["module_ablation"]["matrix_ids"] == ["module_ablation"]
    assert gates["module_ablation"]["variants"] == ["focus_variant"]
    assert gates["baseline_comparison"]["passed"] is True
    assert gates["baseline_comparison"]["matrix_ids"] == ["baseline_scope_comparison"]
    assert gates["baseline_comparison"]["variants"] == ["scope_variant"]
    assert audit["summary"]["ablation_matrix_ids"] == ["module_ablation"]
    assert audit["summary"]["ablation_variants"] == ["focus_variant"]
    assert audit["summary"]["comparison_matrix_ids"] == ["baseline_scope_comparison"]
    assert audit["summary"]["comparison_variants"] == ["scope_variant"]


def test_final_readiness_backfills_support_track_identity_from_experiment_meta(tmp_path):
    ablation_manifest = _write_manifest(
        tmp_path,
        name="ablation-meta-only",
        evidence_mode="ablation",
        target_suite="core",
        preset="focus_variant",
        seed=21,
        config={"enable_replay_bug": False},
        replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
    )
    comparison_manifest = _write_manifest(
        tmp_path,
        name="comparison-meta-only",
        evidence_mode="comparison",
        target_suite="embedded_sql",
        preset="scope_variant",
        seed=22,
        config={"enable_replay_bug": False},
        replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
    )

    ablation_payload = final_readiness.load_json(ablation_manifest)
    ablation_payload["experiment_meta"] = {
        "matrix_id": "module_ablation",
        "comparison_group": "module_ablation",
        "analysis_tags": ["ablation"],
        "variant_by_preset": {
            "focus_variant": {
                "variant_id": "focus_variant",
                "base_preset": "stable_base",
                "comparison_role": "contrast",
                "component_focus": "semantic_normalizer",
                "analysis_tags": ["noise_control"],
            }
        },
    }
    final_readiness.dump_json(ablation_payload, ablation_manifest)

    comparison_payload = final_readiness.load_json(comparison_manifest)
    comparison_payload["experiment_meta"] = {
        "matrix_id": "baseline_scope_comparison",
        "comparison_group": "scope_comparison",
        "analysis_tags": ["comparison"],
        "variant_by_preset": {
            "scope_variant": {
                "variant_id": "scope_variant",
                "base_preset": "stable_base",
                "comparison_role": "contrast",
                "analysis_tags": ["scope"],
            }
        },
    }
    final_readiness.dump_json(comparison_payload, comparison_manifest)

    audit = build_final_readiness(
        [ablation_manifest, comparison_manifest],
        thresholds=ReadinessThresholds(
            min_live_duration_hours=0.0,
            min_live_candidate_families=0,
            min_confirmed_live_families=0,
            min_historical_confirmed=0,
            require_validation=False,
            require_seeded=False,
        ),
    )

    gates = {gate["name"]: gate for gate in audit["gates"]}
    assert gates["module_ablation"]["passed"] is True
    assert gates["baseline_comparison"]["passed"] is True
    assert audit["summary"]["ablation_matrix_ids"] == ["module_ablation"]
    assert audit["summary"]["ablation_variants"] == ["focus_variant"]
    assert audit["summary"]["comparison_matrix_ids"] == ["baseline_scope_comparison"]
    assert audit["summary"]["comparison_variants"] == ["scope_variant"]


def test_final_readiness_records_registered_structured_semantic_focus(tmp_path):
    manifest = _write_manifest(
        tmp_path,
        name="validation-structured-semantic-focus",
        evidence_mode="validation",
        target_suite="datafusion_cross",
        preset="live_common_api_workflow_metamorphic",
        seed=31,
        config={
            "enable_replay_bug": False,
            "guidance_targets": [
                "common_api_workflow",
                "daily_api",
                "input_materialization",
                "filter",
                "join",
                "groupby",
                "sort_limit",
                "topk",
            ],
            "semantic_focus_families": [
                "materialization_boundary",
                "string_semantics",
                "join_membership",
                "set_semantics",
            ],
            "semantic_focus_signals": [
                "common_api_workflow",
                "filter_input_materialization",
                "distinct_input_materialization",
                "case_when_membership",
                "string_pattern_case_when",
            ],
            "effective_guidance_targets": [
                "common_api_workflow",
                "daily_api",
                "input_materialization",
                "filter",
                "join",
                "groupby",
                "sort_limit",
                "topk",
                "semantic_family:materialization_boundary",
                "semantic_family:string_semantics",
                "semantic_family:join_membership",
                "semantic_family:set_semantics",
                "semantic_signal:common_api_workflow",
                "semantic_signal:filter_input_materialization",
                "semantic_signal:distinct_input_materialization",
                "semantic_signal:case_when_membership",
                "semantic_signal:string_pattern_case_when",
            ],
        },
        replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
    )

    audit = build_final_readiness(
        [manifest],
        thresholds=ReadinessThresholds(
            min_live_duration_hours=0.0,
            min_live_candidate_families=0,
            min_confirmed_live_families=0,
            min_historical_confirmed=0,
            require_seeded=False,
            require_ablation=False,
            require_comparison=False,
        ),
    )

    gates = {gate["name"]: gate for gate in audit["gates"]}
    assert gates["structured_semantic_focus"]["passed"] is True
    assert gates["effective_guidance_targets"]["passed"] is True
    run = next(item for item in audit["runs"] if item["preset"] == "live_common_api_workflow_metamorphic")
    assert "materialization_boundary" in run["semantic_focus_families"]
    assert "common_api_workflow" in run["semantic_focus_signals"]
    assert "materialization_boundary" in audit["summary"]["semantic_focus_families"]
    assert "common_api_workflow" in audit["summary"]["semantic_focus_signals"]


def test_final_readiness_flags_registered_semantic_focus_metadata_drift(tmp_path):
    manifest = _write_manifest(
        tmp_path,
        name="validation-semantic-focus-drift",
        evidence_mode="validation",
        target_suite="datafusion_cross",
        preset="live_common_api_workflow_metamorphic",
        seed=32,
        config={
            "enable_replay_bug": False,
            "guidance_targets": [
                "common_api_workflow",
                "daily_api",
                "input_materialization",
                "filter",
                "join",
                "groupby",
                "sort_limit",
                "topk",
            ],
            "semantic_focus_families": [
                "materialization_boundary",
                "string_semantics",
                "join_membership",
                "set_semantics",
            ],
            "semantic_focus_signals": [
                "common_api_workflow",
                "filter_input_materialization",
                "distinct_input_materialization",
                "case_when_membership",
                "string_pattern_case_when",
            ],
            "effective_guidance_targets": [
                "common_api_workflow",
                "daily_api",
                "input_materialization",
                "filter",
                "join",
                "groupby",
                "sort_limit",
                "topk",
                "semantic_family:materialization_boundary",
                "semantic_family:string_semantics",
                "semantic_family:join_membership",
                "semantic_family:set_semantics",
                "semantic_signal:common_api_workflow",
                "semantic_signal:filter_input_materialization",
                "semantic_signal:distinct_input_materialization",
                "semantic_signal:case_when_membership",
                "semantic_signal:string_pattern_case_when",
            ],
        },
        replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
    )
    payload = final_readiness.load_json(manifest)
    payload["experiment_meta"] = {
        "variant_by_preset": {
            "live_common_api_workflow_metamorphic": {
                "semantic_focus_families": ["wrong_family"],
                "semantic_focus_signals": ["wrong_signal"],
            }
        }
    }
    final_readiness.dump_json(payload, manifest)

    audit = build_final_readiness(
        [manifest],
        thresholds=ReadinessThresholds(
            min_live_duration_hours=0.0,
            min_live_candidate_families=0,
            min_confirmed_live_families=0,
            min_historical_confirmed=0,
            require_seeded=False,
            require_ablation=False,
            require_comparison=False,
        ),
    )

    gates = {gate["name"]: gate for gate in audit["gates"]}
    assert gates["structured_semantic_focus"]["passed"] is False
    assert gates["structured_semantic_focus"]["issues"] == [
        "validation:datafusion_cross:live_common_api_workflow_metamorphic"
    ]


def test_final_readiness_flags_effective_guidance_target_drift(tmp_path):
    manifest = _write_manifest(
        tmp_path,
        name="validation-effective-guidance-drift",
        evidence_mode="validation",
        target_suite="datafusion_cross",
        preset="live_common_api_workflow_metamorphic",
        seed=33,
        config={"enable_replay_bug": False},
        replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
    )
    run_file = Path(final_readiness.load_json(manifest)["runs"][0]["run_file"])
    meta_file = run_meta_path(run_file)
    meta_payload = final_readiness.load_json(meta_file)
    meta_payload["config"]["effective_guidance_targets"] = ["wrong_target"]
    final_readiness.dump_json(meta_payload, meta_file)

    audit = build_final_readiness(
        [manifest],
        thresholds=ReadinessThresholds(
            min_live_duration_hours=0.0,
            min_live_candidate_families=0,
            min_confirmed_live_families=0,
            min_historical_confirmed=0,
            require_seeded=False,
            require_ablation=False,
            require_comparison=False,
        ),
    )

    gates = {gate["name"]: gate for gate in audit["gates"]}
    assert gates["effective_guidance_targets"]["passed"] is False
    assert gates["effective_guidance_targets"]["issues"] == [
        "validation:datafusion_cross:live_common_api_workflow_metamorphic"
    ]


def test_final_readiness_summary_only_mode_does_not_claim_paper_readiness(tmp_path):
    manifest = _write_manifest(
        tmp_path,
        name="live-summary-only",
        evidence_mode="live",
        target_suite="datafusion_cross",
        preset="live_datafusion",
        seed=6,
        findings=[
            {
                "triage_verdict": "candidate_implementation_bug",
                "root_cause": "summary_only_candidate",
                "suspicious_backends": ["datafusion"],
                "discovery_origin": "organic",
            }
        ],
        config={"enable_replay_bug": False},
        replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
    )

    audit = build_final_readiness(
        [manifest],
        scan_run_logs=False,
        thresholds=ReadinessThresholds(
            min_live_duration_hours=0.0,
            min_live_candidate_families=0,
            min_confirmed_live_families=0,
            min_historical_confirmed=0,
            require_validation=False,
            require_seeded=False,
            require_ablation=False,
            require_comparison=False,
        ),
        policy=ReadinessPolicy(required_live_suites=("datafusion_cross",), required_live_families=("query_engine",)),
    )

    gates = {gate["name"]: gate for gate in audit["gates"]}
    assert audit["ready"] is False
    assert gates["run_log_scan"]["passed"] is False
    assert audit["summary"]["run_logs_scanned"] == 0
    assert audit["summary"]["run_logs_scan_skipped"] == 1
    assert audit["summary"]["live_runs"] == 1
    assert audit["summary"]["rewardable_live_candidate_families"] == {}


def test_final_readiness_counts_legacy_historical_replay_without_run_replay_flag(tmp_path):
    manifest = _write_manifest(
        tmp_path,
        name="legacy-historical-duckdb-22075",
        evidence_mode="historical",
        target_suite="cross_family",
        preset="join_groupby_stress",
        seed=22075,
        known_bug_id="duckdb-22075",
        config={},
    )

    audit = build_final_readiness(
        [manifest],
        thresholds=ReadinessThresholds(
            min_live_duration_hours=0.0,
            min_live_candidate_families=0,
            min_confirmed_live_families=0,
            min_historical_confirmed=1,
            require_validation=False,
            require_seeded=False,
            require_ablation=False,
            require_comparison=False,
        ),
    )

    assert audit["summary"]["historical_confirmed_bug_ids"] == ["duckdb-22075"]
    assert audit["runs"][0]["enable_replay_bug"] is True


def test_final_readiness_does_not_default_legacy_manifest_to_live(tmp_path):
    manifest = _write_manifest(
        tmp_path,
        name="legacy-with-candidate",
        evidence_mode=None,
        target_suite="datafusion_cross",
        preset="old_live_datafusion",
        seed=7,
        findings=[
            {
                "triage_verdict": "candidate_implementation_bug",
                "root_cause": "legacy_candidate",
                "suspicious_backends": ["datafusion"],
                "discovery_origin": "organic",
            }
        ],
        config={"enable_replay_bug": False},
        replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
    )

    audit = build_final_readiness(
        [manifest],
        thresholds=ReadinessThresholds(
            min_live_duration_hours=0.0,
            min_live_candidate_families=0,
            min_confirmed_live_families=0,
            min_historical_confirmed=0,
            require_validation=False,
            require_seeded=False,
            require_ablation=False,
            require_comparison=False,
        ),
        policy=ReadinessPolicy(required_live_suites=("datafusion_cross",), required_live_families=("query_engine",)),
    )

    assert audit["summary"]["live_runs"] == 0
    assert audit["summary"]["ignored_evidence_runs"] == 1
    assert audit["summary"]["rewardable_live_candidate_families"] == {}
    assert {gate["name"]: gate["passed"] for gate in audit["gates"]}["live_suite_breadth"] is False


def test_final_readiness_counts_only_fresh_policy_live_runs_as_latest_evidence(tmp_path):
    manifest = _write_manifest(
        tmp_path,
        name="live-without-replay-filter",
        evidence_mode="live",
        target_suite="datafusion_cross",
        preset="old_live_datafusion",
        seed=9,
        findings=[
            {
                "triage_verdict": "candidate_implementation_bug",
                "root_cause": "stale_candidate",
                "suspicious_backends": ["datafusion"],
                "discovery_origin": "organic",
            }
        ],
        config={"enable_replay_bug": False},
    )

    audit = build_final_readiness(
        [manifest],
        thresholds=ReadinessThresholds(
            min_live_duration_hours=0.0,
            min_live_candidate_families=0,
            min_confirmed_live_families=0,
            min_historical_confirmed=0,
            require_validation=False,
            require_seeded=False,
            require_ablation=False,
            require_comparison=False,
        ),
        policy=ReadinessPolicy(required_live_suites=("datafusion_cross",), required_live_families=("query_engine",)),
    )

    assert audit["summary"]["explicit_live_runs"] == 1
    assert audit["summary"]["live_runs"] == 0
    assert audit["summary"]["replay_policy_rejected_live_runs"] == 1
    assert audit["summary"]["rewardable_live_candidate_families"] == {}
    assert {gate["name"]: gate["passed"] for gate in audit["gates"]}["live_suite_breadth"] is False


def test_final_readiness_rejects_seeded_suite_as_latest_evidence(tmp_path):
    manifest = _write_manifest(
        tmp_path,
        name="seeded-mislabeled-live",
        evidence_mode="live",
        target_suite="seeded_filter",
        preset="guided_filter",
        seed=8,
        findings=[
            {
                "triage_verdict": "candidate_implementation_bug",
                "root_cause": "filter_predicate",
                "suspicious_backends": ["buggy_filter"],
                "discovery_origin": "organic",
            }
        ],
        config={"enable_replay_bug": False},
        replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
    )

    audit = build_final_readiness(
        [manifest],
        thresholds=ReadinessThresholds(
            min_live_duration_hours=0.0,
            min_live_candidate_families=0,
            min_confirmed_live_families=0,
            min_historical_confirmed=0,
            require_validation=False,
            require_seeded=False,
            require_ablation=False,
            require_comparison=False,
        ),
        policy=ReadinessPolicy(required_live_suites=("seeded_filter",), required_live_families=("seeded_fault",)),
    )

    assert audit["summary"]["live_runs"] == 0
    assert audit["summary"]["seeded_runs"] == 0
    assert audit["summary"]["ignored_evidence_runs"] == 1
    assert audit["summary"]["rewardable_live_candidate_families"] == {}


def test_final_readiness_excludes_known_saturated_live_families_from_latest_evidence(tmp_path):
    manifest = _write_manifest(
        tmp_path,
        name="live-known-family",
        evidence_mode="live",
        target_suite="datafusion_cross",
        preset="live_datafusion",
        seed=1,
        findings=[
            {
                "triage_verdict": "candidate_implementation_bug",
                "root_cause": "grouped_topk_null_sort_key",
                "suspicious_backends": ["datafusion"],
                "discovery_origin": "organic",
                "paper_status": "confirmed_bug",
            }
        ],
        config={
            "enable_replay_bug": False,
            "known_saturated_bug_families": ["grouped_topk_null_sort_key@datafusion"],
        },
        replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
    )

    audit = build_final_readiness(
        [manifest],
        thresholds=ReadinessThresholds(
            min_live_duration_hours=0.0,
            min_live_candidate_families=0,
            min_confirmed_live_families=0,
            min_historical_confirmed=0,
            require_validation=False,
            require_seeded=False,
            require_ablation=False,
            require_comparison=False,
        ),
        policy=ReadinessPolicy(required_live_suites=("datafusion_cross",), required_live_families=("query_engine",)),
    )

    assert audit["summary"]["known_saturated_live_candidate_families"] == {
        "grouped_topk_null_sort_key@datafusion": 1
    }
    assert audit["summary"]["rewardable_live_candidate_families"] == {}
    assert audit["summary"]["confirmed_live_candidate_families"] == {}


def test_final_readiness_counts_external_upstream_confirmation_without_rewarding_known_family(tmp_path):
    manifest = _write_manifest(
        tmp_path,
        name="live-known-family",
        evidence_mode="live",
        target_suite="datafusion_cross",
        preset="live_datafusion",
        seed=1,
        findings=[
            {
                "triage_verdict": "candidate_implementation_bug",
                "root_cause": "grouped_topk_null_sort_key",
                "suspicious_backends": ["datafusion"],
                "discovery_origin": "organic",
                "paper_status": "candidate_bug_needs_external_confirmation",
            }
        ],
        config={
            "enable_replay_bug": False,
            "known_saturated_bug_families": ["grouped_topk_null_sort_key@datafusion"],
        },
        replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
    )
    confirmation_file = tmp_path / "latest_confirmations.json"
    dump_json(
        {
            "schema_version": 1,
            "confirmations": [
                {
                    "family": "grouped_topk_null_sort_key@datafusion",
                    "issue_url": "https://github.com/apache/datafusion/issues/22190",
                    "upstream_status": "upstream_labeled_bug",
                    "labels": ["bug"],
                }
            ],
        },
        confirmation_file,
    )

    audit = build_final_readiness(
        [manifest],
        latest_confirmation_files=[confirmation_file],
        thresholds=ReadinessThresholds(
            min_live_duration_hours=0.0,
            min_live_candidate_families=0,
            min_confirmed_live_families=1,
            min_historical_confirmed=0,
            require_validation=False,
            require_seeded=False,
            require_ablation=False,
            require_comparison=False,
        ),
        policy=ReadinessPolicy(required_live_suites=("datafusion_cross",), required_live_families=("query_engine",)),
    )

    assert audit["summary"]["rewardable_live_candidate_families"] == {}
    assert audit["summary"]["external_confirmed_live_candidate_families"] == {
        "grouped_topk_null_sort_key@datafusion": 1
    }
    assert audit["summary"]["confirmed_live_candidate_families"] == {
        "grouped_topk_null_sort_key@datafusion": 1
    }
    assert {gate["name"]: gate["passed"] for gate in audit["gates"]}["latest_confirmed_bug_families"] is True


def test_final_readiness_policy_keeps_top_level_requirements_out_of_engine(tmp_path):
    manifest = _write_manifest(
        tmp_path,
        name="live-only",
        evidence_mode="live",
        target_suite="datafusion_cross",
        preset="live_datafusion",
        seed=1,
        config={"enable_replay_bug": False},
        replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
    )

    audit = build_final_readiness(
        [manifest],
        thresholds=ReadinessThresholds(
            min_live_duration_hours=0.0,
            min_live_candidate_families=0,
            min_confirmed_live_families=0,
            min_historical_confirmed=0,
            require_validation=False,
            require_seeded=False,
            require_ablation=False,
            require_comparison=False,
        ),
        policy=ReadinessPolicy(
            required_live_suites=("datafusion_cross",),
            required_live_families=("dataframe", "embedded_sql", "query_engine"),
        ),
    )

    assert audit["ready"] is True
    assert audit["policy"]["required_live_suites"] == ("datafusion_cross",)


def test_analyze_final_readiness_writes_markdown_and_json(tmp_path, monkeypatch):
    reports_dir = tmp_path / "reports"
    monkeypatch.setattr(final_readiness, "REPORTS_DIR", reports_dir)
    manifest = _write_manifest(
        tmp_path,
        name="live-only",
        evidence_mode="live",
        target_suite="datafusion_cross",
        preset="live_datafusion",
        seed=1,
        config={"enable_replay_bug": False},
        replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
    )

    md_path, json_path = final_readiness.analyze_final_readiness(
        [manifest],
        thresholds=ReadinessThresholds(
            min_live_duration_hours=0.0,
            min_live_candidate_families=0,
            min_confirmed_live_families=0,
            min_historical_confirmed=0,
            require_validation=False,
            require_seeded=False,
            require_ablation=False,
            require_comparison=False,
        ),
    )

    assert md_path.exists()
    assert json_path.exists()
    assert "Final Experiment Readiness" in md_path.read_text(encoding="utf-8")


def test_final_readiness_default_manifest_resolution_uses_latest_limit(tmp_path, monkeypatch):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    manifests = []
    for idx in range(DEFAULT_FINAL_READINESS_MANIFEST_LIMIT + 2):
        path = runs_dir / f"experiment-{idx:02d}.json"
        path.write_text('{"runs": []}\n', encoding="utf-8")
        timestamp_ns = 1_800_000_000_000_000_000 + idx
        path.touch()
        os.utime(path, ns=(timestamp_ns, timestamp_ns))
        manifests.append(path)
    monkeypatch.setattr(final_readiness, "RUNS_DIR", runs_dir)

    selected = final_readiness._resolve_manifest_files(None)

    assert selected == manifests[-DEFAULT_FINAL_READINESS_MANIFEST_LIMIT:]
    assert final_readiness._resolve_manifest_files(None, manifest_limit=None) == manifests
    assert final_readiness._resolve_manifest_files([manifests[0]], manifest_limit=1) == [manifests[0]]


def test_final_readiness_rejects_live_runs_without_latest_frozen_authority_provenance(tmp_path, monkeypatch):
    manifest = _write_manifest(
        tmp_path,
        name="live-provenance-mismatch",
        evidence_mode="live",
        target_suite="datafusion_cross",
        preset="live_datafusion",
        seed=1,
        config={"enable_replay_bug": False},
        replay_filter={"enabled": True, "filtered_candidates": 0, "fallback_candidates": 0},
        run_provenance={
            "vcs": {"git_commit": "old-head", "workspace_dirty": True},
            "harness": {"authority": False, "freeze_intent": False, "latest_code_claim": False},
        },
    )
    monkeypatch.setattr(final_readiness, "current_workspace_git_commit", lambda: "new-head")

    audit = build_final_readiness(
        [manifest],
        thresholds=ReadinessThresholds(
            min_live_duration_hours=0.0,
            min_live_candidate_families=0,
            min_confirmed_live_families=0,
            min_historical_confirmed=0,
            require_validation=False,
            require_seeded=False,
            require_ablation=False,
            require_comparison=False,
        ),
        policy=ReadinessPolicy(required_live_suites=("datafusion_cross",), required_live_families=("query_engine",)),
    )

    gates = {gate["name"]: gate for gate in audit["gates"]}
    assert gates["latest_live_provenance"]["passed"] is False
    assert sorted(gates["latest_live_provenance"]["issues"]) == [
        "live:datafusion_cross:live_datafusion:dirty_or_unknown_workspace",
        "live:datafusion_cross:live_datafusion:freeze_not_declared",
        "live:datafusion_cross:live_datafusion:head_mismatch",
        "live:datafusion_cross:live_datafusion:latest_code_not_declared",
        "live:datafusion_cross:live_datafusion:missing_git_status_artifact",
        "live:datafusion_cross:live_datafusion:missing_launcher_env_artifact",
        "live:datafusion_cross:live_datafusion:missing_manifest_artifact",
        "live:datafusion_cross:live_datafusion:missing_pip_freeze_artifact",
        "live:datafusion_cross:live_datafusion:missing_strategy_snapshot_artifact",
        "live:datafusion_cross:live_datafusion:non_authority",
    ]


def _write_manifest(
    root: Path,
    *,
    name: str,
    evidence_mode: str | None,
    target_suite: str,
    preset: str,
    seed: int,
    findings: list[dict] | None = None,
    known_bug_id: str = "",
    config: dict | None = None,
    replay_filter: dict | None = None,
    run_updates: dict | None = None,
    include_stage_profile: bool = True,
    run_provenance: dict | None = None,
    write_paper_run_journal: bool = True,
) -> Path:
    runs_dir = root / "runs"
    run_file = runs_dir / f"run-{name}.jsonl"
    backends = list(TARGET_SUITES[target_suite])
    append_jsonl(
        {
            "case_index": 0,
            "case": {"case_id": f"case-{name}", "seed": seed, "program": {"operations": []}},
            "findings": findings or [],
        },
        run_file,
    )
    run_meta = {
        "executed_cases": 1,
        "elapsed_s": 1.0,
        "throughput_cases_s": 1.0,
        "backends": backends,
        "targets": describe_targets(backends),
        "config": config or {},
        "replay_bug_filter": replay_filter or {},
        "run_provenance": run_provenance
        or {
            "schema_version": "run-provenance-v1",
            "vcs": {
                "git_commit": final_readiness.current_workspace_git_commit(),
                "git_commit_short": "current-head",
                "git_branch": "main",
                "workspace_dirty": False,
            },
            "launch": {
                "source": "closed_loop_tmux",
                "session": "authority-test",
                "duration": "24h" if evidence_mode == "live" else "short",
                "batch_duration": "10m",
                "log_prefix": name,
                "launch_script": "start_closed_loop_24h_tmux.sh" if evidence_mode == "live" else "manual",
            },
            "harness": {
                "authority": evidence_mode == "live",
                "freeze_intent": evidence_mode == "live",
                "latest_code_claim": evidence_mode == "live",
                "evidence_role": "latest_live_authority_24h" if evidence_mode == "live" else "",
            },
            "freeze_artifacts": {
                "manifest": str(root / "reports" / f"{name}.freeze.json"),
                "pip_freeze": str(root / "reports" / f"{name}.pip-freeze.txt"),
                "git_status": str(root / "reports" / f"{name}.git-status.txt"),
                "git_diff": str(root / "reports" / f"{name}.git-diff.patch"),
                "launcher_env": str(root / "reports" / f"{name}.launcher-env.txt"),
                "strategy_snapshot": str(root / "reports" / f"{name}.strategy-snapshot.json"),
            },
        },
    }
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    freeze_artifacts = run_meta["run_provenance"].get("freeze_artifacts", {})
    for artifact_name, artifact_path in freeze_artifacts.items():
        path = Path(str(artifact_path))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{artifact_name}\n", encoding="utf-8")
    if include_stage_profile:
        run_meta["stage_profile"] = {
            "totals_ms": {
                "generate_mutate_ms": 0.1,
                "backend_execution_ms": 0.2,
                "normalize_ms": 0.05,
                "oracle_classification_ms": 0.03,
                "scheduler_feedback_ms": 0.02,
                "logging_artifact_ms": 0.01,
                "total_case_wall_ms": 0.41,
            }
        }
    dump_json(run_meta, run_meta_path(run_file))
    if write_paper_run_journal:
        append_jsonl(
            {
                "run_file": str(run_file),
                "evidence_mode": evidence_mode,
                "target_suite": target_suite,
                "preset": preset,
                "seed": seed,
            },
            reports_dir / "paper-run-journal.jsonl",
        )
    manifest = runs_dir / f"experiment-{name}.json"
    run_payload = {
        "target_suite": target_suite,
        "preset": preset,
        "seed": seed,
        "known_bug_id": known_bug_id,
        "run_file": str(run_file),
        "backends": backends,
        "report": "",
    }
    if run_updates:
        run_payload.update(run_updates)
    manifest_payload = {
        "target_suite": target_suite,
        "target_suites": [target_suite],
        "known_bug_id": known_bug_id,
        "backends": backends,
        "targets": describe_targets(backends),
        "replay_bug_policy": {"enable_replay_bug": evidence_mode == "historical"},
        "runs": [run_payload],
    }
    if evidence_mode is not None:
        manifest_payload["evidence_mode"] = evidence_mode
        run_payload["evidence_mode"] = evidence_mode
    dump_json(manifest_payload, manifest)
    return manifest
