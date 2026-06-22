import pytest

from datadiff.experiment_metadata import (
    component_focus,
    canonical_comparison_role,
    experiment_row_group_id,
    experiment_row_variant_id,
    experiment_row_variant_key,
    experiment_row_variant_label,
    is_contrast_experiment_row,
    is_contrast_variant,
    is_ablation_experiment_row,
    is_reference_experiment_row,
    is_reference_variant,
    manifest_experiment_meta,
    merge_experiment_meta,
    normalize_experiment_meta,
    parse_experiment_meta,
    contrast_variant_id_for_suite,
    reference_row_for_group,
    resolved_run_semantics,
)
from datadiff.experiment_catalog import registered_experiment_meta_defaults, resolve_historical_experiment_meta
from datadiff.preset_catalog import build_experiment_config


def test_resolved_run_semantics_ignores_empty_explicit_variant_overlay() -> None:
    experiment_meta = {
        "matrix_id": "module_ablation",
        "matrix_title": "Module Ablation",
        "comparison_group": "module_ablation",
        "rq_tags": ["RQ2", "RQ4"],
        "analysis_tags": ["ablation", "noise_control"],
        "counts_as_real_bugs": False,
        "scope_by_target_suite": {"core": "core"},
        "variant_by_preset": {
            "baseline": {
                "variant_id": "baseline",
                "variant_title": "baseline",
                "base_preset": "baseline",
                "comparison_role": "baseline",
                "component_focus": "",
                "overlays": [],
                "factors": {"type_aware_generation": True},
                "oracle_profile": "differential",
                "rq_tags": ["RQ2", "RQ4"],
                "analysis_tags": ["ablation"],
            }
        },
        "variant": {
            "variant_id": "",
            "variant_title": "",
            "base_preset": "",
            "comparison_role": "",
            "component_focus": "",
            "overlays": [],
            "factors": {},
            "oracle_profile": "",
            "rq_tags": [],
            "analysis_tags": [],
        },
    }

    run_semantics = resolved_run_semantics({"preset": "baseline", "target_suite": "core"}, experiment_meta)

    assert run_semantics["variant_id"] == "baseline"
    assert run_semantics["scope_kind"] == "core"
    assert run_semantics["oracle_profile"] == "differential"
    assert run_semantics["canonical_comparison_role"] == "baseline"
    assert run_semantics["factors"] == {"type_aware_generation": True}


def test_component_focus_recognizes_witness_oracle_factor() -> None:
    assert component_focus({"factors": {"witness_oracle": True}}) == "witness_oracle"


def test_parse_experiment_meta_rejects_non_object_json() -> None:
    with pytest.raises(ValueError, match="experiment meta must decode to an object"):
        parse_experiment_meta("[]")


def test_normalize_experiment_meta_preserves_structured_fields() -> None:
    normalized = normalize_experiment_meta(
        {
            "matrix_id": "module_ablation",
            "comparison_group": "module_ablation",
            "rq_tags": ["RQ2", "", "RQ4"],
            "analysis_tags": ["ablation", "", "noise_control"],
            "target_suites": ["core", ""],
            "scope_by_target_suite": {"core": "core", "": "drop"},
            "variant_by_preset": {
                "baseline": {
                    "variant_id": "baseline",
                    "factors": {"type_aware_generation": True},
                    "overlays": ["enable_x", ""],
                    "semantic_focus_families": ["join_membership", ""],
                    "semantic_focus_signals": ["row_value_absence_filter", ""],
                    "analysis_tags": ["ablation", ""],
                }
            },
            "historical": {
                "bug_id": "bug-1",
                "expected_root_causes": ["join_semantics", ""],
                "expected_suspicious_backends": ["duckdb", ""],
            },
        }
    )

    assert normalized["matrix_id"] == "module_ablation"
    assert normalized["comparison_group"] == "module_ablation"
    assert normalized["rq_tags"] == ["RQ2", "RQ4"]
    assert normalized["analysis_tags"] == ["ablation", "noise_control"]
    assert normalized["target_suites"] == ["core"]
    assert normalized["scope_by_target_suite"] == {"core": "core"}
    assert normalized["variant_by_preset"]["baseline"]["overlays"] == ["enable_x"]
    assert normalized["variant_by_preset"]["baseline"]["semantic_focus_families"] == ["join_membership"]
    assert normalized["variant_by_preset"]["baseline"]["semantic_focus_signals"] == ["row_value_absence_filter"]
    assert normalized["variant_by_preset"]["baseline"]["factors"] == {"type_aware_generation": True}
    assert normalized["historical"]["expected_root_causes"] == ["join_semantics"]
    assert normalized["historical"]["expected_suspicious_backends"] == ["duckdb"]


def test_manifest_experiment_meta_normalizes_manifest_payload() -> None:
    meta = manifest_experiment_meta(
        {
            "experiment_meta": {
                "matrix_id": "module_ablation",
                "comparison_group": "module_ablation",
                "rq_tags": ["RQ2", "", "RQ4"],
                "analysis_tags": ["ablation", "", "noise_control"],
                "variant_by_preset": {
                    "baseline": {
                        "variant_id": "baseline",
                        "overlays": ["enable_x", ""],
                        "semantic_focus_families": ["join_membership", ""],
                        "semantic_focus_signals": ["row_value_absence_filter", ""],
                        "analysis_tags": ["ablation", ""],
                    }
                },
            }
        }
    )

    assert meta["matrix_id"] == "module_ablation"
    assert meta["comparison_group"] == "module_ablation"
    assert meta["rq_tags"] == ["RQ2", "RQ4"]
    assert meta["analysis_tags"] == ["ablation", "noise_control"]
    assert meta["variant_by_preset"]["baseline"]["overlays"] == ["enable_x"]
    assert meta["variant_by_preset"]["baseline"]["semantic_focus_families"] == ["join_membership"]
    assert meta["variant_by_preset"]["baseline"]["semantic_focus_signals"] == ["row_value_absence_filter"]
    assert meta["variant_by_preset"]["baseline"]["analysis_tags"] == ["ablation"]


def test_resolved_run_semantics_preserves_structured_semantic_focus() -> None:
    experiment_meta = {
        "matrix_id": "module_ablation",
        "comparison_group": "module_ablation",
        "variant_by_preset": {
            "focus_variant": {
                "variant_id": "focus_variant",
                "comparison_role": "contrast",
                "semantic_focus_families": ["join_membership"],
                "semantic_focus_signals": ["row_value_absence_filter"],
            }
        },
    }

    run_semantics = resolved_run_semantics(
        {"preset": "focus_variant", "target_suite": "core"},
        experiment_meta,
    )

    assert run_semantics["semantic_focus_families"] == ["join_membership"]
    assert run_semantics["semantic_focus_signals"] == ["row_value_absence_filter"]


def test_manifest_experiment_meta_backfills_registered_final_matrix_defaults() -> None:
    meta = manifest_experiment_meta(
        {
            "evidence_mode": "seeded",
            "target_suite": "seeded_filter",
            "runs": [
                {
                    "target_suite": "seeded_filter",
                    "preset": "guided_filter",
                }
            ],
        }
    )

    assert meta["matrix_id"] == "seeded_sensitivity"
    assert meta["comparison_group"] == "seeded_sensitivity"
    assert meta["scope_by_target_suite"] == {"seeded_filter": "seeded_fault_injection"}
    assert meta["variant_by_preset"]["guided_filter"]["comparison_role"] == "contrast"


def test_manifest_experiment_meta_preserves_registered_identity_under_partial_variant_override() -> None:
    meta = manifest_experiment_meta(
        {
            "evidence_mode": "validation",
            "target_suite": "datafusion_cross",
            "runs": [
                {
                    "target_suite": "datafusion_cross",
                    "preset": "live_common_api_workflow_metamorphic",
                }
            ],
            "experiment_meta": {
                "variant_by_preset": {
                    "live_common_api_workflow_metamorphic": {
                        "semantic_focus_families": ["wrong_family"],
                        "semantic_focus_signals": ["wrong_signal"],
                    }
                }
            },
        }
    )

    assert meta["matrix_id"] == "final_validation"
    assert meta["comparison_group"] == "validation_smoke"
    variant = meta["variant_by_preset"]["live_common_api_workflow_metamorphic"]
    assert variant["variant_id"] == "live_common_api_workflow_metamorphic"
    assert variant["base_preset"] == "live_common_api_workflow"
    assert variant["oracle_profile"] == "both"
    assert variant["semantic_focus_families"] == ["wrong_family"]
    assert variant["semantic_focus_signals"] == ["wrong_signal"]


def test_merge_experiment_meta_merges_nested_variant_and_historical_payloads() -> None:
    merged = merge_experiment_meta(
        {
            "matrix_id": "historical_replay",
            "variant": {
                "variant_id": "bug-1",
                "analysis_tags": ["historical"],
                "factors": {"historical_status": "pending"},
            },
            "historical": {
                "bug_id": "bug-1",
                "status": "pending_merge",
                "expected_root_causes": ["join_semantics"],
            },
        },
        {
            "target_version": "sha-1",
            "variant": {"scope_kind": "core"},
            "historical": {"expected_suspicious_backends": ["duckdb"]},
        },
    )

    assert merged["target_version"] == "sha-1"
    assert merged["variant"]["variant_id"] == "bug-1"
    assert merged["variant"]["scope_kind"] == "core"
    assert merged["variant"]["factors"] == {"historical_status": "pending"}
    assert merged["historical"]["bug_id"] == "bug-1"
    assert merged["historical"]["status"] == "pending_merge"
    assert merged["historical"]["expected_root_causes"] == ["join_semantics"]
    assert merged["historical"]["expected_suspicious_backends"] == ["duckdb"]


def test_merge_experiment_meta_preserves_registered_variant_catalog_when_override_is_matrix_level_only() -> None:
    merged = merge_experiment_meta(
        {
            "matrix_id": "baseline_scope_comparison",
            "scope_by_target_suite": {"latest_no_datafusion": "cross_ecosystem"},
            "variant_by_preset": {
                "baseline": {
                    "variant_id": "baseline",
                    "comparison_role": "baseline",
                },
                "live_cross_family": {
                    "variant_id": "live_cross_family",
                    "comparison_role": "contrast",
                    "scope_kind": "cross_ecosystem",
                },
            },
        },
        {
            "matrix_id": "baseline_scope_comparison",
            "comparison_group": "scope_comparison",
            "analysis_tags": ["comparison", "scope"],
        },
    )

    assert merged["comparison_group"] == "scope_comparison"
    assert merged["scope_by_target_suite"] == {"latest_no_datafusion": "cross_ecosystem"}
    assert merged["variant_by_preset"]["baseline"]["comparison_role"] == "baseline"
    assert merged["variant_by_preset"]["live_cross_family"]["comparison_role"] == "contrast"
    assert merged["variant_by_preset"]["live_cross_family"]["scope_kind"] == "cross_ecosystem"


def test_resolve_historical_experiment_meta_uses_registered_spec() -> None:
    meta = resolve_historical_experiment_meta(
        known_bug_id="datafusion-22190",
        target_suite="core",
        target_version="pre-fix-sha",
    )

    assert meta["matrix_id"] == "historical_replay"
    assert meta["comparison_group"] == "historical_replay"
    assert meta["known_bug_id"] == "datafusion-22190"
    assert meta["historical"]["status"] == "pending_merge"
    assert meta["historical"]["target_version"] == "pre-fix-version-required"
    assert meta["variant"]["variant_id"] == "datafusion-22190"
    assert meta["variant"]["factors"]["historical_status"] == "pending_merge"


def test_resolve_historical_experiment_meta_builds_unregistered_fallback() -> None:
    meta = resolve_historical_experiment_meta(
        known_bug_id="fixture-test",
        target_suite="core",
        target_version="pre-fix-sha",
    )

    assert meta["matrix_id"] == "historical_replay"
    assert meta["comparison_group"] == "historical_replay"
    assert meta["known_bug_id"] == "fixture-test"
    assert meta["scope_kind"] == "core"
    assert meta["historical"]["status"] == "unregistered"
    assert meta["historical"]["target_version"] == "pre-fix-sha"
    assert meta["variant"]["variant_id"] == "fixture-test"
    assert meta["variant"]["factors"]["historical_status"] == "unregistered"


def test_registered_experiment_meta_defaults_resolves_historical_metadata() -> None:
    meta = registered_experiment_meta_defaults(
        evidence_mode="historical",
        known_bug_id="fixture-test",
        target_suite="core",
        target_version="pre-fix-sha",
    )

    assert meta["matrix_id"] == "historical_replay"
    assert meta["historical"]["bug_id"] == "fixture-test"


def test_registered_final_matrix_variant_metadata_keeps_executable_overlay_details() -> None:
    meta = manifest_experiment_meta(
        {
            "evidence_mode": "validation",
            "runs": [
                {"target_suite": "latest_no_datafusion", "preset": "live_common_api_workflow_metamorphic"},
                {"target_suite": "latest_no_datafusion", "preset": "live_deep_organic_metamorphic"},
            ],
        }
    )

    workflow_variant = meta["variant_by_preset"]["live_common_api_workflow_metamorphic"]
    deep_variant = meta["variant_by_preset"]["live_deep_organic_metamorphic"]

    assert workflow_variant["base_preset"] == "live_common_api_workflow"
    assert workflow_variant["overlays"] == [
        "enable_metamorphic_oracle",
        "candidate_pool_8",
        "metamorphic_variant_limit_6",
    ]
    assert "materialization_boundary" in workflow_variant["semantic_focus_families"]
    assert "common_api_workflow" in workflow_variant["semantic_focus_signals"]
    assert deep_variant["base_preset"] == "live_deep_organic"
    assert deep_variant["overlays"] == [
        "enable_metamorphic_oracle",
        "candidate_pool_10",
        "metamorphic_variant_limit_8",
    ]
    assert "join_membership" in deep_variant["semantic_focus_families"]


def test_guided_groupby_preset_uses_groupby_focused_discovery_targets() -> None:
    config = build_experiment_config("guided_groupby")

    assert config.generator_profile == "discovery"
    assert config.guidance_strategy == "guided"
    assert config.guidance_candidate_pool == 8
    assert config.family_saturation_threshold == 16
    assert {
        "groupby_aggregation",
        "multi_key_groupby",
        "null_agg_topk",
        "join_filter_groupby",
        "groupby_having_topk",
    }.issubset(config.guidance_targets)
    assert config.discovery_biases
    assert config.discovery_biases[0].keep_in_pool is True


def test_registered_experiment_meta_defaults_promotes_uniform_scope_kind_for_single_suite_selection() -> None:
    meta = registered_experiment_meta_defaults(
        evidence_mode="comparison",
        target_suite="embedded_sql",
    )

    assert meta["scope_kind"] == "sql_oriented"
    assert meta["variant_by_preset"]["guided"]["scope_kind"] == "sql_oriented"
    assert meta["variant_by_preset"]["workflow"]["scope_kind"] == "sql_oriented"


def test_registered_experiment_meta_for_manifest_keeps_multi_scope_matrix_unset_at_matrix_level() -> None:
    meta = manifest_experiment_meta(
        {
            "evidence_mode": "comparison",
            "runs": [
                {"target_suite": "embedded_sql", "preset": "baseline"},
                {"target_suite": "latest_no_datafusion", "preset": "live_cross_family"},
            ],
        }
    )

    assert meta["scope_kind"] == ""
    assert meta["variant_by_preset"]["baseline"]["scope_kind"] == ""
    assert meta["variant_by_preset"]["guided"]["scope_kind"] == ""
    assert meta["variant_by_preset"]["live_cross_family"]["scope_kind"] == "cross_ecosystem"


def test_is_ablation_experiment_row_uses_structured_ablation_semantics() -> None:
    assert is_ablation_experiment_row(
        {
            "evidence_mode": "ablation",
            "matrix_id": "module_ablation",
            "preset": "focus_variant",
            "comparison_role": "contrast",
            "component_focus": "semantic_normalizer",
            "analysis_tags": "ablation,noise_control",
        }
    )
    assert is_ablation_experiment_row(
        {
            "evidence_mode": "ablation",
            "preset": "focus_variant",
            "comparison_role": "contrast",
            "component_focus": "semantic_normalizer",
            "analysis_tags": "ablation,noise_control",
        }
    )
    assert not is_ablation_experiment_row(
        {
            "evidence_mode": "comparison",
            "preset": "focus_variant",
            "comparison_role": "contrast",
            "component_focus": "semantic_normalizer",
            "analysis_tags": "comparison,scope",
        }
    )


def test_canonical_comparison_role_normalizes_targeted_to_contrast() -> None:
    assert canonical_comparison_role({"comparison_role": "targeted"}) == "contrast"
    assert canonical_comparison_role({"comparison_role": "contrast"}) == "contrast"
    assert canonical_comparison_role({"comparison_role": "baseline"}) == "baseline"


def test_is_comparison_and_ablation_rows_accept_targeted_as_contrast_semantics() -> None:
    assert is_ablation_experiment_row(
        {
            "evidence_mode": "ablation",
            "comparison_role": "targeted",
            "component_focus": "semantic_normalizer",
            "analysis_tags": "ablation,guided",
        }
    )


def test_experiment_row_accessors_prefer_structured_variant_identity() -> None:
    row = {
        "target_suite": "core",
        "comparison_group": "module_ablation",
        "matrix_id": "module_ablation",
        "variant_id": "focus_variant",
        "variant_title": "Focus Variant",
        "preset": "legacy_preset_name",
    }

    assert experiment_row_group_id(row) == ("core", "module_ablation", "module_ablation")
    assert experiment_row_variant_id(row) == "focus_variant"
    assert experiment_row_variant_label(row) == "Focus Variant"
    assert experiment_row_variant_key(row) == (
        "core",
        "module_ablation",
        "module_ablation",
        "focus_variant",
        "legacy_preset_name",
    )


def test_reference_and_contrast_helpers_keep_canonical_semantics() -> None:
    reference_row = {
        "target_suite": "seeded_join",
        "comparison_group": "seeded_sensitivity",
        "matrix_id": "seeded_sensitivity",
        "variant_id": "baseline",
        "preset": "baseline",
        "comparison_role": "baseline",
    }
    contrast_row = {
        "target_suite": "seeded_join",
        "comparison_group": "seeded_sensitivity",
        "matrix_id": "seeded_sensitivity",
        "variant_id": "join_focus",
        "preset": "join_focus",
        "comparison_role": "targeted",
    }

    assert is_reference_variant(reference_row) is True
    assert is_reference_experiment_row(reference_row) is True
    assert is_contrast_variant(contrast_row) is True
    assert is_contrast_experiment_row(contrast_row) is True
    assert reference_row_for_group([contrast_row, reference_row]) == reference_row
    assert contrast_variant_id_for_suite("seeded_join") == "guided_join"
