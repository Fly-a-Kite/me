from __future__ import annotations

from datadiff import cli
from datadiff.config import ExperimentConfig
from datadiff.experiment_catalog import (
    FINAL_EXPERIMENT_MATRICES,
    FINAL_LIVE_DISCOVERY_MATRIX,
    FINAL_MODULE_ABLATION_MATRIX,
    FINAL_VALIDATION_MATRIX,
    build_final_experiment_variant_config,
)
from datadiff.preset_catalog import build_catalog_preset
from datadiff.preset_catalog import build_experiment_config


def test_final_experiment_variants_roundtrip_to_real_configs() -> None:
    for matrix in FINAL_EXPERIMENT_MATRICES:
        for variant in matrix.variants:
            config = build_final_experiment_variant_config(matrix.id, preset=variant.preset)
            assert isinstance(config, ExperimentConfig)
            assert config.to_dict() == cli._preset_config(variant.preset).to_dict()


def test_final_validation_overlay_variants_resolve_via_base_preset_and_overlays() -> None:
    common_api = build_final_experiment_variant_config(
        FINAL_VALIDATION_MATRIX.id,
        preset="live_common_api_workflow_metamorphic",
    )
    deep_organic = build_final_experiment_variant_config(
        FINAL_VALIDATION_MATRIX.id,
        preset="live_deep_organic_metamorphic",
    )

    assert common_api.to_dict() == build_experiment_config(
        "live_common_api_workflow",
        ("enable_metamorphic_oracle", "candidate_pool_8", "metamorphic_variant_limit_6"),
    ).to_dict()
    assert deep_organic.to_dict() == build_experiment_config(
        "live_deep_organic",
        ("enable_metamorphic_oracle", "candidate_pool_10", "metamorphic_variant_limit_8"),
    ).to_dict()


def test_final_live_variants_use_catalog_owned_live_target_registry() -> None:
    for preset in [variant.preset for variant in FINAL_LIVE_DISCOVERY_MATRIX.variants]:
        config = build_final_experiment_variant_config(FINAL_LIVE_DISCOVERY_MATRIX.id, preset=preset)
        expected = cli.LIVE_PRESET_TARGETS_BY_NAME[preset]
        assert config.guidance_targets == list(expected)


def test_final_ablation_variants_preserve_structured_overlay_identity() -> None:
    baseline = build_final_experiment_variant_config(FINAL_MODULE_ABLATION_MATRIX.id, preset="baseline")
    no_feedback = build_final_experiment_variant_config(FINAL_MODULE_ABLATION_MATRIX.id, preset="no_feedback")
    reducer = build_final_experiment_variant_config(FINAL_MODULE_ABLATION_MATRIX.id, preset="reducer")

    assert baseline.to_dict() == build_catalog_preset("baseline").to_dict()
    assert no_feedback.enable_feedback is False
    assert reducer.enable_reducer is True


def test_catalog_config_builders_no_longer_require_cli_owned_live_targets() -> None:
    assert build_catalog_preset("live_cross_family").to_dict() == cli._preset_config("live_cross_family").to_dict()
    assert build_experiment_config(
        "live_deep_organic",
        ("enable_metamorphic_oracle", "candidate_pool_10", "metamorphic_variant_limit_8"),
    ).to_dict() == cli._preset_config("live_deep_organic_metamorphic").to_dict()
