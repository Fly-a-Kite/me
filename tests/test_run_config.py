from datadiff.config import ExperimentConfig
from datadiff.exploration_objectives import ExplorationObjectiveRule
from datadiff import runner as runner_module
from datadiff.run_config import (
    _config_layer_payload,
    _config_payload_with_effective_guidance_targets,
    _configured_guidance_targets,
    _effective_guidance_targets,
)


def test_configured_guidance_targets_normalizes_structured_targets():
    config = ExperimentConfig(
        guidance_targets=["groupby", "groupby", ""],
        semantic_focus_families=["conditional_semantics", "conditional_semantics"],
        semantic_focus_signals=["left_join_case_when_membership"],
        exploration_objective_rules=[
            ExplorationObjectiveRule(
                objective="adaptive consistency",
                exact_features=frozenset({"op:join"}),
            )
        ],
    )

    targets = _configured_guidance_targets(config)

    assert targets == [
        "groupby",
        "semantic_family:conditional_semantics",
        "semantic_signal:left_join_case_when_membership",
        "exploration_objective:adaptive_consistency",
    ]
    assert _effective_guidance_targets(config) == targets


def test_config_payloads_preserve_flat_and_nested_compatibility():
    config = ExperimentConfig(
        guidance_targets=["topk"],
        semantic_focus_families=["join_membership"],
        log_level="minimal",
    )

    flat = _config_payload_with_effective_guidance_targets(config)
    layered = _config_layer_payload(config)

    assert flat["log_level"] == "minimal"
    assert flat["effective_guidance_targets"] == ["topk", "semantic_family:join_membership"]
    assert layered["logging"]["log_level"] == "minimal"
    assert layered["guidance"]["effective_targets"] == ["topk", "semantic_family:join_membership"]


def test_experiment_config_from_payload_ignores_derived_report_fields():
    config = ExperimentConfig.from_payload(
        {
            "generator_profile": "discovery_fresh",
            "guidance_targets": ["join"],
            "effective_guidance_targets": ["join"],
            "unknown_future_field": True,
        }
    )

    assert config.generator_profile == "discovery_fresh"
    assert config.guidance_targets == ["join"]


def test_runner_reexports_run_config_helpers_for_compatibility():
    assert runner_module._configured_guidance_targets is _configured_guidance_targets
    assert runner_module._config_layer_payload is _config_layer_payload
