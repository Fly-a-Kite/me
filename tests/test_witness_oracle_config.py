from argparse import Namespace

from datadiff.cli_config import config_from_args
from datadiff.preset_catalog import build_experiment_config


def _args(**overrides):
    payload = {
        "disable_type_aware_generation": False,
        "disable_normalizer": False,
        "disable_differential_oracle": False,
        "enable_metamorphic_oracle": False,
        "enable_witness_oracle": False,
        "disable_feedback": False,
        "enable_replay_bug": False,
        "enable_reducer": False,
        "disable_artifact": False,
        "disable_parallel_backend_execution": False,
        "disable_preflight_validation": False,
        "disable_preflight_repair": False,
        "persist_feedback_corpus": False,
        "feedback_persist_limit": 4096,
        "enable_local_source_scheduler": False,
        "local_source_exploration_weight": 0.5,
        "disable_adaptive_components": "",
        "no_compress_run_log": False,
        "artifact_limit": None,
        "profile": "common",
        "profile_pool": "",
        "version_pair_pool": "",
        "profile_learning_weight": 0.0,
        "semantic_objective_learning_weight": 0.0,
        "metamorphic_relation_learning_weight": 0.0,
        "version_pair_learning_weight": 0.0,
        "backend_pair_learning_weight": 0.0,
        "backend_pair_priority_limit": 3,
        "targets": "",
        "semantic_focus_families": "",
        "semantic_focus_signals": "",
        "exploration_objective_rules": "",
        "disable_family_saturation": False,
        "family_saturation_threshold": 8,
        "family_saturation_penalty": 1.25,
        "saturated_family_reward": 0.02,
        "known_saturated_bug_families": "",
        "replay_bug_source_issues": "",
        "issue_replay_saturation_threshold": 1,
        "issue_replay_saturation_penalty": 1.0,
        "issue_replay_global_saturation_threshold": 4,
        "issue_replay_global_saturation_penalty": 1.5,
        "issue_inspired_source_saturation_threshold": 3,
        "issue_inspired_source_saturation_penalty": 1.25,
        "candidate_recheck_count": 0,
        "metamorphic_variant_limit": 4,
        "metamorphic_relation_order": "",
        "log_level": "compact",
        "target_version": "",
        "fixed_version": "",
        "strategy_snapshot": "",
        "strategy_learning": "",
        "freeze_strategy_snapshot": False,
    }
    payload.update(overrides)
    return Namespace(**payload)


def test_cli_config_enables_witness_oracle_only_when_requested():
    default_config = config_from_args(_args())
    enabled_config = config_from_args(_args(enable_witness_oracle=True))

    assert default_config.enable_witness_oracle is False
    assert enabled_config.enable_witness_oracle is True
    assert enabled_config.oracle.enable_witness is True


def test_preset_overlay_enables_witness_oracle():
    config = build_experiment_config("baseline", overlays=("enable_witness_oracle",))

    assert config.enable_witness_oracle is True
    assert config.enable_differential_oracle is True
    assert config.enable_metamorphic_oracle is False
