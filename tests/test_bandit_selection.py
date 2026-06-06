from types import SimpleNamespace

from datadiff.bandit_selection import (
    _backend_pair_context_features,
    _backend_pair_pool,
    _backend_pair_reward,
    _case_learning_context_features,
    _config_for_version_pair,
    _config_payload_for_version_pair,
    _generator_profile_context_features,
    _generator_profile_pool,
    _generator_profile_reward,
    _metamorphic_relation_order_from_selection,
    _normalize_backend_pair_id,
    _record_backend_pair_feedback,
    _runtime_cost_signal,
    _select_adaptive_action,
    _select_generator_profile,
    _semantic_objective_pool,
    _version_pair_context_features,
    _version_pair_id,
    _version_pair_pool,
)
from datadiff.config import ExperimentConfig
from datadiff.datagen import generate_case
from datadiff.exploration_objectives import EXPLORATION_OBJECTIVE_PREFIX, ExplorationObjectiveRule
from datadiff.finding_outcomes import row_reward_signals
from datadiff.operation_combo import describe_operation_combo
from datadiff import runner as runner_module


def test_version_pair_pool_includes_default_and_deduplicates_candidates():
    config = ExperimentConfig(
        target_version="latest",
        fixed_version="fixed",
        version_pair_pool=["latest->preview", "latest->fixed", "latest->preview"],
    )

    pool, metadata = _version_pair_pool(config)

    assert _version_pair_id(config) == "latest->fixed"
    assert pool == ("latest->preview", "latest->fixed")
    assert metadata["default_pair"] == "latest->fixed"
    assert metadata["selected"] == ["latest->preview", "latest->fixed"]


def test_config_for_version_pair_and_payload_are_copy_on_write():
    config = ExperimentConfig(target_version="latest", fixed_version="fixed")
    same = _config_for_version_pair(config, "latest->fixed")
    changed = _config_for_version_pair(config, "latest->preview")
    payload = _config_payload_for_version_pair({"target_version": "latest"}, "latest->preview")

    assert same is config
    assert changed is not config
    assert changed.target_version == "latest"
    assert changed.fixed_version == "preview"
    assert config.fixed_version == "fixed"
    assert payload["fixed_version"] == "preview"
    assert payload["selected_version_pair"] == "latest->preview"


def test_context_features_collect_case_guidance_version_and_backend_signals():
    case = generate_case(9, profile="common")
    config = ExperimentConfig(
        oracle_mode="both",
        guidance_strategy="guided",
        semantic_focus_families=["window"],
        semantic_focus_signals=["null_semantics"],
        target_version="latest",
        fixed_version="fixed",
    )
    operation_combo = describe_operation_combo(case.program.operations)
    guidance_row = {
        "matched_targets": ["join"],
        "features": ["semantic_family:window", f"{EXPLORATION_OBJECTIVE_PREFIX}null_boundary"],
    }

    case_features = _case_learning_context_features(
        case,
        config,
        backends=["pandas", "duckdb"],
        target_capabilities=["op:join"],
        operation_combo=operation_combo,
        guidance_row=guidance_row,
    )
    version_features = _version_pair_context_features(
        config=config,
        backends=["pandas", "duckdb"],
        target_capabilities=["op:join"],
    )

    assert "oracle_mode:both" in case_features
    assert "guidance_strategy:guided" in case_features
    assert "backend_count:few" in case_features
    assert "backend:pandas" in case_features
    assert "capability:op:join" in case_features
    assert "matched_target:join" in case_features
    assert "version_pair:present" in case_features
    assert "semantic_family:window" in case_features
    assert "semantic_signal:null_semantics" in version_features


def test_backend_pair_pool_context_and_reward_helpers_are_canonical():
    pool = _backend_pair_pool(["right", "left", "right", "third"])
    context = _backend_pair_context_features(
        case_learning_context=("root:join",),
        case_fingerprint={"feature_tokens": ["fingerprint:a"]},
        disagreement_descriptor={
            "feature_tokens": ["disagree_pair:left|right", "shape:mismatch"],
            "mismatch_class": "row_count",
            "primary_root_cause": "join",
        },
        selected_version_pair="latest->fixed",
        target_capabilities=["op:join"],
    )

    assert pool == ("left|right", "left|third", "right|third")
    assert _normalize_backend_pair_id("right|left") == "left|right"
    assert _normalize_backend_pair_id("left|left") == ""
    assert "root:join" in context
    assert "fingerprint:a" in context
    assert "shape:mismatch" in context
    assert "disagree_pair:left|right" not in context
    assert "mismatch:row_count" in context
    assert "version_pair:latest->fixed" in context
    assert _backend_pair_reward(
        "left|right",
        disagrees=True,
        reward_signals={"candidate_bug": True, "rewardable_semantic_divergence": True},
        priority_pairs={"left|right"},
    ) == 4.0
    assert _backend_pair_reward(
        "left|third",
        disagrees=False,
        reward_signals={"false_positive": True},
        priority_pairs={"left|third"},
    ) == -0.45
    assert _runtime_cost_signal({"duration_ms": 2500.0}) == 2.0


def test_generator_profile_reward_filters_non_rewardable_raw_signal():
    false_positive_row = {
        "signal_new_behavior": True,
        "duration_ms": 10.0,
        "preflight": {"valid": True, "fallback_used": False},
        "findings": [
            {
                "triage_verdict": "normalizer_false_positive",
                "root_cause": "order_only_normalization_mismatch",
                "false_positive": True,
            }
        ],
    }
    source_issue_row = {
        "signal_new_behavior": True,
        "preflight": {"valid": True, "fallback_used": False},
        "findings": [
            {
                "triage_verdict": "candidate_implementation_bug",
                "root_cause": "csv_long_numeric_roundtrip",
                "suspicious_backends": ["duckdb"],
                "source_issue": "duckdb/duckdb#12345",
            }
        ],
    }
    candidate_row = {
        "signal_new_behavior": True,
        "preflight": {"valid": True, "fallback_used": False},
        "findings": [
            {
                "triage_verdict": "candidate_implementation_bug",
                "root_cause": "topk_filter_pushdown",
                "suspicious_backends": ["datafusion"],
                "discovery_origin": "organic",
            }
        ],
    }

    assert _generator_profile_reward(
        false_positive_row,
        reward_signals=row_reward_signals(false_positive_row),
    ) < 0.0
    assert _generator_profile_reward(
        source_issue_row,
        reward_signals=row_reward_signals(source_issue_row),
    ) <= 0.0
    assert _generator_profile_reward(
        candidate_row,
        reward_signals=row_reward_signals(candidate_row),
    ) > 3.0


def test_record_backend_pair_feedback_records_all_pool_pairs_and_rewards_disagreements():
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    class FakeLearning:
        def record_outcome(self, *args, **kwargs):
            calls.append((args, kwargs))

    feedback = SimpleNamespace(adaptive_learning=FakeLearning())
    row = {
        "duration_ms": 900.0,
        "preflight": {"valid": False, "fallback_used": True},
        "backend_pair_selection": {
            "strategy": "contextual_bandit",
            "scope": "backend_pair",
            "action_pool": ["right|left", "left|third", "bad"],
            "priority": ["left|right"],
        },
        "disagreement_descriptor": {
            "pair_disagrees": [
                {"left": "right", "right": "left", "disagrees": True},
                {"left": "left", "right": "third", "disagrees": False},
            ]
        },
    }

    result = _record_backend_pair_feedback(
        feedback,
        row,
        context_features=("root:join",),
        version_id="latest->fixed",
        reward_signals={"candidate_bug": True},
    )

    assert result["recorded"] == 2
    assert result["disagree_pairs"] == ["left|right"]
    assert result["rewarded_pairs"] == [{"pair": "left|right", "reward": 3.25}]
    assert [call[0] for call in calls] == [("backend_pair", "left|right"), ("backend_pair", "left|third")]
    assert calls[0][1]["runtime_cost"] == 0.45
    assert calls[0][1]["preflight_valid"] is False
    assert calls[0][1]["fallback_used"] is True


def test_generator_profile_pool_context_and_selection_delegate_to_dense_learning():
    class FakeLearning:
        def __init__(self):
            self.bandits = {
                "generator_profile": SimpleNamespace(
                    arms={
                        "common": SimpleNamespace(pulls=1),
                        "discovery_fresh": SimpleNamespace(pulls=1),
                    }
                )
            }

        def choose_dense(self, scope, action_ids, **kwargs):
            return SimpleNamespace(action_id="discovery_fresh")

        def rank_top(self, scope, action_ids, *, limit, **kwargs):
            return [
                {"action_id": "discovery_fresh", "score": 2.0},
                {"action_id": "common", "score": 1.0},
            ]

    config = ExperimentConfig(
        generator_profile="common",
        generator_profile_pool=["common", "discovery_fresh", "common"],
        semantic_focus_families=["join"],
        guidance_targets=["join"],
    )
    pool, metadata = _generator_profile_pool(config, target_capabilities=[])
    features = _generator_profile_context_features(
        config=config,
        backends=["pandas", "duckdb"],
        target_specs=[{"family": "dataframe", "layer": "python_dataframe"}],
        target_capabilities=["op:join"],
        candidate_pool=5,
    )
    profile, selection = _select_generator_profile(
        SimpleNamespace(adaptive_learning=FakeLearning()),
        pool,
        context_features=features,
        learning_weight=1.0,
        pool_metadata=metadata,
    )

    assert pool == ("common", "discovery_fresh")
    assert metadata["selected"] == ["common", "discovery_fresh"]
    assert "candidate_pool:medium" in features
    assert "target_family:dataframe" in features
    assert "capability:op:join" in features
    assert profile == "discovery_fresh"
    assert selection["profile"] == "discovery_fresh"
    assert selection["profile_pool_metadata"] == metadata


def test_semantic_objective_pool_and_metamorphic_order_dedupe_preserve_priority():
    config = ExperimentConfig(
        exploration_objective_rules=[
            ExplorationObjectiveRule(
                objective="window_frame",
                exact_features=frozenset({"semantic_family:window"}),
            )
        ]
    )
    pool = _semantic_objective_pool(
        (
            "semantic_family:window",
            f"{EXPLORATION_OBJECTIVE_PREFIX}null_boundary",
            f"{EXPLORATION_OBJECTIVE_PREFIX}null_boundary",
        ),
        config,
        {"matched_targets": [f"{EXPLORATION_OBJECTIVE_PREFIX}sort_boundary"]},
    )
    order = _metamorphic_relation_order_from_selection(
        "row_permutation",
        configured_order=["input_partition_union_all", "row_permutation"],
    )

    assert pool[:3] == (
        f"{EXPLORATION_OBJECTIVE_PREFIX}null_boundary",
        f"{EXPLORATION_OBJECTIVE_PREFIX}sort_boundary",
        f"{EXPLORATION_OBJECTIVE_PREFIX}window_frame",
    )
    assert f"{EXPLORATION_OBJECTIVE_PREFIX}boundary_depth" in pool
    assert order == ["row_permutation", "input_partition_union_all"]


def test_metamorphic_order_rotates_fixed_default_but_preserves_explicit_priority():
    rotated = _metamorphic_relation_order_from_selection(
        "first",
        configured_order=[],
        relation_pool=("first", "second", "third"),
        rotation_seed=1,
        selection_strategy="fixed",
    )
    configured = _metamorphic_relation_order_from_selection(
        "first",
        configured_order=["third"],
        relation_pool=("first", "second", "third"),
        rotation_seed=1,
        selection_strategy="fixed",
    )
    learned = _metamorphic_relation_order_from_selection(
        "second",
        configured_order=["third"],
        relation_pool=("first", "second", "third"),
        rotation_seed=1,
        selection_strategy="contextual_bandit",
    )

    assert rotated == ["second", "third", "first"]
    assert configured == ["third", "first", "second"]
    assert learned == ["second", "third"]


def test_runner_reexports_selection_helpers_for_compatibility():
    assert runner_module._select_adaptive_action is _select_adaptive_action
    assert runner_module._version_pair_pool is _version_pair_pool
    assert runner_module._record_backend_pair_feedback is _record_backend_pair_feedback
