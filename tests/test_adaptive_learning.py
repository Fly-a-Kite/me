from datadiff.adaptive_learning import (
    AdaptiveLearningState,
    ContinualPriorityMemory,
    ContextualBandit,
    ExplorationMemory,
    OnlineRewardModel,
    VersionFeedbackMemory,
    context_features_from_mapping,
    continual_learning_summary,
    continual_priority_seed,
)


def test_contextual_bandit_prefers_rewarded_action_after_feedback():
    bandit = ContextualBandit(exploration_weight=0.0, uncertainty_weight=0.0)

    bandit.record("profile_a", scope="generator_profile", context_features=["target:core"], reward=3.0)
    bandit.record("profile_b", scope="generator_profile", context_features=["target:core"], reward=-1.0)

    ranked = bandit.rank(["profile_b", "profile_a"], scope="generator_profile", context_features=["target:core"])

    assert ranked[0]["action_id"] == "profile_a"
    assert ranked[0]["reward_signal"] > ranked[1]["reward_signal"]


def test_contextual_bandit_self_calibrates_top_level_component_scales():
    bandit = ContextualBandit(
        exploration_weight=0.0,
        model_weight=0.0,
        uncertainty_weight=0.0,
        version_weight=0.0,
        continual_priority_weight=0.0,
        active_learning_weight=0.0,
    )

    bandit.record("profile_a", scope="generator_profile", context_features=["target:core"], reward=3.0)
    bandit.record("profile_a", scope="generator_profile", context_features=["target:core"], reward=3.0)

    row = bandit.score_action("profile_a", scope="generator_profile", context_features=["target:core"])
    restored = ContextualBandit.from_state_dict(bandit.to_state_dict())

    assert bandit.weight_calibrator.total_updates == 2
    assert bandit.weight_calibrator.scale("reward_signal") > 1.0
    assert row["calibrated_component_scales"]["reward_signal"] > 1.0
    assert restored.weight_calibrator.scale("reward_signal") == bandit.weight_calibrator.scale("reward_signal")


def test_contextual_bandit_downranks_high_runtime_cost_action():
    bandit = ContextualBandit(exploration_weight=0.0, uncertainty_weight=0.0, model_weight=0.0)

    bandit.record("lean", scope="generator_profile", context_features=["target:core"], reward=2.0)
    bandit.record(
        "costly",
        scope="generator_profile",
        context_features=["target:core"],
        reward=2.0,
        runtime_cost=2.0,
    )

    restored = ContextualBandit.from_state_dict(bandit.to_state_dict())
    ranked = restored.rank(["costly", "lean"], scope="generator_profile", context_features=["target:core"])

    assert ranked[0]["action_id"] == "lean"
    assert ranked[1]["health_penalty"] > ranked[0]["health_penalty"]
    assert restored.arms["costly"].runtime_cost_total == 2.0


def test_online_reward_model_uncertainty_is_higher_for_unseen_features():
    model = OnlineRewardModel()
    seen_features = ("profile:guided", "semantic_family:null")
    unseen_features = ("profile:fresh", "semantic_family:cast")

    for _ in range(5):
        model.update(seen_features, 2.0)

    assert model.uncertainty(unseen_features) > model.uncertainty(seen_features)
    assert model.predict(seen_features) > 0.0


def test_online_reward_model_round_trips_explicit_zero_hyperparameters():
    model = OnlineRewardModel(learning_rate=0.0, l2=0.0, max_abs_reward=0.0)
    restored = OnlineRewardModel.from_state_dict(model.to_state_dict())

    assert restored.learning_rate == 0.0
    assert restored.l2 == 0.0
    assert restored.max_abs_reward == 0.0


def test_contextual_bandit_can_ablate_online_reward_model_without_losing_arm_stats():
    bandit = ContextualBandit(exploration_weight=0.0)

    error = bandit.record(
        "profile_a",
        scope="generator_profile",
        context_features=["target_suite:core"],
        reward=3.0,
        enable_reward_model=False,
    )
    row = bandit.score_action(
        "profile_a",
        scope="generator_profile",
        context_features=["target_suite:core"],
        enable_reward_model=False,
    )

    assert error == 0.0
    assert bandit.arms["profile_a"].pulls == 1
    assert bandit.arms["profile_a"].total_reward == 3.0
    assert bandit.reward_model.total_updates == 0
    assert row["model_prediction"] == 0.0
    assert row["uncertainty"] == 0.0


def test_version_feedback_memory_transfers_signal_to_new_version():
    memory = VersionFeedbackMemory()
    memory.record(
        scope="semantic_objective",
        action_id="semantic_family:null",
        version_id="v1",
        reward=3.0,
    )
    memory.record(
        scope="semantic_objective",
        action_id="semantic_family:null",
        version_id="v1",
        reward=3.0,
    )

    assert memory.transfer_signal(
        scope="semantic_objective",
        action_id="semantic_family:null",
        version_id="v2",
    ) > 0.0


def test_exploration_memory_prioritizes_cold_high_potential_regions():
    memory = ExplorationMemory()
    for _ in range(8):
        memory.record(
            scope="semantic_objective",
            action_id="partition",
            context_features=["target_suite:core", "semantic_family:null"],
            version_id="v1",
            reward=1.0,
        )

    cold_bonus = memory.exploration_bonus(
        scope="semantic_objective",
        action_id="materialization",
        context_features=["target_suite:new_backend", "semantic_family:cast"],
        version_id="v2",
    )
    warm_bonus = memory.exploration_bonus(
        scope="semantic_objective",
        action_id="partition",
        context_features=["target_suite:core", "semantic_family:null"],
        version_id="v1",
    )

    restored = ExplorationMemory.from_state_dict(memory.to_state_dict())

    assert cold_bonus > warm_bonus
    assert restored.total_records == 8
    assert restored.exploration_bonus(
        scope="semantic_objective",
        action_id="materialization",
        context_features=["target_suite:new_backend", "semantic_family:cast"],
        version_id="v2",
    ) == cold_bonus


def test_contextual_bandit_exposes_active_learning_bonus_for_unexplored_actions():
    bandit = ContextualBandit(
        exploration_weight=0.0,
        model_weight=0.0,
        uncertainty_weight=0.0,
        version_weight=0.0,
        active_learning_weight=1.0,
    )
    memory = ExplorationMemory()
    for _ in range(6):
        memory.record(
            scope="mutation_operator",
            action_id="known_operator",
            context_features=["target_suite:core"],
            reward=0.5,
            version_id="v1",
        )

    ranked = bandit.rank(
        ["known_operator", "fresh_operator"],
        scope="mutation_operator",
        context_features=["target_suite:core"],
        version_id="v2",
        exploration_memory=memory,
    )

    assert ranked[0]["action_id"] == "fresh_operator"
    assert ranked[0]["exploration_bonus"] > ranked[1]["exploration_bonus"]


def test_contextual_bandit_uses_continual_priority_memory_for_cold_start():
    bandit = ContextualBandit(
        exploration_weight=0.0,
        model_weight=0.0,
        uncertainty_weight=0.0,
        version_weight=0.0,
        continual_priority_weight=1.0,
        active_learning_weight=0.0,
    )
    memory = ContinualPriorityMemory()
    memory.ingest_ledger(
        {
            "families": [
                {"family": "null_semantics@engine", "status": "regression"},
                {"family": "cast_semantics@engine", "status": "fixed"},
            ]
        }
    )

    ranked = bandit.rank(
        ["semantic_family:cast_semantics", "semantic_family:null_semantics"],
        scope="semantic_objective",
        context_features=["target_suite:engine"],
        continual_priority_memory=memory,
    )

    assert ranked[0]["action_id"] == "semantic_family:null_semantics"
    assert ranked[0]["continual_priority_signal"] > ranked[1]["continual_priority_signal"]


def test_continual_priority_memory_downweights_unhealthy_cross_version_families():
    memory = ContinualPriorityMemory()
    summary = memory.ingest_ledger(
        {
            "families": [
                {
                    "family": "healthy_semantics@engine",
                    "status": "regression",
                    "health_observations": [
                        {
                            "invalid_rate": 0.0,
                            "fallback_rate": 0.0,
                            "false_positive_rate": 0.0,
                            "runtime_ms_per_case": 5.0,
                        }
                    ],
                },
                {
                    "family": "noisy_semantics@engine",
                    "status": "regression",
                    "health_observations": [
                        {
                            "invalid_rate": 1.0,
                            "fallback_rate": 1.0,
                            "false_positive_rate": 1.0,
                            "runtime_ms_per_case": 1200.0,
                        }
                    ],
                },
            ]
        }
    )

    healthy = memory.priority_signal(
        scope="semantic_objective",
        action_id="semantic_family:healthy_semantics",
        context_features=["target_suite:engine"],
    )
    noisy = memory.priority_signal(
        scope="semantic_objective",
        action_id="semantic_family:noisy_semantics",
        context_features=["target_suite:engine"],
    )

    assert summary["imported_health_feedback_count"] == 2
    assert summary["health_feedback_feature_count"] > 0
    assert healthy > noisy
    assert memory.to_state_dict()["family_health_penalties"]["noisy_semantics@engine"] > 0.0


def test_contextual_bandit_can_ablate_continual_learning_signals_and_penalties():
    bandit = ContextualBandit(
        exploration_weight=0.0,
        model_weight=0.0,
        uncertainty_weight=0.0,
        version_weight=1.0,
        continual_priority_weight=1.0,
        active_learning_weight=0.0,
    )
    version_memory = VersionFeedbackMemory()
    version_memory.record(
        scope="semantic_objective",
        action_id="semantic_family:null_semantics",
        version_id="v1",
        reward=3.0,
        runtime_cost=2.0,
        preflight_valid=False,
        false_positive=True,
    )
    continual_memory = ContinualPriorityMemory()
    continual_memory.ingest_ledger(
        {"families": [{"family": "null_semantics@engine", "status": "regression"}]}
    )

    enabled = bandit.score_action(
        "semantic_family:null_semantics",
        scope="semantic_objective",
        context_features=["target_suite:engine"],
        version_id="v1",
        version_memory=version_memory,
        continual_priority_memory=continual_memory,
        enable_continual_learning=True,
    )
    disabled = bandit.score_action(
        "semantic_family:null_semantics",
        scope="semantic_objective",
        context_features=["target_suite:engine"],
        version_id="v1",
        version_memory=version_memory,
        continual_priority_memory=continual_memory,
        enable_continual_learning=False,
    )

    assert enabled["version_signal"] > 0.0
    assert enabled["continual_priority_signal"] > 0.0
    assert enabled["health_penalty"] > 0.0
    assert disabled["version_signal"] == 0.0
    assert disabled["continual_priority_signal"] == 0.0
    assert disabled["health_penalty"] == 0.0


def test_adaptive_learning_state_can_disable_active_learning_bonus():
    state = AdaptiveLearningState()
    state.record_outcome(
        "generator_profile",
        "known_profile",
        context_features=["target_suite:core"],
        version_id="v1",
        reward=1.0,
    )

    enabled = state.score_action(
        "generator_profile",
        "fresh_profile",
        context_features=["target_suite:fresh"],
        version_id="v2",
        enable_active_learning=True,
    )
    disabled = state.score_action(
        "generator_profile",
        "fresh_profile",
        context_features=["target_suite:fresh"],
        version_id="v2",
        enable_active_learning=False,
    )

    assert enabled["exploration_bonus"] > 0.0
    assert disabled["exploration_bonus"] == 0.0


def test_adaptive_learning_state_health_summary_reports_compact_diagnostics():
    state = AdaptiveLearningState()
    state.record_outcome(
        "generator_profile",
        "discovery",
        context_features=["target_suite:core", "semantic_family:null"],
        version_id="v1",
        reward=2.0,
        runtime_cost=0.5,
    )
    state.record_outcome(
        "mutation_operator",
        "append_filter",
        context_features=["target_suite:core"],
        version_id="v1",
        reward=-1.0,
        preflight_valid=False,
        false_positive=True,
    )

    summary = state.health_summary()

    assert summary["schema_version"] == "adaptive-learning-health-v1"
    assert summary["bandit_count"] == 2
    assert summary["arm_count"] == 2
    assert summary["total_pulls"] == 2
    assert summary["max_health_penalty"] > 0.0
    assert summary["exploration_memory"]["total_records"] == 2
    assert summary["version_memory_key_count"] == 2


def test_adaptive_learning_state_persists_scoped_actions_and_version_memory():
    state = AdaptiveLearningState()
    state.record_outcome(
        "mutation_operator",
        "append_range_filter",
        context_features=["semantic_family:filtering"],
        version_id="v1",
        reward=2.0,
    )

    restored = AdaptiveLearningState.from_state_dict(state.to_state_dict())
    ranked = restored.rank(
        "mutation_operator",
        ["append_range_filter", "append_sort_probe"],
        context_features=["semantic_family:filtering"],
        version_id="v2",
    )

    assert "mutation_operator" in restored.bandits
    assert ranked[0]["action_id"] == "append_range_filter"
    assert ranked[0]["version_signal"] > 0.0
    assert ranked[0]["exploration_bonus"] >= 0.0
    assert restored.exploration_memory.total_records == 1


def test_backend_pair_scope_learns_rewarded_pair_priority():
    state = AdaptiveLearningState()
    context = ["fp_type_mix:num1", "mismatch:value"]
    for _ in range(4):
        state.record_outcome(
            "backend_pair",
            "left|right",
            context_features=context,
            version_id="latest->fixed",
            reward=2.0,
        )
        state.record_outcome(
            "backend_pair",
            "left|third",
            context_features=context,
            version_id="latest->fixed",
            reward=-0.1,
        )

    ranked = state.rank_top(
        "backend_pair",
        ["left|third", "left|right"],
        limit=2,
        context_features=context,
        version_id="latest->fixed",
    )

    assert ranked[0]["action_id"] == "left|right"
    assert ranked[0]["mean_reward"] > ranked[1]["mean_reward"]
    assert "backend_pair" in state.to_state_dict()["bandits"]


def test_adaptive_learning_state_imports_and_persists_continual_priority_seed():
    state = AdaptiveLearningState()
    seed = continual_priority_seed(
        {
            "families": [
                {"family": "partition_order@engine", "status": "new"},
            ]
        }
    )
    state.continual_priority_memory = ContinualPriorityMemory.from_state_dict(
        seed["continual_priority_memory"]
    )

    restored = AdaptiveLearningState.from_state_dict(state.to_state_dict())
    row = restored.score_action(
        "semantic_objective",
        "partition_order",
        context_features=["target_suite:engine"],
    )

    assert row["continual_priority_signal"] > 0.0
    assert restored.health_summary()["continual_priority_memory"]["family_count"] == 1


def test_continual_priority_memory_merges_multiple_imported_seeds():
    first = ContinualPriorityMemory()
    first.ingest_ledger(
        {
            "families": [
                {"family": "null_semantics@engine", "status": "regression"},
            ]
        }
    )
    second = ContinualPriorityMemory()
    second.ingest_ledger(
        {
            "families": [
                {"family": "cast_semantics@engine", "status": "new"},
            ]
        }
    )

    summary = first.merge(second)

    assert summary["family_count"] == 2
    assert summary["imported_ledger_count"] == 2
    assert first.family_priorities["null_semantics@engine"] == 1.0
    assert first.family_priorities["cast_semantics@engine"] == 0.9
    assert first.priority_signal(
        scope="semantic_objective",
        action_id="semantic_family:cast_semantics",
        context_features=["target_suite:engine"],
    ) > 0.0


def test_context_features_from_mapping_extracts_transferable_action_context():
    features = context_features_from_mapping(
        {
            "target_suite": "core",
            "preset": "guided",
            "generator_profile": "discovery",
            "target_version": "v2",
            "fixed_version": "v1",
            "backends": ["duckdb", "sqlite"],
            "semantic_focus_families": ["null_semantics"],
            "semantic_focus_signals": "order_sensitive",
            "guidance_targets": ["semantic_family:cast_semantics"],
            "enable_metamorphic_oracle": True,
            "enable_feedback": True,
            "enable_local_source_scheduler": True,
            "metamorphic_variant_limit": 4,
        }
    )

    assert "target_suite:core" in features
    assert "generator_profile:discovery" in features
    assert "semantic_focus_families:null_semantics" in features
    assert "semantic_focus_signals:order_sensitive" in features
    assert "guidance_targets:semantic_family_cast_semantics" in features
    assert "metamorphic:enabled" in features
    assert "backend_count:few" in features


def test_context_features_ignore_metamorphic_limit_when_oracle_is_disabled():
    disabled = context_features_from_mapping(
        {
            "target_suite": "core",
            "metamorphic_variant_limit": 8,
            "enable_metamorphic_oracle": False,
        }
    )
    enabled = context_features_from_mapping(
        {
            "target_suite": "core",
            "effective_metamorphic_variant_limit": 8,
            "enable_metamorphic_oracle": True,
        }
    )

    assert not any(feature.startswith("metamorphic_limit:") for feature in disabled)
    assert "metamorphic_limit:large" in enabled


def test_continual_learning_summary_prioritizes_regressions_and_new_families():
    summary = continual_learning_summary(
        [
            {"family": "fixed@engine", "status": "fixed"},
            {"family": "new@engine", "status": "new"},
            {"family": "regressed@engine", "status": "regression"},
            {"family": "persistent@engine", "status": "persistent"},
        ],
        version_order=["v1", "v2"],
    )

    priorities = {row["family"]: row["priority"] for row in summary["family_priorities"]}

    assert summary["schema_version"] == "cross-version-continual-learning-v1"
    assert priorities["regressed@engine"] > priorities["new@engine"] > priorities["persistent@engine"]
    assert priorities["persistent@engine"] > priorities["fixed@engine"]
    assert summary["adaptive_learning_seed"]["schema_version"] == "continual-priority-seed-v1"
    assert summary["adaptive_learning_seed"]["summary"]["family_count"] == 4
