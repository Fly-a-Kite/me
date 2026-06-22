import pytest
from pathlib import Path

from datadiff import scheduler as scheduler_module
from datadiff.scheduler import (
    AdaptiveBudgetScheduler,
    AdaptiveScheduleConfig,
    BatchObservation,
    LocalSourceScheduler,
    _batch_reward,
    summarize_batch_run,
)
from datadiff.adaptive_learning import AdaptiveLearningState
from datadiff.util import append_jsonl, dump_json, run_meta_path


def test_local_source_scheduler_prefers_feedback_after_productive_mutation():
    scheduler = LocalSourceScheduler(exploration_weight=0.0)

    assert scheduler.choose_source(feedback_available=False) == "generated"
    assert scheduler.choose_source(feedback_available=True) == "generated"
    scheduler.record_result(
        "generated",
        has_finding=False,
        is_new_behavior=False,
        preflight_valid=True,
        fallback_used=False,
    )

    assert scheduler.choose_source(feedback_available=True) == "feedback_mutation"
    scheduler.record_result(
        "feedback_mutation",
        has_finding=True,
        is_new_behavior=True,
        preflight_valid=True,
        fallback_used=False,
        candidate_bug=True,
    )

    assert scheduler.choose_source(feedback_available=True) == "feedback_mutation"
    snapshot = {row["source"]: row for row in scheduler.snapshot()}
    assert snapshot["generated"]["pulls"] == 1
    assert snapshot["feedback_mutation"]["pulls"] == 1
    assert snapshot["feedback_mutation"]["mean_reward"] > snapshot["generated"]["mean_reward"]


def test_local_source_scheduler_discounts_repeated_candidate_bug_family():
    scheduler = LocalSourceScheduler(exploration_weight=0.0)

    first_reward = scheduler.record_result(
        "generated",
        has_finding=True,
        is_new_behavior=True,
        preflight_valid=True,
        fallback_used=False,
        candidate_bug=True,
        candidate_bug_families=["topk_filter_pushdown@datafusion"],
        candidate_bug_signatures=["sig-a"],
    )
    repeated_family_reward = scheduler.record_result(
        "generated",
        has_finding=True,
        is_new_behavior=True,
        preflight_valid=True,
        fallback_used=False,
        candidate_bug=True,
        candidate_bug_families=["topk_filter_pushdown@datafusion"],
        candidate_bug_signatures=["sig-b"],
    )
    repeated_signature_reward = scheduler.record_result(
        "generated",
        has_finding=True,
        is_new_behavior=False,
        preflight_valid=True,
        fallback_used=False,
        candidate_bug=True,
        candidate_bug_families=["topk_filter_pushdown@datafusion"],
        candidate_bug_signatures=["sig-b"],
    )

    assert first_reward == 3.805036923076923
    assert 0.0 < repeated_signature_reward < repeated_family_reward < first_reward
    snapshot = {row["source"]: row for row in scheduler.snapshot()}
    assert snapshot["generated"]["candidate_bug_family_count"] == 1
    assert snapshot["generated"]["candidate_bug_signature_count"] == 2


def test_local_source_scheduler_downweights_known_saturated_family():
    scheduler = LocalSourceScheduler(
        exploration_weight=0.0,
        known_saturated_bug_families=["topk_filter_pushdown@datafusion"],
    )

    reward = scheduler.record_result(
        "generated",
        has_finding=True,
        is_new_behavior=False,
        preflight_valid=True,
        fallback_used=False,
        candidate_bug=True,
        candidate_bug_families=["topk_filter_pushdown@datafusion"],
        candidate_bug_signatures=["sig-a"],
    )

    assert 0.0 < reward < 0.05


def test_local_source_scheduler_keeps_feedback_mutation_sampling_floor():
    scheduler = LocalSourceScheduler(exploration_weight=0.0, min_feedback_share=0.20)

    for _ in range(10):
        scheduler.record_result(
            "generated",
            has_finding=False,
            is_new_behavior=True,
            preflight_valid=True,
            fallback_used=False,
        )
    scheduler.record_result(
        "feedback_mutation",
        has_finding=False,
        is_new_behavior=False,
        preflight_valid=True,
        fallback_used=False,
    )

    assert scheduler.choose_source(feedback_available=True) == "feedback_mutation"


def test_local_source_scheduler_reward_signal_is_bounded_by_confidence():
    scheduler = LocalSourceScheduler(exploration_weight=0.0)

    scheduler.record_result(
        "generated",
        has_finding=True,
        is_new_behavior=True,
        preflight_valid=True,
        fallback_used=False,
        candidate_bug=True,
        reward_adjustment=100.0,
    )

    snapshot = {row["source"]: row for row in scheduler.snapshot()}
    generated = snapshot["generated"]

    assert generated["mean_reward"] > generated["reward_signal"] > 0.0
    assert generated["reward_signal"] < 2.5


def test_adaptive_budget_scheduler_warmup_then_exploits_high_reward_arm():
    scheduler = AdaptiveBudgetScheduler(
        [
            {"arm_id": "a", "target_suite": "core", "preset": "baseline", "seed": 1},
            {"arm_id": "b", "target_suite": "core_datafusion", "preset": "guided", "seed": 1001},
        ],
        total_cases_budget=12,
        total_duration_budget_s=None,
        config=AdaptiveScheduleConfig(batch_cases=4, warmup_batches=1, exploration_weight=0.0),
    )

    batch_a = scheduler.next_batch()
    assert batch_a.arm_id == "a"
    scheduler.record_result(
        batch_a,
        BatchObservation(
            cases=4,
            elapsed_s=1.0,
            throughput_cases_s=4.0,
            findings=0,
            candidate_bug_cases=0,
            new_behavior_cases=0,
        ),
        next_seed=5,
    )

    batch_b = scheduler.next_batch()
    assert batch_b.arm_id == "b"
    scheduler.record_result(
        batch_b,
        BatchObservation(
            cases=4,
            elapsed_s=1.0,
            throughput_cases_s=4.0,
            findings=2,
            candidate_bug_cases=2,
            candidate_bug_families={"groupby_aggregation@datafusion"},
            new_behavior_cases=1,
        ),
        next_seed=1005,
    )

    next_batch = scheduler.next_batch()
    assert next_batch.arm_id == "b"
    snapshot = {row["arm_id"]: row for row in scheduler.snapshot()}
    assert snapshot["b"]["mean_reward"] > snapshot["a"]["mean_reward"]


def test_adaptive_budget_scheduler_applies_good_turing_exploration_from_closed_loop_state():
    scheduler = AdaptiveBudgetScheduler(
        [
            {"arm_id": "a", "target_suite": "core", "preset": "baseline", "seed": 1},
            {"arm_id": "b", "target_suite": "core_datafusion", "preset": "guided", "seed": 1001},
        ],
        total_cases_budget=8,
        total_duration_budget_s=None,
        config=AdaptiveScheduleConfig(batch_cases=4, warmup_batches=1, exploration_weight=0.5),
    )

    batch = scheduler.next_batch()
    scheduler.record_result(
        batch,
        BatchObservation(
            cases=4,
            elapsed_s=1.0,
            throughput_cases_s=4.0,
            findings=0,
            candidate_bug_cases=0,
        ),
        next_seed=5,
        closed_loop_state={
            "feedback": {
                "discovery_rate_estimator": {
                    "family_counts": {
                        "family:a": 1,
                        "family:b": 1,
                        "family:c": 1,
                        "family:d": 1,
                    },
                    "total_observations": 4,
                }
            }
        },
    )

    snapshot = {row["arm_id"]: row for row in scheduler.snapshot()}
    row = snapshot[batch.arm_id]
    assert row["base_exploration_weight"] == 0.5
    assert row["current_exploration_weight"] > row["base_exploration_weight"]
    assert row["bayesian_exploration_observation_count"] == 1
    assert row["bayesian_unseen_probability"] == 1.0
    assert row["bayesian_exploration_bucket"] == "very_high"


def test_adaptive_budget_scheduler_can_disable_good_turing_exploration():
    scheduler = AdaptiveBudgetScheduler(
        [{"arm_id": "a", "target_suite": "core", "preset": "baseline", "seed": 1}],
        total_cases_budget=4,
        total_duration_budget_s=None,
        config=AdaptiveScheduleConfig(
            batch_cases=4,
            warmup_batches=1,
            exploration_weight=0.5,
            enable_bayesian_exploration=False,
        ),
    )

    batch = scheduler.next_batch()
    scheduler.record_result(
        batch,
        BatchObservation(
            cases=4,
            elapsed_s=1.0,
            throughput_cases_s=4.0,
            findings=0,
            candidate_bug_cases=0,
        ),
        next_seed=5,
        closed_loop_state={
            "feedback": {
                "discovery_rate_estimator": {
                    "family_counts": {"family:a": 1, "family:b": 1},
                    "total_observations": 2,
                }
            }
        },
    )

    row = scheduler.snapshot()[0]
    assert row["base_exploration_weight"] == 0.5
    assert row["current_exploration_weight"] == 0.5
    assert row["bayesian_exploration_enabled"] is False
    assert row["bayesian_exploration_observation_count"] == 0


def test_adaptive_budget_scheduler_reward_signal_prefers_repeated_stable_arm_over_single_spike():
    scheduler = AdaptiveBudgetScheduler(
        [
            {"arm_id": "spiky", "target_suite": "core", "preset": "baseline", "seed": 1},
            {"arm_id": "stable", "target_suite": "dataframe", "preset": "baseline", "seed": 1001},
        ],
        total_cases_budget=20,
        total_duration_budget_s=None,
        config=AdaptiveScheduleConfig(
            batch_cases=1,
            warmup_batches=1,
            exploration_weight=0.0,
            freshness_weight=0.0,
            stale_penalty=0.5,
            group_fairness_weight=0.0,
        ),
    )

    spiky = scheduler.next_batch()
    assert spiky.arm_id == "spiky"
    scheduler.record_result(
        spiky,
        BatchObservation(
            cases=1,
            elapsed_s=0.1,
            throughput_cases_s=10.0,
            findings=0,
            candidate_bug_cases=0,
            signal_new_behavior_cases=0,
            quality_oracle_count=1,
            quality_pass_count=1,
            quality_fail_count=0,
            quality_score_total=400.0,
        ),
        next_seed=2,
    )

    stable = scheduler.next_batch()
    assert stable.arm_id == "stable"
    scheduler.record_result(
        stable,
        BatchObservation(
            cases=1,
            elapsed_s=0.1,
            throughput_cases_s=10.0,
            findings=0,
            candidate_bug_cases=0,
            signal_new_behavior_cases=1,
            quality_oracle_count=3,
            quality_pass_count=3,
            quality_fail_count=0,
            quality_score_total=6.0,
            productive_mutation_cases=1,
            feedback_mutation_cases=1,
            guided_productive_cases=1,
            source_reward_adjustment_total=1.5,
            guidance_reward_adjustment_total=0.5,
            seed_schedule_delta_total=4.0,
        ),
        next_seed=1002,
    )

    for next_seed in (1003, 1004):
        stable = scheduler.next_batch()
        assert stable.arm_id == "stable"
        scheduler.record_result(
            stable,
            BatchObservation(
                cases=1,
                elapsed_s=0.1,
                throughput_cases_s=10.0,
                findings=0,
                candidate_bug_cases=0,
                signal_new_behavior_cases=1,
                quality_oracle_count=3,
                quality_pass_count=3,
                quality_fail_count=0,
                quality_score_total=6.0,
                productive_mutation_cases=1,
                feedback_mutation_cases=1,
                guided_productive_cases=1,
                source_reward_adjustment_total=1.5,
                guidance_reward_adjustment_total=0.5,
                seed_schedule_delta_total=4.0,
            ),
            next_seed=next_seed,
        )

    snapshot = {row["arm_id"]: row for row in scheduler.snapshot()}

    assert snapshot["spiky"]["mean_reward"] < snapshot["stable"]["mean_reward"]
    assert snapshot["stable"]["reward_signal"] > snapshot["spiky"]["reward_signal"]
    assert scheduler.next_batch().arm_id == "stable"


def test_batch_reward_suppresses_auxiliary_positive_terms_without_rewardable_signal():
    auxiliary_only = BatchObservation(
        cases=4,
        elapsed_s=1.0,
        throughput_cases_s=8.0,
        findings=1,
        candidate_bug_cases=0,
        signal_new_behavior_cases=0,
        semantic_divergence_count=1,
        resolved_semantic_divergence_count=1,
        quality_oracle_count=3,
        quality_pass_count=3,
        quality_fail_count=0,
        quality_score_total=500.0,
        feedback_mutation_cases=1,
        productive_mutation_cases=1,
        guided_productive_cases=1,
        source_reward_adjustment_total=3.0,
        guidance_reward_adjustment_total=3.0,
        seed_schedule_delta_total=5.0,
    )
    rewardable = BatchObservation(
        cases=4,
        elapsed_s=1.0,
        throughput_cases_s=8.0,
        findings=1,
        candidate_bug_cases=0,
        signal_new_behavior_cases=0,
        semantic_divergence_count=1,
        rewardable_semantic_divergence_count=1,
        needs_confirmation_count=1,
        quality_oracle_count=3,
        quality_pass_count=3,
        quality_fail_count=0,
        quality_score_total=3.0,
        feedback_mutation_cases=1,
        productive_mutation_cases=1,
        guided_productive_cases=1,
        source_reward_adjustment_total=0.5,
        guidance_reward_adjustment_total=0.5,
        seed_schedule_delta_total=0.5,
    )

    auxiliary_only_reward = scheduler_module._batch_reward(
        auxiliary_only,
        new_global_family_count=0,
        new_local_family_count=0,
    )
    rewardable_reward = scheduler_module._batch_reward(
        rewardable,
        new_global_family_count=0,
        new_local_family_count=0,
    )

    assert auxiliary_only_reward < 0.0
    assert rewardable_reward > auxiliary_only_reward


def test_batch_reward_penalizes_runtime_cost_without_losing_signal_terms():
    lean = BatchObservation(
        cases=4,
        elapsed_s=1.0,
        throughput_cases_s=4.0,
        findings=0,
        candidate_bug_cases=0,
        signal_new_behavior_cases=2,
        feedback_mutation_cases=2,
        productive_mutation_cases=2,
        guided_productive_cases=2,
        scheduler_feedback_share=0.02,
    )
    costly = BatchObservation(
        cases=4,
        elapsed_s=1.0,
        throughput_cases_s=4.0,
        findings=0,
        candidate_bug_cases=0,
        signal_new_behavior_cases=2,
        feedback_mutation_cases=2,
        productive_mutation_cases=2,
        guided_productive_cases=2,
        invalid_mutation_cases=1,
        redundant_mutation_cases=1,
        guided_redundant_cases=2,
        scheduler_feedback_share=0.80,
    )

    lean_reward = scheduler_module._batch_reward(
        lean,
        new_global_family_count=0,
        new_local_family_count=0,
    )
    costly_reward = scheduler_module._batch_reward(
        costly,
        new_global_family_count=0,
        new_local_family_count=0,
    )

    assert lean_reward > costly_reward
    assert scheduler_module._runtime_cost_penalty(costly) > scheduler_module._runtime_cost_penalty(lean)


def test_batch_reward_cost_normalizes_same_signal_by_elapsed_time():
    fast = BatchObservation(
        cases=4,
        elapsed_s=0.5,
        throughput_cases_s=8.0,
        findings=0,
        candidate_bug_cases=0,
        signal_new_behavior_cases=2,
    )
    slow = BatchObservation(
        cases=4,
        elapsed_s=2.0,
        throughput_cases_s=2.0,
        findings=0,
        candidate_bug_cases=0,
        signal_new_behavior_cases=2,
    )

    assert scheduler_module._batch_reward(
        fast,
        new_global_family_count=0,
        new_local_family_count=0,
    ) > scheduler_module._batch_reward(
        slow,
        new_global_family_count=0,
        new_local_family_count=0,
    )


def test_adaptive_budget_scheduler_carries_closed_loop_state_between_batches():
    scheduler = AdaptiveBudgetScheduler(
        [
            {"arm_id": "a", "target_suite": "core", "preset": "guided", "seed": 1},
        ],
        total_cases_budget=2,
        total_duration_budget_s=None,
        config=AdaptiveScheduleConfig(batch_cases=1, warmup_batches=1, exploration_weight=0.0),
    )

    first = scheduler.next_batch()
    assert first.job["persist_closed_loop_state"] is True
    assert "closed_loop_state" not in first.job

    scheduler.record_result(
        first,
        BatchObservation(
            cases=1,
            elapsed_s=0.1,
            throughput_cases_s=10.0,
            findings=0,
            candidate_bug_cases=0,
            new_behavior_cases=1,
        ),
        next_seed=2,
        closed_loop_state={"seen_signatures": ["disc-a"]},
    )

    second = scheduler.next_batch()
    assert second.job["persist_closed_loop_state"] is True
    assert second.job["closed_loop_state"] == {"seen_signatures": ["disc-a"]}
    assert scheduler.snapshot()[0]["closed_loop_state_present"] is True


def test_adaptive_budget_scheduler_records_transferable_learning_actions():
    learning = AdaptiveLearningState()
    scheduler = AdaptiveBudgetScheduler(
        [
            {
                "arm_id": "guided-a",
                "target_suite": "core",
                "preset": "guided",
                "seed": 1,
                "generator_profile": "discovery",
                "guidance_targets": ["semantic_family:null_semantics"],
                "semantic_objectives": ["partition"],
                "mutation_operators": ["append_range_filter"],
                "metamorphic_relations": ["order"],
                "target_version": "v2",
                "fixed_version": "v1",
            },
        ],
        total_cases_budget=1,
        total_duration_budget_s=None,
        config=AdaptiveScheduleConfig(batch_cases=1, warmup_batches=1, exploration_weight=0.0),
        learning_state=learning,
    )

    batch = scheduler.next_batch()
    scheduler.record_result(
        batch,
        BatchObservation(
            cases=1,
            elapsed_s=0.1,
            throughput_cases_s=10.0,
            findings=1,
            candidate_bug_cases=1,
            candidate_bug_families={"family@engine"},
            new_behavior_cases=1,
        ),
        next_seed=2,
    )

    assert {"batch_arm", "generator_profile", "semantic_objective", "mutation_operator", "metamorphic_relation", "version_pair"}.issubset(
        learning.bandits
    )
    assert learning.bandits["mutation_operator"].arms["append_range_filter"].pulls == 1
    assert learning.bandits["metamorphic_relation"].arms["order"].pulls == 1
    assert learning.bandits["version_pair"].arms["v2->v1"].pulls == 1


def test_adaptive_budget_scheduler_records_runtime_cost_in_learning_health():
    learning = AdaptiveLearningState()
    scheduler = AdaptiveBudgetScheduler(
        [
            {
                "arm_id": "guided-a",
                "target_suite": "core",
                "preset": "guided",
                "seed": 1,
                "generator_profile": "discovery",
            },
        ],
        total_cases_budget=1,
        total_duration_budget_s=None,
        config=AdaptiveScheduleConfig(batch_cases=1, warmup_batches=1, exploration_weight=0.0),
        learning_state=learning,
    )

    batch = scheduler.next_batch()
    scheduler.record_result(
        batch,
        BatchObservation(
            cases=1,
            elapsed_s=0.1,
            throughput_cases_s=10.0,
            findings=0,
            candidate_bug_cases=0,
            signal_new_behavior_cases=1,
            feedback_mutation_cases=1,
            invalid_mutation_cases=1,
            scheduler_feedback_share=0.90,
        ),
        next_seed=2,
    )

    arm = learning.bandits["batch_arm"].arms["guided-a"]
    score = learning.score_action(
        "batch_arm",
        "guided-a",
        context_features=("target_suite:core", "preset:guided"),
    )

    assert arm.runtime_cost_total > 0.0
    assert score["health_penalty"] > 0.0


def test_adaptive_budget_scheduler_can_ablate_runtime_cost_learning():
    learning = AdaptiveLearningState()
    scheduler = AdaptiveBudgetScheduler(
        [
            {
                "arm_id": "guided-a",
                "target_suite": "core",
                "preset": "guided",
                "seed": 1,
            },
        ],
        total_cases_budget=1,
        total_duration_budget_s=None,
        config=AdaptiveScheduleConfig(
            batch_cases=1,
            warmup_batches=1,
            exploration_weight=0.0,
            enable_runtime_cost_learning=False,
        ),
        learning_state=learning,
    )

    batch = scheduler.next_batch()
    scheduler.record_result(
        batch,
        BatchObservation(
            cases=1,
            elapsed_s=0.1,
            throughput_cases_s=10.0,
            findings=0,
            candidate_bug_cases=0,
            signal_new_behavior_cases=1,
            feedback_mutation_cases=1,
            invalid_mutation_cases=1,
            scheduler_feedback_share=0.90,
        ),
        next_seed=2,
    )

    assert learning.bandits["batch_arm"].arms["guided-a"].runtime_cost_total == 0.0


def test_batch_reward_can_disable_elapsed_cost_normalization_only():
    observation = BatchObservation(
        cases=1,
        elapsed_s=0.1,
        throughput_cases_s=10.0,
        findings=0,
        candidate_bug_cases=0,
        signal_new_behavior_cases=1,
        feedback_mutation_cases=1,
        invalid_mutation_cases=1,
        scheduler_feedback_share=0.90,
    )

    normalized = _batch_reward(
        observation,
        new_global_family_count=0,
        new_local_family_count=0,
        enable_runtime_cost=True,
        enable_cost_normalized_reward=True,
    )
    unnormalized_with_penalty = _batch_reward(
        observation,
        new_global_family_count=0,
        new_local_family_count=0,
        enable_runtime_cost=True,
        enable_cost_normalized_reward=False,
    )
    unnormalized_without_penalty = _batch_reward(
        observation,
        new_global_family_count=0,
        new_local_family_count=0,
        enable_runtime_cost=False,
        enable_cost_normalized_reward=False,
    )

    assert normalized != unnormalized_with_penalty
    assert unnormalized_with_penalty < unnormalized_without_penalty


def test_adaptive_budget_scheduler_can_ablate_online_reward_model_updates():
    learning = AdaptiveLearningState()
    scheduler = AdaptiveBudgetScheduler(
        [
            {
                "arm_id": "guided-a",
                "target_suite": "core",
                "preset": "guided",
                "seed": 1,
                "generator_profile": "discovery",
            },
        ],
        total_cases_budget=1,
        total_duration_budget_s=None,
        config=AdaptiveScheduleConfig(
            batch_cases=1,
            warmup_batches=1,
            exploration_weight=0.0,
            enable_online_reward_model=False,
        ),
        learning_state=learning,
    )

    batch = scheduler.next_batch()
    scheduler.record_result(
        batch,
        BatchObservation(
            cases=1,
            elapsed_s=0.1,
            throughput_cases_s=10.0,
            findings=1,
            candidate_bug_cases=1,
            candidate_bug_families={"family@engine"},
            signal_new_behavior_cases=1,
        ),
        next_seed=2,
    )

    arm_bandit = learning.bandits["batch_arm"]
    profile_bandit = learning.bandits["generator_profile"]
    score = learning.score_action(
        "batch_arm",
        "guided-a",
        context_features=("target_suite:core", "preset:guided"),
        enable_reward_model=False,
    )

    assert arm_bandit.arms["guided-a"].pulls == 1
    assert profile_bandit.arms["discovery"].pulls == 1
    assert arm_bandit.reward_model.total_updates == 0
    assert profile_bandit.reward_model.total_updates == 0
    assert score["model_prediction"] == 0.0
    assert score["uncertainty"] == 0.0


def test_adaptive_budget_scheduler_learning_signal_can_rank_prior_rewarded_arm():
    learning = AdaptiveLearningState()
    learning.record_outcome(
        "batch_arm",
        "b",
        context_features=("target_suite:core", "preset:guided"),
        reward=4.0,
    )
    scheduler = AdaptiveBudgetScheduler(
        [
            {"arm_id": "a", "target_suite": "core", "preset": "guided", "seed": 1},
            {"arm_id": "b", "target_suite": "core", "preset": "guided", "seed": 1001},
        ],
        total_cases_budget=2,
        total_duration_budget_s=None,
        config=AdaptiveScheduleConfig(
            batch_cases=1,
            warmup_batches=0,
            exploration_weight=0.0,
            freshness_weight=0.0,
            stale_penalty=0.0,
            group_fairness_weight=0.0,
            learning_weight=1.0,
        ),
        learning_state=learning,
    )
    for arm in scheduler.arms.values():
        arm.pulls = 1
    scheduler.total_batches_completed = 2

    assert scheduler.next_batch().arm_id == "b"


def test_adaptive_budget_scheduler_uses_transferable_action_learning_to_rank_arms():
    learning = AdaptiveLearningState()
    learning.record_outcome(
        "generator_profile",
        "productive_profile",
        context_features=("target_suite:core", "preset:guided"),
        reward=5.0,
    )
    learning.record_outcome(
        "generator_profile",
        "stale_profile",
        context_features=("target_suite:core", "preset:guided"),
        reward=-2.0,
    )
    scheduler = AdaptiveBudgetScheduler(
        [
            {
                "arm_id": "stale-arm",
                "target_suite": "core",
                "preset": "guided",
                "seed": 1,
                "generator_profile": "stale_profile",
            },
            {
                "arm_id": "productive-arm",
                "target_suite": "core",
                "preset": "guided",
                "seed": 1001,
                "generator_profile": "productive_profile",
            },
        ],
        total_cases_budget=2,
        total_duration_budget_s=None,
        config=AdaptiveScheduleConfig(
            batch_cases=1,
            warmup_batches=0,
            exploration_weight=0.0,
            freshness_weight=0.0,
            stale_penalty=0.0,
            group_fairness_weight=0.0,
            learning_weight=1.0,
        ),
        learning_state=learning,
    )
    for arm in scheduler.arms.values():
        arm.pulls = 1
    scheduler.total_batches_completed = 2

    assert scheduler.next_batch().arm_id == "productive-arm"


def test_adaptive_budget_scheduler_uses_continual_priority_to_rank_arms():
    learning = AdaptiveLearningState()
    learning.ingest_continual_ledger(
        {
            "families": [
                {"family": "cast_semantics@engine", "status": "fixed"},
                {"family": "null_semantics@engine", "status": "regression"},
            ]
        }
    )
    scheduler = AdaptiveBudgetScheduler(
        [
            {
                "arm_id": "cast-arm",
                "target_suite": "engine",
                "preset": "guided",
                "seed": 1,
                "semantic_focus_families": ["cast_semantics"],
            },
            {
                "arm_id": "null-arm",
                "target_suite": "engine",
                "preset": "guided",
                "seed": 1001,
                "semantic_focus_families": ["null_semantics"],
            },
        ],
        total_cases_budget=2,
        total_duration_budget_s=None,
        config=AdaptiveScheduleConfig(
            batch_cases=1,
            warmup_batches=0,
            exploration_weight=0.0,
            freshness_weight=0.0,
            stale_penalty=0.0,
            group_fairness_weight=0.0,
            learning_weight=1.0,
        ),
        learning_state=learning,
    )
    for arm in scheduler.arms.values():
        arm.pulls = 1
    scheduler.total_batches_completed = 2

    assert scheduler.next_batch().arm_id == "null-arm"
    null_score = learning.score_action(
        "semantic_objective",
        "semantic_family:null_semantics",
        context_features=("target_suite:engine",),
    )
    cast_score = learning.score_action(
        "semantic_objective",
        "semantic_family:cast_semantics",
        context_features=("target_suite:engine",),
    )
    assert null_score["continual_priority_signal"] > cast_score["continual_priority_signal"]


def test_adaptive_budget_scheduler_can_ablate_continual_learning_priority():
    learning = AdaptiveLearningState()
    learning.ingest_continual_ledger(
        {
            "families": [
                {"family": "null_semantics@engine", "status": "regression"},
            ]
        }
    )
    scheduler = AdaptiveBudgetScheduler(
        [
            {
                "arm_id": "neutral-arm",
                "target_suite": "engine",
                "preset": "guided",
                "seed": 1,
                "semantic_focus_families": ["cast_semantics"],
            },
            {
                "arm_id": "priority-arm",
                "target_suite": "engine",
                "preset": "guided",
                "seed": 1001,
                "semantic_focus_families": ["null_semantics"],
            },
        ],
        total_cases_budget=2,
        total_duration_budget_s=None,
        config=AdaptiveScheduleConfig(
            batch_cases=1,
            warmup_batches=0,
            exploration_weight=0.0,
            freshness_weight=0.0,
            stale_penalty=0.0,
            group_fairness_weight=0.0,
            learning_weight=1.0,
            enable_continual_learning=False,
        ),
        learning_state=learning,
    )
    for arm in scheduler.arms.values():
        arm.pulls = 1
    scheduler.total_batches_completed = 2

    priority_score = learning.score_action(
        "semantic_objective",
        "semantic_family:null_semantics",
        context_features=("target_suite:engine",),
        enable_continual_learning=False,
    )

    assert priority_score["version_signal"] == 0.0
    assert priority_score["continual_priority_signal"] == 0.0
    assert scheduler.next_batch().arm_id == "neutral-arm"


def test_adaptive_budget_scheduler_uses_active_learning_bonus_for_cold_contexts():
    learning = AdaptiveLearningState()
    for _ in range(8):
        learning.record_outcome(
            "generator_profile",
            "known_profile",
            context_features=("target_suite:core", "preset:guided"),
            version_id="v1",
            reward=0.1,
        )
    scheduler = AdaptiveBudgetScheduler(
        [
            {
                "arm_id": "known-arm",
                "target_suite": "core",
                "preset": "guided",
                "seed": 1,
                "generator_profile": "known_profile",
                "target_version": "v1",
            },
            {
                "arm_id": "cold-arm",
                "target_suite": "embedded_sql_cross",
                "preset": "guided",
                "seed": 1001,
                "generator_profile": "cold_profile",
                "target_version": "v2",
            },
        ],
        total_cases_budget=2,
        total_duration_budget_s=None,
        config=AdaptiveScheduleConfig(
            batch_cases=1,
            warmup_batches=0,
            exploration_weight=0.0,
            freshness_weight=0.0,
            stale_penalty=0.0,
            group_fairness_weight=0.0,
            learning_weight=1.0,
        ),
        learning_state=learning,
    )
    for arm in scheduler.arms.values():
        arm.pulls = 1
    scheduler.total_batches_completed = 2

    assert scheduler.next_batch().arm_id == "cold-arm"
    cold_signal = scheduler.snapshot()[0]["learning_signal"]
    assert cold_signal >= 0.0


def test_adaptive_budget_scheduler_can_ablate_active_learning_bonus():
    learning = AdaptiveLearningState()
    for _ in range(8):
        learning.record_outcome(
            "generator_profile",
            "known_profile",
            context_features=("target_suite:core", "preset:guided"),
            version_id="v1",
            reward=0.1,
        )
    scheduler = AdaptiveBudgetScheduler(
        [
            {
                "arm_id": "known-arm",
                "target_suite": "core",
                "preset": "guided",
                "seed": 1,
                "generator_profile": "known_profile",
                "target_version": "v1",
            },
            {
                "arm_id": "cold-arm",
                "target_suite": "embedded_sql_cross",
                "preset": "guided",
                "seed": 1001,
                "generator_profile": "cold_profile",
                "target_version": "v2",
            },
        ],
        total_cases_budget=2,
        total_duration_budget_s=None,
        config=AdaptiveScheduleConfig(
            batch_cases=1,
            warmup_batches=0,
            exploration_weight=0.0,
            freshness_weight=0.0,
            stale_penalty=0.0,
            group_fairness_weight=0.0,
            learning_weight=1.0,
            enable_active_learning=False,
        ),
        learning_state=learning,
    )
    for arm in scheduler.arms.values():
        arm.pulls = 1
    scheduler.total_batches_completed = 2

    cold_score = learning.score_action(
        "generator_profile",
        "cold_profile",
        context_features=("target_suite:embedded_sql_cross", "preset:guided"),
        version_id="v2",
        enable_active_learning=False,
    )

    assert cold_score["exploration_bonus"] == 0.0
    assert scheduler.next_batch().arm_id == "known-arm"


def test_adaptive_budget_scheduler_next_round_uses_unique_arms_and_reserved_budget():
    scheduler = AdaptiveBudgetScheduler(
        [
            {"arm_id": "a", "target_suite": "core", "preset": "baseline", "seed": 1},
            {"arm_id": "b", "target_suite": "dataframe", "preset": "baseline", "seed": 1001},
            {"arm_id": "c", "target_suite": "datafusion_cross", "preset": "guided", "seed": 2001},
        ],
        total_cases_budget=3,
        total_duration_budget_s=None,
        config=AdaptiveScheduleConfig(batch_cases=2, warmup_batches=1, exploration_weight=0.0),
    )

    round_batches = scheduler.next_round(2)

    assert len(round_batches) == 2
    assert {batch.arm_id for batch in round_batches} == {"a", "b"}
    assert [batch.cases for batch in round_batches] == [2, 1]
    assert scheduler.has_budget() is False


def test_adaptive_budget_scheduler_rebalances_underrepresented_groups():
    scheduler = AdaptiveBudgetScheduler(
        [
            {"arm_id": "core-a", "target_suite": "core", "preset": "baseline", "seed": 1},
            {"arm_id": "core-b", "target_suite": "core", "preset": "baseline", "seed": 101},
            {"arm_id": "df-a", "target_suite": "dataframe", "preset": "baseline", "seed": 1001},
            {"arm_id": "df-b", "target_suite": "dataframe", "preset": "baseline", "seed": 1101},
        ],
        total_cases_budget=24,
        total_duration_budget_s=None,
        config=AdaptiveScheduleConfig(
            batch_cases=1,
            warmup_batches=1,
            exploration_weight=0.0,
            group_fairness_weight=0.0,
            max_group_pull_gap=3,
        ),
    )

    # Warm up every arm once.
    warmup_batches = [scheduler.next_batch() for _ in range(4)]
    assert {batch.arm_id for batch in warmup_batches} == {"core-a", "core-b", "df-a", "df-b"}
    for batch in warmup_batches:
        scheduler.record_result(
            batch,
            BatchObservation(
                cases=1,
                elapsed_s=0.1,
                throughput_cases_s=10.0,
                findings=0,
                candidate_bug_cases=0,
                new_behavior_cases=0,
            ),
            next_seed=batch.seed + 1,
        )

    # Simulate a productive streak that over-concentrated the dataframe group.
    dominant = scheduler.arms["df-a"]
    dominant.pulls += 3
    dominant.total_reward += 12.0
    dominant.last_reward = 4.0
    scheduler.total_batches_completed += 3

    rebalanced = scheduler.next_batch()
    assert rebalanced.group_key == "core:baseline"


def test_adaptive_budget_scheduler_annealing_can_choose_non_greedy_arm():
    def build_scheduler(temperature: float) -> AdaptiveBudgetScheduler:
        scheduler = AdaptiveBudgetScheduler(
            [
                {"arm_id": "best", "target_suite": "core", "preset": "baseline", "seed": 1},
                {"arm_id": "mid", "target_suite": "core", "preset": "baseline", "seed": 1001},
                {"arm_id": "low", "target_suite": "core", "preset": "baseline", "seed": 2001},
            ],
            total_cases_budget=3,
            total_duration_budget_s=None,
            config=AdaptiveScheduleConfig(
                batch_cases=1,
                warmup_batches=1,
                exploration_weight=0.0,
                freshness_weight=0.0,
                stale_penalty=0.0,
                group_fairness_weight=0.0,
                max_group_pull_gap=0,
                annealing_initial_temperature=temperature,
                annealing_decay=1.0,
                annealing_min_temperature=0.0,
            ),
        )
        for arm_id, total_reward in {"best": 20.0, "mid": 12.0, "low": 4.0}.items():
            arm = scheduler.arms[arm_id]
            arm.pulls = 4
            arm.total_reward = total_reward
            arm.last_reward = total_reward / 4.0
        scheduler.total_batches_completed = 3
        return scheduler

    greedy = build_scheduler(temperature=0.0)
    annealed = build_scheduler(temperature=100.0)

    assert greedy.next_batch().arm_id == "best"
    assert annealed.next_batch().arm_id == "low"
    assert {row["annealing_temperature"] for row in annealed.snapshot()} == {100.0}


def test_summarize_batch_run_counts_scheduler_signals(tmp_path: Path):
    run_file = tmp_path / "run-scheduler.jsonl"
    append_jsonl(
        {
            "case": {"case_id": "case-1", "seed": 1},
            "case_index": 0,
            "elapsed_s": 0.1,
            "is_new_behavior": True,
            "signal_new_behavior": False,
            "candidate_source": "feedback_mutation",
            "stored_in_feedback_corpus": True,
            "feedback_summary": {
                "candidate_source": "feedback_mutation",
                "stored_in_feedback_corpus": True,
                "quality_oracle_count": 3,
                "quality_pass_count": 3,
                "quality_fail_count": 0,
                "quality_score_total": 5.5,
                "source_reward_adjustment": 0.4,
                "guidance_reward_adjustment": 0.25,
                "seed_schedule_delta": 3.6,
                "mutation_oracle_verdict": "productive_mutation",
                "feedback_oracle_verdict": "finding_yield",
                "guidance_oracle_verdict": "guided_productive",
            },
            "findings": [
                {
                    "root_cause": "grouped_topk_null_sort_key",
                    "triage_verdict": "candidate_implementation_bug",
                    "suspicious_backends": ["datafusion"],
                    "signature": "sig-1",
                }
            ],
        },
        run_file,
    )
    append_jsonl(
        {
            "case": {"case_id": "case-2", "seed": 2},
            "case_index": 1,
            "elapsed_s": 0.2,
            "is_new_behavior": False,
            "findings": [
                {
                    "root_cause": "nan_inf_semantics",
                    "triage_verdict": "expected_semantic_divergence",
                    "suspicious_backends": ["duckdb"],
                    "signature": "sig-2",
                }
            ],
        },
        run_file,
    )
    append_jsonl(
        {
            "case": {"case_id": "case-3", "seed": 3},
            "case_index": 2,
            "elapsed_s": 0.3,
            "is_new_behavior": False,
            "findings": [
                {
                    "root_cause": "order_only_normalization_mismatch",
                    "triage_verdict": "normalizer_false_positive",
                    "suspicious_backends": ["sqlite"],
                    "signature": "sig-3",
                }
            ],
        },
        run_file,
    )
    append_jsonl(
        {
            "case": {"case_id": "case-4", "seed": 4},
            "case_index": 3,
            "elapsed_s": 0.4,
            "is_new_behavior": False,
            "findings": [
                {
                    "root_cause": "unknown",
                    "triage_verdict": "needs_manual_confirmation",
                    "suspicious_backends": ["duckdb"],
                    "signature": "sig-4",
                }
            ],
        },
        run_file,
    )
    dump_json(
        {
            "elapsed_s": 0.5,
            "throughput_cases_s": 6.0,
            "stage_profile": {
                "totals_ms": {
                    "generate_mutate_ms": 1.0,
                    "backend_execution_ms": 5.0,
                    "normalize_ms": 1.0,
                    "oracle_classification_ms": 1.0,
                    "scheduler_feedback_ms": 2.0,
                    "logging_artifact_ms": 0.0,
                    "total_case_wall_ms": 10.0,
                },
                "share_of_total": {"scheduler_feedback_ms": 0.2},
            },
        },
        run_meta_path(run_file),
    )

    observation = summarize_batch_run(run_file)

    assert observation.cases == 4
    assert observation.findings == 4
    assert observation.candidate_bug_cases == 1
    assert observation.candidate_bug_families == {"grouped_topk_null_sort_key@datafusion"}
    assert observation.semantic_divergence_count == 1
    assert observation.false_positive_count == 1
    assert observation.needs_confirmation_count == 1
    assert observation.new_behavior_cases == 1
    assert observation.signal_new_behavior_cases == 0
    assert observation.throughput_cases_s == 6.0
    assert observation.first_candidate_bug_case_index == 0
    assert observation.first_candidate_bug_elapsed_s == 0.1
    assert observation.candidate_bug_discovery_auc == 1.0
    assert observation.scheduler_feedback_share == 0.2
    assert observation.feedback_case_count == 4
    assert observation.feedback_mutation_cases == 1
    assert observation.stored_in_feedback_corpus_cases == 1
    assert observation.quality_oracle_count == 3
    assert observation.quality_pass_count == 3
    assert observation.source_reward_adjustment_total == 0.5
    assert observation.guidance_reward_adjustment_total == 0.25
    assert observation.seed_schedule_delta_total == pytest.approx(-0.43)
    assert observation.productive_mutation_cases == 1
    assert observation.feedback_finding_yield_cases == 1
    assert observation.guided_productive_cases == 1


def test_summarize_batch_run_filters_non_rewardable_signal_new_behavior(tmp_path: Path):
    run_file = tmp_path / "run-scheduler-filtered-signal.jsonl"
    rows = [
        {
            "case": {"case_id": "resolved", "seed": 1},
            "case_index": 0,
            "elapsed_s": 0.1,
            "is_new_behavior": True,
            "signal_new_behavior": True,
            "findings": [
                {
                    "root_cause": "nan_inf_semantics",
                    "triage_verdict": "expected_semantic_divergence",
                    "suspicious_backends": ["duckdb"],
                }
            ],
        },
        {
            "case": {"case_id": "false-positive", "seed": 2},
            "case_index": 1,
            "elapsed_s": 0.2,
            "is_new_behavior": True,
            "signal_new_behavior": True,
            "findings": [
                {
                    "root_cause": "order_only_normalization_mismatch",
                    "triage_verdict": "normalizer_false_positive",
                    "false_positive": True,
                    "suspicious_backends": ["sqlite"],
                }
            ],
        },
        {
            "case": {"case_id": "source-issue", "seed": 3},
            "case_index": 2,
            "elapsed_s": 0.3,
            "is_new_behavior": True,
            "signal_new_behavior": True,
            "findings": [
                {
                    "root_cause": "csv_long_numeric_roundtrip",
                    "triage_verdict": "candidate_implementation_bug",
                    "source_issue": "duckdb/duckdb#12345",
                    "suspicious_backends": ["duckdb"],
                }
            ],
        },
        {
            "case": {"case_id": "pure-behavior", "seed": 4},
            "case_index": 3,
            "elapsed_s": 0.4,
            "is_new_behavior": True,
            "signal_new_behavior": True,
            "findings": [],
        },
        {
            "case": {"case_id": "candidate", "seed": 5},
            "case_index": 4,
            "elapsed_s": 0.5,
            "is_new_behavior": True,
            "signal_new_behavior": True,
            "findings": [
                {
                    "root_cause": "topk_filter_pushdown",
                    "triage_verdict": "candidate_implementation_bug",
                    "suspicious_backends": ["datafusion"],
                }
            ],
        },
        {
            "case": {"case_id": "semantic-needs-confirmation", "seed": 6},
            "case_index": 5,
            "elapsed_s": 0.6,
            "is_new_behavior": True,
            "signal_new_behavior": True,
            "findings": [
                {
                    "root_cause": "string_expression",
                    "triage_verdict": "semantic_divergence_needs_confirmation",
                    "suspicious_backends": ["sqlite"],
                }
            ],
        },
    ]
    for row in rows:
        append_jsonl(row, run_file)
    dump_json({"elapsed_s": 1.0, "throughput_cases_s": 6.0}, run_meta_path(run_file))

    observation = summarize_batch_run(run_file)

    assert observation.new_behavior_cases == 6
    assert observation.signal_new_behavior_cases == 3
    assert observation.candidate_bug_cases == 1
    assert observation.candidate_bug_families == {"topk_filter_pushdown@datafusion"}
    assert observation.semantic_divergence_count == 2
    assert observation.rewardable_semantic_divergence_count == 1
    assert observation.resolved_semantic_divergence_count == 1
    assert observation.false_positive_count == 1
