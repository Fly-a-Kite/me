from pathlib import Path

from datadiff.scheduler import (
    AdaptiveBudgetScheduler,
    AdaptiveScheduleConfig,
    BatchObservation,
    LocalSourceScheduler,
    summarize_batch_run,
)
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

    assert first_reward == 4.5
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

    assert snapshot["spiky"]["mean_reward"] > snapshot["stable"]["mean_reward"]
    assert snapshot["stable"]["reward_signal"] > snapshot["spiky"]["reward_signal"]
    assert scheduler.next_batch().arm_id == "stable"


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
    assert observation.feedback_case_count == 4
    assert observation.feedback_mutation_cases == 1
    assert observation.stored_in_feedback_corpus_cases == 1
    assert observation.quality_oracle_count == 3
    assert observation.quality_pass_count == 3
    assert observation.source_reward_adjustment_total == 0.4
    assert observation.guidance_reward_adjustment_total == 0.25
    assert observation.seed_schedule_delta_total == 2.85
    assert observation.productive_mutation_cases == 1
    assert observation.feedback_finding_yield_cases == 1
    assert observation.guided_productive_cases == 1
