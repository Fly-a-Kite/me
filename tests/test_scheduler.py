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


def test_summarize_batch_run_counts_scheduler_signals(tmp_path: Path):
    run_file = tmp_path / "run-scheduler.jsonl"
    append_jsonl(
        {
            "case": {"case_id": "case-1", "seed": 1},
            "case_index": 0,
            "elapsed_s": 0.1,
            "is_new_behavior": True,
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
    assert observation.throughput_cases_s == 6.0
    assert observation.first_candidate_bug_case_index == 0
    assert observation.first_candidate_bug_elapsed_s == 0.1
    assert observation.candidate_bug_discovery_auc == 1.0
