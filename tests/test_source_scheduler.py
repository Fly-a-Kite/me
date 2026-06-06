from datadiff.source_scheduler import LocalSourceScheduler


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
        candidate_bug_families=["join_semantics@duckdb"],
    )

    assert scheduler.choose_source(feedback_available=True) == "feedback_mutation"
    snapshot = {row["source"]: row for row in scheduler.snapshot()}
    assert snapshot["feedback_mutation"]["mean_reward"] > snapshot["generated"]["mean_reward"]


def test_local_source_scheduler_round_trips_state_dict():
    scheduler = LocalSourceScheduler(exploration_weight=0.25, min_feedback_share=0.2)
    scheduler.record_result(
        "generated",
        has_finding=True,
        is_new_behavior=True,
        preflight_valid=True,
        fallback_used=False,
        candidate_bug=True,
        candidate_bug_families=["groupby_aggregation@datafusion"],
        candidate_bug_signatures=["sig-a"],
    )

    restored = LocalSourceScheduler.from_state_dict(
        scheduler.to_state_dict(),
        exploration_weight=0.25,
        enable_family_saturation=True,
        family_saturation_threshold=8,
        saturated_family_reward=0.02,
        known_saturated_bug_families=[],
    )

    assert restored.total_pulls == 1
    assert restored.arms["generated"].candidate_bug_families["groupby_aggregation@datafusion"] == 1
    assert restored.arms["generated"].candidate_bug_signatures["sig-a"] == 1
