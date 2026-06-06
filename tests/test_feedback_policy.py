from types import SimpleNamespace

from datadiff.dsl import Case, ColumnSpec, Program, TableData
from datadiff.feedback_policy import (
    FeedbackStorageContext,
    FeedbackStoragePolicy,
    feedback_storage_decision,
    source_scheduler_snapshot,
)


def _case(seed: int = 1, operation: dict | None = None) -> Case:
    return Case(
        f"case-{seed}",
        seed,
        [TableData("t0", [ColumnSpec("x", "int")], [{"x": seed}])],
        Program("prog", seed, [operation or {"op": "select", "columns": ["x"]}]),
    )


def test_feedback_storage_policy_keeps_generated_cases_and_probe_guard():
    assert feedback_storage_decision(_case(), FeedbackStorageContext()) == (True, "")
    assert feedback_storage_decision(
        _case(2, {"op": "running_sum", "source": "x", "as": "run_x"}),
        FeedbackStorageContext(),
    ) == (False, "calibration_probe_case")


def test_feedback_storage_policy_skips_plain_feedback_children():
    context = FeedbackStorageContext(
        candidate_source="feedback_mutation",
        seed_lineage={"depth": 1},
    )

    assert feedback_storage_decision(_case(), context) == (False, "feedback_mutation_child")


def test_feedback_storage_policy_admits_shallow_candidate_bug_children():
    context = FeedbackStorageContext(
        candidate_source="feedback_mutation",
        seed_lineage={"depth": 2},
        candidate_bug_families=("groupby_aggregation@datafusion",),
    )
    policy = FeedbackStoragePolicy(
        allow_feedback_mutation_child_candidate_bug=True,
        max_feedback_mutation_child_depth=2,
    )

    assert feedback_storage_decision(_case(), context, policy) == (True, "")


def test_feedback_storage_policy_rejects_deep_feedback_children():
    context = FeedbackStorageContext(
        candidate_source="feedback_mutation",
        seed_lineage={"depth": 3},
        candidate_bug_families=("groupby_aggregation@datafusion",),
    )
    policy = FeedbackStoragePolicy(
        allow_feedback_mutation_child_candidate_bug=True,
        max_feedback_mutation_child_depth=2,
    )

    assert feedback_storage_decision(_case(), context, policy) == (False, "feedback_mutation_child")


def test_source_scheduler_snapshot_is_side_effect_free_adapter():
    scheduler = SimpleNamespace(snapshot=lambda: [{"source": "generated"}])

    assert source_scheduler_snapshot(SimpleNamespace(source_scheduler=scheduler)) == [{"source": "generated"}]
    assert source_scheduler_snapshot(SimpleNamespace(source_scheduler=None)) == []
    assert source_scheduler_snapshot(None) == []
