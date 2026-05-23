from types import SimpleNamespace

from datadiff import feedback
from datadiff.dsl import Case, ColumnSpec, Program, TableData
from datadiff.feedback import FeedbackState
from datadiff.mutator import MUTATION_OPERATOR_NAMES, PROBE_MUTATION_OPERATOR_NAMES, ROOT_TARGETED_MUTATION_OPERATOR_NAMES
from datadiff.scheduler import LocalSourceScheduler


def _case(seed: int) -> Case:
    return Case(
        f"case-{seed}",
        seed,
        [TableData("t0", [ColumnSpec("x", "int")], [{"x": seed}])],
        Program(f"prog-{seed}", seed, []),
    )


def test_feedback_persistence_is_bounded_per_run(tmp_path, monkeypatch):
    monkeypatch.setattr(feedback, "CORPUS_DIR", tmp_path / "corpus")
    state = FeedbackState(persist_to_disk=True, max_persisted=2)

    assert state.record(_case(1), "0000000000000001", False) is True
    assert state.last_persisted_to_disk is True
    assert state.record(_case(2), "0000000000000002", False) is True
    assert state.last_persisted_to_disk is True
    assert state.record(_case(3), "0000000000000003", True) is True
    assert state.last_persisted_to_disk is False

    persisted = sorted((tmp_path / "corpus" / "interesting").glob("*.json"))
    assert len(persisted) == 2
    assert state.persisted_count == 2


def test_feedback_persistence_limit_zero_disables_disk_writes(tmp_path, monkeypatch):
    monkeypatch.setattr(feedback, "CORPUS_DIR", tmp_path / "corpus")
    state = FeedbackState(persist_to_disk=True, max_persisted=0)

    assert state.record(_case(1), "0000000000000001", True) is True

    assert state.last_persisted_to_disk is False
    assert not (tmp_path / "corpus" / "interesting").exists()


def test_feedback_record_caps_candidate_bug_family_storage():
    state = FeedbackState(max_cases_per_candidate_family=2)

    assert state.record(
        _case(1),
        "0000000000000001",
        True,
        candidate_bug_families=["grouped_topk_null_sort_key@datafusion"],
    )
    assert state.record(
        _case(2),
        "0000000000000002",
        True,
        candidate_bug_families=["grouped_topk_null_sort_key@datafusion"],
    )
    assert not state.record(
        _case(3),
        "0000000000000003",
        True,
        candidate_bug_families=["grouped_topk_null_sort_key@datafusion"],
    )

    assert state.last_record_skip_reason == "candidate_family_saturated"
    assert len(state.interesting_cases) == 2


def test_feedback_source_scheduler_prefers_productive_mutations():
    scheduler = LocalSourceScheduler(exploration_weight=0.0)
    state = FeedbackState(source_scheduler=scheduler, interesting_cases=[_case(1)])
    generated = _case(7)

    first = state.choose_case(7, generated)
    assert first.case_id == generated.case_id
    assert state.last_candidate_source == "generated"
    assert state.last_candidate_metadata["seed_lineage"]["root_seed"] == 7
    assert state.last_candidate_metadata["seed_lineage"]["depth"] == 0
    assert state.last_candidate_metadata["mutation"]["operator"] == "generated"

    generated_reward = state.record_candidate_result(
        "generated",
        has_finding=False,
        is_new_behavior=False,
        preflight={"valid": True, "fallback_used": False},
    )
    assert generated_reward == -0.1
    assert state.last_source_reward == -0.1

    second = state.choose_case(8, generated)
    assert second.case_id.endswith("-mut-8")
    assert state.last_candidate_source == "feedback_mutation"
    assert state.last_candidate_metadata["seed_lineage"]["parent_case_id"] == "case-1"
    assert state.last_candidate_metadata["seed_lineage"]["mutation_seed"] == 8
    assert state.last_candidate_metadata["seed_lineage"]["depth"] == 1
    assert state.last_candidate_metadata["mutation"]["operator"] in MUTATION_OPERATOR_NAMES

    feedback_reward = state.record_candidate_result(
        "feedback_mutation",
        has_finding=True,
        is_new_behavior=True,
        preflight={"valid": True, "fallback_used": False},
        candidate_bug=True,
    )
    assert feedback_reward == 4.5
    assert state.last_source_reward == 4.5

    third = state.choose_case(9, generated)
    assert third.case_id.endswith("-mut-9")
    assert state.last_candidate_source == "feedback_mutation"


def test_feedback_mutation_falls_back_to_generated_when_attempts_do_not_change(monkeypatch):
    scheduler = LocalSourceScheduler(exploration_weight=0.0)
    state = FeedbackState(source_scheduler=scheduler, interesting_cases=[_case(1), _case(2)])
    generated = _case(7)

    state.record_candidate_result(
        "generated",
        has_finding=False,
        is_new_behavior=False,
        preflight={"valid": True, "fallback_used": False},
    )

    def unchanged_mutation(case, seed, *, allow_probe_operators=True):
        metadata = {
            "candidate_source": "feedback_mutation",
            "seed_lineage": {
                "root_seed": case.seed,
                "parent_seed": case.seed,
                "parent_case_id": case.case_id,
                "mutation_seed": seed,
                "depth": 1,
            },
            "mutation": {
                "operator": "value",
                "detail": "value:int:x",
                "changed": False,
            },
        }
        return SimpleNamespace(case=case, metadata=metadata)

    monkeypatch.setattr(feedback, "mutate_case_with_metadata", unchanged_mutation)

    selected = state.choose_case(8, generated)

    assert selected.case_id == generated.case_id
    assert state.last_candidate_source == "generated"
    assert state.last_candidate_metadata["mutation"]["operator"] == "generated"


def test_feedback_mutations_avoid_direct_probe_append_operators():
    scheduler = LocalSourceScheduler(exploration_weight=0.0, min_feedback_share=1.0)
    state = FeedbackState(source_scheduler=scheduler, interesting_cases=[_case(1)])
    generated = _case(7)
    scheduler.record_result(
        "generated",
        has_finding=False,
        is_new_behavior=False,
        preflight_valid=True,
        fallback_used=False,
    )

    seen = set()
    for seed in range(8, 80):
        selected = state.choose_case(seed, generated)
        operator_name = state.last_candidate_metadata["mutation"]["operator"]
        seen.add(operator_name)
        assert selected.case_id.endswith(f"-mut-{seed}")
        assert operator_name not in PROBE_MUTATION_OPERATOR_NAMES
        assert operator_name not in ROOT_TARGETED_MUTATION_OPERATOR_NAMES

    assert seen
