from collections import Counter
from types import SimpleNamespace

from datadiff import feedback
from datadiff.dsl import Case, ColumnSpec, Program, TableData
from datadiff.feedback import FeedbackState
from datadiff.mutator import (
    MUTATION_OPERATOR_NAMES,
    PROBE_MUTATION_OPERATOR_NAMES,
    ROOT_TARGETED_MUTATION_OPERATOR_NAMES,
    SPECIALIZED_DISCOVERY_MUTATION_OPERATOR_NAMES,
)
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


def test_feedback_record_deduplicates_on_discovery_signature():
    state = FeedbackState()

    assert state.record(_case(1), "0000000000000001", False, discovery_signature="disc-a") is True
    assert state.record(_case(2), "0000000000000002", False, discovery_signature="disc-a") is False

    assert state.last_record_skip_reason == "duplicate_uninteresting_behavior"
    assert len(state.interesting_cases) == 1


def test_feedback_state_round_trip_preserves_corpus_and_source_scheduler():
    source_scheduler = LocalSourceScheduler(exploration_weight=0.0, min_feedback_share=0.2)
    state = FeedbackState(
        persist_to_disk=True,
        max_persisted=3,
        max_cases_per_profile=5,
        source_scheduler=source_scheduler,
    )
    case = _case(7)
    case.metadata["mixed_generator_profile"] = "join_ordered_agg_topk"

    assert state.record(
        case,
        "0000000000000007",
        True,
        discovery_signature="disc-7",
        candidate_bug_families=["grouped_topk_null_sort_key@datafusion"],
        target_keys=["target:groupby_sorted_input"],
        schedule_delta=1.25,
    )
    source_scheduler.record_result(
        "feedback_mutation",
        has_finding=True,
        is_new_behavior=True,
        preflight_valid=True,
        fallback_used=False,
        candidate_bug=True,
        candidate_bug_families=["grouped_topk_null_sort_key@datafusion"],
        candidate_bug_signatures=["sig-7"],
    )

    state_dict = state.to_state_dict()
    restored_scheduler = LocalSourceScheduler.from_state_dict(
        state_dict["source_scheduler"],
        exploration_weight=0.0,
        enable_family_saturation=True,
        family_saturation_threshold=8,
        saturated_family_reward=0.02,
        known_saturated_bug_families=None,
    )
    restored = FeedbackState.from_state_dict(
        state_dict,
        persist_to_disk=False,
        max_persisted=9,
        max_cases_per_profile=5,
        source_scheduler=restored_scheduler,
    )

    assert restored.seen_signatures == {"disc-7"}
    assert [item.case_id for item in restored.interesting_cases] == [case.case_id]
    assert restored.case_family_keys == [["grouped_topk_null_sort_key@datafusion"]]
    assert restored.case_target_keys == [["target:groupby_sorted_input"]]
    assert restored.case_schedule_rewards[0] == state.case_schedule_rewards[0]
    assert restored.case_schedule_feedback_totals == [0.0]
    assert restored.case_schedule_feedback_counts == [0]
    assert restored.recent_mutation_parent_counts == Counter()
    assert restored.recent_mutation_operator_counts == Counter()
    assert restored.source_scheduler is not None
    assert restored.source_scheduler.total_pulls == 1
    assert restored.source_scheduler.snapshot()[1]["source"] == "generated"


def test_feedback_state_from_legacy_state_dict_backfills_schedule_feedback_arrays():
    legacy_case = _case(11)
    state_dict = {
        "seen_signatures": ["disc-11"],
        "interesting_cases": [legacy_case.to_dict()],
        "case_utilities": [1.0],
        "case_family_keys": [[]],
        "case_profile_keys": [""],
        "case_target_keys": [["target:legacy"]],
        "case_mutation_pulls": [2],
        "case_schedule_rewards": [3.5],
    }

    restored = FeedbackState.from_state_dict(
        state_dict,
        persist_to_disk=False,
        max_persisted=0,
        max_cases_per_profile=16,
        source_scheduler=None,
    )

    assert restored.seen_signatures == {"disc-11"}
    assert [item.case_id for item in restored.interesting_cases] == [legacy_case.case_id]
    assert restored.case_schedule_rewards == [3.5]
    assert restored.case_schedule_feedback_totals == [0.0]
    assert restored.case_schedule_feedback_counts == [0]
    assert restored._case_seed_schedule_reward(0) == 3.5


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


def test_feedback_record_caps_nonfinding_profile_storage():
    state = FeedbackState(max_cases_per_profile=2)

    first = _case(1)
    first.metadata["mixed_generator_profile"] = "join_ordered_agg_topk"
    second = _case(2)
    second.metadata["mixed_generator_profile"] = "join_ordered_agg_topk"
    third = _case(3)
    third.metadata["mixed_generator_profile"] = "join_ordered_agg_topk"
    finding_case = _case(4)
    finding_case.metadata["mixed_generator_profile"] = "join_ordered_agg_topk"

    assert state.record(first, "0000000000000001", False)
    assert state.record(second, "0000000000000002", False)
    assert not state.record(third, "0000000000000003", False)
    assert state.last_record_skip_reason == "profile_saturated"
    assert state.record(finding_case, "0000000000000004", True)

    assert state.stored_profiles["join_ordered_agg_topk"] == 3
    assert len(state.interesting_cases) == 3


def test_feedback_record_replaces_low_utility_seed_when_corpus_is_full():
    state = FeedbackState(max_corpus=2, max_cases_per_profile=0)
    simple = _case(1)
    other_simple = _case(2)
    rich = Case(
        "case-rich",
        3,
        [
            TableData("t0", [ColumnSpec("id", "int"), ColumnSpec("g", "str"), ColumnSpec("x", "int")], [{"id": 1, "g": None, "x": 1}]),
            TableData("t1", [ColumnSpec("id", "int")], [{"id": 1}]),
        ],
        Program(
            "prog-rich",
            3,
            [
                {"op": "join", "table": "t1", "left_on": "id", "right_on": "id", "how": "left"},
                {"op": "filter", "column": "x", "cmp": ">", "value": 0},
                {"op": "groupby", "keys": ["g"], "aggs": [{"column": "x", "func": "count", "as": "count_x"}]},
            ],
        ),
    )

    assert state.record(simple, "0000000000000001", False)
    assert state.record(other_simple, "0000000000000002", False)
    assert state.record(rich, "0000000000000003", False)

    assert rich in state.interesting_cases
    assert len(state.interesting_cases) == 2
    assert state.last_record_skip_reason == ""


def test_feedback_mutation_prefers_productive_parent_over_complex_parent(monkeypatch):
    state = FeedbackState(source_scheduler=LocalSourceScheduler(exploration_weight=0.0, min_feedback_share=1.0))
    productive_simple = _case(1)
    complex_unproven = Case(
        "case-complex",
        2,
        [
            TableData("t0", [ColumnSpec("id", "int"), ColumnSpec("g", "str"), ColumnSpec("x", "int")], [{"id": 1, "g": None, "x": 1}]),
            TableData("t1", [ColumnSpec("id", "int")], [{"id": 1}]),
        ],
        Program(
            "prog-high",
            2,
            [
                {"op": "join", "table": "t1", "left_on": "id", "right_on": "id", "how": "left"},
                {"op": "filter", "column": "x", "cmp": ">", "value": 0},
                {"op": "groupby", "keys": ["g"], "aggs": [{"column": "x", "func": "count", "as": "count_x"}]},
            ],
        ),
    )
    state.record(productive_simple, "0000000000000001", False)
    state.record(complex_unproven, "0000000000000002", False)
    state.case_schedule_rewards[state.interesting_cases.index(productive_simple)] = 2.0
    state.source_scheduler.record_result(
        "generated",
        has_finding=False,
        is_new_behavior=False,
        preflight_valid=True,
        fallback_used=False,
    )
    parents = []

    def tracked_mutation(case, seed, *, allow_probe_operators=True, operator_scores=None):
        parents.append(case.case_id)
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
                "changed": True,
            },
        }
        return SimpleNamespace(case=case, metadata=metadata)

    monkeypatch.setattr(feedback, "mutate_case_with_metadata", tracked_mutation)

    selected = state.choose_case(8, _case(8))

    assert selected.case_id == "case-1"
    assert parents == ["case-1"]
    assert state.case_mutation_pulls[state.interesting_cases.index(productive_simple)] == 1


def test_feedback_mutation_parent_selection_ignores_retention_utility_without_reward_signal():
    state = FeedbackState()
    simple = _case(1)
    complex_case = Case(
        "case-complex",
        2,
        [
            TableData("t0", [ColumnSpec("id", "int"), ColumnSpec("g", "str"), ColumnSpec("x", "int")], [{"id": 1, "g": None, "x": 1}]),
            TableData("t1", [ColumnSpec("id", "int")], [{"id": 1}]),
        ],
        Program(
            "prog-complex",
            2,
            [
                {"op": "join", "table": "t1", "left_on": "id", "right_on": "id", "how": "left"},
                {"op": "filter", "column": "x", "cmp": ">", "value": 0},
                {"op": "mutate", "column": "x2", "expr": {"kind": "add_const", "source": "x", "value": 0}},
                {"op": "groupby", "keys": ["g"], "aggs": [{"column": "x2", "func": "count", "as": "count_x"}]},
            ],
        ),
    )

    assert state.record(simple, "0000000000000001", False)
    assert state.record(complex_case, "0000000000000002", False)

    assert state.case_utilities[state.interesting_cases.index(complex_case)] > state.case_utilities[
        state.interesting_cases.index(simple)
    ]
    assert state._choose_mutation_seed_index(8) == state.interesting_cases.index(simple)


def test_feedback_schedule_score_rewards_rare_target_keys_without_changing_retention_utility():
    state = FeedbackState()
    first_common = _case(1)
    repeated_common = _case(3)
    rare = _case(2)
    assert state.record(first_common, "0000000000000001", False, target_keys=["target:common"])
    assert state.record(repeated_common, "0000000000000003", False, target_keys=["target:common"])
    assert state.record(rare, "0000000000000002", False, target_keys=["target:rare"])

    common_index = state.interesting_cases.index(repeated_common)
    rare_index = state.interesting_cases.index(rare)

    assert state.case_utilities[common_index] == state.case_utilities[rare_index]
    assert state.case_schedule_rewards[rare_index] == state.case_schedule_rewards[common_index]
    assert state._case_target_novelty_score(rare_index) > state._case_target_novelty_score(common_index)
    assert state._case_seed_schedule_score(rare_index) > state._case_seed_schedule_score(common_index)
    assert state.case_target_keys[rare_index] == ["target:rare"]


def test_feedback_cluster_keys_group_similar_seed_structure():
    state = FeedbackState()
    first = _case(1)
    second = _case(2)
    different_target = _case(3)

    assert state.record(first, "0000000000000001", False, target_keys=["semantic_family:null_semantics"])
    assert state.record(second, "0000000000000002", False, target_keys=["semantic_family:null_semantics"])
    assert state.record(
        different_target,
        "0000000000000003",
        False,
        target_keys=["semantic_family:ordering_semantics"],
    )

    first_index = state.interesting_cases.index(first)
    second_index = state.interesting_cases.index(second)
    different_index = state.interesting_cases.index(different_target)

    assert state.case_cluster_keys[first_index] == state.case_cluster_keys[second_index]
    assert state.case_cluster_keys[first_index] != state.case_cluster_keys[different_index]
    assert state.stored_cluster_keys[state.case_cluster_keys[first_index]] == 2


def test_feedback_cluster_keys_include_capability_dimension_before_truncation():
    state = FeedbackState()
    first = _case(1)
    same = _case(2)
    different_capability = _case(3)

    target_keys = [
        "target:common_a",
        "target:common_b",
        "target:common_c",
        "capability:op:join",
    ]
    assert state.record(first, "0000000000000001", False, target_keys=target_keys)
    assert state.record(same, "0000000000000002", False, target_keys=target_keys)
    assert state.record(
        different_capability,
        "0000000000000003",
        False,
        target_keys=[
            "target:common_a",
            "target:common_b",
            "target:common_c",
            "capability:op:sort",
        ],
    )

    first_index = state.interesting_cases.index(first)
    same_index = state.interesting_cases.index(same)
    different_index = state.interesting_cases.index(different_capability)

    assert state.case_cluster_keys[first_index] == state.case_cluster_keys[same_index]
    assert state.case_cluster_keys[first_index] != state.case_cluster_keys[different_index]
    assert "capability_op_join" in state.case_cluster_keys[first_index]
    assert "capability_op_sort" in state.case_cluster_keys[different_index]


def test_feedback_cluster_schedule_cools_recently_reused_seed_family():
    state = FeedbackState()
    first = _case(1)
    second = _case(2)
    assert state.record(first, "0000000000000001", False, target_keys=["semantic_family:null_semantics"])
    assert state.record(second, "0000000000000002", False, target_keys=["semantic_family:null_semantics"])
    first_index = state.interesting_cases.index(first)
    second_index = state.interesting_cases.index(second)

    before = state._case_seed_schedule_score(second_index)
    state._record_mutation_seed_pull(first_index)
    after = state._case_seed_schedule_score(second_index)

    assert state._recent_cluster_pull_count(state.case_cluster_keys[second_index]) == 1
    assert after < before


def test_feedback_cluster_reward_transfers_to_similar_seed_and_round_trips():
    state = FeedbackState()
    first = _case(1)
    second = _case(2)
    assert state.record(first, "0000000000000001", False, target_keys=["semantic_family:cast_semantics"])
    assert state.record(second, "0000000000000002", False, target_keys=["semantic_family:cast_semantics"])
    first_index = state.interesting_cases.index(first)
    second_index = state.interesting_cases.index(second)
    cluster_key = state.case_cluster_keys[first_index]

    baseline = state._case_seed_schedule_score(second_index)
    state.last_feedback_parent_index = first_index
    state.record_feedback_candidate_reward("feedback_mutation", 3.0)

    assert state.cluster_schedule_feedback_totals[cluster_key] == 3.0
    assert state.cluster_schedule_feedback_counts[cluster_key] == 1
    assert state._case_seed_schedule_score(second_index) > baseline

    state._record_mutation_seed_pull(first_index)
    restored = FeedbackState.from_state_dict(
        state.to_state_dict(),
        persist_to_disk=False,
        max_persisted=0,
        max_cases_per_profile=16,
        source_scheduler=None,
    )

    assert restored.case_cluster_keys == state.case_cluster_keys
    assert restored.stored_cluster_keys == state.stored_cluster_keys
    assert restored.cluster_schedule_feedback_totals == state.cluster_schedule_feedback_totals
    assert restored.cluster_schedule_feedback_counts == state.cluster_schedule_feedback_counts
    assert restored.quality_archive.to_state_dict() == state.quality_archive.to_state_dict()
    assert restored._recent_cluster_pull_count(cluster_key) == 1


def test_feedback_quality_archive_is_rebuilt_from_legacy_state_dict():
    legacy_case = _case(11)
    state_dict = {
        "seen_signatures": ["disc-11"],
        "interesting_cases": [legacy_case.to_dict()],
        "case_utilities": [2.0],
        "case_family_keys": [[]],
        "case_profile_keys": [""],
        "case_target_keys": [["semantic_family:null_semantics"]],
        "case_mutation_pulls": [0],
        "case_schedule_rewards": [0.0],
    }

    restored = FeedbackState.from_state_dict(
        state_dict,
        persist_to_disk=False,
        max_persisted=0,
        max_cases_per_profile=16,
        source_scheduler=None,
    )
    cluster_key = restored.case_cluster_keys[0]

    assert not restored.quality_archive.is_empty()
    assert restored.quality_archive.elite_indexes(cluster_key) == [0]
    assert restored.quality_archive.seed_elite_bonus(cluster_key, 0) > 0.0


def test_feedback_quality_archive_health_penalty_lowers_cluster_schedule_score():
    state = FeedbackState()
    assert state.record(_case(1), "0000000000000001", False, target_keys=["semantic_family:cast_semantics"])
    index = 0
    cluster_key = state.case_cluster_keys[index]
    baseline = state._case_seed_schedule_score(index)
    state.last_feedback_parent_index = index

    state.record_feedback_outcome_reward(
        "feedback_mutation",
        0.0,
        preflight_valid=False,
        fallback_used=True,
        false_positive=True,
    )

    assert state.quality_archive.cluster_health_penalty(cluster_key) > 0.0
    assert state._case_seed_schedule_score(index) < baseline


def test_feedback_quality_archive_can_be_disabled_for_ablation():
    state = FeedbackState(enable_quality_archive=False)

    assert state.record(
        _case(1),
        "0000000000000001",
        False,
        target_keys=["semantic_family:filtering"],
    )
    context = state.candidate_quality_context(
        _case(2),
        target_keys=["semantic_family:filtering"],
    )
    state.last_feedback_parent_index = 0
    state.record_feedback_outcome_reward("feedback_mutation", 2.0)

    assert state.quality_archive.is_empty()
    assert context["archive_enabled"] is False
    assert "archive_known" not in context
    assert state.to_state_dict()["enable_quality_archive"] is False


def test_feedback_candidate_quality_context_reports_archive_signals_without_mutating_archive():
    state = FeedbackState()
    assert state.record(_case(1), "0000000000000001", False, target_keys=["semantic_family:cast_semantics"])
    cluster_key = state.case_cluster_keys[0]
    state.last_feedback_parent_index = 0
    state.record_feedback_outcome_reward("feedback_mutation", 2.0)

    known_before = state.quality_archive.to_state_dict()
    known_context = state.candidate_quality_context(
        _case(2),
        target_keys=["semantic_family:cast_semantics"],
    )

    assert known_context["cluster_key"] == cluster_key
    assert known_context["archive_known"] is True
    assert known_context["archive_elite_indexes"] == [0]
    assert known_context["archive_seed_count"] == 1
    assert known_context["archive_outcome_count"] == 1
    assert known_context["archive_cluster_reward"] > 0.0
    assert known_context["cluster_count"] == 1
    assert state.quality_archive.to_state_dict() == known_before

    unknown_before = state.quality_archive.to_state_dict()
    unknown_context = state.candidate_quality_context(
        _case(3),
        target_keys=["semantic_family:ordering_semantics"],
    )

    assert unknown_context["archive_known"] is False
    assert unknown_context["archive_seed_count"] == 0
    assert unknown_context["archive_outcome_count"] == 0
    assert state.quality_archive.to_state_dict() == unknown_before


def test_feedback_candidate_reward_updates_parent_and_operator_scores():
    state = FeedbackState()
    state.record(_case(1), "0000000000000001", False)
    state.last_feedback_parent_index = 0
    state.last_feedback_operator = "value"

    state.record_feedback_candidate_reward("feedback_mutation", 2.5)

    assert state.case_schedule_rewards == [0.0]
    assert state.case_schedule_feedback_totals == [2.5]
    assert state.case_schedule_feedback_counts == [1]
    assert state._case_seed_schedule_reward(0) > state.case_schedule_rewards[0]
    assert state.mutation_operator_pulls["value"] == 1
    assert state.mutation_operator_rewards["value"] == 2.5
    score = state._mutation_operator_score_snapshot()["value"]
    assert 1.0 < score < 1.2
    assert state.adaptive_learning.bandits["mutation_operator"].arms["value"].pulls == 1


def test_feedback_mutation_operator_learning_state_round_trips_and_influences_scores():
    state = FeedbackState()
    assert state.record(
        _case(1),
        "0000000000000001",
        False,
        target_keys=["semantic_family:filtering"],
    )
    state.last_feedback_parent_index = 0
    state.last_feedback_operator = "append_range_filter"
    state.record_feedback_outcome_reward("feedback_mutation", 3.0)

    restored = FeedbackState.from_state_dict(
        state.to_state_dict(),
        persist_to_disk=False,
        max_persisted=0,
        max_cases_per_profile=16,
        source_scheduler=None,
    )
    scores = restored._mutation_operator_score_snapshot(target_keys=["semantic_family:filtering"])

    assert "mutation_operator" in restored.adaptive_learning.bandits
    assert restored.adaptive_learning.bandits["mutation_operator"].arms["append_range_filter"].pulls == 1
    assert scores["append_range_filter"] > scores["append_order_projection"]


def test_feedback_mutation_operator_learning_can_be_disabled_for_ablation():
    state = FeedbackState(enable_mutation_operator_learning=False)
    assert state.record(
        _case(1),
        "0000000000000001",
        False,
        target_keys=["semantic_family:filtering"],
    )
    state.last_feedback_parent_index = 0
    state.last_feedback_operator = "append_range_filter"

    state.record_feedback_outcome_reward("feedback_mutation", 3.0)

    assert "mutation_operator" not in state.adaptive_learning.bandits
    assert state.mutation_operator_pulls["append_range_filter"] == 1
    restored = FeedbackState.from_state_dict(
        state.to_state_dict(),
        persist_to_disk=False,
        max_persisted=0,
        max_cases_per_profile=16,
        source_scheduler=None,
    )
    assert restored.enable_mutation_operator_learning is False


def test_feedback_canonical_method_aliases_match_legacy_behavior():
    state = FeedbackState()
    generated = _case(7)

    assert state.select_case(7, generated) == state.choose_case(7, generated)
    assert (
        state.record_candidate_outcome(
            "generated",
            has_finding=False,
            is_new_behavior=False,
            preflight={"valid": True, "fallback_used": False},
        )
        == state.record_candidate_result(
            "generated",
            has_finding=False,
            is_new_behavior=False,
            preflight={"valid": True, "fallback_used": False},
        )
    )


def test_feedback_record_feedback_outcome_reward_matches_legacy_alias():
    state = FeedbackState()
    state.record(_case(1), "0000000000000001", False)
    state.last_feedback_parent_index = 0
    state.last_feedback_operator = "value"

    state.record_feedback_outcome_reward("feedback_mutation", 1.5)

    assert state.case_schedule_feedback_totals == [1.5]
    assert state.case_schedule_feedback_counts == [1]
    assert state.mutation_operator_rewards["value"] == 1.5


def test_feedback_parent_schedule_reward_uses_bounded_feedback_mean():
    state = FeedbackState()
    state.record(_case(1), "0000000000000001", False)
    state.last_feedback_parent_index = 0

    for reward in [3.0, 3.0, -1.0, 5.0]:
        state.record_feedback_candidate_reward("feedback_mutation", reward)

    assert state.case_schedule_rewards == [0.0]
    assert state.case_schedule_feedback_totals == [10.0]
    assert state.case_schedule_feedback_counts == [4]
    assert 0.0 < state._case_feedback_reward_signal(0) <= 2.5
    assert state._case_seed_schedule_reward(0) == state.case_schedule_rewards[0] + state._case_feedback_reward_signal(0)


def test_feedback_operator_score_snapshot_cools_recently_reused_operator():
    state = FeedbackState()
    for _ in range(12):
        state.mutation_operator_pulls["append_range_filter"] += 1
        state.mutation_operator_rewards["append_range_filter"] += 0.5
        state.recent_mutation_operators.append("append_range_filter")

    scores = state._mutation_operator_score_snapshot()

    assert scores["append_range_filter"] < scores["__untried__"]


def test_feedback_operator_score_snapshot_prefers_target_affine_operator():
    state = FeedbackState()
    state.mutation_operator_pulls["append_range_filter"] = 4
    state.mutation_operator_rewards["append_range_filter"] = 0.4
    state.mutation_operator_pulls["append_left_join_case_membership"] = 4
    state.mutation_operator_rewards["append_left_join_case_membership"] = 0.4

    for _ in range(3):
        state.mutation_operator_target_pulls["append_left_join_case_membership|risk:left_join_case_when_membership"] += 1
        state.mutation_operator_target_rewards["append_left_join_case_membership|risk:left_join_case_when_membership"] += 1.5
        state.mutation_operator_target_pulls["append_range_filter|risk:left_join_case_when_membership"] += 1
        state.mutation_operator_target_rewards["append_range_filter|risk:left_join_case_when_membership"] -= 0.2

    scores = state._mutation_operator_score_snapshot(target_keys=["risk:left_join_case_when_membership"])

    assert scores["append_left_join_case_membership"] > scores["append_range_filter"]


def test_feedback_operator_score_snapshot_keeps_stats_only_operator():
    state = FeedbackState()
    state.mutation_operator_pulls["legacy_operator"] = 2
    state.mutation_operator_rewards["legacy_operator"] = 1.5

    scores = state._mutation_operator_score_snapshot()

    assert "legacy_operator" in scores


def test_feedback_normalizes_legacy_risk_keys_to_semantic_signal_keys():
    state = FeedbackState.from_state_dict(
        {
            "case_target_keys": [["risk:left_join_case_when_membership"]],
            "stored_target_keys": {"risk:left_join_case_when_membership": 2},
            "mutation_operator_target_pulls": {
                "append_left_join_case_membership|risk:left_join_case_when_membership": 3
            },
            "mutation_operator_target_rewards": {
                "append_left_join_case_membership|risk:left_join_case_when_membership": 1.5
            },
        },
        persist_to_disk=False,
        max_persisted=0,
        max_cases_per_profile=16,
        source_scheduler=None,
    )

    assert state.case_target_keys == [["semantic_signal:left_join_case_when_membership"]]
    assert state.stored_target_keys == {"semantic_signal:left_join_case_when_membership": 2}
    assert state.mutation_operator_target_pulls == {
        "append_left_join_case_membership|semantic_signal:left_join_case_when_membership": 3
    }
    assert state.mutation_operator_target_rewards == {
        "append_left_join_case_membership|semantic_signal:left_join_case_when_membership": 1.5
    }


def test_feedback_operator_score_snapshot_prefers_semantic_family_affine_operator():
    state = FeedbackState()
    state.mutation_operator_pulls["append_case_when"] = 4
    state.mutation_operator_rewards["append_case_when"] = 0.4
    state.mutation_operator_pulls["append_range_filter"] = 4
    state.mutation_operator_rewards["append_range_filter"] = 0.4

    for _ in range(3):
        state.mutation_operator_target_pulls["append_case_when|semantic_family:conditional_semantics"] += 1
        state.mutation_operator_target_rewards["append_case_when|semantic_family:conditional_semantics"] += 1.0
        state.mutation_operator_target_pulls["append_range_filter|semantic_family:conditional_semantics"] += 1
        state.mutation_operator_target_rewards["append_range_filter|semantic_family:conditional_semantics"] -= 0.2

    scores = state._mutation_operator_score_snapshot(target_keys=["semantic_family:conditional_semantics"])

    assert scores["append_case_when"] > scores["append_range_filter"]


def test_feedback_operator_score_snapshot_uses_structural_semantic_family_affinity_for_untried_operator():
    state = FeedbackState()
    state.mutation_operator_pulls["append_range_filter"] = 4
    state.mutation_operator_rewards["append_range_filter"] = 0.4

    scores = state._mutation_operator_score_snapshot(target_keys=["semantic_family:conditional_semantics"])

    assert scores["append_left_join_case_membership"] > scores["append_range_filter"]


def test_feedback_operator_score_snapshot_uses_structural_semantic_signal_affinity_for_untried_operator():
    state = FeedbackState()
    state.mutation_operator_pulls["append_range_filter"] = 4
    state.mutation_operator_rewards["append_range_filter"] = 0.4

    scores = state._mutation_operator_score_snapshot(
        target_keys=["semantic_signal:left_join_case_when_membership"]
    )

    assert scores["append_left_join_case_membership"] > scores["append_range_filter"]


def test_feedback_operator_score_snapshot_uses_structural_objective_affinity_for_untried_operator():
    state = FeedbackState()
    state.mutation_operator_pulls["append_range_filter"] = 4
    state.mutation_operator_rewards["append_range_filter"] = 0.4

    scores = state._mutation_operator_score_snapshot(
        target_keys=["exploration_objective:cross_model_consistency"]
    )

    assert scores["append_left_join_case_membership"] > scores["append_range_filter"]


def test_feedback_mutation_parent_selection_rotates_after_pulls():
    state = FeedbackState()
    low = _case(1)
    high = Case(
        "case-high",
        2,
        [
            TableData("t0", [ColumnSpec("id", "int"), ColumnSpec("g", "str"), ColumnSpec("x", "int")], [{"id": 1, "g": None, "x": 1}]),
            TableData("t1", [ColumnSpec("id", "int")], [{"id": 1}]),
        ],
        Program(
            "prog-high",
            2,
            [
                {"op": "join", "table": "t1", "left_on": "id", "right_on": "id", "how": "left"},
                {"op": "filter", "column": "x", "cmp": ">", "value": 0},
                {"op": "groupby", "keys": ["g"], "aggs": [{"column": "x", "func": "count", "as": "count_x"}]},
            ],
        ),
    )
    state.record(low, "0000000000000001", False)
    state.record(high, "0000000000000002", False)
    high_index = state.interesting_cases.index(high)
    low_index = state.interesting_cases.index(low)
    state.case_mutation_pulls[high_index] = 100

    selected_index = state._choose_mutation_seed_index(8)

    assert selected_index == low_index


def test_feedback_mutation_parent_selection_cools_recently_reused_parent():
    state = FeedbackState()
    first = _case(1)
    second = _case(2)
    assert state.record(first, "0000000000000001", False)
    assert state.record(second, "0000000000000002", False)
    first_index = state.interesting_cases.index(first)
    second_index = state.interesting_cases.index(second)
    state.case_schedule_rewards[first_index] = 3.0
    state.case_schedule_rewards[second_index] = 1.0
    for _ in range(6):
        state.recent_mutation_parent_indexes.append(first_index)

    selected_index = state._choose_mutation_seed_index(8)

    assert selected_index == second_index


def test_feedback_recent_pull_counters_backfill_from_legacy_deques():
    state = FeedbackState.from_state_dict(
        {
            "recent_mutation_parent_indexes": [1, 1, 2],
            "recent_mutation_operators": ["append_range_filter", "append_range_filter", "append_case_when"],
        },
        persist_to_disk=False,
        max_persisted=0,
        max_cases_per_profile=16,
        source_scheduler=None,
    )

    assert state._recent_parent_pull_count(1) == 2
    assert state._recent_parent_pull_count(2) == 1
    assert state._recent_operator_pull_count("append_range_filter") == 2
    assert state._recent_operator_pull_count("append_case_when") == 1


def test_feedback_replacement_resets_mutation_pulls():
    state = FeedbackState(max_corpus=1, max_cases_per_profile=0)
    low = _case(1)
    high = Case(
        "case-high",
        2,
        [
            TableData("t0", [ColumnSpec("id", "int"), ColumnSpec("g", "str"), ColumnSpec("x", "int")], [{"id": 1, "g": None, "x": 1}]),
            TableData("t1", [ColumnSpec("id", "int")], [{"id": 1}]),
        ],
        Program(
            "prog-high",
            2,
            [
                {"op": "join", "table": "t1", "left_on": "id", "right_on": "id", "how": "left"},
                {"op": "filter", "column": "x", "cmp": ">", "value": 0},
                {"op": "groupby", "keys": ["g"], "aggs": [{"column": "x", "func": "count", "as": "count_x"}]},
            ],
        ),
    )

    assert state.record(low, "0000000000000001", False)
    state.case_mutation_pulls[0] = 7
    assert state.record(high, "0000000000000002", False)

    assert state.interesting_cases == [high]
    assert state.case_mutation_pulls == [0]


def test_feedback_record_rejects_low_utility_seed_when_corpus_is_full():
    state = FeedbackState(max_corpus=1, max_cases_per_profile=0)
    rich = Case(
        "case-rich",
        3,
        [
            TableData("t0", [ColumnSpec("id", "int"), ColumnSpec("g", "str"), ColumnSpec("x", "int")], [{"id": 1, "g": None, "x": 1}]),
            TableData("t1", [ColumnSpec("id", "int")], [{"id": 1}]),
        ],
        Program(
            "prog-rich",
            3,
            [
                {"op": "join", "table": "t1", "left_on": "id", "right_on": "id", "how": "left"},
                {"op": "filter", "column": "x", "cmp": ">", "value": 0},
                {"op": "groupby", "keys": ["g"], "aggs": [{"column": "x", "func": "count", "as": "count_x"}]},
            ],
        ),
    )
    simple = _case(4)

    assert state.record(rich, "0000000000000003", False)
    assert not state.record(simple, "0000000000000004", False)

    assert state.interesting_cases == [rich]
    assert state.last_record_skip_reason == "corpus_full_low_utility"


def test_feedback_replacement_decrements_profile_counts():
    state = FeedbackState(max_corpus=1, max_cases_per_profile=1)
    simple = _case(1)
    simple.metadata["mixed_generator_profile"] = "simple_profile"
    rich = Case(
        "case-rich",
        2,
        [
            TableData("t0", [ColumnSpec("id", "int"), ColumnSpec("g", "str"), ColumnSpec("x", "int")], [{"id": 1, "g": None, "x": 1}]),
            TableData("t1", [ColumnSpec("id", "int")], [{"id": 1}]),
        ],
        Program(
            "prog-rich",
            2,
            [
                {"op": "join", "table": "t1", "left_on": "id", "right_on": "id", "how": "left"},
                {"op": "groupby", "keys": ["g"], "aggs": [{"column": "x", "func": "count", "as": "count_x"}]},
            ],
        ),
        metadata={"mixed_generator_profile": "rich_profile"},
    )
    another_simple = Case(
        "case-simple-rich",
        3,
        [
            TableData("t0", [ColumnSpec("id", "int"), ColumnSpec("g", "str"), ColumnSpec("x", "int")], [{"id": 1, "g": None, "x": 1}]),
            TableData("t1", [ColumnSpec("id", "int")], [{"id": 1}]),
        ],
        Program(
            "prog-simple-rich",
            3,
            [
                {"op": "join", "table": "t1", "left_on": "id", "right_on": "id", "how": "left"},
                {"op": "filter", "column": "x", "cmp": ">", "value": 0},
                {"op": "mutate", "column": "x2", "expr": {"kind": "add_const", "source": "x", "value": 0}},
                {"op": "groupby", "keys": ["g"], "aggs": [{"column": "x2", "func": "count", "as": "count_x"}]},
            ],
        ),
        metadata={"mixed_generator_profile": "simple_profile"},
    )

    assert state.record(simple, "0000000000000001", False)
    assert state.record(rich, "0000000000000002", False)
    assert state.record(another_simple, "0000000000000003", False)

    assert state.interesting_cases == [another_simple]
    assert state.stored_profiles["simple_profile"] == 1
    assert state.stored_profiles["rich_profile"] == 0


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
    decision = state.last_candidate_metadata["feedback_decision"]
    assert decision["parent_case_id"] == "case-1"
    assert decision["mutation_seed"] == 8
    assert decision["retention_utility"] >= 1.0
    assert decision["schedule_score"] > 0.0
    assert decision["mutation_pulls"] == 1
    assert decision["selected_operator"] == state.last_candidate_metadata["mutation"]["operator"]

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


def test_feedback_source_scheduler_does_not_reward_resolved_semantic_divergence():
    scheduler = LocalSourceScheduler(exploration_weight=0.0)
    state = FeedbackState(source_scheduler=scheduler)

    reward = state.record_candidate_result(
        "generated",
        has_finding=False,
        is_new_behavior=False,
        preflight={"valid": True, "fallback_used": False},
        semantic_divergence=False,
    )

    assert reward == -0.1
    assert state.last_source_reward == -0.1


def test_feedback_source_scheduler_rewards_semantic_divergence_needing_confirmation():
    scheduler = LocalSourceScheduler(exploration_weight=0.0)
    state = FeedbackState(source_scheduler=scheduler)

    reward = state.record_candidate_result(
        "generated",
        has_finding=True,
        is_new_behavior=False,
        preflight={"valid": True, "fallback_used": False},
        semantic_divergence=True,
    )

    assert reward == 0.2
    assert state.last_source_reward == 0.2


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

    def unchanged_mutation(case, seed, *, allow_probe_operators=True, operator_scores=None):
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
        assert operator_name not in SPECIALIZED_DISCOVERY_MUTATION_OPERATOR_NAMES

    assert seen
