from collections import Counter
from types import SimpleNamespace

from datadiff import feedback
from datadiff.champion_corpus import ChampionRegistry, ChampionSeed
from datadiff.dsl import Case, ColumnSpec, Program, TableData
from datadiff.feedback import FeedbackState
from datadiff.mutator import (
    MUTATION_OPERATOR_NAMES,
    PROBE_MUTATION_OPERATOR_NAMES,
    ROOT_TARGETED_MUTATION_OPERATOR_NAMES,
    SHRINK_MUTATION_OPERATOR_NAMES,
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


def _semantic_plan_case(seed: int, label: str) -> Case:
    case = _case(seed)
    case.metadata["interaction_descriptor"] = {
        "tokens": [
            {
                "category": "semantic_physical",
                "components": [
                    "backend=datafusion",
                    f"operator={label}",
                    "op=limit",
                ],
            },
            {
                "category": "plan_state",
                "components": [
                    "backend=datafusion",
                    "mode=query_engine",
                    "kind=physical",
                    f"operators={label}",
                ],
            },
        ]
    }
    case.metadata["input_layouts"] = {
        "t0": {
            "representation": "chunked",
            "chunk_count": seed + 1,
            "dictionary_columns": ["x"] if seed % 2 else [],
        }
    }
    return case


def _fingerprint_payload(offset: int = 0) -> dict[str, object]:
    return {
        "minhash_signature": [offset + index for index in range(64)],
        "op_skeleton_hash": f"ops-{offset}",
        "type_mix_token": "num1",
        "null_density_bucket": 0,
        "row_mass_bucket": 1,
        "column_count": 1,
        "feature_tokens": [f"fp_op:ops-{offset}", "fp_type_mix:num1"],
    }


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


def test_feedback_record_persists_disagreement_descriptor_metadata():
    state = FeedbackState()
    case = _case(9)
    descriptor = {
        "backend_groups": [["duckdb"], ["sqlite"]],
        "pair_count": 1,
        "feature_tokens": ["disagree_pair:duckdb|sqlite"],
    }

    assert state.record(
        case,
        "0000000000000009",
        True,
        disagreement_descriptor=descriptor,
    )

    assert state.interesting_cases[0].metadata["disagreement_descriptor"] == descriptor


def test_feedback_records_backend_disagreement_behavioral_axis():
    state = FeedbackState()
    case = _case(12)
    descriptor = {
        "pair_disagrees": [
            {"left": "duckdb", "right": "pandas", "disagrees": True},
            {"left": "pandas", "right": "sqlite", "disagrees": False},
        ],
        "mismatch_class": "value",
    }

    assert state.record(
        case,
        "0000000000000012",
        True,
        disagreement_descriptor=descriptor,
    )

    stored_descriptor = state.case_behavioral_descriptors[0]
    assert stored_descriptor["backend_disagreement_axis"] == "pair:duckdb_pandas"
    assert state.quality_archive.axis_seed_count("bd_backend_disagreement", "pair:duckdb_pandas") == 1


def test_feedback_can_disable_backend_disagreement_behavioral_axis():
    state = FeedbackState(enable_disagreement_bd_axis=False)
    descriptor = {
        "pair_disagrees": [
            {"left": "duckdb", "right": "pandas", "disagrees": True},
        ],
        "mismatch_class": "value",
    }

    assert state.record(
        _case(12),
        "0000000000000012",
        True,
        disagreement_descriptor=descriptor,
    )

    stored_descriptor = state.case_behavioral_descriptors[0]
    assert stored_descriptor["backend_disagreement_axis"] == "pair:none"
    assert state.quality_archive.axis_seed_count("bd_backend_disagreement", "pair:duckdb_pandas") == 0
    assert state.quality_archive.axis_seed_count("bd_backend_disagreement", "pair:none") == 1


def test_feedback_record_persists_case_fingerprint_metadata():
    state = FeedbackState()
    case = _case(10)
    fingerprint = _fingerprint_payload(10)

    assert state.record(
        case,
        "0000000000000010",
        True,
        case_fingerprint=fingerprint,
    )

    assert state.interesting_cases[0].metadata["case_fingerprint"] == fingerprint


def test_feedback_record_skips_redundant_nonfinding_fingerprint():
    state = FeedbackState()
    fingerprint = _fingerprint_payload(20)

    assert state.record(_case(20), "0000000000000020", False, case_fingerprint=fingerprint)
    assert not state.record(_case(21), "0000000000000021", False, case_fingerprint=fingerprint)

    assert state.last_record_skip_reason == "fingerprint_redundant"
    assert len(state.interesting_cases) == 1


def test_feedback_can_disable_minhash_redundant_fingerprint_filter():
    state = FeedbackState(enable_minhash_dedup=False)
    fingerprint = _fingerprint_payload(20)

    assert state.record(_case(20), "0000000000000020", False, case_fingerprint=fingerprint)
    assert state.record(_case(21), "0000000000000021", False, case_fingerprint=fingerprint)

    assert state.last_record_skip_reason == ""
    assert len(state.interesting_cases) == 2


def test_feedback_record_keeps_finding_and_candidate_family_despite_redundant_fingerprint():
    state = FeedbackState()
    fingerprint = _fingerprint_payload(30)

    assert state.record(_case(30), "0000000000000030", False, case_fingerprint=fingerprint)
    assert state.record(_case(31), "0000000000000031", True, case_fingerprint=fingerprint)
    assert state.record(
        _case(32),
        "0000000000000032",
        False,
        candidate_bug_families=["window_boundary@duckdb"],
        case_fingerprint=fingerprint,
    )

    assert state.last_record_skip_reason == ""
    assert len(state.interesting_cases) == 3


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


def test_feedback_schedule_penalizes_overused_candidate_family():
    state = FeedbackState(max_cases_per_candidate_family=4)
    repeated = _case(1)
    fresh = _case(2)

    assert state.record(
        repeated,
        "0000000000000001",
        True,
        candidate_bug_families=["repeated_family@engine"],
    )
    assert state.record(
        fresh,
        "0000000000000002",
        True,
        candidate_bug_families=["fresh_family@engine"],
    )
    state.stored_candidate_bug_families["repeated_family@engine"] = 16
    repeated_index = state.interesting_cases.index(repeated)
    fresh_index = state.interesting_cases.index(fresh)

    assert state._case_family_reuse_penalty(repeated_index) > state._case_family_reuse_penalty(fresh_index)
    assert state._case_seed_schedule_score(repeated_index) < state._case_seed_schedule_score(fresh_index)


def test_feedback_promotes_stable_candidate_family_to_champion(tmp_path):
    registry = ChampionRegistry(tmp_path / "champions.jsonl")
    state = FeedbackState(
        max_cases_per_candidate_family=0,
        champion_registry=registry,
        champion_version_id="v1",
        champion_promotion_threshold=3,
    )

    assert state.record(_case(1), "0000000000000001", True, candidate_bug_families=["family@engine"])
    assert state.record(_case(2), "0000000000000002", True, candidate_bug_families=["family@engine"])
    assert state.record(_case(3), "0000000000000003", True, candidate_bug_families=["family@engine"])

    champions = registry.champions_for_version("v2")
    assert len(champions) == 1
    assert champions[0].bug_family_keys == ("family@engine",)
    assert state.champion_family_hits["family@engine"] == 3
    assert "family@engine" in state.champion_promoted_families


def test_feedback_can_disable_champion_corpus_promotion(tmp_path):
    registry = ChampionRegistry(tmp_path / "champions.jsonl")
    state = FeedbackState(
        max_cases_per_candidate_family=0,
        champion_registry=registry,
        champion_version_id="v1",
        champion_promotion_threshold=1,
        enable_champion_corpus=False,
    )

    assert state.record(_case(1), "0000000000000001", True, candidate_bug_families=["family@engine"])

    assert registry.champions_for_version("v2") == []
    assert state.champion_family_hits == {}


def test_feedback_champion_graft_donor_scope_ranks_records_and_round_trips():
    state = FeedbackState(champion_version_id="v3")
    base = _case(1)
    donor_a = ChampionSeed(
        case_id="donor-a",
        version_id="v1",
        bug_family_keys=("family-a",),
        case_payload=_case(11).to_dict(),
        stability=2,
    )
    donor_b = ChampionSeed(
        case_id="donor-b",
        version_id="v2",
        bug_family_keys=("family-b",),
        case_payload=_case(12).to_dict(),
        stability=2,
    )
    context = state._champion_graft_donor_context_features(
        base,
        target_keys=["semantic_family:filtering"],
    )
    state.adaptive_learning.record_outcome(
        "champion_graft_donor",
        "donor-b",
        context_features=context,
        version_id="v3",
        reward=4.0,
    )

    ordered, selection = state._choose_champion_graft_donors(
        base,
        [donor_a, donor_b],
        target_keys=["semantic_family:filtering"],
    )
    state.last_candidate_metadata = {
        "champion_graft_selection": selection,
    }
    state.record_feedback_outcome_reward("feedback_mutation", 2.0)

    assert ordered[0].case_id == "donor-b"
    assert selection["scope"] == "champion_graft_donor"
    assert selection["selected_donor_case_id"] == "donor-b"
    assert state.adaptive_learning.bandits["champion_graft_donor"].arms["donor-b"].pulls == 2

    restored = FeedbackState.from_state_dict(
        state.to_state_dict(),
        persist_to_disk=False,
        max_persisted=0,
        max_cases_per_profile=16,
        source_scheduler=None,
    )

    assert restored.enable_champion_graft_donor_bandit is True
    assert restored.adaptive_learning.bandits["champion_graft_donor"].arms["donor-b"].pulls == 2


def test_feedback_champion_graft_donor_scope_can_be_disabled_for_ablation():
    state = FeedbackState(enable_champion_graft_donor_bandit=False)
    donor_a = ChampionSeed(
        case_id="donor-a",
        version_id="v1",
        bug_family_keys=("family-a",),
        case_payload=_case(11).to_dict(),
        stability=2,
    )
    donor_b = ChampionSeed(
        case_id="donor-b",
        version_id="v2",
        bug_family_keys=("family-b",),
        case_payload=_case(12).to_dict(),
        stability=2,
    )

    ordered, selection = state._choose_champion_graft_donors(
        _case(1),
        [donor_a, donor_b],
        target_keys=[],
    )
    state.last_candidate_metadata = {
        "champion_graft_selection": {
            "selected_donor_case_id": "donor-a",
            "context_features": [],
        }
    }
    state.record_feedback_outcome_reward("feedback_mutation", 2.0)

    assert [donor.case_id for donor in ordered] == ["donor-a", "donor-b"]
    assert selection == {}
    assert "champion_graft_donor" not in state.adaptive_learning.bandits
    restored = FeedbackState.from_state_dict(
        state.to_state_dict(),
        persist_to_disk=False,
        max_persisted=0,
        max_cases_per_profile=16,
        source_scheduler=None,
    )
    assert restored.enable_champion_graft_donor_bandit is False


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

    def tracked_mutation(
        case,
        seed,
        *,
        allow_probe_operators=True,
        operator_scores=None,
        target_keys=None,
        plan_depth=None,
        disagreement=None,
        operator_pulls=None,
        recent_operator_pulls=None,
    ):
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

    selected = state.select_case(8, _case(8))

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


def test_seed_frontier_priority_matches_frontier_row_fields():
    state = FeedbackState(enable_seed_energy_tier_bandit=False)
    assert state.record(_case(1), "0000000000000001", False, target_keys=["target:common"])
    assert state.record(_case(2), "0000000000000002", False, target_keys=["target:rare"])
    index = 1

    row = state._seed_frontier_row(index)
    priority = state._seed_frontier_priority(index)

    assert priority == (
        -float(row["schedule_score"]),
        -float(row["reward_prior"]),
        -float(row["target_novelty_score"]),
        int(row["mutation_pulls"]),
        int(row["recent_parent_pulls"]) + int(row["recent_cluster_pulls"]),
        int(row["index"]),
    )


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
    assert restored.case_behavioral_descriptors == state.case_behavioral_descriptors
    assert restored.stored_cluster_keys == state.stored_cluster_keys
    assert restored.cluster_schedule_feedback_totals == state.cluster_schedule_feedback_totals
    assert restored.cluster_schedule_feedback_counts == state.cluster_schedule_feedback_counts
    assert restored.quality_archive.to_state_dict() == state.quality_archive.to_state_dict()
    assert restored._recent_cluster_pull_count(cluster_key) == 1


def test_feedback_lineage_backfills_and_round_trips_from_seed_metadata():
    state = FeedbackState()
    root = _case(1)
    child = _case(2)
    child.metadata["seed_lineage"] = {
        "root_seed": 1,
        "parent_index": 0,
        "parent_seed": 1,
        "parent_case_id": "case-1",
        "mutation_seed": 2,
        "depth": 1,
    }

    assert state.record(root, "0000000000000001", False)
    assert state.record(child, "0000000000000002", False)
    restored = FeedbackState.from_state_dict(
        state.to_state_dict(),
        persist_to_disk=False,
        max_persisted=0,
        max_cases_per_profile=16,
        source_scheduler=None,
    )

    assert restored.lineage.nodes[1].parent_index == 0
    assert 1 in restored.lineage.nodes[0].children
    assert restored._seed_frontier_row(1)["lineage_rarity"] == restored.lineage.rarity_score(1)


def test_feedback_can_disable_lineage_rarity_seed_frontier_signal():
    state = FeedbackState(enable_lineage_rarity=False)
    root = _case(1)
    child = _case(2)
    child.metadata["seed_lineage"] = {
        "root_seed": 1,
        "parent_index": 0,
        "parent_seed": 1,
        "parent_case_id": "case-1",
        "mutation_seed": 2,
        "depth": 1,
    }

    assert state.record(root, "0000000000000001", False)
    assert state.record(child, "0000000000000002", False)

    assert state.lineage.rarity_score(1) > 0.0
    assert state._seed_frontier_row(1)["lineage_rarity"] == 0.0


def test_feedback_lineage_reward_cools_successful_sibling_branch():
    state = FeedbackState()
    root = _case(1)
    left = _case(2)
    right = _case(3)
    left.metadata["seed_lineage"] = {"parent_index": 0, "depth": 1}
    right.metadata["seed_lineage"] = {"parent_index": 0, "depth": 1}

    assert state.record(root, "0000000000000001", False)
    assert state.record(left, "0000000000000002", False)
    assert state.record(right, "0000000000000003", False)
    before = state.lineage.rarity_score(2)
    state.last_feedback_parent_index = 1
    state.record_feedback_outcome_reward("feedback_mutation", 3.0)

    assert state.lineage.nodes[1].pulls == 1
    assert state.lineage.nodes[1].reward_total == 3.0
    assert state.lineage.rarity_score(2) < before


def test_feedback_discovery_rate_estimator_updates_bandit_exploration_weight():
    state = FeedbackState()
    state.adaptive_learning.score_action("semantic_objective", "objective-a")
    initial_weight = state.adaptive_learning.bandits["semantic_objective"].exploration_weight

    for index in range(4):
        assert state.record(
            _case(index + 1),
            f"{index + 1:016x}",
            True,
            candidate_bug_families=[f"family:{index}"],
        )

    increased = state.adaptive_learning.bandits["semantic_objective"].exploration_weight
    restored = FeedbackState.from_state_dict(
        state.to_state_dict(),
        persist_to_disk=False,
        max_persisted=0,
        max_cases_per_profile=16,
        source_scheduler=None,
    )

    assert increased > initial_weight
    assert restored.discovery_rate_estimator.unique_family_count == 4
    assert restored.adaptive_learning.bandits["semantic_objective"].exploration_weight == increased


def test_feedback_bayesian_exploration_can_be_disabled_for_ablation():
    state = FeedbackState(enable_bayesian_exploration=False)
    state.adaptive_learning.score_action("semantic_objective", "objective-a")
    initial_weight = state.adaptive_learning.bandits["semantic_objective"].exploration_weight

    for index in range(4):
        assert state.record(
            _case(index + 1),
            f"{index + 1:016x}",
            True,
            candidate_bug_families=[f"family:{index}"],
        )

    restored = FeedbackState.from_state_dict(
        state.to_state_dict(),
        persist_to_disk=False,
        max_persisted=0,
        max_cases_per_profile=16,
        source_scheduler=None,
    )

    assert state.adaptive_learning.bandits["semantic_objective"].exploration_weight == initial_weight
    assert restored.enable_bayesian_exploration is False
    assert restored.adaptive_learning.bandits["semantic_objective"].exploration_weight == initial_weight


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
    assert restored.case_behavioral_descriptors[0]["target_class_axis"] == "target:semantic_family_null_semantics"


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


def test_feedback_hierarchical_archive_round_trips_and_can_be_disabled():
    state = FeedbackState(enable_hierarchical_archive=False)

    restored = FeedbackState.from_state_dict(
        state.to_state_dict(),
        persist_to_disk=False,
        max_persisted=0,
        max_cases_per_profile=16,
        source_scheduler=None,
    )

    assert state.quality_archive.enable_hierarchical is False
    assert restored.enable_hierarchical_archive is False
    assert restored.quality_archive.enable_hierarchical is False
    assert restored.to_state_dict()["enable_hierarchical_archive"] is False


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
    assert known_context["archive_axis_reward"] > 0.0
    assert known_context["behavioral_descriptor"]["target_class_axis"] == "target:semantic_family_cast_semantics"
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


def test_feedback_bd_axis_bandit_records_axis_rewards_and_round_trips():
    state = FeedbackState()
    assert state.record(
        _case(1),
        "0000000000000001",
        False,
        target_keys=["semantic_family:cast_semantics"],
    )
    state.last_feedback_parent_index = 0

    state.record_feedback_outcome_reward("feedback_mutation", 3.0)

    assert "bd_axis_weights" in state.adaptive_learning.bandits
    bandit = state.adaptive_learning.bandits["bd_axis_weights"]
    assert bandit.arms["bd_profile"].pulls == 1
    assert bandit.arms["bd_target_class"].pulls == 1
    assert bandit.arms["bd_backend_disagreement"].pulls == 1
    assert bandit.total_pulls == len(state.case_behavioral_descriptors[0]["axis_tuples"])
    weights = state._bd_axis_weights_for_seed(0)
    assert set(weights) == {axis for axis, _value in state.case_behavioral_descriptors[0]["axis_tuples"]}
    assert max(weights.values()) > 1.0
    context = state.candidate_quality_context(
        _case(2),
        target_keys=["semantic_family:cast_semantics"],
    )
    assert set(context["archive_axis_weights"]) == set(weights)
    assert max(context["archive_axis_weights"].values()) > 1.0

    restored = FeedbackState.from_state_dict(
        state.to_state_dict(),
        persist_to_disk=False,
        max_persisted=0,
        max_cases_per_profile=16,
        source_scheduler=None,
    )

    assert restored.enable_bd_axis_bandit is True
    assert restored.adaptive_learning.bandits["bd_axis_weights"].arms["bd_target_class"].pulls == 1
    assert restored._bd_axis_weights_for_seed(0) == weights


def test_feedback_bd_axis_bandit_can_be_disabled_for_ablation():
    state = FeedbackState(enable_bd_axis_bandit=False)
    assert state.record(
        _case(1),
        "0000000000000001",
        False,
        target_keys=["semantic_family:cast_semantics"],
    )
    state.last_feedback_parent_index = 0

    state.record_feedback_outcome_reward("feedback_mutation", 3.0)

    assert "bd_axis_weights" not in state.adaptive_learning.bandits
    assert state._bd_axis_weights_for_seed(0) == {}
    restored = FeedbackState.from_state_dict(
        state.to_state_dict(),
        persist_to_disk=False,
        max_persisted=0,
        max_cases_per_profile=16,
        source_scheduler=None,
    )
    assert restored.enable_bd_axis_bandit is False


def test_feedback_bd_axis_scores_are_cached_by_learning_and_context_revision(monkeypatch):
    state = FeedbackState()
    assert state.record(
        _case(1),
        "0000000000000001",
        False,
        target_keys=["semantic_family:cast_semantics"],
    )
    state.last_feedback_parent_index = 0
    state.record_feedback_outcome_reward("feedback_mutation", 3.0)

    score_calls = 0
    learning_type = type(state.adaptive_learning)
    original_score_action = learning_type.score_action

    def tracked_score_action(self, scope, action_id, **kwargs):
        nonlocal score_calls
        if scope == "bd_axis_weights":
            score_calls += 1
        return original_score_action(self, scope, action_id, **kwargs)

    monkeypatch.setattr(learning_type, "score_action", tracked_score_action)

    first = state._bd_axis_weights_for_seed(0)
    cold_calls = score_calls
    assert state._bd_axis_weights_for_seed(0) == first
    state._record_mutation_seed_pull(0)
    warm = state._bd_axis_weights_for_seed(0)
    warm_calls = score_calls
    state._record_mutation_seed_pull(0)
    assert state._bd_axis_weights_for_seed(0) == warm

    assert cold_calls > 0
    assert warm_calls > cold_calls
    assert score_calls == warm_calls


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
    assert 0.8 < score < 1.1
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


def test_feedback_value_catalog_learning_records_used_entries_and_round_trips():
    state = FeedbackState()
    assert state.record(
        _case(1),
        "0000000000000001",
        False,
        target_keys=["semantic_family:numeric_semantics"],
    )
    state.last_feedback_parent_index = 0
    state.last_feedback_operator = "value"
    state.last_candidate_metadata = {
        "mutation": {
            "operator": "value",
            "value_catalog_entries": [
                {"entry_id": "int.zero", "column": "x"},
                {"entry_id": "int.zero", "column": "x"},
            ],
        },
        "disagreement_descriptor": {
            "column_classes": {"x": "numeric"},
            "mismatch_class": "value",
        },
    }

    state.record_feedback_outcome_reward("feedback_mutation", 3.0)

    assert "value_catalog_entry" in state.adaptive_learning.bandits
    assert state.adaptive_learning.bandits["value_catalog_entry"].arms["int.zero"].pulls == 1

    restored = FeedbackState.from_state_dict(
        state.to_state_dict(),
        persist_to_disk=False,
        max_persisted=0,
        max_cases_per_profile=16,
        source_scheduler=None,
    )
    scores = restored._value_catalog_score_snapshot(
        target_keys=["semantic_family:numeric_semantics"],
        disagreement={"column_classes": {"x": "numeric"}, "mismatch_class": "value"},
    )

    assert restored.enable_value_catalog is True
    assert scores["int.zero"] > scores["int.neg_one"]


def test_feedback_value_catalog_can_be_disabled_for_ablation():
    state = FeedbackState(enable_value_catalog=False)
    assert state.record(_case(1), "0000000000000001", False)
    state.last_feedback_parent_index = 0
    state.last_feedback_operator = "value"
    state.last_candidate_metadata = {
        "mutation": {
            "operator": "value",
            "value_catalog_entries": [{"entry_id": "int.zero", "column": "x"}],
        },
    }

    state.record_feedback_outcome_reward("feedback_mutation", 3.0)

    assert "value_catalog_entry" not in state.adaptive_learning.bandits
    assert state._value_catalog_score_snapshot(target_keys=["semantic_family:numeric_semantics"]) == {}
    restored = FeedbackState.from_state_dict(
        state.to_state_dict(),
        persist_to_disk=False,
        max_persisted=0,
        max_cases_per_profile=16,
        source_scheduler=None,
    )
    assert restored.enable_value_catalog is False


def test_feedback_operator_swarm_updates_round_trips_and_records_bandit_scope():
    state = FeedbackState()
    assert state.record(
        _case(1),
        "0000000000000001",
        False,
        target_keys=["semantic_family:filtering"],
    )
    state.last_feedback_parent_index = 0
    state.last_feedback_operator = "append_range_filter"
    state.last_feedback_swarm_particle_id = 0

    state.record_feedback_outcome_reward("feedback_mutation", 3.0)

    assert state.operator_swarm.operator_pulls["append_range_filter"] == 1
    assert state.operator_swarm.select_particle(0).pulls == 1
    assert "mutation_operator_swarm" in state.adaptive_learning.bandits
    assert state.adaptive_learning.bandits["mutation_operator_swarm"].arms["0"].pulls == 1

    restored = FeedbackState.from_state_dict(
        state.to_state_dict(),
        persist_to_disk=False,
        max_persisted=0,
        max_cases_per_profile=16,
        source_scheduler=None,
    )

    assert restored.last_feedback_swarm_particle_id == 0
    assert restored.operator_swarm.operator_pulls == state.operator_swarm.operator_pulls
    assert restored.operator_swarm.operator_rewards == state.operator_swarm.operator_rewards
    assert restored.operator_swarm.select_particle(0).pulls == state.operator_swarm.select_particle(0).pulls
    assert abs(
        restored.operator_swarm.select_particle(0).weights["append_range_filter"]
        - state.operator_swarm.select_particle(0).weights["append_range_filter"]
    ) < 1e-12


def test_feedback_operator_swarm_score_snapshot_biases_selected_particle():
    state = FeedbackState()
    baseline = state._mutation_operator_score_snapshot()["append_range_filter"]

    for _ in range(12):
        state.operator_swarm.update(0, 3.0, operator="append_range_filter")

    scores = state._mutation_operator_score_snapshot(swarm_particle_id=0)

    assert scores["append_range_filter"] > baseline


def test_feedback_operator_swarm_can_be_disabled_for_ablation():
    state = FeedbackState(enable_operator_swarm=False)
    assert state.record(_case(1), "0000000000000001", False)
    state.last_feedback_parent_index = 0
    state.last_feedback_operator = "append_range_filter"
    state.last_feedback_swarm_particle_id = 0

    state.record_feedback_outcome_reward("feedback_mutation", 3.0)

    assert state.operator_swarm.operator_pulls.get("append_range_filter", 0) == 0
    assert "mutation_operator_swarm" not in state.adaptive_learning.bandits
    restored = FeedbackState.from_state_dict(
        state.to_state_dict(),
        persist_to_disk=False,
        max_persisted=0,
        max_cases_per_profile=16,
        source_scheduler=None,
    )
    assert restored.enable_operator_swarm is False


def test_feedback_canonical_methods_are_available():
    state = FeedbackState()
    generated = _case(7)

    selected = state.select_case(7, generated)
    assert selected == generated
    assert (
        state.record_candidate_outcome(
            "generated",
            has_finding=False,
            is_new_behavior=False,
            preflight={"valid": True, "fallback_used": False},
        )
        is None
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


def test_feedback_seed_energy_is_visible_and_rewards_productive_diverse_seed():
    state = FeedbackState()
    plain = _case(1)
    rich = _case(2)

    assert state.record(plain, "0000000000000001", False, target_keys=["target:common"])
    assert state.record(
        rich,
        "0000000000000002",
        False,
        target_keys=[
            "semantic_family:window_semantics",
            "semantic_signal:null_boundary",
            "exploration_objective:composite_comparison",
        ],
    )
    rich_index = state.interesting_cases.index(rich)
    state.last_feedback_parent_index = rich_index
    state.record_feedback_candidate_reward("feedback_mutation", 3.0)

    plain_energy = state._case_seed_energy(state.interesting_cases.index(plain))
    rich_energy = state._case_seed_energy(rich_index)
    row = state._seed_frontier_row(rich_index)
    snapshot = state._feedback_decision_snapshot(
        rich_index,
        mutation_seed=99,
        attempt=0,
        operator_scores={},
        plan_depth=1,
    )

    assert rich_energy >= plain_energy
    assert row["seed_energy"] == rich_energy
    assert snapshot["seed_energy"] == rich_energy
    assert snapshot["frontier_head"][0]["seed_energy"] >= 1


def test_feedback_semantic_plan_qd_protects_cold_strata_in_schedule_and_energy(monkeypatch):
    state = FeedbackState(corpus_mode="semantic_plan_qd")
    cold = _semantic_plan_case(1, "sort")
    warm = _semantic_plan_case(2, "hash_join")
    assert state.record(cold, "0000000000000101", True)
    assert state.record(warm, "0000000000000102", True)
    cold_index = state.interesting_cases.index(cold)
    warm_index = state.interesting_cases.index(warm)
    monkeypatch.setattr(
        type(state.seed_budget_allocator),
        "energy_for_seed",
        lambda self, index, **kwargs: 4,
    )

    state._record_mutation_seed_pull(warm_index)
    state._record_mutation_seed_pull(warm_index)
    cold_trace = state._case_seed_energy_decision(cold_index)
    warm_trace = state._case_seed_energy_decision(warm_index)
    snapshot = state._feedback_decision_snapshot(
        cold_index,
        mutation_seed=101,
        attempt=0,
        operator_scores={},
        plan_depth=1,
    )

    assert cold_trace["cold_debt"] > warm_trace["cold_debt"] == 0
    assert cold_trace["adjusted_energy"] > warm_trace["adjusted_energy"]
    assert state._case_seed_schedule_score(cold_index) > state._case_seed_schedule_score(warm_index)
    assert snapshot["quality_diversity_energy_decision"] == cold_trace
    assert snapshot["cold_stratum_debt"] == cold_trace["cold_debt"]
    assert snapshot["energy_audit_floor"] > 0.0


def test_feedback_semantic_plan_qd_round_trips_and_rebuilds_on_mode_change():
    state = FeedbackState(corpus_mode="semantic_plan_qd")
    case = _semantic_plan_case(3, "aggregate")
    assert state.record(case, "0000000000000103", True)
    payload = state.to_state_dict()

    restored = FeedbackState.from_state_dict(
        payload,
        persist_to_disk=False,
        max_persisted=0,
        max_cases_per_profile=16,
        source_scheduler=None,
    )
    legacy = FeedbackState.from_state_dict(
        payload,
        persist_to_disk=False,
        max_persisted=0,
        max_cases_per_profile=16,
        source_scheduler=None,
        corpus_mode="legacy_qd",
    )

    assert payload["corpus_mode"] == "semantic_plan_qd"
    assert restored.corpus_mode == "semantic_plan_qd"
    assert dict(restored.case_behavioral_descriptors[0]["axis_tuples"])["bd_plan"].startswith("plan:")
    assert legacy.corpus_mode == "legacy_qd"
    assert "bd_plan" not in dict(legacy.case_behavioral_descriptors[0]["axis_tuples"])


def test_feedback_saturated_family_energy_trace_preserves_audit_opportunity():
    state = FeedbackState(
        corpus_mode="semantic_plan_qd",
        max_cases_per_candidate_family=1,
    )
    case = _semantic_plan_case(4, "limit")
    assert state.record(
        case,
        "0000000000000104",
        True,
        candidate_bug_families=["root:known-family"],
    )
    state._record_mutation_seed_pull(0)
    state._record_mutation_seed_pull(0)

    trace = state._case_seed_energy_decision(0)

    assert trace["saturated_family"] is True
    assert trace["saturation_multiplier"] < 1.0
    assert trace["effective_energy_multiplier"] >= trace["audit_floor"]
    assert trace["adjusted_energy"] >= 1


def test_feedback_seed_energy_tier_scope_learns_energy_multiplier_and_round_trips():
    state = FeedbackState()
    assert state.record(_case(1), "0000000000000001", False)
    state.adaptive_learning.record_outcome(
        "seed_energy_tier",
        "high",
        context_features=state._seed_energy_tier_context_features(0),
        reward=3.0,
    )

    energy = state._case_seed_energy(0)
    snapshot = state._feedback_decision_snapshot(
        0,
        mutation_seed=99,
        attempt=0,
        operator_scores={},
        plan_depth=1,
    )
    state.last_feedback_parent_index = 0
    state.last_feedback_decision = snapshot
    state.record_feedback_outcome_reward("feedback_mutation", 2.0)

    assert snapshot["seed_energy_tier"] == "high"
    assert energy >= 2
    assert state.adaptive_learning.bandits["seed_energy_tier"].arms["high"].pulls == 2

    restored = FeedbackState.from_state_dict(
        state.to_state_dict(),
        persist_to_disk=False,
        max_persisted=0,
        max_cases_per_profile=16,
        source_scheduler=None,
    )

    assert restored.enable_seed_energy_tier_bandit is True
    assert restored.adaptive_learning.bandits["seed_energy_tier"].arms["high"].pulls == 2


def test_feedback_seed_energy_tier_scope_can_be_disabled_for_ablation():
    state = FeedbackState(enable_seed_energy_tier_bandit=False)
    assert state.record(_case(1), "0000000000000001", False)
    state.last_feedback_parent_index = 0
    state.last_feedback_decision = {
        "seed_energy_tier": "high",
        "seed_energy_tier_context": list(state._seed_energy_tier_context_features(0)),
    }

    state.record_feedback_outcome_reward("feedback_mutation", 2.0)

    assert "seed_energy_tier" not in state.adaptive_learning.bandits
    assert state._choose_seed_energy_tier(0) == "med"
    restored = FeedbackState.from_state_dict(
        state.to_state_dict(),
        persist_to_disk=False,
        max_persisted=0,
        max_cases_per_profile=16,
        source_scheduler=None,
    )
    assert restored.enable_seed_energy_tier_bandit is False


def test_feedback_select_case_batch_uses_seed_energy(monkeypatch):
    state = FeedbackState()
    base = _case(1)
    generated = _case(9)
    assert state.record(base, "0000000000000001", False)

    monkeypatch.setattr(FeedbackState, "_case_seed_energy", lambda self, index: 3)

    def changed_mutation(case, seed, **kwargs):
        mutated = _case(seed)
        mutated.case_id = f"{case.case_id}-mut-{seed}"
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
                "detail": f"value:{seed}",
                "changed": True,
            },
            "mutation_plan": {
                "strategy": "test",
                "planned_depth": 1,
                "executed_depth": 1,
                "changed": True,
            },
        }
        return SimpleNamespace(case=mutated, metadata=metadata)

    monkeypatch.setattr(feedback, "mutate_case_with_metadata", changed_mutation)

    batch = state.select_case_batch(6, generated, max_batch=3)

    assert [case.case_id for case in batch] == [
        "case-1-mut-6",
        "case-1-mut-7",
        "case-1-mut-8",
    ]
    assert state.last_candidate_source == "feedback_mutation"
    assert state.last_feedback_parent_index == 0
    assert state.last_feedback_operator == "value"
    assert len(state.last_candidate_batch_metadata) == 3
    assert state.case_mutation_pulls[0] == 3


def test_feedback_seed_energy_batch_reuses_selection_frontier_snapshot(monkeypatch):
    state = FeedbackState()
    base = _case(1)
    assert state.record(base, "0000000000000001", False)
    monkeypatch.setattr(FeedbackState, "_case_seed_energy", lambda self, index: 3)

    snapshot_calls = 0
    original_snapshot = FeedbackState._seed_frontier_snapshot

    def tracked_snapshot(self, **kwargs):
        nonlocal snapshot_calls
        snapshot_calls += 1
        return original_snapshot(self, **kwargs)

    def changed_mutation(case, seed, **kwargs):
        mutated = _case(seed)
        mutated.case_id = f"{case.case_id}-mut-{seed}"
        return SimpleNamespace(
            case=mutated,
            metadata={
                "candidate_source": "feedback_mutation",
                "seed_lineage": {
                    "root_seed": case.seed,
                    "parent_seed": case.seed,
                    "parent_case_id": case.case_id,
                    "mutation_seed": seed,
                    "depth": 1,
                },
                "mutation": {"operator": "value", "changed": True},
                "mutation_plan": {"changed": True, "steps": []},
            },
        )

    monkeypatch.setattr(FeedbackState, "_seed_frontier_snapshot", tracked_snapshot)
    monkeypatch.setattr(feedback, "mutate_case_with_metadata", changed_mutation)

    batch = state.select_case_batch(6, _case(9), max_batch=3)

    assert len(batch) == 3
    assert snapshot_calls == 1
    assert {
        metadata["feedback_decision"]["frontier_snapshot_timing"]
        for metadata in state.last_candidate_batch_metadata
    } == {"selection"}


def test_feedback_select_case_batch_can_disable_seed_energy_batch(monkeypatch):
    state = FeedbackState(enable_seed_energy_batch=False)
    base = _case(1)
    generated = _case(9)
    assert state.record(base, "0000000000000001", False)

    monkeypatch.setattr(FeedbackState, "_case_seed_energy", lambda self, index: 3)

    def changed_mutation(case, seed, **kwargs):
        mutated = _case(seed)
        mutated.case_id = f"{case.case_id}-mut-{seed}"
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
                "detail": f"value:{seed}",
                "changed": True,
            },
            "mutation_plan": {
                "strategy": "test",
                "planned_depth": 1,
                "executed_depth": 1,
                "changed": True,
            },
        }
        return SimpleNamespace(case=mutated, metadata=metadata)

    monkeypatch.setattr(feedback, "mutate_case_with_metadata", changed_mutation)

    batch = state.select_case_batch(6, generated, max_batch=3, enqueue_remaining=True)

    assert [case.case_id for case in batch] == ["case-1-mut-6"]
    assert state.last_candidate_source == "feedback_mutation"
    assert state.last_feedback_parent_index == 0
    assert len(state.last_candidate_batch_metadata) == 1
    assert not state.pending_candidate_batch
    assert state.case_mutation_pulls[0] == 1


def test_feedback_operator_score_snapshot_cools_recently_reused_operator():
    state = FeedbackState()
    for _ in range(12):
        state.mutation_operator_pulls["append_range_filter"] += 1
        state.mutation_operator_rewards["append_range_filter"] += 0.5
        state.recent_mutation_operators.append("append_range_filter")

    scores = state._mutation_operator_score_snapshot()

    assert scores["append_range_filter"] < scores["__untried__"]


def test_feedback_operator_learning_bonus_cache_survives_recent_pull(monkeypatch):
    state = FeedbackState()
    learning_context = state._mutation_operator_learning_context_features([])
    state.adaptive_learning.record_outcome(
        "mutation_operator",
        "append_range_filter",
        context_features=learning_context,
        reward=3.0,
    )
    score_calls = 0
    learning_type = type(state.adaptive_learning)
    original_score_action = learning_type.score_action

    def tracked_score_action(self, scope, action_id, **kwargs):
        nonlocal score_calls
        if scope == "mutation_operator":
            score_calls += 1
        return original_score_action(self, scope, action_id, **kwargs)

    monkeypatch.setattr(learning_type, "score_action", tracked_score_action)

    first = state._mutation_operator_score_snapshot()
    initial_score_calls = score_calls
    state._record_recent_operator_pull("append_range_filter")
    second = state._mutation_operator_score_snapshot()

    assert initial_score_calls > 0
    assert score_calls == initial_score_calls
    assert second["append_range_filter"] < first["append_range_filter"]


def test_feedback_value_catalog_cache_is_not_invalidated_by_recent_operator(monkeypatch):
    state = FeedbackState()
    state.adaptive_learning.record_outcome(
        "value_catalog_entry",
        "int.zero",
        context_features=(),
        reward=3.0,
    )
    score_calls = 0
    learning_type = type(state.adaptive_learning)
    original_score_action = learning_type.score_action

    def tracked_score_action(self, scope, action_id, **kwargs):
        nonlocal score_calls
        if scope == "value_catalog_entry":
            score_calls += 1
        return original_score_action(self, scope, action_id, **kwargs)

    monkeypatch.setattr(learning_type, "score_action", tracked_score_action)

    first = state._value_catalog_score_snapshot(target_keys=["semantic_family:numeric_semantics"])
    initial_score_calls = score_calls
    state._record_recent_operator_pull("append_range_filter")
    second = state._value_catalog_score_snapshot(target_keys=["semantic_family:numeric_semantics"])
    state._invalidate_mutation_score_caches()
    state._value_catalog_score_snapshot(target_keys=["semantic_family:numeric_semantics"])

    assert initial_score_calls > 0
    assert second == first
    assert score_calls == 2 * initial_score_calls


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


def test_feedback_mutation_parent_selection_uses_global_frontier_not_local_window():
    state = FeedbackState()
    for seed in range(20):
        assert state.record(_case(seed), f"{seed + 1:016x}", False)
    state.case_schedule_rewards[19] = 5.0

    selected_index = state._choose_mutation_seed_index(0)

    assert selected_index == 19


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


def test_feedback_quota_replacement_preserves_rare_cluster_singleton():
    state = FeedbackState(max_corpus=3, max_cases_per_profile=0)
    common_low = _case(1)
    rare = _case(2)
    common_mid = _case(3)
    common_high = Case(
        "case-common-high",
        4,
        [
            TableData("t0", [ColumnSpec("id", "int"), ColumnSpec("g", "str"), ColumnSpec("x", "int")], [{"id": 1, "g": None, "x": 1}]),
            TableData("t1", [ColumnSpec("id", "int")], [{"id": 1}]),
        ],
        Program(
            "prog-common-high",
            4,
            [
                {"op": "join", "table": "t1", "left_on": "id", "right_on": "id", "how": "left"},
                {"op": "filter", "column": "x", "cmp": ">", "value": 0},
                {"op": "mutate", "column": "x2", "expr": {"kind": "add_const", "source": "x", "value": 1}},
                {"op": "groupby", "keys": ["g"], "aggs": [{"column": "x2", "func": "count", "as": "count_x"}]},
            ],
        ),
    )

    assert state.record(common_low, "0000000000000001", False, target_keys=["semantic_family:common"])
    assert state.record(rare, "0000000000000002", False, target_keys=["semantic_family:rare"])
    assert state.record(common_mid, "0000000000000003", False, target_keys=["semantic_family:common"])
    assert state.record(common_high, "0000000000000004", False, target_keys=["semantic_family:common"])

    case_ids = [case.case_id for case in state.interesting_cases]
    assert "case-1" not in case_ids
    assert "case-2" in case_ids
    assert "case-common-high" in case_ids
    rare_index = case_ids.index("case-2")
    assert "semantic_family_rare" in state.case_cluster_keys[rare_index]
    assert state.stored_cluster_keys[state.case_cluster_keys[rare_index]] == 1


def test_feedback_seed_quota_round_trips_and_can_be_disabled():
    state = FeedbackState(enable_seed_quota=False)

    restored = FeedbackState.from_state_dict(
        state.to_state_dict(),
        persist_to_disk=False,
        max_persisted=0,
        max_cases_per_profile=16,
        source_scheduler=None,
    )

    assert restored.enable_seed_quota is False
    assert restored.seed_eviction_policy.enabled is False


def test_feedback_seed_energy_batch_round_trips_and_can_be_disabled():
    state = FeedbackState(enable_seed_energy_batch=False)

    restored = FeedbackState.from_state_dict(
        state.to_state_dict(),
        persist_to_disk=False,
        max_persisted=0,
        max_cases_per_profile=16,
        source_scheduler=None,
    )

    assert restored.enable_seed_energy_batch is False
    assert restored.to_state_dict()["enable_seed_energy_batch"] is False


def test_feedback_per_operator_energy_round_trips_and_can_be_disabled():
    state = FeedbackState(enable_per_operator_energy=False)

    restored = FeedbackState.from_state_dict(
        state.to_state_dict(),
        persist_to_disk=False,
        max_persisted=0,
        max_cases_per_profile=16,
        source_scheduler=None,
    )

    assert restored.enable_per_operator_energy is False
    assert restored.to_state_dict()["enable_per_operator_energy"] is False


def test_feedback_ir_rewrite_mutations_round_trip_and_filter_operator_profiles():
    state = FeedbackState(enable_ir_rewrite_mutations=False)
    state.mutation_operator_rewards["ir_swap_adjacent"] = 10.0
    state.mutation_operator_pulls["ir_swap_adjacent"] = 1
    state.recent_mutation_operator_counts["ir_pushdown_filter"] = 1

    scores = state._mutation_operator_score_snapshot()

    assert state.enable_ir_rewrite_mutations is False
    assert "ir_swap_adjacent" not in scores
    assert "ir_pushdown_filter" not in scores
    assert "ir_swap_adjacent" not in state.operator_swarm.operator_names
    assert "ir_pushdown_filter" not in state.operator_swarm.operator_names

    restored = FeedbackState.from_state_dict(
        state.to_state_dict(),
        persist_to_disk=False,
        max_persisted=0,
        max_cases_per_profile=16,
        source_scheduler=None,
    )

    assert restored.enable_ir_rewrite_mutations is False
    assert restored.to_state_dict()["enable_ir_rewrite_mutations"] is False
    assert "ir_swap_adjacent" not in restored.operator_swarm.operator_names
    assert "ir_pushdown_filter" not in restored.operator_swarm.operator_names


def test_feedback_shrink_mutations_round_trip_and_filter_operator_swarm():
    state = FeedbackState(enable_shrink_mutations=False)

    assert state.enable_shrink_mutations is False
    assert set(state.operator_swarm.operator_names).isdisjoint(SHRINK_MUTATION_OPERATOR_NAMES)

    restored = FeedbackState.from_state_dict(
        state.to_state_dict(),
        persist_to_disk=False,
        max_persisted=0,
        max_cases_per_profile=16,
        source_scheduler=None,
    )

    assert restored.enable_shrink_mutations is False
    assert restored.to_state_dict()["enable_shrink_mutations"] is False
    assert set(restored.operator_swarm.operator_names).isdisjoint(SHRINK_MUTATION_OPERATOR_NAMES)


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

    first = state.select_case(7, generated)
    assert first.case_id == generated.case_id
    assert state.last_candidate_source == "generated"
    assert state.last_candidate_metadata["seed_lineage"]["root_seed"] == 7
    assert state.last_candidate_metadata["seed_lineage"]["depth"] == 0
    assert state.last_candidate_metadata["mutation"]["operator"] == "generated"

    generated_reward = state.record_candidate_outcome(
        "generated",
        has_finding=False,
        is_new_behavior=False,
        preflight={"valid": True, "fallback_used": False},
    )
    assert generated_reward == -0.1
    assert state.last_source_reward == -0.1

    second = state.select_case(8, generated)
    selected_mutation_seed = state.last_candidate_metadata["seed_lineage"]["mutation_seed"]
    assert 8 <= selected_mutation_seed <= 12
    assert second.case_id.endswith(f"-mut-{selected_mutation_seed}")
    assert second.seed == selected_mutation_seed
    assert state.last_candidate_source == "feedback_mutation"
    assert state.last_candidate_metadata["seed_lineage"]["parent_case_id"] == "case-1"
    assert state.last_candidate_metadata["seed_lineage"]["depth"] == 1
    assert state.last_candidate_metadata["mutation"]["operator"] in MUTATION_OPERATOR_NAMES
    decision = state.last_candidate_metadata["feedback_decision"]
    assert decision["parent_case_id"] == "case-1"
    assert decision["mutation_seed"] == selected_mutation_seed
    assert decision["attempt"] == selected_mutation_seed - 8
    assert decision["retention_utility"] >= 1.0
    assert decision["schedule_score"] > 0.0
    assert decision["mutation_pulls"] == 1
    assert decision["frontier_rank"] == 1
    assert decision["frontier_head"][0]["case_id"] == "case-1"
    assert decision["planned_mutation_depth"] >= 1
    assert decision["selected_operator"] == state.last_candidate_metadata["mutation"]["operator"]
    assert decision["mutation_plan"]["changed"] is True
    assert any(step["productive"] for step in decision["mutation_plan"]["steps"])

    feedback_reward = state.record_candidate_outcome(
        "feedback_mutation",
        has_finding=True,
        is_new_behavior=True,
        preflight={"valid": True, "fallback_used": False},
        candidate_bug=True,
    )
    assert feedback_reward == 3.791276923076923
    assert state.last_source_reward == 3.791276923076923

    third = state.select_case(9, generated)
    assert third.case_id.endswith("-mut-9")
    assert state.last_candidate_source == "feedback_mutation"


def test_feedback_select_case_passes_frontier_and_planning_context_to_mutation(monkeypatch):
    scheduler = LocalSourceScheduler(exploration_weight=0.0, min_feedback_share=1.0)
    state = FeedbackState(source_scheduler=scheduler)
    base = _case(1)
    descriptor = {
        "column_classes": {"x": "numeric"},
        "mismatch_class": "value",
        "feature_tokens": ["disagree_class:numeric", "mismatch:value"],
    }
    base.metadata["disagreement_descriptor"] = descriptor
    generated = _case(9)
    assert state.record(
        base,
        "0000000000000001",
        False,
        target_keys=[
            "semantic_signal:left_join_case_when_membership",
            "exploration_objective:cross_model_consistency",
        ],
    )
    scheduler.record_result(
        "generated",
        has_finding=False,
        is_new_behavior=False,
        preflight_valid=True,
        fallback_used=False,
    )
    captured: dict[str, object] = {}

    def planned_mutation(
        case,
        seed,
        *,
        allow_probe_operators=True,
        operator_scores=None,
        target_keys=None,
        plan_depth=None,
        disagreement=None,
        operator_pulls=None,
        recent_operator_pulls=None,
    ):
        captured["case_id"] = case.case_id
        captured["target_keys"] = list(target_keys or [])
        captured["plan_depth"] = plan_depth
        captured["operator_scores"] = dict(operator_scores or {})
        captured["disagreement"] = disagreement
        captured["operator_pulls"] = dict(operator_pulls or {})
        captured["recent_operator_pulls"] = dict(recent_operator_pulls or {})
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
                "multi_step": True,
                "plan_depth": int(plan_depth or 1),
                "executed_steps": int(plan_depth or 1),
            },
            "mutation_plan": {
                "strategy": "adaptive_multi_step_planning",
                "planned_depth": int(plan_depth or 1),
                "executed_depth": int(plan_depth or 1),
                "changed": True,
                "steps": [{"operator": "value", "detail": "value:int:x"}],
            },
        }
        return SimpleNamespace(case=case, metadata=metadata)

    monkeypatch.setattr(feedback, "mutate_case_with_metadata", planned_mutation)

    selected = state.select_case(10, generated)

    assert selected.case_id == "case-1"
    assert captured["case_id"] == "case-1"
    assert captured["plan_depth"] >= 3
    assert captured["target_keys"] == [
        "semantic_signal:left_join_case_when_membership",
        "exploration_objective:cross_model_consistency",
    ]
    assert captured["disagreement"] == descriptor
    assert isinstance(captured["operator_pulls"], dict)
    assert isinstance(captured["recent_operator_pulls"], dict)
    assert state.last_feedback_decision["planned_mutation_depth"] == captured["plan_depth"]
    assert state.last_feedback_decision["frontier_head"][0]["case_id"] == "case-1"
    assert state.last_feedback_decision["mutation_plan"]["planned_depth"] == captured["plan_depth"]


def test_feedback_source_scheduler_does_not_reward_resolved_semantic_divergence():
    scheduler = LocalSourceScheduler(exploration_weight=0.0)
    state = FeedbackState(source_scheduler=scheduler)

    reward = state.record_candidate_outcome(
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

    reward = state.record_candidate_outcome(
        "generated",
        has_finding=True,
        is_new_behavior=False,
        preflight={"valid": True, "fallback_used": False},
        semantic_divergence=True,
    )

    assert reward == 0.058480000000000004
    assert state.last_source_reward == 0.058480000000000004


def test_feedback_mutation_falls_back_to_generated_when_attempts_do_not_change(monkeypatch):
    scheduler = LocalSourceScheduler(exploration_weight=0.0)
    state = FeedbackState(source_scheduler=scheduler, interesting_cases=[_case(1), _case(2)])
    generated = _case(7)

    state.record_candidate_outcome(
        "generated",
        has_finding=False,
        is_new_behavior=False,
        preflight={"valid": True, "fallback_used": False},
    )

    def unchanged_mutation(
        case,
        seed,
        *,
        allow_probe_operators=True,
        operator_scores=None,
        target_keys=None,
        plan_depth=None,
        disagreement=None,
        operator_pulls=None,
        recent_operator_pulls=None,
    ):
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

    selected = state.select_case(8, generated)

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
        selected = state.select_case(seed, generated)
        operator_name = state.last_candidate_metadata["mutation"]["operator"]
        seen.add(operator_name)
        if state.last_candidate_source == "feedback_mutation":
            mutation_seed = state.last_candidate_metadata["seed_lineage"]["mutation_seed"]
            assert selected.case_id.endswith(f"-mut-{mutation_seed}")
            assert mutation_seed >= seed
        assert operator_name not in PROBE_MUTATION_OPERATOR_NAMES
        assert operator_name not in ROOT_TARGETED_MUTATION_OPERATOR_NAMES
        assert operator_name not in SPECIALIZED_DISCOVERY_MUTATION_OPERATOR_NAMES

    assert seen - {"generated"}
