from collections import Counter

import pytest

from datadiff.datagen import generate_case
from datadiff.config import DiscoveryBias
from datadiff.dsl import Case, ColumnSpec, Program, TableData
from datadiff.exploration_objectives import ExplorationObjectiveRule
from datadiff.guidance import (
    GuidanceState,
    _apply_candidate_pool_discovery_balance,
    _candidate_pool_bucket_factors,
    _candidate_pool_discovery_bonus_from_weights,
    _discovery_diversity_bonus,
    _discovery_metrics,
    _discovery_stale_penalty,
    _guidance_reward,
    _recent_discovery_loop_penalty,
    derive_case_features,
    extract_case_features,
    parse_guidance_targets,
)
from datadiff.mutator import mutate_case_with_metadata
from datadiff.semantic_family import derive_semantic_families


def _case(seed: int, operations: list[dict]) -> Case:
    table = TableData(
        name="t0",
        columns=[
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("g", "str", nullable=True),
            ColumnSpec("x", "int", nullable=True),
            ColumnSpec("flag", "bool", nullable=True),
        ],
        rows=[
            {"id": 0, "g": "alpha", "x": 1, "flag": True},
            {"id": 1, "g": "中文", "x": None, "flag": None},
            {"id": 1, "g": "", "x": -1, "flag": False},
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}",
        seed=seed,
        tables=[table],
        program=Program(program_id=f"prog-{seed:08d}", seed=seed, operations=operations),
    )


def _organic_discovery_case_matching_pattern(pattern: str) -> Case:
    for seed in range(260000, 260500):
        case = generate_case(seed, profile="discovery")
        if "source_issue" in case.metadata:
            continue
        if case.metadata.get("mixed_generator_profile") == pattern:
            continue
        if f"pattern:{pattern}" in extract_case_features(case):
            return case
    raise AssertionError(f"no organic discovery case matched pattern:{pattern}")


def _common_api_workflow_case_matching_template(template: str) -> Case:
    for seed in range(0, 2000):
        case = generate_case(seed, profile="common_api_workflow")
        if case.metadata.get("workflow_template") == template:
            return case
    raise AssertionError(f"no common_api_workflow case matched workflow_template:{template}")


def test_extract_case_features_tracks_structure_and_values():
    case = _case(
        1,
        [
            {"op": "filter", "column": "x", "cmp": ">=", "value": 0},
            {"op": "sort", "columns": ["g", "id", "x"], "ascending": False},
        ],
    )

    features = extract_case_features(case)
    assert "op:filter" in features
    assert "cmp:>=" in features
    assert "op:sort" in features
    assert "sort:desc" in features
    assert "combo:filter_sort" in features
    assert "combo_frequency:exploratory" in features
    assert "has:null" in features
    assert "has:unicode_string" in features
    assert "has:empty_string" in features


def test_derive_case_features_matches_compatibility_alias():
    case = _case(99, [{"op": "filter", "column": "x", "cmp": ">=", "value": 0}])

    assert derive_case_features(case) == extract_case_features(case)


def test_guidance_features_bucket_quality_archive_context_without_cluster_identity():
    case = _case(101, [{"op": "filter", "column": "x", "cmp": ">=", "value": 0}])
    case.metadata["quality_archive_context"] = {
        "cluster_key": "profile=generic|targets=semantic_family_cast_semantics|ops=filter",
        "archive_known": True,
        "archive_elite_indexes": [3, 7],
        "archive_seed_count": 2,
        "archive_outcome_count": 5,
        "archive_cluster_reward": 0.8,
        "archive_health_penalty": 0.2,
        "cluster_count": 4,
        "cluster_feedback_reward": -0.2,
        "cluster_novelty_score": 0.1,
        "recent_cluster_pulls": 3,
    }

    features = extract_case_features(case)

    assert "quality_archive:known" in features
    assert "quality_archive:seed_count:few" in features
    assert "quality_archive:outcome_count:some" in features
    assert "quality_archive:cluster_count:few" in features
    assert "quality_archive:recent_cluster_pulls:some" in features
    assert "quality_archive:reward:positive" in features
    assert "quality_archive:feedback_reward:weak_negative" in features
    assert "quality_archive:health:medium" in features
    assert "quality_archive:novelty:low" in features
    assert "quality_archive:elite:present" in features
    assert not any("profile=generic" in feature for feature in features)
    assert not any("semantic_family_cast_semantics" in feature for feature in features)


def test_guidance_matches_dynamic_exploration_objective_rule():
    rule = ExplorationObjectiveRule("adaptive_predicate", exact_features=frozenset({"op:filter"}))
    case = _case(100, [{"op": "filter", "column": "x", "cmp": ">=", "value": 0}])
    guidance = GuidanceState(
        targets=["exploration_objective:adaptive_predicate"],
        exploration_objective_rules=[rule],
    )

    decision = guidance.choose_case([case])

    assert "exploration_objective:adaptive_predicate" in decision.features
    assert decision.matched_targets == ["exploration_objective:adaptive_predicate"]


def test_extract_case_features_tracks_string_contains_filter():
    case = _case(13, [{"op": "filter", "column": "g", "cmp": "str_contains", "value": "a"}])

    features = extract_case_features(case)

    assert "cmp:str_contains" in features
    assert "filter:string-pattern" in features
    assert "filter:string-contains" in features
    assert "pattern:string_contains_filter" in features
    assert "pattern:string_pattern_filter" in features


def test_extract_case_features_tracks_input_materialization_boundaries():
    filter_case = _case(70, [{"op": "filter", "column": "g", "cmp": "in_set", "value": ["alpha"]}])
    drop_case = _case(71, [{"op": "drop_nulls", "columns": ["g", "x"]}])
    fill_case = _case(72, [{"op": "fill_null", "column": "g", "value": "missing"}])
    distinct_case = _case(73, [{"op": "distinct", "columns": ["g", "x"]}])

    filter_features = extract_case_features(filter_case)
    drop_features = extract_case_features(drop_case)
    fill_features = extract_case_features(fill_case)
    distinct_features = extract_case_features(distinct_case)

    assert {"materialization:input", "materialization:filter", "pattern:filter_input_materialization"}.issubset(
        filter_features
    )
    assert {
        "materialization:input",
        "materialization:leading-cleanup",
        "materialization:drop_nulls",
        "pattern:drop_nulls_input_materialization",
    }.issubset(drop_features)
    assert {"materialization:fill_null", "pattern:fill_null_input_materialization"}.issubset(fill_features)
    assert {"materialization:distinct", "pattern:distinct_input_materialization"}.issubset(distinct_features)

    guidance = GuidanceState(targets=["filter_input_materialization", "cleanup_input_materialization"])
    assert guidance.choose_case([filter_case]).matched_targets == ["filter_input_materialization"]
    assert guidance.choose_case([drop_case]).matched_targets == ["cleanup_input_materialization"]


def test_guidance_downweights_no_yield_materialization_target():
    stale = _case(74, [{"op": "drop_nulls", "columns": ["g", "x"]}])
    fresh = _case(75, [{"op": "fill_null", "column": "g", "value": "missing"}])
    guidance = GuidanceState(targets=["drop_nulls_input_materialization", "fill_null_input_materialization"])
    row = {
        "findings": [],
        "is_new_behavior": False,
        "preflight": {"valid": True, "fallback_used": False},
    }

    for _ in range(10):
        guidance.record_result(stale, row)

    stale_decision = guidance._score_case(stale, 2)
    selected = guidance.choose_case([stale, fresh])

    assert stale_decision.score_breakdown["target_no_yield_penalty"] < 0
    assert selected.case is fresh


def test_guidance_matches_canonical_semantic_signal_target_directly():
    case = _common_api_workflow_case_matching_template("sql_left_join_case_membership_groupby")

    guidance = GuidanceState(targets=["semantic_signal:left_join_case_when_membership"])
    decision = guidance.choose_case([case])

    assert "semantic_signal:left_join_case_when_membership" in decision.features
    assert decision.matched_targets == ["semantic_signal:left_join_case_when_membership"]


def test_guidance_state_round_trip_preserves_discovery_and_learning_state():
    case = _case(
        175,
        [{"op": "filter", "column": "x", "cmp": "==", "value": 1}],
    )
    guidance = GuidanceState(
        targets=["filter"],
        known_saturated_bug_families=["grouped_topk_null_sort_key@duckdb"],
        active_backends=["pandas", "duckdb"],
    )
    row = {
        "findings": [
            {
                "kind": "semantic_output_mismatch",
                "root_cause": "grouped_topk_null_sort_key",
                "triage_verdict": "candidate_implementation_bug",
                "suspicious_backends": ["duckdb"],
                "signature": "sig-guidance-1",
            }
        ],
        "is_new_behavior": True,
        "preflight": {"valid": True, "fallback_used": False},
    }

    guidance.record_result(case, row)
    state_dict = guidance.to_state_dict()
    restored = GuidanceState.from_state_dict(
        state_dict,
        targets=["filter"],
        enable_family_saturation=True,
        family_saturation_threshold=8,
        family_saturation_penalty=1.25,
        saturated_family_reward=0.02,
        known_saturated_bug_families=["grouped_topk_null_sort_key@duckdb"],
        issue_replay_saturation_threshold=1,
        issue_replay_saturation_penalty=1.0,
        issue_replay_global_saturation_threshold=4,
        issue_replay_global_saturation_penalty=1.5,
        issue_inspired_source_saturation_threshold=3,
        issue_inspired_source_saturation_penalty=1.25,
        active_backends=["pandas", "duckdb"],
    )

    assert restored.targets == ["filter"]
    assert restored.feature_counts == guidance.feature_counts
    assert restored.finding_feature_counts == guidance.finding_feature_counts
    assert restored.root_cause_counts == guidance.root_cause_counts
    assert restored.candidate_bug_family_counts == guidance.candidate_bug_family_counts
    assert restored.candidate_bug_signature_counts == guidance.candidate_bug_signature_counts
    assert list(restored.recent_discovery_windows) == list(guidance.recent_discovery_windows)
    assert restored.recent_discovery_stale_counts == guidance.recent_discovery_stale_counts
    assert restored.recent_discovery_signal_counts == guidance.recent_discovery_signal_counts
    assert restored.online_weights.total_updates == guidance.online_weights.total_updates
    assert restored.online_weights.feature_stats.keys() == guidance.online_weights.feature_stats.keys()


def test_guidance_state_round_trip_rebuilds_family_saturation_hits_for_active_backends():
    saturated_case = _case(
        1751,
        [
            {
                "op": "mutate",
                "column": "ratio",
                "expr": {"kind": "reverse_division_columns", "source": "x", "numerator": 1.0},
            }
        ],
    )
    guidance = GuidanceState(
        targets=["polars_reverse_division_columns"],
        active_backends=["polars"],
        family_saturation_threshold=3,
        known_saturated_bug_families=["reverse_division_operand_order@polars"],
    )
    for idx in range(3):
        guidance.record_result(
            saturated_case,
            {
                "findings": [
                    {
                        "root_cause": "reverse_division_operand_order",
                        "triage_verdict": "candidate_implementation_bug",
                        "suspicious_backends": ["polars"],
                        "signature": f"sig-roundtrip-{idx}",
                    }
                ],
                "preflight": {"valid": True, "fallback_used": False},
            },
        )

    state_dict = guidance.to_state_dict()
    restored = GuidanceState.from_state_dict(
        state_dict,
        targets=["polars_reverse_division_columns"],
        active_backends=["polars"],
        family_saturation_threshold=3,
        known_saturated_bug_families=["reverse_division_operand_order@polars"],
    )

    assert restored.predicted_saturated_family_roots(saturated_case) == ["reverse_division_operand_order"]
    assert restored.choose_case([saturated_case]).score_breakdown["family_saturation_active"] == 1.0


def test_guidance_state_round_trip_rebuilds_recent_discovery_counters_from_windows():
    case = _case(1760, [{"op": "filter", "column": "x", "cmp": "==", "value": 1}])
    guidance = GuidanceState(targets=["filter"])
    stale_row = {
        "findings": [],
        "is_new_behavior": False,
        "preflight": {"valid": True, "fallback_used": False},
    }
    signal_row = {
        "findings": [
            {
                "kind": "semantic_output_mismatch",
                "root_cause": "filter_semantics",
                "triage_verdict": "candidate_implementation_bug",
                "suspicious_backends": ["duckdb"],
                "signature": "sig-guidance-sync",
            }
        ],
        "is_new_behavior": True,
        "preflight": {"valid": True, "fallback_used": False},
    }

    guidance.record_result(case, stale_row)
    guidance.record_result(case, signal_row)

    state_dict = guidance.to_state_dict()
    state_dict.pop("recent_discovery_stale_counts", None)
    state_dict.pop("recent_discovery_signal_counts", None)
    restored = GuidanceState.from_state_dict(state_dict, targets=["filter"])

    assert restored.recent_discovery_stale_counts == guidance.recent_discovery_stale_counts
    assert restored.recent_discovery_signal_counts == guidance.recent_discovery_signal_counts


def test_guidance_state_from_state_dict_accepts_discovery_bias_overrides():
    guidance = GuidanceState(
        targets=["filter"],
        discovery_biases=[DiscoveryBias(targets=["filter"], feature_prefixes=["pattern:"], score_bonus=1.0)],
    )
    state_dict = guidance.to_state_dict()
    override = [DiscoveryBias(targets=["join"], feature_prefixes=["semantic_signal:"], score_bonus=2.0)]

    restored = GuidanceState.from_state_dict(
        state_dict,
        targets=["filter"],
        discovery_biases=override,
    )

    assert [bias.targets for bias in restored.discovery_biases] == [["join"]]
    assert [bias.feature_prefixes for bias in restored.discovery_biases] == [["semantic_signal:"]]
    assert restored.discovery_biases[0].score_bonus == 2.0


def test_guidance_downweights_stale_discovery_buckets_without_signal():
    stale = _case(76, [{"op": "select", "columns": ["id"]}])
    fresh = _case(77, [{"op": "filter", "column": "x", "cmp": "==", "value": 1}])
    guidance = GuidanceState()
    row = {
        "findings": [],
        "is_new_behavior": False,
        "preflight": {"valid": True, "fallback_used": False},
    }

    for _ in range(12):
        guidance.record_result(stale, row)

    stale_decision = guidance._score_case(stale, 2)
    selected = guidance.choose_case([stale, fresh])

    assert "opseq:select" in stale_decision.discovery_buckets
    assert stale_decision.score_breakdown["discovery_stale_penalty"] < 0.0
    assert selected.case is fresh


def test_guidance_downweights_recent_discovery_loops_before_long_term_staleness():
    stale = _case(176, [{"op": "select", "columns": ["id"]}])
    fresh = _case(177, [{"op": "filter", "column": "x", "cmp": "==", "value": 1}])
    guidance = GuidanceState()
    row = {
        "findings": [],
        "is_new_behavior": False,
        "preflight": {"valid": True, "fallback_used": False},
    }

    for _ in range(4):
        guidance.record_result(stale, row)

    stale_decision = guidance._score_case(stale, 2)
    selected = guidance.choose_case([stale, fresh])

    assert stale_decision.score_breakdown["discovery_stale_penalty"] == 0.0
    assert stale_decision.score_breakdown["recent_discovery_loop_penalty"] < 0.0
    assert stale_decision.score_breakdown["recent_discovery_loop_active"] == 1.0
    assert selected.case is fresh


def test_guidance_recent_discovery_loop_treats_candidate_findings_as_signal():
    case = _case(178, [{"op": "filter", "column": "x", "cmp": "==", "value": 1}])
    guidance = GuidanceState(targets=["filter"])

    for _ in range(4):
        guidance.record_result(
            case,
            {
                "findings": [
                    {
                        "kind": "semantic_output_mismatch",
                        "root_cause": "filter_semantics",
                        "triage_verdict": "candidate_implementation_bug",
                        "suspicious_backends": ["duckdb"],
                    }
                ],
                "is_new_behavior": False,
                "preflight": {"valid": True, "fallback_used": False},
            },
        )

    decision = guidance._score_case(case, 1)

    assert decision.score_breakdown["recent_discovery_window_count"] == 4.0
    assert decision.score_breakdown["recent_discovery_loop_penalty"] == 0.0
    assert decision.score_breakdown["recent_discovery_loop_active"] == 0.0


def test_guidance_balances_discovery_buckets_within_candidate_pool():
    repeated_filter = _case(78, [{"op": "filter", "column": "x", "cmp": "==", "value": 1}])
    common_filter = _case(79, [{"op": "filter", "column": "x", "cmp": "==", "value": 1}])
    rare_groupby = _case(
        80,
        [{"op": "groupby", "keys": ["g"], "aggs": [{"column": "x", "func": "sum", "as": "sum_x"}]}],
    )
    guidance = GuidanceState()

    for _ in range(8):
        guidance.record_result(repeated_filter, {"findings": [], "is_new_behavior": False})

    common_decision = guidance.choose_case([repeated_filter, common_filter])
    balanced_decision = guidance.choose_case([repeated_filter, common_filter, rare_groupby])

    assert common_decision.score_breakdown["candidate_pool_diversity_bonus"] > 0.0
    assert balanced_decision.case is rare_groupby
    assert (
        balanced_decision.score_breakdown["candidate_pool_diversity_bonus"]
        > common_decision.score_breakdown["candidate_pool_diversity_bonus"]
    )


def test_discovery_metrics_match_legacy_scoring_helpers():
    case = _case(
        781,
        [{"op": "filter", "column": "x", "cmp": "==", "value": 1}],
    )
    guidance = GuidanceState()
    for _ in range(8):
        guidance.record_result(
            case,
            {"findings": [], "is_new_behavior": False, "preflight": {"valid": True, "fallback_used": False}},
        )
    analysis = guidance._case_analysis(case)

    diversity_bonus, stale_penalty, recent_loop_penalty = _discovery_metrics(
        analysis.discovery_bucket_weights,
        guidance.discovery_bucket_counts,
        guidance.discovery_bucket_signal_counts,
        guidance.recent_discovery_stale_counts,
        guidance.recent_discovery_signal_counts,
        recent_window_count=len(guidance.recent_discovery_windows),
    )

    assert diversity_bonus == _discovery_diversity_bonus(analysis.discovery_buckets, guidance.discovery_bucket_counts)
    assert stale_penalty == _discovery_stale_penalty(
        analysis.discovery_buckets,
        guidance.discovery_bucket_counts,
        guidance.discovery_bucket_signal_counts,
    )
    assert recent_loop_penalty == _recent_discovery_loop_penalty(
        analysis.discovery_buckets,
        guidance.recent_discovery_stale_counts,
        guidance.recent_discovery_signal_counts,
        recent_window_count=len(guidance.recent_discovery_windows),
    )


def test_candidate_pool_bucket_factor_helper_matches_direct_bonus_formula():
    guidance = GuidanceState()
    repeated_filter = _case(782, [{"op": "filter", "column": "x", "cmp": "==", "value": 1}])
    rare_groupby = _case(
        783,
        [{"op": "groupby", "keys": ["g"], "aggs": [{"column": "x", "func": "sum", "as": "sum_x"}]}],
    )
    for _ in range(6):
        guidance.record_result(
            repeated_filter,
            {"findings": [], "is_new_behavior": False, "preflight": {"valid": True, "fallback_used": False}},
        )
    repeated_analysis = guidance._case_analysis(repeated_filter)
    rare_analysis = guidance._case_analysis(rare_groupby)
    pool_counts = Counter()
    pool_counts.update(repeated_analysis.discovery_buckets)
    pool_counts.update(rare_analysis.discovery_buckets)
    factors = _candidate_pool_bucket_factors(
        pool_counts,
        guidance.discovery_bucket_counts,
        guidance.recent_discovery_stale_counts,
    )

    repeated_bonus = _candidate_pool_discovery_bonus_from_weights(
        repeated_analysis.discovery_bucket_weights,
        factors,
    )
    rare_bonus = _candidate_pool_discovery_bonus_from_weights(
        rare_analysis.discovery_bucket_weights,
        factors,
    )

    assert repeated_bonus > 0.0
    assert rare_bonus > repeated_bonus


def test_guidance_discovery_bias_keeps_high_risk_case_in_pool():
    stale = _case(301, [{"op": "select", "columns": ["id"]}])
    risky = _case(
        302,
        [
            {"op": "join", "table": "t1", "left_on": "id", "right_on": "id", "how": "left"},
            {"op": "filter", "column": "g", "cmp": "not_in_set", "value": ["alpha"]},
        ],
    )
    risky.tables.append(
        TableData("t1", [ColumnSpec("id", "int", nullable=False), ColumnSpec("j", "int")], [{"id": 1, "j": 2}])
    )
    guidance = GuidanceState(
        discovery_biases=[
            DiscoveryBias(
                targets=["negative_set_membership_filter"],
                feature_prefixes=["filter:"],
                score_bonus=0.5,
                novelty_bonus=0.2,
                contribution_bonus=0.2,
                keep_in_pool=True,
            )
        ]
    )
    row = {
        "findings": [],
        "is_new_behavior": False,
        "preflight": {"valid": True, "fallback_used": False},
    }

    for _ in range(8):
        guidance.record_result(stale, row)

    decision = guidance.choose_case([stale, risky])

    assert decision.case is risky
    assert decision.discovery_bias_hits
    assert decision.score_breakdown["discovery_bias_keep_in_pool"] == 1.0


def test_guidance_discovery_bias_candidate_pool_bonus_contributes_to_pool_scoring():
    stale = _case(304, [{"op": "select", "columns": ["id"]}])
    risky = _case(
        305,
        [{"op": "filter", "column": "g", "cmp": "not_in_set", "value": ["alpha"]}],
    )
    guidance = GuidanceState(
        discovery_biases=[
            DiscoveryBias(
                targets=["negative_set_membership_filter"],
                feature_prefixes=["filter:"],
                candidate_pool_bonus=0.3,
            )
        ]
    )

    stale_decision = guidance._score_case(stale, 2)
    risky_decision = guidance._score_case(risky, 2)
    _apply_candidate_pool_discovery_balance(
        [stale_decision, risky_decision],
        guidance.discovery_bucket_counts,
        guidance.recent_discovery_stale_counts,
    )

    assert risky_decision.score_breakdown["candidate_pool_bias_bonus"] == 0.3
    assert risky_decision.score_breakdown["candidate_pool_shared_bonus"] > 0.0
    assert risky_decision.score_breakdown["candidate_pool_diversity_bonus"] == pytest.approx(
        risky_decision.score_breakdown["candidate_pool_shared_bonus"] + 0.3
    )
    assert stale_decision.score_breakdown["candidate_pool_bias_bonus"] == 0.0
    assert stale_decision.score_breakdown["candidate_pool_diversity_bonus"] == pytest.approx(
        stale_decision.score_breakdown["candidate_pool_shared_bonus"]
    )


def test_guidance_discovery_bias_matches_raw_semantic_family_against_canonical_target():
    case = _case(
        303,
        [
            {"op": "mutate", "column": "s_clean", "expr": {"kind": "string_strip", "source": "g"}},
            {"op": "mutate", "column": "s_key", "expr": {"kind": "string_lower", "source": "s_clean"}},
            {"op": "semi_join", "table": "t1", "left_on": "s_key", "right_on": "s_key"},
        ],
    )
    case = Case(
        case.case_id,
        case.seed,
        [
            case.tables[0],
            TableData("t1", [ColumnSpec("s_key", "str", nullable=False)], [{"s_key": "alpha"}]),
        ],
        case.program,
        metadata=case.metadata,
    )
    guidance = GuidanceState(
        targets=["semantic_family:join_membership"],
        discovery_biases=[
            DiscoveryBias(
                targets=["join_membership"],
                feature_prefixes=["join:"],
                score_bonus=0.25,
                keep_in_pool=True,
            )
        ],
    )

    decision = guidance.choose_case([case])

    assert decision.matched_targets == ["semantic_family:join_membership"]
    assert decision.discovery_bias_hits == ["join_membership|join:"]
    assert decision.score_breakdown["discovery_bias_keep_in_pool"] == 1.0


def test_guidance_discovery_bias_matches_raw_semantic_signal_against_canonical_target():
    case = _common_api_workflow_case_matching_template("sql_left_join_case_membership_groupby")
    guidance = GuidanceState(
        targets=["semantic_signal:left_join_case_when_membership"],
        discovery_biases=[
            DiscoveryBias(
                targets=["left_join_case_when_membership"],
                feature_prefixes=["semantic_signal:"],
                score_bonus=0.25,
                keep_in_pool=True,
            )
        ],
    )

    decision = guidance.choose_case([case])

    assert decision.matched_targets == ["semantic_signal:left_join_case_when_membership"]
    assert decision.discovery_bias_hits == ["left_join_case_when_membership|semantic_signal:"]
    assert decision.score_breakdown["discovery_bias_keep_in_pool"] == 1.0


def test_guidance_discovery_staleness_treats_findings_as_signal():
    case = _case(78, [{"op": "filter", "column": "x", "cmp": "==", "value": 1}])
    guidance = GuidanceState(targets=["filter"])

    for idx in range(12):
        guidance.record_result(
            case,
            {
                "findings": [
                    {
                        "kind": "semantic_output_mismatch",
                        "root_cause": "filter_semantics",
                        "triage_verdict": "candidate_implementation_bug",
                        "suspicious_backends": ["duckdb"],
                        "signature": f"sig-candidate-{idx}",
                    }
                ],
                "is_new_behavior": False,
                "preflight": {"valid": True, "fallback_used": False},
            },
        )

    decision = guidance.choose_case([case])

    assert decision.score_breakdown["discovery_stale_active"] == 0.0
    assert decision.score_breakdown["discovery_stale_penalty"] == 0.0


def test_guidance_discovery_staleness_ignores_expected_semantic_divergence():
    case = _case(81, [{"op": "filter", "column": "x", "cmp": "==", "value": 1}])
    guidance = GuidanceState(targets=["filter"])

    for idx in range(12):
        guidance.record_result(
            case,
            {
                "findings": [
                    {
                        "kind": "semantic_output_mismatch",
                        "root_cause": "filter_predicate",
                        "triage_verdict": "expected_semantic_divergence",
                        "signature": f"sig-expected-{idx}",
                    }
                ],
                "is_new_behavior": False,
                "preflight": {"valid": True, "fallback_used": False},
            },
        )

    decision = guidance.choose_case([case])

    assert decision.score_breakdown["discovery_stale_active"] == 1.0
    assert decision.score_breakdown["discovery_stale_penalty"] < 0.0


def test_guidance_discovery_staleness_ignores_new_expected_semantic_divergence():
    case = _case(82, [{"op": "filter", "column": "x", "cmp": "==", "value": 1}])
    guidance = GuidanceState(targets=["filter"])

    for idx in range(12):
        guidance.record_result(
            case,
            {
                "findings": [
                    {
                        "kind": "semantic_output_mismatch",
                        "root_cause": "filter_predicate",
                        "triage_verdict": "expected_semantic_divergence",
                        "signature": f"sig-expected-new-{idx}",
                    }
                ],
                "is_new_behavior": True,
                "preflight": {"valid": True, "fallback_used": False},
            },
        )

    decision = guidance.choose_case([case])

    assert decision.score_breakdown["discovery_stale_active"] == 1.0
    assert decision.score_breakdown["discovery_stale_penalty"] < 0.0


def test_guidance_common_api_targets_map_to_features():
    row_number_case = _case(
        79,
        [
            {
                "op": "row_number_filter",
                "partition_by": ["g"],
                "order_by": [{"column": "x", "ascending": False}],
                "cmp": "<=",
                "n": 1,
            }
        ],
    )
    nunique_case = _case(
        80,
        [{"op": "groupby", "keys": ["g"], "aggs": [{"column": "x", "func": "nunique", "as": "unique_x"}]}],
    )
    workflow_case = _case(
        81,
        [
            {"op": "aggregate", "aggs": [{"column": "x", "func": "sum", "as": "sum_x"}]},
            {"op": "filter", "column": "sum_x", "cmp": ">=", "value": 0},
        ],
    )

    row_number_decision = GuidanceState(targets=["row_number_filter", "top_n_per_group"]).choose_case(
        [row_number_case]
    )
    nunique_decision = GuidanceState(targets=["nunique", "count_distinct", "distinct_count_aggregation"]).choose_case(
        [nunique_case]
    )
    workflow_decision = GuidanceState(targets=["filtered_global_aggregation"]).choose_case([workflow_case])

    assert row_number_decision.matched_targets == ["row_number_filter", "top_n_per_group"]
    assert nunique_decision.matched_targets == ["nunique", "count_distinct", "distinct_count_aggregation"]
    assert workflow_decision.matched_targets == ["filtered_global_aggregation"]


def test_extract_case_features_tracks_negative_set_membership_filter():
    case = _case(14, [{"op": "filter", "column": "g", "cmp": "not_in_set", "value": ["alpha"]}])
    guidance = GuidanceState(targets=["negative_set_membership_filter"])

    features = extract_case_features(case)
    decision = guidance.choose_case([case])

    assert "filter:set-membership" in features
    assert "filter:negative-set-membership" in features
    assert "semantic_signal:negative_set_membership" in features
    assert "combo_risk:negative_set_membership" in features
    assert decision.matched_targets == ["negative_set_membership_filter"]


def test_extract_case_features_tracks_string_strip_expression():
    case = _case(16, [{"op": "mutate", "column": "g_clean", "expr": {"kind": "string_strip", "source": "g"}}])

    features = extract_case_features(case)

    assert "op:mutate" in features
    assert "expr:string_strip" in features
    assert "string:strip" in features


def test_extract_case_features_tracks_string_upper_expression():
    case = _case(15, [{"op": "mutate", "column": "g_upper", "expr": {"kind": "string_upper", "source": "g"}}])
    guidance = GuidanceState(targets=["string_upper"])

    features = extract_case_features(case)
    decision = guidance.choose_case([case])

    assert "op:mutate" in features
    assert "expr:string_upper" in features
    assert "string:upper" in features
    assert decision.matched_targets == ["string_upper"]


def test_guidance_tracks_unicode_case_mapping_boundary():
    case = _case(17, [{"op": "mutate", "column": "g_upper", "expr": {"kind": "string_upper", "source": "g"}}])
    guidance = GuidanceState(targets=["unicode_case_mapping"])

    features = extract_case_features(case)
    decision = guidance.choose_case([case])

    assert "has:unicode_string" in features
    assert "expr:string_upper" in features
    assert decision.matched_targets == ["unicode_case_mapping"]
    assert decision.score_breakdown["resolved_semantic_boundary_penalty"] == 0.0


def test_guidance_demotes_known_unicode_case_boundary_in_general_pool():
    boundary_case = _case(
        18,
        [
            {"op": "mutate", "column": "g_upper", "expr": {"kind": "string_upper", "source": "g"}},
            {"op": "groupby", "keys": ["g_upper"], "aggs": [{"column": "x", "func": "count", "as": "count_x"}]},
        ],
    )
    fresh_case = _case(
        19,
        [
            {"op": "mutate", "column": "g_len", "expr": {"kind": "string_length", "source": "g"}},
            {"op": "groupby", "keys": ["g_len"], "aggs": [{"column": "x", "func": "count", "as": "count_x"}]},
        ],
    )
    guidance = GuidanceState(targets=["groupby_sorted_input", "expressions"])

    boundary_decision = guidance._score_case(boundary_case, 2)
    fresh_decision = guidance._score_case(fresh_case, 2)
    selected = guidance.choose_case([boundary_case, fresh_case])

    assert "expr:string_upper" in boundary_decision.features
    assert "has:unicode_string" in boundary_decision.features
    assert boundary_decision.score_breakdown["resolved_semantic_boundary_penalty"] < 0.0
    assert fresh_decision.score_breakdown["resolved_semantic_boundary_penalty"] == 0.0
    assert selected.case is fresh_case


def test_guidance_preserves_explicit_unicode_case_boundary_target():
    boundary_case = _case(
        20,
        [
            {"op": "mutate", "column": "g_upper", "expr": {"kind": "string_upper", "source": "g"}},
            {"op": "groupby", "keys": ["g_upper"], "aggs": [{"column": "x", "func": "count", "as": "count_x"}]},
        ],
    )
    ordinary_case = _case(
        21,
        [
            {"op": "mutate", "column": "g_len", "expr": {"kind": "string_length", "source": "g"}},
            {"op": "groupby", "keys": ["g_len"], "aggs": [{"column": "x", "func": "count", "as": "count_x"}]},
        ],
    )
    guidance = GuidanceState(targets=["unicode_case_mapping"])

    boundary_decision = guidance._score_case(boundary_case, 2)
    selected = guidance.choose_case([boundary_case, ordinary_case])

    assert boundary_decision.score_breakdown["resolved_semantic_boundary_penalty"] == 0.0
    assert selected.case is boundary_case


def test_guidance_demotes_float_aggregate_precision_boundary_in_general_pool():
    boundary_case = Case(
        case_id="case-float-aggregate-boundary",
        seed=22,
        tables=[
            TableData(
                "t0",
                [
                    ColumnSpec("id", "int", nullable=False),
                    ColumnSpec("y", "float", nullable=True),
                ],
                [
                    {"id": 1, "y": 0.1},
                    {"id": 1, "y": 0.2},
                    {"id": 2, "y": 1.0},
                ],
            )
        ],
        program=Program(
            "prog-float-aggregate-boundary",
            22,
            [
                {"op": "groupby", "keys": ["id"], "aggs": [{"column": "y", "func": "sum", "as": "sum_y"}]},
                {"op": "sort", "columns": ["sum_y", "id"], "ascending": False},
            ],
        ),
    )
    fresh_case = _case(
        23,
        [
            {"op": "groupby", "keys": ["g"], "aggs": [{"column": "x", "func": "count", "as": "count_x"}]},
            {"op": "sort", "columns": ["count_x", "g"], "ascending": False},
        ],
    )
    guidance = GuidanceState(targets=["groupby", "aggregation", "sort_limit"])

    boundary_decision = guidance._score_case(boundary_case, 2)
    fresh_decision = guidance._score_case(fresh_case, 2)
    selected = guidance.choose_case([boundary_case, fresh_case])

    assert "agg:precision-float" in boundary_decision.features
    assert boundary_decision.score_breakdown["resolved_semantic_boundary_penalty"] < 0.0
    assert fresh_decision.score_breakdown["resolved_semantic_boundary_penalty"] == 0.0
    assert selected.case is fresh_case


def test_guidance_tracks_precision_float_aggregate_from_joined_right_table():
    case = Case(
        case_id="case-joined-float-aggregate-boundary",
        seed=25,
        tables=[
            TableData(
                "t0",
                [ColumnSpec("id", "int", nullable=False)],
                [{"id": 1}, {"id": 2}],
            ),
            TableData(
                "t1",
                [
                    ColumnSpec("id", "int", nullable=False),
                    ColumnSpec("z", "float", nullable=True),
                ],
                [{"id": 1, "z": 0.1}, {"id": 1, "z": 0.2}, {"id": 2, "z": 1.0}],
            ),
        ],
        program=Program(
            "prog-joined-float-aggregate-boundary",
            25,
            [
                {"op": "join", "table": "t1", "left_on": "id", "right_on": "id", "how": "left"},
                {"op": "groupby", "keys": ["id"], "aggs": [{"column": "z", "func": "sum", "as": "sum_z"}]},
                {"op": "sort", "columns": ["sum_z", "id"], "ascending": False},
            ],
        ),
    )
    guidance = GuidanceState(targets=["join", "groupby", "aggregation"])

    features = extract_case_features(case)
    decision = guidance._score_case(case, 1)

    assert "agg_source_type:float" in features
    assert "agg:precision-float" in features
    assert decision.score_breakdown["resolved_semantic_boundary_penalty"] < 0.0


def test_guidance_preserves_explicit_float_precision_probe_target():
    boundary_case = generate_case(135, profile="duckdb_float_literal_precision")
    ordinary_case = _case(24, [{"op": "filter", "column": "x", "cmp": ">=", "value": 0}])
    guidance = GuidanceState(targets=["duckdb_float_literal_precision"])

    boundary_decision = guidance._score_case(boundary_case, 2)
    selected = guidance.choose_case([boundary_case, ordinary_case])

    assert "pattern:duckdb_float_literal_precision" in boundary_decision.features
    assert boundary_decision.score_breakdown["resolved_semantic_boundary_penalty"] == 0.0
    assert selected.case is boundary_case


def test_extract_case_features_tracks_string_length_expression():
    case = _case(26, [{"op": "mutate", "column": "g_len", "expr": {"kind": "string_length", "source": "g"}}])
    guidance = GuidanceState(targets=["string_length"])

    features = extract_case_features(case)
    decision = guidance.choose_case([case])

    assert "op:mutate" in features
    assert "expr:string_length" in features
    assert "string:length" in features
    assert decision.matched_targets == ["string_length"]


def test_extract_case_features_tracks_string_lower_expression():
    case = _case(27, [{"op": "mutate", "column": "g_lower", "expr": {"kind": "string_lower", "source": "g"}}])
    guidance = GuidanceState(targets=["string_lower"])

    features = extract_case_features(case)
    decision = guidance.choose_case([case])

    assert "op:mutate" in features
    assert "expr:string_lower" in features
    assert "string:lower" in features
    assert decision.matched_targets == ["string_lower"]


def test_extract_case_features_tracks_string_null_if_empty_expression():
    case = _case(25, [{"op": "mutate", "column": "g_norm", "expr": {"kind": "string_null_if_empty", "source": "g"}}])
    guidance = GuidanceState(targets=["string_null_if_empty"])

    features = extract_case_features(case)
    decision = guidance.choose_case([case])

    assert "op:mutate" in features
    assert "expr:string_null_if_empty" in features
    assert "string:null-if-empty" in features
    assert "null:empty-string" in features
    assert decision.matched_targets == ["string_null_if_empty"]


def test_extract_case_features_tracks_string_replace_expression():
    case = _case(
        17,
        [{"op": "mutate", "column": "g_token", "expr": {"kind": "string_replace", "source": "g", "old": " ", "new": "_"}}],
    )

    features = extract_case_features(case)

    assert "op:mutate" in features
    assert "expr:string_replace" in features
    assert "string:replace" in features


def test_extract_case_features_tracks_string_slice_expression():
    case = _case(
        18,
        [{"op": "mutate", "column": "g_prefix", "expr": {"kind": "string_slice", "source": "g", "start": 0, "length": 3}}],
    )

    features = extract_case_features(case)

    assert "op:mutate" in features
    assert "expr:string_slice" in features
    assert "string:slice" in features


def test_extract_case_features_tracks_string_split_part_expression():
    case = _case(
        24,
        [{"op": "mutate", "column": "g_token", "expr": {"kind": "string_split_part", "source": "g", "sep": " ", "index": 0}}],
    )
    guidance = GuidanceState(targets=["string_split_part"])

    features = extract_case_features(case)
    decision = guidance.choose_case([case])

    assert "op:mutate" in features
    assert "expr:string_split_part" in features
    assert "string:split-first" in features
    assert decision.matched_targets == ["string_split_part"]


def test_extract_case_features_tracks_string_basename_expression():
    case = Case(
        "case-string-basename",
        28,
        [
            TableData(
                "t0",
                [ColumnSpec("path_value", "str", nullable=True)],
                [
                    {"path_value": "/tmp/alpha.csv"},
                    {"path_value": "D:\\archive\\beta.parquet"},
                    {"path_value": None},
                ],
            )
        ],
        Program(
            "prog-string-basename",
            28,
            [{"op": "mutate", "column": "path_base", "expr": {"kind": "string_basename", "source": "path_value"}}],
        ),
    )
    guidance = GuidanceState(targets=["string_basename"])

    features = extract_case_features(case)
    decision = guidance.choose_case([case])

    assert "op:mutate" in features
    assert "expr:string_basename" in features
    assert "path:basename" in features
    assert decision.matched_targets == ["string_basename"]


def test_guidance_recognizes_normalized_string_join_key():
    case = Case(
        "case-normalized-string-join",
        29,
        [
            TableData(
                "t0",
                [
                    ColumnSpec("id", "int", nullable=False),
                    ColumnSpec("s", "str", nullable=True),
                    ColumnSpec("x", "int", nullable=True),
                ],
                [
                    {"id": 0, "s": " Alpha ", "x": 1},
                    {"id": 1, "s": None, "x": 2},
                ],
            ),
            TableData(
                "t_string_lookup",
                [
                    ColumnSpec("s_key", "str", nullable=False),
                    ColumnSpec("label", "str", nullable=False),
                ],
                [{"s_key": "alpha", "label": "letter-a"}],
            ),
        ],
        Program(
            "prog-normalized-string-join",
            29,
            [
                {"op": "mutate", "column": "s_clean", "expr": {"kind": "string_strip", "source": "s"}},
                {"op": "mutate", "column": "s_key", "expr": {"kind": "string_lower", "source": "s_clean"}},
                {"op": "join", "table": "t_string_lookup", "left_on": "s_key", "right_on": "s_key", "how": "left"},
                {"op": "fill_null", "column": "label", "value": "unmatched"},
                {"op": "groupby", "keys": ["label"], "aggs": [{"column": "id", "func": "count", "as": "count_id"}]},
            ],
        ),
    )
    guidance = GuidanceState(targets=["normalized_string_join"])

    features = extract_case_features(case)
    decision = guidance.choose_case([case])

    assert "combo_risk:cleaned_join_key" in features
    assert "combo_risk:normalized_string_join_key" in features
    assert "combo_risk:chained_string_normalized_join_key" in features
    assert "semantic_family:string_semantics" in features
    assert "semantic_family:join_membership" in features
    assert decision.matched_targets == ["normalized_string_join"]


def test_derive_semantic_families_groups_high_risk_semantics_into_stable_labels():
    families = derive_semantic_families(
        [
            "combo_risk:left_join_case_when_membership",
            "combo_risk:join_case_when_membership_aggregation",
            "pattern:conditional_expression",
            "filter:null-predicate",
        ]
    )

    assert "join_membership" in families
    assert "conditional_semantics" in families
    assert "null_semantics" in families


def test_guidance_can_target_semantic_family_directly():
    case = _case(
        140,
        [
            {"op": "mutate", "column": "s_clean", "expr": {"kind": "string_strip", "source": "g"}},
            {"op": "mutate", "column": "s_key", "expr": {"kind": "string_lower", "source": "s_clean"}},
            {"op": "semi_join", "table": "t1", "left_on": "s_key", "right_on": "s_key"},
            {"op": "sort", "keys": [{"column": "s_key", "ascending": True, "nulls": "last"}]},
            {"op": "limit", "n": 1},
        ],
    )
    case = Case(
        case.case_id,
        case.seed,
        [
            case.tables[0],
            TableData("t1", [ColumnSpec("s_key", "str", nullable=False)], [{"s_key": "alpha"}]),
        ],
        case.program,
        metadata=case.metadata,
    )
    decision = GuidanceState(targets=["join_membership", "string_semantics"]).choose_case([case])

    assert decision.matched_targets == ["join_membership", "string_semantics"]


def test_guidance_recognizes_normalized_string_membership_key():
    case = Case(
        "case-normalized-string-membership",
        30,
        [
            TableData(
                "t0",
                [
                    ColumnSpec("id", "int", nullable=False),
                    ColumnSpec("s", "str", nullable=True),
                    ColumnSpec("x", "int", nullable=True),
                ],
                [
                    {"id": 0, "s": " Alpha ", "x": 1},
                    {"id": 1, "s": "Beta", "x": 2},
                ],
            ),
            TableData(
                "t_string_membership",
                [ColumnSpec("s_key", "str", nullable=False)],
                [{"s_key": "alpha"}],
            ),
        ],
        Program(
            "prog-normalized-string-membership",
            30,
            [
                {"op": "mutate", "column": "s_clean", "expr": {"kind": "string_strip", "source": "s"}},
                {"op": "mutate", "column": "s_key", "expr": {"kind": "string_lower", "source": "s_clean"}},
                {"op": "filter", "column": "s_key", "cmp": "is_not_null", "value": None},
                {"op": "semi_join", "table": "t_string_membership", "left_on": "s_key", "right_on": "s_key"},
                {"op": "sort", "keys": [{"column": "s_key", "ascending": True, "nulls": "last"}]},
                {"op": "limit", "n": 1},
            ],
        ),
    )
    guidance = GuidanceState(targets=["normalized_string_membership"])

    features = extract_case_features(case)
    decision = guidance.choose_case([case])

    assert "combo_risk:normalized_string_membership_key" in features
    assert "combo_risk:chained_string_normalized_membership_key" in features
    assert "combo_risk:normalized_string_membership_topk" in features
    assert decision.matched_targets == ["normalized_string_membership"]


def test_guidance_recognizes_sql_distinct_null_topk_template():
    case = _common_api_workflow_case_matching_template("sql_distinct_null_coalesce_topk")
    guidance = GuidanceState(targets=["sql_distinct_null_topk", "coalesced_distinct_topk"])

    features = extract_case_features(case)
    decision = guidance.choose_case([case])

    assert case.metadata["workflow_template"] == "sql_distinct_null_coalesce_topk"
    assert "common_api_template:sql_distinct_null_coalesce_topk" in features
    assert "combo_risk:sql_distinct_null_topk" in features
    assert "combo_risk:coalesced_distinct_topk" in features
    assert decision.matched_targets == ["sql_distinct_null_topk", "coalesced_distinct_topk"]


def test_guidance_recognizes_left_join_coalesce_membership_template():
    case = _common_api_workflow_case_matching_template("sql_left_join_coalesce_membership")
    guidance = GuidanceState(targets=["left_join_coalesce_membership", "join_coalesce_membership"])

    features = extract_case_features(case)
    decision = guidance.choose_case([case])

    assert case.metadata["workflow_template"] == "sql_left_join_coalesce_membership"
    assert "combo_risk:left_join_coalesce_membership" in features
    assert "combo_risk:join_coalesce_membership" in features
    assert decision.matched_targets == ["left_join_coalesce_membership", "join_coalesce_membership"]


def test_guidance_recognizes_case_when_membership_template():
    case = _common_api_workflow_case_matching_template("sql_case_membership_distinct_topk")
    guidance = GuidanceState(targets=["case_when_membership"])

    features = extract_case_features(case)
    decision = guidance.choose_case([case])

    assert case.metadata["workflow_template"] == "sql_case_membership_distinct_topk"
    assert "combo_risk:case_when_membership_predicate" in features
    assert "combo_risk:case_when_membership_topk" in features
    assert decision.matched_targets == ["case_when_membership"]


def test_guidance_recognizes_union_coalesce_distinct_topk_template():
    case = _common_api_workflow_case_matching_template("sql_union_coalesce_distinct_topk")
    guidance = GuidanceState(targets=["union_coalesce_distinct_topk"])

    features = extract_case_features(case)
    decision = guidance.choose_case([case])

    assert case.metadata["workflow_template"] == "sql_union_coalesce_distinct_topk"
    assert "combo_risk:union_coalesce_distinct_topk" in features
    assert decision.matched_targets == ["union_coalesce_distinct_topk"]


def test_guidance_recognizes_left_join_case_membership_template():
    case = _common_api_workflow_case_matching_template("sql_left_join_case_membership_groupby")
    guidance = GuidanceState(targets=["left_join_case_membership"])

    features = extract_case_features(case)
    decision = guidance.choose_case([case])

    assert case.metadata["workflow_template"] == "sql_left_join_case_membership_groupby"
    assert "combo_risk:left_join_case_when_membership" in features
    assert "combo_risk:join_case_when_membership_aggregation" in features
    assert decision.matched_targets == ["left_join_case_membership"]


def test_guidance_recognizes_left_join_null_predicate_aggregate_template():
    case = _common_api_workflow_case_matching_template("sql_left_join_null_predicate_aggregate")
    guidance = GuidanceState(targets=["left_join_null_predicate_aggregation"])

    features = extract_case_features(case)
    decision = guidance.choose_case([case])

    assert case.metadata["workflow_template"] == "sql_left_join_null_predicate_aggregate"
    assert "combo_risk:left_join_null_predicate_filter" in features
    assert "combo_risk:left_join_null_predicate_aggregation" in features
    assert decision.matched_targets == ["left_join_null_predicate_aggregation"]


def test_guidance_recognizes_coalesce_case_distinct_groupby_template():
    case = _common_api_workflow_case_matching_template("sql_coalesce_case_distinct_groupby")
    guidance = GuidanceState(targets=["coalesce_case_distinct_aggregation"])

    features = extract_case_features(case)
    decision = guidance.choose_case([case])

    assert case.metadata["workflow_template"] == "sql_coalesce_case_distinct_groupby"
    assert "combo_risk:coalesce_case_distinct_aggregation" in features
    assert decision.matched_targets == ["coalesce_case_distinct_aggregation"]


def test_guidance_recognizes_numeric_text_cast_membership_template():
    case = _common_api_workflow_case_matching_template("sql_numeric_text_cast_membership_groupby")
    guidance = GuidanceState(targets=["numeric_text_cast_membership"])

    features = extract_case_features(case)
    decision = guidance.choose_case([case])

    assert case.metadata["workflow_template"] == "sql_numeric_text_cast_membership_groupby"
    assert "combo_risk:numeric_text_cast_membership_aggregation" in features
    assert decision.matched_targets == ["numeric_text_cast_membership"]


def test_guidance_recognizes_boolean_membership_case_aggregate_template():
    case = _common_api_workflow_case_matching_template("sql_bool_membership_case_aggregate")
    guidance = GuidanceState(targets=["boolean_membership_case_aggregation"])

    features = extract_case_features(case)
    decision = guidance.choose_case([case])

    assert case.metadata["workflow_template"] == "sql_bool_membership_case_aggregate"
    assert "combo_risk:boolean_case_when_predicate" in features
    assert "combo_risk:boolean_membership_case_aggregation" in features
    assert decision.matched_targets == ["boolean_membership_case_aggregation"]


def test_guidance_recognizes_left_join_boolean_case_groupby_template():
    case = _common_api_workflow_case_matching_template("sql_left_join_bool_case_groupby")
    guidance = GuidanceState(targets=["left_join_boolean_case_aggregation"])

    features = extract_case_features(case)
    decision = guidance.choose_case([case])

    assert case.metadata["workflow_template"] == "sql_left_join_bool_case_groupby"
    assert "combo_risk:boolean_case_when_aggregation" in features
    assert "combo_risk:left_join_boolean_case_aggregation" in features
    assert decision.matched_targets == ["left_join_boolean_case_aggregation"]


def test_guidance_recognizes_left_join_boolean_coalesce_case_groupby_template():
    case = _common_api_workflow_case_matching_template("sql_left_join_bool_coalesce_case_groupby")
    guidance = GuidanceState(targets=["left_join_boolean_coalesce_aggregation"])

    features = extract_case_features(case)
    decision = guidance.choose_case([case])

    assert case.metadata["workflow_template"] == "sql_left_join_bool_coalesce_case_groupby"
    assert "combo_risk:boolean_coalesce_case_aggregation" in features
    assert "combo_risk:left_join_boolean_coalesce_aggregation" in features
    assert decision.matched_targets == ["left_join_boolean_coalesce_aggregation"]


def test_guidance_recognizes_boolean_antijoin_case_aggregate_template():
    case = _common_api_workflow_case_matching_template("sql_bool_antijoin_case_aggregate")
    guidance = GuidanceState(targets=["boolean_membership_case_aggregation"])

    features = extract_case_features(case)
    decision = guidance.choose_case([case])

    assert case.metadata["workflow_template"] == "sql_bool_antijoin_case_aggregate"
    assert "combo_risk:boolean_case_when_predicate" in features
    assert "combo_risk:boolean_membership_case_aggregation" in features
    assert decision.matched_targets == ["boolean_membership_case_aggregation"]


def test_guidance_recognizes_left_join_boolean_coalesce_filter_groupby_template():
    case = _common_api_workflow_case_matching_template("sql_left_join_bool_coalesce_filter_groupby")
    guidance = GuidanceState(targets=["left_join_boolean_coalesce_filter_aggregation"])

    features = extract_case_features(case)
    decision = guidance.choose_case([case])

    assert case.metadata["workflow_template"] == "sql_left_join_bool_coalesce_filter_groupby"
    assert "combo_risk:boolean_coalesce_filter_aggregation" in features
    assert "combo_risk:left_join_boolean_coalesce_filter_aggregation" in features
    assert decision.matched_targets == ["left_join_boolean_coalesce_filter_aggregation"]


def test_guidance_recognizes_numeric_text_cast_bool_antijoin_template():
    case = _common_api_workflow_case_matching_template("sql_numeric_text_cast_bool_antijoin_groupby")
    guidance = GuidanceState(targets=["numeric_text_cast_membership"])

    features = extract_case_features(case)
    decision = guidance.choose_case([case])

    assert case.metadata["workflow_template"] == "sql_numeric_text_cast_bool_antijoin_groupby"
    assert "combo_risk:numeric_text_cast_membership_aggregation" in features
    assert "combo_risk:boolean_membership_case_aggregation" in features
    assert decision.matched_targets == ["numeric_text_cast_membership"]


def test_guidance_recognizes_multi_key_semijoin_case_groupby_template():
    case = _common_api_workflow_case_matching_template("sql_multi_key_semijoin_case_groupby")
    guidance = GuidanceState(targets=["multi_key_membership_aggregation"])

    features = extract_case_features(case)
    decision = guidance.choose_case([case])

    assert case.metadata["workflow_template"] == "sql_multi_key_semijoin_case_groupby"
    assert "combo_risk:multi_key_semi_anti_join" in features
    assert "combo_risk:multi_key_membership_aggregation" in features
    assert decision.matched_targets == ["multi_key_membership_aggregation"]


def test_guidance_recognizes_multi_key_antijoin_case_groupby_template():
    case = _common_api_workflow_case_matching_template("sql_multi_key_antijoin_case_groupby")
    guidance = GuidanceState(targets=["multi_key_membership_aggregation"])

    features = extract_case_features(case)
    decision = guidance.choose_case([case])

    assert case.metadata["workflow_template"] == "sql_multi_key_antijoin_case_groupby"
    assert "combo_risk:multi_key_semi_anti_join" in features
    assert "combo_risk:multi_key_membership_aggregation" in features
    assert decision.matched_targets == ["multi_key_membership_aggregation"]


def test_extract_case_features_tracks_date_part_expression():
    case = _case(
        19,
        [{"op": "mutate", "column": "year", "expr": {"kind": "date_part", "source": "g", "part": "year"}}],
    )
    guidance = GuidanceState(targets=["date_part"])

    features = extract_case_features(case)
    decision = guidance.choose_case([case])

    assert "op:mutate" in features
    assert "expr:date_part" in features
    assert "date:part" in features
    assert "date_part:year" in features
    assert decision.matched_targets == ["date_part"]


def test_extract_case_features_tracks_cast_boundary_expression():
    case = Case(
        "case-cast-boundary-guidance",
        20,
        [
            TableData(
                "t0",
                [
                    ColumnSpec("id", "int", nullable=False),
                    ColumnSpec("num_s", "str", nullable=True),
                ],
                [
                    {"id": 1, "num_s": "10"},
                    {"id": 2, "num_s": None},
                ],
            )
        ],
        Program(
            "prog-cast-boundary-guidance",
            20,
            [
                {
                    "op": "mutate",
                    "column": "num_i",
                    "expr": {"kind": "cast", "source": "num_s", "to": "int", "input_domain": "integer_string"},
                },
                {"op": "mutate", "column": "num_label", "expr": {"kind": "cast", "source": "num_i", "to": "str"}},
            ],
        ),
    )
    guidance = GuidanceState(targets=["casts"])

    features = extract_case_features(case)
    decision = guidance.choose_case([case])

    assert "op:mutate" in features
    assert "expr:cast" in features
    assert "cast_to:int" in features
    assert "cast_to:str" in features
    assert "cast:str_to_int" in features
    assert "cast:int_to_str" in features
    assert "cast_domain:integer_string" in features
    assert decision.matched_targets == ["casts"]


def test_extract_case_features_tracks_string_concat_expression():
    case = _case(
        19,
        [{"op": "mutate", "column": "label", "expr": {"kind": "string_concat", "source": "g", "other": "g", "sep": "-"}}],
    )

    features = extract_case_features(case)

    assert "op:mutate" in features
    assert "expr:string_concat" in features
    assert "string:concat" in features


def test_extract_case_features_tracks_string_contains_expression():
    case = _case(
        21,
        [{"op": "mutate", "column": "has_a", "expr": {"kind": "string_contains", "source": "g", "needle": "a"}}],
    )

    features = extract_case_features(case)

    assert "op:mutate" in features
    assert "expr:string_contains" in features
    assert "string:contains" in features


def test_extract_case_features_tracks_string_starts_and_ends_with_expression():
    starts = _case(
        22,
        [{"op": "mutate", "column": "starts_a", "expr": {"kind": "string_starts_with", "source": "g", "needle": "a"}}],
    )
    ends = _case(
        23,
        [{"op": "mutate", "column": "ends_a", "expr": {"kind": "string_ends_with", "source": "g", "needle": "a"}}],
    )

    starts_features = extract_case_features(starts)
    ends_features = extract_case_features(ends)

    assert "expr:string_starts_with" in starts_features
    assert "string:starts-with" in starts_features
    assert "expr:string_ends_with" in ends_features
    assert "string:ends-with" in ends_features


def test_extract_case_features_tracks_bool_not_expression():
    case = _case(20, [{"op": "mutate", "column": "not_flag", "expr": {"kind": "bool_not", "source": "flag"}}])

    features = extract_case_features(case)

    assert "op:mutate" in features
    assert "expr:bool_not" in features
    assert "boolean:not" in features


def test_extract_case_features_tracks_numeric_clip_expression():
    case = _case(
        14,
        [{"op": "mutate", "column": "x_clip", "expr": {"kind": "clip", "source": "x", "lower": -2, "upper": 2}}],
    )

    features = extract_case_features(case)

    assert "op:mutate" in features
    assert "expr:clip" in features
    assert "numeric:clip" in features


def test_extract_case_features_tracks_numeric_abs_expression():
    case = _case(15, [{"op": "mutate", "column": "x_abs", "expr": {"kind": "abs", "source": "x"}}])

    features = extract_case_features(case)

    assert "op:mutate" in features
    assert "expr:abs" in features
    assert "numeric:abs" in features


def test_extract_case_features_tracks_fill_null_semantics():
    case = _case(2, [{"op": "fill_null", "column": "x", "value": 0}])

    features = extract_case_features(case)

    assert "op:fill_null" in features
    assert "null:fill" in features
    assert "fill_null_type:int" in features
    assert "fill_null:zero" in features
    assert "pattern:fill_null_null_semantics" in features


def test_extract_case_features_tracks_coalesce_semantics():
    case = _case(8, [{"op": "coalesce", "columns": ["g", "id"], "as": "label", "fallback": "missing"}])

    features = extract_case_features(case)

    assert "op:coalesce" in features
    assert "null:coalesce" in features
    assert "coalesce:columns" in features
    assert "coalesce_columns:two" in features
    assert "coalesce:fallback" in features
    assert "pattern:coalesce_null_semantics" in features


def test_extract_case_features_tracks_case_when_semantics():
    case = _case(
        3,
        [
            {
                "op": "case_when",
                "as": "label",
                "condition": {"column": "x", "cmp": ">=", "value": 0},
                "then": "yes",
                "else": "no",
            }
        ],
    )

    features = extract_case_features(case)

    assert "op:case_when" in features
    assert "conditional:case_when" in features
    assert "case_when_type:int" in features
    assert "case_when_cmp:>=" in features
    assert "case_when_output:str" in features
    assert "pattern:conditional_expression" in features


def test_extract_case_features_tracks_union_all_semantics():
    case = _case(4, [{"op": "union_all", "table": "t_append"}])

    features = extract_case_features(case)

    assert "op:union_all" in features
    assert "table:row-append" in features
    assert "union_all:append" in features
    assert "pattern:union_all_row_append" in features


def test_extract_case_features_tracks_drop_nulls_semantics():
    case = _case(5, [{"op": "drop_nulls", "columns": ["g", "x"]}])

    features = extract_case_features(case)

    assert "op:drop_nulls" in features
    assert "null:drop" in features
    assert "drop_nulls:subset" in features
    assert "drop_nulls_columns:two" in features
    assert "pattern:drop_nulls_null_filter" in features


def test_extract_case_features_tracks_semi_and_anti_join_semantics():
    semi = _case(6, [{"op": "semi_join", "table": "t_lookup", "left_on": "id", "right_on": "id"}])
    anti = _case(7, [{"op": "anti_join", "table": "t_lookup", "left_on": "id", "right_on": "id"}])

    semi_features = extract_case_features(semi)
    anti_features = extract_case_features(anti)

    assert "op:semi_join" in semi_features
    assert "join:existence" in semi_features
    assert "membership:semi_join" in semi_features
    assert "pattern:semi_join_membership" in semi_features
    assert "pattern:semi_anti_join_null_keys" in semi_features
    assert "op:anti_join" in anti_features
    assert "membership:anti_join" in anti_features
    assert "pattern:anti_join_exclusion" in anti_features
    assert "pattern:semi_anti_join_null_keys" in anti_features


def test_extract_case_features_tracks_multi_key_groupby():
    case = _case(
        8,
        [
            {
                "op": "groupby",
                "keys": ["g", "flag"],
                "aggs": [{"column": "x", "func": "sum", "as": "sum_x"}],
            }
        ],
    )
    guidance = GuidanceState(targets=["multi_key_groupby"])

    features = extract_case_features(case)
    decision = guidance.choose_case([case])

    assert "groupby:multi-key" in features
    assert "combo_risk:multi_key_groupby" in features
    assert decision.matched_targets == ["multi_key_groupby"]


def test_extract_case_features_tracks_groupby_sorted_input_target():
    case = _case(
        81,
        [
            {
                "op": "groupby",
                "keys": ["g"],
                "aggs": [
                    {"column": "x", "func": "nunique", "as": "nunique_x"},
                    {"column": "flag", "func": "any", "as": "any_flag"},
                ],
            }
        ],
    )
    guidance = GuidanceState(targets=["groupby_sorted_input"])

    features = extract_case_features(case)
    decision = guidance.choose_case([case])

    assert "groupby:exact-aggregate" in features
    assert "groupby:sorted-input" in features
    assert "pattern:groupby_sorted_input" in features
    assert decision.matched_targets == ["groupby_sorted_input"]


def test_extract_case_features_marks_issue_replay_sources():
    replay_case = _case(11, [{"op": "scalar_subquery_probe", "as": "probe_ok"}])
    replay_case.metadata["source_issue"] = "https://github.com/example/project/issues/1"
    inspired_case = _case(12, [{"op": "filter", "column": "x", "cmp": ">=", "value": 0}])
    inspired_case.metadata["source_issue"] = "https://github.com/example/project/issues/2"

    replay_features = extract_case_features(replay_case)
    inspired_features = extract_case_features(inspired_case)

    assert "source_issue:https://github.com/example/project/issues/1" in replay_features
    assert "source_issue:https://github.com/example/project/issues/2" in inspired_features
    assert "source:issue_replay" in replay_features
    assert "source:issue_inspired" not in replay_features
    assert "source:issue_inspired" in inspired_features
    assert "source:issue_replay" not in inspired_features


def test_guidance_prefers_targeted_candidate():
    filter_case = _case(1, [{"op": "filter", "column": "x", "cmp": ">", "value": 0}])
    groupby_case = _case(
        2,
        [{"op": "groupby", "keys": ["g"], "aggs": [{"column": "x", "func": "sum", "as": "sum_x"}]}],
    )
    guidance = GuidanceState(targets=["groupby"])

    decision = guidance.choose_case([filter_case, groupby_case])

    assert decision.case is groupby_case
    assert decision.matched_targets == ["groupby"]
    assert decision.candidate_count == 2


def test_guidance_recognizes_null_groupby_topk_pattern():
    ordinary_case = _case(
        1,
        [{"op": "groupby", "keys": ["g"], "aggs": [{"column": "x", "func": "count", "as": "count_x"}]}],
    )
    topk_case = Case(
        "case-null-groupby-topk",
        2,
        [
            TableData(
                "t0",
                [ColumnSpec("x", "int"), ColumnSpec("s", "str")],
                [{"x": 1, "s": None}],
            )
        ],
        Program(
            "prog-null-groupby-topk",
            2,
            [
                {"op": "mutate", "column": "m_0", "expr": {"kind": "string_length", "source": "s"}},
                {
                    "op": "groupby",
                    "keys": ["m_0"],
                    "aggs": [{"column": "x", "func": "count", "as": "count_x"}],
                },
                {"op": "select", "columns": ["m_0"]},
                {"op": "sort", "columns": ["m_0"], "ascending": False},
                {"op": "limit", "n": 5},
            ],
        ),
    )
    guidance = GuidanceState(targets=["null_groupby_topk"])

    features = extract_case_features(topk_case)
    decision = guidance.choose_case([ordinary_case, topk_case])

    assert "groupby:null-key" in features
    assert "sort:null-order" in features
    assert "pattern:null_groupby_topk" in features
    assert decision.case is topk_case
    assert decision.matched_targets == ["null_groupby_topk"]


def test_guidance_recognizes_null_agg_topk_pattern():
    case = Case(
        "case-null-agg-topk",
        3,
        [
            TableData(
                "t0",
                [ColumnSpec("g", "str", nullable=False), ColumnSpec("x", "int")],
                [{"g": "a", "x": None}, {"g": "b", "x": 5}],
            )
        ],
        Program(
            "prog-null-agg-topk",
            3,
            [
                {"op": "groupby", "keys": ["g"], "aggs": [{"column": "x", "func": "min", "as": "min_x"}]},
                {"op": "select", "columns": ["min_x"]},
                {"op": "sort", "columns": ["min_x"], "ascending": True},
                {"op": "limit", "n": 4},
            ],
        ),
    )
    guidance = GuidanceState(targets=["null_agg_topk"])

    features = extract_case_features(case)
    decision = guidance.choose_case([case])

    assert "groupby:null-agg-output" in features
    assert "sort:null-order" in features
    assert "pattern:null_agg_topk" in features
    assert decision.matched_targets == ["null_agg_topk"]


def test_guidance_recognizes_filter_null_agg_topk_pattern():
    case = generate_case(30, profile="filter_null_agg_topk")
    guidance = GuidanceState(targets=["filter_null_agg_topk"])

    features = extract_case_features(case)
    decision = guidance.choose_case([case])

    assert "groupby:null-agg-output" in features
    assert "sort:null-order" in features
    assert "pattern:filter_null_agg_topk" in features
    assert decision.matched_targets == ["filter_null_agg_topk"]


def test_guidance_recognizes_join_null_agg_topk_pattern():
    case = Case(
        "case-join-null-agg-topk",
        31,
        [
            TableData(
                "t0",
                [ColumnSpec("id", "int", nullable=False), ColumnSpec("g", "str", nullable=False)],
                [{"id": 0, "g": "a"}, {"id": 1, "g": "b"}],
            ),
            TableData(
                "t1",
                [ColumnSpec("id", "int", nullable=False), ColumnSpec("j", "int", nullable=True)],
                [{"id": 0, "j": 5}],
            ),
        ],
        Program(
            "prog-join-null-agg-topk",
            31,
            [
                {"op": "join", "table": "t1", "left_on": "id", "right_on": "id", "how": "left"},
                {"op": "mutate", "column": "m_0", "expr": {"kind": "cast", "source": "j", "to": "float"}},
                {"op": "groupby", "keys": ["g"], "aggs": [{"column": "m_0", "func": "min", "as": "min_j"}]},
                {"op": "select", "columns": ["min_j"]},
                {"op": "sort", "columns": ["min_j"], "ascending": True},
                {"op": "limit", "n": 3},
            ],
        ),
    )
    guidance = GuidanceState(targets=["join_null_agg_topk"])

    features = extract_case_features(case)
    decision = guidance.choose_case([case])

    assert "groupby:null-agg-output" in features
    assert "sort:null-order" in features
    assert "pattern:join_null_agg_topk" in features
    assert decision.matched_targets == ["join_null_agg_topk"]


def test_guidance_generated_target_profiles_stay_aligned_with_patterns():
    profiles = [
        "null_groupby_topk",
        "null_agg_topk",
        "filter_null_agg_topk",
        "join_null_agg_topk",
        "join_null_key_topk",
        "wide_offset_topk",
        "empty_filter_groupby",
        "join_filter_groupby",
        "join_null_truth_filter",
        "float_group_key",
        "join_null_sort",
        "global_null_aggregate",
        "string_count_groupby",
        "unique_count_groupby",
        "bool_null_groupby_agg",
        "large_int_filter_groupby",
        "set_membership_filter",
        "null_predicate_filter",
        "boolean_predicate_filter",
        "post_topk_range_filter",
        "tuple_absence_filter",
        "row_value_absence_filter",
        "running_sum_precision",
        "partitioned_running_sum",
        "path_basename_keyed_pick",
        "sortedness_null_placement",
        "simple_case_random_subject",
        "group_quantile_key_probe",
        "scalar_subquery_double_parentheses",
        "window_avg_rows_frame",
        "struct_distinct_unnest",
        "bit_compare_unequal_length",
        "round_even_float_scale",
        "duckdb_float_literal_precision",
        "polars_timestamp_precision_filter",
        "series_rtruediv_operand_order",
        "pandas_uint64_isin_precision",
        "duckdb_tuple_anti_null_semantics",
        "duckdb_json_predicate_order_semantics",
        "pandas_sparse_array_mask_semantics",
        "polars_float_wrap_numerical_semantics",
        "pandas_index_bool_result_type",
        "polars_empty_literal_groupby_semantics",
        "pandas_arrow_string_eq_sum_semantics",
        "pandas_arrow_timestamp_loc_slice_semantics",
        "pandas_arrow_timestamp_index_attr_semantics",
        "pandas_eval_inplace_aliasing_semantics",
        "pyarrow_dataset_isin_all_match_semantics",
        "pyarrow_large_string_partition_schema_semantics",
        "pyarrow_hash_pivot_wider_order_semantics",
        "pyarrow_list_flatten_parent_indices_semantics",
        "polars_rolling_mean_by_null_count_semantics",
    ]
    for profile in profiles:
        for seed in range(50):
            case = generate_case(seed, profile=profile)
            guidance = GuidanceState(targets=[profile])

            features = extract_case_features(case)
            decision = guidance.choose_case([case])

            assert f"pattern:{profile}" in features
            assert decision.matched_targets == [profile]


def test_guidance_recognizes_distinct_null_topk_pattern():
    case = _case(
        31,
        [
            {"op": "distinct", "columns": ["g"]},
            {"op": "sort", "keys": [{"column": "g", "ascending": True, "nulls": "first"}]},
            {"op": "limit", "n": 1},
        ],
    )
    case.tables[0].rows[1]["g"] = None
    guidance = GuidanceState(targets=["distinct_null_topk"])

    features = extract_case_features(case)
    decision = guidance.choose_case([case])

    assert "sort:null-order" in features
    assert "pattern:distinct_null_topk" in features
    assert decision.matched_targets == ["distinct_null_topk"]


def test_guidance_recognizes_join_filter_groupby_pattern():
    case = Case(
        "case-join-filter-groupby",
        32,
        [
            TableData(
                "t0",
                [
                    ColumnSpec("id", "int", nullable=False),
                    ColumnSpec("g", "str", nullable=False),
                    ColumnSpec("x", "int", nullable=True),
                ],
                [{"id": 0, "g": "a", "x": 1}, {"id": 1, "g": "b", "x": 2}],
            ),
            TableData(
                "t1",
                [
                    ColumnSpec("id", "int", nullable=False),
                    ColumnSpec("j", "int", nullable=True),
                    ColumnSpec("z", "float", nullable=True),
                ],
                [{"id": 0, "j": 5, "z": 1.0}, {"id": 1, "j": 7, "z": 0.5}],
            ),
        ],
        Program(
            "prog-join-filter-groupby",
            32,
            [
                {"op": "join", "table": "t1", "left_on": "id", "right_on": "id", "how": "inner"},
                {"op": "filter", "column": "j", "cmp": ">=", "value": 0},
                {"op": "mutate", "column": "m_0", "expr": {"kind": "add_const", "source": "x", "value": 1}},
                {"op": "groupby", "keys": ["g"], "aggs": [{"column": "m_0", "func": "sum", "as": "sum_m_0"}]},
                {"op": "sort", "columns": ["sum_m_0", "g"], "ascending": True},
                {"op": "limit", "n": 2},
            ],
        ),
    )
    guidance = GuidanceState(targets=["join_filter_groupby"])

    features = extract_case_features(case)
    decision = guidance.choose_case([case])

    assert "pattern:join_filter_groupby" in features
    assert decision.matched_targets == ["join_filter_groupby"]


def test_guidance_recognizes_join_null_truth_filter_pattern():
    case = generate_case(123, profile="join_null_truth_filter")
    guidance = GuidanceState(targets=["join_null_truth_filter"])

    features = extract_case_features(case)
    decision = guidance.choose_case([case])

    assert "filter:truth-test" in features
    assert "filter:truth:is_not_true" in features
    assert "pattern:join_null_truth_filter" in features
    assert decision.matched_targets == ["join_null_truth_filter"]


def test_guidance_recognizes_empty_filter_groupby_pattern():
    case = generate_case(123, profile="empty_filter_groupby")
    guidance = GuidanceState(targets=["empty_filter_groupby"])

    features = extract_case_features(case)
    decision = guidance.choose_case([case])

    assert "filter:empty-output" in features
    assert "pattern:empty_filter_groupby" in features
    assert decision.matched_targets == ["empty_filter_groupby"]


def test_guidance_recognizes_empty_filter_aggregate_pattern():
    case = _case(
        124,
        [
            {"op": "filter", "column": "x", "cmp": ">", "value": 999},
            {
                "op": "aggregate",
                "aggs": [
                    {"column": "id", "func": "count", "as": "row_count"},
                    {"column": "x", "func": "sum", "as": "sum_x"},
                ],
            },
        ],
    )
    guidance = GuidanceState(targets=["empty_filter_aggregate"])

    features = extract_case_features(case)
    decision = guidance.choose_case([case])

    assert "filter:empty-output" in features
    assert "pattern:empty_filter_aggregate" in features
    assert decision.matched_targets == ["empty_filter_aggregate"]


def test_guidance_recognizes_csv_long_numeric_roundtrip_pattern():
    case = generate_case(153, profile="csv_long_numeric_roundtrip")
    guidance = GuidanceState(targets=["csv_long_numeric_roundtrip", "csv_numeric_inference"])

    features = extract_case_features(case)
    decision = guidance.choose_case([case])

    assert "csv:long-numeric-roundtrip" in features
    assert "csv:numeric-inference" in features
    assert "pattern:csv_long_numeric_roundtrip" in features
    assert decision.matched_targets == ["csv_long_numeric_roundtrip", "csv_numeric_inference"]


def test_guidance_uses_actual_left_join_output_for_sort_null_order():
    case = Case(
        "case-join-null-sort",
        4,
        [
            TableData(
                "t0",
                [ColumnSpec("id", "int", nullable=False), ColumnSpec("x", "int", nullable=True)],
                [{"id": 0, "x": 1}, {"id": 1, "x": 2}],
            ),
            TableData(
                "t1",
                [ColumnSpec("id", "int", nullable=False), ColumnSpec("j", "int", nullable=True)],
                [{"id": 1, "j": 10}],
            ),
        ],
        Program(
            "prog-join-null-sort",
            4,
            [
                {"op": "join", "table": "t1", "left_on": "id", "right_on": "id", "how": "left"},
                {"op": "select", "columns": ["id", "j"]},
                {"op": "sort", "columns": ["j", "id"], "ascending": True},
                {"op": "limit", "n": 2},
            ],
        ),
    )

    features = extract_case_features(case)

    assert "sort:null-order" in features
    assert "pattern:join_null_sort" in features


def test_guidance_penalizes_saturated_finding_features():
    join_case = _case(1, [{"op": "join", "table": "t1", "how": "inner", "on": ["id"]}])
    novel_case = _case(2, [{"op": "mutate", "column": "x2", "expr": {"kind": "cast", "column": "x", "to": "str"}}])
    guidance = GuidanceState()

    for idx in range(80):
        guidance.record_result(
            join_case,
            {
                "findings": [
                    {
                        "kind": "semantic_output_mismatch",
                        "root_cause": "join_semantics",
                    }
                ]
            },
        )

    decision = guidance.choose_case([join_case, novel_case])

    assert decision.case is novel_case
    assert decision.score_breakdown["root_saturation_penalty"] <= 0.0


def test_guidance_keeps_target_priority_under_saturation():
    join_case = Case(
        "case-join",
        1,
        [
            TableData(
                "t0",
                [
                    ColumnSpec("id", "int", nullable=False),
                    ColumnSpec("g", "str", nullable=True),
                    ColumnSpec("x", "int", nullable=True),
                ],
                [
                    {"id": 0, "g": "alpha", "x": 1},
                    {"id": 1, "g": "中文", "x": None},
                    {"id": 1, "g": "", "x": -1},
                ],
            ),
            TableData(
                "t1",
                [
                    ColumnSpec("id", "int", nullable=False),
                    ColumnSpec("j", "int", nullable=True),
                ],
                [
                    {"id": 1, "j": 10},
                    {"id": 2, "j": 20},
                ],
            ),
        ],
        Program("prog-join", 1, [{"op": "join", "table": "t1", "left_on": "id", "right_on": "id", "how": "left"}]),
    )
    filter_case = _case(2, [{"op": "filter", "column": "x", "cmp": ">", "value": 0}])
    guidance = GuidanceState(targets=["join"])

    for idx in range(80):
        guidance.record_result(
            join_case,
            {
                "findings": [
                    {
                        "kind": "semantic_output_mismatch",
                        "root_cause": "join_semantics",
                    }
                ]
            },
        )

    decision = guidance.choose_case([join_case, filter_case])

    assert decision.case is join_case
    assert decision.matched_targets == ["join"]
    assert decision.score_breakdown["target_bonus"] == 3.0


def test_guidance_prioritizes_specific_pattern_target_over_generic_target_count():
    generic_case = _case(
        1,
        [
            {"op": "filter", "column": "x", "cmp": ">=", "value": 0},
            {"op": "mutate", "column": "m_0", "expr": {"kind": "add_const", "source": "x", "value": 1}},
            {"op": "groupby", "keys": ["g"], "aggs": [{"column": "m_0", "func": "sum", "as": "sum_m_0"}]},
            {"op": "sort", "columns": ["sum_m_0"], "ascending": True},
            {"op": "limit", "n": 2},
        ],
    )
    pattern_case = generate_case(40, profile="discovery")
    guidance_targets = [
        "common_workflow",
        "operation_combo",
        "topk",
        "join",
        "join_null_key_topk",
        "groupby",
        "mutate",
        "filter",
        "nulls",
        "aggregation",
        "sort_limit",
        "expressions",
    ]
    guidance = GuidanceState(targets=guidance_targets)

    generic_decision = guidance.choose_case([generic_case])
    pattern_decision = guidance.choose_case([pattern_case])
    decision = guidance.choose_case([generic_case, pattern_case])

    assert "join_null_key_topk" in pattern_decision.matched_targets
    assert len(generic_decision.matched_targets) > len(pattern_decision.matched_targets)
    assert decision.case is pattern_case
    assert decision.score_breakdown["specific_target_matches"] == 1.0
    assert decision.score_breakdown["target_priority"] > generic_decision.score_breakdown["target_priority"]


def test_guidance_prioritizes_issue_template_over_organic_pattern_match():
    template_case = generate_case(260020, profile="discovery")
    organic_case = _organic_discovery_case_matching_pattern("join_null_key_topk")
    guidance_targets = [
        "common_workflow",
        "operation_combo",
        "topk",
        "join",
        "join_null_key_topk",
        "groupby",
        "mutate",
        "filter",
        "nulls",
        "aggregation",
        "sort_limit",
        "expressions",
    ]
    guidance = GuidanceState(targets=guidance_targets)

    template_decision = guidance.choose_case([template_case])
    organic_decision = guidance.choose_case([organic_case])
    decision = guidance.choose_case([organic_case, template_case])

    assert template_case.metadata["mixed_generator_profile"] == "join_null_key_topk"
    assert "join_null_key_topk" in template_decision.matched_targets
    assert "join_null_key_topk" in organic_decision.matched_targets
    assert organic_decision.score_breakdown["target_template_matches"] == 0.0
    assert template_decision.score_breakdown["target_template_matches"] == 1.0
    assert decision.case is template_case


def test_guidance_template_priority_decays_after_repeated_template_selection():
    template_case = generate_case(260020, profile="discovery")
    organic_case = _organic_discovery_case_matching_pattern("join_null_key_topk")
    guidance = GuidanceState(
        targets=[
            "common_workflow",
            "operation_combo",
            "topk",
            "join",
            "join_null_key_topk",
            "groupby",
            "mutate",
            "filter",
            "nulls",
            "aggregation",
            "sort_limit",
            "expressions",
        ]
    )

    first = guidance.choose_case([organic_case, template_case])
    for _ in range(8):
        guidance.record_result(template_case, {"findings": []})
    after_repeated_template_hits = guidance.choose_case([organic_case, template_case])

    assert first.case is template_case
    assert first.score_breakdown["target_template_bonus"] > 0.0
    assert after_repeated_template_hits.case is organic_case
    assert (
        after_repeated_template_hits.score_breakdown["target_priority"]
        >= guidance.choose_case([template_case]).score_breakdown["target_priority"]
    )


def test_guidance_uses_data_sensitivity_and_path_coverage_breakdown():
    simple_case = _case(1, [{"op": "select", "columns": ["id"]}])
    sensitive_case = _case(
        2,
        [
            {"op": "filter", "column": "x", "cmp": ">=", "value": 0},
            {"op": "mutate", "column": "m_0", "expr": {"kind": "string_length", "source": "g"}},
        ],
    )
    guidance = GuidanceState()

    decision = guidance.choose_case([simple_case, sensitive_case])

    assert decision.case is sensitive_case
    assert "data_sensitivity" in decision.score_breakdown
    assert "path_coverage_proxy" in decision.score_breakdown
    assert "frontier_conformance" in decision.score_breakdown
    assert "contribution_potential" in decision.score_breakdown
    assert decision.score_breakdown["data_sensitivity"] > 0.0
    assert decision.score_breakdown["path_coverage_proxy"] > 0.0
    assert decision.score_breakdown["frontier_conformance"] > 0.0


def test_guidance_scores_realistic_operation_combos():
    simple_case = _case(1, [{"op": "select", "columns": ["id"]}])
    combo_case = _case(
        2,
        [
            {"op": "filter", "column": "x", "cmp": ">=", "value": 0},
            {"op": "mutate", "column": "m_0", "expr": {"kind": "add_const", "source": "x", "value": 1}},
            {"op": "select", "columns": ["id", "m_0"]},
            {"op": "sort", "columns": ["m_0", "id"], "ascending": True},
            {"op": "limit", "n": 2},
        ],
    )
    guidance = GuidanceState(targets=["common_workflow", "topk"])

    features = extract_case_features(combo_case)
    decision = guidance.choose_case([simple_case, combo_case])

    assert "combo:filter_mutate_select_sort_limit" in features
    assert "combo_frequency:high" in features
    assert "combo_risk:topk_ordering" in features
    assert decision.case is combo_case
    assert decision.matched_targets == ["common_workflow", "topk"]
    assert decision.score_breakdown["combo_priority"] > 0.0


def test_guidance_updates_online_feature_weights_from_feedback():
    combo_case = _case(
        3,
        [
            {"op": "filter", "column": "x", "cmp": ">=", "value": 0},
            {"op": "mutate", "column": "m_0", "expr": {"kind": "add_const", "source": "x", "value": 1}},
            {"op": "select", "columns": ["id", "m_0"]},
            {"op": "sort", "columns": ["m_0", "id"], "ascending": True},
            {"op": "limit", "n": 2},
        ],
    )
    guidance = GuidanceState()

    before = guidance.choose_case([combo_case])
    guidance.record_result(
        combo_case,
        {
            "findings": [{"kind": "semantic_output_mismatch", "root_cause": "topk_ordering"}],
            "is_new_behavior": True,
            "preflight": {"valid": True, "fallback_used": False},
        },
    )
    after = guidance.choose_case([combo_case])

    assert before.score_breakdown["online_weight_mean"] == 1.0
    assert after.score_breakdown["online_weight_mean"] > before.score_breakdown["online_weight_mean"]
    assert after.score_breakdown["online_weight_updates"] == 1.0
    assert guidance.online_weights.multiplier("combo:filter_mutate_select_sort_limit") > 1.0
    assert after.online_weights


def test_guidance_online_weights_learn_quality_archive_context_features():
    case = _case(304, [{"op": "filter", "column": "x", "cmp": ">=", "value": 0}])
    case.metadata["quality_archive_context"] = {
        "cluster_key": "profile=generic|targets=semantic_family_cast_semantics|ops=filter",
        "archive_known": True,
        "archive_elite_indexes": [0],
        "archive_seed_count": 1,
        "archive_outcome_count": 3,
        "archive_cluster_reward": 0.7,
        "archive_health_penalty": 0.0,
        "cluster_count": 2,
        "cluster_feedback_reward": 0.6,
        "cluster_novelty_score": 0.05,
        "recent_cluster_pulls": 0,
    }
    guidance = GuidanceState()

    guidance.record_result(
        case,
        {
            "findings": [{"kind": "semantic_output_mismatch", "root_cause": "cast_semantics"}],
            "is_new_behavior": True,
            "preflight": {"valid": True, "fallback_used": False},
        },
    )

    assert "quality_archive:known" in guidance.online_weights.feature_stats
    assert "quality_archive:reward:positive" in guidance.online_weights.feature_stats
    assert guidance.online_weights.multiplier("quality_archive:known") > 1.0
    assert guidance.online_weights.prefix_stats["quality_archive"].pulls > 0


def test_guidance_online_weights_do_not_overreact_to_single_large_reward():
    combo_case = _case(
        303,
        [
            {"op": "filter", "column": "x", "cmp": ">=", "value": 0},
            {"op": "mutate", "column": "m_0", "expr": {"kind": "add_const", "source": "x", "value": 1}},
            {"op": "select", "columns": ["id", "m_0"]},
            {"op": "sort", "columns": ["m_0", "id"], "ascending": True},
            {"op": "limit", "n": 2},
        ],
    )
    guidance = GuidanceState()
    feature = "combo:filter_mutate_select_sort_limit"

    guidance.online_weights.record(extract_case_features(combo_case), 100.0)

    snapshot = guidance.online_weights.snapshot(limit=128)
    row = next(item for item in snapshot if item["feature"] == feature)

    assert row["mean_reward"] == 100.0
    assert 0.0 < row["reward_signal"] < row["mean_reward"]
    assert 1.0 < row["multiplier"] < guidance.online_weights.max_multiplier


def test_guidance_online_weights_prefer_repeated_stable_signal_over_single_spike():
    spiky_case = _case(
        304,
        [
            {"op": "filter", "column": "x", "cmp": ">=", "value": 0},
            {"op": "select", "columns": ["id", "x"]},
        ],
    )
    stable_case = _case(
        305,
        [
            {"op": "filter", "column": "x", "cmp": ">=", "value": 0},
            {"op": "mutate", "column": "m_0", "expr": {"kind": "add_const", "source": "x", "value": 1}},
            {"op": "select", "columns": ["id", "m_0"]},
            {"op": "sort", "columns": ["m_0", "id"], "ascending": True},
            {"op": "limit", "n": 2},
        ],
    )
    guidance = GuidanceState()

    guidance.online_weights.record(extract_case_features(spiky_case), 100.0)
    for _ in range(3):
        guidance.online_weights.record(extract_case_features(stable_case), 4.0)

    assert (
        guidance.online_weights.multiplier("combo:filter_mutate_select_sort_limit")
        > guidance.online_weights.multiplier("combo:filter_select")
    )


def test_guidance_reward_discounts_repeated_candidate_bug_family():
    row = {
        "findings": [
            {
                "root_cause": "topk_filter_pushdown",
                "triage_verdict": "candidate_implementation_bug",
                "suspicious_backends": ["datafusion"],
                "signature": "sig-a",
            }
        ],
        "is_new_behavior": True,
        "preflight": {"valid": True, "fallback_used": False},
    }

    fresh = _guidance_reward(row)
    repeated = _guidance_reward(
        row,
        root_cause_counts={"topk_filter_pushdown": 12},
        candidate_bug_family_counts={"topk_filter_pushdown@datafusion": 12},
        candidate_bug_signature_counts={"sig-a": 1},
    )

    assert fresh == 4.5
    assert 0.0 < repeated < 1.0


def test_guidance_features_keep_feedback_mutation_issue_profile():
    base = generate_case(260000, profile="discovery")

    mutated = mutate_case_with_metadata(base, 260001).case
    features = extract_case_features(mutated)

    assert "source:feedback_mutation" in features
    assert "mutation_depth:one" in features
    assert "generator_profile:discovery" in features
    assert f"mixed_generator_profile:{base.metadata['mixed_generator_profile']}" in features


def test_guidance_moves_off_repeated_discovery_template_findings():
    repeated_template = generate_case(55, profile="discovery")
    alternate_template = generate_case(56, profile="discovery")
    guidance = GuidanceState(
        targets=[
            "common_workflow",
            "operation_combo",
            "join",
            "groupby",
            "mutate",
            "filter",
            "aggregation",
            "sort_limit",
            "topk",
            "expressions",
            "nulls",
            "strings",
        ]
    )
    for idx in range(8):
        guidance.record_result(
            repeated_template,
            {
                "findings": [
                    {
                        "root_cause": "topk_filter_pushdown",
                        "triage_verdict": "candidate_implementation_bug",
                        "suspicious_backends": ["datafusion"],
                        "signature": f"sig-{idx}",
                    }
                ]
            },
        )

    repeated_decision = guidance.choose_case([repeated_template])
    decision = guidance.choose_case([repeated_template, alternate_template])

    assert repeated_decision.score_breakdown["profile_saturation_penalty"] < 0.0
    assert decision.case is alternate_template


def test_guidance_family_saturation_demotes_known_specific_family():
    saturated_case = _case(
        70,
        [
            {
                "op": "mutate",
                "column": "ratio",
                "expr": {"kind": "reverse_division_columns", "source": "x", "numerator": 1.0},
            }
        ],
    )
    fresh_case = _case(71, [{"op": "filter", "column": "x", "cmp": ">=", "value": 0}])
    guidance = GuidanceState(
        targets=["polars_reverse_division_columns", "filter"],
        active_backends=["polars"],
        known_saturated_bug_families=["reverse_division_operand_order@polars"],
    )

    saturated_decision = guidance.choose_case([saturated_case])
    decision = guidance.choose_case([saturated_case, fresh_case])

    assert guidance.predicted_saturated_family_roots(saturated_case) == ["reverse_division_operand_order"]
    assert guidance.predicted_saturated_family_roots(fresh_case) == []
    assert "pattern:polars_reverse_division_columns" in saturated_decision.features
    assert saturated_decision.score_breakdown["family_saturation_active"] == 1.0
    assert saturated_decision.score_breakdown["family_saturation_penalty"] < 0.0
    assert decision.case is fresh_case


def test_guidance_family_saturation_uses_untargeted_fallback():
    saturated_case = _case(
        170,
        [
            {
                "op": "mutate",
                "column": "ratio",
                "expr": {"kind": "reverse_division_columns", "source": "x", "numerator": 1.0},
            }
        ],
    )
    fallback_case = _case(171, [{"op": "select", "columns": ["id"]}])
    guidance = GuidanceState(
        targets=["polars_reverse_division_columns"],
        active_backends=["polars"],
        known_saturated_bug_families=["reverse_division_operand_order@polars"],
    )

    decision = guidance.choose_case([saturated_case, fallback_case])

    assert decision.case is fallback_case
    assert decision.matched_targets == []


def test_guidance_family_saturation_catches_joined_order_offset_without_projection():
    saturated_case = _case(
        172,
        [
            {"op": "join", "table": "t1", "left_on": "id", "right_on": "id", "how": "inner"},
            {"op": "sort", "columns": ["x"], "ascending": False},
            {"op": "offset", "n": 1},
            {"op": "sort", "columns": ["id"], "ascending": True},
        ],
    )
    fallback_case = _case(173, [{"op": "filter", "column": "x", "cmp": ">=", "value": 0}])
    guidance = GuidanceState(
        targets=["join", "sort_offset"],
        active_backends=["datafusion"],
        known_saturated_bug_families=["joined_order_offset_projection@datafusion"],
    )

    saturated_decision = guidance.choose_case([saturated_case])
    decision = guidance.choose_case([saturated_case, fallback_case])

    assert saturated_decision.score_breakdown["family_saturation_active"] == 1.0
    assert decision.case is fallback_case


def test_guidance_family_saturation_catches_topk_filter_combo_risk():
    saturated_case = _case(
        174,
        [
            {"op": "sort", "columns": ["x"], "ascending": False},
            {"op": "offset", "n": 1},
            {"op": "filter", "column": "x", "cmp": ">=", "value": 0},
        ],
    )
    fallback_case = _case(175, [{"op": "select", "columns": ["id"]}])
    guidance = GuidanceState(
        targets=["sort_offset"],
        active_backends=["datafusion"],
        known_saturated_bug_families=["topk_filter_pushdown@datafusion"],
    )

    saturated_decision = guidance.choose_case([saturated_case])
    decision = guidance.choose_case([saturated_case, fallback_case])

    assert "semantic_signal:topk_filter_pushdown" in saturated_decision.features
    assert "combo_risk:topk_filter_pushdown" in saturated_decision.features
    assert saturated_decision.score_breakdown["family_saturation_active"] == 1.0
    assert decision.case is fallback_case


def test_guidance_family_saturation_catches_ordered_topk_projection():
    saturated_case = _case(
        178,
        [
            {"op": "sort", "columns": ["x", "g"], "ascending": False},
            {"op": "offset", "n": 1},
            {"op": "select", "columns": ["g", "id", "x"]},
            {"op": "sort", "columns": ["id"], "ascending": False},
        ],
    )
    fallback_case = _case(179, [{"op": "filter", "column": "x", "cmp": ">=", "value": 0}])
    guidance = GuidanceState(
        targets=["sort_offset"],
        active_backends=["datafusion"],
        known_saturated_bug_families=["ordered_topk_projection@datafusion"],
    )

    saturated_decision = guidance.choose_case([saturated_case])
    decision = guidance.choose_case([saturated_case, fallback_case])

    assert "semantic_signal:projection_ordering" in saturated_decision.features
    assert "combo_risk:projection_ordering" in saturated_decision.features
    assert saturated_decision.score_breakdown["family_saturation_active"] == 1.0
    assert decision.case is fallback_case


def test_guidance_restores_legacy_combo_risk_state_as_canonical_semantic_signal():
    restored = GuidanceState.from_state_dict(
        {
            "feature_counts": {"combo_risk:topk_filter_pushdown": 3, "op:sort": 2},
            "finding_feature_counts": {"combo_risk:topk_filter_pushdown": 1},
            "online_weights": {
                "total_updates": 2,
                "feature_stats": {
                    "combo_risk:topk_filter_pushdown": {"pulls": 2, "total_reward": 1.5},
                },
                "prefix_stats": {
                    "combo_risk": {"pulls": 2, "total_reward": 1.5},
                },
            },
        },
        targets=["topk"],
        active_backends=["datafusion"],
    )

    assert restored.feature_counts["semantic_signal:topk_filter_pushdown"] == 3
    assert restored.feature_counts["combo_risk:topk_filter_pushdown"] == 0
    assert restored.finding_feature_counts["semantic_signal:topk_filter_pushdown"] == 1
    assert "semantic_signal:topk_filter_pushdown" in restored.online_weights.feature_stats
    assert "combo_risk:topk_filter_pushdown" not in restored.online_weights.feature_stats
    assert restored.online_weights.multiplier("semantic_signal:topk_filter_pushdown") > 1.0


def test_guidance_family_saturation_catches_negative_zero_comparison():
    saturated_case = _case(
        176,
        [
            {
                "op": "mutate",
                "column": "m_0",
                "expr": {"kind": "arith_const", "source": "x", "op": "mul", "value": -1},
            },
            {"op": "filter", "column": "m_0", "cmp": "ge_is_not_true", "value": 0.0},
        ],
    )
    fallback_case = _case(177, [{"op": "select", "columns": ["id"]}])
    guidance = GuidanceState(
        targets=["mutate", "filter"],
        active_backends=["datafusion"],
        known_saturated_bug_families=["negative_zero_comparison@datafusion"],
    )

    saturated_decision = guidance.choose_case([saturated_case])
    decision = guidance.choose_case([saturated_case, fallback_case])

    assert saturated_decision.score_breakdown["family_saturation_active"] == 1.0
    assert decision.case is fallback_case


def test_guidance_family_saturation_demotes_runtime_repeated_family():
    saturated_case = _case(
        72,
        [
            {
                "op": "mutate",
                "column": "ratio",
                "expr": {"kind": "reverse_division_columns", "source": "x", "numerator": 1.0},
            }
        ],
    )
    fresh_case = _case(73, [{"op": "filter", "column": "x", "cmp": ">=", "value": 0}])
    guidance = GuidanceState(
        targets=["polars_reverse_division_columns", "filter"],
        active_backends=["polars"],
        family_saturation_threshold=3,
    )
    for idx in range(3):
        guidance.record_result(
            saturated_case,
            {
                "findings": [
                    {
                        "root_cause": "reverse_division_operand_order",
                        "triage_verdict": "candidate_implementation_bug",
                        "suspicious_backends": ["polars"],
                        "signature": f"sig-runtime-{idx}",
                    }
                ],
                "preflight": {"valid": True, "fallback_used": False},
            },
        )

    saturated_decision = guidance.choose_case([saturated_case])
    decision = guidance.choose_case([saturated_case, fresh_case])

    assert saturated_decision.score_breakdown["family_saturation_active"] == 1.0
    assert saturated_decision.score_breakdown["family_saturation_penalty"] < 0.0
    assert decision.case is fresh_case


def test_candidate_family_prefilter_uses_lightweight_root_matcher_without_case_analysis(monkeypatch):
    saturated_case = _case(
        272,
        [
            {
                "op": "mutate",
                "column": "ratio",
                "expr": {"kind": "reverse_division_columns", "source": "x", "numerator": 1.0},
            }
        ],
    )
    guidance = GuidanceState(
        targets=["polars_reverse_division_columns"],
        active_backends=["polars"],
        family_saturation_threshold=3,
    )
    for idx in range(3):
        guidance.record_result(
            saturated_case,
            {
                "findings": [
                    {
                        "root_cause": "reverse_division_operand_order",
                        "triage_verdict": "candidate_implementation_bug",
                        "suspicious_backends": ["polars"],
                        "signature": f"sig-prefilter-{idx}",
                    }
                ],
                "preflight": {"valid": True, "fallback_used": False},
            },
        )

    def _unexpected_case_analysis(self, case):
        raise AssertionError(f"candidate prefilter should not call full case analysis for {case.case_id}")

    monkeypatch.setattr(GuidanceState, "_case_analysis", _unexpected_case_analysis)

    assert guidance.predicted_saturated_family_roots_for_candidate(
        saturated_case,
        include_known_families=False,
    ) == ["reverse_division_operand_order"]


def test_guidance_reward_excludes_known_saturated_candidate_bug_family():
    row = {
        "findings": [
            {
                "root_cause": "reverse_division_operand_order",
                "triage_verdict": "candidate_implementation_bug",
                "suspicious_backends": ["polars"],
                "signature": "sig-reverse-division",
            }
        ],
        "preflight": {"valid": True, "fallback_used": False},
    }

    reward = _guidance_reward(
        row,
        known_saturated_bug_families=["reverse_division_operand_order@polars"],
    )

    assert reward < 0.0


def test_guidance_reward_penalizes_resolved_semantic_divergence():
    resolved_row = {
        "findings": [
            {
                "root_cause": "groupby_aggregation",
                "triage_verdict": "expected_semantic_divergence",
                "suspicious_backends": ["duckdb"],
                "signature": "sig-resolved-semantic",
            }
        ],
        "preflight": {"valid": True, "fallback_used": False},
    }
    unresolved_row = {
        "findings": [
            {
                "root_cause": "groupby_aggregation",
                "triage_verdict": "semantic_divergence_needs_confirmation",
                "suspicious_backends": ["duckdb"],
                "signature": "sig-unresolved-semantic",
            }
        ],
        "preflight": {"valid": True, "fallback_used": False},
    }

    assert _guidance_reward(resolved_row) < 0.0
    assert _guidance_reward(unresolved_row) > 0.0


def test_guidance_issue_replay_saturation_demotes_repeated_replay_family():
    replay_case = generate_case(2159, profile="scalar_subquery_double_parentheses")
    fresh_case = _case(74, [{"op": "filter", "column": "x", "cmp": ">=", "value": 0}])
    guidance = GuidanceState(
        targets=["scalar_subquery_double_parentheses", "filter"],
        active_backends=["duckdb"],
        issue_replay_saturation_threshold=2,
    )
    for idx in range(2):
        guidance.record_result(
            replay_case,
            {
                "findings": [
                    {
                        "root_cause": "scalar_subquery_double_parentheses",
                        "triage_verdict": "candidate_implementation_bug",
                        "suspicious_backends": ["duckdb"],
                        "signature": f"sig-replay-{idx}",
                        "discovery_origin": "issue_replay",
                    }
                ],
                "preflight": {"valid": True, "fallback_used": False},
            },
        )

    replay_decision = guidance.choose_case([replay_case])
    decision = guidance.choose_case([replay_case, fresh_case])

    assert replay_decision.score_breakdown["issue_replay_saturation_active"] == 1.0
    assert replay_decision.score_breakdown["issue_replay_saturation_penalty"] < 0.0
    assert decision.case is fresh_case


def test_guidance_global_issue_replay_saturation_demotes_distinct_replay_probe():
    replay_case = _case(75, [{"op": "struct_distinct_probe", "as": "probe_ok"}])
    replay_case.metadata["source_issue"] = "https://github.com/example/project/issues/99"
    fresh_case = _case(76, [{"op": "filter", "column": "x", "cmp": ">=", "value": 0}])
    guidance = GuidanceState(
        targets=["struct_distinct_unnest"],
        active_backends=["duckdb"],
        issue_replay_saturation_threshold=99,
        issue_replay_global_saturation_threshold=4,
        issue_replay_global_saturation_penalty=1.5,
    )
    replay_roots = [
        "scalar_subquery_double_parentheses",
        "window_avg_rows_frame",
        "bit_compare_unequal_length",
        "round_even_float_scale",
        "duckdb_float_literal_precision",
        "polars_timestamp_precision_filter",
    ]
    for idx, root in enumerate(replay_roots):
        guidance.record_result(
            replay_case,
            {
                "findings": [
                    {
                        "root_cause": root,
                        "triage_verdict": "candidate_implementation_bug",
                        "suspicious_backends": ["duckdb"],
                        "signature": f"sig-replay-global-{idx}",
                        "discovery_origin": "issue_replay",
                    }
                ],
                "preflight": {"valid": True, "fallback_used": False},
            },
        )

    replay_decision = guidance.choose_case([replay_case])
    decision = guidance.choose_case([replay_case, fresh_case])

    assert guidance.issue_replay_count == len(replay_roots)
    assert replay_decision.score_breakdown["issue_replay_global_saturation_active"] == 1.0
    assert replay_decision.score_breakdown["issue_replay_global_saturation_penalty"] < 0.0
    assert replay_decision.score_breakdown["issue_replay_saturation_active"] == 1.0
    assert decision.case is fresh_case


def test_guidance_issue_inspired_source_saturation_demotes_repeated_source_issue():
    inspired_case = _case(77, [{"op": "filter", "column": "x", "cmp": ">=", "value": 0}])
    inspired_case.metadata["source_issue"] = "https://github.com/apache/datafusion/issues/22190"
    fresh_case = _case(78, [{"op": "mutate", "column": "m_0", "expr": {"kind": "add_const", "source": "x", "value": 1}}])
    guidance = GuidanceState(
        targets=["filter"],
        active_backends=["datafusion"],
        issue_inspired_source_saturation_threshold=3,
        issue_inspired_source_saturation_penalty=1.25,
    )
    for idx in range(3):
        guidance.record_result(
            inspired_case,
            {
                "findings": [
                    {
                        "root_cause": "grouped_topk_null_sort_key",
                        "triage_verdict": "candidate_implementation_bug",
                        "suspicious_backends": ["datafusion"],
                        "signature": f"sig-issue-inspired-{idx}",
                        "discovery_origin": "issue_inspired",
                        "source_issue": "https://github.com/apache/datafusion/issues/22190",
                    }
                ],
                "preflight": {"valid": True, "fallback_used": False},
            },
        )

    inspired_decision = guidance.choose_case([inspired_case])
    decision = guidance.choose_case([inspired_case, fresh_case])

    assert guidance.issue_inspired_source_counts["https://github.com/apache/datafusion/issues/22190"] == 3
    assert inspired_decision.score_breakdown["issue_inspired_source_saturation_active"] == 1.0
    assert inspired_decision.score_breakdown["issue_inspired_source_saturation_penalty"] < 0.0
    assert decision.case is fresh_case


def test_guidance_issue_inspired_source_saturation_tracks_executed_cases_without_findings():
    repeated_case = _case(79, [{"op": "filter", "column": "x", "cmp": ">=", "value": 0}])
    repeated_case.metadata["source_issue"] = "https://github.com/pola-rs/polars/issues/27726"
    fresh_case = _case(80, [{"op": "mutate", "column": "m_0", "expr": {"kind": "add_const", "source": "x", "value": 1}}])
    guidance = GuidanceState(
        targets=["filter"],
        issue_inspired_source_saturation_threshold=2,
        issue_inspired_source_saturation_penalty=1.25,
    )
    for _ in range(2):
        guidance.record_result(
            repeated_case,
            {
                "findings": [],
                "preflight": {"valid": True, "fallback_used": False},
            },
        )

    repeated_decision = guidance.choose_case([repeated_case])
    decision = guidance.choose_case([repeated_case, fresh_case])

    assert guidance.issue_inspired_source_counts["https://github.com/pola-rs/polars/issues/27726"] == 2
    assert repeated_decision.score_breakdown["issue_inspired_source_saturation_active"] == 1.0
    assert repeated_decision.score_breakdown["issue_inspired_source_saturation_penalty"] < 0.0
    assert decision.case is fresh_case


def test_guidance_profile_saturation_tracks_executed_issue_focus_templates_without_findings():
    repeated_case = _case(81, [{"op": "filter", "column": "x", "cmp": ">=", "value": 0}])
    repeated_case.metadata.update(
        {
            "generator_profile": "issue_focus",
            "mixed_generator_profile": "set_membership_filter",
        }
    )
    fresh_case = _case(82, [{"op": "filter", "column": "x", "cmp": "!=", "value": 0}])
    fresh_case.metadata.update(
        {
            "generator_profile": "issue_focus",
            "mixed_generator_profile": "null_predicate_filter",
        }
    )
    guidance = GuidanceState(targets=["set_membership_filter", "null_predicate_filter", "filter"])
    for _ in range(5):
        guidance.record_result(
            repeated_case,
            {
                "findings": [],
                "preflight": {"valid": True, "fallback_used": False},
            },
        )

    repeated_decision = guidance.choose_case([repeated_case])
    decision = guidance.choose_case([repeated_case, fresh_case])

    assert guidance.feature_counts["mixed_generator_profile:set_membership_filter"] == 5
    assert repeated_decision.score_breakdown["profile_saturation_active"] == 1.0
    assert repeated_decision.score_breakdown["profile_saturation_penalty"] < 0.0
    assert decision.case is fresh_case


def test_guidance_profile_saturation_breaks_ties_between_targeted_profiles():
    repeated_case = _case(
        83,
        [
            {"op": "join", "table": "t1", "left_on": "id", "right_on": "id", "how": "left"},
            {"op": "sort", "columns": ["x", "g", "id"], "ascending": True},
            {"op": "groupby", "keys": ["g"], "aggs": [{"column": "x", "func": "count", "as": "count_x"}]},
            {"op": "select", "columns": ["g", "count_x"]},
            {"op": "sort", "columns": ["count_x", "g"], "ascending": False},
            {"op": "limit", "n": 3},
        ],
    )
    repeated_case.metadata.update(
        {
            "generator_profile": "issue_focus",
            "mixed_generator_profile": "join_ordered_agg_topk",
        }
    )
    less_repeated_case = _case(84, [{"op": "filter", "column": "g", "cmp": "in_set", "value": ["alpha"]}])
    less_repeated_case.metadata.update(
        {
            "generator_profile": "issue_focus",
            "mixed_generator_profile": "set_membership_filter",
        }
    )
    guidance = GuidanceState(
        targets=["join_ordered_agg_topk", "set_membership_filter", "join", "filter", "groupby", "sort_limit"]
    )
    for _ in range(20):
        guidance.record_result(repeated_case, {"findings": [], "preflight": {"valid": True, "fallback_used": False}})
    for _ in range(4):
        guidance.record_result(
            less_repeated_case,
            {"findings": [], "preflight": {"valid": True, "fallback_used": False}},
        )

    repeated_decision = guidance.choose_case([repeated_case])
    less_repeated_decision = guidance.choose_case([less_repeated_case])
    decision = guidance.choose_case([repeated_case, less_repeated_case])

    assert repeated_decision.score_breakdown["profile_saturation_active"] == 1.0
    assert less_repeated_decision.score_breakdown["profile_saturation_active"] == 1.0
    assert (
        repeated_decision.score_breakdown["profile_saturation_penalty"]
        < less_repeated_decision.score_breakdown["profile_saturation_penalty"]
    )
    assert decision.case is less_repeated_case


def test_guidance_frontier_conformance_prefers_boundary_case():
    simple_case = _case(1, [{"op": "select", "columns": ["id"]}])
    boundary_case = _case(2, [{"op": "filter", "column": "x", "cmp": "==", "value": 1}])
    guidance = GuidanceState()

    decision = guidance.choose_case([simple_case, boundary_case])

    assert decision.case is boundary_case
    assert "filter:exact-hit" in decision.frontier_buckets
    assert decision.score_breakdown["frontier_conformance"] >= 0.8


def test_guidance_prunes_obviously_redundant_candidates():
    redundant_case = _case(1, [{"op": "select", "columns": ["id"]}])
    useful_case = _case(2, [{"op": "filter", "column": "x", "cmp": "==", "value": 1}])
    guidance = GuidanceState()

    for _ in range(40):
        guidance.record_result(redundant_case, {"findings": []})

    decision = guidance.choose_case([redundant_case, useful_case])

    assert decision.case is useful_case
    assert decision.contributing_candidate_count == 1
    assert decision.pruned_candidate_count == 1


def test_parse_guidance_targets_ignores_empty_parts():
    assert parse_guidance_targets("groupby, ,nulls") == ["groupby", "nulls"]
