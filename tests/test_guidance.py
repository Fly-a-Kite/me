from datadiff.datagen import generate_case
from datadiff.dsl import Case, ColumnSpec, Program, TableData
from datadiff.guidance import GuidanceState, extract_case_features, parse_guidance_targets


def _case(seed: int, operations: list[dict]) -> Case:
    table = TableData(
        name="t0",
        columns=[
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("g", "str", nullable=True),
            ColumnSpec("x", "int", nullable=True),
        ],
        rows=[
            {"id": 0, "g": "alpha", "x": 1},
            {"id": 1, "g": "中文", "x": None},
            {"id": 1, "g": "", "x": -1},
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}",
        seed=seed,
        tables=[table],
        program=Program(program_id=f"prog-{seed:08d}", seed=seed, operations=operations),
    )


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
        "set_membership_filter",
        "null_predicate_filter",
        "boolean_predicate_filter",
        "post_topk_range_filter",
        "tuple_absence_filter",
        "running_sum_precision",
        "sortedness_null_placement",
        "simple_case_random_subject",
        "group_quantile_key_probe",
        "scalar_subquery_double_parentheses",
        "window_avg_rows_frame",
    ]
    for profile in profiles:
        for seed in range(50):
            case = generate_case(seed, profile=profile)
            guidance = GuidanceState(targets=[profile])

            features = extract_case_features(case)
            decision = guidance.choose_case([case])

            assert f"pattern:{profile}" in features
            assert decision.matched_targets == [profile]


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
    pattern_case = generate_case(40, profile="bughunt")
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
    template_case = generate_case(260020, profile="bughunt")
    organic_case = generate_case(260026, profile="bughunt")
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
    template_case = generate_case(260020, profile="bughunt")
    organic_case = generate_case(260026, profile="bughunt")
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
        > guidance.choose_case([template_case]).score_breakdown["target_priority"]
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
