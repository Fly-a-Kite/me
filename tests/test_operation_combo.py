from datadiff.operation_combo import classify_operation_combo, describe_operation_combo, summarize_operation_combo


def test_classify_operation_combo_records_frequency_priority_and_risks():
    combo = classify_operation_combo(
        [
            {"op": "join"},
            {"op": "filter"},
            {"op": "mutate"},
            {"op": "groupby"},
            {"op": "select"},
            {"op": "sort"},
            {"op": "limit"},
        ]
    )

    assert combo["template"] == "join_filter_mutate_groupby_select_sort_limit"
    assert combo["frequency_bucket"] == "high"
    assert combo["priority"] > 1.0
    assert combo["correctness_risks"] == [
        "join_cardinality",
        "join_filter_pushdown",
        "filter_mutate_dependency",
        "groupby_aggregation",
        "grouped_topk",
        "topk_ordering",
        "projection_ordering",
    ]
    assert combo["has_join"] is True
    assert combo["has_groupby"] is True
    assert combo["has_sort_limit"] is True


def test_classify_operation_combo_prioritizes_common_topk_over_empty_program():
    topk = classify_operation_combo([{"op": "sort"}, {"op": "limit"}])
    offset_topk = classify_operation_combo([{"op": "sort"}, {"op": "offset"}])
    empty = classify_operation_combo([])

    assert topk["frequency_bucket"] == "high"
    assert topk["priority"] > empty["priority"]
    assert "topk_ordering" in topk["correctness_risks"]
    assert offset_topk["has_sort_limit"] is True
    assert "topk_ordering" in offset_topk["correctness_risks"]
    assert empty["template"] == "empty"


def test_summarize_operation_combo_matches_compatibility_alias():
    operations = [{"op": "sort"}, {"op": "limit"}]

    assert summarize_operation_combo(operations) == classify_operation_combo(operations)
    assert describe_operation_combo(operations) == classify_operation_combo(operations)


def test_classify_operation_combo_tracks_post_topk_filter_pushdown_risk():
    combo = classify_operation_combo([{"op": "sort"}, {"op": "limit"}, {"op": "filter"}])

    assert combo["has_sort_limit"] is True
    assert "topk_ordering" in combo["correctness_risks"]
    assert "topk_filter_pushdown" in combo["correctness_risks"]


def test_classify_operation_combo_tracks_tuple_absence_null_filter_risk():
    combo = classify_operation_combo([{"op": "tuple_absence_filter"}, {"op": "select"}])

    assert combo["template"] == "tuple_absence_filter_select"
    assert "tuple_absence_null_filter" in combo["correctness_risks"]


def test_classify_operation_combo_tracks_distinct_duplicate_elimination_risk():
    combo = classify_operation_combo([{"op": "filter"}, {"op": "distinct"}, {"op": "sort"}, {"op": "limit"}])

    assert combo["template"] == "filter_distinct_sort_limit"
    assert combo["frequency_bucket"] == "high"
    assert combo["has_distinct"] is True
    assert "distinct_duplicate_elimination" in combo["correctness_risks"]
    assert "distinct_topk_ordering" in combo["correctness_risks"]


def test_classify_operation_combo_tracks_distinct_null_topk_ordering_risk():
    combo = classify_operation_combo(
        [
            {"op": "distinct", "columns": ["s"]},
            {"op": "sort", "keys": [{"column": "s", "ascending": True, "nulls": "first"}]},
            {"op": "limit", "n": 1},
        ]
    )

    assert combo["template"] == "distinct_sort_limit"
    assert "distinct_topk_ordering" in combo["correctness_risks"]
    assert "distinct_null_topk_ordering" in combo["correctness_risks"]


def test_classify_operation_combo_tracks_string_contains_filter_risk():
    combo = classify_operation_combo(
        [
            {"op": "filter", "cmp": "str_contains"},
            {"op": "groupby"},
            {"op": "sort"},
            {"op": "limit"},
        ]
    )

    assert combo["template"] == "filter_groupby_sort_limit"
    assert combo["frequency_bucket"] == "high"
    assert "string_pattern_filter" in combo["correctness_risks"]
    assert "string_pattern_aggregation_cardinality" in combo["correctness_risks"]
    assert "string_pattern_topk_cardinality" in combo["correctness_risks"]


def test_classify_operation_combo_tracks_string_strip_risk():
    combo = classify_operation_combo(
        [
            {"op": "mutate", "expr": {"kind": "string_strip"}},
            {"op": "filter"},
            {"op": "groupby"},
            {"op": "sort"},
            {"op": "limit"},
        ]
    )

    assert combo["template"] == "filter_mutate_groupby_sort_limit"
    assert "string_strip_whitespace" in combo["correctness_risks"]
    assert "string_strip_aggregation_keys" in combo["correctness_risks"]
    assert "string_strip_topk_keys" in combo["correctness_risks"]


def test_classify_operation_combo_tracks_string_case_risk():
    combo = classify_operation_combo(
        [
            {"op": "mutate", "expr": {"kind": "string_upper"}},
            {"op": "groupby"},
            {"op": "sort"},
            {"op": "limit"},
        ]
    )

    assert "string_case_normalization" in combo["correctness_risks"]
    assert "string_case_aggregation_keys" in combo["correctness_risks"]
    assert "string_case_topk_keys" in combo["correctness_risks"]


def test_classify_operation_combo_tracks_string_replace_risk():
    combo = classify_operation_combo(
        [
            {"op": "mutate", "expr": {"kind": "string_replace"}},
            {"op": "filter"},
            {"op": "groupby"},
            {"op": "sort"},
            {"op": "limit"},
        ]
    )

    assert combo["template"] == "filter_mutate_groupby_sort_limit"
    assert "string_replace_literal" in combo["correctness_risks"]
    assert "string_replace_aggregation_keys" in combo["correctness_risks"]
    assert "string_replace_topk_keys" in combo["correctness_risks"]


def test_classify_operation_combo_tracks_string_slice_risk():
    combo = classify_operation_combo(
        [
            {"op": "mutate", "expr": {"kind": "string_slice"}},
            {"op": "filter"},
            {"op": "groupby"},
            {"op": "sort"},
            {"op": "limit"},
        ]
    )

    assert combo["template"] == "filter_mutate_groupby_sort_limit"
    assert "string_slice_prefix" in combo["correctness_risks"]
    assert "string_slice_aggregation_keys" in combo["correctness_risks"]
    assert "string_slice_topk_keys" in combo["correctness_risks"]


def test_classify_operation_combo_tracks_string_length_risk():
    combo = classify_operation_combo(
        [
            {"op": "mutate", "expr": {"kind": "string_length"}},
            {"op": "filter"},
            {"op": "groupby"},
            {"op": "sort"},
            {"op": "limit"},
        ]
    )

    assert combo["template"] == "filter_mutate_groupby_sort_limit"
    assert "string_length" in combo["correctness_risks"]
    assert "string_length_null_boundary" in combo["correctness_risks"]
    assert "string_length_filter" in combo["correctness_risks"]
    assert "string_length_aggregation_keys" in combo["correctness_risks"]
    assert "string_length_topk_keys" in combo["correctness_risks"]


def test_classify_operation_combo_tracks_string_null_if_empty_risk():
    combo = classify_operation_combo(
        [
            {"op": "mutate", "expr": {"kind": "string_null_if_empty"}},
            {"op": "filter"},
            {"op": "groupby"},
            {"op": "sort"},
            {"op": "limit"},
        ]
    )

    assert combo["template"] == "filter_mutate_groupby_sort_limit"
    assert "string_null_if_empty" in combo["correctness_risks"]
    assert "empty_string_null_boundary" in combo["correctness_risks"]
    assert "string_null_if_empty_filter" in combo["correctness_risks"]
    assert "string_null_if_empty_aggregation_keys" in combo["correctness_risks"]
    assert "string_null_if_empty_topk_keys" in combo["correctness_risks"]


def test_classify_operation_combo_tracks_string_split_part_risk():
    combo = classify_operation_combo(
        [
            {"op": "mutate", "expr": {"kind": "string_split_part", "sep": " ", "index": 0}},
            {"op": "filter"},
            {"op": "groupby"},
            {"op": "sort"},
            {"op": "limit"},
        ]
    )

    assert combo["template"] == "filter_mutate_groupby_sort_limit"
    assert "string_split_part_first_token" in combo["correctness_risks"]
    assert "string_split_part_aggregation_keys" in combo["correctness_risks"]
    assert "string_split_part_topk_keys" in combo["correctness_risks"]


def test_classify_operation_combo_tracks_date_part_risk():
    combo = classify_operation_combo(
        [
            {"op": "mutate", "expr": {"kind": "date_part", "part": "year"}},
            {"op": "filter"},
            {"op": "groupby"},
            {"op": "sort"},
            {"op": "limit"},
        ]
    )

    assert combo["template"] == "filter_mutate_groupby_sort_limit"
    assert "date_part_extraction" in combo["correctness_risks"]
    assert "date_part_filter" in combo["correctness_risks"]
    assert "date_part_aggregation_keys" in combo["correctness_risks"]
    assert "date_part_topk_keys" in combo["correctness_risks"]


def test_classify_operation_combo_tracks_type_cast_boundary_risk():
    combo = classify_operation_combo(
        [
            {"op": "mutate", "expr": {"kind": "cast", "to": "int", "input_domain": "integer_string"}},
            {"op": "mutate", "expr": {"kind": "cast", "to": "str"}},
            {"op": "filter"},
            {"op": "groupby"},
            {"op": "sort"},
            {"op": "limit"},
        ]
    )

    assert combo["template"] == "filter_mutate_groupby_sort_limit"
    assert "type_cast" in combo["correctness_risks"]
    assert "type_cast_to_int" in combo["correctness_risks"]
    assert "type_cast_to_str" in combo["correctness_risks"]
    assert "type_cast_numeric_string" in combo["correctness_risks"]
    assert "type_cast_aggregation_keys" in combo["correctness_risks"]
    assert "type_cast_topk_keys" in combo["correctness_risks"]


def test_classify_operation_combo_tracks_string_concat_risk():
    combo = classify_operation_combo(
        [
            {"op": "mutate", "expr": {"kind": "string_concat"}},
            {"op": "filter"},
            {"op": "groupby"},
            {"op": "sort"},
            {"op": "limit"},
        ]
    )

    assert combo["template"] == "filter_mutate_groupby_sort_limit"
    assert "string_concat_null_propagation" in combo["correctness_risks"]
    assert "string_concat_aggregation_keys" in combo["correctness_risks"]
    assert "string_concat_topk_keys" in combo["correctness_risks"]


def test_classify_operation_combo_tracks_string_contains_expr_risk():
    combo = classify_operation_combo(
        [
            {"op": "mutate", "expr": {"kind": "string_contains"}},
            {"op": "filter"},
            {"op": "groupby"},
            {"op": "sort"},
            {"op": "limit"},
        ]
    )

    assert combo["template"] == "filter_mutate_groupby_sort_limit"
    assert "string_pattern_nullable_bool" in combo["correctness_risks"]
    assert "string_pattern_bool_filter" in combo["correctness_risks"]
    assert "string_pattern_aggregation_keys" in combo["correctness_risks"]
    assert "string_pattern_topk_keys" in combo["correctness_risks"]
    assert "string_contains_nullable_bool" in combo["correctness_risks"]
    assert "string_contains_bool_filter" in combo["correctness_risks"]
    assert "string_contains_aggregation_keys" in combo["correctness_risks"]
    assert "string_contains_topk_keys" in combo["correctness_risks"]


def test_classify_operation_combo_tracks_starts_and_ends_with_expr_risk():
    for kind in ("string_starts_with", "string_ends_with"):
        combo = classify_operation_combo(
            [
                {"op": "mutate", "expr": {"kind": kind}},
                {"op": "filter"},
                {"op": "groupby"},
                {"op": "sort"},
                {"op": "limit"},
            ]
        )

        assert "string_pattern_nullable_bool" in combo["correctness_risks"]
        assert "string_pattern_bool_filter" in combo["correctness_risks"]
        assert "string_pattern_aggregation_keys" in combo["correctness_risks"]
        assert "string_pattern_topk_keys" in combo["correctness_risks"]


def test_classify_operation_combo_tracks_nullable_boolean_negation_risk():
    combo = classify_operation_combo(
        [
            {"op": "mutate", "expr": {"kind": "bool_not"}},
            {"op": "filter"},
            {"op": "groupby"},
            {"op": "sort"},
            {"op": "limit"},
        ]
    )

    assert combo["template"] == "filter_mutate_groupby_sort_limit"
    assert "nullable_boolean_negation" in combo["correctness_risks"]
    assert "nullable_boolean_filter" in combo["correctness_risks"]
    assert "nullable_boolean_aggregation_keys" in combo["correctness_risks"]
    assert "nullable_boolean_topk_keys" in combo["correctness_risks"]


def test_classify_operation_combo_tracks_nullable_boolean_reduction_risk():
    combo = classify_operation_combo(
        [
            {
                "op": "groupby",
                "keys": ["g"],
                "aggs": [
                    {"column": "flag", "func": "any", "as": "any_flag"},
                    {"column": "flag", "func": "all", "as": "all_flag"},
                ],
            },
            {"op": "filter", "column": "any_flag", "cmp": "bool_is_not_false", "value": None},
            {"op": "sort"},
            {"op": "limit"},
        ]
    )

    assert combo["template"] == "filter_groupby_sort_limit"
    assert "nullable_boolean_reduction" in combo["correctness_risks"]
    assert "nullable_boolean_reduction_filter" in combo["correctness_risks"]
    assert "nullable_boolean_reduction_aggregation" in combo["correctness_risks"]
    assert "nullable_boolean_reduction_topk" in combo["correctness_risks"]


def test_classify_operation_combo_tracks_numeric_clip_risk():
    combo = classify_operation_combo(
        [
            {"op": "mutate", "expr": {"kind": "clip"}},
            {"op": "filter"},
            {"op": "groupby"},
            {"op": "sort"},
            {"op": "limit"},
        ]
    )

    assert combo["template"] == "filter_mutate_groupby_sort_limit"
    assert "numeric_clip_bounds" in combo["correctness_risks"]
    assert "numeric_clip_aggregation_keys" in combo["correctness_risks"]
    assert "numeric_clip_topk_keys" in combo["correctness_risks"]


def test_classify_operation_combo_tracks_numeric_abs_risk():
    combo = classify_operation_combo(
        [
            {"op": "mutate", "expr": {"kind": "abs"}},
            {"op": "filter"},
            {"op": "groupby"},
            {"op": "sort"},
            {"op": "limit"},
        ]
    )

    assert combo["template"] == "filter_mutate_groupby_sort_limit"
    assert "numeric_abs_sign" in combo["correctness_risks"]
    assert "numeric_abs_aggregation_keys" in combo["correctness_risks"]
    assert "numeric_abs_topk_keys" in combo["correctness_risks"]


def test_classify_operation_combo_tracks_fill_null_semantic_risk():
    combo = classify_operation_combo([{"op": "fill_null"}, {"op": "groupby"}, {"op": "sort"}])

    assert combo["template"] == "fill_null_groupby_sort"
    assert combo["frequency_bucket"] == "high"
    assert combo["has_fill_null"] is True
    assert "fill_null_null_semantics" in combo["correctness_risks"]
    assert "fill_null_groupby_keys" in combo["correctness_risks"]
    assert "fill_null_topk_keys" in combo["correctness_risks"]


def test_classify_operation_combo_tracks_coalesce_semantic_risk():
    combo = classify_operation_combo([{"op": "coalesce"}, {"op": "groupby"}, {"op": "sort"}])

    assert combo["template"] == "coalesce_groupby_sort"
    assert combo["frequency_bucket"] == "high"
    assert combo["has_coalesce"] is True
    assert "coalesce_null_semantics" in combo["correctness_risks"]
    assert "coalesce_column_precedence" in combo["correctness_risks"]
    assert "coalesce_aggregation_keys" in combo["correctness_risks"]


def test_classify_operation_combo_tracks_fill_null_coalesce_groupby_topk_risk():
    combo = classify_operation_combo(
        [{"op": "coalesce"}, {"op": "fill_null"}, {"op": "groupby"}, {"op": "sort"}, {"op": "limit"}]
    )

    assert combo["template"] == "fill_null_coalesce_groupby_sort_limit"
    assert "fill_null_null_semantics" in combo["correctness_risks"]
    assert "fill_null_groupby_keys" in combo["correctness_risks"]
    assert "fill_null_topk_keys" in combo["correctness_risks"]
    assert "coalesce_null_semantics" in combo["correctness_risks"]
    assert "coalesce_aggregation_keys" in combo["correctness_risks"]
    assert "coalesce_topk_keys" in combo["correctness_risks"]


def test_classify_operation_combo_tracks_case_when_semantic_risk():
    combo = classify_operation_combo([{"op": "case_when"}, {"op": "groupby"}, {"op": "sort"}])

    assert combo["template"] == "case_when_groupby_sort"
    assert combo["frequency_bucket"] == "high"
    assert combo["has_case_when"] is True
    assert "conditional_expression" in combo["correctness_risks"]
    assert "case_when_groupby_keys" in combo["correctness_risks"]


def test_classify_operation_combo_tracks_union_all_row_append_risk():
    combo = classify_operation_combo([{"op": "union_all"}, {"op": "filter"}, {"op": "groupby"}, {"op": "sort"}])

    assert combo["template"] == "union_all_filter_groupby_sort"
    assert combo["frequency_bucket"] == "high"
    assert combo["has_union_all"] is True
    assert "union_all_row_append" in combo["correctness_risks"]
    assert "union_all_filter_pushdown" in combo["correctness_risks"]
    assert "union_all_aggregation_cardinality" in combo["correctness_risks"]


def test_classify_operation_combo_tracks_drop_nulls_semantic_risk():
    combo = classify_operation_combo([{"op": "drop_nulls"}, {"op": "groupby"}, {"op": "sort"}])

    assert combo["template"] == "drop_nulls_groupby_sort"
    assert combo["frequency_bucket"] == "high"
    assert combo["has_drop_nulls"] is True
    assert "drop_nulls_null_filter" in combo["correctness_risks"]
    assert "drop_nulls_aggregation_cardinality" in combo["correctness_risks"]


def test_classify_operation_combo_tracks_semi_and_anti_join_membership_risks():
    semi = classify_operation_combo([{"op": "semi_join"}, {"op": "filter"}, {"op": "select"}, {"op": "sort"}, {"op": "limit"}])
    anti = classify_operation_combo([{"op": "anti_join"}, {"op": "fill_null"}, {"op": "groupby"}, {"op": "sort"}])

    assert semi["template"] == "semi_join_filter_select_sort_limit"
    assert semi["frequency_bucket"] == "high"
    assert semi["has_semi_join"] is True
    assert "semi_join_membership" in semi["correctness_risks"]
    assert "semi_anti_join_filter_pushdown" in semi["correctness_risks"]
    assert "semi_anti_join_topk_cardinality" in semi["correctness_risks"]
    assert anti["template"] == "anti_join_fill_null_groupby_sort"
    assert anti["frequency_bucket"] == "high"
    assert anti["has_anti_join"] is True
    assert "anti_join_exclusion" in anti["correctness_risks"]
    assert "semi_anti_join_aggregation_cardinality" in anti["correctness_risks"]


def test_classify_operation_combo_tracks_normalized_string_join_risks():
    combo = classify_operation_combo(
        [
            {"op": "mutate", "expr": {"kind": "string_strip", "source": "s"}},
            {"op": "mutate", "expr": {"kind": "string_lower", "source": "s_clean"}},
            {"op": "join", "left_on": "s_key", "right_on": "s_key"},
            {"op": "fill_null"},
            {"op": "groupby"},
            {"op": "sort"},
            {"op": "limit"},
        ]
    )

    assert combo["template"] == "join_fill_null_mutate_groupby_sort_limit"
    assert "cleaned_join_key" in combo["correctness_risks"]
    assert "normalized_string_join_key" in combo["correctness_risks"]
    assert "chained_string_normalized_join_key" in combo["correctness_risks"]
    assert "normalized_string_join_aggregation" in combo["correctness_risks"]
    assert "normalized_string_join_topk" in combo["correctness_risks"]


def test_classify_operation_combo_tracks_normalized_string_membership_risks():
    combo = classify_operation_combo(
        [
            {"op": "mutate", "expr": {"kind": "string_strip", "source": "s"}},
            {"op": "mutate", "expr": {"kind": "string_lower", "source": "s_clean"}},
            {"op": "filter"},
            {"op": "semi_join", "left_on": "s_key", "right_on": "s_key"},
            {"op": "sort"},
            {"op": "limit"},
        ]
    )

    assert "semi_join_membership" in combo["correctness_risks"]
    assert "normalized_string_membership_key" in combo["correctness_risks"]
    assert "chained_string_normalized_membership_key" in combo["correctness_risks"]
    assert "normalized_string_membership_topk" in combo["correctness_risks"]


def test_classify_operation_combo_tracks_sql_distinct_null_topk_risks():
    combo = classify_operation_combo(
        [
            {"op": "mutate", "expr": {"kind": "string_null_if_empty", "source": "s"}},
            {"op": "coalesce", "columns": ["s_norm", "g"], "as": "label", "fallback": "missing"},
            {"op": "distinct", "columns": ["label", "flag"]},
            {"op": "sort", "keys": [{"column": "label", "ascending": True, "nulls": "first"}]},
            {"op": "offset", "n": 1},
            {"op": "limit", "n": 2},
        ]
    )

    assert combo["template"] == "coalesce_mutate_distinct_sort_offset_limit"
    assert combo["frequency_bucket"] == "high"
    assert "distinct_null_topk_ordering" in combo["correctness_risks"]
    assert "sql_distinct_null_topk" in combo["correctness_risks"]
    assert "coalesced_distinct_topk" in combo["correctness_risks"]


def test_classify_operation_combo_tracks_left_join_coalesce_membership_risks():
    combo = classify_operation_combo(
        [
            {"op": "join", "how": "left"},
            {"op": "coalesce", "columns": ["tag", "g"], "as": "segment_key", "fallback": "missing"},
            {"op": "fill_null", "column": "j", "value": 0},
            {"op": "anti_join", "left_on": "segment_key", "right_on": "segment_key"},
            {"op": "sort"},
            {"op": "limit"},
        ]
    )

    assert combo["template"] == "join_anti_join_fill_null_coalesce_sort_limit"
    assert "join_coalesce_membership" in combo["correctness_risks"]
    assert "left_join_coalesce_membership" in combo["correctness_risks"]
    assert "join_coalesce_membership_topk" in combo["correctness_risks"]


def test_classify_operation_combo_tracks_case_when_membership_risks():
    combo = classify_operation_combo(
        [
            {"op": "mutate", "expr": {"kind": "string_strip", "source": "g"}},
            {
                "op": "case_when",
                "condition": {"column": "g_clean", "cmp": "in_set", "value": ["a"]},
                "as": "membership_bucket",
            },
            {"op": "distinct", "columns": ["membership_bucket", "flag"]},
            {"op": "sort"},
            {"op": "limit"},
        ]
    )

    assert combo["template"] == "case_when_mutate_distinct_sort_limit"
    assert combo["frequency_bucket"] == "high"
    assert "case_when_membership_predicate" in combo["correctness_risks"]
    assert "case_when_membership_topk" in combo["correctness_risks"]


def test_classify_operation_combo_tracks_normalized_string_case_when_risks():
    combo = classify_operation_combo(
        [
            {"op": "mutate", "expr": {"kind": "string_strip", "source": "s"}},
            {"op": "mutate", "expr": {"kind": "string_lower", "source": "s_clean"}},
            {
                "op": "case_when",
                "condition": {"column": "s_key", "cmp": "in_set", "value": ["alpha"]},
                "then": "interesting",
                "else": "other",
            },
            {"op": "groupby"},
            {"op": "sort"},
            {"op": "limit"},
        ]
    )

    assert combo["template"] == "case_when_mutate_groupby_sort_limit"
    assert "case_when_membership_predicate" in combo["correctness_risks"]
    assert "case_when_membership_aggregation" in combo["correctness_risks"]
    assert "case_when_membership_topk" in combo["correctness_risks"]
    assert "normalized_string_case_when" in combo["correctness_risks"]
    assert "normalized_string_case_when_aggregation" in combo["correctness_risks"]
    assert "normalized_string_case_when_topk" in combo["correctness_risks"]


def test_classify_operation_combo_tracks_string_pattern_case_when_risks():
    combo = classify_operation_combo(
        [
            {"op": "mutate", "expr": {"kind": "string_starts_with", "source": "s", "needle": "space"}},
            {
                "op": "case_when",
                "condition": {"column": "starts_space", "cmp": "bool_is_true", "value": None},
                "then": "starts-space",
                "else": "other",
            },
            {"op": "groupby"},
            {"op": "sort"},
            {"op": "limit"},
        ]
    )

    assert combo["template"] == "case_when_mutate_groupby_sort_limit"
    assert "string_pattern_nullable_bool" in combo["correctness_risks"]
    assert "boolean_case_when_predicate" in combo["correctness_risks"]
    assert "string_pattern_case_when" in combo["correctness_risks"]
    assert "string_pattern_case_when_aggregation" in combo["correctness_risks"]
    assert "string_pattern_case_when_topk" in combo["correctness_risks"]


def test_classify_operation_combo_tracks_union_coalesce_distinct_topk_risk():
    combo = classify_operation_combo(
        [
            {"op": "union_all"},
            {"op": "mutate", "expr": {"kind": "string_null_if_empty", "source": "s"}},
            {"op": "coalesce", "columns": ["s_norm", "g"], "as": "label", "fallback": "missing"},
            {"op": "distinct", "columns": ["label", "flag"]},
            {"op": "sort", "keys": [{"column": "label", "ascending": True, "nulls": "first"}]},
            {"op": "offset", "n": 1},
            {"op": "limit", "n": 2},
        ]
    )

    assert combo["template"] == "union_all_coalesce_mutate_distinct_sort_offset_limit"
    assert combo["frequency_bucket"] == "high"
    assert "union_coalesce_distinct_topk" in combo["correctness_risks"]


def test_classify_operation_combo_tracks_left_join_case_membership_risk():
    combo = classify_operation_combo(
        [
            {"op": "join", "how": "left"},
            {"op": "case_when", "condition": {"column": "tag", "cmp": "in_set", "value": ["dim-a"]}},
            {"op": "fill_null", "column": "j", "value": 0},
            {"op": "groupby"},
            {"op": "sort"},
        ]
    )

    assert "join_case_when_membership" in combo["correctness_risks"]
    assert "left_join_case_when_membership" in combo["correctness_risks"]
    assert "join_case_when_membership_aggregation" in combo["correctness_risks"]


def test_classify_operation_combo_tracks_left_join_null_predicate_aggregation_risk():
    combo = classify_operation_combo(
        [
            {"op": "join", "how": "left"},
            {"op": "filter", "column": "tag", "cmp": "is_null", "value": None},
            {"op": "aggregate", "aggs": [{"column": "id", "func": "count", "as": "count_id"}]},
        ]
    )

    assert combo["template"] == "join_filter_aggregate"
    assert combo["frequency_bucket"] == "high"
    assert "left_join_null_predicate_filter" in combo["correctness_risks"]
    assert "left_join_null_predicate_aggregation" in combo["correctness_risks"]


def test_classify_operation_combo_tracks_coalesce_case_distinct_aggregation_risk():
    combo = classify_operation_combo(
        [
            {"op": "mutate", "expr": {"kind": "string_null_if_empty", "source": "s"}},
            {"op": "coalesce", "columns": ["s_norm", "g"], "as": "label"},
            {"op": "case_when", "condition": {"column": "label", "cmp": "in_set", "value": ["a"]}},
            {"op": "distinct", "columns": ["label", "bucket"]},
            {"op": "groupby"},
            {"op": "sort"},
        ]
    )

    assert combo["template"] == "coalesce_case_when_mutate_groupby_distinct_sort"
    assert combo["frequency_bucket"] == "high"
    assert "coalesce_case_distinct_aggregation" in combo["correctness_risks"]


def test_classify_operation_combo_tracks_numeric_text_cast_membership_aggregation_risk():
    combo = classify_operation_combo(
        [
            {
                "op": "mutate",
                "expr": {"kind": "cast", "source": "num_s", "to": "int", "input_domain": "integer_string"},
            },
            {"op": "filter", "column": "num_value", "cmp": "is_not_null", "value": None},
            {"op": "semi_join", "left_on": "num_value", "right_on": "num_value"},
            {"op": "groupby"},
            {"op": "sort"},
        ]
    )

    assert combo["template"] == "semi_join_filter_mutate_groupby_sort"
    assert combo["frequency_bucket"] == "high"
    assert "type_cast_membership_aggregation" in combo["correctness_risks"]
    assert "numeric_text_cast_membership_aggregation" in combo["correctness_risks"]


def test_classify_operation_combo_tracks_boolean_membership_case_aggregation_risk():
    combo = classify_operation_combo(
        [
            {"op": "filter", "column": "flag", "cmp": "bool_is_not_unknown", "value": None},
            {"op": "semi_join", "left_on": "flag", "right_on": "flag_key"},
            {
                "op": "case_when",
                "condition": {"column": "flag", "cmp": "bool_is_true", "value": None},
                "then": "true_member",
                "else": "false_member",
            },
            {"op": "groupby"},
            {"op": "sort"},
        ]
    )

    assert combo["template"] == "semi_join_filter_case_when_groupby_sort"
    assert combo["frequency_bucket"] == "high"
    assert "boolean_case_when_predicate" in combo["correctness_risks"]
    assert "boolean_membership_case_aggregation" in combo["correctness_risks"]


def test_classify_operation_combo_tracks_boolean_antijoin_case_aggregation_risk():
    combo = classify_operation_combo(
        [
            {"op": "filter", "column": "flag", "cmp": "bool_is_not_unknown", "value": None},
            {"op": "anti_join", "left_on": "flag", "right_on": "flag_key"},
            {
                "op": "case_when",
                "condition": {"column": "flag", "cmp": "bool_is_false", "value": None},
                "then": "false_non_member",
                "else": "other_non_member",
            },
            {"op": "groupby"},
            {"op": "sort"},
        ]
    )

    assert combo["template"] == "anti_join_filter_case_when_groupby_sort"
    assert combo["frequency_bucket"] == "high"
    assert "boolean_case_when_predicate" in combo["correctness_risks"]
    assert "boolean_membership_case_aggregation" in combo["correctness_risks"]


def test_classify_operation_combo_tracks_left_join_boolean_case_aggregation_risk():
    combo = classify_operation_combo(
        [
            {"op": "join", "how": "left", "left_on": "id", "right_on": "id"},
            {
                "op": "case_when",
                "condition": {"column": "tag", "cmp": "is_null", "value": None},
                "then": True,
                "else": False,
            },
            {"op": "groupby"},
            {"op": "sort"},
        ]
    )

    assert combo["template"] == "join_case_when_groupby_sort"
    assert combo["frequency_bucket"] == "high"
    assert "boolean_case_when" in combo["correctness_risks"]
    assert "boolean_case_when_aggregation" in combo["correctness_risks"]
    assert "left_join_boolean_case_aggregation" in combo["correctness_risks"]


def test_classify_operation_combo_tracks_left_join_boolean_coalesce_case_aggregation_risk():
    combo = classify_operation_combo(
        [
            {"op": "join", "how": "left", "left_on": "id", "right_on": "id"},
            {"op": "coalesce", "columns": ["dim_flag", "flag"], "as": "flag_effective", "fallback": False},
            {
                "op": "case_when",
                "condition": {"column": "flag_effective", "cmp": "bool_is_true", "value": None},
                "then": "effective_true",
                "else": "effective_false_or_missing",
            },
            {"op": "groupby"},
            {"op": "sort"},
        ]
    )

    assert combo["template"] == "join_coalesce_case_when_groupby_sort"
    assert combo["frequency_bucket"] == "high"
    assert "boolean_case_when_predicate" in combo["correctness_risks"]
    assert "boolean_coalesce_case_aggregation" in combo["correctness_risks"]
    assert "left_join_boolean_coalesce_aggregation" in combo["correctness_risks"]


def test_classify_operation_combo_tracks_left_join_boolean_coalesce_filter_aggregation_risk():
    combo = classify_operation_combo(
        [
            {"op": "join", "how": "left", "left_on": "id", "right_on": "id"},
            {"op": "coalesce", "columns": ["dim_flag", "flag"], "as": "flag_effective", "fallback": False},
            {"op": "filter", "column": "flag_effective", "cmp": "==", "value": False},
            {
                "op": "case_when",
                "condition": {"column": "flag_effective", "cmp": "bool_is_false", "value": None},
                "then": "effective_false",
                "else": "effective_true_or_missing",
            },
            {"op": "groupby"},
            {"op": "sort"},
        ]
    )

    assert combo["template"] == "join_filter_coalesce_case_when_groupby_sort"
    assert combo["frequency_bucket"] == "high"
    assert "boolean_coalesce_filter_aggregation" in combo["correctness_risks"]
    assert "left_join_boolean_coalesce_filter_aggregation" in combo["correctness_risks"]


def test_classify_operation_combo_tracks_numeric_text_bool_antijoin_aggregation_risk():
    combo = classify_operation_combo(
        [
            {
                "op": "mutate",
                "column": "num_value",
                "expr": {"kind": "cast", "source": "num_s", "to": "int", "input_domain": "integer_string"},
            },
            {"op": "filter", "column": "num_value", "cmp": "is_not_null", "value": None},
            {"op": "filter", "column": "flag", "cmp": "bool_is_not_false", "value": None},
            {"op": "anti_join", "left_on": "num_value", "right_on": "num_value"},
            {
                "op": "case_when",
                "condition": {"column": "flag", "cmp": "bool_is_true", "value": None},
                "then": "true_unmatched_number",
                "else": "null_or_false_unmatched_number",
            },
            {"op": "groupby"},
            {"op": "sort"},
        ]
    )

    assert combo["template"] == "anti_join_filter_case_when_mutate_groupby_sort"
    assert combo["frequency_bucket"] == "high"
    assert "numeric_text_cast_membership_aggregation" in combo["correctness_risks"]
    assert "boolean_membership_case_aggregation" in combo["correctness_risks"]


def test_classify_operation_combo_tracks_multi_key_membership_aggregation_risk():
    combo = classify_operation_combo(
        [
            {"op": "mutate", "column": "g_clean", "expr": {"kind": "string_strip", "source": "g"}},
            {"op": "filter", "column": "g_clean", "cmp": "is_not_null", "value": None},
            {
                "op": "semi_join",
                "left_on": ["id", "g_clean"],
                "right_on": ["id", "g_clean"],
            },
            {
                "op": "case_when",
                "condition": {"column": "flag", "cmp": "bool_is_true", "value": None},
                "then": "matched_true",
                "else": "matched_false_or_null",
            },
            {"op": "groupby"},
            {"op": "sort"},
        ]
    )

    assert combo["template"] == "semi_join_filter_case_when_mutate_groupby_sort"
    assert combo["frequency_bucket"] == "high"
    assert "multi_key_semi_anti_join" in combo["correctness_risks"]
    assert "multi_key_membership_aggregation" in combo["correctness_risks"]
    assert "boolean_membership_case_aggregation" in combo["correctness_risks"]


def test_classify_operation_combo_tracks_running_sum_precision_risk():
    combo = classify_operation_combo([{"op": "running_sum"}, {"op": "sort"}, {"op": "limit"}])

    assert combo["template"] == "running_sum_sort_limit"
    assert "running_sum_precision" in combo["correctness_risks"]
    assert combo["has_sort_limit"] is True


def test_classify_operation_combo_tracks_partitioned_running_sum_risk():
    combo = classify_operation_combo([{"op": "running_sum", "partition_by": ["g"]}, {"op": "sort"}])

    assert combo["template"] == "running_sum_sort"
    assert "partitioned_running_sum" in combo["correctness_risks"]


def test_classify_operation_combo_tracks_path_projection_keyed_pick_risk():
    combo = classify_operation_combo(
        [
            {"op": "mutate", "expr": {"kind": "string_basename", "source": "path_value"}},
            {"op": "row_number_filter"},
            {"op": "select"},
        ]
    )

    assert combo["template"] == "mutate_row_number_filter_select"
    assert "keyed_row_pick" in combo["correctness_risks"]
    assert "path_projection_keyed_pick" in combo["correctness_risks"]


def test_classify_operation_combo_tracks_string_basename_groupby_and_topk_risks():
    groupby_combo = classify_operation_combo(
        [
            {"op": "mutate", "expr": {"kind": "string_basename", "source": "path_value"}},
            {"op": "fill_null"},
            {"op": "groupby"},
            {"op": "sort"},
        ]
    )
    topk_combo = classify_operation_combo(
        [
            {"op": "mutate", "expr": {"kind": "string_basename", "source": "path_value"}},
            {"op": "sort"},
            {"op": "limit"},
        ]
    )

    assert "path_projection" in groupby_combo["correctness_risks"]
    assert "string_basename_path_separator" in groupby_combo["correctness_risks"]
    assert "string_basename_aggregation_keys" in groupby_combo["correctness_risks"]
    assert "string_basename_topk_keys" in topk_combo["correctness_risks"]


def test_classify_operation_combo_tracks_sortedness_null_placement_risk():
    combo = classify_operation_combo([{"op": "sort"}, {"op": "sortedness_check"}])

    assert combo["template"] == "sort_sortedness_check"
    assert "sortedness_null_placement" in combo["correctness_risks"]


def test_classify_operation_combo_tracks_random_case_subject_risk():
    combo = classify_operation_combo([{"op": "random_case_probe"}])

    assert combo["template"] == "random_case_probe"
    assert "simple_case_random_subject" in combo["correctness_risks"]


def test_classify_operation_combo_tracks_group_quantile_key_expression_risk():
    combo = classify_operation_combo([{"op": "group_quantile_probe"}])

    assert combo["template"] == "group_quantile_probe"
    assert "group_quantile_key_expression" in combo["correctness_risks"]


def test_classify_operation_combo_tracks_scalar_subquery_double_parentheses_risk():
    combo = classify_operation_combo([{"op": "scalar_subquery_probe"}])

    assert combo["template"] == "scalar_subquery_probe"
    assert "scalar_subquery_double_parentheses" in combo["correctness_risks"]


def test_classify_operation_combo_tracks_window_avg_rows_frame_risk():
    combo = classify_operation_combo([{"op": "window_avg_probe"}])

    assert combo["template"] == "window_avg_probe"
    assert "window_avg_rows_frame" in combo["correctness_risks"]


def test_classify_operation_combo_tracks_struct_distinct_unnest_risk():
    combo = classify_operation_combo([{"op": "struct_distinct_probe"}])

    assert combo["template"] == "struct_distinct_probe"
    assert "struct_distinct_unnest" in combo["correctness_risks"]


def test_classify_operation_combo_tracks_bit_compare_unequal_length_risk():
    combo = classify_operation_combo([{"op": "bit_compare_probe"}])

    assert combo["template"] == "bit_compare_probe"
    assert "bit_compare_unequal_length" in combo["correctness_risks"]


def test_classify_operation_combo_tracks_round_even_float_scale_risk():
    combo = classify_operation_combo([{"op": "round_even_probe"}])

    assert combo["template"] == "round_even_probe"
    assert "round_even_float_scale" in combo["correctness_risks"]


def test_classify_operation_combo_tracks_duckdb_float_literal_precision_risk():
    combo = classify_operation_combo([{"op": "float_literal_precision_probe"}])

    assert combo["template"] == "float_literal_precision_probe"
    assert "duckdb_float_literal_precision" in combo["correctness_risks"]


def test_classify_operation_combo_tracks_polars_timestamp_precision_filter_risk():
    combo = classify_operation_combo([{"op": "timestamp_precision_filter_probe"}])

    assert combo["template"] == "timestamp_precision_filter_probe"
    assert "polars_timestamp_precision_filter" in combo["correctness_risks"]


def test_classify_operation_combo_tracks_series_rtruediv_operand_order_risk():
    combo = classify_operation_combo([{"op": "series_rtruediv_probe"}])

    assert combo["template"] == "series_rtruediv_probe"
    assert "series_rtruediv_operand_order" in combo["correctness_risks"]


def test_classify_operation_combo_tracks_datafusion_grouped_null_topk_risk():
    combo = classify_operation_combo(
        [{"op": "datafusion_grouped_null_topk_probe"}]
    )

    assert combo["template"] == "datafusion_grouped_null_topk_probe"
    assert "grouped_topk_null_sort_key" in combo["correctness_risks"]


def test_classify_operation_combo_tracks_pandas_uint64_isin_precision_risk():
    combo = classify_operation_combo([{"op": "uint64_isin_probe"}])

    assert combo["template"] == "uint64_isin_probe"
    assert "pandas_uint64_isin_precision" in combo["correctness_risks"]


def test_classify_operation_combo_tracks_duckdb_tuple_anti_null_semantics_risk():
    combo = classify_operation_combo([{"op": "tuple_anti_null_probe"}])

    assert combo["template"] == "tuple_anti_null_probe"
    assert "duckdb_tuple_anti_null_semantics" in combo["correctness_risks"]


def test_classify_operation_combo_tracks_datafusion_setop_all_duplicate_count_risk():
    combo = classify_operation_combo([{"op": "setop_all_duplicate_probe"}])

    assert combo["template"] == "setop_all_duplicate_probe"
    assert "datafusion_setop_all_duplicate_count" in combo["correctness_risks"]


def test_classify_operation_combo_tracks_duckdb_json_predicate_order_semantics_risk():
    combo = classify_operation_combo([{"op": "json_predicate_order_probe"}])

    assert combo["template"] == "json_predicate_order_probe"
    assert "duckdb_json_predicate_order_semantics" in combo["correctness_risks"]


def test_classify_operation_combo_tracks_pandas_sparse_array_mask_semantics_risk():
    combo = classify_operation_combo([{"op": "sparse_mask_probe"}])

    assert combo["template"] == "sparse_mask_probe"
    assert "pandas_sparse_array_mask_semantics" in combo["correctness_risks"]


def test_classify_operation_combo_tracks_polars_float_wrap_numerical_semantics_risk():
    combo = classify_operation_combo([{"op": "float_wrap_probe"}])

    assert combo["template"] == "float_wrap_probe"
    assert "polars_float_wrap_numerical_semantics" in combo["correctness_risks"]


def test_classify_operation_combo_tracks_pandas_index_bool_result_type_risk():
    combo = classify_operation_combo([{"op": "index_bool_probe"}])

    assert combo["template"] == "index_bool_probe"
    assert "pandas_index_bool_result_type" in combo["correctness_risks"]


def test_classify_operation_combo_tracks_polars_empty_literal_groupby_semantics_risk():
    combo = classify_operation_combo([{"op": "empty_literal_groupby_probe"}])

    assert combo["template"] == "empty_literal_groupby_probe"
    assert "polars_empty_literal_groupby_semantics" in combo["correctness_risks"]


def test_classify_operation_combo_tracks_pandas_arrow_string_eq_sum_semantics_risk():
    combo = classify_operation_combo([{"op": "arrow_string_eq_sum_probe"}])

    assert combo["template"] == "arrow_string_eq_sum_probe"
    assert "pandas_arrow_string_eq_sum_semantics" in combo["correctness_risks"]


def test_classify_operation_combo_tracks_pandas_arrow_timestamp_loc_slice_semantics_risk():
    combo = classify_operation_combo([{"op": "arrow_timestamp_loc_slice_probe"}])

    assert combo["template"] == "arrow_timestamp_loc_slice_probe"
    assert "pandas_arrow_timestamp_loc_slice_semantics" in combo["correctness_risks"]


def test_classify_operation_combo_tracks_pandas_arrow_timestamp_index_attr_semantics_risk():
    combo = classify_operation_combo([{"op": "arrow_timestamp_index_attr_probe"}])

    assert combo["template"] == "arrow_timestamp_index_attr_probe"
    assert "pandas_arrow_timestamp_index_attr_semantics" in combo["correctness_risks"]


def test_classify_operation_combo_tracks_pandas_eval_inplace_aliasing_risk():
    combo = classify_operation_combo([{"op": "eval_inplace_alias_probe"}])

    assert combo["template"] == "eval_inplace_alias_probe"
    assert "pandas_eval_inplace_aliasing_semantics" in combo["correctness_risks"]


def test_classify_operation_combo_tracks_pandas_bool_reduction_skipna_semantics_risk():
    combo = classify_operation_combo([{"op": "bool_reduction_skipna_probe"}])

    assert combo["template"] == "bool_reduction_skipna_probe"
    assert "pandas_bool_reduction_skipna_semantics" in combo["correctness_risks"]


def test_classify_operation_combo_tracks_pyarrow_dataset_isin_all_match_semantics_risk():
    combo = classify_operation_combo([{"op": "dataset_isin_all_match_probe"}])

    assert combo["template"] == "dataset_isin_all_match_probe"
    assert "pyarrow_dataset_isin_all_match_semantics" in combo["correctness_risks"]


def test_classify_operation_combo_tracks_pyarrow_run_end_null_compute_semantics_risk():
    combo = classify_operation_combo([{"op": "run_end_null_compute_probe"}])

    assert combo["template"] == "run_end_null_compute_probe"
    assert "pyarrow_run_end_null_compute_semantics" in combo["correctness_risks"]


def test_classify_operation_combo_tracks_pyarrow_large_string_partition_schema_risk():
    combo = classify_operation_combo([{"op": "large_string_partition_probe"}])

    assert combo["template"] == "large_string_partition_probe"
    assert "pyarrow_large_string_partition_schema_semantics" in combo["correctness_risks"]


def test_classify_operation_combo_tracks_pyarrow_hash_pivot_wider_order_risk():
    combo = classify_operation_combo([{"op": "hash_pivot_wider_probe"}])

    assert combo["template"] == "hash_pivot_wider_probe"
    assert "pyarrow_hash_pivot_wider_order_semantics" in combo["correctness_risks"]


def test_classify_operation_combo_tracks_pyarrow_list_flatten_parent_indices_risk():
    combo = classify_operation_combo([{"op": "list_flatten_parent_indices_probe"}])

    assert combo["template"] == "list_flatten_parent_indices_probe"
    assert "pyarrow_list_flatten_parent_indices_semantics" in combo["correctness_risks"]


def test_classify_operation_combo_tracks_polars_rolling_mean_by_null_count_semantics_risk():
    combo = classify_operation_combo([{"op": "rolling_mean_by_null_count_probe"}])

    assert combo["template"] == "rolling_mean_by_null_count_probe"
    assert "polars_rolling_mean_by_null_count_semantics" in combo["correctness_risks"]


def test_classify_operation_combo_tracks_csv_long_numeric_roundtrip_risk():
    combo = classify_operation_combo([{"op": "csv_long_numeric_roundtrip_probe"}])

    assert combo["template"] == "csv_long_numeric_roundtrip_probe"
    assert "csv_long_numeric_roundtrip" in combo["correctness_risks"]


def test_classify_operation_combo_tracks_global_aggregation_pipeline():
    combo = classify_operation_combo(
        [
            {"op": "join"},
            {"op": "groupby"},
            {"op": "join"},
            {"op": "groupby"},
            {"op": "aggregate"},
        ]
    )

    assert combo["template"] == "join_groupby_aggregate"
    assert combo["has_aggregate"] is True
    assert "global_aggregation" in combo["correctness_risks"]
    assert "join_groupby_pipeline" in combo["correctness_risks"]


def test_classify_operation_combo_tracks_distinct_count_aggregation_risk():
    combo = classify_operation_combo(
        [
            {"op": "join"},
            {
                "op": "groupby",
                "aggs": [{"column": "s", "func": "nunique", "as": "unique_s"}],
            },
        ]
    )
    global_combo = classify_operation_combo(
        [
            {"op": "filter"},
            {
                "op": "aggregate",
                "aggs": [{"column": "s", "func": "nunique", "as": "unique_s"}],
            },
        ]
    )

    assert "distinct_count_aggregation" in combo["correctness_risks"]
    assert "join_distinct_count_aggregation" in combo["correctness_risks"]
    assert "global_distinct_count" in global_combo["correctness_risks"]


def test_classify_operation_combo_tracks_multi_key_join_risk():
    combo = classify_operation_combo(
        [{"op": "join", "left_on": ["id", "g"], "right_on": ["id", "g"]}]
    )

    assert "multi_key_join" in combo["correctness_risks"]


def test_classify_operation_combo_tracks_multi_key_groupby_risk():
    combo = classify_operation_combo(
        [{"op": "groupby", "keys": ["g", "flag"], "aggs": [{"column": "x", "func": "sum", "as": "sum_x"}]}]
    )

    assert "multi_key_groupby" in combo["correctness_risks"]


def test_classify_operation_combo_tracks_negative_set_membership_risk():
    combo = classify_operation_combo(
        [{"op": "filter", "column": "g", "cmp": "not_in_set", "value": ["alpha"]}]
    )

    assert "negative_set_membership" in combo["correctness_risks"]
