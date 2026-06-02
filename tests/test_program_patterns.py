from datadiff.program_patterns import (
    has_distinct_null_topk_pattern,
    has_empty_filter_aggregate_pattern,
    has_float_group_key_pattern,
    has_global_null_aggregate_pattern,
    has_groupby_filter_cast_membership_pattern,
    has_join_null_truth_filter_pattern,
    has_join_ordered_agg_topk_pattern,
    has_join_null_sort_pattern,
    has_ordered_groupby_sort_pattern,
    has_sortedness_null_placement_pattern,
    has_topk_resort_pattern,
    has_wide_offset_topk_pattern,
    program_pattern_features,
)


def test_program_pattern_features_collects_representative_patterns():
    operations = [
        {"op": "sort", "columns": ["x"]},
        {"op": "offset", "n": 2},
        {"op": "limit", "n": 3},
        {"op": "filter", "column": "x", "cmp": "range_closed", "value": [0, 5]},
    ]

    patterns = program_pattern_features(
        operations,
        ["sort:null-order"],
        has_range_filter=True,
    )

    assert "pattern:wide_offset_topk" in patterns
    assert "pattern:post_topk_range_filter" in patterns


def test_join_null_truth_filter_pattern_requires_left_join_truth_filter():
    positive = [
        {"op": "join", "table": "t1", "left_on": "id", "right_on": "id", "how": "left"},
        {"op": "filter", "column": "flag_r", "cmp": "bool_is_not_true", "value": None},
    ]
    negative = [
        {"op": "join", "table": "t1", "left_on": "id", "right_on": "id", "how": "inner"},
        {"op": "filter", "column": "flag_r", "cmp": "bool_is_not_true", "value": None},
    ]

    assert has_join_null_truth_filter_pattern(positive) is True
    assert has_join_null_truth_filter_pattern(negative) is False


def test_groupby_cast_membership_pattern_detects_fractional_literal_against_agg_alias():
    operations = [
        {"op": "groupby", "keys": ["g"], "aggs": [{"column": "x", "func": "mean", "as": "mean_x"}]},
        {"op": "filter", "column": "mean_x", "cmp": "in_set", "value": [1.5, 2.5]},
    ]

    assert has_groupby_filter_cast_membership_pattern(operations) is True


def test_float_group_key_pattern_detects_division_derived_group_key():
    operations = [
        {"op": "mutate", "column": "ratio", "expr": {"kind": "arith_const", "source": "x", "op": "div", "value": 2}},
        {"op": "groupby", "keys": ["ratio"], "aggs": [{"column": "x", "func": "sum", "as": "sum_x"}]},
    ]

    assert has_float_group_key_pattern(operations) is True


def test_ordered_groupby_sort_and_topk_resort_patterns_detect_structure():
    ordered = [
        {"op": "sort", "columns": ["x"], "ascending": True},
        {"op": "groupby", "keys": ["g"], "aggs": [{"column": "x", "func": "sum", "as": "sum_x"}]},
        {"op": "sort", "columns": ["sum_x"], "ascending": False},
    ]
    resort = [
        {"op": "sort", "columns": ["x"], "ascending": True},
        {"op": "limit", "n": 3},
        {"op": "sort", "columns": ["g"], "ascending": True},
    ]

    assert has_ordered_groupby_sort_pattern(ordered) is True
    assert has_topk_resort_pattern(resort) is True


def test_sortedness_null_placement_pattern_detects_mismatch():
    operations = [
        {"op": "sort", "keys": [{"column": "x", "ascending": True, "nulls": "first"}]},
        {"op": "sortedness_check", "column": "x", "ascending": True, "nulls": "last"},
    ]

    assert has_sortedness_null_placement_pattern(operations) is True


def test_join_ordered_agg_topk_pattern_detects_join_sort_groupby_sort_limit():
    operations = [
        {"op": "join", "table": "t1", "left_on": "id", "right_on": "id", "how": "inner"},
        {"op": "sort", "columns": ["x"], "ascending": True},
        {"op": "groupby", "keys": ["g"], "aggs": [{"column": "x", "func": "sum", "as": "sum_x"}]},
        {"op": "sort", "columns": ["sum_x"], "ascending": False},
        {"op": "limit", "n": 2},
    ]

    assert has_join_ordered_agg_topk_pattern(operations) is True


def test_global_null_aggregate_pattern_uses_frontier_buckets():
    operations = [
        {"op": "aggregate", "aggs": [{"column": "x", "func": "sum", "as": "sum_x"}]},
    ]

    assert has_global_null_aggregate_pattern(
        operations,
        ["aggregate:global", "aggregate:null-output"],
    ) is True
    assert has_global_null_aggregate_pattern(
        operations,
        ["aggregate:null-output"],
    ) is False


def test_join_null_sort_pattern_requires_left_join_sort_limit_with_null_frontier():
    operations = [
        {"op": "join", "table": "t1", "left_on": "id", "right_on": "id", "how": "left"},
        {"op": "sort", "columns": ["j", "id"], "ascending": True},
        {"op": "limit", "n": 2},
    ]

    assert has_join_null_sort_pattern(operations, ["sort:null-order"]) is True


def test_distinct_null_topk_and_empty_filter_aggregate_patterns():
    distinct_ops = [
        {"op": "distinct", "columns": ["g"]},
        {"op": "sort", "keys": [{"column": "g", "ascending": True, "nulls": "first"}]},
        {"op": "limit", "n": 1},
    ]
    empty_filter_agg_ops = [
        {"op": "filter", "column": "x", "cmp": ">", "value": 999},
        {"op": "aggregate", "aggs": [{"column": "x", "func": "sum", "as": "sum_x"}]},
    ]

    assert has_distinct_null_topk_pattern(distinct_ops, ["sort:null-order"]) is True
    assert has_empty_filter_aggregate_pattern(empty_filter_agg_ops, ["filter:empty-output"]) is True


def test_running_sum_precision_pattern_feature_is_emitted():
    operations = [{"op": "running_sum", "column": "x", "input_dtype": "float32"}]

    patterns = program_pattern_features(operations, [])

    assert "pattern:running_sum_precision" in patterns
