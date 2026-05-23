from datadiff.operation_combo import classify_operation_combo


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


def test_classify_operation_combo_tracks_post_topk_filter_pushdown_risk():
    combo = classify_operation_combo([{"op": "sort"}, {"op": "limit"}, {"op": "filter"}])

    assert combo["has_sort_limit"] is True
    assert "topk_ordering" in combo["correctness_risks"]
    assert "topk_filter_pushdown" in combo["correctness_risks"]


def test_classify_operation_combo_tracks_tuple_absence_null_filter_risk():
    combo = classify_operation_combo([{"op": "tuple_absence_filter"}, {"op": "select"}])

    assert combo["template"] == "tuple_absence_filter_select"
    assert "tuple_absence_null_filter" in combo["correctness_risks"]


def test_classify_operation_combo_tracks_running_sum_precision_risk():
    combo = classify_operation_combo([{"op": "running_sum"}, {"op": "sort"}, {"op": "limit"}])

    assert combo["template"] == "running_sum_sort_limit"
    assert "running_sum_precision" in combo["correctness_risks"]
    assert combo["has_sort_limit"] is True


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


def test_classify_operation_combo_tracks_series_rtruediv_operand_order_risk():
    combo = classify_operation_combo([{"op": "series_rtruediv_probe"}])

    assert combo["template"] == "series_rtruediv_probe"
    assert "series_rtruediv_operand_order" in combo["correctness_risks"]


def test_classify_operation_combo_tracks_pandas_uint64_isin_precision_risk():
    combo = classify_operation_combo([{"op": "uint64_isin_probe"}])

    assert combo["template"] == "uint64_isin_probe"
    assert "pandas_uint64_isin_precision" in combo["correctness_risks"]


def test_classify_operation_combo_tracks_duckdb_tuple_anti_null_semantics_risk():
    combo = classify_operation_combo([{"op": "tuple_anti_null_probe"}])

    assert combo["template"] == "tuple_anti_null_probe"
    assert "duckdb_tuple_anti_null_semantics" in combo["correctness_risks"]


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
