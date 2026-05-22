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
