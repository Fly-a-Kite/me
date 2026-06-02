from __future__ import annotations

from datadiff.canonicalization import (
    compare_result_batch,
    compare_result_against_anchor,
    compare_row_set_batch,
    compare_row_sets,
    canonicalize_rows,
    canonical_key,
    canonical_keys,
    dedupe_by_canonical_key,
    dedupe_stable_canonical_rows,
    has_duplicate_canonical_key,
    has_duplicate_canonical_rows,
    mapping_projection_key,
    multiset_canonical_row_diff,
    ordered_signature,
    ordered_signatures,
    profile_row_sets,
    profile_rows,
    profile_result,
    profile_result_payload,
    profile_result_payloads,
    result_comparison_key,
    sort_by_canonical_key,
    sorted_row_signatures,
    short_canonical_hash,
    stable_canonical_rows,
    unordered_signature,
    unordered_signatures,
)


def test_canonical_keys_match_scalar_canonical_key_order():
    payloads = [{"b": 2, "a": 1}, {"x": None}, [3, 2, 1]]

    assert canonical_keys(payloads) == [canonical_key(payload) for payload in payloads]


def test_sort_by_canonical_key_orders_rows_stably():
    rows = [[2, "b"], [1, "c"], [1, "a"]]

    assert sort_by_canonical_key(rows) == [[1, "a"], [1, "c"], [2, "b"]]


def test_dedupe_by_canonical_key_preserves_first_occurrence():
    rows = [{"a": 1, "b": 2}, {"b": 2, "a": 1}, {"a": 2, "b": 3}]

    assert dedupe_by_canonical_key(rows) == [{"a": 1, "b": 2}, {"a": 2, "b": 3}]
    assert has_duplicate_canonical_key(rows) is True


def test_mapping_projection_key_supports_value_normalization():
    row = {"g": "a", "x": 1.0, "y": 3}

    raw = mapping_projection_key(row, ["g", "x"])
    normalized = mapping_projection_key(row, ["g", "x"], normalize_value=lambda value: int(value) if value == 1.0 else value)

    assert raw != ""
    assert normalized == canonical_key(["a", 1])


def test_canonical_row_helpers_share_stable_signatures_and_diff():
    rows = [{"b": 2, "a": 1}, {"x": None}, {"a": 1, "b": 2}]
    row_sets = [rows[:2], list(reversed(rows[:2]))]
    diff_right = [rows[1], {"z": 9}]

    stable = stable_canonical_rows(rows)
    diff_right_stable = stable_canonical_rows(diff_right)
    assert len(stable) == 3
    assert has_duplicate_canonical_rows(rows) is True
    assert dedupe_stable_canonical_rows(rows) == [stable[0], stable[1]]
    assert ordered_signature(row_sets[0]) != ordered_signature(row_sets[1])
    assert unordered_signature(row_sets[0]) == unordered_signature(row_sets[1])
    assert ordered_signatures(row_sets) == [ordered_signature(item) for item in row_sets]
    assert unordered_signatures(row_sets) == [unordered_signature(item) for item in row_sets]
    assert sorted_row_signatures(row_sets) == [
        (ordered_signature(item), unordered_signature(item))
        for item in row_sets
    ]
    assert multiset_canonical_row_diff(rows[:2], diff_right) == ([stable[0]], [diff_right_stable[1]])


def test_row_profiles_expose_order_unordered_and_duplicate_state():
    rows = [{"b": 2, "a": 1}, {"x": None}, {"a": 1, "b": 2}]
    row_sets = [rows, rows[:2]]

    single = profile_rows(rows)
    batch = profile_row_sets(row_sets)

    assert single.ordered_signature == ordered_signature(rows)
    assert single.unordered_signature == unordered_signature(rows)
    assert single.has_duplicates is True
    assert [(item.ordered_signature, item.unordered_signature) for item in batch] == sorted_row_signatures(row_sets)
    assert [item.has_duplicates for item in batch] == [True, False]


def test_canonicalize_rows_returns_sorted_rows_keys_and_profile():
    rows = [[2, "b"], [1, "c"], [1, "a"], [1, "a"]]

    canonicalized = canonicalize_rows(rows)

    assert canonicalized.rows == [[1, "a"], [1, "a"], [1, "c"], [2, "b"]]
    assert canonicalized.stable_row_keys == [canonical_key(row) for row in canonicalized.rows]
    assert canonicalized.profile.ordered_signature == ordered_signature(canonicalized.rows)
    assert canonicalized.profile.unordered_signature == unordered_signature(canonicalized.rows)
    assert canonicalized.profile.has_duplicates is True


def test_compare_row_sets_classifies_schema_count_order_and_value():
    assert compare_row_sets([[1]], [[1]], left_columns=["x"], right_columns=["y"]).mismatch_class == "schema"
    assert compare_row_sets([[1]], [[1], [2]], left_columns=["x"], right_columns=["x"]).mismatch_class == "row_count"
    assert compare_row_sets([[1], [2]], [[2], [1]], left_columns=["x"], right_columns=["x"]).mismatch_class == "row_order"
    value_cmp = compare_row_sets([[1], [3]], [[1], [2]], left_columns=["x"], right_columns=["x"])
    assert value_cmp.mismatch_class == "value"
    assert value_cmp.left_only == [canonical_key([3])]
    assert value_cmp.right_only == [canonical_key([2])]
    assert compare_row_sets([[1], [2]], [[1], [2]], left_columns=["x"], right_columns=["x"]).mismatch_class == "none"


def test_row_comparison_exposes_single_compare_view_helpers():
    exact = compare_row_sets([[1], [2]], [[1], [2]], left_columns=["x"], right_columns=["x"])
    order_only = compare_row_sets([[1], [2]], [[2], [1]], left_columns=["x"], right_columns=["x"])
    value_cmp = compare_row_sets([[1], [3]], [[1], [2]], left_columns=["x"], right_columns=["x"])

    assert exact.has_mismatch is False
    assert exact.is_exact_match() is True
    assert exact.is_order_only_mismatch() is False
    assert order_only.has_mismatch is True
    assert order_only.is_order_only_mismatch() is True
    assert order_only.has_value_diff() is False
    assert value_cmp.has_value_diff() is True
    assert value_cmp.diff_size() == 2


def test_compare_row_set_batch_classifies_multi_backend_groups_and_mismatch():
    batch = compare_row_set_batch(
        [[[1], [2]], [[2], [1]], [[1], [3]]],
        column_sets=[["x"], ["x"], ["x"]],
    )
    assert batch.group_ids[0] != batch.group_ids[1]
    assert batch.group_ids[0] != batch.group_ids[2]
    assert batch.mismatch_class == "value"

    order_only = compare_row_set_batch(
        [[[1], [2]], [[2], [1]], [[1], [2]]],
        column_sets=[["x"], ["x"], ["x"]],
    )
    assert order_only.mismatch_class == "row_order"

    schema = compare_row_set_batch(
        [[[1]], [[1]]],
        column_sets=[["x"], ["y"]],
    )
    assert schema.mismatch_class == "schema"


def test_compare_row_set_batch_preserves_equal_groups_across_row_count_mismatch():
    batch = compare_row_set_batch(
        [[[1]], [[1]], [[1], [1]]],
        column_sets=[["x"], ["x"], ["x"]],
    )

    assert batch.group_ids == [0, 0, 1]
    assert batch.groups == [[0, 1], [2]]
    assert batch.group_sizes == [2, 1]
    assert batch.has_mismatch is True
    assert batch.has_clear_majority is True
    assert batch.majority_group == [0, 1]
    assert batch.suspicious_indices == [2]
    assert batch.mismatch_class == "row_count"


def test_compare_row_set_batch_marks_all_indices_suspicious_on_tied_groups():
    batch = compare_row_set_batch(
        [[[1]], [[2]], [[1]], [[2]]],
        column_sets=[["x"], ["x"], ["x"], ["x"]],
    )

    assert batch.group_ids == [0, 1, 0, 1]
    assert batch.group_sizes == [2, 2]
    assert batch.has_mismatch is True
    assert batch.has_clear_majority is False
    assert batch.majority_group is None
    assert batch.suspicious_indices == [0, 1, 2, 3]
    assert batch.mismatch_class == "value"


def test_result_comparison_key_matches_profile_result():
    result_profile = profile_result(
        status="ok",
        columns=["x"],
        rows=[[2], [1]],
    )

    assert result_profile.comparison_key == result_comparison_key(
        status="ok",
        columns=["x"],
        ordered_row_signature=result_profile.ordered_row_signature,
        error_type="",
    )


def test_profile_result_payload_supports_mapping_inputs():
    payload = {"status": "ok", "columns": ["x"], "rows": [[1], [1]], "error_type": ""}

    profile = profile_result_payload(payload)

    assert profile.status == "ok"
    assert profile.columns == ("x",)
    assert profile.has_duplicate_rows is True


def test_profile_result_payloads_support_batch_inputs():
    profiles = profile_result_payloads(
        [
            {"status": "ok", "columns": ["x"], "rows": [[1]], "error_type": ""},
            {"status": "error", "columns": [], "rows": [], "error_type": "ValueError"},
        ]
    )

    assert [profile.status for profile in profiles] == ["ok", "error"]
    assert profiles[0].columns == ("x",)
    assert profiles[1].error_type == "ValueError"


def test_compare_result_batch_classifies_status_error_and_row_semantics():
    status_comparison = compare_result_batch(
        [
            {"status": "ok", "columns": ["x"], "rows": [[1]]},
            {"status": "error", "columns": [], "rows": [], "error_type": "ValueError"},
        ]
    )
    assert status_comparison.mismatch_class == "status"
    assert status_comparison.suspicious_indices == [0, 1]

    error_comparison = compare_result_batch(
        [
            {"status": "error", "columns": [], "rows": [], "error_type": "ValueError"},
            {"status": "error", "columns": [], "rows": [], "error_type": "TypeError"},
        ]
    )
    assert error_comparison.mismatch_class == "error_type"
    assert error_comparison.suspicious_indices == [0, 1]

    row_count_comparison = compare_result_batch(
        [
            {"status": "ok", "columns": ["x"], "rows": [[1]]},
            {"status": "ok", "columns": ["x"], "rows": [[1]]},
            {"status": "ok", "columns": ["x"], "rows": [[1], [1]]},
        ]
    )
    assert row_count_comparison.group_ids == [0, 0, 1]
    assert row_count_comparison.mismatch_class == "row_count"
    assert row_count_comparison.suspicious_indices == [2]

    schema_comparison = compare_result_batch(
        [
            {"status": "ok", "columns": ["x"], "rows": [[1]]},
            {"status": "ok", "columns": ["y"], "rows": [[1]]},
        ]
    )
    assert schema_comparison.mismatch_class == "schema"


def test_batch_result_comparison_exposes_anchor_matching_and_mismatching_indices():
    comparison = compare_result_batch(
        [
            {"status": "ok", "columns": ["x"], "rows": [[1]]},
            {"status": "ok", "columns": ["x"], "rows": [[1]]},
            {"status": "ok", "columns": ["x"], "rows": [[2]]},
        ]
    )

    assert comparison.matching_indices(0) == [0, 1]
    assert comparison.mismatching_indices(0) == [2]


def test_batch_result_comparison_exposes_anchor_summary_and_confidence():
    comparison = compare_result_batch(
        [
            {"status": "ok", "columns": ["x"], "rows": [[1]]},
            {"status": "ok", "columns": ["x"], "rows": [[1]]},
            {"status": "ok", "columns": ["x"], "rows": [[2]]},
        ]
    )

    matching, mismatching, confidence = comparison.anchor_mismatch_summary(
        0,
        ["ref", "a", "b"],
        exclude_anchor_from_matches=True,
    )

    assert matching == ["a"]
    assert mismatching == ["b"]
    assert confidence == "high"


def test_anchor_result_comparison_exposes_stable_label_summary():
    comparison = compare_result_against_anchor(
        [
            {"status": "ok", "columns": ["x"], "rows": [[1]]},
            {"status": "ok", "columns": ["x"], "rows": [[1]]},
            {"status": "ok", "columns": ["x"], "rows": [[2]]},
        ],
        anchor_index=0,
    )

    assert comparison.matching_indices == [0, 1]
    assert comparison.mismatching_indices == [2]
    assert comparison.label_summary(["ref", "a", "b"], exclude_anchor_from_matches=True) == (["a"], ["b"], "high")


def test_anchor_result_comparison_supports_nonzero_anchor_index():
    comparison = compare_result_against_anchor(
        [
            {"status": "ok", "columns": ["x"], "rows": [[1]]},
            {"status": "ok", "columns": ["x"], "rows": [[2]]},
            {"status": "ok", "columns": ["x"], "rows": [[2]]},
        ],
        anchor_index=1,
    )

    assert comparison.matching_indices == [1, 2]
    assert comparison.mismatching_indices == [0]
    assert comparison.label_summary(["a", "ref", "b"], exclude_anchor_from_matches=True) == (["b"], ["a"], "high")


def test_batch_comparison_views_expose_label_level_helpers():
    row_comparison = compare_row_set_batch(
        [[[1]], [[1]], [[2]]],
        column_sets=[["x"], ["x"], ["x"]],
    )
    result_comparison = compare_result_batch(
        [
            {"status": "ok", "columns": ["x"], "rows": [[1]]},
            {"status": "ok", "columns": ["x"], "rows": [[1]]},
            {"status": "ok", "columns": ["x"], "rows": [[2]]},
        ]
    )

    assert row_comparison.suspicious_labels(["a", "b", "c"]) == ["c"]
    assert result_comparison.matching_labels(0, ["ref", "a", "b"]) == ["ref", "a"]
    assert result_comparison.mismatching_labels(0, ["ref", "a", "b"]) == ["b"]


def test_batch_comparison_views_expose_default_confidence_and_actionable_minority():
    row_comparison = compare_row_set_batch(
        [[[1]], [[1]], [[2]]],
        column_sets=[["x"], ["x"], ["x"]],
    )
    tied_comparison = compare_result_batch(
        [
            {"status": "ok", "columns": ["x"], "rows": [[1]]},
            {"status": "ok", "columns": ["x"], "rows": [[2]]},
            {"status": "ok", "columns": ["x"], "rows": [[1]]},
            {"status": "ok", "columns": ["x"], "rows": [[2]]},
        ]
    )

    assert row_comparison.default_confidence() == "high"
    assert row_comparison.has_actionable_minority() is True
    assert tied_comparison.default_confidence() == "medium"
    assert tied_comparison.has_actionable_minority() is False


def test_batch_comparison_views_expose_exact_and_order_only_checks():
    exact = compare_row_set_batch(
        [[[1], [2]], [[1], [2]]],
        column_sets=[["x"], ["x"]],
    )
    order_only = compare_row_set_batch(
        [[[1], [2]], [[2], [1]]],
        column_sets=[["x"], ["x"]],
    )

    assert exact.is_exact_match() is True
    assert exact.is_order_only_mismatch() is False
    assert order_only.is_exact_match() is False
    assert order_only.is_order_only_mismatch() is True


def test_short_canonical_hash_is_stable():
    payload = {"kind": "semantic_output_mismatch", "rows": [[1, None], [2, 3.5]]}

    first = short_canonical_hash(payload, 16)
    second = short_canonical_hash(payload, 16)

    assert first == second
    assert len(first) == 16
