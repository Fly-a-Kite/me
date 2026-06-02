from datadiff.rust_kernel import (
    canonicalize_row_order,
    canonical_sort_key,
    compare_result_batch_anchor_summary,
    compare_result_batch,
    compare_result_batch_summary,
    compare_row_set_batch,
    compare_row_set_batch_summary,
    compare_row_sets,
    compare_row_sets_summary,
    dedupe_canonical_rows,
    has_duplicate_rows,
    json_canonical_dumps,
    native_available,
    ordered_row_signatures,
    profile_result_batch,
    row_profile,
    row_profiles,
    row_set_profiles,
    short_sha256_hex,
    sorted_canonical_rows,
    stable_rows,
    unordered_row_signatures,
    unordered_row_signature,
)
import datadiff.rust_kernel as rust_kernel_module
from datadiff.normalizer import NormalizedResult


def test_rust_kernel_canonical_dumps_is_stable_for_mapping_order():
    left = {"b": [2, 1], "a": {"y": 2, "x": 1}}
    right = {"a": {"x": 1, "y": 2}, "b": [2, 1]}

    assert json_canonical_dumps(left) == json_canonical_dumps(right)


def test_rust_kernel_short_sha256_hex_is_stable():
    payload = {"kind": "semantic_output_mismatch", "rows": [[1, None], [2, 3.5]]}

    first = short_sha256_hex(payload, 16)
    second = short_sha256_hex(payload, 16)

    assert first == second
    assert len(first) == 16


def test_rust_kernel_fallback_or_native_is_available():
    assert native_available() in {True, False}


def test_rust_kernel_canonical_sort_key_is_stable():
    left = [{"b": 2, "a": 1}, {"x": None, "y": [3, 2, 1]}]
    right = [{"a": 1, "b": 2}, {"y": [3, 2, 1], "x": None}]

    assert canonical_sort_key(left) == canonical_sort_key(right)


def test_rust_kernel_row_helpers_handle_duplicates_and_order_insensitive_signature():
    rows = [[{"b": 2, "a": 1}], [{"x": None, "y": [3, 2, 1]}], [{"a": 1, "b": 2}]]

    rendered = stable_rows(rows)

    assert len(rendered) == 3
    assert has_duplicate_rows(rows) is True
    assert sorted_canonical_rows(rows) == sorted(rendered)
    assert dedupe_canonical_rows(rows) == [rendered[0], rendered[1]]
    assert unordered_row_signature(rows) == unordered_row_signature(list(reversed(rows)))


def test_rust_kernel_batch_row_signatures_match_scalar_helpers():
    row_sets = [
        [[{"b": 2, "a": 1}], [{"x": None, "y": [3, 2, 1]}]],
        [[{"a": 1, "b": 2}], [{"y": [3, 2, 1], "x": None}]],
    ]

    assert ordered_row_signatures(row_sets) == [json_canonical_dumps(stable_rows(rows)) for rows in row_sets]
    assert unordered_row_signatures(row_sets) == [unordered_row_signature(rows) for rows in row_sets]


def test_rust_kernel_row_set_profiles_match_scalar_helpers():
    row_sets = [
        [[{"b": 2, "a": 1}], [{"x": None, "y": [3, 2, 1]}]],
        [[{"a": 1, "b": 2}], [{"y": [3, 2, 1], "x": None}]],
    ]

    assert row_set_profiles(row_sets) == [
        (ordered_row_signatures([rows])[0], unordered_row_signatures([rows])[0])
        for rows in row_sets
    ]


def test_rust_kernel_row_profile_includes_duplicate_flag():
    rows = [[{"b": 2, "a": 1}], [{"x": None, "y": [3, 2, 1]}], [{"a": 1, "b": 2}]]

    ordered, unordered, has_duplicates = row_profile(rows)

    assert ordered == json_canonical_dumps(stable_rows(rows))
    assert unordered == unordered_row_signature(rows)
    assert has_duplicates is True


def test_rust_kernel_row_profiles_match_scalar_helpers():
    row_sets = [
        [[{"b": 2, "a": 1}], [{"x": None, "y": [3, 2, 1]}], [{"a": 1, "b": 2}]],
        [[{"x": None, "y": [3, 2, 1]}]],
    ]

    assert row_profiles(row_sets) == [
        (json_canonical_dumps(stable_rows(rows)), unordered_row_signature(rows), has_duplicate_rows(rows))
        for rows in row_sets
    ]


def test_rust_kernel_canonicalize_row_order_returns_indices_keys_and_signature():
    rows = [[2, "b"], [1, "c"], [1, "a"], [1, "a"]]

    indices, ordered_keys, ordered_signature, has_duplicates = canonicalize_row_order(rows)

    ordered_rows = [rows[index] for index in indices]
    assert ordered_rows == [[1, "a"], [1, "a"], [1, "c"], [2, "b"]]
    assert ordered_keys == stable_rows(ordered_rows)
    assert ordered_signature == json_canonical_dumps(ordered_keys)
    assert has_duplicates is True


def test_rust_kernel_compare_row_sets_returns_row_count_order_and_value():
    assert compare_row_sets([[1]], [[1], [2]]) == ("row_count", False, False, [], [])
    assert compare_row_sets([[1], [2]], [[2], [1]]) == ("row_order", True, False, [], [])
    mismatch_class, same_row_count, same_ordered_rows, left_only, right_only = compare_row_sets([[1], [3]], [[1], [2]])
    assert mismatch_class == "value"
    assert same_row_count is True
    assert same_ordered_rows is False
    assert left_only == [json_canonical_dumps([3])]
    assert right_only == [json_canonical_dumps([2])]


def test_rust_kernel_compare_row_sets_summary_returns_unordered_match_state():
    mismatch_class, same_row_count, same_ordered_rows, same_unordered_rows, left_only, right_only = compare_row_sets_summary(
        [[1], [2]],
        [[2], [1]],
    )

    assert mismatch_class == "row_order"
    assert same_row_count is True
    assert same_ordered_rows is False
    assert same_unordered_rows is True
    assert left_only == []
    assert right_only == []


def test_rust_kernel_compare_row_set_batch_returns_group_ids_and_mismatch():
    group_ids, mismatch_class = compare_row_set_batch([[[1], [2]], [[2], [1]], [[1], [3]]])

    assert len(group_ids) == 3
    assert mismatch_class == "value"
    assert group_ids[0] != group_ids[1]
    assert group_ids[0] != group_ids[2]


def test_rust_kernel_compare_row_set_batch_returns_row_count_for_mixed_cardinality():
    group_ids, mismatch_class = compare_row_set_batch([[[1]], [[1]], [[1], [1]]])

    assert group_ids == [0, 0, 1]
    assert mismatch_class == "row_count"


def test_rust_kernel_compare_row_set_batch_summary_returns_majority_and_suspicious_indices():
    group_ids, mismatch_class, suspicious_indices, has_clear_majority, majority_group = compare_row_set_batch_summary(
        [[[1]], [[1]], [[2]]]
    )

    assert group_ids == [0, 0, 1]
    assert mismatch_class == "value"
    assert suspicious_indices == [2]
    assert has_clear_majority is True
    assert majority_group == [0, 1]


def test_rust_kernel_compare_result_batch_returns_status_and_schema_and_row_count():
    status_group_ids, status_mismatch_class = compare_result_batch(
        [
            {"status": "ok", "columns": ["x"], "rows": [[1]], "error_type": ""},
            {"status": "error", "columns": [], "rows": [], "error_type": "ValueError"},
        ]
    )
    assert status_group_ids == [0, 1]
    assert status_mismatch_class == "status"

    schema_group_ids, schema_mismatch_class = compare_result_batch(
        [
            {"status": "ok", "columns": ["x"], "rows": [[1]], "error_type": ""},
            {"status": "ok", "columns": ["y"], "rows": [[1]], "error_type": ""},
        ]
    )
    assert schema_group_ids == [0, 1]
    assert schema_mismatch_class == "schema"

    row_count_group_ids, row_count_mismatch_class = compare_result_batch(
        [
            {"status": "ok", "columns": ["x"], "rows": [[1]], "error_type": ""},
            {"status": "ok", "columns": ["x"], "rows": [[1]], "error_type": ""},
            {"status": "ok", "columns": ["x"], "rows": [[1], [1]], "error_type": ""},
        ]
    )
    assert row_count_group_ids == [0, 0, 1]
    assert row_count_mismatch_class == "row_count"


def test_rust_kernel_compare_result_batch_summary_returns_majority_and_suspicious_indices():
    group_ids, mismatch_class, suspicious_indices, has_clear_majority, majority_group = compare_result_batch_summary(
        [
            {"status": "ok", "columns": ["x"], "rows": [[1]], "error_type": ""},
            {"status": "ok", "columns": ["x"], "rows": [[1]], "error_type": ""},
            {"status": "ok", "columns": ["x"], "rows": [[2]], "error_type": ""},
        ]
    )

    assert group_ids == [0, 0, 1]
    assert mismatch_class == "value"
    assert suspicious_indices == [2]
    assert has_clear_majority is True
    assert majority_group == [0, 1]


def test_rust_kernel_result_compare_interfaces_accept_result_objects():
    results = [
        NormalizedResult("pandas", "ok", ["x"], [[1]]),
        NormalizedResult("duckdb", "ok", ["x"], [[2]]),
        NormalizedResult("sqlite", "ok", ["x"], [[2]]),
    ]

    group_ids, mismatch_class = compare_result_batch(results)
    summary = compare_result_batch_summary(results)
    anchor_summary = compare_result_batch_anchor_summary(results, 1)
    profiles = profile_result_batch(results)

    assert group_ids == [0, 1, 1]
    assert mismatch_class == "value"
    assert summary == ([0, 1, 1], "value", [0], True, [1, 2])
    assert anchor_summary == ([1, 2], [0], "high")
    assert [profile[0] for profile in profiles] == ["ok", "ok", "ok"]


def test_rust_kernel_compare_result_batch_anchor_summary_returns_matching_and_mismatching_indices():
    matching_indices, mismatching_indices, confidence = compare_result_batch_anchor_summary(
        [
            {"status": "ok", "columns": ["x"], "rows": [[1]], "error_type": ""},
            {"status": "ok", "columns": ["x"], "rows": [[1]], "error_type": ""},
            {"status": "ok", "columns": ["x"], "rows": [[2]], "error_type": ""},
        ],
        0,
    )

    assert matching_indices == [0, 1]
    assert mismatching_indices == [2]
    assert confidence == "high"


def test_rust_kernel_profile_result_batch_returns_result_profiles():
    profiles = profile_result_batch(
        [
            {"status": "ok", "columns": ["x"], "rows": [[1], [1]], "error_type": ""},
            {"status": "error", "columns": [], "rows": [], "error_type": "ValueError"},
        ]
    )

    assert profiles[0][0] == "ok"
    assert profiles[0][1] == ["x"]
    assert profiles[0][5] is True
    assert profiles[1][0] == "error"
    assert profiles[1][2] == "ValueError"


def test_rust_kernel_result_compare_and_profile_fallbacks_work_without_native(monkeypatch):
    monkeypatch.setattr(rust_kernel_module, "_load_native", lambda: None)

    group_ids, mismatch_class = rust_kernel_module.compare_result_batch(
        [
            {"status": "ok", "columns": ["x"], "rows": [[1]], "error_type": ""},
            {"status": "ok", "columns": ["x"], "rows": [[1], [1]], "error_type": ""},
        ]
    )
    summary = rust_kernel_module.compare_result_batch_summary(
        [
            {"status": "ok", "columns": ["x"], "rows": [[1]], "error_type": ""},
            {"status": "ok", "columns": ["x"], "rows": [[1], [1]], "error_type": ""},
        ]
    )
    anchor_summary = rust_kernel_module.compare_result_batch_anchor_summary(
        [
            {"status": "ok", "columns": ["x"], "rows": [[1]], "error_type": ""},
            {"status": "ok", "columns": ["x"], "rows": [[1], [1]], "error_type": ""},
        ],
        0,
    )
    profiles = rust_kernel_module.profile_result_batch(
        [
            {"status": "ok", "columns": ["x"], "rows": [[1], [1]], "error_type": ""},
            {"status": "error", "columns": [], "rows": [], "error_type": "ValueError"},
        ]
    )

    assert group_ids == [0, 1]
    assert mismatch_class == "row_count"
    assert summary == ([0, 1], "row_count", [0, 1], False, [])
    assert anchor_summary == ([0], [1], "medium")
    assert profiles[0][0] == "ok"
    assert profiles[0][1] == ["x"]
    assert profiles[1][2] == "ValueError"
