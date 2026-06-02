from __future__ import annotations

from datadiff.dsl import SortKey
from datadiff.ordering_semantics import (
    compare_row_mappings,
    rows_have_duplicate_sort_key,
    sort_boundary_splits_tie,
    sort_key_signature,
    sort_row_mappings,
    sort_window_boundary_splits_tie,
)


def test_sort_row_mappings_respects_null_placement_and_direction():
    rows = [
        {"x": None, "s": "b"},
        {"x": 1, "s": "a"},
        {"x": 1, "s": "c"},
    ]
    keys = [SortKey("x", True, "last"), SortKey("s", False, "last")]

    assert sort_row_mappings(rows, keys) == [
        {"x": 1, "s": "c"},
        {"x": 1, "s": "a"},
        {"x": None, "s": "b"},
    ]


def test_compare_row_mappings_supports_nan_as_null_like_when_requested():
    left = {"x": float("nan")}
    right = {"x": 1.0}
    keys = [SortKey("x", True, "last")]

    assert compare_row_mappings(left, right, keys, null_like=True) == 1


def test_sort_key_signature_and_duplicate_detection_follow_selected_columns():
    rows = [
        {"x": 1, "s": "a"},
        {"x": 1, "s": "b"},
        {"x": 2, "s": "c"},
    ]
    keys = [SortKey("x", True, "last")]

    assert sort_key_signature(rows[0], keys) == sort_key_signature(rows[1], keys)
    assert rows_have_duplicate_sort_key(rows, keys) is True


def test_sort_boundary_tie_detection_marks_cutoff_inside_tie_group():
    rows = [
        {"x": 1, "s": "a"},
        {"x": 1, "s": "b"},
        {"x": 2, "s": "c"},
    ]
    keys = [SortKey("x", True, "last")]

    assert sort_boundary_splits_tie(rows, keys, 1) is True
    assert sort_window_boundary_splits_tie(rows, keys, 0, 1) is True
