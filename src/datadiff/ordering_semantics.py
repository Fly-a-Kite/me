from __future__ import annotations

from functools import cmp_to_key
import math
from typing import Any, Mapping, Sequence

from datadiff.canonicalization import mapping_projection_key
from datadiff.dsl import SortKey


def is_null_like(value: Any) -> bool:
    if value is None:
        return True
    if type(value).__name__ in {"NAType", "NaTType"}:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    try:
        return bool(value != value)
    except Exception:
        return False


def order_key_value(value: Any) -> Any:
    return None if is_null_like(value) else value


def compare_scalar_values(left: Any, right: Any, *, fallback_to_repr: bool = False) -> int:
    try:
        if left < right:
            return -1
        if left > right:
            return 1
        return 0
    except TypeError:
        if not fallback_to_repr:
            raise
        left_text = repr(left)
        right_text = repr(right)
        if left_text < right_text:
            return -1
        if left_text > right_text:
            return 1
        return 0


def compare_row_mappings(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    sort_keys: Sequence[SortKey],
    *,
    null_like: bool = False,
    scalar_fallback_to_repr: bool = False,
) -> int:
    for key in sort_keys:
        left_value = left.get(key.column)
        right_value = right.get(key.column)
        left_null = is_null_like(left_value) if null_like else left_value is None
        right_null = is_null_like(right_value) if null_like else right_value is None
        if left_null and right_null:
            continue
        if left_null:
            return -1 if key.nulls == "first" else 1
        if right_null:
            return 1 if key.nulls == "first" else -1
        cmp = compare_scalar_values(left_value, right_value, fallback_to_repr=scalar_fallback_to_repr)
        if cmp:
            return cmp if key.ascending else -cmp
    return 0


def sort_row_mappings(
    rows: Sequence[Mapping[str, Any]],
    sort_keys: Sequence[SortKey],
    *,
    null_like: bool = False,
    scalar_fallback_to_repr: bool = False,
) -> list[Mapping[str, Any]]:
    return sorted(
        rows,
        key=cmp_to_key(
            lambda left, right: compare_row_mappings(
                left,
                right,
                sort_keys,
                null_like=null_like,
                scalar_fallback_to_repr=scalar_fallback_to_repr,
            )
        ),
    )


def sort_sample_indices(
    samples: Mapping[str, Sequence[Any]],
    sort_keys: Sequence[SortKey],
    *,
    null_like: bool = False,
    scalar_fallback_to_repr: bool = False,
) -> list[int]:
    row_count = min((len(values) for values in samples.values()), default=0)
    if row_count <= 1:
        return list(range(row_count))
    column_views: list[tuple[SortKey, Sequence[Any], list[bool]]] = []
    for key in sort_keys:
        values = samples.get(key.column)
        if values is None:
            continue
        null_flags = [
            is_null_like(value) if null_like else value is None
            for value in values[:row_count]
        ]
        column_views.append((key, values, null_flags))
    if not column_views:
        return list(range(row_count))

    def compare_indices(left_idx: int, right_idx: int) -> int:
        for key, values, null_flags in column_views:
            left_null = null_flags[left_idx]
            right_null = null_flags[right_idx]
            if left_null and right_null:
                continue
            if left_null:
                return -1 if key.nulls == "first" else 1
            if right_null:
                return 1 if key.nulls == "first" else -1
            cmp = compare_scalar_values(
                values[left_idx],
                values[right_idx],
                fallback_to_repr=scalar_fallback_to_repr,
            )
            if cmp:
                return cmp if key.ascending else -cmp
        return 0

    return sorted(range(row_count), key=cmp_to_key(compare_indices))


def reorder_samples_by_index(
    samples: Mapping[str, Sequence[Any]],
    indices: Sequence[int],
    columns: Sequence[str] | None = None,
) -> dict[str, list[Any]]:
    selected_columns = [column for column in (columns or list(samples)) if column in samples]
    if not selected_columns:
        return {}
    row_count = min((len(samples[column]) for column in selected_columns), default=0)
    bounded_indices = [index for index in indices if 0 <= index < row_count]
    return {
        column: [samples[column][index] for index in bounded_indices]
        for column in selected_columns
    }


def is_sorted_scalar_values(
    values: Sequence[Any],
    *,
    ascending: bool = True,
    nulls: str = "last",
    null_like: bool = True,
    scalar_fallback_to_repr: bool = False,
) -> bool:
    for left, right in zip(values, values[1:]):
        left_null = is_null_like(left) if null_like else left is None
        right_null = is_null_like(right) if null_like else right is None
        if left_null and right_null:
            continue
        if left_null:
            if nulls == "last":
                return False
            continue
        if right_null:
            if nulls == "first":
                return False
            continue
        cmp = compare_scalar_values(left, right, fallback_to_repr=scalar_fallback_to_repr)
        if ascending:
            if cmp > 0:
                return False
        elif cmp < 0:
            return False
    return True


def sort_key_signature(row: Mapping[str, Any], sort_keys: Sequence[SortKey]) -> str:
    return mapping_projection_key(row, [sort_key.column for sort_key in sort_keys])


def rows_have_duplicate_sort_key(rows: Sequence[Mapping[str, Any]], sort_keys: Sequence[SortKey]) -> bool:
    seen: set[str] = set()
    for row in rows:
        key = sort_key_signature(row, sort_keys)
        if key in seen:
            return True
        seen.add(key)
    return False


def sort_boundary_splits_tie(rows: Sequence[Mapping[str, Any]], sort_keys: Sequence[SortKey], boundary: int) -> bool:
    if boundary <= 0 or boundary >= len(rows):
        return False
    return sort_key_signature(rows[boundary - 1], sort_keys) == sort_key_signature(rows[boundary], sort_keys)


def sort_window_boundary_splits_tie(
    rows: Sequence[Mapping[str, Any]],
    sort_keys: Sequence[SortKey],
    start: int,
    end: int,
) -> bool:
    return sort_boundary_splits_tie(rows, sort_keys, start) or sort_boundary_splits_tie(rows, sort_keys, end)
