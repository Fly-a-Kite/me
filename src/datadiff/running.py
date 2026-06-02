from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from datadiff.dsl import SortKey, normalize_sort_keys
from datadiff.ordering_semantics import (
    is_null_like,
    order_key_value,
    reorder_samples_by_index,
    sort_row_mappings,
    sort_sample_indices,
)
from datadiff.operation_semantics import op_partition_columns, op_sort_keys, expr_source


def running_sum_partition_columns(op: dict[str, Any]) -> list[str]:
    columns: list[str] = []
    seen: set[str] = set()
    for value in op_partition_columns(op):
        column = str(value)
        if not column or column in seen:
            continue
        columns.append(column)
        seen.add(column)
    return columns


def running_sum_sort_keys(op: dict[str, Any]) -> list[SortKey]:
    keys: list[SortKey] = []
    seen: set[str] = set()
    for column in running_sum_partition_columns(op):
        keys.append(SortKey(column=column, ascending=True, nulls="last"))
        seen.add(column)
    raw_sort_keys = op_sort_keys(op)
    normalized_keys = raw_sort_keys if raw_sort_keys and all(hasattr(key, "column") for key in raw_sort_keys) else normalize_sort_keys({"keys": raw_sort_keys})
    for key in normalized_keys:
        if key.column in seen:
            continue
        keys.append(key)
        seen.add(key.column)
    return keys


def stable_running_sum_values(
    rows: list[dict[str, Any]],
    source: str,
    partition_by: list[str] | tuple[str, ...] = (),
) -> list[float | None]:
    totals: dict[tuple[Any, ...], float] = {}
    seen_values: set[tuple[Any, ...]] = set()
    out: list[float | None] = []
    for row in rows:
        partition_key = tuple(window_key_value(row.get(column)) for column in partition_by)
        total = totals.get(partition_key, 0.0)
        value = row.get(source)
        if _is_null_like(value):
            out.append(total if partition_key in seen_values else None)
            continue
        total += float(value)
        totals[partition_key] = total
        seen_values.add(partition_key)
        out.append(total)
    return out


def stable_running_sum_sample_values(
    samples: Mapping[str, Sequence[Any]],
    source: str,
    partition_by: list[str] | tuple[str, ...] = (),
) -> list[float | None]:
    source_values = samples.get(source)
    if source_values is None:
        return []
    partition_vectors = [samples.get(column, ()) for column in partition_by]
    row_count = min(
        [len(source_values), *(len(values) for values in partition_vectors)],
        default=0,
    )
    totals: dict[tuple[Any, ...], float] = {}
    seen_values: set[tuple[Any, ...]] = set()
    out: list[float | None] = []
    for idx in range(row_count):
        partition_key = tuple(window_key_value(values[idx]) for values in partition_vectors)
        total = totals.get(partition_key, 0.0)
        value = source_values[idx]
        if _is_null_like(value):
            out.append(total if partition_key in seen_values else None)
            continue
        total += float(value)
        totals[partition_key] = total
        seen_values.add(partition_key)
        out.append(total)
    return out


def _is_null_like(value: Any) -> bool:
    return is_null_like(value)


def sort_rows_for_running(rows: list[dict[str, Any]], sort_keys: list[SortKey]) -> list[dict[str, Any]]:
    return list(sort_row_mappings(rows, sort_keys, null_like=True))


def sort_samples_for_running(
    samples: Mapping[str, Sequence[Any]],
    sort_keys: list[SortKey],
) -> dict[str, list[Any]]:
    order = sort_sample_indices(samples, sort_keys, null_like=True)
    return reorder_samples_by_index(samples, order)


def window_key_value(value: Any) -> Any:
    return order_key_value(value)
