from __future__ import annotations

from functools import cmp_to_key
import math
from typing import Any

from datadiff.dsl import SortKey, normalize_sort_keys


def running_sum_partition_columns(op: dict[str, Any]) -> list[str]:
    columns: list[str] = []
    seen: set[str] = set()
    for value in op.get("partition_by", []) or []:
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
    for key in normalize_sort_keys({"keys": op.get("order_by", [])}):
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
        partition_key = tuple(row.get(column) for column in partition_by)
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


def sort_rows_for_running(rows: list[dict[str, Any]], sort_keys: list[SortKey]) -> list[dict[str, Any]]:
    return sorted(rows, key=cmp_to_key(lambda left, right: _compare_rows(left, right, sort_keys)))


def _is_null_like(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    return False


def _compare_rows(left: dict[str, Any], right: dict[str, Any], sort_keys: list[SortKey]) -> int:
    for key in sort_keys:
        left_value = left.get(key.column)
        right_value = right.get(key.column)
        if left_value is None and right_value is None:
            continue
        if left_value is None:
            return -1 if key.nulls == "first" else 1
        if right_value is None:
            return 1 if key.nulls == "first" else -1
        if left_value < right_value:
            return -1 if key.ascending else 1
        if left_value > right_value:
            return 1 if key.ascending else -1
    return 0
