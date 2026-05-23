from __future__ import annotations

from functools import cmp_to_key
from typing import Any

from datadiff.dsl import SortKey


def stable_running_sum_values(rows: list[dict[str, Any]], source: str) -> list[float | None]:
    total = 0.0
    seen = False
    out: list[float | None] = []
    for row in rows:
        value = row.get(source)
        if value is None:
            out.append(total if seen else None)
            continue
        total += float(value)
        seen = True
        out.append(total)
    return out


def sort_rows_for_running(rows: list[dict[str, Any]], sort_keys: list[SortKey]) -> list[dict[str, Any]]:
    return sorted(rows, key=cmp_to_key(lambda left, right: _compare_rows(left, right, sort_keys)))


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
