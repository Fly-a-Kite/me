from __future__ import annotations

from typing import Any

from datadiff.dsl import SortKey, normalize_sort_keys
from datadiff.running import sort_rows_for_running


def row_number_partition_columns(op: dict[str, Any]) -> list[str]:
    columns: list[str] = []
    seen: set[str] = set()
    for value in op.get("partition_by", []) or []:
        column = str(value)
        if not column or column in seen:
            continue
        columns.append(column)
        seen.add(column)
    return columns


def row_number_order_keys(op: dict[str, Any]) -> list[SortKey]:
    return normalize_sort_keys({"keys": op.get("order_by", [])})


def row_number_filter_rows(rows: list[dict[str, Any]], op: dict[str, Any]) -> list[dict[str, Any]]:
    partition_columns = row_number_partition_columns(op)
    order_keys = row_number_order_keys(op)
    sorted_rows = sort_rows_for_running(
        rows,
        [
            *(SortKey(column=column, ascending=True, nulls="last") for column in partition_columns),
            *order_keys,
        ],
    )
    counters: dict[tuple[Any, ...], int] = {}
    out = []
    for row in sorted_rows:
        partition_key = tuple(row.get(column) for column in partition_columns)
        row_number = counters.get(partition_key, 0) + 1
        counters[partition_key] = row_number
        if row_number_matches(row_number, str(op.get("cmp", "==")), int(op.get("value", 1))):
            out.append(row)
    return out


def row_number_matches(row_number: int, comparator: str, value: int) -> bool:
    if comparator == "==":
        return row_number == value
    if comparator == "<":
        return row_number < value
    if comparator == "<=":
        return row_number <= value
    raise ValueError(comparator)
