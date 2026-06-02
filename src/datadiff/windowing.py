from __future__ import annotations

from typing import Any

from datadiff.dsl import SortKey, normalize_sort_keys
from datadiff.operation_semantics import condition_cmp, op_partition_columns, op_sort_keys, op_value
from datadiff.running import sort_rows_for_running, window_key_value


def row_number_partition_columns(op: dict[str, Any]) -> list[str]:
    columns: list[str] = []
    seen: set[str] = set()
    for value in op_partition_columns(op):
        column = str(value)
        if not column or column in seen:
            continue
        columns.append(column)
        seen.add(column)
    return columns


def row_number_order_keys(op: dict[str, Any]) -> list[SortKey]:
    typed_keys = getattr(op, "order_keys", None)
    if isinstance(typed_keys, list):
        return typed_keys
    raw_keys = op_sort_keys(op)
    if raw_keys and all(hasattr(key, "column") for key in raw_keys):
        return raw_keys
    return normalize_sort_keys({"keys": raw_keys})


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
        partition_key = tuple(window_key_value(row.get(column)) for column in partition_columns)
        row_number = counters.get(partition_key, 0) + 1
        counters[partition_key] = row_number
        if row_number_matches(row_number, condition_cmp(op, "=="), int(op_value(op) if op_value(op) is not None else 1)):
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
