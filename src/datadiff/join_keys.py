from __future__ import annotations

from typing import Any

from datadiff.operation_semantics import join_left_keys, join_right_keys


def join_key_columns(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item)]
    column = str(value or "")
    return [column] if column else []


def join_key_pairs(op: dict[str, Any] | Any) -> tuple[list[str], list[str]]:
    return join_left_keys(op), join_right_keys(op)


def join_key_arg(columns: list[str]) -> str | list[str]:
    return columns[0] if len(columns) == 1 else list(columns)


def join_key_value(row: dict[str, Any], columns: list[str]) -> tuple[Any, ...] | None:
    values = tuple(row.get(column) for column in columns)
    if any(value is None for value in values):
        return None
    return values
