from __future__ import annotations

import random
from typing import Any

from datadiff.dsl import TableData
from datadiff.operation_semantics import op_kind, op_n, op_columns


def shrink_drop_tail_op(
    tables: list[TableData],
    operations: list[dict[str, Any]],
    rnd: random.Random,
) -> str:
    del tables, rnd
    if len(operations) <= 1:
        return "shrink_drop_tail_op:none"
    removed = operations.pop()
    return f"shrink_drop_tail_op:{op_kind(removed, 'unknown')}"


def shrink_fold_redundant_op(
    tables: list[TableData],
    operations: list[dict[str, Any]],
    rnd: random.Random,
) -> str:
    del tables, rnd
    if len(operations) < 2:
        return "shrink_fold_redundant_op:none"
    for index in range(len(operations) - 1):
        left = operations[index]
        right = operations[index + 1]
        left_kind = op_kind(left)
        right_kind = op_kind(right)
        if left_kind == right_kind == "sort":
            removed = operations.pop(index)
            return f"shrink_fold_redundant_op:sort:{op_kind(removed, 'unknown')}"
        if left_kind == right_kind == "select":
            merged = _merge_select_ops(left, right)
            if merged is None:
                continue
            operations[index : index + 2] = [merged]
            return "shrink_fold_redundant_op:select"
        if left_kind == right_kind == "limit":
            operations[index : index + 2] = [{"op": "limit", "n": min(op_n(left, 0), op_n(right, 0))}]
            return "shrink_fold_redundant_op:limit"
        if left_kind == right_kind == "offset":
            operations[index : index + 2] = [{"op": "offset", "n": op_n(left, 0) + op_n(right, 0)}]
            return "shrink_fold_redundant_op:offset"
    return "shrink_fold_redundant_op:none"


def shrink_merge_adjacent_filters(
    tables: list[TableData],
    operations: list[dict[str, Any]],
    rnd: random.Random,
) -> str:
    del tables, rnd
    if len(operations) < 2:
        return "shrink_merge_adjacent_filters:none"
    for index in range(len(operations) - 1):
        left = operations[index]
        right = operations[index + 1]
        if op_kind(left) != "filter" or op_kind(right) != "filter":
            continue
        if left.get("column") != right.get("column") or left.get("cmp") != right.get("cmp"):
            continue
        if left.get("value") == right.get("value"):
            operations.pop(index + 1)
            return "shrink_merge_adjacent_filters:duplicate"
        if left.get("cmp") == "range_closed":
            merged = _merge_range_closed_filters(left, right)
            if merged is not None:
                operations[index : index + 2] = [merged]
                return "shrink_merge_adjacent_filters:range_closed"
    return "shrink_merge_adjacent_filters:none"


def shrink_inline_single_use_mutate(
    tables: list[TableData],
    operations: list[dict[str, Any]],
    rnd: random.Random,
) -> str:
    del tables, rnd
    for index, operation in enumerate(operations):
        if op_kind(operation) != "mutate":
            continue
        alias = str(operation.get("column", "") or "")
        expr = operation.get("expr")
        if not alias or not isinstance(expr, dict):
            continue
        source = str(expr.get("source", "") or "")
        if not source or _operation_reference_count(operations[index + 1 :], alias) != 1:
            continue
        if not _replace_operation_reference(operations[index + 1 :], alias, source):
            continue
        operations.pop(index)
        return f"shrink_inline_single_use_mutate:{alias}->{source}"
    return "shrink_inline_single_use_mutate:none"


def _merge_select_ops(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any] | None:
    left_columns = set(op_columns(left))
    right_columns = [column for column in op_columns(right) if column in left_columns]
    if not right_columns:
        return None
    return {"op": "select", "columns": right_columns}


def _merge_range_closed_filters(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any] | None:
    left_value = left.get("value")
    right_value = right.get("value")
    if not (
        isinstance(left_value, list)
        and isinstance(right_value, list)
        and len(left_value) >= 2
        and len(right_value) >= 2
    ):
        return None
    lower = max(left_value[0], right_value[0])
    upper = min(left_value[1], right_value[1])
    if lower > upper:
        return None
    return {"op": "filter", "column": left.get("column"), "cmp": "range_closed", "value": [lower, upper]}


def _operation_reference_count(operations: list[dict[str, Any]], column: str) -> int:
    return sum(_count_references(operation, column) for operation in operations)


def _count_references(value: Any, column: str) -> int:
    if isinstance(value, dict):
        count = 0
        for key, item in value.items():
            if key in {"column", "source"} and item == column:
                count += 1
            if key in {"columns", "partition_by"} and isinstance(item, list):
                count += sum(1 for entry in item if entry == column)
            count += _count_references(item, column)
        return count
    if isinstance(value, list):
        return sum(_count_references(item, column) for item in value)
    return 0


def _replace_operation_reference(operations: list[dict[str, Any]], old: str, new: str) -> bool:
    changed = False
    for operation in operations:
        changed = _replace_reference(operation, old, new) or changed
    return changed


def _replace_reference(value: Any, old: str, new: str) -> bool:
    if isinstance(value, dict):
        changed = False
        for key, item in list(value.items()):
            if key in {"column", "source"} and item == old:
                value[key] = new
                changed = True
            elif key in {"columns", "partition_by"} and isinstance(item, list):
                for index, entry in enumerate(item):
                    if entry == old:
                        item[index] = new
                        changed = True
            else:
                changed = _replace_reference(item, old, new) or changed
        return changed
    if isinstance(value, list):
        changed = False
        for item in value:
            changed = _replace_reference(item, old, new) or changed
        return changed
    return False
