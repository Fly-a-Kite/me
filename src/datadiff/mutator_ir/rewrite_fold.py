from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import Any

from datadiff.dsl import TableData
from datadiff.operation_semantics import op_columns, op_kind, op_n


def legal_redundant_fold_positions(
    tables: Sequence[TableData],
    operations: Sequence[Mapping[str, Any]],
) -> list[int]:
    del tables
    if len(operations) < 2:
        return []
    out: list[int] = []
    for index in range(len(operations) - 1):
        if _fold_pair(operations[index], operations[index + 1]) is not None:
            out.append(index)
    return out


def apply_redundant_op_fold(
    tables: list[TableData],
    operations: list[dict[str, Any]],
    rnd: random.Random,
) -> str:
    positions = legal_redundant_fold_positions(tables, operations)
    if not positions:
        return "ir_fold_redundant_op:no-legal-position"
    index = rnd.choice(positions)
    left_kind = op_kind(operations[index], "unknown")
    right_kind = op_kind(operations[index + 1], "unknown")
    merged = _fold_pair(operations[index], operations[index + 1])
    if merged is None:
        return "ir_fold_redundant_op:no-legal-position"
    operations[index : index + 2] = [merged]
    return f"ir_fold_redundant_op:{index}:{left_kind}+{right_kind}"


def _fold_pair(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any] | None:
    left_kind = op_kind(left)
    right_kind = op_kind(right)
    if left_kind != right_kind:
        return None
    if left_kind == "sort":
        return dict(right)
    if left_kind == "select":
        return _merge_select_ops(left, right)
    if left_kind == "limit":
        return {"op": "limit", "n": min(op_n(left, 0), op_n(right, 0))}
    if left_kind == "offset":
        return {"op": "offset", "n": op_n(left, 0) + op_n(right, 0)}
    if left_kind == "drop_nulls":
        columns = _unique([*op_columns(left), *op_columns(right)])
        return {"op": "drop_nulls", "columns": columns} if columns else None
    return None


def _merge_select_ops(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any] | None:
    left_columns = set(op_columns(left))
    right_columns = [column for column in op_columns(right) if column in left_columns]
    if not right_columns:
        return None
    return {"op": "select", "columns": right_columns}


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = str(value)
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out
