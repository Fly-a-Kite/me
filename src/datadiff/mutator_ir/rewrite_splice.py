from __future__ import annotations

import copy
import random
from collections.abc import Mapping, Sequence
from typing import Any

from datadiff.dsl import TableData
from datadiff.operation_semantics import (
    condition_column,
    op_column,
    op_columns,
    op_kind,
)
from datadiff.program_state import ProgramState, apply_operation_state, state_after_operations

_SPLICEABLE_KINDS = frozenset(
    {
        "filter",
        "drop_nulls",
        "fill_null",
        "sort",
        "limit",
        "offset",
        "select",
        "distinct",
    }
)


def legal_subtree_splice_positions(
    tables: Sequence[TableData],
    operations: Sequence[Mapping[str, Any]],
) -> list[tuple[int, int, int]]:
    if not tables or not operations:
        return []
    out: list[tuple[int, int, int]] = []
    table_by_name = {table.name: table for table in tables}
    prefix_states = _prefix_states(tables, operations, table_by_name)
    for start in range(len(operations)):
        for width in range(1, min(3, len(operations) - start) + 1):
            subtree = operations[start : start + width]
            if not _spliceable_subtree(subtree):
                continue
            for insert_at in range(len(operations) + 1):
                if start <= insert_at <= start + width:
                    continue
                if _subtree_valid_in_state(prefix_states[insert_at], table_by_name, subtree):
                    out.append((start, width, insert_at))
    return out


def apply_subtree_splice(
    tables: list[TableData],
    operations: list[dict[str, Any]],
    rnd: random.Random,
) -> str:
    positions = legal_subtree_splice_positions(tables, operations)
    if not positions:
        return "ir_splice_subtree:no-legal-position"
    start, width, insert_at = rnd.choice(positions)
    subtree = [copy.deepcopy(operation) for operation in operations[start : start + width]]
    adjusted_insert_at = insert_at
    if insert_at > start:
        adjusted_insert_at -= width
    del operations[start : start + width]
    operations[adjusted_insert_at:adjusted_insert_at] = subtree
    kinds = ",".join(op_kind(operation, "unknown") for operation in subtree)
    return f"ir_splice_subtree:{start}:{width}->{adjusted_insert_at}:{kinds}"


def _spliceable_subtree(subtree: Sequence[Mapping[str, Any]]) -> bool:
    return bool(subtree) and all(op_kind(operation) in _SPLICEABLE_KINDS for operation in subtree)


def _subtree_valid_at(
    tables: Sequence[TableData],
    operations: Sequence[Mapping[str, Any]],
    subtree: Sequence[Mapping[str, Any]],
    insert_at: int,
) -> bool:
    if not tables:
        return False
    table_by_name = {table.name: table for table in tables}
    before = state_after_operations(tables[0], operations[:insert_at], extra_tables=tables[1:])
    return _subtree_valid_in_state(before, table_by_name, subtree)


def _prefix_states(
    tables: Sequence[TableData],
    operations: Sequence[Mapping[str, Any]],
    table_by_name: Mapping[str, TableData],
) -> list[ProgramState]:
    state = ProgramState.from_table(tables[0])
    states = [state.copy()]
    for operation in operations:
        apply_operation_state(state, operation, tables=table_by_name)
        states.append(state.copy())
    return states


def _subtree_valid_in_state(
    before: ProgramState,
    table_by_name: Mapping[str, TableData],
    subtree: Sequence[Mapping[str, Any]],
) -> bool:
    trial = before.copy()
    for operation in subtree:
        if not _operation_valid_in_state(operation, trial):
            return False
        apply_operation_state(trial, operation, tables=table_by_name)
    return True


def _operation_valid_in_state(operation: Mapping[str, Any], state: ProgramState) -> bool:
    reads = _read_columns(operation)
    if reads and not reads.issubset(state.available):
        return False
    kind = op_kind(operation)
    if kind in {"select", "distinct"}:
        return bool(op_columns(operation))
    if kind == "fill_null":
        return bool(op_column(operation))
    if kind == "sort":
        return bool(_sort_columns(operation)) or bool(state.columns)
    return True


def _read_columns(operation: Mapping[str, Any]) -> set[str]:
    kind = op_kind(operation)
    if kind == "filter":
        return _nonempty_set(condition_column(operation, op_column(operation)))
    if kind in {"select", "distinct", "drop_nulls"}:
        return set(op_columns(operation))
    if kind in {"fill_null"}:
        return _nonempty_set(op_column(operation))
    if kind == "sort":
        return _sort_columns(operation)
    return set()


def _sort_columns(operation: Mapping[str, Any]) -> set[str]:
    keys = operation.get("keys")
    if isinstance(keys, list):
        return {column for key in keys if (column := _sort_key_column(key))}
    columns = set(op_columns(operation))
    column = op_column(operation)
    if column:
        columns.add(column)
    return columns


def _sort_key_column(value: Any) -> str:
    if isinstance(value, Mapping):
        return str(value.get("column", "") or "")
    return str(getattr(value, "column", "") or "")


def _nonempty_set(value: str) -> set[str]:
    return {str(value)} if str(value) else set()
