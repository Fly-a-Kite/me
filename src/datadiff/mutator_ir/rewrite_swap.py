from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import Any

from datadiff.dsl import TableData
from datadiff.operation_semantics import (
    condition_column,
    groupby_keys,
    op_column,
    op_columns,
    op_kind,
    op_output_alias,
)
from datadiff.program_state import ProgramState, apply_operation_state, state_after_operations

_LOCAL_REORDERABLE_KINDS = frozenset(
    {
        "filter",
        "drop_nulls",
        "sort",
        "limit",
        "offset",
        "fill_null",
    }
)
_BARRIER_KINDS = frozenset(
    {
        "join",
        "groupby",
        "aggregate",
        "union_all",
        "row_number_filter",
        "running_sum",
        "sortedness_check",
    }
)


def legal_adjacent_swap_positions(
    tables: Sequence[TableData],
    operations: Sequence[Mapping[str, Any]],
) -> list[int]:
    if not tables or len(operations) < 2:
        return []
    out: list[int] = []
    table_by_name = {table.name: table for table in tables}
    prefix_states = _prefix_states(tables, operations, table_by_name)
    for index in range(len(operations) - 1):
        if _can_swap_adjacent_in_state(prefix_states[index], table_by_name, operations, index):
            out.append(index)
    return out


def apply_adjacent_independent_swap(
    tables: list[TableData],
    operations: list[dict[str, Any]],
    rnd: random.Random,
) -> str:
    positions = legal_adjacent_swap_positions(tables, operations)
    if not positions:
        return "ir_swap_adjacent:no-legal-position"
    index = rnd.choice(positions)
    left_kind = op_kind(operations[index], "unknown")
    right_kind = op_kind(operations[index + 1], "unknown")
    operations[index], operations[index + 1] = operations[index + 1], operations[index]
    return f"ir_swap_adjacent:{index}:{left_kind}<->{right_kind}"


def _can_swap_adjacent(
    tables: Sequence[TableData],
    operations: Sequence[Mapping[str, Any]],
    index: int,
) -> bool:
    left = operations[index]
    right = operations[index + 1]
    left_kind = op_kind(left)
    right_kind = op_kind(right)
    if left_kind in _BARRIER_KINDS or right_kind in _BARRIER_KINDS:
        return False
    if left_kind not in _LOCAL_REORDERABLE_KINDS or right_kind not in _LOCAL_REORDERABLE_KINDS:
        return False
    before = state_after_operations(tables[0], operations[:index], extra_tables=tables[1:])
    return _can_swap_adjacent_in_state(before, {table.name: table for table in tables}, operations, index)


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


def _can_swap_adjacent_in_state(
    before: ProgramState,
    table_by_name: Mapping[str, TableData],
    operations: Sequence[Mapping[str, Any]],
    index: int,
) -> bool:
    left = operations[index]
    right = operations[index + 1]
    left_kind = op_kind(left)
    right_kind = op_kind(right)
    if left_kind in _BARRIER_KINDS or right_kind in _BARRIER_KINDS:
        return False
    if left_kind not in _LOCAL_REORDERABLE_KINDS or right_kind not in _LOCAL_REORDERABLE_KINDS:
        return False
    if not _operation_valid_in_state(left, before):
        return False
    after_left = before.copy()
    apply_operation_state(after_left, left, tables=table_by_name)
    if not _operation_valid_in_state(right, after_left):
        return False
    if _produced_columns(left) & _read_columns(right):
        return False
    if _produced_columns(right) & _read_columns(left):
        return False
    swapped = before.copy()
    if not _operation_valid_in_state(right, swapped):
        return False
    apply_operation_state(swapped, right, tables=table_by_name)
    if not _operation_valid_in_state(left, swapped):
        return False
    original_after = after_left.copy()
    apply_operation_state(original_after, right, tables=table_by_name)
    apply_operation_state(swapped, left, tables=table_by_name)
    return _state_signature(original_after) == _state_signature(swapped)


def _operation_valid_in_state(operation: Mapping[str, Any], state: ProgramState) -> bool:
    reads = _read_columns(operation)
    return reads.issubset(state.available)


def _read_columns(operation: Mapping[str, Any]) -> set[str]:
    kind = op_kind(operation)
    if kind in {"filter", "fill_null"}:
        return _nonempty_set(op_column(operation))
    if kind in {"select", "distinct", "drop_nulls"}:
        return set(op_columns(operation))
    if kind == "sort":
        keys = operation.get("keys")
        if isinstance(keys, list):
            return {column for key in keys if (column := _sort_key_column(key))}
        return set(op_columns(operation)) or _nonempty_set(op_column(operation))
    if kind == "case_when":
        return _nonempty_set(condition_column(operation))
    if kind == "groupby":
        return set(groupby_keys(operation))
    return set()


def _produced_columns(operation: Mapping[str, Any]) -> set[str]:
    kind = op_kind(operation)
    if kind in {"mutate", "running_sum"}:
        return _nonempty_set(op_column(operation))
    if kind in {"coalesce", "case_when", "sortedness_check"} or kind.endswith("_probe"):
        return _nonempty_set(op_output_alias(operation))
    return set()


def _nonempty_set(value: str) -> set[str]:
    return {str(value)} if str(value) else set()


def _sort_key_column(value: Any) -> str:
    if isinstance(value, Mapping):
        return str(value.get("column", "") or "")
    return str(getattr(value, "column", "") or "")


def _state_signature(state: ProgramState) -> tuple[tuple[str, ...], tuple[tuple[str, str], ...], tuple[str, ...]]:
    return (
        tuple(state.columns),
        tuple(sorted((str(key), str(value)) for key, value in state.column_types.items())),
        tuple(sorted(state.nullable_columns)),
    )
