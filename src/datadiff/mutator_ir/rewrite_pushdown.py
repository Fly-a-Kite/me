from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import Any

from datadiff.dsl import TableData
from datadiff.expression_semantics import expr_output_type
from datadiff.join_keys import join_key_pairs
from datadiff.operation_semantics import (
    condition_column,
    expr_numerator,
    expr_other,
    expr_source,
    join_how,
    op_column,
    op_columns,
    op_kind,
    op_output_alias,
    op_table,
)
from datadiff.program_state import ProgramState, apply_operation_state, state_after_operations

_FILTER_PUSHDOWN_THROUGH_KINDS = frozenset(
    {
        "join",
        "semi_join",
        "anti_join",
        "mutate",
        "select",
        "distinct",
        "sort",
    }
)


def legal_filter_pushdown_positions(
    tables: Sequence[TableData],
    operations: Sequence[Mapping[str, Any]],
) -> list[int]:
    if not tables or len(operations) < 2:
        return []
    out: list[int] = []
    for index in range(1, len(operations)):
        if _can_push_filter_left(tables, operations, index):
            out.append(index)
    return out


def apply_filter_pushdown(
    tables: list[TableData],
    operations: list[dict[str, Any]],
    rnd: random.Random,
) -> str:
    positions = legal_filter_pushdown_positions(tables, operations)
    if not positions:
        return "ir_pushdown_filter:no-legal-position"
    index = rnd.choice(positions)
    through_kind = op_kind(operations[index - 1], "unknown")
    column = condition_column(operations[index], op_column(operations[index], "unknown"))
    operations[index - 1], operations[index] = operations[index], operations[index - 1]
    return f"ir_pushdown_filter:{index}:{through_kind}:filter[{column}]"


def _can_push_filter_left(
    tables: Sequence[TableData],
    operations: Sequence[Mapping[str, Any]],
    filter_index: int,
) -> bool:
    filter_op = operations[filter_index]
    through_op = operations[filter_index - 1]
    if op_kind(filter_op) != "filter":
        return False
    if op_kind(through_op) not in _FILTER_PUSHDOWN_THROUGH_KINDS:
        return False
    table_by_name = {table.name: table for table in tables}
    before = state_after_operations(tables[0], operations[: filter_index - 1], extra_tables=tables[1:])
    filter_reads = _read_columns(filter_op)
    if not filter_reads or not filter_reads.issubset(before.available):
        return False
    if not _operation_valid_in_state(through_op, before, table_by_name):
        return False
    original = before.copy()
    apply_operation_state(original, through_op, tables=table_by_name)
    if not _operation_valid_in_state(filter_op, original, table_by_name):
        return False
    apply_operation_state(original, filter_op, tables=table_by_name)
    swapped = before.copy()
    if not _operation_valid_in_state(filter_op, swapped, table_by_name):
        return False
    apply_operation_state(swapped, filter_op, tables=table_by_name)
    if not _operation_valid_in_state(through_op, swapped, table_by_name):
        return False
    apply_operation_state(swapped, through_op, tables=table_by_name)
    return _state_signature(original) == _state_signature(swapped)


def _operation_valid_in_state(
    operation: Mapping[str, Any],
    state: ProgramState,
    tables: Mapping[str, TableData],
) -> bool:
    kind = op_kind(operation)
    reads = _read_columns(operation)
    if reads and not reads.issubset(state.available):
        return False
    if kind in {"select", "distinct"} and not op_columns(operation):
        return False
    if kind == "mutate":
        column = op_column(operation)
        if not column:
            return False
        return expr_output_type(operation.get("expr", {}), state.column_types) is not None
    if kind in {"join", "semi_join", "anti_join"}:
        return _join_valid_in_state(operation, state, tables)
    return True


def _join_valid_in_state(
    operation: Mapping[str, Any],
    state: ProgramState,
    tables: Mapping[str, TableData],
) -> bool:
    right = tables.get(op_table(operation))
    if right is None:
        return False
    left_keys, right_keys = join_key_pairs(operation)
    if not left_keys or len(left_keys) != len(right_keys):
        return False
    if len(set(left_keys)) != len(left_keys) or len(set(right_keys)) != len(right_keys):
        return False
    if any(left_key not in state.available for left_key in left_keys):
        return False
    right_types = {column.name: column.type for column in right.columns}
    if any(right_key not in right_types for right_key in right_keys):
        return False
    if any(state.column_types.get(left_key) != right_types.get(right_key) for left_key, right_key in zip(left_keys, right_keys)):
        return False
    if op_kind(operation) == "join" and join_how(operation) not in {"inner", "left"}:
        return False
    return True


def _read_columns(operation: Mapping[str, Any]) -> set[str]:
    kind = op_kind(operation)
    if kind == "filter":
        return _nonempty_set(condition_column(operation, op_column(operation)))
    if kind in {"select", "distinct", "drop_nulls"}:
        return set(op_columns(operation))
    if kind == "sort":
        return _sort_columns(operation)
    if kind == "mutate":
        return _expression_read_columns(operation)
    if kind in {"join", "semi_join", "anti_join"}:
        left_keys, _ = join_key_pairs(operation)
        return set(left_keys)
    return set()


def _expression_read_columns(operation: Mapping[str, Any]) -> set[str]:
    columns = _nonempty_set(expr_source(operation))
    columns |= _nonempty_set(expr_other(operation))
    columns |= _nonempty_set(expr_numerator(operation))
    return columns


def _sort_columns(operation: Mapping[str, Any]) -> set[str]:
    keys = operation.get("keys")
    if isinstance(keys, list):
        return {column for key in keys if (column := _sort_key_column(key))}
    columns = set(op_columns(operation))
    column = op_column(operation)
    if column:
        columns.add(column)
    return columns


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
