from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import Any

from datadiff.dsl import TableData
from datadiff.operation_semantics import (
    aggregate_alias,
    aggregate_column,
    aggregate_specs,
    condition_column,
    condition_cmp,
    condition_value,
    groupby_keys,
    op_column,
    op_kind,
)
from datadiff.program_state import ProgramState, apply_operation_state


def legal_filter_above_groupby_positions(
    tables: Sequence[TableData],
    operations: Sequence[Mapping[str, Any]],
) -> list[int]:
    if not tables or len(operations) < 2:
        return []
    out: list[int] = []
    for index in range(len(operations) - 1):
        if _pulled_filter(operations[index], operations[index + 1]) is not None:
            out.append(index)
    return out


def apply_filter_above_groupby(
    tables: list[TableData],
    operations: list[dict[str, Any]],
    rnd: random.Random,
) -> str:
    positions = legal_filter_above_groupby_positions(tables, operations)
    if not positions:
        return "ir_pull_filter_above_groupby:no-legal-position"
    index = rnd.choice(positions)
    pulled = _pulled_filter(operations[index], operations[index + 1])
    if pulled is None:
        return "ir_pull_filter_above_groupby:no-legal-position"
    source_column, filter_op = pulled
    operations[index : index + 2] = [filter_op, operations[index]]
    return f"ir_pull_filter_above_groupby:{index}:{source_column}"


def _pulled_filter(
    groupby_op: Mapping[str, Any],
    filter_op: Mapping[str, Any],
) -> tuple[str, dict[str, Any]] | None:
    if op_kind(groupby_op) != "groupby" or op_kind(filter_op) != "filter":
        return None
    filter_column = condition_column(filter_op, op_column(filter_op))
    if not filter_column:
        return None
    key_set = set(groupby_keys(groupby_op))
    if filter_column in key_set:
        return None
    source_column = ""
    for aggregate in aggregate_specs(groupby_op):
        if aggregate_alias(aggregate) == filter_column:
            source_column = aggregate_column(aggregate)
            break
    if not source_column or source_column in key_set:
        return None
    comparator = condition_cmp(filter_op)
    value = condition_value(filter_op)
    return source_column, {"op": "filter", "column": source_column, "cmp": comparator, "value": value}


def legal_window_wrap_positions(
    tables: Sequence[TableData],
    operations: Sequence[Mapping[str, Any]],
) -> list[int]:
    return [insert_after for insert_after, _ in _legal_window_wrap_candidates(tables, operations)]


def apply_wrap_with_window(
    tables: list[TableData],
    operations: list[dict[str, Any]],
    rnd: random.Random,
) -> str:
    candidates = _legal_window_wrap_candidates(tables, operations)
    if not candidates:
        return "ir_wrap_with_window:no-legal-position"
    insert_after, state = rnd.choice(candidates)
    window_op = _window_op_for_state(state, existing_columns=set(state.available), rnd=rnd)
    if window_op is None:
        return "ir_wrap_with_window:no-legal-position"
    operations.insert(insert_after + 1, window_op)
    return f"ir_wrap_with_window:{insert_after + 1}:{op_kind(window_op)}"


def _legal_window_wrap_candidates(
    tables: Sequence[TableData],
    operations: Sequence[Mapping[str, Any]],
) -> list[tuple[int, ProgramState]]:
    if not tables:
        return []
    states = _prefix_states(tables, operations)
    out: list[tuple[int, ProgramState]] = []
    for insert_after in range(-1, len(operations)):
        state = states[insert_after + 1]
        if _window_op_for_state(state, existing_columns=set(state.available)) is not None:
            out.append((insert_after, state))
    return out


def _prefix_states(
    tables: Sequence[TableData],
    operations: Sequence[Mapping[str, Any]],
) -> list[ProgramState]:
    table_by_name = {table.name: table for table in tables}
    state = ProgramState.from_table(tables[0])
    states = [state.copy()]
    for operation in operations:
        apply_operation_state(state, operation, tables=table_by_name)
        states.append(state.copy())
    return states


def _window_op_for_state(
    state: Any,
    *,
    existing_columns: set[str],
    rnd: random.Random | None = None,
) -> dict[str, Any] | None:
    columns = list(state.columns)
    if not columns:
        return None
    numeric = [column for column in columns if state.column_types.get(column) in {"int", "float"}]
    comparable = [
        column
        for column in columns
        if state.column_types.get(column) in {"int", "float", "str", "bool"}
    ]
    if numeric:
        source = _choose(rnd, numeric)
        order_columns = _window_order_columns(rnd, comparable or columns, source)
        if not order_columns:
            return None
        output_column = _safe_window_alias(f"run_{source}", existing_columns)
        op: dict[str, Any] = {
            "op": "running_sum",
            "source": source,
            "column": output_column,
            "order_by": [_sort_key(column, rnd) for column in order_columns],
            "input_dtype": "float64",
        }
        partition_candidates = [
            column
            for column in columns
            if column != source and state.column_types.get(column) in {"int", "str", "bool"}
        ]
        if partition_candidates:
            op["partition_by"] = [_choose(rnd, partition_candidates)]
        return op
    if comparable:
        order_columns = _window_order_columns(rnd, comparable, comparable[0])
        if not order_columns:
            return None
        partition_candidates = [
            column
            for column in comparable
            if column != order_columns[0] and state.column_types.get(column) in {"int", "str", "bool"}
        ]
        op = {
            "op": "row_number_filter",
            "partition_by": [_choose(rnd, partition_candidates)] if partition_candidates else [],
            "order_by": [_sort_key(column, rnd) for column in order_columns],
            "cmp": "<=",
            "value": 2,
        }
        return op
    return None


def _window_order_columns(rnd: random.Random | None, candidates: Sequence[str], preferred: str) -> list[str]:
    ordered = [preferred, *[column for column in candidates if column != preferred]]
    if rnd is None:
        return ordered[: min(3, len(ordered))]
    width = 1 if len(ordered) == 1 else rnd.randint(1, min(3, len(ordered)))
    head = ordered[0]
    tail = ordered[1:]
    return [head, *rnd.sample(tail, k=max(0, width - 1))]


def _sort_key(column: str, rnd: random.Random | None) -> dict[str, Any]:
    return {
        "column": column,
        "ascending": True if rnd is None else rnd.choice([True, False]),
        "nulls": "last" if rnd is None else rnd.choice(["first", "last"]),
    }


def _safe_window_alias(prefix: str, used: set[str]) -> str:
    candidate = prefix
    suffix = 0
    while candidate in used:
        suffix += 1
        candidate = f"{prefix}_{suffix}"
    return candidate


def _choose(rnd: random.Random | None, values: Sequence[str]) -> str:
    return values[0] if rnd is None else rnd.choice(list(values))
