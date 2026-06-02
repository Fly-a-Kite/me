from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from datadiff.dsl import Case, TableData
from datadiff.expression_semantics import expr_output_type
from datadiff.join_keys import join_key_pairs
from datadiff.operation_semantics import (
    aggregate_alias,
    aggregate_column,
    aggregate_func,
    aggregate_specs,
    case_else_value,
    case_then_value,
    coalesce_fallback,
    coalesce_has_fallback,
    coalesce_sources,
    groupby_keys,
    join_how,
    op_column,
    op_columns,
    op_kind,
    op_output_alias,
    op_table,
)
from datadiff.operation_type_semantics import aggregate_result_type, case_when_output_type
from datadiff.util import unique_preserve_order


@dataclass(slots=True)
class ProgramState:
    columns: list[str]
    column_types: dict[str, str]
    nullable_columns: set[str]

    @classmethod
    def from_table(cls, table: TableData) -> "ProgramState":
        return cls(
            columns=[column.name for column in table.columns],
            column_types={column.name: column.type for column in table.columns},
            nullable_columns={
                column.name
                for column in table.columns
                if column.nullable or any(row.get(column.name) is None for row in table.rows if column.name in row)
            },
        )

    def copy(self) -> "ProgramState":
        return ProgramState(list(self.columns), dict(self.column_types), set(self.nullable_columns))

    @property
    def available(self) -> set[str]:
        return set(self.columns)

    @property
    def numeric(self) -> set[str]:
        return {
            name
            for name in self.columns
            if self.column_types.get(name) in {"int", "float"}
        }

    @property
    def strings(self) -> set[str]:
        return {
            name
            for name in self.columns
            if self.column_types.get(name) == "str"
        }

    @property
    def nullable(self) -> set[str]:
        return {name for name in self.columns if name in self.nullable_columns}

    def replace_projection(self, columns: Sequence[str]) -> None:
        projected = [column for column in unique_preserve_order(columns) if column in self.available]
        self.columns = projected
        self.column_types = {
            column: self.column_types[column]
            for column in projected
            if column in self.column_types
        }
        self.nullable_columns &= set(projected)

    def upsert_column(self, column: str, column_type: str | None = None) -> None:
        self.upsert_column_with_nullability(column, column_type, nullable=True)

    def upsert_column_with_nullability(
        self,
        column: str,
        column_type: str | None = None,
        *,
        nullable: bool = True,
    ) -> None:
        if not column:
            return
        self.columns = [existing for existing in self.columns if existing != column] + [column]
        if column_type is None:
            self.column_types.pop(column, None)
        else:
            self.column_types[column] = column_type
        if nullable:
            self.nullable_columns.add(column)
        else:
            self.nullable_columns.discard(column)

    def collapse_to_alias(self, alias: str, column_type: str = "bool") -> None:
        self.columns = [alias] if alias else []
        self.column_types = {alias: column_type} if alias else {}
        self.nullable_columns = set()


def state_after_operations(
    primary_table: TableData,
    operations: Sequence[Any],
    *,
    extra_tables: Sequence[TableData] | None = None,
) -> ProgramState:
    state = ProgramState.from_table(primary_table)
    tables = {table.name: table for table in [primary_table, *(extra_tables or [])]}
    for operation in operations:
        apply_operation_state(state, operation, tables=tables)
    return state


def state_before_operation(case: Case, op_index: int) -> ProgramState:
    if not case.tables:
        return ProgramState([], {}, set())
    return state_after_operations(
        case.tables[0],
        case.program.operations[:op_index],
        extra_tables=case.tables[1:],
    )


def state_before_first_operation(case: Case, kind: str) -> ProgramState:
    for index, operation in enumerate(case.program.operations):
        if op_kind(operation) == kind:
            return state_before_operation(case, index)
    if not case.tables:
        return ProgramState([], {}, set())
    return ProgramState.from_table(case.tables[0])


def apply_operation_state(
    state: ProgramState,
    operation: Any,
    *,
    tables: Mapping[str, TableData] | None = None,
) -> ProgramState:
    kind = op_kind(operation)
    if kind == "join":
        _apply_join_state(state, operation, tables or {})
    elif kind in {"select", "distinct"}:
        state.replace_projection(op_columns(operation))
    elif kind == "mutate":
        out_type = expr_output_type(operation.get("expr", {}), state.column_types)
        state.upsert_column_with_nullability(op_column(operation), out_type, nullable=True)
    elif kind == "running_sum":
        state.upsert_column_with_nullability(op_column(operation), "float", nullable=True)
    elif kind == "fill_null":
        state.nullable_columns.discard(op_column(operation))
    elif kind == "coalesce":
        alias = op_output_alias(operation)
        output_type = _coalesce_output_type(state, operation)
        state.upsert_column_with_nullability(
            alias,
            output_type,
            nullable=_coalesce_output_nullable(state, operation),
        )
    elif kind == "case_when":
        alias = op_output_alias(operation)
        output_type = case_when_output_type(
            case_then_value(operation),
            case_else_value(operation),
        )
        state.upsert_column_with_nullability(alias, output_type, nullable=True)
    elif kind == "groupby":
        _apply_groupby_state(state, operation)
    elif kind == "aggregate":
        _apply_aggregate_state(state, operation)
    elif kind == "sortedness_check" or kind.endswith("_probe"):
        state.collapse_to_alias(op_output_alias(operation), "bool")
    return state


def _apply_join_state(
    state: ProgramState,
    operation: Any,
    tables: Mapping[str, TableData],
) -> None:
    right = tables.get(op_table(operation))
    if right is None:
        return
    _, right_keys = join_key_pairs(operation)
    right_key_set = set(right_keys)
    for column in right.columns:
        if column.name in right_key_set or column.name in state.available:
            continue
        state.columns.append(column.name)
        state.column_types[column.name] = column.type
        if column.nullable or join_how(operation) == "left":
            state.nullable_columns.add(column.name)


def _apply_groupby_state(state: ProgramState, operation: Any) -> None:
    previous_types = dict(state.column_types)
    previous_nullable = set(state.nullable_columns)
    keys = [key for key in groupby_keys(operation) if key in state.available]
    next_columns = list(keys)
    next_types = {
        key: previous_types[key]
        for key in keys
        if key in previous_types
    }
    next_nullable = {
        key
        for key in keys
        if key in previous_nullable
    }
    for aggregate in aggregate_specs(operation):
        alias = aggregate_alias(aggregate)
        if not alias:
            continue
        next_columns.append(alias)
        next_types[alias] = aggregate_result_type(
            previous_types.get(aggregate_column(aggregate)),
            aggregate_func(aggregate),
        )
        next_nullable.add(alias)
    state.columns = unique_preserve_order(next_columns)
    state.column_types = next_types
    state.nullable_columns = next_nullable


def _apply_aggregate_state(state: ProgramState, operation: Any) -> None:
    previous_types = dict(state.column_types)
    next_columns: list[str] = []
    next_types: dict[str, str] = {}
    next_nullable: set[str] = set()
    for aggregate in aggregate_specs(operation):
        alias = aggregate_alias(aggregate)
        if not alias:
            continue
        next_columns.append(alias)
        next_types[alias] = aggregate_result_type(
            previous_types.get(aggregate_column(aggregate)),
            aggregate_func(aggregate),
        )
        next_nullable.add(alias)
    state.columns = unique_preserve_order(next_columns)
    state.column_types = next_types
    state.nullable_columns = next_nullable


def _coalesce_output_type(state: ProgramState, operation: Any) -> str | None:
    source_columns = [column for column in coalesce_sources(operation) if column in state.column_types]
    if not source_columns:
        return None
    first_type = state.column_types.get(source_columns[0])
    if first_type is None:
        return None
    if any(state.column_types.get(column) != first_type for column in source_columns):
        return None
    return first_type


def _coalesce_output_nullable(state: ProgramState, operation: Any) -> bool:
    source_columns = [column for column in coalesce_sources(operation) if column in state.column_types]
    if not source_columns:
        return True
    if coalesce_has_fallback(operation) and coalesce_fallback(operation) is not None:
        return False
    return all(column in state.nullable_columns for column in source_columns)
