from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from datadiff.dsl import TableData
from datadiff.operation_semantics import op_kind, op_table
from datadiff.program_state import ProgramState, apply_operation_state


@dataclass(frozen=True, slots=True)
class TableSchema:
    name: str
    columns: tuple[str, ...]
    column_types: dict[str, str]
    nullable_columns: frozenset[str]

    @classmethod
    def from_table(cls, table: TableData) -> "TableSchema":
        return cls(
            name=table.name,
            columns=tuple(column.name for column in table.columns),
            column_types={column.name: column.type for column in table.columns},
            nullable_columns=frozenset(
                column.name
                for column in table.columns
                if column.nullable
                or any(row.get(column.name) is None for row in table.rows if column.name in row)
            ),
        )


@dataclass(frozen=True, slots=True)
class TypedProgramState:
    columns: tuple[str, ...]
    column_types: dict[str, str]
    nullable_columns: frozenset[str]
    available_tables: tuple[TableSchema, ...] = ()
    row_count_estimate: int = 0
    is_ordered: bool = False
    is_grouped: bool = False
    operations_so_far: tuple[str, ...] = ()
    joined_tables: frozenset[str] = frozenset()

    @classmethod
    def from_table(
        cls,
        table: TableData,
        *,
        extra_tables: Sequence[TableData] | None = None,
    ) -> "TypedProgramState":
        base = ProgramState.from_table(table)
        return cls(
            columns=tuple(base.columns),
            column_types=dict(base.column_types),
            nullable_columns=frozenset(base.nullable_columns),
            available_tables=tuple(TableSchema.from_table(table) for table in (extra_tables or ())),
            row_count_estimate=len(table.rows),
        )

    @property
    def available(self) -> set[str]:
        return set(self.columns)

    @property
    def numeric(self) -> tuple[str, ...]:
        return tuple(
            column
            for column in self.columns
            if self.column_types.get(column) in {"int", "float"}
        )

    @property
    def strings(self) -> tuple[str, ...]:
        return tuple(
            column
            for column in self.columns
            if self.column_types.get(column) == "str"
        )

    @property
    def booleans(self) -> tuple[str, ...]:
        return tuple(
            column
            for column in self.columns
            if self.column_types.get(column) == "bool"
        )

    @property
    def comparable(self) -> tuple[str, ...]:
        return tuple(
            column
            for column in self.columns
            if self.column_types.get(column) in {"int", "float", "str", "bool"}
        )

    def compatible_join_tables(self) -> tuple[TableSchema, ...]:
        out: list[TableSchema] = []
        for table in self.available_tables:
            if table.name in self.joined_tables:
                continue
            if _compatible_key_pairs(self, table):
                out.append(table)
        return tuple(out)

    def after_operation(
        self,
        operation: Mapping[str, Any],
        *,
        tables: Mapping[str, TableData] | None = None,
    ) -> "TypedProgramState":
        mutable = ProgramState(
            columns=list(self.columns),
            column_types=dict(self.column_types),
            nullable_columns=set(self.nullable_columns),
        )
        apply_operation_state(mutable, operation, tables=tables or {})
        kind = op_kind(operation)
        joined = set(self.joined_tables)
        if kind == "join":
            table_name = op_table(operation)
            if table_name:
                joined.add(table_name)
        row_count = _estimate_row_count(self.row_count_estimate, kind, operation)
        return TypedProgramState(
            columns=tuple(mutable.columns),
            column_types=dict(mutable.column_types),
            nullable_columns=frozenset(mutable.nullable_columns),
            available_tables=self.available_tables,
            row_count_estimate=row_count,
            is_ordered=kind in {"sort", "running_sum", "row_number_filter"}
            or (self.is_ordered and kind not in {"groupby", "aggregate", "join", "union_all"}),
            is_grouped=self.is_grouped or kind in {"groupby", "aggregate"},
            operations_so_far=(*self.operations_so_far, kind),
            joined_tables=frozenset(joined),
        )


def compatible_join_key_pairs(state: TypedProgramState, table: TableSchema) -> tuple[tuple[str, str], ...]:
    return tuple(_compatible_key_pairs(state, table))


def _compatible_key_pairs(state: TypedProgramState, table: TableSchema) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for left in state.columns:
        left_type = state.column_types.get(left)
        if left_type not in {"int", "str", "bool"}:
            continue
        for right in table.columns:
            if table.column_types.get(right) == left_type:
                pairs.append((left, right))
    return pairs


def _estimate_row_count(current: int, kind: str, operation: Mapping[str, Any]) -> int:
    rows = max(0, int(current))
    if kind in {"filter", "drop_nulls", "semi_join", "anti_join", "tuple_absence_filter"}:
        return max(0, rows // 2)
    if kind == "limit":
        try:
            return min(rows, max(0, int(operation.get("n", rows))))
        except (TypeError, ValueError):
            return rows
    if kind == "offset":
        try:
            return max(0, rows - max(0, int(operation.get("n", 0))))
        except (TypeError, ValueError):
            return rows
    if kind in {"groupby", "aggregate"}:
        return max(1, min(rows, max(1, rows // 3)))
    return rows
