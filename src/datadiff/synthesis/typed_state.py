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
    structural_risk_tags: frozenset[str] = frozenset()
    coverage_axes: frozenset[str] = frozenset()
    expandability_score: float = 0.0
    validity_score: float = 1.0

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
            structural_risk_tags=frozenset({"base_table"}),
            coverage_axes=frozenset({"shape:base_table", "shape:row_preserving"}),
            expandability_score=0.35,
            validity_score=1.0,
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
        structural_risk_tags = set(self.structural_risk_tags)
        coverage_axes = set(self.coverage_axes)
        structural_risk_tags.update(_structural_risk_tags_for_operation(kind, operation))
        coverage_axes.update(_coverage_axes_for_operation(kind, operation))
        expandability = _next_expandability_score(self.expandability_score, kind)
        validity = _next_validity_score(self.validity_score, kind)
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
            structural_risk_tags=frozenset(structural_risk_tags),
            coverage_axes=frozenset(coverage_axes),
            expandability_score=expandability,
            validity_score=validity,
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


def _structural_risk_tags_for_operation(kind: str, operation: Mapping[str, Any]) -> tuple[str, ...]:
    tags: list[str] = []
    if kind == "join":
        tags.extend(("join_cardinality", "join_null_semantics"))
    if kind in {"groupby", "aggregate"}:
        tags.append("aggregation_boundary")
    if kind in {"sort", "running_sum", "row_number_filter", "limit", "offset"}:
        tags.append("ordering_boundary")
    if kind in {"filter", "drop_nulls"}:
        tags.append("predicate_boundary")
    if kind in {"mutate", "coalesce", "case_when", "fill_null"}:
        tags.append("expression_boundary")
    return tuple(tags)


def _coverage_axes_for_operation(kind: str, operation: Mapping[str, Any]) -> tuple[str, ...]:
    axes = [f"op:{kind}"]
    if kind == "join":
        axes.append("shape:multi_table")
    if kind in {"groupby", "aggregate"}:
        axes.append("shape:grouped")
    if kind in {"sort", "running_sum", "row_number_filter"}:
        axes.append("shape:ordered")
    if kind in {"filter", "drop_nulls", "semi_join", "anti_join"}:
        axes.append("shape:row_reducing")
    if kind in {"mutate", "coalesce", "case_when", "fill_null"}:
        axes.append("shape:column_expanding")
    return tuple(axes)


def _next_expandability_score(current: float, kind: str) -> float:
    score = float(current)
    if kind in {"mutate", "coalesce", "case_when", "join"}:
        score += 0.12
    elif kind in {"groupby", "aggregate", "select", "distinct"}:
        score -= 0.10
    elif kind in {"limit", "offset"}:
        score -= 0.08
    return max(0.0, min(1.0, score))


def _next_validity_score(current: float, kind: str) -> float:
    score = float(current)
    if kind == "join":
        score -= 0.08
    elif kind in {"groupby", "aggregate"}:
        score -= 0.06
    elif kind in {"mutate", "case_when", "coalesce"}:
        score -= 0.03
    return max(0.35, min(1.0, score))
