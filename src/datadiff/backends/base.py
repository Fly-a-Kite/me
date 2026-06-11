from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from datadiff.dsl import ColumnSpec, Program, TableData


@dataclass(slots=True)
class PreparedTable:
    name: str
    columns: list[ColumnSpec]
    rows: list[dict[str, Any]]
    column_names: tuple[str, ...]
    row_tuples: tuple[tuple[Any, ...], ...]
    columns_data: dict[str, list[Any]]


def prepare_table(table: TableData | PreparedTable) -> PreparedTable:
    if isinstance(table, PreparedTable):
        return table
    column_names = tuple(column.name for column in table.columns)
    row_tuples = tuple(
        tuple(row.get(column_name) for column_name in column_names)
        for row in table.rows
    )
    columns_data = {
        column_name: [row_values[index] for row_values in row_tuples]
        for index, column_name in enumerate(column_names)
    }
    return PreparedTable(
        name=table.name,
        columns=list(table.columns),
        rows=table.rows,
        column_names=column_names,
        row_tuples=row_tuples,
        columns_data=columns_data,
    )


def prepare_tables(tables: Sequence[TableData | PreparedTable]) -> list[PreparedTable]:
    return [prepare_table(table) for table in tables]


@dataclass(slots=True)
class BackendResult:
    backend: str
    status: str  # ok / error / timeout / missing
    data: Any = None
    error_type: str = ""
    error: str = ""
    duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "status": self.status,
            "data": self.data,
            "error_type": self.error_type,
            "error": self.error,
            "duration_ms": self.duration_ms,
        }

    def summary_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "status": self.status,
            "error_type": self.error_type,
            "error": self.error,
            "duration_ms": self.duration_ms,
        }


class Backend:
    name = "base"

    def run(
        self,
        tables: list[TableData | PreparedTable],
        program: Program,
        timeout_s: float = 5.0,
    ) -> BackendResult:
        raise NotImplementedError
