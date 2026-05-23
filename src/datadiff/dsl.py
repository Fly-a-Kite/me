from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

ColumnType = Literal["int", "float", "bool", "str"]
SortNulls = Literal["first", "last"]
Value = Any


@dataclass(slots=True)
class ColumnSpec:
    name: str
    type: ColumnType
    nullable: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ColumnSpec":
        return cls(**data)


@dataclass(slots=True)
class TableData:
    name: str
    columns: list[ColumnSpec]
    rows: list[dict[str, Value]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "columns": [c.to_dict() for c in self.columns],
            "rows": self.rows,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TableData":
        return cls(
            name=data["name"],
            columns=[ColumnSpec.from_dict(c) for c in data["columns"]],
            rows=data["rows"],
        )

    def column_type(self, name: str) -> ColumnType:
        for c in self.columns:
            if c.name == name:
                return c.type
        raise KeyError(name)

    def numeric_columns(self) -> list[str]:
        return [c.name for c in self.columns if c.type in {"int", "float"}]

    def comparable_columns(self) -> list[str]:
        return [c.name for c in self.columns if c.type in {"int", "float", "str", "bool"}]


@dataclass(slots=True)
class Program:
    program_id: str
    seed: int
    operations: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Program":
        return cls(**data)

    @property
    def order_sensitive(self) -> bool:
        row_order_preserving = {
            "filter",
            "tuple_absence_filter",
            "running_sum",
            "select",
            "mutate",
            "limit",
            "offset",
        }
        for op in reversed(self.operations):
            kind = op.get("op")
            if kind in {"sort", "running_sum"}:
                return True
            if kind in row_order_preserving:
                continue
            return False
        return False

    def op_sequence(self) -> list[str]:
        return [str(op.get("op", "unknown")) for op in self.operations]


@dataclass(frozen=True, slots=True)
class SortKey:
    column: str
    ascending: bool = True
    nulls: SortNulls = "last"

    def to_dict(self) -> dict[str, Any]:
        return {"column": self.column, "ascending": self.ascending, "nulls": self.nulls}


def normalize_sort_keys(op: dict[str, Any]) -> list[SortKey]:
    """Return the canonical per-column sort keys for old and new sort ops."""

    if "keys" in op:
        keys = []
        for item in op.get("keys", []):
            if not isinstance(item, dict):
                raise ValueError(f"sort key must be a mapping: {item!r}")
            column = item.get("column")
            ascending = item.get("ascending", True)
            nulls = item.get("nulls", "last")
            if not isinstance(column, str) or not column:
                raise ValueError(f"sort key column must be a non-empty string: {column!r}")
            if not isinstance(ascending, bool):
                raise ValueError(f"sort key ascending must be a boolean: {ascending!r}")
            if nulls not in {"first", "last"}:
                raise ValueError(f"sort key nulls must be 'first' or 'last': {nulls!r}")
            keys.append(SortKey(column=column, ascending=ascending, nulls=nulls))
        return keys

    columns = op.get("columns", [])
    ascending = op.get("ascending", True)
    if not isinstance(ascending, bool):
        raise ValueError(f"sort ascending must be a boolean: {ascending!r}")
    keys = []
    for column in columns:
        if not isinstance(column, str) or not column:
            raise ValueError(f"sort column must be a non-empty string: {column!r}")
        keys.append(SortKey(column=column, ascending=ascending, nulls="last"))
    return keys


def sort_columns(op: dict[str, Any]) -> list[str]:
    return [key.column for key in normalize_sort_keys(op)]


@dataclass(slots=True)
class Case:
    case_id: str
    seed: int
    tables: list[TableData]
    program: Program
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = {
            "case_id": self.case_id,
            "seed": self.seed,
            "tables": [t.to_dict() for t in self.tables],
            "program": self.program.to_dict(),
        }
        if self.metadata:
            data["metadata"] = self.metadata
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Case":
        return cls(
            case_id=data["case_id"],
            seed=data["seed"],
            tables=[TableData.from_dict(t) for t in data["tables"]],
            program=Program.from_dict(data["program"]),
            metadata=dict(data.get("metadata", {})),
        )
