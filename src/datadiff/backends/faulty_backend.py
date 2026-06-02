from __future__ import annotations

from typing import Any

from datadiff.backends.base import BackendResult
from datadiff.backends.pandas_backend import PandasBackend
from datadiff.dsl import Program, TableData
from datadiff.operation_semantics import op_column, op_kind


class FaultyPandasBackend(PandasBackend):
    def __init__(self, name: str, fault: str) -> None:
        self.name = name
        self.fault = fault

    def run(self, tables: list[TableData], program: Program, timeout_s: float = 5.0) -> BackendResult:
        result = super().run(tables, program, timeout_s=timeout_s)
        result.backend = self.name
        if result.status != "ok":
            return result
        if not _program_triggers_fault(program, self.fault):
            return result
        try:
            result.data = _inject_fault(result.data, program, self.fault)
        except Exception as exc:  # noqa: BLE001
            return BackendResult(
                self.name,
                "error",
                error_type=type(exc).__name__,
                error=str(exc),
                duration_ms=result.duration_ms,
            )
        return result


def _program_triggers_fault(program: Program, fault: str) -> bool:
    ops = program.operations
    if fault == "filter":
        return any(op_kind(op) == "filter" for op in ops)
    if fault == "groupby":
        return any(op_kind(op) == "groupby" for op in ops)
    if fault == "join":
        return any(op_kind(op) == "join" for op in ops)
    if fault == "mutate":
        return any(op_kind(op) == "mutate" for op in ops)
    return False


def _inject_fault(df: Any, program: Program, fault: str) -> Any:
    out = df.copy()
    if out.empty:
        return out
    if fault == "filter":
        return out.iloc[:-1].copy() if len(out) > 1 else out.iloc[0:0].copy()
    if fault == "groupby":
        column = _first_numeric_column(out)
        if column is not None:
            out.loc[out.index[0], column] = out.loc[out.index[0], column] + 1
        return out
    if fault == "join":
        return out.iloc[1:].copy() if len(out) > 1 else out.iloc[0:0].copy()
    if fault == "mutate":
        column = _last_mutated_output_column(program)
        if column not in out.columns:
            column = _first_non_key_column(out) or _first_column(out)
        if column is not None:
            out.loc[out.index[0], column] = _perturb_value(out.loc[out.index[0], column])
        return out
    return out


def _first_numeric_column(df: Any) -> str | None:
    for column in df.columns:
        try:
            if str(df[column].dtype).startswith(("int", "float")):
                return str(column)
        except Exception:  # noqa: BLE001
            continue
    return _first_column(df)


def _last_mutated_output_column(program: Program) -> str | None:
    for op in reversed(program.operations):
        if op_kind(op) == "mutate":
            return op_column(op)
    return None


def _first_non_key_column(df: Any) -> str | None:
    for column in df.columns:
        name = str(column)
        if name not in {"id", "g", "s"}:
            return name
    return None


def _first_column(df: Any) -> str | None:
    return str(df.columns[0]) if len(df.columns) else None


def _perturb_value(value: Any) -> Any:
    if value is None:
        return "__fault__"
    if isinstance(value, bool):
        return not value
    if isinstance(value, (int, float)):
        return value + 1
    return f"{value}__fault__"
