from __future__ import annotations

import math
import time
from collections.abc import Mapping
from typing import Any

from datadiff.backends.base import Backend, BackendResult, NativeRows, PreparedTable
from datadiff.dsl import Program
from datadiff.filtering import evaluate_filter_predicate
from datadiff.operation_semantics import (
    expr_source,
    op_column,
    op_columns,
    op_comparator,
    op_kind,
    op_n,
    op_value,
)


class NumPyBackend(Backend):
    """Narrow, independently executed NumPy column adapter for P6 portability."""

    name = "numpy"
    session_reuse_policy = "stateless"
    input_physical_layout = "numpy_object_columns"

    def execute_lowered(
        self,
        tables: list[PreparedTable],
        program: Program,
        timeout_s: float = 5.0,
    ) -> BackendResult:
        del timeout_s
        started = time.perf_counter()
        try:
            import numpy as np

            if len(tables) != 1:
                return self._missing(
                    started,
                    "NumPy adapter supports exactly one input table",
                )
            table = tables[0]
            columns = list(table.column_names)
            arrays = {
                column: np.asarray(table.columns_data[column], dtype=object)
                for column in columns
            }
            for operation in program.operations:
                kind = op_kind(operation, default="unknown")
                if kind == "select":
                    selected = list(op_columns(operation))
                    _require_columns(arrays, selected)
                    columns = selected
                    arrays = {column: arrays[column] for column in columns}
                elif kind == "filter":
                    column = op_column(operation)
                    _require_columns(arrays, [column])
                    comparator = op_comparator(operation)
                    expected = op_value(operation)
                    mask = np.fromiter(
                        (
                            _predicate(value, comparator, expected)
                            for value in arrays[column]
                        ),
                        dtype=bool,
                        count=len(arrays[column]),
                    )
                    arrays = {name: values[mask] for name, values in arrays.items()}
                elif kind == "mutate":
                    target = op_column(operation)
                    expression = operation.get("expr", {})
                    source = expr_source(operation)
                    _require_columns(arrays, [source])
                    arrays[target] = np.fromiter(
                        (
                            _evaluate_expression(value, expression)
                            for value in arrays[source]
                        ),
                        dtype=object,
                        count=len(arrays[source]),
                    )
                    if target not in columns:
                        columns.append(target)
                elif kind in {"limit", "offset"}:
                    count = max(0, op_n(operation))
                    selection = slice(0, count) if kind == "limit" else slice(count, None)
                    arrays = {
                        name: values[selection]
                        for name, values in arrays.items()
                    }
                else:
                    return self._missing(
                        started,
                        f"unsupported NumPy operation: {kind}",
                    )

            row_count = len(arrays[columns[0]]) if columns else 0
            rows = [
                [
                    _python_scalar(arrays[column][row_index])
                    for column in columns
                ]
                for row_index in range(row_count)
            ]
            return BackendResult(
                backend=self.name,
                status="ok",
                data=NativeRows(columns=columns, row_values=rows),
                duration_ms=(time.perf_counter() - started) * 1000.0,
            )
        except Exception as exc:  # noqa: BLE001
            return BackendResult(
                backend=self.name,
                status="error",
                error_type=type(exc).__name__,
                error=str(exc),
                duration_ms=(time.perf_counter() - started) * 1000.0,
            )

    def _missing(self, started: float, message: str) -> BackendResult:
        return BackendResult(
            backend=self.name,
            status="missing",
            error_type="UnsupportedNumPyOperation",
            error=message,
            duration_ms=(time.perf_counter() - started) * 1000.0,
        )


def _require_columns(arrays: dict[str, Any], columns: list[str]) -> None:
    missing = [column for column in columns if column not in arrays]
    if missing:
        raise KeyError("unknown NumPy column(s): " + ", ".join(missing))


def _predicate(value: Any, comparator: str, expected: Any) -> bool:
    return evaluate_filter_predicate(value, comparator, expected)


def _evaluate_expression(value: Any, expression: Any) -> Any:
    if value is None:
        return None
    if not isinstance(expression, Mapping):
        raise TypeError("NumPy mutation expression must be a mapping")
    kind = str(expression.get("kind", ""))
    if kind == "add_const":
        return value + expression.get("value", 0)
    if kind == "arith_const":
        operand = expression.get("value", 0)
        operator = str(expression.get("op", "add"))
        if operator == "add":
            return value + operand
        if operator == "sub":
            return value - operand
        if operator == "mul":
            return value * operand
        if operator == "div":
            if operand == 0:
                return math.nan
            return value / operand
        if operator == "mod":
            if operand == 0:
                return None
            return value % operand
        raise ValueError(f"unsupported NumPy arithmetic operator: {operator}")
    if kind == "abs":
        return abs(value)
    if kind == "bool_not":
        return not bool(value)
    if kind == "string_lower":
        return str(value).lower()
    if kind == "string_upper":
        return str(value).upper()
    if kind == "string_strip":
        return str(value).strip()
    raise ValueError(f"unsupported NumPy expression: {kind}")


def _python_scalar(value: Any) -> Any:
    item = getattr(value, "item", None)
    return item() if callable(item) else value
