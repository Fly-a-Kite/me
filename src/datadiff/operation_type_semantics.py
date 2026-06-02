from __future__ import annotations

from typing import Any

from datadiff.expression_semantics import aggregate_output_type, literal_output_type


def aggregate_accepts_type(source_type: str, func: str, is_numeric_column: bool) -> bool:
    if func in {"count", "nunique"}:
        return True
    if func in {"any", "all"}:
        return source_type == "bool"
    if func in {"min", "max"} and source_type == "bool":
        return True
    return is_numeric_column


def aggregate_result_type(source_type: str | None, func: str) -> str:
    return aggregate_output_type(source_type, func)


def case_when_output_type(then_value: Any, else_value: Any) -> str | None:
    then_type = literal_output_type(then_value)
    else_type = literal_output_type(else_value)
    if then_type is None or else_type is None:
        return None
    if then_type == else_type:
        return then_type
    if {then_type, else_type} <= {"int", "float"}:
        return "float"
    return None
