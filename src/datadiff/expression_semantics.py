from __future__ import annotations

from typing import Any, Mapping

from datadiff.dsl import Expression, coerce_expression
from datadiff.operation_semantics import (
    expr_index,
    expr_input_domain,
    expr_kind,
    expr_length,
    expr_lower,
    expr_needle,
    expr_new,
    expr_numerator,
    expr_old,
    expr_operator,
    expr_other,
    expr_part,
    expr_separator,
    expr_source,
    expr_start,
    expr_target_type,
    expr_upper,
    expr_value,
)
from datadiff.pathing import path_basename


def literal_output_type(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    return None


def aggregate_output_type(source_type: str | None, func: str) -> str:
    if func in {"count", "nunique"}:
        return "int"
    if func in {"any", "all"}:
        return "bool"
    if func == "mean":
        return "float"
    return source_type or "float"


def cast_output_type(source_type: str, expr: Mapping[str, Any] | Expression) -> str | None:
    wrapped = expr if isinstance(expr, _ExprOp) else _expr_wrapper(expr)
    target = expr_target_type(wrapped)
    input_domain = expr_input_domain(wrapped)
    if target == "float" and source_type in {"int", "float"}:
        return "float"
    if target == "float" and source_type == "str" and input_domain in {"numeric_string", "integer_string"}:
        return "float"
    if target == "int" and source_type == "int":
        return "int"
    if target == "int" and source_type == "str" and input_domain == "integer_string":
        return "int"
    if target == "str" and source_type in {"int", "float", "str"}:
        return "str"
    return None


def expr_output_type(expr: Mapping[str, Any] | Expression, col_types: Mapping[str, str]) -> str | None:
    wrapped = expr if isinstance(expr, _ExprOp) else _expr_wrapper(expr)
    source = expr_source(wrapped)
    source_type = col_types.get(source)
    if source_type is None:
        return None
    kind = expr_kind(wrapped)
    if kind == "add_const":
        return source_type if source_type in {"int", "float"} else None
    if kind == "arith_const":
        operator = expr_operator(wrapped)
        if source_type not in {"int", "float"} or operator not in {"sub", "mul", "div", "mod"}:
            return None
        return "float" if operator == "div" or source_type == "float" else source_type
    if kind == "reverse_division_columns":
        numerator = expr_numerator(wrapped)
        if source_type not in {"int", "float"} or col_types.get(numerator) not in {"int", "float"}:
            return None
        return "float"
    if kind == "abs":
        return source_type if source_type in {"int", "float"} else None
    if kind == "clip":
        lower = expr_lower(wrapped)
        upper = expr_upper(wrapped)
        if source_type not in {"int", "float"}:
            return None
        if lower is None or upper is None:
            return None
        if not all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in (lower, upper)):
            return None
        if lower > upper:
            return None
        if source_type == "int":
            return source_type if type(lower) is int and type(upper) is int else None
        return source_type
    if kind == "bool_not":
        return "bool" if source_type == "bool" else None
    if kind == "cast":
        return cast_output_type(source_type, wrapped)
    if kind == "string_length":
        return "int" if source_type == "str" else None
    if kind in {"string_lower", "string_upper", "string_strip", "string_null_if_empty", "string_basename"}:
        return "str" if source_type == "str" else None
    if kind == "string_replace":
        old = expr_old(wrapped)
        new = expr_new(wrapped)
        if source_type != "str" or not isinstance(old, str) or old == "" or not isinstance(new, str):
            return None
        return "str"
    if kind == "string_slice":
        start = expr_start(wrapped)
        length = expr_length(wrapped)
        if source_type != "str" or type(start) is not int or type(length) is not int or start < 0 or length < 0:
            return None
        return "str"
    if kind == "string_split_part":
        sep = expr_separator(wrapped)
        index = expr_index(wrapped)
        if source_type != "str" or not isinstance(sep, str) or sep == "" or type(index) is not int or index != 0:
            return None
        return "str"
    if kind == "string_concat":
        other = expr_other(wrapped)
        raw_sep = dict(expr).get("sep") if not isinstance(expr, Expression) else wrapped.expression.get("sep")
        if source_type != "str" or col_types.get(other) != "str" or not isinstance(raw_sep, str):
            return None
        return "str"
    if kind in {"string_contains", "string_starts_with", "string_ends_with"}:
        needle = expr_needle(wrapped)
        if source_type != "str" or not isinstance(needle, str):
            return None
        return "bool"
    if kind == "date_part":
        if source_type != "str" or expr_part(wrapped) not in {"year", "month", "day"}:
            return None
        return "int"
    return None


def eval_expr_on_value(
    expr: Mapping[str, Any] | Expression,
    value: Any,
    *,
    other_value: Any = None,
) -> Any:
    wrapped = expr if isinstance(expr, _ExprOp) else _expr_wrapper(expr)
    if value is None:
        return None
    kind = expr_kind(wrapped)
    try:
        if kind == "add_const":
            return value + expr_value(wrapped)
        if kind == "arith_const":
            operator = expr_operator(wrapped)
            operand = expr_value(wrapped)
            if operator == "sub":
                return value - operand
            if operator == "mul":
                return value * operand
            if operator == "div":
                return value / operand
            if operator == "mod":
                return value % operand
            return None
        if kind == "abs":
            return abs(value)
        if kind == "clip":
            return min(max(value, expr_lower(wrapped)), expr_upper(wrapped))
        if kind == "bool_not":
            return not value if isinstance(value, bool) else None
        if kind == "cast":
            target = expr_target_type(wrapped)
            if target == "float":
                return float(value)
            if target == "int":
                return int(value)
            if target == "str":
                return str(value)
            return None
        if kind == "string_length":
            return len(value) if isinstance(value, str) else None
        if kind == "string_lower":
            return value.lower() if isinstance(value, str) else None
        if kind == "string_upper":
            return value.upper() if isinstance(value, str) else None
        if kind == "string_strip":
            return value.strip() if isinstance(value, str) else None
        if kind == "string_null_if_empty":
            return None if value == "" else value if isinstance(value, str) else None
        if kind == "string_replace":
            old = expr_old(wrapped)
            new = expr_new(wrapped)
            return value.replace(old, new) if isinstance(value, str) else None
        if kind == "string_slice":
            start = expr_start(wrapped)
            length = expr_length(wrapped)
            return value[start : start + length] if isinstance(value, str) else None
        if kind == "string_split_part":
            sep = expr_separator(wrapped)
            index = expr_index(wrapped)
            return value.split(sep, 1)[index] if isinstance(value, str) and isinstance(sep, str) else None
        if kind == "string_concat":
            sep = expr_separator(wrapped)
            return value + sep + other_value if isinstance(value, str) and isinstance(other_value, str) else None
        if kind == "string_contains":
            needle = expr_needle(wrapped)
            return needle in value if isinstance(value, str) and isinstance(needle, str) else None
        if kind == "string_starts_with":
            needle = expr_needle(wrapped)
            return value.startswith(needle) if isinstance(value, str) and isinstance(needle, str) else None
        if kind == "string_ends_with":
            needle = expr_needle(wrapped)
            return value.endswith(needle) if isinstance(value, str) and isinstance(needle, str) else None
        if kind == "date_part":
            spans = {"year": (0, 4), "month": (5, 7), "day": (8, 10)}
            span = spans.get(expr_part(wrapped))
            if span is None or not isinstance(value, str):
                return None
            start, stop = span
            return int(value[start:stop])
        if kind == "string_basename":
            text = str(value)
            slash = max(text.rfind("/"), text.rfind("\\"))
            return text[slash + 1 :] if slash >= 0 else text
    except Exception:
        return None
    return None


def eval_expr_on_row(
    expr: Mapping[str, Any] | Expression,
    row: Mapping[str, Any],
) -> Any:
    wrapped = expr if isinstance(expr, _ExprOp) else _expr_wrapper(expr)
    source = expr_source(wrapped)
    value = row.get(source)
    if value is None:
        return None
    kind = expr_kind(wrapped)
    if kind == "reverse_division_columns":
        numerator = row.get(expr_numerator(wrapped))
        if numerator is None:
            return None
        try:
            return numerator / value
        except Exception:
            return None
    if kind == "string_concat":
        other = row.get(expr_other(wrapped))
        if other is None:
            return None
        return eval_expr_on_value(wrapped, value, other_value=other)
    if kind == "string_basename":
        return path_basename(value)
    return eval_expr_on_value(wrapped, value)


def _expr_wrapper(expr: Mapping[str, Any] | Expression) -> Any:
    return _ExprOp(expr)


class _ExprOp:
    def __init__(self, expr: Mapping[str, Any] | Expression) -> None:
        self.expression = coerce_expression(expr)

    def get(self, key: str, default: Any = None) -> Any:
        if key == "expr":
            return self.expression
        return default
