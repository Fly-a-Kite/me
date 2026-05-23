from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Any

BASE_FILTER_COMPARATORS = (">", ">=", "<", "<=", "==", "!=")
_COMPARATOR_TOKENS = {
    "gt": ">",
    "ge": ">=",
    "lt": "<",
    "le": "<=",
    "eq": "==",
    "ne": "!=",
}
_SQL_COMPARATORS = {
    ">": ">",
    ">=": ">=",
    "<": "<",
    "<=": "<=",
    "==": "=",
    "!=": "<>",
}
TRUTH_TESTS = (
    "is_true",
    "is_not_true",
    "is_false",
    "is_not_false",
    "is_unknown",
    "is_not_unknown",
)
TRUTH_FILTER_COMPARATORS = tuple(
    f"{token}_{truth_test}"
    for token in _COMPARATOR_TOKENS
    for truth_test in TRUTH_TESTS
)
FILTER_COMPARATORS = frozenset(BASE_FILTER_COMPARATORS + TRUTH_FILTER_COMPARATORS)


@dataclass(frozen=True, slots=True)
class FilterComparator:
    base: str
    truth_test: str | None = None


def parse_filter_comparator(raw: Any) -> FilterComparator | None:
    comparator = str(raw)
    if comparator in BASE_FILTER_COMPARATORS:
        return FilterComparator(comparator)
    for token, base in _COMPARATOR_TOKENS.items():
        prefix = f"{token}_"
        if comparator.startswith(prefix):
            truth_test = comparator[len(prefix):]
            if truth_test in TRUTH_TESTS:
                return FilterComparator(base, truth_test)
    return None


def is_filter_comparator(raw: Any) -> bool:
    return parse_filter_comparator(raw) is not None


def filter_comparator_supports_type(column_type: str, raw: Any) -> bool:
    parsed = parse_filter_comparator(raw)
    if parsed is None:
        return False
    if column_type in {"str", "bool"}:
        return parsed.base in {"==", "!="}
    return column_type in {"int", "float"}


def evaluate_filter_predicate(left: Any, raw: Any, right: Any) -> bool:
    parsed = parse_filter_comparator(raw)
    if parsed is None:
        raise ValueError(raw)
    comparison = _compare_three_valued(left, parsed.base, right)
    if parsed.truth_test is None:
        return comparison is True
    return _apply_truth_test(comparison, parsed.truth_test)


def sql_filter_condition(column_sql: str, literal_sql: str, raw: Any) -> str:
    parsed = parse_filter_comparator(raw)
    if parsed is None:
        raise ValueError(raw)
    base = f"{column_sql} {_SQL_COMPARATORS[parsed.base]} {literal_sql}"
    truth_test = parsed.truth_test
    if truth_test is None:
        return base
    if truth_test == "is_true":
        return f"({base}) IS TRUE"
    if truth_test == "is_not_true":
        return f"NOT (({base}) IS TRUE)"
    if truth_test == "is_false":
        return f"({base}) IS FALSE"
    if truth_test == "is_not_false":
        return f"NOT (({base}) IS FALSE)"
    if truth_test == "is_unknown":
        return f"({base}) IS NULL"
    if truth_test == "is_not_unknown":
        return f"({base}) IS NOT NULL"
    raise ValueError(truth_test)


def _compare_three_valued(left: Any, comparator: str, right: Any) -> bool | None:
    if _is_nullish_scalar(left) or _is_nullish_scalar(right):
        return None
    try:
        if comparator == ">":
            return bool(left > right)
        if comparator == ">=":
            return bool(left >= right)
        if comparator == "<":
            return bool(left < right)
        if comparator == "<=":
            return bool(left <= right)
        if comparator == "==":
            return bool(left == right)
        if comparator == "!=":
            return bool(left != right)
    except Exception:
        return None
    raise ValueError(comparator)


def _is_nullish_scalar(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, Real) and not isinstance(value, bool):
        try:
            return math.isnan(float(value))
        except (TypeError, ValueError, OverflowError):
            return False
    cls = type(value)
    if cls.__module__.startswith("pandas.") and cls.__name__ in {"NAType", "NaTType"}:
        return True
    return False


def _apply_truth_test(value: bool | None, truth_test: str) -> bool:
    if truth_test == "is_true":
        return value is True
    if truth_test == "is_not_true":
        return value is not True
    if truth_test == "is_false":
        return value is False
    if truth_test == "is_not_false":
        return value is not False
    if truth_test == "is_unknown":
        return value is None
    if truth_test == "is_not_unknown":
        return value is not None
    raise ValueError(truth_test)
