from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Any

BASE_FILTER_COMPARATORS = (">", ">=", "<", "<=", "==", "!=")
SET_FILTER_COMPARATORS = ("in_set", "not_in_set")
NULL_FILTER_COMPARATORS = ("is_null", "is_not_null")
RANGE_FILTER_COMPARATORS = ("range_closed",)
STRING_FILTER_COMPARATORS = ("str_contains", "str_starts_with", "str_ends_with")
BOOLEAN_FILTER_COMPARATORS = (
    "bool_is_true",
    "bool_is_not_true",
    "bool_is_false",
    "bool_is_not_false",
    "bool_is_unknown",
    "bool_is_not_unknown",
)
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
FILTER_COMPARATORS = frozenset(
    BASE_FILTER_COMPARATORS
    + TRUTH_FILTER_COMPARATORS
    + SET_FILTER_COMPARATORS
    + NULL_FILTER_COMPARATORS
    + RANGE_FILTER_COMPARATORS
    + STRING_FILTER_COMPARATORS
    + BOOLEAN_FILTER_COMPARATORS
)


@dataclass(frozen=True, slots=True)
class FilterComparator:
    base: str
    truth_test: str | None = None


def parse_filter_comparator(raw: Any) -> FilterComparator | None:
    comparator = str(raw)
    if comparator in BASE_FILTER_COMPARATORS:
        return FilterComparator(comparator)
    if comparator in SET_FILTER_COMPARATORS:
        return FilterComparator(comparator)
    if comparator in NULL_FILTER_COMPARATORS:
        return FilterComparator(comparator)
    if comparator in RANGE_FILTER_COMPARATORS:
        return FilterComparator(comparator)
    if comparator in STRING_FILTER_COMPARATORS:
        return FilterComparator(comparator)
    if comparator in BOOLEAN_FILTER_COMPARATORS:
        return FilterComparator("bool_predicate", comparator.removeprefix("bool_"))
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
    if parsed.base in SET_FILTER_COMPARATORS:
        return parsed.truth_test is None and column_type in {"int", "float", "str", "bool"}
    if parsed.base in NULL_FILTER_COMPARATORS:
        return parsed.truth_test is None and column_type in {"int", "float", "str", "bool"}
    if parsed.base in RANGE_FILTER_COMPARATORS:
        return parsed.truth_test is None and column_type in {"int", "float"}
    if parsed.base in STRING_FILTER_COMPARATORS:
        return parsed.truth_test is None and column_type == "str"
    if parsed.base == "bool_predicate":
        return column_type == "bool" and parsed.truth_test in TRUTH_TESTS
    if column_type in {"str", "bool"}:
        return parsed.base in {"==", "!="}
    return column_type in {"int", "float"}


def evaluate_filter_predicate(left: Any, raw: Any, right: Any) -> bool:
    parsed = parse_filter_comparator(raw)
    if parsed is None:
        raise ValueError(raw)
    if parsed.base == "in_set":
        if _is_nullish_scalar(left):
            return False
        if not isinstance(right, (list, tuple, frozenset, set)):
            raise ValueError("in_set literal must be a collection")
        return left in right
    if parsed.base == "not_in_set":
        if _is_nullish_scalar(left):
            return False
        if not isinstance(right, (list, tuple, frozenset, set)):
            raise ValueError("not_in_set literal must be a collection")
        return left not in right
    if parsed.base == "is_null":
        return _is_nullish_scalar(left)
    if parsed.base == "is_not_null":
        return not _is_nullish_scalar(left)
    if parsed.base == "range_closed":
        lower, upper = _range_bounds(right)
        return _compare_three_valued(left, ">=", lower) is True and _compare_three_valued(left, "<=", upper) is True
    if parsed.base == "str_contains":
        if _is_nullish_scalar(left):
            return False
        if not isinstance(left, str) or not isinstance(right, str) or right == "":
            raise ValueError("string pattern comparators require a non-empty string literal and string values")
        return right in left
    if parsed.base == "str_starts_with":
        if _is_nullish_scalar(left):
            return False
        if not isinstance(left, str) or not isinstance(right, str) or right == "":
            raise ValueError("string pattern comparators require a non-empty string literal and string values")
        return left.startswith(right)
    if parsed.base == "str_ends_with":
        if _is_nullish_scalar(left):
            return False
        if not isinstance(left, str) or not isinstance(right, str) or right == "":
            raise ValueError("string pattern comparators require a non-empty string literal and string values")
        return left.endswith(right)
    if parsed.base == "bool_predicate":
        return _apply_truth_test(_bool_three_valued(left), parsed.truth_test or "")
    comparison = _compare_three_valued(left, parsed.base, right)
    if parsed.truth_test is None:
        return comparison is True
    return _apply_truth_test(comparison, parsed.truth_test)


def sql_filter_condition(column_sql: str, literal_sql: str, raw: Any) -> str:
    parsed = parse_filter_comparator(raw)
    if parsed is None:
        raise ValueError(raw)
    if parsed.base == "in_set":
        return f"{column_sql} IN {literal_sql}"
    if parsed.base == "not_in_set":
        return f"{column_sql} IS NOT NULL AND {column_sql} NOT IN {literal_sql}"
    if parsed.base == "is_null":
        return f"{column_sql} IS NULL"
    if parsed.base == "is_not_null":
        return f"{column_sql} IS NOT NULL"
    if parsed.base == "range_closed":
        lower_sql, upper_sql = _range_bounds_sql(literal_sql)
        return f"{column_sql} BETWEEN {lower_sql} AND {upper_sql}"
    if parsed.base == "str_contains":
        return f"instr({column_sql}, {literal_sql}) > 0"
    if parsed.base == "str_starts_with":
        return f"substr({column_sql}, 1, length({literal_sql})) = {literal_sql}"
    if parsed.base == "str_ends_with":
        return (
            f"substr({column_sql}, length({column_sql}) - length({literal_sql}) + 1, "
            f"length({literal_sql})) = {literal_sql}"
        )
    if parsed.base == "bool_predicate":
        truth_test = parsed.truth_test
        if truth_test == "is_true":
            return f"{column_sql} IS TRUE"
        if truth_test == "is_not_true":
            return f"{column_sql} IS NOT TRUE"
        if truth_test == "is_false":
            return f"{column_sql} IS FALSE"
        if truth_test == "is_not_false":
            return f"{column_sql} IS NOT FALSE"
        if truth_test == "is_unknown":
            return f"{column_sql} IS NULL"
        if truth_test == "is_not_unknown":
            return f"{column_sql} IS NOT NULL"
        raise ValueError(truth_test)
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


def _range_bounds(value: Any) -> tuple[Any, Any]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("range_closed literal must contain exactly two bounds")
    return value[0], value[1]


def _range_bounds_sql(literal_sql: str) -> tuple[str, str]:
    text = literal_sql.strip()
    if text.startswith("(") and text.endswith(")"):
        text = text[1:-1]
    parts = [part.strip() for part in text.split(",", 1)]
    if len(parts) != 2 or not all(parts):
        raise ValueError("range_closed SQL literal must contain exactly two bounds")
    return parts[0], parts[1]


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


def _bool_three_valued(value: Any) -> bool | None:
    if _is_nullish_scalar(value):
        return None
    if isinstance(value, bool):
        return value
    cls = type(value)
    if cls.__module__.startswith("numpy") and cls.__name__ in {"bool", "bool_"}:
        return bool(value)
    return None


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
