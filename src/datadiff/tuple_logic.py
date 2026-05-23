from __future__ import annotations

import math
from typing import Any


TruthValue = bool | None


def evaluate_tuple_absence(
    left_row: dict[str, Any],
    left_columns: list[str],
    right_rows: list[dict[str, Any]],
    right_columns: list[str],
) -> bool:
    """Return whether SQL row-value NOT IN is TRUE for the left row."""

    left_values = [left_row.get(column) for column in left_columns]
    right_values = [
        [right_row.get(column) for column in right_columns]
        for right_row in right_rows
    ]
    return _tuple_not_in_truth(left_values, right_values) is True


def _tuple_not_in_truth(left_values: list[Any], right_values: list[list[Any]]) -> TruthValue:
    found_unknown = False
    for right in right_values:
        equality = _row_equality_truth(left_values, right)
        if equality is True:
            return False
        if equality is None:
            found_unknown = True
    return None if found_unknown else True


def _row_equality_truth(left_values: list[Any], right_values: list[Any]) -> TruthValue:
    found_unknown = False
    for left, right in zip(left_values, right_values):
        equality = _scalar_equality_truth(left, right)
        if equality is False:
            return False
        if equality is None:
            found_unknown = True
    return None if found_unknown else True


def _scalar_equality_truth(left: Any, right: Any) -> TruthValue:
    if _is_nullish(left) or _is_nullish(right):
        return None
    return left == right


def _is_nullish(value: Any) -> bool:
    if value is None:
        return True
    return isinstance(value, float) and math.isnan(value)
