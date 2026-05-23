from __future__ import annotations

import math
from typing import Any


def is_sorted_values(values: list[Any], *, ascending: bool = True, nulls: str = "last") -> bool:
    for left, right in zip(values, values[1:]):
        if _is_null_like(left) and _is_null_like(right):
            continue
        if _is_null_like(left):
            if nulls == "last":
                return False
            continue
        if _is_null_like(right):
            if nulls == "first":
                return False
            continue
        if ascending:
            if left > right:
                return False
        elif left < right:
            return False
    return True


def _is_null_like(value: Any) -> bool:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return True
    try:
        import pandas as pd

        return bool(pd.isna(value))
    except Exception:
        return False
