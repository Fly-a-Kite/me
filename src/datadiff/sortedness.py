from __future__ import annotations

from typing import Any

from datadiff.ordering_semantics import is_sorted_scalar_values


def is_sorted_values(values: list[Any], *, ascending: bool = True, nulls: str = "last") -> bool:
    return is_sorted_scalar_values(values, ascending=ascending, nulls=nulls, null_like=True, scalar_fallback_to_repr=True)
