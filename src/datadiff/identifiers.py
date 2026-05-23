from __future__ import annotations

import re
from collections.abc import Iterable

RESERVED_OUTPUT_NAMES = frozenset(
    {
        "aggregate",
        "all",
        "and",
        "as",
        "asc",
        "between",
        "by",
        "case",
        "count",
        "cross",
        "desc",
        "distinct",
        "false",
        "filter",
        "from",
        "full",
        "group",
        "groupby",
        "having",
        "inner",
        "join",
        "left",
        "limit",
        "max",
        "min",
        "mutate",
        "not",
        "null",
        "offset",
        "on",
        "or",
        "order",
        "outer",
        "right",
        "select",
        "sort",
        "sum",
        "table",
        "true",
        "where",
    }
)
RESERVED_OUTPUT_PREFIXES = ("__datadiff_",)


def is_reserved_output_name(name: str) -> bool:
    lowered = name.lower()
    return lowered in RESERVED_OUTPUT_NAMES or any(lowered.startswith(prefix) for prefix in RESERVED_OUTPUT_PREFIXES)


def make_safe_output_name(base: str, *, used: Iterable[str] = ()) -> str:
    candidate = re.sub(r"[^0-9A-Za-z_]+", "_", str(base)).strip("_") or "derived"
    if candidate[0].isdigit():
        candidate = f"col_{candidate}"
    if is_reserved_output_name(candidate):
        candidate = f"derived_{candidate}"

    used_lower = {str(name).lower() for name in used}
    if candidate.lower() not in used_lower:
        return candidate

    stem = candidate
    suffix = 1
    while f"{stem}_{suffix}".lower() in used_lower or is_reserved_output_name(f"{stem}_{suffix}"):
        suffix += 1
    return f"{stem}_{suffix}"
