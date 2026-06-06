from __future__ import annotations

import random
from typing import Any


STRING_PATTERN_FALLBACKS = ("a", "A", "beta", "中文", "space", "value", "missing")


def string_pattern_literal_for_type(rnd: random.Random, typ: str) -> str:
    if typ != "str":
        raise ValueError(typ)
    return rnd.choice(STRING_PATTERN_FALLBACKS)


def string_pattern_literal_for_column(
    rnd: random.Random,
    table: Any,
    col: str,
    comparator: str,
) -> str:
    values = [row.get(col) for row in table.rows if isinstance(row.get(col), str) and row.get(col) != ""]
    if values and rnd.random() < 0.75:
        value = str(rnd.choice(values))
        if comparator == "str_starts_with":
            return value[: rnd.randint(1, min(len(value), 4))]
        if comparator == "str_ends_with":
            width = rnd.randint(1, min(len(value), 4))
            return value[-width:]
        if " " in value and rnd.random() < 0.5:
            return rnd.choice([" ", value.split(" ", 1)[0]])
        if len(value) == 1:
            return value
        start = rnd.randint(0, len(value) - 1)
        end = rnd.randint(start + 1, min(len(value), start + 4))
        return value[start:end]
    return string_pattern_literal_for_type(rnd, "str")
