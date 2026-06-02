from __future__ import annotations

from typing import Any

from datadiff.ordering_semantics import is_null_like


def path_basename(value: Any) -> str | None:
    if is_null_like(value):
        return None
    text = str(value)
    slash = max(text.rfind("/"), text.rfind("\\"))
    return text[slash + 1 :] if slash >= 0 else text
