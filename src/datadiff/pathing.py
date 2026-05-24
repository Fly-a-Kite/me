from __future__ import annotations

from typing import Any


def path_basename(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    slash = max(text.rfind("/"), text.rfind("\\"))
    return text[slash + 1 :] if slash >= 0 else text
