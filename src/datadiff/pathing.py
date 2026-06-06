from __future__ import annotations

from pathlib import Path
from typing import Any

from datadiff.ordering_semantics import is_null_like
from datadiff.util import PROJECT_ROOT


def path_basename(value: Any) -> str | None:
    if is_null_like(value):
        return None
    text = str(value)
    slash = max(text.rfind("/"), text.rfind("\\"))
    return text[slash + 1 :] if slash >= 0 else text


def resolve_project_path(path: str | Path | None, *, project_root: Path = PROJECT_ROOT) -> Path:
    resolved = Path(path or "")
    return resolved if resolved.is_absolute() else project_root / resolved


def project_display_path(path: str | Path | None, *, project_root: Path = PROJECT_ROOT) -> str:
    if path in {None, ""}:
        return ""
    resolved = resolve_project_path(path, project_root=project_root)
    try:
        return str(resolved.resolve().relative_to(project_root))
    except ValueError:
        return str(resolved)
