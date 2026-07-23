from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from datadiff.backends.base import Backend
from datadiff.dsl import Case
from datadiff.targets import target_spec


PHYSICAL_LAYOUT_RESOLUTION_SCHEMA_VERSION = "physical-layout-resolution-v1"
_LOGICAL_LAYOUT_NAMES = frozenset({"", "default", "logical_rows", "native"})


@dataclass(frozen=True, slots=True)
class PhysicalLayoutResolution:
    backend: str
    previous_layout: str
    default_layout: str
    applied_layout: str
    requested_layouts: tuple[str, ...]
    ignored_layouts: tuple[str, ...]
    source: str

    @property
    def changed(self) -> bool:
        return self.applied_layout != self.previous_layout

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PHYSICAL_LAYOUT_RESOLUTION_SCHEMA_VERSION,
            "backend": self.backend,
            "previous_layout": self.previous_layout,
            "default_layout": self.default_layout,
            "applied_layout": self.applied_layout,
            "requested_layouts": list(self.requested_layouts),
            "ignored_layouts": list(self.ignored_layouts),
            "source": self.source,
            "changed": self.changed,
        }


def resolve_case_physical_layout(
    case: Case,
    backend_name: str,
    backend: Backend,
) -> PhysicalLayoutResolution:
    """Resolve one adapter-wide layout from table-level ``input_layouts``.

    The current adapters expose one physical-layout switch per backend rather
    than one switch per input table.  Uniform supported table directives are
    therefore applied to that backend.  Layouts belonging to another backend
    family are retained in CCS-IR but ignored at execution time.
    """

    resolved_backend = str(backend_name)
    previous_layout = str(backend.input_physical_layout or "default")
    supported_layouts = _supported_layouts(resolved_backend, backend)
    default_layout = _default_layout(previous_layout, supported_layouts)
    requested = _requested_layouts(case, resolved_backend)

    applicable: list[str] = []
    ignored: list[str] = []
    for layout, explicit_backend in requested:
        if layout in _LOGICAL_LAYOUT_NAMES:
            continue
        if explicit_backend or layout in supported_layouts:
            applicable.append(layout)
        else:
            ignored.append(layout)

    unique_applicable = tuple(dict.fromkeys(applicable))
    if len(unique_applicable) > 1:
        raise ValueError(
            "adapter-wide physical layout cannot represent mixed input layouts "
            f"for {resolved_backend}: {', '.join(unique_applicable)}"
        )
    applied_layout = unique_applicable[0] if unique_applicable else default_layout
    return PhysicalLayoutResolution(
        backend=resolved_backend,
        previous_layout=previous_layout,
        default_layout=default_layout,
        applied_layout=applied_layout,
        requested_layouts=tuple(dict.fromkeys(layout for layout, _explicit in requested)),
        ignored_layouts=tuple(dict.fromkeys(ignored)),
        source="input_layouts" if unique_applicable else "backend_default",
    )


def _supported_layouts(backend_name: str, backend: Backend) -> tuple[str, ...]:
    declared = tuple(
        str(layout)
        for layout in getattr(backend, "supported_physical_layouts", ())
        if str(layout)
    )
    if declared:
        return declared
    try:
        return tuple(target_spec(backend_name).capability_model.physical_layouts)
    except ValueError:
        return ()


def _default_layout(previous_layout: str, supported_layouts: tuple[str, ...]) -> str:
    if previous_layout not in _LOGICAL_LAYOUT_NAMES:
        return previous_layout
    if "contiguous" in supported_layouts:
        return "contiguous"
    if len(supported_layouts) == 1:
        return supported_layouts[0]
    return previous_layout or "default"


def _requested_layouts(case: Case, backend_name: str) -> list[tuple[str, bool]]:
    metadata = case.metadata if isinstance(case.metadata, Mapping) else {}
    layout_map = metadata.get("input_layouts", {})
    if not isinstance(layout_map, Mapping):
        return []

    backend_map = layout_map.get("backends", layout_map.get("_backends", {}))
    if isinstance(backend_map, Mapping):
        backend_layout = str(backend_map.get(backend_name, "") or "")
        if backend_layout:
            return [(backend_layout, True)]

    requested: list[tuple[str, bool]] = []
    for index, table in enumerate(case.tables):
        raw = layout_map.get(table.name, layout_map.get(str(index)))
        if isinstance(raw, str):
            requested.append((str(raw), False))
            continue
        if not isinstance(raw, Mapping):
            continue
        per_backend = raw.get("backend_layouts", raw.get("backends", {}))
        if isinstance(per_backend, Mapping):
            explicit = str(per_backend.get(backend_name, "") or "")
            if explicit:
                requested.append((explicit, True))
                continue
        requested.append((str(raw.get("representation", "logical_rows") or "logical_rows"), False))
    return requested
