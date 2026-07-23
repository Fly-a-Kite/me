"""Physical fresh-discovery declaration view.

The module contains declarations only.  Runtime authority must not branch on
the reporting-only ``test_family_id`` carried by a compiled cell.
"""

from __future__ import annotations

from .declarations import _legacy_templates_for
from .model import ProvenanceClass, TargetTemplate


def fresh_target_templates() -> tuple[TargetTemplate, ...]:
    templates = _legacy_templates_for(ProvenanceClass.FRESH_DISCOVERY)
    if not templates or any(
        item.provenance_class != ProvenanceClass.FRESH_DISCOVERY
        for item in templates
    ):
        raise ValueError("fresh registry contains non-fresh declaration")
    return templates


__all__ = ["fresh_target_templates"]
