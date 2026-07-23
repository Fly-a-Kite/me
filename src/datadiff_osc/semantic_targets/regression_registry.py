"""Physical known-regression declaration view."""

from __future__ import annotations

from .declarations import _legacy_templates_for
from .model import ProvenanceClass, TargetTemplate


def regression_target_templates() -> tuple[TargetTemplate, ...]:
    templates = _legacy_templates_for(ProvenanceClass.KNOWN_REGRESSION)
    if not templates or any(
        item.provenance_class != ProvenanceClass.KNOWN_REGRESSION
        for item in templates
    ):
        raise ValueError("regression registry contains non-regression declaration")
    return templates


__all__ = ["regression_target_templates"]
