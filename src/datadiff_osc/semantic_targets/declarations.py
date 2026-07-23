from __future__ import annotations

"""Declarative OSC target templates and the narrow v1 registry bridge.

The legacy registry is used only as frozen declaration/construction evidence.
Neither target activation nor contract verdicts dispatch on its family/root IDs.
"""

from dataclasses import replace
from typing import Iterable

from datadiff.family_witness_registry import latest_family_witness_registrations
from datadiff.semantic_family_universe_v3 import (
    semantic_family_v3_definitions,
    semantic_family_v3_witness_specs,
)

from datadiff_osc._canonical import stable_digest

from .model import AxisSpec, ContrastRelation, ProvenanceClass, TargetTemplate


DECLARATION_SCHEMA_VERSION = "osc-target-declarations-v1"

_REGRESSION_SOURCE_MARKERS = frozenset(
    {
        "confirmed_root",
        "confirmed-root",
        "issue_replay",
        "issue-replay",
        "known_root",
        "known-root",
    }
)


def _contains_regression_marker(*values: object) -> bool:
    text = "\0".join(str(value) for value in values).lower()
    return any(marker in text for marker in _REGRESSION_SOURCE_MARKERS)


def _validate_provenance_closure(
    definitions: tuple[object, ...],
    registrations: tuple[object, ...],
) -> None:
    """Reject hidden or cross-partition root provenance before any digesting."""

    by_family = {str(item.family_id): item for item in registrations}
    root_partitions: dict[str, set[ProvenanceClass]] = {}
    for definition in definitions:
        observed = (
            ProvenanceClass.FRESH_DISCOVERY
            if definition.source == "coverage_expansion"
            else ProvenanceClass.KNOWN_REGRESSION
        )
        root_id = str(definition.root_id)
        root_partitions.setdefault(root_id, set()).add(observed)
        if observed is not ProvenanceClass.FRESH_DISCOVERY:
            continue
        registration = by_family.get(str(definition.family_id))
        if registration is None:
            raise ValueError("fresh declaration has no constructor registration")
        if _contains_regression_marker(
            definition.source,
            definition.root_id,
            definition.mechanism_id,
            registration.builder,
        ):
            raise ValueError("fresh declaration contains known-root provenance")
    overlaps = sorted(
        root_id
        for root_id, partitions in root_partitions.items()
        if len(partitions) > 1
    )
    if overlaps:
        raise ValueError(
            "fresh and regression declarations share root identity: "
            + ",".join(overlaps)
        )


def _axis_relation(name: str) -> ContrastRelation:
    if name in {
        "layout",
        "physical_layout",
        "execution_mode",
        "mode",
        "input_order",
        "projection_mode",
        "name_mode",
    }:
        return ContrastRelation.METAMORPHIC_EQUIVALENCE
    if name in {"boundary", "direction"}:
        return ContrastRelation.MONOTONIC_BOUNDARY
    return ContrastRelation.DIFFERENTIAL_ISOLATION


def _oracle_relation(comparison_view: str) -> str:
    view = str(comparison_view or "").strip().lower()
    if view == "exact":
        return "sequence_equal"
    if view in {"set", "set_equal"}:
        return "set_equal"
    if view in {"sequence", "ordered", "sequence_equal"}:
        return "sequence_equal"
    return "bag_equal"


def _capability_atoms(capabilities: Iterable[str]) -> frozenset[str]:
    """Keep capability permission separate from semantic activation atoms.

    Legacy declarations may list the union of capabilities across pipeline
    alternatives.  Treating that union as an all-of semantic clause makes valid
    cells unreachable and conflates capability support with target activation.
    The original values remain available in ``TargetTemplate.required_capabilities``.
    """

    _ = tuple(str(item) for item in capabilities)  # eagerly validate iterability
    return frozenset()


def _legacy_templates_for(
    provenance: ProvenanceClass,
) -> tuple[TargetTemplate, ...]:
    """Translate one declaration partition without importing an evaluator.

    This bridge is deliberately declaration-only.  Runtime extraction, matching,
    repair and scheduling never consume a family, root or builder identity.
    """

    if provenance not in {
        ProvenanceClass.FRESH_DISCOVERY,
        ProvenanceClass.KNOWN_REGRESSION,
    }:
        raise ValueError("only fresh and regression declarations enter v1 registry")

    definition_values = tuple(semantic_family_v3_definitions())
    registration_values = tuple(latest_family_witness_registrations())
    _validate_provenance_closure(definition_values, registration_values)
    definitions = {item.family_id: item for item in definition_values}
    witness_specs = {
        item.family_id: item for item in semantic_family_v3_witness_specs()
    }
    templates: list[TargetTemplate] = []
    for registration in registration_values:
        definition = definitions[registration.family_id]
        observed_provenance = (
            ProvenanceClass.FRESH_DISCOVERY
            if definition.source == "coverage_expansion"
            else ProvenanceClass.KNOWN_REGRESSION
        )
        if observed_provenance != provenance:
            continue
        if provenance == ProvenanceClass.FRESH_DISCOVERY:
            if definition.source != "coverage_expansion":
                raise ValueError("fresh declaration source is not coverage expansion")
            if _contains_regression_marker(
                definition.source,
                definition.root_id,
                registration.builder,
                definition.mechanism_id,
            ):
                raise ValueError("fresh declaration contains known-root provenance")
        axes = tuple(
            AxisSpec(
                name=str(name),
                values=tuple(str(value) for value in values),
                baseline=str(values[0]),
                relation=_axis_relation(str(name)),
            )
            for name, values in registration.axes
        )
        comparison_view = getattr(
            witness_specs.get(registration.family_id), "comparison_view", ""
        )
        declaration_payload = {
            "schema_version": DECLARATION_SCHEMA_VERSION,
            "test_family_id": registration.family_id,
            "source": definition.source,
            "root_id": definition.root_id,
            "mechanism": definition.mechanism_id,
            "axes": [
                {
                    "name": axis.name,
                    "values": axis.values,
                    "baseline": axis.baseline,
                    "relation": axis.relation.value,
                }
                for axis in axes
            ],
            "target_backend": definition.target_backend,
            "control_backends": definition.control_backends,
            "constructor": registration.builder,
            "oracle_relation": _oracle_relation(comparison_view),
        }
        template_id = stable_digest("osc-target-template-id", declaration_payload)
        evidence_id = (
            definition.root_id
            if provenance == ProvenanceClass.KNOWN_REGRESSION
            else stable_digest(
                "osc-fresh-target-evidence",
                {
                    "declaration": definition.root_id,
                    "mechanism": definition.mechanism_id,
                },
            )
        )
        templates.append(
            TargetTemplate(
                template_id=template_id,
                test_family_id=registration.family_id,
                provenance_class=provenance,
                semantic_dimensions=tuple(definition.semantic_dimensions),
                axes=axes,
                required_all_atoms=_capability_atoms(
                    definition.required_capabilities
                ),
                required_any_atom_groups=(),
                forbidden_atoms=frozenset(),
                target_backend=definition.target_backend,
                control_backends=tuple(definition.control_backends),
                required_capabilities=frozenset(
                    str(item) for item in definition.required_capabilities
                ),
                risk_class=definition.mechanism_id,
                priority=int(definition.priority),
                cost_class=(
                    "high"
                    if len(definition.control_backends) >= 4
                    else "medium"
                    if len(definition.control_backends) >= 2
                    else "low"
                ),
                constructor_provider=f"legacy_adapter:{registration.builder}",
                oracle_relation=_oracle_relation(comparison_view),
                provenance_evidence_id=evidence_id,
            )
        )
    return tuple(sorted(templates, key=lambda item: item.template_id))


def legacy_v4_target_templates() -> tuple[TargetTemplate, ...]:
    """Return the physically separated fresh and regression declaration views."""

    # Lazy imports avoid making either physical registry depend on the other.
    from .fresh_registry import fresh_target_templates
    from .regression_registry import regression_target_templates

    templates = (*fresh_target_templates(), *regression_target_templates())
    if len({item.template_id for item in templates}) != len(templates):
        raise ValueError("target registry contains duplicate template identity")
    return tuple(sorted(templates, key=lambda item: item.template_id))


def partition_templates(
    templates: Iterable[TargetTemplate],
) -> tuple[tuple[TargetTemplate, ...], tuple[TargetTemplate, ...]]:
    materialized = tuple(templates)
    fresh = tuple(sorted((
        item
        for item in materialized
        if item.provenance_class == ProvenanceClass.FRESH_DISCOVERY
    ), key=lambda item: item.template_id))
    regression = tuple(sorted((
        item
        for item in materialized
        if item.provenance_class == ProvenanceClass.KNOWN_REGRESSION
    ), key=lambda item: item.template_id))
    if len(fresh) + len(regression) != len(materialized):
        raise ValueError("only fresh and regression declarations enter v1 registry")
    if len({item.template_id for item in materialized}) != len(materialized):
        raise ValueError("target registry contains duplicate template identity")
    return fresh, regression


def declarations_digest(templates: Iterable[TargetTemplate]) -> str:
    return stable_digest(
        "osc-target-declarations",
        tuple(sorted((item.digest for item in templates))),
    )


def with_constructor_provider(
    template: TargetTemplate, provider: str
) -> TargetTemplate:
    """Small helper for tests and future native-constructor migration."""

    return replace(template, constructor_provider=str(provider))
