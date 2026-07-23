from __future__ import annotations

from math import prod

import pytest

from datadiff_osc.semantic_targets.declarations import (
    legacy_v4_target_templates,
    partition_templates,
)
from datadiff_osc.semantic_targets.fresh_registry import fresh_target_templates
from datadiff_osc.semantic_targets.model import ProvenanceClass
from datadiff_osc.semantic_targets.regression_registry import (
    regression_target_templates,
)


def _cell_count(templates):
    return sum(prod(len(axis.values) for axis in item.axes) for item in templates)


def _obligation_count(templates):
    return sum(item.cell_count * len(item.control_backends) for item in templates)


def test_physical_registry_views_are_typed_disjoint_and_declaration_derived():
    fresh = fresh_target_templates()
    regression = regression_target_templates()
    assert (len(fresh), _cell_count(fresh), _obligation_count(fresh)) == (16, 232, 502)
    assert (len(regression), _cell_count(regression), _obligation_count(regression)) == (
        9,
        144,
        228,
    )
    assert {item.template_id for item in fresh}.isdisjoint(
        item.template_id for item in regression
    )
    assert {item.provenance_class for item in fresh} == {
        ProvenanceClass.FRESH_DISCOVERY
    }
    assert {item.provenance_class for item in regression} == {
        ProvenanceClass.KNOWN_REGRESSION
    }
    assert partition_templates(legacy_v4_target_templates()) == (fresh, regression)


def test_fresh_registry_rejects_known_root_markers(monkeypatch):
    import datadiff_osc.semantic_targets.declarations as declarations

    original = declarations.semantic_family_v3_definitions

    def contaminated():
        values = list(original())
        index = next(i for i, item in enumerate(values) if item.source == "coverage_expansion")
        from dataclasses import replace

        values[index] = replace(values[index], mechanism_id="known-root-replay")
        return tuple(values)

    monkeypatch.setattr(declarations, "semantic_family_v3_definitions", contaminated)
    with pytest.raises(ValueError, match="known-root provenance"):
        declarations._legacy_templates_for(ProvenanceClass.FRESH_DISCOVERY)


def test_fresh_registry_rejects_known_root_marker_in_root_id(monkeypatch):
    import datadiff_osc.semantic_targets.declarations as declarations

    original = declarations.semantic_family_v3_definitions

    def contaminated():
        values = list(original())
        index = next(i for i, item in enumerate(values) if item.source == "coverage_expansion")
        from dataclasses import replace

        values[index] = replace(
            values[index], root_id="target-family-known-root-replay"
        )
        return tuple(values)

    monkeypatch.setattr(declarations, "semantic_family_v3_definitions", contaminated)
    with pytest.raises(ValueError, match="known-root provenance"):
        declarations._legacy_templates_for(ProvenanceClass.FRESH_DISCOVERY)


def test_registry_rejects_cross_partition_root_identity(monkeypatch):
    import datadiff_osc.semantic_targets.declarations as declarations

    original = declarations.semantic_family_v3_definitions

    def contaminated():
        values = list(original())
        fresh = next(item for item in values if item.source == "coverage_expansion")
        index = next(i for i, item in enumerate(values) if item.source == "confirmed_root")
        from dataclasses import replace

        values[index] = replace(values[index], root_id=fresh.root_id)
        return tuple(values)

    monkeypatch.setattr(declarations, "semantic_family_v3_definitions", contaminated)
    with pytest.raises(ValueError, match="share root identity"):
        declarations._legacy_templates_for(ProvenanceClass.FRESH_DISCOVERY)


def test_fresh_registry_rejects_regression_constructor_provider(monkeypatch):
    import datadiff_osc.semantic_targets.declarations as declarations

    original = declarations.latest_family_witness_registrations

    def contaminated():
        values = list(original())
        definitions = {
            item.family_id: item
            for item in declarations.semantic_family_v3_definitions()
        }
        index = next(
            i
            for i, item in enumerate(values)
            if definitions[item.family_id].source == "coverage_expansion"
        )
        from dataclasses import replace

        values[index] = replace(
            values[index], builder="issue-replay.known-root-constructor"
        )
        return tuple(values)

    monkeypatch.setattr(
        declarations, "latest_family_witness_registrations", contaminated
    )
    with pytest.raises(ValueError, match="known-root provenance"):
        declarations._legacy_templates_for(ProvenanceClass.FRESH_DISCOVERY)


def test_fresh_root_identity_participates_in_template_hash(monkeypatch):
    import datadiff_osc.semantic_targets.declarations as declarations

    original_provider = declarations.semantic_family_v3_definitions
    original_definitions = original_provider()
    selected = next(
        item for item in original_definitions if item.source == "coverage_expansion"
    )
    original_template = next(
        item
        for item in declarations._legacy_templates_for(
            ProvenanceClass.FRESH_DISCOVERY
        )
        if item.test_family_id == selected.family_id
    )

    def renamed():
        from dataclasses import replace

        return tuple(
            replace(item, root_id="target-family-legitimate-renamed")
            if item.family_id == selected.family_id
            else item
            for item in original_definitions
        )

    monkeypatch.setattr(declarations, "semantic_family_v3_definitions", renamed)
    changed_template = next(
        item
        for item in declarations._legacy_templates_for(
            ProvenanceClass.FRESH_DISCOVERY
        )
        if item.test_family_id == selected.family_id
    )
    assert changed_template.template_id != original_template.template_id


def test_registry_is_declaration_only_and_never_imports_runtime_evaluators():
    import inspect
    import datadiff_osc.semantic_targets.declarations as declarations

    source = inspect.getsource(declarations)
    assert "evaluate_semantic_activation" not in source
    assert "semantic_core.activation" not in source
