from __future__ import annotations

from dataclasses import replace
from math import prod

import pytest

from datadiff_osc.semantic_targets.compiler import compile_target_universe
from datadiff_osc.semantic_targets.declarations import legacy_v4_target_templates
from datadiff_osc.semantic_targets.model import ProvenanceClass


def _baseline_star_edges(template):
    return sum(
        (len(axis.values) - 1)
        * prod(len(other.values) for other in template.axes if other.name != axis.name)
        for axis in template.axes
    )


def test_compiler_recomputes_exact_typed_denominators_without_magic_counts():
    templates = legacy_v4_target_templates()
    universe = compile_target_universe(templates)
    fresh = [item for item in templates if item.provenance_class == ProvenanceClass.FRESH_DISCOVERY]
    assert len(universe.fresh_cells) == sum(item.cell_count for item in fresh) == 232
    assert len(universe.regression_cells) == sum(
        item.cell_count for item in templates if item not in fresh
    ) == 144
    assert len(universe.fresh_edges) == sum(_baseline_star_edges(item) for item in fresh) == 384
    assert len(universe.fresh_backend_pair_obligations) == sum(
        item.cell_count * len(item.control_backends) for item in fresh
    ) == 502
    assert len(universe.regression_backend_pair_obligations) == 228
    assert {item.obligation_id for item in universe.fresh_backend_pair_obligations}.isdisjoint(
        item.obligation_id for item in universe.regression_backend_pair_obligations
    )


def test_compilation_is_order_independent_and_edges_change_exactly_one_axis():
    templates = legacy_v4_target_templates()
    left = compile_target_universe(templates)
    right = compile_target_universe(reversed(templates))
    assert left.digest == right.digest
    cells = {item.target_cell_id: item for item in left.fresh_cells}
    for edge in left.fresh_edges:
        base = cells[edge.base_cell_id].coordinate_map
        sibling = cells[edge.sibling_cell_id].coordinate_map
        changed = [name for name in base if base[name] != sibling[name]]
        assert changed == [edge.changed_axis]


def test_unknown_pipeline_fails_closed():
    template = next(
        item for item in legacy_v4_target_templates()
        if any(axis.name == "pipeline" for axis in item.axes)
    )
    axes = tuple(
        replace(axis, values=("unknown_pipeline",), baseline="unknown_pipeline")
        if axis.name == "pipeline"
        else axis
        for axis in template.axes
    )
    with pytest.raises(ValueError, match="unknown target pipeline"):
        compile_target_universe((replace(template, axes=axes),))


def test_composite_axes_require_observable_operation_atoms():
    universe = compile_target_universe(legacy_v4_target_templates())
    by_coordinates = [
        item for item in universe.fresh_cells
        if item.coordinate_map.get("aggregate_pair") == "count_sum"
    ]
    assert by_coordinates
    assert all({"op:aggregate", "agg:count", "agg:sum"} <= item.required_all_atoms for item in by_coordinates)
