from __future__ import annotations

from dataclasses import replace

import pytest

from datadiff.dsl import Case, ColumnSpec, Program, TableData
from datadiff_osc.contract_engine.compatibility import (
    compile_v1_profile_facade,
    program_semantics_from_v1_case,
    v1_facade_payload,
    v1_profile_relation,
)
from datadiff_osc.contract_engine.model import EndpointRequirement
from datadiff_osc.contract_engine.mutations import (
    COMPARATOR_WEAKENING_MUTANTS,
    HYPEREDGE_MUTANTS,
    audit_mutants,
    mutate_contract,
)


def test_v1_facade_maps_structured_views_and_refuses_implicit_rounding():
    assert v1_profile_relation({"view": "ordered_value"}) == ("sequence_equal", {})
    assert v1_profile_relation({"view": "bag_value"}) == ("bag_equal", {})
    assert v1_profile_relation({"view": "error_equivalent"}) == ("error_category_equal", {})
    with pytest.raises(ValueError, match="explicit OSC"):
        v1_profile_relation({"view": "numeric_tolerant", "numeric_decimals": 10})


def test_v1_case_bridge_preserves_running_sum_internal_order_precision_fix():
    case = Case(
        "case-running-join",
        1,
        [
            TableData("t0", [ColumnSpec("id", "int"), ColumnSpec("x", "float")], [{"id": 1, "x": 1.0}]),
            TableData("t1", [ColumnSpec("id", "int"), ColumnSpec("y", "str")], [{"id": 1, "y": "a"}]),
        ],
        Program(
            "program-running-join",
            1,
            [
                {"op": "running_sum", "source": "x", "column": "run_x", "order_by": [{"column": "id", "ascending": True, "nulls": "last"}]},
                {"op": "join", "table": "t1", "left_on": "id", "right_on": "id", "how": "inner"},
            ],
        ),
    )
    semantics = program_semantics_from_v1_case(case)
    compiled = compile_v1_profile_facade(
        semantics,
        {"view": "bag_value"},
        (EndpointRequirement("a"), EndpointRequirement("b")),
    )
    payload = v1_facade_payload(compiled)

    assert semantics.steps[0].kind == "running_sum"
    assert semantics.steps[1].kind == "join"
    assert compiled.forward.final_state.presentation_order.value == "none"
    assert compiled.contract.obligations[-1].relation_id == "bag_equal"
    assert payload["boundary_axes"] == []


def test_mutant_audit_records_killed_and_survived(simple_compiled):
    result = audit_mutants(
        "smoke",
        ("ordered_to_bag", "logical_dtype_ignored"),
        lambda mutant: mutant == "ordered_to_bag",
    )
    assert result.kill_rate == 0.5
    assert result.killed_ids == ("ordered_to_bag",)
    assert result.survived_ids == ("logical_dtype_ignored",)


@pytest.mark.parametrize(
    "mutant",
    [
        "ordered_to_bag", "bag_to_set", "logical_dtype_ignored",
        "column_name_order_ignored", "error_category_collapsed",
        "partial_order_freedom_globalized", "numeric_tolerance_expanded",
        "layout_relation_ignores_values", "presentation_evaluation_order_confused",
    ],
)
def test_executable_comparator_mutants_change_contract_digest(simple_compiled, mutant):
    mutated = mutate_contract(simple_compiled.contract, mutant)
    assert mutated.digest != simple_compiled.contract.digest


def test_mutation_palettes_are_frozen_and_complete():
    assert len(COMPARATOR_WEAKENING_MUTANTS) == 12
    assert len(HYPEREDGE_MUTANTS) == 7

