from __future__ import annotations

import pytest

from datadiff_osc.contract_engine import (
    AGGREGATE_KINDS,
    EXPRESSION_KINDS,
    OPERATION_KINDS,
    AbstractState,
    EndpointRequirement,
    ProgramSemantics,
    SemanticStep,
    compile_hypercontract,
    default_rule_registry,
)
from datadiff_osc.contract_engine.domains import (
    LayoutDomain,
    MultiplicityDomain,
    OrderDomain,
)


def _program(*steps: SemanticStep) -> ProgramSemantics:
    return ProgramSemantics(
        "program-analysis",
        AbstractState.initial(
            schema=(("id", "int", False), ("x", "float", True)),
            has_nulls=True,
            row_count=3,
            layout=LayoutDomain.CONTIGUOUS,
        ),
        tuple(steps),
    )


def _requirements() -> tuple[EndpointRequirement, ...]:
    return (EndpointRequirement("a"), EndpointRequirement("b"))


def test_rule_registry_exactly_covers_frozen_universe():
    registry = default_rule_registry()

    assert len(registry.operation) == 21
    assert len(registry.expression) == 21
    assert len(registry.aggregate) == 8
    assert tuple(item.subject for item in registry.operation) == OPERATION_KINDS
    assert tuple(item.subject for item in registry.expression) == EXPRESSION_KINDS
    assert tuple(item.subject for item in registry.aggregate) == AGGREGATE_KINDS


def test_unknown_operation_fails_closed_with_invalid_derivation():
    compiled = compile_hypercontract(
        _program(SemanticStep.build(step_id="unknown", kind="mystery_op")),
        endpoint_requirements=_requirements(),
        relation_id="bag_equal",
    )

    assert compiled.valid is False
    assert compiled.derivation.unresolved_facts == ("operation:mystery_op",)
    assert compiled.forward.final_state.definedness.value == "unknown"


def test_running_sum_evaluation_order_is_separate_from_final_join_presentation():
    compiled = compile_hypercontract(
        _program(
            SemanticStep.build(
                step_id="running",
                kind="running_sum",
                arguments={"total_order": True, "partition_by": ["id"]},
            ),
            SemanticStep.build(step_id="join", kind="join"),
        ),
        endpoint_requirements=_requirements(),
        relation_id="bag_equal",
    )

    state = compiled.forward.final_state
    assert state.presentation_order == OrderDomain.NONE
    assert state.evaluation_order == OrderDomain.UNKNOWN
    assert compiled.backward.input_demand.numeric is True
    assert compiled.backward.final_demand.presentation_order is False
    assert compiled.valid is True


def test_final_sort_reestablishes_presentation_order_after_join():
    compiled = compile_hypercontract(
        _program(
            SemanticStep.build(step_id="running", kind="running_sum", arguments={"total_order": True}),
            SemanticStep.build(step_id="join", kind="join"),
            SemanticStep.build(step_id="sort", kind="sort", arguments={"total_order": True}),
        ),
        endpoint_requirements=_requirements(),
        relation_id="sequence_equal",
    )

    assert compiled.forward.final_state.presentation_order == OrderDomain.TOTAL
    assert compiled.backward.final_demand.presentation_order is True
    assert compiled.backward.input_demand.presentation_order is False


def test_distinct_requires_uniqueness_but_kills_input_duplicate_demand():
    compiled = compile_hypercontract(
        _program(SemanticStep.build(step_id="distinct", kind="distinct")),
        endpoint_requirements=_requirements(),
        relation_id="set_equal_unique",
    )

    assert compiled.forward.final_state.multiplicity == MultiplicityDomain.SET_OUTPUT
    application = compiled.forward.applications[0]
    assert {"set_membership", "output_uniqueness", "exact_cardinality"} <= set(
        application.obligations
    )
    assert compiled.backward.input_demand.duplicate_multiplicity is False


@pytest.mark.parametrize("kind", EXPRESSION_KINDS)
def test_every_expression_rule_has_forward_and_backward_trace(kind):
    compiled = compile_hypercontract(
        _program(
            SemanticStep.build(
                step_id=f"expr-{kind}",
                kind="mutate",
                arguments={"domain_proven": True},
                expression_kinds=(kind,),
            )
        ),
        endpoint_requirements=_requirements(),
        relation_id="bag_equal",
    )
    assert compiled.valid, kind
    assert any(item.subject == kind for item in compiled.forward.applications)


@pytest.mark.parametrize("kind", AGGREGATE_KINDS)
def test_every_aggregate_rule_has_forward_and_backward_trace(kind):
    compiled = compile_hypercontract(
        _program(
            SemanticStep.build(
                step_id=f"agg-{kind}", kind="aggregate", aggregate_kinds=(kind,)
            )
        ),
        endpoint_requirements=_requirements(),
        relation_id="bag_equal",
    )
    assert compiled.valid, kind
    assert any(item.subject == kind for item in compiled.forward.applications)

