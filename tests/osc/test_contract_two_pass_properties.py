from __future__ import annotations

from datadiff_osc.contract_engine import (
    AbstractState,
    EndpointRequirement,
    Observation,
    ObservationDemand,
    ProgramSemantics,
    SemanticStep,
    VerdictKind,
    compile_hypercontract,
    default_rule_registry,
    evaluate_applicability,
)
from datadiff_osc.contract_engine.demand import backward_analyze
from datadiff_osc.contract_engine.domains import MultiplicityDomain, OrderDomain
from datadiff_osc.contract_engine.monitor import monitor_exact
from datadiff_osc.contract_engine.overlays import BackendOverlay


def _requirements():
    return (EndpointRequirement("a"), EndpointRequirement("b"))


def _program(*steps):
    return ProgramSemantics(
        "two-pass-program",
        AbstractState.initial(schema=(("x", "int", False),), row_count=2),
        tuple(steps),
    )


def test_backward_trace_is_bound_to_node_local_forward_state():
    compiled = compile_hypercontract(
        _program(
            SemanticStep.build(step_id="distinct", kind="distinct"),
            SemanticStep.build(step_id="sort", kind="sort", arguments={"total_order": True}),
        ),
        endpoint_requirements=_requirements(),
        relation_id="sequence_equal",
    )
    assert tuple(item.forward_state_digest for item in compiled.backward.traces) == (
        compiled.forward.states[1].digest,
        compiled.forward.states[0].digest,
    )
    assert all(item.forward_state_digest for item in compiled.backward.traces)


def test_distinct_only_discharges_duplicate_demand_with_proven_forward_state():
    registry = default_rule_registry()
    transformer = registry.resolve("operation", "distinct")
    step = SemanticStep.build(step_id="distinct", kind="distinct")
    demand = ObservationDemand(duplicate_multiplicity=True)
    unknown = transformer.backward(demand, step, AbstractState())
    proven = transformer.backward(
        demand,
        step,
        AbstractState(multiplicity=MultiplicityDomain.SET_OUTPUT),
    )
    assert unknown.duplicate_multiplicity is True
    assert proven.duplicate_multiplicity is False


def test_sort_does_not_discharge_order_under_unknown_forward_state():
    transformer = default_rule_registry().resolve("operation", "sort")
    step = SemanticStep.build(step_id="sort", kind="sort")
    demand = ObservationDemand(presentation_order=True)
    unknown = transformer.backward(demand, step, AbstractState())
    proven = transformer.backward(
        demand,
        step,
        AbstractState(presentation_order=OrderDomain.TOTAL),
    )
    assert unknown.presentation_order is True
    assert proven.presentation_order is False


def test_misaligned_forward_states_fail_closed():
    analysis = backward_analyze(
        ObservationDemand(),
        (SemanticStep.build(step_id="select", kind="select"),),
        default_rule_registry(),
        forward_states=(AbstractState(), AbstractState()),
    )
    assert "forward_state_alignment" in analysis.unresolved_rules


def test_unknown_relation_parameter_invalidates_derivation():
    compiled = compile_hypercontract(
        _program(SemanticStep.build(step_id="select", kind="select")),
        endpoint_requirements=_requirements(),
        relation_id="partial_order_equal",
        relation_parameters={"key_indices": [0], "global_tie_freedom": True},
    )
    assert not compiled.valid
    assert any("unknown_parameter" in item for item in compiled.derivation.unresolved_facts)


def test_semantic_permission_is_component_relation_and_overlay_bound():
    overlay = BackendOverlay(
        overlay_id="local-permission",
        backend="backend",
        version_spec="*",
        adapter_revision="adapter",
        exact_preconditions=("fact:exact",),
        affected_component="numeric",
        permitted_relation="numeric_exact",
        evidence_source="frozen:test",
        expiry_policy="revalidate",
    )
    permission = overlay.permission()
    program = _program(SemanticStep.build(step_id="select", kind="select"))
    valid = compile_hypercontract(
        program,
        endpoint_requirements=_requirements(),
        relation_id="numeric_exact",
        overlay_digests=(overlay.digest,),
        semantic_permissions=(permission,),
        permission_facts=frozenset({"fact:exact"}),
    )
    assert valid.valid
    wrong_relation = compile_hypercontract(
        program,
        endpoint_requirements=_requirements(),
        relation_id="bag_equal",
        overlay_digests=(overlay.digest,),
        semantic_permissions=(permission,),
        permission_facts=frozenset({"fact:exact"}),
    )
    assert not wrong_relation.valid
    assert any("permission_relation_out_of_scope" in item for item in wrong_relation.derivation.unresolved_facts)


def test_containment_compilation_does_not_inject_cardinality_equality():
    compiled = compile_hypercontract(
        _program(SemanticStep.build(step_id="select", kind="select")),
        endpoint_requirements=_requirements(),
        relation_id="containment",
    )

    relation_ids = tuple(item.relation_id for item in compiled.contract.obligations)
    assert "containment" in relation_ids
    assert "cardinality_equal" not in relation_ids
    assert compiled.valid


def test_compiled_containment_accepts_strict_bag_containment(
    endpoints, int_schema
):
    requirements = tuple(
        EndpointRequirement(endpoint.endpoint_id) for endpoint in endpoints
    )
    compiled = compile_hypercontract(
        _program(SemanticStep.build(step_id="select", kind="select")),
        endpoint_requirements=requirements,
        relation_id="containment",
    )
    applicability = evaluate_applicability(
        compiled.contract,
        endpoints,
        facts=frozenset(compiled.contract.preconditions),
    )
    observations = (
        Observation.build(
            endpoint_id=endpoints[0].endpoint_id,
            status="ok",
            schema=int_schema,
            rows=[[1]],
            execution_metadata={"endpoint_digest": endpoints[0].digest},
        ),
        Observation.build(
            endpoint_id=endpoints[1].endpoint_id,
            status="ok",
            schema=int_schema,
            rows=[[1], [2]],
            execution_metadata={"endpoint_digest": endpoints[1].digest},
        ),
    )

    assert (
        monitor_exact(
            compiled.contract,
            observations,
            applicability,
            endpoints=endpoints,
        ).kind
        == VerdictKind.SATISFIED
    )


def test_explicit_status_relation_compiles_without_value_obligations():
    compiled = compile_hypercontract(
        _program(SemanticStep.build(step_id="select", kind="select")),
        endpoint_requirements=_requirements(),
        relation_id="status_ok",
    )

    assert compiled.contract.observations == ("status",)
    assert tuple(item.relation_id for item in compiled.contract.obligations) == (
        "status_ok",
    )
    assert compiled.valid
