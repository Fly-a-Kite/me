from __future__ import annotations

import pytest

from datadiff_osc.contract_engine import (
    AbstractState,
    Endpoint,
    EndpointRequirement,
    ProgramSemantics,
    SchemaField,
    SemanticStep,
    compile_hypercontract,
    evaluate_applicability,
)
from datadiff_osc.contract_engine.domains import LayoutDomain


@pytest.fixture
def endpoints() -> tuple[Endpoint, Endpoint]:
    capabilities = {"op:select", "op:sort", "op:limit", "type:int"}
    return (
        Endpoint.build(
            endpoint_id="left",
            case_digest="case-digest",
            backend="left_backend",
            backend_version="1.0.0",
            adapter_revision="adapter-v1",
            execution_mode="eager",
            physical_layout="contiguous",
            capabilities=capabilities,
        ),
        Endpoint.build(
            endpoint_id="right",
            case_digest="case-digest",
            backend="right_backend",
            backend_version="2.0.0",
            adapter_revision="adapter-v1",
            execution_mode="eager",
            physical_layout="contiguous",
            capabilities=capabilities,
        ),
    )


@pytest.fixture
def simple_compiled(endpoints):
    program = ProgramSemantics(
        "program-simple",
        AbstractState.initial(
            schema=(("x", "int", False),),
            row_count=2,
            layout=LayoutDomain.CONTIGUOUS,
        ),
        (SemanticStep.build(step_id="select", kind="select"),),
    )
    requirements = tuple(
        EndpointRequirement(
            endpoint_id=item.endpoint_id,
            backend=item.backend,
            version_spec="*",
            required_capabilities=frozenset({"op:select"}),
        )
        for item in endpoints
    )
    return compile_hypercontract(
        program,
        endpoint_requirements=requirements,
        relation_id="bag_equal",
    )


@pytest.fixture
def simple_applicability(simple_compiled, endpoints):
    return evaluate_applicability(
        simple_compiled.contract,
        endpoints,
        facts=frozenset(simple_compiled.contract.preconditions),
        unresolved_rules=simple_compiled.derivation.unresolved_facts,
    )


@pytest.fixture
def int_schema() -> tuple[SchemaField, ...]:
    return (SchemaField("x", "int", False),)

