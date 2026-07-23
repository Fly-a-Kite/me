from __future__ import annotations

from datadiff_osc.contract_engine import Observation, VerdictKind, evaluate_applicability
from datadiff_osc.contract_engine.compatibility import (
    V1_COMPARISON_VIEWS,
    compile_v1_profile_facade,
    v1_facade_payload,
    v1_profile_relation,
)
from datadiff_osc.contract_engine.model import EndpointRequirement, SchemaField
from datadiff_osc.contract_engine.monitor import monitor_exact


def _requirements(endpoints):
    return tuple(EndpointRequirement(item.endpoint_id) for item in endpoints)


def test_all_five_v1_views_compile_without_legacy_authority(simple_compiled, endpoints):
    profiles = {
        "exact": {"view": "exact"},
        "ordered_value": {"view": "ordered_value"},
        "bag_value": {"view": "bag_value"},
        "numeric_tolerant": {
            "view": "numeric_tolerant",
            "osc_numeric_parameters": {"abs_tol": "0.000001", "rel_tol": 0, "ulp_tol": 0},
        },
        "error_equivalent": {"view": "error_equivalent"},
    }
    assert tuple(profiles) == V1_COMPARISON_VIEWS
    for view, profile in profiles.items():
        compiled = compile_v1_profile_facade(
            # The facade consumes OSC ProgramSemantics, never legacy verdict code.
            _program_from(simple_compiled),
            profile,
            _requirements(endpoints),
        )
        assert compiled.valid, view
        assert v1_facade_payload(compiled)["comparison_view"] == view


def _program_from(compiled):
    from datadiff_osc.contract_engine import ProgramSemantics, SemanticStep

    return ProgramSemantics(
        compiled.derivation.program_digest + "-v1",
        compiled.forward.initial_state,
        (SemanticStep.build(step_id="select", kind="select"),),
    )


def test_exact_preserves_dtype_while_ordered_value_does_not_add_dtype_authority(
    simple_compiled, endpoints
):
    program = _program_from(simple_compiled)
    exact = compile_v1_profile_facade(program, {"view": "exact"}, _requirements(endpoints))
    ordered = compile_v1_profile_facade(
        program, {"view": "ordered_value"}, _requirements(endpoints)
    )
    assert exact.backward.final_demand.schema_types is True
    assert ordered.backward.final_demand.schema_types is False
    assert ordered.backward.final_demand.presentation_order is True


def test_numeric_v1_view_is_tolerant_bag_with_exact_matching(
    simple_compiled, endpoints
):
    compiled = compile_v1_profile_facade(
        _program_from(simple_compiled),
        {
            "view": "numeric_tolerant",
            "osc_numeric_parameters": {"abs_tol": "0.01", "rel_tol": 0, "ulp_tol": 0},
        },
        _requirements(endpoints),
    )
    applicability = evaluate_applicability(
        compiled.contract,
        endpoints,
        facts=frozenset(compiled.contract.preconditions),
    )
    schema = (SchemaField("x", "float", False),)
    observations = (
        Observation.build(
            endpoint_id="left", status="ok", schema=schema,
            rows=[[1.0], [2.0]],
            execution_metadata={"endpoint_digest": endpoints[0].digest},
        ),
        Observation.build(
            endpoint_id="right", status="ok", schema=schema,
            rows=[[2.005], [1.005]],
            execution_metadata={"endpoint_digest": endpoints[1].digest},
        ),
    )
    result = monitor_exact(
        compiled.contract,
        observations,
        applicability,
        endpoints=endpoints,
    )
    assert result.kind == VerdictKind.SATISFIED
    assert dict(compiled.contract.obligations[-1].parameters)["collection"] == "bag"
    assert "bag_bipartite_matching" in v1_facade_payload(compiled)["precision_fixes"]


def test_v1_decimal_rounding_is_never_silently_reauthorized():
    try:
        v1_profile_relation(
            {"view": "numeric_tolerant", "osc_numeric_parameters": {}}
        )
    except ValueError as exc:
        assert "explicit" in str(exc)
    else:
        raise AssertionError("empty numeric precision unexpectedly accepted")
