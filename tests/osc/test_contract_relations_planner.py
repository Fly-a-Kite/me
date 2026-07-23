from __future__ import annotations

from dataclasses import replace

import pytest

from datadiff_osc.contract_engine import (
    Observation,
    VerdictKind,
    evaluate_applicability,
    execute_staged_comparison,
)
from datadiff_osc.contract_engine.capability import decide_capabilities
from datadiff_osc.contract_engine.hypergraph import default_entailment_graph
from datadiff_osc.contract_engine.model import RelationObligation
from datadiff_osc.contract_engine.monitor import monitor_exact
from datadiff_osc.contract_engine.overlays import BackendOverlay, OverlayRegistry
from datadiff_osc.contract_engine.planner import cluster_fingerprints
from datadiff_osc.contract_engine.relations import (
    default_relation_registry,
    validate_relation_obligation,
)
from datadiff_osc.schemas import (
    ExecutionStatus,
    FailureKind,
    StructuredExecutionOutcome,
)


def _observations(schema, right_rows=((2,), (1,)), *, endpoints):
    return (
        Observation.build(
            endpoint_id="left", status="ok", schema=schema, rows=[[1], [2]],
            execution_metadata={"endpoint_digest": endpoints[0].digest},
        ),
        Observation.build(
            endpoint_id="right", status="ok", schema=schema, rows=right_rows,
            execution_metadata={"endpoint_digest": endpoints[1].digest},
        ),
    )


def _ok_outcomes():
    return (
        StructuredExecutionOutcome("left", ExecutionStatus.OK, FailureKind.NONE),
        StructuredExecutionOutcome("right", ExecutionStatus.OK, FailureKind.NONE),
    )


def test_bag_satisfied_and_sequence_violated(
    simple_compiled, simple_applicability, endpoints, int_schema
):
    observations = _observations(int_schema, endpoints=endpoints)
    bag = monitor_exact(
        simple_compiled.contract,
        observations,
        simple_applicability,
        endpoints=endpoints,
    )

    value = simple_compiled.contract.obligations[-1]
    ordered_contract = replace(
        simple_compiled.contract,
        contract_id="ordered-contract",
        obligations=(*simple_compiled.contract.obligations[:-1], replace(value, relation_id="sequence_equal")),
    )
    ordered = monitor_exact(
        ordered_contract,
        observations,
        replace(simple_applicability, contract_digest=ordered_contract.digest),
        endpoints=endpoints,
    )

    assert bag.kind == VerdictKind.SATISFIED
    assert ordered.kind == VerdictKind.VIOLATED


def test_bag_relation_detects_multiplicity(
    simple_compiled, simple_applicability, endpoints, int_schema
):
    observations = _observations(
        int_schema, right_rows=((1,), (1,), (2,)), endpoints=endpoints
    )
    assert (
        monitor_exact(
            simple_compiled.contract,
            observations,
            simple_applicability,
            endpoints=endpoints,
        ).kind
        == VerdictKind.VIOLATED
    )


def test_set_relation_detects_duplicate_output(
    simple_compiled, simple_applicability, endpoints, int_schema
):
    value = replace(simple_compiled.contract.obligations[-1], relation_id="set_equal_unique")
    contract = replace(simple_compiled.contract, contract_id="set-contract", obligations=(*simple_compiled.contract.obligations[:-1], value))
    applicability = replace(simple_applicability, contract_digest=contract.digest)
    observations = _observations(
        int_schema, right_rows=((1,), (1,), (2,)), endpoints=endpoints
    )
    result = monitor_exact(
        contract, observations, applicability, endpoints=endpoints
    )

    assert result.kind == VerdictKind.VIOLATED
    assert "duplicate" in result.components[-1].reason


def test_staged_planner_screens_equal_and_exact_escalates_mismatch(
    simple_compiled, simple_applicability, endpoints, int_schema
):
    equal = _observations(int_schema, endpoints=endpoints)
    screened = execute_staged_comparison(
        simple_compiled.contract,
        equal,
        simple_applicability,
        endpoints=endpoints,
        execution_outcomes=_ok_outcomes(),
    )
    mismatch = _observations(
        int_schema, right_rows=((1,), (3,)), endpoints=endpoints
    )
    escalated = execute_staged_comparison(
        simple_compiled.contract,
        mismatch,
        simple_applicability,
        endpoints=endpoints,
        execution_outcomes=_ok_outcomes(),
    )
    authority = monitor_exact(
        simple_compiled.contract,
        mismatch,
        simple_applicability,
        endpoints=endpoints,
    )

    assert screened.verdict.kind == VerdictKind.SATISFIED
    assert screened.exact_escalated is False
    assert screened.materialized_endpoint_ids == ()
    assert escalated.exact_escalated is True
    assert escalated.verdict.kind == authority.kind == VerdictKind.VIOLATED


def test_schema_difference_cannot_screen_as_equal(
    simple_compiled, simple_applicability, endpoints, int_schema
):
    different_schema = (replace(int_schema[0], logical_type="float"),)
    observations = (
        Observation.build(
            endpoint_id="left", status="ok", schema=int_schema, rows=[[1]],
            execution_metadata={"endpoint_digest": endpoints[0].digest},
        ),
        Observation.build(
            endpoint_id="right", status="ok", schema=different_schema, rows=[[1]],
            execution_metadata={"endpoint_digest": endpoints[1].digest},
        ),
    )
    result = execute_staged_comparison(
        simple_compiled.contract,
        observations,
        simple_applicability,
        endpoints=endpoints,
        execution_outcomes=_ok_outcomes(),
    )
    assert result.exact_escalated is True
    assert result.verdict.kind == VerdictKind.VIOLATED


def test_fresh_confirmation_and_native_reproduction_forbid_cache(
    simple_compiled, simple_applicability, endpoints, int_schema
):
    with pytest.raises(ValueError, match="forbidden"):
        execute_staged_comparison(
            simple_compiled.contract,
            _observations(int_schema, endpoints=endpoints),
            simple_applicability,
            evidence_tier="fresh_confirmation",
            endpoints=endpoints,
            cache_used=True,
            execution_outcomes=_ok_outcomes(),
        )


def test_capability_unsupported_is_inapplicable_not_satisfied(simple_compiled, endpoints):
    restricted = replace(
        simple_compiled.contract.endpoint_requirements[0],
        required_capabilities=frozenset({"op:missing"}),
    )
    contract = replace(
        simple_compiled.contract,
        contract_id="unsupported-contract",
        endpoint_requirements=(restricted, *simple_compiled.contract.endpoint_requirements[1:]),
    )
    certificate = evaluate_applicability(
        contract, endpoints, facts=frozenset(contract.preconditions)
    )

    assert certificate.verdict == VerdictKind.INAPPLICABLE
    assert certificate.unsupported_evidence
    assert all(item.missing_capability == "op:missing" for item in certificate.unsupported_evidence)


def test_timeout_is_inconclusive_and_never_unsupported(
    simple_compiled, simple_applicability, endpoints, int_schema
):
    observations = (
        Observation.build(
            endpoint_id="left", status="ok", schema=int_schema, rows=[[1]],
            execution_metadata={"endpoint_digest": endpoints[0].digest},
        ),
        Observation.build(
            endpoint_id="right", status="timeout", schema=int_schema,
            execution_metadata={"endpoint_digest": endpoints[1].digest},
        ),
    )
    result = monitor_exact(
        simple_compiled.contract,
        observations,
        simple_applicability,
        endpoints=endpoints,
    )
    assert result.kind == VerdictKind.INCONCLUSIVE
    assert "retry or external adjudication" in result.components[0].reason


def test_overlay_is_component_scoped_versioned_and_rejects_root_authority(endpoints):
    overlay = BackendOverlay(
        overlay_id="pandas-null-materialization-v1",
        backend="left_backend",
        version_spec=">=1.0,<2.0",
        adapter_revision="adapter-v1",
        exact_preconditions=("dtype:nullable_int", "observed:adapter_nan"),
        affected_component="nullable_scalar_materialization",
        permitted_relation="nan_represents_null",
        evidence_source="frozen-corpus:test",
        expiry_policy="revalidate_on_backend_or_adapter_change",
    )
    registry = OverlayRegistry((overlay,))
    permissions = registry.resolve(
        endpoints[0],
        frozenset({"dtype:nullable_int", "observed:adapter_nan"}),
        "nullable_scalar_materialization",
    )

    assert len(permissions) == 1
    assert not registry.resolve(
        endpoints[1],
        frozenset({"dtype:nullable_int", "observed:adapter_nan"}),
        "nullable_scalar_materialization",
    )
    with pytest.raises(ValueError, match="family/root/candidate"):
        BackendOverlay(
            overlay_id="root_id:known",
            backend="left_backend",
            version_spec="*",
            adapter_revision="adapter-v1",
            exact_preconditions=("x",),
            affected_component="value",
            permitted_relation="bag_equal",
            evidence_source="bad",
            expiry_policy="never",
        )


def test_entailment_is_partial_order_not_global_chain():
    graph = default_entailment_graph()
    assert graph.entails("ordered_exact_dtype", "bag_exact_value")
    assert graph.entails("bag_exact_value", "ordered_exact_value") is False
    assert graph.entails("numeric_exact", "numeric_tolerant")
    assert graph.entails("bag_exact_value", "numeric_tolerant") is False


def test_unknown_relation_fails_closed(
    simple_compiled, simple_applicability, endpoints, int_schema
):
    value = replace(simple_compiled.contract.obligations[-1], relation_id="unknown_relation")
    contract = replace(simple_compiled.contract, contract_id="unknown-relation", obligations=(*simple_compiled.contract.obligations[:-1], value))
    applicability = replace(simple_applicability, contract_digest=contract.digest)
    verdict = monitor_exact(
        contract,
        _observations(int_schema, endpoints=endpoints),
        applicability,
        endpoints=endpoints,
    )
    assert verdict.kind == VerdictKind.INCONCLUSIVE


@pytest.mark.parametrize(
    ("allowed_rows", "fixed_rows", "actual_rows"),
    (
        (((1,), (2,), (3,)), ((1,),), ((1,), (1,))),
        (((1,), (1,), (2,)), ((1,), (1,)), ((1,), (2,))),
    ),
)
def test_partial_order_topk_rejects_multiplicity_exhaustion(
    int_schema, allowed_rows, fixed_rows, actual_rows
):
    obligation = RelationObligation.build(
        obligation_id="topk-counter",
        relation_id="partial_order_topk",
        endpoint_ids=("left", "right"),
        components=("row_membership", "duplicate_multiplicity"),
        parameters={
            "n": 2,
            "allowed_rows": allowed_rows,
            "fixed_rows": fixed_rows,
        },
    )
    observations = (
        Observation.build(
            endpoint_id="left", status="ok", schema=int_schema, rows=actual_rows
        ),
        Observation.build(
            endpoint_id="right", status="ok", schema=int_schema, rows=((1,), (2,))
        ),
    )

    verdict = default_relation_registry().evaluate(obligation, observations)

    assert verdict.kind == VerdictKind.VIOLATED
    assert "left" in dict(verdict.evidence)["failing_endpoints"]


def test_partial_order_topk_accepts_only_available_duplicate_counts(int_schema):
    obligation = RelationObligation.build(
        obligation_id="topk-legal-duplicates",
        relation_id="partial_order_topk",
        endpoint_ids=("left", "right"),
        components=("row_membership", "duplicate_multiplicity"),
        parameters={
            "n": 2,
            "allowed_rows": ((1,), (1,), (2,)),
            "fixed_rows": ((1,),),
        },
    )
    observations = (
        Observation.build(
            endpoint_id="left", status="ok", schema=int_schema, rows=((1,), (1,))
        ),
        Observation.build(
            endpoint_id="right", status="ok", schema=int_schema, rows=((1,), (2,))
        ),
    )

    assert (
        default_relation_registry().evaluate(obligation, observations).kind
        == VerdictKind.SATISFIED
    )


def test_partial_order_topk_rejects_fixed_counts_outside_allowed_multiset():
    obligation = RelationObligation.build(
        obligation_id="topk-invalid-fixed-count",
        relation_id="partial_order_topk",
        endpoint_ids=("left", "right"),
        components=("row_membership", "duplicate_multiplicity"),
        parameters={
            "n": 2,
            "allowed_rows": ((1,), (2,), (3,)),
            "fixed_rows": ((1,), (1,)),
        },
    )

    assert "topk_fixed_rows_outside_allowed_rows" in validate_relation_obligation(
        obligation
    )


def test_explicit_status_ok_interprets_semantic_error_as_violation(
    simple_compiled, simple_applicability, endpoints, int_schema
):
    status = RelationObligation.build(
        obligation_id="explicit-status-ok",
        relation_id="status_ok",
        endpoint_ids=("left", "right"),
        components=("status",),
        strength="status_ok",
    )
    contract = replace(
        simple_compiled.contract,
        contract_id="explicit-status-contract",
        obligations=(status,),
    )
    applicability = replace(simple_applicability, contract_digest=contract.digest)
    observations = (
        Observation.build(
            endpoint_id="left", status="ok", schema=int_schema, rows=[[1]],
            execution_metadata={"endpoint_digest": endpoints[0].digest},
        ),
        Observation.build(
            endpoint_id="right",
            status="semantic_error",
            error_category="domain",
            execution_metadata={"endpoint_digest": endpoints[1].digest},
        ),
    )

    verdict = monitor_exact(
        contract, observations, applicability, endpoints=endpoints
    )

    assert verdict.kind == VerdictKind.VIOLATED
    assert verdict.components[0].kind == VerdictKind.VIOLATED


def test_value_relation_cannot_consume_semantic_error(
    simple_compiled, simple_applicability, endpoints, int_schema
):
    value = replace(
        simple_compiled.contract.obligations[-1],
        obligation_id="value-only",
    )
    contract = replace(
        simple_compiled.contract,
        contract_id="value-semantic-error",
        obligations=(value,),
    )
    applicability = replace(simple_applicability, contract_digest=contract.digest)
    observations = (
        Observation.build(
            endpoint_id="left", status="ok", schema=int_schema, rows=[[1]],
            execution_metadata={"endpoint_digest": endpoints[0].digest},
        ),
        Observation.build(
            endpoint_id="right",
            status="semantic_error",
            error_category="domain",
            execution_metadata={"endpoint_digest": endpoints[1].digest},
        ),
    )

    verdict = monitor_exact(
        contract, observations, applicability, endpoints=endpoints
    )

    assert verdict.kind == VerdictKind.INCONCLUSIVE
