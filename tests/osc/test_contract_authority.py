from __future__ import annotations

from dataclasses import replace

import pytest

from datadiff_osc.contract_engine import (
    Observation,
    VerdictKind,
    evaluate_applicability,
    execute_staged_comparison,
    plan_comparison,
)
from datadiff_osc.contract_engine.derivation import DerivationCertificate
from datadiff_osc.contract_engine.model import ComponentVerdict, RelationObligation
from datadiff_osc.contract_engine.monitor import monitor_exact
from datadiff_osc.contract_engine.relations import default_relation_registry
from datadiff_osc.schemas import (
    ExecutionStatus,
    FailureKind,
    StructuredExecutionOutcome,
)


def _observations(schema, left=1, right=1, *, endpoints=None):
    metadata = (
        tuple({"endpoint_digest": item.digest} for item in endpoints)
        if endpoints is not None
        else (None, None)
    )
    return (
        Observation.build(
            endpoint_id="left",
            status="ok",
            schema=schema,
            rows=[[left]],
            execution_metadata=metadata[0],
        ),
        Observation.build(
            endpoint_id="right",
            status="ok",
            schema=schema,
            rows=[[right]],
            execution_metadata=metadata[1],
        ),
    )


def _ok_outcomes(endpoint_ids=("left", "right")):
    return tuple(
        StructuredExecutionOutcome(endpoint_id, ExecutionStatus.OK, FailureKind.NONE)
        for endpoint_id in endpoint_ids
    )


def test_empty_derivation_certificate_is_not_valid():
    certificate = DerivationCertificate("", "", "", "", (), (), (), (), ())
    assert certificate.valid is False


def test_compiled_contract_binds_all_derivation_digests(simple_compiled):
    assert simple_compiled.valid
    stale = replace(
        simple_compiled,
        derivation=replace(simple_compiled.derivation, forward_digest="stale"),
    )
    assert stale.valid is False


def test_compiled_contract_rejects_obligation_rebinding(simple_compiled):
    value = replace(
        simple_compiled.contract.obligations[-1], relation_id="sequence_equal"
    )
    rebound = replace(
        simple_compiled,
        contract=replace(
            simple_compiled.contract,
            obligations=(*simple_compiled.contract.obligations[:-1], value),
        ),
    )
    assert rebound.valid is False


def test_forged_empty_applicability_decisions_fail_closed(
    simple_compiled, simple_applicability, endpoints, int_schema
):
    forged = replace(
        simple_applicability,
        endpoint_digests=("fake-a", "fake-b"),
        capability_decisions=(),
    )
    verdict = monitor_exact(
        simple_compiled.contract,
        _observations(int_schema, endpoints=endpoints),
        forged,
        endpoints=endpoints,
    )
    assert verdict.kind == VerdictKind.INCONCLUSIVE
    assert "certificate" in verdict.reason


def test_contract_digest_rebinding_alone_cannot_reuse_applicability(
    simple_compiled, simple_applicability, endpoints, int_schema
):
    changed_requirement = replace(
        simple_compiled.contract.endpoint_requirements[0],
        required_capabilities=frozenset({"op:select", "op:new"}),
    )
    contract = replace(
        simple_compiled.contract,
        contract_id="changed-scope",
        endpoint_requirements=(
            changed_requirement,
            simple_compiled.contract.endpoint_requirements[1],
        ),
    )
    stale = replace(simple_applicability, contract_digest=contract.digest)
    verdict = monitor_exact(
        contract,
        _observations(int_schema, endpoints=endpoints),
        stale,
        endpoints=endpoints,
    )
    assert verdict.kind == VerdictKind.INCONCLUSIVE
    assert "capability_decision_invalid:left" in verdict.reason


def test_large_integer_exactness_never_routes_through_binary64(
    simple_compiled, simple_applicability, endpoints, int_schema
):
    obligation = RelationObligation.build(
        obligation_id="large-int",
        relation_id="numeric_exact",
        endpoint_ids=("left", "right"),
        components=("numeric",),
        strength="numeric_exact",
        relation_properties={"reflexive", "symmetric", "transitive"},
    )
    contract = replace(
        simple_compiled.contract,
        contract_id="large-int-contract",
        obligations=(obligation,),
    )
    applicability = replace(simple_applicability, contract_digest=contract.digest)
    observations = _observations(
        int_schema, 2**60, 2**60 + 1, endpoints=endpoints
    )
    assert (
        monitor_exact(
            contract, observations, applicability, endpoints=endpoints
        ).kind
        == VerdictKind.VIOLATED
    )


def test_unknown_evidence_tier_is_rejected(simple_compiled):
    with pytest.raises(ValueError, match="unknown evidence tier"):
        plan_comparison(simple_compiled.contract, evidence_tier="candidate")


def test_non_screening_tier_always_uses_exact_authority(
    simple_compiled, simple_applicability, endpoints, int_schema
):
    certificate = execute_staged_comparison(
        simple_compiled.contract,
        _observations(int_schema, endpoints=endpoints),
        simple_applicability,
        evidence_tier="audit",
        endpoints=endpoints,
        execution_outcomes=_ok_outcomes(),
    )
    assert certificate.exact_escalated
    assert certificate.comparison_stage == "S3_EXACT_MATERIALIZED"


def test_duplicate_endpoint_observation_cannot_screen_as_pass(
    simple_compiled, simple_applicability, endpoints, int_schema
):
    left_metadata = {"endpoint_digest": endpoints[0].digest}
    right_metadata = {"endpoint_digest": endpoints[1].digest}
    observations = (
        Observation.build(
            endpoint_id="left", status="ok", schema=int_schema, rows=[[1]],
            execution_metadata=left_metadata,
        ),
        Observation.build(
            endpoint_id="left", status="ok", schema=int_schema, rows=[[1]],
            execution_metadata=left_metadata,
        ),
        Observation.build(
            endpoint_id="right", status="ok", schema=int_schema, rows=[[1]],
            execution_metadata=right_metadata,
        ),
    )
    result = execute_staged_comparison(
        simple_compiled.contract,
        observations,
        simple_applicability,
        endpoints=endpoints,
        execution_outcomes=_ok_outcomes(),
    )
    assert result.exact_escalated
    assert result.verdict.kind == VerdictKind.INCONCLUSIVE


def test_structured_outcome_must_match_observation_status(
    simple_compiled, simple_applicability, endpoints, int_schema
):
    outcomes = (
        StructuredExecutionOutcome(
            "left", ExecutionStatus.OK, FailureKind.NONE
        ),
        StructuredExecutionOutcome(
            "right", ExecutionStatus.TIMEOUT, FailureKind.TIMEOUT
        ),
    )
    verdict = monitor_exact(
        simple_compiled.contract,
        _observations(int_schema, endpoints=endpoints),
        simple_applicability,
        endpoints=endpoints,
        execution_outcomes=outcomes,
    )
    assert verdict.kind == VerdictKind.INCONCLUSIVE
    assert "outcome_status_mismatch:right" in verdict.reason


def test_staged_screening_rejects_outcome_status_mismatch_before_fingerprints(
    simple_compiled, simple_applicability, endpoints, int_schema
):
    outcomes = (
        StructuredExecutionOutcome(
            "left", ExecutionStatus.OK, FailureKind.NONE
        ),
        StructuredExecutionOutcome(
            "right", ExecutionStatus.TIMEOUT, FailureKind.TIMEOUT
        ),
    )

    certificate = execute_staged_comparison(
        simple_compiled.contract,
        _observations(int_schema, endpoints=endpoints),
        simple_applicability,
        endpoints=endpoints,
        execution_outcomes=outcomes,
    )

    assert certificate.comparison_stage == "S0_STATIC"
    assert certificate.verdict.kind == VerdictKind.INCONCLUSIVE
    assert certificate.component_fingerprints == ()


def test_staged_screening_requires_execution_outcomes_before_fingerprints(
    simple_compiled, simple_applicability, endpoints, int_schema
):
    certificate = execute_staged_comparison(
        simple_compiled.contract,
        _observations(int_schema, endpoints=endpoints),
        simple_applicability,
        endpoints=endpoints,
    )

    assert certificate.comparison_stage == "S0_STATIC"
    assert certificate.verdict.kind == VerdictKind.INCONCLUSIVE
    assert certificate.component_fingerprints == ()
    assert "execution_outcomes_missing" in certificate.verdict.reason


@pytest.mark.parametrize(
    ("endpoint_ids", "expected_error"),
    (
        pytest.param(
            ("left",),
            "execution_outcome_endpoint_order_mismatch",
            id="incomplete",
        ),
        pytest.param(
            ("right", "left"),
            "execution_outcome_endpoint_order_mismatch",
            id="reordered",
        ),
        pytest.param(
            ("left", "left"),
            "duplicate_execution_outcome_endpoint",
            id="duplicate",
        ),
        pytest.param(
            ("left", "unexpected"),
            "outcome_without_observation:unexpected",
            id="mismatched-endpoint",
        ),
    ),
)
def test_staged_screening_rejects_unbound_outcome_endpoints_before_fingerprints(
    simple_compiled,
    simple_applicability,
    endpoints,
    int_schema,
    endpoint_ids,
    expected_error,
):
    certificate = execute_staged_comparison(
        simple_compiled.contract,
        _observations(int_schema, endpoints=endpoints),
        simple_applicability,
        endpoints=endpoints,
        execution_outcomes=_ok_outcomes(endpoint_ids),
    )

    assert certificate.comparison_stage == "S0_STATIC"
    assert certificate.verdict.kind == VerdictKind.INCONCLUSIVE
    assert certificate.component_fingerprints == ()
    assert expected_error in certificate.verdict.reason


def test_staged_screening_accepts_bound_ok_outcomes(
    simple_compiled, simple_applicability, endpoints, int_schema
):
    certificate = execute_staged_comparison(
        simple_compiled.contract,
        _observations(int_schema, endpoints=endpoints),
        simple_applicability,
        endpoints=endpoints,
        execution_outcomes=_ok_outcomes(),
    )

    assert certificate.comparison_stage == "S2_COMPONENT_FINGERPRINT"
    assert certificate.exact_escalated is False
    assert certificate.verdict.kind == VerdictKind.SATISFIED


def test_bound_missing_outcome_enters_exact_and_remains_inconclusive(
    simple_compiled, simple_applicability, endpoints, int_schema
):
    observations = (
        Observation.build(
            endpoint_id="left",
            status="ok",
            schema=int_schema,
            rows=[[1]],
            execution_metadata={"endpoint_digest": endpoints[0].digest},
        ),
        Observation.build(
            endpoint_id="right",
            status="missing",
            schema=int_schema,
            execution_metadata={"endpoint_digest": endpoints[1].digest},
        ),
    )
    outcomes = (
        StructuredExecutionOutcome("left", ExecutionStatus.OK, FailureKind.NONE),
        StructuredExecutionOutcome(
            "right", ExecutionStatus.MISSING, FailureKind.MISSING_RESULT
        ),
    )

    certificate = execute_staged_comparison(
        simple_compiled.contract,
        observations,
        simple_applicability,
        endpoints=endpoints,
        execution_outcomes=outcomes,
    )

    assert certificate.comparison_stage == "S3_EXACT_MATERIALIZED"
    assert certificate.exact_escalated is True
    assert certificate.verdict.kind == VerdictKind.INCONCLUSIVE
    assert any(
        "structured execution failure" in component.reason
        for component in certificate.verdict.components
    )


def test_unsupported_outcome_requires_exact_applicability_evidence(
    simple_compiled, endpoints, int_schema
):
    restricted = replace(
        simple_compiled.contract.endpoint_requirements[0],
        required_capabilities=frozenset({"op:missing"}),
    )
    contract = replace(
        simple_compiled.contract,
        contract_id="unsupported-outcome-binding",
        endpoint_requirements=(
            restricted,
            simple_compiled.contract.endpoint_requirements[1],
        ),
    )
    applicability = evaluate_applicability(
        contract,
        endpoints,
        facts=frozenset(contract.preconditions),
    )
    observations = (
        Observation.build(
            endpoint_id="left",
            status="unsupported",
            schema=int_schema,
            execution_metadata={"endpoint_digest": endpoints[0].digest},
        ),
        Observation.build(
            endpoint_id="right",
            status="ok",
            schema=int_schema,
            rows=[[1]],
            execution_metadata={"endpoint_digest": endpoints[1].digest},
        ),
    )
    evidence_digest = applicability.unsupported_evidence[0].digest
    valid_outcomes = (
        StructuredExecutionOutcome(
            "left",
            ExecutionStatus.UNSUPPORTED,
            FailureKind.UNSUPPORTED_CAPABILITY,
            unsupported_evidence_digest=evidence_digest,
        ),
        StructuredExecutionOutcome("right", ExecutionStatus.OK, FailureKind.NONE),
    )
    mismatched_outcomes = (
        replace(valid_outcomes[0], unsupported_evidence_digest="stale-evidence"),
        valid_outcomes[1],
    )

    missing = execute_staged_comparison(
        contract,
        observations,
        applicability,
        endpoints=endpoints,
    )
    mismatched = execute_staged_comparison(
        contract,
        observations,
        applicability,
        endpoints=endpoints,
        execution_outcomes=mismatched_outcomes,
    )
    valid = execute_staged_comparison(
        contract,
        observations,
        applicability,
        endpoints=endpoints,
        execution_outcomes=valid_outcomes,
    )

    assert missing.comparison_stage == "S0_STATIC"
    assert missing.verdict.kind == VerdictKind.INCONCLUSIVE
    assert missing.component_fingerprints == ()
    assert mismatched.comparison_stage == "S0_STATIC"
    assert mismatched.verdict.kind == VerdictKind.INCONCLUSIVE
    assert mismatched.component_fingerprints == ()
    assert "unsupported_evidence_binding_mismatch:left" in mismatched.verdict.reason
    assert valid.comparison_stage == "S0_STATIC"
    assert valid.verdict.kind == VerdictKind.INAPPLICABLE
    assert valid.component_fingerprints == ()


def test_contract_registry_digest_is_separate_from_derivation_rule_registry(
    simple_compiled,
):
    assert simple_compiled.contract.registry_digest == default_relation_registry().digest
    assert (
        simple_compiled.contract.registry_digest
        != simple_compiled.derivation.rule_registry_digest
    )


def test_same_metadata_replacement_evaluator_is_rejected_before_exact(
    simple_compiled, simple_applicability, endpoints, int_schema
):
    trusted = default_relation_registry()
    invoked = False

    def forged_evaluator(obligation, observations):
        nonlocal invoked
        invoked = True
        return ComponentVerdict.build(
            obligation.obligation_id,
            VerdictKind.SATISFIED,
            "forged evaluator",
        )

    forged = replace(
        trusted,
        definitions=tuple(
            replace(item, evaluator=forged_evaluator)
            if item.relation_id == "bag_equal"
            else item
            for item in trusted.definitions
        ),
    )
    assert forged.digest == trusted.digest

    verdict = monitor_exact(
        simple_compiled.contract,
        _observations(int_schema, 1, 2, endpoints=endpoints),
        simple_applicability,
        registry=forged,
        endpoints=endpoints,
    )

    assert verdict.kind == VerdictKind.INCONCLUSIVE
    assert "registry" in verdict.reason
    assert invoked is False


@pytest.mark.parametrize("evidence_tier", ("screening", "audit"))
def test_same_metadata_replacement_evaluator_is_rejected_before_planning(
    simple_compiled, evidence_tier
):
    trusted = default_relation_registry()

    def forged_evaluator(obligation, observations):
        return ComponentVerdict.build(
            obligation.obligation_id,
            VerdictKind.SATISFIED,
            "forged evaluator",
        )

    forged = replace(
        trusted,
        definitions=tuple(
            replace(item, evaluator=forged_evaluator)
            if item.relation_id == "bag_equal"
            else item
            for item in trusted.definitions
        ),
    )

    with pytest.raises(ValueError, match="registry"):
        plan_comparison(
            simple_compiled.contract,
            evidence_tier=evidence_tier,
            registry=forged,
        )


@pytest.mark.parametrize(
    ("scope", "changes"),
    (
        ("backend_scope", {"backend": "other_backend"}),
        ("version_scope", {"backend_version": "9.0.0"}),
        ("mode_scope", {"execution_mode": "streaming"}),
        ("layout_scope", {"physical_layout": "chunked"}),
    ),
)
def test_endpoint_scope_rebinding_fails_closed_end_to_end(
    simple_compiled, endpoints, int_schema, scope, changes
):
    strict_requirements = tuple(
        replace(
            requirement,
            version_spec=endpoint.backend_version,
            execution_modes=frozenset({endpoint.execution_mode}),
            physical_layouts=frozenset({endpoint.physical_layout}),
        )
        for requirement, endpoint in zip(
            simple_compiled.contract.endpoint_requirements, endpoints, strict=True
        )
    )
    contract = replace(
        simple_compiled.contract,
        contract_id="strict-endpoint-scope",
        endpoint_requirements=strict_requirements,
    )
    certificate = evaluate_applicability(
        contract,
        endpoints,
        facts=frozenset(contract.preconditions),
    )
    rebound_endpoints = (replace(endpoints[0], **changes), endpoints[1])
    forged = replace(
        certificate,
        endpoint_digests=tuple(item.digest for item in rebound_endpoints),
    )
    observations = tuple(
        Observation.build(
            endpoint_id=item.endpoint_id,
            status="ok",
            schema=int_schema,
            rows=[[1]],
            execution_metadata={"endpoint_digest": item.digest},
        )
        for item in rebound_endpoints
    )

    verdict = monitor_exact(
        contract,
        observations,
        forged,
        endpoints=rebound_endpoints,
    )

    assert verdict.kind == VerdictKind.INCONCLUSIVE
    assert scope in verdict.reason


def test_missing_observation_endpoint_digest_fails_closed_when_endpoints_supplied(
    simple_compiled, simple_applicability, endpoints, int_schema
):
    verdict = monitor_exact(
        simple_compiled.contract,
        _observations(int_schema),
        simple_applicability,
        endpoints=endpoints,
    )

    assert verdict.kind == VerdictKind.INCONCLUSIVE
    assert "observation_endpoint_digest_missing" in verdict.reason


def test_staged_screening_rejects_endpoint_scope_rebinding_at_s0(
    simple_compiled, endpoints, int_schema
):
    strict_requirement = replace(
        simple_compiled.contract.endpoint_requirements[0],
        version_spec=endpoints[0].backend_version,
    )
    contract = replace(
        simple_compiled.contract,
        contract_id="strict-screening-scope",
        endpoint_requirements=(
            strict_requirement,
            simple_compiled.contract.endpoint_requirements[1],
        ),
    )
    certificate = evaluate_applicability(
        contract,
        endpoints,
        facts=frozenset(contract.preconditions),
    )
    rebound_endpoints = (
        replace(endpoints[0], backend_version="9.0.0"),
        endpoints[1],
    )
    forged = replace(
        certificate,
        endpoint_digests=tuple(item.digest for item in rebound_endpoints),
    )
    observations = tuple(
        Observation.build(
            endpoint_id=item.endpoint_id,
            status="ok",
            schema=int_schema,
            rows=[[1]],
            execution_metadata={"endpoint_digest": item.digest},
        )
        for item in rebound_endpoints
    )

    result = execute_staged_comparison(
        contract,
        observations,
        forged,
        endpoints=rebound_endpoints,
        execution_outcomes=_ok_outcomes(),
    )

    assert result.comparison_stage == "S0_STATIC"
    assert result.verdict.kind == VerdictKind.INCONCLUSIVE


@pytest.mark.parametrize("include_digest_metadata", (False, True))
def test_authority_requires_original_endpoint_tuple_even_if_digests_match(
    simple_compiled, endpoints, int_schema, include_digest_metadata
):
    strict_requirement = replace(
        simple_compiled.contract.endpoint_requirements[0],
        version_spec=endpoints[0].backend_version,
    )
    contract = replace(
        simple_compiled.contract,
        contract_id="endpoint-tuple-required",
        endpoint_requirements=(
            strict_requirement,
            simple_compiled.contract.endpoint_requirements[1],
        ),
    )
    certificate = evaluate_applicability(
        contract,
        endpoints,
        facts=frozenset(contract.preconditions),
    )
    rebound_endpoints = (
        replace(endpoints[0], backend_version="9.0.0"),
        endpoints[1],
    )
    forged = replace(
        certificate,
        endpoint_digests=tuple(item.digest for item in rebound_endpoints),
    )
    observations = tuple(
        Observation.build(
            endpoint_id=item.endpoint_id,
            status="ok",
            schema=int_schema,
            rows=[[1]],
            execution_metadata=(
                {"endpoint_digest": item.digest}
                if include_digest_metadata
                else None
            ),
        )
        for item in rebound_endpoints
    )

    verdict = monitor_exact(contract, observations, forged)

    assert verdict.kind == VerdictKind.INCONCLUSIVE
    assert "authority_endpoint_tuple_missing" in verdict.reason
