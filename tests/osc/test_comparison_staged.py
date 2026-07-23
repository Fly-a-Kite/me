from __future__ import annotations

from dataclasses import replace

import pytest

from datadiff_osc.comparison import staged as staged_module
from datadiff_osc.comparison.staged import (
    StagedComparisonEngine,
    cluster_component_fingerprints,
)
from datadiff_osc.contract_engine import Observation
from datadiff_osc.contract_engine.fingerprints import fingerprint_component
from datadiff_osc.contract_engine.observers import LazyObservationView, ObserverSpec
from datadiff_osc.schemas import (
    ContractFingerprint,
    EvidenceTier,
    ExecutionStatus,
    FailureKind,
    StagedComparisonRequest,
    StructuredExecutionOutcome,
    VerdictKind,
)


def _request(contract, tier=EvidenceTier.SCREENING, *, observers=None, force_exact=False):
    return StagedComparisonRequest(
        result_group_digest="result-group-1",
        contract_fingerprint=ContractFingerprint(contract.digest, contract.registry_digest),
        endpoint_ids=("left", "right"),
        observer_ids=tuple(observers or contract.observations),
        evidence_tier=tier,
        force_exact=force_exact,
    )


def _observations(schema, endpoints, right_rows=((1,), (2,))):
    return (
        Observation.build(
            endpoint_id="left",
            status="ok",
            schema=schema,
            rows=[[1], [2]],
            execution_metadata={"endpoint_digest": endpoints[0].digest},
        ),
        Observation.build(
            endpoint_id="right",
            status="ok",
            schema=schema,
            rows=right_rows,
            execution_metadata={"endpoint_digest": endpoints[1].digest},
        ),
    )


def _outcomes(right_status=ExecutionStatus.OK):
    failure_kind = {
        ExecutionStatus.OK: FailureKind.NONE,
        ExecutionStatus.TIMEOUT: FailureKind.TIMEOUT,
        ExecutionStatus.MISSING: FailureKind.MISSING_RESULT,
    }[right_status]
    return (
        StructuredExecutionOutcome(
            endpoint_id="left",
            status=ExecutionStatus.OK,
            failure_kind=FailureKind.NONE,
        ),
        StructuredExecutionOutcome(
            endpoint_id="right",
            status=right_status,
            failure_kind=failure_kind,
            reason=("bounded timeout" if right_status is ExecutionStatus.TIMEOUT else ""),
        ),
    )


def _timeout_observations(schema, endpoints):
    return (
        Observation.build(
            endpoint_id="left",
            status="ok",
            schema=schema,
            rows=[[1], [2]],
            execution_metadata={"endpoint_digest": endpoints[0].digest},
        ),
        Observation.build(
            endpoint_id="right",
            status="timeout",
            schema=schema,
            error_category="timeout",
            error_type="TimeoutError",
            error_message="bounded timeout",
            execution_metadata={"endpoint_digest": endpoints[1].digest},
        ),
    )


def test_frozen_request_adapter_screens_equal_and_escalates_mismatch(
    simple_compiled, simple_applicability, endpoints, int_schema
):
    contract = simple_compiled.contract
    outcomes = _outcomes()
    screened, screened_certificate = StagedComparisonEngine.compare(
        _request(contract),
        contract,
        _observations(int_schema, endpoints),
        simple_applicability,
        execution_outcomes=outcomes,
        endpoints=endpoints,
    )
    mismatch, mismatch_certificate = StagedComparisonEngine.compare(
        _request(contract),
        contract,
        _observations(int_schema, endpoints, right_rows=((1,), (3,))),
        simple_applicability,
        execution_outcomes=outcomes,
        endpoints=endpoints,
    )
    assert screened.verdict_kind == VerdictKind.SATISFIED
    assert screened.exact_escalated is False
    assert screened_certificate.comparison_stage == "S2_COMPONENT_FINGERPRINT"
    assert screened.observation_certificate_digest == screened_certificate.digest
    assert mismatch.verdict_kind == VerdictKind.VIOLATED
    assert mismatch.exact_escalated is True
    assert mismatch_certificate.comparison_stage == "S3_EXACT_MATERIALIZED"
    assert mismatch.observation_certificate_digest == mismatch_certificate.digest


def test_wrapper_requires_execution_outcomes_keyword(
    simple_compiled, simple_applicability, endpoints, int_schema
):
    contract = simple_compiled.contract
    with pytest.raises(TypeError, match="execution_outcomes"):
        StagedComparisonEngine.compare(
            _request(contract),
            contract,
            _observations(int_schema, endpoints),
            simple_applicability,
            endpoints=endpoints,
        )


def test_wrapper_passes_caller_outcome_tuple_unchanged(
    monkeypatch, simple_compiled, simple_applicability, endpoints, int_schema
):
    contract = simple_compiled.contract
    outcomes = _outcomes()
    captured = []
    authority = staged_module.execute_staged_comparison

    def capture(*args, **kwargs):
        captured.append(kwargs["execution_outcomes"])
        return authority(*args, **kwargs)

    monkeypatch.setattr(staged_module, "execute_staged_comparison", capture)
    StagedComparisonEngine.compare(
        _request(contract),
        contract,
        _observations(int_schema, endpoints),
        simple_applicability,
        execution_outcomes=outcomes,
        endpoints=endpoints,
    )
    assert captured == [outcomes]
    assert captured[0] is outcomes


@pytest.mark.parametrize("mode", ["status_mismatch", "endpoint_reorder"])
def test_wrapper_outcome_binding_mismatch_fails_closed_before_fingerprints(
    simple_compiled, simple_applicability, endpoints, int_schema, mode
):
    contract = simple_compiled.contract
    outcomes = _outcomes(ExecutionStatus.TIMEOUT)
    if mode == "endpoint_reorder":
        outcomes = tuple(reversed(_outcomes()))
    result, certificate = StagedComparisonEngine.compare(
        _request(contract),
        contract,
        _observations(int_schema, endpoints),
        simple_applicability,
        execution_outcomes=outcomes,
        endpoints=endpoints,
    )
    assert result.verdict_kind is VerdictKind.INCONCLUSIVE
    assert certificate.comparison_stage == "S0_STATIC"
    assert certificate.exact_escalated is False
    assert certificate.component_fingerprints == ()


def test_explicit_bound_timeout_outcome_enters_s3_exact(
    simple_compiled, simple_applicability, endpoints, int_schema
):
    contract = simple_compiled.contract
    result, certificate = StagedComparisonEngine.compare(
        _request(contract),
        contract,
        _timeout_observations(int_schema, endpoints),
        simple_applicability,
        execution_outcomes=_outcomes(ExecutionStatus.TIMEOUT),
        endpoints=endpoints,
    )
    assert result.verdict_kind is VerdictKind.INCONCLUSIVE
    assert result.exact_escalated is True
    assert certificate.comparison_stage == "S3_EXACT_MATERIALIZED"


@pytest.mark.parametrize(
    "tier",
    [
        EvidenceTier.AUDIT,
        EvidenceTier.FINDING,
        EvidenceTier.FRESH_CONFIRMATION,
        EvidenceTier.NATIVE_REPRODUCTION,
    ],
)
def test_every_non_screening_tier_uses_exact_authority(
    simple_compiled, simple_applicability, endpoints, int_schema, tier
):
    contract = simple_compiled.contract
    result, _ = StagedComparisonEngine.compare(
        _request(contract, tier),
        contract,
        _observations(int_schema, endpoints),
        simple_applicability,
        execution_outcomes=_outcomes(),
        endpoints=endpoints,
    )
    assert result.exact_escalated is True


def test_unknown_observer_forces_exact_and_never_becomes_fast_path_authority(
    simple_compiled, simple_applicability, endpoints, int_schema
):
    contract = simple_compiled.contract
    result, _ = StagedComparisonEngine.compare(
        _request(contract, observers=("unknown-observer",)),
        contract,
        _observations(int_schema, endpoints),
        simple_applicability,
        execution_outcomes=_outcomes(),
        endpoints=endpoints,
    )
    assert result.exact_escalated is True


def test_fresh_cache_use_and_misbound_inputs_fail_closed(
    simple_compiled, simple_applicability, endpoints, int_schema
):
    contract = simple_compiled.contract
    with pytest.raises(ValueError, match="cache is forbidden"):
        StagedComparisonEngine.compare(
            _request(contract, EvidenceTier.FRESH_CONFIRMATION),
            contract,
            _observations(int_schema, endpoints),
            simple_applicability,
            execution_outcomes=_outcomes(),
            endpoints=endpoints,
            cache_used=True,
        )
    with pytest.raises(ValueError, match="contract digest"):
        StagedComparisonEngine.compare(
            replace(
                _request(contract),
                contract_fingerprint=ContractFingerprint("wrong", contract.registry_digest),
            ),
            contract,
            _observations(int_schema, endpoints),
            simple_applicability,
            execution_outcomes=_outcomes(),
            endpoints=endpoints,
        )
    with pytest.raises(ValueError, match="frozen endpoint IDs"):
        StagedComparisonEngine.compare(
            _request(contract),
            contract,
            _observations(int_schema, endpoints)[:1],
            simple_applicability,
            execution_outcomes=_outcomes(),
            endpoints=endpoints,
        )


def test_missing_endpoint_authority_returns_inconclusive_before_screening(
    simple_compiled, simple_applicability, endpoints, int_schema
):
    contract = simple_compiled.contract

    result, certificate = StagedComparisonEngine.compare(
        _request(contract),
        contract,
        _observations(int_schema, endpoints),
        simple_applicability,
        execution_outcomes=_outcomes(),
    )

    assert result.verdict_kind == VerdictKind.INCONCLUSIVE
    assert certificate.comparison_stage == "S0_STATIC"
    assert certificate.exact_escalated is False


def test_component_clustering_is_linear_ordered_and_uncertainty_is_explicit(
    simple_compiled, endpoints, int_schema
):
    contract = simple_compiled.contract
    observations = _observations(int_schema, endpoints)
    spec = ObserverSpec("sequence", "sequence")
    fingerprints = tuple(
        fingerprint_component(LazyObservationView(item), spec, contract.digest)
        for item in observations
    )
    clustered = cluster_component_fingerprints(
        reversed(fingerprints), endpoint_order=("left", "right")
    )
    assert len(clustered.clusters) == 1
    assert clustered.clusters[0].endpoint_ids == ("left", "right")
    assert clustered.exact_required is False

    missing = cluster_component_fingerprints(
        fingerprints[:1], endpoint_order=("left", "right")
    )
    duplicate = cluster_component_fingerprints(
        (fingerprints[0], fingerprints[0]), endpoint_order=("left", "right")
    )
    incompatible = cluster_component_fingerprints(
        (fingerprints[0], replace(fingerprints[1], observer_id="other")),
        endpoint_order=("left", "right"),
    )
    forged_schema = cluster_component_fingerprints(
        (fingerprints[0], replace(fingerprints[1], schema_digest="other-schema")),
        endpoint_order=("left", "right"),
    )
    assert missing.missing_endpoint_ids == ("right",)
    assert duplicate.duplicate_endpoint_ids == ("left",)
    assert incompatible.incompatible_fingerprint_count == 1
    assert all(
        item.exact_required
        for item in (missing, duplicate, incompatible, forged_schema)
    )
