from __future__ import annotations

from copy import deepcopy
import json

import datadiff_osc
from datadiff_osc._canonical import canonical_json, stable_digest
from datadiff_osc.contract_engine.applicability import evaluate_applicability
from datadiff_osc.contract_engine.evidence import (
    ExactEscalation,
    ObservationCertificate,
)
from datadiff_osc.contract_engine.fingerprints import ComponentFingerprint
from datadiff_osc.contract_engine.model import (
    CONTRACT_SCHEMA_VERSION,
    ComponentVerdict,
    Endpoint,
    EndpointRequirement,
    HyperContract,
    RelationObligation,
    Verdict,
)
from datadiff_osc.contract_engine.relations import default_relation_registry
from datadiff_osc.contract_engine.replay import (
    CONTRACT_REPLAY_ENVELOPE_NOT_ALLOWED,
    CONTRACT_REPLAY_PAYLOAD_INVALID,
    CONTRACT_REPLAY_SCHEMA_VERSION_MISMATCH,
    CONTRACT_REPLAY_SUBJECT_IDS_MISMATCH,
    CONTRACT_REPLAY_SUBJECT_KIND_NOT_DERIVABLE,
    replay_contract_admission,
)
from datadiff_osc.schemas import (
    ContractFingerprint,
    EvidenceTier,
    ResultGroup,
    TargetFingerprint,
    VerdictKind,
)


def _endpoint() -> Endpoint:
    return Endpoint.build(
        endpoint_id="endpoint-a",
        case_digest=stable_digest("test-case", "case-a"),
        backend="backend-a",
        backend_version="1.2.3",
        adapter_revision="adapter-r1",
        execution_mode="eager",
        physical_layout="contiguous",
        optimizer_config={
            "blob": b"\x00\xff",
            "negative_zero": -0.0,
            "nested": (1, "value"),
        },
        capabilities={"execute", "stable_schema"},
    )


def _contract() -> HyperContract:
    requirement = EndpointRequirement(
        endpoint_id="endpoint-a",
        backend="backend-a",
        version_spec="1.2.3",
        execution_modes=frozenset({"eager"}),
        physical_layouts=frozenset({"contiguous"}),
        required_capabilities=frozenset({"execute"}),
    )
    obligation = RelationObligation.build(
        obligation_id="status-obligation",
        relation_id="status_ok",
        endpoint_ids=("endpoint-a",),
        components=("status",),
        strength="status_exact",
        relation_properties={"reflexive", "symmetric", "transitive"},
    )
    return HyperContract(
        contract_id="contract-a",
        endpoint_requirements=(requirement,),
        preconditions=("no_unresolved_rules", "endpoint_scope_matches"),
        observations=("status",),
        obligations=(obligation,),
        strength="status_exact",
        derivation_digest=stable_digest("test-derivation", "contract-a"),
        registry_digest=default_relation_registry().digest,
    )


def _result_group() -> ResultGroup:
    contract = _contract()
    return ResultGroup(
        result_group_id="result-group-a",
        task_ids=("task-a",),
        endpoint_ids=("endpoint-a",),
        contract_fingerprint=ContractFingerprint(
            contract_digest=contract.digest,
            registry_digest=contract.registry_digest,
        ),
        target_fingerprint=TargetFingerprint(
            universe_digest=stable_digest("test-universe", "u"),
            taxonomy_digest=stable_digest("test-taxonomy", "t"),
            template_digest=stable_digest("test-template", "x"),
        ),
        evidence_tier=EvidenceTier.AUDIT,
        result_order_key=("task-a", "endpoint-a"),
    )


def _applicability_certificate():
    contract = _contract()
    endpoint = _endpoint()
    certificate = evaluate_applicability(
        contract,
        (endpoint,),
        facts=frozenset(contract.preconditions),
    )
    assert certificate.valid
    return certificate


def _observation_certificate() -> ObservationCertificate:
    contract = _contract()
    applicability = _applicability_certificate()
    fingerprint = ComponentFingerprint(
        endpoint_id="endpoint-a",
        observer_id="status-observer",
        observer_digest=stable_digest("test-observer", "status"),
        contract_digest=contract.digest,
        row_count=1,
        schema_digest="1" * 64,
        payload_digest="2" * 64,
    )
    component = ComponentVerdict.build(
        "status-obligation",
        VerdictKind.SATISFIED,
        "status is valid",
        {"negative_zero": -0.0},
    )
    certificate = ObservationCertificate(
        contract_digest=contract.digest,
        applicability_digest=applicability.digest,
        executed_endpoint_ids=("endpoint-a",),
        observer_ids=("status-observer",),
        component_fingerprints=(fingerprint,),
        relation_trace=("status-obligation: one fingerprint cluster",),
        comparison_stage="S2_COMPONENT_FINGERPRINT",
        exact_escalated=False,
        materialized_endpoint_ids=(),
        verdict=Verdict.aggregate((component,), reason="screening equality"),
    )
    assert certificate.valid
    return certificate


def _exact_escalation() -> ExactEscalation:
    escalation = ExactEscalation(
        obligation_id="status-obligation",
        reason="fingerprints diverged",
        endpoint_ids=("endpoint-a",),
        materialized_bytes=128,
    )
    assert escalation.valid
    return escalation


def _decoded_payload(value: object) -> object:
    """Match the decoded canonical payload handed to a domain replayer."""

    return json.loads(canonical_json(value))


def _assert_admitted(
    *,
    envelope_type: str,
    schema_version: str,
    value: object,
    subject_kind: str,
    subject_id: str,
) -> None:
    assert replay_contract_admission(
        envelope_type=envelope_type,
        schema_version=schema_version,
        payload=_decoded_payload(value),
        subject_kind=subject_kind,
        subject_ids=(subject_id,),
    ) == ()


def _valid_admissions():
    contract = _contract()
    endpoint = _endpoint()
    group = _result_group()
    applicability = _applicability_certificate()
    observation = _observation_certificate()
    escalation = _exact_escalation()
    return (
        (
            "HyperContract",
            CONTRACT_SCHEMA_VERSION,
            contract,
            "hypercontract_digests",
            contract.digest,
        ),
        (
            "Endpoint",
            CONTRACT_SCHEMA_VERSION,
            endpoint,
            "endpoint_digests",
            endpoint.digest,
        ),
        (
            "ResultGroup",
            "osc-result-group-v1",
            group,
            "result_group_digests",
            group.digest,
        ),
        (
            "ApplicabilityCertificate",
            "osc-applicability-certificate-v1",
            applicability,
            "applicability_certificate_digests",
            applicability.digest,
        ),
        (
            "ObservationCertificate",
            "osc-observation-certificate-v1",
            observation,
            "observation_certificate_digests",
            observation.digest,
        ),
        (
            "ExactEscalationCertificate",
            "osc-exact-escalation-v1",
            escalation,
            "exact_escalation_certificate_digests",
            escalation.digest,
        ),
    )


def test_all_six_contract_envelopes_replay_exact_intrinsic_digests():
    for envelope_type, version, value, subject_kind, subject_id in _valid_admissions():
        _assert_admitted(
            envelope_type=envelope_type,
            schema_version=version,
            value=value,
            subject_kind=subject_kind,
            subject_id=subject_id,
        )


def test_result_group_replay_derives_its_staged_parity_identity():
    group = _result_group()
    _assert_admitted(
        envelope_type="ResultGroup",
        schema_version="osc-result-group-v1",
        value=group,
        subject_kind="staged_parity_groups",
        subject_id=group.result_group_id,
    )


def test_exact_escalation_certificate_privately_maps_to_existing_type():
    escalation = _exact_escalation()
    _assert_admitted(
        envelope_type="ExactEscalationCertificate",
        schema_version="osc-exact-escalation-v1",
        value=escalation,
        subject_kind="exact_escalation_certificate_digests",
        subject_id=escalation.digest,
    )
    assert "ExactEscalationCertificate" not in datadiff_osc.__all__


def test_unknown_envelope_and_wrong_schema_version_fail_closed():
    contract = _contract()
    payload = _decoded_payload(contract)
    assert replay_contract_admission(
        envelope_type="LedgerEvent",
        schema_version="osc-ledger-event-v1",
        payload=payload,
        subject_kind="hypercontract_digests",
        subject_ids=(contract.digest,),
    ) == (CONTRACT_REPLAY_ENVELOPE_NOT_ALLOWED,)
    assert replay_contract_admission(
        envelope_type="HyperContract",
        schema_version="osc-hypercontract-v999",
        payload=payload,
        subject_kind="hypercontract_digests",
        subject_ids=(contract.digest,),
    ) == (CONTRACT_REPLAY_SCHEMA_VERSION_MISMATCH,)


def test_missing_extra_and_aggregate_payloads_fail_closed():
    contract = _contract()
    payload = _decoded_payload(contract)
    assert isinstance(payload, dict)
    missing = deepcopy(payload)
    del missing["derivation_digest"]
    extra = deepcopy(payload)
    extra["credited_ids"] = ["root-a"]

    for forged in (missing, extra, [payload]):
        assert replay_contract_admission(
            envelope_type="HyperContract",
            schema_version=CONTRACT_SCHEMA_VERSION,
            payload=forged,
            subject_kind="hypercontract_digests",
            subject_ids=(contract.digest,),
        ) == (CONTRACT_REPLAY_PAYLOAD_INVALID,)


def test_hypercontract_replay_rejects_registry_relation_and_property_forgery():
    contract = _contract()
    payload = _decoded_payload(contract)
    assert isinstance(payload, dict)

    wrong_registry = deepcopy(payload)
    wrong_registry["registry_digest"] = stable_digest("forged-registry", "x")
    unknown_relation = deepcopy(payload)
    unknown_relation["obligations"][0]["relation_id"] = "unknown-relation"
    untrusted_property = deepcopy(payload)
    untrusted_property["obligations"][0]["relation_properties"].append("total")

    for forged in (wrong_registry, unknown_relation, untrusted_property):
        assert replay_contract_admission(
            envelope_type="HyperContract",
            schema_version=CONTRACT_SCHEMA_VERSION,
            payload=forged,
            subject_kind="hypercontract_digests",
            subject_ids=(contract.digest,),
        ) == (CONTRACT_REPLAY_PAYLOAD_INVALID,)


def test_endpoint_replay_rejects_noncanonical_immutable_values():
    endpoint = _endpoint()
    payload = _decoded_payload(endpoint)
    assert isinstance(payload, dict)
    forged = deepcopy(payload)
    negative_zero = next(
        item for item in forged["optimizer_config"] if item[0] == "negative_zero"
    )
    negative_zero[1] = {"$float": "0x1p+0"}

    assert replay_contract_admission(
        envelope_type="Endpoint",
        schema_version=CONTRACT_SCHEMA_VERSION,
        payload=forged,
        subject_kind="endpoint_digests",
        subject_ids=(endpoint.digest,),
    ) == (CONTRACT_REPLAY_PAYLOAD_INVALID,)


def test_result_group_replay_rejects_nested_fingerprint_forgery():
    group = _result_group()
    payload = _decoded_payload(group)
    assert isinstance(payload, dict)
    forged = deepcopy(payload)
    forged["contract_fingerprint"]["registry_digest"] = "forged"

    assert replay_contract_admission(
        envelope_type="ResultGroup",
        schema_version="osc-result-group-v1",
        payload=forged,
        subject_kind="result_group_digests",
        subject_ids=(group.digest,),
    ) == (CONTRACT_REPLAY_PAYLOAD_INVALID,)


def test_applicability_replay_rejects_internally_inconsistent_decisions():
    certificate = _applicability_certificate()
    payload = _decoded_payload(certificate)
    assert isinstance(payload, dict)
    forged = deepcopy(payload)
    forged["capability_decisions"][0]["supported"] = False

    assert replay_contract_admission(
        envelope_type="ApplicabilityCertificate",
        schema_version="osc-applicability-certificate-v1",
        payload=forged,
        subject_kind="applicability_certificate_digests",
        subject_ids=(certificate.digest,),
    ) == (CONTRACT_REPLAY_PAYLOAD_INVALID,)


def test_observation_replay_rejects_invalid_stage_and_contradictory_verdict():
    certificate = _observation_certificate()
    payload = _decoded_payload(certificate)
    assert isinstance(payload, dict)
    invalid_stage = deepcopy(payload)
    invalid_stage["comparison_stage"] = "SHAPED_STAGE"
    contradictory = deepcopy(payload)
    contradictory["verdict"]["kind"] = VerdictKind.VIOLATED.value

    for forged in (invalid_stage, contradictory):
        assert replay_contract_admission(
            envelope_type="ObservationCertificate",
            schema_version="osc-observation-certificate-v1",
            payload=forged,
            subject_kind="observation_certificate_digests",
            subject_ids=(certificate.digest,),
        ) == (CONTRACT_REPLAY_PAYLOAD_INVALID,)


def test_subject_identity_substitution_fails_for_every_contract_type():
    for envelope_type, version, value, subject_kind, _ in _valid_admissions():
        assert replay_contract_admission(
            envelope_type=envelope_type,
            schema_version=version,
            payload=_decoded_payload(value),
            subject_kind=subject_kind,
            subject_ids=("same-cardinality-substitute",),
        ) == (CONTRACT_REPLAY_SUBJECT_IDS_MISMATCH,)


def test_context_only_root_mutant_and_false_positive_claims_fail_closed():
    contract = _contract()
    for subject_kind in (
        "confirmed_roots_total",
        "high_risk_mutants_total",
        "false_positive_fixes",
    ):
        assert replay_contract_admission(
            envelope_type="HyperContract",
            schema_version=CONTRACT_SCHEMA_VERSION,
            payload=_decoded_payload(contract),
            subject_kind=subject_kind,
            subject_ids=("caller-authored-claim",),
        ) == (CONTRACT_REPLAY_SUBJECT_KIND_NOT_DERIVABLE,)


def test_unknown_subject_kind_and_non_tuple_subjects_fail_closed():
    endpoint = _endpoint()
    payload = _decoded_payload(endpoint)
    assert replay_contract_admission(
        envelope_type="Endpoint",
        schema_version=CONTRACT_SCHEMA_VERSION,
        payload=payload,
        subject_kind="endpoint_ids",
        subject_ids=(endpoint.endpoint_id,),
    ) == (CONTRACT_REPLAY_SUBJECT_KIND_NOT_DERIVABLE,)
    assert replay_contract_admission(
        envelope_type="Endpoint",
        schema_version=CONTRACT_SCHEMA_VERSION,
        payload=payload,
        subject_kind="endpoint_digests",
        subject_ids=[endpoint.digest],  # type: ignore[arg-type]
    ) == (CONTRACT_REPLAY_SUBJECT_IDS_MISMATCH,)


def test_contract_replay_remains_outside_the_public_api():
    assert "replay_contract_admission" not in datadiff_osc.__all__
