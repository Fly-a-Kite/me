from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import datadiff_osc
import datadiff_osc._phase6_gate_authority as authority_module
import pytest
from datadiff_osc._canonical import canonical_envelope, canonical_json, stable_digest
from datadiff_osc.contract_engine._phase6_comparison_receipts import (
    CONTRACT_COMPARISON_RECEIPT_SCHEMA_VERSION,
    CanonicalEndpointValue,
    ContractComparisonReceipt,
)
from datadiff_osc.contract_engine.evidence import ObservationCertificate
from datadiff_osc.contract_engine.fingerprints import ComponentFingerprint
from datadiff_osc.contract_engine.model import (
    ComponentVerdict,
    Endpoint,
    EndpointRequirement,
    HyperContract,
    RelationObligation,
    Verdict,
)
from datadiff_osc.contract_engine.planner import plan_comparison
from datadiff_osc.contract_engine.relations import default_relation_registry
from datadiff_osc.contract_engine.replay import (
    CONTRACT_REPLAY_PAYLOAD_INVALID,
    CONTRACT_REPLAY_SCHEMA_VERSION_MISMATCH,
    CONTRACT_REPLAY_SUBJECT_IDS_MISMATCH,
    replay_contract_admission,
)
from datadiff_osc.schemas import (
    ContractFingerprint,
    EvidenceTier,
    ResourceTokens,
    ResultGroup,
    StagedComparisonRequest,
    StagedComparisonResult,
    TaskIdentity,
    TaskKind,
    TaskSpec,
    VerdictKind,
)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _contract() -> HyperContract:
    endpoint_ids = ("endpoint-a", "endpoint-b")
    requirements = tuple(
        EndpointRequirement(
            endpoint_id=endpoint_id,
            backend="backend-a",
            version_spec="1.2.3",
            execution_modes=frozenset({"eager"}),
            physical_layouts=frozenset({"contiguous"}),
            required_capabilities=frozenset({"execute"}),
        )
        for endpoint_id in endpoint_ids
    )
    obligation = RelationObligation.build(
        obligation_id="status-obligation",
        relation_id="status_ok",
        endpoint_ids=endpoint_ids,
        components=("status",),
        strength="status_exact",
        relation_properties={"reflexive", "symmetric", "transitive"},
    )
    return HyperContract(
        contract_id="phase6-comparison-contract",
        endpoint_requirements=requirements,
        preconditions=("no_unresolved_rules", "endpoint_scope_matches"),
        observations=("status",),
        obligations=(obligation,),
        strength="status_exact",
        derivation_digest=stable_digest("phase6-comparison-derivation", "v1"),
        registry_digest=default_relation_registry().digest,
    )


def _endpoints(source_case_digest: str) -> tuple[Endpoint, ...]:
    return tuple(
        Endpoint.build(
            endpoint_id=endpoint_id,
            case_digest=source_case_digest,
            backend="backend-a",
            backend_version="1.2.3",
            adapter_revision="adapter-r1",
            execution_mode="eager",
            physical_layout="contiguous",
            optimizer_config={"ordinal": ordinal},
            capabilities={"execute"},
        )
        for ordinal, endpoint_id in enumerate(("endpoint-a", "endpoint-b"))
    )


def _identity(kind: TaskKind) -> TaskIdentity:
    return TaskIdentity(
        protocol_digest=stable_digest("phase6-comparison-protocol", "v1"),
        task_kind=kind,
        epoch_index=7,
        decision_index=11,
        seed_lineage_digest=stable_digest("phase6-comparison-seed", "v1"),
        contrast_set_id="contrast-phase6",
    )


def _certificate(
    *,
    contract: HyperContract,
    endpoint_ids: tuple[str, ...],
    fingerprints: tuple[ComponentFingerprint, ...],
    exact: bool,
    equal: bool,
) -> ObservationCertificate:
    component = ComponentVerdict.build(
        "status-obligation",
        VerdictKind.SATISFIED if equal else VerdictKind.VIOLATED,
        "raw comparison outcome",
        {},
    )
    return ObservationCertificate(
        contract_digest=contract.digest,
        applicability_digest=stable_digest("phase6-comparison-applicability", "v1"),
        executed_endpoint_ids=endpoint_ids,
        observer_ids=("status",),
        component_fingerprints=fingerprints,
        relation_trace=("status-obligation: raw comparison",),
        comparison_stage="S3_EXACT_MATERIALIZED" if exact else "S2_COMPONENT_FINGERPRINT",
        exact_escalated=exact,
        materialized_endpoint_ids=endpoint_ids if exact else (),
        verdict=Verdict.aggregate((component,), reason="raw comparison"),
    )


def _receipt(*, canonical_right: object = ("value", "same")) -> ContractComparisonReceipt:
    contract = _contract()
    source_case_digest = stable_digest("phase6-comparison-case", "v1")
    endpoints = _endpoints(source_case_digest)
    endpoint_ids = tuple(item.endpoint_id for item in endpoints)
    fingerprint_identity = _identity(TaskKind.FINGERPRINT_CLUSTER)
    exact_identity = _identity(TaskKind.FULL_DIFF)
    group = ResultGroup(
        result_group_id="phase6-comparison-group",
        task_ids=(fingerprint_identity.task_id, exact_identity.task_id),
        endpoint_ids=endpoint_ids,
        contract_fingerprint=ContractFingerprint(
            contract_digest=contract.digest,
            registry_digest=contract.registry_digest,
        ),
        target_fingerprint=None,
        evidence_tier=EvidenceTier.SCREENING,
        result_order_key=("phase6-comparison", *endpoint_ids),
    )
    fingerprint_request = StagedComparisonRequest(
        result_group_digest=group.digest,
        contract_fingerprint=group.contract_fingerprint,
        endpoint_ids=endpoint_ids,
        observer_ids=("status",),
        evidence_tier=EvidenceTier.SCREENING,
    )
    exact_request = StagedComparisonRequest(
        result_group_digest=group.digest,
        contract_fingerprint=group.contract_fingerprint,
        endpoint_ids=endpoint_ids,
        observer_ids=("status",),
        evidence_tier=EvidenceTier.SCREENING,
        force_exact=True,
    )
    resources = ResourceTokens(
        cpu_tokens=1,
        rss_bytes=0,
        io_class="phase6-test",
        backend_internal_threads=0,
    )
    fingerprint_task = TaskSpec(
        identity=fingerprint_identity,
        dependency_task_ids=(),
        resources=resources,
        payload_digest=fingerprint_request.digest,
    )
    exact_task = TaskSpec(
        identity=exact_identity,
        dependency_task_ids=(fingerprint_identity.task_id,),
        resources=resources,
        payload_digest=exact_request.digest,
    )
    fingerprints = tuple(
        ComponentFingerprint(
            endpoint_id=endpoint_id,
            observer_id="status",
            observer_digest=stable_digest("phase6-comparison-observer", "status"),
            contract_digest=contract.digest,
            row_count=1,
            schema_digest="1" * 64,
            payload_digest="2" * 64,
        )
        for endpoint_id in endpoint_ids
    )
    canonical_values = (
        CanonicalEndpointValue("endpoint-a", ("value", "same")),
        CanonicalEndpointValue("endpoint-b", canonical_right),
    )
    fingerprint_certificate = _certificate(
        contract=contract,
        endpoint_ids=endpoint_ids,
        fingerprints=fingerprints,
        exact=False,
        equal=True,
    )
    exact_equal = canonical_right == ("value", "same")
    exact_certificate = _certificate(
        contract=contract,
        endpoint_ids=endpoint_ids,
        fingerprints=fingerprints,
        exact=True,
        equal=exact_equal,
    )
    plan_digest = plan_comparison(contract, evidence_tier="screening").digest
    fingerprint_result = StagedComparisonResult(
        request_digest=fingerprint_request.digest,
        plan_digest=plan_digest,
        observation_certificate_digest=fingerprint_certificate.digest,
        evidence_tier=EvidenceTier.SCREENING,
        comparison_stage=fingerprint_certificate.comparison_stage,
        exact_escalated=False,
        endpoint_order=endpoint_ids,
        component_fingerprint_digests=tuple(item.digest for item in fingerprints),
        verdict_kind=fingerprint_certificate.verdict.kind,
    )
    exact_result = StagedComparisonResult(
        request_digest=exact_request.digest,
        plan_digest=plan_digest,
        observation_certificate_digest=exact_certificate.digest,
        evidence_tier=EvidenceTier.SCREENING,
        comparison_stage=exact_certificate.comparison_stage,
        exact_escalated=True,
        endpoint_order=endpoint_ids,
        component_fingerprint_digests=tuple(item.digest for item in fingerprints),
        verdict_kind=exact_certificate.verdict.kind,
    )
    return ContractComparisonReceipt(
        source_snapshot_digest=stable_digest("phase6-comparison-source", "v1"),
        source_case_digest=source_case_digest,
        result_group=group,
        contract=contract,
        endpoints=endpoints,
        fingerprint_task=fingerprint_task,
        fingerprint_request=fingerprint_request,
        fingerprint_result=fingerprint_result,
        fingerprint_certificate=fingerprint_certificate,
        exact_task=exact_task,
        exact_request=exact_request,
        exact_result=exact_result,
        exact_certificate=exact_certificate,
        canonical_values=canonical_values,
    )


def _payload(value: object) -> object:
    return json.loads(canonical_json(value))


def _replay(receipt: ContractComparisonReceipt, subject_id: str | None = None):
    return replay_contract_admission(
        envelope_type="ContractComparisonReceipt",
        schema_version=CONTRACT_COMPARISON_RECEIPT_SCHEMA_VERSION,
        payload=_payload(receipt),
        subject_kind="contract_comparison_decision_ids",
        subject_ids=(subject_id or receipt.comparison_decision_id,),
    )


def test_private_receipt_derives_raw_identity_and_replays_exactly():
    receipt = _receipt()

    assert "ContractComparisonReceipt" not in datadiff_osc.__all__
    assert receipt.fingerprint_equal is True
    assert receipt.full_canonical_equal is True
    assert receipt.partitions_equivalent is True
    assert receipt.fingerprint_partition == (("endpoint-a", "endpoint-b"),)
    assert receipt.pairwise_canonical_partition == (("endpoint-a", "endpoint-b"),)
    assert _replay(receipt) == ()


def test_fingerprint_collision_is_typed_failure_evidence_not_a_pass_claim():
    receipt = _receipt(canonical_right=("value", "different"))

    assert receipt.fingerprint_equal is True
    assert receipt.full_canonical_equal is False
    assert receipt.partitions_equivalent is False
    assert receipt.exact_result.verdict_kind is VerdictKind.VIOLATED
    assert _replay(receipt) == ()


def test_source_and_same_cardinality_endpoint_substitutions_cannot_claim_old_identity():
    receipt = _receipt()
    source_changed = replace(
        receipt,
        source_snapshot_digest=stable_digest("phase6-comparison-source", "changed"),
    )
    substituted_endpoint = Endpoint.build(
        endpoint_id="endpoint-b",
        case_digest=receipt.source_case_digest,
        backend="backend-a",
        backend_version="1.2.3",
        adapter_revision="substituted-adapter-r2",
        execution_mode="eager",
        physical_layout="contiguous",
        capabilities={"execute"},
    )
    substituted = replace(
        receipt,
        endpoints=(receipt.endpoints[0], substituted_endpoint),
    )

    for forged in (source_changed, substituted):
        assert forged.comparison_decision_id != receipt.comparison_decision_id
        assert _replay(forged, receipt.comparison_decision_id) == (
            CONTRACT_REPLAY_SUBJECT_IDS_MISMATCH,
        )


def test_constructor_rejects_context_chain_and_exactness_mutants():
    receipt = _receipt()
    mutants = (
        lambda: replace(
            receipt,
            source_case_digest=stable_digest("phase6-comparison-case", "wrong"),
        ),
        lambda: replace(receipt, endpoints=tuple(reversed(receipt.endpoints))),
        lambda: replace(
            receipt,
            endpoints=(
                receipt.endpoints[0],
                Endpoint.build(
                    endpoint_id="endpoint-b",
                    case_digest=receipt.source_case_digest,
                    backend="wrong-backend",
                    backend_version="1.2.3",
                    adapter_revision="adapter-r1",
                    execution_mode="eager",
                    physical_layout="contiguous",
                    capabilities={"execute"},
                ),
            ),
        ),
        lambda: replace(
            receipt,
            fingerprint_task=replace(
                receipt.fingerprint_task,
                identity=replace(
                    receipt.fingerprint_task.identity, task_kind=TaskKind.FULL_DIFF
                ),
            ),
        ),
        lambda: replace(
            receipt,
            fingerprint_task=replace(
                receipt.fingerprint_task,
                payload_digest=stable_digest("forged", "payload"),
            ),
        ),
        lambda: replace(
            receipt,
            exact_task=replace(
                receipt.exact_task,
                identity=replace(
                    receipt.exact_task.identity,
                    seed_lineage_digest=stable_digest(
                        "phase6-comparison-seed", "wrong"
                    ),
                ),
            ),
        ),
        lambda: replace(
            receipt,
            exact_request=replace(receipt.exact_request, force_exact=False),
        ),
    )

    for mutant in mutants:
        with pytest.raises(ValueError):
            mutant()


def test_replay_rejects_missing_extra_digest_only_and_malformed_canonical_values():
    receipt = _receipt()
    payload = _payload(receipt)
    assert isinstance(payload, dict)
    missing = deepcopy(payload)
    del missing["canonical_values"]
    extra = deepcopy(payload)
    extra["fingerprint_equal"] = True
    digest_only = deepcopy(payload)
    digest_only["canonical_values"][0] = {
        "endpoint_id": "endpoint-a",
        "canonical_digest": "0" * 64,
        "schema_version": "osc-phase6-canonical-endpoint-value-v1",
    }
    malformed = deepcopy(payload)
    malformed["canonical_values"][0]["canonical_value"] = {"ambiguous": "mapping"}
    duplicate = deepcopy(payload)
    duplicate["canonical_values"][1]["endpoint_id"] = "endpoint-a"
    bad_request = deepcopy(payload)
    bad_request["fingerprint_request"]["endpoint_ids"] = ["endpoint-a", "endpoint-c"]
    bad_result = deepcopy(payload)
    bad_result["exact_result"]["request_digest"] = stable_digest("forged", "result")
    bad_certificate = deepcopy(payload)
    bad_certificate["fingerprint_certificate"]["executed_endpoint_ids"] = [
        "endpoint-a"
    ]
    bad_seed = deepcopy(payload)
    bad_seed["exact_task"]["identity"]["seed_lineage_digest"] = stable_digest(
        "forged", "seed"
    )
    bad_fingerprint = deepcopy(payload)
    bad_fingerprint["fingerprint_certificate"]["component_fingerprints"][1][
        "payload_digest"
    ] = "3" * 64

    for forged in (
        missing,
        extra,
        digest_only,
        malformed,
        duplicate,
        bad_request,
        bad_result,
        bad_certificate,
        bad_seed,
        bad_fingerprint,
    ):
        assert replay_contract_admission(
            envelope_type="ContractComparisonReceipt",
            schema_version=CONTRACT_COMPARISON_RECEIPT_SCHEMA_VERSION,
            payload=forged,
            subject_kind="contract_comparison_decision_ids",
            subject_ids=(receipt.comparison_decision_id,),
        ) == (CONTRACT_REPLAY_PAYLOAD_INVALID,)


def test_old_schema_and_wrong_subject_fail_closed():
    receipt = _receipt()

    assert replay_contract_admission(
        envelope_type="ContractComparisonReceipt",
        schema_version="osc-phase6-contract-comparison-receipt-v0",
        payload=_payload(receipt),
        subject_kind="contract_comparison_decision_ids",
        subject_ids=(receipt.comparison_decision_id,),
    ) == (CONTRACT_REPLAY_SCHEMA_VERSION_MISMATCH,)
    assert _replay(receipt, stable_digest("substituted-decision", "v1")) == (
        CONTRACT_REPLAY_SUBJECT_IDS_MISMATCH,
    )


def _verified_root_comparison_admission(
    tmp_path: Path,
    *,
    canonical_right: object = ("value", "different"),
):
    receipt = _receipt(canonical_right=canonical_right)
    envelope = canonical_envelope(
        "ContractComparisonReceipt",
        CONTRACT_COMPARISON_RECEIPT_SCHEMA_VERSION,
        receipt,
    )
    path = tmp_path / "comparison-receipt.json"
    path.write_text(envelope, encoding="utf-8")
    errors: list[str] = []
    verified = authority_module._verify_typed_admission(
        root=tmp_path.resolve(),
        producer_kind="contract_exact_replay",
        value={
            "admission_id": "private-comparison-receipt",
            "subject_kind": "contract_comparison_decision_ids",
            "subject_ids": [receipt.comparison_decision_id],
            "envelope_path": path.name,
            "envelope_sha256": _sha256(envelope.encode()),
            "envelope_type": "ContractComparisonReceipt",
            "envelope_schema_version": CONTRACT_COMPARISON_RECEIPT_SCHEMA_VERSION,
        },
        index=0,
        errors=errors,
    )

    assert errors == []
    assert verified is not None
    return receipt, path, verified


def test_root_retains_a_valid_private_receipt_only_for_later_exact_binding(
    tmp_path: Path,
):
    receipt, _, verified = _verified_root_comparison_admission(tmp_path)

    assert verified.subject_ids == (receipt.comparison_decision_id,)
    assert verified.envelope_type == "ContractComparisonReceipt"


@pytest.mark.parametrize(
    "field",
    (
        "source_snapshot_digest",
        "source_case_digest",
        "result_group_id",
        "result_group_digest",
        "contract_fingerprint_digest",
        "endpoint_set_digest",
        "fingerprint_task_id",
        "fingerprint_result_digest",
        "exact_task_id",
        "exact_result_digest",
        "fingerprint_partition_digest",
        "canonical_partition_digest",
        "producer_receipt_digest",
    ),
)
def test_root_rejects_each_substituted_comparison_dynamic_binding_field(
    tmp_path: Path,
    field: str,
):
    receipt, _, verified = _verified_root_comparison_admission(tmp_path)
    producer = authority_module.VerifiedProducerReceipt(
        "comparison-producer",
        "contract_exact_replay",
        "comparison-artifact",
        (verified,),
    )
    record = authority_module._contract_comparison_dynamic_record(
        receipt,
        producer_receipt_digest=producer.digest,
    )
    record[field] = stable_digest("comparison-dynamic-substitute", field)
    errors: list[str] = []

    authority_module._verify_contract_comparison_dynamic_bindings(
        source_digest=receipt.source_snapshot_digest,
        records=(record,),
        producer_receipt=producer,
        errors=errors,
    )

    assert errors == [
        "contract_comparison_record_binding_mismatch:"
        f"{receipt.comparison_decision_id}:{field}"
    ]


def test_root_rejects_receipt_identity_duplicate_and_post_admission_mutation(
    tmp_path: Path,
):
    receipt, path, verified = _verified_root_comparison_admission(tmp_path)
    duplicate = replace(
        verified,
        admission_id="private-comparison-receipt-duplicate",
    )
    duplicate_producer = authority_module.VerifiedProducerReceipt(
        "comparison-producer",
        "contract_exact_replay",
        "comparison-artifact",
        (verified, duplicate),
    )
    duplicate_record = authority_module._contract_comparison_dynamic_record(
        receipt,
        producer_receipt_digest=duplicate_producer.digest,
    )
    duplicate_errors: list[str] = []

    authority_module._verify_contract_comparison_dynamic_bindings(
        source_digest=receipt.source_snapshot_digest,
        records=(duplicate_record,),
        producer_receipt=duplicate_producer,
        errors=duplicate_errors,
    )

    assert duplicate_errors == [
        "contract_comparison_duplicate_decision_admission:"
        f"{receipt.comparison_decision_id}"
    ]

    producer = authority_module.VerifiedProducerReceipt(
        "comparison-producer",
        "contract_exact_replay",
        "comparison-artifact",
        (verified,),
    )
    record = authority_module._contract_comparison_dynamic_record(
        receipt,
        producer_receipt_digest=producer.digest,
    )
    path.write_text("{}", encoding="utf-8")
    mutation_errors: list[str] = []

    authority_module._verify_contract_comparison_dynamic_bindings(
        source_digest=receipt.source_snapshot_digest,
        records=(record,),
        producer_receipt=producer,
        errors=mutation_errors,
    )

    assert mutation_errors[0] == (
        "contract_comparison_admission_hash_mismatch:private-comparison-receipt"
    )
    assert mutation_errors[1] == (
        "contract_comparison_record_not_admitted:"
        f"{receipt.comparison_decision_id}"
    )


def test_root_rejects_source_rebind_and_noncomparison_admission_shape(
    tmp_path: Path,
):
    receipt, _, verified = _verified_root_comparison_admission(tmp_path)
    producer = authority_module.VerifiedProducerReceipt(
        "comparison-producer",
        "contract_exact_replay",
        "comparison-artifact",
        (verified,),
    )
    record = authority_module._contract_comparison_dynamic_record(
        receipt,
        producer_receipt_digest=producer.digest,
    )
    source_errors: list[str] = []

    authority_module._verify_contract_comparison_dynamic_bindings(
        source_digest=stable_digest("rebound-root-source", "wrong"),
        records=(record,),
        producer_receipt=producer,
        errors=source_errors,
    )

    assert source_errors == [
        "contract_comparison_receipt_source_snapshot_mismatch:"
        f"{receipt.comparison_decision_id}"
    ]

    wrong_shape = replace(verified, envelope_type="ResultGroup")
    wrong_producer = authority_module.VerifiedProducerReceipt(
        "comparison-producer",
        "contract_exact_replay",
        "comparison-artifact",
        (wrong_shape,),
    )
    wrong_record = authority_module._contract_comparison_dynamic_record(
        receipt,
        producer_receipt_digest=wrong_producer.digest,
    )
    shape_errors: list[str] = []

    authority_module._verify_contract_comparison_dynamic_bindings(
        source_digest=receipt.source_snapshot_digest,
        records=(wrong_record,),
        producer_receipt=wrong_producer,
        errors=shape_errors,
    )

    assert shape_errors[0] == (
        "contract_comparison_admission_shape_mismatch:private-comparison-receipt"
    )
    assert shape_errors[1] == (
        "contract_comparison_record_not_admitted:"
        f"{receipt.comparison_decision_id}"
    )


def test_root_binds_a_collision_as_non_authoritative_evidence_only(tmp_path: Path):
    receipt, _, verified = _verified_root_comparison_admission(tmp_path)
    producer = authority_module.VerifiedProducerReceipt(
        "comparison-producer",
        "contract_exact_replay",
        "comparison-artifact",
        (verified,),
    )
    record = authority_module._contract_comparison_dynamic_record(
        receipt,
        producer_receipt_digest=producer.digest,
    )
    errors: list[str] = []

    authority_module._verify_contract_comparison_dynamic_bindings(
        source_digest=receipt.source_snapshot_digest,
        records=(record,),
        producer_receipt=producer,
        errors=errors,
    )

    assert receipt.fingerprint_equal is True
    assert receipt.full_canonical_equal is False
    assert receipt.partitions_equivalent is False
    assert errors == []
