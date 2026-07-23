"""Private fail-closed semantic replay for Contract-owned envelopes.

This module reconstructs only the Contract types assigned by the Phase-6
replay contract.  Successful replay proves an exact typed/canonical object and
an intrinsic subject match; it does not grant gate or evidence authority.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

from datadiff_osc._replay_support import (
    ReplayValidationError,
    assert_payload_roundtrip,
    replay_enum,
    replay_immutable_value,
    require_bool,
    require_exact_mapping,
    require_nonnegative_int,
    require_text,
    require_tuple,
)
from datadiff_osc.contract_engine.applicability import ApplicabilityCertificate
from datadiff_osc.contract_engine.capability import (
    CapabilityDecision,
    UnsupportedEvidence,
)
from datadiff_osc.contract_engine._phase6_comparison_receipts import (
    CANONICAL_ENDPOINT_VALUE_SCHEMA_VERSION,
    CONTRACT_COMPARISON_RECEIPT_SCHEMA_VERSION,
    CanonicalEndpointValue,
    ContractComparisonReceipt,
)
from datadiff_osc.contract_engine.evidence import (
    ExactEscalation,
    ObservationCertificate,
)
from datadiff_osc.contract_engine.fingerprints import (
    FINGERPRINT_SCHEMA_VERSION,
    HASH_ALGORITHM_VERSION,
    ComponentFingerprint,
)
from datadiff_osc.contract_engine.model import (
    CONTRACT_SCHEMA_VERSION,
    ComponentVerdict,
    Endpoint,
    EndpointRequirement,
    HyperContract,
    RelationObligation,
    Verdict,
)
from datadiff_osc.contract_engine.relations import (
    default_relation_registry,
    relation_registry_binding_errors,
    validate_relation_obligation,
)
from datadiff_osc.schemas import (
    CANONICAL_SCHEMA_VERSION,
    ContractFingerprint,
    EvidenceTier,
    RESOURCE_TOKEN_SCHEMA_VERSION,
    ResultGroup,
    ResourceTokens,
    STAGED_COMPARISON_SCHEMA_VERSION,
    StagedComparisonRequest,
    StagedComparisonResult,
    TASK_SCHEMA_VERSION,
    TargetFingerprint,
    TaskIdentity,
    TaskKind,
    TaskSpec,
    VerdictKind,
)


CONTRACT_REPLAY_ENVELOPE_NOT_ALLOWED = "contract_replay_envelope_not_allowed"
CONTRACT_REPLAY_SCHEMA_VERSION_MISMATCH = (
    "contract_replay_schema_version_mismatch"
)
CONTRACT_REPLAY_PAYLOAD_INVALID = "contract_replay_payload_invalid"
CONTRACT_REPLAY_CANONICAL_MISMATCH = "contract_replay_canonical_mismatch"
CONTRACT_REPLAY_SUBJECT_KIND_NOT_DERIVABLE = (
    "contract_replay_subject_kind_not_derivable"
)
CONTRACT_REPLAY_SUBJECT_IDS_MISMATCH = "contract_replay_subject_ids_mismatch"


_APPLICABILITY_SCHEMA_VERSION = "osc-applicability-certificate-v1"
_CAPABILITY_DECISION_SCHEMA_VERSION = "osc-capability-decision-v1"
_UNSUPPORTED_EVIDENCE_SCHEMA_VERSION = "osc-unsupported-evidence-v1"
_OBSERVATION_CERTIFICATE_SCHEMA_VERSION = "osc-observation-certificate-v1"
_EXACT_ESCALATION_SCHEMA_VERSION = "osc-exact-escalation-v1"
_RESULT_GROUP_SCHEMA_VERSION = "osc-result-group-v1"
_CONTRACT_FINGERPRINT_SCHEMA_VERSION = "osc-contract-fingerprint-v1"
_TARGET_FINGERPRINT_SCHEMA_VERSION = "osc-target-fingerprint-v1"
_TASK_SPEC_SCHEMA_VERSION = "osc-task-spec-v1"


_ReplayValue: TypeAlias = (
    HyperContract
    | Endpoint
    | ResultGroup
    | ApplicabilityCertificate
    | ObservationCertificate
    | ExactEscalation
    | ContractComparisonReceipt
)
_Replayer: TypeAlias = Callable[[object], _ReplayValue]


_EXPECTED_SCHEMA_VERSIONS = {
    "HyperContract": CONTRACT_SCHEMA_VERSION,
    # Endpoint is part of the frozen HyperContract envelope family and has no
    # schema_version field of its own.
    "Endpoint": CONTRACT_SCHEMA_VERSION,
    "ResultGroup": _RESULT_GROUP_SCHEMA_VERSION,
    "ApplicabilityCertificate": _APPLICABILITY_SCHEMA_VERSION,
    "ObservationCertificate": _OBSERVATION_CERTIFICATE_SCHEMA_VERSION,
    "ExactEscalationCertificate": _EXACT_ESCALATION_SCHEMA_VERSION,
    "ContractComparisonReceipt": CONTRACT_COMPARISON_RECEIPT_SCHEMA_VERSION,
}

_CONTEXT_ONLY_SUBJECT_KINDS = frozenset(
    {
        "confirmed_roots_total",
        "high_risk_mutants_total",
        "false_positive_fixes",
    }
)


def _invalid(path: str, detail: str) -> ReplayValidationError:
    return ReplayValidationError(f"{path}: {detail}")


def _require_string(value: object, *, path: str) -> str:
    if not isinstance(value, str):
        raise _invalid(path, "expected a string")
    return value


def _replay_text(value: object, path: str) -> str:
    return require_text(value, path=path)


def _text_tuple(
    value: object,
    *,
    path: str,
    min_length: int = 0,
    unique: bool = False,
) -> tuple[str, ...]:
    return tuple(
        require_tuple(
            value,
            path=path,
            item_replayer=_replay_text,
            min_length=min_length,
            unique=unique,
        )
    )


def _text_frozenset(value: object, *, path: str) -> frozenset[str]:
    values = require_tuple(
        value,
        path=path,
        item_replayer=_replay_text,
        unique=True,
        sorted_values=True,
    )
    return frozenset(values)


def _replay_pair(value: object, path: str) -> tuple[str, object]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise _invalid(path, "expected a canonical key/value pair")
    return (
        require_text(value[0], path=f"{path}[0]"),
        replay_immutable_value(value[1], f"{path}[1]"),
    )


def _replay_pairs(value: object, *, path: str) -> tuple[tuple[str, object], ...]:
    pairs = tuple(
        require_tuple(value, path=path, item_replayer=_replay_pair)
    )
    keys = tuple(item[0] for item in pairs)
    if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
        raise _invalid(path, "keys must be unique and canonically sorted")
    return pairs


def _require_exact_version(
    value: object,
    expected: str,
    *,
    path: str,
) -> str:
    version = require_text(value, path=path)
    if version != expected:
        raise _invalid(path, "unsupported schema version")
    return version


def _require_sha256(value: object, *, path: str) -> str:
    digest = require_text(value, path=path)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise _invalid(path, "expected a lowercase SHA-256 digest")
    return digest


def _reconstruct_endpoint_requirement(
    payload: object,
    path: str,
) -> EndpointRequirement:
    value = require_exact_mapping(
        payload,
        fields={
            "endpoint_id",
            "backend",
            "version_spec",
            "execution_modes",
            "physical_layouts",
            "required_capabilities",
        },
        path=path,
    )
    return EndpointRequirement(
        endpoint_id=require_text(value["endpoint_id"], path=f"{path}.endpoint_id"),
        backend=require_text(value["backend"], path=f"{path}.backend"),
        version_spec=require_text(
            value["version_spec"], path=f"{path}.version_spec"
        ),
        execution_modes=_text_frozenset(
            value["execution_modes"], path=f"{path}.execution_modes"
        ),
        physical_layouts=_text_frozenset(
            value["physical_layouts"], path=f"{path}.physical_layouts"
        ),
        required_capabilities=_text_frozenset(
            value["required_capabilities"],
            path=f"{path}.required_capabilities",
        ),
    )


def _reconstruct_relation_obligation(
    payload: object,
    path: str,
) -> RelationObligation:
    value = require_exact_mapping(
        payload,
        fields={
            "obligation_id",
            "relation_id",
            "endpoint_ids",
            "components",
            "parameters",
            "strength",
            "relation_properties",
        },
        path=path,
    )
    return RelationObligation(
        obligation_id=require_text(
            value["obligation_id"], path=f"{path}.obligation_id"
        ),
        relation_id=require_text(
            value["relation_id"], path=f"{path}.relation_id"
        ),
        endpoint_ids=_text_tuple(
            value["endpoint_ids"],
            path=f"{path}.endpoint_ids",
            min_length=1,
            unique=True,
        ),
        components=_text_tuple(
            value["components"],
            path=f"{path}.components",
            min_length=1,
            unique=True,
        ),
        parameters=_replay_pairs(
            value["parameters"], path=f"{path}.parameters"
        ),
        strength=_require_string(value["strength"], path=f"{path}.strength"),
        relation_properties=_text_frozenset(
            value["relation_properties"], path=f"{path}.relation_properties"
        ),
    )


def _reconstruct_hypercontract(payload: object) -> HyperContract:
    path = "payload.HyperContract"
    value = require_exact_mapping(
        payload,
        fields={
            "contract_id",
            "endpoint_requirements",
            "preconditions",
            "observations",
            "obligations",
            "strength",
            "derivation_digest",
            "registry_digest",
            "schema_version",
        },
        path=path,
    )
    requirements = tuple(
        require_tuple(
            value["endpoint_requirements"],
            path=f"{path}.endpoint_requirements",
            item_replayer=_reconstruct_endpoint_requirement,
            min_length=1,
        )
    )
    obligations = tuple(
        require_tuple(
            value["obligations"],
            path=f"{path}.obligations",
            item_replayer=_reconstruct_relation_obligation,
            min_length=1,
        )
    )
    contract = HyperContract(
        contract_id=require_text(value["contract_id"], path=f"{path}.contract_id"),
        endpoint_requirements=requirements,
        preconditions=_text_tuple(
            value["preconditions"], path=f"{path}.preconditions", unique=True
        ),
        observations=_text_tuple(
            value["observations"],
            path=f"{path}.observations",
            min_length=1,
            unique=True,
        ),
        obligations=obligations,
        strength=require_text(value["strength"], path=f"{path}.strength"),
        derivation_digest=require_text(
            value["derivation_digest"], path=f"{path}.derivation_digest"
        ),
        registry_digest=require_text(
            value["registry_digest"], path=f"{path}.registry_digest"
        ),
        schema_version=_require_exact_version(
            value["schema_version"],
            CONTRACT_SCHEMA_VERSION,
            path=f"{path}.schema_version",
        ),
    )
    registry = default_relation_registry()
    if relation_registry_binding_errors(registry, contract.registry_digest):
        raise _invalid(path, "relation registry is not the trusted authority")
    for obligation in contract.obligations:
        try:
            definition = registry.resolve(obligation.relation_id)
        except KeyError as exc:
            raise _invalid(path, "obligation uses an unknown relation") from exc
        if validate_relation_obligation(obligation):
            raise _invalid(path, "relation obligation is semantically invalid")
        if not obligation.relation_properties <= definition.properties:
            raise _invalid(path, "relation properties exceed trusted properties")
    return contract


def _reconstruct_endpoint(payload: object) -> Endpoint:
    path = "payload.Endpoint"
    value = require_exact_mapping(
        payload,
        fields={
            "endpoint_id",
            "case_digest",
            "backend",
            "backend_version",
            "adapter_revision",
            "execution_mode",
            "physical_layout",
            "optimizer_config",
            "capabilities",
        },
        path=path,
    )
    return Endpoint(
        endpoint_id=require_text(value["endpoint_id"], path=f"{path}.endpoint_id"),
        case_digest=require_text(value["case_digest"], path=f"{path}.case_digest"),
        backend=require_text(value["backend"], path=f"{path}.backend"),
        backend_version=require_text(
            value["backend_version"], path=f"{path}.backend_version"
        ),
        adapter_revision=require_text(
            value["adapter_revision"], path=f"{path}.adapter_revision"
        ),
        execution_mode=require_text(
            value["execution_mode"], path=f"{path}.execution_mode"
        ),
        physical_layout=require_text(
            value["physical_layout"], path=f"{path}.physical_layout"
        ),
        optimizer_config=_replay_pairs(
            value["optimizer_config"], path=f"{path}.optimizer_config"
        ),
        capabilities=_text_frozenset(
            value["capabilities"], path=f"{path}.capabilities"
        ),
    )


def _reconstruct_contract_fingerprint(
    payload: object,
    path: str,
) -> ContractFingerprint:
    value = require_exact_mapping(
        payload,
        fields={
            "contract_digest",
            "registry_digest",
            "canonical_schema_version",
            "schema_version",
        },
        path=path,
    )
    fingerprint = ContractFingerprint(
        contract_digest=require_text(
            value["contract_digest"], path=f"{path}.contract_digest"
        ),
        registry_digest=require_text(
            value["registry_digest"], path=f"{path}.registry_digest"
        ),
        canonical_schema_version=_require_exact_version(
            value["canonical_schema_version"],
            CANONICAL_SCHEMA_VERSION,
            path=f"{path}.canonical_schema_version",
        ),
        schema_version=_require_exact_version(
            value["schema_version"],
            _CONTRACT_FINGERPRINT_SCHEMA_VERSION,
            path=f"{path}.schema_version",
        ),
    )
    if relation_registry_binding_errors(
        default_relation_registry(), fingerprint.registry_digest
    ):
        raise _invalid(path, "fingerprint registry is not the trusted authority")
    return fingerprint


def _reconstruct_target_fingerprint(
    payload: object,
    path: str,
) -> TargetFingerprint:
    value = require_exact_mapping(
        payload,
        fields={
            "universe_digest",
            "taxonomy_digest",
            "template_digest",
            "canonical_schema_version",
            "schema_version",
        },
        path=path,
    )
    return TargetFingerprint(
        universe_digest=require_text(
            value["universe_digest"], path=f"{path}.universe_digest"
        ),
        taxonomy_digest=require_text(
            value["taxonomy_digest"], path=f"{path}.taxonomy_digest"
        ),
        template_digest=require_text(
            value["template_digest"], path=f"{path}.template_digest"
        ),
        canonical_schema_version=_require_exact_version(
            value["canonical_schema_version"],
            CANONICAL_SCHEMA_VERSION,
            path=f"{path}.canonical_schema_version",
        ),
        schema_version=_require_exact_version(
            value["schema_version"],
            _TARGET_FINGERPRINT_SCHEMA_VERSION,
            path=f"{path}.schema_version",
        ),
    )


def _reconstruct_result_group(payload: object) -> ResultGroup:
    path = "payload.ResultGroup"
    value = require_exact_mapping(
        payload,
        fields={
            "result_group_id",
            "task_ids",
            "endpoint_ids",
            "contract_fingerprint",
            "target_fingerprint",
            "evidence_tier",
            "result_order_key",
            "schema_version",
        },
        path=path,
    )
    target_payload = value["target_fingerprint"]
    target_fingerprint = (
        None
        if target_payload is None
        else _reconstruct_target_fingerprint(
            target_payload, f"{path}.target_fingerprint"
        )
    )
    return ResultGroup(
        result_group_id=require_text(
            value["result_group_id"], path=f"{path}.result_group_id"
        ),
        task_ids=_text_tuple(
            value["task_ids"], path=f"{path}.task_ids", min_length=1, unique=True
        ),
        endpoint_ids=_text_tuple(
            value["endpoint_ids"],
            path=f"{path}.endpoint_ids",
            min_length=1,
            unique=True,
        ),
        contract_fingerprint=_reconstruct_contract_fingerprint(
            value["contract_fingerprint"], f"{path}.contract_fingerprint"
        ),
        target_fingerprint=target_fingerprint,
        evidence_tier=replay_enum(
            EvidenceTier, value["evidence_tier"], path=f"{path}.evidence_tier"
        ),
        result_order_key=_text_tuple(
            value["result_order_key"],
            path=f"{path}.result_order_key",
            min_length=1,
        ),
        schema_version=_require_exact_version(
            value["schema_version"],
            _RESULT_GROUP_SCHEMA_VERSION,
            path=f"{path}.schema_version",
        ),
    )


def _reconstruct_unsupported_evidence(
    payload: object,
    path: str,
) -> UnsupportedEvidence:
    value = require_exact_mapping(
        payload,
        fields={
            "endpoint_id",
            "missing_capability",
            "backend_version",
            "adapter_revision",
            "evidence_source",
            "reason",
            "schema_version",
        },
        path=path,
    )
    evidence = UnsupportedEvidence(
        endpoint_id=require_text(value["endpoint_id"], path=f"{path}.endpoint_id"),
        missing_capability=require_text(
            value["missing_capability"], path=f"{path}.missing_capability"
        ),
        backend_version=require_text(
            value["backend_version"], path=f"{path}.backend_version"
        ),
        adapter_revision=require_text(
            value["adapter_revision"], path=f"{path}.adapter_revision"
        ),
        evidence_source=require_text(
            value["evidence_source"], path=f"{path}.evidence_source"
        ),
        reason=require_text(value["reason"], path=f"{path}.reason"),
        schema_version=_require_exact_version(
            value["schema_version"],
            _UNSUPPORTED_EVIDENCE_SCHEMA_VERSION,
            path=f"{path}.schema_version",
        ),
    )
    if not evidence.valid:
        raise _invalid(path, "unsupported evidence is invalid")
    return evidence


def _reconstruct_capability_decision(
    payload: object,
    path: str,
) -> CapabilityDecision:
    value = require_exact_mapping(
        payload,
        fields={
            "endpoint_id",
            "supported",
            "required",
            "actual",
            "missing",
            "evidence",
            "schema_version",
        },
        path=path,
    )
    decision = CapabilityDecision(
        endpoint_id=require_text(value["endpoint_id"], path=f"{path}.endpoint_id"),
        supported=require_bool(value["supported"], path=f"{path}.supported"),
        required=_text_frozenset(value["required"], path=f"{path}.required"),
        actual=_text_frozenset(value["actual"], path=f"{path}.actual"),
        missing=_text_frozenset(value["missing"], path=f"{path}.missing"),
        evidence=tuple(
            require_tuple(
                value["evidence"],
                path=f"{path}.evidence",
                item_replayer=_reconstruct_unsupported_evidence,
            )
        ),
        schema_version=_require_exact_version(
            value["schema_version"],
            _CAPABILITY_DECISION_SCHEMA_VERSION,
            path=f"{path}.schema_version",
        ),
    )
    if not decision.valid:
        raise _invalid(path, "capability decision is invalid")
    return decision


def _reconstruct_applicability_certificate(
    payload: object,
) -> ApplicabilityCertificate:
    path = "payload.ApplicabilityCertificate"
    value = require_exact_mapping(
        payload,
        fields={
            "contract_digest",
            "endpoint_digests",
            "capability_decisions",
            "satisfied_preconditions",
            "missing_preconditions",
            "unsupported_evidence",
            "selected_atoms",
            "activated_atoms",
            "mutation_preserved",
            "verdict",
            "reason",
            "schema_version",
        },
        path=path,
    )
    certificate = ApplicabilityCertificate(
        contract_digest=require_text(
            value["contract_digest"], path=f"{path}.contract_digest"
        ),
        endpoint_digests=_text_tuple(
            value["endpoint_digests"],
            path=f"{path}.endpoint_digests",
            min_length=1,
            unique=True,
        ),
        capability_decisions=tuple(
            require_tuple(
                value["capability_decisions"],
                path=f"{path}.capability_decisions",
                item_replayer=_reconstruct_capability_decision,
            )
        ),
        satisfied_preconditions=_text_tuple(
            value["satisfied_preconditions"],
            path=f"{path}.satisfied_preconditions",
            unique=True,
        ),
        missing_preconditions=_text_tuple(
            value["missing_preconditions"],
            path=f"{path}.missing_preconditions",
            unique=True,
        ),
        unsupported_evidence=tuple(
            require_tuple(
                value["unsupported_evidence"],
                path=f"{path}.unsupported_evidence",
                item_replayer=_reconstruct_unsupported_evidence,
            )
        ),
        selected_atoms=_text_tuple(
            value["selected_atoms"], path=f"{path}.selected_atoms", unique=True
        ),
        activated_atoms=_text_tuple(
            value["activated_atoms"], path=f"{path}.activated_atoms", unique=True
        ),
        mutation_preserved=require_bool(
            value["mutation_preserved"], path=f"{path}.mutation_preserved"
        ),
        verdict=replay_enum(
            VerdictKind, value["verdict"], path=f"{path}.verdict"
        ),
        reason=_require_string(value["reason"], path=f"{path}.reason"),
        schema_version=_require_exact_version(
            value["schema_version"],
            _APPLICABILITY_SCHEMA_VERSION,
            path=f"{path}.schema_version",
        ),
    )
    if not certificate.well_formed:
        raise _invalid(path, "applicability certificate is not well formed")
    return certificate


def _reconstruct_component_fingerprint(
    payload: object,
    path: str,
) -> ComponentFingerprint:
    value = require_exact_mapping(
        payload,
        fields={
            "endpoint_id",
            "observer_id",
            "observer_digest",
            "contract_digest",
            "row_count",
            "schema_digest",
            "payload_digest",
            "hash_algorithm",
            "schema_version",
        },
        path=path,
    )
    hash_algorithm = _require_exact_version(
        value["hash_algorithm"], HASH_ALGORITHM_VERSION, path=f"{path}.hash_algorithm"
    )
    schema_version = _require_exact_version(
        value["schema_version"],
        FINGERPRINT_SCHEMA_VERSION,
        path=f"{path}.schema_version",
    )
    return ComponentFingerprint(
        endpoint_id=require_text(value["endpoint_id"], path=f"{path}.endpoint_id"),
        observer_id=require_text(value["observer_id"], path=f"{path}.observer_id"),
        observer_digest=require_text(
            value["observer_digest"], path=f"{path}.observer_digest"
        ),
        contract_digest=require_text(
            value["contract_digest"], path=f"{path}.contract_digest"
        ),
        row_count=require_nonnegative_int(
            value["row_count"], path=f"{path}.row_count"
        ),
        schema_digest=_require_sha256(
            value["schema_digest"], path=f"{path}.schema_digest"
        ),
        payload_digest=_require_sha256(
            value["payload_digest"], path=f"{path}.payload_digest"
        ),
        hash_algorithm=hash_algorithm,
        schema_version=schema_version,
    )


def _reconstruct_component_verdict(
    payload: object,
    path: str,
) -> ComponentVerdict:
    value = require_exact_mapping(
        payload,
        fields={"component_id", "kind", "reason", "evidence"},
        path=path,
    )
    return ComponentVerdict(
        component_id=require_text(
            value["component_id"], path=f"{path}.component_id"
        ),
        kind=replay_enum(VerdictKind, value["kind"], path=f"{path}.kind"),
        reason=_require_string(value["reason"], path=f"{path}.reason"),
        evidence=_replay_pairs(value["evidence"], path=f"{path}.evidence"),
    )


def _reconstruct_verdict(payload: object, path: str) -> Verdict:
    value = require_exact_mapping(
        payload,
        fields={"kind", "reason", "components"},
        path=path,
    )
    components = tuple(
        require_tuple(
            value["components"],
            path=f"{path}.components",
            item_replayer=_reconstruct_component_verdict,
        )
    )
    component_ids = tuple(item.component_id for item in components)
    if len(component_ids) != len(set(component_ids)):
        raise _invalid(path, "component verdict identities must be unique")
    verdict = Verdict(
        kind=replay_enum(VerdictKind, value["kind"], path=f"{path}.kind"),
        reason=_require_string(value["reason"], path=f"{path}.reason"),
        components=components,
    )
    if components and Verdict.aggregate(components).kind != verdict.kind:
        raise _invalid(path, "aggregate verdict kind contradicts its components")
    return verdict


def _reconstruct_observation_certificate(
    payload: object,
) -> ObservationCertificate:
    path = "payload.ObservationCertificate"
    value = require_exact_mapping(
        payload,
        fields={
            "contract_digest",
            "applicability_digest",
            "executed_endpoint_ids",
            "observer_ids",
            "component_fingerprints",
            "relation_trace",
            "comparison_stage",
            "exact_escalated",
            "materialized_endpoint_ids",
            "verdict",
            "cache_used",
            "schema_version",
        },
        path=path,
    )
    fingerprints = tuple(
        require_tuple(
            value["component_fingerprints"],
            path=f"{path}.component_fingerprints",
            item_replayer=_reconstruct_component_fingerprint,
        )
    )
    observer_ids = _text_tuple(
        value["observer_ids"], path=f"{path}.observer_ids", unique=True
    )
    certificate = ObservationCertificate(
        contract_digest=require_text(
            value["contract_digest"], path=f"{path}.contract_digest"
        ),
        applicability_digest=require_text(
            value["applicability_digest"], path=f"{path}.applicability_digest"
        ),
        executed_endpoint_ids=_text_tuple(
            value["executed_endpoint_ids"],
            path=f"{path}.executed_endpoint_ids",
            min_length=1,
            unique=True,
        ),
        observer_ids=observer_ids,
        component_fingerprints=fingerprints,
        relation_trace=_text_tuple(
            value["relation_trace"], path=f"{path}.relation_trace", min_length=1
        ),
        comparison_stage=require_text(
            value["comparison_stage"], path=f"{path}.comparison_stage"
        ),
        exact_escalated=require_bool(
            value["exact_escalated"], path=f"{path}.exact_escalated"
        ),
        materialized_endpoint_ids=_text_tuple(
            value["materialized_endpoint_ids"],
            path=f"{path}.materialized_endpoint_ids",
            unique=True,
        ),
        verdict=_reconstruct_verdict(value["verdict"], f"{path}.verdict"),
        cache_used=require_bool(value["cache_used"], path=f"{path}.cache_used"),
        schema_version=_require_exact_version(
            value["schema_version"],
            _OBSERVATION_CERTIFICATE_SCHEMA_VERSION,
            path=f"{path}.schema_version",
        ),
    )
    expected_observers = tuple(
        dict.fromkeys(item.observer_id for item in certificate.component_fingerprints)
    )
    if certificate.observer_ids != expected_observers:
        raise _invalid(path, "observer identities do not bind the fingerprints")
    if not certificate.exact_escalated and certificate.materialized_endpoint_ids:
        raise _invalid(path, "non-exact certificate cannot claim materialization")
    if not certificate.valid:
        raise _invalid(path, "observation certificate is invalid")
    return certificate


def _reconstruct_exact_escalation_certificate(payload: object) -> ExactEscalation:
    path = "payload.ExactEscalationCertificate"
    value = require_exact_mapping(
        payload,
        fields={
            "obligation_id",
            "reason",
            "endpoint_ids",
            "materialized_bytes",
            "schema_version",
        },
        path=path,
    )
    escalation = ExactEscalation(
        obligation_id=require_text(
            value["obligation_id"], path=f"{path}.obligation_id"
        ),
        reason=require_text(value["reason"], path=f"{path}.reason"),
        endpoint_ids=_text_tuple(
            value["endpoint_ids"],
            path=f"{path}.endpoint_ids",
            min_length=1,
            unique=True,
        ),
        materialized_bytes=require_nonnegative_int(
            value["materialized_bytes"], path=f"{path}.materialized_bytes"
        ),
        schema_version=_require_exact_version(
            value["schema_version"],
            _EXACT_ESCALATION_SCHEMA_VERSION,
            path=f"{path}.schema_version",
        ),
    )
    if not escalation.valid:
        raise _invalid(path, "exact escalation is invalid")
    return escalation


def _reconstruct_task_identity(payload: object, path: str) -> TaskIdentity:
    value = require_exact_mapping(
        payload,
        fields={
            "protocol_digest",
            "task_kind",
            "epoch_index",
            "decision_index",
            "seed_lineage_digest",
            "contrast_set_id",
            "endpoint_id",
            "backend",
            "attempt",
            "schema_version",
        },
        path=path,
    )
    return TaskIdentity(
        protocol_digest=require_text(
            value["protocol_digest"], path=f"{path}.protocol_digest"
        ),
        task_kind=replay_enum(TaskKind, value["task_kind"], path=f"{path}.task_kind"),
        epoch_index=require_nonnegative_int(
            value["epoch_index"], path=f"{path}.epoch_index"
        ),
        decision_index=require_nonnegative_int(
            value["decision_index"], path=f"{path}.decision_index"
        ),
        seed_lineage_digest=require_text(
            value["seed_lineage_digest"], path=f"{path}.seed_lineage_digest"
        ),
        contrast_set_id=_require_string(
            value["contrast_set_id"], path=f"{path}.contrast_set_id"
        ),
        endpoint_id=_require_string(
            value["endpoint_id"], path=f"{path}.endpoint_id"
        ),
        backend=_require_string(value["backend"], path=f"{path}.backend"),
        attempt=require_nonnegative_int(value["attempt"], path=f"{path}.attempt"),
        schema_version=_require_exact_version(
            value["schema_version"], TASK_SCHEMA_VERSION, path=f"{path}.schema_version"
        ),
    )


def _reconstruct_resource_tokens(payload: object, path: str) -> ResourceTokens:
    value = require_exact_mapping(
        payload,
        fields={
            "cpu_tokens",
            "rss_bytes",
            "io_class",
            "backend_internal_threads",
            "exclusive_state",
            "schema_version",
        },
        path=path,
    )
    return ResourceTokens(
        cpu_tokens=require_nonnegative_int(
            value["cpu_tokens"], path=f"{path}.cpu_tokens"
        ),
        rss_bytes=require_nonnegative_int(value["rss_bytes"], path=f"{path}.rss_bytes"),
        io_class=require_text(value["io_class"], path=f"{path}.io_class"),
        backend_internal_threads=require_nonnegative_int(
            value["backend_internal_threads"],
            path=f"{path}.backend_internal_threads",
        ),
        exclusive_state=_require_string(
            value["exclusive_state"], path=f"{path}.exclusive_state"
        ),
        schema_version=_require_exact_version(
            value["schema_version"],
            RESOURCE_TOKEN_SCHEMA_VERSION,
            path=f"{path}.schema_version",
        ),
    )


def _reconstruct_task_spec(payload: object, path: str) -> TaskSpec:
    value = require_exact_mapping(
        payload,
        fields={
            "identity",
            "dependency_task_ids",
            "resources",
            "payload_digest",
            "schema_version",
        },
        path=path,
    )
    return TaskSpec(
        identity=_reconstruct_task_identity(value["identity"], f"{path}.identity"),
        dependency_task_ids=_text_tuple(
            value["dependency_task_ids"],
            path=f"{path}.dependency_task_ids",
            unique=True,
        ),
        resources=_reconstruct_resource_tokens(value["resources"], f"{path}.resources"),
        payload_digest=require_text(value["payload_digest"], path=f"{path}.payload_digest"),
        schema_version=_require_exact_version(
            value["schema_version"], _TASK_SPEC_SCHEMA_VERSION, path=f"{path}.schema_version"
        ),
    )


def _reconstruct_staged_request(
    payload: object,
    path: str,
) -> StagedComparisonRequest:
    value = require_exact_mapping(
        payload,
        fields={
            "result_group_digest",
            "contract_fingerprint",
            "endpoint_ids",
            "observer_ids",
            "evidence_tier",
            "force_exact",
            "schema_version",
        },
        path=path,
    )
    return StagedComparisonRequest(
        result_group_digest=require_text(
            value["result_group_digest"], path=f"{path}.result_group_digest"
        ),
        contract_fingerprint=_reconstruct_contract_fingerprint(
            value["contract_fingerprint"], f"{path}.contract_fingerprint"
        ),
        endpoint_ids=_text_tuple(
            value["endpoint_ids"], path=f"{path}.endpoint_ids", min_length=1, unique=True
        ),
        observer_ids=_text_tuple(
            value["observer_ids"], path=f"{path}.observer_ids", min_length=1, unique=True
        ),
        evidence_tier=replay_enum(
            EvidenceTier, value["evidence_tier"], path=f"{path}.evidence_tier"
        ),
        force_exact=require_bool(value["force_exact"], path=f"{path}.force_exact"),
        schema_version=_require_exact_version(
            value["schema_version"],
            STAGED_COMPARISON_SCHEMA_VERSION,
            path=f"{path}.schema_version",
        ),
    )


def _reconstruct_staged_result(
    payload: object,
    path: str,
) -> StagedComparisonResult:
    value = require_exact_mapping(
        payload,
        fields={
            "request_digest",
            "plan_digest",
            "observation_certificate_digest",
            "evidence_tier",
            "comparison_stage",
            "exact_escalated",
            "endpoint_order",
            "component_fingerprint_digests",
            "verdict_kind",
            "schema_version",
        },
        path=path,
    )
    return StagedComparisonResult(
        request_digest=require_text(value["request_digest"], path=f"{path}.request_digest"),
        plan_digest=require_text(value["plan_digest"], path=f"{path}.plan_digest"),
        observation_certificate_digest=require_text(
            value["observation_certificate_digest"],
            path=f"{path}.observation_certificate_digest",
        ),
        evidence_tier=replay_enum(
            EvidenceTier, value["evidence_tier"], path=f"{path}.evidence_tier"
        ),
        comparison_stage=require_text(
            value["comparison_stage"], path=f"{path}.comparison_stage"
        ),
        exact_escalated=require_bool(
            value["exact_escalated"], path=f"{path}.exact_escalated"
        ),
        endpoint_order=_text_tuple(
            value["endpoint_order"], path=f"{path}.endpoint_order", min_length=1, unique=True
        ),
        component_fingerprint_digests=_text_tuple(
            value["component_fingerprint_digests"],
            path=f"{path}.component_fingerprint_digests",
        ),
        verdict_kind=replay_enum(
            VerdictKind, value["verdict_kind"], path=f"{path}.verdict_kind"
        ),
        schema_version=_require_exact_version(
            value["schema_version"],
            STAGED_COMPARISON_SCHEMA_VERSION,
            path=f"{path}.schema_version",
        ),
    )


def _reconstruct_canonical_endpoint_value(
    payload: object,
    path: str,
) -> CanonicalEndpointValue:
    value = require_exact_mapping(
        payload,
        fields={"endpoint_id", "canonical_value", "schema_version"},
        path=path,
    )
    return CanonicalEndpointValue(
        endpoint_id=require_text(value["endpoint_id"], path=f"{path}.endpoint_id"),
        canonical_value=replay_immutable_value(
            value["canonical_value"], f"{path}.canonical_value"
        ),
        schema_version=_require_exact_version(
            value["schema_version"],
            CANONICAL_ENDPOINT_VALUE_SCHEMA_VERSION,
            path=f"{path}.schema_version",
        ),
    )


def _reconstruct_contract_comparison_receipt(
    payload: object,
) -> ContractComparisonReceipt:
    path = "payload.ContractComparisonReceipt"
    value = require_exact_mapping(
        payload,
        fields={
            "source_snapshot_digest",
            "source_case_digest",
            "result_group",
            "contract",
            "endpoints",
            "fingerprint_task",
            "fingerprint_request",
            "fingerprint_result",
            "fingerprint_certificate",
            "exact_task",
            "exact_request",
            "exact_result",
            "exact_certificate",
            "canonical_values",
            "schema_version",
        },
        path=path,
    )
    return ContractComparisonReceipt(
        source_snapshot_digest=require_text(
            value["source_snapshot_digest"], path=f"{path}.source_snapshot_digest"
        ),
        source_case_digest=require_text(
            value["source_case_digest"], path=f"{path}.source_case_digest"
        ),
        result_group=_reconstruct_result_group(value["result_group"]),
        contract=_reconstruct_hypercontract(value["contract"]),
        endpoints=tuple(
            require_tuple(
                value["endpoints"],
                path=f"{path}.endpoints",
                item_replayer=lambda item, _path: _reconstruct_endpoint(item),
                min_length=1,
            )
        ),
        fingerprint_task=_reconstruct_task_spec(
            value["fingerprint_task"], f"{path}.fingerprint_task"
        ),
        fingerprint_request=_reconstruct_staged_request(
            value["fingerprint_request"], f"{path}.fingerprint_request"
        ),
        fingerprint_result=_reconstruct_staged_result(
            value["fingerprint_result"], f"{path}.fingerprint_result"
        ),
        fingerprint_certificate=_reconstruct_observation_certificate(
            value["fingerprint_certificate"]
        ),
        exact_task=_reconstruct_task_spec(value["exact_task"], f"{path}.exact_task"),
        exact_request=_reconstruct_staged_request(
            value["exact_request"], f"{path}.exact_request"
        ),
        exact_result=_reconstruct_staged_result(
            value["exact_result"], f"{path}.exact_result"
        ),
        exact_certificate=_reconstruct_observation_certificate(
            value["exact_certificate"]
        ),
        canonical_values=tuple(
            require_tuple(
                value["canonical_values"],
                path=f"{path}.canonical_values",
                item_replayer=_reconstruct_canonical_endpoint_value,
                min_length=1,
            )
        ),
        schema_version=_require_exact_version(
            value["schema_version"],
            CONTRACT_COMPARISON_RECEIPT_SCHEMA_VERSION,
            path=f"{path}.schema_version",
        ),
    )


def _reconstruct_contract_comparison_receipt_payload(
    payload: object,
) -> ContractComparisonReceipt:
    """Reconstruct the private comparison receipt for Root cross-binding.

    This deliberately stays outside ``__all__``: Root needs the Contract
    owner's exact reconstruction for its private dynamic-record binding, while
    the public API continues to expose no Phase-6 comparison-receipt surface.
    """

    reconstructed = _reconstruct_contract_comparison_receipt(payload)
    assert_payload_roundtrip(payload, reconstructed, path="payload")
    return reconstructed


_REPLAYERS: dict[str, _Replayer] = {
    "HyperContract": _reconstruct_hypercontract,
    "Endpoint": _reconstruct_endpoint,
    "ResultGroup": _reconstruct_result_group,
    "ApplicabilityCertificate": _reconstruct_applicability_certificate,
    "ObservationCertificate": _reconstruct_observation_certificate,
    "ExactEscalationCertificate": _reconstruct_exact_escalation_certificate,
    "ContractComparisonReceipt": _reconstruct_contract_comparison_receipt,
}


def _contract_replay_subject_ids(
    envelope_type: str,
    reconstructed: _ReplayValue,
    subject_kind: str,
) -> tuple[str, ...] | None:
    if subject_kind in _CONTEXT_ONLY_SUBJECT_KINDS:
        return None
    if (
        envelope_type == "HyperContract"
        and subject_kind == "hypercontract_digests"
        and isinstance(reconstructed, HyperContract)
    ):
        return (reconstructed.digest,)
    if (
        envelope_type == "Endpoint"
        and subject_kind == "endpoint_digests"
        and isinstance(reconstructed, Endpoint)
    ):
        return (reconstructed.digest,)
    if envelope_type == "ResultGroup" and isinstance(reconstructed, ResultGroup):
        if subject_kind == "result_group_digests":
            return (reconstructed.digest,)
        if subject_kind == "staged_parity_groups":
            return (reconstructed.result_group_id,)
    if (
        envelope_type == "ApplicabilityCertificate"
        and subject_kind == "applicability_certificate_digests"
        and isinstance(reconstructed, ApplicabilityCertificate)
    ):
        return (reconstructed.digest,)
    if (
        envelope_type == "ObservationCertificate"
        and subject_kind == "observation_certificate_digests"
        and isinstance(reconstructed, ObservationCertificate)
    ):
        return (reconstructed.digest,)
    if (
        envelope_type == "ExactEscalationCertificate"
        and subject_kind == "exact_escalation_certificate_digests"
        and isinstance(reconstructed, ExactEscalation)
    ):
        return (reconstructed.digest,)
    if (
        envelope_type == "ContractComparisonReceipt"
        and subject_kind == "contract_comparison_decision_ids"
        and isinstance(reconstructed, ContractComparisonReceipt)
    ):
        return (reconstructed.comparison_decision_id,)
    return None


def replay_contract_admission(
    *,
    envelope_type: str,
    schema_version: str,
    payload: object,
    subject_kind: str,
    subject_ids: tuple[str, ...],
) -> tuple[str, ...]:
    """Replay one Contract envelope and verify only its intrinsic subjects."""

    expected_version = _EXPECTED_SCHEMA_VERSIONS.get(envelope_type)
    if expected_version is None:
        return (CONTRACT_REPLAY_ENVELOPE_NOT_ALLOWED,)
    if schema_version != expected_version:
        return (CONTRACT_REPLAY_SCHEMA_VERSION_MISMATCH,)

    try:
        if envelope_type == "ContractComparisonReceipt":
            reconstructed = _reconstruct_contract_comparison_receipt_payload(
                payload
            )
        else:
            reconstructed = _REPLAYERS[envelope_type](payload)
    except (ReplayValidationError, KeyError, OverflowError, TypeError, ValueError):
        return (CONTRACT_REPLAY_PAYLOAD_INVALID,)

    try:
        if envelope_type != "ContractComparisonReceipt":
            assert_payload_roundtrip(payload, reconstructed, path="payload")
    except (ReplayValidationError, TypeError, ValueError):
        return (CONTRACT_REPLAY_CANONICAL_MISMATCH,)

    derived = _contract_replay_subject_ids(
        envelope_type, reconstructed, subject_kind
    )
    if derived is None:
        return (CONTRACT_REPLAY_SUBJECT_KIND_NOT_DERIVABLE,)
    if (
        not isinstance(subject_ids, tuple)
        or any(not isinstance(item, str) or not item for item in subject_ids)
        or subject_ids != tuple(sorted(subject_ids))
        or len(subject_ids) != len(set(subject_ids))
        or subject_ids != derived
    ):
        return (CONTRACT_REPLAY_SUBJECT_IDS_MISMATCH,)
    return ()


__all__ = ["replay_contract_admission"]
