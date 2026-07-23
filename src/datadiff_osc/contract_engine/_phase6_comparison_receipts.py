"""Private typed evidence for one full-canonical Contract comparison decision.

The receipt is intentionally not part of the public API.  It establishes only
intrinsic, replayable comparison facts; Root must still bind a receipt to an
authorized producer and a frozen corpus before it can contribute to a gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datadiff_osc._canonical import (
    assert_deeply_immutable,
    canonical_json,
    stable_digest,
    to_primitive,
)
from datadiff_osc.contract_engine.evidence import ObservationCertificate
from datadiff_osc.contract_engine.model import Endpoint, HyperContract
from datadiff_osc.contract_engine.overlays import version_matches
from datadiff_osc.contract_engine.planner import plan_comparison
from datadiff_osc.schemas import (
    ResultGroup,
    StagedComparisonRequest,
    StagedComparisonResult,
    TaskKind,
    TaskSpec,
    VerdictKind,
)


CANONICAL_ENDPOINT_VALUE_SCHEMA_VERSION = (
    "osc-phase6-canonical-endpoint-value-v1"
)
CONTRACT_COMPARISON_RECEIPT_SCHEMA_VERSION = (
    "osc-phase6-contract-comparison-receipt-v1"
)


def _require_text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_tuple(name: str, value: object, expected_type: type) -> tuple:
    if not isinstance(value, tuple) or not value:
        raise ValueError(f"{name} must be a non-empty tuple")
    if any(not isinstance(item, expected_type) for item in value):
        raise ValueError(f"{name} must contain only {expected_type.__name__}")
    return value


def _partition(
    items: tuple[tuple[str, str], ...],
) -> tuple[tuple[str, ...], ...]:
    groups: dict[str, list[str]] = {}
    for endpoint_id, key in items:
        groups.setdefault(key, []).append(endpoint_id)
    return tuple(sorted(tuple(sorted(group)) for group in groups.values()))


@dataclass(frozen=True, slots=True)
class CanonicalEndpointValue:
    """The raw canonical value used by the explicit full-diff comparator."""

    endpoint_id: str
    canonical_value: object
    schema_version: str = CANONICAL_ENDPOINT_VALUE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_text("canonical endpoint ID", self.endpoint_id)
        if self.schema_version != CANONICAL_ENDPOINT_VALUE_SCHEMA_VERSION:
            raise ValueError("canonical endpoint value schema version mismatch")
        try:
            assert_deeply_immutable(self.canonical_value)
            canonical_json(self.canonical_value)
        except (TypeError, ValueError) as exc:
            raise ValueError("canonical endpoint value must be deeply immutable") from exc

    @property
    def digest(self) -> str:
        return stable_digest("osc-phase6-canonical-endpoint-value", self)


@dataclass(frozen=True, slots=True)
class ContractComparisonReceipt:
    """One raw fingerprint-versus-full-canonical comparison decision.

    ``fingerprint_equal`` and ``full_canonical_equal`` are derived properties.
    A collision is therefore valid evidence of a future gate failure, never a
    caller-authored claim that can be relabelled as success.
    """

    source_snapshot_digest: str
    source_case_digest: str
    result_group: ResultGroup
    contract: HyperContract
    endpoints: tuple[Endpoint, ...]
    fingerprint_task: TaskSpec
    fingerprint_request: StagedComparisonRequest
    fingerprint_result: StagedComparisonResult
    fingerprint_certificate: ObservationCertificate
    exact_task: TaskSpec
    exact_request: StagedComparisonRequest
    exact_result: StagedComparisonResult
    exact_certificate: ObservationCertificate
    canonical_values: tuple[CanonicalEndpointValue, ...]
    schema_version: str = CONTRACT_COMPARISON_RECEIPT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_text("source snapshot digest", self.source_snapshot_digest)
        _require_text("source case digest", self.source_case_digest)
        if self.schema_version != CONTRACT_COMPARISON_RECEIPT_SCHEMA_VERSION:
            raise ValueError("contract comparison receipt schema version mismatch")
        if not isinstance(self.result_group, ResultGroup):
            raise ValueError("comparison receipt requires a ResultGroup")
        if not isinstance(self.contract, HyperContract):
            raise ValueError("comparison receipt requires a HyperContract")
        endpoints = _require_tuple("comparison endpoints", self.endpoints, Endpoint)
        canonical_values = _require_tuple(
            "canonical endpoint values", self.canonical_values, CanonicalEndpointValue
        )
        endpoint_ids = tuple(item.endpoint_id for item in endpoints)
        if endpoint_ids != tuple(sorted(endpoint_ids)):
            raise ValueError("comparison endpoints must use canonical endpoint ordering")
        if len(endpoint_ids) != len(set(endpoint_ids)):
            raise ValueError("comparison endpoints must be unique")
        if self.result_group.endpoint_ids != endpoint_ids:
            raise ValueError("result group endpoint order mismatch")
        if tuple(item.endpoint_id for item in self.contract.endpoint_requirements) != endpoint_ids:
            raise ValueError("contract endpoint requirement order mismatch")
        for requirement, endpoint in zip(self.contract.endpoint_requirements, endpoints):
            if requirement.backend not in {"*", endpoint.backend}:
                raise ValueError("endpoint backend contract binding mismatch")
            if not version_matches(endpoint.backend_version, requirement.version_spec):
                raise ValueError("endpoint version contract binding mismatch")
            if (
                requirement.execution_modes
                and endpoint.execution_mode not in requirement.execution_modes
            ):
                raise ValueError("endpoint execution-mode contract binding mismatch")
            if (
                requirement.physical_layouts
                and endpoint.physical_layout not in requirement.physical_layouts
            ):
                raise ValueError("endpoint layout contract binding mismatch")
            if not requirement.required_capabilities <= endpoint.capabilities:
                raise ValueError("endpoint capability contract binding mismatch")
        if any(item.case_digest != self.source_case_digest for item in endpoints):
            raise ValueError("endpoint source-case binding mismatch")
        if (
            self.result_group.contract_fingerprint.contract_digest
            != self.contract.digest
            or self.result_group.contract_fingerprint.registry_digest
            != self.contract.registry_digest
        ):
            raise ValueError("result group contract fingerprint mismatch")
        if tuple(item.endpoint_id for item in canonical_values) != endpoint_ids:
            raise ValueError("canonical endpoint value ordering mismatch")
        if len({item.endpoint_id for item in canonical_values}) != len(canonical_values):
            raise ValueError("canonical endpoint values must be unique")

        self._validate_chain(
            label="fingerprint",
            task=self.fingerprint_task,
            request=self.fingerprint_request,
            result=self.fingerprint_result,
            certificate=self.fingerprint_certificate,
            expected_task_kind=TaskKind.FINGERPRINT_CLUSTER,
            require_exact=False,
            endpoint_ids=endpoint_ids,
        )
        self._validate_chain(
            label="exact",
            task=self.exact_task,
            request=self.exact_request,
            result=self.exact_result,
            certificate=self.exact_certificate,
            expected_task_kind=TaskKind.FULL_DIFF,
            require_exact=True,
            endpoint_ids=endpoint_ids,
        )
        self._validate_pair_context()
        if self.fingerprint_result.verdict_kind != self._equality_verdict(
            self.fingerprint_equal
        ):
            raise ValueError("fingerprint result verdict contradicts raw fingerprints")
        if self.exact_result.verdict_kind != self._equality_verdict(
            self.full_canonical_equal
        ):
            raise ValueError("exact result verdict contradicts raw canonical values")

    def _validate_chain(
        self,
        *,
        label: str,
        task: TaskSpec,
        request: StagedComparisonRequest,
        result: StagedComparisonResult,
        certificate: ObservationCertificate,
        expected_task_kind: TaskKind,
        require_exact: bool,
        endpoint_ids: tuple[str, ...],
    ) -> None:
        if not isinstance(task, TaskSpec):
            raise ValueError(f"{label} chain requires a TaskSpec")
        if not isinstance(request, StagedComparisonRequest):
            raise ValueError(f"{label} chain requires a StagedComparisonRequest")
        if not isinstance(result, StagedComparisonResult):
            raise ValueError(f"{label} chain requires a StagedComparisonResult")
        if not isinstance(certificate, ObservationCertificate) or not certificate.valid:
            raise ValueError(f"{label} chain requires a valid ObservationCertificate")
        if task.identity.task_kind is not expected_task_kind:
            raise ValueError(f"{label} task kind mismatch")
        if task.payload_digest != request.digest:
            raise ValueError(f"{label} task payload does not bind request")
        if request.result_group_digest != self.result_group.digest:
            raise ValueError(f"{label} request result group mismatch")
        if request.contract_fingerprint != self.result_group.contract_fingerprint:
            raise ValueError(f"{label} request contract fingerprint mismatch")
        if request.endpoint_ids != endpoint_ids:
            raise ValueError(f"{label} request endpoint order mismatch")
        if certificate.contract_digest != self.contract.digest:
            raise ValueError(f"{label} certificate contract mismatch")
        if certificate.executed_endpoint_ids != endpoint_ids:
            raise ValueError(f"{label} certificate endpoint order mismatch")
        if certificate.observer_ids != request.observer_ids:
            raise ValueError(f"{label} request observer binding mismatch")
        if result.request_digest != request.digest:
            raise ValueError(f"{label} result request binding mismatch")
        if result.observation_certificate_digest != certificate.digest:
            raise ValueError(f"{label} result certificate binding mismatch")
        if result.evidence_tier is not request.evidence_tier:
            raise ValueError(f"{label} result evidence tier mismatch")
        if result.endpoint_order != endpoint_ids:
            raise ValueError(f"{label} result endpoint order mismatch")
        if result.comparison_stage != certificate.comparison_stage:
            raise ValueError(f"{label} result comparison stage mismatch")
        if result.exact_escalated is not certificate.exact_escalated:
            raise ValueError(f"{label} result exact-escalation mismatch")
        if result.verdict_kind is not certificate.verdict.kind:
            raise ValueError(f"{label} result verdict mismatch")
        if result.component_fingerprint_digests != tuple(
            item.digest for item in certificate.component_fingerprints
        ):
            raise ValueError(f"{label} result fingerprint binding mismatch")
        expected_plan_digest = plan_comparison(
            self.contract, evidence_tier=request.evidence_tier.value
        ).digest
        if result.plan_digest != expected_plan_digest:
            raise ValueError(f"{label} result plan binding mismatch")
        if require_exact:
            if not request.force_exact:
                raise ValueError("full-diff request must force exact execution")
            if not result.exact_escalated:
                raise ValueError("full-diff result must record exact escalation")
            if certificate.comparison_stage != "S3_EXACT_MATERIALIZED":
                raise ValueError("full-diff certificate must be exact-materialized")
            if certificate.materialized_endpoint_ids != endpoint_ids:
                raise ValueError("full-diff materialized endpoint binding mismatch")
        else:
            if request.force_exact or request.evidence_tier.requires_exact:
                raise ValueError("fingerprint request cannot claim exact authority")
            if result.exact_escalated or certificate.materialized_endpoint_ids:
                raise ValueError("fingerprint result cannot claim materialization")
            if certificate.comparison_stage != "S2_COMPONENT_FINGERPRINT":
                raise ValueError("fingerprint certificate must remain at S2")

    def _validate_pair_context(self) -> None:
        fingerprint_identity = self.fingerprint_task.identity
        exact_identity = self.exact_task.identity
        if fingerprint_identity.task_id == exact_identity.task_id:
            raise ValueError("comparison tasks must be distinct")
        if self.fingerprint_task.dependency_task_ids:
            raise ValueError("fingerprint task cannot depend on an unbound task")
        if self.exact_task.dependency_task_ids != (fingerprint_identity.task_id,):
            raise ValueError("full-diff task must depend on fingerprint task")
        for name in (
            "protocol_digest",
            "epoch_index",
            "decision_index",
            "seed_lineage_digest",
            "contrast_set_id",
            "endpoint_id",
            "backend",
        ):
            if getattr(fingerprint_identity, name) != getattr(exact_identity, name):
                raise ValueError(f"comparison task {name} mismatch")
        if self.fingerprint_result.digest == self.exact_result.digest:
            raise ValueError("comparison results must be distinct")
        if self.fingerprint_certificate.digest == self.exact_certificate.digest:
            raise ValueError("comparison certificates must be distinct")
        if set((self.fingerprint_task.identity.task_id, self.exact_task.identity.task_id)) - set(
            self.result_group.task_ids
        ):
            raise ValueError("result group does not bind both comparison tasks")
        fingerprints = self.fingerprint_certificate.component_fingerprints
        endpoint_ids = tuple(item.endpoint_id for item in self.endpoints)
        if len(fingerprints) != len(endpoint_ids):
            raise ValueError("fingerprint chain requires one raw fingerprint per endpoint")
        if tuple(item.endpoint_id for item in fingerprints) != endpoint_ids:
            raise ValueError("fingerprint component endpoint order mismatch")
        if len({item.observer_id for item in fingerprints}) != 1:
            raise ValueError("fingerprint chain requires one canonical observer")
        if any(item.contract_digest != self.contract.digest for item in fingerprints):
            raise ValueError("fingerprint component contract mismatch")
        if (
            self.exact_certificate.component_fingerprints
            != self.fingerprint_certificate.component_fingerprints
        ):
            raise ValueError("full-diff fingerprint component binding mismatch")

    @staticmethod
    def _equality_verdict(value: bool) -> VerdictKind:
        return VerdictKind.SATISFIED if value else VerdictKind.VIOLATED

    @property
    def endpoint_set_digest(self) -> str:
        return stable_digest(
            "osc-phase6-contract-ordered-endpoint-set-v1",
            tuple((item.endpoint_id, item.digest) for item in self.endpoints),
        )

    @property
    def fingerprint_partition(self) -> tuple[tuple[str, ...], ...]:
        return _partition(
            tuple(
                (item.endpoint_id, item.payload_digest)
                for item in self.fingerprint_certificate.component_fingerprints
            )
        )

    @property
    def pairwise_canonical_partition(self) -> tuple[tuple[str, ...], ...]:
        return _partition(
            tuple(
                (item.endpoint_id, canonical_json(item.canonical_value))
                for item in self.canonical_values
            )
        )

    @property
    def fingerprint_equal(self) -> bool:
        return len(self.fingerprint_partition) == 1

    @property
    def full_canonical_equal(self) -> bool:
        return len(self.pairwise_canonical_partition) == 1

    @property
    def partitions_equivalent(self) -> bool:
        return self.fingerprint_partition == self.pairwise_canonical_partition

    @property
    def comparison_decision_id(self) -> str:
        return stable_digest(
            "osc-phase6-contract-comparison-decision-id-v1",
            {
                "source_snapshot_digest": self.source_snapshot_digest,
                "source_case_digest": self.source_case_digest,
                "result_group_digest": self.result_group.digest,
                "contract_fingerprint_digest": self.result_group.contract_fingerprint.digest,
                "endpoint_set_digest": self.endpoint_set_digest,
                "fingerprint_task_id": self.fingerprint_task.identity.task_id,
                "fingerprint_result_id": self.fingerprint_result.digest,
                "fingerprint_component_digests": tuple(
                    item.digest
                    for item in self.fingerprint_certificate.component_fingerprints
                ),
                "exact_task_id": self.exact_task.identity.task_id,
                "exact_result_id": self.exact_result.digest,
                "canonical_value_digests": tuple(
                    item.digest for item in self.canonical_values
                ),
                "schema_version": self.schema_version,
            },
        )

    @property
    def digest(self) -> str:
        return stable_digest("osc-phase6-contract-comparison-receipt", self)

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(self)
