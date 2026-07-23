"""Frozen-request adapter around the HyperContract staged authority entrypoint."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from datadiff_osc.contract_engine import (
    ApplicabilityCertificate,
    Endpoint,
    HyperContract,
    Observation,
    ObservationCertificate,
    execute_staged_comparison,
    plan_comparison,
)
from datadiff_osc.contract_engine.fingerprints import ComponentFingerprint
from datadiff_osc.schemas import (
    StagedComparisonRequest,
    StagedComparisonResult,
    StructuredExecutionOutcome,
)


@dataclass(frozen=True, slots=True)
class FingerprintCluster:
    """One component-fingerprint bucket in frozen endpoint order."""

    observer_id: str
    observer_digest: str
    payload_digest: str
    schema_digest: str
    row_count: int
    hash_algorithm: str
    endpoint_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FingerprintClustering:
    """Linear-time clustering result with explicit uncertainty evidence."""

    clusters: tuple[FingerprintCluster, ...]
    missing_endpoint_ids: tuple[str, ...] = ()
    duplicate_endpoint_ids: tuple[str, ...] = ()
    incompatible_fingerprint_count: int = 0

    @property
    def exact_required(self) -> bool:
        return bool(
            self.missing_endpoint_ids
            or self.duplicate_endpoint_ids
            or self.incompatible_fingerprint_count
            or len(self.clusters) != 1
        )


def cluster_component_fingerprints(
    fingerprints: Iterable[ComponentFingerprint],
    *,
    endpoint_order: tuple[str, ...],
) -> FingerprintClustering:
    """Cluster one observer component in O(B) time without pairwise comparisons.

    Determinism comes from the caller's frozen ``endpoint_order``.  The function
    performs two linear passes and deliberately does not sort fingerprints.
    Missing, duplicate, or observer/contract-incompatible values are uncertainty
    and therefore force exact comparison.
    """

    if not endpoint_order or len(endpoint_order) != len(set(endpoint_order)):
        raise ValueError("endpoint_order must be non-empty and unique")
    allowed_endpoints = set(endpoint_order)
    by_endpoint: dict[str, ComponentFingerprint] = {}
    duplicate: list[str] = []
    reference: tuple[str, str, str, str, str] | None = None
    incompatible = 0
    for item in fingerprints:
        if item.endpoint_id not in allowed_endpoints:
            incompatible += 1
            continue
        if item.endpoint_id in by_endpoint:
            duplicate.append(item.endpoint_id)
            continue
        compatibility = (
            item.observer_id,
            item.observer_digest,
            item.contract_digest,
            item.hash_algorithm,
            item.schema_version,
        )
        if reference is None:
            reference = compatibility
        elif compatibility != reference:
            incompatible += 1
        by_endpoint[item.endpoint_id] = item

    buckets: dict[tuple[str, str, int], list[str]] = {}
    missing: list[str] = []
    for endpoint_id in endpoint_order:
        item = by_endpoint.get(endpoint_id)
        if item is None:
            missing.append(endpoint_id)
            continue
        key = (item.payload_digest, item.schema_digest, item.row_count)
        buckets.setdefault(key, []).append(endpoint_id)

    observer_id, observer_digest, _, hash_algorithm, _ = reference or (
        "",
        "",
        "",
        "",
        "",
    )
    clusters = tuple(
        FingerprintCluster(
            observer_id=observer_id,
            observer_digest=observer_digest,
            payload_digest=key[0],
            schema_digest=key[1],
            row_count=key[2],
            hash_algorithm=hash_algorithm,
            endpoint_ids=tuple(endpoint_ids),
        )
        for key, endpoint_ids in buckets.items()
    )
    return FingerprintClustering(
        clusters=clusters,
        missing_endpoint_ids=tuple(missing),
        duplicate_endpoint_ids=tuple(dict.fromkeys(duplicate)),
        incompatible_fingerprint_count=incompatible,
    )


class StagedComparisonEngine:
    """Bind a frozen request to the contract engine without copying authority."""

    @staticmethod
    def compare(
        request: StagedComparisonRequest,
        contract: HyperContract,
        observations: Iterable[Observation],
        applicability: ApplicabilityCertificate,
        *,
        execution_outcomes: tuple[StructuredExecutionOutcome, ...],
        endpoints: tuple[Endpoint, ...] | None = None,
        cache_used: bool = False,
    ) -> tuple[StagedComparisonResult, ObservationCertificate]:
        if request.contract_fingerprint.contract_digest != contract.digest:
            raise ValueError("comparison request contract digest does not match contract")
        if request.contract_fingerprint.registry_digest != contract.registry_digest:
            raise ValueError("comparison request registry digest does not match contract")
        if applicability.contract_digest != contract.digest:
            raise ValueError("applicability certificate is bound to another contract")
        if cache_used and not request.evidence_tier.cache_allowed:
            raise ValueError(
                f"cache is forbidden for evidence tier {request.evidence_tier.value}"
            )

        materialized = tuple(observations)
        by_id: dict[str, Observation] = {}
        for item in materialized:
            if item.endpoint_id in by_id:
                raise ValueError(f"duplicate observation endpoint: {item.endpoint_id}")
            by_id[item.endpoint_id] = item
        if set(by_id) != set(request.endpoint_ids):
            raise ValueError("comparison observations do not match frozen endpoint IDs")
        ordered = tuple(by_id[endpoint_id] for endpoint_id in request.endpoint_ids)

        expected_observers = tuple(dict.fromkeys(contract.observations))
        observer_uncertainty = set(request.observer_ids) != set(expected_observers)
        force_exact = request.exact_required or observer_uncertainty
        tier = request.evidence_tier.value
        plan = plan_comparison(contract, evidence_tier=tier)
        certificate = execute_staged_comparison(
            contract,
            ordered,
            applicability,
            evidence_tier=tier,
            endpoints=endpoints,
            force_exact=force_exact,
            cache_used=cache_used,
            execution_outcomes=execution_outcomes,
        )
        if certificate.contract_digest != contract.digest:
            raise RuntimeError("contract authority returned a misbound certificate")
        if certificate.applicability_digest != applicability.digest:
            raise RuntimeError("contract authority returned a misbound applicability digest")
        if tuple(certificate.executed_endpoint_ids) != request.endpoint_ids:
            raise RuntimeError("contract authority changed frozen endpoint ordering")
        if (
            force_exact
            and not certificate.exact_escalated
            and certificate.comparison_stage != "S0_STATIC"
        ):
            raise RuntimeError("uncertain or non-screening comparison did not escalate exact")

        result = StagedComparisonResult(
            request_digest=request.digest,
            plan_digest=plan.digest,
            observation_certificate_digest=certificate.digest,
            evidence_tier=request.evidence_tier,
            comparison_stage=certificate.comparison_stage,
            exact_escalated=certificate.exact_escalated,
            endpoint_order=request.endpoint_ids,
            component_fingerprint_digests=tuple(
                item.digest for item in certificate.component_fingerprints
            ),
            verdict_kind=certificate.verdict.kind,
        )
        return result, certificate
