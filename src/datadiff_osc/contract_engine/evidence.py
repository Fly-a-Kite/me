from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datadiff_osc._canonical import stable_digest, to_primitive
from datadiff_osc.contract_engine.fingerprints import ComponentFingerprint
from datadiff_osc.contract_engine.model import Verdict


@dataclass(frozen=True, slots=True)
class ObservationCertificate:
    contract_digest: str
    applicability_digest: str
    executed_endpoint_ids: tuple[str, ...]
    observer_ids: tuple[str, ...]
    component_fingerprints: tuple[ComponentFingerprint, ...]
    relation_trace: tuple[str, ...]
    comparison_stage: str
    exact_escalated: bool
    materialized_endpoint_ids: tuple[str, ...]
    verdict: Verdict
    cache_used: bool = False
    schema_version: str = "osc-observation-certificate-v1"

    @property
    def valid(self) -> bool:
        valid_stages = {
            "S0_STATIC",
            "S1_STATUS_SCHEMA_CARDINALITY",
            "S2_COMPONENT_FINGERPRINT",
            "S3_EXACT_MATERIALIZED",
            "S4_CONFIRMATION_NATIVE",
        }
        exact_stage = self.comparison_stage in {
            "S3_EXACT_MATERIALIZED", "S4_CONFIRMATION_NATIVE"
        }
        return (
            self.schema_version == "osc-observation-certificate-v1"
            and bool(self.contract_digest)
            and bool(self.applicability_digest)
            and bool(self.executed_endpoint_ids)
            and len(self.executed_endpoint_ids) == len(set(self.executed_endpoint_ids))
            and all(self.executed_endpoint_ids)
            and len(self.observer_ids) == len(set(self.observer_ids))
            and all(self.observer_ids)
            and self.comparison_stage in valid_stages
            and self.exact_escalated == exact_stage
            and (
                not exact_stage
                or self.materialized_endpoint_ids == self.executed_endpoint_ids
            )
            and all(
                item.contract_digest == self.contract_digest
                and item.endpoint_id in self.executed_endpoint_ids
                for item in self.component_fingerprints
            )
            and self.verdict.kind.value in {
                "SATISFIED", "VIOLATED", "INAPPLICABLE", "INCONCLUSIVE"
            }
        )

    def binding_errors(
        self,
        *,
        contract_digest: str,
        applicability_digest: str,
        endpoint_ids: tuple[str, ...],
    ) -> tuple[str, ...]:
        errors: list[str] = []
        if not self.valid:
            errors.append("observation_certificate_invalid")
        if self.contract_digest != contract_digest:
            errors.append("observation_contract_digest_mismatch")
        if self.applicability_digest != applicability_digest:
            errors.append("observation_applicability_digest_mismatch")
        if self.executed_endpoint_ids != endpoint_ids:
            errors.append("observation_endpoint_binding_mismatch")
        return tuple(errors)

    @property
    def digest(self) -> str:
        return stable_digest("osc-observation-certificate", self)

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(self)


@dataclass(frozen=True, slots=True)
class ExactEscalation:
    obligation_id: str
    reason: str
    endpoint_ids: tuple[str, ...]
    materialized_bytes: int
    schema_version: str = "osc-exact-escalation-v1"

    @property
    def valid(self) -> bool:
        return (
            bool(self.obligation_id)
            and bool(self.reason)
            and bool(self.endpoint_ids)
            and len(self.endpoint_ids) == len(set(self.endpoint_ids))
            and self.materialized_bytes >= 0
        )

    @property
    def digest(self) -> str:
        return stable_digest("osc-exact-escalation", self)
