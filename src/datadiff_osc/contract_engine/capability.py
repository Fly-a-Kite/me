from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datadiff_osc._canonical import stable_digest, to_primitive
from datadiff_osc.contract_engine.model import Endpoint


@dataclass(frozen=True, slots=True)
class UnsupportedEvidence:
    endpoint_id: str
    missing_capability: str
    backend_version: str
    adapter_revision: str
    evidence_source: str
    reason: str
    schema_version: str = "osc-unsupported-evidence-v1"

    @property
    def valid(self) -> bool:
        return all(
            isinstance(item, str) and bool(item)
            for item in (
                self.endpoint_id,
                self.missing_capability,
                self.backend_version,
                self.adapter_revision,
                self.evidence_source,
                self.reason,
            )
        )

    @property
    def digest(self) -> str:
        return stable_digest("osc-unsupported-evidence", self)


@dataclass(frozen=True, slots=True)
class CapabilityDecision:
    endpoint_id: str
    supported: bool
    required: frozenset[str]
    actual: frozenset[str]
    missing: frozenset[str]
    evidence: tuple[UnsupportedEvidence, ...] = ()
    schema_version: str = "osc-capability-decision-v1"

    @property
    def valid(self) -> bool:
        if not self.endpoint_id:
            return False
        if any(
            not isinstance(item, str) or not item
            for item in (*self.required, *self.actual, *self.missing)
        ):
            return False
        expected_missing = self.required - self.actual
        if self.missing != expected_missing or self.supported != (not expected_missing):
            return False
        evidence_capabilities = tuple(item.missing_capability for item in self.evidence)
        return (
            len(evidence_capabilities) == len(set(evidence_capabilities))
            and frozenset(evidence_capabilities) == self.missing
            and all(
                item.valid
                and item.endpoint_id == self.endpoint_id
                and item.missing_capability in self.missing
                for item in self.evidence
            )
        )

    @property
    def digest(self) -> str:
        return stable_digest("osc-capability-decision", self)

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(self)


@dataclass(frozen=True, slots=True)
class SemanticPermission:
    """A semantic permission is deliberately not an execution capability."""

    permission_id: str
    affected_component: str
    relation_id: str
    exact_preconditions: tuple[str, ...]
    overlay_digest: str
    schema_version: str = "osc-semantic-permission-v1"

    @property
    def valid(self) -> bool:
        return (
            all(
                isinstance(item, str) and bool(item)
                for item in (
                    self.permission_id,
                    self.affected_component,
                    self.relation_id,
                    self.overlay_digest,
                )
            )
            and bool(self.exact_preconditions)
            and len(self.exact_preconditions) == len(set(self.exact_preconditions))
            and all(self.exact_preconditions)
        )

    def permits(
        self,
        *,
        component: str,
        relation_id: str,
        facts: frozenset[str],
        overlay_digest: str = "",
    ) -> bool:
        """Check the exact, component-local permission boundary."""

        return (
            self.valid
            and self.affected_component == component
            and self.relation_id == relation_id
            and set(self.exact_preconditions) <= set(facts)
            and (not overlay_digest or self.overlay_digest == overlay_digest)
        )


def decide_capabilities(
    endpoint: Endpoint,
    required: frozenset[str],
    *,
    evidence_source: str = "endpoint_declaration",
) -> CapabilityDecision:
    missing = frozenset(required - endpoint.capabilities)
    evidence = tuple(
        UnsupportedEvidence(
            endpoint_id=endpoint.endpoint_id,
            missing_capability=capability,
            backend_version=endpoint.backend_version,
            adapter_revision=endpoint.adapter_revision,
            evidence_source=evidence_source,
            reason="required capability absent from the versioned endpoint declaration",
        )
        for capability in sorted(missing)
    )
    return CapabilityDecision(
        endpoint_id=endpoint.endpoint_id,
        supported=not missing,
        required=frozenset(required),
        actual=endpoint.capabilities,
        missing=missing,
        evidence=evidence,
    )
