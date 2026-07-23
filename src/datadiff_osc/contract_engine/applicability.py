from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datadiff_osc._canonical import pairs_dict, stable_digest, to_primitive
from datadiff_osc.contract_engine.capability import (
    CapabilityDecision,
    UnsupportedEvidence,
    decide_capabilities,
)
from datadiff_osc.contract_engine.model import (
    Endpoint,
    HyperContract,
    Observation,
    VerdictKind,
)
from datadiff_osc.contract_engine.overlays import version_matches


@dataclass(frozen=True, slots=True)
class ApplicabilityCertificate:
    contract_digest: str
    endpoint_digests: tuple[str, ...]
    capability_decisions: tuple[CapabilityDecision, ...]
    satisfied_preconditions: tuple[str, ...]
    missing_preconditions: tuple[str, ...]
    unsupported_evidence: tuple[UnsupportedEvidence, ...]
    selected_atoms: tuple[str, ...] = ()
    activated_atoms: tuple[str, ...] = ()
    mutation_preserved: bool = True
    verdict: VerdictKind = VerdictKind.INCONCLUSIVE
    reason: str = ""
    schema_version: str = "osc-applicability-certificate-v1"

    @property
    def valid(self) -> bool:
        return self.verdict == VerdictKind.SATISFIED and self.well_formed

    @property
    def well_formed(self) -> bool:
        decision_ids = tuple(item.endpoint_id for item in self.capability_decisions)
        evidence = tuple(
            item for decision in self.capability_decisions for item in decision.evidence
        )
        selected = set(self.selected_atoms)
        activated = set(self.activated_atoms)
        common = (
            self.schema_version == "osc-applicability-certificate-v1"
            and bool(self.contract_digest)
            and bool(self.endpoint_digests)
            and all(self.endpoint_digests)
            and len(self.endpoint_digests) == len(set(self.endpoint_digests))
            and len(decision_ids) == len(set(decision_ids))
            and all(item.valid for item in self.capability_decisions)
            and evidence == self.unsupported_evidence
            and len(self.satisfied_preconditions)
            == len(set(self.satisfied_preconditions))
            and len(self.missing_preconditions) == len(set(self.missing_preconditions))
            and not (
                set(self.satisfied_preconditions) & set(self.missing_preconditions)
            )
            and len(self.selected_atoms) == len(selected)
            and len(self.activated_atoms) == len(activated)
        )
        if not common:
            return False
        if self.verdict == VerdictKind.SATISFIED:
            return (
                bool(self.capability_decisions)
                and len(self.capability_decisions) == len(self.endpoint_digests)
                and not self.missing_preconditions
                and not self.unsupported_evidence
                and all(item.supported for item in self.capability_decisions)
                and self.mutation_preserved
                and selected <= activated
            )
        if self.verdict == VerdictKind.INAPPLICABLE:
            return (
                bool(self.unsupported_evidence)
                and not self.missing_preconditions
                and self.mutation_preserved
                and selected <= activated
                and all(
                    not item.supported
                    for item in self.capability_decisions
                    if item.evidence
                )
            )
        if self.verdict == VerdictKind.INCONCLUSIVE:
            return bool(
                self.missing_preconditions
                or not self.mutation_preserved
                or not selected <= activated
                or not self.reason
            ) or not self.unsupported_evidence
        return False

    @property
    def digest(self) -> str:
        return stable_digest("osc-applicability-certificate", self)

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(self)


def evaluate_applicability(
    contract: HyperContract,
    endpoints: tuple[Endpoint, ...],
    *,
    facts: frozenset[str],
    unresolved_rules: tuple[str, ...] = (),
    selected_atoms: tuple[str, ...] = (),
    activated_atoms: tuple[str, ...] = (),
    mutation_preserved: bool = True,
) -> ApplicabilityCertificate:
    endpoint_ids = tuple(endpoint.endpoint_id for endpoint in endpoints)
    duplicate_endpoint_ids = {
        endpoint_id for endpoint_id in endpoint_ids if endpoint_ids.count(endpoint_id) > 1
    }
    by_id = {endpoint.endpoint_id: endpoint for endpoint in endpoints}
    required_ids = tuple(item.endpoint_id for item in contract.endpoint_requirements)
    decisions: list[CapabilityDecision] = []
    scope_errors: list[str] = []
    if duplicate_endpoint_ids:
        scope_errors.extend(
            f"duplicate_endpoint:{item}" for item in sorted(duplicate_endpoint_ids)
        )
    extra_endpoint_ids = sorted(set(endpoint_ids) - set(required_ids))
    scope_errors.extend(f"unexpected_endpoint:{item}" for item in extra_endpoint_ids)
    for requirement in contract.endpoint_requirements:
        endpoint = by_id.get(requirement.endpoint_id)
        if endpoint is None:
            scope_errors.append(f"missing_endpoint:{requirement.endpoint_id}")
            continue
        scope_errors.extend(_endpoint_scope_errors(requirement, endpoint))
        decisions.append(decide_capabilities(endpoint, requirement.required_capabilities))

    precondition_facts = set(facts)
    if not unresolved_rules:
        precondition_facts.add("no_unresolved_rules")
    else:
        precondition_facts.discard("no_unresolved_rules")
    if not scope_errors:
        precondition_facts.add("endpoint_scope_matches")
    else:
        precondition_facts.discard("endpoint_scope_matches")
    satisfied_preconditions = tuple(
        item for item in contract.preconditions if item in precondition_facts
    )
    missing_preconditions = tuple(
        item for item in contract.preconditions if item not in precondition_facts
    )
    unsupported = tuple(
        evidence for decision in decisions for evidence in decision.evidence
    )
    activation_missing = sorted(set(selected_atoms) - set(activated_atoms))
    if unresolved_rules:
        verdict = VerdictKind.INCONCLUSIVE
        reason = "semantic rules are unresolved: " + ",".join(unresolved_rules)
    elif scope_errors or missing_preconditions or not mutation_preserved or activation_missing:
        verdict = VerdictKind.INCONCLUSIVE
        reason = ";".join(
            (
                *scope_errors,
                *(f"missing_precondition:{item}" for item in missing_preconditions),
                *(() if mutation_preserved else ("mutation_not_preserved",)),
                *(f"selected_atom_not_activated:{item}" for item in activation_missing),
            )
        )
    elif unsupported:
        verdict = VerdictKind.INAPPLICABLE
        reason = "required endpoint capability is explicitly unsupported"
    else:
        verdict = VerdictKind.SATISFIED
        reason = "all endpoint scopes, capabilities, and preconditions are satisfied"
    return ApplicabilityCertificate(
        contract_digest=contract.digest,
        endpoint_digests=tuple(
            by_id[item].digest for item in required_ids if item in by_id
        ),
        capability_decisions=tuple(decisions),
        satisfied_preconditions=satisfied_preconditions,
        missing_preconditions=missing_preconditions,
        unsupported_evidence=unsupported,
        selected_atoms=tuple(selected_atoms),
        activated_atoms=tuple(activated_atoms),
        mutation_preserved=mutation_preserved,
        verdict=verdict,
        reason=reason,
    )


def applicability_binding_errors(
    contract: HyperContract,
    endpoints: tuple[Endpoint, ...],
    certificate: ApplicabilityCertificate,
) -> tuple[str, ...]:
    """Return deterministic fail-closed binding errors for a certificate."""

    errors: list[str] = []
    if not certificate.well_formed:
        errors.append("applicability_certificate_malformed")
    if certificate.contract_digest != contract.digest:
        errors.append("applicability_contract_digest_mismatch")
    endpoint_ids = tuple(item.endpoint_id for item in endpoints)
    required_ids = tuple(item.endpoint_id for item in contract.endpoint_requirements)
    if len(endpoint_ids) != len(set(endpoint_ids)):
        errors.append("duplicate_endpoint_identity")
    if endpoint_ids != required_ids:
        errors.append("endpoint_order_or_membership_mismatch")
    expected_digests = tuple(item.digest for item in endpoints)
    if certificate.endpoint_digests != expected_digests:
        errors.append("endpoint_digest_mismatch")
    decision_ids = tuple(item.endpoint_id for item in certificate.capability_decisions)
    if decision_ids != required_ids:
        errors.append("capability_decision_binding_mismatch")
    for requirement, endpoint, decision in zip(
        contract.endpoint_requirements,
        endpoints,
        certificate.capability_decisions,
    ):
        errors.extend(_endpoint_scope_errors(requirement, endpoint))
        if (
            decision.endpoint_id != endpoint.endpoint_id
            or decision.required != requirement.required_capabilities
            or decision.actual != endpoint.capabilities
            or not decision.valid
        ):
            errors.append(f"capability_decision_invalid:{endpoint.endpoint_id}")
    contract_preconditions = set(contract.preconditions)
    if (
        set(certificate.satisfied_preconditions)
        | set(certificate.missing_preconditions)
    ) != contract_preconditions:
        errors.append("precondition_partition_mismatch")
    return tuple(dict.fromkeys(errors))


def observation_endpoint_binding_errors(
    contract: HyperContract,
    observations: tuple[Observation, ...],
    certificate: ApplicabilityCertificate,
    *,
    endpoints: tuple[Endpoint, ...] | None = None,
) -> tuple[str, ...]:
    """Bind observation provenance to the scoped endpoints and certificate."""

    if endpoints is None:
        return tuple(
            dict.fromkeys(
                (
                    *applicability_contract_binding_errors(contract, certificate),
                    "authority_endpoint_tuple_missing",
                )
            )
        )

    errors = list(applicability_binding_errors(contract, endpoints, certificate))
    expected_digests = {item.endpoint_id: item.digest for item in endpoints}

    metadata_by_endpoint: dict[str, dict[str, Any]] = {}
    for observation in observations:
        try:
            metadata_by_endpoint[observation.endpoint_id] = pairs_dict(
                observation.execution_metadata
            )
        except (TypeError, ValueError):
            errors.append(
                f"observation_execution_metadata_invalid:{observation.endpoint_id}"
            )
    for observation in observations:
        metadata = metadata_by_endpoint.get(observation.endpoint_id, {})
        observed_digest = metadata.get("endpoint_digest")
        if not isinstance(observed_digest, str) or not observed_digest:
            errors.append(
                f"observation_endpoint_digest_missing:{observation.endpoint_id}"
            )
        elif observed_digest != expected_digests.get(observation.endpoint_id):
            errors.append(
                f"observation_endpoint_digest_mismatch:{observation.endpoint_id}"
            )
    return tuple(dict.fromkeys(errors))


def applicability_contract_binding_errors(
    contract: HyperContract,
    certificate: ApplicabilityCertificate,
) -> tuple[str, ...]:
    """Validate all bindings available without the original Endpoint objects."""

    errors: list[str] = []
    if not certificate.well_formed:
        errors.append("applicability_certificate_malformed")
    if certificate.contract_digest != contract.digest:
        errors.append("applicability_contract_digest_mismatch")
    required_ids = tuple(item.endpoint_id for item in contract.endpoint_requirements)
    decision_ids = tuple(item.endpoint_id for item in certificate.capability_decisions)
    if decision_ids != required_ids:
        errors.append("capability_decision_contract_scope_mismatch")
    if len(certificate.endpoint_digests) != len(required_ids):
        errors.append("endpoint_digest_count_mismatch")
    for requirement, decision in zip(
        contract.endpoint_requirements, certificate.capability_decisions
    ):
        if (
            decision.endpoint_id != requirement.endpoint_id
            or decision.required != requirement.required_capabilities
            or not decision.valid
        ):
            errors.append(f"capability_requirement_mismatch:{requirement.endpoint_id}")
    partition = (
        set(certificate.satisfied_preconditions)
        | set(certificate.missing_preconditions)
    )
    if partition != set(contract.preconditions):
        errors.append("precondition_partition_mismatch")
    return tuple(dict.fromkeys(errors))


def _endpoint_scope_errors(requirement: Any, endpoint: Endpoint) -> tuple[str, ...]:
    errors: list[str] = []
    if requirement.backend not in {"*", endpoint.backend}:
        errors.append(f"backend_scope:{requirement.endpoint_id}")
    if not version_matches(endpoint.backend_version, requirement.version_spec):
        errors.append(f"version_scope:{requirement.endpoint_id}")
    if (
        requirement.execution_modes
        and endpoint.execution_mode not in requirement.execution_modes
    ):
        errors.append(f"mode_scope:{requirement.endpoint_id}")
    if (
        requirement.physical_layouts
        and endpoint.physical_layout not in requirement.physical_layouts
    ):
        errors.append(f"layout_scope:{requirement.endpoint_id}")
    return tuple(errors)
