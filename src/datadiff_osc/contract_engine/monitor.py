from __future__ import annotations

from typing import Iterable

from datadiff_osc.contract_engine.applicability import (
    ApplicabilityCertificate,
    observation_endpoint_binding_errors,
)
from datadiff_osc.contract_engine.model import (
    ComponentVerdict,
    Endpoint,
    HyperContract,
    Observation,
    RelationObligation,
    Verdict,
    VerdictKind,
)
from datadiff_osc.contract_engine.hypergraph import EntailmentGraph
from datadiff_osc.contract_engine.relations import (
    RelationRegistry,
    default_relation_registry,
    relation_registry_binding_errors,
)
from datadiff_osc.schemas import ExecutionStatus, StructuredExecutionOutcome


_ERROR_RELATIONS = frozenset(
    {"status_ok", "status_equal", "error_category_equal", "accept_reject_equal"}
)


def monitor_exact(
    contract: HyperContract,
    observations: Iterable[Observation],
    applicability: ApplicabilityCertificate,
    *,
    registry: RelationRegistry | None = None,
    endpoints: tuple[Endpoint, ...] | None = None,
    execution_outcomes: tuple[StructuredExecutionOutcome, ...] | None = None,
    entailment_graph: EntailmentGraph | None = None,
) -> Verdict:
    resolved_registry = registry or default_relation_registry()
    registry_errors = relation_registry_binding_errors(
        resolved_registry, contract.registry_digest
    )
    if registry_errors:
        return Verdict(
            VerdictKind.INCONCLUSIVE,
            "relation registry is not contract-bound authority: "
            + ";".join(registry_errors),
        )
    materialized = tuple(observations)
    applicability_errors = observation_endpoint_binding_errors(
        contract,
        materialized,
        applicability,
        endpoints=endpoints,
    )
    if applicability_errors:
        return Verdict(
            VerdictKind.INCONCLUSIVE,
            "applicability certificate is malformed or not contract-bound: "
            + ";".join(applicability_errors),
        )
    if execution_outcomes is not None:
        outcome_errors = execution_outcome_binding_errors(
            contract, materialized, applicability, execution_outcomes
        )
        if outcome_errors:
            return Verdict(
                VerdictKind.INCONCLUSIVE,
                "structured execution outcomes are not bound: "
                + ";".join(outcome_errors),
            )
    if applicability.verdict == VerdictKind.INAPPLICABLE:
        if not applicability.unsupported_evidence:
            return Verdict(
                VerdictKind.INCONCLUSIVE,
                "inapplicable verdict lacks bound unsupported-capability evidence",
            )
        return Verdict(VerdictKind.INAPPLICABLE, applicability.reason)
    if applicability.verdict != VerdictKind.SATISFIED:
        return Verdict(VerdictKind.INCONCLUSIVE, applicability.reason)

    observation_ids = tuple(item.endpoint_id for item in materialized)
    required_ids = tuple(item.endpoint_id for item in contract.endpoint_requirements)
    if len(observation_ids) != len(set(observation_ids)):
        return Verdict(
            VerdictKind.INCONCLUSIVE,
            "duplicate endpoint observations violate endpoint binding",
        )
    if set(observation_ids) != set(required_ids):
        missing = sorted(set(required_ids) - set(observation_ids))
        unexpected = sorted(set(observation_ids) - set(required_ids))
        return Verdict.aggregate(
            (
                ComponentVerdict.build(
                    "endpoint-binding",
                    VerdictKind.INCONCLUSIVE,
                    "observation endpoint membership does not match the contract",
                    {"missing": missing, "unexpected": unexpected},
                ),
            ),
            reason="exact contract monitor",
        )
    by_id = {item.endpoint_id: item for item in materialized}
    components: list[ComponentVerdict] = []
    satisfied_obligations: list[RelationObligation] = []
    for obligation in contract.obligations:
        proof = next(
            (
                candidate
                for stronger in satisfied_obligations
                if entailment_graph is not None
                for candidate in (entailment_graph.prove(stronger, obligation),)
                if candidate is not None and candidate.valid
            ),
            None,
        )
        if proof is not None:
            components.append(
                ComponentVerdict.build(
                    obligation.obligation_id,
                    VerdictKind.SATISFIED,
                    "discharged by proof-scoped acyclic entailment",
                    {"entailment_proof_digest": proof.digest},
                )
            )
            satisfied_obligations.append(obligation)
            continue
        selected = tuple(by_id[item] for item in obligation.endpoint_ids if item in by_id)
        if len(selected) != len(obligation.endpoint_ids):
            missing = sorted(set(obligation.endpoint_ids) - set(by_id))
            components.append(
                ComponentVerdict.build(
                    obligation.obligation_id,
                    VerdictKind.INCONCLUSIVE,
                    "required observations are missing",
                    {"missing_endpoints": missing},
                )
            )
            continue
        unsupported = [item.endpoint_id for item in selected if item.status == "unsupported"]
        if unsupported:
            components.append(
                ComponentVerdict.build(
                    obligation.obligation_id,
                    VerdictKind.INCONCLUSIVE,
                    "unsupported observation lacks an inapplicable applicability decision",
                    {"unsupported_endpoints": unsupported},
                )
            )
            continue
        execution_failures = [
            (item.endpoint_id, item.status)
            for item in selected
            if item.status in {"timeout", "crash", "adapter_error", "missing"}
        ]
        if execution_failures:
            components.append(
                ComponentVerdict.build(
                    obligation.obligation_id,
                    VerdictKind.INCONCLUSIVE,
                    "structured execution failure requires retry or external adjudication",
                    {"execution_failures": execution_failures},
                )
            )
            continue
        if (
            obligation.relation_id not in _ERROR_RELATIONS
            and any(item.status != "ok" for item in selected)
        ):
            components.append(
                ComponentVerdict.build(
                    obligation.obligation_id,
                    VerdictKind.INCONCLUSIVE,
                    "value relation cannot consume non-OK semantic observations",
                    {"statuses": [(item.endpoint_id, item.status) for item in selected]},
                )
            )
            continue
        evaluated = resolved_registry.evaluate(obligation, selected)
        components.append(evaluated)
        if evaluated.kind == VerdictKind.SATISFIED:
            satisfied_obligations.append(obligation)
    return Verdict.aggregate(tuple(components), reason="exact contract monitor")


def execution_outcome_binding_errors(
    contract: HyperContract,
    observations: tuple[Observation, ...],
    applicability: ApplicabilityCertificate,
    outcomes: tuple[StructuredExecutionOutcome, ...],
) -> tuple[str, ...]:
    """Validate endpoint/status/evidence binding for frozen runtime outcomes."""

    errors: list[str] = []
    required_ids = tuple(item.endpoint_id for item in contract.endpoint_requirements)
    outcome_ids = tuple(item.endpoint_id for item in outcomes)
    if outcome_ids != required_ids:
        errors.append("execution_outcome_endpoint_order_mismatch")
    if len(outcome_ids) != len(set(outcome_ids)):
        errors.append("duplicate_execution_outcome_endpoint")
    observations_by_id = {item.endpoint_id: item for item in observations}
    evidence_by_digest = {
        item.digest: item for item in applicability.unsupported_evidence
    }
    for outcome in outcomes:
        observation = observations_by_id.get(outcome.endpoint_id)
        if observation is None:
            errors.append(f"outcome_without_observation:{outcome.endpoint_id}")
            continue
        if observation.status != outcome.status.value:
            errors.append(f"outcome_status_mismatch:{outcome.endpoint_id}")
        if outcome.status == ExecutionStatus.UNSUPPORTED:
            evidence = evidence_by_digest.get(outcome.unsupported_evidence_digest)
            if evidence is None or evidence.endpoint_id != outcome.endpoint_id:
                errors.append(
                    f"unsupported_evidence_binding_mismatch:{outcome.endpoint_id}"
                )
    return tuple(dict.fromkeys(errors))
