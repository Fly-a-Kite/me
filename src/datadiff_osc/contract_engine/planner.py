from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable

from datadiff_osc._canonical import canonical_json, stable_digest, to_primitive
from datadiff_osc.contract_engine.applicability import (
    ApplicabilityCertificate,
    observation_endpoint_binding_errors,
)
from datadiff_osc.contract_engine.evidence import ObservationCertificate
from datadiff_osc.contract_engine.fingerprints import (
    ComponentFingerprint,
    fingerprint_component,
)
from datadiff_osc.contract_engine.model import (
    ComponentVerdict,
    Endpoint,
    HyperContract,
    Observation,
    Verdict,
    VerdictKind,
)
from datadiff_osc.contract_engine.monitor import (
    execution_outcome_binding_errors,
    monitor_exact,
)
from datadiff_osc.contract_engine.hypergraph import EntailmentGraph
from datadiff_osc.contract_engine.observers import LazyObservationView, ObserverSpec
from datadiff_osc.contract_engine.relations import (
    RelationRegistry,
    default_relation_registry,
    relation_registry_binding_errors,
    validate_relation_obligation,
)
from datadiff_osc.schemas import (
    EvidenceTier,
    ExecutionStatus,
    StagedComparisonRequest,
    StagedComparisonResult,
    StructuredExecutionOutcome,
)


class ComparisonStage(str, Enum):
    S0_STATIC = "S0_STATIC"
    S1_STATUS_SCHEMA_CARDINALITY = "S1_STATUS_SCHEMA_CARDINALITY"
    S2_COMPONENT_FINGERPRINT = "S2_COMPONENT_FINGERPRINT"
    S3_EXACT_MATERIALIZED = "S3_EXACT_MATERIALIZED"
    S4_CONFIRMATION_NATIVE = "S4_CONFIRMATION_NATIVE"


@dataclass(frozen=True, slots=True)
class PlanNode:
    node_id: str
    stage: ComparisonStage
    observer_id: str
    depends_on: tuple[str, ...] = ()
    conditional: bool = False


@dataclass(frozen=True, slots=True)
class ComparisonPlan:
    contract_digest: str
    nodes: tuple[PlanNode, ...]
    evidence_tier: str
    result_cache_allowed: bool
    schema_version: str = "osc-staged-comparison-plan-v1"

    @property
    def digest(self) -> str:
        return stable_digest("osc-comparison-plan", self)

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(self)


def plan_comparison(
    contract: HyperContract,
    *,
    evidence_tier: str = "screening",
    registry: RelationRegistry | None = None,
) -> ComparisonPlan:
    tier = _evidence_tier(evidence_tier)
    resolved = registry or default_relation_registry()
    registry_errors = relation_registry_binding_errors(
        resolved, contract.registry_digest
    )
    if registry_errors:
        raise ValueError(
            "relation registry is not contract-bound authority: "
            + ";".join(registry_errors)
        )
    nodes: list[PlanNode] = [
        PlanNode("s0-applicability", ComparisonStage.S0_STATIC, "applicability"),
        PlanNode(
            "s1-status", ComparisonStage.S1_STATUS_SCHEMA_CARDINALITY,
            "status", ("s0-applicability",),
        ),
        PlanNode(
            "s1-schema", ComparisonStage.S1_STATUS_SCHEMA_CARDINALITY,
            "schema", ("s1-status",),
        ),
        PlanNode(
            "s1-cardinality", ComparisonStage.S1_STATUS_SCHEMA_CARDINALITY,
            "cardinality", ("s1-schema",),
        ),
    ]
    prior = "s1-cardinality"
    for index, obligation in enumerate(contract.obligations):
        try:
            observer = resolved.resolve(obligation.relation_id).fingerprint_observer
        except KeyError:
            observer = "invalid_relation"
        node_id = f"s2-{index:03d}-{observer}"
        nodes.append(
            PlanNode(
                node_id,
                ComparisonStage.S2_COMPONENT_FINGERPRINT,
                observer,
                (prior,),
            )
        )
        prior = node_id
    nodes.append(
        PlanNode(
            "s3-exact", ComparisonStage.S3_EXACT_MATERIALIZED,
            "exact", (prior,), conditional=True,
        )
    )
    if tier in {EvidenceTier.FRESH_CONFIRMATION, EvidenceTier.NATIVE_REPRODUCTION}:
        nodes.append(
            PlanNode(
                "s4-confirmation", ComparisonStage.S4_CONFIRMATION_NATIVE,
                tier.value, ("s3-exact",),
            )
        )
    return ComparisonPlan(
        contract_digest=contract.digest,
        nodes=tuple(nodes),
        evidence_tier=tier.value,
        result_cache_allowed=tier.cache_allowed,
    )


def cluster_fingerprints(
    fingerprints: Iterable[ComponentFingerprint],
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """O(B) insertion clustering; each endpoint fingerprint is visited once."""

    clusters: dict[str, list[str]] = {}
    for item in fingerprints:
        clusters.setdefault(item.payload_digest, []).append(item.endpoint_id)
    return tuple(
        (digest, tuple(sorted(endpoint_ids)))
        for digest, endpoint_ids in sorted(clusters.items())
    )


def execute_staged_comparison(
    contract: HyperContract,
    observations: tuple[Observation, ...],
    applicability: ApplicabilityCertificate,
    *,
    evidence_tier: str = "screening",
    registry: RelationRegistry | None = None,
    endpoints: tuple[Endpoint, ...] | None = None,
    force_exact: bool = False,
    cache_used: bool = False,
    execution_outcomes: tuple[StructuredExecutionOutcome, ...] | None = None,
    entailment_graph: EntailmentGraph | None = None,
) -> ObservationCertificate:
    tier = _evidence_tier(evidence_tier)
    resolved = registry or default_relation_registry()
    plan = plan_comparison(contract, evidence_tier=tier.value, registry=resolved)
    if cache_used and not plan.result_cache_allowed:
        raise ValueError(f"result cache is forbidden for evidence tier {tier.value}")
    applicability_errors = observation_endpoint_binding_errors(
        contract,
        observations,
        applicability,
        endpoints=endpoints,
    )
    typed_outcomes: tuple[StructuredExecutionOutcome, ...] = ()
    if execution_outcomes is None:
        outcome_errors = ("execution_outcomes_missing",)
    elif not isinstance(execution_outcomes, tuple) or any(
        not isinstance(item, StructuredExecutionOutcome)
        for item in execution_outcomes
    ):
        outcome_errors = ("execution_outcomes_not_typed_tuple",)
    else:
        typed_outcomes = execution_outcomes
        outcome_errors = execution_outcome_binding_errors(
            contract,
            observations,
            applicability,
            typed_outcomes,
        )
    if (
        applicability_errors
        or outcome_errors
        or applicability.verdict != VerdictKind.SATISFIED
    ):
        applicability_bound = not applicability_errors
        outcomes_bound = not outcome_errors
        if applicability_errors:
            reason = (
                "applicability certificate is malformed or not contract-bound: "
                + ";".join(applicability_errors)
            )
            trace = "S0 applicability binding stopped comparison"
        elif outcome_errors:
            reason = (
                "structured execution outcomes are missing or not bound: "
                + ";".join(outcome_errors)
            )
            trace = "S0 execution outcome admission stopped comparison"
        else:
            reason = applicability.reason
            trace = "S0 applicability stopped comparison"
        verdict = Verdict(
            VerdictKind.INAPPLICABLE
            if applicability_bound
            and outcomes_bound
            and applicability.verdict == VerdictKind.INAPPLICABLE
            and applicability.unsupported_evidence
            else VerdictKind.INCONCLUSIVE,
            reason,
        )
        return ObservationCertificate(
            contract_digest=contract.digest,
            applicability_digest=applicability.digest,
            executed_endpoint_ids=tuple(item.endpoint_id for item in observations),
            observer_ids=(),
            component_fingerprints=(),
            relation_trace=(trace,),
            comparison_stage=ComparisonStage.S0_STATIC.value,
            exact_escalated=False,
            materialized_endpoint_ids=(),
            verdict=verdict,
            cache_used=cache_used,
        )

    by_id = {item.endpoint_id: item for item in observations}
    views = {key: LazyObservationView(value) for key, value in by_id.items()}
    fingerprints: list[ComponentFingerprint] = []
    trace: list[str] = []
    must_exact = force_exact or tier.requires_exact
    required_ids = tuple(item.endpoint_id for item in contract.endpoint_requirements)
    observation_ids = tuple(item.endpoint_id for item in observations)
    if (
        len(observation_ids) != len(set(observation_ids))
        or set(observation_ids) != set(required_ids)
    ):
        must_exact = True
        trace.append("observation endpoint binding requires exact fail-closed evaluation")
    if any(item.status != ExecutionStatus.OK for item in typed_outcomes):
        must_exact = True
        trace.append(
            "non-OK structured execution outcome requires exact status/error evaluation"
        )
    screened_components: list[ComponentVerdict] = []
    for obligation in contract.obligations:
        validation_errors = validate_relation_obligation(obligation)
        if validation_errors:
            must_exact = True
            trace.append(
                f"{obligation.obligation_id}: invalid relation parameters "
                + ";".join(validation_errors)
            )
            continue
        try:
            definition = resolved.resolve(obligation.relation_id)
        except KeyError:
            must_exact = True
            trace.append(f"unknown relation {obligation.relation_id}; exact monitor will fail closed")
            continue
        selected: list[ComponentFingerprint] = []
        for endpoint_id in obligation.endpoint_ids:
            view = views.get(endpoint_id)
            if view is None:
                must_exact = True
                continue
            try:
                item = fingerprint_component(
                    view,
                    ObserverSpec(
                        definition.fingerprint_observer,
                        definition.fingerprint_observer,
                    ),
                    contract.digest,
                )
            except KeyError:
                must_exact = True
                trace.append(
                    f"{obligation.obligation_id}: unknown observer; exact escalation"
                )
                continue
            fingerprints.append(item)
            selected.append(item)
        clusters = cluster_fingerprints(selected)
        equivalence = {"reflexive", "symmetric", "transitive"} <= definition.properties
        if len(selected) != len(obligation.endpoint_ids) or len(clusters) != 1 or not equivalence:
            must_exact = True
            trace.append(
                f"{obligation.obligation_id}: fingerprint clusters={len(clusters)} equivalence={equivalence}; exact escalation"
            )
        else:
            screened_components.append(
                ComponentVerdict.build(
                    obligation.obligation_id,
                    VerdictKind.SATISFIED,
                    "versioned component fingerprints are equal",
                    {"observer": definition.fingerprint_observer},
                )
            )
            trace.append(f"{obligation.obligation_id}: one fingerprint cluster")

    if must_exact:
        verdict = monitor_exact(
            contract,
            observations,
            applicability,
            registry=resolved,
            endpoints=endpoints,
            execution_outcomes=typed_outcomes,
            entailment_graph=entailment_graph,
        )
        stage = ComparisonStage.S3_EXACT_MATERIALIZED.value
        materialized = tuple(item.endpoint_id for item in observations)
    else:
        verdict = Verdict.aggregate(
            tuple(screened_components), reason="screening fingerprint equality"
        )
        stage = ComparisonStage.S2_COMPONENT_FINGERPRINT.value
        materialized = ()
    return ObservationCertificate(
        contract_digest=contract.digest,
        applicability_digest=applicability.digest,
        executed_endpoint_ids=tuple(item.endpoint_id for item in observations),
        observer_ids=tuple(dict.fromkeys(item.observer_id for item in fingerprints)),
        component_fingerprints=tuple(fingerprints),
        relation_trace=tuple(trace),
        comparison_stage=stage,
        exact_escalated=must_exact,
        materialized_endpoint_ids=materialized,
        verdict=verdict,
        cache_used=cache_used,
    )


def execute_staged_request(
    request: StagedComparisonRequest,
    contract: HyperContract,
    observations: tuple[Observation, ...],
    applicability: ApplicabilityCertificate,
    *,
    registry: RelationRegistry | None = None,
    endpoints: tuple[Endpoint, ...] | None = None,
    cache_used: bool = False,
    execution_outcomes: tuple[StructuredExecutionOutcome, ...] | None = None,
    entailment_graph: EntailmentGraph | None = None,
) -> StagedComparisonResult:
    """Execute and bind the frozen cross-subsystem staged request/result pair."""

    required_ids = tuple(item.endpoint_id for item in contract.endpoint_requirements)
    if request.contract_fingerprint.contract_digest != contract.digest:
        raise ValueError("staged request contract fingerprint mismatch")
    if request.contract_fingerprint.registry_digest != contract.registry_digest:
        raise ValueError("staged request registry fingerprint mismatch")
    if request.endpoint_ids != required_ids:
        raise ValueError("staged request endpoint binding mismatch")
    plan = plan_comparison(
        contract,
        evidence_tier=request.evidence_tier.value,
        registry=registry,
    )
    planned_observers = tuple(
        dict.fromkeys(
            node.observer_id
            for node in plan.nodes
            if node.observer_id not in {"applicability", "exact"}
            and node.stage != ComparisonStage.S4_CONFIRMATION_NATIVE
        )
    )
    if not set(request.observer_ids) <= set(planned_observers):
        raise ValueError("staged request contains an observer outside the plan")
    certificate = execute_staged_comparison(
        contract,
        observations,
        applicability,
        evidence_tier=request.evidence_tier.value,
        registry=registry,
        endpoints=endpoints,
        force_exact=request.force_exact,
        cache_used=cache_used,
        execution_outcomes=execution_outcomes,
        entailment_graph=entailment_graph,
    )
    return StagedComparisonResult(
        request_digest=request.digest,
        plan_digest=plan.digest,
        observation_certificate_digest=certificate.digest,
        evidence_tier=request.evidence_tier,
        comparison_stage=certificate.comparison_stage,
        exact_escalated=certificate.exact_escalated,
        endpoint_order=certificate.executed_endpoint_ids,
        component_fingerprint_digests=tuple(
            item.digest for item in certificate.component_fingerprints
        ),
        verdict_kind=certificate.verdict.kind,
    )


def _evidence_tier(value: str | EvidenceTier) -> EvidenceTier:
    if isinstance(value, EvidenceTier):
        return value
    try:
        return EvidenceTier(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"unknown evidence tier: {value!r}") from exc
