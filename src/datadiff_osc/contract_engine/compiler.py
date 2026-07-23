from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datadiff_osc._canonical import stable_digest, to_primitive
from datadiff_osc.contract_engine.demand import (
    BackwardAnalysis,
    backward_analyze,
    demand_for_relation,
)
from datadiff_osc.contract_engine.derivation import DerivationCertificate
from datadiff_osc.contract_engine.domains import AbstractState, ObservationDemand
from datadiff_osc.contract_engine.forward import ForwardAnalysis, forward_analyze
from datadiff_osc.contract_engine.model import (
    EndpointRequirement,
    HyperContract,
    RelationObligation,
)
from datadiff_osc.contract_engine.capability import SemanticPermission
from datadiff_osc.contract_engine.overlays import semantic_permission_errors
from datadiff_osc.contract_engine.rules import (
    ContractRuleRegistry,
    SemanticStep,
    default_rule_registry,
)
from datadiff_osc.contract_engine.relations import (
    RelationRegistry,
    default_relation_registry,
    relation_registry_binding_errors,
    validate_relation_obligation,
)


@dataclass(frozen=True, slots=True)
class ProgramSemantics:
    program_digest: str
    initial_state: AbstractState
    steps: tuple[SemanticStep, ...]

    @property
    def digest(self) -> str:
        return stable_digest("osc-program-semantics", self)


@dataclass(frozen=True, slots=True)
class CompiledContract:
    contract: HyperContract
    derivation: DerivationCertificate
    forward: ForwardAnalysis
    backward: BackwardAnalysis

    @property
    def valid(self) -> bool:
        authority_registry = default_relation_registry()
        authority_fact = "contract_authority:" + _contract_authority_digest(
            endpoint_requirements=self.contract.endpoint_requirements,
            preconditions=self.contract.preconditions,
            observations=self.contract.observations,
            obligations=self.contract.obligations,
            strength=self.contract.strength,
            registry_digest=self.contract.registry_digest,
        )
        expected_premises = tuple(
            item
            for item in self.contract.preconditions
            if item not in {"no_unresolved_rules", "endpoint_scope_matches"}
        )
        expected_outputs = tuple(
            dict.fromkeys(
                (
                    *(
                        fact
                        for application in self.forward.applications
                        for fact in application.obligations
                    ),
                    authority_fact,
                )
            )
        )
        expected_contract_id = _contract_id(
            self.derivation.program_digest,
            self.contract.obligations[-1].relation_id,
            tuple(item.endpoint_id for item in self.contract.endpoint_requirements),
            self.derivation.digest,
        )
        return (
            self.derivation.valid
            and self.contract.derivation_digest == self.derivation.digest
            and not authority_registry.authority_binding_errors()
            and self.contract.registry_digest == authority_registry.digest
            and self.derivation.forward_digest == self.forward.digest
            and self.derivation.backward_digest == self.backward.digest
            and self.derivation.rule_ids
            == tuple(item.rule_id for item in self.forward.applications)
            and self.derivation.premise_fact_ids == expected_premises
            and self.derivation.output_fact_ids == expected_outputs
            and self.contract.contract_id == expected_contract_id
            and self.derivation.unresolved_facts
            == tuple(
                dict.fromkeys(
                    (*self.forward.unresolved_rules, *self.backward.unresolved_rules)
                )
            )
        )

    @property
    def digest(self) -> str:
        return stable_digest("osc-compiled-contract", self)

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(self)


def compile_hypercontract(
    program: ProgramSemantics,
    *,
    endpoint_requirements: tuple[EndpointRequirement, ...],
    relation_id: str,
    relation_parameters: dict[str, Any] | None = None,
    strength: str = "",
    registry: ContractRuleRegistry | None = None,
    relation_registry: RelationRegistry | None = None,
    overlay_digests: tuple[str, ...] = (),
    final_demand: ObservationDemand | None = None,
    semantic_permissions: tuple[SemanticPermission, ...] = (),
    permission_facts: frozenset[str] = frozenset(),
) -> CompiledContract:
    resolved_rule_registry = registry or default_rule_registry()
    resolved_relation_registry = relation_registry or default_relation_registry()
    forward = forward_analyze(
        program.initial_state, program.steps, resolved_rule_registry
    )
    demand = final_demand or demand_for_relation(relation_id)
    backward = backward_analyze(
        demand,
        program.steps,
        resolved_rule_registry,
        forward_states=forward.states,
    )
    rule_ids = tuple(item.rule_id for item in forward.applications)
    premise_facts = tuple(
        dict.fromkeys(
            fact
            for application in forward.applications
            for fact in application.preconditions
        )
    )
    output_facts = tuple(
        dict.fromkeys(
            fact
            for application in forward.applications
            for fact in application.obligations
        )
    )
    permission_errors = semantic_permission_errors(
        semantic_permissions,
        components=demand.components,
        relation_id=relation_id,
        facts=permission_facts,
        overlay_digests=overlay_digests,
    )
    if not permission_errors:
        premise_facts = tuple(
            dict.fromkeys(
                (
                    *premise_facts,
                    *(
                        f"semantic_permission:{item.permission_id}:"
                        f"{item.affected_component}"
                        for item in semantic_permissions
                    ),
                )
            )
        )
    endpoint_ids = tuple(item.endpoint_id for item in endpoint_requirements)
    obligation_seed = {
        "program": program.program_digest,
        "relation": relation_id,
        "endpoints": endpoint_ids,
    }
    obligations: list[RelationObligation] = []
    explicit_status_relation = relation_id in {
        "status_ok",
        "status_equal",
        "error_category_equal",
        "accept_reject_equal",
    }
    if demand.status and not explicit_status_relation:
        obligations.append(
            RelationObligation.build(
                obligation_id="status-" + stable_digest("osc-obligation-id", {**obligation_seed, "component": "status"})[-24:],
                relation_id="status_ok",
                endpoint_ids=endpoint_ids,
                components=("status",),
                strength="status_ok",
            )
        )
    if demand.schema_names or demand.schema_types or demand.nullability:
        schema_mode = "full" if demand.schema_names and (demand.schema_types or demand.nullability) else "types" if demand.schema_types or demand.nullability else "names"
        obligations.append(
            RelationObligation.build(
                obligation_id="schema-" + stable_digest("osc-obligation-id", {**obligation_seed, "component": "schema"})[-24:],
                relation_id="schema_equal",
                endpoint_ids=endpoint_ids,
                components=("schema_names", "schema_types", "nullability"),
                parameters={"mode": schema_mode},
                strength="schema_exact",
                relation_properties={"reflexive", "symmetric", "transitive"},
            )
        )
    if demand.cardinality and relation_id not in {
        "cardinality_equal", "cardinality_nonincreasing",
        "cardinality_nondecreasing", "cardinality_bounded",
    }:
        obligations.append(
            RelationObligation.build(
                obligation_id="cardinality-" + stable_digest("osc-obligation-id", {**obligation_seed, "component": "cardinality"})[-24:],
                relation_id="cardinality_equal",
                endpoint_ids=endpoint_ids,
                components=("cardinality",),
                strength="cardinality_exact",
                relation_properties={"reflexive", "symmetric", "transitive"},
            )
        )
    obligations.append(
        RelationObligation.build(
            obligation_id=f"relation-{stable_digest('osc-obligation-id', obligation_seed)[-24:]}",
            relation_id=relation_id,
            endpoint_ids=endpoint_ids,
            components=demand.components,
            parameters=relation_parameters,
            strength=strength or relation_id,
            relation_properties=_relation_properties(relation_id),
        )
    )
    relation_errors = tuple(
        error
        for obligation in obligations
        for error in validate_relation_obligation(obligation)
    )
    preconditions = tuple(
        dict.fromkeys(
            (*premise_facts, "no_unresolved_rules", "endpoint_scope_matches")
        )
    )
    authority_fact = "contract_authority:" + _contract_authority_digest(
        endpoint_requirements=endpoint_requirements,
        preconditions=preconditions,
        observations=demand.components,
        obligations=tuple(obligations),
        strength=strength or relation_id,
        registry_digest=resolved_relation_registry.digest,
    )
    output_facts = tuple(dict.fromkeys((*output_facts, authority_fact)))
    unresolved = tuple(
        dict.fromkeys(
            (
                *forward.unresolved_rules,
                *backward.unresolved_rules,
                *(f"relation:{item}" for item in relation_errors),
                *(
                    f"relation_registry:{item}"
                    for item in relation_registry_binding_errors(
                        resolved_relation_registry,
                        default_relation_registry().digest,
                    )
                ),
                *(f"permission:{item}" for item in permission_errors),
            )
        )
    )
    derivation = DerivationCertificate(
        program_digest=program.program_digest,
        rule_registry_digest=resolved_rule_registry.digest,
        forward_digest=forward.digest,
        backward_digest=backward.digest,
        rule_ids=rule_ids,
        premise_fact_ids=premise_facts,
        output_fact_ids=output_facts,
        backend_overlay_digests=tuple(overlay_digests),
        unresolved_facts=unresolved,
    )
    contract = HyperContract(
        contract_id=_contract_id(
            program.program_digest, relation_id, endpoint_ids, derivation.digest
        ),
        endpoint_requirements=endpoint_requirements,
        preconditions=preconditions,
        observations=demand.components,
        obligations=tuple(obligations),
        strength=strength or relation_id,
        derivation_digest=derivation.digest,
        registry_digest=resolved_relation_registry.digest,
    )
    return CompiledContract(contract, derivation, forward, backward)


def _contract_authority_digest(
    *,
    endpoint_requirements: tuple[EndpointRequirement, ...],
    preconditions: tuple[str, ...],
    observations: tuple[str, ...],
    obligations: tuple[RelationObligation, ...],
    strength: str,
    registry_digest: str,
) -> str:
    return stable_digest(
        "osc-contract-authority",
        {
            "endpoint_requirements": endpoint_requirements,
            "preconditions": preconditions,
            "observations": observations,
            "obligations": obligations,
            "strength": strength,
            "registry_digest": registry_digest,
        },
    )


def _contract_id(
    program_digest: str,
    relation_id: str,
    endpoint_ids: tuple[str, ...],
    derivation_digest: str,
) -> str:
    seed = {
        "program": program_digest,
        "relation": relation_id,
        "endpoints": endpoint_ids,
        "derivation": derivation_digest,
    }
    return f"contract-{stable_digest('osc-contract-id', seed)[-32:]}"


def _relation_properties(relation_id: str) -> frozenset[str]:
    if relation_id in {
        "sequence_equal", "bag_equal", "set_equal_unique",
        "schema_equal", "error_category_equal", "numeric_exact",
        "layout_sequence_equal", "layout_bag_equal",
        "mode_sequence_equal", "mode_bag_equal",
    }:
        return frozenset({"reflexive", "symmetric", "transitive"})
    if relation_id in {"containment", "cardinality_nonincreasing", "cardinality_nondecreasing"}:
        return frozenset({"reflexive", "transitive"})
    return frozenset()
