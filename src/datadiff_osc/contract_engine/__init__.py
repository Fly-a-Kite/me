"""Stable public API for the OSC HyperContract engine."""

from datadiff_osc.contract_engine.applicability import (
    ApplicabilityCertificate,
    evaluate_applicability,
)
from datadiff_osc.contract_engine.compiler import (
    CompiledContract,
    ProgramSemantics,
    compile_hypercontract,
)
from datadiff_osc.contract_engine.derivation import DerivationCertificate
from datadiff_osc.contract_engine.domains import AbstractState, ObservationDemand
from datadiff_osc.contract_engine.evidence import ObservationCertificate
from datadiff_osc.contract_engine.model import (
    ComponentVerdict,
    Endpoint,
    EndpointRequirement,
    HyperContract,
    Observation,
    RelationObligation,
    SchemaField,
    Verdict,
    VerdictKind,
)
from datadiff_osc.contract_engine.planner import (
    ComparisonPlan,
    execute_staged_comparison,
    plan_comparison,
    execute_staged_request,
)
from datadiff_osc.contract_engine.relations import (
    RelationRegistry,
    default_relation_registry,
)
from datadiff_osc.contract_engine.rules import (
    AGGREGATE_KINDS,
    EXPRESSION_KINDS,
    OPERATION_KINDS,
    ContractRuleRegistry,
    ContractTransformer,
    SemanticStep,
    UnknownSemanticRule,
    default_rule_registry,
)

__all__ = [
    "AGGREGATE_KINDS",
    "AbstractState",
    "ApplicabilityCertificate",
    "ComparisonPlan",
    "CompiledContract",
    "ComponentVerdict",
    "ContractRuleRegistry",
    "ContractTransformer",
    "DerivationCertificate",
    "EXPRESSION_KINDS",
    "Endpoint",
    "EndpointRequirement",
    "HyperContract",
    "OPERATION_KINDS",
    "Observation",
    "ObservationCertificate",
    "ObservationDemand",
    "ProgramSemantics",
    "RelationObligation",
    "RelationRegistry",
    "SchemaField",
    "SemanticStep",
    "UnknownSemanticRule",
    "Verdict",
    "VerdictKind",
    "compile_hypercontract",
    "default_relation_registry",
    "default_rule_registry",
    "evaluate_applicability",
    "execute_staged_comparison",
    "execute_staged_request",
    "plan_comparison",
]
