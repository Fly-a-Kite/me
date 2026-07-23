"""Stable semantic boundary used by generation, execution, and evidence layers."""

from datadiff.semantic_core.activation import (
    SEMANTIC_ACTIVATION_SCHEMA_VERSION,
    SemanticActivation,
    evaluate_semantic_activation,
)
from datadiff.semantic_core.contract_coverage import audit_contract_coverage
from datadiff.semantic_core.contract_test_evidence import (
    build_p6_contract_test_evidence,
)

__all__ = [
    "SEMANTIC_ACTIVATION_SCHEMA_VERSION",
    "SemanticActivation",
    "audit_contract_coverage",
    "build_p6_contract_test_evidence",
    "evaluate_semantic_activation",
]
