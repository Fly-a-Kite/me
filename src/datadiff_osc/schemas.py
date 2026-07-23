"""Frozen cross-subsystem schemas for Oracle-Carrying Semantic Contrast.

This module deliberately has no dependency on contract, target, search, or
runtime implementation modules.  It is the unique source for identities and
evidence records that cross those ownership boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from datadiff_osc._canonical import (
    CANONICAL_SCHEMA_VERSION,
    frozen_pairs,
    stable_digest,
    to_primitive,
)


PUBLIC_SCHEMA_VERSION = "osc-public-schemas-v1"
SEMANTIC_ATOM_SCHEMA_VERSION = "osc-semantic-atom-v1"
SEED_LINEAGE_SCHEMA_VERSION = "osc-seed-lineage-v1"
TASK_SCHEMA_VERSION = "osc-task-v1"
EVIDENCE_SCHEMA_VERSION = "osc-evidence-envelope-v1"
LEDGER_EVENT_SCHEMA_VERSION = "osc-ledger-event-v1"
STAGED_COMPARISON_SCHEMA_VERSION = "osc-staged-comparison-v1"
RUNTIME_CACHE_KEY_SCHEMA_VERSION = "osc-runtime-cache-key-v1"
RESOURCE_TOKEN_SCHEMA_VERSION = "osc-resource-tokens-v1"
INFEASIBLE_EVIDENCE_SCHEMA_VERSION = "osc-infeasible-construction-v1"


def _require_text(name: str, value: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")


def _require_unique(name: str, values: tuple[str, ...]) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{name} must be unique")


class VerdictKind(str, Enum):
    SATISFIED = "SATISFIED"
    VIOLATED = "VIOLATED"
    INAPPLICABLE = "INAPPLICABLE"
    INCONCLUSIVE = "INCONCLUSIVE"


class EvidenceTier(str, Enum):
    SCREENING = "screening"
    AUDIT = "audit"
    FINDING = "finding"
    FRESH_CONFIRMATION = "fresh_confirmation"
    NATIVE_REPRODUCTION = "native_reproduction"

    @property
    def requires_exact(self) -> bool:
        return self is not EvidenceTier.SCREENING

    @property
    def cache_allowed(self) -> bool:
        return self not in {
            EvidenceTier.FRESH_CONFIRMATION,
            EvidenceTier.NATIVE_REPRODUCTION,
        }


class ExecutionStatus(str, Enum):
    OK = "ok"
    SEMANTIC_ERROR = "semantic_error"
    UNSUPPORTED = "unsupported"
    TIMEOUT = "timeout"
    CRASH = "crash"
    ADAPTER_ERROR = "adapter_error"
    MISSING = "missing"


class FailureKind(str, Enum):
    NONE = "none"
    SEMANTIC_DOMAIN_ERROR = "semantic_domain_error"
    UNSUPPORTED_CAPABILITY = "unsupported_capability"
    TIMEOUT = "timeout"
    CRASH = "crash"
    ADAPTER_ERROR = "adapter_error"
    MISSING_RESULT = "missing_result"
    UNKNOWN = "unknown"


class CoverageLevel(str, Enum):
    CONSTRUCTED = "constructed"
    ACTIVATED = "activated"
    EXECUTED = "executed"
    OBSERVED = "observed"


class SeedStage(str, Enum):
    TARGET = "target"
    AXES = "axes"
    CONSTRUCTOR = "constructor"
    DATA = "data"
    BOUNDARY = "boundary"
    MUTATION = "mutation"
    BACKEND = "backend"
    ORACLE_SAMPLE = "oracle_sample"


class TaskKind(str, Enum):
    EPOCH_DECISION = "epoch_decision"
    CONTRAST_CONSTRUCTION = "contrast_construction"
    ENDPOINT_PREFLIGHT = "endpoint_preflight"
    ENDPOINT_ACTIVATION = "endpoint_activation"
    BACKEND_EXECUTION = "backend_execution"
    FINGERPRINT_CLUSTER = "fingerprint_cluster"
    FULL_DIFF = "full_diff"
    FRESH_RECHECK = "fresh_recheck"
    LOCALIZATION = "localization"
    REDUCTION = "reduction"
    NATIVE_REPRODUCTION = "native_reproduction"


class EvidenceState(str, Enum):
    FINDING = "finding"
    STABLE_SURVIVOR = "stable_survivor"
    NATIVE_REPRODUCIBLE = "native_reproducible"
    MINIMIZED = "minimized"
    UNIQUE_ROOT = "unique_root"
    INDEPENDENTLY_CONFIRMED = "independently_confirmed"
    NOT_A_CANDIDATE = "not_a_candidate"


@dataclass(frozen=True, slots=True)
class AtomProvenance:
    source_kind: str
    source_digest: str
    source_path: str
    evidence_digest: str
    schema_version: str = "osc-atom-provenance-v1"

    def __post_init__(self) -> None:
        for name in ("source_kind", "source_digest", "source_path", "evidence_digest"):
            _require_text(name, getattr(self, name))

    @property
    def digest(self) -> str:
        return stable_digest("osc-atom-provenance", self)


@dataclass(frozen=True, slots=True)
class SemanticAtom:
    atom_id: str
    namespace: str
    value: str
    provenance: tuple[AtomProvenance, ...]
    schema_version: str = SEMANTIC_ATOM_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("atom_id", "namespace", "value"):
            _require_text(name, getattr(self, name))
        if not self.atom_id.startswith(f"{self.namespace}:"):
            raise ValueError("semantic atom ID must be namespaced")
        if not self.provenance:
            raise ValueError("semantic atom requires provenance evidence")
        _require_unique("semantic atom provenance", tuple(item.digest for item in self.provenance))

    @property
    def digest(self) -> str:
        return stable_digest("osc-semantic-atom", self)

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(self)


@dataclass(frozen=True, slots=True)
class ContractFingerprint:
    contract_digest: str
    registry_digest: str
    canonical_schema_version: str = CANONICAL_SCHEMA_VERSION
    schema_version: str = "osc-contract-fingerprint-v1"

    def __post_init__(self) -> None:
        _require_text("contract_digest", self.contract_digest)
        _require_text("registry_digest", self.registry_digest)
        if self.canonical_schema_version != CANONICAL_SCHEMA_VERSION:
            raise ValueError("contract fingerprint canonical schema mismatch")

    @property
    def digest(self) -> str:
        return stable_digest("osc-contract-fingerprint", self)


@dataclass(frozen=True, slots=True)
class TargetFingerprint:
    universe_digest: str
    taxonomy_digest: str
    template_digest: str
    canonical_schema_version: str = CANONICAL_SCHEMA_VERSION
    schema_version: str = "osc-target-fingerprint-v1"

    def __post_init__(self) -> None:
        for name in ("universe_digest", "taxonomy_digest", "template_digest"):
            _require_text(name, getattr(self, name))
        if self.canonical_schema_version != CANONICAL_SCHEMA_VERSION:
            raise ValueError("target fingerprint canonical schema mismatch")

    @property
    def digest(self) -> str:
        return stable_digest("osc-target-fingerprint", self)


@dataclass(frozen=True, slots=True)
class SeedLineage:
    protocol_digest: str
    master_seed: int
    lane_id: str
    case_index: int
    stage_name: SeedStage
    counter: int = 0
    parent_digest: str = ""
    schema_version: str = SEED_LINEAGE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_text("protocol_digest", self.protocol_digest)
        _require_text("lane_id", self.lane_id)
        if self.master_seed < 0 or self.case_index < 0 or self.counter < 0:
            raise ValueError("seed lineage numeric fields must be non-negative")

    @property
    def digest(self) -> str:
        return stable_digest("osc-seed-lineage", self)

    @property
    def subseed(self) -> int:
        return int(self.digest.rsplit("-", 1)[-1][:16], 16)

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(self)


@dataclass(frozen=True, slots=True)
class StructuredExecutionOutcome:
    endpoint_id: str
    status: ExecutionStatus
    failure_kind: FailureKind
    reason: str = ""
    unsupported_evidence_digest: str = ""
    schema_version: str = "osc-structured-execution-outcome-v1"

    def __post_init__(self) -> None:
        _require_text("endpoint_id", self.endpoint_id)
        required_failure = {
            ExecutionStatus.OK: FailureKind.NONE,
            ExecutionStatus.SEMANTIC_ERROR: FailureKind.SEMANTIC_DOMAIN_ERROR,
            ExecutionStatus.UNSUPPORTED: FailureKind.UNSUPPORTED_CAPABILITY,
            ExecutionStatus.TIMEOUT: FailureKind.TIMEOUT,
            ExecutionStatus.CRASH: FailureKind.CRASH,
            ExecutionStatus.ADAPTER_ERROR: FailureKind.ADAPTER_ERROR,
            ExecutionStatus.MISSING: FailureKind.MISSING_RESULT,
        }[self.status]
        if self.failure_kind != required_failure:
            raise ValueError(
                f"execution status {self.status.value} requires failure kind "
                f"{required_failure.value}"
            )
        if self.status == ExecutionStatus.UNSUPPORTED and not self.unsupported_evidence_digest:
            raise ValueError("unsupported outcome requires structured capability evidence")
        if self.status != ExecutionStatus.UNSUPPORTED and self.unsupported_evidence_digest:
            raise ValueError("only unsupported outcomes may carry unsupported evidence")

    @property
    def digest(self) -> str:
        return stable_digest("osc-execution-outcome", self)


@dataclass(frozen=True, slots=True)
class InfeasibleConstructionEvidence:
    target_fingerprint: TargetFingerprint
    assignment_digest: str
    reason_code: str
    missing_requirements: tuple[str, ...]
    attempts: int
    trace_digest: str
    schema_version: str = INFEASIBLE_EVIDENCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("assignment_digest", "reason_code", "trace_digest"):
            _require_text(name, getattr(self, name))
        if self.attempts < 1:
            raise ValueError("infeasible construction evidence requires an attempt")
        if not self.missing_requirements:
            raise ValueError("infeasible construction evidence requires missing requirements")
        _require_unique("missing construction requirements", self.missing_requirements)

    @property
    def digest(self) -> str:
        return stable_digest("osc-infeasible-construction", self)


@dataclass(frozen=True, slots=True)
class ResourceTokens:
    cpu_tokens: int
    rss_bytes: int
    io_class: str
    backend_internal_threads: int
    exclusive_state: str = ""
    schema_version: str = RESOURCE_TOKEN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.cpu_tokens < 1 or self.rss_bytes < 0 or self.backend_internal_threads < 0:
            raise ValueError("resource tokens are outside their valid range")
        _require_text("io_class", self.io_class)

    @property
    def digest(self) -> str:
        return stable_digest("osc-resource-tokens", self)


@dataclass(frozen=True, slots=True)
class TaskIdentity:
    protocol_digest: str
    task_kind: TaskKind
    epoch_index: int
    decision_index: int
    seed_lineage_digest: str
    contrast_set_id: str = ""
    endpoint_id: str = ""
    backend: str = ""
    attempt: int = 0
    schema_version: str = TASK_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_text("protocol_digest", self.protocol_digest)
        _require_text("seed_lineage_digest", self.seed_lineage_digest)
        if self.epoch_index < 0 or self.decision_index < 0 or self.attempt < 0:
            raise ValueError("task identity indices must be non-negative")
        if self.task_kind == TaskKind.BACKEND_EXECUTION and not (
            self.endpoint_id and self.backend
        ):
            raise ValueError("backend execution task requires endpoint and backend")

    @property
    def task_id(self) -> str:
        return stable_digest("osc-task-id", self)

    @property
    def digest(self) -> str:
        return self.task_id


@dataclass(frozen=True, slots=True)
class TaskSpec:
    identity: TaskIdentity
    dependency_task_ids: tuple[str, ...]
    resources: ResourceTokens
    payload_digest: str
    schema_version: str = "osc-task-spec-v1"

    def __post_init__(self) -> None:
        _require_unique("task dependencies", self.dependency_task_ids)
        if self.identity.task_id in self.dependency_task_ids:
            raise ValueError("task cannot depend on itself")
        _require_text("payload_digest", self.payload_digest)

    @property
    def digest(self) -> str:
        return stable_digest("osc-task-spec", self)


@dataclass(frozen=True, slots=True)
class ResultGroup:
    result_group_id: str
    task_ids: tuple[str, ...]
    endpoint_ids: tuple[str, ...]
    contract_fingerprint: ContractFingerprint
    target_fingerprint: TargetFingerprint | None
    evidence_tier: EvidenceTier
    result_order_key: tuple[str, ...]
    schema_version: str = "osc-result-group-v1"

    def __post_init__(self) -> None:
        _require_text("result_group_id", self.result_group_id)
        if not self.task_ids or not self.endpoint_ids:
            raise ValueError("result group requires tasks and endpoints")
        _require_unique("result group task IDs", self.task_ids)
        _require_unique("result group endpoint IDs", self.endpoint_ids)
        if not self.result_order_key:
            raise ValueError("result group requires an explicit stable ordering key")

    @property
    def digest(self) -> str:
        return stable_digest("osc-result-group", self)


@dataclass(frozen=True, slots=True)
class EvidenceEnvelope:
    evidence_id: str
    state: EvidenceState
    result_group_digest: str
    task_id: str
    seed_lineage_digest: str
    contract_fingerprint: ContractFingerprint
    target_fingerprint: TargetFingerprint | None
    derivation_certificate_digest: str
    applicability_certificate_digest: str
    activation_certificate_digest: str
    observation_certificate_digest: str
    execution_outcomes: tuple[StructuredExecutionOutcome, ...]
    verdict_kind: VerdictKind
    artifact_refs: tuple[str, ...] = ()
    metadata: tuple[tuple[str, Any], ...] = ()
    schema_version: str = EVIDENCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in (
            "evidence_id",
            "result_group_digest",
            "task_id",
            "seed_lineage_digest",
            "derivation_certificate_digest",
            "applicability_certificate_digest",
            "activation_certificate_digest",
            "observation_certificate_digest",
        ):
            _require_text(name, getattr(self, name))
        if not self.execution_outcomes:
            raise ValueError("evidence envelope requires execution outcomes")
        _require_unique(
            "evidence endpoint outcomes",
            tuple(item.endpoint_id for item in self.execution_outcomes),
        )
        _require_unique("evidence artifact references", self.artifact_refs)

    @classmethod
    def build(
        cls,
        *,
        metadata: dict[str, Any] | None = None,
        **values: Any,
    ) -> "EvidenceEnvelope":
        return cls(metadata=frozen_pairs(metadata), **values)

    @property
    def digest(self) -> str:
        return stable_digest("osc-evidence-envelope", self)

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(self)


@dataclass(frozen=True, slots=True)
class LedgerEvent:
    task_id: str
    level: CoverageLevel
    event_order_key: tuple[str, ...]
    cell_ids: tuple[str, ...] = ()
    edge_ids: tuple[str, ...] = ()
    backend_pair_obligation_ids: tuple[str, ...] = ()
    activation_certificate_digest: str = ""
    applicability_certificate_digest: str = ""
    observation_certificate_digest: str = ""
    execution_outcome_digests: tuple[str, ...] = ()
    schema_version: str = LEDGER_EVENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_text("task_id", self.task_id)
        if not self.event_order_key:
            raise ValueError("ledger event requires a stable ordering key")
        for name, values in (
            ("cell IDs", self.cell_ids),
            ("edge IDs", self.edge_ids),
            ("backend-pair obligation IDs", self.backend_pair_obligation_ids),
            ("execution outcome digests", self.execution_outcome_digests),
        ):
            _require_unique(name, values)
        if self.level in {CoverageLevel.ACTIVATED, CoverageLevel.EXECUTED, CoverageLevel.OBSERVED}:
            _require_text("activation_certificate_digest", self.activation_certificate_digest)
        if self.level in {CoverageLevel.EXECUTED, CoverageLevel.OBSERVED}:
            _require_text("applicability_certificate_digest", self.applicability_certificate_digest)
            if not self.execution_outcome_digests:
                raise ValueError("executed ledger event requires execution outcomes")
        if self.level == CoverageLevel.OBSERVED:
            _require_text("observation_certificate_digest", self.observation_certificate_digest)

    @property
    def event_id(self) -> str:
        return stable_digest("osc-ledger-event-id", self)

    @property
    def digest(self) -> str:
        return self.event_id


@dataclass(frozen=True, slots=True)
class StagedComparisonRequest:
    result_group_digest: str
    contract_fingerprint: ContractFingerprint
    endpoint_ids: tuple[str, ...]
    observer_ids: tuple[str, ...]
    evidence_tier: EvidenceTier
    force_exact: bool = False
    schema_version: str = STAGED_COMPARISON_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_text("result_group_digest", self.result_group_digest)
        if not self.endpoint_ids or not self.observer_ids:
            raise ValueError("comparison request requires endpoints and observers")
        _require_unique("comparison endpoint IDs", self.endpoint_ids)
        _require_unique("comparison observer IDs", self.observer_ids)

    @property
    def exact_required(self) -> bool:
        return self.force_exact or self.evidence_tier.requires_exact

    @property
    def digest(self) -> str:
        return stable_digest("osc-staged-comparison-request", self)


@dataclass(frozen=True, slots=True)
class StagedComparisonResult:
    request_digest: str
    plan_digest: str
    observation_certificate_digest: str
    evidence_tier: EvidenceTier
    comparison_stage: str
    exact_escalated: bool
    endpoint_order: tuple[str, ...]
    component_fingerprint_digests: tuple[str, ...]
    verdict_kind: VerdictKind
    schema_version: str = STAGED_COMPARISON_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in (
            "request_digest",
            "plan_digest",
            "observation_certificate_digest",
            "comparison_stage",
        ):
            _require_text(name, getattr(self, name))
        if not self.endpoint_order:
            raise ValueError("comparison result requires endpoint ordering")
        _require_unique("comparison result endpoint order", self.endpoint_order)
        if self.evidence_tier.requires_exact and not self.exact_escalated:
            raise ValueError("non-screening comparison result must use exact authority")

    @property
    def digest(self) -> str:
        return stable_digest("osc-staged-comparison-result", self)


@dataclass(frozen=True, slots=True)
class RuntimeCacheKey:
    endpoint_digest: str
    contract_fingerprint: ContractFingerprint
    semantic_schema_version: str
    backend: str
    backend_version: str
    adapter_revision: str
    execution_mode: str
    physical_layout: str
    evidence_tier: EvidenceTier
    canonical_schema_version: str = CANONICAL_SCHEMA_VERSION
    schema_version: str = RUNTIME_CACHE_KEY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in (
            "endpoint_digest",
            "semantic_schema_version",
            "backend",
            "backend_version",
            "adapter_revision",
            "execution_mode",
            "physical_layout",
        ):
            _require_text(name, getattr(self, name))
        if self.canonical_schema_version != CANONICAL_SCHEMA_VERSION:
            raise ValueError("runtime cache key canonical schema mismatch")

    @property
    def cache_allowed(self) -> bool:
        return self.evidence_tier.cache_allowed

    @property
    def digest(self) -> str:
        return stable_digest("osc-runtime-cache-key", self)


__all__ = [
    "AtomProvenance",
    "ContractFingerprint",
    "CoverageLevel",
    "EvidenceEnvelope",
    "EvidenceState",
    "EvidenceTier",
    "ExecutionStatus",
    "FailureKind",
    "InfeasibleConstructionEvidence",
    "LedgerEvent",
    "ResourceTokens",
    "ResultGroup",
    "RuntimeCacheKey",
    "SeedLineage",
    "SeedStage",
    "SemanticAtom",
    "StagedComparisonRequest",
    "StagedComparisonResult",
    "StructuredExecutionOutcome",
    "TargetFingerprint",
    "TaskIdentity",
    "TaskKind",
    "TaskSpec",
    "VerdictKind",
]
