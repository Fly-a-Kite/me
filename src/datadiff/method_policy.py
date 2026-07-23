from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal


ComparisonMode = Literal["legacy", "contract"]
ObservationMode = Literal["legacy", "lossless"]
IRMode = Literal["legacy_dict", "ccs_ir"]
ObligationMode = Literal["legacy_all", "ccs_guided"]
ObligationPriorityMode = Literal[
    "complete_builder_order",
    "ccs_risk_priority",
    "ccs_risk_priority_v2",
]
UnknownContractPolicy = Literal["legacy_fallback", "fail_closed"]
GenerationMode = Literal[
    "forward_random",
    "goal_first",
    "goal_first_witness",
    "goal_first_witness_v2",
    "goal_first_witness_v3",
    "goal_first_witness_v4",
    "goal_first_witness_v5",
    "goal_first_witness_v6",
    "goal_first_witness_v7",
    "goal_first_witness_global_v1",
    "goal_first_witness_global_v2",
    "goal_first_witness_global_v3",
    "goal_first_witness_global_v4",
]
BoundaryMode = Literal["disabled", "fault_model_targeted"]
CorpusMode = Literal["legacy_qd", "semantic_plan_qd"]
ExecutionMode = Literal["cartesian", "lattice"]
SelectorMode = Literal["all", "static", "shared_cost", "plan", "random"]
CacheMode = Literal["disabled", "lattice"]
ConfirmationMode = Literal["none", "full"]
PlanGuidanceMode = Literal["disabled", "semantic", "physical"]
PlanCollectionMode = Literal["disabled", "full", "tiered"]
PlanDetail = Literal["disabled", "fingerprint", "full"]
EvidenceTier = Literal[
    "screening",
    "finding",
    "fresh_confirmation",
    "native_reproduction",
]
LocalizationMode = Literal["reducer_only", "prefix_adaptive"]
BackendParallelism = Literal["sequential", "parallel_control"]
LogDetail = Literal["minimal", "compact", "full"]


@dataclass(frozen=True, slots=True)
class SemanticPolicy:
    comparison_mode: ComparisonMode
    observation_mode: ObservationMode
    ir_mode: IRMode
    obligation_mode: ObligationMode
    obligation_priority_mode: ObligationPriorityMode
    unknown_contract: UnknownContractPolicy = "fail_closed"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class GenerationPolicy:
    mode: GenerationMode
    boundary_mode: BoundaryMode
    corpus_mode: CorpusMode

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PlanCollectionPolicy:
    mode: PlanCollectionMode

    def detail_for(self, evidence_tier: EvidenceTier) -> PlanDetail:
        if self.mode == "disabled":
            return "disabled"
        if self.mode == "full":
            return "full"
        if evidence_tier == "screening":
            return "fingerprint"
        return "full"

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "screening_detail": self.detail_for("screening"),
            "finding_detail": self.detail_for("finding"),
            "fresh_confirmation_detail": self.detail_for("fresh_confirmation"),
            "native_reproduction_detail": self.detail_for("native_reproduction"),
        }


@dataclass(frozen=True, slots=True)
class ExecutionPolicy:
    mode: ExecutionMode
    selector: SelectorMode
    node_budget: float | None
    cache_mode: CacheMode
    confirmation_mode: ConfirmationMode
    plan_guidance: PlanGuidanceMode
    plan_collection: PlanCollectionPolicy

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["plan_collection"] = self.plan_collection.to_dict()
        return payload


@dataclass(frozen=True, slots=True)
class EvidencePolicy:
    localization_mode: LocalizationMode
    localization_trigger: Literal["never", "finding"] = "finding"
    full_plan_trigger: Literal["never", "finding"] = "finding"
    candidate_root_separation: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ResourcePolicy:
    backend_parallelism: BackendParallelism = "sequential"
    backend_session_reuse: bool = True
    default_log_detail: LogDetail = "compact"
    worker_rss_limit_mib: int = 2048
    fresh_evidence_cache_allowed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class MethodPolicy:
    semantic: SemanticPolicy
    generation: GenerationPolicy
    execution: ExecutionPolicy
    evidence: EvidencePolicy
    resource: ResourcePolicy

    def to_dict(self) -> dict[str, Any]:
        return {
            "semantic": self.semantic.to_dict(),
            "generation": self.generation.to_dict(),
            "execution": self.execution.to_dict(),
            "evidence": self.evidence.to_dict(),
            "resource": self.resource.to_dict(),
        }
