from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from typing import Any, Literal, Mapping

from datadiff.canonicalization import short_canonical_hash
from datadiff.method_policy import (
    EvidencePolicy,
    ExecutionPolicy,
    GenerationPolicy,
    MethodPolicy,
    PlanCollectionMode,
    PlanCollectionPolicy,
    ResourcePolicy,
    SemanticPolicy,
    UnknownContractPolicy,
)


METHOD_ARM_SCHEMA_VERSION = "latticefuzz-method-arm-v6"
METHOD_VERSION = "latticefuzz-2026.07-p4.8b-shared-cost1"
P5_METHOD_VERSION = "latticefuzz-2026.07-p5-method1"
P8_METHOD_VERSION = "latticefuzz-2026.07-p8-candidate1"
P8_WITNESS_METHOD_VERSION = "latticefuzz-2026.07-p8-witness1"
P8_WITNESS_V2_METHOD_VERSION = "latticefuzz-2026.07-p8-witness2-efficient7"
P8_WITNESS_V3_METHOD_VERSION = "latticefuzz-2026.07-p8-witness3-root1"
P8_WITNESS_V4_METHOD_VERSION = "latticefuzz-2026.07-p8-witness4-pyarrow-layout1"
P8_WITNESS_V5_METHOD_VERSION = "latticefuzz-2026.07-p8-witness5-polars-reflected1"
P8_WITNESS_V6_METHOD_VERSION = "latticefuzz-2026.07-p8-witness6-datafusion-null-topk1"
P8_WITNESS_V7_METHOD_VERSION = "latticefuzz-2026.07-p8-witness7-duckdb-join-filter1"
P8_WITNESS_GLOBAL_V1_METHOD_VERSION = (
    "latticefuzz-2026.07-p8-witness-global1-all-confirmed-roots"
)
P8_WITNESS_GLOBAL_V2_METHOD_VERSION = (
    "latticefuzz-2026.07-p8-witness-global2-family-universe1"
)
P8_WITNESS_GLOBAL_V3_METHOD_VERSION = (
    "latticefuzz-2026.07-p8-witness-global3-family-universe2-preflight2"
)
P8_WITNESS_GLOBAL_V4_METHOD_VERSION = (
    "latticefuzz-2026.07-p8-witness-global4-family-universe3-backend-scope1"
)
DEFAULT_METHOD_ARM_ID = "p8_candidate_v1"
DEFAULT_COVERAGE_METHOD_ARM_ID = "p8_semantic_witness_global_v4"

ComparisonMode = Literal["legacy", "contract"]
ObservationMode = Literal["legacy", "lossless"]
IRMode = Literal["legacy_dict", "ccs_ir"]
ObligationMode = Literal["legacy_all", "ccs_guided"]
ObligationPriorityMode = Literal[
    "complete_builder_order",
    "ccs_risk_priority",
    "ccs_risk_priority_v2",
]
ExecutionMode = Literal["cartesian", "lattice"]
SelectorMode = Literal["all", "static", "shared_cost", "plan", "random"]
ConfirmationMode = Literal["none", "full"]
CacheMode = Literal["disabled", "lattice"]
PlanGuidanceMode = Literal["disabled", "semantic", "physical"]
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
LocalizationMode = Literal["reducer_only", "prefix_adaptive"]

_SETTING_FIELDS = (
    "comparison_mode",
    "observation_mode",
    "ir_mode",
    "obligation_mode",
    "obligation_priority_mode",
    "execution_mode",
    "selector_mode",
    "plan_guidance",
    "node_budget",
    "cache_mode",
    "confirmation_mode",
    "generation_mode",
    "boundary_mode",
    "corpus_mode",
    "localization_mode",
    "unknown_contract_policy",
    "plan_collection_mode",
)
_OVERRIDABLE_FIELDS = frozenset(_SETTING_FIELDS)


@dataclass(frozen=True, slots=True)
class MethodArm:
    arm_id: str
    method_version: str
    parent_arm_id: str
    unique_factor: str
    changed_dimensions: tuple[str, ...]
    comparison_mode: ComparisonMode
    observation_mode: ObservationMode
    ir_mode: IRMode
    obligation_mode: ObligationMode
    obligation_priority_mode: ObligationPriorityMode
    execution_mode: ExecutionMode
    selector_mode: SelectorMode
    plan_guidance: PlanGuidanceMode
    node_budget: float | None
    cache_mode: CacheMode
    confirmation_mode: ConfirmationMode
    generation_mode: GenerationMode = "forward_random"
    boundary_mode: BoundaryMode = "disabled"
    corpus_mode: CorpusMode = "legacy_qd"
    localization_mode: LocalizationMode = "reducer_only"
    unknown_contract_policy: UnknownContractPolicy = "fail_closed"
    plan_collection_mode: PlanCollectionMode = "disabled"

    def __post_init__(self) -> None:
        if not self.arm_id:
            raise ValueError("method arm id must not be empty")
        if self.comparison_mode not in {"legacy", "contract"}:
            raise ValueError(f"unsupported comparison mode: {self.comparison_mode}")
        if self.observation_mode not in {"legacy", "lossless"}:
            raise ValueError(f"unsupported observation mode: {self.observation_mode}")
        if self.ir_mode not in {"legacy_dict", "ccs_ir"}:
            raise ValueError(f"unsupported IR mode: {self.ir_mode}")
        if self.obligation_mode not in {"legacy_all", "ccs_guided"}:
            raise ValueError(f"unsupported obligation mode: {self.obligation_mode}")
        if self.obligation_mode == "ccs_guided" and self.ir_mode != "ccs_ir":
            raise ValueError("CCS-guided obligations require ir_mode='ccs_ir'")
        if self.obligation_priority_mode not in {
            "complete_builder_order",
            "ccs_risk_priority",
            "ccs_risk_priority_v2",
        }:
            raise ValueError(
                "unsupported obligation priority mode: "
                f"{self.obligation_priority_mode}"
            )
        if (
            self.obligation_priority_mode in {"ccs_risk_priority", "ccs_risk_priority_v2"}
            and self.obligation_mode != "ccs_guided"
        ):
            raise ValueError("CCS risk priority requires obligation_mode='ccs_guided'")
        if self.execution_mode not in {"cartesian", "lattice"}:
            raise ValueError(f"unsupported execution mode: {self.execution_mode}")
        if self.selector_mode not in {"all", "static", "shared_cost", "plan", "random"}:
            raise ValueError(f"unsupported selector mode: {self.selector_mode}")
        if self.plan_guidance not in {"disabled", "semantic", "physical"}:
            raise ValueError(f"unsupported plan guidance mode: {self.plan_guidance}")
        if self.confirmation_mode not in {"none", "full"}:
            raise ValueError(f"unsupported confirmation mode: {self.confirmation_mode}")
        if self.cache_mode not in {"disabled", "lattice"}:
            raise ValueError(f"unsupported cache mode: {self.cache_mode}")
        if self.generation_mode not in {
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
        }:
            raise ValueError(f"unsupported generation mode: {self.generation_mode}")
        if self.boundary_mode not in {"disabled", "fault_model_targeted"}:
            raise ValueError(f"unsupported boundary mode: {self.boundary_mode}")
        if self.corpus_mode not in {"legacy_qd", "semantic_plan_qd"}:
            raise ValueError(f"unsupported corpus mode: {self.corpus_mode}")
        if self.localization_mode not in {"reducer_only", "prefix_adaptive"}:
            raise ValueError(f"unsupported localization mode: {self.localization_mode}")
        if self.unknown_contract_policy not in {"legacy_fallback", "fail_closed"}:
            raise ValueError(
                f"unsupported unknown contract policy: {self.unknown_contract_policy}"
            )
        if self.plan_collection_mode not in {"disabled", "full", "tiered"}:
            raise ValueError(
                f"unsupported plan collection mode: {self.plan_collection_mode}"
            )
        if self.plan_guidance != "physical" and self.plan_collection_mode != "disabled":
            raise ValueError("physical plan collection requires plan_guidance='physical'")
        if self.execution_mode == "cartesian":
            if self.selector_mode != "all":
                raise ValueError("cartesian arms must use selector_mode='all'")
            if self.node_budget is not None:
                raise ValueError("cartesian arms must not declare a node budget")
            if self.cache_mode != "disabled":
                raise ValueError("cartesian arms must disable lattice result caching")
        else:
            if self.selector_mode == "all":
                raise ValueError("lattice arms must declare a subgraph selector")
            if self.node_budget is None or float(self.node_budget) < 2.0:
                raise ValueError("lattice arms require a node budget of at least two")
        unknown_changes = set(self.changed_dimensions) - _OVERRIDABLE_FIELDS
        if unknown_changes:
            raise ValueError(
                "method arm declares unknown changed dimensions: "
                + ", ".join(sorted(unknown_changes))
            )

    def settings(self) -> dict[str, Any]:
        payload = asdict(self)
        return {key: payload[key] for key in _SETTING_FIELDS}

    @property
    def policy(self) -> MethodPolicy:
        return MethodPolicy(
            semantic=SemanticPolicy(
                comparison_mode=self.comparison_mode,
                observation_mode=self.observation_mode,
                ir_mode=self.ir_mode,
                obligation_mode=self.obligation_mode,
                obligation_priority_mode=self.obligation_priority_mode,
                unknown_contract=self.unknown_contract_policy,
            ),
            generation=GenerationPolicy(
                mode=self.generation_mode,
                boundary_mode=self.boundary_mode,
                corpus_mode=self.corpus_mode,
            ),
            execution=ExecutionPolicy(
                mode=self.execution_mode,
                selector=self.selector_mode,
                node_budget=self.node_budget,
                cache_mode=self.cache_mode,
                confirmation_mode=self.confirmation_mode,
                plan_guidance=self.plan_guidance,
                plan_collection=PlanCollectionPolicy(
                    "full"
                    if self.plan_guidance == "physical"
                    and self.plan_collection_mode == "disabled"
                    else self.plan_collection_mode
                ),
            ),
            evidence=EvidencePolicy(localization_mode=self.localization_mode),
            resource=ResourcePolicy(),
        )

    def policies(self) -> dict[str, Any]:
        return self.policy.to_dict()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["changed_dimensions"] = list(self.changed_dimensions)
        return payload


@dataclass(frozen=True, slots=True)
class MethodArmResolution:
    base_arm: MethodArm
    effective_arm: MethodArm
    overrides: tuple[tuple[str, Any], ...] = ()

    @property
    def registered(self) -> bool:
        return not self.overrides

    @property
    def effective_arm_id(self) -> str:
        if self.registered:
            return self.base_arm.arm_id
        suffix = short_canonical_hash(dict(self.overrides), 10)
        return f"{self.base_arm.arm_id}+override-{suffix}"

    def manifest(self) -> dict[str, Any]:
        payload = {
            "schema_version": METHOD_ARM_SCHEMA_VERSION,
            "arm_id": self.effective_arm_id,
            "base_arm_id": self.base_arm.arm_id,
            "registered": self.registered,
            "method_version": self.effective_arm.method_version,
            "parent_arm_id": self.base_arm.parent_arm_id,
            "unique_factor": self.base_arm.unique_factor,
            "changed_dimensions": list(self.base_arm.changed_dimensions),
            "settings": self.effective_arm.settings(),
            "policies": self.effective_arm.policies(),
            "overrides": {key: value for key, value in self.overrides},
        }
        payload["digest"] = f"method-{short_canonical_hash(payload, 64)}"
        return payload


_P4_EFFICIENT_CONTROL = MethodArm(
    arm_id="contract_lattice_shared_cost_full",
    method_version=METHOD_VERSION,
    parent_arm_id="contract_lattice_shared_cost",
    unique_factor="candidate_triggered_full_backend_confirmation",
    changed_dimensions=("confirmation_mode",),
    comparison_mode="contract",
    observation_mode="lossless",
    ir_mode="ccs_ir",
    obligation_mode="ccs_guided",
    obligation_priority_mode="complete_builder_order",
    execution_mode="lattice",
    selector_mode="shared_cost",
    plan_guidance="disabled",
    node_budget=3.0,
    cache_mode="lattice",
    confirmation_mode="full",
)
_P5_PROMOTED_METHOD = MethodArm(
    arm_id="p5_promoted_method",
    method_version=P5_METHOD_VERSION,
    parent_arm_id="p5_physical_goal_boundary_localize",
    unique_factor="semantic_plan_quality_diversity_archive",
    changed_dimensions=("corpus_mode",),
    comparison_mode="contract",
    observation_mode="lossless",
    ir_mode="ccs_ir",
    obligation_mode="ccs_guided",
    obligation_priority_mode="complete_builder_order",
    execution_mode="lattice",
    selector_mode="shared_cost",
    plan_guidance="physical",
    node_budget=3.0,
    cache_mode="lattice",
    confirmation_mode="full",
    generation_mode="goal_first",
    boundary_mode="fault_model_targeted",
    corpus_mode="semantic_plan_qd",
    localization_mode="prefix_adaptive",
)
_P8_CANDIDATE = replace(
    _P5_PROMOTED_METHOD,
    arm_id="p8_candidate_v1",
    method_version=P8_METHOD_VERSION,
    parent_arm_id="p5_promoted_method",
    unique_factor="tiered_finding_triggered_physical_plan_evidence",
    changed_dimensions=("plan_collection_mode",),
    plan_collection_mode="tiered",
)
_P8_SEMANTIC_WITNESS = replace(
    _P8_CANDIDATE,
    arm_id="p8_semantic_witness_v1",
    method_version=P8_WITNESS_METHOD_VERSION,
    parent_arm_id="p8_candidate_v1",
    unique_factor="deterministic_semantic_activation_witness_enforcement",
    changed_dimensions=("generation_mode",),
    generation_mode="goal_first_witness",
)
_P8_SEMANTIC_WITNESS_V2 = replace(
    _P8_CANDIDATE,
    arm_id="p8_semantic_witness_v2",
    method_version=P8_WITNESS_V2_METHOD_VERSION,
    parent_arm_id="p8_candidate_v1",
    unique_factor="diversity_preserving_deterministic_semantic_witness_generation",
    changed_dimensions=("generation_mode",),
    generation_mode="goal_first_witness_v2",
)
_P8_SEMANTIC_WITNESS_V3 = replace(
    _P8_CANDIDATE,
    arm_id="p8_semantic_witness_v3",
    method_version=P8_WITNESS_V3_METHOD_VERSION,
    parent_arm_id="p8_candidate_v1",
    unique_factor="root_guided_cross_interaction_semantic_witness_generation",
    changed_dimensions=("generation_mode",),
    generation_mode="goal_first_witness_v3",
)
_P8_SEMANTIC_WITNESS_V4 = replace(
    _P8_CANDIDATE,
    arm_id="p8_semantic_witness_v4",
    method_version=P8_WITNESS_V4_METHOD_VERSION,
    parent_arm_id="p8_candidate_v1",
    unique_factor="pyarrow_physical_layout_boolean_groupby_witness_generation",
    changed_dimensions=("generation_mode",),
    generation_mode="goal_first_witness_v4",
)
_P8_SEMANTIC_WITNESS_V5 = replace(
    _P8_CANDIDATE,
    arm_id="p8_semantic_witness_v5",
    method_version=P8_WITNESS_V5_METHOD_VERSION,
    parent_arm_id="p8_candidate_v1",
    unique_factor="polars_reflected_arithmetic_operand_order_witness_generation",
    changed_dimensions=("generation_mode",),
    generation_mode="goal_first_witness_v5",
)
_P8_SEMANTIC_WITNESS_V6 = replace(
    _P8_CANDIDATE,
    arm_id="p8_semantic_witness_v6",
    method_version=P8_WITNESS_V6_METHOD_VERSION,
    parent_arm_id="p8_candidate_v1",
    unique_factor="datafusion_grouped_null_topk_witness_generation",
    changed_dimensions=("generation_mode",),
    generation_mode="goal_first_witness_v6",
)
_P8_SEMANTIC_WITNESS_V7 = replace(
    _P8_CANDIDATE,
    arm_id="p8_semantic_witness_v7",
    method_version=P8_WITNESS_V7_METHOD_VERSION,
    parent_arm_id="p8_candidate_v1",
    unique_factor="duckdb_join_filter_pushdown_limit_witness_generation",
    changed_dimensions=("generation_mode",),
    generation_mode="goal_first_witness_v7",
)
_P8_SEMANTIC_WITNESS_GLOBAL_V1 = replace(
    _P8_CANDIDATE,
    arm_id="p8_semantic_witness_global_v1",
    method_version=P8_WITNESS_GLOBAL_V1_METHOD_VERSION,
    parent_arm_id="p8_candidate_v1",
    unique_factor="all_confirmed_root_family_witness_portfolio_generation",
    changed_dimensions=("generation_mode",),
    generation_mode="goal_first_witness_global_v1",
)
_P8_SEMANTIC_WITNESS_GLOBAL_V2 = replace(
    _P8_CANDIDATE,
    arm_id="p8_semantic_witness_global_v2",
    method_version=P8_WITNESS_GLOBAL_V2_METHOD_VERSION,
    parent_arm_id="p8_candidate_v1",
    unique_factor="backend_semantic_family_universe_witness_portfolio_generation",
    changed_dimensions=("generation_mode",),
    generation_mode="goal_first_witness_global_v2",
)
_P8_SEMANTIC_WITNESS_GLOBAL_V3 = replace(
    _P8_CANDIDATE,
    arm_id="p8_semantic_witness_global_v3",
    method_version=P8_WITNESS_GLOBAL_V3_METHOD_VERSION,
    parent_arm_id="p8_candidate_v1",
    unique_factor="expanded_backend_semantic_family_universe_witness_generation",
    changed_dimensions=("generation_mode",),
    generation_mode="goal_first_witness_global_v3",
)
_P8_SEMANTIC_WITNESS_GLOBAL_V4 = replace(
    _P8_CANDIDATE,
    arm_id="p8_semantic_witness_global_v4",
    method_version=P8_WITNESS_GLOBAL_V4_METHOD_VERSION,
    parent_arm_id="p8_candidate_v1",
    unique_factor="issue_risk_backend_scoped_semantic_family_universe_generation",
    changed_dimensions=("generation_mode",),
    generation_mode="goal_first_witness_global_v4",
)

LIVE_METHOD_ARM_IDS: tuple[str, ...] = (
    "p8_candidate_v1",
    "p8_semantic_witness_v1",
    "p8_semantic_witness_v2",
    "p8_semantic_witness_v3",
    "p8_semantic_witness_v4",
    "p8_semantic_witness_v5",
    "p8_semantic_witness_v6",
    "p8_semantic_witness_v7",
    "p8_semantic_witness_global_v1",
    "p8_semantic_witness_global_v2",
    "p8_semantic_witness_global_v3",
    "p8_semantic_witness_global_v4",
    "contract_lattice_shared_cost_full",
    "p5_promoted_method",
)
RESEARCH_CONTROL_ARM_IDS: tuple[str, ...] = (
    "legacy_cartesian",
    "contract_cartesian",
    "contract_ccs_cartesian",
    "contract_ccs_obligations_cartesian",
    "contract_ccs_risk_obligations_cartesian",
    "contract_lattice_static",
    "contract_lattice_plan",
    "contract_lattice_shared_cost",
    "contract_lattice_full",
    "p4_lattice_static_cache_off",
    "p4_lattice_random_cache_off",
    "p4_lattice_shared_cost_cache_off",
    "p4_lattice_plan_unguided_cache_off",
    "p4_lattice_plan_cache_off",
    "p5_physical_plan",
    "p5_goal_first",
    "p5_boundary_targeted",
    "p5_priority_v2",
    "p5_prefix_localization",
    "p5_quality_diversity",
    "p5_physical_goal_first",
    "p5_physical_goal_boundary",
    "p5_physical_goal_boundary_localize",
    "p5_physical_goal_boundary_priority",
    "p5_physical_goal_boundary_priority_localize",
    "p5_full_method",
)
REGISTERED_METHOD_ARM_IDS: tuple[str, ...] = (
    "legacy_cartesian",
    "contract_cartesian",
    "contract_ccs_cartesian",
    "contract_ccs_obligations_cartesian",
    "contract_ccs_risk_obligations_cartesian",
    "contract_lattice_static",
    "contract_lattice_plan",
    "contract_lattice_shared_cost",
    "contract_lattice_shared_cost_full",
    "contract_lattice_full",
    "p4_lattice_static_cache_off",
    "p4_lattice_random_cache_off",
    "p4_lattice_shared_cost_cache_off",
    "p4_lattice_plan_unguided_cache_off",
    "p4_lattice_plan_cache_off",
    "p5_physical_plan",
    "p5_goal_first",
    "p5_boundary_targeted",
    "p5_priority_v2",
    "p5_prefix_localization",
    "p5_quality_diversity",
    "p5_physical_goal_first",
    "p5_physical_goal_boundary",
    "p5_physical_goal_boundary_localize",
    "p5_promoted_method",
    "p5_physical_goal_boundary_priority",
    "p5_physical_goal_boundary_priority_localize",
    "p5_full_method",
    "p8_candidate_v1",
    "p8_semantic_witness_v1",
    "p8_semantic_witness_v2",
    "p8_semantic_witness_v3",
    "p8_semantic_witness_v4",
    "p8_semantic_witness_v5",
    "p8_semantic_witness_v6",
    "p8_semantic_witness_v7",
    "p8_semantic_witness_global_v1",
    "p8_semantic_witness_global_v2",
    "p8_semantic_witness_global_v3",
    "p8_semantic_witness_global_v4",
)

_LIVE_METHOD_ARMS = {
    arm.arm_id: arm
    for arm in (
        _P8_CANDIDATE,
        _P8_SEMANTIC_WITNESS,
        _P8_SEMANTIC_WITNESS_V2,
        _P8_SEMANTIC_WITNESS_V3,
        _P8_SEMANTIC_WITNESS_V4,
        _P8_SEMANTIC_WITNESS_V5,
        _P8_SEMANTIC_WITNESS_V6,
        _P8_SEMANTIC_WITNESS_V7,
        _P8_SEMANTIC_WITNESS_GLOBAL_V1,
        _P8_SEMANTIC_WITNESS_GLOBAL_V2,
        _P8_SEMANTIC_WITNESS_GLOBAL_V3,
        _P8_SEMANTIC_WITNESS_GLOBAL_V4,
        _P4_EFFICIENT_CONTROL,
        _P5_PROMOTED_METHOD,
    )
}
_RESEARCH_METHOD_ARMS: dict[str, MethodArm] | None = None


def method_arm_ids() -> tuple[str, ...]:
    return REGISTERED_METHOD_ARM_IDS


def live_method_arm_ids() -> tuple[str, ...]:
    return LIVE_METHOD_ARM_IDS


def research_control_arm_ids() -> tuple[str, ...]:
    return RESEARCH_CONTROL_ARM_IDS


def registered_method_arms() -> tuple[MethodArm, ...]:
    return tuple(method_arm(arm_id) for arm_id in REGISTERED_METHOD_ARM_IDS)


def method_arm(arm_id: str) -> MethodArm:
    resolved_id = str(arm_id or DEFAULT_METHOD_ARM_ID).strip()
    live = _LIVE_METHOD_ARMS.get(resolved_id)
    if live is not None:
        return live
    control = _research_method_arms().get(resolved_id)
    if control is not None:
        return control
    allowed = ", ".join(REGISTERED_METHOD_ARM_IDS)
    raise ValueError(f"unknown method arm '{resolved_id}'; expected one of: {allowed}")


def resolve_method_arm(
    arm_id: str,
    overrides: Mapping[str, Any] | None = None,
) -> MethodArmResolution:
    base = method_arm(arm_id)
    normalized_overrides = _normalize_overrides(overrides)
    effective = replace(base, **normalized_overrides) if normalized_overrides else base
    return MethodArmResolution(
        base_arm=base,
        effective_arm=effective,
        overrides=tuple(sorted(normalized_overrides.items())),
    )


def arm_difference(left: MethodArm, right: MethodArm) -> dict[str, tuple[Any, Any]]:
    left_settings = left.settings()
    right_settings = right.settings()
    return {
        key: (left_settings[key], right_settings[key])
        for key in _SETTING_FIELDS
        if left_settings[key] != right_settings[key]
    }


def validate_method_arm_registry() -> None:
    dataclass_fields = {item.name for item in fields(MethodArm)}
    if not _OVERRIDABLE_FIELDS <= dataclass_fields:
        raise AssertionError("overridable method dimensions are not MethodArm fields")
    for arm in registered_method_arms():
        if not arm.parent_arm_id:
            continue
        parent = method_arm(arm.parent_arm_id)
        actual = set(arm_difference(parent, arm))
        declared = set(arm.changed_dimensions)
        if actual != declared:
            raise AssertionError(
                f"arm {arm.arm_id} changes {sorted(actual)} but declares {sorted(declared)}"
            )


def _research_method_arms() -> dict[str, MethodArm]:
    global _RESEARCH_METHOD_ARMS
    if _RESEARCH_METHOD_ARMS is None:
        from datadiff.research_controls.method_arms import build_research_control_arms

        _RESEARCH_METHOD_ARMS = build_research_control_arms(_LIVE_METHOD_ARMS)
    return _RESEARCH_METHOD_ARMS


def _normalize_overrides(overrides: Mapping[str, Any] | None) -> dict[str, Any]:
    if overrides is None:
        return {}
    if not isinstance(overrides, Mapping):
        raise TypeError("method arm overrides must be a mapping")
    unknown = {str(key) for key in overrides} - _OVERRIDABLE_FIELDS
    if unknown:
        raise ValueError(
            "unsupported method arm override fields: " + ", ".join(sorted(unknown))
        )
    normalized = {str(key): value for key, value in overrides.items()}
    for key in (
        "plan_guidance",
        "generation_mode",
        "boundary_mode",
        "corpus_mode",
        "localization_mode",
        "obligation_priority_mode",
    ):
        if key in normalized:
            normalized[key] = str(normalized[key])
    if "node_budget" in normalized and normalized["node_budget"] is not None:
        normalized["node_budget"] = float(normalized["node_budget"])
    return normalized


if arm_difference(_P5_PROMOTED_METHOD, _P8_CANDIDATE) != {
    "plan_collection_mode": ("disabled", "tiered")
}:
    raise AssertionError("live P8 candidate must differ from frozen P5 by one dimension")
if arm_difference(_P8_CANDIDATE, _P8_SEMANTIC_WITNESS) != {
    "generation_mode": ("goal_first", "goal_first_witness")
}:
    raise AssertionError("semantic-witness arm must differ from P8 by one dimension")
if arm_difference(_P8_CANDIDATE, _P8_SEMANTIC_WITNESS_V2) != {
    "generation_mode": ("goal_first", "goal_first_witness_v2")
}:
    raise AssertionError("semantic-witness v2 arm must differ from P8 by one dimension")
if arm_difference(_P8_CANDIDATE, _P8_SEMANTIC_WITNESS_V3) != {
    "generation_mode": ("goal_first", "goal_first_witness_v3")
}:
    raise AssertionError("semantic-witness v3 arm must differ from P8 by one dimension")
if arm_difference(_P8_CANDIDATE, _P8_SEMANTIC_WITNESS_V4) != {
    "generation_mode": ("goal_first", "goal_first_witness_v4")
}:
    raise AssertionError("semantic-witness v4 arm must differ from P8 by one dimension")
if arm_difference(_P8_CANDIDATE, _P8_SEMANTIC_WITNESS_V5) != {
    "generation_mode": ("goal_first", "goal_first_witness_v5")
}:
    raise AssertionError("semantic-witness v5 arm must differ from P8 by one dimension")
if arm_difference(_P8_CANDIDATE, _P8_SEMANTIC_WITNESS_V6) != {
    "generation_mode": ("goal_first", "goal_first_witness_v6")
}:
    raise AssertionError("semantic-witness v6 arm must differ from P8 by one dimension")
if arm_difference(_P8_CANDIDATE, _P8_SEMANTIC_WITNESS_V7) != {
    "generation_mode": ("goal_first", "goal_first_witness_v7")
}:
    raise AssertionError("semantic-witness v7 arm must differ from P8 by one dimension")
if arm_difference(_P8_CANDIDATE, _P8_SEMANTIC_WITNESS_GLOBAL_V1) != {
    "generation_mode": ("goal_first", "goal_first_witness_global_v1")
}:
    raise AssertionError(
        "global semantic-witness arm must differ from P8 by one dimension"
    )
if arm_difference(_P8_CANDIDATE, _P8_SEMANTIC_WITNESS_GLOBAL_V2) != {
    "generation_mode": ("goal_first", "goal_first_witness_global_v2")
}:
    raise AssertionError(
        "global-v2 semantic-witness arm must differ from P8 by one dimension"
    )
if arm_difference(_P8_CANDIDATE, _P8_SEMANTIC_WITNESS_GLOBAL_V3) != {
    "generation_mode": ("goal_first", "goal_first_witness_global_v3")
}:
    raise AssertionError(
        "global-v3 semantic-witness arm must differ from P8 by one dimension"
    )
if arm_difference(_P8_CANDIDATE, _P8_SEMANTIC_WITNESS_GLOBAL_V4) != {
    "generation_mode": ("goal_first", "goal_first_witness_global_v4")
}:
    raise AssertionError(
        "global-v4 semantic-witness arm must differ from P8 by one dimension"
    )
