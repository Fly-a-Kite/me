from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from datadiff.coverage_debt import CoverageDebtLedger, CoverageDebtPolicy
from datadiff.risk_limiting_audit import (
    AUDIT_AXES,
    AuditAxisPolicy,
    DeterministicSentinelAuditor,
    _validated_sha256,
)
from datadiff.research_controls.rlcmf_shadow.rlcmf_voi import VOI_FEATURES, VOIRaiseThreshold


RLCMF_MANIFEST_SCHEMA_VERSION = "rlcmf-manifest-v1"
RLCMF_PRIMARY_METRIC = "independently_confirmed_unique_real_roots_per_cpu_hour"
RLCMF_TOP_LEVEL_FIELDS = {
    "schema_version",
    "manifest_id",
    "outcome_blind",
    "seed_plan",
    "implementation",
    "fidelity",
    "audit",
    "coverage_debt",
    "estimators",
    "safety",
    "voi",
    "budget",
    "promotion",
}
RLCMF_REQUIRED_ROLLBACK_TRIGGERS = {
    "non_reproduced_candidate",
    "miss_rate_bound",
    "recall_bound",
    "misclassification_bound",
    "coverage_debt_cap",
    "manifest_integrity",
    "propensity_integrity",
    "ledger_integrity",
    "model_integrity",
}
RLCMF_REQUIRED_SAFETY_AXIS_THRESHOLDS = {
    "maximum_event_miss_rate",
    "minimum_event_recall",
    "maximum_misclassification_rate",
    "caution_margin",
    "maximum_uncaptured_full_event_rate",
}
RLCMF_EVENT_METRICS = {
    "any_new_candidate_family",
    "any_new_candidate_root",
    "any_new_confirmed_family",
    "any_new_confirmed_root",
}
RLCMF_REQUIRED_PROMOTION_THRESHOLDS = {
    "minimum_exact_family_recall",
    "minimum_root_recall",
    "maximum_non_reproduced_candidate_cases",
    "maximum_misclassification_rate",
    "minimum_primary_metric_ratio",
}


def canonical_manifest_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")


def manifest_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_manifest_bytes(payload)).hexdigest()


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return value


def _positive_int(value: Any, *, field: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{field} must be positive")
    return parsed


@dataclass(frozen=True, slots=True)
class ValidatedRLCMFManifest:
    payload: dict[str, Any]
    sha256: str
    audit_policies: tuple[AuditAxisPolicy, ...]
    debt_policies: tuple[CoverageDebtPolicy, ...]

    def build_auditor(self) -> DeterministicSentinelAuditor:
        audit = _mapping(self.payload["audit"], field="audit")
        return DeterministicSentinelAuditor(
            manifest_sha256=self.sha256,
            salt=str(audit["salt"]),
            policies=self.audit_policies,
            serialization=str(audit["serialization"]),
        )

    def build_debt_ledger(self) -> CoverageDebtLedger:
        return CoverageDebtLedger(
            manifest_sha256=self.sha256,
            policies=self.debt_policies,
        )

    def build_event_accumulator(
        self,
        *,
        axis: str,
        metric: str | None = None,
        estimator: str | None = None,
    ):
        from datadiff.research_controls.rlcmf_shadow.rlcmf_estimation import RiskLimitingEventAccumulator

        if axis not in {policy.axis for policy in self.audit_policies}:
            raise ValueError(f"axis is not registered for RLCMF audit: {axis}")
        estimators = _mapping(self.payload["estimators"], field="estimators")
        alpha_allocation = _mapping(
            estimators["alpha_allocation"],
            field="estimators.alpha_allocation",
        )
        event_metrics = _mapping(
            estimators["event_metrics"],
            field="estimators.event_metrics",
        )
        registered_metric = str(event_metrics[axis])
        if metric is not None and str(metric) != registered_metric:
            raise ValueError(
                f"event metric for {axis} differs from the frozen manifest"
            )
        from datadiff.research_controls.rlcmf_shadow.rlcmf_safety_evidence import RLCMF_AXIS_ALPHA_COMPONENTS

        return RiskLimitingEventAccumulator(
            manifest_sha256=self.sha256,
            axis=axis,
            metric=registered_metric,
            estimator=str(estimator or estimators["event"]),
            event_bound=float(estimators["event_bound"]),
            weight_clip=float(estimators["weight_clip"]),
            alpha=float(alpha_allocation[axis]) / RLCMF_AXIS_ALPHA_COMPONENTS,
            minimum_effective_audits=float(
                estimators["minimum_effective_audits"]
            ),
            confidence_sequence=str(estimators["confidence_sequence"]),
        )

    def build_sentinel_coordinator(self):
        from datadiff.research_controls.rlcmf_shadow.rlcmf_sentinel import RLCMFSentinelCoordinator
        from datadiff.research_controls.rlcmf_shadow.rlcmf_safety_evidence import (
            CounterfactualAxisSafetyTracker,
        )

        estimators = _mapping(self.payload["estimators"], field="estimators")
        alpha_allocation = _mapping(
            estimators["alpha_allocation"],
            field="estimators.alpha_allocation",
        )
        accumulators = {
            policy.axis: self.build_event_accumulator(axis=policy.axis)
            for policy in self.audit_policies
        }

        return RLCMFSentinelCoordinator(
            auditor=self.build_auditor(),
            debt_ledger=self.build_debt_ledger(),
            event_accumulators=accumulators,
            safety_trackers={
                axis: CounterfactualAxisSafetyTracker(
                    miss_accumulator=accumulator,
                    axis_alpha=float(alpha_allocation[axis]),
                )
                for axis, accumulator in accumulators.items()
            },
        )

    def build_safety_controller(self):
        from datadiff.research_controls.rlcmf_shadow.rlcmf_safety import RiskLimitingSafetyController

        safety = _mapping(self.payload["safety"], field="safety")
        estimators = _mapping(self.payload["estimators"], field="estimators")
        budget = _mapping(self.payload["budget"], field="budget")
        promotion = _mapping(self.payload["promotion"], field="promotion")
        thresholds = _mapping(
            promotion["thresholds"],
            field="promotion.thresholds",
        )
        return RiskLimitingSafetyController.from_manifest_contract(
            manifest_sha256=self.sha256,
            seeds=tuple(int(seed) for seed in self.payload["seed_plan"]["seeds"]),
            audit_policies=self.audit_policies,
            calibration_cases_per_seed=int(
                budget["equal_calibration_cases_per_seed"]
            ),
            minimum_effective_audits=float(
                estimators["minimum_effective_audits"]
            ),
            decision_epoch_cases=int(safety["decision_epoch_cases"]),
            propensity_ladder=tuple(safety["propensity_ladder"]),
            axis_thresholds=_mapping(
                safety["axis_thresholds"],
                field="safety.axis_thresholds",
            ),
            maximum_non_reproduced_candidate_cases=int(
                thresholds["maximum_non_reproduced_candidate_cases"]
            ),
            rollback_triggers=tuple(str(value) for value in safety["rollback_triggers"]),
            recovery_enabled=bool(safety["recovery"]["enabled"]),
            horizon_payload=(
                _mapping(safety["horizon"], field="safety.horizon")
                if "horizon" in safety
                else None
            ),
            total_horizon_cases=(
                len(self.payload["seed_plan"]["seeds"])
                * int(budget["max_cases_per_seed"])
            ),
        )

    def build_voi_scheduler(self):
        from datadiff.research_controls.rlcmf_shadow.rlcmf_voi import RootCauseVOIScheduler

        safety = _mapping(self.payload["safety"], field="safety")
        voi = _mapping(self.payload["voi"], field="voi")
        return RootCauseVOIScheduler.from_manifest_contract(
            manifest_sha256=self.sha256,
            audit_policies=self.audit_policies,
            propensity_ladder=tuple(safety["propensity_ladder"]),
            payload=voi,
        )


def validate_rlcmf_manifest(payload: Mapping[str, Any]) -> ValidatedRLCMFManifest:
    manifest = dict(payload)
    unknown_fields = sorted(set(manifest) - RLCMF_TOP_LEVEL_FIELDS)
    if unknown_fields:
        raise ValueError(f"unknown RLCMF manifest fields: {unknown_fields}")
    if manifest.get("schema_version") != RLCMF_MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported RLCMF manifest schema")
    if manifest.get("outcome_blind") is not True:
        raise ValueError("RLCMF manifest must be outcome_blind")
    if not str(manifest.get("manifest_id", "")).strip():
        raise ValueError("manifest_id cannot be empty")

    seed_plan = _mapping(manifest.get("seed_plan"), field="seed_plan")
    if not str(seed_plan.get("algorithm", "")).strip() or not str(
        seed_plan.get("plan_id", "")
    ).strip():
        raise ValueError("seed_plan algorithm and plan_id are required")
    seeds = seed_plan.get("seeds")
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("seed_plan.seeds must be a non-empty list")
    normalized_seeds = [int(seed) for seed in seeds]
    if any(seed < 0 for seed in normalized_seeds):
        raise ValueError("seed_plan seeds must be non-negative")
    if len(set(normalized_seeds)) != len(normalized_seeds):
        raise ValueError("seed_plan seeds must be unique")
    raw_seed_derivation = seed_plan.get("derivation")
    if raw_seed_derivation is not None:
        seed_derivation = _mapping(
            raw_seed_derivation,
            field="seed_plan.derivation",
        )
        expected_seed_derivation_fields = {
            "serialization",
            "digest",
            "minimum",
            "maximum",
            "count",
            "collision_rule",
            "schedule",
        }
        if set(seed_derivation) != expected_seed_derivation_fields:
            raise ValueError("seed derivation contract is incomplete")
        if seed_derivation["serialization"] != "canonical-json-sort-keys-v1":
            raise ValueError("unsupported seed derivation serialization")
        if seed_derivation["digest"] != "sha256-first64-big-endian-v1":
            raise ValueError("unsupported seed derivation digest")
        if seed_derivation["collision_rule"] != (
            "increment-counter-skip-duplicate-v1"
        ):
            raise ValueError("unsupported seed collision rule")
        if seed_derivation["schedule"] != "round-robin-in-trace-order-v1":
            raise ValueError("unsupported seed scheduling rule")
        from datadiff.research_controls.rlcmf_shadow.rlcmf_replay import derive_outcome_blind_seeds

        reproduced_seeds = derive_outcome_blind_seeds(
            plan_id=str(seed_plan["plan_id"]),
            count=int(seed_derivation["count"]),
            minimum=int(seed_derivation["minimum"]),
            maximum=int(seed_derivation["maximum"]),
        )
        if reproduced_seeds != normalized_seeds:
            raise ValueError("frozen seed list differs from its derivation contract")
    schedule_counts: dict[int, int] | None = None
    raw_schedule = seed_plan.get("schedule")
    if raw_schedule is not None:
        if not isinstance(raw_schedule, list) or not raw_schedule:
            raise ValueError("seed_plan.schedule must be a non-empty list")
        schedule_counts = {seed: 0 for seed in normalized_seeds}
        expected_schedule_fields = {
            "trace_index",
            "campaign_seed",
            "case_sha256",
            "metamorphic_relation_order",
        }
        for index, raw_entry in enumerate(raw_schedule):
            entry = _mapping(raw_entry, field=f"seed_plan.schedule[{index}]")
            if set(entry) != expected_schedule_fields:
                raise ValueError(
                    "seed schedule entries must contain exactly trace_index, "
                    "campaign_seed, case_sha256, and metamorphic_relation_order"
                )
            if int(entry["trace_index"]) != index:
                raise ValueError("seed_plan.schedule trace indexes must be contiguous")
            campaign_seed = int(entry["campaign_seed"])
            if campaign_seed not in schedule_counts:
                raise ValueError("seed schedule references an unregistered campaign seed")
            _validated_sha256(
                str(entry["case_sha256"]),
                field=f"seed_plan.schedule[{index}].case_sha256",
            )
            relation_order = entry["metamorphic_relation_order"]
            if not isinstance(relation_order, list):
                raise ValueError("scheduled metamorphic relation order must be a list")
            normalized_relations = [str(value).strip() for value in relation_order]
            if any(not value for value in normalized_relations) or len(
                set(normalized_relations)
            ) != len(normalized_relations):
                raise ValueError(
                    "scheduled metamorphic relation order must be unique and non-empty"
                )
            schedule_counts[campaign_seed] += 1

    implementation = _mapping(manifest.get("implementation"), field="implementation")
    for field in ("source_sha256", "environment_sha256", "config_sha256"):
        _validated_sha256(str(implementation.get(field, "")), field=f"implementation.{field}")
    model_sha = str(implementation.get("model_sha256", "") or "")
    if model_sha:
        _validated_sha256(model_sha, field="implementation.model_sha256")

    fidelity = _mapping(manifest.get("fidelity"), field="fidelity")
    _mapping(fidelity.get("reference_policy"), field="fidelity.reference_policy")
    _mapping(fidelity.get("low_policy"), field="fidelity.low_policy")
    actions = fidelity.get("actions")
    if not isinstance(actions, list) or not actions:
        raise ValueError("fidelity.actions must be a non-empty list")
    action_ids: set[str] = set()
    action_axes: set[str] = set()
    for index, action_value in enumerate(actions):
        action = _mapping(action_value, field=f"fidelity.actions[{index}]")
        action_id = str(action.get("action_id", "")).strip()
        axis = str(action.get("axis", ""))
        if not action_id or action_id in action_ids:
            raise ValueError("fidelity action ids must be non-empty and unique")
        if axis not in AUDIT_AXES:
            raise ValueError(f"unknown fidelity action axis: {axis}")
        action_ids.add(action_id)
        action_axes.add(axis)

    audit = _mapping(manifest.get("audit"), field="audit")
    if audit.get("hash_algorithm") != "sha256-first64-uniform-v1":
        raise ValueError("unsupported audit hash algorithm")
    if audit.get("serialization") != "canonical-json-sort-keys-v1":
        raise ValueError("unsupported audit serialization contract")
    if not str(audit.get("salt", "")).strip():
        raise ValueError("audit.salt cannot be empty")
    raw_axes = _mapping(audit.get("axes"), field="audit.axes")
    audit_policies = tuple(
        AuditAxisPolicy.from_payload(str(axis), _mapping(value, field=f"audit.axes.{axis}"))
        for axis, value in raw_axes.items()
    )
    registered_axes = {policy.axis for policy in audit_policies}
    if "joint" not in registered_axes:
        raise ValueError("a nonzero joint audit policy is required")
    if not action_axes.issubset(registered_axes):
        missing = sorted(action_axes - registered_axes)
        raise ValueError(f"fidelity actions lack audit policies: {missing}")

    debt = _mapping(manifest.get("coverage_debt"), field="coverage_debt")
    if debt.get("forced_settlement") is not True:
        raise ValueError("coverage debt must enable forced settlement")
    raw_strata = debt.get("strata")
    if not isinstance(raw_strata, list) or not raw_strata:
        raise ValueError("coverage_debt.strata must be a non-empty list")
    debt_policies: list[CoverageDebtPolicy] = []
    stratum_ids: set[str] = set()
    debt_policy_keys: set[tuple[str, str]] = set()
    for index, value in enumerate(raw_strata):
        row = _mapping(value, field=f"coverage_debt.strata[{index}]")
        stratum_id = str(row.get("stratum_id", "")).strip()
        axis = str(row.get("axis", ""))
        if not stratum_id or stratum_id in stratum_ids:
            raise ValueError("coverage-debt stratum ids must be non-empty and unique")
        if axis not in registered_axes:
            raise ValueError(f"coverage-debt axis lacks audit policy: {axis}")
        stratum_ids.add(stratum_id)
        debt_policy = CoverageDebtPolicy(
            axis=axis,
            stratum=str(row.get("stratum", stratum_id)),
            max_age_cases=_positive_int(
                row.get("max_age_cases"),
                field=f"coverage_debt.strata[{index}].max_age_cases",
            ),
            max_outstanding=_positive_int(
                row.get("max_outstanding"),
                field=f"coverage_debt.strata[{index}].max_outstanding",
            ),
        )
        if debt_policy.key in debt_policy_keys:
            raise ValueError("duplicate coverage-debt axis/stratum policies")
        debt_policy_keys.add(debt_policy.key)
        debt_policies.append(debt_policy)
    covered_debt_axes = {policy.axis for policy in debt_policies}
    missing_debt_axes = sorted(registered_axes - covered_debt_axes)
    if missing_debt_axes:
        raise ValueError(
            f"registered audit axes lack coverage-debt policies: {missing_debt_axes}"
        )

    estimators = _mapping(manifest.get("estimators"), field="estimators")
    if estimators.get("event") not in {"horvitz_thompson", "doubly_robust"}:
        raise ValueError("unsupported event estimator")
    if estimators.get("root") not in {
        "cluster_inclusion_plus_capture_recapture",
        "paired_full_reference_only",
    }:
        raise ValueError("unsupported root estimator")
    if estimators.get("confidence_sequence") not in {
        "clipped-freedman-v1",
        "asymmetric-clipped-freedman-partial-id-v2",
    }:
        raise ValueError("unsupported confidence sequence")
    alpha = float(estimators.get("alpha", 0.0))
    if not 0 < alpha < 1:
        raise ValueError("estimators.alpha must satisfy 0 < alpha < 1")
    minimum_effective = float(estimators.get("minimum_effective_audits", 0.0))
    if minimum_effective < 1:
        raise ValueError("minimum_effective_audits must be at least one")
    alpha_allocation = _mapping(
        estimators.get("alpha_allocation"),
        field="estimators.alpha_allocation",
    )
    if set(alpha_allocation) != registered_axes:
        raise ValueError(
            "estimators.alpha_allocation must cover every registered audit axis"
        )
    allocated_alpha = 0.0
    for axis, value in alpha_allocation.items():
        axis_alpha = float(value)
        if not 0 < axis_alpha < 1:
            raise ValueError(f"estimators.alpha_allocation.{axis} is invalid")
        allocated_alpha += axis_alpha
    if allocated_alpha > alpha + 1e-15:
        raise ValueError("axis alpha allocation cannot exceed estimators.alpha")
    event_bound = float(estimators.get("event_bound", 0.0))
    if event_bound <= 0:
        raise ValueError("estimators.event_bound must be positive")
    event_metrics = _mapping(
        estimators.get("event_metrics"),
        field="estimators.event_metrics",
    )
    if set(event_metrics) != registered_axes:
        raise ValueError("estimators.event_metrics must cover every registered audit axis")
    unsupported_event_metrics = sorted(
        {
            str(value)
            for value in event_metrics.values()
            if str(value) not in RLCMF_EVENT_METRICS
        }
    )
    if unsupported_event_metrics:
        raise ValueError(
            f"unsupported RLCMF event metrics: {unsupported_event_metrics}"
        )
    if event_bound != 1.0:
        raise ValueError("RLCMF v1 indicator event metrics require event_bound=1")
    minimum_floor = min(policy.effective_floor for policy in audit_policies)
    weight_clip = float(estimators.get("weight_clip", 0.0))
    if weight_clip < 1.0 / float(minimum_floor):
        raise ValueError(
            "weight_clip cannot be lower than the maximum registered inverse propensity"
        )

    safety = _mapping(manifest.get("safety"), field="safety")
    if safety.get("states") != [
        "calibration_full",
        "adaptive_safe",
        "caution",
        "full_rollback",
    ]:
        raise ValueError("safety states must use the registered RLCMF order")
    decision_rule = safety.get("decision_rule")
    if decision_rule is not None and decision_rule != (
        "asymmetric-confirmed-violation-v2"
    ):
        raise ValueError("unsupported RLCMF safety decision rule")
    triggers = safety.get("rollback_triggers")
    if not isinstance(triggers, list) or not triggers:
        raise ValueError("at least one rollback trigger is required")
    missing_triggers = sorted(RLCMF_REQUIRED_ROLLBACK_TRIGGERS - set(triggers))
    if missing_triggers:
        raise ValueError(f"missing mandatory rollback triggers: {missing_triggers}")
    safety_ladder = safety.get("propensity_ladder")
    if not isinstance(safety_ladder, list) or not safety_ladder:
        raise ValueError("safety.propensity_ladder cannot be empty")
    normalized_safety_ladder = [float(value) for value in safety_ladder]
    if normalized_safety_ladder != sorted(set(normalized_safety_ladder)):
        raise ValueError("safety propensity ladder must be strictly increasing")
    if any(value <= 0 or value > 1 for value in normalized_safety_ladder):
        raise ValueError("safety propensity ladder values must satisfy 0 < p <= 1")
    if normalized_safety_ladder[-1] != 1.0:
        raise ValueError("safety propensity ladder must end at full fidelity 1.0")
    for policy in audit_policies:
        policy_values = {float(value) for value in policy.ladder}
        if not set(normalized_safety_ladder).issubset(policy_values):
            raise ValueError(
                f"safety propensity ladder is not registered for axis {policy.axis}"
            )
        if float(policy.initial) not in normalized_safety_ladder:
            raise ValueError(
                f"initial propensity is absent from the safety ladder for axis {policy.axis}"
            )
    _positive_int(
        safety.get("decision_epoch_cases"),
        field="safety.decision_epoch_cases",
    )
    axis_thresholds = _mapping(
        safety.get("axis_thresholds"),
        field="safety.axis_thresholds",
    )
    if set(axis_thresholds) != registered_axes:
        raise ValueError("safety.axis_thresholds must cover every registered audit axis")
    for axis, raw_thresholds in axis_thresholds.items():
        axis_values = _mapping(
            raw_thresholds,
            field=f"safety.axis_thresholds.{axis}",
        )
        if set(axis_values) != RLCMF_REQUIRED_SAFETY_AXIS_THRESHOLDS:
            raise ValueError(
                f"safety thresholds for {axis} must contain exactly "
                f"{sorted(RLCMF_REQUIRED_SAFETY_AXIS_THRESHOLDS)}"
            )
        for key in (
            "maximum_event_miss_rate",
            "minimum_event_recall",
            "maximum_misclassification_rate",
            "maximum_uncaptured_full_event_rate",
        ):
            value = float(axis_values[key])
            if not 0 <= value <= 1:
                raise ValueError(f"safety.axis_thresholds.{axis}.{key} must be in [0, 1]")
        caution_margin = float(axis_values["caution_margin"])
        if not 0 <= caution_margin < 1:
            raise ValueError(
                f"safety.axis_thresholds.{axis}.caution_margin must be in [0, 1)"
            )
    recovery = _mapping(safety.get("recovery"), field="safety.recovery")
    if set(recovery) != {"enabled"} or recovery.get("enabled") is not False:
        raise ValueError("RLCMF v1 requires terminal full rollback with recovery disabled")
    if "horizon" in safety:
        horizon = _mapping(safety["horizon"], field="safety.horizon")
        expected_horizon_fields = {
            "enabled",
            "allocation_rule",
            "minimum_low_path_cases",
            "debt_reserve_cases",
            "target_confidence_width",
            "full_audit_deadline_remaining_cases",
        }
        if set(horizon) != expected_horizon_fields:
            raise ValueError(
                "safety.horizon must contain exactly the registered fields"
            )
        if horizon["enabled"] is not True:
            raise ValueError("safety.horizon.enabled must be true when present")
        if horizon["allocation_rule"] != (
            "minimum-required-ladder-with-low-path-reserve-v1"
        ):
            raise ValueError("unsupported horizon allocation rule")
        for key in (
            "minimum_low_path_cases",
            "debt_reserve_cases",
            "full_audit_deadline_remaining_cases",
        ):
            value = int(horizon[key])
            if value < 0:
                raise ValueError(f"safety.horizon.{key} cannot be negative")
        width = float(horizon["target_confidence_width"])
        if not 0 < width <= 1:
            raise ValueError(
                "safety.horizon.target_confidence_width must be in (0, 1]"
            )

    voi = _mapping(manifest.get("voi"), field="voi")
    expected_voi_fields = {
        "enabled",
        "model",
        "model_sha256",
        "features",
        "weights",
        "minimum_cost_ms",
        "raise_thresholds",
        "maximum_extra_expected_cost_ms_per_case",
        "raise_joint_with_component",
        "permitted_influence",
    }
    if set(voi) != expected_voi_fields:
        raise ValueError(
            f"voi must contain exactly the registered fields: {sorted(expected_voi_fields)}"
        )
    if voi.get("enabled") is not True:
        raise ValueError("RLCMF v1 requires the registered VOI scheduler")
    if voi.get("model") != "registered-linear-root-voi-v1":
        raise ValueError("unsupported RLCMF VOI model")
    _validated_sha256(str(voi.get("model_sha256", "")), field="voi.model_sha256")
    if voi.get("features") != list(VOI_FEATURES):
        raise ValueError("voi.features must use the registered order")
    weights = _mapping(voi.get("weights"), field="voi.weights")
    if set(weights) != set(VOI_FEATURES):
        raise ValueError("voi.weights must cover every registered feature")
    for feature, raw_weight in weights.items():
        weight = float(raw_weight)
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f"voi.weights.{feature} must be finite and non-negative")
    minimum_cost_ms = float(voi.get("minimum_cost_ms", 0.0))
    if not math.isfinite(minimum_cost_ms) or minimum_cost_ms <= 0:
        raise ValueError("voi.minimum_cost_ms must be positive")
    raw_raise_thresholds = voi.get("raise_thresholds")
    if not isinstance(raw_raise_thresholds, list) or not raw_raise_thresholds:
        raise ValueError("voi.raise_thresholds must be a non-empty list")
    raise_thresholds = [
        VOIRaiseThreshold(
            minimum_voi_per_cpu_ms=float(
                _mapping(row, field=f"voi.raise_thresholds[{index}]")[
                    "minimum_voi_per_cpu_ms"
                ]
            ),
            ladder_steps=int(row["ladder_steps"]),
        )
        for index, row in enumerate(raw_raise_thresholds)
    ]
    threshold_scores = [row.minimum_voi_per_cpu_ms for row in raise_thresholds]
    threshold_steps = [row.ladder_steps for row in raise_thresholds]
    if threshold_scores != sorted(set(threshold_scores)) or threshold_steps != sorted(
        threshold_steps
    ):
        raise ValueError("voi raise thresholds must increase monotonically")
    maximum_extra = float(
        voi.get("maximum_extra_expected_cost_ms_per_case", -1.0)
    )
    if not math.isfinite(maximum_extra) or maximum_extra < 0:
        raise ValueError(
            "voi.maximum_extra_expected_cost_ms_per_case cannot be negative"
        )
    if voi.get("raise_joint_with_component") is not True:
        raise ValueError("RLCMF VOI component raises must retain a joint anchor")
    if voi.get("permitted_influence") != "raise_only":
        raise ValueError("RLCMF VOI influence must be raise_only")

    budget = _mapping(manifest.get("budget"), field="budget")
    calibration = _positive_int(
        budget.get("equal_calibration_cases_per_seed"),
        field="budget.equal_calibration_cases_per_seed",
    )
    max_cases = _positive_int(
        budget.get("max_cases_per_seed"),
        field="budget.max_cases_per_seed",
    )
    if calibration > max_cases:
        raise ValueError("equal calibration budget cannot exceed per-seed cap")
    if schedule_counts is not None:
        under_calibrated = sorted(
            seed
            for seed, count in schedule_counts.items()
            if count < calibration
        )
        if under_calibrated:
            raise ValueError(
                f"seed schedule underfunds equal calibration: {under_calibrated}"
            )
        over_budget = sorted(
            seed for seed, count in schedule_counts.items() if count > max_cases
        )
        if over_budget:
            raise ValueError(f"seed schedule exceeds per-seed cap: {over_budget}")
    if not str(budget.get("stopping_rule", "")).strip():
        raise ValueError("budget.stopping_rule cannot be empty")
    if budget.get("post_outcome_seed_replacement_allowed") is not False:
        raise ValueError("post-outcome seed replacement must be forbidden")
    if budget.get("favorable_early_stopping_allowed") is not False:
        raise ValueError("favorable early stopping must be forbidden")
    raw_paired_analysis = budget.get("paired_analysis")
    if raw_paired_analysis is not None:
        paired_analysis = _mapping(
            raw_paired_analysis,
            field="budget.paired_analysis",
        )
        required_analysis_fields = {
            "analysis_id",
            "unit",
            "pairing",
            "metrics",
            "primary_cost_metric",
            "confidence_level",
            "interval_method",
            "bootstrap_replicates",
            "block_length",
            "seed_derivation",
            "decision_rule",
        }
        if set(paired_analysis) != required_analysis_fields:
            raise ValueError(
                "budget.paired_analysis must contain the complete registered contract"
            )
        if not str(paired_analysis["analysis_id"]).strip():
            raise ValueError("paired analysis id cannot be empty")
        if paired_analysis["unit"] != "frozen_trace_case":
            raise ValueError("paired analysis unit must be frozen_trace_case")
        if paired_analysis["pairing"] != "common_case_full_static_shadow":
            raise ValueError("paired analysis must use the common-case shadow")
        metrics = paired_analysis["metrics"]
        allowed_metrics = {
            "backend_calls",
            "backend_reported_ms",
            "wall_ms",
            "process_cpu_ms",
        }
        if (
            not isinstance(metrics, list)
            or not metrics
            or len(set(metrics)) != len(metrics)
            or not set(metrics).issubset(allowed_metrics)
        ):
            raise ValueError("paired analysis metrics are invalid")
        if paired_analysis["primary_cost_metric"] not in metrics:
            raise ValueError("primary cost metric must be registered in metrics")
        confidence_level = float(paired_analysis["confidence_level"])
        if not 0.5 < confidence_level < 1:
            raise ValueError("paired analysis confidence level must be in (0.5, 1)")
        if paired_analysis["interval_method"] != (
            "circular-moving-block-bootstrap-percentile-v1"
        ):
            raise ValueError("unsupported paired analysis interval method")
        if int(paired_analysis["bootstrap_replicates"]) < 1_000:
            raise ValueError("paired analysis requires at least 1000 replicates")
        block_length = int(paired_analysis["block_length"])
        if block_length < 1 or (schedule_counts and block_length > sum(schedule_counts.values())):
            raise ValueError("paired analysis block length is outside the trace")
        if paired_analysis["seed_derivation"] != (
            "sha256-manifest-metric-counter-block-v1"
        ):
            raise ValueError("unsupported paired analysis seed derivation")
        if paired_analysis["decision_rule"] != (
            "upper-ratio-ci-below-one-v1"
        ):
            raise ValueError("unsupported paired analysis decision rule")

    promotion = _mapping(manifest.get("promotion"), field="promotion")
    if promotion.get("primary_metric") != RLCMF_PRIMARY_METRIC:
        raise ValueError("promotion primary metric does not match RLCMF")
    if promotion.get("requires_all_gates") is not True:
        raise ValueError("RLCMF promotion must require every registered gate")
    thresholds = _mapping(promotion.get("thresholds"), field="promotion.thresholds")
    if not thresholds:
        raise ValueError("promotion.thresholds cannot be empty")
    missing_thresholds = sorted(
        RLCMF_REQUIRED_PROMOTION_THRESHOLDS - set(thresholds)
    )
    if missing_thresholds:
        raise ValueError(
            f"missing mandatory promotion thresholds: {missing_thresholds}"
        )
    for key in ("minimum_exact_family_recall", "minimum_root_recall"):
        value = float(thresholds[key])
        if not 0 <= value <= 1:
            raise ValueError(f"promotion threshold {key} must be in [0, 1]")
    maximum_non_reproduced = int(
        thresholds["maximum_non_reproduced_candidate_cases"]
    )
    if maximum_non_reproduced < 0:
        raise ValueError(
            "maximum_non_reproduced_candidate_cases cannot be negative"
        )
    maximum_misclassification = float(
        thresholds["maximum_misclassification_rate"]
    )
    if not 0 <= maximum_misclassification <= 1:
        raise ValueError("maximum_misclassification_rate must be in [0, 1]")
    if float(thresholds["minimum_primary_metric_ratio"]) <= 0:
        raise ValueError("minimum_primary_metric_ratio must be positive")

    canonical_payload = json.loads(canonical_manifest_bytes(manifest))
    return ValidatedRLCMFManifest(
        payload=canonical_payload,
        sha256=manifest_sha256(canonical_payload),
        audit_policies=audit_policies,
        debt_policies=tuple(debt_policies),
    )


def freeze_rlcmf_manifest(path: Path, payload: Mapping[str, Any]) -> ValidatedRLCMFManifest:
    validated = validate_rlcmf_manifest(payload)
    encoded = canonical_manifest_bytes(validated.payload)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != encoded:
            raise FileExistsError(
                f"existing RLCMF manifest differs; choose a new manifest id/path: {path}"
            )
    else:
        with path.open("xb") as handle:
            handle.write(encoded)
    return validated


def load_frozen_rlcmf_manifest(
    path: Path,
    *,
    expected_sha256: str | None = None,
) -> ValidatedRLCMFManifest:
    path = Path(path)
    raw = path.read_bytes()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid RLCMF manifest JSON: {path}") from exc
    validated = validate_rlcmf_manifest(payload)
    if raw != canonical_manifest_bytes(validated.payload):
        raise ValueError("RLCMF manifest is not in canonical frozen form")
    if expected_sha256 is not None:
        expected = _validated_sha256(expected_sha256, field="expected_sha256")
        if validated.sha256 != expected:
            raise ValueError("RLCMF manifest digest mismatch")
    return validated
