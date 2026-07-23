from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping

from datadiff.coverage_debt import CoverageDebtLedger
from datadiff.risk_limiting_audit import (
    AUDIT_AXES,
    AuditDecision,
    AuditOpportunity,
    CoupledAuditDecisionSet,
    DeterministicSentinelAuditor,
    _probability_text,
)
from datadiff.research_controls.rlcmf_shadow.rlcmf_estimation import (
    EventAuditRecord,
    RiskLimitingEventAccumulator,
)
from datadiff.research_controls.rlcmf_shadow.rlcmf_safety import SafetyDecision
from datadiff.research_controls.rlcmf_shadow.rlcmf_safety_evidence import (
    CounterfactualAxisSafetyTracker,
    CounterfactualSafetyObservation,
)
from datadiff.research_controls.rlcmf_shadow.rlcmf_voi import VOIAllocationDecision


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


@dataclass(frozen=True, slots=True)
class SentinelAuditPlan:
    plan_id: str
    axis: str
    stratum: str
    item_key: str
    estimated_omitted_units: float
    decision: AuditDecision
    created_debt_id: str
    forced_debt_id: str
    voi_allocation_id: str
    coupling_id: str = ""

    @property
    def selected(self) -> bool:
        return self.decision.selected

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "rlcmf-sentinel-plan-v1",
            "plan_id": self.plan_id,
            "axis": self.axis,
            "stratum": self.stratum,
            "item_key": self.item_key,
            "estimated_omitted_units": self.estimated_omitted_units,
            "selected": self.selected,
            "created_debt_id": self.created_debt_id,
            "forced_debt_id": self.forced_debt_id,
            "voi_allocation_id": self.voi_allocation_id,
            "coupling_id": self.coupling_id,
            "decision": self.decision.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class SentinelAuditResult:
    result_id: str
    plan_id: str
    axis: str
    selected: bool
    outcome: float | None
    prediction: float
    estimator_record: EventAuditRecord
    created_debt_id: str
    settled_debt_id: str
    safety_observation: CounterfactualSafetyObservation | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "rlcmf-sentinel-result-v1",
            "result_id": self.result_id,
            "plan_id": self.plan_id,
            "axis": self.axis,
            "selected": self.selected,
            "outcome": self.outcome,
            "prediction": self.prediction,
            "created_debt_id": self.created_debt_id,
            "settled_debt_id": self.settled_debt_id,
            "estimator_record": self.estimator_record.to_dict(),
            "safety_observation": (
                self.safety_observation.to_dict()
                if self.safety_observation is not None
                else None
            ),
        }


class RLCMFSentinelCoordinator:
    """Tie pre-outcome audit draws to debt and event-estimator accounting."""

    def __init__(
        self,
        *,
        auditor: DeterministicSentinelAuditor,
        debt_ledger: CoverageDebtLedger,
        event_accumulators: Mapping[str, RiskLimitingEventAccumulator],
        safety_trackers: Mapping[str, CounterfactualAxisSafetyTracker] | None = None,
        allow_direct_propensity_for_testing: bool = False,
    ) -> None:
        self.auditor = auditor
        self.debt_ledger = debt_ledger
        self.manifest_sha256 = auditor.manifest_sha256
        if debt_ledger.manifest_sha256 != self.manifest_sha256:
            raise ValueError("auditor and debt ledger manifest hashes differ")
        self.event_accumulators = dict(event_accumulators)
        self.allow_direct_propensity_for_testing = bool(
            allow_direct_propensity_for_testing
        )
        if not self.event_accumulators:
            raise ValueError("at least one event accumulator is required")
        for axis, accumulator in self.event_accumulators.items():
            if axis not in AUDIT_AXES or accumulator.axis != axis:
                raise ValueError("event accumulator key must match its audit axis")
            if accumulator.manifest_sha256 != self.manifest_sha256:
                raise ValueError("event accumulator manifest hash differs")
            if axis not in self.auditor.policies:
                raise ValueError(f"event accumulator axis lacks audit policy: {axis}")
        self.safety_trackers = dict(safety_trackers or {})
        if not set(self.safety_trackers).issubset(self.event_accumulators):
            raise ValueError("safety tracker axes must have event accumulators")
        for axis, tracker in self.safety_trackers.items():
            if tracker.axis != axis:
                raise ValueError("safety tracker key must match its audit axis")
            if tracker.miss_accumulator is not self.event_accumulators[axis]:
                raise ValueError(
                    "safety tracker must share the coordinator miss accumulator"
                )
        self.plans: dict[str, SentinelAuditPlan] = {}
        self.results: dict[str, SentinelAuditResult] = {}
        self.coupled_decision_sets: dict[str, CoupledAuditDecisionSet] = {}
        self._opportunities: set[tuple[int, int, str, str, int]] = set()

    def plan_axis(
        self,
        *,
        campaign_seed: int,
        case_index: int,
        axis: str,
        stratum: str,
        item_key: str,
        estimated_omitted_units: float,
        safety_decision: SafetyDecision | None = None,
        voi_decision: VOIAllocationDecision | None = None,
        requested_propensity: float | int | str | None = None,
        decision_epoch: int | None = None,
    ) -> SentinelAuditPlan:
        if axis not in self.event_accumulators:
            raise KeyError(f"no RLCMF event accumulator for axis: {axis}")
        normalized_stratum = str(stratum or "").strip()
        normalized_item_key = str(item_key or "").strip()
        if not normalized_stratum or not normalized_item_key:
            raise ValueError("sentinel stratum and item_key cannot be empty")
        units = float(estimated_omitted_units)
        if not math.isfinite(units) or units <= 0:
            raise ValueError("estimated_omitted_units must be positive")
        if safety_decision is not None:
            if requested_propensity is not None or decision_epoch is not None:
                raise ValueError(
                    "safety_decision cannot be combined with a direct propensity"
                )
            if safety_decision.manifest_sha256 != self.manifest_sha256:
                raise ValueError("safety decision manifest hash differs")
            if not safety_decision.verify_digest():
                raise ValueError("safety decision digest is invalid")
            if safety_decision.completed_cases > int(case_index):
                raise ValueError("safety decision cannot use future case history")
            requested = safety_decision.propensity_for(axis)
            if voi_decision is not None:
                if voi_decision.manifest_sha256 != self.manifest_sha256:
                    raise ValueError("VOI allocation manifest hash differs")
                if not voi_decision.verify_digest():
                    raise ValueError("VOI allocation digest is invalid")
                if voi_decision.safety_decision_id != safety_decision.decision_id:
                    raise ValueError("VOI allocation does not match the safety decision")
                if voi_decision.allocation_epoch != int(case_index):
                    raise ValueError("VOI allocation epoch does not match the case")
                requested = voi_decision.propensity_for(axis)
                if requested < safety_decision.propensity_for(axis):
                    raise ValueError("VOI allocation cannot lower safety propensity")
            epoch = safety_decision.decision_epoch
        else:
            if voi_decision is not None:
                raise ValueError("VOI allocation requires a safety decision")
            if not self.allow_direct_propensity_for_testing:
                raise ValueError(
                    "production sentinel planning requires a verified safety decision"
                )
            if requested_propensity is None or decision_epoch is None:
                raise ValueError(
                    "direct sentinel planning requires propensity and decision_epoch"
                )
            requested = requested_propensity
            epoch = int(decision_epoch)

        opportunity = AuditOpportunity(
            campaign_seed=int(campaign_seed),
            case_index=int(case_index),
            axis=axis,
            item_key=normalized_item_key,
            decision_epoch=epoch,
        )
        opportunity_key = (
            opportunity.campaign_seed,
            opportunity.case_index,
            opportunity.axis,
            opportunity.item_key,
            opportunity.decision_epoch,
        )
        if opportunity_key in self._opportunities:
            raise ValueError("duplicate RLCMF audit opportunity")

        forced_candidates = [
            item
            for item in self.debt_ledger.forced_settlements(
                current_case=opportunity.case_index
            )
            if item.axis == axis
            and item.stratum == normalized_stratum
            and self.debt_ledger.entries[item.debt_id].debt_kind == "quota"
        ]
        forced_debt_id = forced_candidates[0].debt_id if forced_candidates else ""
        reason = (
            f"coverage_debt_{forced_candidates[0].reason}"
            if forced_candidates
            else "registered_random_sentinel"
        )
        decision = self.auditor.decide(
            opportunity,
            requested_propensity=requested,
            forced=bool(forced_candidates),
            reason=reason,
        )
        created_debt_id = ""
        if not decision.selected:
            debt = self.debt_ledger.add_omission(
                axis=axis,
                stratum=normalized_stratum,
                item_key=normalized_item_key,
                created_case=opportunity.case_index,
                estimated_units=units,
                debt_kind="quota",
                exact_replayable=False,
                metadata={
                    "audit_decision_id": decision.decision_id,
                    "requested_propensity_exact": _probability_text(
                        decision.requested_propensity
                    ),
                    "effective_propensity_exact": _probability_text(
                        decision.propensity
                    ),
                    "selection_threshold_u64": decision.selection_threshold_u64,
                    "random_u64": decision.random_u64,
                    "decision_epoch": opportunity.decision_epoch,
                    "voi_allocation_id": (
                        voi_decision.allocation_id if voi_decision is not None else ""
                    ),
                },
            )
            created_debt_id = debt.debt_id
        plan_material = {
            "schema_version": "rlcmf-sentinel-plan-id-v1",
            "manifest_sha256": self.manifest_sha256,
            "axis": axis,
            "stratum": normalized_stratum,
            "item_key": normalized_item_key,
            "estimated_omitted_units": units,
            "audit_decision_id": decision.decision_id,
            "created_debt_id": created_debt_id,
            "forced_debt_id": forced_debt_id,
            "voi_allocation_id": (
                voi_decision.allocation_id if voi_decision is not None else ""
            ),
        }
        plan_id = hashlib.sha256(_canonical_json_bytes(plan_material)).hexdigest()
        plan = SentinelAuditPlan(
            plan_id=plan_id,
            axis=axis,
            stratum=normalized_stratum,
            item_key=normalized_item_key,
            estimated_omitted_units=units,
            decision=decision,
            created_debt_id=created_debt_id,
            forced_debt_id=forced_debt_id,
            voi_allocation_id=(
                voi_decision.allocation_id if voi_decision is not None else ""
            ),
        )
        self.plans[plan_id] = plan
        self._opportunities.add(opportunity_key)
        return plan

    def plan_coupled_axes(
        self,
        *,
        campaign_seed: int,
        case_index: int,
        opportunities: Mapping[str, Mapping[str, Any]],
        coupling_key: str,
        safety_decision: SafetyDecision | None = None,
        voi_decision: VOIAllocationDecision | None = None,
        requested_propensities: Mapping[str, float | int | str] | None = None,
        decision_epoch: int | None = None,
        require_joint_subset: bool = True,
    ) -> tuple[dict[str, SentinelAuditPlan], CoupledAuditDecisionSet]:
        if not opportunities:
            raise ValueError("coupled sentinel planning requires opportunities")
        axes = tuple(sorted(str(axis) for axis in opportunities))
        if any(axis not in self.event_accumulators for axis in axes):
            raise KeyError("coupled sentinel axis lacks an event accumulator")
        if safety_decision is not None:
            if requested_propensities is not None or decision_epoch is not None:
                raise ValueError(
                    "safety_decision cannot be combined with direct coupled propensities"
                )
            if safety_decision.manifest_sha256 != self.manifest_sha256:
                raise ValueError("safety decision manifest hash differs")
            if not safety_decision.verify_digest():
                raise ValueError("safety decision digest is invalid")
            if safety_decision.completed_cases > int(case_index):
                raise ValueError("safety decision cannot use future case history")
            requested = {
                axis: safety_decision.propensity_for(axis) for axis in axes
            }
            if voi_decision is not None:
                if voi_decision.manifest_sha256 != self.manifest_sha256:
                    raise ValueError("VOI allocation manifest hash differs")
                if not voi_decision.verify_digest():
                    raise ValueError("VOI allocation digest is invalid")
                if voi_decision.safety_decision_id != safety_decision.decision_id:
                    raise ValueError("VOI allocation does not match the safety decision")
                if voi_decision.allocation_epoch != int(case_index):
                    raise ValueError("VOI allocation epoch does not match the case")
                requested = {
                    axis: voi_decision.propensity_for(axis) for axis in axes
                }
                if any(
                    requested[axis] < safety_decision.propensity_for(axis)
                    for axis in axes
                ):
                    raise ValueError("VOI allocation cannot lower safety propensity")
            epoch = safety_decision.decision_epoch
        else:
            if voi_decision is not None:
                raise ValueError("VOI allocation requires a safety decision")
            if not self.allow_direct_propensity_for_testing:
                raise ValueError(
                    "production coupled planning requires a verified safety decision"
                )
            if requested_propensities is None or decision_epoch is None:
                raise ValueError(
                    "direct coupled planning requires propensities and decision_epoch"
                )
            if set(requested_propensities) != set(axes):
                raise ValueError(
                    "direct coupled propensity axes must match opportunities"
                )
            requested = dict(requested_propensities)
            epoch = int(decision_epoch)

        normalized: dict[str, dict[str, Any]] = {}
        audit_opportunities: dict[str, AuditOpportunity] = {}
        forced_debt_ids: dict[str, str] = {}
        reasons: dict[str, str] = {}
        opportunity_keys: dict[str, tuple[int, int, str, str, int]] = {}
        for axis in axes:
            payload = opportunities[axis]
            stratum = str(payload.get("stratum", "") or "").strip()
            item_key = str(payload.get("item_key", "") or "").strip()
            units = float(payload.get("estimated_omitted_units", 0.0))
            if not stratum or not item_key:
                raise ValueError("coupled sentinel stratum and item_key cannot be empty")
            if not math.isfinite(units) or units <= 0:
                raise ValueError("estimated_omitted_units must be positive")
            opportunity = AuditOpportunity(
                campaign_seed=int(campaign_seed),
                case_index=int(case_index),
                axis=axis,
                item_key=item_key,
                decision_epoch=int(epoch),
            )
            key = (
                opportunity.campaign_seed,
                opportunity.case_index,
                opportunity.axis,
                opportunity.item_key,
                opportunity.decision_epoch,
            )
            if key in self._opportunities:
                raise ValueError("duplicate RLCMF audit opportunity")
            forced_candidates = [
                item
                for item in self.debt_ledger.forced_settlements(
                    current_case=opportunity.case_index
                )
                if item.axis == axis
                and item.stratum == stratum
                and self.debt_ledger.entries[item.debt_id].debt_kind == "quota"
            ]
            forced_debt_ids[axis] = (
                forced_candidates[0].debt_id if forced_candidates else ""
            )
            reasons[axis] = (
                f"coverage_debt_{forced_candidates[0].reason}"
                if forced_candidates
                else "registered_coupled_sentinel"
            )
            normalized[axis] = {
                "stratum": stratum,
                "item_key": item_key,
                "estimated_omitted_units": units,
            }
            audit_opportunities[axis] = opportunity
            opportunity_keys[axis] = key

        decision_set = self.auditor.decide_coupled(
            audit_opportunities,
            requested_propensities=requested,
            forced_axes=[axis for axis in axes if forced_debt_ids[axis]],
            reasons=reasons,
            coupling_key=coupling_key,
            require_joint_subset=require_joint_subset,
        )
        plans: dict[str, SentinelAuditPlan] = {}
        voi_allocation_id = (
            voi_decision.allocation_id if voi_decision is not None else ""
        )
        for axis in axes:
            row = normalized[axis]
            decision = decision_set.decisions[axis]
            created_debt_id = ""
            if not decision.selected:
                debt = self.debt_ledger.add_omission(
                    axis=axis,
                    stratum=row["stratum"],
                    item_key=row["item_key"],
                    created_case=int(case_index),
                    estimated_units=row["estimated_omitted_units"],
                    debt_kind="quota",
                    exact_replayable=False,
                    metadata={
                        "audit_decision_id": decision.decision_id,
                        "coupling_id": decision_set.coupling_id,
                        "requested_propensity_exact": _probability_text(
                            decision.requested_propensity
                        ),
                        "effective_propensity_exact": _probability_text(
                            decision.propensity
                        ),
                        "selection_threshold_u64": decision.selection_threshold_u64,
                        "random_u64": decision.random_u64,
                        "decision_epoch": decision.opportunity.decision_epoch,
                        "voi_allocation_id": voi_allocation_id,
                    },
                )
                created_debt_id = debt.debt_id
            plan_material = {
                "schema_version": "rlcmf-sentinel-plan-id-v1",
                "manifest_sha256": self.manifest_sha256,
                "axis": axis,
                "stratum": row["stratum"],
                "item_key": row["item_key"],
                "estimated_omitted_units": row["estimated_omitted_units"],
                "audit_decision_id": decision.decision_id,
                "created_debt_id": created_debt_id,
                "forced_debt_id": forced_debt_ids[axis],
                "voi_allocation_id": voi_allocation_id,
                "coupling_id": decision_set.coupling_id,
            }
            plan_id = hashlib.sha256(
                _canonical_json_bytes(plan_material)
            ).hexdigest()
            plan = SentinelAuditPlan(
                plan_id=plan_id,
                axis=axis,
                stratum=row["stratum"],
                item_key=row["item_key"],
                estimated_omitted_units=row["estimated_omitted_units"],
                decision=decision,
                created_debt_id=created_debt_id,
                forced_debt_id=forced_debt_ids[axis],
                voi_allocation_id=voi_allocation_id,
                coupling_id=decision_set.coupling_id,
            )
            self.plans[plan_id] = plan
            self._opportunities.add(opportunity_keys[axis])
            plans[axis] = plan
        self.coupled_decision_sets[decision_set.coupling_id] = decision_set
        return plans, decision_set

    def finalize(
        self,
        plan: SentinelAuditPlan,
        *,
        outcome: float | None,
        prediction: float = 0.0,
        full_event: float | None = None,
        potential_misclassification: float | None = None,
    ) -> SentinelAuditResult:
        registered = self.plans.get(plan.plan_id)
        if registered != plan:
            raise ValueError("sentinel plan is not registered by this coordinator")
        if plan.plan_id in self.results:
            raise ValueError("sentinel plan has already been finalized")
        accumulator = self.event_accumulators[plan.axis]
        tracker = self.safety_trackers.get(plan.axis)
        if tracker is not None:
            if plan.selected:
                if any(
                    value is None
                    for value in (
                        outcome,
                        full_event,
                        potential_misclassification,
                    )
                ):
                    raise ValueError(
                        "selected audit requires complete counterfactual safety outcomes"
                    )
                if float(outcome) > float(full_event):
                    raise ValueError(
                        "a missed event cannot exist without a full event"
                    )
            elif full_event is not None or potential_misclassification is not None:
                raise ValueError("unselected audit cannot reveal safety outcomes")
        estimator_record = accumulator.add_decision(
            plan.decision,
            outcome=outcome,
            prediction=prediction,
        )
        safety_observation = None
        if tracker is not None:
            safety_observation = tracker.observe_auxiliary(
                plan.decision,
                miss_event=outcome,
                full_event=full_event,
                potential_misclassification=potential_misclassification,
            )
        elif full_event is not None or potential_misclassification is not None:
            raise ValueError(
                f"axis {plan.axis} has no registered counterfactual safety tracker"
            )
        settled_debt_id = ""
        if plan.selected:
            if plan.forced_debt_id:
                entry = self.debt_ledger.entries.get(plan.forced_debt_id)
                if entry is not None and entry.status == "outstanding":
                    settled = self.debt_ledger.settle(
                        entry.debt_id,
                        settled_case=plan.decision.opportunity.case_index,
                        reason="forced_current_case_sentinel",
                        audit_decision_id=plan.decision.decision_id,
                    )
                    settled_debt_id = settled.debt_id
            else:
                settled = self.debt_ledger.settle_oldest(
                    axis=plan.axis,
                    stratum=plan.stratum,
                    settled_case=plan.decision.opportunity.case_index,
                    reason="random_current_case_sentinel",
                    audit_decision_id=plan.decision.decision_id,
                    debt_kind="quota",
                )
                if settled is not None:
                    settled_debt_id = settled.debt_id
        result_material = {
            "schema_version": "rlcmf-sentinel-result-id-v1",
            "manifest_sha256": self.manifest_sha256,
            "plan_id": plan.plan_id,
            "outcome": outcome,
            "prediction": float(prediction),
            "estimator_decision_id": estimator_record.audit_decision_id,
            "created_debt_id": plan.created_debt_id,
            "settled_debt_id": settled_debt_id,
            "safety_observation": (
                safety_observation.to_dict()
                if safety_observation is not None
                else None
            ),
        }
        result_id = hashlib.sha256(_canonical_json_bytes(result_material)).hexdigest()
        result = SentinelAuditResult(
            result_id=result_id,
            plan_id=plan.plan_id,
            axis=plan.axis,
            selected=plan.selected,
            outcome=outcome,
            prediction=float(prediction),
            estimator_record=estimator_record,
            created_debt_id=plan.created_debt_id,
            settled_debt_id=settled_debt_id,
            safety_observation=safety_observation,
        )
        self.results[plan.plan_id] = result
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "rlcmf-sentinel-coordinator-v1",
            "manifest_sha256": self.manifest_sha256,
            "plan_count": len(self.plans),
            "finalized_count": len(self.results),
            "pending_plan_ids": sorted(set(self.plans) - set(self.results)),
            "allow_direct_propensity_for_testing": (
                self.allow_direct_propensity_for_testing
            ),
            "coupled_decision_sets": [
                self.coupled_decision_sets[key].to_dict()
                for key in sorted(self.coupled_decision_sets)
            ],
            "plans": [self.plans[key].to_dict() for key in sorted(self.plans)],
            "results": [self.results[key].to_dict() for key in sorted(self.results)],
            "debt": self.debt_ledger.to_dict(),
            "estimators": {
                axis: accumulator.to_dict()
                for axis, accumulator in sorted(self.event_accumulators.items())
            },
            "safety_trackers": {
                axis: tracker.to_dict()
                for axis, tracker in sorted(self.safety_trackers.items())
            },
        }
