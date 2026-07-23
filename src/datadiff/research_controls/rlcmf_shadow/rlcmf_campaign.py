from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from datadiff.research_controls.rlcmf_shadow.rlcmf_manifest import ValidatedRLCMFManifest
from datadiff.research_controls.rlcmf_shadow.rlcmf_safety import SafetyDecision
from datadiff.research_controls.rlcmf_shadow.rlcmf_sentinel import (
    RLCMFSentinelCoordinator,
    SentinelAuditPlan,
)
from datadiff.research_controls.rlcmf_shadow.rlcmf_voi import (
    RootCauseVOIForecast,
    RootCauseVOIForecastSnapshot,
    VOIAllocationDecision,
    VOIAllocationPair,
)


@dataclass(frozen=True, slots=True)
class OmissionOpportunity:
    axis: str
    stratum: str
    item_key: str
    estimated_omitted_units: float

    @classmethod
    def from_payload(
        cls,
        axis: str,
        payload: Mapping[str, Any],
    ) -> "OmissionOpportunity":
        return cls(
            axis=str(axis),
            stratum=str(payload["stratum"]),
            item_key=str(payload["item_key"]),
            estimated_omitted_units=float(payload["estimated_omitted_units"]),
        )


class RLCMFCampaignRuntime:
    """Sequential manifest-driven control plane for audited campaign cases."""

    def __init__(self, manifest: ValidatedRLCMFManifest) -> None:
        self.manifest = manifest
        self.controller = manifest.build_safety_controller()
        self.voi_scheduler = manifest.build_voi_scheduler()
        self.coordinator: RLCMFSentinelCoordinator = (
            manifest.build_sentinel_coordinator()
        )
        self.case_counts = {
            int(seed): 0 for seed in manifest.payload["seed_plan"]["seeds"]
        }
        self.completed_cases = 0
        self.non_reproduced_candidate_cases = 0
        self._pending_seed: int | None = None
        self._pending_plans: dict[str, SentinelAuditPlan] = {}
        self._pending_voi_decision: VOIAllocationDecision | None = None
        self._pending_coupling_id: str = ""
        self.voi_decisions: list[VOIAllocationDecision] = []
        self.voi_allocation_pairs: list[VOIAllocationPair] = []
        self.current_safety_decision = self.controller.evaluate(
            completed_cases=0,
            case_counts_by_seed=self.case_counts,
            evidence=self._axis_evidence(),
        )

    def _axis_evidence(self):
        return [
            tracker.snapshot().to_axis_evidence()
            for _axis, tracker in sorted(self.coordinator.safety_trackers.items())
        ]

    def plan_case(
        self,
        *,
        campaign_seed: int,
        omissions: Mapping[str, OmissionOpportunity | Mapping[str, Any]],
        voi_forecasts: tuple[RootCauseVOIForecast, ...] | list[RootCauseVOIForecast] = (),
        voi_snapshot: RootCauseVOIForecastSnapshot | None = None,
        coupled: bool = False,
        coupling_key: str = "",
    ) -> dict[str, SentinelAuditPlan]:
        if self._pending_seed is not None:
            raise ValueError("the previous RLCMF case has not been completed")
        seed = int(campaign_seed)
        if seed not in self.case_counts:
            raise ValueError("campaign seed is absent from the frozen manifest")
        if voi_snapshot is not None:
            if voi_forecasts:
                raise ValueError("VOI snapshot cannot be combined with raw forecasts")
            allocation_pair = self.voi_scheduler.allocate_with_control(
                self.current_safety_decision,
                snapshot=voi_snapshot,
                allocation_epoch=self.completed_cases,
            )
            voi_decision = allocation_pair.voi_decision
            self.voi_allocation_pairs.append(allocation_pair)
        else:
            voi_decision = self.voi_scheduler.allocate(
                self.current_safety_decision,
                forecasts=voi_forecasts,
                allocation_epoch=self.completed_cases,
            )
        normalized_opportunities: dict[str, OmissionOpportunity] = {}
        for axis, raw_opportunity in sorted(omissions.items()):
            opportunity = (
                raw_opportunity
                if isinstance(raw_opportunity, OmissionOpportunity)
                else OmissionOpportunity.from_payload(axis, raw_opportunity)
            )
            if opportunity.axis != axis:
                raise ValueError("omission mapping key must match its axis")
            normalized_opportunities[axis] = opportunity
        if coupled and normalized_opportunities:
            resolved_coupling_key = str(coupling_key or "").strip() or (
                f"seed={seed}:case={self.completed_cases}:all-axes"
            )
            plans, decision_set = self.coordinator.plan_coupled_axes(
                campaign_seed=seed,
                case_index=self.completed_cases,
                opportunities={
                    axis: {
                        "stratum": opportunity.stratum,
                        "item_key": opportunity.item_key,
                        "estimated_omitted_units": (
                            opportunity.estimated_omitted_units
                        ),
                    }
                    for axis, opportunity in normalized_opportunities.items()
                },
                coupling_key=resolved_coupling_key,
                safety_decision=self.current_safety_decision,
                voi_decision=voi_decision,
            )
            self._pending_coupling_id = decision_set.coupling_id
        else:
            plans = {}
            for axis, opportunity in sorted(normalized_opportunities.items()):
                plans[axis] = self.coordinator.plan_axis(
                    campaign_seed=seed,
                    case_index=self.completed_cases,
                    axis=axis,
                    stratum=opportunity.stratum,
                    item_key=opportunity.item_key,
                    estimated_omitted_units=opportunity.estimated_omitted_units,
                    safety_decision=self.current_safety_decision,
                    voi_decision=voi_decision,
                )
            self._pending_coupling_id = ""
        self._pending_seed = seed
        self._pending_plans = plans
        self._pending_voi_decision = voi_decision
        self.voi_decisions.append(voi_decision)
        return dict(plans)

    def complete_case(
        self,
        *,
        non_reproduced_candidate_increment: int = 0,
        coverage_debt_violation: bool = False,
        confirmed_misclassification_violation: bool = False,
        manifest_integrity: bool = True,
        propensity_integrity: bool = True,
        ledger_integrity: bool = True,
        model_integrity: bool = True,
    ) -> SafetyDecision:
        if self._pending_seed is None:
            raise ValueError("no RLCMF case is pending completion")
        pending_ids = {plan.plan_id for plan in self._pending_plans.values()}
        finalized_ids = set(self.coordinator.results)
        missing = sorted(pending_ids - finalized_ids)
        if missing:
            raise ValueError(f"RLCMF case has unfinalized sentinel plans: {missing}")
        increment = int(non_reproduced_candidate_increment)
        if increment < 0:
            raise ValueError("non-reproduced candidate increment cannot be negative")
        self.non_reproduced_candidate_cases += increment
        self.case_counts[self._pending_seed] += 1
        self.completed_cases += 1
        self.current_safety_decision = self.controller.evaluate(
            completed_cases=self.completed_cases,
            case_counts_by_seed=self.case_counts,
            evidence=self._axis_evidence(),
            non_reproduced_candidate_cases=(
                self.non_reproduced_candidate_cases
            ),
            coverage_debt_violation=coverage_debt_violation,
            confirmed_misclassification_violation=(
                confirmed_misclassification_violation
            ),
            manifest_integrity=manifest_integrity,
            propensity_integrity=propensity_integrity,
            ledger_integrity=ledger_integrity,
            model_integrity=model_integrity,
        )
        self._pending_seed = None
        self._pending_plans = {}
        self._pending_voi_decision = None
        self._pending_coupling_id = ""
        return self.current_safety_decision

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "rlcmf-campaign-runtime-v1",
            "manifest_sha256": self.manifest.sha256,
            "completed_cases": self.completed_cases,
            "case_counts_by_seed": dict(sorted(self.case_counts.items())),
            "non_reproduced_candidate_cases": (
                self.non_reproduced_candidate_cases
            ),
            "pending_seed": self._pending_seed,
            "pending_plan_ids": sorted(
                plan.plan_id for plan in self._pending_plans.values()
            ),
            "pending_coupling_id": self._pending_coupling_id,
            "pending_voi_decision": (
                self._pending_voi_decision.to_dict()
                if self._pending_voi_decision is not None
                else None
            ),
            "voi_decisions": [
                decision.to_dict() for decision in self.voi_decisions
            ],
            "voi_allocation_pairs": [
                pair.to_dict() for pair in self.voi_allocation_pairs
            ],
            "current_safety_decision": self.current_safety_decision.to_dict(),
            "controller": self.controller.to_dict(),
            "sentinel": self.coordinator.to_dict(),
        }
