from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping

from datadiff.risk_limiting_audit import AuditDecision
from datadiff.research_controls.rlcmf_shadow.rlcmf_estimation import (
    EventAuditRecord,
    EventEstimateSnapshot,
    RiskLimitingEventAccumulator,
)
from datadiff.research_controls.rlcmf_shadow.rlcmf_safety import AxisSafetyEvidence


RLCMF_AXIS_ALPHA_COMPONENTS = 3


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def _snapshot_id(snapshot: EventEstimateSnapshot) -> str:
    return hashlib.sha256(_canonical_json_bytes(snapshot.to_dict())).hexdigest()


@dataclass(frozen=True, slots=True)
class CounterfactualSafetyObservation:
    full_event_record: EventAuditRecord
    potential_misclassification_record: EventAuditRecord

    def to_dict(self) -> dict[str, Any]:
        return {
            "full_event_record": self.full_event_record.to_dict(),
            "potential_misclassification_record": (
                self.potential_misclassification_record.to_dict()
            ),
        }


@dataclass(frozen=True, slots=True)
class CounterfactualSafetySnapshot:
    axis: str
    metric: str
    miss: EventEstimateSnapshot
    full_event: EventEstimateSnapshot
    potential_misclassification: EventEstimateSnapshot
    event_recall_point: float | None
    event_recall_lower: float | None
    event_recall_upper: float | None
    capture_ready: bool
    effective_sample_size: float

    def to_axis_evidence(self) -> AxisSafetyEvidence:
        return AxisSafetyEvidence(
            axis=self.axis,
            effective_sample_size=self.effective_sample_size,
            miss_rate_lower=self.miss.confidence_lower,
            miss_rate_upper=self.miss.confidence_upper,
            event_recall_lower=(
                self.event_recall_lower if self.capture_ready else None
            ),
            event_recall_upper=self.event_recall_upper,
            misclassification_rate_lower=(
                self.potential_misclassification.confidence_lower
            ),
            misclassification_rate_upper=(
                self.potential_misclassification.confidence_upper
            ),
            full_event_rate_upper=self.full_event.confidence_upper,
            evidence_ids=(
                _snapshot_id(self.miss),
                _snapshot_id(self.full_event),
                _snapshot_id(self.potential_misclassification),
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "rlcmf-counterfactual-safety-snapshot-v1",
            "axis": self.axis,
            "metric": self.metric,
            "miss": self.miss.to_dict(),
            "full_event": self.full_event.to_dict(),
            "potential_misclassification": (
                self.potential_misclassification.to_dict()
            ),
            "event_recall_point": self.event_recall_point,
            "event_recall_lower": self.event_recall_lower,
            "event_recall_upper": self.event_recall_upper,
            "capture_ready": self.capture_ready,
            "effective_sample_size": self.effective_sample_size,
            "misclassification_scope": (
                "conservative_low_only_candidate_disagreement_proxy"
            ),
        }


class CounterfactualAxisSafetyTracker:
    """Simultaneous miss, recall-denominator, and disagreement bounds."""

    def __init__(
        self,
        *,
        miss_accumulator: RiskLimitingEventAccumulator,
        axis_alpha: float,
    ) -> None:
        self.miss_accumulator = miss_accumulator
        self.axis = miss_accumulator.axis
        self.metric = miss_accumulator.metric
        self.axis_alpha = float(axis_alpha)
        if not math.isfinite(self.axis_alpha) or not 0 < self.axis_alpha < 1:
            raise ValueError("axis_alpha must satisfy 0 < alpha < 1")
        component_alpha = self.axis_alpha / RLCMF_AXIS_ALPHA_COMPONENTS
        if not math.isclose(
            miss_accumulator.alpha,
            component_alpha,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError(
                "miss accumulator alpha must equal one third of the axis allocation"
            )
        shared = {
            "manifest_sha256": miss_accumulator.manifest_sha256,
            "axis": self.axis,
            "estimator": miss_accumulator.estimator,
            "event_bound": 1.0,
            "weight_clip": miss_accumulator.weight_clip,
            "alpha": component_alpha,
            "minimum_effective_audits": (
                miss_accumulator.minimum_effective_audits
            ),
            "confidence_sequence": (
                miss_accumulator.confidence_sequence_name
            ),
        }
        self.full_event_accumulator = RiskLimitingEventAccumulator(
            metric=f"{self.metric}:full_event_denominator",
            **shared,
        )
        self.potential_misclassification_accumulator = (
            RiskLimitingEventAccumulator(
                metric=f"{self.metric}:low_only_disagreement",
                **shared,
            )
        )

    def observe_auxiliary(
        self,
        decision: AuditDecision,
        *,
        miss_event: float | None,
        full_event: float | None,
        potential_misclassification: float | None,
    ) -> CounterfactualSafetyObservation:
        if decision.opportunity.axis != self.axis:
            raise ValueError("safety tracker axis differs from audit decision")
        if decision.selected:
            if any(
                value is None
                for value in (
                    miss_event,
                    full_event,
                    potential_misclassification,
                )
            ):
                raise ValueError(
                    "selected counterfactual audit requires all safety outcomes"
                )
            if float(miss_event) > float(full_event):
                raise ValueError("a missed event cannot exist without a full event")
        elif any(
            value is not None
            for value in (
                miss_event,
                full_event,
                potential_misclassification,
            )
        ):
            raise ValueError("unselected audit cannot reveal safety outcomes")
        full_record = self.full_event_accumulator.add_decision(
            decision,
            outcome=full_event,
        )
        disagreement_record = (
            self.potential_misclassification_accumulator.add_decision(
                decision,
                outcome=potential_misclassification,
            )
        )
        return CounterfactualSafetyObservation(
            full_event_record=full_record,
            potential_misclassification_record=disagreement_record,
        )

    def snapshot(self) -> CounterfactualSafetySnapshot:
        miss = self.miss_accumulator.snapshot()
        full = self.full_event_accumulator.snapshot()
        disagreement = self.potential_misclassification_accumulator.snapshot()
        effective_sample_size = min(
            miss.effective_sample_size,
            full.effective_sample_size,
            disagreement.effective_sample_size,
        )
        capture_ready = bool(
            effective_sample_size >= miss.minimum_effective_audits
            and full.confidence_lower > 0
        )
        if full.estimated_mean_bounded > 0:
            point = max(
                0.0,
                min(
                    1.0,
                    1.0
                    - miss.estimated_mean_bounded
                    / full.estimated_mean_bounded,
                ),
            )
        else:
            point = None
        if full.confidence_lower > 0:
            lower = max(
                0.0,
                1.0 - miss.confidence_upper / full.confidence_lower,
            )
        else:
            lower = None
        if full.confidence_upper > 0:
            upper = min(
                1.0,
                1.0 - miss.confidence_lower / full.confidence_upper,
            )
        else:
            upper = None
        return CounterfactualSafetySnapshot(
            axis=self.axis,
            metric=self.metric,
            miss=miss,
            full_event=full,
            potential_misclassification=disagreement,
            event_recall_point=point,
            event_recall_lower=lower,
            event_recall_upper=upper,
            capture_ready=capture_ready,
            effective_sample_size=effective_sample_size,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "rlcmf-counterfactual-safety-tracker-v1",
            "axis": self.axis,
            "metric": self.metric,
            "axis_alpha": self.axis_alpha,
            "alpha_split": {
                "miss": self.miss_accumulator.alpha,
                "full_event": self.full_event_accumulator.alpha,
                "potential_misclassification": (
                    self.potential_misclassification_accumulator.alpha
                ),
            },
            "snapshot": self.snapshot().to_dict(),
        }
