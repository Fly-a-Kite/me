from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

from datadiff.risk_limiting_audit import (
    AUDIT_AXES,
    AuditDecision,
    _probability_text,
    _validated_sha256,
)


EVENT_ESTIMATORS = ("horvitz_thompson", "doubly_robust")
LEGACY_CONFIDENCE_SEQUENCE = "clipped-freedman-v1"
ASYMMETRIC_PARTIAL_ID_CONFIDENCE_SEQUENCE = (
    "asymmetric-clipped-freedman-partial-id-v2"
)
CONFIDENCE_SEQUENCES = (
    LEGACY_CONFIDENCE_SEQUENCE,
    ASYMMETRIC_PARTIAL_ID_CONFIDENCE_SEQUENCE,
)


def _finite(value: float | int, *, field: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be finite") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{field} must be finite")
    return parsed


def _positive(value: float | int, *, field: str) -> float:
    parsed = _finite(value, field=field)
    if parsed <= 0:
        raise ValueError(f"{field} must be positive")
    return parsed


def _propensity(value: Decimal | float | int | str) -> Decimal:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError("propensity must be a finite probability") from exc
    if not parsed.is_finite() or parsed <= 0 or parsed > 1:
        raise ValueError("propensity must satisfy 0 < p <= 1")
    return parsed


@dataclass(frozen=True, slots=True)
class EventAuditRecord:
    index: int
    audit_decision_id: str
    axis: str
    selected: bool
    propensity: Decimal
    outcome: float | None
    prediction: float
    contribution: float
    predictable_variance_bound: float
    inclusion_weight: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "audit_decision_id": self.audit_decision_id,
            "axis": self.axis,
            "selected": self.selected,
            "propensity": float(self.propensity),
            "propensity_exact": _probability_text(self.propensity),
            "outcome": self.outcome,
            "prediction": self.prediction,
            "contribution": self.contribution,
            "predictable_variance_bound": self.predictable_variance_bound,
            "inclusion_weight": self.inclusion_weight,
        }


@dataclass(frozen=True, slots=True)
class EventEstimateSnapshot:
    manifest_sha256: str
    axis: str
    metric: str
    estimator: str
    opportunity_count: int
    selected_audit_count: int
    expected_audit_count: float
    estimated_total_raw: float
    estimated_mean_raw: float
    estimated_mean_bounded: float
    confidence_lower: float
    confidence_upper: float
    confidence_radius_total: float
    confidence_lower_radius_total: float
    confidence_upper_radius_total: float
    predictable_variance_bound: float
    martingale_increment_bound: float
    confidence_lower_increment_bound: float
    confidence_upper_increment_bound: float
    identification_lower: float
    identification_upper: float
    identification_interval_applied: bool
    confidence_intersection_fallback: bool
    selected_weight_sum: float
    selected_weight_square_sum: float
    effective_sample_size: float
    minimum_effective_audits: float
    confidence_ready: bool
    maximum_inverse_propensity: float
    event_bound: float
    weight_clip: float
    alpha: float
    confidence_sequence: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "rlcmf-event-estimate-v1",
            **asdict(self),
        }


class ClippedFreedmanConfidenceSequence:
    """Two-sided, variance-stitched Freedman confidence sequence.

    The registered event and prediction bounds imply a bounded martingale
    difference after inverse-propensity correction. Variance epochs receive a
    summable alpha allocation, so optional stopping and repeated inspection do
    not invalidate the interval.
    """

    def __init__(
        self,
        *,
        alpha: float,
        event_bound: float,
        weight_clip: float,
        stitching_eta: float = 2.0,
    ) -> None:
        self.alpha = _finite(alpha, field="alpha")
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must satisfy 0 < alpha < 1")
        self.event_bound = _positive(event_bound, field="event_bound")
        self.weight_clip = _finite(weight_clip, field="weight_clip")
        if self.weight_clip < 1:
            raise ValueError("weight_clip must be at least one")
        self.stitching_eta = _finite(stitching_eta, field="stitching_eta")
        if self.stitching_eta <= 1:
            raise ValueError("stitching_eta must be greater than one")

    @property
    def martingale_increment_bound(self) -> float:
        return self.event_bound * max(1.0, self.weight_clip - 1.0)

    def variance_increment(
        self,
        propensity: Decimal | float | int | str,
        *,
        prediction: float | int | None = None,
    ) -> float:
        probability = float(_propensity(propensity))
        if prediction is None:
            residual_bound = self.event_bound
        else:
            predicted = _finite(prediction, field="prediction")
            if not 0 <= predicted <= self.event_bound:
                raise ValueError(
                    "prediction must stay within the registered event bound"
                )
            residual_bound = max(predicted, self.event_bound - predicted)
        return residual_bound**2 * (1.0 - probability) / probability

    def one_sided_increment_bounds(
        self,
        propensity: Decimal | float | int | str,
        *,
        prediction: float | int,
    ) -> tuple[float, float]:
        """Return predictable bounds for lower- and upper-CS errors.

        The first value bounds ``estimate - truth`` and controls the lower
        confidence limit. The second bounds ``truth - estimate`` and controls
        the upper confidence limit.
        """

        probability = float(_propensity(propensity))
        predicted = _finite(prediction, field="prediction")
        if not 0 <= predicted <= self.event_bound:
            raise ValueError(
                "prediction must stay within the registered event bound"
            )
        if probability == 1.0:
            return (0.0, 0.0)
        inverse_odds = (1.0 - probability) / probability
        lower_cs_error = max(
            predicted,
            (self.event_bound - predicted) * inverse_odds,
        )
        upper_cs_error = max(
            self.event_bound - predicted,
            predicted * inverse_odds,
        )
        return (lower_cs_error, upper_cs_error)

    def _variance_epoch(self, variance: float) -> tuple[int, float]:
        variance = _finite(variance, field="predictable_variance")
        if variance < 0:
            raise ValueError("predictable_variance cannot be negative")
        if variance == 0:
            return (0, 0.0)
        base = self.event_bound**2
        if variance <= base:
            return (0, base)
        epoch = int(
            math.ceil(
                math.log(variance / base) / math.log(self.stitching_eta)
                - 1e-15
            )
        )
        epoch = max(1, epoch)
        cap = base * self.stitching_eta**epoch
        while cap + (1e-15 * max(1.0, cap)) < variance:
            epoch += 1
            cap *= self.stitching_eta
        return (epoch, cap)

    def radius(self, predictable_variance: float) -> float:
        epoch, variance_cap = self._variance_epoch(predictable_variance)
        if variance_cap == 0:
            return 0.0
        # Allocate alpha/2 to each tail and 6/(pi^2 (k+1)^2) across epochs.
        epoch_alpha_one_tail = (
            (self.alpha / 2.0)
            * 6.0
            / (math.pi**2 * (epoch + 1) ** 2)
        )
        log_term = math.log(1.0 / epoch_alpha_one_tail)
        linear = self.martingale_increment_bound * log_term / 3.0
        return linear + math.sqrt(
            linear**2 + 2.0 * variance_cap * log_term
        )

    def _bound_epoch(self, increment_bound: float) -> tuple[int, float]:
        bound = _finite(increment_bound, field="increment_bound")
        if bound < 0:
            raise ValueError("increment_bound cannot be negative")
        if bound > self.martingale_increment_bound + (
            1e-15 * max(1.0, self.martingale_increment_bound)
        ):
            raise ValueError("increment_bound exceeds the registered weight clip")
        if bound == 0:
            return (0, 0.0)
        if bound <= self.event_bound:
            return (0, self.event_bound)
        epoch = int(
            math.ceil(
                math.log(bound / self.event_bound) / math.log(self.stitching_eta)
                - 1e-15
            )
        )
        epoch = max(1, epoch)
        cap = self.event_bound * self.stitching_eta**epoch
        while cap + (1e-15 * max(1.0, cap)) < bound:
            epoch += 1
            cap *= self.stitching_eta
        return (epoch, cap)

    def asymmetric_radius(
        self,
        predictable_variance: float,
        *,
        increment_bound: float,
    ) -> float:
        """One-tail radius stitched over variance and realized-bound epochs."""

        variance_epoch, variance_cap = self._variance_epoch(predictable_variance)
        if variance_cap == 0:
            return 0.0
        bound_epoch, bound_cap = self._bound_epoch(increment_bound)
        variance_weight = 6.0 / (math.pi**2 * (variance_epoch + 1) ** 2)
        bound_weight = 6.0 / (math.pi**2 * (bound_epoch + 1) ** 2)
        cell_alpha_one_tail = (
            (self.alpha / 2.0) * variance_weight * bound_weight
        )
        log_term = math.log(1.0 / cell_alpha_one_tail)
        linear = bound_cap * log_term / 3.0
        return linear + math.sqrt(
            linear**2 + 2.0 * variance_cap * log_term
        )


class RiskLimitingEventAccumulator:
    """Audit-linked HT/DR event estimator with an anytime-valid safety bound."""

    def __init__(
        self,
        *,
        manifest_sha256: str,
        axis: str,
        metric: str,
        estimator: str,
        event_bound: float,
        weight_clip: float,
        alpha: float,
        minimum_effective_audits: float,
        confidence_sequence: str = LEGACY_CONFIDENCE_SEQUENCE,
    ) -> None:
        self.manifest_sha256 = _validated_sha256(
            manifest_sha256,
            field="manifest_sha256",
        )
        self.axis = str(axis)
        if self.axis not in AUDIT_AXES:
            raise ValueError(f"unknown estimator axis: {self.axis}")
        self.metric = str(metric or "").strip()
        if not self.metric:
            raise ValueError("metric cannot be empty")
        self.estimator = str(estimator)
        if self.estimator not in EVENT_ESTIMATORS:
            raise ValueError(f"unsupported event estimator: {self.estimator}")
        self.event_bound = _positive(event_bound, field="event_bound")
        self.weight_clip = _finite(weight_clip, field="weight_clip")
        if self.weight_clip < 1:
            raise ValueError("weight_clip must be at least one")
        self.alpha = _finite(alpha, field="alpha")
        self.minimum_effective_audits = _positive(
            minimum_effective_audits,
            field="minimum_effective_audits",
        )
        self.confidence_sequence_name = str(confidence_sequence)
        if self.confidence_sequence_name not in CONFIDENCE_SEQUENCES:
            raise ValueError(
                f"unsupported confidence sequence: {self.confidence_sequence_name}"
            )
        self.confidence_sequence = ClippedFreedmanConfidenceSequence(
            alpha=self.alpha,
            event_bound=self.event_bound,
            weight_clip=self.weight_clip,
        )
        self.records: list[EventAuditRecord] = []
        self._decision_ids: set[str] = set()

    def add_decision(
        self,
        decision: AuditDecision,
        *,
        outcome: float | None = None,
        prediction: float = 0.0,
    ) -> EventAuditRecord:
        if decision.opportunity.axis != self.axis:
            raise ValueError(
                f"audit axis {decision.opportunity.axis} does not match {self.axis}"
            )
        return self.add_observation(
            audit_decision_id=decision.decision_id,
            selected=decision.selected,
            propensity=decision.propensity,
            outcome=outcome,
            prediction=prediction,
        )

    def add_observation(
        self,
        *,
        audit_decision_id: str,
        selected: bool,
        propensity: Decimal | float | int | str,
        outcome: float | None = None,
        prediction: float = 0.0,
    ) -> EventAuditRecord:
        decision_id = _validated_sha256(
            audit_decision_id,
            field="audit_decision_id",
        )
        if decision_id in self._decision_ids:
            raise ValueError(f"duplicate audit decision: {decision_id}")
        probability = _propensity(propensity)
        inverse_probability = Decimal(1) / probability
        if inverse_probability > Decimal(str(self.weight_clip)):
            raise ValueError(
                "inverse propensity exceeds the registered weight clip"
            )
        is_selected = bool(selected)
        if probability == 1 and not is_selected:
            raise ValueError("probability-one audit cannot be unselected")
        predicted = _finite(prediction, field="prediction")
        if not 0 <= predicted <= self.event_bound:
            raise ValueError("prediction must stay within the registered event bound")
        if self.estimator == "horvitz_thompson" and predicted != 0.0:
            raise ValueError(
                "horvitz_thompson requires prediction=0; use doubly_robust "
                "for predictable nonzero predictions"
            )
        if is_selected:
            if outcome is None:
                raise ValueError("selected audit requires a full-fidelity outcome")
            observed = _finite(outcome, field="outcome")
            if not 0 <= observed <= self.event_bound:
                raise ValueError("outcome must stay within the registered event bound")
        else:
            if outcome is not None:
                raise ValueError(
                    "unselected audit cannot reveal a full-fidelity outcome"
                )
            observed = None

        p = float(probability)
        if self.estimator == "horvitz_thompson":
            contribution = (float(observed) / p) if is_selected else 0.0
        else:
            contribution = predicted
            if is_selected:
                contribution += (float(observed) - predicted) / p
        variance_bound = self.confidence_sequence.variance_increment(
            probability,
            prediction=(
                predicted
                if self.confidence_sequence_name
                == ASYMMETRIC_PARTIAL_ID_CONFIDENCE_SEQUENCE
                else None
            ),
        )
        inclusion_weight = (1.0 / p) if is_selected else 0.0
        record = EventAuditRecord(
            index=len(self.records),
            audit_decision_id=decision_id,
            axis=self.axis,
            selected=is_selected,
            propensity=probability,
            outcome=observed,
            prediction=predicted,
            contribution=contribution,
            predictable_variance_bound=variance_bound,
            inclusion_weight=inclusion_weight,
        )
        self.records.append(record)
        self._decision_ids.add(decision_id)
        return record

    def snapshot(self) -> EventEstimateSnapshot:
        count = len(self.records)
        selected = [record for record in self.records if record.selected]
        selected_weights = [record.inclusion_weight for record in selected]
        weight_sum = math.fsum(selected_weights)
        weight_square_sum = math.fsum(weight**2 for weight in selected_weights)
        effective_sample_size = (
            weight_sum**2 / weight_square_sum if weight_square_sum > 0 else 0.0
        )
        estimated_total = math.fsum(record.contribution for record in self.records)
        variance = math.fsum(
            record.predictable_variance_bound for record in self.records
        )
        selected_outcome_total = math.fsum(
            float(record.outcome)
            for record in selected
            if record.outcome is not None
        )
        if count:
            identification_lower = selected_outcome_total / count
            identification_upper = (
                selected_outcome_total
                + (count - len(selected)) * self.event_bound
            ) / count
        else:
            identification_lower = 0.0
            identification_upper = self.event_bound
        identification_interval_applied = bool(
            self.confidence_sequence_name
            == ASYMMETRIC_PARTIAL_ID_CONFIDENCE_SEQUENCE
        )
        confidence_intersection_fallback = False
        if identification_interval_applied:
            increment_bounds = [
                self.confidence_sequence.one_sided_increment_bounds(
                    record.propensity,
                    prediction=record.prediction,
                )
                for record in self.records
            ]
            lower_increment_bound = max(
                (bounds[0] for bounds in increment_bounds),
                default=0.0,
            )
            upper_increment_bound = max(
                (bounds[1] for bounds in increment_bounds),
                default=0.0,
            )
            lower_radius = self.confidence_sequence.asymmetric_radius(
                variance,
                increment_bound=lower_increment_bound,
            )
            upper_radius = self.confidence_sequence.asymmetric_radius(
                variance,
                increment_bound=upper_increment_bound,
            )
        else:
            lower_increment_bound = (
                self.confidence_sequence.martingale_increment_bound
            )
            upper_increment_bound = lower_increment_bound
            lower_radius = self.confidence_sequence.radius(variance)
            upper_radius = lower_radius
        if count:
            mean_raw = estimated_total / count
            lower = max(0.0, (estimated_total - lower_radius) / count)
            upper = min(
                self.event_bound,
                (estimated_total + upper_radius) / count,
            )
            if identification_interval_applied:
                intersected_lower = max(lower, identification_lower)
                intersected_upper = min(upper, identification_upper)
                if intersected_lower <= intersected_upper + 1e-15:
                    lower = min(intersected_lower, intersected_upper)
                    upper = max(intersected_lower, intersected_upper)
                else:
                    lower = identification_lower
                    upper = identification_upper
                    confidence_intersection_fallback = True
            bounded = min(self.event_bound, max(0.0, mean_raw))
        else:
            mean_raw = 0.0
            lower = 0.0
            upper = self.event_bound
            bounded = 0.0
        return EventEstimateSnapshot(
            manifest_sha256=self.manifest_sha256,
            axis=self.axis,
            metric=self.metric,
            estimator=self.estimator,
            opportunity_count=count,
            selected_audit_count=len(selected),
            expected_audit_count=math.fsum(
                float(record.propensity) for record in self.records
            ),
            estimated_total_raw=estimated_total,
            estimated_mean_raw=mean_raw,
            estimated_mean_bounded=bounded,
            confidence_lower=lower,
            confidence_upper=upper,
            confidence_radius_total=max(lower_radius, upper_radius),
            confidence_lower_radius_total=lower_radius,
            confidence_upper_radius_total=upper_radius,
            predictable_variance_bound=variance,
            martingale_increment_bound=(
                self.confidence_sequence.martingale_increment_bound
            ),
            confidence_lower_increment_bound=lower_increment_bound,
            confidence_upper_increment_bound=upper_increment_bound,
            identification_lower=identification_lower,
            identification_upper=identification_upper,
            identification_interval_applied=identification_interval_applied,
            confidence_intersection_fallback=confidence_intersection_fallback,
            selected_weight_sum=weight_sum,
            selected_weight_square_sum=weight_square_sum,
            effective_sample_size=effective_sample_size,
            minimum_effective_audits=self.minimum_effective_audits,
            confidence_ready=effective_sample_size >= self.minimum_effective_audits,
            maximum_inverse_propensity=max(
                (1.0 / float(record.propensity) for record in self.records),
                default=0.0,
            ),
            event_bound=self.event_bound,
            weight_clip=self.weight_clip,
            alpha=self.alpha,
            confidence_sequence=self.confidence_sequence_name,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "rlcmf-event-accumulator-v1",
            "manifest_sha256": self.manifest_sha256,
            "axis": self.axis,
            "metric": self.metric,
            "estimator": self.estimator,
            "event_bound": self.event_bound,
            "weight_clip": self.weight_clip,
            "alpha": self.alpha,
            "minimum_effective_audits": self.minimum_effective_audits,
            "confidence_sequence": self.confidence_sequence_name,
            "records": [record.to_dict() for record in self.records],
            "snapshot": self.snapshot().to_dict(),
        }
