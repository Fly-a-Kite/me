from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field, replace
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable, Mapping

from datadiff.risk_limiting_audit import (
    AUDIT_AXES,
    AuditAxisPolicy,
    _probability_text,
    _validated_sha256,
)


SAFETY_STATES = (
    "calibration_full",
    "adaptive_safe",
    "caution",
    "full_rollback",
)
MANDATORY_ROLLBACK_TRIGGERS = {
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


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def _rate(value: float | int, *, field: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a finite rate") from exc
    if not math.isfinite(parsed) or not 0 <= parsed <= 1:
        raise ValueError(f"{field} must be in [0, 1]")
    return parsed


def _optional_rate(value: float | int | None, *, field: str) -> float | None:
    return None if value is None else _rate(value, field=field)


def _nonnegative(value: float | int, *, field: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be finite and non-negative") from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError(f"{field} must be finite and non-negative")
    return parsed


def _probability(value: Decimal | float | int | str) -> Decimal:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError("safety propensity must be finite") from exc
    if not parsed.is_finite() or parsed <= 0 or parsed > 1:
        raise ValueError("safety propensity must satisfy 0 < p <= 1")
    return parsed


@dataclass(frozen=True, slots=True)
class AxisSafetyThresholds:
    axis: str
    maximum_event_miss_rate: float
    minimum_event_recall: float
    maximum_misclassification_rate: float
    caution_margin: float
    maximum_uncaptured_full_event_rate: float = 0.0

    def __post_init__(self) -> None:
        if self.axis not in AUDIT_AXES:
            raise ValueError(f"unknown safety axis: {self.axis}")
        object.__setattr__(
            self,
            "maximum_event_miss_rate",
            _rate(
                self.maximum_event_miss_rate,
                field=f"{self.axis}.maximum_event_miss_rate",
            ),
        )
        object.__setattr__(
            self,
            "maximum_uncaptured_full_event_rate",
            _rate(
                self.maximum_uncaptured_full_event_rate,
                field=f"{self.axis}.maximum_uncaptured_full_event_rate",
            ),
        )
        object.__setattr__(
            self,
            "minimum_event_recall",
            _rate(
                self.minimum_event_recall,
                field=f"{self.axis}.minimum_event_recall",
            ),
        )
        object.__setattr__(
            self,
            "maximum_misclassification_rate",
            _rate(
                self.maximum_misclassification_rate,
                field=f"{self.axis}.maximum_misclassification_rate",
            ),
        )
        margin = _rate(self.caution_margin, field=f"{self.axis}.caution_margin")
        if margin == 1:
            raise ValueError("caution_margin must be smaller than one")
        object.__setattr__(self, "caution_margin", margin)

    @classmethod
    def from_payload(
        cls,
        axis: str,
        payload: Mapping[str, Any],
    ) -> "AxisSafetyThresholds":
        return cls(
            axis=str(axis),
            maximum_event_miss_rate=float(payload["maximum_event_miss_rate"]),
            minimum_event_recall=float(payload["minimum_event_recall"]),
            maximum_misclassification_rate=float(
                payload["maximum_misclassification_rate"]
            ),
            caution_margin=float(payload["caution_margin"]),
            maximum_uncaptured_full_event_rate=float(
                payload["maximum_uncaptured_full_event_rate"]
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "axis": self.axis,
            "maximum_event_miss_rate": self.maximum_event_miss_rate,
            "minimum_event_recall": self.minimum_event_recall,
            "maximum_misclassification_rate": self.maximum_misclassification_rate,
            "caution_margin": self.caution_margin,
            "maximum_uncaptured_full_event_rate": (
                self.maximum_uncaptured_full_event_rate
            ),
        }


@dataclass(frozen=True, slots=True)
class AxisSafetyEvidence:
    axis: str
    effective_sample_size: float
    miss_rate_upper: float | None
    event_recall_lower: float | None
    misclassification_rate_upper: float | None
    evidence_ids: tuple[str, ...] = ()
    full_event_rate_upper: float | None = None
    miss_rate_lower: float | None = None
    event_recall_upper: float | None = None
    misclassification_rate_lower: float | None = None

    def __post_init__(self) -> None:
        if self.axis not in AUDIT_AXES:
            raise ValueError(f"unknown safety-evidence axis: {self.axis}")
        object.__setattr__(
            self,
            "effective_sample_size",
            _nonnegative(
                self.effective_sample_size,
                field=f"{self.axis}.effective_sample_size",
            ),
        )
        for field in (
            "miss_rate_lower",
            "miss_rate_upper",
            "event_recall_lower",
            "event_recall_upper",
            "misclassification_rate_lower",
            "misclassification_rate_upper",
            "full_event_rate_upper",
        ):
            object.__setattr__(
                self,
                field,
                _optional_rate(
                    getattr(self, field),
                    field=f"{self.axis}.{field}",
                ),
            )
        for lower_field, upper_field in (
            ("miss_rate_lower", "miss_rate_upper"),
            ("event_recall_lower", "event_recall_upper"),
            (
                "misclassification_rate_lower",
                "misclassification_rate_upper",
            ),
        ):
            lower = getattr(self, lower_field)
            upper = getattr(self, upper_field)
            if lower is not None and upper is not None and lower > upper:
                raise ValueError(
                    f"{self.axis}.{lower_field} cannot exceed {upper_field}"
                )
        normalized_ids = tuple(
            _validated_sha256(value, field=f"{self.axis}.evidence_id")
            for value in self.evidence_ids
        )
        if len(set(normalized_ids)) != len(normalized_ids):
            raise ValueError("duplicate safety evidence ids are not allowed")
        object.__setattr__(self, "evidence_ids", normalized_ids)

    @property
    def complete(self) -> bool:
        return all(
            value is not None
            for value in (
                self.miss_rate_upper,
                self.event_recall_lower,
                self.misclassification_rate_upper,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "axis": self.axis,
            "effective_sample_size": self.effective_sample_size,
            "miss_rate_lower": self.miss_rate_lower,
            "miss_rate_upper": self.miss_rate_upper,
            "event_recall_lower": self.event_recall_lower,
            "event_recall_upper": self.event_recall_upper,
            "misclassification_rate_lower": self.misclassification_rate_lower,
            "misclassification_rate_upper": self.misclassification_rate_upper,
            "evidence_ids": list(self.evidence_ids),
            "full_event_rate_upper": self.full_event_rate_upper,
        }


@dataclass(frozen=True, slots=True)
class HorizonAuditPolicy:
    total_horizon_cases: int
    minimum_low_path_cases: int
    debt_reserve_cases: int
    target_confidence_width: float
    full_audit_deadline_remaining_cases: int

    def __post_init__(self) -> None:
        total = int(self.total_horizon_cases)
        minimum_low = int(self.minimum_low_path_cases)
        debt_reserve = int(self.debt_reserve_cases)
        deadline = int(self.full_audit_deadline_remaining_cases)
        width = _rate(
            self.target_confidence_width,
            field="horizon.target_confidence_width",
        )
        if total <= 0:
            raise ValueError("horizon total_horizon_cases must be positive")
        if minimum_low < 0 or minimum_low >= total:
            raise ValueError("horizon minimum_low_path_cases is outside the horizon")
        if debt_reserve < 0 or debt_reserve >= total:
            raise ValueError("horizon debt_reserve_cases is outside the horizon")
        if deadline < 0 or deadline >= total:
            raise ValueError(
                "horizon full_audit_deadline_remaining_cases is outside the horizon"
            )
        if width <= 0:
            raise ValueError("horizon target_confidence_width must be positive")
        object.__setattr__(self, "total_horizon_cases", total)
        object.__setattr__(self, "minimum_low_path_cases", minimum_low)
        object.__setattr__(self, "debt_reserve_cases", debt_reserve)
        object.__setattr__(self, "target_confidence_width", width)
        object.__setattr__(
            self,
            "full_audit_deadline_remaining_cases",
            deadline,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "rlcmf-horizon-audit-policy-v1",
            "total_horizon_cases": self.total_horizon_cases,
            "minimum_low_path_cases": self.minimum_low_path_cases,
            "debt_reserve_cases": self.debt_reserve_cases,
            "target_confidence_width": self.target_confidence_width,
            "full_audit_deadline_remaining_cases": (
                self.full_audit_deadline_remaining_cases
            ),
            "allocation_rule": "minimum-required-ladder-with-low-path-reserve-v1",
        }


@dataclass(frozen=True, slots=True)
class SafetyDecision:
    manifest_sha256: str
    decision_id: str
    previous_decision_id: str
    serial: int
    completed_cases: int
    decision_epoch: int
    previous_state: str
    state: str
    reasons: tuple[str, ...]
    caution_axes: tuple[str, ...]
    calibration_complete: bool
    hard_rollback: bool
    audit_propensities: tuple[tuple[str, Decimal], ...]
    case_counts_by_seed: tuple[tuple[int, int], ...]
    evidence: tuple[AxisSafetyEvidence, ...]
    non_reproduced_candidate_cases: int
    coverage_debt_violation: bool
    confirmed_misclassification_violation: bool
    manifest_integrity: bool
    propensity_integrity: bool
    ledger_integrity: bool
    model_integrity: bool
    horizon_snapshot: Mapping[str, Any] = field(default_factory=dict)

    def propensity_for(self, axis: str) -> Decimal:
        try:
            return dict(self.audit_propensities)[axis]
        except KeyError as exc:
            raise KeyError(f"no safety propensity for axis: {axis}") from exc

    @property
    def full_fidelity(self) -> bool:
        return all(value == 1 for _, value in self.audit_propensities)

    def _digest_payload(self) -> dict[str, Any]:
        return {
            "schema_version": "rlcmf-safety-decision-v1",
            "manifest_sha256": self.manifest_sha256,
            "previous_decision_id": self.previous_decision_id,
            "serial": self.serial,
            "completed_cases": self.completed_cases,
            "decision_epoch": self.decision_epoch,
            "previous_state": self.previous_state,
            "state": self.state,
            "reasons": list(self.reasons),
            "caution_axes": list(self.caution_axes),
            "calibration_complete": self.calibration_complete,
            "hard_rollback": self.hard_rollback,
            "audit_propensities_exact": {
                axis: _probability_text(value)
                for axis, value in self.audit_propensities
            },
            "case_counts_by_seed": {
                str(seed): count for seed, count in self.case_counts_by_seed
            },
            "evidence": [item.to_dict() for item in self.evidence],
            "non_reproduced_candidate_cases": self.non_reproduced_candidate_cases,
            "coverage_debt_violation": self.coverage_debt_violation,
            "confirmed_misclassification_violation": (
                self.confirmed_misclassification_violation
            ),
            "manifest_integrity": self.manifest_integrity,
            "propensity_integrity": self.propensity_integrity,
            "ledger_integrity": self.ledger_integrity,
            "model_integrity": self.model_integrity,
            "horizon_snapshot": dict(self.horizon_snapshot),
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._digest_payload()
        payload["decision_id"] = self.decision_id
        payload["audit_propensities"] = {
            axis: float(value) for axis, value in self.audit_propensities
        }
        payload["full_fidelity"] = self.full_fidelity
        return payload

    def verify_digest(self) -> bool:
        return hashlib.sha256(_canonical_json_bytes(self._digest_payload())).hexdigest() == (
            self.decision_id
        )


class RiskLimitingSafetyController:
    """Fail-closed RLCMF controller with preregistered adaptation epochs."""

    def __init__(
        self,
        *,
        manifest_sha256: str,
        seeds: Iterable[int],
        audit_policies: Iterable[AuditAxisPolicy],
        calibration_cases_per_seed: int,
        minimum_effective_audits: float,
        decision_epoch_cases: int,
        propensity_ladder: Iterable[Decimal | float | int | str],
        axis_thresholds: Iterable[AxisSafetyThresholds],
        maximum_non_reproduced_candidate_cases: int,
        rollback_triggers: Iterable[str],
        recovery_enabled: bool = False,
        horizon_policy: HorizonAuditPolicy | None = None,
    ) -> None:
        self.manifest_sha256 = _validated_sha256(
            manifest_sha256,
            field="manifest_sha256",
        )
        seed_items = tuple(int(seed) for seed in seeds)
        if not seed_items or any(seed < 0 for seed in seed_items):
            raise ValueError("registered seeds must be non-empty and non-negative")
        if len(set(seed_items)) != len(seed_items):
            raise ValueError("registered seeds must be unique")
        self.seeds = tuple(sorted(seed_items))
        policy_items = tuple(audit_policies)
        self.audit_policies = {policy.axis: policy for policy in policy_items}
        if not self.audit_policies or len(self.audit_policies) != len(policy_items):
            raise ValueError("audit policies must be non-empty and unique by axis")
        threshold_items = tuple(axis_thresholds)
        self.axis_thresholds = {item.axis: item for item in threshold_items}
        if set(self.axis_thresholds) != set(self.audit_policies):
            raise ValueError("axis safety thresholds must cover every audit policy")
        self.calibration_cases_per_seed = int(calibration_cases_per_seed)
        if self.calibration_cases_per_seed <= 0:
            raise ValueError("calibration_cases_per_seed must be positive")
        self.minimum_effective_audits = _nonnegative(
            minimum_effective_audits,
            field="minimum_effective_audits",
        )
        if self.minimum_effective_audits < 1:
            raise ValueError("minimum_effective_audits must be at least one")
        self.decision_epoch_cases = int(decision_epoch_cases)
        if self.decision_epoch_cases <= 0:
            raise ValueError("decision_epoch_cases must be positive")
        ladder = tuple(_probability(value) for value in propensity_ladder)
        if not ladder or tuple(sorted(set(ladder))) != ladder or ladder[-1] != 1:
            raise ValueError(
                "propensity_ladder must be strictly increasing and end at one"
            )
        self.propensity_ladder = ladder
        for policy in self.audit_policies.values():
            for value in ladder:
                policy.validate_requested(value)
            if policy.initial not in ladder:
                raise ValueError(
                    f"initial propensity for {policy.axis} is absent from safety ladder"
                )
        self.maximum_non_reproduced_candidate_cases = int(
            maximum_non_reproduced_candidate_cases
        )
        if self.maximum_non_reproduced_candidate_cases < 0:
            raise ValueError(
                "maximum_non_reproduced_candidate_cases cannot be negative"
            )
        self.rollback_triggers = frozenset(str(value) for value in rollback_triggers)
        missing = sorted(MANDATORY_ROLLBACK_TRIGGERS - self.rollback_triggers)
        if missing:
            raise ValueError(f"missing mandatory rollback triggers: {missing}")
        if recovery_enabled:
            raise ValueError("RLCMF v1 supports terminal full rollback only")
        self.recovery_enabled = False
        self.horizon_policy = horizon_policy

        self.state = "calibration_full"
        self._adaptive_propensities = {
            axis: policy.initial for axis, policy in self.audit_policies.items()
        }
        self._last_adaptation_epoch = -1
        self._last_completed_cases = -1
        self._last_case_counts = {seed: 0 for seed in self.seeds}
        self._last_non_reproduced = 0
        self.decisions: list[SafetyDecision] = []

    @classmethod
    def from_manifest_contract(
        cls,
        *,
        manifest_sha256: str,
        seeds: Iterable[int],
        audit_policies: Iterable[AuditAxisPolicy],
        calibration_cases_per_seed: int,
        minimum_effective_audits: float,
        decision_epoch_cases: int,
        propensity_ladder: Iterable[Decimal | float | int | str],
        axis_thresholds: Mapping[str, Any],
        maximum_non_reproduced_candidate_cases: int,
        rollback_triggers: Iterable[str],
        recovery_enabled: bool,
        horizon_payload: Mapping[str, Any] | None = None,
        total_horizon_cases: int | None = None,
    ) -> "RiskLimitingSafetyController":
        parsed_thresholds = []
        for axis, payload in axis_thresholds.items():
            if not isinstance(payload, Mapping):
                raise ValueError(f"safety thresholds for {axis} must be an object")
            parsed_thresholds.append(
                AxisSafetyThresholds.from_payload(str(axis), payload)
            )
        horizon_policy = None
        if horizon_payload is not None:
            if total_horizon_cases is None:
                raise ValueError("horizon policy requires total_horizon_cases")
            horizon_policy = HorizonAuditPolicy(
                total_horizon_cases=int(total_horizon_cases),
                minimum_low_path_cases=int(
                    horizon_payload["minimum_low_path_cases"]
                ),
                debt_reserve_cases=int(horizon_payload["debt_reserve_cases"]),
                target_confidence_width=float(
                    horizon_payload["target_confidence_width"]
                ),
                full_audit_deadline_remaining_cases=int(
                    horizon_payload["full_audit_deadline_remaining_cases"]
                ),
            )
        return cls(
            manifest_sha256=manifest_sha256,
            seeds=seeds,
            audit_policies=audit_policies,
            calibration_cases_per_seed=calibration_cases_per_seed,
            minimum_effective_audits=minimum_effective_audits,
            decision_epoch_cases=decision_epoch_cases,
            propensity_ladder=propensity_ladder,
            axis_thresholds=parsed_thresholds,
            maximum_non_reproduced_candidate_cases=(
                maximum_non_reproduced_candidate_cases
            ),
            rollback_triggers=rollback_triggers,
            recovery_enabled=recovery_enabled,
            horizon_policy=horizon_policy,
        )

    def _next_propensity(self, current: Decimal) -> Decimal:
        for value in self.propensity_ladder:
            if value > current:
                return value
        return Decimal(1)

    def _ladder_ceiling(self, value: float | Decimal) -> Decimal:
        target = Decimal(str(value))
        for probability in self.propensity_ladder:
            if probability >= target:
                return probability
        return Decimal(1)

    @staticmethod
    def _confidence_width(evidence: AxisSafetyEvidence | None) -> float | None:
        if evidence is None:
            return None
        widths = []
        for lower, upper in (
            (evidence.miss_rate_lower, evidence.miss_rate_upper),
            (evidence.event_recall_lower, evidence.event_recall_upper),
            (
                evidence.misclassification_rate_lower,
                evidence.misclassification_rate_upper,
            ),
        ):
            if lower is not None and upper is not None:
                widths.append(max(0.0, float(upper) - float(lower)))
        return max(widths) if widths else None

    def _horizon_target(
        self,
        *,
        axis: str,
        evidence: AxisSafetyEvidence | None,
        completed_cases: int,
    ) -> tuple[Decimal, dict[str, Any]]:
        policy = self.horizon_policy
        if policy is None:
            return self._next_propensity(self._adaptive_propensities[axis]), {}
        remaining = max(0, policy.total_horizon_cases - int(completed_cases))
        usable = max(1, remaining - policy.debt_reserve_cases)
        effective_sample_size = (
            float(evidence.effective_sample_size) if evidence is not None else 0.0
        )
        needed_for_minimum = max(
            0.0,
            self.minimum_effective_audits - effective_sample_size,
        )
        width = self._confidence_width(evidence)
        needed_for_width = 0.0
        if width is not None and width > policy.target_confidence_width:
            target_effective_sample_size = effective_sample_size * (
                width / policy.target_confidence_width
            ) ** 2
            needed_for_width = max(
                0.0,
                target_effective_sample_size - effective_sample_size,
            )
        required_future_audits = max(needed_for_minimum, needed_for_width)
        required_fraction = min(1.0, required_future_audits / usable)
        current = self._adaptive_propensities[axis]
        target = self._ladder_ceiling(max(float(current), required_fraction))
        full_deadline = remaining <= (
            policy.full_audit_deadline_remaining_cases
            + policy.debt_reserve_cases
        )
        low_path_reserve_active = remaining > (
            policy.minimum_low_path_cases + policy.debt_reserve_cases
        )
        if target == 1 and low_path_reserve_active and not full_deadline:
            target = self.propensity_ladder[-2]
        return target, {
            "axis": axis,
            "remaining_cases": remaining,
            "usable_cases_after_debt_reserve": usable,
            "effective_sample_size": effective_sample_size,
            "confidence_width": width,
            "target_confidence_width": policy.target_confidence_width,
            "needed_audits_for_minimum_ess": needed_for_minimum,
            "needed_audits_for_confidence_width": needed_for_width,
            "required_future_audits": required_future_audits,
            "required_future_audit_fraction": required_fraction,
            "target_propensity_exact": _probability_text(target),
            "full_audit_deadline": full_deadline,
            "low_path_reserve_active": low_path_reserve_active,
        }

    def _normalize_case_counts(
        self,
        counts: Mapping[int, int],
        *,
        completed_cases: int,
    ) -> tuple[tuple[int, int], ...]:
        normalized = {int(seed): int(count) for seed, count in counts.items()}
        if set(normalized) != set(self.seeds):
            raise ValueError("case counts must cover exactly the registered seed list")
        if any(count < 0 for count in normalized.values()):
            raise ValueError("case counts cannot be negative")
        if sum(normalized.values()) != completed_cases:
            raise ValueError("completed_cases must equal the registered seed case total")
        for seed, count in normalized.items():
            if count < self._last_case_counts[seed]:
                raise ValueError("per-seed case counts cannot decrease")
        return tuple(sorted(normalized.items()))

    def evaluate(
        self,
        *,
        completed_cases: int,
        case_counts_by_seed: Mapping[int, int],
        evidence: Iterable[AxisSafetyEvidence],
        non_reproduced_candidate_cases: int = 0,
        coverage_debt_violation: bool = False,
        confirmed_misclassification_violation: bool = False,
        manifest_integrity: bool = True,
        propensity_integrity: bool = True,
        ledger_integrity: bool = True,
        model_integrity: bool = True,
    ) -> SafetyDecision:
        completed = int(completed_cases)
        if completed < 0 or completed <= self._last_completed_cases:
            raise ValueError("completed_cases must increase on every safety evaluation")
        case_counts = self._normalize_case_counts(
            case_counts_by_seed,
            completed_cases=completed,
        )
        non_reproduced = int(non_reproduced_candidate_cases)
        if non_reproduced < self._last_non_reproduced:
            raise ValueError("non-reproduced candidate count cannot decrease")
        evidence_items = tuple(sorted(evidence, key=lambda item: item.axis))
        evidence_by_axis = {item.axis: item for item in evidence_items}
        if len(evidence_by_axis) != len(evidence_items):
            raise ValueError("duplicate axis safety evidence is not allowed")
        unknown_evidence = sorted(set(evidence_by_axis) - set(self.audit_policies))
        if unknown_evidence:
            raise ValueError(f"evidence contains unregistered axes: {unknown_evidence}")

        previous_state = self.state
        decision_epoch = completed // self.decision_epoch_cases
        calibration_complete = all(
            count >= self.calibration_cases_per_seed for _, count in case_counts
        )
        hard_reasons: list[str] = []
        if not manifest_integrity:
            hard_reasons.append("manifest_integrity")
        if not propensity_integrity:
            hard_reasons.append("propensity_integrity")
        if not ledger_integrity:
            hard_reasons.append("ledger_integrity")
        if not model_integrity:
            hard_reasons.append("model_integrity")
        if coverage_debt_violation:
            hard_reasons.append("coverage_debt_cap")
        if confirmed_misclassification_violation:
            hard_reasons.append("misclassification_bound:confirmed_violation")
        if non_reproduced > self.maximum_non_reproduced_candidate_cases:
            hard_reasons.append("non_reproduced_candidate")

        caution_reasons: list[str] = []
        caution_axes: set[str] = set()
        zero_capture_safe_axes: set[str] = set()
        horizon_axis_snapshots: dict[str, Any] = {}
        for axis in sorted(self.audit_policies):
            axis_evidence = evidence_by_axis.get(axis)
            if axis_evidence is None:
                caution_axes.add(axis)
                caution_reasons.append(f"{axis}:missing_safety_evidence")
                continue
            thresholds = self.axis_thresholds[axis]
            if (
                axis_evidence.miss_rate_lower is not None
                and axis_evidence.miss_rate_lower
                > thresholds.maximum_event_miss_rate
            ):
                hard_reasons.append(
                    f"miss_rate_bound:{axis}:confirmed_violation"
                )
            if (
                axis_evidence.event_recall_upper is not None
                and axis_evidence.event_recall_upper
                < thresholds.minimum_event_recall
            ):
                hard_reasons.append(f"recall_bound:{axis}:confirmed_violation")
            if (
                axis_evidence.misclassification_rate_lower is not None
                and axis_evidence.misclassification_rate_lower
                > thresholds.maximum_misclassification_rate
            ):
                hard_reasons.append(
                    f"misclassification_bound:{axis}:confirmed_violation"
                )
            if axis_evidence.effective_sample_size < self.minimum_effective_audits:
                caution_axes.add(axis)
                caution_reasons.append(f"{axis}:insufficient_effective_audits")
                continue
            if (
                axis_evidence.miss_rate_upper is None
                or axis_evidence.misclassification_rate_upper is None
            ):
                caution_axes.add(axis)
                caution_reasons.append(f"{axis}:incomplete_safety_bounds")
                continue
            if axis_evidence.miss_rate_upper > thresholds.maximum_event_miss_rate:
                caution_axes.add(axis)
                caution_reasons.append(f"{axis}:miss_rate_uncertainty")
            if axis_evidence.event_recall_lower is None:
                if axis_evidence.full_event_rate_upper is None:
                    caution_axes.add(axis)
                    caution_reasons.append(f"{axis}:incomplete_capture_bounds")
                elif (
                    axis_evidence.full_event_rate_upper
                    > thresholds.maximum_uncaptured_full_event_rate
                ):
                    caution_axes.add(axis)
                    caution_reasons.append(f"{axis}:uncaptured_event_hazard")
                else:
                    zero_capture_safe_axes.add(axis)
                    if (
                        thresholds.maximum_uncaptured_full_event_rate
                        - axis_evidence.full_event_rate_upper
                        <= thresholds.caution_margin
                    ):
                        caution_axes.add(axis)
                        caution_reasons.append(
                            f"{axis}:uncaptured_event_hazard_slack"
                        )
            elif axis_evidence.event_recall_lower < thresholds.minimum_event_recall:
                caution_axes.add(axis)
                caution_reasons.append(f"{axis}:recall_uncertainty")
            if (
                axis_evidence.misclassification_rate_upper
                > thresholds.maximum_misclassification_rate
            ):
                caution_axes.add(axis)
                caution_reasons.append(f"{axis}:misclassification_uncertainty")
            if (
                axis_evidence.miss_rate_upper
                <= thresholds.maximum_event_miss_rate
                and
                thresholds.maximum_event_miss_rate
                - axis_evidence.miss_rate_upper
                <= thresholds.caution_margin
            ):
                caution_axes.add(axis)
                caution_reasons.append(f"{axis}:miss_rate_slack")
            if axis_evidence.event_recall_lower is not None and (
                axis_evidence.event_recall_lower
                >= thresholds.minimum_event_recall
                and axis_evidence.event_recall_lower
                - thresholds.minimum_event_recall
                <= thresholds.caution_margin
            ):
                caution_axes.add(axis)
                caution_reasons.append(f"{axis}:recall_slack")
            if (
                axis_evidence.misclassification_rate_upper
                <= thresholds.maximum_misclassification_rate
                and
                thresholds.maximum_misclassification_rate
                - axis_evidence.misclassification_rate_upper
                <= thresholds.caution_margin
            ):
                caution_axes.add(axis)
                caution_reasons.append(f"{axis}:misclassification_slack")

        if previous_state == "full_rollback":
            state = "full_rollback"
            reasons = tuple(sorted(set(hard_reasons + ["rollback_latched"])))
        elif hard_reasons:
            state = "full_rollback"
            reasons = tuple(sorted(set(hard_reasons)))
        elif not calibration_complete:
            state = "calibration_full"
            incomplete = [
                f"calibration_incomplete:{seed}"
                for seed, count in case_counts
                if count < self.calibration_cases_per_seed
            ]
            reasons = tuple(incomplete)
        elif caution_axes:
            state = "caution"
            reasons = tuple(sorted(set(caution_reasons)))
            axes_to_raise = set(caution_axes)
            axes_to_raise.add("joint")
            axes_to_raise.intersection_update(self.audit_policies)
            if decision_epoch > self._last_adaptation_epoch:
                targets = {}
                for axis in sorted(axes_to_raise):
                    raised, horizon_axis_snapshot = self._horizon_target(
                        axis=axis,
                        evidence=evidence_by_axis.get(axis),
                        completed_cases=completed,
                    )
                    targets[axis] = raised
                    if horizon_axis_snapshot:
                        horizon_axis_snapshots[axis] = horizon_axis_snapshot
                if self.horizon_policy is not None and "joint" in targets:
                    joint_target = targets["joint"]
                    for axis in self.audit_policies:
                        if axis == "joint":
                            continue
                        targets[axis] = max(
                            targets.get(axis, self._adaptive_propensities[axis]),
                            joint_target,
                        )
                        snapshot = horizon_axis_snapshots.setdefault(
                            axis,
                            {
                                "axis": axis,
                                "joint_nesting_only": True,
                            },
                        )
                        snapshot["joint_nesting_floor_exact"] = (
                            _probability_text(joint_target)
                        )
                        snapshot["target_propensity_exact"] = (
                            _probability_text(targets[axis])
                        )
                for axis, raised in sorted(targets.items()):
                    self.audit_policies[axis].validate_requested(raised)
                    self._adaptive_propensities[axis] = max(
                        self._adaptive_propensities[axis],
                        raised,
                    )
                self._last_adaptation_epoch = decision_epoch
        else:
            state = "adaptive_safe"
            reasons = tuple(
                ["all_registered_safety_bounds_pass"]
                + [
                    f"zero_capture_event_hazard_bound_pass:{axis}"
                    for axis in sorted(zero_capture_safe_axes)
                ]
            )

        self.state = state
        if state in {"calibration_full", "full_rollback"}:
            propensities = tuple(
                (axis, Decimal(1)) for axis in sorted(self.audit_policies)
            )
        else:
            propensities = tuple(sorted(self._adaptive_propensities.items()))
        previous_decision_id = (
            self.decisions[-1].decision_id if self.decisions else ""
        )
        decision_without_id = SafetyDecision(
            manifest_sha256=self.manifest_sha256,
            decision_id="0" * 64,
            previous_decision_id=previous_decision_id,
            serial=len(self.decisions),
            completed_cases=completed,
            decision_epoch=decision_epoch,
            previous_state=previous_state,
            state=state,
            reasons=reasons,
            caution_axes=tuple(sorted(caution_axes)),
            calibration_complete=calibration_complete,
            hard_rollback=state == "full_rollback",
            audit_propensities=propensities,
            case_counts_by_seed=case_counts,
            evidence=evidence_items,
            non_reproduced_candidate_cases=non_reproduced,
            coverage_debt_violation=bool(coverage_debt_violation),
            confirmed_misclassification_violation=bool(
                confirmed_misclassification_violation
            ),
            manifest_integrity=bool(manifest_integrity),
            propensity_integrity=bool(propensity_integrity),
            ledger_integrity=bool(ledger_integrity),
            model_integrity=bool(model_integrity),
            horizon_snapshot=(
                {
                    "schema_version": "rlcmf-horizon-safety-snapshot-v1",
                    "policy": self.horizon_policy.to_dict(),
                    "completed_cases": completed,
                    "remaining_cases": max(
                        0,
                        self.horizon_policy.total_horizon_cases - completed,
                    ),
                    "axes": dict(sorted(horizon_axis_snapshots.items())),
                }
                if self.horizon_policy is not None
                else {}
            ),
        )
        decision_id = hashlib.sha256(
            _canonical_json_bytes(decision_without_id._digest_payload())
        ).hexdigest()
        decision = replace(decision_without_id, decision_id=decision_id)
        self.decisions.append(decision)
        self._last_completed_cases = completed
        self._last_case_counts = dict(case_counts)
        self._last_non_reproduced = non_reproduced
        return decision

    def verify_log(self) -> bool:
        previous_id = ""
        for serial, decision in enumerate(self.decisions):
            if decision.serial != serial or decision.previous_decision_id != previous_id:
                return False
            if not decision.verify_digest():
                return False
            previous_id = decision.decision_id
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "rlcmf-safety-controller-v1",
            "manifest_sha256": self.manifest_sha256,
            "state": self.state,
            "registered_seeds": list(self.seeds),
            "calibration_cases_per_seed": self.calibration_cases_per_seed,
            "minimum_effective_audits": self.minimum_effective_audits,
            "decision_epoch_cases": self.decision_epoch_cases,
            "propensity_ladder_exact": [
                _probability_text(value) for value in self.propensity_ladder
            ],
            "axis_thresholds": {
                axis: thresholds.to_dict()
                for axis, thresholds in sorted(self.axis_thresholds.items())
            },
            "maximum_non_reproduced_candidate_cases": (
                self.maximum_non_reproduced_candidate_cases
            ),
            "recovery_enabled": self.recovery_enabled,
            "horizon_policy": (
                self.horizon_policy.to_dict()
                if self.horizon_policy is not None
                else None
            ),
            "decision_count": len(self.decisions),
            "decisions": [decision.to_dict() for decision in self.decisions],
            "log_integrity": self.verify_log(),
        }
