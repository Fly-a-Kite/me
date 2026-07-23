from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable, Mapping

from datadiff.risk_limiting_audit import (
    AUDIT_AXES,
    AuditAxisPolicy,
    _probability_text,
    _validated_sha256,
)
from datadiff.research_controls.rlcmf_shadow.rlcmf_safety import SafetyDecision


VOI_FEATURES = (
    "root_entropy_reduction",
    "unseen_root_hazard",
    "disagreement_diversity",
    "coverage_debt_pressure",
    "reproducibility_uncertainty",
    "saturation_penalty",
)


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def _finite(value: float | int, *, field: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be finite") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{field} must be finite")
    return parsed


def _unit(value: float | int, *, field: str) -> float:
    parsed = _finite(value, field=field)
    if not 0 <= parsed <= 1:
        raise ValueError(f"{field} must be in [0, 1]")
    return parsed


def _probability(value: Decimal | float | int | str) -> Decimal:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError("VOI propensity must be finite") from exc
    if not parsed.is_finite() or parsed <= 0 or parsed > 1:
        raise ValueError("VOI propensity must satisfy 0 < p <= 1")
    return parsed


@dataclass(frozen=True, slots=True)
class RootCauseVOIForecast:
    action_id: str
    axis: str
    history_sha256: str
    expected_cost_ms: float
    root_entropy_reduction: float
    unseen_root_hazard: float
    disagreement_diversity: float
    coverage_debt_pressure: float
    reproducibility_uncertainty: float
    saturation_penalty: float
    history_case_count: int = 0
    cost_metric: str = "process_cpu_ms"
    feature_source: str = "predictable_history_only"

    def __post_init__(self) -> None:
        if not str(self.action_id or "").strip():
            raise ValueError("VOI action_id cannot be empty")
        if self.axis not in AUDIT_AXES:
            raise ValueError(f"unknown VOI axis: {self.axis}")
        object.__setattr__(
            self,
            "history_sha256",
            _validated_sha256(self.history_sha256, field="history_sha256"),
        )
        cost = _finite(self.expected_cost_ms, field="expected_cost_ms")
        if cost <= 0:
            raise ValueError("expected_cost_ms must be positive")
        object.__setattr__(self, "expected_cost_ms", cost)
        history_case_count = int(self.history_case_count)
        if history_case_count < 0:
            raise ValueError("VOI history_case_count cannot be negative")
        if self.cost_metric != "process_cpu_ms":
            raise ValueError("VOI expected cost must use process_cpu_ms")
        if self.feature_source != "predictable_history_only":
            raise ValueError("VOI features must come from predictable history only")
        object.__setattr__(self, "history_case_count", history_case_count)
        for feature in VOI_FEATURES:
            object.__setattr__(
                self,
                feature,
                _unit(getattr(self, feature), field=feature),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "axis": self.axis,
            "history_sha256": self.history_sha256,
            "expected_cost_ms": self.expected_cost_ms,
            "history_case_count": self.history_case_count,
            "cost_metric": self.cost_metric,
            "feature_source": self.feature_source,
            **{feature: getattr(self, feature) for feature in VOI_FEATURES},
        }


@dataclass(frozen=True, slots=True)
class RootCauseVOIForecastSnapshot:
    snapshot_id: str
    manifest_sha256: str
    model_sha256: str
    history_sha256: str
    history_case_count: int
    forecasts: tuple[RootCauseVOIForecast, ...]

    @classmethod
    def create(
        cls,
        *,
        manifest_sha256: str,
        model_sha256: str,
        history_payload: Mapping[str, Any],
        history_case_count: int,
        forecasts: Iterable[RootCauseVOIForecast],
    ) -> "RootCauseVOIForecastSnapshot":
        manifest_digest = _validated_sha256(
            manifest_sha256,
            field="manifest_sha256",
        )
        model_digest = _validated_sha256(model_sha256, field="model_sha256")
        count = int(history_case_count)
        if count < 0:
            raise ValueError("VOI snapshot history_case_count cannot be negative")
        history_sha256 = hashlib.sha256(
            _canonical_json_bytes(history_payload)
        ).hexdigest()
        forecast_items = tuple(
            sorted(forecasts, key=lambda row: (row.axis, row.action_id))
        )
        if any(
            forecast.history_sha256 != history_sha256
            or forecast.history_case_count != count
            for forecast in forecast_items
        ):
            raise ValueError(
                "VOI forecasts must bind to the same predictable history snapshot"
            )
        material = {
            "schema_version": "rlcmf-voi-forecast-snapshot-v1",
            "manifest_sha256": manifest_digest,
            "model_sha256": model_digest,
            "history_sha256": history_sha256,
            "history_case_count": count,
            "forecasts": [forecast.to_dict() for forecast in forecast_items],
        }
        snapshot_id = hashlib.sha256(_canonical_json_bytes(material)).hexdigest()
        return cls(
            snapshot_id=snapshot_id,
            manifest_sha256=manifest_digest,
            model_sha256=model_digest,
            history_sha256=history_sha256,
            history_case_count=count,
            forecasts=forecast_items,
        )

    def verify_digest(self) -> bool:
        material = {
            "schema_version": "rlcmf-voi-forecast-snapshot-v1",
            "manifest_sha256": self.manifest_sha256,
            "model_sha256": self.model_sha256,
            "history_sha256": self.history_sha256,
            "history_case_count": self.history_case_count,
            "forecasts": [forecast.to_dict() for forecast in self.forecasts],
        }
        return hashlib.sha256(_canonical_json_bytes(material)).hexdigest() == (
            self.snapshot_id
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "rlcmf-voi-forecast-snapshot-v1",
            "snapshot_id": self.snapshot_id,
            "manifest_sha256": self.manifest_sha256,
            "model_sha256": self.model_sha256,
            "history_sha256": self.history_sha256,
            "history_case_count": self.history_case_count,
            "feature_source": "predictable_history_only",
            "cost_metric": "process_cpu_ms",
            "forecasts": [forecast.to_dict() for forecast in self.forecasts],
        }


@dataclass(frozen=True, slots=True)
class VOIAllocationPair:
    snapshot_id: str
    voi_decision: "VOIAllocationDecision"
    no_voi_control: "VOIAllocationDecision"

    def to_dict(self) -> dict[str, Any]:
        voi = dict(self.voi_decision.audit_propensities)
        control = dict(self.no_voi_control.audit_propensities)
        return {
            "schema_version": "rlcmf-voi-allocation-pair-v1",
            "snapshot_id": self.snapshot_id,
            "voi_decision": self.voi_decision.to_dict(),
            "no_voi_control": self.no_voi_control.to_dict(),
            "propensity_delta": {
                axis: float(voi[axis] - control[axis])
                for axis in sorted(control)
            },
            "raise_only": all(voi[axis] >= control[axis] for axis in control),
        }


@dataclass(frozen=True, slots=True)
class VOIRaiseThreshold:
    minimum_voi_per_cpu_ms: float
    ladder_steps: int

    def __post_init__(self) -> None:
        minimum = _finite(
            self.minimum_voi_per_cpu_ms,
            field="minimum_voi_per_cpu_ms",
        )
        if minimum < 0:
            raise ValueError("minimum_voi_per_cpu_ms cannot be negative")
        if int(self.ladder_steps) <= 0:
            raise ValueError("ladder_steps must be positive")
        object.__setattr__(self, "minimum_voi_per_cpu_ms", minimum)
        object.__setattr__(self, "ladder_steps", int(self.ladder_steps))

    def to_dict(self) -> dict[str, Any]:
        return {
            "minimum_voi_per_cpu_ms": self.minimum_voi_per_cpu_ms,
            "ladder_steps": self.ladder_steps,
        }


@dataclass(frozen=True, slots=True)
class VOIAllocationDecision:
    manifest_sha256: str
    allocation_id: str
    safety_decision_id: str
    allocation_epoch: int
    audit_propensities: tuple[tuple[str, Decimal], ...]
    selected_actions: tuple[tuple[str, str], ...]
    score_per_cpu_ms: tuple[tuple[str, float], ...]
    extra_expected_cost_ms: float
    maximum_extra_expected_cost_ms: float
    reasons: tuple[str, ...]
    forecasts: tuple[RootCauseVOIForecast, ...]
    model_sha256: str

    def propensity_for(self, axis: str) -> Decimal:
        try:
            return dict(self.audit_propensities)[axis]
        except KeyError as exc:
            raise KeyError(f"no VOI propensity for axis: {axis}") from exc

    def _digest_payload(self) -> dict[str, Any]:
        return {
            "schema_version": "rlcmf-voi-allocation-v1",
            "manifest_sha256": self.manifest_sha256,
            "safety_decision_id": self.safety_decision_id,
            "allocation_epoch": self.allocation_epoch,
            "audit_propensities_exact": {
                axis: _probability_text(value)
                for axis, value in self.audit_propensities
            },
            "selected_actions": dict(self.selected_actions),
            "score_per_cpu_ms": dict(self.score_per_cpu_ms),
            "extra_expected_cost_ms": self.extra_expected_cost_ms,
            "maximum_extra_expected_cost_ms": self.maximum_extra_expected_cost_ms,
            "reasons": list(self.reasons),
            "forecasts": [forecast.to_dict() for forecast in self.forecasts],
            "model_sha256": self.model_sha256,
        }

    def verify_digest(self) -> bool:
        return hashlib.sha256(_canonical_json_bytes(self._digest_payload())).hexdigest() == (
            self.allocation_id
        )

    def to_dict(self) -> dict[str, Any]:
        payload = self._digest_payload()
        payload["allocation_id"] = self.allocation_id
        payload["audit_propensities"] = {
            axis: float(value) for axis, value in self.audit_propensities
        }
        return payload


class RootCauseVOIScheduler:
    """Raise-only value-of-information allocation under a frozen CPU budget."""

    def __init__(
        self,
        *,
        manifest_sha256: str,
        model_sha256: str,
        audit_policies: Iterable[AuditAxisPolicy],
        propensity_ladder: Iterable[Decimal | float | int | str],
        feature_weights: Mapping[str, float],
        minimum_cost_ms: float,
        raise_thresholds: Iterable[VOIRaiseThreshold],
        maximum_extra_expected_cost_ms: float,
        raise_joint_with_component: bool,
    ) -> None:
        self.manifest_sha256 = _validated_sha256(
            manifest_sha256,
            field="manifest_sha256",
        )
        self.model_sha256 = _validated_sha256(
            model_sha256,
            field="model_sha256",
        )
        policy_items = tuple(audit_policies)
        self.audit_policies = {policy.axis: policy for policy in policy_items}
        if not self.audit_policies or len(self.audit_policies) != len(policy_items):
            raise ValueError("VOI audit policies must be non-empty and unique")
        ladder = tuple(_probability(value) for value in propensity_ladder)
        if not ladder or tuple(sorted(set(ladder))) != ladder or ladder[-1] != 1:
            raise ValueError("VOI propensity ladder must increase strictly to one")
        self.propensity_ladder = ladder
        for policy in self.audit_policies.values():
            for value in ladder:
                policy.validate_requested(value)
        if set(feature_weights) != set(VOI_FEATURES):
            raise ValueError("VOI feature weights must match the registered feature set")
        self.feature_weights = {
            feature: _finite(value, field=f"weight.{feature}")
            for feature, value in feature_weights.items()
        }
        if any(value < 0 for value in self.feature_weights.values()):
            raise ValueError("VOI feature weights cannot be negative")
        self.minimum_cost_ms = _finite(minimum_cost_ms, field="minimum_cost_ms")
        if self.minimum_cost_ms <= 0:
            raise ValueError("minimum_cost_ms must be positive")
        thresholds = tuple(raise_thresholds)
        if not thresholds:
            raise ValueError("at least one VOI raise threshold is required")
        scores = [row.minimum_voi_per_cpu_ms for row in thresholds]
        steps = [row.ladder_steps for row in thresholds]
        if scores != sorted(set(scores)) or steps != sorted(steps):
            raise ValueError("VOI thresholds and ladder steps must increase monotonically")
        self.raise_thresholds = thresholds
        self.maximum_extra_expected_cost_ms = _finite(
            maximum_extra_expected_cost_ms,
            field="maximum_extra_expected_cost_ms",
        )
        if self.maximum_extra_expected_cost_ms < 0:
            raise ValueError("maximum_extra_expected_cost_ms cannot be negative")
        self.raise_joint_with_component = bool(raise_joint_with_component)
        if self.raise_joint_with_component and "joint" not in self.audit_policies:
            raise ValueError("joint coupling requires a registered joint audit axis")

    @classmethod
    def from_manifest_contract(
        cls,
        *,
        manifest_sha256: str,
        audit_policies: Iterable[AuditAxisPolicy],
        propensity_ladder: Iterable[Decimal | float | int | str],
        payload: Mapping[str, Any],
    ) -> "RootCauseVOIScheduler":
        raw_thresholds = payload.get("raise_thresholds", [])
        if not isinstance(raw_thresholds, list):
            raise ValueError("voi.raise_thresholds must be a list")
        return cls(
            manifest_sha256=manifest_sha256,
            model_sha256=str(payload["model_sha256"]),
            audit_policies=audit_policies,
            propensity_ladder=propensity_ladder,
            feature_weights=dict(payload["weights"]),
            minimum_cost_ms=float(payload["minimum_cost_ms"]),
            raise_thresholds=[
                VOIRaiseThreshold(
                    minimum_voi_per_cpu_ms=float(
                        row["minimum_voi_per_cpu_ms"]
                    ),
                    ladder_steps=int(row["ladder_steps"]),
                )
                for row in raw_thresholds
                if isinstance(row, Mapping)
            ],
            maximum_extra_expected_cost_ms=float(
                payload["maximum_extra_expected_cost_ms_per_case"]
            ),
            raise_joint_with_component=bool(
                payload["raise_joint_with_component"]
            ),
        )

    def _value(self, forecast: RootCauseVOIForecast) -> float:
        positive = sum(
            self.feature_weights[feature] * getattr(forecast, feature)
            for feature in VOI_FEATURES
            if feature != "saturation_penalty"
        )
        penalty = (
            self.feature_weights["saturation_penalty"]
            * forecast.saturation_penalty
        )
        return max(0.0, positive - penalty)

    def _score(self, forecast: RootCauseVOIForecast) -> float:
        return self._value(forecast) / max(
            self.minimum_cost_ms,
            forecast.expected_cost_ms,
        )

    def _target_steps(self, score: float) -> int:
        steps = 0
        for threshold in self.raise_thresholds:
            if score >= threshold.minimum_voi_per_cpu_ms:
                steps = threshold.ladder_steps
        return steps

    def _next(self, current: Decimal) -> Decimal:
        for value in self.propensity_ladder:
            if value > current:
                return value
        return Decimal(1)

    def allocate(
        self,
        safety_decision: SafetyDecision,
        *,
        forecasts: Iterable[RootCauseVOIForecast],
        allocation_epoch: int,
    ) -> VOIAllocationDecision:
        if safety_decision.manifest_sha256 != self.manifest_sha256:
            raise ValueError("VOI safety decision manifest hash differs")
        if not safety_decision.verify_digest():
            raise ValueError("VOI safety decision digest is invalid")
        epoch = int(allocation_epoch)
        if epoch < 0:
            raise ValueError("allocation_epoch cannot be negative")
        forecast_items = tuple(
            sorted(forecasts, key=lambda row: (row.axis, row.action_id))
        )
        if len({row.action_id for row in forecast_items}) != len(forecast_items):
            raise ValueError("VOI action ids must be unique per allocation")
        future_forecasts = [
            row.action_id
            for row in forecast_items
            if row.history_case_count > epoch
        ]
        if future_forecasts:
            raise ValueError(
                "VOI forecast uses current/future full outcome history: "
                f"{future_forecasts}"
            )
        unknown_axes = sorted(
            {row.axis for row in forecast_items} - set(self.audit_policies)
        )
        if unknown_axes:
            raise ValueError(f"VOI forecasts contain unregistered axes: {unknown_axes}")
        baseline = {
            axis: safety_decision.propensity_for(axis)
            for axis in self.audit_policies
        }
        current = dict(baseline)
        score_by_axis: dict[str, float] = {}
        action_by_axis: dict[str, str] = {}
        cost_by_axis: dict[str, float] = {}
        target_steps: dict[str, int] = {}
        for forecast in forecast_items:
            score = self._score(forecast)
            if score > score_by_axis.get(forecast.axis, -1.0):
                score_by_axis[forecast.axis] = score
                action_by_axis[forecast.axis] = forecast.action_id
                cost_by_axis[forecast.axis] = forecast.expected_cost_ms
                target_steps[forecast.axis] = self._target_steps(score)

        reasons: list[str] = []
        extra_cost = 0.0
        applied_steps = {axis: 0 for axis in self.audit_policies}
        if safety_decision.state in {"calibration_full", "full_rollback"}:
            reasons.append(f"safety_state_{safety_decision.state}_dominates_voi")
        elif self.maximum_extra_expected_cost_ms <= 0:
            reasons.append("voi_extra_budget_is_zero")
        else:
            while True:
                candidates = [
                    axis
                    for axis, steps in target_steps.items()
                    if applied_steps[axis] < steps and current[axis] < 1
                ]
                candidates.sort(
                    key=lambda axis: (
                        -score_by_axis[axis],
                        action_by_axis[axis],
                        axis,
                    )
                )
                applied = False
                for axis in candidates:
                    next_value = self._next(current[axis])
                    package: list[tuple[str, Decimal, float]] = [
                        (
                            axis,
                            next_value,
                            (float(next_value) - float(current[axis]))
                            * cost_by_axis[axis],
                        )
                    ]
                    if (
                        self.raise_joint_with_component
                        and axis != "joint"
                        and current.get("joint", Decimal(1)) < 1
                        and applied_steps.get("joint", 0) == 0
                    ):
                        joint_next = self._next(current["joint"])
                        joint_cost = cost_by_axis.get(
                            "joint",
                            cost_by_axis[axis],
                        )
                        package.append(
                            (
                                "joint",
                                joint_next,
                                (float(joint_next) - float(current["joint"]))
                                * joint_cost,
                            )
                        )
                    package_cost = sum(row[2] for row in package)
                    if (
                        extra_cost + package_cost
                        > self.maximum_extra_expected_cost_ms + 1e-12
                    ):
                        continue
                    for package_axis, value, _cost in package:
                        self.audit_policies[package_axis].validate_requested(value)
                        if value < baseline[package_axis]:
                            raise AssertionError("VOI attempted to lower a safety propensity")
                        if value > current[package_axis]:
                            current[package_axis] = value
                            applied_steps[package_axis] += 1
                    extra_cost += package_cost
                    applied = True
                    break
                if not applied:
                    break
            if any(value > baseline[axis] for axis, value in current.items()):
                reasons.append("voi_raise_applied_within_registered_budget")
            else:
                reasons.append("no_voi_raise_passed_threshold_and_budget")

        propensities = tuple(sorted(current.items()))
        selected_actions = tuple(
            sorted(
                (axis, action_by_axis[axis])
                for axis in action_by_axis
                if current[axis] > baseline[axis]
            )
        )
        score_rows = tuple(sorted(score_by_axis.items()))
        decision_without_id = VOIAllocationDecision(
            manifest_sha256=self.manifest_sha256,
            allocation_id="0" * 64,
            safety_decision_id=safety_decision.decision_id,
            allocation_epoch=epoch,
            audit_propensities=propensities,
            selected_actions=selected_actions,
            score_per_cpu_ms=score_rows,
            extra_expected_cost_ms=extra_cost,
            maximum_extra_expected_cost_ms=self.maximum_extra_expected_cost_ms,
            reasons=tuple(reasons),
            forecasts=forecast_items,
            model_sha256=self.model_sha256,
        )
        allocation_id = hashlib.sha256(
            _canonical_json_bytes(decision_without_id._digest_payload())
        ).hexdigest()
        return VOIAllocationDecision(
            manifest_sha256=decision_without_id.manifest_sha256,
            allocation_id=allocation_id,
            safety_decision_id=decision_without_id.safety_decision_id,
            allocation_epoch=decision_without_id.allocation_epoch,
            audit_propensities=decision_without_id.audit_propensities,
            selected_actions=decision_without_id.selected_actions,
            score_per_cpu_ms=decision_without_id.score_per_cpu_ms,
            extra_expected_cost_ms=decision_without_id.extra_expected_cost_ms,
            maximum_extra_expected_cost_ms=(
                decision_without_id.maximum_extra_expected_cost_ms
            ),
            reasons=decision_without_id.reasons,
            forecasts=decision_without_id.forecasts,
            model_sha256=decision_without_id.model_sha256,
        )

    def allocate_snapshot(
        self,
        safety_decision: SafetyDecision,
        *,
        snapshot: RootCauseVOIForecastSnapshot,
        allocation_epoch: int,
    ) -> VOIAllocationDecision:
        if snapshot.manifest_sha256 != self.manifest_sha256:
            raise ValueError("VOI snapshot manifest hash differs")
        if snapshot.model_sha256 != self.model_sha256:
            raise ValueError("VOI snapshot model hash differs")
        if not snapshot.verify_digest():
            raise ValueError("VOI snapshot digest is invalid")
        if snapshot.history_case_count > int(allocation_epoch):
            raise ValueError("VOI snapshot includes current/future full outcomes")
        return self.allocate(
            safety_decision,
            forecasts=snapshot.forecasts,
            allocation_epoch=allocation_epoch,
        )

    def allocate_with_control(
        self,
        safety_decision: SafetyDecision,
        *,
        snapshot: RootCauseVOIForecastSnapshot,
        allocation_epoch: int,
    ) -> VOIAllocationPair:
        voi_decision = self.allocate_snapshot(
            safety_decision,
            snapshot=snapshot,
            allocation_epoch=allocation_epoch,
        )
        no_voi_control = self.allocate(
            safety_decision,
            forecasts=(),
            allocation_epoch=allocation_epoch,
        )
        pair = VOIAllocationPair(
            snapshot_id=snapshot.snapshot_id,
            voi_decision=voi_decision,
            no_voi_control=no_voi_control,
        )
        if not pair.to_dict()["raise_only"]:
            raise AssertionError("VOI allocation pair violated raise-only control")
        return pair
