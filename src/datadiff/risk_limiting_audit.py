from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_FLOOR, localcontext
from typing import Any, Iterable, Mapping


AUDIT_AXES = ("candidate", "backend", "relation", "joint")
_UINT64_RANGE = 1 << 64


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def _validated_sha256(value: str, *, field: str) -> str:
    normalized = str(value or "").strip().lower()
    if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
        raise ValueError(f"{field} must be a lowercase 64-character SHA-256 digest")
    return normalized


def _probability(value: Decimal | float | int | str, *, field: str) -> Decimal:
    try:
        probability = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{field} must be a finite probability") from exc
    if not probability.is_finite() or probability <= 0 or probability > 1:
        raise ValueError(f"{field} must satisfy 0 < p <= 1")
    return probability


def _probability_text(value: Decimal) -> str:
    text = format(value, "f").rstrip("0").rstrip(".")
    return text if "." in text or text == "1" else f"{text}.0"


def _probability_threshold(value: Decimal) -> int:
    if value == 1:
        return _UINT64_RANGE
    with localcontext() as context:
        context.prec = max(100, len(value.as_tuple().digits) + 32)
        return int(
            (value * Decimal(_UINT64_RANGE)).to_integral_value(
                rounding=ROUND_FLOOR
            )
        )


def _effective_probability(threshold: int) -> Decimal:
    if threshold < 0 or threshold > _UINT64_RANGE:
        raise ValueError("probability threshold is outside the uint64 design range")
    with localcontext() as context:
        context.prec = 100
        return Decimal(threshold) / Decimal(_UINT64_RANGE)


@dataclass(frozen=True, slots=True)
class AuditOpportunity:
    campaign_seed: int
    case_index: int
    axis: str
    item_key: str
    decision_epoch: int = 0

    def __post_init__(self) -> None:
        if int(self.campaign_seed) < 0:
            raise ValueError("campaign_seed must be non-negative")
        if int(self.case_index) < 0:
            raise ValueError("case_index must be non-negative")
        if str(self.axis) not in AUDIT_AXES:
            raise ValueError(f"unknown audit axis: {self.axis}")
        if not str(self.item_key or "").strip():
            raise ValueError("item_key cannot be empty")
        if int(self.decision_epoch) < 0:
            raise ValueError("decision_epoch must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "campaign_seed": int(self.campaign_seed),
            "case_index": int(self.case_index),
            "axis": str(self.axis),
            "item_key": str(self.item_key),
            "decision_epoch": int(self.decision_epoch),
        }


@dataclass(frozen=True, slots=True)
class AuditAxisPolicy:
    axis: str
    floor: Decimal
    initial: Decimal
    maximum: Decimal
    ladder: tuple[Decimal, ...]

    @classmethod
    def from_payload(cls, axis: str, payload: Mapping[str, Any]) -> "AuditAxisPolicy":
        floor = _probability(payload.get("floor", 0), field=f"{axis}.floor")
        initial = _probability(payload.get("initial", 0), field=f"{axis}.initial")
        maximum = _probability(payload.get("maximum", 0), field=f"{axis}.maximum")
        raw_ladder = payload.get("ladder", ())
        if not isinstance(raw_ladder, (list, tuple)) or not raw_ladder:
            raise ValueError(f"{axis}.ladder must contain at least one probability")
        ladder = tuple(
            _probability(value, field=f"{axis}.ladder") for value in raw_ladder
        )
        return cls(
            axis=str(axis),
            floor=floor,
            initial=initial,
            maximum=maximum,
            ladder=ladder,
        )

    def __post_init__(self) -> None:
        if self.axis not in AUDIT_AXES:
            raise ValueError(f"unknown audit axis: {self.axis}")
        if not (self.floor <= self.initial <= self.maximum):
            raise ValueError(
                f"{self.axis} probabilities must satisfy floor <= initial <= maximum"
            )
        if tuple(sorted(set(self.ladder))) != self.ladder:
            raise ValueError(f"{self.axis}.ladder must be strictly increasing")
        if self.ladder[0] < self.floor or self.ladder[-1] > self.maximum:
            raise ValueError(f"{self.axis}.ladder must stay within floor and maximum")
        if self.initial not in self.ladder:
            raise ValueError(f"{self.axis}.initial must appear in the propensity ladder")
        if _probability_threshold(self.floor) <= 0:
            raise ValueError(
                f"{self.axis}.floor is below the representable uint64 propensity floor"
            )

    def validate_requested(self, value: Decimal | float | int | str | None) -> Decimal:
        requested = self.initial if value is None else _probability(
            value,
            field=f"{self.axis}.requested_propensity",
        )
        if requested < self.floor or requested > self.maximum:
            raise ValueError(
                f"{self.axis} requested propensity must stay within registered bounds"
            )
        if requested not in self.ladder:
            raise ValueError(
                f"{self.axis} requested propensity must be on the registered ladder"
            )
        return requested

    @property
    def effective_floor(self) -> Decimal:
        return _effective_probability(_probability_threshold(self.floor))

    def next_propensity(self, current: Decimal | float | int | str) -> Decimal:
        value = self.validate_requested(current)
        for candidate in self.ladder:
            if candidate > value:
                return candidate
        return self.maximum

    def to_dict(self) -> dict[str, Any]:
        return {
            "axis": self.axis,
            "floor": float(self.floor),
            "initial": float(self.initial),
            "maximum": float(self.maximum),
            "ladder": [float(value) for value in self.ladder],
            "floor_exact": _probability_text(self.floor),
            "initial_exact": _probability_text(self.initial),
            "maximum_exact": _probability_text(self.maximum),
            "ladder_exact": [_probability_text(value) for value in self.ladder],
            "effective_floor_exact": _probability_text(
                self.effective_floor
            ),
        }


@dataclass(frozen=True, slots=True)
class AuditDecision:
    decision_id: str
    opportunity: AuditOpportunity
    requested_propensity: Decimal
    propensity: Decimal
    selection_threshold_u64: int
    random_u64: int
    selected: bool
    forced: bool
    reason: str
    policy_floor: Decimal
    policy_maximum: Decimal

    @property
    def inclusion_weight(self) -> float:
        return 1.0 / float(self.propensity) if self.selected else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "rlcmf-audit-decision-v1",
            "decision_id": self.decision_id,
            "opportunity": self.opportunity.to_dict(),
            "propensity": float(self.propensity),
            "propensity_exact": _probability_text(self.propensity),
            "requested_propensity": float(self.requested_propensity),
            "requested_propensity_exact": _probability_text(
                self.requested_propensity
            ),
            "selection_threshold_u64": int(self.selection_threshold_u64),
            "random_u64": int(self.random_u64),
            "uniform_draw": self.random_u64 / _UINT64_RANGE,
            "selected": bool(self.selected),
            "forced": bool(self.forced),
            "reason": str(self.reason),
            "policy_floor": float(self.policy_floor),
            "policy_maximum": float(self.policy_maximum),
            "inclusion_weight": self.inclusion_weight,
        }


@dataclass(frozen=True, slots=True)
class CoupledAuditDecisionSet:
    """Comonotone shared-draw decisions with exact registered marginals."""

    coupling_id: str
    coupling_key: str
    random_u64: int
    decisions: Mapping[str, AuditDecision]
    marginal_propensity_exact: Mapping[str, str]
    pairwise_joint_propensity_exact: Mapping[str, str]
    pairwise_union_propensity_exact: Mapping[str, str]
    joint_implies_components: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "rlcmf-coupled-audit-decision-set-v1",
            "coupling_id": self.coupling_id,
            "coupling_key": self.coupling_key,
            "design": "shared-uniform-comonotone-v1",
            "random_u64": int(self.random_u64),
            "uniform_draw": self.random_u64 / _UINT64_RANGE,
            "marginal_propensity_exact": dict(
                sorted(self.marginal_propensity_exact.items())
            ),
            "pairwise_joint_propensity_exact": dict(
                sorted(self.pairwise_joint_propensity_exact.items())
            ),
            "pairwise_union_propensity_exact": dict(
                sorted(self.pairwise_union_propensity_exact.items())
            ),
            "joint_implies_components": self.joint_implies_components,
            "decisions": {
                axis: decision.to_dict()
                for axis, decision in sorted(self.decisions.items())
            },
        }


class DeterministicSentinelAuditor:
    """Outcome-blind, reproducible Bernoulli audit decisions."""

    def __init__(
        self,
        *,
        manifest_sha256: str,
        salt: str,
        policies: Iterable[AuditAxisPolicy],
        serialization: str = "canonical-json-sort-keys-v1",
    ) -> None:
        self.manifest_sha256 = _validated_sha256(
            manifest_sha256,
            field="manifest_sha256",
        )
        self.salt = str(salt or "").strip()
        if not self.salt:
            raise ValueError("audit salt cannot be empty")
        self.serialization = str(serialization or "").strip()
        if self.serialization != "canonical-json-sort-keys-v1":
            raise ValueError("unsupported audit serialization contract")
        policy_items = tuple(policies)
        self.policies = {policy.axis: policy for policy in policy_items}
        if not self.policies:
            raise ValueError("at least one audit-axis policy is required")
        if len(self.policies) != len(policy_items):
            raise ValueError("duplicate audit-axis policies are not allowed")

    @classmethod
    def from_manifest(cls, manifest: Mapping[str, Any], *, manifest_sha256: str) -> "DeterministicSentinelAuditor":
        audit = manifest.get("audit", {})
        if not isinstance(audit, Mapping):
            raise ValueError("manifest.audit must be an object")
        if audit.get("hash_algorithm") != "sha256-first64-uniform-v1":
            raise ValueError("unsupported audit hash algorithm")
        if audit.get("serialization") != "canonical-json-sort-keys-v1":
            raise ValueError("unsupported audit serialization contract")
        axes = audit.get("axes", {})
        if not isinstance(axes, Mapping):
            raise ValueError("manifest.audit.axes must be an object")
        policies = [
            AuditAxisPolicy.from_payload(str(axis), payload)
            for axis, payload in axes.items()
            if isinstance(payload, Mapping)
        ]
        if len(policies) != len(axes):
            raise ValueError("every audit axis must have an object policy")
        return cls(
            manifest_sha256=manifest_sha256,
            salt=str(audit.get("salt", "")),
            policies=policies,
            serialization=str(audit.get("serialization", "")),
        )

    def decide(
        self,
        opportunity: AuditOpportunity,
        *,
        requested_propensity: Decimal | float | int | str | None = None,
        forced: bool = False,
        reason: str = "registered_random_sentinel",
    ) -> AuditDecision:
        policy = self.policies.get(opportunity.axis)
        if policy is None:
            raise KeyError(f"no registered policy for audit axis: {opportunity.axis}")
        registered = policy.validate_requested(requested_propensity)
        if not str(reason or "").strip():
            raise ValueError("audit decision reason cannot be empty")
        selection_threshold = (
            _UINT64_RANGE if forced else _probability_threshold(registered)
        )
        propensity = _effective_probability(selection_threshold)
        material = {
            "schema_version": "rlcmf-audit-draw-v1",
            "manifest_sha256": self.manifest_sha256,
            "salt": self.salt,
            "opportunity": opportunity.to_dict(),
        }
        random_u64 = int.from_bytes(
            hashlib.sha256(_canonical_json_bytes(material)).digest()[:8],
            "big",
        )
        selected = bool(random_u64 < selection_threshold)
        decision_material = {
            **material,
            "requested_propensity_exact": _probability_text(registered),
            "effective_propensity_exact": _probability_text(propensity),
            "selection_threshold_u64": selection_threshold,
            "random_u64": random_u64,
            "selected": selected,
            "forced": bool(forced),
            "reason": str(reason),
        }
        decision_id = hashlib.sha256(
            _canonical_json_bytes(decision_material)
        ).hexdigest()
        return AuditDecision(
            decision_id=decision_id,
            opportunity=opportunity,
            requested_propensity=registered,
            propensity=propensity,
            selection_threshold_u64=selection_threshold,
            random_u64=random_u64,
            selected=selected,
            forced=bool(forced),
            reason=str(reason),
            policy_floor=policy.floor,
            policy_maximum=policy.maximum,
        )

    def decide_coupled(
        self,
        opportunities: Mapping[str, AuditOpportunity],
        *,
        requested_propensities: Mapping[
            str, Decimal | float | int | str | None
        ],
        forced_axes: Iterable[str] = (),
        reasons: Mapping[str, str] | None = None,
        coupling_key: str,
        require_joint_subset: bool = True,
    ) -> CoupledAuditDecisionSet:
        normalized_key = str(coupling_key or "").strip()
        if not normalized_key:
            raise ValueError("coupled audit coupling_key cannot be empty")
        axes = tuple(sorted(str(axis) for axis in opportunities))
        if not axes:
            raise ValueError("coupled audit requires at least one opportunity")
        if set(axes) != set(requested_propensities):
            raise ValueError("coupled audit propensity axes must match opportunities")
        normalized_opportunities: dict[str, AuditOpportunity] = {}
        registered: dict[str, Decimal] = {}
        base_coordinates: set[tuple[int, int, int]] = set()
        for axis in axes:
            opportunity = opportunities[axis]
            if opportunity.axis != axis:
                raise ValueError("coupled audit mapping key must match opportunity axis")
            policy = self.policies.get(axis)
            if policy is None:
                raise KeyError(f"no registered policy for audit axis: {axis}")
            normalized_opportunities[axis] = opportunity
            registered[axis] = policy.validate_requested(
                requested_propensities[axis]
            )
            base_coordinates.add(
                (
                    opportunity.campaign_seed,
                    opportunity.case_index,
                    opportunity.decision_epoch,
                )
            )
        if len(base_coordinates) != 1:
            raise ValueError(
                "coupled audit opportunities must share seed, case, and decision epoch"
            )
        forced = {str(axis) for axis in forced_axes}
        if not forced.issubset(axes):
            raise ValueError("coupled audit forced axes must be registered opportunities")
        if "joint" in forced:
            forced.update(axis for axis in axes if axis != "joint")
        effective_requested = {
            axis: Decimal(1) if axis in forced else registered[axis]
            for axis in axes
        }
        if require_joint_subset and "joint" in effective_requested:
            joint_probability = effective_requested["joint"]
            violating = sorted(
                axis
                for axis, probability in effective_requested.items()
                if axis != "joint" and probability < joint_probability
            )
            if violating:
                raise ValueError(
                    "joint propensity must not exceed coupled component propensities: "
                    f"{violating}"
                )
        reason_map = {axis: "registered_coupled_sentinel" for axis in axes}
        reason_map.update({str(axis): str(reason) for axis, reason in (reasons or {}).items()})
        if set(reason_map) != set(axes) or any(
            not str(reason_map[axis] or "").strip() for axis in axes
        ):
            raise ValueError("coupled audit reasons must be non-empty for every axis")

        draw_material = {
            "schema_version": "rlcmf-coupled-audit-draw-v1",
            "manifest_sha256": self.manifest_sha256,
            "salt": self.salt,
            "coupling_key": normalized_key,
            "opportunities": {
                axis: normalized_opportunities[axis].to_dict() for axis in axes
            },
        }
        random_u64 = int.from_bytes(
            hashlib.sha256(_canonical_json_bytes(draw_material)).digest()[:8],
            "big",
        )
        coupling_material = {
            **draw_material,
            "requested_propensity_exact": {
                axis: _probability_text(registered[axis]) for axis in axes
            },
            "forced_axes": sorted(forced),
            "random_u64": random_u64,
            "require_joint_subset": bool(require_joint_subset),
        }
        coupling_id = hashlib.sha256(
            _canonical_json_bytes(coupling_material)
        ).hexdigest()
        decisions: dict[str, AuditDecision] = {}
        marginal: dict[str, str] = {}
        for axis in axes:
            policy = self.policies[axis]
            forced_axis = axis in forced
            threshold = (
                _UINT64_RANGE
                if forced_axis
                else _probability_threshold(registered[axis])
            )
            propensity = _effective_probability(threshold)
            selected = bool(random_u64 < threshold)
            decision_material = {
                "schema_version": "rlcmf-coupled-axis-decision-id-v1",
                "coupling_id": coupling_id,
                "axis": axis,
                "opportunity": normalized_opportunities[axis].to_dict(),
                "requested_propensity_exact": _probability_text(registered[axis]),
                "effective_propensity_exact": _probability_text(propensity),
                "selection_threshold_u64": threshold,
                "random_u64": random_u64,
                "selected": selected,
                "forced": forced_axis,
                "reason": reason_map[axis],
            }
            decision_id = hashlib.sha256(
                _canonical_json_bytes(decision_material)
            ).hexdigest()
            decisions[axis] = AuditDecision(
                decision_id=decision_id,
                opportunity=normalized_opportunities[axis],
                requested_propensity=registered[axis],
                propensity=propensity,
                selection_threshold_u64=threshold,
                random_u64=random_u64,
                selected=selected,
                forced=forced_axis,
                reason=reason_map[axis],
                policy_floor=policy.floor,
                policy_maximum=policy.maximum,
            )
            marginal[axis] = _probability_text(propensity)

        pairwise_joint = {}
        pairwise_union = {}
        for left_index, left in enumerate(axes):
            for right in axes[left_index + 1 :]:
                key = f"{left}|{right}"
                left_probability = decisions[left].propensity
                right_probability = decisions[right].propensity
                pairwise_joint[key] = _probability_text(
                    min(left_probability, right_probability)
                )
                pairwise_union[key] = _probability_text(
                    max(left_probability, right_probability)
                )
        joint_implies_components = bool(
            "joint" not in decisions
            or all(
                not decisions["joint"].selected or decisions[axis].selected
                for axis in axes
                if axis != "joint"
            )
        )
        return CoupledAuditDecisionSet(
            coupling_id=coupling_id,
            coupling_key=normalized_key,
            random_u64=random_u64,
            decisions=decisions,
            marginal_propensity_exact=marginal,
            pairwise_joint_propensity_exact=pairwise_joint,
            pairwise_union_propensity_exact=pairwise_union,
            joint_implies_components=joint_implies_components,
        )

    def verify_coupled(self, decision_set: CoupledAuditDecisionSet) -> bool:
        try:
            expected = self.decide_coupled(
                {
                    axis: decision.opportunity
                    for axis, decision in decision_set.decisions.items()
                },
                requested_propensities={
                    axis: decision.requested_propensity
                    for axis, decision in decision_set.decisions.items()
                },
                forced_axes=[
                    axis
                    for axis, decision in decision_set.decisions.items()
                    if decision.forced
                ],
                reasons={
                    axis: decision.reason
                    for axis, decision in decision_set.decisions.items()
                },
                coupling_key=decision_set.coupling_key,
                require_joint_subset=True,
            )
        except (KeyError, ValueError):
            return False
        return expected == decision_set

    def verify(self, decision: AuditDecision) -> bool:
        expected = self.decide(
            decision.opportunity,
            requested_propensity=(
                decision.requested_propensity
            ),
            forced=decision.forced,
            reason=decision.reason,
        )
        return expected == decision

    def policy_summary(self) -> dict[str, Any]:
        return {
            "manifest_sha256": self.manifest_sha256,
            "salt_sha256": hashlib.sha256(self.salt.encode("utf-8")).hexdigest(),
            "hash_algorithm": "sha256-first64-uniform-v1",
            "serialization": self.serialization,
            "axes": {
                axis: policy.to_dict()
                for axis, policy in sorted(self.policies.items())
            },
        }
