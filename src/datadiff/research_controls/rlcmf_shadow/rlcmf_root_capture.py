from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping, Sequence


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def _sha256(value: str, *, field_name: str) -> str:
    normalized = str(value or "").strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return normalized


def _probability(value: Decimal | float | int | str, *, field_name: str) -> Decimal:
    try:
        probability = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{field_name} must be a probability") from exc
    if not probability.is_finite() or probability <= 0 or probability > 1:
        raise ValueError(f"{field_name} must satisfy 0 < p <= 1")
    return probability


def _probability_text(value: Decimal) -> str:
    text = format(value, "f").rstrip("0").rstrip(".")
    return text or "0"


@dataclass(frozen=True, slots=True)
class AtomicRootCaptureGroup:
    group_id: str
    opportunity_id: str
    axis_arms: tuple[str, ...]
    inclusion_propensity: Decimal
    selected: bool
    independence_block_id: str
    coupling_id: str = ""
    pairwise_joint_propensity_exact: Mapping[str, str] = field(
        default_factory=dict
    )
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.group_id or "").strip():
            raise ValueError("root capture group_id cannot be empty")
        if not str(self.opportunity_id or "").strip():
            raise ValueError("root capture opportunity_id cannot be empty")
        normalized_arms = tuple(sorted({str(arm) for arm in self.axis_arms if str(arm)}))
        if not normalized_arms:
            raise ValueError("root capture group requires at least one axis/arm")
        probability = _probability(
            self.inclusion_propensity,
            field_name="inclusion_propensity",
        )
        block = str(self.independence_block_id or "").strip()
        if not block:
            raise ValueError("root capture independence_block_id cannot be empty")
        pairwise = {}
        for other_group_id, value in self.pairwise_joint_propensity_exact.items():
            other = str(other_group_id or "").strip()
            if not other or other == self.group_id:
                raise ValueError("invalid pairwise root capture group identifier")
            pairwise[other] = _probability_text(
                _probability(value, field_name="pairwise_joint_propensity")
            )
        object.__setattr__(self, "group_id", str(self.group_id).strip())
        object.__setattr__(self, "opportunity_id", str(self.opportunity_id).strip())
        object.__setattr__(self, "axis_arms", normalized_arms)
        object.__setattr__(self, "inclusion_propensity", probability)
        object.__setattr__(self, "independence_block_id", block)
        object.__setattr__(self, "coupling_id", str(self.coupling_id or "").strip())
        object.__setattr__(
            self,
            "pairwise_joint_propensity_exact",
            dict(sorted(pairwise.items())),
        )
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "rlcmf-atomic-root-capture-group-v1",
            "group_id": self.group_id,
            "opportunity_id": self.opportunity_id,
            "axis_arms": list(self.axis_arms),
            "inclusion_propensity": float(self.inclusion_propensity),
            "inclusion_propensity_exact": _probability_text(
                self.inclusion_propensity
            ),
            "selected": self.selected,
            "independence_block_id": self.independence_block_id,
            "coupling_id": self.coupling_id,
            "pairwise_joint_propensity_exact": dict(
                self.pairwise_joint_propensity_exact
            ),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class RootOpportunityFrame:
    root_id: str
    capable_group_ids: tuple[str, ...]
    complete: bool
    union_inclusion_propensity: Decimal | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        root_id = str(self.root_id or "").strip()
        if not root_id:
            raise ValueError("root opportunity frame root_id cannot be empty")
        groups = tuple(sorted({str(value) for value in self.capable_group_ids if str(value)}))
        if not groups:
            raise ValueError("root opportunity frame requires capable groups")
        union = self.union_inclusion_propensity
        if union is not None:
            union = _probability(union, field_name="union_inclusion_propensity")
        object.__setattr__(self, "root_id", root_id)
        object.__setattr__(self, "capable_group_ids", groups)
        object.__setattr__(self, "union_inclusion_propensity", union)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "rlcmf-root-opportunity-frame-v1",
            "root_id": self.root_id,
            "capable_group_ids": list(self.capable_group_ids),
            "complete": self.complete,
            "union_inclusion_propensity_exact": (
                _probability_text(self.union_inclusion_propensity)
                if self.union_inclusion_propensity is not None
                else None
            ),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class RootCaptureEvidence:
    capture_id: str
    root_id: str
    group_id: str
    evidence_level: str
    candidate_family_keys: tuple[str, ...]
    raw_to_root_mapping: Mapping[str, str]
    native_reproduction_id: str
    independent_triage_id: str
    metadata: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "rlcmf-root-capture-evidence-v1",
            "capture_id": self.capture_id,
            "root_id": self.root_id,
            "group_id": self.group_id,
            "evidence_level": self.evidence_level,
            "candidate_family_keys": list(self.candidate_family_keys),
            "raw_to_root_mapping": dict(self.raw_to_root_mapping),
            "native_reproduction_id": self.native_reproduction_id,
            "independent_triage_id": self.independent_triage_id,
            "metadata": dict(self.metadata),
        }


class RootCaptureLedger:
    """Root-level capture records without event-weight-to-unique-root relabeling."""

    def __init__(
        self,
        *,
        manifest_sha256: str,
        root_clusterer_sha256: str,
        design_id: str,
    ) -> None:
        self.manifest_sha256 = _sha256(
            manifest_sha256,
            field_name="manifest_sha256",
        )
        self.root_clusterer_sha256 = _sha256(
            root_clusterer_sha256,
            field_name="root_clusterer_sha256",
        )
        self.design_id = str(design_id or "").strip()
        if not self.design_id:
            raise ValueError("root capture design_id cannot be empty")
        self.groups: dict[str, AtomicRootCaptureGroup] = {}
        self.frames: dict[str, RootOpportunityFrame] = {}
        self.captures: dict[str, RootCaptureEvidence] = {}
        self._root_group_captures: set[tuple[str, str]] = set()

    def register_group(
        self,
        *,
        opportunity_id: str,
        axis_arms: Sequence[str],
        inclusion_propensity: Decimal | float | int | str,
        selected: bool,
        independence_block_id: str,
        coupling_id: str = "",
        pairwise_joint_propensity_exact: Mapping[str, str] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> AtomicRootCaptureGroup:
        material = {
            "schema_version": "rlcmf-atomic-root-capture-group-id-v1",
            "manifest_sha256": self.manifest_sha256,
            "design_id": self.design_id,
            "opportunity_id": str(opportunity_id),
            "axis_arms": sorted({str(value) for value in axis_arms}),
            "independence_block_id": str(independence_block_id),
            "coupling_id": str(coupling_id or ""),
        }
        group_id = hashlib.sha256(_canonical_json_bytes(material)).hexdigest()
        if group_id in self.groups:
            raise ValueError(f"duplicate root capture group: {group_id}")
        group = AtomicRootCaptureGroup(
            group_id=group_id,
            opportunity_id=str(opportunity_id),
            axis_arms=tuple(axis_arms),
            inclusion_propensity=_probability(
                inclusion_propensity,
                field_name="inclusion_propensity",
            ),
            selected=bool(selected),
            independence_block_id=str(independence_block_id),
            coupling_id=str(coupling_id or ""),
            pairwise_joint_propensity_exact=dict(
                pairwise_joint_propensity_exact or {}
            ),
            metadata=dict(metadata or {}),
        )
        self.groups[group_id] = group
        return group

    def register_root_frame(
        self,
        *,
        root_id: str,
        capable_group_ids: Sequence[str],
        complete: bool,
        union_inclusion_propensity: Decimal | float | int | str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> RootOpportunityFrame:
        normalized_root = str(root_id or "").strip()
        if normalized_root in self.frames:
            raise ValueError(f"duplicate root opportunity frame: {normalized_root}")
        missing = sorted(set(capable_group_ids) - set(self.groups))
        if missing:
            raise ValueError(f"root opportunity frame references unknown groups: {missing}")
        frame = RootOpportunityFrame(
            root_id=normalized_root,
            capable_group_ids=tuple(capable_group_ids),
            complete=bool(complete),
            union_inclusion_propensity=(
                _probability(
                    union_inclusion_propensity,
                    field_name="union_inclusion_propensity",
                )
                if union_inclusion_propensity is not None
                else None
            ),
            metadata=dict(metadata or {}),
        )
        self.frames[normalized_root] = frame
        return frame

    def record_capture(
        self,
        *,
        root_id: str,
        group_id: str,
        evidence_level: str,
        candidate_family_keys: Sequence[str],
        raw_to_root_mapping: Mapping[str, str],
        native_reproduction_id: str = "",
        independent_triage_id: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> RootCaptureEvidence:
        normalized_root = str(root_id or "").strip()
        group = self.groups.get(str(group_id))
        if group is None:
            raise KeyError(f"unknown root capture group: {group_id}")
        if not group.selected:
            raise ValueError("an unselected atomic group cannot record a root capture")
        level = str(evidence_level or "").strip()
        if level not in {"candidate_root", "independently_confirmed_real_root"}:
            raise ValueError("unknown root capture evidence_level")
        native_id = str(native_reproduction_id or "").strip()
        triage_id = str(independent_triage_id or "").strip()
        if level == "independently_confirmed_real_root" and (
            not native_id or not triage_id
        ):
            raise ValueError(
                "confirmed real-root capture requires native reproduction and independent triage"
            )
        key = (normalized_root, group.group_id)
        if key in self._root_group_captures:
            raise ValueError("duplicate root capture within one atomic group")
        mappings = {
            str(raw): str(mapped)
            for raw, mapped in raw_to_root_mapping.items()
        }
        if not mappings or any(mapped != normalized_root for mapped in mappings.values()):
            raise ValueError("raw-to-root mapping must map every key to the captured root")
        families = tuple(sorted({str(value) for value in candidate_family_keys if str(value)}))
        capture_material = {
            "schema_version": "rlcmf-root-capture-id-v1",
            "manifest_sha256": self.manifest_sha256,
            "root_clusterer_sha256": self.root_clusterer_sha256,
            "root_id": normalized_root,
            "group_id": group.group_id,
            "evidence_level": level,
            "candidate_family_keys": list(families),
            "raw_to_root_mapping": dict(sorted(mappings.items())),
        }
        capture_id = hashlib.sha256(
            _canonical_json_bytes(capture_material)
        ).hexdigest()
        evidence = RootCaptureEvidence(
            capture_id=capture_id,
            root_id=normalized_root,
            group_id=group.group_id,
            evidence_level=level,
            candidate_family_keys=families,
            raw_to_root_mapping=dict(sorted(mappings.items())),
            native_reproduction_id=native_id,
            independent_triage_id=triage_id,
            metadata=dict(metadata or {}),
        )
        self.captures[capture_id] = evidence
        self._root_group_captures.add(key)
        return evidence

    def estimator_inputs(self, root_id: str) -> dict[str, Any]:
        normalized_root = str(root_id or "").strip()
        captures = sorted(
            (
                capture
                for capture in self.captures.values()
                if capture.root_id == normalized_root
            ),
            key=lambda capture: capture.capture_id,
        )
        frame = self.frames.get(normalized_root)
        base = {
            "schema_version": "rlcmf-root-estimator-inputs-v1",
            "root_id": normalized_root,
            "root_clusterer_sha256": self.root_clusterer_sha256,
            "direct_capture_count": len(captures),
            "candidate_capture_count": sum(
                capture.evidence_level == "candidate_root" for capture in captures
            ),
            "confirmed_capture_count": sum(
                capture.evidence_level == "independently_confirmed_real_root"
                for capture in captures
            ),
            "capture_ids": [capture.capture_id for capture in captures],
        }
        if frame is None:
            return {
                **base,
                "status": "not_identified",
                "reason": "root_opportunity_frame_missing",
                "root_inclusion_propensity_exact": None,
                "second_order_ready": False,
            }
        if not frame.complete:
            return {
                **base,
                "status": "not_identified",
                "reason": "root_opportunity_frame_incomplete",
                "root_inclusion_propensity_exact": None,
                "second_order_ready": False,
                "frame": frame.to_dict(),
            }
        groups = [self.groups[group_id] for group_id in frame.capable_group_ids]
        if frame.union_inclusion_propensity is not None:
            inclusion = frame.union_inclusion_propensity
            inclusion_source = "registered_exact_union"
        elif len({group.independence_block_id for group in groups}) == len(groups):
            miss_probability = Decimal(1)
            for group in groups:
                miss_probability *= Decimal(1) - group.inclusion_propensity
            inclusion = Decimal(1) - miss_probability
            inclusion_source = "independent_atomic_group_product"
        else:
            return {
                **base,
                "status": "not_identified",
                "reason": "correlated_capture_groups_lack_exact_union_probability",
                "root_inclusion_propensity_exact": None,
                "second_order_ready": False,
                "frame": frame.to_dict(),
            }
        pairwise_missing = []
        for left_index, left in enumerate(groups):
            for right in groups[left_index + 1 :]:
                if (
                    right.group_id not in left.pairwise_joint_propensity_exact
                    and left.group_id not in right.pairwise_joint_propensity_exact
                ):
                    pairwise_missing.append(f"{left.group_id}|{right.group_id}")
        second_order_ready = len(groups) <= 1 or not pairwise_missing
        return {
            **base,
            "status": "identified",
            "reason": "identified_root_capture_frame",
            "root_inclusion_propensity_exact": _probability_text(inclusion),
            "root_inclusion_propensity": float(inclusion),
            "inclusion_source": inclusion_source,
            "atomic_capture_group_ids": list(frame.capable_group_ids),
            "second_order_ready": second_order_ready,
            "missing_second_order_pairs": pairwise_missing,
            "frame": frame.to_dict(),
            "warning": (
                "root-level inclusion is not the sum of event-level HT weights"
            ),
        }

    def readiness(self) -> dict[str, Any]:
        roots = sorted(
            set(self.frames)
            | {capture.root_id for capture in self.captures.values()}
        )
        inputs = {root_id: self.estimator_inputs(root_id) for root_id in roots}
        confirmed_roots = sorted(
            {
                capture.root_id
                for capture in self.captures.values()
                if capture.evidence_level == "independently_confirmed_real_root"
            }
        )
        candidate_roots = sorted(
            {
                capture.root_id
                for capture in self.captures.values()
                if capture.evidence_level == "candidate_root"
            }
        )
        return {
            "schema_version": "rlcmf-root-capture-readiness-v1",
            "root_clusterer_sha256": self.root_clusterer_sha256,
            "atomic_group_count": len(self.groups),
            "root_frame_count": len(self.frames),
            "capture_count": len(self.captures),
            "candidate_roots": candidate_roots,
            "independently_confirmed_real_roots": confirmed_roots,
            "identified_roots": sorted(
                root_id
                for root_id, payload in inputs.items()
                if payload["status"] == "identified"
            ),
            "not_identified_roots": {
                root_id: payload["reason"]
                for root_id, payload in inputs.items()
                if payload["status"] != "identified"
            },
            "root_inputs": inputs,
            "primary_root_efficiency_ready": bool(confirmed_roots)
            and all(
                inputs[root_id]["status"] == "identified"
                for root_id in confirmed_roots
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "rlcmf-root-capture-ledger-v1",
            "manifest_sha256": self.manifest_sha256,
            "root_clusterer_sha256": self.root_clusterer_sha256,
            "design_id": self.design_id,
            "groups": [
                self.groups[group_id].to_dict() for group_id in sorted(self.groups)
            ],
            "frames": [
                self.frames[root_id].to_dict() for root_id in sorted(self.frames)
            ],
            "captures": [
                self.captures[capture_id].to_dict()
                for capture_id in sorted(self.captures)
            ],
            "readiness": self.readiness(),
        }
