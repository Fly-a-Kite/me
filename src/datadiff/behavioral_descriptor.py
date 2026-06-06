from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datadiff.dsl import Case
from datadiff.fingerprint import CaseFingerprint, compute_fingerprint


@dataclass(frozen=True, slots=True)
class BehavioralDescriptor:
    profile_axis: str
    op_skeleton_axis: str
    target_class_axis: str
    null_density_axis: str
    type_mix_axis: str
    row_mass_axis: str
    column_count_axis: str
    backend_disagreement_axis: str = "pair:none"

    def axis_tuples(self) -> tuple[tuple[str, str], ...]:
        return (
            ("bd_profile", self.profile_axis),
            ("bd_op_skeleton", self.op_skeleton_axis),
            ("bd_target_class", self.target_class_axis),
            ("bd_null_density", self.null_density_axis),
            ("bd_type_mix", self.type_mix_axis),
            ("bd_row_mass", self.row_mass_axis),
            ("bd_column_count", self.column_count_axis),
            ("bd_backend_disagreement", self.backend_disagreement_axis),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_axis": self.profile_axis,
            "op_skeleton_axis": self.op_skeleton_axis,
            "target_class_axis": self.target_class_axis,
            "null_density_axis": self.null_density_axis,
            "type_mix_axis": self.type_mix_axis,
            "row_mass_axis": self.row_mass_axis,
            "column_count_axis": self.column_count_axis,
            "backend_disagreement_axis": self.backend_disagreement_axis,
            "axis_tuples": [list(item) for item in self.axis_tuples()],
        }

    @classmethod
    def from_dict(cls, data: Any) -> "BehavioralDescriptor":
        if not isinstance(data, dict):
            return cls.empty()
        return cls(
            profile_axis=_axis_text(data.get("profile_axis"), "generic"),
            op_skeleton_axis=_axis_text(data.get("op_skeleton_axis"), "op:none"),
            target_class_axis=_axis_text(data.get("target_class_axis"), "target:none"),
            null_density_axis=_axis_text(data.get("null_density_axis"), "null:0"),
            type_mix_axis=_axis_text(data.get("type_mix_axis"), "type:none"),
            row_mass_axis=_axis_text(data.get("row_mass_axis"), "rows:0"),
            column_count_axis=_axis_text(data.get("column_count_axis"), "cols:0"),
            backend_disagreement_axis=_axis_text(data.get("backend_disagreement_axis"), "pair:none"),
        )

    @classmethod
    def empty(cls) -> "BehavioralDescriptor":
        return cls(
            profile_axis="generic",
            op_skeleton_axis="op:none",
            target_class_axis="target:none",
            null_density_axis="null:0",
            type_mix_axis="type:none",
            row_mass_axis="rows:0",
            column_count_axis="cols:0",
            backend_disagreement_axis="pair:none",
        )


def compute_behavioral_descriptor(
    case: Case,
    *,
    profile_key: str = "",
    target_keys: list[str] | tuple[str, ...] = (),
) -> BehavioralDescriptor:
    fingerprint = _case_fingerprint(case)
    return BehavioralDescriptor(
        profile_axis=_axis_token(profile_key or _case_profile_key(case) or "generic"),
        op_skeleton_axis=f"op:{_axis_token(fingerprint.op_skeleton_hash or 'none')}",
        target_class_axis=f"target:{_axis_token(_target_class_axis(target_keys))}",
        null_density_axis=f"null:{int(fingerprint.null_density_bucket)}",
        type_mix_axis=f"type:{_axis_token(fingerprint.type_mix_token or 'none')}",
        row_mass_axis=f"rows:{int(fingerprint.row_mass_bucket)}",
        column_count_axis=f"cols:{_column_count_bucket(int(fingerprint.column_count))}",
        backend_disagreement_axis=f"pair:{_axis_token(_backend_disagreement_axis(case))}",
    )


def _case_fingerprint(case: Case) -> CaseFingerprint:
    metadata = case.metadata if isinstance(case.metadata, dict) else {}
    payload = metadata.get("case_fingerprint")
    if isinstance(payload, dict):
        return CaseFingerprint.from_dict(payload)
    return compute_fingerprint(case, None)


def _case_profile_key(case: Case) -> str:
    metadata = case.metadata if isinstance(case.metadata, dict) else {}
    for key in ("mixed_generator_profile", "generator_profile"):
        value = str(metadata.get(key, "") or "").strip()
        if value:
            return value
    return ""


def _backend_disagreement_axis(case: Case) -> str:
    metadata = case.metadata if isinstance(case.metadata, dict) else {}
    descriptor = metadata.get("disagreement_descriptor")
    if not isinstance(descriptor, dict):
        return "none"
    pairs: list[str] = []
    raw_pairs = descriptor.get("pair_disagrees", [])
    if isinstance(raw_pairs, dict):
        for key, disagrees in raw_pairs.items():
            if bool(disagrees):
                pair = _normalize_pair_key(str(key))
                if pair:
                    pairs.append(pair)
    elif isinstance(raw_pairs, list):
        for item in raw_pairs:
            if not isinstance(item, dict) or not bool(item.get("disagrees")):
                continue
            pair = _normalize_pair_key(f"{item.get('left', '')}|{item.get('right', '')}")
            if pair:
                pairs.append(pair)
    if not pairs:
        mismatch = str(descriptor.get("mismatch_class", "") or "").strip()
        return f"mismatch:{mismatch}" if mismatch and mismatch != "none" else "none"
    return "__".join(sorted(dict.fromkeys(pairs))[:4])


def _normalize_pair_key(value: str) -> str:
    parts = [part.strip() for part in str(value or "").split("|", 1)]
    if len(parts) != 2 or not parts[0] or not parts[1] or parts[0] == parts[1]:
        return ""
    left, right = sorted(parts)
    return f"{left}|{right}"


def _target_class_axis(target_keys: list[str] | tuple[str, ...]) -> str:
    normalized = [str(item).strip() for item in target_keys if str(item).strip()]
    if not normalized:
        return "none"
    for prefix in ("semantic_family:", "semantic_signal:", "capability:", "exploration_objective:", "target:"):
        for item in normalized:
            if item.startswith(prefix):
                return item
    return normalized[0]


def _axis_token(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return "none"
    out = []
    for char in text:
        out.append(char if char.isalnum() else "_")
    return "_".join(part for part in "".join(out).split("_") if part) or "none"


def _axis_text(value: Any, default: str) -> str:
    text = str(value or "").strip()
    return text if text else default


def _column_count_bucket(column_count: int) -> int:
    value = max(0, int(column_count or 0))
    if value <= 0:
        return 0
    if value <= 2:
        return 1
    if value <= 4:
        return 2
    if value <= 8:
        return 3
    return 4
