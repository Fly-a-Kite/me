from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from datadiff.canonicalization import short_canonical_hash
from datadiff.dsl import Case
from datadiff.normalizer import NormalizedResult
from datadiff.operation_semantics import operation_names
from datadiff.rust_kernel import compute_minhash

MINHASH_SIGNATURE_SIZE = 64


@dataclass(frozen=True, slots=True)
class CaseFingerprint:
    minhash_signature: tuple[int, ...]
    op_skeleton_hash: str
    type_mix_token: str
    null_density_bucket: int
    row_mass_bucket: int
    column_count: int

    def feature_tokens(self) -> tuple[str, ...]:
        return (
            f"fp_op:{self.op_skeleton_hash}",
            f"fp_type_mix:{self.type_mix_token}",
            f"fp_null_density:{self.null_density_bucket}",
            f"fp_row_mass:{self.row_mass_bucket}",
            f"fp_column_count:{_column_count_bucket(self.column_count)}",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "minhash_signature": list(self.minhash_signature),
            "op_skeleton_hash": self.op_skeleton_hash,
            "type_mix_token": self.type_mix_token,
            "null_density_bucket": self.null_density_bucket,
            "row_mass_bucket": self.row_mass_bucket,
            "column_count": self.column_count,
            "feature_tokens": list(self.feature_tokens()),
        }

    @classmethod
    def from_dict(cls, data: Any) -> "CaseFingerprint":
        if not isinstance(data, Mapping):
            return empty_fingerprint()
        return cls(
            minhash_signature=tuple(
                int(item)
                for item in list(data.get("minhash_signature", []) or [])
            ),
            op_skeleton_hash=str(data.get("op_skeleton_hash", "") or ""),
            type_mix_token=str(data.get("type_mix_token", "") or "none"),
            null_density_bucket=_bucket_int(data.get("null_density_bucket", 0)),
            row_mass_bucket=_bucket_int(data.get("row_mass_bucket", 0)),
            column_count=max(0, int(data.get("column_count", 0) or 0)),
        )


def empty_fingerprint() -> CaseFingerprint:
    return CaseFingerprint(
        minhash_signature=tuple(0 for _ in range(MINHASH_SIGNATURE_SIZE)),
        op_skeleton_hash="",
        type_mix_token="none",
        null_density_bucket=0,
        row_mass_bucket=0,
        column_count=0,
    )


def compute_fingerprint(
    case: Case,
    normalized_anchor: NormalizedResult | Mapping[str, Any] | None,
) -> CaseFingerprint:
    anchor = _coerce_anchor(normalized_anchor)
    op_sequence = operation_names(case.program.operations, default="unknown")
    type_mix = _type_mix_token(case)
    null_density_bucket = _null_density_bucket(case, anchor)
    row_mass_bucket = _row_mass_bucket(case, anchor)
    column_count = len(anchor.columns) if anchor is not None else _input_column_count(case)
    tokens = _fingerprint_tokens(
        case,
        anchor,
        op_sequence=op_sequence,
        type_mix_token=type_mix,
        null_density_bucket=null_density_bucket,
        row_mass_bucket=row_mass_bucket,
        column_count=column_count,
    )
    return CaseFingerprint(
        minhash_signature=tuple(compute_minhash(tokens, MINHASH_SIGNATURE_SIZE)),
        op_skeleton_hash=short_canonical_hash(
            {"ops": op_sequence},
            16,
        ),
        type_mix_token=type_mix,
        null_density_bucket=null_density_bucket,
        row_mass_bucket=row_mass_bucket,
        column_count=column_count,
    )


def jaccard_distance(left: CaseFingerprint | Mapping[str, Any], right: CaseFingerprint | Mapping[str, Any]) -> float:
    left_fp = left if isinstance(left, CaseFingerprint) else CaseFingerprint.from_dict(left)
    right_fp = right if isinstance(right, CaseFingerprint) else CaseFingerprint.from_dict(right)
    left_sig = tuple(left_fp.minhash_signature)
    right_sig = tuple(right_fp.minhash_signature)
    if not left_sig and not right_sig:
        return 0.0
    if len(left_sig) != len(right_sig) or not left_sig or not right_sig:
        return 1.0
    matches = sum(1 for left_value, right_value in zip(left_sig, right_sig) if left_value == right_value)
    return 1.0 - (matches / float(len(left_sig)))


def _coerce_anchor(value: NormalizedResult | Mapping[str, Any] | None) -> NormalizedResult | None:
    if value is None:
        return None
    if isinstance(value, NormalizedResult):
        return value
    if isinstance(value, Mapping):
        return NormalizedResult.from_dict(value)
    return None


def _fingerprint_tokens(
    case: Case,
    anchor: NormalizedResult | None,
    *,
    op_sequence: Sequence[str],
    type_mix_token: str,
    null_density_bucket: int,
    row_mass_bucket: int,
    column_count: int,
) -> list[str]:
    tokens = [
        f"ops:{'|'.join(op_sequence) if op_sequence else 'empty'}",
        f"type_mix:{type_mix_token}",
        f"null_density:{null_density_bucket}",
        f"row_mass:{row_mass_bucket}",
        f"column_count:{_column_count_bucket(column_count)}",
    ]
    for index, op_name in enumerate(op_sequence):
        tokens.append(f"op:{index}:{op_name}")
        tokens.append(f"op_any:{op_name}")
    for table in case.tables:
        tokens.append(f"table:{table.name}:cols:{len(table.columns)}")
        for column in table.columns:
            tokens.append(f"input_col:{column.name}:{column.type}:{int(column.nullable)}")
    if anchor is None:
        return tokens
    tokens.append(f"anchor_status:{anchor.status}")
    if anchor.error_type:
        tokens.append(f"anchor_error:{anchor.error_type}")
    for column in anchor.columns:
        tokens.append(f"output_col:{column}")
    for row_key in sorted(anchor.stable_row_keys):
        tokens.append(f"row:{row_key}")
    return tokens


def _type_mix_token(case: Case) -> str:
    counts: Counter[str] = Counter()
    for table in case.tables:
        for column in table.columns:
            if column.type in {"int", "float"}:
                counts["num"] += 1
            elif column.type == "str":
                counts["str"] += 1
            elif column.type == "bool":
                counts["bool"] += 1
            else:
                counts["other"] += 1
    parts = [f"{name}{counts[name]}" for name in ("num", "str", "bool", "other") if counts[name]]
    return "_".join(parts) if parts else "none"


def _null_density_bucket(case: Case, anchor: NormalizedResult | None) -> int:
    total = 0
    nulls = 0
    for table in case.tables:
        for row in table.rows:
            for value in row.values():
                total += 1
                if value is None:
                    nulls += 1
    if anchor is not None and anchor.status == "ok":
        for row in anchor.rows:
            if not isinstance(row, Sequence) or isinstance(row, (str, bytes, bytearray)):
                continue
            for value in row:
                total += 1
                if value is None:
                    nulls += 1
    if total <= 0:
        return 0
    return min(7, int((nulls / float(total)) * 8.0))


def _row_mass_bucket(case: Case, anchor: NormalizedResult | None) -> int:
    row_count = sum(len(table.rows) for table in case.tables)
    if anchor is not None and anchor.status == "ok":
        row_count += len(anchor.rows)
    if row_count <= 0:
        return 0
    if row_count <= 1:
        return 1
    if row_count <= 3:
        return 2
    if row_count <= 7:
        return 3
    if row_count <= 15:
        return 4
    if row_count <= 31:
        return 5
    if row_count <= 127:
        return 6
    return 7


def _input_column_count(case: Case) -> int:
    return sum(len(table.columns) for table in case.tables)


def _column_count_bucket(column_count: int) -> int:
    return min(7, max(0, int(column_count or 0)))


def _bucket_int(value: Any) -> int:
    return min(7, max(0, int(value or 0)))
