from __future__ import annotations

import hashlib
import importlib.util
import json
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
import sys
from typing import Any


def canonical_json_bytes(payload: Any) -> bytes:
    return json_canonical_dumps(payload).encode("utf-8")


def json_canonical_dumps(payload: Any) -> str:
    native = _load_native()
    if native is not None:
        try:
            return str(native.json_canonical_dumps(payload))
        except Exception:
            pass
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def short_sha256_hex(payload: Any, length: int = 16) -> str:
    native = _load_native()
    if native is not None:
        try:
            return str(native.short_sha256_hex(payload, length))
        except Exception:
            pass
    raw = json_canonical_dumps(payload).encode()
    return hashlib.sha256(raw).hexdigest()[:length]


def canonical_sort_key(payload: Any) -> str:
    native = _load_native()
    if native is not None:
        try:
            return str(native.canonical_sort_key(payload))
        except Exception:
            pass
    return json_canonical_dumps(payload)


def canonical_group_keys(payloads: Any) -> list[str]:
    native = _load_native()
    if native is not None:
        try:
            return [str(item) for item in native.canonical_group_keys(payloads)]
        except Exception:
            pass
    return [canonical_sort_key(payload) for payload in payloads]


def stable_rows(rows: Any) -> list[str]:
    native = _load_native()
    if native is not None:
        try:
            return [str(item) for item in native.stable_rows(rows)]
        except Exception:
            pass
    return [json_canonical_dumps(row) for row in rows]


def _result_field_fallback(result: Any, key: str, default: Any = "") -> Any:
    if isinstance(result, Mapping):
        return result.get(key, default)
    return getattr(result, key, default)


def _result_payload_fallback(result: Any) -> dict[str, Any]:
    return {
        "status": str(_result_field_fallback(result, "status", "")),
        "columns": [str(column) for column in list(_result_field_fallback(result, "columns", []) or [])],
        "rows": [list(row) for row in list(_result_field_fallback(result, "rows", []) or [])],
        "error_type": str(_result_field_fallback(result, "error_type", "")),
    }


def _row_set_profiles_fallback(row_sets: Any) -> list[tuple[str, str, bool]]:
    profiles: list[tuple[str, str, bool]] = []
    for rows in row_sets:
        stable = stable_rows(rows)
        ordered = json_canonical_dumps(stable)
        sorted_stable = sorted(stable)
        unordered = json_canonical_dumps(sorted_stable)
        has_duplicates = any(left == right for left, right in zip(sorted_stable, sorted_stable[1:]))
        profiles.append((ordered, unordered, has_duplicates))
    return profiles


def _group_ids_from_signatures_fallback(signatures: list[str]) -> list[int]:
    group_map: dict[str, int] = {}
    group_ids: list[int] = []
    for signature in signatures:
        if signature not in group_map:
            group_map[signature] = len(group_map)
        group_ids.append(group_map[signature])
    return group_ids


def _comparison_summary_from_group_ids_fallback(group_ids: list[int]) -> tuple[list[int], bool, list[int]]:
    groups: dict[int, list[int]] = {}
    for index, group_id in enumerate(group_ids):
        groups.setdefault(group_id, []).append(index)
    grouped = list(groups.values())
    if len(grouped) < 2:
        return [], False, []
    max_size = max(len(group) for group in grouped)
    max_groups = [group for group in grouped if len(group) == max_size]
    if len(max_groups) != 1:
        return list(range(len(group_ids))), False, []
    majority_group = list(max_groups[0])
    majority_lookup = set(majority_group)
    suspicious_indices = [index for index in range(len(group_ids)) if index not in majority_lookup]
    return suspicious_indices, True, majority_group


def _result_payloads_fallback(results: Any) -> list[dict[str, Any]]:
    return [_result_payload_fallback(result) for result in results]


def _result_profiles_fallback(results: Any) -> list[tuple[str, list[str], str, str, str, bool, str]]:
    payloads = _result_payloads_fallback(results)
    if not payloads:
        return []
    row_profiles = _row_set_profiles_fallback([payload["rows"] for payload in payloads])
    return [
        (
            payload["status"],
            payload["columns"],
            payload["error_type"],
            ordered_signature,
            unordered_signature,
            has_duplicates,
            json_canonical_dumps(
                {
                    "status": payload["status"],
                    "columns": payload["columns"],
                    "row_signature": ordered_signature,
                    "error_type": payload["error_type"],
                }
            ),
        )
        for payload, (ordered_signature, unordered_signature, has_duplicates) in zip(payloads, row_profiles)
    ]


def sorted_canonical_rows(rows: Any) -> list[str]:
    native = _load_native()
    if native is not None:
        try:
            return [str(item) for item in native.sorted_canonical_rows(rows)]
        except Exception:
            pass
    stable = stable_rows(rows)
    stable.sort()
    return stable


def dedupe_canonical_rows(rows: Any) -> list[str]:
    native = _load_native()
    if native is not None:
        try:
            return [str(item) for item in native.dedupe_canonical_rows(rows)]
        except Exception:
            pass
    seen: set[str] = set()
    out: list[str] = []
    for row in stable_rows(rows):
        if row in seen:
            continue
        seen.add(row)
        out.append(row)
    return out


def ordered_row_signature(rows: Any) -> str:
    native = _load_native()
    if native is not None:
        try:
            return str(native.ordered_row_signature(rows))
        except Exception:
            pass
    return json_canonical_dumps(stable_rows(rows))


def ordered_row_signatures(row_sets: Any) -> list[str]:
    native = _load_native()
    if native is not None:
        try:
            return [str(item) for item in native.ordered_row_signatures(row_sets)]
        except Exception:
            pass
    return [ordered for ordered, _, _ in _row_set_profiles_fallback(row_sets)]


def has_duplicate_rows(rows: Any) -> bool:
    native = _load_native()
    if native is not None:
        try:
            return bool(native.has_duplicate_rows(rows))
        except Exception:
            pass
    seen: set[str] = set()
    for row in stable_rows(rows):
        if row in seen:
            return True
        seen.add(row)
    return False


def unordered_row_signature(rows: Any) -> str:
    native = _load_native()
    if native is not None:
        try:
            return str(native.unordered_row_signature(rows))
        except Exception:
            pass
    return json_canonical_dumps(sorted_canonical_rows(rows))


def unordered_row_signatures(row_sets: Any) -> list[str]:
    native = _load_native()
    if native is not None:
        try:
            return [str(item) for item in native.unordered_row_signatures(row_sets)]
        except Exception:
            pass
    return [unordered for _, unordered, _ in _row_set_profiles_fallback(row_sets)]


def row_profile(rows: Any) -> tuple[str, str, bool]:
    native = _load_native()
    if native is not None:
        try:
            ordered, unordered, has_duplicates = native.row_profile(rows)
            return str(ordered), str(unordered), bool(has_duplicates)
        except Exception:
            pass
    stable = stable_rows(rows)
    ordered = json_canonical_dumps(stable)
    seen: set[str] = set()
    has_duplicates = False
    for row in stable:
        if row in seen:
            has_duplicates = True
            break
        seen.add(row)
    sorted_stable = sorted(stable)
    unordered = json_canonical_dumps(sorted_stable)
    return ordered, unordered, has_duplicates


def row_profiles(row_sets: Any) -> list[tuple[str, str, bool]]:
    native = _load_native()
    if native is not None:
        try:
            return [
                (str(ordered), str(unordered), bool(has_duplicates))
                for ordered, unordered, has_duplicates in native.row_profiles(row_sets)
            ]
        except Exception:
            pass
    return _row_set_profiles_fallback(row_sets)


def canonicalize_row_order(rows: Any) -> tuple[list[int], list[str], str, bool]:
    native = _load_native()
    if native is not None:
        try:
            indices, ordered_keys, ordered_signature, has_duplicates = native.canonicalize_row_order(rows)
            return (
                [int(index) for index in indices],
                [str(item) for item in ordered_keys],
                str(ordered_signature),
                bool(has_duplicates),
            )
        except Exception:
            pass
    stable = stable_rows(rows)
    ordered_pairs = sorted(enumerate(stable), key=lambda pair: pair[1])
    indices = [index for index, _ in ordered_pairs]
    ordered_keys = [key for _, key in ordered_pairs]
    ordered_signature = json_canonical_dumps(ordered_keys)
    has_duplicates = any(left == right for left, right in zip(ordered_keys, ordered_keys[1:]))
    return indices, ordered_keys, ordered_signature, has_duplicates


def row_set_profiles(row_sets: Any) -> list[tuple[str, str]]:
    native = _load_native()
    if native is not None:
        try:
            return [(str(ordered), str(unordered)) for ordered, unordered in native.row_set_profiles(row_sets)]
        except Exception:
            pass
    return [(ordered, unordered) for ordered, unordered, _ in row_profiles(row_sets)]


def multiset_row_diff(left_rows: Any, right_rows: Any) -> tuple[list[str], list[str]]:
    native = _load_native()
    if native is not None:
        try:
            left_only, right_only = native.multiset_row_diff(left_rows, right_rows)
            return [str(item) for item in left_only], [str(item) for item in right_only]
        except Exception:
            pass

    left_counts: dict[str, int] = {}
    right_counts: dict[str, int] = {}
    for row in stable_rows(left_rows):
        left_counts[row] = left_counts.get(row, 0) + 1
    for row in stable_rows(right_rows):
        right_counts[row] = right_counts.get(row, 0) + 1

    left_only: list[str] = []
    right_only: list[str] = []
    for row, count in left_counts.items():
        for _ in range(max(0, count - right_counts.get(row, 0))):
            left_only.append(row)
    for row, count in right_counts.items():
        for _ in range(max(0, count - left_counts.get(row, 0))):
            right_only.append(row)
    return left_only, right_only


def compare_row_sets_summary(left_rows: Any, right_rows: Any) -> tuple[str, bool, bool, bool, list[str], list[str]]:
    native = _load_native()
    if native is not None:
        try:
            mismatch_class, same_row_count, same_ordered_rows, same_unordered_rows, left_only, right_only = native.compare_row_sets_summary(
                left_rows,
                right_rows,
            )
            return (
                str(mismatch_class),
                bool(same_row_count),
                bool(same_ordered_rows),
                bool(same_unordered_rows),
                [str(item) for item in left_only],
                [str(item) for item in right_only],
            )
        except Exception:
            pass

    left_values = list(left_rows)
    right_values = list(right_rows)
    left_profile = row_profile(left_values)
    right_profile = row_profile(right_values)
    same_row_count = len(left_values) == len(right_values)
    if not same_row_count:
        return "row_count", False, False, False, [], []
    same_ordered_rows = left_profile[0] == right_profile[0]
    if same_ordered_rows:
        return "none", True, True, True, [], []
    same_unordered_rows = left_profile[1] == right_profile[1]
    if same_unordered_rows:
        return "row_order", True, False, True, [], []
    left_only, right_only = multiset_row_diff(left_values, right_values)
    return "value", True, False, False, left_only, right_only


def compare_row_sets(left_rows: Any, right_rows: Any) -> tuple[str, bool, bool, list[str], list[str]]:
    mismatch_class, same_row_count, same_ordered_rows, _, left_only, right_only = compare_row_sets_summary(
        left_rows,
        right_rows,
    )
    return mismatch_class, same_row_count, same_ordered_rows, left_only, right_only


def compare_row_set_batch(row_sets: Any) -> tuple[list[int], str]:
    native = _load_native()
    if native is not None:
        try:
            group_ids, mismatch_class = native.compare_row_set_batch(row_sets)
            return [int(item) for item in group_ids], str(mismatch_class)
        except Exception:
            pass

    values = [list(rows) for rows in row_sets]
    if not values:
        return [], "none"
    profiles = _row_set_profiles_fallback(values)
    ordered_signatures = [ordered for ordered, _, _ in profiles]
    group_ids = _group_ids_from_signatures_fallback(ordered_signatures)
    row_counts = [len(rows) for rows in values]
    if len(set(row_counts)) > 1:
        return group_ids, "row_count"
    if len(set(ordered_signatures)) == 1:
        return group_ids, "none"
    unordered_signatures = [unordered for _, unordered, _ in profiles]
    if len(set(unordered_signatures)) == 1:
        return group_ids, "row_order"
    return group_ids, "value"


def compare_row_set_batch_summary(row_sets: Any) -> tuple[list[int], str, list[int], bool, list[int]]:
    native = _load_native()
    if native is not None:
        try:
            group_ids, mismatch_class, suspicious_indices, has_clear_majority, majority_group = native.compare_row_set_batch_summary(
                row_sets
            )
            return (
                [int(item) for item in group_ids],
                str(mismatch_class),
                [int(item) for item in suspicious_indices],
                bool(has_clear_majority),
                [int(item) for item in majority_group],
            )
        except Exception:
            pass

    group_ids, mismatch_class = compare_row_set_batch(row_sets)
    suspicious_indices, has_clear_majority, majority_group = _comparison_summary_from_group_ids_fallback(group_ids)
    return group_ids, mismatch_class, suspicious_indices, has_clear_majority, majority_group


def compare_result_batch(results: Any) -> tuple[list[int], str]:
    payloads = _result_payloads_fallback(results)
    native = _load_native()
    if native is not None:
        try:
            group_ids, mismatch_class = native.compare_result_batch(payloads)
            return [int(item) for item in group_ids], str(mismatch_class)
        except Exception:
            pass
    profiles = _result_profiles_fallback(payloads)
    if not profiles:
        return [], "none"
    comparison_keys = [profile[6] for profile in profiles]
    group_ids = _group_ids_from_signatures_fallback(comparison_keys)
    statuses = {profile[0] for profile in profiles}
    if len(statuses) > 1:
        return group_ids, "status"
    first_status = profiles[0][0]
    if first_status != "ok":
        error_types = {profile[2] for profile in profiles}
        if len(error_types) > 1:
            return group_ids, "error_type"
        mismatch_class = "none" if len(set(group_ids)) == 1 else "status"
        return group_ids, mismatch_class
    columns = [tuple(profile[1]) for profile in profiles]
    if len(set(columns)) > 1:
        return group_ids, "schema"
    row_groups, mismatch_class = compare_row_set_batch([payload["rows"] for payload in payloads])
    if row_groups != group_ids:
        group_ids = row_groups
    return group_ids, mismatch_class


def compare_result_batch_summary(results: Any) -> tuple[list[int], str, list[int], bool, list[int]]:
    payloads = _result_payloads_fallback(results)
    native = _load_native()
    if native is not None:
        try:
            group_ids, mismatch_class, suspicious_indices, has_clear_majority, majority_group = native.compare_result_batch_summary(
                payloads
            )
            return (
                [int(item) for item in group_ids],
                str(mismatch_class),
                [int(item) for item in suspicious_indices],
                bool(has_clear_majority),
                [int(item) for item in majority_group],
            )
        except Exception:
            pass
    group_ids, mismatch_class = compare_result_batch(results)
    suspicious_indices, has_clear_majority, majority_group = _comparison_summary_from_group_ids_fallback(group_ids)
    return group_ids, mismatch_class, suspicious_indices, has_clear_majority, majority_group


def compare_result_batch_anchor_summary(results: Any, anchor_index: int) -> tuple[list[int], list[int], str]:
    payloads = _result_payloads_fallback(results)
    native = _load_native()
    if native is not None:
        try:
            matching_indices, mismatching_indices, confidence = native.compare_result_batch_anchor_summary(
                payloads,
                int(anchor_index),
            )
            return (
                [int(item) for item in matching_indices],
                [int(item) for item in mismatching_indices],
                str(confidence),
            )
        except Exception:
            pass
    group_ids, _mismatch_class = compare_result_batch(results)
    if anchor_index < 0 or anchor_index >= len(group_ids):
        raise IndexError("anchor_index out of range")
    anchor_group = group_ids[anchor_index]
    matching_indices = [index for index, group_id in enumerate(group_ids) if group_id == anchor_group]
    mismatching_indices = [index for index, group_id in enumerate(group_ids) if group_id != anchor_group]
    confidence = "high" if not mismatching_indices or len(mismatching_indices) < max(1, len(group_ids) - 1) else "medium"
    return matching_indices, mismatching_indices, confidence


def profile_result_batch(results: Any) -> list[tuple[str, list[str], str, str, str, bool, str]]:
    payloads = _result_payloads_fallback(results)
    native = _load_native()
    if native is not None:
        try:
            return [
                (
                    str(status),
                    [str(column) for column in columns],
                    str(error_type),
                    str(ordered_signature),
                    str(unordered_signature),
                    bool(has_duplicates),
                    str(comparison_key),
                )
                for status, columns, error_type, ordered_signature, unordered_signature, has_duplicates, comparison_key in native.profile_result_batch(payloads)
            ]
        except Exception:
            pass
    return _result_profiles_fallback(payloads)


def native_available() -> bool:
    return _load_native() is not None


@lru_cache(maxsize=1)
def _load_native() -> Any | None:
    try:
        from datadiff._rust_kernel import rust_kernel as native

        return native
    except Exception:
        pass

    for candidate in _local_native_candidates():
        try:
            spec = importlib.util.spec_from_file_location("rust_kernel", candidate)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules.setdefault("rust_kernel", module)
            spec.loader.exec_module(module)
            return module
        except Exception:
            continue
    return None


def _local_native_candidates() -> list[Path]:
    root = Path(__file__).resolve().parents[2]
    candidates: list[Path] = []
    native_dir = root / "rust_kernel" / "python" / "datadiff" / "_rust_kernel"
    for pattern in ("rust_kernel*.so", "rust_kernel*.pyd", "rust_kernel*.dylib"):
        candidates.extend(native_dir.glob(pattern))
    target_dir = root / "rust_kernel" / "target" / "release"
    candidates.extend(
        [
            target_dir / "librust_kernel.so",
            target_dir / "librust_kernel.dylib",
            target_dir / "rust_kernel.pyd",
            target_dir / "deps" / "librust_kernel.so",
            target_dir / "deps" / "librust_kernel.dylib",
            target_dir / "deps" / "rust_kernel.pyd",
        ]
    )
    return [path for path in candidates if path.is_file()]
