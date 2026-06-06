from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

from datadiff.rust_kernel import (
    canonicalize_row_order,
    canonical_group_keys,
    canonical_sort_key,
    compare_result_batch_anchor_summary as kernel_compare_result_batch_anchor_summary,
    compare_result_batch_summary as kernel_compare_result_batch_summary,
    compare_row_set_batch_summary as kernel_compare_row_set_batch_summary,
    compare_row_sets_summary as kernel_compare_row_sets_summary,
    dedupe_canonical_rows,
    profile_result_batch as kernel_profile_result_batch,
    row_profile,
    row_profiles,
    short_sha256_hex,
    sorted_canonical_rows,
    stable_rows,
    multiset_row_diff,
)

_T = TypeVar("_T")


@dataclass(frozen=True, slots=True)
class RowSetProfile:
    ordered_signature: str
    unordered_signature: str
    has_duplicates: bool


@dataclass(frozen=True, slots=True)
class CanonicalizedRows:
    rows: list[Any]
    stable_row_keys: list[str]
    profile: RowSetProfile


@dataclass(frozen=True, slots=True)
class _SingleComparisonView:
    mismatch_class: str
    same_schema: bool
    same_row_count: bool
    same_ordered_rows: bool
    same_unordered_rows: bool
    left_only: list[str]
    right_only: list[str]

    @property
    def has_mismatch(self) -> bool:
        return self.mismatch_class != "none"

    def is_exact_match(self) -> bool:
        return self.mismatch_class == "none"

    def is_order_only_mismatch(self) -> bool:
        return self.mismatch_class == "row_order"

    def diff_size(self) -> int:
        return len(self.left_only) + len(self.right_only)

    def has_value_diff(self) -> bool:
        return bool(self.left_only or self.right_only)


@dataclass(frozen=True, slots=True)
class RowComparison(_SingleComparisonView):
    pass


@dataclass(frozen=True, slots=True)
class ResultProfile:
    status: str
    columns: tuple[str, ...]
    error_type: str
    row_profile: RowSetProfile
    comparison_key: str

    @property
    def ordered_row_signature(self) -> str:
        return self.row_profile.ordered_signature

    @property
    def unordered_row_signature(self) -> str:
        return self.row_profile.unordered_signature

    @property
    def has_duplicate_rows(self) -> bool:
        return self.row_profile.has_duplicates


@dataclass(frozen=True, slots=True)
class _BatchComparisonView:
    group_ids: list[int]
    mismatch_class: str
    suspicious_indices_value: list[int] | None = None
    has_clear_majority_value: bool | None = None
    majority_group_value: list[int] | None = None

    @property
    def groups(self) -> list[list[int]]:
        return _groups_from_group_ids(self.group_ids)

    @property
    def group_sizes(self) -> list[int]:
        return _group_sizes_from_group_ids(self.group_ids)

    @property
    def has_mismatch(self) -> bool:
        return len(self.groups) > 1

    @property
    def has_clear_majority(self) -> bool:
        if self.has_clear_majority_value is not None:
            return self.has_clear_majority_value
        return _has_clear_majority_group(self.group_ids)

    @property
    def majority_group(self) -> list[int] | None:
        if self.majority_group_value is not None:
            if not self.majority_group_value:
                return None
            return list(self.majority_group_value)
        return _majority_group_from_group_ids(self.group_ids)

    @property
    def suspicious_indices(self) -> list[int]:
        if self.suspicious_indices_value is not None:
            return list(self.suspicious_indices_value)
        return _suspicious_indices_from_group_ids(self.group_ids)

    def is_exact_match(self) -> bool:
        return self.mismatch_class == "none"

    def is_order_only_mismatch(self) -> bool:
        return self.mismatch_class == "row_order"

    def labels_for_indices(self, labels: Sequence[str], indices: Sequence[int]) -> list[str]:
        if len(labels) != len(self.group_ids):
            raise ValueError("labels must match comparison length")
        return [str(labels[index]) for index in indices]

    def suspicious_labels(self, labels: Sequence[str]) -> list[str]:
        return self.labels_for_indices(labels, self.suspicious_indices)

    def default_confidence(self) -> str:
        return "high" if self.has_clear_majority else "medium"

    def has_actionable_minority(self) -> bool:
        return bool(self.suspicious_indices) and self.has_clear_majority


@dataclass(frozen=True, slots=True)
class BatchRowComparison(_BatchComparisonView):
    pass


@dataclass(frozen=True, slots=True)
class BatchResultComparison(_BatchComparisonView):
    def matching_indices(self, anchor_index: int) -> list[int]:
        if anchor_index < 0 or anchor_index >= len(self.group_ids):
            raise IndexError("anchor_index out of range")
        anchor_group = self.group_ids[anchor_index]
        return [index for index, group_id in enumerate(self.group_ids) if group_id == anchor_group]

    def mismatching_indices(self, anchor_index: int) -> list[int]:
        if anchor_index < 0 or anchor_index >= len(self.group_ids):
            raise IndexError("anchor_index out of range")
        anchor_group = self.group_ids[anchor_index]
        return [index for index, group_id in enumerate(self.group_ids) if group_id != anchor_group]

    def matching_labels(self, anchor_index: int, labels: Sequence[str]) -> list[str]:
        return self.labels_for_indices(labels, self.matching_indices(anchor_index))

    def mismatching_labels(self, anchor_index: int, labels: Sequence[str]) -> list[str]:
        return self.labels_for_indices(labels, self.mismatching_indices(anchor_index))

    def anchor_confidence(self, anchor_index: int) -> str:
        mismatching = self.mismatching_indices(anchor_index)
        if not mismatching:
            return "high"
        return "high" if len(mismatching) < max(1, len(self.group_ids) - 1) else "medium"

    def anchor_matching_labels(
        self,
        anchor_index: int,
        labels: Sequence[str],
        *,
        exclude_anchor: bool = False,
    ) -> list[str]:
        values = self.matching_labels(anchor_index, labels)
        if not exclude_anchor:
            return values
        if anchor_index >= len(labels):
            raise IndexError("anchor_index out of range")
        anchor_label = str(labels[anchor_index])
        return [label for label in values if label != anchor_label]

    def anchor_mismatch_summary(
        self,
        anchor_index: int,
        labels: Sequence[str],
        *,
        exclude_anchor_from_matches: bool = False,
    ) -> tuple[list[str], list[str], str]:
        return (
            self.anchor_matching_labels(anchor_index, labels, exclude_anchor=exclude_anchor_from_matches),
            self.mismatching_labels(anchor_index, labels),
            self.anchor_confidence(anchor_index),
        )


@dataclass(frozen=True, slots=True)
class AnchorResultComparison:
    anchor_index: int
    matching_indices: list[int]
    mismatching_indices: list[int]
    confidence: str

    def matching_labels(
        self,
        labels: Sequence[str],
        *,
        exclude_anchor: bool = False,
    ) -> list[str]:
        if len(labels) <= self.anchor_index:
            raise IndexError("anchor_index out of range")
        values = [str(labels[index]) for index in self.matching_indices]
        if not exclude_anchor:
            return values
        anchor_label = str(labels[self.anchor_index])
        return [label for label in values if label != anchor_label]

    def mismatching_labels(self, labels: Sequence[str]) -> list[str]:
        return [str(labels[index]) for index in self.mismatching_indices]

    def label_summary(
        self,
        labels: Sequence[str],
        *,
        exclude_anchor_from_matches: bool = False,
    ) -> tuple[list[str], list[str], str]:
        return (
            self.matching_labels(labels, exclude_anchor=exclude_anchor_from_matches),
            self.mismatching_labels(labels),
            self.confidence,
        )


def _group_ids_from_keys(keys: Sequence[str]) -> list[int]:
    groups: dict[str, int] = {}
    group_ids: list[int] = []
    for key in keys:
        group_ids.append(groups.setdefault(key, len(groups)))
    return group_ids


def _groups_from_group_ids(group_ids: Sequence[int]) -> list[list[int]]:
    grouped: dict[int, list[int]] = {}
    for index, group_id in enumerate(group_ids):
        grouped.setdefault(group_id, []).append(index)
    return list(grouped.values())


def _group_sizes_from_group_ids(group_ids: Sequence[int]) -> list[int]:
    return [len(group) for group in _groups_from_group_ids(group_ids)]


def _has_clear_majority_group(group_ids: Sequence[int]) -> bool:
    if len(group_ids) < 2:
        return False
    sizes = _group_sizes_from_group_ids(group_ids)
    if len(sizes) < 2:
        return False
    max_size = max(sizes)
    return sizes.count(max_size) == 1


def _majority_group_from_group_ids(group_ids: Sequence[int]) -> list[int] | None:
    if not _has_clear_majority_group(group_ids):
        return None
    groups = _groups_from_group_ids(group_ids)
    return max(groups, key=len)


def _suspicious_indices_from_group_ids(group_ids: Sequence[int]) -> list[int]:
    groups = _groups_from_group_ids(group_ids)
    if len(groups) < 2:
        return []
    majority_group = _majority_group_from_group_ids(group_ids)
    if majority_group is None:
        return list(range(len(group_ids)))
    majority = set(majority_group)
    return [index for index in range(len(group_ids)) if index not in majority]

def canonical_key(payload: Any) -> str:
    return canonical_sort_key(payload)


def canonical_keys(payloads: Iterable[Any]) -> list[str]:
    items = list(payloads)
    if not items:
        return []
    return canonical_group_keys(items)


def mapping_projection_payload(
    row: Mapping[str, Any],
    columns: Sequence[str],
    *,
    normalize_value: Callable[[Any], Any] | None = None,
) -> list[Any]:
    if normalize_value is None:
        return [row.get(column) for column in columns]
    return [normalize_value(row.get(column)) for column in columns]


def mapping_projection_key(
    row: Mapping[str, Any],
    columns: Sequence[str],
    *,
    normalize_value: Callable[[Any], Any] | None = None,
) -> str:
    return canonical_key(mapping_projection_payload(row, columns, normalize_value=normalize_value))


def sort_by_canonical_key(
    items: Iterable[_T],
    *,
    payload: Callable[[_T], Any] | None = None,
) -> list[_T]:
    values = list(items)
    if len(values) < 2:
        return list(values)
    payload_fn = payload or (lambda item: item)
    keys = canonical_keys(payload_fn(item) for item in values)
    return [item for key, item in sorted(zip(keys, values), key=lambda pair: pair[0])]


def dedupe_by_canonical_key(
    items: Iterable[_T],
    *,
    payload: Callable[[_T], Any] | None = None,
) -> list[_T]:
    values = list(items)
    if not values:
        return []
    payload_fn = payload or (lambda item: item)
    keys = canonical_keys(payload_fn(item) for item in values)
    seen: set[str] = set()
    out: list[_T] = []
    for item, key in zip(values, keys):
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def has_duplicate_canonical_key(
    items: Iterable[_T],
    *,
    payload: Callable[[_T], Any] | None = None,
) -> bool:
    values = list(items)
    if len(values) < 2:
        return False
    payload_fn = payload or (lambda item: item)
    keys = canonical_keys(payload_fn(item) for item in values)
    return len(set(keys)) != len(keys)


def sorted_canonical_keys(payloads: Iterable[Any]) -> list[str]:
    items = list(payloads)
    if not items:
        return []
    return sorted_canonical_rows(items)


def short_canonical_hash(payload: Any, length: int = 16) -> str:
    return short_sha256_hex(payload, length)


def result_comparison_key(
    *,
    status: str,
    columns: Sequence[str],
    ordered_row_signature: str,
    error_type: str = "",
) -> str:
    return canonical_key(
        {
            "status": status,
            "columns": list(columns),
            "row_signature": ordered_row_signature,
            "error_type": error_type,
        }
    )


def stable_canonical_rows(rows: Iterable[Any]) -> list[str]:
    return stable_rows(list(rows))


def canonicalize_rows(rows: Iterable[Any]) -> CanonicalizedRows:
    values = list(rows)
    if not values:
        empty_signature = canonical_key([])
        return CanonicalizedRows(
            rows=[],
            stable_row_keys=[],
            profile=RowSetProfile(
                ordered_signature=empty_signature,
                unordered_signature=empty_signature,
                has_duplicates=False,
            ),
        )
    order_indices, ordered_keys, ordered_signature, has_duplicates = canonicalize_row_order(values)
    ordered_rows = [values[index] for index in order_indices]
    return CanonicalizedRows(
        rows=ordered_rows,
        stable_row_keys=ordered_keys,
        profile=RowSetProfile(
            ordered_signature=ordered_signature,
            unordered_signature=ordered_signature,
            has_duplicates=has_duplicates,
        ),
    )


def profile_rows(rows: Iterable[Any]) -> RowSetProfile:
    return profile_row_sets([rows])[0]


def profile_row_sets(row_sets: Iterable[Any]) -> list[RowSetProfile]:
    items = list(row_sets)
    if not items:
        return []
    return [
        RowSetProfile(
            ordered_signature=ordered,
            unordered_signature=unordered,
            has_duplicates=has_duplicates,
        )
        for ordered, unordered, has_duplicates in row_profiles(items)
    ]


def profile_result(
    *,
    status: str,
    columns: Sequence[str],
    rows: Iterable[Any],
    error_type: str = "",
    row_profile_value: RowSetProfile | None = None,
) -> ResultProfile:
    normalized_columns = tuple(str(column) for column in columns)
    profile = row_profile_value or profile_rows(rows)
    return ResultProfile(
        status=str(status),
        columns=normalized_columns,
        error_type=str(error_type),
        row_profile=profile,
        comparison_key=result_comparison_key(
            status=str(status),
            columns=normalized_columns,
            ordered_row_signature=profile.ordered_signature,
            error_type=str(error_type),
        ),
    )


def profile_result_payload(result: Mapping[str, Any] | Any) -> ResultProfile:
    return profile_result_payloads([result])[0]


def profile_result_payloads(results: Iterable[Mapping[str, Any] | Any]) -> list[ResultProfile]:
    items = list(results)
    if not items:
        return []
    native_profiles = kernel_profile_result_batch(items)
    return [
        ResultProfile(
            status=status,
            columns=tuple(columns),
            error_type=error_type,
            row_profile=RowSetProfile(
                ordered_signature=ordered_signature,
                unordered_signature=unordered_signature,
                has_duplicates=has_duplicates,
            ),
            comparison_key=comparison_key,
        )
        for status, columns, error_type, ordered_signature, unordered_signature, has_duplicates, comparison_key in native_profiles
    ]


def sorted_row_signatures(row_sets: Iterable[Any]) -> list[tuple[str, str]]:
    profiles = profile_row_sets(row_sets)
    return [(profile.ordered_signature, profile.unordered_signature) for profile in profiles]


def ordered_signature(rows: Iterable[Any]) -> str:
    return ordered_signatures([rows])[0]


def ordered_signatures(row_sets: Iterable[Any]) -> list[str]:
    return [profile.ordered_signature for profile in profile_row_sets(row_sets)]


def unordered_signature(rows: Iterable[Any]) -> str:
    return unordered_signatures([rows])[0]


def unordered_signatures(row_sets: Iterable[Any]) -> list[str]:
    return [profile.unordered_signature for profile in profile_row_sets(row_sets)]


def has_duplicate_canonical_rows(rows: Iterable[Any]) -> bool:
    return profile_rows(rows).has_duplicates


def dedupe_stable_canonical_rows(rows: Iterable[Any]) -> list[str]:
    return dedupe_canonical_rows(list(rows))


def multiset_canonical_row_diff(left_rows: Iterable[Any], right_rows: Iterable[Any]) -> tuple[list[str], list[str]]:
    return multiset_row_diff(list(left_rows), list(right_rows))


def compare_row_sets(
    left_rows: Iterable[Any],
    right_rows: Iterable[Any],
    *,
    left_columns: Sequence[str] | None = None,
    right_columns: Sequence[str] | None = None,
) -> RowComparison:
    left_list = list(left_rows)
    right_list = list(right_rows)
    same_schema = (
        True
        if left_columns is None or right_columns is None
        else tuple(left_columns) == tuple(right_columns)
    )
    if not same_schema:
        return RowComparison(
            mismatch_class="schema",
            same_schema=False,
            same_row_count=len(left_list) == len(right_list),
            same_ordered_rows=False,
            same_unordered_rows=False,
            left_only=[],
            right_only=[],
        )

    mismatch_class, same_row_count, same_ordered_rows, same_unordered_rows, left_only, right_only = kernel_compare_row_sets_summary(
        left_list,
        right_list,
    )
    return RowComparison(
        mismatch_class=mismatch_class,
        same_schema=True,
        same_row_count=same_row_count,
        same_ordered_rows=same_ordered_rows,
        same_unordered_rows=same_unordered_rows,
        left_only=left_only,
        right_only=right_only,
    )


def compare_row_set_batch(
    row_sets: Iterable[Iterable[Any]],
    *,
    column_sets: Sequence[Sequence[str]] | None = None,
) -> BatchRowComparison:
    values = [list(rows) for rows in row_sets]
    if not values:
        return BatchRowComparison(group_ids=[], mismatch_class="none")
    if column_sets is not None:
        if len(column_sets) != len(values):
            raise ValueError("column_sets must match row_sets length")
        normalized_columns = [tuple(columns) for columns in column_sets]
        if len(set(normalized_columns)) > 1:
            profiles = profile_row_sets(values)
            identity_keys = canonical_keys(
                {
                    "columns": list(columns),
                    "row_signature": profile.ordered_signature,
                }
                for columns, profile in zip(normalized_columns, profiles)
            )
            return BatchRowComparison(
                group_ids=_group_ids_from_keys(identity_keys),
                mismatch_class="schema",
            )
    group_ids, mismatch_class, suspicious_indices, has_clear_majority, majority_group = kernel_compare_row_set_batch_summary(values)
    return BatchRowComparison(
        group_ids=group_ids,
        mismatch_class=mismatch_class,
        suspicious_indices_value=suspicious_indices,
        has_clear_majority_value=has_clear_majority,
        majority_group_value=majority_group,
    )


def compare_result_batch(results: Iterable[Mapping[str, Any] | Any]) -> BatchResultComparison:
    items = list(results)
    if not items:
        return BatchResultComparison(group_ids=[], mismatch_class="none")
    fast = _compare_result_batch_fast(items)
    if fast is not None:
        return fast
    group_ids, mismatch_class, suspicious_indices, has_clear_majority, majority_group = kernel_compare_result_batch_summary(items)
    return BatchResultComparison(
        group_ids=group_ids,
        mismatch_class=mismatch_class,
        suspicious_indices_value=suspicious_indices,
        has_clear_majority_value=has_clear_majority,
        majority_group_value=majority_group,
    )


def compare_result_against_anchor(
    results: Iterable[Mapping[str, Any] | Any],
    *,
    anchor_index: int = 0,
) -> AnchorResultComparison:
    items = list(results)
    if not items:
        raise IndexError("anchor_index out of range")
    fast = _compare_result_batch_fast(items)
    if fast is not None:
        if anchor_index < 0 or anchor_index >= len(items):
            raise IndexError("anchor_index out of range")
        return AnchorResultComparison(
            anchor_index=anchor_index,
            matching_indices=fast.matching_indices(anchor_index),
            mismatching_indices=fast.mismatching_indices(anchor_index),
            confidence=fast.anchor_confidence(anchor_index),
        )
    matching_indices, mismatching_indices, confidence = kernel_compare_result_batch_anchor_summary(
        items,
        anchor_index,
    )
    if anchor_index < 0 or anchor_index >= len(items):
        raise IndexError("anchor_index out of range")
    return AnchorResultComparison(
        anchor_index=anchor_index,
        matching_indices=matching_indices,
        mismatching_indices=mismatching_indices,
        confidence=confidence,
    )


def _result_field(result: Mapping[str, Any] | Any, key: str, default: Any = None) -> Any:
    if isinstance(result, Mapping):
        return result.get(key, default)
    return getattr(result, key, default)


def _compare_result_batch_fast(
    results: Sequence[Mapping[str, Any] | Any],
) -> BatchResultComparison | None:
    profiles: list[ResultProfile] = []
    row_counts: list[int] = []
    for result in results:
        if isinstance(result, ResultProfile):
            profiles.append(result)
            row_counts.append(0)
            continue
        status = str(_result_field(result, "status", "") or "")
        columns = tuple(str(column) for column in list(_result_field(result, "columns", []) or []))
        error_type = str(_result_field(result, "error_type", "") or "")
        ordered_signature = str(_result_field(result, "ordered_row_signature", "") or "")
        unordered_signature = str(_result_field(result, "unordered_row_signature", "") or "")
        has_duplicate_rows = _result_field(result, "has_duplicate_rows", None)
        comparison_key = str(_result_field(result, "comparison_key", "") or "")
        if (
            not ordered_signature
            or not unordered_signature
            or not isinstance(has_duplicate_rows, bool)
            or not comparison_key
        ):
            return None
        profiles.append(
            ResultProfile(
                status=status,
                columns=columns,
                error_type=error_type,
                row_profile=RowSetProfile(
                    ordered_signature=ordered_signature,
                    unordered_signature=unordered_signature,
                    has_duplicates=has_duplicate_rows,
                ),
                comparison_key=comparison_key,
            )
        )
        row_count = _result_field(result, "row_count", None)
        if row_count is None:
            rows = _result_field(result, "rows", []) or []
            row_count = len(rows)
        row_counts.append(int(row_count))

    group_ids = _group_ids_from_keys([profile.comparison_key for profile in profiles])
    statuses = {profile.status for profile in profiles}
    if len(statuses) > 1:
        mismatch_class = "status"
    else:
        first_status = profiles[0].status
        if first_status != "ok":
            error_types = {profile.error_type for profile in profiles}
            if len(error_types) > 1:
                mismatch_class = "error_type"
            else:
                mismatch_class = "none" if len(set(group_ids)) == 1 else "status"
        else:
            if len({profile.columns for profile in profiles}) > 1:
                mismatch_class = "schema"
            elif len(set(row_counts)) > 1:
                mismatch_class = "row_count"
            elif len({profile.ordered_row_signature for profile in profiles}) == 1:
                mismatch_class = "none"
            elif len({profile.unordered_row_signature for profile in profiles}) == 1:
                mismatch_class = "row_order"
            else:
                mismatch_class = "value"

    suspicious_indices = _suspicious_indices_from_group_ids(group_ids)
    has_clear_majority = _has_clear_majority_group(group_ids)
    majority_group = _majority_group_from_group_ids(group_ids) or []
    return BatchResultComparison(
        group_ids=group_ids,
        mismatch_class=mismatch_class,
        suspicious_indices_value=suspicious_indices,
        has_clear_majority_value=has_clear_majority,
        majority_group_value=majority_group,
    )
