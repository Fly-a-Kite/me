from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

from datadiff.canonicalization import compare_result_batch
from datadiff.normalizer import NormalizedResult

_MAX_COLUMN_CLASS_VALUES = 512


@dataclass(frozen=True, slots=True)
class DisagreementDescriptor:
    backend_groups: tuple[tuple[str, ...], ...] = ()
    pair_disagrees: dict[tuple[str, str], bool] = field(default_factory=dict)
    column_classes: dict[str, str] = field(default_factory=dict)
    primary_root_cause: str = ""
    mismatch_class: str = "none"
    backend_statuses: dict[str, str] = field(default_factory=dict)

    @property
    def pair_count(self) -> int:
        return sum(1 for disagrees in self.pair_disagrees.values() if disagrees)

    def feature_tokens(self) -> tuple[str, ...]:
        tokens: list[str] = []
        if self.mismatch_class:
            tokens.append(f"mismatch:{self.mismatch_class}")
        tokens.append(f"group_count:{len(self.backend_groups)}")
        for group in self.backend_groups:
            if group:
                tokens.append(f"group:{'+'.join(group)}")
        for backend, status in sorted(self.backend_statuses.items()):
            if status:
                tokens.append(f"status:{backend}:{status}")
        for (left, right), disagrees in sorted(self.pair_disagrees.items()):
            if disagrees:
                tokens.append(f"disagree_pair:{left}|{right}")
        for column, klass in sorted(self.column_classes.items()):
            if klass:
                tokens.append(f"column:{column}:{klass}")
                tokens.append(f"disagree_class:{klass}")
        if self.primary_root_cause:
            tokens.append(f"root:{self.primary_root_cause}")
        return tuple(sorted(dict.fromkeys(tokens)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend_groups": [list(group) for group in self.backend_groups],
            "pair_disagrees": [
                {"left": left, "right": right, "disagrees": bool(disagrees)}
                for (left, right), disagrees in sorted(self.pair_disagrees.items())
            ],
            "pair_count": self.pair_count,
            "column_classes": {
                column: klass for column, klass in sorted(self.column_classes.items())
            },
            "primary_root_cause": self.primary_root_cause,
            "mismatch_class": self.mismatch_class,
            "backend_statuses": {
                backend: status for backend, status in sorted(self.backend_statuses.items())
            },
            "feature_tokens": list(self.feature_tokens()),
        }

    @classmethod
    def from_dict(cls, data: Any) -> "DisagreementDescriptor":
        if not isinstance(data, Mapping):
            return cls()
        return cls(
            backend_groups=_coerce_backend_groups(data.get("backend_groups")),
            pair_disagrees=_coerce_pair_disagrees(data.get("pair_disagrees")),
            column_classes=_coerce_string_map(data.get("column_classes")),
            primary_root_cause=str(data.get("primary_root_cause", "") or ""),
            mismatch_class=str(data.get("mismatch_class", "none") or "none"),
            backend_statuses=_coerce_string_map(data.get("backend_statuses")),
        )


def compute_descriptor(
    normalized: Mapping[str, NormalizedResult] | None,
    findings: Sequence[Any] | None,
) -> DisagreementDescriptor:
    if not normalized:
        return DisagreementDescriptor()

    items = sorted(
        ((str(backend), result) for backend, result in normalized.items()),
        key=lambda item: item[0],
    )
    labels = [backend for backend, _ in items]
    results = [_coerce_result(result, backend=backend) for backend, result in items]
    comparison = compare_result_batch(results)
    group_ids = list(comparison.group_ids)
    if len(group_ids) != len(labels):
        group_ids = list(range(len(labels)))
    backend_groups = _backend_groups(labels, group_ids)
    pair_disagrees = {
        (left, right): group_ids[left_index] != group_ids[right_index]
        for (left_index, left), (right_index, right) in combinations(enumerate(labels), 2)
    }
    return DisagreementDescriptor(
        backend_groups=backend_groups,
        pair_disagrees=pair_disagrees,
        column_classes=_column_classes(results),
        primary_root_cause=_primary_root_cause(findings or ()),
        mismatch_class=str(comparison.mismatch_class or "none"),
        backend_statuses=_backend_statuses(results),
    )


def _coerce_result(result: Any, *, backend: str) -> NormalizedResult:
    if isinstance(result, NormalizedResult):
        return result
    if isinstance(result, Mapping):
        return NormalizedResult.from_dict(result, backend=backend)
    return NormalizedResult(
        backend=backend,
        status=str(getattr(result, "status", "") or ""),
        columns=[str(column) for column in list(getattr(result, "columns", []) or [])],
        rows=[list(row) for row in list(getattr(result, "rows", []) or [])],
        error_type=str(getattr(result, "error_type", "") or ""),
        error=str(getattr(result, "error", "") or ""),
    )


def _backend_groups(labels: Sequence[str], group_ids: Sequence[int]) -> tuple[tuple[str, ...], ...]:
    groups: dict[int, list[str]] = {}
    for label, group_id in zip(labels, group_ids):
        groups.setdefault(int(group_id), []).append(str(label))
    return tuple(tuple(groups[group_id]) for group_id in sorted(groups, key=lambda key: groups[key][0]))


def _backend_statuses(results: Sequence[NormalizedResult]) -> dict[str, str]:
    out: dict[str, str] = {}
    for result in results:
        status = str(result.status or "unknown")
        if status == "error" and result.error_type:
            status = f"error:{result.error_type}"
        out[str(result.backend)] = status
    return out


def _column_classes(results: Sequence[NormalizedResult]) -> dict[str, str]:
    values_by_column: dict[str, list[Any]] = {}
    saw_column: set[str] = set()
    for result in results:
        if result.status != "ok":
            continue
        columns = [str(column) for column in result.columns]
        for column in columns:
            saw_column.add(column)
            values_by_column.setdefault(column, [])
        for row in result.rows:
            if not isinstance(row, Sequence) or isinstance(row, (str, bytes, bytearray)):
                continue
            for index, value in enumerate(row[: len(columns)]):
                column = columns[index]
                values = values_by_column.setdefault(column, [])
                if len(values) < _MAX_COLUMN_CLASS_VALUES:
                    values.append(value)
    return {
        column: _classify_values(values_by_column.get(column, []))
        for column in sorted(saw_column)
    }


def _classify_values(values: Sequence[Any]) -> str:
    if not values:
        return "unknown"
    kinds: set[str] = set()
    saw_null = False
    for value in values:
        if value is None:
            saw_null = True
            continue
        if isinstance(value, bool):
            kinds.add("bool")
        elif isinstance(value, (int, float)):
            kinds.add("numeric")
        elif isinstance(value, str):
            kinds.add("string")
        elif isinstance(value, (Mapping, list, tuple)):
            kinds.add("complex")
        else:
            kinds.add("object")
    if not kinds:
        return "null" if saw_null else "unknown"
    if len(kinds) == 1:
        return next(iter(kinds))
    return "mixed"


def _primary_root_cause(findings: Sequence[Any]) -> str:
    fallback = ""
    for finding in findings:
        value = _finding_value(finding, "root_cause")
        if not value:
            continue
        text = str(value)
        if not fallback:
            fallback = text
        if text != "unknown":
            return text
    return "" if fallback == "unknown" else fallback


def _finding_value(finding: Any, key: str) -> Any:
    if isinstance(finding, Mapping):
        return finding.get(key)
    return getattr(finding, key, None)


def _coerce_backend_groups(value: Any) -> tuple[tuple[str, ...], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return ()
    groups: list[tuple[str, ...]] = []
    for group in value:
        if isinstance(group, Sequence) and not isinstance(group, (str, bytes, bytearray)):
            labels = tuple(str(item) for item in group if str(item))
            if labels:
                groups.append(labels)
    return tuple(groups)


def _coerce_pair_disagrees(value: Any) -> dict[tuple[str, str], bool]:
    if isinstance(value, Mapping):
        out: dict[tuple[str, str], bool] = {}
        for key, disagrees in value.items():
            pair = _parse_pair_key(key)
            if pair:
                out[pair] = bool(disagrees)
        return out
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return {}
    out = {}
    for item in value:
        if isinstance(item, Mapping):
            left = str(item.get("left", "") or "")
            right = str(item.get("right", "") or "")
            if left and right:
                out[tuple(sorted((left, right)))] = bool(item.get("disagrees"))
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)) and len(item) >= 3:
            left, right, disagrees = item[:3]
            if str(left) and str(right):
                out[tuple(sorted((str(left), str(right))))] = bool(disagrees)
    return out


def _parse_pair_key(key: Any) -> tuple[str, str] | None:
    if isinstance(key, tuple) and len(key) == 2:
        left, right = str(key[0]), str(key[1])
    else:
        text = str(key)
        separator = "|" if "|" in text else ","
        parts = [part.strip() for part in text.split(separator, 1)]
        if len(parts) != 2:
            return None
        left, right = parts
    if not left or not right:
        return None
    return tuple(sorted((left, right)))


def _coerce_string_map(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    return {
        str(key): str(item)
        for key, item in value.items()
        if str(key) and str(item)
    }
