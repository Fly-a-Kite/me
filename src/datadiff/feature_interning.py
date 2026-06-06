from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from threading import Lock
from typing import Any


@dataclass(slots=True)
class FeatureInterner:
    _text_to_id: dict[str, int] = field(default_factory=dict)
    _id_to_text: list[str] = field(default_factory=list)
    _lock: Lock = field(default_factory=Lock, repr=False)

    def intern(self, value: str) -> int:
        text = str(value)
        feature_id = self._text_to_id.get(text)
        if feature_id is not None:
            return feature_id
        with self._lock:
            feature_id = self._text_to_id.get(text)
            if feature_id is not None:
                return feature_id
            feature_id = len(self._id_to_text)
            self._id_to_text.append(text)
            self._text_to_id[text] = feature_id
            return feature_id

    def resolve(self, feature_id: int) -> str:
        if 0 <= int(feature_id) < len(self._id_to_text):
            return self._id_to_text[int(feature_id)]
        return ""

    def intern_many(self, values: Iterable[str]) -> tuple[int, ...]:
        return tuple(self.intern(value) for value in values)

    def intern_set(self, values: Iterable[str]) -> frozenset[int]:
        return frozenset(self.intern_many(values))

    def export_mapping(self, values: Mapping[int, Any]) -> dict[str, Any]:
        return {
            self.resolve(feature_id): value
            for feature_id, value in sorted(values.items(), key=lambda item: self.resolve(int(item[0])))
            if self.resolve(feature_id)
        }

    def export_counter(self, values: Mapping[int, Any]) -> dict[str, int]:
        return {
            self.resolve(feature_id): int(value or 0)
            for feature_id, value in sorted(values.items(), key=lambda item: self.resolve(int(item[0])))
            if self.resolve(feature_id)
        }

    def import_float_mapping(self, values: Mapping[str, Any] | None) -> dict[int, float]:
        out: dict[int, float] = {}
        for feature, value in (values or {}).items():
            text = str(feature)
            if not text:
                continue
            out[self.intern(text)] = float(value or 0.0)
        return out

    def import_counter(self, values: Mapping[str, Any] | None) -> Counter[int]:
        out: Counter[int] = Counter()
        for feature, value in (values or {}).items():
            text = str(feature)
            if not text:
                continue
            out[self.intern(text)] = int(value or 0)
        return out


GLOBAL_FEATURE_INTERNER = FeatureInterner()


def intern_feature(value: str) -> int:
    return GLOBAL_FEATURE_INTERNER.intern(value)


def resolve_feature_id(feature_id: int) -> str:
    return GLOBAL_FEATURE_INTERNER.resolve(feature_id)


def intern_feature_ids(values: Iterable[str]) -> tuple[int, ...]:
    return GLOBAL_FEATURE_INTERNER.intern_many(values)


def intern_feature_id_set(values: Iterable[str]) -> frozenset[int]:
    return GLOBAL_FEATURE_INTERNER.intern_set(values)


def export_feature_mapping(values: Mapping[int, Any]) -> dict[str, Any]:
    return GLOBAL_FEATURE_INTERNER.export_mapping(values)


def export_feature_counter(values: Mapping[int, Any]) -> dict[str, int]:
    return GLOBAL_FEATURE_INTERNER.export_counter(values)


def import_feature_float_mapping(values: Mapping[str, Any] | None) -> dict[int, float]:
    return GLOBAL_FEATURE_INTERNER.import_float_mapping(values)


def import_feature_counter(values: Mapping[str, Any] | None) -> Counter[int]:
    return GLOBAL_FEATURE_INTERNER.import_counter(values)
