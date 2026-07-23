from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from datadiff_osc._canonical import canonical_json, stable_digest, to_primitive
from datadiff_osc.contract_engine.model import Observation


OBSERVER_SCHEMA_VERSION = "osc-component-observer-v1"


@dataclass(frozen=True, slots=True)
class ObserverSpec:
    observer_id: str
    component: str
    version: str = OBSERVER_SCHEMA_VERSION

    @property
    def digest(self) -> str:
        return stable_digest("osc-observer-spec", self)


@dataclass(slots=True)
class LazyObservationView:
    observation: Observation
    _cache: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    requested: list[str] = field(default_factory=list, init=False)

    def get(self, observer_id: str) -> Any:
        if observer_id not in self._cache:
            self._cache[observer_id] = observe_component(self.observation, observer_id)
            self.requested.append(observer_id)
        return self._cache[observer_id]


def observe_component(observation: Observation, observer_id: str) -> Any:
    if observer_id == "status":
        return observation.status
    if observer_id == "schema":
        return observation.schema
    if observer_id == "schema_names":
        return tuple(field.name for field in observation.schema)
    if observer_id == "schema_types":
        return tuple((field.logical_type, field.nullable) for field in observation.schema)
    if observer_id == "cardinality":
        return observation.cardinality
    if observer_id in {"sequence", "numeric_exact"}:
        return observation.rows
    if observer_id == "bag":
        return tuple(sorted(Counter(canonical_json(row) for row in observation.rows).items()))
    if observer_id == "set_unique":
        rows = tuple(canonical_json(row) for row in observation.rows)
        return {"rows": tuple(sorted(set(rows))), "unique": len(rows) == len(set(rows))}
    if observer_id == "partial_order":
        return observation.rows
    if observer_id == "error":
        return (observation.status, observation.error_category, observation.error_type)
    if observer_id == "layout":
        return dict(observation.execution_metadata).get("physical_layout", "unknown")
    raise KeyError(f"unknown observer: {observer_id}")

