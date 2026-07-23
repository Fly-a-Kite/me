from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

from datadiff_osc._canonical import canonical_json, stable_digest, to_primitive
from datadiff_osc.contract_engine.model import Observation
from datadiff_osc.contract_engine.observers import LazyObservationView, ObserverSpec


FINGERPRINT_SCHEMA_VERSION = "osc-component-fingerprint-v1"
HASH_ALGORITHM_VERSION = "sha256-v1"


@dataclass(frozen=True, slots=True)
class ComponentFingerprint:
    endpoint_id: str
    observer_id: str
    observer_digest: str
    contract_digest: str
    row_count: int
    schema_digest: str
    payload_digest: str
    hash_algorithm: str = HASH_ALGORITHM_VERSION
    schema_version: str = FINGERPRINT_SCHEMA_VERSION

    @property
    def digest(self) -> str:
        return stable_digest("osc-component-fingerprint", self)

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(self)


def fingerprint_component(
    view: LazyObservationView,
    observer: ObserverSpec,
    contract_digest: str,
) -> ComponentFingerprint:
    payload = view.get(observer.observer_id)
    observation = view.observation
    schema_digest = hashlib.sha256(canonical_json(observation.schema).encode()).hexdigest()
    payload_digest = hashlib.sha256(
        canonical_json(
            {
                "observer": observer,
                "contract_digest": contract_digest,
                "schema": observation.schema,
                "row_count": observation.cardinality,
                "status": observation.status,
                "payload": payload,
            }
        ).encode()
    ).hexdigest()
    return ComponentFingerprint(
        endpoint_id=observation.endpoint_id,
        observer_id=observer.observer_id,
        observer_digest=observer.digest,
        contract_digest=contract_digest,
        row_count=observation.cardinality,
        schema_digest=schema_digest,
        payload_digest=payload_digest,
    )
