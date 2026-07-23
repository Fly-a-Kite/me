from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from typing import Any

from datadiff.backends.base import BackendResult, PreparedTable
from datadiff.dsl import Program


ADAPTER_SPI_VERSION = "backend-adapter-spi-v1"


@dataclass(frozen=True, slots=True)
class LoweredCase:
    tables: tuple[PreparedTable, ...]
    program: Program
    lowering_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class NativeExportResult:
    status: str
    format: str
    payload: str = ""
    skip_reason: str = ""

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AdapterSPIManifest:
    backend: str
    session_reuse_policy: str
    lowering: str
    execution: str
    normalization: str
    plan_collection: str
    native_export: str
    lifecycle: str
    result_contract: str = "backend-result-v1"
    schema_version: str = ADAPTER_SPI_VERSION

    @property
    def digest(self) -> str:
        payload = json.dumps(
            asdict(self),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "digest": self.digest}


def validate_backend_result(result: BackendResult, *, backend_name: str) -> None:
    if not isinstance(result, BackendResult):
        raise TypeError(
            f"adapter {backend_name} returned {type(result).__name__}, expected BackendResult"
        )
    if result.backend != backend_name:
        raise ValueError(
            f"adapter result backend mismatch: expected {backend_name}, got {result.backend}"
        )
    if result.status not in {"ok", "error", "timeout", "missing"}:
        raise ValueError(f"adapter {backend_name} returned invalid status: {result.status}")
