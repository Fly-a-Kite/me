"""Private re-readable provenance for one Phase-6 target-version receipt.

This module binds one complete target-version receipt to the exact raw
version-audit artifacts it deterministically derives from.  It is deliberately
not a public API, raw gate artifact, producer admission, metric, or gate
decision.  Root must still verify a separately hashed canonical envelope
before retaining any target-version admission.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

from datadiff_osc._canonical import (
    assert_deeply_immutable,
    canonical_envelope,
    stable_digest,
    to_primitive,
)
from datadiff_osc._replay_support import (
    ReplayValidationError,
    assert_payload_roundtrip,
    require_exact_mapping,
    require_text,
)
from datadiff_osc.runtime._private_receipts import TargetVersionReceipt
from datadiff_osc.runtime._semantic_replay import _replay_target_receipt


TARGET_VERSION_PRODUCTION_PROVENANCE_SCHEMA_VERSION = (
    "osc-phase6-target-version-production-provenance-v1"
)
TARGET_VERSION_PRODUCTION_PROVENANCE_ENVELOPE_TYPE = (
    "TargetVersionProductionProvenance"
)
_PRODUCER_FILENAME = "_phase6_formal_target_version_producer.py"
_BRIDGE_FILENAME = "_phase6_target_version_provenance.py"


def _module_sha256(filename: str) -> str:
    try:
        raw = Path(__file__).with_name(filename).read_bytes()
    except OSError as exc:
        raise ValueError(
            f"target version provenance implementation bytes unavailable:{filename}"
        ) from exc
    return hashlib.sha256(raw).hexdigest()


def _require_sha256(value: object, *, path: str) -> str:
    text = require_text(value, path=path)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ReplayValidationError(f"{path}: expected a lowercase SHA-256 digest")
    return text


def target_version_execution_transcript_digest(
    receipt: TargetVersionReceipt,
) -> str:
    """Bind the exact audit/environment metadata of one target-version replay."""

    if not isinstance(receipt, TargetVersionReceipt):
        raise TypeError("receipt must be a TargetVersionReceipt")
    return stable_digest(
        "osc-phase6-target-version-execution-transcript-v1",
        {
            "source_digest": receipt.source_digest,
            "audit_plan_digest": receipt.audit_plan_digest,
            "environment_digest": receipt.environment_digest,
            "python_runtime_digest": receipt.python_runtime_digest,
            "package_ids": receipt.package_ids,
            "installed_versions": tuple(
                item.installed_version for item in receipt.packages
            ),
            "latest_versions": tuple(
                item.latest_version for item in receipt.packages
            ),
            "version_source_sha256s": tuple(
                item.version_source_sha256 for item in receipt.packages
            ),
        },
    )


def _replay_target_version_receipt(value: object, path: str) -> TargetVersionReceipt:
    """Use Runtime's owning receipt replayer without accepting a caller receipt."""

    try:
        return _replay_target_receipt(value, path)
    except (TypeError, ValueError) as exc:
        raise ReplayValidationError(f"{path}: receipt reconstruction failed:{exc}") from exc


@dataclass(frozen=True, slots=True)
class TargetVersionProductionProvenance:
    """Canonical raw version-audit bindings and the exact receipt they derive.

    The implementation hashes are derived from local bytes by the builder and
    re-checked by reconstruction.  They are not caller-selected declarations.
    """

    receipt: TargetVersionReceipt
    source_snapshot_digest: str
    manifest_sha256: str
    execution_transcript_digest: str
    producer_module_sha256: str
    bridge_module_sha256: str
    schema_version: str = TARGET_VERSION_PRODUCTION_PROVENANCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.receipt, TargetVersionReceipt):
            raise ValueError("target version provenance requires a TargetVersionReceipt")
        if self.schema_version != TARGET_VERSION_PRODUCTION_PROVENANCE_SCHEMA_VERSION:
            raise ValueError("target version provenance schema version mismatch")
        if (
            not isinstance(self.source_snapshot_digest, str)
            or not self.source_snapshot_digest
        ):
            raise ValueError("target version provenance source snapshot digest is invalid")
        if self.source_snapshot_digest != self.receipt.source_digest:
            raise ValueError("target version provenance source binding mismatch")
        if (
            not isinstance(self.manifest_sha256, str)
            or len(self.manifest_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.manifest_sha256
            )
        ):
            raise ValueError(
                "target version provenance raw binding is invalid:manifest_sha256"
            )
        if self.execution_transcript_digest != target_version_execution_transcript_digest(
            self.receipt
        ):
            raise ValueError("target version provenance transcript binding mismatch")
        for field_name, expected in (
            ("producer_module_sha256", _module_sha256(_PRODUCER_FILENAME)),
            ("bridge_module_sha256", _module_sha256(_BRIDGE_FILENAME)),
        ):
            actual = getattr(self, field_name)
            if not isinstance(actual, str) or actual != expected:
                raise ValueError(
                    f"target version provenance implementation binding mismatch:{field_name}"
                )
        assert_deeply_immutable(self)

    @property
    def provenance_id(self) -> str:
        return stable_digest(
            "osc-phase6-target-version-production-provenance-id-v1",
            {
                "source_snapshot_digest": self.receipt.source_digest,
                "audit_plan_digest": self.receipt.audit_plan_digest,
                "receipt_digest": self.receipt.digest,
                "producer_module_sha256": self.producer_module_sha256,
                "bridge_module_sha256": self.bridge_module_sha256,
            },
        )

    @property
    def digest(self) -> str:
        return stable_digest("osc-phase6-target-version-production-provenance", self)

    def to_dict(self) -> dict[str, object]:
        value = to_primitive(self)
        assert isinstance(value, dict)
        return value


def build_target_version_production_provenance(
    *,
    receipt: TargetVersionReceipt,
    manifest_sha256: str,
) -> TargetVersionProductionProvenance:
    """Build the only private provenance object from complete typed input."""

    if not isinstance(receipt, TargetVersionReceipt):
        raise TypeError("receipt must be a TargetVersionReceipt")
    return TargetVersionProductionProvenance(
        receipt=receipt,
        source_snapshot_digest=receipt.source_digest,
        manifest_sha256=manifest_sha256,
        execution_transcript_digest=target_version_execution_transcript_digest(receipt),
        producer_module_sha256=_module_sha256(_PRODUCER_FILENAME),
        bridge_module_sha256=_module_sha256(_BRIDGE_FILENAME),
    )


def reconstruct_target_version_receipt_payload(
    payload: object,
) -> TargetVersionReceipt:
    """Reconstruct the private target-version receipt for Root cross-object binding."""

    receipt = _replay_target_version_receipt(payload, "payload.TargetVersionReceipt")
    assert_payload_roundtrip(payload, receipt)
    return receipt


def reconstruct_target_version_production_provenance_payload(
    payload: object,
) -> TargetVersionProductionProvenance:
    """Fail-closed canonical reconstruction used only by the Root bridge."""

    path = "payload.TargetVersionProductionProvenance"
    data = require_exact_mapping(
        payload,
        fields=(
            "receipt",
            "source_snapshot_digest",
            "manifest_sha256",
            "execution_transcript_digest",
            "producer_module_sha256",
            "bridge_module_sha256",
            "schema_version",
        ),
        path=path,
    )
    provenance = TargetVersionProductionProvenance(
        receipt=_replay_target_version_receipt(data["receipt"], f"{path}.receipt"),
        source_snapshot_digest=require_text(
            data["source_snapshot_digest"],
            path=f"{path}.source_snapshot_digest",
        ),
        manifest_sha256=_require_sha256(
            data["manifest_sha256"], path=f"{path}.manifest_sha256"
        ),
        execution_transcript_digest=require_text(
            data["execution_transcript_digest"],
            path=f"{path}.execution_transcript_digest",
        ),
        producer_module_sha256=_require_sha256(
            data["producer_module_sha256"],
            path=f"{path}.producer_module_sha256",
        ),
        bridge_module_sha256=_require_sha256(
            data["bridge_module_sha256"],
            path=f"{path}.bridge_module_sha256",
        ),
        schema_version=require_text(data["schema_version"], path=f"{path}.schema_version"),
    )
    assert_payload_roundtrip(payload, provenance, path=path)
    return provenance


def canonical_target_version_production_provenance_envelope(
    provenance: TargetVersionProductionProvenance,
) -> str:
    if not isinstance(provenance, TargetVersionProductionProvenance):
        raise TypeError("provenance must be a TargetVersionProductionProvenance")
    return canonical_envelope(
        TARGET_VERSION_PRODUCTION_PROVENANCE_ENVELOPE_TYPE,
        TARGET_VERSION_PRODUCTION_PROVENANCE_SCHEMA_VERSION,
        provenance,
    )


__all__ = [
    "TARGET_VERSION_PRODUCTION_PROVENANCE_ENVELOPE_TYPE",
    "TARGET_VERSION_PRODUCTION_PROVENANCE_SCHEMA_VERSION",
    "TargetVersionProductionProvenance",
    "build_target_version_production_provenance",
    "canonical_target_version_production_provenance_envelope",
    "reconstruct_target_version_production_provenance_payload",
    "reconstruct_target_version_receipt_payload",
    "target_version_execution_transcript_digest",
]
