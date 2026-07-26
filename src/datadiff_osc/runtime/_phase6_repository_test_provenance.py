"""Private re-readable provenance for one Phase-6 repository-test receipt.

This module binds one complete R2 repository-test receipt to the exact raw
execution artifacts it deterministically derives from.  It is deliberately not
a public API, raw gate artifact, producer admission, metric, or gate decision.
Root must still verify a separately hashed canonical envelope before retaining
any repository-test admission.
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
from datadiff_osc.runtime._private_receipts import RepositoryTestReceipt
from datadiff_osc.runtime._semantic_replay import _replay_repository_receipt


REPOSITORY_TEST_PRODUCTION_PROVENANCE_SCHEMA_VERSION = (
    "osc-phase6-repository-test-production-provenance-v1"
)
REPOSITORY_TEST_PRODUCTION_PROVENANCE_ENVELOPE_TYPE = (
    "RepositoryTestProductionProvenance"
)
_R2_PRODUCER_FILENAME = "_phase6_formal_repository_test_producer.py"
_BRIDGE_FILENAME = "_phase6_repository_test_provenance.py"


def _module_sha256(filename: str) -> str:
    try:
        raw = Path(__file__).with_name(filename).read_bytes()
    except OSError as exc:
        raise ValueError(
            f"repository test provenance implementation bytes unavailable:{filename}"
        ) from exc
    return hashlib.sha256(raw).hexdigest()


def _require_sha256(value: object, *, path: str) -> str:
    text = require_text(value, path=path)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ReplayValidationError(f"{path}: expected a lowercase SHA-256 digest")
    return text


def repository_test_execution_transcript_digest(
    receipt: RepositoryTestReceipt,
) -> str:
    """Bind the exact command/exit metadata of one repository-test execution."""

    if not isinstance(receipt, RepositoryTestReceipt):
        raise TypeError("receipt must be a RepositoryTestReceipt")
    return stable_digest(
        "osc-phase6-repository-test-execution-transcript-v1",
        {
            "source_digest": receipt.source_digest,
            "test_plan_digest": receipt.test_plan_digest,
            "collection_digest": receipt.collection_digest,
            "config_digest": receipt.config_digest,
            "environment_digest": receipt.environment_digest,
            "command_digest": receipt.command_digest,
            "junit_sha256": receipt.junit_sha256,
            "log_sha256": receipt.log_sha256,
            "exit_code": receipt.exit_code,
        },
    )


def _replay_repository_test_receipt(value: object, path: str) -> RepositoryTestReceipt:
    """Use Runtime's owning receipt replayer without accepting a caller receipt."""

    try:
        return _replay_repository_receipt(value, path)
    except (TypeError, ValueError) as exc:
        raise ReplayValidationError(f"{path}: receipt reconstruction failed:{exc}") from exc


@dataclass(frozen=True, slots=True)
class RepositoryTestProductionProvenance:
    """Canonical raw execution bindings and the exact R2 receipt they derive.

    The implementation hashes are derived from local bytes by the builder and
    re-checked by reconstruction.  They are not caller-selected declarations.
    """

    receipt: RepositoryTestReceipt
    source_snapshot_digest: str
    collection_sha256: str
    junit_sha256: str
    log_sha256: str
    execution_transcript_digest: str
    producer_module_sha256: str
    bridge_module_sha256: str
    schema_version: str = REPOSITORY_TEST_PRODUCTION_PROVENANCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.receipt, RepositoryTestReceipt):
            raise ValueError("repository test provenance requires a RepositoryTestReceipt")
        if self.schema_version != REPOSITORY_TEST_PRODUCTION_PROVENANCE_SCHEMA_VERSION:
            raise ValueError("repository test provenance schema version mismatch")
        if (
            not isinstance(self.source_snapshot_digest, str)
            or not self.source_snapshot_digest
        ):
            raise ValueError("repository test provenance source snapshot digest is invalid")
        if self.source_snapshot_digest != self.receipt.source_digest:
            raise ValueError("repository test provenance source binding mismatch")
        for field_name in ("collection_sha256", "junit_sha256", "log_sha256"):
            value = getattr(self, field_name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(
                    f"repository test provenance raw binding is invalid:{field_name}"
                )
        if self.junit_sha256 != self.receipt.junit_sha256:
            raise ValueError("repository test provenance junit binding mismatch")
        if self.log_sha256 != self.receipt.log_sha256:
            raise ValueError("repository test provenance log binding mismatch")
        if self.execution_transcript_digest != repository_test_execution_transcript_digest(
            self.receipt
        ):
            raise ValueError("repository test provenance transcript binding mismatch")
        for field_name, expected in (
            ("producer_module_sha256", _module_sha256(_R2_PRODUCER_FILENAME)),
            ("bridge_module_sha256", _module_sha256(_BRIDGE_FILENAME)),
        ):
            actual = getattr(self, field_name)
            if not isinstance(actual, str) or actual != expected:
                raise ValueError(
                    f"repository test provenance implementation binding mismatch:{field_name}"
                )
        assert_deeply_immutable(self)

    @property
    def provenance_id(self) -> str:
        return stable_digest(
            "osc-phase6-repository-test-production-provenance-id-v1",
            {
                "source_snapshot_digest": self.receipt.source_digest,
                "collection_digest": self.receipt.collection_digest,
                "receipt_digest": self.receipt.digest,
                "producer_module_sha256": self.producer_module_sha256,
                "bridge_module_sha256": self.bridge_module_sha256,
            },
        )

    @property
    def digest(self) -> str:
        return stable_digest("osc-phase6-repository-test-production-provenance", self)

    def to_dict(self) -> dict[str, object]:
        value = to_primitive(self)
        assert isinstance(value, dict)
        return value


def build_repository_test_production_provenance(
    *,
    receipt: RepositoryTestReceipt,
    collection_sha256: str,
) -> RepositoryTestProductionProvenance:
    """Build the only private provenance object from complete typed input."""

    if not isinstance(receipt, RepositoryTestReceipt):
        raise TypeError("receipt must be a RepositoryTestReceipt")
    return RepositoryTestProductionProvenance(
        receipt=receipt,
        source_snapshot_digest=receipt.source_digest,
        collection_sha256=collection_sha256,
        junit_sha256=receipt.junit_sha256,
        log_sha256=receipt.log_sha256,
        execution_transcript_digest=repository_test_execution_transcript_digest(receipt),
        producer_module_sha256=_module_sha256(_R2_PRODUCER_FILENAME),
        bridge_module_sha256=_module_sha256(_BRIDGE_FILENAME),
    )


def reconstruct_repository_test_receipt_payload(
    payload: object,
) -> RepositoryTestReceipt:
    """Reconstruct the private repository receipt for Root cross-object binding."""

    receipt = _replay_repository_test_receipt(payload, "payload.RepositoryTestReceipt")
    assert_payload_roundtrip(payload, receipt)
    return receipt


def reconstruct_repository_test_production_provenance_payload(
    payload: object,
) -> RepositoryTestProductionProvenance:
    """Fail-closed canonical reconstruction used only by the Root bridge."""

    path = "payload.RepositoryTestProductionProvenance"
    data = require_exact_mapping(
        payload,
        fields=(
            "receipt",
            "source_snapshot_digest",
            "collection_sha256",
            "junit_sha256",
            "log_sha256",
            "execution_transcript_digest",
            "producer_module_sha256",
            "bridge_module_sha256",
            "schema_version",
        ),
        path=path,
    )
    provenance = RepositoryTestProductionProvenance(
        receipt=_replay_repository_test_receipt(data["receipt"], f"{path}.receipt"),
        source_snapshot_digest=require_text(
            data["source_snapshot_digest"],
            path=f"{path}.source_snapshot_digest",
        ),
        collection_sha256=_require_sha256(
            data["collection_sha256"], path=f"{path}.collection_sha256"
        ),
        junit_sha256=_require_sha256(data["junit_sha256"], path=f"{path}.junit_sha256"),
        log_sha256=_require_sha256(data["log_sha256"], path=f"{path}.log_sha256"),
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


def canonical_repository_test_production_provenance_envelope(
    provenance: RepositoryTestProductionProvenance,
) -> str:
    if not isinstance(provenance, RepositoryTestProductionProvenance):
        raise TypeError("provenance must be a RepositoryTestProductionProvenance")
    return canonical_envelope(
        REPOSITORY_TEST_PRODUCTION_PROVENANCE_ENVELOPE_TYPE,
        REPOSITORY_TEST_PRODUCTION_PROVENANCE_SCHEMA_VERSION,
        provenance,
    )


__all__ = [
    "REPOSITORY_TEST_PRODUCTION_PROVENANCE_ENVELOPE_TYPE",
    "REPOSITORY_TEST_PRODUCTION_PROVENANCE_SCHEMA_VERSION",
    "RepositoryTestProductionProvenance",
    "build_repository_test_production_provenance",
    "canonical_repository_test_production_provenance_envelope",
    "reconstruct_repository_test_production_provenance_payload",
    "reconstruct_repository_test_receipt_payload",
    "repository_test_execution_transcript_digest",
]
