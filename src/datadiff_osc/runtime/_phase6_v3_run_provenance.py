"""Private re-readable provenance for one Phase-6 v3 run receipt.

This module binds one complete ``V3RunReceipt`` to the exact frozen builder
semantics it deterministically derives from.  It is deliberately not a public
API, raw gate artifact, producer admission, metric, or gate decision.  Root
must still verify a separately hashed canonical envelope before retaining any
v3 run admission.
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
from datadiff_osc.runtime._phase6_v3_run_receipt_builder import (
    _reject_synthetic_markers,
    v3_case_pipeline_digest,
    v3_execution_run_digest,
)
from datadiff_osc.runtime._private_receipts import V3RunReceipt
from datadiff_osc.runtime._semantic_replay import _replay_v3_receipt


V3_RUN_PRODUCTION_PROVENANCE_SCHEMA_VERSION = (
    "osc-phase6-v3-run-production-provenance-v1"
)
V3_RUN_PRODUCTION_PROVENANCE_ENVELOPE_TYPE = "V3RunProductionProvenance"
_PRODUCER_FILENAME = "_phase6_v3_run_receipt_builder.py"
_BRIDGE_FILENAME = "_phase6_v3_run_provenance.py"


def _module_sha256(filename: str) -> str:
    try:
        raw = Path(__file__).with_name(filename).read_bytes()
    except OSError as exc:
        raise ValueError(
            f"v3 run provenance implementation bytes unavailable:{filename}"
        ) from exc
    return hashlib.sha256(raw).hexdigest()


def _require_sha256(value: object, *, path: str) -> str:
    text = require_text(value, path=path)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ReplayValidationError(f"{path}: expected a lowercase SHA-256 digest")
    return text


def v3_run_execution_transcript_digest(receipt: V3RunReceipt) -> str:
    """Bind the exact run-identity fields of one v3 run replay."""

    if not isinstance(receipt, V3RunReceipt):
        raise TypeError("receipt must be a V3RunReceipt")
    return stable_digest(
        "osc-phase6-v3-run-execution-transcript-v1",
        {
            "source_digest": receipt.source_digest,
            "protocol_digest": receipt.protocol_digest,
            "v3_gate_plan_digest": receipt.v3_gate_plan_digest,
            "run_id": receipt.run_id,
            "lane_id": receipt.lane_id,
            "seed": receipt.seed,
            "execution_run_digest": receipt.execution_run_digest,
            "run_completed": receipt.run_completed,
            "case_ids": receipt.case_ids,
            "executed_case_ids": receipt.executed_case_ids,
            "backend_task_ids": receipt.backend_task_ids,
        },
    )


def _require_frozen_digest_semantics(receipt: V3RunReceipt) -> None:
    """Reject any receipt whose digests do not recompute under the freeze."""

    for case in receipt.cases:
        expected_pipeline = v3_case_pipeline_digest(
            case_id=case.case_id,
            case_task_id=case.case_task_id,
            result_digest=case.result_digest,
            execution_outcome_digest=case.execution_outcome_digest,
            evidence_envelope_digest=case.evidence_envelope_digest,
            backend_task_ids=tuple(item.task_id for item in case.backend_tasks),
        )
        if case.pipeline_digest != expected_pipeline:
            raise ValueError(
                "v3 run provenance pipeline digest binding mismatch:"
                f"{case.case_id}"
            )
    expected_run_digest = v3_execution_run_digest(
        v3_gate_plan_digest=receipt.v3_gate_plan_digest,
        lane_id=receipt.lane_id,
        seed=receipt.seed,
        run_id=receipt.run_id,
        case_ids=tuple(item.case_id for item in receipt.cases),
        case_result_digests=tuple(item.result_digest for item in receipt.cases),
    )
    if receipt.execution_run_digest != expected_run_digest:
        raise ValueError("v3 run provenance execution run digest binding mismatch")


def _replay_v3_run_receipt(value: object, path: str) -> V3RunReceipt:
    """Use Runtime's owning receipt replayer without accepting a caller receipt."""

    try:
        return _replay_v3_receipt(value, path)
    except (TypeError, ValueError) as exc:
        raise ReplayValidationError(f"{path}: receipt reconstruction failed:{exc}") from exc


@dataclass(frozen=True, slots=True)
class V3RunProductionProvenance:
    """Canonical frozen-semantics bindings and the exact receipt they derive.

    The implementation hashes are derived from local bytes by the builder and
    re-checked by reconstruction.  They are not caller-selected declarations.
    """

    receipt: V3RunReceipt
    source_snapshot_digest: str
    execution_transcript_digest: str
    producer_module_sha256: str
    bridge_module_sha256: str
    schema_version: str = V3_RUN_PRODUCTION_PROVENANCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.receipt, V3RunReceipt):
            raise ValueError("v3 run provenance requires a V3RunReceipt")
        if self.schema_version != V3_RUN_PRODUCTION_PROVENANCE_SCHEMA_VERSION:
            raise ValueError("v3 run provenance schema version mismatch")
        if (
            not isinstance(self.source_snapshot_digest, str)
            or not self.source_snapshot_digest
        ):
            raise ValueError("v3 run provenance source snapshot digest is invalid")
        if self.source_snapshot_digest != self.receipt.source_digest:
            raise ValueError("v3 run provenance source binding mismatch")
        _reject_synthetic_markers(self.receipt)
        _require_frozen_digest_semantics(self.receipt)
        if self.execution_transcript_digest != v3_run_execution_transcript_digest(
            self.receipt
        ):
            raise ValueError("v3 run provenance transcript binding mismatch")
        for field_name, expected in (
            ("producer_module_sha256", _module_sha256(_PRODUCER_FILENAME)),
            ("bridge_module_sha256", _module_sha256(_BRIDGE_FILENAME)),
        ):
            actual = getattr(self, field_name)
            if not isinstance(actual, str) or actual != expected:
                raise ValueError(
                    f"v3 run provenance implementation binding mismatch:{field_name}"
                )
        assert_deeply_immutable(self)

    @property
    def provenance_id(self) -> str:
        return stable_digest(
            "osc-phase6-v3-run-production-provenance-id-v1",
            {
                "source_snapshot_digest": self.receipt.source_digest,
                "v3_gate_plan_digest": self.receipt.v3_gate_plan_digest,
                "run_id": self.receipt.run_id,
                "receipt_digest": self.receipt.digest,
                "producer_module_sha256": self.producer_module_sha256,
                "bridge_module_sha256": self.bridge_module_sha256,
            },
        )

    @property
    def digest(self) -> str:
        return stable_digest("osc-phase6-v3-run-production-provenance", self)

    def to_dict(self) -> dict[str, object]:
        value = to_primitive(self)
        assert isinstance(value, dict)
        return value


def build_v3_run_production_provenance(
    *,
    receipt: V3RunReceipt,
) -> V3RunProductionProvenance:
    """Build the only private provenance object from one complete typed receipt."""

    if not isinstance(receipt, V3RunReceipt):
        raise TypeError("receipt must be a V3RunReceipt")
    return V3RunProductionProvenance(
        receipt=receipt,
        source_snapshot_digest=receipt.source_digest,
        execution_transcript_digest=v3_run_execution_transcript_digest(receipt),
        producer_module_sha256=_module_sha256(_PRODUCER_FILENAME),
        bridge_module_sha256=_module_sha256(_BRIDGE_FILENAME),
    )


def reconstruct_v3_run_receipt_payload(payload: object) -> V3RunReceipt:
    """Reconstruct the private v3 run receipt for Root cross-object binding."""

    receipt = _replay_v3_run_receipt(payload, "payload.V3RunReceipt")
    assert_payload_roundtrip(payload, receipt)
    return receipt


def reconstruct_v3_run_production_provenance_payload(
    payload: object,
) -> V3RunProductionProvenance:
    """Fail-closed canonical reconstruction used only by the Root bridge."""

    path = "payload.V3RunProductionProvenance"
    data = require_exact_mapping(
        payload,
        fields=(
            "receipt",
            "source_snapshot_digest",
            "execution_transcript_digest",
            "producer_module_sha256",
            "bridge_module_sha256",
            "schema_version",
        ),
        path=path,
    )
    provenance = V3RunProductionProvenance(
        receipt=_replay_v3_run_receipt(data["receipt"], f"{path}.receipt"),
        source_snapshot_digest=require_text(
            data["source_snapshot_digest"],
            path=f"{path}.source_snapshot_digest",
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


def canonical_v3_run_production_provenance_envelope(
    provenance: V3RunProductionProvenance,
) -> str:
    if not isinstance(provenance, V3RunProductionProvenance):
        raise TypeError("provenance must be a V3RunProductionProvenance")
    return canonical_envelope(
        V3_RUN_PRODUCTION_PROVENANCE_ENVELOPE_TYPE,
        V3_RUN_PRODUCTION_PROVENANCE_SCHEMA_VERSION,
        provenance,
    )


__all__ = [
    "V3_RUN_PRODUCTION_PROVENANCE_ENVELOPE_TYPE",
    "V3_RUN_PRODUCTION_PROVENANCE_SCHEMA_VERSION",
    "V3RunProductionProvenance",
    "build_v3_run_production_provenance",
    "canonical_v3_run_production_provenance_envelope",
    "reconstruct_v3_run_production_provenance_payload",
    "reconstruct_v3_run_receipt_payload",
    "v3_run_execution_transcript_digest",
]
