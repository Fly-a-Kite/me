from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

import datadiff_osc
import datadiff_osc._phase6_gate_authority as authority_module
from datadiff_osc._canonical import canonical_envelope, canonical_json, stable_digest
from datadiff_osc.runtime._phase6_target_version_provenance import (
    TARGET_VERSION_PRODUCTION_PROVENANCE_ENVELOPE_TYPE,
    TARGET_VERSION_PRODUCTION_PROVENANCE_SCHEMA_VERSION,
    TargetVersionProductionProvenance,
    build_target_version_production_provenance,
    canonical_target_version_production_provenance_envelope,
    reconstruct_target_version_production_provenance_payload,
    target_version_execution_transcript_digest,
)
from datadiff_osc.runtime._private_receipts import (
    TargetPackageVersionBinding,
    TargetVersionReceipt,
)


def _digest(label: str) -> str:
    return stable_digest("phase6-target-version-provenance-test", label)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _receipt(
    *,
    source_digest: str | None = None,
    environment_digest: str | None = None,
) -> TargetVersionReceipt:
    source = source_digest or _digest("source")
    packages = tuple(
        TargetPackageVersionBinding(
            package_id=package_id,
            distribution_name=distribution,
            import_name=import_name,
            installed_version=version,
            latest_version=version,
            installed_metadata_digest=_digest(f"metadata-{package_id}"),
            version_source_kind="pypi_json",
            version_source_digest=_digest(f"version-source-{package_id}"),
            version_source_sha256=_sha256(f"version-source-bytes-{package_id}"),
        )
        for package_id, distribution, import_name, version in (
            ("target:alpha", "alpha-dist", "alpha", "1.2.3"),
            ("target:beta", "beta-dist", "beta", "4.5.6"),
        )
    )
    return TargetVersionReceipt(
        source_digest=source,
        audit_plan_digest=_digest("audit-plan"),
        environment_digest=environment_digest or _digest("environment"),
        python_runtime_digest=_digest("python-runtime"),
        packages=packages,
    )


def _provenance(
    *,
    source_digest: str | None = None,
    environment_digest: str | None = None,
) -> TargetVersionProductionProvenance:
    return build_target_version_production_provenance(
        receipt=_receipt(
            source_digest=source_digest, environment_digest=environment_digest
        ),
        manifest_sha256=_sha256("target-version-proof-manifest"),
    )


def _write_text(path: Path, text: str) -> str:
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _index_payload(
    *,
    provenance: TargetVersionProductionProvenance,
    receipt_path: Path,
    receipt_sha256: str,
    provenance_path: Path | None,
    provenance_sha256: str | None,
    provenance_admission_id: str = "target-version-admission-001",
    include_provenance: bool = True,
) -> dict[str, object]:
    source_digest = provenance.receipt.source_digest
    producer: dict[str, object] = {
        "schema_version": authority_module.TYPED_PRODUCER_RECEIPT_SCHEMA_VERSION,
        "receipt_id": "target-version-proof-producer",
        "producer_kind": "target_version_replay",
        "artifact_id": "target-version-proof-artifact",
        "admissions": [
            {
                "admission_id": "target-version-admission-001",
                "subject_kind": "target_packages",
                "subject_ids": list(provenance.receipt.package_ids),
                "envelope_path": receipt_path.name,
                "envelope_sha256": receipt_sha256,
                "envelope_type": "TargetVersionReceipt",
                "envelope_schema_version": provenance.receipt.schema_version,
            }
        ],
    }
    if include_provenance:
        assert provenance_path is not None
        assert provenance_sha256 is not None
        producer["target_version_provenance"] = [
            {
                "admission_id": provenance_admission_id,
                "provenance_path": provenance_path.name,
                "provenance_sha256": provenance_sha256,
                "provenance_type": (
                    TARGET_VERSION_PRODUCTION_PROVENANCE_ENVELOPE_TYPE
                ),
                "provenance_schema_version": (
                    TARGET_VERSION_PRODUCTION_PROVENANCE_SCHEMA_VERSION
                ),
            }
        ]
    return {
        "schema_version": authority_module.ARTIFACT_RECEIPT_SCHEMA_VERSION,
        "source_digest": source_digest,
        "dynamic_plan_sha256": _sha256("target-version-proof-dynamic-plan"),
        "producer_receipts": [producer],
        "artifacts": [],
    }


def _verify_index(tmp_path: Path, payload: dict[str, object]):
    index_path = tmp_path / "target-version-proof-index.json"
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    index_sha256 = _write_text(index_path, raw)
    return authority_module.verify_artifact_receipt_index(
        index_path=index_path,
        expected_index_sha256=index_sha256,
        expected_source_digest=str(payload["source_digest"]),
        expected_dynamic_plan_sha256=str(payload["dynamic_plan_sha256"]),
    )


def _provenance_index(tmp_path: Path):
    provenance = _provenance()
    receipt_path = tmp_path / "target-version-proof-receipt.json"
    receipt_sha256 = _write_text(
        receipt_path,
        canonical_envelope(
            "TargetVersionReceipt",
            provenance.receipt.schema_version,
            provenance.receipt,
        ),
    )
    provenance_path = tmp_path / "target-version-production-proof.json"
    provenance_sha256 = _write_text(
        provenance_path,
        canonical_target_version_production_provenance_envelope(provenance),
    )
    payload = _index_payload(
        provenance=provenance,
        receipt_path=receipt_path,
        receipt_sha256=receipt_sha256,
        provenance_path=provenance_path,
        provenance_sha256=provenance_sha256,
    )
    return provenance, receipt_path, provenance_path, payload


def test_private_target_version_provenance_rebuilds_receipt_without_public_authority():
    provenance = _provenance()
    envelope = canonical_target_version_production_provenance_envelope(provenance)
    decoded = json.loads(envelope)

    assert "TargetVersionProductionProvenance" not in datadiff_osc.__all__
    assert provenance.source_snapshot_digest == provenance.receipt.source_digest
    assert provenance.execution_transcript_digest == (
        target_version_execution_transcript_digest(provenance.receipt)
    )
    assert provenance.provenance_id
    assert reconstruct_target_version_production_provenance_payload(
        decoded["payload"]
    ) == provenance

    changed = _provenance(source_digest=_digest("other-source"))
    assert changed.receipt.digest != provenance.receipt.digest
    assert changed.provenance_id != provenance.provenance_id


@pytest.mark.parametrize(
    "mutate",
    (
        lambda payload: payload.pop("receipt"),
        lambda payload: payload.update({"caller_digest": _digest("forged")}),
        lambda payload: payload["receipt"].update(
            {"source_digest": _digest("forged-source")}
        ),
        lambda payload: payload["receipt"].update(
            {"environment_digest": _digest("forged-environment")}
        ),
        lambda payload: payload.update(
            {"execution_transcript_digest": _digest("forged-transcript")}
        ),
        lambda payload: payload.update({"producer_module_sha256": "0" * 64}),
        lambda payload: payload.update({"manifest_sha256": "not-a-sha"}),
    ),
)
def test_private_target_version_provenance_rejects_malformed_or_misbound_payload(
    mutate,
):
    provenance = _provenance()
    payload = json.loads(canonical_json(provenance))

    mutate(payload)

    with pytest.raises((ValueError, TypeError)):
        reconstruct_target_version_production_provenance_payload(payload)


def test_private_target_version_provenance_rejects_wrong_object_and_module_binding():
    with pytest.raises(TypeError, match="TargetVersionReceipt"):
        build_target_version_production_provenance(
            receipt=object(),  # type: ignore[arg-type]
            manifest_sha256=_sha256("manifest"),
        )
    with pytest.raises(TypeError, match="TargetVersionProductionProvenance"):
        canonical_target_version_production_provenance_envelope(object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="implementation binding mismatch"):
        replace(_provenance(), producer_module_sha256="0" * 64)


@pytest.mark.parametrize(
    ("mutate", "expected_error"),
    (
        (
            lambda provenance: replace(provenance, schema_version="osc-forged-v1"),
            "schema version mismatch",
        ),
        (
            lambda provenance: replace(
                provenance, source_snapshot_digest=_digest("other-source")
            ),
            "source binding mismatch",
        ),
        (
            lambda provenance: replace(provenance, manifest_sha256="xyz"),
            "raw binding is invalid:manifest_sha256",
        ),
        (
            lambda provenance: replace(
                provenance,
                execution_transcript_digest=_digest("forged-transcript"),
            ),
            "transcript binding mismatch",
        ),
        (
            lambda provenance: replace(provenance, producer_module_sha256="0" * 64),
            "implementation binding mismatch:producer_module_sha256",
        ),
        (
            lambda provenance: replace(provenance, bridge_module_sha256="0" * 64),
            "implementation binding mismatch:bridge_module_sha256",
        ),
    ),
)
def test_private_target_version_provenance_rejects_every_field_rebinding(
    mutate, expected_error
):
    provenance = _provenance()

    with pytest.raises(ValueError, match=expected_error):
        mutate(provenance)


def test_private_target_version_provenance_rejects_non_receipt_object():
    provenance = _provenance()

    with pytest.raises(ValueError, match="requires a TargetVersionReceipt"):
        TargetVersionProductionProvenance(
            receipt=None,  # type: ignore[arg-type]
            source_snapshot_digest=provenance.source_snapshot_digest,
            manifest_sha256=provenance.manifest_sha256,
            execution_transcript_digest=provenance.execution_transcript_digest,
            producer_module_sha256=provenance.producer_module_sha256,
            bridge_module_sha256=provenance.bridge_module_sha256,
        )


def test_root_retains_only_exact_target_version_provenance_pair_in_incomplete_index(
    tmp_path,
):
    provenance, _, _, payload = _provenance_index(tmp_path)

    verified = _verify_index(tmp_path, payload)

    assert not verified.valid
    producer = verified.producer_map()["target_version_replay"]
    assert tuple(item.admission_id for item in producer.admissions) == (
        "target-version-admission-001",
    )
    assert tuple(item.admission_id for item in producer.target_version_provenance) == (
        "target-version-admission-001",
    )
    assert producer.target_version_provenance[0].receipt_digest == (
        provenance.receipt.digest
    )
    assert producer.target_version_provenance[0].audit_plan_digest == (
        provenance.receipt.audit_plan_digest
    )
    assert any(
        error.startswith("typed_producer_receipt_producers_missing:")
        for error in verified.verification_errors
    )
    assert "artifact_receipt_index_empty" in verified.verification_errors
    assert not any(
        error.startswith("target_version_provenance_")
        for error in verified.verification_errors
    )


@pytest.mark.parametrize(
    "case",
    (
        "missing",
        "wrong_admission",
        "duplicate",
        "proof_bytes",
        "receipt_bytes",
        "source_rebind",
    ),
)
def test_root_target_version_provenance_pairing_fails_closed_on_binding_variants(
    tmp_path, case
):
    provenance, receipt_path, provenance_path, payload = _provenance_index(tmp_path)
    producer = payload["producer_receipts"][0]
    assert isinstance(producer, dict)

    if case == "missing":
        del producer["target_version_provenance"]
    elif case == "wrong_admission":
        proof = producer["target_version_provenance"]
        assert isinstance(proof, list) and isinstance(proof[0], dict)
        proof[0]["admission_id"] = "other-admission"
    elif case == "duplicate":
        proof = producer["target_version_provenance"]
        assert isinstance(proof, list) and isinstance(proof[0], dict)
        proof.append(dict(proof[0]))
    elif case == "proof_bytes":
        provenance_path.write_text(
            provenance_path.read_text(encoding="utf-8") + "\n",
            encoding="utf-8",
        )
    elif case == "receipt_bytes":
        receipt_path.write_text(
            receipt_path.read_text(encoding="utf-8") + "\n",
            encoding="utf-8",
        )
    elif case == "source_rebind":
        rebound = _provenance(source_digest=_digest("rebound-source"))
        rebound_text = canonical_target_version_production_provenance_envelope(
            rebound
        )
        rebound_sha256 = _write_text(provenance_path, rebound_text)
        proof = producer["target_version_provenance"]
        assert isinstance(proof, list) and isinstance(proof[0], dict)
        proof[0]["provenance_sha256"] = rebound_sha256
    else:  # pragma: no cover - protects the parametrized case list.
        raise AssertionError(case)

    verified = _verify_index(tmp_path, payload)

    producer_result = verified.producer_map()["target_version_replay"]
    assert producer_result.target_version_provenance == ()
    assert not any(
        admission.envelope_type == "TargetVersionReceipt"
        for admission in producer_result.admissions
    )
    assert any(
        error.startswith("target_version_provenance_")
        or error.startswith("typed_admission_")
        for error in verified.verification_errors
    )
    assert not verified.valid
    assert provenance.receipt.audit_plan_digest
