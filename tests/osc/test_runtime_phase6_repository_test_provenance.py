from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

import datadiff_osc
import datadiff_osc._phase6_gate_authority as authority_module
from datadiff_osc._canonical import canonical_envelope, canonical_json, stable_digest
from datadiff_osc.runtime._phase6_repository_test_provenance import (
    REPOSITORY_TEST_PRODUCTION_PROVENANCE_ENVELOPE_TYPE,
    REPOSITORY_TEST_PRODUCTION_PROVENANCE_SCHEMA_VERSION,
    RepositoryTestProductionProvenance,
    build_repository_test_production_provenance,
    canonical_repository_test_production_provenance_envelope,
    reconstruct_repository_test_production_provenance_payload,
    repository_test_execution_transcript_digest,
)
from datadiff_osc.runtime._private_receipts import (
    RepositoryTestNodeBinding,
    RepositoryTestReceipt,
    repository_test_collection_digest,
)


def _digest(label: str) -> str:
    return stable_digest("phase6-repository-provenance-test", label)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _receipt(
    *,
    source_digest: str | None = None,
    config_digest: str | None = None,
) -> RepositoryTestReceipt:
    source = source_digest or _digest("source")
    plan_digest = _digest("test-plan")
    command_digest = _digest("command")
    junit_sha = _sha256("repository-proof-junit")
    log_sha = _sha256("repository-proof-log")
    node_ids = ("tests/a.py::test_a", "tests/b.py::test_b")
    collection_digest = repository_test_collection_digest(node_ids)
    nodes = tuple(
        RepositoryTestNodeBinding(
            node_id=node_id,
            outcome="passed",
            result_digest=stable_digest(
                "phase6-repository-provenance-test-node",
                {
                    "source_digest": source,
                    "node_id": node_id,
                },
            ),
        )
        for node_id in node_ids
    )
    return RepositoryTestReceipt(
        source_digest=source,
        test_plan_digest=plan_digest,
        collection_digest=collection_digest,
        config_digest=config_digest or _digest("pytest-config"),
        environment_digest=_digest("environment"),
        command_digest=command_digest,
        junit_sha256=junit_sha,
        log_sha256=log_sha,
        exit_code=0,
        nodes=nodes,
    )


def _provenance(
    *,
    source_digest: str | None = None,
    config_digest: str | None = None,
) -> RepositoryTestProductionProvenance:
    return build_repository_test_production_provenance(
        receipt=_receipt(source_digest=source_digest, config_digest=config_digest),
        collection_sha256=_sha256("repository-proof-collection"),
    )


def _write_text(path: Path, text: str) -> str:
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _index_payload(
    *,
    provenance: RepositoryTestProductionProvenance,
    receipt_path: Path,
    receipt_sha256: str,
    provenance_path: Path | None,
    provenance_sha256: str | None,
    provenance_admission_id: str = "repository-admission-001",
    include_provenance: bool = True,
) -> dict[str, object]:
    source_digest = provenance.receipt.source_digest
    producer: dict[str, object] = {
        "schema_version": authority_module.TYPED_PRODUCER_RECEIPT_SCHEMA_VERSION,
        "receipt_id": "repository-proof-producer",
        "producer_kind": "repository_test_replay",
        "artifact_id": "repository-proof-artifact",
        "admissions": [
            {
                "admission_id": "repository-admission-001",
                "subject_kind": "repository_tests",
                "subject_ids": list(provenance.receipt.node_ids),
                "envelope_path": receipt_path.name,
                "envelope_sha256": receipt_sha256,
                "envelope_type": "RepositoryTestReceipt",
                "envelope_schema_version": provenance.receipt.schema_version,
            }
        ],
    }
    if include_provenance:
        assert provenance_path is not None
        assert provenance_sha256 is not None
        producer["repository_test_provenance"] = [
            {
                "admission_id": provenance_admission_id,
                "provenance_path": provenance_path.name,
                "provenance_sha256": provenance_sha256,
                "provenance_type": (
                    REPOSITORY_TEST_PRODUCTION_PROVENANCE_ENVELOPE_TYPE
                ),
                "provenance_schema_version": (
                    REPOSITORY_TEST_PRODUCTION_PROVENANCE_SCHEMA_VERSION
                ),
            }
        ]
    return {
        "schema_version": authority_module.ARTIFACT_RECEIPT_SCHEMA_VERSION,
        "source_digest": source_digest,
        "dynamic_plan_sha256": _sha256("repository-proof-dynamic-plan"),
        "producer_receipts": [producer],
        "artifacts": [],
    }


def _verify_index(tmp_path: Path, payload: dict[str, object]):
    index_path = tmp_path / "repository-proof-index.json"
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
    receipt_path = tmp_path / "repository-proof-receipt.json"
    receipt_sha256 = _write_text(
        receipt_path,
        canonical_envelope(
            "RepositoryTestReceipt",
            provenance.receipt.schema_version,
            provenance.receipt,
        ),
    )
    provenance_path = tmp_path / "repository-production-proof.json"
    provenance_sha256 = _write_text(
        provenance_path,
        canonical_repository_test_production_provenance_envelope(provenance),
    )
    payload = _index_payload(
        provenance=provenance,
        receipt_path=receipt_path,
        receipt_sha256=receipt_sha256,
        provenance_path=provenance_path,
        provenance_sha256=provenance_sha256,
    )
    return provenance, receipt_path, provenance_path, payload


def test_private_repository_provenance_rebuilds_r2_receipt_without_public_authority():
    provenance = _provenance()
    envelope = canonical_repository_test_production_provenance_envelope(provenance)
    decoded = json.loads(envelope)

    assert "RepositoryTestProductionProvenance" not in datadiff_osc.__all__
    assert provenance.source_snapshot_digest == provenance.receipt.source_digest
    assert provenance.junit_sha256 == provenance.receipt.junit_sha256
    assert provenance.log_sha256 == provenance.receipt.log_sha256
    assert provenance.execution_transcript_digest == (
        repository_test_execution_transcript_digest(provenance.receipt)
    )
    assert provenance.provenance_id
    assert reconstruct_repository_test_production_provenance_payload(
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
            {"junit_sha256": hashlib.sha256(b"forged-junit").hexdigest()}
        ),
        lambda payload: payload["receipt"].update(
            {"config_digest": _digest("forged-config")}
        ),
        lambda payload: payload.update(
            {"execution_transcript_digest": _digest("forged-transcript")}
        ),
        lambda payload: payload.update({"producer_module_sha256": "0" * 64}),
        lambda payload: payload.update({"collection_sha256": "not-a-sha"}),
    ),
)
def test_private_repository_provenance_rejects_malformed_or_misbound_payload(mutate):
    provenance = _provenance()
    payload = json.loads(canonical_json(provenance))

    mutate(payload)

    with pytest.raises((ValueError, TypeError)):
        reconstruct_repository_test_production_provenance_payload(payload)


def test_private_repository_provenance_rejects_wrong_object_and_module_binding():
    with pytest.raises(TypeError, match="RepositoryTestReceipt"):
        build_repository_test_production_provenance(
            receipt=object(),  # type: ignore[arg-type]
            collection_sha256=_sha256("collection"),
        )
    with pytest.raises(TypeError, match="RepositoryTestProductionProvenance"):
        canonical_repository_test_production_provenance_envelope(object())  # type: ignore[arg-type]
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
            lambda provenance: replace(provenance, collection_sha256="xyz"),
            "raw binding is invalid:collection_sha256",
        ),
        (
            lambda provenance: replace(
                provenance,
                junit_sha256=hashlib.sha256(b"other-junit").hexdigest(),
            ),
            "junit binding mismatch",
        ),
        (
            lambda provenance: replace(
                provenance,
                log_sha256=hashlib.sha256(b"other-log").hexdigest(),
            ),
            "log binding mismatch",
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
def test_private_repository_provenance_rejects_every_field_rebinding(
    mutate, expected_error
):
    provenance = _provenance()

    with pytest.raises(ValueError, match=expected_error):
        mutate(provenance)


def test_private_repository_provenance_rejects_non_receipt_object():
    provenance = _provenance()

    with pytest.raises(ValueError, match="requires a RepositoryTestReceipt"):
        RepositoryTestProductionProvenance(
            receipt=None,  # type: ignore[arg-type]
            source_snapshot_digest=provenance.source_snapshot_digest,
            collection_sha256=provenance.collection_sha256,
            junit_sha256=provenance.junit_sha256,
            log_sha256=provenance.log_sha256,
            execution_transcript_digest=provenance.execution_transcript_digest,
            producer_module_sha256=provenance.producer_module_sha256,
            bridge_module_sha256=provenance.bridge_module_sha256,
        )


def test_root_retains_only_exact_repository_provenance_pair_in_incomplete_index(
    tmp_path,
):
    provenance, _, _, payload = _provenance_index(tmp_path)

    verified = _verify_index(tmp_path, payload)

    assert not verified.valid
    producer = verified.producer_map()["repository_test_replay"]
    assert tuple(item.admission_id for item in producer.admissions) == (
        "repository-admission-001",
    )
    assert tuple(item.admission_id for item in producer.repository_test_provenance) == (
        "repository-admission-001",
    )
    assert producer.repository_test_provenance[0].receipt_digest == (
        provenance.receipt.digest
    )
    assert producer.repository_test_provenance[0].collection_digest == (
        provenance.receipt.collection_digest
    )
    assert any(
        error.startswith("typed_producer_receipt_producers_missing:")
        for error in verified.verification_errors
    )
    assert "artifact_receipt_index_empty" in verified.verification_errors
    assert not any(
        error.startswith("repository_test_provenance_")
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
def test_root_repository_provenance_pairing_fails_closed_on_binding_variants(
    tmp_path, case
):
    provenance, receipt_path, provenance_path, payload = _provenance_index(tmp_path)
    producer = payload["producer_receipts"][0]
    assert isinstance(producer, dict)

    if case == "missing":
        del producer["repository_test_provenance"]
    elif case == "wrong_admission":
        proof = producer["repository_test_provenance"]
        assert isinstance(proof, list) and isinstance(proof[0], dict)
        proof[0]["admission_id"] = "other-admission"
    elif case == "duplicate":
        proof = producer["repository_test_provenance"]
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
        rebound_text = canonical_repository_test_production_provenance_envelope(
            rebound
        )
        rebound_sha256 = _write_text(provenance_path, rebound_text)
        proof = producer["repository_test_provenance"]
        assert isinstance(proof, list) and isinstance(proof[0], dict)
        proof[0]["provenance_sha256"] = rebound_sha256
    else:  # pragma: no cover - protects the parametrized case list.
        raise AssertionError(case)

    verified = _verify_index(tmp_path, payload)

    producer_result = verified.producer_map()["repository_test_replay"]
    assert producer_result.repository_test_provenance == ()
    assert not any(
        admission.envelope_type == "RepositoryTestReceipt"
        for admission in producer_result.admissions
    )
    assert any(
        error.startswith("repository_test_provenance_")
        or error.startswith("typed_admission_")
        for error in verified.verification_errors
    )
    assert not verified.valid
    assert provenance.receipt.collection_digest
