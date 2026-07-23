from __future__ import annotations

from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

import datadiff_osc
import datadiff_osc._phase6_gate_authority as authority_module
from datadiff_osc._canonical import canonical_envelope, canonical_json, stable_digest
from datadiff_osc.runtime._phase6_contract_performance_provenance import (
    CONTRACT_PERFORMANCE_PRODUCTION_PROVENANCE_ENVELOPE_TYPE,
    CONTRACT_PERFORMANCE_PRODUCTION_PROVENANCE_SCHEMA_VERSION,
    ContractPerformanceProductionProvenance,
    build_contract_performance_production_provenance,
    canonical_contract_performance_production_provenance_envelope,
    reconstruct_contract_performance_production_provenance_payload,
)
from datadiff_osc.runtime._phase6_contract_performance_receipts import (
    CONTRACT_PERFORMANCE_EVIDENCE_RECEIPT_SCHEMA_VERSION,
)


def _sha256(value: str | bytes) -> str:
    raw = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(raw).hexdigest()


def _digest(label: str) -> str:
    return stable_digest("test-contract-performance-provenance", label)


def _observation():
    """Reuse the immutable R4 legal construction only as a test input fixture."""

    path = Path(__file__).with_name(
        "test_runtime_phase6_contract_performance_producers.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_phase6_contract_performance_producers_fixture", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._observation()


def _write_text(path: Path, text: str) -> str:
    path.write_text(text, encoding="utf-8")
    return _sha256(text)


def _index_payload(
    *,
    provenance: ContractPerformanceProductionProvenance,
    receipt_path: Path,
    receipt_sha256: str,
    provenance_path: Path | None,
    provenance_sha256: str | None,
    provenance_admission_id: str = "contract-admission-001",
    include_provenance: bool = True,
) -> dict[str, object]:
    source_digest = provenance.receipt.source_snapshot_digest
    producer: dict[str, object] = {
        "schema_version": authority_module.TYPED_PRODUCER_RECEIPT_SCHEMA_VERSION,
        "receipt_id": "contract-proof-producer",
        "producer_kind": "performance_paired_replay",
        "artifact_id": "contract-proof-artifact",
        "admissions": [
            {
                "admission_id": "contract-admission-001",
                "subject_kind": "contract_performance_samples",
                "subject_ids": [provenance.receipt.sample_id],
                "envelope_path": receipt_path.name,
                "envelope_sha256": receipt_sha256,
                "envelope_type": "ContractPerformanceEvidenceReceipt",
                "envelope_schema_version": provenance.receipt.schema_version,
            }
        ],
    }
    if include_provenance:
        assert provenance_path is not None
        assert provenance_sha256 is not None
        producer["contract_performance_provenance"] = [
            {
                "admission_id": provenance_admission_id,
                "provenance_path": provenance_path.name,
                "provenance_sha256": provenance_sha256,
                "provenance_type": (
                    CONTRACT_PERFORMANCE_PRODUCTION_PROVENANCE_ENVELOPE_TYPE
                ),
                "provenance_schema_version": (
                    CONTRACT_PERFORMANCE_PRODUCTION_PROVENANCE_SCHEMA_VERSION
                ),
            }
        ]
    return {
        "schema_version": authority_module.ARTIFACT_RECEIPT_SCHEMA_VERSION,
        "source_digest": source_digest,
        "dynamic_plan_sha256": _sha256("contract-proof-dynamic-plan"),
        "producer_receipts": [producer],
        "artifacts": [],
    }


def _verify_index(tmp_path: Path, payload: dict[str, object]):
    index_path = tmp_path / "contract-proof-index.json"
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    index_sha256 = _write_text(index_path, raw)
    return authority_module.verify_artifact_receipt_index(
        index_path=index_path,
        expected_index_sha256=index_sha256,
        expected_source_digest=str(payload["source_digest"]),
        expected_dynamic_plan_sha256=str(payload["dynamic_plan_sha256"]),
    )


def _provenance_index(tmp_path: Path):
    provenance = build_contract_performance_production_provenance(_observation())
    receipt_path = tmp_path / "contract-proof-receipt.json"
    receipt_sha256 = _write_text(
        receipt_path,
        canonical_envelope(
            "ContractPerformanceEvidenceReceipt",
            CONTRACT_PERFORMANCE_EVIDENCE_RECEIPT_SCHEMA_VERSION,
            provenance.receipt,
        ),
    )
    provenance_path = tmp_path / "contract-production-proof.json"
    provenance_sha256 = _write_text(
        provenance_path,
        canonical_contract_performance_production_provenance_envelope(provenance),
    )
    payload = _index_payload(
        provenance=provenance,
        receipt_path=receipt_path,
        receipt_sha256=receipt_sha256,
        provenance_path=provenance_path,
        provenance_sha256=provenance_sha256,
    )
    return provenance, receipt_path, provenance_path, payload


def test_private_contract_performance_provenance_rebuilds_r4_receipt():
    observation = _observation()
    provenance = build_contract_performance_production_provenance(observation)
    envelope = canonical_contract_performance_production_provenance_envelope(provenance)
    decoded = json.loads(envelope)

    assert "ContractPerformanceProductionProvenance" not in datadiff_osc.__all__
    assert provenance.receipt.source_snapshot_digest == (
        observation.comparison.source_snapshot_digest
    )
    assert provenance.receipt.sample_id
    assert provenance.provenance_id
    assert reconstruct_contract_performance_production_provenance_payload(
        decoded["payload"]
    ) == provenance

    changed = build_contract_performance_production_provenance(
        replace(
            observation,
            treatment_counters=replace(
                observation.treatment_counters,
                comparison_cpu_ns=observation.treatment_counters.comparison_cpu_ns + 1,
            ),
        )
    )
    assert changed.receipt.sample_id != provenance.receipt.sample_id
    assert changed.provenance_id != provenance.provenance_id


@pytest.mark.parametrize(
    "mutate",
    (
        lambda payload: payload.pop("receipt"),
        lambda payload: payload.update({"caller_digest": _digest("forged")}),
        lambda payload: payload["receipt"].update(
            {"treatment_comparison_cpu_ns": 3_001}
        ),
        lambda payload: payload["observation"]["baseline_counters"].update(
            {"materialized_bytes": 0}
        ),
        lambda payload: payload.update({"producer_module_sha256": "0" * 64}),
    ),
)
def test_private_contract_performance_provenance_rejects_malformed_or_misbound_payload(
    mutate,
):
    provenance = build_contract_performance_production_provenance(_observation())
    payload = json.loads(canonical_json(provenance))

    mutate(payload)

    with pytest.raises((ValueError, TypeError)):
        reconstruct_contract_performance_production_provenance_payload(payload)


def test_private_contract_performance_provenance_rejects_wrong_object_and_module():
    with pytest.raises(TypeError, match="ContractPerformanceProductionObservation"):
        build_contract_performance_production_provenance(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="ContractPerformanceProductionProvenance"):
        canonical_contract_performance_production_provenance_envelope(
            object()  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="implementation binding mismatch"):
        replace(
            build_contract_performance_production_provenance(_observation()),
            producer_module_sha256="0" * 64,
        )


def test_root_retains_only_exact_contract_provenance_pair_in_incomplete_index(
    tmp_path,
):
    provenance, _, _, payload = _provenance_index(tmp_path)

    verified = _verify_index(tmp_path, payload)

    assert not verified.valid
    producer = verified.producer_map()["performance_paired_replay"]
    assert tuple(item.admission_id for item in producer.admissions) == (
        "contract-admission-001",
    )
    assert tuple(
        item.admission_id for item in producer.contract_performance_provenance
    ) == ("contract-admission-001",)
    assert producer.contract_performance_provenance[0].receipt_digest == (
        provenance.receipt.digest
    )
    assert any(
        error.startswith("typed_producer_receipt_payload_types_missing:")
        for error in verified.verification_errors
    )
    assert "artifact_receipt_index_empty" in verified.verification_errors
    assert not any(
        error.startswith("contract_performance_provenance_")
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
        "wrong_type",
    ),
)
def test_root_contract_provenance_pairing_fails_closed_on_binding_variants(
    tmp_path,
    case,
):
    provenance, receipt_path, provenance_path, payload = _provenance_index(tmp_path)
    producer = payload["producer_receipts"][0]
    assert isinstance(producer, dict)

    if case == "missing":
        del producer["contract_performance_provenance"]
    elif case == "wrong_admission":
        proof = producer["contract_performance_provenance"]
        assert isinstance(proof, list) and isinstance(proof[0], dict)
        proof[0]["admission_id"] = "other-admission"
    elif case == "duplicate":
        proof = producer["contract_performance_provenance"]
        assert isinstance(proof, list) and isinstance(proof[0], dict)
        proof.append(dict(proof[0]))
    elif case == "proof_bytes":
        provenance_path.write_text(
            provenance_path.read_text(encoding="utf-8") + "\n", encoding="utf-8"
        )
    elif case == "receipt_bytes":
        receipt_path.write_text(
            receipt_path.read_text(encoding="utf-8") + "\n", encoding="utf-8"
        )
    elif case == "source_rebind":
        rebound = build_contract_performance_production_provenance(
            replace(
                provenance.observation,
                comparison=replace(
                    provenance.observation.comparison,
                    source_snapshot_digest=_digest("rebound-source"),
                ),
            )
        )
        rebound_sha256 = _write_text(
            provenance_path,
            canonical_contract_performance_production_provenance_envelope(rebound),
        )
        proof = producer["contract_performance_provenance"]
        assert isinstance(proof, list) and isinstance(proof[0], dict)
        proof[0]["provenance_sha256"] = rebound_sha256
    else:
        proof = producer["contract_performance_provenance"]
        assert isinstance(proof, list) and isinstance(proof[0], dict)
        proof[0]["provenance_type"] = "ParallelScalingProductionProvenance"

    verified = _verify_index(tmp_path, payload)
    producer_result = verified.producer_map()["performance_paired_replay"]

    assert producer_result.admissions == ()
    assert producer_result.contract_performance_provenance == ()
    assert any(
        error.startswith("contract_performance_provenance_")
        or error.startswith("typed_admission_root_provenance_pending_phase6:")
        or error.startswith("typed_producer_receipt_payload_types_missing:")
        for error in verified.verification_errors
    )
