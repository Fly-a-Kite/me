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
from datadiff_osc.runtime._phase6_contract_performance_execution_provenance import (
    CONTRACT_PERFORMANCE_EXECUTION_PROVENANCE_ENVELOPE_TYPE,
    CONTRACT_PERFORMANCE_EXECUTION_PROVENANCE_SCHEMA_VERSION,
    ContractPerformanceExecutionProvenance,
    ContractPerformanceExecutionTranscript,
    build_contract_performance_execution_provenance,
    canonical_contract_performance_execution_provenance_envelope,
    reconstruct_contract_performance_execution_provenance_payload,
)
from datadiff_osc.runtime._phase6_contract_performance_provenance import (
    build_contract_performance_production_provenance,
    canonical_contract_performance_production_provenance_envelope,
)
from datadiff_osc.runtime._phase6_contract_performance_receipts import (
    CONTRACT_PERFORMANCE_EVIDENCE_RECEIPT_SCHEMA_VERSION,
)


ROOT = Path(__file__).resolve().parents[2]


def _sha256(value: str | bytes) -> str:
    raw = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(raw).hexdigest()


def _digest(label: str) -> str:
    return stable_digest("test-contract-performance-execution-provenance", label)


def _write_text(path: Path, text: str) -> str:
    path.write_text(text, encoding="utf-8")
    return _sha256(text)


def _r9_fixture_module():
    path = Path(__file__).with_name("test_runtime_phase6_contract_performance_provenance.py")
    spec = importlib.util.spec_from_file_location(
        "_phase6_contract_performance_execution_r9_fixture", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _source_snapshot(tmp_path: Path):
    git_state = authority_module._current_git_state(ROOT)
    assert git_state is not None
    payload = authority_module.source_snapshot_payload(
        repo_root=ROOT,
        relative_paths=sorted(authority_module._required_source_paths(ROOT)),
        git_head=git_state[0],
        git_status_digest=git_state[1],
    )
    path = tmp_path / "source-snapshot.json"
    expected_sha = _write_text(path, canonical_json(payload))
    return authority_module.verify_source_snapshot(
        repo_root=ROOT,
        snapshot_path=path,
        expected_snapshot_sha256=expected_sha,
    )


def _production_provenance(source_digest: str):
    fixture = _r9_fixture_module()
    observation = fixture._observation()
    rebound = replace(
        observation,
        comparison=replace(
            observation.comparison,
            source_snapshot_digest=source_digest,
        ),
    )
    return build_contract_performance_production_provenance(rebound)


def _verify_index(tmp_path: Path, payload: dict[str, object]):
    index_path = tmp_path / "contract-execution-index.json"
    index_sha256 = _write_text(index_path, canonical_json(payload))
    return authority_module.verify_artifact_receipt_index(
        index_path=index_path,
        expected_index_sha256=index_sha256,
        expected_source_digest=str(payload["source_digest"]),
        expected_dynamic_plan_sha256=str(payload["dynamic_plan_sha256"]),
    )


def _execution_index(tmp_path: Path):
    source = _source_snapshot(tmp_path)
    production = _production_provenance(source.source_digest)
    execution = build_contract_performance_execution_provenance(production)
    fixture = _r9_fixture_module()

    receipt_path = tmp_path / "contract-execution-receipt.json"
    receipt_sha256 = _write_text(
        receipt_path,
        canonical_envelope(
            "ContractPerformanceEvidenceReceipt",
            CONTRACT_PERFORMANCE_EVIDENCE_RECEIPT_SCHEMA_VERSION,
            production.receipt,
        ),
    )
    production_path = tmp_path / "contract-production-proof.json"
    production_sha256 = _write_text(
        production_path,
        canonical_contract_performance_production_provenance_envelope(production),
    )
    execution_path = tmp_path / "contract-execution-proof.json"
    execution_sha256 = _write_text(
        execution_path,
        canonical_contract_performance_execution_provenance_envelope(execution),
    )
    artifact_path = tmp_path / "contract-execution-artifact.json"
    artifact_path.write_bytes(execution.artifact_bytes)

    payload = fixture._index_payload(
        provenance=production,
        receipt_path=receipt_path,
        receipt_sha256=receipt_sha256,
        provenance_path=production_path,
        provenance_sha256=production_sha256,
    )
    producer = payload["producer_receipts"][0]
    assert isinstance(producer, dict)
    producer["artifact_id"] = execution.artifact_id
    provisional = _verify_index(tmp_path, payload)
    producer_digest = provisional.producer_map()["performance_paired_replay"].digest
    payload["artifacts"] = [
        {
            "artifact_id": execution.artifact_id,
            "path": artifact_path.name,
            "sha256": execution.artifact_sha256,
            "provenance_digest": execution.digest,
            "producer_kind": "performance_paired_replay",
            "producer_receipt_digest": producer_digest,
            "contract_performance_execution_provenance": [
                {
                    "admission_id": "contract-admission-001",
                    "provenance_path": execution_path.name,
                    "provenance_sha256": execution_sha256,
                    "provenance_type": (
                        CONTRACT_PERFORMANCE_EXECUTION_PROVENANCE_ENVELOPE_TYPE
                    ),
                    "provenance_schema_version": (
                        CONTRACT_PERFORMANCE_EXECUTION_PROVENANCE_SCHEMA_VERSION
                    ),
                }
            ],
        }
    ]
    return (
        source,
        production,
        execution,
        execution_path,
        artifact_path,
        payload,
    )


def test_private_execution_provenance_rebuilds_r9_proof_and_raw_artifact(tmp_path):
    source, production, execution, _, _, _, = _execution_index(tmp_path)
    envelope = canonical_contract_performance_execution_provenance_envelope(execution)
    decoded = json.loads(envelope)

    assert "ContractPerformanceExecutionProvenance" not in datadiff_osc.__all__
    assert source.valid
    assert execution.source_snapshot_digest == source.source_digest
    assert execution.receipt == production.receipt
    assert execution.transcript.exit_code == 0
    assert execution.transcript.stdout == execution.artifact_bytes
    assert execution.transcript.output_sha256 == execution.artifact_sha256
    assert reconstruct_contract_performance_execution_provenance_payload(
        decoded["payload"]
    ) == execution

    raw_artifact = json.loads(execution.artifact_bytes)
    assert raw_artifact["artifact_id"] == execution.artifact_id
    assert raw_artifact["source_digest"] == source.source_digest
    assert len(raw_artifact["series"]) == 6

    changed_production = build_contract_performance_production_provenance(
        replace(
            production.observation,
            treatment_counters=replace(
                production.observation.treatment_counters,
                comparison_cpu_ns=(
                    production.observation.treatment_counters.comparison_cpu_ns + 1
                ),
            ),
        )
    )
    changed = build_contract_performance_execution_provenance(changed_production)
    assert changed.receipt.sample_id != execution.receipt.sample_id
    assert changed.artifact_id != execution.artifact_id
    assert changed.artifact_bytes != execution.artifact_bytes
    assert changed.provenance_id != execution.provenance_id


@pytest.mark.parametrize(
    "mutate",
    (
        lambda payload: payload.pop("artifact_bytes"),
        lambda payload: payload.update({"caller_digest": _digest("forged")}),
        lambda payload: payload["transcript"].update({"exit_code": 1}),
        lambda payload: payload["transcript"].update({"stdout": {"$bytes": "AA=="}}),
        lambda payload: payload["production_provenance"]["receipt"].update(
            {"treatment_comparison_cpu_ns": 3_001}
        ),
        lambda payload: payload.update({"execution_module_sha256": "0" * 64}),
    ),
)
def test_private_execution_provenance_rejects_malformed_or_misbound_payload(
    tmp_path,
    mutate,
):
    source = _source_snapshot(tmp_path)
    execution = build_contract_performance_execution_provenance(
        _production_provenance(source.source_digest)
    )
    payload = json.loads(canonical_json(execution))
    mutate(payload)

    with pytest.raises((TypeError, ValueError)):
        reconstruct_contract_performance_execution_provenance_payload(payload)


def test_private_execution_provenance_rejects_wrong_objects_and_nonzero_exit(tmp_path):
    source = _source_snapshot(tmp_path)
    execution = build_contract_performance_execution_provenance(
        _production_provenance(source.source_digest)
    )

    with pytest.raises(TypeError, match="ContractPerformanceProductionProvenance"):
        build_contract_performance_execution_provenance(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="ContractPerformanceExecutionProvenance"):
        canonical_contract_performance_execution_provenance_envelope(
            object()  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="exact zero exit"):
        replace(execution.transcript, exit_code=1)
    with pytest.raises(ValueError, match="output/artifact mismatch"):
        replace(
            execution,
            transcript=ContractPerformanceExecutionTranscript(
                command=execution.transcript.command,
                cwd=execution.transcript.cwd,
                environment=execution.transcript.environment,
                stdout=b"{}",
                stderr=b"",
                exit_code=0,
            ),
        )


def test_root_retains_only_exact_execution_provenance_in_incomplete_index(tmp_path):
    source, _, execution, _, _, payload = _execution_index(tmp_path)
    verified = _verify_index(tmp_path, payload)

    assert source.valid
    assert not verified.valid
    producer = verified.producer_map()["performance_paired_replay"]
    assert tuple(item.admission_id for item in producer.admissions) == (
        "contract-admission-001",
    )
    assert len(verified.artifacts) == 1
    artifact = verified.artifacts[0]
    binding = artifact.contract_performance_execution_provenance
    assert binding is not None
    assert binding.artifact_id == execution.artifact_id
    assert binding.receipt_digest == execution.receipt.digest
    assert binding.output_sha256 == execution.artifact_sha256
    assert artifact.provenance_digest == execution.digest
    assert any(
        error.startswith("typed_producer_receipt_payload_types_missing:")
        for error in verified.verification_errors
    )
    assert any(
        error.startswith("typed_producer_receipt_producers_missing:")
        for error in verified.verification_errors
    )
    assert not any(
        error.startswith("contract_performance_execution_provenance_")
        for error in verified.verification_errors
    )


@pytest.mark.parametrize(
    "case",
    (
        "missing",
        "wrong_admission",
        "duplicate",
        "proof_bytes",
        "artifact_bytes",
        "source_rebind",
        "wrong_type",
        "nonzero_exit",
        "output",
    ),
)
def test_root_execution_provenance_binding_fails_closed(tmp_path, case):
    source, production, _, execution_path, artifact_path, payload = _execution_index(
        tmp_path
    )
    artifact = payload["artifacts"][0]
    assert isinstance(artifact, dict)
    bindings = artifact["contract_performance_execution_provenance"]
    assert isinstance(bindings, list) and isinstance(bindings[0], dict)

    if case == "missing":
        del artifact["contract_performance_execution_provenance"]
    elif case == "wrong_admission":
        bindings[0]["admission_id"] = "other-admission"
    elif case == "duplicate":
        bindings.append(dict(bindings[0]))
    elif case == "proof_bytes":
        execution_path.write_text(
            execution_path.read_text(encoding="utf-8") + "\n", encoding="utf-8"
        )
    elif case == "artifact_bytes":
        artifact_path.write_bytes(artifact_path.read_bytes() + b"\n")
    elif case == "source_rebind":
        rebound_production = build_contract_performance_production_provenance(
            replace(
                production.observation,
                comparison=replace(
                    production.observation.comparison,
                    source_snapshot_digest=_digest("rebound-source"),
                ),
            )
        )
        rebound = build_contract_performance_execution_provenance(rebound_production)
        bindings[0]["provenance_sha256"] = _write_text(
            execution_path,
            canonical_contract_performance_execution_provenance_envelope(rebound),
        )
    elif case == "wrong_type":
        bindings[0]["provenance_type"] = "ParallelScalingProductionProvenance"
    else:
        envelope = json.loads(execution_path.read_text(encoding="utf-8"))
        if case == "nonzero_exit":
            envelope["payload"]["transcript"]["exit_code"] = 1
        else:
            envelope["payload"]["transcript"]["stdout"] = {"$bytes": "AA=="}
        bindings[0]["provenance_sha256"] = _write_text(
            execution_path, canonical_json(envelope)
        )

    verified = _verify_index(tmp_path, payload)
    assert len(verified.artifacts) == 1
    assert verified.artifacts[0].contract_performance_execution_provenance is None
    assert any(
        error.startswith("contract_performance_execution_provenance_")
        or error.startswith("artifact_receipt_hash_mismatch:")
        for error in verified.verification_errors
    )
    assert source.valid
