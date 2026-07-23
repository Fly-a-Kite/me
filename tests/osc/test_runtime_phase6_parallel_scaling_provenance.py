from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

import datadiff_osc
import datadiff_osc._phase6_gate_authority as authority_module
from datadiff_osc._canonical import canonical_envelope, canonical_json, stable_digest
from datadiff_osc.contract_engine.relations import default_relation_registry
from datadiff_osc.runtime._phase6_parallel_scaling_producers import (
    ParallelScalingBenchmarkContext,
    ParallelScalingProductionObservation,
)
from datadiff_osc.runtime._phase6_parallel_scaling_provenance import (
    PARALLEL_SCALING_PRODUCTION_PROVENANCE_ENVELOPE_TYPE,
    PARALLEL_SCALING_PRODUCTION_PROVENANCE_SCHEMA_VERSION,
    ParallelScalingProductionProvenance,
    build_parallel_scaling_production_provenance,
    canonical_parallel_scaling_production_provenance_envelope,
    reconstruct_parallel_scaling_production_provenance_payload,
)
from datadiff_osc.runtime._receipt_producers import (
    ParallelScalingObservation,
    RawCounterMeasurement,
    RawMonotonicInterval,
    TypedExecutionResult,
    WorkerBoundExecutionResult,
)
from datadiff_osc.schemas import (
    ContractFingerprint,
    EvidenceEnvelope,
    EvidenceState,
    EvidenceTier,
    ExecutionStatus,
    FailureKind,
    ResourceTokens,
    ResultGroup,
    SeedLineage,
    SeedStage,
    StructuredExecutionOutcome,
    TaskIdentity,
    TaskKind,
    TaskSpec,
    VerdictKind,
)


def _digest(label: str) -> str:
    return stable_digest("phase6-parallel-provenance-test", label)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _seed() -> SeedLineage:
    return SeedLineage(
        protocol_digest=_digest("protocol"),
        master_seed=79,
        lane_id="phase6-parallel-provenance",
        case_index=23,
        stage_name=SeedStage.ORACLE_SAMPLE,
    )


def _task(*, seed: SeedLineage, decision_index: int, attempt: int) -> TaskSpec:
    return TaskSpec(
        identity=TaskIdentity(
            protocol_digest=seed.protocol_digest,
            task_kind=TaskKind.BACKEND_EXECUTION,
            epoch_index=4,
            decision_index=decision_index,
            seed_lineage_digest=seed.digest,
            contrast_set_id="phase6-parallel-provenance-contrast",
            endpoint_id="endpoint-a",
            backend="backend-a",
            attempt=attempt,
        ),
        dependency_task_ids=(),
        resources=ResourceTokens(
            cpu_tokens=1,
            rss_bytes=0,
            io_class="phase6-provenance-test",
            backend_internal_threads=0,
        ),
        payload_digest=_digest("payload:endpoint-a"),
    )


def _evidence(*, label: str, result: TypedExecutionResult) -> EvidenceEnvelope:
    return EvidenceEnvelope.build(
        evidence_id=f"phase6-parallel-proof-{label}",
        state=EvidenceState.NOT_A_CANDIDATE,
        result_group_digest=result.result_group.digest,
        task_id=result.task.identity.task_id,
        seed_lineage_digest=result.seed_lineage.digest,
        contract_fingerprint=result.result_group.contract_fingerprint,
        target_fingerprint=result.result_group.target_fingerprint,
        derivation_certificate_digest=_digest(f"derivation:{label}"),
        applicability_certificate_digest=_digest(f"applicability:{label}"),
        activation_certificate_digest=_digest(f"activation:{label}"),
        observation_certificate_digest=_digest(f"observation:{label}"),
        execution_outcomes=(result.outcome,),
        verdict_kind=VerdictKind.SATISFIED,
    )


def _observation(*, source_digest: str | None = None) -> ParallelScalingProductionObservation:
    seed = _seed()
    one_task = _task(seed=seed, decision_index=301, attempt=1)
    six_task = _task(seed=seed, decision_index=302, attempt=2)
    group = ResultGroup(
        result_group_id="phase6-parallel-provenance-group",
        task_ids=(one_task.identity.task_id, six_task.identity.task_id),
        endpoint_ids=("endpoint-a",),
        contract_fingerprint=ContractFingerprint(
            contract_digest=_digest("contract"),
            registry_digest=default_relation_registry().digest,
        ),
        target_fingerprint=None,
        evidence_tier=EvidenceTier.SCREENING,
        result_order_key=("phase6-parallel-provenance", "endpoint-a"),
    )

    def result(task: TaskSpec) -> TypedExecutionResult:
        return TypedExecutionResult(
            case_id="phase6-parallel-provenance-case",
            result_group=group,
            task=task,
            seed_lineage=seed,
            outcome=StructuredExecutionOutcome(
                endpoint_id="endpoint-a",
                status=ExecutionStatus.OK,
                failure_kind=FailureKind.NONE,
                reason="parallel provenance execution",
            ),
        )

    one_result = result(one_task)
    six_result = result(six_task)
    one_bindings = (WorkerBoundExecutionResult(1, one_result),)
    six_bindings = (WorkerBoundExecutionResult(6, six_result),)
    return ParallelScalingProductionObservation(
        benchmark_context=ParallelScalingBenchmarkContext(
            source_snapshot_digest=source_digest or _digest("source"),
            benchmark_plan=(
                ("sample_scope", "one-parallel-proof"),
                ("scheduler", "fixed"),
            ),
            environment=(("machine", "test"), ("python", "3.12")),
            one_worker_config=(("isolation", "process"),),
            six_worker_config=(("isolation", "process"),),
        ),
        parallel_sample=ParallelScalingObservation(
            ordinal=1,
            workload_id="phase6-parallel-provenance-workload",
            one_worker_results=one_bindings,
            six_worker_results=six_bindings,
            one_worker_measurement=RawCounterMeasurement(
                "tasks",
                0,
                4,
                RawMonotonicInterval("monotonic-phase6-proof", 10, 4_000_000_010),
            ),
            six_worker_measurement=RawCounterMeasurement(
                "tasks",
                10,
                14,
                RawMonotonicInterval("monotonic-phase6-proof", 20, 1_000_000_020),
            ),
        ),
        one_worker_evidence=(_evidence(label="one", result=one_result),),
        six_worker_evidence=(_evidence(label="six", result=six_result),),
    )


def _write_text(path: Path, text: str) -> str:
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _index_payload(
    *,
    provenance: ParallelScalingProductionProvenance,
    receipt_path: Path,
    receipt_sha256: str,
    provenance_path: Path | None,
    provenance_sha256: str | None,
    provenance_admission_id: str = "parallel-admission-001",
    include_provenance: bool = True,
) -> dict[str, object]:
    source_digest = provenance.receipt.source_snapshot_digest
    producer: dict[str, object] = {
        "schema_version": authority_module.TYPED_PRODUCER_RECEIPT_SCHEMA_VERSION,
        "receipt_id": "parallel-proof-producer",
        "producer_kind": "performance_paired_replay",
        "artifact_id": "parallel-proof-artifact",
        "admissions": [
            {
                "admission_id": "parallel-admission-001",
                "subject_kind": "parallel_scaling_samples",
                "subject_ids": [provenance.receipt.sample_id],
                "envelope_path": receipt_path.name,
                "envelope_sha256": receipt_sha256,
                "envelope_type": "ParallelScalingEvidenceReceipt",
                "envelope_schema_version": provenance.receipt.schema_version,
            }
        ],
    }
    if include_provenance:
        assert provenance_path is not None
        assert provenance_sha256 is not None
        producer["parallel_scaling_provenance"] = [
            {
                "admission_id": provenance_admission_id,
                "provenance_path": provenance_path.name,
                "provenance_sha256": provenance_sha256,
                "provenance_type": (
                    PARALLEL_SCALING_PRODUCTION_PROVENANCE_ENVELOPE_TYPE
                ),
                "provenance_schema_version": (
                    PARALLEL_SCALING_PRODUCTION_PROVENANCE_SCHEMA_VERSION
                ),
            }
        ]
    return {
        "schema_version": authority_module.ARTIFACT_RECEIPT_SCHEMA_VERSION,
        "source_digest": source_digest,
        "dynamic_plan_sha256": _sha256("parallel-proof-dynamic-plan"),
        "producer_receipts": [producer],
        "artifacts": [],
    }


def _verify_index(tmp_path: Path, payload: dict[str, object]):
    index_path = tmp_path / "parallel-proof-index.json"
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    index_sha256 = _write_text(index_path, raw)
    return authority_module.verify_artifact_receipt_index(
        index_path=index_path,
        expected_index_sha256=index_sha256,
        expected_source_digest=str(payload["source_digest"]),
        expected_dynamic_plan_sha256=str(payload["dynamic_plan_sha256"]),
    )


def _provenance_index(tmp_path: Path):
    provenance = build_parallel_scaling_production_provenance(_observation())
    receipt_path = tmp_path / "parallel-proof-receipt.json"
    receipt_sha256 = _write_text(
        receipt_path,
        canonical_envelope(
            "ParallelScalingEvidenceReceipt",
            provenance.receipt.schema_version,
            provenance.receipt,
        ),
    )
    provenance_path = tmp_path / "parallel-production-proof.json"
    provenance_sha256 = _write_text(
        provenance_path,
        canonical_parallel_scaling_production_provenance_envelope(provenance),
    )
    payload = _index_payload(
        provenance=provenance,
        receipt_path=receipt_path,
        receipt_sha256=receipt_sha256,
        provenance_path=provenance_path,
        provenance_sha256=provenance_sha256,
    )
    return provenance, receipt_path, provenance_path, payload


def test_private_parallel_provenance_rebuilds_r6_receipt_without_public_authority():
    observation = _observation()
    provenance = build_parallel_scaling_production_provenance(observation)
    envelope = canonical_parallel_scaling_production_provenance_envelope(provenance)
    decoded = json.loads(envelope)

    assert "ParallelScalingProductionProvenance" not in datadiff_osc.__all__
    assert provenance.receipt.source_snapshot_digest == (
        observation.benchmark_context.source_snapshot_digest
    )
    assert provenance.receipt.sample_id
    assert provenance.provenance_id
    assert reconstruct_parallel_scaling_production_provenance_payload(
        decoded["payload"]
    ) == provenance

    changed = build_parallel_scaling_production_provenance(
        replace(
            observation,
            benchmark_context=replace(
                observation.benchmark_context,
                source_snapshot_digest=_digest("other-source"),
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
            {"six_worker_result_digest": _digest("forged-result")}
        ),
        lambda payload: payload["observation"]["parallel_sample"][
            "one_worker_measurement"
        ].update({"end_count": 5}),
        lambda payload: payload.update({"producer_module_sha256": "0" * 64}),
    ),
)
def test_private_parallel_provenance_rejects_malformed_or_misbound_payload(mutate):
    provenance = build_parallel_scaling_production_provenance(_observation())
    payload = json.loads(canonical_json(provenance))

    mutate(payload)

    with pytest.raises((ValueError, TypeError)):
        reconstruct_parallel_scaling_production_provenance_payload(payload)


def test_private_parallel_provenance_rejects_wrong_object_and_module_binding():
    with pytest.raises(TypeError, match="ParallelScalingProductionObservation"):
        build_parallel_scaling_production_provenance(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="ParallelScalingProductionProvenance"):
        canonical_parallel_scaling_production_provenance_envelope(object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="implementation binding mismatch"):
        replace(
            build_parallel_scaling_production_provenance(_observation()),
            producer_module_sha256="0" * 64,
        )


def test_root_retains_only_exact_parallel_provenance_pair_in_incomplete_index(tmp_path):
    provenance, _, _, payload = _provenance_index(tmp_path)

    verified = _verify_index(tmp_path, payload)

    assert not verified.valid
    producer = verified.producer_map()["performance_paired_replay"]
    assert tuple(item.admission_id for item in producer.admissions) == (
        "parallel-admission-001",
    )
    assert tuple(item.admission_id for item in producer.parallel_scaling_provenance) == (
        "parallel-admission-001",
    )
    assert producer.parallel_scaling_provenance[0].receipt_digest == provenance.receipt.digest
    assert any(
        error.startswith("typed_producer_receipt_payload_types_missing:")
        for error in verified.verification_errors
    )
    assert "artifact_receipt_index_empty" in verified.verification_errors
    assert not any(
        error.startswith("parallel_scaling_provenance_")
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
def test_root_parallel_provenance_pairing_fails_closed_on_binding_variants(
    tmp_path, case
):
    provenance, receipt_path, provenance_path, payload = _provenance_index(tmp_path)
    producer = payload["producer_receipts"][0]
    assert isinstance(producer, dict)

    if case == "missing":
        del producer["parallel_scaling_provenance"]
    elif case == "wrong_admission":
        proof = producer["parallel_scaling_provenance"]
        assert isinstance(proof, list) and isinstance(proof[0], dict)
        proof[0]["admission_id"] = "other-admission"
    elif case == "duplicate":
        proof = producer["parallel_scaling_provenance"]
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
        rebound = build_parallel_scaling_production_provenance(
            _observation(source_digest=_digest("rebound-source"))
        )
        rebound_text = canonical_parallel_scaling_production_provenance_envelope(
            rebound
        )
        rebound_sha256 = _write_text(provenance_path, rebound_text)
        proof = producer["parallel_scaling_provenance"]
        assert isinstance(proof, list) and isinstance(proof[0], dict)
        proof[0]["provenance_sha256"] = rebound_sha256
    else:  # pragma: no cover - protects the parametrized case list.
        raise AssertionError(case)

    verified = _verify_index(tmp_path, payload)

    producer_result = verified.producer_map()["performance_paired_replay"]
    assert producer_result.parallel_scaling_provenance == ()
    assert not any(
        admission.envelope_type == "ParallelScalingEvidenceReceipt"
        for admission in producer_result.admissions
    )
    assert any(
        error.startswith("parallel_scaling_provenance_")
        or error.startswith("typed_admission_")
        for error in verified.verification_errors
    )
    assert not verified.valid
    assert provenance.receipt.sample_id
