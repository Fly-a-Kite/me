from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

import datadiff_osc
import datadiff_osc._phase6_gate_authority as authority_module
from datadiff_osc._canonical import canonical_envelope, canonical_json, decode_canonical_envelope
from datadiff_osc.runtime._phase6_parallel_scaling_receipts import (
    PARALLEL_SCALING_EVIDENCE_RECEIPT_SCHEMA_VERSION,
    ParallelScalingEvidenceReceipt,
)
from datadiff_osc.runtime._semantic_replay import replay_runtime_admission


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _receipt(*, source_digest: str | None = None) -> ParallelScalingEvidenceReceipt:
    task_set = _sha("parallel-task-set")
    return ParallelScalingEvidenceReceipt(
        source_snapshot_digest=source_digest or _sha("parallel-source"),
        workload_id="parallel-workload-001",
        benchmark_plan_digest=_sha("parallel-plan"),
        environment_digest=_sha("parallel-environment"),
        one_worker_config_digest=_sha("parallel-config-one"),
        six_worker_config_digest=_sha("parallel-config-six"),
        one_worker_task_set_digest=task_set,
        six_worker_task_set_digest=task_set,
        one_worker_result_digest=_sha("parallel-result-one"),
        six_worker_result_digest=_sha("parallel-result-six"),
        one_worker_evidence_digest=_sha("parallel-evidence-one"),
        six_worker_evidence_digest=_sha("parallel-evidence-six"),
        one_worker_clock_id="monotonic-clock-1",
        one_worker_start_ns=10,
        one_worker_end_ns=60_000_000_010,
        one_worker_task_counter_start=0,
        one_worker_task_counter_end=60,
        six_worker_clock_id="monotonic-clock-1",
        six_worker_start_ns=20,
        six_worker_end_ns=12_000_000_020,
        six_worker_task_counter_start=5,
        six_worker_task_counter_end=65,
    )


def _payload(receipt: ParallelScalingEvidenceReceipt) -> dict[str, object]:
    value = json.loads(canonical_json(receipt))
    assert isinstance(value, dict)
    return value


def _admission(tmp_path: Path, receipt: ParallelScalingEvidenceReceipt):
    path = tmp_path / "parallel-scaling-envelope.json"
    envelope = canonical_envelope(
        "ParallelScalingEvidenceReceipt",
        receipt.schema_version,
        receipt,
    )
    path.write_text(envelope, encoding="utf-8")
    decoded = decode_canonical_envelope(envelope)
    return authority_module.VerifiedTypedAdmission(
        admission_id="parallel-scaling-admission",
        subject_kind="parallel_scaling_samples",
        subject_ids=(receipt.sample_id,),
        envelope_path=str(path),
        byte_sha256=hashlib.sha256(envelope.encode()).hexdigest(),
        envelope_type="ParallelScalingEvidenceReceipt",
        envelope_schema_version=receipt.schema_version,
        envelope_digest=authority_module.stable_digest(
            "osc-root-typed-admission-envelope", decoded
        ),
    ), path


def test_private_parallel_receipt_derives_identity_and_replays_exactly():
    receipt = _receipt()
    payload = _payload(receipt)

    assert "ParallelScalingEvidenceReceipt" not in datadiff_osc.__all__
    assert receipt.one_worker_count == 1
    assert receipt.six_worker_count == 6
    assert receipt.one_worker_task_count == 60
    assert receipt.six_worker_task_count == 60
    assert receipt.one_worker_elapsed_ns == 60_000_000_000
    assert receipt.six_worker_elapsed_ns == 12_000_000_000
    assert replay_runtime_admission(
        envelope_type="ParallelScalingEvidenceReceipt",
        schema_version=PARALLEL_SCALING_EVIDENCE_RECEIPT_SCHEMA_VERSION,
        payload=payload,
        subject_kind="parallel_scaling_samples",
        subject_ids=(receipt.sample_id,),
    ) == ()
    assert replay_runtime_admission(
        envelope_type="ParallelScalingEvidenceReceipt",
        schema_version=PARALLEL_SCALING_EVIDENCE_RECEIPT_SCHEMA_VERSION,
        payload=payload,
        subject_kind="parallel_scaling_samples",
        subject_ids=(_sha("forged-sample"),),
    ) == (
        "runtime_replay_subject_mismatch:"
        "ParallelScalingEvidenceReceipt:parallel_scaling_samples",
    )


@pytest.mark.parametrize(
    ("changes", "match"),
    (
        ({"one_worker_config_digest": _sha("parallel-config-six")}, "configurations"),
        ({"six_worker_task_set_digest": _sha("other-task-set")}, "task-set"),
        ({"six_worker_result_digest": _sha("parallel-result-one")}, "result identities"),
        ({"six_worker_evidence_digest": _sha("parallel-evidence-one")}, "evidence identities"),
        ({"one_worker_count": 2}, "exactly 1 and 6"),
        ({"six_worker_count": 5}, "exactly 1 and 6"),
        ({"six_worker_clock_id": "other-clock"}, "clocks must match"),
        ({"one_worker_end_ns": 10}, "interval must be positive"),
        ({"six_worker_task_counter_end": 5}, "positive progress"),
        ({"one_worker_task_counter_end": 61}, "same task count"),
        ({"one_worker_start_ns": True}, "non-negative integer"),
        ({"benchmark_plan_digest": "not-a-digest"}, "SHA-256"),
        (
            {"schema_version": "osc-phase6-parallel-scaling-evidence-receipt-v0"},
            "schema",
        ),
    ),
)
def test_private_parallel_receipt_rejects_invalid_identity_and_raw_context(
    changes, match
):
    with pytest.raises(ValueError, match=match):
        replace(_receipt(), **changes)


def test_private_parallel_replay_rejects_payload_extra_missing_and_counter_forgery():
    receipt = _receipt()
    payload = _payload(receipt)

    missing = dict(payload)
    del missing["six_worker_clock_id"]
    extra = dict(payload)
    extra["caller_efficiency"] = 0.99
    forged = dict(payload)
    forged["one_worker_task_counter_end"] = 0

    for value in (missing, extra, forged):
        errors = replay_runtime_admission(
            envelope_type="ParallelScalingEvidenceReceipt",
            schema_version=receipt.schema_version,
            payload=value,
            subject_kind="parallel_scaling_samples",
            subject_ids=(receipt.sample_id,),
        )
        assert len(errors) == 1
        assert errors[0].startswith(
            "runtime_replay_invalid_payload:ParallelScalingEvidenceReceipt:"
        )

    with pytest.raises(TypeError):
        ParallelScalingEvidenceReceipt(
            **_receipt().to_dict(),
            one_worker_elapsed_ns=1,  # type: ignore[call-arg]
        )


def test_legacy_paired_receipt_is_not_owned_for_parallel_scaling_admission():
    assert replay_runtime_admission(
        envelope_type="PairedBenchmarkReceipt",
        schema_version="osc-root-paired-benchmark-receipt-v1",
        payload={},
        subject_kind="parallel_scaling_samples",
        subject_ids=("legacy-parallel-sample",),
    ) == (
        "runtime_replay_type_not_owned:PairedBenchmarkReceipt@"
        "osc-root-paired-benchmark-receipt-v1",
    )


def test_root_binds_parallel_receipt_to_exact_record_and_rejects_substitutions(
    tmp_path,
):
    source_digest = _sha("parallel-source")
    receipt = _receipt(source_digest=source_digest)
    admission, path = _admission(tmp_path, receipt)
    producer = authority_module.VerifiedProducerReceipt(
        receipt_id="parallel-producer",
        producer_kind="performance_paired_replay",
        artifact_id="parallel-artifact",
        admissions=(admission,),
    )
    record = authority_module._parallel_scaling_dynamic_record(
        receipt,
        producer_receipt_digest=producer.digest,
    )

    errors: list[str] = []
    authority_module._verify_parallel_scaling_dynamic_bindings(
        source_digest=source_digest,
        records=(record,),
        performance_producer_receipt=producer,
        errors=errors,
    )
    assert errors == []
    assert (
        "ParallelScalingEvidenceReceipt",
        PARALLEL_SCALING_EVIDENCE_RECEIPT_SCHEMA_VERSION,
    ) in authority_module._ALLOWED_TYPED_ENVELOPES["performance_paired_replay"]
    assert (
        "PairedBenchmarkReceipt",
        "osc-root-paired-benchmark-receipt-v1",
    ) not in authority_module._ALLOWED_TYPED_ENVELOPES["performance_paired_replay"]

    forged = dict(record)
    forged["six_worker_result_digest"] = _sha("forged-six-result")
    errors = []
    authority_module._verify_parallel_scaling_dynamic_bindings(
        source_digest=source_digest,
        records=(forged,),
        performance_producer_receipt=producer,
        errors=errors,
    )
    assert errors == [
        "parallel_scaling_record_binding_mismatch:"
        f"{receipt.sample_id}:six_worker_result_digest"
    ]

    errors = []
    authority_module._verify_parallel_scaling_dynamic_bindings(
        source_digest=_sha("parallel-other-source"),
        records=(record,),
        performance_producer_receipt=producer,
        errors=errors,
    )
    assert errors == [
        "parallel_scaling_receipt_source_snapshot_mismatch:"
        f"{receipt.sample_id}"
    ]

    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    errors = []
    authority_module._verify_parallel_scaling_dynamic_bindings(
        source_digest=source_digest,
        records=(record,),
        performance_producer_receipt=producer,
        errors=errors,
    )
    assert errors
    assert errors[0] == (
        f"parallel_scaling_admission_hash_mismatch:{admission.admission_id}"
    )


def test_root_parallel_binding_rejects_duplicate_and_same_cardinality_admissions(
    tmp_path,
):
    source_digest = _sha("parallel-source")
    receipt = _receipt(source_digest=source_digest)
    admission, _ = _admission(tmp_path, receipt)
    producer = authority_module.VerifiedProducerReceipt(
        receipt_id="parallel-producer",
        producer_kind="performance_paired_replay",
        artifact_id="parallel-artifact",
        admissions=(admission, admission),
    )
    record = authority_module._parallel_scaling_dynamic_record(
        receipt,
        producer_receipt_digest=producer.digest,
    )

    errors: list[str] = []
    authority_module._verify_parallel_scaling_dynamic_bindings(
        source_digest=source_digest,
        records=(record, dict(record)),
        performance_producer_receipt=producer,
        errors=errors,
    )

    assert f"parallel_scaling_duplicate_dynamic_record:{receipt.sample_id}" in errors
    assert f"parallel_scaling_duplicate_sample_admission:{receipt.sample_id}" in errors
