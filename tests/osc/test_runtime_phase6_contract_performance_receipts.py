from __future__ import annotations

from dataclasses import replace
import hashlib
import json

import pytest

import datadiff_osc
from datadiff_osc._canonical import canonical_envelope, canonical_json, decode_canonical_envelope
from datadiff_osc.runtime._phase6_contract_performance_receipts import (
    BASELINE_STRATEGY,
    CONTRACT_PERFORMANCE_EVIDENCE_RECEIPT_SCHEMA_VERSION,
    TREATMENT_STRATEGY,
    ContractPerformanceEvidenceReceipt,
)
from datadiff_osc.runtime._private_receipts import (
    ContractPerformanceSampleBinding,
    PairedBenchmarkReceipt,
    ParallelScalingSampleBinding,
)
from datadiff_osc.runtime._semantic_replay import (
    _reconstruct_contract_performance_evidence_receipt_payload,
    replay_runtime_admission,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _receipt() -> ContractPerformanceEvidenceReceipt:
    return ContractPerformanceEvidenceReceipt(
        source_snapshot_digest=_sha("source-snapshot"),
        source_case_digest=_sha("source-case"),
        comparison_decision_id=_sha("comparison-decision"),
        contract_comparison_producer_receipt_digest=_sha("comparison-producer"),
        result_group_id="result-group-001",
        result_group_digest=_sha("result-group"),
        contract_fingerprint_digest=_sha("contract-fingerprint"),
        endpoint_set_digest=_sha("endpoint-set"),
        fingerprint_task_id="fingerprint-task-001",
        fingerprint_result_digest=_sha("fingerprint-result"),
        exact_task_id="exact-task-001",
        exact_result_digest=_sha("exact-result"),
        fingerprint_partition_digest=_sha("fingerprint-partition"),
        canonical_partition_digest=_sha("canonical-partition"),
        comparison_algorithm_digest=_sha("comparison-algorithm"),
        benchmark_plan_digest=_sha("benchmark-plan"),
        environment_digest=_sha("environment"),
        baseline_config_digest=_sha("baseline-config"),
        treatment_config_digest=_sha("treatment-config"),
        baseline_task_id="baseline-task-001",
        baseline_task_spec_digest=_sha("baseline-task-spec"),
        baseline_result_digest=_sha("baseline-result"),
        baseline_seed_lineage_digest=_sha("shared-seed-lineage"),
        baseline_execution_outcome_digest=_sha("baseline-outcome"),
        baseline_evidence_envelope_digest=_sha("baseline-evidence"),
        treatment_task_id="treatment-task-001",
        treatment_task_spec_digest=_sha("treatment-task-spec"),
        treatment_result_digest=_sha("treatment-result"),
        treatment_seed_lineage_digest=_sha("shared-seed-lineage"),
        treatment_execution_outcome_digest=_sha("treatment-outcome"),
        treatment_evidence_envelope_digest=_sha("treatment-evidence"),
        baseline_materialized_bytes=4096,
        treatment_materialized_bytes=1024,
        baseline_backend_pair_comparisons=6,
        treatment_backend_pair_comparisons=2,
        baseline_comparison_cpu_ns=10_000,
        treatment_comparison_cpu_ns=3_000,
    )


def _payload(receipt: ContractPerformanceEvidenceReceipt) -> dict[str, object]:
    return json.loads(canonical_json(receipt))


def _legacy_paired_receipt() -> PairedBenchmarkReceipt:
    return PairedBenchmarkReceipt(
        source_digest=_sha("legacy-source"),
        benchmark_plan_digest=_sha("legacy-plan"),
        environment_digest=_sha("legacy-environment"),
        baseline_config_digest=_sha("legacy-baseline"),
        treatment_config_digest=_sha("legacy-treatment"),
        contract_samples=(
            ContractPerformanceSampleBinding(
                sample_id="legacy-contract-sample",
                case_id="legacy-case",
                baseline_task_id="legacy-baseline-task",
                treatment_task_id="legacy-treatment-task",
                compile_match_ms=1.0,
                case_wall_ms=5.0,
                baseline_throughput_cases_s=100.0,
                treatment_throughput_cases_s=95.0,
            ),
        ),
        parallel_samples=(
            ParallelScalingSampleBinding(
                sample_id="legacy-parallel-sample",
                workload_id="legacy-workload",
                one_worker_task_set_digest=_sha("legacy-task-set"),
                six_worker_task_set_digest=_sha("legacy-task-set"),
                one_worker_result_digest=_sha("legacy-result-one"),
                six_worker_result_digest=_sha("legacy-result-six"),
                one_worker_elapsed_seconds=60.0,
                six_worker_elapsed_seconds=12.0,
            ),
        ),
    )


def test_private_raw_pair_derives_identity_and_roundtrips_exactly():
    receipt = _receipt()
    payload = _payload(receipt)

    assert "ContractPerformanceEvidenceReceipt" not in datadiff_osc.__all__
    assert receipt.baseline_strategy == BASELINE_STRATEGY
    assert receipt.treatment_strategy == TREATMENT_STRATEGY
    assert receipt.sample_id == _reconstruct_contract_performance_evidence_receipt_payload(
        payload
    ).sample_id
    assert receipt.sample_id != replace(
        receipt, treatment_comparison_cpu_ns=receipt.treatment_comparison_cpu_ns + 1
    ).sample_id
    assert replay_runtime_admission(
        envelope_type="ContractPerformanceEvidenceReceipt",
        schema_version=CONTRACT_PERFORMANCE_EVIDENCE_RECEIPT_SCHEMA_VERSION,
        payload=payload,
        subject_kind="contract_performance_samples",
        subject_ids=(receipt.sample_id,),
    ) == ()


@pytest.mark.parametrize(
    ("changes", "match"),
    (
        ({"baseline_materialized_bytes": 0}, "baseline_materialized_bytes"),
        ({"baseline_backend_pair_comparisons": True}, "integer"),
        ({"treatment_comparison_cpu_ns": -1}, "non-negative"),
        ({"baseline_task_id": "treatment-task-001"}, "task identities"),
        ({"baseline_result_digest": _sha("treatment-result")}, "result identities"),
        ({"baseline_config_digest": _sha("treatment-config")}, "configurations"),
        ({"treatment_seed_lineage_digest": _sha("different-seed")}, "seed lineage"),
        ({"baseline_strategy": "optional_materialization"}, "baseline strategy"),
        ({"treatment_strategy": "fingerprint_only"}, "treatment strategy"),
        ({"schema_version": "osc-phase6-contract-performance-evidence-receipt-v0"}, "schema"),
    ),
)
def test_private_raw_pair_rejects_invalid_context_and_counters(changes, match):
    with pytest.raises(ValueError, match=match):
        replace(_receipt(), **changes)


def test_private_raw_pair_replay_rejects_subject_and_payload_substitutions():
    receipt = _receipt()
    payload = _payload(receipt)

    assert replay_runtime_admission(
        envelope_type="ContractPerformanceEvidenceReceipt",
        schema_version=receipt.schema_version,
        payload=payload,
        subject_kind="contract_performance_samples",
        subject_ids=(_sha("forged-sample"),),
    ) == (
        "runtime_replay_subject_mismatch:"
        "ContractPerformanceEvidenceReceipt:contract_performance_samples",
    )

    payload["baseline_materialized_bytes"] = 0
    errors = replay_runtime_admission(
        envelope_type="ContractPerformanceEvidenceReceipt",
        schema_version=receipt.schema_version,
        payload=payload,
        subject_kind="contract_performance_samples",
        subject_ids=(receipt.sample_id,),
    )
    assert len(errors) == 1
    assert errors[0].startswith(
        "runtime_replay_invalid_payload:ContractPerformanceEvidenceReceipt:"
    )


def test_legacy_paired_receipt_is_not_owned_for_phase6_performance_admission():
    legacy = _legacy_paired_receipt()
    decoded = decode_canonical_envelope(
        canonical_envelope("PairedBenchmarkReceipt", legacy.schema_version, legacy)
    )

    assert replay_runtime_admission(
        envelope_type=decoded["type"],
        schema_version=decoded["schema_version"],
        payload=decoded["payload"],
        subject_kind="contract_performance_samples",
        subject_ids=(legacy.contract_sample_ids[0],),
    ) == (
        "runtime_replay_type_not_owned:PairedBenchmarkReceipt@"
        "osc-root-paired-benchmark-receipt-v1",
    )
