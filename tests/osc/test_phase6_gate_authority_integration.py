from __future__ import annotations

from dataclasses import replace
import hashlib
import importlib.util
import inspect
from pathlib import Path
import shutil
from types import SimpleNamespace
import subprocess
import sys
import textwrap

import pytest

import datadiff_osc._phase6_gate_authority as authority_module
from datadiff_osc._canonical import (
    canonical_envelope,
    canonical_json,
    decode_canonical_envelope,
    stable_digest,
)
from datadiff_osc.contract_engine.model import CONTRACT_SCHEMA_VERSION, Endpoint
from datadiff_osc.contract_engine.relations import default_relation_registry
from datadiff_osc._phase6_gate_authority import (
    ARTIFACT_RECEIPT_SCHEMA_VERSION,
    DYNAMIC_PLAN_SCHEMA_VERSION,
    EXPECTED_COMPILED_UNIVERSE_DIGEST,
    EXPECTED_GATE_SPEC_SET_DIGEST,
    IdentityUniverse,
    RootGateBuildResult,
    RootGatePreflightReport,
    VerifiedArtifactReceipt,
    build_phase6_gate_authority_bundle,
    derive_static_gate_universes,
    preflight_phase6_gate_authority_bundle,
    publish_phase6_gate_authority_bundle,
    source_snapshot_payload,
    verify_artifact_receipt_index,
    verify_dynamic_gate_plan,
    verify_source_snapshot,
)
from datadiff_osc.schemas import (
    ContractFingerprint,
    EvidenceTier,
    ExecutionStatus,
    FailureKind,
    ResultGroup,
    SeedLineage,
    SeedStage,
    StructuredExecutionOutcome,
    TargetFingerprint,
)
from datadiff_osc.search.semantic_replay import replay_search_admission
from datadiff_osc.semantic_targets.compiler import compile_target_universe
from datadiff_osc.semantic_targets.declarations import legacy_v4_target_templates
from datadiff_osc.semantic_targets.model import TargetAssignment
from datadiff_osc.runtime.gate_artifacts import (
    GATE_AUTHORITY_PLAN_SCHEMA_VERSION,
    RAW_GATE_ARTIFACT_SCHEMA_VERSION,
)
from datadiff_osc.runtime.gates import default_pre24_gate_specs
from datadiff_osc.runtime._phase6_contract_performance_receipts import (
    CONTRACT_PERFORMANCE_EVIDENCE_RECEIPT_SCHEMA_VERSION,
    ContractPerformanceEvidenceReceipt,
)
from datadiff_osc.semantic_targets.declarations import legacy_v4_target_templates


ROOT = Path(__file__).resolve().parents[2]


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_json(path: Path, payload: object, *, newline: bool = True) -> str:
    raw = (canonical_json(payload) + ("\n" if newline else "")).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return _sha256(raw)


def _endpoint() -> Endpoint:
    return Endpoint.build(
        endpoint_id="endpoint-root-dispatch",
        case_digest=stable_digest("test-root-dispatch-case", "case-a"),
        backend="backend-a",
        backend_version="1.2.3",
        adapter_revision="adapter-r1",
        execution_mode="eager",
        physical_layout="contiguous",
        capabilities={"execute"},
    )


def _result_group() -> ResultGroup:
    return ResultGroup(
        result_group_id="staged-group-a",
        task_ids=("task-a",),
        endpoint_ids=("endpoint-a",),
        contract_fingerprint=ContractFingerprint(
            contract_digest=stable_digest("test-contract", "contract-a"),
            registry_digest=default_relation_registry().digest,
        ),
        target_fingerprint=TargetFingerprint(
            universe_digest=stable_digest("test-universe", "universe-a"),
            taxonomy_digest=stable_digest("test-taxonomy", "taxonomy-a"),
            template_digest=stable_digest("test-template", "template-a"),
        ),
        evidence_tier=EvidenceTier.AUDIT,
        result_order_key=("task-a", "endpoint-a"),
    )


def _search_assignment():
    universe = compile_target_universe(legacy_v4_target_templates())
    cell = universe.fresh_cells[0]
    assignment = TargetAssignment(
        (cell.target_cell_id,),
        SeedLineage(
            "root-search-replay-protocol",
            23,
            "root-search-replay-lane",
            cell.construction_index,
            SeedStage.CONSTRUCTOR,
        ),
    )
    return assignment, cell


def _verify_single_admission(
    tmp_path: Path,
    *,
    producer_kind: str,
    envelope_type: str,
    schema_version: str,
    payload: object,
    subject_kind: str,
    subject_ids: list[str],
):
    envelope_path = tmp_path / "single-admission-envelope.json"
    envelope_text = canonical_envelope(envelope_type, schema_version, payload)
    envelope_path.write_text(envelope_text, encoding="utf-8")
    errors: list[str] = []
    verified = authority_module._verify_typed_admission(
        root=tmp_path.resolve(),
        producer_kind=producer_kind,
        value={
            "admission_id": "single-admission",
            "subject_kind": subject_kind,
            "subject_ids": subject_ids,
            "envelope_path": envelope_path.name,
            "envelope_sha256": _sha256(envelope_text.encode()),
            "envelope_type": envelope_type,
            "envelope_schema_version": schema_version,
        },
        index=0,
        errors=errors,
    )
    return verified, tuple(errors)


def _source_snapshot(tmp_path: Path, *, repo_root: Path = ROOT):
    git_state = authority_module._current_git_state(repo_root)
    assert git_state is not None
    payload = source_snapshot_payload(
        repo_root=repo_root,
        relative_paths=sorted(authority_module._required_source_paths(repo_root)),
        git_head=git_state[0],
        git_status_digest=git_state[1],
    )
    path = tmp_path / "source-snapshot.json"
    expected_sha = _write_json(path, payload)
    return verify_source_snapshot(
        repo_root=repo_root,
        snapshot_path=path,
        expected_snapshot_sha256=expected_sha,
    )


def _dynamic_payload(source_digest: str) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": DYNAMIC_PLAN_SCHEMA_VERSION,
        "source_digest": source_digest,
        "producer_receipts_digest": stable_digest(
            "osc-root-producer-receipt-set", ()
        ),
    }
    payload.update({section: [] for section in authority_module._DYNAMIC_RECORD_FIELDS})
    return payload


def _incomplete_dynamic_plan(tmp_path: Path, source_digest: str):
    path = tmp_path / "dynamic-plan.json"
    expected_sha = _write_json(path, _dynamic_payload(source_digest))
    return verify_dynamic_gate_plan(
        plan_path=path,
        expected_plan_sha256=expected_sha,
        expected_source_digest=source_digest,
    )


def _comparison_dynamic_record(source_digest: str) -> dict[str, str]:
    return {
        "comparison_decision_id": stable_digest("test-comparison-decision", "a"),
        "source_snapshot_digest": source_digest,
        "source_case_digest": stable_digest("test-comparison-case", "a"),
        "result_group_id": "test-comparison-group-a",
        "result_group_digest": stable_digest("test-result-group", "a"),
        "contract_fingerprint_digest": stable_digest("test-contract-fingerprint", "a"),
        "endpoint_set_digest": stable_digest("test-endpoint-set", "a"),
        "fingerprint_task_id": "test-fingerprint-task-a",
        "fingerprint_result_digest": stable_digest("test-fingerprint-result", "a"),
        "exact_task_id": "test-exact-task-a",
        "exact_result_digest": stable_digest("test-exact-result", "a"),
        "fingerprint_partition_digest": stable_digest(
            "test-fingerprint-partition", "a"
        ),
        "canonical_partition_digest": stable_digest("test-canonical-partition", "a"),
        "producer_receipt_digest": stable_digest("test-producer-receipt", "a"),
    }


def _contract_performance_receipt(
    *,
    source_digest: str,
    comparison_producer_digest: str,
) -> ContractPerformanceEvidenceReceipt:
    return ContractPerformanceEvidenceReceipt(
        source_snapshot_digest=source_digest,
        source_case_digest=stable_digest("test-performance-case", "a"),
        comparison_decision_id=stable_digest("test-performance-decision", "a"),
        contract_comparison_producer_receipt_digest=comparison_producer_digest,
        result_group_id="test-performance-group-a",
        result_group_digest=stable_digest("test-performance-group", "a"),
        contract_fingerprint_digest=stable_digest("test-performance-contract", "a"),
        endpoint_set_digest=stable_digest("test-performance-endpoints", "a"),
        fingerprint_task_id="test-performance-fingerprint-task-a",
        fingerprint_result_digest=stable_digest(
            "test-performance-fingerprint-result", "a"
        ),
        exact_task_id="test-performance-exact-task-a",
        exact_result_digest=stable_digest("test-performance-exact-result", "a"),
        fingerprint_partition_digest=stable_digest(
            "test-performance-fingerprint-partition", "a"
        ),
        canonical_partition_digest=stable_digest(
            "test-performance-canonical-partition", "a"
        ),
        comparison_algorithm_digest=stable_digest(
            "test-performance-algorithm", "a"
        ),
        benchmark_plan_digest=stable_digest("test-performance-plan", "a"),
        environment_digest=stable_digest("test-performance-environment", "a"),
        baseline_config_digest=stable_digest("test-performance-baseline", "a"),
        treatment_config_digest=stable_digest("test-performance-treatment", "a"),
        baseline_task_id="test-performance-baseline-task-a",
        baseline_task_spec_digest=stable_digest(
            "test-performance-baseline-task-spec", "a"
        ),
        baseline_result_digest=stable_digest("test-performance-baseline-result", "a"),
        baseline_seed_lineage_digest=stable_digest(
            "test-performance-shared-seed", "a"
        ),
        baseline_execution_outcome_digest=stable_digest(
            "test-performance-baseline-outcome", "a"
        ),
        baseline_evidence_envelope_digest=stable_digest(
            "test-performance-baseline-evidence", "a"
        ),
        treatment_task_id="test-performance-treatment-task-a",
        treatment_task_spec_digest=stable_digest(
            "test-performance-treatment-task-spec", "a"
        ),
        treatment_result_digest=stable_digest(
            "test-performance-treatment-result", "a"
        ),
        treatment_seed_lineage_digest=stable_digest(
            "test-performance-shared-seed", "a"
        ),
        treatment_execution_outcome_digest=stable_digest(
            "test-performance-treatment-outcome", "a"
        ),
        treatment_evidence_envelope_digest=stable_digest(
            "test-performance-treatment-evidence", "a"
        ),
        baseline_materialized_bytes=4096,
        treatment_materialized_bytes=1024,
        baseline_backend_pair_comparisons=6,
        treatment_backend_pair_comparisons=2,
        baseline_comparison_cpu_ns=10_000,
        treatment_comparison_cpu_ns=3_000,
    )


def _comparison_record_for_performance(
    receipt: ContractPerformanceEvidenceReceipt,
    *,
    producer_receipt_digest: str,
) -> dict[str, str]:
    return {
        "comparison_decision_id": receipt.comparison_decision_id,
        "source_snapshot_digest": receipt.source_snapshot_digest,
        "source_case_digest": receipt.source_case_digest,
        "result_group_id": receipt.result_group_id,
        "result_group_digest": receipt.result_group_digest,
        "contract_fingerprint_digest": receipt.contract_fingerprint_digest,
        "endpoint_set_digest": receipt.endpoint_set_digest,
        "fingerprint_task_id": receipt.fingerprint_task_id,
        "fingerprint_result_digest": receipt.fingerprint_result_digest,
        "exact_task_id": receipt.exact_task_id,
        "exact_result_digest": receipt.exact_result_digest,
        "fingerprint_partition_digest": receipt.fingerprint_partition_digest,
        "canonical_partition_digest": receipt.canonical_partition_digest,
        "producer_receipt_digest": producer_receipt_digest,
    }


def _performance_admission(
    tmp_path: Path,
    receipt: ContractPerformanceEvidenceReceipt,
):
    path = tmp_path / "contract-performance-envelope.json"
    envelope = canonical_envelope(
        "ContractPerformanceEvidenceReceipt",
        CONTRACT_PERFORMANCE_EVIDENCE_RECEIPT_SCHEMA_VERSION,
        receipt,
    )
    path.write_text(envelope, encoding="utf-8")
    decoded = decode_canonical_envelope(envelope)
    return authority_module.VerifiedTypedAdmission(
        admission_id="contract-performance-admission",
        subject_kind="contract_performance_samples",
        subject_ids=(receipt.sample_id,),
        envelope_path=str(path),
        byte_sha256=_sha256(envelope.encode()),
        envelope_type="ContractPerformanceEvidenceReceipt",
        envelope_schema_version=CONTRACT_PERFORMANCE_EVIDENCE_RECEIPT_SCHEMA_VERSION,
        envelope_digest=stable_digest("osc-root-typed-admission-envelope", decoded),
    ), path


@pytest.mark.parametrize(
    "legacy_schema",
    ("osc-root-phase6-dynamic-plan-v1", "osc-root-phase6-dynamic-plan-v2"),
)
def test_old_dynamic_plan_schema_is_never_migrated_or_accepted(
    tmp_path, legacy_schema
):
    source = _source_snapshot(tmp_path)
    payload = _dynamic_payload(source.source_digest)
    payload["schema_version"] = legacy_schema
    path = tmp_path / "old-dynamic-plan.json"
    expected_sha = _write_json(path, payload)

    verified = verify_dynamic_gate_plan(
        plan_path=path,
        expected_plan_sha256=expected_sha,
        expected_source_digest=source.source_digest,
    )

    assert verified.valid is False
    assert "dynamic_plan_schema_version_mismatch" in verified.verification_errors


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    (
        ("missing", "dynamic_plan_record_schema:contract_comparison_decisions:0"),
        ("extra", "dynamic_plan_record_schema:contract_comparison_decisions:0"),
        ("duplicate", "dynamic_plan_duplicate_identity:contract_comparison_decisions"),
        (
            "unordered",
            "dynamic_plan_record_identities_not_sorted:"
            "contract_comparison_decisions",
        ),
    ),
)
def test_contract_comparison_dynamic_records_reject_shape_and_identity_forgeries(
    tmp_path,
    mutation,
    expected_error,
):
    source = _source_snapshot(tmp_path)
    payload = _dynamic_payload(source.source_digest)
    record = _comparison_dynamic_record(source.source_digest)
    if mutation == "missing":
        del record["canonical_partition_digest"]
        payload["contract_comparison_decisions"] = [record]
    elif mutation == "extra":
        record["caller_claimed_equal"] = "true"
        payload["contract_comparison_decisions"] = [record]
    elif mutation == "duplicate":
        payload["contract_comparison_decisions"] = [record, dict(record)]
    else:
        other = dict(record)
        other["comparison_decision_id"] = stable_digest(
            "test-comparison-decision", "b"
        )
        payload["contract_comparison_decisions"] = sorted(
            (record, other),
            key=lambda item: item["comparison_decision_id"],
            reverse=True,
        )
    path = tmp_path / f"comparison-{mutation}.json"
    expected_sha = _write_json(path, payload)

    verified = verify_dynamic_gate_plan(
        plan_path=path,
        expected_plan_sha256=expected_sha,
        expected_source_digest=source.source_digest,
    )

    assert expected_error in verified.verification_errors
    assert "contract_comparison_decision_floor_not_met" in verified.verification_errors


@pytest.mark.parametrize(
    ("field", "value", "expected_error"),
    (
        (
            "baseline_materialized_bytes",
            0,
            "dynamic_plan_record_positive_integer:"
            "contract_performance_samples:0:baseline_materialized_bytes",
        ),
        (
            "treatment_comparison_cpu_ns",
            -1,
            "dynamic_plan_record_nonnegative_integer:"
            "contract_performance_samples:0:treatment_comparison_cpu_ns",
        ),
        (
            "baseline_strategy",
            "caller_chosen_strategy",
            "contract_performance_strategy_mismatch",
        ),
    ),
)
def test_contract_performance_dynamic_records_reject_raw_pair_forgeries(
    tmp_path, field, value, expected_error
):
    source = _source_snapshot(tmp_path)
    comparison_digest = stable_digest("test-comparison-producer", "a")
    receipt = _contract_performance_receipt(
        source_digest=source.source_digest,
        comparison_producer_digest=comparison_digest,
    )
    payload = _dynamic_payload(source.source_digest)
    record = authority_module._contract_performance_dynamic_record(
        receipt,
        producer_receipt_digest=stable_digest("test-performance-producer", "a"),
    )
    record[field] = value
    payload["contract_performance_samples"] = [record]
    path = tmp_path / f"contract-performance-{field}.json"
    expected_sha = _write_json(path, payload)

    verified = verify_dynamic_gate_plan(
        plan_path=path,
        expected_plan_sha256=expected_sha,
        expected_source_digest=source.source_digest,
    )

    assert expected_error in verified.verification_errors
    assert "contract_performance_sample_floor_not_met" in verified.verification_errors


def test_root_binds_raw_pair_to_exact_runtime_and_r2_comparison_context(tmp_path):
    source_digest = stable_digest("test-performance-source", "a")
    comparison_producer = authority_module.VerifiedProducerReceipt(
        receipt_id="comparison-producer",
        producer_kind="contract_exact_replay",
        artifact_id="comparison-artifact",
        admissions=(),
    )
    receipt = _contract_performance_receipt(
        source_digest=source_digest,
        comparison_producer_digest=comparison_producer.digest,
    )
    admission, path = _performance_admission(tmp_path, receipt)
    performance_producer = authority_module.VerifiedProducerReceipt(
        receipt_id="performance-producer",
        producer_kind="performance_paired_replay",
        artifact_id="performance-artifact",
        admissions=(admission,),
    )
    record = authority_module._contract_performance_dynamic_record(
        receipt,
        producer_receipt_digest=performance_producer.digest,
    )
    comparison_record = _comparison_record_for_performance(
        receipt,
        producer_receipt_digest=comparison_producer.digest,
    )

    errors: list[str] = []
    authority_module._verify_contract_performance_dynamic_bindings(
        source_digest=source_digest,
        records=(record,),
        comparison_records=(comparison_record,),
        comparison_producer_receipt=comparison_producer,
        performance_producer_receipt=performance_producer,
        errors=errors,
    )

    assert errors == []
    assert (
        "ContractPerformanceEvidenceReceipt",
        CONTRACT_PERFORMANCE_EVIDENCE_RECEIPT_SCHEMA_VERSION,
    ) in authority_module._ALLOWED_TYPED_ENVELOPES["performance_paired_replay"]
    assert (
        "PairedBenchmarkReceipt",
        "osc-root-paired-benchmark-receipt-v1",
    ) not in authority_module._ALLOWED_TYPED_ENVELOPES["performance_paired_replay"]

    forged_record = dict(record)
    forged_record["treatment_comparison_cpu_ns"] += 1
    errors = []
    authority_module._verify_contract_performance_dynamic_bindings(
        source_digest=source_digest,
        records=(forged_record,),
        comparison_records=(comparison_record,),
        comparison_producer_receipt=comparison_producer,
        performance_producer_receipt=performance_producer,
        errors=errors,
    )
    assert errors == [
        "contract_performance_record_binding_mismatch:"
        f"{receipt.sample_id}:treatment_comparison_cpu_ns"
    ]

    forged_comparison = dict(comparison_record)
    forged_comparison["exact_result_digest"] = stable_digest(
        "test-forged-exact-result", "a"
    )
    errors = []
    authority_module._verify_contract_performance_dynamic_bindings(
        source_digest=source_digest,
        records=(record,),
        comparison_records=(forged_comparison,),
        comparison_producer_receipt=comparison_producer,
        performance_producer_receipt=performance_producer,
        errors=errors,
    )
    assert errors == [
        "contract_performance_comparison_binding_mismatch:"
        f"{receipt.sample_id}:exact_result_digest"
    ]

    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    errors = []
    authority_module._verify_contract_performance_dynamic_bindings(
        source_digest=source_digest,
        records=(record,),
        comparison_records=(comparison_record,),
        comparison_producer_receipt=comparison_producer,
        performance_producer_receipt=performance_producer,
        errors=errors,
    )
    assert any(
        error == f"contract_performance_admission_hash_mismatch:{admission.admission_id}"
        for error in errors
    )


def _empty_receipts(tmp_path: Path, source_digest: str, plan_sha256: str):
    path = tmp_path / "artifact-receipts.json"
    expected_sha = _write_json(
        path,
        {
            "schema_version": ARTIFACT_RECEIPT_SCHEMA_VERSION,
            "source_digest": source_digest,
            "dynamic_plan_sha256": plan_sha256,
            "producer_receipts": [],
            "artifacts": [],
        },
    )
    return verify_artifact_receipt_index(
        index_path=path,
        expected_index_sha256=expected_sha,
        expected_source_digest=source_digest,
        expected_dynamic_plan_sha256=plan_sha256,
    )


def _incomplete_bundle(tmp_path: Path):
    source = _source_snapshot(tmp_path)
    dynamic = _incomplete_dynamic_plan(tmp_path, source.source_digest)
    receipts = _empty_receipts(tmp_path, source.source_digest, dynamic.byte_sha256)
    result = build_phase6_gate_authority_bundle(
        source=source,
        templates=legacy_v4_target_templates(),
        dynamic_plan=dynamic,
        artifact_receipts=receipts,
    )
    return source, dynamic, receipts, result


def test_static_universes_recompute_frozen_41_38_and_exact_counts(tmp_path):
    source = _source_snapshot(tmp_path)
    static = derive_static_gate_universes(
        source=source,
        templates=legacy_v4_target_templates(),
    )
    universes = static.universe_map()

    assert source.valid
    assert static.valid
    assert static.compiled_universe.digest == EXPECTED_COMPILED_UNIVERSE_DIGEST
    assert len(default_pre24_gate_specs()) == 41
    assert len({item.denominator_key for item in default_pre24_gate_specs()}) == 38
    assert len(universes) == 21
    assert len(universes["fresh_families_declared"].record_ids) == 16
    assert len(universes["fresh_cells_declared"].record_ids) == 232
    assert len(universes["regression_families_declared"].record_ids) == 9
    assert len(universes["regression_cells_declared"].record_ids) == 144
    assert len(universes["contrast_edges_declared"].record_ids) == 384
    assert len(universes["backend_pair_obligations_declared"].record_ids) == 502
    assert len(universes["confirmed_roots_total"].record_ids) == 9
    assert len(universes["high_risk_mutants_total"].record_ids) == 32
    assert len(universes["parallel_invariance_checks_total"].record_ids) == 13


def test_incomplete_inputs_build_all_38_universes_but_never_publish(tmp_path):
    _, _, receipts, result = _incomplete_bundle(tmp_path)

    assert len(result.denominator_universes) == 38
    assert result.publishable is False
    assert result.twenty_four_hour_run_authorized is False
    assert "trusted_dynamic_plan_invalid" in result.errors
    assert "trusted_artifact_receipts_invalid" in result.errors
    assert "typed_admission_semantic_replay_pending_phase6" not in receipts.verification_errors
    assert "typed_producer_receipts_missing" in receipts.verification_errors
    assert "root_raw_metrics_missing:" in "\n".join(result.errors)


def test_every_allowed_envelope_pair_has_exactly_one_domain_replay_owner():
    allowed = {
        pair
        for pairs in authority_module._ALLOWED_TYPED_ENVELOPES.values()
        for pair in pairs
    }
    owned = authority_module._TYPED_SEMANTIC_REPLAYERS

    assert set(owned) == allowed
    assert len(owned) == 27
    assert {
        owner: sum(1 for value in owned.values() if value[0] == owner)
        for owner in ("contract", "search", "runtime")
    } == {"contract": 7, "search": 7, "runtime": 13}
    assert all(
        owned[pair] == ("search", replay_search_admission)
        for pair in authority_module._SEARCH_REPLAY_TYPES
    )


def test_root_rebuilds_valid_contract_payload_and_retains_intrinsic_subject(tmp_path):
    endpoint = _endpoint()
    verified, errors = _verify_single_admission(
        tmp_path,
        producer_kind="contract_exact_replay",
        envelope_type="Endpoint",
        schema_version=CONTRACT_SCHEMA_VERSION,
        payload=endpoint,
        subject_kind="endpoint_digests",
        subject_ids=[endpoint.digest],
    )

    assert errors == ()
    assert verified is not None
    assert verified.subject_ids == (endpoint.digest,)


def test_root_discards_payload_identity_forgery_after_real_object_replay(tmp_path):
    endpoint = _endpoint()
    forged_payload = endpoint.to_dict()
    forged_payload["backend"] = "substituted-backend"
    verified, errors = _verify_single_admission(
        tmp_path,
        producer_kind="contract_exact_replay",
        envelope_type="Endpoint",
        schema_version=CONTRACT_SCHEMA_VERSION,
        payload=forged_payload,
        subject_kind="endpoint_digests",
        subject_ids=[endpoint.digest],
    )

    assert verified is None
    assert any(
        error.startswith(
            "typed_admission_semantic_replay_failed:single-admission:"
        )
        and "subject_ids_mismatch" in error
        for error in errors
    )


def test_root_discards_caller_subject_substitution_for_valid_payload(tmp_path):
    endpoint = _endpoint()
    verified, errors = _verify_single_admission(
        tmp_path,
        producer_kind="contract_exact_replay",
        envelope_type="Endpoint",
        schema_version=CONTRACT_SCHEMA_VERSION,
        payload=endpoint,
        subject_kind="endpoint_digests",
        subject_ids=[stable_digest("substituted-endpoint", "not-the-payload")],
    )

    assert verified is None
    assert any(
        error.startswith(
            "typed_admission_semantic_replay_failed:single-admission:"
        )
        and "subject_ids_mismatch" in error
        for error in errors
    )


def test_root_routes_search_intrinsic_subject_to_the_real_search_replayer(tmp_path):
    assignment, cell = _search_assignment()
    verified, errors = _verify_single_admission(
        tmp_path,
        producer_kind="reachability_admission_replay",
        envelope_type="TargetAssignment",
        schema_version="osc-root-target-assignment-v1",
        payload=assignment,
        subject_kind="fresh_cells_declared",
        subject_ids=[cell.target_cell_id],
    )

    assert errors == ()
    assert verified is not None
    assert verified.subject_ids == (cell.target_cell_id,)


def test_root_discards_search_subject_substitution(tmp_path):
    assignment, _ = _search_assignment()
    universe = compile_target_universe(legacy_v4_target_templates())
    substituted_cell = universe.fresh_cells[1]
    verified, errors = _verify_single_admission(
        tmp_path,
        producer_kind="reachability_admission_replay",
        envelope_type="TargetAssignment",
        schema_version="osc-root-target-assignment-v1",
        payload=assignment,
        subject_kind="fresh_cells_declared",
        subject_ids=[substituted_cell.target_cell_id],
    )

    assert verified is None
    assert any(
        error.startswith(
            "typed_admission_semantic_replay_failed:single-admission:"
            "search_replay_subject_mismatch:fresh_cells_declared:"
        )
        for error in errors
    )


def test_search_context_failure_is_not_retained_in_producer_receipt(tmp_path):
    source = _source_snapshot(tmp_path)
    dynamic = _incomplete_dynamic_plan(tmp_path, source.source_digest)
    assignment, cell = _search_assignment()
    envelope_path = tmp_path / "search-assignment-envelope.json"
    envelope_text = canonical_envelope(
        "TargetAssignment", "osc-root-target-assignment-v1", assignment
    )
    envelope_path.write_text(envelope_text, encoding="utf-8")
    envelope_sha = _sha256(envelope_text.encode())

    def admission(admission_id, subject_kind, subject_ids):
        return {
            "admission_id": admission_id,
            "subject_kind": subject_kind,
            "subject_ids": subject_ids,
            "envelope_path": envelope_path.name,
            "envelope_sha256": envelope_sha,
            "envelope_type": "TargetAssignment",
            "envelope_schema_version": "osc-root-target-assignment-v1",
        }

    receipt_path = tmp_path / "search-context-receipts.json"
    receipt_sha = _write_json(
        receipt_path,
        {
            "schema_version": ARTIFACT_RECEIPT_SCHEMA_VERSION,
            "source_digest": source.source_digest,
            "dynamic_plan_sha256": dynamic.byte_sha256,
            "producer_receipts": [
                {
                    "schema_version": (
                        authority_module.TYPED_PRODUCER_RECEIPT_SCHEMA_VERSION
                    ),
                    "receipt_id": "search-receipt",
                    "producer_kind": "reachability_admission_replay",
                    "artifact_id": "search-artifact",
                    "admissions": [
                        admission(
                            "search-context",
                            "scheduled_target_attempts",
                            ["caller-authored-attempt"],
                        ),
                        admission(
                            "search-valid",
                            "fresh_cells_declared",
                            [cell.target_cell_id],
                        ),
                    ],
                }
            ],
            "artifacts": [],
        },
    )
    receipts = verify_artifact_receipt_index(
        index_path=receipt_path,
        expected_index_sha256=receipt_sha,
        expected_source_digest=source.source_digest,
        expected_dynamic_plan_sha256=dynamic.byte_sha256,
    )

    retained = receipts.producer_map()["reachability_admission_replay"].admissions
    assert tuple(item.admission_id for item in retained) == ("search-valid",)
    assert any(
        error.startswith(
            "typed_admission_semantic_replay_failed:search-context:"
            "search_replay_context_required:scheduled_target_attempts"
        )
        for error in receipts.verification_errors
    )
    assert receipts.valid is False


def test_root_discards_payload_that_violates_reconstructed_type_invariant(tmp_path):
    endpoint = _endpoint()
    forged_payload = endpoint.to_dict()
    forged_payload["capabilities"] = [""]
    verified, errors = _verify_single_admission(
        tmp_path,
        producer_kind="contract_exact_replay",
        envelope_type="Endpoint",
        schema_version=CONTRACT_SCHEMA_VERSION,
        payload=forged_payload,
        subject_kind="endpoint_digests",
        subject_ids=[endpoint.digest],
    )

    assert verified is None
    assert any(
        error.startswith(
            "typed_admission_semantic_replay_failed:single-admission:"
        )
        and "payload_invalid" in error
        for error in errors
    )


def test_runtime_typed_replay_success_still_waits_for_root_provenance(tmp_path):
    outcome = StructuredExecutionOutcome(
        endpoint_id="endpoint-a",
        status=ExecutionStatus.OK,
        failure_kind=FailureKind.NONE,
    )
    verified, errors = _verify_single_admission(
        tmp_path,
        producer_kind="v3_typed_run_replay",
        envelope_type="StructuredExecutionOutcome",
        schema_version="osc-structured-execution-outcome-v1",
        payload=outcome,
        subject_kind="execution_outcomes",
        subject_ids=[outcome.digest],
    )

    assert verified is None
    assert errors == (
        "typed_admission_root_provenance_pending_phase6:single-admission",
    )


def test_staged_parity_group_waits_for_root_cross_object_binding(tmp_path):
    group = _result_group()
    verified, errors = _verify_single_admission(
        tmp_path,
        producer_kind="contract_exact_replay",
        envelope_type="ResultGroup",
        schema_version="osc-result-group-v1",
        payload=group,
        subject_kind="staged_parity_groups",
        subject_ids=[group.result_group_id],
    )

    assert verified is None
    assert errors == (
        "typed_admission_root_context_pending_phase6:single-admission:"
        "staged_parity_groups:CR-OSC-6-002",
    )


@pytest.mark.parametrize(
    ("forgery", "expected_error"),
    (
        ("hash", "typed_admission_hash_mismatch:transport-forgery"),
        ("type", "typed_admission_type_mismatch:transport-forgery"),
        ("version", "typed_admission_version_mismatch:transport-forgery"),
        ("invalid", "typed_admission_invalid_envelope:transport-forgery:"),
    ),
)
def test_transport_forgery_is_discarded_before_domain_replay(
    tmp_path, monkeypatch, forgery, expected_error
):
    endpoint = _endpoint()
    if forgery == "type":
        raw = canonical_envelope(
            "HyperContract", CONTRACT_SCHEMA_VERSION, endpoint
        ).encode()
    elif forgery == "version":
        raw = canonical_envelope(
            "Endpoint", "osc-result-group-v1", endpoint
        ).encode()
    elif forgery == "invalid":
        raw = b'{"not":"a canonical envelope"}'
    else:
        raw = canonical_envelope(
            "Endpoint", CONTRACT_SCHEMA_VERSION, endpoint
        ).encode()
    path = tmp_path / "transport-forgery.json"
    path.write_bytes(raw)
    calls = 0

    def forbidden_replay(**_kwargs):
        nonlocal calls
        calls += 1
        return ()

    monkeypatch.setitem(
        authority_module._TYPED_SEMANTIC_REPLAYERS,
        ("Endpoint", CONTRACT_SCHEMA_VERSION),
        ("contract", forbidden_replay),
    )
    errors: list[str] = []
    verified = authority_module._verify_typed_admission(
        root=tmp_path.resolve(),
        producer_kind="contract_exact_replay",
        value={
            "admission_id": "transport-forgery",
            "subject_kind": "endpoint_digests",
            "subject_ids": [endpoint.digest],
            "envelope_path": path.name,
            "envelope_sha256": "0" * 64 if forgery == "hash" else _sha256(raw),
            "envelope_type": "Endpoint",
            "envelope_schema_version": CONTRACT_SCHEMA_VERSION,
        },
        index=0,
        errors=errors,
    )

    assert verified is None
    assert calls == 0
    if forgery == "invalid":
        assert any(error.startswith(expected_error) for error in errors)
    else:
        assert expected_error in errors


def test_missing_or_wrong_typed_inputs_return_no_go_instead_of_throwing():
    result = build_phase6_gate_authority_bundle(
        source=None,
        templates=legacy_v4_target_templates(),
        dynamic_plan=None,
        artifact_receipts=None,
    )

    assert result.publishable is False
    assert len(result.denominator_universes) == 38
    assert result.twenty_four_hour_run_authorized is False
    assert "trusted_source_snapshot_invalid" in result.errors
    assert "trusted_dynamic_plan_invalid" in result.errors
    assert "trusted_artifact_receipts_invalid" in result.errors


def test_dynamic_plan_requires_performance_pairs_and_real_package_identities(tmp_path):
    source = _source_snapshot(tmp_path)
    payload = _dynamic_payload(source.source_digest)
    digest = stable_digest("test-producer-receipt", "packages")
    payload["target_packages"] = [
        {
            "package_id": f"fake-package-{index}",
            "distribution_name": f"fake-dist-{index}",
            "import_name": f"fake-import-{index}",
            "expected_version": "1.0.0",
            "version_source_digest": stable_digest("test-version-source", index),
            "producer_receipt_digest": digest,
        }
        for index in range(7)
    ]
    path = tmp_path / "fake-packages-plan.json"
    expected_sha = _write_json(path, payload)
    verified = verify_dynamic_gate_plan(
        plan_path=path,
        expected_plan_sha256=expected_sha,
        expected_source_digest=source.source_digest,
    )

    assert "target_package_identity_set_mismatch" in verified.verification_errors
    assert "contract_performance_sample_floor_not_met" in verified.verification_errors
    assert "parallel_scaling_sample_plan_missing" in verified.verification_errors


def test_plan_and_receipt_bundle_are_cryptographically_cross_bound(tmp_path):
    source = _source_snapshot(tmp_path)
    dynamic = _incomplete_dynamic_plan(tmp_path, source.source_digest)
    receipt_path = tmp_path / "wrong-plan-receipts.json"
    receipt_sha = _write_json(
        receipt_path,
        {
            "schema_version": ARTIFACT_RECEIPT_SCHEMA_VERSION,
            "source_digest": source.source_digest,
            "dynamic_plan_sha256": "0" * 64,
            "producer_receipts": [],
            "artifacts": [],
        },
    )
    receipts = verify_artifact_receipt_index(
        index_path=receipt_path,
        expected_index_sha256=receipt_sha,
        expected_source_digest=source.source_digest,
        expected_dynamic_plan_sha256=dynamic.byte_sha256,
    )

    assert "artifact_receipt_index_dynamic_plan_mismatch" in receipts.verification_errors
    assert receipts.valid is False


def test_aggregate_ledger_envelope_cannot_masquerade_as_typed_admission(tmp_path):
    source = _source_snapshot(tmp_path)
    dynamic = _incomplete_dynamic_plan(tmp_path, source.source_digest)
    envelope_path = tmp_path / "ledger-envelope.json"
    envelope_text = canonical_envelope(
        "LedgerEvent",
        "osc-ledger-event-v1",
        {"counts": {"cells": 232}, "credited_ids": ["fake"]},
    )
    envelope_path.write_text(envelope_text, encoding="utf-8")
    envelope_sha = _sha256(envelope_text.encode())
    receipt_path = tmp_path / "ledger-receipts.json"
    receipt_sha = _write_json(
        receipt_path,
        {
            "schema_version": ARTIFACT_RECEIPT_SCHEMA_VERSION,
            "source_digest": source.source_digest,
            "dynamic_plan_sha256": dynamic.byte_sha256,
            "producer_receipts": [
                {
                    "schema_version": authority_module.TYPED_PRODUCER_RECEIPT_SCHEMA_VERSION,
                    "receipt_id": "coverage-receipt",
                    "producer_kind": "coverage_admission_replay",
                    "artifact_id": "coverage-artifact",
                    "admissions": [
                        {
                            "admission_id": "aggregate-ledger",
                            "subject_kind": "fresh_cells_declared",
                            "subject_ids": ["fake"],
                            "envelope_path": envelope_path.name,
                            "envelope_sha256": envelope_sha,
                            "envelope_type": "LedgerEvent",
                            "envelope_schema_version": "osc-ledger-event-v1",
                        }
                    ],
                }
            ],
            "artifacts": [],
        },
    )
    receipts = verify_artifact_receipt_index(
        index_path=receipt_path,
        expected_index_sha256=receipt_sha,
        expected_source_digest=source.source_digest,
        expected_dynamic_plan_sha256=dynamic.byte_sha256,
    )

    assert any(
        error.startswith("typed_admission_type_not_allowed:coverage_admission_replay")
        for error in receipts.verification_errors
    )
    assert receipts.valid is False


def test_synthetic_typed_payload_is_rejected_even_with_external_hash(tmp_path):
    source = _source_snapshot(tmp_path)
    dynamic = _incomplete_dynamic_plan(tmp_path, source.source_digest)
    envelope_path = tmp_path / "synthetic-envelope.json"
    envelope_text = canonical_envelope(
        "RepositoryTestReceipt",
        "osc-root-repository-test-receipt-v1",
        {"mode": "synthetic", "node_ids": ["test_fake.py::test_fake"]},
    )
    envelope_path.write_text(envelope_text, encoding="utf-8")
    receipt_path = tmp_path / "synthetic-receipts.json"
    receipt_sha = _write_json(
        receipt_path,
        {
            "schema_version": ARTIFACT_RECEIPT_SCHEMA_VERSION,
            "source_digest": source.source_digest,
            "dynamic_plan_sha256": dynamic.byte_sha256,
            "producer_receipts": [
                {
                    "schema_version": authority_module.TYPED_PRODUCER_RECEIPT_SCHEMA_VERSION,
                    "receipt_id": "repository-receipt",
                    "producer_kind": "repository_test_replay",
                    "artifact_id": "repository-artifact",
                    "admissions": [
                        {
                            "admission_id": "repository-tests",
                            "subject_kind": "repository_tests",
                            "subject_ids": ["test_fake.py::test_fake"],
                            "envelope_path": envelope_path.name,
                            "envelope_sha256": _sha256(envelope_text.encode()),
                            "envelope_type": "RepositoryTestReceipt",
                            "envelope_schema_version": "osc-root-repository-test-receipt-v1",
                        }
                    ],
                }
            ],
            "artifacts": [],
        },
    )
    receipts = verify_artifact_receipt_index(
        index_path=receipt_path,
        expected_index_sha256=receipt_sha,
        expected_source_digest=source.source_digest,
        expected_dynamic_plan_sha256=dynamic.byte_sha256,
    )

    assert "typed_admission_synthetic_rejected:repository-tests" in receipts.verification_errors
    assert receipts.producer_map()["repository_test_replay"].admissions == ()


def test_same_cardinality_identity_substitution_changes_authority_digest():
    provenance = stable_digest("test-provenance", "same")
    real = IdentityUniverse("metric", ("a", "b"), provenance)
    substituted = IdentityUniverse("metric", ("x", "y"), provenance)

    assert len(real.record_ids) == len(substituted.record_ids) == 2
    assert real.identity_digest != substituted.identity_digest
    assert real.authority_payload() != substituted.authority_payload()


def test_source_snapshot_rehash_detects_post_verification_mutation(tmp_path):
    repo = tmp_path / "copied-repo"
    for relative in authority_module._required_source_paths(ROOT):
        destination = repo / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    subprocess.run(("git", "init", "-q", str(repo)), check=True)
    subprocess.run(("git", "-C", str(repo), "add", "."), check=True)
    subprocess.run(
        (
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Phase5 Test",
            "-c",
            "user.email=phase5@example.invalid",
            "commit",
            "-qm",
            "snapshot",
        ),
        check=True,
    )
    snapshot = _source_snapshot(tmp_path / "snapshot", repo_root=repo)

    assert snapshot.valid
    target = repo / "src/datadiff_osc/runtime/gates.py"
    target.write_bytes(target.read_bytes() + b"\n# post-snapshot mutation\n")
    assert snapshot.valid is False
    reverified = verify_source_snapshot(
        repo_root=repo,
        snapshot_path=Path(snapshot.snapshot_path),
        expected_snapshot_sha256=snapshot.expected_snapshot_sha256,
    )
    assert "source_snapshot_file_hash_mismatch:src/datadiff_osc/runtime/gates.py" in reverified.verification_errors


def test_direct_publishable_and_zero_gate_claims_are_rejected(tmp_path):
    _, _, _, result = _incomplete_bundle(tmp_path)
    fake_artifact = VerifiedArtifactReceipt(
        "fake",
        str(tmp_path / "missing.json"),
        "0" * 64,
        stable_digest("fake-provenance", "fake"),
        "contract_exact_replay",
        stable_digest("fake-receipt", "fake"),
        (),
    )
    with pytest.raises(ValueError, match="frozen 41/38"):
        replace(
            result,
            errors=(),
            artifact_receipts=(fake_artifact,),
            publishable=True,
        )
    with pytest.raises(ValueError, match="eligibility"):
        RootGatePreflightReport(
            source_digest="source",
            build_result_digest="build",
            authority_plan_sha256="0" * 64,
            artifact_manifest_sha256="0" * 64,
            gate_report_digest="report",
            verification_errors=(),
            full_gate_count=0,
            all_gates_pass=True,
            phase6_gate_eligible=True,
        )


def test_nonpublishable_build_cannot_create_output_or_parent(tmp_path):
    source, dynamic, receipts, result = _incomplete_bundle(tmp_path)
    output = tmp_path / "never-created" / "authority.json"

    with pytest.raises(ValueError, match="non-publishable"):
        publish_phase6_gate_authority_bundle(
            result=result,
            source=source,
            templates=legacy_v4_target_templates(),
            dynamic_plan=dynamic,
            artifact_receipts=receipts,
            output_path=output,
        )
    assert not output.exists()
    assert not output.parent.exists()


def _mock_publishable_result(result: RootGateBuildResult, tmp_path: Path):
    artifacts = tuple(
        VerifiedArtifactReceipt(
            f"artifact-{kind}",
            str(tmp_path / f"{kind}.json"),
            "0" * 64,
            stable_digest("test-artifact-provenance", kind),
            kind,
            stable_digest("test-producer-receipt", kind),
            (),
        )
        for kind in sorted(authority_module._REQUIRED_PRODUCER_KINDS)
    )
    payload = {
        "schema_version": GATE_AUTHORITY_PLAN_SCHEMA_VERSION,
        "source_digest": result.source_digest,
        "gate_spec_set_digest": EXPECTED_GATE_SPEC_SET_DIGEST,
        "artifacts": [
            {
                "artifact_id": item.artifact_id,
                "sha256": item.byte_sha256,
                "schema_version": RAW_GATE_ARTIFACT_SCHEMA_VERSION,
                "provenance_digest": item.provenance_digest,
            }
            for item in artifacts
        ],
        "denominator_universes": [
            item.authority_payload() for item in result.denominator_universes
        ],
    }
    return RootGateBuildResult(
        source_digest=result.source_digest,
        compiled_universe_digest=EXPECTED_COMPILED_UNIVERSE_DIGEST,
        frozen_spec_set_digest=EXPECTED_GATE_SPEC_SET_DIGEST,
        denominator_universes=result.denominator_universes,
        artifact_receipts=artifacts,
        authority_plan_json=canonical_json(payload),
        errors=(),
        publishable=True,
    )


def test_publish_postlink_exception_rolls_back_authority(tmp_path, monkeypatch):
    _, _, _, result = _incomplete_bundle(tmp_path)
    publishable = _mock_publishable_result(result, tmp_path)
    monkeypatch.setattr(
        authority_module,
        "build_phase6_gate_authority_bundle",
        lambda **_: publishable,
    )
    calls = 0

    def fake_verify(_path, *, expected_sha256, expected_source_digest):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("post-link verification fault")
        return SimpleNamespace(
            valid=True,
            verification_errors=(),
            byte_sha256=expected_sha256,
        )

    monkeypatch.setattr(authority_module, "verify_gate_authority_plan", fake_verify)
    output = tmp_path / "publish" / "authority.json"
    with pytest.raises(RuntimeError, match="post-link"):
        publish_phase6_gate_authority_bundle(
            result=publishable,
            source=None,
            templates=(),
            dynamic_plan=None,
            artifact_receipts=None,
            output_path=output,
        )

    assert not output.exists()
    assert not tuple(output.parent.glob(".osc-phase6-authority-*.tmp"))


def test_link_success_then_wrapper_exception_uses_inode_rollback(
    tmp_path, monkeypatch
):
    _, _, _, result = _incomplete_bundle(tmp_path)
    publishable = _mock_publishable_result(result, tmp_path)
    monkeypatch.setattr(
        authority_module,
        "build_phase6_gate_authority_bundle",
        lambda **_: publishable,
    )

    def fake_verify(_path, *, expected_sha256, expected_source_digest):
        return SimpleNamespace(
            valid=True,
            verification_errors=(),
            byte_sha256=expected_sha256,
        )

    real_link = authority_module.os.link

    def link_then_raise(source, destination):
        real_link(source, destination)
        raise RuntimeError("link wrapper raised after kernel success")

    monkeypatch.setattr(authority_module, "verify_gate_authority_plan", fake_verify)
    monkeypatch.setattr(authority_module.os, "link", link_then_raise)
    output = tmp_path / "link-wrapper-fault" / "authority.json"
    with pytest.raises(RuntimeError, match="kernel success"):
        publish_phase6_gate_authority_bundle(
            result=publishable,
            source=None,
            templates=(),
            dynamic_plan=None,
            artifact_receipts=None,
            output_path=output,
        )

    assert not output.exists()
    assert not tuple(output.parent.glob(".osc-phase6-authority-*.tmp"))


def test_publish_race_preserves_symlink_competitor(tmp_path, monkeypatch):
    _, _, _, result = _incomplete_bundle(tmp_path)
    publishable = _mock_publishable_result(result, tmp_path)
    monkeypatch.setattr(
        authority_module,
        "build_phase6_gate_authority_bundle",
        lambda **_: publishable,
    )

    def fake_verify(_path, *, expected_sha256, expected_source_digest):
        return SimpleNamespace(
            valid=True,
            verification_errors=(),
            byte_sha256=expected_sha256,
        )

    def symlink_then_eexist(source, destination):
        Path(destination).symlink_to(Path(source))
        raise FileExistsError("symlink competitor won")

    monkeypatch.setattr(authority_module, "verify_gate_authority_plan", fake_verify)
    monkeypatch.setattr(authority_module.os, "link", symlink_then_eexist)
    output = tmp_path / "symlink-race" / "authority.json"
    with pytest.raises(FileExistsError, match="competitor"):
        publish_phase6_gate_authority_bundle(
            result=publishable,
            source=None,
            templates=(),
            dynamic_plan=None,
            artifact_receipts=None,
            output_path=output,
        )

    assert output.is_symlink()


def test_postlink_replacement_preserves_competitor_inode(tmp_path, monkeypatch):
    _, _, _, result = _incomplete_bundle(tmp_path)
    publishable = _mock_publishable_result(result, tmp_path)
    monkeypatch.setattr(
        authority_module,
        "build_phase6_gate_authority_bundle",
        lambda **_: publishable,
    )
    calls = 0

    def replace_before_second_verify(path, *, expected_sha256, expected_source_digest):
        nonlocal calls
        calls += 1
        if calls == 2:
            candidate = Path(path)
            candidate.unlink()
            candidate.write_text("competitor", encoding="utf-8")
            raise RuntimeError("destination replaced after link")
        return SimpleNamespace(
            valid=True,
            verification_errors=(),
            byte_sha256=expected_sha256,
        )

    monkeypatch.setattr(
        authority_module,
        "verify_gate_authority_plan",
        replace_before_second_verify,
    )
    output = tmp_path / "postlink-replacement" / "authority.json"
    with pytest.raises(RuntimeError, match="replaced after link"):
        publish_phase6_gate_authority_bundle(
            result=publishable,
            source=None,
            templates=(),
            dynamic_plan=None,
            artifact_receipts=None,
            output_path=output,
        )

    assert output.read_text(encoding="utf-8") == "competitor"


def test_publish_fsyncs_directory_and_never_authorizes_24h(tmp_path, monkeypatch):
    _, _, _, result = _incomplete_bundle(tmp_path)
    publishable = _mock_publishable_result(result, tmp_path)
    monkeypatch.setattr(
        authority_module,
        "build_phase6_gate_authority_bundle",
        lambda **_: publishable,
    )

    def fake_verify(_path, *, expected_sha256, expected_source_digest):
        return SimpleNamespace(
            valid=True,
            verification_errors=(),
            byte_sha256=expected_sha256,
        )

    synced: list[Path] = []
    monkeypatch.setattr(authority_module, "verify_gate_authority_plan", fake_verify)
    monkeypatch.setattr(authority_module, "_fsync_directory", synced.append)
    output = tmp_path / "publish-success" / "authority.json"
    published = publish_phase6_gate_authority_bundle(
        result=publishable,
        source=None,
        templates=(),
        dynamic_plan=None,
        artifact_receipts=None,
        output_path=output,
    )

    assert output.exists()
    assert synced == [output.parent]
    assert published.twenty_four_hour_run_authorized is False


def test_preflight_missing_inputs_returns_structured_no_go(tmp_path):
    source, dynamic, receipts, _ = _incomplete_bundle(tmp_path)
    report = preflight_phase6_gate_authority_bundle(
        source=source,
        templates=legacy_v4_target_templates(),
        dynamic_plan=dynamic,
        artifact_receipts=receipts,
        authority_plan_path=tmp_path / "missing-authority.json",
        expected_authority_plan_sha256="0" * 64,
        artifact_manifest_path=tmp_path / "missing-manifest.json",
        expected_artifact_manifest_sha256="0" * 64,
    )

    assert report.phase6_gate_eligible is False
    assert report.twenty_four_hour_run_authorized is False
    assert report.verification_errors
    assert "root_gate_build_not_publishable" in report.verification_errors


def test_preflight_symlink_loop_returns_structured_no_go(tmp_path):
    source, dynamic, receipts, _ = _incomplete_bundle(tmp_path)
    first = tmp_path / "authority-loop-a.json"
    second = tmp_path / "authority-loop-b.json"
    first.symlink_to(second.name)
    second.symlink_to(first.name)

    report = preflight_phase6_gate_authority_bundle(
        source=source,
        templates=legacy_v4_target_templates(),
        dynamic_plan=dynamic,
        artifact_receipts=receipts,
        authority_plan_path=first,
        expected_authority_plan_sha256="0" * 64,
        artifact_manifest_path=tmp_path / "missing-manifest.json",
        expected_artifact_manifest_sha256="0" * 64,
    )

    assert report.phase6_gate_eligible is False
    assert report.twenty_four_hour_run_authorized is False
    assert any(
        error.startswith("authority_plan_verification_failed:RuntimeError")
        for error in report.verification_errors
    )


@pytest.mark.parametrize(
    ("target", "build_error", "preflight_error"),
    (
        (
            "source",
            "trusted_source_snapshot_verification_failed:RuntimeError",
            "source_snapshot_reverification_failed:RuntimeError",
        ),
        (
            "dynamic",
            "trusted_dynamic_plan_verification_failed:RuntimeError",
            "dynamic_plan_reverification_failed:RuntimeError",
        ),
        (
            "receipts",
            "trusted_artifact_receipts_verification_failed:RuntimeError",
            "artifact_receipts_reverification_failed:RuntimeError",
        ),
    ),
)
def test_trusted_input_symlink_races_return_structured_no_go(
    tmp_path, target, build_error, preflight_error
):
    source, dynamic, receipts, _ = _incomplete_bundle(tmp_path)
    if target == "dynamic":
        dynamic = replace(dynamic, verification_errors=())
    if target == "receipts":
        receipts = replace(receipts, verification_errors=())
    selected = {
        "source": Path(source.snapshot_path),
        "dynamic": Path(dynamic.plan_path),
        "receipts": Path(receipts.index_path),
    }[target]
    sibling = selected.with_name(selected.name + ".loop")
    selected.unlink()
    selected.symlink_to(sibling.name)
    sibling.symlink_to(selected.name)

    result = build_phase6_gate_authority_bundle(
        source=source,
        templates=legacy_v4_target_templates(),
        dynamic_plan=dynamic,
        artifact_receipts=receipts,
    )
    assert result.publishable is False
    assert any(error.startswith(build_error) for error in result.errors)

    report = preflight_phase6_gate_authority_bundle(
        source=source,
        templates=legacy_v4_target_templates(),
        dynamic_plan=dynamic,
        artifact_receipts=receipts,
        authority_plan_path=tmp_path / "missing-authority.json",
        expected_authority_plan_sha256="0" * 64,
        artifact_manifest_path=tmp_path / "missing-manifest.json",
        expected_artifact_manifest_sha256="0" * 64,
    )
    assert report.phase6_gate_eligible is False
    assert report.twenty_four_hour_run_authorized is False
    assert any(error.startswith(preflight_error) for error in report.verification_errors)


def test_preflight_binds_runtime_consumed_manifest_hash(tmp_path, monkeypatch):
    source, dynamic, receipts, _ = _incomplete_bundle(tmp_path)
    consumed_sha = "1" * 64
    evidence = SimpleNamespace(
        verification_errors=(),
        manifest_sha256=consumed_sha,
        artifacts=(object(),),
    )
    gate_report = SimpleNamespace(
        verification_errors=(),
        artifact_manifest_sha256="2" * 64,
        authority_plan_sha256="0" * 64,
        authority_eligible=False,
        results=tuple(range(41)),
        digest=stable_digest("test-gate-report", "split"),
        all_gates_pass=False,
    )
    monkeypatch.setattr(
        authority_module,
        "verify_gate_artifact_manifest",
        lambda *args, **kwargs: evidence,
    )
    monkeypatch.setattr(
        authority_module,
        "calculate_gate_report",
        lambda *args, **kwargs: gate_report,
    )
    report = preflight_phase6_gate_authority_bundle(
        source=source,
        templates=legacy_v4_target_templates(),
        dynamic_plan=dynamic,
        artifact_receipts=receipts,
        authority_plan_path=tmp_path / "missing-authority.json",
        expected_authority_plan_sha256="0" * 64,
        artifact_manifest_path=tmp_path / "missing-manifest.json",
        expected_artifact_manifest_sha256="0" * 64,
    )

    assert "runtime_evidence_manifest_sha_mismatch" in report.verification_errors
    assert "runtime_gate_report_manifest_sha_mismatch" in report.verification_errors
    assert report.artifact_manifest_sha256 == consumed_sha
    assert report.phase6_gate_eligible is False


def test_build_cli_exits_2_and_leaves_no_authority_output(tmp_path):
    source = _source_snapshot(tmp_path)
    dynamic = _incomplete_dynamic_plan(tmp_path, source.source_digest)
    receipts = _empty_receipts(tmp_path, source.source_digest, dynamic.byte_sha256)
    output = tmp_path / "cli-output" / "authority.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/osc/build_phase6_gate_authority_bundle.py"),
            "--repo-root",
            str(ROOT),
            "--source-snapshot",
            source.snapshot_path,
            "--source-snapshot-sha256",
            source.expected_snapshot_sha256,
            "--dynamic-plan",
            dynamic.plan_path,
            "--dynamic-plan-sha256",
            dynamic.expected_plan_sha256,
            "--artifact-receipts",
            receipts.index_path,
            "--artifact-receipts-sha256",
            receipts.expected_index_sha256,
            "--output",
            str(output),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 2
    assert not output.exists()
    assert not output.parent.exists()
    assert '"publishable":false' in completed.stdout
    assert '"twenty_four_hour_run_authorized":false' in completed.stdout


def test_build_cli_has_no_fallible_report_after_publish_commit(tmp_path, monkeypatch):
    script_path = ROOT / "scripts/osc/build_phase6_gate_authority_bundle.py"
    spec = importlib.util.spec_from_file_location("phase6_authority_build_cli", script_path)
    assert spec is not None
    assert spec.loader is not None
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)

    source = SimpleNamespace(source_digest="source")
    dynamic = SimpleNamespace(byte_sha256="dynamic")
    receipts = SimpleNamespace()
    result = SimpleNamespace(
        publishable=True,
        to_dict=lambda: {
            "publishable": True,
            "twenty_four_hour_run_authorized": False,
        },
    )
    monkeypatch.setattr(cli, "verify_source_snapshot", lambda **_: source)
    monkeypatch.setattr(cli, "verify_dynamic_gate_plan", lambda **_: dynamic)
    monkeypatch.setattr(cli, "verify_artifact_receipt_index", lambda **_: receipts)
    monkeypatch.setattr(cli, "legacy_v4_target_templates", lambda: ())
    monkeypatch.setattr(cli, "build_phase6_gate_authority_bundle", lambda **_: result)

    output = tmp_path / "published" / "authority.json"

    def publish(**kwargs):
        destination = kwargs["output_path"]
        destination.parent.mkdir(parents=True)
        destination.write_text("authority", encoding="utf-8")
        return {"authority_plan_sha256": "0" * 64}

    monkeypatch.setattr(cli, "publish_phase6_gate_authority_bundle", publish)

    class FailOnSecondWrite:
        calls = 0
        flushes = 0

        def write(self, value):
            self.calls += 1
            if self.calls > 1:
                raise BrokenPipeError("post-publish report failure")
            return len(value)

        def flush(self):
            self.flushes += 1

    stdout = FailOnSecondWrite()
    monkeypatch.setattr(cli.sys, "stdout", stdout)
    exit_code = cli.main(
        [
            "--source-snapshot",
            str(tmp_path / "source.json"),
            "--source-snapshot-sha256",
            "0" * 64,
            "--dynamic-plan",
            str(tmp_path / "dynamic.json"),
            "--dynamic-plan-sha256",
            "1" * 64,
            "--artifact-receipts",
            str(tmp_path / "receipts.json"),
            "--artifact-receipts-sha256",
            "2" * 64,
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    assert stdout.calls == 1
    assert stdout.flushes == 1
    assert output.read_text(encoding="utf-8") == "authority"


def test_build_cli_flushes_report_before_publish_commit(tmp_path, monkeypatch):
    script_path = ROOT / "scripts/osc/build_phase6_gate_authority_bundle.py"
    spec = importlib.util.spec_from_file_location(
        "phase6_authority_flush_build_cli", script_path
    )
    assert spec is not None
    assert spec.loader is not None
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)

    source = SimpleNamespace(source_digest="source")
    dynamic = SimpleNamespace(byte_sha256="dynamic")
    receipts = SimpleNamespace()
    result = SimpleNamespace(
        publishable=True,
        to_dict=lambda: {
            "publishable": True,
            "twenty_four_hour_run_authorized": False,
        },
    )
    monkeypatch.setattr(cli, "verify_source_snapshot", lambda **_: source)
    monkeypatch.setattr(cli, "verify_dynamic_gate_plan", lambda **_: dynamic)
    monkeypatch.setattr(cli, "verify_artifact_receipt_index", lambda **_: receipts)
    monkeypatch.setattr(cli, "legacy_v4_target_templates", lambda: ())
    monkeypatch.setattr(cli, "build_phase6_gate_authority_bundle", lambda **_: result)

    output = tmp_path / "must-not-publish" / "authority.json"
    publish_called = False

    def publish(**_kwargs):
        nonlocal publish_called
        publish_called = True

    monkeypatch.setattr(cli, "publish_phase6_gate_authority_bundle", publish)

    class FlushFails:
        def write(self, value):
            return len(value)

        def flush(self):
            raise BrokenPipeError("buffered report was not delivered")

    monkeypatch.setattr(cli.sys, "stdout", FlushFails())
    exit_code = cli.main(
        [
            "--source-snapshot",
            str(tmp_path / "source.json"),
            "--source-snapshot-sha256",
            "0" * 64,
            "--dynamic-plan",
            str(tmp_path / "dynamic.json"),
            "--dynamic-plan-sha256",
            "1" * 64,
            "--artifact-receipts",
            str(tmp_path / "receipts.json"),
            "--artifact-receipts-sha256",
            "2" * 64,
            "--output",
            str(output),
        ]
    )

    assert exit_code == 2
    assert publish_called is False
    assert not output.exists()


def test_build_cli_real_closed_pipe_exits_2_without_publish(tmp_path):
    script_path = ROOT / "scripts/osc/build_phase6_gate_authority_bundle.py"
    publish_marker = tmp_path / "publish-was-called"
    output = tmp_path / "closed-pipe" / "authority.json"
    probe = textwrap.dedent(
        f"""
        import importlib.util
        from pathlib import Path
        from types import SimpleNamespace
        import sys

        spec = importlib.util.spec_from_file_location(
            "phase6_authority_real_pipe_cli", Path({str(script_path)!r})
        )
        assert spec is not None and spec.loader is not None
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        source = SimpleNamespace(source_digest="source")
        dynamic = SimpleNamespace(byte_sha256="dynamic")
        receipts = SimpleNamespace()
        result = SimpleNamespace(
            publishable=True,
            to_dict=lambda: {{
                "publishable": True,
                "twenty_four_hour_run_authorized": False,
            }},
        )
        cli.verify_source_snapshot = lambda **_: source
        cli.verify_dynamic_gate_plan = lambda **_: dynamic
        cli.verify_artifact_receipt_index = lambda **_: receipts
        cli.legacy_v4_target_templates = lambda: ()
        cli.build_phase6_gate_authority_bundle = lambda **_: result
        def publish(**_kwargs):
            Path({str(publish_marker)!r}).write_text("called", encoding="utf-8")
        cli.publish_phase6_gate_authority_bundle = publish
        sys.stdin.buffer.read(1)
        raise SystemExit(cli.main([
            "--source-snapshot", {str(tmp_path / "source.json")!r},
            "--source-snapshot-sha256", "0" * 64,
            "--dynamic-plan", {str(tmp_path / "dynamic.json")!r},
            "--dynamic-plan-sha256", "1" * 64,
            "--artifact-receipts", {str(tmp_path / "receipts.json")!r},
            "--artifact-receipts-sha256", "2" * 64,
            "--output", {str(output)!r},
        ]))
        """
    )
    process = subprocess.Popen(
        [sys.executable, "-c", probe],
        cwd=ROOT,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdin is not None
    assert process.stdout is not None
    assert process.stderr is not None
    process.stdout.close()
    process.stdin.write(b"x")
    process.stdin.close()
    stderr = process.stderr.read().decode()
    returncode = process.wait()

    assert returncode == 2
    assert "phase6 authority build refused: BrokenPipeError" in stderr
    assert "Exception ignored" not in stderr
    assert not publish_marker.exists()
    assert not output.exists()


def test_builder_does_not_accept_manifest_claims_or_authority_flags():
    parameters = set(inspect.signature(build_phase6_gate_authority_bundle).parameters)
    assert parameters == {"source", "templates", "dynamic_plan", "artifact_receipts"}
    assert "manifest" not in parameters
    assert "claimed_metrics" not in parameters
    assert "thresholds" not in parameters
    assert "twenty_four_hour_run_authorized" not in parameters
