from __future__ import annotations

from dataclasses import replace
import hashlib

import pytest

from datadiff_osc._canonical import to_primitive
from datadiff_osc.runtime._private_receipts import (
    CandidateClassificationReceipt,
    CandidateRecheckBinding,
    ContractPerformanceSampleBinding,
    PairedBenchmarkReceipt,
    ParallelScalingSampleBinding,
    RepositoryTestNodeBinding,
    RepositoryTestReceipt,
    TargetPackageVersionBinding,
    TargetVersionReceipt,
    V3BackendTaskBinding,
    V3CaseBinding,
    V3RunReceipt,
    repository_test_collection_digest,
)
from datadiff_osc.schemas import EvidenceState, ExecutionStatus


def _digest(label: str) -> str:
    return f"test-{label}-digest"


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _backend(case_index: int, backend_index: int = 0) -> V3BackendTaskBinding:
    suffix = f"{case_index:03d}-{backend_index}"
    return V3BackendTaskBinding(
        task_id=f"backend-task-{suffix}",
        endpoint_digest=_digest(f"endpoint-{suffix}"),
        task_spec_digest=_digest(f"task-spec-{suffix}"),
        execution_outcome_digest=_digest(f"outcome-{suffix}"),
        status=ExecutionStatus.OK,
    )


def _case(case_index: int) -> V3CaseBinding:
    suffix = f"{case_index:03d}"
    return V3CaseBinding(
        case_id=f"case-{suffix}",
        case_index=case_index,
        seed_lineage_digest=_digest(f"seed-{suffix}"),
        case_task_id=f"case-task-{suffix}",
        case_task_spec_digest=_digest(f"case-task-spec-{suffix}"),
        result_digest=_digest(f"result-{suffix}"),
        pipeline_digest=_digest(f"pipeline-{suffix}"),
        execution_outcome_digest=_digest(f"case-outcome-{suffix}"),
        evidence_envelope_digest=_digest(f"evidence-{suffix}"),
        executed=True,
        iteration_failure=False,
        pipeline_error=False,
        backend_tasks=(_backend(case_index),),
    )


def _v3_receipt() -> V3RunReceipt:
    return V3RunReceipt(
        source_digest=_digest("source"),
        protocol_digest=_digest("protocol"),
        v3_gate_plan_digest=_digest("v3-plan"),
        run_id="run-001",
        lane_id="lane-001",
        seed=1001,
        execution_run_digest=_digest("execution-run"),
        run_completed=True,
        cases=tuple(_case(index) for index in range(100)),
    )


def _repository_receipt() -> RepositoryTestReceipt:
    nodes = (
        RepositoryTestNodeBinding("tests/a.py::test_a", "passed", _digest("a")),
        RepositoryTestNodeBinding("tests/b.py::test_b", "skipped", _digest("b")),
    )
    return RepositoryTestReceipt(
        source_digest=_digest("source"),
        test_plan_digest=_digest("test-plan"),
        collection_digest=repository_test_collection_digest(
            tuple(item.node_id for item in nodes)
        ),
        config_digest=_digest("pytest-config"),
        environment_digest=_digest("environment"),
        command_digest=_digest("command"),
        junit_sha256=_sha("junit"),
        log_sha256=_sha("log"),
        exit_code=0,
        nodes=nodes,
    )


def _package(
    package_id: str = "pandas", *, installed: str = "3.0", latest: str = "3.0"
) -> TargetPackageVersionBinding:
    return TargetPackageVersionBinding(
        package_id=package_id,
        distribution_name=package_id,
        import_name=package_id,
        installed_version=installed,
        latest_version=latest,
        installed_metadata_digest=_digest(f"metadata-{package_id}"),
        version_source_kind="pypi_json",
        version_source_digest=_digest(f"source-{package_id}"),
        version_source_sha256=_sha(f"source-{package_id}"),
    )


def _target_receipt(*packages: TargetPackageVersionBinding) -> TargetVersionReceipt:
    return TargetVersionReceipt(
        source_digest=_digest("source"),
        audit_plan_digest=_digest("audit-plan"),
        environment_digest=_digest("environment"),
        python_runtime_digest=_digest("python"),
        packages=packages or (_package(),),
    )


def _rechecks(prefix: str) -> tuple[CandidateRecheckBinding, ...]:
    return tuple(
        CandidateRecheckBinding(
            recheck_id=f"{prefix}-{ordinal}",
            ordinal=ordinal,
            task_id=f"{prefix}-task-{ordinal}",
            seed_lineage_digest=_digest(f"{prefix}-seed-{ordinal}"),
            execution_outcome_digest=_digest(f"{prefix}-outcome-{ordinal}"),
            completed=True,
            reproduced=True,
        )
        for ordinal in (1, 2, 3)
    )


def _candidate_receipt() -> CandidateClassificationReceipt:
    return CandidateClassificationReceipt(
        source_digest=_digest("source"),
        v3_gate_plan_digest=_digest("v3-plan"),
        classification_policy_digest=_digest("classification-policy"),
        classification_id="classification-001",
        source_case_id="case-001",
        result_group_digest=_digest("result-group"),
        evidence_envelope_digest=_digest("evidence"),
        evidence_state=EvidenceState.STABLE_SURVIVOR,
        finding_record_id="finding-001",
        candidate_id="candidate-001",
        classification_digest=_digest("classification"),
        campaign_rechecks=_rechecks("campaign"),
        pipeline_rechecks=_rechecks("pipeline"),
        bug_claimed=False,
    )


def _paired_receipt() -> PairedBenchmarkReceipt:
    return PairedBenchmarkReceipt(
        source_digest=_digest("source"),
        benchmark_plan_digest=_digest("benchmark-plan"),
        environment_digest=_digest("environment"),
        baseline_config_digest=_digest("baseline-config"),
        treatment_config_digest=_digest("treatment-config"),
        contract_samples=(
            ContractPerformanceSampleBinding(
                sample_id="contract-001",
                case_id="case-001",
                baseline_task_id="baseline-task-001",
                treatment_task_id="treatment-task-001",
                compile_match_ms=1.0,
                case_wall_ms=20.0,
                baseline_throughput_cases_s=100.0,
                treatment_throughput_cases_s=95.0,
            ),
        ),
        parallel_samples=(
            ParallelScalingSampleBinding(
                sample_id="parallel-001",
                workload_id="workload-001",
                one_worker_task_set_digest=_digest("task-set"),
                six_worker_task_set_digest=_digest("task-set"),
                one_worker_result_digest=_digest("one-result"),
                six_worker_result_digest=_digest("six-result"),
                one_worker_elapsed_seconds=60.0,
                six_worker_elapsed_seconds=12.0,
            ),
        ),
    )


def test_v3_receipt_binds_exact_100_case_and_backend_task_identities():
    receipt = _v3_receipt()

    assert len(receipt.case_ids) == 100
    assert receipt.case_ids[0] == "case-000"
    assert receipt.case_ids[-1] == "case-099"
    assert receipt.executed_case_ids == receipt.case_ids
    assert len(receipt.backend_task_ids) == 100


def test_v3_receipt_rejects_duplicate_missing_or_relabelled_cases():
    receipt = _v3_receipt()

    with pytest.raises(ValueError, match="exactly 100"):
        replace(receipt, cases=receipt.cases[:-1])
    with pytest.raises(ValueError, match="indices"):
        replace(
            receipt,
            cases=(replace(receipt.cases[0], case_index=1), *receipt.cases[1:]),
        )
    with pytest.raises(ValueError, match="case identities"):
        replace(
            receipt,
            cases=(
                receipt.cases[0],
                replace(receipt.cases[1], case_id=receipt.cases[0].case_id),
                *receipt.cases[2:],
            ),
        )


def test_v3_receipt_preserves_non_ok_backend_outcomes():
    receipt = _v3_receipt()
    failed_backend = replace(receipt.cases[0].backend_tasks[0], status=ExecutionStatus.TIMEOUT)
    failed_case = replace(receipt.cases[0], backend_tasks=(failed_backend,))
    failed_receipt = replace(receipt, cases=(failed_case, *receipt.cases[1:]))

    assert failed_receipt.cases[0].backend_tasks[0].status is ExecutionStatus.TIMEOUT
    assert to_primitive(failed_receipt)["cases"][0]["backend_tasks"][0]["status"] == "timeout"


def test_repository_receipt_recomputes_collection_identity():
    receipt = _repository_receipt()

    assert receipt.collection_digest == repository_test_collection_digest(
        receipt.node_ids
    )
    with pytest.raises(ValueError, match="collection digest mismatch"):
        replace(receipt, collection_digest=_digest("substituted-collection"))


def test_repository_receipt_rejects_zero_exit_with_failures():
    receipt = _repository_receipt()
    failed = replace(receipt.nodes[0], outcome="failed")
    nodes = (failed, receipt.nodes[1])

    with pytest.raises(ValueError, match="zero repository test exit"):
        replace(
            receipt,
            nodes=nodes,
            collection_digest=repository_test_collection_digest(
                tuple(item.node_id for item in nodes)
            ),
        )


def test_target_receipt_derives_mismatch_without_caller_boolean():
    receipt = _target_receipt(_package(installed="2.9", latest="3.0"))

    assert receipt.mismatched_package_ids == ("pandas",)
    assert receipt.all_match_latest is False
    assert "up_to_date" not in to_primitive(receipt)


def test_target_receipt_rejects_offline_override_as_authority():
    package = _package()

    with pytest.raises(ValueError, match="not authority eligible"):
        replace(package, version_source_kind="offline_override")


def test_candidate_receipt_never_claims_or_implies_confirmed_bug():
    receipt = _candidate_receipt()

    assert receipt.bug_claimed is False
    assert "confirmed_bug" not in to_primitive(receipt)
    with pytest.raises(ValueError, match="bug_claimed"):
        replace(receipt, bug_claimed=True)
    with pytest.raises(ValueError, match="confirmed evidence states"):
        replace(receipt, evidence_state=EvidenceState.UNIQUE_ROOT)
    with pytest.raises(ValueError, match="confirmed evidence states"):
        replace(receipt, evidence_state=EvidenceState.INDEPENDENTLY_CONFIRMED)


def test_candidate_receipt_requires_exact_campaign_and_pipeline_ordinals():
    receipt = _candidate_receipt()

    with pytest.raises(ValueError, match="campaign recheck ordinals"):
        replace(receipt, campaign_rechecks=receipt.campaign_rechecks[:-1])
    invalid = (
        receipt.pipeline_rechecks[0],
        replace(receipt.pipeline_rechecks[1], ordinal=3),
        replace(receipt.pipeline_rechecks[2], ordinal=4),
    )
    with pytest.raises(ValueError, match="pipeline recheck ordinals"):
        replace(receipt, pipeline_rechecks=invalid)


def test_zero_candidate_receipt_admits_only_classified_v3_case():
    receipt = replace(
        _candidate_receipt(),
        evidence_state=EvidenceState.NOT_A_CANDIDATE,
        finding_record_id="",
        candidate_id="",
        campaign_rechecks=(),
        pipeline_rechecks=(),
    )

    assert receipt.is_candidate is False
    assert receipt.source_case_id == "case-001"
    assert receipt.campaign_recheck_ids == receipt.pipeline_recheck_ids == ()


def test_paired_receipt_derives_all_scalar_metrics_from_raw_pairs():
    receipt = _paired_receipt()

    assert receipt.compile_match_samples_ms == (1.0,)
    assert receipt.compile_wall_shares == pytest.approx((0.05,))
    assert receipt.throughput_regressions == pytest.approx((0.05,))
    assert receipt.six_worker_efficiencies == pytest.approx((5.0 / 6.0,))
    payload = to_primitive(receipt)
    assert "compile_match_p95_ms" not in payload
    assert "six_worker_efficiency" not in payload
    zero_treatment = replace(
        receipt.contract_samples[0], treatment_throughput_cases_s=0.0
    )
    assert zero_treatment.throughput_regression == 1.0
    with pytest.raises(ValueError, match="finite number"):
        replace(receipt.contract_samples[0], compile_match_ms=float("nan"))


def test_paired_receipt_rejects_task_set_or_sample_identity_substitution():
    receipt = _paired_receipt()
    parallel = receipt.parallel_samples[0]

    with pytest.raises(ValueError, match="task-set identity mismatch"):
        replace(
            parallel,
            six_worker_task_set_digest=_digest("substituted-task-set"),
        )
    with pytest.raises(ValueError, match="globally unique"):
        replace(
            receipt,
            parallel_samples=(replace(parallel, sample_id="contract-001"),),
        )
