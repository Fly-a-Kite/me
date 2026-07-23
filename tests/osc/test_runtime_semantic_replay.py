from __future__ import annotations

import hashlib
import inspect

import pytest

from datadiff_osc._canonical import (
    canonical_envelope,
    canonical_roundtrip,
    decode_canonical_envelope,
    to_primitive,
)
from datadiff_osc.parallel.invariance import (
    AuthorityEvidenceSnapshot,
    InvarianceReport,
)
from datadiff_osc.runtime._private_receipts import (
    CandidateClassificationReceipt,
    CandidateRecheckBinding,
    RepositoryTestNodeBinding,
    RepositoryTestReceipt,
    TargetPackageVersionBinding,
    TargetVersionReceipt,
    V3BackendTaskBinding,
    V3CaseBinding,
    V3RunReceipt,
    repository_test_collection_digest,
)
from datadiff_osc.runtime._phase6_contract_performance_receipts import (
    ContractPerformanceEvidenceReceipt,
)
from datadiff_osc.runtime._semantic_replay import (
    RUNTIME_REPLAY_REQUIRES_ROOT_PROVENANCE,
    replay_runtime_admission,
)
from datadiff_osc.schemas import (
    ContractFingerprint,
    EvidenceEnvelope,
    EvidenceState,
    ExecutionStatus,
    FailureKind,
    ResourceTokens,
    SeedLineage,
    SeedStage,
    StructuredExecutionOutcome,
    TargetFingerprint,
    TaskIdentity,
    TaskKind,
    TaskSpec,
    VerdictKind,
)


def _digest(label: str) -> str:
    return f"semantic-{label}-digest"


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _outcome() -> StructuredExecutionOutcome:
    return StructuredExecutionOutcome(
        endpoint_id="endpoint-001",
        status=ExecutionStatus.OK,
        failure_kind=FailureKind.NONE,
    )


def _lineage() -> SeedLineage:
    return SeedLineage(
        protocol_digest=_digest("protocol"),
        master_seed=101,
        lane_id="lane-001",
        case_index=7,
        stage_name=SeedStage.BACKEND,
    )


def _identity() -> TaskIdentity:
    return TaskIdentity(
        protocol_digest=_digest("protocol"),
        task_kind=TaskKind.BACKEND_EXECUTION,
        epoch_index=1,
        decision_index=7,
        seed_lineage_digest=_lineage().digest,
        contrast_set_id="contrast-001",
        endpoint_id="endpoint-001",
        backend="backend-001",
    )


def _task() -> TaskSpec:
    return TaskSpec(
        identity=_identity(),
        dependency_task_ids=(),
        resources=ResourceTokens(
            cpu_tokens=1,
            rss_bytes=1024,
            io_class="local",
            backend_internal_threads=1,
        ),
        payload_digest=_digest("payload"),
    )


def _evidence() -> EvidenceEnvelope:
    return EvidenceEnvelope.build(
        evidence_id="evidence-001",
        state=EvidenceState.FINDING,
        result_group_digest=_digest("result-group"),
        task_id=_identity().task_id,
        seed_lineage_digest=_lineage().digest,
        contract_fingerprint=ContractFingerprint(
            contract_digest=_digest("contract"),
            registry_digest=_digest("registry"),
        ),
        target_fingerprint=TargetFingerprint(
            universe_digest=_digest("universe"),
            taxonomy_digest=_digest("taxonomy"),
            template_digest=_digest("template"),
        ),
        derivation_certificate_digest=_digest("derivation"),
        applicability_certificate_digest=_digest("applicability"),
        activation_certificate_digest=_digest("activation"),
        observation_certificate_digest=_digest("observation"),
        execution_outcomes=(_outcome(),),
        verdict_kind=VerdictKind.VIOLATED,
        artifact_refs=("artifact-001",),
        metadata={
            "bytes": b"\x00runtime",
            "nested": ("value", 1.25, (True, None)),
        },
    )


def _snapshot() -> AuthorityEvidenceSnapshot:
    return AuthorityEvidenceSnapshot(
        execution_run_digest=_digest("execution-run"),
        worker_count=1,
        assignment_digest=_digest("assignment"),
        epoch_digest=_digest("epoch"),
        task_multiset_digest=_digest("task-multiset"),
        seed_multiset_digest=_digest("seed-multiset"),
        authority_verdict_digest=_digest("verdict"),
        coverage_bitmap_digest=_digest("coverage"),
        ledger_digest=_digest("ledger"),
        certificate_digests=(_digest("certificate"),),
    )


def _invariance() -> InvarianceReport:
    return InvarianceReport(
        worker_counts=(1, 6),
        assignment_invariant=True,
        epoch_invariant=True,
        task_multiset_invariant=True,
        task_set_invariant=True,
        seed_lineage_invariant=True,
        result_order_invariant=True,
        outcome_invariant=True,
        retry_trace_invariant=True,
        authority_verdict_invariant=True,
        coverage_bitmap_invariant=True,
        ledger_invariant=True,
        certificate_invariant=True,
        authority_evidence_complete=True,
        authority_evidence_errors=(),
        run_digests=(_digest("run-1"), _digest("run-6")),
        outcome_digests=(_digest("outcomes"), _digest("outcomes")),
        authority_evidence_digests=(
            _digest("authority-1"),
            _digest("authority-6"),
        ),
    )


def _v3() -> V3RunReceipt:
    cases = tuple(
        V3CaseBinding(
            case_id=f"case-{index:03d}",
            case_index=index,
            seed_lineage_digest=_digest(f"case-seed-{index:03d}"),
            case_task_id=f"case-task-{index:03d}",
            case_task_spec_digest=_digest(f"case-spec-{index:03d}"),
            result_digest=_digest(f"result-{index:03d}"),
            pipeline_digest=_digest(f"pipeline-{index:03d}"),
            execution_outcome_digest=_digest(f"case-outcome-{index:03d}"),
            evidence_envelope_digest=_digest(f"case-evidence-{index:03d}"),
            executed=True,
            iteration_failure=False,
            pipeline_error=False,
            backend_tasks=(
                V3BackendTaskBinding(
                    task_id=f"backend-task-{index:03d}",
                    endpoint_digest=_digest(f"endpoint-{index:03d}"),
                    task_spec_digest=_digest(f"backend-spec-{index:03d}"),
                    execution_outcome_digest=_digest(
                        f"backend-outcome-{index:03d}"
                    ),
                    status=ExecutionStatus.OK,
                ),
            ),
        )
        for index in range(100)
    )
    return V3RunReceipt(
        source_digest=_digest("source"),
        protocol_digest=_digest("protocol"),
        v3_gate_plan_digest=_digest("v3-plan"),
        run_id="run-001",
        lane_id="lane-001",
        seed=101,
        execution_run_digest=_digest("execution-run"),
        run_completed=True,
        cases=cases,
    )


def _repository() -> RepositoryTestReceipt:
    nodes = (
        RepositoryTestNodeBinding(
            "tests/osc/test_a.py::test_a", "passed", _digest("test-a")
        ),
    )
    return RepositoryTestReceipt(
        source_digest=_digest("source"),
        test_plan_digest=_digest("test-plan"),
        collection_digest=repository_test_collection_digest(
            tuple(item.node_id for item in nodes)
        ),
        config_digest=_digest("config"),
        environment_digest=_digest("environment"),
        command_digest=_digest("command"),
        junit_sha256=_sha("junit"),
        log_sha256=_sha("log"),
        exit_code=0,
        nodes=nodes,
    )


def _target() -> TargetVersionReceipt:
    package = TargetPackageVersionBinding(
        package_id="pandas",
        distribution_name="pandas",
        import_name="pandas",
        installed_version="3.0.0",
        latest_version="3.0.0",
        installed_metadata_digest=_digest("pandas-metadata"),
        version_source_kind="root_frozen_public_index",
        version_source_digest=_digest("pandas-index"),
        version_source_sha256=_sha("pandas-index"),
    )
    return TargetVersionReceipt(
        source_digest=_digest("source"),
        audit_plan_digest=_digest("audit-plan"),
        environment_digest=_digest("environment"),
        python_runtime_digest=_digest("python-runtime"),
        packages=(package,),
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


def _candidate() -> CandidateClassificationReceipt:
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


def _contract_performance() -> ContractPerformanceEvidenceReceipt:
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


def _owned_values() -> tuple[tuple[str, str, object, str, tuple[str, ...]], ...]:
    outcome = _outcome()
    evidence = _evidence()
    task = _task()
    identity = _identity()
    lineage = _lineage()
    invariance = _invariance()
    snapshot = _snapshot()
    v3 = _v3()
    repository = _repository()
    target = _target()
    candidate = _candidate()
    performance = _contract_performance()
    return (
        (
            "StructuredExecutionOutcome",
            outcome.schema_version,
            outcome,
            "execution_outcomes",
            (outcome.digest,),
        ),
        (
            "EvidenceEnvelope",
            evidence.schema_version,
            evidence,
            "evidence_envelopes",
            (evidence.digest,),
        ),
        ("TaskSpec", task.schema_version, task, "task_specs", (task.digest,)),
        (
            "TaskIdentity",
            identity.schema_version,
            identity,
            "task_identities",
            (identity.task_id,),
        ),
        (
            "SeedLineage",
            lineage.schema_version,
            lineage,
            "seed_lineages",
            (lineage.digest,),
        ),
        (
            "InvarianceReport",
            invariance.schema_version,
            invariance,
            "parallel_invariance_checks_total",
            tuple(
                sorted(
                    f"parallel:{name}"
                    for name in (
                        "assignment_invariant",
                        "authority_evidence_complete",
                        "authority_verdict_invariant",
                        "certificate_invariant",
                        "coverage_bitmap_invariant",
                        "epoch_invariant",
                        "ledger_invariant",
                        "outcome_invariant",
                        "result_order_invariant",
                        "retry_trace_invariant",
                        "seed_lineage_invariant",
                        "task_multiset_invariant",
                        "task_set_invariant",
                    )
                )
            ),
        ),
        (
            "AuthorityEvidenceSnapshot",
            snapshot.schema_version,
            snapshot,
            "authority_evidence_snapshots",
            (snapshot.digest,),
        ),
        ("V3RunReceipt", v3.schema_version, v3, "v3_runs", (v3.run_id,)),
        (
            "RepositoryTestReceipt",
            repository.schema_version,
            repository,
            "repository_tests",
            repository.node_ids,
        ),
        (
            "TargetVersionReceipt",
            target.schema_version,
            target,
            "target_packages",
            target.package_ids,
        ),
        (
            "CandidateClassificationReceipt",
            candidate.schema_version,
            candidate,
            "candidate_records",
            (candidate.finding_record_id,),
        ),
        (
            "ContractPerformanceEvidenceReceipt",
            performance.schema_version,
            performance,
            "contract_performance_samples",
            (performance.sample_id,),
        ),
    )


def _replay(
    type_name: str,
    schema_version: str,
    value: object,
    subject_kind: str,
    subject_ids: tuple[str, ...],
) -> tuple[str, ...]:
    decoded = decode_canonical_envelope(
        canonical_envelope(type_name, schema_version, value)
    )
    return replay_runtime_admission(
        envelope_type=type_name,
        schema_version=schema_version,
        payload=decoded["payload"],
        subject_kind=subject_kind,
        subject_ids=subject_ids,
    )


def test_runtime_replay_signature_is_exactly_frozen():
    signature = inspect.signature(replay_runtime_admission)

    assert tuple(signature.parameters) == (
        "envelope_type",
        "schema_version",
        "payload",
        "subject_kind",
        "subject_ids",
    )
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in signature.parameters.values()
    )


def test_runtime_replays_all_twelve_owned_types():
    values = _owned_values()

    assert len(values) == 12
    for type_name, version, value, kind, identities in values:
        assert _replay(type_name, version, value, kind, identities) == ()


def test_runtime_rejects_unknown_type_schema_subject_and_extra_field():
    outcome = _outcome()
    payload = to_primitive(outcome)

    assert _replay("Unknown", outcome.schema_version, outcome, "x", ("x",))
    assert _replay(
        "StructuredExecutionOutcome",
        "unknown-schema",
        outcome,
        "execution_outcomes",
        (outcome.digest,),
    )
    assert _replay(
        "StructuredExecutionOutcome",
        outcome.schema_version,
        outcome,
        "candidate_records",
        (outcome.digest,),
    )
    assert replay_runtime_admission(
        envelope_type="StructuredExecutionOutcome",
        schema_version=outcome.schema_version,
        payload={**payload, "extra": True},
        subject_kind="execution_outcomes",
        subject_ids=(outcome.digest,),
    )


def test_runtime_rejects_claimed_subject_identity_substitution():
    repository = _repository()

    errors = _replay(
        "RepositoryTestReceipt",
        repository.schema_version,
        repository,
        "repository_tests",
        ("tests/osc/test_fake.py::test_fake",),
    )

    assert errors == (
        "runtime_replay_subject_mismatch:RepositoryTestReceipt:repository_tests",
    )


def test_runtime_reconstructs_nested_enums_and_dataclasses():
    task = _task()
    evidence = _evidence()

    assert _replay(
        "TaskSpec", task.schema_version, task, "task_specs", (task.digest,)
    ) == ()
    assert _replay(
        "EvidenceEnvelope",
        evidence.schema_version,
        evidence,
        "evidence_envelopes",
        (evidence.digest,),
    ) == ()


def test_evidence_metadata_canonical_immutable_roundtrip():
    evidence = _evidence()
    payload = to_primitive(evidence)

    assert payload["metadata"][0][0] == "bytes"
    assert _replay(
        "EvidenceEnvelope",
        evidence.schema_version,
        evidence,
        "evidence_envelopes",
        (evidence.digest,),
    ) == ()


def test_invariance_report_returns_exact_thirteen_subjects():
    entry = next(item for item in _owned_values() if item[0] == "InvarianceReport")

    assert len(entry[4]) == 13
    assert _replay(*entry) == ()
    assert _replay(*entry[:-1], entry[4][:-1])


@pytest.mark.parametrize("subject_kind", ["candidate_records", "confirmed_bugs"])
def test_outcome_and_evidence_cannot_claim_candidate_or_bug_subjects(subject_kind):
    for type_name, value in (
        ("StructuredExecutionOutcome", _outcome()),
        ("EvidenceEnvelope", _evidence()),
    ):
        errors = _replay(
            type_name,
            value.schema_version,
            value,
            subject_kind,
            ("forged",),
        )
        assert errors


def test_task_and_seed_payloads_cannot_claim_v3_case_or_backend_credit():
    task = _task()
    lineage = _lineage()

    assert _replay(
        "TaskSpec", task.schema_version, task, "v3_backend_tasks", (task.digest,)
    )
    assert _replay(
        "SeedLineage",
        lineage.schema_version,
        lineage,
        "v3_cases",
        (lineage.digest,),
    )


def test_private_receipt_canonical_roundtrip_is_byte_stable():
    for type_name, version, value, kind, identities in _owned_values()[7:]:
        encoded = canonical_envelope(type_name, version, value)
        decoded = decode_canonical_envelope(encoded)

        assert canonical_roundtrip(encoded) == encoded
        assert replay_runtime_admission(
            envelope_type=decoded["type"],
            schema_version=decoded["schema_version"],
            payload=decoded["payload"],
            subject_kind=kind,
            subject_ids=identities,
        ) == ()


def test_private_receipts_derive_every_declared_runtime_subject_family():
    v3 = _v3()
    candidate = _candidate()
    performance = _contract_performance()
    claims = (
        ("V3RunReceipt", v3, "v3_cases", v3.case_ids),
        (
            "V3RunReceipt",
            v3,
            "v3_executed_cases",
            v3.executed_case_ids,
        ),
        ("V3RunReceipt", v3, "v3_backend_tasks", v3.backend_task_ids),
        (
            "CandidateClassificationReceipt",
            candidate,
            "classified_v3_cases",
            (candidate.source_case_id,),
        ),
        (
            "CandidateClassificationReceipt",
            candidate,
            "campaign_rechecks",
            candidate.campaign_recheck_ids,
        ),
        (
            "CandidateClassificationReceipt",
            candidate,
            "pipeline_rechecks",
            candidate.pipeline_recheck_ids,
        ),
        (
            "ContractPerformanceEvidenceReceipt",
            performance,
            "contract_performance_samples",
            (performance.sample_id,),
        ),
    )

    for type_name, value, kind, identities in claims:
        assert _replay(type_name, value.schema_version, value, kind, identities) == ()


def test_zero_candidate_cannot_create_candidate_or_recheck_subjects():
    candidate = _candidate()
    zero = CandidateClassificationReceipt(
        source_digest=candidate.source_digest,
        v3_gate_plan_digest=candidate.v3_gate_plan_digest,
        classification_policy_digest=candidate.classification_policy_digest,
        classification_id="classification-zero",
        source_case_id="case-zero",
        result_group_digest=candidate.result_group_digest,
        evidence_envelope_digest=candidate.evidence_envelope_digest,
        evidence_state=EvidenceState.NOT_A_CANDIDATE,
        finding_record_id="",
        candidate_id="",
        classification_digest=_digest("classification-zero"),
        campaign_rechecks=(),
        pipeline_rechecks=(),
        bug_claimed=False,
    )

    assert _replay(
        "CandidateClassificationReceipt",
        zero.schema_version,
        zero,
        "classified_v3_cases",
        (zero.source_case_id,),
    ) == ()
    for kind in ("candidate_records", "campaign_rechecks", "pipeline_rechecks"):
        assert _replay(
            "CandidateClassificationReceipt",
            zero.schema_version,
            zero,
            kind,
            (),
        )


def test_semantic_replay_never_infers_real_producer_provenance():
    assert RUNTIME_REPLAY_REQUIRES_ROOT_PROVENANCE is True
