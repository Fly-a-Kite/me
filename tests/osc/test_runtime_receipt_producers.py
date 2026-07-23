from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path

import pytest

from datadiff_osc._canonical import canonical_json
from datadiff_osc.runtime._receipt_producers import (
    CandidateClassificationObservation,
    CandidateRecheckObservation,
    ContractPerformanceObservation,
    PairedBenchmarkObservation,
    ParallelScalingObservation,
    RawCounterMeasurement,
    RawMonotonicInterval,
    RepositoryNodeObservation,
    RepositoryTestObservation,
    TargetPackageObservation,
    TargetVersionObservation,
    TypedCaseObservation,
    TypedExecutionResult,
    WorkerBoundExecutionResult,
    build_candidate_classification_receipt,
    build_paired_benchmark_receipt,
    build_repository_test_receipt,
    build_target_version_receipt,
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


PUBLIC_FREEZE_SHA256 = (
    "cb505dbe665e517e53314ddae8ba9302d05ac794d326e9be1e792f8bd4e22131"
)


def _fields(**values: str) -> tuple[tuple[str, str], ...]:
    return tuple(sorted(values.items()))


def _lineage(*, counter: int, case_index: int = 1) -> SeedLineage:
    return SeedLineage(
        protocol_digest="protocol-digest",
        master_seed=31_000_001,
        lane_id="formal-lane",
        case_index=case_index,
        stage_name=SeedStage.ORACLE_SAMPLE,
        counter=counter,
    )


def _task(
    *,
    lineage: SeedLineage,
    decision_index: int,
    endpoint_id: str = "endpoint-a",
) -> TaskSpec:
    return TaskSpec(
        identity=TaskIdentity(
            protocol_digest=lineage.protocol_digest,
            task_kind=TaskKind.BACKEND_EXECUTION,
            epoch_index=0,
            decision_index=decision_index,
            seed_lineage_digest=lineage.digest,
            endpoint_id=endpoint_id,
            backend="backend-a",
            attempt=decision_index,
        ),
        dependency_task_ids=(),
        resources=ResourceTokens(
            cpu_tokens=1,
            rss_bytes=1024,
            io_class="bounded",
            backend_internal_threads=1,
        ),
        payload_digest=f"payload-{decision_index}-{endpoint_id}",
    )


def _outcome(
    endpoint_id: str = "endpoint-a",
    *,
    status: ExecutionStatus = ExecutionStatus.OK,
) -> StructuredExecutionOutcome:
    failure = {
        ExecutionStatus.OK: FailureKind.NONE,
        ExecutionStatus.SEMANTIC_ERROR: FailureKind.SEMANTIC_DOMAIN_ERROR,
        ExecutionStatus.UNSUPPORTED: FailureKind.UNSUPPORTED_CAPABILITY,
        ExecutionStatus.TIMEOUT: FailureKind.TIMEOUT,
        ExecutionStatus.CRASH: FailureKind.CRASH,
        ExecutionStatus.ADAPTER_ERROR: FailureKind.ADAPTER_ERROR,
        ExecutionStatus.MISSING: FailureKind.MISSING_RESULT,
    }[status]
    return StructuredExecutionOutcome(
        endpoint_id=endpoint_id,
        status=status,
        failure_kind=failure,
        unsupported_evidence_digest=("capability-evidence" if status is ExecutionStatus.UNSUPPORTED else ""),
    )


def _group(
    tasks: tuple[TaskSpec, ...],
    *,
    group_id: str = "result-group-001",
    endpoints: tuple[str, ...] = ("endpoint-a",),
) -> ResultGroup:
    return ResultGroup(
        result_group_id=group_id,
        task_ids=tuple(item.identity.task_id for item in tasks),
        endpoint_ids=endpoints,
        contract_fingerprint=ContractFingerprint(
            contract_digest="contract-digest",
            registry_digest="registry-digest",
        ),
        target_fingerprint=None,
        evidence_tier=EvidenceTier.FINDING,
        result_order_key=endpoints,
    )


def _result(
    *,
    task: TaskSpec,
    lineage: SeedLineage,
    group: ResultGroup,
    case_id: str = "case-001",
    outcome: StructuredExecutionOutcome | None = None,
) -> TypedExecutionResult:
    return TypedExecutionResult(
        case_id=case_id,
        result_group=group,
        task=task,
        seed_lineage=lineage,
        outcome=outcome or _outcome(task.identity.endpoint_id),
    )


def _evidence(
    result: TypedExecutionResult,
    *,
    state: EvidenceState,
    suffix: str,
) -> EvidenceEnvelope:
    return EvidenceEnvelope(
        evidence_id=f"evidence-{suffix}",
        state=state,
        result_group_digest=result.result_group.digest,
        task_id=result.task.identity.task_id,
        seed_lineage_digest=result.seed_lineage.digest,
        contract_fingerprint=result.result_group.contract_fingerprint,
        target_fingerprint=result.result_group.target_fingerprint,
        derivation_certificate_digest=f"derivation-{suffix}",
        applicability_certificate_digest=f"applicability-{suffix}",
        activation_certificate_digest=f"activation-{suffix}",
        observation_certificate_digest=f"observation-{suffix}",
        execution_outcomes=(result.outcome,),
        verdict_kind=VerdictKind.VIOLATED,
    )


def _typed_case() -> TypedCaseObservation:
    payload = {
        "case_id": "case-001",
        "seed": 17,
        "tables": [
            {
                "name": "t",
                "columns": [{"name": "x", "nullable": False, "type": "int"}],
                "rows": [{"x": 1}],
            }
        ],
        "program": {"operations": [], "program_id": "program-001", "seed": 17},
    }
    return TypedCaseObservation(
        case_id="case-001",
        case_json=canonical_json(payload).encode("utf-8"),
    )


def _repository_observation() -> RepositoryTestObservation:
    return RepositoryTestObservation(
        source_digest="source-digest",
        test_plan=("collect", "execute"),
        config=_fields(pytest="frozen", plugins="none"),
        environment=_fields(PYTHONHASHSEED="0", TZ="UTC"),
        command=("python", "-m", "pytest", "-q"),
        junit_xml=(
            b'<testsuite tests="2" failures="0" errors="0" skipped="1">'
            b'<testcase classname="tests.a" name="test_a" time="0.1" />'
            b'<testcase classname="tests.b" name="test_b" time="0.0">'
            b'<skipped message="not selected" /></testcase></testsuite>'
        ),
        log=b"1 passed, 1 skipped in 0.10s\n",
        exit_code=0,
    )


def test_repository_builder_strictly_derives_collection_and_outcomes():
    observation = _repository_observation()
    receipt = build_repository_test_receipt(observation)

    assert receipt.node_ids == ("tests.a::test_a", "tests.b::test_b")
    assert tuple(item.outcome for item in receipt.nodes) == ("passed", "skipped")
    assert receipt.junit_sha256 == hashlib.sha256(observation.junit_xml).hexdigest()
    assert receipt.log_sha256 == hashlib.sha256(observation.log).hexdigest()
    assert receipt.failed_node_ids == ()


def test_repository_failing_junit_log_or_exit_cannot_be_spoofed_as_pass():
    failing = replace(
        _repository_observation(),
        junit_xml=(
            b'<testsuite tests="2" failures="1" errors="0" skipped="0">'
            b'<testcase classname="tests.a" name="test_a" time="0.1" />'
            b'<testcase classname="tests.b" name="test_b" time="0.0">'
            b'<failure message="boom">trace</failure></testcase></testsuite>'
        ),
        log=b"1 failed, 1 passed in 0.10s\n",
        exit_code=1,
    )
    receipt = build_repository_test_receipt(failing)
    assert receipt.failed_node_ids == ("tests.b::test_b",)
    assert receipt.exit_code == 1

    with pytest.raises(ValueError, match="log summary conflicts"):
        build_repository_test_receipt(replace(failing, log=b"2 passed in 0.10s\n"))
    with pytest.raises(ValueError, match="exit code conflicts"):
        build_repository_test_receipt(replace(failing, exit_code=0))
    with pytest.raises(ValueError, match="failure marker conflicts"):
        build_repository_test_receipt(
            replace(
                _repository_observation(),
                log=b"FAILED tests.a::test_a\n1 passed, 1 skipped in 0.10s\n",
            )
        )


def test_repository_caller_summary_malformed_and_duplicate_junit_reject():
    observation = _repository_observation()
    forged_node = RepositoryNodeObservation(
        node_id="tests.a::test_a",
        outcome="passed",
        result_fields=_fields(time="forged"),
    )
    with pytest.raises(ValueError, match="caller.*conflicts"):
        build_repository_test_receipt(replace(observation, nodes=(forged_node,)))
    with pytest.raises(ValueError, match="DTD"):
        build_repository_test_receipt(
            replace(
                observation,
                junit_xml=b'<!DOCTYPE testsuite><testsuite><testcase classname="x" name="y" /></testsuite>',
                log=b"1 passed\n",
            )
        )
    with pytest.raises(ValueError, match="duplicate testcase"):
        build_repository_test_receipt(
            replace(
                observation,
                junit_xml=(
                    b'<testsuite tests="2"><testcase classname="x" name="y" />'
                    b'<testcase classname="x" name="y" /></testsuite>'
                ),
                log=b"2 passed\n",
            )
        )
    with pytest.raises(ValueError, match="forbidden child"):
        build_repository_test_receipt(
            replace(
                observation,
                junit_xml=(
                    b'<testsuite><failure message="suite-level spoof" />'
                    b'<testcase classname="x" name="y" /></testsuite>'
                ),
                log=b"1 passed\n",
            )
        )


def _target_observation() -> TargetVersionObservation:
    return TargetVersionObservation(
        source_digest="source-digest",
        audit_plan=_fields(package_set="six", policy="exact"),
        environment=_fields(venv="phase6"),
        python_runtime=_fields(implementation="cpython", version="3.12"),
        packages=(
            TargetPackageObservation(
                distribution_name="pandas",
                import_name="pandas",
                installed_version="3.0.3",
                latest_version="3.0.3",
                installed_metadata=(
                    b"Metadata-Version: 2.4\nName: pandas\nVersion: 3.0.3\n\n"
                ),
                version_source_kind="pypi_json",
                version_source=b'{"info":{"name":"pandas","version":"3.0.3"}}',
            ),
        ),
    )


def test_target_builder_derives_names_versions_and_source_hashes_from_raw():
    observation = _target_observation()
    receipt = build_target_version_receipt(observation)
    package = receipt.packages[0]

    assert package.distribution_name == "pandas"
    assert package.installed_version == package.latest_version == "3.0.3"
    assert package.package_id.startswith("osc-root-target-package-id-")
    assert package.version_source_sha256 == hashlib.sha256(
        observation.packages[0].version_source
    ).hexdigest()
    assert receipt.all_match_latest


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        ("distribution_name", "forged", "distribution name conflicts"),
        ("installed_version", "0", "installed version conflicts"),
        ("latest_version", "999", "latest version conflicts"),
    ],
)
def test_target_raw_scalar_conflicts_reject(field_name: str, value: str, message: str):
    observation = _target_observation()
    package = replace(observation.packages[0], **{field_name: value})
    with pytest.raises(ValueError, match=message):
        build_target_version_receipt(replace(observation, packages=(package,)))


def test_target_malformed_or_duplicate_raw_metadata_rejects():
    observation = _target_observation()
    duplicate_metadata = replace(
        observation.packages[0],
        installed_metadata=(
            b"Metadata-Version: 2.4\nName: pandas\nName: forged\nVersion: 3.0.3\n\n"
        ),
    )
    with pytest.raises(ValueError, match="exactly one name"):
        build_target_version_receipt(replace(observation, packages=(duplicate_metadata,)))
    duplicate_json = replace(
        observation.packages[0],
        version_source=(
            b'{"info":{"name":"pandas","version":"3.0.3","version":"9"}}'
        ),
    )
    with pytest.raises(ValueError, match="duplicate JSON key"):
        build_target_version_receipt(replace(observation, packages=(duplicate_json,)))
    nonstandard_json = replace(
        observation.packages[0],
        version_source=(
            b'{"extra":NaN,"info":{"name":"pandas","version":"3.0.3"}}'
        ),
    )
    with pytest.raises(ValueError, match="non-standard JSON constant"):
        build_target_version_receipt(
            replace(observation, packages=(nonstandard_json,))
        )
    with pytest.raises(ValueError, match="not authority eligible"):
        replace(observation.packages[0], version_source_kind="offline_override")


def _candidate_observation(
    *, lineage_counters: tuple[int, ...] = tuple(range(100, 107))
) -> CandidateClassificationObservation:
    if len(lineage_counters) != 7:
        raise ValueError("candidate fixture requires exactly seven lineages")
    lineages = tuple(_lineage(counter=index) for index in lineage_counters)
    tasks = tuple(
        _task(lineage=lineage, decision_index=100 + index)
        for index, lineage in enumerate(lineages)
    )
    group = _group(tasks, group_id="candidate-result-group")
    results = tuple(
        _result(task=task, lineage=lineage, group=group)
        for task, lineage in zip(tasks, lineages)
    )
    evidences = tuple(
        _evidence(result, state=EvidenceState.STABLE_SURVIVOR, suffix=str(index))
        for index, result in enumerate(results)
    )
    campaign = tuple(
        CandidateRecheckObservation(
            ordinal=ordinal,
            result=results[ordinal],
            evidence=evidences[ordinal],
        )
        for ordinal in (1, 2, 3)
    )
    pipeline = tuple(
        CandidateRecheckObservation(
            ordinal=ordinal,
            result=results[ordinal + 3],
            evidence=evidences[ordinal + 3],
        )
        for ordinal in (1, 2, 3)
    )
    return CandidateClassificationObservation(
        source_digest="source-digest",
        v3_gate_plan_digest="v3-plan-digest",
        classification_policy=_fields(policy="candidate-only", version="2"),
        source_case=_typed_case(),
        source_result=results[0],
        source_evidence=evidences[0],
        evidence_state=EvidenceState.STABLE_SURVIVOR,
        classification_facts=_fields(family="family-a", verdict="candidate"),
        campaign_rechecks=campaign,
        pipeline_rechecks=pipeline,
        bug_claimed=False,
    )


def test_candidate_valid_disjoint_typed_3_plus_3_round_trips():
    observation = _candidate_observation()
    receipt = build_candidate_classification_receipt(observation)

    assert receipt.source_case_id == "case-001"
    assert receipt.result_group_digest == observation.source_result.result_group.digest
    assert receipt.evidence_envelope_digest == observation.source_evidence.digest
    assert tuple(item.ordinal for item in receipt.campaign_rechecks) == (1, 2, 3)
    assert tuple(item.ordinal for item in receipt.pipeline_rechecks) == (1, 2, 3)
    assert all(item.completed and item.reproduced for item in receipt.campaign_rechecks)
    assert all(item.completed and item.reproduced for item in receipt.pipeline_rechecks)
    assert receipt.evidence_state is EvidenceState.STABLE_SURVIVOR
    assert receipt.bug_claimed is False


@pytest.mark.parametrize("stage", ("campaign_rechecks", "pipeline_rechecks"))
def test_candidate_source_execution_and_evidence_reuse_rejects(stage: str):
    observation = _candidate_observation()
    rechecks = getattr(observation, stage)
    reused = replace(
        rechecks[0],
        result=observation.source_result,
        evidence=observation.source_evidence,
    )

    with pytest.raises(
        ValueError,
        match="source/campaign/pipeline identities must be globally disjoint",
    ) as exc_info:
        build_candidate_classification_receipt(
            replace(observation, **{stage: (reused, *rechecks[1:])})
        )

    message = str(exc_info.value)
    for identity_label in (
        "task IDs",
        "TaskSpec digests",
        "result IDs",
        "seed lineages",
        "evidence envelope digests",
    ):
        assert identity_label in message


@pytest.mark.parametrize("stage", ("campaign_rechecks", "pipeline_rechecks"))
def test_candidate_source_collision_rejects_before_incomplete_verdict(stage: str):
    observation = _candidate_observation()
    rechecks = getattr(observation, stage)
    collided = replace(
        rechecks[0],
        result=observation.source_result,
        evidence=replace(
            observation.source_evidence,
            verdict_kind=VerdictKind.INCONCLUSIVE,
        ),
    )

    with pytest.raises(
        ValueError,
        match="source/campaign/pipeline identities must be globally disjoint",
    ):
        build_candidate_classification_receipt(
            replace(observation, **{stage: (collided, *rechecks[1:])})
        )


@pytest.mark.parametrize(
    ("source_verdict", "stage"),
    (
        (VerdictKind.INCONCLUSIVE, "campaign_rechecks"),
        (VerdictKind.INCONCLUSIVE, "pipeline_rechecks"),
        (VerdictKind.SATISFIED, "campaign_rechecks"),
        (VerdictKind.SATISFIED, "pipeline_rechecks"),
    ),
)
def test_candidate_source_reuse_disjointness_precedes_nonviolated_source_verdict(
    source_verdict: VerdictKind,
    stage: str,
):
    observation = _candidate_observation()
    source_evidence = replace(
        observation.source_evidence,
        verdict_kind=source_verdict,
    )
    rechecks = getattr(observation, stage)
    reused = replace(
        rechecks[0],
        result=observation.source_result,
        evidence=source_evidence,
    )

    with pytest.raises(
        ValueError,
        match="source/campaign/pipeline identities must be globally disjoint",
    ) as exc_info:
        build_candidate_classification_receipt(
            replace(
                observation,
                source_evidence=source_evidence,
                **{stage: (reused, *rechecks[1:])},
            )
        )

    assert "candidate source lacks a typed VIOLATED witness" not in str(exc_info.value)


@pytest.mark.parametrize(
    ("source_verdict", "stage"),
    (
        (VerdictKind.INCONCLUSIVE, "campaign_rechecks"),
        (VerdictKind.INCONCLUSIVE, "pipeline_rechecks"),
        (VerdictKind.SATISFIED, "campaign_rechecks"),
        (VerdictKind.SATISFIED, "pipeline_rechecks"),
    ),
)
def test_candidate_exact_recheck_task_binding_precedes_nonviolated_source_verdict(
    source_verdict: VerdictKind,
    stage: str,
):
    observation = _candidate_observation()
    source_evidence = replace(
        observation.source_evidence,
        verdict_kind=source_verdict,
    )
    rechecks = getattr(observation, stage)
    wrong_evidence = replace(
        rechecks[0].evidence,
        evidence_id=f"evidence-unique-wrong-task-{source_verdict.name}-{stage}",
        task_id=f"wrong-task-{source_verdict.name}-{stage}",
    )
    wrong_recheck = replace(rechecks[0], evidence=wrong_evidence)
    candidate = replace(
        observation,
        source_evidence=source_evidence,
        **{stage: (wrong_recheck, *rechecks[1:])},
    )
    execution_evidence = (
        (candidate.source_result, candidate.source_evidence),
        *((item.result, item.evidence) for item in candidate.campaign_rechecks),
        *((item.result, item.evidence) for item in candidate.pipeline_rechecks),
    )
    assert len(execution_evidence) == 7
    assert len({result.task.identity.task_id for result, _ in execution_evidence}) == 7
    assert len({result.task.digest for result, _ in execution_evidence}) == 7
    assert len({result.result_id for result, _ in execution_evidence}) == 7
    assert len({result.seed_lineage.digest for result, _ in execution_evidence}) == 7
    assert len({evidence.digest for _, evidence in execution_evidence}) == 7

    with pytest.raises(ValueError, match="candidate evidence/task mismatch") as exc_info:
        build_candidate_classification_receipt(candidate)

    assert "candidate source lacks a typed VIOLATED witness" not in str(exc_info.value)


@pytest.mark.parametrize("stage", ("campaign_rechecks", "pipeline_rechecks"))
@pytest.mark.parametrize(
    ("binding", "message"),
    (
        ("result_group", "candidate evidence/result-group mismatch"),
        ("seed", "candidate evidence/seed mismatch"),
        ("outcome", "candidate evidence/outcome mismatch"),
    ),
)
def test_candidate_exact_recheck_binding_variants_precede_source_verdict(
    stage: str,
    binding: str,
    message: str,
):
    observation = _candidate_observation()
    rechecks = getattr(observation, stage)
    source_evidence = replace(
        observation.source_evidence,
        verdict_kind=VerdictKind.INCONCLUSIVE,
    )
    evidence = rechecks[0].evidence
    if binding == "result_group":
        evidence = replace(
            evidence,
            evidence_id=f"evidence-unique-wrong-result-group-{stage}",
            result_group_digest=f"wrong-result-group-{stage}",
        )
    elif binding == "seed":
        evidence = replace(
            evidence,
            evidence_id=f"evidence-unique-wrong-seed-{stage}",
            seed_lineage_digest=f"wrong-seed-{stage}",
        )
    else:
        evidence = replace(
            evidence,
            evidence_id=f"evidence-unique-wrong-outcome-{stage}",
            execution_outcomes=(
                replace(
                    rechecks[0].result.outcome,
                    reason=f"wrong-outcome-{stage}",
                ),
            ),
        )
    candidate = replace(
        observation,
        source_evidence=source_evidence,
        **{stage: (replace(rechecks[0], evidence=evidence), *rechecks[1:])},
    )

    with pytest.raises(ValueError, match=message) as exc_info:
        build_candidate_classification_receipt(candidate)

    assert "candidate source lacks a typed VIOLATED witness" not in str(exc_info.value)


@pytest.mark.parametrize(
    "source_verdict", (VerdictKind.INCONCLUSIVE, VerdictKind.SATISFIED)
)
@pytest.mark.parametrize("stage", ("campaign_rechecks", "pipeline_rechecks"))
@pytest.mark.parametrize(
    ("context", "expected_suffix"),
    (
        ("case", "recheck case mismatch"),
        ("result_group", "recheck result-group mismatch"),
        ("outcome", "recheck requires an OK outcome"),
        ("ordinal", "recheck ordinals must be exactly 1, 2, 3"),
    ),
)
def test_candidate_exact_recheck_typed_context_precedes_nonviolated_source_verdict(
    source_verdict: VerdictKind,
    stage: str,
    context: str,
    expected_suffix: str,
):
    observation = _candidate_observation()
    source_evidence = replace(
        observation.source_evidence,
        verdict_kind=source_verdict,
    )
    rechecks = getattr(observation, stage)
    first = rechecks[0]
    stage_name = stage.removesuffix("_rechecks")
    if context == "case":
        changed = replace(
            first,
            result=replace(
                first.result,
                case_id=f"wrong-case-{source_verdict.name}-{stage}",
            ),
        )
    elif context == "result_group":
        result = replace(
            first.result,
            result_group=replace(
                first.result.result_group,
                result_group_id=f"wrong-result-group-{source_verdict.name}-{stage}",
            ),
        )
        changed = replace(
            first,
            result=result,
            evidence=replace(
                first.evidence,
                evidence_id=f"evidence-unique-context-group-{source_verdict.name}-{stage}",
                result_group_digest=result.result_group.digest,
            ),
        )
    elif context == "outcome":
        outcome = _outcome(status=ExecutionStatus.SEMANTIC_ERROR)
        changed = replace(
            first,
            result=replace(first.result, outcome=outcome),
            evidence=replace(
                first.evidence,
                evidence_id=f"evidence-unique-context-outcome-{source_verdict.name}-{stage}",
                execution_outcomes=(outcome,),
            ),
        )
    else:
        changed = replace(first, ordinal=4)
    candidate = replace(
        observation,
        source_evidence=source_evidence,
        **{stage: (changed, *rechecks[1:])},
    )
    execution_evidence = (
        (candidate.source_result, candidate.source_evidence),
        *((item.result, item.evidence) for item in candidate.campaign_rechecks),
        *((item.result, item.evidence) for item in candidate.pipeline_rechecks),
    )
    assert len(execution_evidence) == 7
    assert len({result.task.identity.task_id for result, _ in execution_evidence}) == 7
    assert len({result.task.digest for result, _ in execution_evidence}) == 7
    assert len({result.result_id for result, _ in execution_evidence}) == 7
    assert len({result.seed_lineage.digest for result, _ in execution_evidence}) == 7
    assert len({evidence.digest for _, evidence in execution_evidence}) == 7

    with pytest.raises(
        ValueError, match=rf"{stage_name} {expected_suffix}"
    ) as exc_info:
        build_candidate_classification_receipt(candidate)

    assert "candidate source lacks a typed VIOLATED witness" not in str(exc_info.value)


@pytest.mark.parametrize(
    ("context", "message"),
    (
        ("case", "candidate source case/result mismatch"),
        ("outcome", "candidate source result requires an OK outcome"),
    ),
)
def test_candidate_source_typed_context_precedes_nonviolated_source_verdict(
    context: str, message: str
):
    observation = _candidate_observation()
    source_evidence = replace(
        observation.source_evidence,
        verdict_kind=VerdictKind.INCONCLUSIVE,
    )
    if context == "case":
        source_result = replace(observation.source_result, case_id="wrong-source-case")
    else:
        outcome = _outcome(status=ExecutionStatus.SEMANTIC_ERROR)
        source_result = replace(observation.source_result, outcome=outcome)
        source_evidence = replace(source_evidence, execution_outcomes=(outcome,))
    candidate = replace(
        observation,
        source_result=source_result,
        source_evidence=source_evidence,
    )

    with pytest.raises(ValueError, match=message) as exc_info:
        build_candidate_classification_receipt(candidate)

    assert "candidate source lacks a typed VIOLATED witness" not in str(exc_info.value)


@pytest.mark.parametrize(
    "source_verdict", (VerdictKind.INCONCLUSIVE, VerdictKind.SATISFIED)
)
def test_candidate_canonical_typed_case_conflict_precedes_nonviolated_source_verdict(
    source_verdict: VerdictKind,
):
    observation = _candidate_observation()
    candidate = replace(
        observation,
        source_case=replace(
            observation.source_case,
            case_id=f"forged-case-{source_verdict.name}",
        ),
        source_evidence=replace(
            observation.source_evidence,
            verdict_kind=source_verdict,
        ),
    )

    with pytest.raises(ValueError, match="caller case ID conflicts") as exc_info:
        build_candidate_classification_receipt(candidate)

    assert "candidate source case/result mismatch" not in str(exc_info.value)
    assert "candidate source lacks a typed VIOLATED witness" not in str(exc_info.value)


@pytest.mark.parametrize(
    ("collision", "identity_label"),
    (
        ("task_id", "task IDs"),
        ("task_spec", "TaskSpec digests"),
        ("result_id", "result IDs"),
        ("seed_lineage", "seed lineages"),
        ("evidence_digest", "evidence envelope digests"),
    ),
)
def test_candidate_each_source_identity_collision_rejects(
    collision: str, identity_label: str
):
    observation = _candidate_observation(
        lineage_counters=(100, 100, 102, 103, 104, 105, 106)
        if collision == "seed_lineage"
        else tuple(range(100, 107))
    )
    first = observation.campaign_rechecks[0]
    source = observation.source_result

    if collision == "task_id":
        result = replace(
            source,
            task=replace(source.task, payload_digest="task-id-collision-payload"),
            outcome=replace(source.outcome, reason="task-id-collision"),
        )
        evidence = _evidence(
            result,
            state=observation.evidence_state,
            suffix="task-id-collision",
        )
    elif collision == "task_spec":
        result = replace(
            source,
            outcome=replace(source.outcome, reason="task-spec-collision"),
        )
        evidence = _evidence(
            result,
            state=observation.evidence_state,
            suffix="task-spec-collision",
        )
    elif collision == "result_id":
        result = source
        evidence = _evidence(
            result,
            state=observation.evidence_state,
            suffix="result-id-collision",
        )
    elif collision == "seed_lineage":
        result = first.result
        evidence = first.evidence
    else:
        result = replace(
            source,
            task=replace(source.task, payload_digest="evidence-collision-payload"),
        )
        evidence = observation.source_evidence

    collided = replace(first, result=result, evidence=evidence)
    with pytest.raises(ValueError, match="globally disjoint") as exc_info:
        build_candidate_classification_receipt(
            replace(
                observation,
                campaign_rechecks=(collided, *observation.campaign_rechecks[1:]),
            )
        )
    assert identity_label in str(exc_info.value)


def test_stable_survivor_zero_or_insufficient_reproduction_rejects():
    observation = _candidate_observation()
    zero_campaign = tuple(
        replace(
            item,
            evidence=replace(item.evidence, verdict_kind=VerdictKind.SATISFIED),
        )
        for item in observation.campaign_rechecks
    )
    with pytest.raises(ValueError, match="campaign reproduction threshold"):
        build_candidate_classification_receipt(
            replace(observation, campaign_rechecks=zero_campaign)
        )
    insufficient_pipeline = (
        *observation.pipeline_rechecks[:2],
        replace(
            observation.pipeline_rechecks[2],
            evidence=replace(
                observation.pipeline_rechecks[2].evidence,
                verdict_kind=VerdictKind.SATISFIED,
            ),
        ),
    )
    with pytest.raises(ValueError, match="pipeline reproduction threshold"):
        build_candidate_classification_receipt(
            replace(observation, pipeline_rechecks=insufficient_pipeline)
        )


def test_reproduction_is_derived_from_exact_typed_verdicts_without_caller_bool():
    observation = _candidate_observation()
    first = observation.campaign_rechecks[0]
    with pytest.raises(TypeError):
        CandidateRecheckObservation(
            ordinal=first.ordinal,
            result=first.result,
            evidence=first.evidence,
            reproduced=True,  # type: ignore[call-arg]
        )

    def with_verdict(item, verdict):
        return replace(item, evidence=replace(item.evidence, verdict_kind=verdict))

    all_satisfied = replace(
        observation,
        campaign_rechecks=tuple(
            with_verdict(item, VerdictKind.SATISFIED)
            for item in observation.campaign_rechecks
        ),
        pipeline_rechecks=tuple(
            with_verdict(item, VerdictKind.SATISFIED)
            for item in observation.pipeline_rechecks
        ),
    )
    with pytest.raises(ValueError, match="campaign reproduction threshold"):
        build_candidate_classification_receipt(all_satisfied)

    incomplete = replace(
        observation,
        campaign_rechecks=(
            with_verdict(first, VerdictKind.INCONCLUSIVE),
            *observation.campaign_rechecks[1:],
        ),
    )
    with pytest.raises(ValueError, match="reproduction context is incomplete"):
        build_candidate_classification_receipt(incomplete)

    source_satisfied = replace(
        observation,
        source_evidence=replace(
            observation.source_evidence, verdict_kind=VerdictKind.SATISFIED
        ),
    )
    with pytest.raises(ValueError, match="source lacks a typed VIOLATED witness"):
        build_candidate_classification_receipt(source_satisfied)


def test_candidate_cross_stage_task_result_reuse_and_context_swaps_reject():
    observation = _candidate_observation()
    reused = replace(
        observation.pipeline_rechecks[0],
        result=observation.campaign_rechecks[0].result,
        evidence=observation.campaign_rechecks[0].evidence,
    )
    with pytest.raises(ValueError, match="globally disjoint"):
        build_candidate_classification_receipt(
            replace(
                observation,
                pipeline_rechecks=(reused, *observation.pipeline_rechecks[1:]),
            )
        )

    wrong_case_result = replace(
        observation.campaign_rechecks[0].result, case_id="other-case"
    )
    with pytest.raises(ValueError, match="case mismatch"):
        build_candidate_classification_receipt(
            replace(
                observation,
                campaign_rechecks=(
                    replace(observation.campaign_rechecks[0], result=wrong_case_result),
                    *observation.campaign_rechecks[1:],
                ),
            )
        )

    swapped_evidence = replace(
        observation.campaign_rechecks[0],
        evidence=replace(
            observation.campaign_rechecks[1].evidence,
            evidence_id="evidence-unique-wrong-task-binding",
        ),
    )
    with pytest.raises(ValueError, match="evidence/task mismatch"):
        build_candidate_classification_receipt(
            replace(
                observation,
                campaign_rechecks=(swapped_evidence, *observation.campaign_rechecks[1:]),
            )
        )


@pytest.mark.parametrize(
    "state", [EvidenceState.UNIQUE_ROOT, EvidenceState.INDEPENDENTLY_CONFIRMED]
)
def test_candidate_forbids_confirmed_states_and_bug_claims(state: EvidenceState):
    observation = _candidate_observation()
    with pytest.raises(ValueError, match="confirmed evidence states"):
        build_candidate_classification_receipt(replace(observation, evidence_state=state))
    with pytest.raises(ValueError, match="bug_claimed"):
        build_candidate_classification_receipt(replace(observation, bug_claimed=True))


def test_candidate_non_ok_recheck_and_wrong_typed_case_reject():
    observation = _candidate_observation()
    bad_outcome = _outcome(status=ExecutionStatus.SEMANTIC_ERROR)
    bad_result = replace(observation.campaign_rechecks[0].result, outcome=bad_outcome)
    bad_evidence = replace(
        observation.campaign_rechecks[0].evidence,
        execution_outcomes=(bad_outcome,),
    )
    with pytest.raises(ValueError, match="requires an OK outcome"):
        build_candidate_classification_receipt(
            replace(
                observation,
                campaign_rechecks=(
                    replace(
                        observation.campaign_rechecks[0],
                        result=bad_result,
                        evidence=bad_evidence,
                    ),
                    *observation.campaign_rechecks[1:],
                ),
            )
        )
    with pytest.raises(ValueError, match="caller case ID conflicts"):
        build_candidate_classification_receipt(
            replace(
                observation,
                source_case=replace(observation.source_case, case_id="forged-case"),
            )
        )


def _paired_results(
    *,
    start_counter: int,
    group_id: str,
) -> tuple[TypedExecutionResult, TypedExecutionResult]:
    lineage = _lineage(counter=start_counter)
    baseline_task = _task(lineage=lineage, decision_index=start_counter)
    treatment_task = _task(lineage=lineage, decision_index=start_counter + 1)
    group = _group((baseline_task, treatment_task), group_id=group_id)
    return (
        _result(task=baseline_task, lineage=lineage, group=group),
        _result(task=treatment_task, lineage=lineage, group=group),
    )


def _parallel_results(
    *,
    start_counter: int,
    group_id: str,
) -> tuple[TypedExecutionResult, TypedExecutionResult]:
    lineage = _lineage(counter=start_counter)
    one_task = _task(lineage=lineage, decision_index=start_counter)
    six_task = replace(
        one_task,
        identity=replace(
            one_task.identity,
            decision_index=start_counter + 1,
            attempt=one_task.identity.attempt + 1,
        ),
    )
    group = _group((one_task, six_task), group_id=group_id)
    return (
        _result(task=one_task, lineage=lineage, group=group),
        _result(task=six_task, lineage=lineage, group=group),
    )


def _paired_observation() -> PairedBenchmarkObservation:
    baseline, treatment = _paired_results(
        start_counter=200, group_id="contract-pair-group"
    )
    one_worker, six_worker = _parallel_results(
        start_counter=300, group_id="parallel-pair-group"
    )
    return PairedBenchmarkObservation(
        source_digest="source-digest",
        benchmark_plan=_fields(samples="frozen", workers="1,6"),
        environment=_fields(PYTHONHASHSEED="0"),
        baseline_config=_fields(mode="baseline"),
        treatment_config=_fields(mode="treatment"),
        contract_samples=(
            ContractPerformanceObservation(
                ordinal=1,
                baseline_result=baseline,
                treatment_result=treatment,
                compile_match_interval=RawMonotonicInterval("clock-1", 0, 1_000_000),
                case_wall_interval=RawMonotonicInterval("clock-1", 0, 20_000_000),
                baseline_throughput=RawCounterMeasurement(
                    "cases",
                    0,
                    100,
                    RawMonotonicInterval("clock-1", 0, 1_000_000_000),
                ),
                treatment_throughput=RawCounterMeasurement(
                    "cases",
                    0,
                    100,
                    RawMonotonicInterval("clock-1", 0, 1_052_631_579),
                ),
            ),
        ),
        parallel_samples=(
            ParallelScalingObservation(
                ordinal=1,
                workload_id="workload-001",
                one_worker_results=(WorkerBoundExecutionResult(1, one_worker),),
                six_worker_results=(WorkerBoundExecutionResult(6, six_worker),),
                one_worker_measurement=RawCounterMeasurement(
                    "tasks",
                    0,
                    60,
                    RawMonotonicInterval("clock-1", 0, 60_000_000_000),
                ),
                six_worker_measurement=RawCounterMeasurement(
                    "tasks",
                    0,
                    60,
                    RawMonotonicInterval("clock-1", 0, 12_000_000_000),
                ),
            ),
        ),
    )


def test_paired_builder_derives_all_scalars_from_raw_monotonic_values():
    receipt = build_paired_benchmark_receipt(_paired_observation())

    contract = receipt.contract_samples[0]
    parallel = receipt.parallel_samples[0]
    assert contract.sample_id.startswith("osc-root-contract-performance-sample-id-0001-")
    assert parallel.sample_id.startswith("osc-root-parallel-scaling-sample-id-0001-")
    assert contract.compile_match_ms == pytest.approx(1.0)
    assert contract.case_wall_ms == pytest.approx(20.0)
    assert contract.baseline_throughput_cases_s == pytest.approx(100.0)
    assert contract.treatment_throughput_cases_s == pytest.approx(95.0)
    assert receipt.compile_wall_shares == pytest.approx((0.05,))
    assert receipt.throughput_regressions == pytest.approx((0.05,))
    assert receipt.six_worker_efficiencies == pytest.approx((5.0 / 6.0,))
    assert parallel.one_worker_task_set_digest == parallel.six_worker_task_set_digest
    assert parallel.one_worker_result_digest != parallel.six_worker_result_digest


def test_paired_seed_context_task_result_and_non_ok_swaps_reject():
    observation = _paired_observation()
    sample = observation.contract_samples[0]
    with pytest.raises(ValueError, match="tasks must be distinct"):
        replace(sample, treatment_result=sample.baseline_result)

    other_lineage = _lineage(counter=999)
    other_task = _task(lineage=other_lineage, decision_index=999)
    old_group = sample.treatment_result.result_group
    other_group = replace(
        old_group,
        task_ids=(sample.baseline_result.task.identity.task_id, other_task.identity.task_id),
    )
    other_result = _result(task=other_task, lineage=other_lineage, group=other_group)
    with pytest.raises(ValueError, match="result group mismatch|seed lineage mismatch"):
        replace(sample, treatment_result=other_result)

    bad_outcome = _outcome(status=ExecutionStatus.CRASH)
    bad_result = replace(sample.treatment_result, outcome=bad_outcome)
    with pytest.raises(ValueError, match="requires an OK outcome"):
        replace(sample, treatment_result=bad_result)


def test_paired_rejects_scalar_spoofing_invalid_raw_and_parallel_reuse():
    observation = _paired_observation()
    sample = observation.contract_samples[0]
    with pytest.raises(TypeError):
        ContractPerformanceObservation(
            ordinal=sample.ordinal,
            baseline_result=sample.baseline_result,
            treatment_result=sample.treatment_result,
            compile_match_interval=sample.compile_match_interval,
            case_wall_interval=sample.case_wall_interval,
            baseline_throughput=sample.baseline_throughput,
            treatment_throughput=sample.treatment_throughput,
            compile_match_ms=0.0,  # type: ignore[call-arg]
        )
    with pytest.raises(ValueError, match="positive progress"):
        RawCounterMeasurement(
            "cases", 5, 5, RawMonotonicInterval("clock-1", 0, 1)
        )
    with pytest.raises(ValueError, match="run backwards"):
        RawMonotonicInterval("clock-1", 2, 1)

    parallel = observation.parallel_samples[0]
    with pytest.raises(ValueError, match="globally distinct"):
        replace(
            parallel,
            six_worker_results=(
                WorkerBoundExecutionResult(
                    6, parallel.one_worker_results[0].result
                ),
            ),
        )


def _replace_parallel_six_task(
    sample: ParallelScalingObservation,
    six_task: TaskSpec,
    *,
    six_lineage: SeedLineage | None = None,
    six_case_id: str | None = None,
) -> ParallelScalingObservation:
    one = sample.one_worker_results[0].result
    six = sample.six_worker_results[0].result
    old_six_id = six.task.identity.task_id
    new_group = replace(
        six.result_group,
        task_ids=tuple(
            six_task.identity.task_id if item == old_six_id else item
            for item in six.result_group.task_ids
        ),
    )
    rebound_one = replace(one, result_group=new_group)
    rebound_six = replace(
        six,
        task=six_task,
        seed_lineage=six_lineage or six.seed_lineage,
        case_id=six_case_id or six.case_id,
        result_group=new_group,
    )
    return replace(
        sample,
        one_worker_results=(WorkerBoundExecutionResult(1, rebound_one),),
        six_worker_results=(WorkerBoundExecutionResult(6, rebound_six),),
    )


def test_parallel_logical_projection_rejects_non_instance_field_swaps():
    sample = _paired_observation().parallel_samples[0]
    six = sample.six_worker_results[0].result
    task = six.task
    identity = task.identity
    resource_changes = (
        replace(task.resources, cpu_tokens=task.resources.cpu_tokens + 1),
        replace(task.resources, rss_bytes=task.resources.rss_bytes + 1),
        replace(task.resources, io_class="other"),
        replace(
            task.resources,
            backend_internal_threads=task.resources.backend_internal_threads + 1,
        ),
        replace(task.resources, exclusive_state="exclusive"),
        replace(task.resources, schema_version="forged-resource-schema"),
    )
    forged_tasks = (
        replace(task, identity=replace(identity, task_kind=TaskKind.LOCALIZATION)),
        replace(task, identity=replace(identity, epoch_index=identity.epoch_index + 1)),
        replace(task, identity=replace(identity, contrast_set_id="other-contrast")),
        replace(task, identity=replace(identity, backend="other-backend")),
        replace(task, payload_digest="other-payload"),
        replace(task, dependency_task_ids=(sample.one_worker_results[0].result.task.identity.task_id,)),
        *(replace(task, resources=value) for value in resource_changes),
    )
    for forged_task in forged_tasks:
        with pytest.raises(ValueError, match="logical task-set|dependency context"):
            _replace_parallel_six_task(sample, forged_task)

    with pytest.raises(ValueError, match="case mismatch"):
        _replace_parallel_six_task(sample, task, six_case_id="other-case")

    other_lineage = replace(six.seed_lineage, protocol_digest="other-protocol")
    protocol_task = replace(
        task,
        identity=replace(
            identity,
            protocol_digest=other_lineage.protocol_digest,
            seed_lineage_digest=other_lineage.digest,
        ),
    )
    with pytest.raises(ValueError, match="seed lineage mismatch|protocol mismatch"):
        _replace_parallel_six_task(
            sample, protocol_task, six_lineage=other_lineage
        )

    endpoint_task = replace(
        task, identity=replace(identity, endpoint_id="endpoint-b")
    )
    with pytest.raises(ValueError, match="endpoint"):
        _replace_parallel_six_task(sample, endpoint_task)

    with pytest.raises(ValueError, match="six-worker arm"):
        replace(
            sample,
            six_worker_results=(WorkerBoundExecutionResult(1, six),),
        )


def test_parallel_distinct_dependency_ids_normalize_to_same_logical_graph():
    lineage = _lineage(counter=400)
    one_a = _task(lineage=lineage, decision_index=400, endpoint_id="endpoint-a")
    six_a = replace(
        one_a,
        identity=replace(one_a.identity, decision_index=500, attempt=500),
    )
    one_b = _task(lineage=lineage, decision_index=401, endpoint_id="endpoint-b")
    one_b = replace(one_b, dependency_task_ids=(one_a.identity.task_id,))
    six_b = replace(
        one_b,
        identity=replace(one_b.identity, decision_index=501, attempt=501),
        dependency_task_ids=(six_a.identity.task_id,),
    )
    group = _group(
        (one_a, one_b, six_a, six_b),
        group_id="parallel-dependency-group",
        endpoints=("endpoint-a", "endpoint-b"),
    )
    one_results = (
        _result(task=one_a, lineage=lineage, group=group),
        _result(task=one_b, lineage=lineage, group=group),
    )
    six_results = (
        _result(task=six_a, lineage=lineage, group=group),
        _result(task=six_b, lineage=lineage, group=group),
    )
    sample = ParallelScalingObservation(
        ordinal=2,
        workload_id="workload-dependency-graph",
        one_worker_results=tuple(
            WorkerBoundExecutionResult(1, item) for item in one_results
        ),
        six_worker_results=tuple(
            WorkerBoundExecutionResult(6, item) for item in six_results
        ),
        one_worker_measurement=RawCounterMeasurement(
            "tasks", 0, 2, RawMonotonicInterval("clock-1", 0, 2_000_000_000)
        ),
        six_worker_measurement=RawCounterMeasurement(
            "tasks", 0, 2, RawMonotonicInterval("clock-1", 0, 500_000_000)
        ),
    )
    observation = replace(_paired_observation(), parallel_samples=(sample,))
    binding = build_paired_benchmark_receipt(observation).parallel_samples[0]

    assert one_b.dependency_task_ids != six_b.dependency_task_ids
    assert binding.one_worker_task_set_digest == binding.six_worker_task_set_digest
    assert binding.one_worker_result_digest != binding.six_worker_result_digest


def test_public_api_freeze_remains_byte_identical():
    path = Path(__file__).parents[2] / "src/datadiff_osc/public_api_freeze.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == PUBLIC_FREEZE_SHA256
