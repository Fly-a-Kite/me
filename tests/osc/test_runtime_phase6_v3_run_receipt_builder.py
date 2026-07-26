from __future__ import annotations

from dataclasses import replace
import functools

import pytest

from datadiff_osc._canonical import (
    canonical_envelope,
    decode_canonical_envelope,
    stable_digest,
)
from datadiff_osc.contract_engine import Endpoint
from datadiff_osc.runtime._phase6_v3_run_receipt_builder import (
    V3_CASE_PIPELINE_DIGEST_NAMESPACE,
    V3_EXECUTION_RUN_DIGEST_NAMESPACE,
    build_v3_backend_task_binding,
    build_v3_case_binding,
    build_v3_run_receipt,
    v3_case_pipeline_digest,
    v3_execution_run_digest,
)
from datadiff_osc.runtime._private_receipts import V3CaseBinding, V3RunReceipt
from datadiff_osc.runtime._receipt_producers import TypedExecutionResult
from datadiff_osc.runtime._semantic_replay import replay_runtime_admission
from datadiff_osc.runtime.protocols import V3GatePlan, build_v3_gate_plan
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
    StructuredExecutionOutcome,
    TaskIdentity,
    TaskKind,
    TaskSpec,
    VerdictKind,
)
from datadiff_osc.search.context_receipts import FormalRunBinding


_NS = "phase6-v3-run-receipt-builder-test"
_LANE_IDS = tuple(f"lane-{index:02d}" for index in range(11))
_SEEDS = (30700001, 30700102)
_BACKENDS = ("backend-left", "backend-right")


def _digest(label: str) -> str:
    return stable_digest(_NS, label)


_PROTOCOL_DIGEST = _digest("protocol")
_SOURCE_DIGEST = _digest("source")


def _gate_plan(*, source_digest: str | None = None) -> V3GatePlan:
    return build_v3_gate_plan(
        lane_ids=_LANE_IDS,
        seeds=_SEEDS,
        source_digest=source_digest or _SOURCE_DIGEST,
        protocol_digest=_PROTOCOL_DIGEST,
    )


def _run_binding(*, master_seed: int = _SEEDS[0]) -> FormalRunBinding:
    return FormalRunBinding.build(
        protocol_digest=_PROTOCOL_DIGEST,
        lane_id=_LANE_IDS[0],
        seed_block_id=f"seed-block-{master_seed}",
        master_seed=master_seed,
        indexed_case_ids=tuple(
            (f"case-{index:03d}", index) for index in range(100)
        ),
    )


def _typed_case(
    case_index: int,
    lineage: SeedLineage,
    *,
    status: ExecutionStatus = ExecutionStatus.OK,
) -> tuple[
    TypedExecutionResult,
    EvidenceEnvelope,
    tuple[TypedExecutionResult, ...],
    tuple[Endpoint, ...],
]:
    suffix = f"{case_index:03d}"
    case_id = f"case-{suffix}"
    resources = ResourceTokens(
        cpu_tokens=1,
        rss_bytes=0,
        io_class="osc-private-v3-builder-test",
        backend_internal_threads=0,
    )
    endpoints = tuple(
        Endpoint.build(
            endpoint_id=f"{case_id}:{backend}",
            case_digest=_digest(f"case-bytes-{suffix}"),
            backend=backend,
            backend_version="1.0.0",
            adapter_revision="adapter-v1",
            execution_mode="eager",
            physical_layout="contiguous",
            capabilities=frozenset({"op:select", "type:int"}),
        )
        for backend in _BACKENDS
    )
    failure_kind = {
        ExecutionStatus.OK: FailureKind.NONE,
        ExecutionStatus.CRASH: FailureKind.CRASH,
    }[status]
    outcomes = tuple(
        StructuredExecutionOutcome(
            endpoint_id=endpoint.endpoint_id,
            status=status,
            failure_kind=failure_kind,
        )
        for endpoint in endpoints
    )
    backend_tasks = tuple(
        TaskSpec(
            identity=TaskIdentity(
                protocol_digest=_PROTOCOL_DIGEST,
                task_kind=TaskKind.BACKEND_EXECUTION,
                epoch_index=0,
                decision_index=case_index,
                seed_lineage_digest=lineage.digest,
                endpoint_id=endpoint.endpoint_id,
                backend=endpoint.backend,
                attempt=0,
            ),
            dependency_task_ids=(),
            resources=resources,
            payload_digest=_digest(
                f"backend-payload-{suffix}-{endpoint.endpoint_id}"
            ),
        )
        for endpoint in endpoints
    )
    case_task = TaskSpec(
        identity=TaskIdentity(
            protocol_digest=_PROTOCOL_DIGEST,
            task_kind=TaskKind.FULL_DIFF,
            epoch_index=0,
            decision_index=case_index,
            seed_lineage_digest=lineage.digest,
        ),
        dependency_task_ids=(),
        resources=resources,
        payload_digest=_digest(f"case-payload-{suffix}"),
    )
    group = ResultGroup(
        result_group_id=_digest(f"result-group-{suffix}"),
        task_ids=(case_task.identity.task_id,)
        + tuple(item.identity.task_id for item in backend_tasks),
        endpoint_ids=tuple(item.endpoint_id for item in endpoints),
        contract_fingerprint=ContractFingerprint(
            contract_digest=_digest("contract"),
            registry_digest=_digest("registry"),
        ),
        target_fingerprint=None,
        evidence_tier=EvidenceTier.AUDIT,
        result_order_key=(case_id,),
    )
    case_result = TypedExecutionResult(
        case_id=case_id,
        result_group=group,
        task=case_task,
        seed_lineage=lineage,
        outcome=outcomes[0],
    )
    backend_results = tuple(
        TypedExecutionResult(
            case_id=case_id,
            result_group=group,
            task=task,
            seed_lineage=lineage,
            outcome=outcome,
        )
        for task, outcome in zip(backend_tasks, outcomes, strict=True)
    )
    evidence = EvidenceEnvelope.build(
        evidence_id=_digest(f"evidence-{suffix}"),
        state=EvidenceState.NOT_A_CANDIDATE,
        result_group_digest=group.digest,
        task_id=case_task.identity.task_id,
        seed_lineage_digest=lineage.digest,
        contract_fingerprint=group.contract_fingerprint,
        target_fingerprint=None,
        derivation_certificate_digest=_digest(f"derivation-{suffix}"),
        applicability_certificate_digest=_digest(f"applicability-{suffix}"),
        activation_certificate_digest=_digest(f"activation-{suffix}"),
        observation_certificate_digest=_digest(f"observation-{suffix}"),
        execution_outcomes=outcomes,
        verdict_kind=VerdictKind.SATISFIED,
        artifact_refs=(),
        metadata={},
    )
    return case_result, evidence, backend_results, endpoints


def _case_lineages(run: FormalRunBinding) -> dict[int, SeedLineage]:
    return {item.case_index: item.seed_lineage for item in run.case_bindings}


def _built_case(
    case_index: int,
    lineage: SeedLineage,
    *,
    status: ExecutionStatus = ExecutionStatus.OK,
    executed: bool = True,
    iteration_failure: bool = False,
    pipeline_error: bool = False,
    forced_index: int | None = None,
) -> V3CaseBinding:
    case_result, evidence, backend_results, endpoints = _typed_case(
        case_index, lineage, status=status
    )
    return build_v3_case_binding(
        case_index=case_index if forced_index is None else forced_index,
        case_result=case_result,
        evidence=evidence,
        backend_results=backend_results,
        endpoints=endpoints,
        executed=executed,
        iteration_failure=iteration_failure,
        pipeline_error=pipeline_error,
    )


@functools.lru_cache(maxsize=1)
def _built_cases() -> tuple[V3CaseBinding, ...]:
    lineages = _case_lineages(_run_binding())
    return tuple(_built_case(index, lineages[index]) for index in range(100))


def _receipt() -> V3RunReceipt:
    return build_v3_run_receipt(
        gate_plan=_gate_plan(),
        run_binding=_run_binding(),
        cases=_built_cases(),
    )


def test_frozen_digest_functions_are_deterministic_and_order_canonical():
    pipeline = v3_case_pipeline_digest(
        case_id="case-000",
        case_task_id="case-task-000",
        result_digest=_digest("result"),
        execution_outcome_digest=_digest("outcome"),
        evidence_envelope_digest=_digest("evidence"),
        backend_task_ids=("task-b", "task-a"),
    )
    assert pipeline.startswith(f"{V3_CASE_PIPELINE_DIGEST_NAMESPACE}-")
    assert pipeline == v3_case_pipeline_digest(
        case_id="case-000",
        case_task_id="case-task-000",
        result_digest=_digest("result"),
        execution_outcome_digest=_digest("outcome"),
        evidence_envelope_digest=_digest("evidence"),
        backend_task_ids=("task-a", "task-b"),
    )
    run_digest = v3_execution_run_digest(
        v3_gate_plan_digest=_digest("plan"),
        lane_id="lane-00",
        seed=_SEEDS[0],
        run_id="run-000",
        case_ids=("case-001", "case-000"),
        case_result_digests=(_digest("result-1"), _digest("result-0")),
    )
    assert run_digest.startswith(f"{V3_EXECUTION_RUN_DIGEST_NAMESPACE}-")
    assert run_digest == v3_execution_run_digest(
        v3_gate_plan_digest=_digest("plan"),
        lane_id="lane-00",
        seed=_SEEDS[0],
        run_id="run-000",
        case_ids=("case-000", "case-001"),
        case_result_digests=(_digest("result-0"), _digest("result-1")),
    )
    with pytest.raises(ValueError, match="must be unique"):
        v3_case_pipeline_digest(
            case_id="case-000",
            case_task_id="case-task-000",
            result_digest=_digest("result"),
            execution_outcome_digest=_digest("outcome"),
            evidence_envelope_digest=_digest("evidence"),
            backend_task_ids=("task-a", "task-a"),
        )
    with pytest.raises(ValueError, match="one-to-one"):
        v3_execution_run_digest(
            v3_gate_plan_digest=_digest("plan"),
            lane_id="lane-00",
            seed=_SEEDS[0],
            run_id="run-000",
            case_ids=("case-000", "case-001"),
            case_result_digests=(_digest("result-0"),),
        )


def test_case_binding_derives_every_digest_from_typed_objects():
    lineage = _case_lineages(_run_binding())[0]
    case_result, evidence, backend_results, endpoints = _typed_case(0, lineage)

    binding = build_v3_case_binding(
        case_index=0,
        case_result=case_result,
        evidence=evidence,
        backend_results=backend_results,
        endpoints=endpoints,
        executed=True,
        iteration_failure=False,
        pipeline_error=False,
    )

    assert binding.case_id == case_result.case_id
    assert binding.seed_lineage_digest == lineage.digest
    assert binding.case_task_id == case_result.task.identity.task_id
    assert binding.case_task_spec_digest == case_result.task.digest
    assert binding.result_digest == case_result.result_group.digest
    assert binding.execution_outcome_digest == case_result.outcome.digest
    assert binding.evidence_envelope_digest == evidence.digest
    assert binding.pipeline_digest == v3_case_pipeline_digest(
        case_id=binding.case_id,
        case_task_id=binding.case_task_id,
        result_digest=binding.result_digest,
        execution_outcome_digest=binding.execution_outcome_digest,
        evidence_envelope_digest=binding.evidence_envelope_digest,
        backend_task_ids=tuple(item.task_id for item in binding.backend_tasks),
    )
    assert len(binding.backend_tasks) == len(_BACKENDS)
    endpoint_digests = {item.digest for item in endpoints}
    for task_binding, backend_result in zip(
        binding.backend_tasks,
        sorted(backend_results, key=lambda item: item.task.identity.task_id),
        strict=True,
    ):
        assert task_binding.task_id == backend_result.task.identity.task_id
        assert task_binding.task_spec_digest == backend_result.task.digest
        assert task_binding.execution_outcome_digest == (
            backend_result.outcome.digest
        )
        assert task_binding.endpoint_digest in endpoint_digests
        assert task_binding.status is ExecutionStatus.OK


def test_backend_binding_rejects_non_backend_task_and_endpoint_mismatch():
    lineage = _case_lineages(_run_binding())[0]
    case_result, _, backend_results, endpoints = _typed_case(0, lineage)

    with pytest.raises(ValueError, match="backend execution task"):
        build_v3_backend_task_binding(result=case_result, endpoint=endpoints[0])
    with pytest.raises(ValueError, match="endpoint identity mismatch"):
        build_v3_backend_task_binding(
            result=backend_results[0], endpoint=endpoints[1]
        )
    with pytest.raises(TypeError, match="TypedExecutionResult"):
        build_v3_backend_task_binding(
            result=object(),  # type: ignore[arg-type]
            endpoint=endpoints[0],
        )
    with pytest.raises(TypeError, match="Endpoint"):
        build_v3_backend_task_binding(
            result=backend_results[0],
            endpoint=object(),  # type: ignore[arg-type]
        )


def test_case_binding_rejects_cross_binding_forgeries():
    lineages = _case_lineages(_run_binding())
    case_result, evidence, backend_results, endpoints = _typed_case(
        0, lineages[0]
    )
    _, other_evidence, other_backend_results, other_endpoints = _typed_case(
        1, lineages[1]
    )

    with pytest.raises(ValueError, match="evidence/result-group mismatch"):
        build_v3_case_binding(
            case_index=0,
            case_result=case_result,
            evidence=other_evidence,
            backend_results=backend_results,
            endpoints=endpoints,
            executed=True,
            iteration_failure=False,
            pipeline_error=False,
        )
    with pytest.raises(ValueError, match="not bound to the case"):
        build_v3_case_binding(
            case_index=0,
            case_result=case_result,
            evidence=evidence,
            backend_results=other_backend_results,
            endpoints=other_endpoints,
            executed=True,
            iteration_failure=False,
            pipeline_error=False,
        )
    with pytest.raises(ValueError, match="one-to-one with backend executions"):
        build_v3_case_binding(
            case_index=0,
            case_result=case_result,
            evidence=evidence,
            backend_results=backend_results,
            endpoints=endpoints[:1],
            executed=True,
            iteration_failure=False,
            pipeline_error=False,
        )
    with pytest.raises(ValueError, match="unique endpoints"):
        build_v3_case_binding(
            case_index=0,
            case_result=case_result,
            evidence=evidence,
            backend_results=(backend_results[0], backend_results[0]),
            endpoints=endpoints,
            executed=True,
            iteration_failure=False,
            pipeline_error=False,
        )


def test_builder_assembles_completed_hundred_case_receipt():
    receipt = _receipt()
    gate_plan = _gate_plan()
    run = _run_binding()

    assert receipt.source_digest == gate_plan.source_digest
    assert receipt.protocol_digest == gate_plan.protocol_digest
    assert receipt.v3_gate_plan_digest == gate_plan.digest
    assert receipt.run_id == run.run_id
    assert receipt.lane_id == run.lane_id
    assert receipt.seed == run.master_seed
    assert receipt.run_completed is True
    assert len(receipt.cases) == 100
    assert receipt.execution_run_digest == v3_execution_run_digest(
        v3_gate_plan_digest=gate_plan.digest,
        lane_id=run.lane_id,
        seed=run.master_seed,
        run_id=run.run_id,
        case_ids=tuple(item.case_id for item in receipt.cases),
        case_result_digests=tuple(
            item.result_digest for item in receipt.cases
        ),
    )
    assert build_v3_run_receipt(
        gate_plan=gate_plan,
        run_binding=run,
        cases=_built_cases(),
        run_completed=True,
    ) == receipt


def test_built_receipt_replays_through_runtime_admission_for_every_subject_kind():
    receipt = _receipt()
    decoded = decode_canonical_envelope(
        canonical_envelope("V3RunReceipt", receipt.schema_version, receipt)
    )
    for subject_kind, subject_ids in (
        ("v3_runs", (receipt.run_id,)),
        ("v3_cases", receipt.case_ids),
        ("v3_executed_cases", receipt.executed_case_ids),
        ("v3_backend_tasks", receipt.backend_task_ids),
    ):
        assert (
            replay_runtime_admission(
                envelope_type="V3RunReceipt",
                schema_version=receipt.schema_version,
                payload=decoded["payload"],
                subject_kind=subject_kind,
                subject_ids=subject_ids,
            )
            == ()
        )


def test_builder_derives_incomplete_run_from_failure_flags():
    lineages = _case_lineages(_run_binding())
    cases = list(_built_cases())
    cases[7] = _built_case(
        7,
        lineages[7],
        status=ExecutionStatus.CRASH,
        executed=True,
        iteration_failure=True,
        pipeline_error=False,
    )

    receipt = build_v3_run_receipt(
        gate_plan=_gate_plan(),
        run_binding=_run_binding(),
        cases=tuple(cases),
    )

    assert receipt.run_completed is False
    assert len(receipt.executed_case_ids) == 100
    failed = receipt.cases[7]
    assert failed.iteration_failure is True
    assert all(
        item.status is ExecutionStatus.CRASH for item in failed.backend_tasks
    )


def test_builder_rejects_wrong_case_count():
    with pytest.raises(ValueError, match="exactly 100 case bindings"):
        build_v3_run_receipt(
            gate_plan=_gate_plan(),
            run_binding=_run_binding(),
            cases=_built_cases()[:99],
        )


def test_builder_rejects_duplicate_case_identities():
    lineages = _case_lineages(_run_binding())
    cases = list(_built_cases())
    cases[1] = _built_case(0, lineages[0], forced_index=1)

    with pytest.raises(ValueError, match="case identities must be unique"):
        build_v3_run_receipt(
            gate_plan=_gate_plan(),
            run_binding=_run_binding(),
            cases=tuple(cases),
        )


def test_builder_rejects_duplicate_backend_task_identities():
    cases = list(_built_cases())
    cases[1] = replace(cases[1], backend_tasks=cases[0].backend_tasks)

    with pytest.raises(
        ValueError, match="backend task identities must be globally unique"
    ):
        build_v3_run_receipt(
            gate_plan=_gate_plan(),
            run_binding=_run_binding(),
            cases=tuple(cases),
        )


def test_builder_rejects_case_index_gap():
    lineages = _case_lineages(_run_binding())
    cases = list(_built_cases())
    cases[99] = _built_case(99, lineages[99], forced_index=100)

    with pytest.raises(ValueError, match="exactly 0 through 99"):
        build_v3_run_receipt(
            gate_plan=_gate_plan(),
            run_binding=_run_binding(),
            cases=tuple(cases),
        )


def test_builder_rejects_completed_claim_over_failed_case():
    lineages = _case_lineages(_run_binding())
    cases = list(_built_cases())
    cases[3] = _built_case(
        3,
        lineages[3],
        status=ExecutionStatus.CRASH,
        executed=False,
        iteration_failure=False,
        pipeline_error=True,
    )

    with pytest.raises(
        ValueError, match="run_completed conflicts with the case execution flags"
    ):
        build_v3_run_receipt(
            gate_plan=_gate_plan(),
            run_binding=_run_binding(),
            cases=tuple(cases),
            run_completed=True,
        )


def test_builder_rejects_tampered_pipeline_digest():
    cases = list(_built_cases())
    cases[5] = replace(cases[5], pipeline_digest=_digest("forged-pipeline"))

    with pytest.raises(ValueError, match="pipeline digest does not recompute"):
        build_v3_run_receipt(
            gate_plan=_gate_plan(),
            run_binding=_run_binding(),
            cases=tuple(cases),
        )


def test_builder_rejects_run_binding_and_plan_misbindings():
    with pytest.raises(ValueError, match="seed lineage does not recompute"):
        build_v3_run_receipt(
            gate_plan=_gate_plan(),
            run_binding=_run_binding(master_seed=_SEEDS[1]),
            cases=_built_cases(),
        )
    foreign_plan = build_v3_gate_plan(
        lane_ids=tuple(f"other-lane-{index:02d}" for index in range(11)),
        seeds=_SEEDS,
        source_digest=_SOURCE_DIGEST,
        protocol_digest=_PROTOCOL_DIGEST,
    )
    with pytest.raises(ValueError, match="lane is not part of the v3 gate plan"):
        build_v3_run_receipt(
            gate_plan=foreign_plan,
            run_binding=_run_binding(),
            cases=_built_cases(),
        )
    with pytest.raises(TypeError, match="V3GatePlan"):
        build_v3_run_receipt(
            gate_plan=object(),  # type: ignore[arg-type]
            run_binding=_run_binding(),
            cases=_built_cases(),
        )
    with pytest.raises(TypeError, match="FormalRunBinding"):
        build_v3_run_receipt(
            gate_plan=_gate_plan(),
            run_binding=object(),  # type: ignore[arg-type]
            cases=_built_cases(),
        )
