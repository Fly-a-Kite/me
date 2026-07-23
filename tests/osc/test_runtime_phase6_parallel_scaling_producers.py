from __future__ import annotations

from dataclasses import replace

import pytest

import datadiff_osc
from datadiff_osc._canonical import stable_digest
from datadiff_osc.runtime._phase6_parallel_scaling_producers import (
    ParallelScalingBenchmarkContext,
    ParallelScalingProductionObservation,
    build_parallel_scaling_evidence_receipt,
)
from datadiff_osc.runtime._phase6_parallel_scaling_receipts import (
    ParallelScalingEvidenceReceipt,
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
    return stable_digest("phase6-parallel-typed-producer-test", label)


def _seed() -> SeedLineage:
    return SeedLineage(
        protocol_digest=_digest("protocol"),
        master_seed=73,
        lane_id="phase6-parallel-typed",
        case_index=19,
        stage_name=SeedStage.ORACLE_SAMPLE,
    )


def _task(
    *,
    seed: SeedLineage,
    endpoint_id: str,
    decision_index: int,
    attempt: int,
) -> TaskSpec:
    return TaskSpec(
        identity=TaskIdentity(
            protocol_digest=seed.protocol_digest,
            task_kind=TaskKind.BACKEND_EXECUTION,
            epoch_index=3,
            decision_index=decision_index,
            seed_lineage_digest=seed.digest,
            contrast_set_id="phase6-parallel-contrast",
            endpoint_id=endpoint_id,
            backend="backend-a",
            attempt=attempt,
        ),
        dependency_task_ids=(),
        resources=ResourceTokens(
            cpu_tokens=1,
            rss_bytes=0,
            io_class="phase6-test",
            backend_internal_threads=0,
        ),
        payload_digest=_digest(f"payload:{endpoint_id}"),
    )


def _evidence(*, label: str, result: TypedExecutionResult) -> EvidenceEnvelope:
    return EvidenceEnvelope.build(
        evidence_id=f"phase6-parallel-evidence-{label}",
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


def _observation(
    endpoint_ids: tuple[str, ...] = ("endpoint-a",),
) -> ParallelScalingProductionObservation:
    seed = _seed()
    one_tasks = tuple(
        _task(
            seed=seed,
            endpoint_id=endpoint_id,
            decision_index=100 + ordinal,
            attempt=1,
        )
        for ordinal, endpoint_id in enumerate(endpoint_ids)
    )
    six_tasks = tuple(
        _task(
            seed=seed,
            endpoint_id=endpoint_id,
            decision_index=200 + ordinal,
            attempt=2,
        )
        for ordinal, endpoint_id in enumerate(endpoint_ids)
    )
    group = ResultGroup(
        result_group_id="phase6-parallel-typed-group",
        task_ids=tuple(
            task.identity.task_id for task in one_tasks + six_tasks
        ),
        endpoint_ids=endpoint_ids,
        contract_fingerprint=ContractFingerprint(
            contract_digest=_digest("contract"),
            registry_digest=_digest("registry"),
        ),
        target_fingerprint=None,
        evidence_tier=EvidenceTier.SCREENING,
        result_order_key=("phase6-parallel-typed", *endpoint_ids),
    )

    def result(task: TaskSpec) -> TypedExecutionResult:
        return TypedExecutionResult(
            case_id="phase6-parallel-typed-case",
            result_group=group,
            task=task,
            seed_lineage=seed,
            outcome=StructuredExecutionOutcome(
                endpoint_id=task.identity.endpoint_id,
                status=ExecutionStatus.OK,
                failure_kind=FailureKind.NONE,
                reason="parallel typed execution",
            ),
        )

    one_bindings = tuple(
        WorkerBoundExecutionResult(1, result(task)) for task in one_tasks
    )
    six_bindings = tuple(
        WorkerBoundExecutionResult(6, result(task)) for task in six_tasks
    )
    sample = ParallelScalingObservation(
        ordinal=1,
        workload_id="phase6-parallel-workload-001",
        one_worker_results=one_bindings,
        six_worker_results=six_bindings,
        one_worker_measurement=RawCounterMeasurement(
            "tasks",
            0,
            4,
            RawMonotonicInterval("monotonic-phase6", 10, 4_000_000_010),
        ),
        six_worker_measurement=RawCounterMeasurement(
            "tasks",
            10,
            14,
            RawMonotonicInterval("monotonic-phase6", 20, 1_000_000_020),
        ),
    )
    one_evidence = tuple(
        _evidence(label=f"one-{ordinal}", result=binding.result)
        for ordinal, binding in enumerate(
            sorted(one_bindings, key=lambda item: item.result.task.identity.task_id)
        )
    )
    six_evidence = tuple(
        _evidence(label=f"six-{ordinal}", result=binding.result)
        for ordinal, binding in enumerate(
            sorted(six_bindings, key=lambda item: item.result.task.identity.task_id)
        )
    )
    return ParallelScalingProductionObservation(
        benchmark_context=ParallelScalingBenchmarkContext(
            source_snapshot_digest=_digest("source"),
            benchmark_plan=(
                ("sample_scope", "one-typed-parallel-sample"),
                ("scheduler", "fixed"),
            ),
            environment=(("machine", "test"), ("python", "3.12")),
            one_worker_config=(("isolation", "process"),),
            six_worker_config=(("isolation", "process"),),
        ),
        parallel_sample=sample,
        one_worker_evidence=one_evidence,
        six_worker_evidence=six_evidence,
    )


def test_typed_parallel_builder_derives_r5_receipt_without_public_authority():
    observation = _observation()
    receipt = build_parallel_scaling_evidence_receipt(observation)

    assert isinstance(receipt, ParallelScalingEvidenceReceipt)
    assert "ParallelScalingProductionObservation" not in datadiff_osc.__all__
    assert receipt.source_snapshot_digest == observation.benchmark_context.source_snapshot_digest
    assert receipt.workload_id == observation.parallel_sample.workload_id
    assert receipt.one_worker_count == 1
    assert receipt.six_worker_count == 6
    assert receipt.one_worker_task_count == receipt.six_worker_task_count == 4
    assert receipt.one_worker_task_set_digest == receipt.six_worker_task_set_digest
    assert receipt.one_worker_config_digest != receipt.six_worker_config_digest
    assert receipt.one_worker_result_digest != receipt.six_worker_result_digest
    assert receipt.one_worker_evidence_digest != receipt.six_worker_evidence_digest
    assert observation.one_worker_evidence[0].state is EvidenceState.NOT_A_CANDIDATE
    assert observation.six_worker_evidence[0].state is EvidenceState.NOT_A_CANDIDATE

    source_rebound = build_parallel_scaling_evidence_receipt(
        replace(
            observation,
            benchmark_context=replace(
                observation.benchmark_context,
                source_snapshot_digest=_digest("source-rebound"),
            ),
        )
    )
    assert source_rebound.sample_id != receipt.sample_id

    changed_raw = replace(
        observation.parallel_sample,
        one_worker_measurement=RawCounterMeasurement(
            "tasks",
            0,
            5,
            RawMonotonicInterval("monotonic-phase6", 10, 5_000_000_010),
        ),
        six_worker_measurement=RawCounterMeasurement(
            "tasks",
            10,
            15,
            RawMonotonicInterval("monotonic-phase6", 20, 1_100_000_020),
        ),
    )
    changed = build_parallel_scaling_evidence_receipt(
        replace(observation, parallel_sample=changed_raw)
    )
    assert changed.sample_id != receipt.sample_id

    changed_config = build_parallel_scaling_evidence_receipt(
        replace(
            observation,
            benchmark_context=replace(
                observation.benchmark_context,
                six_worker_config=(("isolation", "fresh-process"),),
            ),
        )
    )
    assert changed_config.sample_id != receipt.sample_id


@pytest.mark.parametrize(
    ("mutate", "match"),
    (
        (
            lambda value: replace(
                value,
                one_worker_evidence=(
                    replace(
                        value.one_worker_evidence[0],
                        task_id=value.parallel_sample.six_worker_results[0]
                        .result.task.identity.task_id,
                    ),
                ),
            ),
            "canonical task order",
        ),
        (
            lambda value: replace(
                value,
                one_worker_evidence=(
                    replace(
                        value.one_worker_evidence[0],
                        execution_outcomes=(
                            replace(
                                value.parallel_sample.one_worker_results[0].result.outcome,
                                reason="forged outcome",
                            ),
                        ),
                    ),
                ),
            ),
            "evidence/outcome mismatch",
        ),
        (
            lambda value: replace(
                value,
                one_worker_evidence=(
                    replace(
                        value.one_worker_evidence[0],
                        state=EvidenceState.STABLE_SURVIVOR,
                    ),
                ),
            ),
            "NOT_A_CANDIDATE",
        ),
        (
            lambda value: replace(
                value,
                one_worker_evidence=(
                    replace(
                        value.one_worker_evidence[0],
                        verdict_kind=VerdictKind.INCONCLUSIVE,
                    ),
                ),
            ),
            "must be authoritative",
        ),
        (lambda value: replace(value, one_worker_evidence=()), "exactly one item"),
        (
            lambda value: replace(
                value,
                one_worker_evidence=value.one_worker_evidence
                + value.one_worker_evidence,
            ),
            "exactly one item",
        ),
    ),
)
def test_typed_parallel_builder_fails_closed_on_evidence_mutations(mutate, match):
    with pytest.raises(ValueError, match=match):
        mutate(_observation())


def test_typed_parallel_builder_rejects_reordered_or_duplicate_evidence_vectors():
    observation = _observation(("endpoint-a", "endpoint-b"))

    with pytest.raises(ValueError, match="canonical task order"):
        replace(
            observation,
            one_worker_evidence=tuple(reversed(observation.one_worker_evidence)),
        )
    with pytest.raises(ValueError, match="canonical task order"):
        replace(
            observation,
            six_worker_evidence=(
                observation.six_worker_evidence[0],
                observation.six_worker_evidence[0],
            ),
        )


def test_typed_parallel_builder_rejects_frozen_worker_counter_and_graph_mismatches():
    observation = _observation()
    sample = observation.parallel_sample

    with pytest.raises(ValueError, match="six-worker arm"):
        replace(
            sample,
            six_worker_results=(
                WorkerBoundExecutionResult(1, sample.six_worker_results[0].result),
            ),
        )
    with pytest.raises(ValueError, match="same task count"):
        replace(
            sample,
            six_worker_measurement=RawCounterMeasurement(
                "tasks",
                10,
                15,
                RawMonotonicInterval("monotonic-phase6", 20, 1_100_000_020),
            ),
        )

    forged_task = replace(
        sample.six_worker_results[0].result.task,
        dependency_task_ids=("missing-logical-dependency",),
    )
    forged_result = replace(sample.six_worker_results[0].result, task=forged_task)
    with pytest.raises(ValueError, match="dependency context"):
        replace(
            sample,
            six_worker_results=(WorkerBoundExecutionResult(6, forged_result),),
        )

    non_ok = replace(
        sample.six_worker_results[0].result.outcome,
        status=ExecutionStatus.SEMANTIC_ERROR,
        failure_kind=FailureKind.SEMANTIC_DOMAIN_ERROR,
    )
    with pytest.raises(ValueError, match="requires OK outcomes"):
        replace(
            sample,
            six_worker_results=(
                WorkerBoundExecutionResult(
                    6,
                    replace(sample.six_worker_results[0].result, outcome=non_ok),
                ),
            ),
        )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    (
        (
            {"benchmark_plan": (("z", "1"), ("a", "2"))},
            "uniquely sorted",
        ),
        (
            {"environment": (("a", "1"), ("a", "2"))},
            "uniquely sorted",
        ),
        (
            {"one_worker_config": (("worker_count", "1"),)},
            "derived worker or metric",
        ),
        (
            {"six_worker_config": ()},
            "non-empty immutable field tuple",
        ),
        ({"source_snapshot_digest": "not-a-digest"}, "SHA-256"),
    ),
)
def test_typed_parallel_context_rejects_ambiguous_or_caller_authored_fields(
    kwargs, match
):
    observation = _observation()
    values = {
        "source_snapshot_digest": observation.benchmark_context.source_snapshot_digest,
        "benchmark_plan": observation.benchmark_context.benchmark_plan,
        "environment": observation.benchmark_context.environment,
        "one_worker_config": observation.benchmark_context.one_worker_config,
        "six_worker_config": observation.benchmark_context.six_worker_config,
    }
    values.update(kwargs)

    with pytest.raises(ValueError, match=match):
        ParallelScalingBenchmarkContext(**values)

    with pytest.raises(TypeError):
        ParallelScalingBenchmarkContext(  # type: ignore[call-arg]
            **values,
            caller_efficiency="1.0",
        )


def test_typed_parallel_builder_rejects_wrong_objects_and_legacy_shaped_input():
    observation = _observation()

    with pytest.raises(TypeError, match="ParallelScalingProductionObservation"):
        build_parallel_scaling_evidence_receipt(object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="parallel sample"):
        ParallelScalingProductionObservation(
            benchmark_context=observation.benchmark_context,
            parallel_sample=object(),  # type: ignore[arg-type]
            one_worker_evidence=observation.one_worker_evidence,
            six_worker_evidence=observation.six_worker_evidence,
        )
    with pytest.raises(TypeError):
        ParallelScalingProductionObservation(  # type: ignore[call-arg]
            benchmark_context=observation.benchmark_context,
            parallel_sample=observation.parallel_sample,
            one_worker_evidence=observation.one_worker_evidence,
            six_worker_evidence=observation.six_worker_evidence,
            sample_id="caller-authored",
        )
