from __future__ import annotations

from dataclasses import replace

import pytest

import datadiff_osc
import datadiff_osc._phase6_gate_authority as authority_module
from datadiff_osc._canonical import stable_digest
from datadiff_osc.contract_engine._phase6_comparison_receipts import (
    CanonicalEndpointValue,
    ContractComparisonReceipt,
)
from datadiff_osc.contract_engine.evidence import ObservationCertificate
from datadiff_osc.contract_engine.fingerprints import ComponentFingerprint
from datadiff_osc.contract_engine.model import (
    ComponentVerdict,
    Endpoint,
    EndpointRequirement,
    HyperContract,
    RelationObligation,
    Verdict,
)
from datadiff_osc.contract_engine.planner import plan_comparison
from datadiff_osc.contract_engine.relations import default_relation_registry
from datadiff_osc.runtime._phase6_contract_performance_producers import (
    ContractPerformanceBenchmarkContext,
    ContractPerformanceProductionObservation,
    RawContractPerformanceCounters,
    build_contract_performance_evidence_receipt,
)
from datadiff_osc.runtime._phase6_contract_performance_receipts import (
    BASELINE_STRATEGY,
    TREATMENT_STRATEGY,
    ContractPerformanceEvidenceReceipt,
)
from datadiff_osc.runtime._receipt_producers import TypedExecutionResult
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
    StagedComparisonRequest,
    StagedComparisonResult,
    StructuredExecutionOutcome,
    TaskIdentity,
    TaskKind,
    TaskSpec,
    VerdictKind,
)


_PROTOCOL = stable_digest("phase6-raw-pair-protocol", "v1")


def _contract() -> HyperContract:
    endpoint_ids = ("endpoint-a", "endpoint-b")
    requirements = tuple(
        EndpointRequirement(
            endpoint_id=endpoint_id,
            backend="backend-a",
            version_spec="1.2.3",
            execution_modes=frozenset({"eager"}),
            physical_layouts=frozenset({"contiguous"}),
            required_capabilities=frozenset({"execute"}),
        )
        for endpoint_id in endpoint_ids
    )
    obligation = RelationObligation.build(
        obligation_id="status-obligation",
        relation_id="status_ok",
        endpoint_ids=endpoint_ids,
        components=("status",),
        strength="status_exact",
        relation_properties={"reflexive", "symmetric", "transitive"},
    )
    return HyperContract(
        contract_id="phase6-raw-pair-contract",
        endpoint_requirements=requirements,
        preconditions=("no_unresolved_rules", "endpoint_scope_matches"),
        observations=("status",),
        obligations=(obligation,),
        strength="status_exact",
        derivation_digest=stable_digest("phase6-raw-pair-derivation", "v1"),
        registry_digest=default_relation_registry().digest,
    )


def _certificate(
    *,
    contract: HyperContract,
    endpoint_ids: tuple[str, ...],
    fingerprints: tuple[ComponentFingerprint, ...],
    exact: bool,
) -> ObservationCertificate:
    component = ComponentVerdict.build(
        "status-obligation",
        VerdictKind.SATISFIED,
        "raw comparison outcome",
        {},
    )
    return ObservationCertificate(
        contract_digest=contract.digest,
        applicability_digest=stable_digest("phase6-raw-pair-applicability", "v1"),
        executed_endpoint_ids=endpoint_ids,
        observer_ids=("status",),
        component_fingerprints=fingerprints,
        relation_trace=("status-obligation: raw comparison",),
        comparison_stage="S3_EXACT_MATERIALIZED" if exact else "S2_COMPONENT_FINGERPRINT",
        exact_escalated=exact,
        materialized_endpoint_ids=endpoint_ids if exact else (),
        verdict=Verdict.aggregate((component,), reason="raw comparison"),
    )


def _task_identity(
    *,
    kind: TaskKind,
    seed: SeedLineage,
    attempt: int,
) -> TaskIdentity:
    return TaskIdentity(
        protocol_digest=_PROTOCOL,
        task_kind=kind,
        epoch_index=7,
        decision_index=11,
        seed_lineage_digest=seed.digest,
        contrast_set_id="contrast-phase6-raw-pair",
        attempt=attempt,
    )


def _evidence(
    *,
    label: str,
    result: TypedExecutionResult,
    verdict: VerdictKind,
) -> EvidenceEnvelope:
    return EvidenceEnvelope.build(
        evidence_id=f"phase6-raw-pair-evidence-{label}",
        state=EvidenceState.NOT_A_CANDIDATE,
        result_group_digest=result.result_group.digest,
        task_id=result.task.identity.task_id,
        seed_lineage_digest=result.seed_lineage.digest,
        contract_fingerprint=result.result_group.contract_fingerprint,
        target_fingerprint=result.result_group.target_fingerprint,
        derivation_certificate_digest=stable_digest("phase6-raw-pair-derivation", label),
        applicability_certificate_digest=stable_digest(
            "phase6-raw-pair-applicability", label
        ),
        activation_certificate_digest=stable_digest("phase6-raw-pair-activation", label),
        observation_certificate_digest=stable_digest(
            "phase6-raw-pair-observation", label
        ),
        execution_outcomes=(result.outcome,),
        verdict_kind=verdict,
    )


def _observation() -> ContractPerformanceProductionObservation:
    contract = _contract()
    source_case_digest = stable_digest("phase6-raw-pair-case", "v1")
    endpoints = tuple(
        Endpoint.build(
            endpoint_id=endpoint_id,
            case_digest=source_case_digest,
            backend="backend-a",
            backend_version="1.2.3",
            adapter_revision="adapter-r1",
            execution_mode="eager",
            physical_layout="contiguous",
            optimizer_config={"ordinal": ordinal},
            capabilities={"execute"},
        )
        for ordinal, endpoint_id in enumerate(("endpoint-a", "endpoint-b"))
    )
    endpoint_ids = tuple(item.endpoint_id for item in endpoints)
    seed = SeedLineage(
        protocol_digest=_PROTOCOL,
        master_seed=17,
        lane_id="phase6-contract",
        case_index=11,
        stage_name=SeedStage.ORACLE_SAMPLE,
    )
    fingerprint_identity = _task_identity(
        kind=TaskKind.FINGERPRINT_CLUSTER, seed=seed, attempt=0
    )
    exact_identity = _task_identity(kind=TaskKind.FULL_DIFF, seed=seed, attempt=0)
    baseline_identity = _task_identity(
        kind=TaskKind.FULL_DIFF, seed=seed, attempt=1
    )
    treatment_identity = _task_identity(
        kind=TaskKind.FULL_DIFF, seed=seed, attempt=2
    )
    group = ResultGroup(
        result_group_id="phase6-raw-pair-group",
        task_ids=(
            fingerprint_identity.task_id,
            exact_identity.task_id,
            baseline_identity.task_id,
            treatment_identity.task_id,
        ),
        endpoint_ids=endpoint_ids,
        contract_fingerprint=ContractFingerprint(
            contract_digest=contract.digest,
            registry_digest=contract.registry_digest,
        ),
        target_fingerprint=None,
        evidence_tier=EvidenceTier.SCREENING,
        result_order_key=("phase6-raw-pair", *endpoint_ids),
    )
    fingerprint_request = StagedComparisonRequest(
        result_group_digest=group.digest,
        contract_fingerprint=group.contract_fingerprint,
        endpoint_ids=endpoint_ids,
        observer_ids=("status",),
        evidence_tier=EvidenceTier.SCREENING,
    )
    exact_request = StagedComparisonRequest(
        result_group_digest=group.digest,
        contract_fingerprint=group.contract_fingerprint,
        endpoint_ids=endpoint_ids,
        observer_ids=("status",),
        evidence_tier=EvidenceTier.SCREENING,
        force_exact=True,
    )
    resources = ResourceTokens(
        cpu_tokens=1,
        rss_bytes=0,
        io_class="phase6-test",
        backend_internal_threads=0,
    )
    fingerprint_task = TaskSpec(
        identity=fingerprint_identity,
        dependency_task_ids=(),
        resources=resources,
        payload_digest=fingerprint_request.digest,
    )
    exact_task = TaskSpec(
        identity=exact_identity,
        dependency_task_ids=(fingerprint_identity.task_id,),
        resources=resources,
        payload_digest=exact_request.digest,
    )
    baseline_task = TaskSpec(
        identity=baseline_identity,
        dependency_task_ids=(fingerprint_identity.task_id,),
        resources=resources,
        payload_digest=stable_digest("phase6-raw-pair-arm-payload", "baseline"),
    )
    treatment_task = TaskSpec(
        identity=treatment_identity,
        dependency_task_ids=(fingerprint_identity.task_id,),
        resources=resources,
        payload_digest=stable_digest("phase6-raw-pair-arm-payload", "treatment"),
    )
    fingerprints = tuple(
        ComponentFingerprint(
            endpoint_id=endpoint_id,
            observer_id="status",
            observer_digest=stable_digest("phase6-raw-pair-observer", "status"),
            contract_digest=contract.digest,
            row_count=1,
            schema_digest="1" * 64,
            payload_digest="2" * 64,
        )
        for endpoint_id in endpoint_ids
    )
    fingerprint_certificate = _certificate(
        contract=contract,
        endpoint_ids=endpoint_ids,
        fingerprints=fingerprints,
        exact=False,
    )
    exact_certificate = _certificate(
        contract=contract,
        endpoint_ids=endpoint_ids,
        fingerprints=fingerprints,
        exact=True,
    )
    plan_digest = plan_comparison(contract, evidence_tier="screening").digest
    fingerprint_result = StagedComparisonResult(
        request_digest=fingerprint_request.digest,
        plan_digest=plan_digest,
        observation_certificate_digest=fingerprint_certificate.digest,
        evidence_tier=EvidenceTier.SCREENING,
        comparison_stage=fingerprint_certificate.comparison_stage,
        exact_escalated=False,
        endpoint_order=endpoint_ids,
        component_fingerprint_digests=tuple(item.digest for item in fingerprints),
        verdict_kind=fingerprint_certificate.verdict.kind,
    )
    exact_result = StagedComparisonResult(
        request_digest=exact_request.digest,
        plan_digest=plan_digest,
        observation_certificate_digest=exact_certificate.digest,
        evidence_tier=EvidenceTier.SCREENING,
        comparison_stage=exact_certificate.comparison_stage,
        exact_escalated=True,
        endpoint_order=endpoint_ids,
        component_fingerprint_digests=tuple(item.digest for item in fingerprints),
        verdict_kind=exact_certificate.verdict.kind,
    )
    comparison = ContractComparisonReceipt(
        source_snapshot_digest=stable_digest("phase6-raw-pair-source", "v1"),
        source_case_digest=source_case_digest,
        result_group=group,
        contract=contract,
        endpoints=endpoints,
        fingerprint_task=fingerprint_task,
        fingerprint_request=fingerprint_request,
        fingerprint_result=fingerprint_result,
        fingerprint_certificate=fingerprint_certificate,
        exact_task=exact_task,
        exact_request=exact_request,
        exact_result=exact_result,
        exact_certificate=exact_certificate,
        canonical_values=(
            CanonicalEndpointValue("endpoint-a", ("value", "same")),
            CanonicalEndpointValue("endpoint-b", ("value", "same")),
        ),
    )
    baseline_result = TypedExecutionResult(
        case_id="phase6-raw-pair-case-id",
        result_group=group,
        task=baseline_task,
        seed_lineage=seed,
        outcome=StructuredExecutionOutcome(
            endpoint_id="endpoint-a",
            status=ExecutionStatus.OK,
            failure_kind=FailureKind.NONE,
            reason="baseline exact execution",
        ),
    )
    treatment_result = TypedExecutionResult(
        case_id="phase6-raw-pair-case-id",
        result_group=group,
        task=treatment_task,
        seed_lineage=seed,
        outcome=StructuredExecutionOutcome(
            endpoint_id="endpoint-a",
            status=ExecutionStatus.OK,
            failure_kind=FailureKind.NONE,
            reason="treatment exact execution",
        ),
    )
    return ContractPerformanceProductionObservation(
        comparison=comparison,
        contract_comparison_producer_receipt_digest=stable_digest(
            "phase6-raw-pair-r2-producer", "v1"
        ),
        benchmark_context=ContractPerformanceBenchmarkContext(
            benchmark_plan=(
                ("comparator", "full_canonical"),
                ("sample_scope", "one_exact_pair"),
            ),
            environment=(("machine", "test"), ("python", "3.12")),
            baseline_config=(("materialization", "unconditional"),),
            treatment_config=(("planner", "staged_exact"),),
        ),
        baseline_result=baseline_result,
        treatment_result=treatment_result,
        baseline_evidence=_evidence(
            label="baseline",
            result=baseline_result,
            verdict=comparison.exact_result.verdict_kind,
        ),
        treatment_evidence=_evidence(
            label="treatment",
            result=treatment_result,
            verdict=comparison.exact_result.verdict_kind,
        ),
        baseline_counters=RawContractPerformanceCounters(
            materialized_bytes=4096,
            backend_pair_comparisons=6,
            comparison_cpu_ns=10_000,
        ),
        treatment_counters=RawContractPerformanceCounters(
            materialized_bytes=1024,
            backend_pair_comparisons=2,
            comparison_cpu_ns=3_000,
        ),
    )


def test_typed_raw_pair_builder_derives_the_r3_receipt_without_public_authority():
    observation = _observation()
    receipt = build_contract_performance_evidence_receipt(observation)

    assert isinstance(receipt, ContractPerformanceEvidenceReceipt)
    assert "ContractPerformanceProductionObservation" not in datadiff_osc.__all__
    assert receipt.baseline_strategy == BASELINE_STRATEGY
    assert receipt.treatment_strategy == TREATMENT_STRATEGY
    assert receipt.source_snapshot_digest == observation.comparison.source_snapshot_digest
    assert receipt.source_case_digest == observation.comparison.source_case_digest
    assert receipt.comparison_decision_id == observation.comparison.comparison_decision_id
    assert receipt.result_group_digest == observation.comparison.result_group.digest
    assert receipt.baseline_task_id == observation.baseline_result.task.identity.task_id
    assert receipt.treatment_task_id == observation.treatment_result.task.identity.task_id
    assert receipt.baseline_result_digest == observation.baseline_result.result_id
    assert receipt.treatment_result_digest == observation.treatment_result.result_id
    assert receipt.baseline_evidence_envelope_digest == observation.baseline_evidence.digest
    assert receipt.treatment_evidence_envelope_digest == observation.treatment_evidence.digest
    assert observation.baseline_evidence.state is EvidenceState.NOT_A_CANDIDATE
    assert observation.treatment_evidence.state is EvidenceState.NOT_A_CANDIDATE

    changed = build_contract_performance_evidence_receipt(
        replace(
            observation,
            treatment_counters=replace(
                observation.treatment_counters,
                comparison_cpu_ns=observation.treatment_counters.comparison_cpu_ns + 1,
            ),
        )
    )
    assert changed.sample_id != receipt.sample_id

    root_record = authority_module._contract_performance_dynamic_record(
        receipt,
        producer_receipt_digest=stable_digest("phase6-raw-pair-r4-producer", "v1"),
    )
    assert root_record["sample_id"] == receipt.sample_id
    assert root_record["baseline_comparison_cpu_ns"] == 10_000
    assert root_record["treatment_comparison_cpu_ns"] == 3_000


def test_changed_valid_comparison_source_derives_a_new_receipt_not_the_old_identity():
    observation = _observation()
    original = build_contract_performance_evidence_receipt(observation)
    rebound = replace(
        observation.comparison,
        source_snapshot_digest=stable_digest("phase6-raw-pair-source", "rebound"),
    )

    changed = build_contract_performance_evidence_receipt(
        replace(observation, comparison=rebound)
    )

    assert changed.source_snapshot_digest == rebound.source_snapshot_digest
    assert changed.comparison_decision_id != original.comparison_decision_id
    assert changed.sample_id != original.sample_id


@pytest.mark.parametrize(
    ("mutate", "match"),
    (
        (
            lambda value: replace(
                value,
                baseline_result=replace(value.baseline_result, case_id="other-case"),
            ),
            "case mismatch",
        ),
        (
            lambda value: replace(
                value,
                treatment_result=replace(
                    value.treatment_result,
                    task=replace(
                        value.treatment_result.task,
                        resources=replace(
                            value.treatment_result.task.resources,
                            cpu_tokens=2,
                        ),
                    ),
                ),
            ),
            "resource envelope mismatch",
        ),
        (
            lambda value: replace(
                value,
                treatment_result=replace(
                    value.treatment_result,
                    task=value.baseline_result.task,
                ),
            ),
            "task identities",
        ),
        (
            lambda value: replace(
                value,
                treatment_result=replace(
                    value.treatment_result,
                    outcome=value.baseline_result.outcome,
                ),
            ),
            "outcome identities",
        ),
        (
            lambda value: replace(
                value,
                treatment_evidence=replace(
                    value.treatment_evidence,
                    task_id=value.baseline_result.task.identity.task_id,
                ),
            ),
            "treatment evidence/task mismatch",
        ),
        (
            lambda value: replace(
                value,
                treatment_evidence=replace(
                    value.treatment_evidence,
                    execution_outcomes=(value.baseline_result.outcome,),
                ),
            ),
            "treatment evidence/outcome mismatch",
        ),
        (
            lambda value: replace(
                value,
                treatment_evidence=replace(
                    value.treatment_evidence,
                    verdict_kind=VerdictKind.VIOLATED,
                ),
            ),
            "treatment evidence verdict",
        ),
        (
            lambda value: replace(
                value,
                treatment_evidence=replace(
                    value.treatment_evidence,
                    state=EvidenceState.STABLE_SURVIVOR,
                ),
            ),
            "NOT_A_CANDIDATE",
        ),
        (
            lambda value: replace(
                value,
                baseline_counters=RawContractPerformanceCounters(
                    materialized_bytes=0,
                    backend_pair_comparisons=6,
                    comparison_cpu_ns=10_000,
                ),
            ),
            "baseline raw counters must be positive",
        ),
    ),
)
def test_typed_raw_pair_builder_fails_closed_on_single_binding_mutations(mutate, match):
    with pytest.raises(ValueError, match=match):
        mutate(_observation())


def test_typed_raw_pair_builder_rejects_result_group_substitution():
    observation = _observation()
    substituted_group = replace(
        observation.baseline_result.result_group,
        result_group_id="substituted-result-group",
    )
    substituted_result = replace(
        observation.baseline_result,
        result_group=substituted_group,
    )

    with pytest.raises(ValueError, match="baseline result group"):
        replace(observation, baseline_result=substituted_result)


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
            {"baseline_config": (("strategy", "forged"),)},
            "cannot choose a comparison strategy",
        ),
        (
            {"treatment_config": ()},
            "non-empty immutable field tuple",
        ),
    ),
)
def test_benchmark_context_rejects_ambiguous_or_strategy_spoofed_fields(kwargs, match):
    observation = _observation()
    values = {
        "benchmark_plan": observation.benchmark_context.benchmark_plan,
        "environment": observation.benchmark_context.environment,
        "baseline_config": observation.benchmark_context.baseline_config,
        "treatment_config": observation.benchmark_context.treatment_config,
    }
    values.update(kwargs)

    with pytest.raises(ValueError, match=match):
        ContractPerformanceBenchmarkContext(**values)


@pytest.mark.parametrize(
    ("values", "match"),
    (
        ((True, 1, 1), "non-negative integer"),
        ((1.0, 1, 1), "non-negative integer"),
        ((1, -1, 1), "non-negative integer"),
    ),
)
def test_raw_counter_shape_is_typed_and_fail_closed(values, match):
    with pytest.raises(ValueError, match=match):
        RawContractPerformanceCounters(
            materialized_bytes=values[0],
            backend_pair_comparisons=values[1],
            comparison_cpu_ns=values[2],
        )


def test_zero_treatment_counters_are_raw_observations_but_zero_baseline_is_not():
    observation = _observation()
    zero_treatment = replace(
        observation,
        treatment_counters=RawContractPerformanceCounters(
            materialized_bytes=0,
            backend_pair_comparisons=0,
            comparison_cpu_ns=0,
        ),
    )

    receipt = build_contract_performance_evidence_receipt(zero_treatment)
    assert receipt.treatment_materialized_bytes == 0
    assert receipt.treatment_backend_pair_comparisons == 0
    assert receipt.treatment_comparison_cpu_ns == 0


def test_observation_rejects_wrong_typed_object_before_any_receipt_exists():
    observation = _observation()

    with pytest.raises(ValueError, match="comparison receipt"):
        ContractPerformanceProductionObservation(
            comparison=object(),
            contract_comparison_producer_receipt_digest=(
                observation.contract_comparison_producer_receipt_digest
            ),
            benchmark_context=observation.benchmark_context,
            baseline_result=observation.baseline_result,
            treatment_result=observation.treatment_result,
            baseline_evidence=observation.baseline_evidence,
            treatment_evidence=observation.treatment_evidence,
            baseline_counters=observation.baseline_counters,
            treatment_counters=observation.treatment_counters,
        )
