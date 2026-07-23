from __future__ import annotations

from dataclasses import replace
import hashlib
import json

import pytest

import datadiff_osc.runtime as public_runtime
from datadiff.family_witness_registry import latest_family_witness_registrations
from datadiff_osc._canonical import canonical_json
from datadiff_osc.contract_engine import (
    Endpoint,
    Observation,
    SchemaField,
    Verdict,
    evaluate_applicability,
    execute_staged_comparison,
)
from datadiff_osc.generation.construction import (
    ConstructionOutcome,
    fragments_for_cell,
    plan_backward,
)
from datadiff_osc.generation.extraction import AtomExtractor
from datadiff_osc.probe_contracts import materialize_coverage_hypercontract
from datadiff_osc.runtime._phase6_real_adapter_execution_sample import (
    DIRECT_SERIAL_UNCACHED_EXECUTION_MODE,
    RealAdapterExecutionRecord,
    RealAdapterExecutionSample,
    _close_adapters,
    capture_real_adapter_execution_sample,
)
from datadiff_osc.runtime._phase6_reachability_execution_binding import (
    ReachabilityExecutionBinding,
    reachability_execution_evidence_id,
    reachability_execution_payload_digest,
    reachability_execution_result_group_id,
    reachability_execution_result_order_key,
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
    SeedStage,
    StructuredExecutionOutcome,
    TaskIdentity,
    TaskKind,
    TaskSpec,
    VerdictKind,
)
from datadiff_osc.search.context_receipts import (
    CanonicalCaseBinding,
    FormalLanePlanReceipt,
    FormalRunBinding,
    ObservationContextReceipt,
    ScheduledTargetAttemptReceipt,
)
from datadiff_osc.search.epochs import derive_stage_lineage
from datadiff_osc.search.lane_registry import formal_lane_registry
from datadiff_osc.search.matcher import TargetMatcher
from datadiff_osc.semantic_targets.compiler import compile_target_universe
from datadiff_osc.semantic_targets.declarations import legacy_v4_target_templates
from datadiff_osc.semantic_targets.model import TargetAssignment


def _fresh_binding() -> ReachabilityExecutionBinding:
    """Build one valid R1 context without invoking any adapter."""

    universe = compile_target_universe(legacy_v4_target_templates())
    cells_by_id = {item.target_cell_id: item for item in universe.fresh_cells}
    edge = next(
        item
        for item in universe.fresh_edges
        if cells_by_id[item.base_cell_id].test_family_id
        == "pandas_nullable_bool_reduction"
    )
    cell = cells_by_id[edge.base_cell_id]
    pair = next(
        item
        for item in universe.fresh_backend_pair_obligations
        if item.target_cell_id == cell.target_cell_id
    )
    assert (pair.target_backend, pair.control_backend) == ("pandas", "polars")

    lane_id = formal_lane_registry().lane_ids[0]
    run = FormalRunBinding.build(
        protocol_digest="phase6-private-real-adapter-diagnostic-r2",
        lane_id=lane_id,
        seed_block_id="private-real-adapter-seed-block-a",
        master_seed=29,
        indexed_case_ids=(("case-000", 0),),
    )
    plan = FormalLanePlanReceipt.build(
        protocol_digest=run.protocol_digest,
        lane_id=lane_id,
        planned_case_ids=("case-000",),
        seed_block_ids=(run.seed_block_id,),
        run_bindings=(run,),
    )
    _run, case_binding = plan.case_run_binding("case-000", 0)
    assignment = TargetAssignment(
        (cell.target_cell_id,),
        case_binding.seed_lineage,
    )
    registration = next(
        item
        for item in latest_family_witness_registrations()
        if item.family_id == cell.test_family_id
    )
    case = registration.generate_case(cell.construction_index)
    canonical_case = CanonicalCaseBinding.build(binding_case_id="case-000", case=case)
    extraction = AtomExtractor().extract(case)
    activation = TargetMatcher(universe).match(assignment, extraction)
    assert activation.valid
    construction_plan = plan_backward(
        frozenset({f"oracle:{cell.observation_contract}"}),
        fragments_for_cell(cell),
    )
    assert construction_plan is not None
    construction = ConstructionOutcome(
        case=case,
        certificate=activation,
        plan=construction_plan,
        infeasible=None,
        attempts=1,
    )
    endpoints = tuple(
        Endpoint.build(
            endpoint_id=f"{cell.target_cell_id[-12:]}:{backend}",
            case_digest=extraction.source_digest,
            backend=backend,
            backend_version="1.0.0",
            adapter_revision="adapter-v1",
            execution_mode=cell.coordinate_map.get(
                "execution_mode", cell.coordinate_map.get("mode", "eager")
            ),
            physical_layout=cell.coordinate_map.get(
                "physical_layout", cell.coordinate_map.get("layout", "contiguous")
            ),
            capabilities=cell.required_capabilities,
        )
        for backend in (pair.target_backend, pair.control_backend)
    )
    contract = materialize_coverage_hypercontract(
        universe,
        cell_ids=(cell.target_cell_id,),
        edge_ids=(),
        backend_pair_obligation_ids=(pair.obligation_id,),
        extraction=extraction,
        endpoints=endpoints,
    )
    applicability = evaluate_applicability(
        contract,
        endpoints,
        facts=frozenset(contract.preconditions),
    )
    assert applicability.valid
    outcomes = tuple(
        StructuredExecutionOutcome(
            endpoint_id=endpoint.endpoint_id,
            status=ExecutionStatus.OK,
            failure_kind=FailureKind.NONE,
        )
        for endpoint in endpoints
    )
    observations = tuple(
        Observation.build(
            endpoint_id=endpoint.endpoint_id,
            status="ok",
            schema=(SchemaField("x", "int", False),),
            rows=[[1], [2]],
            execution_metadata={"endpoint_digest": endpoint.digest},
        )
        for endpoint in endpoints
    )
    observation = execute_staged_comparison(
        contract,
        observations,
        applicability,
        evidence_tier="audit",
        endpoints=endpoints,
        force_exact=True,
        execution_outcomes=outcomes,
    )
    assert observation.valid
    assert observation.verdict.kind is VerdictKind.SATISFIED

    lineage = derive_stage_lineage(assignment.seed_lineage, SeedStage.BACKEND)
    preliminary_schedule = ScheduledTargetAttemptReceipt.build(
        plan=plan,
        case_id="case-000",
        case_index=0,
        assignment=assignment,
    )
    tasks = tuple(
        TaskSpec(
            identity=TaskIdentity(
                protocol_digest=plan.protocol_digest,
                task_kind=TaskKind.BACKEND_EXECUTION,
                epoch_index=0,
                decision_index=0,
                seed_lineage_digest=lineage.digest,
                endpoint_id=endpoint.endpoint_id,
                backend=endpoint.backend,
                attempt=0,
            ),
            dependency_task_ids=(),
            resources=ResourceTokens(
                cpu_tokens=1,
                rss_bytes=0,
                io_class="osc-private-reachability-context",
                backend_internal_threads=0,
            ),
            payload_digest=reachability_execution_payload_digest(
                plan=plan,
                canonical_case=canonical_case,
                scheduled_attempt_id=preliminary_schedule.attempt_id,
                assignment=assignment,
                target_cell_id=cell.target_cell_id,
                backend_pair=pair,
                contract=contract,
                endpoint=endpoint,
                seed_lineage=lineage,
            ),
        )
        for endpoint in endpoints
    )
    task_ids = tuple(item.identity.task_id for item in tasks)
    result_group = ResultGroup(
        result_group_id=reachability_execution_result_group_id(
            plan=plan,
            canonical_case=canonical_case,
            scheduled_attempt_id=preliminary_schedule.attempt_id,
            assignment=assignment,
            backend_pair=pair,
            contract=contract,
            endpoints=endpoints,
            task_ids=task_ids,
        ),
        task_ids=task_ids,
        endpoint_ids=tuple(item.endpoint_id for item in endpoints),
        contract_fingerprint=ContractFingerprint(contract.digest, contract.registry_digest),
        target_fingerprint=activation.target_fingerprint,
        evidence_tier=EvidenceTier.AUDIT,
        result_order_key=reachability_execution_result_order_key(
            plan=plan,
            canonical_case=canonical_case,
            scheduled_attempt_id=preliminary_schedule.attempt_id,
            backend_pair=pair,
            contract=contract,
            endpoints=endpoints,
            task_ids=task_ids,
        ),
    )
    results = tuple(
        TypedExecutionResult(
            case_id=canonical_case.source_case_id,
            result_group=result_group,
            task=task,
            seed_lineage=lineage,
            outcome=outcome,
        )
        for task, outcome in zip(tasks, outcomes, strict=True)
    )
    scheduled_attempt = ScheduledTargetAttemptReceipt.build(
        plan=plan,
        case_id="case-000",
        case_index=0,
        assignment=assignment,
        runtime_task_ref=result_group.digest,
    )
    observation_context = ObservationContextReceipt.build(
        plan=plan,
        case_id="case-000",
        assignment=assignment,
        activation_certificate_ref=activation.digest,
        runtime_task_refs=tuple(sorted(item.task.identity.task_id for item in results)),
        runtime_outcome_refs=tuple(sorted(item.outcome.digest for item in results)),
        observation_certificate_ref=observation.digest,
    )
    evidence = tuple(
        EvidenceEnvelope.build(
            evidence_id=reachability_execution_evidence_id(
                canonical_case=canonical_case,
                scheduled_attempt_id=scheduled_attempt.attempt_id,
                result=result,
                activation=activation,
                applicability=applicability,
                observation=observation,
            ),
            state=EvidenceState.NOT_A_CANDIDATE,
            result_group_digest=result_group.digest,
            task_id=result.task.identity.task_id,
            seed_lineage_digest=result.seed_lineage.digest,
            contract_fingerprint=result_group.contract_fingerprint,
            target_fingerprint=activation.target_fingerprint,
            derivation_certificate_digest=contract.derivation_digest,
            applicability_certificate_digest=applicability.digest,
            activation_certificate_digest=activation.digest,
            observation_certificate_digest=observation.digest,
            execution_outcomes=(result.outcome,),
            verdict_kind=observation.verdict.kind,
            artifact_refs=(),
            metadata={},
        )
        for result in results
    )
    return ReachabilityExecutionBinding(
        scheduled_attempt=scheduled_attempt,
        observation_context=observation_context,
        canonical_case=canonical_case,
        construction=construction,
        extraction=extraction,
        activation=activation,
        backend_pair=pair,
        contract=contract,
        endpoints=endpoints,
        applicability=applicability,
        results=results,
        evidence=evidence,
        observation=observation,
    )


@pytest.fixture(scope="module")
def real_sample() -> RealAdapterExecutionSample:
    """The single actual pair invocation for this entire scoped suite."""

    return capture_real_adapter_execution_sample(_fresh_binding())


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_real_adapter_sample_is_private_and_exact(real_sample):
    assert tuple(record.backend for record in real_sample.adapter_records) == (
        "pandas",
        "polars",
    )
    assert all(record.adapter_type for record in real_sample.adapter_records)
    assert all(
        json.loads(record.raw_summary_json)["status"] == "ok"
        and json.loads(record.normalized_result_json)["status"] == "ok"
        for record in real_sample.adapter_records
    )
    assert real_sample.execution_mode == DIRECT_SERIAL_UNCACHED_EXECUTION_MODE
    assert real_sample.sample_id
    assert real_sample.digest
    assert real_sample.authority_eligible is False
    assert real_sample.gate_credit is False
    assert real_sample.raw_artifact_admitted is False
    assert real_sample.coverage_event_created is False
    assert real_sample.candidate_confirmed is False
    assert real_sample.bug_claimed is False
    assert "RealAdapterExecutionSample" not in public_runtime.__all__
    assert not hasattr(public_runtime, "RealAdapterExecutionSample")


def test_negative_review_rejects_reordered_or_tampered_real_records(real_sample):
    first, second = real_sample.adapter_records

    with pytest.raises(ValueError, match="record order"):
        replace(real_sample, adapter_records=(second, first))

    with pytest.raises(ValueError, match="raw summary SHA mismatch"):
        replace(first, raw_summary_sha256="0" * 64)

    with pytest.raises(ValueError, match="malformed canonical JSON"):
        replace(first, raw_summary_json="{not-json}", raw_summary_sha256=_sha256("{not-json}"))

    raw_payload = json.loads(first.raw_summary_json)
    raw_payload["status"] = "error"
    non_ok_raw = canonical_json(raw_payload)
    with pytest.raises(ValueError, match="status is not OK"):
        replace(
            first,
            raw_summary_json=non_ok_raw,
            raw_summary_sha256=_sha256(non_ok_raw),
        )

    raw_payload["status"] = "ok"
    raw_payload["backend"] = "unexpected"
    unexpected_raw = canonical_json(raw_payload)
    normalized_payload = json.loads(first.normalized_result_json)
    normalized_payload["backend"] = "unexpected"
    unexpected_normalized = canonical_json(normalized_payload)
    unexpected_record = RealAdapterExecutionRecord(
        backend="unexpected",
        adapter_type=first.adapter_type,
        raw_summary_json=unexpected_raw,
        raw_summary_sha256=_sha256(unexpected_raw),
        normalized_result_json=unexpected_normalized,
        normalized_result_sha256=_sha256(unexpected_normalized),
    )
    with pytest.raises(ValueError, match="record order"):
        replace(real_sample, adapter_records=(unexpected_record, second))


def test_independent_counterexamples_keep_structural_context_non_authoritative(real_sample):
    with pytest.raises(ValueError, match="exactly two typed records"):
        RealAdapterExecutionSample(
            binding=real_sample.binding,
            execution_mode=DIRECT_SERIAL_UNCACHED_EXECUTION_MODE,
            adapter_records=(),
        )
    with pytest.raises(TypeError, match="requires ReachabilityExecutionBinding"):
        capture_real_adapter_execution_sample(object())  # type: ignore[arg-type]

    stale_binding = _fresh_binding()
    stale_case = stale_binding.canonical_case.case
    stale_case.case_id = "stale-private-r2-case"
    object.__setattr__(
        stale_binding,
        "canonical_case",
        CanonicalCaseBinding.build(binding_case_id="case-000", case=stale_case),
    )
    with pytest.raises(ValueError, match="not exact R1 context"):
        capture_real_adapter_execution_sample(stale_binding)


def test_exceptional_inputs_fail_closed_without_a_second_adapter_execution():
    class _CloseFailure:
        def close(self):
            raise RuntimeError("close failure")

    with pytest.raises(ValueError, match="real adapter close failed"):
        _close_adapters((_CloseFailure(),))
