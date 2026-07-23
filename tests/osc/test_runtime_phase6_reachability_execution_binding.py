from __future__ import annotations

from dataclasses import dataclass, replace

import pytest

import datadiff_osc.runtime as public_runtime
from datadiff.family_witness_registry import latest_family_witness_registrations
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
    SeedLineage,
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


@dataclass(frozen=True)
class _BaseContext:
    universe: object
    cell: object
    pair: object
    plan: FormalLanePlanReceipt
    assignment: TargetAssignment
    canonical_case: CanonicalCaseBinding
    construction: ConstructionOutcome
    extraction: object
    activation: object
    endpoints: tuple[Endpoint, ...]
    contract: object
    applicability: object
    outcomes: tuple[StructuredExecutionOutcome, ...]
    observation: object


@dataclass(frozen=True)
class _RuntimeContext:
    scheduled_attempt: ScheduledTargetAttemptReceipt
    observation_context: ObservationContextReceipt
    results: tuple[TypedExecutionResult, ...]
    evidence: tuple[EvidenceEnvelope, ...]


def _base_context() -> _BaseContext:
    templates = legacy_v4_target_templates()
    universe = compile_target_universe(templates)
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

    registry = formal_lane_registry()
    lane_id = registry.lane_ids[0]
    run = FormalRunBinding.build(
        protocol_digest="phase6-private-reachability-binding",
        lane_id=lane_id,
        seed_block_id="private-seed-block-a",
        master_seed=17,
        indexed_case_ids=(("case-000", 0),),
    )
    plan = FormalLanePlanReceipt.build(
        protocol_digest=run.protocol_digest,
        lane_id=lane_id,
        planned_case_ids=("case-000",),
        seed_block_ids=(run.seed_block_id,),
        run_bindings=(run,),
    )
    _formal_run, case_binding = plan.case_run_binding("case-000", 0)
    assignment = TargetAssignment(
        (cell.target_cell_id,),
        case_binding.seed_lineage,
    )

    registrations = {
        item.family_id: item for item in latest_family_witness_registrations()
    }
    case = registrations[cell.test_family_id].generate_case(cell.construction_index)
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
    return _BaseContext(
        universe=universe,
        cell=cell,
        pair=pair,
        plan=plan,
        assignment=assignment,
        canonical_case=canonical_case,
        construction=construction,
        extraction=extraction,
        activation=activation,
        endpoints=endpoints,
        contract=contract,
        applicability=applicability,
        outcomes=outcomes,
        observation=observation,
    )


def _runtime_context(
    base: _BaseContext,
    *,
    lineage: SeedLineage | None = None,
    observation: object | None = None,
    outcomes: tuple[StructuredExecutionOutcome, ...] | None = None,
) -> _RuntimeContext:
    actual_lineage = lineage or derive_stage_lineage(
        base.assignment.seed_lineage,
        SeedStage.BACKEND,
    )
    actual_observation = observation or base.observation
    actual_outcomes = outcomes or base.outcomes
    seed_schedule = ScheduledTargetAttemptReceipt.build(
        plan=base.plan,
        case_id="case-000",
        case_index=0,
        assignment=base.assignment,
    )
    tasks = tuple(
        TaskSpec(
            identity=TaskIdentity(
                protocol_digest=base.plan.protocol_digest,
                task_kind=TaskKind.BACKEND_EXECUTION,
                epoch_index=0,
                decision_index=0,
                seed_lineage_digest=actual_lineage.digest,
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
                plan=base.plan,
                canonical_case=base.canonical_case,
                scheduled_attempt_id=seed_schedule.attempt_id,
                assignment=base.assignment,
                target_cell_id=base.cell.target_cell_id,
                backend_pair=base.pair,
                contract=base.contract,
                endpoint=endpoint,
                seed_lineage=actual_lineage,
            ),
        )
        for endpoint in base.endpoints
    )
    task_ids = tuple(item.identity.task_id for item in tasks)
    group = ResultGroup(
        result_group_id=reachability_execution_result_group_id(
            plan=base.plan,
            canonical_case=base.canonical_case,
            scheduled_attempt_id=seed_schedule.attempt_id,
            assignment=base.assignment,
            backend_pair=base.pair,
            contract=base.contract,
            endpoints=base.endpoints,
            task_ids=task_ids,
        ),
        task_ids=task_ids,
        endpoint_ids=tuple(item.endpoint_id for item in base.endpoints),
        contract_fingerprint=ContractFingerprint(
            base.contract.digest,
            base.contract.registry_digest,
        ),
        target_fingerprint=base.activation.target_fingerprint,
        evidence_tier=EvidenceTier.AUDIT,
        result_order_key=reachability_execution_result_order_key(
            plan=base.plan,
            canonical_case=base.canonical_case,
            scheduled_attempt_id=seed_schedule.attempt_id,
            backend_pair=base.pair,
            contract=base.contract,
            endpoints=base.endpoints,
            task_ids=task_ids,
        ),
    )
    results = tuple(
        TypedExecutionResult(
            case_id=base.canonical_case.source_case_id,
            result_group=group,
            task=task,
            seed_lineage=actual_lineage,
            outcome=outcome,
        )
        for task, outcome in zip(tasks, actual_outcomes, strict=True)
    )
    scheduled_attempt = ScheduledTargetAttemptReceipt.build(
        plan=base.plan,
        case_id="case-000",
        case_index=0,
        assignment=base.assignment,
        runtime_task_ref=group.digest,
    )
    observation_context = ObservationContextReceipt.build(
        plan=base.plan,
        case_id="case-000",
        assignment=base.assignment,
        activation_certificate_ref=base.activation.digest,
        runtime_task_refs=tuple(
            sorted(item.task.identity.task_id for item in results)
        ),
        runtime_outcome_refs=tuple(sorted(item.outcome.digest for item in results)),
        observation_certificate_ref=actual_observation.digest,
    )
    evidence = tuple(
        EvidenceEnvelope.build(
            evidence_id=reachability_execution_evidence_id(
                canonical_case=base.canonical_case,
                scheduled_attempt_id=scheduled_attempt.attempt_id,
                result=result,
                activation=base.activation,
                applicability=base.applicability,
                observation=actual_observation,
            ),
            state=EvidenceState.NOT_A_CANDIDATE,
            result_group_digest=group.digest,
            task_id=result.task.identity.task_id,
            seed_lineage_digest=result.seed_lineage.digest,
            contract_fingerprint=group.contract_fingerprint,
            target_fingerprint=base.activation.target_fingerprint,
            derivation_certificate_digest=base.contract.derivation_digest,
            applicability_certificate_digest=base.applicability.digest,
            activation_certificate_digest=base.activation.digest,
            observation_certificate_digest=actual_observation.digest,
            execution_outcomes=(result.outcome,),
            verdict_kind=actual_observation.verdict.kind,
            artifact_refs=(),
            metadata={},
        )
        for result in results
    )
    return _RuntimeContext(
        scheduled_attempt=scheduled_attempt,
        observation_context=observation_context,
        results=results,
        evidence=evidence,
    )


def _binding(
    base: _BaseContext,
    *,
    runtime: _RuntimeContext | None = None,
    **changes: object,
) -> ReachabilityExecutionBinding:
    actual_runtime = runtime or _runtime_context(base)
    values: dict[str, object] = {
        "scheduled_attempt": actual_runtime.scheduled_attempt,
        "observation_context": actual_runtime.observation_context,
        "canonical_case": base.canonical_case,
        "construction": base.construction,
        "extraction": base.extraction,
        "activation": base.activation,
        "backend_pair": base.pair,
        "contract": base.contract,
        "endpoints": base.endpoints,
        "applicability": base.applicability,
        "results": actual_runtime.results,
        "evidence": actual_runtime.evidence,
        "observation": base.observation,
    }
    values.update(changes)
    return ReachabilityExecutionBinding(**values)


def _alternate_plan_context(
    base: _BaseContext,
    runtime: _RuntimeContext,
) -> ObservationContextReceipt:
    original_run = base.plan.run_bindings[0]
    alternate_run = FormalRunBinding.build(
        protocol_digest=base.plan.protocol_digest,
        lane_id=base.plan.lane_id,
        seed_block_id="private-seed-block-b",
        master_seed=original_run.master_seed,
        indexed_case_ids=(("case-000", 0),),
    )
    alternate_plan = FormalLanePlanReceipt.build(
        protocol_digest=base.plan.protocol_digest,
        lane_id=base.plan.lane_id,
        planned_case_ids=("case-000",),
        seed_block_ids=(alternate_run.seed_block_id,),
        run_bindings=(alternate_run,),
    )
    return ObservationContextReceipt.build(
        plan=alternate_plan,
        case_id="case-000",
        assignment=base.assignment,
        activation_certificate_ref=base.activation.digest,
        runtime_task_refs=tuple(
            sorted(item.task.identity.task_id for item in runtime.results)
        ),
        runtime_outcome_refs=tuple(
            sorted(item.outcome.digest for item in runtime.results)
        ),
        observation_certificate_ref=base.observation.digest,
    )


def test_private_binding_is_exact_but_never_authoritative():
    base = _base_context()
    binding = _binding(base)

    assert binding.authority_eligible is False
    assert binding.gate_credit is False
    assert binding.bug_claimed is False
    assert binding.digest
    assert "ReachabilityExecutionBinding" not in public_runtime.__all__
    assert not hasattr(public_runtime, "ReachabilityExecutionBinding")


def test_negative_review_rejects_search_contract_endpoint_and_opaque_ref_substitution():
    base = _base_context()
    runtime = _runtime_context(base)

    with pytest.raises(ValueError, match="plan does not match"):
        _binding(
            base,
            runtime=runtime,
            observation_context=_alternate_plan_context(base, runtime),
        )

    opaque_context = ObservationContextReceipt.build(
        plan=base.plan,
        case_id="case-000",
        assignment=base.assignment,
        activation_certificate_ref="opaque-activation",
        runtime_task_refs=("opaque-task",),
        runtime_outcome_refs=("opaque-outcome",),
        observation_certificate_ref="opaque-observation",
    )
    with pytest.raises(ValueError, match="activation reference"):
        _binding(base, runtime=runtime, observation_context=opaque_context)

    alternate_cell = next(
        item
        for item in base.universe.fresh_cells
        if item.target_cell_id != base.cell.target_cell_id
    )
    alternate_assignment = TargetAssignment(
        (alternate_cell.target_cell_id,),
        base.assignment.seed_lineage,
    )
    alternate_schedule = ScheduledTargetAttemptReceipt.build(
        plan=base.plan,
        case_id="case-000",
        case_index=0,
        assignment=alternate_assignment,
        runtime_task_ref=runtime.scheduled_attempt.runtime_task_ref,
    )
    with pytest.raises(ValueError, match="assignment does not match"):
        _binding(base, runtime=runtime, scheduled_attempt=alternate_schedule)

    alternate_pair = next(
        item
        for item in base.universe.fresh_backend_pair_obligations
        if item.obligation_id != base.pair.obligation_id
    )
    with pytest.raises(ValueError, match="backend pair"):
        _binding(base, runtime=runtime, backend_pair=alternate_pair)

    with pytest.raises(ValueError):
        _binding(
            base,
            runtime=runtime,
            contract=replace(base.contract, strength="tampered-strength"),
        )

    with pytest.raises(ValueError):
        _binding(
            base,
            runtime=runtime,
            endpoints=(
                replace(base.endpoints[0], adapter_revision="tampered-adapter"),
                base.endpoints[1],
            ),
        )


def test_independent_counterexamples_reject_typed_runtime_before_verdict_masking():
    base = _base_context()
    runtime = _runtime_context(base)

    bad_task = replace(runtime.results[0].task, payload_digest="tampered-payload")
    bad_result = replace(runtime.results[0], task=bad_task)
    inconclusive = replace(
        base.observation,
        verdict=Verdict(VerdictKind.INCONCLUSIVE, "would be a verdict mask"),
    )
    with pytest.raises(ValueError, match="typed Runtime task"):
        _binding(
            base,
            runtime=runtime,
            results=(bad_result, runtime.results[1]),
            observation=inconclusive,
        )

    wrong_stage_runtime = _runtime_context(
        base,
        lineage=derive_stage_lineage(
            base.assignment.seed_lineage,
            SeedStage.MUTATION,
        ),
    )
    with pytest.raises(ValueError, match="exact backend lineage"):
        _binding(base, runtime=wrong_stage_runtime)

    non_ok_outcomes = (
        StructuredExecutionOutcome(
            endpoint_id=base.endpoints[0].endpoint_id,
            status=ExecutionStatus.SEMANTIC_ERROR,
            failure_kind=FailureKind.SEMANTIC_DOMAIN_ERROR,
        ),
        base.outcomes[1],
    )
    non_ok_runtime = _runtime_context(base, outcomes=non_ok_outcomes)
    with pytest.raises(ValueError, match="exact OK endpoint result"):
        _binding(base, runtime=non_ok_runtime)

    with pytest.raises(ValueError, match="Runtime task IDs"):
        _binding(
            base,
            runtime=runtime,
            results=(runtime.results[0], runtime.results[0]),
            evidence=(runtime.evidence[0], runtime.evidence[0]),
        )

    with pytest.raises(ValueError, match="Evidence IDs"):
        _binding(
            base,
            runtime=runtime,
            evidence=(runtime.evidence[0], runtime.evidence[0]),
        )

    with pytest.raises(ValueError, match="non-candidate evidence"):
        _binding(
            base,
            runtime=runtime,
            evidence=(
                replace(
                    runtime.evidence[0],
                    state=EvidenceState.INDEPENDENTLY_CONFIRMED,
                ),
                runtime.evidence[1],
            ),
        )


@pytest.mark.parametrize(
    "kind",
    (VerdictKind.INAPPLICABLE, VerdictKind.INCONCLUSIVE),
)
def test_exceptional_input_rejects_nonfinal_or_unelevated_observation(kind):
    base = _base_context()
    nonfinal_observation = replace(
        base.observation,
        verdict=Verdict(kind, "not admissible for private context"),
    )
    nonfinal_runtime = _runtime_context(base, observation=nonfinal_observation)
    with pytest.raises(ValueError, match="non-final observation verdict"):
        _binding(
            base,
            runtime=nonfinal_runtime,
            observation=nonfinal_observation,
        )

    unelevated_violated = replace(
        base.observation,
        verdict=Verdict(VerdictKind.VIOLATED, "not exact"),
        comparison_stage="S2_COMPONENT_FINGERPRINT",
        exact_escalated=False,
        materialized_endpoint_ids=(),
    )
    unelevated_runtime = _runtime_context(base, observation=unelevated_violated)
    with pytest.raises(ValueError, match="lacks exact materialized"):
        _binding(
            base,
            runtime=unelevated_runtime,
            observation=unelevated_violated,
        )


def test_exceptional_input_rejects_malformed_construction_extraction_and_activation():
    base = _base_context()
    runtime = _runtime_context(base)

    with pytest.raises(ValueError, match="malformed or unresolved"):
        _binding(
            base,
            runtime=runtime,
            construction=replace(base.construction, case=None),
        )
    with pytest.raises(ValueError, match="malformed or unresolved"):
        _binding(
            base,
            runtime=runtime,
            construction=replace(base.construction, plan=None),
        )
    with pytest.raises(ValueError, match="extraction does not exactly recompute"):
        _binding(
            base,
            runtime=runtime,
            extraction=replace(base.extraction, source_digest="extraction-drift"),
        )
    with pytest.raises(ValueError, match="activation certificate is not valid"):
        _binding(
            base,
            runtime=runtime,
            activation=replace(base.activation, mutation_preserved=False),
        )
