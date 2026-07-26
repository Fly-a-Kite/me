"""Private builders that freeze the Phase-6 v3 run-receipt digest semantics.

This module is the only construction authority for ``V3RunReceipt`` and its
bindings.  Every digest inside a built receipt is derived from complete typed
execution objects (``TypedExecutionResult``, ``EvidenceEnvelope``,
``Endpoint``, ``V3GatePlan``, ``FormalRunBinding``); no caller-supplied digest
is ever accepted.  The two previously undefined receipt fields —
``pipeline_digest`` and ``execution_run_digest`` — are frozen here under
versioned namespaces.  The module creates no authority artifact, is not wired
into gate authority, and never promotes a candidate.
"""

from __future__ import annotations

from datadiff_osc._canonical import stable_digest, to_primitive
from datadiff_osc.contract_engine import Endpoint
from datadiff_osc.runtime._private_receipts import (
    V3BackendTaskBinding,
    V3CaseBinding,
    V3RunReceipt,
)
from datadiff_osc.runtime._receipt_producers import TypedExecutionResult
from datadiff_osc.runtime.protocols import V3GatePlan
from datadiff_osc.schemas import EvidenceEnvelope, TaskKind
from datadiff_osc.search.context_receipts import FormalRunBinding


V3_CASE_PIPELINE_DIGEST_NAMESPACE = "osc-phase6-v3-case-pipeline-v1"
V3_EXECUTION_RUN_DIGEST_NAMESPACE = "osc-phase6-v3-execution-run-v1"


def _require_text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_bool(name: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _require_nonnegative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _require_unique_texts(name: str, values: object) -> tuple[str, ...]:
    if not isinstance(values, tuple) or not values:
        raise ValueError(f"{name} must be a non-empty tuple")
    if any(not isinstance(item, str) or not item for item in values):
        raise ValueError(f"{name} must contain non-empty strings")
    if len(values) != len(set(values)):
        raise ValueError(f"{name} must be unique")
    return values


def _reject_synthetic_markers(value: object) -> None:
    """Byte hygiene: no formal v3 identity or digest may carry the marker."""

    primitive = to_primitive(value)

    def _walk(item: object) -> None:
        if isinstance(item, str):
            if "synthetic" in item.lower():
                raise ValueError(
                    "synthetic marker is not admissible for formal v3 evidence"
                )
        elif isinstance(item, dict):
            for key, entry in item.items():
                _walk(key)
                _walk(entry)
        elif isinstance(item, (list, tuple)):
            for entry in item:
                _walk(entry)

    _walk(primitive)


def v3_case_pipeline_digest(
    *,
    case_id: str,
    case_task_id: str,
    result_digest: str,
    execution_outcome_digest: str,
    evidence_envelope_digest: str,
    backend_task_ids: tuple[str, ...],
) -> str:
    """Frozen v1 semantics of the v3 case ``pipeline_digest``.

    The pipeline digest is ``stable_digest`` under the frozen namespace
    ``osc-phase6-v3-case-pipeline-v1`` over the case's processing-chain
    identity, encoded as exactly this mapping:

    - ``case_id``: the canonical case identity executed inside the run;
    - ``case_task_id``: the task identity of the case-level typed execution;
    - ``result_digest``: the digest of the case's typed ``ResultGroup``;
    - ``execution_outcome_digest``: the digest of the case-level structured
      execution outcome;
    - ``evidence_envelope_digest``: the digest of the one per-case
      ``EvidenceEnvelope``;
    - ``backend_task_ids``: the sorted tuple of per-backend execution task
      identities (input order is irrelevant; the sorted order is canonical).

    Any change to this payload shape or namespace is a schema break and
    requires a new namespace version.
    """

    _require_text("case_id", case_id)
    _require_text("case_task_id", case_task_id)
    _require_text("result_digest", result_digest)
    _require_text("execution_outcome_digest", execution_outcome_digest)
    _require_text("evidence_envelope_digest", evidence_envelope_digest)
    _require_unique_texts("backend_task_ids", backend_task_ids)
    return stable_digest(
        V3_CASE_PIPELINE_DIGEST_NAMESPACE,
        {
            "case_id": case_id,
            "case_task_id": case_task_id,
            "result_digest": result_digest,
            "execution_outcome_digest": execution_outcome_digest,
            "evidence_envelope_digest": evidence_envelope_digest,
            "backend_task_ids": tuple(sorted(backend_task_ids)),
        },
    )


def v3_execution_run_digest(
    *,
    v3_gate_plan_digest: str,
    lane_id: str,
    seed: int,
    run_id: str,
    case_ids: tuple[str, ...],
    case_result_digests: tuple[str, ...],
) -> str:
    """Frozen v1 semantics of the v3 ``execution_run_digest``.

    The execution-run digest is ``stable_digest`` under the frozen namespace
    ``osc-phase6-v3-execution-run-v1`` over exactly this mapping:

    - ``v3_gate_plan_digest``: the digest of the governing ``V3GatePlan``;
    - ``lane_id``: the formal lane executed;
    - ``seed``: the non-negative master seed of the run;
    - ``run_id``: the formal run identity from the ``FormalRunBinding``;
    - ``case_ids``: the sorted tuple of the run's case identities;
    - ``case_result_digests``: the sorted tuple of the per-case typed
      ``ResultGroup`` digests (input order is irrelevant for both tuples;
      the sorted order is canonical).

    Any change to this payload shape or namespace is a schema break and
    requires a new namespace version.
    """

    _require_text("v3_gate_plan_digest", v3_gate_plan_digest)
    _require_text("lane_id", lane_id)
    _require_nonnegative_int("seed", seed)
    _require_text("run_id", run_id)
    _require_unique_texts("case_ids", case_ids)
    if not isinstance(case_result_digests, tuple) or not case_result_digests:
        raise ValueError("case_result_digests must be a non-empty tuple")
    if any(
        not isinstance(item, str) or not item for item in case_result_digests
    ):
        raise ValueError("case_result_digests must contain non-empty strings")
    if len(case_result_digests) != len(case_ids):
        raise ValueError(
            "case_result_digests must correspond one-to-one with case_ids"
        )
    return stable_digest(
        V3_EXECUTION_RUN_DIGEST_NAMESPACE,
        {
            "v3_gate_plan_digest": v3_gate_plan_digest,
            "lane_id": lane_id,
            "seed": seed,
            "run_id": run_id,
            "case_ids": tuple(sorted(case_ids)),
            "case_result_digests": tuple(sorted(case_result_digests)),
        },
    )


def build_v3_backend_task_binding(
    *,
    result: TypedExecutionResult,
    endpoint: Endpoint,
) -> V3BackendTaskBinding:
    """Derive one backend task binding from a typed per-backend execution."""

    if not isinstance(result, TypedExecutionResult):
        raise TypeError("result must be a TypedExecutionResult")
    if not isinstance(endpoint, Endpoint):
        raise TypeError("endpoint must be an Endpoint")
    identity = result.task.identity
    if identity.task_kind is not TaskKind.BACKEND_EXECUTION:
        raise ValueError(
            "v3 backend binding requires a backend execution task"
        )
    if endpoint.endpoint_id != result.outcome.endpoint_id:
        raise ValueError("v3 backend endpoint identity mismatch")
    if identity.backend and endpoint.backend != identity.backend:
        raise ValueError("v3 backend adapter identity mismatch")
    return V3BackendTaskBinding(
        task_id=identity.task_id,
        endpoint_digest=endpoint.digest,
        task_spec_digest=result.task.digest,
        execution_outcome_digest=result.outcome.digest,
        status=result.outcome.status,
    )


def build_v3_case_binding(
    *,
    case_index: int,
    case_result: TypedExecutionResult,
    evidence: EvidenceEnvelope,
    backend_results: tuple[TypedExecutionResult, ...],
    endpoints: tuple[Endpoint, ...],
    executed: bool,
    iteration_failure: bool,
    pipeline_error: bool,
) -> V3CaseBinding:
    """Assemble one case binding entirely from typed execution objects.

    Every digest in the returned binding — seed lineage, task spec, result
    group, execution outcome, evidence envelope, endpoint, and the frozen
    pipeline digest — is derived from the typed inputs after their
    cross-bindings are revalidated.  The caller supplies only the case index
    and the three raw execution flags.
    """

    if not isinstance(case_result, TypedExecutionResult):
        raise TypeError("case_result must be a TypedExecutionResult")
    if not isinstance(evidence, EvidenceEnvelope):
        raise TypeError("evidence must be an EvidenceEnvelope")
    _require_nonnegative_int("case_index", case_index)
    _require_bool("executed", executed)
    _require_bool("iteration_failure", iteration_failure)
    _require_bool("pipeline_error", pipeline_error)
    if not isinstance(backend_results, tuple) or not backend_results:
        raise ValueError("backend_results must be a non-empty tuple")
    if any(
        not isinstance(item, TypedExecutionResult) for item in backend_results
    ):
        raise ValueError(
            "backend_results must contain TypedExecutionResult values"
        )
    if not isinstance(endpoints, tuple) or not endpoints:
        raise ValueError("endpoints must be a non-empty tuple")
    if any(not isinstance(item, Endpoint) for item in endpoints):
        raise ValueError("endpoints must contain Endpoint values")
    endpoint_map = {item.endpoint_id: item for item in endpoints}
    if len(endpoint_map) != len(endpoints):
        raise ValueError("endpoints must have unique identities")

    _reject_synthetic_markers(case_result.case_id)
    group = case_result.result_group

    # Evidence envelope cross-binding (mirrors the typed candidate binding
    # invariants without pinning an evidence state).
    if evidence.result_group_digest != group.digest:
        raise ValueError("v3 case evidence/result-group mismatch")
    if evidence.task_id != case_result.task.identity.task_id:
        raise ValueError("v3 case evidence/task mismatch")
    if evidence.seed_lineage_digest != case_result.seed_lineage.digest:
        raise ValueError("v3 case evidence/seed mismatch")
    if evidence.contract_fingerprint.digest != group.contract_fingerprint.digest:
        raise ValueError("v3 case evidence/contract mismatch")
    if evidence.target_fingerprint != group.target_fingerprint:
        raise ValueError("v3 case evidence/target mismatch")
    evidence_outcomes = {
        item.endpoint_id: item for item in evidence.execution_outcomes
    }
    if evidence_outcomes.get(case_result.outcome.endpoint_id) != case_result.outcome:
        raise ValueError("v3 case evidence/outcome mismatch")

    # Per-backend cross-bindings against the one case-level result group.
    backend_endpoint_ids = tuple(
        item.outcome.endpoint_id for item in backend_results
    )
    if len(backend_endpoint_ids) != len(set(backend_endpoint_ids)):
        raise ValueError("v3 backend executions must target unique endpoints")
    if set(backend_endpoint_ids) != set(endpoint_map):
        raise ValueError(
            "endpoints must correspond one-to-one with backend executions"
        )
    bindings = []
    for backend_result in backend_results:
        if backend_result.case_id != case_result.case_id:
            raise ValueError("v3 backend execution is not bound to the case")
        if backend_result.result_group.digest != group.digest:
            raise ValueError("v3 backend execution result-group mismatch")
        if evidence_outcomes.get(backend_result.outcome.endpoint_id) != (
            backend_result.outcome
        ):
            raise ValueError("v3 backend evidence/outcome mismatch")
        bindings.append(
            build_v3_backend_task_binding(
                result=backend_result,
                endpoint=endpoint_map[backend_result.outcome.endpoint_id],
            )
        )
    backend_tasks = tuple(sorted(bindings, key=lambda item: item.task_id))
    backend_task_ids = tuple(item.task_id for item in backend_tasks)
    if len(backend_task_ids) != len(set(backend_task_ids)):
        raise ValueError("v3 backend task identities must be unique per case")

    return V3CaseBinding(
        case_id=case_result.case_id,
        case_index=case_index,
        seed_lineage_digest=case_result.seed_lineage.digest,
        case_task_id=case_result.task.identity.task_id,
        case_task_spec_digest=case_result.task.digest,
        result_digest=group.digest,
        pipeline_digest=v3_case_pipeline_digest(
            case_id=case_result.case_id,
            case_task_id=case_result.task.identity.task_id,
            result_digest=group.digest,
            execution_outcome_digest=case_result.outcome.digest,
            evidence_envelope_digest=evidence.digest,
            backend_task_ids=backend_task_ids,
        ),
        execution_outcome_digest=case_result.outcome.digest,
        evidence_envelope_digest=evidence.digest,
        executed=executed,
        iteration_failure=iteration_failure,
        pipeline_error=pipeline_error,
        backend_tasks=backend_tasks,
    )


def _derive_run_completed(cases: tuple[V3CaseBinding, ...]) -> bool:
    return all(
        case.executed and not case.iteration_failure and not case.pipeline_error
        for case in cases
    )


def build_v3_run_receipt(
    *,
    gate_plan: V3GatePlan,
    run_binding: FormalRunBinding,
    cases: tuple[V3CaseBinding, ...],
    run_completed: bool | None = None,
) -> V3RunReceipt:
    """Assemble one ``V3RunReceipt`` from exactly 100 built case bindings.

    Run-level identity (``source_digest``, ``protocol_digest``,
    ``v3_gate_plan_digest``) is derived from the typed ``V3GatePlan``;
    ``run_id``, ``lane_id``, and ``seed`` are derived from the typed
    ``FormalRunBinding``; ``run_completed`` is derived from the case flags.
    When the optional ``run_completed`` claim is passed it must match the
    derived value.  All ``V3RunReceipt`` structural invariants are enforced
    here with explicit messages before construction, and every case binding
    must recompute under the frozen pipeline-digest semantics and against the
    formal run binding's per-case seed lineages.
    """

    if not isinstance(gate_plan, V3GatePlan):
        raise TypeError("gate_plan must be a V3GatePlan")
    if not isinstance(run_binding, FormalRunBinding):
        raise TypeError("run_binding must be a FormalRunBinding")
    if run_completed is not None:
        _require_bool("run_completed", run_completed)
    if run_binding.protocol_digest != gate_plan.protocol_digest:
        raise ValueError("formal run protocol conflicts with the v3 gate plan")
    if run_binding.lane_id not in gate_plan.lane_ids:
        raise ValueError("formal run lane is not part of the v3 gate plan")
    if run_binding.master_seed not in gate_plan.seeds:
        raise ValueError("formal run seed is not part of the v3 gate plan")

    if not isinstance(cases, tuple):
        raise ValueError("cases must be an immutable tuple")
    if any(not isinstance(item, V3CaseBinding) for item in cases):
        raise ValueError("cases must contain built V3CaseBinding values")
    if len(cases) != 100:
        raise ValueError(
            "v3 run receipt builder requires exactly 100 case bindings"
        )
    ordered = tuple(sorted(cases, key=lambda item: item.case_index))
    if tuple(item.case_index for item in ordered) != tuple(range(100)):
        raise ValueError(
            "v3 case indices must cover exactly 0 through 99 without gaps"
        )
    case_ids = tuple(item.case_id for item in ordered)
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("v3 case identities must be unique")
    case_task_ids = tuple(item.case_task_id for item in ordered)
    if len(case_task_ids) != len(set(case_task_ids)):
        raise ValueError("v3 case task identities must be unique")
    backend_task_ids = tuple(
        task.task_id for case in ordered for task in case.backend_tasks
    )
    if len(backend_task_ids) != len(set(backend_task_ids)):
        raise ValueError("v3 backend task identities must be globally unique")

    for case in ordered:
        expected_pipeline = v3_case_pipeline_digest(
            case_id=case.case_id,
            case_task_id=case.case_task_id,
            result_digest=case.result_digest,
            execution_outcome_digest=case.execution_outcome_digest,
            evidence_envelope_digest=case.evidence_envelope_digest,
            backend_task_ids=tuple(
                item.task_id for item in case.backend_tasks
            ),
        )
        if case.pipeline_digest != expected_pipeline:
            raise ValueError(
                "v3 case pipeline digest does not recompute under the frozen "
                f"semantics:{case.case_id}"
            )

    bound_cases = {item.case_id: item for item in run_binding.case_bindings}
    if set(bound_cases) != set(case_ids) or len(bound_cases) != len(case_ids):
        raise ValueError(
            "formal run binding does not cover the receipt cases exactly"
        )
    for case in ordered:
        bound = bound_cases[case.case_id]
        if bound.case_index != case.case_index:
            raise ValueError(
                f"formal case index mismatch:{case.case_id}"
            )
        if bound.seed_lineage.digest != case.seed_lineage_digest:
            raise ValueError(
                "v3 case seed lineage does not recompute from the formal run "
                f"binding:{case.case_id}"
            )

    derived_run_completed = _derive_run_completed(ordered)
    if run_completed is not None and run_completed != derived_run_completed:
        raise ValueError(
            "claimed run_completed conflicts with the case execution flags"
        )

    receipt = V3RunReceipt(
        source_digest=gate_plan.source_digest,
        protocol_digest=gate_plan.protocol_digest,
        v3_gate_plan_digest=gate_plan.digest,
        run_id=run_binding.run_id,
        lane_id=run_binding.lane_id,
        seed=run_binding.master_seed,
        execution_run_digest=v3_execution_run_digest(
            v3_gate_plan_digest=gate_plan.digest,
            lane_id=run_binding.lane_id,
            seed=run_binding.master_seed,
            run_id=run_binding.run_id,
            case_ids=case_ids,
            case_result_digests=tuple(
                item.result_digest for item in ordered
            ),
        ),
        run_completed=derived_run_completed,
        cases=ordered,
    )
    _reject_synthetic_markers(receipt)
    return receipt


__all__ = [
    "V3_CASE_PIPELINE_DIGEST_NAMESPACE",
    "V3_EXECUTION_RUN_DIGEST_NAMESPACE",
    "build_v3_backend_task_binding",
    "build_v3_case_binding",
    "build_v3_run_receipt",
    "v3_case_pipeline_digest",
    "v3_execution_run_digest",
]
