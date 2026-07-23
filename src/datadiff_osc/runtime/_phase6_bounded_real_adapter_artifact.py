"""Private, diagnostic-only serialization for one bounded real adapter pair.

This module deliberately does not implement a Phase-6 producer receipt,
dynamic plan, coverage event, or gate result.  It only turns one exact private
``RealAdapterExecutionSample`` into a canonical external diagnostic artifact
after the caller has already verified a clean source snapshot.  The source
snapshot and the artifact are exact-byte bound, but neither is gate credit.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

from datadiff.family_witness_registry import latest_family_witness_registrations
from datadiff_osc._canonical import canonical_json, stable_digest
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


BOUNDED_REAL_ADAPTER_ARTIFACT_SCHEMA_VERSION = (
    "osc-private-bounded-real-adapter-artifact-v1"
)
BOUNDED_REAL_ADAPTER_PROTOCOL_DIGEST = "phase6-bounded-real-adapter-artifact-r1"
BOUNDED_REAL_ADAPTER_FAMILY_ID = "pandas_nullable_bool_reduction"
BOUNDED_REAL_ADAPTER_BACKENDS = ("pandas", "polars")
BOUNDED_REAL_ADAPTER_SEED_BLOCK_ID = "bounded-real-adapter-seed-block-r1"
BOUNDED_REAL_ADAPTER_MASTER_SEED = 29
_ARTIFACT_FILENAME = "real-adapter-execution-artifact.json"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _require_text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError(f"{name} must be non-empty text")
    return value


def _strict_json(payload: str, *, name: str) -> dict[str, Any]:
    if not isinstance(payload, str) or not payload:
        raise ValueError(f"{name} must be canonical JSON text")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{name} contains duplicate JSON fields")
            result[key] = value
        return result

    try:
        decoded = json.loads(payload, object_pairs_hook=reject_duplicates)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} is malformed canonical JSON") from exc
    if not isinstance(decoded, dict) or canonical_json(decoded) != payload:
        raise ValueError(f"{name} is not canonical JSON")
    return decoded


@dataclass(frozen=True, slots=True)
class SourceSnapshotBinding:
    """The immutable source properties needed by the private artifact only."""

    source_digest: str
    snapshot_byte_sha256: str
    git_head: str
    git_status_digest: str

    def __post_init__(self) -> None:
        for name, value in (
            ("source digest", self.source_digest),
            ("source Git status digest", self.git_status_digest),
        ):
            _require_text(value, name=name)
        _require_sha256(self.snapshot_byte_sha256, name="source snapshot SHA")
        if (
            not isinstance(self.git_head, str)
            or len(self.git_head) not in {40, 64}
            or any(character not in "0123456789abcdef" for character in self.git_head)
        ):
            raise ValueError("source Git head must be a lowercase object ID")

    @classmethod
    def from_verified(cls, verified: object) -> "SourceSnapshotBinding":
        """Require a current verified source object without exporting its type."""

        try:
            valid = verified.valid
            source_digest = verified.source_digest
            snapshot_byte_sha256 = verified.byte_sha256
            git_head = verified.git_head
            git_status_digest = verified.git_status_digest
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError("bounded artifact requires a verified source snapshot") from exc
        if valid is not True:
            raise ValueError("bounded artifact source snapshot is not verified")
        return cls(
            source_digest=source_digest,
            snapshot_byte_sha256=snapshot_byte_sha256,
            git_head=git_head,
            git_status_digest=git_status_digest,
        )

    @property
    def digest(self) -> str:
        return stable_digest("osc-private-bounded-real-adapter-source", self)


@dataclass(frozen=True, slots=True)
class BoundedAdapterRecord:
    backend: str
    adapter_type: str
    raw_summary_json: str
    raw_summary_sha256: str
    normalized_result_json: str
    normalized_result_sha256: str

    def __post_init__(self) -> None:
        _require_text(self.backend, name="artifact record backend")
        _require_text(self.adapter_type, name="artifact record adapter type")
        raw = _strict_json(self.raw_summary_json, name="artifact raw summary")
        normalized = _strict_json(
            self.normalized_result_json,
            name="artifact normalized result",
        )
        if _sha256(self.raw_summary_json.encode("utf-8")) != _require_sha256(
            self.raw_summary_sha256, name="artifact raw summary SHA"
        ):
            raise ValueError("artifact raw summary SHA mismatch")
        if _sha256(self.normalized_result_json.encode("utf-8")) != _require_sha256(
            self.normalized_result_sha256, name="artifact normalized result SHA"
        ):
            raise ValueError("artifact normalized result SHA mismatch")
        if raw.get("backend") != self.backend or normalized.get("backend") != self.backend:
            raise ValueError("artifact record backend does not match payload")
        if raw.get("status") != "ok" or normalized.get("status") != "ok":
            raise ValueError("artifact record is not an OK result")
        if raw.get("error_type", "") or raw.get("error", ""):
            raise ValueError("artifact raw summary carries an error")
        if normalized.get("error_type", "") or normalized.get("error", ""):
            raise ValueError("artifact normalized result carries an error")

    @classmethod
    def from_sample_record(cls, record: RealAdapterExecutionRecord) -> "BoundedAdapterRecord":
        if not isinstance(record, RealAdapterExecutionRecord):
            raise TypeError("artifact record requires RealAdapterExecutionRecord")
        return cls(
            backend=record.backend,
            adapter_type=record.adapter_type,
            raw_summary_json=record.raw_summary_json,
            raw_summary_sha256=record.raw_summary_sha256,
            normalized_result_json=record.normalized_result_json,
            normalized_result_sha256=record.normalized_result_sha256,
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "backend": self.backend,
            "adapter_type": self.adapter_type,
            "raw_summary_json": self.raw_summary_json,
            "raw_summary_sha256": self.raw_summary_sha256,
            "normalized_result_json": self.normalized_result_json,
            "normalized_result_sha256": self.normalized_result_sha256,
        }


@dataclass(frozen=True, slots=True)
class BoundedRealAdapterArtifact:
    source: SourceSnapshotBinding
    runner_sha256: str
    family_id: str
    target_cell_id: str
    backend_pair_obligation_id: str
    protocol_digest: str
    lane_id: str
    seed_block_id: str
    master_seed: int
    seed_lineage_digest: str
    scheduled_attempt_id: str
    source_case_id: str
    result_group_id: str
    result_group_digest: str
    task_ids: tuple[str, ...]
    result_ids: tuple[str, ...]
    outcome_digests: tuple[str, ...]
    evidence_digests: tuple[str, ...]
    execution_mode: str
    sample_id: str
    sample_digest: str
    records: tuple[BoundedAdapterRecord, ...]
    schema_version: str = BOUNDED_REAL_ADAPTER_ARTIFACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != BOUNDED_REAL_ADAPTER_ARTIFACT_SCHEMA_VERSION:
            raise ValueError("bounded artifact schema version mismatch")
        _require_sha256(self.runner_sha256, name="bounded runner SHA")
        for name, value in (
            ("bounded family", self.family_id),
            ("bounded target cell", self.target_cell_id),
            ("bounded backend obligation", self.backend_pair_obligation_id),
            ("bounded protocol", self.protocol_digest),
            ("bounded lane", self.lane_id),
            ("bounded seed block", self.seed_block_id),
            ("bounded seed lineage", self.seed_lineage_digest),
            ("bounded scheduled attempt", self.scheduled_attempt_id),
            ("bounded source case", self.source_case_id),
            ("bounded result group ID", self.result_group_id),
            ("bounded result group digest", self.result_group_digest),
            ("bounded sample ID", self.sample_id),
            ("bounded sample digest", self.sample_digest),
        ):
            _require_text(value, name=name)
        if self.family_id != BOUNDED_REAL_ADAPTER_FAMILY_ID:
            raise ValueError("bounded artifact family is not authorized")
        if self.protocol_digest != BOUNDED_REAL_ADAPTER_PROTOCOL_DIGEST:
            raise ValueError("bounded artifact protocol is not authorized")
        if self.seed_block_id != BOUNDED_REAL_ADAPTER_SEED_BLOCK_ID:
            raise ValueError("bounded artifact seed block is not authorized")
        if self.master_seed != BOUNDED_REAL_ADAPTER_MASTER_SEED:
            raise ValueError("bounded artifact master seed is not authorized")
        if self.execution_mode != DIRECT_SERIAL_UNCACHED_EXECUTION_MODE:
            raise ValueError("bounded artifact execution mode is not authorized")
        for name, values in (
            ("bounded task IDs", self.task_ids),
            ("bounded result IDs", self.result_ids),
            ("bounded outcome digests", self.outcome_digests),
            ("bounded evidence digests", self.evidence_digests),
        ):
            if (
                not isinstance(values, tuple)
                or len(values) != 2
                or any(not isinstance(item, str) or not item for item in values)
                or len(values) != len(set(values))
            ):
                raise ValueError(f"{name} must be two unique texts")
        if (
            not isinstance(self.records, tuple)
            or len(self.records) != 2
            or any(not isinstance(item, BoundedAdapterRecord) for item in self.records)
            or tuple(item.backend for item in self.records) != BOUNDED_REAL_ADAPTER_BACKENDS
        ):
            raise ValueError("bounded artifact must retain the exact pandas/polars record order")

    @property
    def artifact_id(self) -> str:
        return stable_digest("osc-private-bounded-real-adapter-artifact-id", self)

    @property
    def authority_eligible(self) -> bool:
        return False

    @property
    def gate_credit(self) -> bool:
        return False

    @property
    def dynamic_plan_emitted(self) -> bool:
        return False

    @property
    def typed_receipt_index_emitted(self) -> bool:
        return False

    @property
    def coverage_event_created(self) -> bool:
        return False

    @property
    def candidate_confirmed(self) -> bool:
        return False

    @property
    def bug_claimed(self) -> bool:
        return False

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "artifact_id": self.artifact_id,
            "source": {
                "source_digest": self.source.source_digest,
                "snapshot_byte_sha256": self.source.snapshot_byte_sha256,
                "git_head": self.source.git_head,
                "git_status_digest": self.source.git_status_digest,
            },
            "runner_sha256": self.runner_sha256,
            "family_id": self.family_id,
            "target_cell_id": self.target_cell_id,
            "backend_pair_obligation_id": self.backend_pair_obligation_id,
            "protocol_digest": self.protocol_digest,
            "lane_id": self.lane_id,
            "seed_block_id": self.seed_block_id,
            "master_seed": self.master_seed,
            "seed_lineage_digest": self.seed_lineage_digest,
            "scheduled_attempt_id": self.scheduled_attempt_id,
            "source_case_id": self.source_case_id,
            "result_group_id": self.result_group_id,
            "result_group_digest": self.result_group_digest,
            "task_ids": list(self.task_ids),
            "result_ids": list(self.result_ids),
            "outcome_digests": list(self.outcome_digests),
            "evidence_digests": list(self.evidence_digests),
            "execution_mode": self.execution_mode,
            "sample_id": self.sample_id,
            "sample_digest": self.sample_digest,
            "records": [item.to_dict() for item in self.records],
            "diagnostic_only": True,
            "authority_eligible": False,
            "gate_credit": False,
            "dynamic_plan_emitted": False,
            "typed_receipt_index_emitted": False,
            "coverage_event_created": False,
            "candidate_confirmed": False,
            "bug_claimed": False,
        }

    @classmethod
    def from_dict(cls, payload: object) -> "BoundedRealAdapterArtifact":
        if not isinstance(payload, dict):
            raise ValueError("bounded artifact payload must be a mapping")
        expected = {
            "schema_version", "artifact_id", "source", "runner_sha256", "family_id",
            "target_cell_id", "backend_pair_obligation_id", "protocol_digest", "lane_id",
            "seed_block_id", "master_seed", "seed_lineage_digest", "scheduled_attempt_id",
            "source_case_id", "result_group_id", "result_group_digest", "task_ids",
            "result_ids", "outcome_digests", "evidence_digests", "execution_mode",
            "sample_id", "sample_digest", "records", "diagnostic_only",
            "authority_eligible", "gate_credit", "dynamic_plan_emitted",
            "typed_receipt_index_emitted", "coverage_event_created", "candidate_confirmed",
            "bug_claimed",
        }
        if set(payload) != expected:
            raise ValueError("bounded artifact fields do not match schema")
        source_payload = payload["source"]
        if not isinstance(source_payload, dict) or set(source_payload) != {
            "source_digest", "snapshot_byte_sha256", "git_head", "git_status_digest"
        }:
            raise ValueError("bounded artifact source fields do not match schema")
        records = payload["records"]
        if not isinstance(records, list):
            raise ValueError("bounded artifact records must be a list")
        values = cls(
            source=SourceSnapshotBinding(**source_payload),
            runner_sha256=payload["runner_sha256"],
            family_id=payload["family_id"],
            target_cell_id=payload["target_cell_id"],
            backend_pair_obligation_id=payload["backend_pair_obligation_id"],
            protocol_digest=payload["protocol_digest"],
            lane_id=payload["lane_id"],
            seed_block_id=payload["seed_block_id"],
            master_seed=payload["master_seed"],
            seed_lineage_digest=payload["seed_lineage_digest"],
            scheduled_attempt_id=payload["scheduled_attempt_id"],
            source_case_id=payload["source_case_id"],
            result_group_id=payload["result_group_id"],
            result_group_digest=payload["result_group_digest"],
            task_ids=tuple(payload["task_ids"]),
            result_ids=tuple(payload["result_ids"]),
            outcome_digests=tuple(payload["outcome_digests"]),
            evidence_digests=tuple(payload["evidence_digests"]),
            execution_mode=payload["execution_mode"],
            sample_id=payload["sample_id"],
            sample_digest=payload["sample_digest"],
            records=tuple(BoundedAdapterRecord(**item) for item in records),
            schema_version=payload["schema_version"],
        )
        if payload["artifact_id"] != values.artifact_id:
            raise ValueError("bounded artifact identity mismatch")
        if any(
            payload[name] is not expected_value
            for name, expected_value in (
                ("diagnostic_only", True),
                ("authority_eligible", False),
                ("gate_credit", False),
                ("dynamic_plan_emitted", False),
                ("typed_receipt_index_emitted", False),
                ("coverage_event_created", False),
                ("candidate_confirmed", False),
                ("bug_claimed", False),
            )
        ):
            raise ValueError("bounded artifact authority flags are invalid")
        return values


def build_bounded_real_adapter_binding() -> ReachabilityExecutionBinding:
    """Rebuild the one authority-selected private context without adapters."""

    universe = compile_target_universe(legacy_v4_target_templates())
    cells = {item.target_cell_id: item for item in universe.fresh_cells}
    edge = next(
        item
        for item in universe.fresh_edges
        if cells[item.base_cell_id].test_family_id == BOUNDED_REAL_ADAPTER_FAMILY_ID
    )
    cell = cells[edge.base_cell_id]
    pair = next(
        item
        for item in universe.fresh_backend_pair_obligations
        if item.target_cell_id == cell.target_cell_id
    )
    if (pair.target_backend, pair.control_backend) != BOUNDED_REAL_ADAPTER_BACKENDS:
        raise ValueError("bounded real adapter pair is not the authorized pair")
    lane_id = formal_lane_registry().lane_ids[0]
    run = FormalRunBinding.build(
        protocol_digest=BOUNDED_REAL_ADAPTER_PROTOCOL_DIGEST,
        lane_id=lane_id,
        seed_block_id=BOUNDED_REAL_ADAPTER_SEED_BLOCK_ID,
        master_seed=BOUNDED_REAL_ADAPTER_MASTER_SEED,
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
    assignment = TargetAssignment((cell.target_cell_id,), case_binding.seed_lineage)
    registration = next(
        item
        for item in latest_family_witness_registrations()
        if item.family_id == cell.test_family_id
    )
    case = registration.generate_case(cell.construction_index)
    canonical_case = CanonicalCaseBinding.build(binding_case_id="case-000", case=case)
    extraction = AtomExtractor().extract(case)
    activation = TargetMatcher(universe).match(assignment, extraction)
    if not activation.valid:
        raise ValueError("bounded real adapter activation is invalid")
    construction_plan = plan_backward(
        frozenset({f"oracle:{cell.observation_contract}"}), fragments_for_cell(cell)
    )
    if construction_plan is None:
        raise ValueError("bounded real adapter construction plan is unavailable")
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
        for backend in BOUNDED_REAL_ADAPTER_BACKENDS
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
        contract, endpoints, facts=frozenset(contract.preconditions)
    )
    if not applicability.valid:
        raise ValueError("bounded real adapter applicability is invalid")
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
    if not observation.valid or observation.verdict.kind is not VerdictKind.SATISFIED:
        raise ValueError("bounded real adapter observation is invalid")
    lineage = derive_stage_lineage(assignment.seed_lineage, SeedStage.BACKEND)
    preliminary = ScheduledTargetAttemptReceipt.build(
        plan=plan, case_id="case-000", case_index=0, assignment=assignment
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
                scheduled_attempt_id=preliminary.attempt_id,
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
            scheduled_attempt_id=preliminary.attempt_id,
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
            scheduled_attempt_id=preliminary.attempt_id,
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
    scheduled = ScheduledTargetAttemptReceipt.build(
        plan=plan,
        case_id="case-000",
        case_index=0,
        assignment=assignment,
        runtime_task_ref=result_group.digest,
    )
    context = ObservationContextReceipt.build(
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
                scheduled_attempt_id=scheduled.attempt_id,
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
        scheduled_attempt=scheduled,
        observation_context=context,
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


def build_bounded_real_adapter_artifact(
    *,
    source: SourceSnapshotBinding,
    sample: RealAdapterExecutionSample,
    runner_sha256: str,
) -> BoundedRealAdapterArtifact:
    """Revalidate every private input before making serializable bytes."""

    if not isinstance(source, SourceSnapshotBinding):
        raise TypeError("bounded artifact requires SourceSnapshotBinding")
    if not isinstance(sample, RealAdapterExecutionSample):
        raise TypeError("bounded artifact requires RealAdapterExecutionSample")
    checked = RealAdapterExecutionSample(
        binding=sample.binding,
        execution_mode=sample.execution_mode,
        adapter_records=sample.adapter_records,
        schema_version=sample.schema_version,
    )
    if checked != sample:
        raise ValueError("bounded artifact sample changed during revalidation")
    binding = checked.binding
    scheduled = binding.scheduled_attempt
    run, _case = scheduled.plan.case_run_binding(scheduled.case_id, scheduled.case_index)
    if (
        scheduled.target_family_id != BOUNDED_REAL_ADAPTER_FAMILY_ID
        or (binding.backend_pair.target_backend, binding.backend_pair.control_backend)
        != BOUNDED_REAL_ADAPTER_BACKENDS
        or run.protocol_digest != BOUNDED_REAL_ADAPTER_PROTOCOL_DIGEST
        or run.seed_block_id != BOUNDED_REAL_ADAPTER_SEED_BLOCK_ID
        or run.master_seed != BOUNDED_REAL_ADAPTER_MASTER_SEED
    ):
        raise ValueError("bounded artifact sample is outside the authorized target/seed")
    results = binding.results
    result_group = results[0].result_group
    if any(item.result_group != result_group for item in results):
        raise ValueError("bounded artifact results do not share one exact result group")
    records = tuple(BoundedAdapterRecord.from_sample_record(item) for item in checked.adapter_records)
    return BoundedRealAdapterArtifact(
        source=source,
        runner_sha256=runner_sha256,
        family_id=scheduled.target_family_id,
        target_cell_id=scheduled.target_cell_id,
        backend_pair_obligation_id=binding.backend_pair.obligation_id,
        protocol_digest=run.protocol_digest,
        lane_id=run.lane_id,
        seed_block_id=run.seed_block_id,
        master_seed=run.master_seed,
        seed_lineage_digest=results[0].seed_lineage.digest,
        scheduled_attempt_id=scheduled.attempt_id,
        source_case_id=binding.canonical_case.source_case_id,
        result_group_id=result_group.result_group_id,
        result_group_digest=result_group.digest,
        task_ids=tuple(item.task.identity.task_id for item in results),
        result_ids=tuple(item.result_id for item in results),
        outcome_digests=tuple(item.outcome.digest for item in results),
        evidence_digests=tuple(item.digest for item in binding.evidence),
        execution_mode=checked.execution_mode,
        sample_id=checked.sample_id,
        sample_digest=checked.digest,
        records=records,
    )


def artifact_bytes(artifact: BoundedRealAdapterArtifact) -> bytes:
    if not isinstance(artifact, BoundedRealAdapterArtifact):
        raise TypeError("artifact bytes require BoundedRealAdapterArtifact")
    return (canonical_json(artifact.to_dict()) + "\n").encode("utf-8")


def write_bounded_real_adapter_artifact(
    *, artifact: BoundedRealAdapterArtifact, output_path: Path
) -> str:
    """Atomically create exactly one non-overwritable external artifact file."""

    if not isinstance(output_path, Path) or not output_path.is_absolute():
        raise ValueError("artifact output path must be absolute")
    if output_path.name != _ARTIFACT_FILENAME:
        raise ValueError("artifact output filename is not authorized")
    parent = output_path.parent
    if parent.is_symlink() or not parent.is_dir():
        raise ValueError("artifact output parent must be a real existing directory")
    destination = parent.resolve() / output_path.name
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"bounded artifact already exists: {destination}")
    payload = artifact_bytes(artifact)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".osc-phase6-bounded-artifact-", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"bounded artifact already exists: {destination}")
        try:
            os.link(temporary, destination, follow_symlinks=False)
        except FileExistsError as exc:
            raise FileExistsError(
                f"bounded artifact already exists: {destination}"
            ) from exc
        temporary.unlink()
        if _sha256(destination.read_bytes()) != _sha256(payload):
            raise ValueError("bounded artifact write did not preserve exact bytes")
    except BaseException:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return _sha256(payload)


def verify_bounded_real_adapter_artifact(
    *,
    artifact_path: Path,
    expected_sha256: str,
    expected_source: SourceSnapshotBinding,
    expected_runner_sha256: str,
) -> BoundedRealAdapterArtifact:
    """Re-read external bytes and reject any substitution before returning it."""

    _require_sha256(expected_sha256, name="expected artifact SHA")
    _require_sha256(expected_runner_sha256, name="expected runner SHA")
    if not isinstance(expected_source, SourceSnapshotBinding):
        raise TypeError("artifact verification requires SourceSnapshotBinding")
    if not isinstance(artifact_path, Path) or not artifact_path.is_absolute():
        raise ValueError("artifact verification path must be absolute")
    try:
        raw = artifact_path.read_bytes()
    except OSError as exc:
        raise ValueError("bounded artifact could not be read") from exc
    if _sha256(raw) != expected_sha256:
        raise ValueError("bounded artifact SHA mismatch")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("bounded artifact is not UTF-8") from exc
    if not text.endswith("\n"):
        raise ValueError("bounded artifact is missing canonical newline")
    payload = _strict_json(text[:-1], name="bounded artifact")
    artifact = BoundedRealAdapterArtifact.from_dict(payload)
    if artifact.source != expected_source:
        raise ValueError("bounded artifact source binding mismatch")
    if artifact.runner_sha256 != expected_runner_sha256:
        raise ValueError("bounded artifact runner binding mismatch")
    return artifact


def capture_and_write_bounded_real_adapter_artifact(
    *,
    source: SourceSnapshotBinding,
    runner_sha256: str,
    output_path: Path,
) -> tuple[BoundedRealAdapterArtifact, str]:
    """Execute the only future pair once, then write its diagnostic artifact."""

    binding = build_bounded_real_adapter_binding()
    sample = capture_real_adapter_execution_sample(binding)
    artifact = build_bounded_real_adapter_artifact(
        source=source, sample=sample, runner_sha256=runner_sha256
    )
    return artifact, write_bounded_real_adapter_artifact(
        artifact=artifact, output_path=output_path
    )
