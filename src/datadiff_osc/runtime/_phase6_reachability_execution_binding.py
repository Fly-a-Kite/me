"""Private, non-authoritative Phase-6 reachability execution context binding.

This module deliberately binds already-typed Search and Runtime objects without
running an adapter, recording a raw artifact, or issuing a coverage event.  A
valid ReachabilityExecutionBinding is diagnostic-only structural
context: it is never evidence that a target was really executed and cannot
earn Phase-6 gate credit.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

from datadiff_osc._canonical import stable_digest
from datadiff_osc.contract_engine.applicability import (
    ApplicabilityCertificate,
    applicability_binding_errors,
)
from datadiff_osc.contract_engine.evidence import ObservationCertificate
from datadiff_osc.contract_engine.model import Endpoint, HyperContract
from datadiff_osc.generation.construction import (
    ConstructionOutcome,
    fragments_for_cell,
    plan_backward,
)
from datadiff_osc.generation.extraction import AtomExtractor, ExtractionResult
from datadiff_osc.probe_contracts import (
    ObservationPolicyError,
    materialize_coverage_hypercontract,
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
    TaskIdentity,
    TaskKind,
    TaskSpec,
    VerdictKind,
)
from datadiff_osc.search.context_receipts import (
    CanonicalCaseBinding,
    FormalLanePlanReceipt,
    ObservationContextReceipt,
    ScheduledTargetAttemptReceipt,
)
from datadiff_osc.search.epochs import derive_stage_lineage
from datadiff_osc.search.matcher import TargetMatcher
from datadiff_osc.semantic_targets.compiler import compile_target_universe
from datadiff_osc.semantic_targets.declarations import legacy_v4_target_templates
from datadiff_osc.semantic_targets.model import (
    ActivationCertificate,
    BackendPairObligation,
    CompiledTargetUniverse,
    TargetAssignment,
    TargetCell,
)


REACHABILITY_EXECUTION_BINDING_SCHEMA_VERSION = (
    "osc-private-reachability-execution-binding-v1"
)
_STRUCTURAL_IO_CLASS = "osc-private-reachability-context"
_EXACT_OBSERVATION_STAGES = frozenset(
    {"S3_EXACT_MATERIALIZED", "S4_CONFIRMATION_NATIVE"}
)


@cache
def _compiled_universe() -> CompiledTargetUniverse:
    """Recompile the target universe instead of trusting a caller reference."""

    return compile_target_universe(legacy_v4_target_templates())


def _require_unique_texts(name: str, values: tuple[str, ...]) -> None:
    if (
        not isinstance(values, tuple)
        or not values
        or any(not isinstance(item, str) or not item for item in values)
        or len(values) != len(set(values))
    ):
        raise ValueError(f"{name} must be a non-empty unique text tuple")


def _require_typed_tuple(name: str, values: object, expected_type: type[object]) -> tuple:
    if not isinstance(values, tuple) or not values:
        raise ValueError(f"{name} must be a non-empty immutable tuple")
    if any(not isinstance(item, expected_type) for item in values):
        raise ValueError(f"{name} must contain only {expected_type.__name__}")
    return values


def reachability_execution_payload_digest(
    *,
    plan: FormalLanePlanReceipt,
    canonical_case: CanonicalCaseBinding,
    scheduled_attempt_id: str,
    assignment: TargetAssignment,
    target_cell_id: str,
    backend_pair: BackendPairObligation,
    contract: HyperContract,
    endpoint: Endpoint,
    seed_lineage: SeedLineage,
) -> str:
    """Return the exact private payload identity for one backend task.

    This digest names structural context only.  It is intentionally not a raw
    execution artifact or a producer-provenance assertion.
    """

    return stable_digest(
        "osc-private-reachability-execution-payload-v1",
        {
            "plan_digest": plan.digest,
            "canonical_case_binding_digest": canonical_case.digest,
            "scheduled_attempt_id": scheduled_attempt_id,
            "assignment_digest": assignment.digest,
            "target_cell_id": target_cell_id,
            "backend_pair_digest": backend_pair.digest,
            "contract_digest": contract.digest,
            "endpoint_digest": endpoint.digest,
            "seed_lineage_digest": seed_lineage.digest,
        },
    )


def reachability_execution_result_group_id(
    *,
    plan: FormalLanePlanReceipt,
    canonical_case: CanonicalCaseBinding,
    scheduled_attempt_id: str,
    assignment: TargetAssignment,
    backend_pair: BackendPairObligation,
    contract: HyperContract,
    endpoints: tuple[Endpoint, ...],
    task_ids: tuple[str, ...],
) -> str:
    """Return the deterministic group identity for this one cell/pair context."""

    return stable_digest(
        "osc-private-reachability-execution-result-group-id-v1",
        {
            "plan_digest": plan.digest,
            "canonical_case_binding_digest": canonical_case.digest,
            "scheduled_attempt_id": scheduled_attempt_id,
            "assignment_digest": assignment.digest,
            "backend_pair_digest": backend_pair.digest,
            "contract_digest": contract.digest,
            "endpoint_ids": tuple(item.endpoint_id for item in endpoints),
            "task_ids": task_ids,
        },
    )


def reachability_execution_result_order_key(
    *,
    plan: FormalLanePlanReceipt,
    canonical_case: CanonicalCaseBinding,
    scheduled_attempt_id: str,
    backend_pair: BackendPairObligation,
    contract: HyperContract,
    endpoints: tuple[Endpoint, ...],
    task_ids: tuple[str, ...],
) -> tuple[str, ...]:
    """Return an explicit deterministic order for the private result group."""

    return (
        REACHABILITY_EXECUTION_BINDING_SCHEMA_VERSION,
        plan.protocol_digest,
        plan.lane_id,
        scheduled_attempt_id,
        canonical_case.digest,
        backend_pair.obligation_id,
        contract.digest,
        *(item.endpoint_id for item in endpoints),
        *task_ids,
    )


def reachability_execution_evidence_id(
    *,
    canonical_case: CanonicalCaseBinding,
    scheduled_attempt_id: str,
    result: TypedExecutionResult,
    activation: ActivationCertificate,
    applicability: ApplicabilityCertificate,
    observation: ObservationCertificate,
) -> str:
    """Bind evidence identity to one exact typed Runtime result."""

    return stable_digest(
        "osc-private-reachability-execution-evidence-id-v1",
        {
            "canonical_case_binding_digest": canonical_case.digest,
            "scheduled_attempt_id": scheduled_attempt_id,
            "result_id": result.result_id,
            "task_spec_digest": result.task.digest,
            "seed_lineage_digest": result.seed_lineage.digest,
            "execution_outcome_digest": result.outcome.digest,
            "activation_certificate_digest": activation.digest,
            "applicability_certificate_digest": applicability.digest,
            "observation_certificate_digest": observation.digest,
        },
    )


@dataclass(frozen=True, slots=True)
class ReachabilityExecutionBinding:
    """One exact, private Search-to-Runtime context binding.

    The record is intentionally narrower than a coverage event.  It binds one
    fresh target cell and one selected fresh backend-pair obligation, but it
    never creates execution provenance, raw evidence, a candidate promotion,
    or gate authority.
    """

    scheduled_attempt: ScheduledTargetAttemptReceipt
    observation_context: ObservationContextReceipt
    canonical_case: CanonicalCaseBinding
    construction: ConstructionOutcome
    extraction: ExtractionResult
    activation: ActivationCertificate
    backend_pair: BackendPairObligation
    contract: HyperContract
    endpoints: tuple[Endpoint, ...]
    applicability: ApplicabilityCertificate
    results: tuple[TypedExecutionResult, ...]
    evidence: tuple[EvidenceEnvelope, ...]
    observation: ObservationCertificate
    schema_version: str = REACHABILITY_EXECUTION_BINDING_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != REACHABILITY_EXECUTION_BINDING_SCHEMA_VERSION:
            raise ValueError("reachability execution binding schema mismatch")
        self._validate_types()
        universe, cell = self._validate_search_and_construction_context()
        self._validate_contract_and_applicability(universe, cell)
        # Runtime task/result/evidence checks deliberately precede every
        # observation-verdict branch so a bad typed context cannot be masked.
        self._validate_runtime_and_evidence_context(cell)
        self._validate_observation_context()

    def _validate_types(self) -> None:
        typed_values = (
            ("scheduled attempt", self.scheduled_attempt, ScheduledTargetAttemptReceipt),
            ("observation context", self.observation_context, ObservationContextReceipt),
            ("canonical case", self.canonical_case, CanonicalCaseBinding),
            ("construction outcome", self.construction, ConstructionOutcome),
            ("extraction", self.extraction, ExtractionResult),
            ("activation certificate", self.activation, ActivationCertificate),
            ("backend pair", self.backend_pair, BackendPairObligation),
            ("contract", self.contract, HyperContract),
            ("applicability certificate", self.applicability, ApplicabilityCertificate),
            ("observation certificate", self.observation, ObservationCertificate),
        )
        for name, value, expected_type in typed_values:
            if not isinstance(value, expected_type):
                raise TypeError(f"{name} must be a {expected_type.__name__}")
        _require_typed_tuple("endpoints", self.endpoints, Endpoint)
        _require_typed_tuple("typed execution results", self.results, TypedExecutionResult)
        _require_typed_tuple("evidence envelopes", self.evidence, EvidenceEnvelope)

    def _validate_search_and_construction_context(
        self,
    ) -> tuple[CompiledTargetUniverse, TargetCell]:
        scheduled = self.scheduled_attempt
        context = self.observation_context
        if not scheduled.runtime_task_ref:
            raise ValueError("scheduled attempt lacks an exact runtime task reference")
        if context.plan != scheduled.plan:
            raise ValueError("observation context plan does not match scheduled attempt")
        for name in (
            "case_id",
            "case_index",
            "formal_run_id",
            "seed_block_id",
            "assignment",
        ):
            if getattr(context, name) != getattr(scheduled, name):
                raise ValueError(
                    f"observation context {name} does not match scheduled attempt"
                )
        if self.canonical_case.binding_case_id != scheduled.case_id:
            raise ValueError("canonical binding case ID does not match scheduled case")

        universe = _compiled_universe()
        cells = {item.target_cell_id: item for item in universe.fresh_cells}
        cell = cells.get(scheduled.target_cell_id)
        if cell is None:
            raise ValueError("scheduled attempt is not bound to a fresh target cell")
        if scheduled.target_family_id != cell.test_family_id:
            raise ValueError("scheduled attempt target family does not match fresh cell")
        assignment = scheduled.assignment
        if (
            assignment.selected_cell_ids != (cell.target_cell_id,)
            or assignment.selected_edge_ids
            or assignment.selected_tile_ids
            or assignment.parameters
        ):
            raise ValueError(
                "private reachability binding requires exactly one unparameterized cell"
            )
        if context.context_cell_ids != (cell.target_cell_id,):
            raise ValueError("observation context cell scope does not match scheduled cell")
        if context.context_family_ids != (cell.test_family_id,):
            raise ValueError(
                "observation context family scope does not match scheduled family"
            )
        if context.context_edge_ids:
            raise ValueError("private reachability binding cannot bind contrast edges")

        case = self.canonical_case.case
        construction = self.construction
        if (
            not construction.successful
            or construction.case is None
            or construction.certificate is None
            or construction.plan is None
            or construction.infeasible is not None
            or isinstance(construction.attempts, bool)
            or not isinstance(construction.attempts, int)
            or construction.attempts < 1
        ):
            raise ValueError("construction outcome is malformed or unresolved")
        if construction.case != case:
            raise ValueError("construction case does not match canonical case bytes")
        expected_plan = plan_backward(
            frozenset({f"oracle:{cell.observation_contract}"}),
            fragments_for_cell(cell),
        )
        if expected_plan is None or construction.plan != expected_plan:
            raise ValueError("construction plan does not exactly recompute")

        expected_extraction = AtomExtractor().extract(case)
        if self.extraction != expected_extraction:
            raise ValueError("extraction does not exactly recompute from canonical case")
        matcher = TargetMatcher(universe)
        expected_activation = matcher.match(
            assignment,
            expected_extraction,
            preflight_valid=self.activation.preflight_valid,
            mutation_preserved=self.activation.mutation_preserved,
            degraded_reasons=self.activation.degraded_reasons,
        )
        if self.activation != expected_activation:
            raise ValueError("activation does not exactly recompute from extraction")
        if not self.activation.valid:
            raise ValueError("activation certificate is not valid")
        if construction.certificate != self.activation:
            raise ValueError(
                "construction certificate does not match recomputed activation"
            )

        pairs = {
            item.obligation_id: item
            for item in universe.fresh_backend_pair_obligations
        }
        expected_pair = pairs.get(self.backend_pair.obligation_id)
        if expected_pair != self.backend_pair:
            raise ValueError("backend pair is not the exact fresh pair obligation")
        if (
            self.backend_pair.target_cell_id != cell.target_cell_id
            or self.backend_pair.observation_contract != cell.observation_contract
        ):
            raise ValueError("backend pair does not bind the scheduled target cell")
        return universe, cell

    def _validate_contract_and_applicability(
        self,
        universe: CompiledTargetUniverse,
        cell: TargetCell,
    ) -> None:
        if len(self.endpoints) != 2:
            raise ValueError("one backend pair requires exactly two endpoints")
        endpoint_ids = tuple(item.endpoint_id for item in self.endpoints)
        _require_unique_texts("endpoint IDs", endpoint_ids)
        expected_backends = (
            self.backend_pair.target_backend,
            self.backend_pair.control_backend,
        )
        if tuple(item.backend for item in self.endpoints) != expected_backends:
            raise ValueError("endpoint order does not match the selected backend pair")
        if any(
            item.case_digest != self.extraction.source_digest for item in self.endpoints
        ):
            raise ValueError("endpoint case digest does not match extracted case")
        try:
            expected_contract = materialize_coverage_hypercontract(
                universe,
                cell_ids=(cell.target_cell_id,),
                edge_ids=(),
                backend_pair_obligation_ids=(self.backend_pair.obligation_id,),
                extraction=self.extraction,
                endpoints=self.endpoints,
            )
        except ObservationPolicyError as exc:
            raise ValueError(
                f"exact private contract materialization failed: {exc}"
            ) from exc
        if self.contract != expected_contract:
            raise ValueError("contract does not equal exact pair materialization")
        binding_errors = applicability_binding_errors(
            self.contract,
            self.endpoints,
            self.applicability,
        )
        if binding_errors:
            raise ValueError(
                "applicability certificate is not exactly contract/endpoint-bound: "
                + ";".join(binding_errors)
            )
        if not self.applicability.valid:
            raise ValueError("non-applicable context is not admissible")
        observed_atoms = set(self.activation.observed_atoms)
        if not set(self.applicability.selected_atoms) <= observed_atoms:
            raise ValueError(
                "applicability selected atoms are not bound to activation evidence"
            )
        if not set(self.applicability.activated_atoms) <= observed_atoms:
            raise ValueError(
                "applicability activated atoms are not bound to activation evidence"
            )

    def _validate_runtime_and_evidence_context(self, cell: TargetCell) -> None:
        results = self.results
        evidence = self.evidence
        if len(results) != len(self.endpoints):
            raise ValueError("typed results must exactly cover the endpoint tuple")
        if len(evidence) != len(results):
            raise ValueError("evidence must exactly cover typed execution results")

        task_ids = tuple(item.task.identity.task_id for item in results)
        task_digests = tuple(item.task.digest for item in results)
        result_ids = tuple(item.result_id for item in results)
        outcome_digests = tuple(item.outcome.digest for item in results)
        endpoint_ids = tuple(item.endpoint_id for item in self.endpoints)
        result_endpoint_ids = tuple(item.outcome.endpoint_id for item in results)
        _require_unique_texts("Runtime task IDs", task_ids)
        _require_unique_texts("Runtime TaskSpec digests", task_digests)
        _require_unique_texts("Runtime result IDs", result_ids)
        _require_unique_texts("Runtime outcome digests", outcome_digests)
        if result_endpoint_ids != endpoint_ids:
            raise ValueError("typed Runtime outcomes do not preserve endpoint order")

        result_group = results[0].result_group
        if any(item.result_group != result_group for item in results):
            raise ValueError("typed Runtime results do not share one exact result group")
        backend_lineage = derive_stage_lineage(
            self.scheduled_attempt.assignment.seed_lineage,
            SeedStage.BACKEND,
        )
        for result, endpoint in zip(results, self.endpoints, strict=True):
            if result.case_id != self.canonical_case.source_case_id:
                raise ValueError("typed Runtime result case does not match canonical case")
            if result.seed_lineage != backend_lineage:
                raise ValueError("typed Runtime result does not use exact backend lineage")
            expected_identity = TaskIdentity(
                protocol_digest=self.scheduled_attempt.plan.protocol_digest,
                task_kind=TaskKind.BACKEND_EXECUTION,
                epoch_index=0,
                decision_index=self.scheduled_attempt.case_index,
                seed_lineage_digest=backend_lineage.digest,
                endpoint_id=endpoint.endpoint_id,
                backend=endpoint.backend,
                attempt=0,
            )
            expected_task = TaskSpec(
                identity=expected_identity,
                dependency_task_ids=(),
                resources=ResourceTokens(
                    cpu_tokens=1,
                    rss_bytes=0,
                    io_class=_STRUCTURAL_IO_CLASS,
                    backend_internal_threads=0,
                ),
                payload_digest=reachability_execution_payload_digest(
                    plan=self.scheduled_attempt.plan,
                    canonical_case=self.canonical_case,
                    scheduled_attempt_id=self.scheduled_attempt.attempt_id,
                    assignment=self.scheduled_attempt.assignment,
                    target_cell_id=cell.target_cell_id,
                    backend_pair=self.backend_pair,
                    contract=self.contract,
                    endpoint=endpoint,
                    seed_lineage=backend_lineage,
                ),
            )
            if result.task != expected_task:
                raise ValueError(
                    "typed Runtime task does not exactly bind Search/context inputs"
                )
            if (
                result.outcome.endpoint_id != endpoint.endpoint_id
                or result.outcome.status is not ExecutionStatus.OK
                or result.outcome.failure_kind is not FailureKind.NONE
            ):
                raise ValueError("typed Runtime outcome is not an exact OK endpoint result")

        expected_group = ResultGroup(
            result_group_id=reachability_execution_result_group_id(
                plan=self.scheduled_attempt.plan,
                canonical_case=self.canonical_case,
                scheduled_attempt_id=self.scheduled_attempt.attempt_id,
                assignment=self.scheduled_attempt.assignment,
                backend_pair=self.backend_pair,
                contract=self.contract,
                endpoints=self.endpoints,
                task_ids=task_ids,
            ),
            task_ids=task_ids,
            endpoint_ids=endpoint_ids,
            contract_fingerprint=ContractFingerprint(
                self.contract.digest,
                self.contract.registry_digest,
            ),
            target_fingerprint=self.activation.target_fingerprint,
            evidence_tier=EvidenceTier.AUDIT,
            result_order_key=reachability_execution_result_order_key(
                plan=self.scheduled_attempt.plan,
                canonical_case=self.canonical_case,
                scheduled_attempt_id=self.scheduled_attempt.attempt_id,
                backend_pair=self.backend_pair,
                contract=self.contract,
                endpoints=self.endpoints,
                task_ids=task_ids,
            ),
        )
        if result_group != expected_group:
            raise ValueError("Runtime result group does not exactly bind this context")
        if self.scheduled_attempt.runtime_task_ref != result_group.digest:
            raise ValueError(
                "scheduled runtime task reference does not equal typed result group"
            )

        context = self.observation_context
        if not context.runtime_context_complete:
            raise ValueError("opaque observation context is incomplete")
        if context.activation_certificate_ref != self.activation.digest:
            raise ValueError("observation activation reference is not exact")
        if context.runtime_task_refs != tuple(sorted(task_ids)):
            raise ValueError("observation context task refs are not exact typed tasks")
        if context.runtime_outcome_refs != tuple(sorted(outcome_digests)):
            raise ValueError(
                "observation context outcome refs are not exact typed outcomes"
            )
        if context.observation_certificate_ref != self.observation.digest:
            raise ValueError("observation certificate reference is not exact")

        evidence_ids = tuple(item.evidence_id for item in evidence)
        evidence_digests = tuple(item.digest for item in evidence)
        _require_unique_texts("Evidence IDs", evidence_ids)
        _require_unique_texts("Evidence envelope digests", evidence_digests)
        for result, envelope in zip(results, evidence, strict=True):
            if envelope.state is not EvidenceState.NOT_A_CANDIDATE:
                raise ValueError(
                    "private reachability context only admits non-candidate evidence"
                )
            expected_evidence_id = reachability_execution_evidence_id(
                canonical_case=self.canonical_case,
                scheduled_attempt_id=self.scheduled_attempt.attempt_id,
                result=result,
                activation=self.activation,
                applicability=self.applicability,
                observation=self.observation,
            )
            if envelope.evidence_id != expected_evidence_id:
                raise ValueError("evidence ID does not bind exact typed result identity")
            if (
                envelope.result_group_digest != result_group.digest
                or envelope.task_id != result.task.identity.task_id
                or envelope.seed_lineage_digest != result.seed_lineage.digest
                or envelope.contract_fingerprint
                != result_group.contract_fingerprint
                or envelope.target_fingerprint != self.activation.target_fingerprint
                or envelope.derivation_certificate_digest
                != self.contract.derivation_digest
                or envelope.applicability_certificate_digest != self.applicability.digest
                or envelope.activation_certificate_digest != self.activation.digest
                or envelope.observation_certificate_digest != self.observation.digest
                or envelope.execution_outcomes != (result.outcome,)
                or envelope.verdict_kind is not self.observation.verdict.kind
            ):
                raise ValueError(
                    "evidence envelope does not exactly bind typed Runtime context"
                )
            if envelope.artifact_refs or envelope.metadata:
                raise ValueError(
                    "private reachability context cannot carry artifact or metadata claims"
                )

    def _validate_observation_context(self) -> None:
        endpoint_ids = tuple(item.endpoint_id for item in self.endpoints)
        binding_errors = self.observation.binding_errors(
            contract_digest=self.contract.digest,
            applicability_digest=self.applicability.digest,
            endpoint_ids=endpoint_ids,
        )
        if binding_errors:
            raise ValueError(
                "observation certificate is not exactly contract/applicability-bound: "
                + ";".join(binding_errors)
            )
        if (
            not self.observation.exact_escalated
            or self.observation.comparison_stage not in _EXACT_OBSERVATION_STAGES
        ):
            raise ValueError(
                "observation context lacks exact materialized comparison authority"
            )
        if self.observation.cache_used:
            raise ValueError("private reachability context cannot use a cached result")
        if self.observation.verdict.kind in {
            VerdictKind.INAPPLICABLE,
            VerdictKind.INCONCLUSIVE,
        }:
            raise ValueError("non-final observation verdict is not admissible")
        if self.observation.verdict.kind not in {
            VerdictKind.SATISFIED,
            VerdictKind.VIOLATED,
        }:
            raise ValueError("observation verdict kind is unsupported")

    @property
    def authority_eligible(self) -> bool:
        """This private structural binding never grants authority."""

        return False

    @property
    def gate_credit(self) -> bool:
        """This private structural binding never grants Phase-6 gate credit."""

        return False

    @property
    def bug_claimed(self) -> bool:
        """No candidate or bug claim can be derived from this binding."""

        return False

    @property
    def digest(self) -> str:
        return stable_digest("osc-private-reachability-execution-binding", self)
