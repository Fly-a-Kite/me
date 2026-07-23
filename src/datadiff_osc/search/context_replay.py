"""Exact canonical replay for private Search context foundation receipts.

Replay validates intrinsic declarations and typed bindings.  Runtime-dependent
subjects intentionally remain context-required: opaque references are never
execution provenance and this module cannot grant Root or gate authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

from datadiff_osc._canonical import stable_digest, to_primitive
from datadiff_osc._replay_support import (
    ReplayValidationError,
    assert_payload_roundtrip,
    replay_enum,
    replay_immutable_value,
    require_bool,
    require_exact_mapping,
    require_nonnegative_int,
    require_text,
    require_tuple,
)
from datadiff_osc.generation.extraction import ExtractionResult
from datadiff_osc.generation.mutation import MutationOutcome
from datadiff_osc.schemas import SeedLineage, SeedStage
from datadiff_osc.search.semantic_replay import _reconstruct_search_payload
from datadiff_osc.semantic_targets.model import TargetAssignment

from .context_receipts import (
    CANONICAL_CASE_BINDING_SCHEMA_VERSION,
    FOCUS_HIT_RECEIPT_SCHEMA_VERSION,
    FORMAL_CASE_RUN_BINDING_SCHEMA_VERSION,
    FORMAL_LANE_PLAN_RECEIPT_SCHEMA_VERSION,
    FORMAL_RUN_BINDING_SCHEMA_VERSION,
    INTRINSIC_FOCUS_PROOF_SCHEMA_VERSION,
    INTRINSIC_SEARCH_UNIVERSE_SCHEMA_VERSION,
    MUTATION_ATTEMPT_RECEIPT_SCHEMA_VERSION,
    OBSERVATION_CONTEXT_RECEIPT_SCHEMA_VERSION,
    SCHEDULED_TARGET_ATTEMPT_RECEIPT_SCHEMA_VERSION,
    CanonicalCaseBinding,
    FocusHitReceipt,
    FormalCaseRunBinding,
    FormalLanePlanReceipt,
    FormalRunBinding,
    IntrinsicFocusProof,
    IntrinsicSearchUniverse,
    MutationAttemptReceipt,
    ObservationContextReceipt,
    ScheduledTargetAttemptReceipt,
    search_context_subject_spec,
)
from .lane_registry import (
    FORMAL_FOCUS_OBLIGATION_SCHEMA_VERSION,
    FORMAL_FOCUS_RULE_SCHEMA_VERSION,
    FORMAL_LANE_DECLARATION_SCHEMA_VERSION,
    FORMAL_LANE_REGISTRY_SCHEMA_VERSION,
    FormalFocusObligation,
    FormalFocusRule,
    FormalLaneDeclaration,
    FormalLaneRegistry,
)


CONTEXT_REPLAY_RESULT_SCHEMA_VERSION = "osc-private-search-context-replay-v1"


class ContextReplayStatus(str, Enum):
    VALIDATED_INTRINSIC = "validated_intrinsic"
    CONTEXT_REQUIRED = "context_required"
    NOT_EVALUATED = "not_evaluated"
    REJECTED = "rejected"


def _has_duplicate(values: tuple[object, ...]) -> bool:
    return any(value in values[:index] for index, value in enumerate(values))


@dataclass(frozen=True, slots=True)
class ContextReplayResult:
    receipt_type: str
    subject_kind: str
    status: ContextReplayStatus
    derived_subject_ids: tuple[str, ...]
    errors: tuple[str, ...]
    authority_eligible: bool = False
    schema_version: str = CONTEXT_REPLAY_RESULT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CONTEXT_REPLAY_RESULT_SCHEMA_VERSION:
            raise ValueError("Search context replay result schema mismatch")
        if self.authority_eligible:
            raise ValueError("private Search context replay cannot grant authority")
        if not self.receipt_type or not self.subject_kind:
            raise ValueError("Search context replay result identity is empty")
        if (
            _has_duplicate(self.derived_subject_ids)
            or self.derived_subject_ids != tuple(sorted(self.derived_subject_ids))
        ):
            raise ValueError("derived Search subjects must be unique and sorted")
        if self.status is ContextReplayStatus.VALIDATED_INTRINSIC and self.errors:
            raise ValueError("validated intrinsic replay cannot contain errors")
        if self.status is not ContextReplayStatus.VALIDATED_INTRINSIC and not self.errors:
            raise ValueError("non-validating replay must explain why it stopped")

    @property
    def digest(self) -> str:
        return stable_digest("osc-private-search-context-replay-result", self)

    @property
    def admitted(self) -> bool:
        return False

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(self)


def _text(value: object, path: str) -> str:
    return require_text(value, path=path)


def _optional_text(value: object, path: str) -> str:
    if not isinstance(value, str):
        raise ReplayValidationError(f"{path}: expected a string")
    return value


def _text_tuple(
    value: object,
    *,
    path: str,
    min_length: int = 0,
) -> tuple[str, ...]:
    return tuple(
        require_tuple(
            value,
            path=path,
            item_replayer=_text,
            min_length=min_length,
            unique=True,
            sorted_values=True,
        )
    )


def _text_tuple_preserve_order(
    value: object,
    *,
    path: str,
    min_length: int = 0,
) -> tuple[str, ...]:
    return tuple(
        require_tuple(
            value,
            path=path,
            item_replayer=_text,
            min_length=min_length,
            unique=True,
        )
    )


def _replay_lane_declaration(value: object, path: str) -> FormalLaneDeclaration:
    data = require_exact_mapping(
        value,
        fields={
            "lane_id",
            "target_suite",
            "preset",
            "theme",
            "legacy_default",
            "semantic_focus_families",
            "semantic_focus_signals",
            "executable_declaration_digest",
            "schema_version",
        },
        path=path,
    )
    return FormalLaneDeclaration(
        lane_id=require_text(data["lane_id"], path=f"{path}.lane_id"),
        target_suite=require_text(
            data["target_suite"], path=f"{path}.target_suite"
        ),
        preset=require_text(data["preset"], path=f"{path}.preset"),
        theme=require_text(data["theme"], path=f"{path}.theme"),
        legacy_default=require_bool(
            data["legacy_default"], path=f"{path}.legacy_default"
        ),
        semantic_focus_families=_text_tuple_preserve_order(
            data["semantic_focus_families"],
            path=f"{path}.semantic_focus_families",
            min_length=1,
        ),
        semantic_focus_signals=_text_tuple_preserve_order(
            data["semantic_focus_signals"],
            path=f"{path}.semantic_focus_signals",
            min_length=1,
        ),
        executable_declaration_digest=require_text(
            data["executable_declaration_digest"],
            path=f"{path}.executable_declaration_digest",
        ),
        schema_version=require_text(
            data["schema_version"], path=f"{path}.schema_version"
        ),
    )


def _replay_focus_obligation(value: object, path: str) -> FormalFocusObligation:
    data = require_exact_mapping(
        value,
        fields={
            "obligation_id",
            "lane_id",
            "signal_id",
            "lane_declaration_digest",
            "schema_version",
        },
        path=path,
    )
    return FormalFocusObligation(
        obligation_id=require_text(
            data["obligation_id"], path=f"{path}.obligation_id"
        ),
        lane_id=require_text(data["lane_id"], path=f"{path}.lane_id"),
        signal_id=require_text(data["signal_id"], path=f"{path}.signal_id"),
        lane_declaration_digest=require_text(
            data["lane_declaration_digest"],
            path=f"{path}.lane_declaration_digest",
        ),
        schema_version=require_text(
            data["schema_version"], path=f"{path}.schema_version"
        ),
    )


def _replay_focus_rule(value: object, path: str) -> FormalFocusRule:
    data = require_exact_mapping(
        value,
        fields={
            "rule_id",
            "signal_id",
            "source_generator",
            "required_feature",
            "source_kind",
            "schema_version",
        },
        path=path,
    )
    return FormalFocusRule(
        rule_id=require_text(data["rule_id"], path=f"{path}.rule_id"),
        signal_id=require_text(data["signal_id"], path=f"{path}.signal_id"),
        source_generator=require_text(
            data["source_generator"], path=f"{path}.source_generator"
        ),
        required_feature=require_text(
            data["required_feature"], path=f"{path}.required_feature"
        ),
        source_kind=require_text(
            data["source_kind"], path=f"{path}.source_kind"
        ),
        schema_version=require_text(
            data["schema_version"], path=f"{path}.schema_version"
        ),
    )


def _replay_registry(value: object, path: str) -> FormalLaneRegistry:
    data = require_exact_mapping(
        value,
        fields={
            "lane_declarations",
            "focus_obligations",
            "unique_focus_signal_ids",
            "schema_version",
        },
        path=path,
    )
    return FormalLaneRegistry(
        lane_declarations=tuple(
            require_tuple(
                data["lane_declarations"],
                path=f"{path}.lane_declarations",
                item_replayer=_replay_lane_declaration,
                min_length=11,
            )
        ),
        focus_obligations=tuple(
            require_tuple(
                data["focus_obligations"],
                path=f"{path}.focus_obligations",
                item_replayer=_replay_focus_obligation,
                min_length=37,
            )
        ),
        unique_focus_signal_ids=_text_tuple(
            data["unique_focus_signal_ids"],
            path=f"{path}.unique_focus_signal_ids",
            min_length=20,
        ),
        schema_version=require_text(
            data["schema_version"], path=f"{path}.schema_version"
        ),
    )


def _replay_intrinsic_universe(value: object, path: str) -> IntrinsicSearchUniverse:
    data = require_exact_mapping(
        value,
        fields={
            "registry",
            "fresh_family_ids",
            "fresh_cell_ids",
            "contrast_edge_ids",
            "backend_pair_obligation_ids",
            "schema_version",
        },
        path=path,
    )
    return IntrinsicSearchUniverse(
        registry=_replay_registry(data["registry"], f"{path}.registry"),
        fresh_family_ids=_text_tuple(
            data["fresh_family_ids"],
            path=f"{path}.fresh_family_ids",
            min_length=16,
        ),
        fresh_cell_ids=_text_tuple(
            data["fresh_cell_ids"], path=f"{path}.fresh_cell_ids", min_length=232
        ),
        contrast_edge_ids=_text_tuple(
            data["contrast_edge_ids"],
            path=f"{path}.contrast_edge_ids",
            min_length=384,
        ),
        backend_pair_obligation_ids=_text_tuple(
            data["backend_pair_obligation_ids"],
            path=f"{path}.backend_pair_obligation_ids",
            min_length=502,
        ),
        schema_version=require_text(
            data["schema_version"], path=f"{path}.schema_version"
        ),
    )


def _replay_seed_lineage(value: object, path: str) -> SeedLineage:
    data = require_exact_mapping(
        value,
        fields={
            "protocol_digest",
            "master_seed",
            "lane_id",
            "case_index",
            "stage_name",
            "counter",
            "parent_digest",
            "schema_version",
        },
        path=path,
    )
    lineage = SeedLineage(
        protocol_digest=require_text(
            data["protocol_digest"], path=f"{path}.protocol_digest"
        ),
        master_seed=require_nonnegative_int(
            data["master_seed"], path=f"{path}.master_seed"
        ),
        lane_id=require_text(data["lane_id"], path=f"{path}.lane_id"),
        case_index=require_nonnegative_int(
            data["case_index"], path=f"{path}.case_index"
        ),
        stage_name=replay_enum(
            SeedStage, data["stage_name"], path=f"{path}.stage_name"
        ),
        counter=require_nonnegative_int(
            data["counter"], path=f"{path}.counter"
        ),
        parent_digest=_optional_text(
            data["parent_digest"], f"{path}.parent_digest"
        ),
        schema_version=require_text(
            data["schema_version"], path=f"{path}.schema_version"
        ),
    )
    if lineage.schema_version != "osc-seed-lineage-v1":
        raise ReplayValidationError(f"{path}.schema_version: seed lineage mismatch")
    return lineage


def _replay_case_run_binding(value: object, path: str) -> FormalCaseRunBinding:
    data = require_exact_mapping(
        value,
        fields={"case_id", "case_index", "seed_lineage", "schema_version"},
        path=path,
    )
    return FormalCaseRunBinding(
        case_id=require_text(data["case_id"], path=f"{path}.case_id"),
        case_index=require_nonnegative_int(
            data["case_index"], path=f"{path}.case_index"
        ),
        seed_lineage=_replay_seed_lineage(
            data["seed_lineage"], f"{path}.seed_lineage"
        ),
        schema_version=require_text(
            data["schema_version"], path=f"{path}.schema_version"
        ),
    )


def _replay_run_binding(value: object, path: str) -> FormalRunBinding:
    data = require_exact_mapping(
        value,
        fields={
            "run_id",
            "protocol_digest",
            "lane_id",
            "seed_block_id",
            "master_seed",
            "run_lineage",
            "case_bindings",
            "schema_version",
        },
        path=path,
    )
    return FormalRunBinding(
        run_id=require_text(data["run_id"], path=f"{path}.run_id"),
        protocol_digest=require_text(
            data["protocol_digest"], path=f"{path}.protocol_digest"
        ),
        lane_id=require_text(data["lane_id"], path=f"{path}.lane_id"),
        seed_block_id=require_text(
            data["seed_block_id"], path=f"{path}.seed_block_id"
        ),
        master_seed=require_nonnegative_int(
            data["master_seed"], path=f"{path}.master_seed"
        ),
        run_lineage=_replay_seed_lineage(
            data["run_lineage"], f"{path}.run_lineage"
        ),
        case_bindings=tuple(
            require_tuple(
                data["case_bindings"],
                path=f"{path}.case_bindings",
                item_replayer=_replay_case_run_binding,
                min_length=1,
            )
        ),
        schema_version=require_text(
            data["schema_version"], path=f"{path}.schema_version"
        ),
    )


def _replay_plan(value: object, path: str) -> FormalLanePlanReceipt:
    data = require_exact_mapping(
        value,
        fields={
            "registry",
            "protocol_digest",
            "lane_id",
            "planned_case_ids",
            "seed_block_ids",
            "run_bindings",
            "focus_obligation_ids",
            "schema_version",
        },
        path=path,
    )
    return FormalLanePlanReceipt(
        registry=_replay_registry(data["registry"], f"{path}.registry"),
        protocol_digest=require_text(
            data["protocol_digest"], path=f"{path}.protocol_digest"
        ),
        lane_id=require_text(data["lane_id"], path=f"{path}.lane_id"),
        planned_case_ids=_text_tuple(
            data["planned_case_ids"],
            path=f"{path}.planned_case_ids",
            min_length=1,
        ),
        seed_block_ids=_text_tuple(
            data["seed_block_ids"], path=f"{path}.seed_block_ids", min_length=1
        ),
        run_bindings=tuple(
            require_tuple(
                data["run_bindings"],
                path=f"{path}.run_bindings",
                item_replayer=_replay_run_binding,
                min_length=1,
            )
        ),
        focus_obligation_ids=_text_tuple(
            data["focus_obligation_ids"],
            path=f"{path}.focus_obligation_ids",
            min_length=1,
        ),
        schema_version=require_text(
            data["schema_version"], path=f"{path}.schema_version"
        ),
    )


def _replay_search_value(
    *,
    type_name: str,
    schema_version: str,
    value: object,
    path: str,
) -> object:
    try:
        return _reconstruct_search_payload(
            envelope_type=type_name,
            schema_version=schema_version,
            payload=value,
        )
    except Exception as exc:
        raise ReplayValidationError(
            f"{path}: invalid {type_name} ({type(exc).__name__}: {exc})"
        ) from exc


def _replay_assignment(value: object, path: str) -> TargetAssignment:
    reconstructed = _replay_search_value(
        type_name="TargetAssignment",
        schema_version="osc-root-target-assignment-v1",
        value=value,
        path=path,
    )
    if not isinstance(reconstructed, TargetAssignment):
        raise ReplayValidationError(f"{path}: Search replay returned wrong type")
    return reconstructed


def _replay_scheduled(value: object, path: str) -> ScheduledTargetAttemptReceipt:
    data = require_exact_mapping(
        value,
        fields={
            "plan",
            "case_id",
            "case_index",
            "formal_run_id",
            "seed_block_id",
            "assignment",
            "target_cell_id",
            "target_family_id",
            "runtime_task_ref",
            "attempt_id",
            "schema_version",
        },
        path=path,
    )
    return ScheduledTargetAttemptReceipt(
        plan=_replay_plan(data["plan"], f"{path}.plan"),
        case_id=require_text(data["case_id"], path=f"{path}.case_id"),
        case_index=require_nonnegative_int(
            data["case_index"], path=f"{path}.case_index"
        ),
        formal_run_id=require_text(
            data["formal_run_id"], path=f"{path}.formal_run_id"
        ),
        seed_block_id=require_text(
            data["seed_block_id"], path=f"{path}.seed_block_id"
        ),
        assignment=_replay_assignment(data["assignment"], f"{path}.assignment"),
        target_cell_id=require_text(
            data["target_cell_id"], path=f"{path}.target_cell_id"
        ),
        target_family_id=require_text(
            data["target_family_id"], path=f"{path}.target_family_id"
        ),
        runtime_task_ref=_optional_text(
            data["runtime_task_ref"], f"{path}.runtime_task_ref"
        ),
        attempt_id=require_text(data["attempt_id"], path=f"{path}.attempt_id"),
        schema_version=require_text(
            data["schema_version"], path=f"{path}.schema_version"
        ),
    )


def _replay_canonical_case(value: object, path: str) -> CanonicalCaseBinding:
    data = require_exact_mapping(
        value,
        fields={
            "binding_case_id",
            "source_case_id",
            "canonical_case_json",
            "case_digest",
            "schema_version",
        },
        path=path,
    )
    raw = replay_immutable_value(
        data["canonical_case_json"], f"{path}.canonical_case_json"
    )
    if not isinstance(raw, bytes):
        raise ReplayValidationError(
            f"{path}.canonical_case_json: expected canonical bytes"
        )
    return CanonicalCaseBinding(
        binding_case_id=require_text(
            data["binding_case_id"], path=f"{path}.binding_case_id"
        ),
        source_case_id=require_text(
            data["source_case_id"], path=f"{path}.source_case_id"
        ),
        canonical_case_json=raw,
        case_digest=require_text(
            data["case_digest"], path=f"{path}.case_digest"
        ),
        schema_version=require_text(
            data["schema_version"], path=f"{path}.schema_version"
        ),
    )


def _replay_mutation_outcome(value: object, path: str) -> MutationOutcome:
    reconstructed = _replay_search_value(
        type_name="MutationOutcome",
        schema_version="osc-root-mutation-outcome-v1",
        value=value,
        path=path,
    )
    if not isinstance(reconstructed, MutationOutcome):
        raise ReplayValidationError(f"{path}: Search replay returned wrong type")
    return reconstructed


def _replay_mutation(value: object, path: str) -> MutationAttemptReceipt:
    data = require_exact_mapping(
        value,
        fields={
            "plan",
            "case_id",
            "case_index",
            "formal_run_id",
            "seed_block_id",
            "assignment",
            "source_case",
            "target_cell_id",
            "target_family_id",
            "operator_id",
            "operator_digest",
            "operator_application_policy_id",
            "mutation_lineage",
            "mutation_seed",
            "outcome",
            "activation_certificate",
            "outcome_digest",
            "runtime_task_ref",
            "attempt_id",
            "schema_version",
        },
        path=path,
    )
    outcome = _replay_mutation_outcome(data["outcome"], f"{path}.outcome")
    if data["activation_certificate"] != to_primitive(outcome.certificate):
        raise ReplayValidationError(
            f"{path}.activation_certificate: certificate/outcome payload mismatch"
        )
    return MutationAttemptReceipt(
        plan=_replay_plan(data["plan"], f"{path}.plan"),
        case_id=require_text(data["case_id"], path=f"{path}.case_id"),
        case_index=require_nonnegative_int(
            data["case_index"], path=f"{path}.case_index"
        ),
        formal_run_id=require_text(
            data["formal_run_id"], path=f"{path}.formal_run_id"
        ),
        seed_block_id=require_text(
            data["seed_block_id"], path=f"{path}.seed_block_id"
        ),
        assignment=_replay_assignment(data["assignment"], f"{path}.assignment"),
        source_case=_replay_canonical_case(
            data["source_case"], f"{path}.source_case"
        ),
        target_cell_id=require_text(
            data["target_cell_id"], path=f"{path}.target_cell_id"
        ),
        target_family_id=require_text(
            data["target_family_id"], path=f"{path}.target_family_id"
        ),
        operator_id=require_text(
            data["operator_id"], path=f"{path}.operator_id"
        ),
        operator_digest=require_text(
            data["operator_digest"], path=f"{path}.operator_digest"
        ),
        operator_application_policy_id=require_text(
            data["operator_application_policy_id"],
            path=f"{path}.operator_application_policy_id",
        ),
        mutation_lineage=_replay_seed_lineage(
            data["mutation_lineage"], f"{path}.mutation_lineage"
        ),
        mutation_seed=require_nonnegative_int(
            data["mutation_seed"], path=f"{path}.mutation_seed"
        ),
        outcome=outcome,
        activation_certificate=outcome.certificate,
        outcome_digest=require_text(
            data["outcome_digest"], path=f"{path}.outcome_digest"
        ),
        runtime_task_ref=_optional_text(
            data["runtime_task_ref"], f"{path}.runtime_task_ref"
        ),
        attempt_id=require_text(data["attempt_id"], path=f"{path}.attempt_id"),
        schema_version=require_text(
            data["schema_version"], path=f"{path}.schema_version"
        ),
    )


def _replay_observation(value: object, path: str) -> ObservationContextReceipt:
    data = require_exact_mapping(
        value,
        fields={
            "plan",
            "case_id",
            "case_index",
            "formal_run_id",
            "seed_block_id",
            "assignment",
            "context_cell_ids",
            "context_family_ids",
            "context_edge_ids",
            "activation_certificate_ref",
            "runtime_task_refs",
            "runtime_outcome_refs",
            "observation_certificate_ref",
            "context_id",
            "schema_version",
        },
        path=path,
    )
    return ObservationContextReceipt(
        plan=_replay_plan(data["plan"], f"{path}.plan"),
        case_id=require_text(data["case_id"], path=f"{path}.case_id"),
        case_index=require_nonnegative_int(
            data["case_index"], path=f"{path}.case_index"
        ),
        formal_run_id=require_text(
            data["formal_run_id"], path=f"{path}.formal_run_id"
        ),
        seed_block_id=require_text(
            data["seed_block_id"], path=f"{path}.seed_block_id"
        ),
        assignment=_replay_assignment(data["assignment"], f"{path}.assignment"),
        context_cell_ids=_text_tuple(
            data["context_cell_ids"],
            path=f"{path}.context_cell_ids",
            min_length=1,
        ),
        context_family_ids=_text_tuple(
            data["context_family_ids"],
            path=f"{path}.context_family_ids",
            min_length=1,
        ),
        context_edge_ids=_text_tuple(
            data["context_edge_ids"], path=f"{path}.context_edge_ids"
        ),
        activation_certificate_ref=_optional_text(
            data["activation_certificate_ref"],
            f"{path}.activation_certificate_ref",
        ),
        runtime_task_refs=_text_tuple(
            data["runtime_task_refs"], path=f"{path}.runtime_task_refs"
        ),
        runtime_outcome_refs=_text_tuple(
            data["runtime_outcome_refs"], path=f"{path}.runtime_outcome_refs"
        ),
        observation_certificate_ref=_optional_text(
            data["observation_certificate_ref"],
            f"{path}.observation_certificate_ref",
        ),
        context_id=require_text(data["context_id"], path=f"{path}.context_id"),
        schema_version=require_text(
            data["schema_version"], path=f"{path}.schema_version"
        ),
    )


def _replay_extraction(value: object, path: str) -> ExtractionResult:
    reconstructed = _replay_search_value(
        type_name="ExtractionResult",
        schema_version="osc-atom-extraction-v1",
        value=value,
        path=path,
    )
    if not isinstance(reconstructed, ExtractionResult):
        raise ReplayValidationError(f"{path}: Search replay returned wrong type")
    return reconstructed


def _replay_focus_proof(value: object, path: str) -> IntrinsicFocusProof:
    data = require_exact_mapping(
        value,
        fields={
            "case_id",
            "rule",
            "case_lineage",
            "source_case",
            "extraction",
            "proof_id",
            "schema_version",
        },
        path=path,
    )
    return IntrinsicFocusProof(
        case_id=require_text(data["case_id"], path=f"{path}.case_id"),
        rule=_replay_focus_rule(data["rule"], f"{path}.rule"),
        case_lineage=_replay_seed_lineage(
            data["case_lineage"], f"{path}.case_lineage"
        ),
        source_case=_replay_canonical_case(
            data["source_case"], f"{path}.source_case"
        ),
        extraction=_replay_extraction(
            data["extraction"], f"{path}.extraction"
        ),
        proof_id=require_text(data["proof_id"], path=f"{path}.proof_id"),
        schema_version=require_text(
            data["schema_version"], path=f"{path}.schema_version"
        ),
    )


def _replay_focus(value: object, path: str) -> FocusHitReceipt:
    data = require_exact_mapping(
        value,
        fields={
            "plan",
            "observation_context",
            "obligation",
            "lane_id",
            "signal_id",
            "case_id",
            "intrinsic_proofs",
            "focus_context_id",
            "schema_version",
        },
        path=path,
    )
    return FocusHitReceipt(
        plan=_replay_plan(data["plan"], f"{path}.plan"),
        observation_context=_replay_observation(
            data["observation_context"], f"{path}.observation_context"
        ),
        obligation=_replay_focus_obligation(
            data["obligation"], f"{path}.obligation"
        ),
        lane_id=require_text(data["lane_id"], path=f"{path}.lane_id"),
        signal_id=require_text(data["signal_id"], path=f"{path}.signal_id"),
        case_id=require_text(data["case_id"], path=f"{path}.case_id"),
        intrinsic_proofs=tuple(
            require_tuple(
                data["intrinsic_proofs"],
                path=f"{path}.intrinsic_proofs",
                item_replayer=_replay_focus_proof,
                min_length=1,
                unique=True,
            )
        ),
        focus_context_id=require_text(
            data["focus_context_id"], path=f"{path}.focus_context_id"
        ),
        schema_version=require_text(
            data["schema_version"], path=f"{path}.schema_version"
        ),
    )


_REPLAYERS: dict[tuple[str, str], Callable[[object, str], object]] = {
    ("FormalLaneRegistry", FORMAL_LANE_REGISTRY_SCHEMA_VERSION): _replay_registry,
    (
        "IntrinsicSearchUniverse",
        INTRINSIC_SEARCH_UNIVERSE_SCHEMA_VERSION,
    ): _replay_intrinsic_universe,
    (
        "FormalCaseRunBinding",
        FORMAL_CASE_RUN_BINDING_SCHEMA_VERSION,
    ): _replay_case_run_binding,
    ("FormalRunBinding", FORMAL_RUN_BINDING_SCHEMA_VERSION): _replay_run_binding,
    (
        "FormalLanePlanReceipt",
        FORMAL_LANE_PLAN_RECEIPT_SCHEMA_VERSION,
    ): _replay_plan,
    (
        "CanonicalCaseBinding",
        CANONICAL_CASE_BINDING_SCHEMA_VERSION,
    ): _replay_canonical_case,
    (
        "ScheduledTargetAttemptReceipt",
        SCHEDULED_TARGET_ATTEMPT_RECEIPT_SCHEMA_VERSION,
    ): _replay_scheduled,
    (
        "MutationAttemptReceipt",
        MUTATION_ATTEMPT_RECEIPT_SCHEMA_VERSION,
    ): _replay_mutation,
    (
        "ObservationContextReceipt",
        OBSERVATION_CONTEXT_RECEIPT_SCHEMA_VERSION,
    ): _replay_observation,
    (
        "IntrinsicFocusProof",
        INTRINSIC_FOCUS_PROOF_SCHEMA_VERSION,
    ): _replay_focus_proof,
    ("FocusHitReceipt", FOCUS_HIT_RECEIPT_SCHEMA_VERSION): _replay_focus,
}


def reconstruct_context_receipt(
    *,
    receipt_type: str,
    schema_version: str,
    payload: object,
) -> object:
    key = (receipt_type, schema_version)
    replayer = _REPLAYERS.get(key)
    if replayer is None:
        raise ReplayValidationError(
            f"receipt: unsupported Search context type/version {receipt_type}@{schema_version}"
        )
    reconstructed = replayer(payload, "payload")
    assert_payload_roundtrip(payload, reconstructed)
    return reconstructed


def _intrinsic_subjects(value: object, subject_kind: str) -> tuple[str, ...] | None:
    if isinstance(value, FormalLaneRegistry):
        projections = {
            "formal_lanes": tuple(sorted(value.lane_ids)),
            "focus_signals": tuple(
                sorted(item.obligation_id for item in value.focus_obligations)
            ),
            "unique_focus_signals": value.unique_focus_signal_ids,
        }
        return projections.get(subject_kind)
    if isinstance(value, IntrinsicSearchUniverse):
        projections = {
            "fresh_cells_declared": value.fresh_cell_ids,
            "contrast_edges_declared": value.contrast_edge_ids,
            "backend_pair_obligations_declared": value.backend_pair_obligation_ids,
        }
        return projections.get(subject_kind)
    if isinstance(value, FormalLanePlanReceipt) and subject_kind == "formal_lane_plans":
        return (value.lane_id,)
    if (
        isinstance(value, ScheduledTargetAttemptReceipt)
        and subject_kind == "scheduled_target_attempts"
    ):
        return (value.attempt_id,)
    if isinstance(value, MutationAttemptReceipt) and subject_kind == "mutation_attempts":
        return (value.attempt_id,)
    if (
        isinstance(value, ObservationContextReceipt)
        and subject_kind == "observation_contexts"
    ):
        return (value.context_id,)
    if isinstance(value, FocusHitReceipt) and subject_kind == "focus_hits":
        return (value.focus_context_id,)
    return None


def replay_context_admission(
    *,
    receipt_type: str,
    schema_version: str,
    payload: object,
    subject_kind: str,
    subject_ids: tuple[str, ...],
) -> ContextReplayResult:
    """Validate intrinsic subjects while retaining a non-authoritative result."""

    if not isinstance(receipt_type, str) or not receipt_type:
        return _rejected("invalid", subject_kind or "invalid", "receipt_type_invalid")
    if not isinstance(subject_kind, str) or not subject_kind:
        return _rejected(receipt_type, "invalid", "subject_kind_invalid")
    if (
        not isinstance(subject_ids, tuple)
        or not subject_ids
        or any(not isinstance(item, str) or not item for item in subject_ids)
        or _has_duplicate(subject_ids)
        or subject_ids != tuple(sorted(subject_ids))
    ):
        return _rejected(receipt_type, subject_kind, "subject_ids_invalid")
    try:
        spec = search_context_subject_spec(subject_kind)
    except KeyError:
        return _rejected(receipt_type, subject_kind, "subject_not_registered")
    if spec.producer_type != receipt_type:
        return _rejected(receipt_type, subject_kind, "subject_producer_type_mismatch")
    try:
        reconstructed = reconstruct_context_receipt(
            receipt_type=receipt_type,
            schema_version=schema_version,
            payload=payload,
        )
        derived = _intrinsic_subjects(reconstructed, subject_kind)
        if derived is None:
            return _rejected(receipt_type, subject_kind, "subject_not_intrinsic")
        derived = tuple(sorted(derived))
        if derived != subject_ids:
            return ContextReplayResult(
                receipt_type,
                subject_kind,
                ContextReplayStatus.REJECTED,
                derived,
                ("subject_identity_mismatch",),
            )
        if spec.runtime_context_required:
            return ContextReplayResult(
                receipt_type,
                subject_kind,
                ContextReplayStatus.CONTEXT_REQUIRED,
                derived,
                (f"runtime_context_required:{subject_kind}",),
            )
        return ContextReplayResult(
            receipt_type,
            subject_kind,
            ContextReplayStatus.VALIDATED_INTRINSIC,
            derived,
            (),
        )
    except Exception as exc:
        return _rejected(
            receipt_type,
            subject_kind,
            f"invalid_receipt:{type(exc).__name__}:{exc}",
        )


def _rejected(
    receipt_type: str,
    subject_kind: str,
    error: str,
) -> ContextReplayResult:
    return ContextReplayResult(
        receipt_type or "invalid",
        subject_kind or "invalid",
        ContextReplayStatus.REJECTED,
        (),
        (error,),
    )
