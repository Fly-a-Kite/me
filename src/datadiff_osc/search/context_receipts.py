"""Private intrinsic Search receipts for the Phase-6 context foundation.

The values below establish exact Search-local identities only.  Opaque Runtime
references remain comparison-only and no receipt in this module proves that a
task ran, an outcome was observed, a focus hit was executed, or a gate passed.
"""

from __future__ import annotations

import copy
from contextvars import Context, ContextVar
from dataclasses import dataclass, fields, is_dataclass
import builtins
import dis
from enum import Enum
from functools import cache
import hashlib
import inspect
import json
import math
from pathlib import Path
import random
import sys
from types import CodeType, MappingProxyType, ModuleType
from typing import Any, Callable, Iterable, TypeVar

from datadiff.datagen import generate_case
from datadiff.dsl import Case, Program
from datadiff.guidance import extract_case_features
from datadiff.mutator import ALL_MUTATION_OPERATOR_PROFILES, MutationOperator

from datadiff_osc._canonical import canonical_json, stable_digest, to_primitive
from datadiff_osc.generation.extraction import AtomExtractor, ExtractionResult
from datadiff_osc.generation.mutation import MutationOutcome, TargetPreservingMutator
from datadiff_osc.schemas import SeedLineage, SeedStage
from datadiff_osc.search.epochs import derive_stage_lineage
from datadiff_osc.search.semantic_replay import (
    _reconstruct_search_payload,
    _replay_case,
)
from datadiff_osc.semantic_targets.compiler import compile_target_universe
from datadiff_osc.semantic_targets.declarations import legacy_v4_target_templates
from datadiff_osc.semantic_targets.model import (
    ActivationCertificate,
    CompiledTargetUniverse,
    TargetAssignment,
    TargetCell,
)

from .matcher import TargetMatcher

from .lane_registry import (
    FormalFocusObligation,
    FormalFocusRule,
    FormalLaneRegistry,
    formal_focus_rule,
    formal_lane_registry,
)


FORMAL_CASE_RUN_BINDING_SCHEMA_VERSION = (
    "osc-private-formal-case-run-binding-v1"
)
FORMAL_RUN_BINDING_SCHEMA_VERSION = "osc-private-formal-run-binding-v1"
FORMAL_LANE_PLAN_RECEIPT_SCHEMA_VERSION = (
    "osc-private-formal-lane-plan-receipt-v2"
)
CANONICAL_CASE_BINDING_SCHEMA_VERSION = (
    "osc-private-canonical-case-binding-v1"
)
SCHEDULED_TARGET_ATTEMPT_RECEIPT_SCHEMA_VERSION = (
    "osc-private-scheduled-target-attempt-receipt-v2"
)
MUTATION_ATTEMPT_RECEIPT_SCHEMA_VERSION = (
    "osc-private-mutation-attempt-receipt-v9"
)
MUTATION_OPERATOR_APPLICATION_POLICY_SCHEMA_VERSION = (
    "osc-private-mutation-operator-application-policy-v7"
)
OBSERVATION_CONTEXT_RECEIPT_SCHEMA_VERSION = (
    "osc-private-observation-context-receipt-v2"
)
INTRINSIC_FOCUS_PROOF_SCHEMA_VERSION = "osc-private-intrinsic-focus-proof-v1"
FOCUS_HIT_RECEIPT_SCHEMA_VERSION = "osc-private-focus-hit-receipt-v2"
INTRINSIC_SEARCH_UNIVERSE_SCHEMA_VERSION = (
    "osc-private-intrinsic-search-universe-v1"
)
SEARCH_CONTEXT_SUBJECT_SPEC_SCHEMA_VERSION = (
    "osc-private-search-context-subject-spec-v1"
)

EXPECTED_COMPILED_UNIVERSE_DIGEST = (
    "osc-compiled-target-universe-"
    "c822459944b38c1ae8abc6c271725bece0509457046c16e1088d455e2ac3982e"
)


_T = TypeVar("_T")


def _require_text(value: str, name: str, *, allow_empty: bool = False) -> None:
    if not isinstance(value, str) or (not value and not allow_empty):
        qualifier = "string" if allow_empty else "non-empty string"
        raise ValueError(f"{name} must be a {qualifier}")


def _has_duplicate(values: tuple[_T, ...]) -> bool:
    return any(value in values[:index] for index, value in enumerate(values))


def _unique_sorted_text_input(
    values: Iterable[str],
    name: str,
    *,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    materialized = tuple(values)
    if not allow_empty and not materialized:
        raise ValueError(f"{name} must be non-empty")
    if any(not isinstance(item, str) or not item for item in materialized):
        raise ValueError(f"{name} must contain non-empty strings")
    if _has_duplicate(materialized):
        raise ValueError(f"{name} contains duplicates")
    return tuple(sorted(materialized))


def _require_unique_sorted_text(
    values: tuple[str, ...],
    name: str,
    *,
    allow_empty: bool = False,
) -> None:
    if not isinstance(values, tuple):
        raise TypeError(f"{name} must be a tuple")
    if not allow_empty and not values:
        raise ValueError(f"{name} must be non-empty")
    if any(not isinstance(item, str) or not item for item in values):
        raise ValueError(f"{name} must contain non-empty strings")
    if _has_duplicate(values):
        raise ValueError(f"{name} contains duplicates")
    if values != tuple(sorted(values)):
        raise ValueError(f"{name} must be sorted")


def _unique_sorted_objects(
    values: Iterable[_T],
    name: str,
    *,
    identity: Callable[[_T], str],
) -> tuple[_T, ...]:
    materialized = tuple(values)
    if not materialized:
        raise ValueError(f"{name} must be non-empty")
    identities = tuple(identity(item) for item in materialized)
    if _has_duplicate(identities):
        raise ValueError(f"{name} contains duplicates")
    return tuple(sorted(materialized, key=identity))


@cache
def _compiled_universe() -> CompiledTargetUniverse:
    universe = compile_target_universe(legacy_v4_target_templates())
    counts = (
        len({item.test_family_id for item in universe.fresh_cells}),
        len(universe.fresh_cells),
        len(universe.fresh_edges),
        len(universe.fresh_backend_pair_obligations),
    )
    if universe.digest != EXPECTED_COMPILED_UNIVERSE_DIGEST:
        raise ValueError("compiled Search universe digest drift")
    if counts != (16, 232, 384, 502):
        raise ValueError(f"compiled Search universe count drift: {counts}")
    return universe


@cache
def _fresh_cell_lookup() -> dict[str, TargetCell]:
    return {item.target_cell_id: item for item in _compiled_universe().fresh_cells}


def _assignment_context(
    assignment: TargetAssignment,
    *,
    require_single_cell: bool,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    if not isinstance(assignment, TargetAssignment):
        raise TypeError("Search receipt requires a TargetAssignment")
    if not isinstance(assignment.selected_cell_ids, tuple):
        raise TypeError("Search receipt assignment cells must be a tuple")
    selected = assignment.selected_cell_ids
    if (
        not selected
        or any(not isinstance(item, str) or not item for item in selected)
        or _has_duplicate(selected)
    ):
        raise ValueError("Search receipt assignment cells must be non-empty and unique")
    if require_single_cell and len(selected) != 1:
        raise ValueError("attempt receipt requires exactly one fresh target cell")
    cells = _fresh_cell_lookup()
    unknown = tuple(item for item in selected if item not in cells)
    if unknown:
        raise ValueError(f"Search receipt contains non-fresh cells: {unknown}")

    universe = _compiled_universe()
    edges = {item.contrast_edge_id: item for item in universe.fresh_edges}
    if not isinstance(assignment.selected_edge_ids, tuple):
        raise TypeError("Search receipt assignment edges must be a tuple")
    edge_ids = assignment.selected_edge_ids
    if (
        any(not isinstance(item, str) or not item for item in edge_ids)
        or _has_duplicate(edge_ids)
        or any(item not in edges for item in edge_ids)
    ):
        raise ValueError("Search receipt assignment contains invalid contrast edges")
    if edge_ids:
        expected: list[str] = []
        for edge_id in edge_ids:
            edge = edges[edge_id]
            for cell_id in (edge.base_cell_id, edge.sibling_cell_id):
                if cell_id not in expected:
                    expected.append(cell_id)
        if selected != tuple(expected):
            raise ValueError(
                "Search receipt contrast cells must follow base/sibling direction"
            )
    elif len(selected) > 1:
        raise ValueError("multi-cell Search context requires declared contrast edges")
    if not isinstance(assignment.selected_tile_ids, tuple):
        raise TypeError("Search receipt assignment tiles must be a tuple")
    if assignment.selected_tile_ids:
        raise ValueError("Phase-6 context foundation does not admit shadow tiles")
    families = tuple(sorted({cells[cell_id].test_family_id for cell_id in selected}))
    return tuple(sorted(selected)), families, tuple(sorted(edge_ids))


@dataclass(frozen=True, slots=True)
class IntrinsicSearchUniverse:
    registry: FormalLaneRegistry
    fresh_family_ids: tuple[str, ...]
    fresh_cell_ids: tuple[str, ...]
    contrast_edge_ids: tuple[str, ...]
    backend_pair_obligation_ids: tuple[str, ...]
    schema_version: str = INTRINSIC_SEARCH_UNIVERSE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != INTRINSIC_SEARCH_UNIVERSE_SCHEMA_VERSION:
            raise ValueError("intrinsic Search universe schema mismatch")
        if self.registry != formal_lane_registry():
            raise ValueError("intrinsic Search universe registry drift")
        universe = _compiled_universe()
        expected = (
            tuple(sorted({item.test_family_id for item in universe.fresh_cells})),
            tuple(item.target_cell_id for item in universe.fresh_cells),
            tuple(item.contrast_edge_id for item in universe.fresh_edges),
            tuple(
                item.obligation_id
                for item in universe.fresh_backend_pair_obligations
            ),
        )
        if (
            self.fresh_family_ids,
            self.fresh_cell_ids,
            self.contrast_edge_ids,
            self.backend_pair_obligation_ids,
        ) != expected:
            raise ValueError("intrinsic Search universe identity drift")

    @property
    def digest(self) -> str:
        return stable_digest("osc-private-intrinsic-search-universe", self)

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(self)


@cache
def intrinsic_search_universe() -> IntrinsicSearchUniverse:
    universe = _compiled_universe()
    return IntrinsicSearchUniverse(
        registry=formal_lane_registry(),
        fresh_family_ids=tuple(
            sorted({item.test_family_id for item in universe.fresh_cells})
        ),
        fresh_cell_ids=tuple(item.target_cell_id for item in universe.fresh_cells),
        contrast_edge_ids=tuple(item.contrast_edge_id for item in universe.fresh_edges),
        backend_pair_obligation_ids=tuple(
            item.obligation_id for item in universe.fresh_backend_pair_obligations
        ),
    )


@dataclass(frozen=True, slots=True)
class FormalCaseRunBinding:
    case_id: str
    case_index: int
    seed_lineage: SeedLineage
    schema_version: str = FORMAL_CASE_RUN_BINDING_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != FORMAL_CASE_RUN_BINDING_SCHEMA_VERSION:
            raise ValueError("formal case/run binding schema mismatch")
        _require_text(self.case_id, "case_id")
        if (
            isinstance(self.case_index, bool)
            or not isinstance(self.case_index, int)
            or self.case_index < 0
        ):
            raise ValueError("formal case index must be non-negative")
        if not isinstance(self.seed_lineage, SeedLineage):
            raise TypeError("formal case/run binding requires typed SeedLineage")

    @property
    def digest(self) -> str:
        return stable_digest("osc-private-formal-case-run-binding", self)


def _formal_run_id(
    *,
    protocol_digest: str,
    lane_id: str,
    seed_block_id: str,
    master_seed: int,
    run_lineage: SeedLineage,
) -> str:
    return stable_digest(
        "osc-private-formal-run-id-v1",
        {
            "protocol_digest": protocol_digest,
            "lane_id": lane_id,
            "seed_block_id": seed_block_id,
            "master_seed": master_seed,
            "run_lineage_digest": run_lineage.digest,
        },
    )


@dataclass(frozen=True, slots=True)
class FormalRunBinding:
    run_id: str
    protocol_digest: str
    lane_id: str
    seed_block_id: str
    master_seed: int
    run_lineage: SeedLineage
    case_bindings: tuple[FormalCaseRunBinding, ...]
    schema_version: str = FORMAL_RUN_BINDING_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != FORMAL_RUN_BINDING_SCHEMA_VERSION:
            raise ValueError("formal run binding schema mismatch")
        for name, value in (
            ("run_id", self.run_id),
            ("protocol_digest", self.protocol_digest),
            ("lane_id", self.lane_id),
            ("seed_block_id", self.seed_block_id),
        ):
            _require_text(value, name)
        if (
            isinstance(self.master_seed, bool)
            or not isinstance(self.master_seed, int)
            or self.master_seed < 0
        ):
            raise ValueError("formal run master seed must be non-negative")
        expected_root = SeedLineage(
            protocol_digest=self.protocol_digest,
            master_seed=self.master_seed,
            lane_id=self.lane_id,
            case_index=0,
            stage_name=SeedStage.TARGET,
            counter=0,
            parent_digest="",
        )
        if self.run_lineage != expected_root:
            raise ValueError("formal run root lineage does not recompute exactly")
        expected_run_id = _formal_run_id(
            protocol_digest=self.protocol_digest,
            lane_id=self.lane_id,
            seed_block_id=self.seed_block_id,
            master_seed=self.master_seed,
            run_lineage=self.run_lineage,
        )
        if self.run_id != expected_run_id:
            raise ValueError("formal run identity mismatch")
        if not isinstance(self.case_bindings, tuple) or not self.case_bindings:
            raise ValueError("formal run requires case bindings")
        case_ids = tuple(item.case_id for item in self.case_bindings)
        case_indexes = tuple(item.case_index for item in self.case_bindings)
        if _has_duplicate(case_ids) or _has_duplicate(case_indexes):
            raise ValueError("formal run case mappings must be one-to-one")
        if case_ids != tuple(sorted(case_ids)):
            raise ValueError("formal run case mappings must be case-sorted")
        for item in self.case_bindings:
            expected = derive_stage_lineage(
                self.run_lineage,
                SeedStage.TARGET,
                case_index=item.case_index,
                counter=item.case_index,
            )
            if item.seed_lineage != expected:
                raise ValueError("formal case lineage does not recompute exactly")

    @classmethod
    def build(
        cls,
        *,
        protocol_digest: str,
        lane_id: str,
        seed_block_id: str,
        master_seed: int,
        indexed_case_ids: Iterable[tuple[str, int]],
    ) -> "FormalRunBinding":
        raw = tuple(indexed_case_ids)
        if not raw:
            raise ValueError("indexed_case_ids must be non-empty")
        if any(
            not isinstance(item, tuple) or len(item) != 2 for item in raw
        ):
            raise TypeError("indexed_case_ids must contain (case_id, index) tuples")
        case_ids = tuple(item[0] for item in raw)
        indexes = tuple(item[1] for item in raw)
        if any(not isinstance(item, str) or not item for item in case_ids):
            raise ValueError("indexed_case_ids contains an invalid case ID")
        if any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in indexes
        ):
            raise ValueError("indexed_case_ids contains an invalid index")
        if _has_duplicate(case_ids) or _has_duplicate(indexes):
            raise ValueError("indexed_case_ids contains duplicates")
        root = SeedLineage(
            protocol_digest=protocol_digest,
            master_seed=master_seed,
            lane_id=lane_id,
            case_index=0,
            stage_name=SeedStage.TARGET,
            counter=0,
            parent_digest="",
        )
        cases = tuple(
            sorted(
                (
                    FormalCaseRunBinding(
                        case_id=case_id,
                        case_index=case_index,
                        seed_lineage=derive_stage_lineage(
                            root,
                            SeedStage.TARGET,
                            case_index=case_index,
                            counter=case_index,
                        ),
                    )
                    for case_id, case_index in raw
                ),
                key=lambda item: item.case_id,
            )
        )
        return cls(
            run_id=_formal_run_id(
                protocol_digest=protocol_digest,
                lane_id=lane_id,
                seed_block_id=seed_block_id,
                master_seed=master_seed,
                run_lineage=root,
            ),
            protocol_digest=protocol_digest,
            lane_id=lane_id,
            seed_block_id=seed_block_id,
            master_seed=master_seed,
            run_lineage=root,
            case_bindings=cases,
        )

    @property
    def digest(self) -> str:
        return stable_digest("osc-private-formal-run-binding", self)


@dataclass(frozen=True, slots=True)
class FormalLanePlanReceipt:
    registry: FormalLaneRegistry
    protocol_digest: str
    lane_id: str
    planned_case_ids: tuple[str, ...]
    seed_block_ids: tuple[str, ...]
    run_bindings: tuple[FormalRunBinding, ...]
    focus_obligation_ids: tuple[str, ...]
    schema_version: str = FORMAL_LANE_PLAN_RECEIPT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != FORMAL_LANE_PLAN_RECEIPT_SCHEMA_VERSION:
            raise ValueError("formal lane plan receipt schema mismatch")
        if self.registry != formal_lane_registry():
            raise ValueError("formal lane plan registry is not the frozen registry")
        _require_text(self.protocol_digest, "protocol_digest")
        self.registry.declaration_for(self.lane_id)
        _require_unique_sorted_text(self.planned_case_ids, "planned_case_ids")
        _require_unique_sorted_text(self.seed_block_ids, "seed_block_ids")
        _require_unique_sorted_text(
            self.focus_obligation_ids, "focus_obligation_ids"
        )
        expected_obligations = tuple(
            sorted(
                item.obligation_id
                for item in self.registry.obligations_for(self.lane_id)
            )
        )
        if self.focus_obligation_ids != expected_obligations:
            raise ValueError("formal lane plan focus obligations are caller-substituted")
        if not isinstance(self.run_bindings, tuple) or not self.run_bindings:
            raise ValueError("formal lane plan requires exact run bindings")
        if self.run_bindings != tuple(
            sorted(self.run_bindings, key=lambda item: item.seed_block_id)
        ):
            raise ValueError("formal run bindings must be seed-block sorted")
        run_ids = tuple(item.run_id for item in self.run_bindings)
        blocks = tuple(item.seed_block_id for item in self.run_bindings)
        master_seeds = tuple(item.master_seed for item in self.run_bindings)
        if (
            _has_duplicate(run_ids)
            or _has_duplicate(blocks)
            or _has_duplicate(master_seeds)
        ):
            raise ValueError("formal run, block and master-seed mappings must be unique")
        if blocks != self.seed_block_ids:
            raise ValueError("formal run bindings do not exactly cover seed blocks")
        all_cases = tuple(
            item
            for run in self.run_bindings
            for item in run.case_bindings
        )
        mapped_case_ids = tuple(item.case_id for item in all_cases)
        if _has_duplicate(mapped_case_ids):
            raise ValueError("formal cases cannot map to multiple runs")
        if tuple(sorted(mapped_case_ids)) != self.planned_case_ids:
            raise ValueError("formal runs do not exactly cover planned cases")
        indexed = tuple(sorted(all_cases, key=lambda item: item.case_id))
        for expected_index, binding in enumerate(indexed):
            if binding.case_index != expected_index:
                raise ValueError("formal case index mapping is not canonical")
        for run in self.run_bindings:
            if (
                run.protocol_digest != self.protocol_digest
                or run.lane_id != self.lane_id
            ):
                raise ValueError("formal run is bound to another protocol or lane")

    @classmethod
    def build(
        cls,
        *,
        protocol_digest: str,
        lane_id: str,
        planned_case_ids: Iterable[str],
        seed_block_ids: Iterable[str],
        run_bindings: Iterable[FormalRunBinding],
        registry: FormalLaneRegistry | None = None,
    ) -> "FormalLanePlanReceipt":
        cases = _unique_sorted_text_input(planned_case_ids, "planned_case_ids")
        blocks = _unique_sorted_text_input(seed_block_ids, "seed_block_ids")
        runs = _unique_sorted_objects(
            run_bindings,
            "run_bindings",
            identity=lambda item: item.seed_block_id,
        )
        resolved = registry or formal_lane_registry()
        return cls(
            registry=resolved,
            protocol_digest=protocol_digest,
            lane_id=lane_id,
            planned_case_ids=cases,
            seed_block_ids=blocks,
            run_bindings=runs,
            focus_obligation_ids=tuple(
                sorted(
                    item.obligation_id
                    for item in resolved.obligations_for(lane_id)
                )
            ),
        )

    def case_run_binding(
        self,
        case_id: str,
        case_index: int | None = None,
    ) -> tuple[FormalRunBinding, FormalCaseRunBinding]:
        for run in self.run_bindings:
            for case in run.case_bindings:
                if case.case_id == case_id:
                    if case_index is not None and case.case_index != case_index:
                        raise ValueError(
                            "Search receipt case identity is not index-bound to the plan"
                        )
                    return run, case
        raise ValueError("Search receipt case is outside the formal lane plan")

    @property
    def digest(self) -> str:
        return stable_digest("osc-private-formal-lane-plan-receipt", self)

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(self)


def _validate_plan_assignment(
    plan: FormalLanePlanReceipt,
    case_id: str,
    case_index: int,
    assignment: TargetAssignment,
) -> tuple[FormalRunBinding, FormalCaseRunBinding]:
    if not isinstance(assignment, TargetAssignment):
        raise TypeError("Search receipt assignment requires TargetAssignment")
    run, binding = plan.case_run_binding(case_id, case_index)
    if assignment.seed_lineage != binding.seed_lineage:
        raise ValueError("assignment lineage is not the exact formal case mapping")
    return run, binding


def _attempt_identity(namespace: str, payload: object) -> str:
    return stable_digest(namespace, payload)


@dataclass(frozen=True, slots=True)
class ScheduledTargetAttemptReceipt:
    plan: FormalLanePlanReceipt
    case_id: str
    case_index: int
    formal_run_id: str
    seed_block_id: str
    assignment: TargetAssignment
    target_cell_id: str
    target_family_id: str
    runtime_task_ref: str
    attempt_id: str
    schema_version: str = SCHEDULED_TARGET_ATTEMPT_RECEIPT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEDULED_TARGET_ATTEMPT_RECEIPT_SCHEMA_VERSION:
            raise ValueError("scheduled target attempt receipt schema mismatch")
        _require_text(self.case_id, "case_id")
        _require_text(self.runtime_task_ref, "runtime_task_ref", allow_empty=True)
        run, _binding = _validate_plan_assignment(
            self.plan, self.case_id, self.case_index, self.assignment
        )
        if (self.formal_run_id, self.seed_block_id) != (
            run.run_id,
            run.seed_block_id,
        ):
            raise ValueError("scheduled attempt run/block is caller-substituted")
        cells, families, _edges = _assignment_context(
            self.assignment, require_single_cell=True
        )
        if (self.target_cell_id,) != cells or (self.target_family_id,) != families:
            raise ValueError("scheduled attempt cell/family is caller-substituted")
        expected = _scheduled_attempt_id(
            self.plan,
            self.case_id,
            self.case_index,
            self.formal_run_id,
            self.seed_block_id,
            self.assignment,
            self.target_cell_id,
            self.target_family_id,
        )
        if self.attempt_id != expected:
            raise ValueError("scheduled target attempt identity mismatch")

    @classmethod
    def build(
        cls,
        *,
        plan: FormalLanePlanReceipt,
        case_id: str,
        case_index: int,
        assignment: TargetAssignment,
        runtime_task_ref: str = "",
    ) -> "ScheduledTargetAttemptReceipt":
        run, _binding = _validate_plan_assignment(
            plan, case_id, case_index, assignment
        )
        cells, families, _edges = _assignment_context(
            assignment, require_single_cell=True
        )
        attempt_id = _scheduled_attempt_id(
            plan,
            case_id,
            case_index,
            run.run_id,
            run.seed_block_id,
            assignment,
            cells[0],
            families[0],
        )
        return cls(
            plan,
            case_id,
            case_index,
            run.run_id,
            run.seed_block_id,
            assignment,
            cells[0],
            families[0],
            runtime_task_ref,
            attempt_id,
        )

    @property
    def digest(self) -> str:
        return stable_digest("osc-private-scheduled-target-attempt-receipt", self)

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(self)


def _scheduled_attempt_id(
    plan: FormalLanePlanReceipt,
    case_id: str,
    case_index: int,
    formal_run_id: str,
    seed_block_id: str,
    assignment: TargetAssignment,
    target_cell_id: str,
    target_family_id: str,
) -> str:
    return _attempt_identity(
        "osc-private-scheduled-target-attempt-id-v2",
        {
            "plan_digest": plan.digest,
            "case_id": case_id,
            "case_index": case_index,
            "formal_run_id": formal_run_id,
            "seed_block_id": seed_block_id,
            "assignment_digest": assignment.digest,
            "target_cell_id": target_cell_id,
            "target_family_id": target_family_id,
        },
    )


def _reject_duplicate_json_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"canonical Case JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _canonical_case_bytes(case: Case) -> bytes:
    if not isinstance(case, Case):
        raise TypeError("canonical case binding requires a Case")
    return canonical_json(case).encode("utf-8")


def _decode_canonical_case(payload: bytes) -> Case:
    if not isinstance(payload, bytes) or not payload:
        raise ValueError("canonical Case JSON must be non-empty bytes")
    try:
        decoded = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_pairs,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("canonical Case JSON is malformed") from exc
    try:
        case = _replay_case(decoded, "canonical_case_json")
    except Exception as exc:
        raise ValueError(f"canonical Case payload is invalid: {type(exc).__name__}") from exc
    if _canonical_case_bytes(case) != payload:
        raise ValueError("canonical Case JSON is not in canonical byte form")
    return case


@dataclass(frozen=True, slots=True)
class CanonicalCaseBinding:
    binding_case_id: str
    source_case_id: str
    canonical_case_json: bytes
    case_digest: str
    schema_version: str = CANONICAL_CASE_BINDING_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CANONICAL_CASE_BINDING_SCHEMA_VERSION:
            raise ValueError("canonical case binding schema mismatch")
        _require_text(self.binding_case_id, "binding_case_id")
        _require_text(self.source_case_id, "source_case_id")
        case = _decode_canonical_case(self.canonical_case_json)
        if case.case_id != self.source_case_id:
            raise ValueError("canonical source Case identity mismatch")
        expected = stable_digest(
            "osc-private-canonical-case-v1", self.canonical_case_json
        )
        if self.case_digest != expected:
            raise ValueError("canonical source Case digest mismatch")

    @classmethod
    def build(
        cls,
        *,
        binding_case_id: str,
        case: Case,
    ) -> "CanonicalCaseBinding":
        payload = _canonical_case_bytes(case)
        return cls(
            binding_case_id=binding_case_id,
            source_case_id=case.case_id,
            canonical_case_json=payload,
            case_digest=stable_digest("osc-private-canonical-case-v1", payload),
        )

    @property
    def case(self) -> Case:
        return _decode_canonical_case(self.canonical_case_json)

    @property
    def digest(self) -> str:
        return stable_digest("osc-private-canonical-case-binding", self)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


_LIVE_CONTEXT_MAX_DEPTH = 64
_UNINSPECTABLE_DYNAMIC_DEPENDENCY_NAMES = frozenset(
    {
        "__import__",
        "eval",
        "exec",
        "getattr",
        "globals",
        "locals",
        "vars",
    }
)
_UNINSPECTABLE_IMPORT_OPNAMES = frozenset(
    {"IMPORT_FROM", "IMPORT_NAME", "IMPORT_STAR"}
)
_LiveContextMemoKey = tuple[
    int,
    bool,
    bool,
    tuple[tuple[str, ...], ...],
]
_LiveContextMemo = dict[_LiveContextMemoKey, str]


def _stable_code_constant_payload(value: object) -> object:
    if value is None:
        return ("none",)
    if value is Ellipsis:
        return ("ellipsis",)
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, int):
        return ("int", value)
    if isinstance(value, float):
        return ("float", value.hex())
    if isinstance(value, complex):
        return ("complex", value.real.hex(), value.imag.hex())
    if isinstance(value, str):
        return ("str", value)
    if isinstance(value, bytes):
        return ("bytes", value)
    if isinstance(value, tuple):
        return (
            "tuple",
            tuple(_stable_code_constant_payload(item) for item in value),
        )
    if isinstance(value, frozenset):
        items = tuple(_stable_code_constant_payload(item) for item in value)
        return (
            "frozenset",
            tuple(
                item
                for _digest, item in sorted(
                    (
                        stable_digest("osc-private-code-constant-order-v2", item),
                        item,
                    )
                    for item in items
                )
            ),
        )
    if isinstance(value, CodeType):
        return ("nested-code", _stable_code_payload(value))
    raise ValueError(
        f"code constant type {type(value).__name__} is unsupported"
    )


def _stable_code_payload(code: CodeType) -> tuple[object, ...]:
    """Exclude CPython quickening state while binding executable structure."""

    return (
        "python-code-structure-v2",
        code.co_argcount,
        code.co_posonlyargcount,
        code.co_kwonlyargcount,
        code.co_nlocals,
        code.co_stacksize,
        code.co_flags,
        code.co_code,
        tuple(_stable_code_constant_payload(item) for item in code.co_consts),
        code.co_names,
        code.co_varnames,
        code.co_filename,
        code.co_name,
        getattr(code, "co_qualname", code.co_name),
        code.co_firstlineno,
        code.co_linetable,
        getattr(code, "co_exceptiontable", b""),
        code.co_freevars,
        code.co_cellvars,
    )


def _stable_code_digest(code: CodeType) -> str:
    return stable_digest("osc-private-python-code-structure-v2", _stable_code_payload(code))


def _read_nonempty_file_sha256(path: object, name: str) -> str:
    if not isinstance(path, str) or not path:
        raise ValueError(f"{name} source path is unavailable")
    try:
        payload = Path(path).read_bytes()
    except (OSError, ValueError) as exc:
        raise ValueError(f"{name} source module cannot be read exactly") from exc
    if not payload:
        raise ValueError(f"{name} source module is empty")
    return _sha256_bytes(payload)


def _module_static_payload(value: ModuleType, name: str) -> dict[str, object]:
    module_name = getattr(value, "__name__", "")
    _require_text(module_name, f"{name} module name")
    source_path = getattr(value, "__file__", None)
    if isinstance(source_path, str) and source_path:
        source_sha256 = _read_nonempty_file_sha256(source_path, name)
        source_kind = "file"
    elif module_name == "builtins" or module_name in sys.builtin_module_names:
        source_sha256 = ""
        source_kind = "builtin"
    else:
        raise ValueError(f"{name} module implementation is uninspectable")
    return {
        "kind": "module",
        "module": module_name,
        "source_kind": source_kind,
        "source_sha256": source_sha256,
        "python_implementation": sys.implementation.name,
        "python_cache_tag": str(sys.implementation.cache_tag or ""),
    }


def _class_static_payload(value: type[object], name: str) -> dict[str, object]:
    module_name = getattr(value, "__module__", "")
    qualname = getattr(value, "__qualname__", "")
    _require_text(module_name, f"{name} class module")
    _require_text(qualname, f"{name} class qualname")
    module = sys.modules.get(module_name)
    module_payload: dict[str, object] | None = None
    if isinstance(module, ModuleType):
        module_payload = _module_static_payload(module, f"{name} owning module")
    try:
        class_source = inspect.getsource(value).encode("utf-8")
    except (OSError, TypeError):
        class_source = b""
    if not class_source and module_payload is None:
        raise ValueError(f"{name} class implementation is uninspectable")
    return {
        "kind": "class",
        "module": module_name,
        "qualname": qualname,
        "class_source_sha256": (
            _sha256_bytes(class_source) if class_source else ""
        ),
        "module_identity": module_payload,
        "bases": tuple(
            (base.__module__, base.__qualname__) for base in value.__bases__
        ),
    }


def _code_dependency_specs(
    code: CodeType,
) -> dict[str, set[tuple[str, ...]]]:
    result: dict[str, set[tuple[str, ...]]] = {}
    try:
        instructions = tuple(dis.get_instructions(code))
    except (TypeError, ValueError) as exc:
        raise ValueError("callable bytecode dependencies cannot be inspected") from exc
    for instruction in instructions:
        if instruction.opname in _UNINSPECTABLE_IMPORT_OPNAMES:
            raise ValueError(
                "callable uses a dynamic import dependency that cannot be identity-bound"
            )
    for index, instruction in enumerate(instructions):
        if instruction.opname not in {"LOAD_GLOBAL", "LOAD_NAME"}:
            continue
        dependency_name = instruction.argval
        if not isinstance(dependency_name, str) or not dependency_name:
            raise ValueError("callable contains an unresolved global dependency")
        if dependency_name in _UNINSPECTABLE_DYNAMIC_DEPENDENCY_NAMES:
            raise ValueError(
                f"callable uses dynamic dependency resolver {dependency_name!r}"
            )
        attributes: list[str] = []
        for following in instructions[index + 1 :]:
            if following.opname not in {"LOAD_ATTR", "LOAD_METHOD"}:
                break
            attribute = following.argval
            if not isinstance(attribute, str) or not attribute:
                raise ValueError("callable contains an unresolved attribute dependency")
            attributes.append(attribute)
        result.setdefault(dependency_name, set()).add(tuple(attributes))
    for constant in code.co_consts:
        if isinstance(constant, CodeType):
            nested = _code_dependency_specs(constant)
            for dependency_name, paths in nested.items():
                result.setdefault(dependency_name, set()).update(paths)
    return result


def _resolve_attribute_path(
    value: object,
    attributes: tuple[str, ...],
    name: str,
) -> tuple[tuple[tuple[str, ...], object], ...]:
    """Resolve and retain every static owner in one declared attribute path."""

    current = value
    resolved: list[tuple[tuple[str, ...], object]] = []
    for index, attribute in enumerate(attributes, start=1):
        try:
            if isinstance(current, ModuleType):
                namespace = vars(current)
                if attribute not in namespace:
                    raise AttributeError(attribute)
                current = namespace[attribute]
            else:
                current = inspect.getattr_static(current, attribute)
        except (AttributeError, TypeError) as exc:
            raise ValueError(f"{name} attribute dependency is unresolved") from exc
        resolved.append((attributes[:index], current))
    return tuple(resolved)


def _attribute_path_live_bindings(
    value: object,
    name: str,
    *,
    attribute_paths: tuple[tuple[str, ...], ...],
    stack: tuple[int, ...],
    memo: _LiveContextMemo,
    recursive_dependencies: bool,
) -> tuple[tuple[tuple[str, ...], tuple[tuple[tuple[str, ...], str], ...]], ...]:
    """Bind each owner, including mutable intermediate values, in every path."""

    def owner_digest(
        owner: object,
        owner_path: tuple[str, ...],
        *,
        terminal: bool,
    ) -> str:
        # A terminal Python method's implementation bytes are authority-relevant,
        # but recursively walking every dependency of a library method would make
        # a policy depend on dynamic implementation internals outside this path.
        # Intermediate owners always use the strict recursive route below.
        if terminal and (inspect.isfunction(owner) or inspect.ismethod(owner)):
            return _live_context_digest(
                owner,
                f"{name}.{'.'.join(owner_path)}",
                stack=stack,
                memo=memo,
                inspect_function_globals=False,
                recursive_dependencies=False,
            )
        return _live_context_digest(
            owner,
            f"{name}.{'.'.join(owner_path)}",
            stack=stack,
            memo=memo,
            inspect_function_globals=True,
            recursive_dependencies=True,
        )

    bindings: list[
        tuple[tuple[str, ...], tuple[tuple[tuple[str, ...], str], ...]]
    ] = []
    for attributes in attribute_paths:
        if not attributes:
            continue
        resolved = _resolve_attribute_path(value, attributes, name)
        bindings.append(
            (
                attributes,
                tuple(
                    (
                        owner_path,
                        owner_digest(
                            owner,
                            owner_path,
                            terminal=owner_path == attributes,
                        ),
                    )
                    for owner_path, owner in resolved
                ),
            )
        )
    return tuple(bindings)


def _python_function_static_payload(
    value: Callable[..., object],
    name: str,
) -> tuple[object, ...]:
    """Bind a non-recursive policy dependency without inheriting its caches."""

    code = getattr(value, "__code__", None)
    if not isinstance(code, CodeType):
        raise ValueError(f"{name} has no inspectable Python code object")
    try:
        source_file = inspect.getsourcefile(value)
        callable_source = inspect.getsource(value).encode("utf-8")
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} implementation cannot be inspected exactly") from exc
    if not source_file or not callable_source:
        raise ValueError(f"{name} implementation context is incomplete")
    return (
        "nonrecursive-python-function",
        getattr(value, "__module__", ""),
        getattr(value, "__qualname__", ""),
        _sha256_bytes(callable_source),
        _stable_code_digest(code),
        _read_nonempty_file_sha256(source_file, name),
    )


def _live_context_digest(
    value: object,
    name: str,
    *,
    stack: tuple[int, ...],
    memo: _LiveContextMemo,
    inspect_function_globals: bool,
    recursive_dependencies: bool,
    module_attribute_paths: tuple[tuple[str, ...], ...] = (),
) -> str:
    if len(stack) >= _LIVE_CONTEXT_MAX_DEPTH:
        raise ValueError(f"{name} live dependency depth is unsupported")
    if value is None:
        return stable_digest("osc-private-live-context-scalar-v3", ("none", None))
    if isinstance(value, bool):
        return stable_digest("osc-private-live-context-scalar-v3", ("bool", value))
    if isinstance(value, int):
        return stable_digest("osc-private-live-context-scalar-v3", ("int", value))
    if isinstance(value, float):
        rendered = (
            value.hex()
            if math.isfinite(value)
            else "nan"
            if math.isnan(value)
            else "+inf"
            if value > 0
            else "-inf"
        )
        return stable_digest("osc-private-live-context-scalar-v3", ("float", rendered))
    if isinstance(value, str):
        return stable_digest("osc-private-live-context-scalar-v3", ("str", value))
    if isinstance(value, bytes):
        return stable_digest("osc-private-live-context-scalar-v3", ("bytes", value))
    if isinstance(value, Enum):
        return stable_digest(
            "osc-private-live-context-enum-v3",
            (
                type(value).__module__,
                type(value).__qualname__,
                value.name,
                _live_context_digest(
                    value.value,
                    f"{name}.value",
                    stack=stack,
                    memo=memo,
                    inspect_function_globals=inspect_function_globals,
                    recursive_dependencies=recursive_dependencies,
                ),
            ),
        )

    identity = id(value)
    if identity in stack:
        if inspect.isfunction(value):
            return stable_digest(
                "osc-private-live-context-recursive-function-reference-v4",
                _python_function_static_payload(value, name),
            )
        raise ValueError(f"{name} live dependency graph is cyclicly unresolved")
    canonical_attribute_paths = tuple(sorted(set(module_attribute_paths)))
    memo_key: _LiveContextMemoKey = (
        identity,
        inspect_function_globals,
        recursive_dependencies,
        canonical_attribute_paths,
    )
    cached = memo.get(memo_key)
    if cached is not None:
        return cached
    next_stack = (*stack, identity)

    if isinstance(value, tuple):
        payload: object = (
            "tuple",
            tuple(
                _live_context_digest(
                    item,
                    f"{name}[{index}]",
                    stack=next_stack,
                    memo=memo,
                    inspect_function_globals=inspect_function_globals,
                    recursive_dependencies=recursive_dependencies,
                )
                for index, item in enumerate(value)
            ),
        )
    elif isinstance(value, frozenset):
        payload = (
            "frozenset",
            tuple(
                sorted(
                    _live_context_digest(
                        item,
                        f"{name}.item",
                        stack=next_stack,
                        memo=memo,
                        inspect_function_globals=inspect_function_globals,
                        recursive_dependencies=recursive_dependencies,
                    )
                    for item in value
                )
            ),
        )
    elif isinstance(value, MappingProxyType):
        if any(not isinstance(key, str) or not key for key in value):
            raise ValueError(f"{name} immutable mapping keys are not canonical")
        payload = (
            "mappingproxy",
            tuple(
                (
                    key,
                    _live_context_digest(
                        value[key],
                        f"{name}[{key!r}]",
                        stack=next_stack,
                        memo=memo,
                        inspect_function_globals=inspect_function_globals,
                        recursive_dependencies=recursive_dependencies,
                    ),
                )
                for key in sorted(value)
            ),
        )
    elif isinstance(value, ContextVar):
        try:
            default_value = Context().run(value.get)
        except LookupError:
            default_digest = ""
            has_default = False
        else:
            default_digest = _live_context_digest(
                default_value,
                f"{name}.default",
                stack=next_stack,
                memo=memo,
                inspect_function_globals=inspect_function_globals,
                recursive_dependencies=recursive_dependencies,
            )
            has_default = True
        try:
            active_value = value.get()
        except LookupError:
            active_digest = ""
            has_active_value = False
        else:
            active_digest = _live_context_digest(
                active_value,
                f"{name}.active",
                stack=next_stack,
                memo=memo,
                inspect_function_globals=inspect_function_globals,
                recursive_dependencies=recursive_dependencies,
            )
            has_active_value = True
        payload = (
            "contextvar",
            value.name,
            has_default,
            default_digest,
            has_active_value,
            active_digest,
        )
    elif isinstance(value, staticmethod):
        payload = (
            "staticmethod",
            _live_context_digest(
                value.__func__,
                f"{name}.__func__",
                stack=next_stack,
                memo=memo,
                inspect_function_globals=recursive_dependencies,
                recursive_dependencies=recursive_dependencies,
            ),
        )
    elif isinstance(value, classmethod):
        payload = (
            "classmethod",
            _live_context_digest(
                value.__func__,
                f"{name}.__func__",
                stack=next_stack,
                memo=memo,
                inspect_function_globals=recursive_dependencies,
                recursive_dependencies=recursive_dependencies,
            ),
        )
    elif isinstance(value, property):
        payload = (
            "property",
            _live_context_digest(
                value.fget,
                f"{name}.fget",
                stack=next_stack,
                memo=memo,
                inspect_function_globals=recursive_dependencies,
                recursive_dependencies=recursive_dependencies,
            ),
            _live_context_digest(
                value.fset,
                f"{name}.fset",
                stack=next_stack,
                memo=memo,
                inspect_function_globals=recursive_dependencies,
                recursive_dependencies=recursive_dependencies,
            ),
            _live_context_digest(
                value.fdel,
                f"{name}.fdel",
                stack=next_stack,
                memo=memo,
                inspect_function_globals=recursive_dependencies,
                recursive_dependencies=recursive_dependencies,
            ),
        )
    elif isinstance(value, ModuleType):
        payload = (
            "module",
            _module_static_payload(value, name),
            _attribute_path_live_bindings(
                value,
                name,
                attribute_paths=canonical_attribute_paths,
                stack=next_stack,
                memo=memo,
                recursive_dependencies=recursive_dependencies,
            ),
        )
    elif inspect.isfunction(value):
        if not recursive_dependencies and not inspect_function_globals:
            payload = _python_function_static_payload(value, name)
        else:
            payload = _python_function_live_payload(
                value,
                name,
                stack=next_stack,
                memo=memo,
                inspect_globals=inspect_function_globals,
                recursive_dependencies=recursive_dependencies,
            )
    elif inspect.ismethod(value):
        payload = (
            "bound_method",
            _live_context_digest(
                value.__func__,
                f"{name}.__func__",
                stack=next_stack,
                memo=memo,
                inspect_function_globals=inspect_function_globals,
                recursive_dependencies=recursive_dependencies,
            ),
            _live_context_digest(
                value.__self__,
                f"{name}.__self__",
                stack=next_stack,
                memo=memo,
                inspect_function_globals=inspect_function_globals,
                recursive_dependencies=recursive_dependencies,
            ),
        )
    elif inspect.isbuiltin(value) or inspect.ismethoddescriptor(value):
        builtin_name = getattr(value, "__name__", "")
        if builtin_name in _UNINSPECTABLE_DYNAMIC_DEPENDENCY_NAMES:
            raise ValueError(
                f"{name} uses unsupported dynamic builtin {builtin_name!r}"
            )
        bound_owner = getattr(value, "__self__", None)
        if bound_owner is not None and not isinstance(bound_owner, ModuleType):
            raise ValueError(f"{name} uses a stateful bound builtin callable")
        payload = (
            "builtin_callable",
            getattr(value, "__module__", "builtins") or "builtins",
            getattr(value, "__qualname__", getattr(value, "__name__", "")),
            sys.implementation.name,
            str(sys.implementation.cache_tag or ""),
            sys.version,
        )
    elif inspect.isclass(value):
        payload = (
            "class",
            _class_static_payload(value, name),
            _attribute_path_live_bindings(
                value,
                name,
                attribute_paths=canonical_attribute_paths,
                stack=next_stack,
                memo=memo,
                recursive_dependencies=recursive_dependencies,
            ),
        )
    elif (
        type(value).__module__ == "typing"
        and type(value).__qualname__ == "_SpecialGenericAlias"
    ):
        alias_module = getattr(value, "__module__", "")
        alias_qualname = getattr(value, "__qualname__", "")
        origin = getattr(value, "__origin__", None)
        if (
            not isinstance(alias_module, str)
            or not alias_module
            or not isinstance(alias_qualname, str)
            or not alias_qualname
            or not inspect.isclass(origin)
        ):
            raise ValueError(f"{name} typing alias is not inspectable")
        payload = (
            "typing_special_generic_alias",
            alias_module,
            alias_qualname,
            repr(value),
            _live_context_digest(
                origin,
                f"{name}.__origin__",
                stack=next_stack,
                memo=memo,
                inspect_function_globals=inspect_function_globals,
                recursive_dependencies=recursive_dependencies,
            ),
        )
    elif callable(value):
        raise ValueError(
            f"{name} callable object type {type(value).__name__} is unsupported"
        )
    elif is_dataclass(value) and not isinstance(value, type):
        params = getattr(type(value), "__dataclass_params__", None)
        if params is None or not params.frozen:
            raise ValueError(f"{name} uses mutable dataclass live context")
        payload = (
            "frozen_dataclass",
            type(value).__module__,
            type(value).__qualname__,
            tuple(
                (
                    field.name,
                    _live_context_digest(
                        getattr(value, field.name),
                        f"{name}.{field.name}",
                        stack=next_stack,
                        memo=memo,
                        inspect_function_globals=inspect_function_globals,
                        recursive_dependencies=recursive_dependencies,
                    ),
                )
                for field in fields(value)
            ),
        )
    elif isinstance(value, CodeType):
        payload = ("code", _stable_code_digest(value))
    elif isinstance(value, list):
        raise ValueError(f"{name} uses mutable live context")
    elif isinstance(value, dict):
        raise ValueError(f"{name} uses mutable live context")
    elif isinstance(value, set):
        raise ValueError(f"{name} uses mutable live context")
    elif isinstance(value, bytearray):
        raise ValueError(f"{name} uses mutable live context")
    else:
        raise ValueError(
            f"{name} live dependency type {type(value).__name__} is unsupported"
        )

    digest = stable_digest("osc-private-live-context-node-v4", payload)
    memo[memo_key] = digest
    return digest


def _python_function_live_payload(
    value: Callable[..., object],
    name: str,
    *,
    stack: tuple[int, ...],
    memo: _LiveContextMemo,
    inspect_globals: bool,
    recursive_dependencies: bool,
) -> tuple[object, ...]:
    code = getattr(value, "__code__", None)
    if not isinstance(code, CodeType):
        raise ValueError(f"{name} has no replayable Python code object")
    try:
        source_file = inspect.getsourcefile(value)
        callable_source = inspect.getsource(value).encode("utf-8")
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} implementation cannot be inspected exactly") from exc
    if not source_file or not callable_source:
        raise ValueError(f"{name} implementation context is incomplete")
    module_source_sha256 = _read_nonempty_file_sha256(source_file, name)

    defaults = value.__defaults__ or ()
    kwdefaults = value.__kwdefaults__ or {}
    if any(not isinstance(key, str) or not key for key in kwdefaults):
        raise ValueError(f"{name} keyword defaults are not canonical")
    closure_values: list[str] = []
    for index, cell in enumerate(value.__closure__ or ()):
        try:
            cell_value = cell.cell_contents
        except ValueError as exc:
            raise ValueError(f"{name} closure cell is empty") from exc
        closure_values.append(
            _live_context_digest(
                cell_value,
                f"{name}.closure[{index}]",
                stack=stack,
                memo=memo,
                inspect_function_globals=recursive_dependencies,
                recursive_dependencies=recursive_dependencies,
            )
        )

    function_attributes = getattr(value, "__dict__", {})
    if any(not isinstance(key, str) or not key for key in function_attributes):
        raise ValueError(f"{name} function attributes are not canonical")
    annotations = getattr(value, "__annotations__", {})
    if any(not isinstance(key, str) or not key for key in annotations):
        raise ValueError(f"{name} annotations are not canonical")

    dependencies: list[tuple[object, ...]] = []
    if inspect_globals:
        dependency_specs = _code_dependency_specs(code)
        globals_map = value.__globals__
        builtin_namespace = globals_map.get("__builtins__", builtins)
        if isinstance(builtin_namespace, ModuleType):
            builtin_map = vars(builtin_namespace)
        elif isinstance(builtin_namespace, dict):
            builtin_map = builtin_namespace
        else:
            raise ValueError(f"{name} builtins namespace is unsupported")
        for dependency_name in sorted(dependency_specs):
            if dependency_name in globals_map:
                dependency = globals_map[dependency_name]
                origin = "global"
            elif dependency_name in builtin_map:
                dependency = builtin_map[dependency_name]
                origin = "builtin"
            else:
                raise ValueError(
                    f"{name} global dependency {dependency_name!r} is unresolved"
                )
            attribute_paths = tuple(sorted(dependency_specs[dependency_name]))
            dependencies.append(
                (
                    dependency_name,
                    origin,
                    attribute_paths,
                    _live_context_digest(
                        dependency,
                        f"{name}.global[{dependency_name}]",
                        stack=stack,
                        memo=memo,
                        inspect_function_globals=recursive_dependencies,
                        recursive_dependencies=recursive_dependencies,
                        module_attribute_paths=attribute_paths,
                    ),
                )
            )

    return (
        "python_function",
        getattr(value, "__module__", ""),
        getattr(value, "__qualname__", ""),
        _sha256_bytes(callable_source),
        _stable_code_digest(code),
        module_source_sha256,
        _live_context_digest(
            defaults,
            f"{name}.__defaults__",
            stack=stack,
            memo=memo,
            inspect_function_globals=recursive_dependencies,
            recursive_dependencies=recursive_dependencies,
        ),
        tuple(
            (
                key,
                _live_context_digest(
                    kwdefaults[key],
                    f"{name}.__kwdefaults__[{key}]",
                    stack=stack,
                    memo=memo,
                    inspect_function_globals=recursive_dependencies,
                    recursive_dependencies=recursive_dependencies,
                ),
            )
            for key in sorted(kwdefaults)
        ),
        tuple(closure_values),
        tuple(
            (
                key,
                _live_context_digest(
                    function_attributes[key],
                    f"{name}.__dict__[{key}]",
                    stack=stack,
                    memo=memo,
                    inspect_function_globals=recursive_dependencies,
                    recursive_dependencies=recursive_dependencies,
                ),
            )
            for key in sorted(function_attributes)
        ),
        tuple(
            (
                key,
                _live_context_digest(
                    annotations[key],
                    f"{name}.__annotations__[{key}]",
                    stack=stack,
                    memo=memo,
                    inspect_function_globals=recursive_dependencies,
                    recursive_dependencies=recursive_dependencies,
                ),
            )
            for key in sorted(annotations)
        ),
        tuple(dependencies),
    )


def _callable_implementation_digest(
    value: object,
    name: str,
    *,
    recursive_dependencies: bool = True,
) -> str:
    """Bind executable bytes and the complete supported live callable context."""

    _require_text(name, "callable implementation name")
    if not inspect.isfunction(value):
        raise ValueError(f"{name} is not a fully inspectable Python function")
    implementation_digest = _live_context_digest(
        value,
        name,
        stack=(),
        memo={},
        inspect_function_globals=True,
        recursive_dependencies=recursive_dependencies,
    )
    return stable_digest(
        "osc-private-callable-implementation-v5",
        {
            "name": name,
            "live_context_schema_version": "osc-private-live-context-v5",
            "recursive_dependencies": recursive_dependencies,
            "implementation_context_digest": implementation_digest,
        },
    )


def _policy_external_function_live_digest(
    value: object,
    name: str,
    *,
    bind_cross_module_globals: bool = False,
) -> str:
    """Bind a cross-module function at the selected policy boundary.

    The policy graph follows functions defined in the policy callable's source
    module recursively.  A function reached through another module is still
    bound to its source/code/module/defaults/kwdefaults/closure/attributes, but
    its own module is an independently source-sealed implementation boundary.
    This keeps a deterministic policy identity available for the established
    extraction stack, whose transitive generic serialization helpers use dynamic
    reflection outside the policy module.  Callers that make a cross-module
    function part of an execution-authorizing policy must opt into complete
    live-global binding.  That stricter mode recursively binds every supported
    direct dependency and fails closed for unresolved, mutable, cyclic,
    uninspectable, dynamic, or unsupported context.
    """

    if not inspect.isfunction(value):
        raise ValueError(f"{name} is not an inspectable Python function")
    if bind_cross_module_globals:
        return stable_digest(
            "osc-private-policy-external-function-live-context-v3",
            _python_function_live_payload(
                value,
                name,
                stack=(),
                memo={},
                inspect_globals=True,
                recursive_dependencies=True,
            ),
        )
    return stable_digest(
        "osc-private-policy-external-function-live-context-v1",
        _python_function_live_payload(
            value,
            name,
            stack=(),
            memo={},
            inspect_globals=False,
            recursive_dependencies=False,
        ),
    )


def _policy_same_module_function_live_digest(
    value: Callable[..., object],
    name: str,
    *,
    policy_module: str,
    stack: tuple[int, ...],
    memo: dict[tuple[int, str, bool], str],
    bind_cross_module_globals: bool,
) -> str:
    """Recursively bind direct Python-function dependencies in one module."""

    if not inspect.isfunction(value):
        raise ValueError(f"{name} is not an inspectable Python function")
    identity = id(value)
    memo_key = (identity, policy_module, bind_cross_module_globals)
    if identity in stack:
        # Recursive Python functions are source/code-sealed references, not
        # mutable graph edges.  Other cyclic values are rejected by the normal
        # live-context encoder below.
        return stable_digest(
            "osc-private-policy-recursive-function-reference-v1",
            _python_function_static_payload(value, name),
        )
    cached = memo.get(memo_key)
    if cached is not None:
        return cached

    code = getattr(value, "__code__", None)
    if not isinstance(code, CodeType):
        raise ValueError(f"{name} has no replayable Python code object")
    try:
        source_file = inspect.getsourcefile(value)
        callable_source = inspect.getsource(value).encode("utf-8")
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} implementation cannot be inspected exactly") from exc
    if not source_file or not callable_source:
        raise ValueError(f"{name} implementation context is incomplete")
    module_source_sha256 = _read_nonempty_file_sha256(source_file, name)
    next_stack = (*stack, identity)

    defaults = value.__defaults__ or ()
    kwdefaults = value.__kwdefaults__ or {}
    if any(not isinstance(key, str) or not key for key in kwdefaults):
        raise ValueError(f"{name} keyword defaults are not canonical")
    function_attributes = getattr(value, "__dict__", {})
    if any(not isinstance(key, str) or not key for key in function_attributes):
        raise ValueError(f"{name} function attributes are not canonical")
    annotations = getattr(value, "__annotations__", {})
    if any(not isinstance(key, str) or not key for key in annotations):
        raise ValueError(f"{name} annotations are not canonical")

    live_memo: _LiveContextMemo = {}

    def nonfunction_live_digest(
        dependency: object,
        dependency_name: str,
        attribute_paths: tuple[tuple[str, ...], ...],
    ) -> str:
        return _live_context_digest(
            dependency,
            dependency_name,
            stack=next_stack,
            memo=live_memo,
            inspect_function_globals=True,
            recursive_dependencies=True,
            module_attribute_paths=attribute_paths,
        )

    closure_values: list[str] = []
    for index, cell in enumerate(value.__closure__ or ()):
        try:
            cell_value = cell.cell_contents
        except ValueError as exc:
            raise ValueError(f"{name} closure cell is empty") from exc
        if inspect.isfunction(cell_value):
            if getattr(cell_value, "__module__", "") == policy_module:
                closure_values.append(
                    _policy_same_module_function_live_digest(
                        cell_value,
                        f"{name}.closure[{index}]",
                        policy_module=policy_module,
                        stack=next_stack,
                        memo=memo,
                        bind_cross_module_globals=bind_cross_module_globals,
                    )
                )
            else:
                closure_values.append(
                    _policy_external_function_live_digest(
                        cell_value,
                        f"{name}.closure[{index}]",
                        bind_cross_module_globals=bind_cross_module_globals,
                    )
                )
        else:
            closure_values.append(
                nonfunction_live_digest(
                    cell_value,
                    f"{name}.closure[{index}]",
                    (),
                )
            )

    dependency_specs = _code_dependency_specs(code)
    globals_map = value.__globals__
    builtin_namespace = globals_map.get("__builtins__", builtins)
    if isinstance(builtin_namespace, ModuleType):
        builtin_map = vars(builtin_namespace)
    elif isinstance(builtin_namespace, dict):
        builtin_map = builtin_namespace
    else:
        raise ValueError(f"{name} builtins namespace is unsupported")
    dependencies: list[tuple[object, ...]] = []
    for dependency_name in sorted(dependency_specs):
        if dependency_name in globals_map:
            dependency = globals_map[dependency_name]
            origin = "global"
        elif dependency_name in builtin_map:
            dependency = builtin_map[dependency_name]
            origin = "builtin"
        else:
            raise ValueError(
                f"{name} global dependency {dependency_name!r} is unresolved"
            )
        attribute_paths = tuple(sorted(dependency_specs[dependency_name]))
        if inspect.isfunction(dependency) and not any(attribute_paths):
            if getattr(dependency, "__module__", "") == policy_module:
                dependency_digest = _policy_same_module_function_live_digest(
                    dependency,
                    f"{name}.global[{dependency_name}]",
                    policy_module=policy_module,
                    stack=next_stack,
                    memo=memo,
                    bind_cross_module_globals=bind_cross_module_globals,
                )
            else:
                dependency_digest = _policy_external_function_live_digest(
                    dependency,
                    f"{name}.global[{dependency_name}]",
                    bind_cross_module_globals=bind_cross_module_globals,
                )
        else:
            dependency_digest = nonfunction_live_digest(
                dependency,
                f"{name}.global[{dependency_name}]",
                attribute_paths,
            )
        dependencies.append(
            (dependency_name, origin, attribute_paths, dependency_digest)
        )

    payload = (
        "policy_same_module_python_function",
        getattr(value, "__module__", ""),
        getattr(value, "__qualname__", ""),
        _sha256_bytes(callable_source),
        _stable_code_digest(code),
        module_source_sha256,
        nonfunction_live_digest(defaults, f"{name}.__defaults__", ()),
        tuple(
            (
                key,
                nonfunction_live_digest(
                    kwdefaults[key],
                    f"{name}.__kwdefaults__[{key}]",
                    (),
                ),
            )
            for key in sorted(kwdefaults)
        ),
        tuple(closure_values),
        tuple(
            (
                key,
                nonfunction_live_digest(
                    function_attributes[key],
                    f"{name}.__dict__[{key}]",
                    (),
                ),
            )
            for key in sorted(function_attributes)
        ),
        tuple(
            (
                key,
                nonfunction_live_digest(
                    annotations[key],
                    f"{name}.__annotations__[{key}]",
                    (),
                ),
            )
            for key in sorted(annotations)
        ),
        tuple(dependencies),
    )
    digest = stable_digest("osc-private-policy-live-context-node-v2", payload)
    memo[memo_key] = digest
    return digest


def _policy_callable_implementation_digest(
    value: object,
    name: str,
    *,
    bind_cross_module_globals: bool = False,
) -> str:
    """Bind policy callable code plus its complete supported live context."""

    _require_text(name, "policy callable implementation name")
    if not inspect.isfunction(value):
        raise ValueError(f"{name} is not an inspectable Python function")
    if not isinstance(bind_cross_module_globals, bool):
        raise TypeError("cross-module global binding mode must be boolean")
    policy_module = getattr(value, "__module__", "")
    _require_text(policy_module, f"{name} module")
    return stable_digest(
        (
            "osc-private-policy-callable-implementation-v6"
            if bind_cross_module_globals
            else "osc-private-policy-callable-implementation-v5"
        ),
        {
            "live_context_schema_version": (
                "osc-private-policy-live-context-v3"
                if bind_cross_module_globals
                else "osc-private-policy-live-context-v2"
            ),
            "implementation_context_digest": (
                _policy_same_module_function_live_digest(
                    value,
                    name,
                    policy_module=policy_module,
                    stack=(),
                    memo={},
                    bind_cross_module_globals=bind_cross_module_globals,
                )
            ),
            "cross_module_dependency_boundary": (
                "complete-live-direct-dependency-graph"
                if bind_cross_module_globals
                else "source-code-module-defaults-kwdefaults-closure-attributes"
            ),
        },
    )


def _registered_mutation_operator(operator_id: str) -> object:
    _require_text(operator_id, "operator_id")
    registry = ALL_MUTATION_OPERATOR_PROFILES
    if not isinstance(registry, MappingProxyType):
        raise ValueError("mutation operator registry is not immutable")
    operator = registry.get(operator_id)
    if operator is None:
        raise ValueError("mutation operator is not registered")
    if operator.name != operator_id or not inspect.isfunction(operator.apply):
        raise ValueError("mutation operator registry entry is not replayable")
    return operator


@dataclass(frozen=True, slots=True)
class _RegisteredMutationOperatorSnapshot:
    registry: object
    operator: MutationOperator
    apply: Callable[[list[object], list[dict[str, Any]], random.Random], str]
    profile_identity: tuple[object, ...]
    digest: str


_MUTATION_OPERATOR_TUPLE_FIELDS = (
    "semantic_family_affinity",
    "semantic_signal_affinity",
    "exploration_objective_affinity",
    "divergence_affinity",
    "structural_risk_tags",
    "coverage_axes",
)
_MUTATION_OPERATOR_PROFILE_FIELDS = (
    "name",
    "apply",
    *_MUTATION_OPERATOR_TUPLE_FIELDS,
    "expandability_bias",
    "validity_floor",
)


def _capture_registered_mutation_operator_snapshot(
    operator_id: str,
) -> _RegisteredMutationOperatorSnapshot:
    registry = ALL_MUTATION_OPERATOR_PROFILES
    if not isinstance(registry, MappingProxyType):
        raise ValueError("mutation operator registry is not immutable")
    operator = _registered_mutation_operator(operator_id)
    if type(operator) is not MutationOperator:
        raise ValueError("mutation operator registry profile type is not frozen")
    profile_schema = tuple(field.name for field in fields(operator))
    if profile_schema != _MUTATION_OPERATOR_PROFILE_FIELDS:
        raise ValueError("mutation operator profile schema is not frozen")
    profile_fields: list[tuple[str, object]] = []
    for field_name in _MUTATION_OPERATOR_TUPLE_FIELDS:
        field_value = getattr(operator, field_name)
        if (
            not isinstance(field_value, tuple)
            or any(not isinstance(item, str) or not item for item in field_value)
        ):
            raise ValueError(f"mutation operator profile {field_name} is not canonical")
        profile_fields.append((field_name, field_value))
    for field_name in ("expandability_bias", "validity_floor"):
        field_value = getattr(operator, field_name)
        if (
            not isinstance(field_value, float)
            or isinstance(field_value, bool)
            or not math.isfinite(field_value)
        ):
            raise ValueError(f"mutation operator profile {field_name} is not finite")
        profile_fields.append((field_name, field_value.hex()))
    apply = operator.apply
    implementation_digest = _callable_implementation_digest(
        apply,
        f"registered mutation operator {operator_id}",
        recursive_dependencies=True,
    )
    profile_identity: tuple[object, ...] = (
        type(operator).__module__,
        type(operator).__qualname__,
        profile_schema,
        operator.name,
        getattr(apply, "__module__", type(apply).__module__),
        getattr(apply, "__qualname__", type(apply).__qualname__),
        implementation_digest,
        tuple(profile_fields),
    )
    if (
        ALL_MUTATION_OPERATOR_PROFILES is not registry
        or registry.get(operator_id) is not operator
        or operator.apply is not apply
    ):
        raise ValueError("mutation operator live registry changed during capture")
    digest = stable_digest(
        "osc-private-registered-mutation-operator-v4",
        profile_identity,
    )
    return _RegisteredMutationOperatorSnapshot(
        registry=registry,
        operator=operator,
        apply=apply,
        profile_identity=profile_identity,
        digest=digest,
    )


def _assert_registered_mutation_operator_snapshot(
    snapshot: _RegisteredMutationOperatorSnapshot,
) -> None:
    if ALL_MUTATION_OPERATOR_PROFILES is not snapshot.registry:
        raise ValueError("mutation operator live registry changed during replay")
    current = _capture_registered_mutation_operator_snapshot(
        snapshot.operator.name
    )
    if (
        current.registry is not snapshot.registry
        or current.operator is not snapshot.operator
        or current.apply is not snapshot.apply
        or current.profile_identity != snapshot.profile_identity
        or current.digest != snapshot.digest
    ):
        raise ValueError("mutation operator live identity changed during replay")


def registered_mutation_operator_digest(operator_id: str) -> str:
    """Recompute the complete live identity; cache state carries no authority."""

    return _capture_registered_mutation_operator_snapshot(operator_id).digest


def _apply_registered_mutation_operator(
    case: Case,
    mutation_seed: int,
    *,
    operator_snapshot: _RegisteredMutationOperatorSnapshot,
) -> Case:
    """Apply one already captured live profile to an isolated Case clone."""

    if not isinstance(case, Case):
        raise TypeError("registered mutation application requires a Case")
    if (
        isinstance(mutation_seed, bool)
        or not isinstance(mutation_seed, int)
        or mutation_seed < 0
    ):
        raise ValueError("registered mutation application seed must be non-negative")
    isolated = Case.from_dict(copy.deepcopy(case.to_dict()))
    raw_operations = copy.deepcopy(isolated.program.to_dict()["operations"])
    rng = random.Random()
    rng.seed(mutation_seed, version=2)
    try:
        detail = Context().run(
            operator_snapshot.apply,
            isolated.tables,
            raw_operations,
            rng,
        )
    except Exception as exc:
        raise ValueError(
            f"registered mutation operator {operator_snapshot.operator.name!r} did not replay"
        ) from exc
    if not isinstance(detail, str):
        raise ValueError("registered mutation operator returned an invalid detail")
    isolated.program = Program(
        program_id=isolated.program.program_id,
        seed=isolated.program.seed,
        operations=raw_operations,
    )
    return isolated


def mutation_operator_application_policy_id() -> str:
    """Identity of the closed selected-operator/target-preservation policy."""

    return stable_digest(
        "osc-private-mutation-operator-application-policy-v7",
        {
            "schema_version": (
                MUTATION_OPERATOR_APPLICATION_POLICY_SCHEMA_VERSION
            ),
            "registry": "datadiff.mutator.ALL_MUTATION_OPERATOR_PROFILES",
            "registry_selection": "exact-operator-id-no-reselection-or-fallback",
            "rng": {
                "type": "random.Random",
                "seed_source": "derived-SeedStage.MUTATION-subseed",
                "seed_version": 2,
                "state_version": random.Random.VERSION,
                "python_implementation": sys.implementation.name,
                "python_cache_tag": str(sys.implementation.cache_tag or ""),
            },
            "source_clone": "Case.from_dict(copy.deepcopy(case.to_dict()))",
            "ambient_context": "fresh-empty-contextvars.Context",
            "candidate_identity": "preserve-isolated-source-case-identities",
            "operator_call": (
                "captured-profile.apply(candidate.tables,canonical-raw-operations,rng)"
            ),
            "adapter_implementation_digest": _policy_callable_implementation_digest(
                _apply_registered_mutation_operator,
                "selected mutation operator adapter",
            ),
            "compiled_universe_digest": _compiled_universe().digest,
            "extractor_implementation_digest": _policy_callable_implementation_digest(
                AtomExtractor.extract,
                "AtomExtractor.extract",
            ),
            "matcher_implementation_digest": _policy_callable_implementation_digest(
                TargetMatcher.match,
                "TargetMatcher.match",
            ),
            "target_preserving_implementation_digest": (
                _policy_callable_implementation_digest(
                    TargetPreservingMutator.mutate,
                    "TargetPreservingMutator.mutate",
                    bind_cross_module_globals=True,
                )
            ),
            "repair_policy": {
                "allow_repair": True,
                "owner": "TargetPreservingMutator",
                "caller_selectable": False,
            },
            "determinism_check": (
                "one-live-profile-before-between-after-two-fresh-reconstructions"
            ),
        },
    )


@dataclass(frozen=True, slots=True)
class _RegisteredMutationReplayContext:
    operator_id: str
    operator_snapshot: _RegisteredMutationOperatorSnapshot
    application_policy_id: str


def _assert_registered_mutation_replay_context(
    context: _RegisteredMutationReplayContext,
) -> None:
    _assert_registered_mutation_operator_snapshot(context.operator_snapshot)
    if mutation_operator_application_policy_id() != context.application_policy_id:
        raise ValueError("mutation operator application policy changed during replay")


def _capture_registered_mutation_replay_context(
    operator_id: str,
) -> _RegisteredMutationReplayContext:
    snapshot = _capture_registered_mutation_operator_snapshot(operator_id)
    context = _RegisteredMutationReplayContext(
        operator_id=operator_id,
        operator_snapshot=snapshot,
        application_policy_id=mutation_operator_application_policy_id(),
    )
    _assert_registered_mutation_replay_context(context)
    return context


def _validate_typed_search_value(
    type_name: str,
    schema_version: str,
    value: object,
) -> object:
    reconstructed = _reconstruct_search_payload(
        envelope_type=type_name,
        schema_version=schema_version,
        payload=to_primitive(value),
    )
    if reconstructed != value:
        raise ValueError(f"typed {type_name} does not replay exactly")
    return reconstructed


def reconstruct_registered_mutation_outcome(
    *,
    source_case: Case,
    assignment: TargetAssignment,
    operator_id: str,
    _replay_context: _RegisteredMutationReplayContext | None = None,
) -> MutationOutcome:
    """Causally rebuild the exact outcome for one frozen selected operator."""

    if not isinstance(source_case, Case):
        raise TypeError("registered mutation reconstruction requires a Case")
    if not isinstance(assignment, TargetAssignment):
        raise TypeError(
            "registered mutation reconstruction requires a TargetAssignment"
        )
    # Keep malformed operator IDs outside the replay-context boundary so callers
    # receive the stable registry error rather than a generic replay failure.
    if _replay_context is None:
        _registered_mutation_operator(operator_id)
    try:
        replay_context = (
            _capture_registered_mutation_replay_context(operator_id)
            if _replay_context is None
            else _replay_context
        )
        if (
            replay_context.operator_id != operator_id
            or replay_context.operator_snapshot.operator.name != operator_id
        ):
            raise ValueError("selected mutation operator replay context is swapped")
        _assert_registered_mutation_replay_context(replay_context)

        def replay_once() -> MutationOutcome:
            extractor = AtomExtractor()
            mutator = TargetPreservingMutator(
                TargetMatcher(_compiled_universe()),
                extractor=extractor,
            )

            def selected_operator(candidate: Case, seed: int) -> Case:
                _assert_registered_mutation_operator_snapshot(
                    replay_context.operator_snapshot
                )
                result = _apply_registered_mutation_operator(
                    candidate,
                    seed,
                    operator_snapshot=replay_context.operator_snapshot,
                )
                _assert_registered_mutation_operator_snapshot(
                    replay_context.operator_snapshot
                )
                return result

            return mutator.mutate(
                Case.from_dict(copy.deepcopy(source_case.to_dict())),
                assignment,
                selected_operator,
                allow_repair=True,
            )

        _assert_registered_mutation_replay_context(replay_context)
        first = replay_once()
        _assert_registered_mutation_replay_context(replay_context)
        second = replay_once()
        _assert_registered_mutation_replay_context(replay_context)
    except Exception as exc:
        raise ValueError("selected mutation operator context is not replayable") from exc
    if first != second:
        raise ValueError("selected mutation operator is not deterministic")
    _validate_typed_search_value(
        "MutationOutcome", "osc-root-mutation-outcome-v1", first
    )
    return first


@dataclass(frozen=True, slots=True)
class MutationAttemptReceipt:
    plan: FormalLanePlanReceipt
    case_id: str
    case_index: int
    formal_run_id: str
    seed_block_id: str
    assignment: TargetAssignment
    source_case: CanonicalCaseBinding
    target_cell_id: str
    target_family_id: str
    operator_id: str
    operator_digest: str
    operator_application_policy_id: str
    mutation_lineage: SeedLineage
    mutation_seed: int
    outcome: MutationOutcome
    activation_certificate: ActivationCertificate
    outcome_digest: str
    runtime_task_ref: str
    attempt_id: str
    schema_version: str = MUTATION_ATTEMPT_RECEIPT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != MUTATION_ATTEMPT_RECEIPT_SCHEMA_VERSION:
            raise ValueError("mutation attempt receipt schema mismatch")
        _require_text(self.case_id, "case_id")
        _require_text(self.operator_id, "operator_id")
        _require_text(self.runtime_task_ref, "runtime_task_ref", allow_empty=True)
        run, binding = _validate_plan_assignment(
            self.plan, self.case_id, self.case_index, self.assignment
        )
        if (self.formal_run_id, self.seed_block_id) != (
            run.run_id,
            run.seed_block_id,
        ):
            raise ValueError("mutation attempt run/block is caller-substituted")
        if not isinstance(self.source_case, CanonicalCaseBinding):
            raise TypeError("mutation attempt requires a canonical source Case")
        if self.source_case.binding_case_id != self.case_id:
            raise ValueError("mutation source Case is bound to another formal case")
        source = self.source_case.case
        if source.seed != binding.seed_lineage.subseed:
            raise ValueError("mutation source Case seed is not lineage-derived")
        cells, families, _edges = _assignment_context(
            self.assignment, require_single_cell=True
        )
        if (self.target_cell_id,) != cells or (self.target_family_id,) != families:
            raise ValueError("mutation attempt cell/family is caller-substituted")
        replay_context = _capture_registered_mutation_replay_context(
            self.operator_id
        )
        if self.operator_digest != replay_context.operator_snapshot.digest:
            raise ValueError("mutation operator identity mismatch")
        if (
            self.operator_application_policy_id
            != replay_context.application_policy_id
        ):
            raise ValueError("mutation operator application policy mismatch")
        expected_lineage = derive_stage_lineage(
            binding.seed_lineage, SeedStage.MUTATION
        )
        if self.mutation_lineage != expected_lineage:
            raise ValueError("mutation lineage does not recompute exactly")
        if self.mutation_seed != expected_lineage.subseed:
            raise ValueError("mutation seed does not recompute from lineage")
        if not isinstance(self.outcome, MutationOutcome):
            raise TypeError("mutation attempt requires typed MutationOutcome")
        _validate_typed_search_value(
            "MutationOutcome", "osc-root-mutation-outcome-v1", self.outcome
        )
        if self.outcome.mutation_seed != self.mutation_seed:
            raise ValueError("mutation outcome seed is not attempt-bound")
        expected_outcome = reconstruct_registered_mutation_outcome(
            source_case=source,
            assignment=self.assignment,
            operator_id=self.operator_id,
            _replay_context=replay_context,
        )
        if self.outcome != expected_outcome:
            raise ValueError(
                "mutation outcome does not reproduce from the selected operator"
            )
        if not isinstance(self.activation_certificate, ActivationCertificate):
            raise TypeError("mutation attempt requires typed ActivationCertificate")
        if self.activation_certificate != self.outcome.certificate:
            raise ValueError("mutation certificate is not outcome-bound")
        if self.activation_certificate.assignment_digest != self.assignment.digest:
            raise ValueError("mutation certificate is not assignment-bound")
        if self.activation_certificate.selected_cell_ids != self.assignment.selected_cell_ids:
            raise ValueError("mutation certificate target cells are swapped")
        if self.outcome.status == "rejected":
            if (
                self.outcome.accepted_case is not None
                or self.activation_certificate.mutation_preserved
                or self.activation_certificate.valid
            ):
                raise ValueError("rejected mutation cannot earn preservation credit")
        elif self.outcome.status in {"accepted", "repaired"}:
            if (
                self.outcome.accepted_case is None
                or not self.activation_certificate.mutation_preserved
                or not self.activation_certificate.valid
            ):
                raise ValueError("accepted mutation lacks exact preservation evidence")
            original = AtomExtractor().extract(source)
            accepted = AtomExtractor().extract(self.outcome.accepted_case)
            if original.source_digest == accepted.source_digest:
                raise ValueError("accepted mutation is a semantic no-op")
        else:
            raise ValueError("mutation outcome status is not frozen")
        expected_outcome_digest = stable_digest(
            "osc-private-bound-mutation-outcome-v1", self.outcome
        )
        if self.outcome_digest != expected_outcome_digest:
            raise ValueError("mutation outcome digest mismatch")
        expected_attempt = _mutation_attempt_id(
            self.plan,
            self.case_id,
            self.case_index,
            self.formal_run_id,
            self.seed_block_id,
            self.assignment,
            self.source_case,
            self.target_cell_id,
            self.target_family_id,
            self.operator_id,
            self.operator_digest,
            self.operator_application_policy_id,
            self.mutation_lineage,
            self.outcome_digest,
        )
        if self.attempt_id != expected_attempt:
            raise ValueError("mutation attempt identity mismatch")

    @classmethod
    def build(
        cls,
        *,
        plan: FormalLanePlanReceipt,
        case_id: str,
        case_index: int,
        assignment: TargetAssignment,
        source_case: Case,
        operator_id: str,
        outcome: MutationOutcome,
        runtime_task_ref: str = "",
    ) -> "MutationAttemptReceipt":
        run, binding = _validate_plan_assignment(
            plan, case_id, case_index, assignment
        )
        cells, families, _edges = _assignment_context(
            assignment, require_single_cell=True
        )
        source_binding = CanonicalCaseBinding.build(
            binding_case_id=case_id, case=source_case
        )
        replay_context = _capture_registered_mutation_replay_context(operator_id)
        operator_digest = replay_context.operator_snapshot.digest
        operator_application_policy_id = replay_context.application_policy_id
        mutation_lineage = derive_stage_lineage(
            binding.seed_lineage, SeedStage.MUTATION
        )
        rebuilt_outcome = reconstruct_registered_mutation_outcome(
            source_case=source_case,
            assignment=assignment,
            operator_id=operator_id,
            _replay_context=replay_context,
        )
        if not isinstance(outcome, MutationOutcome):
            raise TypeError("mutation attempt requires typed MutationOutcome")
        if outcome != rebuilt_outcome:
            raise ValueError(
                "mutation outcome does not reproduce from the selected operator"
            )
        outcome_digest = stable_digest(
            "osc-private-bound-mutation-outcome-v1", outcome
        )
        attempt_id = _mutation_attempt_id(
            plan,
            case_id,
            case_index,
            run.run_id,
            run.seed_block_id,
            assignment,
            source_binding,
            cells[0],
            families[0],
            operator_id,
            operator_digest,
            operator_application_policy_id,
            mutation_lineage,
            outcome_digest,
        )
        return cls(
            plan,
            case_id,
            case_index,
            run.run_id,
            run.seed_block_id,
            assignment,
            source_binding,
            cells[0],
            families[0],
            operator_id,
            operator_digest,
            operator_application_policy_id,
            mutation_lineage,
            mutation_lineage.subseed,
            outcome,
            outcome.certificate,
            outcome_digest,
            runtime_task_ref,
            attempt_id,
        )

    @property
    def digest(self) -> str:
        return stable_digest("osc-private-mutation-attempt-receipt-v9", self)

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(self)


def _mutation_attempt_id(
    plan: FormalLanePlanReceipt,
    case_id: str,
    case_index: int,
    formal_run_id: str,
    seed_block_id: str,
    assignment: TargetAssignment,
    source_case: CanonicalCaseBinding,
    target_cell_id: str,
    target_family_id: str,
    operator_id: str,
    operator_digest: str,
    operator_application_policy_id: str,
    mutation_lineage: SeedLineage,
    outcome_digest: str,
) -> str:
    return _attempt_identity(
        "osc-private-mutation-attempt-id-v9",
        {
            "plan_digest": plan.digest,
            "case_id": case_id,
            "case_index": case_index,
            "formal_run_id": formal_run_id,
            "seed_block_id": seed_block_id,
            "assignment_digest": assignment.digest,
            "source_case_digest": source_case.digest,
            "target_cell_id": target_cell_id,
            "target_family_id": target_family_id,
            "operator_id": operator_id,
            "operator_digest": operator_digest,
            "operator_application_policy_id": (
                operator_application_policy_id
            ),
            "mutation_lineage_digest": mutation_lineage.digest,
            "outcome_digest": outcome_digest,
        },
    )


@dataclass(frozen=True, slots=True)
class ObservationContextReceipt:
    plan: FormalLanePlanReceipt
    case_id: str
    case_index: int
    formal_run_id: str
    seed_block_id: str
    assignment: TargetAssignment
    context_cell_ids: tuple[str, ...]
    context_family_ids: tuple[str, ...]
    context_edge_ids: tuple[str, ...]
    activation_certificate_ref: str
    runtime_task_refs: tuple[str, ...]
    runtime_outcome_refs: tuple[str, ...]
    observation_certificate_ref: str
    context_id: str
    schema_version: str = OBSERVATION_CONTEXT_RECEIPT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != OBSERVATION_CONTEXT_RECEIPT_SCHEMA_VERSION:
            raise ValueError("observation context receipt schema mismatch")
        _require_text(self.case_id, "case_id")
        run, _binding = _validate_plan_assignment(
            self.plan, self.case_id, self.case_index, self.assignment
        )
        if (self.formal_run_id, self.seed_block_id) != (
            run.run_id,
            run.seed_block_id,
        ):
            raise ValueError("observation run/block is caller-substituted")
        cells, families, edges = _assignment_context(
            self.assignment, require_single_cell=False
        )
        for name, values in (
            ("context_cell_ids", self.context_cell_ids),
            ("context_family_ids", self.context_family_ids),
        ):
            _require_unique_sorted_text(values, name)
        _require_unique_sorted_text(
            self.context_edge_ids, "context_edge_ids", allow_empty=True
        )
        if (
            self.context_cell_ids,
            self.context_family_ids,
            self.context_edge_ids,
        ) != (cells, families, edges):
            raise ValueError("observation context identities are caller-substituted")
        for name, value in (
            ("activation_certificate_ref", self.activation_certificate_ref),
            ("observation_certificate_ref", self.observation_certificate_ref),
        ):
            _require_text(value, name, allow_empty=True)
        _require_unique_sorted_text(
            self.runtime_task_refs, "runtime_task_refs", allow_empty=True
        )
        _require_unique_sorted_text(
            self.runtime_outcome_refs, "runtime_outcome_refs", allow_empty=True
        )
        expected = _observation_context_id(
            self.plan,
            self.case_id,
            self.case_index,
            self.formal_run_id,
            self.seed_block_id,
            self.assignment,
            cells,
            families,
            edges,
            self.activation_certificate_ref,
            self.runtime_task_refs,
            self.runtime_outcome_refs,
            self.observation_certificate_ref,
        )
        if self.context_id != expected:
            raise ValueError("observation context identity mismatch")

    @classmethod
    def build(
        cls,
        *,
        plan: FormalLanePlanReceipt,
        case_id: str,
        assignment: TargetAssignment,
        activation_certificate_ref: str = "",
        runtime_task_refs: Iterable[str] = (),
        runtime_outcome_refs: Iterable[str] = (),
        observation_certificate_ref: str = "",
    ) -> "ObservationContextReceipt":
        case_index = assignment.seed_lineage.case_index
        run, _binding = _validate_plan_assignment(
            plan, case_id, case_index, assignment
        )
        cells, families, edges = _assignment_context(
            assignment, require_single_cell=False
        )
        tasks = _unique_sorted_text_input(
            runtime_task_refs, "runtime_task_refs", allow_empty=True
        )
        outcomes = _unique_sorted_text_input(
            runtime_outcome_refs, "runtime_outcome_refs", allow_empty=True
        )
        context_id = _observation_context_id(
            plan,
            case_id,
            case_index,
            run.run_id,
            run.seed_block_id,
            assignment,
            cells,
            families,
            edges,
            activation_certificate_ref,
            tasks,
            outcomes,
            observation_certificate_ref,
        )
        return cls(
            plan,
            case_id,
            case_index,
            run.run_id,
            run.seed_block_id,
            assignment,
            cells,
            families,
            edges,
            activation_certificate_ref,
            tasks,
            outcomes,
            observation_certificate_ref,
            context_id,
        )

    @property
    def runtime_context_complete(self) -> bool:
        """Comparison-only convenience; it never means context is verified."""

        return bool(
            self.activation_certificate_ref
            and self.runtime_task_refs
            and self.runtime_outcome_refs
            and self.observation_certificate_ref
        )

    @property
    def digest(self) -> str:
        return stable_digest("osc-private-observation-context-receipt", self)

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(self)


def _observation_context_id(
    plan: FormalLanePlanReceipt,
    case_id: str,
    case_index: int,
    formal_run_id: str,
    seed_block_id: str,
    assignment: TargetAssignment,
    cells: tuple[str, ...],
    families: tuple[str, ...],
    edges: tuple[str, ...],
    activation_ref: str,
    task_refs: tuple[str, ...],
    outcome_refs: tuple[str, ...],
    observation_ref: str,
) -> str:
    return stable_digest(
        "osc-private-observation-context-id-v2",
        {
            "plan_digest": plan.digest,
            "case_id": case_id,
            "case_index": case_index,
            "formal_run_id": formal_run_id,
            "seed_block_id": seed_block_id,
            "assignment_digest": assignment.digest,
            "cell_ids": cells,
            "family_ids": families,
            "edge_ids": edges,
            "comparison_only_runtime_refs": {
                "activation": activation_ref,
                "tasks": task_refs,
                "outcomes": outcome_refs,
                "observation": observation_ref,
            },
        },
    )


@dataclass(frozen=True, slots=True)
class IntrinsicFocusProof:
    case_id: str
    rule: FormalFocusRule
    case_lineage: SeedLineage
    source_case: CanonicalCaseBinding
    extraction: ExtractionResult
    proof_id: str
    schema_version: str = INTRINSIC_FOCUS_PROOF_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != INTRINSIC_FOCUS_PROOF_SCHEMA_VERSION:
            raise ValueError("intrinsic focus proof schema mismatch")
        _require_text(self.case_id, "case_id")
        if not isinstance(self.rule, FormalFocusRule):
            raise TypeError("intrinsic focus proof requires FormalFocusRule")
        if self.rule != formal_focus_rule(self.rule.signal_id):
            raise ValueError("intrinsic focus rule is not frozen")
        if not isinstance(self.case_lineage, SeedLineage):
            raise TypeError("intrinsic focus proof requires typed SeedLineage")
        if self.case_lineage.stage_name is not SeedStage.TARGET:
            raise ValueError("intrinsic focus source must use TARGET lineage")
        if self.source_case.binding_case_id != self.case_id:
            raise ValueError("intrinsic focus source is bound to another case")
        case = self.source_case.case
        if case.seed != self.case_lineage.subseed:
            raise ValueError("intrinsic focus Case seed is not lineage-derived")
        regenerated = generate_case(
            self.case_lineage.subseed,
            profile=self.rule.source_generator,
        )
        expected_binding = CanonicalCaseBinding.build(
            binding_case_id=self.case_id,
            case=regenerated,
        )
        if self.source_case != expected_binding:
            raise ValueError("intrinsic focus source Case does not recompute")
        if not isinstance(self.extraction, ExtractionResult):
            raise TypeError("intrinsic focus proof requires typed ExtractionResult")
        _validate_typed_search_value(
            "ExtractionResult", "osc-atom-extraction-v1", self.extraction
        )
        recomputed_extraction = AtomExtractor().extract(case)
        if self.extraction != recomputed_extraction:
            raise ValueError("intrinsic focus atom extraction does not recompute")
        recomputed_features = extract_case_features(case)
        if self.rule.required_feature not in recomputed_features:
            raise ValueError("intrinsic focus obligation signal does not recompute")
        expected_id = _intrinsic_focus_proof_id(
            self.case_id,
            self.rule,
            self.case_lineage,
            self.source_case,
            self.extraction,
        )
        if self.proof_id != expected_id:
            raise ValueError("intrinsic focus proof identity mismatch")

    @classmethod
    def build(
        cls,
        *,
        plan: FormalLanePlanReceipt,
        case_id: str,
        obligation_id: str,
    ) -> "IntrinsicFocusProof":
        obligation = plan.registry.obligation(obligation_id)
        if obligation.obligation_id not in plan.focus_obligation_ids:
            raise ValueError("focus obligation is outside the formal lane plan")
        _run, binding = plan.case_run_binding(case_id)
        rule = formal_focus_rule(obligation.signal_id)
        case = generate_case(
            binding.seed_lineage.subseed,
            profile=rule.source_generator,
        )
        source = CanonicalCaseBinding.build(binding_case_id=case_id, case=case)
        extraction = AtomExtractor().extract(case)
        proof_id = _intrinsic_focus_proof_id(
            case_id, rule, binding.seed_lineage, source, extraction
        )
        return cls(
            case_id,
            rule,
            binding.seed_lineage,
            source,
            extraction,
            proof_id,
        )

    @property
    def signal_id(self) -> str:
        return self.rule.signal_id

    @property
    def digest(self) -> str:
        return stable_digest("osc-private-intrinsic-focus-proof", self)


def _intrinsic_focus_proof_id(
    case_id: str,
    rule: FormalFocusRule,
    lineage: SeedLineage,
    source_case: CanonicalCaseBinding,
    extraction: ExtractionResult,
) -> str:
    return stable_digest(
        "osc-private-intrinsic-focus-proof-id-v1",
        {
            "case_id": case_id,
            "rule_digest": rule.digest,
            "case_lineage_digest": lineage.digest,
            "source_case_digest": source_case.digest,
            "extraction_digest": extraction.digest,
        },
    )


@dataclass(frozen=True, slots=True)
class FocusHitReceipt:
    plan: FormalLanePlanReceipt
    observation_context: ObservationContextReceipt
    obligation: FormalFocusObligation
    lane_id: str
    signal_id: str
    case_id: str
    intrinsic_proofs: tuple[IntrinsicFocusProof, ...]
    focus_context_id: str
    schema_version: str = FOCUS_HIT_RECEIPT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != FOCUS_HIT_RECEIPT_SCHEMA_VERSION:
            raise ValueError("focus hit receipt schema mismatch")
        if self.plan != self.observation_context.plan:
            raise ValueError("focus receipt plan is not observation-bound")
        if self.case_id != self.observation_context.case_id:
            raise ValueError("focus receipt case is not observation-bound")
        expected_obligation = self.plan.registry.obligation(
            self.obligation.obligation_id
        )
        if self.obligation != expected_obligation:
            raise ValueError("focus receipt obligation is caller-substituted")
        if self.obligation.obligation_id not in self.plan.focus_obligation_ids:
            raise ValueError("focus obligation is outside the formal lane plan")
        if (self.lane_id, self.signal_id) != (
            self.obligation.lane_id,
            self.obligation.signal_id,
        ):
            raise ValueError("focus lane/signal identity is caller-substituted")
        if self.lane_id != self.plan.lane_id:
            raise ValueError("focus obligation belongs to another lane")
        if not isinstance(self.intrinsic_proofs, tuple) or not self.intrinsic_proofs:
            raise ValueError("focus context requires intrinsic proofs")
        proof_ids = tuple(item.proof_id for item in self.intrinsic_proofs)
        if _has_duplicate(proof_ids):
            raise ValueError("focus evidence contains duplicates")
        if self.intrinsic_proofs != tuple(
            sorted(self.intrinsic_proofs, key=lambda item: item.proof_id)
        ):
            raise ValueError("focus intrinsic proofs must be proof-sorted")
        for proof in self.intrinsic_proofs:
            if (
                proof.case_id != self.case_id
                or proof.signal_id != self.signal_id
                or proof.case_lineage != self.observation_context.assignment.seed_lineage
            ):
                raise ValueError("focus proof case, lineage or signal is swapped")
        expected_id = _focus_context_id(
            self.plan,
            self.observation_context,
            self.obligation,
            self.case_id,
            self.intrinsic_proofs,
        )
        if self.focus_context_id != expected_id:
            raise ValueError("focus context identity mismatch")

    @classmethod
    def build(
        cls,
        *,
        plan: FormalLanePlanReceipt,
        observation_context: ObservationContextReceipt,
        obligation_id: str,
        intrinsic_proofs: Iterable[IntrinsicFocusProof],
    ) -> "FocusHitReceipt":
        obligation = plan.registry.obligation(obligation_id)
        proofs = _unique_sorted_objects(
            intrinsic_proofs,
            "intrinsic_proofs",
            identity=lambda item: item.proof_id,
        )
        focus_id = _focus_context_id(
            plan,
            observation_context,
            obligation,
            observation_context.case_id,
            proofs,
        )
        return cls(
            plan,
            observation_context,
            obligation,
            obligation.lane_id,
            obligation.signal_id,
            observation_context.case_id,
            proofs,
            focus_id,
        )

    @property
    def digest(self) -> str:
        return stable_digest("osc-private-focus-hit-receipt", self)

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(self)


def _focus_context_id(
    plan: FormalLanePlanReceipt,
    observation: ObservationContextReceipt,
    obligation: FormalFocusObligation,
    case_id: str,
    proofs: tuple[IntrinsicFocusProof, ...],
) -> str:
    return stable_digest(
        "osc-private-focus-hit-context-id-v2",
        {
            "plan_digest": plan.digest,
            "observation_context_digest": observation.digest,
            "obligation_digest": obligation.digest,
            "case_id": case_id,
            "intrinsic_proof_digests": tuple(item.digest for item in proofs),
        },
    )


@dataclass(frozen=True, slots=True)
class SearchContextSubjectSpec:
    subject_kind: str
    producer_type: str
    identity_kind: str
    runtime_context_required: bool
    schema_version: str = SEARCH_CONTEXT_SUBJECT_SPEC_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SEARCH_CONTEXT_SUBJECT_SPEC_SCHEMA_VERSION:
            raise ValueError("Search context subject spec schema mismatch")
        for name, value in (
            ("subject_kind", self.subject_kind),
            ("producer_type", self.producer_type),
            ("identity_kind", self.identity_kind),
        ):
            _require_text(value, name)


# This is a context routing table, not a gate specification or threshold set.
SEARCH_CONTEXT_SUBJECT_SPECS: tuple[SearchContextSubjectSpec, ...] = (
    SearchContextSubjectSpec("formal_lanes", "FormalLaneRegistry", "lane_id", False),
    SearchContextSubjectSpec(
        "formal_lane_plans", "FormalLanePlanReceipt", "lane_id", False
    ),
    SearchContextSubjectSpec(
        "focus_signals", "FormalLaneRegistry", "lane_signal_obligation_id", False
    ),
    SearchContextSubjectSpec(
        "unique_focus_signals", "FormalLaneRegistry", "signal_id", False
    ),
    SearchContextSubjectSpec(
        "fresh_cells_declared", "IntrinsicSearchUniverse", "target_cell_id", False
    ),
    SearchContextSubjectSpec(
        "contrast_edges_declared", "IntrinsicSearchUniverse", "contrast_edge_id", False
    ),
    SearchContextSubjectSpec(
        "backend_pair_obligations_declared",
        "IntrinsicSearchUniverse",
        "backend_pair_obligation_id",
        False,
    ),
    SearchContextSubjectSpec(
        "scheduled_target_attempts",
        "ScheduledTargetAttemptReceipt",
        "attempt_id",
        True,
    ),
    SearchContextSubjectSpec(
        "mutation_attempts", "MutationAttemptReceipt", "attempt_id", True
    ),
    SearchContextSubjectSpec(
        "observation_contexts",
        "ObservationContextReceipt",
        "context_id",
        True,
    ),
    SearchContextSubjectSpec(
        "focus_hits", "FocusHitReceipt", "focus_context_id", True
    ),
)


def search_context_subject_spec(subject_kind: str) -> SearchContextSubjectSpec:
    for item in SEARCH_CONTEXT_SUBJECT_SPECS:
        if item.subject_kind == subject_kind:
            return item
    raise KeyError(subject_kind)
