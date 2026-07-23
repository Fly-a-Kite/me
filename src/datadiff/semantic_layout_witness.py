"""Bounded PyArrow physical-layout witnesses for Boolean group-by kernels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from datadiff.compact_novelty import DenseProductBitmap
from datadiff.dsl import Case, ColumnSpec, Program, TableData


PHYSICAL_LAYOUT_WITNESS_SCHEMA_VERSION = "pyarrow-layout-witness-v1"
PHYSICAL_LAYOUT_WITNESS_BITMAP_SCHEMA_VERSION = (
    "pyarrow-layout-witness-trigger-bitmap-v1"
)
PHYSICAL_LAYOUT_WITNESS_GOAL_ID = "pyarrow_layout_bool_groupby"
PHYSICAL_LAYOUT_WITNESS_ROOT_ID = "pyarrow-sliced-bool-hash-aggregate-001"
PHYSICAL_LAYOUT_WITNESS_LAYOUTS = ("contiguous", "sliced", "chunked")
PHYSICAL_LAYOUT_WITNESS_VARIANTS = ("any_all", "all_any")
PHYSICAL_LAYOUT_WITNESS_DATA_PATTERNS = (
    "single_group_false_null",
    "two_groups_false_null",
    "null_key_false_null",
)
PHYSICAL_LAYOUT_WITNESS_CELL_COUNT = (
    len(PHYSICAL_LAYOUT_WITNESS_LAYOUTS)
    * len(PHYSICAL_LAYOUT_WITNESS_VARIANTS)
    * len(PHYSICAL_LAYOUT_WITNESS_DATA_PATTERNS)
)

_PATTERN_ROWS: dict[str, tuple[tuple[int | None, bool | None], ...]] = {
    "single_group_false_null": ((10, False), (10, None)),
    "two_groups_false_null": ((10, False), (11, None)),
    "null_key_false_null": ((10, False), (None, None)),
}


@dataclass(frozen=True, slots=True)
class PhysicalLayoutWitnessCell:
    cell_index: int
    layout: str
    variant_id: str
    data_pattern_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PHYSICAL_LAYOUT_WITNESS_SCHEMA_VERSION,
            "cell_index": self.cell_index,
            "layout": self.layout,
            "variant_id": self.variant_id,
            "data_pattern_id": self.data_pattern_id,
        }


@dataclass(frozen=True, slots=True)
class PhysicalLayoutWitnessGenerationResult:
    case: Case
    trace: dict[str, Any]


@dataclass(frozen=True, slots=True)
class PhysicalLayoutTriggerObservation:
    registered: bool
    cell_index: int | None
    first_seen: bool | None
    layout: str
    variant_id: str
    data_pattern_id: str
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PHYSICAL_LAYOUT_WITNESS_BITMAP_SCHEMA_VERSION,
            "registered": self.registered,
            "cell_index": self.cell_index,
            "first_seen": self.first_seen,
            "layout": self.layout,
            "variant_id": self.variant_id,
            "data_pattern_id": self.data_pattern_id,
            "reason": self.reason,
        }


class PhysicalLayoutWitnessBitmap:
    """Exact shadow bitmap over layout x variant x data pattern."""

    __slots__ = ("_product", "unregistered_observations")

    def __init__(self) -> None:
        self._product = DenseProductBitmap(
            (
                PHYSICAL_LAYOUT_WITNESS_LAYOUTS,
                PHYSICAL_LAYOUT_WITNESS_VARIANTS,
                PHYSICAL_LAYOUT_WITNESS_DATA_PATTERNS,
            )
        )
        self.unregistered_observations = 0

    def observe_case(self, case: Case) -> PhysicalLayoutTriggerObservation:
        metadata = case.metadata if isinstance(case.metadata, Mapping) else {}
        witness = metadata.get("physical_layout_witness", {})
        if not isinstance(witness, Mapping):
            witness = {}
        layout = str(witness.get("layout", "") or "")
        variant_id = str(witness.get("variant_id", "") or "")
        data_pattern_id = str(witness.get("data_pattern_id", "") or "")
        try:
            cell_index, first_seen = self._product.observe(
                (layout, variant_id, data_pattern_id)
            )
        except (KeyError, ValueError) as exc:
            self.unregistered_observations += 1
            return PhysicalLayoutTriggerObservation(
                registered=False,
                cell_index=None,
                first_seen=None,
                layout=layout,
                variant_id=variant_id,
                data_pattern_id=data_pattern_id,
                reason=f"unregistered_layout_witness_dimension:{exc}",
            )
        declared_index = witness.get("cell_index")
        if declared_index is not None and int(declared_index) != cell_index:
            self.unregistered_observations += 1
            return PhysicalLayoutTriggerObservation(
                registered=False,
                cell_index=cell_index,
                first_seen=first_seen,
                layout=layout,
                variant_id=variant_id,
                data_pattern_id=data_pattern_id,
                reason=(
                    "layout_witness_cell_index_mismatch:"
                    f"declared={declared_index},encoded={cell_index}"
                ),
            )
        return PhysicalLayoutTriggerObservation(
            registered=True,
            cell_index=cell_index,
            first_seen=first_seen,
            layout=layout,
            variant_id=variant_id,
            data_pattern_id=data_pattern_id,
        )

    def snapshot(self, *, include_data: bool = False) -> dict[str, Any]:
        return {
            **self._product.snapshot(include_data=include_data),
            "schema_version": PHYSICAL_LAYOUT_WITNESS_BITMAP_SCHEMA_VERSION,
            "dimension_names": ["layout", "variant", "data_pattern"],
            "dimension_values": [
                list(PHYSICAL_LAYOUT_WITNESS_LAYOUTS),
                list(PHYSICAL_LAYOUT_WITNESS_VARIANTS),
                list(PHYSICAL_LAYOUT_WITNESS_DATA_PATTERNS),
            ],
            "unregistered_observation_count": self.unregistered_observations,
            "behavioral_role": "shadow_first_seen_hint_only",
            "case_discard_authority": False,
            "confirmed_bug_dedupe_authority": False,
        }


def physical_layout_witness_cell(seed: int) -> PhysicalLayoutWitnessCell:
    cell_index = int(seed) % PHYSICAL_LAYOUT_WITNESS_CELL_COUNT
    cells_per_layout = (
        len(PHYSICAL_LAYOUT_WITNESS_VARIANTS)
        * len(PHYSICAL_LAYOUT_WITNESS_DATA_PATTERNS)
    )
    layout_index, remainder = divmod(cell_index, cells_per_layout)
    variant_index, data_pattern_index = divmod(
        remainder,
        len(PHYSICAL_LAYOUT_WITNESS_DATA_PATTERNS),
    )
    return PhysicalLayoutWitnessCell(
        cell_index=cell_index,
        layout=PHYSICAL_LAYOUT_WITNESS_LAYOUTS[layout_index],
        variant_id=PHYSICAL_LAYOUT_WITNESS_VARIANTS[variant_index],
        data_pattern_id=PHYSICAL_LAYOUT_WITNESS_DATA_PATTERNS[data_pattern_index],
    )


def generate_physical_layout_witness_case(
    seed: int,
    *,
    profile: str = "",
) -> PhysicalLayoutWitnessGenerationResult:
    """Build one in-memory cell without consulting the canonical corpus."""

    cell = physical_layout_witness_cell(seed)
    rows = [
        {"g": group, "flag": flag}
        for group, flag in _PATTERN_ROWS[cell.data_pattern_id]
    ]
    aggregates = [
        {"column": "flag", "func": function, "as": f"flag_{function}"}
        for function in (
            ("any", "all") if cell.variant_id == "any_all" else ("all", "any")
        )
    ]
    table = TableData(
        "t0",
        [
            ColumnSpec("g", "int", nullable=True),
            ColumnSpec("flag", "bool", nullable=True),
        ],
        rows,
    )
    program = Program(
        f"prog-{int(seed):08d}-pyarrow-layout-{cell.cell_index:02d}",
        int(seed),
        [{"op": "groupby", "keys": ["g"], "aggs": aggregates}],
    )
    data_pattern_index = PHYSICAL_LAYOUT_WITNESS_DATA_PATTERNS.index(
        cell.data_pattern_id
    )
    variant_index = PHYSICAL_LAYOUT_WITNESS_VARIANTS.index(cell.variant_id)
    witness = {
        **cell.to_dict(),
        "goal_id": PHYSICAL_LAYOUT_WITNESS_GOAL_ID,
        "root_ids": [PHYSICAL_LAYOUT_WITNESS_ROOT_ID],
        "variant_index": variant_index,
        "data_pattern_index": data_pattern_index,
        "slice_offset": 1 if cell.layout == "sliced" else 0,
        "chunk_count": 2 if cell.layout == "chunked" else 1,
        "canonical_case_replay": False,
        "runtime_corpus_io": False,
        "bounded_for_cache_reuse": True,
    }
    family_witness = {
        "schema_version": "family-witness-case-v1",
        "family_id": "pyarrow_sliced_bool_groupby_any_all",
        "generation_mode": "goal_first_witness_v4",
        "goal_id": PHYSICAL_LAYOUT_WITNESS_GOAL_ID,
        "root_ids": [PHYSICAL_LAYOUT_WITNESS_ROOT_ID],
        "cell_index": cell.cell_index,
        "axes": {
            "layout": cell.layout,
            "variant": cell.variant_id,
            "data_pattern": cell.data_pattern_id,
        },
        "palette_dimensions": [
            cell.layout,
            cell.variant_id,
            cell.data_pattern_id,
        ],
        "canonical_case_replay": False,
        "runtime_corpus_io": False,
    }
    trace = {
        "schema_version": PHYSICAL_LAYOUT_WITNESS_SCHEMA_VERSION,
        "generation_mode": "goal_first_witness_v4",
        "selected_goal": {
            "goal_id": PHYSICAL_LAYOUT_WITNESS_GOAL_ID,
            "target_contract": (
                "sliced_boolean_groupby_any_all_matches_zero_offset_layouts"
            ),
            "fault_models": [
                "boolean_value_bitmap_offset",
                "hash_aggregate_layout_semantics",
            ],
        },
        "selection_reason": "bounded_layout_variant_pattern_cycle_v1",
        "builder_variant": {
            "variant_id": cell.variant_id,
            "variant_family": "v4",
            "selection_index": variant_index,
            "data_pattern": {
                "pattern_id": cell.data_pattern_id,
                "pattern_index": data_pattern_index,
                "pattern_count": len(PHYSICAL_LAYOUT_WITNESS_DATA_PATTERNS),
            },
            "layout": cell.layout,
            "cell_index": cell.cell_index,
            "construction_seed": cell.cell_index,
            "case_seed": int(seed),
            "epoch_reuse": int(seed) != cell.cell_index,
            "root_guidance": [PHYSICAL_LAYOUT_WITNESS_ROOT_ID],
        },
        "requested_profile": str(profile or ""),
        "activation_requirement": {
            "required": True,
            "policy": "deterministic_physical_layout_static_witness",
        },
        "physical_layout_witness": witness,
        "family_witness": family_witness,
        "valid": True,
        "constructible": True,
        "skip_reason": "",
    }
    case = Case(
        case_id=f"case-{int(seed):08d}-pyarrow-layout-{cell.cell_index:02d}",
        seed=int(seed),
        tables=[table],
        program=program,
        metadata={
            "generator_profile": "pyarrow_layout_bool_groupby_witness",
            "generation_mode": "goal_first_witness_v4",
            "goal_id": PHYSICAL_LAYOUT_WITNESS_GOAL_ID,
            "goal_fault_models": [
                "boolean_value_bitmap_offset",
                "hash_aggregate_layout_semantics",
            ],
            "semantic_activation_syntactic_reached": True,
            "goal_first_generation": trace,
            "goal_builder_variant": dict(trace["builder_variant"]),
            "semantic_witness_builder_variant": cell.variant_id,
            "semantic_witness_data_pattern": dict(
                trace["builder_variant"]["data_pattern"]
            ),
            "input_layouts": {
                "t0": {
                    "representation": cell.layout,
                    "chunk_count": witness["chunk_count"],
                    "dictionary_columns": [],
                    "slice_offset": witness["slice_offset"],
                    "layout_witness_root_id": PHYSICAL_LAYOUT_WITNESS_ROOT_ID,
                }
            },
            "physical_layout_witness": witness,
            "family_witness": family_witness,
        },
    )
    from datadiff.semantic_core.activation import evaluate_semantic_activation

    activation = evaluate_semantic_activation(
        case,
        goal_id=PHYSICAL_LAYOUT_WITNESS_GOAL_ID,
        syntactic_reached=True,
    ).to_dict()
    case.metadata["semantic_activation"] = activation
    trace["semantic_activation"] = activation
    return PhysicalLayoutWitnessGenerationResult(case=case, trace=trace)
