"""Deterministic DataFusion grouped-null TopK witnesses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datadiff.dsl import Case, ColumnSpec, Program, TableData
from datadiff.family_witness_registry import (
    FamilyWitnessBitmap,
    family_witness_registration,
)


DATAFUSION_GROUPED_NULL_TOPK_WITNESS_SCHEMA_VERSION = (
    "datafusion-grouped-null-topk-witness-v1"
)
DATAFUSION_GROUPED_NULL_TOPK_FAMILY_ID = "datafusion_grouped_null_topk"
DATAFUSION_GROUPED_NULL_TOPK_GOAL_ID = "datafusion_grouped_null_topk"
DATAFUSION_GROUPED_NULL_TOPK_ROOT_ID = "datafusion-grouped-null-topk-001"
DATAFUSION_GROUPED_NULL_TOPK_AGGREGATES = ("min", "max")
DATAFUSION_GROUPED_NULL_TOPK_EXPOSURE_MODES = (
    "full_nulls_last",
    "top1_nulls_first",
)
DATAFUSION_GROUPED_NULL_TOPK_DATA_PATTERNS = (
    "singleton_null",
    "null_value_groups",
    "repeated_null_group",
)

_PATTERN_ROWS: dict[str, tuple[tuple[str, int | None], ...]] = {
    "singleton_null": (("a", None),),
    "null_value_groups": (("a", None), ("b", 1)),
    "repeated_null_group": (("a", None), ("a", None), ("b", 2)),
}

_REGISTRATION = family_witness_registration(
    DATAFUSION_GROUPED_NULL_TOPK_FAMILY_ID
)
DATAFUSION_GROUPED_NULL_TOPK_CELL_COUNT = _REGISTRATION.cell_count

if _REGISTRATION.axis_values != (
    DATAFUSION_GROUPED_NULL_TOPK_AGGREGATES,
    DATAFUSION_GROUPED_NULL_TOPK_EXPOSURE_MODES,
    DATAFUSION_GROUPED_NULL_TOPK_DATA_PATTERNS,
):
    raise AssertionError("DataFusion grouped-null TopK registry axes mismatch")


@dataclass(frozen=True, slots=True)
class DataFusionGroupedNullTopKWitnessCell:
    cell_index: int
    aggregate: str
    exposure_mode: str
    data_pattern_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": DATAFUSION_GROUPED_NULL_TOPK_WITNESS_SCHEMA_VERSION,
            "cell_index": self.cell_index,
            "aggregate": self.aggregate,
            "exposure_mode": self.exposure_mode,
            "data_pattern_id": self.data_pattern_id,
        }


@dataclass(frozen=True, slots=True)
class DataFusionGroupedNullTopKGenerationResult:
    case: Case
    trace: dict[str, Any]


class DataFusionGroupedNullTopKBitmap(FamilyWitnessBitmap):
    def __init__(self) -> None:
        super().__init__(_REGISTRATION)


def datafusion_grouped_null_topk_witness_cell(
    seed: int,
) -> DataFusionGroupedNullTopKWitnessCell:
    cell_index, axes = _REGISTRATION.cell_for_seed(seed)
    return DataFusionGroupedNullTopKWitnessCell(
        cell_index=cell_index,
        aggregate=axes["aggregate"],
        exposure_mode=axes["exposure_mode"],
        data_pattern_id=axes["data_pattern"],
    )


def datafusion_grouped_null_topk_expected_values(
    *,
    aggregate: str,
    exposure_mode: str,
    rows: tuple[tuple[str, int | None], ...] | list[tuple[str, int | None]],
) -> list[int | None]:
    grouped: dict[str, list[int | None]] = {}
    for group, value in rows:
        grouped.setdefault(str(group), []).append(value)
    values: list[int | None] = []
    for group_values in grouped.values():
        nonnull = [value for value in group_values if value is not None]
        if not nonnull:
            values.append(None)
        elif aggregate == "min":
            values.append(min(nonnull))
        elif aggregate == "max":
            values.append(max(nonnull))
        else:
            raise ValueError(f"unsupported aggregate: {aggregate}")

    if exposure_mode == "full_nulls_last":
        nulls = "last"
        limit = 20
    elif exposure_mode == "top1_nulls_first":
        nulls = "first"
        limit = 1
    else:
        raise ValueError(f"unsupported exposure mode: {exposure_mode}")
    ascending = aggregate == "min"

    def sort_key(value: int | None) -> tuple[int, int]:
        if value is None:
            return (1 if nulls == "last" else 0, 0)
        return (0 if nulls == "last" else 1, value if ascending else -value)

    return sorted(values, key=sort_key)[:limit]


def generate_datafusion_grouped_null_topk_witness_case(
    seed: int,
    *,
    profile: str = "",
) -> DataFusionGroupedNullTopKGenerationResult:
    """Build one bounded in-memory grouped-null TopK contract probe."""

    cell = datafusion_grouped_null_topk_witness_cell(seed)
    pattern_rows = _PATTERN_ROWS[cell.data_pattern_id]
    direction = "asc" if cell.aggregate == "min" else "desc"
    nulls, limit = (
        ("last", 20)
        if cell.exposure_mode == "full_nulls_last"
        else ("first", 1)
    )
    expected_values = datafusion_grouped_null_topk_expected_values(
        aggregate=cell.aggregate,
        exposure_mode=cell.exposure_mode,
        rows=pattern_rows,
    )
    table = TableData(
        "t0",
        [
            ColumnSpec("g", "str", nullable=False),
            ColumnSpec("x", "int", nullable=True),
        ],
        [{"g": group, "x": value} for group, value in pattern_rows],
    )
    operation = {
        "op": "datafusion_grouped_null_topk_probe",
        "as": "grouped_null_topk_mismatch",
        "aggregate": cell.aggregate,
        "exposure_mode": cell.exposure_mode,
        "data_pattern_id": cell.data_pattern_id,
        "group_column": "g",
        "value_column": "x",
        "direction": direction,
        "nulls": nulls,
        "limit": limit,
        "expected_values": list(expected_values),
    }
    program = Program(
        f"prog-{int(seed):08d}-datafusion-null-topk-{cell.cell_index:02d}",
        int(seed),
        [operation],
    )
    family_witness = {
        "schema_version": "family-witness-case-v1",
        "family_id": DATAFUSION_GROUPED_NULL_TOPK_FAMILY_ID,
        "generation_mode": "goal_first_witness_v6",
        "goal_id": DATAFUSION_GROUPED_NULL_TOPK_GOAL_ID,
        "root_ids": [DATAFUSION_GROUPED_NULL_TOPK_ROOT_ID],
        "cell_index": cell.cell_index,
        "axes": {
            "aggregate": cell.aggregate,
            "exposure_mode": cell.exposure_mode,
            "data_pattern": cell.data_pattern_id,
        },
        "palette_dimensions": [
            cell.aggregate,
            cell.exposure_mode,
            cell.data_pattern_id,
        ],
        "canonical_case_replay": False,
        "runtime_corpus_io": False,
    }
    null_groups = {
        group
        for group, _value in pattern_rows
        if all(
            candidate is None
            for candidate_group, candidate in pattern_rows
            if candidate_group == group
        )
    }
    witness = {
        **cell.to_dict(),
        "goal_id": DATAFUSION_GROUPED_NULL_TOPK_GOAL_ID,
        "root_ids": [DATAFUSION_GROUPED_NULL_TOPK_ROOT_ID],
        "rows": [
            {"g": group, "x": value} for group, value in pattern_rows
        ],
        "direction": direction,
        "nulls": nulls,
        "limit": limit,
        "expected_values": list(expected_values),
        "group_count": len({group for group, _value in pattern_rows}),
        "all_null_group_count": len(null_groups),
        "canonical_case_replay": False,
        "runtime_corpus_io": False,
        "bounded_for_cache_reuse": True,
    }
    variant_id = f"{cell.aggregate}:{cell.exposure_mode}"
    trace = {
        "schema_version": DATAFUSION_GROUPED_NULL_TOPK_WITNESS_SCHEMA_VERSION,
        "generation_mode": "goal_first_witness_v6",
        "selected_goal": {
            "goal_id": DATAFUSION_GROUPED_NULL_TOPK_GOAL_ID,
            "target_contract": "grouped_null_aggregate_row_survives_ordered_topk",
            "fault_models": ["topk_null_group_loss"],
        },
        "selection_reason": "bounded_aggregate_exposure_pattern_cycle_v1",
        "builder_variant": {
            "variant_id": variant_id,
            "variant_family": "v6",
            "aggregate": cell.aggregate,
            "exposure_mode": cell.exposure_mode,
            "data_pattern": {
                "pattern_id": cell.data_pattern_id,
                "pattern_index": DATAFUSION_GROUPED_NULL_TOPK_DATA_PATTERNS.index(
                    cell.data_pattern_id
                ),
                "pattern_count": len(
                    DATAFUSION_GROUPED_NULL_TOPK_DATA_PATTERNS
                ),
            },
            "cell_index": cell.cell_index,
            "construction_seed": cell.cell_index,
            "case_seed": int(seed),
            "root_guidance": [DATAFUSION_GROUPED_NULL_TOPK_ROOT_ID],
        },
        "requested_profile": str(profile or ""),
        "activation_requirement": {
            "required": True,
            "policy": "deterministic_native_grouped_null_topk_witness",
        },
        "datafusion_grouped_null_topk_witness": witness,
        "family_witness": family_witness,
        "valid": True,
        "constructible": True,
        "skip_reason": "",
    }
    case = Case(
        case_id=(
            f"case-{int(seed):08d}-datafusion-null-topk-"
            f"{cell.cell_index:02d}"
        ),
        seed=int(seed),
        tables=[table],
        program=program,
        metadata={
            "generator_profile": "datafusion_grouped_null_topk_witness",
            "generation_mode": "goal_first_witness_v6",
            "goal_id": DATAFUSION_GROUPED_NULL_TOPK_GOAL_ID,
            "goal_fault_models": ["topk_null_group_loss"],
            "semantic_activation_syntactic_reached": True,
            "goal_first_generation": trace,
            "goal_builder_variant": dict(trace["builder_variant"]),
            "semantic_witness_builder_variant": variant_id,
            "semantic_witness_data_pattern": dict(
                trace["builder_variant"]["data_pattern"]
            ),
            "datafusion_grouped_null_topk_witness": witness,
            "family_witness": family_witness,
        },
    )
    from datadiff.semantic_core.activation import evaluate_semantic_activation

    activation = evaluate_semantic_activation(
        case,
        goal_id=DATAFUSION_GROUPED_NULL_TOPK_GOAL_ID,
        syntactic_reached=True,
    ).to_dict()
    case.metadata["semantic_activation"] = activation
    trace["semantic_activation"] = activation
    return DataFusionGroupedNullTopKGenerationResult(case=case, trace=trace)
