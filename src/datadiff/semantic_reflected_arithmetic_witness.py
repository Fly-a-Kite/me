"""Deterministic Polars reflected-arithmetic operand-order witnesses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datadiff.dsl import Case, ColumnSpec, Program, TableData
from datadiff.family_witness_registry import (
    FamilyWitnessBitmap,
    family_witness_registration,
)


REFLECTED_ARITHMETIC_WITNESS_SCHEMA_VERSION = (
    "polars-reflected-arithmetic-witness-v1"
)
REFLECTED_ARITHMETIC_FAMILY_ID = "polars_reflected_arithmetic_operand_order"
REFLECTED_ARITHMETIC_GOAL_ID = "polars_reflected_arithmetic"
REFLECTED_ARITHMETIC_ROOT_ID = (
    "polars-reflected-arithmetic-operand-order-001"
)
REFLECTED_ARITHMETIC_OPERATORS = (
    "rsub",
    "rtruediv",
    "rfloordiv",
    "rmod",
    "rpow",
)
REFLECTED_ARITHMETIC_NAME_MODES = ("distinct_names", "shared_name")
REFLECTED_ARITHMETIC_DATA_PATTERNS = (
    "canonical_positive",
    "unit_boundary",
    "mixed_order",
)

_PATTERN_VALUES: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {
    "canonical_positive": ((2, 3, 4), (5, 7, 9)),
    "unit_boundary": ((1, 2, 3), (2, 5, 4)),
    "mixed_order": ((4, 2, 3), (3, 8, 5)),
}

_REGISTRATION = family_witness_registration(REFLECTED_ARITHMETIC_FAMILY_ID)
REFLECTED_ARITHMETIC_CELL_COUNT = _REGISTRATION.cell_count

if _REGISTRATION.axis_values != (
    REFLECTED_ARITHMETIC_OPERATORS,
    REFLECTED_ARITHMETIC_NAME_MODES,
    REFLECTED_ARITHMETIC_DATA_PATTERNS,
):
    raise AssertionError("reflected-arithmetic registry axes do not match builder axes")


@dataclass(frozen=True, slots=True)
class ReflectedArithmeticWitnessCell:
    cell_index: int
    operator: str
    name_mode: str
    data_pattern_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": REFLECTED_ARITHMETIC_WITNESS_SCHEMA_VERSION,
            "cell_index": self.cell_index,
            "operator": self.operator,
            "name_mode": self.name_mode,
            "data_pattern_id": self.data_pattern_id,
        }


@dataclass(frozen=True, slots=True)
class ReflectedArithmeticWitnessGenerationResult:
    case: Case
    trace: dict[str, Any]


class ReflectedArithmeticWitnessBitmap(FamilyWitnessBitmap):
    def __init__(self) -> None:
        super().__init__(_REGISTRATION)


def reflected_arithmetic_witness_cell(seed: int) -> ReflectedArithmeticWitnessCell:
    cell_index, axes = _REGISTRATION.cell_for_seed(seed)
    return ReflectedArithmeticWitnessCell(
        cell_index=cell_index,
        operator=axes["operator"],
        name_mode=axes["name_mode"],
        data_pattern_id=axes["data_pattern"],
    )


def reflected_arithmetic_expected_values(
    operator: str,
    lhs_values: tuple[int, ...] | list[int],
    rhs_values: tuple[int, ...] | list[int],
) -> list[int | float]:
    return [
        _apply_operator(operator, rhs, lhs)
        for lhs, rhs in zip(lhs_values, rhs_values, strict=True)
    ]


def reflected_arithmetic_reversed_values(
    operator: str,
    lhs_values: tuple[int, ...] | list[int],
    rhs_values: tuple[int, ...] | list[int],
) -> list[int | float]:
    return [
        _apply_operator(operator, lhs, rhs)
        for lhs, rhs in zip(lhs_values, rhs_values, strict=True)
    ]


def generate_reflected_arithmetic_witness_case(
    seed: int,
    *,
    profile: str = "",
) -> ReflectedArithmeticWitnessGenerationResult:
    """Build one bounded native-Series contract probe without corpus I/O."""

    cell = reflected_arithmetic_witness_cell(seed)
    lhs_values, rhs_values = _PATTERN_VALUES[cell.data_pattern_id]
    lhs_name, rhs_name = (
        ("lhs", "rhs")
        if cell.name_mode == "distinct_names"
        else ("value", "value")
    )
    expected_values = reflected_arithmetic_expected_values(
        cell.operator,
        lhs_values,
        rhs_values,
    )
    reversed_values = reflected_arithmetic_reversed_values(
        cell.operator,
        lhs_values,
        rhs_values,
    )
    table = TableData(
        "t0",
        [ColumnSpec("probe_id", "int", nullable=False)],
        [{"probe_id": 0}],
    )
    operation = {
        "op": "series_reflected_arithmetic_probe",
        "as": "reflected_arithmetic_mismatch",
        "operator": cell.operator,
        "name_mode": cell.name_mode,
        "data_pattern_id": cell.data_pattern_id,
        "lhs_name": lhs_name,
        "rhs_name": rhs_name,
        "lhs_values": list(lhs_values),
        "rhs_values": list(rhs_values),
        "expected_values": expected_values,
    }
    program = Program(
        f"prog-{int(seed):08d}-polars-reflected-{cell.cell_index:02d}",
        int(seed),
        [operation],
    )
    family_witness = {
        "schema_version": "family-witness-case-v1",
        "family_id": REFLECTED_ARITHMETIC_FAMILY_ID,
        "generation_mode": "goal_first_witness_v5",
        "goal_id": REFLECTED_ARITHMETIC_GOAL_ID,
        "root_ids": [REFLECTED_ARITHMETIC_ROOT_ID],
        "cell_index": cell.cell_index,
        "axes": {
            "operator": cell.operator,
            "name_mode": cell.name_mode,
            "data_pattern": cell.data_pattern_id,
        },
        "palette_dimensions": [
            cell.operator,
            cell.name_mode,
            cell.data_pattern_id,
        ],
        "canonical_case_replay": False,
        "runtime_corpus_io": False,
    }
    witness = {
        **cell.to_dict(),
        "goal_id": REFLECTED_ARITHMETIC_GOAL_ID,
        "root_ids": [REFLECTED_ARITHMETIC_ROOT_ID],
        "lhs_name": lhs_name,
        "rhs_name": rhs_name,
        "lhs_values": list(lhs_values),
        "rhs_values": list(rhs_values),
        "expected_values": expected_values,
        "reversed_values": reversed_values,
        "static_noncommutative": expected_values != reversed_values,
        "canonical_case_replay": False,
        "runtime_corpus_io": False,
        "bounded_for_cache_reuse": True,
    }
    trace = {
        "schema_version": REFLECTED_ARITHMETIC_WITNESS_SCHEMA_VERSION,
        "generation_mode": "goal_first_witness_v5",
        "selected_goal": {
            "goal_id": REFLECTED_ARITHMETIC_GOAL_ID,
            "target_contract": (
                "series_reflected_noncommutative_operator_preserves_operand_order"
            ),
            "fault_models": [
                "reflected_operator_operand_order",
                "reflected_power_column_context",
            ],
        },
        "selection_reason": "bounded_operator_name_pattern_cycle_v1",
        "builder_variant": {
            "variant_id": f"{cell.operator}:{cell.name_mode}",
            "variant_family": "v5",
            "operator": cell.operator,
            "name_mode": cell.name_mode,
            "data_pattern": {
                "pattern_id": cell.data_pattern_id,
                "pattern_index": REFLECTED_ARITHMETIC_DATA_PATTERNS.index(
                    cell.data_pattern_id
                ),
                "pattern_count": len(REFLECTED_ARITHMETIC_DATA_PATTERNS),
            },
            "cell_index": cell.cell_index,
            "construction_seed": cell.cell_index,
            "case_seed": int(seed),
            "root_guidance": [REFLECTED_ARITHMETIC_ROOT_ID],
        },
        "requested_profile": str(profile or ""),
        "activation_requirement": {
            "required": True,
            "policy": "deterministic_native_series_contract_witness",
        },
        "reflected_arithmetic_witness": witness,
        "family_witness": family_witness,
        "valid": True,
        "constructible": True,
        "skip_reason": "",
    }
    case = Case(
        case_id=(
            f"case-{int(seed):08d}-polars-reflected-{cell.cell_index:02d}"
        ),
        seed=int(seed),
        tables=[table],
        program=program,
        metadata={
            "generator_profile": "polars_reflected_arithmetic_witness",
            "generation_mode": "goal_first_witness_v5",
            "goal_id": REFLECTED_ARITHMETIC_GOAL_ID,
            "goal_fault_models": [
                "reflected_operator_operand_order",
                "reflected_power_column_context",
            ],
            "semantic_activation_syntactic_reached": True,
            "goal_first_generation": trace,
            "goal_builder_variant": dict(trace["builder_variant"]),
            "semantic_witness_builder_variant": trace["builder_variant"][
                "variant_id"
            ],
            "semantic_witness_data_pattern": dict(
                trace["builder_variant"]["data_pattern"]
            ),
            "reflected_arithmetic_witness": witness,
            "family_witness": family_witness,
        },
    )
    from datadiff.semantic_core.activation import evaluate_semantic_activation

    activation = evaluate_semantic_activation(
        case,
        goal_id=REFLECTED_ARITHMETIC_GOAL_ID,
        syntactic_reached=True,
    ).to_dict()
    case.metadata["semantic_activation"] = activation
    trace["semantic_activation"] = activation
    return ReflectedArithmeticWitnessGenerationResult(case=case, trace=trace)


def _apply_operator(operator: str, left: int, right: int) -> int | float:
    if operator == "rsub":
        return left - right
    if operator == "rtruediv":
        return left / right
    if operator == "rfloordiv":
        return left // right
    if operator == "rmod":
        return left % right
    if operator == "rpow":
        return left**right
    raise ValueError(f"unsupported reflected arithmetic operator: {operator}")
