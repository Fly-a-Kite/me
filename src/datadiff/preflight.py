from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from datadiff.case_validation import validate_case_program
from datadiff.datagen import repair_operations
from datadiff.dsl import Case, ColumnSpec, Program, TableData


@dataclass(slots=True)
class PreflightResult:
    case: Case
    valid: bool
    repaired: bool
    fallback_used: bool
    errors_before: list[str]
    errors_after: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "repaired": self.repaired,
            "fallback_used": self.fallback_used,
            "errors_before": list(self.errors_before),
            "errors_after": list(self.errors_after),
        }


def preflight_case(
    case: Case,
    *,
    enable_validation: bool = True,
    enable_repair: bool = True,
) -> PreflightResult:
    if not enable_validation and not enable_repair:
        return PreflightResult(case, True, False, False, [], [])

    errors_before = validate_case_program(case) if enable_validation else []
    if not errors_before:
        return PreflightResult(case, True, False, False, errors_before, [])

    if not enable_repair:
        return PreflightResult(case, False, False, False, errors_before, errors_before)

    repaired, fallback_used = _repair_case(case)
    errors_after = validate_case_program(repaired) if enable_validation else []
    if errors_after:
        repaired = _fallback_case(case)
        errors_after = validate_case_program(repaired) if enable_validation else []
        fallback_used = True
    return PreflightResult(
        repaired,
        valid=not errors_after,
        repaired=True,
        fallback_used=fallback_used,
        errors_before=errors_before,
        errors_after=errors_after,
    )


def _repair_case(case: Case) -> tuple[Case, bool]:
    if not case.tables:
        return _fallback_case(case), True
    operations = repair_operations(
        case.tables[0],
        copy.deepcopy(case.program.operations),
        extra_tables=case.tables[1:],
    )
    fallback_used = False
    if not operations:
        operations = _fallback_operations(case)
        fallback_used = True
    return Case(
        case_id=case.case_id,
        seed=case.seed,
        tables=case.tables,
        program=Program(
            program_id=case.program.program_id,
            seed=case.program.seed,
            operations=operations,
        ),
        metadata=dict(case.metadata),
    ), fallback_used


def _fallback_case(case: Case) -> Case:
    tables = _dedupe_case_tables(case.tables)
    return Case(
        case_id=case.case_id,
        seed=case.seed,
        tables=tables,
        program=Program(
            program_id=case.program.program_id,
            seed=case.program.seed,
            operations=_fallback_operations_for_tables(tables),
        ),
        metadata=dict(case.metadata),
    )


def _fallback_operations(case: Case) -> list[dict[str, Any]]:
    return [{"op": "limit", "n": len(case.tables[0].rows) if case.tables else 0}]


def _fallback_operations_for_tables(tables: list[TableData]) -> list[dict[str, Any]]:
    return [{"op": "limit", "n": len(tables[0].rows) if tables else 0}]


def _dedupe_case_tables(tables: list[TableData]) -> list[TableData]:
    result: list[TableData] = []
    used_table_names: set[str] = set()
    for index, table in enumerate(tables):
        table_name = table.name
        if table_name in used_table_names:
            table_name = f"{table.name}_{index}"
        used_table_names.add(table_name)
        seen_columns: set[str] = set()
        columns: list[ColumnSpec] = []
        for column in table.columns:
            if column.name in seen_columns:
                continue
            seen_columns.add(column.name)
            columns.append(ColumnSpec(column.name, column.type, nullable=column.nullable))
        result.append(TableData(table_name, columns, list(table.rows)))
    return result
