from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from datadiff.dsl import Case, Program, TableData
from datadiff.synthesis.program_synthesizer import synthesize_program
from datadiff.synthesis.typed_state import TypedProgramState

RepairOperationsFn = Callable[[TableData, list[dict[str, Any]], Sequence[TableData]], list[dict[str, Any]]]


def build_typed_grammar_case(
    seed: int,
    table: TableData,
    *,
    extra_tables: Sequence[TableData] | None = None,
    repair_operations_fn: RepairOperationsFn,
    schema_spec: Any | None = None,
    max_ops: int = 7,
) -> Case:
    extras = list(extra_tables or [])
    program = synthesize_program(
        seed,
        table,
        max_ops=max_ops,
        extra_tables=extras,
    )
    repaired = repair_operations_fn(
        table,
        [op.to_dict() for op in program.operations],
        extras,
    )
    if repaired:
        program = Program(program.program_id, program.seed, repaired)
    else:
        program = Program(program.program_id, program.seed, [{"op": "limit", "n": len(table.rows)}])
    typed_state = TypedProgramState.from_table(table, extra_tables=extras)
    table_by_name = {item.name: item for item in [table, *extras]}
    for operation in program.operations:
        typed_state = typed_state.after_operation(operation.to_dict(), tables=table_by_name)
    metadata: dict[str, Any] = {
        "synthesis_model": "typed_grammar",
        "typed_grammar_semantics": {
            "structural_risk_tags": sorted(typed_state.structural_risk_tags),
            "coverage_axes": sorted(typed_state.coverage_axes),
            "expandability_score": float(typed_state.expandability_score),
            "validity_score": float(typed_state.validity_score),
            "row_count_estimate": int(typed_state.row_count_estimate),
            "is_ordered": bool(typed_state.is_ordered),
            "is_grouped": bool(typed_state.is_grouped),
        },
    }
    if schema_spec is not None and hasattr(schema_spec, "to_dict"):
        metadata["lhs_schema_spec"] = schema_spec.to_dict()
    return Case(
        case_id=f"case-{seed:08d}-typed-grammar",
        seed=seed,
        tables=[table, *extras],
        program=program,
        metadata=metadata,
    )
