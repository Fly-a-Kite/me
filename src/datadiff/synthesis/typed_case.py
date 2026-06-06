from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from datadiff.dsl import Case, Program, TableData
from datadiff.synthesis.program_synthesizer import synthesize_program

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
    metadata: dict[str, Any] = {"synthesis_model": "typed_grammar"}
    if schema_spec is not None and hasattr(schema_spec, "to_dict"):
        metadata["lhs_schema_spec"] = schema_spec.to_dict()
    return Case(
        case_id=f"case-{seed:08d}-typed-grammar",
        seed=seed,
        tables=[table, *extras],
        program=program,
        metadata=metadata,
    )
