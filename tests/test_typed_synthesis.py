from datadiff.classification_oracle import validate_case_program
from datadiff.datagen import generate_case, generate_join_table, generate_table, repair_operations
from datadiff.dsl import ColumnSpec, Program, TableData
from datadiff.program_state import state_after_operations
from datadiff.synthesis.program_synthesizer import default_grammar_registry, synthesize_program
from datadiff.synthesis.typed_state import TypedProgramState


def test_typed_program_state_tracks_forward_schema_changes():
    table = TableData(
        "t0",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("g", "str", nullable=True),
            ColumnSpec("x", "int", nullable=True),
        ],
        [{"id": 1, "g": "a", "x": 2}, {"id": 2, "g": None, "x": None}],
    )

    state = TypedProgramState.from_table(table)
    state = state.after_operation({"op": "mutate", "column": "mx", "expr": {"kind": "add_const", "source": "x", "value": 1}})
    state = state.after_operation({"op": "groupby", "keys": ["g"], "aggs": [{"column": "mx", "func": "sum", "as": "sum_mx"}]})

    assert state.columns == ("g", "sum_mx")
    assert state.column_types["sum_mx"] in {"int", "float"}
    assert state.is_grouped is True


def test_default_grammar_registry_exposes_rule_metadata():
    summary = default_grammar_registry().to_summary()

    assert summary["rule_count"] >= 10
    assert {"filter", "mutate", "groupby", "join"}.issubset(
        {row["op_kind"] for row in summary["rules"]}
    )


def test_synthesize_program_produces_valid_repaired_programs():
    table = generate_table(17, profile="common")
    extra = [generate_join_table(17, profile="common")]

    program = synthesize_program(17, table, max_ops=7, extra_tables=extra)
    repaired = repair_operations(table, [op.to_dict() for op in program.operations], extra_tables=extra)
    state = state_after_operations(table, repaired, extra_tables=extra)

    assert program.program_id.endswith("-typed-grammar")
    assert repaired
    assert state.columns


def test_generate_case_typed_grammar_profile_is_valid():
    case = generate_case(42, profile="typed_grammar")

    assert case.case_id.endswith("-typed-grammar")
    assert case.metadata["synthesis_model"] == "typed_grammar"
    assert case.program.program_id.endswith("-typed-grammar")
    assert case.program.operations
    assert validate_case_program(case) == []
