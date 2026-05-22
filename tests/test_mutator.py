import random

from datadiff.dsl import Case, ColumnSpec, Program, TableData
from datadiff.mutator import (
    MUTATION_OPERATOR_NAMES,
    _available_columns,
    _random_operation,
    mutate_case,
    mutate_case_with_metadata,
)


def test_random_operation_can_mutate_string_only_available_columns():
    table = TableData(
        "t0",
        [ColumnSpec("s", "str")],
        [{"s": "Alpha"}, {"s": "中文"}],
    )

    for seed in range(200):
        op = _random_operation([table], [], random.Random(seed))
        if op is None or op["op"] != "mutate":
            continue
        assert op["expr"]["kind"] in {"string_length", "string_lower"}
        assert op["expr"]["source"] == "s"


def test_available_columns_are_deduplicated_after_overwrite_and_select():
    table = TableData(
        "t0",
        [ColumnSpec("x", "int"), ColumnSpec("g", "str")],
        [{"x": 1, "g": "Alpha"}],
    )

    available = _available_columns(
        [table],
        [
            {"op": "mutate", "column": "m_1", "expr": {"kind": "add_const", "source": "x", "value": 1}},
            {"op": "mutate", "column": "m_1", "expr": {"kind": "add_const", "source": "x", "value": 2}},
            {"op": "select", "columns": ["g", "m_1", "m_1"]},
        ],
    )

    assert available == ["g", "m_1"]


def test_mutate_case_with_metadata_records_lineage_and_operator():
    base = Case(
        "case-base",
        10,
        [TableData("t0", [ColumnSpec("x", "int")], [{"x": 1}, {"x": None}])],
        Program("prog-base", 10, [{"op": "limit", "n": 2}]),
        metadata={"seed_lineage": {"root_seed": 3, "depth": 2}},
    )

    result = mutate_case_with_metadata(base, 99)

    assert result.case.case_id == "case-base-mut-99"
    assert result.case.metadata == result.metadata
    assert result.metadata["seed_lineage"] == {
        "root_seed": 3,
        "parent_seed": 10,
        "parent_case_id": "case-base",
        "mutation_seed": 99,
        "depth": 3,
    }
    assert result.metadata["mutation"]["operator"] in MUTATION_OPERATOR_NAMES
    assert isinstance(result.metadata["mutation"]["detail"], str)
    assert isinstance(result.metadata["mutation"]["changed"], bool)
    assert isinstance(mutate_case(base, 100), Case)


def test_mutation_operator_registry_covers_row_value_and_operation_mutations():
    assert {"value", "nullify_value", "duplicate_row", "drop_row", "shuffle_rows"}.issubset(MUTATION_OPERATOR_NAMES)
    assert {"append_op", "drop_op", "tweak_op"}.issubset(MUTATION_OPERATOR_NAMES)
