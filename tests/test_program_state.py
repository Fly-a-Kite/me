from datadiff.datagen import generate_join_table, generate_table
from datadiff.dsl import Case, ColumnSpec, Program, TableData
from datadiff.program_state import state_after_operations, state_before_first_operation, state_before_operation


def test_program_state_tracks_columns_and_types_across_core_ops():
    base = TableData(
        "t0",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("g", "str"),
            ColumnSpec("x", "int"),
            ColumnSpec("y", "float"),
        ],
        [],
    )
    right = TableData(
        "t1",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("tag", "str"),
            ColumnSpec("z", "float"),
        ],
        [],
    )
    operations = [
        {"op": "join", "table": "t1", "left_on": "id", "right_on": "id", "how": "left"},
        {"op": "mutate", "column": "x2", "expr": {"kind": "arith_const", "source": "x", "op": "div", "value": 2}},
        {"op": "coalesce", "columns": ["tag", "g"], "as": "name"},
        {
            "op": "groupby",
            "keys": ["name"],
            "aggs": [{"column": "x2", "func": "sum", "as": "sum_x2"}],
        },
    ]

    state = state_after_operations(base, operations, extra_tables=[right])

    assert state.columns == ["name", "sum_x2"]
    assert state.column_types == {"name": "str", "sum_x2": "float"}


def test_program_state_before_operation_matches_prefix_state():
    case = Case(
        "case-state-prefix",
        1,
        [
            TableData(
                "t0",
                [ColumnSpec("id", "int", nullable=False), ColumnSpec("x", "int"), ColumnSpec("g", "str")],
                [],
            ),
            TableData(
                "t1",
                [ColumnSpec("id", "int", nullable=False), ColumnSpec("tag", "str")],
                [],
            ),
        ],
        Program(
            "prog-state-prefix",
            1,
            [
                {"op": "join", "table": "t1", "left_on": "id", "right_on": "id", "how": "left"},
                {"op": "mutate", "column": "x2", "expr": {"kind": "add_const", "source": "x", "value": 1}},
                {"op": "select", "columns": ["g", "x2", "tag"]},
                {"op": "groupby", "keys": ["g"], "aggs": [{"column": "x2", "func": "sum", "as": "sum_x2"}]},
            ],
        ),
    )

    before_groupby = state_before_first_operation(case, "groupby")
    before_select = state_before_operation(case, 2)

    assert before_select.columns == ["id", "x", "g", "tag", "x2"]
    assert before_groupby.columns == ["g", "x2", "tag"]
    assert before_groupby.column_types["x2"] == "int"
    assert before_groupby.column_types["tag"] == "str"
