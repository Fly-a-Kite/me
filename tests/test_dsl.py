from datadiff.dsl import (
    AggregateOp,
    AggregateSpec,
    ArithConstExpr,
    BooleanPredicateCondition,
    CaseWhenOp,
    Expression,
    FilterOp,
    GroupByOp,
    MutateOp,
    Operation,
    Program,
    RowNumberFilterOp,
    RunningSumOp,
    SortKeySpec,
    SortOp,
    StringLowerExpr,
)
from datadiff.operation_semantics import op_kind


def test_program_coerces_operations_into_typed_ir_nodes():
    program = Program(
        "prog-typed",
        7,
        [
            {
                "op": "mutate",
                "column": "m_0",
                "expr": {"kind": "string_lower", "source": "s"},
            },
            {
                "op": "case_when",
                "as": "bucket",
                "condition": {"column": "flag", "cmp": "bool_is_true", "value": None},
                "then": "hit",
                "else": "miss",
            },
            {
                "op": "groupby",
                "keys": ["g"],
                "aggs": [{"column": "x", "func": "sum", "as": "sum_x"}],
            },
            {
                "op": "sort",
                "keys": [{"column": "g", "ascending": True, "nulls": "last"}],
            },
        ],
    )

    assert isinstance(program.operations[0], Operation)
    assert isinstance(program.operations[0], MutateOp)
    assert isinstance(program.operations[0]["expr"], Expression)
    assert isinstance(program.operations[0]["expr"], StringLowerExpr)
    assert isinstance(program.operations[1], CaseWhenOp)
    assert isinstance(program.operations[1]["condition"], BooleanPredicateCondition)
    assert isinstance(program.operations[2], GroupByOp)
    assert isinstance(program.operations[2]["aggs"][0], AggregateSpec)
    assert isinstance(program.operations[3], SortOp)
    assert isinstance(program.operations[3]["keys"][0], SortKeySpec)


def test_typed_ir_nodes_preserve_legacy_mapping_behavior():
    op = Operation({"op": "filter", "column": "x", "cmp": ">", "value": 1})

    assert op == {"op": "filter", "column": "x", "cmp": ">", "value": 1}
    assert dict(op) == {"op": "filter", "column": "x", "cmp": ">", "value": 1}
    assert {**op, "value": 2} == {"op": "filter", "column": "x", "cmp": ">", "value": 2}


def test_program_to_dict_preserves_legacy_json_shape():
    payload = {
        "program_id": "prog-roundtrip",
        "seed": 11,
        "operations": [
            {"op": "mutate", "column": "m_0", "expr": {"kind": "add_const", "source": "x", "value": 1}},
            {"op": "groupby", "keys": ["g"], "aggs": [{"column": "m_0", "func": "max", "as": "max_m_0"}]},
        ],
    }

    program = Program.from_dict(payload)

    assert program.to_dict() == payload


def test_program_coerces_filter_and_aggregate_into_specific_operation_nodes():
    program = Program(
        "prog-specific",
        13,
        [
            {"op": "filter", "column": "x", "cmp": ">", "value": 1},
            {"op": "aggregate", "aggs": [{"column": "x", "func": "sum", "as": "sum_x"}]},
        ],
    )

    assert isinstance(program.operations[0], FilterOp)
    assert isinstance(program.operations[1], AggregateOp)


def test_program_accepts_legacy_kind_operation_field():
    program = Program(
        "prog-kind-compat",
        17,
        [
            {"kind": "filter", "column": "x", "cmp": ">", "value": 1},
            {"kind": "sort", "keys": [{"column": "x", "ascending": True, "nulls": "last"}]},
        ],
    )

    assert isinstance(program.operations[0], FilterOp)
    assert isinstance(program.operations[1], SortOp)
    assert op_kind(program.operations[0]) == "filter"
    assert program.operations[0]["op"] == "filter"
    assert program.operations[0]["kind"] == "filter"


def test_typed_ir_exposes_stable_properties_for_common_nodes():
    program = Program(
        "prog-properties",
        21,
        [
            {"op": "mutate", "column": "m_0", "expr": {"kind": "arith_const", "source": "x", "op": "div", "value": 2}},
            {"op": "filter", "column": "m_0", "cmp": ">=", "value": 1},
            {"op": "groupby", "keys": ["g"], "aggs": [{"column": "m_0", "func": "sum", "as": "sum_m_0"}]},
            {"op": "sort", "keys": [{"column": "sum_m_0", "ascending": False, "nulls": "last"}]},
        ],
    )

    mutate = program.operations[0]
    filter_op = program.operations[1]
    groupby = program.operations[2]
    sort = program.operations[3]

    assert isinstance(mutate, MutateOp)
    assert mutate.column == "m_0"
    assert isinstance(mutate.expression, ArithConstExpr)
    assert mutate.expression.source == "x"
    assert mutate.expression.operator == "div"
    assert isinstance(filter_op, FilterOp)
    assert filter_op.column == "m_0"
    assert filter_op.comparator == ">="
    assert isinstance(groupby, GroupByOp)
    assert groupby.group_keys == ["g"]
    assert groupby.aggregates[0].alias == "sum_m_0"
    assert isinstance(sort, SortOp)
    assert sort.sort_keys[0].column == "sum_m_0"


def test_typed_ir_rejects_properties_that_shadow_mapping_methods():
    namespace = {
        "__module__": "tests.test_dsl",
        "keys": property(lambda self: []),
    }

    try:
        type("BadNode", (Operation,), namespace)
    except TypeError as exc:
        assert "reserved IR property names" in str(exc)
        assert "keys" in str(exc)
    else:
        raise AssertionError("expected reserved property name conflict")


def test_typed_ir_exposes_join_window_and_groupby_shape_helpers():
    program = Program(
        "prog-helpers",
        34,
        [
            {"op": "join", "table": "t1", "left_on": ["id", "g"], "right_on": ["rid", "rg"], "how": "left"},
            {
                "op": "row_number_filter",
                "partition_by": ["g"],
                "order_by": [{"column": "id", "ascending": False, "nulls": "first"}],
                "cmp": "==",
                "value": 1,
            },
            {
                "op": "running_sum",
                "source": "x",
                "column": "run_x",
                "partition_by": ["g"],
                "order_by": [{"column": "id", "ascending": True, "nulls": "last"}],
            },
            {
                "op": "groupby",
                "keys": ["g"],
                "aggs": [{"column": "x", "func": "sum", "as": "sum_x"}],
            },
        ],
    )

    join = program.operations[0]
    row_number = program.operations[1]
    running_sum = program.operations[2]
    groupby = program.operations[3]

    assert join.left_keys == ["id", "g"]
    assert join.right_keys == ["rid", "rg"]
    assert join.table == "t1"
    assert isinstance(row_number, RowNumberFilterOp)
    assert row_number.partition_columns == ["g"]
    assert row_number.order_keys[0].column == "id"
    assert row_number.order_keys[0].ascending is False
    assert row_number.order_keys[0].nulls == "first"
    assert isinstance(running_sum, RunningSumOp)
    assert running_sum.source == "x"
    assert running_sum.column == "run_x"
    assert running_sum.partition_columns == ["g"]
    assert running_sum.order_keys[0].column == "id"
    assert groupby.aggregate_aliases == ["sum_x"]
