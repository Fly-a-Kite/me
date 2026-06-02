from datadiff.dsl import Program
from datadiff.operation_semantics import (
    aggregate_functions,
    expr_operator,
    expr_target_type,
    groupby_agg_aliases,
    is_multi_key_join,
    join_left_keys,
    join_right_keys,
    op_branches,
    op_literal,
    op_n,
    op_output_alias,
    op_quantiles,
    op_rows,
    op_values,
    operation_count,
)


def test_operation_semantics_accepts_typed_ir_nodes():
    program = Program(
        "prog-semantics",
        55,
        [
            {
                "op": "join",
                "table": "t_lookup",
                "left_on": ["id", "g"],
                "right_on": ["rid", "rg"],
                "how": "left",
            },
            {
                "op": "mutate",
                "column": "m_0",
                "expr": {"kind": "arith_const", "source": "x", "op": "div", "value": 2},
            },
            {
                "op": "mutate",
                "column": "m_1",
                "expr": {"kind": "cast", "source": "m_0", "to": "int"},
            },
            {
                "op": "groupby",
                "keys": ["g"],
                "aggs": [
                    {"column": "x", "func": "sum", "as": "sum_x"},
                    {"column": "id", "func": "nunique", "as": "n_id"},
                ],
            },
        ],
    )

    join = program.operations[0]
    arith = program.operations[1]
    cast = program.operations[2]
    groupby = program.operations[3]

    assert join_left_keys(join) == ["id", "g"]
    assert join_right_keys(join) == ["rid", "rg"]
    assert is_multi_key_join(join) is True
    assert expr_operator(arith) == "div"
    assert expr_target_type(cast) == "int"
    assert groupby_agg_aliases(groupby) == {"sum_x", "n_id"}
    assert aggregate_functions(groupby) == ["sum", "nunique"]
    assert operation_count(program.operations, "mutate") == 2
    assert operation_count(program.operations, "groupby") == 1


def test_operation_semantics_accepts_typed_probe_and_limit_nodes():
    program = Program(
        "prog-probes",
        77,
        [
            {
                "op": "group_quantile_probe",
                "as": "quantile_ok",
                "values": [1, 2, 3],
                "quantiles": [0.0, 0.5, 1.0],
            },
            {
                "op": "float_literal_precision_probe",
                "as": "float_ok",
                "literal": "0.10000000000000001",
            },
            {"op": "limit", "n": 3},
            {"op": "offset", "n": 2},
        ],
    )

    quantile_probe = program.operations[0]
    float_probe = program.operations[1]
    limit_op = program.operations[2]
    offset_op = program.operations[3]

    assert op_output_alias(quantile_probe) == "quantile_ok"
    assert op_values(quantile_probe) == [1, 2, 3]
    assert op_quantiles(quantile_probe) == [0.0, 0.5, 1.0]
    assert op_output_alias(float_probe) == "float_ok"
    assert op_literal(float_probe) == "0.10000000000000001"
    assert op_n(limit_op) == 3
    assert op_n(offset_op) == 2


def test_operation_semantics_accepts_typed_random_case_probe_fields():
    program = Program(
        "prog-random-probe",
        88,
        [
            {
                "op": "random_case_probe",
                "as": "probe_ok",
                "rows": 4,
                "branches": 3,
            }
        ],
    )

    probe = program.operations[0]

    assert op_output_alias(probe) == "probe_ok"
    assert op_rows(probe) == 4
    assert op_branches(probe) == 3
