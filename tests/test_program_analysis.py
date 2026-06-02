from datadiff.dsl import Case, ColumnSpec, Program, TableData
from datadiff.program_analysis import (
    case_uses_arithmetic_float_lineage,
    case_uses_precision_sensitive_float_arithmetic,
)


def _base_case(ops):
    return Case(
        "case-analysis",
        1,
        [
            TableData(
                "t0",
                [ColumnSpec("x", "int"), ColumnSpec("f", "float"), ColumnSpec("g", "str")],
                [{"x": 1, "f": 1.25, "g": "a"}, {"x": 2, "f": 2.5, "g": "b"}],
            )
        ],
        Program("prog-analysis", 1, ops),
    )


def test_case_uses_precision_sensitive_float_arithmetic_detects_sorted_float_lineage():
    case = _base_case(
        [
            {"op": "mutate", "column": "f2", "expr": {"kind": "arith_const", "source": "f", "op": "mul", "value": 3}},
            {"op": "sort", "columns": ["f2"]},
        ]
    )

    assert case_uses_precision_sensitive_float_arithmetic(case) is True


def test_case_uses_precision_sensitive_float_arithmetic_ignores_non_precision_paths():
    case = _base_case(
        [
            {"op": "filter", "column": "x", "cmp": ">", "value": 0},
            {"op": "sort", "columns": ["x"]},
        ]
    )

    assert case_uses_precision_sensitive_float_arithmetic(case) is False


def test_case_uses_arithmetic_float_lineage_detects_groupby_key_derived_from_float_division():
    case = _base_case(
        [
            {"op": "mutate", "column": "bucket", "expr": {"kind": "arith_const", "source": "f", "op": "div", "value": 2}},
            {
                "op": "groupby",
                "keys": ["bucket"],
                "aggs": [{"column": "x", "func": "count", "as": "n"}],
            },
        ]
    )

    assert case_uses_arithmetic_float_lineage(case, ["bucket"]) is True


def test_case_uses_arithmetic_float_lineage_ignores_non_float_groupby_keys():
    case = _base_case(
        [
            {"op": "mutate", "column": "x2", "expr": {"kind": "add_const", "source": "x", "value": 1}},
            {
                "op": "groupby",
                "keys": ["g"],
                "aggs": [{"column": "x2", "func": "sum", "as": "sum_x2"}],
            },
        ]
    )

    assert case_uses_arithmetic_float_lineage(case, ["g"]) is False
