from datadiff.classification_oracle import classify_finding
from datadiff.dsl import Case, ColumnSpec, Program, TableData
from datadiff.semantic_contracts import (
    finding_contract_boundary_precedes_reference,
    finding_contract_axes,
    finding_contract_axes_for_case,
    finding_matches_contract_boundary,
    semantic_contract_lattice_payload,
)


def test_semantic_contract_lattice_declares_boundary_axes_for_operations():
    case = Case(
        "case-contract",
        1,
        [
            TableData(
                "t0",
                [ColumnSpec("s", "str"), ColumnSpec("x", "float")],
                [{"s": "1", "x": float("nan")}, {"s": None, "x": 2.0}],
            )
        ],
        Program(
            "prog-contract",
            1,
            [
                {"op": "mutate", "column": "sx", "expr": {"kind": "cast", "source": "s", "to": "int"}},
                {"op": "filter", "column": "sx", "cmp": ">=", "value": None},
                {"op": "limit", "n": 1},
            ],
        ),
    )

    payload = semantic_contract_lattice_payload(case)

    assert payload["schema_version"] == "semantic-contract-lattice-v1"
    assert payload["axes"]["dtype_coercion"]["policy"] == "boundary"
    assert payload["axes"]["error_equivalence"]["policy"] == "boundary"
    assert payload["axes"]["null"]["policy"] == "boundary"
    assert payload["axes"]["nan"]["policy"] == "boundary"
    assert payload["axes"]["ordering"]["policy"] == "boundary"
    assert "dtype_coercion" in payload["boundary_axes"]


def test_finding_contract_boundary_only_matches_declared_boundary_axes():
    boundary_case = Case(
        "case-cast-boundary",
        2,
        [TableData("t0", [ColumnSpec("s", "str")], [{"s": "1"}, {"s": "bad"}])],
        Program("prog-cast-boundary", 2, [{"op": "mutate", "column": "i", "expr": {"kind": "cast", "source": "s", "to": "int"}}]),
    )
    strict_case = Case(
        "case-strict",
        3,
        [TableData("t0", [ColumnSpec("x", "int")], [{"x": 1}])],
        Program("prog-strict", 3, [{"op": "select", "columns": ["x"]}]),
    )
    finding = {"root_cause": "type_cast", "mismatch_class": "status"}

    assert finding_contract_axes(finding) == ("dtype_coercion", "error_equivalence")
    assert finding_matches_contract_boundary(boundary_case, finding)
    assert not finding_matches_contract_boundary(strict_case, finding)


def test_finding_contract_boundary_matches_value_or_row_count_on_dynamic_axes():
    running_case = Case(
        "case-running-boundary",
        770187,
        [TableData("t0", [ColumnSpec("id", "int"), ColumnSpec("x", "float")], [{"id": 1, "x": 1.0}])],
        Program(
            "prog-running-boundary",
            770187,
            [
                {
                    "op": "running_sum",
                    "source": "x",
                    "column": "run_x",
                    "order_by": [{"column": "id", "ascending": True, "nulls": "last"}],
                }
            ],
        ),
    )
    grouped_topk_case = Case(
        "case-grouped-topk-boundary",
        772577,
        [TableData("t0", [ColumnSpec("k", "int", nullable=True), ColumnSpec("s", "str")], [{"k": None, "s": "a"}])],
        Program(
            "prog-grouped-topk-boundary",
            772577,
            [
                {"op": "groupby", "keys": ["k"], "aggs": [{"column": "s", "func": "count", "as": "count_s"}]},
                {"op": "sort", "columns": ["k"], "ascending": True},
                {"op": "limit", "n": 1},
            ],
        ),
    )

    assert finding_matches_contract_boundary(
        running_case,
        {"root_cause": "running_sum_precision", "mismatch_class": "row_count"},
    )
    assert finding_matches_contract_boundary(
        running_case,
        {"root_cause": "running_sum_precision", "mismatch_class": "value"},
    )
    assert finding_matches_contract_boundary(
        grouped_topk_case,
        {"root_cause": "grouped_topk_null_sort_key", "mismatch_class": "value"},
    )


def test_case_aware_finding_axes_follow_operation_contract_boundary():
    case = Case(
        "case-conditional-null-boundary",
        760180,
        [
            TableData(
                "t0",
                [ColumnSpec("id", "int", nullable=False), ColumnSpec("x", "int", nullable=True)],
                [{"id": 4, "x": None}],
            ),
            TableData(
                "t1",
                [
                    ColumnSpec("id", "int", nullable=False),
                    ColumnSpec("x", "int", nullable=True),
                    ColumnSpec("flag", "bool", nullable=True),
                ],
                [{"id": 4, "x": None, "flag": True}],
            ),
        ],
        Program(
            "prog-conditional-null-boundary",
            760180,
            [
                {"op": "join", "table": "t1", "left_on": ["id", "x"], "right_on": ["id", "x"], "how": "left"},
                {"op": "mutate", "column": "m_0", "expr": {"kind": "bool_not", "source": "flag"}},
                {"op": "case_when", "as": "cw_0", "condition": {"column": "m_0", "cmp": "!=", "value": False}, "then": True, "else": False},
            ],
        ),
    )
    finding = {"root_cause": "conditional_expression", "mismatch_class": "value"}

    assert finding_contract_axes(finding) == ()
    assert finding_contract_axes_for_case(case, finding) == ("null",)
    assert finding_matches_contract_boundary(case, finding)


def test_special_float_aggregate_contract_boundary_precedes_value_reference():
    case = Case(
        "case-special-float-aggregate-boundary",
        760183,
        [
            TableData(
                "t0",
                [ColumnSpec("g", "str"), ColumnSpec("x", "float")],
                [{"g": "b", "x": float("inf")}, {"g": "b", "x": float("nan")}],
            )
        ],
        Program(
            "prog-special-float-aggregate-boundary",
            760183,
            [
                {
                    "op": "groupby",
                    "keys": ["g"],
                    "aggs": [
                        {"column": "x", "func": "mean", "as": "mean_x"},
                        {"column": "x", "func": "min", "as": "min_x"},
                    ],
                }
            ],
        ),
    )

    finding = {
        "kind": "semantic_output_mismatch",
        "root_cause": "nan_inf_semantics",
        "mismatch_class": "value",
    }

    assert finding_contract_boundary_precedes_reference(case, finding)


def test_contract_boundary_does_not_precede_reference_for_execution_or_mixed_axes():
    cast_case = Case(
        "case-cast-reference-decisive",
        760184,
        [TableData("t0", [ColumnSpec("s", "str")], [{"s": "bad"}])],
        Program(
            "prog-cast-reference-decisive",
            760184,
            [{"op": "mutate", "column": "i", "expr": {"kind": "cast", "source": "s", "to": "int"}}],
        ),
    )
    ordinary_aggregate_case = Case(
        "case-ordinary-aggregate-reference-decisive",
        760185,
        [TableData("t0", [ColumnSpec("g", "str"), ColumnSpec("x", "int")], [{"g": "a", "x": 1}])],
        Program(
            "prog-ordinary-aggregate-reference-decisive",
            760185,
            [{"op": "groupby", "keys": ["g"], "aggs": [{"column": "x", "func": "sum", "as": "sum_x"}]}],
        ),
    )

    assert not finding_contract_boundary_precedes_reference(
        cast_case,
        {
            "kind": "accept_reject_mismatch",
            "root_cause": "type_cast",
            "mismatch_class": "accept_reject",
        },
    )
    assert not finding_contract_boundary_precedes_reference(
        ordinary_aggregate_case,
        {
            "kind": "semantic_output_mismatch",
            "root_cause": "groupby_aggregation",
            "mismatch_class": "value",
        },
    )


def test_classification_uses_semantic_contract_lattice_boundary():
    case = Case(
        "case-contract-classification",
        4,
        [TableData("t0", [ColumnSpec("x", "int")], [{"x": 1}, {"x": 2}])],
        Program("prog-contract-classification", 4, [{"op": "mutate", "column": "xf", "expr": {"kind": "cast", "source": "x", "to": "float"}}]),
    )
    finding = {
        "kind": "semantic_output_mismatch",
        "root_cause": "type_cast",
        "mismatch_class": "status",
        "confidence": "high",
        "suspicious_backends": ["duckdb"],
    }

    classification = classify_finding(case, finding, {}, {}, {"generator_profile": "common"}, ["pandas", "duckdb"])

    assert classification.verdict == "expected_semantic_divergence"
    assert "semantic-contract axis" in classification.evidence
