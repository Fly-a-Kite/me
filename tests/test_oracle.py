from datadiff.datagen import generate_case
from datadiff.dsl import Case, ColumnSpec, Program, TableData
from datadiff.normalizer import NormalizedResult
from datadiff.oracle import evaluate_case


def test_oracle_accept_reject_mismatch():
    case = generate_case(1)
    findings = evaluate_case(
        case,
        {
            "a": NormalizedResult("a", "ok", ["x"], [[1]]),
            "b": NormalizedResult("b", "error", [], [], "ValueError", "bad"),
        },
    )
    assert findings
    assert findings[0].kind == "accept_reject_mismatch"
    assert findings[0].severity == "high"
    assert findings[0].oracle == "differential"
    assert findings[0].root_cause


def test_oracle_semantic_output_mismatch():
    case = generate_case(2)
    findings = evaluate_case(
        case,
        {
            "a": NormalizedResult("a", "ok", ["x"], [[1]]),
            "b": NormalizedResult("b", "ok", ["x"], [[2]]),
        },
    )
    assert findings
    assert findings[0].kind == "semantic_output_mismatch"
    assert findings[0].confidence in {"high", "medium"}
    assert findings[0].mismatch_class == "value"


def test_oracle_labels_row_order_only_mismatch():
    case = Program("prog-order", 1, [{"op": "sort", "columns": ["x"], "ascending": True}])
    findings = evaluate_case(
        Case(
            "case-order",
            1,
            [TableData("t0", [ColumnSpec("x", "int")], [{"x": 1}, {"x": 2}])],
            case,
        ),
        {
            "a": NormalizedResult("a", "ok", ["x"], [[1], [2]]),
            "b": NormalizedResult("b", "ok", ["x"], [[2], [1]]),
        },
    )

    assert findings
    assert findings[0].mismatch_class == "row_order"


def test_oracle_no_mismatch_for_equal_results():
    case = generate_case(3)
    findings = evaluate_case(
        case,
        {
            "a": NormalizedResult("a", "ok", ["x"], [[1]]),
            "b": NormalizedResult("b", "ok", ["x"], [[1]]),
        },
    )
    assert findings == []


def test_oracle_classifies_special_float_before_filter():
    case = Case(
        "case-nan",
        1,
        [
            TableData(
                "t0",
                [ColumnSpec("y", "float")],
                [{"y": float("nan")}, {"y": 1.0}],
            )
        ],
        Program("prog-nan", 1, [{"op": "filter", "column": "y", "cmp": ">", "value": 0.0}]),
    )
    findings = evaluate_case(
        case,
        {
            "a": NormalizedResult("a", "ok", ["y"], [[1]]),
            "b": NormalizedResult("b", "ok", ["y"], [[None], [1]]),
        },
    )
    assert findings
    assert findings[0].root_cause == "nan_inf_semantics"


def test_oracle_classifies_modulo_before_join_or_cast():
    case = Case(
        "case-mod-join-cast",
        11,
        [
            TableData("t0", [ColumnSpec("id", "int"), ColumnSpec("x", "int")], [{"id": 1, "x": -3}]),
            TableData("t1", [ColumnSpec("id", "int"), ColumnSpec("j", "int")], [{"id": 1, "j": 10}]),
        ],
        Program(
            "prog-mod-join-cast",
            11,
            [
                {"op": "join", "table": "t1", "left_on": "id", "right_on": "id", "how": "left"},
                {"op": "mutate", "column": "m", "expr": {"kind": "arith_const", "op": "mod", "source": "x", "value": 2}},
                {"op": "mutate", "column": "xf", "expr": {"kind": "cast", "source": "x", "to": "float"}},
            ],
        ),
    )
    findings = evaluate_case(
        case,
        {
            "a": NormalizedResult("a", "ok", ["id", "j", "m", "x", "xf"], [[1, 10, 1, -3, -3.0]]),
            "b": NormalizedResult("b", "ok", ["id", "j", "m", "x", "xf"], [[1, 10, -1, -3, -3.0]]),
        },
    )

    assert findings
    assert findings[0].root_cause == "arithmetic_expression"


def test_oracle_classifies_groupby_after_join_as_aggregation():
    case = Case(
        "case-join-groupby",
        12,
        [
            TableData("t0", [ColumnSpec("id", "int"), ColumnSpec("x", "int")], [{"id": 1, "x": 3}]),
            TableData("t1", [ColumnSpec("id", "int"), ColumnSpec("j", "int")], [{"id": 1, "j": 10}]),
        ],
        Program(
            "prog-join-groupby",
            12,
            [
                {"op": "join", "table": "t1", "left_on": "id", "right_on": "id", "how": "left"},
                {
                    "op": "groupby",
                    "keys": ["id"],
                    "aggs": [{"column": "x", "func": "count", "as": "count_x"}],
                },
            ],
        ),
    )
    findings = evaluate_case(
        case,
        {
            "a": NormalizedResult("a", "ok", ["count_x", "id"], [[1, 1]]),
            "b": NormalizedResult("b", "ok", ["count_x", "id"], [[2, 1]]),
        },
    )

    assert findings
    assert findings[0].root_cause == "groupby_aggregation"


def test_oracle_classifies_outer_join_truth_filter_before_plain_join():
    case = Case(
        "case-outer-join-truth-filter",
        16,
        [
            TableData("t0", [ColumnSpec("id", "int", nullable=False)], [{"id": 1}, {"id": 2}]),
            TableData("t1", [ColumnSpec("id", "int", nullable=False), ColumnSpec("j", "int")], [{"id": 1, "j": 100}]),
        ],
        Program(
            "prog-outer-join-truth-filter",
            16,
            [
                {"op": "join", "table": "t1", "left_on": "id", "right_on": "id", "how": "left"},
                {"op": "filter", "column": "j", "cmp": "gt_is_not_true", "value": 150},
            ],
        ),
    )

    findings = evaluate_case(
        case,
        {
            "a": NormalizedResult("a", "ok", ["id", "j"], [[1, 100], [2, None]]),
            "b": NormalizedResult("b", "ok", ["id", "j"], [[1, 100]]),
        },
    )

    assert findings
    assert findings[0].root_cause == "outer_join_truth_filter"


def test_oracle_classifies_post_topk_filter_pushdown():
    case = generate_case(370000, profile="post_topk_range_filter")

    findings = evaluate_case(
        case,
        {
            "reference": NormalizedResult("reference", "ok", ["id", "score", "x"], [[2, 90, 1], [4, 80, 2]]),
            "datafusion": NormalizedResult(
                "datafusion",
                "ok",
                ["id", "score", "x"],
                [[2, 90, 1], [4, 80, 2], [6, 70, 0], [7, 65, 2], [8, 60, 1], [9, None, 1]],
            ),
        },
    )

    assert findings
    assert findings[0].root_cause == "topk_filter_pushdown"


def test_oracle_classifies_tuple_absence_null_filter():
    case = generate_case(370017, profile="tuple_absence_filter")

    findings = evaluate_case(
        case,
        {
            "reference": NormalizedResult("reference", "ok", ["a", "b", "payload", "row_id"], [[2, 2, "survivor", 1]]),
            "duckdb": NormalizedResult("duckdb", "ok", ["a", "b", "payload", "row_id"], []),
        },
    )

    assert findings
    assert findings[0].root_cause == "tuple_absence_null_filter"


def test_oracle_classifies_running_sum_precision():
    case = generate_case(370018, profile="running_sum_precision")

    findings = evaluate_case(
        case,
        {
            "reference": NormalizedResult("reference", "ok", ["row_id", "run_x"], [[19999, 10]]),
            "polars": NormalizedResult("polars", "ok", ["row_id", "run_x"], [[19999, 10.0003185272]]),
        },
    )

    assert findings
    assert findings[0].root_cause == "running_sum_precision"


def test_oracle_classifies_sortedness_null_placement():
    case = generate_case(370019, profile="sortedness_null_placement")

    findings = evaluate_case(
        case,
        {
            "reference": NormalizedResult("reference", "ok", ["sorted_ok_x"], [[False]]),
            "polars": NormalizedResult("polars", "ok", ["sorted_ok_x"], [[True]]),
        },
    )

    assert findings
    assert findings[0].root_cause == "sortedness_null_placement"


def test_oracle_classifies_simple_case_random_subject():
    case = generate_case(370020, profile="simple_case_random_subject")

    findings = evaluate_case(
        case,
        {
            "reference": NormalizedResult("reference", "ok", ["unexpected_else_seen"], [[False]]),
            "duckdb": NormalizedResult("duckdb", "ok", ["unexpected_else_seen"], [[True]]),
        },
    )

    assert findings
    assert findings[0].root_cause == "simple_case_random_subject"


def test_oracle_classifies_group_quantile_key_expression():
    case = generate_case(370021, profile="group_quantile_key_probe")

    findings = evaluate_case(
        case,
        {
            "reference": NormalizedResult("reference", "ok", ["quantile_key_mismatch"], [[False]]),
            "polars": NormalizedResult("polars", "ok", ["quantile_key_mismatch"], [[True]]),
        },
    )

    assert findings
    assert findings[0].root_cause == "group_quantile_key_expression"


def test_oracle_classifies_scalar_subquery_double_parentheses():
    case = generate_case(370022, profile="scalar_subquery_double_parentheses")

    findings = evaluate_case(
        case,
        {
            "reference": NormalizedResult("reference", "ok", ["scalar_subquery_mismatch"], [[False]]),
            "duckdb": NormalizedResult("duckdb", "ok", ["scalar_subquery_mismatch"], [[True]]),
        },
    )

    assert findings
    assert findings[0].root_cause == "scalar_subquery_double_parentheses"


def test_oracle_classifies_window_avg_rows_frame():
    case = generate_case(370023, profile="window_avg_rows_frame")

    findings = evaluate_case(
        case,
        {
            "reference": NormalizedResult("reference", "ok", ["window_avg_mismatch"], [[False]]),
            "polars": NormalizedResult("polars", "ok", ["window_avg_mismatch"], [[True]]),
        },
    )

    assert findings
    assert findings[0].root_cause == "window_avg_rows_frame"


def test_oracle_classifies_struct_distinct_unnest():
    case = generate_case(370024, profile="struct_distinct_unnest")

    findings = evaluate_case(
        case,
        {
            "reference": NormalizedResult("reference", "ok", ["struct_distinct_mismatch"], [[False]]),
            "duckdb": NormalizedResult("duckdb", "ok", ["struct_distinct_mismatch"], [[True]]),
        },
    )

    assert findings
    assert findings[0].root_cause == "struct_distinct_unnest"


def test_oracle_classifies_bit_compare_unequal_length():
    case = generate_case(370025, profile="bit_compare_unequal_length")

    findings = evaluate_case(
        case,
        {
            "reference": NormalizedResult("reference", "ok", ["bit_compare_mismatch"], [[False]]),
            "duckdb": NormalizedResult("duckdb", "ok", ["bit_compare_mismatch"], [[True]]),
        },
    )

    assert findings
    assert findings[0].root_cause == "bit_compare_unequal_length"


def test_oracle_classifies_round_even_float_scale():
    case = generate_case(370026, profile="round_even_float_scale")

    findings = evaluate_case(
        case,
        {
            "reference": NormalizedResult("reference", "ok", ["round_even_mismatch"], [[False]]),
            "duckdb": NormalizedResult("duckdb", "ok", ["round_even_mismatch"], [[True]]),
        },
    )

    assert findings
    assert findings[0].root_cause == "round_even_float_scale"


def test_oracle_classifies_series_rtruediv_operand_order():
    case = generate_case(370027, profile="series_rtruediv_operand_order")

    findings = evaluate_case(
        case,
        {
            "reference": NormalizedResult("reference", "ok", ["series_rtruediv_mismatch"], [[False]]),
            "polars": NormalizedResult("polars", "ok", ["series_rtruediv_mismatch"], [[True]]),
        },
    )

    assert findings
    assert findings[0].root_cause == "series_rtruediv_operand_order"


def test_oracle_classifies_grouped_topk_null_sort_key():
    case = Case(
        "case-null-agg-topk",
        13,
        [
            TableData(
                "t0",
                [ColumnSpec("g", "str", nullable=False), ColumnSpec("x", "int")],
                [{"g": "a", "x": None}, {"g": "b", "x": 5}],
            )
        ],
        Program(
            "prog-null-agg-topk",
            13,
            [
                {"op": "groupby", "keys": ["g"], "aggs": [{"column": "x", "func": "min", "as": "min_x"}]},
                {"op": "select", "columns": ["min_x"]},
                {"op": "sort", "columns": ["min_x"], "ascending": True},
                {"op": "limit", "n": 4},
            ],
        ),
    )
    findings = evaluate_case(
        case,
        {
            "a": NormalizedResult("a", "ok", ["min_x"], [[None], [5]]),
            "b": NormalizedResult("b", "ok", ["min_x"], [[5]]),
        },
    )

    assert findings
    assert findings[0].root_cause == "grouped_topk_null_sort_key"


def test_oracle_classifies_grouped_topk_null_sort_key_after_join():
    case = Case(
        "case-join-null-agg-topk",
        15,
        [
            TableData(
                "t0",
                [ColumnSpec("id", "int", nullable=False), ColumnSpec("g", "str"), ColumnSpec("x", "int")],
                [{"id": 1, "g": "a", "x": None}, {"id": 2, "g": "b", "x": 5}],
            ),
            TableData(
                "t1",
                [ColumnSpec("id", "int", nullable=False), ColumnSpec("tag", "str")],
                [{"id": 1, "tag": "left"}, {"id": 2, "tag": "right"}],
            ),
        ],
        Program(
            "prog-join-null-agg-topk",
            15,
            [
                {"op": "join", "table": "t1", "left_on": "id", "right_on": "id", "how": "inner"},
                {"op": "groupby", "keys": ["g"], "aggs": [{"column": "x", "func": "min", "as": "min_x"}]},
                {"op": "select", "columns": ["min_x"]},
                {"op": "sort", "columns": ["min_x"], "ascending": True},
                {"op": "limit", "n": 4},
            ],
        ),
    )
    findings = evaluate_case(
        case,
        {
            "a": NormalizedResult("a", "ok", ["min_x"], [[None], [5]]),
            "b": NormalizedResult("b", "ok", ["min_x"], [[5]]),
        },
    )

    assert findings
    assert findings[0].root_cause == "grouped_topk_null_sort_key"


def test_oracle_classifies_grouped_topk_null_sort_key_from_join_null_key():
    case = Case(
        "case-join-null-key-topk",
        17,
        [
            TableData(
                "t0",
                [ColumnSpec("id", "int", nullable=False), ColumnSpec("x", "int")],
                [{"id": 1, "x": -1}],
            ),
            TableData(
                "t1",
                [ColumnSpec("id", "int", nullable=False), ColumnSpec("j", "int")],
                [{"id": 9, "j": 9}],
            ),
        ],
        Program(
            "prog-join-null-key-topk",
            17,
            [
                {"op": "join", "table": "t1", "left_on": "id", "right_on": "id", "how": "left"},
                {"op": "groupby", "keys": ["j"], "aggs": [{"column": "x", "func": "count", "as": "count_x"}]},
                {"op": "select", "columns": ["j"]},
                {"op": "sort", "columns": ["j"], "ascending": True},
                {"op": "limit", "n": 4},
            ],
        ),
    )

    findings = evaluate_case(
        case,
        {
            "a": NormalizedResult("a", "ok", ["j"], [[None]]),
            "b": NormalizedResult("b", "ok", ["j"], []),
        },
    )

    assert findings
    assert findings[0].root_cause == "grouped_topk_null_sort_key"


def test_oracle_classifies_float_group_key_instability():
    case = Case(
        "case-float-group-key-instability",
        14,
        [
            TableData(
                "t0",
                [ColumnSpec("x", "int")],
                [{"x": 0}, {"x": 0}],
            )
        ],
        Program(
            "prog-float-group-key-instability",
            14,
            [
                {"op": "mutate", "column": "m_0", "expr": {"kind": "add_const", "source": "x", "value": -1}},
                {"op": "filter", "column": "m_0", "cmp": "==", "value": -1},
                {"op": "mutate", "column": "m_1", "expr": {"kind": "arith_const", "op": "mul", "source": "m_0", "value": 10}},
                {"op": "mutate", "column": "m_3", "expr": {"kind": "arith_const", "op": "div", "source": "m_1", "value": 3}},
                {"op": "groupby", "keys": ["m_3"], "aggs": [{"column": "m_0", "func": "min", "as": "min_m_0"}]},
            ],
        ),
    )
    findings = evaluate_case(
        case,
        {
            "a": NormalizedResult("a", "ok", ["m_3", "min_m_0"], [[-3.3333333333, -1]]),
            "b": NormalizedResult(
                "b",
                "ok",
                ["m_3", "min_m_0"],
                [[-3.3333333333, -1], [-3.3333333333, -1]],
            ),
        },
    )

    assert findings
    assert findings[0].root_cause == "float_group_key_instability"
