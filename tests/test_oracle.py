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


def test_oracle_classifies_pandas_uint64_isin_precision():
    case = generate_case(370028, profile="pandas_uint64_isin_precision")

    findings = evaluate_case(
        case,
        {
            "reference": NormalizedResult("reference", "ok", ["uint64_isin_mismatch"], [[False]]),
            "pandas": NormalizedResult("pandas", "ok", ["uint64_isin_mismatch"], [[True]]),
        },
    )

    assert findings
    assert findings[0].root_cause == "pandas_uint64_isin_precision"


def test_oracle_classifies_duckdb_tuple_anti_null_semantics():
    case = generate_case(370029, profile="duckdb_tuple_anti_null_semantics")

    findings = evaluate_case(
        case,
        {
            "reference": NormalizedResult("reference", "ok", ["tuple_anti_null_mismatch"], [[False]]),
            "duckdb": NormalizedResult("duckdb", "ok", ["tuple_anti_null_mismatch"], [[True]]),
        },
    )

    assert findings
    assert findings[0].root_cause == "duckdb_tuple_anti_null_semantics"


def test_oracle_classifies_duckdb_json_predicate_order_semantics():
    case = generate_case(370040, profile="duckdb_json_predicate_order_semantics")

    findings = evaluate_case(
        case,
        {
            "reference": NormalizedResult("reference", "ok", ["json_predicate_order_mismatch"], [[False]]),
            "duckdb": NormalizedResult("duckdb", "ok", ["json_predicate_order_mismatch"], [[True]]),
        },
    )

    assert findings
    assert findings[0].root_cause == "duckdb_json_predicate_order_semantics"


def test_oracle_classifies_pandas_sparse_array_mask_semantics():
    case = generate_case(370030, profile="pandas_sparse_array_mask_semantics")

    findings = evaluate_case(
        case,
        {
            "reference": NormalizedResult("reference", "ok", ["sparse_mask_mismatch"], [[False]]),
            "pandas": NormalizedResult("pandas", "ok", ["sparse_mask_mismatch"], [[True]]),
        },
    )

    assert findings
    assert findings[0].root_cause == "pandas_sparse_array_mask_semantics"


def test_oracle_classifies_polars_float_wrap_numerical_semantics():
    case = generate_case(370031, profile="polars_float_wrap_numerical_semantics")

    findings = evaluate_case(
        case,
        {
            "reference": NormalizedResult("reference", "ok", ["float_wrap_mismatch"], [[False]]),
            "polars": NormalizedResult("polars", "ok", ["float_wrap_mismatch"], [[True]]),
            "polars_lazy": NormalizedResult("polars_lazy", "ok", ["float_wrap_mismatch"], [[True]]),
        },
    )

    assert findings
    assert findings[0].root_cause == "polars_float_wrap_numerical_semantics"


def test_oracle_classifies_pandas_index_bool_result_type():
    case = generate_case(370032, profile="pandas_index_bool_result_type")

    findings = evaluate_case(
        case,
        {
            "reference": NormalizedResult("reference", "ok", ["index_bool_mismatch"], [[False]]),
            "pandas": NormalizedResult("pandas", "ok", ["index_bool_mismatch"], [[True]]),
        },
    )

    assert findings
    assert findings[0].root_cause == "pandas_index_bool_result_type"


def test_oracle_classifies_polars_empty_literal_groupby_semantics():
    case = generate_case(370033, profile="polars_empty_literal_groupby_semantics")

    findings = evaluate_case(
        case,
        {
            "reference": NormalizedResult("reference", "ok", ["empty_literal_groupby_mismatch"], [[False]]),
            "polars": NormalizedResult("polars", "ok", ["empty_literal_groupby_mismatch"], [[True]]),
            "polars_lazy": NormalizedResult("polars_lazy", "ok", ["empty_literal_groupby_mismatch"], [[True]]),
        },
    )

    assert findings
    assert findings[0].root_cause == "polars_empty_literal_groupby_semantics"


def test_oracle_classifies_pandas_arrow_string_eq_sum_semantics():
    case = generate_case(370034, profile="pandas_arrow_string_eq_sum_semantics")

    findings = evaluate_case(
        case,
        {
            "reference": NormalizedResult("reference", "ok", ["arrow_string_eq_sum_mismatch"], [[False]]),
            "pandas": NormalizedResult("pandas", "ok", ["arrow_string_eq_sum_mismatch"], [[True]]),
        },
    )

    assert findings
    assert findings[0].root_cause == "pandas_arrow_string_eq_sum_semantics"


def test_oracle_classifies_pandas_arrow_timestamp_loc_slice_semantics():
    case = generate_case(370035, profile="pandas_arrow_timestamp_loc_slice_semantics")

    findings = evaluate_case(
        case,
        {
            "reference": NormalizedResult("reference", "ok", ["arrow_timestamp_loc_slice_mismatch"], [[False]]),
            "pandas": NormalizedResult("pandas", "ok", ["arrow_timestamp_loc_slice_mismatch"], [[True]]),
        },
    )

    assert findings
    assert findings[0].root_cause == "pandas_arrow_timestamp_loc_slice_semantics"


def test_oracle_classifies_pandas_arrow_timestamp_index_attr_semantics():
    case = generate_case(370036, profile="pandas_arrow_timestamp_index_attr_semantics")

    findings = evaluate_case(
        case,
        {
            "reference": NormalizedResult("reference", "ok", ["arrow_timestamp_index_attr_mismatch"], [[False]]),
            "pandas": NormalizedResult("pandas", "ok", ["arrow_timestamp_index_attr_mismatch"], [[True]]),
        },
    )

    assert findings
    assert findings[0].root_cause == "pandas_arrow_timestamp_index_attr_semantics"


def test_oracle_classifies_pandas_eval_inplace_aliasing_semantics():
    case = generate_case(370041, profile="pandas_eval_inplace_aliasing_semantics")

    findings = evaluate_case(
        case,
        {
            "reference": NormalizedResult("reference", "ok", ["eval_inplace_alias_mismatch"], [[False]]),
            "pandas": NormalizedResult("pandas", "ok", ["eval_inplace_alias_mismatch"], [[True]]),
        },
    )

    assert findings
    assert findings[0].root_cause == "pandas_eval_inplace_aliasing_semantics"


def test_oracle_classifies_pyarrow_dataset_isin_all_match_semantics():
    case = generate_case(370037, profile="pyarrow_dataset_isin_all_match_semantics")

    findings = evaluate_case(
        case,
        {
            "reference": NormalizedResult("reference", "ok", ["dataset_isin_all_match_mismatch"], [[False]]),
            "pyarrow": NormalizedResult("pyarrow", "ok", ["dataset_isin_all_match_mismatch"], [[True]]),
        },
    )

    assert findings
    assert findings[0].root_cause == "pyarrow_dataset_isin_all_match_semantics"


def test_oracle_classifies_last_output_probe_when_mutation_appends_multiple_probes():
    case = Case(
        "case-multi-probe-root",
        17,
        [TableData("t0", [ColumnSpec("probe_id", "int")], [{"probe_id": 0}])],
        Program(
            "prog-multi-probe-root",
            17,
            [
                {"op": "group_quantile_probe", "as": "quantile_key_mismatch"},
                {"op": "dataset_isin_all_match_probe", "as": "dataset_isin_all_match_mismatch"},
            ],
        ),
    )

    findings = evaluate_case(
        case,
        {
            "reference": NormalizedResult("reference", "ok", ["dataset_isin_all_match_mismatch"], [[False]]),
            "pyarrow": NormalizedResult("pyarrow", "ok", ["dataset_isin_all_match_mismatch"], [[True]]),
            "polars": NormalizedResult("polars", "ok", ["dataset_isin_all_match_mismatch"], [[False]]),
        },
    )

    assert findings
    assert findings[0].root_cause == "pyarrow_dataset_isin_all_match_semantics"


def test_oracle_classifies_pyarrow_large_string_partition_schema_semantics():
    case = generate_case(370039, profile="pyarrow_large_string_partition_schema_semantics")

    findings = evaluate_case(
        case,
        {
            "reference": NormalizedResult("reference", "ok", ["large_string_partition_mismatch"], [[False]]),
            "pyarrow": NormalizedResult("pyarrow", "ok", ["large_string_partition_mismatch"], [[True]]),
        },
    )

    assert findings
    assert findings[0].root_cause == "pyarrow_large_string_partition_schema_semantics"


def test_oracle_classifies_pyarrow_hash_pivot_wider_order_semantics():
    case = generate_case(370040, profile="pyarrow_hash_pivot_wider_order_semantics")

    findings = evaluate_case(
        case,
        {
            "reference": NormalizedResult("reference", "ok", ["hash_pivot_wider_mismatch"], [[False]]),
            "pyarrow": NormalizedResult("pyarrow", "ok", ["hash_pivot_wider_mismatch"], [[True]]),
        },
    )

    assert findings
    assert findings[0].root_cause == "pyarrow_hash_pivot_wider_order_semantics"


def test_oracle_classifies_polars_rolling_mean_by_null_count_semantics():
    case = generate_case(370038, profile="polars_rolling_mean_by_null_count_semantics")

    findings = evaluate_case(
        case,
        {
            "reference": NormalizedResult("reference", "ok", ["rolling_mean_by_null_count_mismatch"], [[False]]),
            "polars": NormalizedResult("polars", "ok", ["rolling_mean_by_null_count_mismatch"], [[True]]),
            "polars_lazy": NormalizedResult("polars_lazy", "ok", ["rolling_mean_by_null_count_mismatch"], [[True]]),
        },
    )

    assert findings
    assert findings[0].root_cause == "polars_rolling_mean_by_null_count_semantics"


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
