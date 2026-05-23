from datadiff.classification_oracle import classify_finding, validate_case_program
from datadiff.dsl import Case, ColumnSpec, Program, TableData
from datadiff.normalizer import NormalizedResult


def _case(ops):
    return Case(
        "case-x",
        1,
        [TableData("t0", [ColumnSpec("x", "int"), ColumnSpec("s", "str")], [{"x": 1, "s": "Alpha"}])],
        Program("prog-x", 1, ops),
    )


def test_classification_marks_invalid_generated_program_false_positive():
    case = _case([{"op": "select", "columns": ["missing"]}])
    finding = {
        "kind": "accept_reject_mismatch",
        "root_cause": "schema_projection",
        "confidence": "high",
        "suspicious_backends": ["pandas"],
    }

    classification = classify_finding(case, finding, {}, {}, {"generator_profile": "common"}, ["pandas", "duckdb"])

    assert classification.verdict == "generator_false_positive"
    assert classification.false_positive is True
    assert classification.false_positive_reason == "invalid_generated_program"
    assert validate_case_program(case)


def test_validate_case_program_accepts_and_rejects_per_column_sort_keys():
    valid = _case(
        [
            {
                "op": "sort",
                "keys": [
                    {"column": "x", "ascending": False, "nulls": "first"},
                    {"column": "s", "ascending": True, "nulls": "last"},
                ],
            }
        ]
    )
    invalid = _case([{"op": "sort", "keys": [{"column": "x", "ascending": True, "nulls": "middle"}]}])

    assert validate_case_program(valid) == []
    assert "invalid sort keys" in validate_case_program(invalid)[0]


def test_validate_case_program_accepts_and_rejects_offset():
    valid = _case([{"op": "sort", "columns": ["x"], "ascending": True}, {"op": "offset", "n": 1}])
    negative = _case([{"op": "offset", "n": -1}])
    invalid = _case([{"op": "offset", "n": "one"}])

    assert validate_case_program(valid) == []
    assert "negative offset" in validate_case_program(negative)[0]
    assert "non-integer offset" in validate_case_program(invalid)[0]


def test_classification_marks_normalizer_error_false_positive():
    case = _case([{"op": "filter", "column": "x", "cmp": ">", "value": 0}])
    finding = {
        "kind": "semantic_output_mismatch",
        "root_cause": "filter_predicate",
        "confidence": "high",
        "suspicious_backends": ["duckdb"],
    }
    normalized = {
        "pandas": NormalizedResult("pandas", "ok", ["x"], [[1]]),
        "duckdb": NormalizedResult("duckdb", "normalization_error", [], [], "TypeError", "bad"),
    }

    classification = classify_finding(case, finding, normalized, {}, {"generator_profile": "common"}, ["pandas", "duckdb"])

    assert classification.verdict == "normalizer_false_positive"
    assert classification.false_positive is True
    assert classification.false_positive_reason == "normalization_error"


def test_classification_marks_order_insensitive_order_only_mismatch_false_positive():
    case = _case([])
    finding = {
        "kind": "semantic_output_mismatch",
        "root_cause": "unknown",
        "confidence": "medium",
        "suspicious_backends": ["pandas", "duckdb"],
    }
    normalized = {
        "pandas": NormalizedResult("pandas", "ok", ["x"], [[1], [2]]),
        "duckdb": NormalizedResult("duckdb", "ok", ["x"], [[2], [1]]),
    }

    classification = classify_finding(case, finding, normalized, {}, {"generator_profile": "common"}, ["pandas", "duckdb"])

    assert classification.verdict == "normalizer_false_positive"
    assert classification.false_positive is True
    assert classification.false_positive_reason == "order_only_normalization_mismatch"


def test_classification_uses_ordered_reference_for_order_sensitive_mismatch():
    case = Case(
        "case-order-sensitive",
        41,
        [
            TableData(
                "t0",
                [ColumnSpec("x", "int"), ColumnSpec("s", "str")],
                [{"x": 1, "s": "a"}, {"x": 2, "s": "b"}],
            )
        ],
        Program(
            "prog-order-sensitive",
            41,
            [{"op": "sort", "columns": ["x"], "ascending": False}, {"op": "select", "columns": ["s"]}],
        ),
    )
    finding = {
        "kind": "semantic_output_mismatch",
        "root_cause": "ordering_or_limit",
        "confidence": "medium",
        "suspicious_backends": ["duckdb"],
    }
    normalized = {
        "pandas": NormalizedResult("pandas", "ok", ["s"], [["b"], ["a"]]),
        "duckdb": NormalizedResult("duckdb", "ok", ["s"], [["a"], ["b"]]),
    }

    classification = classify_finding(case, finding, normalized, {}, {"generator_profile": "common"}, ["pandas", "duckdb"])

    assert classification.verdict == "candidate_implementation_bug"
    assert classification.false_positive is False
    assert classification.false_positive_reason == ""
    assert classification.implicated_backends == ["duckdb"]


def test_classification_marks_nan_semantics_as_documented_divergence():
    case = Case(
        "case-nan",
        2,
        [TableData("t0", [ColumnSpec("y", "float")], [{"y": float("nan")}])],
        Program("prog-nan", 2, [{"op": "filter", "column": "y", "cmp": ">", "value": 0.0}]),
    )
    finding = {
        "kind": "semantic_output_mismatch",
        "root_cause": "nan_inf_semantics",
        "confidence": "high",
        "suspicious_backends": ["polars"],
    }

    classification = classify_finding(case, finding, {}, {}, {"generator_profile": "edge_float"}, ["pandas", "polars"])

    assert classification.verdict == "documented_semantic_divergence"
    assert classification.false_positive is False
    assert classification.documentation_refs


def test_classification_does_not_call_non_polars_nan_divergence_documented():
    case = Case(
        "case-nan-duckdb",
        22,
        [TableData("t0", [ColumnSpec("y", "float")], [{"y": float("nan")}])],
        Program("prog-nan-duckdb", 22, [{"op": "filter", "column": "y", "cmp": ">", "value": 0.0}]),
    )
    finding = {
        "kind": "semantic_output_mismatch",
        "root_cause": "nan_inf_semantics",
        "confidence": "high",
        "suspicious_backends": ["duckdb"],
    }

    classification = classify_finding(case, finding, {}, {}, {"generator_profile": "edge_float"}, ["pandas", "duckdb"])

    assert classification.verdict == "expected_semantic_divergence"
    assert classification.documentation_refs == []


def test_classification_does_not_call_multi_backend_nan_divergence_documented():
    case = Case(
        "case-nan-all",
        23,
        [TableData("t0", [ColumnSpec("y", "float")], [{"y": float("nan")}])],
        Program("prog-nan-all", 23, [{"op": "filter", "column": "y", "cmp": ">", "value": 0.0}]),
    )
    finding = {
        "kind": "semantic_output_mismatch",
        "root_cause": "nan_inf_semantics",
        "confidence": "medium",
        "suspicious_backends": ["duckdb", "pandas", "polars", "sqlite"],
    }

    classification = classify_finding(
        case,
        finding,
        {},
        {},
        {"generator_profile": "edge_float"},
        ["pandas", "polars", "duckdb", "sqlite"],
    )

    assert classification.verdict == "expected_semantic_divergence"
    assert classification.documentation_refs == []


def test_classification_marks_clear_minority_as_candidate_bug():
    case = _case([{"op": "filter", "column": "x", "cmp": ">", "value": 0}])
    finding = {
        "kind": "semantic_output_mismatch",
        "root_cause": "filter_predicate",
        "confidence": "high",
        "suspicious_backends": ["duckdb"],
    }

    classification = classify_finding(case, finding, {}, {}, {"generator_profile": "common"}, ["pandas", "polars", "duckdb"])

    assert classification.verdict == "candidate_implementation_bug"
    assert classification.false_positive is False


def test_classification_does_not_treat_edge_float_profile_as_boundary_by_itself():
    case = _case([{"op": "filter", "column": "x", "cmp": ">", "value": 0}])
    finding = {
        "kind": "semantic_output_mismatch",
        "root_cause": "filter_predicate",
        "confidence": "high",
        "suspicious_backends": ["duckdb"],
    }

    classification = classify_finding(case, finding, {}, {}, {"generator_profile": "edge_float"}, ["pandas", "duckdb"])

    assert classification.verdict == "candidate_implementation_bug"


def test_classification_marks_unicode_lower_as_semantic_boundary():
    case = Case(
        "case-lower",
        3,
        [TableData("t0", [ColumnSpec("s", "str")], [{"s": "Ä"}])],
        Program("prog-lower", 3, [{"op": "mutate", "column": "m_0", "expr": {"kind": "string_lower", "source": "s"}}]),
    )
    finding = {
        "kind": "semantic_output_mismatch",
        "root_cause": "string_expression",
        "confidence": "high",
        "suspicious_backends": ["sqlite"],
    }

    classification = classify_finding(case, finding, {}, {}, {"generator_profile": "common"}, ["pandas", "sqlite"])

    assert classification.verdict == "expected_semantic_divergence"


def test_classification_uses_dsl_reference_to_identify_mismatching_backend():
    case = Case(
        "case-reference",
        31,
        [TableData("t0", [ColumnSpec("x", "int")], [{"x": 1}, {"x": 2}])],
        Program("prog-reference", 31, [{"op": "filter", "column": "x", "cmp": ">", "value": 1}]),
    )
    finding = {
        "kind": "semantic_output_mismatch",
        "root_cause": "filter_predicate",
        "confidence": "medium",
        "suspicious_backends": ["pandas", "duckdb"],
    }
    normalized = {
        "pandas": NormalizedResult("pandas", "ok", ["x"], [[2]]),
        "duckdb": NormalizedResult("duckdb", "ok", ["x"], [[1], [2]]),
    }

    classification = classify_finding(
        case,
        finding,
        normalized,
        {},
        {"generator_profile": "common"},
        ["pandas", "duckdb"],
    )

    assert classification.verdict == "candidate_implementation_bug"
    assert "duckdb" in classification.evidence
    assert "pandas" in classification.evidence


def test_classification_reference_understands_null_aware_truth_filter():
    case = Case(
        "case-reference-truth-filter",
        33,
        [
            TableData(
                "t0",
                [ColumnSpec("id", "int", nullable=False), ColumnSpec("g", "str")],
                [{"id": 1, "g": "a"}, {"id": 2, "g": "b"}],
            ),
            TableData(
                "t1",
                [ColumnSpec("id", "int", nullable=False), ColumnSpec("j", "int")],
                [{"id": 1, "j": 100}],
            ),
        ],
        Program(
            "prog-reference-truth-filter",
            33,
            [
                {"op": "join", "table": "t1", "left_on": "id", "right_on": "id", "how": "left"},
                {"op": "filter", "column": "j", "cmp": "gt_is_not_true", "value": 150},
                {"op": "select", "columns": ["id", "g", "j"]},
            ],
        ),
    )
    finding = {
        "kind": "semantic_output_mismatch",
        "root_cause": "outer_join_truth_filter",
        "confidence": "medium",
        "suspicious_backends": ["datafusion"],
    }
    normalized = {
        "duckdb": NormalizedResult("duckdb", "ok", ["g", "id", "j"], [["a", 1, 100], ["b", 2, None]]),
        "datafusion": NormalizedResult("datafusion", "ok", ["g", "id", "j"], [["a", 1, 100]]),
    }

    classification = classify_finding(
        case,
        finding,
        normalized,
        {},
        {"generator_profile": "join_null_truth_filter"},
        ["duckdb", "datafusion"],
    )

    assert classification.verdict == "candidate_implementation_bug"
    assert classification.implicated_backends == ["datafusion"]


def test_classification_does_not_treat_explicit_null_predicate_as_null_comparison_noise():
    case = Case(
        "case-reference-null-predicate",
        32,
        [
            TableData(
                "t0",
                [ColumnSpec("id", "int"), ColumnSpec("s", "str")],
                [{"id": 1, "s": "alpha"}, {"id": 2, "s": None}],
            )
        ],
        Program("prog-reference-null-predicate", 32, [{"op": "filter", "column": "s", "cmp": "is_null", "value": None}]),
    )
    finding = {
        "kind": "semantic_output_mismatch",
        "root_cause": "filter_predicate",
        "confidence": "high",
        "suspicious_backends": ["datafusion"],
    }
    normalized = {
        "pandas": NormalizedResult("pandas", "ok", ["id", "s"], [[2, None]]),
        "duckdb": NormalizedResult("duckdb", "ok", ["id", "s"], [[2, None]]),
        "datafusion": NormalizedResult("datafusion", "ok", ["id", "s"], []),
    }

    classification = classify_finding(
        case,
        finding,
        normalized,
        {},
        {"generator_profile": "null_predicate_filter"},
        ["pandas", "duckdb", "datafusion"],
    )

    assert classification.verdict == "candidate_implementation_bug"
    assert classification.false_positive is False


def test_classification_reference_understands_join_null_key_topk():
    case = Case(
        "case-reference-join-null-key-topk",
        34,
        [
            TableData("t0", [ColumnSpec("id", "int", nullable=False), ColumnSpec("x", "int")], [{"id": 1, "x": -1}]),
            TableData("t1", [ColumnSpec("id", "int", nullable=False), ColumnSpec("j", "int")], [{"id": 9, "j": 9}]),
        ],
        Program(
            "prog-reference-join-null-key-topk",
            34,
            [
                {"op": "join", "table": "t1", "left_on": "id", "right_on": "id", "how": "left"},
                {"op": "groupby", "keys": ["j"], "aggs": [{"column": "x", "func": "count", "as": "count_x"}]},
                {"op": "select", "columns": ["j"]},
                {"op": "sort", "columns": ["j"], "ascending": True},
                {"op": "limit", "n": 4},
            ],
        ),
    )
    finding = {
        "kind": "semantic_output_mismatch",
        "root_cause": "grouped_topk_null_sort_key",
        "confidence": "medium",
        "suspicious_backends": ["datafusion"],
    }
    normalized = {
        "duckdb": NormalizedResult("duckdb", "ok", ["j"], [[None]]),
        "datafusion": NormalizedResult("datafusion", "ok", ["j"], []),
    }

    classification = classify_finding(
        case,
        finding,
        normalized,
        {},
        {"generator_profile": "join_null_key_topk"},
        ["duckdb", "datafusion"],
    )

    assert classification.verdict == "candidate_implementation_bug"
    assert classification.implicated_backends == ["datafusion"]


def test_classification_marks_modulo_as_expected_semantic_divergence_before_reference():
    case = Case(
        "case-mod",
        32,
        [TableData("t0", [ColumnSpec("x", "int")], [{"x": -3}, {"x": 4}])],
        Program(
            "prog-mod",
            32,
            [{"op": "mutate", "column": "m", "expr": {"kind": "arith_const", "op": "mod", "source": "x", "value": 2}}],
        ),
    )
    finding = {
        "kind": "semantic_output_mismatch",
        "root_cause": "arithmetic_expression",
        "confidence": "high",
        "suspicious_backends": ["sqlite"],
    }
    normalized = {
        "pandas": NormalizedResult("pandas", "ok", ["m", "x"], [[1, -3], [0, 4]]),
        "sqlite": NormalizedResult("sqlite", "ok", ["m", "x"], [[-1, -3], [0, 4]]),
    }

    classification = classify_finding(
        case,
        finding,
        normalized,
        {},
        {"generator_profile": "common"},
        ["pandas", "sqlite"],
    )

    assert classification.verdict == "expected_semantic_divergence"
    assert "modulo" in classification.evidence


def test_classification_marks_single_backend_metamorphic_violation_as_candidate_bug():
    case = _case([{"op": "filter", "column": "x", "cmp": ">", "value": 0}])
    finding = {
        "kind": "metamorphic_filter_idempotence_violation",
        "root_cause": "metamorphic_filter_idempotence",
        "oracle": "metamorphic",
        "confidence": "medium",
        "suspicious_backends": ["duckdb"],
    }

    classification = classify_finding(case, finding, {}, {}, {"generator_profile": "common"}, ["pandas", "duckdb"])

    assert classification.verdict == "candidate_implementation_bug"
    assert classification.confidence == "high"


def test_validate_case_rejects_cross_type_filter_literal():
    case = Case(
        "case-invalid-filter-type",
        4,
        [TableData("t0", [ColumnSpec("g", "str")], [{"g": "Alpha"}])],
        Program(
            "prog-invalid-filter-type",
            4,
            [
                {"op": "mutate", "column": "m_0", "expr": {"kind": "string_lower", "source": "g"}},
                {"op": "filter", "column": "m_0", "cmp": "<", "value": 0.5},
            ],
        ),
    )

    errors = validate_case_program(case)

    assert any("not supported for str filter" in error for error in errors)


def test_validate_case_accepts_null_aware_truth_filter_comparator():
    valid = Case(
        "case-truth-filter",
        4,
        [TableData("t0", [ColumnSpec("x", "int")], [{"x": 1}, {"x": None}])],
        Program("prog-truth-filter", 4, [{"op": "filter", "column": "x", "cmp": "gt_is_not_true", "value": 0}]),
    )
    invalid = Case(
        "case-invalid-truth-filter",
        4,
        [TableData("t0", [ColumnSpec("s", "str")], [{"s": "a"}])],
        Program("prog-invalid-truth-filter", 4, [{"op": "filter", "column": "s", "cmp": "gt_is_not_true", "value": "a"}]),
    )

    assert validate_case_program(valid) == []
    assert any("not supported for str filter" in error for error in validate_case_program(invalid))


def test_validate_case_accepts_typed_set_membership_filter():
    valid = Case(
        "case-set-filter",
        5,
        [TableData("t0", [ColumnSpec("s", "str")], [{"s": "alpha"}, {"s": None}])],
        Program("prog-set-filter", 5, [{"op": "filter", "column": "s", "cmp": "in_set", "value": ["alpha", "中文"]}]),
    )
    invalid_null = Case(
        "case-set-filter-null",
        6,
        [TableData("t0", [ColumnSpec("s", "str")], [{"s": "alpha"}])],
        Program("prog-set-filter-null", 6, [{"op": "filter", "column": "s", "cmp": "in_set", "value": ["alpha", None]}]),
    )
    invalid_type = Case(
        "case-set-filter-type",
        7,
        [TableData("t0", [ColumnSpec("x", "int")], [{"x": 1}])],
        Program("prog-set-filter-type", 7, [{"op": "filter", "column": "x", "cmp": "in_set", "value": [1, "2"]}]),
    )

    assert validate_case_program(valid) == []
    assert any("must not contain NULL" in error for error in validate_case_program(invalid_null))
    assert any("not compatible with int column" in error for error in validate_case_program(invalid_type))


def test_validate_case_accepts_typed_range_filter():
    valid = Case(
        "case-range-filter",
        13,
        [TableData("t0", [ColumnSpec("x", "int")], [{"x": 1}, {"x": None}])],
        Program("prog-range-filter", 13, [{"op": "filter", "column": "x", "cmp": "range_closed", "value": [0, 2]}]),
    )
    invalid_type = Case(
        "case-range-filter-type",
        14,
        [TableData("t0", [ColumnSpec("s", "str")], [{"s": "alpha"}])],
        Program("prog-range-filter-type", 14, [{"op": "filter", "column": "s", "cmp": "range_closed", "value": ["a", "z"]}]),
    )
    invalid_order = Case(
        "case-range-filter-order",
        15,
        [TableData("t0", [ColumnSpec("x", "int")], [{"x": 1}])],
        Program("prog-range-filter-order", 15, [{"op": "filter", "column": "x", "cmp": "range_closed", "value": [2, 0]}]),
    )

    assert validate_case_program(valid) == []
    assert any("not supported for str filter" in error for error in validate_case_program(invalid_type))
    assert any("lower bound must be <= upper bound" in error for error in validate_case_program(invalid_order))


def test_validate_case_accepts_tuple_absence_filter():
    valid = Case(
        "case-tuple-absence",
        18,
        [
            TableData("t0", [ColumnSpec("a", "int"), ColumnSpec("b", "int")], [{"a": 1, "b": 1}]),
            TableData("t1", [ColumnSpec("a", "int"), ColumnSpec("b", "int")], [{"a": None, "b": 4}]),
        ],
        Program(
            "prog-tuple-absence",
            18,
            [{"op": "tuple_absence_filter", "columns": ["a", "b"], "table": "t1", "right_columns": ["a", "b"]}],
        ),
    )
    invalid_type = Case(
        "case-tuple-absence-type",
        19,
        [
            TableData("t0", [ColumnSpec("a", "int"), ColumnSpec("b", "int")], [{"a": 1, "b": 1}]),
            TableData("t1", [ColumnSpec("a", "int"), ColumnSpec("s", "str")], [{"a": 1, "s": "x"}]),
        ],
        Program(
            "prog-tuple-absence-type",
            19,
            [{"op": "tuple_absence_filter", "columns": ["a", "b"], "table": "t1", "right_columns": ["a", "s"]}],
        ),
    )

    assert validate_case_program(valid) == []
    assert any("tuple absence type mismatch" in error for error in validate_case_program(invalid_type))


def test_validate_case_accepts_running_sum():
    valid = Case(
        "case-running-sum",
        20,
        [TableData("t0", [ColumnSpec("row_id", "int"), ColumnSpec("x", "float")], [{"row_id": 0, "x": 0.5}])],
        Program(
            "prog-running-sum",
            20,
            [
                {
                    "op": "running_sum",
                    "source": "x",
                    "column": "run_x",
                    "order_by": [{"column": "row_id", "ascending": True, "nulls": "last"}],
                    "input_dtype": "float32",
                }
            ],
        ),
    )
    invalid_type = Case(
        "case-running-sum-type",
        21,
        [TableData("t0", [ColumnSpec("row_id", "int"), ColumnSpec("s", "str")], [{"row_id": 0, "s": "x"}])],
        Program(
            "prog-running-sum-type",
            21,
            [
                {
                    "op": "running_sum",
                    "source": "s",
                    "column": "run_s",
                    "order_by": [{"column": "row_id", "ascending": True, "nulls": "last"}],
                }
            ],
        ),
    )
    invalid_order = Case(
        "case-running-sum-order",
        22,
        [TableData("t0", [ColumnSpec("x", "float")], [{"x": 0.5}])],
        Program(
            "prog-running-sum-order",
            22,
            [{"op": "running_sum", "source": "x", "column": "run_x", "order_by": []}],
        ),
    )

    assert validate_case_program(valid) == []
    assert any("not numeric" in error for error in validate_case_program(invalid_type))
    assert any("has no order_by" in error for error in validate_case_program(invalid_order))


def test_validate_case_accepts_sortedness_check():
    valid = Case(
        "case-sortedness",
        23,
        [TableData("t0", [ColumnSpec("x", "int")], [{"x": 1}, {"x": None}])],
        Program(
            "prog-sortedness",
            23,
            [
                {"op": "sort", "keys": [{"column": "x", "ascending": True, "nulls": "last"}]},
                {
                    "op": "sortedness_check",
                    "column": "x",
                    "as": "sorted_ok_x",
                    "ascending": True,
                    "nulls": "first",
                },
            ],
        ),
    )
    missing_column = Case(
        "case-sortedness-missing",
        24,
        [TableData("t0", [ColumnSpec("x", "int")], [{"x": 1}])],
        Program("prog-sortedness-missing", 24, [{"op": "sortedness_check", "column": "y", "as": "sorted_ok_y"}]),
    )
    bad_alias = Case(
        "case-sortedness-alias",
        25,
        [TableData("t0", [ColumnSpec("x", "int")], [{"x": 1}])],
        Program("prog-sortedness-alias", 25, [{"op": "sortedness_check", "column": "x", "as": "select"}]),
    )
    bad_ascending = Case(
        "case-sortedness-ascending",
        26,
        [TableData("t0", [ColumnSpec("x", "int")], [{"x": 1}])],
        Program(
            "prog-sortedness-ascending",
            26,
            [{"op": "sortedness_check", "column": "x", "as": "sorted_ok_x", "ascending": "yes"}],
        ),
    )
    bad_nulls = Case(
        "case-sortedness-nulls",
        27,
        [TableData("t0", [ColumnSpec("x", "int")], [{"x": 1}])],
        Program(
            "prog-sortedness-nulls",
            27,
            [{"op": "sortedness_check", "column": "x", "as": "sorted_ok_x", "nulls": "middle"}],
        ),
    )

    assert validate_case_program(valid) == []
    assert any("sortedness_check column" in error for error in validate_case_program(missing_column))
    assert any("reserved" in error for error in validate_case_program(bad_alias))
    assert any("ascending must be boolean" in error for error in validate_case_program(bad_ascending))
    assert any("nulls must be 'first' or 'last'" in error for error in validate_case_program(bad_nulls))


def test_validate_case_accepts_random_case_probe():
    valid = Case(
        "case-random-case-probe",
        28,
        [TableData("t0", [ColumnSpec("probe_id", "int")], [{"probe_id": 0}])],
        Program(
            "prog-random-case-probe",
            28,
            [{"op": "random_case_probe", "as": "unexpected_else_seen", "rows": 100_000, "branches": 3}],
        ),
    )
    bad_alias = Case(
        "case-random-case-probe-alias",
        29,
        [TableData("t0", [ColumnSpec("probe_id", "int")], [{"probe_id": 0}])],
        Program("prog-random-case-probe-alias", 29, [{"op": "random_case_probe", "as": "where"}]),
    )
    bad_rows = Case(
        "case-random-case-probe-rows",
        30,
        [TableData("t0", [ColumnSpec("probe_id", "int")], [{"probe_id": 0}])],
        Program("prog-random-case-probe-rows", 30, [{"op": "random_case_probe", "as": "ok", "rows": 0}]),
    )
    bad_branches = Case(
        "case-random-case-probe-branches",
        31,
        [TableData("t0", [ColumnSpec("probe_id", "int")], [{"probe_id": 0}])],
        Program("prog-random-case-probe-branches", 31, [{"op": "random_case_probe", "as": "ok", "branches": 17}]),
    )

    assert validate_case_program(valid) == []
    assert any("reserved" in error for error in validate_case_program(bad_alias))
    assert any("rows must be positive" in error for error in validate_case_program(bad_rows))
    assert any("branches must be between" in error for error in validate_case_program(bad_branches))


def test_validate_case_accepts_group_quantile_probe():
    valid = Case(
        "case-group-quantile-probe",
        32,
        [TableData("t0", [ColumnSpec("probe_id", "int")], [{"probe_id": 0}])],
        Program(
            "prog-group-quantile-probe",
            32,
            [
                {
                    "op": "group_quantile_probe",
                    "as": "quantile_key_mismatch",
                    "values": [1, 2, 3],
                    "quantiles": [0.0, 0.5, 1.0],
                }
            ],
        ),
    )
    bad_alias = Case(
        "case-group-quantile-probe-alias",
        33,
        [TableData("t0", [ColumnSpec("probe_id", "int")], [{"probe_id": 0}])],
        Program("prog-group-quantile-probe-alias", 33, [{"op": "group_quantile_probe", "as": "where"}]),
    )
    bad_values = Case(
        "case-group-quantile-probe-values",
        34,
        [TableData("t0", [ColumnSpec("probe_id", "int")], [{"probe_id": 0}])],
        Program(
            "prog-group-quantile-probe-values",
            34,
            [{"op": "group_quantile_probe", "as": "ok", "values": [1], "quantiles": [0.0, 1.1]}],
        ),
    )

    assert validate_case_program(valid) == []
    assert any("reserved" in error for error in validate_case_program(bad_alias))
    assert any("numeric values and quantiles" in error for error in validate_case_program(bad_values))


def test_validate_case_accepts_scalar_subquery_probe():
    valid = Case(
        "case-scalar-subquery-probe",
        35,
        [TableData("t0", [ColumnSpec("probe_id", "int")], [{"probe_id": 0}])],
        Program("prog-scalar-subquery-probe", 35, [{"op": "scalar_subquery_probe", "as": "scalar_subquery_mismatch"}]),
    )
    bad_alias = Case(
        "case-scalar-subquery-probe-alias",
        36,
        [TableData("t0", [ColumnSpec("probe_id", "int")], [{"probe_id": 0}])],
        Program("prog-scalar-subquery-probe-alias", 36, [{"op": "scalar_subquery_probe", "as": "where"}]),
    )

    assert validate_case_program(valid) == []
    assert any("reserved" in error for error in validate_case_program(bad_alias))


def test_validate_case_accepts_window_avg_probe():
    valid = Case(
        "case-window-avg-probe",
        37,
        [TableData("t0", [ColumnSpec("probe_id", "int")], [{"probe_id": 0}])],
        Program("prog-window-avg-probe", 37, [{"op": "window_avg_probe", "as": "window_avg_mismatch"}]),
    )
    bad_alias = Case(
        "case-window-avg-probe-alias",
        38,
        [TableData("t0", [ColumnSpec("probe_id", "int")], [{"probe_id": 0}])],
        Program("prog-window-avg-probe-alias", 38, [{"op": "window_avg_probe", "as": "where"}]),
    )

    assert validate_case_program(valid) == []
    assert any("reserved" in error for error in validate_case_program(bad_alias))


def test_validate_case_accepts_struct_distinct_probe():
    valid = Case(
        "case-struct-distinct-probe",
        39,
        [TableData("t0", [ColumnSpec("probe_id", "int")], [{"probe_id": 0}])],
        Program("prog-struct-distinct-probe", 39, [{"op": "struct_distinct_probe", "as": "struct_distinct_mismatch"}]),
    )
    bad_alias = Case(
        "case-struct-distinct-probe-alias",
        40,
        [TableData("t0", [ColumnSpec("probe_id", "int")], [{"probe_id": 0}])],
        Program("prog-struct-distinct-probe-alias", 40, [{"op": "struct_distinct_probe", "as": "where"}]),
    )

    assert validate_case_program(valid) == []
    assert any("reserved" in error for error in validate_case_program(bad_alias))


def test_validate_case_accepts_bit_compare_probe():
    valid = Case(
        "case-bit-compare-probe",
        41,
        [TableData("t0", [ColumnSpec("probe_id", "int")], [{"probe_id": 0}])],
        Program("prog-bit-compare-probe", 41, [{"op": "bit_compare_probe", "as": "bit_compare_mismatch"}]),
    )
    bad_alias = Case(
        "case-bit-compare-probe-alias",
        42,
        [TableData("t0", [ColumnSpec("probe_id", "int")], [{"probe_id": 0}])],
        Program("prog-bit-compare-probe-alias", 42, [{"op": "bit_compare_probe", "as": "where"}]),
    )

    assert validate_case_program(valid) == []
    assert any("reserved" in error for error in validate_case_program(bad_alias))


def test_classification_reference_understands_bit_compare_probe():
    case = Case(
        "case-bit-compare-reference",
        43,
        [TableData("t0", [ColumnSpec("probe_id", "int")], [{"probe_id": 0}])],
        Program("prog-bit-compare-reference", 43, [{"op": "bit_compare_probe", "as": "bit_compare_mismatch"}]),
    )
    finding = {
        "kind": "semantic_output_mismatch",
        "root_cause": "bit_compare_unequal_length",
        "confidence": "high",
        "suspicious_backends": ["duckdb"],
    }
    normalized = {
        "pandas": NormalizedResult("pandas", "ok", ["bit_compare_mismatch"], [[False]]),
        "duckdb": NormalizedResult("duckdb", "ok", ["bit_compare_mismatch"], [[True]]),
    }

    classification = classify_finding(
        case,
        finding,
        normalized,
        {},
        {"generator_profile": "bit_compare_unequal_length"},
        ["pandas", "duckdb"],
    )

    assert classification.verdict == "candidate_implementation_bug"
    assert classification.implicated_backends == ["duckdb"]


def test_validate_case_accepts_round_even_probe():
    valid = Case(
        "case-round-even-probe",
        44,
        [TableData("t0", [ColumnSpec("probe_id", "int")], [{"probe_id": 0}])],
        Program("prog-round-even-probe", 44, [{"op": "round_even_probe", "as": "round_even_mismatch"}]),
    )
    bad_alias = Case(
        "case-round-even-probe-alias",
        45,
        [TableData("t0", [ColumnSpec("probe_id", "int")], [{"probe_id": 0}])],
        Program("prog-round-even-probe-alias", 45, [{"op": "round_even_probe", "as": "where"}]),
    )

    assert validate_case_program(valid) == []
    assert any("reserved" in error for error in validate_case_program(bad_alias))


def test_classification_reference_understands_round_even_probe():
    case = Case(
        "case-round-even-reference",
        46,
        [TableData("t0", [ColumnSpec("probe_id", "int")], [{"probe_id": 0}])],
        Program("prog-round-even-reference", 46, [{"op": "round_even_probe", "as": "round_even_mismatch"}]),
    )
    finding = {
        "kind": "semantic_output_mismatch",
        "root_cause": "round_even_float_scale",
        "confidence": "high",
        "suspicious_backends": ["duckdb"],
    }
    normalized = {
        "pandas": NormalizedResult("pandas", "ok", ["round_even_mismatch"], [[False]]),
        "duckdb": NormalizedResult("duckdb", "ok", ["round_even_mismatch"], [[True]]),
    }

    classification = classify_finding(
        case,
        finding,
        normalized,
        {},
        {"generator_profile": "round_even_float_scale"},
        ["pandas", "duckdb"],
    )

    assert classification.verdict == "candidate_implementation_bug"
    assert classification.implicated_backends == ["duckdb"]


def test_validate_case_accepts_series_rtruediv_probe():
    valid = Case(
        "case-series-rtruediv-probe",
        47,
        [TableData("t0", [ColumnSpec("probe_id", "int")], [{"probe_id": 0}])],
        Program(
            "prog-series-rtruediv-probe",
            47,
            [{"op": "series_rtruediv_probe", "as": "series_rtruediv_mismatch"}],
        ),
    )
    bad_alias = Case(
        "case-series-rtruediv-probe-alias",
        48,
        [TableData("t0", [ColumnSpec("probe_id", "int")], [{"probe_id": 0}])],
        Program("prog-series-rtruediv-probe-alias", 48, [{"op": "series_rtruediv_probe", "as": "where"}]),
    )

    assert validate_case_program(valid) == []
    assert any("reserved" in error for error in validate_case_program(bad_alias))


def test_classification_reference_understands_series_rtruediv_probe():
    case = Case(
        "case-series-rtruediv-reference",
        49,
        [TableData("t0", [ColumnSpec("probe_id", "int")], [{"probe_id": 0}])],
        Program(
            "prog-series-rtruediv-reference",
            49,
            [{"op": "series_rtruediv_probe", "as": "series_rtruediv_mismatch"}],
        ),
    )
    finding = {
        "kind": "semantic_output_mismatch",
        "root_cause": "series_rtruediv_operand_order",
        "confidence": "high",
        "suspicious_backends": ["polars"],
    }
    normalized = {
        "pandas": NormalizedResult("pandas", "ok", ["series_rtruediv_mismatch"], [[False]]),
        "polars": NormalizedResult("polars", "ok", ["series_rtruediv_mismatch"], [[True]]),
    }

    classification = classify_finding(
        case,
        finding,
        normalized,
        {},
        {"generator_profile": "series_rtruediv_operand_order"},
        ["pandas", "polars"],
    )

    assert classification.verdict == "candidate_implementation_bug"
    assert classification.implicated_backends == ["polars"]


def test_validate_case_accepts_uint64_isin_probe():
    valid = Case(
        "case-uint64-isin-probe",
        50,
        [TableData("t0", [ColumnSpec("probe_id", "int")], [{"probe_id": 0}])],
        Program("prog-uint64-isin-probe", 50, [{"op": "uint64_isin_probe", "as": "uint64_isin_mismatch"}]),
    )
    bad_alias = Case(
        "case-uint64-isin-probe-alias",
        51,
        [TableData("t0", [ColumnSpec("probe_id", "int")], [{"probe_id": 0}])],
        Program("prog-uint64-isin-probe-alias", 51, [{"op": "uint64_isin_probe", "as": "where"}]),
    )

    assert validate_case_program(valid) == []
    assert any("reserved" in error for error in validate_case_program(bad_alias))


def test_classification_reference_understands_uint64_isin_probe():
    case = Case(
        "case-uint64-isin-reference",
        52,
        [TableData("t0", [ColumnSpec("probe_id", "int")], [{"probe_id": 0}])],
        Program("prog-uint64-isin-reference", 52, [{"op": "uint64_isin_probe", "as": "uint64_isin_mismatch"}]),
    )
    finding = {
        "kind": "semantic_output_mismatch",
        "root_cause": "pandas_uint64_isin_precision",
        "confidence": "high",
        "suspicious_backends": ["pandas"],
    }
    normalized = {
        "reference": NormalizedResult("reference", "ok", ["uint64_isin_mismatch"], [[False]]),
        "pandas": NormalizedResult("pandas", "ok", ["uint64_isin_mismatch"], [[True]]),
    }

    classification = classify_finding(
        case,
        finding,
        normalized,
        {},
        {"generator_profile": "pandas_uint64_isin_precision"},
        ["reference", "pandas"],
    )

    assert classification.verdict == "candidate_implementation_bug"
    assert classification.implicated_backends == ["pandas"]


def test_validate_case_accepts_tuple_anti_null_probe():
    valid = Case(
        "case-tuple-anti-null-probe",
        53,
        [TableData("t0", [ColumnSpec("probe_id", "int")], [{"probe_id": 0}])],
        Program("prog-tuple-anti-null-probe", 53, [{"op": "tuple_anti_null_probe", "as": "tuple_anti_null_mismatch"}]),
    )
    bad_alias = Case(
        "case-tuple-anti-null-probe-alias",
        54,
        [TableData("t0", [ColumnSpec("probe_id", "int")], [{"probe_id": 0}])],
        Program("prog-tuple-anti-null-probe-alias", 54, [{"op": "tuple_anti_null_probe", "as": "where"}]),
    )

    assert validate_case_program(valid) == []
    assert any("reserved" in error for error in validate_case_program(bad_alias))


def test_classification_reference_understands_tuple_anti_null_probe():
    case = Case(
        "case-tuple-anti-null-reference",
        55,
        [TableData("t0", [ColumnSpec("probe_id", "int")], [{"probe_id": 0}])],
        Program("prog-tuple-anti-null-reference", 55, [{"op": "tuple_anti_null_probe", "as": "tuple_anti_null_mismatch"}]),
    )
    finding = {
        "kind": "semantic_output_mismatch",
        "root_cause": "duckdb_tuple_anti_null_semantics",
        "confidence": "high",
        "suspicious_backends": ["duckdb"],
    }
    normalized = {
        "reference": NormalizedResult("reference", "ok", ["tuple_anti_null_mismatch"], [[False]]),
        "duckdb": NormalizedResult("duckdb", "ok", ["tuple_anti_null_mismatch"], [[True]]),
    }

    classification = classify_finding(
        case,
        finding,
        normalized,
        {},
        {"generator_profile": "duckdb_tuple_anti_null_semantics"},
        ["reference", "duckdb"],
    )

    assert classification.verdict == "candidate_implementation_bug"
    assert classification.implicated_backends == ["duckdb"]


def test_validate_case_accepts_json_predicate_order_probe():
    valid = Case(
        "case-json-predicate-order-probe",
        86,
        [TableData("t0", [ColumnSpec("probe_id", "int")], [{"probe_id": 0}])],
        Program(
            "prog-json-predicate-order-probe",
            86,
            [{"op": "json_predicate_order_probe", "as": "json_predicate_order_mismatch"}],
        ),
    )
    bad_alias = Case(
        "case-json-predicate-order-probe-alias",
        87,
        [TableData("t0", [ColumnSpec("probe_id", "int")], [{"probe_id": 0}])],
        Program(
            "prog-json-predicate-order-probe-alias",
            87,
            [{"op": "json_predicate_order_probe", "as": "select"}],
        ),
    )

    assert validate_case_program(valid) == []
    assert any("reserved" in error for error in validate_case_program(bad_alias))


def test_classification_reference_understands_json_predicate_order_probe():
    case = Case(
        "case-json-predicate-order-reference",
        88,
        [TableData("t0", [ColumnSpec("probe_id", "int")], [{"probe_id": 0}])],
        Program(
            "prog-json-predicate-order-reference",
            88,
            [{"op": "json_predicate_order_probe", "as": "json_predicate_order_mismatch"}],
        ),
    )
    finding = {
        "kind": "semantic_output_mismatch",
        "root_cause": "duckdb_json_predicate_order_semantics",
        "confidence": "high",
        "suspicious_backends": ["duckdb", "duckdb_persistent"],
    }
    normalized = {
        "reference": NormalizedResult("reference", "ok", ["json_predicate_order_mismatch"], [[False]]),
        "duckdb": NormalizedResult("duckdb", "ok", ["json_predicate_order_mismatch"], [[True]]),
    }

    classification = classify_finding(
        case,
        finding,
        normalized,
        {},
        {"generator_profile": "duckdb_json_predicate_order_semantics"},
        ["reference", "duckdb"],
    )

    assert classification.verdict == "candidate_implementation_bug"
    assert classification.implicated_backends == ["duckdb"]


def test_validate_case_accepts_sparse_mask_probe():
    valid = Case(
        "case-sparse-mask-probe",
        56,
        [TableData("t0", [ColumnSpec("probe_id", "int")], [{"probe_id": 0}])],
        Program("prog-sparse-mask-probe", 56, [{"op": "sparse_mask_probe", "as": "sparse_mask_mismatch"}]),
    )
    bad_alias = Case(
        "case-sparse-mask-probe-alias",
        57,
        [TableData("t0", [ColumnSpec("probe_id", "int")], [{"probe_id": 0}])],
        Program("prog-sparse-mask-probe-alias", 57, [{"op": "sparse_mask_probe", "as": "where"}]),
    )

    assert validate_case_program(valid) == []
    assert any("reserved" in error for error in validate_case_program(bad_alias))


def test_classification_reference_understands_sparse_mask_probe():
    case = Case(
        "case-sparse-mask-reference",
        58,
        [TableData("t0", [ColumnSpec("probe_id", "int")], [{"probe_id": 0}])],
        Program("prog-sparse-mask-reference", 58, [{"op": "sparse_mask_probe", "as": "sparse_mask_mismatch"}]),
    )
    finding = {
        "kind": "semantic_output_mismatch",
        "root_cause": "pandas_sparse_array_mask_semantics",
        "confidence": "high",
        "suspicious_backends": ["pandas"],
    }
    normalized = {
        "reference": NormalizedResult("reference", "ok", ["sparse_mask_mismatch"], [[False]]),
        "pandas": NormalizedResult("pandas", "ok", ["sparse_mask_mismatch"], [[True]]),
    }

    classification = classify_finding(
        case,
        finding,
        normalized,
        {},
        {"generator_profile": "pandas_sparse_array_mask_semantics"},
        ["reference", "pandas"],
    )

    assert classification.verdict == "candidate_implementation_bug"
    assert classification.implicated_backends == ["pandas"]


def test_validate_case_accepts_float_wrap_probe():
    valid = Case(
        "case-float-wrap-probe",
        59,
        [TableData("t0", [ColumnSpec("probe_id", "int")], [{"probe_id": 0}])],
        Program("prog-float-wrap-probe", 59, [{"op": "float_wrap_probe", "as": "float_wrap_mismatch"}]),
    )
    bad_alias = Case(
        "case-float-wrap-probe-alias",
        60,
        [TableData("t0", [ColumnSpec("probe_id", "int")], [{"probe_id": 0}])],
        Program("prog-float-wrap-probe-alias", 60, [{"op": "float_wrap_probe", "as": "where"}]),
    )

    assert validate_case_program(valid) == []
    assert any("reserved" in error for error in validate_case_program(bad_alias))


def test_classification_reference_understands_float_wrap_probe():
    case = Case(
        "case-float-wrap-reference",
        61,
        [TableData("t0", [ColumnSpec("probe_id", "int")], [{"probe_id": 0}])],
        Program("prog-float-wrap-reference", 61, [{"op": "float_wrap_probe", "as": "float_wrap_mismatch"}]),
    )
    finding = {
        "kind": "semantic_output_mismatch",
        "root_cause": "polars_float_wrap_numerical_semantics",
        "confidence": "high",
        "suspicious_backends": ["polars", "polars_lazy"],
    }
    normalized = {
        "reference": NormalizedResult("reference", "ok", ["float_wrap_mismatch"], [[False]]),
        "polars": NormalizedResult("polars", "ok", ["float_wrap_mismatch"], [[True]]),
        "polars_lazy": NormalizedResult("polars_lazy", "ok", ["float_wrap_mismatch"], [[True]]),
    }

    classification = classify_finding(
        case,
        finding,
        normalized,
        {},
        {"generator_profile": "polars_float_wrap_numerical_semantics"},
        ["reference", "polars", "polars_lazy"],
    )

    assert classification.verdict == "candidate_implementation_bug"
    assert classification.implicated_backends == ["polars", "polars_lazy"]


def test_validate_case_accepts_index_bool_probe():
    valid = Case(
        "case-index-bool-probe",
        62,
        [TableData("t0", [ColumnSpec("probe_id", "int")], [{"probe_id": 0}])],
        Program("prog-index-bool-probe", 62, [{"op": "index_bool_probe", "as": "index_bool_mismatch"}]),
    )
    bad_alias = Case(
        "case-index-bool-probe-alias",
        63,
        [TableData("t0", [ColumnSpec("probe_id", "int")], [{"probe_id": 0}])],
        Program("prog-index-bool-probe-alias", 63, [{"op": "index_bool_probe", "as": "where"}]),
    )

    assert validate_case_program(valid) == []
    assert any("reserved" in error for error in validate_case_program(bad_alias))


def test_classification_reference_understands_index_bool_probe():
    case = Case(
        "case-index-bool-reference",
        64,
        [TableData("t0", [ColumnSpec("probe_id", "int")], [{"probe_id": 0}])],
        Program("prog-index-bool-reference", 64, [{"op": "index_bool_probe", "as": "index_bool_mismatch"}]),
    )
    finding = {
        "kind": "semantic_output_mismatch",
        "root_cause": "pandas_index_bool_result_type",
        "confidence": "high",
        "suspicious_backends": ["pandas"],
    }
    normalized = {
        "reference": NormalizedResult("reference", "ok", ["index_bool_mismatch"], [[False]]),
        "pandas": NormalizedResult("pandas", "ok", ["index_bool_mismatch"], [[True]]),
    }

    classification = classify_finding(
        case,
        finding,
        normalized,
        {},
        {"generator_profile": "pandas_index_bool_result_type"},
        ["reference", "pandas"],
    )

    assert classification.verdict == "candidate_implementation_bug"
    assert classification.implicated_backends == ["pandas"]


def test_validate_case_accepts_empty_literal_groupby_probe():
    valid = Case(
        "case-empty-literal-groupby-probe",
        65,
        [TableData("t0", [ColumnSpec("probe_id", "int")], [{"probe_id": 0}])],
        Program(
            "prog-empty-literal-groupby-probe",
            65,
            [{"op": "empty_literal_groupby_probe", "as": "empty_literal_groupby_mismatch"}],
        ),
    )
    bad_alias = Case(
        "case-empty-literal-groupby-probe-alias",
        66,
        [TableData("t0", [ColumnSpec("probe_id", "int")], [{"probe_id": 0}])],
        Program("prog-empty-literal-groupby-probe-alias", 66, [{"op": "empty_literal_groupby_probe", "as": "where"}]),
    )

    assert validate_case_program(valid) == []
    assert any("reserved" in error for error in validate_case_program(bad_alias))


def test_classification_reference_understands_empty_literal_groupby_probe():
    case = Case(
        "case-empty-literal-groupby-reference",
        67,
        [TableData("t0", [ColumnSpec("probe_id", "int")], [{"probe_id": 0}])],
        Program(
            "prog-empty-literal-groupby-reference",
            67,
            [{"op": "empty_literal_groupby_probe", "as": "empty_literal_groupby_mismatch"}],
        ),
    )
    finding = {
        "kind": "semantic_output_mismatch",
        "root_cause": "polars_empty_literal_groupby_semantics",
        "confidence": "high",
        "suspicious_backends": ["polars", "polars_lazy"],
    }
    normalized = {
        "reference": NormalizedResult("reference", "ok", ["empty_literal_groupby_mismatch"], [[False]]),
        "polars": NormalizedResult("polars", "ok", ["empty_literal_groupby_mismatch"], [[True]]),
        "polars_lazy": NormalizedResult("polars_lazy", "ok", ["empty_literal_groupby_mismatch"], [[True]]),
    }

    classification = classify_finding(
        case,
        finding,
        normalized,
        {},
        {"generator_profile": "polars_empty_literal_groupby_semantics"},
        ["reference", "polars", "polars_lazy"],
    )

    assert classification.verdict == "candidate_implementation_bug"
    assert classification.implicated_backends == ["polars", "polars_lazy"]


def test_validate_case_accepts_arrow_string_eq_sum_probe():
    valid = Case(
        "case-arrow-string-eq-sum-probe",
        68,
        [TableData("t0", [ColumnSpec("probe_id", "int")], [{"probe_id": 0}])],
        Program(
            "prog-arrow-string-eq-sum-probe",
            68,
            [{"op": "arrow_string_eq_sum_probe", "as": "arrow_string_eq_sum_mismatch"}],
        ),
    )
    bad_alias = Case(
        "case-arrow-string-eq-sum-probe-alias",
        69,
        [TableData("t0", [ColumnSpec("probe_id", "int")], [{"probe_id": 0}])],
        Program("prog-arrow-string-eq-sum-probe-alias", 69, [{"op": "arrow_string_eq_sum_probe", "as": "where"}]),
    )

    assert validate_case_program(valid) == []
    assert any("reserved" in error for error in validate_case_program(bad_alias))


def test_classification_reference_understands_arrow_string_eq_sum_probe():
    case = Case(
        "case-arrow-string-eq-sum-reference",
        70,
        [TableData("t0", [ColumnSpec("probe_id", "int")], [{"probe_id": 0}])],
        Program(
            "prog-arrow-string-eq-sum-reference",
            70,
            [{"op": "arrow_string_eq_sum_probe", "as": "arrow_string_eq_sum_mismatch"}],
        ),
    )
    finding = {
        "kind": "semantic_output_mismatch",
        "root_cause": "pandas_arrow_string_eq_sum_semantics",
        "confidence": "high",
        "suspicious_backends": ["pandas"],
    }
    normalized = {
        "reference": NormalizedResult("reference", "ok", ["arrow_string_eq_sum_mismatch"], [[False]]),
        "pandas": NormalizedResult("pandas", "ok", ["arrow_string_eq_sum_mismatch"], [[True]]),
    }

    classification = classify_finding(
        case,
        finding,
        normalized,
        {},
        {"generator_profile": "pandas_arrow_string_eq_sum_semantics"},
        ["reference", "pandas"],
    )

    assert classification.verdict == "candidate_implementation_bug"
    assert classification.implicated_backends == ["pandas"]


def test_validate_case_accepts_arrow_timestamp_loc_slice_probe():
    valid = Case(
        "case-arrow-timestamp-loc-slice-probe",
        71,
        [TableData("t0", [ColumnSpec("probe_id", "int")], [{"probe_id": 0}])],
        Program(
            "prog-arrow-timestamp-loc-slice-probe",
            71,
            [{"op": "arrow_timestamp_loc_slice_probe", "as": "arrow_timestamp_loc_slice_mismatch"}],
        ),
    )
    bad_alias = Case(
        "case-arrow-timestamp-loc-slice-probe-alias",
        72,
        [TableData("t0", [ColumnSpec("probe_id", "int")], [{"probe_id": 0}])],
        Program(
            "prog-arrow-timestamp-loc-slice-probe-alias",
            72,
            [{"op": "arrow_timestamp_loc_slice_probe", "as": "where"}],
        ),
    )

    assert validate_case_program(valid) == []
    assert any("reserved" in error for error in validate_case_program(bad_alias))


def test_classification_reference_understands_arrow_timestamp_loc_slice_probe():
    case = Case(
        "case-arrow-timestamp-loc-slice-reference",
        73,
        [TableData("t0", [ColumnSpec("probe_id", "int")], [{"probe_id": 0}])],
        Program(
            "prog-arrow-timestamp-loc-slice-reference",
            73,
            [{"op": "arrow_timestamp_loc_slice_probe", "as": "arrow_timestamp_loc_slice_mismatch"}],
        ),
    )
    finding = {
        "kind": "semantic_output_mismatch",
        "root_cause": "pandas_arrow_timestamp_loc_slice_semantics",
        "confidence": "high",
        "suspicious_backends": ["pandas"],
    }
    normalized = {
        "reference": NormalizedResult("reference", "ok", ["arrow_timestamp_loc_slice_mismatch"], [[False]]),
        "pandas": NormalizedResult("pandas", "ok", ["arrow_timestamp_loc_slice_mismatch"], [[True]]),
    }

    classification = classify_finding(
        case,
        finding,
        normalized,
        {},
        {"generator_profile": "pandas_arrow_timestamp_loc_slice_semantics"},
        ["reference", "pandas"],
    )

    assert classification.verdict == "candidate_implementation_bug"
    assert classification.implicated_backends == ["pandas"]


def test_validate_case_accepts_arrow_timestamp_index_attr_probe():
    valid = Case(
        "case-arrow-timestamp-index-attr-probe",
        74,
        [TableData("t0", [ColumnSpec("probe_id", "int")], [{"probe_id": 0}])],
        Program(
            "prog-arrow-timestamp-index-attr-probe",
            74,
            [{"op": "arrow_timestamp_index_attr_probe", "as": "arrow_timestamp_index_attr_mismatch"}],
        ),
    )
    bad_alias = Case(
        "case-arrow-timestamp-index-attr-probe-alias",
        75,
        [TableData("t0", [ColumnSpec("probe_id", "int")], [{"probe_id": 0}])],
        Program(
            "prog-arrow-timestamp-index-attr-probe-alias",
            75,
            [{"op": "arrow_timestamp_index_attr_probe", "as": "where"}],
        ),
    )

    assert validate_case_program(valid) == []
    assert any("reserved" in error for error in validate_case_program(bad_alias))


def test_classification_reference_understands_arrow_timestamp_index_attr_probe():
    case = Case(
        "case-arrow-timestamp-index-attr-reference",
        76,
        [TableData("t0", [ColumnSpec("probe_id", "int")], [{"probe_id": 0}])],
        Program(
            "prog-arrow-timestamp-index-attr-reference",
            76,
            [{"op": "arrow_timestamp_index_attr_probe", "as": "arrow_timestamp_index_attr_mismatch"}],
        ),
    )
    finding = {
        "kind": "semantic_output_mismatch",
        "root_cause": "pandas_arrow_timestamp_index_attr_semantics",
        "confidence": "high",
        "suspicious_backends": ["pandas"],
    }
    normalized = {
        "reference": NormalizedResult("reference", "ok", ["arrow_timestamp_index_attr_mismatch"], [[False]]),
        "pandas": NormalizedResult("pandas", "ok", ["arrow_timestamp_index_attr_mismatch"], [[True]]),
    }

    classification = classify_finding(
        case,
        finding,
        normalized,
        {},
        {"generator_profile": "pandas_arrow_timestamp_index_attr_semantics"},
        ["reference", "pandas"],
    )

    assert classification.verdict == "candidate_implementation_bug"
    assert classification.implicated_backends == ["pandas"]


def test_validate_case_accepts_dataset_isin_all_match_probe():
    valid = Case(
        "case-dataset-isin-all-match-probe",
        77,
        [TableData("t0", [ColumnSpec("probe_id", "int")], [{"probe_id": 0}])],
        Program(
            "prog-dataset-isin-all-match-probe",
            77,
            [{"op": "dataset_isin_all_match_probe", "as": "dataset_isin_all_match_mismatch"}],
        ),
    )
    bad_alias = Case(
        "case-dataset-isin-all-match-probe-alias",
        78,
        [TableData("t0", [ColumnSpec("probe_id", "int")], [{"probe_id": 0}])],
        Program("prog-dataset-isin-all-match-probe-alias", 78, [{"op": "dataset_isin_all_match_probe", "as": "where"}]),
    )

    assert validate_case_program(valid) == []
    assert any("reserved" in error for error in validate_case_program(bad_alias))


def test_classification_reference_understands_dataset_isin_all_match_probe():
    case = Case(
        "case-dataset-isin-all-match-reference",
        79,
        [TableData("t0", [ColumnSpec("probe_id", "int")], [{"probe_id": 0}])],
        Program(
            "prog-dataset-isin-all-match-reference",
            79,
            [{"op": "dataset_isin_all_match_probe", "as": "dataset_isin_all_match_mismatch"}],
        ),
    )
    finding = {
        "kind": "semantic_output_mismatch",
        "root_cause": "pyarrow_dataset_isin_all_match_semantics",
        "confidence": "high",
        "suspicious_backends": ["pyarrow"],
    }
    normalized = {
        "reference": NormalizedResult("reference", "ok", ["dataset_isin_all_match_mismatch"], [[False]]),
        "pyarrow": NormalizedResult("pyarrow", "ok", ["dataset_isin_all_match_mismatch"], [[True]]),
    }

    classification = classify_finding(
        case,
        finding,
        normalized,
        {},
        {"generator_profile": "pyarrow_dataset_isin_all_match_semantics"},
        ["reference", "pyarrow"],
    )

    assert classification.verdict == "candidate_implementation_bug"
    assert classification.implicated_backends == ["pyarrow"]


def test_validate_case_accepts_eval_inplace_alias_probe():
    valid = Case(
        "case-eval-inplace-alias-probe",
        89,
        [TableData("t0", [ColumnSpec("probe_id", "int")], [{"probe_id": 0}])],
        Program(
            "prog-eval-inplace-alias-probe",
            89,
            [{"op": "eval_inplace_alias_probe", "as": "eval_inplace_alias_mismatch"}],
        ),
    )
    bad_alias = Case(
        "case-eval-inplace-alias-probe-alias",
        90,
        [TableData("t0", [ColumnSpec("probe_id", "int")], [{"probe_id": 0}])],
        Program(
            "prog-eval-inplace-alias-probe-alias",
            90,
            [{"op": "eval_inplace_alias_probe", "as": "where"}],
        ),
    )

    assert validate_case_program(valid) == []
    assert any("reserved" in error for error in validate_case_program(bad_alias))


def test_classification_reference_understands_eval_inplace_alias_probe():
    case = Case(
        "case-eval-inplace-alias-reference",
        91,
        [TableData("t0", [ColumnSpec("probe_id", "int")], [{"probe_id": 0}])],
        Program(
            "prog-eval-inplace-alias-reference",
            91,
            [{"op": "eval_inplace_alias_probe", "as": "eval_inplace_alias_mismatch"}],
        ),
    )
    finding = {
        "kind": "semantic_output_mismatch",
        "root_cause": "pandas_eval_inplace_aliasing_semantics",
        "confidence": "high",
        "suspicious_backends": ["pandas"],
    }
    normalized = {
        "reference": NormalizedResult("reference", "ok", ["eval_inplace_alias_mismatch"], [[False]]),
        "pandas": NormalizedResult("pandas", "ok", ["eval_inplace_alias_mismatch"], [[True]]),
    }

    classification = classify_finding(
        case,
        finding,
        normalized,
        {},
        {"generator_profile": "pandas_eval_inplace_aliasing_semantics"},
        ["reference", "pandas"],
    )

    assert classification.verdict == "candidate_implementation_bug"
    assert classification.implicated_backends == ["pandas"]


def test_validate_case_accepts_rolling_mean_by_null_count_probe():
    valid = Case(
        "case-rolling-mean-by-null-count-probe",
        80,
        [TableData("t0", [ColumnSpec("probe_id", "int")], [{"probe_id": 0}])],
        Program(
            "prog-rolling-mean-by-null-count-probe",
            80,
            [{"op": "rolling_mean_by_null_count_probe", "as": "rolling_mean_by_null_count_mismatch"}],
        ),
    )
    bad_alias = Case(
        "case-rolling-mean-by-null-count-probe-alias",
        81,
        [TableData("t0", [ColumnSpec("probe_id", "int")], [{"probe_id": 0}])],
        Program(
            "prog-rolling-mean-by-null-count-probe-alias",
            81,
            [{"op": "rolling_mean_by_null_count_probe", "as": "where"}],
        ),
    )

    assert validate_case_program(valid) == []
    assert any("reserved" in error for error in validate_case_program(bad_alias))


def test_validate_case_accepts_large_string_partition_probe():
    valid = Case(
        "case-large-string-partition-probe",
        83,
        [TableData("t0", [ColumnSpec("probe_id", "int")], [{"probe_id": 0}])],
        Program(
            "prog-large-string-partition-probe",
            83,
            [{"op": "large_string_partition_probe", "as": "large_string_partition_mismatch"}],
        ),
    )
    bad_alias = Case(
        "case-large-string-partition-probe-alias",
        84,
        [TableData("t0", [ColumnSpec("probe_id", "int")], [{"probe_id": 0}])],
        Program(
            "prog-large-string-partition-probe-alias",
            84,
            [{"op": "large_string_partition_probe", "as": "from"}],
        ),
    )

    assert validate_case_program(valid) == []
    assert any("reserved" in error for error in validate_case_program(bad_alias))


def test_validate_case_accepts_hash_pivot_wider_probe():
    valid = Case(
        "case-hash-pivot-wider-probe",
        86,
        [TableData("t0", [ColumnSpec("probe_id", "int")], [{"probe_id": 0}])],
        Program(
            "prog-hash-pivot-wider-probe",
            86,
            [{"op": "hash_pivot_wider_probe", "as": "hash_pivot_wider_mismatch"}],
        ),
    )
    bad_alias = Case(
        "case-hash-pivot-wider-probe-alias",
        87,
        [TableData("t0", [ColumnSpec("probe_id", "int")], [{"probe_id": 0}])],
        Program(
            "prog-hash-pivot-wider-probe-alias",
            87,
            [{"op": "hash_pivot_wider_probe", "as": "select"}],
        ),
    )

    assert validate_case_program(valid) == []
    assert any("reserved" in error for error in validate_case_program(bad_alias))


def test_classification_reference_understands_large_string_partition_probe():
    case = Case(
        "case-large-string-partition-reference",
        85,
        [TableData("t0", [ColumnSpec("probe_id", "int")], [{"probe_id": 0}])],
        Program(
            "prog-large-string-partition-reference",
            85,
            [{"op": "large_string_partition_probe", "as": "large_string_partition_mismatch"}],
        ),
    )
    finding = {
        "kind": "semantic_output_mismatch",
        "root_cause": "pyarrow_large_string_partition_schema_semantics",
        "confidence": "high",
        "suspicious_backends": ["pyarrow"],
    }
    normalized = {
        "reference": NormalizedResult("reference", "ok", ["large_string_partition_mismatch"], [[False]]),
        "pyarrow": NormalizedResult("pyarrow", "ok", ["large_string_partition_mismatch"], [[True]]),
    }

    classification = classify_finding(
        case,
        finding,
        normalized,
        {},
        {"generator_profile": "pyarrow_large_string_partition_schema_semantics"},
        ["reference", "pyarrow"],
    )

    assert classification.verdict == "candidate_implementation_bug"
    assert classification.implicated_backends == ["pyarrow"]


def test_classification_reference_understands_hash_pivot_wider_probe():
    case = Case(
        "case-hash-pivot-wider-reference",
        88,
        [TableData("t0", [ColumnSpec("probe_id", "int")], [{"probe_id": 0}])],
        Program(
            "prog-hash-pivot-wider-reference",
            88,
            [{"op": "hash_pivot_wider_probe", "as": "hash_pivot_wider_mismatch"}],
        ),
    )
    finding = {
        "kind": "semantic_output_mismatch",
        "root_cause": "pyarrow_hash_pivot_wider_order_semantics",
        "confidence": "high",
        "suspicious_backends": ["pyarrow"],
    }
    normalized = {
        "reference": NormalizedResult("reference", "ok", ["hash_pivot_wider_mismatch"], [[False]]),
        "pyarrow": NormalizedResult("pyarrow", "ok", ["hash_pivot_wider_mismatch"], [[True]]),
    }

    classification = classify_finding(
        case,
        finding,
        normalized,
        {},
        {"generator_profile": "pyarrow_hash_pivot_wider_order_semantics"},
        ["reference", "pyarrow"],
    )

    assert classification.verdict == "candidate_implementation_bug"
    assert classification.implicated_backends == ["pyarrow"]


def test_classification_reference_understands_rolling_mean_by_null_count_probe():
    case = Case(
        "case-rolling-mean-by-null-count-reference",
        82,
        [TableData("t0", [ColumnSpec("probe_id", "int")], [{"probe_id": 0}])],
        Program(
            "prog-rolling-mean-by-null-count-reference",
            82,
            [{"op": "rolling_mean_by_null_count_probe", "as": "rolling_mean_by_null_count_mismatch"}],
        ),
    )
    finding = {
        "kind": "semantic_output_mismatch",
        "root_cause": "polars_rolling_mean_by_null_count_semantics",
        "confidence": "high",
        "suspicious_backends": ["polars", "polars_lazy"],
    }
    normalized = {
        "reference": NormalizedResult("reference", "ok", ["rolling_mean_by_null_count_mismatch"], [[False]]),
        "polars": NormalizedResult("polars", "ok", ["rolling_mean_by_null_count_mismatch"], [[True]]),
        "polars_lazy": NormalizedResult("polars_lazy", "ok", ["rolling_mean_by_null_count_mismatch"], [[True]]),
    }

    classification = classify_finding(
        case,
        finding,
        normalized,
        {},
        {"generator_profile": "polars_rolling_mean_by_null_count_semantics"},
        ["reference", "polars", "polars_lazy"],
    )

    assert classification.verdict == "candidate_implementation_bug"
    assert classification.implicated_backends == ["polars", "polars_lazy"]


def test_validate_case_accepts_explicit_null_predicate_filter():
    valid = Case(
        "case-null-predicate",
        8,
        [TableData("t0", [ColumnSpec("s", "str")], [{"s": "alpha"}, {"s": None}])],
        Program("prog-null-predicate", 8, [{"op": "filter", "column": "s", "cmp": "is_null", "value": None}]),
    )
    invalid_literal = Case(
        "case-null-predicate-literal",
        9,
        [TableData("t0", [ColumnSpec("s", "str")], [{"s": "alpha"}])],
        Program("prog-null-predicate-literal", 9, [{"op": "filter", "column": "s", "cmp": "is_not_null", "value": "alpha"}]),
    )

    assert validate_case_program(valid) == []
    assert any("must be NULL for is_not_null" in error for error in validate_case_program(invalid_literal))


def test_validate_case_accepts_boolean_predicate_filter():
    valid = Case(
        "case-bool-predicate",
        10,
        [TableData("t0", [ColumnSpec("flag", "bool")], [{"flag": True}, {"flag": None}])],
        Program("prog-bool-predicate", 10, [{"op": "filter", "column": "flag", "cmp": "bool_is_not_true", "value": None}]),
    )
    invalid_type = Case(
        "case-bool-predicate-type",
        11,
        [TableData("t0", [ColumnSpec("x", "int")], [{"x": 1}])],
        Program("prog-bool-predicate-type", 11, [{"op": "filter", "column": "x", "cmp": "bool_is_true", "value": None}]),
    )
    invalid_literal = Case(
        "case-bool-predicate-literal",
        12,
        [TableData("t0", [ColumnSpec("flag", "bool")], [{"flag": True}])],
        Program("prog-bool-predicate-literal", 12, [{"op": "filter", "column": "flag", "cmp": "bool_is_false", "value": False}]),
    )

    assert validate_case_program(valid) == []
    assert any("not supported for int filter" in error for error in validate_case_program(invalid_type))
    assert any("must be NULL for bool_is_false" in error for error in validate_case_program(invalid_literal))


def test_validate_case_rejects_duplicate_select_columns():
    case = Case(
        "case-dup-select",
        5,
        [TableData("t0", [ColumnSpec("x", "int"), ColumnSpec("g", "str")], [{"x": 1, "g": "Alpha"}])],
        Program("prog-dup-select", 5, [{"op": "select", "columns": ["x", "x", "g"]}]),
    )

    errors = validate_case_program(case)

    assert any("select contains duplicate columns" in error for error in errors)


def test_validate_case_allows_post_groupby_join_and_global_aggregate():
    case = Case(
        "case-post-groupby-join-aggregate",
        6,
        [
            TableData("t0", [ColumnSpec("id", "int"), ColumnSpec("x", "int")], [{"id": 1, "x": 2}]),
            TableData("t1", [ColumnSpec("id", "int"), ColumnSpec("z", "int")], [{"id": 1, "z": 3}]),
        ],
        Program(
            "prog-post-groupby-join-aggregate",
            6,
            [
                {"op": "groupby", "keys": ["id"], "aggs": [{"column": "x", "func": "sum", "as": "sum_x"}]},
                {"op": "join", "table": "t1", "left_on": "id", "right_on": "id", "how": "inner"},
                {"op": "aggregate", "aggs": [{"column": "sum_x", "func": "sum", "as": "total_x"}]},
            ],
        ),
    )

    assert validate_case_program(case) == []


def test_validate_case_rejects_groupby_alias_that_collides_with_key():
    case = Case(
        "case-groupby-alias-collision",
        7,
        [TableData("t0", [ColumnSpec("g", "str"), ColumnSpec("x", "int")], [{"g": "a", "x": 1}])],
        Program(
            "prog-groupby-alias-collision",
            7,
            [{"op": "groupby", "keys": ["g"], "aggs": [{"column": "x", "func": "max", "as": "g"}]}],
        ),
    )

    errors = validate_case_program(case)

    assert any("aliases collide with keys" in error for error in errors)


def test_validate_case_allows_count_and_nunique_on_string_columns_only():
    count_string = Case(
        "case-count-string",
        8,
        [
            TableData(
                "t0",
                [ColumnSpec("g", "str"), ColumnSpec("s", "str"), ColumnSpec("x", "int")],
                [{"g": "a", "s": "alpha", "x": 1}, {"g": "a", "s": None, "x": 2}],
            )
        ],
        Program(
            "prog-count-string",
            8,
            [{"op": "groupby", "keys": ["g"], "aggs": [{"column": "s", "func": "count", "as": "count_s"}]}],
        ),
    )
    nunique_string = Case(
        "case-nunique-string",
        9,
        [
            TableData(
                "t0",
                [ColumnSpec("g", "str"), ColumnSpec("s", "str")],
                [{"g": "a", "s": "alpha"}, {"g": "a", "s": "alpha"}, {"g": "a", "s": None}],
            )
        ],
        Program(
            "prog-nunique-string",
            9,
            [{"op": "groupby", "keys": ["g"], "aggs": [{"column": "s", "func": "nunique", "as": "uniq_s"}]}],
        ),
    )
    sum_string = Case(
        "case-sum-string",
        10,
        [
            TableData(
                "t0",
                [ColumnSpec("g", "str"), ColumnSpec("s", "str")],
                [{"g": "a", "s": "alpha"}],
            )
        ],
        Program(
            "prog-sum-string",
            9,
            [{"op": "groupby", "keys": ["g"], "aggs": [{"column": "s", "func": "sum", "as": "sum_s"}]}],
        ),
    )

    assert validate_case_program(count_string) == []
    assert validate_case_program(nunique_string) == []
    assert any("not numeric" in error for error in validate_case_program(sum_string))


def test_validate_case_rejects_reserved_output_aliases():
    groupby_case = Case(
        "case-groupby-reserved-alias",
        8,
        [TableData("t0", [ColumnSpec("g", "str"), ColumnSpec("x", "int")], [{"g": "a", "x": 1}])],
        Program(
            "prog-groupby-reserved-alias",
            8,
            [{"op": "groupby", "keys": ["g"], "aggs": [{"column": "x", "func": "max", "as": "select"}]}],
        ),
    )
    aggregate_case = Case(
        "case-aggregate-reserved-alias",
        9,
        [TableData("t0", [ColumnSpec("x", "int")], [{"x": 1}])],
        Program(
            "prog-aggregate-reserved-alias",
            9,
            [{"op": "aggregate", "aggs": [{"column": "x", "func": "max", "as": "where"}]}],
        ),
    )
    mutate_case = Case(
        "case-mutate-reserved-column",
        10,
        [TableData("t0", [ColumnSpec("x", "int")], [{"x": 1}])],
        Program(
            "prog-mutate-reserved-column",
            10,
            [{"op": "mutate", "column": "__datadiff_tmp", "expr": {"kind": "add_const", "source": "x", "value": 1}}],
        ),
    )

    assert any("reserved names" in error for error in validate_case_program(groupby_case))
    assert any("reserved names" in error for error in validate_case_program(aggregate_case))
    assert any("mutate output column" in error and "reserved" in error for error in validate_case_program(mutate_case))
