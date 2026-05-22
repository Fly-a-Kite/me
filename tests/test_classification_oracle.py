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
