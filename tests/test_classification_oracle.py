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
