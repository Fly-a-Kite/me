from datadiff.classification_oracle import annotate_findings, classify_finding, validate_case_program
from datadiff.datagen import generate_case
from datadiff.dsl import Case, ColumnSpec, Program, TableData
from datadiff.normalizer import NormalizedResult
from datadiff.oracle import Finding


def _case(ops):
    return Case(
        "case-x",
        1,
        [
            TableData(
                "t0",
                [ColumnSpec("x", "int"), ColumnSpec("s", "str"), ColumnSpec("flag", "bool")],
                [{"x": 1, "s": "Alpha", "flag": True}],
            )
        ],
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


def test_validate_case_program_accepts_and_rejects_string_contains_filter():
    valid = _case([{"op": "filter", "column": "s", "cmp": "str_contains", "value": "Al"}])
    valid_prefix = _case([{"op": "filter", "column": "s", "cmp": "str_starts_with", "value": "A"}])
    valid_suffix = _case([{"op": "filter", "column": "s", "cmp": "str_ends_with", "value": "a"}])
    wrong_type = _case([{"op": "filter", "column": "x", "cmp": "str_contains", "value": "1"}])
    empty_literal = _case([{"op": "filter", "column": "s", "cmp": "str_contains", "value": ""}])
    null_literal = _case([{"op": "filter", "column": "s", "cmp": "str_contains", "value": None}])

    assert validate_case_program(valid) == []
    assert validate_case_program(valid_prefix) == []
    assert validate_case_program(valid_suffix) == []
    assert "not supported for int filter" in validate_case_program(wrong_type)[0]
    assert "non-empty string for str_contains" in validate_case_program(empty_literal)[0]
    assert "non-empty string for str_contains" in validate_case_program(null_literal)[0]


def test_validate_case_program_accepts_and_rejects_numeric_clip_mutate():
    valid = _case([{"op": "mutate", "column": "x_clip", "expr": {"kind": "clip", "source": "x", "lower": -2, "upper": 2}}])
    missing = _case([{"op": "mutate", "column": "bad", "expr": {"kind": "clip", "source": "missing", "lower": -2, "upper": 2}}])
    reversed_bounds = _case([{"op": "mutate", "column": "bad", "expr": {"kind": "clip", "source": "x", "lower": 2, "upper": -2}}])
    string_source = _case([{"op": "mutate", "column": "bad", "expr": {"kind": "clip", "source": "s", "lower": -2, "upper": 2}}])

    assert validate_case_program(valid) == []
    assert "invalid mutate expression" in validate_case_program(missing)[0]
    assert "invalid mutate expression" in validate_case_program(reversed_bounds)[0]
    assert "invalid mutate expression" in validate_case_program(string_source)[0]


def test_validate_case_program_accepts_and_rejects_numeric_abs_mutate():
    valid = _case([{"op": "mutate", "column": "x_abs", "expr": {"kind": "abs", "source": "x"}}])
    missing = _case([{"op": "mutate", "column": "bad", "expr": {"kind": "abs", "source": "missing"}}])
    string_source = _case([{"op": "mutate", "column": "bad", "expr": {"kind": "abs", "source": "s"}}])

    assert validate_case_program(valid) == []
    assert "invalid mutate expression" in validate_case_program(missing)[0]
    assert "invalid mutate expression" in validate_case_program(string_source)[0]


def test_validate_case_program_accepts_and_rejects_string_strip_mutate():
    valid = _case([{"op": "mutate", "column": "s_clean", "expr": {"kind": "string_strip", "source": "s"}}])
    missing = _case([{"op": "mutate", "column": "bad", "expr": {"kind": "string_strip", "source": "missing"}}])
    numeric_source = _case([{"op": "mutate", "column": "bad", "expr": {"kind": "string_strip", "source": "x"}}])

    assert validate_case_program(valid) == []
    assert "invalid mutate expression" in validate_case_program(missing)[0]
    assert "invalid mutate expression" in validate_case_program(numeric_source)[0]


def test_validate_case_program_accepts_and_rejects_string_replace_mutate():
    valid = _case(
        [
            {
                "op": "mutate",
                "column": "s_token",
                "expr": {"kind": "string_replace", "source": "s", "old": " ", "new": "_"},
            }
        ]
    )
    missing = _case(
        [
            {
                "op": "mutate",
                "column": "bad",
                "expr": {"kind": "string_replace", "source": "missing", "old": " ", "new": "_"},
            }
        ]
    )
    numeric_source = _case(
        [
            {
                "op": "mutate",
                "column": "bad",
                "expr": {"kind": "string_replace", "source": "x", "old": " ", "new": "_"},
            }
        ]
    )
    empty_pattern = _case(
        [
            {
                "op": "mutate",
                "column": "bad",
                "expr": {"kind": "string_replace", "source": "s", "old": "", "new": "_"},
            }
        ]
    )

    assert validate_case_program(valid) == []
    assert "invalid mutate expression" in validate_case_program(missing)[0]
    assert "invalid mutate expression" in validate_case_program(numeric_source)[0]
    assert "invalid mutate expression" in validate_case_program(empty_pattern)[0]


def test_validate_case_program_accepts_and_rejects_string_slice_mutate():
    valid = _case(
        [
            {
                "op": "mutate",
                "column": "s_prefix",
                "expr": {"kind": "string_slice", "source": "s", "start": 0, "length": 3},
            }
        ]
    )
    missing = _case(
        [
            {
                "op": "mutate",
                "column": "bad",
                "expr": {"kind": "string_slice", "source": "missing", "start": 0, "length": 3},
            }
        ]
    )
    numeric_source = _case(
        [
            {
                "op": "mutate",
                "column": "bad",
                "expr": {"kind": "string_slice", "source": "x", "start": 0, "length": 3},
            }
        ]
    )
    negative_start = _case(
        [
            {
                "op": "mutate",
                "column": "bad",
                "expr": {"kind": "string_slice", "source": "s", "start": -1, "length": 3},
            }
        ]
    )

    assert validate_case_program(valid) == []
    assert "invalid mutate expression" in validate_case_program(missing)[0]
    assert "invalid mutate expression" in validate_case_program(numeric_source)[0]
    assert "invalid mutate expression" in validate_case_program(negative_start)[0]


def test_validate_case_program_accepts_and_rejects_string_null_if_empty_mutate():
    valid = _case(
        [
            {
                "op": "mutate",
                "column": "s_norm",
                "expr": {"kind": "string_null_if_empty", "source": "s"},
            }
        ]
    )
    missing = _case(
        [
            {
                "op": "mutate",
                "column": "bad",
                "expr": {"kind": "string_null_if_empty", "source": "missing"},
            }
        ]
    )
    numeric_source = _case(
        [
            {
                "op": "mutate",
                "column": "bad",
                "expr": {"kind": "string_null_if_empty", "source": "x"},
            }
        ]
    )

    assert validate_case_program(valid) == []
    assert "invalid mutate expression" in validate_case_program(missing)[0]
    assert "invalid mutate expression" in validate_case_program(numeric_source)[0]


def test_validate_case_program_accepts_and_rejects_string_split_part_mutate():
    valid = _case(
        [
            {
                "op": "mutate",
                "column": "s_token",
                "expr": {"kind": "string_split_part", "source": "s", "sep": " ", "index": 0},
            }
        ]
    )
    numeric_source = _case(
        [
            {
                "op": "mutate",
                "column": "bad",
                "expr": {"kind": "string_split_part", "source": "x", "sep": " ", "index": 0},
            }
        ]
    )
    empty_separator = _case(
        [
            {
                "op": "mutate",
                "column": "bad",
                "expr": {"kind": "string_split_part", "source": "s", "sep": "", "index": 0},
            }
        ]
    )
    later_token = _case(
        [
            {
                "op": "mutate",
                "column": "bad",
                "expr": {"kind": "string_split_part", "source": "s", "sep": " ", "index": 1},
            }
        ]
    )

    assert validate_case_program(valid) == []
    assert "invalid mutate expression" in validate_case_program(numeric_source)[0]
    assert "invalid mutate expression" in validate_case_program(empty_separator)[0]
    assert "invalid mutate expression" in validate_case_program(later_token)[0]


def test_validate_case_program_accepts_and_rejects_date_part_mutate():
    valid = _case(
        [
            {
                "op": "mutate",
                "column": "year",
                "expr": {"kind": "date_part", "source": "s", "part": "year"},
            }
        ]
    )
    numeric_source = _case(
        [
            {
                "op": "mutate",
                "column": "bad",
                "expr": {"kind": "date_part", "source": "x", "part": "year"},
            }
        ]
    )
    bad_part = _case(
        [
            {
                "op": "mutate",
                "column": "bad",
                "expr": {"kind": "date_part", "source": "s", "part": "hour"},
            }
        ]
    )

    assert validate_case_program(valid) == []
    assert "invalid mutate expression" in validate_case_program(numeric_source)[0]
    assert "invalid mutate expression" in validate_case_program(bad_part)[0]


def test_validate_case_program_accepts_cast_boundary_mutate():
    numeric_text_case = Case(
        "case-cast-boundary",
        1,
        [
            TableData(
                "t0",
                [ColumnSpec("x", "int"), ColumnSpec("num_s", "str"), ColumnSpec("label", "str")],
                [{"x": 1, "num_s": "10", "label": "alpha"}],
            )
        ],
        Program(
            "prog-cast-boundary",
            1,
            [
                {"op": "mutate", "column": "num_i", "expr": {"kind": "cast", "source": "num_s", "to": "int", "input_domain": "integer_string"}},
                {"op": "mutate", "column": "num_f", "expr": {"kind": "cast", "source": "num_i", "to": "float"}},
                {"op": "mutate", "column": "num_label", "expr": {"kind": "cast", "source": "num_i", "to": "str"}},
            ],
        ),
    )
    missing_domain = _case([{"op": "mutate", "column": "bad", "expr": {"kind": "cast", "source": "s", "to": "int"}}])
    bad_target = _case([{"op": "mutate", "column": "bad", "expr": {"kind": "cast", "source": "x", "to": "bool"}}])
    bool_to_str = _case([{"op": "mutate", "column": "bad", "expr": {"kind": "cast", "source": "flag", "to": "str"}}])

    assert validate_case_program(numeric_text_case) == []
    assert "invalid mutate expression" in validate_case_program(missing_domain)[0]
    assert "invalid mutate expression" in validate_case_program(bad_target)[0]
    assert "invalid mutate expression" in validate_case_program(bool_to_str)[0]


def test_validate_case_program_accepts_and_rejects_string_concat_mutate():
    valid = _case(
        [
            {
                "op": "mutate",
                "column": "label",
                "expr": {"kind": "string_concat", "source": "s", "other": "s", "sep": "-"},
            }
        ]
    )
    missing = _case(
        [
            {
                "op": "mutate",
                "column": "bad",
                "expr": {"kind": "string_concat", "source": "s", "other": "missing", "sep": "-"},
            }
        ]
    )
    numeric_other = _case(
        [
            {
                "op": "mutate",
                "column": "bad",
                "expr": {"kind": "string_concat", "source": "s", "other": "x", "sep": "-"},
            }
        ]
    )
    non_string_separator = _case(
        [
            {
                "op": "mutate",
                "column": "bad",
                "expr": {"kind": "string_concat", "source": "s", "other": "s", "sep": 1},
            }
        ]
    )

    assert validate_case_program(valid) == []
    assert "invalid mutate expression" in validate_case_program(missing)[0]
    assert "invalid mutate expression" in validate_case_program(numeric_other)[0]
    assert "invalid mutate expression" in validate_case_program(non_string_separator)[0]


def test_validate_case_program_accepts_and_rejects_string_contains_mutate():
    valid = _case(
        [
            {
                "op": "mutate",
                "column": "has_a",
                "expr": {"kind": "string_contains", "source": "s", "needle": "a"},
            }
        ]
    )
    missing = _case(
        [
            {
                "op": "mutate",
                "column": "bad",
                "expr": {"kind": "string_contains", "source": "missing", "needle": "a"},
            }
        ]
    )
    numeric_source = _case(
        [
            {
                "op": "mutate",
                "column": "bad",
                "expr": {"kind": "string_contains", "source": "x", "needle": "a"},
            }
        ]
    )
    non_string_needle = _case(
        [
            {
                "op": "mutate",
                "column": "bad",
                "expr": {"kind": "string_contains", "source": "s", "needle": 1},
            }
        ]
    )

    assert validate_case_program(valid) == []
    assert "invalid mutate expression" in validate_case_program(missing)[0]
    assert "invalid mutate expression" in validate_case_program(numeric_source)[0]
    assert "invalid mutate expression" in validate_case_program(non_string_needle)[0]


def test_validate_case_program_accepts_string_starts_and_ends_with_mutate():
    starts = _case(
        [
            {
                "op": "mutate",
                "column": "starts_a",
                "expr": {"kind": "string_starts_with", "source": "s", "needle": "A"},
            }
        ]
    )
    ends = _case(
        [
            {
                "op": "mutate",
                "column": "ends_a",
                "expr": {"kind": "string_ends_with", "source": "s", "needle": "a"},
            }
        ]
    )
    numeric_source = _case(
        [
            {
                "op": "mutate",
                "column": "bad",
                "expr": {"kind": "string_starts_with", "source": "x", "needle": "A"},
            }
        ]
    )
    non_string_needle = _case(
        [
            {
                "op": "mutate",
                "column": "bad",
                "expr": {"kind": "string_ends_with", "source": "s", "needle": 1},
            }
        ]
    )

    assert validate_case_program(starts) == []
    assert validate_case_program(ends) == []
    assert "invalid mutate expression" in validate_case_program(numeric_source)[0]
    assert "invalid mutate expression" in validate_case_program(non_string_needle)[0]


def test_validate_case_program_accepts_and_rejects_bool_not_mutate():
    valid = _case([{"op": "mutate", "column": "not_flag", "expr": {"kind": "bool_not", "source": "flag"}}])
    missing = _case([{"op": "mutate", "column": "bad", "expr": {"kind": "bool_not", "source": "missing"}}])
    numeric_source = _case([{"op": "mutate", "column": "bad", "expr": {"kind": "bool_not", "source": "x"}}])

    assert validate_case_program(valid) == []
    assert "invalid mutate expression" in validate_case_program(missing)[0]
    assert "invalid mutate expression" in validate_case_program(numeric_source)[0]


def test_annotate_findings_marks_issue_replay_origin():
    case = Case(
        "case-replay",
        1,
        [TableData("t0", [ColumnSpec("probe_id", "int", nullable=False)], [{"probe_id": 0}])],
        Program("prog-replay", 1, [{"op": "group_quantile_probe", "as": "quantile_key_mismatch"}]),
        metadata={"source_issue": "https://github.com/example/project/issues/1"},
    )
    finding = Finding(
        finding_id="finding-1",
        kind="semantic_output_mismatch",
        severity="critical",
        suspicious_backends=["polars"],
        evidence="mismatch",
        signature="sig-1",
        root_cause="group_quantile_key_expression",
    )

    annotate_findings(case, [finding], {}, {}, {"generator_profile": "bughunt"}, ["pandas", "polars"])

    assert finding.discovery_origin == "issue_replay"
    assert finding.source_issue == "https://github.com/example/project/issues/1"


def test_annotate_findings_marks_structural_issue_inspired_origin():
    case = Case(
        "case-inspired",
        1,
        [TableData("t0", [ColumnSpec("x", "int")], [{"x": 1}])],
        Program("prog-inspired", 1, [{"op": "filter", "column": "x", "cmp": ">", "value": 0}]),
        metadata={"source_issue": "https://github.com/example/project/issues/2"},
    )
    finding = Finding(
        finding_id="finding-2",
        kind="semantic_output_mismatch",
        severity="critical",
        suspicious_backends=["duckdb"],
        evidence="mismatch",
        signature="sig-2",
        root_cause="filter_predicate",
    )

    annotate_findings(case, [finding], {}, {}, {"generator_profile": "bughunt"}, ["pandas", "duckdb"])

    assert finding.discovery_origin == "issue_inspired"
    assert finding.source_issue == "https://github.com/example/project/issues/2"


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


def test_validate_case_program_accepts_and_rejects_distinct_columns():
    valid = _case([{"op": "distinct", "columns": ["x", "s"]}])
    missing = _case([{"op": "distinct", "columns": ["x", "missing"]}])
    duplicate = _case([{"op": "distinct", "columns": ["x", "x"]}])
    empty = _case([{"op": "distinct", "columns": []}])

    assert validate_case_program(valid) == []
    assert "distinct columns unavailable" in validate_case_program(missing)[0]
    assert "distinct contains duplicate columns" in validate_case_program(duplicate)[0]
    assert "distinct has no columns" in validate_case_program(empty)[0]


def test_validate_case_program_accepts_and_rejects_fill_null():
    valid = _case([{"op": "fill_null", "column": "x", "value": 0}])
    missing = _case([{"op": "fill_null", "column": "missing", "value": 0}])
    null_value = _case([{"op": "fill_null", "column": "x", "value": None}])
    incompatible = _case([{"op": "fill_null", "column": "x", "value": "zero"}])

    assert validate_case_program(valid) == []
    assert "fill_null column unavailable" in validate_case_program(missing)[0]
    assert "fill_null value must not be NULL" in validate_case_program(null_value)[0]
    assert "fill_null literal" in validate_case_program(incompatible)[0]


def test_validate_case_program_accepts_and_rejects_coalesce():
    valid = Case(
        "case-coalesce-valid",
        1,
        [
            TableData(
                "t0",
                [ColumnSpec("g", "str"), ColumnSpec("s", "str"), ColumnSpec("x", "int")],
                [{"g": None, "s": "fallback", "x": 1}],
            )
        ],
        Program("prog-coalesce-valid", 1, [{"op": "coalesce", "columns": ["g", "s"], "as": "label", "fallback": "missing"}]),
    )
    missing = _case([{"op": "coalesce", "columns": ["s", "missing"], "as": "label"}])
    duplicate = _case([{"op": "coalesce", "columns": ["s", "s"], "as": "label"}])
    narrow = _case([{"op": "coalesce", "columns": ["s"], "as": "label"}])
    reserved = _case([{"op": "coalesce", "columns": ["x", "s"], "as": "select"}])
    type_mismatch = _case([{"op": "coalesce", "columns": ["x", "s"], "as": "label"}])
    bad_fallback = Case(
        "case-coalesce-bad-fallback",
        1,
        [TableData("t0", [ColumnSpec("x", "int"), ColumnSpec("y", "int")], [{"x": None, "y": 1}])],
        Program("prog-coalesce-bad-fallback", 1, [{"op": "coalesce", "columns": ["x", "y"], "as": "label", "fallback": "zero"}]),
    )

    assert validate_case_program(valid) == []
    assert "coalesce columns unavailable" in validate_case_program(missing)[0]
    assert "coalesce contains duplicate columns" in validate_case_program(duplicate)[0]
    assert "coalesce needs at least two columns" in validate_case_program(narrow)[0]
    assert "coalesce output alias" in validate_case_program(reserved)[0]
    assert "coalesce column type mismatch" in validate_case_program(type_mismatch)[0]
    assert "coalesce fallback literal" in validate_case_program(bad_fallback)[0]


def test_validate_case_program_accepts_and_rejects_case_when():
    valid = _case(
        [
            {
                "op": "case_when",
                "as": "label",
                "condition": {"column": "x", "cmp": ">=", "value": 0},
                "then": "yes",
                "else": "no",
            }
        ]
    )
    missing = _case(
        [
            {
                "op": "case_when",
                "as": "label",
                "condition": {"column": "missing", "cmp": ">=", "value": 0},
                "then": "yes",
                "else": "no",
            }
        ]
    )
    reserved = _case(
        [
            {
                "op": "case_when",
                "as": "select",
                "condition": {"column": "x", "cmp": ">=", "value": 0},
                "then": "yes",
                "else": "no",
            }
        ]
    )
    bad_branch_types = _case(
        [
            {
                "op": "case_when",
                "as": "label",
                "condition": {"column": "x", "cmp": ">=", "value": 0},
                "then": "yes",
                "else": 0,
            }
        ]
    )

    assert validate_case_program(valid) == []
    assert "case_when condition column unavailable" in validate_case_program(missing)[0]
    assert "case_when output alias" in validate_case_program(reserved)[0]
    assert "case_when branch literals are incompatible" in validate_case_program(bad_branch_types)[0]


def test_validate_case_program_accepts_and_rejects_union_all():
    valid = Case(
        "case-union-valid",
        1,
        [
            TableData("t0", [ColumnSpec("x", "int"), ColumnSpec("s", "str")], [{"x": 1, "s": "a"}]),
            TableData("t_append", [ColumnSpec("x", "int"), ColumnSpec("s", "str")], [{"x": 2, "s": "b"}]),
        ],
        Program("prog-union-valid", 1, [{"op": "union_all", "table": "t_append"}]),
    )
    missing_table = _case([{"op": "union_all", "table": "missing"}])
    missing_column = Case(
        "case-union-missing-column",
        1,
        [
            TableData("t0", [ColumnSpec("x", "int"), ColumnSpec("s", "str")], [{"x": 1, "s": "a"}]),
            TableData("t_append", [ColumnSpec("x", "int")], [{"x": 2}]),
        ],
        Program("prog-union-missing-column", 1, [{"op": "union_all", "table": "t_append"}]),
    )
    type_mismatch = Case(
        "case-union-type-mismatch",
        1,
        [
            TableData("t0", [ColumnSpec("x", "int"), ColumnSpec("s", "str")], [{"x": 1, "s": "a"}]),
            TableData("t_append", [ColumnSpec("x", "float"), ColumnSpec("s", "str")], [{"x": 2.0, "s": "b"}]),
        ],
        Program("prog-union-type-mismatch", 1, [{"op": "union_all", "table": "t_append"}]),
    )

    assert validate_case_program(valid) == []
    assert "unknown union_all table" in validate_case_program(missing_table)[0]
    assert "union_all columns unavailable" in validate_case_program(missing_column)[0]
    assert "union_all column type mismatch" in validate_case_program(type_mismatch)[0]


def test_validate_case_program_accepts_and_rejects_drop_nulls_columns():
    valid = _case([{"op": "drop_nulls", "columns": ["x", "s"]}])
    missing = _case([{"op": "drop_nulls", "columns": ["x", "missing"]}])
    duplicate = _case([{"op": "drop_nulls", "columns": ["x", "x"]}])
    empty = _case([{"op": "drop_nulls", "columns": []}])

    assert validate_case_program(valid) == []
    assert "drop_nulls columns unavailable" in validate_case_program(missing)[0]
    assert "drop_nulls contains duplicate columns" in validate_case_program(duplicate)[0]
    assert "drop_nulls has no columns" in validate_case_program(empty)[0]


def test_validate_case_program_accepts_and_rejects_semi_and_anti_join_keys():
    valid = Case(
        "case-semi-valid",
        1,
        [
            TableData("t0", [ColumnSpec("id", "int"), ColumnSpec("s", "str")], [{"id": 1, "s": "a"}]),
            TableData("t_lookup", [ColumnSpec("id", "int"), ColumnSpec("tag", "str")], [{"id": 1, "tag": "keep"}]),
        ],
        Program("prog-semi-valid", 1, [{"op": "semi_join", "table": "t_lookup", "left_on": "id", "right_on": "id"}]),
    )
    missing_table = _case([{"op": "anti_join", "table": "missing", "left_on": "x", "right_on": "x"}])
    missing_left = Case(
        "case-semi-missing-left",
        1,
        [
            TableData("t0", [ColumnSpec("id", "int")], [{"id": 1}]),
            TableData("t_lookup", [ColumnSpec("id", "int")], [{"id": 1}]),
        ],
        Program("prog-semi-missing-left", 1, [{"op": "semi_join", "table": "t_lookup", "left_on": "missing", "right_on": "id"}]),
    )
    missing_right = Case(
        "case-anti-missing-right",
        1,
        [
            TableData("t0", [ColumnSpec("id", "int")], [{"id": 1}]),
            TableData("t_lookup", [ColumnSpec("other", "int")], [{"other": 1}]),
        ],
        Program("prog-anti-missing-right", 1, [{"op": "anti_join", "table": "t_lookup", "left_on": "id", "right_on": "id"}]),
    )
    type_mismatch = Case(
        "case-semi-type-mismatch",
        1,
        [
            TableData("t0", [ColumnSpec("id", "int")], [{"id": 1}]),
            TableData("t_lookup", [ColumnSpec("id", "str")], [{"id": "1"}]),
        ],
        Program("prog-semi-type-mismatch", 1, [{"op": "semi_join", "table": "t_lookup", "left_on": "id", "right_on": "id"}]),
    )

    assert validate_case_program(valid) == []
    assert "unknown anti_join table" in validate_case_program(missing_table)[0]
    assert "semi_join left key" in validate_case_program(missing_left)[0]
    assert "anti_join right key" in validate_case_program(missing_right)[0]
    assert "semi_join key type mismatch" in validate_case_program(type_mismatch)[0]


def test_validate_case_program_accepts_and_rejects_offset():
    valid = _case([{"op": "sort", "columns": ["x"], "ascending": True}, {"op": "offset", "n": 1}])
    negative = _case([{"op": "offset", "n": -1}])
    invalid = _case([{"op": "offset", "n": "one"}])

    assert validate_case_program(valid) == []
    assert "negative offset" in validate_case_program(negative)[0]
    assert "non-integer offset" in validate_case_program(invalid)[0]


def test_validate_case_program_accepts_and_rejects_row_number_filter_and_basename_expr():
    valid = _case(
        [
            {"op": "mutate", "column": "base_s", "expr": {"kind": "string_basename", "source": "s"}},
            {
                "op": "row_number_filter",
                "partition_by": ["s"],
                "order_by": [{"column": "x", "ascending": True, "nulls": "last"}],
                "cmp": "==",
                "value": 1,
            },
        ]
    )
    missing_order = _case(
        [
            {
                "op": "row_number_filter",
                "partition_by": ["s"],
                "order_by": [{"column": "missing", "ascending": True, "nulls": "last"}],
                "cmp": "==",
                "value": 1,
            }
        ]
    )
    bad_expr = _case([{"op": "mutate", "column": "base_x", "expr": {"kind": "string_basename", "source": "x"}}])

    assert validate_case_program(valid) == []
    assert "row_number_filter order_by columns unavailable" in validate_case_program(missing_order)[0]
    assert "invalid mutate expression" in validate_case_program(bad_expr)[0]


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


def test_classification_reference_marks_row_count_minority_backend():
    case = Case(
        "case-reference-row-count",
        4101,
        [TableData("t0", [ColumnSpec("x", "int")], [{"x": 1}])],
        Program("prog-reference-row-count", 4101, []),
    )
    finding = {
        "kind": "semantic_output_mismatch",
        "root_cause": "unknown",
        "confidence": "high",
        "suspicious_backends": ["duckdb"],
    }
    normalized = {
        "pandas": NormalizedResult("pandas", "ok", ["x"], [[1]]),
        "duckdb": NormalizedResult("duckdb", "ok", ["x"], [[1], [1]]),
    }

    classification = classify_finding(case, finding, normalized, {}, {"generator_profile": "common"}, ["pandas", "duckdb"])

    assert classification.verdict == "candidate_implementation_bug"
    assert classification.implicated_backends == ["duckdb"]
    assert classification.confidence == "high"


def test_classification_reference_marks_schema_minority_backend():
    case = Case(
        "case-reference-schema",
        4102,
        [TableData("t0", [ColumnSpec("x", "int")], [{"x": 1}])],
        Program("prog-reference-schema", 4102, []),
    )
    finding = {
        "kind": "semantic_output_mismatch",
        "root_cause": "schema_projection",
        "confidence": "high",
        "suspicious_backends": ["duckdb"],
    }
    normalized = {
        "pandas": NormalizedResult("pandas", "ok", ["x"], [[1]]),
        "duckdb": NormalizedResult("duckdb", "ok", ["y"], [[1]]),
    }

    classification = classify_finding(case, finding, normalized, {}, {"generator_profile": "common"}, ["pandas", "duckdb"])

    assert classification.verdict == "candidate_implementation_bug"
    assert classification.implicated_backends == ["duckdb"]


def test_classification_excludes_order_sensitive_sort_tie_order_noise():
    case = Case(
        "case-sort-tie-order",
        42,
        [
            TableData(
                "t0",
                [ColumnSpec("x", "int"), ColumnSpec("s", "str")],
                [{"x": 1, "s": "a"}, {"x": 1, "s": "b"}],
            )
        ],
        Program(
            "prog-sort-tie-order",
            42,
            [{"op": "sort", "columns": ["x"], "ascending": True}, {"op": "select", "columns": ["s"]}],
        ),
    )
    finding = {
        "kind": "semantic_output_mismatch",
        "root_cause": "ordering_or_limit",
        "confidence": "medium",
        "suspicious_backends": ["duckdb"],
    }
    normalized = {
        "pandas": NormalizedResult("pandas", "ok", ["s"], [["a"], ["b"]]),
        "duckdb": NormalizedResult("duckdb", "ok", ["s"], [["b"], ["a"]]),
    }

    classification = classify_finding(case, finding, normalized, {}, {"generator_profile": "common"}, ["pandas", "duckdb"])

    assert classification.verdict == "normalizer_false_positive"
    assert classification.false_positive is True
    assert classification.false_positive_reason == "sort_tie_order_underconstrained"


def test_classification_excludes_sort_limit_tie_cutoff_noise():
    case = Case(
        "case-sort-limit-tie-cutoff",
        45,
        [
            TableData(
                "t0",
                [ColumnSpec("id", "int"), ColumnSpec("g", "str"), ColumnSpec("s", "str")],
                [
                    {"id": 0, "g": "", "s": "first"},
                    {"id": 0, "g": "", "s": "second"},
                    {"id": 2, "g": "", "s": "left"},
                    {"id": 2, "g": "", "s": "right"},
                ],
            )
        ],
        Program(
            "prog-sort-limit-tie-cutoff",
            45,
            [
                {
                    "op": "mutate",
                    "column": "g_token",
                    "expr": {"kind": "string_replace", "source": "g", "old": " ", "new": "_"},
                },
                {
                    "op": "sort",
                    "keys": [
                        {"column": "g_token", "ascending": True, "nulls": "last"},
                        {"column": "id", "ascending": True, "nulls": "last"},
                    ],
                },
                {"op": "select", "columns": ["id", "g_token", "s"]},
                {"op": "limit", "n": 3},
            ],
        ),
    )
    finding = {
        "kind": "semantic_output_mismatch",
        "root_cause": "string_expression",
        "confidence": "high",
        "suspicious_backends": ["duckdb"],
    }
    normalized = {
        "pandas": NormalizedResult(
            "pandas",
            "ok",
            ["g_token", "id", "s"],
            [["", 0, "first"], ["", 0, "second"], ["", 2, "left"]],
        ),
        "duckdb": NormalizedResult(
            "duckdb",
            "ok",
            ["g_token", "id", "s"],
            [["", 0, "first"], ["", 0, "second"], ["", 2, "right"]],
        ),
        "datafusion": NormalizedResult(
            "datafusion",
            "ok",
            ["g_token", "id", "s"],
            [["", 0, "first"], ["", 0, "second"], ["", 2, "left"]],
        ),
    }

    classification = classify_finding(
        case,
        finding,
        normalized,
        {},
        {"generator_profile": "common_api_workflow"},
        ["pandas", "duckdb", "datafusion"],
    )

    assert classification.verdict == "normalizer_false_positive"
    assert classification.false_positive is True
    assert classification.false_positive_reason == "sort_tie_order_underconstrained"


def test_classification_keeps_unique_sort_limit_mismatch_as_candidate_bug():
    case = Case(
        "case-unique-sort-limit",
        46,
        [
            TableData(
                "t0",
                [ColumnSpec("id", "int"), ColumnSpec("s", "str")],
                [{"id": 1, "s": "a"}, {"id": 2, "s": "b"}],
            )
        ],
        Program(
            "prog-unique-sort-limit",
            46,
            [
                {"op": "sort", "columns": ["id"], "ascending": True},
                {"op": "select", "columns": ["id", "s"]},
                {"op": "limit", "n": 1},
            ],
        ),
    )
    finding = {
        "kind": "semantic_output_mismatch",
        "root_cause": "ordering_or_limit",
        "confidence": "high",
        "suspicious_backends": ["duckdb"],
    }
    normalized = {
        "pandas": NormalizedResult("pandas", "ok", ["id", "s"], [[1, "a"]]),
        "duckdb": NormalizedResult("duckdb", "ok", ["id", "s"], [[2, "b"]]),
    }

    classification = classify_finding(case, finding, normalized, {}, {"generator_profile": "common"}, ["pandas", "duckdb"])

    assert classification.verdict == "candidate_implementation_bug"
    assert classification.false_positive is False


def test_classification_excludes_limit_before_any_defined_order():
    case = Case(
        "case-limit-before-order",
        43,
        [
            TableData(
                "t0",
                [ColumnSpec("x", "int"), ColumnSpec("s", "str")],
                [{"x": 1, "s": "a"}, {"x": 2, "s": "b"}],
            )
        ],
        Program(
            "prog-limit-before-order",
            43,
            [{"op": "limit", "n": 1}, {"op": "sort", "columns": ["x"], "ascending": True}],
        ),
    )
    finding = {
        "kind": "semantic_output_mismatch",
        "root_cause": "ordering_or_limit",
        "confidence": "high",
        "suspicious_backends": ["sqlite"],
    }
    normalized = {
        "pandas": NormalizedResult("pandas", "ok", ["x", "s"], [[1, "a"]]),
        "sqlite": NormalizedResult("sqlite", "ok", ["x", "s"], [[2, "b"]]),
    }

    classification = classify_finding(case, finding, normalized, {}, {"generator_profile": "common"}, ["pandas", "sqlite"])

    assert classification.verdict == "generator_false_positive"
    assert classification.false_positive is True
    assert classification.false_positive_reason == "limit_offset_order_underconstrained"


def test_classification_marks_float_precision_order_boundary_not_candidate_bug():
    case = Case(
        "case-float-precision-order-boundary",
        135030,
        [
            TableData(
                "t0",
                [ColumnSpec("id", "int"), ColumnSpec("flag", "bool"), ColumnSpec("x", "int")],
                [
                    {"id": 1, "flag": False, "x": 1},
                    {"id": 2, "flag": False, "x": 2},
                ],
            ),
            TableData(
                "t1",
                [ColumnSpec("id", "int"), ColumnSpec("tag", "str")],
                [{"id": 1, "tag": None}, {"id": 2, "tag": "space value"}],
            ),
        ],
        Program(
            "prog-float-precision-order-boundary",
            135030,
            [
                {"op": "join", "table": "t1", "left_on": "id", "right_on": "id", "how": "inner"},
                {"op": "mutate", "column": "m_0", "expr": {"kind": "arith_const", "op": "div", "source": "id", "value": 5}},
                {
                    "op": "groupby",
                    "keys": ["tag"],
                    "aggs": [
                        {"column": "flag", "func": "min", "as": "min_flag"},
                        {"column": "x", "func": "nunique", "as": "nunique_x"},
                        {"column": "m_0", "func": "sum", "as": "sum_m_0"},
                    ],
                },
                {"op": "sort", "columns": ["sum_m_0", "min_flag", "nunique_x", "tag"], "ascending": False},
                {"op": "limit", "n": 24},
            ],
        ),
    )
    finding = {
        "kind": "semantic_output_mismatch",
        "root_cause": "groupby_aggregation",
        "confidence": "high",
        "suspicious_backends": ["duckdb"],
    }
    normalized = {
        "duckdb": NormalizedResult(
            "duckdb",
            "ok",
            ["min_flag", "nunique_x", "sum_m_0", "tag"],
            [[False, 2, 1.6, "space value"], [False, 5, 1.5999999999999999, None]],
        ),
        "pandas": NormalizedResult(
            "pandas",
            "ok",
            ["min_flag", "nunique_x", "sum_m_0", "tag"],
            [[False, 5, 1.6, None], [False, 2, 1.5999999999999999, "space value"]],
        ),
    }

    classification = classify_finding(case, finding, normalized, {}, {"generator_profile": "bughunt"}, ["pandas", "duckdb"])

    assert classification.verdict == "expected_semantic_divergence"
    assert classification.paper_status == "valid_finding_not_bug"
    assert classification.false_positive is False


def test_classification_excludes_groupby_limit_without_order():
    case = Case(
        "case-groupby-limit-no-order",
        44,
        [
            TableData(
                "t0",
                [ColumnSpec("x", "int"), ColumnSpec("s", "str")],
                [{"x": 1, "s": "a"}, {"x": 2, "s": "b"}],
            )
        ],
        Program(
            "prog-groupby-limit-no-order",
            44,
            [
                {"op": "groupby", "keys": ["x"], "aggs": [{"column": "x", "func": "min", "as": "min_x"}]},
                {"op": "limit", "n": 1},
            ],
        ),
    )
    finding = {
        "kind": "semantic_output_mismatch",
        "root_cause": "groupby_aggregation",
        "confidence": "high",
        "suspicious_backends": ["duckdb"],
    }
    normalized = {
        "pandas": NormalizedResult("pandas", "ok", ["min_x", "x"], [[1, 1]]),
        "duckdb": NormalizedResult("duckdb", "ok", ["min_x", "x"], [[2, 2]]),
    }

    classification = classify_finding(case, finding, normalized, {}, {"generator_profile": "common"}, ["pandas", "duckdb"])

    assert classification.verdict == "generator_false_positive"
    assert classification.false_positive is True
    assert classification.false_positive_reason == "limit_offset_order_underconstrained"


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


def test_classification_marks_unicode_upper_as_semantic_boundary():
    case = Case(
        "case-upper",
        4,
        [TableData("t0", [ColumnSpec("s", "str")], [{"s": "δ"}])],
        Program("prog-upper", 4, [{"op": "mutate", "column": "m_0", "expr": {"kind": "string_upper", "source": "s"}}]),
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
    assert classification.confidence == "high"
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


def test_classification_reference_understands_csv_long_numeric_roundtrip_probe():
    case = generate_case(153, profile="csv_long_numeric_roundtrip")
    finding = {
        "kind": "semantic_output_mismatch",
        "root_cause": "csv_long_numeric_roundtrip",
        "confidence": "high",
        "suspicious_backends": ["duckdb"],
    }
    normalized = {
        "pandas": NormalizedResult("pandas", "ok", ["csv_long_numeric_roundtrip_mismatch"], [[False]]),
        "duckdb": NormalizedResult("duckdb", "ok", ["csv_long_numeric_roundtrip_mismatch"], [[True]]),
    }

    classification = classify_finding(
        case,
        finding,
        normalized,
        {},
        {"generator_profile": "csv_long_numeric_roundtrip"},
        ["pandas", "duckdb"],
    )

    assert validate_case_program(case) == []
    assert classification.verdict == "candidate_implementation_bug"
    assert classification.implicated_backends == ["duckdb"]
    assert "Independent DSL reference" in classification.evidence


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
        [
            TableData(
                "t0",
                [ColumnSpec("row_id", "int"), ColumnSpec("g", "str"), ColumnSpec("x", "float")],
                [{"row_id": 0, "g": "a", "x": 0.5}],
            )
        ],
        Program(
            "prog-running-sum",
            20,
            [
                {
                    "op": "running_sum",
                    "source": "x",
                    "column": "run_x",
                    "partition_by": ["g"],
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
    invalid_partition = Case(
        "case-running-sum-partition",
        23,
        [TableData("t0", [ColumnSpec("row_id", "int"), ColumnSpec("x", "float")], [{"row_id": 0, "x": 0.5}])],
        Program(
            "prog-running-sum-partition",
            23,
            [
                {
                    "op": "running_sum",
                    "source": "x",
                    "column": "run_x",
                    "partition_by": ["missing"],
                    "order_by": [{"column": "row_id", "ascending": True, "nulls": "last"}],
                }
            ],
        ),
    )

    assert validate_case_program(valid) == []
    assert any("not numeric" in error for error in validate_case_program(invalid_type))
    assert any("has no order_by" in error for error in validate_case_program(invalid_order))
    assert any("partition_by columns unavailable" in error for error in validate_case_program(invalid_partition))


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


def test_validate_case_accepts_setop_all_duplicate_probe():
    valid = Case(
        "case-setop-all-duplicate-probe",
        56,
        [TableData("t0", [ColumnSpec("probe_id", "int")], [{"probe_id": 0}])],
        Program(
            "prog-setop-all-duplicate-probe",
            56,
            [{"op": "setop_all_duplicate_probe", "as": "setop_all_duplicate_mismatch"}],
        ),
    )
    bad_alias = Case(
        "case-setop-all-duplicate-probe-alias",
        57,
        [TableData("t0", [ColumnSpec("probe_id", "int")], [{"probe_id": 0}])],
        Program(
            "prog-setop-all-duplicate-probe-alias",
            57,
            [{"op": "setop_all_duplicate_probe", "as": "where"}],
        ),
    )

    assert validate_case_program(valid) == []
    assert any("reserved" in error for error in validate_case_program(bad_alias))


def test_classification_reference_understands_setop_all_duplicate_probe():
    case = Case(
        "case-setop-all-duplicate-reference",
        58,
        [TableData("t0", [ColumnSpec("probe_id", "int")], [{"probe_id": 0}])],
        Program(
            "prog-setop-all-duplicate-reference",
            58,
            [{"op": "setop_all_duplicate_probe", "as": "setop_all_duplicate_mismatch"}],
        ),
    )
    finding = {
        "kind": "semantic_output_mismatch",
        "root_cause": "datafusion_setop_all_duplicate_count",
        "confidence": "high",
        "suspicious_backends": ["datafusion"],
    }
    normalized = {
        "reference": NormalizedResult("reference", "ok", ["setop_all_duplicate_mismatch"], [[False]]),
        "datafusion": NormalizedResult("datafusion", "ok", ["setop_all_duplicate_mismatch"], [[True]]),
    }

    classification = classify_finding(
        case,
        finding,
        normalized,
        {},
        {"generator_profile": "datafusion_setop_all_duplicate_count"},
        ["reference", "datafusion"],
    )

    assert classification.verdict == "candidate_implementation_bug"
    assert classification.implicated_backends == ["datafusion"]


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


def test_validate_case_accepts_run_end_null_compute_probe():
    valid = Case(
        "case-run-end-null-compute-probe",
        95,
        [TableData("t0", [ColumnSpec("probe_id", "int")], [{"probe_id": 0}])],
        Program(
            "prog-run-end-null-compute-probe",
            95,
            [{"op": "run_end_null_compute_probe", "as": "run_end_null_compute_mismatch"}],
        ),
    )
    bad_alias = Case(
        "case-run-end-null-compute-probe-alias",
        96,
        [TableData("t0", [ColumnSpec("probe_id", "int")], [{"probe_id": 0}])],
        Program("prog-run-end-null-compute-probe-alias", 96, [{"op": "run_end_null_compute_probe", "as": "where"}]),
    )

    assert validate_case_program(valid) == []
    assert any("reserved" in error for error in validate_case_program(bad_alias))


def test_classification_reference_understands_run_end_null_compute_probe():
    case = Case(
        "case-run-end-null-compute-reference",
        97,
        [TableData("t0", [ColumnSpec("probe_id", "int")], [{"probe_id": 0}])],
        Program(
            "prog-run-end-null-compute-reference",
            97,
            [{"op": "run_end_null_compute_probe", "as": "run_end_null_compute_mismatch"}],
        ),
    )
    finding = {
        "kind": "semantic_output_mismatch",
        "root_cause": "pyarrow_run_end_null_compute_semantics",
        "confidence": "high",
        "suspicious_backends": ["pyarrow"],
    }
    normalized = {
        "reference": NormalizedResult("reference", "ok", ["run_end_null_compute_mismatch"], [[False]]),
        "pyarrow": NormalizedResult("pyarrow", "ok", ["run_end_null_compute_mismatch"], [[True]]),
    }

    classification = classify_finding(
        case,
        finding,
        normalized,
        {},
        {"generator_profile": "pyarrow_run_end_null_compute_semantics"},
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


def test_validate_case_accepts_bool_reduction_skipna_probe():
    valid = Case(
        "case-bool-reduction-skipna-probe",
        92,
        [TableData("t0", [ColumnSpec("probe_id", "int")], [{"probe_id": 0}])],
        Program(
            "prog-bool-reduction-skipna-probe",
            92,
            [{"op": "bool_reduction_skipna_probe", "as": "bool_reduction_skipna_mismatch"}],
        ),
    )
    bad_alias = Case(
        "case-bool-reduction-skipna-probe-alias",
        93,
        [TableData("t0", [ColumnSpec("probe_id", "int")], [{"probe_id": 0}])],
        Program(
            "prog-bool-reduction-skipna-probe-alias",
            93,
            [{"op": "bool_reduction_skipna_probe", "as": "where"}],
        ),
    )

    assert validate_case_program(valid) == []
    assert any("reserved" in error for error in validate_case_program(bad_alias))


def test_classification_reference_understands_bool_reduction_skipna_probe():
    case = Case(
        "case-bool-reduction-skipna-reference",
        94,
        [TableData("t0", [ColumnSpec("probe_id", "int")], [{"probe_id": 0}])],
        Program(
            "prog-bool-reduction-skipna-reference",
            94,
            [{"op": "bool_reduction_skipna_probe", "as": "bool_reduction_skipna_mismatch"}],
        ),
    )
    finding = {
        "kind": "semantic_output_mismatch",
        "root_cause": "pandas_bool_reduction_skipna_semantics",
        "confidence": "high",
        "suspicious_backends": ["pandas"],
    }
    normalized = {
        "reference": NormalizedResult("reference", "ok", ["bool_reduction_skipna_mismatch"], [[False]]),
        "pandas": NormalizedResult("pandas", "ok", ["bool_reduction_skipna_mismatch"], [[True]]),
    }

    classification = classify_finding(
        case,
        finding,
        normalized,
        {},
        {"generator_profile": "pandas_bool_reduction_skipna_semantics"},
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


def test_validate_case_accepts_list_flatten_parent_indices_probe():
    valid = Case(
        "case-list-flatten-parent-indices-probe",
        98,
        [TableData("t0", [ColumnSpec("probe_id", "int")], [{"probe_id": 0}])],
        Program(
            "prog-list-flatten-parent-indices-probe",
            98,
            [{"op": "list_flatten_parent_indices_probe", "as": "list_flatten_parent_indices_mismatch"}],
        ),
    )
    bad_alias = Case(
        "case-list-flatten-parent-indices-probe-alias",
        99,
        [TableData("t0", [ColumnSpec("probe_id", "int")], [{"probe_id": 0}])],
        Program(
            "prog-list-flatten-parent-indices-probe-alias",
            99,
            [{"op": "list_flatten_parent_indices_probe", "as": "where"}],
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


def test_classification_reference_understands_list_flatten_parent_indices_probe():
    case = Case(
        "case-list-flatten-parent-indices-reference",
        100,
        [TableData("t0", [ColumnSpec("probe_id", "int")], [{"probe_id": 0}])],
        Program(
            "prog-list-flatten-parent-indices-reference",
            100,
            [{"op": "list_flatten_parent_indices_probe", "as": "list_flatten_parent_indices_mismatch"}],
        ),
    )
    finding = {
        "kind": "semantic_output_mismatch",
        "root_cause": "pyarrow_list_flatten_parent_indices_semantics",
        "confidence": "high",
        "suspicious_backends": ["pyarrow"],
    }
    normalized = {
        "reference": NormalizedResult("reference", "ok", ["list_flatten_parent_indices_mismatch"], [[False]]),
        "pyarrow": NormalizedResult("pyarrow", "ok", ["list_flatten_parent_indices_mismatch"], [[True]]),
    }

    classification = classify_finding(
        case,
        finding,
        normalized,
        {},
        {"generator_profile": "pyarrow_list_flatten_parent_indices_semantics"},
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


def test_validate_case_allows_mean_on_numeric_columns_only():
    mean_numeric = Case(
        "case-mean-numeric",
        108,
        [
            TableData(
                "t0",
                [ColumnSpec("g", "str"), ColumnSpec("x", "int"), ColumnSpec("s", "str")],
                [{"g": "a", "x": 1, "s": "one"}, {"g": "a", "x": 3, "s": "two"}],
            )
        ],
        Program(
            "prog-mean-numeric",
            108,
            [{"op": "groupby", "keys": ["g"], "aggs": [{"column": "x", "func": "mean", "as": "mean_x"}]}],
        ),
    )
    mean_string = Case(
        "case-mean-string",
        109,
        [
            TableData(
                "t0",
                [ColumnSpec("g", "str"), ColumnSpec("x", "int"), ColumnSpec("s", "str")],
                [{"g": "a", "x": 1, "s": "one"}, {"g": "a", "x": 3, "s": "two"}],
            )
        ],
        Program(
            "prog-mean-string",
            109,
            [{"op": "groupby", "keys": ["g"], "aggs": [{"column": "s", "func": "mean", "as": "mean_s"}]}],
        ),
    )

    assert validate_case_program(mean_numeric) == []
    assert any("not numeric" in error for error in validate_case_program(mean_string))


def test_validate_case_allows_any_and_all_on_bool_columns_only():
    bool_case = Case(
        "case-bool-any-all",
        11,
        [
            TableData(
                "t0",
                [ColumnSpec("g", "str"), ColumnSpec("flag", "bool"), ColumnSpec("x", "int")],
                [{"g": "a", "flag": True, "x": 1}, {"g": "a", "flag": None, "x": 2}],
            )
        ],
        Program(
            "prog-bool-any-all",
            11,
            [
                {
                    "op": "groupby",
                    "keys": ["g"],
                    "aggs": [
                        {"column": "flag", "func": "any", "as": "flag_any"},
                        {"column": "flag", "func": "all", "as": "flag_all"},
                    ],
                }
            ],
        ),
    )
    int_case = Case(
        "case-int-any",
        12,
        [TableData("t0", [ColumnSpec("g", "str"), ColumnSpec("x", "int")], [{"g": "a", "x": 1}])],
        Program(
            "prog-int-any",
            12,
            [{"op": "groupby", "keys": ["g"], "aggs": [{"column": "x", "func": "any", "as": "x_any"}]}],
        ),
    )

    assert validate_case_program(bool_case) == []
    assert any("not numeric" in error for error in validate_case_program(int_case))


def test_classification_reference_understands_bool_any_all_all_null_input():
    case = Case(
        "case-bool-any-all-reference",
        13,
        [TableData("t0", [ColumnSpec("flag", "bool")], [{"flag": None}, {"flag": None}])],
        Program(
            "prog-bool-any-all-reference",
            13,
            [
                {
                    "op": "aggregate",
                    "aggs": [
                        {"column": "flag", "func": "any", "as": "any_flag"},
                        {"column": "flag", "func": "all", "as": "all_flag"},
                    ],
                }
            ],
        ),
    )
    finding = {
        "kind": "semantic_output_mismatch",
        "root_cause": "groupby_aggregation",
        "confidence": "medium",
        "suspicious_backends": ["sqlite"],
    }
    normalized = {
        "pandas": NormalizedResult("pandas", "ok", ["all_flag", "any_flag"], [[None, None]]),
        "sqlite": NormalizedResult("sqlite", "ok", ["all_flag", "any_flag"], [[False, False]]),
    }

    classification = classify_finding(
        case,
        finding,
        normalized,
        {},
        {"generator_profile": "bool_null_groupby_agg"},
        ["pandas", "sqlite"],
    )

    assert classification.verdict == "candidate_implementation_bug"
    assert classification.implicated_backends == ["sqlite"]


def test_classification_reference_anchor_summary_flags_backend_minority():
    case = Case(
        "case-reference-anchor-summary",
        113,
        [TableData("t0", [ColumnSpec("flag", "bool")], [{"flag": None}, {"flag": None}])],
        Program(
            "prog-reference-anchor-summary",
            113,
            [
                {
                    "op": "aggregate",
                    "aggs": [
                        {"column": "flag", "func": "any", "as": "any_flag"},
                        {"column": "flag", "func": "all", "as": "all_flag"},
                    ],
                }
            ],
        ),
    )
    finding = {
        "kind": "semantic_output_mismatch",
        "root_cause": "groupby_aggregation",
        "confidence": "medium",
        "suspicious_backends": ["sqlite"],
    }
    normalized = {
        "pandas": NormalizedResult("pandas", "ok", ["all_flag", "any_flag"], [[None, None]]),
        "sqlite": NormalizedResult("sqlite", "ok", ["all_flag", "any_flag"], [[False, False]]),
    }

    classification = classify_finding(
        case,
        finding,
        normalized,
        {},
        {"generator_profile": "bool_null_groupby_agg"},
        ["pandas", "sqlite"],
    )

    assert classification.verdict == "candidate_implementation_bug"
    assert classification.implicated_backends == ["sqlite"]
    assert "disagrees with ['sqlite']" in classification.evidence


def test_classification_excludes_stale_pyarrow_empty_global_bool_aggregate_adapter_error():
    case = Case(
        "case-pyarrow-empty-global-bool-aggregate-stale",
        14,
        [
            TableData(
                "t0",
                [ColumnSpec("id", "int", nullable=False), ColumnSpec("flag", "bool")],
                [{"id": 0, "flag": None}, {"id": 1, "flag": True}],
            )
        ],
        Program(
            "prog-pyarrow-empty-global-bool-aggregate-stale",
            14,
            [
                {"op": "filter", "column": "id", "cmp": "<", "value": 0},
                {
                    "op": "aggregate",
                    "aggs": [
                        {"column": "flag", "func": "any", "as": "any_flag"},
                        {"column": "flag", "func": "all", "as": "all_flag"},
                    ],
                },
                {"op": "filter", "column": "any_flag", "cmp": "bool_is_not_true", "value": None},
            ],
        ),
    )
    finding = {
        "kind": "accept_reject_mismatch",
        "root_cause": "groupby_aggregation",
        "confidence": "high",
        "suspicious_backends": ["pyarrow"],
    }
    normalized = {
        "pandas": NormalizedResult("pandas", "ok", ["all_flag", "any_flag"], [[None, None]]),
        "pyarrow": NormalizedResult("pyarrow", "error", [], [], "ArrowInvalid", "Invalid null value"),
    }
    raw_results = {"pyarrow": {"status": "error", "error_type": "ArrowInvalid", "error": "Invalid null value"}}

    classification = classify_finding(
        case,
        finding,
        normalized,
        raw_results,
        {"generator_profile": "common_api_workflow"},
        ["pandas", "pyarrow"],
    )

    assert classification.verdict == "normalizer_false_positive"
    assert classification.false_positive is True
    assert classification.false_positive_reason == "pyarrow_empty_global_bool_aggregate_adapter_error"


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
