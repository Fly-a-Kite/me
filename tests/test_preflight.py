from datadiff.dsl import Case, ColumnSpec, Program, TableData
from datadiff.preflight import preflight_case


def test_preflight_repairs_invalid_program_with_fallback():
    case = Case(
        "case-invalid",
        1,
        [TableData("t0", [ColumnSpec("x", "int")], [{"x": 1}])],
        Program("prog-invalid", 1, [{"op": "select", "columns": ["missing"]}]),
        metadata={"seed_lineage": {"root_seed": 1, "depth": 1}, "mutation": {"operator": "drop_op"}},
    )

    result = preflight_case(case)

    assert result.valid is True
    assert result.repaired is True
    assert result.fallback_used is True
    assert result.case.program.operations == [{"op": "limit", "n": 1}]
    assert result.case.metadata == case.metadata
    assert result.errors_before
    assert result.errors_after == []


def test_preflight_can_report_invalid_without_repair():
    case = Case(
        "case-invalid",
        1,
        [TableData("t0", [ColumnSpec("x", "int")], [{"x": 1}])],
        Program("prog-invalid", 1, [{"op": "filter", "column": "missing", "cmp": ">", "value": 0}]),
    )

    result = preflight_case(case, enable_repair=False)

    assert result.valid is False
    assert result.repaired is False
    assert result.errors_after == result.errors_before


def test_preflight_handles_case_without_tables():
    case = Case("case-empty", 1, [], Program("prog-empty", 1, [{"op": "limit", "n": 1}]))

    result = preflight_case(case)

    assert result.valid is False
    assert result.repaired is True
    assert result.fallback_used is True
    assert result.errors_after == ["case has no tables"]


def test_preflight_repairs_cross_type_filter_literal():
    case = Case(
        "case-invalid-filter-type",
        2,
        [TableData("t0", [ColumnSpec("g", "str")], [{"g": "Alpha"}])],
        Program(
            "prog-invalid-filter-type",
            2,
            [
                {"op": "mutate", "column": "m_0", "expr": {"kind": "string_lower", "source": "g"}},
                {"op": "filter", "column": "m_0", "cmp": "<", "value": 0.5},
            ],
        ),
    )

    result = preflight_case(case)

    assert result.valid is True
    assert result.repaired is True
    assert result.case.program.operations == [
        {"op": "mutate", "column": "m_0", "expr": {"kind": "string_lower", "source": "g"}}
    ]


def test_preflight_repairs_duplicate_select_columns():
    case = Case(
        "case-dup-select",
        3,
        [TableData("t0", [ColumnSpec("x", "int"), ColumnSpec("g", "str")], [{"x": 1, "g": "Alpha"}])],
        Program("prog-dup-select", 3, [{"op": "select", "columns": ["x", "x", "g"]}]),
    )

    result = preflight_case(case)

    assert result.valid is True
    assert result.repaired is True
    assert result.fallback_used is False
    assert result.case.program.operations == [{"op": "select", "columns": ["x", "g"]}]
