from datadiff.dsl import Case, ColumnSpec, Program, TableData
from datadiff.triage import build_triage_report, supports_standalone_reproducer, write_standalone_reproducer


def test_triage_marks_polars_nan_filter_as_documented_divergence():
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
    report = build_triage_report(
        case,
        original_findings=[{"kind": "semantic_output_mismatch"}],
        reproduced_findings=[
            {
                "kind": "semantic_output_mismatch",
                "root_cause": "nan_inf_semantics",
                "confidence": "high",
                "suspicious_backends": ["polars"],
            }
        ],
        config={"generator_profile": "edge_float"},
        backends=["pandas", "polars", "duckdb", "sqlite"],
    )
    assert report["verdict"] == "documented_semantic_divergence"
    assert report["paper_status"] == "valid_finding_not_bug"
    assert report["documentation_refs"]


def test_triage_marks_clear_common_subset_minority_as_candidate_bug():
    case = Case(
        "case-common",
        2,
        [TableData("t0", [ColumnSpec("x", "int")], [{"x": 1}, {"x": 2}])],
        Program("prog-common", 2, [{"op": "filter", "column": "x", "cmp": ">", "value": 1}]),
    )
    report = build_triage_report(
        case,
        original_findings=[{"kind": "semantic_output_mismatch"}],
        reproduced_findings=[
            {
                "kind": "semantic_output_mismatch",
                "root_cause": "filter_predicate",
                "confidence": "high",
                "suspicious_backends": ["engine_x"],
            }
        ],
        config={"generator_profile": "common"},
        backends=["engine_a", "engine_b", "engine_x"],
    )
    assert report["verdict"] == "candidate_implementation_bug"


def test_triage_does_not_treat_edge_float_profile_as_boundary_by_itself():
    case = Case(
        "case-edge-common",
        22,
        [TableData("t0", [ColumnSpec("x", "int")], [{"x": 1}, {"x": 2}])],
        Program("prog-edge-common", 22, [{"op": "filter", "column": "x", "cmp": ">", "value": 1}]),
    )
    report = build_triage_report(
        case,
        original_findings=[{"kind": "semantic_output_mismatch"}],
        reproduced_findings=[
            {
                "kind": "semantic_output_mismatch",
                "root_cause": "filter_predicate",
                "confidence": "high",
                "suspicious_backends": ["engine_x"],
            }
        ],
        config={"generator_profile": "edge_float"},
        backends=["engine_a", "engine_b", "engine_x"],
    )

    assert report["verdict"] == "candidate_implementation_bug"


def test_standalone_reproducer_supports_known_root_causes():
    assert supports_standalone_reproducer(
        {
            "generator_profile": "common",
            "features": {"contains_nan": False, "contains_inf": False},
            "reproduced_roots": ["grouped_topk_null_sort_key"],
        }
    ) is True
    assert supports_standalone_reproducer(
        {
            "generator_profile": "bughunt",
            "features": {"contains_nan": False, "contains_inf": False},
            "reproduced_roots": ["groupby_aggregation"],
            "suspicious_backends": ["datafusion"],
        }
    ) is True
    assert supports_standalone_reproducer(
        {
            "generator_profile": "bughunt",
            "features": {"contains_nan": False, "contains_inf": False},
            "reproduced_roots": ["outer_join_truth_filter"],
            "suspicious_backends": ["datafusion"],
        }
    ) is True
    assert supports_standalone_reproducer(
        {
            "generator_profile": "bughunt",
            "features": {"contains_nan": False, "contains_inf": False},
            "reproduced_roots": ["joined_order_offset_projection"],
            "suspicious_backends": ["datafusion"],
        }
    ) is True
    assert supports_standalone_reproducer(
        {
            "generator_profile": "common",
            "features": {"contains_nan": False, "contains_inf": False},
            "reproduced_roots": ["reverse_division_operand_order"],
        }
    ) is True
    assert supports_standalone_reproducer(
        {
            "generator_profile": "common",
            "features": {"contains_nan": False, "contains_inf": False},
            "reproduced_roots": ["tuple_absence_null_filter"],
        }
    ) is True
    assert supports_standalone_reproducer(
        {
            "generator_profile": "common",
            "features": {"contains_nan": True, "contains_inf": False},
            "reproduced_roots": ["nan_inf_semantics"],
        }
    ) is True
    assert supports_standalone_reproducer(
        {
            "generator_profile": "common",
            "features": {"contains_nan": False, "contains_inf": False},
            "reproduced_roots": ["filter_predicate"],
        }
    ) is False


def test_write_datafusion_standalone_reproducer_for_grouped_topk(tmp_path):
    path = write_standalone_reproducer(
        tmp_path,
        {
            "reproduced_roots": ["grouped_topk_null_sort_key"],
        },
    )

    assert path.name == "standalone_datafusion_groupby_null_sortkey_limit.py"
    text = path.read_text(encoding="utf-8")
    assert "ORDER BY min_x ASC NULLS LAST LIMIT 20" in text
    assert "DataFusion dropped the group" in text


def test_write_datafusion_standalone_reproducer_for_groupby_limit_offset(tmp_path):
    path = write_standalone_reproducer(
        tmp_path,
        {
            "reproduced_roots": ["groupby_aggregation"],
            "suspicious_backends": ["datafusion"],
        },
    )

    assert path.name == "standalone_datafusion_groupby_limit_offset.py"
    text = path.read_text(encoding="utf-8")
    assert "COUNT(DISTINCT j)" in text
    assert "outer ORDER BY/OFFSET" in text


def test_write_datafusion_standalone_reproducer_for_sort_offset_groupby_aggregation(tmp_path):
    path = write_standalone_reproducer(
        tmp_path,
        {
            "reproduced_roots": ["groupby_aggregation"],
            "suspicious_backends": ["datafusion"],
            "features": {
                "operation_sequence": ["join", "mutate", "sort", "offset", "groupby"],
                "uses_limit": False,
            },
        },
    )

    assert path.name == "standalone_datafusion_sort_offset_groupby_aggregation.py"
    text = path.read_text(encoding="utf-8")
    assert "OFFSET 11" in text
    assert "MIN(m_0)" in text


def test_write_datafusion_standalone_reproducer_for_negative_zero_truth_filter(tmp_path):
    path = write_standalone_reproducer(
        tmp_path,
        {
            "reproduced_roots": ["outer_join_truth_filter"],
            "suspicious_backends": ["datafusion"],
        },
    )

    assert path.name == "standalone_datafusion_negative_zero_truth_filter.py"
    text = path.read_text(encoding="utf-8")
    assert "-0.0 >= 0.0" in text
    assert "IS NOT TRUE" in text


def test_write_datafusion_standalone_reproducer_for_negative_zero_root(tmp_path):
    path = write_standalone_reproducer(
        tmp_path,
        {
            "reproduced_roots": ["negative_zero_comparison"],
            "suspicious_backends": ["datafusion"],
        },
    )

    assert path.name == "standalone_datafusion_negative_zero_truth_filter.py"


def test_write_datafusion_standalone_reproducer_for_ordered_topk_projection(tmp_path):
    path = write_standalone_reproducer(
        tmp_path,
        {
            "reproduced_roots": ["ordered_topk_projection"],
            "suspicious_backends": ["datafusion"],
        },
    )

    assert path.name == "standalone_datafusion_ordered_topk_projection.py"
    text = path.read_text(encoding="utf-8")
    assert "ORDER BY x DESC NULLS LAST, g DESC NULLS LAST" in text
    assert "OFFSET 1" in text


def test_write_datafusion_standalone_reproducer_for_truth_filter_offset(tmp_path):
    path = write_standalone_reproducer(
        tmp_path,
        {
            "reproduced_roots": ["outer_join_truth_filter"],
            "suspicious_backends": ["datafusion"],
            "features": {
                "operation_sequence": ["join", "mutate", "filter", "sort", "offset", "sort"],
                "uses_filter": True,
                "uses_groupby": False,
                "uses_limit": False,
            },
        },
    )

    assert path.name == "standalone_datafusion_truth_filter_offset.py"
    text = path.read_text(encoding="utf-8")
    assert "NOT ((q.\"m_0\" <= 0.5) IS FALSE)" in text
    assert "OFFSET 2" in text


def test_write_datafusion_standalone_reproducer_for_joined_order_offset_projection(tmp_path):
    path = write_standalone_reproducer(
        tmp_path,
        {
            "reproduced_roots": ["joined_order_offset_projection"],
            "suspicious_backends": ["datafusion"],
        },
    )

    assert path.name == "standalone_datafusion_joined_order_offset_projection.py"
    text = path.read_text(encoding="utf-8")
    assert "OFFSET 1" in text
    assert "__datadiff_order_0_0" in text
    assert "DataFusion returned the row that OFFSET should skip" in text


def test_write_polars_standalone_reproducer_for_reverse_division(tmp_path):
    path = write_standalone_reproducer(
        tmp_path,
        {
            "reproduced_roots": ["reverse_division_operand_order"],
        },
    )

    assert path.name == "standalone_polars_reverse_division_columns.py"
    text = path.read_text(encoding="utf-8")
    assert "Series.__rtruediv__" in text
    assert "numerator_value" in text


def test_write_duckdb_standalone_reproducer_for_tuple_absence_null_filter(tmp_path):
    path = write_standalone_reproducer(
        tmp_path,
        {
            "reproduced_roots": ["tuple_absence_null_filter"],
        },
    )

    assert path.name == "standalone_duckdb_tuple_absence_null_filter.py"
    text = path.read_text(encoding="utf-8")
    assert "NOT IN" in text
    assert "duckdb row equality" in text


def test_triage_marks_boundary_semantics_as_expected_divergence():
    case = Case(
        "case-boundary",
        3,
        [TableData("t0", [ColumnSpec("x", "int")], [{"x": -3}, {"x": 4}])],
        Program(
            "prog-boundary",
            3,
            [{"op": "mutate", "column": "m", "expr": {"kind": "arith_const", "op": "mod", "source": "x", "value": 2}}],
        ),
    )
    report = build_triage_report(
        case,
        original_findings=[{"kind": "semantic_output_mismatch"}],
        reproduced_findings=[
            {
                "kind": "semantic_output_mismatch",
                "root_cause": "arithmetic_expression",
                "confidence": "high",
                "suspicious_backends": ["sqlite"],
            }
        ],
        config={"generator_profile": "common"},
        backends=["pandas", "sqlite"],
    )
    assert report["verdict"] == "expected_semantic_divergence"
    assert report["paper_status"] == "valid_finding_not_bug"
