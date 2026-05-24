import importlib.util

import pytest

from datadiff.config import ExperimentConfig
from datadiff.datagen import generate_case
from datadiff.dsl import Case, ColumnSpec, Program, TableData
from datadiff.runner import run_loaded_case
from datadiff.triage import build_triage_report


REQUIRED_BACKENDS = ["pandas", "polars", "duckdb", "sqlite"]
DATAFUSION_BACKENDS = ["pandas", "duckdb", "datafusion"]
PYARROW_BACKENDS = ["pandas", "duckdb", "pyarrow"]
POLARS_LAZY_BACKENDS = ["polars", "polars_lazy"]
EMBEDDED_SQL_BACKENDS = ["duckdb", "sqlite"]


@pytest.mark.skipif(
    any(name != "sqlite" and importlib.util.find_spec(name) is None for name in REQUIRED_BACKENDS),
    reason="data backends are not installed",
)
def test_polars_nan_filter_divergence_is_documented_not_bug():
    case = Case(
        "case-polars-nan-filter",
        14096,
        [TableData("t0", [ColumnSpec("y", "float")], [{"y": float("nan")}])],
        Program(
            "prog-polars-nan-filter",
            14096,
            [
                {
                    "op": "mutate",
                    "column": "m_0",
                    "expr": {"kind": "add_const", "source": "y", "value": 10},
                },
                {"op": "filter", "column": "m_0", "cmp": ">", "value": 10.0},
            ],
        ),
    )
    config = ExperimentConfig(generator_profile="edge_float")
    result = run_loaded_case(case, REQUIRED_BACKENDS, config=config, save_artifact=False)

    assert result["status"] == "bug"
    assert result["findings"][0]["kind"] == "semantic_output_mismatch"
    assert result["findings"][0]["root_cause"] == "nan_inf_semantics"
    assert result["findings"][0]["suspicious_backends"] == ["polars"]

    report = build_triage_report(
        case,
        original_findings=[{"kind": "semantic_output_mismatch"}],
        reproduced_findings=result["findings"],
        config=config.to_dict(),
        backends=REQUIRED_BACKENDS,
    )
    assert report["verdict"] == "documented_semantic_divergence"
    assert report["paper_status"] == "valid_finding_not_bug"


@pytest.mark.skipif(
    any(name != "sqlite" and importlib.util.find_spec(name) is None for name in REQUIRED_BACKENDS),
    reason="data backends are not installed",
)
def test_polars_nan_groupby_count_divergence_is_documented_not_bug():
    case = Case(
        "case-polars-nan-count",
        40069,
        [
            TableData(
                "t0",
                [
                    ColumnSpec("flag", "bool"),
                    ColumnSpec("y", "float"),
                ],
                [{"flag": False, "y": float("nan")}],
            )
        ],
        Program(
            "prog-polars-nan-count",
            40069,
            [
                {
                    "op": "groupby",
                    "keys": ["flag"],
                    "aggs": [{"column": "y", "func": "count", "as": "count_y"}],
                }
            ],
        ),
    )
    config = ExperimentConfig(generator_profile="edge_float")
    result = run_loaded_case(case, REQUIRED_BACKENDS, config=config, save_artifact=False)

    assert result["status"] == "bug"
    assert result["normalized"]["polars"]["rows"] == [[1, False]]
    assert result["normalized"]["pandas"]["rows"] == [[0, False]]
    assert result["normalized"]["duckdb"]["rows"] == [[0, False]]
    assert result["normalized"]["sqlite"]["rows"] == [[0, False]]

    report = build_triage_report(
        case,
        original_findings=[{"kind": "semantic_output_mismatch"}],
        reproduced_findings=result["findings"],
        config=config.to_dict(),
        backends=REQUIRED_BACKENDS,
    )
    assert report["verdict"] == "documented_semantic_divergence"
    assert report["paper_status"] == "valid_finding_not_bug"


@pytest.mark.skipif(
    any(name != "sqlite" and importlib.util.find_spec(name) is None for name in REQUIRED_BACKENDS),
    reason="data backends are not installed",
)
def test_polars_string_length_arithmetic_does_not_wrap_unsigned():
    case = Case(
        "case-polars-len-arith-wrap",
        911117,
        [
            TableData(
                "t0",
                [ColumnSpec("g", "str", nullable=True)],
                [{"g": "A"}],
            )
        ],
        Program(
            "prog-polars-len-arith-wrap",
            911117,
            [
                {"op": "mutate", "column": "m_0", "expr": {"kind": "string_length", "source": "g"}},
                {"op": "mutate", "column": "m_1", "expr": {"kind": "arith_const", "op": "sub", "source": "m_0", "value": 10}},
                {"op": "mutate", "column": "m_2", "expr": {"kind": "arith_const", "op": "sub", "source": "m_1", "value": -2}},
                {"op": "mutate", "column": "m_3", "expr": {"kind": "arith_const", "op": "sub", "source": "m_1", "value": 10}},
                {
                    "op": "groupby",
                    "keys": ["m_3"],
                    "aggs": [
                        {"column": "m_1", "func": "count", "as": "count_m_1"},
                        {"column": "m_2", "func": "min", "as": "min_m_2"},
                        {"column": "m_0", "func": "count", "as": "count_m_0"},
                    ],
                },
            ],
        ),
    )

    result = run_loaded_case(case, REQUIRED_BACKENDS, config=ExperimentConfig(), save_artifact=False)

    assert result["status"] == "ok"
    expected_rows = [[1, 1, -19, -7]]
    assert result["normalized"]["pandas"]["rows"] == expected_rows
    assert result["normalized"]["duckdb"]["rows"] == expected_rows
    assert result["normalized"]["sqlite"]["rows"] == expected_rows
    assert result["normalized"]["polars"]["rows"] == expected_rows


@pytest.mark.skipif(
    any(importlib.util.find_spec(name) is None for name in DATAFUSION_BACKENDS),
    reason="datafusion comparison backends are not installed",
)
def test_datafusion_grouped_topk_null_sort_key_is_candidate_bug():
    case = Case(
        "case-datafusion-grouped-topk-null-sort-key",
        917531,
        [
            TableData(
                "t0",
                [
                    ColumnSpec("g", "str", nullable=True),
                    ColumnSpec("x", "int", nullable=True),
                ],
                [{"g": "a", "x": None}],
            )
        ],
        Program(
            "prog-datafusion-grouped-topk-null-sort-key",
            917531,
            [
                {
                    "op": "groupby",
                    "keys": ["g"],
                    "aggs": [{"column": "x", "func": "min", "as": "min_x"}],
                },
                {"op": "sort", "columns": ["min_x"], "ascending": True},
                {"op": "limit", "n": 20},
            ],
        ),
    )

    result = run_loaded_case(
        case,
        DATAFUSION_BACKENDS,
        config=ExperimentConfig(),
        save_artifact=False,
    )

    assert result["status"] == "bug"
    assert result["normalized"]["pandas"]["rows"] == [["a", None]]
    assert result["normalized"]["duckdb"]["rows"] == [["a", None]]
    assert result["normalized"]["datafusion"]["rows"] == []
    assert result["findings"][0]["root_cause"] == "grouped_topk_null_sort_key"
    assert result["findings"][0]["suspicious_backends"] == ["datafusion"]

    report = build_triage_report(
        case,
        original_findings=[{"kind": "semantic_output_mismatch"}],
        reproduced_findings=result["findings"],
        config=ExperimentConfig().to_dict(),
        backends=DATAFUSION_BACKENDS,
    )
    assert report["verdict"] == "candidate_implementation_bug"
    assert report["paper_status"] == "candidate_bug_needs_external_confirmation"


@pytest.mark.skipif(
    any(importlib.util.find_spec(name) is None for name in DATAFUSION_BACKENDS),
    reason="datafusion comparison backends are not installed",
)
def test_datafusion_groupby_limit_offset_is_candidate_bug():
    case = Case(
        "case-datafusion-groupby-limit-offset",
        22234,
        [
            TableData("t0", [ColumnSpec("id", "int", nullable=False)], [{"id": 0}, {"id": 1}]),
            TableData(
                "t1",
                [
                    ColumnSpec("id", "int", nullable=False),
                    ColumnSpec("j", "int", nullable=True),
                ],
                [{"id": 1, "j": 1}],
            ),
        ],
        Program(
            "prog-datafusion-groupby-limit-offset",
            22234,
            [
                {"op": "join", "table": "t1", "left_on": "id", "right_on": "id", "how": "left"},
                {
                    "op": "groupby",
                    "keys": ["id"],
                    "aggs": [
                        {"column": "j", "func": "nunique", "as": "nunique_j"},
                        {"column": "id", "func": "count", "as": "count_id"},
                    ],
                },
                {
                    "op": "sort",
                    "keys": [
                        {"column": "id", "ascending": False, "nulls": "last"},
                        {"column": "count_id", "ascending": False, "nulls": "last"},
                        {"column": "nunique_j", "ascending": False, "nulls": "last"},
                    ],
                },
                {"op": "limit", "n": 8},
                {
                    "op": "sort",
                    "keys": [
                        {"column": "id", "ascending": False, "nulls": "last"},
                        {"column": "count_id", "ascending": False, "nulls": "last"},
                        {"column": "nunique_j", "ascending": True, "nulls": "last"},
                    ],
                },
                {"op": "offset", "n": 1},
            ],
        ),
    )

    result = run_loaded_case(
        case,
        DATAFUSION_BACKENDS,
        config=ExperimentConfig(),
        save_artifact=False,
    )

    assert result["status"] == "bug"
    assert result["normalized"]["pandas"]["rows"] == [[1, 0, 0]]
    assert result["normalized"]["duckdb"]["rows"] == [[1, 0, 0]]
    assert result["normalized"]["datafusion"]["rows"] == []
    assert result["findings"][0]["root_cause"] == "groupby_aggregation"
    assert result["findings"][0]["suspicious_backends"] == ["datafusion"]

    report = build_triage_report(
        case,
        original_findings=[{"kind": "semantic_output_mismatch"}],
        reproduced_findings=result["findings"],
        config=ExperimentConfig().to_dict(),
        backends=DATAFUSION_BACKENDS,
    )
    assert report["verdict"] == "candidate_implementation_bug"


@pytest.mark.skipif(
    any(importlib.util.find_spec(name) is None for name in DATAFUSION_BACKENDS),
    reason="datafusion comparison backends are not installed",
)
def test_datafusion_negative_zero_truth_filter_is_candidate_bug():
    case = Case(
        "case-datafusion-negative-zero-truth-filter",
        141204,
        [
            TableData(
                "t0",
                [
                    ColumnSpec("id", "int", nullable=False),
                    ColumnSpec("y", "float", nullable=True),
                ],
                [{"id": 1, "y": 0.0}],
            ),
            TableData("t1", [ColumnSpec("id", "int", nullable=False)], [{"id": 18}]),
        ],
        Program(
            "prog-datafusion-negative-zero-truth-filter",
            141204,
            [
                {"op": "join", "table": "t1", "left_on": "id", "right_on": "id", "how": "left"},
                {
                    "op": "mutate",
                    "column": "m_0",
                    "expr": {"kind": "arith_const", "op": "mul", "source": "y", "value": -1},
                },
                {"op": "filter", "column": "m_0", "cmp": "ge_is_not_true", "value": 0.0},
            ],
        ),
    )

    result = run_loaded_case(
        case,
        DATAFUSION_BACKENDS,
        config=ExperimentConfig(),
        save_artifact=False,
    )

    assert result["status"] == "bug"
    assert result["normalized"]["pandas"]["rows"] == []
    assert result["normalized"]["duckdb"]["rows"] == []
    assert result["normalized"]["datafusion"]["rows"] == [[1, 0, 0]]
    assert result["findings"][0]["root_cause"] == "outer_join_truth_filter"
    assert result["findings"][0]["suspicious_backends"] == ["datafusion"]

    report = build_triage_report(
        case,
        original_findings=[{"kind": "semantic_output_mismatch"}],
        reproduced_findings=result["findings"],
        config=ExperimentConfig().to_dict(),
        backends=DATAFUSION_BACKENDS,
    )
    assert report["verdict"] == "candidate_implementation_bug"


@pytest.mark.skipif(
    any(importlib.util.find_spec(name) is None for name in PYARROW_BACKENDS),
    reason="pyarrow comparison backends are not installed",
)
def test_pyarrow_groupby_filter_cast_membership_does_not_cast_fractional_literals_to_ints():
    case = generate_case(123, profile="pyarrow_groupby_filter_cast_membership")

    result = run_loaded_case(
        case,
        PYARROW_BACKENDS,
        config=ExperimentConfig(),
        save_artifact=False,
    )

    assert result["status"] == "ok"
    assert result["normalized"]["pandas"]["rows"] == []
    assert result["normalized"]["duckdb"]["rows"] == []
    assert result["normalized"]["pyarrow"]["rows"] == []
    assert result["findings"] == []

    report = build_triage_report(
        case,
        original_findings=[{"kind": "semantic_output_mismatch"}],
        reproduced_findings=result["findings"],
        config=ExperimentConfig().to_dict(),
        backends=PYARROW_BACKENDS,
    )
    assert report["verdict"] == "not_reproduced"
    assert report["paper_status"] == "not_usable_until_reproduced"


@pytest.mark.skipif(
    importlib.util.find_spec("polars") is None,
    reason="polars comparison backends are not installed",
)
def test_polars_reverse_division_columns_is_candidate_bug():
    case = generate_case(135, profile="polars_reverse_division_columns")

    result = run_loaded_case(
        case,
        POLARS_LAZY_BACKENDS,
        config=ExperimentConfig(),
        save_artifact=False,
    )

    assert result["status"] == "bug"
    assert result["normalized"]["polars"]["rows"] == [[0, 0.5], [1, 0.6666666667], [1, 0.75]]
    assert result["normalized"]["polars_lazy"]["rows"] == [[0, 2], [1, 1.3333333333], [1, 1.5]]
    assert result["findings"][0]["root_cause"] == "reverse_division_operand_order"
    assert result["findings"][0]["suspicious_backends"] == ["polars"]

    report = build_triage_report(
        case,
        original_findings=[{"kind": "semantic_output_mismatch"}],
        reproduced_findings=result["findings"],
        config=ExperimentConfig().to_dict(),
        backends=POLARS_LAZY_BACKENDS,
    )
    assert report["verdict"] == "candidate_implementation_bug"
    assert report["paper_status"] == "candidate_bug_needs_external_confirmation"


@pytest.mark.skipif(
    importlib.util.find_spec("duckdb") is None,
    reason="duckdb comparison backend is not installed",
)
def test_row_value_absence_filter_is_organic_candidate_bug():
    case = generate_case(125, profile="row_value_absence_filter")

    result = run_loaded_case(
        case,
        EMBEDDED_SQL_BACKENDS,
        config=ExperimentConfig(generator_profile="row_value_absence_filter"),
        save_artifact=False,
    )

    assert result["status"] == "bug"
    assert result["normalized"]["duckdb"]["rows"] == []
    assert result["normalized"]["sqlite"]["rows"] == [["survivor", 2, 2, 1]]
    assert result["findings"][0]["root_cause"] == "tuple_absence_null_filter"
    assert result["findings"][0]["suspicious_backends"] == ["duckdb"]
    assert result["findings"][0]["discovery_origin"] == "organic"
    assert result["findings"][0]["source_issue"] == ""

    report = build_triage_report(
        case,
        original_findings=[{"kind": "semantic_output_mismatch"}],
        reproduced_findings=result["findings"],
        config=ExperimentConfig(generator_profile="row_value_absence_filter").to_dict(),
        backends=EMBEDDED_SQL_BACKENDS,
    )
    assert report["verdict"] == "candidate_implementation_bug"
    assert report["paper_status"] == "candidate_bug_needs_external_confirmation"


@pytest.mark.skipif(
    any(importlib.util.find_spec(name) is None for name in DATAFUSION_BACKENDS),
    reason="datafusion comparison backends are not installed",
)
def test_datafusion_grouped_topk_null_max_sort_key_is_candidate_bug():
    case = Case(
        "case-datafusion-grouped-topk-null-max-sort-key",
        917532,
        [
            TableData(
                "t0",
                [
                    ColumnSpec("g", "str", nullable=True),
                    ColumnSpec("x", "int", nullable=True),
                ],
                [{"g": "a", "x": None}],
            )
        ],
        Program(
            "prog-datafusion-grouped-topk-null-max-sort-key",
            917532,
            [
                {
                    "op": "groupby",
                    "keys": ["g"],
                    "aggs": [{"column": "x", "func": "max", "as": "max_x"}],
                },
                {"op": "sort", "columns": ["max_x"], "ascending": False},
                {"op": "limit", "n": 20},
            ],
        ),
    )

    result = run_loaded_case(
        case,
        DATAFUSION_BACKENDS,
        config=ExperimentConfig(),
        save_artifact=False,
    )

    assert result["status"] == "bug"
    assert result["normalized"]["pandas"]["rows"] == [["a", None]]
    assert result["normalized"]["duckdb"]["rows"] == [["a", None]]
    assert result["normalized"]["datafusion"]["rows"] == []
    assert result["findings"][0]["root_cause"] == "grouped_topk_null_sort_key"
    assert result["findings"][0]["suspicious_backends"] == ["datafusion"]


@pytest.mark.skipif(
    any(importlib.util.find_spec(name) is None for name in DATAFUSION_BACKENDS),
    reason="datafusion comparison backends are not installed",
)
def test_datafusion_joined_order_offset_projection_is_candidate_bug():
    case = Case(
        "case-datafusion-joined-order-offset-projection",
        109514,
        [
            TableData(
                "t0",
                [
                    ColumnSpec("id", "int", nullable=False),
                    ColumnSpec("g", "str", nullable=True),
                    ColumnSpec("x", "int", nullable=True),
                    ColumnSpec("y", "float", nullable=True),
                    ColumnSpec("flag", "bool", nullable=True),
                    ColumnSpec("s", "str", nullable=True),
                ],
                [
                    {"id": 0, "g": "filfM", "x": None, "y": -0.5, "flag": True, "s": "Iqm"},
                    {"id": 0, "g": "a", "x": -1, "y": 1.0, "flag": True, "s": "alpha"},
                ],
            ),
            TableData(
                "t1",
                [
                    ColumnSpec("id", "int", nullable=False),
                    ColumnSpec("j", "int", nullable=True),
                    ColumnSpec("z", "float", nullable=True),
                    ColumnSpec("tag", "str", nullable=True),
                ],
                [{"id": 0, "j": 0, "z": -39.802, "tag": "A"}],
            ),
        ],
        Program(
            "prog-datafusion-joined-order-offset-projection",
            109514,
            [
                {"op": "join", "table": "t1", "left_on": "id", "right_on": "id", "how": "inner"},
                {"op": "mutate", "column": "m_0", "expr": {"kind": "add_const", "source": "id", "value": -2}},
                {
                    "op": "sort",
                    "keys": [
                        {"column": "id", "ascending": True, "nulls": "first"},
                        {"column": "flag", "ascending": True, "nulls": "last"},
                        {"column": "g", "ascending": True, "nulls": "first"},
                        {"column": "j", "ascending": False, "nulls": "first"},
                        {"column": "m_0", "ascending": True, "nulls": "last"},
                        {"column": "s", "ascending": True, "nulls": "first"},
                        {"column": "tag", "ascending": True, "nulls": "last"},
                        {"column": "x", "ascending": False, "nulls": "first"},
                        {"column": "y", "ascending": False, "nulls": "first"},
                        {"column": "z", "ascending": True, "nulls": "last"},
                    ],
                },
                {"op": "offset", "n": 1},
                {
                    "op": "sort",
                    "keys": [
                        {"column": "s", "ascending": False, "nulls": "first"},
                        {"column": "g", "ascending": False, "nulls": "first"},
                        {"column": "j", "ascending": True, "nulls": "last"},
                    ],
                },
                {"op": "select", "columns": ["g"]},
            ],
        ),
    )

    result = run_loaded_case(
        case,
        DATAFUSION_BACKENDS,
        config=ExperimentConfig(),
        save_artifact=False,
    )

    assert result["status"] == "bug"
    assert result["normalized"]["pandas"]["rows"] == [["filfM"]]
    assert result["normalized"]["duckdb"]["rows"] == [["filfM"]]
    assert result["normalized"]["datafusion"]["rows"] == [["a"]]
    assert result["findings"][0]["root_cause"] == "joined_order_offset_projection"
    assert result["findings"][0]["suspicious_backends"] == ["datafusion"]
