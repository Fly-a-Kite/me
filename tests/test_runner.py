import importlib.util
from pathlib import Path

import pytest

from datadiff.config import ExperimentConfig
from datadiff.datagen import generate_case
from datadiff.dsl import Case, ColumnSpec, Program, TableData
from datadiff.normalizer import NormalizedResult
from datadiff.oracle import Finding
from datadiff import runner as runner_module
from datadiff.runner import run_fuzz, run_loaded_case
from datadiff.targets import resolve_target_backends
from datadiff.util import load_json, read_jsonl, run_meta_path


REQUIRED_BACKENDS = ["pandas", "polars", "duckdb", "sqlite"]
SQL_ORDER_BACKENDS = ["pandas", "duckdb", "sqlite"]
DATAFUSION_BACKENDS = ["pandas", "duckdb", "datafusion"]
PYARROW_BACKENDS = ["pandas", "duckdb", "pyarrow"]
BOOL_AGG_BACKENDS = ["pandas", "polars", "polars_lazy", "duckdb", "sqlite", "pyarrow", "datafusion"]
BOOL_AGG_BACKEND_PACKAGES = ["pandas", "polars", "duckdb", "pyarrow", "datafusion"]
MEAN_AGG_BACKENDS = [
    "pandas",
    "polars",
    "polars_lazy",
    "polars_streaming",
    "duckdb",
    "sqlite",
    "pyarrow",
    "datafusion",
]
MEAN_AGG_BACKEND_PACKAGES = ["pandas", "polars", "duckdb", "pyarrow", "datafusion"]
LATEST_ALL_ENGINE_PACKAGES = ["pandas", "pyarrow", "polars", "duckdb", "datafusion"]
PATH_KEYED_PICK_BACKENDS = ["pandas", "sqlite", "pyarrow", "polars", "polars_lazy", "datafusion"]
PATH_KEYED_PICK_PACKAGES = ["pandas", "pyarrow", "polars", "datafusion"]


@pytest.mark.skipif(
    any(name != "sqlite" and importlib.util.find_spec(name) is None for name in REQUIRED_BACKENDS),
    reason="data backends are not installed",
)
def test_run_loaded_case_smoke():
    case = generate_case(11)
    row = run_loaded_case(case, REQUIRED_BACKENDS, save_artifact=False)
    assert row["case"]["case_id"] == "case-00000011"
    assert set(row["normalized"]) == set(REQUIRED_BACKENDS)
    assert row["status"] in {"ok", "bug"}
    assert row["behavior_signature"]


@pytest.mark.skipif(
    any(name != "sqlite" and importlib.util.find_spec(name) is None for name in SQL_ORDER_BACKENDS),
    reason="data backends are not installed",
)
def test_run_loaded_case_supports_large_int64_values_in_duckdb():
    case = generate_case(123, profile="large_int_filter_groupby")
    row = run_loaded_case(case, SQL_ORDER_BACKENDS, save_artifact=False)

    assert row["status"] == "ok"
    assert row["findings"] == []
    assert row["normalized"]["duckdb"]["status"] == "ok"


@pytest.mark.skipif(
    any(importlib.util.find_spec(name) is None for name in BOOL_AGG_BACKEND_PACKAGES),
    reason="bool aggregation backends are not installed",
)
def test_run_loaded_case_supports_bool_any_all_aggregate_semantics():
    case = generate_case(123, profile="bool_null_groupby_agg")
    row = run_loaded_case(case, BOOL_AGG_BACKENDS, save_artifact=False)

    assert row["status"] == "ok"
    assert row["findings"] == []
    assert row["normalized"]["sqlite"]["status"] == "ok"


@pytest.mark.skipif(
    any(importlib.util.find_spec(name) is None for name in MEAN_AGG_BACKEND_PACKAGES),
    reason="mean aggregation backends are not installed",
)
def test_run_loaded_case_supports_mean_aggregate_semantics():
    case = Case(
        "case-mean-aggregate",
        4,
        [
            TableData(
                "t0",
                [
                    ColumnSpec("g", "str"),
                    ColumnSpec("x", "int"),
                ],
                [
                    {"g": "a", "x": 1},
                    {"g": "a", "x": 3},
                    {"g": "b", "x": None},
                ],
            )
        ],
        Program(
            "prog-mean-aggregate",
            4,
            [
                {"op": "groupby", "keys": ["g"], "aggs": [{"column": "x", "func": "mean", "as": "mean_x"}]},
                {"op": "sort", "columns": ["g"], "ascending": True},
            ],
        ),
    )

    row = run_loaded_case(case, MEAN_AGG_BACKENDS, save_artifact=False)

    assert row["status"] == "ok"
    assert row["findings"] == []
    assert row["normalized"]["polars_streaming"]["status"] == "ok"
    assert ["a", 2.0] in row["normalized"]["sqlite"]["rows"]
    assert ["b", None] in row["normalized"]["sqlite"]["rows"]


@pytest.mark.skipif(
    any(importlib.util.find_spec(name) is None for name in MEAN_AGG_BACKEND_PACKAGES),
    reason="partitioned running-sum backends are not installed",
)
def test_run_loaded_case_supports_partitioned_running_sum_semantics():
    case = generate_case(126, profile="partitioned_running_sum")

    row = run_loaded_case(case, MEAN_AGG_BACKENDS, save_artifact=False)

    assert row["status"] == "ok"
    assert row["findings"] == []
    assert row["normalized"]["polars_streaming"]["status"] == "ok"
    assert row["normalized"]["duckdb"]["rows"] == row["normalized"]["sqlite"]["rows"]


@pytest.mark.skipif(
    any(importlib.util.find_spec(name) is None for name in PATH_KEYED_PICK_PACKAGES),
    reason="path keyed-pick backends are not installed",
)
def test_run_loaded_case_supports_path_basename_keyed_pick_semantics_without_duckdb():
    case = generate_case(107, profile="path_basename_keyed_pick")
    row = run_loaded_case(case, PATH_KEYED_PICK_BACKENDS, save_artifact=False)

    assert row["status"] == "ok"
    assert row["findings"] == []
    assert row["normalized"]["sqlite"]["rows"] == row["normalized"]["datafusion"]["rows"]


@pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None or importlib.util.find_spec("duckdb") is None,
    reason="pandas/duckdb backends are not installed",
)
def test_run_loaded_case_flags_duckdb_path_projection_keyed_pick_candidate():
    case = generate_case(107, profile="path_basename_keyed_pick")
    config = ExperimentConfig(enable_artifact=False, enable_replay_bug=True)
    row = run_loaded_case(case, ["pandas", "sqlite", "duckdb"], config=config, save_artifact=False)

    assert row["status"] == "bug"
    assert row["findings"]
    finding = row["findings"][0]
    assert finding["root_cause"] == "path_projection_keyed_pick"
    assert finding["suspicious_backends"] == ["duckdb"]
    assert finding["triage_verdict"] == "candidate_implementation_bug"
    assert finding["discovery_origin"] == "issue_replay"


@pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None or importlib.util.find_spec("duckdb") is None,
    reason="pandas/duckdb backends are not installed",
)
def test_run_loaded_case_flags_duckdb_float_literal_precision_candidate():
    case = generate_case(135, profile="duckdb_float_literal_precision")
    config = ExperimentConfig(enable_artifact=False, enable_replay_bug=True)
    row = run_loaded_case(case, ["pandas", "sqlite", "duckdb"], config=config, save_artifact=False)

    assert row["status"] == "bug"
    assert row["findings"]
    finding = row["findings"][0]
    assert finding["root_cause"] == "duckdb_float_literal_precision"
    assert finding["suspicious_backends"] == ["duckdb"]
    assert finding["triage_verdict"] == "candidate_implementation_bug"
    assert finding["discovery_origin"] == "issue_replay"


@pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None or importlib.util.find_spec("polars") is None,
    reason="pandas/polars backends are not installed",
)
def test_run_loaded_case_flags_polars_timestamp_precision_filter_candidate():
    case = generate_case(136, profile="polars_timestamp_precision_filter")
    config = ExperimentConfig(enable_artifact=False, enable_replay_bug=True)
    row = run_loaded_case(case, ["pandas", "polars", "polars_lazy"], config=config, save_artifact=False)

    assert row["status"] == "bug"
    assert row["findings"]
    finding = row["findings"][0]
    assert finding["root_cause"] == "polars_timestamp_precision_filter"
    assert finding["suspicious_backends"] == ["polars", "polars_lazy"]
    assert finding["triage_verdict"] == "candidate_implementation_bug"
    assert finding["discovery_origin"] == "issue_replay"


@pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None or importlib.util.find_spec("duckdb") is None,
    reason="pandas/duckdb backends are not installed",
)
def test_run_loaded_case_classifies_bool_reduction_skipna_probe_when_backend_mismatches(monkeypatch):
    from datadiff.backends import pandas_backend

    monkeypatch.setattr(pandas_backend, "_pandas_bool_reduction_skipna_mismatch", lambda pd: True)
    case = generate_case(151, profile="pandas_bool_reduction_skipna_semantics")

    row = run_loaded_case(case, ["pandas", "duckdb", "sqlite"], save_artifact=False)

    assert row["status"] == "bug"
    assert row["findings"]
    assert row["findings"][0]["root_cause"] == "pandas_bool_reduction_skipna_semantics"
    assert row["findings"][0]["suspicious_backends"] == ["pandas"]


@pytest.mark.skipif(
    importlib.util.find_spec("pyarrow") is None,
    reason="pyarrow backend is not installed",
)
def test_run_loaded_case_detects_pyarrow_run_end_null_compute_probe():
    case = generate_case(146, profile="pyarrow_run_end_null_compute_semantics")

    row = run_loaded_case(case, ["pyarrow", "duckdb", "sqlite"], save_artifact=False)

    assert row["status"] == "bug"
    assert row["findings"]
    assert row["findings"][0]["root_cause"] == "pyarrow_run_end_null_compute_semantics"
    assert row["findings"][0]["suspicious_backends"] == ["pyarrow"]


@pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None or importlib.util.find_spec("polars") is None,
    reason="pandas/polars backends are not installed",
)
def test_run_loaded_case_preserves_polars_large_ints_through_normalizer():
    case = Case(
        "case-polars-large-int-groupby",
        1,
        [
            TableData(
                "t0",
                [
                    ColumnSpec("id", "int", nullable=False),
                    ColumnSpec("g", "str"),
                    ColumnSpec("x", "int"),
                    ColumnSpec("flag", "bool"),
                ],
                [
                    {"id": 0, "g": "alpha", "x": 9007199254740991, "flag": True},
                    {"id": 1, "g": "alpha", "x": 9007199254740992, "flag": False},
                    {"id": 2, "g": "alpha", "x": 9007199254740993, "flag": None},
                    {"id": 3, "g": "beta", "x": -9007199254740991, "flag": True},
                    {"id": 4, "g": "beta", "x": -9007199254740992, "flag": False},
                    {"id": 5, "g": None, "x": 0, "flag": None},
                    {"id": 6, "g": None, "x": 42, "flag": True},
                    {"id": 7, "g": "gamma", "x": None, "flag": False},
                    {"id": 9, "g": "delta", "x": -9007199254740995, "flag": None},
                ],
            )
        ],
        Program(
            "prog-polars-large-int-groupby",
            1,
            [
                {
                    "op": "groupby",
                    "keys": ["g"],
                    "aggs": [
                        {"column": "x", "func": "count", "as": "x_seen_count"},
                        {"column": "x", "func": "min", "as": "x_min_value"},
                        {"column": "x", "func": "max", "as": "x_max_value"},
                        {"column": "flag", "func": "count", "as": "flag_seen_count"},
                    ],
                },
                {
                    "op": "sort",
                    "keys": [
                        {"column": "x_max_value", "ascending": False, "nulls": "last"},
                        {"column": "g", "ascending": True, "nulls": "first"},
                        {"column": "flag_seen_count", "ascending": True, "nulls": "last"},
                        {"column": "x_min_value", "ascending": True, "nulls": "last"},
                        {"column": "x_seen_count", "ascending": True, "nulls": "last"},
                    ],
                },
                {"op": "limit", "n": 5},
            ],
        ),
    )

    row = run_loaded_case(case, ["pandas", "polars", "polars_lazy"], save_artifact=False)

    assert row["status"] == "ok"
    assert row["findings"] == []
    expected_alpha = [2, "alpha", 9007199254740993, 9007199254740991, 3]
    expected_delta = [0, "delta", -9007199254740995, -9007199254740995, 1]
    for backend in ["pandas", "polars", "polars_lazy"]:
        assert expected_alpha in row["normalized"][backend]["rows"]
        assert expected_delta in row["normalized"][backend]["rows"]


def test_run_loaded_case_recheck_marks_non_reproducible_candidate(monkeypatch):
    case = Case(
        "case-flaky",
        1,
        [TableData("t0", [ColumnSpec("x", "int")], [{"x": 1}])],
        Program("prog-flaky", 1, [{"op": "select", "columns": ["x"]}]),
    )
    calls = {"evaluate": 0}

    def fake_execute_case(*args, **kwargs):
        return {}, {
            "left": NormalizedResult("left", "ok", ["x"], [[1]]),
            "right": NormalizedResult("right", "ok", ["x"], [[1]]),
        }

    def fake_evaluate_case(*args, **kwargs):
        calls["evaluate"] += 1
        if calls["evaluate"] == 1:
            return [
                Finding(
                    finding_id="finding-flaky",
                    kind="semantic_output_mismatch",
                    severity="critical",
                    suspicious_backends=["right"],
                    evidence="first pass only",
                    signature="flaky",
                    root_cause="filter_predicate",
                    mismatch_class="row_count",
                )
            ]
        return []

    def fake_annotate_findings(case, findings, **kwargs):
        for finding in findings:
            finding.triage_verdict = "candidate_implementation_bug"
            finding.paper_status = "candidate_bug_needs_external_confirmation"
            finding.triage_confidence = "high"

    monkeypatch.setattr(runner_module, "_execute_case", fake_execute_case)
    monkeypatch.setattr(runner_module, "evaluate_case", fake_evaluate_case)
    monkeypatch.setattr(runner_module, "annotate_findings", fake_annotate_findings)

    row = run_loaded_case(
        case,
        ["left", "right"],
        config=ExperimentConfig(candidate_recheck_count=1),
        save_artifact=False,
        target_specs=[],
    )

    assert row["status"] == "ok"
    assert row["candidate_recheck"]["enabled"] is True
    assert row["candidate_recheck"]["non_reproduced_keys"] == [
        "semantic_output_mismatch:filter_predicate@right:row_count"
    ]
    assert row["findings"][0]["triage_verdict"] == "non_reproducible_candidate"
    assert row["findings"][0]["false_positive"] is True
    assert row["findings"][0]["false_positive_reason"] == "candidate_not_reproduced_on_immediate_recheck"


@pytest.mark.skipif(
    any(name != "sqlite" and importlib.util.find_spec(name) is None for name in REQUIRED_BACKENDS),
    reason="data backends are not installed",
)
def test_run_loaded_case_supports_join_expressions_and_multi_agg():
    case = Case(
        "case-join-expr",
        99,
        [
            TableData(
                "t0",
                [
                    ColumnSpec("id", "int", nullable=False),
                    ColumnSpec("g", "str"),
                    ColumnSpec("x", "int"),
                ],
                [
                    {"id": 1, "g": "Alpha", "x": 2},
                    {"id": 2, "g": "中文", "x": None},
                    {"id": 3, "g": None, "x": 5},
                ],
            ),
            TableData(
                "t1",
                [
                    ColumnSpec("id", "int", nullable=False),
                    ColumnSpec("j", "int"),
                    ColumnSpec("tag", "str"),
                ],
                [
                    {"id": 1, "j": 10, "tag": "One"},
                    {"id": 1, "j": 20, "tag": "Two"},
                    {"id": 3, "j": None, "tag": "Three"},
                ],
            ),
        ],
        Program(
            "prog-join-expr",
            99,
            [
                {"op": "join", "table": "t1", "left_on": "id", "right_on": "id", "how": "left"},
                {"op": "mutate", "column": "x_mul", "expr": {"kind": "arith_const", "source": "x", "op": "mul", "value": 2}},
                {"op": "mutate", "column": "g_len", "expr": {"kind": "string_length", "source": "g"}},
                {"op": "mutate", "column": "tag_l", "expr": {"kind": "string_lower", "source": "tag"}},
                {
                    "op": "groupby",
                    "keys": ["id"],
                    "aggs": [
                        {"column": "x_mul", "func": "sum", "as": "sum_x_mul"},
                        {"column": "g_len", "func": "max", "as": "max_g_len"},
                        {"column": "j", "func": "count", "as": "count_j"},
                    ],
                },
            ],
        ),
    )
    row = run_loaded_case(case, REQUIRED_BACKENDS, save_artifact=False)
    assert row["status"] == "ok"
    assert set(row["normalized"]) == set(REQUIRED_BACKENDS)


@pytest.mark.skipif(
    any(name != "sqlite" and importlib.util.find_spec(name) is None for name in REQUIRED_BACKENDS),
    reason="data backends are not installed",
)
def test_run_loaded_case_supports_null_aware_truth_filter():
    case = generate_case(123, profile="join_null_truth_filter")

    row = run_loaded_case(case, REQUIRED_BACKENDS, save_artifact=False)

    assert row["status"] == "ok"
    assert set(row["normalized"]) == set(REQUIRED_BACKENDS)
    assert {tuple(tuple(item) for item in result["rows"]) for result in row["normalized"].values()} == {
        (("a", 1, 100, "p"), ("c", 3, None, None), ("d", 4, None, None))
    }


@pytest.mark.skipif(
    any(name != "sqlite" and importlib.util.find_spec(name) is None for name in REQUIRED_BACKENDS),
    reason="data backends are not installed",
)
def test_run_loaded_case_pandas_join_nulls_do_not_pass_string_neq_filter():
    case = Case(
        "case-join-null-string-filter",
        223,
        [
            TableData(
                "t0",
                [ColumnSpec("id", "int", nullable=False), ColumnSpec("label", "str")],
                [{"id": 1, "label": "matched"}, {"id": 2, "label": "unmatched"}],
            ),
            TableData(
                "t1",
                [ColumnSpec("id", "int", nullable=False), ColumnSpec("tag", "str")],
                [{"id": 1, "tag": "keep"}],
            ),
        ],
        Program(
            "prog-join-null-string-filter",
            223,
            [
                {"op": "join", "table": "t1", "left_on": "id", "right_on": "id", "how": "left"},
                {"op": "filter", "column": "tag", "cmp": "!=", "value": ""},
                {"op": "select", "columns": ["id", "label", "tag"]},
            ],
        ),
    )

    row = run_loaded_case(case, REQUIRED_BACKENDS, save_artifact=False)

    assert row["status"] == "ok"
    assert {tuple(tuple(item) for item in result["rows"]) for result in row["normalized"].values()} == {
        ((1, "matched", "keep"),)
    }


@pytest.mark.skipif(
    any(name != "sqlite" and importlib.util.find_spec(name) is None for name in REQUIRED_BACKENDS),
    reason="data backends are not installed",
)
def test_run_loaded_case_pandas_empty_filter_preserves_columns_for_groupby():
    case = Case(
        "case-empty-filter-groupby",
        224,
        [
            TableData(
                "t0",
                [
                    ColumnSpec("id", "int", nullable=False),
                    ColumnSpec("tag", "str"),
                    ColumnSpec("x", "int"),
                ],
                [{"id": 1, "tag": "a", "x": 10}],
            )
        ],
        Program(
            "prog-empty-filter-groupby",
            224,
            [
                {"op": "filter", "column": "id", "cmp": ">", "value": 100},
                {"op": "groupby", "keys": ["tag"], "aggs": [{"column": "x", "func": "sum", "as": "sum_x"}]},
                {"op": "select", "columns": ["sum_x"]},
            ],
        ),
    )

    row = run_loaded_case(case, REQUIRED_BACKENDS, save_artifact=False)

    assert row["status"] == "ok"
    assert {tuple(tuple(item) for item in result["rows"]) for result in row["normalized"].values()} == {()}


@pytest.mark.skipif(
    any(name != "sqlite" and importlib.util.find_spec(name) is None for name in REQUIRED_BACKENDS),
    reason="data backends are not installed",
)
def test_run_loaded_case_pandas_empty_tuple_absence_preserves_columns():
    case = Case(
        "case-empty-tuple-absence-schema",
        225,
        [
            TableData(
                "t0",
                [
                    ColumnSpec("id", "int", nullable=False),
                    ColumnSpec("flag", "bool"),
                    ColumnSpec("x", "int"),
                ],
                [{"id": 1, "flag": True, "x": 10}],
            ),
            TableData(
                "t1",
                [ColumnSpec("id", "int", nullable=False)],
                [{"id": 1}],
            ),
        ],
        Program(
            "prog-empty-tuple-absence-schema",
            225,
            [{"op": "tuple_absence_filter", "columns": ["id"], "table": "t1", "right_columns": ["id"]}],
        ),
    )

    row = run_loaded_case(case, REQUIRED_BACKENDS, save_artifact=False)

    assert row["status"] == "ok"
    assert {tuple(result["columns"]) for result in row["normalized"].values()} == {("flag", "id", "x")}
    assert {tuple(tuple(item) for item in result["rows"]) for result in row["normalized"].values()} == {()}


@pytest.mark.skipif(
    any(name != "sqlite" and importlib.util.find_spec(name) is None for name in REQUIRED_BACKENDS),
    reason="data backends are not installed",
)
def test_run_loaded_case_drops_right_join_keys_for_mismatched_key_names():
    case = Case(
        "case-mismatched-join-key-schema",
        221,
        [
            TableData(
                "t0",
                [
                    ColumnSpec("fromnode", "int", nullable=False),
                    ColumnSpec("tonode", "int", nullable=False),
                ],
                [{"fromnode": 1, "tonode": 10}, {"fromnode": 2, "tonode": 20}],
            ),
            TableData(
                "t1",
                [
                    ColumnSpec("p6_fromnode", "int", nullable=False),
                    ColumnSpec("p6_tonode", "int", nullable=False),
                ],
                [{"p6_fromnode": 10, "p6_tonode": 100}, {"p6_fromnode": 20, "p6_tonode": 200}],
            ),
            TableData(
                "t2",
                [
                    ColumnSpec("p4_fromnode", "int", nullable=False),
                    ColumnSpec("p4_tonode_join", "int", nullable=False),
                    ColumnSpec("p4_tonode", "int", nullable=False),
                ],
                [
                    {"p4_fromnode": 9, "p4_tonode_join": 1, "p4_tonode": 900},
                    {"p4_fromnode": 8, "p4_tonode_join": 2, "p4_tonode": 800},
                ],
            ),
        ],
        Program(
            "prog-mismatched-join-key-schema",
            221,
            [
                {
                    "op": "join",
                    "table": "t1",
                    "left_on": "tonode",
                    "right_on": "p6_fromnode",
                    "how": "inner",
                },
                {
                    "op": "join",
                    "table": "t2",
                    "left_on": "fromnode",
                    "right_on": "p4_tonode_join",
                    "how": "inner",
                },
            ],
        ),
    )

    row = run_loaded_case(case, REQUIRED_BACKENDS, save_artifact=False)

    assert row["status"] == "ok"
    assert set(row["normalized"]) == set(REQUIRED_BACKENDS)
    for result in row["normalized"].values():
        assert result["columns"] == ["fromnode", "p4_fromnode", "p4_tonode", "p6_tonode", "tonode"]
        assert "p6_fromnode" not in result["columns"]
        assert "p4_tonode_join" not in result["columns"]
    assert {tuple(tuple(item) for item in result["rows"]) for result in row["normalized"].values()} == {
        ((1, 9, 900, 100, 10), (2, 8, 800, 200, 20))
    }


@pytest.mark.skipif(
    any(name != "sqlite" and importlib.util.find_spec(name) is None for name in REQUIRED_BACKENDS),
    reason="data backends are not installed",
)
def test_run_loaded_case_supports_post_groupby_join_and_global_aggregate():
    case = Case(
        "case-post-groupby-join-aggregate",
        220,
        [
            TableData(
                "t0",
                [ColumnSpec("id", "int", nullable=False), ColumnSpec("x", "int")],
                [{"id": 1, "x": 2}, {"id": 1, "x": 3}, {"id": 2, "x": 5}],
            ),
            TableData(
                "t1",
                [ColumnSpec("id", "int", nullable=False), ColumnSpec("z", "int")],
                [{"id": 1, "z": 10}, {"id": 2, "z": 20}],
            ),
        ],
        Program(
            "prog-post-groupby-join-aggregate",
            220,
            [
                {"op": "groupby", "keys": ["id"], "aggs": [{"column": "x", "func": "sum", "as": "sum_x"}]},
                {"op": "join", "table": "t1", "left_on": "id", "right_on": "id", "how": "inner"},
                {
                    "op": "aggregate",
                    "aggs": [
                        {"column": "sum_x", "func": "sum", "as": "total_x"},
                        {"column": "z", "func": "count", "as": "matched_groups"},
                    ],
                },
            ],
        ),
    )

    row = run_loaded_case(case, REQUIRED_BACKENDS, save_artifact=False)

    assert row["status"] == "ok"
    assert set(row["normalized"]) == set(REQUIRED_BACKENDS)
    assert {tuple(tuple(item) for item in result["rows"]) for result in row["normalized"].values()} == {
        ((2, 10),)
    }


def _ordered_groupby_sort_projection_case() -> Case:
    return Case(
        "case-ordered-groupby-sort-select-drops-key",
        1064,
        [
            TableData(
                "t0",
                [
                    ColumnSpec("id", "int", nullable=False),
                    ColumnSpec("g", "str", nullable=False),
                    ColumnSpec("x", "int"),
                    ColumnSpec("z", "int"),
                ],
                [
                    {"id": 0, "g": "a", "x": 1, "z": 1},
                    {"id": 1, "g": "b", "x": 2, "z": 2},
                    {"id": 2, "g": "b", "x": 0, "z": 0},
                    {"id": 3, "g": "d", "x": 0, "z": 1},
                    {"id": 4, "g": "a", "x": None, "z": 1},
                    {"id": 5, "g": "b", "x": 0, "z": 8},
                    {"id": 6, "g": "c", "x": 2, "z": 1},
                    {"id": 7, "g": "d", "x": -2, "z": -1},
                ],
            )
        ],
        Program(
            "prog-ordered-groupby-sort-select-drops-key",
            1064,
            [
                {
                    "op": "sort",
                    "keys": [
                        {"column": "x", "ascending": True, "nulls": "last"},
                        {"column": "g", "ascending": True, "nulls": "last"},
                        {"column": "id", "ascending": True, "nulls": "last"},
                        {"column": "z", "ascending": True, "nulls": "last"},
                    ],
                },
                {"op": "groupby", "keys": ["g"], "aggs": [{"column": "x", "func": "max", "as": "max_x"}]},
                {"op": "select", "columns": ["g", "max_x"]},
                {
                    "op": "sort",
                    "keys": [
                        {"column": "max_x", "ascending": False, "nulls": "last"},
                        {"column": "g", "ascending": True, "nulls": "last"},
                    ],
                },
                {"op": "select", "columns": ["g"]},
            ],
        ),
    )


@pytest.mark.skipif(
    any(name != "sqlite" and importlib.util.find_spec(name) is None for name in SQL_ORDER_BACKENDS),
    reason="SQL order-preservation backends are not installed",
)
def test_sql_backends_preserve_sort_when_select_drops_sort_key():
    row = run_loaded_case(_ordered_groupby_sort_projection_case(), SQL_ORDER_BACKENDS, save_artifact=False)

    assert row["findings"] == []
    assert {tuple(tuple(item) for item in result["rows"]) for result in row["normalized"].values()} == {
        (("b",), ("c",), ("a",), ("d",))
    }


@pytest.mark.skipif(
    any(name != "sqlite" and importlib.util.find_spec(name) is None for name in SQL_ORDER_BACKENDS),
    reason="SQL order-preservation backends are not installed",
)
def test_sql_backends_apply_limit_after_select_drops_sort_key():
    case = Case(
        "case-sort-select-limit-drops-key",
        1065,
        [
            TableData(
                "t0",
                [ColumnSpec("x", "int"), ColumnSpec("s", "str")],
                [{"x": 1, "s": "a"}, {"x": 3, "s": "c"}, {"x": 2, "s": "b"}],
            )
        ],
        Program(
            "prog-sort-select-limit-drops-key",
            1065,
            [
                {"op": "sort", "columns": ["x"], "ascending": False},
                {"op": "select", "columns": ["s"]},
                {"op": "limit", "n": 1},
            ],
        ),
    )

    row = run_loaded_case(case, SQL_ORDER_BACKENDS, save_artifact=False)

    assert row["findings"] == []
    assert {tuple(tuple(item) for item in result["rows"]) for result in row["normalized"].values()} == {
        (("c",),)
    }


@pytest.mark.skipif(
    any(name != "sqlite" and importlib.util.find_spec(name) is None for name in SQL_ORDER_BACKENDS),
    reason="SQL order-preservation backends are not installed",
)
def test_sql_backends_apply_offset_then_limit_after_select_drops_sort_key():
    case = Case(
        "case-sort-select-offset-limit-drops-key",
        1066,
        [
            TableData(
                "t0",
                [ColumnSpec("x", "int"), ColumnSpec("s", "str")],
                [{"x": 1, "s": "a"}, {"x": 3, "s": "c"}, {"x": 2, "s": "b"}],
            )
        ],
        Program(
            "prog-sort-select-offset-limit-drops-key",
            1066,
            [
                {"op": "sort", "columns": ["x"], "ascending": False},
                {"op": "select", "columns": ["s"]},
                {"op": "offset", "n": 1},
                {"op": "limit", "n": 1},
            ],
        ),
    )

    row = run_loaded_case(case, SQL_ORDER_BACKENDS, save_artifact=False)

    assert row["findings"] == []
    assert {tuple(tuple(item) for item in result["rows"]) for result in row["normalized"].values()} == {
        (("b",),)
    }


@pytest.mark.skipif(
    any(importlib.util.find_spec(name) is None for name in DATAFUSION_BACKENDS),
    reason="DataFusion order-preservation backends are not installed",
)
def test_datafusion_backend_preserves_sort_when_select_drops_sort_key():
    row = run_loaded_case(_ordered_groupby_sort_projection_case(), DATAFUSION_BACKENDS, save_artifact=False)

    assert row["findings"] == []
    assert {tuple(tuple(item) for item in result["rows"]) for result in row["normalized"].values()} == {
        (("b",), ("c",), ("a",), ("d",))
    }


@pytest.mark.skipif(
    any(importlib.util.find_spec(name) is None for name in LATEST_ALL_ENGINE_PACKAGES),
    reason="latest all-engine methodology backends are not installed",
)
def test_latest_all_engines_methodology_case_uses_common_semantics():
    backends = resolve_target_backends(target_suite="latest_all_engines")
    case = Case(
        "case-latest-all-engines-methodology",
        9101,
        [
            TableData(
                "t0",
                [
                    ColumnSpec("id", "int", nullable=False),
                    ColumnSpec("g", "str", nullable=False),
                    ColumnSpec("x", "int", nullable=False),
                ],
                [
                    {"id": 1, "g": "Alpha", "x": 2},
                    {"id": 2, "g": "Beta", "x": 4},
                    {"id": 3, "g": "Gamma", "x": 5},
                    {"id": 4, "g": "Beta", "x": 1},
                ],
            ),
            TableData(
                "t1",
                [
                    ColumnSpec("id", "int", nullable=False),
                    ColumnSpec("j", "int", nullable=False),
                    ColumnSpec("tag", "str", nullable=False),
                ],
                [
                    {"id": 1, "j": 10, "tag": "One"},
                    {"id": 2, "j": 20, "tag": "Two"},
                    {"id": 3, "j": -1, "tag": "Drop"},
                    {"id": 4, "j": 0, "tag": "Two"},
                ],
            ),
        ],
        Program(
            "prog-latest-all-engines-methodology",
            9101,
            [
                {"op": "join", "table": "t1", "left_on": "id", "right_on": "id", "how": "inner"},
                {"op": "filter", "column": "j", "cmp": ">=", "value": 0},
                {"op": "mutate", "column": "x_plus", "expr": {"kind": "add_const", "source": "x", "value": 1}},
                {"op": "mutate", "column": "tag_norm", "expr": {"kind": "string_lower", "source": "tag"}},
                {
                    "op": "groupby",
                    "keys": ["tag_norm"],
                    "aggs": [
                        {"column": "x_plus", "func": "sum", "as": "sum_x_plus"},
                        {"column": "j", "func": "count", "as": "count_j"},
                    ],
                },
                {"op": "select", "columns": ["tag_norm", "sum_x_plus", "count_j"]},
                {"op": "sort", "columns": ["tag_norm", "sum_x_plus", "count_j"], "ascending": True},
                {"op": "limit", "n": 10},
            ],
        ),
    )

    row = run_loaded_case(case, backends, save_artifact=False)

    assert row["status"] == "ok"
    assert row["findings"] == []
    assert set(row["normalized"]) == set(backends)
    assert {tuple(result["columns"]) for result in row["normalized"].values()} == {
        ("count_j", "sum_x_plus", "tag_norm")
    }
    assert {tuple(tuple(item) for item in result["rows"]) for result in row["normalized"].values()} == {
        ((1, 3, "one"), (2, 7, "two"))
    }


@pytest.mark.skipif(
    any(importlib.util.find_spec(name) is None for name in ["pandas", "duckdb", "datafusion", "pyarrow"]),
    reason="datafusion test backends are not installed",
)
def test_datafusion_backend_matches_common_join_groupby_case():
    case = Case(
        "case-datafusion-join-groupby",
        777,
        [
            TableData(
                "t0",
                [
                    ColumnSpec("id", "int", nullable=False),
                    ColumnSpec("g", "str"),
                    ColumnSpec("x", "int"),
                ],
                [
                    {"id": 1, "g": "alpha", "x": 2},
                    {"id": 2, "g": None, "x": None},
                    {"id": 3, "g": "beta", "x": 5},
                ],
            ),
            TableData(
                "t1",
                [
                    ColumnSpec("id", "int", nullable=False),
                    ColumnSpec("z", "float"),
                    ColumnSpec("tag", "str"),
                ],
                [
                    {"id": 1, "z": 10.0, "tag": "One"},
                    {"id": 1, "z": 20.0, "tag": "Two"},
                    {"id": 3, "z": None, "tag": "Three"},
                ],
            ),
        ],
        Program(
            "prog-datafusion-join-groupby",
            777,
            [
                {"op": "join", "table": "t1", "left_on": "id", "right_on": "id", "how": "left"},
                {"op": "mutate", "column": "x_div", "expr": {"kind": "arith_const", "source": "x", "op": "div", "value": 2}},
                {
                    "op": "groupby",
                    "keys": ["g"],
                    "aggs": [
                        {"column": "x_div", "func": "sum", "as": "sum_x_div"},
                        {"column": "z", "func": "min", "as": "min_z"},
                    ],
                },
            ],
        ),
    )

    row = run_loaded_case(case, DATAFUSION_BACKENDS, save_artifact=False)

    assert row["status"] == "ok"
    assert set(row["normalized"]) == set(DATAFUSION_BACKENDS)


@pytest.mark.skipif(
    any(importlib.util.find_spec(name) is None for name in ["pandas", "duckdb", "datafusion", "pyarrow"]),
    reason="datafusion test backends are not installed",
)
def test_datafusion_setop_all_duplicate_probe_runs_and_classifies_mismatch():
    case = generate_case(137, profile="datafusion_setop_all_duplicate_count")

    row = run_loaded_case(
        case,
        DATAFUSION_BACKENDS,
        config=ExperimentConfig(enable_metamorphic_oracle=False),
        save_artifact=False,
    )

    assert row["normalized"]["pandas"]["rows"] == [[False]]
    assert row["normalized"]["duckdb"]["rows"] == [[False]]
    assert row["normalized"]["datafusion"]["rows"] in ([[False]], [[True]])
    if row["normalized"]["datafusion"]["rows"] == [[True]]:
        assert row["status"] == "bug"
        assert row["findings"]
        assert row["findings"][0]["root_cause"] == "datafusion_setop_all_duplicate_count"
        assert row["findings"][0]["triage_verdict"] == "candidate_implementation_bug"
        assert row["findings"][0]["suspicious_backends"] == ["datafusion"]
    else:
        assert row["status"] == "ok"
        assert row["findings"] == []


@pytest.mark.skipif(
    any(importlib.util.find_spec(name) is None for name in ["pandas", "duckdb", "datafusion", "pyarrow"]),
    reason="datafusion test backends are not installed",
)
def test_datafusion_backend_applies_limit_to_sorted_rows():
    case = Case(
        "case-datafusion-sort-limit",
        778,
        [
            TableData(
                "t0",
                [ColumnSpec("x", "int")],
                [{"x": 1}, {"x": 5}, {"x": 3}, {"x": 2}],
            )
        ],
        Program(
            "prog-datafusion-sort-limit",
            778,
            [
                {"op": "sort", "columns": ["x"], "ascending": False},
                {"op": "limit", "n": 2},
            ],
        ),
    )

    row = run_loaded_case(
        case,
        DATAFUSION_BACKENDS,
        config=ExperimentConfig(enable_metamorphic_oracle=False),
        save_artifact=False,
    )

    assert row["status"] == "ok"
    assert {tuple(tuple(r) for r in result["rows"]) for result in row["normalized"].values()} == {
        ((5,), (3,))
    }


@pytest.mark.skipif(
    any(name != "sqlite" and importlib.util.find_spec(name) is None for name in REQUIRED_BACKENDS),
    reason="data backends are not installed",
)
def test_run_loaded_case_supports_per_column_sort_null_order():
    case = Case(
        "case-per-column-sort-null-order",
        3015,
        [
            TableData(
                "t0",
                [
                    ColumnSpec("a", "int"),
                    ColumnSpec("b", "int"),
                    ColumnSpec("label", "str", nullable=False),
                ],
                [
                    {"a": None, "b": 1, "label": "n1"},
                    {"a": None, "b": None, "label": "nnull"},
                    {"a": None, "b": 5, "label": "n5"},
                    {"a": 1, "b": 7, "label": "a1b7"},
                    {"a": 1, "b": None, "label": "a1null"},
                    {"a": 0, "b": 2, "label": "a0b2"},
                ],
            )
        ],
        Program(
            "prog-per-column-sort-null-order",
            3015,
            [
                {
                    "op": "sort",
                    "keys": [
                        {"column": "a", "ascending": True, "nulls": "first"},
                        {"column": "b", "ascending": False, "nulls": "last"},
                        {"column": "label", "ascending": True, "nulls": "last"},
                    ],
                },
                {"op": "limit", "n": 4},
            ],
        ),
    )

    row = run_loaded_case(case, REQUIRED_BACKENDS, save_artifact=False)

    assert row["status"] == "ok"
    assert set(row["normalized"]) == set(REQUIRED_BACKENDS)
    assert {frozenset(tuple(item) for item in result["rows"]) for result in row["normalized"].values()} == {
        frozenset(
            {
                (None, None, "nnull"),
                (None, 1, "n1"),
                (None, 5, "n5"),
                (0, 2, "a0b2"),
            }
        )
    }


@pytest.mark.skipif(
    any(name != "sqlite" and importlib.util.find_spec(name) is None for name in REQUIRED_BACKENDS),
    reason="data backends are not installed",
)
def test_run_loaded_case_supports_sort_offset_limit():
    case = Case(
        "case-sort-offset-limit",
        22656,
        [
            TableData(
                "t0",
                [
                    ColumnSpec("x", "int"),
                    ColumnSpec("label", "str", nullable=False),
                ],
                [
                    {"x": 4, "label": "d"},
                    {"x": 1, "label": "a"},
                    {"x": 3, "label": "c"},
                    {"x": 2, "label": "b"},
                    {"x": 5, "label": "e"},
                ],
            )
        ],
        Program(
            "prog-sort-offset-limit",
            22656,
            [
                {"op": "sort", "columns": ["x"], "ascending": True},
                {"op": "offset", "n": 2},
                {"op": "limit", "n": 2},
            ],
        ),
    )

    row = run_loaded_case(case, REQUIRED_BACKENDS, save_artifact=False)

    assert row["status"] == "ok"
    assert {tuple(tuple(r) for r in result["rows"]) for result in row["normalized"].values()} == {
        (("c", 3), ("d", 4))
    }


@pytest.mark.skipif(
    any(importlib.util.find_spec(name) is None for name in ["pandas", "duckdb", "datafusion", "pyarrow"]),
    reason="datafusion test backends are not installed",
)
def test_datafusion_backend_preserves_sort_through_select_before_limit():
    case = Case(
        "case-datafusion-sort-select-limit",
        780,
        [
            TableData(
                "t0",
                [
                    ColumnSpec("id", "int", nullable=False),
                    ColumnSpec("y", "float"),
                    ColumnSpec("flag", "bool"),
                    ColumnSpec("s", "str"),
                ],
                [
                    {"id": 0, "y": -1.0, "flag": True, "s": "alpha"},
                    {"id": 16, "y": None, "flag": True, "s": "alpha"},
                    {"id": 0, "y": 0.5, "flag": True, "s": "wtiApSjd"},
                ],
            ),
            TableData(
                "t1",
                [
                    ColumnSpec("id", "int", nullable=False),
                    ColumnSpec("j", "int"),
                    ColumnSpec("z", "float"),
                    ColumnSpec("tag", "str"),
                ],
                [
                    {"id": 0, "j": 2, "z": -0.5, "tag": "gamma"},
                    {"id": 0, "j": None, "z": 0.5, "tag": "beta"},
                    {"id": 0, "j": 10, "z": -1.0, "tag": "zh"},
                    {"id": 0, "j": -2, "z": 1.0, "tag": "f"},
                    {"id": 16, "j": 16, "z": 16.0, "tag": "tag_16"},
                ],
            ),
        ],
        Program(
            "prog-datafusion-sort-select-limit",
            780,
            [
                {"op": "join", "table": "t1", "left_on": "id", "right_on": "id", "how": "left"},
                {"op": "select", "columns": ["flag", "id", "j", "s", "y", "z"]},
                {"op": "sort", "columns": ["s", "flag", "id", "j", "y", "z"], "ascending": True},
                {"op": "limit", "n": 4},
            ],
        ),
    )

    row = run_loaded_case(
        case,
        DATAFUSION_BACKENDS,
        config=ExperimentConfig(enable_metamorphic_oracle=True, metamorphic_variant_limit=20),
        save_artifact=False,
    )

    assert row["status"] == "ok"
    variant = row["metamorphic"]["sort_select_commutation:swap-1-2"]
    assert variant["normalized"]["datafusion"]["rows"] == row["normalized"]["datafusion"]["rows"]


@pytest.mark.skipif(
    any(importlib.util.find_spec(name) is None for name in ["pandas", "duckdb", "pyarrow"]),
    reason="pyarrow test backends are not installed",
)
def test_pyarrow_backend_matches_common_join_groupby_case():
    case = Case(
        "case-pyarrow-join-groupby",
        779,
        [
            TableData(
                "t0",
                [
                    ColumnSpec("id", "int", nullable=False),
                    ColumnSpec("g", "str"),
                    ColumnSpec("x", "int"),
                ],
                [
                    {"id": 1, "g": "alpha", "x": 2},
                    {"id": 2, "g": None, "x": None},
                    {"id": 3, "g": "beta", "x": 5},
                ],
            ),
            TableData(
                "t1",
                [
                    ColumnSpec("id", "int", nullable=False),
                    ColumnSpec("z", "float"),
                    ColumnSpec("tag", "str"),
                ],
                [
                    {"id": 1, "z": 10.0, "tag": "One"},
                    {"id": 1, "z": 20.0, "tag": "Two"},
                    {"id": 3, "z": None, "tag": "Three"},
                ],
            ),
        ],
        Program(
            "prog-pyarrow-join-groupby",
            779,
            [
                {"op": "join", "table": "t1", "left_on": "id", "right_on": "id", "how": "left"},
                {"op": "mutate", "column": "x_div", "expr": {"kind": "arith_const", "source": "x", "op": "div", "value": 2}},
                {
                    "op": "groupby",
                    "keys": ["g"],
                    "aggs": [
                        {"column": "x_div", "func": "sum", "as": "sum_x_div"},
                        {"column": "z", "func": "min", "as": "min_z"},
                    ],
                },
                {"op": "sort", "columns": ["g", "sum_x_div", "min_z"], "ascending": True},
            ],
        ),
    )

    row = run_loaded_case(case, PYARROW_BACKENDS, save_artifact=False)

    assert row["status"] == "ok"
    assert set(row["normalized"]) == set(PYARROW_BACKENDS)


@pytest.mark.skipif(importlib.util.find_spec("polars") is None, reason="polars is not installed")
def test_polars_backend_preserves_empty_table_schema():
    case = Case(
        "case-empty-polars-schema",
        521,
        [TableData("t0", [ColumnSpec("g", "str")], [])],
        Program(
            "prog-empty-polars-schema",
            521,
            [
                {"op": "mutate", "column": "m_0", "expr": {"kind": "string_length", "source": "g"}},
                {"op": "filter", "column": "m_0", "cmp": "==", "value": -1},
            ],
        ),
    )

    row = run_loaded_case(case, ["polars"], save_artifact=False)

    assert row["normalized"]["polars"]["status"] == "ok"
    assert row["findings"] == []


@pytest.mark.skipif(importlib.util.find_spec("polars") is None, reason="polars is not installed")
def test_polars_lazy_backend_matches_polars_eager_on_common_case():
    case = generate_case(91, profile="bughunt")

    row = run_loaded_case(case, ["polars", "polars_lazy"], save_artifact=False)

    assert row["status"] == "ok"
    assert row["findings"] == []
    assert set(row["normalized"]) == {"polars", "polars_lazy"}


@pytest.mark.skipif(
    any(name != "sqlite" and importlib.util.find_spec(name) is None for name in REQUIRED_BACKENDS),
    reason="data backends are not installed",
)
def test_sql_backends_treat_mutate_as_column_replacement():
    case = Case(
        "case-replace-mutate",
        6534,
        [
            TableData(
                "t0",
                [ColumnSpec("y", "float")],
                [{"y": None}],
            )
        ],
        Program(
            "prog-replace-mutate",
            6534,
            [
                {"op": "filter", "column": "y", "cmp": ">=", "value": 0.0},
                {"op": "mutate", "column": "m_1", "expr": {"kind": "cast", "source": "y", "to": "float"}},
                {"op": "mutate", "column": "m_1", "expr": {"kind": "add_const", "source": "m_1", "value": -1}},
            ],
        ),
    )

    row = run_loaded_case(case, REQUIRED_BACKENDS, save_artifact=False)

    assert row["status"] == "ok"
    assert {tuple(result["columns"]) for result in row["normalized"].values()} == {("m_1", "y")}


@pytest.mark.skipif(
    any(name != "sqlite" and importlib.util.find_spec(name) is None for name in REQUIRED_BACKENDS),
    reason="data backends are not installed",
)
def test_run_fuzz_records_duration_and_feedback():
    run_file = run_fuzz(cases=2, seed=21, backends=REQUIRED_BACKENDS, duration_s=None)
    rows = read_jsonl(run_file)
    assert len(rows) == 2
    assert all("elapsed_s" in row for row in rows)
    assert all("stored_in_feedback_corpus" in row for row in rows)


def test_feedback_storage_skips_calibration_probe_cases():
    probe_cases = [
        Case(
            "case-probe-feedback",
            1,
            [TableData("t0", [ColumnSpec("probe_id", "int")], [{"probe_id": 0}])],
            Program(
                "prog-probe-feedback",
                1,
                [{"op": "dataset_isin_all_match_probe", "as": "dataset_isin_all_match_mismatch"}],
            ),
        ),
        Case(
            "case-sortedness-feedback",
            2,
            [TableData("t0", [ColumnSpec("x", "int")], [{"x": 1}, {"x": None}])],
            Program("prog-sortedness-feedback", 2, [{"op": "sortedness_check", "column": "x", "as": "is_sorted"}]),
        ),
        Case(
            "case-running-feedback",
            3,
            [TableData("t0", [ColumnSpec("x", "int")], [{"x": 1}, {"x": None}])],
            Program("prog-running-feedback", 3, [{"op": "running_sum", "source": "x", "as": "run_x"}]),
        ),
        Case(
            "case-tuple-absence-feedback",
            4,
            [TableData("t0", [ColumnSpec("x", "int")], [{"x": 1}, {"x": None}])],
            Program("prog-tuple-absence-feedback", 4, [{"op": "tuple_absence_filter", "columns": ["x"]}]),
        ),
    ]
    ordinary_case = Case(
        "case-ordinary-feedback",
        5,
        [TableData("t0", [ColumnSpec("x", "int")], [{"x": 1}])],
        Program("prog-ordinary-feedback", 5, [{"op": "filter", "column": "x", "cmp": ">", "value": 0}]),
    )

    for probe_case in probe_cases:
        assert runner_module._feedback_storage_decision(probe_case) == (False, "calibration_probe_case")
    assert runner_module._feedback_storage_decision(ordinary_case) == (True, "")
    assert runner_module._feedback_storage_decision(
        ordinary_case,
        candidate_source="feedback_mutation",
        seed_lineage={"depth": 1},
    ) == (False, "feedback_mutation_child")


def test_run_fuzz_can_persist_generated_cases_and_checkpoint(tmp_path):
    case_log = tmp_path / "generated.cases.jsonl"
    run_file = run_fuzz(
        cases=2,
        seed=31,
        backends=[],
        duration_s=None,
        save_cases=True,
        case_log_file=case_log,
        checkpoint_interval_s=0.0,
    )

    generated = read_jsonl(case_log)
    assert len(generated) == 2
    assert [row["seed"] for row in generated] == [31, 32]
    assert generated[0]["case"]["case_id"] == "case-00000031"

    meta = load_json(run_meta_path(run_file))
    checkpoint = load_json(Path(meta["checkpoint_file"]))
    assert meta["case_log_file"] == str(case_log)
    assert meta["executed_cases"] == 2
    assert "preflight" in meta
    assert "quality_oracles" in meta
    rows = read_jsonl(run_file)
    assert all("quality_oracles" in row for row in rows)
    assert all("preflight" in row for row in rows)
    assert generated[0]["seed_lineage"]["depth"] == 0
    assert generated[0]["mutation"]["operator"] == "generated"
    assert generated[0]["operation_combo"]["operation_count"] == len(generated[0]["case"]["program"]["operations"])
    assert rows[0]["seed_lineage"]["depth"] == 0
    assert rows[0]["mutation"]["operator"] == "generated"
    assert "operation_combo" in rows[0]
    assert "source_reward" in rows[0]
    assert checkpoint["status"] == "completed"
    assert checkpoint["next_seed"] == 33


def test_run_fuzz_records_guidance_metadata(tmp_path):
    case_log = tmp_path / "guided.cases.jsonl"
    config = ExperimentConfig(
        guidance_strategy="guided",
        guidance_candidate_pool=4,
        guidance_targets=["groupby"],
    )

    run_file = run_fuzz(cases=1, seed=41, backends=[], config=config, case_log_file=case_log)

    row = read_jsonl(run_file)[0]
    case_log_row = read_jsonl(case_log)[0]
    meta = load_json(run_meta_path(run_file))
    assert row["guidance"]["candidate_count"] == 4
    assert row["guidance"]["contributing_candidate_count"] >= 1
    assert row["guidance"]["pruned_candidate_count"] >= 0
    assert "frontier_conformance" in row["guidance"]
    assert "data_sensitivity" in row["guidance"]
    assert "path_coverage_proxy" in row["guidance"]
    assert "combo_priority" in row["guidance"]
    assert "online_weight_mean" in row["guidance"]
    assert "profile_saturation_penalty" in row["guidance"]
    assert "profile_saturation_active" in row["guidance"]
    assert "issue_replay_global_saturation_penalty" in row["guidance"]
    assert "issue_replay_global_saturation_active" in row["guidance"]
    assert "issue_inspired_source_saturation_penalty" in row["guidance"]
    assert "issue_inspired_source_saturation_active" in row["guidance"]
    assert case_log_row["candidate_pool_size"] == 4
    assert meta["guidance"]["strategy"] == "guided"
    assert meta["next_seed"] == 45


def test_run_fuzz_fresh_policy_filters_replay_profile(tmp_path):
    case_log = tmp_path / "fresh.cases.jsonl"
    config = ExperimentConfig(
        generator_profile="datafusion_setop_all_duplicate_count",
        enable_replay_bug=False,
    )

    run_file = run_fuzz(cases=1, seed=137, backends=[], config=config, case_log_file=case_log)

    row = read_jsonl(run_file)[0]
    case_log_row = read_jsonl(case_log)[0]
    meta = load_json(run_meta_path(run_file))
    assert row["candidate_source"] == "generated_fresh_fallback"
    assert row["replay_filter"]["enabled"] is True
    assert row["replay_filter"]["filtered_before_candidate"] > 0
    assert row["replay_filter"]["fallback_used"] is True
    assert row["replay_filter"]["last_skip_reason"] == "issue_replay_probe"
    assert row["case"].get("metadata", {}).get("generator_profile") != "datafusion_setop_all_duplicate_count"
    assert case_log_row["replay_filter"] == row["replay_filter"]
    assert meta["replay_bug_filter"]["enabled"] is True
    assert meta["replay_bug_filter"]["filtered_candidates"] > 0


def test_run_fuzz_fresh_bughunt_uses_generation_time_replay_gate(tmp_path):
    case_log = tmp_path / "fresh-bughunt.cases.jsonl"
    config = ExperimentConfig(generator_profile="bughunt", enable_replay_bug=False)

    run_file = run_fuzz(cases=1, seed=20, backends=[], config=config, case_log_file=case_log)

    row = read_jsonl(run_file)[0]
    case_log_row = read_jsonl(case_log)[0]
    meta = load_json(run_meta_path(run_file))
    assert meta["config"]["generator_profile"] == "bughunt"
    assert meta["effective_generator_profile"] == "bughunt"
    assert meta["replay_bug_filter"]["filtered_candidates"] > 0
    assert row["case"].get("metadata", {}).get("mixed_generator_profile") != "wide_offset_topk"
    assert row["replay_filter"]["filtered_before_candidate"] > 0
    assert row["replay_filter"]["last_skip_reason"] == "known_replay_source_issue"
    assert case_log_row["case"].get("metadata", {}).get("mixed_generator_profile") != "wide_offset_topk"


def test_run_fuzz_custom_replay_source_gate_keeps_requested_generator_profile(tmp_path):
    case_log = tmp_path / "custom-source-gate.cases.jsonl"
    config = ExperimentConfig(
        generator_profile="bughunt",
        enable_replay_bug=False,
        replay_bug_source_issues=[],
    )

    run_file = run_fuzz(cases=1, seed=20, backends=[], config=config, case_log_file=case_log)

    case_log_row = read_jsonl(case_log)[0]
    meta = load_json(run_meta_path(run_file))
    assert meta["effective_generator_profile"] == "bughunt"
    assert meta["replay_bug_filter"]["filtered_candidates"] == 0
    assert case_log_row["case"]["metadata"]["mixed_generator_profile"] == "wide_offset_topk"


def test_run_fuzz_replay_policy_allows_replay_profile(tmp_path):
    case_log = tmp_path / "replay.cases.jsonl"
    config = ExperimentConfig(
        generator_profile="datafusion_setop_all_duplicate_count",
        enable_replay_bug=True,
    )

    run_file = run_fuzz(cases=1, seed=137, backends=[], config=config, case_log_file=case_log)

    row = read_jsonl(run_file)[0]
    case_log_row = read_jsonl(case_log)[0]
    meta = load_json(run_meta_path(run_file))
    assert row["candidate_source"] == "generated"
    assert row["replay_filter"]["enabled"] is False
    assert row["replay_filter"]["filtered_before_candidate"] == 0
    assert case_log_row["case"]["metadata"]["generator_profile"] == "datafusion_setop_all_duplicate_count"
    assert meta["replay_bug_filter"]["enabled"] is False
    assert meta["replay_bug_filter"]["filtered_candidates"] == 0


def test_run_fuzz_uses_feedback_source_marker_for_candidate_source(tmp_path, monkeypatch):
    instances = []

    class FakeFeedbackState:
        def __init__(self, **kwargs):
            self.last_persisted_to_disk = False
            self.last_candidate_source = "generated"
            self.last_candidate_metadata = {}
            self.source_scheduler = None
            self.recorded_sources = []
            instances.append(self)

        def choose_case(self, seed, generated):
            self.last_candidate_source = "feedback_mutation"
            self.last_candidate_metadata = {
                "seed_lineage": {
                    "root_seed": generated.seed,
                    "parent_seed": 1,
                    "parent_case_id": "case-parent",
                    "mutation_seed": seed,
                    "depth": 1,
                },
                "mutation": {
                    "operator": "value",
                    "detail": "value:int:x",
                    "changed": True,
                },
            }
            return generated

        def record(self, case, behavior_signature, has_finding):
            return True

        def record_candidate_result(self, candidate_source, *, has_finding, is_new_behavior, preflight, **reward_signals):
            self.recorded_sources.append(candidate_source)
            return 1.25

    monkeypatch.setattr(runner_module, "FeedbackState", FakeFeedbackState)

    case_log = tmp_path / "feedback.cases.jsonl"
    config = ExperimentConfig(enable_local_source_scheduler=True)

    run_file = run_fuzz(cases=1, seed=47, backends=[], config=config, case_log_file=case_log)

    row = read_jsonl(run_file)[0]
    case_log_row = read_jsonl(case_log)[0]
    assert row["candidate_source"] == "feedback_mutation"
    assert case_log_row["candidate_source"] == "feedback_mutation"
    assert row["seed_lineage"]["parent_case_id"] == "case-parent"
    assert row["mutation"]["operator"] == "value"
    assert row["operation_combo"]["operation_count"] == len(row["case"]["program"]["operations"])
    assert row["source_reward"] == 1.25
    assert row["stored_in_feedback_corpus"] is False
    assert row["feedback_skip_reason"] == "feedback_mutation_child"
    assert case_log_row["seed_lineage"]["parent_case_id"] == "case-parent"
    assert case_log_row["mutation"]["operator"] == "value"
    assert "operation_combo" in case_log_row
    assert instances[0].recorded_sources == ["feedback_mutation"]


def test_run_fuzz_compact_log_omits_repeated_run_metadata():
    run_file = run_fuzz(cases=1, seed=51, backends=[], duration_s=None)

    row = read_jsonl(run_file)[0]
    meta = load_json(run_meta_path(run_file))
    assert run_file.name.endswith(".jsonl.gz")
    assert meta["log_level"] == "compact"
    assert "environment" not in row
    assert "targets" not in row
    assert "config" not in row
    assert "tables" not in row["case"]
    assert "normalized" in row
    assert "seed_lineage" in row
    assert "mutation" in row
    assert "operation_combo" in row


def test_run_fuzz_minimal_log_keeps_only_backend_status():
    config = ExperimentConfig(log_level="minimal")

    run_file = run_fuzz(cases=1, seed=52, backends=[], config=config)

    row = read_jsonl(run_file)[0]
    assert "normalized" not in row
    assert "raw_results" not in row
    assert row["backend_status"] == {}
    assert "frontier_conformance" in row["guidance"]


def test_run_fuzz_can_disable_run_log_compression():
    config = ExperimentConfig(compress_run_log=False)

    run_file = run_fuzz(cases=1, seed=53, backends=[], config=config)

    assert run_file.name.endswith(".jsonl")
    assert not run_file.name.endswith(".jsonl.gz")
    assert load_json(run_meta_path(run_file))["config"]["compress_run_log"] is False


def test_run_loaded_case_honors_metamorphic_variant_limit():
    case = generate_case(61, profile="bughunt")
    config = ExperimentConfig(enable_metamorphic_oracle=True, metamorphic_variant_limit=2)

    row = run_loaded_case(case, [], config=config, save_artifact=False)

    assert len(row["metamorphic"]) <= 2
    assert row["config"]["metamorphic_variant_limit"] == 2
