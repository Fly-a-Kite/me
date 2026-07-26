import importlib.util

import pytest

from datadiff.config import ExperimentConfig
from datadiff.dsl import Case, ColumnSpec, Program, TableData
from datadiff.runner import run_loaded_case


pytestmark = pytest.mark.skipif(importlib.util.find_spec("pandas") is None, reason="pandas is not installed")


def test_seeded_filter_fault_is_classified_as_candidate_bug():
    case = Case(
        "case-seeded-filter",
        1,
        [TableData("t0", [ColumnSpec("x", "int")], [{"x": 1}, {"x": 2}, {"x": 3}])],
        Program("prog-seeded-filter", 1, [{"op": "filter", "column": "x", "cmp": ">", "value": 1}]),
    )

    row = run_loaded_case(
        case,
        ["pandas", "buggy_filter"],
        config=ExperimentConfig(enable_artifact=False),
        save_artifact=False,
    )

    assert row["status"] == "bug"
    assert row["findings"][0]["kind"] == "semantic_output_mismatch"
    assert row["findings"][0]["triage_verdict"] == "candidate_implementation_bug"
    assert "buggy_filter" in row["findings"][0]["triage_evidence"]
    assert row["raw_results"]["buggy_filter"]["status"] == "ok"


def test_seeded_groupby_fault_is_classified_as_candidate_bug():
    case = Case(
        "case-seeded-groupby",
        2,
        [
            TableData(
                "t0",
                [ColumnSpec("g", "str"), ColumnSpec("x", "int")],
                [{"g": "a", "x": 1}, {"g": "a", "x": 2}, {"g": "b", "x": 5}],
            )
        ],
        Program(
            "prog-seeded-groupby",
            2,
            [{"op": "groupby", "keys": ["g"], "aggs": [{"column": "x", "func": "sum", "as": "sum_x"}]}],
        ),
    )

    row = run_loaded_case(
        case,
        ["pandas", "buggy_groupby"],
        config=ExperimentConfig(enable_artifact=False),
        save_artifact=False,
    )

    assert row["status"] == "bug"
    assert row["findings"][0]["kind"] == "semantic_output_mismatch"
    assert row["findings"][0]["triage_verdict"] == "candidate_implementation_bug"
    assert "buggy_groupby" in row["findings"][0]["triage_evidence"]
    assert row["raw_results"]["buggy_groupby"]["status"] == "ok"


def test_seeded_join_fault_is_classified_as_candidate_bug():
    case = Case(
        "case-seeded-join",
        3,
        [
            TableData("t0", [ColumnSpec("id", "int"), ColumnSpec("x", "int")], [{"id": 1, "x": 10}, {"id": 2, "x": 20}]),
            TableData("t1", [ColumnSpec("id", "int"), ColumnSpec("j", "int")], [{"id": 1, "j": 100}, {"id": 2, "j": 200}]),
        ],
        Program("prog-seeded-join", 3, [{"op": "join", "table": "t1", "left_on": "id", "right_on": "id", "how": "left"}]),
    )

    row = run_loaded_case(
        case,
        ["pandas", "buggy_join"],
        config=ExperimentConfig(enable_artifact=False),
        save_artifact=False,
    )

    assert row["status"] == "bug"
    assert row["findings"][0]["kind"] == "semantic_output_mismatch"
    assert row["findings"][0]["triage_verdict"] == "candidate_implementation_bug"
    assert "buggy_join" in row["findings"][0]["triage_evidence"]
    assert row["raw_results"]["buggy_join"]["status"] == "ok"


def test_seeded_mutate_fault_is_classified_as_candidate_bug():
    case = Case(
        "case-seeded-mutate",
        4,
        [TableData("t0", [ColumnSpec("x", "int")], [{"x": 1}, {"x": 2}])],
        Program(
            "prog-seeded-mutate",
            4,
            [{"op": "mutate", "column": "m", "expr": {"kind": "arith_const", "source": "x", "op": "mul", "value": 2}}],
        ),
    )

    row = run_loaded_case(
        case,
        ["pandas", "buggy_mutate"],
        config=ExperimentConfig(enable_artifact=False),
        save_artifact=False,
    )

    assert row["status"] == "bug"
    assert row["findings"][0]["kind"] == "semantic_output_mismatch"
    assert row["findings"][0]["triage_verdict"] == "candidate_implementation_bug"
    assert "buggy_mutate" in row["findings"][0]["triage_evidence"]
    assert row["raw_results"]["buggy_mutate"]["status"] == "ok"
