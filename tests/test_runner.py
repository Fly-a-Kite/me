import importlib.util
from pathlib import Path

import pytest

from datadiff.config import ExperimentConfig
from datadiff.datagen import COMMON_API_WORKFLOW_TEMPLATES, generate_case
from datadiff.dsl import Case, ColumnSpec, Program, TableData
from datadiff.normalizer import NormalizedResult
from datadiff.oracle import Finding
from datadiff import runner as runner_module
from datadiff.runner import run_fuzz, run_loaded_case
from datadiff.targets import resolve_target_backends
from datadiff.util import closed_loop_state_path, load_json, read_jsonl, run_meta_path


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
DISTINCT_BACKENDS = ["pandas", "polars", "polars_lazy", "duckdb", "sqlite", "pyarrow", "datafusion"]
FILL_NULL_BACKENDS = ["pandas", "polars", "polars_lazy", "duckdb", "sqlite", "pyarrow", "datafusion"]
COALESCE_BACKENDS = ["pandas", "polars", "polars_lazy", "duckdb", "sqlite", "pyarrow", "datafusion"]
CASE_WHEN_BACKENDS = ["pandas", "polars", "polars_lazy", "duckdb", "sqlite", "pyarrow", "datafusion"]
STRING_CONTAINS_BACKENDS = ["pandas", "polars", "polars_lazy", "duckdb", "sqlite", "pyarrow", "datafusion"]
STRING_LENGTH_BACKENDS = ["pandas", "polars", "polars_lazy", "duckdb", "sqlite", "pyarrow", "datafusion"]
STRING_LOWER_BACKENDS = ["pandas", "polars", "polars_lazy", "duckdb", "sqlite", "pyarrow", "datafusion"]
STRING_STRIP_BACKENDS = ["pandas", "polars", "polars_lazy", "duckdb", "sqlite", "pyarrow", "datafusion"]
STRING_NULL_IF_EMPTY_BACKENDS = ["pandas", "polars", "polars_lazy", "duckdb", "sqlite", "pyarrow", "datafusion"]
STRING_REPLACE_BACKENDS = ["pandas", "polars", "polars_lazy", "duckdb", "sqlite", "pyarrow", "datafusion"]
STRING_SLICE_BACKENDS = ["pandas", "polars", "polars_lazy", "duckdb", "sqlite", "pyarrow", "datafusion"]
STRING_SPLIT_PART_BACKENDS = ["pandas", "polars", "polars_lazy", "duckdb", "sqlite", "pyarrow", "datafusion"]
STRING_CONCAT_BACKENDS = ["pandas", "polars", "polars_lazy", "duckdb", "sqlite", "pyarrow", "datafusion"]
DATE_PART_BACKENDS = ["pandas", "polars", "polars_lazy", "duckdb", "sqlite", "pyarrow", "datafusion"]
TYPE_CAST_BACKENDS = ["pandas", "polars", "polars_lazy", "duckdb", "sqlite", "pyarrow", "datafusion"]
BOOL_NOT_BACKENDS = ["pandas", "polars", "polars_lazy", "duckdb", "sqlite", "pyarrow", "datafusion"]
ABS_BACKENDS = ["pandas", "polars", "polars_lazy", "duckdb", "sqlite", "pyarrow", "datafusion"]
CLIP_BACKENDS = ["pandas", "polars", "polars_lazy", "duckdb", "sqlite", "pyarrow", "datafusion"]
UNION_ALL_BACKENDS = ["pandas", "polars", "polars_lazy", "duckdb", "sqlite", "pyarrow", "datafusion"]
DROP_NULLS_BACKENDS = ["pandas", "polars", "polars_lazy", "duckdb", "sqlite", "pyarrow", "datafusion"]
SEMI_ANTI_JOIN_BACKENDS = ["pandas", "polars", "polars_lazy", "duckdb", "sqlite", "pyarrow", "datafusion"]
MULTI_KEY_JOIN_BACKENDS = ["pandas", "polars", "polars_lazy", "duckdb", "sqlite", "pyarrow", "datafusion"]
SQL_FREEZE_ORDER_BACKENDS = ["pandas", "duckdb", "sqlite", "datafusion"]


def _common_api_workflow_seed_for_template(template: str) -> int:
    for seed in range(len(COMMON_API_WORKFLOW_TEMPLATES)):
        case = generate_case(seed, profile="common_api_workflow")
        if case.metadata.get("workflow_template") == template:
            return seed
    raise AssertionError(f"missing common_api_workflow template: {template}")


def test_guidance_summary_includes_resolved_semantic_boundary_penalty():
    summary = runner_module._guidance_summary(
        {
            "score": 1.0,
            "matched_targets": ["groupby"],
            "score_breakdown": {
                "resolved_semantic_boundary_penalty": -2.0,
                "family_saturation_penalty": -1.0,
                "family_saturation_active": 1.0,
            },
        }
    )

    assert summary["resolved_semantic_boundary_penalty"] == -2.0
    assert summary["matched_semantic_targets"] == ["groupby"]
    assert summary["family_diversity_guard_penalty"] == -1.0
    assert summary["family_diversity_guard_active"] == 1.0


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
    assert set(row["stage_profile"]) == {
        "generate_mutate_ms",
        "backend_execution_ms",
        "normalize_ms",
        "oracle_classification_ms",
        "scheduler_feedback_ms",
        "logging_artifact_ms",
        "total_case_wall_ms",
    }
    assert row["stage_profile"]["total_case_wall_ms"] >= row["stage_profile"]["backend_execution_ms"]


@pytest.mark.skipif(
    any(name != "sqlite" and importlib.util.find_spec(name) is None for name in LATEST_ALL_ENGINE_PACKAGES),
    reason="latest data backends are not installed",
)
def test_run_loaded_case_supports_distinct_across_latest_engines():
    case = Case(
        "case-distinct-latest-engines",
        31,
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
                    {"id": 1, "g": "a", "x": None, "flag": True},
                    {"id": 1, "g": "a", "x": None, "flag": True},
                    {"id": 2, "g": None, "x": 3, "flag": None},
                    {"id": 2, "g": None, "x": 3, "flag": None},
                    {"id": 3, "g": "b", "x": 3, "flag": False},
                ],
            )
        ],
        Program(
            "prog-distinct-latest-engines",
            31,
            [
                {"op": "distinct", "columns": ["g", "x", "flag"]},
                {
                    "op": "sort",
                    "keys": [
                        {"column": "g", "ascending": True, "nulls": "last"},
                        {"column": "x", "ascending": True, "nulls": "last"},
                        {"column": "flag", "ascending": False, "nulls": "last"},
                    ],
                },
            ],
        ),
    )

    row = run_loaded_case(case, DISTINCT_BACKENDS, save_artifact=False)

    assert row["status"] == "ok"
    assert row["findings"] == []
    assert set(row["normalized"]) == set(DISTINCT_BACKENDS)


@pytest.mark.skipif(
    any(name != "sqlite" and importlib.util.find_spec(name) is None for name in LATEST_ALL_ENGINE_PACKAGES),
    reason="latest data backends are not installed",
)
def test_run_loaded_case_supports_fill_null_across_latest_engines():
    case = Case(
        "case-fill-null-latest-engines",
        32,
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
                    {"id": 1, "g": "a", "x": None, "flag": True},
                    {"id": 2, "g": None, "x": 3, "flag": None},
                    {"id": 3, "g": "b", "x": None, "flag": False},
                    {"id": 4, "g": None, "x": -1, "flag": None},
                ],
            )
        ],
        Program(
            "prog-fill-null-latest-engines",
            32,
            [
                {"op": "fill_null", "column": "g", "value": "missing"},
                {"op": "fill_null", "column": "x", "value": 0},
                {"op": "fill_null", "column": "flag", "value": False},
                {
                    "op": "sort",
                    "keys": [
                        {"column": "g", "ascending": True, "nulls": "last"},
                        {"column": "x", "ascending": True, "nulls": "last"},
                        {"column": "id", "ascending": True, "nulls": "last"},
                    ],
                },
            ],
        ),
    )

    row = run_loaded_case(case, FILL_NULL_BACKENDS, save_artifact=False)

    assert row["status"] == "ok"
    assert row["findings"] == []
    assert set(row["normalized"]) == set(FILL_NULL_BACKENDS)


@pytest.mark.skipif(
    any(name != "sqlite" and importlib.util.find_spec(name) is None for name in LATEST_ALL_ENGINE_PACKAGES),
    reason="latest data backends are not installed",
)
def test_run_loaded_case_supports_negative_set_membership_filter_across_latest_engines():
    case = Case(
        "case-negative-set-membership-filter-latest-engines",
        37,
        [
            TableData(
                "t0",
                [
                    ColumnSpec("id", "int", nullable=False),
                    ColumnSpec("g", "str"),
                    ColumnSpec("x", "int"),
                ],
                [
                    {"id": 1, "g": "a", "x": 10},
                    {"id": 2, "g": "b", "x": 20},
                    {"id": 3, "g": None, "x": 30},
                    {"id": 4, "g": "space value", "x": 40},
                    {"id": 5, "g": "", "x": 50},
                ],
            )
        ],
        Program(
            "prog-negative-set-membership-filter-latest-engines",
            37,
            [
                {"op": "filter", "column": "g", "cmp": "not_in_set", "value": ["a", "space value"]},
                {"op": "sort", "keys": [{"column": "id", "ascending": True, "nulls": "last"}]},
                {"op": "select", "columns": ["id", "g"]},
            ],
        ),
    )

    row = run_loaded_case(case, DISTINCT_BACKENDS, save_artifact=False)

    assert row["status"] == "ok"
    assert row["findings"] == []
    for result in row["normalized"].values():
        assert result["columns"] == ["g", "id"]
        assert result["rows"] == [["b", 2], ["", 5]]


@pytest.mark.skipif(
    any(name != "sqlite" and importlib.util.find_spec(name) is None for name in LATEST_ALL_ENGINE_PACKAGES),
    reason="latest data backends are not installed",
)
def test_run_loaded_case_supports_coalesce_across_latest_engines():
    case = Case(
        "case-coalesce-latest-engines",
        40,
        [
            TableData(
                "t0",
                [
                    ColumnSpec("id", "int", nullable=False),
                    ColumnSpec("g", "str"),
                    ColumnSpec("s", "str"),
                    ColumnSpec("x", "int"),
                    ColumnSpec("y", "int"),
                ],
                [
                    {"id": 1, "g": None, "s": "fallback-a", "x": None, "y": 10},
                    {"id": 2, "g": "group-b", "s": "fallback-b", "x": 2, "y": None},
                    {"id": 3, "g": None, "s": None, "x": None, "y": None},
                    {"id": 4, "g": "group-a", "s": None, "x": 4, "y": 40},
                ],
            )
        ],
        Program(
            "prog-coalesce-latest-engines",
            40,
            [
                {"op": "coalesce", "columns": ["g", "s"], "as": "label", "fallback": "missing"},
                {"op": "coalesce", "columns": ["x", "y"], "as": "score", "fallback": 0},
                {
                    "op": "sort",
                    "keys": [
                        {"column": "score", "ascending": True, "nulls": "last"},
                        {"column": "id", "ascending": True, "nulls": "last"},
                    ],
                },
                {"op": "select", "columns": ["id", "label", "score"]},
            ],
        ),
    )

    row = run_loaded_case(case, COALESCE_BACKENDS, save_artifact=False)

    assert row["status"] == "ok"
    assert row["findings"] == []
    assert set(row["normalized"]) == set(COALESCE_BACKENDS)
    for result in row["normalized"].values():
        assert result["rows"] == [
            [3, "missing", 0],
            [2, "group-b", 2],
            [4, "group-a", 4],
            [1, "fallback-a", 10],
        ]


@pytest.mark.skipif(
    any(name != "sqlite" and importlib.util.find_spec(name) is None for name in LATEST_ALL_ENGINE_PACKAGES),
    reason="latest data backends are not installed",
)
def test_run_loaded_case_supports_fill_null_coalesce_groupby_topk_across_latest_engines():
    case = Case(
        "case-fill-null-coalesce-groupby-topk-latest-engines",
        41,
        [
            TableData(
                "t0",
                [
                    ColumnSpec("id", "int", nullable=False),
                    ColumnSpec("g", "str"),
                    ColumnSpec("s", "str"),
                    ColumnSpec("x", "int"),
                    ColumnSpec("flag", "bool"),
                ],
                [
                    {"id": 1, "g": None, "s": "b", "x": None, "flag": True},
                    {"id": 2, "g": "a", "s": None, "x": 2, "flag": False},
                    {"id": 3, "g": None, "s": None, "x": None, "flag": False},
                    {"id": 4, "g": "b", "s": "ignored", "x": 5, "flag": True},
                ],
            )
        ],
        Program(
            "prog-fill-null-coalesce-groupby-topk-latest-engines",
            41,
            [
                {"op": "coalesce", "columns": ["g", "s"], "as": "label", "fallback": "missing"},
                {"op": "fill_null", "column": "x", "value": 0},
                {
                    "op": "groupby",
                    "keys": ["label"],
                    "aggs": [
                        {"column": "x", "func": "sum", "as": "sum_x"},
                        {"column": "id", "func": "count", "as": "count_id"},
                        {"column": "flag", "func": "any", "as": "any_flag"},
                    ],
                },
                {
                    "op": "sort",
                    "keys": [
                        {"column": "sum_x", "ascending": False, "nulls": "last"},
                        {"column": "label", "ascending": True, "nulls": "last"},
                    ],
                },
                {"op": "select", "columns": ["label", "sum_x", "count_id", "any_flag"]},
                {"op": "limit", "n": 3},
            ],
        ),
    )

    row = run_loaded_case(case, COALESCE_BACKENDS, save_artifact=False)

    assert row["status"] == "ok"
    assert row["findings"] == []
    assert set(row["normalized"]) == set(COALESCE_BACKENDS)
    for result in row["normalized"].values():
        assert result["rows"] == [
            [True, 2, "b", 5],
            [False, 1, "a", 2],
            [False, 1, "missing", 0],
        ]


@pytest.mark.skipif(
    any(name != "sqlite" and importlib.util.find_spec(name) is None for name in LATEST_ALL_ENGINE_PACKAGES),
    reason="latest data backends are not installed",
)
@pytest.mark.parametrize(
    ("comparator", "literal", "expected_ids"),
    [
        ("str_contains", "a", [1, 2, 5]),
        ("str_starts_with", "A", [1]),
        ("str_ends_with", "e", [2]),
    ],
)
def test_run_loaded_case_supports_string_pattern_filter_across_latest_engines(comparator, literal, expected_ids):
    case = Case(
        f"case-{comparator}-filter-latest-engines",
        41,
        [
            TableData(
                "t0",
                [
                    ColumnSpec("id", "int", nullable=False),
                    ColumnSpec("s", "str"),
                ],
                [
                    {"id": 1, "s": "Alpha"},
                    {"id": 2, "s": "space value"},
                    {"id": 3, "s": None},
                    {"id": 4, "s": ""},
                    {"id": 5, "s": "Beta"},
                    {"id": 6, "s": "中文"},
                ],
            )
        ],
        Program(
            f"prog-{comparator}-filter-latest-engines",
            41,
            [
                {"op": "filter", "column": "s", "cmp": comparator, "value": literal},
                {"op": "sort", "keys": [{"column": "id", "ascending": True, "nulls": "last"}]},
                {"op": "select", "columns": ["id", "s"]},
            ],
        ),
    )

    row = run_loaded_case(case, STRING_CONTAINS_BACKENDS, save_artifact=False)

    assert row["status"] == "ok"
    assert row["findings"] == []
    assert set(row["normalized"]) == set(STRING_CONTAINS_BACKENDS)
    for result in row["normalized"].values():
        id_index = result["columns"].index("id")
        assert [record[id_index] for record in result["rows"]] == expected_ids


@pytest.mark.skipif(
    any(name != "sqlite" and importlib.util.find_spec(name) is None for name in LATEST_ALL_ENGINE_PACKAGES),
    reason="latest data backends are not installed",
)
def test_run_loaded_case_supports_string_upper_across_latest_engines():
    case = Case(
        "case-string-upper-latest-engines",
        43,
        [
            TableData(
                "t0",
                [
                    ColumnSpec("id", "int", nullable=False),
                    ColumnSpec("s", "str"),
                ],
                [
                    {"id": 1, "s": "Alpha"},
                    {"id": 2, "s": "space value"},
                    {"id": 3, "s": None},
                    {"id": 4, "s": ""},
                    {"id": 5, "s": "Beta"},
                ],
            )
        ],
        Program(
            "prog-string-upper-latest-engines",
            43,
            [
                {"op": "mutate", "column": "s_upper", "expr": {"kind": "string_upper", "source": "s"}},
                {"op": "sort", "keys": [{"column": "id", "ascending": True, "nulls": "last"}]},
                {"op": "select", "columns": ["id", "s_upper"]},
            ],
        ),
    )

    row = run_loaded_case(case, STRING_STRIP_BACKENDS, save_artifact=False)

    assert row["status"] == "ok"
    assert row["findings"] == []
    assert set(row["normalized"]) == set(STRING_STRIP_BACKENDS)
    for result in row["normalized"].values():
        assert result["rows"] == [
            [1, "ALPHA"],
            [2, "SPACE VALUE"],
            [3, None],
            [4, ""],
            [5, "BETA"],
        ]


@pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None or importlib.util.find_spec("duckdb") is None,
    reason="pandas/duckdb backends are not installed",
)
def test_run_loaded_case_does_not_count_expected_unicode_case_mapping_as_bug():
    case = Case(
        "case-unicode-case-boundary",
        430,
        [
            TableData(
                "t0",
                [
                    ColumnSpec("s", "str"),
                    ColumnSpec("x", "int"),
                    ColumnSpec("y", "float"),
                    ColumnSpec("z", "float"),
                ],
                [{"s": "δelta", "x": 1, "y": None, "z": 0.5}],
            )
        ],
        Program(
            "prog-unicode-case-boundary",
            430,
            [
                {"op": "mutate", "column": "s_upper", "expr": {"kind": "string_upper", "source": "s"}},
                {"op": "coalesce", "columns": ["y", "z"], "as": "co_y", "fallback": 0.0},
                {"op": "groupby", "keys": ["s_upper"], "aggs": [{"column": "x", "func": "nunique", "as": "nunique_x"}]},
            ],
        ),
    )

    row = run_loaded_case(
        case,
        ["pandas", "duckdb", "sqlite"],
        config=ExperimentConfig(candidate_recheck_count=2),
        save_artifact=False,
    )

    assert row["status"] == "ok"
    assert row["findings"]
    assert row["findings"][0]["root_cause"] == "unicode_case_mapping"
    assert row["findings"][0]["triage_verdict"] == "expected_semantic_divergence"
    assert row["candidate_recheck"]["skip_reason"] == "no_countable_candidate_findings"


@pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None or importlib.util.find_spec("duckdb") is None,
    reason="pandas/duckdb backends are not installed",
)
def test_run_fuzz_does_not_store_expected_semantic_divergence_as_feedback_seed(tmp_path, monkeypatch):
    case = Case(
        "case-unicode-case-feedback-boundary",
        431,
        [
            TableData(
                "t0",
                [
                    ColumnSpec("s", "str"),
                    ColumnSpec("x", "int"),
                    ColumnSpec("y", "float"),
                    ColumnSpec("z", "float"),
                ],
                [{"s": "δelta", "x": 1, "y": None, "z": 0.5}],
            )
        ],
        Program(
            "prog-unicode-case-feedback-boundary",
            431,
            [
                {"op": "mutate", "column": "s_upper", "expr": {"kind": "string_upper", "source": "s"}},
                {"op": "coalesce", "columns": ["y", "z"], "as": "co_y", "fallback": 0.0},
                {"op": "groupby", "keys": ["s_upper"], "aggs": [{"column": "x", "func": "nunique", "as": "nunique_x"}]},
            ],
        ),
    )

    monkeypatch.setattr(runner_module, "generate_case", lambda *args, **kwargs: case)
    case_log = tmp_path / "expected-semantic-feedback.cases.jsonl"
    config = ExperimentConfig(enable_local_source_scheduler=True, candidate_recheck_count=2)

    run_file = run_fuzz(cases=1, seed=431, backends=["pandas", "duckdb", "sqlite"], config=config, case_log_file=case_log)
    row = read_jsonl(run_file)[0]

    assert row["findings"][0]["triage_verdict"] == "expected_semantic_divergence"
    assert row["stored_in_feedback_corpus"] is False
    assert row["feedback_eligible"] is False
    assert row["feedback_skip_reason"] == "resolved_semantic_divergence"
    assert row["source_reward"] < 0.0


@pytest.mark.skipif(
    any(name != "sqlite" and importlib.util.find_spec(name) is None for name in LATEST_ALL_ENGINE_PACKAGES),
    reason="latest data backends are not installed",
)
def test_run_loaded_case_supports_string_lower_across_latest_engines():
    case = Case(
        "case-string-lower-latest-engines",
        53,
        [
            TableData(
                "t0",
                [
                    ColumnSpec("id", "int", nullable=False),
                    ColumnSpec("s", "str"),
                ],
                [
                    {"id": 1, "s": "Alpha"},
                    {"id": 2, "s": "SPACE VALUE"},
                    {"id": 3, "s": None},
                    {"id": 4, "s": ""},
                    {"id": 5, "s": "Beta"},
                    {"id": 6, "s": "中文"},
                ],
            )
        ],
        Program(
            "prog-string-lower-latest-engines",
            53,
            [
                {"op": "mutate", "column": "s_lower", "expr": {"kind": "string_lower", "source": "s"}},
                {"op": "sort", "keys": [{"column": "id", "ascending": True, "nulls": "last"}]},
                {"op": "select", "columns": ["id", "s_lower"]},
            ],
        ),
    )

    row = run_loaded_case(case, STRING_LOWER_BACKENDS, save_artifact=False)

    assert row["status"] == "ok"
    assert row["findings"] == []
    assert set(row["normalized"]) == set(STRING_LOWER_BACKENDS)
    for result in row["normalized"].values():
        assert result["rows"] == [
            [1, "alpha"],
            [2, "space value"],
            [3, None],
            [4, ""],
            [5, "beta"],
            [6, "中文"],
        ]


@pytest.mark.skipif(
    any(name != "sqlite" and importlib.util.find_spec(name) is None for name in LATEST_ALL_ENGINE_PACKAGES),
    reason="latest data backends are not installed",
)
def test_run_loaded_case_supports_string_length_across_latest_engines():
    case = Case(
        "case-string-length-latest-engines",
        52,
        [
            TableData(
                "t0",
                [
                    ColumnSpec("id", "int", nullable=False),
                    ColumnSpec("s", "str"),
                ],
                [
                    {"id": 1, "s": ""},
                    {"id": 2, "s": None},
                    {"id": 3, "s": "alpha"},
                    {"id": 4, "s": " space"},
                    {"id": 5, "s": "中文"},
                ],
            )
        ],
        Program(
            "prog-string-length-latest-engines",
            52,
            [
                {"op": "mutate", "column": "s_len", "expr": {"kind": "string_length", "source": "s"}},
                {"op": "sort", "keys": [{"column": "id", "ascending": True, "nulls": "last"}]},
                {"op": "select", "columns": ["id", "s_len"]},
            ],
        ),
    )

    row = run_loaded_case(case, STRING_LENGTH_BACKENDS, save_artifact=False)

    assert row["status"] == "ok"
    assert row["findings"] == []
    assert set(row["normalized"]) == set(STRING_LENGTH_BACKENDS)
    for result in row["normalized"].values():
        assert result["rows"] == [
            [1, 0],
            [2, None],
            [3, 5],
            [4, 6],
            [5, 2],
        ]


@pytest.mark.skipif(
    any(name != "sqlite" and importlib.util.find_spec(name) is None for name in LATEST_ALL_ENGINE_PACKAGES),
    reason="latest data backends are not installed",
)
def test_run_loaded_case_supports_string_null_if_empty_across_latest_engines():
    case = Case(
        "case-string-null-if-empty-latest-engines",
        51,
        [
            TableData(
                "t0",
                [
                    ColumnSpec("id", "int", nullable=False),
                    ColumnSpec("s", "str"),
                ],
                [
                    {"id": 1, "s": ""},
                    {"id": 2, "s": None},
                    {"id": 3, "s": "alpha"},
                    {"id": 4, "s": " space"},
                ],
            )
        ],
        Program(
            "prog-string-null-if-empty-latest-engines",
            51,
            [
                {"op": "mutate", "column": "s_norm", "expr": {"kind": "string_null_if_empty", "source": "s"}},
                {"op": "sort", "keys": [{"column": "id", "ascending": True, "nulls": "last"}]},
                {"op": "select", "columns": ["id", "s_norm"]},
            ],
        ),
    )

    row = run_loaded_case(case, STRING_NULL_IF_EMPTY_BACKENDS, save_artifact=False)

    assert row["status"] == "ok"
    assert row["findings"] == []
    assert set(row["normalized"]) == set(STRING_NULL_IF_EMPTY_BACKENDS)
    for result in row["normalized"].values():
        assert result["rows"] == [
            [1, None],
            [2, None],
            [3, "alpha"],
            [4, " space"],
        ]


@pytest.mark.skipif(
    any(name != "sqlite" and importlib.util.find_spec(name) is None for name in LATEST_ALL_ENGINE_PACKAGES),
    reason="latest data backends are not installed",
)
def test_run_loaded_case_supports_string_strip_across_latest_engines():
    case = Case(
        "case-string-strip-latest-engines",
        44,
        [
            TableData(
                "t0",
                [
                    ColumnSpec("id", "int", nullable=False),
                    ColumnSpec("s", "str"),
                ],
                [
                    {"id": 1, "s": " Alpha "},
                    {"id": 2, "s": "space value"},
                    {"id": 3, "s": None},
                    {"id": 4, "s": ""},
                    {"id": 5, "s": "  中文  "},
                ],
            )
        ],
        Program(
            "prog-string-strip-latest-engines",
            44,
            [
                {"op": "mutate", "column": "s_clean", "expr": {"kind": "string_strip", "source": "s"}},
                {"op": "sort", "keys": [{"column": "id", "ascending": True, "nulls": "last"}]},
                {"op": "select", "columns": ["id", "s_clean"]},
            ],
        ),
    )

    row = run_loaded_case(case, STRING_STRIP_BACKENDS, save_artifact=False)

    assert row["status"] == "ok"
    assert row["findings"] == []
    assert set(row["normalized"]) == set(STRING_STRIP_BACKENDS)
    for result in row["normalized"].values():
        assert result["rows"] == [
            [1, "Alpha"],
            [2, "space value"],
            [3, None],
            [4, ""],
            [5, "中文"],
        ]


@pytest.mark.skipif(
    any(name != "sqlite" and importlib.util.find_spec(name) is None for name in LATEST_ALL_ENGINE_PACKAGES),
    reason="latest data backends are not installed",
)
def test_run_loaded_case_supports_string_replace_across_latest_engines():
    case = Case(
        "case-string-replace-latest-engines",
        45,
        [
            TableData(
                "t0",
                [
                    ColumnSpec("id", "int", nullable=False),
                    ColumnSpec("s", "str"),
                ],
                [
                    {"id": 1, "s": "Alpha Beta"},
                    {"id": 2, "s": "space value"},
                    {"id": 3, "s": None},
                    {"id": 4, "s": ""},
                    {"id": 5, "s": "  padded  "},
                ],
            )
        ],
        Program(
            "prog-string-replace-latest-engines",
            45,
            [
                {"op": "mutate", "column": "s_token", "expr": {"kind": "string_replace", "source": "s", "old": " ", "new": "_"}},
                {"op": "sort", "keys": [{"column": "id", "ascending": True, "nulls": "last"}]},
                {"op": "select", "columns": ["id", "s_token"]},
            ],
        ),
    )

    row = run_loaded_case(case, STRING_REPLACE_BACKENDS, save_artifact=False)

    assert row["status"] == "ok"
    assert row["findings"] == []
    assert set(row["normalized"]) == set(STRING_REPLACE_BACKENDS)
    for result in row["normalized"].values():
        assert result["rows"] == [
            [1, "Alpha_Beta"],
            [2, "space_value"],
            [3, None],
            [4, ""],
            [5, "__padded__"],
        ]


@pytest.mark.skipif(
    any(name != "sqlite" and importlib.util.find_spec(name) is None for name in LATEST_ALL_ENGINE_PACKAGES),
    reason="latest data backends are not installed",
)
def test_run_loaded_case_supports_string_slice_across_latest_engines():
    case = Case(
        "case-string-slice-latest-engines",
        46,
        [
            TableData(
                "t0",
                [
                    ColumnSpec("id", "int", nullable=False),
                    ColumnSpec("s", "str"),
                ],
                [
                    {"id": 1, "s": "Alpha"},
                    {"id": 2, "s": "space value"},
                    {"id": 3, "s": None},
                    {"id": 4, "s": ""},
                    {"id": 5, "s": "中文"},
                ],
            )
        ],
        Program(
            "prog-string-slice-latest-engines",
            46,
            [
                {"op": "mutate", "column": "s_prefix", "expr": {"kind": "string_slice", "source": "s", "start": 0, "length": 3}},
                {"op": "sort", "keys": [{"column": "id", "ascending": True, "nulls": "last"}]},
                {"op": "select", "columns": ["id", "s_prefix"]},
            ],
        ),
    )

    row = run_loaded_case(case, STRING_SLICE_BACKENDS, save_artifact=False)

    assert row["status"] == "ok"
    assert row["findings"] == []
    assert set(row["normalized"]) == set(STRING_SLICE_BACKENDS)
    for result in row["normalized"].values():
        assert result["rows"] == [
            [1, "Alp"],
            [2, "spa"],
            [3, None],
            [4, ""],
            [5, "中文"],
        ]


@pytest.mark.skipif(
    any(name != "sqlite" and importlib.util.find_spec(name) is None for name in LATEST_ALL_ENGINE_PACKAGES),
    reason="latest data backends are not installed",
)
def test_run_loaded_case_supports_string_split_part_across_latest_engines():
    case = Case(
        "case-string-split-part-latest-engines",
        50,
        [
            TableData(
                "t0",
                [
                    ColumnSpec("id", "int", nullable=False),
                    ColumnSpec("s", "str"),
                ],
                [
                    {"id": 1, "s": "alpha beta"},
                    {"id": 2, "s": None},
                    {"id": 3, "s": "plain"},
                    {"id": 4, "s": " leading"},
                    {"id": 5, "s": ""},
                ],
            )
        ],
        Program(
            "prog-string-split-part-latest-engines",
            50,
            [
                {"op": "mutate", "column": "token", "expr": {"kind": "string_split_part", "source": "s", "sep": " ", "index": 0}},
                {"op": "sort", "keys": [{"column": "id", "ascending": True, "nulls": "last"}]},
                {"op": "select", "columns": ["id", "token"]},
            ],
        ),
    )

    row = run_loaded_case(case, STRING_SPLIT_PART_BACKENDS, save_artifact=False)

    assert row["status"] == "ok"
    assert row["findings"] == []
    assert set(row["normalized"]) == set(STRING_SPLIT_PART_BACKENDS)
    for result in row["normalized"].values():
        assert result["rows"] == [
            [1, "alpha"],
            [2, None],
            [3, "plain"],
            [4, ""],
            [5, ""],
        ]


@pytest.mark.skipif(
    any(name != "sqlite" and importlib.util.find_spec(name) is None for name in LATEST_ALL_ENGINE_PACKAGES),
    reason="latest data backends are not installed",
)
def test_run_loaded_case_supports_date_part_across_latest_engines():
    case = Case(
        "case-date-part-latest-engines",
        48,
        [
            TableData(
                "t0",
                [
                    ColumnSpec("id", "int", nullable=False),
                    ColumnSpec("dt", "str"),
                    ColumnSpec("x", "int"),
                ],
                [
                    {"id": 1, "dt": "2024-01-03", "x": 5},
                    {"id": 2, "dt": "2024-02-14T08:30:00", "x": 7},
                    {"id": 3, "dt": None, "x": 11},
                    {"id": 4, "dt": "2025-12-31", "x": 13},
                ],
            )
        ],
        Program(
            "prog-date-part-latest-engines",
            48,
            [
                {"op": "mutate", "column": "year", "expr": {"kind": "date_part", "source": "dt", "part": "year"}},
                {"op": "mutate", "column": "month", "expr": {"kind": "date_part", "source": "dt", "part": "month"}},
                {"op": "mutate", "column": "day", "expr": {"kind": "date_part", "source": "dt", "part": "day"}},
                {
                    "op": "sort",
                    "keys": [
                        {"column": "year", "ascending": True, "nulls": "last"},
                        {"column": "month", "ascending": True, "nulls": "last"},
                        {"column": "day", "ascending": True, "nulls": "last"},
                    ],
                },
                {"op": "select", "columns": ["id", "year", "month", "day"]},
            ],
        ),
    )

    row = run_loaded_case(case, DATE_PART_BACKENDS, save_artifact=False)

    assert row["status"] == "ok"
    assert row["findings"] == []
    assert set(row["normalized"]) == set(DATE_PART_BACKENDS)
    for result in row["normalized"].values():
        assert result["columns"] == ["day", "id", "month", "year"]
        assert result["rows"] == [
            [3, 1, 1, 2024],
            [14, 2, 2, 2024],
            [31, 4, 12, 2025],
            [None, 3, None, None],
        ]


@pytest.mark.skipif(
    any(name != "sqlite" and importlib.util.find_spec(name) is None for name in LATEST_ALL_ENGINE_PACKAGES),
    reason="latest data backends are not installed",
)
def test_run_loaded_case_supports_type_cast_boundaries_across_latest_engines():
    case = Case(
        "case-type-cast-boundary-latest-engines",
        49,
        [
            TableData(
                "t0",
                [
                    ColumnSpec("id", "int", nullable=False),
                    ColumnSpec("num_s", "str"),
                    ColumnSpec("x", "int"),
                ],
                [
                    {"id": 1, "num_s": "10", "x": -2},
                    {"id": 2, "num_s": "0", "x": 0},
                    {"id": 3, "num_s": None, "x": 5},
                    {"id": 4, "num_s": "-1", "x": None},
                ],
            )
        ],
        Program(
            "prog-type-cast-boundary-latest-engines",
            49,
            [
                {"op": "mutate", "column": "num_i", "expr": {"kind": "cast", "source": "num_s", "to": "int", "input_domain": "integer_string"}},
                {"op": "mutate", "column": "num_f", "expr": {"kind": "cast", "source": "num_i", "to": "float"}},
                {"op": "mutate", "column": "id_label", "expr": {"kind": "cast", "source": "id", "to": "str"}},
                {
                    "op": "sort",
                    "keys": [
                        {"column": "num_i", "ascending": True, "nulls": "last"},
                        {"column": "id", "ascending": True, "nulls": "last"},
                    ],
                },
                {"op": "select", "columns": ["id_label", "num_s", "num_i", "num_f"]},
            ],
        ),
    )

    row = run_loaded_case(case, TYPE_CAST_BACKENDS, save_artifact=False)

    assert row["status"] == "ok"
    assert row["findings"] == []
    assert set(row["normalized"]) == set(TYPE_CAST_BACKENDS)
    for result in row["normalized"].values():
        assert result["columns"] == ["id_label", "num_f", "num_i", "num_s"]
        assert result["rows"] == [
            ["4", -1.0, -1, "-1"],
            ["2", 0.0, 0, "0"],
            ["1", 10.0, 10, "10"],
            ["3", None, None, None],
        ]


@pytest.mark.skipif(
    any(name != "sqlite" and importlib.util.find_spec(name) is None for name in LATEST_ALL_ENGINE_PACKAGES),
    reason="latest data backends are not installed",
)
def test_run_loaded_case_supports_string_concat_across_latest_engines():
    case = Case(
        "case-string-concat-latest-engines",
        47,
        [
            TableData(
                "t0",
                [
                    ColumnSpec("id", "int", nullable=False),
                    ColumnSpec("g", "str"),
                    ColumnSpec("s", "str"),
                ],
                [
                    {"id": 1, "g": "a", "s": "Alpha"},
                    {"id": 2, "g": "space value", "s": "Beta"},
                    {"id": 3, "g": None, "s": "Gamma"},
                    {"id": 4, "g": "b", "s": None},
                    {"id": 5, "g": "中", "s": "文"},
                ],
            )
        ],
        Program(
            "prog-string-concat-latest-engines",
            47,
            [
                {"op": "mutate", "column": "label", "expr": {"kind": "string_concat", "source": "g", "other": "s", "sep": "-"}},
                {"op": "sort", "keys": [{"column": "id", "ascending": True, "nulls": "last"}]},
                {"op": "select", "columns": ["id", "label"]},
            ],
        ),
    )

    row = run_loaded_case(case, STRING_CONCAT_BACKENDS, save_artifact=False)

    assert row["status"] == "ok"
    assert row["findings"] == []
    assert set(row["normalized"]) == set(STRING_CONCAT_BACKENDS)
    for result in row["normalized"].values():
        assert result["rows"] == [
            [1, "a-Alpha"],
            [2, "space value-Beta"],
            [3, None],
            [4, None],
            [5, "中-文"],
        ]


@pytest.mark.skipif(
    any(name != "sqlite" and importlib.util.find_spec(name) is None for name in LATEST_ALL_ENGINE_PACKAGES),
    reason="latest data backends are not installed",
)
def test_run_loaded_case_supports_string_contains_expr_across_latest_engines():
    case = Case(
        "case-string-contains-expr-latest-engines",
        49,
        [
            TableData(
                "t0",
                [
                    ColumnSpec("id", "int", nullable=False),
                    ColumnSpec("s", "str"),
                ],
                [
                    {"id": 1, "s": "Alpha"},
                    {"id": 2, "s": "space value"},
                    {"id": 3, "s": None},
                    {"id": 4, "s": ""},
                    {"id": 5, "s": "padded"},
                ],
            )
        ],
        Program(
            "prog-string-contains-expr-latest-engines",
            49,
            [
                {"op": "mutate", "column": "has_a", "expr": {"kind": "string_contains", "source": "s", "needle": "a"}},
                {"op": "sort", "keys": [{"column": "id", "ascending": True, "nulls": "last"}]},
                {"op": "select", "columns": ["id", "has_a"]},
            ],
        ),
    )

    row = run_loaded_case(case, STRING_CONTAINS_BACKENDS, save_artifact=False)

    assert row["status"] == "ok"
    assert row["findings"] == []
    assert set(row["normalized"]) == set(STRING_CONTAINS_BACKENDS)
    for result in row["normalized"].values():
        assert result["columns"] == ["has_a", "id"]
        assert result["rows"] == [
            [True, 1],
            [True, 2],
            [None, 3],
            [False, 4],
            [True, 5],
        ]


@pytest.mark.skipif(
    any(name != "sqlite" and importlib.util.find_spec(name) is None for name in LATEST_ALL_ENGINE_PACKAGES),
    reason="latest data backends are not installed",
)
def test_run_loaded_case_supports_string_starts_and_ends_with_expr_across_latest_engines():
    case = Case(
        "case-string-starts-ends-expr-latest-engines",
        50,
        [
            TableData(
                "t0",
                [
                    ColumnSpec("id", "int", nullable=False),
                    ColumnSpec("s", "str"),
                ],
                [
                    {"id": 1, "s": "Alpha"},
                    {"id": 2, "s": "space value"},
                    {"id": 3, "s": None},
                    {"id": 4, "s": ""},
                    {"id": 5, "s": "padded"},
                ],
            )
        ],
        Program(
            "prog-string-starts-ends-expr-latest-engines",
            50,
            [
                {"op": "mutate", "column": "starts_a", "expr": {"kind": "string_starts_with", "source": "s", "needle": "A"}},
                {"op": "mutate", "column": "ends_d", "expr": {"kind": "string_ends_with", "source": "s", "needle": "d"}},
                {"op": "sort", "keys": [{"column": "id", "ascending": True, "nulls": "last"}]},
                {"op": "select", "columns": ["id", "starts_a", "ends_d"]},
            ],
        ),
    )

    row = run_loaded_case(case, STRING_CONTAINS_BACKENDS, save_artifact=False)

    assert row["status"] == "ok"
    assert row["findings"] == []
    assert set(row["normalized"]) == set(STRING_CONTAINS_BACKENDS)
    for result in row["normalized"].values():
        assert result["columns"] == ["ends_d", "id", "starts_a"]
        assert result["rows"] == [
            [False, 1, True],
            [False, 2, False],
            [None, 3, None],
            [False, 4, False],
            [True, 5, False],
        ]


@pytest.mark.skipif(
    any(name != "sqlite" and importlib.util.find_spec(name) is None for name in LATEST_ALL_ENGINE_PACKAGES),
    reason="latest data backends are not installed",
)
def test_run_loaded_case_supports_bool_not_across_latest_engines():
    case = Case(
        "case-bool-not-latest-engines",
        48,
        [
            TableData(
                "t0",
                [
                    ColumnSpec("id", "int", nullable=False),
                    ColumnSpec("flag", "bool"),
                ],
                [
                    {"id": 1, "flag": True},
                    {"id": 2, "flag": False},
                    {"id": 3, "flag": None},
                ],
            )
        ],
        Program(
            "prog-bool-not-latest-engines",
            48,
            [
                {"op": "mutate", "column": "not_flag", "expr": {"kind": "bool_not", "source": "flag"}},
                {"op": "sort", "keys": [{"column": "id", "ascending": True, "nulls": "last"}]},
                {"op": "select", "columns": ["id", "not_flag"]},
            ],
        ),
    )

    row = run_loaded_case(case, BOOL_NOT_BACKENDS, save_artifact=False)

    assert row["status"] == "ok"
    assert row["findings"] == []
    assert set(row["normalized"]) == set(BOOL_NOT_BACKENDS)
    for result in row["normalized"].values():
        assert result["rows"] == [
            [1, False],
            [2, True],
            [3, None],
        ]


@pytest.mark.skipif(
    any(name != "sqlite" and importlib.util.find_spec(name) is None for name in LATEST_ALL_ENGINE_PACKAGES),
    reason="latest data backends are not installed",
)
def test_run_loaded_case_supports_nullable_bool_reduction_filter_topk_across_latest_engines():
    case = Case(
        "case-nullable-bool-reduction-filter-topk",
        52,
        [
            TableData(
                "t0",
                [
                    ColumnSpec("id", "int", nullable=False),
                    ColumnSpec("g", "str"),
                    ColumnSpec("flag", "bool"),
                    ColumnSpec("x", "int"),
                ],
                [
                    {"id": 1, "g": "a", "flag": True, "x": 1},
                    {"id": 2, "g": "a", "flag": None, "x": 2},
                    {"id": 3, "g": "b", "flag": False, "x": 3},
                    {"id": 4, "g": "b", "flag": None, "x": 4},
                    {"id": 5, "g": "c", "flag": True, "x": 5},
                    {"id": 6, "g": "c", "flag": False, "x": 6},
                    {"id": 7, "g": "d", "flag": None, "x": 7},
                ],
            )
        ],
        Program(
            "prog-nullable-bool-reduction-filter-topk",
            52,
            [
                {
                    "op": "groupby",
                    "keys": ["g"],
                    "aggs": [
                        {"column": "flag", "func": "any", "as": "any_flag"},
                        {"column": "flag", "func": "all", "as": "all_flag"},
                        {"column": "flag", "func": "count", "as": "count_flag"},
                    ],
                },
                {"op": "filter", "column": "any_flag", "cmp": "bool_is_true", "value": None},
                {
                    "op": "sort",
                    "keys": [
                        {"column": "all_flag", "ascending": False, "nulls": "last"},
                        {"column": "count_flag", "ascending": False, "nulls": "last"},
                        {"column": "g", "ascending": True, "nulls": "last"},
                    ],
                },
                {"op": "select", "columns": ["g", "any_flag", "all_flag", "count_flag"]},
                {"op": "limit", "n": 3},
            ],
        ),
    )

    row = run_loaded_case(case, BOOL_AGG_BACKENDS, save_artifact=False)

    assert row["status"] == "ok"
    assert row["findings"] == []
    assert set(row["normalized"]) == set(BOOL_AGG_BACKENDS)
    for result in row["normalized"].values():
        assert result["columns"] == ["all_flag", "any_flag", "count_flag", "g"]
        assert result["rows"] == [
            [True, True, 1, "a"],
            [False, True, 2, "c"],
        ]


@pytest.mark.skipif(
    any(name != "sqlite" and importlib.util.find_spec(name) is None for name in ["pandas", "duckdb", "pyarrow"]),
    reason="pyarrow comparison backends are not installed",
)
def test_pyarrow_global_bool_aggregate_empty_input_keeps_nullable_bool_schema():
    case = Case(
        "case-pyarrow-empty-global-bool-aggregate-filter",
        53,
        [
            TableData(
                "t0",
                [
                    ColumnSpec("id", "int", nullable=False),
                    ColumnSpec("x", "int"),
                    ColumnSpec("y", "float"),
                    ColumnSpec("flag", "bool"),
                ],
                [
                    {"id": 1, "x": 10, "y": 1.5, "flag": True},
                    {"id": 2, "x": None, "y": None, "flag": None},
                ],
            )
        ],
        Program(
            "prog-pyarrow-empty-global-bool-aggregate-filter",
            53,
            [
                {"op": "filter", "column": "id", "cmp": "<", "value": 0},
                {
                    "op": "aggregate",
                    "aggs": [
                        {"column": "id", "func": "count", "as": "row_count"},
                        {"column": "x", "func": "sum", "as": "sum_x"},
                        {"column": "y", "func": "mean", "as": "mean_y"},
                        {"column": "flag", "func": "any", "as": "any_flag"},
                        {"column": "flag", "func": "all", "as": "all_flag"},
                    ],
                },
                {"op": "filter", "column": "any_flag", "cmp": "bool_is_false", "value": None},
            ],
        ),
    )

    row = run_loaded_case(case, ["pandas", "duckdb", "pyarrow"], save_artifact=False)

    assert row["status"] == "ok"
    assert row["findings"] == []
    assert row["raw_results"]["pyarrow"]["status"] == "ok"
    assert set(row["normalized"]) == {"pandas", "duckdb", "pyarrow"}
    for result in row["normalized"].values():
        assert result["rows"] == []


@pytest.mark.skipif(
    any(name != "sqlite" and importlib.util.find_spec(name) is None for name in LATEST_ALL_ENGINE_PACKAGES),
    reason="latest data backends are not installed",
)
def test_run_loaded_case_supports_numeric_clip_across_latest_engines():
    case = Case(
        "case-numeric-clip-latest-engines",
        42,
        [
            TableData(
                "t0",
                [
                    ColumnSpec("id", "int", nullable=False),
                    ColumnSpec("x", "int"),
                    ColumnSpec("y", "float"),
                ],
                [
                    {"id": 1, "x": -10, "y": -2.5},
                    {"id": 2, "x": -1, "y": -0.5},
                    {"id": 3, "x": 0, "y": None},
                    {"id": 4, "x": 5, "y": 3.5},
                    {"id": 5, "x": None, "y": 0.5},
                ],
            )
        ],
        Program(
            "prog-numeric-clip-latest-engines",
            42,
            [
                {"op": "mutate", "column": "x_clip", "expr": {"kind": "clip", "source": "x", "lower": -2, "upper": 2}},
                {"op": "mutate", "column": "y_clip", "expr": {"kind": "clip", "source": "y", "lower": -1.0, "upper": 1.0}},
                {"op": "sort", "keys": [{"column": "id", "ascending": True, "nulls": "last"}]},
                {"op": "select", "columns": ["id", "x_clip", "y_clip"]},
            ],
        ),
    )

    row = run_loaded_case(case, CLIP_BACKENDS, save_artifact=False)

    assert row["status"] == "ok"
    assert row["findings"] == []
    assert set(row["normalized"]) == set(CLIP_BACKENDS)
    for result in row["normalized"].values():
        assert result["rows"] == [
            [1, -2, -1.0],
            [2, -1, -0.5],
            [3, 0, None],
            [4, 2, 1.0],
            [5, None, 0.5],
        ]


@pytest.mark.skipif(
    any(name != "sqlite" and importlib.util.find_spec(name) is None for name in LATEST_ALL_ENGINE_PACKAGES),
    reason="latest data backends are not installed",
)
def test_run_loaded_case_supports_numeric_abs_across_latest_engines():
    case = Case(
        "case-numeric-abs-latest-engines",
        43,
        [
            TableData(
                "t0",
                [
                    ColumnSpec("id", "int", nullable=False),
                    ColumnSpec("x", "int"),
                    ColumnSpec("y", "float"),
                ],
                [
                    {"id": 1, "x": -10, "y": -2.5},
                    {"id": 2, "x": -1, "y": -0.0},
                    {"id": 3, "x": 0, "y": None},
                    {"id": 4, "x": 5, "y": 3.5},
                    {"id": 5, "x": None, "y": 0.5},
                ],
            )
        ],
        Program(
            "prog-numeric-abs-latest-engines",
            43,
            [
                {"op": "mutate", "column": "x_abs", "expr": {"kind": "abs", "source": "x"}},
                {"op": "mutate", "column": "y_abs", "expr": {"kind": "abs", "source": "y"}},
                {"op": "sort", "keys": [{"column": "id", "ascending": True, "nulls": "last"}]},
                {"op": "select", "columns": ["id", "x_abs", "y_abs"]},
            ],
        ),
    )

    row = run_loaded_case(case, ABS_BACKENDS, save_artifact=False)

    assert row["status"] == "ok"
    assert row["findings"] == []
    assert set(row["normalized"]) == set(ABS_BACKENDS)
    for result in row["normalized"].values():
        assert result["rows"] == [
            [1, 10, 2.5],
            [2, 1, 0.0],
            [3, 0, None],
            [4, 5, 3.5],
            [5, None, 0.5],
        ]


@pytest.mark.skipif(
    any(name != "sqlite" and importlib.util.find_spec(name) is None for name in LATEST_ALL_ENGINE_PACKAGES),
    reason="latest data backends are not installed",
)
def test_run_loaded_case_supports_case_when_across_latest_engines():
    case = Case(
        "case-case-when-latest-engines",
        34,
        [
            TableData(
                "t0",
                [
                    ColumnSpec("id", "int", nullable=False),
                    ColumnSpec("x", "int"),
                    ColumnSpec("flag", "bool"),
                ],
                [
                    {"id": 1, "x": None, "flag": True},
                    {"id": 2, "x": 3, "flag": None},
                    {"id": 3, "x": -2, "flag": False},
                    {"id": 4, "x": 0, "flag": None},
                ],
            )
        ],
        Program(
            "prog-case-when-latest-engines",
            34,
            [
                {
                    "op": "case_when",
                    "as": "x_bucket",
                    "condition": {"column": "x", "cmp": ">=", "value": 0},
                    "then": "nonnegative",
                    "else": "negative_or_null",
                },
                {
                    "op": "case_when",
                    "as": "flag_state",
                    "condition": {"column": "flag", "cmp": "bool_is_true", "value": None},
                    "then": "true_flag",
                    "else": "not_true",
                },
                {
                    "op": "sort",
                    "keys": [
                        {"column": "x_bucket", "ascending": True, "nulls": "last"},
                        {"column": "flag_state", "ascending": True, "nulls": "last"},
                        {"column": "id", "ascending": True, "nulls": "last"},
                    ],
                },
            ],
        ),
    )

    row = run_loaded_case(case, CASE_WHEN_BACKENDS, save_artifact=False)

    assert row["status"] == "ok"
    assert row["findings"] == []
    assert set(row["normalized"]) == set(CASE_WHEN_BACKENDS)


@pytest.mark.skipif(
    any(name != "sqlite" and importlib.util.find_spec(name) is None for name in LATEST_ALL_ENGINE_PACKAGES),
    reason="latest data backends are not installed",
)
def test_run_loaded_case_supports_union_all_across_latest_engines():
    columns = [
        ColumnSpec("id", "int", nullable=False),
        ColumnSpec("g", "str"),
        ColumnSpec("x", "int"),
        ColumnSpec("flag", "bool"),
    ]
    case = Case(
        "case-union-all-latest-engines",
        36,
        [
            TableData(
                "t0",
                columns,
                [
                    {"id": 1, "g": "a", "x": 1, "flag": True},
                    {"id": 2, "g": None, "x": 3, "flag": None},
                    {"id": 3, "g": "b", "x": None, "flag": False},
                ],
            ),
            TableData(
                "t_append",
                list(columns),
                [
                    {"id": 4, "g": "a", "x": 5, "flag": None},
                    {"id": 5, "g": "c", "x": None, "flag": True},
                    {"id": 6, "g": None, "x": -1, "flag": False},
                ],
            ),
        ],
        Program(
            "prog-union-all-latest-engines",
            36,
            [
                {"op": "union_all", "table": "t_append"},
                {"op": "filter", "column": "id", "cmp": ">=", "value": 2},
                {
                    "op": "sort",
                    "keys": [
                        {"column": "g", "ascending": True, "nulls": "last"},
                        {"column": "id", "ascending": True, "nulls": "last"},
                    ],
                },
            ],
        ),
    )

    row = run_loaded_case(case, UNION_ALL_BACKENDS, save_artifact=False)

    assert row["status"] == "ok"
    assert row["findings"] == []
    assert set(row["normalized"]) == set(UNION_ALL_BACKENDS)


@pytest.mark.skipif(
    any(name != "sqlite" and importlib.util.find_spec(name) is None for name in LATEST_ALL_ENGINE_PACKAGES),
    reason="latest data backends are not installed",
)
def test_run_loaded_case_supports_union_all_after_mutate_nullable_metadata_change():
    columns = [
        ColumnSpec("id", "int", nullable=False),
        ColumnSpec("g", "str"),
        ColumnSpec("x", "int"),
    ]
    case = Case(
        "case-union-all-after-mutate-id",
        37,
        [
            TableData(
                "t0",
                columns,
                [
                    {"id": 1, "g": "a", "x": 1},
                    {"id": 2, "g": "b", "x": None},
                ],
            ),
            TableData(
                "t_append",
                list(columns),
                [
                    {"id": 3, "g": "a", "x": 2},
                    {"id": 4, "g": "b", "x": 4},
                ],
            ),
        ],
        Program(
            "prog-union-all-after-mutate-id",
            37,
            [
                {"op": "mutate", "column": "id", "expr": {"kind": "add_const", "source": "id", "value": 0}},
                {"op": "union_all", "table": "t_append"},
                {
                    "op": "groupby",
                    "keys": ["g"],
                    "aggs": [
                        {"column": "id", "func": "count", "as": "count_id"},
                        {"column": "x", "func": "sum", "as": "sum_x"},
                    ],
                },
                {"op": "sort", "keys": [{"column": "g", "ascending": True, "nulls": "last"}]},
            ],
        ),
    )

    row = run_loaded_case(case, UNION_ALL_BACKENDS, save_artifact=False)

    assert row["status"] == "ok"
    assert row["findings"] == []


@pytest.mark.skipif(
    any(name != "sqlite" and importlib.util.find_spec(name) is None for name in LATEST_ALL_ENGINE_PACKAGES),
    reason="latest data backends are not installed",
)
def test_run_loaded_case_supports_union_all_after_groupby_count_nunique():
    grouped_columns = [
        ColumnSpec("g", "str", nullable=True),
        ColumnSpec("unique_s", "int", nullable=True),
        ColumnSpec("unique_x", "int", nullable=True),
        ColumnSpec("count_id", "int", nullable=True),
    ]
    case = Case(
        "case-union-all-after-groupby-count-nunique",
        138,
        [
            TableData(
                "t0",
                [
                    ColumnSpec("id", "int", nullable=False),
                    ColumnSpec("g", "str", nullable=True),
                    ColumnSpec("x", "int", nullable=True),
                    ColumnSpec("s", "str", nullable=True),
                ],
                [
                    {"id": 1, "g": "a", "x": 10, "s": "alpha"},
                    {"id": 2, "g": "a", "x": 10, "s": "beta"},
                    {"id": 3, "g": "b", "x": None, "s": None},
                    {"id": 4, "g": "c", "x": 20, "s": "alpha"},
                ],
            ),
            TableData(
                "t_append",
                grouped_columns,
                [
                    {"g": "append", "unique_s": 2, "unique_x": 1, "count_id": 1},
                    {"g": "b", "unique_s": 0, "unique_x": 0, "count_id": 3},
                ],
            ),
        ],
        Program(
            "prog-union-all-after-groupby-count-nunique",
            138,
            [
                {
                    "op": "groupby",
                    "keys": ["g"],
                    "aggs": [
                        {"column": "s", "func": "nunique", "as": "unique_s"},
                        {"column": "x", "func": "nunique", "as": "unique_x"},
                        {"column": "id", "func": "count", "as": "count_id"},
                    ],
                },
                {"op": "union_all", "table": "t_append"},
                {
                    "op": "sort",
                    "keys": [
                        {"column": "g", "ascending": True, "nulls": "last"},
                        {"column": "unique_s", "ascending": True, "nulls": "last"},
                        {"column": "unique_x", "ascending": True, "nulls": "last"},
                        {"column": "count_id", "ascending": True, "nulls": "last"},
                    ],
                },
            ],
        ),
    )

    row = run_loaded_case(case, UNION_ALL_BACKENDS, save_artifact=False)

    assert row["status"] == "ok"
    assert row["findings"] == []


@pytest.mark.skipif(
    any(name != "sqlite" and importlib.util.find_spec(name) is None for name in LATEST_ALL_ENGINE_PACKAGES),
    reason="latest data backends are not installed",
)
def test_run_loaded_case_checks_input_partition_union_all_metamorphic_boundary():
    case = Case(
        "case-input-partition-union-boundary",
        64,
        [
            TableData(
                "t0",
                [
                    ColumnSpec("id", "int", nullable=False),
                    ColumnSpec("g", "str"),
                    ColumnSpec("x", "int"),
                    ColumnSpec("flag", "bool"),
                    ColumnSpec("s", "str"),
                ],
                [
                    {"id": 1, "g": "a", "x": 10, "flag": True, "s": "Alpha"},
                    {"id": 2, "g": "b", "x": None, "flag": False, "s": ""},
                    {"id": 3, "g": "a", "x": 30, "flag": None, "s": "Beta"},
                    {"id": 4, "g": "b", "x": 40, "flag": True, "s": None},
                    {"id": 5, "g": None, "x": 50, "flag": False, "s": "Alpha"},
                    {"id": 6, "g": "a", "x": -5, "flag": True, "s": ""},
                ],
            )
        ],
        Program(
            "prog-input-partition-union-boundary",
            64,
            [
                {"op": "mutate", "column": "s_norm", "expr": {"kind": "string_null_if_empty", "source": "s"}},
                {"op": "fill_null", "column": "s_norm", "value": "missing"},
                {
                    "op": "groupby",
                    "keys": ["g", "s_norm"],
                    "aggs": [
                        {"column": "id", "func": "count", "as": "count_id"},
                        {"column": "x", "func": "sum", "as": "sum_x"},
                        {"column": "flag", "func": "any", "as": "any_flag"},
                    ],
                },
            ],
        ),
    )

    row = run_loaded_case(
        case,
        UNION_ALL_BACKENDS,
        config=ExperimentConfig(enable_metamorphic_oracle=True, metamorphic_variant_limit=4),
        save_artifact=False,
    )

    assert row["status"] == "ok"
    assert row["findings"] == []
    variant = row["metamorphic"]["input_partition_union_all:split-3-3"]
    assert [table["name"] for table in variant["case"]["tables"]] == ["t0", "t_partition_tail"]
    for backend in UNION_ALL_BACKENDS:
        assert variant["normalized"][backend]["rows"] == row["normalized"][backend]["rows"]


@pytest.mark.skipif(
    any(name != "sqlite" and importlib.util.find_spec(name) is None for name in LATEST_ALL_ENGINE_PACKAGES),
    reason="latest data backends are not installed",
)
def test_run_loaded_case_checks_filter_input_materialization_metamorphic_boundary():
    case = Case(
        "case-filter-input-materialization-boundary",
        65,
        [
            TableData(
                "t0",
                [
                    ColumnSpec("id", "int", nullable=False),
                    ColumnSpec("g", "str"),
                    ColumnSpec("x", "int"),
                    ColumnSpec("s", "str"),
                ],
                [
                    {"id": 1, "g": "a", "x": 10, "s": "Alpha"},
                    {"id": 2, "g": "b", "x": None, "s": None},
                    {"id": 3, "g": "a", "x": 30, "s": "Beta"},
                    {"id": 4, "g": "b", "x": 40, "s": "Gamma"},
                    {"id": 5, "g": None, "x": 50, "s": "Alpha"},
                    {"id": 6, "g": "a", "x": -5, "s": ""},
                ],
            )
        ],
        Program(
            "prog-filter-input-materialization-boundary",
            65,
            [
                {"op": "filter", "column": "s", "cmp": "in_set", "value": ["Alpha", "Beta"]},
                {"op": "fill_null", "column": "g", "value": "missing"},
                {
                    "op": "groupby",
                    "keys": ["g"],
                    "aggs": [
                        {"column": "id", "func": "count", "as": "count_id"},
                        {"column": "x", "func": "sum", "as": "sum_x"},
                    ],
                },
            ],
        ),
    )

    row = run_loaded_case(
        case,
        UNION_ALL_BACKENDS,
        config=ExperimentConfig(enable_metamorphic_oracle=True, metamorphic_variant_limit=4),
        save_artifact=False,
    )

    assert row["status"] == "ok"
    assert row["findings"] == []
    variant = row["metamorphic"]["filter_input_materialization:s-6to3"]
    assert [op["op"] for op in variant["case"]["program"]["operations"]] == ["fill_null", "groupby"]
    assert variant["case"]["tables"][0]["rows"] == [
        {"id": 1, "g": "a", "x": 10, "s": "Alpha"},
        {"id": 3, "g": "a", "x": 30, "s": "Beta"},
        {"id": 5, "g": None, "x": 50, "s": "Alpha"},
    ]
    for backend in UNION_ALL_BACKENDS:
        assert variant["normalized"][backend]["rows"] == row["normalized"][backend]["rows"]


@pytest.mark.skipif(
    any(name != "sqlite" and importlib.util.find_spec(name) is None for name in LATEST_ALL_ENGINE_PACKAGES),
    reason="latest data backends are not installed",
)
def test_run_loaded_case_checks_cleanup_input_materialization_metamorphic_boundaries():
    base_columns = [
        ColumnSpec("id", "int", nullable=False),
        ColumnSpec("g", "str"),
        ColumnSpec("x", "int"),
        ColumnSpec("flag", "bool"),
    ]
    base_rows = [
        {"id": 1, "g": "a", "x": 10, "flag": True},
        {"id": 2, "g": None, "x": 20, "flag": False},
        {"id": 3, "g": "a", "x": 10, "flag": True},
        {"id": 4, "g": "b", "x": None, "flag": None},
        {"id": 5, "g": "b", "x": 40, "flag": False},
    ]
    cases = [
        (
            "drop_nulls_input_materialization:g-x-5to3",
            Case(
                "case-drop-nulls-input-materialization-boundary",
                66,
                [TableData("t0", base_columns, base_rows)],
                Program(
                    "prog-drop-nulls-input-materialization-boundary",
                    66,
                    [
                        {"op": "drop_nulls", "columns": ["g", "x"]},
                        {
                            "op": "groupby",
                            "keys": ["g"],
                            "aggs": [{"column": "x", "func": "sum", "as": "sum_x"}],
                        },
                    ],
                ),
            ),
        ),
        (
            "fill_null_input_materialization:g",
            Case(
                "case-fill-null-input-materialization-boundary",
                67,
                [TableData("t0", base_columns, base_rows)],
                Program(
                    "prog-fill-null-input-materialization-boundary",
                    67,
                    [
                        {"op": "fill_null", "column": "g", "value": "missing"},
                        {
                            "op": "groupby",
                            "keys": ["g"],
                            "aggs": [{"column": "id", "func": "count", "as": "count_id"}],
                        },
                    ],
                ),
            ),
        ),
        (
            "distinct_input_materialization:g-x-5to4",
            Case(
                "case-distinct-input-materialization-boundary",
                68,
                [TableData("t0", base_columns, base_rows)],
                Program(
                    "prog-distinct-input-materialization-boundary",
                    68,
                    [
                        {"op": "distinct", "columns": ["g", "x"]},
                        {
                            "op": "groupby",
                            "keys": ["g"],
                            "aggs": [{"column": "x", "func": "sum", "as": "sum_x"}],
                        },
                    ],
                ),
            ),
        ),
    ]

    for variant_name, case in cases:
        row = run_loaded_case(
            case,
            UNION_ALL_BACKENDS,
            config=ExperimentConfig(enable_metamorphic_oracle=True, metamorphic_variant_limit=4),
            save_artifact=False,
        )

        assert row["status"] == "ok"
        assert row["findings"] == []
        assert variant_name in row["metamorphic"]
        variant = row["metamorphic"][variant_name]
        for backend in UNION_ALL_BACKENDS:
            assert variant["normalized"][backend]["rows"] == row["normalized"][backend]["rows"]


@pytest.mark.skipif(
    any(name != "sqlite" and importlib.util.find_spec(name) is None for name in LATEST_ALL_ENGINE_PACKAGES),
    reason="latest data backends are not installed",
)
def test_run_loaded_case_supports_drop_nulls_across_latest_engines():
    case = Case(
        "case-drop-nulls-latest-engines",
        38,
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
                    {"id": 1, "g": "a", "x": 1, "flag": True},
                    {"id": 2, "g": None, "x": 3, "flag": None},
                    {"id": 3, "g": "b", "x": None, "flag": False},
                    {"id": 4, "g": "b", "x": 5, "flag": None},
                ],
            )
        ],
        Program(
            "prog-drop-nulls-latest-engines",
            38,
            [
                {"op": "drop_nulls", "columns": ["g", "x"]},
                {
                    "op": "groupby",
                    "keys": ["g"],
                    "aggs": [
                        {"column": "id", "func": "count", "as": "count_id"},
                        {"column": "x", "func": "sum", "as": "sum_x"},
                    ],
                },
                {
                    "op": "sort",
                    "keys": [
                        {"column": "g", "ascending": True, "nulls": "last"},
                    ],
                },
            ],
        ),
    )

    row = run_loaded_case(case, DROP_NULLS_BACKENDS, save_artifact=False)

    assert row["status"] == "ok"
    assert row["findings"] == []


@pytest.mark.skipif(
    any(name != "sqlite" and importlib.util.find_spec(name) is None for name in LATEST_ALL_ENGINE_PACKAGES),
    reason="latest data backends are not installed",
)
@pytest.mark.parametrize(
    ("kind", "expected_ids"),
    [
        ("semi_join", [1, 1, 3]),
        ("anti_join", [2, 4, None]),
    ],
)
def test_run_loaded_case_supports_semi_and_anti_join_across_latest_engines(kind, expected_ids):
    case = Case(
        f"case-{kind}-latest-engines",
        39,
        [
            TableData(
                "t0",
                [
                    ColumnSpec("id", "int", nullable=True),
                    ColumnSpec("g", "str"),
                    ColumnSpec("x", "int"),
                ],
                [
                    {"id": 1, "g": "a", "x": 10},
                    {"id": 1, "g": "a2", "x": 11},
                    {"id": 2, "g": "b", "x": 20},
                    {"id": 3, "g": "c", "x": 30},
                    {"id": 4, "g": "d", "x": 40},
                    {"id": None, "g": "null-left", "x": 99},
                ],
            ),
            TableData(
                "t_lookup",
                [
                    ColumnSpec("id", "int", nullable=True),
                    ColumnSpec("tag", "str"),
                ],
                [
                    {"id": 1, "tag": "one"},
                    {"id": 1, "tag": "duplicate"},
                    {"id": 3, "tag": "three"},
                    {"id": None, "tag": "null-right"},
                ],
            ),
        ],
        Program(
            f"prog-{kind}-latest-engines",
            39,
            [
                {"op": kind, "table": "t_lookup", "left_on": "id", "right_on": "id"},
                {"op": "sort", "keys": [{"column": "x", "ascending": True, "nulls": "last"}]},
            ],
        ),
    )

    row = run_loaded_case(case, SEMI_ANTI_JOIN_BACKENDS, save_artifact=False)

    assert row["status"] == "ok"
    assert row["findings"] == []
    for result in row["normalized"].values():
        id_index = result["columns"].index("id")
        assert [record[id_index] for record in result["rows"]] == expected_ids
    assert set(row["normalized"]) == set(DROP_NULLS_BACKENDS)


@pytest.mark.skipif(
    any(name != "sqlite" and importlib.util.find_spec(name) is None for name in LATEST_ALL_ENGINE_PACKAGES),
    reason="latest data backends are not installed",
)
@pytest.mark.parametrize(
    ("kind", "expected_pairs"),
    [
        ("semi_join", [(1, "a"), (2, "b")]),
        ("anti_join", [(1, "z"), (3, "c"), (None, "c")]),
    ],
)
def test_run_loaded_case_supports_multi_key_semi_and_anti_join_across_latest_engines(kind, expected_pairs):
    case = Case(
        f"case-multi-key-{kind}-latest-engines",
        390,
        [
            TableData(
                "t0",
                [
                    ColumnSpec("id", "int", nullable=True),
                    ColumnSpec("g", "str", nullable=True),
                    ColumnSpec("x", "int"),
                ],
                [
                    {"id": 1, "g": "a", "x": 10},
                    {"id": 1, "g": "z", "x": 11},
                    {"id": 2, "g": "b", "x": 20},
                    {"id": 3, "g": "c", "x": 30},
                    {"id": None, "g": "c", "x": 40},
                ],
            ),
            TableData(
                "t_lookup",
                [
                    ColumnSpec("rid", "int", nullable=True),
                    ColumnSpec("rg", "str", nullable=True),
                ],
                [
                    {"rid": 1, "rg": "a"},
                    {"rid": 1, "rg": "a"},
                    {"rid": 2, "rg": "b"},
                    {"rid": 3, "rg": None},
                    {"rid": None, "rg": "c"},
                ],
            ),
        ],
        Program(
            f"prog-multi-key-{kind}-latest-engines",
            390,
            [
                {"op": kind, "table": "t_lookup", "left_on": ["id", "g"], "right_on": ["rid", "rg"]},
                {"op": "sort", "keys": [{"column": "x", "ascending": True, "nulls": "last"}]},
            ],
        ),
    )

    row = run_loaded_case(case, SEMI_ANTI_JOIN_BACKENDS, save_artifact=False)

    assert row["status"] == "ok"
    assert row["findings"] == []
    for result in row["normalized"].values():
        id_index = result["columns"].index("id")
        g_index = result["columns"].index("g")
        assert [(record[id_index], record[g_index]) for record in result["rows"]] == expected_pairs


@pytest.mark.skipif(
    any(name != "sqlite" and importlib.util.find_spec(name) is None for name in LATEST_ALL_ENGINE_PACKAGES),
    reason="latest data backends are not installed",
)
@pytest.mark.parametrize("kind", ["semi_join", "anti_join"])
def test_run_loaded_case_checks_semi_anti_join_rewrite_metamorphic_across_latest_engines(kind):
    case = Case(
        f"case-{kind}-rewrite-metamorphic",
        391,
        [
            TableData(
                "t0",
                [
                    ColumnSpec("id", "int", nullable=True),
                    ColumnSpec("g", "str", nullable=True),
                    ColumnSpec("x", "int"),
                ],
                [
                    {"id": 1, "g": "a", "x": 10},
                    {"id": 1, "g": "z", "x": 11},
                    {"id": 2, "g": "b", "x": 20},
                    {"id": 3, "g": "c", "x": 30},
                    {"id": None, "g": "c", "x": 40},
                ],
            ),
            TableData(
                "t_lookup",
                [
                    ColumnSpec("rid", "int", nullable=True),
                    ColumnSpec("rg", "str", nullable=True),
                    ColumnSpec("label", "str"),
                ],
                [
                    {"rid": 1, "rg": "a", "label": "match-a"},
                    {"rid": 1, "rg": "a", "label": "duplicate-a"},
                    {"rid": 2, "rg": "b", "label": "match-b"},
                    {"rid": 3, "rg": None, "label": "null-rg"},
                    {"rid": None, "rg": "c", "label": "null-rid"},
                ],
            ),
        ],
        Program(
            f"prog-{kind}-rewrite-metamorphic",
            391,
            [
                {"op": kind, "table": "t_lookup", "left_on": ["id", "g"], "right_on": ["rid", "rg"]},
                {"op": "sort", "keys": [{"column": "x", "ascending": True, "nulls": "last"}]},
            ],
        ),
    )

    row = run_loaded_case(
        case,
        SEMI_ANTI_JOIN_BACKENDS,
        config=ExperimentConfig(enable_metamorphic_oracle=True, metamorphic_variant_limit=20),
        save_artifact=False,
    )

    assert row["status"] == "ok"
    assert row["findings"] == []
    assert any(name.startswith("semi_anti_join_rewrite:") for name in row["metamorphic"])


@pytest.mark.skipif(
    any(name != "sqlite" and importlib.util.find_spec(name) is None for name in LATEST_ALL_ENGINE_PACKAGES),
    reason="latest data backends are not installed",
)
def test_run_loaded_case_supports_multi_key_join_across_latest_engines():
    case = Case(
        "case-multi-key-join",
        45,
        [
            TableData(
                "t0",
                [
                    ColumnSpec("id", "int", nullable=False),
                    ColumnSpec("g", "str", nullable=False),
                    ColumnSpec("x", "int", nullable=True),
                ],
                [
                    {"id": 1, "g": "a", "x": 10},
                    {"id": 1, "g": "b", "x": 20},
                    {"id": 2, "g": "a", "x": 30},
                    {"id": 3, "g": "c", "x": 40},
                ],
            ),
            TableData(
                "t1",
                [
                    ColumnSpec("rid", "int", nullable=False),
                    ColumnSpec("rg", "str", nullable=False),
                    ColumnSpec("tag", "str", nullable=True),
                ],
                [
                    {"rid": 1, "rg": "a", "tag": "match-a"},
                    {"rid": 1, "rg": "c", "tag": "wrong-g"},
                    {"rid": 2, "rg": "a", "tag": "match-b"},
                ],
            ),
        ],
        Program(
            "prog-multi-key-join",
            45,
            [
                {
                    "op": "join",
                    "table": "t1",
                    "left_on": ["id", "g"],
                    "right_on": ["rid", "rg"],
                    "how": "left",
                },
                {"op": "fill_null", "column": "tag", "value": "missing"},
                {
                    "op": "sort",
                    "keys": [
                        {"column": "id", "ascending": True, "nulls": "last"},
                        {"column": "g", "ascending": True, "nulls": "last"},
                    ],
                },
                {"op": "select", "columns": ["id", "g", "tag"]},
            ],
        ),
    )

    row = run_loaded_case(case, MULTI_KEY_JOIN_BACKENDS, save_artifact=False)

    assert row["status"] == "ok"
    assert row["findings"] == []
    assert all(result["status"] == "ok" for result in row["normalized"].values())


@pytest.mark.skipif(
    any(name != "sqlite" and importlib.util.find_spec(name) is None for name in ["pandas", "duckdb", "datafusion"]),
    reason="SQL comparison backends are not installed",
)
def test_sql_backends_freeze_pending_order_before_fill_null_on_sort_key():
    case = Case(
        "case-fill-null-after-sort-freezes-order",
        33,
        [
            TableData(
                "t0",
                [ColumnSpec("id", "int", nullable=False), ColumnSpec("x", "int"), ColumnSpec("s", "str")],
                [
                    {"id": 1, "x": None, "s": "b"},
                    {"id": 2, "x": -1, "s": "a"},
                    {"id": 3, "x": None, "s": "a"},
                    {"id": 4, "x": 2, "s": "c"},
                ],
            )
        ],
        Program(
            "prog-fill-null-after-sort-freezes-order",
            33,
            [
                {
                    "op": "sort",
                    "keys": [
                        {"column": "x", "ascending": True, "nulls": "last"},
                        {"column": "s", "ascending": True, "nulls": "last"},
                    ],
                },
                {"op": "fill_null", "column": "x", "value": 0},
            ],
        ),
    )

    row = run_loaded_case(case, SQL_FREEZE_ORDER_BACKENDS, save_artifact=False)

    assert row["status"] == "ok"
    assert row["findings"] == []


@pytest.mark.skipif(
    any(name != "sqlite" and importlib.util.find_spec(name) is None for name in ["pandas", "duckdb", "datafusion"]),
    reason="SQL comparison backends are not installed",
)
def test_sql_backends_freeze_pending_order_before_case_when_on_sort_key():
    case = Case(
        "case-case-when-after-sort-freezes-order",
        35,
        [
            TableData(
                "t0",
                [ColumnSpec("id", "int", nullable=False), ColumnSpec("x", "int"), ColumnSpec("s", "str")],
                [
                    {"id": 1, "x": None, "s": "b"},
                    {"id": 2, "x": -1, "s": "a"},
                    {"id": 3, "x": None, "s": "a"},
                    {"id": 4, "x": 2, "s": "c"},
                ],
            )
        ],
        Program(
            "prog-case-when-after-sort-freezes-order",
            35,
            [
                {
                    "op": "sort",
                    "keys": [
                        {"column": "x", "ascending": True, "nulls": "last"},
                        {"column": "s", "ascending": True, "nulls": "last"},
                    ],
                },
                {
                    "op": "case_when",
                    "as": "x",
                    "condition": {"column": "s", "cmp": "==", "value": "a"},
                    "then": 1,
                    "else": 0,
                },
            ],
        ),
    )

    row = run_loaded_case(case, SQL_FREEZE_ORDER_BACKENDS, save_artifact=False)

    assert row["status"] == "ok"
    assert row["findings"] == []


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
    any(name != "sqlite" and importlib.util.find_spec(name) is None for name in SQL_ORDER_BACKENDS),
    reason="data backends are not installed",
)
def test_run_loaded_case_preserves_sqlite_nullable_large_int_aggregate_results():
    case = Case(
        "case-sqlite-nullable-large-int-groupby",
        1,
        [
            TableData(
                "t0",
                [ColumnSpec("g", "str"), ColumnSpec("x", "int")],
                [
                    {"g": "alpha", "x": 9007199254740991},
                    {"g": "alpha", "x": 9007199254740993},
                    {"g": "null_only", "x": None},
                ],
            )
        ],
        Program(
            "prog-sqlite-nullable-large-int-groupby",
            1,
            [
                {
                    "op": "groupby",
                    "keys": ["g"],
                    "aggs": [
                        {"column": "x", "func": "min", "as": "min_x"},
                        {"column": "x", "func": "max", "as": "max_x"},
                    ],
                },
                {"op": "sort", "columns": ["g"], "ascending": True},
            ],
        ),
    )

    row = run_loaded_case(case, SQL_ORDER_BACKENDS, save_artifact=False)

    assert row["status"] == "ok"
    assert row["findings"] == []
    assert ["alpha", 9007199254740993, 9007199254740991] in row["normalized"]["sqlite"]["rows"]


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
    any(importlib.util.find_spec(name) is None for name in BOOL_AGG_BACKEND_PACKAGES),
    reason="bool aggregation backends are not installed",
)
def test_run_loaded_case_checks_groupby_sorted_input_metamorphic_across_latest_engines():
    case = Case(
        "case-groupby-sorted-input-latest-engines",
        124,
        [
            TableData(
                "t0",
                [
                    ColumnSpec("id", "int", nullable=False),
                    ColumnSpec("g", "str", nullable=True),
                    ColumnSpec("flag", "bool", nullable=True),
                    ColumnSpec("x", "int", nullable=True),
                ],
                [
                    {"id": 1, "g": "b", "flag": True, "x": 2},
                    {"id": 2, "g": "a", "flag": None, "x": 1},
                    {"id": 3, "g": "a", "flag": False, "x": 1},
                    {"id": 4, "g": None, "flag": True, "x": None},
                ],
            )
        ],
        Program(
            "prog-groupby-sorted-input-latest-engines",
            124,
            [
                {
                    "op": "groupby",
                    "keys": ["g"],
                    "aggs": [
                        {"column": "x", "func": "nunique", "as": "nunique_x"},
                        {"column": "flag", "func": "any", "as": "any_flag"},
                        {"column": "flag", "func": "count", "as": "count_flag"},
                    ],
                }
            ],
        ),
    )

    row = run_loaded_case(
        case,
        BOOL_AGG_BACKENDS,
        config=ExperimentConfig(enable_metamorphic_oracle=True, metamorphic_variant_limit=20),
        save_artifact=False,
    )

    assert row["status"] == "ok"
    assert row["findings"] == []
    assert any(name.startswith("groupby_sorted_input:") for name in row["metamorphic"])


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
    any(importlib.util.find_spec(name) is None for name in PATH_KEYED_PICK_PACKAGES),
    reason="path basename workflow backends are not installed",
)
@pytest.mark.parametrize(
    ("seed", "template"),
    [
        (84, "string_basename_groupby"),
        (85, "string_basename_topk"),
    ],
)
def test_run_loaded_case_supports_common_api_path_basename_workflows_without_duckdb(seed, template):
    case = generate_case(seed, profile="common_api_workflow")

    row = run_loaded_case(case, PATH_KEYED_PICK_BACKENDS, save_artifact=False)

    assert case.metadata["workflow_template"] == template
    assert case.metadata["discovery_origin"] == "organic"
    assert row["status"] == "ok"
    assert row["findings"] == []


@pytest.mark.skipif(
    any(name != "sqlite" and importlib.util.find_spec(name) is None for name in LATEST_ALL_ENGINE_PACKAGES),
    reason="normalized string join workflow backends are not installed",
)
@pytest.mark.parametrize(
    ("seed", "template"),
    [
        (86, "normalized_string_join_groupby"),
        (87, "normalized_string_join_topk"),
    ],
)
def test_run_loaded_case_supports_common_api_normalized_string_join_workflows(seed, template):
    case = generate_case(seed, profile="common_api_workflow")

    row = run_loaded_case(case, ["pandas", "pyarrow", "polars", "polars_lazy", "duckdb", "sqlite", "datafusion"], save_artifact=False)

    assert case.metadata["workflow_template"] == template
    assert case.metadata["discovery_origin"] == "organic"
    assert row["status"] == "ok"
    assert row["findings"] == []


@pytest.mark.skipif(
    any(name != "sqlite" and importlib.util.find_spec(name) is None for name in LATEST_ALL_ENGINE_PACKAGES),
    reason="normalized string membership workflow backends are not installed",
)
@pytest.mark.parametrize(
    ("seed", "template"),
    [
        (88, "normalized_string_semi_join_topk"),
        (89, "normalized_string_anti_join_groupby"),
    ],
)
def test_run_loaded_case_supports_common_api_normalized_string_membership_workflows(seed, template):
    case = generate_case(seed, profile="common_api_workflow")

    row = run_loaded_case(case, ["pandas", "pyarrow", "polars", "polars_lazy", "duckdb", "sqlite", "datafusion"], save_artifact=False)

    assert case.metadata["workflow_template"] == template
    assert case.metadata["discovery_origin"] == "organic"
    assert row["status"] == "ok"
    assert row["findings"] == []


@pytest.mark.skipif(
    any(name != "sqlite" and importlib.util.find_spec(name) is None for name in LATEST_ALL_ENGINE_PACKAGES),
    reason="string language-risk workflow backends are not installed",
)
@pytest.mark.parametrize(
    ("seed", "template"),
    [
        (_common_api_workflow_seed_for_template("normalized_string_case_when_groupby"), "normalized_string_case_when_groupby"),
        (_common_api_workflow_seed_for_template("normalized_string_case_when_topk"), "normalized_string_case_when_topk"),
        (_common_api_workflow_seed_for_template("string_pattern_case_when_groupby"), "string_pattern_case_when_groupby"),
        (_common_api_workflow_seed_for_template("string_pattern_case_when_topk"), "string_pattern_case_when_topk"),
    ],
)
def test_run_loaded_case_supports_common_api_string_language_risk_workflows(seed, template):
    case = generate_case(seed, profile="common_api_workflow")

    row = run_loaded_case(case, ["pandas", "pyarrow", "polars", "polars_lazy", "duckdb", "sqlite", "datafusion"], save_artifact=False)

    assert case.metadata["workflow_template"] == template
    assert case.metadata["discovery_origin"] == "organic"
    assert row["status"] == "ok"
    assert row["findings"] == []


@pytest.mark.skipif(
    any(name != "sqlite" and importlib.util.find_spec(name) is None for name in LATEST_ALL_ENGINE_PACKAGES),
    reason="SQL rewrite workflow backends are not installed",
)
@pytest.mark.parametrize(
    ("seed", "template"),
    [
        (_common_api_workflow_seed_for_template("sql_distinct_null_coalesce_topk"), "sql_distinct_null_coalesce_topk"),
        (_common_api_workflow_seed_for_template("sql_left_join_coalesce_membership"), "sql_left_join_coalesce_membership"),
        (_common_api_workflow_seed_for_template("sql_case_membership_distinct_topk"), "sql_case_membership_distinct_topk"),
        (_common_api_workflow_seed_for_template("sql_union_coalesce_distinct_topk"), "sql_union_coalesce_distinct_topk"),
        (_common_api_workflow_seed_for_template("sql_left_join_case_membership_groupby"), "sql_left_join_case_membership_groupby"),
        (_common_api_workflow_seed_for_template("sql_left_join_null_predicate_aggregate"), "sql_left_join_null_predicate_aggregate"),
        (_common_api_workflow_seed_for_template("sql_coalesce_case_distinct_groupby"), "sql_coalesce_case_distinct_groupby"),
        (_common_api_workflow_seed_for_template("sql_numeric_text_cast_membership_groupby"), "sql_numeric_text_cast_membership_groupby"),
        (_common_api_workflow_seed_for_template("sql_bool_membership_case_aggregate"), "sql_bool_membership_case_aggregate"),
        (_common_api_workflow_seed_for_template("sql_left_join_bool_case_groupby"), "sql_left_join_bool_case_groupby"),
        (_common_api_workflow_seed_for_template("sql_left_join_bool_coalesce_case_groupby"), "sql_left_join_bool_coalesce_case_groupby"),
        (_common_api_workflow_seed_for_template("sql_bool_antijoin_case_aggregate"), "sql_bool_antijoin_case_aggregate"),
        (_common_api_workflow_seed_for_template("sql_left_join_bool_coalesce_filter_groupby"), "sql_left_join_bool_coalesce_filter_groupby"),
        (_common_api_workflow_seed_for_template("sql_numeric_text_cast_bool_antijoin_groupby"), "sql_numeric_text_cast_bool_antijoin_groupby"),
        (_common_api_workflow_seed_for_template("sql_multi_key_semijoin_case_groupby"), "sql_multi_key_semijoin_case_groupby"),
        (_common_api_workflow_seed_for_template("sql_multi_key_antijoin_case_groupby"), "sql_multi_key_antijoin_case_groupby"),
    ],
)
def test_run_loaded_case_supports_common_api_sql_rewrite_workflows(seed, template):
    case = generate_case(seed, profile="common_api_workflow")

    row = run_loaded_case(case, ["pandas", "pyarrow", "polars", "polars_lazy", "duckdb", "sqlite", "datafusion"], save_artifact=False)

    assert case.metadata["workflow_template"] == template
    assert case.metadata["discovery_origin"] == "organic"
    assert row["status"] == "ok"
    assert row["findings"] == []


@pytest.mark.skipif(
    any(name != "sqlite" and importlib.util.find_spec(name) is None for name in LATEST_ALL_ENGINE_PACKAGES),
    reason="row-number workflow backends are not installed",
)
def test_run_loaded_case_row_number_filter_treats_pandas_nan_as_null_order_key():
    case = Case(
        "case-row-number-null-float-order-key",
        35408,
        [
            TableData(
                "t0",
                [
                    ColumnSpec("row_nr", "int", nullable=False),
                    ColumnSpec("id", "int", nullable=False),
                    ColumnSpec("g", "str", nullable=True),
                    ColumnSpec("x", "int", nullable=True),
                    ColumnSpec("y", "float", nullable=True),
                    ColumnSpec("flag", "bool", nullable=True),
                    ColumnSpec("s", "str", nullable=True),
                ],
                [
                    {"row_nr": 0, "id": 0, "g": "space value", "x": None, "y": None, "flag": None, "s": "Alpha"},
                    {"row_nr": 1, "id": 1, "g": "b", "x": 1, "y": -0.5, "flag": False, "s": "Beta"},
                    {"row_nr": 2, "id": 2, "g": "", "x": 0, "y": 0.5, "flag": True, "s": ""},
                    {"row_nr": 3, "id": 3, "g": " padded ", "x": 10, "y": 1.5, "flag": True, "s": "space value"},
                    {"row_nr": 4, "id": 4, "g": None, "x": 5, "y": 10.0, "flag": True, "s": "  padded  "},
                    {"row_nr": 5, "id": 5, "g": "b", "x": 10, "y": -0.5, "flag": None, "s": "Alpha"},
                    {"row_nr": 6, "id": 1, "g": "b", "x": -2, "y": None, "flag": True, "s": None},
                    {"row_nr": 7, "id": 0, "g": "b", "x": None, "y": 0.5, "flag": False, "s": "Beta"},
                    {"row_nr": 8, "id": 1, "g": " padded ", "x": 0, "y": 10.0, "flag": True, "s": " alpha "},
                    {"row_nr": 9, "id": 2, "g": "space value", "x": 0, "y": 0.0, "flag": True, "s": "Beta"},
                    {"row_nr": 10, "id": 10, "g": "", "x": -10, "y": 1.5, "flag": None, "s": "space value"},
                    {"row_nr": 11, "id": 4, "g": "a", "x": 0, "y": -0.5, "flag": False, "s": "  padded  "},
                ],
            )
        ],
        Program(
            "prog-row-number-null-float-order-key",
            35408,
            [
                {
                    "op": "row_number_filter",
                    "partition_by": ["id"],
                    "order_by": [
                        {"column": "y", "ascending": False, "nulls": "last"},
                        {"column": "row_nr", "ascending": True, "nulls": "last"},
                    ],
                    "cmp": "==",
                    "value": 1,
                },
                {
                    "op": "sort",
                    "keys": [
                        {"column": "id", "ascending": True, "nulls": "last"},
                        {"column": "row_nr", "ascending": True, "nulls": "last"},
                    ],
                },
                {"op": "select", "columns": ["id", "g", "y", "s"]},
            ],
        ),
    )

    row = run_loaded_case(case, ["pandas", "pyarrow", "polars", "polars_lazy", "duckdb", "sqlite", "datafusion"], save_artifact=False)

    assert row["status"] == "ok"
    assert row["findings"] == []


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
    importlib.util.find_spec("pyarrow") is None,
    reason="pyarrow backend is not installed",
)
def test_run_loaded_case_supports_pyarrow_list_flatten_parent_indices_probe():
    case = generate_case(150, profile="pyarrow_list_flatten_parent_indices_semantics")

    row = run_loaded_case(case, ["pyarrow", "duckdb", "sqlite"], save_artifact=False)

    assert row["status"] == "ok"
    assert row["findings"] == []
    for result in row["normalized"].values():
        assert result["columns"] == ["list_flatten_parent_indices_mismatch"]
        assert result["rows"] == [[False]]


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


def test_feedback_target_keys_include_guidance_targets_and_operation_risks():
    guidance_row = {
        "matched_targets": ["semi_anti_join_rewrite", "strings", "groupby"],
        "features": [
            "pattern:semi_anti_join_rewrite",
            "semantic_family:join_membership",
            "combo_risk:groupby_aggregation",
            "rows:many",
        ],
        "config": {
            "semantic_focus_families": ["conditional_semantics"],
            "semantic_focus_signals": ["left_join_case_when_membership"],
        },
    }
    operation_combo = {
        "template": "join_filter_groupby",
        "correctness_risks": ["semi_anti_join_filter_pushdown", "groupby_aggregation"],
    }

    keys = runner_module._feedback_target_keys(guidance_row, operation_combo)

    assert "target:semi_anti_join_rewrite" in keys
    assert "feature:pattern:semi_anti_join_rewrite" in keys
    assert "feature:semantic_family:join_membership" in keys
    assert "semantic_family:join_membership" in keys
    assert "combo:join_filter_groupby" in keys
    assert "semantic_signal:semi_anti_join_filter_pushdown" in keys
    assert "risk:semi_anti_join_filter_pushdown" in keys
    assert "semantic_family:conditional_semantics" in keys
    assert "semantic_signal:left_join_case_when_membership" in keys
    assert "risk:left_join_case_when_membership" in keys
    assert "rows:many" not in keys
    assert "target:strings" not in keys
    assert "target:groupby" not in keys
    assert "feature:combo_risk:groupby_aggregation" not in keys


def test_configured_guidance_targets_include_structured_semantic_focus_targets():
    config = ExperimentConfig(
        guidance_targets=["groupby"],
        semantic_focus_families=["conditional_semantics"],
        semantic_focus_signals=["left_join_case_when_membership"],
    )

    targets = runner_module._configured_guidance_targets(config)

    assert "groupby" in targets
    assert "semantic_family:conditional_semantics" in targets
    assert "semantic_signal:left_join_case_when_membership" in targets
    assert "conditional_semantics" not in targets
    assert "left_join_case_when_membership" not in targets


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
    assert row["guidance"]["strategy"] == "guided"
    assert row["guidance"]["candidate_count"] == 4
    assert row["guidance"]["contributing_candidate_count"] >= 1
    assert row["guidance"]["pruned_candidate_count"] >= 0
    assert "frontier_conformance" in row["guidance"]
    assert "data_sensitivity" in row["guidance"]
    assert "path_coverage_proxy" in row["guidance"]
    assert "combo_priority" in row["guidance"]
    assert "online_weight_mean" in row["guidance"]
    assert "discovery_bucket_count" in row["guidance"]
    assert "discovery_diversity_bonus" in row["guidance"]
    assert "candidate_pool_diversity_bonus" in row["guidance"]
    assert "discovery_stale_penalty" in row["guidance"]
    assert "discovery_stale_active" in row["guidance"]
    assert "recent_discovery_loop_penalty" in row["guidance"]
    assert "recent_discovery_loop_active" in row["guidance"]
    assert "recent_discovery_window_count" in row["guidance"]
    assert "profile_saturation_penalty" in row["guidance"]
    assert "profile_saturation_active" in row["guidance"]
    assert "issue_replay_global_saturation_penalty" in row["guidance"]
    assert "issue_replay_global_saturation_active" in row["guidance"]
    assert "issue_inspired_source_saturation_penalty" in row["guidance"]
    assert "issue_inspired_source_saturation_active" in row["guidance"]
    assert case_log_row["candidate_pool_size"] == 4
    assert meta["guidance"]["strategy"] == "guided"
    assert meta["next_seed"] == 45


def test_run_fuzz_does_not_filter_static_known_saturated_family_before_execution(tmp_path):
    case_log = tmp_path / "saturation-filter.cases.jsonl"
    config = ExperimentConfig(
        generator_profile="join_null_key_topk",
        guidance_strategy="guided",
        guidance_candidate_pool=1,
        guidance_targets=["join_null_key_topk", "topk"],
        enable_replay_bug=True,
        known_saturated_bug_families=["grouped_topk_null_sort_key@datafusion"],
    )

    run_file = run_fuzz(cases=1, seed=9700, backends=[], config=config, case_log_file=case_log)

    row = read_jsonl(run_file)[0]
    case_log_row = read_jsonl(case_log)[0]
    meta = load_json(run_meta_path(run_file))

    assert row["candidate_source"] == "generated"
    assert row["family_saturation_filter"]["enabled"] is True
    assert row["family_saturation_filter"]["filtered_before_candidate"] == 0
    assert row["family_saturation_filter"]["fallback_used"] is False
    assert row["family_saturation_filter"]["last_skip_reason"] == ""
    assert case_log_row["family_saturation_filter"] == row["family_saturation_filter"]
    assert meta["family_saturation_filter"]["enabled"] is True
    assert meta["family_saturation_filter"]["filtered_candidates"] == 0
    assert meta["family_saturation_filter"]["fallback_candidates"] == 0


def test_run_fuzz_uses_candidate_family_prefilter_interface(monkeypatch, tmp_path):
    case_log = tmp_path / "guided-prefilter.cases.jsonl"
    config = ExperimentConfig(
        generator_profile="join_null_key_topk",
        guidance_strategy="guided",
        guidance_candidate_pool=1,
        guidance_targets=["join_null_key_topk", "topk"],
        enable_replay_bug=True,
    )
    calls = {"candidate_prefilter": 0}

    def fake_candidate_prefilter(self, case, *, include_known_families=True):
        calls["candidate_prefilter"] += 1
        return []

    def unexpected_full_prediction(self, case, *, include_known_families=True):
        raise AssertionError("candidate generation should use predicted_saturated_family_roots_for_candidate")

    monkeypatch.setattr(
        runner_module.GuidanceState,
        "predicted_saturated_family_roots_for_candidate",
        fake_candidate_prefilter,
    )
    monkeypatch.setattr(
        runner_module.GuidanceState,
        "predicted_saturated_family_roots",
        unexpected_full_prediction,
    )

    run_fuzz(cases=1, seed=9701, backends=[], config=config, case_log_file=case_log)

    assert calls["candidate_prefilter"] >= 1


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
                "feedback_decision": {
                    "parent_index": 0,
                    "parent_case_id": "case-parent",
                    "retention_utility": 1.0,
                    "schedule_reward": 0.0,
                    "schedule_score": 1.0,
                    "mutation_pulls": 1,
                    "selected_operator": "value",
                },
            }
            return generated

        def record(
            self,
            case,
            behavior_signature,
            has_finding,
            *,
            novelty_signature=None,
            discovery_signature=None,
            candidate_bug_families=None,
            target_keys=None,
            schedule_delta=0.0,
        ):
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
    assert row["feedback_decision"]["parent_case_id"] == "case-parent"
    assert row["feedback_decision"]["schedule_score"] == 1.0
    assert row["operation_combo"]["operation_count"] == len(row["case"]["program"]["operations"])
    assert row["source_reward"] == 1.25
    assert row["stored_in_feedback_corpus"] is False
    assert row["feedback_skip_reason"] == "feedback_mutation_child"
    assert case_log_row["seed_lineage"]["parent_case_id"] == "case-parent"
    assert case_log_row["mutation"]["operator"] == "value"
    assert case_log_row["feedback_decision"]["selected_operator"] == "value"
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
    assert "stage_profile" in row
    assert meta["stage_profile"]["case_count"] == 1
    assert meta["stage_profile"]["totals_ms"]["total_case_wall_ms"] >= 0.0


def test_run_fuzz_records_run_provenance(monkeypatch):
    expected = {
        "schema_version": "run-provenance-v1",
        "vcs": {"git_commit": "abc123", "git_commit_short": "abc123", "git_branch": "main", "workspace_dirty": False},
        "launch": {"source": "closed_loop_tmux", "session": "s", "duration": "12h", "batch_duration": "10m", "log_prefix": "x", "launch_script": "start.sh"},
        "harness": {"authority": True, "freeze_intent": True, "latest_code_claim": True, "evidence_role": "latest_live_authority_12h"},
    }
    monkeypatch.setattr(runner_module, "collect_run_provenance", lambda: expected)

    run_file = run_fuzz(cases=1, seed=54, backends=[], duration_s=None)

    meta = load_json(run_meta_path(run_file))
    assert meta["run_provenance"] == expected


def test_run_fuzz_minimal_log_keeps_only_backend_status():
    config = ExperimentConfig(log_level="minimal")

    run_file = run_fuzz(cases=1, seed=52, backends=[], config=config)

    row = read_jsonl(run_file)[0]
    assert "normalized" not in row
    assert "raw_results" not in row
    assert row["backend_status"] == {}
    assert row["guidance"]["strategy"] == "random"
    assert "frontier_conformance" in row["guidance"]


def test_run_fuzz_can_disable_run_log_compression():
    config = ExperimentConfig(compress_run_log=False)

    run_file = run_fuzz(cases=1, seed=53, backends=[], config=config)

    assert run_file.name.endswith(".jsonl")
    assert not run_file.name.endswith(".jsonl.gz")
    assert load_json(run_meta_path(run_file))["config"]["compress_run_log"] is False


def test_run_fuzz_new_behavior_uses_discovery_signature(tmp_path, monkeypatch):
    rows = [
        {
            "behavior_signature": "behavior-a",
            "discovery_signature": "discovery-shared",
        },
        {
            "behavior_signature": "behavior-b",
            "discovery_signature": "discovery-shared",
        },
    ]
    call_index = {"value": 0}

    def fake_run_loaded_case(
        case,
        backends,
        config=None,
        save_artifact=True,
        backend_instances=None,
        environment=None,
        target_specs=None,
        config_payload=None,
    ):
        index = call_index["value"]
        call_index["value"] += 1
        signatures = rows[index]
        return {
            "run_at": "2026-05-31T00:00:00Z",
            "case": case.to_dict(),
            "targets": target_specs or [],
            "raw_results": {},
            "normalized": {},
            "metamorphic": {},
            "findings": [],
            "candidate_recheck": {"enabled": False, "attempts": 0, "reproduced_keys": [], "non_reproduced_keys": []},
            "config": config_payload or (config or ExperimentConfig()).to_dict(),
            "environment": environment or {},
            "status": "ok",
            "duration_ms": 0.1,
            "behavior_signature": signatures["behavior_signature"],
            "discovery_signature": signatures["discovery_signature"],
        }

    monkeypatch.setattr(runner_module, "run_loaded_case", fake_run_loaded_case)

    run_file = run_fuzz(cases=2, seed=61, backends=[], config=ExperimentConfig(log_level="minimal"))

    first, second = read_jsonl(run_file)
    assert first["behavior_signature"] != second["behavior_signature"]
    assert first["discovery_signature"] == second["discovery_signature"]
    assert first["is_new_behavior"] is True
    assert second["is_new_behavior"] is False
    assert first["signal_new_behavior"] is True
    assert second["signal_new_behavior"] is False
    assert second["feedback_summary"]["feedback_oracle_verdict"] == "redundant_behavior"


def test_run_fuzz_persists_closed_loop_state_across_runs(monkeypatch):
    rows = [
        {
            "behavior_signature": "behavior-a",
            "discovery_signature": "discovery-shared",
        },
        {
            "behavior_signature": "behavior-b",
            "discovery_signature": "discovery-shared",
        },
    ]
    call_index = {"value": 0}

    def fake_run_loaded_case(
        case,
        backends,
        config=None,
        save_artifact=True,
        backend_instances=None,
        environment=None,
        target_specs=None,
        config_payload=None,
    ):
        index = call_index["value"]
        call_index["value"] += 1
        signatures = rows[index]
        return {
            "run_at": "2026-05-31T00:00:00Z",
            "case": case.to_dict(),
            "targets": target_specs or [],
            "raw_results": {},
            "normalized": {},
            "metamorphic": {},
            "findings": [],
            "candidate_recheck": {"enabled": False, "attempts": 0, "reproduced_keys": [], "non_reproduced_keys": []},
            "config": config_payload or (config or ExperimentConfig()).to_dict(),
            "environment": environment or {},
            "status": "ok",
            "duration_ms": 0.1,
            "behavior_signature": signatures["behavior_signature"],
            "discovery_signature": signatures["discovery_signature"],
        }

    monkeypatch.setattr(runner_module, "run_loaded_case", fake_run_loaded_case)

    config = ExperimentConfig(enable_feedback=False, log_level="minimal")
    first_run = run_fuzz(
        cases=1,
        seed=71,
        backends=[],
        config=config,
        persist_closed_loop_state=True,
    )
    first_row = read_jsonl(first_run)[0]
    first_meta = load_json(run_meta_path(first_run))
    first_state = load_json(closed_loop_state_path(first_run))

    second_run = run_fuzz(
        cases=1,
        seed=72,
        backends=[],
        config=config,
        closed_loop_state=first_state,
        persist_closed_loop_state=True,
    )
    second_row = read_jsonl(second_run)[0]
    second_meta = load_json(run_meta_path(second_run))
    second_state = load_json(closed_loop_state_path(second_run))

    assert first_row["is_new_behavior"] is True
    assert second_row["is_new_behavior"] is False
    assert first_meta["closed_loop_state_file"].endswith(".state.json.gz")
    assert first_meta["closed_loop_state_summary"]["seen_signature_count"] == 1
    assert second_meta["closed_loop_state_summary"]["seen_signature_count"] == 1
    assert first_state["seen_signatures"] == ["discovery-shared"]
    assert second_state["seen_signatures"] == ["discovery-shared"]
    assert len(first_state["signal_seen_signatures"]) == 1
    assert second_state["signal_seen_signatures"] == first_state["signal_seen_signatures"]


def test_run_fuzz_signal_new_behavior_uses_coarser_signal_signature(monkeypatch):
    call_index = {"value": 0}

    def fake_generate_case(seed, *, type_aware=True, profile="common"):
        return Case(
            case_id=f"case-{seed}",
            seed=seed,
            tables=[TableData("t0", [ColumnSpec("x", "int")], [{"x": 1}])],
            program=Program(f"prog-{seed}", seed, [{"op": "select", "columns": ["x"]}]),
        )

    def fake_run_loaded_case(
        case,
        backends,
        config=None,
        save_artifact=True,
        backend_instances=None,
        environment=None,
        target_specs=None,
        config_payload=None,
    ):
        index = call_index["value"]
        call_index["value"] += 1
        return {
            "run_at": "2026-05-31T00:00:00Z",
            "case": case.to_dict(),
            "targets": target_specs or [],
            "raw_results": {},
            "normalized": {},
            "metamorphic": {},
            "findings": [],
            "candidate_recheck": {"enabled": False, "attempts": 0, "reproduced_keys": [], "non_reproduced_keys": []},
            "config": config_payload or (config or ExperimentConfig()).to_dict(),
            "environment": environment or {},
            "status": "ok",
            "duration_ms": 0.1,
            "behavior_signature": f"behavior-{index}",
            "discovery_signature": f"discovery-{index}",
        }

    monkeypatch.setattr(runner_module, "generate_case", fake_generate_case)
    monkeypatch.setattr(runner_module, "run_loaded_case", fake_run_loaded_case)

    run_file = run_fuzz(cases=2, seed=81, backends=[], config=ExperimentConfig(log_level="minimal"))

    first, second = read_jsonl(run_file)
    assert first["is_new_behavior"] is True
    assert second["is_new_behavior"] is True
    assert first["signal_signature"] == second["signal_signature"]
    assert first["signal_new_behavior"] is True
    assert second["signal_new_behavior"] is False


def test_run_loaded_case_honors_metamorphic_variant_limit():
    case = generate_case(61, profile="bughunt")
    config = ExperimentConfig(enable_metamorphic_oracle=True, metamorphic_variant_limit=2)

    row = run_loaded_case(case, [], config=config, save_artifact=False)

    assert len(row["metamorphic"]) <= 2
    assert row["config"]["metamorphic_variant_limit"] == 2
