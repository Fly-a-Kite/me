import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from datadiff.config import ExperimentConfig
from datadiff.champion_corpus import ChampionRegistry
from datadiff.datagen import COMMON_API_WORKFLOW_TEMPLATES, generate_case
from datadiff.dsl import Case, ColumnSpec, Program, TableData
from datadiff.execution_cost_model import BackendCostModel
from datadiff.normalizer import NormalizedResult
from datadiff.oracle import Finding
from datadiff.osc_diagnostic_facade import not_evaluated_diagnostic_ref_set
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


@pytest.mark.parametrize(
    ("log_level", "has_findings"),
    [
        ("full", False),
        ("compact", False),
        ("compact", True),
        ("minimal", False),
    ],
)
def test_runner_preserves_only_fail_closed_opaque_diagnostics(
    log_level, has_findings
):
    case_digest = "case-runner-diagnostic"
    variant_digest = "case-runner-variant-diagnostic"
    row = {
        "case": {"case_id": "case-runner", "seed": 1},
        "normalized": {"duckdb": {"status": "error"}},
        "raw_results": {"duckdb": {"status": "error"}},
        "experiment_manifest": {"case_digest": case_digest},
        "osc_diagnostic_refs": not_evaluated_diagnostic_ref_set(
            ["duckdb"],
            case_digest=case_digest,
            reason_code="legacy_error_diagnostic",
        ),
        "metamorphic": {
            "target:b": {
                "case": {"case_id": "case-runner-variant", "seed": 2},
                "normalized": {"duckdb": {"status": "error"}},
                "raw_results": {"duckdb": {"status": "error"}},
                "experiment_manifest": {"case_digest": variant_digest},
                "osc_diagnostic_refs": not_evaluated_diagnostic_ref_set(
                    ["duckdb"],
                    case_digest=variant_digest,
                    reason_code="legacy_variant_error_diagnostic",
                ),
            }
        },
        "findings": [{"kind": "diagnostic-test"}] if has_findings else [],
    }
    row["osc_diagnostic_refs"]["authority_eligible"] = True
    row["metamorphic"]["target:b"]["osc_diagnostic_refs"][
        "authority_eligible"
    ] = True

    log_row = runner_module._compact_log_row(row, log_level)
    runner_module._attach_opaque_diagnostic_refs_for_log(row, log_row)

    refs = log_row["osc_diagnostic_refs"]
    assert log_row["experiment_manifest"]["case_digest"] == case_digest
    assert refs["evaluation_status"] == "not_evaluated"
    assert refs["authority_scope"] == "diagnostic_only"
    assert refs["authority_eligible"] is False
    assert refs["refs"][0]["backend"] == "duckdb"
    assert refs["refs"][0]["ref"]["reason_code"] == "invalid_diagnostic_refs"
    variant = log_row["osc_metamorphic_diagnostic_refs"]["target:b"]
    assert variant["experiment_manifest"]["case_digest"] == variant_digest
    assert variant["backend_status"] == {"duckdb": "error"}
    assert variant["osc_diagnostic_refs"]["case_digest"] == variant_digest
    assert variant["osc_diagnostic_refs"]["authority_eligible"] is False
    assert variant["osc_diagnostic_refs"]["refs"][0]["ref"][
        "reason_code"
    ] == "invalid_diagnostic_refs"
    if "metamorphic" in log_row:
        assert log_row["metamorphic"]["target:b"]["osc_diagnostic_refs"] == (
            variant["osc_diagnostic_refs"]
        )
    assert "candidate" not in refs
    assert "confirmed_bug" not in refs


def test_registered_family_confirmation_skip_is_limited_to_known_saturated_findings():
    known = {
        "kind": "semantic_output_mismatch",
        "root_cause": "topk_filter_pushdown",
        "triage_verdict": "candidate_implementation_bug",
        "suspicious_backends": ["duckdb"],
    }
    novel = {
        **known,
        "root_cause": "new_optimizer_family",
    }

    assert runner_module._row_has_only_known_saturated_candidate_findings(
        {"findings": [known]},
        ["topk_filter_pushdown@duckdb"],
    ) is True
    assert runner_module._row_has_only_known_saturated_candidate_findings(
        {"findings": [known, novel]},
        ["topk_filter_pushdown@duckdb"],
    ) is False


def test_select_adaptive_action_uses_choose_dense_for_best_action():
    calls: list[tuple[str, str, tuple[str, ...]]] = []

    class FakeLearning:
        def __init__(self):
            self.bandits = {
                "semantic_objective": SimpleNamespace(
                    arms={
                        "objective_a": SimpleNamespace(pulls=3),
                        "objective_b": SimpleNamespace(pulls=5),
                    }
                )
            }

        def choose_dense(self, scope, action_ids, **kwargs):
            calls.append(("choose_dense", scope, tuple(action_ids)))
            return SimpleNamespace(action_id="objective_b")

        def rank_top(self, scope, action_ids, *, limit, **kwargs):
            calls.append(("rank_top", scope, tuple(action_ids)))
            assert limit == 8
            return [
                {"action_id": "objective_a", "score": 9.0},
                {"action_id": "objective_b", "score": 8.0},
            ]

    feedback = SimpleNamespace(adaptive_learning=FakeLearning())

    action, selection = runner_module._select_adaptive_action(
        feedback,
        scope="semantic_objective",
        action_pool=("objective_a", "objective_b"),
        context_features=("family:agg",),
        version_id="latest->fixed",
        learning_weight=1.0,
        enabled=True,
    )

    assert action == "objective_b"
    assert selection["action"] == "objective_b"
    assert [row["action_id"] for row in selection["ranked"]] == ["objective_a", "objective_b"]
    assert calls == [
        ("choose_dense", "semantic_objective", ("objective_a", "objective_b")),
        ("rank_top", "semantic_objective", ("objective_a", "objective_b")),
    ]


def test_select_generator_profile_uses_choose_dense_for_best_profile():
    calls: list[tuple[str, str, tuple[str, ...]]] = []

    class FakeLearning:
        def __init__(self):
            self.bandits = {
                "generator_profile": SimpleNamespace(
                    arms={
                        "common": SimpleNamespace(pulls=4),
                        "discovery_fresh": SimpleNamespace(pulls=6),
                    }
                )
            }

        def choose_dense(self, scope, action_ids, **kwargs):
            calls.append(("choose_dense", scope, tuple(action_ids)))
            return SimpleNamespace(action_id="discovery_fresh")

        def rank_top(self, scope, action_ids, *, limit, **kwargs):
            calls.append(("rank_top", scope, tuple(action_ids)))
            assert limit == 8
            return [
                {"action_id": "common", "score": 7.0},
                {"action_id": "discovery_fresh", "score": 6.5},
            ]

    feedback = SimpleNamespace(adaptive_learning=FakeLearning())

    profile, selection = runner_module._select_generator_profile(
        feedback,
        ("common", "discovery_fresh"),
        context_features=("backend:pandas",),
        learning_weight=1.0,
        pool_metadata={"capability_aware": True},
    )

    assert profile == "discovery_fresh"
    assert selection["profile"] == "discovery_fresh"
    assert [row["action_id"] for row in selection["ranked"]] == ["common", "discovery_fresh"]
    assert calls == [
        ("choose_dense", "generator_profile", ("common", "discovery_fresh")),
        ("rank_top", "generator_profile", ("common", "discovery_fresh")),
    ]


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


def test_runner_execute_case_passes_parallel_execution_config(monkeypatch):
    case = Case(
        "case-parallel-config",
        91,
        [TableData("t0", [ColumnSpec("x", "int")], [{"x": 1}])],
        Program("prog-parallel-config", 91, []),
    )
    captured: dict[str, object] = {}

    def fake_execute_case_impl(case_arg, backends_arg, config_arg, **kwargs):
        captured["case_id"] = case_arg.case_id
        captured["backends"] = list(backends_arg)
        captured["parallel"] = kwargs.get("parallel")
        return {}, {}

    monkeypatch.setattr(runner_module, "_execute_case_impl", fake_execute_case_impl)

    runner_module._execute_case(
        case,
        ["left", "right"],
        ExperimentConfig(enable_parallel_backend_execution=False),
        backend_instances={},
    )

    assert captured == {
        "case_id": "case-parallel-config",
        "backends": ["left", "right"],
        "parallel": False,
    }


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
def test_run_loaded_case_supports_coalesce_null_fallback_across_latest_engines():
    case = Case(
        "case-coalesce-null-fallback-latest-engines",
        402,
        [
            TableData(
                "t0",
                [
                    ColumnSpec("id", "int", nullable=False),
                    ColumnSpec("x", "int"),
                ],
                [
                    {"id": 1, "x": None},
                    {"id": 2, "x": 20},
                    {"id": 3, "x": None},
                ],
            )
        ],
        Program(
            "prog-coalesce-null-fallback-latest-engines",
            402,
            [
                {"op": "coalesce", "columns": ["x", "id"], "as": "x_or_id", "fallback": None},
                {
                    "op": "sort",
                    "keys": [
                        {"column": "id", "ascending": True, "nulls": "last"},
                    ],
                },
                {"op": "select", "columns": ["id", "x_or_id"]},
            ],
        ),
    )

    row = run_loaded_case(case, COALESCE_BACKENDS, save_artifact=False)

    assert row["status"] == "ok"
    assert row["findings"] == []
    assert row["raw_results"]["pyarrow"]["status"] == "ok"
    for result in row["normalized"].values():
        assert result["rows"] == [[1, 1], [2, 20], [3, 3]]


@pytest.mark.skipif(
    any(importlib.util.find_spec(name) is None for name in ["pandas", "duckdb"]),
    reason="pandas and duckdb are not installed",
)
def test_duckdb_backends_support_nul_string_literal_in_coalesce_fallback():
    case = Case(
        "case-duckdb-nul-coalesce-fallback",
        404,
        [
            TableData(
                "t0",
                [
                    ColumnSpec("id", "int", nullable=False),
                    ColumnSpec("tag", "str"),
                ],
                [{"id": 1, "tag": None}],
            )
        ],
        Program(
            "prog-duckdb-nul-coalesce-fallback",
            404,
            [
                {"op": "coalesce", "columns": ["tag"], "as": "tag_or_fallback", "fallback": "a\x00z"},
                {"op": "select", "columns": ["id", "tag_or_fallback"]},
            ],
        ),
    )

    row = run_loaded_case(case, ["duckdb", "duckdb_persistent", "pandas"], save_artifact=False)

    assert row["status"] == "ok"
    assert row["findings"] == []
    assert row["raw_results"]["duckdb"]["status"] == "ok"
    assert row["raw_results"]["duckdb_persistent"]["status"] == "ok"
    for result in row["normalized"].values():
        assert result["rows"] == [[1, "a\x00z"]]


@pytest.mark.skipif(
    any(name != "sqlite" and importlib.util.find_spec(name) is None for name in LATEST_ALL_ENGINE_PACKAGES),
    reason="latest data backends are not installed",
)
def test_run_loaded_case_supports_float_filter_with_extreme_int_literal_across_latest_engines():
    case = Case(
        "case-float-filter-extreme-int-literal-latest-engines",
        403,
        [
            TableData(
                "t0",
                [
                    ColumnSpec("id", "int", nullable=False),
                    ColumnSpec("y", "float"),
                ],
                [
                    {"id": 1, "y": None},
                    {"id": 2, "y": 0.0},
                    {"id": 3, "y": -1.0},
                ],
            )
        ],
        Program(
            "prog-float-filter-extreme-int-literal-latest-engines",
            403,
            [
                {"op": "filter", "column": "y", "cmp": "gt_is_not_true", "value": -9223372036854775808},
                {
                    "op": "sort",
                    "keys": [
                        {"column": "id", "ascending": True, "nulls": "last"},
                    ],
                },
                {"op": "select", "columns": ["id", "y"]},
            ],
        ),
    )

    row = run_loaded_case(case, COALESCE_BACKENDS, save_artifact=False)

    assert row["status"] == "ok"
    assert row["findings"] == []
    assert row["raw_results"]["pyarrow"]["status"] == "ok"
    for result in row["normalized"].values():
        assert result["rows"] == [[1, None]]


@pytest.mark.skipif(
    any(importlib.util.find_spec(name) is None for name in ["pandas", "duckdb"]),
    reason="pandas and duckdb are not installed",
)
def test_run_loaded_case_preserves_duckdb_huge_integer_sum_without_fetchdf_precision_loss():
    case = Case(
        "case-duckdb-huge-integer-sum-fetchall-precision",
        405,
        [
            TableData(
                "t0",
                [ColumnSpec("v", "int")],
                [
                    {"v": -9223372036854775808},
                    {"v": 15},
                ],
            )
        ],
        Program(
            "prog-duckdb-huge-integer-sum-fetchall-precision",
            405,
            [
                {
                    "op": "aggregate",
                    "aggs": [
                        {"func": "count", "column": "v", "as": "count_v"},
                        {"func": "sum", "column": "v", "as": "sum_v"},
                        {"func": "mean", "column": "v", "as": "mean_v"},
                    ],
                }
            ],
        ),
    )

    row = run_loaded_case(case, ["duckdb", "pandas", "sqlite"], save_artifact=False)

    assert row["status"] == "ok"
    assert row["findings"] == []
    for result in row["normalized"].values():
        assert result["rows"] == [[2, -4.611686018427388e18, -9223372036854775793]]


@pytest.mark.skipif(
    any(importlib.util.find_spec(name) is None for name in ["pandas", "duckdb", "pyarrow"]),
    reason="pandas, duckdb, and pyarrow are not installed",
)
def test_run_loaded_case_keeps_duckdb_topk_tie_probe_deterministic_by_default(monkeypatch):
    monkeypatch.delenv("DATADIFF_DUCKDB_THREADS", raising=False)
    case = Case(
        "case-duckdb-topk-tie-default-single-thread",
        406,
        [
            TableData(
                "t0",
                [
                    ColumnSpec("id", "int"),
                    ColumnSpec("g", "str"),
                    ColumnSpec("x", "int"),
                    ColumnSpec("y", "float"),
                    ColumnSpec("flag", "bool"),
                    ColumnSpec("s", "str"),
                ],
                [
                    {"id": 0, "g": "OOI", "x": -10, "y": 0.5, "flag": False, "s": "δelta"},
                    {"id": 1, "g": "beta", "x": 10, "y": 0.5, "flag": False, "s": "中文"},
                    {"id": 2, "g": "space value", "x": 10, "y": 1.0, "flag": False, "s": None},
                    {"id": 1, "g": "δelta", "x": None, "y": None, "flag": True, "s": "a"},
                    {"id": 1, "g": None, "x": -1, "y": 0.5, "flag": True, "s": "a"},
                    {"id": 5, "g": "a", "x": 68, "y": 0.5, "flag": False, "s": "gamma"},
                    {"id": 1, "g": "space value", "x": -1, "y": -0.5, "flag": True, "s": "XZULm"},
                    {"id": 1, "g": "", "x": 0, "y": 1.0, "flag": True, "s": "Z"},
                ],
            )
        ],
        Program(
            "prog-duckdb-topk-tie-default-single-thread",
            406,
            [
                {"op": "mutate", "column": "m_0", "expr": {"kind": "cast", "source": "id", "to": "str"}},
                {
                    "op": "case_when",
                    "as": "cw_0",
                    "condition": {"column": "m_0", "cmp": "is_null", "value": None},
                    "then": True,
                    "else": False,
                },
                {
                    "op": "row_number_filter",
                    "partition_by": ["x"],
                    "order_by": [
                        {"column": "y", "ascending": True, "nulls": "last"},
                        {"column": "id", "ascending": False, "nulls": "first"},
                        {"column": "g", "ascending": False, "nulls": "first"},
                    ],
                    "cmp": "==",
                    "value": 1,
                },
                {
                    "op": "groupby",
                    "keys": ["m_0", "s"],
                    "aggs": [{"func": "nunique", "column": "flag", "as": "nunique_flag"}],
                },
                {"op": "filter", "column": "nunique_flag", "cmp": "ge_is_not_true", "value": 1000000},
                {
                    "op": "case_when",
                    "as": "cw_0",
                    "condition": {"column": "m_0", "cmp": "is_null", "value": None},
                    "then": True,
                    "else": False,
                },
                {
                    "op": "case_when",
                    "as": "cw_0",
                    "condition": {"column": "m_0", "cmp": "is_null", "value": None},
                    "then": True,
                    "else": False,
                },
                {"op": "filter", "column": "m_0", "cmp": "is_not_null", "value": None},
                {"op": "filter", "column": "nunique_flag", "cmp": "range_closed", "value": [0, 10]},
                {
                    "op": "row_number_filter",
                    "partition_by": [],
                    "order_by": [{"column": "cw_0", "ascending": True, "nulls": "first"}],
                    "cmp": "==",
                    "value": 1,
                },
                {
                    "op": "sort",
                    "keys": [
                        {"column": "m_0", "ascending": False, "nulls": "last"},
                        {"column": "cw_0", "ascending": False, "nulls": "last"},
                        {"column": "nunique_flag", "ascending": True, "nulls": "last"},
                        {"column": "s", "ascending": False, "nulls": "last"},
                    ],
                },
                {"op": "select", "columns": ["cw_0", "nunique_flag", "s"]},
                {"op": "limit", "n": 3},
            ],
        ),
    )

    row = run_loaded_case(case, ["duckdb", "pandas", "pyarrow"], save_artifact=False)

    assert row["status"] == "ok"
    assert row["findings"] == []
    for result in row["normalized"].values():
        assert result["rows"] == [[False, 1, "δelta"]]


@pytest.mark.skipif(
    importlib.util.find_spec("polars") is None,
    reason="polars is not installed",
)
def test_run_loaded_case_supports_arrow_timestamp_index_attr_probe_on_polars_adapters():
    case = Case(
        "case-arrow-timestamp-index-attr-probe-polars-adapters",
        404,
        [
            TableData(
                "t0",
                [ColumnSpec("probe_id", "int", nullable=False)],
                [{"probe_id": 0}],
            )
        ],
        Program(
            "prog-arrow-timestamp-index-attr-probe-polars-adapters",
            404,
            [{"op": "arrow_timestamp_index_attr_probe", "as": "arrow_timestamp_index_attr_mismatch"}],
        ),
    )

    row = run_loaded_case(case, ["polars", "polars_lazy"], save_artifact=False)

    assert row["status"] == "ok"
    assert row["findings"] == []
    assert row["raw_results"]["polars"]["status"] == "ok"
    assert row["raw_results"]["polars_lazy"]["status"] == "ok"
    for result in row["normalized"].values():
        assert result["rows"] == [[False]]


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
    config = ExperimentConfig(
        method_arm="contract_lattice_shared_cost_full",
        enable_local_source_scheduler=True,
        candidate_recheck_count=2,
    )

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
    any(importlib.util.find_spec(name) is None for name in ["pandas", "duckdb", "pyarrow", "datafusion"]),
    reason="datafusion and pyarrow test backends are not installed",
)
def test_arrow_backends_preserve_nullable_large_int_after_union_all():
    large = 9_007_199_254_740_993
    case = Case(
        "case-arrow-union-all-nullable-large-int",
        2085,
        [
            TableData(
                "t0",
                [
                    ColumnSpec("id", "int"),
                    ColumnSpec("i2", "int"),
                ],
                [{"id": large, "i2": -1}],
            ),
            TableData(
                "t1",
                [ColumnSpec("id", "int")],
                [{"id": large}],
            ),
            TableData(
                "t_union",
                [
                    ColumnSpec("id", "int"),
                    ColumnSpec("i2", "int"),
                ],
                [{"id": None, "i2": None}],
            ),
        ],
        Program(
            "prog-arrow-union-all-nullable-large-int",
            2085,
            [
                {"op": "join", "table": "t1", "left_on": "id", "right_on": "id", "how": "inner"},
                {"op": "union_all", "table": "t_union"},
            ],
        ),
    )

    row = run_loaded_case(case, ["datafusion", "duckdb", "pandas", "pyarrow"], save_artifact=False)

    assert row["status"] == "ok"
    assert row["findings"] == []
    for result in row["normalized"].values():
        assert result["columns"] == ["i2", "id"]
        assert result["rows"] == [[-1, large], [None, None]]


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
def test_run_loaded_case_preserves_left_order_for_sorted_semi_join_before_limit_across_latest_engines():
    case = Case(
        "case-semi-join-order-before-limit-latest-engines",
        392,
        [
            TableData(
                "t0",
                [
                    ColumnSpec("id", "int", nullable=False),
                    ColumnSpec("g", "str", nullable=True),
                    ColumnSpec("x", "int", nullable=True),
                ],
                [
                    {"id": 3, "g": "c", "x": 30},
                    {"id": 1, "g": "a", "x": 10},
                    {"id": 4, "g": "d", "x": 40},
                    {"id": 2, "g": "b", "x": 20},
                    {"id": 5, "g": "e", "x": 50},
                ],
            ),
            TableData(
                "t_lookup",
                [ColumnSpec("id", "int", nullable=False)],
                [{"id": 4}, {"id": 1}, {"id": 3}, {"id": 2}],
            ),
        ],
        Program(
            "prog-semi-join-order-before-limit-latest-engines",
            392,
            [
                {"op": "sort", "columns": ["x"], "ascending": False},
                {"op": "semi_join", "table": "t_lookup", "left_on": "id", "right_on": "id"},
                {"op": "limit", "n": 3},
            ],
        ),
    )

    row = run_loaded_case(case, SEMI_ANTI_JOIN_BACKENDS, save_artifact=False)

    assert row["status"] == "ok"
    assert row["findings"] == []
    for result in row["normalized"].values():
        id_index = result["columns"].index("id")
        assert [record[id_index] for record in result["rows"]] == [4, 3, 2]


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
def test_run_loaded_case_tracks_duckdb_path_projection_regression_or_upstream_fix():
    case = generate_case(107, profile="path_basename_keyed_pick")
    config = ExperimentConfig(enable_artifact=False, enable_replay_bug=True)
    row = run_loaded_case(case, ["pandas", "sqlite", "duckdb"], config=config, save_artifact=False)

    if row["status"] == "ok":
        assert row["findings"] == []
        expected_rows = row["normalized"]["pandas"]["rows"]
        assert row["normalized"]["sqlite"]["rows"] == expected_rows
        assert row["normalized"]["duckdb"]["rows"] == expected_rows
        return

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
        config=ExperimentConfig(
            method_arm="contract_lattice_shared_cost_full",
            candidate_recheck_count=1,
        ),
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


def test_run_loaded_case_recheck_does_not_reuse_persistent_execution_session(monkeypatch):
    case = Case(
        "case-stateful-recheck",
        2,
        [TableData("t0", [ColumnSpec("x", "int")], [{"x": 1}])],
        Program("prog-stateful-recheck", 2, [{"op": "select", "columns": ["x"]}]),
    )

    class StatefulSession:
        def __init__(self):
            self.execute_calls = 0

        def execute_case(self, *args, **kwargs):
            self.execute_calls += 1
            return {}, {
                "left": NormalizedResult("left", "ok", ["x"], [[1]]),
                "right": NormalizedResult("right", "ok", ["x"], [[2]]),
            }

        def execute_cases(self, *args, **kwargs):
            raise AssertionError("metamorphic execution is disabled")

    session = StatefulSession()
    fresh_calls = {"count": 0}

    def fresh_execute_case(*args, **kwargs):
        fresh_calls["count"] += 1
        return {}, {
            "left": NormalizedResult("left", "ok", ["x"], [[1]]),
            "right": NormalizedResult("right", "ok", ["x"], [[1]]),
        }

    def fake_evaluate_case(_case, normalized):
        if normalized["left"].comparison_key == normalized["right"].comparison_key:
            return []
        return [
            Finding(
                finding_id="finding-stateful",
                kind="semantic_output_mismatch",
                severity="critical",
                suspicious_backends=["right"],
                evidence="persistent instance only",
                signature="stateful",
                root_cause="filter_predicate",
                mismatch_class="value",
            )
        ]

    def fake_annotate_findings(case_arg, findings, **kwargs):
        for finding in findings:
            finding.triage_verdict = "candidate_implementation_bug"
            finding.paper_status = "candidate_bug_needs_external_confirmation"
            finding.triage_confidence = "high"

    monkeypatch.setattr(runner_module, "_execute_case", fresh_execute_case)
    monkeypatch.setattr(runner_module, "evaluate_case", fake_evaluate_case)
    monkeypatch.setattr(runner_module, "annotate_findings", fake_annotate_findings)

    row = run_loaded_case(
        case,
        ["left", "right"],
        config=ExperimentConfig(
            method_arm="contract_lattice_shared_cost_full",
            candidate_recheck_count=1,
        ),
        save_artifact=False,
        target_specs=[],
        execution_session=session,
    )

    assert session.execute_calls == 1
    assert fresh_calls["count"] == 1
    assert row["status"] == "ok"
    assert row["candidate_recheck"]["non_reproduced_keys"] == [
        "semantic_output_mismatch:filter_predicate@right:value"
    ]


def test_run_loaded_case_emits_disagreement_descriptor_and_fingerprint(monkeypatch):
    case = Case(
        "case-disagreement",
        11,
        [TableData("t0", [ColumnSpec("x", "int")], [{"x": 1}])],
        Program("prog-disagreement", 11, [{"op": "select", "columns": ["x"]}]),
    )

    def fake_execute_case(*args, **kwargs):
        return {}, {
            "left": NormalizedResult("left", "ok", ["x"], [[1]]),
            "right": NormalizedResult("right", "ok", ["x"], [[2]]),
        }

    def fake_evaluate_case(*args, **kwargs):
        return [
            Finding(
                finding_id="finding-disagreement",
                kind="semantic_output_mismatch",
                severity="critical",
                suspicious_backends=["right"],
                evidence="mismatch",
                signature="sig",
                root_cause="window_boundary",
                mismatch_class="value",
            )
        ]

    monkeypatch.setattr(runner_module, "_execute_case", fake_execute_case)
    monkeypatch.setattr(runner_module, "evaluate_case", fake_evaluate_case)
    monkeypatch.setattr(runner_module, "annotate_findings", lambda *args, **kwargs: None)

    row = run_loaded_case(case, ["left", "right"], save_artifact=False, target_specs=[])

    descriptor = row["disagreement_descriptor"]
    assert descriptor["backend_groups"] == [["left"], ["right"]]
    assert descriptor["pair_count"] == 1
    assert descriptor["primary_root_cause"] == "window_boundary"
    assert "disagree_pair:left|right" in descriptor["feature_tokens"]
    assert row["case"]["metadata"]["disagreement_descriptor"] == descriptor
    assert case.metadata["disagreement_descriptor"] == descriptor

    fingerprint = row["case_fingerprint"]
    assert len(fingerprint["minhash_signature"]) == 64
    assert fingerprint["type_mix_token"] == "num1"
    assert fingerprint["column_count"] == 1
    assert "fp_type_mix:num1" in fingerprint["feature_tokens"]
    assert row["case"]["metadata"]["case_fingerprint"] == fingerprint
    assert case.metadata["case_fingerprint"] == fingerprint


def test_run_loaded_case_cross_validates_differential_and_metamorphic_findings(monkeypatch):
    case = Case(
        "case-oracle-cross-validation",
        12,
        [TableData("t0", [ColumnSpec("x", "int")], [{"x": 1}])],
        Program("prog-oracle-cross-validation", 12, [{"op": "select", "columns": ["x"]}]),
    )
    variant_case = Case(
        "case-oracle-cross-validation-mr",
        12,
        [TableData("t0", [ColumnSpec("x", "int")], [{"x": 1}])],
        Program("prog-oracle-cross-validation-mr", 12, [{"op": "select", "columns": ["x"]}]),
    )
    calls = {"execute": 0}

    def fake_execute_case(*args, **kwargs):
        calls["execute"] += 1
        return {}, {
            "left": NormalizedResult("left", "ok", ["x"], [[1]]),
            "right": NormalizedResult("right", "ok", ["x"], [[2]]),
        }

    def fake_evaluate_case(*args, **kwargs):
        return [
            Finding(
                finding_id="finding-diff",
                kind="semantic_output_mismatch",
                severity="critical",
                suspicious_backends=["right"],
                evidence="differential mismatch",
                signature="diff",
                root_cause="join_semantics",
                oracle="differential",
                mismatch_class="value",
            )
        ]

    def fake_evaluate_metamorphic_variants(*args, **kwargs):
        return [
            Finding(
                finding_id="finding-mr",
                kind="metamorphic_join_semantics_violation",
                severity="high",
                suspicious_backends=["right"],
                evidence="metamorphic mismatch",
                signature="mr",
                root_cause="metamorphic_join_semantics",
                oracle="metamorphic",
            )
        ]

    def fake_annotate_findings(case, findings, **kwargs):
        for finding in findings:
            finding.triage_verdict = "needs_manual_confirmation"
            finding.paper_status = "needs_manual_confirmation"
            finding.triage_confidence = "medium"
            finding.adjudication = {"verdict": "needs_manual_confirmation"}

    monkeypatch.setattr(runner_module, "_execute_case", fake_execute_case)
    monkeypatch.setattr(runner_module, "evaluate_case", fake_evaluate_case)
    monkeypatch.setattr(runner_module, "evaluate_metamorphic_variants", fake_evaluate_metamorphic_variants)
    monkeypatch.setattr(
        runner_module,
        "all_metamorphic_variants",
        lambda case_arg: [SimpleNamespace(name="join_semantics:mr", relation="join_semantics", case=variant_case)],
    )
    monkeypatch.setattr(runner_module, "annotate_findings", fake_annotate_findings)

    row = run_loaded_case(
        case,
        ["left", "right"],
        config=ExperimentConfig(
            method_arm="contract_lattice_shared_cost_full",
            enable_metamorphic_oracle=True,
            metamorphic_variant_limit=1,
        ),
        save_artifact=False,
        target_specs=[],
    )

    assert calls["execute"] == 2
    assert row["oracle_cross_validation"]["cross_validated_count"] == 1
    assert row["oracle_cross_validation"]["metamorphic_only_count"] == 0
    diff_finding = next(finding for finding in row["findings"] if finding["oracle"] == "differential")
    mr_finding = next(finding for finding in row["findings"] if finding["oracle"] == "metamorphic")
    assert diff_finding["confidence"] == "high"
    assert diff_finding["adjudication"]["metamorphic_support"] == "corroborated"
    assert diff_finding["adjudication"]["oracle_complex"]["cross_validated"] is True
    assert mr_finding["adjudication"]["metamorphic_support"] == "corroborates_differential"


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


@pytest.mark.skipif(
    any(importlib.util.find_spec(name) is None for name in ["pandas", "duckdb", "pyarrow"]),
    reason="pyarrow test backends are not installed",
)
def test_pyarrow_backend_supports_duplicate_groupby_source_aggregates():
    case = Case(
        "case-pyarrow-duplicate-groupby-source-aggs",
        8801,
        [
            TableData(
                "t0",
                [ColumnSpec("id", "int", nullable=False)],
                [{"id": 1}],
            ),
            TableData(
                "t1",
                [
                    ColumnSpec("id", "int", nullable=False),
                    ColumnSpec("j", "int"),
                ],
                [{"id": 1, "j": 7}],
            ),
        ],
        Program(
            "prog-pyarrow-duplicate-groupby-source-aggs",
            8801,
            [
                {"op": "join", "table": "t1", "left_on": "id", "right_on": "id", "how": "inner"},
                {
                    "op": "groupby",
                    "keys": ["j"],
                    "aggs": [
                        {"column": "j", "func": "count", "as": "count_joined_rows"},
                        {"column": "j", "func": "count", "as": "count_j"},
                    ],
                },
            ],
        ),
    )

    row = run_loaded_case(case, PYARROW_BACKENDS, save_artifact=False)

    assert row["status"] == "ok"
    assert row["raw_results"]["pyarrow"]["status"] == "ok"
    assert set(row["normalized"]) == set(PYARROW_BACKENDS)
    assert {tuple(result["columns"]) for result in row["normalized"].values()} == {
        ("count_j", "count_joined_rows", "j")
    }
    assert {tuple(tuple(item) for item in result["rows"]) for result in row["normalized"].values()} == {
        ((1, 1, 7),)
    }


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
    case = generate_case(91, profile="discovery")

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


def test_run_fuzz_case_budget_terminates_when_every_iteration_fails(monkeypatch):
    def fail_generation(*args, **kwargs):
        raise RuntimeError("synthetic generation failure")

    monkeypatch.setattr(runner_module, "generate_case", fail_generation)

    run_file = run_fuzz(
        cases=2,
        seed=2100,
        backends=[],
        duration_s=None,
        config=ExperimentConfig(method_arm="contract_lattice_shared_cost_full"),
    )

    rows = read_jsonl(run_file)
    meta = load_json(run_meta_path(run_file))
    assert len(rows) == 2
    assert all(row["kind"] == "case_iteration_error" for row in rows)
    assert all(
        row["method_arm"]["arm_id"] == "contract_lattice_shared_cost_full"
        for row in rows
    )
    assert all(
        row["experiment_manifest"]["method_arm_id"]
        == "contract_lattice_shared_cost_full"
        for row in rows
    )
    assert [row["attempt_index"] for row in rows] == [0, 1]
    assert meta["attempted_cases"] == 2
    assert meta["executed_cases"] == 0
    assert meta["case_iteration_failures"] == 2


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
            "exploration_objective:cross_model_consistency",
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

    keys = runner_module._feedback_target_keys(
        guidance_row,
        operation_combo,
        target_capabilities=["op:join", "table:multi"],
    )

    assert "target:semi_anti_join_rewrite" in keys
    assert "feature:pattern:semi_anti_join_rewrite" in keys
    assert "feature:semantic_family:join_membership" in keys
    assert "feature:exploration_objective:cross_model_consistency" in keys
    assert "semantic_family:join_membership" in keys
    assert "exploration_objective:cross_model_consistency" in keys
    assert "combo:join_filter_groupby" in keys
    assert "semantic_signal:semi_anti_join_filter_pushdown" in keys
    assert "risk:semi_anti_join_filter_pushdown" in keys
    assert "semantic_family:conditional_semantics" in keys
    assert "semantic_signal:left_join_case_when_membership" in keys
    assert "risk:left_join_case_when_membership" in keys
    assert "capability:op:join" in keys
    assert "capability:table:multi" in keys
    assert "rows:many" not in keys
    assert "target:strings" not in keys
    assert "target:groupby" not in keys
    assert "feature:combo_risk:groupby_aggregation" not in keys


def test_configured_guidance_targets_include_structured_semantic_focus_targets():
    config = ExperimentConfig(
        guidance_targets=["groupby"],
        semantic_focus_families=["conditional_semantics"],
        semantic_focus_signals=["left_join_case_when_membership"],
        exploration_objective_rules=[
            {"objective": "adaptive consistency", "exact_features": ["op:join"]}
        ],
    )

    targets = runner_module._configured_guidance_targets(config)

    assert "groupby" in targets
    assert "semantic_family:conditional_semantics" in targets
    assert "semantic_signal:left_join_case_when_membership" in targets
    assert "exploration_objective:adaptive_consistency" in targets
    assert "conditional_semantics" not in targets
    assert "left_join_case_when_membership" not in targets


def test_run_fuzz_can_persist_generated_cases_and_checkpoint(tmp_path):
    case_log = tmp_path / "generated.cases.jsonl"
    run_file = run_fuzz(
        cases=2,
        seed=31,
        backends=[],
        duration_s=None,
        config=ExperimentConfig(method_arm="contract_lattice_shared_cost_full"),
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


def test_run_fuzz_adapts_candidate_pool_and_records_policy_metadata(tmp_path):
    case_log = tmp_path / "adaptive-pool.cases.jsonl"
    config = ExperimentConfig(
        guidance_strategy="guided",
        guidance_candidate_pool=4,
        enable_adaptive_candidate_pool=True,
        adaptive_candidate_pool_min_size=2,
        adaptive_candidate_pool_full_sweep_interval=0,
        adaptive_candidate_pool_calibration_cases=2,
        adaptive_candidate_pool_candidate_burst_cases=0,
    )

    run_file = run_fuzz(
        cases=4,
        seed=45,
        backends=[],
        config=config,
        case_log_file=case_log,
    )

    rows = read_jsonl(run_file)
    case_rows = read_jsonl(case_log)
    meta = load_json(run_meta_path(run_file))

    assert [row["candidate_pool_size"] for row in rows] == [4, 4, 2, 2]
    assert [row["candidate_pool_size"] for row in case_rows] == [4, 4, 2, 2]
    assert [row["candidate_pool_sampling"]["mode"] for row in rows] == [
        "calibration_full_pool",
        "calibration_full_pool",
        "reduced_pool",
        "reduced_pool",
    ]
    assert case_rows[2]["candidate_pool_sampling"]["sampled"] is True
    policy = meta["guidance"]["adaptive_candidate_pool"]
    assert policy["enabled"] is True
    assert policy["selected_candidate_budget"] == 12
    assert policy["mean_pool_size"] == 3.0
    assert policy["full_pool_cases"] == 2
    assert policy["reduced_pool_cases"] == 2


def test_run_fuzz_adaptive_pool_can_preserve_full_pool_seed_stride(tmp_path):
    case_log = tmp_path / "adaptive-pool-stride.cases.jsonl"
    config = ExperimentConfig(
        guidance_strategy="guided",
        guidance_candidate_pool=4,
        enable_adaptive_candidate_pool=True,
        adaptive_candidate_pool_min_size=2,
        adaptive_candidate_pool_full_sweep_interval=0,
        adaptive_candidate_pool_calibration_cases=1,
        adaptive_candidate_pool_candidate_burst_cases=0,
        adaptive_candidate_pool_preserve_seed_stride=True,
    )

    run_file = run_fuzz(
        cases=3,
        seed=45,
        backends=[],
        config=config,
        case_log_file=case_log,
    )

    rows = read_jsonl(run_file)
    case_rows = read_jsonl(case_log)
    meta = load_json(run_meta_path(run_file))

    assert [row["candidate_seed_start"] for row in rows] == [45, 49, 53]
    assert [row["candidate_seed_start"] for row in case_rows] == [45, 49, 53]
    assert [row["candidate_pool_size"] for row in rows] == [4, 2, 2]
    assert rows[1]["candidate_pool_sampling"]["generated_seed_advance"] == 2
    assert rows[1]["candidate_pool_sampling"]["applied_seed_advance"] == 4
    assert rows[1]["candidate_pool_sampling"]["skipped_seed_slots"] == 2
    policy = meta["guidance"]["adaptive_candidate_pool"]
    assert policy["preserve_seed_stride"] is True
    assert policy["generated_seed_advance"] == 8
    assert policy["applied_seed_advance"] == 12
    assert policy["skipped_seed_slots"] == 4
    assert meta["next_seed"] == 57


def test_run_fuzz_records_acceptance_compensated_seed_horizon_audit(tmp_path):
    case_log = tmp_path / "adaptive-pool-horizon.cases.jsonl"
    config = ExperimentConfig(
        guidance_strategy="guided",
        guidance_candidate_pool=4,
        enable_adaptive_candidate_pool=True,
        adaptive_candidate_pool_min_size=2,
        adaptive_candidate_pool_full_sweep_interval=0,
        adaptive_candidate_pool_calibration_cases=0,
        adaptive_candidate_pool_candidate_burst_cases=0,
        adaptive_candidate_pool_compensate_seed_horizon=True,
    )

    run_file = run_fuzz(
        cases=2,
        seed=57,
        backends=[],
        config=config,
        case_log_file=case_log,
    )

    rows = read_jsonl(run_file)
    meta = load_json(run_meta_path(run_file))

    assert all(
        row["candidate_pool_sampling"]["seed_horizon_policy"]
        == "acceptance_compensated"
        for row in rows
    )
    assert all(
        row["candidate_pool_sampling"]["full_pool_equivalent_seed_advance"]
        >= row["candidate_pool_sampling"]["generated_seed_advance"]
        for row in rows
    )
    policy = meta["guidance"]["adaptive_candidate_pool"]
    assert policy["preserve_seed_stride"] is True
    assert policy["compensate_seed_horizon"] is True
    assert policy["applied_seed_advance"] >= policy["generated_seed_advance"]


def test_sample_confirmation_can_reuse_full_confirmation_as_recheck_evidence():
    config = ExperimentConfig(
        candidate_recheck_count=2,
        backend_sample_confirmation_recheck_count=1,
    )
    payload = config.to_dict()

    confirmation, confirmation_payload = runner_module._sample_confirmation_config(
        config,
        payload,
    )

    assert config.candidate_recheck_count == 2
    assert confirmation.candidate_recheck_count == 1
    assert confirmation.backend_sample_confirmation_recheck_count == 1
    assert confirmation_payload["candidate_recheck_count"] == 1


def test_sample_confirmation_keeps_default_recheck_count_without_override():
    config = ExperimentConfig(candidate_recheck_count=2)
    payload = config.to_dict()

    confirmation, confirmation_payload = runner_module._sample_confirmation_config(
        config,
        payload,
    )

    assert confirmation is config
    assert confirmation_payload is payload


def test_promoted_default_confirms_status_ok_finding_signal_on_all_backends(
    monkeypatch,
):
    class FakeTargetContext:
        common_capabilities = ("op:select", "table:single")

        def target_dicts(self):
            return [{"name": backend} for backend in configured_backends]

        def to_dict(self):
            return {"common_capabilities": list(self.common_capabilities)}

    configured_backends = [
        "pandas",
        "polars",
        "duckdb",
        "pyarrow",
        "datafusion",
    ]
    calls: list[tuple[str, ...]] = []
    cost_model = BackendCostModel(
        source_path="profile.json",
        source_profile_id="p4.1-test",
        source_result_digest="result-test",
        metric="process_cpu",
        raw_costs={backend: 1.0 for backend in configured_backends},
        normalized_costs={backend: 1.0 for backend in configured_backends},
        digest="cost-model-p4.1-test",
    )

    def fake_generate_case(seed, **kwargs):
        return Case(
            case_id=f"case-{seed}",
            seed=seed,
            tables=[TableData("t0", [ColumnSpec("x", "int")], [{"x": seed}])],
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
        metamorphic_relation_order=None,
    ):
        calls.append(tuple(backends))
        finding = {
            "kind": "metamorphic_mismatch",
            "root_cause": "status_ok_metamorphic_root",
            "triage_verdict": "candidate_implementation_bug",
            "suspicious_backends": ["duckdb"],
            "signature": "status-ok-finding",
        }
        resolved_config = config or ExperimentConfig()
        return {
            "run_at": "2026-07-14T00:00:00Z",
            "case": case.to_dict(),
            "targets": target_specs or [],
            "raw_results": {},
            "normalized": {},
            "metamorphic": {},
            "findings": [finding],
            "candidate_recheck": {
                "enabled": False,
                "attempts": 0,
                "reproduced_keys": [],
                "non_reproduced_keys": [],
            },
            "config": config_payload or resolved_config.to_dict(),
            "method_arm": resolved_config.method_arm_manifest,
            "experiment_manifest": {},
            "environment": environment or {},
            "status": "ok",
            "duration_ms": 0.1,
            "stage_profile": {},
            "wall_time_profile": {},
            "process_cpu_profile": {},
            "execution_profile": {
                "backend_calls": len(backends),
                "backend_reported_total_ms": float(len(backends)),
            },
            "behavior_signature": f"behavior-{len(calls)}",
            "discovery_signature": f"discovery-{len(calls)}",
        }

    monkeypatch.setattr(runner_module, "generate_case", fake_generate_case)
    monkeypatch.setattr(runner_module, "run_loaded_case", fake_run_loaded_case)
    monkeypatch.setattr(runner_module, "target_context", lambda names: FakeTargetContext())
    monkeypatch.setattr(runner_module, "load_backend_cost_model", lambda root: cost_model)

    config = ExperimentConfig(
        enable_backend_session_reuse=False,
        enable_feedback=False,
        enable_artifact=False,
        enable_champion_corpus=False,
        enable_lhs_seeding=False,
        compress_run_log=False,
        log_level="full",
    )
    run_file = run_fuzz(
        cases=1,
        seed=5073,
        backends=configured_backends,
        config=config,
    )
    row = read_jsonl(run_file)[0]
    meta = load_json(run_meta_path(run_file))

    assert config.method_arm == "p8_candidate_v1"
    assert config.enable_parallel_backend_execution is False
    assert len(calls) == 2
    assert 2 <= len(calls[0]) < len(configured_backends)
    assert calls[1] == tuple(configured_backends)
    assert row["backend_sampling"]["confirmation_executed"] is True
    assert row["backend_sampling"]["screening"]["candidate_signal"] is True
    assert row["backend_sampling"]["confirmation_candidate_signal"] is True
    assert row["backend_sampling"]["candidate_burst_decision"][
        "candidate_detected"
    ] is True
    lattice = row["execution_lattice_selection"]["lattice_selection"]
    assert lattice["selector_mode"] == "shared_cost"
    assert lattice["budget_satisfied"] is True
    assert lattice["total_cost"] <= lattice["budget"]
    assert meta["method_arm"]["arm_id"] == "p8_candidate_v1"
    assert meta["execution"]["lattice"]["cost_model"]["digest"] == (
        "cost-model-p4.1-test"
    )


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
        method_arm="contract_lattice_shared_cost_full",
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


def test_run_fuzz_fresh_discovery_uses_generation_time_replay_gate(tmp_path):
    case_log = tmp_path / "fresh-discovery.cases.jsonl"
    config = ExperimentConfig(
        method_arm="contract_lattice_shared_cost_full",
        generator_profile="discovery",
        enable_replay_bug=False,
    )

    run_file = run_fuzz(cases=1, seed=20, backends=[], config=config, case_log_file=case_log)

    row = read_jsonl(run_file)[0]
    case_log_row = read_jsonl(case_log)[0]
    meta = load_json(run_meta_path(run_file))
    assert meta["config"]["generator_profile"] == "discovery"
    assert meta["effective_generator_profile"] == "discovery"
    assert meta["replay_bug_filter"]["filtered_candidates"] > 0
    assert row["case"].get("metadata", {}).get("mixed_generator_profile") != "wide_offset_topk"
    assert row["replay_filter"]["filtered_before_candidate"] > 0
    assert row["replay_filter"]["last_skip_reason"] == "known_replay_source_issue"
    assert case_log_row["case"].get("metadata", {}).get("mixed_generator_profile") != "wide_offset_topk"


def test_run_fuzz_custom_replay_source_gate_keeps_requested_generator_profile(tmp_path):
    case_log = tmp_path / "custom-source-gate.cases.jsonl"
    config = ExperimentConfig(
        method_arm="contract_lattice_shared_cost_full",
        generator_profile="discovery",
        enable_replay_bug=False,
        replay_bug_source_issues=[],
    )

    run_file = run_fuzz(cases=1, seed=20, backends=[], config=config, case_log_file=case_log)

    case_log_row = read_jsonl(case_log)[0]
    meta = load_json(run_meta_path(run_file))
    assert meta["effective_generator_profile"] == "discovery"
    assert meta["replay_bug_filter"]["filtered_candidates"] == 0
    assert case_log_row["case"]["metadata"]["mixed_generator_profile"] == "wide_offset_topk"


def test_run_fuzz_replay_policy_allows_replay_profile(tmp_path):
    case_log = tmp_path / "replay.cases.jsonl"
    config = ExperimentConfig(
        method_arm="contract_lattice_shared_cost_full",
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


def test_run_fuzz_enforces_fresh_source_quota_before_feedback(tmp_path, monkeypatch):
    instances = []

    class FakeFeedbackState:
        def __init__(self, **kwargs):
            self.last_persisted_to_disk = False
            self.last_candidate_source = "generated"
            self.last_candidate_metadata = {}
            self.source_scheduler = None
            self.recorded_sources = []
            self.quality_context_calls = []
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
                    "target_keys": ["semantic_family:cast_semantics"],
                },
            }
            return generated

        def candidate_quality_context(self, case, *, target_keys=None, profile_key=None):
            self.quality_context_calls.append(
                {
                    "case_id": case.case_id,
                    "target_keys": list(target_keys or []),
                    "profile_key": profile_key or "",
                }
            )
            return {
                "cluster_key": "profile=generic|targets=semantic_family_cast_semantics|ops=empty",
                "profile_key": "generic",
                "target_keys": list(target_keys or []),
                "target_key_count": len(target_keys or []),
                "archive_known": True,
                "archive_elite_indexes": [0],
                "archive_seed_count": 1,
                "archive_outcome_count": 2,
                "archive_cluster_reward": 0.75,
                "archive_health_penalty": 0.05,
                "cluster_count": 3,
                "cluster_feedback_reward": 0.5,
                "cluster_feedback_count": 2,
                "cluster_novelty_score": 0.1,
                "recent_cluster_pulls": 1,
            }

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
            disagreement_descriptor=None,
            case_fingerprint=None,
            schedule_delta=0.0,
        ):
            return True

        def record_candidate_result(self, candidate_source, *, has_finding, is_new_behavior, preflight, **reward_signals):
            self.recorded_sources.append(candidate_source)
            return 1.25

    monkeypatch.setattr(runner_module, "FeedbackState", FakeFeedbackState)

    case_log = tmp_path / "feedback.cases.jsonl"
    config = ExperimentConfig(
        method_arm="contract_lattice_shared_cost_full",
        enable_local_source_scheduler=True,
    )

    run_file = run_fuzz(cases=1, seed=47, backends=[], config=config, case_log_file=case_log)

    row = read_jsonl(run_file)[0]
    case_log_row = read_jsonl(case_log)[0]
    # The coordinator starts below its 60% fresh-generation floor, so it
    # deliberately bypasses a feedback mutation even when one is available.
    assert row["candidate_source"] == "generated"
    assert case_log_row["candidate_source"] == "generated"
    assert row["seed_lineage"]["parent_case_id"] == ""
    assert row["mutation"]["operator"] == "generated"
    assert row["feedback_decision"] == {}
    assert row["quality_archive_context"]["archive_known"] is True
    assert row["quality_archive_context"]["archive_cluster_reward"] == 0.75
    assert row["quality_archive_context"]["target_keys"]
    assert any(
        key.startswith("capability:")
        for key in row["quality_archive_context"]["target_keys"]
    )
    assert row["operation_combo"]["operation_count"] == len(row["case"]["program"]["operations"])
    assert row["source_reward"] == 1.25
    assert row["stored_in_feedback_corpus"] is True
    assert row["feedback_skip_reason"] == ""
    assert case_log_row["seed_lineage"]["parent_case_id"] == ""
    assert case_log_row["mutation"]["operator"] == "generated"
    assert case_log_row["feedback_decision"] == {}
    assert case_log_row["quality_archive_context"] == row["quality_archive_context"]
    assert case_log_row["case"]["metadata"]["quality_archive_context"] == row["quality_archive_context"]
    assert "operation_combo" in case_log_row
    assert instances[0].recorded_sources == ["generated"]
    assert instances[0].quality_context_calls == [
        {
            "case_id": "case-00000047",
            "target_keys": row["quality_archive_context"]["target_keys"],
            "profile_key": "",
        }
    ]


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
    assert row["disagreement_descriptor"]["pair_count"] == 0
    assert len(row["case_fingerprint"]["minhash_signature"]) == 64
    assert "stage_profile" in row
    assert "wall_time_profile" in row
    assert "process_cpu_profile" in row
    assert row["method_arm"]["arm_id"] == "p8_candidate_v1"
    assert row["experiment_manifest"]["method_arm_digest"] == row["method_arm"]["digest"]
    assert meta["method_arm"] == row["method_arm"]
    assert meta["experiment_manifest"]["method_arm_id"] == (
        "p8_candidate_v1"
    )
    assert row["fuzz_iteration"]["case_id"] == row["case"]["case_id"]
    assert row["fuzz_iteration"]["stage_timings"]["total_case_wall_ms"] == row["stage_profile"]["total_case_wall_ms"]
    assert meta["stage_profile"]["case_count"] == 1
    assert meta["stage_profile"]["totals_ms"]["total_case_wall_ms"] >= 0.0
    assert meta["wall_time_profile"]["totals_ms"] == row["wall_time_profile"]
    assert meta["process_cpu_profile"]["totals_ms"] == row["process_cpu_profile"]
    assert row["wall_time_profile"]["total_wall_ms"] == pytest.approx(
        sum(
            value
            for key, value in row["wall_time_profile"].items()
            if key != "total_wall_ms"
        )
    )
    assert row["process_cpu_profile"]["total_process_cpu_ms"] == pytest.approx(
        sum(
            value
            for key, value in row["process_cpu_profile"].items()
            if key != "total_process_cpu_ms"
        )
    )


def test_run_fuzz_fresh_session_control_records_nonpersistent_mode():
    config = ExperimentConfig(
        enable_backend_session_reuse=False,
        enable_parallel_backend_execution=False,
        enable_metamorphic_oracle=False,
        enable_feedback=False,
        enable_artifact=False,
    )

    run_file = run_fuzz(cases=1, seed=5101, backends=[], config=config)
    meta = load_json(run_meta_path(run_file))

    assert meta["execution"]["session"]["persistent"] is False
    assert meta["execution"]["session"]["mode"] == "fresh_per_execution"
    assert meta["config"]["enable_backend_session_reuse"] is False


@pytest.mark.skipif(
    any(importlib.util.find_spec(name) is None for name in ["pandas", "duckdb"]),
    reason="pandas and duckdb are not installed",
)
def test_run_fuzz_uses_fresh_execution_when_target_forbids_session_reuse(tmp_path):
    def generate_case_for_storage(seed, **_kwargs):
        return Case(
            f"case-storage-{seed}",
            seed,
            [TableData("t0", [ColumnSpec("x", "int", nullable=False)], [{"x": 1}])],
            Program(f"prog-storage-{seed}", seed, [{"op": "select", "columns": ["x"]}]),
        )

    config = ExperimentConfig(
        candidate_recheck_count=0,
        compress_run_log=False,
        coordinator_candidate_pool=1,
        enable_artifact=False,
        enable_backend_sampling=False,
        enable_backend_session_reuse=True,
        enable_champion_corpus=False,
        enable_feedback=False,
        enable_lhs_seeding=False,
        enable_metamorphic_oracle=False,
        enable_parallel_backend_execution=False,
        guidance_candidate_pool=1,
        guidance_strategy="random",
        log_level="full",
    )

    run_file = run_fuzz(
        cases=1,
        seed=5102,
        backends=["pandas", "duckdb_persistent"],
        config=config,
        runs_dir=tmp_path / "runs",
        corpus_dir=tmp_path / "corpus",
        generate_case_fn=generate_case_for_storage,
    )
    meta = load_json(run_meta_path(run_file))
    row = read_jsonl(run_file)[0]

    assert meta["executed_cases"] == 1
    assert meta["case_iteration_failures"] == 0
    assert row["status"] == "ok"
    resolution = meta["execution"]["session_reuse_resolution"]
    assert resolution["requested"] is True
    assert resolution["effective"] is False
    assert resolution["reason"] == "fresh_only_backend_present"
    assert resolution["fresh_only_backends"] == ["duckdb_persistent"]
    assert meta["execution"]["session"]["mode"] == "fresh_per_execution"


def test_run_fuzz_records_run_provenance(monkeypatch):
    expected = {
        "schema_version": "run-provenance-v1",
        "vcs": {"git_commit": "abc123", "git_commit_short": "abc123", "git_branch": "main", "workspace_dirty": False},
        "launch": {"source": "closed_loop_tmux", "session": "s", "duration": "12h", "batch_duration": "10m", "log_prefix": "x", "launch_script": "start.sh"},
        "harness": {"authority": True, "freeze_intent": True, "latest_code_claim": True, "evidence_role": "latest_live_authority_12h"},
        "freeze_artifacts": {"strategy_snapshot": "reports/strategy-snapshots/frozen.json"},
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
    assert row["execution_profile"]["backend_count"] == 0
    assert row["execution_profile"]["parallel_backend_execution"] is False
    assert row["fuzz_iteration"]["case_id"] == row["case"]["case_id"]
    assert row["fuzz_iteration"]["execution_profile"]["backend_count"] == 0
    assert row["guidance"]["strategy"] == "random"
    assert "frontier_conformance" in row["guidance"]


def test_run_fuzz_can_disable_run_log_compression():
    config = ExperimentConfig(compress_run_log=False)

    run_file = run_fuzz(cases=1, seed=53, backends=[], config=config)

    assert run_file.name.endswith(".jsonl")
    assert not run_file.name.endswith(".jsonl.gz")
    assert load_json(run_meta_path(run_file))["config"]["compress_run_log"] is False


def test_run_fuzz_records_layered_config_metadata():
    config = ExperimentConfig(
        guidance_strategy="guided",
        guidance_candidate_pool=3,
        guidance_targets=["topk"],
        enable_local_source_scheduler=True,
        enable_parallel_backend_execution=False,
        log_level="compact",
    )

    run_file = run_fuzz(cases=1, seed=55, backends=[], config=config)

    meta = load_json(run_meta_path(run_file))
    assert meta["config"]["guidance_strategy"] == "guided"
    assert "guidance" not in meta["config"]
    assert meta["config_layers"]["guidance"]["strategy"] == "guided"
    assert meta["config_layers"]["guidance"]["candidate_pool"] == 3
    assert meta["config_layers"]["guidance"]["targets"] == ["topk"]
    assert "topk" in meta["config_layers"]["guidance"]["effective_targets"]
    assert meta["config_layers"]["feedback"]["enable_local_source_scheduler"] is True
    assert meta["config_layers"]["execution"]["enable_parallel_backend_execution"] is False
    assert meta["config_layers"]["method"]["arm_id"] == (
        "p8_candidate_v1"
    )
    assert meta["config"]["config_digest"] == meta["config_layers"]["config_digest"]


def test_run_fuzz_records_lattice_arm_without_implicit_boolean_reconstruction():
    config = ExperimentConfig(
        method_arm="contract_lattice_static",
        method_arm_overrides={"node_budget": 2},
        log_level="minimal",
    )

    run_file = run_fuzz(cases=1, seed=5501, backends=[], config=config)

    row = read_jsonl(run_file)[0]
    meta = load_json(run_meta_path(run_file))
    assert row["method_arm"]["base_arm_id"] == "contract_lattice_static"
    assert row["method_arm"]["registered"] is False
    assert row["method_arm"]["settings"]["execution_mode"] == "lattice"
    assert meta["execution"]["lattice"]["selector_mode"] == "static"
    assert meta["method_arm"]["digest"] == row["method_arm"]["digest"]


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


def test_run_fuzz_injects_cross_version_champion_corpus(tmp_path, monkeypatch):
    champion_path = tmp_path / "champions.jsonl"
    registry = ChampionRegistry(champion_path)
    donor = Case(
        "case-donor",
        9,
        [TableData("t0", [ColumnSpec("x", "int")], [{"x": 1}])],
        Program("prog-donor", 9, [{"op": "filter", "column": "x", "cmp": ">", "value": 0}]),
    )
    assert registry.promote_if_stable(donor, ["family@engine"], threshold=1, version_id="old", stability=3)
    monkeypatch.setattr(runner_module, "DEFAULT_CHAMPION_CORPUS_PATH", champion_path)

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
            "environment": {},
            "status": "ok",
            "duration_ms": 0.1,
            "behavior_signature": "behavior-a",
            "discovery_signature": "discovery-a",
        }

    monkeypatch.setattr(runner_module, "run_loaded_case", fake_run_loaded_case)

    run_file = run_fuzz(
        cases=1,
        seed=81,
        backends=[],
        config=ExperimentConfig(log_level="minimal", target_version="new"),
        persist_closed_loop_state=True,
    )

    meta = load_json(run_meta_path(run_file))
    state = load_json(closed_loop_state_path(run_file))
    feedback = state["feedback"]

    assert meta["champion_corpus"]["injected_count"] == 1
    assert feedback["interesting_cases"][0]["metadata"]["champion_seed"]["source_version_id"] == "old"
    assert feedback["case_family_keys"][0] == ["family@engine"]


def test_run_fuzz_can_disable_cross_version_champion_corpus(tmp_path, monkeypatch):
    champion_path = tmp_path / "champions.jsonl"
    registry = ChampionRegistry(champion_path)
    donor = Case(
        "case-donor",
        9,
        [TableData("t0", [ColumnSpec("x", "int")], [{"x": 1}])],
        Program("prog-donor", 9, [{"op": "filter", "column": "x", "cmp": ">", "value": 0}]),
    )
    assert registry.promote_if_stable(donor, ["family@engine"], threshold=1, version_id="old", stability=3)
    monkeypatch.setattr(runner_module, "DEFAULT_CHAMPION_CORPUS_PATH", champion_path)

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
            "environment": {},
            "status": "ok",
            "duration_ms": 0.1,
            "behavior_signature": "behavior-a",
            "discovery_signature": "discovery-a",
        }

    monkeypatch.setattr(runner_module, "run_loaded_case", fake_run_loaded_case)

    run_file = run_fuzz(
        cases=1,
        seed=82,
        backends=[],
        config=ExperimentConfig(
            log_level="minimal",
            target_version="new",
            enable_champion_corpus=False,
        ),
        persist_closed_loop_state=True,
    )

    meta = load_json(run_meta_path(run_file))
    state = load_json(closed_loop_state_path(run_file))

    assert meta["champion_corpus"]["enabled"] is False
    assert meta["champion_corpus"]["injected_count"] == 0
    assert all(
        "champion_seed" not in case.get("metadata", {})
        for case in state["feedback"]["interesting_cases"]
    )


def test_run_fuzz_passes_lhs_schema_spec_during_cold_start(monkeypatch):
    observed_specs = []

    def fake_generate_case(seed, *, type_aware=True, profile="common", schema_spec=None):
        observed_specs.append(schema_spec)
        return Case(
            f"case-{seed}",
            seed,
            [TableData("t0", [ColumnSpec("x", "int")], [{"x": seed}])],
            Program(f"prog-{seed}", seed, [{"op": "limit", "n": 1}]),
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
            "environment": {},
            "status": "ok",
            "duration_ms": 0.1,
            "behavior_signature": f"behavior-{case.seed}",
            "discovery_signature": f"discovery-{case.seed}",
        }

    monkeypatch.setattr(runner_module, "generate_case", fake_generate_case)
    monkeypatch.setattr(runner_module, "run_loaded_case", fake_run_loaded_case)

    run_file = run_fuzz(
        cases=2,
        seed=90,
        backends=[],
        config=ExperimentConfig(
            method_arm="contract_lattice_shared_cost_full",
            log_level="minimal",
        ),
    )
    meta = load_json(run_meta_path(run_file))

    assert observed_specs
    assert all(spec is not None for spec in observed_specs)
    assert meta["lhs_seeding"]["enabled"] is True


def test_run_fuzz_adaptive_profile_pool_learns_and_persists(monkeypatch):
    generated_profiles: list[str] = []

    def fake_generate_case(seed, *, type_aware=True, profile="common"):
        generated_profiles.append(profile)
        return Case(
            case_id=f"case-{seed}-{profile}",
            seed=seed,
            tables=[TableData("t0", [ColumnSpec("x", "int")], [{"x": seed}])],
            program=Program(f"prog-{seed}", seed, [{"op": "select", "columns": ["x"]}]),
            metadata={"generator_profile": profile},
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
        profile = case.metadata.get("generator_profile", "common")
        findings = []
        status = "ok"
        if profile == "discovery_fresh":
            status = "bug"
            findings = [
                {
                    "kind": "differential_mismatch",
                    "root_cause": "adaptive_profile_probe",
                    "triage_verdict": "candidate_implementation_bug",
                    "suspicious_backends": ["engine"],
                    "signature": f"sig-{case.seed}",
                }
            ]
        return {
            "run_at": "2026-05-31T00:00:00Z",
            "case": case.to_dict(),
            "targets": [],
            "raw_results": {},
            "normalized": {},
            "metamorphic": {},
            "findings": findings,
            "candidate_recheck": {"enabled": False, "attempts": 0, "reproduced_keys": [], "non_reproduced_keys": []},
            "config": config_payload or (config or ExperimentConfig()).to_dict(),
            "environment": {},
            "status": status,
            "duration_ms": 0.1,
            "behavior_signature": f"behavior-{case.case_id}",
            "discovery_signature": f"discovery-{case.case_id}",
        }

    monkeypatch.setattr(runner_module, "generate_case", fake_generate_case)
    monkeypatch.setattr(runner_module, "run_loaded_case", fake_run_loaded_case)

    config = ExperimentConfig(
        method_arm="contract_lattice_shared_cost_full",
        enable_feedback=True,
        generator_profile="common",
        generator_profile_pool=["common", "discovery_fresh"],
        generator_profile_learning_weight=1.0,
        log_level="minimal",
    )
    first_run = run_fuzz(
        cases=2,
        seed=91,
        backends=[],
        config=config,
        persist_closed_loop_state=True,
    )
    first_state = load_json(closed_loop_state_path(first_run))
    first_rows = read_jsonl(first_run)
    first_meta = load_json(run_meta_path(first_run))

    generated_profiles.clear()
    second_run = run_fuzz(
        cases=1,
        seed=101,
        backends=[],
        config=config,
        closed_loop_state=first_state,
        persist_closed_loop_state=True,
    )
    second_row = read_jsonl(second_run)[0]
    second_state = load_json(closed_loop_state_path(second_run))

    assert {row["selected_generator_profile"] for row in first_rows} == {"common", "discovery_fresh"}
    assert "generator_profile" in first_state["feedback"]["adaptive_learning"]["bandits"]
    assert (
        first_meta["closed_loop_state_summary"]["adaptive_learning_health"]["schema_version"]
        == "adaptive-learning-health-v1"
    )
    assert first_meta["closed_loop_state_summary"]["adaptive_learning_health"]["total_pulls"] >= 2
    assert generated_profiles[0] == "discovery_fresh"
    assert second_row["generator_profile_selection"]["strategy"] == "contextual_bandit"
    assert second_row["selected_generator_profile"] == "discovery_fresh"
    assert second_state["feedback"]["adaptive_learning"]["bandits"]["generator_profile"]["total_pulls"] >= 3


def test_run_fuzz_profile_pool_filters_profiles_by_target_capability(monkeypatch):
    generated_profiles: list[str] = []

    class FakeTargetContext:
        common_capabilities = ("table:single", "op:filter", "op:join", "op:sort", "op:limit")

        def target_dicts(self):
            return []

        def to_dict(self):
            return {"common_capabilities": list(self.common_capabilities)}

    def fake_generate_case(seed, *, type_aware=True, profile="common"):
        generated_profiles.append(profile)
        return Case(
            case_id=f"case-{seed}-{profile}",
            seed=seed,
            tables=[TableData("t0", [ColumnSpec("x", "int")], [{"x": seed}])],
            program=Program(f"prog-{seed}", seed, [{"op": "select", "columns": ["x"]}]),
            metadata={"generator_profile": profile},
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
        return {
            "run_at": "2026-05-31T00:00:00Z",
            "case": case.to_dict(),
            "targets": [],
            "raw_results": {},
            "normalized": {},
            "metamorphic": {},
            "findings": [],
            "candidate_recheck": {"enabled": False, "attempts": 0, "reproduced_keys": [], "non_reproduced_keys": []},
            "config": config_payload or (config or ExperimentConfig()).to_dict(),
            "environment": {},
            "status": "ok",
            "duration_ms": 0.1,
            "behavior_signature": f"behavior-{case.case_id}",
            "discovery_signature": f"discovery-{case.case_id}",
        }

    monkeypatch.setattr(runner_module, "target_context", lambda backends: FakeTargetContext())
    monkeypatch.setattr(runner_module, "generate_case", fake_generate_case)
    monkeypatch.setattr(runner_module, "run_loaded_case", fake_run_loaded_case)

    config = ExperimentConfig(
        method_arm="contract_lattice_shared_cost_full",
        generator_profile="common",
        generator_profile_pool=["common", "partitioned_running_sum", "join_null_sort"],
        generator_profile_learning_weight=1.0,
        log_level="compact",
    )
    run_file = run_fuzz(cases=2, seed=111, backends=[], config=config)
    rows = read_jsonl(run_file)
    meta = load_json(run_meta_path(run_file))

    assert "partitioned_running_sum" not in generated_profiles
    assert set(generated_profiles).issubset({"common", "join_null_sort"})
    assert meta["generator_profile_pool"] == ["common", "join_null_sort"]
    dropped = meta["generator_profile_pool_metadata"]["dropped"]
    assert dropped[0]["profile"] == "partitioned_running_sum"
    assert "op:running_sum" in dropped[0]["missing"]
    assert rows[0]["generator_profile_selection"]["profile_pool_metadata"]["dropped"] == dropped


def test_run_fuzz_profile_capability_filter_can_be_disabled(monkeypatch):
    generated_profiles: list[str] = []

    class FakeTargetContext:
        common_capabilities = ("table:single", "op:filter", "op:join", "op:sort", "op:limit")

        def target_dicts(self):
            return []

        def to_dict(self):
            return {"common_capabilities": list(self.common_capabilities)}

    def fake_generate_case(seed, *, type_aware=True, profile="common"):
        generated_profiles.append(profile)
        return Case(
            case_id=f"case-{seed}-{profile}",
            seed=seed,
            tables=[TableData("t0", [ColumnSpec("x", "int")], [{"x": seed}])],
            program=Program(f"prog-{seed}", seed, [{"op": "select", "columns": ["x"]}]),
            metadata={"generator_profile": profile},
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
        return {
            "run_at": "2026-05-31T00:00:00Z",
            "case": case.to_dict(),
            "targets": [],
            "raw_results": {},
            "normalized": {},
            "metamorphic": {},
            "findings": [],
            "candidate_recheck": {"enabled": False, "attempts": 0, "reproduced_keys": [], "non_reproduced_keys": []},
            "config": config_payload or (config or ExperimentConfig()).to_dict(),
            "environment": {},
            "status": "ok",
            "duration_ms": 0.1,
            "behavior_signature": f"behavior-{case.case_id}",
            "discovery_signature": f"discovery-{case.case_id}",
        }

    monkeypatch.setattr(runner_module, "target_context", lambda backends: FakeTargetContext())
    monkeypatch.setattr(runner_module, "generate_case", fake_generate_case)
    monkeypatch.setattr(runner_module, "run_loaded_case", fake_run_loaded_case)

    config = ExperimentConfig(
        method_arm="contract_lattice_shared_cost_full",
        generator_profile="common",
        generator_profile_pool=["common", "partitioned_running_sum"],
        generator_profile_learning_weight=1.0,
        enable_profile_capability_filter=False,
        log_level="compact",
    )
    run_file = run_fuzz(cases=2, seed=211, backends=[], config=config)
    rows = read_jsonl(run_file)
    meta = load_json(run_meta_path(run_file))

    assert "partitioned_running_sum" in generated_profiles
    assert meta["generator_profile_pool"] == ["common", "partitioned_running_sum"]
    assert meta["generator_profile_pool_metadata"]["dropped"] == []
    assert meta["generator_profile_pool_metadata"]["capability_filter_enabled"] is False
    assert rows[0]["generator_profile_selection"]["profile_pool_metadata"]["capability_aware"] is False


def test_run_fuzz_records_per_case_objective_mr_and_version_learning(monkeypatch):
    received_relation_orders: list[list[str]] = []

    def fake_generate_case(seed, *, type_aware=True, profile="common"):
        return Case(
            case_id=f"case-{seed}",
            seed=seed,
            tables=[
                TableData(
                    "t0",
                    [ColumnSpec("id", "int", nullable=False), ColumnSpec("x", "int")],
                    [{"id": 1, "x": 1}, {"id": 2, "x": 2}],
                )
            ],
            program=Program(f"prog-{seed}", seed, [{"op": "select", "columns": ["id", "x"]}]),
            metadata={"generator_profile": profile},
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
        metamorphic_relation_order=None,
    ):
        received_relation_orders.append(list(metamorphic_relation_order or []))
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
            "behavior_signature": f"behavior-{case.case_id}",
            "discovery_signature": f"discovery-{case.case_id}",
        }

    monkeypatch.setattr(runner_module, "generate_case", fake_generate_case)
    monkeypatch.setattr(runner_module, "run_loaded_case", fake_run_loaded_case)

    config = ExperimentConfig(
        method_arm="contract_lattice_shared_cost_full",
        enable_feedback=True,
        enable_metamorphic_oracle=True,
        metamorphic_variant_limit=1,
        semantic_objective_learning_weight=1.0,
        metamorphic_relation_learning_weight=1.0,
        version_pair_learning_weight=1.0,
        target_version="latest",
        fixed_version="fixed",
        persist_feedback_corpus=False,
        log_level="compact",
    )
    run_file = run_fuzz(
        cases=1,
        seed=91,
        backends=[],
        config=config,
        persist_closed_loop_state=True,
    )
    row = read_jsonl(run_file)[0]
    state = load_json(closed_loop_state_path(run_file))
    bandits = state["feedback"]["adaptive_learning"]["bandits"]

    assert row["semantic_objective_selection"]["strategy"] == "contextual_bandit_warmup"
    assert row["metamorphic_relation_selection"]["action"] == "input_partition_union_all"
    assert row["version_pair_selection"]["action"] == "latest->fixed"
    assert row["selected_version_pair"] == "latest->fixed"
    assert received_relation_orders[0][0] == "input_partition_union_all"
    assert bandits["semantic_objective"]["total_pulls"] == 1
    assert bandits["metamorphic_relation"]["total_pulls"] == 1
    assert bandits["version_pair"]["total_pulls"] == 1


def test_run_fuzz_version_pair_pool_learns_and_updates_case_config(monkeypatch):
    observed_version_pairs: list[str] = []

    def fake_generate_case(seed, *, type_aware=True, profile="common"):
        return Case(
            case_id=f"case-{seed}",
            seed=seed,
            tables=[
                TableData(
                    "t0",
                    [ColumnSpec("id", "int", nullable=False), ColumnSpec("x", "int")],
                    [{"id": 1, "x": 1}, {"id": 2, "x": 2}],
                )
            ],
            program=Program(f"prog-{seed}", seed, [{"op": "select", "columns": ["id", "x"]}]),
            metadata={"generator_profile": profile},
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
        metamorphic_relation_order=None,
    ):
        payload = config_payload or (config or ExperimentConfig()).to_dict()
        pair = f"{payload.get('target_version', '')}->{payload.get('fixed_version', '')}".rstrip("->")
        observed_version_pairs.append(pair)
        findings = []
        status = "ok"
        if pair == "latest->preview":
            status = "bug"
            findings = [
                {
                    "kind": "differential_mismatch",
                    "root_cause": "version_pair_probe",
                    "triage_verdict": "candidate_implementation_bug",
                    "suspicious_backends": ["engine"],
                    "signature": f"sig-{case.seed}",
                }
            ]
        return {
            "run_at": "2026-05-31T00:00:00Z",
            "case": case.to_dict(),
            "targets": target_specs or [],
            "raw_results": {},
            "normalized": {},
            "metamorphic": {},
            "findings": findings,
            "candidate_recheck": {"enabled": False, "attempts": 0, "reproduced_keys": [], "non_reproduced_keys": []},
            "config": payload,
            "environment": environment or {},
            "status": status,
            "duration_ms": 0.1,
            "behavior_signature": f"behavior-{case.case_id}-{pair}",
            "discovery_signature": f"discovery-{case.case_id}-{pair}",
        }

    monkeypatch.setattr(runner_module, "generate_case", fake_generate_case)
    monkeypatch.setattr(runner_module, "run_loaded_case", fake_run_loaded_case)

    config = ExperimentConfig(
        enable_feedback=True,
        version_pair_pool=["latest->fixed", "latest->preview"],
        version_pair_learning_weight=1.0,
        target_version="latest",
        fixed_version="fixed",
        persist_feedback_corpus=False,
        log_level="compact",
    )
    first_run = run_fuzz(
        cases=2,
        seed=191,
        backends=[],
        config=config,
        persist_closed_loop_state=True,
    )
    first_rows = read_jsonl(first_run)
    first_meta = load_json(run_meta_path(first_run))
    first_state = load_json(closed_loop_state_path(first_run))

    observed_version_pairs.clear()
    second_run = run_fuzz(
        cases=1,
        seed=201,
        backends=[],
        config=config,
        closed_loop_state=first_state,
        persist_closed_loop_state=True,
    )
    second_row = read_jsonl(second_run)[0]
    second_state = load_json(closed_loop_state_path(second_run))

    assert {row["selected_version_pair"] for row in first_rows} == {"latest->fixed", "latest->preview"}
    assert first_meta["version_pair_pool"] == ["latest->fixed", "latest->preview"]
    assert second_row["version_pair_selection"]["strategy"] == "contextual_bandit"
    assert second_row["selected_version_pair"] == "latest->preview"
    assert observed_version_pairs[0] == "latest->preview"
    assert second_row["config"]["target_version"] == "latest"
    assert second_row["config"]["fixed_version"] == "preview"
    assert second_state["feedback"]["adaptive_learning"]["bandits"]["version_pair"]["total_pulls"] >= 3


def test_run_fuzz_records_backend_pair_priority_and_learns_pair_rewards(monkeypatch):
    class FakeTargetContext:
        common_capabilities = ("op:select", "table:single")

        def target_dicts(self):
            return []

        def to_dict(self):
            return {"common_capabilities": list(self.common_capabilities)}

    def fake_generate_case(seed, *, type_aware=True, profile="common"):
        return Case(
            case_id=f"case-{seed}",
            seed=seed,
            tables=[TableData("t0", [ColumnSpec("x", "int")], [{"x": seed}])],
            program=Program(f"prog-{seed}", seed, [{"op": "select", "columns": ["x"]}]),
            metadata={"generator_profile": profile},
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
        metamorphic_relation_order=None,
    ):
        descriptor = {
            "backend_groups": [["left"], ["right"], ["third"]],
            "pair_disagrees": [
                {"left": "left", "right": "right", "disagrees": True},
                {"left": "left", "right": "third", "disagrees": False},
                {"left": "right", "right": "third", "disagrees": True},
            ],
            "pair_count": 2,
            "column_classes": {"x": "numeric"},
            "primary_root_cause": "window_boundary",
            "mismatch_class": "value",
            "backend_statuses": {backend: "ok" for backend in backends},
            "feature_tokens": [
                "mismatch:value",
                "root:window_boundary",
                "disagree_pair:left|right",
                "disagree_pair:right|third",
            ],
        }
        fingerprint = {
            "minhash_signature": [case.seed] * 64,
            "op_skeleton_hash": "select-hash",
            "type_mix_token": "num1",
            "null_density_bucket": 0,
            "row_mass_bucket": 1,
            "column_count": 1,
            "feature_tokens": ["fp_op:select-hash", "fp_type_mix:num1", "fp_column_count:1"],
        }
        findings = [
            {
                "kind": "semantic_output_mismatch",
                "root_cause": "window_boundary",
                "triage_verdict": "candidate_implementation_bug",
                "suspicious_backends": ["right"],
                "signature": f"sig-{case.seed}",
                "mismatch_class": "value",
            }
        ]
        row_case = case.to_dict()
        row_case.setdefault("metadata", {})["disagreement_descriptor"] = descriptor
        row_case.setdefault("metadata", {})["case_fingerprint"] = fingerprint
        return {
            "run_at": "2026-05-31T00:00:00Z",
            "case": row_case,
            "targets": target_specs or [],
            "raw_results": {},
            "normalized": {},
            "metamorphic": {},
            "findings": findings,
            "candidate_recheck": {"enabled": False, "attempts": 0, "reproduced_keys": [], "non_reproduced_keys": []},
            "config": config_payload or (config or ExperimentConfig()).to_dict(),
            "environment": environment or {},
            "status": "bug",
            "duration_ms": 0.1,
            "behavior_signature": f"behavior-{case.case_id}",
            "discovery_signature": f"discovery-{case.case_id}",
            "disagreement_descriptor": descriptor,
            "case_fingerprint": fingerprint,
        }

    monkeypatch.setattr(runner_module, "generate_case", fake_generate_case)
    monkeypatch.setattr(runner_module, "run_loaded_case", fake_run_loaded_case)
    monkeypatch.setattr(runner_module, "make_backend", lambda backend: object())
    monkeypatch.setattr(runner_module, "target_context", lambda backends: FakeTargetContext())

    config = ExperimentConfig(
        method_arm="contract_ccs_cartesian",
        enable_feedback=True,
        backend_pair_learning_weight=1.0,
        backend_pair_priority_limit=2,
        persist_feedback_corpus=False,
        log_level="compact",
    )
    run_file = run_fuzz(
        cases=2,
        seed=301,
        backends=["third", "right", "left"],
        config=config,
        persist_closed_loop_state=True,
    )
    rows = read_jsonl(run_file)
    meta = load_json(run_meta_path(run_file))
    state = load_json(closed_loop_state_path(run_file))
    bandit = state["feedback"]["adaptive_learning"]["bandits"]["backend_pair"]
    rewards = {
        arm["action_id"]: arm["total_reward"]
        for arm in bandit["arms"]
    }

    assert meta["backend_pair_pool"] == ["left|right", "left|third", "right|third"]
    assert rows[0]["backend_pair_selection"]["strategy"] == "contextual_bandit_warmup"
    assert rows[0]["backend_pair_priority"] == ["left|right", "left|third"]
    assert rows[0]["backend_pair_feedback"]["recorded"] == 3
    assert set(rows[0]["backend_pair_feedback"]["disagree_pairs"]) == {"left|right", "right|third"}
    assert rows[1]["backend_pair_selection"]["strategy"] == "contextual_bandit"
    assert rows[1]["backend_pair_priority"][0] in {"left|right", "right|third"}
    assert bandit["total_pulls"] == 6
    assert rewards["left|right"] > rewards["left|third"]


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

    run_file = run_fuzz(
        cases=2,
        seed=81,
        backends=[],
        config=ExperimentConfig(
            method_arm="contract_lattice_shared_cost_full",
            log_level="minimal",
            enable_feedback=False,
            enable_champion_corpus=False,
        ),
    )

    first, second = read_jsonl(run_file)
    assert first["is_new_behavior"] is True
    assert second["is_new_behavior"] is True
    assert first["signal_signature"] == second["signal_signature"]
    assert first["signal_new_behavior"] is True
    assert second["signal_new_behavior"] is False


def test_run_loaded_case_honors_metamorphic_variant_limit():
    case = generate_case(61, profile="discovery")
    config = ExperimentConfig(enable_metamorphic_oracle=True, metamorphic_variant_limit=2)

    row = run_loaded_case(case, [], config=config, save_artifact=False)

    assert len(row["metamorphic"]) <= 2
    assert row["config"]["metamorphic_variant_limit"] == 2


def test_run_loaded_case_prioritizes_configured_metamorphic_relation_order():
    case = generate_case(7)
    case.program = Program(case.program.program_id, case.program.seed, [{"op": "select", "columns": ["id"]}])
    config = ExperimentConfig(enable_metamorphic_oracle=True, metamorphic_variant_limit=1)

    row = run_loaded_case(
        case,
        [],
        config=config,
        save_artifact=False,
        metamorphic_relation_order=["row_permutation"],
    )

    assert row["metamorphic"]["row_permutation:reverse"]["relation"] == "row_permutation"
    assert row["metamorphic_selection"]["relation_order"] == ["row_permutation"]
