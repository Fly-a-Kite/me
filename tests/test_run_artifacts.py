from datadiff import runner as runner_module
from datadiff.adjudication import build_adjudication
from datadiff.config import ExperimentConfig
from datadiff.dsl import Case, ColumnSpec, Program, TableData
from datadiff.run_artifacts import process_reducer_and_artifacts


def _case(seed: int = 1, *, rows: int = 2, ops: int = 2) -> Case:
    operations = [{"op": "select", "columns": ["x"]}]
    if ops > 1:
        operations.append({"op": "filter", "column": "x", "cmp": ">=", "value": 0})
    return Case(
        f"case-{seed}",
        seed,
        [TableData("t0", [ColumnSpec("x", "int")], [{"x": idx} for idx in range(rows)])],
        Program(f"prog-{seed}", seed, operations),
    )


def _countable_finding(root: str = "filter_predicate") -> dict:
    return {
        "kind": "semantic_output_mismatch",
        "root_cause": root,
        "triage_verdict": "candidate_implementation_bug",
        "suspicious_backends": ["duckdb"],
        "adjudication": build_adjudication("candidate_implementation_bug"),
    }


def _row(*, findings: list[dict] | None = None, bug_dir: str = "") -> dict:
    row = {
        "stage_profile": {"backend_execution_ms": 1.0},
        "findings": list(findings or []),
    }
    if bug_dir:
        row["bug_dir"] = bug_dir
    return row


def test_process_reducer_and_artifacts_preserves_non_finding_rows():
    case = _case()
    row = _row()

    result = process_reducer_and_artifacts(
        row=row,
        case=case,
        backends=["pandas"],
        config=ExperimentConfig(enable_artifact=True),
        effective_config=ExperimentConfig(),
        backend_instances=None,
        environment={},
        target_specs=[],
        effective_config_payload={},
        saved_artifacts_count=0,
        generate_mutate_elapsed_ms=2.5,
        run_loaded_case_fn=lambda *args, **kwargs: {},
    )

    assert result.row is row
    assert result.countable_row_findings == []
    assert result.saved_artifact_delta == 0
    assert result.scheduler_elapsed_ms == 0.0
    assert result.row_stage_profile["generate_mutate_ms"] == 2.5
    assert "artifact_saved" not in row


def test_process_reducer_and_artifacts_marks_saved_and_skipped_artifacts():
    case = _case()
    finding = _countable_finding()

    saved = process_reducer_and_artifacts(
        row=_row(findings=[finding], bug_dir="/tmp/bug"),
        case=case,
        backends=["pandas"],
        config=ExperimentConfig(enable_artifact=True),
        effective_config=ExperimentConfig(),
        backend_instances=None,
        environment={},
        target_specs=[],
        effective_config_payload={},
        saved_artifacts_count=0,
        generate_mutate_elapsed_ms=0.0,
        run_loaded_case_fn=lambda *args, **kwargs: {},
    )
    skipped = process_reducer_and_artifacts(
        row=_row(findings=[finding]),
        case=case,
        backends=["pandas"],
        config=ExperimentConfig(enable_artifact=True),
        effective_config=ExperimentConfig(),
        backend_instances=None,
        environment={},
        target_specs=[],
        effective_config_payload={},
        saved_artifacts_count=1,
        generate_mutate_elapsed_ms=0.0,
        run_loaded_case_fn=lambda *args, **kwargs: {},
    )

    assert saved.saved_artifact_delta == 1
    assert saved.row["artifact_saved"] is True
    assert skipped.saved_artifact_delta == 0
    assert skipped.row["artifact_saved"] is False
    assert skipped.row["artifact_skipped_reason"] == "artifact_limit_reached"


def test_process_reducer_and_artifacts_runs_reducer_with_minimal_config_and_reexecutes():
    original = _case(3, rows=3, ops=2)
    reduced = _case(3, rows=1, ops=1)
    calls: dict[str, object] = {}

    def fake_reduce_case(case, *, backends, config, target_kinds, target_roots, target_suspicious_backends):
        calls["reduce_case"] = {
            "case": case,
            "backends": list(backends),
            "config": config,
            "target_kinds": list(target_kinds),
            "target_roots": list(target_roots),
            "target_suspicious_backends": [list(item) for item in target_suspicious_backends],
        }
        return reduced

    def fake_run_loaded_case(case, **kwargs):
        calls["run_loaded_case"] = {"case": case, "kwargs": dict(kwargs)}
        return {
            "stage_profile": {"backend_execution_ms": 3.0},
            "execution_profile": {"backend_reported_total_ms": 7.0},
            "findings": [_countable_finding("reduced_root")],
            "bug_dir": "/tmp/reduced-bug",
        }

    original_row = _row(findings=[_countable_finding("original_root")])
    original_row["backend_sampling"] = {"mode": "coverage_sample"}
    original_row["execution_profile"] = {"combined_backend_reported_total_ms": 5.0}
    result = process_reducer_and_artifacts(
        row=original_row,
        case=original,
        backends=["pandas", "duckdb"],
        config=ExperimentConfig(enable_reducer=True, enable_artifact=True, artifact_limit=1),
        effective_config=ExperimentConfig(
            enable_type_aware_generation=False,
            enable_normalizer=False,
            enable_differential_oracle=True,
            enable_metamorphic_oracle=True,
            oracle_mode="both",
            generator_profile="common",
            metamorphic_variant_limit=5,
            target_version="latest",
            fixed_version="fixed",
        ),
        backend_instances={"pandas": object()},
        environment={"python": "test"},
        target_specs=[{"backend": "pandas"}],
        effective_config_payload={"target_version": "latest"},
        saved_artifacts_count=0,
        generate_mutate_elapsed_ms=4.0,
        run_loaded_case_fn=fake_run_loaded_case,
        reduce_case_fn=fake_reduce_case,
    )

    reduce_call = calls["reduce_case"]
    reducer_config = reduce_call["config"]
    assert reduce_call["case"] is original
    assert reduce_call["backends"] == ["pandas", "duckdb"]
    assert reduce_call["target_kinds"] == ["semantic_output_mismatch"]
    assert reduce_call["target_roots"] == ["original_root"]
    assert reduce_call["target_suspicious_backends"] == [["duckdb"]]
    assert reducer_config.enable_feedback is False
    assert reducer_config.enable_reducer is False
    assert reducer_config.enable_artifact is False
    assert reducer_config.enable_metamorphic_oracle is True
    assert reducer_config.target_version == "latest"
    assert reducer_config.fixed_version == "fixed"

    run_call = calls["run_loaded_case"]
    assert run_call["case"] is reduced
    assert run_call["kwargs"]["save_artifact"] is True
    assert run_call["kwargs"]["backend_instances"].keys() == {"pandas"}
    assert run_call["kwargs"]["environment"] == {"python": "test"}
    assert run_call["kwargs"]["target_specs"] == [{"backend": "pandas"}]
    assert run_call["kwargs"]["config_payload"] == {"target_version": "latest"}

    assert result.row["original_case"]["case_id"] == "case-3"
    assert result.row["reduction"] == {
        "original_rows": 3,
        "reduced_rows": 1,
        "original_ops": 2,
        "reduced_ops": 1,
    }
    assert result.row["artifact_saved"] is True
    assert result.saved_artifact_delta == 1
    assert result.row_stage_profile["generate_mutate_ms"] == 4.0
    assert result.row_stage_profile["backend_execution_ms"] == 4.0
    assert result.row_stage_profile["total_case_wall_ms"] == 8.0
    assert result.row["backend_sampling"] == {"mode": "coverage_sample"}
    assert result.row["execution_profile"]["original_candidate_backend_reported_total_ms"] == 5.0
    assert result.row["execution_profile"]["combined_backend_reported_total_ms"] == 12.0
    assert result.row["duration_ms"] == 8.0
    assert result.scheduler_elapsed_ms >= 0.0


def test_runner_reexports_artifact_processing_helper_for_compatibility():
    assert runner_module.process_reducer_and_artifacts is process_reducer_and_artifacts
