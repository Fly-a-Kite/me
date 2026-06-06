from types import SimpleNamespace

from datadiff import runner as runner_module
from datadiff.adjudication import build_adjudication
from datadiff.config import ExperimentConfig
from datadiff.dsl import Case, ColumnSpec, Program, TableData
from datadiff.normalizer import NormalizedResult
from datadiff.oracle import Finding
from datadiff.run_loaded import (
    execute_case_for_run_loaded,
    parallel_backend_execution_active,
    run_loaded_case_impl,
)


def _case(seed: int = 1, *, case_id: str | None = None) -> Case:
    return Case(
        case_id or f"case-{seed}",
        seed,
        [TableData("t0", [ColumnSpec("x", "int")], [{"x": seed}])],
        Program(f"prog-{seed}", seed, [{"op": "select", "columns": ["x"]}]),
    )


def _normalized(left_value: int = 1, right_value: int = 1) -> dict[str, NormalizedResult]:
    return {
        "left": NormalizedResult("left", "ok", ["x"], [[left_value]]),
        "right": NormalizedResult("right", "ok", ["x"], [[right_value]]),
    }


def _raw(duration_ms: float = 1.0) -> dict[str, dict]:
    return {
        "left": {"status": "ok", "duration_ms": duration_ms},
        "right": {"status": "ok", "duration_ms": duration_ms},
    }


def _countable_finding(**overrides) -> Finding:
    payload = {
        "finding_id": "finding-1",
        "kind": "semantic_output_mismatch",
        "severity": "critical",
        "suspicious_backends": ["right"],
        "evidence": "mismatch",
        "signature": "sig-1",
        "root_cause": "filter_predicate",
        "mismatch_class": "value",
        "triage_verdict": "candidate_implementation_bug",
        "paper_status": "candidate_bug_needs_external_confirmation",
        "triage_confidence": "high",
        "adjudication": build_adjudication("candidate_implementation_bug"),
    }
    payload.update(overrides)
    return Finding(**payload)


def test_parallel_backend_execution_active_requires_distinct_parallel_backends():
    enabled = ExperimentConfig(enable_parallel_backend_execution=True)
    disabled = ExperimentConfig(enable_parallel_backend_execution=False)

    assert parallel_backend_execution_active(enabled, ["left", "right"]) is True
    assert parallel_backend_execution_active(enabled, ["left", "left"]) is False
    assert parallel_backend_execution_active(enabled, ["left"]) is False
    assert parallel_backend_execution_active(disabled, ["left", "right"]) is False


def test_run_loaded_case_impl_emits_row_shape_metadata_and_signatures():
    case = _case(10)
    calls: dict[str, object] = {}

    def fake_execute(case_arg, backends_arg, config_arg, **kwargs):
        calls["execute"] = {
            "case_id": case_arg.case_id,
            "backends": list(backends_arg),
            "backend_instances": kwargs.get("backend_instances"),
        }
        return _raw(2.0), _normalized(1, 1)

    row = run_loaded_case_impl(
        case,
        ["left", "right"],
        config=ExperimentConfig(enable_parallel_backend_execution=False),
        save_artifact=False,
        backend_instances={"left": object()},
        environment={"python": "test"},
        target_specs=[{"name": "left"}, {"name": "right"}],
        config_payload={"custom": True},
        execute_case_fn=fake_execute,
        parallel_backend_execution_active_fn=lambda config, backends: False,
        evaluate_case_fn=lambda case_arg, normalized: [],
        collect_environment_fn=lambda: {"python": "unused"},
        describe_targets_fn=lambda backends: [{"name": "unused"}],
    )

    assert calls["execute"]["case_id"] == "case-10"
    assert calls["execute"]["backends"] == ["left", "right"]
    assert row["status"] == "ok"
    assert row["config"] == {"custom": True}
    assert row["environment"] == {"python": "test"}
    assert row["targets"] == [{"name": "left"}, {"name": "right"}]
    assert row["normalized"]["left"]["rows"] == [[1]]
    assert row["metamorphic"] == {}
    assert row["candidate_recheck"]["enabled"] is False
    assert row["behavior_signature"]
    assert row["discovery_signature"]
    assert row["disagreement_descriptor"]["pair_count"] == 0
    assert row["case"]["metadata"]["case_fingerprint"] == row["case_fingerprint"]
    assert case.metadata["disagreement_descriptor"] == row["disagreement_descriptor"]
    assert row["execution_profile"]["parallel_backend_execution"] is False


def test_run_loaded_case_impl_executes_selected_metamorphic_variants_and_cross_validates():
    base_case = _case(20, case_id="case-base")
    variant_a = _case(21, case_id="case-a")
    variant_b = _case(22, case_id="case-b")
    calls = {"execute": [], "relation_order": None, "variant_names": None}

    variants = [
        SimpleNamespace(name="slow:a", relation="slow", case=variant_a),
        SimpleNamespace(name="target:b", relation="target", case=variant_b),
    ]

    def fake_select(variant_rows, *, limit, relation_order):
        calls["relation_order"] = list(relation_order)
        return [variant for variant in variant_rows if variant.relation == "target"][:limit]

    def fake_execute(case_arg, backends_arg, config_arg, **kwargs):
        calls["execute"].append(case_arg.case_id)
        return _raw(), _normalized(1, 2)

    def fake_evaluate_case(case_arg, normalized):
        return [
            _countable_finding(
                finding_id="finding-diff",
                root_cause="target",
                oracle="differential",
            )
        ]

    def fake_evaluate_mr(case_arg, normalized, variant_results):
        calls["variant_names"] = sorted(variant_results)
        return [
            _countable_finding(
                finding_id="finding-mr",
                kind="metamorphic_target_violation",
                root_cause="metamorphic_target",
                oracle="metamorphic",
            )
        ]

    row = run_loaded_case_impl(
        base_case,
        ["left", "right"],
        config=ExperimentConfig(enable_metamorphic_oracle=True, metamorphic_variant_limit=1),
        save_artifact=False,
        target_specs=[],
        metamorphic_relation_order=["target"],
        execute_case_fn=fake_execute,
        all_metamorphic_variants_fn=lambda case_arg: variants,
        select_metamorphic_variants_fn=fake_select,
        evaluate_case_fn=fake_evaluate_case,
        evaluate_metamorphic_variants_fn=fake_evaluate_mr,
        annotate_findings_fn=lambda *args, **kwargs: None,
        candidate_recheck_fn=lambda *args, **kwargs: {"enabled": True, "attempts": 1},
    )

    assert calls["execute"] == ["case-base", "case-b"]
    assert calls["relation_order"] == ["target"]
    assert calls["variant_names"] == ["target:b"]
    assert list(row["metamorphic"]) == ["target:b"]
    assert row["metamorphic_selection"]["relation_order"] == ["target"]
    assert row["metamorphic_selection"]["executed_relations"] == ["target"]
    assert row["oracle_cross_validation"]["cross_validated_count"] == 1
    assert row["oracle_cross_validation"]["metamorphic_only_count"] == 0
    assert row["status"] == "bug"


def test_run_loaded_case_impl_saves_artifact_for_countable_findings():
    case = _case(30)
    calls: dict[str, object] = {}

    def fake_save(case_arg, **kwargs):
        calls["save"] = {
            "case_id": case_arg.case_id,
            "findings": [finding.finding_id for finding in kwargs["findings"]],
            "normalized": kwargs["normalized"],
            "config": kwargs["config"],
        }
        return "/tmp/datadiff-bug"

    row = run_loaded_case_impl(
        case,
        ["left", "right"],
        config=ExperimentConfig(enable_artifact=True),
        save_artifact=True,
        target_specs=[],
        config_payload={"artifact": True},
        execute_case_fn=lambda *args, **kwargs: (_raw(), _normalized(1, 2)),
        evaluate_case_fn=lambda case_arg, normalized: [_countable_finding()],
        annotate_findings_fn=lambda *args, **kwargs: None,
        save_bug_artifact_fn=fake_save,
    )

    assert row["status"] == "bug"
    assert row["bug_dir"] == "/tmp/datadiff-bug"
    assert row["stage_profile"]["logging_artifact_ms"] >= 0.0
    assert calls["save"]["case_id"] == "case-30"
    assert calls["save"]["findings"] == ["finding-1"]
    assert calls["save"]["config"] == {"artifact": True}


def test_runner_run_loaded_case_wrapper_keeps_monkeypatch_compatibility(monkeypatch):
    case = _case(40)
    calls: dict[str, object] = {}

    def fake_execute(case_arg, backends_arg, config_arg, **kwargs):
        calls["execute"] = {
            "case_id": case_arg.case_id,
            "backends": list(backends_arg),
        }
        return _raw(), _normalized(1, 1)

    monkeypatch.setattr(runner_module, "_execute_case", fake_execute)
    monkeypatch.setattr(runner_module, "evaluate_case", lambda *args, **kwargs: [])

    row = runner_module.run_loaded_case(case, ["left", "right"], save_artifact=False, target_specs=[])

    assert calls["execute"] == {"case_id": "case-40", "backends": ["left", "right"]}
    assert row["status"] == "ok"
    assert runner_module.run_loaded_case_impl is run_loaded_case_impl
    assert runner_module.execute_case_for_run_loaded is execute_case_for_run_loaded
