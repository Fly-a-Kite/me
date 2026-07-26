from types import SimpleNamespace

import pytest

from datadiff import runner as runner_module
from datadiff import run_loaded as run_loaded_module
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
from datadiff.semantic_values import LOSSLESS_VALUE_SCHEMA_VERSION, encode_semantic_value


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


class _ManualClock:
    def __init__(self) -> None:
        self.seconds = 0.0

    def __call__(self) -> float:
        return self.seconds

    def advance_ms(self, milliseconds: float) -> None:
        self.seconds += float(milliseconds) / 1000.0


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
        config=ExperimentConfig(
            method_arm="contract_lattice_shared_cost_full",
            enable_parallel_backend_execution=False,
        ),
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
    assert row["config"]["custom"] is True
    assert row["config"]["config_digest"].startswith("config-")
    assert row["method_arm"]["arm_id"] == "contract_lattice_shared_cost_full"
    assert row["method_arm"]["settings"]["ir_mode"] == "ccs_ir"
    assert row["experiment_manifest"]["method_arm_digest"] == row["method_arm"]["digest"]
    assert row["experiment_manifest"]["case_digest"].startswith("case-")
    assert row["experiment_manifest"]["ir_mode"] == "ccs_ir"
    assert row["experiment_manifest"]["program_ir_digest"] == row["ccs_ir_digest"]
    assert row["program_ir"]["digest"] == row["ccs_ir_digest"]
    assert row["ccs_ir"]["digest"] == row["ccs_ir_digest"]
    assert row["environment"] == {"python": "test"}
    assert row["targets"] == [{"name": "left"}, {"name": "right"}]
    assert row["normalized"]["left"]["rows"] == [[1]]
    assert row["osc_diagnostic_refs"]["evaluation_status"] == "not_evaluated"
    assert row["osc_diagnostic_refs"]["authority_scope"] == "diagnostic_only"
    assert row["osc_diagnostic_refs"]["authority_eligible"] is False
    assert row["osc_diagnostic_refs"]["case_digest"] == row["experiment_manifest"][
        "case_digest"
    ]
    assert row["metamorphic"] == {}
    assert row["witness_oracle"]["contract_present"] is False
    assert row["witness_oracle"]["enabled"] is False
    assert row["candidate_recheck"]["enabled"] is False
    assert row["behavior_signature"]
    assert row["discovery_signature"]
    assert row["disagreement_descriptor"]["pair_count"] == 0
    assert row["case"]["metadata"]["case_fingerprint"] == row["case_fingerprint"]
    assert row["semantic_contract_lattice"]["schema_version"] == "semantic-contract-lattice-v1"
    assert row["case"]["metadata"]["semantic_contract_lattice"] == row["semantic_contract_lattice"]
    assert row["semantic_comparison_profile"]["schema_version"] == "contract-comparison-v1"
    assert row["semantic_comparison_profile"]["view"] == "bag_value"
    assert row["case"]["metadata"]["semantic_comparison_profile"] == row["semantic_comparison_profile"]
    assert case.metadata["disagreement_descriptor"] == row["disagreement_descriptor"]
    assert case.metadata["semantic_contract_lattice"] == row["semantic_contract_lattice"]
    assert row["execution_profile"]["parallel_backend_execution"] is False


def test_screening_skips_prefix_localization_for_expected_semantic_divergence(
    monkeypatch,
):
    expected = _countable_finding(
        triage_verdict="expected_semantic_divergence",
        paper_status="valid_finding_not_bug",
        adjudication=build_adjudication(
            "expected_semantic_divergence",
            semantic_gate="expected_boundary",
            attribution_gate="semantic_boundary",
        ),
    )

    def unexpected_localization(*args, **kwargs):
        raise AssertionError("screening must not localize non-countable findings")

    monkeypatch.setattr(
        run_loaded_module,
        "localize_first_divergence",
        unexpected_localization,
    )
    row = run_loaded_case_impl(
        _case(11),
        ["left", "right"],
        config=ExperimentConfig(method_arm="p8_candidate_v1"),
        save_artifact=False,
        target_specs=[],
        execute_case_fn=lambda *args, **kwargs: (_raw(), _normalized(1, 2)),
        evaluate_case_fn=lambda *args, **kwargs: [expected],
        annotate_findings_fn=lambda *args, **kwargs: None,
    )

    assert row["status"] == "ok"
    assert row["findings"][0]["paper_status"] == "valid_finding_not_bug"
    assert row["localization"]["attempted"] is False
    assert row["localization"]["skip_reason"] == (
        "no_countable_candidate_findings_at_screening_tier"
    )
    assert row["execution_profile"]["backend_calls"] == 2


def test_screening_skips_recheck_and_localization_for_known_saturated_finding(
    monkeypatch,
):
    saturated = _countable_finding(root_cause="filter_predicate")

    def unexpected_recheck(*args, **kwargs):
        raise AssertionError("known saturated screening findings must not recheck")

    def unexpected_localization(*args, **kwargs):
        raise AssertionError("known saturated screening findings must not localize")

    monkeypatch.setattr(
        run_loaded_module,
        "localize_first_divergence",
        unexpected_localization,
    )
    row = run_loaded_case_impl(
        _case(12),
        ["left", "right"],
        config=ExperimentConfig(
            method_arm="p8_candidate_v1",
            known_saturated_bug_families=["filter_predicate@right"],
        ),
        save_artifact=False,
        target_specs=[],
        execute_case_fn=lambda *args, **kwargs: (_raw(), _normalized(1, 2)),
        evaluate_case_fn=lambda *args, **kwargs: [saturated],
        annotate_findings_fn=lambda *args, **kwargs: None,
        candidate_recheck_fn=unexpected_recheck,
    )

    assert row["status"] == "bug"
    assert row["candidate_recheck"]["enabled"] is False
    assert row["candidate_recheck"]["skip_reason"] == (
        "known_saturated_findings_do_not_require_screening_recheck"
    )
    assert row["localization"]["attempted"] is False
    assert row["localization"]["skip_reason"] == (
        "known_saturated_findings_do_not_require_screening_localization"
    )
    assert row["execution_profile"]["backend_calls"] == 2


def test_run_loaded_case_impl_executes_selected_metamorphic_variants_and_cross_validates():
    base_case = _case(20, case_id="case-base")
    variant_a = _case(21, case_id="case-a")
    variant_b = _case(22, case_id="case-b")
    calls = {"execute": [], "relation_order": None, "recheck_relation_order": None, "variant_names": None}

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

    def fake_recheck(case_arg, backends_arg, config_arg, findings_arg):
        calls["recheck_relation_order"] = list(config_arg.metamorphic_relation_order)
        return {"enabled": True, "attempts": 1}

    row = run_loaded_case_impl(
        base_case,
        ["left", "right"],
        config=ExperimentConfig(
            method_arm="contract_lattice_shared_cost_full",
            enable_metamorphic_oracle=True,
            metamorphic_variant_limit=1,
        ),
        save_artifact=False,
        target_specs=[],
        metamorphic_relation_order=["target"],
        execute_case_fn=fake_execute,
        all_metamorphic_variants_fn=lambda case_arg: variants,
        select_metamorphic_variants_fn=fake_select,
        evaluate_case_fn=fake_evaluate_case,
        evaluate_metamorphic_variants_fn=fake_evaluate_mr,
        annotate_findings_fn=lambda *args, **kwargs: None,
        candidate_recheck_fn=fake_recheck,
    )

    assert calls["execute"] == ["case-base", "case-b"]
    assert calls["relation_order"] == ["target"]
    assert calls["recheck_relation_order"] == ["target"]
    assert calls["variant_names"] == ["target:b"]
    assert list(row["metamorphic"]) == ["target:b"]
    variant_refs = row["metamorphic"]["target:b"]["osc_diagnostic_refs"]
    assert variant_refs["authority_scope"] == "diagnostic_only"
    assert variant_refs["authority_eligible"] is False
    assert variant_refs["case_digest"] == row["metamorphic"]["target:b"][
        "experiment_manifest"
    ]["case_digest"]
    assert variant_refs["case_digest"] != row["osc_diagnostic_refs"]["case_digest"]
    assert row["metamorphic_selection"]["relation_order"] == ["target"]
    assert row["metamorphic_selection"]["applicable_relations"] == ["slow", "target"]
    assert row["metamorphic_selection"]["executed_relations"] == ["target"]
    assert row["metamorphic_selection"]["omitted_relations"] == ["slow"]
    assert row["metamorphic_selection"]["applicable_variant_count"] == 2
    assert row["metamorphic_selection"]["executed_variant_count"] == 1
    assert row["metamorphic_selection"]["omitted_variant_count"] == 1
    assert row["metamorphic_selection"]["executed_variants"] == ["target:b"]
    assert row["metamorphic_selection"]["omitted_variants"] == ["slow:a"]
    assert row["oracle_cross_validation"]["cross_validated_count"] == 1
    assert row["oracle_cross_validation"]["metamorphic_only_count"] == 0
    assert row["status"] == "bug"
    assert row["execution_profile"]["base_backend_reported_total_ms"] == 2.0
    assert row["execution_profile"]["metamorphic_backend_reported_total_ms"] == 2.0
    assert row["execution_profile"]["candidate_recheck_backend_reported_total_ms"] == 0.0
    assert row["execution_profile"]["backend_reported_total_ms"] == 4.0
    assert row["execution_profile"]["base_backend_calls"] == 2
    assert row["execution_profile"]["metamorphic_backend_calls"] == 2
    assert row["execution_profile"]["candidate_recheck_backend_calls"] == 0
    assert row["execution_profile"]["backend_calls"] == 4


def test_run_loaded_case_impl_uses_executable_ccs_obligation_registry():
    case = Case(
        "case-ccs-obligation-runtime",
        23,
        [
            TableData(
                "t0",
                [ColumnSpec("id", "int", nullable=False)],
                [{"id": 1}, {"id": 2}],
            )
        ],
        Program(
            "prog-ccs-obligation-runtime",
            23,
            [{"op": "filter", "column": "id", "cmp": ">=", "value": 1}],
        ),
    )

    row = run_loaded_case_impl(
        case,
        ["left", "right"],
        config=ExperimentConfig(
            method_arm="contract_ccs_obligations_cartesian",
            enable_metamorphic_oracle=True,
            metamorphic_variant_limit=2,
        ),
        save_artifact=False,
        target_specs=[],
        execute_case_fn=lambda *args, **kwargs: (_raw(), _normalized(1, 1)),
        evaluate_case_fn=lambda *args, **kwargs: [],
        evaluate_metamorphic_variants_fn=lambda *args, **kwargs: [],
    )

    selection = row["test_obligation_selection"]
    assert selection["registry_applied"] is True
    assert selection["mode"] == "ccs_guided"
    assert selection["relations"][:2] == [
        "input_partition_union_all",
        "filter_rejecting_row_injection",
    ]
    assert selection["program_obligation_count"] == 6
    assert not any(
        obligation["obligation_id"] == "filter_input_materialization"
        for obligation in selection["obligations"]
    )
    assert row["metamorphic_selection"]["executed_relations"] == [
        "input_partition_union_all",
        "filter_rejecting_row_injection",
    ]
    assert row["metamorphic_selection"]["obligation_mode"] == "ccs_guided"
    assert row["execution_profile"]["obligation_mode"] == "ccs_guided"


def test_run_loaded_case_impl_accounts_for_candidate_recheck_backend_work():
    row = run_loaded_case_impl(
        _case(23),
        ["left", "right"],
        config=ExperimentConfig(method_arm="contract_lattice_shared_cost_full"),
        save_artifact=False,
        target_specs=[],
        execute_case_fn=lambda *args, **kwargs: (_raw(2.0), _normalized(1, 2)),
        evaluate_case_fn=lambda case_arg, normalized: [_countable_finding()],
        annotate_findings_fn=lambda *args, **kwargs: None,
        candidate_recheck_fn=lambda *args, **kwargs: {
            "enabled": True,
            "attempts": 2,
            "backend_reported_total_ms": 12.0,
            "backend_calls": 8,
        },
    )

    assert row["execution_profile"]["base_backend_reported_total_ms"] == 4.0
    assert row["execution_profile"]["metamorphic_backend_reported_total_ms"] == 0.0
    assert row["execution_profile"]["candidate_recheck_backend_reported_total_ms"] == 12.0
    assert row["execution_profile"]["backend_reported_total_ms"] == 16.0
    assert row["execution_profile"]["backend_calls"] == 10


def test_run_loaded_case_impl_batches_metamorphic_variant_execution_when_available():
    base_case = _case(24, case_id="case-batch-base")
    variants = [
        SimpleNamespace(name="mr:a", relation="a", case=_case(25, case_id="case-batch-a")),
        SimpleNamespace(name="mr:b", relation="b", case=_case(26, case_id="case-batch-b")),
    ]
    calls = {"single": [], "batch": []}

    def fake_execute(case_arg, backends_arg, config_arg, **kwargs):
        calls["single"].append(case_arg.case_id)
        return _raw(), _normalized()

    def fake_execute_batch(cases_arg, backends_arg, config_arg, **kwargs):
        calls["batch"].append([case.case_id for case in cases_arg])
        return [(_raw(), _normalized()) for _case_arg in cases_arg]

    row = run_loaded_case_impl(
        base_case,
        ["left", "right"],
        config=ExperimentConfig(enable_metamorphic_oracle=True, metamorphic_variant_limit=2),
        save_artifact=False,
        target_specs=[],
        execute_case_fn=fake_execute,
        execute_cases_fn=fake_execute_batch,
        all_metamorphic_variants_fn=lambda case_arg: variants,
        select_metamorphic_variants_fn=lambda rows, **kwargs: list(rows),
        evaluate_case_fn=lambda case_arg, normalized: [],
        evaluate_metamorphic_variants_fn=lambda *args, **kwargs: [],
    )

    assert calls["single"] == ["case-batch-base"]
    assert calls["batch"] == [["case-batch-a", "case-batch-b"]]
    assert list(row["metamorphic"]) == ["mr:a", "mr:b"]
    assert row["execution_profile"]["base_backend_reported_total_ms"] == 2.0
    assert row["execution_profile"]["metamorphic_backend_reported_total_ms"] == 4.0
    assert row["execution_profile"]["backend_reported_total_ms"] == 6.0
    assert row["execution_profile"]["backend_calls"] == 6


def test_run_loaded_case_impl_separates_backend_and_oracle_timing():
    clock = _ManualClock()
    base_case = _case(27, case_id="case-timing-base")
    variant_case = _case(28, case_id="case-timing-variant")
    variants = [SimpleNamespace(name="mr:timing", relation="timing", case=variant_case)]

    def fake_execute(case_arg, backends_arg, config_arg, **kwargs):
        clock.advance_ms(10.0 if case_arg.case_id == base_case.case_id else 20.0)
        return _raw(3.0), _normalized(1, 2)

    def fake_evaluate_case(case_arg, normalized):
        clock.advance_ms(2.0)
        return [_countable_finding()]

    def fake_all_variants(case_arg):
        clock.advance_ms(1.0)
        return variants

    def fake_select(rows, **kwargs):
        clock.advance_ms(1.0)
        return list(rows)

    def fake_evaluate_mr(*args, **kwargs):
        clock.advance_ms(3.0)
        return []

    def fake_annotate(*args, **kwargs):
        clock.advance_ms(4.0)

    def fake_recheck(*args, **kwargs):
        clock.advance_ms(30.0)
        return {
            "enabled": True,
            "attempts": 1,
            "backend_reported_total_ms": 12.0,
            "backend_calls": 4,
        }

    row = run_loaded_case_impl(
        base_case,
        ["left", "right"],
        config=ExperimentConfig(
            method_arm="contract_lattice_shared_cost_full",
            enable_metamorphic_oracle=True,
            metamorphic_variant_limit=1,
        ),
        save_artifact=False,
        environment={},
        target_specs=[],
        execute_case_fn=fake_execute,
        evaluate_case_fn=fake_evaluate_case,
        all_metamorphic_variants_fn=fake_all_variants,
        select_metamorphic_variants_fn=fake_select,
        evaluate_metamorphic_variants_fn=fake_evaluate_mr,
        annotate_findings_fn=fake_annotate,
        candidate_recheck_fn=fake_recheck,
        perf_counter_fn=clock,
    )

    assert row["execution_profile"]["timing_schema_version"] == "execution-timing-v3"
    assert row["execution_profile"]["base_backend_wall_ms"] == pytest.approx(10.0)
    assert row["execution_profile"]["metamorphic_backend_wall_ms"] == pytest.approx(20.0)
    assert row["execution_profile"]["candidate_recheck_wall_ms"] == pytest.approx(30.0)
    assert row["execution_profile"]["candidate_recheck_timing_source"] == "callback_wall_fallback"
    assert row["execution_profile"]["differential_oracle_ms"] == pytest.approx(2.0)
    assert row["execution_profile"]["metamorphic_construction_ms"] == pytest.approx(2.0)
    assert row["execution_profile"]["metamorphic_oracle_ms"] == pytest.approx(3.0)
    assert row["execution_profile"]["classification_annotation_ms"] == pytest.approx(4.0)
    assert row["stage_profile"]["backend_execution_ms"] == pytest.approx(42.0)
    assert row["stage_profile"]["normalize_ms"] == pytest.approx(18.0)
    assert row["stage_profile"]["oracle_classification_ms"] == pytest.approx(11.0)
    assert row["stage_profile"]["total_case_wall_ms"] == pytest.approx(71.0)
    assert row["execution_profile"]["observed_case_wall_ms"] == pytest.approx(71.0)
    assert row["execution_profile"]["accounted_case_wall_ms"] == pytest.approx(71.0)
    assert row["execution_profile"]["accounting_overrun_ms"] == pytest.approx(0.0)
    assert row["wall_time_profile"]["execution_pipeline_ms"] == pytest.approx(60.0)
    assert row["wall_time_profile"]["oracle_classification_ms"] == pytest.approx(11.0)
    assert row["wall_time_profile"]["total_wall_ms"] == pytest.approx(71.0)


def test_run_loaded_case_impl_injected_clocks_separate_wall_backend_and_process_cpu():
    wall_clock = _ManualClock()
    process_cpu_clock = _ManualClock()
    case = _case(29, case_id="case-injected-clocks")

    def fake_execute(*args, **kwargs):
        wall_clock.advance_ms(10.0)
        process_cpu_clock.advance_ms(6.0)
        return _raw(3.0), _normalized(1, 1)

    def fake_evaluate(*args, **kwargs):
        wall_clock.advance_ms(2.0)
        process_cpu_clock.advance_ms(1.0)
        return []

    row = run_loaded_case_impl(
        case,
        ["left", "right"],
        config=ExperimentConfig(enable_metamorphic_oracle=False),
        save_artifact=False,
        environment={},
        target_specs=[],
        execute_case_fn=fake_execute,
        evaluate_case_fn=fake_evaluate,
        perf_counter_fn=wall_clock,
        process_cpu_fn=process_cpu_clock,
    )

    assert row["execution_profile"]["backend_reported_total_ms"] == pytest.approx(6.0)
    assert row["wall_time_profile"] == pytest.approx(
        {
            "generate_mutate_ms": 0.0,
            "execution_pipeline_ms": 10.0,
            "oracle_classification_ms": 2.0,
            "scheduler_feedback_ms": 0.0,
            "logging_artifact_ms": 0.0,
            "total_wall_ms": 12.0,
        }
    )
    assert row["process_cpu_profile"] == pytest.approx(
        {
            "generate_mutate_ms": 0.0,
            "execution_pipeline_ms": 6.0,
            "oracle_classification_ms": 1.0,
            "scheduler_feedback_ms": 0.0,
            "logging_artifact_ms": 0.0,
            "total_process_cpu_ms": 7.0,
        }
    )
    assert row["execution_profile"]["observed_case_wall_ms"] == pytest.approx(12.0)
    assert row["execution_profile"]["observed_case_process_cpu_ms"] == pytest.approx(7.0)


def test_run_loaded_case_impl_records_complete_key_reuse_provenance():
    raw = _raw(2.0)
    raw["left"].update(
        {
            "execution_cache_hit": True,
            "execution_cache_lookup": "hit",
            "execution_cache_key": "exec-left",
            "execution_cache_entry_bytes": 123,
            "execution_cache_provenance": {"environment_digest": "env-a"},
        }
    )
    raw["right"].update(
        {
            "execution_cache_hit": False,
            "execution_cache_lookup": "miss",
            "execution_cache_key": "exec-right",
            "execution_cache_entry_bytes": 100,
            "execution_cache_provenance": {"environment_digest": "env-a"},
        }
    )

    row = run_loaded_case_impl(
        _case(30),
        ["left", "right"],
        config=ExperimentConfig(enable_metamorphic_oracle=False),
        save_artifact=False,
        environment={},
        target_specs=[],
        execute_case_fn=lambda *args, **kwargs: (raw, _normalized(1, 1)),
        evaluate_case_fn=lambda *args, **kwargs: [],
    )

    reuse = row["execution_reuse"]
    assert reuse["summary"] == {
        "event_count": 2,
        "cache_hit_count": 1,
        "cache_miss_count": 1,
        "not_applicable_count": 0,
        "reused_entry_bytes": 123,
    }
    assert reuse["events"][0]["reuse_reason"] == "complete_key_match"
    assert reuse["events"][1]["reuse_reason"] == "no_matching_complete_key"


def test_run_loaded_case_impl_records_witness_summary_without_counting_when_disabled():
    case = _case(50)
    case.metadata["witness_contract"] = {
        "kind": "row_containment",
        "row": {"x": 50},
    }

    row = run_loaded_case_impl(
        case,
        ["left", "right"],
        config=ExperimentConfig(enable_witness_oracle=False),
        save_artifact=False,
        target_specs=[],
        execute_case_fn=lambda *args, **kwargs: (_raw(), _normalized(50, 51)),
        evaluate_case_fn=lambda case_arg, normalized: [],
    )

    assert row["status"] == "ok"
    assert row["findings"] == []
    assert row["witness_oracle"]["contract_present"] is True
    assert row["witness_oracle"]["enabled"] is False
    assert row["witness_oracle"]["failing_backends"] == ["right"]


def test_run_loaded_case_impl_counts_witness_finding_when_enabled():
    case = _case(60)
    case.metadata["witness_contract"] = {
        "kind": "row_containment",
        "row": {"x": 60},
        "reason": "PQS-style pivot row should remain present",
    }

    row = run_loaded_case_impl(
        case,
        ["left", "right"],
        config=ExperimentConfig(enable_witness_oracle=True),
        save_artifact=False,
        target_specs=[],
        execute_case_fn=lambda *args, **kwargs: (_raw(), _normalized(60, 61)),
        evaluate_case_fn=lambda case_arg, normalized: [],
        annotate_findings_fn=lambda *args, **kwargs: None,
    )

    assert row["status"] == "bug"
    assert row["witness_oracle"]["enabled"] is True
    assert row["witness_oracle"]["failing_backends"] == ["right"]
    assert [finding["oracle"] for finding in row["findings"]] == ["witness"]
    assert row["findings"][0]["kind"] == "witness_row_containment_violation"
    assert row["findings"][0]["suspicious_backends"] == ["right"]
    assert row["findings"][0]["triage_verdict"] == "candidate_implementation_bug"
    assert row["findings"][0]["adjudication"]["semantic_gate"] == "witness_contract"


def test_run_loaded_case_impl_can_infer_witness_contract_when_enabled():
    case = Case(
        "case-infer-witness",
        70,
        [TableData("t0", [ColumnSpec("x", "int")], [{"x": 70}])],
        Program("prog-infer-witness", 70, [{"op": "filter", "column": "x", "cmp": "==", "value": 70}]),
    )

    row = run_loaded_case_impl(
        case,
        ["left", "right"],
        config=ExperimentConfig(enable_witness_oracle=True),
        save_artifact=False,
        target_specs=[],
        execute_case_fn=lambda *args, **kwargs: (_raw(), _normalized(70, 71)),
        evaluate_case_fn=lambda case_arg, normalized: [],
        annotate_findings_fn=lambda *args, **kwargs: None,
    )

    assert row["witness_oracle"]["contract"]["source"] == "inferred_row_preserving_pipeline"
    assert row["witness_oracle"]["failing_backends"] == ["right"]
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
            "ccs_ir": kwargs["ccs_ir"],
        }
        return "/tmp/datadiff-bug"

    row = run_loaded_case_impl(
        case,
        ["left", "right"],
        config=ExperimentConfig(
            method_arm="contract_lattice_shared_cost_full",
            enable_artifact=True,
        ),
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
    assert calls["save"]["config"]["artifact"] is True
    assert calls["save"]["ccs_ir"].digest.startswith("ccs-ir-")
    assert calls["save"]["config"]["method_arm_manifest"]["arm_id"] == (
        "contract_lattice_shared_cost_full"
    )
    assert calls["save"]["config"]["experiment_manifest"]["manifest_digest"].startswith(
        "experiment-"
    )


def test_run_loaded_case_impl_switches_legacy_and_contract_oracle_arms():
    case = Case(
        "case-arm-switch",
        31,
        [TableData("t0", [ColumnSpec("x", "float")], [{"x": float("nan")}])],
        Program("prog-arm-switch", 31, [{"op": "select", "columns": ["x"]}]),
    )
    normalized = {
        "left": NormalizedResult(
            "left",
            "ok",
            ["x"],
            [[None]],
            lossless_rows=[[encode_semantic_value(None)]],
            lossless_schema_version=LOSSLESS_VALUE_SCHEMA_VERSION,
        ),
        "right": NormalizedResult(
            "right",
            "ok",
            ["x"],
            [[None]],
            lossless_rows=[[encode_semantic_value(float("nan"))]],
            lossless_schema_version=LOSSLESS_VALUE_SCHEMA_VERSION,
        ),
    }

    def execute(*args, **kwargs):
        return _raw(), normalized

    common = {
        "save_artifact": False,
        "target_specs": [],
        "execute_case_fn": execute,
        "annotate_findings_fn": lambda *args, **kwargs: None,
    }
    legacy = run_loaded_case_impl(
        case,
        ["left", "right"],
        config=ExperimentConfig(method_arm="legacy_cartesian"),
        **common,
    )
    contract = run_loaded_case_impl(
        case,
        ["left", "right"],
        config=ExperimentConfig(method_arm="contract_cartesian"),
        **common,
    )

    assert legacy["findings"] == []
    assert legacy["execution_profile"]["comparison_mode"] == "legacy"
    assert contract["findings"][0]["mismatch_class"] == "value"
    assert contract["execution_profile"]["comparison_mode"] == "contract"


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
