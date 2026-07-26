from datadiff.adjudication import build_adjudication
from datadiff.config import ExperimentConfig
from datadiff.dsl import Case, ColumnSpec, Program, TableData
from datadiff.oracle import Finding
from datadiff.run_recheck import candidate_recheck_impl


def _case(seed: int = 1) -> Case:
    return Case(
        f"case-{seed}",
        seed,
        [TableData("t0", [ColumnSpec("x", "int")], [{"x": seed}])],
        Program(f"prog-{seed}", seed, [{"op": "select", "columns": ["x"]}]),
    )


def _finding(root: str = "filter_predicate", *, backend: str = "right") -> Finding:
    return Finding(
        finding_id=f"finding-{root}",
        kind="semantic_output_mismatch",
        severity="critical",
        suspicious_backends=[backend],
        evidence="mismatch",
        signature=f"sig-{root}",
        root_cause=root,
        mismatch_class="row_count",
        triage_verdict="candidate_implementation_bug",
        adjudication=build_adjudication("candidate_implementation_bug"),
    )


def _finding_row(root: str = "filter_predicate", *, backend: str = "right") -> dict:
    return {
        "finding_id": f"finding-{root}",
        "kind": "semantic_output_mismatch",
        "root_cause": root,
        "suspicious_backends": [backend],
        "mismatch_class": "row_count",
    }


def test_candidate_recheck_impl_short_circuits_without_attempts_or_findings():
    calls = {"run_loaded": 0}

    def fake_run_loaded_case(*args, **kwargs):
        calls["run_loaded"] += 1
        return {"findings": []}

    no_attempts = candidate_recheck_impl(
        _case(),
        ["left", "right"],
        ExperimentConfig(candidate_recheck_count=0),
        [_finding()],
        run_loaded_case_fn=fake_run_loaded_case,
    )
    no_findings = candidate_recheck_impl(
        _case(),
        ["left", "right"],
        ExperimentConfig(candidate_recheck_count=2),
        [],
        run_loaded_case_fn=fake_run_loaded_case,
    )

    assert no_attempts == {"enabled": False, "attempts": 0, "reproduced_keys": [], "non_reproduced_keys": []}
    assert no_findings == {"enabled": False, "attempts": 0, "reproduced_keys": [], "non_reproduced_keys": []}
    assert calls["run_loaded"] == 0


def test_candidate_recheck_impl_uses_non_recursive_no_artifact_config_and_intersects_attempts():
    case = _case(2)
    finding = _finding()
    calls: list[dict] = []
    rows = [
        {
            "findings": [_finding_row()],
            "raw_results": {
                "left": {
                    "physical_plan": {
                        "detail": "full",
                        "observations": [{"raw_text": "Scan t"}],
                    }
                }
            },
            "execution_profile": {
                "combined_backend_reported_total_ms": 6.0,
                "combined_backend_calls": 6,
            },
            "stage_profile": {
                "backend_execution_ms": 4.0,
                "oracle_classification_ms": 1.0,
            },
        },
        {
            "findings": [],
            "execution_profile": {
                "backend_reported_total_ms": 4.0,
                "backend_calls": 4,
            },
            "stage_profile": {
                "backend_execution_ms": 3.0,
                "normalize_ms": 1.0,
                "oracle_classification_ms": 2.0,
            },
        },
    ]

    def fake_run_loaded_case(case_arg, **kwargs):
        calls.append(
            {
                "case": case_arg,
                "backends": kwargs["backends"],
                "candidate_recheck_count": kwargs["config"].candidate_recheck_count,
                "evidence_tier": kwargs["config"].evidence_tier,
                "enable_artifact": kwargs["config"].enable_artifact,
                "save_artifact": kwargs["save_artifact"],
                "backend_instances": kwargs["backend_instances"],
                "target_specs": kwargs["target_specs"],
            }
        )
        return rows[len(calls) - 1]

    result = candidate_recheck_impl(
        case,
        ["left", "right"],
        ExperimentConfig(candidate_recheck_count=2, enable_artifact=True),
        [finding],
        run_loaded_case_fn=fake_run_loaded_case,
    )

    assert len(calls) == 2
    assert all(call["case"] is case for call in calls)
    assert all(call["backends"] == ["left", "right"] for call in calls)
    assert all(call["candidate_recheck_count"] == 0 for call in calls)
    assert all(call["evidence_tier"] == "fresh_confirmation" for call in calls)
    assert all(call["enable_artifact"] is False for call in calls)
    assert all(call["save_artifact"] is False for call in calls)
    assert all(call["backend_instances"] is None for call in calls)
    assert all(call["target_specs"] == [] for call in calls)
    assert result["enabled"] is True
    assert result["attempts"] == 2
    assert result["attempt_summaries"][0]["reproduced_keys"] == [
        "semantic_output_mismatch:filter_predicate@right:row_count"
    ]
    assert result["attempt_summaries"][1]["reproduced_keys"] == []
    assert result["attempt_summaries"][0]["backend_reported_total_ms"] == 6.0
    assert result["attempt_summaries"][1]["backend_reported_total_ms"] == 4.0
    assert result["backend_reported_total_ms"] == 10.0
    assert result["attempt_summaries"][0]["backend_calls"] == 6
    assert result["attempt_summaries"][1]["backend_calls"] == 4
    assert result["backend_calls"] == 10
    assert result["attempt_summaries"][0]["physical_plans"]["left"]["detail"] == (
        "full"
    )
    assert result["attempt_summaries"][0]["stage_profile"]["total_case_wall_ms"] == 5.0
    assert result["attempt_summaries"][1]["stage_profile"]["total_case_wall_ms"] == 6.0
    assert result["stage_profile_totals"]["backend_execution_ms"] == 7.0
    assert result["stage_profile_totals"]["normalize_ms"] == 1.0
    assert result["stage_profile_totals"]["oracle_classification_ms"] == 3.0
    assert result["stage_profile_totals"]["total_case_wall_ms"] == 11.0
    assert result["reproduced_keys"] == []
    assert result["non_reproduced_keys"] == [
        "semantic_output_mismatch:filter_predicate@right:row_count"
    ]
    assert finding.triage_verdict == "non_reproducible_candidate"
    assert finding.false_positive is True


def test_candidate_recheck_impl_keeps_stably_reproduced_findings_countable():
    finding = _finding()

    def fake_run_loaded_case(*args, **kwargs):
        return {"findings": [_finding_row()]}

    result = candidate_recheck_impl(
        _case(3),
        ["left", "right"],
        ExperimentConfig(candidate_recheck_count=2),
        [finding],
        run_loaded_case_fn=fake_run_loaded_case,
    )

    assert result["reproduced_keys"] == [
        "semantic_output_mismatch:filter_predicate@right:row_count"
    ]
    assert result["non_reproduced_keys"] == []
    assert finding.triage_verdict == "candidate_implementation_bug"
    assert finding.false_positive is False
