from pathlib import Path

from datadiff.fuzz_loop import (
    FuzzBudget,
    FuzzIteration,
    IntervalGate,
    RunCounters,
    RunPaths,
    StageTimings,
)


def test_fuzz_budget_defaults_to_100_cases_without_explicit_budget():
    budget = FuzzBudget.start(cases=None, duration_s=None, now=10.0)

    assert budget.cases == 100
    assert budget.duration_s is None
    assert budget.started == 10.0


def test_fuzz_budget_allows_first_iteration_for_duration_only_runs():
    budget = FuzzBudget.start(cases=None, duration_s=5.0, now=10.0)

    assert budget.should_start_iteration(executed=0, now=20.0) is True
    assert budget.should_start_iteration(executed=1, now=14.9) is True
    assert budget.should_start_iteration(executed=1, now=15.0) is False
    assert budget.should_stop_after_iteration(now=15.0) is True


def test_fuzz_budget_stops_at_case_limit():
    budget = FuzzBudget.start(cases=2, duration_s=None, now=10.0)

    assert budget.should_start_iteration(executed=0, now=10.0) is True
    assert budget.should_start_iteration(executed=1, now=10.0) is True
    assert budget.should_start_iteration(executed=2, now=10.0) is False


def test_run_paths_builds_run_case_and_checkpoint_paths():
    paths = RunPaths.build(
        run_id="run-test",
        runs_dir=Path("/tmp/runs"),
        corpus_dir=Path("/tmp/corpus"),
        compress_run_log=True,
        save_cases=True,
        case_log_file=None,
        checkpoint_interval_s=10.0,
    )

    assert paths.run_id == "run-test"
    assert paths.run_file == Path("/tmp/runs/run-test.jsonl.gz")
    assert paths.case_log_file == Path("/tmp/corpus/generated/run-test.cases.jsonl")
    assert paths.checkpoint_file == Path("/tmp/runs/run-test.checkpoint.json")


def test_run_paths_respects_explicit_case_log_and_uncompressed_log():
    explicit_case_log = Path("/tmp/custom.cases.jsonl")

    paths = RunPaths.build(
        run_id="run-test",
        runs_dir=Path("/tmp/runs"),
        corpus_dir=Path("/tmp/corpus"),
        compress_run_log=False,
        save_cases=False,
        case_log_file=explicit_case_log,
        checkpoint_interval_s=None,
    )

    assert paths.run_file == Path("/tmp/runs/run-test.jsonl")
    assert paths.case_log_file == explicit_case_log
    assert paths.checkpoint_file is None


def test_interval_gate_tracks_due_and_mark():
    gate = IntervalGate(interval_s=2.5, last_fire_s=10.0)

    assert gate.due(12.4) is False
    assert gate.due(12.5) is True

    gate.mark(12.5)

    assert gate.last_fire_s == 12.5
    assert gate.due(14.9) is False
    assert gate.due(15.0) is True


def test_interval_gate_is_disabled_without_interval():
    gate = IntervalGate(interval_s=None, last_fire_s=10.0)

    assert gate.due(100.0) is False


def test_run_counters_record_completed_case_and_summaries():
    counters = RunCounters()

    counters.record_completed_case(
        row={
            "findings": [{"kind": "semantic_output_mismatch"}, {"kind": "exception_mismatch"}],
            "is_new_behavior": True,
            "signal_new_behavior": False,
        },
        preflight={
            "repaired": True,
            "fallback_used": True,
            "valid": False,
        },
    )

    assert counters.executed == 1
    assert counters.findings == 2
    assert counters.new_behavior_cases == 1
    assert counters.signal_new_behavior_cases == 0
    assert counters.preflight_summary() == {
        "repaired_cases": 1,
        "fallback_cases": 1,
        "invalid_cases": 1,
    }


def test_run_counters_filter_non_rewardable_signal_new_behavior():
    counters = RunCounters(
        known_saturated_bug_families=("csv_long_numeric_roundtrip@duckdb",)
    )
    rows = [
        {
            "is_new_behavior": True,
            "signal_new_behavior": True,
            "findings": [
                {
                    "triage_verdict": "normalizer_false_positive",
                    "root_cause": "order_only_normalization_mismatch",
                    "false_positive": True,
                }
            ],
        },
        {
            "is_new_behavior": True,
            "signal_new_behavior": True,
            "findings": [
                {
                    "triage_verdict": "candidate_implementation_bug",
                    "root_cause": "csv_long_numeric_roundtrip",
                    "suspicious_backends": ["duckdb", "pyarrow"],
                }
            ],
        },
        {
            "is_new_behavior": True,
            "signal_new_behavior": True,
            "findings": [
                {
                    "triage_verdict": "candidate_implementation_bug",
                    "root_cause": "known_source_issue",
                    "suspicious_backends": ["duckdb"],
                    "source_issue": "duckdb/duckdb#12345",
                }
            ],
        },
        {
            "is_new_behavior": True,
            "signal_new_behavior": True,
            "findings": [
                {
                    "triage_verdict": "candidate_implementation_bug",
                    "root_cause": "topk_filter_pushdown",
                    "suspicious_backends": ["datafusion"],
                    "discovery_origin": "organic",
                }
            ],
        },
        {
            "is_new_behavior": True,
            "signal_new_behavior": True,
            "findings": [],
        },
    ]

    for row in rows:
        counters.record_completed_case(row=row, preflight={"valid": True})

    assert counters.executed == 5
    assert counters.new_behavior_cases == 5
    assert counters.signal_new_behavior_cases == 2


def test_run_counters_record_quality_oracles_and_filter_summaries():
    counters = RunCounters()
    counters.replay_filtered_candidates = 3
    counters.replay_fallback_candidates = 1
    counters.saturated_family_filtered_candidates = 2
    counters.saturated_family_fallback_candidates = 1

    counters.record_quality_oracle({"name": "validity", "verdict": "pass"})
    counters.record_quality_oracle({"name": "validity", "verdict": "pass"})
    counters.record_quality_oracle({"name": "novelty", "verdict": "warn"})

    assert counters.quality_oracles == {
        "validity:pass": 2,
        "novelty:warn": 1,
    }
    assert counters.replay_filter_summary(enabled=True) == {
        "enabled": True,
        "filtered_candidates": 3,
        "fallback_candidates": 1,
    }
    assert counters.family_saturation_filter_summary(enabled=False) == {
        "enabled": False,
        "filtered_candidates": 2,
        "fallback_candidates": 1,
    }


def test_stage_timings_from_mapping_computes_missing_total():
    timings = StageTimings.from_mapping(
        {
            "generate_mutate_ms": 1.0,
            "backend_execution_ms": 2.0,
            "normalize_ms": 3.0,
        }
    )

    assert timings.total_case_wall_ms == 6.0
    assert timings.to_dict()["backend_execution_ms"] == 2.0


def test_fuzz_iteration_round_trips_from_runner_row():
    iteration = FuzzIteration.from_row(
        {
            "run_id": "run-a",
            "case_index": 3,
            "candidate_seed_start": 100,
            "candidate_pool_size": 4,
            "candidate_source": "feedback_mutation",
            "status": "bug",
            "case": {"case_id": "case-a", "seed": 101},
            "findings": [{"kind": "semantic_output_mismatch"}],
            "is_new_behavior": True,
            "signal_new_behavior": True,
            "selected_generator_profile": "typed_grammar",
            "selected_semantic_objective": "objective:null_boundary",
            "selected_metamorphic_relation": "limit_idempotence",
            "selected_version_pair": "latest->fixed",
            "backend_pair_priority": ["pandas|polars"],
            "stage_profile": {
                "generate_mutate_ms": 1.0,
                "backend_execution_ms": 2.0,
                "total_case_wall_ms": 3.0,
            },
            "execution_profile": {"parallel_backend_execution": True, "backend_count": 2},
            "guidance": {"strategy": "guided"},
            "preflight": {"valid": True},
            "operation_combo": {"operation_count": 2},
            "source_reward": 1.25,
            "stored_in_feedback_corpus": True,
            "feedback_eligible": True,
        }
    )

    payload = iteration.to_dict()
    assert payload["case_id"] == "case-a"
    assert payload["finding_count"] == 1
    assert payload["candidate_source"] == "feedback_mutation"
    assert payload["backend_pair_priority"] == ["pandas|polars"]
    assert payload["stage_timings"]["total_case_wall_ms"] == 3.0
    assert payload["execution_profile"]["parallel_backend_execution"] is True
    assert payload["reward"]["source_reward"] == 1.25
