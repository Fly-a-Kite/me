from pathlib import Path

from datadiff.adaptive_benchmark import (
    ADAPTIVE_BENCHMARK_SCHEMA_VERSION,
    ReplayRunSource,
    _extract_replay_events,
    run_adaptive_benchmark,
    write_adaptive_benchmark_markdown,
)
from datadiff.util import append_jsonl, dump_json


def test_adaptive_benchmark_splits_learning_and_systems_metrics(tmp_path: Path):
    profile_path = tmp_path / "adaptive.prof"

    payload = run_adaptive_benchmark(
        mode="all",
        learning_rounds=8,
        systems_iterations=64,
        profile_iterations=48,
        action_pool_size=6,
        profile_top_n=8,
        profile_output=profile_path,
    )

    assert payload["schema_version"] == ADAPTIVE_BENCHMARK_SCHEMA_VERSION
    assert "learning_effectiveness" in payload
    assert "systems_benchmark" in payload
    assert payload["learning_effectiveness"]["summary"]["baseline_variant"] == "reward_signal_only"
    assert payload["systems_benchmark"]["summary"]["action_pool_size"] == 6
    assert payload["systems_benchmark"]["profiler"]["enabled"] is True
    assert Path(payload["systems_benchmark"]["profiler"]["profile_output"]) == profile_path
    assert profile_path.exists()
    assert any(
        row["scenario_id"] == "reward_model_context_split"
        for row in payload["learning_effectiveness"]["scenarios"]
    )
    assert any(
        row["name"] == "bandit_rank_dense"
        for row in payload["systems_benchmark"]["benchmarks"]
    )


def test_adaptive_benchmark_markdown_renders_key_sections(tmp_path: Path):
    payload = run_adaptive_benchmark(
        mode="learning",
        learning_rounds=6,
    )
    markdown_path = tmp_path / "adaptive-benchmark.md"

    write_adaptive_benchmark_markdown(payload, markdown_path)

    content = markdown_path.read_text(encoding="utf-8")
    assert "# Adaptive Benchmark" in content
    assert "## Learning Effectiveness" in content
    assert "reward_model_context_split" in content
    assert "reward_signal_only" in content


def test_adaptive_replay_recomputes_polluted_selection_reward(tmp_path: Path):
    run_file = tmp_path / "run-polluted-reward.jsonl"
    append_jsonl(
        {
            "case_index": 0,
            "duration_ms": 10.0,
            "is_new_behavior": True,
            "signal_new_behavior": True,
            "findings": [
                {
                    "triage_verdict": "normalizer_false_positive",
                    "root_cause": "order_only_normalization_mismatch",
                    "false_positive": True,
                }
            ],
            "preflight": {"valid": True, "fallback_used": False},
            "generator_profile_selection": {
                "strategy": "contextual_bandit",
                "profile": "profile_bad",
                "profile_pool": ["profile_bad", "profile_good"],
                "reward": 99.0,
            },
            "selected_generator_profile": "profile_bad",
            "case_learning_context": ["semantic_family:groupby"],
        },
        run_file,
    )

    events, row_count = _extract_replay_events(
        ReplayRunSource(
            order_index=0,
            run_file=run_file,
            source_refs=("test",),
            meta={"config": {}},
        )
    )

    assert row_count == 1
    assert len(events) == 1
    assert events[0].reward < 0.0


def test_adaptive_replay_filters_known_saturated_families_from_reward(tmp_path: Path):
    run_file = tmp_path / "run-known-family.jsonl"
    append_jsonl(
        {
            "case_index": 0,
            "duration_ms": 0.0,
            "is_new_behavior": True,
            "signal_new_behavior": True,
            "findings": [
                {
                    "triage_verdict": "candidate_implementation_bug",
                    "root_cause": "csv_long_numeric_roundtrip",
                    "suspicious_backends": ["duckdb", "pyarrow"],
                }
            ],
            "preflight": {"valid": True, "fallback_used": False},
            "generator_profile_selection": {
                "strategy": "contextual_bandit",
                "profile": "profile_known",
                "profile_pool": ["profile_known", "profile_fresh"],
                "reward": 99.0,
            },
            "selected_generator_profile": "profile_known",
            "case_learning_context": ["semantic_family:csv"],
        },
        run_file,
    )

    events, _ = _extract_replay_events(
        ReplayRunSource(
            order_index=0,
            run_file=run_file,
            source_refs=("test",),
            meta={"config": {"known_saturated_bug_families": ["csv_long_numeric_roundtrip@duckdb"]}},
        )
    )

    assert len(events) == 1
    assert events[0].reward == 0.0


def test_adaptive_benchmark_replays_real_run_logs_and_renders_section(tmp_path: Path):
    run_file = tmp_path / "run-replay.jsonl"
    append_jsonl(
        {
            "case_index": 0,
            "duration_ms": 10.0,
            "is_new_behavior": True,
            "signal_new_behavior": True,
            "findings": [
                {
                    "kind": "semantic_output_mismatch",
                    "root_cause": "family_a",
                    "triage_verdict": "candidate_implementation_bug",
                    "suspicious_backends": ["engine"],
                }
            ],
            "preflight": {"valid": True, "fallback_used": False},
            "generator_profile_selection": {
                "strategy": "contextual_bandit",
                "profile": "profile_join",
                "profile_pool": ["profile_groupby", "profile_join"],
                "reward": 3.2,
            },
            "selected_generator_profile": "profile_join",
            "version_pair_selection": {
                "strategy": "contextual_bandit",
                "action": "latest->preview",
                "action_pool": ["latest->fixed", "latest->preview"],
                "reward": 3.2,
            },
            "selected_version_pair": "latest->preview",
            "semantic_objective_selection": {
                "strategy": "contextual_bandit",
                "action": "exploration_objective:join_null_boundary",
                "action_pool": [
                    "exploration_objective:join_groupby_mix",
                    "exploration_objective:join_null_boundary",
                ],
                "reward": 3.2,
            },
            "selected_semantic_objective": "exploration_objective:join_null_boundary",
            "metamorphic_relation_selection": {
                "strategy": "contextual_bandit",
                "action": "input_partition_union_all",
                "action_pool": ["limit_pushdown", "input_partition_union_all"],
                "reward": 3.2,
            },
            "selected_metamorphic_relation": "input_partition_union_all",
            "case_learning_context": [
                "oracle_mode:differential",
                "guidance_strategy:guided",
                "backend_count:single",
                "backend:duckdb",
                "semantic_family:join",
                "semantic_signal:null_boundary",
                "exploration_objective:join_null_boundary",
                "matched_target:join",
                "capability:op:join",
                "feedback:enabled",
            ],
        },
        run_file,
    )
    append_jsonl(
        {
            "case_index": 1,
            "duration_ms": 40.0,
            "is_new_behavior": False,
            "signal_new_behavior": False,
            "findings": [],
            "preflight": {"valid": False, "fallback_used": True},
            "generator_profile_selection": {
                "strategy": "contextual_bandit",
                "profile": "profile_groupby",
                "profile_pool": ["profile_groupby", "profile_join"],
                "reward": -0.75,
            },
            "selected_generator_profile": "profile_groupby",
            "version_pair_selection": {
                "strategy": "contextual_bandit",
                "action": "latest->fixed",
                "action_pool": ["latest->fixed", "latest->preview"],
                "reward": -0.75,
            },
            "selected_version_pair": "latest->fixed",
            "semantic_objective_selection": {
                "strategy": "contextual_bandit",
                "action": "exploration_objective:join_groupby_mix",
                "action_pool": [
                    "exploration_objective:join_groupby_mix",
                    "exploration_objective:join_null_boundary",
                ],
                "reward": -0.75,
            },
            "selected_semantic_objective": "exploration_objective:join_groupby_mix",
            "metamorphic_relation_selection": {
                "strategy": "contextual_bandit",
                "action": "limit_pushdown",
                "action_pool": ["limit_pushdown", "input_partition_union_all"],
                "reward": -0.75,
            },
            "selected_metamorphic_relation": "limit_pushdown",
            "case_learning_context": [
                "oracle_mode:differential",
                "guidance_strategy:guided",
                "backend_count:single",
                "backend:duckdb",
                "semantic_family:groupby",
                "semantic_signal:stable_order",
                "exploration_objective:join_groupby_mix",
                "matched_target:groupby",
                "capability:op:groupby",
                "feedback:enabled",
            ],
        },
        run_file,
    )
    dump_json(
        {
            "config": {
                "oracle_mode": "differential",
                "guidance_strategy": "guided",
                "enable_feedback": True,
                "enable_metamorphic_oracle": True,
                "semantic_focus_families": ["join"],
                "semantic_focus_signals": ["null_boundary"],
                "guidance_targets": ["join"],
                "target_version": "latest",
                "fixed_version": "fixed",
            },
            "backends": ["duckdb"],
            "targets": [{"family": "analytics", "layer": "sql"}],
            "common_capabilities": ["op:join", "op:groupby"],
            "version_pair": "latest->fixed",
            "adaptive_learning_seed": {
                "continual_priority_memory": {
                    "feature_priority_totals": {"semantic_family:join": 2.0},
                    "feature_counts": {"semantic_family:join": 2},
                    "feature_health_penalty_totals": {},
                    "feature_health_counts": {},
                    "family_priorities": {"join_family@engine": 0.9},
                    "family_health_penalties": {},
                    "status_counts": {"regression": 1},
                    "imported_ledger_count": 1,
                    "imported_family_count": 1,
                    "imported_health_feedback_count": 0,
                }
            },
        },
        run_file.with_name("run-replay.meta.json"),
    )

    payload = run_adaptive_benchmark(mode="replay", replay_run_files=[run_file])

    assert payload["schema_version"] == ADAPTIVE_BENCHMARK_SCHEMA_VERSION
    replay = payload["real_run_replay"]
    assert replay["schema_version"] == "adaptive-real-run-replay-v1"
    assert replay["summary"]["run_file_count"] == 1
    assert replay["summary"]["event_count"] == 8
    assert replay["summary"]["baseline_variant"] == "reward_signal_only"
    assert replay["summary"]["scope_event_counts"]["generator_profile"] == 2
    assert {row["name"] for row in replay["variants"]} == {
        "reward_signal_only",
        "no_reward_model",
        "full_adaptive",
    }
    assert replay["seed_summary"]["imported_continual_seed_count"] == 1
    assert replay["sources"][0]["event_count"] == 8

    markdown_path = tmp_path / "adaptive-replay.md"
    write_adaptive_benchmark_markdown(payload, markdown_path)
    content = markdown_path.read_text(encoding="utf-8")
    assert "## Real Run Replay" in content
    assert "reward_signal_only" in content
    assert "full_adaptive" in content
