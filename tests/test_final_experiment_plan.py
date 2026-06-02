from __future__ import annotations

import importlib.util
import sys
from argparse import Namespace
from pathlib import Path
import json


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_final_experiments.py"
    spec = importlib.util.spec_from_file_location("run_final_experiments", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _args(**overrides):
    data = {
        "track": "all",
        "duration": "24h",
        "validation_cases": 200,
        "validation_seeds": "1,101",
        "live_seeds": "1,1001,2001",
        "historical_seeds": None,
        "seeded_cases": 5000,
        "seeded_seeds": "1,1001,2001,3001,4001",
        "ablation_cases": 2000,
        "ablation_seeds": "1,1001,2001",
        "comparison_cases": 2000,
        "comparison_seeds": "1,1001,2001",
        "jobs": 1,
        "artifact_limit": 50,
        "log_level": "compact",
        "include_pending_historical": False,
        "skip_run_reports": True,
        "strategy_snapshot": "",
        "execute": False,
    }
    data.update(overrides)
    return Namespace(**data)


def test_default_final_plan_includes_only_confirmed_historical_specs():
    module = _module()

    commands = module.build_plan(_args())

    assert {command.track for command in commands} == {
        "validation",
        "live",
        "historical",
        "seeded",
        "ablation",
        "comparison",
    }
    historical = [command for command in commands if command.track == "historical"]
    assert [command.name for command in historical] == ["duckdb-22075", "duckdb-22656"]
    assert all(command.count_as_real_bugs for command in historical)
    assert sum(1 for command in commands if command.track == "live") == 11
    assert sum(1 for command in commands if command.track == "validation") == 1
    assert sum(1 for command in commands if command.track == "seeded") == 1
    assert sum(1 for command in commands if command.track == "ablation") == 1
    assert sum(1 for command in commands if command.track == "comparison") == 1


def test_pending_historical_specs_are_explicit_case_studies():
    module = _module()

    commands = module.build_plan(_args(track="historical", include_pending_historical=True))

    by_name = {command.name: command for command in commands}
    assert set(by_name) == {
        "duckdb-3015",
        "duckdb-11261",
        "duckdb-22075",
        "duckdb-22656",
        "arrow-42231",
        "datafusion-22190",
        "datafusion-22441",
    }
    assert by_name["duckdb-22075"].count_as_real_bugs is True
    assert by_name["duckdb-22656"].count_as_real_bugs is True
    assert "duckdb_storage_cross" in by_name["duckdb-22656"].command
    assert "storage_offset" in by_name["duckdb-22656"].command
    assert _flag_value(by_name["duckdb-22656"].command, "--artifact-limit") == "0"
    assert _flag_value(by_name["duckdb-22656"].command, "--log-level") == "minimal"
    assert by_name["datafusion-22190"].count_as_real_bugs is False
    assert by_name["datafusion-22441"].count_as_real_bugs is False
    assert by_name["duckdb-3015"].count_as_real_bugs is False
    assert by_name["duckdb-11261"].count_as_real_bugs is False
    assert by_name["arrow-42231"].count_as_real_bugs is False
    assert "wide_offset_topk" in _flag_value(by_name["duckdb-11261"].command, "--presets")
    assert "join_groupby_stress" in _flag_value(by_name["arrow-42231"].command, "--presets")
    assert "--evidence-mode" in by_name["datafusion-22190"].command
    assert "historical" in by_name["datafusion-22190"].command
    assert "join_null_truth_filter" in _flag_value(by_name["datafusion-22441"].command, "--presets")
    assert "replay-fixture" in by_name["duckdb-3015"].command
    assert "DATADIFF_DUCKDB_3015_FIXTURE" in by_name["duckdb-3015"].command


def test_live_and_seeded_commands_record_evidence_mode():
    module = _module()

    commands = module.build_plan(_args())
    by_track = {command.track: command for command in commands if command.track == "seeded"}
    live = [command for command in commands if command.track == "live"]
    live_names = {command.name for command in live}
    expected_live_names = {
        f"{campaign.suite}:{campaign.preset}"
        for campaign in module.FINAL_LIVE_DISCOVERY_MATRIX.campaigns
    }

    assert live
    assert live_names == expected_live_names
    assert "datafusion_cross:live_datafusion_fresh" in live_names
    assert "latest_no_datafusion:live_cross_family" in live_names
    assert "polars_cross:live_polars_issue_focus" in live_names
    assert "embedded_sql_cross:live_duckdb_issue_focus" in live_names
    assert "arrow_cross:live_arrow_issue_focus" in live_names
    assert "latest_no_datafusion:live_issue_focus" in live_names
    assert all("--evidence-mode" in command.command and "live" in command.command for command in live)
    assert all("--run-theme" in command.command and "--paper-notes" in command.command for command in live)
    assert "--evidence-mode" in by_track["seeded"].command
    assert "seeded" in by_track["seeded"].command
    assert "--run-theme" in by_track["seeded"].command


def test_validation_command_gates_short_before_long_runs():
    module = _module()

    commands = module.build_plan(_args(track="validation", validation_cases=17, validation_seeds="3"))
    assert len(commands) == 1
    validation = commands[0]

    assert validation.track == "validation"
    assert validation.name == "short_validation_smoke"
    assert validation.count_as_real_bugs is False
    assert _flag_value(validation.command, "--cases") == "17"
    assert _flag_value(validation.command, "--seeds") == "3"
    assert _flag_value(validation.command, "--evidence-mode") == "validation"
    assert _flag_value(validation.command, "--log-level") == "compact"
    assert "datafusion_cross" in _flag_value(validation.command, "--target-suites")
    assert "latest_no_datafusion" in _flag_value(validation.command, "--target-suites")
    assert "live_common_api_workflow_metamorphic" in _flag_value(validation.command, "--presets")
    assert "live_deep_organic_metamorphic" in _flag_value(validation.command, "--presets")
    assert validation.replay_bug_policy["enable_replay_bug"] is False
    assert "run-health/classify-run" in validation.expected_output


def test_ablation_and_comparison_commands_cover_method_rqs():
    module = _module()

    commands = module.build_plan(_args(track="all"))
    by_name = {command.name: command for command in commands}
    ablation = by_name["module_ablation"]
    comparison = by_name["baseline_and_related_scope"]

    assert ablation.count_as_real_bugs is False
    assert comparison.count_as_real_bugs is False
    assert "no_type_aware" in _flag_value(ablation.command, "--presets")
    assert "no_normalizer" in _flag_value(ablation.command, "--presets")
    assert "no_feedback" in _flag_value(ablation.command, "--presets")
    assert "oracle_only_metamorphic" in _flag_value(ablation.command, "--presets")
    assert "core_datafusion" in _flag_value(ablation.command, "--target-suites")
    assert "core_arrow" in _flag_value(ablation.command, "--target-suites")
    assert "embedded_sql" in _flag_value(comparison.command, "--target-suites")
    assert "latest_all_engines" in _flag_value(comparison.command, "--target-suites")
    assert "bughunt_guided" in _flag_value(comparison.command, "--presets")
    assert "live_cross_family" in _flag_value(comparison.command, "--presets")
    assert _flag_value(ablation.command, "--evidence-mode") == "ablation"
    assert _flag_value(comparison.command, "--evidence-mode") == "comparison"
    assert ablation.replay_bug_policy["enable_replay_bug"] is False
    assert comparison.replay_bug_policy["enable_replay_bug"] is False


def test_comparison_matrix_variants_are_explicitly_marked_as_contrast_or_baseline():
    module = _module()

    roles = {
        variant.preset: variant.comparison_role
        for variant in module.FINAL_COMPARISON_MATRIX.variants
    }

    assert roles["baseline"] == "baseline"
    assert roles["guided"] == "contrast"
    assert roles["bughunt"] == "contrast"
    assert roles["bughunt_guided"] == "contrast"
    assert roles["metamorphic"] == "contrast"
    assert roles["oracle_only_metamorphic"] == "contrast"
    assert roles["workflow"] == "contrast"
    assert roles["workflow_metamorphic"] == "contrast"
    assert roles["live_cross_family"] == "contrast"


def test_final_plan_commands_include_structured_experiment_meta():
    module = _module()

    commands = module.build_plan(_args(track="all"))
    by_name = {command.name: command for command in commands}

    validation = by_name["short_validation_smoke"]
    validation_meta = json.loads(_flag_value(validation.command, "--experiment-meta"))
    assert validation_meta["matrix_id"] == "final_validation"
    assert validation_meta["comparison_group"] == "validation_smoke"
    assert "validation" in validation_meta["analysis_tags"]

    live = by_name["latest_no_datafusion:live_issue_focus"]
    live_meta = json.loads(_flag_value(live.command, "--experiment-meta"))
    assert live_meta["matrix_id"] == "live_discovery"
    assert live_meta["variant"]["variant_id"] == "live_issue_focus"
    assert live_meta["scope_kind"] == "cross_ecosystem"
    assert live_meta["variant"]["scope_kind"] == "cross_ecosystem"
    assert live_meta["counts_as_real_bugs"] is True

    ablation = by_name["module_ablation"]
    ablation_meta = json.loads(_flag_value(ablation.command, "--experiment-meta"))
    assert ablation_meta["matrix_id"] == "module_ablation"
    assert ablation_meta["comparison_group"] == "module_ablation"
    assert "RQ2" in ablation_meta["rq_tags"]

    comparison = by_name["baseline_and_related_scope"]
    comparison_meta = json.loads(_flag_value(comparison.command, "--experiment-meta"))
    assert comparison_meta["matrix_id"] == "baseline_scope_comparison"
    assert comparison_meta["comparison_group"] == "scope_comparison"
    assert "scope" in comparison_meta["analysis_tags"]

    historical = by_name["duckdb-22075"]
    historical_meta = json.loads(_flag_value(historical.command, "--experiment-meta"))
    assert historical_meta["matrix_id"] == "historical_replay"
    assert historical_meta["comparison_group"] == "historical_replay"
    assert historical_meta["historical"]["bug_id"] == "duckdb-22075"
    assert historical_meta["historical"]["status"] == "confirmed_fixed"
    assert historical_meta["variant"]["variant_id"] == "duckdb-22075"


def test_final_plan_freezes_dynamic_strategy_snapshot():
    module = _module()

    commands = module.build_plan(_args(track="validation"))

    assert len(commands) == 1
    command = commands[0]
    snapshot_path = _flag_value(command.command, "--strategy-snapshot")
    assert snapshot_path
    assert "--freeze-strategy-snapshot" in command.command
    assert Path(snapshot_path).is_file()


def test_final_plan_explicitly_separates_fresh_and_replay_policy():
    module = _module()

    commands = module.build_plan(_args())
    live = [command for command in commands if command.track == "live"]
    historical = [command for command in commands if command.track == "historical"]
    support = [command for command in commands if command.track in {"validation", "ablation", "comparison"}]

    assert live
    assert all("--enable-replay-bug" not in command.command for command in live)
    assert all("--replay-bug-source-issues" in command.command for command in live)
    assert all(command.replay_bug_policy["enable_replay_bug"] is False for command in live)
    assert support
    assert all("--enable-replay-bug" not in command.command for command in support)
    assert all("--replay-bug-source-issues" in command.command for command in support)
    assert all(command.replay_bug_policy["enable_replay_bug"] is False for command in support)
    assert historical
    assert all("--enable-replay-bug" in command.command for command in historical)
    assert all(command.replay_bug_policy["enable_replay_bug"] is True for command in historical)
    assert "https://github.com/duckdb/duckdb/issues/22075" in _flag_value(
        {command.name: command for command in historical}["duckdb-22075"].command,
        "--replay-bug-source-issues",
    )


def test_final_plan_keeps_paper_run_journal_enabled():
    module = _module()

    commands = module.build_plan(_args(track="all"))
    assert commands
    assert all("--skip-paper-journal" not in command.command for command in commands)

    plan_path = module.write_plan(commands, _args(track="all"))
    payload = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    assert "paper_run_journal" in payload["policy"]
    assert "paper-run-journal.jsonl" in payload["policy"]["paper_run_journal"]


def test_execute_all_is_rejected_for_mixed_environments(capsys):
    module = _module()

    rc = module.run_with_args(_args(execute=True))

    assert rc == 2
    assert "different Python environments" in capsys.readouterr().err


def _flag_value(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]
