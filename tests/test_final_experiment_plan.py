from __future__ import annotations

import importlib.util
import sys
from argparse import Namespace
from pathlib import Path


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
        "live_seeds": "1,1001,2001",
        "historical_seeds": None,
        "seeded_cases": 5000,
        "seeded_seeds": "1,1001,2001,3001,4001",
        "jobs": 1,
        "artifact_limit": 50,
        "log_level": "compact",
        "include_pending_historical": False,
        "skip_run_reports": True,
        "execute": False,
    }
    data.update(overrides)
    return Namespace(**data)


def test_default_final_plan_includes_only_confirmed_historical_specs():
    module = _module()

    commands = module.build_plan(_args())

    assert {command.track for command in commands} == {"live", "historical", "seeded"}
    historical = [command for command in commands if command.track == "historical"]
    assert [command.name for command in historical] == ["duckdb-22075", "duckdb-22656"]
    assert all(command.count_as_real_bugs for command in historical)
    assert sum(1 for command in commands if command.track == "live") == 5
    assert sum(1 for command in commands if command.track == "seeded") == 1


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

    assert live
    assert all("--evidence-mode" in command.command and "live" in command.command for command in live)
    assert all("--run-theme" in command.command and "--paper-notes" in command.command for command in live)
    assert "--evidence-mode" in by_track["seeded"].command
    assert "seeded" in by_track["seeded"].command
    assert "--run-theme" in by_track["seeded"].command


def test_final_plan_explicitly_separates_fresh_and_replay_policy():
    module = _module()

    commands = module.build_plan(_args())
    live = [command for command in commands if command.track == "live"]
    historical = [command for command in commands if command.track == "historical"]

    assert live
    assert all("--enable-replay-bug" not in command.command for command in live)
    assert all("--replay-bug-source-issues" in command.command for command in live)
    assert all(command.replay_bug_policy["enable_replay_bug"] is False for command in live)
    assert historical
    assert all("--enable-replay-bug" in command.command for command in historical)
    assert all(command.replay_bug_policy["enable_replay_bug"] is True for command in historical)
    assert "https://github.com/duckdb/duckdb/issues/22075" in _flag_value(
        {command.name: command for command in historical}["duckdb-22075"].command,
        "--replay-bug-source-issues",
    )


def test_execute_all_is_rejected_for_mixed_environments(capsys):
    module = _module()

    rc = module.main_with_args_for_test(_args(execute=True))

    assert rc == 2
    assert "different Python environments" in capsys.readouterr().err


def _flag_value(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]
