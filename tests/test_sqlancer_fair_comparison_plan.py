from __future__ import annotations

import importlib.util
import sys
from argparse import Namespace
from pathlib import Path


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "plan_sqlancer_fair_comparison.py"
    spec = importlib.util.spec_from_file_location("plan_sqlancer_fair_comparison", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _args(tmp_path: Path, **overrides):
    data = {
        "run_id": "pilot",
        "output_dir": str(tmp_path),
        "duration_seconds": 7200,
        "timeout_slack_seconds": 300,
        "seeds": "1,1001,2001",
        "sqlancer_root": "/tmp/sqlancer",
        "sqlancer_suite": ["duckdb-query-partitioning", "duckdb-norec"],
        "sqlancer_num_queries": 1000000000,
        "sqlancer_max_generated_databases": 1000000,
        "sqlancer_manifest_dir": str(tmp_path / "sqlancer"),
        "datadiff_target_suite": "embedded_sql",
        "datadiff_preset": "live_duckdb_issue_focus",
        "datadiff_batch_duration": "10m",
        "datadiff_adaptive_learning_weight": 0.75,
        "datadiff_scheduler_annealing_temperature": 0.35,
        "datadiff_scheduler_annealing_decay": 0.985,
        "datadiff_scheduler_annealing_min_temperature": 0.02,
        "datadiff_artifact_limit": 50,
        "datadiff_log_level": "minimal",
        "print_commands": False,
    }
    data.update(overrides)
    return Namespace(**data)


def test_plan_uses_one_sqlancer_command_per_oracle_and_one_datadiff_command_per_seed(tmp_path):
    module = _module()

    commands = module.build_commands(_args(tmp_path), seeds=[1, 1001, 2001], run_id="pilot")

    sqlancer = [command for command in commands if command["tool"] == "sqlancer"]
    datadiff = [command for command in commands if command["tool"] == "datadiff"]
    assert [command["suite"] for command in sqlancer] == ["duckdb-query-partitioning", "duckdb-norec"]
    assert [command["seed"] for command in datadiff] == [1, 1001, 2001]
    assert all("--num-threads 1" in command["shell"] for command in sqlancer)
    assert all("--timeout-seconds 7200" in command["shell"] for command in sqlancer)
    assert all("--no-log-each-select" not in command["shell"] for command in sqlancer)
    assert all("--no-log-execution-time" in command["shell"] for command in sqlancer)
    assert all("--duration 7200s" in command["shell"] for command in datadiff)
    assert all("--target-suite embedded_sql" in command["shell"] for command in datadiff)
    assert all("--seeds 1,1001,2001" not in command["shell"] for command in datadiff)
    assert all("--evidence-mode comparison" in command["shell"] for command in datadiff)
    assert all("--evidence-mode live" not in command["shell"] for command in datadiff)
    assert all("sqlancer-fair-comparison:" in command["shell"] for command in datadiff)
    assert all("final-live:" not in command["shell"] for command in datadiff)
    assert all("external_sqlancer_scope_limited" in command["shell"] for command in datadiff)
    assert all('"counts_as_real_bugs": false' in command["shell"] for command in datadiff)


def test_run_with_args_writes_manifest_and_script(tmp_path):
    module = _module()

    rc = module.run_with_args(_args(tmp_path, duration_seconds=20))

    assert rc == 0
    manifest = tmp_path / "pilot.json"
    script = tmp_path / "pilot.sh"
    assert manifest.is_file()
    assert script.is_file()
    assert "sqlancer-fair-comparison-plan-v1" in manifest.read_text(encoding="utf-8")
    assert "scripts/run_sqlancer_baseline.py" in script.read_text(encoding="utf-8")
