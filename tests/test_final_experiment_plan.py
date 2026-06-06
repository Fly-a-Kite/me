from __future__ import annotations

import importlib.util
import subprocess
import sys
from argparse import Namespace
from pathlib import Path
import json


ADAPTIVE_COMPONENT_DISABLE_FLAGS = {
    "active_learning": "active-learning",
    "backend_pair_learning": "backend-pair-learning",
    "bd_axis_bandit": "bd-axis-bandit",
    "bayesian_exploration": "bayesian-exploration",
    "champion_corpus": "champion-corpus",
    "continual_learning": "continual-learning",
    "cost_normalized_reward": "cost-normalized-reward",
    "disagreement_bd_axis": "disagreement-bd-axis",
    "divergence_conditioned": "divergence-conditioned",
    "hierarchical_archive": "hierarchical-archive",
    "ir_rewrite_mutations": "ir-rewrite-mutations",
    "lhs_seeding": "lhs-seeding",
    "lineage_rarity": "lineage-rarity",
    "minhash_dedup": "minhash-dedup",
    "online_reward_model": "online-reward-model",
    "operator_swarm": "operator-swarm",
    "per_operator_energy": "per-operator-energy",
    "quality_archive": "quality-archive",
    "runtime_cost_learning": "runtime-cost-learning",
    "scheduler_annealing": "scheduler-annealing",
    "scheduler_learning": "scheduler-learning",
    "seed_energy_batch": "seed-energy-batch",
    "seed_quota": "seed-quota",
    "shrink_mutations": "shrink-mutations",
    "value_catalog": "value-catalog",
}


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
        "live_batch_duration": "10m",
        "adaptive_learning_weight": 0.75,
        "scheduler_annealing_temperature": 0.35,
        "scheduler_annealing_decay": 0.985,
        "scheduler_annealing_min_temperature": 0.02,
        "continual_learning_ledgers": "",
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
        "reset_strategy_snapshot": False,
        "ledger_run_files": "",
        "ledger_versions": "",
        "previous_ledger": "",
        "ledger_output": "reports/final-version-ledger.json",
        "ledger_evidence_manifest": "reports/experiment-final-version-ledger.json",
        "manifest_index": "reports/final-experiment-manifest-index.json",
        "import_manifest": [],
        "import_extra_manifest": [],
        "paper_run_journal": "reports/paper-run-journal.jsonl",
        "reset_manifest_index": False,
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
        "postprocess",
    }
    historical = [command for command in commands if command.track == "historical"]
    assert [command.name for command in historical] == ["duckdb-22075", "duckdb-22656"]
    assert all(command.count_as_real_bugs for command in historical)
    assert sum(1 for command in commands if command.track == "live") == 11
    assert sum(1 for command in commands if command.track == "validation") == 1
    assert sum(1 for command in commands if command.track == "seeded") == 1
    assert sum(1 for command in commands if command.track == "ablation") == 27
    assert sum(1 for command in commands if command.track == "comparison") == 2
    assert sum(1 for command in commands if command.track == "postprocess") == 1


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
    assert all("--persist-closed-loop-state" in command.command for command in live)
    assert all(_flag_value(command.command, "--schedule") == "adaptive" for command in live)
    assert all(_flag_value(command.command, "--batch-duration") == "10m" for command in live)
    assert all(_flag_value(command.command, "--adaptive-learning-weight") == "0.75" for command in live)
    assert all(_flag_value(command.command, "--scheduler-annealing-temperature") == "0.35" for command in live)
    assert all(_flag_value(command.command, "--scheduler-annealing-decay") == "0.985" for command in live)
    assert all(_flag_value(command.command, "--scheduler-annealing-min-temperature") == "0.02" for command in live)
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


def test_final_plan_accepts_matrix_alias_tracks():
    module = _module()

    module_only = module.build_plan(_args(track="module_ablation"))
    assert [command.name for command in module_only] == ["module_ablation"]

    adaptive_only = module.build_plan(_args(track="adaptive_component_ablation"))
    assert adaptive_only
    assert all(command.name.startswith("adaptive_component_ablation:") for command in adaptive_only)

    comparison_only = module.build_plan(_args(track="baseline_scope_comparison"))
    assert [command.name for command in comparison_only] == ["baseline_and_related_scope"]

    ledger_only = module.build_plan(_args(track="version_ledger"))
    assert [command.name for command in ledger_only] == ["cross_version_regression_ledger"]


def test_final_plan_parse_args_accepts_matrix_alias_track(monkeypatch):
    module = _module()

    monkeypatch.setattr(sys, "argv", ["run_final_experiments.py", "--track", "module_ablation"])
    args = module.parse_args()

    assert args.track == "module_ablation"


def test_ablation_and_comparison_commands_cover_method_rqs():
    module = _module()

    commands = module.build_plan(_args(track="all"))
    by_name = {command.name: command for command in commands}
    ablation = by_name["module_ablation"]
    adaptive_reference = by_name["adaptive_component_ablation:adaptive_reference"]
    adaptive_contrasts = {
        component: by_name[f"adaptive_component_ablation:no_{component}"]
        for component in ADAPTIVE_COMPONENT_DISABLE_FLAGS
    }
    comparison = by_name["baseline_and_related_scope"]

    assert ablation.count_as_real_bugs is False
    assert adaptive_reference.count_as_real_bugs is False
    assert all(command.count_as_real_bugs is False for command in adaptive_contrasts.values())
    assert comparison.count_as_real_bugs is False
    assert "no_type_aware" in _flag_value(ablation.command, "--presets")
    assert "no_normalizer" in _flag_value(ablation.command, "--presets")
    assert "no_feedback" in _flag_value(ablation.command, "--presets")
    assert "oracle_only_metamorphic" in _flag_value(ablation.command, "--presets")
    assert "core_datafusion" in _flag_value(ablation.command, "--target-suites")
    assert "core_arrow" in _flag_value(ablation.command, "--target-suites")
    assert "embedded_sql" in _flag_value(comparison.command, "--target-suites")
    assert "latest_all_engines" in _flag_value(comparison.command, "--target-suites")
    assert "discovery_guided" in _flag_value(comparison.command, "--presets")
    assert "live_cross_family" in _flag_value(comparison.command, "--presets")
    assert _flag_value(ablation.command, "--evidence-mode") == "ablation"
    assert _flag_value(adaptive_reference.command, "--schedule") == "adaptive"
    for component, command in adaptive_contrasts.items():
        assert _flag_value(command.command, "--schedule") == "adaptive"
        assert (
            _flag_value(command.command, "--disable-adaptive-components")
            == ADAPTIVE_COMPONENT_DISABLE_FLAGS[component]
        )
    assert "--disable-adaptive-components" not in adaptive_reference.command
    assert _flag_value(comparison.command, "--evidence-mode") == "comparison"
    assert ablation.replay_bug_policy["enable_replay_bug"] is False
    assert adaptive_reference.replay_bug_policy["enable_replay_bug"] is False
    assert all(
        command.replay_bug_policy["enable_replay_bug"] is False
        for command in adaptive_contrasts.values()
    )
    assert comparison.replay_bug_policy["enable_replay_bug"] is False


def test_comparison_matrix_variants_are_explicitly_marked_as_contrast_or_baseline():
    module = _module()

    roles = {
        variant.preset: variant.comparison_role
        for variant in module.FINAL_COMPARISON_MATRIX.variants
    }

    assert roles["baseline"] == "baseline"
    assert roles["guided"] == "contrast"
    assert roles["discovery"] == "contrast"
    assert roles["discovery_guided"] == "contrast"
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

    for component in ADAPTIVE_COMPONENT_DISABLE_FLAGS:
        command = by_name[f"adaptive_component_ablation:no_{component}"]
        adaptive_meta = json.loads(_flag_value(command.command, "--experiment-meta"))
        assert adaptive_meta["matrix_id"] == "adaptive_component_ablation"
        assert adaptive_meta["comparison_group"] == "adaptive_component_ablation"
        assert adaptive_meta["variant"]["variant_id"] == f"no_{component}"
        assert adaptive_meta["variant"]["component_focus"] == component
        assert adaptive_meta["variant"]["factors"] == {component: False}

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


def test_final_plan_can_append_cross_version_ledger_evidence_command():
    module = _module()

    commands = module.build_plan(
        _args(
            track="comparison",
            ledger_run_files="runs/v1.jsonl,runs/v2.jsonl",
            ledger_versions="v1,v2",
            previous_ledger="reports/previous-ledger.json",
            ledger_output="reports/final-ledger.json",
            ledger_evidence_manifest="reports/experiment-final-ledger.json",
        )
    )
    by_name = {command.name: command for command in commands}
    command = by_name["cross_version_regression_ledger"]

    assert _flag_value(command.command, "--run-files") == "runs/v1.jsonl,runs/v2.jsonl"
    assert _flag_value(command.command, "--versions") == "v1,v2"
    assert _flag_value(command.command, "--previous-ledger") == "reports/previous-ledger.json"
    assert _flag_value(command.command, "--output") == "reports/final-ledger.json"
    assert _flag_value(command.command, "--evidence-manifest-output") == "reports/experiment-final-ledger.json"
    assert command.count_as_real_bugs is False
    assert command.experiment_meta["comparison_group"] == "cross_version_continual_learning"


def test_final_plan_adds_manifest_index_driven_cross_version_ledger_by_default():
    module = _module()

    commands = module.build_plan(_args(track="comparison"))
    by_name = {command.name: command for command in commands}
    command = by_name["cross_version_regression_ledger"]

    assert "--run-files" not in command.command
    assert _flag_value(command.command, "--manifest-index") == "reports/final-experiment-manifest-index.json"
    assert _flag_value(command.command, "--output") == "reports/final-version-ledger.json"
    assert _flag_value(command.command, "--evidence-manifest-output") == (
        "reports/experiment-final-version-ledger.json"
    )


def test_final_plan_postprocess_readiness_audit_includes_extra_ledger_manifest():
    module = _module()

    commands = module.build_plan(
        _args(
            track="all",
            ledger_run_files="runs/v1.jsonl,runs/v2.jsonl",
            ledger_evidence_manifest="reports/experiment-final-ledger.json",
        )
    )
    by_name = {command.name: command for command in commands}
    audit = by_name["final_readiness_audit"]

    assert audit.track == "postprocess"
    assert audit.count_as_real_bugs is False
    assert _flag_value(audit.command, "--manifest-index") == "reports/final-experiment-manifest-index.json"
    assert _flag_value(audit.command, "--extra-manifest") == "reports/experiment-final-ledger.json"
    assert "--all-manifests" not in audit.command
    assert "--full-run-log-scan" in audit.command
    assert "--fail-on-missing" in audit.command
    assert audit.experiment_meta["matrix_id"] == "final_readiness_audit"


def test_final_plan_postprocess_readiness_audit_can_run_as_single_track():
    module = _module()

    commands = module.build_plan(_args(track="postprocess"))

    assert [command.name for command in commands] == ["final_readiness_audit"]
    assert _flag_value(commands[0].command, "--manifest-index") == "reports/final-experiment-manifest-index.json"
    assert "--all-manifests" not in commands[0].command
    assert _flag_value(commands[0].command, "--extra-manifest") == (
        "reports/experiment-final-version-ledger.json"
    )


def test_final_plan_manifest_index_records_executed_command_evidence(tmp_path):
    module = _module()
    index = tmp_path / "final-index.json"
    plan = tmp_path / "plan.json"
    args = _args(manifest_index=str(index))
    command = module.FinalCommand(
        track="validation",
        name="short_validation_smoke",
        command=["datadiff", "experiment"],
        purpose="test",
        count_as_real_bugs=False,
        expected_output="manifest",
    )
    observed = {"returncode": 0}
    module._ingest_command_evidence_line(observed, "experiment manifest: runs/experiment-final.json\n")
    module._ingest_command_evidence_line(observed, "evidence_manifest=reports/experiment-ledger.json\n")

    module._write_initial_manifest_index(index, plan_path=plan, args=args, commands=[command])
    module._append_manifest_index_command(index, command, observed)

    payload = json.loads(index.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "final-experiment-manifest-index-v1"
    assert payload["manifest_files"] == ["runs/experiment-final.json"]
    assert payload["extra_manifest_files"] == ["reports/experiment-ledger.json"]
    assert payload["commands"][0]["status"] == "completed"
    assert payload["commands"][0]["manifest_files"] == ["runs/experiment-final.json"]
    assert payload["commands"][0]["extra_manifest_files"] == ["reports/experiment-ledger.json"]


def test_final_plan_execute_appends_existing_manifest_index_by_default(tmp_path, monkeypatch):
    module = _module()
    index = tmp_path / "final-index.json"
    index.write_text(
        json.dumps(
            {
                "schema_version": "final-experiment-manifest-index-v1",
                "manifest_files": ["runs/existing.json"],
                "extra_manifest_files": [],
                "commands": [],
            }
        ),
        encoding="utf-8",
    )
    command = module.FinalCommand(
        track="validation",
        name="short_validation_smoke",
        command=["datadiff", "experiment"],
        purpose="test",
        count_as_real_bugs=False,
        expected_output="manifest",
    )
    args = _args(track="validation", execute=True, manifest_index=str(index))

    monkeypatch.setattr(module, "build_plan", lambda parsed_args: [command])
    monkeypatch.setattr(module, "write_plan", lambda commands, parsed_args: tmp_path / "plan.json")
    monkeypatch.setattr(
        module,
        "_execute_command_with_manifest_capture",
        lambda item: {
            "returncode": 0,
            "manifest_files": ["runs/new.json"],
            "extra_manifest_files": [],
            "final_readiness_files": [],
        },
    )

    assert module.run_with_args(args) == 0
    payload = json.loads(index.read_text(encoding="utf-8"))
    assert payload["manifest_files"] == ["runs/existing.json", "runs/new.json"]
    assert payload["commands"][0]["name"] == "short_validation_smoke"


def test_final_plan_imports_existing_evidence_into_manifest_index_and_journal(tmp_path):
    module = _module()
    reports_dir = tmp_path / "reports"
    runs_dir = tmp_path / "runs"
    reports_dir.mkdir()
    runs_dir.mkdir()
    run_file = runs_dir / "run-live.jsonl"
    run_file.write_text("", encoding="utf-8")
    (runs_dir / "run-live.meta.json").write_text(
        json.dumps(
            {
                "target_suite": "datafusion_cross",
                "preset": "live_datafusion",
                "seed": 1,
                "executed_cases": 1,
                "throughput_cases_s": 1.0,
                "experiment_meta": {"matrix_id": "live_discovery"},
            }
        ),
        encoding="utf-8",
    )
    manifest = runs_dir / "experiment-live.json"
    manifest.write_text(
        json.dumps(
            {
                "experiment_meta": {"matrix_id": "live_discovery"},
                "evidence_mode": "live",
                "run_theme": "imported-live",
                "runs": [
                    {
                        "run_file": str(run_file),
                        "target_suite": "datafusion_cross",
                        "preset": "live_datafusion",
                        "seed": 1,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    extra_manifest = reports_dir / "experiment-final-version-ledger.json"
    extra_manifest.write_text(
        json.dumps({"schema_version": "version-ledger-evidence-manifest-v1", "version_ledger_file": "reports/ledger.json"}),
        encoding="utf-8",
    )
    index = reports_dir / "final-index.json"
    args = _args(
        track="postprocess",
        manifest_index=str(index),
        import_manifest=[str(manifest)],
        import_extra_manifest=[str(extra_manifest)],
        paper_run_journal=str(reports_dir / "paper-run-journal.jsonl"),
    )

    rc = module.run_with_args(args)

    assert rc == 0
    payload = json.loads(index.read_text(encoding="utf-8"))
    assert payload["manifest_files"] == [str(manifest)]
    assert payload["extra_manifest_files"] == [str(extra_manifest)]
    assert payload["commands"][0]["name"] == "import_existing_evidence"
    assert payload["commands"][0]["paper_run_journal_files"] == [str(reports_dir / "paper-run-journal.jsonl")]
    journal = (reports_dir / "paper-run-journal.jsonl").read_text(encoding="utf-8")
    assert str(run_file) in journal


def test_final_plan_execute_fails_successful_experiment_without_manifest(tmp_path, monkeypatch):
    module = _module()
    index = tmp_path / "final-index.json"
    command = module.FinalCommand(
        track="validation",
        name="short_validation_smoke",
        command=["datadiff", "experiment"],
        purpose="test",
        count_as_real_bugs=False,
        expected_output="manifest",
    )
    args = _args(track="validation", execute=True, manifest_index=str(index))

    monkeypatch.setattr(module, "build_plan", lambda parsed_args: [command])
    monkeypatch.setattr(module, "write_plan", lambda commands, parsed_args: tmp_path / "plan.json")
    monkeypatch.setattr(
        module,
        "_execute_command_with_manifest_capture",
        lambda item: {
            "returncode": 0,
            "manifest_files": [],
            "extra_manifest_files": [],
            "final_readiness_files": [],
        },
    )

    try:
        module.run_with_args(args)
    except subprocess.CalledProcessError as exc:
        assert exc.returncode == 2
    else:
        raise AssertionError("expected missing manifest evidence to fail execute")

    payload = json.loads(index.read_text(encoding="utf-8"))
    assert payload["commands"][0]["status"] == "failed"
    assert payload["commands"][0]["evidence_issues"] == ["missing_experiment_manifest"]


def test_final_plan_execute_fails_successful_final_readiness_without_json(tmp_path, monkeypatch):
    module = _module()
    command = module.FinalCommand(
        track="postprocess",
        name="final_readiness_audit",
        command=["datadiff", "final-readiness"],
        purpose="test",
        count_as_real_bugs=False,
        expected_output="readiness json",
    )
    args = _args(track="postprocess", execute=True)

    monkeypatch.setattr(module, "build_plan", lambda parsed_args: [command])
    monkeypatch.setattr(module, "write_plan", lambda commands, parsed_args: tmp_path / "plan.json")
    monkeypatch.setattr(
        module,
        "_execute_command_with_manifest_capture",
        lambda item: {
            "returncode": 0,
            "manifest_files": [],
            "extra_manifest_files": [],
            "final_readiness_files": [],
        },
    )

    try:
        module.run_with_args(args)
    except subprocess.CalledProcessError as exc:
        assert exc.returncode == 2
    else:
        raise AssertionError("expected missing final-readiness evidence to fail execute")


def test_final_plan_postprocess_execute_refuses_incomplete_manifest_index(
    tmp_path, monkeypatch, capsys
):
    module = _module()
    index = tmp_path / "final-index.json"
    manifest = tmp_path / "validation.json"
    manifest.write_text(
        json.dumps({"experiment_meta": {"matrix_id": "final_validation"}, "runs": []}),
        encoding="utf-8",
    )
    index.write_text(
        json.dumps(
            {
                "schema_version": "final-experiment-manifest-index-v1",
                "manifest_files": [str(manifest)],
                "extra_manifest_files": [],
                "commands": [{"name": "short_validation_smoke", "status": "completed"}],
            }
        ),
        encoding="utf-8",
    )
    command = module.FinalCommand(
        track="postprocess",
        name="final_readiness_audit",
        command=["datadiff", "final-readiness", "--manifest-index", str(index)],
        purpose="test",
        count_as_real_bugs=False,
        expected_output="readiness json",
    )
    args = _args(track="postprocess", execute=True, manifest_index=str(index))

    monkeypatch.setattr(module, "build_plan", lambda parsed_args: [command])
    monkeypatch.setattr(module, "write_plan", lambda commands, parsed_args: tmp_path / "plan.json")

    def fail_execute(item):
        raise AssertionError("postprocess command should not execute with incomplete evidence")

    monkeypatch.setattr(module, "_execute_command_with_manifest_capture", fail_execute)

    assert module.run_with_args(args) == 2
    err = capsys.readouterr().err
    assert "manifest index is incomplete" in err
    assert "missing_required_matrix_ids:" in err
    assert "adaptive_component_ablation" in err
    assert "baseline_scope_comparison" in err
    assert "unrecorded_version_ledger_evidence_manifest" in err


def test_final_plan_postprocess_preflight_accepts_complete_manifest_index(tmp_path):
    module = _module()
    index = tmp_path / "final-index.json"
    manifests = []
    for matrix_id in module.FINAL_REQUIRED_MATRIX_IDS:
        manifest = tmp_path / f"{matrix_id}.json"
        manifest.write_text(
            json.dumps(
                {
                    "experiment_meta": {"matrix_id": matrix_id},
                    "runs": [{"matrix_id": matrix_id}],
                }
            ),
            encoding="utf-8",
        )
        manifests.append(str(manifest))
    ledger = tmp_path / "final-version-ledger.json"
    ledger.write_text(
        json.dumps(
            {
                "schema_version": "version-ledger-v1",
                "summary": {"version_count": 2, "family_count": 1},
                "health": {
                    "schema_version": "version-ledger-health-v1",
                    "health_observation_count": 2,
                },
                "health_feedback_report": {
                    "schema_version": "version-ledger-health-feedback-report-v1",
                    "health_observation_count": 2,
                },
            }
        ),
        encoding="utf-8",
    )
    ledger_manifest = tmp_path / "experiment-final-version-ledger.json"
    ledger_manifest.write_text(
        json.dumps(
            {
                "schema_version": "version-ledger-evidence-manifest-v1",
                "version_ledger_file": str(ledger),
                "runs": [{"version_ledger_file": str(ledger)}],
            }
        ),
        encoding="utf-8",
    )
    index.write_text(
        json.dumps(
            {
                "schema_version": "final-experiment-manifest-index-v1",
                "manifest_files": manifests,
                "extra_manifest_files": [str(ledger_manifest)],
                "commands": [{"name": "all-tracks", "status": "completed"}],
            }
        ),
        encoding="utf-8",
    )

    issues = module._postprocess_evidence_preflight_issues(
        index,
        args=_args(
            track="postprocess",
            manifest_index=str(index),
            ledger_evidence_manifest=str(ledger_manifest),
        ),
    )

    assert issues == []


def test_final_plan_imported_evidence_satisfies_postprocess_preflight(tmp_path):
    module = _module()
    reports_dir = tmp_path / "reports"
    runs_dir = tmp_path / "runs"
    reports_dir.mkdir()
    runs_dir.mkdir()

    run_file = runs_dir / "run-live.jsonl"
    run_file.write_text("", encoding="utf-8")
    (runs_dir / "run-live.meta.json").write_text(
        json.dumps(
            {
                "target_suite": "datafusion_cross",
                "preset": "live_datafusion",
                "seed": 1,
                "executed_cases": 1,
                "throughput_cases_s": 1.0,
                "experiment_meta": {"matrix_id": "live_discovery"},
            }
        ),
        encoding="utf-8",
    )

    manifests = []
    for matrix_id in module.FINAL_REQUIRED_MATRIX_IDS:
        manifest = runs_dir / f"{matrix_id}.json"
        manifest.write_text(
            json.dumps(
                {
                    "experiment_meta": {"matrix_id": matrix_id},
                    "evidence_mode": "live" if matrix_id == "live_discovery" else "comparison",
                    "runs": [
                        {
                            "run_file": str(run_file),
                            "target_suite": "datafusion_cross",
                            "preset": "live_datafusion",
                            "seed": 1,
                            "matrix_id": matrix_id,
                            "experiment_meta": {"matrix_id": matrix_id},
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        manifests.append(manifest)

    ledger = reports_dir / "final-version-ledger.json"
    ledger.write_text(
        json.dumps(
            {
                "schema_version": "version-ledger-v1",
                "summary": {"version_count": 2, "family_count": 1},
                "health": {
                    "schema_version": "version-ledger-health-v1",
                    "health_observation_count": 2,
                },
                "health_feedback_report": {
                    "schema_version": "version-ledger-health-feedback-report-v1",
                    "health_observation_count": 2,
                },
            }
        ),
        encoding="utf-8",
    )
    ledger_manifest = reports_dir / "experiment-final-version-ledger.json"
    ledger_manifest.write_text(
        json.dumps(
            {
                "schema_version": "version-ledger-evidence-manifest-v1",
                "version_ledger_file": str(ledger),
                "runs": [{"version_ledger_file": str(ledger)}],
            }
        ),
        encoding="utf-8",
    )

    index = reports_dir / "final-index.json"
    import_args = _args(
        track="postprocess",
        manifest_index=str(index),
        import_manifest=[str(path) for path in manifests],
        import_extra_manifest=[str(ledger_manifest)],
        paper_run_journal=str(reports_dir / "paper-run-journal.jsonl"),
    )

    assert module.run_with_args(import_args) == 0
    issues = module._postprocess_evidence_preflight_issues(
        index,
        args=_args(
            track="postprocess",
            manifest_index=str(index),
            ledger_evidence_manifest=str(ledger_manifest),
        ),
    )

    assert issues == []


def test_final_plan_propagates_continual_learning_ledgers_to_adaptive_runs():
    module = _module()

    commands = module.build_plan(
        _args(
            track="all",
            live_batch_duration="15m",
            adaptive_learning_weight=0.9,
            continual_learning_ledgers="reports/ledger-a.json,reports/ledger-b.json",
        )
    )
    adaptive_commands = [
        command
        for command in commands
        if "--schedule" in command.command and _flag_value(command.command, "--schedule") == "adaptive"
    ]

    assert adaptive_commands
    assert any(command.track == "live" for command in adaptive_commands)
    assert any(command.name.startswith("adaptive_component_ablation:") for command in adaptive_commands)
    assert all(
        _flag_value(command.command, "--continual-learning-ledgers")
        == "reports/ledger-a.json,reports/ledger-b.json"
        for command in adaptive_commands
    )
    assert all(_flag_value(command.command, "--adaptive-learning-weight") == "0.9" for command in adaptive_commands)
    assert all(
        _flag_value(command.command, "--scheduler-annealing-temperature") == "0.35"
        for command in adaptive_commands
    )
    assert all(
        _flag_value(command.command, "--scheduler-annealing-decay") == "0.985"
        for command in adaptive_commands
    )
    assert all(
        _flag_value(command.command, "--scheduler-annealing-min-temperature") == "0.02"
        for command in adaptive_commands
    )
    assert all(
        _flag_value(command.command, "--batch-duration") == "15m"
        for command in adaptive_commands
        if command.track == "live"
    )


def test_final_plan_freezes_dynamic_strategy_snapshot():
    module = _module()

    commands = module.build_plan(_args(track="validation", reset_strategy_snapshot=True))

    assert len(commands) == 1
    command = commands[0]
    snapshot_path = _flag_value(command.command, "--strategy-snapshot")
    assert snapshot_path
    assert snapshot_path.endswith("reports/strategy-snapshots/final-frozen-strategy-snapshot.json")
    assert "--freeze-strategy-snapshot" in command.command
    assert Path(snapshot_path).is_file()

    comparison = module.build_plan(_args(track="comparison"))[0]
    assert _flag_value(comparison.command, "--strategy-snapshot") == snapshot_path


def test_final_plan_respects_explicit_strategy_snapshot():
    module = _module()

    commands = module.build_plan(_args(track="validation", strategy_snapshot="reports/custom-strategy.json"))

    assert _flag_value(commands[0].command, "--strategy-snapshot") == "reports/custom-strategy.json"


def test_final_plan_explicitly_separates_fresh_and_replay_policy():
    module = _module()

    commands = module.build_plan(_args())
    live = [command for command in commands if command.track == "live"]
    historical = [command for command in commands if command.track == "historical"]
    support = [command for command in commands if command.track in {"validation", "ablation", "comparison"}]
    experiment_support = [
        command
        for command in support
        if command.name != "cross_version_regression_ledger"
    ]

    assert live
    assert all("--enable-replay-bug" not in command.command for command in live)
    assert all("--replay-bug-source-issues" in command.command for command in live)
    assert all(command.replay_bug_policy["enable_replay_bug"] is False for command in live)
    assert support
    assert all("--enable-replay-bug" not in command.command for command in support)
    assert all("--replay-bug-source-issues" in command.command for command in experiment_support)
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
