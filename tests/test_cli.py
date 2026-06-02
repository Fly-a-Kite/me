import json
import os
from pathlib import Path

import pytest

from datadiff import cli
from datadiff.config import DiscoveryBias
from datadiff.cli import _experiment_target_runs, _preset_config, build_parser
from datadiff.preset_catalog import build_experiment_config
from datadiff.preset_catalog import PRESET_CATALOG
from datadiff.dsl import Case, ColumnSpec, Program, TableData
from datadiff.util import append_jsonl, closed_loop_state_path, dump_json, run_meta_path


def test_cli_parses_fuzz_ablation_flags():
    parser = build_parser()
    args = parser.parse_args(
        [
            "fuzz",
            "--cases",
            "10",
            "--seed",
            "5",
            "--duration",
            "10s",
            "--profile",
            "edge_float",
            "--disable-normalizer",
            "--disable-feedback",
            "--enable-replay-bug",
            "--disable-preflight-repair",
            "--persist-feedback-corpus",
            "--feedback-persist-limit",
            "12",
            "--enable-local-source-scheduler",
            "--local-source-exploration-weight",
            "0.25",
            "--metamorphic-variant-limit",
            "9",
            "--log-level",
            "minimal",
        ]
    )
    assert args.cmd == "fuzz"
    assert args.cases == 10
    assert args.seed == 5
    assert args.duration == "10s"
    assert args.profile == "edge_float"
    assert args.disable_normalizer is True
    assert args.disable_feedback is True
    assert args.enable_replay_bug is True
    assert args.disable_preflight_repair is True
    assert args.persist_feedback_corpus is True
    assert args.feedback_persist_limit == 12
    assert args.enable_local_source_scheduler is True
    assert args.local_source_exploration_weight == 0.25
    assert args.metamorphic_variant_limit == 9
    assert args.log_level == "minimal"
    assert args.no_compress_run_log is False


def test_cli_parses_no_compress_run_log():
    parser = build_parser()
    args = parser.parse_args(["fuzz", "--no-compress-run-log"])
    assert args.cmd == "fuzz"
    assert args.no_compress_run_log is True


def test_cli_parses_bug_audit_options():
    parser = build_parser()
    args = parser.parse_args(
        [
            "bug-audit",
            "--probes",
            "polars_reflected_arithmetic,pyarrow_sliced_bool_groupby",
            "--write-issues",
            "--issue-dir",
            "new_issue/generated",
            "--overwrite-issues",
            "--fail-on-candidate",
        ]
    )
    assert args.cmd == "bug-audit"
    assert args.probes == "polars_reflected_arithmetic,pyarrow_sliced_bool_groupby"
    assert args.write_issues is True
    assert args.issue_dir == "new_issue/generated"
    assert args.overwrite_issues is True
    assert args.fail_on_candidate is True


def test_cli_parses_bug_hunt_options():
    parser = build_parser()
    args = parser.parse_args(
        [
            "bug-hunt",
            "--cases",
            "10",
            "--seed",
            "5",
            "--target-suite",
            "latest_no_datafusion",
            "--preset",
            "live_arrow_issue_focus",
            "--no-write-issues",
            "--no-overwrite-issues",
            "--skip-run-report",
            "--fail-on-fresh-candidate",
        ]
    )
    assert args.cmd == "bug-hunt"
    assert args.cases == 10
    assert args.seed == 5
    assert args.target_suite == "latest_no_datafusion"
    assert args.preset == "live_arrow_issue_focus"
    assert args.write_issues is False
    assert args.overwrite_issues is False
    assert args.skip_run_report is True
    assert args.fail_on_fresh_candidate is True


def test_cli_parses_bug_sprint_options():
    parser = build_parser()
    args = parser.parse_args(
        [
            "bug-sprint",
            "--cases",
            "12",
            "--seeds",
            "3,5",
            "--lanes",
            "arrow_layout,datafusion_optimizer",
            "--list-lanes",
            "--json",
            "--no-write-issues",
            "--no-overwrite-issues",
            "--skip-run-report",
            "--fail-on-fresh-candidate",
            "--watch-health",
        ]
    )
    assert args.cmd == "bug-sprint"
    assert args.cases == 12
    assert args.seeds == "3,5"
    assert args.lanes == "arrow_layout,datafusion_optimizer"
    assert args.list_lanes is True
    assert args.json is True
    assert args.write_issues is False
    assert args.overwrite_issues is False
    assert args.skip_run_report is True
    assert args.fail_on_fresh_candidate is True
    assert args.watch_health is True


def test_cli_bug_hunt_writes_integrated_manifest(tmp_path, monkeypatch):
    from datadiff.bug_audit import BugAuditRun

    run_file = tmp_path / "run.jsonl"
    report_file = tmp_path / "report.md"
    csv_file = tmp_path / "report.csv"
    manifest_file = tmp_path / "hunt.json"
    issue_file = tmp_path / "issue.md"

    def fake_run_bug_audit(*, probe_ids=None, output_dir=None):
        assert probe_ids is None
        return BugAuditRun(
            generated_at="2026-05-26T12:00:00Z",
            evidence_mode="deterministic_probe",
            methodology="test",
            environment={},
            results=[],
            candidate_bug_families=["audit_family@pyarrow"],
            output_json=str(tmp_path / "audit.json"),
            output_markdown=str(tmp_path / "audit.md"),
        )

    def fake_write_issue_drafts(run, *, issue_dir=None, overwrite=False):
        assert overwrite is True
        return [issue_file]

    def fake_run_fuzz(*, cases, seed, backends, config, duration_s, **kwargs):
        assert cases == 7
        assert seed == 9
        assert backends
        assert config.generator_profile == "bughunt_fresh"
        assert "pyarrow_sliced_bool_groupby_any_all@pyarrow" in config.known_saturated_bug_families
        assert duration_s is None
        append_jsonl(
            {
                "case": {"case_id": "case-fresh", "seed": 9},
                "config": config.to_dict(),
                "findings": [
                    {
                        "root_cause": "new_family",
                        "suspicious_backends": ["polars"],
                        "triage_verdict": "candidate_implementation_bug",
                        "false_positive": False,
                    }
                ],
                "normalized": {"polars": {"status": "ok", "rows": [[1]]}},
            },
            run_file,
        )
        return run_file

    monkeypatch.setattr(cli, "run_probe_audit", fake_run_bug_audit)
    monkeypatch.setattr(cli, "write_probe_issue_drafts", fake_write_issue_drafts)
    monkeypatch.setattr(cli, "run_fuzz", fake_run_fuzz)
    monkeypatch.setattr(cli, "write_report", lambda path: (report_file, csv_file))
    monkeypatch.setattr(
        cli,
        "build_candidate_pipeline",
        lambda **kwargs: {
            "manifest_path": "new_issue/generated/candidate-pipelines/pipeline-test/manifest.json",
            "markdown_path": "new_issue/generated/candidate-pipelines/pipeline-test/manifest.md",
            "summary": {"candidate_count": 1, "reproduced_count": 1, "reduced_count": 1},
        },
    )
    monkeypatch.setattr(
        cli,
        "_summarize_run_classification",
        lambda path, limit, refresh: {
            "run_file": str(path),
            "refresh": refresh,
            "triage_verdicts": {"candidate_implementation_bug": 1},
            "candidate_bug_families": {"new_family@polars": 1},
            "fresh_candidate_bug_families": {"new_family@polars": 1},
            "known_saturated_candidate_bug_families": {},
            "known_saturated_reference_count": 1,
            "false_positive_reasons": {},
            "examples": {},
        },
    )

    parser = build_parser()
    args = parser.parse_args(
        [
            "bug-hunt",
            "--cases",
            "7",
            "--seed",
            "9",
            "--output-manifest",
            str(manifest_file),
        ]
    )

    assert args.func(args) == 0
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "bug-hunt-v1"
    assert manifest["generated_by"] == "datadiff bug-hunt"
    assert manifest["bug_audit"]["candidate_bug_families"] == ["audit_family@pyarrow"]
    assert manifest["classification"]["fresh_candidate_bug_families"] == {"new_family@polars": 1}
    assert manifest["fuzz_run"]["run_file"]
    assert manifest["fuzz_run"]["fresh_candidate_evidence_rows"] == 1
    assert manifest["candidate_pipeline"]["summary"]["candidate_count"] == 1
    evidence = json.loads((tmp_path / "hunt-fresh-candidates.json").read_text(encoding="utf-8"))
    assert evidence["candidate_row_count"] == 1
    assert evidence["candidate_rows"][0]["case"]["case_id"] == "case-fresh"


def test_cli_bug_sprint_writes_targeted_manifest(tmp_path, monkeypatch):
    from datadiff.bug_audit import BugAuditRun

    run_file = tmp_path / "run-sprint.jsonl"
    report_file = tmp_path / "report.md"
    csv_file = tmp_path / "report.csv"
    manifest_file = tmp_path / "sprint.json"
    issue_file = tmp_path / "issue.md"

    def fake_run_bug_audit(*, probe_ids=None, output_dir=None):
        assert probe_ids is None
        return BugAuditRun(
            generated_at="2026-05-26T12:00:00Z",
            evidence_mode="deterministic_probe",
            methodology="test",
            environment={},
            results=[],
            candidate_bug_families=["audit_family@pyarrow"],
            output_json=str(tmp_path / "audit.json"),
            output_markdown=str(tmp_path / "audit.md"),
        )

    def fake_write_issue_drafts(run, *, issue_dir=None, overwrite=False):
        assert overwrite is True
        return [issue_file]

    def fake_run_fuzz(*, cases, seed, backends, config, duration_s, **kwargs):
        assert cases == 7
        assert seed == 9
        assert backends == ["pandas", "duckdb", "pyarrow"]
        assert config.generator_profile == "bughunt_fresh"
        assert config.enable_metamorphic_oracle is True
        assert duration_s is None
        append_jsonl(
            {
                "case": {"case_id": "case-sprint", "seed": seed},
                "config": config.to_dict(),
                "findings": [
                    {
                        "root_cause": "new_arrow_family",
                        "suspicious_backends": ["pyarrow"],
                        "triage_verdict": "candidate_implementation_bug",
                        "false_positive": False,
                    }
                ],
                "normalized": {"pyarrow": {"status": "ok", "rows": [[1]]}},
            },
            run_file,
        )
        return run_file

    monkeypatch.setattr(cli, "run_probe_audit", fake_run_bug_audit)
    monkeypatch.setattr(cli, "write_probe_issue_drafts", fake_write_issue_drafts)
    monkeypatch.setattr(cli, "run_fuzz", fake_run_fuzz)
    monkeypatch.setattr(cli, "write_report", lambda path: (report_file, csv_file))
    monkeypatch.setattr(
        cli,
        "build_candidate_pipeline",
        lambda **kwargs: {
            "manifest_path": "new_issue/generated/candidate-pipelines/pipeline-test/manifest.json",
            "markdown_path": "new_issue/generated/candidate-pipelines/pipeline-test/manifest.md",
            "summary": {"candidate_count": 1, "reproduced_count": 1, "needs_dedup_check_count": 1},
        },
    )
    monkeypatch.setattr(
        cli,
        "_summarize_run_classification",
        lambda path, limit, refresh: {
            "run_file": str(path),
            "refresh": refresh,
            "triage_verdicts": {"candidate_implementation_bug": 1},
            "candidate_bug_families": {"new_arrow_family@pyarrow": 1},
            "fresh_candidate_bug_families": {"new_arrow_family@pyarrow": 1},
            "issue_inspired_unsaturated_candidate_bug_families": {},
            "known_saturated_candidate_bug_families": {},
            "known_saturated_reference_count": 1,
            "false_positive_reasons": {},
            "examples": {},
        },
    )

    parser = build_parser()
    args = parser.parse_args(
        [
            "bug-sprint",
            "--cases",
            "7",
            "--seeds",
            "9",
            "--lanes",
            "arrow_layout",
            "--output-manifest",
            str(manifest_file),
        ]
    )

    assert args.func(args) == 0
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "bug-sprint-v1"
    assert manifest["generated_by"] == "datadiff bug-sprint"
    assert manifest["status"] == "completed"
    assert manifest["bug_audit"]["candidate_bug_families"] == ["audit_family@pyarrow"]
    assert manifest["lane_ids"] == ["arrow_layout"]
    assert manifest["lanes"][0]["preset"] == "live_arrow_deep_organic_metamorphic"
    assert manifest["lanes"][0]["semantic_focus_families"] == [
        "null_semantics",
        "type_coercion",
        "backend_specific_semantics",
    ]
    assert manifest["lanes"][0]["semantic_focus_signals"] == [
        "pyarrow_list_flatten_parent_indices_semantics",
        "pyarrow_hash_pivot_wider_order_semantics",
    ]
    assert manifest["scheduler"]["strategy"] == "adaptive_lane_yield_novelty_false_positive_weighting"
    assert manifest["scheduler"]["lanes"][0]["lane_id"] == "arrow_layout"
    assert manifest["summary"]["fresh_candidate_bug_families"] == {"new_arrow_family@pyarrow": 1}
    assert manifest["summary"]["candidate_pipeline"]["candidate_count"] == 1
    assert manifest["progress"]["planned_run_count"] == 1
    assert manifest["progress"]["completed_run_count"] == 1
    assert manifest["runs"][0]["target_suite"] == "arrow_cross"
    assert manifest["runs"][0]["semantic_focus_families"] == [
        "null_semantics",
        "type_coercion",
        "backend_specific_semantics",
    ]
    assert manifest["runs"][0]["semantic_focus_signals"] == [
        "pyarrow_list_flatten_parent_indices_semantics",
        "pyarrow_hash_pivot_wider_order_semantics",
    ]
    assert manifest["runs"][0]["status"] == "completed"
    assert manifest["runs"][0]["candidate_pipeline"]["summary"]["candidate_count"] == 1
    evidence = json.loads((tmp_path / "sprint-arrow_layout-seed9-fresh-candidates.json").read_text(encoding="utf-8"))
    assert evidence["candidate_row_count"] == 1
    assert evidence["candidate_rows"][0]["case"]["case_id"] == "case-sprint"


def test_cli_bug_sprint_writes_running_manifest_before_lane_executes(tmp_path, monkeypatch):
    run_file = tmp_path / "run-progress.jsonl"
    manifest_file = tmp_path / "sprint-progress.json"

    def fake_run_fuzz(*, cases, seed, backends, config, duration_s, **kwargs):
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        assert manifest["status"] == "running"
        assert manifest["progress"]["planned_run_count"] == 1
        assert manifest["progress"]["completed_run_count"] == 0
        assert manifest["progress"]["current_lane_id"] == "arrow_layout"
        assert manifest["progress"]["current_seed"] == 9
        assert manifest["runs"][0]["status"] == "running"
        append_jsonl(
            {
                "status": "ok",
                "case": {"case_id": "case-progress", "seed": seed},
                "config": config.to_dict(),
                "findings": [],
                "normalized": {"pyarrow": {"status": "ok", "rows": [[1]]}},
            },
            run_file,
        )
        return run_file

    monkeypatch.setattr(cli, "run_fuzz", fake_run_fuzz)
    monkeypatch.setattr(cli, "write_report", lambda path: (tmp_path / "report.md", tmp_path / "report.csv"))
    monkeypatch.setattr(
        cli,
        "build_candidate_pipeline",
        lambda **kwargs: {
            "manifest_path": "new_issue/generated/candidate-pipelines/pipeline-watch/manifest.json",
            "markdown_path": "new_issue/generated/candidate-pipelines/pipeline-watch/manifest.md",
            "summary": {"candidate_count": 1, "reproduced_count": 1},
        },
    )
    monkeypatch.setattr(
        cli,
        "_summarize_run_classification",
        lambda path, limit, refresh: {
            "run_file": str(path),
            "refresh": refresh,
            "triage_verdicts": {},
            "candidate_bug_families": {},
            "fresh_candidate_bug_families": {},
            "issue_inspired_unsaturated_candidate_bug_families": {},
            "known_saturated_candidate_bug_families": {},
            "known_saturated_reference_count": 1,
            "false_positive_reasons": {},
            "examples": {},
        },
    )

    args = build_parser().parse_args(
        [
            "bug-sprint",
            "--skip-bug-audit",
            "--cases",
            "7",
            "--seeds",
            "9",
            "--lanes",
            "arrow_layout",
            "--output-manifest",
            str(manifest_file),
        ]
    )

    assert args.func(args) == 0
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["progress"]["completed_run_count"] == 1
    assert manifest["runs"][0]["status"] == "completed"


def test_cli_bug_sprint_watch_health_stops_remaining_lanes(tmp_path, monkeypatch):
    run_file = tmp_path / "run-watch.jsonl"
    manifest_file = tmp_path / "sprint-watch.json"
    calls = []

    def fake_run_fuzz(*, cases, seed, backends, config, duration_s, **kwargs):
        calls.append((seed, tuple(backends)))
        append_jsonl(
            {
                "status": "bug",
                "case": {"case_id": f"case-watch-{seed}", "seed": seed},
                "config": config.to_dict(),
                "findings": [
                    {
                        "kind": "semantic_output_mismatch",
                        "root_cause": "fresh_family",
                        "triage_verdict": "candidate_implementation_bug",
                        "suspicious_backends": ["pyarrow"],
                        "false_positive": False,
                    }
                ],
                "normalized": {"pyarrow": {"status": "ok", "rows": [[1]]}},
            },
            run_file,
        )
        return run_file

    monkeypatch.setattr(cli, "run_fuzz", fake_run_fuzz)
    monkeypatch.setattr(cli, "write_report", lambda path: (tmp_path / "report.md", tmp_path / "report.csv"))
    monkeypatch.setattr(
        cli,
        "_summarize_run_classification",
        lambda path, limit, refresh: {
            "run_file": str(path),
            "refresh": refresh,
            "triage_verdicts": {"candidate_implementation_bug": 1},
            "candidate_bug_families": {"fresh_family@pyarrow": 1},
            "fresh_candidate_bug_families": {"fresh_family@pyarrow": 1},
            "issue_inspired_unsaturated_candidate_bug_families": {},
            "known_saturated_candidate_bug_families": {},
            "known_saturated_reference_count": 1,
            "false_positive_reasons": {},
            "examples": {},
        },
    )

    parser = build_parser()
    args = parser.parse_args(
        [
            "bug-sprint",
            "--skip-bug-audit",
            "--cases",
            "7",
            "--seeds",
            "9,10",
            "--lanes",
            "arrow_layout,datafusion_optimizer",
            "--output-manifest",
            str(manifest_file),
            "--watch-health",
        ]
    )

    assert args.func(args) == 0
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert len(calls) == 1
    assert manifest["status"] == "completed"
    assert manifest["stopped_by_health"] is True
    assert manifest["health_stop_reason"] == "fresh_candidate"
    assert manifest["summary"]["run_count"] == 1
    assert manifest["progress"]["planned_run_count"] == 4
    assert manifest["progress"]["completed_run_count"] == 1
    assert manifest["runs"][0]["health"]["statuses"] == {"bug": 1}
    assert manifest["runs"][0]["health"]["fresh_candidate_bug_families"] == {"fresh_family@pyarrow": 1}


def test_cli_bug_sprint_lists_lanes_as_json(capsys):
    parser = build_parser()
    args = parser.parse_args(["bug-sprint", "--list-lanes", "--json"])

    assert args.func(args) == 0
    catalog = json.loads(capsys.readouterr().out)
    assert catalog["arrow_layout"]["default"] is True
    assert catalog["arrow_layout"]["preset"] == "live_arrow_deep_organic_metamorphic"
    assert catalog["common_api_workflow"]["default"] is True
    assert catalog["common_api_workflow"]["preset"] == "live_common_api_workflow_metamorphic"
    assert catalog["datafusion_common_api"]["default"] is True
    assert catalog["datafusion_common_api"]["target_suite"] == "datafusion_cross"
    assert catalog["datafusion_common_api"]["preset"] == "live_datafusion_common_api_metamorphic"
    assert catalog["duckdb_storage"]["default"] is True
    assert catalog["duckdb_storage"]["target_suite"] == "duckdb_storage_cross"
    assert catalog["duckdb_storage"]["preset"] == "live_embedded_sql_deep_organic_metamorphic"
    assert catalog["common_api_workflow"]["discovery_biases"]
    assert catalog["datafusion_optimizer"]["discovery_biases"][0]["keep_in_pool"] is True
    assert catalog["deep_probe_rotation"]["default"] is False
    assert catalog["deep_probe_rotation"]["target_suite"] == "latest_all_engines"
    assert catalog["deep_probe_rotation"]["preset"] == "live_deep_probe_rotation_metamorphic"
    assert catalog["arrow_probe_stress"]["default"] is False


def test_cli_targets_json_exposes_hidden_methodology_and_extension_contract(capsys):
    parser = build_parser()
    args = parser.parse_args(["targets", "--json"])

    assert args.func(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["methodology"]["name"] == "capability_aware_closed_loop_semantic_differential_fuzzing"
    assert "backend_adapter_lowering_boundary" in payload["methodology"]["reusable_layers"]
    datafusion = next(item for item in payload["targets"] if item["name"] == "datafusion")
    assert datafusion["execution_model"] == "arrow_query_engine"
    assert datafusion["adapter_args"] == []
    assert datafusion["adapter_kwargs"] == {}
    assert "implement_backend_adapter" in datafusion["extension_contract"]
    buggy_join = next(item for item in payload["targets"] if item["name"] == "buggy_join")
    assert buggy_join["adapter"] == "datadiff.backends.faulty_backend.FaultyPandasBackend"
    assert buggy_join["adapter_args"] == ["buggy_join", "join"]


def test_cli_parses_bug_sprint_status_command():
    parser = build_parser()
    args = parser.parse_args(
        [
            "bug-sprint-status",
            "--manifest",
            "new_issue/generated/bug-sprint-live.json",
            "--limit",
            "2",
            "--json",
            "--fail-on-fresh-candidate",
            "--fail-on-bug",
        ]
    )
    assert args.cmd == "bug-sprint-status"
    assert args.manifest == "new_issue/generated/bug-sprint-live.json"
    assert args.limit == 2
    assert args.json is True
    assert args.fail_on_fresh_candidate is True
    assert args.fail_on_bug is True


def test_cli_parses_candidate_pipeline_command():
    parser = build_parser()
    args = parser.parse_args(
        [
            "candidate-pipeline",
            "--manifest",
            "new_issue/generated/bug-sprint-live.json",
            "--evidence-files",
            "new_issue/generated/a.json,new_issue/generated/b.json",
            "--output-dir",
            "new_issue/generated/candidate-pipelines",
            "--recheck-attempts",
            "3",
            "--no-reduce",
            "--no-standalone-reproducer",
            "--json",
            "--fail-on-ready",
        ]
    )
    assert args.cmd == "candidate-pipeline"
    assert args.manifest == "new_issue/generated/bug-sprint-live.json"
    assert args.evidence_files == "new_issue/generated/a.json,new_issue/generated/b.json"
    assert args.output_dir == "new_issue/generated/candidate-pipelines"
    assert args.recheck_attempts == 3
    assert args.no_reduce is True
    assert args.no_standalone_reproducer is True
    assert args.json is True
    assert args.fail_on_ready is True


def test_cli_bug_sprint_status_summarizes_running_manifest(tmp_path, monkeypatch, capsys):
    manifest_file = tmp_path / "bug-sprint-running.json"
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    historical_manifest = tmp_path / "bug-sprint-history-manifest.json"
    run_file = runs_dir / "run-live.jsonl.gz"
    append_jsonl(
        {
            "case": {"case_id": "case-live", "seed": 53001},
            "status": "bug",
            "findings": [
                {
                    "kind": "semantic_output_mismatch",
                    "root_cause": "fresh_family",
                    "triage_verdict": "candidate_implementation_bug",
                    "suspicious_backends": ["duckdb"],
                }
            ],
        },
        run_file,
    )
    historical_manifest.write_text(
        json.dumps(
            {
                "schema_version": "bug-sprint-v1",
                "generated_at": "1999-12-31T23:00:00Z",
                "runs": [
                    {
                        "lane_id": "arrow_layout",
                        "status": "completed",
                        "completed_at": "1999-12-31T23:10:00Z",
                        "classification": {
                            "fresh_candidate_bug_families": {"history_family@pyarrow": 2},
                            "false_positive_reasons": {},
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    manifest_file.write_text(
        json.dumps(
                {
                    "schema_version": "bug-sprint-v1",
                    "status": "running",
                    "started_at": "2000-01-01T00:00:00Z",
                "completed_at": "",
                "stopped_by_health": False,
                "health_stop_reason": "",
                "lanes": [
                    {
                        "id": "arrow_layout",
                        "target_suite": "arrow_cross",
                        "preset": "live_arrow_deep_organic_metamorphic",
                        "theme": "Arrow",
                    },
                    {
                        "id": "common_api_workflow",
                        "target_suite": "latest_no_datafusion",
                        "preset": "live_common_api_workflow_metamorphic",
                        "theme": "Common API",
                    },
                ],
                "progress": {
                    "planned_run_count": 4,
                    "completed_run_count": 0,
                    "remaining_run_count": 4,
                    "current_lane_id": "common_api_workflow",
                    "current_seed": 53001,
                },
                "runs": [
                    {
                        "lane_id": "common_api_workflow",
                        "seed": 53001,
                        "status": "running",
                    }
                ],
                "summary": {
                    "run_count": 0,
                    "fresh_candidate_bug_families": {},
                    "issue_inspired_unsaturated_candidate_bug_families": {},
                    "known_saturated_candidate_bug_families": {},
                    "triage_verdicts": {},
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(cli, "RUNS_DIR", runs_dir)

    args = build_parser().parse_args(["bug-sprint-status", "--manifest", str(manifest_file), "--json"])
    assert args.func(args) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["manifest_status"] == "running"
    assert summary["progress"]["planned_run_count"] == 4
    assert summary["progress"]["current_lane_id"] == "common_api_workflow"
    assert summary["current_run"]["status"] == "running"
    assert summary["latest_observed_run_file"].endswith("run-live.jsonl.gz")
    assert summary["latest_observed_run_health"]["statuses"] == {"bug": 1}
    assert summary["latest_observed_run_health"]["fresh_candidate_bug_families"] == {"fresh_family@duckdb": 1}
    lane_summary = {item["lane_id"]: item for item in summary["recent_lane_yield_summary"]}
    assert lane_summary["arrow_layout"]["fresh_candidate_total"] == 2
    assert lane_summary["arrow_layout"]["score"] > lane_summary["common_api_workflow"]["score"]

    args = build_parser().parse_args(
        ["bug-sprint-status", "--manifest", str(manifest_file), "--fail-on-fresh-candidate"]
    )
    assert args.func(args) == 2


def test_cli_parses_artifact_limit():
    parser = build_parser()
    args = parser.parse_args(["longrun", "--artifact-limit", "25"])
    assert args.cmd == "longrun"
    assert args.artifact_limit == 25


def test_cli_parses_guided_fuzz_options():
    parser = build_parser()
    args = parser.parse_args(
        [
            "fuzz",
            "--strategy",
            "guided",
            "--candidate-pool",
            "12",
            "--targets",
            "groupby,nulls",
            "--replay-bug-source-issues",
            "https://github.com/example/project/issues/1",
        ]
    )
    assert args.cmd == "fuzz"
    assert args.strategy == "guided"
    assert args.candidate_pool == 12
    assert args.targets == "groupby,nulls"
    assert args.replay_bug_source_issues == "https://github.com/example/project/issues/1"


def test_cli_parses_workflow_profile():
    parser = build_parser()
    args = parser.parse_args(["fuzz", "--profile", "workflow"])
    assert args.cmd == "fuzz"
    assert args.profile == "workflow"


def test_cli_parses_bughunt_profile():
    parser = build_parser()
    args = parser.parse_args(["fuzz", "--profile", "bughunt"])
    assert args.cmd == "fuzz"
    assert args.profile == "bughunt"


def test_cli_parses_bughunt_fresh_profile():
    parser = build_parser()
    args = parser.parse_args(["fuzz", "--profile", "bughunt_fresh"])
    assert args.cmd == "fuzz"
    assert args.profile == "bughunt_fresh"

    args = parser.parse_args(["longrun", "--profile", "bughunt_fresh"])
    assert args.cmd == "longrun"
    assert args.profile == "bughunt_fresh"


def test_cli_parses_bughunt_no_groupby_profile():
    parser = build_parser()
    args = parser.parse_args(["fuzz", "--profile", "bughunt_no_groupby"])
    assert args.cmd == "fuzz"
    assert args.profile == "bughunt_no_groupby"


def test_cli_parses_pyarrow_groupby_filter_cast_membership_profile():
    parser = build_parser()
    args = parser.parse_args(["fuzz", "--profile", "pyarrow_groupby_filter_cast_membership"])
    assert args.cmd == "fuzz"
    assert args.profile == "pyarrow_groupby_filter_cast_membership"


def test_cli_parses_polars_reverse_division_columns_profile():
    parser = build_parser()
    args = parser.parse_args(["fuzz", "--profile", "polars_reverse_division_columns"])
    assert args.cmd == "fuzz"
    assert args.profile == "polars_reverse_division_columns"


def test_cli_parses_row_value_absence_filter_profile():
    parser = build_parser()
    args = parser.parse_args(["fuzz", "--profile", "row_value_absence_filter"])
    assert args.cmd == "fuzz"
    assert args.profile == "row_value_absence_filter"


def test_cli_parses_null_groupby_topk_profile():
    parser = build_parser()
    args = parser.parse_args(["fuzz", "--profile", "null_groupby_topk"])
    assert args.cmd == "fuzz"
    assert args.profile == "null_groupby_topk"

    args = parser.parse_args(["longrun", "--profile", "null_groupby_topk"])
    assert args.cmd == "longrun"
    assert args.profile == "null_groupby_topk"


def test_cli_parses_null_agg_topk_profile():
    parser = build_parser()
    args = parser.parse_args(["fuzz", "--profile", "null_agg_topk"])
    assert args.cmd == "fuzz"
    assert args.profile == "null_agg_topk"

    args = parser.parse_args(["longrun", "--profile", "null_agg_topk"])
    assert args.cmd == "longrun"
    assert args.profile == "null_agg_topk"


def test_cli_parses_filter_null_agg_topk_profile():
    parser = build_parser()
    args = parser.parse_args(["fuzz", "--profile", "filter_null_agg_topk"])
    assert args.cmd == "fuzz"
    assert args.profile == "filter_null_agg_topk"

    args = parser.parse_args(["longrun", "--profile", "filter_null_agg_topk"])
    assert args.cmd == "longrun"
    assert args.profile == "filter_null_agg_topk"


def test_cli_parses_join_null_agg_topk_profile():
    parser = build_parser()
    args = parser.parse_args(["fuzz", "--profile", "join_null_agg_topk"])
    assert args.cmd == "fuzz"
    assert args.profile == "join_null_agg_topk"

    args = parser.parse_args(["longrun", "--profile", "join_null_agg_topk"])
    assert args.cmd == "longrun"
    assert args.profile == "join_null_agg_topk"


def test_cli_parses_additional_issue_inspired_profiles():
    parser = build_parser()
    for profile in ["join_null_key_topk", "wide_offset_topk", "empty_filter_groupby"]:
        args = parser.parse_args(["fuzz", "--profile", profile])
        assert args.cmd == "fuzz"
        assert args.profile == profile

        args = parser.parse_args(["longrun", "--profile", profile])
        assert args.cmd == "longrun"
        assert args.profile == profile


def test_cli_parses_join_filter_groupby_profile():
    parser = build_parser()
    args = parser.parse_args(["fuzz", "--profile", "join_filter_groupby"])
    assert args.cmd == "fuzz"
    assert args.profile == "join_filter_groupby"

    args = parser.parse_args(["longrun", "--profile", "join_filter_groupby"])
    assert args.cmd == "longrun"
    assert args.profile == "join_filter_groupby"


def test_cli_parses_join_null_truth_filter_profile():
    parser = build_parser()
    args = parser.parse_args(["fuzz", "--profile", "join_null_truth_filter"])
    assert args.cmd == "fuzz"
    assert args.profile == "join_null_truth_filter"

    args = parser.parse_args(["longrun", "--profile", "join_null_truth_filter"])
    assert args.cmd == "longrun"
    assert args.profile == "join_null_truth_filter"


def test_cli_parses_join_groupby_stress_profile():
    parser = build_parser()
    args = parser.parse_args(["fuzz", "--profile", "join_groupby_stress"])
    assert args.cmd == "fuzz"
    assert args.profile == "join_groupby_stress"

    args = parser.parse_args(["longrun", "--profile", "join_groupby_stress"])
    assert args.cmd == "longrun"
    assert args.profile == "join_groupby_stress"


def test_cli_parses_storage_offset_profile():
    parser = build_parser()
    args = parser.parse_args(["fuzz", "--profile", "storage_offset"])
    assert args.cmd == "fuzz"
    assert args.profile == "storage_offset"

    args = parser.parse_args(["longrun", "--profile", "storage_offset"])
    assert args.cmd == "longrun"
    assert args.profile == "storage_offset"


def test_cli_parses_float_group_key_profile():
    parser = build_parser()
    args = parser.parse_args(["fuzz", "--profile", "float_group_key"])
    assert args.cmd == "fuzz"
    assert args.profile == "float_group_key"

    args = parser.parse_args(["longrun", "--profile", "float_group_key"])
    assert args.cmd == "longrun"
    assert args.profile == "float_group_key"


def test_cli_parses_issue_focus_profile():
    parser = build_parser()
    args = parser.parse_args(["fuzz", "--profile", "issue_focus"])
    assert args.cmd == "fuzz"
    assert args.profile == "issue_focus"

    args = parser.parse_args(["longrun", "--profile", "issue_focus"])
    assert args.cmd == "longrun"
    assert args.profile == "issue_focus"

    args = parser.parse_args(["fuzz", "--profile", "deep_probe_rotation"])
    assert args.cmd == "fuzz"
    assert args.profile == "deep_probe_rotation"

    args = parser.parse_args(["longrun", "--profile", "deep_probe_rotation"])
    assert args.cmd == "longrun"
    assert args.profile == "deep_probe_rotation"

    args = parser.parse_args(["fuzz", "--profile", "bool_null_groupby_agg"])
    assert args.cmd == "fuzz"
    assert args.profile == "bool_null_groupby_agg"

    args = parser.parse_args(["fuzz", "--profile", "large_int_filter_groupby"])
    assert args.cmd == "fuzz"
    assert args.profile == "large_int_filter_groupby"


def test_cli_parses_join_null_sort_profile():
    parser = build_parser()
    args = parser.parse_args(["fuzz", "--profile", "join_null_sort"])
    assert args.cmd == "fuzz"
    assert args.profile == "join_null_sort"

    args = parser.parse_args(["longrun", "--profile", "join_null_sort"])
    assert args.cmd == "longrun"
    assert args.profile == "join_null_sort"


def test_cli_parses_order_sensitive_bug_hunt_profiles():
    parser = build_parser()
    for profile in [
        "ordered_groupby_sort",
        "topk_resort",
        "join_ordered_agg_topk",
        "global_null_aggregate",
        "string_count_groupby",
        "unique_count_groupby",
        "set_membership_filter",
        "null_predicate_filter",
        "boolean_predicate_filter",
        "post_topk_range_filter",
        "tuple_absence_filter",
        "running_sum_precision",
        "partitioned_running_sum",
        "path_basename_keyed_pick",
        "sortedness_null_placement",
        "simple_case_random_subject",
        "group_quantile_key_probe",
        "scalar_subquery_double_parentheses",
        "window_avg_rows_frame",
        "struct_distinct_unnest",
        "bit_compare_unequal_length",
        "round_even_float_scale",
        "duckdb_float_literal_precision",
        "polars_timestamp_precision_filter",
        "series_rtruediv_operand_order",
        "pandas_uint64_isin_precision",
        "duckdb_tuple_anti_null_semantics",
        "duckdb_json_predicate_order_semantics",
        "pandas_sparse_array_mask_semantics",
        "polars_float_wrap_numerical_semantics",
        "pandas_index_bool_result_type",
        "polars_empty_literal_groupby_semantics",
        "pandas_arrow_string_eq_sum_semantics",
        "pandas_arrow_timestamp_loc_slice_semantics",
        "pandas_arrow_timestamp_index_attr_semantics",
        "pandas_eval_inplace_aliasing_semantics",
        "pyarrow_dataset_isin_all_match_semantics",
        "pyarrow_large_string_partition_schema_semantics",
        "pyarrow_hash_pivot_wider_order_semantics",
        "pyarrow_list_flatten_parent_indices_semantics",
        "polars_rolling_mean_by_null_count_semantics",
        "csv_long_numeric_roundtrip",
    ]:
        args = parser.parse_args(["fuzz", "--profile", profile])
        assert args.cmd == "fuzz"
        assert args.profile == profile

        args = parser.parse_args(["longrun", "--profile", profile])
        assert args.cmd == "longrun"
        assert args.profile == profile


def test_cli_parses_target_suite():
    parser = build_parser()
    args = parser.parse_args(["fuzz", "--target-suite", "dataframe"])
    assert args.cmd == "fuzz"
    assert args.target_suite == "dataframe"
    assert args.backends is None


def test_cli_parses_targets_command():
    parser = build_parser()
    args = parser.parse_args(["targets"])
    assert args.cmd == "targets"


def test_cli_parses_targets_json_command():
    parser = build_parser()
    args = parser.parse_args(["targets", "--json"])
    assert args.cmd == "targets"
    assert args.json is True


def test_cli_prune_corpus_dry_run_and_yes(tmp_path, monkeypatch, capsys):
    corpus_dir = tmp_path / "corpus"
    interesting = corpus_dir / "interesting"
    interesting.mkdir(parents=True)
    files = []
    for idx in range(3):
        path = interesting / f"{idx}.json"
        path.write_text("{}", encoding="utf-8")
        os.utime(path, (idx + 1, idx + 1))
        files.append(path)
    monkeypatch.setattr(cli, "CORPUS_DIR", corpus_dir)

    parser = build_parser()
    args = parser.parse_args(["prune-corpus", "--keep", "1"])
    assert args.func(args) == 0
    assert all(path.exists() for path in files)
    assert "dry_run=true" in capsys.readouterr().out

    args = parser.parse_args(["prune-corpus", "--keep", "1", "--yes"])
    assert args.func(args) == 0
    remaining = sorted(path.name for path in interesting.glob("*.json"))
    assert remaining == ["2.json"]


def test_cli_parses_experiment_command():
    parser = build_parser()
    args = parser.parse_args(
        [
            "experiment",
            "--cases",
            "5",
            "--seeds",
            "1,2",
            "--presets",
            "baseline,metamorphic",
        ]
    )
    assert args.cmd == "experiment"
    assert args.cases == 5
    assert args.seeds == "1,2"
    assert args.presets == "baseline,metamorphic"
    assert args.no_compress_run_log is False
    assert args.target_suites is None
    assert args.metamorphic_variant_limit is None


def test_cli_parses_multi_target_suite_experiment():
    parser = build_parser()
    args = parser.parse_args(
        [
            "experiment",
            "--cases",
            "5",
            "--target-suites",
            "dataframe,embedded_sql,cross_family",
        ]
    )
    assert args.cmd == "experiment"
    assert args.target_suites == "dataframe,embedded_sql,cross_family"
    assert _experiment_target_runs(args) == [
        ("dataframe", ["pandas", "polars"]),
        ("embedded_sql", ["duckdb", "sqlite"]),
        ("cross_family", ["pandas", "duckdb"]),
    ]


def test_cli_parses_workflow_experiment_preset():
    parser = build_parser()
    args = parser.parse_args(["experiment", "--presets", "workflow"])
    assert args.cmd == "experiment"
    assert args.presets == "workflow"


def test_cli_parses_workflow_metamorphic_experiment_preset():
    parser = build_parser()
    args = parser.parse_args(["experiment", "--presets", "workflow_metamorphic"])
    assert args.cmd == "experiment"
    assert args.presets == "workflow_metamorphic"
    config = _preset_config(args.presets)
    assert config.generator_profile == "workflow"
    assert config.enable_metamorphic_oracle is True
    assert config.oracle_mode == "both"


def test_cli_parses_edge_float_guided_experiment_preset():
    parser = build_parser()
    args = parser.parse_args(["experiment", "--presets", "edge_float_guided"])
    assert args.cmd == "experiment"
    config = _preset_config(args.presets)
    assert config.generator_profile == "edge_float"
    assert config.guidance_strategy == "guided"
    assert config.guidance_candidate_pool == 8
    assert config.guidance_targets == ["edge_float", "numeric", "expressions"]


def test_cli_parses_edge_float_metamorphic_experiment_preset():
    parser = build_parser()
    args = parser.parse_args(["experiment", "--presets", "edge_float_metamorphic"])
    assert args.cmd == "experiment"
    config = _preset_config(args.presets)
    assert config.generator_profile == "edge_float"
    assert config.enable_metamorphic_oracle is True
    assert config.oracle_mode == "both"


def test_cli_catalog_preserves_edge_float_base_and_guided_variants():
    base = _preset_config("edge_float")
    guided = _preset_config("edge_float_guided")
    assert base.generator_profile == "edge_float"
    assert base.guidance_strategy == "random"
    assert guided.generator_profile == base.generator_profile
    assert guided.guidance_strategy == "guided"
    assert guided.guidance_targets == ["edge_float", "numeric", "expressions"]


def test_cli_catalog_does_not_expose_unregistered_profile_overlays():
    with pytest.raises(ValueError, match="unknown experiment preset"):
        _preset_config("storage_offset_metamorphic")
    with pytest.raises(ValueError, match="unknown experiment preset"):
        _preset_config("path_basename_keyed_pick_metamorphic")


def test_cli_parses_targeted_guided_experiment_presets():
    assert _preset_config("guided_filter").guidance_targets == ["filter"]
    assert _preset_config("guided_join").generator_profile == "bughunt_no_groupby"
    assert _preset_config("guided_join").guidance_targets == ["join", "sort_limit"]
    assert _preset_config("guided_mutate").guidance_targets == ["mutate", "expressions"]
    assert _preset_config("null_groupby_topk").generator_profile == "null_groupby_topk"
    assert _preset_config("null_groupby_topk").guidance_targets[0] == "null_groupby_topk"
    assert _preset_config("null_agg_topk").generator_profile == "null_agg_topk"
    assert _preset_config("null_agg_topk").guidance_targets[0] == "null_agg_topk"
    assert _preset_config("null_agg_topk_metamorphic").enable_metamorphic_oracle is True
    assert _preset_config("filter_null_agg_topk").generator_profile == "filter_null_agg_topk"
    assert _preset_config("filter_null_agg_topk").guidance_targets[0] == "filter_null_agg_topk"
    assert _preset_config("filter_null_agg_topk_metamorphic").enable_metamorphic_oracle is True
    assert _preset_config("join_null_agg_topk").generator_profile == "join_null_agg_topk"
    assert _preset_config("join_null_agg_topk").guidance_targets[0] == "join_null_agg_topk"
    assert _preset_config("join_null_agg_topk_metamorphic").enable_metamorphic_oracle is True
    assert _preset_config("join_null_key_topk").generator_profile == "join_null_key_topk"
    assert _preset_config("join_null_key_topk").guidance_targets[0] == "join_null_key_topk"
    assert _preset_config("join_null_key_topk_metamorphic").enable_metamorphic_oracle is True
    assert _preset_config("wide_offset_topk").generator_profile == "wide_offset_topk"
    assert _preset_config("wide_offset_topk").guidance_targets[0] == "wide_offset_topk"
    assert _preset_config("wide_offset_topk_metamorphic").enable_metamorphic_oracle is True
    assert _preset_config("empty_filter_groupby").generator_profile == "empty_filter_groupby"
    assert _preset_config("empty_filter_groupby").guidance_targets[0] == "empty_filter_groupby"
    assert _preset_config("empty_filter_groupby_metamorphic").enable_metamorphic_oracle is True
    assert _preset_config("join_filter_groupby").generator_profile == "join_filter_groupby"
    assert _preset_config("join_filter_groupby").guidance_targets[0] == "join_filter_groupby"
    assert _preset_config("join_filter_groupby_metamorphic").enable_metamorphic_oracle is True
    assert _preset_config("join_null_truth_filter").generator_profile == "join_null_truth_filter"
    assert _preset_config("join_null_truth_filter").guidance_targets[0] == "join_null_truth_filter"
    assert _preset_config("join_null_truth_filter_metamorphic").enable_metamorphic_oracle is True
    assert _preset_config("join_groupby_stress").generator_profile == "join_groupby_stress"
    assert "global_aggregation" in _preset_config("join_groupby_stress").guidance_targets
    assert _preset_config("join_groupby_stress_metamorphic").enable_metamorphic_oracle is True
    assert _preset_config("storage_offset").generator_profile == "storage_offset"
    assert {"sort_offset", "offset"}.issubset(_preset_config("storage_offset").guidance_targets)
    assert _preset_config("float_group_key").generator_profile == "float_group_key"
    assert _preset_config("float_group_key").guidance_targets[0] == "float_group_key"
    assert _preset_config("float_group_key_metamorphic").enable_metamorphic_oracle is True
    assert _preset_config("global_null_aggregate").generator_profile == "global_null_aggregate"
    assert _preset_config("global_null_aggregate").guidance_targets[0] == "global_null_aggregate"
    assert _preset_config("global_null_aggregate_metamorphic").enable_metamorphic_oracle is True
    assert _preset_config("string_count_groupby").generator_profile == "string_count_groupby"
    assert _preset_config("string_count_groupby").guidance_targets[0] == "string_count_groupby"
    assert _preset_config("string_count_groupby_metamorphic").enable_metamorphic_oracle is True
    assert _preset_config("unique_count_groupby").generator_profile == "unique_count_groupby"
    assert _preset_config("unique_count_groupby").guidance_targets[0] == "unique_count_groupby"
    assert _preset_config("unique_count_groupby_metamorphic").enable_metamorphic_oracle is True
    assert _preset_config("set_membership_filter").generator_profile == "set_membership_filter"
    assert _preset_config("set_membership_filter").guidance_targets[0] == "set_membership_filter"
    assert _preset_config("set_membership_filter_metamorphic").enable_metamorphic_oracle is True
    assert (
        _preset_config("pyarrow_groupby_filter_cast_membership").generator_profile
        == "pyarrow_groupby_filter_cast_membership"
    )
    assert (
        _preset_config("pyarrow_groupby_filter_cast_membership").guidance_targets[0]
        == "pyarrow_groupby_filter_cast_membership"
    )
    assert _preset_config("pyarrow_groupby_filter_cast_membership_metamorphic").enable_metamorphic_oracle is True
    assert _preset_config("null_predicate_filter").generator_profile == "null_predicate_filter"
    assert _preset_config("null_predicate_filter").guidance_targets[0] == "null_predicate_filter"
    assert _preset_config("null_predicate_filter_metamorphic").enable_metamorphic_oracle is True
    assert _preset_config("boolean_predicate_filter").generator_profile == "boolean_predicate_filter"
    assert _preset_config("boolean_predicate_filter").guidance_targets[0] == "boolean_predicate_filter"
    assert _preset_config("boolean_predicate_filter_metamorphic").enable_metamorphic_oracle is True
    assert _preset_config("post_topk_range_filter").generator_profile == "post_topk_range_filter"
    assert _preset_config("post_topk_range_filter").guidance_targets[0] == "post_topk_range_filter"
    assert _preset_config("post_topk_range_filter_metamorphic").enable_metamorphic_oracle is True
    assert _preset_config("tuple_absence_filter").generator_profile == "tuple_absence_filter"
    assert _preset_config("tuple_absence_filter").guidance_targets[0] == "tuple_absence_filter"
    assert _preset_config("tuple_absence_filter_metamorphic").enable_metamorphic_oracle is True
    assert _preset_config("row_value_absence_filter").generator_profile == "row_value_absence_filter"
    assert _preset_config("row_value_absence_filter").guidance_targets[0] == "row_value_absence_filter"
    assert _preset_config("row_value_absence_filter_metamorphic").enable_metamorphic_oracle is True
    assert _preset_config("running_sum_precision").generator_profile == "running_sum_precision"
    assert _preset_config("running_sum_precision").guidance_targets[0] == "running_sum_precision"
    assert _preset_config("running_sum_precision_metamorphic").enable_metamorphic_oracle is True
    assert _preset_config("partitioned_running_sum").generator_profile == "partitioned_running_sum"
    assert _preset_config("partitioned_running_sum").guidance_targets[0] == "partitioned_running_sum"
    assert _preset_config("partitioned_running_sum_metamorphic").enable_metamorphic_oracle is True
    assert _preset_config("path_basename_keyed_pick").generator_profile == "path_basename_keyed_pick"
    assert _preset_config("path_basename_keyed_pick").guidance_targets[0] == "path_basename_keyed_pick"
    assert _preset_config("path_basename_keyed_pick_replay").enable_replay_bug is True
    assert _preset_config("path_basename_keyed_pick_replay").guidance_targets[0] == "path_basename_keyed_pick"
    assert _preset_config("sortedness_null_placement").generator_profile == "sortedness_null_placement"
    assert _preset_config("sortedness_null_placement").guidance_targets[0] == "sortedness_null_placement"
    assert _preset_config("sortedness_null_placement_metamorphic").enable_metamorphic_oracle is True
    assert _preset_config("simple_case_random_subject").generator_profile == "simple_case_random_subject"
    assert _preset_config("simple_case_random_subject").guidance_targets[0] == "simple_case_random_subject"
    assert _preset_config("simple_case_random_subject_metamorphic").enable_metamorphic_oracle is True
    assert _preset_config("group_quantile_key_probe").generator_profile == "group_quantile_key_probe"
    assert _preset_config("group_quantile_key_probe").guidance_targets[0] == "group_quantile_key_probe"
    assert _preset_config("group_quantile_key_probe_metamorphic").enable_metamorphic_oracle is True
    assert _preset_config("scalar_subquery_double_parentheses").generator_profile == "scalar_subquery_double_parentheses"
    assert _preset_config("scalar_subquery_double_parentheses").guidance_targets[0] == "scalar_subquery_double_parentheses"
    assert _preset_config("scalar_subquery_double_parentheses_metamorphic").enable_metamorphic_oracle is True
    assert _preset_config("window_avg_rows_frame").generator_profile == "window_avg_rows_frame"
    assert _preset_config("window_avg_rows_frame").guidance_targets[0] == "window_avg_rows_frame"
    assert _preset_config("window_avg_rows_frame_metamorphic").enable_metamorphic_oracle is True
    assert _preset_config("struct_distinct_unnest").generator_profile == "struct_distinct_unnest"
    assert _preset_config("struct_distinct_unnest").guidance_targets[0] == "struct_distinct_unnest"
    assert _preset_config("struct_distinct_unnest_metamorphic").enable_metamorphic_oracle is True
    assert _preset_config("bit_compare_unequal_length").generator_profile == "bit_compare_unequal_length"
    assert _preset_config("bit_compare_unequal_length").guidance_targets[0] == "bit_compare_unequal_length"
    assert _preset_config("bit_compare_unequal_length_metamorphic").enable_metamorphic_oracle is True
    assert _preset_config("round_even_float_scale").generator_profile == "round_even_float_scale"
    assert _preset_config("round_even_float_scale").guidance_targets[0] == "round_even_float_scale"
    assert _preset_config("round_even_float_scale_metamorphic").enable_metamorphic_oracle is True
    assert _preset_config("duckdb_float_literal_precision").generator_profile == "duckdb_float_literal_precision"
    assert _preset_config("duckdb_float_literal_precision").guidance_targets[0] == "duckdb_float_literal_precision"
    assert _preset_config("duckdb_float_literal_precision_metamorphic").enable_metamorphic_oracle is True
    assert _preset_config("polars_timestamp_precision_filter").generator_profile == "polars_timestamp_precision_filter"
    assert _preset_config("polars_timestamp_precision_filter").guidance_targets[0] == "polars_timestamp_precision_filter"
    assert _preset_config("polars_timestamp_precision_filter_metamorphic").enable_metamorphic_oracle is True
    assert _preset_config("series_rtruediv_operand_order").generator_profile == "series_rtruediv_operand_order"
    assert _preset_config("series_rtruediv_operand_order").guidance_targets[0] == "series_rtruediv_operand_order"
    assert _preset_config("series_rtruediv_operand_order_metamorphic").enable_metamorphic_oracle is True
    assert _preset_config("polars_reverse_division_columns").generator_profile == "polars_reverse_division_columns"
    assert _preset_config("polars_reverse_division_columns").guidance_targets[0] == "polars_reverse_division_columns"
    assert _preset_config("polars_reverse_division_columns_metamorphic").enable_metamorphic_oracle is True
    assert _preset_config("pandas_uint64_isin_precision").generator_profile == "pandas_uint64_isin_precision"
    assert _preset_config("pandas_uint64_isin_precision").guidance_targets[0] == "pandas_uint64_isin_precision"
    assert _preset_config("pandas_uint64_isin_precision_metamorphic").enable_metamorphic_oracle is True
    assert _preset_config("duckdb_tuple_anti_null_semantics").generator_profile == "duckdb_tuple_anti_null_semantics"
    assert _preset_config("duckdb_tuple_anti_null_semantics").guidance_targets[0] == "duckdb_tuple_anti_null_semantics"
    assert _preset_config("duckdb_tuple_anti_null_semantics_metamorphic").enable_metamorphic_oracle is True
    assert (
        _preset_config("datafusion_setop_all_duplicate_count").generator_profile
        == "datafusion_setop_all_duplicate_count"
    )
    assert (
        _preset_config("datafusion_setop_all_duplicate_count").guidance_targets[0]
        == "datafusion_setop_all_duplicate_count"
    )
    replay = _preset_config("datafusion_setop_all_duplicate_count_replay")
    assert replay.generator_profile == "datafusion_setop_all_duplicate_count"
    assert replay.enable_replay_bug is True
    assert _preset_config("datafusion_setop_all_duplicate_count_metamorphic").enable_metamorphic_oracle is True
    assert _preset_config("duckdb_json_predicate_order_semantics").generator_profile == "duckdb_json_predicate_order_semantics"
    assert _preset_config("duckdb_json_predicate_order_semantics").guidance_targets[0] == "duckdb_json_predicate_order_semantics"
    assert _preset_config("duckdb_json_predicate_order_semantics_metamorphic").enable_metamorphic_oracle is True
    assert _preset_config("pandas_sparse_array_mask_semantics").generator_profile == "pandas_sparse_array_mask_semantics"
    assert _preset_config("pandas_sparse_array_mask_semantics").guidance_targets[0] == "pandas_sparse_array_mask_semantics"
    assert _preset_config("pandas_sparse_array_mask_semantics_metamorphic").enable_metamorphic_oracle is True
    assert _preset_config("polars_float_wrap_numerical_semantics").generator_profile == "polars_float_wrap_numerical_semantics"
    assert _preset_config("polars_float_wrap_numerical_semantics").guidance_targets[0] == "polars_float_wrap_numerical_semantics"
    assert _preset_config("polars_float_wrap_numerical_semantics_metamorphic").enable_metamorphic_oracle is True
    assert _preset_config("pandas_index_bool_result_type").generator_profile == "pandas_index_bool_result_type"
    assert _preset_config("pandas_index_bool_result_type").guidance_targets[0] == "pandas_index_bool_result_type"
    assert _preset_config("pandas_index_bool_result_type_metamorphic").enable_metamorphic_oracle is True
    assert _preset_config("polars_empty_literal_groupby_semantics").generator_profile == "polars_empty_literal_groupby_semantics"
    assert (
        _preset_config("polars_empty_literal_groupby_semantics").guidance_targets[0]
        == "polars_empty_literal_groupby_semantics"
    )
    assert _preset_config("polars_empty_literal_groupby_semantics_metamorphic").enable_metamorphic_oracle is True
    assert _preset_config("pandas_arrow_string_eq_sum_semantics").generator_profile == "pandas_arrow_string_eq_sum_semantics"
    assert (
        _preset_config("pandas_arrow_string_eq_sum_semantics").guidance_targets[0]
        == "pandas_arrow_string_eq_sum_semantics"
    )
    assert _preset_config("pandas_arrow_string_eq_sum_semantics_metamorphic").enable_metamorphic_oracle is True
    assert (
        _preset_config("pandas_arrow_timestamp_loc_slice_semantics").generator_profile
        == "pandas_arrow_timestamp_loc_slice_semantics"
    )
    assert (
        _preset_config("pandas_arrow_timestamp_loc_slice_semantics").guidance_targets[0]
        == "pandas_arrow_timestamp_loc_slice_semantics"
    )
    assert _preset_config("pandas_arrow_timestamp_loc_slice_semantics_metamorphic").enable_metamorphic_oracle is True
    assert (
        _preset_config("pandas_arrow_timestamp_index_attr_semantics").generator_profile
        == "pandas_arrow_timestamp_index_attr_semantics"
    )
    assert (
        _preset_config("pandas_arrow_timestamp_index_attr_semantics").guidance_targets[0]
        == "pandas_arrow_timestamp_index_attr_semantics"
    )
    assert _preset_config("pandas_arrow_timestamp_index_attr_semantics_metamorphic").enable_metamorphic_oracle is True
    assert (
        _preset_config("pandas_eval_inplace_aliasing_semantics").generator_profile
        == "pandas_eval_inplace_aliasing_semantics"
    )
    assert (
        _preset_config("pandas_eval_inplace_aliasing_semantics").guidance_targets[0]
        == "pandas_eval_inplace_aliasing_semantics"
    )
    assert _preset_config("pandas_eval_inplace_aliasing_semantics_metamorphic").enable_metamorphic_oracle is True
    assert (
        _preset_config("pandas_bool_reduction_skipna_semantics").generator_profile
        == "pandas_bool_reduction_skipna_semantics"
    )
    assert (
        _preset_config("pandas_bool_reduction_skipna_semantics").guidance_targets[0]
        == "pandas_bool_reduction_skipna_semantics"
    )
    assert _preset_config("pandas_bool_reduction_skipna_semantics_metamorphic").enable_metamorphic_oracle is True
    assert (
        _preset_config("pyarrow_dataset_isin_all_match_semantics").generator_profile
        == "pyarrow_dataset_isin_all_match_semantics"
    )
    assert (
        _preset_config("pyarrow_dataset_isin_all_match_semantics").guidance_targets[0]
        == "pyarrow_dataset_isin_all_match_semantics"
    )
    assert _preset_config("pyarrow_dataset_isin_all_match_semantics_metamorphic").enable_metamorphic_oracle is True
    assert (
        _preset_config("pyarrow_run_end_null_compute_semantics").generator_profile
        == "pyarrow_run_end_null_compute_semantics"
    )
    assert (
        _preset_config("pyarrow_run_end_null_compute_semantics").guidance_targets[0]
        == "pyarrow_run_end_null_compute_semantics"
    )
    assert _preset_config("pyarrow_run_end_null_compute_semantics_metamorphic").enable_metamorphic_oracle is True
    assert (
        _preset_config("pyarrow_large_string_partition_schema_semantics").generator_profile
        == "pyarrow_large_string_partition_schema_semantics"
    )
    assert (
        _preset_config("pyarrow_large_string_partition_schema_semantics").guidance_targets[0]
        == "pyarrow_large_string_partition_schema_semantics"
    )
    assert _preset_config("pyarrow_large_string_partition_schema_semantics_metamorphic").enable_metamorphic_oracle is True
    assert (
        _preset_config("pyarrow_hash_pivot_wider_order_semantics").generator_profile
        == "pyarrow_hash_pivot_wider_order_semantics"
    )
    assert (
        _preset_config("pyarrow_hash_pivot_wider_order_semantics").guidance_targets[0]
        == "pyarrow_hash_pivot_wider_order_semantics"
    )
    assert _preset_config("pyarrow_hash_pivot_wider_order_semantics_metamorphic").enable_metamorphic_oracle is True
    assert (
        _preset_config("pyarrow_list_flatten_parent_indices_semantics").generator_profile
        == "pyarrow_list_flatten_parent_indices_semantics"
    )
    assert (
        _preset_config("pyarrow_list_flatten_parent_indices_semantics").guidance_targets[0]
        == "pyarrow_list_flatten_parent_indices_semantics"
    )
    assert (
        _preset_config("pyarrow_list_flatten_parent_indices_semantics_metamorphic").enable_metamorphic_oracle is True
    )
    assert (
        _preset_config("polars_rolling_mean_by_null_count_semantics").generator_profile
        == "polars_rolling_mean_by_null_count_semantics"
    )
    assert (
        _preset_config("polars_rolling_mean_by_null_count_semantics").guidance_targets[0]
        == "polars_rolling_mean_by_null_count_semantics"
    )
    assert _preset_config("polars_rolling_mean_by_null_count_semantics_metamorphic").enable_metamorphic_oracle is True
    assert _preset_config("csv_long_numeric_roundtrip").generator_profile == "csv_long_numeric_roundtrip"
    assert _preset_config("csv_long_numeric_roundtrip").guidance_targets[0] == "csv_long_numeric_roundtrip"
    assert _preset_config("csv_long_numeric_roundtrip").candidate_recheck_count == 2
    assert _preset_config("csv_long_numeric_roundtrip_metamorphic").enable_metamorphic_oracle is True
    assert _preset_config("join_null_sort").generator_profile == "join_null_sort"
    assert _preset_config("join_null_sort").guidance_targets[0] == "join_null_sort"
    assert _preset_config("join_null_sort_metamorphic").enable_metamorphic_oracle is True


def test_cli_parses_bughunt_guided_metamorphic_preset():
    config = _preset_config("bughunt_guided_metamorphic")
    assert config.generator_profile == "bughunt"
    assert config.enable_metamorphic_oracle is True
    assert config.guidance_strategy == "guided"
    assert config.metamorphic_variant_limit == 8


def test_cli_parses_bughunt_no_groupby_guided_metamorphic_preset():
    config = _preset_config("bughunt_no_groupby_guided_metamorphic")
    assert config.generator_profile == "bughunt_no_groupby"
    assert config.enable_metamorphic_oracle is True
    assert config.guidance_strategy == "guided"
    assert "groupby" not in config.guidance_targets
    assert config.metamorphic_variant_limit == 8


def test_catalog_backed_preset_overlays_preserve_base_semantics():
    workflow = _preset_config("workflow")
    workflow_metamorphic = _preset_config("workflow_metamorphic")
    assert workflow_metamorphic.generator_profile == workflow.generator_profile
    assert workflow_metamorphic.enable_metamorphic_oracle is True
    assert workflow_metamorphic.oracle_mode == "both"

    guided = _preset_config("bughunt_guided")
    guided_metamorphic = _preset_config("bughunt_guided_metamorphic")
    assert guided_metamorphic.generator_profile == guided.generator_profile
    assert guided_metamorphic.guidance_strategy == guided.guidance_strategy
    assert guided_metamorphic.guidance_candidate_pool == guided.guidance_candidate_pool
    assert guided_metamorphic.guidance_targets == guided.guidance_targets
    assert guided_metamorphic.enable_metamorphic_oracle is True
    assert guided_metamorphic.oracle_mode == "both"
    assert guided_metamorphic.metamorphic_variant_limit == 8


def test_build_experiment_config_matches_final_harness_overlay_variants():
    workflow_metamorphic = build_experiment_config(
        "workflow",
        ("enable_metamorphic_oracle", "metamorphic_variant_limit_4"),
    )
    assert workflow_metamorphic.to_dict() == _preset_config("workflow_metamorphic").to_dict()

    no_feedback = build_experiment_config(
        "baseline",
        ("disable_feedback_corpus",),
    )
    assert no_feedback.to_dict() == _preset_config("no_feedback").to_dict()

    guided_join = build_experiment_config(
        "baseline",
        ("enable_guidance", "target_join"),
    )
    assert guided_join.to_dict() == _preset_config("guided_join").to_dict()

    bughunt_guided = build_experiment_config(
        "bughunt",
        ("enable_guidance", "target_bughunt_guided"),
    )
    assert bughunt_guided.to_dict() == _preset_config("bughunt_guided").to_dict()

    deep_organic = build_experiment_config(
        "live_deep_organic",
        ("enable_metamorphic_oracle", "candidate_pool_10", "metamorphic_variant_limit_8"),
    )
    assert deep_organic.to_dict() == _preset_config("live_deep_organic_metamorphic").to_dict()


def test_catalog_overlay_aliases_are_structured_in_preset_catalog():
    guided_join = PRESET_CATALOG["guided_join"]
    no_feedback = PRESET_CATALOG["no_feedback"]
    bughunt_guided = PRESET_CATALOG["bughunt_guided"]

    assert guided_join.base_preset == "baseline"
    assert guided_join.overlays == ("enable_guidance", "target_join")
    assert no_feedback.base_preset == "baseline"
    assert no_feedback.overlays == ("disable_feedback_corpus",)
    assert bughunt_guided.base_preset == "bughunt"
    assert bughunt_guided.overlays == ("enable_guidance", "target_bughunt_guided")


def test_catalog_backed_replay_overlay_preserves_live_base_targets():
    live = _preset_config("live_datafusion")
    replay = _preset_config("live_datafusion_replay")
    assert replay.enable_replay_bug is True
    assert replay.generator_profile == live.generator_profile
    assert replay.guidance_targets == live.guidance_targets
    assert replay.local_source_exploration_weight == live.local_source_exploration_weight


def test_cli_parses_bughunt_experiment_presets():
    assert _preset_config("bughunt").generator_profile == "bughunt"
    assert _preset_config("bughunt_no_groupby").generator_profile == "bughunt_no_groupby"
    guided = _preset_config("bughunt_guided")
    assert guided.generator_profile == "bughunt"
    assert guided.guidance_strategy == "guided"
    assert guided.guidance_targets == ["join", "groupby", "mutate", "filter", "expressions"]
    no_groupby_guided = _preset_config("bughunt_no_groupby_guided")
    assert no_groupby_guided.generator_profile == "bughunt_no_groupby"
    assert no_groupby_guided.guidance_targets == ["join", "mutate", "filter", "expressions", "sort_limit"]
    metamorphic = _preset_config("bughunt_metamorphic")
    assert metamorphic.generator_profile == "bughunt"
    assert metamorphic.enable_metamorphic_oracle is True
    no_groupby_metamorphic = _preset_config("bughunt_no_groupby_metamorphic")
    assert no_groupby_metamorphic.generator_profile == "bughunt_no_groupby"
    assert no_groupby_metamorphic.enable_metamorphic_oracle is True


def test_cli_parses_live_datafusion_presets():
    live = _preset_config("live_datafusion")
    assert live.generator_profile == "bughunt"
    assert live.enable_replay_bug is False
    assert live.guidance_strategy == "guided"
    assert live.guidance_candidate_pool == 12
    assert live.enable_local_source_scheduler is True
    assert live.local_source_exploration_weight == 0.35
    assert live.enable_family_saturation is True
    assert live.family_saturation_threshold == 4
    assert live.family_saturation_penalty == 6.0
    assert live.saturated_family_reward == 0.0
    assert live.candidate_recheck_count == 2
    assert live.issue_replay_saturation_threshold == 1
    assert live.issue_replay_saturation_penalty == 1.0
    assert live.issue_replay_global_saturation_threshold == 2
    assert live.issue_replay_global_saturation_penalty == 2.0
    assert live.issue_inspired_source_saturation_threshold == 3
    assert live.issue_inspired_source_saturation_penalty == 1.25
    assert "distinct_null_topk@datafusion" in live.known_saturated_bug_families
    assert "groupby_aggregation@datafusion" in live.known_saturated_bug_families
    assert "grouped_topk_null_sort_key@datafusion" in live.known_saturated_bug_families
    assert "joined_order_offset_projection@datafusion" in live.known_saturated_bug_families
    assert "negative_zero_comparison@datafusion" in live.known_saturated_bug_families
    assert "ordered_topk_projection@datafusion" in live.known_saturated_bug_families
    assert "outer_join_truth_filter@datafusion" in live.known_saturated_bug_families
    assert "topk_filter_pushdown@datafusion" in live.known_saturated_bug_families
    assert {
        "common_workflow",
        "operation_combo",
        "topk",
        "join",
        "groupby",
        "join_null_key_topk",
        "datafusion_setop_all_duplicate_count",
    }.issubset(live.guidance_targets)

    metamorphic = _preset_config("live_datafusion_metamorphic")
    assert metamorphic.enable_metamorphic_oracle is True
    assert metamorphic.oracle_mode == "both"
    assert metamorphic.metamorphic_variant_limit == 6

    fresh = _preset_config("live_datafusion_fresh")
    assert fresh.generator_profile == "bughunt_no_groupby"
    assert fresh.enable_replay_bug is False
    assert fresh.guidance_strategy == "guided"
    assert fresh.enable_local_source_scheduler is True
    assert fresh.local_source_exploration_weight == 0.45
    assert fresh.family_saturation_threshold == 4
    assert fresh.family_saturation_penalty == 6.0
    assert fresh.saturated_family_reward == 0.0
    assert fresh.issue_replay_global_saturation_threshold == 2
    assert fresh.issue_replay_global_saturation_penalty == 2.0
    assert "groupby" not in fresh.guidance_targets
    assert "sort_offset" not in fresh.guidance_targets
    assert "wide_offset_topk" not in fresh.guidance_targets
    assert {"join", "mutate", "filter", "truth_filter", "set_membership_filter", "setop_all_duplicates"}.issubset(
        fresh.guidance_targets
    )
    assert "joined_order_offset_projection@datafusion" in fresh.known_saturated_bug_families
    assert "negative_zero_comparison@datafusion" in fresh.known_saturated_bug_families
    assert "ordered_topk_projection@datafusion" in fresh.known_saturated_bug_families
    assert "distinct_null_topk@datafusion" in fresh.known_saturated_bug_families

    fresh_metamorphic = _preset_config("live_datafusion_fresh_metamorphic")
    assert fresh_metamorphic.generator_profile == "bughunt_no_groupby"
    assert fresh_metamorphic.enable_metamorphic_oracle is True
    assert fresh_metamorphic.oracle_mode == "both"
    assert fresh_metamorphic.metamorphic_variant_limit == 6

    replay = _preset_config("live_datafusion_replay")
    assert replay.generator_profile == "bughunt"
    assert replay.enable_replay_bug is True
    assert replay.guidance_targets == live.guidance_targets


def test_cli_parses_non_datafusion_live_presets():
    arrow = _preset_config("live_arrow")
    assert arrow.generator_profile == "bughunt"
    assert arrow.guidance_strategy == "guided"
    assert {
        "join",
        "groupby",
        "strings",
        "casts",
        "topk",
        "global_null_aggregate",
        "string_count_groupby",
        "unique_count_groupby",
        "set_membership_filter",
        "pyarrow_groupby_filter_cast_membership",
        "null_predicate_filter",
        "boolean_predicate_filter",
        "post_topk_range_filter",
        "tuple_absence_filter",
        "running_sum_precision",
        "sortedness_null_placement",
        "simple_case_random_subject",
        "group_quantile_key_probe",
        "scalar_subquery_double_parentheses",
        "window_avg_rows_frame",
        "struct_distinct_unnest",
        "bit_compare_unequal_length",
        "round_even_float_scale",
        "duckdb_float_literal_precision",
        "polars_timestamp_precision_filter",
        "series_rtruediv_operand_order",
        "pandas_uint64_isin_precision",
        "duckdb_tuple_anti_null_semantics",
        "duckdb_json_predicate_order_semantics",
        "pandas_sparse_array_mask_semantics",
        "polars_float_wrap_numerical_semantics",
        "pandas_index_bool_result_type",
        "polars_empty_literal_groupby_semantics",
        "pandas_arrow_string_eq_sum_semantics",
        "pandas_arrow_timestamp_loc_slice_semantics",
        "pandas_arrow_timestamp_index_attr_semantics",
        "pandas_eval_inplace_aliasing_semantics",
        "pandas_bool_reduction_skipna_semantics",
        "pyarrow_dataset_isin_all_match_semantics",
        "pyarrow_large_string_partition_schema_semantics",
        "pyarrow_hash_pivot_wider_order_semantics",
        "pyarrow_list_flatten_parent_indices_semantics",
        "polars_rolling_mean_by_null_count_semantics",
    }.issubset(arrow.guidance_targets)
    assert arrow.enable_local_source_scheduler is True
    assert arrow.issue_replay_global_saturation_threshold == 2
    assert arrow.issue_replay_global_saturation_penalty == 2.0

    polars_lazy = _preset_config("live_polars_lazy")
    assert polars_lazy.generator_profile == "bughunt"
    assert {
        "filter",
        "mutate",
        "sort_limit",
        "topk",
        "global_aggregation",
        "polars_reverse_division_columns",
    }.issubset(polars_lazy.guidance_targets)
    assert polars_lazy.local_source_exploration_weight == 0.45
    assert polars_lazy.issue_replay_global_saturation_threshold == 2
    assert polars_lazy.issue_replay_global_saturation_penalty == 2.0
    assert "reverse_division_operand_order@polars" in polars_lazy.known_saturated_bug_families

    polars_streaming = _preset_config("live_polars_streaming")
    assert polars_streaming.generator_profile == "bughunt"
    assert {
        "groupby",
        "aggregation",
        "numeric_mean_aggregate",
        "sort_limit",
    }.issubset(polars_streaming.guidance_targets)
    assert polars_streaming.local_source_exploration_weight == 0.45

    embedded_sql = _preset_config("live_embedded_sql")
    assert embedded_sql.generator_profile == "bughunt"
    assert {
        "join",
        "filter",
        "groupby",
        "groupby_sorted_input",
        "aggregation",
        "casts",
        "row_value_absence_filter",
    }.issubset(embedded_sql.guidance_targets)
    assert "tuple_absence_null_filter@duckdb" in embedded_sql.known_saturated_bug_families
    assert "csv_long_numeric_roundtrip@duckdb" in embedded_sql.known_saturated_bug_families
    assert embedded_sql.issue_replay_global_saturation_threshold == 2
    assert embedded_sql.issue_replay_global_saturation_penalty == 2.0

    cross_family = _preset_config("live_cross_family")
    assert cross_family.generator_profile == "bughunt"
    assert {
        "common_workflow",
        "operation_combo",
        "join",
        "groupby",
        "groupby_sorted_input",
        "topk",
        "pyarrow_groupby_filter_cast_membership",
    }.issubset(cross_family.guidance_targets)
    assert "csv_long_numeric_roundtrip@duckdb" in cross_family.known_saturated_bug_families
    assert "csv_long_numeric_roundtrip@pyarrow" in cross_family.known_saturated_bug_families
    assert cross_family.issue_replay_global_saturation_threshold == 2
    assert cross_family.issue_replay_global_saturation_penalty == 2.0

    deep_organic = _preset_config("live_deep_organic")
    assert deep_organic.generator_profile == "bughunt_fresh"
    assert deep_organic.guidance_candidate_pool == 14
    assert deep_organic.enable_replay_bug is False
    assert deep_organic.enable_local_source_scheduler is True
    assert deep_organic.candidate_recheck_count == 2
    assert {"stateful_ordering", "join_membership", "string_semantics"}.issubset(
        deep_organic.semantic_focus_families
    )
    assert {
        "groupby_sorted_input",
        "running_sum_precision",
        "normalized_string_membership",
    }.issubset(deep_organic.semantic_focus_signals)
    assert {
        "running_sum_precision",
        "groupby_sorted_input",
        "sortedness_null_placement",
        "window_avg_rows_frame",
        "struct_distinct_unnest",
        "bit_compare_unequal_length",
        "round_even_float_scale",
        "pandas_arrow_timestamp_loc_slice_semantics",
        "pyarrow_dataset_isin_all_match_semantics",
        "pyarrow_list_flatten_parent_indices_semantics",
        "polars_rolling_mean_by_null_count_semantics",
    }.issubset(deep_organic.guidance_targets)
    assert "polars_reverse_division_columns" not in deep_organic.guidance_targets
    assert "csv_long_numeric_roundtrip" not in deep_organic.guidance_targets

    common_api = _preset_config("live_common_api_workflow_metamorphic")
    assert common_api.generator_profile == "common_api_workflow"
    assert common_api.enable_metamorphic_oracle is True
    assert common_api.guidance_candidate_pool == 8
    assert common_api.metamorphic_variant_limit == 6
    assert {
        "materialization_boundary",
        "string_semantics",
        "join_membership",
        "set_semantics",
    }.issubset(common_api.semantic_focus_families)
    assert {
        "common_api_workflow",
        "filter_input_materialization",
        "distinct_input_materialization",
        "case_when_membership",
        "string_pattern_case_when",
    }.issubset(common_api.semantic_focus_signals)
    assert len(common_api.discovery_biases) == 1
    assert common_api.discovery_biases[0].targets == [
        "common_api_workflow",
        "filter_input_materialization",
        "distinct_input_materialization",
        "case_when_membership",
        "string_pattern_case_when",
    ]
    assert "semantic_signal:" in common_api.discovery_biases[0].feature_prefixes
    assert "combo_risk:" in common_api.discovery_biases[0].feature_prefixes
    assert common_api.discovery_biases[0].keep_in_pool is True
    assert {
        "common_api_workflow",
        "daily_api",
        "input_materialization",
        "materialization_boundary",
        "filter_input_materialization",
        "cleanup_input_materialization",
        "drop_nulls_input_materialization",
        "fill_null_input_materialization",
        "distinct_input_materialization",
        "input_partition_union",
        "join",
        "multi_key_join",
        "groupby",
        "multi_key_groupby",
        "groupby_sorted_input",
        "normalized_string_join",
        "normalized_string_membership",
        "left_join_coalesce_membership",
        "join_coalesce_membership",
        "sql_distinct_null_topk",
        "coalesced_distinct_topk",
        "case_when_membership",
        "union_coalesce_distinct_topk",
        "left_join_case_membership",
        "left_join_null_predicate_aggregation",
        "coalesce_case_distinct_aggregation",
        "numeric_text_cast_membership",
        "multi_key_membership_aggregation",
        "boolean_membership_case_aggregation",
        "left_join_boolean_case_aggregation",
        "boolean_coalesce_case_aggregation",
        "left_join_boolean_coalesce_aggregation",
        "boolean_coalesce_filter_aggregation",
        "left_join_boolean_coalesce_filter_aggregation",
        "embedded_sql_rewrite",
        "union_all",
        "drop_nulls",
        "semi_join",
        "anti_join",
        "distinct",
        "nunique",
        "count_distinct",
        "distinct_count_aggregation",
        "distinct_null_topk",
        "fill_null",
        "coalesce",
        "case_when",
        "set_membership_filter",
        "negative_set_membership_filter",
        "range_filter",
        "row_number_filter",
        "top_n_per_group",
        "multi_key_join",
        "running_sum",
        "partitioned_running_sum",
        "filtered_global_aggregation",
        "empty_filter_aggregate",
        "boolean_aggregation",
        "bool_any_all",
        "string_length",
        "string_lower",
        "string_upper",
        "string_null_if_empty",
        "string_starts_with",
        "string_ends_with",
        "string_split_part",
        "date_part",
        "date_extraction",
    }.issubset(common_api.guidance_targets)

    datafusion_common_api = _preset_config("live_datafusion_common_api_metamorphic")
    assert datafusion_common_api.generator_profile == "common_api_workflow"
    assert datafusion_common_api.enable_metamorphic_oracle is True
    assert datafusion_common_api.guidance_targets == [
        "common_api_workflow",
        "daily_api",
        "common_workflow",
        "operation_combo",
        "input_materialization",
        "materialization_boundary",
        "filter_input_materialization",
        "cleanup_input_materialization",
        "fill_null_input_materialization",
        "distinct_input_materialization",
        "input_partition_union",
        "distinct_null_topk",
        "distinct",
        "duplicate_elimination",
        "nunique",
        "count_distinct",
        "distinct_count_aggregation",
        "sort_limit",
        "nulls",
        "boolean_aggregation",
        "bool_any_all",
        "row_number_filter",
        "top_n_per_group",
        "multi_key_join",
        "multi_key_groupby",
        "groupby_sorted_input",
        "normalized_string_join",
        "normalized_string_membership",
        "left_join_coalesce_membership",
        "join_coalesce_membership",
        "sql_distinct_null_topk",
        "coalesced_distinct_topk",
        "case_when_membership",
        "union_coalesce_distinct_topk",
        "left_join_case_membership",
        "left_join_null_predicate_aggregation",
        "coalesce_case_distinct_aggregation",
        "numeric_text_cast_membership",
        "multi_key_membership_aggregation",
        "boolean_membership_case_aggregation",
        "left_join_boolean_case_aggregation",
        "boolean_coalesce_case_aggregation",
        "left_join_boolean_coalesce_aggregation",
        "boolean_coalesce_filter_aggregation",
        "left_join_boolean_coalesce_filter_aggregation",
        "negative_set_membership_filter",
        "running_sum",
        "partitioned_running_sum",
        "filtered_global_aggregation",
        "empty_filter_aggregate",
        "having_filter_after_aggregation",
        "strings",
        "string_length",
        "string_lower",
        "string_upper",
        "string_null_if_empty",
        "string_starts_with",
        "string_ends_with",
        "string_split_part",
        "date_part",
        "date_extraction",
        "numeric",
    ]

    arrow_deep = _preset_config("live_arrow_deep_organic_metamorphic")
    assert arrow_deep.generator_profile == "bughunt_fresh"
    assert arrow_deep.enable_metamorphic_oracle is True
    assert "pyarrow_run_end_null_compute_semantics" in arrow_deep.guidance_targets
    assert "csv_long_numeric_roundtrip" not in arrow_deep.guidance_targets

    polars_deep = _preset_config("live_polars_deep_organic_metamorphic")
    assert polars_deep.generator_profile == "bughunt_fresh"
    assert polars_deep.enable_metamorphic_oracle is True
    assert "polars_rolling_mean_by_null_count_semantics" in polars_deep.guidance_targets
    assert "polars_reverse_division_columns" not in polars_deep.guidance_targets

    streaming_deep = _preset_config("live_polars_streaming_deep_organic_metamorphic")
    assert streaming_deep.generator_profile == "bughunt_fresh"
    assert streaming_deep.enable_metamorphic_oracle is True
    assert "partitioned_running_sum" in streaming_deep.guidance_targets

    datafusion_deep = _preset_config("live_datafusion_deep_organic_metamorphic")
    assert datafusion_deep.generator_profile == "bughunt_fresh"
    assert datafusion_deep.enable_metamorphic_oracle is True
    assert "datafusion_setop_all_duplicate_count" not in datafusion_deep.guidance_targets
    assert "row_value_absence_filter" in datafusion_deep.guidance_targets
    assert len(datafusion_deep.discovery_biases) == 1
    assert datafusion_deep.discovery_biases[0].targets == [
        "row_value_absence_filter",
        "normalized_string_join",
        "negative_set_membership_filter",
    ]

    embedded_deep = _preset_config("live_embedded_sql_deep_organic_metamorphic")
    assert embedded_deep.generator_profile == "bughunt_fresh"
    assert embedded_deep.enable_metamorphic_oracle is True
    assert "duckdb_json_predicate_order_semantics" not in embedded_deep.guidance_targets
    assert "row_value_absence_filter" in embedded_deep.guidance_targets
    assert "groupby_sorted_input" in embedded_deep.guidance_targets

    issue_focus = _preset_config("live_issue_focus")
    assert issue_focus.generator_profile == "issue_focus"
    assert issue_focus.enable_replay_bug is False
    assert issue_focus.candidate_recheck_count == 2
    assert issue_focus.guidance_candidate_pool == 14
    assert {
        "row_value_absence_filter",
        "polars_reverse_division_columns",
        "join_filter_groupby",
        "bool_null_groupby_agg",
        "large_int_filter_groupby",
        "pandas_bool_reduction_skipna_semantics",
        "bool_reduction_skipna_probe",
    }.issubset(
        issue_focus.guidance_targets
    )

    polars_issue = _preset_config("live_polars_issue_focus")
    assert polars_issue.generator_profile == "issue_focus"
    assert {
        "empty_filter_groupby",
        "polars_reverse_division_columns",
        "topk_resort",
        "boolean_aggregation",
        "bool_any_all",
        "large_integer",
        "csv_long_numeric_roundtrip",
        "csv_numeric_inference",
    }.issubset(
        polars_issue.guidance_targets
    )

    duckdb_issue = _preset_config("live_duckdb_issue_focus")
    assert duckdb_issue.generator_profile == "issue_focus"
    assert {"row_value_absence_filter", "null_predicate_filter", "join_filter_groupby"}.issubset(
        duckdb_issue.guidance_targets
    )

    arrow_issue = _preset_config("live_arrow_issue_focus")
    assert arrow_issue.generator_profile == "issue_focus"
    assert {
        "pyarrow_groupby_filter_cast_membership",
        "unique_count_groupby",
        "bool_null_groupby_agg",
        "bool_any_all",
        "large_int_filter_groupby",
        "strings",
    }.issubset(
        arrow_issue.guidance_targets
    )

    deep_probe = _preset_config("live_deep_probe_rotation_metamorphic")
    assert deep_probe.generator_profile == "deep_probe_rotation"
    assert deep_probe.enable_metamorphic_oracle is True
    assert deep_probe.guidance_candidate_pool == 4
    assert deep_probe.local_source_exploration_weight == 0.25
    assert {
        "dataset_isin_all_match_probe",
        "run_end_null_compute_probe",
        "json_predicate_order_probe",
        "rolling_mean_by_null_count_probe",
        "csv_long_numeric_roundtrip_probe",
    }.issubset(deep_probe.guidance_targets)


def test_bug_sprint_config_can_merge_lane_discovery_biases():
    parser = build_parser()
    args = parser.parse_args(["bug-sprint", "--lanes", "datafusion_optimizer"])
    config = cli._bug_sprint_config_from_args(
        args,
        "live_datafusion_deep_organic_metamorphic",
        lane_discovery_biases=[
            {
                "targets": ["negative_set_membership_filter"],
                "feature_prefixes": ["join:"],
                "score_bonus": 0.5,
                "novelty_bonus": 0.2,
                "contribution_bonus": 0.2,
                "candidate_pool_bonus": 0.1,
                "keep_in_pool": True,
            }
        ],
    )

    assert len(config.discovery_biases) == 2
    assert {tuple(bias.targets) for bias in config.discovery_biases} == {
        ("negative_set_membership_filter",),
        ("row_value_absence_filter", "normalized_string_join", "negative_set_membership_filter"),
    }
    assert any(bias.keep_in_pool for bias in config.discovery_biases)


def test_bug_sprint_config_deduplicates_lane_discovery_biases():
    parser = build_parser()
    args = parser.parse_args(["bug-sprint", "--lanes", "datafusion_optimizer"])
    lane_bias = {
        "targets": ["row_value_absence_filter", "normalized_string_join", "negative_set_membership_filter"],
        "feature_prefixes": ["join:", "filter:", "pattern:"],
        "score_bonus": 0.45,
        "novelty_bonus": 0.20,
        "contribution_bonus": 0.20,
        "candidate_pool_bonus": 0.10,
        "keep_in_pool": True,
    }

    config = cli._bug_sprint_config_from_args(
        args,
        "live_datafusion_deep_organic_metamorphic",
        lane_discovery_biases=[lane_bias],
    )

    assert len(config.discovery_biases) == 1
    assert config.discovery_biases[0].targets == lane_bias["targets"]


def test_bug_sprint_config_merges_lane_semantic_focus():
    parser = build_parser()
    args = parser.parse_args(["bug-sprint", "--lanes", "datafusion_optimizer"])

    config = cli._bug_sprint_config_from_args(
        args,
        "live_datafusion_deep_organic_metamorphic",
        lane_semantic_focus_families=["join_membership", "string_semantics"],
        lane_semantic_focus_signals=["row_value_absence_filter", "normalized_string_join"],
    )

    assert config.semantic_focus_families == ["join_membership", "string_semantics"]
    assert config.semantic_focus_signals == [
        "row_value_absence_filter",
        "normalized_string_join",
    ]


def test_bug_sprint_config_deduplicates_lane_semantic_focus():
    parser = build_parser()
    args = parser.parse_args(["bug-sprint", "--lanes", "datafusion_optimizer"])

    config = cli._bug_sprint_config_from_args(
        args,
        "live_datafusion_deep_organic_metamorphic",
        lane_semantic_focus_families=["join_membership", "string_semantics", "topk_ordering"],
        lane_semantic_focus_signals=[
            "row_value_absence_filter",
            "normalized_string_join",
            "negative_set_membership_filter",
        ],
    )

    assert config.semantic_focus_families == ["join_membership", "string_semantics", "topk_ordering"]
    assert config.semantic_focus_signals == [
        "row_value_absence_filter",
        "normalized_string_join",
        "negative_set_membership_filter",
    ]


def test_preset_config_roundtrip_preserves_discovery_bias_objects():
    payload = _preset_config("live_common_api_workflow_metamorphic").to_dict()

    config = cli.ExperimentConfig(**payload)

    assert config.discovery_biases
    assert all(isinstance(bias, DiscoveryBias) for bias in config.discovery_biases)


def test_cli_parses_non_datafusion_live_metamorphic_presets():
    for preset in [
        "live_arrow_metamorphic",
        "live_polars_lazy_metamorphic",
        "live_polars_streaming_metamorphic",
        "live_embedded_sql_metamorphic",
        "live_cross_family_metamorphic",
        "live_deep_organic_metamorphic",
        "live_common_api_workflow_metamorphic",
        "live_issue_focus_metamorphic",
        "live_polars_issue_focus_metamorphic",
        "live_duckdb_issue_focus_metamorphic",
        "live_arrow_issue_focus_metamorphic",
        "live_deep_probe_rotation_metamorphic",
    ]:
        config = _preset_config(preset)
        assert config.enable_metamorphic_oracle is True
        assert config.oracle_mode == "both"
        assert config.guidance_strategy == "guided"
        assert config.enable_local_source_scheduler is True

    deep_organic = _preset_config("live_deep_organic_metamorphic")
    assert deep_organic.generator_profile == "bughunt_fresh"
    assert deep_organic.guidance_candidate_pool == 10
    assert deep_organic.metamorphic_variant_limit == 8


def test_cli_experiment_duration_can_run_without_case_cap():
    parser = build_parser()
    args = parser.parse_args(["experiment", "--duration", "1s", "--seeds", "1"])
    assert args.cmd == "experiment"
    assert args.cases is None
    assert args.duration == "1s"


def test_cli_parses_analyze_experiment_command():
    parser = build_parser()
    args = parser.parse_args(
        [
            "analyze-experiment",
            "--manifest",
            "runs/experiment-x.json",
            "--reference-preset",
            "baseline",
            "--compare-presets",
            "guided_filter,guided_join",
            "--refresh",
        ]
    )
    assert args.cmd == "analyze-experiment"
    assert args.manifest == "runs/experiment-x.json"
    assert args.reference_preset == "baseline"
    assert args.baseline_preset is None
    assert args.compare_presets == "guided_filter,guided_join"
    assert args.refresh is True


def test_cli_parses_analyze_experiment_legacy_baseline_alias():
    parser = build_parser()
    args = parser.parse_args(
        [
            "analyze-experiment",
            "--manifest",
            "runs/experiment-x.json",
            "--baseline-preset",
            "baseline",
        ]
    )
    assert args.cmd == "analyze-experiment"
    assert args.reference_preset == "baseline"
    assert args.baseline_preset == "baseline"


def test_cli_parses_analyze_seeded_sensitivity_command():
    parser = build_parser()
    args = parser.parse_args(
        [
            "analyze-seeded-sensitivity",
            "--manifest",
            "runs/experiment-seeded.json",
        ]
    )
    assert args.cmd == "analyze-seeded-sensitivity"
    assert args.manifest == "runs/experiment-seeded.json"


def test_cli_parses_methodology_report_command():
    parser = build_parser()
    args = parser.parse_args(
        [
            "methodology-report",
            "--manifest",
            "runs/experiment-methodology.json",
            "--refresh",
            "--summary-only",
            "--json",
        ]
    )
    assert args.cmd == "methodology-report"
    assert args.manifest == "runs/experiment-methodology.json"
    assert args.refresh is True
    assert args.summary_only is True
    assert args.json is True


def test_cli_methodology_report_can_print_json(tmp_path, monkeypatch, capsys):
    md_path = tmp_path / "methodology.md"
    json_path = tmp_path / "methodology.json"
    json_path.write_text(
        json.dumps({"schema_version": "methodology-report-v1", "ok": True}),
        encoding="utf-8",
    )
    calls = []

    def fake_write_methodology_report(manifest_file=None, *, refresh=False, scan_run_logs=True):
        calls.append((manifest_file, refresh, scan_run_logs))
        return md_path, json_path

    monkeypatch.setattr(cli, "write_methodology_report", fake_write_methodology_report)

    parser = build_parser()
    args = parser.parse_args(
        [
            "methodology-report",
            "--manifest",
            str(tmp_path / "experiment.json"),
            "--refresh",
            "--summary-only",
            "--json",
        ]
    )

    assert args.func(args) == 0
    assert calls == [(tmp_path / "experiment.json", True, False)]
    assert json.loads(capsys.readouterr().out)["schema_version"] == "methodology-report-v1"


def test_cli_parses_final_readiness_command():
    parser = build_parser()
    args = parser.parse_args(
        [
            "final-readiness",
            "--manifest",
            "runs/experiment-a.json",
            "--manifest",
            "runs/experiment-b.json",
            "--latest-manifests",
            "12",
            "--all-manifests",
            "--summary-only",
            "--full-run-log-scan",
            "--latest-confirmation-file",
            "experiments/latest_confirmations.json",
            "--min-live-cases-per-suite",
            "100",
            "--min-live-duration-hours",
            "1",
            "--min-live-candidate-families",
            "2",
            "--min-confirmed-live-families",
            "1",
            "--min-historical-confirmed",
            "2",
            "--required-live-suites",
            "datafusion_cross,arrow_cross",
            "--required-live-families",
            "dataframe,query_engine",
            "--no-require-validation",
            "--no-require-seeded",
            "--no-require-ablation",
            "--no-require-comparison",
            "--json",
            "--fail-on-missing",
        ]
    )
    assert args.cmd == "final-readiness"
    assert args.manifest == ["runs/experiment-a.json", "runs/experiment-b.json"]
    assert args.latest_manifests == 12
    assert args.all_manifests is True
    assert args.summary_only is True
    assert args.full_run_log_scan is True
    assert args.latest_confirmation_file == ["experiments/latest_confirmations.json"]
    assert args.min_live_cases_per_suite == 100
    assert args.min_live_duration_hours == 1.0
    assert args.min_live_candidate_families == 2
    assert args.min_confirmed_live_families == 1
    assert args.min_historical_confirmed == 2
    assert args.required_live_suites == "datafusion_cross,arrow_cross"
    assert args.required_live_families == "dataframe,query_engine"
    assert args.no_require_validation is True
    assert args.no_require_seeded is True
    assert args.no_require_ablation is True
    assert args.no_require_comparison is True
    assert args.json is True
    assert args.fail_on_missing is True


def test_cli_final_readiness_prints_json_and_can_fail_on_missing(tmp_path, monkeypatch, capsys):
    audit = {"schema_version": "final-readiness-v1", "ready": False, "summary": {}}
    md_path = tmp_path / "final-readiness.md"
    json_path = tmp_path / "final-readiness.json"
    md_path.write_text("# Final\n", encoding="utf-8")
    json_path.write_text(json.dumps(audit), encoding="utf-8")
    calls = []

    def fake_analyze_final_readiness(*args, **kwargs):
        calls.append((args, kwargs))
        return md_path, json_path

    monkeypatch.setattr(cli, "analyze_final_readiness", fake_analyze_final_readiness)

    args = build_parser().parse_args(["final-readiness", "--json", "--fail-on-missing", "--latest-manifests", "3"])

    assert args.func(args) == 2
    assert json.loads(capsys.readouterr().out)["schema_version"] == "final-readiness-v1"
    assert calls[0][1]["manifest_limit"] == 3
    assert calls[0][1]["scan_run_logs"] is False


def test_cli_final_readiness_all_manifests_removes_default_limit(tmp_path, monkeypatch):
    audit = {"schema_version": "final-readiness-v1", "ready": True, "summary": {}}
    md_path = tmp_path / "final-readiness.md"
    json_path = tmp_path / "final-readiness.json"
    md_path.write_text("# Final\n", encoding="utf-8")
    json_path.write_text(json.dumps(audit), encoding="utf-8")
    calls = []

    def fake_analyze_final_readiness(*args, **kwargs):
        calls.append((args, kwargs))
        return md_path, json_path

    monkeypatch.setattr(cli, "analyze_final_readiness", fake_analyze_final_readiness)

    args = build_parser().parse_args(["final-readiness", "--all-manifests"])

    assert args.func(args) == 0
    assert calls[0][1]["manifest_limit"] is None
    assert calls[0][1]["scan_run_logs"] is False


def test_cli_final_readiness_explicit_manifest_scans_run_logs_by_default(tmp_path, monkeypatch):
    audit = {"schema_version": "final-readiness-v1", "ready": True, "summary": {}}
    md_path = tmp_path / "final-readiness.md"
    json_path = tmp_path / "final-readiness.json"
    md_path.write_text("# Final\n", encoding="utf-8")
    json_path.write_text(json.dumps(audit), encoding="utf-8")
    calls = []

    def fake_analyze_final_readiness(*args, **kwargs):
        calls.append((args, kwargs))
        return md_path, json_path

    monkeypatch.setattr(cli, "analyze_final_readiness", fake_analyze_final_readiness)

    args = build_parser().parse_args(["final-readiness", "--manifest", "runs/experiment-final.json"])

    assert args.func(args) == 0
    assert calls[0][0][0] == [Path("runs/experiment-final.json")]
    assert calls[0][1]["manifest_limit"] is None
    assert calls[0][1]["scan_run_logs"] is True


def test_cli_final_readiness_full_run_log_scan_overrides_no_manifest_summary_default(tmp_path, monkeypatch):
    audit = {"schema_version": "final-readiness-v1", "ready": True, "summary": {}}
    md_path = tmp_path / "final-readiness.md"
    json_path = tmp_path / "final-readiness.json"
    md_path.write_text("# Final\n", encoding="utf-8")
    json_path.write_text(json.dumps(audit), encoding="utf-8")
    calls = []

    def fake_analyze_final_readiness(*args, **kwargs):
        calls.append((args, kwargs))
        return md_path, json_path

    monkeypatch.setattr(cli, "analyze_final_readiness", fake_analyze_final_readiness)

    args = build_parser().parse_args(["final-readiness", "--full-run-log-scan", "--latest-manifests", "2"])

    assert args.func(args) == 0
    assert calls[0][1]["manifest_limit"] == 2
    assert calls[0][1]["scan_run_logs"] is True


def test_cli_parses_review_readiness_command():
    parser = build_parser()
    args = parser.parse_args(
        [
            "review-readiness",
            "--json",
            "--write-report",
            "--output-dir",
            "reports",
            "--latest-confirmation-file",
            "experiments/latest_confirmations.json",
            "--target-confirmed",
            "20",
            "--min-audit-candidates",
            "1",
            "--min-bug-workflows",
            "1",
            "--min-generated-issue-drafts",
            "1",
            "--min-issue-bundle-families",
            "1",
            "--min-pending-issue-drafts",
            "1",
            "--min-old-known-issues",
            "1",
            "--fail-on-missing",
        ]
    )
    assert args.cmd == "review-readiness"
    assert args.json is True
    assert args.write_report is True
    assert args.output_dir == "reports"
    assert args.latest_confirmation_file == ["experiments/latest_confirmations.json"]
    assert args.target_confirmed == 20
    assert args.min_issue_bundle_families == 1
    assert args.fail_on_missing is True


def test_cli_parses_issue_readiness_command():
    parser = build_parser()
    args = parser.parse_args(
        [
            "issue-readiness",
            "--json",
            "--latest-confirmations",
            "experiments/latest_confirmations.json,other.json",
            "--new-issue-dir",
            "new_issue",
            "--old-issue-dir",
            "old_issue",
            "--generated-issue-dir",
            "new_issue/generated",
            "--include-generated",
            "--write-report",
            "--output-dir",
            "reports",
            "--fail-on-no-ready",
        ]
    )
    assert args.cmd == "issue-readiness"
    assert args.json is True
    assert args.latest_confirmations == "experiments/latest_confirmations.json,other.json"
    assert args.include_generated is True
    assert args.write_report is True
    assert args.fail_on_no_ready is True


def test_cli_issue_readiness_prints_json(monkeypatch, capsys):
    audit = {
        "schema_version": "issue-readiness-v1",
        "summary": {
            "issue_document_count": 1,
            "ready_to_submit_count": 1,
            "ready_to_submit_family_count": 1,
            "ready_to_submit": ["new_issue/ready.md"],
            "needs_dedup_check_count": 0,
            "needs_dedup_check_family_count": 0,
            "needs_dedup_check": [],
            "needs_reproducer_or_evidence_count": 0,
            "needs_reproducer_or_evidence": [],
            "already_submitted_or_confirmed_count": 0,
            "not_latest_reproducible_count": 0,
        },
        "issues": [],
    }
    monkeypatch.setattr(cli, "build_issue_readiness", lambda **kwargs: audit)

    args = build_parser().parse_args(["issue-readiness", "--json"])

    assert args.func(args) == 0
    assert json.loads(capsys.readouterr().out)["schema_version"] == "issue-readiness-v1"


def test_cli_parses_issue_bundle_command():
    parser = build_parser()
    args = parser.parse_args(
        [
            "issue-bundle",
            "--json",
            "--latest-confirmations",
            "experiments/latest_confirmations.json,other.json",
            "--new-issue-dir",
            "new_issue",
            "--old-issue-dir",
            "old_issue",
            "--generated-issue-dir",
            "new_issue/generated",
            "--statuses",
            "ready_to_submit",
            "--output-dir",
            "new_issue/generated/issue-bundles",
            "--run-reproducers",
            "--timeout",
            "5",
            "--repeat",
            "3",
            "--primary-per-family",
            "--fail-on-missing-reproducer",
            "--fail-on-compile-error",
        ]
    )
    assert args.cmd == "issue-bundle"
    assert args.json is True
    assert args.latest_confirmations == "experiments/latest_confirmations.json,other.json"
    assert args.statuses == "ready_to_submit"
    assert args.run_reproducers is True
    assert args.timeout == 5.0
    assert args.repeat == 3
    assert args.primary_per_family is True
    assert args.fail_on_missing_reproducer is True
    assert args.fail_on_compile_error is True


def test_cli_issue_bundle_prints_json(monkeypatch, capsys):
    manifest = {
        "schema_version": "issue-bundle-v1",
        "manifest_path": "new_issue/generated/issue-bundles/manifest.json",
        "markdown_path": "new_issue/generated/issue-bundles/manifest.md",
        "summary": {
            "issue_count": 1,
            "family_count": 1,
            "extracted_reproducer_count": 1,
            "missing_reproducer_count": 0,
            "compile_failure_count": 0,
            "executed_reproducer_count": 0,
            "nonzero_exit_count": 0,
            "timeout_count": 0,
        },
        "issues": [],
    }
    captured = {}

    def fake_build_issue_bundle(**kwargs):
        captured.update(kwargs)
        return manifest

    monkeypatch.setattr(cli, "build_issue_bundle", fake_build_issue_bundle)

    args = build_parser().parse_args(["issue-bundle", "--json", "--primary-per-family"])

    assert args.func(args) == 0
    assert json.loads(capsys.readouterr().out)["schema_version"] == "issue-bundle-v1"
    assert captured["primary_per_family"] is True


def test_cli_parses_analyze_ablation_audit_command():
    parser = build_parser()
    args = parser.parse_args(
        [
            "analyze-ablation-audit",
            "--manifest",
            "runs/experiment-ablation.json",
            "--reference-presets",
            "baseline,guided",
            "--trusted-presets",
            "baseline,guided",
            "--ablation-presets",
            "no_type_aware,no_normalizer",
            "--refresh",
        ]
    )
    assert args.cmd == "analyze-ablation-audit"
    assert args.manifest == "runs/experiment-ablation.json"
    assert args.reference_presets == "baseline,guided"
    assert args.trusted_presets == "baseline,guided"
    assert args.ablation_presets == "no_type_aware,no_normalizer"
    assert args.refresh is True


def test_cli_ablation_audit_help_prefers_reference_wording(capsys):
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["analyze-ablation-audit", "--help"])
    help_text = capsys.readouterr().out

    assert "--reference-presets" in help_text
    assert "legacy alias for --reference-presets" in help_text


def test_cli_parses_analyze_pattern_variants_command():
    parser = build_parser()
    args = parser.parse_args(
        [
            "analyze-pattern-variants",
            "--manifest",
            "runs/experiment-pattern.json",
            "--pattern",
            "null_agg_topk",
        ]
    )
    assert args.cmd == "analyze-pattern-variants"
    assert args.manifest == "runs/experiment-pattern.json"
    assert args.pattern == "null_agg_topk"


def test_classify_run_refresh_recomputes_current_oracle_roots(tmp_path, capsys):
    case = Case(
        "case-stale-root",
        14,
        [TableData("t0", [ColumnSpec("x", "int")], [{"x": 0}, {"x": 0}])],
        Program(
            "prog-stale-root",
            14,
            [
                {"op": "mutate", "column": "m_0", "expr": {"kind": "add_const", "source": "x", "value": -1}},
                {"op": "filter", "column": "m_0", "cmp": "==", "value": -1},
                {"op": "mutate", "column": "m_1", "expr": {"kind": "arith_const", "op": "mul", "source": "m_0", "value": 10}},
                {"op": "mutate", "column": "m_3", "expr": {"kind": "arith_const", "op": "div", "source": "m_1", "value": 3}},
                {"op": "groupby", "keys": ["m_3"], "aggs": [{"column": "m_0", "func": "min", "as": "min_m_0"}]},
            ],
        ),
    )
    run_file = tmp_path / "run-stale.jsonl"
    append_jsonl(
        {
            "status": "bug",
            "case": case.to_dict(),
            "findings": [
                {
                    "kind": "semantic_output_mismatch",
                    "root_cause": "groupby_aggregation",
                    "triage_verdict": "candidate_implementation_bug",
                    "suspicious_backends": ["b"],
                    "signature": "stale",
                    "confidence": "high",
                }
            ],
            "normalized": {
                "a": {"backend": "a", "status": "ok", "columns": ["m_3", "min_m_0"], "rows": [[-3.3333333333, -1]]},
                "c": {"backend": "c", "status": "ok", "columns": ["m_3", "min_m_0"], "rows": [[-3.3333333333, -1]]},
                "b": {
                    "backend": "b",
                    "status": "ok",
                    "columns": ["m_3", "min_m_0"],
                    "rows": [[-3.3333333333, -1], [-3.3333333333, -1]],
                },
            },
        },
        run_file,
    )

    parser = build_parser()
    args = parser.parse_args(["classify-run", "--run-file", str(run_file), "--refresh"])
    assert args.func(args) == 0
    out = capsys.readouterr().out

    assert "refresh=true" in out
    assert "float_group_key_instability@b: 1" in out
    assert "root=float_group_key_instability" in out


def test_cli_experiment_parses_no_compress_run_log():
    parser = build_parser()
    args = parser.parse_args(["experiment", "--no-compress-run-log"])
    assert args.cmd == "experiment"
    assert args.no_compress_run_log is True


def test_cli_experiment_parses_artifact_limit():
    parser = build_parser()
    args = parser.parse_args(["experiment", "--artifact-limit", "20"])
    assert args.cmd == "experiment"
    assert args.artifact_limit == 20


def test_cli_experiment_parses_skip_run_reports():
    parser = build_parser()
    args = parser.parse_args(["experiment", "--skip-run-reports"])
    assert args.cmd == "experiment"
    assert args.skip_run_reports is True


def test_cli_experiment_parses_jobs():
    parser = build_parser()
    args = parser.parse_args(["experiment", "--jobs", "4"])
    assert args.cmd == "experiment"
    assert args.jobs == 4


def test_cli_experiment_parses_auto_jobs_and_parallel_cost():
    parser = build_parser()
    args = parser.parse_args(["experiment", "--jobs", "auto", "--max-parallel-cost", "9.5"])
    assert args.cmd == "experiment"
    assert args.jobs == "auto"
    assert args.max_parallel_cost == 9.5


def test_cli_experiment_parses_adaptive_schedule_flags():
    parser = build_parser()
    args = parser.parse_args(
        [
            "experiment",
            "--schedule",
            "adaptive",
            "--batch-cases",
            "25",
            "--batch-duration",
            "30s",
            "--warmup-batches",
            "2",
            "--exploration-weight",
            "0.5",
            "--group-fairness-weight",
            "0.3",
            "--max-group-pull-gap",
            "4",
            "--local-source-exploration-weight",
            "0.2",
        ]
    )
    assert args.cmd == "experiment"
    assert args.schedule == "adaptive"
    assert args.batch_cases == 25
    assert args.batch_duration == "30s"
    assert args.warmup_batches == 2
    assert args.exploration_weight == 0.5
    assert args.group_fairness_weight == 0.3
    assert args.max_group_pull_gap == 4
    assert args.local_source_exploration_weight == 0.2


def test_cli_experiment_parses_evidence_mode_flags():
    parser = build_parser()
    args = parser.parse_args(
        [
            "experiment",
            "--evidence-mode",
            "historical",
            "--known-bug-id",
            "datafusion-22190",
            "--target-version",
            "pre-fix-sha",
            "--enable-replay-bug",
            "--replay-bug-source-issues",
            "https://github.com/apache/datafusion/issues/22190",
        ]
    )
    assert args.evidence_mode == "historical"
    assert args.known_bug_id == "datafusion-22190"
    assert args.target_version == "pre-fix-sha"
    assert args.enable_replay_bug is True
    assert args.replay_bug_source_issues == "https://github.com/apache/datafusion/issues/22190"

    validation_args = parser.parse_args(["experiment", "--evidence-mode", "validation"])
    assert validation_args.evidence_mode == "validation"
    ablation_args = parser.parse_args(["experiment", "--evidence-mode", "ablation"])
    assert ablation_args.evidence_mode == "ablation"
    comparison_args = parser.parse_args(["experiment", "--evidence-mode", "comparison"])
    assert comparison_args.evidence_mode == "comparison"


def test_cli_parses_paper_run_journal_flags():
    parser = build_parser()
    args = parser.parse_args(
        [
            "experiment",
            "--run-theme",
            "final-live:datafusion",
            "--paper-notes",
            "24h latest-version run",
            "--skip-paper-journal",
        ]
    )
    assert args.run_theme == "final-live:datafusion"
    assert args.paper_notes == "24h latest-version run"
    assert args.skip_paper_journal is True


def test_run_experiment_job_propagates_local_source_scheduler(monkeypatch):
    captured = {}

    def fake_run_fuzz(*, cases, seed, backends, config, duration_s, **kwargs):
        captured["enable_local_source_scheduler"] = config.enable_local_source_scheduler
        captured["local_source_exploration_weight"] = config.local_source_exploration_weight
        captured["checkpoint_interval_s"] = kwargs["checkpoint_interval_s"]
        captured["progress_interval_s"] = kwargs["progress_interval_s"]
        return Path("runs/fake.jsonl")

    monkeypatch.setattr(cli, "run_fuzz", fake_run_fuzz)
    monkeypatch.setattr(cli, "write_report", lambda run_file: (Path(""), Path("")))

    result = cli._run_experiment_job(
        {
            "order": 0,
            "target_suite": "core",
            "backends": ["pandas"],
            "preset": "baseline",
            "seed": 1,
            "cases": 1,
            "duration_s": None,
            "log_level": "compact",
            "compress_run_log": True,
            "artifact_limit": None,
            "metamorphic_variant_limit": None,
            "enable_local_source_scheduler": True,
            "local_source_exploration_weight": 0.125,
            "skip_run_reports": True,
        }
    )

    assert captured["enable_local_source_scheduler"] is True
    assert captured["local_source_exploration_weight"] == 0.125
    assert captured["checkpoint_interval_s"] == 60.0
    assert captured["progress_interval_s"] == 60.0
    assert result["run"]["run_file"] == "runs/fake.jsonl"


def test_run_experiment_job_preserves_live_preset_source_scheduler(monkeypatch):
    captured = {}

    def fake_run_fuzz(*, cases, seed, backends, config, duration_s, **kwargs):
        captured["enable_local_source_scheduler"] = config.enable_local_source_scheduler
        captured["local_source_exploration_weight"] = config.local_source_exploration_weight
        captured["guidance_candidate_pool"] = config.guidance_candidate_pool
        captured["checkpoint_interval_s"] = kwargs["checkpoint_interval_s"]
        return Path("runs/fake-live.jsonl")

    monkeypatch.setattr(cli, "run_fuzz", fake_run_fuzz)
    monkeypatch.setattr(cli, "write_report", lambda run_file: (Path(""), Path("")))

    cli._run_experiment_job(
        {
            "order": 0,
            "target_suite": "datafusion_cross",
            "backends": ["pandas", "duckdb", "datafusion"],
            "preset": "live_datafusion",
            "seed": 1,
            "cases": 1,
            "duration_s": None,
            "log_level": "compact",
            "compress_run_log": True,
            "artifact_limit": None,
            "metamorphic_variant_limit": None,
            "enable_local_source_scheduler": False,
            "local_source_exploration_weight": 0.5,
            "skip_run_reports": True,
        }
    )

    assert captured["enable_local_source_scheduler"] is True
    assert captured["local_source_exploration_weight"] == 0.35
    assert captured["guidance_candidate_pool"] == 12
    assert captured["checkpoint_interval_s"] == 60.0


def test_run_experiment_job_propagates_replay_policy(monkeypatch):
    captured = {}

    def fake_run_fuzz(*, cases, seed, backends, config, duration_s, **kwargs):
        captured["enable_replay_bug"] = config.enable_replay_bug
        captured["replay_bug_source_issues"] = config.replay_bug_source_issues
        captured["checkpoint_interval_s"] = kwargs["checkpoint_interval_s"]
        return Path("runs/fake-replay.jsonl")

    monkeypatch.setattr(cli, "run_fuzz", fake_run_fuzz)
    monkeypatch.setattr(cli, "write_report", lambda run_file: (Path(""), Path("")))

    cli._run_experiment_job(
        {
            "order": 0,
            "target_suite": "datafusion_cross",
            "backends": ["pandas", "duckdb", "datafusion"],
            "preset": "live_datafusion",
            "seed": 1,
            "cases": 1,
            "duration_s": None,
            "log_level": "compact",
            "compress_run_log": True,
            "artifact_limit": None,
            "metamorphic_variant_limit": None,
            "enable_replay_bug": True,
            "replay_bug_source_issues": ["https://github.com/apache/datafusion/issues/22190"],
            "enable_local_source_scheduler": False,
            "local_source_exploration_weight": 0.5,
            "skip_run_reports": True,
        }
    )

    assert captured["enable_replay_bug"] is True
    assert captured["replay_bug_source_issues"] == ["https://github.com/apache/datafusion/issues/22190"]
    assert captured["checkpoint_interval_s"] == 60.0


def test_cli_experiment_static_manifest_records_local_source_scheduler(tmp_path, monkeypatch, capsys):
    runs_dir = tmp_path / "runs"
    reports_dir = tmp_path / "reports"
    bugs_dir = tmp_path / "bugs"
    corpus_dir = tmp_path / "corpus"
    monkeypatch.setattr(cli, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(cli, "REPORTS_DIR", reports_dir)
    monkeypatch.setattr(cli, "BUGS_DIR", bugs_dir)
    monkeypatch.setattr(cli, "CORPUS_DIR", corpus_dir)

    def fake_run_fuzz(*, cases, seed, backends, config, duration_s, **kwargs):
        run_file = runs_dir / "run-static.jsonl"
        append_jsonl(
            {
                "case": {"case_id": "case-0", "seed": seed},
                "is_new_behavior": False,
                "findings": [],
            },
            run_file,
        )
        meta_path = Path(str(run_file).replace(".jsonl", ".meta.json"))
        meta_path.write_text(
            json.dumps(
                {
                    "elapsed_s": 0.1,
                    "throughput_cases_s": 10.0,
                    "next_seed": seed + (cases or 1),
                }
            ),
            encoding="utf-8",
        )
        return run_file

    monkeypatch.setattr(cli, "run_fuzz", fake_run_fuzz)
    monkeypatch.setattr(cli, "write_report", lambda run_file: (Path(""), Path("")))

    parser = build_parser()
    args = parser.parse_args(
        [
            "experiment",
            "--target-suites",
            "core",
            "--presets",
            "baseline",
            "--seeds",
            "1",
            "--cases",
            "1",
            "--enable-local-source-scheduler",
            "--local-source-exploration-weight",
            "0.25",
            "--skip-run-reports",
        ]
    )

    assert args.func(args) == 0
    manifests = sorted(runs_dir.glob("experiment-*.json"))
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert manifest["schedule"] == "matrix_order"
    assert manifest["evidence_mode"] == "live"
    assert manifest["local_source_scheduler"]["enabled"] is True
    assert manifest["local_source_scheduler"]["exploration_weight"] == 0.25


def test_cli_experiment_manifest_records_structured_experiment_meta(tmp_path, monkeypatch):
    runs_dir = tmp_path / "runs"
    reports_dir = tmp_path / "reports"
    bugs_dir = tmp_path / "bugs"
    corpus_dir = tmp_path / "corpus"
    monkeypatch.setattr(cli, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(cli, "REPORTS_DIR", reports_dir)
    monkeypatch.setattr(cli, "BUGS_DIR", bugs_dir)
    monkeypatch.setattr(cli, "CORPUS_DIR", corpus_dir)

    def fake_run_fuzz(*, cases, seed, backends, config, duration_s, **kwargs):
        run_file = runs_dir / "run-meta.jsonl"
        append_jsonl(
            {
                "case": {"case_id": "case-0", "seed": seed},
                "is_new_behavior": False,
                "findings": [],
            },
            run_file,
        )
        run_meta_path(run_file).write_text(
            json.dumps({"elapsed_s": 0.1, "throughput_cases_s": 10.0, "next_seed": seed + 1}),
            encoding="utf-8",
        )
        return run_file

    monkeypatch.setattr(cli, "run_fuzz", fake_run_fuzz)
    monkeypatch.setattr(cli, "write_report", lambda run_file: (Path(""), Path("")))

    parser = build_parser()
    args = parser.parse_args(
        [
            "experiment",
            "--target-suites",
            "core",
            "--presets",
            "baseline",
            "--seeds",
            "1",
            "--cases",
            "1",
            "--experiment-meta",
            json.dumps(
                {
                    "matrix_id": "module_ablation",
                    "matrix_title": "Module Ablation",
                    "comparison_group": "module_ablation",
                    "rq_tags": ["RQ2", "RQ4"],
                    "analysis_tags": ["ablation", "noise_control"],
                    "counts_as_real_bugs": False,
                    "scope_by_target_suite": {"core": "core"},
                    "variant_by_preset": {
                        "baseline": {
                            "variant_id": "baseline",
                            "variant_title": "baseline",
                            "base_preset": "baseline",
                            "comparison_role": "baseline",
                            "component_focus": "",
                            "overlays": [],
                            "factors": {"type_aware_generation": True},
                            "oracle_profile": "differential",
                            "rq_tags": ["RQ2", "RQ4"],
                            "analysis_tags": ["ablation"],
                        }
                    },
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            "--skip-run-reports",
        ]
    )

    assert args.func(args) == 0
    manifest = json.loads(sorted(runs_dir.glob("experiment-*.json"))[0].read_text(encoding="utf-8"))
    assert manifest["experiment_meta"]["matrix_id"] == "module_ablation"
    assert manifest["experiment_meta"]["comparison_group"] == "module_ablation"
    run = manifest["runs"][0]
    assert run["matrix_id"] == "module_ablation"
    assert run["variant_id"] == "baseline"
    assert run["scope_kind"] == "core"
    assert run["oracle_profile"] == "differential"
    assert run["factors"] == {"type_aware_generation": True}
    assert run["comparison_role"] == "baseline"
    assert run["canonical_comparison_role"] == "baseline"
    assert run["component_focus"] == ""
    assert run["semantic_focus_families"] == []
    assert run["semantic_focus_signals"] == []
    assert run["analysis_tags"] == ["ablation", "noise_control"]


def test_cli_experiment_manifest_records_structured_semantic_focus_for_catalog_preset(
    tmp_path,
    monkeypatch,
):
    runs_dir = tmp_path / "runs"
    reports_dir = tmp_path / "reports"
    bugs_dir = tmp_path / "bugs"
    corpus_dir = tmp_path / "corpus"
    monkeypatch.setattr(cli, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(cli, "REPORTS_DIR", reports_dir)
    monkeypatch.setattr(cli, "BUGS_DIR", bugs_dir)
    monkeypatch.setattr(cli, "CORPUS_DIR", corpus_dir)

    def fake_run_fuzz(*, cases, seed, backends, config, duration_s, **kwargs):
        run_file = runs_dir / "run-semantic-focus.jsonl"
        append_jsonl(
            {
                "case": {"case_id": "case-0", "seed": seed},
                "is_new_behavior": False,
                "findings": [],
            },
            run_file,
        )
        run_meta_path(run_file).write_text(
            json.dumps(
                {
                    "elapsed_s": 0.1,
                    "throughput_cases_s": 10.0,
                    "next_seed": seed + 1,
                    "config": config.to_dict(),
                }
            ),
            encoding="utf-8",
        )
        return run_file

    monkeypatch.setattr(cli, "run_fuzz", fake_run_fuzz)
    monkeypatch.setattr(cli, "write_report", lambda run_file: (Path(""), Path("")))

    parser = build_parser()
    args = parser.parse_args(
        [
            "experiment",
            "--target-suites",
            "latest_no_datafusion",
            "--presets",
            "live_common_api_workflow_metamorphic",
            "--evidence-mode",
            "validation",
            "--seeds",
            "1",
            "--cases",
            "1",
            "--skip-run-reports",
        ]
    )

    assert args.func(args) == 0
    manifest = json.loads(sorted(runs_dir.glob("experiment-*.json"))[0].read_text(encoding="utf-8"))
    assert manifest["experiment_meta"]["matrix_id"] == "final_validation"
    run = manifest["runs"][0]
    preset_config = _preset_config("live_common_api_workflow_metamorphic")
    assert "materialization_boundary" in run["semantic_focus_families"]
    assert "common_api_workflow" in run["semantic_focus_signals"]
    assert run["configured_guidance_targets"] == list(preset_config.guidance_targets)
    assert run["configured_effective_guidance_targets"] == cli._configured_guidance_targets(preset_config)
    assert "materialization_boundary" in run["configured_semantic_focus_families"]
    assert "common_api_workflow" in run["configured_semantic_focus_signals"]


def test_cli_experiment_structured_variant_drives_run_config_from_base_and_overlays(
    tmp_path,
    monkeypatch,
):
    runs_dir = tmp_path / "runs"
    reports_dir = tmp_path / "reports"
    bugs_dir = tmp_path / "bugs"
    corpus_dir = tmp_path / "corpus"
    monkeypatch.setattr(cli, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(cli, "REPORTS_DIR", reports_dir)
    monkeypatch.setattr(cli, "BUGS_DIR", bugs_dir)
    monkeypatch.setattr(cli, "CORPUS_DIR", corpus_dir)

    captured: dict[str, object] = {}

    def fake_run_fuzz(*, cases, seed, backends, config, duration_s, **kwargs):
        captured["config"] = config.to_dict()
        run_file = runs_dir / "run-structured-config.jsonl"
        append_jsonl(
            {
                "case": {"case_id": "case-0", "seed": seed},
                "is_new_behavior": False,
                "findings": [],
            },
            run_file,
        )
        run_meta_path(run_file).write_text(
            json.dumps({"elapsed_s": 0.1, "throughput_cases_s": 10.0, "next_seed": seed + 1}),
            encoding="utf-8",
        )
        return run_file

    monkeypatch.setattr(cli, "run_fuzz", fake_run_fuzz)
    monkeypatch.setattr(cli, "write_report", lambda run_file: (Path(""), Path("")))

    parser = build_parser()
    args = parser.parse_args(
        [
            "experiment",
            "--target-suites",
            "core",
            "--presets",
            "custom_guided_join_variant",
            "--seeds",
            "1",
            "--cases",
            "1",
            "--experiment-meta",
            json.dumps(
                {
                    "matrix_id": "custom_matrix",
                    "comparison_group": "custom_group",
                    "variant_by_preset": {
                        "custom_guided_join_variant": {
                            "variant_id": "custom_guided_join_variant",
                            "variant_title": "custom_guided_join_variant",
                            "base_preset": "baseline",
                            "comparison_role": "contrast",
                            "overlays": ["enable_guidance", "target_join"],
                            "oracle_profile": "differential",
                        }
                    },
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            "--skip-run-reports",
        ]
    )

    assert args.func(args) == 0
    config = dict(captured["config"])
    assert config["guidance_strategy"] == "guided"
    assert config["generator_profile"] == "bughunt_no_groupby"
    assert config["guidance_targets"] == ["join", "sort_limit"]


def test_cli_experiment_adaptive_live_accepts_structured_guided_variant(
    tmp_path,
    monkeypatch,
):
    runs_dir = tmp_path / "runs"
    reports_dir = tmp_path / "reports"
    bugs_dir = tmp_path / "bugs"
    corpus_dir = tmp_path / "corpus"
    monkeypatch.setattr(cli, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(cli, "REPORTS_DIR", reports_dir)
    monkeypatch.setattr(cli, "BUGS_DIR", bugs_dir)
    monkeypatch.setattr(cli, "CORPUS_DIR", corpus_dir)

    def fake_run_fuzz(*, cases, seed, backends, config, duration_s, **kwargs):
        run_file = runs_dir / "run-adaptive-structured.jsonl"
        append_jsonl(
            {
                "case": {"case_id": "case-0", "seed": seed},
                "is_new_behavior": False,
                "findings": [],
            },
            run_file,
        )
        run_meta_path(run_file).write_text(
            json.dumps({"elapsed_s": 0.1, "throughput_cases_s": 10.0, "next_seed": seed + 1}),
            encoding="utf-8",
        )
        return run_file

    monkeypatch.setattr(cli, "run_fuzz", fake_run_fuzz)
    monkeypatch.setattr(cli, "write_report", lambda run_file: (Path(""), Path("")))

    parser = build_parser()
    args = parser.parse_args(
        [
            "experiment",
            "--target-suites",
            "core",
            "--presets",
            "custom_guided_join_variant",
            "--seeds",
            "1",
            "--cases",
            "1",
            "--schedule",
            "adaptive",
            "--batch-cases",
            "1",
            "--jobs",
            "1",
            "--experiment-meta",
            json.dumps(
                {
                    "matrix_id": "custom_matrix",
                    "comparison_group": "custom_group",
                    "variant_by_preset": {
                        "custom_guided_join_variant": {
                            "variant_id": "custom_guided_join_variant",
                            "variant_title": "custom_guided_join_variant",
                            "base_preset": "baseline",
                            "comparison_role": "contrast",
                            "overlays": ["enable_guidance", "target_join"],
                        }
                    },
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            "--skip-run-reports",
        ]
    )

    assert args.func(args) == 0
    manifest = json.loads(sorted(runs_dir.glob("experiment-*.json"))[0].read_text(encoding="utf-8"))
    assert manifest["schedule"] == "adaptive"


def test_cli_experiment_manifest_records_live_preset_source_scheduler(tmp_path, monkeypatch, capsys):
    runs_dir = tmp_path / "runs"
    reports_dir = tmp_path / "reports"
    bugs_dir = tmp_path / "bugs"
    corpus_dir = tmp_path / "corpus"
    monkeypatch.setattr(cli, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(cli, "REPORTS_DIR", reports_dir)
    monkeypatch.setattr(cli, "BUGS_DIR", bugs_dir)
    monkeypatch.setattr(cli, "CORPUS_DIR", corpus_dir)

    def fake_run_fuzz(*, cases, seed, backends, config, duration_s, **kwargs):
        run_file = runs_dir / "run-live.jsonl"
        append_jsonl(
            {
                "case": {"case_id": "case-0", "seed": seed},
                "is_new_behavior": False,
                "findings": [],
            },
            run_file,
        )
        run_meta_path(run_file).write_text(
            json.dumps({"elapsed_s": 0.1, "throughput_cases_s": 10.0, "next_seed": seed + 1}),
            encoding="utf-8",
        )
        return run_file

    monkeypatch.setattr(cli, "run_fuzz", fake_run_fuzz)
    monkeypatch.setattr(cli, "write_report", lambda run_file: (Path(""), Path("")))

    parser = build_parser()
    args = parser.parse_args(
        [
            "experiment",
            "--target-suite",
            "datafusion_cross",
            "--presets",
            "live_datafusion",
            "--seeds",
            "1",
            "--cases",
            "1",
            "--skip-run-reports",
        ]
    )

    assert args.func(args) == 0
    manifests = sorted(runs_dir.glob("experiment-*.json"))
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert manifest["local_source_scheduler"]["enabled"] is True
    assert manifest["local_source_scheduler"]["exploration_weight"] == 0.35


def test_cli_experiment_manifest_records_historical_evidence_metadata(tmp_path, monkeypatch, capsys):
    runs_dir = tmp_path / "runs"
    reports_dir = tmp_path / "reports"
    monkeypatch.setattr(cli, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(cli, "REPORTS_DIR", reports_dir)

    def fake_run_fuzz(*, cases, seed, backends, config, duration_s, **kwargs):
        run_file = runs_dir / "run-historical.jsonl"
        append_jsonl({"case": {"case_id": "case-0", "seed": seed}, "findings": []}, run_file)
        run_meta_path(run_file).write_text(
            json.dumps({"elapsed_s": 0.1, "throughput_cases_s": 10.0, "next_seed": seed + 1}),
            encoding="utf-8",
        )
        return run_file

    monkeypatch.setattr(cli, "run_fuzz", fake_run_fuzz)
    monkeypatch.setattr(cli, "write_report", lambda run_file: (Path(""), Path("")))

    parser = build_parser()
    args = parser.parse_args(
        [
            "experiment",
            "--target-suites",
            "core",
            "--presets",
            "baseline",
            "--seeds",
            "1",
            "--cases",
            "1",
            "--evidence-mode",
            "historical",
            "--known-bug-id",
            "datafusion-22190",
            "--target-version",
            "pre-fix-sha",
            "--skip-run-reports",
        ]
    )

    assert args.func(args) == 0
    manifest = json.loads(next(runs_dir.glob("experiment-*.json")).read_text(encoding="utf-8"))
    assert manifest["evidence_mode"] == "historical"
    assert manifest["known_bug_id"] == "datafusion-22190"
    assert manifest["target_version"] == "pre-fix-sha"
    assert manifest["experiment_meta"]["matrix_id"] == "historical_replay"
    assert manifest["experiment_meta"]["comparison_group"] == "historical_replay"
    assert manifest["experiment_meta"]["historical"]["bug_id"] == "datafusion-22190"
    assert manifest["experiment_meta"]["historical"]["status"] == "pending_merge"
    assert manifest["runs"][0]["evidence_mode"] == "historical"
    assert manifest["runs"][0]["matrix_id"] == "historical_replay"
    assert manifest["runs"][0]["variant_id"] == "datafusion-22190"
    assert manifest["runs"][0]["canonical_comparison_role"] == "contrast"
    assert "historical" in manifest["runs"][0]["analysis_tags"]


def test_cli_experiment_manifest_backfills_registered_validation_metadata(tmp_path, monkeypatch):
    runs_dir = tmp_path / "runs"
    reports_dir = tmp_path / "reports"
    monkeypatch.setattr(cli, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(cli, "REPORTS_DIR", reports_dir)

    def fake_run_fuzz(*, cases, seed, backends, config, duration_s, **kwargs):
        run_file = runs_dir / "run-validation.jsonl"
        append_jsonl({"case": {"case_id": "case-0", "seed": seed}, "findings": []}, run_file)
        run_meta_path(run_file).write_text(
            json.dumps({"elapsed_s": 0.1, "throughput_cases_s": 10.0, "next_seed": seed + 1}),
            encoding="utf-8",
        )
        return run_file

    monkeypatch.setattr(cli, "run_fuzz", fake_run_fuzz)
    monkeypatch.setattr(cli, "write_report", lambda run_file: (Path(""), Path("")))

    parser = build_parser()
    args = parser.parse_args(
        [
            "experiment",
            "--target-suite",
            "datafusion_cross",
            "--presets",
            "live_deep_organic_metamorphic",
            "--seeds",
            "1",
            "--cases",
            "1",
            "--evidence-mode",
            "validation",
            "--skip-run-reports",
        ]
    )

    assert args.func(args) == 0
    manifest = json.loads(next(runs_dir.glob("experiment-*.json")).read_text(encoding="utf-8"))
    assert manifest["experiment_meta"]["matrix_id"] == "final_validation"
    assert manifest["experiment_meta"]["comparison_group"] == "validation_smoke"
    assert manifest["experiment_meta"]["target_suites"] == ["datafusion_cross"]
    assert manifest["runs"][0]["matrix_id"] == "final_validation"
    assert manifest["runs"][0]["variant_id"] == "live_deep_organic_metamorphic"
    assert manifest["runs"][0]["canonical_comparison_role"] == "contrast"
    assert "validation" in manifest["runs"][0]["analysis_tags"]


def test_cli_experiment_manifest_restores_registered_variant_catalog_for_comparison_matrix(tmp_path, monkeypatch):
    runs_dir = tmp_path / "runs"
    reports_dir = tmp_path / "reports"
    monkeypatch.setattr(cli, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(cli, "REPORTS_DIR", reports_dir)

    def fake_run_fuzz(*, cases, seed, backends, config, duration_s, **kwargs):
        run_file = runs_dir / f"run-{seed}.jsonl"
        append_jsonl({"case": {"case_id": f"case-{seed}", "seed": seed}, "findings": []}, run_file)
        run_meta_path(run_file).write_text(
            json.dumps({"elapsed_s": 0.1, "throughput_cases_s": 10.0, "next_seed": seed + 1}),
            encoding="utf-8",
        )
        return run_file

    monkeypatch.setattr(cli, "run_fuzz", fake_run_fuzz)
    monkeypatch.setattr(cli, "write_report", lambda run_file: (Path(""), Path("")))

    parser = build_parser()
    args = parser.parse_args(
        [
            "experiment",
            "--target-suite",
            "latest_no_datafusion",
            "--presets",
            "baseline,live_cross_family",
            "--seeds",
            "1",
            "--cases",
            "1",
            "--evidence-mode",
            "comparison",
            "--experiment-meta",
            json.dumps(
                {
                    "matrix_id": "baseline_scope_comparison",
                    "comparison_group": "scope_comparison",
                    "analysis_tags": ["comparison", "scope"],
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            "--skip-run-reports",
        ]
    )

    assert args.func(args) == 0
    manifest = json.loads(next(runs_dir.glob("experiment-*.json")).read_text(encoding="utf-8"))
    variant_by_preset = manifest["experiment_meta"]["variant_by_preset"]
    assert "baseline" in variant_by_preset
    assert "live_cross_family" in variant_by_preset
    baseline_run = next(run for run in manifest["runs"] if run["preset"] == "baseline")
    contrast_run = next(run for run in manifest["runs"] if run["preset"] == "live_cross_family")
    assert baseline_run["matrix_id"] == "baseline_scope_comparison"
    assert baseline_run["canonical_comparison_role"] == "baseline"
    assert contrast_run["variant_id"] == "live_cross_family"
    assert contrast_run["comparison_role"] == "contrast"
    assert contrast_run["scope_kind"] != ""


def test_cli_experiment_manifest_keeps_registered_identity_under_partial_variant_override(
    tmp_path,
    monkeypatch,
):
    runs_dir = tmp_path / "runs"
    reports_dir = tmp_path / "reports"
    monkeypatch.setattr(cli, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(cli, "REPORTS_DIR", reports_dir)

    def fake_run_fuzz(*, cases, seed, backends, config, duration_s, **kwargs):
        run_file = runs_dir / "run-partial-override.jsonl"
        append_jsonl({"case": {"case_id": "case-0", "seed": seed}, "findings": []}, run_file)
        run_meta_path(run_file).write_text(
            json.dumps(
                {
                    "elapsed_s": 0.1,
                    "throughput_cases_s": 10.0,
                    "next_seed": seed + 1,
                    "config": config.to_dict(),
                }
            ),
            encoding="utf-8",
        )
        return run_file

    monkeypatch.setattr(cli, "run_fuzz", fake_run_fuzz)
    monkeypatch.setattr(cli, "write_report", lambda run_file: (Path(""), Path("")))

    parser = build_parser()
    args = parser.parse_args(
        [
            "experiment",
            "--target-suite",
            "datafusion_cross",
            "--presets",
            "live_common_api_workflow_metamorphic",
            "--seeds",
            "1",
            "--cases",
            "1",
            "--evidence-mode",
            "validation",
            "--experiment-meta",
            json.dumps(
                {
                    "variant_by_preset": {
                        "live_common_api_workflow_metamorphic": {
                            "semantic_focus_families": ["wrong_family"],
                            "semantic_focus_signals": ["wrong_signal"],
                        }
                    }
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            "--skip-run-reports",
        ]
    )

    assert args.func(args) == 0
    manifest = json.loads(next(runs_dir.glob("experiment-*.json")).read_text(encoding="utf-8"))
    assert manifest["experiment_meta"]["matrix_id"] == "final_validation"
    assert manifest["experiment_meta"]["comparison_group"] == "validation_smoke"
    variant = manifest["experiment_meta"]["variant_by_preset"]["live_common_api_workflow_metamorphic"]
    assert variant["variant_id"] == "live_common_api_workflow_metamorphic"
    assert variant["base_preset"] == "live_common_api_workflow"
    assert variant["semantic_focus_families"] == ["wrong_family"]
    assert variant["semantic_focus_signals"] == ["wrong_signal"]

    run = manifest["runs"][0]
    assert run["matrix_id"] == "final_validation"
    assert run["variant_id"] == "live_common_api_workflow_metamorphic"
    assert run["scope_kind"] == "cross_ecosystem"
    assert run["oracle_profile"] == "both"
    assert run["semantic_focus_families"] == ["wrong_family"]
    assert run["semantic_focus_signals"] == ["wrong_signal"]
    expected_config = _preset_config("live_common_api_workflow_metamorphic")
    assert run["configured_guidance_targets"] == list(expected_config.guidance_targets)
    assert run["configured_effective_guidance_targets"] == cli._configured_guidance_targets(expected_config)


def test_cli_replay_fixture_records_single_case_run_and_journal(tmp_path, monkeypatch, capsys):
    pyarrow = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    from datadiff.fixture_replay import fixture_sha256

    runs_dir = tmp_path / "runs"
    reports_dir = tmp_path / "reports"
    monkeypatch.setattr(cli, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(cli, "REPORTS_DIR", reports_dir)
    monkeypatch.setattr(cli, "BUGS_DIR", tmp_path / "bugs")
    monkeypatch.setattr(cli, "CORPUS_DIR", tmp_path / "corpus")

    fixture_path = tmp_path / "fixture.parquet"
    table = pyarrow.table(
        {
            "a": pyarrow.array([None, 2, 1], type=pyarrow.int64()),
            "b": pyarrow.array(["n", "z", "a"], type=pyarrow.string()),
        }
    )
    pq.write_table(table, fixture_path)
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(
        json.dumps(
            {
                "case_id": "case-cli-fixture",
                "seed": 77,
                "target_suite": "cross_family",
                "fixture": {
                    "name": "declared",
                    "columns": ["a", "b"],
                    "sha256": fixture_sha256(fixture_path),
                },
                "operations": [
                    {
                        "op": "sort",
                        "keys": [
                            {"column": "a", "ascending": True, "nulls": "first"},
                            {"column": "b", "ascending": True, "nulls": "last"},
                        ],
                    },
                    {"op": "limit", "n": 2},
                ],
            }
        ),
        encoding="utf-8",
    )

    parser = build_parser()
    args = parser.parse_args(
        [
            "replay-fixture",
            "--spec",
            str(spec_path),
            "--fixture",
            str(fixture_path),
            "--backends",
            "pandas",
            "--known-bug-id",
            "fixture-test",
            "--target-version",
            "target==1.0",
            "--artifact-limit",
            "0",
            "--no-compress-run-log",
        ]
    )

    assert args.func(args) == 0
    run_file = next(runs_dir.glob("run-fixture-*.jsonl"))
    meta = json.loads(run_meta_path(run_file).read_text(encoding="utf-8"))
    journal = (reports_dir / "paper-run-journal.jsonl").read_text(encoding="utf-8")
    output = capsys.readouterr().out
    assert "status=ok" in output
    assert meta["preset"] == "fixture_replay"
    assert meta["evidence_mode"] == "historical"
    assert meta["known_bug_id"] == "fixture-test"
    assert meta["fixture_sha256"] == fixture_sha256(fixture_path)
    assert meta["config"]["enable_replay_bug"] is True
    assert meta["experiment_meta"]["matrix_id"] == "historical_replay"
    assert meta["experiment_meta"]["historical"]["bug_id"] == "fixture-test"
    assert meta["experiment_meta"]["variant"]["variant_id"] == "fixture-test"
    assert '"known_bug_id": "fixture-test"' in journal


def test_cli_historical_status_marks_counted_and_pending(capsys):
    parser = build_parser()
    args = parser.parse_args(["historical-status", "--include-pending", "--json"])

    assert args.func(args) == 0
    payload = json.loads(capsys.readouterr().out)
    by_id = {row["bug_id"]: row for row in payload["historical_bugs"]}
    assert by_id["duckdb-22075"]["counted"] is True
    assert by_id["duckdb-22656"]["counted"] is True
    assert by_id["duckdb-22656"]["target_suite"] == "duckdb_storage_cross"
    assert by_id["duckdb-3015"]["counted"] is False
    assert by_id["duckdb-11261"]["counted"] is False
    assert by_id["duckdb-11261"]["target_suite"] == "duckdb_storage_cross"
    assert by_id["arrow-42231"]["counted"] is False
    assert by_id["arrow-42231"]["target_suite"] == "arrow_cross"
    assert by_id["duckdb-3015"]["replay_kind"] == "fixture"
    assert by_id["duckdb-3015"]["fixture_env_status"] in {"set", "unset"}


def test_cli_experiment_manifest_names_do_not_collide_within_same_second(tmp_path, monkeypatch):
    runs_dir = tmp_path / "runs"
    reports_dir = tmp_path / "reports"
    bugs_dir = tmp_path / "bugs"
    corpus_dir = tmp_path / "corpus"
    monkeypatch.setattr(cli, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(cli, "REPORTS_DIR", reports_dir)
    monkeypatch.setattr(cli, "BUGS_DIR", bugs_dir)
    monkeypatch.setattr(cli, "CORPUS_DIR", corpus_dir)
    monkeypatch.setattr(cli, "utc_now", lambda: "2026-05-19T15:42:27Z")
    ticks = iter([111, 222])
    monkeypatch.setattr(cli.time, "time_ns", lambda: next(ticks))

    def fake_run_fuzz(*, cases, seed, backends, config, duration_s, **kwargs):
        run_file = runs_dir / f"run-{seed}.jsonl"
        append_jsonl(
            {
                "case": {"case_id": f"case-{seed}", "seed": seed},
                "is_new_behavior": False,
                "findings": [],
            },
            run_file,
        )
        meta_path = Path(str(run_file).replace(".jsonl", ".meta.json"))
        meta_path.write_text(
            json.dumps(
                {
                    "elapsed_s": 0.1,
                    "throughput_cases_s": 10.0,
                    "next_seed": seed + (cases or 1),
                }
            ),
            encoding="utf-8",
        )
        return run_file

    monkeypatch.setattr(cli, "run_fuzz", fake_run_fuzz)
    monkeypatch.setattr(cli, "write_report", lambda run_file: (Path(""), Path("")))

    parser = build_parser()
    argv = [
        "experiment",
        "--target-suites",
        "core",
        "--presets",
        "baseline",
        "--seeds",
        "1",
        "--cases",
        "1",
        "--skip-run-reports",
    ]

    args = parser.parse_args(argv)
    assert args.func(args) == 0
    args = parser.parse_args(argv)
    assert args.func(args) == 0

    manifests = sorted(runs_dir.glob("experiment-*.json"))
    assert [path.name for path in manifests] == [
        "experiment-20260519T154227-111.json",
        "experiment-20260519T154227-222.json",
    ]


def test_experiment_parallel_scheduler_starts_heavy_jobs_first():
    fast = {
        "order": 0,
        "preset": "null_groupby_topk",
        "backends": ["pandas", "duckdb", "datafusion"],
    }
    slow = {
        "order": 1,
        "preset": "float_group_key_metamorphic",
        "backends": ["pandas", "polars", "polars_lazy", "duckdb", "sqlite", "datafusion"],
    }

    assert sorted([fast, slow], key=cli._experiment_job_sort_key) == [slow, fast]


def test_experiment_parallelism_auto_uses_bounded_workers(monkeypatch):
    monkeypatch.setattr(cli.os, "cpu_count", lambda: 20)
    parser = build_parser()
    args = parser.parse_args(["experiment", "--jobs", "auto"])
    planned = [
        {"order": 0, "preset": "live_cross_family", "backends": ["pandas", "duckdb", "datafusion"]},
        {"order": 1, "preset": "live_polars_lazy", "backends": ["polars", "polars_lazy"]},
        {"order": 2, "preset": "live_arrow", "backends": ["pandas", "duckdb", "pyarrow"]},
        {"order": 3, "preset": "baseline", "backends": ["pandas", "duckdb"]},
        {"order": 4, "preset": "baseline", "backends": ["pandas", "sqlite"]},
    ]

    parallelism = cli._resolve_experiment_parallelism(args, planned)

    assert parallelism["worker_count"] == 5
    assert parallelism["bounded_submission"] is True
    assert parallelism["cost_limited"] is True
    assert parallelism["worker_thread_limit"] == 4


def test_cli_experiment_adaptive_scheduler_reuses_budget_on_high_yield_arm(tmp_path, monkeypatch, capsys):
    runs_dir = tmp_path / "runs"
    reports_dir = tmp_path / "reports"
    bugs_dir = tmp_path / "bugs"
    corpus_dir = tmp_path / "corpus"
    monkeypatch.setattr(cli, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(cli, "REPORTS_DIR", reports_dir)
    monkeypatch.setattr(cli, "BUGS_DIR", bugs_dir)
    monkeypatch.setattr(cli, "CORPUS_DIR", corpus_dir)

    counter = {"index": 0}

    def fake_run_fuzz(
        *,
        cases,
        seed,
        backends,
        config,
        duration_s,
        persist_closed_loop_state=False,
        closed_loop_state=None,
        **kwargs,
    ):
        idx = counter["index"]
        counter["index"] += 1
        assert config.enable_local_source_scheduler is True
        assert config.local_source_exploration_weight == 0.125
        assert persist_closed_loop_state is True
        run_file = runs_dir / f"run-{idx}.jsonl"
        target_suite = "datafusion_cross" if "datafusion" in backends else "core"
        if target_suite == "datafusion_cross":
            append_jsonl(
                {
                    "case": {"case_id": f"case-{idx}", "seed": seed},
                    "is_new_behavior": True,
                    "findings": [
                        {
                            "root_cause": "grouped_topk_null_sort_key",
                            "triage_verdict": "candidate_implementation_bug",
                            "suspicious_backends": ["datafusion"],
                            "signature": f"sig-{idx}",
                        }
                    ],
                },
                run_file,
            )
        else:
            append_jsonl(
                {
                    "case": {"case_id": f"case-{idx}", "seed": seed},
                    "is_new_behavior": False,
                    "findings": [],
                },
                run_file,
            )
        meta_path = Path(str(run_file).replace(".jsonl", ".meta.json"))
        meta_path.write_text(
            json.dumps(
                {
                    "elapsed_s": 0.1,
                    "throughput_cases_s": 10.0,
                    "next_seed": seed + cases,
                }
            ),
            encoding="utf-8",
        )
        return run_file

    monkeypatch.setattr(cli, "run_fuzz", fake_run_fuzz)
    monkeypatch.setattr(cli, "write_report", lambda run_file: (Path(""), Path("")))

    parser = build_parser()
    args = parser.parse_args(
        [
            "experiment",
            "--target-suites",
            "core,datafusion_cross",
            "--presets",
            "live_deep_organic",
            "--seeds",
            "1",
            "--cases",
            "2",
            "--schedule",
            "adaptive",
            "--batch-cases",
            "1",
            "--jobs",
            "1",
            "--local-source-exploration-weight",
            "0.125",
            "--skip-run-reports",
        ]
    )

    assert args.func(args) == 0
    out = capsys.readouterr().out
    assert "reward=" in out

    manifests = sorted(runs_dir.glob("experiment-*.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert manifest["schedule"] == "adaptive"
    assert manifest["adaptive_config"]["fine_grained_local_source_scheduler"] is True
    assert manifest["adaptive_config"]["local_source_exploration_weight"] == 0.125
    assert manifest["adaptive_config"]["group_fairness_weight"] == 0.4
    assert manifest["adaptive_config"]["max_group_pull_gap"] == 3
    assert manifest["adaptive_config"]["prefer_group_diversity_in_round"] is True
    suites = [run["target_suite"] for run in manifest["runs"]]
    assert suites.count("datafusion_cross") >= suites.count("core")


def test_cli_experiment_adaptive_scheduler_supports_parallel_rounds(tmp_path, monkeypatch, capsys):
    runs_dir = tmp_path / "runs"
    reports_dir = tmp_path / "reports"
    bugs_dir = tmp_path / "bugs"
    corpus_dir = tmp_path / "corpus"
    monkeypatch.setattr(cli, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(cli, "REPORTS_DIR", reports_dir)
    monkeypatch.setattr(cli, "BUGS_DIR", bugs_dir)
    monkeypatch.setattr(cli, "CORPUS_DIR", corpus_dir)
    monkeypatch.setattr(cli, "ProcessPoolExecutor", cli.ThreadPoolExecutor)

    counter = {"index": 0}

    def fake_run_fuzz(
        *,
        cases,
        seed,
        backends,
        config,
        duration_s,
        persist_closed_loop_state=False,
        closed_loop_state=None,
        **kwargs,
    ):
        idx = counter["index"]
        counter["index"] += 1
        assert persist_closed_loop_state is True
        run_file = runs_dir / f"run-{idx}.jsonl"
        target_suite = (
            "datafusion_cross"
            if "datafusion" in backends
            else "dataframe"
            if "polars" in backends
            else "core"
        )
        if target_suite == "datafusion_cross":
            payload = {
                "case": {"case_id": f"case-{idx}", "seed": seed},
                "is_new_behavior": True,
                "findings": [
                    {
                        "root_cause": "grouped_topk_null_sort_key",
                        "triage_verdict": "candidate_implementation_bug",
                        "suspicious_backends": ["datafusion"],
                        "signature": f"sig-{idx}",
                    }
                ],
            }
        elif target_suite == "dataframe":
            payload = {
                "case": {"case_id": f"case-{idx}", "seed": seed},
                "is_new_behavior": True,
                "findings": [],
            }
        else:
            payload = {
                "case": {"case_id": f"case-{idx}", "seed": seed},
                "is_new_behavior": False,
                "findings": [],
            }
        append_jsonl(payload, run_file)
        meta_path = Path(str(run_file).replace(".jsonl", ".meta.json"))
        meta_path.write_text(
            json.dumps(
                {
                    "elapsed_s": 0.1,
                    "throughput_cases_s": 10.0,
                    "next_seed": seed + cases,
                }
            ),
            encoding="utf-8",
        )
        return run_file

    monkeypatch.setattr(cli, "run_fuzz", fake_run_fuzz)
    monkeypatch.setattr(cli, "write_report", lambda run_file: (Path(""), Path("")))

    parser = build_parser()
    args = parser.parse_args(
        [
            "experiment",
            "--target-suites",
            "core,dataframe,datafusion_cross",
            "--presets",
            "live_deep_organic",
            "--seeds",
            "1",
            "--cases",
            "2",
            "--schedule",
            "adaptive",
            "--batch-cases",
            "1",
            "--jobs",
            "2",
            "--skip-run-reports",
        ]
    )

    assert args.func(args) == 0
    out = capsys.readouterr().out
    assert "reward=" in out

    manifests = sorted(runs_dir.glob("experiment-*.json"))
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert manifest["adaptive_config"]["jobs"] == 2
    suites = [run["target_suite"] for run in manifest["runs"]]
    assert suites.count("datafusion_cross") >= 2
    state_by_suite = {row["target_suite"]: row for row in manifest["adaptive_state"]}
    assert state_by_suite["datafusion_cross"]["pulls"] >= 2
    assert state_by_suite["datafusion_cross"]["reward_signal"] > state_by_suite["core"]["reward_signal"]
    assert any(run["scheduler_reward"] > 0 for run in manifest["runs"])


def test_cli_experiment_adaptive_scheduler_passes_closed_loop_state_between_batches(
    tmp_path,
    monkeypatch,
    capsys,
):
    runs_dir = tmp_path / "runs"
    reports_dir = tmp_path / "reports"
    bugs_dir = tmp_path / "bugs"
    corpus_dir = tmp_path / "corpus"
    monkeypatch.setattr(cli, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(cli, "REPORTS_DIR", reports_dir)
    monkeypatch.setattr(cli, "BUGS_DIR", bugs_dir)
    monkeypatch.setattr(cli, "CORPUS_DIR", corpus_dir)

    seen_states: list[dict[str, object]] = []
    counter = {"index": 0}

    def fake_run_fuzz(
        *,
        cases,
        seed,
        backends,
        config,
        duration_s,
        persist_closed_loop_state=False,
        closed_loop_state=None,
        **kwargs,
    ):
        idx = counter["index"]
        counter["index"] += 1
        seen_states.append(
            {
                "persist_closed_loop_state": persist_closed_loop_state,
                "closed_loop_state": closed_loop_state,
            }
        )
        run_file = runs_dir / f"run-state-{idx}.jsonl"
        append_jsonl(
            {
                "case": {"case_id": f"case-{idx}", "seed": seed},
                "is_new_behavior": idx == 0,
                "findings": [],
            },
            run_file,
        )
        state_path = closed_loop_state_path(run_file)
        dump_json(
            {
                "seen_signatures": ["discovery-shared"],
                "feedback": None,
                "source_scheduler": None,
                "guidance": None,
            },
            state_path,
            compact=True,
        )
        run_meta_path(run_file).write_text(
            json.dumps(
                {
                    "elapsed_s": 0.1,
                    "throughput_cases_s": 10.0,
                    "next_seed": seed + cases,
                    "closed_loop_state_file": str(state_path),
                    "closed_loop_state_summary": {"seen_signature_count": 1},
                }
            ),
            encoding="utf-8",
        )
        return run_file

    monkeypatch.setattr(cli, "run_fuzz", fake_run_fuzz)
    monkeypatch.setattr(cli, "write_report", lambda run_file: (Path(""), Path("")))

    parser = build_parser()
    args = parser.parse_args(
        [
            "experiment",
            "--target-suites",
            "core",
            "--presets",
            "live_deep_organic",
            "--seeds",
            "1",
            "--cases",
            "2",
            "--schedule",
            "adaptive",
            "--batch-cases",
            "1",
            "--jobs",
            "1",
            "--skip-run-reports",
        ]
    )

    assert args.func(args) == 0
    assert "reward=" in capsys.readouterr().out
    assert len(seen_states) == 2
    assert seen_states[0]["persist_closed_loop_state"] is True
    assert seen_states[0]["closed_loop_state"] is None
    assert seen_states[1]["persist_closed_loop_state"] is True
    assert seen_states[1]["closed_loop_state"] == {
        "seen_signatures": ["discovery-shared"],
        "feedback": None,
        "source_scheduler": None,
        "guidance": None,
    }

    manifest = json.loads(next(runs_dir.glob("experiment-*.json")).read_text(encoding="utf-8"))
    assert [run["closed_loop_state_present"] for run in manifest["runs"]] == [True, True]


def test_cli_experiment_rejects_live_adaptive_baseline_preset(tmp_path, monkeypatch, capsys):
    runs_dir = tmp_path / "runs"
    reports_dir = tmp_path / "reports"
    bugs_dir = tmp_path / "bugs"
    corpus_dir = tmp_path / "corpus"
    monkeypatch.setattr(cli, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(cli, "REPORTS_DIR", reports_dir)
    monkeypatch.setattr(cli, "BUGS_DIR", bugs_dir)
    monkeypatch.setattr(cli, "CORPUS_DIR", corpus_dir)

    called = {"run_fuzz": False}

    def fake_run_fuzz(*, cases, seed, backends, config, duration_s, **kwargs):
        called["run_fuzz"] = True
        raise AssertionError("run_fuzz should not be called for invalid live adaptive baseline experiments")

    monkeypatch.setattr(cli, "run_fuzz", fake_run_fuzz)

    parser = build_parser()
    args = parser.parse_args(
        [
            "experiment",
            "--target-suites",
            "core,datafusion_cross",
            "--presets",
            "baseline",
            "--seeds",
            "1",
            "--cases",
            "2",
            "--schedule",
            "adaptive",
            "--batch-cases",
            "1",
            "--jobs",
            "1",
            "--skip-run-reports",
        ]
    )

    assert args.func(args) == 2
    assert called["run_fuzz"] is False
    assert "adaptive live experiments require guided presets" in capsys.readouterr().out


def test_cli_experiment_summary_parses_refresh():
    parser = build_parser()
    args = parser.parse_args(["experiment-summary", "--refresh"])
    assert args.cmd == "experiment-summary"
    assert args.refresh is True


def test_cli_parses_longrun_defaults():
    parser = build_parser()
    args = parser.parse_args(["longrun"])
    assert args.cmd == "longrun"
    assert args.duration == "24h"
    assert args.cases is None
    assert args.strategy == "guided"
    assert args.candidate_pool == 8
    assert args.checkpoint_interval == "60s"
    assert args.progress_interval == "60s"
    assert args.save_cases is False
    assert args.no_save_cases is False
    assert args.log_level == "compact"


def test_cli_parses_experiment_summary_command():
    parser = build_parser()
    args = parser.parse_args(["experiment-summary", "--manifest", "runs/experiment-x.json"])
    assert args.cmd == "experiment-summary"
    assert args.manifest == "runs/experiment-x.json"


def test_cli_parses_report_csv_limit():
    parser = build_parser()
    args = parser.parse_args(["report", "--run-file", "runs/run-x.jsonl.gz", "--csv-limit", "100"])
    assert args.cmd == "report"
    assert args.csv_limit == 100


def test_cli_parses_artifact_validation_command():
    parser = build_parser()
    args = parser.parse_args(["validate-artifact", "--bug", "bugs/bug_x"])
    assert args.cmd == "validate-artifact"
    assert args.bug == "bugs/bug_x"


def test_cli_parses_classify_run_command():
    parser = build_parser()
    args = parser.parse_args(["classify-run", "--run-file", "runs/run-x.jsonl", "--limit", "2", "--json"])
    assert args.cmd == "classify-run"
    assert args.run_file == "runs/run-x.jsonl"
    assert args.limit == 2
    assert args.json is True


def test_cli_classify_run_reads_compressed_jsonl(tmp_path, capsys):
    run_file = tmp_path / "run-x.jsonl.gz"
    append_jsonl({"case": {"case_id": "case-x", "seed": 1}, "findings": []}, run_file)

    parser = build_parser()
    args = parser.parse_args(["classify-run", "--run-file", str(run_file), "--limit", "2"])

    assert args.func(args) == 0
    out = capsys.readouterr().out
    assert f"run_file={run_file}" in out
    assert "- none" in out


def test_cli_classify_run_reports_candidate_bug_families(tmp_path, capsys):
    run_file = tmp_path / "run-family.jsonl.gz"
    append_jsonl(
        {
            "case": {"case_id": "case-x", "seed": 1},
            "findings": [
                {
                    "kind": "semantic_output_mismatch",
                    "root_cause": "grouped_topk_null_sort_key",
                    "triage_verdict": "candidate_implementation_bug",
                    "suspicious_backends": ["datafusion"],
                    "signature": "sig-x",
                }
            ],
        },
        run_file,
    )

    parser = build_parser()
    args = parser.parse_args(["classify-run", "--run-file", str(run_file), "--limit", "1"])

    assert args.func(args) == 0
    out = capsys.readouterr().out
    assert "candidate bug families:" in out
    assert "- grouped_topk_null_sort_key@datafusion: 1" in out


def test_cli_classify_run_splits_fresh_and_known_saturated_families(tmp_path, capsys):
    run_file = tmp_path / "run-family-split.jsonl.gz"
    append_jsonl(
        {
            "case": {"case_id": "case-known", "seed": 1},
            "config": {"known_saturated_bug_families": ["known_family@duckdb"]},
            "findings": [
                {
                    "kind": "semantic_output_mismatch",
                    "root_cause": "known_family",
                    "triage_verdict": "candidate_implementation_bug",
                    "suspicious_backends": ["duckdb"],
                    "signature": "sig-known",
                },
                {
                    "kind": "semantic_output_mismatch",
                    "root_cause": "csv_long_numeric_roundtrip",
                    "triage_verdict": "candidate_implementation_bug",
                    "suspicious_backends": ["duckdb", "pyarrow"],
                    "signature": "sig-known-combined",
                },
                {
                    "kind": "semantic_output_mismatch",
                    "root_cause": "fresh_family",
                    "triage_verdict": "candidate_implementation_bug",
                    "suspicious_backends": ["polars"],
                    "signature": "sig-fresh",
                },
                {
                    "kind": "semantic_output_mismatch",
                    "root_cause": "inspired_family",
                    "triage_verdict": "candidate_implementation_bug",
                    "suspicious_backends": ["pyarrow"],
                    "signature": "sig-inspired",
                    "discovery_origin": "issue_inspired",
                    "source_issue": "https://github.com/example/project/issues/1",
                },
            ],
        },
        run_file,
    )

    parser = build_parser()
    args = parser.parse_args(["classify-run", "--run-file", str(run_file), "--limit", "1"])

    assert args.func(args) == 0
    out = capsys.readouterr().out
    assert "offline paper buckets:" in out
    assert "- new_bug: 1" in out
    assert "- known_bug: 3" in out
    assert "fresh candidate bug families:" in out
    assert "- fresh_family@polars: 1" in out
    assert "issue-inspired unsaturated candidate bug families:" in out
    assert "- inspired_family@pyarrow: 1" in out
    assert "known saturated candidate bug families:" in out
    assert "- known_family@duckdb: 1" in out
    assert "- csv_long_numeric_roundtrip@duckdb,pyarrow: 1" in out


def test_cli_classify_run_can_emit_json_with_offline_buckets(tmp_path, capsys):
    run_file = tmp_path / "run-classify-json.jsonl.gz"
    append_jsonl(
        {
            "case": {"case_id": "case-json", "seed": 1, "metadata": {"source_issue": "https://example.test/1"}},
            "findings": [
                {
                    "kind": "semantic_output_mismatch",
                    "root_cause": "source_backed_family",
                    "triage_verdict": "candidate_implementation_bug",
                    "suspicious_backends": ["pyarrow"],
                    "signature": "sig-source",
                },
                {
                    "kind": "semantic_output_mismatch",
                    "root_cause": "unicode_case_mapping",
                    "triage_verdict": "expected_semantic_divergence",
                    "suspicious_backends": ["sqlite"],
                    "signature": "sig-semantic",
                },
            ],
        },
        run_file,
    )

    parser = build_parser()
    args = parser.parse_args(["classify-run", "--run-file", str(run_file), "--json"])

    assert args.func(args) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["offline_buckets"] == {
        "known_bug": 1,
        "semantic_divergence": 1,
    }


def test_cli_parses_run_health_command():
    parser = build_parser()
    args = parser.parse_args(
        [
            "run-health",
            "--run-file",
            "runs/run-x.jsonl.gz",
            "--limit",
            "2",
            "--json",
            "--fail-on-fresh-candidate",
            "--fail-on-bug",
        ]
    )
    assert args.cmd == "run-health"
    assert args.run_file == "runs/run-x.jsonl.gz"
    assert args.limit == 2
    assert args.json is True
    assert args.fail_on_fresh_candidate is True
    assert args.fail_on_bug is True


def test_cli_run_health_summarizes_compressed_run(tmp_path, capsys):
    run_file = tmp_path / "run-health.jsonl.gz"
    append_jsonl({"case": {"case_id": "case-ok", "seed": 1}, "status": "ok", "findings": []}, run_file)
    append_jsonl(
        {
            "case": {"case_id": "case-bug", "seed": 2},
            "status": "bug",
            "findings": [
                {
                    "kind": "semantic_output_mismatch",
                    "root_cause": "fresh_family",
                    "triage_verdict": "candidate_implementation_bug",
                    "suspicious_backends": ["duckdb"],
                }
            ],
        },
        run_file,
    )

    parser = build_parser()
    args = parser.parse_args(["run-health", "--run-file", str(run_file), "--limit", "1"])

    assert args.func(args) == 0
    out = capsys.readouterr().out
    assert f"run_file={run_file}" in out
    assert "partial=false" in out
    assert "rows=2" in out
    assert "- ok: 1" in out
    assert "- bug: 1" in out
    assert "- fresh_family@duckdb: 1" in out
    assert "case-bug" in out


def test_cli_run_health_can_fail_on_bug_or_fresh_candidate(tmp_path, capsys):
    run_file = tmp_path / "run-health-fail.jsonl.gz"
    append_jsonl(
        {
            "case": {"case_id": "case-bug", "seed": 2},
            "status": "bug",
            "findings": [
                {
                    "kind": "semantic_output_mismatch",
                    "root_cause": "fresh_family",
                    "triage_verdict": "candidate_implementation_bug",
                    "suspicious_backends": ["duckdb"],
                }
            ],
        },
        run_file,
    )

    parser = build_parser()
    args = parser.parse_args(["run-health", "--run-file", str(run_file), "--fail-on-bug"])
    assert args.func(args) == 2
    capsys.readouterr()

    args = parser.parse_args(["run-health", "--run-file", str(run_file), "--fail-on-fresh-candidate"])
    assert args.func(args) == 2


def test_cli_run_health_tolerates_in_progress_gzip(tmp_path, capsys):
    run_file = tmp_path / "run-health-partial.jsonl.gz"
    append_jsonl({"case": {"case_id": "case-ok", "seed": 1}, "status": "ok", "findings": []}, run_file)
    data = run_file.read_bytes()
    run_file.write_bytes(data[:-4])

    parser = build_parser()
    args = parser.parse_args(["run-health", "--run-file", str(run_file)])

    assert args.func(args) == 0
    out = capsys.readouterr().out
    assert "partial=true" in out
    assert "rows=1" in out


def test_cli_run_health_surfaces_runtime_summary_from_checkpoint(tmp_path, capsys):
    run_file = tmp_path / "run-health-runtime.jsonl.gz"
    append_jsonl({"case": {"case_id": "case-ok", "seed": 1}, "status": "ok", "findings": []}, run_file)
    meta_path = run_meta_path(run_file)
    checkpoint_path = tmp_path / "run-health-runtime.checkpoint.json"
    state_path = closed_loop_state_path(run_file)
    dump_json({"seen_signatures": ["sig-1"]}, state_path, compact=True)
    dump_json(
        {
            "run_file": str(run_file),
            "checkpoint_file": str(checkpoint_path),
            "status": "running",
            "executed_cases": 4,
            "elapsed_s": 2.0,
            "throughput_cases_s": 2.0,
            "next_seed": 5,
            "stage_profile": {
                "case_count": 4,
                "avg_ms_per_case": {
                    "generate_mutate_ms": 1.5,
                    "scheduler_feedback_ms": 0.5,
                },
                "share_of_total": {
                    "generate_mutate_ms": 0.6,
                    "scheduler_feedback_ms": 0.2,
                },
            },
            "closed_loop_state_file": str(state_path),
            "closed_loop_state_summary": {"seen_signature_count": 1},
        },
        checkpoint_path,
        compact=True,
    )
    dump_json({"status": "running"}, meta_path, compact=True)

    parser = build_parser()
    args = parser.parse_args(["run-health", "--run-file", str(run_file), "--json"])

    assert args.func(args) == 0
    summary = json.loads(capsys.readouterr().out)
    runtime = summary["runtime"]
    assert runtime["status"] == "running"
    assert runtime["executed_cases"] == 4
    assert runtime["throughput_cases_s"] == 2.0
    assert runtime["checkpoint_file"].endswith("run-health-runtime.checkpoint.json")
    assert runtime["meta_file"].endswith("run-health-runtime.meta.json")
    assert runtime["closed_loop_state_file"].endswith("run-health-runtime.state.json.gz")
    assert runtime["closed_loop_state_bytes"] > 0
    assert runtime["evidence_bytes_per_case"] > 0.0
    assert runtime["stage_profile"]["avg_ms_per_case"]["generate_mutate_ms"] == 1.5
    assert runtime["stage_profile"]["share_of_total"]["scheduler_feedback_ms"] == 0.2


def test_cli_parses_artifact_triage_command():
    parser = build_parser()
    args = parser.parse_args(
        ["triage-artifact", "--bug", "bugs/bug_x", "--reduce", "--standalone-reproducer"]
    )
    assert args.cmd == "triage-artifact"
    assert args.bug == "bugs/bug_x"
    assert args.reduce is True
    assert args.standalone_reproducer is True


def test_cli_triage_artifact_writes_datafusion_standalone(tmp_path, monkeypatch, capsys):
    bug_dir = tmp_path / "bug_datafusion"
    bug_dir.mkdir()
    case = Case(
        "case-datafusion",
        1,
        [TableData("t0", [ColumnSpec("g", "str"), ColumnSpec("x", "int", nullable=True)], [{"g": "a", "x": None}])],
        Program(
            "prog-datafusion",
            1,
            [
                {"op": "groupby", "keys": ["g"], "aggs": [{"column": "x", "func": "min", "as": "min_x"}]},
                {"op": "sort", "columns": ["min_x"], "ascending": True},
                {"op": "limit", "n": 20},
            ],
        ),
    )
    (bug_dir / "case.json").write_text(json.dumps(case.to_dict()), encoding="utf-8")
    (bug_dir / "findings.json").write_text(
        json.dumps(
            [
                {
                    "kind": "semantic_output_mismatch",
                    "root_cause": "grouped_topk_null_sort_key",
                    "confidence": "high",
                    "suspicious_backends": ["datafusion"],
                }
            ]
        ),
        encoding="utf-8",
    )
    (bug_dir / "config.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        cli,
        "run_loaded_case",
        lambda *args, **kwargs: {
            "status": "bug",
            "findings": [
                {
                    "kind": "semantic_output_mismatch",
                    "root_cause": "grouped_topk_null_sort_key",
                    "confidence": "high",
                    "suspicious_backends": ["datafusion"],
                }
            ],
        },
    )

    parser = build_parser()
    args = parser.parse_args(
        [
            "triage-artifact",
            "--bug",
            str(bug_dir),
            "--backends",
            "pandas,duckdb,datafusion",
            "--standalone-reproducer",
        ]
    )

    assert args.func(args) == 0
    out = capsys.readouterr().out
    standalone = bug_dir / "standalone_datafusion_groupby_null_sortkey_limit.py"
    triage = json.loads((bug_dir / "triage.json").read_text(encoding="utf-8"))
    assert f"standalone_reproducer={standalone}" in out
    assert standalone.exists()
    assert "ORDER BY min_x ASC NULLS LAST LIMIT 20" in standalone.read_text(encoding="utf-8")
    assert triage["verdict"] == "candidate_implementation_bug"
