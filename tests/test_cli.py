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
from datadiff.version_ledger import VersionObservation, build_version_ledger


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


def test_cli_config_parses_exploration_objective_rules():
    parser = build_parser()
    args = parser.parse_args(
        [
            "fuzz",
            "--exploration-objective-rules",
            json.dumps(
                {
                    "objective": "adaptive consistency",
                    "exact_features": ["op:join"],
                }
            ),
        ]
    )

    config = cli._config_from_args(args)

    assert [rule.objective for rule in config.exploration_objective_rules] == [
        "adaptive_consistency"
    ]
    assert config.exploration_objective_rules[0].exact_features == frozenset({"op:join"})


def test_cli_semantic_registry_command_emits_json(capsys):
    parser = build_parser()
    args = parser.parse_args(
        [
            "semantic-registry",
            "--target-suite",
            "core",
            "--exploration-objective-rules",
            json.dumps({"objective": "adaptive consistency", "exact_features": ["op:join"]}),
            "--json",
        ]
    )

    assert args.func(args) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["schema_version"] == "semantic-registry-v1"
    assert payload["objectives"][0]["feature"] == "exploration_objective:adaptive_consistency"


def test_cli_version_ledger_command_emits_json(tmp_path, capsys):
    run_a = tmp_path / "run-a.jsonl"
    run_b = tmp_path / "run-b.jsonl"
    append_jsonl(
        {
            "status": "bug",
            "findings": [
                {
                    "triage_verdict": "candidate_implementation_bug",
                    "root_cause": "new_root",
                    "suspicious_backends": ["engine"],
                    "signature": "sig",
                }
            ],
        },
        run_b,
    )
    run_a.write_text("", encoding="utf-8")
    parser = build_parser()
    args = parser.parse_args(
        [
            "version-ledger",
            "--run-files",
            f"{run_a},{run_b}",
            "--versions",
            "v1,v2",
            "--json",
        ]
    )

    assert args.func(args) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["schema_version"] == "version-ledger-v1"
    assert payload["summary"]["new_family_count"] == 1


def test_cli_version_ledger_extracts_health_feedback_from_run_logs(tmp_path, capsys):
    run_a = tmp_path / "run-a.jsonl"
    run_b = tmp_path / "run-b.jsonl.gz"
    append_jsonl(
        {
            "duration_ms": 4.0,
            "preflight": {"valid": True, "fallback_used": False},
            "findings": [
                {
                    "triage_verdict": "candidate_implementation_bug",
                    "root_cause": "persistent_root",
                    "suspicious_backends": ["engine"],
                }
            ],
        },
        run_a,
    )
    append_jsonl(
        {
            "duration_ms": 6.0,
            "preflight": {"valid": False, "fallback_used": True},
            "findings": [
                {
                    "triage_verdict": "candidate_implementation_bug",
                    "root_cause": "persistent_root",
                    "suspicious_backends": ["engine"],
                },
                {
                    "triage_verdict": "candidate_implementation_bug",
                    "root_cause": "new_root",
                    "suspicious_backends": ["engine"],
                },
                {
                    "triage_verdict": "generator_false_positive",
                    "root_cause": "generator_noise",
                    "suspicious_backends": ["engine"],
                },
            ],
        },
        run_b,
    )
    append_jsonl(
        {
            "duration_ms": 2.0,
            "preflight": {"valid": False, "fallback_used": True},
            "findings": [],
        },
        run_b,
    )
    append_jsonl(
        {
            "duration_ms": 1.0,
            "preflight": {"valid": True, "fallback_used": True},
            "findings": [],
        },
        run_b,
    )
    dump_json({"throughput_cases_s": 8.0}, run_meta_path(run_a))
    dump_json(
        {
            "preflight": {"invalid_cases": 2, "fallback_cases": 3},
            "throughput_cases_s": 4.0,
        },
        run_meta_path(run_b),
    )
    parser = build_parser()
    args = parser.parse_args(
        [
            "version-ledger",
            "--run-files",
            f"{run_a},{run_b}",
            "--versions",
            "v1,v2",
            "--json",
        ]
    )

    assert args.func(args) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["summary"]["new_family_count"] == 1
    assert payload["summary"]["persistent_family_count"] == 1
    assert payload["summary"]["health_observation_count"] == 2
    assert payload["summary"]["invalid_case_count"] == 2
    assert payload["summary"]["fallback_case_count"] == 3
    assert payload["summary"]["false_positive_count"] == 1
    assert payload["summary"]["min_throughput_cases_s"] == pytest.approx(4.0)
    assert payload["summary"]["max_invalid_rate"] == pytest.approx(2 / 3)
    assert payload["summary"]["max_false_positive_rate"] == pytest.approx(1 / 3)
    assert payload["health"]["schema_version"] == "version-ledger-health-v1"
    assert payload["health_feedback_report"]["schema_version"] == (
        "version-ledger-health-feedback-report-v1"
    )
    assert payload["health_feedback_report"]["has_health_feedback"] is True
    assert payload["versions"][1]["run_file"] == str(run_b)
    assert payload["versions"][1]["health"]["invalid_case_count"] == 2
    assert payload["versions"][1]["health"]["fallback_case_count"] == 3
    assert payload["versions"][1]["health"]["false_positive_count"] == 1
    assert payload["families"][0]["health_observations"]
    assert payload["continual_learning"]["schema_version"] == "cross-version-continual-learning-v1"
    assert payload["adaptive_learning_seed"]["continual_priority_memory"]["imported_family_count"] >= 1


def test_cli_version_ledger_extracts_health_feedback_from_real_fuzz_run(
    tmp_path, monkeypatch, capsys
):
    from datadiff import runner as runner_module
    from datadiff import util as util_module

    runs_dir = tmp_path / "runs"
    reports_dir = tmp_path / "reports"
    bugs_dir = tmp_path / "bugs"
    corpus_dir = tmp_path / "corpus"
    monkeypatch.setattr(util_module, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(util_module, "REPORTS_DIR", reports_dir)
    monkeypatch.setattr(util_module, "BUGS_DIR", bugs_dir)
    monkeypatch.setattr(util_module, "CORPUS_DIR", corpus_dir)
    monkeypatch.setattr(runner_module, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(runner_module, "CORPUS_DIR", corpus_dir)
    monkeypatch.setattr(cli, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(cli, "REPORTS_DIR", reports_dir)
    monkeypatch.setattr(cli, "BUGS_DIR", bugs_dir)
    monkeypatch.setattr(cli, "CORPUS_DIR", corpus_dir)

    parser = build_parser()
    fuzz_args = parser.parse_args(
        [
            "fuzz",
            "--cases",
            "1",
            "--seed",
            "1337",
            "--backends",
            "pandas,sqlite",
            "--target-version",
            "engine==real-smoke",
            "--no-compress-run-log",
            "--disable-artifact",
            "--skip-paper-journal",
            "--log-level",
            "minimal",
        ]
    )

    assert fuzz_args.func(fuzz_args) == 0
    capsys.readouterr()
    run_file = next(runs_dir.glob("run-*.jsonl"))
    meta = json.loads(run_meta_path(run_file).read_text(encoding="utf-8"))

    ledger_args = parser.parse_args(
        [
            "version-ledger",
            "--run-file",
            str(run_file),
            "--versions",
            "engine==real-smoke",
            "--json",
        ]
    )

    assert ledger_args.func(ledger_args) == 0
    payload = json.loads(capsys.readouterr().out)

    assert meta["executed_cases"] == 1
    assert meta["throughput_cases_s"] > 0.0
    assert payload["version_order"] == ["engine==real-smoke"]
    assert payload["summary"]["version_count"] == 1
    assert payload["summary"]["health_observation_count"] == 1
    assert payload["summary"]["min_throughput_cases_s"] == pytest.approx(
        meta["throughput_cases_s"]
    )
    assert payload["health_feedback_report"]["has_health_feedback"] is True
    assert payload["versions"][0]["health"]["duration_ms_total"] > 0.0
    assert payload["versions"][0]["health"]["throughput_cases_s"] == pytest.approx(
        meta["throughput_cases_s"]
    )


def test_cli_version_ledger_reads_run_logs_from_manifest_index(tmp_path, capsys):
    run_a = tmp_path / "runs" / "run-v1.jsonl"
    run_b = tmp_path / "runs" / "run-v2.jsonl"
    append_jsonl({"duration_ms": 2.0, "findings": []}, run_a)
    append_jsonl(
        {
            "duration_ms": 3.0,
            "findings": [
                {
                    "triage_verdict": "candidate_implementation_bug",
                    "root_cause": "indexed_root",
                    "suspicious_backends": ["engine"],
                }
            ],
        },
        run_b,
    )
    dump_json({"target_version": "engine-v1", "throughput_cases_s": 2.0}, run_meta_path(run_a))
    dump_json({"target_version": "engine-v2", "throughput_cases_s": 3.0}, run_meta_path(run_b))
    manifest_a = tmp_path / "runs" / "experiment-v1.json"
    manifest_b = tmp_path / "runs" / "experiment-v2.json"
    dump_json(
        {
            "evidence_mode": "historical",
            "target_version": "engine-v1",
            "runs": [{"run_file": str(run_a), "target_version": "engine-v1"}],
        },
        manifest_a,
    )
    dump_json(
        {
            "evidence_mode": "historical",
            "target_version": "engine-v2",
            "runs": [{"run_file": str(run_b), "target_version": "engine-v2"}],
        },
        manifest_b,
    )
    index = tmp_path / "reports" / "final-index.json"
    dump_json(
        {
            "schema_version": "final-experiment-manifest-index-v1",
            "manifest_files": [str(manifest_a), str(manifest_b)],
            "commands": [],
        },
        index,
    )
    parser = build_parser()
    args = parser.parse_args(["version-ledger", "--manifest-index", str(index), "--json"])

    assert args.func(args) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["version_order"] == ["engine-v1", "engine-v2"]
    assert payload["summary"]["new_family_count"] == 1
    assert payload["summary"]["health_observation_count"] == 2
    assert payload["versions"][1]["run_file"] == str(run_b)
    assert payload["health_feedback_report"]["has_health_feedback"] is True


def test_cli_version_ledger_manifest_index_does_not_fallback_to_latest_run(
    tmp_path, monkeypatch, capsys
):
    runs_dir = tmp_path / "runs"
    monkeypatch.setattr(cli, "RUNS_DIR", runs_dir)
    latest_run = runs_dir / "run-latest.jsonl"
    append_jsonl(
        {
            "findings": [
                {
                    "triage_verdict": "candidate_implementation_bug",
                    "root_cause": "latest_root",
                    "suspicious_backends": ["engine"],
                }
            ]
        },
        latest_run,
    )
    dump_json({"target_version": "latest-version"}, run_meta_path(latest_run))
    index = tmp_path / "reports" / "empty-index.json"
    dump_json(
        {
            "schema_version": "final-experiment-manifest-index-v1",
            "manifest_files": [],
            "commands": [],
        },
        index,
    )
    parser = build_parser()
    args = parser.parse_args(["version-ledger", "--manifest-index", str(index), "--json"])

    assert args.func(args) == 2
    captured = capsys.readouterr()

    assert captured.out == ""
    assert "no run logs found in --manifest-index" in captured.err


def test_cli_version_ledger_evidence_manifest_requires_two_versions(tmp_path, capsys):
    run_a = tmp_path / "run-a.jsonl"
    append_jsonl({"duration_ms": 2.0, "findings": []}, run_a)
    ledger = tmp_path / "reports" / "ledger.json"
    manifest = tmp_path / "reports" / "experiment-ledger.json"
    parser = build_parser()
    args = parser.parse_args(
        [
            "version-ledger",
            "--run-file",
            str(run_a),
            "--versions",
            "v1",
            "--output",
            str(ledger),
            "--evidence-manifest-output",
            str(manifest),
        ]
    )

    assert args.func(args) == 2
    captured = capsys.readouterr()

    assert captured.out == ""
    assert "requires run logs from at least two versions" in captured.err
    assert not ledger.exists()
    assert not manifest.exists()


def test_cli_version_ledger_writes_final_readiness_evidence_manifest(tmp_path, capsys):
    run_a = tmp_path / "run-a.jsonl"
    run_b = tmp_path / "run-b.jsonl"
    run_a.write_text("", encoding="utf-8")
    append_jsonl(
        {
            "findings": [
                {
                    "triage_verdict": "candidate_implementation_bug",
                    "root_cause": "new_root",
                    "suspicious_backends": ["engine"],
                }
            ],
        },
        run_b,
    )
    ledger = tmp_path / "reports" / "ledger.json"
    manifest = tmp_path / "reports" / "experiment-ledger.json"
    parser = build_parser()
    args = parser.parse_args(
        [
            "version-ledger",
            "--run-files",
            f"{run_a},{run_b}",
            "--versions",
            "v1,v2",
            "--output",
            str(ledger),
            "--evidence-manifest-output",
            str(manifest),
        ]
    )

    assert args.func(args) == 0
    output = capsys.readouterr().out
    payload = json.loads(ledger.read_text(encoding="utf-8"))
    evidence = json.loads(manifest.read_text(encoding="utf-8"))

    assert "evidence_manifest=" in output
    assert payload["schema_version"] == "version-ledger-v1"
    assert evidence["schema_version"] == "version-ledger-evidence-manifest-v1"
    assert evidence["evidence_kind"] == "postprocess_ledger"
    assert evidence["runs"][0]["evidence_kind"] == "postprocess_ledger"
    assert evidence["runs"][0]["version_ledger_file"] == str(ledger)
    assert evidence["champion_transfer"]["schema_version"] == "champion-transfer-evidence-v1"
    assert evidence["experiment_meta"]["comparison_group"] == "cross_version_continual_learning"


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


def test_cli_parses_discovery_run_options():
    parser = build_parser()
    args = parser.parse_args(
        [
            "discovery-run",
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
    assert args.cmd == "discovery-run"
    assert args.cases == 10
    assert args.seed == 5
    assert args.target_suite == "latest_no_datafusion"
    assert args.preset == "live_arrow_issue_focus"
    assert args.write_issues is False
    assert args.overwrite_issues is False
    assert args.skip_run_report is True
    assert args.fail_on_fresh_candidate is True


def test_cli_parses_discovery_campaign_options():
    parser = build_parser()
    args = parser.parse_args(
        [
            "discovery-campaign",
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
    assert args.cmd == "discovery-campaign"
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


def test_cli_discovery_run_writes_integrated_manifest(tmp_path, monkeypatch):
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
        assert config.generator_profile == "discovery_fresh"
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
            "discovery-run",
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
    assert manifest["schema_version"] == "discovery-run-v1"
    assert manifest["generated_by"] == "datadiff discovery-run"
    assert manifest["bug_audit"]["candidate_bug_families"] == ["audit_family@pyarrow"]
    assert manifest["classification"]["fresh_candidate_bug_families"] == {"new_family@polars": 1}
    assert manifest["fuzz_run"]["run_file"]
    assert manifest["fuzz_run"]["fresh_candidate_evidence_rows"] == 1
    assert manifest["candidate_pipeline"]["summary"]["candidate_count"] == 1
    evidence = json.loads((tmp_path / "hunt-fresh-candidates.json").read_text(encoding="utf-8"))
    assert evidence["candidate_row_count"] == 1
    assert evidence["candidate_rows"][0]["case"]["case_id"] == "case-fresh"


def test_cli_discovery_campaign_writes_targeted_manifest(tmp_path, monkeypatch):
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
        assert config.generator_profile == "discovery_fresh"
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
            "discovery-campaign",
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
    assert manifest["schema_version"] == "discovery-campaign-v1"
    assert manifest["generated_by"] == "datadiff discovery-campaign"
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
    assert manifest["scheduler"]["strategy"] == "true_bug_acquisition_good_turing_ucb_boltzmann"
    assert manifest["scheduler"]["bug_discovery_system"]["schema_version"] == "bug-discovery-system-v1"
    assert manifest["scheduler"]["lanes"][0]["lane_id"] == "arrow_layout"
    assert manifest["scheduler"]["lanes"][0]["acquisition_strategy"] == (
        "true_bug_acquisition_good_turing_ucb_boltzmann"
    )
    assert "good_turing_unseen_probability" in manifest["scheduler"]["lanes"][0]
    assert "ucb_bonus" in manifest["scheduler"]["lanes"][0]
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


def test_cli_discovery_campaign_writes_running_manifest_before_lane_executes(tmp_path, monkeypatch):
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
            "discovery-campaign",
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


def test_cli_discovery_campaign_watch_health_stops_remaining_lanes(tmp_path, monkeypatch):
    run_file = tmp_path / "run-watch.jsonl"
    manifest_file = tmp_path / "discovery-campaign-watch.json"
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
            "discovery-campaign",
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


def test_cli_discovery_campaign_lists_lanes_as_json(capsys):
    parser = build_parser()
    args = parser.parse_args(["discovery-campaign", "--list-lanes", "--json"])

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


def test_cli_target_version_audit_json_uses_offline_overrides(capsys):
    parser = build_parser()
    args = parser.parse_args(
        [
            "target-version-audit",
            "--packages",
            "pandas,polars",
            "--latest-versions",
            "pandas=0.0.0,polars=0.0.0",
            "--no-network",
            "--json",
        ]
    )

    assert args.func(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == "target-version-audit-v1"
    assert [row["package"] for row in payload["target_packages"]] == ["pandas", "polars"]
    assert payload["summary"]["outdated_target_package_count"] == 2


def test_cli_parses_discovery_campaign_status_command():
    parser = build_parser()
    args = parser.parse_args(
        [
            "discovery-campaign-status",
            "--manifest",
            "new_issue/generated/discovery-campaign-live.json",
            "--limit",
            "2",
            "--json",
            "--fail-on-fresh-candidate",
            "--fail-on-bug",
        ]
    )
    assert args.cmd == "discovery-campaign-status"
    assert args.manifest == "new_issue/generated/discovery-campaign-live.json"
    assert args.limit == 2
    assert args.json is True
    assert args.fail_on_fresh_candidate is True
    assert args.fail_on_bug is True


def test_cli_parses_discovery_campaign_aggregate_command():
    parser = build_parser()
    args = parser.parse_args(
        [
            "discovery-campaign-aggregate",
            "--manifests",
            "new_issue/generated/discovery-campaign*.json",
            "--limit",
            "4",
            "--output",
            "new_issue/generated/discovery-campaign-aggregate.json",
            "--json",
        ]
    )
    assert args.cmd == "discovery-campaign-aggregate"
    assert args.manifests == "new_issue/generated/discovery-campaign*.json"
    assert args.limit == 4
    assert args.output == "new_issue/generated/discovery-campaign-aggregate.json"
    assert args.json is True


def test_cli_parses_candidate_pipeline_command():
    parser = build_parser()
    args = parser.parse_args(
        [
            "candidate-pipeline",
            "--manifest",
            "new_issue/generated/discovery-campaign-live.json",
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
    assert args.manifest == "new_issue/generated/discovery-campaign-live.json"
    assert args.evidence_files == "new_issue/generated/a.json,new_issue/generated/b.json"
    assert args.output_dir == "new_issue/generated/candidate-pipelines"
    assert args.recheck_attempts == 3
    assert args.no_reduce is True
    assert args.no_standalone_reproducer is True
    assert args.json is True
    assert args.fail_on_ready is True


def test_cli_discovery_campaign_aggregate_summarizes_manifest_glob(tmp_path, capsys):
    run_file = tmp_path / "run-a.jsonl"
    append_jsonl({"case": {"case_id": "case-a", "seed": 1}, "findings": []}, run_file)
    manifest_a = tmp_path / "discovery-campaign-a.json"
    manifest_b = tmp_path / "discovery-campaign-b.json"
    evidence_file = tmp_path / "discovery-campaign-a-embedded_sql-seed1-fresh-candidates.json"
    output_file = tmp_path / "aggregate.json"
    evidence_file.write_text(
        json.dumps(
            {
                "schema_version": "discovery-run-fresh-candidates-v1",
                "source_run_file": str(run_file),
                "fresh_candidate_bug_families": {"fresh_family@duckdb": 2},
                "candidate_row_count": 1,
                "candidate_rows": [
                    {
                        "case": {"case_id": "case-fast", "seed": 1},
                        "case_index": 4,
                        "elapsed_s": 0.25,
                        "candidate_bug_families": {"fresh_family@duckdb": 1},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    manifest_a.write_text(
        json.dumps(
            {
                "schema_version": "discovery-campaign-v1",
                "status": "running",
                "started_at": "2026-06-06T00:00:00Z",
                "generated_at": "2026-06-06T00:00:10Z",
                "seeds": [1],
                "progress": {
                    "planned_run_count": 2,
                    "completed_run_count": 1,
                    "remaining_run_count": 1,
                },
                "summary": {
                    "fresh_candidate_bug_families": {"fresh_family@duckdb": 2},
                    "triage_verdicts": {"candidate_implementation_bug": 2},
                    "candidate_pipeline": {
                        "candidate_count": 3,
                        "processed_candidate_count": 2,
                        "skipped_duplicate_candidate_count": 1,
                        "rechecked_count": 2,
                        "reproduced_count": 1,
                        "reduced_count": 1,
                        "issue_draft_count": 1,
                    },
                },
                "runs": [
                    {
                        "lane_id": "embedded_sql",
                        "theme": "Embedded SQL",
                        "target_suite": "embedded_sql",
                        "preset": "live_embedded_sql",
                        "seed": 1,
                        "cases": 10,
                        "status": "completed",
                        "run_file": str(run_file),
                        "fresh_candidate_evidence": str(evidence_file),
                        "fresh_candidate_evidence_rows": 2,
                        "classification": {
                            "fresh_candidate_bug_families": {"fresh_family@duckdb": 2},
                        },
                        "candidate_pipeline": {
                            "summary": {
                                "candidate_count": 3,
                                "processed_candidate_count": 2,
                                "skipped_duplicate_candidate_count": 1,
                                "rechecked_count": 2,
                                "reproduced_count": 1,
                                "reduced_count": 1,
                                "issue_draft_count": 1,
                                "recheck_pass_rate": 0.5,
                            }
                        },
                        "health": {
                            "runtime": {
                                "executed_cases": 10,
                                "elapsed_s": 2.0,
                            }
                        },
                    },
                    {
                        "lane_id": "embedded_sql",
                        "status": "running",
                        "seed": 2,
                        "cases": 10,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    manifest_b.write_text(
        json.dumps(
            {
                "schema_version": "discovery-campaign-v1",
                "status": "completed",
                "started_at": "2026-06-06T00:00:20Z",
                "completed_at": "2026-06-06T00:00:25Z",
                "seeds": [3],
                "progress": {
                    "planned_run_count": 1,
                    "completed_run_count": 1,
                    "remaining_run_count": 0,
                },
                "summary": {
                    "fresh_candidate_bug_families": {"other_family@polars": 1},
                    "triage_verdicts": {"candidate_implementation_bug": 1},
                },
                "runs": [],
            }
        ),
        encoding="utf-8",
    )

    args = build_parser().parse_args(
        [
            "discovery-campaign-aggregate",
            "--manifests",
            str(tmp_path / "discovery-campaign-*.json"),
            "--output",
            str(output_file),
            "--json",
        ]
    )

    assert args.func(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == "discovery-campaign-aggregate-v1"
    assert payload["manifest_count"] == 2
    assert payload["progress"]["completed_run_count"] == 2
    assert payload["progress"]["running_run_count"] == 1
    assert payload["fresh_candidate_bug_family_count"] == 2
    assert payload["fresh_candidate_evidence_rows"] == 2
    assert payload["candidate_pipeline"]["reproduced_count"] == 1
    assert payload["candidate_pipeline"]["rechecked_count"] == 2
    assert payload["candidate_pipeline"]["processed_candidate_count"] == 2
    assert payload["candidate_pipeline"]["skipped_duplicate_candidate_count"] == 1
    assert payload["candidate_pipeline"]["recheck_pass_rate"] == 0.5
    assert payload["first_candidate"]["case_id"] == "case-fast"
    assert payload["first_candidate"]["elapsed_s"] == 0.25
    assert payload["avg_candidate_bug_discovery_auc"] == 0.6
    assert payload["efficiency"]["serial_cases_per_s"] == 5.0
    assert payload["efficiency"]["parallel_cases_per_s"] == 0.4
    assert payload["efficiency"]["parallel_capacity_cases_s"] == 5.0
    assert payload["icse_experiment_quality"]["dimensions"]["speed"]["passed"] is True
    assert payload["lanes"][0]["lane_id"] == "embedded_sql"
    assert payload["icse_experiment_quality"]["schema_version"] == "icse-experiment-quality-v1"
    assert json.loads(output_file.read_text(encoding="utf-8"))["manifest_count"] == 2


def test_cli_discovery_campaign_status_summarizes_running_manifest(tmp_path, monkeypatch, capsys):
    manifest_file = tmp_path / "discovery-campaign-running.json"
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    historical_manifest = tmp_path / "discovery-campaign-history-manifest.json"
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
                "schema_version": "discovery-campaign-v1",
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
                    "schema_version": "discovery-campaign-v1",
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

    args = build_parser().parse_args(["discovery-campaign-status", "--manifest", str(manifest_file), "--json"])
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
        ["discovery-campaign-status", "--manifest", str(manifest_file), "--fail-on-fresh-candidate"]
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


def test_cli_parses_discovery_profile():
    parser = build_parser()
    args = parser.parse_args(["fuzz", "--profile", "discovery"])
    assert args.cmd == "fuzz"
    assert args.profile == "discovery"


def test_cli_parses_adaptive_profile_pool_flags():
    parser = build_parser()
    args = parser.parse_args(
        [
            "fuzz",
            "--profile",
            "common",
            "--profile-pool",
            "common,discovery_fresh",
            "--profile-learning-weight",
            "0.7",
            "--version-pair-pool",
            "latest->fixed,latest->preview",
            "--semantic-objective-learning-weight",
            "0.5",
            "--metamorphic-relation-learning-weight",
            "0.6",
            "--version-pair-learning-weight",
            "0.8",
            "--backend-pair-learning-weight",
            "0.9",
            "--backend-pair-priority-limit",
            "4",
            "--metamorphic-relation-order",
            "row_permutation,filter_idempotence",
            "--target-version",
            "latest",
            "--fixed-version",
            "fixed",
        ]
    )
    config = cli._config_from_args(args)

    assert args.profile_pool == "common,discovery_fresh"
    assert args.profile_learning_weight == 0.7
    assert config.generator_profile_pool == ["common", "discovery_fresh"]
    assert config.version_pair_pool == ["latest->fixed", "latest->preview"]
    assert config.generator_profile_learning_weight == 0.7
    assert config.semantic_objective_learning_weight == 0.5
    assert config.metamorphic_relation_learning_weight == 0.6
    assert config.version_pair_learning_weight == 0.8
    assert config.backend_pair_learning_weight == 0.9
    assert config.backend_pair_priority_limit == 4
    assert config.metamorphic_relation_order == ["row_permutation", "filter_idempotence"]
    assert config.target_version == "latest"
    assert config.fixed_version == "fixed"


def test_cli_parses_discovery_fresh_profile():
    parser = build_parser()
    args = parser.parse_args(["fuzz", "--profile", "discovery_fresh"])
    assert args.cmd == "fuzz"
    assert args.profile == "discovery_fresh"

    args = parser.parse_args(["longrun", "--profile", "discovery_fresh"])
    assert args.cmd == "longrun"
    assert args.profile == "discovery_fresh"


def test_cli_parses_discovery_no_groupby_profile():
    parser = build_parser()
    args = parser.parse_args(["fuzz", "--profile", "discovery_no_groupby"])
    assert args.cmd == "fuzz"
    assert args.profile == "discovery_no_groupby"


def test_cli_parses_pyarrow_groupby_filter_cast_membership_profile():
    parser = build_parser()
    args = parser.parse_args(["fuzz", "--profile", "pyarrow_groupby_filter_cast_membership"])
    assert args.cmd == "fuzz"
    assert args.profile == "pyarrow_groupby_filter_cast_membership"


def test_cli_parses_pandas_arrow_bool_groupby_reduction_profile():
    parser = build_parser()
    args = parser.parse_args(["fuzz", "--profile", "pandas_arrow_bool_groupby_reduction_semantics"])
    assert args.cmd == "fuzz"
    assert args.profile == "pandas_arrow_bool_groupby_reduction_semantics"


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


def test_cli_parses_order_sensitive_discovery_run_profiles():
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


def test_cli_parses_target_version_audit_command():
    parser = build_parser()
    args = parser.parse_args(["target-version-audit", "--no-network"])
    assert args.cmd == "target-version-audit"
    assert args.no_network is True


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


def test_cli_parses_experiment_per_case_adaptive_flags():
    parser = build_parser()
    args = parser.parse_args(
        [
            "experiment",
            "--presets",
            "baseline",
            "--profile-pool",
            "common,discovery_fresh",
            "--profile-learning-weight",
            "0.7",
            "--semantic-objective-learning-weight",
            "0.5",
            "--enable-metamorphic-oracle",
            "--metamorphic-relation-learning-weight",
            "0.6",
            "--metamorphic-relation-order",
            "row_permutation,filter_idempotence",
            "--version-pair-pool",
            "latest->fixed,latest->preview",
            "--version-pair-learning-weight",
            "0.8",
            "--backend-pair-learning-weight",
            "0.9",
            "--backend-pair-priority-limit",
            "4",
        ]
    )
    assert args.cmd == "experiment"
    assert args.profile_pool == "common,discovery_fresh"
    assert args.profile_learning_weight == 0.7
    assert args.semantic_objective_learning_weight == 0.5
    assert args.enable_metamorphic_oracle is True
    assert args.metamorphic_relation_learning_weight == 0.6
    assert args.metamorphic_relation_order == "row_permutation,filter_idempotence"
    assert args.version_pair_pool == "latest->fixed,latest->preview"
    assert args.version_pair_learning_weight == 0.8
    assert args.backend_pair_learning_weight == 0.9
    assert args.backend_pair_priority_limit == 4


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
    assert _preset_config("guided_join").generator_profile == "discovery_no_groupby"
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
        _preset_config("pandas_arrow_bool_groupby_reduction_semantics").generator_profile
        == "pandas_arrow_bool_groupby_reduction_semantics"
    )
    assert (
        _preset_config("pandas_arrow_bool_groupby_reduction_semantics").guidance_targets[0]
        == "pandas_arrow_bool_groupby_reduction_semantics"
    )
    assert (
        _preset_config("pandas_arrow_bool_groupby_reduction_semantics_metamorphic").enable_metamorphic_oracle is True
    )
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


def test_cli_parses_discovery_guided_metamorphic_preset():
    config = _preset_config("discovery_guided_metamorphic")
    assert config.generator_profile == "discovery"
    assert config.enable_metamorphic_oracle is True
    assert config.guidance_strategy == "guided"
    assert config.metamorphic_variant_limit == 8


def test_cli_parses_discovery_no_groupby_guided_metamorphic_preset():
    config = _preset_config("discovery_no_groupby_guided_metamorphic")
    assert config.generator_profile == "discovery_no_groupby"
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

    guided = _preset_config("discovery_guided")
    guided_metamorphic = _preset_config("discovery_guided_metamorphic")
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

    discovery_guided = build_experiment_config(
        "discovery",
        ("enable_guidance", "target_discovery_guided"),
    )
    assert discovery_guided.to_dict() == _preset_config("discovery_guided").to_dict()

    deep_organic = build_experiment_config(
        "live_deep_organic",
        ("enable_metamorphic_oracle", "candidate_pool_10", "metamorphic_variant_limit_8"),
    )
    assert deep_organic.to_dict() == _preset_config("live_deep_organic_metamorphic").to_dict()


def test_catalog_overlay_aliases_are_structured_in_preset_catalog():
    guided_join = PRESET_CATALOG["guided_join"]
    no_feedback = PRESET_CATALOG["no_feedback"]
    discovery_guided = PRESET_CATALOG["discovery_guided"]

    assert guided_join.base_preset == "baseline"
    assert guided_join.overlays == ("enable_guidance", "target_join")
    assert no_feedback.base_preset == "baseline"
    assert no_feedback.overlays == ("disable_feedback_corpus",)
    assert discovery_guided.base_preset == "discovery"
    assert discovery_guided.overlays == ("enable_guidance", "target_discovery_guided")


def test_catalog_backed_replay_overlay_preserves_live_base_targets():
    live = _preset_config("live_datafusion")
    replay = _preset_config("live_datafusion_replay")
    assert replay.enable_replay_bug is True
    assert replay.generator_profile == live.generator_profile
    assert replay.guidance_targets == live.guidance_targets
    assert replay.local_source_exploration_weight == live.local_source_exploration_weight


def test_cli_parses_discovery_experiment_presets():
    assert _preset_config("discovery").generator_profile == "discovery"
    assert _preset_config("discovery_no_groupby").generator_profile == "discovery_no_groupby"
    guided = _preset_config("discovery_guided")
    assert guided.generator_profile == "discovery"
    assert guided.guidance_strategy == "guided"
    assert guided.guidance_targets == ["join", "groupby", "mutate", "filter", "expressions"]
    no_groupby_guided = _preset_config("discovery_no_groupby_guided")
    assert no_groupby_guided.generator_profile == "discovery_no_groupby"
    assert no_groupby_guided.guidance_targets == ["join", "mutate", "filter", "expressions", "sort_limit"]
    metamorphic = _preset_config("discovery_metamorphic")
    assert metamorphic.generator_profile == "discovery"
    assert metamorphic.enable_metamorphic_oracle is True
    no_groupby_metamorphic = _preset_config("discovery_no_groupby_metamorphic")
    assert no_groupby_metamorphic.generator_profile == "discovery_no_groupby"
    assert no_groupby_metamorphic.enable_metamorphic_oracle is True


def test_cli_parses_live_datafusion_presets():
    live = _preset_config("live_datafusion")
    assert live.generator_profile == "discovery"
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
    assert fresh.generator_profile == "discovery_no_groupby"
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
    assert fresh_metamorphic.generator_profile == "discovery_no_groupby"
    assert fresh_metamorphic.enable_metamorphic_oracle is True
    assert fresh_metamorphic.oracle_mode == "both"
    assert fresh_metamorphic.metamorphic_variant_limit == 6

    replay = _preset_config("live_datafusion_replay")
    assert replay.generator_profile == "discovery"
    assert replay.enable_replay_bug is True
    assert replay.guidance_targets == live.guidance_targets


def test_cli_parses_non_datafusion_live_presets():
    arrow = _preset_config("live_arrow")
    assert arrow.generator_profile == "discovery"
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
    assert polars_lazy.generator_profile == "discovery"
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
    assert polars_streaming.generator_profile == "discovery"
    assert {
        "groupby",
        "aggregation",
        "numeric_mean_aggregate",
        "sort_limit",
    }.issubset(polars_streaming.guidance_targets)
    assert polars_streaming.local_source_exploration_weight == 0.45

    embedded_sql = _preset_config("live_embedded_sql")
    assert embedded_sql.generator_profile == "discovery"
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
    assert cross_family.generator_profile == "discovery"
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
    assert deep_organic.generator_profile == "discovery_fresh"
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
    assert arrow_deep.generator_profile == "discovery_fresh"
    assert arrow_deep.enable_metamorphic_oracle is True
    assert "pyarrow_run_end_null_compute_semantics" in arrow_deep.guidance_targets
    assert "csv_long_numeric_roundtrip" not in arrow_deep.guidance_targets

    polars_deep = _preset_config("live_polars_deep_organic_metamorphic")
    assert polars_deep.generator_profile == "discovery_fresh"
    assert polars_deep.enable_metamorphic_oracle is True
    assert "polars_rolling_mean_by_null_count_semantics" in polars_deep.guidance_targets
    assert "polars_reverse_division_columns" not in polars_deep.guidance_targets

    streaming_deep = _preset_config("live_polars_streaming_deep_organic_metamorphic")
    assert streaming_deep.generator_profile == "discovery_fresh"
    assert streaming_deep.enable_metamorphic_oracle is True
    assert "partitioned_running_sum" in streaming_deep.guidance_targets

    datafusion_deep = _preset_config("live_datafusion_deep_organic_metamorphic")
    assert datafusion_deep.generator_profile == "discovery_fresh"
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
    assert embedded_deep.generator_profile == "discovery_fresh"
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


def test_discovery_campaign_config_can_merge_lane_discovery_biases():
    parser = build_parser()
    args = parser.parse_args(["discovery-campaign", "--lanes", "datafusion_optimizer"])
    config = cli._discovery_campaign_config_from_args(
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


def test_discovery_campaign_config_deduplicates_lane_discovery_biases():
    parser = build_parser()
    args = parser.parse_args(["discovery-campaign", "--lanes", "datafusion_optimizer"])
    lane_bias = {
        "targets": ["row_value_absence_filter", "normalized_string_join", "negative_set_membership_filter"],
        "feature_prefixes": ["join:", "filter:", "pattern:"],
        "score_bonus": 0.45,
        "novelty_bonus": 0.20,
        "contribution_bonus": 0.20,
        "candidate_pool_bonus": 0.10,
        "keep_in_pool": True,
    }

    config = cli._discovery_campaign_config_from_args(
        args,
        "live_datafusion_deep_organic_metamorphic",
        lane_discovery_biases=[lane_bias],
    )

    assert len(config.discovery_biases) == 1
    assert config.discovery_biases[0].targets == lane_bias["targets"]


def test_discovery_campaign_config_merges_lane_semantic_focus():
    parser = build_parser()
    args = parser.parse_args(["discovery-campaign", "--lanes", "datafusion_optimizer"])

    config = cli._discovery_campaign_config_from_args(
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


def test_discovery_campaign_config_deduplicates_lane_semantic_focus():
    parser = build_parser()
    args = parser.parse_args(["discovery-campaign", "--lanes", "datafusion_optimizer"])

    config = cli._discovery_campaign_config_from_args(
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
    assert deep_organic.generator_profile == "discovery_fresh"
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
    assert args.compare_presets == "guided_filter,guided_join"
    assert args.refresh is True


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
            "--extra-manifest",
            "reports/experiment-final-version-ledger.json",
            "--manifest-index",
            "reports/final-experiment-manifest-index.json",
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
            "--no-require-adaptive-component-ablation",
            "--min-adaptive-component-ablations",
            "3",
            "--required-adaptive-component-ablations",
            "scheduler-learning,quality-archive,online-reward-model",
            "--no-require-transferability-scope",
            "--min-transfer-target-families",
            "3",
            "--no-require-cross-version-ledger",
            "--min-cross-version-ledger-versions",
            "4",
            "--min-cross-version-ledger-families",
            "2",
            "--no-require-cross-version-health-feedback",
            "--no-require-runtime-efficiency",
            "--min-throughput-cases-s",
            "0.25",
            "--max-scheduler-feedback-share",
            "0.2",
            "--min-scheduler-feedback-cases",
            "25",
            "--no-require-discovery-responsiveness",
            "--max-first-candidate-elapsed-s",
            "120",
            "--no-require-closed-loop-state-persistence",
            "--no-require-adaptive-live-component-evidence",
            "--json",
            "--fail-on-missing",
        ]
    )
    assert args.cmd == "final-readiness"
    assert args.manifest == ["runs/experiment-a.json", "runs/experiment-b.json"]
    assert args.extra_manifest == ["reports/experiment-final-version-ledger.json"]
    assert args.manifest_index == ["reports/final-experiment-manifest-index.json"]
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
    assert args.no_require_adaptive_component_ablation is True
    assert args.min_adaptive_component_ablations == 3
    assert args.required_adaptive_component_ablations == (
        "scheduler-learning,quality-archive,online-reward-model"
    )
    assert args.no_require_transferability_scope is True
    assert args.min_transfer_target_families == 3
    assert args.no_require_cross_version_ledger is True
    assert args.min_cross_version_ledger_versions == 4
    assert args.min_cross_version_ledger_families == 2
    assert args.no_require_cross_version_health_feedback is True
    assert args.no_require_runtime_efficiency is True
    assert args.min_throughput_cases_s == 0.25
    assert args.max_scheduler_feedback_share == 0.2
    assert args.min_scheduler_feedback_cases == 25
    assert args.no_require_discovery_responsiveness is True
    assert args.max_first_candidate_elapsed_s == 120.0
    assert args.no_require_closed_loop_state_persistence is True
    assert args.no_require_adaptive_live_component_evidence is True
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


def test_cli_final_readiness_passes_extra_manifest_to_audit(tmp_path, monkeypatch):
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

    args = build_parser().parse_args(
        [
            "final-readiness",
            "--latest-manifests",
            "3",
            "--extra-manifest",
            "reports/experiment-final-version-ledger.json",
        ]
    )

    assert args.func(args) == 0
    assert calls[0][0][0] is None
    assert calls[0][1]["manifest_limit"] == 3
    assert calls[0][1]["extra_manifest_files"] == [Path("reports/experiment-final-version-ledger.json")]


def test_cli_final_readiness_reads_manifest_index_for_targeted_audit(tmp_path, monkeypatch):
    audit = {"schema_version": "final-readiness-v1", "ready": True, "summary": {}}
    md_path = tmp_path / "final-readiness.md"
    json_path = tmp_path / "final-readiness.json"
    md_path.write_text("# Final\n", encoding="utf-8")
    json_path.write_text(json.dumps(audit), encoding="utf-8")
    index_path = tmp_path / "final-index.json"
    index_path.write_text(
        json.dumps(
            {
                "schema_version": "final-experiment-manifest-index-v1",
                "manifest_files": ["runs/experiment-validation.json"],
                "extra_manifest_files": ["reports/experiment-ledger.json"],
                "commands": [
                    {
                        "name": "live:datafusion_cross",
                        "manifest_files": ["runs/experiment-live.json"],
                        "extra_manifest_files": ["reports/experiment-ledger.json"],
                        "paper_run_journal_files": ["reports/paper-run-journal.jsonl"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    calls = []

    def fake_analyze_final_readiness(*args, **kwargs):
        calls.append((args, kwargs))
        return md_path, json_path

    monkeypatch.setattr(cli, "analyze_final_readiness", fake_analyze_final_readiness)

    args = build_parser().parse_args(["final-readiness", "--manifest-index", str(index_path)])

    assert args.func(args) == 0
    assert calls[0][0][0] == [
        Path("runs/experiment-validation.json"),
        Path("runs/experiment-live.json"),
    ]
    assert calls[0][1]["extra_manifest_files"] == [Path("reports/experiment-ledger.json")]
    assert calls[0][1]["paper_run_journal_files"] == [Path("reports/paper-run-journal.jsonl")]
    assert calls[0][1]["manifest_limit"] is None
    assert calls[0][1]["scan_run_logs"] is True


def test_cli_final_readiness_passes_extended_thresholds_to_audit(tmp_path, monkeypatch):
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

    args = build_parser().parse_args(
        [
            "final-readiness",
            "--no-require-adaptive-component-ablation",
            "--min-adaptive-component-ablations",
            "3",
            "--required-adaptive-component-ablations",
            "scheduler-learning,quality-archive,online-reward-model",
            "--no-require-transferability-scope",
            "--min-transfer-target-families",
            "3",
            "--no-require-cross-version-ledger",
            "--min-cross-version-ledger-versions",
            "4",
            "--min-cross-version-ledger-families",
            "2",
            "--no-require-cross-version-health-feedback",
            "--no-require-runtime-efficiency",
            "--min-throughput-cases-s",
            "0.25",
            "--max-scheduler-feedback-share",
            "0.2",
            "--min-scheduler-feedback-cases",
            "25",
            "--no-require-discovery-responsiveness",
            "--max-first-candidate-elapsed-s",
            "120",
            "--no-require-closed-loop-state-persistence",
            "--no-require-adaptive-live-component-evidence",
        ]
    )

    assert args.func(args) == 0
    thresholds = calls[0][1]["thresholds"]
    assert thresholds.require_adaptive_component_ablation is False
    assert thresholds.min_adaptive_component_ablations == 3
    assert thresholds.required_adaptive_component_ablations == (
        "scheduler_learning",
        "quality_archive",
        "online_reward_model",
    )
    assert thresholds.require_transferability_scope is False
    assert thresholds.min_transfer_target_families == 3
    assert thresholds.require_cross_version_ledger is False
    assert thresholds.min_cross_version_ledger_versions == 4
    assert thresholds.min_cross_version_ledger_families == 2
    assert thresholds.require_cross_version_health_feedback is False
    assert thresholds.require_runtime_efficiency is False
    assert thresholds.min_throughput_cases_s == 0.25
    assert thresholds.max_scheduler_feedback_share == 0.2
    assert thresholds.min_scheduler_feedback_cases == 25
    assert thresholds.require_discovery_responsiveness is False
    assert thresholds.max_first_candidate_elapsed_s == 120.0
    assert thresholds.require_closed_loop_state_persistence is False
    assert thresholds.require_adaptive_live_component_evidence is False


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
            "--min-discovery-workflows",
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


def test_cli_parses_adaptive_benchmark_command():
    parser = build_parser()
    args = parser.parse_args(
        [
            "adaptive-benchmark",
            "--mode",
            "replay",
            "--learning-rounds",
            "16",
            "--systems-iterations",
            "128",
            "--profile-iterations",
            "64",
            "--action-pool-size",
            "12",
            "--profile-top-n",
            "8",
            "--profile-output",
            "reports/custom.prof",
            "--replay-run-file",
            "runs/run-a.jsonl.gz",
            "--replay-manifest",
            "runs/experiment-a.json",
            "--json",
            "--write-report",
            "--output-dir",
            "reports",
        ]
    )
    assert args.cmd == "adaptive-benchmark"
    assert args.mode == "replay"
    assert args.learning_rounds == 16
    assert args.systems_iterations == 128
    assert args.profile_iterations == 64
    assert args.action_pool_size == 12
    assert args.profile_top_n == 8
    assert args.profile_output == "reports/custom.prof"
    assert args.replay_run_file == ["runs/run-a.jsonl.gz"]
    assert args.replay_manifest == ["runs/experiment-a.json"]
    assert args.json is True
    assert args.write_report is True
    assert args.output_dir == "reports"


def test_cli_adaptive_benchmark_prints_json_and_writes_reports(tmp_path, monkeypatch, capsys):
    captured: dict[str, object] = {}
    profile_file = tmp_path / "custom.prof"

    def fake_run_adaptive_benchmark(**kwargs):
        captured.update(kwargs)
        if kwargs.get("profile_output"):
            Path(kwargs["profile_output"]).write_text("profile", encoding="utf-8")
        return {
            "schema_version": "adaptive-benchmark-v1",
            "generated_at": "2026-06-04T00:00:00Z",
            "mode": "replay",
            "learning_effectiveness": {
                "summary": {
                    "baseline_variant": "reward_signal_only",
                    "best_variant_by_average_reward": "full_adaptive",
                }
            },
            "systems_benchmark": {
                "summary": {
                    "materialized_rank_overhead_ratio": 1.25,
                },
                "profiler": {
                    "enabled": True,
                    "profile_output": str(kwargs.get("profile_output", "")),
                    "top_functions": [],
                },
            },
            "real_run_replay": {
                "summary": {
                    "baseline_variant": "reward_signal_only",
                    "best_variant_by_average_reward": "full_adaptive",
                    "event_count": 42,
                }
            },
        }

    def fake_write_adaptive_benchmark_markdown(payload, path):
        Path(path).write_text(f"# {payload['schema_version']}\n", encoding="utf-8")

    monkeypatch.setattr(cli, "run_adaptive_benchmark", fake_run_adaptive_benchmark)
    monkeypatch.setattr(cli, "write_adaptive_benchmark_markdown", fake_write_adaptive_benchmark_markdown)
    monkeypatch.setattr(cli, "utc_now", lambda: "2026-06-04T01:02:03Z")

    args = build_parser().parse_args(
        [
            "adaptive-benchmark",
            "--json",
            "--write-report",
            "--output-dir",
            str(tmp_path),
            "--profile-output",
            str(profile_file),
            "--learning-rounds",
            "8",
            "--systems-iterations",
            "32",
            "--profile-iterations",
            "16",
            "--action-pool-size",
            "6",
            "--replay-run-file",
            str(tmp_path / "run-a.jsonl.gz"),
            "--replay-manifest",
            str(tmp_path / "experiment-a.json"),
        ]
    )

    assert args.func(args) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["schema_version"] == "adaptive-benchmark-v1"
    assert payload["output_json"] == str(tmp_path / "adaptive-benchmark-20260604T010203.json")
    assert payload["output_markdown"] == str(tmp_path / "adaptive-benchmark-20260604T010203.md")
    assert payload["profile_output"] == str(profile_file)
    assert (tmp_path / "adaptive-benchmark-20260604T010203.json").exists()
    assert (tmp_path / "adaptive-benchmark-20260604T010203.md").exists()
    assert profile_file.exists()
    assert captured["learning_rounds"] == 8
    assert captured["systems_iterations"] == 32
    assert captured["profile_iterations"] == 16
    assert captured["action_pool_size"] == 6
    assert Path(captured["profile_output"]) == profile_file
    assert captured["replay_run_files"] == [tmp_path / "run-a.jsonl.gz"]
    assert captured["replay_manifests"] == [tmp_path / "experiment-a.json"]


def test_cli_parses_analyze_ablation_audit_command():
    parser = build_parser()
    args = parser.parse_args(
        [
            "analyze-ablation-audit",
            "--manifest",
            "runs/experiment-ablation.json",
            "--reference-presets",
            "baseline,guided",
            "--ablation-presets",
            "no_type_aware,no_normalizer",
            "--refresh",
        ]
    )
    assert args.cmd == "analyze-ablation-audit"
    assert args.manifest == "runs/experiment-ablation.json"
    assert args.reference_presets == "baseline,guided"
    assert args.ablation_presets == "no_type_aware,no_normalizer"
    assert args.refresh is True


def test_cli_ablation_audit_help_prefers_reference_wording(capsys):
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["analyze-ablation-audit", "--help"])
    help_text = capsys.readouterr().out

    assert "--reference-presets" in help_text


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
            "--adaptive-learning-weight",
            "0.6",
            "--scheduler-annealing-temperature",
            "0.4",
            "--scheduler-annealing-decay",
            "0.97",
            "--scheduler-annealing-min-temperature",
            "0.03",
            "--continual-learning-ledgers",
            "reports/ledger-a.json,reports/ledger-b.json",
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
    assert args.adaptive_learning_weight == 0.6
    assert args.scheduler_annealing_temperature == 0.4
    assert args.scheduler_annealing_decay == 0.97
    assert args.scheduler_annealing_min_temperature == 0.03
    assert args.continual_learning_ledgers == "reports/ledger-a.json,reports/ledger-b.json"
    assert args.local_source_exploration_weight == 0.2


def test_cli_experiment_parses_strategy_snapshot_flags():
    parser = build_parser()
    args = parser.parse_args(
        [
            "experiment",
            "--strategy-snapshot",
            "reports/frozen-strategy.json",
            "--strategy-learning",
            "reports/strategy-learning.json",
            "--freeze-strategy-snapshot",
        ]
    )

    assert args.cmd == "experiment"
    assert args.strategy_snapshot == "reports/frozen-strategy.json"
    assert args.strategy_learning == "reports/strategy-learning.json"
    assert args.freeze_strategy_snapshot is True


def test_cli_fuzz_can_disable_parallel_backend_execution():
    parser = build_parser()
    args = parser.parse_args(["fuzz", "--disable-parallel-backend-execution"])
    config = cli._config_from_args(args)

    assert args.disable_parallel_backend_execution is True
    assert config.enable_parallel_backend_execution is False


def test_cli_experiment_parses_parallel_backend_execution_ablation_flag():
    parser = build_parser()
    args = parser.parse_args(["experiment", "--disable-parallel-backend-execution"])

    assert args.cmd == "experiment"
    assert args.disable_parallel_backend_execution is True


def test_cli_experiment_parses_adaptive_component_ablation_flags():
    parser = build_parser()
    args = parser.parse_args(
        [
            "experiment",
            "--disable-adaptive-components",
            (
                "contextual-bandit,map-elites,profile-filter,runtime-cost,"
                "uncertainty-sampling,online-reward,cross-version-learning,"
                "simulated-annealing,semantic-objective,mr-learning,version-pair,"
                "backend-pair,typed-ir-rewrite,energy-quota,seed-power,operator-power,"
                "latin-hypercube,champion,bd-axis-weights,good-turing,"
                "operator-swarm,divergence-conditioned,shrink-mutations,"
                "hierarchical-archive,lineage-rarity,minhash-dedup,"
                "disagreement-bd-axis,cost-normalized-reward"
            ),
        ]
    )

    assert args.disable_adaptive_components == {
        "scheduler_learning",
        "semantic_objective_learning",
        "metamorphic_relation_learning",
        "version_pair_learning",
        "backend_pair_learning",
        "bd_axis_bandit",
        "bayesian_exploration",
        "ir_rewrite_mutations",
        "quality_archive",
        "seed_quota",
        "seed_energy_batch",
        "per_operator_energy",
        "lhs_seeding",
        "champion_corpus",
        "profile_capability_filter",
        "runtime_cost_learning",
        "active_learning",
        "online_reward_model",
        "continual_learning",
        "scheduler_annealing",
        "operator_swarm",
        "divergence_conditioned",
        "shrink_mutations",
        "hierarchical_archive",
        "lineage_rarity",
        "minhash_dedup",
        "disagreement_bd_axis",
        "cost_normalized_reward",
    }


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
            "--persist-closed-loop-state",
            "--skip-paper-journal",
        ]
    )
    assert args.run_theme == "final-live:datafusion"
    assert args.paper_notes == "24h latest-version run"
    assert args.persist_closed_loop_state is True
    assert args.skip_paper_journal is True


def test_run_experiment_job_propagates_local_source_scheduler(monkeypatch):
    captured = {}

    def fake_run_fuzz(*, cases, seed, backends, config, duration_s, **kwargs):
        captured["enable_parallel_backend_execution"] = config.enable_parallel_backend_execution
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
            "enable_parallel_backend_execution": False,
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


def test_run_experiment_job_can_persist_closed_loop_state(monkeypatch):
    captured = {}

    def fake_run_fuzz(*, cases, seed, backends, config, duration_s, **kwargs):
        captured["persist_closed_loop_state"] = kwargs.get("persist_closed_loop_state", False)
        return Path("runs/fake.jsonl")

    monkeypatch.setattr(cli, "run_fuzz", fake_run_fuzz)
    monkeypatch.setattr(cli, "write_report", lambda run_file: (Path(""), Path("")))

    cli._run_experiment_job(
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
            "persist_closed_loop_state": True,
            "skip_run_reports": True,
        }
    )

    assert captured["persist_closed_loop_state"] is True


def test_run_experiment_job_applies_strategy_snapshot_flags(monkeypatch):
    captured = {}

    def fake_run_fuzz(*, cases, seed, backends, config, duration_s, **kwargs):
        captured["strategy_snapshot_path"] = config.strategy_snapshot_path
        captured["strategy_learning_path"] = config.strategy_learning_path
        captured["freeze_strategy_snapshot"] = config.freeze_strategy_snapshot
        return Path("runs/fake.jsonl")

    monkeypatch.setattr(cli, "run_fuzz", fake_run_fuzz)
    monkeypatch.setattr(cli, "write_report", lambda run_file: (Path(""), Path("")))

    cli._run_experiment_job(
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
            "strategy_snapshot": "reports/frozen-strategy.json",
            "strategy_learning": "reports/strategy-learning.json",
            "freeze_strategy_snapshot": True,
            "skip_run_reports": True,
        }
    )

    assert captured["strategy_snapshot_path"] == "reports/frozen-strategy.json"
    assert captured["strategy_learning_path"] == "reports/strategy-learning.json"
    assert captured["freeze_strategy_snapshot"] is True


def test_run_experiment_job_propagates_per_case_adaptive_config(monkeypatch):
    captured = {}

    def fake_run_fuzz(*, cases, seed, backends, config, duration_s, **kwargs):
        captured["generator_profile_pool"] = config.generator_profile_pool
        captured["generator_profile_learning_weight"] = config.generator_profile_learning_weight
        captured["semantic_objective_learning_weight"] = config.semantic_objective_learning_weight
        captured["enable_metamorphic_oracle"] = config.enable_metamorphic_oracle
        captured["oracle_mode"] = config.oracle_mode
        captured["metamorphic_relation_learning_weight"] = config.metamorphic_relation_learning_weight
        captured["metamorphic_relation_order"] = config.metamorphic_relation_order
        captured["version_pair_pool"] = config.version_pair_pool
        captured["version_pair_learning_weight"] = config.version_pair_learning_weight
        captured["backend_pair_learning_weight"] = config.backend_pair_learning_weight
        captured["backend_pair_priority_limit"] = config.backend_pair_priority_limit
        return Path("runs/fake-adaptive.jsonl")

    monkeypatch.setattr(cli, "run_fuzz", fake_run_fuzz)
    monkeypatch.setattr(cli, "write_report", lambda run_file: (Path(""), Path("")))

    cli._run_experiment_job(
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
            "profile_pool": "common,discovery_fresh",
            "profile_learning_weight": 0.7,
            "semantic_objective_learning_weight": 0.5,
            "enable_metamorphic_oracle": True,
            "metamorphic_relation_learning_weight": 0.6,
            "metamorphic_relation_order": "row_permutation,filter_idempotence",
            "version_pair_pool": "latest->fixed,latest->preview",
            "version_pair_learning_weight": 0.8,
            "backend_pair_learning_weight": 0.9,
            "backend_pair_priority_limit": 4,
            "skip_run_reports": True,
        }
    )

    assert captured["generator_profile_pool"] == ["common", "discovery_fresh"]
    assert captured["generator_profile_learning_weight"] == 0.7
    assert captured["semantic_objective_learning_weight"] == 0.5
    assert captured["enable_metamorphic_oracle"] is True
    assert captured["oracle_mode"] == "both"
    assert captured["metamorphic_relation_learning_weight"] == 0.6
    assert captured["metamorphic_relation_order"] == ["row_permutation", "filter_idempotence"]
    assert captured["version_pair_pool"] == ["latest->fixed", "latest->preview"]
    assert captured["version_pair_learning_weight"] == 0.8
    assert captured["backend_pair_learning_weight"] == 0.9
    assert captured["backend_pair_priority_limit"] == 4


def test_run_experiment_job_applies_adaptive_component_ablation(monkeypatch):
    captured = {}

    def fake_run_fuzz(*, cases, seed, backends, config, duration_s, **kwargs):
        captured["enable_local_source_scheduler"] = config.enable_local_source_scheduler
        captured["local_source_exploration_weight"] = config.local_source_exploration_weight
        captured["enable_profile_capability_filter"] = config.enable_profile_capability_filter
        captured["enable_mutation_operator_learning"] = config.enable_mutation_operator_learning
        captured["enable_operator_swarm"] = config.enable_operator_swarm
        captured["enable_ir_rewrite_mutations"] = config.enable_ir_rewrite_mutations
        captured["enable_divergence_conditioned_mutations"] = config.enable_divergence_conditioned_mutations
        captured["enable_shrink_mutations"] = config.enable_shrink_mutations
        captured["enable_quality_archive"] = config.enable_quality_archive
        captured["enable_hierarchical_archive"] = config.enable_hierarchical_archive
        captured["enable_bd_axis_bandit"] = config.enable_bd_axis_bandit
        captured["enable_bayesian_exploration"] = config.enable_bayesian_exploration
        captured["enable_seed_quota"] = config.enable_seed_quota
        captured["enable_seed_energy_batch"] = config.enable_seed_energy_batch
        captured["enable_seed_energy_tier_bandit"] = config.enable_seed_energy_tier_bandit
        captured["enable_per_operator_energy"] = config.enable_per_operator_energy
        captured["enable_lineage_rarity"] = config.enable_lineage_rarity
        captured["enable_minhash_dedup"] = config.enable_minhash_dedup
        captured["enable_disagreement_bd_axis"] = config.enable_disagreement_bd_axis
        captured["enable_lhs_seeding"] = config.enable_lhs_seeding
        captured["enable_champion_corpus"] = config.enable_champion_corpus
        captured["enable_champion_graft_donor_bandit"] = config.enable_champion_graft_donor_bandit
        return Path("runs/fake-ablation.jsonl")

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
            "disable_adaptive_components": [
                "local_source_scheduler",
                "profile_capability_filter",
                "mutation_operator_learning",
                "operator_swarm",
                "ir_rewrite_mutations",
                "divergence_conditioned",
                "shrink_mutations",
                "quality_archive",
                "hierarchical_archive",
                "bd_axis_bandit",
                "bayesian_exploration",
                "seed_quota",
                "seed_energy_batch",
                "seed_energy_tier",
                "per_operator_energy",
                "lineage_rarity",
                "minhash_dedup",
                "disagreement_bd_axis",
                "lhs_seeding",
                "champion_corpus",
                "champion_graft_donor",
            ],
            "skip_run_reports": True,
        }
    )

    assert captured["enable_local_source_scheduler"] is False
    assert captured["local_source_exploration_weight"] == 0.0
    assert captured["enable_profile_capability_filter"] is False
    assert captured["enable_mutation_operator_learning"] is False
    assert captured["enable_operator_swarm"] is False
    assert captured["enable_ir_rewrite_mutations"] is False
    assert captured["enable_divergence_conditioned_mutations"] is False
    assert captured["enable_shrink_mutations"] is False
    assert captured["enable_quality_archive"] is False
    assert captured["enable_hierarchical_archive"] is False
    assert captured["enable_bd_axis_bandit"] is False
    assert captured["enable_bayesian_exploration"] is False
    assert captured["enable_seed_quota"] is False
    assert captured["enable_seed_energy_batch"] is False
    assert captured["enable_seed_energy_tier_bandit"] is False
    assert captured["enable_per_operator_energy"] is False
    assert captured["enable_lineage_rarity"] is False
    assert captured["enable_minhash_dedup"] is False
    assert captured["enable_disagreement_bd_axis"] is False
    assert captured["enable_lhs_seeding"] is False
    assert captured["enable_champion_corpus"] is False
    assert captured["enable_champion_graft_donor_bandit"] is False
    assert result["run"]["adaptive_components"]["local_source_scheduler"] is False
    assert result["run"]["disabled_adaptive_components"] == [
        "bayesian_exploration",
        "bd_axis_bandit",
        "champion_corpus",
        "champion_graft_donor",
        "disagreement_bd_axis",
        "divergence_conditioned",
        "hierarchical_archive",
        "ir_rewrite_mutations",
        "lhs_seeding",
        "lineage_rarity",
        "local_source_scheduler",
        "minhash_dedup",
        "mutation_operator_learning",
        "operator_swarm",
        "per_operator_energy",
        "profile_capability_filter",
        "quality_archive",
        "seed_energy_batch",
        "seed_energy_tier",
        "seed_quota",
        "shrink_mutations",
    ]


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
    assert config["generator_profile"] == "discovery_no_groupby"
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
    monkeypatch.setattr(cli, "ProcessPoolExecutor", cli.ThreadPoolExecutor)

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
    manifest_file = next(runs_dir.glob("experiment-*.json"))
    meta = json.loads(run_meta_path(run_file).read_text(encoding="utf-8"))
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    journal = (reports_dir / "paper-run-journal.jsonl").read_text(encoding="utf-8")
    journal_entry = json.loads(journal.strip().splitlines()[-1])
    output = capsys.readouterr().out
    assert "status=ok" in output
    assert f"experiment manifest: {manifest_file}" in output
    assert meta["preset"] == "fixture_replay"
    assert meta["evidence_mode"] == "historical"
    assert meta["known_bug_id"] == "fixture-test"
    assert meta["manifest_file"] == str(manifest_file)
    assert meta["fixture_sha256"] == fixture_sha256(fixture_path)
    assert meta["config"]["enable_replay_bug"] is True
    assert meta["experiment_meta"]["matrix_id"] == "historical_replay"
    assert meta["experiment_meta"]["historical"]["bug_id"] == "fixture-test"
    assert meta["experiment_meta"]["variant"]["variant_id"] == "fixture-test"
    assert manifest["evidence_mode"] == "historical"
    assert manifest["experiment_meta"]["matrix_id"] == "historical_replay"
    assert manifest["fixture_sha256"] == fixture_sha256(fixture_path)
    assert manifest["runs"][0]["run_file"] == str(run_file)
    assert manifest["runs"][0]["preset"] == "fixture_replay"
    assert manifest["runs"][0]["matrix_id"] == "historical_replay"
    assert manifest["runs"][0]["variant_id"] == "fixture-test"
    assert manifest["runs"][0]["fixture_sha256"] == fixture_sha256(fixture_path)
    assert '"known_bug_id": "fixture-test"' in journal
    assert journal_entry["manifest_file"] == str(manifest_file)
    assert journal_entry["matrix_id"] == "historical_replay"


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
            "--adaptive-learning-weight",
            "0.75",
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
    assert manifest["adaptive_config"]["learning_weight"] == 0.75
    assert manifest["adaptive_learning"]["schema_version"] == "adaptive-learning-v1"
    assert "batch_arm" in manifest["adaptive_learning"]["bandits"]
    assert "generator_profile" in manifest["adaptive_learning"]["bandits"]
    assert "guidance_strategy" in manifest["adaptive_learning"]["bandits"]
    assert "oracle_mode" in manifest["adaptive_learning"]["bandits"]
    assert "semantic_objective" in manifest["adaptive_learning"]["bandits"]
    assert "discovery_fresh" in {
        arm["action_id"]
        for arm in manifest["adaptive_learning"]["bandits"]["generator_profile"]["arms"]
    }
    suites = [run["target_suite"] for run in manifest["runs"]]
    assert suites.count("datafusion_cross") >= suites.count("core")


def test_cli_experiment_adaptive_scheduler_imports_continual_learning_ledgers(
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

    ledger_a = build_version_ledger(
        [
            VersionObservation(
                version_id="engine-v1",
                run_file="runs/engine-v1.jsonl",
                case_count=1,
                candidate_families={},
            ),
            VersionObservation(
                version_id="engine-v2",
                run_file="runs/engine-v2.jsonl",
                case_count=1,
                candidate_families={"null_semantics@engine": 1},
            ),
        ]
    )
    ledger_b = build_version_ledger(
        [
            VersionObservation(
                version_id="engine-v2",
                run_file="runs/engine-v2.jsonl",
                case_count=1,
                candidate_families={},
            ),
            VersionObservation(
                version_id="engine-v3",
                run_file="runs/engine-v3.jsonl",
                case_count=1,
                candidate_families={"cast_semantics@engine": 1},
            ),
        ]
    )
    ledger_a_path = tmp_path / "ledger-a.json"
    ledger_b_path = tmp_path / "ledger-b.json"
    dump_json(ledger_a, ledger_a_path)
    dump_json(ledger_b, ledger_b_path)
    old_ledger_path = tmp_path / "old-ledger.json"
    old_ledger = dict(ledger_a)
    old_ledger.pop("health_feedback_report", None)
    dump_json(old_ledger, old_ledger_path)

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
        run_file = runs_dir / f"run-continual-{idx}.jsonl"
        append_jsonl(
            {
                "case": {"case_id": f"case-{idx}", "seed": seed},
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
            "core",
            "--presets",
            "live_deep_organic",
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
            "--adaptive-learning-weight",
            "0.75",
            "--continual-learning-ledgers",
            f"{ledger_a_path},{old_ledger_path},{ledger_b_path}",
            "--skip-run-reports",
        ]
    )

    assert args.func(args) == 0
    assert "reward=" in capsys.readouterr().out
    manifest = json.loads(next(runs_dir.glob("experiment-*.json")).read_text(encoding="utf-8"))
    sources = manifest["adaptive_config"]["continual_learning_sources"]
    memory = manifest["adaptive_learning"]["continual_priority_memory"]

    assert [source["loaded"] for source in sources] == [True, False, True]
    assert sources[0]["family_count"] == 1
    assert sources[1]["reason"] == "missing_health_feedback_report"
    assert sources[2]["family_count"] == 2
    assert memory["imported_ledger_count"] == 2
    assert memory["family_priorities"]["null_semantics@engine"] == 0.9
    assert memory["family_priorities"]["cast_semantics@engine"] == 0.9
    assert manifest["adaptive_learning"]["bandits"]["batch_arm"]["arms"]


def test_cli_experiment_adaptive_component_ablation_manifest_and_runtime(tmp_path, monkeypatch):
    runs_dir = tmp_path / "runs"
    reports_dir = tmp_path / "reports"
    bugs_dir = tmp_path / "bugs"
    corpus_dir = tmp_path / "corpus"
    monkeypatch.setattr(cli, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(cli, "REPORTS_DIR", reports_dir)
    monkeypatch.setattr(cli, "BUGS_DIR", bugs_dir)
    monkeypatch.setattr(cli, "CORPUS_DIR", corpus_dir)

    captured = {}

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
        captured["enable_local_source_scheduler"] = config.enable_local_source_scheduler
        captured["enable_operator_swarm"] = config.enable_operator_swarm
        captured["enable_quality_archive"] = config.enable_quality_archive
        captured["enable_bd_axis_bandit"] = config.enable_bd_axis_bandit
        captured["enable_bayesian_exploration"] = config.enable_bayesian_exploration
        captured["enable_seed_quota"] = config.enable_seed_quota
        captured["enable_seed_energy_batch"] = config.enable_seed_energy_batch
        captured["enable_seed_energy_tier_bandit"] = config.enable_seed_energy_tier_bandit
        captured["enable_ir_rewrite_mutations"] = config.enable_ir_rewrite_mutations
        captured["enable_divergence_conditioned_mutations"] = config.enable_divergence_conditioned_mutations
        captured["enable_shrink_mutations"] = config.enable_shrink_mutations
        captured["enable_per_operator_energy"] = config.enable_per_operator_energy
        captured["enable_hierarchical_archive"] = config.enable_hierarchical_archive
        captured["enable_lineage_rarity"] = config.enable_lineage_rarity
        captured["enable_minhash_dedup"] = config.enable_minhash_dedup
        captured["enable_disagreement_bd_axis"] = config.enable_disagreement_bd_axis
        captured["enable_lhs_seeding"] = config.enable_lhs_seeding
        captured["enable_champion_corpus"] = config.enable_champion_corpus
        captured["enable_champion_graft_donor_bandit"] = config.enable_champion_graft_donor_bandit
        run_file = runs_dir / "run-ablation.jsonl"
        append_jsonl({"case": {"case_id": "case-1", "seed": seed}, "findings": []}, run_file)
        meta_path = Path(str(run_file).replace(".jsonl", ".meta.json"))
        meta_path.write_text(
            json.dumps(
                {
                    "elapsed_s": 0.1,
                    "throughput_cases_s": 10.0,
                    "next_seed": seed + cases,
                    "stage_profile": {
                        "share_of_total": {"scheduler_feedback_ms": 0.9},
                        "totals_ms": {
                            "scheduler_feedback_ms": 9.0,
                            "total_case_wall_ms": 10.0,
                        },
                    },
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
            "datafusion_cross",
            "--presets",
            "live_deep_organic",
            "--seeds",
            "1",
            "--cases",
            "1",
            "--schedule",
            "adaptive",
            "--jobs",
            "1",
            "--adaptive-learning-weight",
            "0.75",
            "--scheduler-annealing-temperature",
            "0.5",
            "--disable-adaptive-components",
            (
                "local-source-scheduler,quality-archive,runtime-cost-learning,"
                "active-learning,online-reward-model,continual-learning,"
                "scheduler-annealing,seed-quota,seed-energy-batch,"
                "seed-energy-tier,"
                "ir-rewrite-mutations,per-operator-energy,lhs-seeding,champion-corpus,"
                "champion-graft-donor,"
                "bd-axis-bandit,bayesian-exploration,operator-swarm,"
                "divergence-conditioned,shrink-mutations,hierarchical-archive,lineage-rarity,"
                "minhash-dedup,disagreement-bd-axis,cost-normalized-reward"
            ),
            "--skip-run-reports",
        ]
    )

    assert args.func(args) == 0
    manifest = json.loads(next(runs_dir.glob("experiment-*.json")).read_text(encoding="utf-8"))

    assert captured["enable_local_source_scheduler"] is False
    assert captured["enable_operator_swarm"] is False
    assert captured["enable_quality_archive"] is False
    assert captured["enable_bd_axis_bandit"] is False
    assert captured["enable_bayesian_exploration"] is False
    assert captured["enable_seed_quota"] is False
    assert captured["enable_seed_energy_batch"] is False
    assert captured["enable_seed_energy_tier_bandit"] is False
    assert captured["enable_ir_rewrite_mutations"] is False
    assert captured["enable_divergence_conditioned_mutations"] is False
    assert captured["enable_shrink_mutations"] is False
    assert captured["enable_per_operator_energy"] is False
    assert captured["enable_hierarchical_archive"] is False
    assert captured["enable_lineage_rarity"] is False
    assert captured["enable_minhash_dedup"] is False
    assert captured["enable_disagreement_bd_axis"] is False
    assert captured["enable_lhs_seeding"] is False
    assert captured["enable_champion_corpus"] is False
    assert captured["enable_champion_graft_donor_bandit"] is False
    assert manifest["adaptive_config"]["learning_weight"] == 0.75
    assert manifest["adaptive_config"]["record_learning_feedback"] is True
    assert manifest["adaptive_config"]["runtime_cost_learning"] is False
    assert manifest["adaptive_config"]["active_learning"] is False
    assert manifest["adaptive_config"]["online_reward_model"] is False
    assert manifest["adaptive_config"]["continual_learning"] is False
    assert manifest["adaptive_config"]["bayesian_exploration"] is False
    assert manifest["adaptive_config"]["cost_normalized_reward"] is False
    assert manifest["adaptive_config"]["scheduler_annealing"] is False
    assert manifest["adaptive_config"]["annealing_initial_temperature"] == 0.0
    assert manifest["adaptive_config"]["fine_grained_local_source_scheduler"] is False
    assert "batch_arm" in manifest["adaptive_learning"]["bandits"]
    assert manifest["adaptive_learning"]["bandits"]["batch_arm"]["reward_model"]["total_updates"] == 0
    batch_arm = manifest["adaptive_learning"]["bandits"]["batch_arm"]["arms"][0]
    assert batch_arm["runtime_cost_total"] == 0.0
    assert manifest["adaptive_methodology"]["components"]["quality_archive"] is False
    assert manifest["adaptive_methodology"]["components"]["bd_axis_bandit"] is False
    assert manifest["adaptive_methodology"]["components"]["bayesian_exploration"] is False
    assert manifest["adaptive_methodology"]["components"]["operator_swarm"] is False
    assert manifest["adaptive_methodology"]["components"]["divergence_conditioned"] is False
    assert manifest["adaptive_methodology"]["components"]["shrink_mutations"] is False
    assert manifest["adaptive_methodology"]["components"]["hierarchical_archive"] is False
    assert manifest["adaptive_methodology"]["components"]["lineage_rarity"] is False
    assert manifest["adaptive_methodology"]["components"]["minhash_dedup"] is False
    assert manifest["adaptive_methodology"]["components"]["disagreement_bd_axis"] is False
    assert manifest["adaptive_methodology"]["components"]["cost_normalized_reward"] is False
    assert manifest["adaptive_methodology"]["components"]["seed_quota"] is False
    assert manifest["adaptive_methodology"]["components"]["seed_energy_batch"] is False
    assert manifest["adaptive_methodology"]["components"]["ir_rewrite_mutations"] is False
    assert manifest["adaptive_methodology"]["components"]["per_operator_energy"] is False
    assert manifest["adaptive_methodology"]["components"]["lhs_seeding"] is False
    assert manifest["adaptive_methodology"]["components"]["champion_corpus"] is False
    assert manifest["adaptive_methodology"]["components"]["runtime_cost_learning"] is False
    assert manifest["adaptive_methodology"]["components"]["active_learning"] is False
    assert manifest["adaptive_methodology"]["components"]["online_reward_model"] is False
    assert manifest["adaptive_methodology"]["components"]["continual_learning"] is False
    assert manifest["adaptive_methodology"]["components"]["scheduler_annealing"] is False
    assert manifest["runs"][0]["adaptive_components"]["scheduler_learning"] is True
    assert manifest["runs"][0]["adaptive_components"]["runtime_cost_learning"] is False
    assert manifest["runs"][0]["adaptive_components"]["active_learning"] is False
    assert manifest["runs"][0]["adaptive_components"]["online_reward_model"] is False
    assert manifest["runs"][0]["adaptive_components"]["continual_learning"] is False
    assert manifest["runs"][0]["adaptive_components"]["bayesian_exploration"] is False
    assert manifest["runs"][0]["adaptive_components"]["scheduler_annealing"] is False
    assert manifest["runs"][0]["adaptive_components"]["bd_axis_bandit"] is False
    assert manifest["runs"][0]["adaptive_components"]["ir_rewrite_mutations"] is False
    assert manifest["runs"][0]["adaptive_components"]["per_operator_energy"] is False


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
            if set(backends) == {"pandas", "polars"}
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


def test_cli_experiment_adaptive_duration_is_total_matrix_budget(tmp_path, monkeypatch):
    runs_dir = tmp_path / "runs"
    reports_dir = tmp_path / "reports"
    bugs_dir = tmp_path / "bugs"
    corpus_dir = tmp_path / "corpus"
    monkeypatch.setattr(cli, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(cli, "REPORTS_DIR", reports_dir)
    monkeypatch.setattr(cli, "BUGS_DIR", bugs_dir)
    monkeypatch.setattr(cli, "CORPUS_DIR", corpus_dir)

    captured: dict[str, object] = {}

    class FakeScheduler:
        def __init__(
            self,
            jobs,
            *,
            total_cases_budget,
            total_duration_budget_s,
            config,
            learning_state=None,
        ):
            captured["job_count"] = len(jobs)
            captured["total_cases_budget"] = total_cases_budget
            captured["total_duration_budget_s"] = total_duration_budget_s
            self.remaining_cases_budget = total_cases_budget
            self.remaining_duration_budget_s = total_duration_budget_s
            self.learning_state = type(
                "LearningStateStub",
                (),
                {"to_state_dict": staticmethod(lambda: {})},
            )()

        def has_budget(self):
            return False

        def snapshot(self):
            return []

    monkeypatch.setattr(cli, "AdaptiveBudgetScheduler", FakeScheduler)

    parser = build_parser()
    args = parser.parse_args(
        [
            "experiment",
            "--target-suites",
            "core,datafusion_cross",
            "--presets",
            "live_deep_organic",
            "--seeds",
            "1,2",
            "--duration",
            "12h",
            "--schedule",
            "adaptive",
            "--batch-duration",
            "10m",
            "--jobs",
            "1",
            "--skip-run-reports",
        ]
    )

    assert args.func(args) == 0
    assert captured["job_count"] == 4
    assert captured["total_cases_budget"] is None
    assert captured["total_duration_budget_s"] == 12 * 3600

    manifest = json.loads(next(runs_dir.glob("experiment-*.json")).read_text(encoding="utf-8"))
    assert manifest["duration_s"] == 12 * 3600
    assert manifest["adaptive_config"]["total_duration_budget_s"] == 12 * 3600


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
