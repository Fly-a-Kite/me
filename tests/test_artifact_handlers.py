from argparse import Namespace
from types import SimpleNamespace
import json

from datadiff.commands.artifact_handlers import (
    _apply_fixture_replay_new_behavior_flags,
    cmd_replay_fixture_impl,
)
from datadiff.config import ExperimentConfig
from datadiff.util import JsonlWriter, dump_json, run_meta_path


def test_fixture_replay_filters_non_rewardable_signal_new_behavior():
    config = ExperimentConfig(known_saturated_bug_families=["known_root@duckdb"])
    rows = [
        (
            {
                "findings": [
                    {
                        "triage_verdict": "candidate_implementation_bug",
                        "false_positive": True,
                        "root_cause": "false_positive_root",
                        "suspicious_backends": ["duckdb"],
                    }
                ]
            },
            False,
        ),
        (
            {
                "findings": [
                    {
                        "triage_verdict": "candidate_implementation_bug",
                        "root_cause": "source_root",
                        "suspicious_backends": ["duckdb"],
                        "source_issue": "fixture-replay-source-issue",
                    }
                ]
            },
            False,
        ),
        (
            {
                "findings": [
                    {
                        "triage_verdict": "candidate_implementation_bug",
                        "root_cause": "known_root",
                        "suspicious_backends": ["duckdb"],
                    }
                ]
            },
            False,
        ),
        (
            {
                "findings": [
                    {
                        "triage_verdict": "candidate_implementation_bug",
                        "root_cause": "fresh_root",
                        "suspicious_backends": ["duckdb"],
                    }
                ]
            },
            True,
        ),
        (
            {
                "findings": [
                    {
                        "triage_verdict": "semantic_divergence_needs_confirmation",
                        "root_cause": "semantic_boundary",
                    }
                ]
            },
            True,
        ),
        (
            {
                "findings": [
                    {
                        "triage_verdict": "expected_semantic_divergence",
                        "root_cause": "documented_boundary",
                    }
                ]
            },
            False,
        ),
    ]

    for row, expected_signal in rows:
        raw_new_behavior, signal_new_behavior = _apply_fixture_replay_new_behavior_flags(row, config)

        assert raw_new_behavior is True
        assert signal_new_behavior is expected_signal
        assert row["is_new_behavior"] is True
        assert row["signal_new_behavior"] is expected_signal


def test_replay_fixture_command_writes_rewardable_signal_counts(tmp_path):
    runs_dir = tmp_path / "runs"
    reports_dir = tmp_path / "reports"
    manifest_path = runs_dir / "experiment-fixture.json"
    case = SimpleNamespace(
        case_id="case-fixture",
        seed=17,
        metadata={"fixture_sha256": "sha256-fixture"},
        tables=[SimpleNamespace(rows=[{"a": 1}])],
        program=SimpleNamespace(operations=[{"op": "noop"}]),
    )
    context = SimpleNamespace(
        common_capabilities=["projection"],
        target_dicts=lambda: [{"name": "duckdb", "family": "sql", "layer": "engine"}],
        to_dict=lambda: {"targets": ["duckdb"]},
    )

    args = Namespace(
        spec="spec.json",
        target_suite="core",
        backends="duckdb",
        evidence_mode="historical",
        known_bug_id="known-root-fixture",
        target_version="target==1.0",
        run_theme="",
        paper_notes="",
        experiment_meta="",
        disable_artifact=True,
        artifact_limit=0,
        log_level="compact",
        no_compress_run_log=True,
        disable_adaptive_components="",
    )

    def config_factory(**kwargs):
        return ExperimentConfig(
            known_saturated_bug_families=["known_root@duckdb"],
            **kwargs,
        )

    def run_loaded_case_func(*args, **kwargs):
        return {
            "status": "bug",
            "findings": [
                {
                    "triage_verdict": "candidate_implementation_bug",
                    "root_cause": "known_root",
                    "suspicious_backends": ["duckdb"],
                    "signature": "known-saturated-signature",
                }
            ],
            "environment": {},
            "bug_dir": "",
        }

    def run_semantics(*args, **kwargs):
        return {
            "matrix_id": "historical_replay",
            "matrix_title": "Historical replay",
            "comparison_group": "historical",
            "purpose": "fixture",
            "counts_as_real_bugs": True,
            "rq_tags": [],
            "analysis_tags": [],
            "variant_id": "known-root-fixture",
            "variant_title": "Known root fixture",
            "base_preset": "fixture_replay",
            "comparison_role": "reference",
            "canonical_comparison_role": "reference",
            "component_focus": [],
            "overlays": [],
            "semantic_focus_families": [],
            "semantic_focus_signals": [],
            "factors": {},
            "oracle_profile": "differential",
            "scope_kind": "single_suite",
        }

    assert cmd_replay_fixture_impl(
        args,
        ensure_dirs_func=lambda: None,
        load_fixture_replay_spec_func=lambda spec: {"target_suite": "core"},
        resolve_fixture_replay_path_func=lambda args: tmp_path / "fixture.parquet",
        build_fixture_replay_case_func=lambda spec, path: case,
        parse_backends_func=lambda value: [value],
        resolve_target_backends_func=lambda value, suite: ["duckdb"],
        parse_experiment_meta_func=lambda value: {},
        normalize_experiment_meta_func=lambda value: dict(value),
        replay_bug_enabled_by_default_func=lambda evidence_mode: False,
        registered_experiment_meta_defaults_func=lambda **kwargs: {},
        merge_experiment_meta_func=lambda defaults, explicit: {**defaults, **explicit},
        experiment_config_factory=config_factory,
        parse_adaptive_components_func=lambda value: set(),
        adaptive_component_config_func=lambda disabled: {},
        run_loaded_case_func=run_loaded_case_func,
        fixture_artifact_budget_allows_func=lambda limit: False,
        describe_targets_func=lambda backends: [{"name": backend} for backend in backends],
        fixture_replay_run_id_func=lambda label: "run-fixture-known-root",
        runs_dir=runs_dir,
        jsonl_writer_factory=JsonlWriter,
        compact_log_row_func=lambda row, log_level: row,
        target_context_func=lambda backends: context,
        experiment_manifest_path_func=lambda: manifest_path,
        resolved_run_semantics_func=run_semantics,
        catalog_preset_metadata_func=lambda *args, **kwargs: {},
        configured_guidance_targets_func=lambda config: [],
        describe_operation_combo_func=lambda operations: {"operations": len(operations)},
        dump_json_func=dump_json,
        run_meta_path_func=run_meta_path,
        record_run_journal_func=lambda run_file, context, journal_file: (
            reports_dir / "paper-run-journal.jsonl",
            reports_dir / "paper-run-journal.md",
        ),
        reports_dir=reports_dir,
        utc_now_func=lambda: "2026-01-01T00:00:00Z",
    ) == 0

    run_file = runs_dir / "run-fixture-known-root.jsonl"
    row = json.loads(run_file.read_text(encoding="utf-8").strip())
    meta = json.loads(run_meta_path(run_file).read_text(encoding="utf-8"))

    assert row["is_new_behavior"] is True
    assert row["signal_new_behavior"] is False
    assert meta["new_behavior_cases"] == 1
    assert meta["signal_new_behavior_cases"] == 0
